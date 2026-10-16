"""
train.py — three ways to run a fine-tune.

This module is the single entry point for the Indonesian fine-tune, regardless
of which UI you prefer. It mirrors the notebook's `src/train.py` call but adds
two switches the notebook doesn't expose:

  * `--runner gradio`  → writes the `setting.json` the Gradio UI reads
                         (and exits — open the UI separately).
  * `--runner cli`     → invokes the **upstream** `finetune_cli.py` directly
                         with our hyperparameters translated to the F5-TTS
                         argparse surface.
  * `--runner module`  → invokes the engine as `python -m f5_tts.train.finetune_cli`
                         (the default — what the notebook does).

All three runners share the same `configs/config.yaml` source of truth, so
switching between Gradio / CLI / module is a flag, not a config rewrite.

CLI:

    # 1. Pre-fill the Gradio UI's Indonesian project (then open the UI):
    python -m src.train --config configs/config.yaml --runner gradio
    python -m src.launch_gradio --config configs/config.yaml

    # 2. Run the upstream CLI verbatim (training happens inline):
    python -m src.train --config configs/config.yaml --runner cli
    python -m src.train --config configs/config.yaml --runner cli --dry

    # 3. Run via python -m (the notebook's preferred path):
    python -m src.train --config configs/config.yaml --runner module

    # 4. FT-Combined: same entry point, different metadata + project name
    python -m src.train --config configs/config.yaml --runner cli \
                        --metadata-csv workspace_full/combined_train.csv \
                        --dataset-name id_tts_full_com
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    ensure_engine_links,
    f5_repo_dir,
    final_model_path,
    load_config,
    persistent_f5_data_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--runner", choices=["gradio", "cli", "module"],
                        default="module",
                        help="How to run the fine-tune (see module docstring).")
    parser.add_argument("--dry", action="store_true",
                        help="Smoke-test the trainer for a single update.")
    parser.add_argument("--metadata-csv", default=None,
                        help="Override the train.csv path (used for FT-Combined).")
    parser.add_argument("--dataset-name", default=None,
                        help="Override the project / dataset name "
                             "(e.g. 'id_tts_full_com' for FT-Combined).")
    parser.add_argument("--exp-name", default="F5TTS_v1_Base",
                        help="F5-TTS experiment name (default: F5TTS_v1_Base).")
    parser.add_argument("--pretrain", default=None,
                        help="Path or HF URL of the pretrained checkpoint. "
                             "Default: auto-download SWivid/F5-TTS F5TTS_v1_Base.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU defaults — mirror Section 3 of the notebook.
# ---------------------------------------------------------------------------

def _apply_gpu_defaults(cfg: dict) -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        name = torch.cuda.get_device_name(0).upper()
    except Exception:
        return

    if "T4" in name:
        cfg["train"]["batch_size_per_gpu"] = 1000
        cfg["train"]["grad_accumulation_steps"] = 2
        cfg["train"]["mixed_precision"] = "fp16"
        try:
            import bitsandbytes  # noqa: F401
            cfg["train"]["bnb_optimizer"] = True
        except Exception:
            cfg["train"]["bnb_optimizer"] = False
            cfg["train"]["batch_size_per_gpu"] = 800
    elif any(x in name for x in ("A100", "H100", "L4", "G4",
                                 "RTX PRO 6000", "BLACKWELL")):
        cfg["train"]["batch_size_per_gpu"] = 1700
        cfg["train"]["grad_accumulation_steps"] = 1
        cfg["train"]["mixed_precision"] = "bf16"
        cfg["train"]["bnb_optimizer"] = False
    else:
        cfg["train"]["batch_size_per_gpu"] = 800
        cfg["train"]["grad_accumulation_steps"] = 2
        cfg["train"]["mixed_precision"] = "fp16"
        cfg["train"]["bnb_optimizer"] = False


# ---------------------------------------------------------------------------
# CLI / module arg builders.
# ---------------------------------------------------------------------------

def _build_cli_args(cfg: dict, dataset_name: str, exp_name: str,
                    target_updates: int) -> list[str]:
    save_per_updates = min(int(cfg["train"]["save_per_updates"]), target_updates)
    last_per_updates = max(100, min(5_000, target_updates // 9 or 5_000))
    warmup_updates = min(int(cfg["train"]["warmup_updates"]), target_updates // 2)

    return [
        "--exp_name", exp_name,
        "--dataset_name", dataset_name,
        "--learning_rate", str(cfg["train"]["peak_learning_rate"]),
        "--batch_size_per_gpu", str(cfg["train"]["batch_size_per_gpu"]),
        "--batch_size_type", "frame",
        "--max_samples", str(cfg["train"]["max_samples"]),
        "--grad_accumulation_steps", str(cfg["train"]["grad_accumulation_steps"]),
        "--max_grad_norm", str(cfg["train"]["grad_clip"]),
        "--epochs", str(cfg["train"]["epochs"]),
        "--num_warmup_updates", str(warmup_updates),
        "--save_per_updates", str(save_per_updates),
        "--keep_last_n_checkpoints", str(cfg["train"]["keep_last_n_checkpoints"]),
        "--last_per_updates", str(last_per_updates),
        "--tokenizer", cfg["train"]["tokenizer"],
    ] + (["--bnb_optimizer"] if cfg["train"]["bnb_optimizer"] else [])


# ---------------------------------------------------------------------------
# Runners.
# ---------------------------------------------------------------------------

def _stage_pretrained(cfg: dict, dataset_name: str, project_dir: Path,
                      pretrain_arg: str | None) -> str | None:
    """
    The Gradio UI / finetune_cli.py both expect the pretrained weights to
    already live inside `<project>/pretrained_<basename>`. We arrange that here
    so the user can just press "Start training" without juggling paths.

    Returns the absolute path to the staged file, or None if nothing to do
    (caller will fall back to the auto-download URL).
    """
    if not cfg["train"].get("stage_pretrained", True):
        return pretrain_arg

    project_dir.mkdir(parents=True, exist_ok=True)
    if pretrain_arg is None:
        # Default upstream URL.
        return "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"

    if pretrain_arg.startswith("hf://") or "://" in pretrain_arg:
        return pretrain_arg  # cached_path will handle it

    src = Path(pretrain_arg)
    if not src.is_file():
        return pretrain_arg
    dst = project_dir / f"pretrained_{src.name}"
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
        print(f"[train] staged {src} -> {dst}")
    return str(dst)


def _runner_module(cfg: dict, args: argparse.Namespace) -> None:
    target_updates = 1 if args.dry else int(cfg["train"]["target_updates"])
    cli_args = _build_cli_args(cfg, args.dataset_name, args.exp_name, target_updates)
    if args.pretrain:
        cli_args += ["--pretrain", args.pretrain]
    cmd = [sys.executable, "-m", "f5_tts.train.finetune_cli", *cli_args]
    env = {**os.environ, "PYTHONPATH": str(f5_repo_dir(cfg) / "src")}
    print(f"[train] module runner: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _runner_cli(cfg: dict, args: argparse.Namespace) -> None:
    """Invoke the upstream `finetune_cli.py` directly (not as a module)."""
    repo = f5_repo_dir(cfg)
    script = repo / "src" / "f5_tts" / "train" / "finetune_cli.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    target_updates = 1 if args.dry else int(cfg["train"]["target_updates"])
    cli_args = _build_cli_args(cfg, args.dataset_name, args.exp_name, target_updates)

    project_dir = repo / "ckpts" / args.dataset_name
    pretrain = _stage_pretrained(cfg, args.dataset_name, project_dir, args.pretrain)
    if pretrain:
        cli_args += ["--pretrain", pretrain]

    cmd = [sys.executable, str(script), *cli_args]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    print(f"[train] cli runner:   {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _runner_gradio(cfg: dict, args: argparse.Namespace) -> None:
    """
    Materialise the Indonesian setting.json and exit. The user then opens the
    Gradio UI separately with `python -m src.launch_gradio`. Splitting the
    two steps means the UI process is independent of the training process
    (which is what the Gradio app itself expects).
    """
    from finetune_settings import build_settings

    project_name = args.dataset_name or cfg["train"]["dataset_name"]
    settings = build_settings(cfg, project_name)

    project_dir = f5_repo_dir(cfg) / "ckpts" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    out = project_dir / "setting.json"
    out.write_text(__import__("json").dumps(settings, indent=2), encoding="utf-8")

    _stage_pretrained(cfg, project_name, project_dir, args.pretrain)

    print(f"[train] gradio runner: wrote {out}")
    print(f"[train] gradio runner: next step → "
          f"python -m src.launch_gradio --config {args.config}")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

def _check_internal_dataset(cfg: dict, dataset_name: str, default_name: str) -> None:
    if dataset_name == default_name:
        persistent = persistent_f5_data_dir(cfg)
        if not (persistent / "raw.arrow").is_file():
            raise FileNotFoundError(
                f"Internal F5 dataset not built yet: {persistent}. "
                "Run src/build_f5_dataset.py first."
            )
    else:
        # Custom project — caller is responsible for having built it.
        pass


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_engine_links(cfg)
    _apply_gpu_defaults(cfg)

    args.dataset_name = args.dataset_name or cfg["train"]["dataset_name"]
    if args.metadata_csv and not Path(args.metadata_csv).is_file():
        raise FileNotFoundError(args.metadata_csv)
    _check_internal_dataset(cfg, args.dataset_name, cfg["train"]["dataset_name"])

    print(f"SECTION 9 — FINE-TUNING")
    print(f"Engine     : {f5_repo_dir(cfg)}")
    print(f"Project    : {args.dataset_name}")
    print(f"Output     : {final_model_path(cfg)}")
    print(f"Runner     : {args.runner}  ({'DRY-RUN' if args.dry else 'FULL'})")
    if args.pretrain:
        print(f"Pretrain   : {args.pretrain}")

    if args.runner == "gradio":
        _runner_gradio(cfg, args)
    elif args.runner == "cli":
        _runner_cli(cfg, args)
    else:  # module
        _runner_module(cfg, args)


if __name__ == "__main__":
    main()
