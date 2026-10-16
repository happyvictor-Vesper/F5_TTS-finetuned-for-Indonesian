"""
finetune_settings.py — materialise the Gradio training UI's `setting.json`.

The vendored F5-TTS Gradio app (`f5_tts.train.finetune_gradio`) stores one
`setting.json` per project under `F5-TTS/ckpts/<project_name>/`. The file
captures every hyperparameter the Gradio UI writes back when you click
"Save Settings", and the same file is read back when you reload a project
in the UI or launch training from it.

This wrapper reads our YAML config and writes the same JSON shape, so the
Indonesian project shows up in the Gradio app pre-filled with the values
from `configs/config.yaml` (45K updates, T4 batch 1000 / A100 1700, fp16/bf16,
`F5TTS_v1_Base` backbone, pinyin tokenizer, etc.). You can still tweak any
field inside the UI before pressing "Start training" — the Gradio app will
overwrite the JSON with whatever you last saved.

CLI:

    python -m src.finetune_settings --config configs/config.yaml
    python -m src.finetune_settings --config configs/config.yaml \
                                       --project-name id_tts_full_com   # FT-Combined
    python -m src.finetune_settings --config configs/config.yaml --print
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    ensure_engine_links,
    f5_ckpt_dir,
    f5_repo_dir,
    load_config,
    persistent_f5_data_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--project-name", default=None,
                        help="Override the project name "
                             "(e.g. 'id_tts_full_com' for FT-Combined).")
    parser.add_argument("--print", action="store_true",
                        help="Print the JSON to stdout instead of writing it.")
    return parser.parse_args()


def build_settings(cfg: dict, project_name: str) -> dict:
    """Translate our YAML config into the shape the Gradio UI writes."""
    f5_dir = f5_repo_dir(cfg)
    pretrained = (
        f5_dir
        / "ckpts"
        / project_name
        / f"pretrained_{cfg['train']['final_model_subpath'].split('/')[-1]}"
    )

    return {
        "exp_name": "F5TTS_v1_Base",
        "learning_rate": float(cfg["train"]["peak_learning_rate"]),
        "batch_size_per_gpu": int(cfg["train"]["batch_size_per_gpu"]),
        "batch_size_type": "frame",
        "max_samples": int(cfg["train"]["max_samples"]),
        "grad_accumulation_steps": int(cfg["train"]["grad_accumulation_steps"]),
        "max_grad_norm": float(cfg["train"]["grad_clip"]),
        "epochs": int(cfg["train"]["epochs"]),
        "num_warmup_updates": int(cfg["train"]["warmup_updates"]),
        "save_per_updates": int(cfg["train"]["save_per_updates"]),
        "keep_last_n_checkpoints": int(cfg["train"]["keep_last_n_checkpoints"]),
        "last_per_updates": max(100, int(cfg["train"]["target_updates"]) // 9),
        "finetune": True,
        "file_checkpoint_train": str(pretrained),
        "tokenizer_type": cfg["train"]["tokenizer"],
        "tokenizer_file": str(
            persistent_f5_data_dir(cfg) / cfg["train"]["vocab_filename"]
        ),
        "mixed_precision": cfg["train"]["mixed_precision"],
        "logger": None,
        "bnb_optimizer": bool(cfg["train"]["bnb_optimizer"]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_engine_links(cfg)

    project_name = args.project_name or cfg["train"]["dataset_name"]
    settings = build_settings(cfg, project_name)

    if args.print:
        print(json.dumps(settings, indent=2))
        return

    ckpt_dir = f5_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "setting.json"
    out.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Project   : {project_name}")
    print(f"Setting   : {out}")
    print(f"LR        : {settings['learning_rate']}")
    print(f"Batch/GPU : {settings['batch_size_per_gpu']}")
    print(f"Mixed prec: {settings['mixed_precision']}")
    print(f"Vocab     : {settings['tokenizer_file']}")
    print(f"Status    : WRITTEN")


if __name__ == "__main__":
    main()
