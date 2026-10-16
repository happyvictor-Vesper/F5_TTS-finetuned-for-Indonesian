"""
launch_gradio.py — boot the vendored F5-TTS Gradio fine-tuning app.

This is the official `f5_tts.train.finetune_gradio` UI, launched with our
project's symlinks already in place and the Indonesian `setting.json`
pre-populated from `configs/config.yaml`.

What it gives you (inside the Gradio tab):

  1. **Dataset & Vocab tab** — points at the data we already prepared
     (`F5-TTS/data/id_tts_full_pinyin/raw.arrow` etc., via the symlink)
     so the "Check Vocab / Extend Vocab" button shows the extended vocab.
  2. **Basic Settings / Training tab** — pre-filled with our hyperparameters
     (45K updates, T4 1000-frame / A100 1700-frame batch, fp16/bf16, AdamW,
     `F5TTS_v1_Base` backbone, pinyin tokenizer, EMA on, last-per-updates =
     5000). Override anything you want and press "Save Settings".
  3. **Start training** — runs `accelerate launch finetune_cli.py` against
     `model_1250000.safetensors` (auto-downloaded from `SWivid/F5-TTS`).

CLI:

    python -m src.launch_gradio --config configs/config.yaml
    python -m src.launch_gradio --config configs/config.yaml --share
    python -m src.launch_gradio --config configs/config.yaml --port 7860
    python -m src.launch_gradio --config configs/config.yaml --no-settings

The `--no-settings` flag skips the setting.json pre-fill; useful if you just
want to drive the UI by hand from scratch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ensure_engine_links, f5_repo_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--project-name", default=None,
                        help="Override the project name (e.g. 'id_tts_full_com').")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Expose a public Gradio share link.")
    parser.add_argument("--no-settings", action="store_true",
                        help="Skip writing the Indonesian setting.json pre-fill.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_engine_links(cfg)

    if not args.no_settings:
        # Materialise setting.json so the Gradio app picks up our defaults
        # when the project is selected.
        from finetune_settings import build_settings
        project_name = args.project_name or cfg["train"]["dataset_name"]
        settings = build_settings(cfg, project_name)
        out = f5_repo_dir(cfg) / "ckpts" / project_name / "setting.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(settings, indent=2), encoding="utf-8")
        print(f"[launch_gradio] setting.json -> {out}")

    repo = f5_repo_dir(cfg)
    gradio_script = repo / "src" / "f5_tts" / "train" / "finetune_gradio.py"
    if not gradio_script.is_file():
        raise FileNotFoundError(gradio_script)

    cmd = [
        sys.executable, str(gradio_script),
        "--port", str(args.port),
        "--host", args.host,
    ]
    if args.share:
        cmd.append("--share")

    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    print(f"[launch_gradio] cmd = {' '.join(cmd)}")
    print(f"[launch_gradio] cwd = {Path.cwd()}")
    print(f"[launch_gradio] data/ckpts symlinks active — Indonesian project is selectable.")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
