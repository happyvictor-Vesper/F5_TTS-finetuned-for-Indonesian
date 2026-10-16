#!/usr/bin/env python3
"""
check_structure.py — verify the project tree is self-consistent.

Run this after `git clone` to confirm the project is laid out the way
Indonesian_TTS_FineTuning.ipynb expects. Exits non-zero on the first
missing or inconsistent thing it finds.

Usage:
    python scripts/check_structure.py [--skip-f5] [--skip-weights]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def exists(p: Path, label: str) -> bool:
    ok = p.exists()
    print(f"  {'OK' if ok else 'MISSING'}  {label:32s}  {p.relative_to(ROOT)}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-f5", action="store_true",
                        help="Don't require the F5-TTS vendored checkout "
                             "(useful for partial checkouts before clone).")
    parser.add_argument("--skip-weights", action="store_true",
                        help="Don't require trained checkpoints / synthesis WAVs.")
    args = parser.parse_args()

    print(f"Project root: {ROOT}")
    print("\n== Top-level structure ==")
    required = [
        ("README.md",       ROOT / "README.md"),
        ("LICENSE",         ROOT / "LICENSE"),
        ("CITATION.cff",    ROOT / "CITATION.cff"),
        ("requirements.txt",ROOT / "requirements.txt"),
        ("config.yaml",     ROOT / "configs" / "config.yaml"),
        ("notebook",        ROOT / "Indonesian_TTS_FineTuning.ipynb"),
    ]
    ok = all(exists(p, lbl) for lbl, p in required)

    print("\n== src/ wrappers ==")
    for f in ("common", "get_dataset", "prepare_data", "build_f5_dataset",
              "train", "infer", "evaluate", "compare", "preview_audio",
              "finalize", "finetune_settings", "launch_gradio"):
        ok &= exists(ROOT / "src" / f"{f}.py", f"src/{f}.py")

    print("\n== docs/ ==")
    ok &= exists(ROOT / "docs" / "alignment_notes.md", "docs/alignment_notes.md")
    ok &= exists(ROOT / "docs" / "gradio_and_cli.md", "docs/gradio_and_cli.md")

    if not args.skip_f5:
        print("\n== Vendored F5-TTS ==")
        f5 = ROOT / "F5-TTS"
        ok &= exists(f5, "F5-TTS/")
        # F5-TTS 1.1.x uses pyproject.toml-based packaging — no __init__.py
        # required (PEP 420 implicit namespace package).
        ok &= exists(f5 / "src" / "f5_tts" / "api.py", "F5-TTS engine package")
        ok &= exists(f5 / "src" / "f5_tts" / "train" / "finetune_cli.py",
                     "F5-TTS finetune CLI")
        ok &= exists(f5 / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py",
                     "F5-TTS prepare_csv_wavs")
        ok &= exists(f5 / "src" / "f5_tts" / "infer" / "utils_infer.py",
                     "F5-TTS inference utils")
        ok &= exists(f5 / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt",
                     "F5-TTS base vocab")

    if not args.skip_weights:
        print("\n== Generated artefacts (optional) ==")
        ok &= exists(ROOT / "workspace_full" / "train.csv", "workspace_full/train.csv")
        ok &= exists(ROOT / "workspace_full" / "val.csv",   "workspace_full/val.csv")
        ok &= exists(ROOT / "workspace_full" / "test.csv",  "workspace_full/test.csv")

    print()
    if ok:
        print("Project structure is consistent. ✓")
        sys.exit(0)
    print("Project structure has gaps. See lines marked MISSING above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
