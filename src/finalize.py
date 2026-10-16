"""
finalize.py — build the project finalisation manifest.

Verifies the final checkpoint, the train/val/test splits, the per-system
evaluation summaries, the comparison artefacts, and the audio preview, then
emits a single `OUTPUT_MANIFEST.json` plus a `samples_manifest.csv` that
points at every per-utterance WAV without copying any binary content.

This is the "Section 13" of the notebook, packaged as a CLI for VS Code:

    python -m src.finalize --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from common import (
    final_model_path,
    load_config,
    results_dir,
    synth_dir,
    ws,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--systems", nargs="*", default=["base", "finetuned"],
                        help="Systems to verify (default: base, finetuned).")
    parser.add_argument("--out-dir", default="FINAL_OUTPUT",
                        help="Where the manifest is written.")
    return parser.parse_args()


REQUIRED_TOP_LEVEL = {
    "Checkpoint":    lambda cfg: final_model_path(cfg),
    "Train CSV":     lambda cfg: ws(cfg, "train.csv"),
    "Validation CSV":lambda cfg: ws(cfg, "val.csv"),
    "Test CSV":      lambda cfg: ws(cfg, "test.csv"),
}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    project = Path(cfg["workspace"]).resolve().parent
    workspace = Path(cfg["workspace"]).resolve()
    out_root = project / args.out_dir
    rdir = results_dir(cfg)

    required = dict(REQUIRED_TOP_LEVEL)
    for sys_name in args.systems:
        required[f"{sys_name} summary"] = rdir / f"{args.split}_{sys_name}_summary.json"
        required[f"{sys_name} details"] = rdir / f"{args.split}_{sys_name}_per_utt.csv"
    required["Comparison table"] = rdir / "comparison_table.csv"
    required["Wilcoxon"]         = rdir / "significance.json"
    required["Conclusion"]       = rdir / "conclusion.json"
    required["Audio preview"]    = workspace / "audio_preview.html"

    print(f"SECTION 13 — FINALIZATION")
    print(f"Project     : {project}")
    print(f"Workspace   : {workspace}")
    print(f"Output      : {out_root}")

    print(f"\nFILE VALIDATION")
    missing = []
    for name, path in required.items():
        ready = path.is_file() and path.stat().st_size > 0
        print(f"  {name:22s}: {'READY' if ready else 'MISSING'}")
        if not ready:
            missing.append(str(path))
    if missing:
        raise RuntimeError("Output belum lengkap:\n" + "\n".join(missing))

    ckpt = final_model_path(cfg)
    if ckpt.stat().st_size <= 4_000_000_000:
        raise RuntimeError(
            f"Checkpoint ukuran mencurigakan: {ckpt.stat().st_size} bytes"
        )

    test_df = pd.read_csv(ws(cfg, f"{args.split}.csv"), dtype={"id": str})
    if test_df.empty or test_df["id"].duplicated().any():
        raise RuntimeError("Test dataset is empty or has duplicate ids.")

    def rel(p: Path) -> str:
        return p.relative_to(project).as_posix()

    sample_rows = []
    missing_audio = []
    for row in test_df.itertuples(index=False):
        sid = str(row.id)
        per_system = {}
        ok = True
        for sys_name in args.systems:
            wav = synth_dir(cfg, sys_name, args.split) / f"{sid}.wav"
            if wav.is_file() and wav.stat().st_size > 1000:
                per_system[sys_name] = rel(wav)
            else:
                ok = False
        if not ok:
            missing_audio.append(sid)
            continue
        sample_rows.append({"id": sid, "text": str(row.text), **per_system})
    if missing_audio:
        raise RuntimeError(
            f"Audio belum lengkap untuk {len(missing_audio)} sampel; "
            f"contoh: {missing_audio[:5]}"
        )

    samples_df = pd.DataFrame(sample_rows)
    samples_path = out_root / "samples_manifest.csv"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(samples_path, index=False)

    evaluation = {}
    for sys_name in args.systems:
        sp = rdir / f"{args.split}_{sys_name}_summary.json"
        if sp.is_file():
            evaluation[sys_name] = json.loads(sp.read_text(encoding="utf-8"))

    sig = json.loads((rdir / "significance.json").read_text(encoding="utf-8")) \
        if (rdir / "significance.json").is_file() else {}
    con = json.loads((rdir / "conclusion.json").read_text(encoding="utf-8")) \
        if (rdir / "conclusion.json").is_file() else {}

    manifest = {
        "project": "Indonesian_TTS_FineTuning",
        "status": "COMPLETED",
        "final_checkpoint": rel(ckpt),
        "checkpoint_size_gib": round(ckpt.stat().st_size / 1024**3, 2),
        "test_samples": len(test_df),
        "datasets": {
            "train":      "workspace_full/train.csv",
            "validation": "workspace_full/val.csv",
            "test":       f"workspace_full/{args.split}.csv",
        },
        "audio": {
            sys_name: rel(synth_dir(cfg, sys_name, args.split))
            for sys_name in args.systems
        },
        "audio_preview": "workspace_full/audio_preview.html",
        "evaluation": evaluation,
        "significance": sig,
        "conclusion":   con,
        "results_directory": rel(rdir),
        "comparison_table":  rel(rdir / "comparison_table.csv"),
        "comparison_plots": [
            rel(p) for p in sorted(rdir.glob("comparison*.png"))
        ],
        "samples_manifest": rel(samples_path),
        "source_code": "src",
        "configuration": "configs/config.yaml",
        "engine": "F5-TTS (vendored @ tag 1.1.22)",
        "storage_policy": (
            "Semua file utama tetap pada lokasi aslinya. "
            "Tidak ada salinan checkpoint, dataset, atau audio."
        ),
    }
    write_json(out_root / "OUTPUT_MANIFEST.json", manifest)

    print(f"\nManifest    : {out_root / 'OUTPUT_MANIFEST.json'}")
    print(f"Samples     : {samples_path}")
    print(f"Status      : COMPLETED")


if __name__ == "__main__":
    main()
