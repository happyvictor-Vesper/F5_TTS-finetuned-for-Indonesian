"""Inference Base atau Fine-tuned final 45k. Resumable dan auto-skip."""
import argparse

from common import (
    final_model_path,
    load_config,
    synth_complete,
    synthesize_manifest,
    ws,
)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--system", required=True, choices=["base", "finetuned"])
ap.add_argument("--n", type=int, default=None)
a = ap.parse_args()

cfg = load_config(a.config)

if a.n is None and synth_complete(cfg, a.system, a.split):
    print(f"✅ synth/{a.split}/{a.system} lengkap -> SKIP")
    raise SystemExit(0)

ckpt = ""

if a.system == "finetuned":
    model = final_model_path(cfg)
    if not model.exists():
        raise FileNotFoundError(f"Model final belum tersedia: {model}")
    if model.stat().st_size < 4_000_000_000:
        raise RuntimeError(f"Ukuran checkpoint mencurigakan: {model.stat().st_size:,} bytes")
    ckpt = str(model)

synthesize_manifest(
    cfg,
    ws(cfg, f"{a.split}.csv"),
    ws(cfg, "synth", a.split, a.system),
    ckpt_file=ckpt,
    n=a.n,
    tag=f"{a.split}-{a.system}",
)
