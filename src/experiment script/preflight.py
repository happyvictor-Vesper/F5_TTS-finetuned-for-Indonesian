"""Cek komponen project sebelum pipeline berjalan."""
import argparse
from pathlib import Path

from common import final_model_path, load_config, persistent_f5_data_dir, ws


def show(label, path, required=True):
    p = Path(path)
    ok = p.exists()
    mark = "✅" if ok else ("❌" if required else "⚠️")
    extra = ""
    if ok and p.is_file():
        extra = f" ({p.stat().st_size:,} bytes)"
    print(f"{mark} {label:<25} {p}{extra}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_project_root"])
    f5 = Path(cfg["f5_repo"])

    print("=== SOURCE + ENGINE ===")

    required = [
        ("F5 engine", f5 / "src/f5_tts"),
        ("finetune CLI", f5 / "src/f5_tts/train/finetune_cli.py"),
        ("get_dataset.py", root / "src/get_dataset.py"),
        ("prepare_data.py", root / "src/prepare_data.py"),
        ("build_f5_dataset.py", root / "src/build_f5_dataset.py"),
        ("train.py", root / "src/train.py"),
        ("infer.py", root / "src/infer.py"),
        ("evaluate.py", root / "src/evaluate.py"),
        ("compare.py", root / "src/compare.py"),
        ("preview_audio.py", root / "src/preview_audio.py"),
        ("finalize.py", root / "src/finalize.py"),
    ]

    failed = []
    for label, path in required:
        if not show(label, path):
            failed.append(str(path))

    print("\n=== STATUS PIPELINE ===")
    show("metadata_all.csv", ws(cfg, "metadata_all.csv"), required=False)
    show("train.csv", ws(cfg, "train.csv"), required=False)
    show("val.csv", ws(cfg, "val.csv"), required=False)
    show("test.csv", ws(cfg, "test.csv"), required=False)
    show("vocab.txt", persistent_f5_data_dir(cfg) / "vocab.txt", required=False)

    model = final_model_path(cfg)
    model_ok = show("model_45000.pt", model, required=False)

    if model_ok and model.stat().st_size > 4_000_000_000:
        print("✅ Model final ada -> Section Training nanti SKIP")
    else:
        print("ℹ️ Model final belum ada -> Section Training nanti dijalankan")

    if failed:
        raise FileNotFoundError("Komponen source belum lengkap:\n" + "\n".join(failed))

    print("\n✅ Preflight selesai")


if __name__ == "__main__":
    main()
