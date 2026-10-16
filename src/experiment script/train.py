"""Full fine-tuning F5-TTS sampai model_45000.pt. Tidak membuat backup checkpoint."""
import argparse
import json
import math
import subprocess
from pathlib import Path

from common import (
    ensure_engine_links,
    final_model_path,
    load_config,
    persistent_ckpt_dir,
    persistent_f5_data_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--num-gpus", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]

    final_model = final_model_path(cfg)
    if final_model.exists() and final_model.stat().st_size > 4_000_000_000 and not args.dry:
        print("✅ model_45000.pt sudah tersedia -> TRAINING SKIP")
        print(final_model)
        return

    if final_model.exists() and final_model.stat().st_size <= 4_000_000_000:
        raise RuntimeError(f'Checkpoint final ada tetapi ukuran mencurigakan. Jangan timpa: {final_model}')

    ensure_engine_links(cfg)

    data = persistent_f5_data_dir(cfg)
    dur_file = data / "duration.json"
    if not dur_file.exists():
        raise RuntimeError("Dataset internal F5 belum dibangun.")

    duration_obj = json.loads(dur_file.read_text(encoding="utf-8"))
    durations = duration_obj["duration"]

    frames_total = sum(durations) * 24000 / 256
    updates_per_epoch = max(
        1.0,
        frames_total / (
            float(tc["batch_size_per_gpu"])
            * args.num_gpus
            * max(1, int(tc["grad_accumulation_steps"]))
        ),
    )

    target_updates = int(tc["target_updates"])
    epochs = max(1, math.ceil(target_updates / updates_per_epoch))

    print(
        f"Durasi train {sum(durations)/3600:.2f} jam | "
        f"~{updates_per_epoch:.0f} update/epoch | "
        f"epochs={epochs} | target={target_updates}"
    )

    repo = Path(cfg["f5_repo"])

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", str(tc["mixed_precision"]),
        "--num_processes", str(args.num_gpus),
        "src/f5_tts/train/finetune_cli.py",
        "--exp_name", str(tc["exp_name"]),
        "--dataset_name", str(tc["dataset_name"]),
        "--tokenizer", str(tc["tokenizer"]),
        "--finetune",
        "--learning_rate", str(tc["learning_rate"]),
        "--batch_size_per_gpu", str(tc["batch_size_per_gpu"]),
        "--batch_size_type", str(tc["batch_size_type"]),
        "--max_samples", str(tc["max_samples"]),
        "--grad_accumulation_steps", str(tc["grad_accumulation_steps"]),
        "--max_grad_norm", str(tc["max_grad_norm"]),
        "--epochs", str(epochs),
        "--num_warmup_updates", str(tc["num_warmup_updates"]),
        "--save_per_updates", str(tc["save_per_updates"]),
        "--last_per_updates", str(tc["last_per_updates"]),
        "--keep_last_n_checkpoints", str(tc["keep_last_n_checkpoints"]),
        "--logger", str(tc["logger"]),
        "--log_samples",
    ]

    if tc.get("bnb_optimizer", False):
        cmd.append("--bnb_optimizer")

    print("CMD:", " ".join(cmd))
    print("Checkpoint langsung ke Drive:", persistent_ckpt_dir(cfg))

    if args.dry:
        print("✅ Dry run selesai")
        return

    subprocess.run(cmd, check=True, cwd=repo)

    if not final_model.exists():
        raise FileNotFoundError(
            f"Training selesai tetapi checkpoint final belum ditemukan: {final_model}"
        )

    print("✅ Training selesai:", final_model)


if __name__ == "__main__":
    main()
