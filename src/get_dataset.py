"""
get_dataset.py — download PapaRazi/id-tts-v2 shards from HuggingFace.

Usage (the notebook uses Section 5 to drive this; CLI is provided for VS Code):

    python -m src.get_dataset --config configs/config.yaml
    python -m src.get_dataset --config configs/config.yaml --shards 1 2 3
    python -m src.get_dataset --config configs/config.yaml --keep-cache

Markers (`.datasetN.done`) are written next to each extracted shard so re-runs
are a no-op. The temporary download cache is removed on success unless
`--keep-cache` is passed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from common import extracted_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML config (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--shards", type=int, nargs="*", default=None,
        help="Subset of shard numbers to fetch (default: all configured parts)",
    )
    parser.add_argument(
        "--cache-dir", default="/content/id_tts_download_cache",
        help="Scratch download directory; cleaned on success unless --keep-cache",
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Do not remove the scratch download directory on success",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    repo = cfg["data"]["repo"]
    parts = args.shards or cfg["data"]["parts"]
    out_dir = extracted_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_dir)
    started = time.time()
    skipped = processed = 0
    extracted_files = 0

    print(f"SECTION 5 — DATASET DOWNLOAD")
    print(f"Repository  : {repo}")
    print(f"Total shards: {len(parts)}")
    print(f"Output      : {out_dir}")

    for idx in parts:
        filename = f"dataset{idx}.zip"
        marker = out_dir / f".dataset{idx}.done"

        print(f"\nShard {idx}/{len(parts)}: {filename}")

        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == "ok":
            print("Status      : SKIP")
            skipped += 1
            continue

        print("Status      : PROCESSING")
        t0 = time.time()
        cache.mkdir(parents=True, exist_ok=True)

        print("Downloading...")
        archive = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache),
        )
        print(f"Download    : {Path(archive).stat().st_size / 1024**3:.2f} GiB")
        print("Extracting...")

        file_count = 0
        with zipfile.ZipFile(archive, "r") as zf:
            for member in tqdm(zf.infolist(),
                               desc=f"Extract shard {idx}",
                               unit="file"):
                zf.extract(member, out_dir)
                if not member.is_dir():
                    file_count += 1

        marker.write_text("ok", encoding="utf-8")
        processed += 1
        extracted_files += file_count
        print(f"Files       : {file_count:,}")
        print(f"Duration    : {(time.time() - t0) / 60:.2f} minutes")

    completed = sum(
        (out_dir / f".dataset{i}.done").is_file()
        and (out_dir / f".dataset{i}.done").read_text(encoding="utf-8").strip() == "ok"
        for i in parts
    )
    if completed != len(parts):
        raise RuntimeError(f"Dataset incomplete: {completed}/{len(parts)}")

    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)

    print(f"\nDATASET RESULT")
    print(f"Completed shards : {completed}/{len(parts)}")
    print(f"Skipped shards   : {skipped}")
    print(f"Processed shards : {processed}")
    print(f"New files        : {extracted_files:,}")
    print(f"Execution time   : {(time.time() - started) / 60:.2f} minutes")
    print(f"Status           : COMPLETED")
    print(f"Output           : {out_dir}")


if __name__ == "__main__":
    main()
