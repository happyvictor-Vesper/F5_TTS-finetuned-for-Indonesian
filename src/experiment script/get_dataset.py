import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    repo = dc["hf_repo"]
    parts = [int(x) for x in dc["parts"]]

    raw_root = Path(dc["raw_dir"])
    extract_dir = raw_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Download ZIP hanya staging sementara di runtime Colab.
    # Hasil persisten yang disimpan adalah extracted dataset di Google Drive.
    temp_cache = Path("/content/id_tts_download_cache")
    temp_cache.mkdir(parents=True, exist_ok=True)

    for idx in parts:
        marker = extract_dir / f".dataset{idx}.done"

        if marker.exists():
            print(f"✅ dataset{idx}.zip sudah tersedia -> SKIP")
            continue

        fname = f"dataset{idx}.zip"
        print(f"⬇️ {fname} belum ada -> DOWNLOAD")

        archive = hf_hub_download(
            repo_id=repo,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(temp_cache),
        )

        print(f"📦 Extract {fname} -> {extract_dir}")

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)

        marker.write_text("ok", encoding="utf-8")
        print(f"✅ {fname} selesai")

    shutil.rmtree(temp_cache, ignore_errors=True)

    missing = [
        i for i in parts
        if not (extract_dir / f".dataset{i}.done").exists()
    ]
    if missing:
        raise RuntimeError(f"Shard belum lengkap: {missing}")

    print("✅ Seluruh shard dataset tersedia di Drive:", extract_dir)


if __name__ == "__main__":
    main()
