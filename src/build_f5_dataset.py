"""
build_f5_dataset.py — convert train.csv into the F5-TTS internal dataset.

Produces, under `<workspace>/f5_data/<dataset_name>_pinyin/`:

  * `raw.arrow`     — tokenised samples in Arrow (F5-TTS reads this directly)
  * `duration.json` — `{ "duration": { "<id>": mel_frame_count, ... } }`
  * `vocab.txt`     — the *extended* character vocabulary (base + new chars)
  * `prepare_report.json` — { samples, hours, dropped_oov, vocab_size, ... }

What it actually does (in order):

  1. Wire up the engine symlinks so F5-TTS finds the data in the right place.
  2. Build a metadata CSV in the format F5-TTS's prepare_csv_wavs.py expects:
       `audio_file|text`  with absolute wav paths.
  3. Copy the base F5-TTS vocab (`Emilia_ZH_EN_pinyin/vocab.txt`) and extend
     it with every character that appears in the training transcripts but not
     in the base vocab. Indonesian is Latin-script so this is just a small
     addition (accent marks, digits, punctuation that PapaRazi uses).
  4. Invoke `F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py` to tokenise
     and pack the samples into `raw.arrow` + `duration.json`.

Because Indonesian uses Latin characters, the default `pinyin` tokenizer
collapses to character-level for it (Han characters get pinyin; everything
else is passed through unchanged).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import (
    ensure_engine_links,
    f5_data_dir_for_tokenizer,
    f5_repo_dir,
    load_config,
    persistent_f5_data_dir,
    wavs_dir,
    ws,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the F5-TTS prepare worker count.")
    return parser.parse_args()


def _base_vocab_path() -> Path:
    """The original Emilia_ZH_EN_pinyin vocab that ships with F5-TTS."""
    return (
        f5_repo_dir(load_config("configs/config.yaml"))
        / "data"
        / "Emilia_ZH_EN_pinyin"
        / "vocab.txt"
    )


def _extend_vocab(train_csv: Path, out_vocab: Path) -> tuple[int, int]:
    """
    Union the base F5-TTS vocab with every character seen in the training
    transcripts. Returns (vocab_size, new_chars_added).
    """
    base = _base_vocab_path()
    if not base.is_file():
        raise FileNotFoundError(f"Base vocab not found: {base}")

    base_chars = set(base.read_text(encoding="utf-8").split())
    transcripts = pd.read_csv(train_csv, dtype={"id": str})["text"].astype(str)
    seen = set()
    for text in transcripts:
        seen.update(text)

    new_chars = sorted(seen - base_chars)
    if new_chars:
        out_vocab.parent.mkdir(parents=True, exist_ok=True)
        with out_vocab.open("a", encoding="utf-8") as f:
            for ch in new_chars:
                f.write(f"{ch}\n")
    final = set(out_vocab.read_text(encoding="utf-8").split())
    return len(final), len(new_chars)


def _build_f5_metadata_csv(train_csv: Path, out_csv: Path, workspace: Path) -> None:
    """
    F5-TTS's prepare_csv_wavs.py expects a CSV with `audio_file|text` and
    *absolute* paths. Our train.csv stores `gt_audio` as a path relative to
    the workspace, so we materialise the absolute version here.
    """
    df = pd.read_csv(train_csv, dtype={"id": str})
    rows = []
    for row in df.itertuples(index=False):
        rel = str(row.gt_audio)
        wav_abs = (workspace / rel).resolve()
        if not wav_abs.is_file():
            raise FileNotFoundError(f"WAV not found: {wav_abs}")
        rows.append({"audio_file": str(wav_abs), "text": str(row.text)})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, header=True)


def _run_f5_prepare(cfg: dict, metadata_csv: Path, out_dir: Path,
                    workers: int | None) -> None:
    """Call F5-TTS's prepare_csv_wavs.py with our paths."""
    repo = f5_repo_dir(cfg)
    prepare_script = repo / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py"
    if not prepare_script.is_file():
        raise FileNotFoundError(
            f"prepare_csv_wavs.py not found at {prepare_script}. "
            "Make sure you cloned F5-TTS at tag 1.1.22."
        )

    cmd = [
        sys.executable, "-m", "f5_tts.train.datasets.prepare_csv_wavs",
        str(metadata_csv),
        str(out_dir),
    ]
    if workers is not None:
        cmd += ["--workers", str(workers)]

    print(f"\nRunning F5-TTS prepare step:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env={**__import__("os").environ,
                                        "PYTHONPATH": str(repo / "src")})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_engine_links(cfg)  # creates the F5-TTS/data symlink

    train_csv = ws(cfg, "train.csv")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}. "
                                "Run src/prepare_data.py first.")

    out_dir = persistent_f5_data_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / cfg["train"]["vocab_filename"]

    # Start with a clean copy of the base vocab, then extend.
    if vocab_path.exists():
        vocab_path.unlink()
    base = _base_vocab_path()
    shutil.copy2(base, vocab_path)

    print(f"SECTION 7 — F5-TTS INTERNAL DATASET")
    print(f"Input       : {train_csv}")
    print(f"Output      : {out_dir}")
    print(f"Base vocab  : {base}")
    print(f"Symlink at  : {f5_data_dir_for_tokenizer(cfg)}")

    vocab_size, new_chars = _extend_vocab(train_csv, vocab_path)
    print(f"Vocab size  : {vocab_size}  (+{new_chars} new chars)")

    # F5-TTS-format metadata (absolute wav paths) for prepare_csv_wavs.py
    f5_meta = out_dir / "f5_metadata.csv"
    _build_f5_metadata_csv(train_csv, f5_meta, Path(cfg["workspace"]).resolve())

    _run_f5_prepare(cfg, f5_meta, out_dir, args.workers)

    raw_arrow = out_dir / "raw.arrow"
    duration_json = out_dir / "duration.json"
    if not (raw_arrow.is_file() and duration_json.is_file() and vocab_path.is_file()):
        raise RuntimeError("F5-TTS dataset build did not produce all expected files.")

    data = json.loads(duration_json.read_text(encoding="utf-8"))
    durations = data["duration"] if isinstance(data, dict) and "duration" in data else data
    total_frames = int(sum(int(v) for v in durations.values()))
    hours = total_frames * 256 / (cfg["data"]["sample_rate"] * 3600.0)

    # OOV check on the extended vocab.
    transcripts = pd.read_csv(train_csv, dtype={"id": str})["text"].astype(str)
    vocab_chars = set(vocab_path.read_text(encoding="utf-8").split())
    oov = int(sum(1 for t in transcripts if any(c not in vocab_chars for c in t)))

    report = {
        "samples": len(transcripts),
        "hours": round(hours, 4),
        "dropped_oov": oov,
        "vocab_size": vocab_size,
        "new_chars_added": new_chars,
        "mel_hop_length": 256,
        "sample_rate": cfg["data"]["sample_rate"],
        "f5_data_link": str(f5_data_dir_for_tokenizer(cfg)),
    }
    write_json(out_dir / "prepare_report.json", report)

    print(f"\nINTERNAL DATASET RESULT")
    print(f"Samples     : {report['samples']:,}")
    print(f"Duration    : {report['hours']:.2f} hours")
    print(f"Dropped OOV : {report['dropped_oov']:,}")
    print(f"Vocab size  : {report['vocab_size']:,}")
    print(f"Status      : COMPLETED")
    print(f"Output      : {out_dir}")


if __name__ == "__main__":
    main()
