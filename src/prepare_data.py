"""
prepare_data.py — preprocess raw PapaRazi/id-tts-v2 clips into F5-TTS-ready metadata.

The notebook drives this from Section 6. The CLI is provided for VS Code:

    python -m src.prepare_data --config configs/config.yaml
    python -m src.prepare_data --config configs/config.yaml --eda

What it does (in order):

  1. Walk the extracted shards and pair every `.wav` with its transcript.
  2. Drop clips that:
       * have no transcript / empty text
       * are too short (< min_duration) or too long (> max_duration)
       * are tagged English (PapaRazi wraps them in `[en]…[/en]`)
       * have a non-24 kHz sample rate (we re-sample but flag pathological files)
  3. Re-sample to mono 24 kHz and write the cleaned WAVs to
     `<workspace>/wavs/<id>.wav`.
  4. Split deterministically (seed = 42) into `train.csv / val.csv / test.csv`
     with the configured val_size and test_size.
  5. Emit `data_stats.json` so the EDA section (6.1) and the paper can quote
     the same drop counts.
  6. If `--eda` is passed, also write the EDA dataframe to
     `workspace_full/results/eda/eda_input.csv` so the EDA section is a
     pure-plotting step on top of an idempotent dataframe.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from tqdm.auto import tqdm

from common import extracted_dir, load_config, wavs_dir, ws, write_json


_EN_TAG_RE = re.compile(r"\[en\](.*?)\[/en\]", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--eda", action="store_true",
                        help="Also write the EDA dataframe (used by Section 6.1).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, deterministic).")
    return parser.parse_args()


def _list_audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def _read_transcript_for(audio_path: Path, transcripts_dir: Path | None) -> str | None:
    """Locate a matching transcript file (best-effort)."""
    candidates = [
        audio_path.with_suffix(".txt"),
        audio_path.with_suffix(".trans.txt"),
    ]
    if transcripts_dir is not None:
        candidates.append(transcripts_dir / f"{audio_path.stem}.txt")
    for c in candidates:
        if c.is_file():
            return c.read_text(encoding="utf-8", errors="ignore").strip()
    return None


def _is_english_tagged(text: str) -> bool:
    return bool(_EN_TAG_RE.search(text or ""))


def _strip_en_tag(text: str) -> str:
    return _EN_TAG_RE.sub("", text or "").strip()


def _resample_to_24k_mono(path: Path) -> tuple[np.ndarray, float]:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav.squeeze(0).cpu().numpy().astype(np.float32), 24000.0


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"[\u2000-\u206f\u2e00-\u2e7f'!\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~]",
                  " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_id(idx: int) -> str:
    return f"id_{idx:07d}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw = extracted_dir(cfg)
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw dataset not found: {raw}. Run get_dataset first.")

    out_wavs = wavs_dir(cfg)
    out_wavs.mkdir(parents=True, exist_ok=True)

    sample_rate = cfg["data"]["sample_rate"]
    min_dur = cfg["data"]["min_duration"]
    max_dur = cfg["data"]["max_duration"]
    min_words = cfg["data"]["min_words"]
    val_size = cfg["data"]["val_size"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["train"]["seed"]

    print(f"SECTION 6 — PREPROCESSING")
    print(f"Source      : {raw}")
    print(f"Output WAVs : {out_wavs}")
    print(f"Sample rate : {sample_rate} Hz")
    print(f"Duration    : {min_dur}–{max_dur} s")

    audio_files = _list_audio_files(raw)
    print(f"Discovered  : {len(audio_files):,} audio files")

    rng = random.Random(seed)
    rows = []
    dropped = {"english": 0, "rate": 0, "dur": 0, "empty": 0, "error": 0}

    for src_path in tqdm(audio_files, desc="Preprocess"):
        try:
            text = _read_transcript_for(src_path, transcripts_dir=None)
            if not text:
                dropped["empty"] += 1
                continue
            if _is_english_tagged(text):
                dropped["english"] += 1
                continue
            text = _strip_en_tag(text)
            words = _normalize_text(text).split()
            if len(words) < min_words:
                dropped["empty"] += 1
                continue

            wav, sr = _resample_to_24k_mono(src_path)
            if sr != sample_rate:
                dropped["rate"] += 1
                continue
            dur = wav.shape[0] / sr
            if dur < min_dur or dur > max_dur:
                dropped["dur"] += 1
                continue

            sample_id = _make_id(len(rows))
            out_path = out_wavs / f"{sample_id}.wav"
            sf.write(str(out_path), wav, int(sr), subtype="PCM_16")

            rows.append({
                "id": sample_id,
                "text": _normalize_text(text),
                "raw_text": text,
                "gt_audio": str(out_path.relative_to(Path(cfg["workspace"]))),
                "duration_sec": round(dur, 3),
                "n_words": len(words),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed on {src_path}: {exc}")
            dropped["error"] += 1
            continue

    df = pd.DataFrame(rows)
    n_valid = len(df)
    n_dropped = sum(dropped.values())
    print(f"\nValid samples  : {n_valid:,}")
    print(f"Dropped total  : {n_dropped:,}")
    for k, v in dropped.items():
        print(f"  - {k:9s}: {v:,}")

    # Deterministic split.
    rng.shuffle(rows)
    test_rows = rows[:test_size]
    val_rows = rows[test_size:test_size + val_size]
    train_rows = rows[test_size + val_size:]

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # Write the metadata CSVs.
    workspace = Path(cfg["workspace"])
    train_df.to_csv(ws(cfg, "train.csv"), index=False)
    val_df.to_csv(ws(cfg, "val.csv"), index=False)
    test_df.to_csv(ws(cfg, "test.csv"), index=False)

    metadata_all_path = ws(cfg, "metadata_all.csv")
    df.to_csv(metadata_all_path, index=False)

    # Aggregate stats for the paper / EDA.
    total_hours = float(df["duration_sec"].sum() / 3600.0)
    train_hours = float(train_df["duration_sec"].sum() / 3600.0)
    stats = {
        "n_initial": len(audio_files),
        "n_valid": n_valid,
        "n_dropped": n_dropped,
        "dropped": dropped,
        "n_train": len(train_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
        "total_hours": round(total_hours, 4),
        "train_hours": round(train_hours, 4),
        "sample_rate": sample_rate,
        "min_duration": min_dur,
        "max_duration": max_dur,
        "min_words": min_words,
        "seed": seed,
    }
    write_json(ws(cfg, "data_stats.json"), stats)

    print(f"\nSPLITS")
    print(f"Train : {len(train_df):,}  ({train_hours:.2f} h)")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")
    print(f"Total : {total_hours:.2f} h")
    print(f"Stats : {ws(cfg, 'data_stats.json')}")

    if args.eda:
        eda_df = df.copy()
        eda_df["eda_duration_sec"] = eda_df["duration_sec"]
        eda_df["eda_chars"] = eda_df["text"].str.len()
        eda_df["eda_words"] = eda_df["text"].str.split().str.len()
        eda_dir = ws(cfg, "results", "eda")
        eda_dir.mkdir(parents=True, exist_ok=True)
        eda_df.to_csv(eda_dir / "eda_input.csv", index=False)
        print(f"EDA input : {eda_dir / 'eda_input.csv'}")


if __name__ == "__main__":
    main()
