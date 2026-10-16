"""Dataset -> 24 kHz mono -> normalisasi -> split train/val/test -> metadata F5-TTS.

Mendukung:
- Dataset hasil extract dari get_dataset.py
- Folder lokal dengan metadata CSV/TSV/TXT/JSON/JSONL/Parquet atau sidecar .txt/.lab
"""
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from common import load_config, ws
from text_norm import normalize_id

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
META_EXTS = {".csv", ".tsv", ".txt", ".lst", ".list", ".json", ".jsonl", ".parquet"}
TEXT_CANDS = ["normalized_text", "text", "sentence", "transcription", "transcript", "raw_text", "utterance", "kalimat", "teks"]
SPK_CANDS = ["speaker", "speaker_id", "client_id", "spk", "speaker_name", "spk_id", "voice", "voice_id"]
AUDIO_CANDS = ["audio", "audio_path", "path", "file", "filename", "wav", "audio_file", "file_name", "wav_path"]


def pick(cols, candidates, given=None):
    if given:
        return given if given in cols else None
    low = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _speaker_from_path(audio: Path, root: Path) -> str:
    try:
        rel = audio.relative_to(root)
        parts = rel.parts[:-1]
    except Exception:
        parts = audio.parts[:-1]
    generic = {"wav", "wavs", "audio", "audios", "clips", "data", "dataset", "train"}
    for p in reversed(parts):
        if p.lower() not in generic and not re.fullmatch(r"dataset\d+", p.lower()):
            return p
    return "spk0"


def _uid_for(audio: Path, root: Path) -> str:
    try:
        rel = audio.relative_to(root).as_posix()
    except Exception:
        rel = str(audio)
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", audio.stem)[:48].strip("_") or "utt"
    digest = hashlib.md5(rel.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{stem}_{digest}"


def _audio_index(root: Path):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    by_name, by_stem, by_rel = {}, {}, {}
    for p in files:
        by_name.setdefault(p.name.lower(), []).append(p)
        by_stem.setdefault(p.stem.lower(), []).append(p)
        try:
            by_rel[p.relative_to(root).as_posix().lower()] = p
        except Exception:
            pass
    return files, by_name, by_stem, by_rel


def _resolve_audio(value, meta_file: Path, root: Path, by_name, by_stem, by_rel):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    raw = str(value).strip().strip('"').strip("'").replace("\\", "/")
    if not raw:
        return None
    raw = raw.removeprefix("./")
    candidates = [
        Path(raw), meta_file.parent / raw, root / raw,
        meta_file.parent / "wavs" / raw, meta_file.parent / "audio" / raw,
        root / "wavs" / raw, root / "audio" / raw,
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_file() and c.suffix.lower() in AUDIO_EXTS:
                return c.resolve()
        except Exception:
            pass
    rel = raw.lower()
    if rel in by_rel:
        return by_rel[rel].resolve()
    name = Path(raw).name.lower()
    if name in by_name and len(by_name[name]) == 1:
        return by_name[name][0].resolve()
    stem = Path(raw).stem.lower()
    if stem in by_stem and len(by_stem[stem]) == 1:
        return by_stem[stem][0].resolve()
    if Path(raw).suffix == "":
        matches = []
        for ext in AUDIO_EXTS:
            matches.extend(by_name.get((Path(raw).name + ext).lower(), []))
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def _iter_table_records(meta: Path):
    suffix = meta.suffix.lower()
    if suffix == ".jsonl":
        try:
            df = pd.read_json(meta, lines=True)
            yield df
        except Exception:
            return
    elif suffix == ".json":
        try:
            obj = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list):
                yield pd.DataFrame(obj)
            elif isinstance(obj, dict):
                if all(isinstance(v, list) for v in obj.values()):
                    yield pd.DataFrame(obj)
                else:
                    yield pd.DataFrame([obj])
        except Exception:
            return
    elif suffix == ".parquet":
        try:
            yield pd.read_parquet(meta)
        except Exception:
            return
    else:
        try:
            df = pd.read_csv(meta, sep=None, engine="python", dtype=str, on_bad_lines="skip")
            if len(df.columns) >= 2:
                yield df
        except Exception:
            return


def discover_pairs(root: Path, audio_column=None, text_column=None, speaker_column=None):
    root = Path(root)
    audio_files, by_name, by_stem, by_rel = _audio_index(root)
    print(f"File audio ditemukan: {len(audio_files):,}")
    if not audio_files:
        raise RuntimeError(f"Tidak ada audio di {root}")

    rows = []
    seen_audio = set()
    meta_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in META_EXTS)
    print(f"Kandidat metadata: {len(meta_files):,}")

    def add(audio_value, text_value, speaker_value, meta_file):
        audio = _resolve_audio(audio_value, meta_file, root, by_name, by_stem, by_rel)
        if audio is None:
            return False
        text = "" if text_value is None else str(text_value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return False
        key = str(audio)
        if key in seen_audio:
            return False
        spk = str(speaker_value).strip() if speaker_value is not None and str(speaker_value).strip().lower() not in {"", "nan", "none"} else _speaker_from_path(audio, root)
        rows.append((audio, text, spk))
        seen_audio.add(key)
        return True

    for meta in meta_files:
        parsed_header = False
        for df in _iter_table_records(meta):
            if df.empty:
                continue
            acol = pick(df.columns, AUDIO_CANDS, audio_column)
            tcol = pick(df.columns, TEXT_CANDS, text_column)
            scol = pick(df.columns, SPK_CANDS, speaker_column)
            if acol is not None and tcol is not None:
                before = len(rows)
                for _, r in df.iterrows():
                    add(r.get(acol), r.get(tcol), r.get(scol) if scol else None, meta)
                if len(rows) > before:
                    parsed_header = True
        if parsed_header or meta.suffix.lower() in {".json", ".jsonl", ".parquet"}:
            continue

        try:
            lines = meta.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        if len(lines) > 200000:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            fields = None
            for delim in ("|", "\t", ","):
                try:
                    parsed = next(csv.reader([line], delimiter=delim))
                except Exception:
                    continue
                if len(parsed) >= 2:
                    fields = [x.strip() for x in parsed]
                    break
            if not fields:
                continue
            resolved_idx = None
            resolved_path = None
            for i, field in enumerate(fields[:4]):
                p = _resolve_audio(field, meta, root, by_name, by_stem, by_rel)
                if p is not None:
                    resolved_idx, resolved_path = i, p
                    break
            if resolved_idx is None:
                continue
            text_fields = [f for i, f in enumerate(fields) if i != resolved_idx and f and f.lower() not in {"text", "transcript", "transcription", "normalized_text"}]
            if not text_fields:
                continue
            text = text_fields[-1] if len(text_fields) >= 2 else text_fields[0]
            if str(resolved_path) not in seen_audio:
                rows.append((resolved_path, text, _speaker_from_path(resolved_path, root)))
                seen_audio.add(str(resolved_path))

    # Sidecar transcript (audio.wav + audio.txt/.lab)
    for audio in audio_files:
        if str(audio.resolve()) in seen_audio:
            continue
        for ext in (".txt", ".lab"):
            side = audio.with_suffix(ext)
            if side.exists():
                text = side.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    rows.append((audio.resolve(), text, _speaker_from_path(audio, root)))
                    seen_audio.add(str(audio.resolve()))
                break

    if not rows:
        ext_counts = Counter(p.suffix.lower() or "<noext>" for p in root.rglob("*") if p.is_file())
        sample = [str(p.relative_to(root)) for p in list(root.rglob("*")) if p.is_file()][:80]
        raise RuntimeError(
            "Metadata pasangan audio-text tidak terdeteksi.\n"
            f"Ekstensi file: {dict(ext_counts.most_common(20))}\n"
            "Contoh file:\n" + "\n".join(sample)
        )
    print(f"Pasangan audio-teks ditemukan: {len(rows):,}")
    return rows


def process_pairs(pairs, cfg):
    dc, seed = cfg["data"], cfg["seed"]
    sr_t = int(dc["sample_rate"])
    wav_dir = ws(cfg, "wavs")
    wav_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    max_items = dc.get("max_items")
    max_hours = dc.get("max_hours")
    rows = []
    drop = dict(dur=0, rate=0, empty=0, error=0, english=0)
    accepted_seconds = 0.0

    for k, (audio_path, text, spk) in enumerate(tqdm(pairs, desc="preprocessing audio")):
        if max_items and k >= int(max_items):
            break
        if max_hours and accepted_seconds >= float(max_hours) * 3600:
            break
        try:
            raw_text = str(text)
            if dc.get("drop_english_tagged", True) and re.search(r"\[en\]", raw_text, flags=re.I):
                drop["english"] += 1
                continue
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != sr_t:
                y = librosa.resample(y, orig_sr=sr, target_sr=sr_t)
            if dc.get("trim_silence", True):
                _, idx = librosa.effects.trim(y, top_db=float(dc.get("trim_top_db", 35)))
                pad = int(0.1 * sr_t)
                y = y[max(0, idx[0] - pad): min(len(y), idx[1] + pad)]
            dur = len(y) / sr_t
            t = normalize_id(raw_text)
            if not t:
                drop["empty"] += 1
                continue
            if not (float(dc["min_dur"]) <= dur <= float(dc["max_dur"])):
                drop["dur"] += 1
                continue
            cps = len(t) / max(dur, 1e-6)
            if not (float(dc["min_chars_per_sec"]) <= cps <= float(dc["max_chars_per_sec"])):
                drop["rate"] += 1
                continue
            peak = float(np.abs(y).max()) if len(y) else 0.0
            if peak > 0:
                y = y / peak * 0.95
            uid = _uid_for(Path(audio_path), Path(audio_path).anchor and Path(audio_path).parent or Path("."))
            # Hindari collision UID walau nama file sama
            uid = f"{uid}_{len(rows):06d}"
            out = wav_dir / f"{uid}.wav"
            # Skip penulisan ulang WAV yang sudah ada (resume preprocessing lebih cepat).
            # Bila filter audio (trim/sample_rate) diubah, jalankan dengan --force agar WAV dibuat ulang.
            if not out.exists() or out.stat().st_size < 20000:
                sf.write(out, y, sr_t, subtype="PCM_16")
            rows.append(dict(id=uid, text=t, gt_audio=str(out.resolve()), speaker=str(spk), dur=round(dur, 3)))
            accepted_seconds += dur
        except Exception as exc:
            drop["error"] += 1
            if drop["error"] <= 20:
                print("skip", audio_path, exc)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Tidak ada data lolos filter. Periksa parser atau longgarkan filter di config.")
    print(f"Lolos filter: {len(df):,} utt, {df.dur.sum()/3600:.2f} jam | dibuang: {drop}")
    return df, drop


def split_and_write(df, cfg, drop):
    dc, seed = cfg["data"], cfg["seed"]
    rng = random.Random(seed)
    groups = list(df.groupby("text").groups.values())
    rng.shuffle(groups)
    n = len(df)
    ntest = min(int(dc["test_size"]), max(1, n // 10))
    nval = min(int(dc["val_size"]), max(1, n // 20))
    split, counts = {}, {"test": 0, "val": 0}
    for g in groups:
        s = "test" if counts["test"] < ntest else "val" if counts["val"] < nval else "train"
        if s in counts:
            counts[s] += len(g)
        for i in g:
            split[i] = s
    df["split"] = df.index.map(split)
    ws(cfg).mkdir(parents=True, exist_ok=True)
    df.to_csv(ws(cfg, "metadata_all.csv"), index=False)

    train = df[df.split == "train"].copy()
    if train.empty:
        raise RuntimeError("Train split kosong.")
    preferred_global = train[(train.dur >= float(dc["ref_min_dur"])) & (train.dur <= float(dc["ref_max_dur"]))]
    if preferred_global.empty:
        preferred_global = train
    pools = {}
    for spk, g in train.groupby("speaker"):
        p = g[(g.dur >= float(dc["ref_min_dur"])) & (g.dur <= float(dc["ref_max_dur"]))]
        pools[spk] = p if not p.empty else g

    for name in ("train", "val", "test"):
        part = df[df.split == name].copy()
        refs = []
        for _, r in part.iterrows():
            pool = pools.get(r["speaker"], preferred_global)
            if len(pool) > 1:
                pool = pool[pool.id != r["id"]]
            if pool.empty:
                pool = preferred_global
            h = int(hashlib.md5(str(r["id"]).encode()).hexdigest(), 16)
            ref = pool.iloc[h % len(pool)]
            refs.append((ref["gt_audio"], ref["text"]))
        if len(part):
            part["ref_audio"], part["ref_text"] = zip(*refs)
        else:
            part["ref_audio"], part["ref_text"] = [], []
        part.to_csv(ws(cfg, f"{name}.csv"), index=False)

    with open(ws(cfg, "f5_metadata.csv"), "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for _, r in train.iterrows():
            f.write(f"{Path(r.gt_audio).resolve()}|{str(r.text).replace('|', ' ')}\n")

    stats = {
        "total": int(len(df)),
        "hours": round(float(df.dur.sum()) / 3600, 2),
        "dropped": drop,
        "n_speakers": int(df.speaker.nunique()),
        "split": {s: int((df.split == s).sum()) for s in ("train", "val", "test")},
        "train_hours": round(float(df[df.split == "train"].dur.sum()) / 3600, 2),
    }
    with open(ws(cfg, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _dataset_ready(cfg):
    required = [
        ws(cfg, "metadata_all.csv"),
        ws(cfg, "train.csv"),
        ws(cfg, "val.csv"),
        ws(cfg, "test.csv"),
        ws(cfg, "f5_metadata.csv"),
        ws(cfg, "data_stats.json"),
    ]
    if not all(p.exists() and p.stat().st_size > 0 for p in required):
        return False

    wav_dir = ws(cfg, "wavs")
    if not wav_dir.exists():
        return False

    try:
        train = pd.read_csv(ws(cfg, "train.csv"))
        val = pd.read_csv(ws(cfg, "val.csv"))
        test = pd.read_csv(ws(cfg, "test.csv"))
    except Exception:
        return False

    if len(val) != int(cfg["data"]["val_size"]) or len(test) != int(cfg["data"]["test_size"]):
        return False

    expected = len(train) + len(val) + len(test)
    actual_wavs = sum(1 for p in wav_dir.glob("*.wav") if p.stat().st_size > 1000)
    return actual_wavs >= expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]

    if _dataset_ready(cfg) and not args.force:
        print("✅ Preprocessing lengkap sudah tersedia -> SKIP")
        print("Workspace:", ws(cfg))
        return

    if args.force and ws(cfg, "wavs").exists():
        shutil.rmtree(ws(cfg, "wavs"))

    if dc["source"] == "paparazi_zip":
        raw_root = Path(dc["raw_dir"])
        extracted = raw_root / "extracted"
        markers = [extracted / f".dataset{i}.done" for i in dc.get("parts", range(1, 11))]
        if not extracted.exists() or not all(m.exists() for m in markers):
            raise FileNotFoundError(
                f"Dataset belum lengkap: {extracted}. "
                "Jalankan src/get_dataset.py terlebih dahulu."
            )
        pairs = discover_pairs(
            extracted,
            dc.get("audio_column"),
            dc.get("text_column"),
            dc.get("speaker_column"),
        )
    elif dc["source"] == "local":
        root = Path(dc["local_dir"]).expanduser().resolve()
        pairs = discover_pairs(root, dc.get("audio_column"), dc.get("text_column"), dc.get("speaker_column"))
    else:
        raise ValueError("data.source harus 'paparazi_zip' atau 'local'")

    df, drop = process_pairs(pairs, cfg)
    split_and_write(df, cfg, drop)

    if not _dataset_ready(cfg):
        raise RuntimeError("Preprocessing selesai tetapi verifikasi output belum lengkap.")

    print("✅ Preprocessing selesai dan tervalidasi.")


if __name__ == "__main__":
    main()
