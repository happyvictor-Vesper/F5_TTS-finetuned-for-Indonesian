"""Section 6 - Membentuk dataset internal F5-TTS dari train.csv."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import load_config, persistent_f5_data_dir, ensure_engine_links, engine_data_dir, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    f5_dir = Path(cfg["f5_repo"])
    sys.path.insert(0, str(f5_dir / "src"))

    from f5_tts.model.utils import convert_char_to_pinyin
    from datasets.arrow_writer import ArrowWriter

    train = ws(cfg, "train.csv")
    out = persistent_f5_data_dir(cfg)
    base_vocab = f5_dir / "src/f5_tts/infer/examples/vocab.txt"

    if not train.exists():
        raise FileNotFoundError(f"train.csv tidak ditemukan: {train}. Jalankan prepare_data.py.")
    if not base_vocab.exists():
        raise FileNotFoundError(f"Vocab F5-TTS tidak ditemukan: {base_vocab}")
    required_out = [out / "raw.arrow", out / "duration.json", out / "vocab.txt", out / "prepare_report.json"]
    if all(p.exists() and p.stat().st_size > 0 for p in required_out) and not args.force:
        print("✅ Dataset internal F5-TTS lengkap -> SKIP:", out)
        ensure_engine_links(cfg)
        return

    df = pd.read_csv(train)
    required = {"text", "gt_audio", "dur"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Kolom kurang: {missing}")

    with base_vocab.open(encoding="utf-8-sig") as f:
        vocab = [line.rstrip("\r\n") for line in f]
    vset = set(vocab)

    texts = df["text"].fillna("").astype(str).tolist()
    all_tokens = []
    batch_size = 2048
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenisasi"):
        all_tokens.extend(convert_char_to_pinyin(texts[i:i + batch_size], polyphone=True))

    items, durations = [], []
    dropped_oov = 0
    for row, tokens in tqdm(zip(df.itertuples(index=False), all_tokens), total=len(df), desc="Build Arrow"):
        if any(token not in vset for token in tokens):
            dropped_oov += 1
            continue
        duration = float(row.dur)
        items.append({"audio_path": str(Path(row.gt_audio).resolve()), "text": tokens, "duration": duration})
        durations.append(duration)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    writer = ArrowWriter(path=str(out / "raw.arrow"))
    for item in tqdm(items, desc="Writing raw.arrow"):
        writer.write(item)
    writer.finalize()

    (out / "duration.json").write_text(json.dumps({"duration": durations}), encoding="utf-8")
    shutil.copy2(base_vocab, out / "vocab.txt")
    report = {
        "samples": len(items),
        "hours": sum(durations) / 3600,
        "dropped_oov": dropped_oov,
        "vocab_size": len(vocab),
        "tokenizer": cfg["train"]["tokenizer"],
    }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ensure_engine_links(cfg)

    print(json.dumps(report, indent=2))
    print("✅ Dataset internal F5-TTS siap:", out)
    print("Engine link:", engine_data_dir(cfg))


if __name__ == "__main__":
    main()
