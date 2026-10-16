"""
preview_audio.py — build a static HTML page that pairs base / fine-tuned audio
from a fixed N-sample slice of the test split.

The notebook does this inline in Section 12; this CLI makes it reproducible
from VS Code:

    python -m src.preview_audio --config configs/config.yaml --n 5

The output is `<workspace>/audio_preview.html` — a self-contained file that
references the per-sample WAVs with relative paths, so it can be opened from
anywhere as long as the directory structure is preserved.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from common import load_config, synth_dir, ws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of test samples to include in the preview.")
    parser.add_argument("--system-a", default="base")
    parser.add_argument("--system-b", default="finetuned")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None,
                        help="Override the output HTML path.")
    return parser.parse_args()


def _build_html(workspace: Path, df: pd.DataFrame, system_a: str, system_b: str,
                split: str) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html lang='id'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>F5-TTS Audio Comparison</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:20px}",
        ".item{padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:10px}",
        ".audio-row{display:flex;gap:20px;flex-wrap:wrap}",
        ".audio-box{flex:1;min-width:260px}",
        "audio{width:100%}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Base vs Fine-Tuned F5-TTS — {split} split</h1>",
        f"<p>Per-sample comparison of <b>{system_a}</b> and <b>{system_b}</b>.</p>",
    ]

    for i, row in enumerate(df.itertuples(index=False), start=1):
        sample_id = str(row.id)
        text = str(row.text)
        a_path = synth_dir_path(workspace, system_a, split) / f"{sample_id}.wav"
        b_path = synth_dir_path(workspace, system_b, split) / f"{sample_id}.wav"
        for p in (a_path, b_path):
            if not p.is_file() or p.stat().st_size < 1000:
                raise FileNotFoundError(f"Missing synth audio: {p}")
        a_src = quote(a_path.relative_to(workspace).as_posix(), safe="/")
        b_src = quote(b_path.relative_to(workspace).as_posix(), safe="/")
        parts.extend([
            "<div class='item'>",
            f"<h3>Sample {i} — {escape(sample_id)}</h3>",
            f"<p><b>Text:</b> {escape(text)}</p>",
            "<div class='audio-row'>",
            "<div class='audio-box'>",
            f"<p><b>{escape(system_a)}</b></p>",
            f'<audio controls preload="metadata" src="{escape(a_src, quote=True)}"></audio>',
            "</div>",
            "<div class='audio-box'>",
            f"<p><b>{escape(system_b)}</b></p>",
            f'<audio controls preload="metadata" src="{escape(b_src, quote=True)}"></audio>',
            "</div>",
            "</div>",
            "</div>",
        ])

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def synth_dir_path(workspace: Path, system: str, split: str) -> Path:
    return workspace / "synth" / split / system


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    workspace = Path(cfg["workspace"])
    manifest = workspace / f"{args.split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    df = pd.read_csv(manifest, dtype={"id": str}).head(args.n)
    if len(df) < args.n:
        raise RuntimeError(f"Need {args.n} samples; found {len(df)}.")

    out = Path(args.out) if args.out else workspace / "audio_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    content = _build_html(workspace, df, args.system_a, args.system_b, args.split)
    if out.is_file() and out.read_text(encoding="utf-8") == content:
        print(f"Status : SKIP  (unchanged)\nOutput : {out}")
        return
    out.write_text(content, encoding="utf-8")
    print(f"Status : COMPLETED\nOutput : {out}")


if __name__ == "__main__":
    main()
