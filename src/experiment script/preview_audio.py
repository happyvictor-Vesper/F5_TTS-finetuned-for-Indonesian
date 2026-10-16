"""Membuat HTML preview audio Base vs Fine-tuned. Auto-skip."""
import argparse
import html

import pandas as pd

from common import load_config, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ws(cfg, "audio_preview.html")

    if out.exists() and out.stat().st_size > 0 and not args.force:
        print("✅ audio_preview.html sudah ada -> SKIP")
        print(out)
        return

    test = pd.read_csv(ws(cfg, "test.csv")).head(args.n)

    blocks = [
        "<meta charset='utf-8'><title>F5-TTS Audio Preview</title>",
        "<style>body{font-family:Arial;max-width:900px;margin:40px auto}.item{margin:28px 0;padding:18px;border:1px solid #ddd;border-radius:10px}audio{width:100%}</style>",
        "<h1>Base vs Fine-tuned 45k</h1>",
    ]

    for _, r in test.iterrows():
        blocks.append("<div class='item'>")
        blocks.append(f"<p><b>Text:</b> {html.escape(str(r['text']))}</p>")

        for system, label in [
            ("base", "Base F5-TTS"),
            ("finetuned", "Fine-tuned 45k"),
        ]:
            wav = ws(cfg, "synth", "test", system, f"{r['id']}.wav")

            if wav.exists():
                blocks.append(
                    f"<p><b>{label}</b></p>"
                    f"<audio controls src='{wav.resolve().as_uri()}'></audio>"
                )
            else:
                blocks.append(f"<p><b>{label}</b>: file belum tersedia</p>")

        blocks.append("</div>")

    out.write_text("\n".join(blocks), encoding="utf-8")
    print("✅ Preview dibuat:", out)


if __name__ == "__main__":
    main()
