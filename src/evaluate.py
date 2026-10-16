"""
evaluate.py — compute WER / CER / SIM / SQUIM on a synth manifest.

Per-(system, split) outputs land in `<workspace>/results/`:

  * `test_<system>_summary.json` — aggregate (WER, CER, SIM_prompt, SIM_gt, n)
  * `test_<system>_per_utt.csv`   — per-utterance (id, ref, hyp, WER, CER, SIM)

The evaluator is the same `Evaluator` class the notebook instantiates
(Section 8 / Section 10), so the numbers are bit-identical between the
notebook and the CLI.

CLI:

    python -m src.evaluate --config configs/config.yaml --split test --system base
    python -m src.evaluate --config configs/config.yaml --split test --system finetuned
    python -m src.evaluate --config configs/config.yaml --split test --system gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import (
    Evaluator,
    for_scoring,
    load_config,
    results_dir,
    synth_dir,
    ws,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--system", required=True, choices=["base", "finetuned", "gt"])
    return parser.parse_args()


def _safe_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Per the paper: an empty hypothesis scores 1.0 on both metrics."""
    if not reference:
        return 1.0, 1.0
    if not hypothesis:
        return 1.0, 1.0
    return float(jiwer.wer(reference, hypothesis)), float(jiwer.cer(reference, hypothesis))


def _wer_breakdown(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """Sub / Del / Ins rates against the reference word count."""
    ref_words = reference.split()
    n = max(len(ref_words), 1)
    if not hypothesis:
        return 0.0, 1.0, 0.0
    out = jiwer.process_words([reference], [hypothesis])
    if not out.alignments or not out.alignments[0]:
        return 0.0, 0.0, 0.0
    ops = out.alignments[0].ops  # list of (tag, ref_idx, hyp_idx)
    sub = del_ = ins = 0
    for tag, *_ in ops:
        if tag == "substitute":
            sub += 1
        elif tag == "delete":
            del_ += 1
        elif tag == "insert":
            ins += 1
    return sub / n, del_ / n, ins / n


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    manifest = ws(cfg, f"{args.split}.csv")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    df = pd.read_csv(manifest, dtype={"id": str})
    synth = synth_dir(cfg, args.system, args.split)
    if not synth.is_dir():
        raise FileNotFoundError(f"Synth dir not found: {synth}. Run src/infer.py first.")

    prompt_audio = Path(cfg["infer"]["ref_audio"])
    if not prompt_audio.is_file():
        alt = Path(cfg["workspace"]).parent / cfg["infer"]["ref_audio"]
        if alt.is_file():
            prompt_audio = alt
        else:
            raise FileNotFoundError(f"Reference audio not found: {prompt_audio}")

    out_dir = results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.split}_{args.system}_summary.json"
    detail_path = out_dir / f"{args.split}_{args.system}_per_utt.csv"

    rows = []
    evaluator = Evaluator(cfg)

    try:
        for row in tqdm(df.itertuples(index=False), total=len(df),
                        desc=f"eval {args.system}/{args.split}"):
            sample_id = str(row.id)
            reference = for_scoring(str(row.text))
            gen_wav = synth / f"{sample_id}.wav"
            if not gen_wav.is_file() or gen_wav.stat().st_size < 1000:
                hypothesis = ""
                sim_prompt = sim_gt = None
            else:
                hypothesis = for_scoring(evaluator.transcribe(str(gen_wav)))
                sim_prompt = evaluator.speaker_similarity(str(prompt_audio), str(gen_wav))
                sim_gt = evaluator.speaker_similarity(
                    str(Path(cfg["workspace"]) / str(row.gt_audio)),
                    str(gen_wav),
                )

            wer, cer = _safe_wer_cer(reference, hypothesis)
            sub, dele, ins = _wer_breakdown(reference, hypothesis)

            rows.append({
                "id": sample_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "WER": wer,
                "CER": cer,
                "Sub%": sub,
                "Del%": dele,
                "Ins%": ins,
                "SIM_prompt": sim_prompt,
                "SIM_gt": sim_gt,
            })
    finally:
        evaluator.free()

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False)

    summary = {
        "n": len(detail),
        "split": args.split,
        "system": args.system,
        "WER": float(detail["WER"].mean()),
        "CER": float(detail["CER"].mean()),
        "Sub%": float(detail["Sub%"].mean()),
        "Del%": float(detail["Del%"].mean()),
        "Ins%": float(detail["Ins%"].mean()),
        "SIM_prompt": float(np.nanmean(detail["SIM_prompt"])) if "SIM_prompt" in detail else None,
        "SIM_gt":     float(np.nanmean(detail["SIM_gt"]))     if "SIM_gt"     in detail else None,
    }
    write_json(summary_path, summary)

    print(f"\nEVALUATION RESULT — system={args.system}  split={args.split}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:11s}: {v:.4f}")
        else:
            print(f"  {k:11s}: {v}")
    print(f"Summary     : {summary_path}")
    print(f"Per-utt     : {detail_path}")


if __name__ == "__main__":
    main()
