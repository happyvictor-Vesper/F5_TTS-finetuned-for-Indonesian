"""
compare.py — paired comparison of two systems on the same test split.

Reads `test_<a>_per_utt.csv` and `test_<b>_per_utt.csv` from
`<workspace>/results/`, joins on `id`, computes the per-pair deltas, and
emits:

  * `comparison_table.csv` / `.md` — aggregate Base vs FT-General (mean, Δ, p)
  * `significance.json`            — Wilcoxon signed-rank p-values per metric
  * `conclusion.json`              — base vs finetuned summary + verdict
  * `comparison_<metric>_box.png`  — paired box plots for WER / CER / SIM
  * `improvement_status.csv`       — per-metric UP / DOWN / FLAT verdict

CLI:

    python -m src.compare --config configs/config.yaml --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import load_config, results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--a", default="base")
    parser.add_argument("--b", default="finetuned")
    return parser.parse_args()


METRICS_LOWER_BETTER = {"WER", "CER", "Sub%", "Del%", "Ins%"}


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank. Returns the p-value (NaN if undefined)."""
    if len(a) < 2:
        return float("nan")
    diff = b - a
    if np.all(diff == 0):
        return float("nan")
    try:
        _, p = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def _box_plot(a: pd.Series, b: pd.Series, label_a: str, label_b: str,
              title: str, out: Path, lower_better: bool) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([a.dropna().values, b.dropna().values],
               tick_labels=[label_a, label_b], showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("value (lower better)" if lower_better else "value (higher better)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = results_dir(cfg)

    pa = out_dir / f"{args.split}_{args.a}_per_utt.csv"
    pb = out_dir / f"{args.split}_{args.b}_per_utt.csv"
    sa = out_dir / f"{args.split}_{args.a}_summary.json"
    sb = out_dir / f"{args.split}_{args.b}_summary.json"

    for p in (pa, pb, sa, sb):
        if not p.is_file():
            raise FileNotFoundError(p)

    df_a = pd.read_csv(pa, dtype={"id": str})
    df_b = pd.read_csv(pb, dtype={"id": str})
    sum_a = json.loads(sa.read_text(encoding="utf-8"))
    sum_b = json.loads(sb.read_text(encoding="utf-8"))

    if df_a["id"].duplicated().any() or df_b["id"].duplicated().any():
        raise RuntimeError("Duplicate ids in one of the per-utt CSVs.")
    if set(df_a["id"]) != set(df_b["id"]):
        raise RuntimeError("Base and FT per-utt CSVs do not share the same id set.")

    merged = df_a.merge(df_b, on="id", suffixes=(f"_{args.a}", f"_{args.b}"))
    metrics = ["WER", "CER", "SIM_prompt", "SIM_gt"]

    table_rows = []
    sig = {}
    for m in metrics:
        a = merged[f"{m}_{args.a}"]
        b = merged[f"{m}_{args.b}"]
        a_clean, b_clean = a.dropna(), b.dropna()
        common = a_clean.index.intersection(b_clean.index)
        p_w = _wilcoxon(a_clean.loc[common].values, b_clean.loc[common].values)
        table_rows.append({
            "metric": m,
            f"{args.a}": float(a.mean()),
            f"{args.b}": float(b.mean()),
            "delta":   float(b.mean() - a.mean()),
            "p_wilcoxon": p_w,
            "n":       int(len(common)),
        })
        sig[m] = {"p_wilcoxon": p_w, "n": int(len(common))}

    table = pd.DataFrame(table_rows).set_index("metric")
    table.to_csv(out_dir / "comparison_table.csv")
    table.round(4).to_markdown(out_dir / "comparison_table.md")

    improvement = []
    for m in metrics:
        old = sum_a[m]; new = sum_b[m]
        better = (new < old) if m in METRICS_LOWER_BETTER else (new > old)
        improvement.append({
            "metric": m,
            args.a: old,
            args.b: new,
            "delta": new - old,
            "status": "IMPROVED" if better else ("UNCHANGED" if new == old else "DECREASED"),
        })
    pd.DataFrame(improvement).to_csv(out_dir / "improvement_status.csv", index=False)

    conclusion = {args.a: sum_a, args.b: sum_b, "metrics": improvement, "significance": sig}
    (out_dir / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "significance.json").write_text(
        json.dumps(sig, indent=2, ensure_ascii=False), encoding="utf-8")

    for m in metrics:
        a = merged[f"{m}_{args.a}"]
        b = merged[f"{m}_{args.b}"]
        if a.dropna().empty and b.dropna().empty:
            continue
        _box_plot(
            a, b, args.a, args.b,
            title=f"{m} — {args.a} vs {args.b}",
            out=out_dir / f"comparison_{m}_box.png",
            lower_better=(m in METRICS_LOWER_BETTER),
        )

    print("\nCOMPARISON TABLE")
    print(table.round(4).to_string())
    print("\nWILCOXON")
    for m, s in sig.items():
        p = s["p_wilcoxon"]
        verdict = "SIGNIFICANT" if (p == p and p < 0.05) else "NOT SIGNIFICANT"
        print(f"  {m:11s}: p={p:.6f}  ({verdict})")
    print(f"\nOutput dir : {out_dir}")


if __name__ == "__main__":
    main()
