"""Perbandingan Base vs Fine-tuned, grafik, kesimpulan, Wilcoxon. Auto-skip."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from common import final_model_path, load_config, ws

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
rd = ws(cfg, "results")
rd.mkdir(parents=True, exist_ok=True)

outputs = [
    rd / "comparison_table.csv",
    rd / "comparison_table.md",
    rd / "comparison_wer.png",
    rd / "comparison_cer.png",
    rd / "comparison_sim_prompt.png",
    rd / "improvement_status.csv",
    rd / "conclusion.json",
    rd / "significance.json",
]

if not a.force and all(p.exists() and p.stat().st_size > 0 for p in outputs):
    print("✅ Comparison lengkap -> SKIP")
    raise SystemExit(0)

def load_summary(name):
    p = rd / f"{a.split}_{name}_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))

gt = load_summary("gt")
base = load_summary("base")
ft = load_summary("finetuned")

base_per = pd.read_csv(rd / f"{a.split}_base_per_utt.csv").set_index("id")
ft_per = pd.read_csv(rd / f"{a.split}_finetuned_per_utt.csv").set_index("id")
ids = base_per.index.intersection(ft_per.index)

comparison = pd.DataFrame(
    {
        "Ground Truth": [gt["WER"], gt["CER"], gt["SIM_prompt"], gt["SIM_gt"]],
        "Base F5-TTS": [base["WER"], base["CER"], base["SIM_prompt"], base["SIM_gt"]],
        "Fine-tuned 45k": [ft["WER"], ft["CER"], ft["SIM_prompt"], ft["SIM_gt"]],
    },
    index=["WER", "CER", "SIM_prompt", "SIM_gt"],
).T

comparison.to_csv(rd / "comparison_table.csv")
(rd / "comparison_table.md").write_text(
    comparison.round(4).to_markdown(),
    encoding="utf-8",
)

wer_reduction = (base["WER"] - ft["WER"]) / base["WER"] * 100
cer_reduction = (base["CER"] - ft["CER"]) / base["CER"] * 100
sim_change = ft["SIM_prompt"] - base["SIM_prompt"]

status = pd.DataFrame([
    {
        "Metric": "WER",
        "Base": base["WER"],
        "Fine-tuned": ft["WER"],
        "Perubahan": f"{wer_reduction:.2f}% lebih rendah",
        "Status": "Meningkat" if ft["WER"] < base["WER"] else "Menurun",
    },
    {
        "Metric": "CER",
        "Base": base["CER"],
        "Fine-tuned": ft["CER"],
        "Perubahan": f"{cer_reduction:.2f}% lebih rendah",
        "Status": "Meningkat" if ft["CER"] < base["CER"] else "Menurun",
    },
    {
        "Metric": "SIM_prompt",
        "Base": base["SIM_prompt"],
        "Fine-tuned": ft["SIM_prompt"],
        "Perubahan": f"{sim_change:+.4f}",
        "Status": "Meningkat" if ft["SIM_prompt"] >= base["SIM_prompt"] else "Menurun",
    },
])
status.to_csv(rd / "improvement_status.csv", index=False)

for metric, title in [
    ("WER", "WER (Semakin Rendah Semakin Baik)"),
    ("CER", "CER (Semakin Rendah Semakin Baik)"),
    ("SIM_prompt", "Speaker Similarity to Prompt (Semakin Tinggi Semakin Baik)"),
]:
    values = comparison[metric].tolist()
    labels = comparison.index.tolist()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(metric)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(rd / f"comparison_{metric.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()

significance = {}

for metric in ["wer", "cer", "sim_prompt"]:
    b = base_per.loc[ids, metric]
    f = ft_per.loc[ids, metric]

    try:
        p_value = float(wilcoxon(b, f).pvalue)
    except ValueError:
        p_value = None

    significance[metric] = {
        "base_mean": float(b.mean()),
        "finetuned_mean": float(f.mean()),
        "p_wilcoxon": p_value,
    }

conclusion = {
    "final_model": str(final_model_path(cfg)),
    "base": {
        "WER": base["WER"],
        "CER": base["CER"],
        "SIM_prompt": base["SIM_prompt"],
        "SIM_gt": base["SIM_gt"],
    },
    "finetuned": {
        "WER": ft["WER"],
        "CER": ft["CER"],
        "SIM_prompt": ft["SIM_prompt"],
        "SIM_gt": ft["SIM_gt"],
    },
    "improvement": {
        "WER_reduction_percent": wer_reduction,
        "CER_reduction_percent": cer_reduction,
        "SIM_prompt_change": sim_change,
    },
    "target_status": {
        "WER_improved": ft["WER"] < base["WER"],
        "CER_improved": ft["CER"] < base["CER"],
        "SIM_prompt_stable_or_improved": ft["SIM_prompt"] >= base["SIM_prompt"],
    },
}

conclusion["all_targets_met"] = all(conclusion["target_status"].values())

(rd / "conclusion.json").write_text(
    json.dumps(conclusion, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
(rd / "significance.json").write_text(
    json.dumps(significance, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(comparison.round(4).to_string())
print(json.dumps(conclusion, indent=2, ensure_ascii=False))
print(json.dumps(significance, indent=2, ensure_ascii=False))
