"""WER, CER, Speaker Similarity. Auto-skip jika hasil lengkap sudah ada."""
import argparse
import json

import pandas as pd

from common import Evaluator, load_config, ws

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--system", required=True, choices=["base", "finetuned", "gt"])
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
rd = ws(cfg, "results")
rd.mkdir(parents=True, exist_ok=True)

summary_path = rd / f"{a.split}_{a.system}_summary.json"
per_path = rd / f"{a.split}_{a.system}_per_utt.csv"

if not a.force and a.n is None and summary_path.exists() and per_path.exists():
    try:
        expected = len(pd.read_csv(ws(cfg, f"{a.split}.csv")))
        actual = len(pd.read_csv(per_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if actual == expected and int(summary.get("n", -1)) == expected:
            print(f"✅ Evaluasi {a.split}/{a.system} lengkap -> SKIP")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass

ev = Evaluator(cfg)

res, summary = ev.score_dir(
    ws(cfg, f"{a.split}.csv"),
    ws(cfg, "synth", a.split, a.system),
    n=a.n,
    use_gt_audio=(a.system == "gt"),
)

res.to_csv(per_path, index=False)

summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(a.system, json.dumps(summary, indent=2, ensure_ascii=False))
ev.free()
