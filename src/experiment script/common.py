"""Helper project: config, path persisten Drive, inference, dan evaluator."""
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import for_scoring, normalize_id  # noqa: E402


def load_config(path="configs/config.yaml"):
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = path.parent.parent
    cfg["_project_root"] = str(base)

    for key in ("workspace", "f5_repo"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((base / p).resolve()) if not p.is_absolute() else str(p.resolve())

    dc = cfg.get("data", {})
    for key in ("raw_dir", "local_dir"):
        if dc.get(key):
            p = Path(dc[key]).expanduser()
            dc[key] = str((base / p).resolve()) if not p.is_absolute() else str(p.resolve())

    go = cfg.get("final_output", {})
    if go.get("folder"):
        p = Path(go["folder"]).expanduser()
        go["folder"] = str((base / p).resolve()) if not p.is_absolute() else str(p.resolve())

    return cfg


def ws(cfg, *parts):
    return Path(cfg["workspace"]).joinpath(*parts)


def persistent_ckpt_dir(cfg):
    return ws(cfg, "checkpoints", cfg["train"]["dataset_name"])


def final_model_path(cfg):
    return persistent_ckpt_dir(cfg) / f'model_{int(cfg["train"]["target_updates"])}.pt'


def persistent_f5_data_dir(cfg):
    tc = cfg["train"]
    return ws(cfg, "f5_data", f'{tc["dataset_name"]}_{tc["tokenizer"]}')


def engine_ckpt_dir(cfg):
    return Path(cfg["f5_repo"]) / "ckpts" / cfg["train"]["dataset_name"]


def engine_data_dir(cfg):
    tc = cfg["train"]
    return Path(cfg["f5_repo"]) / "data" / f'{tc["dataset_name"]}_{tc["tokenizer"]}'


def _replace_with_symlink(link: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise RuntimeError(
                f"Tidak aman menimpa path engine yang sudah berisi data: {link}"
            )

    link.symlink_to(target, target_is_directory=True)


def ensure_engine_links(cfg):
    repo = Path(cfg["f5_repo"])
    if not (repo / "src" / "f5_tts").exists():
        raise RuntimeError(f"Engine F5-TTS belum siap: {repo}")

    _replace_with_symlink(engine_ckpt_dir(cfg), persistent_ckpt_dir(cfg))

    if persistent_f5_data_dir(cfg).exists():
        _replace_with_symlink(engine_data_dir(cfg), persistent_f5_data_dir(cfg))


def vocab_path(cfg):
    p = persistent_f5_data_dir(cfg) / "vocab.txt"
    if not p.exists():
        raise FileNotFoundError(f"Vocab belum tersedia: {p}")
    return p


def expected_test_count(cfg):
    p = ws(cfg, "test.csv")
    if not p.exists():
        return int(cfg["data"]["test_size"])
    try:
        return len(pd.read_csv(p))
    except Exception:
        return int(cfg["data"]["test_size"])


def synth_complete(cfg, system, split="test"):
    manifest = ws(cfg, f"{split}.csv")
    out = ws(cfg, "synth", split, system)
    if not manifest.exists() or not out.exists():
        return False
    try:
        df = pd.read_csv(manifest)
    except Exception:
        return False
    for uid in df["id"].astype(str):
        p = out / f"{uid}.wav"
        if not p.exists() or p.stat().st_size <= 1000:
            return False
    return True


def synthesize_manifest(cfg, manifest_csv, out_dir, ckpt_file="", n=None, tag=""):
    """Synthesis resumable. File WAV yang sudah valid otomatis dilewati."""
    import torch
    from f5_tts.api import F5TTS

    ensure_engine_links(cfg)
    ic = cfg["infer"]

    df = pd.read_csv(manifest_csv)
    if n:
        df = df.iloc[:n]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(model=cfg["train"]["exp_name"], use_ema=True)
    if ckpt_file:
        kwargs.update(
            ckpt_file=str(ckpt_file),
            vocab_file=str(vocab_path(cfg)),
        )

    tts = F5TTS(**kwargs)
    ok = 0
    failed = []

    for _, r in df.iterrows():
        wav_out = out_dir / f"{r['id']}.wav"

        if wav_out.exists() and wav_out.stat().st_size > 1000:
            ok += 1
            continue

        try:
            tts.infer(
                ref_file=str(r["ref_audio"]),
                ref_text=normalize_id(r["ref_text"]),
                gen_text=normalize_id(r["text"]),
                file_wave=str(wav_out),
                nfe_step=ic["nfe_step"],
                cfg_strength=ic["cfg_strength"],
                sway_sampling_coef=ic["sway_sampling_coef"],
                speed=ic["speed"],
                seed=ic["seed"],
                remove_silence=False,
            )
            ok += 1
        except Exception as exc:
            failed.append({"id": str(r["id"]), "error": str(exc)})
            print(f"[WARN] {tag} gagal pada {r['id']}: {exc}")

    report = {
        "tag": tag,
        "requested": int(len(df)),
        "completed": int(ok),
        "failed": int(len(failed)),
        "output_dir": str(out_dir),
    }
    (out_dir / "_inference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if failed:
        (out_dir / "_failed.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Evaluator:
    """Whisper ASR untuk WER/CER + WavLM untuk Speaker Similarity."""

    def __init__(self, cfg):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg["eval"]
        self._asr = None
        self._sv = None
        self._fe = None

    def _load_audio(self, path, sr=16000):
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y

    def transcribe(self, path):
        if self._asr is None:
            from transformers import pipeline

            self._asr = pipeline(
                "automatic-speech-recognition",
                model=os.environ.get("F5_WHISPER_MODEL_PATH", self.cfg["asr_model"]),
                torch_dtype=(
                    self.torch.float16
                    if self.dev == "cuda"
                    else self.torch.float32
                ),
                device=self.dev,
            )

        y = self._load_audio(path)

        out = self._asr(
            {
                "raw": y,
                "sampling_rate": 16000,
            },
            generate_kwargs={
                "language": "indonesian",
                "task": "transcribe",
                "max_new_tokens": 96,
                "condition_on_prev_tokens": False,
            },
        )
        return out["text"]

    def embed(self, path):
        if self._sv is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self._fe = AutoFeatureExtractor.from_pretrained(self.cfg["sv_model"])
            self._sv = (
                WavLMForXVector
                .from_pretrained(self.cfg["sv_model"])
                .to(self.dev)
                .eval()
            )

        y = self._load_audio(path)
        inp = self._fe(y, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(self.dev) for k, v in inp.items()}

        with self.torch.no_grad():
            emb = self._sv(**inp).embeddings

        return self.torch.nn.functional.normalize(emb, dim=-1)[0].cpu()

    def sim(self, a, b):
        return float(self.torch.dot(self.embed(a), self.embed(b)))

    def score_dir(self, manifest_csv, wav_dir, n=None, use_gt_audio=False):
        import jiwer

        df = pd.read_csv(manifest_csv)
        if n:
            df = df.iloc[:n]

        rows = []

        for _, r in df.iterrows():
            wav = (
                str(r["gt_audio"])
                if use_gt_audio
                else str(Path(wav_dir) / f"{r['id']}.wav")
            )

            if not os.path.exists(wav):
                continue

            ref = for_scoring(r["text"])
            hyp = for_scoring(self.transcribe(wav))

            if not ref:
                continue

            rows.append({
                "id": r["id"],
                "ref": ref,
                "hyp": hyp,
                "wer": jiwer.wer(ref, hyp) if hyp else 1.0,
                "cer": jiwer.cer(ref, hyp) if hyp else 1.0,
                "sim_prompt": self.sim(wav, r["ref_audio"]),
                "sim_gt": self.sim(wav, r["gt_audio"]),
            })

        res = pd.DataFrame(rows)

        if res.empty:
            return res, {}

        refs = res["ref"].tolist()
        hyps = res["hyp"].fillna("").tolist()

        summary = {
            "n": len(res),
            "WER": float(jiwer.wer(refs, hyps)),
            "CER": float(jiwer.cer(refs, hyps)),
            "WER_mean": float(res.wer.mean()),
            "CER_mean": float(res.cer.mean()),
            "SIM_prompt": float(res.sim_prompt.mean()),
            "SIM_prompt_std": float(res.sim_prompt.std()),
            "SIM_gt": float(res.sim_gt.mean()),
        }
        return res, summary

    def free(self):
        self._asr = None
        self._sv = None
        self._fe = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
