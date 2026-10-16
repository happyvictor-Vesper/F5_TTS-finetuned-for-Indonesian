"""
common.py — shared utilities for the Indonesian TTS fine-tuning project.

Everything in `src/` is a thin wrapper around the vendored F5-TTS engine at
`./F5-TTS/src/f5_tts/`. This module owns:

  * YAML config loading
  * Workspace / F5-internal / final-model path helpers
  * `ensure_engine_links` — symlinks that let F5-TTS find our prepared data
    and write checkpoints back into our workspace
  * The `Evaluator` class — Whisper ASR + Resemblyzer SIM + SQUIM
  * Text normalisation shared by the training pipeline and the evaluator
  * "synthesis complete?" predicates that the notebook uses to skip work

The notebook imports this module from `src/`, so `sys.path` must contain the
project root when the notebook runs:

    if str(PROJECT_DIR / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR / "src"))
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | os.PathLike) -> dict:
    """Load the YAML config and resolve a few derived fields in place."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is not a mapping: {cfg_path}")

    # Touch the GPU-dependent fields so downstream code can rely on the keys
    # even if the notebook hasn't run Section 3 yet.
    cfg.setdefault("train", {}).setdefault("batch_size_per_gpu", 1000)
    cfg.setdefault("train", {}).setdefault("grad_accumulation_steps", 2)
    cfg.setdefault("train", {}).setdefault("mixed_precision", "fp16")
    cfg.setdefault("train", {}).setdefault("bnb_optimizer", False)
    return cfg


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ws(cfg: dict, *parts: str) -> Path:
    """Join a path under the project workspace."""
    return Path(cfg["workspace"]) / Path(*parts) if parts else Path(cfg["workspace"])


def raw_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["raw_dir"])


def extracted_dir(cfg: dict) -> Path:
    return raw_dir(cfg) / "extracted"


def wavs_dir(cfg: dict) -> Path:
    return ws(cfg, "wavs")


def persistent_f5_data_dir(cfg: dict) -> Path:
    """Where our prepared F5-TTS internal dataset (raw.arrow, duration.json, vocab.txt) lives."""
    return ws(cfg, cfg["f5_data_subpath"])


def f5_data_dir_for_tokenizer(cfg: dict) -> Path:
    """The folder under F5-TTS/data/ that the trainer reads from.

    F5-TTS's `load_dataset` looks at `<f5_tts>/../../data/<dataset>_<tokenizer>/`,
    so for our default (`pinyin`) we need `F5-TTS/data/id_tts_full_pinyin/`.
    `ensure_engine_links` symlinks this folder to `persistent_f5_data_dir`.
    """
    name = f"{cfg['train']['dataset_name']}_{cfg['train']['tokenizer']}"
    return f5_repo_dir(cfg) / "data" / name


def f5_ckpt_dir(cfg: dict) -> Path:
    """Where F5-TTS will write the fine-tune checkpoints."""
    return f5_repo_dir(cfg) / "ckpts" / cfg["train"]["dataset_name"]


def final_model_path(cfg: dict) -> Path:
    """Where the final `model_<target_updates>.pt` lives.

    We symlink F5-TTS/ckpts/<dataset_name>/ into workspace_full/checkpoints/
    so the trainer writes here directly and the notebook path
    `workspace_full/checkpoints/id_tts_full/model_45000.pt` works without copy.
    """
    return ws(cfg, cfg["train"]["final_model_subpath"])


def synth_dir(cfg: dict, system: str, split: str) -> Path:
    """Where synthesised WAVs for a (system, split) pair live."""
    return ws(cfg, "synth", split, system)


def results_dir(cfg: dict) -> Path:
    return ws(cfg, "results")


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

def f5_repo_dir(cfg: dict) -> Path:
    """The vendored F5-TTS checkout we import as a library."""
    return Path(cfg["f5_repo"])


def ensure_engine_links(cfg: dict) -> None:
    """
    Make sure the vendored F5-TTS source is importable AND that the F5-TTS
    internal data / checkpoint layout points at our workspace.

    F5-TTS (1.1.22) is hard-coded to look at:
      * `<install>/f5_tts/../../data/<dataset>_<tokenizer>/{raw.arrow, duration.json, vocab.txt}`
      * `<install>/f5_tts/../../ckpts/<dataset>/`
    which resolves to `F5-TTS/data/...` and `F5-TTS/ckpts/...` under the
    vendored checkout. To keep training artefacts in our workspace, we
    symlink those two folders into `workspace_full/`.

    The notebook installs the engine with `pip install -e F5-TTS` (Section 2);
    this helper additionally wires up the data + ckpt symlinks.
    """
    repo = f5_repo_dir(cfg)
    src_pkg = repo / "src" / "f5_tts"
    if not src_pkg.is_dir():
        raise FileNotFoundError(
            f"Vendored F5-TTS source not found at {src_pkg}. "
            "Run `git clone --depth 1 --branch 1.1.22 "
            "https://github.com/SWivid/F5-TTS.git F5-TTS` first."
        )

    parent = str(repo / "src")
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # 1. Data: F5-TTS/data/<dataset>_<tokenizer>/ -> workspace_full/f5_data/...
    src_data = persistent_f5_data_dir(cfg)
    src_data.mkdir(parents=True, exist_ok=True)
    dst_data = f5_data_dir_for_tokenizer(cfg)
    dst_data.parent.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(src_data, dst_data)

    # 2. Checkpoints: F5-TTS/ckpts/<dataset>/ -> workspace_full/checkpoints/...
    src_ckpts = ws(cfg, "checkpoints")
    src_ckpts.mkdir(parents=True, exist_ok=True)
    dst_ckpts = f5_ckpt_dir(cfg)
    dst_ckpts.parent.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(src_ckpts, dst_ckpts)


def _ensure_symlink(src: Path, dst: Path) -> None:
    """Create a symlink, falling back to copy on systems that disallow it."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Text normalisation (shared between training and evaluation)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[\u2000-\u206f\u2e00-\u2e7f'!\"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + lowercase + strip punctuation + collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# Alias used by the notebook's evaluator
for_scoring = normalize_text


# ---------------------------------------------------------------------------
# "Is this artifact already complete?" predicates
# ---------------------------------------------------------------------------

def synth_complete(cfg: dict, system: str, split: str = "test",
                   expected: int | None = None) -> bool:
    """True if every (sample_id, .wav) is present in synth_dir(cfg, system, split)."""
    manifest = ws(cfg, f"{split}.csv")
    if not manifest.is_file():
        return False

    out = synth_dir(cfg, system, split)
    if not out.is_dir():
        return False

    if expected is None:
        import pandas as pd
        expected = len(pd.read_csv(manifest))

    for sample_id in _read_id_column(manifest):
        wav = out / f"{sample_id}.wav"
        if not wav.is_file() or wav.stat().st_size < 1000:
            return False
    return True


def _read_id_column(csv_path: Path) -> Iterable[str]:
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"id": str})
    return df["id"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Evaluator — Whisper ASR + Resemblyzer SIM + SQUIM quality
# ---------------------------------------------------------------------------

@dataclass
class Evaluator:
    """Wraps Whisper (ASR), Resemblyzer (SIM) and torchaudio SQUIM (quality)."""

    cfg: dict
    asr_model_name: str = "large-v3"
    asr_language: str = "id"
    asr_fp16: bool = True

    def __post_init__(self) -> None:
        self._asr = None  # lazy — costs ~3 GB VRAM
        self._sim = None  # lazy — also a model
        self._squim = None
        self._device = "cuda" if _cuda_available() else "cpu"

    # ---- ASR --------------------------------------------------------------

    def transcribe(self, audio_path: str | os.PathLike) -> str:
        """Transcribe a single audio file with Whisper, language forced to id."""
        if self._asr is None:
            import whisper
            self._asr = whisper.load_model(self.asr_model_name, device=self._device)
        result = self._asr.transcribe(
            str(audio_path),
            language=self.asr_language,
            fp16=self.asr_fp16 and self._device == "cuda",
        )
        return (result.get("text") or "").strip()

    # ---- Speaker similarity ----------------------------------------------

    def speaker_similarity(self, ref_audio: str | os.PathLike,
                           gen_audio: str | os.PathLike) -> float:
        """Cosine similarity between 256-d Resemblyzer d-vectors.

        `resemblyzer.preprocess_wav` accepts either a file path (which it
        resamples to 16 kHz internally) or a numpy array (which it assumes is
        already at 16 kHz). We pass file paths so the resampling happens once
        in the canonical place.
        """
        if self._sim is None:
            from resemblyzer import VoiceEncoder
            self._sim = VoiceEncoder(device=self._device)
        import numpy as np
        from resemblyzer import preprocess_wav

        ref = preprocess_wav(str(ref_audio))
        gen = preprocess_wav(str(gen_audio))

        ref_emb = self._sim.embed_utterance(ref)
        gen_emb = self._sim.embed_utterance(gen)
        return float(np.dot(ref_emb, gen_emb) /
                     (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb)))

    # ---- SQUIM_SUBJECTIVE -------------------------------------------------

    def quality_score(self, gen_audio: str | os.PathLike,
                      ref_audio: str | os.PathLike) -> float:
        """NORESQA-style non-intrusive MOS estimate from torchaudio SQUIM."""
        if self._squim is None:
            from torchaudio.pipelines import SQUIM_SUBJECTIVE as bundle  # type: ignore
            self._squim = bundle.get_model()
        import torch
        import torchaudio

        gen_wav, gen_sr = torchaudio.load(str(gen_audio))
        if gen_sr != 16000:
            gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, 16000)
        ref_wav, ref_sr = torchaudio.load(str(ref_audio))
        if ref_sr != 16000:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 16000)

        gen_wav = gen_wav.mean(0)  # mono
        ref_wav = ref_wav.mean(0)  # mono
        with torch.no_grad():
            score = self._squim(gen_wav.unsqueeze(0), ref_wav.unsqueeze(0))
        return float(score.item() if hasattr(score, "item") else score)

    # ---- Lifecycle --------------------------------------------------------

    def free(self) -> None:
        """Drop all the lazy models so the GPU memory is released."""
        self._asr = None
        self._sim = None
        self._squim = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_audio_24k(path: str | os.PathLike):
    """Load an audio file as a 1-D float32 numpy array at 24 kHz mono."""
    import numpy as np
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Tiny manifest helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
