"""
infer.py — synthesise test-set audio for a given system.

Supported `--system` values (see `configs/config.yaml`):

  * `base`        — unmodified `F5TTS_v1_Base`, zero-shot on Indonesian.
  * `finetuned`   — the FT-General checkpoint (PapaRazi only).
  * `gt`          — passthrough: copy the ground-truth WAVs to the synth dir
                    so the evaluator and the comparison step can run on them
                    without special-casing the path layout.

For both `base` and `finetuned`, inference is run with the same recipe
(Euler ODE, NFE 32, CFG 2.0, Sway -1.0, ref_audio = audio_2.wav) so the only
intentional variable is the model weights.

CLI:

    python -m src.infer --config configs/config.yaml --split test --system base
    python -m src.infer --config configs/config.yaml --split test --system finetuned
    python -m src.infer --config configs/config.yaml --split val   --system base
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm

from common import (
    ensure_engine_links,
    f5_repo_dir,
    final_model_path,
    load_config,
    synth_dir,
    ws,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--system", required=True, choices=["base", "finetuned", "gt"])
    parser.add_argument("--ref-audio", default=None,
                        help="Override the reference audio path (defaults to config).")
    parser.add_argument("--ref-text", default=None,
                        help="Override the reference transcript (defaults to config).")
    return parser.parse_args()


def _load_base_model(cfg: dict, device: str):
    """Load the unmodified F5TTS_v1_Base through the official `F5TTS` API."""
    from f5_tts.api import F5TTS
    print(f"Loading F5TTS_v1_Base (zero-shot) on {device}...")
    return F5TTS(device=device)


def _load_finetuned_model(cfg: dict, device: str):
    """Load the FT-General checkpoint with the vendored F5-TTS engine."""
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    ckpt = final_model_path(cfg)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if ckpt.stat().st_size <= 4_000_000_000:
        raise RuntimeError(f"Checkpoint looks too small: {ckpt.stat().st_size} bytes")

    vocab = ws(cfg, cfg["f5_data_subpath"], cfg["train"]["vocab_filename"])
    if not vocab.is_file():
        raise FileNotFoundError(f"Vocab not found: {vocab}. Run build_f5_dataset.py first.")

    print(f"Loading FT-General checkpoint: {ckpt}")
    print(f"Vocab                        : {vocab}")
    vocoder = load_vocoder(is_local=False, device=device)
    model = load_model(
        DiT,
        cfg["infer"]["model_kwargs"],
        str(ckpt),
        mel_spec_type=cfg["infer"]["mel_spec_type"],
        vocab_file=str(vocab),
        use_ema=True,
        device=device,
    )
    return model, vocoder


def _synth_base(model, ref_audio: str, ref_text: str, gen_text: str) -> tuple:
    """Inference through the official API."""
    wav, sr, _ = model.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
    )
    return wav, sr


def _synth_finetuned(model, vocoder, cfg: dict,
                     ref_audio: str, ref_text: str, gen_text: str) -> tuple:
    """Inference through the lower-level `infer_process` API."""
    from f5_tts.infer.utils_infer import infer_process

    with torch.no_grad():
        audio, sr, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type=cfg["infer"]["mel_spec_type"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return audio, sr


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_engine_links(cfg)

    manifest = ws(cfg, f"{args.split}.csv")
    if not manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest}")
    df = pd.read_csv(manifest, dtype={"id": str})

    out_dir = synth_dir(cfg, args.system, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_audio = args.ref_audio or cfg["infer"]["ref_audio"]
    ref_text = args.ref_text or cfg["infer"]["ref_text"]
    if not Path(ref_audio).is_file():
        # Ref audio lives outside the workspace; allow the caller to point at it.
        alt = Path(cfg["workspace"]).parent / ref_audio
        if alt.is_file():
            ref_audio = str(alt)
        else:
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vocoder = None
    if args.system == "finetuned":
        model, vocoder = _load_finetuned_model(cfg, device)
    elif args.system == "base":
        model = _load_base_model(cfg, device)
    # gt needs no model — handled in the loop below.

    print(f"\nINFERENCE — system={args.system}  split={args.split}")
    print(f"Ref audio   : {ref_audio}")
    print(f"Ref text    : {ref_text!r}")
    print(f"Manifest    : {manifest}  ({len(df):,} samples)")
    print(f"Output      : {out_dir}")

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{args.system}/{args.split}"):
        sample_id = str(row.id)
        text = str(row.text)
        out_path = out_dir / f"{sample_id}.wav"
        if out_path.is_file() and out_path.stat().st_size > 1000:
            continue  # idempotent — skip already-done

        if args.system == "gt":
            gt_src = Path(cfg["workspace"]) / str(row.gt_audio)
            if not gt_src.is_file():
                raise FileNotFoundError(gt_src)
            shutil.copy2(gt_src, out_path)
            continue

        try:
            if args.system == "base":
                wav, sr = _synth_base(model, ref_audio, ref_text, text)
            else:
                wav, sr = _synth_finetuned(model, vocoder, cfg, ref_audio, ref_text, text)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] synthesis failed for {sample_id}: {exc}")
            continue

        # wav is a numpy array (1-D float32) at the model's output sample rate.
        sf.write(str(out_path), wav, int(sr))

    print(f"\nStatus : COMPLETED")
    print(f"Output : {out_dir}")


if __name__ == "__main__":
    main()
