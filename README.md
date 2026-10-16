<p align="center">
  <h1 align="center">🇮🇩 F5-TTS Fine-Tuned for Indonesian</h1>
  <p align="center">
    Two Indonesian fine-tunes of <a href="https://github.com/SWivid/F5-TTS">F5-TTS</a> (flow-matching TTS) released as ready-to-use checkpoints, with a cloud-first inference notebook.
  </p>
</p>

<p align="center">
  <a href="https://huggingface.co/VesperGrovemane/f5_indo"><img src="https://img.shields.io/badge/🤗-HuggingFace-yellow" alt="HF"/></a>
  <a href="https://github.com/SWivid/F5-TTS"><img src="https://img.shields.io/badge/Base-F5--TTS-blue" alt="Base"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange" alt="License"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch"/></a>
</p>

<p align="center">
  <a href="#-quick-demo">🔊 Quick Demo</a> ·
  <a href="#-installation">⚙️ Installation</a> ·
  <a href="#-model-checkpoints">🤗 Checkpoints</a> ·
  <a href="#-datasets">📦 Datasets</a> ·
  <a href="#-citation">📄 Citation</a>
</p>

---

## 📖 Overview

This repository releases **two Indonesian fine-tunes of F5-TTS** for anyone who wants a strong zero-shot Indonesian TTS baseline, plus a side-by-side demo against the unmodified base model.

| Tag | Description | Training data |
|---|---|---|
| `base` | Unmodified `F5TTS_v1_Base` (zero-shot on Indonesian — the English + Chinese base has never seen Indonesian) | — |
| `f5_indo` | Full fine-tune of F5-TTS on a general-purpose Indonesian corpus | [PapaRazi/id-tts-v2](https://huggingface.co/datasets/PapaRazi/id-tts-v2) |
| `f5_indocom` | Full fine-tune on general-purpose + expressive Indonesian speech | PapaRazi/id-tts-v2 + [INESCO](https://data.mendeley.com/datasets/hzrznx3xs5/1) |

All three share the same architecture (DiT, dim 1024, 22 blocks, 16 heads, ff_mult 2, ≈ 335.8 M params) and the same inference recipe — so you can A/B/C compare them through one interface (`script/infer.ipynb`).

For the **why** behind these three conditions and the formal evaluation, see our forthcoming paper (citation TBD — once published it will be linked from this README and the `CITATION.cff`).

---

## 🔊 Quick Demo

Try it in three clicks — no install required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/script/infer.ipynb)

Or run locally:
```bash
jupyter notebook script/infer.ipynb
```

The notebook downloads the checkpoints from HuggingFace, lets you upload a reference voice, and runs all three models on the same sentence so you can A/B/C compare.

---

## ⚙️ Installation

### Option A — pip install (inference only, recommended)

```bash
# 1. Create a fresh environment
conda create -n f5tts-id python=3.10 -y
conda activate f5tts-id

# 2. Install PyTorch (match your CUDA version — see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.4:
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Install f5-tts and friends
pip install f5-tts huggingface_hub soundfile

# 4. (Optional, for Gradio demo) install gradio
pip install gradio
```

### Option B — clone this repo (training, evaluation, custom inference)

```bash
git clone https://github.com/<your-username>/F5-TTS-Indonesian.git
cd F5-TTS-Indonesian

# The src/f5_tts/ folder is a vendored copy of SWivid/F5-TTS — install in editable mode:
pip install -e .

# Install PyTorch the same way as Option A.
```

### Verify the install

```bash
python -c "from f5_tts.api import F5TTS; print('F5-TTS installed')"
```

---

## 🚀 Usage

### 1. Inference with the official `F5TTS` API

```python
from f5_tts.api import F5TTS

# Model A: zero-shot base (auto-downloads SWivid/F5-TTS)
model = F5TTS(device="cuda")  # or "cpu"

wav, sr, _ = model.infer(
    ref_file="ref_audio/audio_2.wav",
    ref_text="saya membeli tiga buah apel di pasar.",
    gen_text="Halo, perkenalkan nama saya Sarah dari Jakarta.",
)
# `wav` is a numpy array, `sr` is the sample rate (24 kHz)
```

### 2. Inference with the fine-tuned models (`f5_indo`, `f5_indocom`)

```python
import torch
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
from huggingface_hub import hf_hub_download

# ---- Load ----
vocoder = load_vocoder(is_local=False, device="cuda")

# Pick which fine-tuned model to use
TAG = "f5_indo"      # or "f5_indocom"
vocab_name = "vocab.txt" if TAG == "f5_indo" else "vocab_com.txt"

ckpt_path  = hf_hub_download(
    "VesperGrovemane/f5_indo", f"{TAG}/{TAG}.pt", local_dir="ckpts",
)
vocab_path = hf_hub_download(
    "VesperGrovemane/f5_indo", f"{TAG}/{vocab_name}", local_dir="ckpts",
)

model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, conv_layers=4, text_dim=512),
    ckpt_path,
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    use_ema=True,
    device="cuda",
)
model.eval()

# ---- Generate ----
with torch.no_grad():
    audio, sr, _ = infer_process(
        ref_audio="ref_audio/audio_2.wav",
        ref_text="saya membeli tiga buah apel di pasar.",
        gen_text="Tolong bantu aku untuk mengisi kuisioner ini yah.",
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type="vocos",
        device="cuda",
    )

# Save
import soundfile as sf
sf.write("output.wav", audio, sr)
```

### 3. Command-line inference (official f5-tts CLI)

```bash
# Base (zero-shot) — Indonesian works only as zero-shot, expect poor intelligibility
f5-tts_infer-cli \
    --model F5TTS_v1_Base \
    --ref_audio ref_audio/audio_2.wav \
    --ref_text "saya membeli tiga buah apel di pasar." \
    --gen_text "Halo, perkenalkan nama saya Sarah dari Jakarta." \
    --output_dir outputs/ \
    --output_file base.wav

# For the fine-tunes, use the Python API (snippet 2 above) — the upstream CLI
# expects the standard F5TTS_v1_Base checkpoint layout.
```

### 4. Cloud notebook (no local install)

Open `script/infer.ipynb` directly in Google Colab (free T4 GPU) or Kaggle. The first cell installs dependencies and downloads the checkpoints from HuggingFace — no manual setup.

---

## 🤗 Model Checkpoints

All checkpoints live on HuggingFace under the same repo:

| Model | File | Size | URL |
|---|---|---|---|
| **Base F5-TTS** (zero-shot) | `F5TTS_v1_Base/model_1250000.safetensors` | ~1.2 GB | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) |
| **f5_indo** | `f5_indo/f5_indo.pt` + `f5_indo/vocab.txt` | 1.35 GB + 14 KB | [🤗 VesperGrovemane/f5_indo](https://huggingface.co/VesperGrovemane/f5_indo/tree/main/f5_indo) |
| **f5_indocom** | `f5_indocom/f5_indocom.pt` + `f5_indocom/vocab_com.txt` | 5.4 GB + 14 KB | [🤗 VesperGrovemane/f5_indo](https://huggingface.co/VesperGrovemane/f5_indo/tree/main/f5_indocom) |
| **Reference voice + transcript** | `ref_audio/audio_2.wav` + `ref_audio/audio_2_transcript.txt` | 118 KB + 38 B | [🤗 ref_audio/](https://huggingface.co/VesperGrovemane/f5_indo/tree/main/ref_audio) |

> **Note:** The Vocos vocoder (`charactr/vocos-mel-24khz`, ~13.5 M params) is downloaded automatically by `f5_tts` on first use.

### Manual download (skip the auto-fetch)

```bash
mkdir -p ckpts/f5_indo ckpts/f5_indocom
wget -O ckpts/f5_indo/f5_indo.pt      https://huggingface.co/VesperGrovemane/f5_indo/resolve/main/f5_indo/f5_indo.pt
wget -O ckpts/f5_indo/vocab.txt       https://huggingface.co/VesperGrovemane/f5_indo/resolve/main/f5_indo/vocab.txt
wget -O ckpts/f5_indocom/f5_indocom.pt https://huggingface.co/VesperGrovemane/f5_indo/resolve/main/f5_indocom/f5_indocom.pt
wget -O ckpts/f5_indocom/vocab_com.txt https://huggingface.co/VesperGrovemane/f5_indo/resolve/main/f5_indocom/vocab_com.txt
```

Then point the loader at `ckpts/<tag>/...` instead of the HF URL.

---

## 🎙️ Reference voice

The default reference clip is **`ref_audio/audio_2.wav`** (2.46 s, 24 kHz, mono), with transcript `"saya membeli tiga buah apel di pasar."`. It is also available at <https://huggingface.co/VesperGrovemane/f5_indo/tree/main/ref_audio>.

You can use your own reference voice — just replace the file and don't forget the correct transcript too. 5–15 s of clean speech at any sample rate works; the loader resamples to 24 kHz mono.

---

## 📦 Datasets

| Dataset | Speakers | Clips | Duration | Style | License |
|---|---|---|---|---|---|
| **[PapaRazi/id-tts-v2](https://huggingface.co/datasets/PapaRazi/id-tts-v2)** | Multi-speaker | 84,641 | ~66.9 h | Web-sourced, ID + EN | CC BY-NC 4.0 |
| **[INESCO](https://data.mendeley.com/datasets/hzrznx3xs5/1)** | 4 (2F, 2M) | 2,400 | ~3.0 h | Acted studio: happy / angry / sad | CC BY 4.0 |
| **Combined (used for `f5_indocom`)** | — | 87,041 | ~69.9 h | INESCO ≈ 4.3 % | Inherits NC from PapaRazi |

For full training and evaluation protocol (data splits, hyperparameters, metrics), please refer to our paper (citation TBD).

---

> **`src/f5_tts/`** is a copy of the official F5-TTS source. We keep it vendored so training and evaluation are reproducible from a single clone, with no submodule surprises. To update, `git pull` the upstream and re-vendor.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: f5_tts` | `pip install f5_tts` |
| `CUDA out of memory` with `f5_indocom` | Skip that model on a free Colab T4 (needs ≥ 8 GB VRAM). Use only `f5_indo` or `base`. |
| `HF download is slow` | Pre-download with `wget` (see "Manual download" above) or set `HF_TOKEN` for higher rate limits. |
| Generated audio sounds bad / wrong language | Make sure `ref_text` accurately matches what your reference audio says, **in the same language** as `gen_text`. F5-TTS is monolingual per-utterance. |
| `torch.cuda.is_available()` is False | Reinstall PyTorch matching your CUDA version. |
| Want a public demo link | See `script/infer.ipynb` (or deploy the Gradio app to HF Spaces — instructions will land here once we publish). |

---

## 📄 Citation

A paper describing the training protocol, evaluation, and findings is forthcoming. 

In the meantime, if you want to reference the work informally, please cite the upstream F5-TTS paper and the two training datasets:

- **Base model** — Chen et al., *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching*, arXiv:2410.06885, 2024. [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Closest prior art** — Aung et al., *ThonburianTTS: Enhancing Neural Flow Matching Models for Authentic Thai Text-to-Speech*, iSAI-NLP 2025. [github.com/biodatlab/thonburian-tts](https://github.com/biodatlab/thonburian-tts)
- **Training data** — [PapaRazi/id-tts-v2](https://huggingface.co/datasets/PapaRazi/id-tts-v2) · [INESCO](https://data.mendeley.com/datasets/hzrznx3xs5/1)

---

## 📜 License

- **Code** in this repository is released under the [MIT License](LICENSE).
- **Model weights (`f5_indo.pt`, `f5_indocom.pt`) and any model-derived artefacts** are released under the [Creative Commons Attribution Non-Commercial 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0). This is the strictest of the two source-dataset licences (PapaRazi/id-tts-v2 is CC BY-NC 4.0; INESCO is CC BY 4.0).

> Commercial use of the released weights is **not permitted** under the inherited PapaRazi licence. For commercial licensing, retrain on a commercially-licensed dataset.

---

## 🙏 Acknowledgements

- The [SWivid F5-TTS team](https://github.com/SWivid/F5-TTS) for the base architecture and training infrastructure.
- [PapaRazi](https://huggingface.co/datasets/PapaRazi/id-tts-v2) and the [INESCO dataset authors](https://data.mendeley.com/datasets/hzrznx3xs5/1) for the speech data.
- The [ThonburianTTS](https://github.com/biodatlab/thonburian-tts) team for the closest published prior art in adapting F5-TTS to a low-resource language.

---

# Reproducible F5-TTS Gradio Fine-Tuning Guide

you can check this link https://github.com/SWivid/F5-TTS/discussions/143#discussion-7331486

This guide explains how to fine-tune **F5-TTS** using the built-in web-based Gradio interface (`f5-tts_finetune-gradio`). This method provides a user-friendly way to adapt the pre-trained model to custom voices, new characters, or specific languages without needing to manually write complex configuration scripts.

---

## 📋 Prerequisites

Before launching the fine-tuning interface, ensure you have set up your environment properly:

1. **Clone the Repository & Install Locally** (Required for training/fine-tuning scripts):
   ```bash
   git clone https://github.com/SWivid/F5-TTS.git
   cd F5-TTS
   pip install -e .
   ```

2. **Verify Dependencies**: Ensure PyTorch with matching CUDA/MPS support and `ffmpeg` are installed on your system.

---

## 🚀 Launching the Fine-Tuning Gradio App

F5-TTS includes a dedicated CLI command to spin up the fine-tuning dashboard:

```bash
f5-tts_finetune-gradio
```

Once executed, open the local URL provided in your terminal (typically `http://127.0.0.1:7860`) in your web browser.

---

## 📂 Step-by-Step Fine-Tuning Workflow

### Step 1: Prepare Your Dataset
* **Audio Format**: Use high-quality audio files (recommended sample rate: **24kHz** or higher to prevent high-frequency noise/artifacts). Avoid heavily compressed or low-sample-rate (e.g., 16kHz) sources if clean studio quality is desired.
* **Dataset Structure**: Prepare your dataset containing target speech audio clips alongside their corresponding text transcriptions. 
* **Auto-Slicing**: The Gradio interface provides utilities to upload and process/slice your dataset directly, though keeping manual control over clean segmentations yields better results.

### Step 2: Configure Training Parameters in the Web UI
Navigate to the training tab in the Gradio app and configure your parameters:
* **Base Model**: Choose the pre-trained checkpoint to build upon (e.g., `F5TTS_v1_Base`).
* **Dataset Path**: Point to your processed dataset directory.
* **Tokenizer**: Select your tokenizer type (`char` is standard for most language adaptations, or custom options if applicable).
* **Learning Rate (LR)**: Typically set around `1e-4` (adjust based on your dataset size and stability).
* **Batch Size / Batch Type**: Frame-based or sample-based batching depending on your GPU VRAM limits.

### Step 3: Run and Monitor Training
* Start the training process directly from the UI.
* Keep an eye on the training loss curves and generated sample checkpoints to ensure the model is converging properly without generating text-skipping or gibberish outputs.

### Step 4: Test and Validate Checkpoints
* Use the built-in testing/inference section within the Gradio layout to evaluate your newly saved checkpoints against reference audio prompts.
* *Note on Inference Best Practices*: Ensure that your reference audio + generated text length stays within safe limits (e.g., under 20 seconds) to avoid word-skipping issues.

---

## 💡 Pro-Tips & Troubleshooting

* **Artifacts or Noise?** If your output audio sounds metallic or contains high-frequency distortion, verify that your training data sample rate is set to **24kHz** rather than lower rates.
* **Word Skipping / Mismatches:** Ensure your transcriptions are clean, punctuation is properly handled, and your tokenizer settings match the configurations outlined in the main repository guidelines.
* **VRAM Limitations:** If you experience Out-Of-Memory (OOM) errors, lower your batch size configuration inside the Gradio parameters panel.
