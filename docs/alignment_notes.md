# Alignment Notes — Indonesian TTS Fine-Tuning Project

This document records how the `src/` wrappers, the `F5-TTS/` vendored engine,
the `configs/config.yaml`. Read this if you are wondering *"why is this file the way it is?"*


## 1. The big picture

```
+----------------------+        +-----------------------+
|                      |        |                       |
|  Indonesian_TTS_     |  uses  |  F5-TTS/              |
|  FineTuning.ipynb    +------->+  src/f5_tts/...       |
|  (the orchestrator)  |        |  (vendored @ 1.1.22)  |
|                      |        |                       |
+----------+-----------+        +----------+------------+
           |                               ^
           |                               |
           v                               |
+----------+-------------------------------+----------+
|                                                     |
|  src/  (our thin wrappers)                          |
|    common.py            paths, Evaluator, text norm |
|    get_dataset.py       PapaRazi shard download      |
|    prepare_data.py      preprocess + split           |
|    build_f5_dataset.py  pack raw.arrow + vocab       |
|    train.py             launch finetune (3 runners)  |
|    finetune_settings.py materialise setting.json     |
|    launch_gradio.py     boot the Gradio UI           |
|    infer.py             synthesise for a system      |
|    evaluate.py          Whisper ASR + SIM + SQUIM    |
|    compare.py           Wilcoxon + box plots         |
|    preview_audio.py     audio_preview.html           |
|    finalize.py          OUTPUT_MANIFEST.json         |
|                                                     |
+-----------------------------------------------------+
           |
           v
+-----------------------------------------------------+
|  configs/config.yaml                                |
|    workspace, dataset, train, infer, eval, systems  |
+-----------------------------------------------------+
```

note: to access the INESCO dataset you need to download the file directly from their website.

Three interchangeable ways to launch the same fine-tune (all reading the
same `config.yaml`):

```
+--------------------+      +---------------------+      +--------------------+
| src/launch_gradio  |      | src/train --runner  |      | src/train --runner |
| (Gradio UI)        |      |   gradio            |      |   cli              |
+--------------------+      | (write setting.json |      | (upstream          |
                            |  + stage pretrained, |      |  finetune_cli.py   |
                            |  then open the UI)   |      |  directly)         |
                            +---------------------+      +--------------------+
```

## 2. The notebook's call surface into `src/`

The notebook imports a fixed set of names from `src/common.py`:

| Symbol | Where it is used | What it does |
|---|---|---|
| `load_config` | Sections 3, 4, 6, 7, 8, 9, 10, 11, 14 | YAML loader with sane defaults |
| `ws` | every section | `Path(cfg["workspace"]) / parts` |
| `extracted_dir` | Section 5 | path under `raw/paparazi_id_tts_v2/extracted/` |
| `persistent_f5_data_dir` | Section 7 | path under `workspace_full/f5_data/...` |
| `final_model_path` | Sections 4, 9, 10, 14 | path to `model_45000.pt` |
| `synth_complete` | Sections 8, 10, 14 | "all WAVs in place?" check |
| `ensure_engine_links` | Section 7 | wires F5-TTS/data + ckpts symlinks |
| `Evaluator` | Sections 8, 10 | ASR + SIM + quality scorer |
| `f5_ckpt_dir` | src/train, src/finetune_settings, src/launch_gradio | `F5-TTS/ckpts/<dataset_name>/` (target of the symlink) |
| `f5_repo_dir` | everywhere that touches the engine | the vendored F5-TTS checkout |

The other modules (`get_dataset.py`, `prepare_data.py`, `build_f5_dataset.py`,
`train.py`, `infer.py`, `evaluate.py`, `compare.py`, `preview_audio.py`,
`finalize.py`, plus `finetune_settings.py` and `launch_gradio.py`) are invoked
as subprocesses by the notebook:

```
python -u src/build_f5_dataset.py  --config configs/config.yaml
python -u src/infer.py             --split test --system base       --config configs/config.yaml
python -u src/infer.py             --split test --system finetuned  --config configs/config.yaml
python -u src/evaluate.py          --split test --system base       --config configs/config.yaml
python -u src/evaluate.py          --split test --system finetuned  --config configs/config.yaml
python -u src/train.py             --config configs/config.yaml        # full run
python -u src/train.py             --dry --config configs/config.yaml  # smoke test
python -u src/compare.py           --split test --config configs/config.yaml
python -u src/preview_audio.py     --n 5 --config configs/config.yaml
python -u src/finalize.py          --config configs/config.yaml
```

And the two extra entry points that bypass the notebook (see
`docs/gradio_and_cli.md` for the long form):

```
# 1. Pre-fill the Gradio UI's Indonesian project, then open it
python -m src.train --config configs/config.yaml --runner gradio
python -m src.launch_gradio --config configs/config.yaml

# 2. Run the upstream F5-TTS finetune_cli.py directly (no python -m)
python -m src.train --config configs/config.yaml --runner cli
```

This split is intentional: the notebook is the orchestrator, the `src/` modules
do their own argument parsing and can be run from a terminal.

## 3. The F5-TTS surface the wrappers touch

The vendored engine at `F5-TTS/src/f5_tts/` exposes (in 1.1.22):

* `f5_tts.api.F5TTS` — high-level API, used by `src/infer.py --system base`.
* `f5_tts.infer.utils_infer.{load_model, load_vocoder, infer_process}` — used
  by `src/infer.py --system finetuned` to load our `model_45000.pt` against
  the extended `vocab.txt`.
* `f5_tts.train.datasets.prepare_csv_wavs` — invoked by
  `src/build_f5_dataset.py` to tokenise the train CSV into `raw.arrow` and
  `duration.json`.
* `f5_tts.train.finetune_cli` — invoked by `src/train.py` for the actual
  fine-tune. It uses the standard F5-TTS argparse interface (no custom
  arg-file).
* `f5_tts.model.{DiT, CFM, Trainer}` — indirectly through the two above.

We do **not** modify any file under `F5-TTS/`. The engine is pinned to tag
`1.1.22` and `git -C F5-TTS describe --tags --exact-match` should print
`1.1.22`. To upgrade, re-clone at a new tag and re-run the pipeline.

## 4. Symlinks: how the trainer finds our data

F5-TTS (1.1.22) hard-codes the data location to
`<f5_tts install>/../../data/<dataset>_<tokenizer>/`. With our editable
install that's `F5-TTS/data/<dataset>_<tokenizer>/`.

`src/common.py:ensure_engine_links` creates two symlinks so the trainer and
the data builder both look at the workspace:

```
F5-TTS/data/<dataset>_<tokenizer>  →  workspace_full/f5_data/<dataset>_<tokenizer>
F5-TTS/ckpts/<dataset>             →  workspace_full/checkpoints
```

`src/build_f5_dataset.py` calls `ensure_engine_links` before invoking
`prepare_csv_wavs`, and `src/train.py` does the same before invoking
`finetune_cli`. The symlinks are idempotent — re-running is a no-op.

## 5. The three "systems" the notebook compares

| `system` arg | What the wrapper does | Where the WAVs go |
|---|---|---|
| `gt` | passthrough: copy the PapaRazi ground-truth WAVs to the synth dir | `workspace_full/synth/test/gt/` |
| `base` | load `F5TTS_v1_Base` via the `F5TTS` API and synthesise at NFE 32 / CFG 2.0 / Sway -1.0 with `ref_audio/audio_2.wav` as prompt | `workspace_full/synth/test/base/` |
| `finetuned` | load `model_45000.pt` (FT-General on PapaRazi) with the extended vocab and run the same inference recipe | `workspace_full/synth/test/finetuned/` |

The README additionally documents the **FT-Combined** (`f5_indocom`) checkpoint
from the paper, which is the same `src/train.py` entry point with a different
`--metadata-csv` and `--dataset-name`. The notebook does not produce
`f5_indocom` itself — that is the paper's headline result and lives on
HuggingFace.

## 6. Config keys the wrappers read

The full schema is in `configs/config.yaml`. The keys each module reads:

| Module | Reads |
|---|---|
| `common.py` | `workspace`, `f5_repo`, `data.raw_dir`, `f5_data_subpath`, `train.dataset_name`, `train.vocab_filename`, `train.final_model_subpath`, `train.tokenizer`, `infer.*`, `eval.*` |
| `get_dataset.py` | `data.repo`, `data.parts`, `data.raw_dir` |
| `prepare_data.py` | `workspace`, `data.{raw_dir, sample_rate, min_duration, max_duration, min_words, val_size, test_size}`, `train.seed` |
| `build_f5_dataset.py` | `workspace`, `f5_data_subpath`, `train.{dataset_name, vocab_filename, tokenizer}`, `data.sample_rate` |
| `train.py` | `f5_repo`, `train.*`, `workspace`, `data.sample_rate` |
| `infer.py` | `workspace`, `infer.*`, `train.{dataset_name, vocab_filename}`, `f5_data_subpath`, `train.final_model_subpath` |
| `evaluate.py` | `workspace`, `eval.*`, `infer.ref_audio` |
| `compare.py` | `workspace` |
| `preview_audio.py` | `workspace`, `infer.ref_audio` |
| `finalize.py` | `workspace`, `train.final_model_subpath` |

## 7. What is *not* in the project on purpose

* **The PapaRazi dataset itself** — it is downloaded on demand by
  `src/get_dataset.py` from `PapaRazi/id-tts-v2` on HuggingFace.
* **The INESCO dataset** — used only by the paper's FT-Combined condition.
  The `src/build_f5_dataset.py` + `src/train.py` flow can rebuild it on
  demand, but the dataset itself is not vendored here.
* **The model checkpoints** — `f5_indo` and `f5_indocom` are released on
  HuggingFace at `VesperGrovemane/f5_indo`. The notebook downloads them
  on first use.
* **The pre-trained `F5TTS_v1_Base`** — auto-downloaded by F5-TTS on first
  inference via `cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/...")`.

## 8. Pinning

| Component | Pinned to |
|---|---|
| F5-TTS engine | tag `1.1.22` (commit `9c614e96`) — see `F5-TTS/.git/HEAD` |
| PapaRazi/id-tts-v2 | shard 1–10 (66.9 h post-filter) |
| Vocos vocoder | `charactr/vocos-mel-24khz` (auto-downloaded by F5-TTS) |
| Whisper ASR | `large-v3` (openai-whisper package) |
| Speaker similarity | Resemblyzer GE2E (256-d d-vector) |
| Objective quality | torchaudio `SQUIM_SUBJECTIVE` |
