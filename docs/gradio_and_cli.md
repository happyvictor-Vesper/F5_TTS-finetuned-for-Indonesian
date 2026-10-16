# Gradio UI & CLI — fine-tuning entry points

The Indonesian project has **three interchangeable ways** to launch the same
fine-tune, all reading `configs/config.yaml` as the single source of truth.

| Entry point | Best for | What it does |
|---|---|---|
| `python -m src.launch_gradio` | interactive tinkering | Boots the vendored F5-TTS Gradio UI with the Indonesian project pre-selected and our `setting.json` pre-filled. |
| `python -m src.train --runner gradio` | preparing the Gradio project | Materialises `setting.json` + stages the pretrained checkpoint. Then run `src.launch_gradio` to open the UI. |
| `python -m src.train --runner cli`    | batch / server training | Invokes the **upstream** `finetune_cli.py` directly with our hyperparameters translated to the F5-TTS argparse surface. |
| `python -m src.train --runner module` | what the notebook does | Invokes `python -m f5_tts.train.finetune_cli` (the path the Colab notebook uses). |

All four paths go through the same `ensure_engine_links` step, which symlinks:

```
F5-TTS/data/<dataset>_<tokenizer>   →  workspace_full/f5_data/<dataset>_<tokenizer>
F5-TTS/ckpts/<dataset>              →  workspace_full/checkpoints
```

so the trainer writes `model_45000.pt` straight into `workspace_full/checkpoints/id_tts_full/`
and the Gradio UI lists the Indonesian project alongside any other project
you've trained.

---

## 1. Gradio UI (interactive)

```bash
# Pre-fill the Indonesian project + stage the pretrained checkpoint.
python -m src.train --config configs/config.yaml --runner gradio

# Open the Gradio UI.
python -m src.launch_gradio --config configs/config.yaml
# → http://localhost:7860  (use --share for a public URL)
```

What you see inside the UI:

* **Project Name** dropdown lists `id_tts_full` (FT-General). Switch to it.
* **Dataset & Vocab** tab: the "Check Vocab / Extend Vocab" button reads
  `workspace_full/f5_data/id_tts_full_pinyin/vocab.txt` (extended for
  Indonesian) and shows how many characters were added on top of the base
  Emilia ZH/EN vocab.
* **Basic Settings / Training** tab: pre-filled with our hyperparameters
  (45K updates, T4 1000-frame / A100 1700-frame batch, fp16/bf16, AdamW,
  `F5TTS_v1_Base`, pinyin tokenizer, EMA on, last-per-updates = 5,000).
  Override anything you want; press "Save Settings" to persist.
* **Start training** runs `accelerate launch finetune_cli.py` against the
  staged `pretrained_model_1250000.safetensors`. The CLI's stdout streams
  into the UI's log panel.

### Reopening with a different project (e.g. FT-Combined)

```bash
python -m src.train --config configs/config.yaml --runner gradio \
                    --dataset-name id_tts_full_com \
                    --metadata-csv workspace_full/combined_train.csv
# First build combined_train.csv with src/prepare_data.py + INESCO.
```

---

## 2. Upstream CLI (batch / non-interactive)

```bash
# Smoke test (1 update).
python -m src.train --config configs/config.yaml --runner cli --dry

# Full run.
python -m src.train --config configs/config.yaml --runner cli

# FT-Combined: same CLI, different metadata + project name.
python -m src.train --config configs/config.yaml --runner cli \
                    --dataset-name id_tts_full_com \
                    --metadata-csv workspace_full/combined_train.csv
```

`--runner cli` calls `python <f5-tts>/src/f5_tts/train/finetune_cli.py` with
the upstream argparse surface — no `-m`, no custom arg-file, just the
flags F5-TTS actually accepts:

```
--exp_name F5TTS_v1_Base
--dataset_name id_tts_full
--learning_rate 1.0e-05
--batch_size_per_gpu 1000           # 1700 on A100/H100/L4
--batch_size_type frame
--max_samples 64
--grad_accumulation_steps 2        # 1 on A100/H100/L4
--max_grad_norm 1.0
--epochs 100
--num_warmup_updates 100
--save_per_updates 7500
--keep_last_n_checkpoints 1
--last_per_updates 5000
--tokenizer pinyin
--bnb_optimizer                    # only on T4 if bitsandbytes is installed
--pretrain <F5-TTS>/ckpts/id_tts_full/pretrained_model_1250000.safetensors
```

The pretrained checkpoint is staged into the project directory by
`_stage_pretrained` so the F5-TTS CLI's `shutil.copy2(ckpt_path, file_checkpoint)`
step is a no-op.

---

## 3. `python -m f5_tts.train.finetune_cli` (what the notebook uses)

```bash
python -m src.train --config configs/config.yaml --runner module
```

Identical hyperparameters to `--runner cli`; just invoked through Python's
module system so the engine's `from f5_tts.model import ...` imports resolve
the same way they do inside the Colab notebook. This is the path the
notebook's Section 9.3 already calls.

---

## 4. Where the artefacts land

```
F5-TTS/ckpts/id_tts_full/                ← symlink target (the trainer writes here)
  ├── pretrained_model_1250000.safetensors  ← copied by _stage_pretrained
  ├── setting.json                       ← written by --runner gradio
  ├── model_45000.pt                     ← the final fine-tune
  └── model_last.pt                      ← snapshot every 5K updates (last_per_updates)

workspace_full/checkpoints/id_tts_full/  ← same files, visible to the notebook
```

Both paths are the same physical files — `workspace_full/checkpoints/` is the
real directory, `F5-TTS/ckpts/id_tts_full/` is the symlink the trainer sees.

---

## 5. Quick troubleshooting

| Symptom | Fix |
|---|---|
| Gradio UI doesn't list `id_tts_full` | Run `python -m src.train --config configs/config.yaml --runner gradio` first. The symlinks must exist. |
| "Internal F5 dataset not built yet" | Run `python -m src.build_f5_dataset --config configs/config.yaml` first. |
| Gradio UI shows "No data" under the project | Same as above — the symlink `F5-TTS/data/id_tts_full_pinyin/` must point at `workspace_full/f5_data/id_tts_full_pinyin/`. |
| `accelerate: command not found` | `pip install accelerate` (or use the Colab default which already has it). |
| `bitsandbytes` import fails on T4 | `pip install bitsandbytes` (T4 only — A100/H100/L4 use full-precision AdamW). |
| "Checkpoint is too small" when loading the fine-tune | The training didn't reach `target_updates`. Re-run with `--target-updates 45000` (or lower for a dry run). |
