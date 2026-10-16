"""Index hasil akhir dalam Drive, tanpa menduplikasi model, data, audio, atau hasil evaluasi."""
import argparse
import json
from pathlib import Path
import pandas as pd
from common import final_model_path, load_config, ws

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(cfg['_project_root'])
    out = Path(cfg['final_output']['folder'])
    result = ws(cfg, 'results')
    model = final_model_path(cfg)
    test_csv = ws(cfg,'test.csv')
    if not model.exists() or model.stat().st_size < 4_000_000_000:
        raise FileNotFoundError(f'Model final 45k belum lengkap: {model}')
    required = [result/'comparison_table.csv',result/'significance.json',result/'conclusion.json',test_csv]
    missing = [p for p in required if not p.exists() or p.stat().st_size < 1]
    if missing: raise RuntimeError('Output belum lengkap: '+', '.join(map(str,missing)))
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'OUTPUT_MANIFEST.json'
    if manifest.exists() and not a.force:
        print('✅ Manifest output akhir sudah tersedia -> SKIP:', manifest)
        return
    t=pd.read_csv(test_csv)
    sample_rows=[]
    n=min(int(cfg['final_output'].get('preview_samples',5)),len(t))
    for r in t.head(n).itertuples(index=False):
        uid=str(r.id)
        base=ws(cfg,'synth','test','base',f'{uid}.wav')
        ft=ws(cfg,'synth','test','finetuned',f'{uid}.wav')
        sample_rows.append({'id':uid,'text':r.text,'base_audio':str(base),'finetuned_audio':str(ft)})
    pd.DataFrame(sample_rows).to_csv(out/'samples_manifest.csv',index=False)
    payload={
        'project':'Indonesian_TTS_FineTuning',
        'checkpoint_45000':str(model),
        'workspace':str(ws(cfg)),
        'dataset_train':str(ws(cfg,'train.csv')),
        'dataset_val':str(ws(cfg,'val.csv')),
        'dataset_test':str(test_csv),
        'base_audio_directory':str(ws(cfg,'synth','test','base')),
        'finetuned_audio_directory':str(ws(cfg,'synth','test','finetuned')),
        'evaluation_results_directory':str(result),
        'comparison_table':str(result/'comparison_table.csv'),
        'wilcoxon':str(result/'significance.json'),
        'plots':sorted(str(p) for p in result.glob('comparison_*.png')),
        'preview':str(ws(cfg,'audio_preview.html')),
        'source':str(root/'src'),
        'notebook':str(root/'Indonesian_TTS_FineTuning.ipynb'),
        'storage_policy':'Semua artefak berada satu kali pada Drive; FINAL_OUTPUT hanya indeks/referensi.',
    }
    manifest.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'README.md').write_text("""# Hasil Akhir Indonesian TTS Fine-Tuning

Semua berkas tersimpan langsung di Google Drive, tanpa backup atau penyalinan checkpoint/audio.

- `../workspace_full/checkpoints/id_tts_full/model_45000.pt` — hasil training.
- `../workspace_full/synth/test/base/` — audio Base.
- `../workspace_full/synth/test/finetuned/` — audio Fine-Tuned.
- `../workspace_full/results/` — hasil WER, CER, SIM, Wilcoxon, tabel dan grafik.
- `../workspace_full/audio_preview.html` — preview audio (tautan lokal bergantung lokasi Drive mount).
- `samples_manifest.csv` — referensi audio contoh.
- `OUTPUT_MANIFEST.json` — indeks semua lokasi output.

Folder `FINAL_OUTPUT` ini **hanya indeks**, bukan salinan hasil, agar penyimpanan tetap ringan.
""",encoding='utf-8')
    print('✅ Output akhir tersimpan:',out)
    print(json.dumps(payload,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
