# Yiddish TTS kit

Everything needed to build the Hasidic-Yiddish speech mix, train the conversational VibeVoice model on it,
and render two-speaker podcasts. Built for the Yiddish-AI team; runs on one RunPod GPU (tested on a
96 GB RTX PRO 6000) and renders on a Mac.

```
data/      build_manifest.py        source metadata -> combined manifest (no audio moves)
           materialize.py           manifest -> local 24 kHz training rows (downloads, chains short clips)
           export_teef_teef_episodes.py, push_teef_dataset.py   how the Teef Teef windows dataset was made
training/  run_vibevoice_podcast.sh launcher for the VibeVoice fork (fine-tune on a manifest dir)
render/    render_yiddish_podcast.py  script JSON + voice clips -> wav; example scripts
docs/      recipe.md (what worked and why), datasets.md (sources, filters, rights)
manifest/  the current combined manifest (csv + parquet) and its summary
```

## The model

[VibeVoice](https://github.com/microsoft/VibeVoice) 1.5B, fine-tuned with our fork
**github.com/mendelg/VibeVoice, branch `podcast-finetuning`**, which adds multi-speaker (podcast)
training rows and fixes two upstream training bugs (the LM never learned to *keep speaking*; the 7B's
output head was mis-tied). A whole two-speaker conversation is generated in one pass, so turn-taking
and reactions are modelled rather than stitched.

## Quick start (Pod)

```bash
git clone --branch podcast-finetuning https://github.com/mendelg/VibeVoice /workspace/VibeVoice
cd /workspace/VibeVoice && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python \
   torch torchaudio --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python .venv/bin/python \
   "transformers==4.51.3" "datasets==3.5.0" "accelerate==1.6.0" peft diffusers librosa soundfile resampy \
   ml-collections absl-py tqdm pytest tensorboard pyarrow && uv pip install --python .venv/bin/python --no-deps -e .
hf auth login                      # member of Yiddish-AI (private sources)

git clone https://github.com/mendelg/yiddish-tts-kit /workspace/yiddish-tts-kit && cd /workspace/yiddish-tts-kit
PY=/workspace/VibeVoice/.venv/bin/python
$PY data/build_manifest.py --out manifest                 # or use the committed manifest/
$PY data/materialize.py --manifest manifest/manifest.csv --out /workspace/vibevoice-data/mix_v1 \
    --sources teef_windows,studio,crowd_recital,crowd_whatsapp --min-quality 0.9
MODEL=vibevoice/VibeVoice-1.5B RUN=mix_v1 MANIFEST_DIR=/workspace/vibevoice-data/mix_v1 VOICE_DROP=0.1 EPOCHS=2 \
    bash training/run_vibevoice_podcast.sh --skip-install
```

Checkpoints land in `/workspace/vibevoice-runs/<RUN>/checkpoint-*/lora`; copy one to the Mac and render:

```bash
python render/render_yiddish_podcast.py --script render/podcast_esrog.json \
    --voices A=host.wav B=guest.wav --checkpoint <run>/checkpoint-400 --cfg-scale 1.5 --device mps --out esrog.wav
```

See `docs/recipe.md` for the settings that mattered and `docs/datasets.md` for the data.
