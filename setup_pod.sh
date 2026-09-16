#!/usr/bin/env bash
# One-shot setup for a fresh RunPod (CUDA 12.8 image, any recent NVIDIA GPU).
# Installs ffmpeg + uv, clones the VibeVoice fork, builds its venv with pinned deps, runs the fork's tests,
# and clones this kit. Re-running is safe; existing pieces are reused.
# Usage:  bash setup_pod.sh            (then: hf auth login  for the private Yiddish-AI datasets)
set -euo pipefail
WORK=${WORK:-/workspace}
FORK_URL=${FORK_URL:-https://github.com/mendelg/VibeVoice.git}
FORK_BRANCH=${FORK_BRANCH:-podcast-finetuning}
REPO=${REPO:-$WORK/VibeVoice}
KIT_URL=${KIT_URL:-https://github.com/mendelg/yiddish-tts-kit.git}
KIT=${KIT:-$WORK/yiddish-tts-kit}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
export HF_HOME=${HF_HOME:-$WORK/hf-cache}
export HF_HUB_ENABLE_HF_TRANSFER=0          # the xet bulk path is faster for many small files
mkdir -p "$HF_HOME"

command -v nvidia-smi >/dev/null || { echo "No GPU visible; use a CUDA RunPod image." >&2; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg git; }
command -v uv >/dev/null || python3 -m pip install -q uv

if [[ ! -d $REPO/.git ]]; then git clone -q --branch "$FORK_BRANCH" "$FORK_URL" "$REPO"; else git -C "$REPO" pull -q; fi
cd "$REPO"
[[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
uv pip install -q --python .venv/bin/python torch torchaudio --index-url "$TORCH_INDEX"
uv pip install -q --python .venv/bin/python "transformers==4.51.3" "datasets==3.5.0" "accelerate==1.6.0" peft diffusers \
    librosa soundfile resampy numpy scipy ml-collections absl-py tqdm pytest tensorboard pyarrow hf_xet
uv pip install -q --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest tests -q 2>&1 | tail -1

if [[ ! -d $KIT/.git ]]; then git clone -q "$KIT_URL" "$KIT"; else git -C "$KIT" pull -q; fi
cat <<MSG

Setup done.
  python:   $REPO/.venv/bin/python
  kit:      $KIT
Next:
  hf auth login                                   # member of Yiddish-AI for the private sources
  cd $KIT
  \$PY data/materialize.py --manifest manifest/manifest.parquet --out $WORK/vibevoice-data/mix_v1 \\
      --sources teef_windows,studio,hasidic24,crowd_recital,crowd_whatsapp --min-quality 0.9
  MODEL=vibevoice/VibeVoice-1.5B RUN=mix_v1 MANIFEST_DIR=$WORK/vibevoice-data/mix_v1 VOICE_DROP=0.1 EPOCHS=2 \\
      bash training/run_vibevoice_podcast.sh --skip-install
MSG
