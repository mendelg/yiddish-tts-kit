#!/usr/bin/env bash
# VibeVoice podcast fine-tune on the Pod, from the Teef Teef data that is already there.
# Steps: fork checkout -> venv -> export episodes -> build windows (dry run first) -> train.
# Usage: bash run_vibevoice_podcast.sh [--dry-run] [--skip-install]
set -euo pipefail
WORK=${WORK:-/workspace}
FORK_URL=${FORK_URL:-https://github.com/mendelg/VibeVoice.git}
FORK_BRANCH=${FORK_BRANCH:-podcast-finetuning}
REPO=${REPO:-$WORK/VibeVoice}
ASR_ROOT=${ASR_ROOT:-$WORK/yiddish-podcast-asr/data}
DATA=${DATA:-$WORK/vibevoice-data}
MODEL=${MODEL:-vibevoice/VibeVoice-1.5B}
RUN=${RUN:-teef_teef_podcast_v3}
OUT=${OUT:-$WORK/vibevoice-runs/$RUN}
MIN_CONF=${MIN_CONF:-0.85}
MAX_SECONDS=${MAX_SECONDS:-90}
MERGE_GAP=${MERGE_GAP:-2.0}
LORA_R=${LORA_R:-32}
PODCAST_DIR=${PODCAST_DIR:-$DATA/podcast_c${MIN_CONF}_m${MAX_SECONDS}_g${MERGE_GAP}}
export HF_HOME=${HF_HOME:-$WORK/hf-cache}
export TOKENIZERS_PARALLELISM=false
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY=0; SKIP_INSTALL=0
for arg in "$@"; do case "$arg" in --dry-run) DRY=1;; --skip-install) SKIP_INSTALL=1;; esac; done

if (( ! SKIP_INSTALL )); then
  command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y ffmpeg; }
  command -v uv >/dev/null || python3 -m pip install -q uv
  if [[ ! -d $REPO/.git ]]; then git clone -q --branch "$FORK_BRANCH" "$FORK_URL" "$REPO"; else git -C "$REPO" fetch -q && git -C "$REPO" checkout -q "$FORK_BRANCH" && git -C "$REPO" pull -q; fi
  cd "$REPO"
  [[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
  # torch first, pinned to the image's CUDA (12.8 on the RunPod images used so far; override with TORCH_INDEX).
  uv pip install --python .venv/bin/python torch torchaudio --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  uv pip install --python .venv/bin/python "transformers==4.51.3" "datasets==3.5.0" "accelerate==1.6.0" peft diffusers librosa soundfile resampy numpy scipy ml-collections absl-py tqdm pytest tensorboard hf_transfer
  uv pip install --python .venv/bin/python --no-deps -e .
  # flash-attn is optional (training and rendering use SDPA otherwise); building it needs a matching nvcc and ~30 min.
  if [[ ${FLASH_ATTN:-0} == 1 ]]; then
    .venv/bin/python -c "import flash_attn" 2>/dev/null || uv pip install --python .venv/bin/python ninja flash-attn --no-build-isolation || echo "flash-attn unavailable; SDPA will be used"
  fi
  .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in torch'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
  .venv/bin/python -m pytest tests -q
fi
cd "$REPO"; PY=$REPO/.venv/bin/python
# RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1; without the package, downloads fail as "file not found".
if ! $PY -c "import hf_transfer" 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=0; fi

# With MANIFEST_DIR pointing at ready train/validation.jsonl (e.g. from data/materialize.py) skip the Teef Teef
# export/window steps entirely; they only apply to the original single-source Pod layout.
if [[ -n ${MANIFEST_DIR:-} && -s $MANIFEST_DIR/train.jsonl ]]; then
  echo "Using manifests in $MANIFEST_DIR ($(wc -l < "$MANIFEST_DIR/train.jsonl") train rows)"
  (( DRY )) && exit 0
else
mkdir -p "$DATA"
KIT_DATA="$(cd "$HERE/.." && pwd)/data"; EXPORT="$HERE/export_teef_teef_episodes.py"; [[ -f $EXPORT ]] || EXPORT="$KIT_DATA/export_teef_teef_episodes.py"
[[ -s $DATA/episodes.jsonl ]] || $PY "$EXPORT" --root "$ASR_ROOT" --out "$DATA/episodes.jsonl"

PREP=( -m vibevoice.finetune.prepare_podcast_jsonl --episodes "$DATA/episodes.jsonl" --out "$PODCAST_DIR"
       --min-seconds 8 --max-seconds "$MAX_SECONDS" --max-gap 2.0 --merge-gap "$MERGE_GAP" --require-speaker-change
       --min-confidence "$MIN_CONF" --prompt-min 3 --prompt-max 12 --validation-fraction 0.1 )
$PY "${PREP[@]}" --dry-run
(( DRY )) && exit 0
[[ -s $PODCAST_DIR/train.jsonl ]] || $PY "${PREP[@]}"
fi
# MANIFEST_DIR= points training at a different train/validation.jsonl pair (e.g. the studio+podcast merge).
MANIFEST_DIR=${MANIFEST_DIR:-$PODCAST_DIR}

GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}')
if [[ $MODEL == *7B* ]]; then
  # 7B: ~15 GB of weights in bf16 plus the fully trained diffusion head; keep micro-batches small.
  if (( GPU_GB >= 130 )); then BS=8; ACC=2; elif (( GPU_GB >= 90 )); then BS=4; ACC=4; elif (( GPU_GB >= 70 )); then BS=2; ACC=8; else BS=1; ACC=16; fi; CKPT=True
  EPOCHS=${EPOCHS:-4}
elif (( GPU_GB >= 130 )); then BS=8; ACC=2; CKPT=False; elif (( GPU_GB >= 70 )); then BS=4; ACC=4; CKPT=False; elif (( GPU_GB >= 40 )); then BS=2; ACC=8; CKPT=True; else BS=1; ACC=16; CKPT=True; fi
BS=${BATCH:-$BS}; ACC=${ACCUM:-$ACC}   # BATCH= / ACCUM= override the automatic choice (keep BATCH*ACCUM = 16)
echo "MODEL $MODEL on GPU ${GPU_GB} GB -> batch $BS x accumulation $ACC, gradient checkpointing $CKPT"

# Weights & Biases: set WANDB_API_KEY (or run `wandb login` once on the Pod) and the run streams to wandb.ai.
# WANDB_PROJECT names the project (default yiddish-vibevoice); the run is named after $RUN. Otherwise TensorBoard only.
if [[ -n ${WANDB_API_KEY:-} ]] || [[ -s ${HOME}/.netrc && $(grep -c api.wandb.ai "${HOME}/.netrc") -gt 0 ]]; then
  export WANDB_PROJECT=${WANDB_PROJECT:-yiddish-vibevoice} WANDB_NAME=${WANDB_NAME:-$RUN} WANDB_DIR=${WANDB_DIR:-$WORK/wandb}
  REPORT_TO=all; echo "wandb: project $WANDB_PROJECT, run $WANDB_NAME"
else
  REPORT_TO=tensorboard; echo "wandb: not configured (set WANDB_API_KEY to enable)"
fi
mkdir -p "$OUT"
$PY -m vibevoice.finetune.train_vibevoice \
  --model_name_or_path "$MODEL" \
  --train_jsonl "$MANIFEST_DIR/train.jsonl" --validation_jsonl "$MANIFEST_DIR/validation.jsonl" \
  --text_column_name text --audio_column_name audio --voice_prompts_column_name voice_prompts \
  --normalize_speaker_ids True --voice_prompt_drop_rate "${VOICE_DROP:-0.2}" \
  --output_dir "$OUT" \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$ACC" --gradient_checkpointing "$CKPT" \
  --learning_rate 2.5e-5 --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs "${EPOCHS:-8}" \
  --logging_steps 10 --save_steps "${SAVE_STEPS:-100}" --save_total_limit "${SAVE_LIMIT:-4}" --eval_strategy steps --eval_steps "${SAVE_STEPS:-100}" --do_eval \
  --report_to $REPORT_TO --remove_unused_columns False --bf16 True --do_train --save_only_model True \
  --gradient_clipping --max_grad_norm 0.8 \
  --ddpm_batch_mul 4 --diffusion_loss_weight 1.4 --train_diffusion_head True \
  --ce_loss_weight "${CE_WEIGHT:-1.0}" --ce_include_speech_tokens "${CE_CONTINUE:-True}" --ce_skip_text_prefix "${CE_SKIP_PREFIX:-True}" \
  --lora_r "$LORA_R" --lora_alpha $((LORA_R * 2)) --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --seed 42 ${EXTRA_ARGS:-}
echo "Done. Adapters: $OUT/lora (and $OUT/checkpoint-*/lora). Render with render_yiddish_podcast.py."
