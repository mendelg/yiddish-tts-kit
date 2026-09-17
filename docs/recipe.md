# Recipe: what worked, what did not (September 2026)

**Model.** VibeVoice 1.5B. The 7B rendered no better to a native listener at 4x the cost.

**Loss.** `--ce_include_speech_tokens True --ce_skip_text_prefix True --ce_loss_weight 1.0`. The upstream trainer
never supervised the "keep speaking" decision and spent its text loss predicting transcripts; on Yiddish that
collapsed generation length within 600 steps. With the fix, full scripts render at CFG 1.3-1.5.

**Data shape.** Multi-speaker rows = real conversation windows (8-90 s of consecutive diarized turns) with one
`Speaker N:` line per turn and one voice prompt per speaker from the same episode. Single-speaker clips are
**chained** into 20-40 s multi-line rows with 0.3-0.8 s pauses; unchained short clips taught the model that a
clean short prompt means a short utterance, so an unseen clean voice stopped after one line.

**Text quality beats quantity.** The yiddish24 (hasidic24) machine transcripts were spot-checked by a native listener and judged good; they stay in.  Whisper transcripts at 0.85 confidence still carry wrong words that become wrong
pronunciations. Verified text (studio set, recital) fixed clarity. Drop `reyd_*` (YIVO/standard Yiddish).

**Voices.** Reference clips must be 8-12 s, one speaker, clean; 18 s prompts derailed generation. Any language
works for the prompt (an English clip clones fine) but its accent bleeds through; a native Yiddish recording is
best. Voice prompt dropout 0.1. More training speakers = less voice drift on unseen voices.

**Inference.** CFG 1.5 default; 2.5-3.0 forces adherence at the cost of flat delivery. 10 diffusion steps is fine.
Loshn-koydesh words (אתרוג) may be mispronounced: spell phonetically in scripts if needed.

**Non-finite gradients (the 150 h mix, Sept 17).** Three runs died at step 101: finite losses, then one optimizer
step turned every weight to nan. Cause: a hot, clipped broadcast chain (peak 1.0, rms 0.165, ~10 dB louder than
studio speech) overflowed the bf16 audio encoder's backward pass. Fixes in the fork: targets are now loudness-
normalized like prompts (`--normalize_target_audio`, default on); `training_step` zeroes non-finite gradients and logs
the row ids; the upstream EMA-of-head callback is off by default (its swap-back hooks never fire). Diagnose with
`data/inspect_rows.py` and `data/probe_rows.py`.

**Ops.** Pod: HF_HUB_ENABLE_HF_TRANSFER blocks the xet bulk path, disable it for many-small-file downloads;
Hub rate limit is 3000 API calls / 5 min (the materializer paces itself). Checkpoints: `--save_only_model`.

**Pronunciation (next round, Sept 17).** A native listener rated words ~75% right, emotion good. Hebrew-script Yiddish
is ambiguous on the page and loshn-koydesh / loanwords do not follow the letter rules, so the model guesses from
spelling learned off noisy transcripts. Fix: feed IPA instead of letters. `data/phonemize.py` wraps the Phonikud-yi
engine bundle (`PHONIKUD_YI_BUNDLE=/path/to/phonikud-yi-engine`, built by Phonikud-yi's `src/make_bundle.py`; the
model export is not in the GitHub repo but the full bundle is public on the Hub as `notmax123/phonikud-yi-engine`). `materialize.py --text-mode ipa` converts every row's text (one engine for
all sources, shipped IPA columns ignored for consistency; original kept in `text_orig`); `render_yiddish_podcast.py
--phonemize` converts scripts at render time so users still type Yiddish. Retrain (~2.5 h on the H200) as `mix_v2_ipa`.
