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


## Current best render recipe (2026-09-17)

Adapter `mix_v2_ipa` checkpoint 2400 (phonetic input), Google Chirp3-HD Hebrew voices Puck (host) + Sadaltager (guest) as
11-13 s reference clips, CFG 1.5, 10 diffusion steps, whole script in one pass, then a 10% slowdown:

```bash
python render/render_yiddish_podcast.py --script dialogue/podcast_tefillin.json \
    --voices A=references/google/Puck_synth.wav B=references/google/Sadaltager_synth.wav \
    --checkpoint outputs/vibevoice/ipa-final --cfg-scale 1.5 --seed 3 --phonemize --out out/tefillin.wav
ffmpeg -i out/tefillin.wav -filter:a atempo=0.90 out/tefillin_slow90.wav        # the pace listeners preferred
python render/eval_renders.py --script dialogue/podcast_tefillin.json out/tefillin.wav   # duration, transcript, reached end
```

- The render prints `engine unsure about ...` for words nobody has reviewed; listen for those first.
- Pace follows the reference clips, so slow down in post (atempo 0.85-0.92) rather than by changing the model.
- If a render comes out short or the transcript shows a missing turn, re-seed first (3, 7, 42); if that fails on a long
  script (> 8 turns), add `--chunk-turns 4`. Chunking joins independent passes, so the voice resets slightly at joins;
  listeners preferred the single-pass render when it was complete.
- Whisper is evidence only: it drops words at its own window cuts (fixed to cut at quiet points) and mishears names.


### 7B on phonetic input (2026-09-18)

`MODEL=vibevoice/VibeVoice-7B EPOCHS=2 SAVE_STEPS=200 BATCH=16 ACCUM=1` on the same 20,728 IPA rows: 4.1 h on an H200
(117 GB used, gradient checkpointing on), eval loss 0.813 -> 0.785 where the 1.5B bottomed at 0.824. With letters as
input the 7B had never beaten the 1.5B; with phonemes it does. Render with `--model vibevoice/VibeVoice-7B` and the 7B
adapter (1.6 GB); about twice the render time of the 1.5B. `--next-round` (gold-contradicting rulings) was NOT used for
this run, so the same override table serves training and rendering.

### Correcting the phonemes ("Phonikud-yi, then Fable")

The engine tags every word HIGH / MED / LOW confidence; LOW is its own human-review queue, not noise. On the training
corpus (2.8M tokens) 65% is HIGH, 14% MED, 22% LOW. Two systematic misses: unpointed פ read as *f* in loanwords and
names (policy, computer, Putin, Trump), and loshn-koydesh vowels (מורא, מצה, מוח, במילא). The fix is a reviewed table,
`data/g2p_overrides.tsv` (word or phrase -> IPA), applied before the engine with longest phrase first, so the same
correction reaches training labels and inference prompts.

```bash
python data/g2p_review.py scripts/*.json                                  # what the engine is unsure about in the scripts
python data/g2p_review.py --manifest manifest/manifest.parquet --min-count 25   # frequency-sorted review queue
# review the rows by ear, append `word<TAB>ipa` to data/g2p_overrides.tsv; loading validates the phone inventory
```

Two tables: `data/g2p_overrides.tsv` (now: unsure words only; never a gold-lexicon word, because the current model was
trained on gold labels) and `data/g2p_overrides_next.tsv` (readings that contradict gold, e.g. komets-vs-pasekh rulings from
`data/review_komets_pasekh.tsv`). The second is applied only with `materialize --next-round` / `render --next-round`, so
training labels and renders change together. The materialize cache is keyed on the overrides file, so editing the table re-phonemizes on the next run. Digits and
Latin words are dropped by the engine: write numbers out in Yiddish words in scripts (the render prints a warning).
