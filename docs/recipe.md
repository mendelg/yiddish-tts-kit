# Recipe: what worked, what did not (September 2026)

**Model.** VibeVoice 1.5B. The 7B rendered no better to a native listener at 4x the cost.

**Loss.** `--ce_include_speech_tokens True --ce_skip_text_prefix True --ce_loss_weight 1.0`. The upstream trainer
never supervised the "keep speaking" decision and spent its text loss predicting transcripts; on Yiddish that
collapsed generation length within 600 steps. With the fix, full scripts render at CFG 1.3-1.5.

**Data shape.** Multi-speaker rows = real conversation windows (8-90 s of consecutive diarized turns) with one
`Speaker N:` line per turn and one voice prompt per speaker from the same episode. Single-speaker clips are
**chained** into 20-40 s multi-line rows with 0.3-0.8 s pauses; unchained short clips taught the model that a
clean short prompt means a short utterance, so an unseen clean voice stopped after one line.

**Text quality beats quantity.** Whisper transcripts at 0.85 confidence still carry wrong words that become wrong
pronunciations. Verified text (studio set, recital) fixed clarity. Drop `reyd_*` (YIVO/standard Yiddish).

**Voices.** Reference clips must be 8-12 s, one speaker, clean; 18 s prompts derailed generation. Any language
works for the prompt (an English clip clones fine) but its accent bleeds through; a native Yiddish recording is
best. Voice prompt dropout 0.1. More training speakers = less voice drift on unseen voices.

**Inference.** CFG 1.5 default; 2.5-3.0 forces adherence at the cost of flat delivery. 10 diffusion steps is fine.
Loshn-koydesh words (אתרוג) may be mispronounced: spell phonetically in scripts if needed.

**Ops.** Pod: HF_HUB_ENABLE_HF_TRANSFER blocks the xet bulk path, disable it for many-small-file downloads;
Hub rate limit is 3000 API calls / 5 min (the materializer paces itself). Checkpoints: `--save_only_model`.
