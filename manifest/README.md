---
license: other
language: [yi]
task_categories: [text-to-speech, automatic-speech-recognition]
tags: [yiddish, hasidic, speech, manifest, multi-speaker, podcast]
size_categories: [10K<n<100K]
configs:
  - config_name: default
    data_files: manifest.parquet
---

# Hasidic Yiddish speech mix (manifest)

One table over six speech sources (five repositories), restricted to **Hasidic (Central) Yiddish** pronunciation, for training
conversational TTS. This repo holds **no audio**: every row links to the clip in its source repository
(`url`, or parquet shard + `row_index` for the ivrit.ai sets). Fetch and convert what you need with
`data/materialize.py` from [mendelg/yiddish-tts-kit](https://github.com/mendelg/yiddish-tts-kit).

| source | rows | hours (known) | speakers | tts_ok rows | text |
|---|---|---|---|---|---|
| teef_windows | 2227 | 15.2 | 47 | 2227 | machine_whisper_conf>=0.85 |
| studio | 5633 | 0.0 | 11 | 5633 | verified_wer0 |
| hasidic24 | 3877 | 8.4 | 19 | 2950 | machine_yiddishlabs |
| crowd_recital | 13722 | 76.6 | 67 | 13722 | recital_aligned_to_read_text |
| crowd_whatsapp | 3613 | 19.8 | 576 | 3613 | crowd_transcribed_aligned |
| broadcast24 | 107237 | 149.8 | 2 | 107237 | machine_whisper_unverified |

Hours are summed from known clip durations. Speaker counts: studio and crowd ids are real speakers; hasidic24
uses one id per yiddish24 category; broadcast24 has two narrators (klar, spitzer) (one speaker each except the two-host programme); Teef Teef labels are
per-episode diarization labels, not linked across episodes.

## Sources and filters

| source | repo | text | filter |
|---|---|---|---|
| teef_windows | Yiddish-AI/teef-teef-podcast-windows | Whisper, mean word probability >= 0.85 | all; 2 speakers per row, script in `text` with `Speaker 0/1:` lines |
| studio | Yiddish-AI/yiddish-tts (default) | verified, WER 0 | `omni_*` recordings only: `reyd_*` (YIVO / standard Yiddish) removed, quarantined rows removed, rows whose audio is absent from the repo removed |
| hasidic24 | Yiddish-AI/yiddish24-hasidic-speech | machine (Yiddish Labs API) | `tts_ok` rows; the two-host programme (`dresdner_shmeltzer_conversation`) has `tts_ok=false` until diarized |
| crowd_recital | ivrit-ai/crowd-recital-yi-whisper-training | aligned to the text that was read | all; use `quality_score` |
| crowd_whatsapp | ivrit-ai/crowd-whatsapp-yi-whisper-training | crowd transcripts, aligned | all; use `quality_score` |
| broadcast24 | Yiddish-AI/yiddish-tts (`yiddish24` config) | Whisper, unverified | `clean` rows (no digits/Latin); two narrators, cap hours per narrator when training |

## Columns

`id`, `source`, `source_repo`, `source_path`, `row_index` (parquet rows), `url`, `text` (plain Hebrew-script
Yiddish; multi-line for multi-turn rows, timestamp tokens stripped), `num_lines`, `num_speakers`, `speaker`,
`speaker_reliable`, `recording_id`, `duration` (s), `sample_rate`, `codec` (`wav` / `mp3_in_parquet`),
`transcript_quality`, `quality_score`, `split` (train / validation / test as in the source), `tts_ok`, `note`.

## Rights

Research use within the team. Audio rights remain with the original speakers and producers (Teef Teef with the
producers' permission; yiddish24; ivrit.ai crowd contributors under their own terms). Keep derived repos private.

Built by `data/build_manifest.py` in the kit; rebuild it when a source changes.
