# Datasets in the mix

| source | repo | what | text | kept |
|---|---|---|---|---|
| teef_windows | Yiddish-AI/teef-teef-podcast-windows (ours) | 2-speaker podcast windows, 24 kHz | Whisper, mean word prob >= 0.85 | all |
| studio | Yiddish-AI/yiddish-tts (default) | 11 studio speakers, 22.05 kHz | WER-0 verified | `omni_*` only: reyd_* (YIVO) dropped, quarantined dropped, rows whose audio is not in the repo dropped |
| hasidic24 | Yiddish-AI/yiddish24-hasidic-speech | bulletin, shiurim, programmes, 24 kHz | machine (Yiddish Labs) | `tts_ok` rows; two-host programme flagged not usable until diarized |
| crowd_recital | ivrit-ai/crowd-recital-yi-whisper-training | crowd read-aloud, 30 s windows, 16 kHz mp3 in parquet | aligned to the read text | all (filter by quality_score) |
| crowd_whatsapp | ivrit-ai/crowd-whatsapp-yi-whisper-training | crowd voice notes, 30 s windows | crowd transcripts | all (filter by quality_score) |

Speaker ids: studio `speaker_id`; hasidic24 category id (one speaker per category except the programmes);
crowd sets `user_id`; Teef Teef per-episode diarization labels (not linked across episodes).

Rights: research use within the team. Audio rights stay with the original speakers/producers (Teef Teef with
permission; yiddish24; ivrit.ai crowd contributors). Keep derived repos private.

The combined manifest is published as **Yiddish-AI/yiddish-hasidic-tts-mix**: one table, each row a link to the
clip in its source repo (no audio copies). `data/materialize.py` fetches and converts what a training run needs.
