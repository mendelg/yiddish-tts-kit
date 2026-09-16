#!/usr/bin/env python3
"""Publish the Teef Teef two-speaker windows as a Hugging Face audio-folder dataset.

Layout pushed:
  windows/metadata.csv + windows/<episode>/<id>.wav   (conversation windows, the training rows)
  prompts/metadata.csv + prompts/<episode>/<file>.wav (per-speaker voice-prompt clips)
  README.md
Audio files are hard-linked into a staging folder (no second copy on disk) and uploaded with
upload_large_folder, which batches commits under the Hub's rate limits and resumes if interrupted.
"""
import argparse, csv, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--src", type=Path, default=Path("/workspace/vibevoice-data/podcast_c0.85_m90_g2.0"))
p.add_argument("--stage", type=Path, default=Path("/workspace/hf-upload/teef-teef-podcast-windows"))
p.add_argument("--repo", default="Yiddish-AI/teef-teef-podcast-windows")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

rows = []
for split in ("train", "validation"):
    for line in (a.src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); r["split"] = split; rows.append(r)
print("windows:", len(rows))

def link(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        try: os.link(src, dest)
        except OSError: import shutil; shutil.copy2(src, dest)

win_rows, prompt_rows, seen_prompts = [], [], {}
for r in rows:
    ep = r["episode"]; src = Path(r["audio"]); rel = f"{ep}/{src.name}"
    if not a.dry_run: link(src, a.stage / "windows" / rel)
    prompt_rels = []
    for k, pth in enumerate(r.get("voice_prompts") or []):
        pp = Path(pth); prel = f"{ep}/{pp.name}"; prompt_rels.append(prel)
        if prel not in seen_prompts:
            seen_prompts[prel] = True
            if not a.dry_run: link(pp, a.stage / "prompts" / prel)
            prompt_rows.append(dict(file_name=prel, episode=ep, speaker=r["speakers"][k] if k < len(r["speakers"]) else "",
                                    speaker_index=k, source_start=pp.stem.split("_", 1)[1] if "_" in pp.stem else ""))
    win_rows.append(dict(file_name=rel, text=r["text"], episode=ep, split=r["split"], start=r["start"], end=r["end"],
                         duration=round(r["duration"], 3), num_turns=r["num_turns"], num_speakers=r["num_speakers"],
                         speakers="|".join(r["speakers"]), speaker_changes=r["speaker_changes"],
                         min_confidence=round(r["min_confidence"], 3), voice_prompts="|".join(prompt_rels)))

hours = sum(r["duration"] for r in win_rows) / 3600
episodes = sorted({r["episode"] for r in win_rows})
print(f"{len(win_rows)} windows, {hours:.1f} h, {len(episodes)} episodes, {len(prompt_rows)} prompt clips")
if a.dry_run: raise SystemExit

a.stage.mkdir(parents=True, exist_ok=True)
for name, data in (("windows", win_rows), ("prompts", prompt_rows)):
    with (a.stage / name / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

n_train = sum(r["split"] == "train" for r in win_rows); n_val = len(win_rows) - n_train
h_train = sum(r["duration"] for r in win_rows if r["split"] == "train") / 3600
(a.stage / "README.md").write_text(f"""---
license: other
language: [yi]
task_categories: [text-to-speech, automatic-speech-recognition]
tags: [yiddish, hasidic, podcast, conversation, multi-speaker]
size_categories: [1K<n<10K]
configs:
  - config_name: windows
    data_files: windows/metadata.csv
    default: true
  - config_name: prompts
    data_files: prompts/metadata.csv
---

# Teef Teef podcast: two-speaker conversation windows

{len(win_rows)} windows ({hours:.1f} h) of real Hasidic Yiddish conversation cut from {len(episodes)} episodes of the
Teef Teef podcast, for training conversational / multi-speaker TTS (built for VibeVoice fine-tuning).
Every window is a stretch of consecutive diarized turns cut from the original recording, so pauses,
overlaps and reactions between the speakers are real. Each window has two speakers.

| split | windows | hours |
|---|---|---|
| train | {n_train} | {h_train:.1f} |
| validation | {n_val} | {hours - h_train:.1f} |

Splits hold out whole episodes.

## Layout

- `windows/<episode>/<id>.wav`: 24 kHz mono 16-bit, 8-90 s. `windows/metadata.csv`:
  - `text`: the script, one line per turn, `Speaker 0: ...` / `Speaker 1: ...`, speakers numbered by first
    appearance in the window; same-speaker turns closer than 2 s are merged into one line
  - `episode`, `start`, `end`: position in the source recording (YouTube id); `duration`, `num_turns`,
    `num_speakers`, `speaker_changes`
  - `speakers`: the episode-level diarization labels behind Speaker 0|Speaker 1
  - `min_confidence`: lowest mean Whisper word probability among the window's turns (all >= 0.85)
  - `voice_prompts`: the prompt clips (below) for Speaker 0|Speaker 1, in that order
- `prompts/<episode>/<speaker>_<start>.wav`: 3-12 s single-speaker clips of the same speaker from the same
  episode, outside the window, for voice conditioning. `prompts/metadata.csv` lists episode and speaker.

```python
from datasets import load_dataset
ds = load_dataset("{a.repo}", "windows")   # audio + text + metadata
```

## How it was built

Whole episodes were transcribed with a Yiddish Whisper model (word timestamps) and diarized with
pyannote. Turns whose mean word probability was below 0.85, turns that were textless for over 1 s, and
turns crossing the ASR's 600 s chunk boundaries end a window rather than entering one. Windows are
8-90 s, gaps between turns up to 2 s, 0.15 s of padding bounded by neighbouring turns. Transcripts are
machine output, not human-reviewed; expect scattered wrong words.

## Rights

Audio: Teef Teef podcast, used with the producers' permission for this project; not for redistribution
outside the team. Transcripts are derived machine output.
""", encoding="utf-8")

from huggingface_hub import HfApi
api = HfApi()
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
print("repo ready:", a.repo, flush=True)
api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(a.stage), print_report=False)
print("UPLOAD DONE", a.repo, flush=True)
