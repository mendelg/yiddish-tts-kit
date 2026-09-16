#!/usr/bin/env python3
"""Export the Pod's already transcribed and diarized episodes as VibeVoice podcast episodes.

Reads, per episode, ``transcripts/<id>/transcript.json`` (Whisper words with probabilities) and
``speaker_clips/<id>/diarization.json`` (pyannote turns) under the ASR data root, and writes one
JSON line per episode in the format ``vibevoice.finetune.prepare_podcast_jsonl`` expects:

    {"id", "audio", "url", "turns": [{"speaker", "start", "end", "text", "confidence"}]}

Each diarization turn's text is the transcript words that start inside it. ``confidence`` is the
mean Whisper word probability of those words (0.0 when a turn crosses one of the 600 s ASR chunk
boundaries, where words are unreliable) so the window builder can treat weak turns as barriers.
Nothing is re-transcribed or re-diarized and no audio is cut here.
"""
import argparse
import json
from pathlib import Path


def turn_words(words, start, end, tol=0.05):
    return [w for w in words if start - tol <= w["start"] < end - tol]


def crosses_chunk_boundary(start, end, chunk=600.0, margin=0.25):
    b = chunk
    while b <= end + margin:
        if start < b + margin and end > b - margin:
            return True
        b += chunk
    return False


def export_episode(root: Path, ident: str, require_source):
    transcript_path = root / "transcripts" / ident / "transcript.json"
    diar_path = root / "speaker_clips" / ident / "diarization.json"
    if not transcript_path.exists() or not diar_path.exists():
        return None, "missing_transcript_or_diarization"
    transcript = json.loads(transcript_path.read_text())
    if require_source and require_source not in (transcript.get("sources") or []):
        return None, "other_source"
    audio = Path(transcript.get("audio") or "")
    if not audio.exists():
        marker = root / "audio" / ident / "download.json"
        if marker.exists():
            audio = Path(json.loads(marker.read_text())["audio"])
    if not audio.exists():
        return None, "missing_audio"
    words = sorted(
        (dict(start=float(w["start"]), end=float(w["end"]), word=w["word"], probability=float(w.get("probability", 0.0)))
         for seg in transcript["segments"] for w in (seg.get("words") or [])
         if str(w.get("word", "")).strip() and float(w.get("end", 0)) > float(w.get("start", 0))),
        key=lambda w: w["start"])
    turns = []
    for t in json.loads(diar_path.read_text())["turns"]:
        start, end = float(t["start"]), float(t["end"])
        ws = turn_words(words, start, end)
        text = "".join(w["word"] for w in ws).strip()
        text = " ".join(text.split())
        if ws:
            confidence = sum(w["probability"] for w in ws) / len(ws)
            min_prob = min(w["probability"] for w in ws)
        else:
            confidence, min_prob = 1.0, 1.0
        if crosses_chunk_boundary(start, end):
            confidence = 0.0
        turns.append(dict(speaker=str(t["speaker"]), start=start, end=end, text=text,
                          confidence=round(confidence, 4), min_word_probability=round(min_prob, 4)))
    return dict(id=ident, audio=str(audio.resolve()), url=transcript.get("url"),
                sources=transcript.get("sources"), turns=turns), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/workspace/yiddish-podcast-asr/data"))
    p.add_argument("--out", type=Path, default=Path("data/vibevoice/episodes.jsonl"))
    p.add_argument("--require-source", default="teef_teef", help="Keep episodes whose transcript lists this source; '' for all")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    idents = sorted(d.name for d in (a.root / "transcripts").iterdir() if d.is_dir())
    if a.limit:
        idents = idents[: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rejected = {}
    written = 0
    hours = 0.0
    speaker_changes = 0
    with a.out.open("w") as f:
        for ident in idents:
            ep, why = export_episode(a.root, ident, a.require_source or None)
            if ep is None:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            written += 1
            hours += sum(t["end"] - t["start"] for t in ep["turns"] if t["text"]) / 3600
            speaker_changes += sum(1 for k in range(1, len(ep["turns"])) if ep["turns"][k]["speaker"] != ep["turns"][k - 1]["speaker"])
    print(json.dumps(dict(episodes=written, rejected=rejected, transcribed_turn_hours=round(hours, 2),
                          speaker_changes=speaker_changes, out=str(a.out)), indent=2))


if __name__ == "__main__":
    main()
