#!/usr/bin/env python3
"""Build the combined Hasidic-Yiddish speech manifest from the source datasets' metadata only.

No audio is downloaded. Each row points at a clip (file URL, or parquet shard + row index) in its source
repo and carries normalized text, speaker, duration, quality and split. `materialize.py` turns the
manifest into local 24 kHz training rows; the manifest itself is published as the mix dataset.

Sources and policy (Hasidic pronunciation only):
  teef_windows    Yiddish-AI/teef-teef-podcast-windows      two-speaker conversation windows, Whisper text >= 0.85
  studio          Yiddish-AI/yiddish-tts (default config)    WER-0 verified clips; reyd_* (YIVO/standard Yiddish) dropped
  hasidic24       Yiddish-AI/yiddish24-hasidic-speech        yiddish24 clips, machine transcripts; tts_ok rows only;
                                                             the two-host conversation programme is kept but flagged
  crowd_recital   ivrit-ai/crowd-recital-yi-whisper-training crowd read-aloud, 30 s windows, aligned to the read text
  crowd_whatsapp  ivrit-ai/crowd-whatsapp-yi-whisper-training crowd voice notes with crowd transcripts, 30 s windows
  broadcast24     Yiddish-AI/yiddish-tts (yiddish24 config)  180 h, two yiddish24 news narrators, unverified Whisper text;
                                                             clean rows only; cap hours per narrator at materialize time
"""
import argparse, csv, io, json, re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

YIVO = re.compile(r"[ִַָּֿׁׂ]|אַ|אָ|פֿ|ײַ|ייִ")
TS = re.compile(r"<\|(\d+\.\d+)\|>")
COLUMNS = ["id", "source", "source_repo", "source_path", "row_index", "url", "text", "num_lines", "num_speakers",
           "speaker", "speaker_reliable", "recording_id", "duration", "sample_rate", "codec", "transcript_quality",
           "quality_score", "split", "tts_ok", "note"]


def url_for(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def row(**kw):
    r = {c: None for c in COLUMNS}; r.update(kw); return r


def teef_windows(api):
    repo = "Yiddish-AI/teef-teef-podcast-windows"
    p = hf_hub_download(repo, "windows/metadata.csv", repo_type="dataset")
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out.append(row(id=f"teef_{Path(r['file_name']).stem}_{r['episode']}", source="teef_windows", source_repo=repo,
                       source_path=f"windows/{r['file_name']}", url=url_for(repo, f"windows/{r['file_name']}"),
                       text=r["text"], num_lines=int(r["num_turns"]), num_speakers=int(r["num_speakers"]),
                       speaker=r["speakers"], speaker_reliable=True, recording_id=r["episode"],
                       duration=float(r["duration"]), sample_rate=24000, codec="wav",
                       transcript_quality="machine_whisper_conf>=0.85", quality_score=float(r["min_confidence"]),
                       split=r["split"], tts_ok=True, note="voice prompts: " + r["voice_prompts"]))
    return out


def studio(api):
    repo = "Yiddish-AI/yiddish-tts"
    present = {f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("audio/")}
    p = hf_hub_download(repo, "metadata.csv", repo_type="dataset")
    out, dropped = [], Counter()
    for r in csv.DictReader(open(p, encoding="utf-8")):
        name = r["file_name"]
        if not name.split("/")[1].startswith("omni_"):
            dropped["reyd_or_other_source"] += 1; continue
        if r["label_status"] != "ok":
            dropped["label_status_" + r["label_status"]] += 1; continue
        if YIVO.search(r["text"]):
            dropped["yivo_marks"] += 1; continue
        if name not in present:
            dropped["audio_missing_in_repo"] += 1; continue
        split = {"omni_train": "train", "omni_dev": "validation", "omni_test": "test"}.get("_".join(name.split("/")[1].split("_")[:2]), "train")
        out.append(row(id=f"studio_{Path(name).stem}", source="studio", source_repo=repo, source_path=name, url=url_for(repo, name),
                       text=" ".join(r["text"].split()), num_lines=1, num_speakers=1, speaker=f"studio_{r['speaker_id']}",
                       speaker_reliable=True, recording_id=Path(name).stem.rsplit("_", 1)[0], duration=None, sample_rate=22050,
                       codec="wav", transcript_quality="verified_wer0", quality_score=1.0, split=split, tts_ok=True, note="phonemes available in source"))
    print("studio dropped:", dict(dropped))
    return out


def hasidic24(api):
    repo = "Yiddish-AI/yiddish24-hasidic-speech"
    p = hf_hub_download(repo, "clips/metadata.csv", repo_type="dataset")
    out, dropped = [], Counter()
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["tts_ok"] != "True":
            dropped["not_tts_ok"] += 1; continue
        conv = r["genre"] == "dresdner_shmeltzer_conversation"
        split = {"train": "train", "dev": "validation", "test": "test"}.get(r["split"], r["split"])
        out.append(row(id=f"y24_{Path(r['file_name']).stem}", source="hasidic24", source_repo=repo, source_path=f"clips/{r['file_name']}",
                       url=url_for(repo, f"clips/{r['file_name']}"), text=" ".join(r["text"].split()), num_lines=1,
                       num_speakers=2 if conv else 1, speaker=f"y24_{r['speaker']}", speaker_reliable=not conv,
                       recording_id=r["source_recording"], duration=float(r["duration"]), sample_rate=24000, codec="wav",
                       transcript_quality="machine_yiddishlabs", quality_score=1.0 - float(r["low_confidence_share"]),
                       split=split, tts_ok=not conv, note=f"genre={r['genre']}" + ("; two hosts, not diarized" if conv else "")))
    print("hasidic24 dropped:", dict(dropped))
    return out


def crowd(api, repo, source, quality_label):
    fs = HfFileSystem()
    files = sorted(f for f in fs.ls(f"datasets/{repo}/data", detail=False) if f.endswith(".parquet"))
    out = []
    for f in files:
        rel = f.split(f"datasets/{repo}/", 1)[1]
        split = "validation" if rel.split("/")[-1].startswith("eval") else "train"
        with fs.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["transcript", "metadata", "has_timestamps"])
        for i, (tr, md) in enumerate(zip(t.column("transcript").to_pylist(), t.column("metadata").to_pylist())):
            marks = TS.findall(tr)
            text_lines = [s.strip() for s in TS.split(tr) if s.strip() and not re.fullmatch(r"\d+\.\d+", s.strip())]
            text = "\n".join(text_lines)
            last = float(marks[-1]) if marks else float(md.get("duration") or 30.0)
            out.append(row(id=f"{source}_{md['entry_id']}_{int(md['seek']*100):07d}", source=source, source_repo=repo, source_path=rel,
                           row_index=i, url=url_for(repo, rel), text=text, num_lines=len(text_lines), num_speakers=1,
                           speaker=f"{source}_{md['user_id']}", speaker_reliable=True, recording_id=md["entry_id"],
                           duration=round(min(last, float(md.get("duration") or 30.0)), 2), sample_rate=16000, codec="mp3_in_parquet",
                           transcript_quality=quality_label, quality_score=round(float(md.get("quality_score") or 0), 4),
                           split=split, tts_ok=True, note=f"seek={md['seek']:.2f}s in entry; segment times in source transcript"))
    return out

def broadcast24(api):
    repo = "Yiddish-AI/yiddish-tts"
    p = hf_hub_download(repo, "yiddish24/metadata.csv", repo_type="dataset")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    recs = sorted({r["source_id"] for r in rows})
    import hashlib
    held = {rec for rec in recs if int(hashlib.sha256(rec.encode()).hexdigest()[:8], 16) % 100 < 3}  # ~3% of recordings -> validation
    out, dropped = [], Counter()
    for r in rows:
        if r["clean"] != "true":
            dropped["not_clean"] += 1; continue
        if YIVO.search(r["text"]):
            dropped["yivo_marks"] += 1; continue
        path = "yiddish24/" + r["file_name"]; narrator = r["source_id"].split("_")[0]
        dur = round(float(r["end_time"]) - float(r["start_time"]), 3)
        out.append(row(id=f"b24_{Path(path).stem}", source="broadcast24", source_repo=repo, source_path=path, url=url_for(repo, path),
                       text=" ".join(r["text"].split()), num_lines=1, num_speakers=1, speaker=f"b24_{narrator}", speaker_reliable=True,
                       recording_id=r["source_id"], duration=dur, sample_rate=16000, codec="wav",
                       transcript_quality="machine_whisper_unverified", quality_score=None,
                       split="validation" if r["source_id"] in held else "train", tts_ok=True,
                       note=f"start={r['start_time']} end={r['end_time']} in yiddish24/source_audio/{r['source_id']}.wav; nikud+ipa in source"))
    print("broadcast24 dropped:", dict(dropped))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("manifest"))
    a = ap.parse_args()
    api = HfApi()
    rows = []
    for fn, args in [(teef_windows, ()), (studio, ()), (hasidic24, ()),
                     (crowd, ("ivrit-ai/crowd-recital-yi-whisper-training", "crowd_recital", "recital_aligned_to_read_text")),
                     (crowd, ("ivrit-ai/crowd-whatsapp-yi-whisper-training", "crowd_whatsapp", "crowd_transcribed_aligned")),
                     (broadcast24, ())]:
        part = fn(api, *args); rows.extend(part); print(f"{part[0]['source']}: {len(part)} rows", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    table = pa.Table.from_pylist([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])
    pq.write_table(table, a.out / "manifest.parquet")
    summary = defaultdict(lambda: dict(rows=0, hours=0.0, speakers=set(), tts_ok=0))
    for r in rows:
        s = summary[r["source"]]; s["rows"] += 1; s["hours"] += (r["duration"] or 0) / 3600; s["speakers"].add(r["speaker"]); s["tts_ok"] += bool(r["tts_ok"])
    table_md = ["| source | rows | hours (known) | speakers | tts_ok rows | text |", "|---|---|---|---|---|---|"]
    q = {r["source"]: r["transcript_quality"] for r in rows}
    for k, s in summary.items():
        table_md.append(f"| {k} | {s['rows']} | {s['hours']:.1f} | {len(s['speakers'])} | {s['tts_ok']} | {q[k]} |")
    (a.out / "summary.md").write_text("\n".join(table_md) + "\n", encoding="utf-8")
    print("\n".join(table_md)); print("total rows", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
