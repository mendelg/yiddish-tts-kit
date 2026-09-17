#!/usr/bin/env python3
"""Turn the combined manifest into local VibeVoice training rows.

For every manifest row selected by --sources/--splits/--tts-ok, fetch the audio from its source repo
(file download, or parquet shard + row extraction), convert to 24 kHz mono 16-bit WAV, and write
train/validation JSONL rows {"text", "audio", "voice_prompts"} the trainer reads with
--normalize_speaker_ids True. Short single clips of one speaker are chained into 20-40 s multi-line rows
(a clean short prompt otherwise teaches "short prompt => short utterance"); rows that already span 15 s+
are kept as they are. Voice prompts are 3-12 s clips of the same speaker (by `speaker`), never the row itself.
Downloads are cached; rerunning only converts what is missing. Files are fetched with the Hub's xet path
(keep HF_HUB_ENABLE_HF_TRANSFER unset) and paced by rate limits automatically.
"""
import argparse, csv, io, json, random, subprocess, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def to_wav24k(src: Path, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)], check=True)
    tmp.replace(dest)


def fetch_files(rows, cache: Path, workers: int):
    """Download file-backed clips one by one (skipping what is already cached), convert to 24 kHz.
    A file that still fails after 3 tries is dropped from `rows` and reported; a rate-limit refusal pauses 5 min."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    todo = [r for r in rows if r["codec"] != "mp3_in_parquet" and r["source"] != "broadcast24"]
    broken = []
    def one(r):
        dest = local_path(r, cache)
        if dest.exists() and dest.stat().st_size > 1000:
            return True
        local = cache / "src" / r["source_repo"].replace("/", "__")
        for attempt in range(3):
            try:
                p = hf_hub_download(r["source_repo"], r["source_path"], repo_type="dataset", local_dir=str(local))
                to_wav24k(Path(p), dest); return True
            except HfHubHTTPError as e:
                if getattr(e.response, "status_code", None) == 429:
                    print("rate limit; sleeping 300 s", flush=True); time.sleep(300); continue
                if getattr(e.response, "status_code", None) == 404:
                    break
                time.sleep(2 + 3 * attempt)
            except Exception:
                time.sleep(2 + 3 * attempt)
        return False
    with ThreadPoolExecutor(workers) as pool:
        for k, (r, ok) in enumerate(zip(todo, pool.map(one, todo)), 1):
            if not ok: broken.append(r)
            if k % 1000 == 0: print(f"files: {k}/{len(todo)} done", flush=True)
    if broken:
        print(f"skipped {len(broken)} files that would not download, e.g. {[b['source_path'] for b in broken[:3]]}", flush=True)
        ids = {b["id"] for b in broken}
        rows[:] = [r for r in rows if r["id"] not in ids]
    print(f"file-backed clips ready: {len(todo) - len(broken)}", flush=True)


def fetch_broadcast(rows, cache: Path, workers: int):
    """broadcast24: download the ~30 min source recordings and cut each clip locally (start/end are in `note`)."""
    import re
    from huggingface_hub import snapshot_download
    rs = [r for r in rows if r["source"] == "broadcast24"]
    if not rs: return
    repo = rs[0]["source_repo"]
    recs = sorted({r["recording_id"] for r in rs})
    local = cache / "src" / repo.replace("/", "__")
    patterns = [f"yiddish24/source_audio/{rec}.wav" for rec in recs]
    for attempt in range(60):
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(local), allow_patterns=patterns, max_workers=4); break
        except Exception as e:
            print(f"{repo} sources: paused by {str(e).splitlines()[0][:120]}; sleeping 300 s", flush=True); time.sleep(300)
    def cut(r):
        dest = local_path(r, cache)
        if dest.exists() and dest.stat().st_size > 1000: return
        m = re.search(r"start=([0-9.]+) end=([0-9.]+)", r["note"]); s, e = float(m.group(1)), float(m.group(2))
        dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix(".tmp.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", str(local / f"yiddish24/source_audio/{r['recording_id']}.wav"),
                        "-t", f"{e - s:.3f}", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)], check=True); tmp.replace(dest)
    with ThreadPoolExecutor(workers) as pool: list(pool.map(cut, rs))
    print(f"broadcast24: {len(rs)} clips cut from {len(recs)} recordings", flush=True)


def fetch_parquet(rows, cache: Path):
    """Extract mp3 bytes for parquet-backed rows, shard by shard, and convert."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    by_shard = defaultdict(list)
    for r in rows:
        if r["codec"] == "mp3_in_parquet":
            by_shard[(r["source_repo"], r["source_path"])].append(r)
    for (repo, shard), rs in by_shard.items():
        outdir = cache / "wav24k" / repo.replace("/", "__") / Path(shard).stem
        if all((outdir / f"{r['row_index']}.wav").exists() for r in rs):
            continue
        local = hf_hub_download(repo, shard, repo_type="dataset", cache_dir=str(cache / "hub"))
        t = pq.read_table(local, columns=["audio"])
        audio = t.column("audio").to_pylist()
        outdir.mkdir(parents=True, exist_ok=True)
        for r in rs:
            dest = outdir / f"{r['row_index']}.wav"
            if dest.exists(): continue
            tmp = dest.with_suffix(".mp3"); tmp.write_bytes(audio[int(r["row_index"])]["bytes"]); to_wav24k(tmp, dest); tmp.unlink()
        print(f"{repo}/{shard}: {len(rs)} clips ready", flush=True)


def local_path(r, cache: Path) -> Path:
    if r["codec"] == "mp3_in_parquet":
        return cache / "wav24k" / r["source_repo"].replace("/", "__") / Path(r["source_path"]).stem / f"{r['row_index']}.wav"
    return cache / "wav24k" / r["source_repo"].replace("/", "__") / Path(r["source_path"]).with_suffix(".wav")


def seconds(p: Path) -> float:
    i = sf.info(str(p)); return i.frames / i.samplerate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("manifest/manifest.parquet"), help=".parquet or .csv")
    ap.add_argument("--cache", type=Path, default=Path("/workspace/mix-cache"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/vibevoice-data/mix_v1"))
    ap.add_argument("--sources", default="teef_windows,studio,hasidic24,crowd_recital,crowd_whatsapp,broadcast24")
    ap.add_argument("--min-quality", type=float, default=0.9, help="Drop crowd_recital/crowd_whatsapp rows below this alignment quality_score")
    ap.add_argument("--chain-below", type=float, default=15.0, help="Chain single-speaker rows shorter than this")
    ap.add_argument("--chain-min", type=float, default=20.0); ap.add_argument("--chain-max", type=float, default=40.0)
    ap.add_argument("--repeat", default="teef_windows=2", help="source=n repeats in the training split")
    ap.add_argument("--cap-per-speaker", type=int, default=3000, help="Max rows per speaker")
    ap.add_argument("--cap-hours-per-speaker", type=float, default=15.0, help="Max audio hours per speaker (uses manifest durations)")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); rng = random.Random(a.seed)
    want = set(a.sources.split(","))
    if a.manifest.suffix == ".parquet":
        import pyarrow.parquet as pq
        raw = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in pq.read_table(a.manifest).to_pylist()]
    else:
        raw = list(csv.DictReader(a.manifest.open(encoding="utf-8")))
    QUALITY_FILTERED = {"crowd_recital", "crowd_whatsapp"}   # alignment scores; other sources' scores mean different things
    rows = [r for r in raw
            if r["source"] in want and r["tts_ok"] == "True" and r["split"] != "test"
            and (r["source"] not in QUALITY_FILTERED or float(r["quality_score"] or 1.0) >= a.min_quality)]
    by_spk = defaultdict(list)
    for r in rows: by_spk[r["speaker"]].append(r)
    for spk, rs in by_spk.items():
        if rs and rs[0]["source"] == "broadcast24":
            # take whole recordings so the clips can be cut from a few source files instead of thousands of downloads
            by_rec = defaultdict(list)
            for r in rs: by_rec[r["recording_id"]].append(r)
            recs = sorted(by_rec); rng.shuffle(recs)
            total, keep = 0.0, []
            for rec in recs:
                d = sum(float(r["duration"] or 0) for r in by_rec[rec])
                if total + d > a.cap_hours_per_speaker * 3600: continue
                keep.extend(by_rec[rec]); total += d
            rs[:] = keep[: a.cap_per_speaker * 10]
            continue
        rng.shuffle(rs); del rs[a.cap_per_speaker:]
        total, keep = 0.0, []
        for r in rs:
            d = float(r["duration"] or 0)
            if total + d > a.cap_hours_per_speaker * 3600: continue
            keep.append(r); total += d
        rs[:] = keep
    rows = [r for rs in by_spk.values() for r in rs]
    print(f"selected {len(rows)} rows from {len(by_spk)} speakers", flush=True)
    if a.dry_run: return
    a.cache.mkdir(parents=True, exist_ok=True)
    fetch_files(rows, a.cache, a.workers); fetch_broadcast(rows, a.cache, a.workers); fetch_parquet(rows, a.cache)
    by_spk = defaultdict(list)
    for r in rows: by_spk[r["speaker"]].append(r)
    for r in rows:
        r["local"] = local_path(r, a.cache); r["dur"] = seconds(r["local"])

    repeats = dict(kv.split("=") for kv in a.repeat.split(",") if kv)
    out_rows = {"train": [], "validation": []}
    chain_dir = a.out / "chains"; chain_dir.mkdir(parents=True, exist_ok=True)
    for spk, rs in by_spk.items():
        prompt_pool = [r for r in rs if 3.0 <= r["dur"] <= 12.0] or rs
        def prompt(exclude):
            c = [q for q in prompt_pool if q["id"] not in exclude]; return str((rng.choice(c) if c else prompt_pool[0])["local"])
        for split in ("train", "validation"):
            part = [r for r in rs if r["split"] == split]
            long_rows = [r for r in part if r["dur"] >= a.chain_below or int(r["num_speakers"]) > 1]
            short = [r for r in part if r not in long_rows]
            for r in long_rows:
                text = r["text"] if r["text"].lstrip().startswith("Speaker") else "\n".join(f"Speaker 0: {l}" for l in r["text"].split("\n") if l.strip())
                vp = [prompt({r["id"]})] if int(r["num_speakers"]) == 1 else None
                out_rows[split].append(dict(id=r["id"], source=r["source"], speaker=spk, text=text, audio=str(r["local"]), voice_prompts=vp, duration=round(r["dur"], 3), num_speakers=int(r["num_speakers"])))
            rng.shuffle(short); i = 0
            while i < len(short):
                target = rng.uniform(a.chain_min, a.chain_max); group, total = [], 0.0
                while i < len(short) and (not group or total + short[i]["dur"] <= target):
                    group.append(short[i]); total += short[i]["dur"] + 0.55; i += 1
                if len(group) == 1 and i < len(short): group.append(short[i]); i += 1
                pauses = [rng.uniform(0.3, 0.8) for _ in group[:-1]]
                cid = f"chain_{spk}_{len(out_rows[split]):06d}"; dest = chain_dir / f"{cid}.wav"
                if not dest.exists():
                    parts = []
                    for k, g in enumerate(group):
                        x, sr = sf.read(str(g["local"]), dtype="float32"); x = x.mean(axis=1) if x.ndim > 1 else x
                        parts.append(x)
                        if k < len(pauses): parts.append(np.zeros(int(pauses[k] * 24000), dtype="float32"))
                    sf.write(str(dest), np.concatenate(parts), 24000, subtype="PCM_16")
                out_rows[split].append(dict(id=cid, source=group[0]["source"] + "_chain", speaker=spk,
                                            text="\n".join(f"Speaker 0: {g['text'].replace(chr(10), ' ')}" for g in group), audio=str(dest),
                                            voice_prompts=[prompt({g["id"] for g in group})], duration=round(sum(g["dur"] for g in group) + sum(pauses), 3), num_speakers=1))
    train = []
    for r in out_rows["train"]:
        n = int(repeats.get(r["source"], 1)); train.append(r)
        train.extend(dict(r, id=f"{r['id']}_r{k}") for k in range(1, n))
    rng.shuffle(train)
    for split, data in (("train", train), ("validation", out_rows["validation"])):
        with (a.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)
    summary = dict(train_rows=len(train), train_hours_incl_repeats=hours(train), validation_rows=len(out_rows["validation"]),
                   validation_hours=hours(out_rows["validation"]), speakers=len(by_spk),
                   by_source={s: sum(r["source"].startswith(s) for r in train) for s in want}, out=str(a.out))
    (a.out / "preparation.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
