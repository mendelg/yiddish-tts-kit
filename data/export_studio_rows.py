#!/usr/bin/env python3
"""Turn the Yiddish-AI/yiddish-tts `default` (studio) subset into VibeVoice rows and merge them with the
Teef Teef podcast windows into one training manifest.

Most studio rows are CHAINS: several clips of the same speaker concatenated with a short pause into a
20-40 s recording, with one "Speaker 0:" line per clip. Single 2-15 s clips taught the model that a
clean, close-miked voice prompt means a short single utterance, so an unseen studio-quality voice made
it stop after one line; chains break that association. A share of single-clip rows is kept for variety.

Voice prompt: another clip of the same speaker (3-12 s). Only label_status == ok rows (WER-0 verified).
The dominant speaker is capped. Podcast windows are repeated `--podcast-repeat` times.
"""
import argparse, csv, json, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studio", type=Path, default=Path("/workspace/yiddish-tts-studio"))
    p.add_argument("--podcast", type=Path, default=Path("/workspace/vibevoice-data/podcast_c0.85_m90_g2.0"))
    p.add_argument("--out", type=Path, default=Path("/workspace/vibevoice-data/combined_studio_v2"))
    p.add_argument("--cap-per-speaker", type=int, default=2500)
    p.add_argument("--single-fraction", type=float, default=0.25, help="Share of clips kept as single-clip rows")
    p.add_argument("--chain-min", type=float, default=20.0)
    p.add_argument("--chain-max", type=float, default=40.0)
    p.add_argument("--pause-min", type=float, default=0.3)
    p.add_argument("--pause-max", type=float, default=0.8)
    p.add_argument("--podcast-repeat", type=int, default=2)
    p.add_argument("--validation-fraction", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    rng = random.Random(a.seed)

    rows = list(csv.DictReader((a.studio / "metadata.csv").open(encoding="utf-8")))
    ok = [r for r in rows if r["label_status"] == "ok" and r["text"].strip() and (a.studio / r["file_name"]).exists()]
    for r in ok:
        r["text"] = " ".join(r["text"].split())
        r["path"] = str((a.studio / r["file_name"]).resolve())
    by_spk = defaultdict(list)
    for r in ok:
        by_spk[r["speaker_id"]].append(r)
    print("studio rows ok:", len(ok), "of", len(rows), "| speakers:", {k: len(v) for k, v in sorted(by_spk.items())}, flush=True)

    # durations + sample rate (needed for prompts and chain lengths)
    def probe(r):
        info = sf.info(r["path"]); return r["file_name"], info.frames / info.samplerate, info.samplerate
    with ThreadPoolExecutor(a.workers) as pool:
        probed = dict((n, (d, s)) for n, d, s in pool.map(probe, ok))
    for r in ok:
        r["dur"], r["sr"] = probed[r["file_name"]]

    chain_dir = a.out / "chains"; chain_dir.mkdir(parents=True, exist_ok=True)
    studio_rows, chain_jobs = [], []
    for spk, items in sorted(by_spk.items()):
        rng.shuffle(items)
        kept = items[: a.cap_per_speaker]
        prompt_pool = [r for r in items if 3.0 <= r["dur"] <= 12.0] or items
        def prompt_for(exclude):
            choices = [q for q in prompt_pool if q["file_name"] not in exclude]
            return (rng.choice(choices) if choices else prompt_pool[0])["path"]
        n_single = int(len(kept) * a.single_fraction)
        singles, to_chain = kept[:n_single], kept[n_single:]
        for r in singles:
            studio_rows.append(dict(id=f"studio_{Path(r['file_name']).stem}", source="yiddish_tts_studio", speaker=spk,
                                    text=f"Speaker 0: {r['text']}", audio=r["path"], voice_prompts=[prompt_for({r['file_name']})],
                                    duration=round(r["dur"], 3), num_speakers=1, num_turns=1))
        # chains: greedy fill to a random target length between chain-min and chain-max
        i = 0
        while i < len(to_chain):
            target = rng.uniform(a.chain_min, a.chain_max)
            group, total = [], 0.0
            while i < len(to_chain) and (not group or total + to_chain[i]["dur"] <= target):
                group.append(to_chain[i]); total += to_chain[i]["dur"] + 0.55; i += 1
            if len(group) == 1 and i < len(to_chain):  # never leave a lone clip as a "chain"
                group.append(to_chain[i]); i += 1
            pauses = [rng.uniform(a.pause_min, a.pause_max) for _ in group[:-1]]
            cid = f"chain_{spk}_{len(chain_jobs):05d}"
            dest = chain_dir / f"{cid}.wav"
            chain_jobs.append((group, pauses, dest))
            studio_rows.append(dict(id=cid, source="yiddish_tts_studio_chain", speaker=spk,
                                    text="\n".join(f"Speaker 0: {r['text']}" for r in group), audio=str(dest.resolve()),
                                    voice_prompts=[prompt_for({r["file_name"] for r in group})],
                                    duration=round(sum(r["dur"] for r in group) + sum(pauses), 3), num_speakers=1, num_turns=len(group)))

    def write_chain(job):
        group, pauses, dest = job
        if dest.exists() and dest.stat().st_size > 1000:
            return
        sr = group[0]["sr"]; parts = []
        for k, r in enumerate(group):
            x, s = sf.read(r["path"], dtype="float32", always_2d=False)
            if x.ndim > 1: x = x.mean(axis=1)
            if s != sr:
                import librosa; x = librosa.resample(x, orig_sr=s, target_sr=sr)
            parts.append(x)
            if k < len(pauses): parts.append(np.zeros(int(pauses[k] * sr), dtype="float32"))
        sf.write(dest, np.concatenate(parts), sr, subtype="PCM_16")
    with ThreadPoolExecutor(a.workers) as pool:
        for k, _ in enumerate(pool.map(write_chain, chain_jobs), 1):
            if k % 500 == 0: print(f"wrote {k}/{len(chain_jobs)} chains", flush=True)
    print(f"chains: {len(chain_jobs)} | single rows: {sum(r['source'] == 'yiddish_tts_studio' for r in studio_rows)}", flush=True)

    rng.shuffle(studio_rows)
    n_val = max(1, int(len(studio_rows) * a.validation_fraction))
    studio_val, studio_train = studio_rows[:n_val], studio_rows[n_val:]

    def read_jsonl(path):
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pod_train = read_jsonl(a.podcast / "train.jsonl"); pod_val = read_jsonl(a.podcast / "validation.jsonl")
    for r in pod_train + pod_val:
        r.setdefault("source", "teef_teef_podcast")
    train = studio_train + [dict(r, id=f"{r['id']}_r{k}") if k else r for k in range(a.podcast_repeat) for r in pod_train]
    rng.shuffle(train)
    val = pod_val + studio_val
    for name, data in (("train", train), ("validation", val)):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = lambda rs: round(sum(float(r.get("duration", 0)) for r in rs) / 3600, 2)
    summary = dict(train_rows=len(train), train_hours_incl_repeats=hours(train),
                   studio_train_rows=len(studio_train), studio_chain_rows=sum(r["source"].endswith("chain") for r in studio_train),
                   studio_hours=hours(studio_train), podcast_rows=len(pod_train), podcast_repeat=a.podcast_repeat,
                   podcast_hours=hours(pod_train), validation_rows=len(val), validation_hours=hours(val),
                   speakers=1 + len(by_spk), out=str(a.out))
    (a.out / "preparation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
