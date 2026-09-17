#!/usr/bin/env python3
"""Print audio and text statistics for given training rows (by id), to find what makes a batch produce
non-finite gradients. CPU only. Usage: inspect_rows.py --data DIR id1 id2 ..."""
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf

def stats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    if len(x) == 0: return "EMPTY"
    return (f"{len(x)/sr:6.1f}s sr={sr} rms={np.sqrt(np.mean(x**2)):.4f} peak={np.abs(x).max():.3f} "
            f"clipped={(np.abs(x) >= 0.999).mean()*100:.2f}% dc={x.mean():+.4f} finite={bool(np.isfinite(x).all())} "
            f"silent_frac={(np.abs(x) < 1e-4).mean()*100:.1f}%")

ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=Path("/workspace/vibevoice-data/mix_v1")); ap.add_argument("ids", nargs="+")
a = ap.parse_args()
want = set(a.ids); rows = {}
for split in ("train", "validation"):
    for l in (a.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["id"] in want: rows[r["id"]] = r
for rid in a.ids:
    r = rows.get(rid)
    if not r: print(f"\n== {rid}: NOT FOUND"); continue
    print(f"\n== {rid}  source={r['source']} speaker={r.get('speaker')} speakers={r.get('num_speakers')} dur={r.get('duration')}")
    print("   audio :", stats(r["audio"]))
    for p in r.get("voice_prompts") or []: print("   prompt:", stats(p), "|", Path(p).name)
    lines = r["text"].split("\n"); words = sum(len(l.split()) for l in lines)
    print(f"   text  : {len(lines)} lines, {words} words, {len(r['text'])} chars; empty lines={sum(1 for l in lines if len(l.split(':',1)[-1].strip())==0)}")
    odd = sorted({c for c in r["text"] if not (c.isspace() or '֐' <= c <= '׿' or c in ".,!?;:'\"-()" or c.isalnum())})
    if odd: print("   odd chars:", odd)
    print("   first :", lines[0][:120].replace("\n", " "))
