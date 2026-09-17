#!/usr/bin/env python3
"""Scan the clips referenced by train/validation.jsonl and drop rows whose audio would break training:
missing, unreadable, shorter than --min-seconds, containing non-finite samples, or (near-)silent
(RMS below --min-rms). Voice prompts are checked too; a bad prompt is replaced by another prompt of the
same speaker when one exists, otherwise the row is dropped. Rewrites the JSONL files in place (backup .bak)."""
import argparse, json, shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def check(path: str, min_seconds: float, min_rms: float):
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception as e:
        return f"unreadable: {e.__class__.__name__}"
    if x.ndim > 1: x = x.mean(axis=1)
    if len(x) < min_seconds * sr: return f"too short ({len(x)/sr:.2f}s)"
    if not np.isfinite(x).all(): return "non-finite samples"
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < min_rms: return f"silent (rms {rms:.2e})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("/workspace/vibevoice-data/mix_v1"))
    ap.add_argument("--min-seconds", type=float, default=0.5)
    ap.add_argument("--min-rms", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = {}
    for split in ("train", "validation"):
        rows[split] = [json.loads(l) for l in (a.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    paths = sorted({r["audio"] for rs in rows.values() for r in rs} | {p for rs in rows.values() for r in rs for p in (r.get("voice_prompts") or [])})
    print(f"checking {len(paths)} files", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        verdicts = dict(zip(paths, pool.map(lambda p: check(p, a.min_seconds, a.min_rms), paths)))
    bad = {p: v for p, v in verdicts.items() if v}
    print(f"bad files: {len(bad)}")
    for p, v in list(bad.items())[:15]: print("  ", v, "|", p)
    # replacement prompts per speaker
    good_prompts = defaultdict(set)
    for rs in rows.values():
        for r in rs:
            for p in r.get("voice_prompts") or []:
                if p not in bad: good_prompts[r.get("speaker")].add(p)
    dropped = defaultdict(int)
    for split, rs in rows.items():
        kept = []
        for r in rs:
            if r["audio"] in bad: dropped[f"{split}:{r.get('source')}"] += 1; continue
            vp = r.get("voice_prompts")
            if vp:
                fixed = []
                for p in vp:
                    if p in bad:
                        alt = [q for q in good_prompts[r.get("speaker")] if q != r["audio"]]
                        if not alt: fixed = None; break
                        fixed.append(sorted(alt)[0])
                    else: fixed.append(p)
                if fixed is None: dropped[f"{split}:{r.get('source')}:prompt"] += 1; continue
                r["voice_prompts"] = fixed
            kept.append(r)
        src = a.data / f"{split}.jsonl"; shutil.copy(src, src.with_suffix(".jsonl.bak"))
        src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        print(f"{split}: kept {len(kept)} of {len(rs)}")
    print("dropped:", dict(dropped) or "nothing")


if __name__ == "__main__":
    main()
