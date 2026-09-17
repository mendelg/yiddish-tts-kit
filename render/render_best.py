#!/usr/bin/env python3
"""Render a script at several seeds until one is COMPLETE (Whisper word recall >= --min-recall and the last line
reached), then write a slowed twin. This is a completeness gate, not a quality judge: Whisper reads straight through
doubled voices, background noise and odd prosody (on bungalow it preferred the glitchy seed over the one listeners
chose), so always listen to the winner and keep the other candidates (<out>_seedN.wav) as alternatives.

    python render/render_best.py --script dialogue/podcast_kapores.json \
        --voices A=references/google/Puck_synth.wav B=references/google/Sadaltager_synth.wav \
        --checkpoint outputs/vibevoice/ipa-final --out out/kapores.wav [--seeds 3,7,42] [--slow 0.90] [--min-recall 0.8]

Every candidate is kept as <out>_seedN.wav; the first complete one is copied to <out> (and <out>_slow90.wav); if none
passes, the highest recall wins. Passes --cfg-scale, --steps and --phonemize through to render_yiddish_podcast.py.
For doubled voices / noise try --steps 20 (cleaner diffusion) or another seed."""
import argparse, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_renders import Transcriber  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True); ap.add_argument("--voices", nargs="+", required=True)
    ap.add_argument("--checkpoint"); ap.add_argument("--model", default="vibevoice/VibeVoice-1.5B")
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--seeds", default="3,7,42")
    ap.add_argument("--cfg-scale", default="1.5"); ap.add_argument("--steps", default="10")
    ap.add_argument("--phonemize", action="store_true"); ap.add_argument("--slow", type=float, default=0.90, help="atempo for the slowed twin; 0 = none")
    ap.add_argument("--min-recall", type=float, default=0.65, help="Stop trying seeds once a candidate reaches this word recall")
    a = ap.parse_args()
    tr = Transcriber(); results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        cand = a.out.with_name(f"{a.out.stem}_seed{seed}.wav")
        cmd = [sys.executable, str(HERE / "render_yiddish_podcast.py"), "--script", a.script, "--voices", *a.voices, "--model", a.model,
               "--cfg-scale", a.cfg_scale, "--steps", a.steps, "--seed", str(seed), "--out", str(cand)]
        if a.checkpoint: cmd += ["--checkpoint", a.checkpoint]
        if a.phonemize: cmd.append("--phonemize")
        p = subprocess.run(cmd, capture_output=True, text=True)
        for line in p.stderr.splitlines():
            if "unsure about" in line or "WARNING: the engine" in line: print(line, file=sys.stderr)
        if p.returncode != 0: print(p.stderr[-2000:], file=sys.stderr); raise SystemExit(f"render failed for seed {seed}")
        r = tr.score(str(cand), a.script); r["seed"] = seed; r["path"] = cand; results.append(r)
        print(f"seed {seed:3d}: {r['sec']:5.1f} s  recall {r['recall']:.2f}  sim {r['sim']:.2f}  end {'yes' if r['reached'] else 'no'}", flush=True)
        if r["recall"] >= a.min_recall and r["reached"]: break
    complete = [r for r in results if r["recall"] >= a.min_recall and r["reached"]]
    best = complete[0] if complete else max(results, key=lambda r: r["recall"])
    if not complete: print("WARNING: no candidate passed the completeness gate; keeping the highest recall. Try more seeds or --chunk-turns.", file=sys.stderr)
    shutil.copyfile(best["path"], a.out); print(f"best: seed {best['seed']} -> {a.out}")
    if a.slow:
        slow = a.out.with_name(f"{a.out.stem}_slow{int(round(a.slow * 100))}.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(a.out), "-filter:a", f"atempo={a.slow}", str(slow)], check=True)
        print(f"slowed: {slow}")


if __name__ == "__main__":
    main()
