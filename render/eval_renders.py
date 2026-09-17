#!/usr/bin/env python3
"""Score rendered podcast wavs against their script with the local Yiddish Whisper (evidence only, Whisper mishears
names): duration, words/second, similarity of the transcript to the script (0-1), word recall, whether the last
script line was reached, and the transcript.
Usage: eval_renders.py --script dialogue/x.json out1.wav out2.wav ...   (Mac, MPS)
Importable: Transcriber().score(wav, script_json) -> dict(sec, wps, sim, recall, reached, hyp)."""
import argparse, difflib, json, re, unicodedata
from collections import Counter
from pathlib import Path

M = "ivrit-ai/yi-whisper-large-v3"


def norm(t):
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if unicodedata.category(c)[0] in "LN" or c == " ")
    for a, b in [("ך", "כ"), ("ם", "מ"), ("ן", "נ"), ("ף", "פ"), ("ץ", "צ"), ("ױ", "וי"), ("ײ", "יי"), ("װ", "וו")]: t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def load_script(path):
    d = json.loads(Path(path).read_text(encoding="utf-8")); turns = d["turns"] if isinstance(d, dict) else d
    return [t["text"] for t in turns]


class Transcriber:
    def __init__(self, device=None):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.proc = WhisperProcessor.from_pretrained(M)
        self.model = WhisperForConditionalGeneration.from_pretrained(M, torch_dtype=torch.float32).to(self.device).eval()

    def transcribe(self, wav):
        import librosa, numpy as np
        x, _ = librosa.load(wav, sr=16000, mono=True); out = []
        # <=30 s windows cut at the quietest point near each boundary: a fixed 30.0 s cut lands mid-word and Whisper
        # drops the words around it, which looks like a skipped turn.
        hop = 1600; rms = librosa.feature.rms(y=x, frame_length=3200, hop_length=hop)[0]
        cuts = [0]
        while len(x) - cuts[-1] > 16000 * 30:
            target = cuts[-1] + 16000 * 27; lo, hi = (target - 16000 * 5) // hop, (target + 16000 * 3) // hop
            cuts.append(int((lo + np.argmin(rms[lo:hi])) * hop))
        cuts.append(len(x))
        for i, j in zip(cuts, cuts[1:]):
            ch = x[i:j]
            if len(ch) < 8000: break
            f = self.proc(ch, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            with self.torch.no_grad(): ids = self.model.generate(f, language="yi", task="transcribe", max_new_tokens=220)
            out.append(self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return " ".join(out).replace("\n", " "), len(x) / 16000

    def score(self, wav, script_path):
        turns = load_script(script_path); script = norm(" ".join(turns)); last = norm(turns[-1]).split()
        raw, sec = self.transcribe(wav); hyp = norm(raw)
        sim = difflib.SequenceMatcher(None, script, hyp).ratio()
        sw, hw = Counter(script.split()), Counter(hyp.split())
        recall = sum(min(c, hw[w]) for w, c in sw.items()) / max(1, sum(sw.values()))
        tail = " ".join(hyp.split()[-12:]); ref = " ".join(last[-4:])
        reached = difflib.SequenceMatcher(None, ref, tail).find_longest_match(0, len(ref), 0, len(tail)).size >= 6
        return dict(sec=sec, wps=len(hyp.split()) / max(sec, 0.1), sim=sim, recall=recall, reached=reached, hyp=raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--script", type=Path, required=True); ap.add_argument("wavs", nargs="+"); a = ap.parse_args()
    tr = Transcriber()
    print(f"{'file':40s} {'sec':>5s} {'w/s':>5s} {'sim':>5s} {'recall':>6s} {'end':>4s}")
    for w in a.wavs:
        r = tr.score(w, a.script)
        print(f"{Path(w).stem[:40]:40s} {r['sec']:5.1f} {r['wps']:5.2f} {r['sim']:5.2f} {r['recall']:6.2f} {'yes' if r['reached'] else 'no':>4s}")
        print("   ", r["hyp"])
