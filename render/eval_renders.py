#!/usr/bin/env python3
"""Score rendered podcast wavs against their script with the local Yiddish Whisper: duration, words/second,
similarity of the transcript to the script (0-1), whether the last script line was reached, and the transcript.
Usage: eval_renders.py --script dialogue/x.json out1.wav out2.wav ...   (Mac, MPS)"""
import argparse, difflib, json, re, unicodedata
from pathlib import Path
import librosa, torch

def norm(t):
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if unicodedata.category(c)[0] in "LN" or c == " ")
    for a, b in [("ך", "כ"), ("ם", "מ"), ("ן", "נ"), ("ף", "פ"), ("ץ", "צ"), ("ױ", "וי"), ("ײ", "יי"), ("װ", "וו")]: t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()

ap = argparse.ArgumentParser(); ap.add_argument("--script", type=Path, required=True); ap.add_argument("wavs", nargs="+"); a = ap.parse_args()
d = json.loads(a.script.read_text(encoding="utf-8")); turns = d["turns"] if isinstance(d, dict) else d
script = norm(" ".join(t["text"] for t in turns)); last = norm(turns[-1]["text"]).split()
from transformers import WhisperProcessor, WhisperForConditionalGeneration
M = "ivrit-ai/yi-whisper-large-v3"
proc = WhisperProcessor.from_pretrained(M); model = WhisperForConditionalGeneration.from_pretrained(M, torch_dtype=torch.float32).to("mps").eval()
print(f"{'file':28s} {'sec':>5s} {'w/s':>5s} {'sim':>5s} {'end':>4s}")
for w in a.wavs:
    x, _ = librosa.load(w, sr=16000, mono=True); out = []
    for i in range(0, len(x), 16000 * 30):
        ch = x[i:i + 16000 * 30]
        if len(ch) < 8000: break
        f = proc(ch, sampling_rate=16000, return_tensors="pt").input_features.to("mps")
        with torch.no_grad(): ids = model.generate(f, language="yi", task="transcribe", max_new_tokens=220)
        out.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    hyp = norm(" ".join(out)); sec = len(x) / 16000
    sim = difflib.SequenceMatcher(None, script, hyp).ratio()
    tail = " ".join(hyp.split()[-12:]); reached = difflib.SequenceMatcher(None, " ".join(last[-4:]), tail).find_longest_match(0, len(" ".join(last[-4:])), 0, len(tail)).size >= 6
    print(f"{Path(w).stem:28s} {sec:5.1f} {len(hyp.split())/sec:5.2f} {sim:5.2f} {'yes' if reached else 'no':>4s}")
    print("   ", " ".join(out)[:400].replace("\n", " "))
