#!/usr/bin/env python3
"""Score a render with the Yiddish Ear (Yiddish-AI/xeus-yi-ear: speech -> Hasidic Yiddish phones, same inventory as the
engine). Phone error rate of what the Ear HEARD against what we FED the model (stress stripped) is a direct pronunciation
score; Whisper only gives words. The Ear collapses on clips longer than ~4 s, so audio is cut into <=3.5 s pieces at quiet
points and the phones are joined.

    python render/ear_score.py --script dialogue/podcast_x.json out.wav [more.wav ...]     # PER vs fed phonemes
    python render/ear_score.py --script dialogue/y.json --real real.wav gen.wav              # also PER of gen vs real

Needs an HF token with access to the Space (~/.cache/huggingface/token) and `gradio_client`.

RESULT 2026-09-17: on our material the Ear is too noisy to judge renders. A REAL Yiddish 24 narrator scored PER 0.475
against the engine's phonemes for his own words; our renders scored 0.49-0.64; v1 vs v4 of a script differed by 0.01. It
was trained on one host's voice and collapses on clips over ~4 s. Kept for when the Ear is retrained on more speakers;
until then the judges are listeners (and Whisper for completeness only)."""
import argparse, difflib, json, os, re, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data"))

PHONES = ["aː", "aj", "ej", "ɔj", "oʊ", "ʦ", "ʧ", "ʤ", "a", "ɛ", "ə", "i", "u", "ɔ", "b", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "x", "ʃ", "ʒ", "ŋ", "g"]


def tokenize(ipa):
    s = re.sub(r"[ˈ\s\.,!?:;\"'()\-]", "", ipa).replace("g", "ɡ"); out = []; i = 0
    while i < len(s):
        for p in PHONES:
            if s.startswith(p, i): out.append(p); i += len(p); break
        else: i += 1
    return out


def per(ref, hyp):
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return errs / max(1, len(ref))


def pieces(wav, max_s=3.5):
    import librosa, numpy as np, soundfile as sf
    x, _ = librosa.load(wav, sr=16000, mono=True)
    hop = 160; rms = librosa.feature.rms(y=x, frame_length=400, hop_length=hop)[0]
    cuts = [0]
    while len(x) - cuts[-1] > 16000 * max_s:
        target = cuts[-1] + int(16000 * 3.0); lo, hi = (target - 16000) // hop, (target + int(16000 * 0.5)) // hop
        cuts.append(int((lo + np.argmin(rms[lo:hi])) * hop))
    cuts.append(len(x)); tmp = tempfile.mkdtemp(); out = []
    for k, (i, j) in enumerate(zip(cuts, cuts[1:])):
        if j - i < 1600: continue
        p = Path(tmp) / f"{k:03d}.wav"; sf.write(str(p), x[i:j], 16000); out.append(str(p))
    return out


class Ear:
    def __init__(self, blank_penalty=1.5):
        from gradio_client import Client
        tok = os.environ.get("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").read_text().strip()
        self.c = Client("Yiddish-AI/xeus-yi-ear", token=tok, verbose=False); self.bp = blank_penalty

    def hear(self, wav):
        from gradio_client import handle_file
        return "".join(self.c.predict(audio=handle_file(p), transcript="", decoder="greedy", blank_penalty=self.bp, beam=8, do_snap=False, api_name="/hear")[0].strip() for p in pieces(wav))


def fed_phonemes(script_path):
    from phonemize import Phonemizer
    ph = Phonemizer()
    d = json.loads(Path(script_path).read_text(encoding="utf-8")) if script_path.endswith(".json") else None
    text = " ".join(t["text"] for t in d["turns"]) if d else Path(script_path).read_text(encoding="utf-8").replace("\n", " ")
    return ph.sentence(text)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--script", required=True); ap.add_argument("--real"); ap.add_argument("wavs", nargs="+")
    ap.add_argument("--blank-penalty", type=float, default=1.5); a = ap.parse_args()
    fed = fed_phonemes(a.script); ref = tokenize(fed); ear = Ear(a.blank_penalty)
    print(f"fed ({len(ref)} phones): {re.sub(chr(712), '', fed)[:160]}...")
    real_tok = None
    if a.real:
        heard = ear.hear(a.real); real_tok = tokenize(heard)
        print(f"{'REAL ' + Path(a.real).name:44s} PER vs fed {per(ref, real_tok):.3f}  | {heard[:120]}")
    for w in a.wavs:
        heard = ear.hear(w); hyp = tokenize(heard)
        extra = f"  PER vs real {per(real_tok, hyp):.3f}" if real_tok else ""
        print(f"{Path(w).name[:44]:44s} PER vs fed {per(ref, hyp):.3f}{extra}  | {heard[:120]}")


if __name__ == "__main__":
    main()
