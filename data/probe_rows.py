#!/usr/bin/env python3
"""Run the given rows' target audio and prompts through VibeVoice's acoustic/semantic encoders in bf16 and
report non-finite outputs and activation peaks, to find which clip overflows. Needs ~4 GB of GPU.
Usage: probe_rows.py --data DIR id1 id2 ..."""
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf, torch

ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=Path("/workspace/vibevoice-data/mix_v1"))
ap.add_argument("--model", default="vibevoice/VibeVoice-1.5B"); ap.add_argument("ids", nargs="+"); a = ap.parse_args()
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
model = VibeVoiceForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()
rows = {}
for split in ("train", "validation"):
    for l in (a.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["id"] in set(a.ids): rows[r["id"]] = r

def load(p):
    x, sr = sf.read(p, dtype="float32"); x = x.mean(axis=1) if x.ndim > 1 else x
    assert sr == 24000; return x

def probe(x, label):
    t = torch.tensor(x)[None].to("cuda", torch.bfloat16); m = torch.ones(1, int(np.ceil(len(x) / 3200)), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        feats, conn = model.forward_speech_features(speech_tensors=t, speech_masks=m, return_unmask=True)
        sem = model.model.semantic_tokenizer.encode(t[:, None, :]) if hasattr(model.model.semantic_tokenizer, "encode") else None
    def rep(name, v):
        if v is None: return f"{name}=n/a"
        v = v.float(); return f"{name}: finite={bool(torch.isfinite(v).all())} max={v.abs().max().item():.1f}"
    sem_t = getattr(sem, "mean", sem) if sem is not None else None
    if isinstance(sem_t, (list, tuple)): sem_t = sem_t[0]
    print(f"   {label:7s} {len(x)/24000:5.1f}s peak={np.abs(x).max():.3f} | {rep('acoustic', feats)} | {rep('connector', conn)} | {rep('semantic', sem_t if torch.is_tensor(sem_t) else None)}")

for rid in a.ids:
    r = rows.get(rid)
    if not r: print(f"== {rid}: NOT FOUND"); continue
    print(f"== {rid}")
    try: probe(load(r["audio"]), "target")
    except Exception as e: print("   target: ERROR", e)
    for p in r.get("voice_prompts") or []:
        try: probe(load(p), "prompt")
        except Exception as e: print("   prompt: ERROR", e)
    x = load(r["audio"]); x = x / (np.abs(x).max() + 1e-6) * 0.5
    try: probe(x, "target@0.5")
    except Exception as e: print("   normalized: ERROR", e)
