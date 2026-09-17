#!/usr/bin/env python3
"""Render a two-speaker Yiddish script with VibeVoice, base model or fine-tuned adapter.

    python render_yiddish_podcast.py --script dialogue/podcast_tefillin.json \
        --voices A=references/curated/podcast_spk0_2.wav B=references/curated/podcast_spk1_1.wav \
        --checkpoint /workspace/vibevoice-runs/teef_teef_podcast/checkpoint-1000 --out outputs/vv/tefillin.wav

The script is the same JSON the Chatterbox harness uses: a list of {"speaker": name, "text": ...}
(or {"turns": [...]}). Speakers are renumbered 0..N-1 by first appearance and the voices are
passed in that order, matching how the fine-tune rows were built.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch


def load_turns(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("turns") or data.get("script") or data["lines"]
    turns = []
    for t in data:
        if isinstance(t, dict):
            turns.append((str(t.get("speaker") or t.get("role") or "host"), str(t["text"]).strip()))
        else:
            turns.append(("host", str(t).strip()))
    return [(s, x) for s, x in turns if x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", required=True, type=Path)
    p.add_argument("--voices", nargs="+", required=True, help="name=path.wav for each speaker name used in the script")
    p.add_argument("--model", default="vibevoice/VibeVoice-1.5B")
    p.add_argument("--checkpoint", default=None, help="Fine-tune output dir (contains lora/) or a checkpoint-N dir")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cfg-scale", type=float, default=1.3)
    p.add_argument("--steps", type=int, default=10, help="Diffusion inference steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--no-voice-prompts", action="store_true", help="Let the model invent the voices (is_prefill=False)")
    p.add_argument("--dry-run", action="store_true", help="Print the normalized script and voices, load nothing")
    p.add_argument("--phonemize", action="store_true", help="Convert the script to IPA with Phonikud-yi first (for adapters trained with --text-mode ipa)")
    a = p.parse_args()

    from vibevoice.finetune.speakers import normalize_script
    from vibevoice.modular.lora_loading import load_lora_assets
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    voices = dict(v.split("=", 1) for v in a.voices)
    turns = load_turns(a.script)
    names = []
    for s, _ in turns:
        if s not in names:
            names.append(s)
    missing = [n for n in names if n not in voices]
    if missing:
        raise SystemExit(f"No voice given for speaker(s) {missing}; pass --voices name=path.wav")
    raw = "\n".join(f"Speaker {names.index(s) + 1}: {x}" for s, x in turns)
    script, prompts, _ = normalize_script(raw, [voices[n] for n in names])
    if a.phonemize:
        import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
        from phonemize import Phonemizer
        script = Phonemizer()(script)
    print(script)
    for k, path in enumerate(prompts):
        print(f"Speaker {k} voice: {path}")
        if not Path(path).exists():
            raise SystemExit(f"Voice file not found: {path}")
    if a.dry_run:
        return

    torch.manual_seed(a.seed)
    dtype = torch.bfloat16 if a.device == "cuda" else (torch.float16 if a.device == "mps" else torch.float32)
    processor = VibeVoiceProcessor.from_pretrained(a.model)
    kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa")
    if a.device == "cuda":
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(a.model, **kwargs).to(a.device)
    if a.checkpoint:
        report = load_lora_assets(model, a.checkpoint)
        print("Loaded adapter components:", {k: v for k, v in vars(report).items() if v})
    model.eval()
    model.set_ddpm_inference_steps(num_steps=a.steps)

    inputs = processor(text=[script], voice_samples=[prompts], padding=True, return_tensors="pt", return_attention_mask=True)
    inputs = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=None, cfg_scale=a.cfg_scale, tokenizer=processor.tokenizer,
                             generation_config={"do_sample": False}, verbose=True, is_prefill=not a.no_voice_prompts)
    wav = outputs.speech_outputs[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(wav, output_path=str(a.out))
    seconds = wav.shape[-1] / 24000
    print(f"Wrote {a.out} ({seconds:.1f} s of audio in {time.time() - t0:.1f} s)")
    (a.out.with_suffix(".json")).write_text(json.dumps(dict(
        script=script, voices=prompts, model=a.model, checkpoint=a.checkpoint, cfg_scale=a.cfg_scale,
        steps=a.steps, seed=a.seed, seconds=seconds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
