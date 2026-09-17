#!/usr/bin/env python3
"""Yiddish text -> IPA through the Phonikud-yi engine bundle (text -> nikud -> IPA, CPU, onnxruntime).

Point PHONIKUD_YI_BUNDLE at the unzipped `phonikud-yi-engine` directory (from Phonikud-yi's
`src/make_bundle.py`), or at a Phonikud-yi checkout that has its model export in place. Results are cached
in a JSON file so re-running materialize does not re-phonemize 40k rows.

    from phonemize import Phonemizer
    ph = Phonemizer(); ph("מיט א פאר יאר צוריק")  -> 'mit a pˈur jur ʦirˈik'

`Speaker N:` prefixes are preserved; only the spoken text after the colon is converted. Digits, Latin words
and punctuation are passed through unchanged (the engine quarantines them), so keep scripts to Yiddish words.
"""
from __future__ import annotations
import json, os, re, sys, threading
from pathlib import Path

SPEAKER = re.compile(r"^(\s*Speaker\s+\d+\s*:\s*)(.*)$", re.IGNORECASE)


class Phonemizer:
    def __init__(self, bundle: str | None = None, cache_path: str | Path | None = None):
        root = Path(bundle or os.environ.get("PHONIKUD_YI_BUNDLE") or os.environ.get("PHONIKUD_YI_DIR") or "")
        candidates = [root, root / "src", Path.home() / "Documents/work projects/Phonikud-yi/src", Path("/workspace/phonikud-yi-engine")]
        self.src = next((c for c in candidates if (c / "yiddish_labels.py").exists()), None)
        if self.src is None:
            raise RuntimeError("Phonikud-yi engine not found. Unzip phonikud-yi-engine.zip and set PHONIKUD_YI_BUNDLE to that directory.")
        sys.path.insert(0, str(self.src))
        from yiddish_labels import text_to_ipa  # noqa: E402
        self._to_ipa = text_to_ipa
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._lock = threading.Lock(); self._dirty = 0

    def sentence(self, text: str) -> str:
        text = " ".join(text.split())
        if not text: return text
        if text in self.cache: return self.cache[text]
        ipa = self._to_ipa(text)
        with self._lock:
            self.cache[text] = ipa; self._dirty += 1
            if self.cache_path and self._dirty >= 500: self.save()
        return ipa

    def __call__(self, text: str) -> str:
        """Phonemize a script: every line keeps its `Speaker N:` prefix, the rest becomes IPA."""
        out = []
        for line in text.split("\n"):
            m = SPEAKER.match(line)
            out.append(m.group(1) + self.sentence(m.group(2)) if m else self.sentence(line))
        return "\n".join(out)

    def save(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"); self._dirty = 0


if __name__ == "__main__":
    ph = Phonemizer()
    for line in sys.stdin if len(sys.argv) == 1 else sys.argv[1:]:
        print(ph(line.rstrip("\n")))
