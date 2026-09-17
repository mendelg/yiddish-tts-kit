#!/usr/bin/env python3
"""Yiddish text -> IPA through the Phonikud-yi engine bundle (text -> nikud -> IPA, CPU, onnxruntime).

Point PHONIKUD_YI_BUNDLE at the unzipped `phonikud-yi-engine` directory (from Phonikud-yi's
`src/make_bundle.py`), or at a Phonikud-yi checkout that has its model export in place. Results are cached
in a JSON file so re-running materialize does not re-phonemize 40k rows.

    from phonemize import Phonemizer
    ph = Phonemizer(); ph("מיט א פאר יאר צוריק")  -> 'mit a pˈur jur ʦirˈik'

`Speaker N:` prefixes are preserved; only the spoken text after the colon is converted. Digits and Latin words
are DROPPED by the engine (quarantined), so write numbers out in words; `Phonemizer.dropped(text)` lists them.

Corrections: `data/g2p_overrides.tsv` (word or phrase -> IPA) is applied before the engine, longest phrase first,
so the reviewed Heimish readings win over the engine's LOW/MED-confidence guesses. Regenerate the review list with
`data/g2p_review.py`. The JSON cache is invalidated automatically when the overrides file changes.
"""
from __future__ import annotations
import hashlib, json, os, re, sys, threading
from pathlib import Path

SPEAKER = re.compile(r"^(\s*Speaker\s+\d+\s*:\s*)(.*)$", re.IGNORECASE)
PUNCT = "״\"'.,!?;:()[]«»-–—…"
HEBREW = re.compile(r"[\u05d0-\u05ea]")
DEFAULT_OVERRIDES = Path(__file__).with_name("g2p_overrides.tsv")
DEFAULT_VERIFIED = Path(__file__).with_name("g2p_verified.txt")


def load_verified(path: str | Path | None = None) -> set[str]:
    p = Path(path) if path else DEFAULT_VERIFIED
    if not p.exists(): return set()
    return {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}


INVENTORY = set("abdfghjklmnprstvzxʃʒʦʧʤŋɡɛəiuɔˈː ")  # engine's closed phone set (+ ej aj ɔj oʊ built from these)
INVENTORY |= set("eo")  # only as part of ej / oʊ
INVENTORY.add("ʊ")


def load_overrides(path: str | Path | None = None) -> dict[str, str]:
    p = Path(path) if path else DEFAULT_OVERRIDES
    out: dict[str, str] = {}
    if not p.exists(): return out
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line: continue
        k, v = line.split("\t", 1); k = " ".join(k.split()); v = v.strip()
        bad = set(v) - INVENTORY
        if bad or "g" in v:  # 'g' must be the IPA ɡ
            raise ValueError(f"{p}:{n}: {k!r} -> {v!r} uses symbols outside the engine inventory: {sorted(bad | ({'g'} if 'g' in v else set()))}")
        if k in out and out[k] != v: raise ValueError(f"{p}:{n}: duplicate key {k!r} with a different value")
        out[k] = v
    return out


def split_punct(tok: str) -> tuple[str, str, str]:
    """'(שבת,' -> ('(', 'שבת', ',')"""
    i, j = 0, len(tok)
    while i < j and tok[i] in PUNCT: i += 1
    while j > i and tok[j - 1] in PUNCT: j -= 1
    return tok[:i], tok[i:j], tok[j:]


class Phonemizer:
    def __init__(self, bundle: str | None = None, cache_path: str | Path | None = None, overrides: str | Path | None = None):
        root = Path(bundle or os.environ.get("PHONIKUD_YI_BUNDLE") or os.environ.get("PHONIKUD_YI_DIR") or "")
        candidates = [root, root / "src", Path.home() / "Documents/work projects/Phonikud-yi/src", Path("/workspace/phonikud-yi-engine")]
        self.src = next((c for c in candidates if (c / "yiddish_labels.py").exists()), None)
        if self.src is None:
            raise RuntimeError("Phonikud-yi engine not found. Unzip phonikud-yi-engine.zip and set PHONIKUD_YI_BUNDLE to that directory.")
        sys.path.insert(0, str(self.src))
        from yiddish_labels import text_to_ipa, token_detail  # noqa: E402
        self._to_ipa = text_to_ipa; self.token_detail = token_detail
        self.overrides = load_overrides(overrides); self.verified = load_verified()
        self.max_words = max([len(k.split()) for k in self.overrides] + [1])
        self.fingerprint = hashlib.sha1(json.dumps(sorted(self.overrides.items()), ensure_ascii=False).encode()).hexdigest()[:12]
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if data.get("_fingerprint") == self.fingerprint: self.cache = {k: v for k, v in data.items() if k != "_fingerprint"}
        self._lock = threading.Lock(); self._dirty = 0

    def engine(self, text: str) -> str:
        return self._to_ipa(text) if text.strip() else text

    def convert(self, text: str) -> str:
        """Overrides (longest phrase first) + engine for everything in between."""
        # Hyphenated compounds (ערלי-וואטינג) are split so each part can hit the overrides table.
        text = re.sub(r"(?<=[\u05d0-\u05ea])-(?=[\u05d0-\u05ea])", " ", text)
        toks = text.split(); out: list[str] = []; span: list[str] = []; i = 0
        def flush():
            if span: out.append(self.engine(" ".join(span))); span.clear()
        while i < len(toks):
            hit = None
            for n in range(min(self.max_words, len(toks) - i), 0, -1):
                cores = [split_punct(t)[1] for t in toks[i:i + n]]
                key = " ".join(c for c in cores if c)
                if len(cores) == n and all(cores) and key in self.overrides: hit = n; break
            if hit:
                flush()
                pre = split_punct(toks[i])[0]; suf = split_punct(toks[i + hit - 1])[2]
                out.append(pre + self.overrides[" ".join(split_punct(t)[1] for t in toks[i:i + hit])] + suf); i += hit
            else:
                span.append(toks[i]); i += 1
        flush()
        return " ".join(out)

    def unsure(self, text: str) -> list[str]:
        """Words the engine marks MED/LOW that nobody has reviewed yet (not overridden, not in g2p_verified.txt)."""
        seen: dict[str, str] = {}
        for sent in self.sentences(text):
            for tok in re.sub(r"(?<=[\u05d0-\u05ea])-(?=[\u05d0-\u05ea])", " ", sent).split():
                w = split_punct(tok)[1]
                if w and HEBREW.search(w) and w not in seen and w not in self.overrides and w not in self.verified:
                    d = self.token_detail(w)
                    if d.get("confidence") in ("LOW", "MED"): seen[w] = f"{w} -> {d.get('ipa_primary')} [{d.get('confidence')}]"
        return list(seen.values())

    @staticmethod
    def dropped(text: str) -> list[str]:
        """Tokens the engine will silently drop: digits, Latin, anything without a Hebrew letter."""
        out = []
        for line in text.split("\n"):
            m = SPEAKER.match(line); body = m.group(2) if m else line
            out += [t for t in body.split() if split_punct(t)[1] and not HEBREW.search(t)]
        return out

    def sentence(self, text: str) -> str:
        text = " ".join(text.split())
        if not text: return text
        if text in self.cache: return self.cache[text]
        ipa = self.convert(text)
        with self._lock:
            self.cache[text] = ipa; self._dirty += 1
            if self.cache_path and self._dirty >= 500: self.save()
        return ipa

    @staticmethod
    def sentences(text: str) -> list[str]:
        """The spoken parts of a script (speaker prefixes removed), one per line."""
        out = []
        for line in text.split("\n"):
            m = SPEAKER.match(line); out.append(m.group(2) if m else line)
        return out

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
            self.cache_path.write_text(json.dumps({"_fingerprint": self.fingerprint, **self.cache}, ensure_ascii=False), encoding="utf-8"); self._dirty = 0


if __name__ == "__main__":
    ph = Phonemizer()
    for line in sys.stdin if len(sys.argv) == 1 else sys.argv[1:]:
        line = line.rstrip("\n")
        if d := ph.dropped(line): print(f"WARNING dropped by the engine (write out in Yiddish words): {d}", file=sys.stderr)
        print(ph(line))
