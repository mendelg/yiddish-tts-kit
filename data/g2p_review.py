#!/usr/bin/env python3
"""List the words Phonikud-yi is unsure about, for review ("correct phonemes through Fable").

    python data/g2p_review.py scripts/*.json            # podcast scripts (turns[].text)
    python data/g2p_review.py --manifest manifest/manifest.parquet --min-count 20
    python data/g2p_review.py --text "א גוט יום טוב"

Prints one row per distinct MED/LOW-confidence word (frequency-sorted): word, count, confidence, route, reason, the
engine's IPA, the current override (if any), and an example sentence. Paste corrected rows into data/g2p_overrides.tsv
as `word<TAB>ipa`. Words already overridden are skipped unless --all.
"""
from __future__ import annotations
import argparse, collections, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from phonemize import Phonemizer, split_punct, HEBREW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scripts", nargs="*", help="podcast script JSON files")
    ap.add_argument("--manifest"); ap.add_argument("--text")
    ap.add_argument("--min-count", type=int, default=1); ap.add_argument("--all", action="store_true")
    ap.add_argument("--conf", default="MED,LOW", help="which confidence levels to list")
    a = ap.parse_args()
    sents: list[str] = []
    for f in a.scripts: sents += [t["text"] for t in json.load(open(f, encoding="utf-8"))["turns"]]
    if a.text: sents.append(a.text)
    if a.manifest:
        import pyarrow.parquet as pq
        sents += [r or "" for r in pq.read_table(a.manifest, columns=["text"]).column("text").to_pylist()]
    ph = Phonemizer(); detail = ph.token_detail
    cnt: collections.Counter[str] = collections.Counter(); example: dict[str, str] = {}
    for s in sents:
        for t in s.split():
            core = split_punct(t)[1]
            if core and HEBREW.search(core): cnt[core] += 1; example.setdefault(core, s)
    levels = set(a.conf.split(",")); tot = sum(cnt.values()); byconf: collections.Counter[str] = collections.Counter(); rows = []
    for w, c in cnt.most_common():
        d = detail(w); conf = d.get("confidence", "?"); byconf[conf] += c
        if conf in levels and c >= a.min_count and (a.all or w not in ph.overrides):
            rows.append([w, c, conf, d.get("route", ""), d.get("reason", ""), d.get("ipa_primary", ""), ph.overrides.get(w, ""), example[w][:90]])
    print(f"# tokens={tot} unique={len(cnt)} " + " ".join(f"{k}={v}({100*v/max(tot,1):.1f}%)" for k, v in byconf.most_common()), file=sys.stderr)
    print("word\tcount\tconf\troute\treason\tengine_ipa\toverride\texample")
    for r in rows: print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()
