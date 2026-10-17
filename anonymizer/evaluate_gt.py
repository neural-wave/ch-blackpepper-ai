"""Score anonymized output against the pseudo ground truth (pseudo_gt/<split>/*.json).

Each GT span is a distinct string in a field; all its occurrences (case-insensitive, whole word) are counted
in the original and in the anonymized field.
  a_removed  share of personal-data occurrences gone                (privacy, category a)
  b_removed  share of unrelated-clinical occurrences gone           (privacy, category b)
  privacy    a + b pooled                                           (proxy for metric A)
  c_kept     share of code-relevant occurrences still present        (proxy for metric B)

python -m anonymizer.evaluate_gt --anonymized outputs/eval_anonymized.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT
from .data import load_encounters


def count(needle: str, hay: str) -> int:
    return len(re.findall(rf"(?<![\w\[]){re.escape(needle)}(?!\w)", hay, re.IGNORECASE))


def load_gt(split: str) -> dict[str, list[dict]]:
    return {json.loads(p.read_text())["encounter_id"]: json.loads(p.read_text())["spans"]
            for p in sorted((ROOT / "pseudo_gt" / split).glob("*.json"))}


def evaluate(original: dict, anonymized: dict, gt: dict) -> dict:
    tot = defaultdict(lambda: [0.0, 0])      # key -> [successes, occurrences]
    misses = defaultdict(Counter)
    for eid, spans in gt.items():
        if eid not in anonymized:
            continue
        orig, out = original[eid], anonymized[eid]
        ds = orig.get("dataset", "?")
        for s in spans:
            n_orig = count(s["text"], orig[s["field"]])
            if n_orig == 0:
                continue
            n_out = min(count(s["text"], out[s["field"]]), n_orig)
            cat = s["category"]
            ok = (n_orig - n_out) if cat in "ab" else n_out
            for key in (cat, f"{cat}:{s['type']}", f"{cat}@{ds}", "privacy" if cat in "ab" else "utility"):
                tot[key][0] += ok
                tot[key][1] += n_orig
            if ok < n_orig:
                misses[f"{cat}:{s['type']}"][f"{s['text'].lower()} ({eid})"] += n_orig - ok

    def rate(k):
        h, t = tot.get(k, (0, 0))
        return round(h / t, 4) if t else None

    res = {
        "a_removed": rate("a"), "b_removed": rate("b"), "privacy": rate("privacy"), "c_kept": rate("utility"),
        "by_type": {k: f"{rate(k)} ({int(tot[k][1])})" for k in sorted(tot) if ":" in k},
        "by_dataset": {k: rate(k) for k in sorted(tot) if "@" in k},
        "occurrences": {k: tot[k][1] for k in ("a", "b", "c")},
        "encounters": len([e for e in gt if e in anonymized]),
    }
    return res, misses


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", default="eval")
    p.add_argument("--anonymized", required=True)
    p.add_argument("--gt", default="eval", help="pseudo_gt subfolder")
    p.add_argument("--misses", type=int, default=0, help="print top N misses per type")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    original = {e["encounter_id"]: e for e in load_encounters(a.original)}
    anonymized = {e["encounter_id"]: e for e in load_encounters(a.anonymized)}
    res, misses = evaluate(original, anonymized, load_gt(a.gt))
    print(json.dumps(res, indent=2))
    if a.misses:
        for k in sorted(misses):
            print(f"\n{k}: " + "; ".join(f"{t} x{n}" for t, n in misses[k].most_common(a.misses)))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
