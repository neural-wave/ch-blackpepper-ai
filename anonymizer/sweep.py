"""Ablation: relevance signal sets x keep_symptoms, scored against the pseudo ground truth.

python -m anonymizer.sweep --split eval --out outputs/sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .data import load_encounters, save_jsonl
from .evaluate_gt import evaluate, load_gt
from .pipeline import Anonymizer

SIGNAL_SETS = [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]
TYPES = ["a:QUASI", "a:OCCUPATION", "b:CONDITION", "b:MEDICATION", "c:CONDITION", "c:MEDICATION", "c:FINDING"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="eval")
    p.add_argument("--out", default="outputs/sweep")
    a = p.parse_args(argv)

    encounters = load_encounters(a.split)
    original = {e["encounter_id"]: e for e in encounters}
    gt = load_gt(a.split)
    anon = Anonymizer(Config())     # models loaded once, knobs switched per run
    rows = []
    for signals in SIGNAL_SETS:
        for keep_symptoms in (True, False):
            anon.cfg.keep_symptoms = keep_symptoms
            anon.scorer.signals = set(signals)
            name = f"s{''.join(map(str, signals))}_sym{'on' if keep_symptoms else 'off'}"
            outs, logs = [], []
            for enc in tqdm(encounters, desc=name, file=sys.stderr):
                out, log = anon.anonymize(enc)
                outs.append(out)
                logs.append(log)
            save_jsonl(outs, Path(a.out) / f"{name}.jsonl")
            save_jsonl(logs, Path(a.out) / f"{name}.decisions.jsonl")
            res, _ = evaluate(original, {o["encounter_id"]: o for o in outs}, gt)
            by_type = {t: res["by_type"].get(t, "-").split(" ")[0] for t in TYPES}
            rows.append({"run": name, "signals": "+".join(map(str, signals)), "keep_symptoms": keep_symptoms,
                         **{k: res[k] for k in ("a_removed", "b_removed", "privacy", "c_kept")}, **by_type})
            print(json.dumps(rows[-1]), flush=True)
    Path(a.out, "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
