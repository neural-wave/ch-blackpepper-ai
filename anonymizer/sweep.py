"""Ablations scored against the pseudo ground truth.

  signals     relevance signal sets x keep_symptoms (saves outputs + decision logs per run)
  thresholds  t_keep x t_remove at the current defaults (metrics only)

python -m anonymizer.sweep --split train --out outputs/sweep_train
python -m anonymizer.sweep --grid thresholds --split train --out outputs/thresholds_train
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
# Relevance scores sit on steps (chapter 0.6, block 0.9, category 0.95, lexical 1.0) above a continuous SapBERT
# range (< ~0.52): t_keep only matters between steps, t_remove inside the continuous range.
T_KEEP = [0.55, 0.6, 0.72, 0.9, 0.93, 0.97, 1.0]
T_REMOVE = [0.3, 0.4, 0.5, 0.6]
TYPES = ["a:QUASI", "a:OCCUPATION", "b:CONDITION", "b:FINDING", "b:MEDICATION", "b:PROCEDURE",
         "c:CONDITION", "c:FINDING", "c:MEDICATION", "c:PROCEDURE"]


def _run(anon, encounters, original, gt, name, out_dir=None, **meta) -> dict:
    outs, logs = [], []
    for enc in tqdm(encounters, desc=name, file=sys.stderr):
        out, log = anon.anonymize(enc)
        outs.append(out)
        logs.append(log)
    if out_dir:
        save_jsonl(outs, Path(out_dir) / f"{name}.jsonl")
        save_jsonl(logs, Path(out_dir) / f"{name}.decisions.jsonl")
    res, _ = evaluate(original, {o["encounter_id"]: o for o in outs}, gt)
    by_type = {t: res["by_type"].get(t, "-").split(" ")[0] for t in TYPES}
    row = {"run": name, **meta, **{k: res[k] for k in ("a_removed", "b_removed", "privacy", "c_kept")}, **by_type}
    print(json.dumps(row), flush=True)
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=["signals", "thresholds"], default="signals")
    p.add_argument("--split", default="eval")
    p.add_argument("--out", default="outputs/sweep")
    p.add_argument("--t-keep", default=",".join(map(str, T_KEEP)))
    p.add_argument("--t-remove", default=",".join(map(str, T_REMOVE)))
    a = p.parse_args(argv)

    encounters = load_encounters(a.split)
    original = {e["encounter_id"]: e for e in encounters}
    gt = load_gt(a.split)
    anon = Anonymizer(Config())     # models loaded once, knobs switched per run
    rows = []
    if a.grid == "signals":
        for signals in SIGNAL_SETS:
            for keep_symptoms in (True, False):
                anon.cfg.keep_symptoms = keep_symptoms
                anon.scorer.signals = set(signals)
                name = f"s{''.join(map(str, signals))}_sym{'on' if keep_symptoms else 'off'}"
                rows.append(_run(anon, encounters, original, gt, name, a.out, signals="+".join(map(str, signals)),
                                 keep_symptoms=keep_symptoms))
    else:
        for tk in map(float, a.t_keep.split(",")):
            for tr in map(float, a.t_remove.split(",")):
                if tr > tk:
                    continue
                anon.cfg.t_keep, anon.cfg.t_remove = tk, tr
                rows.append(_run(anon, encounters, original, gt, f"tk{tk}_tr{tr}", t_keep=tk, t_remove=tr))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
