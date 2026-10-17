"""Stage 1: run the pipeline once per split (Anonymizer.prepare), save every detected clinical span with its
features (anonymizer.decider.span_features) and pseudo-GT label, and the prepared pipeline state.

Experiments replay only Anonymizer.finalize on the saved state, so a decider / theta costs seconds.

python -m experiments.bc_llm.features    # -> outputs/bc_llm/features_{split}.json, prepared_{split}.pkl
"""

from __future__ import annotations

import argparse
import pickle

from tqdm import tqdm

from anonymizer.config import Config
from anonymizer.data import load_encounters
from anonymizer.decider import EMBEDDING_FEATURES, gt_labels  # noqa: F401  (EMBEDDING_FEATURES used by run.py)
from anonymizer.pipeline import Anonymizer

from .common import OUT, save_json

# decider=True only so prepare() computes the span features; the decision itself is never taken here.
CFG = Config(decider=True)


def prepare_split(split: str, limit: int | None = None) -> list[dict]:
    anon = Anonymizer(CFG)
    return [anon.prepare(enc) for enc in tqdm(load_encounters(split)[:limit], desc=f"pipeline {split}")]


def load_states(split: str, limit: int | None = None) -> list[dict]:
    """Prepared states from disk; built once (one pipeline pass) if missing."""
    path = OUT / f"prepared_{split}.pkl"
    if limit is None and path.exists():
        return pickle.loads(path.read_bytes())
    states = prepare_split(split, limit)
    if limit is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(states))
    return states


_FINALIZER = None


def finalize_split(states: list[dict], keep: dict[str, bool] | None = None) -> list[dict]:
    global _FINALIZER
    if _FINALIZER is None:
        _FINALIZER = Anonymizer(CFG)
    return [_FINALIZER.finalize(st, keep)[0] for st in states]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--limit", type=int, default=None, help="first N encounters (smoke test, nothing cached)")
    a = p.parse_args(argv)
    for split in a.splits.split(","):
        states = prepare_split(split, a.limit)
        rows = [r for st in states for r in st["features"]]
        gt_labels(split, rows)
        save_json(rows, OUT / f"features_{split}.json")
        if a.limit is None:
            (OUT / f"prepared_{split}.pkl").write_bytes(pickle.dumps(states))
        print(f"{split}: {len(rows)} clinical spans, {sum(r['y'] is not None for r in rows)} labelled "
              f"-> {OUT / f'features_{split}.json'}, prepared_{split}.pkl")


if __name__ == "__main__":
    main()
