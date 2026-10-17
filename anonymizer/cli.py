"""Command line: anonymize a JSONL file.

python -m anonymizer.cli --input eval --output outputs/eval_anonymized.jsonl
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import torch
from tqdm import tqdm

from .config import Config
from .data import load_encounters, save_jsonl
from .pipeline import Anonymizer, no_code_predictions


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="eval", help="train | eval | path to .jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--log", default=None, help="decision log .jsonl (default: <output>.decisions.jsonl)")
    p.add_argument("--no-codes", action="store_true", help="bonus mode: ignore icd10_codes, predict them instead")
    p.add_argument("--t-keep", type=float, default=Config.t_keep)
    p.add_argument("--t-remove", type=float, default=Config.t_remove)
    p.add_argument("--no-ner", action="store_true", help="skip the de-id transformer (rules only)")
    p.add_argument("--signals", default="1,2,3,4", help="relevance signals to use, e.g. 1,2")
    p.add_argument("--no-keep-symptoms", action="store_true", help="score symptoms like any other span")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)

    cfg = Config(t_keep=a.t_keep, t_remove=a.t_remove, use_deid_ner=not a.no_ner, device=a.device,
                 signals=tuple(int(x) for x in a.signals.split(",")), keep_symptoms=not a.no_keep_symptoms)
    encounters = load_encounters(a.input)[:a.limit]
    t0 = time.time()
    anon = Anonymizer(cfg)
    outputs, logs = [], []
    for enc in tqdm(encounters, desc="anonymizing", file=sys.stderr):
        codes = no_code_predictions(anon, enc) if a.no_codes else None
        out, log = anon.anonymize(enc, codes)
        if a.no_codes:
            log["predicted_codes"] = codes
        outputs.append(out)
        logs.append(log)
    elapsed = time.time() - t0

    save_jsonl(outputs, a.output)
    save_jsonl(logs, a.log or str(a.output).replace(".jsonl", "") + ".decisions.jsonl")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    vram = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None
    print(f"{len(outputs)} encounters in {elapsed:.1f}s ({elapsed / max(len(outputs), 1):.2f}s/enc), "
          f"peak RAM {peak_rss:.0f} MB" + (f", peak VRAM {vram:.0f} MB" if vram is not None else ""))


if __name__ == "__main__":
    main()
