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
    p.add_argument("--signals", default=",".join(map(str, Config.signals)), help="relevance signals to use, e.g. 1,2")
    p.add_argument("--keep-symptoms", action=argparse.BooleanOptionalAction, default=Config.keep_symptoms,
                   help="keep symptoms/findings regardless of relevance score")
    p.add_argument("--no-smalltalk", action="store_true", help="disable the small-talk unit filter")
    p.add_argument("--smalltalk-threshold", type=float, default=Config.smalltalk_threshold)
    p.add_argument("--a-ideas", default=",".join(map(str, Config.a_ideas)),
                   help="personal-data extensions of study G; '' turns them off")
    p.add_argument("--no-cache", action="store_true", help="recompute model outputs instead of reusing .cache/")
    p.add_argument("--no-decider", action="store_true", help="b/c by relevance threshold only (no LLM)")
    p.add_argument("--llm-batch", type=int, default=Config.llm_batch, help="LLM judge spans per forward pass")
    p.add_argument("--vram-cap-gb", type=float, default=None, help="hard CUDA memory cap; OOM if exceeded")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)

    cfg = Config(t_keep=a.t_keep, t_remove=a.t_remove, use_deid_ner=not a.no_ner, device=a.device,
                 signals=tuple(int(x) for x in a.signals.split(",")), keep_symptoms=a.keep_symptoms,
                 smalltalk=not a.no_smalltalk, smalltalk_threshold=a.smalltalk_threshold, cache=not a.no_cache,
                 a_ideas=tuple(int(x) for x in a.a_ideas.split(",") if x), decider=not a.no_decider,
                 llm_batch=a.llm_batch)
    if cfg.decider:
        from . import decider
        decider.load_bundle(cfg)    # fail now, not after minutes of stages 1-4
    if a.vram_cap_gb and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.vram_cap_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory))
    encounters = load_encounters(a.input)[:a.limit]
    t0 = time.time()
    anon = Anonymizer(cfg)
    codes = [no_code_predictions(anon, enc) for enc in encounters] if a.no_codes else None
    results = anon.anonymize_all(encounters, codes, progress=lambda it: tqdm(it, desc="anonymizing", file=sys.stderr))
    outputs, logs = [o for o, _ in results], [l for _, l in results]
    if a.no_codes:
        for log, c in zip(logs, codes):
            log["predicted_codes"] = c
    elapsed = time.time() - t0

    save_jsonl(outputs, a.output)
    save_jsonl(logs, a.log or str(a.output).replace(".jsonl", "") + ".decisions.jsonl")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    vram = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None
    print(f"{len(outputs)} encounters in {elapsed:.1f}s ({elapsed / max(len(outputs), 1):.2f}s/enc), "
          f"peak RAM {peak_rss:.0f} MB" + (f", peak VRAM {vram:.0f} MB" if vram is not None else ""))


if __name__ == "__main__":
    main()
