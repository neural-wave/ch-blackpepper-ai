"""Stage 2: zero-shot LLM judgement per detected clinical span: P(Yes) that the span (in its sentence) is evidence
for, a cause of, or a treatment of one of the encounter's ICD-10 codes. One forward pass per span, no generation:
P(Yes) = softmax over the next-token logits of {Yes, yes} vs {No, no}.

python -m experiments.bc_llm.llm_score --model qwen35_4b                 # 4-bit (default) -> tag qwen35_4b_q4
python -m experiments.bc_llm.llm_score --model Qwen/Qwen3.5-4B --quant none --tag qwen35_4b_bf16   # bf16 ceiling
-> outputs/bc_llm/llm/<tag>_{train,eval}.json  {"meta": {...}, "p": {span key: P(Yes)}}
Presets: see common.LLM_PRESETS. Needs features_{split}.json from stage 1.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch
from tqdm import tqdm

from anonymizer.data import load_encounters
from anonymizer.decider import LLMJudge

from .common import LLM_PRESETS, OUT, load_json, save_json

def score_split(judge: LLMJudge, rows: list[dict], encs: dict, batch: int) -> tuple[np.ndarray, dict]:
    prompts = [judge.prompt(encs[r["eid"]], encs[r["eid"]]["icd10_codes"], r) for r in rows]
    return judge.score(prompts, batch, lambda it: tqdm(it, desc="llm"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"preset ({', '.join(LLM_PRESETS)}) or HF model id")
    p.add_argument("--tag", default=None, help="name used in output files (default: preset name or model id)")
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    # Default 4-bit: the eval box has <= 4 GB VRAM; "none" (bf16) only to measure the quality ceiling.
    p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"], help="bitsandbytes quantization (CUDA)")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None, help="first N spans per split (smoke test)")
    p.add_argument("--vram-cap-gb", type=float, default=None,
                   help="hard CUDA memory cap (e.g. 4): the run fails with OOM if the model needs more")
    a = p.parse_args(argv)
    a.quant = None if a.quant == "none" else a.quant
    model_id = LLM_PRESETS.get(a.model, a.model)
    tag = a.tag or (a.model if a.model in LLM_PRESETS else re.sub(r"[^a-z0-9]+", "_", a.model.lower()))
    if a.tag is None and a.quant:
        tag += {"4bit": "_q4", "8bit": "_q8"}[a.quant]
    device = a.device if a.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.vram_cap_gb and device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_gb / total))
    judge = LLMJudge(model_id, device, a.quant, a.dtype)
    for split in a.splits.split(","):
        rows = load_json(OUT / f"features_{split}.json")[:a.limit]
        encs = {e["encounter_id"]: e for e in load_encounters(split)}
        p_yes, meta = score_split(judge, rows, encs, a.batch)
        meta.update(model=model_id, tag=tag, split=split, spans=len(rows), dtype=a.dtype, quant=a.quant,
                    device=device, gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None)
        path = OUT / "llm" / f"{tag}_{split}.json"
        save_json({"meta": meta, "p": {r["key"]: float(v) for r, v in zip(rows, p_yes)}}, path)
        print(f"{split}: {meta} -> {path}")


if __name__ == "__main__":
    main()
