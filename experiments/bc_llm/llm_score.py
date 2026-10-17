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
import time

import numpy as np
import torch
from tqdm import tqdm

from anonymizer.data import load_encounters

from .common import LLM_PRESETS, OUT, load_json, save_json

PROMPT = """ICD-10 codes assigned to this encounter:
{codes}

Excerpt from the {source}:
"{context}"

The clinical mention in question is marked with [[ ]]: "{span}".
Is this mention evidence for, a cause of, or a treatment of at least one of the ICD-10 codes above (keep it), \
or is it clinical information unrelated to all of these codes (remove it)?
Answer with exactly one word: Yes (evidence for a code) or No (unrelated)."""


def context(text: str, start: int, end: int, width: int = 220) -> str:
    """The span's sentence plus about one sentence either side, span marked with [[ ]]."""
    left, right = text[max(0, start - width):start], text[end:end + width]
    cuts = list(re.finditer(r"[.!?\n]\s", left))
    if len(cuts) >= 2:
        left = left[cuts[-2].end():]
    cuts = list(re.finditer(r"[.!?\n]\s", right))
    if len(cuts) >= 2:
        right = right[:cuts[1].start() + 1]
    return re.sub(r"\s+", " ", f"{left}[[{text[start:end]}]]{right}").strip()


def build_prompts(tok, rows: list[dict], encs: dict) -> list[str]:
    prompts = []
    for r in rows:
        enc = encs[r["eid"]]
        msg = PROMPT.format(
            codes="\n".join(f"- {c['code']}: {c['description']}" for c in enc["icd10_codes"]),
            source="doctor-patient conversation" if r["field"] == "dialogue" else "clinical note",
            context=context(enc[r["field"]], r["start"], r["end"]), span=r["text"])
        # enable_thinking=False: Qwen3.x answers directly; other templates ignore the variable.
        prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False))
    return prompts


def load_model(model_id: str, device: str, dtype: str, quant: str | None, attn: str | None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]}
    if attn:
        kw["attn_implementation"] = attn
    if quant:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=kw["dtype"])
                                     if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        kw["device_map"] = device
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except Exception:
        # multimodal checkpoints (Qwen3.5, Gemma 3 / MedGemma) without a causal-LM auto mapping
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if not quant:
        model = model.to(device)
    return tok, model.eval()


def answer_ids(tok) -> tuple[list[int], list[int]]:
    def first(words):
        return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    return first(["Yes", "yes"]), first(["No", "no"])


@torch.no_grad()
def score(tok, model, prompts: list[str], batch: int, device: str) -> tuple[np.ndarray, dict]:
    yes, no = answer_ids(tok)
    p_yes, mass = np.zeros(len(prompts)), np.zeros(len(prompts))
    order = np.argsort([len(p) for p in prompts])       # similar lengths per batch -> little padding
    n_tok, t0 = 0, time.time()
    for i in tqdm(range(0, len(order), batch), desc="llm"):
        idx = order[i:i + batch]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        n_tok += int(enc["attention_mask"].sum())
        # logits_to_keep=1: only the last position; full logits (batch x tokens x ~250k vocab) dominated peak VRAM.
        try:
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            logits = model(**enc).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        y = torch.logsumexp(logp[:, yes], dim=-1)
        n = torch.logsumexp(logp[:, no], dim=-1)
        p_yes[idx] = torch.sigmoid(y - n).cpu().numpy()
        mass[idx] = (y.exp() + n.exp()).cpu().numpy()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None
    return p_yes, {"sec_per_span": round(dt / max(len(prompts), 1), 4), "tok_per_s": round(n_tok / dt),
                   "seconds": round(dt, 1), "peak_vram_gb": peak and round(peak, 2),
                   "yes_no_mass_mean": round(float(mass.mean()), 4)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"preset ({', '.join(LLM_PRESETS)}) or HF model id")
    p.add_argument("--tag", default=None, help="name used in output files (default: preset name or model id)")
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    # Default 4-bit: the eval box has <= 4 GB VRAM; "none" (bf16) only to measure the quality ceiling.
    p.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"], help="bitsandbytes quantization (CUDA)")
    p.add_argument("--attn", default=None, help="attn_implementation, e.g. sdpa / flash_attention_2")
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
    tok, model = load_model(model_id, device, a.dtype, a.quant, a.attn)
    for split in a.splits.split(","):
        rows = load_json(OUT / f"features_{split}.json")[:a.limit]
        encs = {e["encounter_id"]: e for e in load_encounters(split)}
        prompts = build_prompts(tok, rows, encs)
        p_yes, meta = score(tok, model, prompts, a.batch, device)
        meta.update(model=model_id, tag=tag, split=split, spans=len(rows), dtype=a.dtype, quant=a.quant,
                    device=device, gpu=torch.cuda.get_device_name() if device.startswith("cuda") else None)
        path = OUT / "llm" / f"{tag}_{split}.json"
        save_json({"meta": meta, "p": {r["key"]: float(v) for r, v in zip(rows, p_yes)}}, path)
        print(f"{split}: {meta} -> {path}")


if __name__ == "__main__":
    main()
