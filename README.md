# Patient Data Anonymization — Bosch @ Neural Wave

Anonymizes doctor–patient dialogues and clinical notes while preserving the information that supports the
encounter's ICD-10 codes. Runs fully on-prem (no cloud models, < 4 GB VRAM).

## Deliverables
Report and Video are at .deliverables/

Results json from the pipeline are here: https://drive.google.com/drive/folders/1-7vN66R5Nq-t7xXAtxgRARoFn5lrNb4D?usp=sharing

## Setup
```bash
conda create -n blackpepper python=3.12
conda activate blackpepper
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Hugging Face models (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `obi/deid_roberta_i2b2`, `Qwen/Qwen3.5-4B`)
download on first run. The ICD-10-CM embedding index is built once (~1–2 min) and cached in `.cache/`. Our trained
classifiers ship in `models/` (`decider.joblib`, `smalltalk.joblib`): no training needed for inference.

## End-to-end inference
```bash
./run_inference.sh                                  # eval split -> outputs/eval_anonymized.jsonl + metrics
./run_inference.sh path/to/file.jsonl out.jsonl     # any file with the input schema
./run_inference.sh eval out.jsonl --no-decider      # no LLM judge (no CUDA GPU)
```
Runs under a hard 4 GB VRAM cap. The 4-bit LLM judge needs CUDA; elsewhere it loads in bf16 (slower, ~8 GB).

## Usage
```bash
# anonymize (train | eval | path/to/file.jsonl)
python -m anonymizer.cli --input eval --output outputs/eval_anonymized.jsonl

# bonus: ignore the given ICD codes and predict them instead
python -m anonymizer.cli --input eval --no-codes --output outputs/eval_anonymized_nocodes.jsonl

# local proxy metrics
python -m anonymizer.evaluate --original eval --anonymized outputs/eval_anonymized.jsonl
```

Options: `--no-decider` (relevance threshold instead of the LLM judge), `--vram-cap-gb 4`, `--t-keep` / `--t-remove`
(privacy–utility thresholds), `--no-ner` (rules only), `--device cpu|cuda|mps`, `--limit N`.

## Retraining (optional)
```bash
python -m anonymizer.smalltalk --train      # -> models/smalltalk.joblib
python -m anonymizer.decider --train        # -> models/decider.joblib (runs the LLM on train; GPU)
```

## Output
- `<output>.jsonl` — same schema as the input; `dialogue` and `note` anonymized, `icd10_codes` unchanged.
- `<output>.decisions.jsonl` — per encounter, every decision: span, category (a / b / c), action
  (keep / generalize / perturb / remove / pseudonymize), relevance score, matched code, reason.

## Example
```
before: hi , andrew . ... andrew is a 59-year-old male ... type two diabetes ... ibuprofen
after:  hi , [PATIENT] . ... [PATIENT] is a 50s-year-old male ... type two diabetes ... [other medication]
```
