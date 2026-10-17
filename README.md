# Patient Data Anonymization — Bosch @ Neural Wave

Anonymizes doctor–patient dialogues and clinical notes while preserving the information that supports the
encounter's ICD-10 codes. Runs fully on-prem (no cloud models, < 4 GB VRAM).

- Strategy: [docs/strategy.md](docs/strategy.md)
- Architecture: [docs/architecture.md](docs/architecture.md)

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Hugging Face models (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `obi/deid_roberta_i2b2`) download on first
run. The ICD-10-CM embedding index is built once (~1–2 min) and cached in `.cache/`.

## Usage
```bash
# anonymize (train | eval | path/to/file.jsonl)
python -m anonymizer.cli --input eval --output outputs/eval_anonymized.jsonl

# bonus: ignore the given ICD codes and predict them instead
python -m anonymizer.cli --input eval --no-codes --output outputs/eval_anonymized_nocodes.jsonl

# local proxy metrics
python -m anonymizer.evaluate --original eval --anonymized outputs/eval_anonymized.jsonl
```

Options: `--t-keep` / `--t-remove` (privacy–utility thresholds), `--no-ner` (rules only), `--device cpu|cuda|mps`,
`--limit N`.

## Output
- `<output>.jsonl` — same schema as the input; `dialogue` and `note` anonymized, `icd10_codes` unchanged.
- `<output>.decisions.jsonl` — per encounter, every decision: span, category (a / b / c), action
  (keep / generalize / perturb / remove / pseudonymize), relevance score, matched code, reason.

## Example
```
before: hi , andrew . ... andrew is a 59-year-old male ... type two diabetes ... ibuprofen
after:  hi , [PATIENT] . ... [PATIENT] is a 50s-year-old male ... type two diabetes ... [other medication]
```
