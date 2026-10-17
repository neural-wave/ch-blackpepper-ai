"""Load / save the patient-doctor encounter JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS = {
    "train": DATA_DIR / "patient_doctor_encounters.jsonl",
    "eval": DATA_DIR / "eval_patient_doctor_encounters.jsonl",
}


def load_encounters(split_or_path: str | Path = "train") -> list[dict]:
    """Return encounters (encounter_id, dataset, dialogue, note, icd10_codes) from a split name or path."""
    path = SPLITS.get(str(split_or_path), split_or_path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="train | eval | path to .jsonl")
    parser.add_argument("--show", type=int, default=None, help="print the encounter at this index")
    args = parser.parse_args()

    encounters = load_encounters(args.split)
    print(f"{args.split}: {len(encounters)} encounters")

    if args.show is not None:
        enc = encounters[args.show]
        print(f"\n=== {enc['encounter_id']} ({enc['dataset']}) ===")
        print("\n--- DIALOGUE ---\n" + enc["dialogue"])
        print("\n--- NOTE ---\n" + enc["note"])
        print("\n--- ICD-10 ---")
        for c in enc["icd10_codes"]:
            print(f"{c['code']}: {c['description']}")
