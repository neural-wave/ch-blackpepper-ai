"""Validate pseudo-GT annotation files against the source encounters.

python pseudo_gt/validate.py pseudo_gt/eval/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from anonymizer.data import load_encounters  # noqa: E402

TYPES = {
    "a": {"NAME", "DATE", "AGE", "LOCATION", "CONTACT_ID", "OCCUPATION", "ETHNICITY", "QUASI"},
    "b": {"CONDITION", "MEDICATION", "PROCEDURE", "LAB", "FINDING"},
    "c": {"CONDITION", "MEDICATION", "PROCEDURE", "LAB", "FINDING", "QUASI", "LOCATION", "OCCUPATION"},
}


def validate(path: Path, encounters: dict) -> list[str]:
    errors = []
    try:
        ann = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    enc = encounters.get(ann.get("encounter_id"))
    if enc is None:
        return [f"unknown encounter_id {ann.get('encounter_id')}"]
    codes = {c["code"] for c in enc["icd10_codes"]}
    for i, s in enumerate(ann.get("spans", [])):
        where = f"span {i} {s.get('text')!r}"
        if s.get("field") not in ("dialogue", "note"):
            errors.append(f"{where}: bad field {s.get('field')!r}")
            continue
        if s.get("text", "") not in enc[s["field"]]:
            errors.append(f"{where}: not an exact substring of {s['field']}")
        cat = s.get("category")
        if cat not in TYPES:
            errors.append(f"{where}: bad category {cat!r}")
        elif s.get("type") not in TYPES[cat]:
            errors.append(f"{where}: type {s.get('type')!r} not allowed for category {cat}")
        if cat == "c" and s.get("code") not in codes:
            errors.append(f"{where}: c-span code {s.get('code')!r} not among {sorted(codes)}")
    return errors


if __name__ == "__main__":
    encounters = {e["encounter_id"]: e for split in ("train", "eval") for e in load_encounters(split)}
    bad = 0
    for p in sys.argv[1:]:
        errs = validate(Path(p), encounters)
        bad += bool(errs)
        print(f"{p}: {'OK' if not errs else f'{len(errs)} error(s)'}")
        for e in errs:
            print("   ", e)
    sys.exit(1 if bad else 0)
