# Pseudo ground-truth annotation guidelines

Goal: for each encounter, list the text mentions in `dialogue` and `note` that belong to category **a**, **b** or
**c** (see docs/strategy.md). The ICD-10 codes of the encounter are given and decide b vs c.

## Categories
- **a — sensitive personal data** (must be removed / generalized)
  - `NAME`: any person name — patient, relatives, doctors, nurses, other providers (first, last, nicknames).
    Not: eponyms in medical terms ("Parkinson's disease", "Neer's test", "Clark level"), ambient assistant
    wake words ("dragon", "dax").
  - `DATE`: calendar dates, dates of birth, years, named holidays tied to an event ("on Christmas I fell").
    Not: durations / relative times ("five months ago", "2 weeks", "last year"), habitual weekdays
    ("on saturdays I don't check").
  - `AGE`: exact age expressions ("59-year-old", "i'm 42", "aged 70").
  - `LOCATION`: cities, states, countries, streets, addresses, named hospitals/clinics, named workplaces, schools,
    named stores/restaurants, sports teams (reveal location).
  - `CONTACT_ID`: phone, email, URL, record / insurance / account numbers.
  - `OCCUPATION`: the patient's (or relative's) job, employer, role ("i'm a teacher", "retired firefighter").
  - `ETHNICITY`: ethnicity, nationality, race, religion as a person attribute.
  - `QUASI`: other identifying personal context: hobbies, specific life events, travel destinations, pets' names,
    family structure details that single a person out ("my twin brother", "my daughter's wedding"),
    unusual possessions, specific achievements.
- **b — sensitive clinical, NOT supporting any of the encounter's ICD codes** (must be anonymized)
  - `CONDITION` (diseases, past history, symptoms), `MEDICATION`, `PROCEDURE` (surgeries, imaging, tests),
    `LAB` (test names), `FINDING` (exam findings) that are unrelated to every coded diagnosis.
- **c — code-relevant clinical** (must be kept)
  - same types as b, but the mention is evidence for, the cause of, or the treatment of one of the coded
    diagnoses. Give the supported code in `code`.
  - a hobby / place / mechanism that is the *cause* of a coded injury (e.g. "slipped on ice on the stairs" for
    W00/W10 codes, "playing basketball" for Y93) is **c**, not a.

## Rules
1. `text` must be an **exact substring** of the field (same case, same spacing). Copy it, don't paraphrase.
2. List each **distinct** string once per field and category (all its occurrences are evaluated together).
3. Keep mentions short and specific: "metformin", "type two diabetes", "andrew", "59-year-old".
   Do not annotate whole sentences. Do not annotate generic words alone ("pain", "doctor", "patient", "today").
4. Do not annotate numbers of measurements (blood pressure values, lab values, doses) — handled separately.
5. Negated mentions count: "no history of seasonal allergies" → "seasonal allergies" is b (unless code-relevant).
6. If a string would be both a and c, choose c only when it is truly needed for the code; names are always a.
7. Be exhaustive for a (privacy recall matters most), thorough for b and c.

## Output format
One JSON file per encounter: `pseudo_gt/<split>/<encounter_id>.json` (split = train / eval)
```json
{
  "encounter_id": "D2N001",
  "spans": [
    {"field": "dialogue", "text": "andrew", "category": "a", "type": "NAME", "code": null, "why": "patient first name"},
    {"field": "note", "text": "type 2 diabetes", "category": "c", "type": "CONDITION", "code": "E11.9", "why": "coded diagnosis"},
    {"field": "note", "text": "appendectomy", "category": "b", "type": "PROCEDURE", "code": null, "why": "unrelated surgical history"}
  ]
}
```
Validate with: `python pseudo_gt/validate.py pseudo_gt/eval/<encounter_id>.json`
