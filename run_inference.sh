#!/usr/bin/env bash
# End-to-end inference: anonymize a split or .jsonl, then print the local proxy metrics.
# Usage: ./run_inference.sh [eval|train|path.jsonl] [output.jsonl] [extra anonymizer.cli flags, e.g. --no-decider]
set -euo pipefail
cd "$(dirname "$0")"
# uses the active env (conda per README); falls back to a local .venv
[ -z "${CONDA_PREFIX:-}${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ] && source .venv/bin/activate

INPUT=${1:-eval}
OUTPUT=${2:-outputs/$(basename "${INPUT%.jsonl}")_anonymized.jsonl}
shift $(( $# < 2 ? $# : 2 ))

python -m anonymizer.cli --input "$INPUT" --output "$OUTPUT" --vram-cap-gb 4 "$@"
python -m anonymizer.evaluate --original "$INPUT" --anonymized "$OUTPUT"
