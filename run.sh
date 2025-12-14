#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

python src/01-data-preprocessing.py
python src/02-training.py
python src/03-evaluation.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
