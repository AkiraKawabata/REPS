#!/bin/bash
# This script runs self-evaluation (pairwise tournament) using reps.py

# Usage:
#   sh run_self_eval.sh <INPUT_JSONL> <DATASET> [<CONFIG_FILE>]
# Example:
#   sh run_self_eval.sh output/arc_cot_results.jsonl arc config/model_config.yaml

set -e

export CUDA_VISIBLE_DEVICES=2,3

INPUT_FILE=$1
DATASET=$2
CONFIG_FILE=${3:-"config/model_config.yaml"}

echo "Running self-evaluation..."
echo "  Input file: ${INPUT_FILE}"
echo "  Dataset: ${DATASET}"
echo "  Config: ${CONFIG_FILE}"

python reps.py \
    --input_file "${INPUT_FILE}" \
    --config_file "${CONFIG_FILE}" \
    --dataset "${DATASET}"
