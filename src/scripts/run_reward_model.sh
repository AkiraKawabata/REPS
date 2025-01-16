#!/bin/bash
# This script runs the reward model training using reward_training.py

# Usage:
#   sh run_reward_model.sh <INPUT_FILE> <CONFIG_FILE>
# Example:
#   sh run_reward_model.sh output/arc_cot_results_evaluated_results.jsonl config/model_config.yaml

set -e

export CUDA_VISIBLE_DEVICES=4,5

INPUT_FILE=$1
CONFIG_FILE=$2

echo "Training reward model..."
echo "  Input file: ${INPUT_FILE}"
echo "  Config file: ${CONFIG_FILE}"

python reward_training.py \
    --input_file "${INPUT_FILE}" \
    --config_file "${CONFIG_FILE}"
