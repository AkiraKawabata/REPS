#!/bin/bash
# This script runs the CoT generation process.

# Usage:
#   sh run_cot.sh <DATASET> [<CONFIG_FILE>]
# Example:
#   sh run_cot.sh arc config/model_config.yaml

set -e

export CUDA_VISIBLE_DEVICES=0,1

DATASET=$1
CONFIG_FILE=${2:-"config/model_config.yaml"}

echo "Running CoT generation for ${DATASET} with config ${CONFIG_FILE}"

python cot_generation.py \
    --dataset "${DATASET}" \
    --config_file "${CONFIG_FILE}"
