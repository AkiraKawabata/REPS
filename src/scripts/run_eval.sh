#!/bin/bash
# This script runs the evaluation step using eval.py

# Usage:
#   sh run_eval.sh <DATASET> <MODEL_PATH> <EVAL_DATA_DIR> [<BATCH_SIZE>]
# Example:
#   sh run_eval.sh arc meta-llama/Llama-2-7b-hf data/eval_data 8

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=0  # Specify GPU device if needed

DATASET=$1
MODEL_PATH=$2
EVAL_DATA_DIR=$3
BATCH_SIZE=${4:-8}

echo "Running evaluation..."
echo "  Dataset: ${DATASET}"
echo "  Model path: ${MODEL_PATH}"
echo "  Eval data dir: ${EVAL_DATA_DIR}"
echo "  Batch size: ${BATCH_SIZE}"

python eval.py \
    --dataset "${DATASET}" \
    --model_path "${MODEL_PATH}" \
    --eval_data_dir "${EVAL_DATA_DIR}" \
    --batch_size "${BATCH_SIZE}"
#!/bin/bash
# This script runs the evaluation step using eval.py

# Usage:
#   sh run_eval.sh <DATASET> <MODEL_PATH> <EVAL_DATA_DIR> [<BATCH_SIZE>]
# Example:
#   sh run_eval.sh arc meta-llama/Llama-2-7b-hf data/eval_data 8

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=0  # Specify GPU device if needed

DATASET=$1
MODEL_PATH=$2
EVAL_DATA_DIR=$3
BATCH_SIZE=${4:-8}

echo "Running evaluation..."
echo "  Dataset: ${DATASET}"
echo "  Model path: ${MODEL_PATH}"
echo "  Eval data dir: ${EVAL_DATA_DIR}"
echo "  Batch size: ${BATCH_SIZE}"

python eval.py \
    --dataset "${DATASET}" \
    --model_path "${MODEL_PATH}" \
    --eval_data_dir "${EVAL_DATA_DIR}" \
    --batch_size "${BATCH_SIZE}"
