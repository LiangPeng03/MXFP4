#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The default suite includes MBPP, which executes model-generated Python.
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

exec "${PYTHON_BIN:-python}" "$(dirname "${BASH_SOURCE[0]}")/model_quant.py" \
    --model_name_or_path="${MODEL:-Qwen/Qwen3-0.6B}" \
    --format=mxfp \
    --w_bits=4 \
    --a_bits=4 \
    --transform_class=identity \
    --dataset_name_or_path=c4 \
    --sequence_length=2048 \
    --seed=0 \
    --dtype=bfloat16 \
    --eval_perplexity \
    --eval_openllm \
    "$@"
