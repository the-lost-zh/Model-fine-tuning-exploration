#!/bin/bash
# Parallel experiment launcher for 3-GPU setup.
# Each experiment runs on a dedicated GPU via CUDA_VISIBLE_DEVICES.
# No distributed training code needed.
#
# Usage: bash scripts/run_parallel.sh

set -e

DATASETS=("cub200" "flowers102" "stanford_cars")
METHODS=("full_ft" "linear_probe" "bitfit" "lora" "ssf" "adaptformer" "ssf_sparse")
SEEDS=(42 123 456)
DATA_ROOT="./data"
GPU_COUNT=3

TOTAL_JOBS=0
RUNNING=0

run_on_gpu() {
    local gpu_id=$1
    local method=$2
    local dataset=$3
    local seed=$4
    local output="results/logs/${method}_${dataset}_seed${seed}.json"

    echo "[GPU $gpu_id] Starting $method on $dataset (seed=$seed)"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        --method "$method" --dataset "$dataset" \
        --seed "$seed" --data_root "$DATA_ROOT" \
        --output "$output" \
        > "results/logs/${method}_${dataset}_seed${seed}.log" 2>&1
    echo "[GPU $gpu_id] Finished $method on $dataset (seed=$seed)"
}

# Launch experiments: 3 at a time
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Wait for a free GPU if all are busy
            while [ $RUNNING -ge $GPU_COUNT ]; do
                wait -n
                RUNNING=$((RUNNING - 1))
            done

            gpu_id=$((TOTAL_JOBS % GPU_COUNT))
            run_on_gpu "$gpu_id" "$method" "$dataset" "$seed" &
            RUNNING=$((RUNNING + 1))
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        done
    done
done

# Wait for remaining jobs
wait
echo "All $TOTAL_JOBS experiments completed."
