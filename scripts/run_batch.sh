#!/bin/bash
# ============================================================
# Batch-based experiment runner
# Writes jobs to a file, then processes in batches of 6
# ============================================================
cd "$(dirname "$0")/.."  # cd to project root

export HTTP_PROXY=http://127.0.0.1:37890
export HTTPS_PROXY=http://127.0.0.1:37890
export HF_ENDPOINT=https://huggingface.co
export no_proxy=localhost,127.0.0.1

PYTHON=python
GPUS=(0 1 2 3 4 5)
GPU_COUNT=${#GPUS[@]}

mkdir -p results/logs results/checkpoints

# Build job list file
JOBFILE="/tmp/experiment_jobs_$$.txt"
rm -f "$JOBFILE"

echo "Building experiment list..."

# Phase 1: Main experiments (7 methods x 3 datasets x 3 seeds)
for dataset in cub200 flowers102 stanford_cars; do
    for method in full_ft linear_probe bitfit lora ssf adaptformer ssf_sparse; do
        for seed in 42 123 456; do
            echo "$method|$dataset|$seed||${method}_${dataset}_seed${seed}" >> "$JOBFILE"
        done
    done
done

# Phase 2: Sample efficiency (7 methods x 3 datasets)
for dataset in cub200 flowers102 stanford_cars; do
    for method in full_ft linear_probe bitfit lora ssf adaptformer ssf_sparse; do
        echo "$method|$dataset|42|--exp sample_efficiency|${method}_${dataset}_sample_efficiency" >> "$JOBFILE"
    done
done

# Phase 3: Layer ablation (5 PEFT methods x 3 datasets)
for dataset in cub200 flowers102 stanford_cars; do
    for method in bitfit lora ssf adaptformer ssf_sparse; do
        echo "$method|$dataset|42|--exp layer_ablation|${method}_${dataset}_layer_ablation" >> "$JOBFILE"
    done
done

TOTAL=$(wc -l < "$JOBFILE")
echo "Total experiments: $TOTAL"
echo ""

# Read all jobs into an array
mapfile -t JOBS < "$JOBFILE"
rm -f "$JOBFILE"

# Process in batches
BATCH=0
for ((i=0; i<TOTAL; i+=GPU_COUNT)); do
    BATCH=$((BATCH + 1))
    end=$((i+GPU_COUNT))
    [ $end -gt $TOTAL ] && end=$TOTAL

    echo "=== Batch $BATCH (jobs $((i+1))-$end of $TOTAL) ==="

    pids=()
    for ((j=0; j<GPU_COUNT && i+j<TOTAL; j++)); do
        idx=$((i+j))
        IFS='|' read -r method dataset seed exp_flag name <<< "${JOBS[$idx]}"

        log="results/logs/${name}.log"
        out="results/logs/${name}.json"
        gpu=${GPUS[$j]}

        if [ -f "$out" ]; then
            echo "  SKIP: $name"
            continue
        fi

        echo "  [GPU $gpu] $name"
        CUDA_VISIBLE_DEVICES=$gpu python main.py \
            --method "$method" --dataset "$dataset" --seed "$seed" \
            --data_root "./data" $exp_flag --output "$out" \
            > "$log" 2>&1 &
        pids+=($!)
    done

    # Wait for all jobs in this batch
    for pid in "${pids[@]}"; do
        wait $pid 2>/dev/null || true
    done

    # Report batch results
    for ((j=0; j<GPU_COUNT && i+j<TOTAL; j++)); do
        idx=$((i+j))
        IFS='|' read -r method dataset seed exp_flag name <<< "${JOBS[$idx]}"
        out="results/logs/${name}.json"
        if [ -f "$out" ]; then
            echo "  OK: $name"
        else
            echo "  FAIL: $name"
        fi
    done

    echo "=== Batch $BATCH done ==="
done

echo ""
echo "========== ALL EXPERIMENTS COMPLETE =========="
SUCCESS=$(ls results/logs/*.json 2>/dev/null | wc -l)
echo "Successful: $SUCCESS / $TOTAL"
