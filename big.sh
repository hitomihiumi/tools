#!/usr/bin/env bash
set -euo pipefail

source ./common.sh

PLANNER_GPUS="0,1,2,3,4,5,6,7"
PLANNER_TP_SIZE=8
PLANNER_PP_SIZE=1
PLANNER_MAX_MODEL_LEN=262144
PLANNER_KV_CACHE_DTYPE="nvfp4"
GPU_MEM_UTILIZATION=0.95

mkdir -p "$LOG_DIR"

runpod_kill_gpu_holders

runpod_check_gpu_topology "PLANNER" "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE"

start_vllm "planner" "$PLANNER_MODEL_REPO" "$PLANNER_SERVED_NAME" "$PLANNER_PORT" \
           "$PLANNER_GPUS" "$PLANNER_TP_SIZE" "$PLANNER_PP_SIZE" \
           "$PLANNER_QUANTIZATION" "$PLANNER_KV_CACHE_DTYPE" "$PLANNER_MAX_MODEL_LEN" ""

wait_for_health "planner" "$PLANNER_PORT"

echo "==> Сервер GLM 4.7 з контекстом 262k успішно запущено на всіх 8 GPU!"