#!/bin/bash
# 本地运行 MATH500 power sampling（无需 Slurm）
# 对应 power_samp_math.sh：5 shards × 8 seeds = 40 个任务，默认用 2 块 GPU 并行
#
# 用法:
#   bash llm_experiments/scripts/power_samp_math_local.sh          # 跑全部 40 个任务
#   bash llm_experiments/scripts/power_samp_math_local.sh 12      # 只跑 task_id=12
#   TASK_START=0 TASK_END=7 NUM_GPUS=2 bash ...                   # 只跑 task 0–7
#   HF_HOME=/path/to/hf HF_TOKEN=xxx bash ...                     # 覆盖 HF 配置
#   MODEL=phi bash llm_experiments/scripts/power_samp_math_local.sh  # 换模型
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLM_DIR="${REPO_ROOT}/llm_experiments"

NUM_SHARDS=5
NUM_SEEDS=8
NUM_TASKS=$((NUM_SHARDS * NUM_SEEDS))

NUM_GPUS="${NUM_GPUS:-2}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-$((NUM_TASKS - 1))}"

# --- 按需修改 ---
MODEL="${MODEL:-qwen_math}"   # 可选: qwen, qwen_math, phi, tulu, qwen_math_grpo, phi_grpo
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# HF_TOKEN 建议 export 后运行，勿写死在脚本里

# --- 环境 ---
if command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate psamp
fi

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/models"
export PYTHONPATH="${LLM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${LLM_DIR}"

run_one_task() {
  local task_id=$1
  local gpu=$2
  local seed=$((task_id % NUM_SEEDS))
  local batch_idx=$((task_id / NUM_SEEDS))

  echo "[$(date '+%F %T')] GPU=${gpu} task=${task_id} batch_idx=${batch_idx} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python power_samp_math.py \
    --batch_idx="${batch_idx}" \
    --mcmc_steps=10 \
    --temperature=0.25 \
    --seed="${seed}" \
    --model="${MODEL}"
}

if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  TASK_START=$1
  TASK_END=$1
fi

echo "Repo: ${REPO_ROOT}"
echo "Tasks: ${TASK_START}..${TASK_END} (${NUM_GPUS} GPUs)"
echo "Model: ${MODEL}"
echo "HF_HOME: ${HF_HOME}"

slot=0
for task_id in $(seq "${TASK_START}" "${TASK_END}"); do
  while [[ $(jobs -rp | wc -l) -ge ${NUM_GPUS} ]]; do
    wait -n 2>/dev/null || wait
  done
  gpu=$((slot % NUM_GPUS))
  slot=$((slot + 1))
  run_one_task "${task_id}" "${gpu}" &
done
wait
echo "[$(date '+%F %T')] All tasks finished."
