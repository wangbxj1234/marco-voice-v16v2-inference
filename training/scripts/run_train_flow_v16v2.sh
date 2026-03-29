#!/usr/bin/env bash
set -euo pipefail

# Flow-only DDP training (v16 v2): causal S3 tokens @ 25 Hz, vocab 1024.
# From repo root:  bash training/scripts/run_train_flow_v16v2.sh
#
# Required env / args via variables (see training/docs/FINETUNING.md):
#   MARCO_V16V2_DATA_LIST   — parquet data.list
#   FLOW_INIT_CKPT or MARCO_RESUME_CKPT — flow .pt to load
#
# Optional:
#   MARCO_V16V2_CONFIG (default training/conf/cosyvoice_emosphere_v16_v2.yaml)
#   MARCO_V16V2_EXP, CUDA_VISIBLE_DEVICES, MARCO_START_EPOCH, MARCO_RESUME_TRAIN_STEP, etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../path.sh
source "${REPO_ROOT}/training/path.sh"

CONFIG_REL="${MARCO_V16V2_CONFIG:-training/conf/cosyvoice_emosphere_v16_v2.yaml}"
if [[ "${CONFIG_REL}" != /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_REL}"
else
  CONFIG_PATH="${CONFIG_REL}"
fi

if [ -f "${REPO_ROOT}/../marco/bin/python" ]; then
  PYTHON="${MARCO_PYTHON:-${REPO_ROOT}/../marco/bin/python}"
else
  PYTHON="${MARCO_PYTHON:-python3}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DATA_LIST="${MARCO_V16V2_DATA_LIST:?Set MARCO_V16V2_DATA_LIST to your parquet data.list}"
FLOW_INIT_CKPT="${FLOW_INIT_CKPT:-}"
EXP_TAG="${MARCO_V16V2_EXP:-cosyvoice_emosphere_v16_v2_finetune}"
START_EPOCH="${MARCO_START_EPOCH:-0}"
train_engine="torch_ddp"
MODEL_DIR_REL="exp/${EXP_TAG}/flow/${train_engine}"
CKPT="${MARCO_RESUME_CKPT:-${FLOW_INIT_CKPT}}"

RESUME_TRAIN_STEP="${MARCO_RESUME_TRAIN_STEP:-}"
if [[ -z "${RESUME_TRAIN_STEP}" && "${START_EPOCH}" =~ ^[0-9]+$ && "${START_EPOCH}" -gt 0 ]]; then
  PREV_YAML="${REPO_ROOT}/${MODEL_DIR_REL}/epoch_$((START_EPOCH - 1))_whole.yaml"
  if [[ -f "${PREV_YAML}" ]]; then
    RESUME_TRAIN_STEP="$(awk '/^step:/{print $2; exit}' "${PREV_YAML}")"
    echo "[resume] Using train step from ${PREV_YAML}: ${RESUME_TRAIN_STEP}"
  else
    echo "[warn] MARCO_START_EPOCH=${START_EPOCH} but missing ${PREV_YAML}; LR schedule may reset."
  fi
fi

if [[ ! -f "${DATA_LIST}" ]]; then
  echo "[ERROR] Missing data list: ${DATA_LIST}"
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] Set FLOW_INIT_CKPT (cold start) or MARCO_RESUME_CKPT (resume). Missing: ${CKPT}"
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Missing config: ${CONFIG_PATH}"
  exit 1
fi

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
job_id="${MARCO_V16V2_RDZV_ID:-2043}"
num_workers="${MARCO_V16V2_NUM_WORKERS:-2}"
DS_CFG="${REPO_ROOT}/training/conf/ds_stage2.json"

echo "============================================================"
echo " v16 v2 flow training"
echo "   repo       : ${REPO_ROOT}"
echo "   config     : ${CONFIG_PATH}"
echo "   checkpoint : ${CKPT}"
echo "   start_epoch: ${START_EPOCH}"
if [[ -n "${RESUME_TRAIN_STEP}" ]]; then
  echo "   resume_train_step: ${RESUME_TRAIN_STEP}"
fi
echo "   data.list  : ${DATA_LIST}"
echo "   GPUs       : ${CUDA_VISIBLE_DEVICES} (${num_gpus} processes)"
echo "   output     : ${REPO_ROOT}/${MODEL_DIR_REL}"
echo "============================================================"

cd "${REPO_ROOT}"

"${PYTHON}" -m torch.distributed.run --nnodes=1 --nproc_per_node="${num_gpus}" \
  --rdzv_id="${job_id}" --rdzv_backend="c10d" --rdzv_endpoint="localhost:12371" \
  cosyvoice_emosphere/bin/train.py \
  --train_engine "${train_engine}" \
  --config "${CONFIG_PATH}" \
  --train_data "${DATA_LIST}" \
  --cv_data "${DATA_LIST}" \
  --model flow \
  --checkpoint "${CKPT}" \
  --start_epoch "${START_EPOCH}" \
  ${RESUME_TRAIN_STEP:+--resume_train_step "${RESUME_TRAIN_STEP}"} \
  --model_dir "${MODEL_DIR_REL}" \
  --tensorboard_dir "tensorboard/${EXP_TAG}/flow/${train_engine}" \
  --ddp.dist_backend nccl \
  --num_workers "${num_workers}" \
  --prefetch 100 \
  --pin_memory \
  --deepspeed_config "${DS_CFG}" \
  --deepspeed.save_states model+optimizer

if [[ "${SKIP_POST_AVERAGE:-0}" == "1" ]]; then
  echo ""
  echo "SKIP_POST_AVERAGE=1 — skipping average_model.py (smoke / custom export)."
  echo "Last epoch checkpoint: ${REPO_ROOT}/${MODEL_DIR_REL}/"
else
  echo ""
  echo "Averaging best 5 checkpoints (validation)..."
  "${PYTHON}" cosyvoice_emosphere/bin/average_model.py \
    --dst_model "${MODEL_DIR_REL}/flow.pt" \
    --src_path "${MODEL_DIR_REL}" \
    --num 5 \
    --val_best
  echo ""
  echo "Done. Flow weights: ${REPO_ROOT}/${MODEL_DIR_REL}/flow.pt"
fi
