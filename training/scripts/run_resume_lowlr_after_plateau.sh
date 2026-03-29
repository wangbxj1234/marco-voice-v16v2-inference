#!/usr/bin/env bash
set -euo pipefail

# Resume flow training with **lower peak LR** (2e-4) after a plateau, matching Marco-Voice
# `Training/run_resume_v16_v2_lowlr_ep129.sh` strategy.
#
# Example: resume from YOUR epoch_129 checkpoint into epoch 130+ with low-LR yaml.
#
# Required:
#   MARCO_RESUME_CKPT  — path to epoch_(K-1)_whole.pt
#   PREV_YAML          — path to epoch_(K-1)_whole.yaml (must contain top-level `step:`)
#
# Optional:
#   MARCO_START_EPOCH (default 130)
#   MARCO_V16V2_CONFIG (default training/conf/cosyvoice_emosphere_v16_v2_resume_lowlr.yaml)
#   MARCO_V16V2_EXP, MARCO_V16V2_DATA_LIST, CUDA_VISIBLE_DEVICES, etc.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MARCO_V16V2_CONFIG="${MARCO_V16V2_CONFIG:-training/conf/cosyvoice_emosphere_v16_v2_resume_lowlr.yaml}"
export MARCO_RESUME_CKPT="${MARCO_RESUME_CKPT:?Set MARCO_RESUME_CKPT to epoch_(K-1)_whole.pt}"
export MARCO_START_EPOCH="${MARCO_START_EPOCH:-130}"
PREV_YAML="${PREV_YAML:?Set PREV_YAML to epoch_(K-1)_whole.yaml for step alignment}"
: "${MARCO_V16V2_DATA_LIST:?Set MARCO_V16V2_DATA_LIST to parquet data.list (same as main train script)}"

if [[ ! -f "${PREV_YAML}" ]]; then
  echo "[ERROR] Missing PREV_YAML=${PREV_YAML}"
  exit 1
fi
STEP_LINE="$(awk '/^step:/{print $2; exit}' "${PREV_YAML}")"
export MARCO_RESUME_TRAIN_STEP="${MARCO_RESUME_TRAIN_STEP:-${STEP_LINE}}"

echo "[resume_lowlr] config=${MARCO_V16V2_CONFIG} peak_lr=2e-4 (in yaml)"
echo "[resume_lowlr] ckpt=${MARCO_RESUME_CKPT} start_epoch=${MARCO_START_EPOCH} resume_train_step=${MARCO_RESUME_TRAIN_STEP}"

bash "${REPO_ROOT}/training/scripts/run_train_flow_v16v2.sh"
