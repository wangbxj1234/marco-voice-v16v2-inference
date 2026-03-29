#!/usr/bin/env bash
# Full-stack check: inference + 1-epoch flow smoke train. Run from repo root.
# Required env:
#   TOKENIZER_PT  — causal S3 export .pt (must match training vocabulary 1024 @ 25 Hz)
#   WEIGHTS_DIR   — directory with flow.pt, hift.pt, campplus.onnx (symlinks OK; we copy with -L)
#
# Optional:
#   MARCO_PYTHON  — python interpreter (default: ../marco/bin/python if present, else python3)
#   CUDA_VISIBLE_DEVICES (default 0 for smoke train uses 1 GPU)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/training/path.sh"

TOKENIZER_PT="${TOKENIZER_PT:?Set TOKENIZER_PT}"
WEIGHTS_DIR="${WEIGHTS_DIR:?Set WEIGHTS_DIR to folder containing flow.pt hift.pt campplus.onnx}"

if [[ -x "${ROOT}/../marco/bin/python" ]]; then
  PY="${MARCO_PYTHON:-${ROOT}/../marco/bin/python}"
else
  PY="${MARCO_PYTHON:-python3}"
fi

VERIFY_W="${ROOT}/.verify_runtime_weights"
rm -rf "${VERIFY_W}"
mkdir -p "${VERIFY_W}"
cp -fL "${WEIGHTS_DIR}/flow.pt" "${WEIGHTS_DIR}/hift.pt" "${WEIGHTS_DIR}/campplus.onnx" "${VERIFY_W}/"
if [[ ! -f "${ROOT}/configs/cosyvoice.yaml" ]]; then
  echo "[ERROR] missing configs/cosyvoice.yaml"
  exit 1
fi
cp -f "${ROOT}/configs/cosyvoice.yaml" "${VERIFY_W}/cosyvoice.yaml"

echo "=== [1/3] Inference ==="
"${PY}" "${ROOT}/infer.py" \
  --weights_dir "${VERIFY_W}" \
  --tokenizer_pt "${TOKENIZER_PT}" \
  --prompt_wav "${ROOT}/sample_inputs/synthetic_3s_16k.wav" \
  --out_wav "${ROOT}/outputs/verify_e2e_infer.wav"

echo "=== [2/3] Smoke parquet (extract + make_parquet) ==="
export TOKENIZER_PT
bash "${ROOT}/training/scripts/prepare_smoke_parquet.sh"

echo "=== [3/3] Flow smoke train (1 epoch, 1 GPU, skip average) ==="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MARCO_V16V2_DATA_LIST="${ROOT}/training/smoke_workspace/parquet/data.list"
export FLOW_INIT_CKPT="${VERIFY_W}/flow.pt"
export MARCO_V16V2_CONFIG="${ROOT}/training/conf/cosyvoice_emosphere_v16_v2_smoke1.yaml"
export MARCO_V16V2_EXP="verify_e2e_smoke"
export SKIP_POST_AVERAGE=1
export MARCO_PYTHON="${PY}"
bash "${ROOT}/training/scripts/run_train_flow_v16v2.sh"

echo "OK: end-to-end verify finished (inference wav + 1 training epoch)."
