#!/usr/bin/env bash
set -euo pipefail

# Reproduce flow streaming reconstruction at hop=8 with a fixed flow checkpoint.
# Default scenario: self-reconstruction on long ESD utterance.
# Usage:
#   FLOW_CKPT=/abs/path/to/epoch_92_whole.pt bash training/scripts/reproduce_flow_stream_hop8.sh
# Optional:
#   PROMPT_WAV=... SOURCE_WAV=... FLOW_TIMESTEPS=8 HOP_TOKENS=8 OUT_DIR=...
#   RUN_BASELINE=1 to emit baseline weights/flow.pt comparison pair.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/training/path.sh"

if [ -f "${REPO_ROOT}/../marco/bin/python" ]; then
  PYTHON="${MARCO_PYTHON:-${REPO_ROOT}/../marco/bin/python}"
else
  PYTHON="${MARCO_PYTHON:-python3}"
fi

DEFAULT_FLOW_CKPT="${REPO_ROOT}/exp/cosyvoice_emosphere_v16_v2_flow_stream_ft_v2_formal/flow/torch_ddp/epoch_92_whole.pt"
PUBLISHED_CKPT_URL="${PUBLISHED_CKPT_URL:-https://drive.google.com/file/d/1F4upBZ0mX6BKLO1S2dF3lrd7VyvP35sA/view?usp=drive_link}"

FLOW_CKPT="${FLOW_CKPT:-${DEFAULT_FLOW_CKPT}}"
FLOW_TIMESTEPS="${FLOW_TIMESTEPS:-8}"
HOP_TOKENS="${HOP_TOKENS:-8}"
RUN_BASELINE="${RUN_BASELINE:-1}"

DEFAULT_LONG_WAV="sample_inputs/esd_source_spk0002_neutral_u000282_long.wav"
PROMPT_WAV="${PROMPT_WAV:-${DEFAULT_LONG_WAV}}"
SOURCE_WAV="${SOURCE_WAV:-${DEFAULT_LONG_WAV}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/flow_stream_ep92_demo}"

if [ ! -f "${FLOW_CKPT}" ]; then
  echo "[error] FLOW_CKPT not found: ${FLOW_CKPT}"
  echo "[hint] Download checkpoint from: ${PUBLISHED_CKPT_URL}"
  echo "[hint] Then run:"
  echo "       FLOW_CKPT=/abs/path/to/epoch_92_whole.pt bash training/scripts/reproduce_flow_stream_hop8.sh"
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[run] ep92 stream hop=${HOP_TOKENS}, steps=${FLOW_TIMESTEPS}"
"${PYTHON}" infer.py \
  --weights_dir weights \
  --tokenizer_pt weights/s3_tokenizer.pt \
  --prompt_wav "${PROMPT_WAV}" \
  --source_wav "${SOURCE_WAV}" \
  --flow_ckpt "${FLOW_CKPT}" \
  --stream \
  --hop_tokens "${HOP_TOKENS}" \
  --flow_timesteps "${FLOW_TIMESTEPS}" \
  --out_wav "${OUT_DIR}/ep92_stream_h${HOP_TOKENS}_t${FLOW_TIMESTEPS}.wav"

echo "[run] ep92 offline steps=${FLOW_TIMESTEPS}"
"${PYTHON}" infer.py \
  --weights_dir weights \
  --tokenizer_pt weights/s3_tokenizer.pt \
  --prompt_wav "${PROMPT_WAV}" \
  --source_wav "${SOURCE_WAV}" \
  --flow_ckpt "${FLOW_CKPT}" \
  --flow_timesteps "${FLOW_TIMESTEPS}" \
  --out_wav "${OUT_DIR}/ep92_offline_t${FLOW_TIMESTEPS}.wav"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "[run] baseline stream hop=${HOP_TOKENS}, steps=${FLOW_TIMESTEPS}"
  "${PYTHON}" infer.py \
    --weights_dir weights \
    --tokenizer_pt weights/s3_tokenizer.pt \
    --prompt_wav "${PROMPT_WAV}" \
    --source_wav "${SOURCE_WAV}" \
    --stream \
    --hop_tokens "${HOP_TOKENS}" \
    --flow_timesteps "${FLOW_TIMESTEPS}" \
    --out_wav "${OUT_DIR}/baseline_stream_h${HOP_TOKENS}_t${FLOW_TIMESTEPS}.wav"

  echo "[run] baseline offline steps=${FLOW_TIMESTEPS}"
  "${PYTHON}" infer.py \
    --weights_dir weights \
    --tokenizer_pt weights/s3_tokenizer.pt \
    --prompt_wav "${PROMPT_WAV}" \
    --source_wav "${SOURCE_WAV}" \
    --flow_timesteps "${FLOW_TIMESTEPS}" \
    --out_wav "${OUT_DIR}/baseline_offline_t${FLOW_TIMESTEPS}.wav"
fi

echo "[done] generated files in ${OUT_DIR}"
