#!/usr/bin/env bash
set -euo pipefail
# Build a one-utterance parquet + data.list under training/smoke_workspace/ (all in-repo).
# Requires: TOKENIZER_PT, repo root layout, sample WAV.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../path.sh
source "${REPO_ROOT}/training/path.sh"

if [ -f "${REPO_ROOT}/../marco/bin/python" ]; then
  PYTHON="${MARCO_PYTHON:-${REPO_ROOT}/../marco/bin/python}"
else
  PYTHON="${MARCO_PYTHON:-python3}"
fi

TOKENIZER_PT="${TOKENIZER_PT:?Set TOKENIZER_PT to causal S3 export .pt}"
WAV="${SMOKE_WAV:-${REPO_ROOT}/sample_inputs/esd_source_spk0002_neutral_u000282_long.wav}"
RAW="${REPO_ROOT}/training/smoke_workspace/raw"
PARQ="${REPO_ROOT}/training/smoke_workspace/parquet"

rm -rf "${RAW}" "${PARQ}"
mkdir -p "${RAW}" "${PARQ}"

"${PYTHON}" "${REPO_ROOT}/training/tools/build_smoke_emosphere_dir.py" \
  --out_dir "${RAW}" \
  --wav "${WAV}"

"${PYTHON}" "${REPO_ROOT}/training/tools/extract_speech_token_s3.py" \
  --dir "${RAW}" \
  --tokenizer_pt "${TOKENIZER_PT}" \
  --device "${EXTRACT_DEVICE:-cuda}" \
  --batch_size "${EXTRACT_BATCH_SIZE:-4}"

export SRC_DIR="${RAW}"
export DES_DIR="${PARQ}"
bash "${REPO_ROOT}/training/scripts/prep_parquet_s3causal.sh"

echo "OK: data.list → ${PARQ}/data.list"
