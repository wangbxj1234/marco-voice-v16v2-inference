#!/usr/bin/env bash
set -euo pipefail

# Build Emosphere-style parquet shards (causal S3 tokens @ 25 Hz, 1024 codes).
#
# Prerequisite: SRC_DIR must already contain utt2speech_token.pt. Generate it with:
#   python training/tools/extract_speech_token_s3.py --dir SRC_DIR --tokenizer_pt /path/to/s3_export.pt
#
# Or run the one-shot smoke builder:
#   bash training/scripts/prepare_smoke_parquet.sh   # needs TOKENIZER_PT
#
# Usage:
#   export SRC_DIR=/path/to/processed/train
#   export DES_DIR=/path/to/output_parquet
#   bash training/scripts/prep_parquet_s3causal.sh
#
# After extraction, SRC_DIR/utt2speech_token.pt must exist.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../path.sh
source "${REPO_ROOT}/training/path.sh"

if [ -f "${REPO_ROOT}/../marco/bin/python" ]; then
  PYTHON="${MARCO_PYTHON:-${REPO_ROOT}/../marco/bin/python}"
else
  PYTHON="${MARCO_PYTHON:-python3}"
fi

SRC_DIR="${SRC_DIR:?Set SRC_DIR to processed train dir (with utt2speech_token.pt)}"
DES_DIR="${DES_DIR:?Set DES_DIR to output parquet directory}"

MAKE_PQ="${REPO_ROOT}/training/tools/make_parquet_list_eposhere.py"

if [[ ! -f "${SRC_DIR}/utt2speech_token.pt" ]]; then
  echo "[ERROR] Missing ${SRC_DIR}/utt2speech_token.pt"
  echo "        Run: python training/tools/extract_speech_token_s3.py --dir \"\$SRC_DIR\" --tokenizer_pt ..."
  exit 1
fi
if [[ ! -f "${MAKE_PQ}" ]]; then
  echo "[ERROR] Missing ${MAKE_PQ}"
  exit 1
fi

mkdir -p "${DES_DIR}"
"${PYTHON}" "${MAKE_PQ}" \
  --src_dir "${SRC_DIR}" \
  --des_dir "${DES_DIR}" \
  --num_utts_per_parquet 1000 \
  --num_processes 1

echo "Done. Training data.list is typically: ${DES_DIR}/data.list"
