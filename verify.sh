#!/usr/bin/env bash
# End-to-end sanity checks for a fresh clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Same interpreter as Marco-Voice training when this repo lives under the monorepo (Training/*.sh use ../marco/bin/python).
_MARCO_PY="$(cd "$ROOT/.." && pwd)/marco/bin/python"
if [[ -z "${PYTHON:-}" && -x "$_MARCO_PY" ]]; then
  PYTHON="$_MARCO_PY"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "=== [1/2] Import smoke (no weights) ==="
$PYTHON infer.py --smoke_imports

echo "=== [2/2] Full inference (requires weights/ + TOKENIZER_PT) ==="
if [[ ! -f "$ROOT/weights/flow.pt" ]]; then
  echo "SKIP: no weights/flow.pt — copy weights or run:"
  echo "  cp weights_manifest.example.json weights_manifest.json  # then edit URLs"
  echo "  python scripts/download_weights.py --manifest weights_manifest.json"
  echo "Then: export TOKENIZER_PT=weights/s3_tokenizer.pt   # or absolute path"
  echo "      $PYTHON infer.py --weights_dir weights --tokenizer_pt \"\$TOKENIZER_PT\" --prompt_wav sample_inputs/synthetic_3s_16k.wav"
  exit 0
fi

if [[ -z "${TOKENIZER_PT:-}" ]]; then
  if [[ -f "$ROOT/weights/s3_tokenizer.pt" ]]; then
    TOKENIZER_PT="$ROOT/weights/s3_tokenizer.pt"
  else
    echo "SKIP: set TOKENIZER_PT or place weights/s3_tokenizer.pt"
    exit 0
  fi
fi

$PYTHON infer.py \
  --weights_dir "$ROOT/weights" \
  --tokenizer_pt "$TOKENIZER_PT" \
  --prompt_wav "$ROOT/sample_inputs/synthetic_3s_16k.wav" \
  --out_wav "$ROOT/outputs/verify_out.wav"

echo "OK: wrote outputs/verify_out.wav"
