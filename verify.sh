#!/usr/bin/env bash
# End-to-end sanity checks for a fresh clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

echo "=== [1/2] Import smoke (no weights) ==="
$PYTHON infer.py --smoke_imports

echo "=== [2/2] Full inference (requires weights/ + TOKENIZER_PT) ==="
if [[ ! -f "$ROOT/weights/flow.pt" ]]; then
  echo "SKIP: no weights/flow.pt — copy weights or run:"
  echo "  cp weights_manifest.example.json weights_manifest.json  # then edit URLs"
  echo "  python scripts/download_weights.py --manifest weights_manifest.json"
  echo "Then: export TOKENIZER_PT=/path/to/causal_s3_export.pt"
  echo "      $PYTHON infer.py --tokenizer_pt \"\$TOKENIZER_PT\" --prompt_wav sample_inputs/synthetic_3s_16k.wav"
  exit 0
fi

if [[ -z "${TOKENIZER_PT:-}" ]]; then
  echo "SKIP: set TOKENIZER_PT to your causal S3 tokenizer .pt"
  exit 0
fi

$PYTHON infer.py \
  --weights_dir "$ROOT/weights" \
  --tokenizer_pt "$TOKENIZER_PT" \
  --prompt_wav "$ROOT/sample_inputs/synthetic_3s_16k.wav" \
  --out_wav "$ROOT/outputs/verify_out.wav"

echo "OK: wrote outputs/verify_out.wav"
