# Source before training: UTF-8 + PYTHONPATH for this repo root.
# Usage:  source training/path.sh   (from repo root)
export PYTHONIOENCODING=UTF-8
_TRAINING_PATH_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_ROOT="$(cd "${_TRAINING_PATH_SH_DIR}/.." && pwd)"
export PYTHONPATH="${_REPO_ROOT}:${_REPO_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
