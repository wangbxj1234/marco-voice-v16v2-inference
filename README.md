# Marco Voice — v16 v2 inference (standalone)

Self-contained **git repository** for **causal S3 speech tokens (1024 @ 25 Hz) + Emosphere flow** reconstruction / VC.  
No dependency on the full Marco-Voice training tree: everything required to **run `infer.py`** is under this directory.

## What you get

- `infer.py` — load weights, causal-tokenize with **your** `s3_tokenizer.pt`, run `CosyVoiceModel.vc()`, write 22.05 kHz WAV.
- `cosyvoice_emosphere/` — Emosphere CosyVoice Python package (inference path only).
- `third_party/Matcha-TTS/` — mel backend for YAML.
- `s3tokenizer_train/` — minimal modules to load the causal S3 export checkpoint (`S3TokenizerV1`).
- `emotion_conditioning.py` + `assets/emotion_quantiles/` — emotion2vec + VAD→polar (same as Marco-Voice streaming script).
- `configs/cosyvoice.yaml` — flow definition: **vocab 1024, 25 Hz**, matches v16 v2 training.
- `scripts/download_weights.py` — fill `weights_manifest.json` with HTTPS or `hf:org/repo:file` URLs.
- `sample_inputs/synthetic_3s_16k.wav` — tiny clip for smoke tests.

Large **binary weights are not in git** (`.gitignore`). You host them (Hugging Face, object storage, release assets) and point the manifest at them.

## Quick start

```bash
git clone <YOUR_REPO_URL> marco-voice-v16v2-inference
cd marco-voice-v16v2-inference

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# If needed: install CUDA builds of torch/torchaudio from https://pytorch.org

# 1) Copy manifest and fill URLs for: llm.pt, hift.pt, flow.pt (v16 v2!),
#    campplus.onnx, s3_tokenizer.pt  (speech_tokenizer_v1.onnx optional — omit for this pipeline)
cp weights_manifest.example.json weights_manifest.json
${EDITOR:-vi} weights_manifest.json

python scripts/download_weights.py --manifest weights_manifest.json --output-dir weights

# 2) causal S3 tokenizer .pt — MUST match the export used when building training parquet
export TOKENIZER_PT=/path/to/your/s3tokenizer_export.pt

# 3) Verify imports (no GPU weights load)
python infer.py --smoke_imports

# 4) Run reconstruction on bundled synthetic wav
python infer.py \
  --weights_dir weights \
  --tokenizer_pt "$TOKENIZER_PT" \
  --prompt_wav sample_inputs/synthetic_3s_16k.wav \
  --out_wav outputs/demo.wav

bash verify.sh   # runs smoke; full step if weights + TOKENIZER_PT present
```

### Hugging Face Hub URL format

In `weights_manifest.json` you can use:

```json
"url": "hf:YourOrg/your-weight-repo:flow.pt"
```

(`repo_id` = `YourOrg/your-weight-repo`, file path = `flow.pt`.)  
Set `HF_TOKEN` if the repo is private.

### `weights/` layout after download

- `cosyvoice.yaml` (copied from `configs/` by `download_weights.py`)
- **Required for `infer.py`:** `llm.pt`, `hift.pt`, `flow.pt`, `campplus.onnx`
- **Optional:** `speech_tokenizer_v1.onnx` — only if you call `inference_vc` / `frontend._extract_speech_token` (CosyVoice 50 Hz ONNX path). **Causal-S3 `infer.py` does not need it**; CosyVoice loads without it and uses `--tokenizer_pt` for all speech tokens.
- **Causal tokenizer:** passed as `infer.py --tokenizer_pt` (can also be listed in manifest as `s3_tokenizer.pt` for download scripts).

## Publishing weights (checklist)

1. Upload **v16 v2** `flow.pt` (e.g. best-5 average after training).
2. Upload **v16-compatible** `llm.pt`, `hift.pt`, `campplus.onnx`.
3. Upload **causal** `s3_tokenizer.pt` (same as training `TOKENIZER_PT`).
4. **`speech_tokenizer_v1.onnx` — not required** for this repo’s default inference path.
5. Put direct or `hf:` URLs into `weights_manifest.json` → share as template.

## CLI reference (`infer.py`)

| Argument | Description |
|----------|-------------|
| `--weights_dir` | Default `./weights` |
| `--tokenizer_pt` | Causal S3 export `.pt` (required) |
| `--prompt_wav` | 16 kHz+ WAV (resampled internally) |
| `--source_wav` | Optional; default = prompt (self-reconstruction) |
| `--out_wav` | Default `./outputs/reconstruction.wav` |
| `--flow_ckpt` | Optional override for flow state dict |
| `--hop_tokens` | Optional streaming hop override |
| `--device` | `cuda` or `cpu` |
| `--smoke_imports` | Only test imports |

## Parent Marco-Voice repo

If you keep this folder **inside** the full Marco-Voice monorepo, `../run_vc_s3causal_emosphere.py` is a thin wrapper that forwards to `infer.py` with legacy flag names (`--model_dir`, `--out_dir`, `--output`).

## Publish **only** this tree to GitHub

From the parent repo (or after copying this folder elsewhere):

```bash
cd marco-voice-v16v2-inference
git init
git add -A
git status   # confirm weights/*.pt are NOT staged (.gitignore)
git commit -m "Marco Voice v16 v2 standalone inference"
git branch -M main
git remote add origin git@github.com:YOUR_ORG/marco-voice-v16v2-inference.git
git push -u origin main
```

Clone size is ~**100 MB+** mainly due to `cosyvoice_emosphere/tokenizer/assets/*.tiktoken` (Whisper text tokenizer assets pulled in by the YAML `get_tokenizer` path).

## License

See `LICENSE` and `NOTICE.md`.
