# Third-party notices

This repository vendors minimal inference-time code and assets from:

1. **CosyVoice / cosyvoice_emosphere** (Alibaba et al.) — Apache-2.0. Source: Marco-Voice `Models/marco_voice/cosyvoice_emosphere/`. This repo includes **`bin/train.py`**, **`bin/average_model.py`**, and **`llm/` + `tokenizer/`** so **training YAML** can build the full model object; **`infer.py` still loads only flow+hift** and does not use LLM weights. The high-level `CosyVoice` CLI is not shipped here.

2. **Matcha-TTS** — MIT. `third_party/Matcha-TTS/` provides `matcha.utils.audio.mel_spectrogram` for HyperPyYAML `feat_extractor`.

3. **s3tokenizer_train** (causal S3 tokenizer export loader) — derived from ft_cosy / CosyVoice training tooling. Files: `s3tokenizer_train/{export,model,vq,causal_ops}.py`. Depends on OpenAI **whisper** (MIT) for `ResidualAttentionBlock` and for **log-mel** features fed into the causal S3 model (not the CosyVoice ONNX speech tokenizer).

4. **Emotion conditioning** — logic adapted from Marco-Voice `run_flow_stream_emosphere.py`; quantile stats `assets/emotion_quantiles/q1.pt`, `q3.pt`.

Runtime downloads (not shipped here):

- **emotion2vec_plus_seed** (ModelScope / FunASR)
- **audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim** (Hugging Face)

See `LICENSE` for this distribution’s license header (Apache-2.0 as in upstream Marco-Voice where applicable).
