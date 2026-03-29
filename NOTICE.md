# Third-party notices

This repository vendors minimal inference-time code and assets from:

1. **CosyVoice / cosyvoice_emosphere** (Alibaba et al.) — Apache-2.0. Source: Marco-Voice `Models/marco_voice/cosyvoice_emosphere/` (training binaries under `bin/` omitted).

2. **Matcha-TTS** — MIT. `third_party/Matcha-TTS/` used for `matcha.utils.audio.mel_spectrogram` referenced by HyperPyYAML configs.

3. **s3tokenizer_train** (causal S3 tokenizer export loader) — derived from ft_cosy / CosyVoice training tooling. Files: `s3tokenizer_train/{export,model,vq,causal_ops}.py`. Depends on OpenAI **whisper** (MIT) for `ResidualAttentionBlock` / mel.

4. **Emotion conditioning** — logic adapted from Marco-Voice `run_flow_stream_emosphere.py`; quantile stats `assets/emotion_quantiles/q1.pt`, `q3.pt`.

`speech_tokenizer_v1.onnx` is **not required** for the default `infer.py` path (causal S3 tokens only). It remains supported when present for CosyVoice `inference_vc` / ONNX token extraction.

Runtime downloads (not shipped here):

- **emotion2vec_plus_seed** (ModelScope / FunASR)
- **audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim** (Hugging Face)

See `LICENSE` for this distribution’s license header (Apache-2.0 as in upstream Marco-Voice where applicable).
