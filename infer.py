#!/usr/bin/env python3
"""
v16 v2 inference: causal S3 tokenizer (.pt) + Emosphere flow (1024 @ 25 Hz).

Loads only flow.pt, hift.pt, campplus.onnx + configs/cosyvoice.yaml (no LLM, no CosyVoice text/ONNX tokenizers).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _setup_paths() -> None:
    r = str(_ROOT)
    matcha = str(_ROOT / "third_party" / "Matcha-TTS")
    for p in (r, matcha):
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_paths()

import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from hyperpyyaml import load_hyperpyyaml

from cosyvoice_emosphere.cli.model import CosyVoiceModel
from emotion_conditioning import extract_emotion2vec_768, extract_low_level_emo


def load_wav(wav: str | os.PathLike, target_sr: int) -> torch.Tensor:
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f"wav sr {sample_rate} must be >= {target_sr}"
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def _ensure_3d_mel(mel: torch.Tensor) -> torch.Tensor:
    if mel.ndim == 2:
        return mel.unsqueeze(0)
    if mel.ndim != 3:
        raise ValueError(f"unexpected mel shape: {tuple(mel.shape)}")
    return mel


@torch.no_grad()
def _num_tokens_for_mel_T(model, device, n_mels: int, mel_t: int, cache: dict) -> int:
    if mel_t <= 0:
        return 0
    if mel_t in cache:
        return cache[mel_t]
    z = torch.zeros(1, n_mels, mel_t, device=device)
    n = int(model.tokenize(z).shape[1])
    cache[mel_t] = n
    return n


@torch.no_grad()
def wav16k_to_s3_tokens(
    speech_16k: torch.Tensor,
    s3_model: torch.nn.Module,
    device: torch.device,
    token_len_cache: dict | None = None,
) -> torch.Tensor:
    if speech_16k.shape[1] / 16000.0 > 30.0:
        raise ValueError("audio longer than 30s not supported")
    mel = whisper.log_mel_spectrogram(speech_16k, n_mels=128)
    mel = _ensure_3d_mel(mel)
    tok = s3_model.tokenize(mel.to(device))
    cache = token_len_cache if token_len_cache is not None else {}
    n_mels = int(s3_model.config.n_mels)
    n_tok = _num_tokens_for_mel_T(s3_model, device, n_mels, mel.shape[2], cache)
    t = tok[0, :n_tok].detach().cpu().long().clamp(0, 1023)
    return t.unsqueeze(0)


def load_s3_tokenizer(tokenizer_pt: str, device: torch.device):
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    ckpt = torch.load(tokenizer_pt, map_location="cpu", weights_only=False)
    from s3tokenizer_train.export import S3Config, S3TokenizerV1  # noqa: E402

    model = S3TokenizerV1(S3Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


def _campplus_session(campplus_onnx: str) -> ort.InferenceSession:
    option = ort.SessionOptions()
    option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    return ort.InferenceSession(campplus_onnx, sess_options=option, providers=["CPUExecutionProvider"])


@torch.no_grad()
def extract_spk_embedding(speech_16k: torch.Tensor, session: ort.InferenceSession, device: torch.device) -> torch.Tensor:
    feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    name = session.get_inputs()[0].name
    out = session.run(None, {name: feat.unsqueeze(0).cpu().numpy()})[0].flatten().tolist()
    return torch.tensor([out], device=device)


@torch.no_grad()
def extract_prompt_mel(
    speech_22050: torch.Tensor,
    feat_extractor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    speech_feat = feat_extractor(speech_22050).squeeze(0).transpose(0, 1).to(device)
    speech_feat = speech_feat.unsqueeze(0)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32, device=device)
    return speech_feat, speech_feat_len


def load_vc_stack(weights_dir: Path) -> tuple[CosyVoiceModel, callable]:
    cfg_path = weights_dir / "cosyvoice.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path} (copy from configs/ or run scripts/download_weights.py)")
    with open(cfg_path, encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)
    dummy_llm = torch.nn.Module()
    fm = CosyVoiceModel(dummy_llm, configs["flow"], configs["hift"], fp16=False)
    flow_pt = weights_dir / "flow.pt"
    hift_pt = weights_dir / "hift.pt"
    fm.load(None, str(flow_pt), str(hift_pt))
    return fm, configs["feat_extractor"]


def prepare_weights_dir(weights_dir: Path, config_src: Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst_yaml = weights_dir / "cosyvoice.yaml"
    if not dst_yaml.exists():
        import shutil

        shutil.copy2(config_src, dst_yaml)


def main() -> None:
    ap = argparse.ArgumentParser(description="Marco Voice v16 v2 causal-S3 VC / reconstruction (flow+hift only)")
    ap.add_argument(
        "--weights_dir",
        type=Path,
        default=_ROOT / "weights",
        help="Directory with cosyvoice.yaml, hift.pt, flow.pt, campplus.onnx",
    )
    ap.add_argument("--tokenizer_pt", type=Path, default=None, help="Causal S3 tokenizer export .pt")
    ap.add_argument("--prompt_wav", type=Path, default=None)
    ap.add_argument("--source_wav", type=Path, default=None, help="Default: same as prompt (self-reconstruction)")
    ap.add_argument("--out_wav", type=Path, default=_ROOT / "outputs" / "reconstruction.wav")
    ap.add_argument("--flow_ckpt", type=Path, default=None, help="Override flow weights (else weights_dir/flow.pt)")
    ap.add_argument("--hop_tokens", type=int, default=0)
    ap.add_argument(
        "--emo_id",
        type=str,
        default="auto",
        choices=["Angry", "Happy", "Neutral", "Sad", "Surprise", "auto"],
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt_max_seconds", type=float, default=0.0)
    ap.add_argument(
        "--smoke_imports",
        action="store_true",
        help="Only verify imports and exit (no weights required)",
    )
    args = ap.parse_args()

    if args.smoke_imports:
        print("smoke_imports: OK (CosyVoiceModel, s3tokenizer_train, emotion_conditioning)")
        return

    if args.tokenizer_pt is None or args.prompt_wav is None:
        ap.error("inference requires --tokenizer_pt and --prompt_wav (or use --smoke_imports)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_path = args.source_wav or args.prompt_wav

    prepare_weights_dir(args.weights_dir, _ROOT / "configs" / "cosyvoice.yaml")

    for name in ("hift.pt", "flow.pt", "campplus.onnx"):
        p = args.weights_dir / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run: python scripts/download_weights.py --manifest weights_manifest.json\n"
                "See weights_manifest.example.json and README.md."
            )

    if not args.tokenizer_pt.is_file():
        raise FileNotFoundError(args.tokenizer_pt)
    if not args.prompt_wav.is_file():
        raise FileNotFoundError(args.prompt_wav)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    args.out_wav.parent.mkdir(parents=True, exist_ok=True)

    tok_abs = args.tokenizer_pt.resolve()
    print(f"[causal_s3] loading tokenizer: {tok_abs}")
    s3_model = load_s3_tokenizer(str(tok_abs), device)
    tok_cache: dict = {}

    print(f"Loading flow+hift from {args.weights_dir.resolve()} ...")
    fm, feat_extractor = load_vc_stack(args.weights_dir)
    campplus_path = str((args.weights_dir / "campplus.onnx").resolve())
    spk_session = _campplus_session(campplus_path)

    if args.flow_ckpt is not None:
        ckpt = torch.load(args.flow_ckpt, map_location="cpu", weights_only=False)
        fm.flow.load_state_dict(ckpt, strict=False)
        fm.flow.eval()
        if device.type == "cuda":
            fm.flow.cuda()

    if args.hop_tokens > 0:
        h = int(args.hop_tokens)
        fm.token_min_hop_len = h
        fm.token_max_hop_len = h * 2

    prompt_16k = load_wav(args.prompt_wav, 16000)
    source_16k = load_wav(source_path, 16000)
    if args.prompt_max_seconds and args.prompt_max_seconds > 0:
        max_n = int(16000 * float(args.prompt_max_seconds))
        if prompt_16k.shape[1] > max_n:
            prompt_16k = prompt_16k[:, :max_n]

    prompt_22050 = torchaudio.transforms.Resample(16000, 22050)(prompt_16k)
    prompt_feat, _ = extract_prompt_mel(prompt_22050, feat_extractor, device)
    flow_embedding = extract_spk_embedding(prompt_16k, spk_session, device)

    prompt_tok = wav16k_to_s3_tokens(prompt_16k, s3_model, device, tok_cache)
    source_tok = wav16k_to_s3_tokens(source_16k, s3_model, device, tok_cache)
    print(f"prompt tokens: {prompt_tok.shape[1]}, source tokens: {source_tok.shape[1]}")

    print(f"Emotion from {source_path} ...")
    emotion_vec = extract_emotion2vec_768(str(source_path))
    low_level_emo = extract_low_level_emo(str(source_path), emo_id=args.emo_id)

    chunks = []
    for out in fm.vc(
        source_speech_token=source_tok,
        flow_prompt_speech_token=prompt_tok,
        prompt_speech_feat=prompt_feat,
        flow_embedding=flow_embedding,
        low_level_emo_embedding=low_level_emo,
        emotion_embedding=emotion_vec,
        stream=False,
    ):
        chunks.append(out["tts_speech"])

    if not chunks:
        raise RuntimeError("Model produced no audio chunks.")
    full = torch.cat(chunks, dim=1)
    torchaudio.save(str(args.out_wav), full, 22050)
    print(f"Saved: {args.out_wav.resolve()}")


if __name__ == "__main__":
    main()
