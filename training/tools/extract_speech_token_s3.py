#!/usr/bin/env python3
"""Extract causal S3 speech tokens (1024 @ 25 Hz) into utt2speech_token.pt.

Same algorithm as Marco-Voice / ft_cosy reference: Whisper log-mel (128) + exported S3TokenizerV1.
Repo-root relative imports only (no external trees).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
import whisper
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_wav_scp(path: str) -> Dict[str, str]:
    utt2wav = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1]
    return utt2wav


def _wav_sort_key(wav_path: str) -> int:
    try:
        info = torchaudio.info(wav_path)
        return int(info.num_frames)
    except Exception:
        return 0


def _ensure_3d_mel(mel: torch.Tensor) -> torch.Tensor:
    if mel.ndim == 2:
        return mel.unsqueeze(0)
    if mel.ndim != 3:
        raise ValueError(f"unexpected mel shape: {tuple(mel.shape)}")
    return mel


def _load_mel_cpu(
    wav_path: str,
    n_mels: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resamplers[sr](audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    return _ensure_3d_mel(mel).cpu()


@torch.no_grad()
def _num_tokens_for_mel_T(
    model: torch.nn.Module,
    device: torch.device,
    n_mels: int,
    mel_t: int,
    cache: Dict[int, int],
) -> int:
    if mel_t <= 0:
        return 0
    if mel_t in cache:
        return cache[mel_t]
    z = torch.zeros(1, n_mels, mel_t, device=device)
    n = int(model.tokenize(z).shape[1])
    cache[mel_t] = n
    return n


@torch.no_grad()
def _run_batch(
    model: torch.nn.Module,
    device: torch.device,
    mels: List[torch.Tensor],
    utts: List[str],
    n_mels: int,
    token_len_cache: Dict[int, int],
    out: Dict[str, List[int]],
) -> None:
    lengths = [int(m.shape[2]) for m in mels]
    t_max = max(lengths)
    b = len(mels)
    batch = torch.zeros(b, n_mels, t_max, device=device, dtype=mels[0].dtype)
    for i, m in enumerate(mels):
        t = m.shape[2]
        batch[i, :, :t] = m.to(device, non_blocking=True)
    tok_b = model.tokenize(batch)
    for i, utt in enumerate(utts):
        n_tok = _num_tokens_for_mel_T(model, device, n_mels, lengths[i], token_len_cache)
        out[utt] = tok_b[i, :n_tok].detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="data dir containing wav.scp")
    parser.add_argument("--tokenizer_pt", required=True, help="exported causal S3 tokenizer .pt")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on CUDA (faster; batched vs unbatched may differ slightly).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    from s3tokenizer_train.export import S3Config, S3TokenizerV1  # noqa: E402

    ckpt = torch.load(args.tokenizer_pt, map_location="cpu", weights_only=False)
    model = S3TokenizerV1(S3Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    n_mels = int(model.config.n_mels)
    model = model.to(device).eval()

    wav_scp = os.path.join(args.dir, "wav.scp")
    utt2wav = load_wav_scp(wav_scp)
    items: List[Tuple[str, str]] = sorted(
        utt2wav.items(), key=lambda kv: (_wav_sort_key(kv[1]), kv[0])
    )
    out: Dict[str, List[int]] = {}
    token_len_cache: Dict[int, int] = {}
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    bs = max(1, int(args.batch_size))
    n = len(items)
    n_batch = (n + bs - 1) // bs

    if bs == 1:
        for utt, wav_path in tqdm(items, total=n, desc=f"tokenize:{os.path.basename(args.dir)}"):
            mel = _load_mel_cpu(wav_path, n_mels, resamplers)
            mel = mel.to(device)
            with torch.no_grad():
                tok = model.tokenize(mel)[0].cpu().tolist()
            out[utt] = tok
    else:
        for j in tqdm(range(n_batch), desc=f"tokenize:{os.path.basename(args.dir)}"):
            chunk = items[j * bs : (j + 1) * bs]
            mels = [_load_mel_cpu(p, n_mels, resamplers) for _, p in chunk]
            utts = [u for u, _ in chunk]
            _run_batch(model, device, mels, utts, n_mels, token_len_cache, out)

    out_path = os.path.join(args.dir, "utt2speech_token.pt")
    torch.save(out, out_path)
    print(f"saved: {out_path} ({len(out)} utts)")


if __name__ == "__main__":
    main()
