#!/usr/bin/env python3
"""Recompute speech_token from parquet audio_data (embedded WAV bytes). All deps in-repo."""
from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torchaudio
import whisper
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _ensure_3d_mel(mel: torch.Tensor) -> torch.Tensor:
    if mel.ndim == 2:
        return mel.unsqueeze(0)
    if mel.ndim != 3:
        raise ValueError(f"unexpected mel shape: {tuple(mel.shape)}")
    return mel


def _load_mel_from_bytes(
    raw: bytes,
    resamplers: Dict[int, torchaudio.transforms.Resample],
    max_sec: float = 30.0,
) -> torch.Tensor:
    audio, sr = torchaudio.load(BytesIO(raw))
    audio = audio.mean(dim=0, keepdim=True)
    max_samples = int(max_sec * sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    if sr != 16000:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resamplers[sr](audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128)
    return _ensure_3d_mel(mel).cpu()


@torch.no_grad()
def _num_tokens_for_mel_T(model, device, n_mels: int, mel_t: int, cache: Dict[int, int]) -> int:
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
    model,
    device,
    mels: List[torch.Tensor],
    rows_idx: List[int],
    n_mels: int,
    token_len_cache: Dict[int, int],
    out_tokens: List,
) -> None:
    lengths = [int(m.shape[2]) for m in mels]
    t_max = max(lengths)
    b = len(mels)
    batch = torch.zeros(b, n_mels, t_max, device=device, dtype=mels[0].dtype)
    for i, m in enumerate(mels):
        t = m.shape[2]
        batch[i, :, :t] = m.to(device, non_blocking=True)
    tok_b = model.tokenize(batch)
    for j, irow in enumerate(rows_idx):
        n_tok = _num_tokens_for_mel_T(model, device, n_mels, lengths[j], token_len_cache)
        out_tokens[irow] = tok_b[j, :n_tok].detach().cpu().tolist()


def retokenize_shard(
    in_path: str,
    out_path: str,
    model,
    device: torch.device,
    batch_size: int,
    allow_tf32: bool,
) -> int:
    if device.type == "cuda" and not allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    df = pq.read_table(in_path).to_pandas()
    n = len(df)
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}
    mels: List[torch.Tensor] = []
    idxs: List[int] = []
    out_tokens: List = [None] * n
    token_len_cache: Dict[int, int] = {}
    n_mels = int(model.config.n_mels)

    for i in range(n):
        raw = df.loc[i, "audio_data"]
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        mel = _load_mel_from_bytes(bytes(raw), resamplers)
        mels.append(mel)
        idxs.append(i)
        if len(mels) >= batch_size or i == n - 1:
            order = sorted(range(len(mels)), key=lambda k: mels[k].shape[2], reverse=True)
            m_b = [mels[k] for k in order]
            i_b = [idxs[k] for k in order]
            _run_batch(model, device, m_b, i_b, n_mels, token_len_cache, out_tokens)
            mels = []
            idxs = []

    df["speech_token"] = [np.array(out_tokens[i], dtype=np.int64) for i in range(n)]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path)
    return n


def main() -> None:
    from s3tokenizer_train.export import S3Config, S3TokenizerV1  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_list", required=True, help="data.list of input .tar shards")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_list", required=True)
    ap.add_argument("--tokenizer_pt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--allow_tf32", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.tokenizer_pt, map_location="cpu", weights_only=False)
    model = S3TokenizerV1(S3Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    with open(args.in_list) as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    out_paths = []
    total = 0
    for p in tqdm(paths, desc="shards"):
        base = os.path.basename(p)
        op = os.path.join(args.out_dir, base)
        n = retokenize_shard(p, op, model, device, args.batch_size, args.allow_tf32)
        total += n
        out_paths.append(os.path.abspath(op))

    os.makedirs(os.path.dirname(args.out_list) or ".", exist_ok=True)
    with open(args.out_list, "w", encoding="utf-8") as f:
        for p in out_paths:
            f.write(p + "\n")
    print(f"Done {len(out_paths)} shards, {total} utts -> {args.out_list}")


if __name__ == "__main__":
    main()
