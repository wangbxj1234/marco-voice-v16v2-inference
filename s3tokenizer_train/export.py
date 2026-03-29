"""
Export trained S3Tokenizer: extract Encoder1 + VQ from the full WhisperWithVQ model.

The exported model can be used standalone for speech tokenization.

Usage:
  python -m s3tokenizer_train.export \
    --checkpoint exp/s3tokenizer_v1/final.pt \
    --output exp/s3tokenizer_v1/s3tokenizer.pt
"""

import argparse
import inspect
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.model import LayerNorm

from s3tokenizer_train.causal_ops import causal_conv1d_full
from s3tokenizer_train.model import S3Config, WhisperWithVQ
from s3tokenizer_train.vq import VectorQuantizer

logger = logging.getLogger(__name__)


class S3TokenizerV1(nn.Module):
    """
    Standalone S3Tokenizer for inference.
    Architecture: conv_stem -> 6 transformer blocks -> LN -> pre-VQ downsample -> VQ -> token_ids
    """

    def __init__(self, config: S3Config):
        super().__init__()
        self.config = config
        d = config.n_audio_state

        self.conv1 = nn.Conv1d(config.n_mels, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)

        self.register_buffer("positional_embedding",
                             torch.zeros(config.n_audio_ctx, d))

        from whisper.model import ResidualAttentionBlock
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(d, config.n_audio_head)
            for _ in range(config.n_encoder1_layers)
        ])
        for i, block in enumerate(self.blocks):
            sig = inspect.signature(block.forward)
            if "mask" not in sig.parameters:
                raise RuntimeError(
                    f"encoder block {i} does not support mask argument; strict-causal export requires mask-aware blocks"
                )
        self.ln = LayerNorm(d)
        self.pre_vq_downsample = nn.Conv1d(
            d,
            d,
            kernel_size=3,
            stride=config.pre_vq_stride,
            padding=1,
        )
        self.vq = VectorQuantizer(
            n_codes=config.n_codebook_size,
            dim=d,
            decay=config.vq_decay,
        )

    def _validate_positional_range(self, offset: int, length: int):
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if offset + length > self.positional_embedding.shape[0]:
            raise ValueError(
                f"stream exceeds n_audio_ctx: offset={offset}, length={length}, n_audio_ctx={self.positional_embedding.shape[0]}"
            )

    def _build_causal_attn_mask(self, t: int, device: torch.device, dtype: torch.dtype):
        mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def _encode_stage1_with_offset(self, mel: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = F.gelu(causal_conv1d_full(self.conv1, mel))
        x = F.gelu(causal_conv1d_full(self.conv2, x))
        x = x.permute(0, 2, 1)
        T = x.shape[1]
        self._validate_positional_range(offset, T)
        x = x + self.positional_embedding[offset: offset + T]
        mask = self._build_causal_attn_mask(T, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln(x)
        return x

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) - log mel spectrogram
        Returns:
            indices: (B, T') - speech token ids
        """
        x = self._encode_stage1_with_offset(mel, offset=0)
        x = x.transpose(1, 2)
        x = causal_conv1d_full(self.pre_vq_downsample, x)
        x = x.transpose(1, 2)
        _, indices, _ = self.vq(x)
        return indices

    @torch.no_grad()
    def tokenize(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference-mode tokenization."""
        return self.forward(mel)

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        return {
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
            "mel_hist": torch.zeros(batch_size, self.config.n_mels, 0, device=device, dtype=dtype),
            "emitted_tokens": 0,
            "audio_ctx": self.config.n_audio_ctx,
        }

    @torch.no_grad()
    def stream_step(self, mel_chunk: torch.Tensor, state: dict):
        if mel_chunk.ndim != 3:
            raise ValueError(f"mel_chunk must be 3D (B, n_mels, T), got {tuple(mel_chunk.shape)}")
        if mel_chunk.shape[2] == 0:
            raise ValueError("mel_chunk length must be > 0")
        if mel_chunk.shape[1] != self.config.n_mels:
            raise ValueError(f"expected n_mels={self.config.n_mels}, got {mel_chunk.shape[1]}")
        if mel_chunk.shape[0] != int(state["batch_size"]):
            raise ValueError(
                f"batch mismatch: expected {state['batch_size']}, got {mel_chunk.shape[0]}"
            )
        if mel_chunk.device != state["device"]:
            raise ValueError(f"device mismatch: expected {state['device']}, got {mel_chunk.device}")
        if mel_chunk.dtype != state["dtype"]:
            raise ValueError(f"dtype mismatch: expected {state['dtype']}, got {mel_chunk.dtype}")

        mel_hist = torch.cat([state["mel_hist"], mel_chunk], dim=2)
        stage1_t = (mel_hist.shape[2] + 1) // 2
        if stage1_t > int(state["audio_ctx"]):
            raise ValueError(
                f"stream exceeds n_audio_ctx: stage1_t={stage1_t}, n_audio_ctx={state['audio_ctx']}"
            )
        token_full = self.tokenize(mel_hist)
        emitted = int(state["emitted_tokens"])
        token_new = token_full[:, emitted:]
        new_state = {
            **state,
            "mel_hist": mel_hist,
            "emitted_tokens": token_full.shape[1],
        }
        return token_new, new_state

    @torch.no_grad()
    def finalize_stream(self, state: dict):
        batch = int(state["batch_size"])
        return torch.zeros(batch, 0, dtype=torch.long, device=state["device"]), state


def extract_tokenizer_state_dict(full_state_dict: dict) -> dict:
    """
    Extract Encoder1 + VQ weights from a WhisperWithVQ state dict.
    Maps keys from WhisperWithVQ -> S3TokenizerV1 naming.
    """
    mapping = {
        "conv1.": "conv1.",
        "conv2.": "conv2.",
        "positional_embedding": "positional_embedding",
        "encoder1_blocks.": "blocks.",
        "encoder1_ln.": "ln.",
        "pre_vq_downsample.": "pre_vq_downsample.",
        "vq.": "vq.",
    }

    new_state = OrderedDict()
    for key, val in full_state_dict.items():
        for src_prefix, dst_prefix in mapping.items():
            if key.startswith(src_prefix):
                new_key = dst_prefix + key[len(src_prefix):]
                new_state[new_key] = val
                break

    return new_state


def main():
    parser = argparse.ArgumentParser(description="Export S3Tokenizer from trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", required=True, help="Output path for standalone tokenizer")
    parser.add_argument("--n_encoder1_layers", type=int, default=6)
    parser.add_argument("--n_codebook_size", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    full_state = ckpt["model"] if "model" in ckpt else ckpt

    config = S3Config(
        n_encoder1_layers=args.n_encoder1_layers,
        n_codebook_size=args.n_codebook_size,
    )

    tokenizer_state = extract_tokenizer_state_dict(full_state)

    # Build standalone tokenizer and load weights
    tokenizer = S3TokenizerV1(config)
    missing, unexpected = tokenizer.load_state_dict(tokenizer_state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_data = {
        "config": vars(config),
        "model": tokenizer.state_dict(),
    }
    torch.save(export_data, args.output)
    logger.info(f"Exported S3Tokenizer to {args.output}")

    n_params = sum(p.numel() for p in tokenizer.parameters())
    logger.info(f"Tokenizer params: {n_params / 1e6:.1f}M")


if __name__ == "__main__":
    main()
