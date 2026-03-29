"""
WhisperWithVQ: Whisper-Large-v3 with VQ layer inserted after encoder layer 6.

Training pipeline (Step 2):
  mel -> conv_stem -> encoder_blocks[:6] -> LN
      -> pre-VQ downsample (50 fps -> 25 fps)
      -> VQ
      -> post-VQ upsample (25 fps -> 50 fps)
      -> encoder_blocks[6:] -> LN -> decoder -> logits

Inference (tokenization):
  mel -> conv_stem -> encoder_blocks[:6] -> LN -> pre-VQ downsample -> VQ -> token_ids
"""

import logging
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.model import AudioEncoder, ResidualAttentionBlock, LayerNorm, sinusoids

from s3tokenizer_train.causal_ops import causal_conv1d_full
from s3tokenizer_train.vq import VectorQuantizer

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_encoder1_layers: int = 6
    n_encoder2_layers: int = 26  # 32 - 6
    n_codebook_size: int = 4096
    vq_decay: float = 0.99
    whisper_model: str = "large-v3"
    pre_vq_stride: int = 2


class WhisperWithVQ(nn.Module):
    """
    Full model for S3Tokenizer training.
    Wraps a Whisper model with VQ inserted between encoder layer 6 and 7.
    """

    def __init__(self, config: S3Config, whisper_model: Optional[whisper.Whisper] = None):
        super().__init__()
        self.config = config

        if whisper_model is None:
            logger.info(f"Loading Whisper model: {config.whisper_model}")
            whisper_model = whisper.load_model(config.whisper_model, device="cpu")

        encoder = whisper_model.encoder
        decoder = whisper_model.decoder

        # Conv stem (shared)
        self.conv1 = encoder.conv1
        self.conv2 = encoder.conv2
        self.register_buffer("positional_embedding", encoder.positional_embedding.clone())
        # Extra positional encoding used after VQ, matching the paper's Encoder2 input.
        self.register_buffer("positional_embedding_post_vq", encoder.positional_embedding.clone())

        # Split encoder blocks
        all_blocks = list(encoder.blocks)
        n1 = config.n_encoder1_layers
        self.encoder1_blocks = nn.ModuleList(all_blocks[:n1])
        self.encoder1_ln = LayerNorm(config.n_audio_state)
        for i, block in enumerate(self.encoder1_blocks):
            sig = inspect.signature(block.forward)
            if "mask" not in sig.parameters:
                raise RuntimeError(
                    f"encoder1 block {i} does not support mask argument; strict-causal mode requires mask-aware blocks"
                )

        self.encoder2_blocks = nn.ModuleList(all_blocks[n1:])
        self.encoder2_ln = encoder.ln_post

        # Copy encoder1_ln weights from a fresh LayerNorm (will be trained)
        nn.init.ones_(self.encoder1_ln.weight)
        nn.init.zeros_(self.encoder1_ln.bias)

        # VQ layer
        self.vq = VectorQuantizer(
            n_codes=config.n_codebook_size,
            dim=config.n_audio_state,
            decay=config.vq_decay,
        )
        # Learnable 50->25 fps downsample before VQ.
        self.pre_vq_downsample = nn.Conv1d(
            config.n_audio_state,
            config.n_audio_state,
            kernel_size=3,
            stride=config.pre_vq_stride,
            padding=1,
        )
        # Learnable 25->50 fps upsample after VQ.
        self.post_vq_upsample = nn.ConvTranspose1d(
            config.n_audio_state,
            config.n_audio_state,
            kernel_size=4,
            stride=config.pre_vq_stride,
            padding=1,
        )

        # Decoder (for ASR training)
        self.decoder = decoder

        # Store dims for convenience
        self.dims = whisper_model.dims

        logger.info(
            f"WhisperWithVQ initialized: encoder1={n1} layers, "
            f"encoder2={len(self.encoder2_blocks)} layers, "
            f"VQ={config.n_codebook_size} codes x {config.n_audio_state}d, "
            f"temporal 50->25->50 via conv/deconv"
        )

    def _validate_positional_range(self, offset: int, length: int):
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        max_len = self.positional_embedding.shape[0]
        if offset + length > max_len:
            raise ValueError(
                f"stream exceeds n_audio_ctx: offset={offset}, length={length}, n_audio_ctx={max_len}"
            )

    def _build_causal_attn_mask(self, t: int, device: torch.device, dtype: torch.dtype):
        mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def _encode_stage1_with_offset(self, mel: torch.Tensor, offset: int):
        x = F.gelu(causal_conv1d_full(self.conv1, mel))
        x = F.gelu(causal_conv1d_full(self.conv2, x))
        x = x.permute(0, 2, 1)  # (B, T, D)
        T = x.shape[1]
        self._validate_positional_range(offset, T)
        x = x + self.positional_embedding[offset: offset + T]
        mask = self._build_causal_attn_mask(T, x.device, x.dtype)
        for block in self.encoder1_blocks:
            x = block(x, mask=mask)
        return self.encoder1_ln(x)

    def encode_stage1(self, mel: torch.Tensor):
        """Conv stem + Encoder1 blocks + LayerNorm.

        Args:
            mel: (B, n_mels, T_mel) - log mel spectrogram
        Returns:
            h: (B, T, D) - encoder1 output
        """
        return self._encode_stage1_with_offset(mel, offset=0)

    def encode_stage2(self, x: torch.Tensor):
        """Encoder2 blocks + LayerNorm.

        Args:
            x: (B, T, D) - VQ output
        Returns:
            h: (B, T, D) - full encoder output
        """
        T = x.shape[1]
        x = x + self.positional_embedding_post_vq[:T]
        for block in self.encoder2_blocks:
            x = block(x)
        x = self.encoder2_ln(x)
        return x

    def downsample_pre_vq(self, x: torch.Tensor):
        """Learnable temporal downsample before VQ: (B, T, D) -> (B, T/2, D)."""
        x = x.transpose(1, 2)  # (B, D, T)
        x = causal_conv1d_full(self.pre_vq_downsample, x)
        return x.transpose(1, 2)  # (B, T/2, D)

    def upsample_post_vq(self, x: torch.Tensor, target_len: int):
        """Learnable temporal upsample after VQ: (B, T/2, D) -> (B, T, D)."""
        x = x.transpose(1, 2)  # (B, D, T/2)
        x = self.post_vq_upsample(x)
        x = x.transpose(1, 2)  # (B, T', D)
        # Align to encoder2 expected length (typically 1500) if off by one.
        if x.shape[1] > target_len:
            x = x[:, :target_len, :]
        elif x.shape[1] < target_len:
            x = F.pad(x, (0, 0, 0, target_len - x.shape[1]))
        return x

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor):
        """
        Full forward pass for training.

        Args:
            mel: (B, n_mels, T_mel) - log mel spectrogram (padded to 3000 frames)
            tokens: (B, S) - text token ids with SOT/EOT markers
        Returns:
            logits: (B, S, vocab_size)
            vq_commit_loss: scalar
            vq_indices: (B, T) - for monitoring codebook usage
        """
        h1 = self.encode_stage1(mel)  # 50 fps
        h1_ds = self.downsample_pre_vq(h1)  # 25 fps
        quantized, vq_indices, commit_loss = self.vq(h1_ds)
        h2_in = self.upsample_post_vq(quantized, target_len=h1.shape[1])  # back to 50 fps
        h2 = self.encode_stage2(h2_in)
        logits = self.decoder(tokens, h2)
        return logits, commit_loss, vq_indices

    @torch.no_grad()
    def tokenize(self, mel: torch.Tensor):
        """
        Extract speech tokens (inference only).

        Args:
            mel: (B, n_mels, T_mel)
        Returns:
            indices: (B, T) - speech token ids
        """
        h1 = self.encode_stage1(mel)
        h1_ds = self.downsample_pre_vq(h1)
        return self.vq.quantize(h1_ds)

    def init_stream_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
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

        mel_hist = state["mel_hist"]
        mel_hist = torch.cat([mel_hist, mel_chunk], dim=2)

        # conv2 is stride=2, so stage1 time dimension is approximately ceil(T/2)
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
        return (
            torch.zeros(batch, 0, dtype=torch.long, device=state["device"]),
            state,
        )

    def get_param_groups(self, lr_encoder1: float = 1e-4, lr_vq: float = 1e-4,
                         lr_encoder2: float = 1e-5, lr_decoder: float = 1e-5):
        """Differential learning rates for different model components."""
        encoder1_params = []
        for m in [
            self.conv1,
            self.conv2,
            self.encoder1_blocks,
            self.encoder1_ln,
            self.pre_vq_downsample,
            self.post_vq_upsample,
        ]:
            encoder1_params.extend(p for p in m.parameters() if p.requires_grad)
        # positional_embedding is a buffer, not a parameter

        vq_params = [p for p in self.vq.parameters() if p.requires_grad]

        encoder2_params = []
        for m in [self.encoder2_blocks, self.encoder2_ln]:
            encoder2_params.extend(p for p in m.parameters() if p.requires_grad)

        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]

        return [
            {"params": encoder1_params, "lr": lr_encoder1, "name": "encoder1"},
            {"params": vq_params, "lr": lr_vq, "name": "vq"},
            {"params": encoder2_params, "lr": lr_encoder2, "name": "encoder2"},
            {"params": decoder_params, "lr": lr_decoder, "name": "decoder"},
        ]


def build_model(config: Optional[S3Config] = None, whisper_cache_dir: Optional[str] = None) -> WhisperWithVQ:
    """Build and return a WhisperWithVQ model."""
    if config is None:
        config = S3Config()

    logger.info(f"Loading Whisper {config.whisper_model}...")
    wm = whisper.load_model(
        config.whisper_model,
        device="cpu",
        download_root=whisper_cache_dir,
    )
    model = WhisperWithVQ(config, whisper_model=wm)
    del wm
    return model
