from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def causal_conv1d_full(conv: torch.nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """Run Conv1d with strict-causal left padding."""
    if x.ndim != 3:
        raise ValueError(f"expected 3D input (B,C,T), got shape={tuple(x.shape)}")
    left_pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
    x_pad = F.pad(x, (left_pad, 0))
    return F.conv1d(
        x_pad,
        conv.weight,
        conv.bias,
        stride=conv.stride[0],
        padding=0,
        dilation=conv.dilation[0],
        groups=conv.groups,
    )


def init_causal_conv1d_state(
    batch_size: int,
    channels: int,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor | int]:
    """Initialize stream state for causal_conv1d_step."""
    del kernel_size  # Kept for API stability with potential carry-based implementation.
    return {
        "batch_size": batch_size,
        "channels": channels,
        "x_hist": torch.zeros(batch_size, channels, 0, device=device, dtype=dtype),
        "emitted": 0,
    }


def causal_conv1d_step(
    conv: torch.nn.Conv1d,
    x_chunk: torch.Tensor,
    state: Dict[str, torch.Tensor | int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
    """Chunked causal conv by recomputing full output and emitting delta."""
    if x_chunk.ndim != 3:
        raise ValueError(f"expected x_chunk shape (B,C,T), got {tuple(x_chunk.shape)}")
    batch_size = int(state["batch_size"])
    channels = int(state["channels"])
    if x_chunk.shape[0] != batch_size:
        raise ValueError(f"batch mismatch: expected {batch_size}, got {x_chunk.shape[0]}")
    if x_chunk.shape[1] != channels:
        raise ValueError(f"channel mismatch: expected {channels}, got {x_chunk.shape[1]}")

    x_hist = state["x_hist"]
    if not torch.is_tensor(x_hist):
        raise TypeError("state['x_hist'] must be a tensor")
    x_hist = torch.cat([x_hist, x_chunk], dim=2)
    y_full = causal_conv1d_full(conv, x_hist)

    emitted = int(state["emitted"])
    y_new = y_full[:, :, emitted:]

    return y_new, {
        "batch_size": batch_size,
        "channels": channels,
        "x_hist": x_hist,
        "emitted": y_full.shape[2],
    }
