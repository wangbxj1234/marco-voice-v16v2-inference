"""
Vector Quantization with EMA codebook update, following the S3Tokenizer V1 design.

Architecture:
  - EuclideanCodebook: 4096 codes x 1280-dim
  - Inputs are L2-normalized before nearest-neighbor lookup
  - Codebook updated via Exponential Moving Average (EMA)
  - Gradients pass through via Straight-Through Estimator (STE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanCodebook(nn.Module):
    def __init__(self, n_codes: int = 4096, dim: int = 1280, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_codes, dim)
        nn.init.uniform_(embed, -1.0 / n_codes, 1.0 / n_codes)
        embed = F.normalize(embed, p=2, dim=-1)

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(n_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B*T, D) - L2-normalized input vectors
        Returns:
            quantized: (B*T, D) - quantized vectors (with STE gradient)
            indices: (B*T,) - codebook indices
            commit_loss: scalar - commitment loss (for optional use)
        """
        # Nearest neighbor via negative Euclidean distance
        # dist = -||x - e||^2 = -(||x||^2 - 2*x@e^T + ||e||^2)
        # Since x is L2-normalized, ||x||^2 = 1
        dist = -(
            x.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * x @ self.embed.t()
            + self.embed.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dist.max(dim=-1).indices  # (B*T,)
        quantized = F.embedding(indices, self.embed)  # (B*T, D)

        if self.training:
            self._ema_update(x, indices)

        # Commitment loss (optional, for monitoring)
        commit_loss = F.mse_loss(x.detach(), quantized)

        # Straight-through estimator: copy gradients from quantized to x
        quantized = x + (quantized - x).detach()

        return quantized, indices, commit_loss

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.n_codes).float()  # (B*T, n_codes)
        n_i = one_hot.sum(dim=0)  # (n_codes,)
        sum_i = one_hot.t() @ x  # (n_codes, D)

        self.cluster_size.mul_(self.decay).add_(n_i, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(sum_i, alpha=1 - self.decay)

        # Laplace smoothing
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        )
        self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))


class VectorQuantizer(nn.Module):
    """Wraps EuclideanCodebook with L2 normalization and reshaping."""

    def __init__(self, n_codes: int = 4096, dim: int = 1280, decay: float = 0.99):
        super().__init__()
        self.codebook = EuclideanCodebook(n_codes=n_codes, dim=dim, decay=decay)
        self.dim = dim
        self.n_codes = n_codes

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D) - encoder hidden states
        Returns:
            quantized: (B, T, D) - quantized hidden states (STE gradient)
            indices: (B, T) - token indices
            commit_loss: scalar
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        flat_norm = F.normalize(flat, p=2, dim=-1)

        quantized, indices, commit_loss = self.codebook(flat_norm)

        # Denormalize: scale quantized back to the original magnitude
        # The codebook stores normalized vectors, but the downstream encoder
        # expects the original scale. We use the input's norm for this.
        input_norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quantized = quantized * input_norms

        quantized = quantized.reshape(B, T, D)
        indices = indices.reshape(B, T)

        return quantized, indices, commit_loss

    def quantize(self, x: torch.Tensor):
        """Inference-only: returns just token indices."""
        with torch.no_grad():
            _, indices, _ = self.forward(x)
        return indices
