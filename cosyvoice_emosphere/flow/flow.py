# 

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice_emosphere.utils.mask import make_pad_mask
import numpy as np


# ---------------------------------------------------------------------------
# Auxiliary losses
# ---------------------------------------------------------------------------

def OrthogonalityLoss(speaker_embedding, emotion_embedding, weight: float = 0.1):
    """Penalise cosine overlap between speaker and emotion representations."""
    speaker_embedding_t = speaker_embedding.t()
    dot_product_matrix = torch.matmul(emotion_embedding, speaker_embedding_t)
    emotion_norms = torch.norm(emotion_embedding, dim=1, keepdim=True)
    speaker_norms = torch.norm(speaker_embedding, dim=1, keepdim=True).t()
    normalized_dot_product_matrix = dot_product_matrix / (emotion_norms * speaker_norms + 1e-8)
    ort_loss = torch.norm(normalized_dot_product_matrix, p='fro') ** 2

    cosine_sim = F.cosine_similarity(
        emotion_embedding.unsqueeze(2), speaker_embedding.unsqueeze(1), dim=-1
    )
    cosine_ort_loss = torch.norm(cosine_sim.mean(dim=-1), p='fro') ** 2

    return weight * (ort_loss + cosine_ort_loss)


def PaperContrastiveLoss(h_embeddings, e_embeddings):
    """Paper Eq. 5 — In-batch contrastive learning.

    Minimises |⟨h_i, e_j⟩| for all i ≠ j in the batch, where
      h_i = projected(speaker_i) + projected(emotion_i)
      e_j = projected(emotion_j)

    This encourages the combined speaker+emotion representation of each
    sample to be orthogonal to every *other* sample's pure emotion
    embedding, improving emotion separability and speaker-emotion
    disentanglement.

    Args:
        h_embeddings: (B, D) — projected speaker + projected emotion (summed).
        e_embeddings: (B, D) — projected emotion embeddings.
    Returns:
        scalar loss (0 when B ≤ 1).
    """
    device = h_embeddings.device
    B = h_embeddings.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device)

    # (B, B) dot-product matrix: <h_i, e_j>
    dots = torch.mm(h_embeddings, e_embeddings.t())

    # Exclude diagonal (i == j)
    mask = 1.0 - torch.eye(B, device=device)

    # Average |<h_i, e_j>| over all i ≠ j pairs
    loss = (torch.abs(dots) * mask).sum() / mask.sum()

    if torch.isnan(loss):
        return torch.tensor(0.0, device=device)
    return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class TokenDownsampler(torch.nn.Module):
    """Learnable temporal downsampling via strided convolution with residual skip.

    At initialization the conv weights are zero, so the output equals simple
    strided subsampling (``x[:, ::stride, :]``).  During training the conv
    learns a richer downsampling that preserves more information.

    This reduces the token frame rate (e.g. 50 Hz → 25 Hz for stride=2),
    directly cutting the number of discrete tokens that need to be transmitted.
    """

    def __init__(self, channels: int = 512, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              stride=stride, padding=kernel_size // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
        Args:
            x:      (B, T, C) — embedded token sequence
            x_lens: (B,)      — valid lengths before padding
        Returns:
            out:      (B, T', C)  where T' = ceil(T / stride)
            out_lens: (B,)
        """
        residual = x[:, ::self.stride, :]                       # strided skip
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)        # (B, T', C)
        out = residual + h                                       # identity at init
        out_lens = (x_lens + self.stride - 1) // self.stride
        return out, out_lens


# ---------------------------------------------------------------------------
# Paper Eq.6 — Emotion Cross-Attention
#   Q = W_q(e)       — emotion embedding → query
#   K = W_k(h_LM)    — speech tokens    → keys
#   V = W_v(h_LM)    — speech tokens    → values
#   h_attn = softmax(QK^T / sqrt(d_k)) V
# ---------------------------------------------------------------------------

class EmotionCrossAttention(nn.Module):
    """Emotion-queries-speech cross-attention (Paper §2.1 Eq.6).

    The single emotion vector (B, D_emo) attends over the full speech-token
    sequence (B, T, D_speech), producing a content-aware emotion modulation
    signal (B, D_emo) that is added back as a residual.
    """

    def __init__(self, emotion_dim: int, speech_dim: int,
                 num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Paper: W_q projects emotion, W_k / W_v project speech tokens
        self.to_q = nn.Linear(emotion_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(speech_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(speech_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, emotion_dim),
            nn.LayerNorm(emotion_dim),
        )

    def forward(self, emotion: torch.Tensor,
                speech_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion:       (B, D_emo)        — emotion embedding
            speech_tokens: (B, T, D_speech)  — encoded speech-token sequence
        Returns:
            h_attn: (B, D_emo) — content-aware emotion modulation
        """
        if emotion.dim() == 2:
            emotion = emotion.unsqueeze(1)          # (B, 1, D_emo)

        B = emotion.shape[0]
        S = 1                                       # single query token
        T = speech_tokens.shape[1]

        q = self.to_q(emotion)                      # (B, 1, inner_dim)
        k = self.to_k(speech_tokens)                # (B, T, inner_dim)
        v = self.to_v(speech_tokens)                # (B, T, inner_dim)

        # Multi-head reshape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, T, d)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, T, d)

        # Scaled dot-product attention  (Eq.6)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale          # (B, H, 1, T)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                                        # (B, H, 1, d)
        out = out.transpose(1, 2).reshape(B, S, -1)                        # (B, 1, inner)
        out = self.to_out(out)                                              # (B, 1, D_emo)

        return out.squeeze(1)                                               # (B, D_emo)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 Lort_losss: bool = True,
                 # --- v2 loss knobs ---
                 orth_loss_weight: float = 0.1,
                 contrastive_weight: float = 0.5,
                 contrastive_temperature: float = 0.07,
                 num_emotion_classes: int = 8,
                 emo_cls_weight: float = 0.0,
                 # --- v3 cross-attention (Paper Eq.6) ---
                 emotion_cross_attn_heads: int = 0,     # 0 = disabled
                 emotion_cross_attn_head_dim: int = 64,
                 # --- conditioning norm ---
                 emo_target_norm: float = 4.42,
                 # --- v15: trainable speech tokenizer ---
                 speech_tokenizer_onnx: str = '',
                 # --- v17: token temporal downsampling ---
                 token_downsample_stride: int = 1,
                 # ---------------------------------------------------------
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {
                     'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                     'cfm_params': DictConfig({
                         'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                     'decoder_params': {
                         'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {
                     'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.Lort_losss = Lort_losss
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

        # --- v2 loss config ---
        self.orth_loss_weight = orth_loss_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        # emotion embedding
        self.emo_VAD_inten_proj = nn.Linear(1, 2 * spk_embed_dim, bias=True)
        self.emosty_layer_norm = nn.LayerNorm(2 * spk_embed_dim)

        self.sty_proj = nn.Linear(spk_embed_dim, spk_embed_dim, bias=True)

        self.azimuth_bins = nn.Parameter(torch.linspace(-np.pi / 2, np.pi, 4), requires_grad=False)
        self.azimuth_emb = torch.nn.Embedding(4, spk_embed_dim // 2)
        self.elevation_bins = nn.Parameter(torch.linspace(np.pi / 2, np.pi, 2), requires_grad=False)
        self.elevation_emb = torch.nn.Embedding(2, spk_embed_dim // 2)

        self.spk_embed_proj = nn.Linear(512, spk_embed_dim, bias=True)
        self.emo_proj = nn.Linear(768, spk_embed_dim, bias=True)

        self.emo_mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            Mish(),
            torch.nn.Linear(1024, spk_embed_dim),
        )

        if self.Lort_losss:
            self.map_speaker_embedding = torch.nn.Linear(output_size, spk_embed_dim)

        # --- v3: Paper Eq.6 cross-attention (Q=emotion, K=V=speech) ---
        self.use_emotion_cross_attn = emotion_cross_attn_heads > 0
        if self.use_emotion_cross_attn:
            emotion_dim = 2 * spk_embed_dim   # 384
            speech_dim = output_size           # 80
            self.emotion_cross_attn = EmotionCrossAttention(
                emotion_dim=emotion_dim,
                speech_dim=speech_dim,
                num_heads=emotion_cross_attn_heads,
                head_dim=emotion_cross_attn_head_dim,
            )
            # Paper Eq.8: h_attn modulates h_LM at *every frame*.
            # Project emotion-dim h_attn → speech-dim so it can be added
            # as a per-frame residual to the encoder output h (B, T, 80).
            self.emotion_to_speech_proj = nn.Linear(emotion_dim, speech_dim)
            nn.init.zeros_(self.emotion_to_speech_proj.weight)
            nn.init.zeros_(self.emotion_to_speech_proj.bias)
            logging.info(f"[v3] EmotionCrossAttention enabled: "
                         f"Q({emotion_dim}) × K/V({speech_dim}), "
                         f"heads={emotion_cross_attn_heads}, "
                         f"head_dim={emotion_cross_attn_head_dim}, "
                         f"+ emotion_to_speech_proj({emotion_dim}→{speech_dim})")

        # Target norm for emo_all_emb — keeps conditioning magnitude stable.
        # v0 weights produce norm ~16.36; v5 had drifted to ~4.42.
        # Set via yaml `emo_target_norm` so each version can calibrate correctly.
        self.register_buffer('_emo_target_norm', torch.tensor(float(emo_target_norm)))

        # --- v15: Trainable speech tokenizer ---
        self.use_speech_tokenizer = bool(speech_tokenizer_onnx)
        if self.use_speech_tokenizer:
            import s3tokenizer
            self.speech_tokenizer_encoder = s3tokenizer.S3Tokenizer(
                name='speech_tokenizer_v1'
            )
            self.speech_tokenizer_encoder.init_from_onnx(speech_tokenizer_onnx)
            # We only use the encoder, drop the quantizer buffers from grad graph
            for p in self.speech_tokenizer_encoder.quantizer.parameters():
                p.requires_grad = False

            tokenizer_dim = self.speech_tokenizer_encoder.config.n_audio_state  # 1280
            self.tokenizer_proj = nn.Linear(tokenizer_dim, input_size)  # 1280 → 512

            logging.info(
                f"[v15] Speech tokenizer encoder loaded from {speech_tokenizer_onnx} "
                f"({sum(p.numel() for p in self.speech_tokenizer_encoder.encoder.parameters())/1e6:.1f}M params), "
                f"tokenizer_proj: {tokenizer_dim}→{input_size}"
            )

        # --- v17: token stride subsample ---
        # Directly discard every other token BEFORE embedding.
        # At deployment the sender only transmits 25 tokens/sec — real 300 bps.
        self.token_downsample_stride = token_downsample_stride
        if self.token_downsample_stride > 1:
            logging.info(
                f"[v17] Token stride subsample: stride={token_downsample_stride}, "
                f"effective rate {input_frame_rate}/{token_downsample_stride} "
                f"= {input_frame_rate // token_downsample_stride} Hz, "
                f"bitrate {input_frame_rate // token_downsample_stride * 12} bps "
                f"(no extra parameters)"
            )

    def init_tokenizer_proj_from_codebook(self):
        """Initialize tokenizer_proj so that continuous features → 512-dim
        approximates the old discrete path (quantize → input_embedding).

        Solves least-squares:  tokenizer_proj(normalize(codebook_k)) ≈ embedding_k
        for all k in [0, 4096).
        """
        if not self.use_speech_tokenizer:
            return
        codebook = self.speech_tokenizer_encoder.quantizer._codebook.embed  # (4096, 1280)
        embed = self.input_embedding.weight.data                             # (4096, 512)
        C = F.normalize(codebook.float(), p=2, dim=-1)  # VQ normalizes before lookup
        E = embed.float()
        # Solve C @ W^T = E  →  W^T = pinv(C) @ E  →  W = (pinv(C) @ E)^T
        solution = torch.linalg.lstsq(C, E).solution  # (1280, 512)
        self.tokenizer_proj.weight.data.copy_(solution.t())   # (512, 1280)
        self.tokenizer_proj.bias.data.zero_()
        logging.info("[v15] tokenizer_proj initialized from codebook-embedding least-squares fit")

    # -----------------------------------------------------------------------
    # Shared emotion pipeline → returns (emo_all_emb, emos_proj_embed)
    # -----------------------------------------------------------------------

    def _build_emotion_embedding(self, emotion_embedding, low_level_emo_embedding):
        """Process raw emotion vectors into the full emotion embedding.

        Returns:
            emo_all_emb:    (B, 2*spk_embed_dim)  — full emotion condition
            emos_proj_embed: (B, spk_embed_dim)    — from emo_mlp (for losses)
        """
        emos_proj_embed = self.emo_mlp(emotion_embedding)
        intens_embed = self.emo_VAD_inten_proj(low_level_emo_embedding[:, 0:1])

        ele_embed = 0
        elevation = low_level_emo_embedding[:, 1:2]
        elevation_index = torch.bucketize(elevation, self.elevation_bins)
        elevation_index = elevation_index.squeeze(1)
        elevation_embed = self.elevation_emb(elevation_index)
        ele_embed = elevation_embed + ele_embed

        azi_embed = 0
        azimuth = low_level_emo_embedding[:, 2:3]
        azimuth_index = torch.bucketize(azimuth, self.azimuth_bins)
        azimuth_index = azimuth_index.squeeze(1)
        azimuth_embed = self.azimuth_emb(azimuth_index)
        azi_embed = azimuth_embed + azi_embed

        style_embed = torch.cat((ele_embed, azi_embed), dim=-1)
        style_proj_embed = self.sty_proj(style_embed)

        # Softplus + LayerNorm
        combined_embedding = torch.cat((emos_proj_embed, style_proj_embed), dim=-1)
        emotion_out = F.softplus(combined_embedding)
        emosty_embed = self.emosty_layer_norm(emotion_out)
        emo_all_emb = intens_embed + emosty_embed

        # Normalize to fixed norm to prevent conditioning magnitude drift.
        # Direction encodes emotion; magnitude stays constant for decoder stability.
        if hasattr(self, '_emo_target_norm'):
            emo_norm = emo_all_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            emo_all_emb = emo_all_emb / emo_norm * self._emo_target_norm

        return emo_all_emb, emos_proj_embed

    # -----------------------------------------------------------------------
    # Training forward
    # -----------------------------------------------------------------------

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['utt_embedding'].to(device)
        low_level_emo_embedding = batch['low_level_emotion_embedding'].to(device)
        emotion_embedding = batch['emotion_embedding'].to(device)
        # --- speaker projection ---
        embedding = F.normalize(embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(embedding)

        # --- emotion pipeline ---
        emo_all_emb, emos_proj_embed = self._build_emotion_embedding(
            emotion_embedding, low_level_emo_embedding
        )

        # ------------------------------------------------------------------
        # Projected speaker embedding (needed for both orth and contrastive)
        # ------------------------------------------------------------------
        spk_embedding_ort = None
        if self.Lort_losss and hasattr(self, 'map_speaker_embedding'):
            spk_embedding_ort = self.map_speaker_embedding(spk_embedding)

        # ------------------------------------------------------------------
        # Loss 1: Orthogonality  (speaker ⊥ emotion)
        # ------------------------------------------------------------------
        if spk_embedding_ort is not None and self.orth_loss_weight > 0:
            lort_loss = OrthogonalityLoss(spk_embedding_ort, emos_proj_embed,
                                          weight=self.orth_loss_weight)
        else:
            lort_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------------------------
        # Loss 2: Paper Eq. 5 — In-batch contrastive
        #   h_i = proj(speaker_i) + proj(emotion_i),  e_j = proj(emotion_j)
        #   minimise |⟨h_i, e_j⟩| for all i ≠ j
        # ------------------------------------------------------------------
        if self.contrastive_weight > 0 and spk_embedding_ort is not None:
            h_combined = spk_embedding_ort + emos_proj_embed  # (B, D)
            con_loss = self.contrastive_weight * PaperContrastiveLoss(
                h_combined, emos_proj_embed
            )
        else:
            con_loss = torch.tensor(0.0, device=device)

        # --- encode speech tokens → mu ---
        if self.use_speech_tokenizer and 'tokenizer_mel' in batch:
            # v15 path: raw mel → S3Tokenizer encoder → continuous features
            tok_mel = batch['tokenizer_mel'].to(device)          # (B, 128, T_mel16k)
            tok_mel_len = batch['tokenizer_mel_len'].to(device)  # (B,)
            h_tok, h_tok_len = self.speech_tokenizer_encoder.encoder(tok_mel, tok_mel_len)
            h = self.tokenizer_proj(h_tok)                       # (B, T_tok, 512)
            mask = (~make_pad_mask(h_tok_len)).float().unsqueeze(-1).to(device)
            h = h * mask
            h, h_lengths = self.encoder(h, h_tok_len)
        else:
            # v17: subsample raw tokens BEFORE embedding (true compression)
            if self.token_downsample_stride > 1:
                token = token[:, ::self.token_downsample_stride]
                token_len = (token_len + self.token_downsample_stride - 1) // self.token_downsample_stride
            # legacy path: discrete tokens → embedding → encoder
            mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
            token = self.input_embedding(torch.clamp(token, min=0)) * mask
            h, h_lengths = self.encoder(token, token_len)

        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # ------------------------------------------------------------------
        # Paper Eq.6-8: Emotion cross-attention  Q=emotion, K=V=speech tokens
        # ------------------------------------------------------------------
        encoder_hidden_states = None
        if self.use_emotion_cross_attn:
            h_attn = self.emotion_cross_attn(emo_all_emb, h)          # (B, 384)
            
            # v5/v9: broadcast-add h_attn as a single residual to mu
            h_attn_proj = self.emotion_to_speech_proj(h_attn)     # (B, 80)
            h = h + h_attn_proj.unsqueeze(1)                      # (B, T, 80)

        # --- spk + emo concatenation → decoder global condition ---
        cond_embedding = torch.cat((spk_embedding, emo_all_emb), dim=-1)

        # --- get conditions ---
        conds = torch.zeros(feat.shape, device=device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)

        # ------------------------------------------------------------------
        # Loss 0: Mel reconstruction (CFM)
        # ------------------------------------------------------------------
        mel_loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            cond_embedding,
            cond=conds,
        )

        # L2 anchor: penalize deviation from init weights to prevent catastrophic forgetting
        anchor_loss = torch.tensor(0.0, device=device)
        if hasattr(self, '_anchor_weights') and self._anchor_weight > 0:
            for name, param in self.named_parameters():
                if name in self._anchor_weights and param.requires_grad:
                    ref = self._anchor_weights[name]
                    if ref.device != param.device:
                        ref = ref.to(param.device)
                        self._anchor_weights[name] = ref
                    anchor_loss = anchor_loss + ((param - ref) ** 2).sum()
            anchor_loss = self._anchor_weight * anchor_loss

        total_loss = mel_loss + lort_loss + con_loss + anchor_loss

        return {'loss': total_loss,
                'mel_loss': mel_loss.detach(),
                'orth_loss': lort_loss.detach() if torch.is_tensor(lort_loss) else lort_loss,
                'con_loss': con_loss.detach() if torch.is_tensor(con_loss) else con_loss,
                'anchor_loss': anchor_loss.detach() if torch.is_tensor(anchor_loss) else anchor_loss}

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  low_level_emo_embedding,
                  emotion_embedding,
                  flow_cache):
        if low_level_emo_embedding.ndim == 1:
            low_level_emo_embedding = low_level_emo_embedding.unsqueeze(0)
        if emotion_embedding.ndim == 1:
            emotion_embedding = emotion_embedding.unsqueeze(0)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        spk_embedding = self.spk_embed_affine_layer(embedding)

        # ensure dtype/device alignment
        low_level_emo_embedding = low_level_emo_embedding.to(spk_embedding.device, dtype=spk_embedding.dtype)
        emotion_embedding = emotion_embedding.to(spk_embedding.device, dtype=spk_embedding.dtype)

        # emotion pipeline
        emo_all_emb, _ = self._build_emotion_embedding(emotion_embedding, low_level_emo_embedding)

        # squeeze extra dims if needed
        if emo_all_emb.dim() == 3:
            emo_all_emb = emo_all_emb.squeeze(1)

        # v17: subsample raw tokens BEFORE embedding (true 25Hz compression)
        token_len2_orig_50hz = token.shape[1]
        if self.token_downsample_stride > 1:
            prompt_token = prompt_token[:, ::self.token_downsample_stride]
            prompt_token_len = (prompt_token_len + self.token_downsample_stride - 1) // self.token_downsample_stride
            token = token[:, ::self.token_downsample_stride]
            token_len = (token_len + self.token_downsample_stride - 1) // self.token_downsample_stride

        # concat prompt and source tokens (both at 25Hz if subsampled)
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(spk_embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # effective rate for length regulator
        effective_rate = self.input_frame_rate // self.token_downsample_stride if self.token_downsample_stride > 1 else self.input_frame_rate

        # encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        # mel_len2: original 50Hz count / 50Hz rate = correct duration
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2_orig_50hz / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(
            h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, effective_rate
        )

        # ------------------------------------------------------------------
        # Paper Eq.6-8: Emotion cross-attention
        # ------------------------------------------------------------------
        encoder_hidden_states = None
        if self.use_emotion_cross_attn:
            h_attn = self.emotion_cross_attn(emo_all_emb, h)          # (B, 384)
            
            h_attn_proj = self.emotion_to_speech_proj(h_attn)     # (B, 80)
            h = h + h_attn_proj.unsqueeze(1)                      # (B, T, 80)

        # spk + emo channel concat
        cond_embedding = torch.cat((spk_embedding, emo_all_emb), dim=-1)

        # get conditions
        conds = torch.zeros([token.shape[0], mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2] * token.shape[0]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=cond_embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, flow_cache
