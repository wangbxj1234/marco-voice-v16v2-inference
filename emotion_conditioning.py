# Emotion vectors for Emosphere flow VC (from Marco-Voice run_flow_stream_emosphere.py).
# SPDX: Apache-2.0 (same as parent project)

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

_ASSETS = Path(__file__).resolve().parent / "assets" / "emotion_quantiles"

_emotion2vec_model = None
_emo_wav2vec2_model = None
_emo_wav2vec2_processor = None
_q1_dict = None
_q3_dict = None

_EMO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAD_CENTERS = {
    "Angry": torch.tensor([0.37068613, 0.4814421, 0.37709341]),
    "Happy": torch.tensor([0.36792182, 0.48303542, 0.34562907]),
    "Neutral": torch.tensor([0.4135, 0.5169, 0.3620]),
    "Sad": torch.tensor([0.48519272, 0.57727589, 0.39524114]),
    "Surprise": torch.tensor([0.36877737, 0.48431942, 0.36644742]),
}


def _get_emotion2vec():
    global _emotion2vec_model
    if _emotion2vec_model is None:
        from funasr import AutoModel

        _emotion2vec_model = AutoModel(model="iic/emotion2vec_plus_seed", hub="ms")
    return _emotion2vec_model


def _get_wav2vec2_emo():
    global _emo_wav2vec2_model, _emo_wav2vec2_processor
    if _emo_wav2vec2_model is None:
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class _RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class _EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = _RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = torch.mean(outputs[0], dim=1)
                logits = self.classifier(hidden_states)
                return hidden_states, logits

        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        _emo_wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
        _emo_wav2vec2_model = _EmotionModel.from_pretrained(model_name).to(_EMO_DEVICE)
    return _emo_wav2vec2_model, _emo_wav2vec2_processor


def _get_q1_q3():
    global _q1_dict, _q3_dict
    if _q1_dict is None:
        _q1_dict = torch.load(_ASSETS / "q1.pt", map_location="cpu", weights_only=False)
        _q3_dict = torch.load(_ASSETS / "q3.pt", map_location="cpu", weights_only=False)
    return _q1_dict, _q3_dict


def extract_emotion2vec_768(wav_path: str) -> torch.Tensor:
    m = _get_emotion2vec()
    result = m.generate(
        wav_path,
        output_dir=os.environ.get("EMOTION2VEC_CACHE", "./.cache_emotion2vec"),
        granularity="utterance",
        extract_embedding=True,
    )
    feats = result[0]["feats"]
    if isinstance(feats, np.ndarray):
        return torch.from_numpy(feats).to(torch.float32).flatten()
    return torch.tensor(feats, dtype=torch.float32).flatten()


def _vad_to_polar(vad_diff: np.ndarray, q1_val, q3_val, emo_id: str) -> np.ndarray:
    t = torch.tensor(vad_diff, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    r = torch.sqrt(torch.sum(t**2, dim=1))
    q1_t = torch.tensor(q1_val, dtype=torch.float32) if not isinstance(q1_val, torch.Tensor) else q1_val.float()
    q3_t = torch.tensor(q3_val, dtype=torch.float32) if not isinstance(q3_val, torch.Tensor) else q3_val.float()
    iqr = (q3_t - q1_t) * 1.5
    r_clamp = torch.clamp(r, min=(q1_t - iqr).item(), max=(q3_t + iqr).item())
    r_norm = (r_clamp - (q1_t - iqr)) / ((q3_t + iqr) - (q1_t - iqr))
    theta = torch.acos(torch.clamp(t[:, 2] / (r + 1e-8), -1, 1))
    phi = torch.atan2(t[:, 1], t[:, 0])
    if r_norm.item() == 0:
        theta = torch.zeros_like(theta)
        phi = torch.zeros_like(phi)
    if emo_id == "Neutral":
        r_norm = torch.zeros_like(r_norm)
        theta = torch.zeros_like(theta)
        phi = torch.zeros_like(phi)
    return torch.stack((r_norm, theta, phi), dim=1).squeeze(0).cpu().numpy()


def extract_low_level_emo(wav_path: str, emo_id: str = "auto") -> torch.Tensor:
    emo_model, emo_processor = _get_wav2vec2_emo()
    audio, sr = torchaudio.load(wav_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

    y = emo_processor(audio.numpy(), sampling_rate=16000)
    y = torch.from_numpy(y["input_values"][0].reshape(1, -1)).to(_EMO_DEVICE)
    with torch.no_grad():
        _, vad_logits = emo_model(y)
        vad_raw = vad_logits.detach().cpu().numpy()

    if emo_id == "auto":
        vad_t = torch.tensor(vad_raw, dtype=torch.float32)
        best_emo, best_dist = "Neutral", float("inf")
        for name, center in VAD_CENTERS.items():
            d = torch.norm(vad_t - center.unsqueeze(0)).item()
            if d < best_dist:
                best_dist, best_emo = d, name
        emo_id = best_emo
        print(f"  [auto-detect] closest emotion: {emo_id} (dist={best_dist:.4f})")

    vad_center = VAD_CENTERS[emo_id].cpu().numpy()
    vad_diff = vad_raw - vad_center

    q1_dict, q3_dict = _get_q1_q3()
    polar = _vad_to_polar(vad_diff, q1_dict[emo_id], q3_dict[emo_id], emo_id)
    return torch.tensor(polar, dtype=torch.float32)
