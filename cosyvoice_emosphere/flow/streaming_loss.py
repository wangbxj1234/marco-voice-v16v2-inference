import torch
from torch.nn import functional as F


def tokens_to_mel_frames(tokens: int, input_frame_rate: int, sample_rate: int = 22050, hop_size: int = 256) -> int:
    if tokens <= 0:
        return 0
    frames = int(tokens / float(input_frame_rate) * float(sample_rate) / float(hop_size))
    return max(1, frames)


def build_stream_chunk_pairs(total_frames: int, hop_frames: int, overlap_frames: int, max_pairs: int) -> list[tuple[int, int, int, int]]:
    pairs: list[tuple[int, int, int, int]] = []
    if total_frames <= 0 or hop_frames <= 0 or overlap_frames <= 0 or max_pairs <= 0:
        return pairs
    start = 0
    while len(pairs) < max_pairs:
        left_s = start
        left_e = start + hop_frames + overlap_frames
        right_s = start + hop_frames
        right_e = start + (2 * hop_frames) + overlap_frames
        if right_e > total_frames:
            break
        pairs.append((left_s, left_e, right_s, right_e))
        start += hop_frames
    return pairs


def overlap_consistency_l1(left_chunk: torch.Tensor, right_chunk: torch.Tensor, overlap_frames: int) -> torch.Tensor:
    if overlap_frames <= 0:
        return torch.tensor(0.0, device=left_chunk.device, dtype=left_chunk.dtype)
    if left_chunk.shape[-1] < overlap_frames or right_chunk.shape[-1] < overlap_frames:
        return torch.tensor(0.0, device=left_chunk.device, dtype=left_chunk.dtype)
    left_overlap = left_chunk[:, :, -overlap_frames:]
    right_overlap = right_chunk[:, :, :overlap_frames]
    return F.l1_loss(left_overlap, right_overlap)


def boundary_continuity_l1(left_chunk: torch.Tensor, right_chunk: torch.Tensor, boundary_frames: int) -> torch.Tensor:
    return overlap_consistency_l1(left_chunk, right_chunk, overlap_frames=boundary_frames)


def temporal_delta_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape[-1] < 3:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    pred_d1 = pred[:, :, 1:] - pred[:, :, :-1]
    target_d1 = target[:, :, 1:] - target[:, :, :-1]
    pred_d2 = pred_d1[:, :, 1:] - pred_d1[:, :, :-1]
    target_d2 = target_d1[:, :, 1:] - target_d1[:, :, :-1]
    return F.l1_loss(pred_d1, target_d1) + F.l1_loss(pred_d2, target_d2)


def sample_hop_and_overlap_tokens(
    default_hop_tokens: int,
    default_overlap_tokens: int,
    hop_candidates: list[int] | None,
    hop_probs: list[float] | None,
    overlap_jitter_tokens: int,
    random_value: float | None = None,
    jitter_value: float | None = None,
) -> tuple[int, int]:
    hop_tokens = int(default_hop_tokens)
    choices = [int(x) for x in (hop_candidates or []) if int(x) > 0]
    if choices:
        probs = [float(x) for x in (hop_probs or [])]
        if len(probs) != len(choices) or sum(probs) <= 0:
            probs = [1.0 / float(len(choices))] * len(choices)
        else:
            s = sum(probs)
            probs = [p / s for p in probs]
        rv = torch.rand(1).item() if random_value is None else float(random_value)
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if rv <= acc or idx == len(probs) - 1:
                hop_tokens = choices[idx]
                break

    overlap_tokens = int(default_overlap_tokens)
    jitter = int(overlap_jitter_tokens)
    if jitter > 0:
        jv = torch.rand(1).item() if jitter_value is None else float(jitter_value)
        shift = int(round((2.0 * jv - 1.0) * jitter))
        overlap_tokens = max(1, overlap_tokens + shift)

    return hop_tokens, overlap_tokens


def resolve_curriculum_phase(progress: float, phase_a_ratio: float, phase_b_ratio: float) -> str:
    p = max(0.0, min(1.0, float(progress)))
    a = max(0.0, float(phase_a_ratio))
    b = max(0.0, float(phase_b_ratio))
    if p < a:
        return "A"
    if p < (a + b):
        return "B"
    return "C"
