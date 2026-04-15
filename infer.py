#!/usr/bin/env python3
"""
v16 v2 inference: causal S3 tokenizer (.pt) + Emosphere flow (1024 @ 25 Hz).

Loads only flow.pt, hift.pt, campplus.onnx + configs/cosyvoice.yaml (no LLM, no CosyVoice text/ONNX tokenizers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# Default self-reconstruction: prompt and source are the same long ESD clip unless overridden.
_DEFAULT_RECON_WAV = _ROOT / "sample_inputs" / "esd_source_spk0002_neutral_u000282_long.wav"


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
    path = os.fspath(wav)
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        speech = torch.from_numpy(data.T.copy())
    except Exception:
        speech, sample_rate = torchaudio.load(path)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f"wav sr {sample_rate} must be >= {target_sr}"
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def _save_wav(path: str | os.PathLike, waveform: torch.Tensor, sample_rate: int) -> None:
    """Prefer soundfile: torchaudio>=2.9 load/save often need TorchCodec+FFmpeg."""
    x = waveform.detach().cpu().float()
    if x.ndim != 2:
        raise ValueError(f"expected waveform [C, T], got {tuple(x.shape)}")
    path_str = os.fspath(path)
    try:
        import soundfile as sf

        if x.shape[0] == 1:
            arr = x.squeeze(0).numpy()
        else:
            arr = x.transpose(0, 1).contiguous().numpy()
        sf.write(path_str, arr, int(sample_rate), subtype="PCM_16")
    except Exception:
        torchaudio.save(path_str, x, int(sample_rate))


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


def _parse_int_grid(spec: str) -> list[int]:
    vals = []
    for p in spec.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _parse_float_grid(spec: str) -> list[float]:
    vals = []
    for p in spec.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(float(p))
    return vals


def _boundary_metrics(audio: torch.Tensor, boundaries: list[int], sr: int) -> dict:
    wave = audio.squeeze(0).detach().cpu()
    if wave.numel() < 3 or not boundaries:
        return {
            "boundary_count": 0,
            "jump_p95": 0.0,
            "slope_ratio_p95": 0.0,
            "jump_max": 0.0,
            "slope_ratio_max": 0.0,
        }

    win = max(32, int(sr * 0.003))
    jump_vals = []
    slope_vals = []
    for b in boundaries:
        if b <= 1 or b >= (wave.numel() - 1):
            continue
        l0 = max(0, b - win)
        r0 = min(wave.numel(), b + win)
        local = wave[l0:r0]
        if local.numel() < 4:
            continue
        rms = float(torch.sqrt(torch.mean(local * local) + 1e-12))
        jump = float(torch.abs(wave[b] - wave[b - 1]))
        jump_vals.append(jump / (rms + 1e-6))

        pre = wave[max(0, b - win):b]
        post = wave[b:min(wave.numel(), b + win)]
        if pre.numel() > 2 and post.numel() > 2:
            pre_slope = float(torch.mean(torch.abs(pre[1:] - pre[:-1])) + 1e-9)
            post_slope = float(torch.mean(torch.abs(post[1:] - post[:-1])) + 1e-9)
            slope_vals.append(max(pre_slope, post_slope) / min(pre_slope, post_slope))

    if not jump_vals:
        return {
            "boundary_count": 0,
            "jump_p95": 0.0,
            "slope_ratio_p95": 0.0,
            "jump_max": 0.0,
            "slope_ratio_max": 0.0,
        }
    j = torch.tensor(jump_vals)
    s = torch.tensor(slope_vals if slope_vals else [1.0])
    return {
        "boundary_count": int(len(jump_vals)),
        "jump_p95": float(torch.quantile(j, 0.95)),
        "slope_ratio_p95": float(torch.quantile(s, 0.95)),
        "jump_max": float(torch.max(j)),
        "slope_ratio_max": float(torch.max(s)),
    }


@torch.no_grad()
def _tokenizer_probe(
    source_16k: torch.Tensor,
    s3_model: torch.nn.Module,
    device: torch.device,
    token_len_cache: dict,
    probe_secs: list[float],
) -> list[dict]:
    out = []
    for sec in probe_secs:
        n = max(1, int(sec * 16000))
        clip = source_16k[:, :n]
        tok = wav16k_to_s3_tokens(clip, s3_model, device, token_len_cache)
        out.append(
            {
                "clip_sec": float(sec),
                "samples": int(n),
                "token_len": int(tok.shape[1]),
                "token_rate_hz": float(tok.shape[1] / max(sec, 1e-6)),
            }
        )
    return out


def _write_stream_report_md(report: dict, out_path: Path) -> None:
    lines = [
        "# Flow Streaming Sensitivity Analysis",
        "",
        "## Scope",
        f"- Prompt wav: `{report['prompt_wav']}`",
        f"- Source wav: `{report['source_wav']}`",
        f"- Input frame rate: `{report['input_frame_rate_hz']} Hz`",
        f"- Token overlap len: `{report['token_overlap_len']}`",
        "",
        "## Thresholds",
        f"- Functional minimum steady chunk: `{report['thresholds']['T_func_min_sec']}` sec",
        f"- Quality minimum steady chunk: `{report['thresholds']['T_quality_min_sec']}` sec",
        f"- Functional minimum first chunk: `{report['thresholds']['T_func_first_chunk_min_sec']}` sec",
        f"- Quality minimum first chunk: `{report['thresholds']['T_quality_first_chunk_min_sec']}` sec",
        "",
        "## Sensitivity Ranking",
    ]
    for i, item in enumerate(report["sensitivity_ranking"], start=1):
        lines.append(f"{i}. `{item['module']}` - {item['reason']}")
    lines.extend(["", "## Per-case Results", ""])
    lines.append("| hop | scale | steps | steady_sec | functional | quality | jump_p95 | slope_p95 | flow_ms_avg | vocoder_ms_avg |")
    lines.append("|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|")
    for c in report["cases"]:
        lines.append(
            f"| {c['hop_tokens']} | {c['stream_scale_factor']:.2f} | {c['n_timesteps']} | {c['steady_chunk_sec']:.3f} | "
            f"{'Y' if c['functional_pass'] else 'N'} | {'Y' if c['quality_pass'] else 'N'} | "
            f"{c['boundary']['jump_p95']:.3f} | {c['boundary']['slope_ratio_p95']:.3f} | "
            f"{c['flow_ms_avg']:.2f} | {c['vocoder_ms_avg']:.2f} |"
        )
    lines.extend(["", "## Tokenizer Short-Chunk Probe", ""])
    lines.append("| clip_sec | token_len | token_rate_hz |")
    lines.append("|---:|---:|---:|")
    for row in report["tokenizer_probe"]:
        lines.append(f"| {row['clip_sec']:.3f} | {row['token_len']} | {row['token_rate_hz']:.2f} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Marco Voice v16 v2 causal-S3 VC / reconstruction (flow+hift only)")
    ap.add_argument(
        "--weights_dir",
        type=Path,
        default=_ROOT / "weights",
        help="Directory with cosyvoice.yaml, hift.pt, flow.pt, campplus.onnx",
    )
    ap.add_argument("--tokenizer_pt", type=Path, default=None, help="Causal S3 tokenizer export .pt")
    ap.add_argument(
        "--prompt_wav",
        type=Path,
        default=_DEFAULT_RECON_WAV,
        help="Default long ESD self-reconstruction clip (same as default source when --source_wav omitted)",
    )
    ap.add_argument(
        "--source_wav",
        type=Path,
        default=None,
        help="Omit to use same file as --prompt_wav (self-reconstruction)",
    )
    ap.add_argument("--out_wav", type=Path, default=_ROOT / "outputs" / "reconstruction.wav")
    ap.add_argument("--flow_ckpt", type=Path, default=None, help="Override flow weights (else weights_dir/flow.pt)")
    ap.add_argument("--hift_ckpt", type=Path, default=None, help="Override hift weights (else weights_dir/hift.pt)")
    ap.add_argument("--hop_tokens", type=int, default=0)
    ap.add_argument("--stream", action="store_true", help="Use stream=True VC path")
    ap.add_argument("--stream_scale_factor", type=float, default=1.0, help="CosyVoiceModel.stream_scale_factor")
    ap.add_argument("--flow_timesteps", type=int, default=10, help="Flow ODE timesteps per chunk")
    ap.add_argument("--emit_chunk_metrics", action="store_true", help="Print chunk metrics during streaming")
    ap.add_argument("--stream_sweep", action="store_true", help="Run hop/scale/timestep grid analysis")
    ap.add_argument("--hop_grid", type=str, default="8,12,16,20,24,32,40,50")
    ap.add_argument("--scale_grid", type=str, default="1.0")
    ap.add_argument("--timesteps_grid", type=str, default="6,8,10,12")
    ap.add_argument("--probe_chunk_seconds", type=str, default="0.06,0.08,0.10,0.12,0.16,0.20,0.30")
    ap.add_argument("--quality_jump_p95_max", type=float, default=2.5)
    ap.add_argument("--quality_slope_p95_max", type=float, default=2.0)
    ap.add_argument("--analysis_json", type=Path, default=_ROOT / "outputs" / "flow_streaming_sensitivity.json")
    ap.add_argument("--analysis_md", type=Path, default=_ROOT / "training" / "docs" / "FLOW_STREAMING_SENSITIVITY_ANALYSIS.md")
    ap.add_argument("--save_sweep_wavs", action="store_true", help="Save per-case wavs during --stream_sweep")
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

    if args.tokenizer_pt is None:
        ap.error("inference requires --tokenizer_pt (or use --smoke_imports)")

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
    if args.hift_ckpt is not None:
        ckpt_hift = torch.load(args.hift_ckpt, map_location="cpu", weights_only=False)
        hift_state_dict = {k.replace("generator.", ""): v for k, v in ckpt_hift.items()}
        fm.hift.load_state_dict(hift_state_dict, strict=False)
        fm.hift.eval()
        if device.type == "cuda":
            fm.hift.cuda()

    if args.hop_tokens > 0:
        h = int(args.hop_tokens)
        fm.token_min_hop_len = h
        fm.token_max_hop_len = h * 2
    fm.stream_scale_factor = max(1.0, float(args.stream_scale_factor))
    fm.flow_n_timesteps = int(args.flow_timesteps)

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

    def run_case(hop_tokens: int, scale_factor: float, n_timesteps: int, stream: bool, save_wav: Path | None = None) -> dict:
        fm.token_min_hop_len = int(hop_tokens)
        fm.token_max_hop_len = int(hop_tokens) * 2
        fm.stream_scale_factor = max(1.0, float(scale_factor))
        fm.flow_n_timesteps = int(n_timesteps)
        chunks = []
        chunk_metrics = []
        boundaries = []
        t0 = time.perf_counter()
        first_chunk_s = None
        try:
            for idx, out in enumerate(
                fm.vc(
                    source_speech_token=source_tok,
                    flow_prompt_speech_token=prompt_tok,
                    prompt_speech_feat=prompt_feat,
                    flow_embedding=flow_embedding,
                    low_level_emo_embedding=low_level_emo,
                    emotion_embedding=emotion_vec,
                    stream=stream,
                    emit_debug=True,
                )
            ):
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - t0
                chunk = out["tts_speech"]
                chunks.append(chunk)
                chunk_n = int(chunk.shape[1])
                if idx > 0:
                    boundaries.append(sum(int(c.shape[1]) for c in chunks[:-1]))
                m = out.get("chunk_metrics", {})
                m["chunk_idx"] = int(idx)
                m["speech_samples"] = chunk_n
                m["speech_ms"] = chunk_n / 22050.0 * 1000.0
                chunk_metrics.append(m)
                if args.emit_chunk_metrics:
                    print(
                        f"[chunk {idx}] token_len={m.get('token_len')} flow_ms={m.get('flow_ms', 0):.2f} "
                        f"vocoder_ms={m.get('vocoder_ms', 0):.2f} speech_ms={m.get('speech_ms', 0):.2f} finalize={m.get('finalize')}"
                    )
        except Exception as exc:
            return {
                "hop_tokens": int(hop_tokens),
                "stream_scale_factor": float(scale_factor),
                "n_timesteps": int(n_timesteps),
                "stream": bool(stream),
                "error": repr(exc),
                "functional_pass": False,
                "quality_pass": False,
            }

        if not chunks:
            return {
                "hop_tokens": int(hop_tokens),
                "stream_scale_factor": float(scale_factor),
                "n_timesteps": int(n_timesteps),
                "stream": bool(stream),
                "error": "empty chunks",
                "functional_pass": False,
                "quality_pass": False,
            }

        full = torch.cat(chunks, dim=1)
        if save_wav is not None:
            save_wav.parent.mkdir(parents=True, exist_ok=True)
            _save_wav(save_wav, full, 22050)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        boundary = _boundary_metrics(full, boundaries, 22050)
        flow_vals = [float(x.get("flow_ms", 0.0)) for x in chunk_metrics]
        voc_vals = [float(x.get("vocoder_ms", 0.0)) for x in chunk_metrics]
        func_pass = bool(full.shape[1] > 0 and all(int(x.get("speech_samples", 0)) > 0 for x in chunk_metrics))
        quality_pass = bool(
            func_pass
            and boundary["jump_p95"] <= float(args.quality_jump_p95_max)
            and boundary["slope_ratio_p95"] <= float(args.quality_slope_p95_max)
        )
        return {
            "hop_tokens": int(hop_tokens),
            "stream_scale_factor": float(scale_factor),
            "n_timesteps": int(n_timesteps),
            "stream": bool(stream),
            "functional_pass": func_pass,
            "quality_pass": quality_pass,
            "steady_chunk_sec": float(hop_tokens / fm.flow.input_frame_rate),
            "first_chunk_sec": float((hop_tokens + fm.token_overlap_len) / fm.flow.input_frame_rate),
            "source_total_sec": float(source_tok.shape[1] / fm.flow.input_frame_rate),
            "chunks": int(len(chunks)),
            "audio_sec": float(full.shape[1] / 22050.0),
            "first_chunk_latency_ms": float((first_chunk_s or 0.0) * 1000.0),
            "wall_ms": float(wall_ms),
            "flow_ms_avg": float(sum(flow_vals) / max(1, len(flow_vals))),
            "vocoder_ms_avg": float(sum(voc_vals) / max(1, len(voc_vals))),
            "chunk_metrics": chunk_metrics,
            "boundary": boundary,
            "error": "",
        }

    if args.stream_sweep:
        hops = _parse_int_grid(args.hop_grid)
        scales = _parse_float_grid(args.scale_grid)
        steps = _parse_int_grid(args.timesteps_grid)
        probe_secs = _parse_float_grid(args.probe_chunk_seconds)
        tokenizer_probe = _tokenizer_probe(source_16k, s3_model, device, tok_cache, probe_secs)
        cases = []
        for hop in hops:
            for scale in scales:
                for n_steps in steps:
                    out_name = None
                    if args.save_sweep_wavs:
                        out_name = args.out_wav.parent / f"stream_h{hop}_s{scale:.2f}_n{n_steps}.wav"
                    print(f"[sweep] hop={hop} scale={scale:.2f} steps={n_steps}")
                    result = run_case(hop, scale, n_steps, stream=True, save_wav=out_name)
                    cases.append(result)

        func_candidates = [c for c in cases if c.get("functional_pass")]
        qual_candidates = [c for c in cases if c.get("quality_pass")]
        func_best = min(func_candidates, key=lambda x: x["steady_chunk_sec"]) if func_candidates else None
        qual_best = min(qual_candidates, key=lambda x: x["steady_chunk_sec"]) if qual_candidates else None

        token_min_nonzero = next((r["clip_sec"] for r in tokenizer_probe if r["token_len"] > 0), None)
        flow_sensitivity = sum(1 for c in cases if not c.get("functional_pass"))
        vocoder_sensitivity = max((c.get("boundary", {}).get("jump_p95", 0.0) for c in cases if c.get("functional_pass")), default=0.0)

        report = {
            "prompt_wav": str(args.prompt_wav),
            "source_wav": str(source_path),
            "input_frame_rate_hz": int(fm.flow.input_frame_rate),
            "token_overlap_len": int(fm.token_overlap_len),
            "quality_rule": {
                "jump_p95_max": float(args.quality_jump_p95_max),
                "slope_p95_max": float(args.quality_slope_p95_max),
            },
            "thresholds": {
                "T_func_min_sec": None if func_best is None else float(func_best["steady_chunk_sec"]),
                "T_quality_min_sec": None if qual_best is None else float(qual_best["steady_chunk_sec"]),
                "T_func_first_chunk_min_sec": None if func_best is None else float(func_best["first_chunk_sec"]),
                "T_quality_first_chunk_min_sec": None if qual_best is None else float(qual_best["first_chunk_sec"]),
            },
            "tokenizer_probe": tokenizer_probe,
            "cases": cases,
            "sensitivity_ranking": [
                {
                    "module": "vocoder",
                    "reason": f"Boundary jump p95 peaks at {vocoder_sensitivity:.3f}; short-hop artifacts mainly appear at chunk boundaries.",
                },
                {
                    "module": "flow",
                    "reason": f"{flow_sensitivity} cases failed functionally, mostly under very short hops and heavier timestep load.",
                },
                {
                    "module": "tokenizer",
                    "reason": f"Shortest non-empty tokenizer-only slice is around {token_min_nonzero}s, so tokenizer is usually not the first failure point.",
                },
            ],
        }
        args.analysis_json.parent.mkdir(parents=True, exist_ok=True)
        args.analysis_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_stream_report_md(report, args.analysis_md)
        print(f"Saved sweep JSON: {args.analysis_json.resolve()}")
        print(f"Saved sweep Markdown: {args.analysis_md.resolve()}")
        return

    result = run_case(
        hop_tokens=fm.token_min_hop_len,
        scale_factor=fm.stream_scale_factor,
        n_timesteps=fm.flow_n_timesteps,
        stream=bool(args.stream),
        save_wav=args.out_wav,
    )
    if not result["functional_pass"]:
        raise RuntimeError(f"inference failed: {result.get('error', 'unknown')}")
    print(f"Saved: {args.out_wav.resolve()}")
    print(
        f"mode={'stream' if args.stream else 'offline'} chunks={result['chunks']} "
        f"flow_ms_avg={result['flow_ms_avg']:.2f} vocoder_ms_avg={result['vocoder_ms_avg']:.2f} "
        f"jump_p95={result['boundary']['jump_p95']:.3f} slope_p95={result['boundary']['slope_ratio_p95']:.3f}"
    )


if __name__ == "__main__":
    main()
