#!/usr/bin/env python3
"""Create a minimal Emosphere-style processed train dir (one utterance) for smoke tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True, help="output directory (will be created)")
    ap.add_argument("--wav", type=Path, required=True, help="16 kHz+ WAV path")
    ap.add_argument("--utt", type=str, default="smoke01", help="utterance id")
    ap.add_argument("--spk", type=str, default="spk1", help="speaker id")
    args = ap.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wav_abs = args.wav.resolve()

    (out / "wav.scp").write_text(f"{args.utt}\t{wav_abs}\n", encoding="utf-8")
    (out / "text").write_text(f"{args.utt}\thello world\n", encoding="utf-8")
    (out / "utt2spk").write_text(f"{args.utt}\t{args.spk}\n", encoding="utf-8")

    # Dimensions match Emosphere flow training (campplus / emotion2vec style).
    utt_e = torch.randn(192)
    spk_e = torch.randn(192)
    torch.save({args.utt: utt_e}, out / "utt2embedding.pt")
    torch.save({args.spk: spk_e}, out / "spk2embedding.pt")
    torch.save({args.utt: "Neutral"}, out / "utt_emo.pt")
    torch.save({args.utt: torch.randn(768)}, out / "emotion_embedding.pt")
    torch.save({args.utt: torch.tensor([0.41, 0.52, 0.36], dtype=torch.float32)}, out / "low_level_embedding.pt")

    print(f"OK: wrote Emosphere smoke dir → {out}")


if __name__ == "__main__":
    main()
