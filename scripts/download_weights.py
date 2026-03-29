#!/usr/bin/env python3
"""Download binary weights into ./weights/ from a JSON manifest.

URL schemes:
  https://...           — direct HTTP(S)
  hf:repo_id:filename   — Hugging Face Hub (needs huggingface_hub, optional HF_TOKEN)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

_ROOT = Path(__file__).resolve().parents[1]


def download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("hf:"):
        rest = url[3:]
        if rest.count(":") < 1:
            raise ValueError(f"Bad hf: URL (use hf:org/repo:file.pt): {url}")
        repo_id, fname = rest.split(":", 1)
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(repo_id=repo_id, filename=fname))
        shutil.copy2(p, dest)
        print(f"OK hf {repo_id} {fname} -> {dest}")
        return
    if url.startswith("file://"):
        src = Path(url[7:])
        shutil.copy2(src, dest)
        print(f"OK copy {src} -> {dest}")
        return
    print(f"GET {url} -> {dest}")
    urlretrieve(url, dest)
    print(f"OK {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=_ROOT / "weights_manifest.json")
    ap.add_argument("--output-dir", type=Path, default=_ROOT / "weights")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(
            f"Missing {args.manifest}. Copy weights_manifest.example.json to weights_manifest.json "
            "and fill URLs (or use Hugging Face repo:hf:... entries).",
            file=sys.stderr,
        )
        sys.exit(2)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = data.get("files", [])
    if not files:
        print("Manifest has no files[]", file=sys.stderr)
        sys.exit(2)

    cfg_src = _ROOT / "configs" / "cosyvoice.yaml"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_src, args.output_dir / "cosyvoice.yaml")
    print(f"Copied cosyvoice.yaml -> {args.output_dir / 'cosyvoice.yaml'}")

    for item in files:
        name = item.get("path") or item.get("name")
        url = item.get("url", "").strip()
        if not name:
            raise ValueError(f"bad manifest entry: {item}")
        if not url:
            print(f"SKIP (empty url): {name}")
            continue
        dest = args.output_dir / name
        download_one(url, dest)

    print("Done. Next: python infer.py --tokenizer_pt /path/to/s3_tokenizer.pt --prompt_wav ...")


if __name__ == "__main__":
    main()
