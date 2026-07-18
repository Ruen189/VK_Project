"""
Export TinyParamNet weights to JSON for browser TinyCnnPredictor.

Usage:
  python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train import TinyParamNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyParamNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    def layer(prefix: str, out_c: int, in_c: int, k: int = 3):
        return {
            "w": sd[f"{prefix}.weight"].detach().cpu().flatten().tolist(),
            "b": sd[f"{prefix}.bias"].detach().cpu().flatten().tolist(),
            "outC": out_c,
            "inC": in_c,
            "k": k,
        }

    def fc(prefix: str, out_n: int, in_n: int):
        return {
            "w": sd[f"{prefix}.weight"].detach().cpu().flatten().tolist(),
            "b": sd[f"{prefix}.bias"].detach().cpu().flatten().tolist(),
            "out": out_n,
            "in": in_n,
        }

    payload = {
        "arch": "TinyParamNet",
        "conv1": layer("features.0", 16, 3),
        "conv2": layer("features.2", 32, 16),
        "conv3": layer("features.4", 64, 32),
        "fc1": fc("head.1", 32, 64),
        "fc2": fc("head.3", 3, 32),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB)")
    print("Copy to demo/models/ or open demo after npm run build (copies models/).")


if __name__ == "__main__":
    main()
