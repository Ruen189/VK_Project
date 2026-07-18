"""
Export TinyParamNet checkpoint to ONNX for ONNX Runtime Web.

Usage:
  python export_onnx.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import TinyParamNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TinyParamNet()
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["params"],
        dynamic_axes={"input": {0: "batch"}, "params": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {args.out}")
    print("Next: optional INT8 quantize, then wire OnnxParamPredictor in src/ml/predictor.ts")


if __name__ == "__main__":
    main()
