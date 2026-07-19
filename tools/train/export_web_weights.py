"""
Export TinyParamNet weights for browser TinyCnnPredictor.

Default: binary .bin (float16 or float32). Optional legacy JSON.

Format TPCB v1 (little-endian):
  magic[4] = b'TPCB'
  version  u16 = 1
  dtype    u8  = 0 float32 | 1 float16
  pad      u8  = 0
  c1,c2,c3,fc_hidden  u16 each  (default 32,64,128,64)
  then tensors in order:
    conv1.w, conv1.b, conv2.w, conv2.b, conv3.w, conv3.b, fc1.w, fc1.b, fc2.w, fc2.b

Usage:
  python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.bin
  python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.bin --dtype float16
  python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.json --format json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

from train import TinyParamNet

# Must match train.TinyParamNet (32→64→128, head 128→64→3)
C1, C2, C3, FC_H = 32, 64, 128, 64
MAGIC = b"TPCB"
VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    ap.add_argument("--format", choices=("bin", "json"), default="bin")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinyParamNet()
    model.load_state_dict(ckpt["model"])
    model.eval()
    sd = model.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        export_json(sd, args.out)
    else:
        export_bin(sd, args.out, args.dtype)

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KB, format={args.format}, dtype={args.dtype})")
    print("Then: npm run build  (copies models/ → demo/models/)")


def tensor_list(sd: dict, key: str) -> list[float]:
    return sd[key].detach().cpu().float().flatten().tolist()


def export_json(sd: dict, out: Path) -> None:
    def layer(prefix: str, out_c: int, in_c: int, k: int = 3):
        return {
            "w": tensor_list(sd, f"{prefix}.weight"),
            "b": tensor_list(sd, f"{prefix}.bias"),
            "outC": out_c,
            "inC": in_c,
            "k": k,
        }

    def fc(prefix: str, out_n: int, in_n: int):
        return {
            "w": tensor_list(sd, f"{prefix}.weight"),
            "b": tensor_list(sd, f"{prefix}.bias"),
            "out": out_n,
            "in": in_n,
        }

    payload = {
        "arch": "TinyParamNet",
        "conv1": layer("features.0", C1, 3),
        "conv2": layer("features.2", C2, C1),
        "conv3": layer("features.4", C3, C2),
        "fc1": fc("head.1", FC_H, C3),
        "fc2": fc("head.3", 3, FC_H),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")


def export_bin(sd: dict, out: Path, dtype: str) -> None:
    keys = [
        "features.0.weight",
        "features.0.bias",
        "features.2.weight",
        "features.2.bias",
        "features.4.weight",
        "features.4.bias",
        "head.1.weight",
        "head.1.bias",
        "head.3.weight",
        "head.3.bias",
    ]
    dtype_code = 1 if dtype == "float16" else 0
    header = struct.pack(
        "<4sHBBHHHH",
        MAGIC,
        VERSION,
        dtype_code,
        0,
        C1,
        C2,
        C3,
        FC_H,
    )
    chunks = [header]
    for key in keys:
        t = sd[key].detach().cpu().contiguous().flatten()
        if dtype == "float16":
            t = t.half()
            chunks.append(t.numpy().tobytes())
        else:
            t = t.float()
            chunks.append(t.numpy().tobytes())
    out.write_bytes(b"".join(chunks))


if __name__ == "__main__":
    main()
