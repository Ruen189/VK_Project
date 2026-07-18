"""
Synthetic dataset for parameter regression — labels match browser apply().

Pipeline per sample:
  1. Sample recovery params P (what the model must predict).
  2. bad = inverse_apply(good, P)  so that apply(bad, P) ≈ good
  3. Save bad @ 224×224 and label = P

Formulas mirror src/apply/correct.ts (sRGB [0,1], Rec.709 luma).

Usage:
  python augment.py --input ./raw/div2k/DIV2K_valid_HR --output ./dataset --count 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Install Pillow: pip install Pillow") from e

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Same clips as src/types.ts PARAM_CLIP
BRIGHTNESS = (-0.3, 0.3)
CONTRAST = (0.7, 1.4)
SATURATION = (0.7, 1.5)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma; rgb[..., 3]."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def apply_correction(
    rgb: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """Forward apply — identical math to src/apply/correct.ts."""
    c = (rgb - 0.5) * contrast + 0.5 + brightness
    y = luma(c)[..., None]
    c = y + (c - y) * saturation
    return np.clip(c, 0.0, 1.0)


def inverse_apply(
    rgb: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """
    Inverse of apply_correction (ignoring clamp).
    By construction: apply(inverse_apply(good, P), P) ≈ good (except clipped pixels).
    """
    if abs(saturation) < 1e-6 or abs(contrast) < 1e-6:
        raise ValueError("contrast/saturation too close to zero")
    y = luma(rgb)[..., None]
    c = y + (rgb - y) / saturation
    c = (c - 0.5 - brightness) / contrast + 0.5
    return np.clip(c, 0.0, 1.0)


def sample_recovery_params() -> tuple[float, float, float]:
    """
    Sample P away from identity often enough to create visible degradation.
    Mix: 85% meaningful correction, 15% near-identity (stability).
    """
    if random.random() < 0.15:
        return (
            random.uniform(-0.05, 0.05),
            random.uniform(0.95, 1.05),
            random.uniform(0.95, 1.05),
        )
    return (
        random.uniform(BRIGHTNESS[0], BRIGHTNESS[1]),
        random.uniform(CONTRAST[0], CONTRAST[1]),
        random.uniform(SATURATION[0], SATURATION[1]),
    )


def pil_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def float_to_pil(rgb: np.ndarray) -> Image.Image:
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def clip_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Share of pixels that hit 0/1 after inverse (label may be slightly imperfect)."""
    hit = (after <= 0.0) | (after >= 1.0)
    # only count channels that weren't already clipped in before
    return float(hit.mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dataset with apply()-consistent labels",
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument(
        "--max-clip",
        type=float,
        default=0.35,
        help="Resample params if inverse clips more than this fraction of pixels",
    )
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    files = [p for p in args.input.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"No images in {args.input}")

    bad_dir = args.output / "images"
    bad_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []

    for i in range(args.count):
        src_path = random.choice(files)
        img = Image.open(src_path).convert("RGB")
        img.thumbnail((args.size * 2, args.size * 2))
        img = img.resize((args.size, args.size), Image.Resampling.BILINEAR)
        good = pil_to_float(img)

        brightness = contrast = saturation = 0.0
        bad = good
        for _attempt in range(12):
            brightness, contrast, saturation = sample_recovery_params()
            bad = inverse_apply(good, brightness, contrast, saturation)
            if clip_fraction(good, bad) <= args.max_clip:
                break

        # Optional sanity: reconstruction error after apply (should be tiny without heavy clip)
        recon = apply_correction(bad, brightness, contrast, saturation)
        mse = float(np.mean((recon - good) ** 2))

        name = f"{i:06d}.jpg"
        float_to_pil(bad).save(bad_dir / name, quality=92)
        labels.append(
            {
                "file": name,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "saturation": float(saturation),
                "source": src_path.name,
                "recon_mse": mse,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.count}")

    meta = {
        "formula": "apply-consistent-v1",
        "note": "bad = inverse_apply(good, P); label = P; matches src/apply/correct.ts",
        "count": args.count,
        "size": args.size,
    }
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {args.count} samples → {args.output}")
    print("Labels are recovery params P for browser apply(); rebuild dataset before re-training.")


if __name__ == "__main__":
    main()
