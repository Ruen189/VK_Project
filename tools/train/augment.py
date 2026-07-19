"""
Synthetic dataset for parameter regression — labels match browser apply().

Pipeline per sample:
  1. Sample recovery params P (what the model must predict).
  2. bad = inverse_apply(good, P)  so that apply(bad, P) ≈ good
  3. Save bad @ 224×224 and label = P

Multi-source example (17k total, memes 4× weaker):
  python clean_dataset.py -y
  python augment.py --output ./dataset --size 224 ^
    --source "./raw/div2k/DIV2K_valid_HR:5000" ^
    --source "./raw/Human Faces Dataset/RealImages:7000" ^
    --source "./raw/memes/memes:5000:0.25"

Format --source:  path:count  or  path:count:strength
  strength=1.0 full degrade; 0.25 = 4× weaker (deviations from identity ×0.25)

Single-folder (legacy):
  python augment.py --input ./raw/div2k/DIV2K_valid_HR --output ./dataset --count 10000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Install Pillow: pip install Pillow") from e

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BRIGHTNESS = (-0.3, 0.3)
CONTRAST = (0.7, 1.4)
SATURATION = (0.7, 1.5)


@dataclass
class SourceSpec:
    path: Path
    count: int
    strength: float  # 1.0 = full, 0.25 = 4× weaker
    files: list[Path]


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def apply_correction(
    rgb: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
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
    if abs(saturation) < 1e-6 or abs(contrast) < 1e-6:
        raise ValueError("contrast/saturation too close to zero")
    y = luma(rgb)[..., None]
    c = y + (rgb - y) / saturation
    c = (c - 0.5 - brightness) / contrast + 0.5
    return np.clip(c, 0.0, 1.0)


def sample_recovery_params(strength: float = 1.0) -> tuple[float, float, float]:
    """
    Sample P, then pull toward identity by (1 - strength).
    strength=1 → full range; strength=0.25 → 4× weaker degrade.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if random.random() < 0.15:
        b0 = random.uniform(-0.05, 0.05)
        c0 = random.uniform(0.95, 1.05)
        s0 = random.uniform(0.95, 1.05)
    else:
        b0 = random.uniform(BRIGHTNESS[0], BRIGHTNESS[1])
        c0 = random.uniform(CONTRAST[0], CONTRAST[1])
        s0 = random.uniform(SATURATION[0], SATURATION[1])

    # identity = (0, 1, 1)
    b = 0.0 + (b0 - 0.0) * strength
    c = 1.0 + (c0 - 1.0) * strength
    s = 1.0 + (s0 - 1.0) * strength
    return b, c, s


def pil_to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def float_to_pil(rgb: np.ndarray) -> Image.Image:
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def clip_fraction(before: np.ndarray, after: np.ndarray) -> float:
    hit = (after <= 0.0) | (after >= 1.0)
    return float(hit.mean())


def parse_source(spec: str) -> tuple[Path, int, float]:
    """
    path:count  or  path:count:strength
    Windows-safe: rsplit from the right so 'C:\\foo:5000:0.25' works.
    """
    parts = spec.rsplit(":", 2)
    if len(parts) == 3:
        path_s, count_s, strength_s = parts
        try:
            return Path(path_s), int(count_s), float(strength_s)
        except ValueError:
            pass
    parts = spec.rsplit(":", 1)
    if len(parts) == 2:
        path_s, count_s = parts
        return Path(path_s), int(count_s), 1.0
    raise SystemExit(
        f"Bad --source '{spec}'. Use path:count or path:count:strength "
        f'(example: "./raw/memes/memes:5000:0.25")'
    )


def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTS]


def make_one_sample(
    src_path: Path,
    size: int,
    strength: float,
    max_clip: float,
    source_tag: str,
) -> tuple[Image.Image, dict]:
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((size * 2, size * 2))
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    good = pil_to_float(img)

    brightness = contrast = saturation = 0.0
    bad = good
    for _ in range(12):
        brightness, contrast, saturation = sample_recovery_params(strength)
        bad = inverse_apply(good, brightness, contrast, saturation)
        if clip_fraction(good, bad) <= max_clip:
            break

    recon = apply_correction(bad, brightness, contrast, saturation)
    mse = float(np.mean((recon - good) ** 2))
    meta = {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "source": src_path.name,
        "source_dir": source_tag,
        "strength": float(strength),
        "recon_mse": mse,
    }
    return float_to_pil(bad), meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dataset with apply()-consistent labels (multi-source OK)",
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help='Repeatable. "path:count" or "path:count:strength" (strength 0.25 = 4× weaker)',
    )
    ap.add_argument("--input", type=Path, default=None, help="Legacy single folder")
    ap.add_argument("--count", type=int, default=1000, help="Legacy count for --input")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-clip", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    sources: list[SourceSpec] = []
    if args.source:
        for spec in args.source:
            path, count, strength = parse_source(spec)
            if count <= 0:
                raise SystemExit(f"count must be > 0 in {spec}")
            if not path.is_dir():
                raise SystemExit(f"Source folder not found: {path.resolve()}")
            files = list_images(path)
            if not files:
                raise SystemExit(f"No images in {path}")
            sources.append(SourceSpec(path=path, count=count, strength=strength, files=files))
    elif args.input is not None:
        files = list_images(args.input)
        if not files:
            raise SystemExit(f"No images in {args.input}")
        sources.append(
            SourceSpec(path=args.input, count=args.count, strength=1.0, files=files)
        )
    else:
        raise SystemExit("Provide --source path:count[:strength] (repeatable) or --input + --count")

    total = sum(s.count for s in sources)
    print("Plan:")
    for s in sources:
        print(
            f"  {s.path}  → {s.count} samples, strength={s.strength} "
            f"({len(s.files)} source files)"
        )
    print(f"  TOTAL → {total} files in {args.output / 'images'}")

    bad_dir = args.output / "images"
    if bad_dir.exists():
        shutil.rmtree(bad_dir)
    bad_dir.mkdir(parents=True, exist_ok=True)
    for p in (args.output / "labels.json", args.output / "meta.json"):
        p.unlink(missing_ok=True)

    labels: list[dict] = []
    idx = 0
    for spec in sources:
        tag = str(spec.path)
        print(f"\n=== {tag} ({spec.count}, strength={spec.strength}) ===")
        for local_i in range(spec.count):
            src_path = random.choice(spec.files)
            pil_bad, meta = make_one_sample(
                src_path, args.size, spec.strength, args.max_clip, tag
            )
            name = f"{idx:06d}.jpg"
            pil_bad.save(bad_dir / name, quality=92)
            labels.append({"file": name, **meta})
            idx += 1
            if (local_i + 1) % 100 == 0 or local_i + 1 == spec.count:
                print(f"  {local_i + 1}/{spec.count} (global {idx}/{total})")

    assert idx == total == len(labels)

    meta = {
        "formula": "apply-consistent-v1",
        "note": "bad = inverse_apply(good, P); label = P; multi-source with per-source strength",
        "count": total,
        "size": args.size,
        "sources": [
            {
                "path": str(s.path),
                "count": s.count,
                "strength": s.strength,
                "n_files": len(s.files),
            }
            for s in sources
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {total} samples → {args.output}")
    print("Filenames are unique 000000…; labels.json matches 1:1.")


if __name__ == "__main__":
    main()
