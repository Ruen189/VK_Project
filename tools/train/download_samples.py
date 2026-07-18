"""
Download a small free sample set (picsum.photos) for a quick pipeline smoke-test.

~50–200 photos, a few dozen MB. Not a production dataset — use DIV2K after.

Usage:
  python download_samples.py --out ./raw/samples --count 100
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("raw/samples"))
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--size", type=int, default=512, help="Square edge in pixels")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    size = args.size

    for i in range(args.count):
        # Deterministic seed → reproducible set
        url = f"https://picsum.photos/seed/vk{i}/{size}/{size}.jpg"
        dest = args.out / f"{i:04d}.jpg"
        if dest.exists():
            print(f"skip {dest.name}")
            continue
        print(f"[{i + 1}/{args.count}] {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  failed: {e}")

    n = len(list(args.out.glob("*.jpg")))
    print(f"Done: {n} images in {args.out.resolve()}")
    print(f"Next: python augment.py --input {args.out} --output ./dataset --count 5000")


if __name__ == "__main__":
    main()
