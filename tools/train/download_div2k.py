"""
Download DIV2K high-resolution images for synthetic training.

Default: validation split (~100 HR images in zip ~450 MB).
Full train zip ~3.7 GB via --split train.

Note: DIV2K is one zip per split — the archive still downloads fully.
Use --limit N to extract only the first N PNGs (saves disk after download).

Usage:
  python download_div2k.py --out ./raw/div2k
  python download_div2k.py --out ./raw/div2k --limit 100
  python download_div2k.py --out ./raw/div2k --split train --limit 200
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return

    print(f"Downloading {url}")
    print(f"→ {dest}")

    def hook(block_num: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, done * 100.0 / total)
        mb = done / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def unzip_limited(zip_path: Path, out_dir: Path, limit: int | None) -> Path:
    """Extract PNG members; if limit is set, only the first `limit` image files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}" + (f" (limit={limit})" if limit else "") + " …")

    extracted = 0
    top_dirs: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m
            for m in zf.namelist()
            if m.lower().endswith(".png") and not m.endswith("/")
        ]
        members.sort()
        if limit is not None:
            members = members[: max(0, limit)]

        for name in members:
            zf.extract(name, out_dir)
            extracted += 1
            parts = Path(name).parts
            if parts:
                top_dirs.add(parts[0])
            if extracted % 20 == 0 or extracted == len(members):
                sys.stdout.write(f"\r  extracted {extracted}/{len(members)}")
                sys.stdout.flush()
        print()

    # Prefer DIV2K_* folder if present
    for name in sorted(top_dirs):
        candidate = out_dir / name
        if candidate.is_dir() and "DIV2K" in name:
            return candidate

    candidates = [p for p in out_dir.iterdir() if p.is_dir() and "DIV2K" in p.name]
    if candidates:
        return candidates[0]
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Download DIV2K HR images")
    ap.add_argument("--out", type=Path, default=Path("raw/div2k"))
    ap.add_argument("--split", choices=("valid", "train"), default="valid")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract at most N PNG images (zip still downloads fully)",
    )
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    url = URLS[args.split]
    zip_path = args.out / f"DIV2K_{args.split}_HR.zip"
    download(url, zip_path)
    images_dir = unzip_limited(zip_path, args.out, args.limit)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("Removed zip to save disk space.")

    n = len(list(images_dir.rglob("*.png")))
    print(f"Done. PNG images: {n}")
    print(f"Images folder: {images_dir.resolve()}")
    print()
    print("Next:")
    print(
        f"  python augment.py --input {images_dir} --output ./dataset --count 10000"
    )


if __name__ == "__main__":
    main()
