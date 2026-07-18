"""
Download domain-specific image sets for more diverse training.

Presets:
  faces  — LFW (Labeled Faces in the Wild), real face photos / «аватарки»
  anime  — anime face images via HuggingFace datasets (streaming)

Usage:
  pip install -r requirements.txt
  python download_domain.py --preset faces --out ./raw/faces --limit 300
  python download_domain.py --preset anime --out ./raw/anime --limit 300

Then mix folders into one input for augment, e.g.:
  python augment.py --input ./raw/mixed --output ./dataset --count 20000
(copy/symlink faces+anime+div2k samples into raw/mixed)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"

# Streaming HF dataset of anime-style faces (change if upstream moves)
ANIME_HF_ID = "cchen856/anime-faces"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url}")
    print(f"→ {dest}")

    def hook(block_num: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block_num * block_size, total)
        pct = done * 100.0 / total
        sys.stdout.write(f"\r  {pct:5.1f}%  {done/1e6:.1f}/{total/1e6:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def save_limited_images(src_root: Path, out: Path, limit: int, seed: int) -> int:
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in EXTS]
    random.Random(seed).shuffle(files)
    files = files[:limit]
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files):
        dest = out / f"{i:05d}{p.suffix.lower()}"
        shutil.copy2(p, dest)
    return len(files)


def download_faces(out: Path, limit: int, seed: int, keep_archive: bool) -> None:
    archive = out.parent / "lfw.tgz"
    extract_dir = out.parent / "_lfw_extract"
    download_file(LFW_URL, archive)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting LFW …")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)

    # lfw/<name>/*.jpg
    n = save_limited_images(extract_dir, out, limit, seed)
    shutil.rmtree(extract_dir, ignore_errors=True)
    if not keep_archive:
        archive.unlink(missing_ok=True)
    print(f"Saved {n} face images → {out.resolve()}")


def download_anime(out: Path, limit: int, seed: int, hf_id: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Anime preset needs: pip install datasets huggingface_hub pillow\n"
            f"Original error: {e}"
        ) from e

    out.mkdir(parents=True, exist_ok=True)
    print(f"Streaming HuggingFace dataset: {hf_id}")
    print("(first run may download indexes; review licenses / NSFW policy)")

    ds = load_dataset(hf_id, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2_000)

    saved = 0
    for row in ds:
        img = row.get("image") or row.get("img")
        if img is None:
            for v in row.values():
                if hasattr(v, "save"):
                    img = v
                    break
        if img is None:
            continue
        dest = out / f"{saved:05d}.jpg"
        try:
            if getattr(img, "mode", None) != "RGB":
                img = img.convert("RGB")
            img.save(dest, quality=92)
        except Exception as ex:
            print(f"  skip: {ex}")
            continue
        saved += 1
        if saved % 25 == 0:
            print(f"  {saved}/{limit}")
        if saved >= limit:
            break

    if saved == 0:
        raise SystemExit(
            f"No images from {hf_id}. Try another id via --hf-id, "
            "or put anime faces manually into the out folder."
        )
    print(f"Saved {saved} anime images → {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download faces / anime image presets")
    ap.add_argument("--preset", choices=("faces", "anime"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument(
        "--hf-id",
        default=ANIME_HF_ID,
        help="HuggingFace dataset id for --preset anime",
    )
    args = ap.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    if args.preset == "faces":
        download_faces(args.out, args.limit, args.seed, args.keep_archive)
    else:
        download_anime(args.out, args.limit, args.seed, args.hf_id)

    print()
    print("Next (example mix):")
    print(f"  python augment.py --input {args.out} --output ./dataset --count 10000")


if __name__ == "__main__":
    main()
