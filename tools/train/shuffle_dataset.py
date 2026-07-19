"""
Shuffle dataset samples while keeping images/ ↔ labels.json in sync.

What it does:
  1. Reads labels.json (list of {file, brightness, contrast, ...}).
  2. Shuffles that list (optional --seed for reproducibility).
  3. Renames image files to a temporary scheme, then to new sequential
     names 000000.jpg, 000001.jpg, … matching the shuffled order.
  4. Rewrites labels.json with updated "file" fields (1:1 with images).

Why temp names: avoid collisions like renaming A→B while B still exists.

Usage:
  cd tools/train
  python shuffle_dataset.py --data ./dataset
  python shuffle_dataset.py --data ./dataset --seed 42
  python shuffle_dataset.py --data ./dataset --seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Shuffle dataset images + labels.json together")
    ap.add_argument("--data", type=Path, default=Path("dataset"), help="Dataset root")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible shuffle)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the new order, do not rename or write files",
    )
    args = ap.parse_args()

    root = args.data
    labels_path = root / "labels.json"
    images_dir = root / "images"

    if not labels_path.is_file():
        raise SystemExit(f"Missing {labels_path}")
    if not images_dir.is_dir():
        raise SystemExit(f"Missing {images_dir}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise SystemExit("labels.json must be a non-empty JSON array")

    # Validate every label points to an existing file
    missing = []
    for row in labels:
        f = row.get("file")
        if not f or not (images_dir / f).is_file():
            missing.append(str(f))
    if missing:
        raise SystemExit(
            f"{len(missing)} label(s) reference missing files (e.g. {missing[:5]})"
        )

    if args.seed is not None:
        random.seed(args.seed)

    order = list(range(len(labels)))
    random.shuffle(order)

    print(f"Samples: {len(labels)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("First 8 after shuffle (old_index → new_file ← old_file):")
    for new_i, old_i in enumerate(order[:8]):
        old_name = labels[old_i]["file"]
        print(f"  [{old_i}] {old_name}  →  {new_i:06d}…")

    if args.dry_run:
        print("Dry-run: no files changed.")
        return

    # Phase 1: move each file to a unique temp name (no collisions)
    temp_map: list[tuple[Path, dict]] = []
    for new_i, old_i in enumerate(order):
        row = dict(labels[old_i])  # copy
        old_path = images_dir / row["file"]
        ext = old_path.suffix.lower() or ".jpg"
        tmp_path = images_dir / f"__shuffle_{new_i:06d}{ext}"
        old_path.rename(tmp_path)
        temp_map.append((tmp_path, row))

    # Phase 2: rename temps to final sequential names; update labels
    new_labels: list[dict] = []
    for new_i, (tmp_path, row) in enumerate(temp_map):
        ext = tmp_path.suffix.lower() or ".jpg"
        final_name = f"{new_i:06d}{ext}"
        final_path = images_dir / final_name
        tmp_path.rename(final_path)
        row["file"] = final_name
        new_labels.append(row)

    # Backup then write
    backup = labels_path.with_suffix(".json.bak")
    shutil.copy2(labels_path, backup)
    labels_path.write_text(json.dumps(new_labels, indent=2), encoding="utf-8")

    # Update meta if present
    meta_path = root / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["shuffled"] = True
            if args.seed is not None:
                meta["shuffle_seed"] = args.seed
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass

    print(f"Wrote {labels_path}")
    print(f"Backup: {backup}")
    print("Done. images/ and labels.json stay in sync.")


if __name__ == "__main__":
    main()
