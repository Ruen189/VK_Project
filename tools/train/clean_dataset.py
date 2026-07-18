"""
Remove generated synthetic dataset (images/ + labels.json).
Does NOT delete model checkpoints or exported weights.

Usage:
  python clean_dataset.py                  # ./dataset only
  python clean_dataset.py --path ./dataset
  python clean_dataset.py --raw            # also ./raw (downloaded sources)
  python clean_dataset.py --all            # dataset + raw (checkpoints kept)
  python clean_dataset.py --yes            # no confirmation prompt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def rm_path(path: Path) -> None:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return
    if path.is_file():
        path.unlink()
        print(f"  removed file: {path}")
        return
    shutil.rmtree(path)
    print(f"  removed dir:  {path}")


def clean_dataset_dir(path: Path) -> None:
    """Remove labels.json and images/, then empty dataset folder if empty."""
    labels = path / "labels.json"
    images = path / "images"
    if labels.exists() or images.exists() or path.exists():
        rm_path(labels)
        rm_path(images)
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            rm_path(path)
        elif path.exists() and path.is_dir():
            print(f"  kept non-empty: {path}")
        return
    print(f"  skip (missing): {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean synthetic dataset / labels. Never deletes checkpoints or models/."
    )
    ap.add_argument(
        "--path",
        type=Path,
        default=Path("dataset"),
        help="Dataset folder with images/ and labels.json (default: ./dataset)",
    )
    ap.add_argument("--raw", action="store_true", help="Also delete ./raw (downloaded sources)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Delete dataset + raw. Checkpoints and model weights are never removed.",
    )
    ap.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation")
    args = ap.parse_args()

    targets: list[tuple[str, Path]] = [("dataset", args.path)]
    if args.all or args.raw:
        targets.append(("raw", Path("raw")))

    print("Will remove (models/checkpoints are kept):")
    for kind, t in targets:
        print(f"  - [{kind}] {t.resolve() if t.exists() else t}")

    if not args.yes:
        ans = input("Continue? [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return

    for kind, t in targets:
        if kind == "dataset":
            clean_dataset_dir(t)
        else:
            rm_path(t)

    print("Done. Checkpoints and exported models were not touched.")


if __name__ == "__main__":
    main()
