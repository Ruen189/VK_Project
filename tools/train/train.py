"""
Minimal PyTorch training loop for 3-parameter regression.

Install:
  pip install torch torchvision pillow

Usage:
  python train.py --data ./dataset --epochs 20 --out ./checkpoints/model.pt

Then export ONNX (see export_onnx.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


class ParamDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),  # 0..1 CHW
            ]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        row = self.labels[idx]
        img = Image.open(self.root / "images" / row["file"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(
            [row["brightness"], row["contrast"], row["saturation"]],
            dtype=torch.float32,
        )
        return x, y


class TinyParamNet(nn.Module):
    """CNN for 3-param regression. Channels: 32→64→128, head 128→64→3."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def unique_out_path(path: Path) -> Path:
    """If path exists, use model_2.pt, model_3.pt, … (keeps previous checkpoints)."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --out if it exists (default: save as model_2.pt, model_3.pt, …)",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue training from an existing .pt checkpoint (fine-tune)",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate weights every N epochs (0 = only final). "
        "Files: <stem>_ep005.pt, <stem>_ep010.pt, … next to --out",
    )
    args = ap.parse_args()

    if args.save_every < 0:
        raise SystemExit("--save-every must be >= 0")

    ds = ParamDataset(args.data)
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = TinyParamNet().to(device)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"resumed from {args.resume}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_path = args.out if args.overwrite else unique_out_path(args.out)
    if out_path != args.out:
        print(f"{args.out} exists → saving as {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_every:
        print(f"checkpoints every {args.save_every} epoch(s) → {out_path.stem}_epXXX{out_path.suffix}")

    def save_ckpt(path: Path, epoch: int, train_loss: float, val_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "arch": "TinyParamNet",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        train_loss = total / max(1, n_train)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vtotal += loss_fn(model(x), y).item() * x.size(0)
        val_loss = vtotal / max(1, n_val)
        print(f"epoch {epoch:03d}  train={train_loss:.5f}  val={val_loss:.5f}")

        if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
            mid = out_path.parent / f"{out_path.stem}_ep{epoch:03d}{out_path.suffix}"
            save_ckpt(mid, epoch, train_loss, val_loss)
            print(f"  checkpoint → {mid}")

    save_ckpt(out_path, args.epochs, train_loss, val_loss)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
