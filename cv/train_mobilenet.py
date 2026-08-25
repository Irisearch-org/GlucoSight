#!/usr/bin/env python3
"""Train MobileNetV2 on the Indonesian Food Image dataset (CPU-friendly).

Saves checkpoint to `cv/models/mobilenetv2_food10.pt` compatible with the
inference wrapper in `cv/cv_baseline/classifier.py`.

Usage:
    python cv/train_mobilenet.py
    python cv/train_mobilenet.py --epochs 10 --batch-size 32
"""

import argparse
import json
import random
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "cv" / "indonesian_food_image"
DEFAULT_OUTPUT = PROJECT_ROOT / "models"


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def build_dataloaders(root: Path, image_size=224, batch_size=32, num_workers=2):
    # root should contain train/ and test/ (train will be split into train/val)
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")

    # Transforms approximating the notebook training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    full = datasets.ImageFolder(train_dir, transform=train_tf)
    class_to_idx = full.class_to_idx

    # split
    n = len(full)
    val_size = int(0.2 * n)
    train_size = n - val_size
    train_set, val_set = torch.utils.data.random_split(full, [train_size, val_size])
    # replace val transform
    val_set.dataset = datasets.ImageFolder(train_dir, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, class_to_idx


def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train(args):
    set_seed(args.seed)

    data_root = args.data_root
    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_root, image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    num_classes = len(class_to_idx)
    model = build_model(num_classes)
    device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        t0 = time.time()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
              f"  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save best
            args.output_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "image_size": args.image_size,
            }
            out_path = args.output_dir / args.output_name
            torch.save(ckpt, out_path)
            print(f"  [saved best checkpoint] {out_path}")

    # final metadata
    meta = {
        "model_version": "cv-baseline-v0.1",
        "arch": "mobilenet_v2 (pretrained ImageNet)",
        "classes": sorted(class_to_idx.keys()),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": None,
        "best_val_acc": best_val_acc,
        "config": {
            "IMAGE_SIZE": args.image_size,
            "BATCH_SIZE": args.batch_size,
            "VAL_SPLIT": 0.2,
            "SEED": args.seed,
            "EPOCHS": args.epochs,
            "LR": args.lr,
            "FREEZE_BACKBONE": False,
            "NUM_WORKERS": args.num_workers,
            "MODEL_NAME": args.output_name.replace('.pt','')
        },
        "history": history,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # always save final checkpoint too (so downstream pipeline can load a .pt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "image_size": args.image_size,
    }
    final_path = args.output_dir / args.output_name
    torch.save(final_ckpt, final_path)
    print(f"Saved final checkpoint: {final_path}")

    with open(args.output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "Indonesian Food Image" / "Clean_Data",
                    help="Path containing train/ and test/ folders")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--output-name", default="mobilenetv2_food10.pt")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
