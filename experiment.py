"""Reproducible Oxford-IIIT Pet skip-connection ablation experiment.

Examples:
    python experiment.py prepare
    python experiment.py train --model baseline
    python experiment.py train --model zeroed_skip
    python experiment.py evaluate --checkpoint outputs/checkpoints/baseline_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
NUM_CLASSES = 2
IGNORE_INDEX = 255
SEED = 20261114


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ids(path: Path) -> List[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def split_ids(seed: int = SEED) -> Dict[str, List[str]]:
    """Create the fixed 90/10 train/validation split from official trainval IDs."""
    ids = read_ids(DATA / "annotations" / "trainval.txt")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(len(ids) * 0.9)
    return {"train": sorted(ids[:n_train]), "validation": sorted(ids[n_train:]),
            "test": sorted(read_ids(DATA / "annotations" / "test.txt"))}


class PetSegmentation(Dataset):
    def __init__(self, ids: Sequence[str], size: int = 256, augment: bool = False):
        self.ids, self.size, self.augment = list(ids), size, augment
        self.image_dir = DATA / "images"
        self.mask_dir = DATA / "annotations" / "trimaps"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.ids[index]
        image = Image.open(self.image_dir / f"{name}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{name}.png").convert("L")
        if self.augment and random.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        # Trimap: 1 foreground, 2 background, 3 boundary/not classified.
        mask_array = np.asarray(mask, dtype=np.int64)
        remapped = np.full(mask_array.shape, IGNORE_INDEX, dtype=np.int64)
        remapped[mask_array == 1] = 1
        remapped[mask_array == 2] = 0
        return torch.from_numpy(image_array.transpose(2, 0, 1)).float(), torch.from_numpy(remapped)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """The same architecture is used for both models; only skip tensor values differ."""
    def __init__(self, zeroed_skips: bool = False):
        super().__init__()
        self.zeroed_skips = zeroed_skips
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2 = DoubleConv(3, 32), DoubleConv(32, 64)
        self.enc3, self.bottleneck = DoubleConv(64, 128), DoubleConv(128, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), DoubleConv(64, 32)
        self.head = nn.Conv2d(32, NUM_CLASSES, 1)

    def _skip(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor) if self.zeroed_skips else tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self._skip(e3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._skip(e2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._skip(e1)], dim=1))
        return self.head(d1)


@dataclass
class Config:
    seed: int = SEED
    image_size: int = 128
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0


def device_name() -> str:
    return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def metric_state(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    return torch.bincount(
        (target[valid] * NUM_CLASSES + prediction[valid]).reshape(-1),
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, object]:
    diagonal = confusion.diag().float()
    union = confusion.sum(1).float() + confusion.sum(0).float() - diagonal
    iou = torch.where(union > 0, diagonal / union, torch.zeros_like(union))
    dice = torch.where(confusion.sum(1) + confusion.sum(0) > 0,
                       2 * diagonal / (confusion.sum(1) + confusion.sum(0)).float(),
                       torch.zeros_like(diagonal))
    return {"per_class_iou": iou.tolist(), "miou": float(iou.mean()),
            "dice": float(dice.mean()), "confusion_matrix": confusion.tolist(),
            "valid_pixels": int(confusion.sum())}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, object], float]:
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    started = time.perf_counter()
    for images, masks in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        confusion += metric_state(predictions, masks)
    return metrics_from_confusion(confusion), time.perf_counter() - started


def make_loader(ids: Sequence[str], config: Config, augment: bool, shuffle: bool) -> DataLoader:
    return DataLoader(PetSegmentation(ids, config.image_size, augment), config.batch_size,
                      shuffle=shuffle, num_workers=config.num_workers, pin_memory=torch.cuda.is_available())


def train(model_name: str, config: Config) -> Path:
    if model_name not in {"baseline", "zeroed_skip"}:
        raise ValueError("model must be baseline or zeroed_skip")
    seed_everything(config.seed)
    splits = split_ids(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(model_name == "zeroed_skip").to(device)
    train_loader = make_loader(splits["train"], config, True, True)
    validation_loader = make_loader(splits["validation"], config, False, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    best_loss, best_epoch, updates, epochs_completed = float("inf"), 0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), masks.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            updates += 1
        epochs_completed = epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in validation_loader:
                val_loss += float(loss_fn(model(images.to(device)), masks.to(device)))
        val_loss /= max(1, len(validation_loader))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            OUTPUTS.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "model_name": model_name,
                        "config": asdict(config), "seed": config.seed, "best_epoch": epoch,
                        "trainable_parameters": trainable_parameters(model)},
                       OUTPUTS / "checkpoints" / f"{model_name}_best.pt")
        if epoch - best_epoch >= config.patience:
            break
    duration = time.perf_counter() - started
    summary = {"model": model_name, "max_epochs": config.max_epochs,
               "actual_epochs_completed": epochs_completed, "optimizer_updates": updates,
               "best_validation_epoch": best_epoch, "training_duration_seconds": duration,
               "trainable_parameters": trainable_parameters(model), "hardware": device_name(),
               "configuration": asdict(config), "zeroed_skip_fusion": model_name == "zeroed_skip"}
    OUTPUTS.joinpath("results").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results" / f"{model_name}_training.json").write_text(json.dumps(summary, indent=2))
    return OUTPUTS / "checkpoints" / f"{model_name}_best.pt"


def evaluate_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    seed_everything(checkpoint["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(checkpoint["model_name"] == "zeroed_skip")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    loader = make_loader(split_ids(config.seed)["test"], config, False, False)
    metrics, latency = evaluate_model(model, loader, device)
    result = {"model": checkpoint["model_name"], "metrics_over_valid_non_ignored_pixels": metrics,
              "inference_benchmark_duration_seconds": latency, "hardware": device_name(),
              "trainable_parameters": trainable_parameters(model), "seed": checkpoint["seed"]}
    OUTPUTS.joinpath("results").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results" / f"{checkpoint['model_name']}_evaluation.json").write_text(json.dumps(result, indent=2))
    write_audit()


def write_audit() -> None:
    """Write a comparison audit from whatever model records are currently available."""
    records = {}
    for model_name in ("baseline", "zeroed_skip"):
        training_path = OUTPUTS / "results" / f"{model_name}_training.json"
        evaluation_path = OUTPUTS / "results" / f"{model_name}_evaluation.json"
        if training_path.exists():
            records[model_name] = {"training": json.loads(training_path.read_text())}
        if evaluation_path.exists():
            records.setdefault(model_name, {})["evaluation"] = json.loads(evaluation_path.read_text())
    lines = [
        "# Reproducibility audit",
        "",
        "Both experiments use `outputs/splits/dataset_splits.json`, the same transformations, "
        "optimization settings, and evaluation implementation. Test IDs are never loaded by training.",
        "",
        "| Model | Max epochs | Actual epochs | Optimizer updates | Best validation epoch | "
        "Train seconds | Parameters | Inference seconds | Hardware |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in ("baseline", "zeroed_skip"):
        training = records.get(model_name, {}).get("training", {})
        evaluation = records.get(model_name, {}).get("evaluation", {})
        lines.append(
            f"| {model_name} | {training.get('max_epochs', 'pending')} | "
            f"{training.get('actual_epochs_completed', 'pending')} | "
            f"{training.get('optimizer_updates', 'pending')} | "
            f"{training.get('best_validation_epoch', 'pending')} | "
            f"{training.get('training_duration_seconds', 'pending')} | "
            f"{training.get('trainable_parameters', evaluation.get('trainable_parameters', 'pending'))} | "
            f"{evaluation.get('inference_benchmark_duration_seconds', 'pending')} | "
            f"{training.get('hardware', evaluation.get('hardware', 'pending'))} |"
        )
    train_durations = [
        records.get(name, {}).get("training", {}).get("training_duration_seconds")
        for name in ("baseline", "zeroed_skip")
    ]
    if all(isinstance(value, (int, float)) for value in train_durations):
        lines.extend(["", f"Training duration difference (zeroed_skip - baseline): "
                          f"{train_durations[1] - train_durations[0]:.3f} seconds."])
    lines.extend([
        "",
        "The zeroed-skip model is a controlled zeroed-skip-fusion ablation: it uses the exact "
        "same encoder features, decoder blocks, concatenation widths, and trainable parameters, "
        "but replaces each skip tensor with `torch.zeros_like`.",
        "Checkpoints store their configuration and seed. Evaluation metrics are computed only over "
        "valid, non-ignored pixels; ignore value 255 is not a segmentation class.",
    ])
    OUTPUTS.joinpath("results").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results" / "REPRODUCIBILITY_AUDIT.md").write_text("\n".join(lines) + "\n")


def prepare() -> None:
    splits = split_ids()
    OUTPUTS.joinpath("splits").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "splits" / "dataset_splits.json").write_text(json.dumps(
        {"seed": SEED, "method": "shuffle official trainval IDs with random.Random(seed), 90/10 split",
         "splits": splits}, indent=2))
    ignored = valid = 0
    for name in splits["train"] + splits["validation"] + splits["test"]:
        values = np.asarray(Image.open(DATA / "annotations" / "trimaps" / f"{name}.png"))
        ignored += int((values == 3).sum())
        valid += int((values != 3).sum())
    dataset_image_count = len([
        p for p in (DATA / "images").glob("*.jpg") if not p.name.startswith("._")
    ])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--model", choices=["baseline", "zeroed_skip"])
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        if not args.model:
            parser.error("--model is required for train")
        train(args.model, Config())
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluate")
        evaluate_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
