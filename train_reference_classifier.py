import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import datasets.datasetfactory as df
from analysis.reconstruction_fidelity import (
    OmniglotReferenceClassifier,
    omniglot_model_to_unit_interval,
)
from utils.utils import set_seed


class FilteredVisualDataset(Dataset):
    def __init__(self, base_dataset, num_classes=963, image_transform=None):
        self.base_dataset = base_dataset
        self.indices = [index for index, label in enumerate(base_dataset.targets) if int(label) < num_classes]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base_dataset[self.indices[index]]
        image = omniglot_model_to_unit_interval(image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, int(label)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += len(labels)
    return loss_sum / max(1, total), correct / max(1, total)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_base = df.DatasetFactory.get_dataset("omniglot", background=True, train=True, all=False)
    validation_base = df.DatasetFactory.get_dataset("omniglot", background=True, train=False, all=False)
    train_augmentation = transforms.RandomAffine(
        degrees=args.rotation,
        translate=(args.translation, args.translation),
        scale=(args.scale_min, args.scale_max),
        interpolation=InterpolationMode.BILINEAR,
        fill=1.0,
    ) if args.augment else None
    train_dataset = FilteredVisualDataset(train_base, args.num_classes, train_augmentation)
    validation_dataset = FilteredVisualDataset(validation_base, args.num_classes)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = OmniglotReferenceClassifier(args.num_classes, args.feature_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_correct += logits.argmax(dim=1).eq(labels).sum().item()
            train_total += len(labels)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_total,
            "train_accuracy": train_correct / train_total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_validation_accuracy": max(best_accuracy, validation_accuracy),
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation_accuracy > best_accuracy + args.min_delta:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "num_classes": args.num_classes,
                    "feature_dim": args.feature_dim,
                    "dropout": args.dropout,
                    "label_smoothing": args.label_smoothing,
                    "augmentation": {
                        "enabled": args.augment,
                        "rotation": args.rotation,
                        "translation": args.translation,
                        "scale_min": args.scale_min,
                        "scale_max": args.scale_max,
                    },
                    "input_space": "RGB [0,1]",
                    "training_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "seed": args.seed,
                },
                "best_epoch": epoch,
                "best_validation_accuracy": best_accuracy,
            }, args.output)
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                f"validation accuracy={best_accuracy:.6f}"
            )
            break

    with open(args.output + ".history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    print(
        f"Saved best checkpoint to {args.output}; best epoch={best_epoch}; "
        f"validation accuracy={best_accuracy:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the frozen independent Omniglot reference classifier.")
    parser.add_argument("--output", default="reference_models/omniglot_reference_seed9.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotation", type=float, default=10.0)
    parser.add_argument("--translation", type=float, default=0.10)
    parser.add_argument("--scale-min", type=float, default=0.90)
    parser.add_argument("--scale-max", type=float, default=1.10)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=963)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
