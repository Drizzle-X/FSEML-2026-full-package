import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis.reconstruction_fidelity import CIFAR100ReferenceClassifier
from datasets.cifar100_continual import load_cifar100, cifar100_model_to_unit_interval
from utils.utils import set_seed


class UnitIntervalDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return cifar100_model_to_unit_interval(image.unsqueeze(0)).squeeze(0), int(label)


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_dataset = UnitIntervalDataset(load_cifar100(
        args.data_root, train=True, image_size=28, augment=True, download_if_missing=True,
    ))
    validation_dataset = UnitIntervalDataset(load_cifar100(
        args.data_root, train=False, image_size=28, augment=False, download_if_missing=True,
    ))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = CIFAR100ReferenceClassifier(100, args.dropout).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_accuracy = -1.0
    history = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_correct += logits.argmax(dim=1).eq(labels).sum().item()
            train_total += len(labels)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / train_total,
            "train_accuracy": train_correct / train_total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "dataset": "cifar100",
                    "num_classes": 100,
                    "feature_dim": model.feature_dim,
                    "dropout": args.dropout,
                    "input_space": "RGB [0,1] with normalization inside classifier",
                    "seed": args.seed,
                },
                "best_epoch": epoch,
                "best_validation_accuracy": best_accuracy,
            }, args.output)
        scheduler.step()

    with open(args.output + ".history.json", "w") as handle:
        json.dump(history, handle, indent=2)
    print(f"Saved best checkpoint to {args.output}; validation accuracy={best_accuracy:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the frozen CIFAR100 reference classifier.")
    parser.add_argument("--data-root", default="../data/cifar100")
    parser.add_argument("--output", default="reference_models/cifar100_resnet18_seed9.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
