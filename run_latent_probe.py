import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_probe_data(run_dir):
    probe_dir = run_dir / "latent_probe_data"
    files = sorted(probe_dir.glob("latent_probe_event_*.npz"))
    if not files:
        raise FileNotFoundError(f"No latent probe npz files found in {probe_dir}")

    latents = []
    labels = []
    events = []
    for path in files:
        data = np.load(path)
        z = data["latents"].astype(np.float32)
        y = data["labels"].astype(np.int64).reshape(-1)
        event_index = int(data["event_index"][0]) if "event_index" in data else -1
        latents.append(z)
        labels.append(y)
        events.append(np.full(len(y), event_index, dtype=np.int64))

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0), np.concatenate(events, axis=0), len(files)


def remap_labels(labels):
    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    mapping = {label: idx for idx, label in enumerate(classes.tolist())}
    remapped = np.array([mapping[int(label)] for label in labels], dtype=np.int64)
    return remapped, classes


def stratified_split(labels, test_ratio, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_indices.extend(idx.tolist())
            continue
        test_count = max(1, int(round(len(idx) * test_ratio)))
        test_count = min(test_count, len(idx) - 1)
        test_indices.extend(idx[:test_count].tolist())
        train_indices.extend(idx[test_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.array(train_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def make_batches(x, y, batch_size, device, shuffle=True):
    indices = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx].to(device), y[batch_idx].to(device)


def evaluate(model, x, y, batch_size, device):
    model.eval()
    correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in make_batches(x, y, batch_size, device, shuffle=False):
            logits = model(batch_x)
            pred = logits.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            topk = min(5, logits.size(1))
            top5 = logits.topk(topk, dim=1).indices
            top5_correct += top5.eq(batch_y.view(-1, 1)).any(dim=1).sum().item()
            total += len(batch_y)
    return correct / total, top5_correct / total


def train_probe(latents, labels, args):
    labels, classes = remap_labels(labels)
    train_idx, test_idx = stratified_split(labels, args.test_ratio, args.seed)
    if len(test_idx) == 0:
        raise ValueError("No held-out samples for probe evaluation. Increase collected latent probe data.")

    train_x, test_x = standardize(latents[train_idx], latents[test_idx])
    train_y, test_y = labels[train_idx], labels[test_idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train = torch.from_numpy(train_x.astype(np.float32))
    y_train = torch.from_numpy(train_y.astype(np.int64))
    x_test = torch.from_numpy(test_x.astype(np.float32))
    y_test = torch.from_numpy(test_y.astype(np.int64))

    model = nn.Linear(x_train.size(1), len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in make_batches(x_train, y_train, args.batch_size, device, shuffle=True):
            loss = F.cross_entropy(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        top1, top5 = evaluate(model, x_test, y_test, args.batch_size, device)
        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5
            best_epoch = epoch

    return {
        "num_samples": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "num_classes": int(len(classes)),
        "best_epoch": int(best_epoch),
        "latent_probe_top1_accuracy": best_top1,
        "latent_probe_top5_accuracy": best_top5,
    }


def main(args):
    set_seed(args.seed)
    root = Path(args.analysis_dir).expanduser().resolve()
    run_dirs = [root / run_id for run_id in args.run] if args.run else sorted(path for path in root.iterdir() if path.is_dir())
    rows = []
    for run_dir in run_dirs:
        latents, labels, events, event_count = load_run_probe_data(run_dir)
        result = train_probe(latents, labels, args)
        result.update(
            {
                "run_id": run_dir.name,
                "event_count": int(event_count),
                "event_min": int(events.min()),
                "event_max": int(events.max()),
            }
        )
        rows.append(result)
        print(
            f"{run_dir.name}: top1={result['latent_probe_top1_accuracy']:.4f}, "
            f"top5={result['latent_probe_top5_accuracy']:.4f}, "
            f"classes={result['num_classes']}, samples={result['num_samples']}"
        )

    output_path = Path(args.output).expanduser().resolve() if args.output else root / "latent_probe_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "event_count",
        "event_min",
        "event_max",
        "num_samples",
        "num_train",
        "num_test",
        "num_classes",
        "best_epoch",
        "latent_probe_top1_accuracy",
        "latent_probe_top5_accuracy",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an offline linear probe on saved replay latent codes.")
    parser.add_argument("--analysis-dir", default="./20260507CSV")
    parser.add_argument("--run", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
