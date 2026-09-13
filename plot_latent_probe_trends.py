import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


PALETTE = ["#4b2991", "#932da3", "#d43f96", "#f7667c", "#f89f77", "#edd9a3"]
CONFIG_ORDER = [
    (0.05, 480),
    (0.05, 960),
    (0.05, 1920),
    (0.10, 480),
    (0.10, 960),
    (0.10, 1920),
]


def parse_rate_gap(run_id):
    r_match = re.search(r"r(\d{3})", run_id)
    gi_match = re.search(r"gi(\d+)", run_id)
    if not r_match or not gi_match:
        return None, None
    return int(r_match.group(1)) / 100, int(gi_match.group(1))


def config_label(replay_rate, replay_gap):
    return rf"$r={replay_rate:.2f}$, $G_I={replay_gap}$"


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "lines.linewidth": 1.75,
        }
    )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_probe_events(run_dir):
    files = sorted((run_dir / "latent_probe_data").glob("latent_probe_event_*.npz"))
    if not files:
        raise FileNotFoundError(f"No latent probe npz files found in {run_dir / 'latent_probe_data'}")
    frames = []
    for path in files:
        data = np.load(path)
        event_index = int(data["event_index"][0]) if "event_index" in data else len(frames) + 1
        frames.append(
            {
                "event_index": event_index,
                "latents": data["latents"].astype(np.float32),
                "labels": data["labels"].astype(np.int64).reshape(-1),
            }
        )
    return frames


def remap_labels(labels):
    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    mapping = {label: idx for idx, label in enumerate(classes.tolist())}
    return np.array([mapping[int(label)] for label in labels], dtype=np.int64), classes


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


def standardize(train_x, all_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (all_x - mean) / std


def batches(x, y, batch_size, device, shuffle=True):
    order = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx].to(device), y[idx].to(device), idx


def train_linear_probe(x_train, y_train, num_classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = nn.Linear(x_train.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        for bx, by, _ in batches(x_train, y_train, args.batch_size, device, shuffle=True):
            loss = F.cross_entropy(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, device


def predict(model, x, batch_size, device):
    preds = []
    top5s = []
    model.eval()
    with torch.no_grad():
        dummy_y = torch.zeros(len(x), dtype=torch.long)
        for bx, _, _ in batches(x, dummy_y, batch_size, device, shuffle=False):
            logits = model(bx)
            preds.append(logits.argmax(dim=1).cpu())
            topk = min(5, logits.size(1))
            top5s.append(logits.topk(topk, dim=1).indices.cpu())
    return torch.cat(preds).numpy(), torch.cat(top5s).numpy()


def evaluate_run(run_dir, args):
    event_items = load_probe_events(run_dir)
    latents = np.concatenate([item["latents"] for item in event_items], axis=0)
    raw_labels = np.concatenate([item["labels"] for item in event_items], axis=0)
    event_indices = np.concatenate(
        [np.full(len(item["labels"]), item["event_index"], dtype=np.int64) for item in event_items],
        axis=0,
    )
    labels, classes = remap_labels(raw_labels)
    train_idx, test_idx = stratified_split(labels, args.test_ratio, args.seed)
    standardized = standardize(latents[train_idx], latents)

    x_train = torch.from_numpy(standardized[train_idx].astype(np.float32))
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64))
    x_all = torch.from_numpy(standardized.astype(np.float32))

    model, device = train_linear_probe(x_train, y_train, len(classes), args)
    pred, top5 = predict(model, x_all, args.batch_size, device)

    test_mask = np.zeros(len(labels), dtype=bool)
    test_mask[test_idx] = True
    rows = []
    for event_index in sorted(np.unique(event_indices)):
        mask = test_mask & (event_indices == event_index)
        if not mask.any():
            continue
        event_labels = labels[mask]
        event_pred = pred[mask]
        event_top5 = top5[mask]
        rows.append(
            {
                "run_id": run_dir.name,
                "event_index": int(event_index),
                "test_count": int(mask.sum()),
                "latent_probe_top1_accuracy": float((event_pred == event_labels).mean()),
                "latent_probe_top5_accuracy": float((event_top5 == event_labels[:, None]).any(axis=1).mean()),
            }
        )
    return rows


def smooth_metric(group, metric, mode, window):
    values = group[metric].astype(float)
    counts = group["test_count"].astype(float)
    correct = values * counts
    if mode == "rolling":
        return correct.rolling(window=window, min_periods=1).sum() / counts.rolling(window=window, min_periods=1).sum()
    if mode == "cumulative":
        return correct.cumsum() / counts.cumsum()
    return values


def plot_metric(events, metric, title, output_path, mode="raw", window=5):
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    for index, (replay_rate, replay_gap) in enumerate(CONFIG_ORDER):
        matched = [
            run_id
            for run_id in events["run_id"].drop_duplicates()
            if parse_rate_gap(run_id) == (replay_rate, replay_gap)
        ]
        if not matched:
            continue
        group = events[events["run_id"] == matched[0]].sort_values("event_index")
        y_values = smooth_metric(group, metric, mode, window)
        ax.plot(
            group["event_index"],
            y_values,
            color=PALETTE[index % len(PALETTE)],
            label=config_label(replay_rate, replay_gap),
        )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Replay event")
    ax.set_ylabel("Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True,
        handlelength=1.15,
        columnspacing=0.55,
        labelspacing=0.15,
        borderaxespad=0.0,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cfcfcf")
        legend.get_frame().set_linewidth(0.7)
        legend.get_frame().set_alpha(0.86)
    fig.tight_layout(pad=0.25)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_comparison_grid(events, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.5))
    specs = [
        ("latent_probe_top1_accuracy", "Top-1 Raw", "raw"),
        ("latent_probe_top1_accuracy", "Top-1 Rolling, w=5", "rolling"),
        ("latent_probe_top1_accuracy", "Top-1 Cumulative", "cumulative"),
        ("latent_probe_top5_accuracy", "Top-5 Raw", "raw"),
        ("latent_probe_top5_accuracy", "Top-5 Rolling, w=5", "rolling"),
        ("latent_probe_top5_accuracy", "Top-5 Cumulative", "cumulative"),
    ]
    for ax, (metric, title, mode) in zip(axes.ravel(), specs):
        for index, (replay_rate, replay_gap) in enumerate(CONFIG_ORDER):
            matched = [
                run_id
                for run_id in events["run_id"].drop_duplicates()
                if parse_rate_gap(run_id) == (replay_rate, replay_gap)
            ]
            if not matched:
                continue
            group = events[events["run_id"] == matched[0]].sort_values("event_index")
            ax.plot(
                group["event_index"],
                smooth_metric(group, metric, mode, 5),
                color=PALETTE[index % len(PALETTE)],
                label=config_label(replay_rate, replay_gap),
            )
        ax.set_title(title)
        ax.set_xlabel("Replay event")
        ax.set_ylabel("Accuracy")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.45)
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main(args):
    set_seed(args.seed)
    apply_style()
    root = Path(args.analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "latent_probe_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = (
        [root / run_id for run_id in args.run]
        if args.run
        else sorted(path for path in root.iterdir() if (path / "latent_probe_data").is_dir())
    )

    rows = []
    for run_dir in run_dirs:
        run_rows = evaluate_run(run_dir, args)
        rows.extend(run_rows)
        top1 = np.mean([row["latent_probe_top1_accuracy"] for row in run_rows])
        top5 = np.mean([row["latent_probe_top5_accuracy"] for row in run_rows])
        print(f"{run_dir.name}: event-mean top1={top1:.4f}, top5={top5:.4f}")

    events = pd.DataFrame(rows)
    csv_path = output_dir / "latent_probe_event_trends.csv"
    events.to_csv(csv_path, index=False)
    plot_specs = [
        ("raw", "Latent Probe Top-1 Accuracy", "latent_probe_top1_accuracy_trend"),
        ("raw", "Latent Probe Top-5 Accuracy", "latent_probe_top5_accuracy_trend"),
        ("rolling", "Latent Probe Top-1 Accuracy (Rolling, w=5)", "latent_probe_top1_accuracy_rolling_w5_trend"),
        ("rolling", "Latent Probe Top-5 Accuracy (Rolling, w=5)", "latent_probe_top5_accuracy_rolling_w5_trend"),
        ("cumulative", "Latent Probe Top-1 Accuracy (Cumulative)", "latent_probe_top1_accuracy_cumulative_trend"),
        ("cumulative", "Latent Probe Top-5 Accuracy (Cumulative)", "latent_probe_top5_accuracy_cumulative_trend"),
    ]
    for mode, title, stem in plot_specs:
        metric = "latent_probe_top5_accuracy" if "top5" in stem else "latent_probe_top1_accuracy"
        plot_metric(events, metric, title, output_dir / stem, mode=mode, window=5)
    plot_comparison_grid(events, output_dir / "latent_probe_accuracy_raw_rolling_cumulative_comparison")
    print(f"Saved {csv_path}")
    for _, _, stem in plot_specs:
        print(f"Saved {output_dir / (stem + '.png')}")
    print(f"Saved {output_dir / 'latent_probe_accuracy_raw_rolling_cumulative_comparison.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot event-level latent probe accuracy trends.")
    parser.add_argument("--analysis-dir", default="./20260509CSV")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
