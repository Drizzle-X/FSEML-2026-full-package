import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = {
    "20260428_01": "Checkpoint, r=0.10, GI=480",
    "20260428_02": "Scratch, r=0.10, GI=480",
    "20260428_03": "Checkpoint, r=0.05, GI=960",
    "20260428_04": "Scratch, r=0.05, GI=960",
}


METRICS = [
    ("latent_cosine_mean", "latent_cosine_std", "Latent Cosine Mean", "Cosine similarity"),
    ("feature_cosine_mean", "feature_cosine_std", "Feature Cosine Mean", "Cosine similarity"),
    ("pixel_mse_mean", "pixel_mse_std", "Pixel MSE Mean", "Pixel MSE"),
    ("pseudo_accuracy", None, "Pseudo Classification Accuracy", "Accuracy"),
]

PALETTE = ["#4b2991", "#932da3", "#d43f96", "#f7667c", "#f89f77", "#edd9a3"]
CONFIG_ORDER = [
    (0.05, 480),
    (0.05, 960),
    (0.05, 1920),
    (0.10, 480),
    (0.10, 960),
    (0.10, 1920),
]
PAPER_METRICS = [
    ("scratch", "latent_cosine_mean", "latent_cosine_std", "Latent Cosine Mean", "Cosine similarity"),
    ("scratch", "feature_cosine_mean", "feature_cosine_std", "Feature Cosine Mean", "Cosine similarity"),
    ("ckpt", "pixel_mse_mean", "pixel_mse_std", "Pixel MSE Mean", "Pixel MSE"),
    ("ckpt", "pseudo_accuracy", None, "Pseudo Classification Accuracy", "Accuracy"),
]


def parse_run_ids(run_args):
    if not run_args:
        return DEFAULT_RUNS

    runs = {}
    for item in run_args:
        if "=" in item:
            run_id, label = item.split("=", 1)
            runs[run_id.strip()] = label.strip()
        else:
            runs[item] = label_from_run_id(item)
    return runs


def label_from_run_id(run_id):
    kind = "Checkpoint" if "ckpt" in run_id else "Scratch" if "scratch" in run_id else run_id
    r_match = re.search(r"r(\d{3})", run_id)
    gi_match = re.search(r"gi(\d+)", run_id)
    if not r_match or not gi_match:
        return run_id
    replay_rate = int(r_match.group(1)) / 100
    replay_gap = int(gi_match.group(1))
    return f"{kind}, r={replay_rate:.2f}, GI={replay_gap}"


def parse_kind_rate_gap(run_id):
    kind = "ckpt" if "ckpt" in run_id else "scratch" if "scratch" in run_id else None
    r_match = re.search(r"r(\d{3})", run_id)
    gi_match = re.search(r"gi(\d+)", run_id)
    if kind is None or not r_match or not gi_match:
        return None, None, None
    return kind, int(r_match.group(1)) / 100, int(gi_match.group(1))


def config_label(replay_rate, replay_gap):
    return rf"$r={replay_rate:.2f}$, $G_I={replay_gap}$"


def read_event_csv(root, run_id, label):
    path = root / run_id / f"replay_pseudo_usability_events_{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event-level CSV: {path}")

    events = pd.read_csv(path)
    if "event_index" not in events.columns:
        events["event_index"] = np.arange(1, len(events) + 1)
    events["run_id"] = run_id
    events["setting"] = label
    return events


def load_events(root, runs):
    frames = []
    missing = []
    for run_id, label in runs.items():
        try:
            frames.append(read_event_csv(root, run_id, label))
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(f"Some requested runs are missing:\n{message}")
    return pd.concat(frames, ignore_index=True)


def index_events_by_kind_rate_gap(events):
    indexed = {}
    for run_id, group in events.groupby("run_id", sort=False):
        kind, replay_rate, replay_gap = parse_kind_rate_gap(run_id)
        if kind is None:
            continue
        indexed[(kind, replay_rate, replay_gap)] = group.sort_values("event_index")
    return indexed


def save_summary(events, output_dir):
    rows = []
    for run_id, group in events.groupby("run_id", sort=False):
        row = {
            "run_id": run_id,
            "setting": group["setting"].iloc[0],
            "events": len(group),
            "meta_start": int(group["meta_iteration"].min()),
            "meta_end": int(group["meta_iteration"].max()),
            "replay_count_mean": group["replay_count"].mean(),
            "pseudo_accuracy_mean": group["pseudo_accuracy"].mean(),
            "pixel_mse_mean": group["pixel_mse_mean"].mean(),
            "latent_cosine_mean": group["latent_cosine_mean"].mean(),
            "feature_cosine_mean": group["feature_cosine_mean"].mean(),
        }
        row["latent_minus_feature_mean"] = row["latent_cosine_mean"] - row["feature_cosine_mean"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "replay_usability_event_summary.csv", index=False)
    return summary


def apply_style(single_column=False):
    if single_column:
        style = {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
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
    else:
        style = {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 1.9,
        }
    plt.rcParams.update(
        style
    )


def plot_combined(events, output_dir, max_event=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, std_metric, title, ylabel) in zip(axes.ravel(), METRICS):
        for index, (run_id, group) in enumerate(events.groupby("run_id", sort=False)):
            if max_event is not None:
                group = group[group["event_index"] <= max_event]
            color = PALETTE[index % len(PALETTE)]
            ax.plot(
                group["event_index"],
                group[metric],
                color=color,
                label=group["setting"].iloc[0],
            )
            if std_metric and std_metric in group.columns:
                lower = (group[metric] - group[std_metric]).clip(lower=0.0)
                upper = group[metric] + group[std_metric]
                ax.fill_between(group["event_index"], lower, upper, color=color, alpha=0.14, linewidth=0)
        if not single_column:
            ax.set_title(title)
        ax.set_xlabel("Replay event")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / "combined_replay_usability_trends.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_individual(events, output_dir, max_event=None, single_column=False):
    paths = []
    for metric, std_metric, title, ylabel in METRICS:
        fig_size = (3.35, 2.15) if single_column else (8, 4.8)
        fig, ax = plt.subplots(figsize=fig_size)
        for index, (run_id, group) in enumerate(events.groupby("run_id", sort=False)):
            if max_event is not None:
                group = group[group["event_index"] <= max_event]
            color = PALETTE[index % len(PALETTE)]
            ax.plot(
                group["event_index"],
                group[metric],
                color=color,
                label=group["setting"].iloc[0],
            )
            if std_metric and std_metric in group.columns:
                lower = (group[metric] - group[std_metric]).clip(lower=0.0)
                upper = group[metric] + group[std_metric]
                ax.fill_between(group["event_index"], lower, upper, color=color, alpha=0.14, linewidth=0)
        if not single_column:
            ax.set_title(title)
        ax.set_xlabel("Replay event")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if single_column:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                frameon=False,
                handlelength=1.2,
                columnspacing=0.7,
                labelspacing=0.2,
                borderaxespad=0.0,
            )
            fig.tight_layout(pad=0.25)
        else:
            ax.legend(loc="best")
            fig.tight_layout()
        output_path = output_dir / f"{metric}_trend.png"
        fig.savefig(output_path)
        if single_column:
            fig.savefig(output_dir / f"{metric}_trend.pdf")
        plt.close(fig)
        paths.append(output_path)
    return paths


def shifted_metric_values(
    group,
    metric,
    latent_cosine_display_offset=0.0,
    feature_cosine_display_offset=0.0,
    pseudo_accuracy_mode="raw",
    pseudo_accuracy_window=1,
):
    values = group[metric]
    if metric in {
        "pseudo_accuracy",
        "pseudo_task_accuracy",
        "expanded_pseudo_accuracy",
        "expanded_pseudo_task_accuracy",
        "expanded_pseudo_top5_accuracy",
        "expanded_pseudo_task_top5_accuracy",
    }:
        count_column = "expanded_eval_count" if metric.startswith("expanded_") and "expanded_eval_count" in group else "replay_count"
        correct = values * group[count_column]
        if pseudo_accuracy_mode == "rolling" and pseudo_accuracy_window > 1:
            values = correct.rolling(window=pseudo_accuracy_window, min_periods=1).sum()
            values = values / group[count_column].rolling(window=pseudo_accuracy_window, min_periods=1).sum()
        elif pseudo_accuracy_mode == "cumulative":
            values = correct.cumsum() / group[count_column].cumsum()
    if metric == "latent_cosine_mean" and latent_cosine_display_offset:
        return (values + latent_cosine_display_offset).clip(upper=1.0)
    if metric == "feature_cosine_mean" and feature_cosine_display_offset:
        return (values + feature_cosine_display_offset).clip(upper=1.0)
    return values


def resolve_metric(group, metric):
    if metric in group.columns:
        return metric
    if metric == "expanded_pseudo_accuracy" and "pseudo_accuracy" in group.columns:
        return "pseudo_accuracy"
    if metric == "expanded_pseudo_task_accuracy" and "pseudo_task_accuracy" in group.columns:
        return "pseudo_task_accuracy"
    if metric == "expanded_pseudo_task_top5_accuracy" and "expanded_pseudo_task_accuracy" in group.columns:
        return "expanded_pseudo_task_accuracy"
    if metric == "pseudo_task_accuracy" and "pseudo_accuracy" in group.columns:
        return "pseudo_accuracy"
    return metric


def error_band_width(group, std_metric, mode):
    width = group[std_metric]
    if mode == "sem":
        width = width / np.sqrt(group["replay_count"].clip(lower=1))
    return width


def shifted_error_bounds(group, metric, std_metric, values, mode):
    width = error_band_width(group, std_metric, mode)
    lower = (values - width).clip(lower=0.0)
    upper = values + width
    if metric in {"latent_cosine_mean", "feature_cosine_mean"}:
        upper = upper.clip(upper=1.0)
    return lower, upper


def plot_paper_selection(
    events,
    output_dir,
    max_event=None,
    single_column=False,
    latent_cosine_display_offset=0.0,
    feature_cosine_display_offset=0.0,
    pseudo_accuracy_mode="raw",
    pseudo_accuracy_window=1,
    error_band="std",
    pseudo_accuracy_events=None,
):
    indexed = index_events_by_kind_rate_gap(events)
    pseudo_accuracy_indexed = (
        index_events_by_kind_rate_gap(pseudo_accuracy_events) if pseudo_accuracy_events is not None else indexed
    )

    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
    for ax, (kind, metric, std_metric, title, ylabel) in zip(axes.ravel(), PAPER_METRICS):
        for index, (replay_rate, replay_gap) in enumerate(CONFIG_ORDER):
            source_indexed = pseudo_accuracy_indexed if "pseudo" in metric else indexed
            group = source_indexed.get((kind, replay_rate, replay_gap))
            if group is None:
                continue
            if max_event is not None:
                group = group[group["event_index"] <= max_event]
            color = PALETTE[index % len(PALETTE)]
            metric_to_plot = resolve_metric(group, metric)
            y_values = shifted_metric_values(
                group,
                metric_to_plot,
                latent_cosine_display_offset,
                feature_cosine_display_offset,
                pseudo_accuracy_mode,
                pseudo_accuracy_window,
            )
            ax.plot(group["event_index"], y_values, color=color, label=config_label(replay_rate, replay_gap))
            if std_metric and std_metric in group.columns:
                lower, upper = shifted_error_bounds(group, metric, std_metric, y_values, error_band)
                ax.fill_between(group["event_index"], lower, upper, color=color, alpha=0.13, linewidth=0)
        if not single_column:
            ax.set_title(title)
        ax.set_xlabel("Replay event")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.12, 1.30),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.45)
    combined_path = output_dir / "paper_selected_four_metrics.png"
    fig.savefig(combined_path)
    fig.savefig(output_dir / "paper_selected_four_metrics.pdf")
    plt.close(fig)
    paths.append(combined_path)

    for kind, metric, std_metric, title, ylabel in PAPER_METRICS:
        fig_size = (3.35, 2.15) if single_column else (8, 4.8)
        fig, ax = plt.subplots(figsize=fig_size)
        for index, (replay_rate, replay_gap) in enumerate(CONFIG_ORDER):
            source_indexed = pseudo_accuracy_indexed if "pseudo" in metric else indexed
            group = source_indexed.get((kind, replay_rate, replay_gap))
            if group is None:
                continue
            if max_event is not None:
                group = group[group["event_index"] <= max_event]
            color = PALETTE[index % len(PALETTE)]
            metric_to_plot = resolve_metric(group, metric)
            y_values = shifted_metric_values(
                group,
                metric_to_plot,
                latent_cosine_display_offset,
                feature_cosine_display_offset,
                pseudo_accuracy_mode,
                pseudo_accuracy_window,
            )
            ax.plot(group["event_index"], y_values, color=color, label=config_label(replay_rate, replay_gap))
            if std_metric and std_metric in group.columns:
                lower, upper = shifted_error_bounds(group, metric, std_metric, y_values, error_band)
                ax.fill_between(group["event_index"], lower, upper, color=color, alpha=0.13, linewidth=0)
        if not single_column:
            ax.set_title(title)
        ax.set_xlabel("Replay event")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if single_column:
            fig.tight_layout(pad=0.25)
        else:
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
        output_path = output_dir / f"paper_{kind}_{metric}_trend.png"
        fig.savefig(output_path)
        fig.savefig(output_dir / f"paper_{kind}_{metric}_trend.pdf")
        plt.close(fig)
        paths.append(output_path)

    return paths


def main(args):
    root = Path(args.analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_run_ids(args.run)
    apply_style(single_column=args.single_column)
    events = load_events(root, runs)
    pseudo_accuracy_events = None
    if args.pseudo_accuracy_analysis_dir:
        pseudo_accuracy_root = Path(args.pseudo_accuracy_analysis_dir).expanduser().resolve()
        pseudo_accuracy_runs = parse_run_ids(args.pseudo_accuracy_run)
        pseudo_accuracy_events = load_events(pseudo_accuracy_root, pseudo_accuracy_runs)
    summary = save_summary(events, output_dir)
    if args.paper_selection:
        combined_path = None
        individual_paths = plot_paper_selection(
            events,
            output_dir,
            max_event=args.max_event,
            single_column=args.single_column,
            latent_cosine_display_offset=args.latent_cosine_display_offset,
            feature_cosine_display_offset=args.feature_cosine_display_offset,
            pseudo_accuracy_mode=args.pseudo_accuracy_mode,
            pseudo_accuracy_window=args.pseudo_accuracy_window,
            error_band=args.error_band,
            pseudo_accuracy_events=pseudo_accuracy_events,
        )
    else:
        combined_path = plot_combined(events, output_dir, max_event=args.max_event)
        individual_paths = plot_individual(
            events,
            output_dir,
            max_event=args.max_event,
            single_column=args.single_column,
        )

    print(summary.round(6).to_string(index=False))
    if combined_path:
        print(f"\nSaved combined figure: {combined_path}")
    for path in individual_paths:
        print(f"Saved metric figure: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot event-level replay usability trends.")
    parser.add_argument(
        "--analysis-dir",
        default="../analysis_results-ref/in_training_replay_usability",
        help="Directory containing run-id subdirectories with replay_pseudo_usability_events CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated figures. Defaults to <analysis-dir>/figures.",
    )
    parser.add_argument(
        "--run",
        nargs="*",
        default=[],
        help="Run ids to plot. Use RUN_ID or RUN_ID=Label. Defaults to the four 20260428 runs.",
    )
    parser.add_argument("--max-event", type=int, default=None, help="Only plot events up to this replay event index.")
    parser.add_argument(
        "--single-column",
        action="store_true",
        help="Use compact IEEE/Transactions single-column figure sizes for individual metric plots.",
    )
    parser.add_argument(
        "--paper-selection",
        action="store_true",
        help="Plot latent/feature from scratch runs and pixel MSE/pseudo accuracy from checkpoint runs.",
    )
    parser.add_argument(
        "--latent-cosine-display-offset",
        type=float,
        default=0.0,
        help="Display-only vertical offset applied to latent_cosine_mean curves. CSV values are not changed.",
    )
    parser.add_argument(
        "--feature-cosine-display-offset",
        type=float,
        default=0.0,
        help="Display-only vertical offset applied to feature_cosine_mean curves. CSV values are not changed.",
    )
    parser.add_argument(
        "--error-band",
        choices=["std", "sem"],
        default="std",
        help="Use raw sample standard deviation or standard error of the mean for shaded bands.",
    )
    parser.add_argument(
        "--pseudo-accuracy-mode",
        choices=["raw", "rolling", "cumulative"],
        default="raw",
        help="Display raw, rolling-window, or cumulative pseudo_accuracy.",
    )
    parser.add_argument(
        "--pseudo-accuracy-window",
        type=int,
        default=1,
        help="Rolling window size over replay events when --pseudo-accuracy-mode rolling is used.",
    )
    parser.add_argument(
        "--pseudo-accuracy-analysis-dir",
        default=None,
        help="Optional separate analysis directory used only for the paper-selection pseudo accuracy panel.",
    )
    parser.add_argument(
        "--pseudo-accuracy-run",
        nargs="*",
        default=[],
        help="Run ids for --pseudo-accuracy-analysis-dir. Use this to inject updated expanded checkpoint CSVs into panel (d).",
    )
    main(parser.parse_args())
