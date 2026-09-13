import argparse
import csv
import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "Scratch": "omniglot_scratch_gray_seed9",
    "Warm-start": "omniglot_warm_start_seed9_20260803",
}

EVENT_COLUMNS = {
    "pixel_mse": "fidelity_pixel_mse_mean",
    "ssim": "fidelity_ssim_mean",
    "frozen_feature_cosine": "fidelity_frozen_feature_cosine_mean",
    "independent_real_accuracy": "fidelity_independent_real_accuracy",
    "independent_pseudo_accuracy": "fidelity_independent_pseudo_accuracy",
    "label_preservation_rate": "fidelity_label_preservation_rate",
}


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def entropy_from_counts(counts):
    values = np.asarray(list(counts.values()), dtype=float)
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    final_rows = []
    event_rows = []
    diagnostics = {}
    prediction_rows = []

    for method, run_id in RUNS.items():
        run_dir = os.path.join(args.analysis_root, run_id)
        summary_path = glob.glob(os.path.join(run_dir, "reconstruction_fidelity_final_buffer_summary_*.json"))[0]
        event_path = glob.glob(os.path.join(run_dir, "replay_pseudo_usability_events_*.csv"))[0]
        sample_path = glob.glob(os.path.join(run_dir, "reconstruction_fidelity_final_buffer_samples_*.csv"))[0]
        with open(summary_path) as handle:
            summary = json.load(handle)
        final_rows.append({
            "method": method,
            "sample_count": int(summary["sample_count"]),
            "pixel_mse_mean": summary["pixel_mse_mean"],
            "pixel_mse_std": summary["pixel_mse_std"],
            "ssim_mean": summary["ssim_mean"],
            "ssim_std": summary["ssim_std"],
            "frozen_feature_cosine_mean": summary["frozen_feature_cosine_mean"],
            "frozen_feature_cosine_std": summary["frozen_feature_cosine_std"],
            "independent_real_accuracy": summary["independent_real_accuracy"],
            "independent_pseudo_accuracy": summary["independent_pseudo_accuracy"],
            "label_preservation_rate": summary["label_preservation_rate"],
        })

        raw_events = read_csv(event_path)
        method_events = []
        for index, row in enumerate(raw_events, start=1):
            processed = {
                "method": method,
                "event_index": index,
                "meta_iteration": int(row["meta_iteration"]),
                "sample_count": int(row["fidelity_sample_count"]),
            }
            for output_name, source_name in EVENT_COLUMNS.items():
                processed[output_name] = float(row[source_name])
            event_rows.append(processed)
            method_events.append(processed)

        final_samples = read_csv(sample_path)
        real_predictions = Counter(int(row["real_prediction"]) for row in final_samples)
        pseudo_predictions = Counter(int(row["pseudo_prediction"]) for row in final_samples)
        true_labels = Counter(int(row["label"]) for row in final_samples)
        top_pseudo = pseudo_predictions.most_common(10)
        for rank, (class_id, count) in enumerate(top_pseudo, start=1):
            prediction_rows.append({
                "method": method,
                "rank": rank,
                "pseudo_predicted_class": class_id,
                "count": count,
                "share": count / len(final_samples),
            })

        cosines = [row["frozen_feature_cosine"] for row in method_events]
        pseudo_acc = [row["independent_pseudo_accuracy"] for row in method_events]
        lpr = [row["label_preservation_rate"] for row in method_events]
        diagnostics[method] = {
            "events": len(method_events),
            "first_event": method_events[0],
            "last_event": method_events[-1],
            "early_10_event_mean": {
                key: float(np.mean([row[key] for row in method_events[:10]]))
                for key in ("pixel_mse", "ssim", "frozen_feature_cosine", "independent_pseudo_accuracy", "label_preservation_rate")
            },
            "late_10_event_mean": {
                key: float(np.mean([row[key] for row in method_events[-10:]]))
                for key in ("pixel_mse", "ssim", "frozen_feature_cosine", "independent_pseudo_accuracy", "label_preservation_rate")
            },
            "event_correlation_feature_cosine_vs_pseudo_accuracy": finite_correlation(cosines, pseudo_acc),
            "event_correlation_feature_cosine_vs_lpr": finite_correlation(cosines, lpr),
            "final_unique_true_labels": len(true_labels),
            "final_unique_real_predictions": len(real_predictions),
            "final_unique_pseudo_predictions": len(pseudo_predictions),
            "final_pseudo_prediction_entropy_bits": entropy_from_counts(pseudo_predictions),
            "final_pseudo_prediction_top1_share": top_pseudo[0][1] / len(final_samples),
            "final_pseudo_prediction_top5_share": sum(count for _, count in top_pseudo[:5]) / len(final_samples),
        }

    write_csv(os.path.join(args.output_dir, "final_buffer_main_table.csv"), final_rows)
    write_csv(os.path.join(args.output_dir, "replay_event_metrics.csv"), event_rows)
    write_csv(os.path.join(args.output_dir, "pseudo_prediction_top10.csv"), prediction_rows)
    with open(os.path.join(args.output_dir, "diagnostic_summary.json"), "w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)

    figure, axes = plt.subplots(3, 2, figsize=(12, 13), sharex=True)
    panels = [
        ("pixel_mse", "Pixel MSE ↓"),
        ("ssim", "SSIM ↑"),
        ("frozen_feature_cosine", "Frozen feature cosine ↑"),
        ("independent_pseudo_accuracy", "Independent classifier ACC ↑"),
        ("label_preservation_rate", "Label preservation rate ↑"),
    ]
    colors = {"Scratch": "#2563EB", "Warm-start": "#D97706"}
    for axis, (key, label) in zip(axes.flat, panels):
        for method in RUNS:
            rows = [row for row in event_rows if row["method"] == method]
            axis.plot(
                [row["meta_iteration"] for row in rows],
                [row[key] for row in rows],
                label=method,
                color=colors[method],
                linewidth=1.6,
                alpha=0.9,
            )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        if "accuracy" in key or "rate" in key:
            axis.yaxis.set_major_formatter(lambda value, position: f"{100 * value:.1f}%")
    axes[2, 0].set_xlabel("Meta-training iteration")
    axes[1, 1].set_xlabel("Meta-training iteration")
    axes[0, 0].legend(frameon=False)
    axes[2, 1].axis("off")
    figure.suptitle("Replay reconstruction fidelity across all 83 events", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(os.path.join(args.output_dir, "replay_fidelity_curves.png"), dpi=220)
    figure.savefig(os.path.join(args.output_dir, "replay_fidelity_curves.pdf"))
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", default="reconstruction_fidelity_results_20260803")
    parser.add_argument("--output-dir", default="reconstruction_fidelity_analysis_20260804")
    main(parser.parse_args())
