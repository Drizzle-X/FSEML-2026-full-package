import argparse
import csv
import glob
import json
import os

import numpy as np


METRICS = (
    "pixel_mse",
    "ssim",
    "frozen_feature_cosine",
)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_sample_rows(rows):
    summary = {"sample_count": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    real_correct = np.asarray([int(row["real_correct"]) for row in rows], dtype=np.int64)
    pseudo_correct = np.asarray([int(row["pseudo_correct"]) for row in rows], dtype=np.int64)
    preserved = np.asarray([int(row["label_preserved"]) for row in rows], dtype=np.int64)
    summary["independent_real_accuracy"] = float(real_correct.mean())
    summary["independent_pseudo_accuracy"] = float(pseudo_correct.mean())
    summary["label_preservation_rate"] = (
        float(preserved.sum() / real_correct.sum()) if real_correct.sum() else float("nan")
    )
    return summary


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    event_files = sorted(glob.glob(os.path.join(args.analysis_root, "*", "reconstruction_fidelity_event_samples_*.csv")))
    final_files = sorted(glob.glob(os.path.join(args.analysis_root, "*", "reconstruction_fidelity_final_buffer_summary_*.json")))
    event_summaries = []
    for path in event_files:
        rows = read_rows(path)
        by_event = {}
        for row in rows:
            by_event.setdefault((row["run_id"], row["event_index"], row["meta_iteration"]), []).append(row)
        for (run_id, event_index, meta_iteration), event_rows in sorted(by_event.items(), key=lambda item: int(item[0][1])):
            event_summaries.append({
                "run_id": run_id,
                "event_index": int(event_index),
                "meta_iteration": int(meta_iteration),
                **summarize_sample_rows(event_rows),
            })
    final_summaries = []
    for path in final_files:
        with open(path) as handle:
            final_summaries.append(json.load(handle))
    write_csv(os.path.join(args.output_dir, "replay_event_fidelity_summary.csv"), event_summaries)
    write_csv(os.path.join(args.output_dir, "final_buffer_fidelity_summary.csv"), final_summaries)
    print(f"replay-event rows: {len(event_summaries)}")
    print(f"final-buffer rows: {len(final_summaries)}")
    print(f"summary directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build paper-ready reconstruction-fidelity summary tables.")
    parser.add_argument("--analysis-root", default="reconstruction_fidelity_results")
    parser.add_argument("--output-dir", default="reconstruction_fidelity_results/summaries")
    main(parser.parse_args())
