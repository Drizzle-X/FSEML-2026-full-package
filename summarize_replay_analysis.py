import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    root_dir = Path(args.analysis_dir)
    run_dir = root_dir / args.run_id
    analysis_dir = run_dir if run_dir.exists() else root_dir
    events_path = analysis_dir / f"replay_pseudo_usability_events_{args.run_id}.csv"
    tracked_path = analysis_dir / f"tracked_samples_over_time_{args.run_id}.csv"
    events_v2_path = analysis_dir / f"replay_pseudo_usability_events_{args.run_id}_schema_v2.csv"
    tracked_v2_path = analysis_dir / f"tracked_samples_over_time_{args.run_id}_schema_v2.csv"
    if events_v2_path.exists():
        events_path = events_v2_path
    if tracked_v2_path.exists():
        tracked_path = tracked_v2_path

    events = pd.read_csv(events_path)
    summary = events[
        [
            "pseudo_accuracy",
            "pseudo_confidence_mean",
            "pixel_mse_mean",
            "latent_cosine_mean",
            "feature_cosine_mean",
        ]
    ].agg(["mean", "std", "min", "max"])

    summary_path = analysis_dir / f"replay_pseudo_usability_summary_{args.run_id}.csv"
    summary.to_csv(summary_path)

    if tracked_path.exists():
        tracked = pd.read_csv(tracked_path)
        tracked_summary = tracked.groupby("event_index")[["pixel_mse", "latent_cosine", "feature_cosine"]].agg(
            ["mean", "std", "min", "max"]
        )
        tracked_summary.to_csv(analysis_dir / f"tracked_feature_cosine_summary_{args.run_id}.csv")

        plt.figure(figsize=(9, 5))
        for tracked_index, group in tracked.groupby("tracked_index"):
            plt.plot(
                group["event_index"],
                group["feature_cosine"],
                linewidth=1.2,
                alpha=0.65,
                label=f"sample {tracked_index}",
            )
        plt.plot(
            tracked_summary.index,
            tracked_summary[("feature_cosine", "mean")],
            linewidth=2.4,
            color="black",
            label="mean",
        )
        plt.xlabel("Replay event index")
        plt.ylabel("Feature cosine similarity")
        plt.title(f"Tracked Old-sample Feature Cosine Across Replay Events ({args.run_id})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(analysis_dir / f"tracked_feature_cosine_trend_{args.run_id}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(
            tracked_summary.index,
            tracked_summary[("latent_cosine", "mean")],
            linewidth=2.2,
            label="latent cosine",
        )
        plt.plot(
            tracked_summary.index,
            tracked_summary[("feature_cosine", "mean")],
            linewidth=2.2,
            label="feature cosine",
        )
        plt.xlabel("Replay event index")
        plt.ylabel("Cosine similarity")
        plt.title(f"Tracked Old-sample Similarity Across Replay Events ({args.run_id})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(analysis_dir / f"tracked_latent_feature_cosine_trend_{args.run_id}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(
            tracked_summary.index,
            tracked_summary[("pixel_mse", "mean")],
            linewidth=2.2,
            color="tab:red",
            label="pixel MSE",
        )
        plt.xlabel("Replay event index")
        plt.ylabel("Pixel MSE")
        plt.title(f"Tracked Old-sample Reconstruction Error Across Replay Events ({args.run_id})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(analysis_dir / f"tracked_pixel_mse_trend_{args.run_id}.png", dpi=300)
        plt.close()

    print(summary)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="../analysis_results/in_training_replay_usability")
    parser.add_argument("--run-id", default="20260428_01")
    main(parser.parse_args())
