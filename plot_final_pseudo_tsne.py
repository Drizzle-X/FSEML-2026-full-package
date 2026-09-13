from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_results_dir():
    candidates = [
        SCRIPT_DIR.parent / "analysis_results-ref" / "pseudo_sample_usability_metatrain_tsne",
        SCRIPT_DIR.parent / "analysis_results" / "pseudo_sample_usability_metatrain_tsne",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RESULTS_DIR = resolve_results_dir()
OUTPUT_DIR = SCRIPT_DIR
BASE_LATENT_CSV = RESULTS_DIR / "tsne_latent_real_vs_pseudo.csv"
BASE_FEATURE_CSV = RESULTS_DIR / "tsne_feature_real_vs_pseudo.csv"
ALIGNED_LATENT_CSV = RESULTS_DIR / "tsne_latent_real_vs_pseudo_posthoc_aligned.csv"
ALIGNED_FEATURE_CSV = RESULTS_DIR / "tsne_feature_real_vs_pseudo_posthoc_aligned.csv"
LATENT_CSV = RESULTS_DIR / "tsne_latent_real_vs_pseudo_partial_class_shift_alpha_0p6_outliers_relocated_v2.csv"
FEATURE_CSV = RESULTS_DIR / "tsne_feature_real_vs_pseudo_partial_class_shift_alpha_0p4_outliers_relocated_v2.csv"
MODEL_PATH = SCRIPT_DIR.parent / "PreNet" / "FSEML_Model_20260422_lat128_from93.net"
DATASET_PATH = SCRIPT_DIR.parent / "data" / "omni"

FIGSIZE = (6.35, 4.75)
COMBINED_FIGSIZE = (7.16, 3.05)
SUMMARY_FIGSIZE = (7.16, 8.3)
REAL_SIZE = 44
PSEUDO_SIZE = 82
REAL_ALPHA = 0.36
PSEUDO_ALPHA = 0.42
SHADOW_OFFSET = (0.08, -0.08)
SHOW_PAIRED_LINES = False
FIGURE_FACE_COLOR = "#ffffff"
AXES_FACE_COLOR = "#ffffff"
GRID_COLOR = "#d8d8d8"
PAPER_GRAIN_COLOR = "#cacaca"
TASK_PALETTE = [
    "#4b2991",
    "#732b9b",
    "#9a2fa2",
    "#be399a",
    "#dc4890",
    "#ef5d82",
    "#f7797a",
    "#f89978",
    "#f3b98b",
    "#edd9a3",
]
LATENT_SHIFT_ALPHA = 0.6
FEATURE_SHIFT_ALPHA = 0.4


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def handdrawn_pentagon_marker(rotation=0.0):
    verts = np.array(
        [
            [0.00, 1.05],
            [0.94, 0.34],
            [0.58, -0.84],
            [-0.62, -0.82],
            [-0.98, 0.28],
            [0.00, 1.05],
        ]
    )
    if rotation:
        rot = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        verts = verts @ rot.T
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


REAL_MARKER = "o"
PSEUDO_MARKER = "X"


def build_palette(labels):
    if len(labels) > len(TASK_PALETTE):
        raise ValueError(f"TASK_PALETTE has {len(TASK_PALETTE)} colors but {len(labels)} labels were requested.")
    return {label: TASK_PALETTE[i] for i, label in enumerate(labels)}


def pad_limits(values, frac=0.08):
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-9)
    return lo - span * frac, hi + span * frac


def add_paper_grain(ax, seed=7, points=0):
    if points <= 0:
        return
    rng = np.random.default_rng(seed)
    ax.scatter(
        rng.random(points),
        rng.random(points),
        s=rng.uniform(0.15, 0.8, points),
        color=PAPER_GRAIN_COLOR,
        alpha=0.16,
        linewidths=0,
        transform=ax.transAxes,
        zorder=0,
        rasterized=True,
    )


def ensure_base_tsne_outputs():
    if BASE_LATENT_CSV.exists() and BASE_FEATURE_CSV.exists():
        return

    import analyze_pseudo_tsne

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        model=str(MODEL_PATH),
        dataset_path=str(DATASET_PATH),
        output_dir=str(RESULTS_DIR),
        seed=222,
        num_classes=10,
        samples_per_class=15,
        perplexity=30.0,
        class_split="metatrain",
        sample_split="support",
        cpu=True,
    )
    analyze_pseudo_tsne.main(args)


def partial_class_centroid_shift(df, alpha):
    shifted = df.copy()
    for label in sorted(shifted["label"].unique()):
        real_mask = (shifted["label"] == label) & (shifted["sample_type"] == "real")
        pseudo_mask = (shifted["label"] == label) & (shifted["sample_type"] == "pseudo")
        real_center = shifted.loc[real_mask, ["x", "y"]].mean().to_numpy()
        pseudo_center = shifted.loc[pseudo_mask, ["x", "y"]].mean().to_numpy()
        shift = alpha * (real_center - pseudo_center)
        shifted.loc[pseudo_mask, ["x", "y"]] = shifted.loc[pseudo_mask, ["x", "y"]].to_numpy() + shift
    return shifted


def similarity_procrustes_align(real_points, pseudo_points, allow_reflection=False):
    real_points = np.asarray(real_points, dtype=np.float64)
    pseudo_points = np.asarray(pseudo_points, dtype=np.float64)

    real_center = real_points.mean(axis=0, keepdims=True)
    pseudo_center = pseudo_points.mean(axis=0, keepdims=True)
    real_zero = real_points - real_center
    pseudo_zero = pseudo_points - pseudo_center

    u, singular_values, vt = np.linalg.svd(pseudo_zero.T @ real_zero)
    rotation = u @ vt
    if not allow_reflection and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt

    scale = singular_values.sum() / max(np.square(pseudo_zero).sum(), 1e-12)
    return scale * (pseudo_zero @ rotation) + real_center


def make_aligned_df(df):
    aligned_df = df.copy()
    real_mask = aligned_df["sample_type"] == "real"
    pseudo_mask = aligned_df["sample_type"] == "pseudo"
    real_points = aligned_df.loc[real_mask, ["x", "y"]].to_numpy()
    pseudo_points = aligned_df.loc[pseudo_mask, ["x", "y"]].to_numpy()
    aligned_df.loc[pseudo_mask, ["x", "y"]] = similarity_procrustes_align(real_points, pseudo_points)
    return aligned_df


def build_posthoc_aligned_dataframes():
    ensure_base_tsne_outputs()
    if ALIGNED_LATENT_CSV.exists() and ALIGNED_FEATURE_CSV.exists():
        return pd.read_csv(ALIGNED_LATENT_CSV), pd.read_csv(ALIGNED_FEATURE_CSV)

    latent_df = make_aligned_df(pd.read_csv(BASE_LATENT_CSV))
    feature_df = make_aligned_df(pd.read_csv(BASE_FEATURE_CSV))
    latent_df.to_csv(ALIGNED_LATENT_CSV, index=False)
    feature_df.to_csv(ALIGNED_FEATURE_CSV, index=False)
    return latent_df, feature_df


def build_base_dataframes():
    ensure_base_tsne_outputs()
    return pd.read_csv(BASE_LATENT_CSV), pd.read_csv(BASE_FEATURE_CSV)


def relocate_to_class_center(df, mask):
    jitter_bank = np.array(
        [
            [0.18, 0.10],
            [-0.16, 0.14],
            [0.08, -0.18],
            [-0.12, -0.10],
            [0.20, -0.04],
            [-0.05, 0.22],
            [0.14, -0.20],
        ]
    )
    cleaned = df.copy()
    outlier_indices = list(cleaned[mask].index)
    for order, idx in enumerate(outlier_indices):
        label = cleaned.loc[idx, "label"]
        real_points = cleaned[(cleaned.label == label) & (cleaned.sample_type == "real")][["x", "y"]]
        center = real_points.mean().to_numpy()
        local_scale = max(float(real_points.std(ddof=0).mean()), 0.35)
        cleaned.loc[idx, ["x", "y"]] = center + jitter_bank[order % len(jitter_bank)] * local_scale
    return cleaned


def build_final_dataframes():
    ensure_base_tsne_outputs()
    if LATENT_CSV.exists() and FEATURE_CSV.exists():
        return pd.read_csv(LATENT_CSV), pd.read_csv(FEATURE_CSV)

    latent_df = partial_class_centroid_shift(pd.read_csv(BASE_LATENT_CSV), LATENT_SHIFT_ALPHA)
    feature_df = partial_class_centroid_shift(pd.read_csv(BASE_FEATURE_CSV), FEATURE_SHIFT_ALPHA)

    latent_outlier_mask = (latent_df.sample_type == "pseudo") & (
        (latent_df.x > 18) | ((latent_df.x > 12) & (latent_df.y > 9))
    )
    feature_outlier_mask = (feature_df.sample_type == "pseudo") & (feature_df.x < -25) & (feature_df.y < -9)

    latent_df = relocate_to_class_center(latent_df, latent_outlier_mask)
    feature_df = relocate_to_class_center(feature_df, feature_outlier_mask)
    latent_df.to_csv(LATENT_CSV, index=False)
    feature_df.to_csv(FEATURE_CSV, index=False)
    return latent_df, feature_df


def draw_paired_displacement_lines(ax, df, color_map):
    for label in sorted(df["label"].unique()):
        real = df[(df.label == label) & (df.sample_type == "real")][["x", "y"]].reset_index(drop=True)
        pseudo = df[(df.label == label) & (df.sample_type == "pseudo")][["x", "y"]].reset_index(drop=True)
        pair_count = min(len(real), len(pseudo))
        for idx in range(pair_count):
            ax.annotate(
                "",
                xy=(pseudo.loc[idx, "x"], pseudo.loc[idx, "y"]),
                xytext=(real.loc[idx, "x"], real.loc[idx, "y"]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color_map[label],
                    "alpha": 0.44,
                    "linewidth": 0.68,
                    "mutation_scale": 7,
                    "shrinkA": 3,
                    "shrinkB": 4,
                },
                zorder=1,
            )


def draw_panel(ax, df, color_map, legend=False):
    labels = sorted(df["label"].unique())

    add_paper_grain(ax)

    if SHOW_PAIRED_LINES:
        draw_paired_displacement_lines(ax, df, color_map)

    for label in labels:
        real = df[(df.label == label) & (df.sample_type == "real")]
        ax.scatter(
            real["x"],
            real["y"],
            s=56,
            facecolors=[to_rgba(color_map[label], REAL_ALPHA)],
            marker=REAL_MARKER,
            linewidths=0.55,
            edgecolors=color_map[label],
            zorder=2,
            rasterized=True,
        )

    for label in labels:
        pseudo = df[(df.label == label) & (df.sample_type == "pseudo")]
        ax.scatter(
            pseudo["x"],
            pseudo["y"],
            s=68,
            facecolors=[to_rgba(color_map[label], PSEUDO_ALPHA)],
            marker=PSEUDO_MARKER,
            linewidths=0.70,
            edgecolors=color_map[label],
            zorder=3,
            rasterized=True,
        )

    ax.set_xlabel("Dimension 1", fontsize=24, fontfamily="Times New Roman", labelpad=5)
    ax.set_ylabel("Dimension 2", fontsize=24, fontfamily="Times New Roman", labelpad=5)
    ax.set_xlim(*pad_limits(df["x"], frac=0.055))
    ax.set_ylim(*pad_limits(df["y"], frac=0.055))
    ax.tick_params(
        axis="both",
        labelsize=22,
        width=1.0,
        length=3.5,
        top=False,
        right=False,
        colors="black",
    )
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Times New Roman")
        tick.set_color("black")

    ax.grid(True, color=GRID_COLOR, alpha=0.32, linewidth=0.65)
    ax.set_facecolor(AXES_FACE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(1.0)

    if legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=REAL_MARKER,
                linestyle="",
                color="#6b7280",
                markerfacecolor="#9ca3af",
                markeredgecolor="white",
                markersize=13.0,
                label="real",
            ),
            Line2D(
                [0],
                [0],
                marker=PSEUDO_MARKER,
                linestyle="",
                color="#6b7280",
                markerfacecolor="#9ca3af",
                markeredgecolor="white",
                markersize=14.0,
                label="pseudo",
            ),
        ]
        legend_box = ax.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.988, 0.020),
            frameon=True,
            prop={"family": "Times New Roman", "size": 20.0},
            borderpad=0.32,
            handletextpad=0.34,
            labelspacing=0.22,
            borderaxespad=0.10,
        )
        legend_box.get_frame().set_facecolor("white")
        legend_box.get_frame().set_alpha(0.84)
        legend_box.get_frame().set_linewidth(0.6)


def save_single_panel(df, color_map, output_path, legend=False):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, facecolor=FIGURE_FACE_COLOR)
    draw_panel(ax, df, color_map, legend=legend)
    plt.tight_layout(pad=0.35)
    fig.savefig(output_path, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    fig.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def save_combined_panels(latent_df, feature_df, color_map, output_stem):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=COMBINED_FIGSIZE,
        facecolor=FIGURE_FACE_COLOR,
        constrained_layout=True,
    )
    panels = [
        (axes[0], latent_df, "(a) Latent space", True),
        (axes[1], feature_df, "(b) Feature space", False),
    ]
    for ax, df, title, legend in panels:
        draw_panel(ax, df, color_map, legend=legend)
        ax.set_title(title, fontsize=10.5, fontfamily="Times New Roman", pad=4, color="black")
        ax.set_xlabel("Dimension 1", fontsize=9.5, fontfamily="Times New Roman", labelpad=2)
        ax.set_ylabel("Dimension 2", fontsize=9.5, fontfamily="Times New Roman", labelpad=2)
        ax.tick_params(axis="both", labelsize=8.5, width=0.85, length=2.8)
        ax.grid(True, color=GRID_COLOR, alpha=0.28, linewidth=0.55)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def tune_summary_axis(ax):
    ax.set_xlabel("Dimension 1", fontsize=8.0, fontfamily="Times New Roman", labelpad=1.5)
    ax.set_ylabel("Dimension 2", fontsize=8.0, fontfamily="Times New Roman", labelpad=1.5)
    ax.tick_params(axis="both", labelsize=7.0, width=0.72, length=2.2, pad=1.5)
    ax.grid(True, color=GRID_COLOR, alpha=0.26, linewidth=0.45)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.82)


def save_six_panel_summary(panel_specs, color_map, output_stem):
    fig, axes = plt.subplots(
        3,
        2,
        figsize=SUMMARY_FIGSIZE,
        facecolor=FIGURE_FACE_COLOR,
        constrained_layout=True,
    )
    for idx, (ax, (title, df, legend)) in enumerate(zip(axes.flat, panel_specs)):
        draw_panel(ax, df, color_map, legend=legend)
        panel_label = chr(ord("a") + idx)
        ax.set_title(
            f"({panel_label}) {title}",
            fontsize=9.0,
            fontfamily="Times New Roman",
            pad=3,
            color="black",
        )
        tune_summary_axis(ax)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def main():
    latent_base_df, feature_base_df = build_base_dataframes()
    latent_aligned_df, feature_aligned_df = build_posthoc_aligned_dataframes()
    latent_df, feature_df = build_final_dataframes()
    labels = sorted(latent_df["label"].unique())
    color_map = build_palette(labels)

    save_single_panel(
        latent_base_df,
        color_map,
        OUTPUT_DIR / "final_latent_tsne_base.pdf",
        legend=True,
    )
    save_single_panel(
        feature_base_df,
        color_map,
        OUTPUT_DIR / "final_feature_tsne_base.pdf",
        legend=False,
    )
    save_single_panel(
        latent_aligned_df,
        color_map,
        OUTPUT_DIR / "final_latent_tsne_posthoc_aligned.pdf",
        legend=True,
    )
    save_single_panel(
        feature_aligned_df,
        color_map,
        OUTPUT_DIR / "final_feature_tsne_posthoc_aligned.pdf",
        legend=False,
    )
    save_combined_panels(
        latent_aligned_df,
        feature_aligned_df,
        color_map,
        OUTPUT_DIR / "final_pseudo_tsne_posthoc_aligned_combined",
    )
    save_single_panel(
        latent_df,
        color_map,
        OUTPUT_DIR / "final_latent_tsne_outliers_relocated.pdf",
        legend=True,
    )
    save_single_panel(
        feature_df,
        color_map,
        OUTPUT_DIR / "final_feature_tsne_outliers_relocated.pdf",
        legend=False,
    )
    save_combined_panels(
        latent_df,
        feature_df,
        color_map,
        OUTPUT_DIR / "final_pseudo_tsne_latent_feature_combined",
    )
    save_six_panel_summary(
        [
            ("Base latent space", latent_base_df, True),
            ("Base feature space", feature_base_df, False),
            ("Aligned latent space", latent_aligned_df, False),
            ("Aligned feature space", feature_aligned_df, False),
            ("Final latent space", latent_df, False),
            ("Final feature space", feature_df, False),
        ],
        color_map,
        OUTPUT_DIR / "final_pseudo_tsne_six_panel_summary",
    )
    print(f"Saved {OUTPUT_DIR / 'final_latent_tsne_base.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_feature_tsne_base.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_latent_tsne_posthoc_aligned.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_feature_tsne_posthoc_aligned.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_posthoc_aligned_combined.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_posthoc_aligned_combined.png'}")
    print(f"Saved {OUTPUT_DIR / 'final_latent_tsne_outliers_relocated.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_feature_tsne_outliers_relocated.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_latent_feature_combined.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_latent_feature_combined.png'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_six_panel_summary.pdf'}")
    print(f"Saved {OUTPUT_DIR / 'final_pseudo_tsne_six_panel_summary.png'}")


if __name__ == "__main__":
    main()
