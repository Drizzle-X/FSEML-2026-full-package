from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TREND_DIR = (
    SCRIPT_DIR
    / "20260505CSV"
    / "figures_paper_selected_sem_latent_plus02_feature_plus01_acccum"
)
LATENT_PROBE_DIR = SCRIPT_DIR / "20260509CSV" / "latent_probe_figures"
ALIGNED_OUTPUT_STEM = DEFAULT_TREND_DIR / "paper_selected_four_metrics_with_aligned_tsne"
SHIFTED_OUTPUT_STEM = DEFAULT_TREND_DIR / "paper_selected_four_metrics_with_shifted_tsne_alpha06_alpha04"
EIGHT_PANEL_OUTPUT_STEM = DEFAULT_TREND_DIR / "paper_selected_four_metrics_with_aligned_and_shifted_tsne"

TREND_PANEL_FILES = [
    ("Latent Cosine Mean", DEFAULT_TREND_DIR / "paper_scratch_latent_cosine_mean_trend.png"),
    ("Feature Cosine Mean", DEFAULT_TREND_DIR / "paper_scratch_feature_cosine_mean_trend.png"),
    ("Pixel MSE Mean", DEFAULT_TREND_DIR / "paper_ckpt_pixel_mse_mean_trend.png"),
    ("Latent Probe Cumulative Accuracy", LATENT_PROBE_DIR / "paper_latent_probe_top1_cumulative_accuracy.png"),
]

ALIGNED_PANEL_FILES = TREND_PANEL_FILES + [
    ("Aligned Latent t-SNE", SCRIPT_DIR / "final_latent_tsne_posthoc_aligned.png"),
    ("Aligned Feature t-SNE", SCRIPT_DIR / "final_feature_tsne_posthoc_aligned.png"),
]

SHIFTED_PANEL_FILES = TREND_PANEL_FILES + [
    ("Latent t-SNE", SCRIPT_DIR / "final_latent_tsne_outliers_relocated.png"),
    ("Feature t-SNE", SCRIPT_DIR / "final_feature_tsne_outliers_relocated.png"),
]

EIGHT_PANEL_FILES = TREND_PANEL_FILES + [
    ("Aligned Latent t-SNE", SCRIPT_DIR / "final_latent_tsne_posthoc_aligned.png"),
    ("Aligned Feature t-SNE", SCRIPT_DIR / "final_feature_tsne_posthoc_aligned.png"),
    ("Latent t-SNE", SCRIPT_DIR / "final_latent_tsne_outliers_relocated.png"),
    ("Feature t-SNE", SCRIPT_DIR / "final_feature_tsne_outliers_relocated.png"),
]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def require_files(paths):
    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing input figures:\n{joined}")


def trim_white_margin(image, pad=18, threshold=0.985):
    rgb = image[..., :3]
    if image.shape[-1] == 4:
        alpha = image[..., 3] > 0.01
    else:
        alpha = np.ones(rgb.shape[:2], dtype=bool)
    non_white = np.any(rgb < threshold, axis=2) & alpha
    if not np.any(non_white):
        return image
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]
    row_start = max(rows[0] - pad, 0)
    row_end = min(rows[-1] + pad + 1, image.shape[0])
    col_start = max(cols[0] - pad, 0)
    col_end = min(cols[-1] + pad + 1, image.shape[1])
    return image[row_start:row_end, col_start:col_end]


def compose(panel_files, output_stem):
    apply_style()
    require_files(panel_files)

    fig, axes = plt.subplots(3, 2, figsize=(8.35, 8.85), facecolor="white")
    for index, (ax, (title, path)) in enumerate(zip(axes.flat, panel_files)):
        image = trim_white_margin(mpimg.imread(path))
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.055,
            f"({chr(ord('a') + index)}) {title}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            fontfamily="Times New Roman",
            color="black",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.025, right=0.995, top=0.985, bottom=0.055, wspace=-0.035, hspace=0.18)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")


def compose_eight_panel():
    apply_style()
    require_files(EIGHT_PANEL_FILES)

    fig, axes = plt.subplots(4, 2, figsize=(7.16, 10.7), facecolor="white", constrained_layout=True)
    for index, (ax, (title, path)) in enumerate(zip(axes.flat, EIGHT_PANEL_FILES)):
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_axis_off()
        if index >= 4:
            ax.set_title(title, fontsize=11.5, fontfamily="Times New Roman", pad=4, color="black")
        ax.text(
            0.012,
            0.985,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.8,
            fontfamily="Times New Roman",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0},
        )

    EIGHT_PANEL_OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EIGHT_PANEL_OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(EIGHT_PANEL_OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {EIGHT_PANEL_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {EIGHT_PANEL_OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    compose(ALIGNED_PANEL_FILES, ALIGNED_OUTPUT_STEM)
    compose(SHIFTED_PANEL_FILES, SHIFTED_OUTPUT_STEM)
    compose_eight_panel()
