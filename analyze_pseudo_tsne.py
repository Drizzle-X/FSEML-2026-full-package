import argparse
import csv
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.manifold import TSNE

import datasets.datasetfactory as df


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def collect_balanced_samples(dataset, class_ids, samples_per_class):
    buckets = {int(class_id): [] for class_id in class_ids}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in buckets:
            continue
        if len(buckets[label]) < samples_per_class:
            buckets[label].append(idx)
        if all(len(items) >= samples_per_class for items in buckets.values()):
            break

    selected = []
    for class_id in class_ids:
        selected.extend(buckets[int(class_id)])
    return selected


def extract_representations(model, images):
    with torch.no_grad():
        latent = model.srn.fssae.encode(images)
        pseudo_images = model.srn.fssae.decode(latent)
        pseudo_latent = model.srn.fssae.encode(pseudo_images)

        feature = model.srn.fssae.encoder(images).flatten(start_dim=1)
        pseudo_feature = model.srn.fssae.encoder(pseudo_images).flatten(start_dim=1)
        _, real_logits = model(images)
        _, pseudo_logits = model(pseudo_images)
        real_pred = real_logits.argmax(dim=1)
        pseudo_pred = pseudo_logits.argmax(dim=1)

    return {
        "latent": latent.detach().cpu().numpy(),
        "pseudo_latent": pseudo_latent.detach().cpu().numpy(),
        "feature": feature.detach().cpu().numpy(),
        "pseudo_feature": pseudo_feature.detach().cpu().numpy(),
        "real_logits": real_logits.detach().cpu().numpy(),
        "pseudo_logits": pseudo_logits.detach().cpu().numpy(),
        "real_pred": real_pred.detach().cpu().numpy(),
        "pseudo_pred": pseudo_pred.detach().cpu().numpy(),
    }


def fit_embedding(matrix, method, seed, perplexity):
    if method == "tsne":
        max_perplexity = max(2, min(perplexity, (len(matrix) - 1) // 3))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=max_perplexity,
            random_state=seed,
        ).fit_transform(matrix)

    if method == "umap":
        try:
            import umap
        except ImportError:
            return None
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(matrix)

    raise ValueError(f"Unsupported embedding method: {method}")


def plot_embedding(points, labels, sample_types, class_ids, title, output_path):
    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10", len(class_ids))
    class_to_color = {int(class_id): cmap(i) for i, class_id in enumerate(class_ids)}
    markers = {"real": "o", "pseudo": "^"}

    for sample_type in ["real", "pseudo"]:
        for class_id in class_ids:
            mask = (sample_types == sample_type) & (labels == int(class_id))
            if not np.any(mask):
                continue
            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                s=36,
                marker=markers[sample_type],
                c=[class_to_color[int(class_id)]],
                alpha=0.78 if sample_type == "real" else 0.58,
                edgecolors="none",
                label=f"{class_id} {sample_type}",
            )

    plt.title(title)
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    plt.grid(alpha=0.18)

    handles = []
    labels_for_legend = []
    for class_id in class_ids:
        handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="", color=class_to_color[int(class_id)], markersize=7)
        )
        labels_for_legend.append(f"class {class_id}")
    handles.extend(
        [
            plt.Line2D([0], [0], marker="o", linestyle="", color="black", markersize=7),
            plt.Line2D([0], [0], marker="^", linestyle="", color="black", markersize=7),
        ]
    )
    labels_for_legend.extend(["real", "pseudo"])
    plt.legend(handles, labels_for_legend, loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_csv(output_path, points, labels, sample_types, representation, method):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "method", "label", "sample_type", "x", "y"])
        for point, label, sample_type in zip(points, labels, sample_types):
            writer.writerow([representation, method, int(label), sample_type, float(point[0]), float(point[1])])


def cosine_per_sample(real, pseudo):
    real_tensor = torch.as_tensor(real)
    pseudo_tensor = torch.as_tensor(pseudo)
    return F.cosine_similarity(real_tensor, pseudo_tensor, dim=1).numpy()


def restricted_predictions(logits, class_ids):
    keep = np.asarray(class_ids, dtype=np.int64)
    local_pred = logits[:, keep].argmax(axis=1)
    return keep[local_pred]


def write_metrics(output_dir, labels, class_ids, reps):
    latent_cosine = cosine_per_sample(reps["latent"], reps["pseudo_latent"])
    feature_cosine = cosine_per_sample(reps["feature"], reps["pseudo_feature"])
    real_pred = reps["real_pred"].astype(int)
    pseudo_pred = reps["pseudo_pred"].astype(int)
    real_restricted_pred = restricted_predictions(reps["real_logits"], class_ids)
    pseudo_restricted_pred = restricted_predictions(reps["pseudo_logits"], class_ids)
    real_correct = real_pred == labels
    pseudo_correct = pseudo_pred == labels
    real_restricted_correct = real_restricted_pred == labels
    pseudo_restricted_correct = pseudo_restricted_pred == labels

    per_sample_path = os.path.join(output_dir, "pseudo_usability_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "label",
                "real_pred",
                "real_correct",
                "real_restricted_pred",
                "real_restricted_correct",
                "pseudo_pred",
                "pseudo_correct",
                "pseudo_restricted_pred",
                "pseudo_restricted_correct",
                "latent_cosine",
                "feature_cosine",
            ]
        )
        for idx, metrics in enumerate(
            zip(
                labels,
                real_pred,
                real_correct,
                real_restricted_pred,
                real_restricted_correct,
                pseudo_pred,
                pseudo_correct,
                pseudo_restricted_pred,
                pseudo_restricted_correct,
                latent_cosine,
                feature_cosine,
            )
        ):
            (
                label,
                real_global,
                real_global_correct,
                real_seen,
                real_seen_correct,
                pseudo_global,
                pseudo_global_correct,
                pseudo_seen,
                pseudo_seen_correct,
                latent_cos,
                feature_cos,
            ) = metrics
            writer.writerow(
                [
                    idx,
                    int(label),
                    int(real_global),
                    int(real_global_correct),
                    int(real_seen),
                    int(real_seen_correct),
                    int(pseudo_global),
                    int(pseudo_global_correct),
                    int(pseudo_seen),
                    int(pseudo_seen_correct),
                    float(latent_cos),
                    float(feature_cos),
                ]
            )

    summary = {
        "num_samples": len(labels),
        "num_classes": len(set(labels.tolist())),
        "real_global_classification_accuracy": float(real_correct.mean()),
        "real_seen_classification_accuracy": float(real_restricted_correct.mean()),
        "pseudo_global_classification_accuracy": float(pseudo_correct.mean()),
        "pseudo_seen_classification_accuracy": float(pseudo_restricted_correct.mean()),
        "latent_cosine_mean": float(latent_cosine.mean()),
        "latent_cosine_std": float(latent_cosine.std(ddof=1)),
        "feature_cosine_mean": float(feature_cosine.mean()),
        "feature_cosine_std": float(feature_cosine.std(ddof=1)),
    }

    summary_path = os.path.join(output_dir, "pseudo_usability_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    return summary


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model, device)
    background = args.class_split == "metatrain"
    dataset = df.DatasetFactory.get_dataset(
        "omniglot",
        train=args.sample_split == "support",
        background=background,
        path=args.dataset_path,
    )

    all_class_ids = sorted({int(label) for label in dataset.targets})
    rng = np.random.default_rng(args.seed)
    class_ids = rng.choice(all_class_ids, size=args.num_classes, replace=False).tolist()
    selected_indices = collect_balanced_samples(dataset, class_ids, args.samples_per_class)

    images = []
    labels = []
    for idx in selected_indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(int(label))

    images = torch.stack(images).to(device)
    labels = np.asarray(labels)
    reps = extract_representations(model, images)
    summary = write_metrics(args.output_dir, labels, class_ids, reps)

    for representation in ["latent", "feature"]:
        real = reps[representation]
        pseudo = reps[f"pseudo_{representation}"]
        matrix = np.concatenate([real, pseudo], axis=0)
        combined_labels = np.concatenate([labels, labels], axis=0)
        sample_types = np.asarray(["real"] * len(labels) + ["pseudo"] * len(labels))

        for method in ["tsne", "umap"]:
            points = fit_embedding(matrix, method, args.seed, args.perplexity)
            if points is None:
                continue

            stem = f"{method}_{representation}_real_vs_pseudo"
            plot_embedding(
                points,
                combined_labels,
                sample_types,
                class_ids,
                f"{method.upper()} of real and pseudo samples ({representation} space)",
                os.path.join(args.output_dir, f"{stem}.png"),
            )
            write_csv(
                os.path.join(args.output_dir, f"{stem}.csv"),
                points,
                combined_labels,
                sample_types,
                representation,
                method,
            )

    with open(os.path.join(args.output_dir, "selected_classes.txt"), "w") as f:
        f.write(" ".join(str(int(class_id)) for class_id in class_ids))
        f.write("\n")

    print(f"Saved pseudo-sample embedding visualizations to {args.output_dir}")
    print(f"Selected classes: {' '.join(str(int(class_id)) for class_id in class_ids)}")
    print("Pseudo-sample usability metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../PreNet/FSEML_Model_20260422_lat128_from93.net")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default="../analysis_results/pseudo_sample_usability_metatrain")
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=15)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--class-split", choices=["metatrain", "metatest"], default="metatrain")
    parser.add_argument("--sample-split", choices=["support", "query"], default="support")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
