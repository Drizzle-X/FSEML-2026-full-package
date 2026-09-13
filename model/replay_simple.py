import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from analysis.reconstruction_fidelity import (
    append_event_samples_csv,
    compute_fidelity_metrics,
    compute_fidelity_metrics_batched,
    load_reference_classifier,
    model_to_unit_interval,
    write_summary_csv,
    write_summary_json,
)


class SimpleReplayBuffer:
    """
    Lightweight replay backend that preserves the current project behavior.

    It stores latent codes in a ring buffer, periodically decodes a uniformly
    sampled subset, and concatenates those pseudo-samples to the query batch.
    """

    mode_name = "simple"

    def __init__(self, args):
        self.enabled = getattr(args, "replay", False)
        self.buffer_size = getattr(args, "replay_buffer_size", 1000)
        self.replay_gap = max(1, getattr(args, "replay_gap", 960))
        self.replay_rate = max(0.0, min(1.0, getattr(args, "replay_rate", 0.05)))
        self.replay_content = getattr(args, "replay_content", "decoder").lower()
        if self.replay_content not in {"decoder", "raw"}:
            raise ValueError("replay_content must be 'decoder' or 'raw'.")
        self.replay_seed = int(getattr(args, "seed", 9)) + 104729
        self.cpu_generator = torch.Generator(device="cpu")
        self.cpu_generator.manual_seed(self.replay_seed)
        self.visualize = getattr(args, "visualize_replay", False)
        self.viz_dir = getattr(args, "replay_viz_dir", "replay_visualizations")
        self.viz_max = max(0, getattr(args, "replay_viz_max", 10))
        self.analysis_dir = getattr(args, "replay_analysis_dir", None)
        self.analysis_max_events = max(0, getattr(args, "replay_analysis_max_events", 0))
        self.analysis_plot_events = set(getattr(args, "replay_analysis_plot_events", []))
        self.analysis_cumulative_plot_events = set(getattr(args, "replay_analysis_cumulative_plot_events", []))
        self.analysis_cumulative_plot_max_points = max(
            0,
            getattr(args, "replay_analysis_cumulative_plot_max_points", 0),
        )
        self.analysis_track_events = set(getattr(args, "replay_analysis_track_events", []))
        self.analysis_track_all_events = getattr(args, "replay_analysis_track_all_events", False)
        self.analysis_track_count = max(0, getattr(args, "replay_analysis_track_count", 0))
        self.analysis_expanded_eval_samples = max(0, getattr(args, "replay_analysis_expanded_eval_samples", 0))
        self.analysis_save_probe_data = getattr(args, "replay_analysis_save_probe_data", False)
        self.fidelity_enabled = getattr(args, "reconstruction_fidelity", False)
        self.fidelity_dataset = getattr(args, "dataset", "omniglot")
        self.reference_classifier_path = getattr(args, "reference_classifier", None)
        self.final_fidelity_export = getattr(args, "final_fidelity_export", False)
        self.fidelity_eval_batch_size = max(1, getattr(args, "fidelity_eval_batch_size", 128))
        self.analysis_run_id = getattr(args, "replay_analysis_run_id", "20260428_01")
        if self.analysis_dir and self.analysis_run_id:
            self.analysis_dir = os.path.join(self.analysis_dir, self.analysis_run_id)

        self.viz_count = 0
        self.analysis_count = 0
        self.analysis_header_written = False
        self.tracked_header_written = False
        self.last_replay_batch = None
        self.latents = []
        self.labels = []
        self.features = []
        self.images = []
        self.tracked_items = []
        self.cumulative_embedding_items = []
        self.position = 0
        self.reference_classifier = None
        self.replay_event_count = 0
        self.total_replayed_samples = 0
        self.last_replay_count = 0

        if self.fidelity_enabled and not self.reference_classifier_path:
            raise ValueError("--reference-classifier is required with --reconstruction-fidelity")
        if self.fidelity_enabled and not os.path.isfile(self.reference_classifier_path):
            raise FileNotFoundError(
                f"Reference classifier checkpoint was not found: {self.reference_classifier_path}"
            )

    def _analysis_filename(self, stem, extension):
        if self.analysis_run_id:
            return f"{stem}_{self.analysis_run_id}.{extension}"
        return f"{stem}.{extension}"

    def _get_reference_classifier(self, device):
        if not self.fidelity_enabled:
            return None
        if self.reference_classifier is None:
            self.reference_classifier = load_reference_classifier(self.reference_classifier_path, device)
        return self.reference_classifier

    def should_replay(self, meta_iteration):
        if not self.enabled:
            return False
        if not self.latents:
            return False
        return (meta_iteration + 1) % self.replay_gap == 0

    def status(self):
        return {
            "mode": self.mode_name,
            "content": self.replay_content,
            "enabled": bool(self.enabled),
            "buffer_length": len(self.latents),
            "buffer_capacity": self.buffer_size,
            "replay_events": self.replay_event_count,
            "total_replayed_samples": self.total_replayed_samples,
            "last_replay_count": self.last_replay_count,
            "replay_gap": self.replay_gap,
            "replay_rate": self.replay_rate,
        }

    def store_batch(self, model, x, y):
        if not self.enabled or self.buffer_size <= 0:
            return

        with torch.no_grad():
            srn_outputs = model.forward_srn(x)
            latent = srn_outputs["latent"].detach().cpu()
            feature = model.srn.fssae.encoder(x).flatten(start_dim=1).detach().cpu()
            images = x.detach().cpu()
            labels = y.detach().view(-1).cpu()

        for latent_item, feature_item, image_item, label_item in zip(latent, feature, images, labels):
            latent_item = latent_item.clone()
            feature_item = feature_item.clone()
            image_item = image_item.clone()
            label_item = label_item.clone()
            if len(self.latents) < self.buffer_size:
                self.latents.append(latent_item)
                self.features.append(feature_item)
                self.images.append(image_item)
                self.labels.append(label_item)
            else:
                self.latents[self.position] = latent_item
                self.features[self.position] = feature_item
                self.images[self.position] = image_item
                self.labels[self.position] = label_item
                self.position = (self.position + 1) % self.buffer_size

    def sample(self, model, query_batch_size, device, meta_iteration):
        self.last_replay_batch = None
        if not self.should_replay(meta_iteration):
            return None, None

        replay_count = int(round(query_batch_size * self.replay_rate))
        replay_count = max(1, replay_count)
        replay_count = min(replay_count, len(self.latents))
        if replay_count <= 0:
            return None, None

        if "omni" in self.fidelity_dataset.lower():

            indices = torch.randperm(len(self.latents))[:replay_count].tolist()
        else:

            indices = torch.randperm(
                len(self.latents),
                generator=self.cpu_generator,
            )[:replay_count].tolist()
        latent_batch = torch.stack([self.latents[idx] for idx in indices]).to(device)
        feature_batch = torch.stack([self.features[idx] for idx in indices]).to(device)
        image_batch = torch.stack([self.images[idx] for idx in indices]).to(device)
        label_batch = torch.stack([self.labels[idx] for idx in indices]).to(device)

        if self.replay_content == "raw":
            replay_samples = image_batch.detach()
        else:
            with torch.no_grad():
                replay_samples = model.srn.fssae.decode(latent_batch).detach()

        self.last_replay_batch = {
            "indices": indices,
            "stored_latents": latent_batch.detach(),
            "stored_features": feature_batch.detach(),
            "stored_images": image_batch.detach(),
            "labels": label_batch.detach(),
            "pseudo_samples": replay_samples.detach(),
            "meta_iteration": meta_iteration,
        }
        self.last_replay_count = replay_count
        self.replay_event_count += 1
        self.total_replayed_samples += replay_count
        return replay_samples, label_batch

    def maybe_save_visualization(self, pseudo_samples, pseudo_labels, meta_iteration):
        if not self.visualize:
            return
        if self.viz_count >= self.viz_max:
            return
        if pseudo_samples is None or len(pseudo_samples) == 0:
            return

        os.makedirs(self.viz_dir, exist_ok=True)
        nrow = max(1, int(math.sqrt(len(pseudo_samples))))
        file_name = (
            f"replay_step_{meta_iteration + 1:06d}"
            f"_n{len(pseudo_samples):02d}"
            f"_labels_{'-'.join(map(str, pseudo_labels.detach().cpu().tolist()))}.png"
        )
        output_path = os.path.join(self.viz_dir, file_name)
        save_image(
            pseudo_samples.detach().cpu(),
            output_path,
            nrow=nrow,
            normalize=True,
            value_range=(-1.0, 1.0),
        )
        self.viz_count += 1

    def augment_query(self, model, x, y, meta_iteration):
        pseudo_x, pseudo_y = self.sample(model, len(y), x.device, meta_iteration)
        if pseudo_x is None:
            return x, y, 0

        self.maybe_save_visualization(pseudo_x, pseudo_y, meta_iteration)
        mixed_x = torch.cat([x, pseudo_x], dim=0)
        mixed_y = torch.cat([y, pseudo_y], dim=0)
        return mixed_x, mixed_y, len(pseudo_y)

    def _write_csv_row(self, path, row, header_attr):
        expected_header = ",".join(row.keys())
        if os.path.exists(path):
            with open(path, "r") as f:
                existing_header = f.readline().strip()
            if existing_header and existing_header != expected_header:
                root, ext = os.path.splitext(path)
                path = f"{root}_schema_v2{ext}"

        write_header = not getattr(self, header_attr) and not os.path.exists(path)
        with open(path, "a") as f:
            if write_header:
                f.write(expected_header + "\n")
                setattr(self, header_attr, True)
            f.write(",".join(str(value) for value in row.values()) + "\n")

    def _to_grayscale_for_display(self, image):
        if image.dim() == 3:
            return image.mean(dim=0, keepdim=True)
        if image.dim() == 4 and image.size(1) == 3:
            return image.mean(dim=1, keepdim=True)
        return image

    def _maybe_init_tracked_items(self):
        if self.tracked_items or self.analysis_track_count <= 0:
            return
        count = min(self.analysis_track_count, len(self.latents))
        for idx in range(count):
            self.tracked_items.append(
                {
                    "buffer_index": idx,
                    "label": self.labels[idx].clone(),
                    "image": self.images[idx].clone(),
                    "latent": self.latents[idx].clone(),
                    "feature": self.features[idx].clone(),
                }
            )

    def _save_embedding_plot(self, model, event_index):
        if event_index not in self.analysis_plot_events:
            return
        batch = self.last_replay_batch
        if batch is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE

        os.makedirs(os.path.join(self.analysis_dir, "embeddings"), exist_ok=True)

        device = batch["pseudo_samples"].device
        real_images = batch["stored_images"].to(device)
        pseudo_images = batch["pseudo_samples"].to(device)
        labels = batch["labels"].detach().cpu().numpy()

        with torch.no_grad():
            real_latent = batch["stored_latents"].detach().cpu().numpy()
            pseudo_latent = model.srn.fssae.encode(pseudo_images).detach().cpu().numpy()
            real_feature = batch["stored_features"].detach().cpu().numpy()
            pseudo_feature = model.srn.fssae.encoder(pseudo_images).flatten(start_dim=1).detach().cpu().numpy()

        del real_images

        for representation, real, pseudo in [
            ("latent", real_latent, pseudo_latent),
            ("feature", real_feature, pseudo_feature),
        ]:
            matrix = np.concatenate([real, pseudo], axis=0)
            all_labels = np.concatenate([labels, labels], axis=0)
            sample_types = np.asarray(["stored"] * len(labels) + ["pseudo"] * len(labels))

            methods = [("tsne", None)]
            try:
                import umap

                methods.append(("umap", umap.UMAP(n_components=2, random_state=event_index, n_neighbors=5, min_dist=0.1)))
            except Exception:
                pass

            for method_name, reducer in methods:
                if method_name == "tsne":
                    perplexity = max(2, min(5, (len(matrix) - 1) // 3))
                    if len(matrix) <= 3:
                        continue
                    points = TSNE(
                        n_components=2,
                        init="pca",
                        learning_rate="auto",
                        perplexity=perplexity,
                        random_state=event_index,
                    ).fit_transform(matrix)
                else:
                    points = reducer.fit_transform(matrix)

                plt.figure(figsize=(7, 5))
                unique_labels = sorted(set(int(label) for label in all_labels))
                cmap = plt.get_cmap("tab10", max(1, len(unique_labels)))
                color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
                for sample_type, marker, alpha in [("stored", "o", 0.75), ("pseudo", "^", 0.6)]:
                    for label in unique_labels:
                        mask = (sample_types == sample_type) & (all_labels == label)
                        if not np.any(mask):
                            continue
                        plt.scatter(
                            points[mask, 0],
                            points[mask, 1],
                            marker=marker,
                            c=[color_map[label]],
                            alpha=alpha,
                            s=42,
                            edgecolors="none",
                        )
                plt.title(f"{method_name.upper()} replay event {event_index} ({representation})")
                plt.xlabel("dimension 1")
                plt.ylabel("dimension 2")
                plt.grid(alpha=0.18)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.analysis_dir,
                        "embeddings",
                        self._analysis_filename(f"{method_name}_{representation}_event_{event_index:03d}", "png"),
                    ),
                    dpi=300,
                )
                plt.close()

    def _append_cumulative_embedding_items(
        self,
        labels,
        stored_latents,
        pseudo_latents,
        stored_features,
        pseudo_features,
        event_index,
    ):
        self.cumulative_embedding_items.append(
            {
                "event_index": event_index,
                "labels": labels.detach().cpu(),
                "stored_latents": stored_latents.detach().cpu(),
                "pseudo_latents": pseudo_latents.detach().cpu(),
                "stored_features": stored_features.detach().cpu(),
                "pseudo_features": pseudo_features.detach().cpu(),
            }
        )

    def _save_cumulative_embedding_plot(self, event_index):
        if event_index not in self.analysis_cumulative_plot_events:
            return
        if not self.cumulative_embedding_items:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE

        os.makedirs(os.path.join(self.analysis_dir, "cumulative_embeddings"), exist_ok=True)

        labels = torch.cat([item["labels"] for item in self.cumulative_embedding_items]).numpy()
        event_ids = np.concatenate(
            [
                np.full(len(item["labels"]), item["event_index"], dtype=np.int64)
                for item in self.cumulative_embedding_items
            ]
        )

        for representation, stored_key, pseudo_key in [
            ("latent", "stored_latents", "pseudo_latents"),
            ("feature", "stored_features", "pseudo_features"),
        ]:
            stored = torch.cat([item[stored_key] for item in self.cumulative_embedding_items]).numpy()
            pseudo = torch.cat([item[pseudo_key] for item in self.cumulative_embedding_items]).numpy()
            matrix = np.concatenate([stored, pseudo], axis=0)
            all_labels = np.concatenate([labels, labels], axis=0)
            all_event_ids = np.concatenate([event_ids, event_ids], axis=0)
            sample_types = np.asarray(["stored"] * len(labels) + ["pseudo"] * len(labels))

            if self.analysis_cumulative_plot_max_points and len(matrix) > self.analysis_cumulative_plot_max_points:
                rng = np.random.default_rng(event_index)
                selected = rng.choice(len(matrix), size=self.analysis_cumulative_plot_max_points, replace=False)
                matrix = matrix[selected]
                all_labels = all_labels[selected]
                all_event_ids = all_event_ids[selected]
                sample_types = sample_types[selected]

            methods = [("tsne", None)]
            try:
                import umap

                methods.append(
                    (
                        "umap",
                        umap.UMAP(
                            n_components=2,
                            random_state=event_index,
                            n_neighbors=min(15, max(2, len(matrix) - 1)),
                            min_dist=0.1,
                        ),
                    )
                )
            except Exception:
                pass

            for method_name, reducer in methods:
                if method_name == "tsne":
                    if len(matrix) <= 3:
                        continue
                    perplexity = max(2, min(30, (len(matrix) - 1) // 3))
                    points = TSNE(
                        n_components=2,
                        init="pca",
                        learning_rate="auto",
                        perplexity=perplexity,
                        random_state=event_index,
                    ).fit_transform(matrix)
                else:
                    points = reducer.fit_transform(matrix)

                plt.figure(figsize=(8, 6))
                unique_labels = sorted(set(int(label) for label in all_labels))
                cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
                color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
                for sample_type, marker, alpha in [("stored", "o", 0.55), ("pseudo", "^", 0.45)]:
                    for label in unique_labels:
                        mask = (sample_types == sample_type) & (all_labels == label)
                        if not np.any(mask):
                            continue
                        plt.scatter(
                            points[mask, 0],
                            points[mask, 1],
                            marker=marker,
                            c=[color_map[label]],
                            alpha=alpha,
                            s=30,
                            edgecolors="none",
                        )

                plt.title(
                    f"Cumulative {method_name.upper()} through replay event {event_index} "
                    f"({representation}, n={len(matrix)})"
                )
                plt.xlabel("dimension 1")
                plt.ylabel("dimension 2")
                plt.grid(alpha=0.18)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.analysis_dir,
                        "cumulative_embeddings",
                        self._analysis_filename(
                            f"cumulative_{method_name}_{representation}_through_event_{event_index:03d}",
                            "png",
                        ),
                    ),
                    dpi=300,
                )
                plt.close()

                csv_path = os.path.join(
                    self.analysis_dir,
                    "cumulative_embeddings",
                    self._analysis_filename(
                        f"cumulative_{method_name}_{representation}_through_event_{event_index:03d}",
                        "csv",
                    ),
                )
                with open(csv_path, "w") as f:
                    f.write("run_id,method,representation,event_index,label,sample_type,x,y\n")
                    for point, label, sample_type, source_event in zip(
                        points,
                        all_labels,
                        sample_types,
                        all_event_ids,
                    ):
                        f.write(
                            f"{self.analysis_run_id},{method_name},{representation},{int(source_event)},"
                            f"{int(label)},{sample_type},{float(point[0])},{float(point[1])}\n"
                        )

    def _record_tracked_items(self, model, fast_weights, event_index):
        del fast_weights
        if not self.analysis_track_all_events and event_index not in self.analysis_track_events:
            return
        self._maybe_init_tracked_items()
        if not self.tracked_items:
            return

        os.makedirs(os.path.join(self.analysis_dir, "tracked_samples"), exist_ok=True)
        path = os.path.join(self.analysis_dir, self._analysis_filename("tracked_samples_over_time", "csv"))

        device = next(model.parameters()).device
        with torch.no_grad():
            for tracked_index, item in enumerate(self.tracked_items):
                label = item["label"].to(device).long().view(1)
                original_image = item["image"].to(device)
                stored_latent = item["latent"].to(device).view(1, -1)
                stored_feature = item["feature"].to(device).view(1, -1)

                pseudo_image = model.srn.fssae.decode(stored_latent)
                pseudo_latent = model.srn.fssae.encode(pseudo_image)
                pseudo_feature = model.srn.fssae.encoder(pseudo_image).flatten(start_dim=1)
                pixel_mse = F.mse_loss(pseudo_image, original_image.view_as(pseudo_image), reduction="mean").item()

                row = {
                    "run_id": self.analysis_run_id,
                    "event_index": event_index,
                    "meta_iteration": self.last_replay_batch["meta_iteration"] + 1,
                    "tracked_index": tracked_index,
                    "buffer_index": item["buffer_index"],
                    "label": int(label.item()),
                    "pixel_mse": pixel_mse,
                    "latent_cosine": F.cosine_similarity(stored_latent, pseudo_latent, dim=1).item(),
                    "feature_cosine": F.cosine_similarity(stored_feature, pseudo_feature, dim=1).item(),
                }
                self._write_csv_row(path, row, "tracked_header_written")

                sample_dir = os.path.join(
                    self.analysis_dir,
                    "tracked_samples",
                    f"sample_{tracked_index:02d}_label_{int(label.item())}",
                )
                os.makedirs(sample_dir, exist_ok=True)

                original_path = os.path.join(sample_dir, self._analysis_filename("original", "png"))
                if not os.path.exists(original_path):
                    save_image(
                        original_image.detach().cpu(),
                        original_path,
                        normalize=True,
                        value_range=(-1.0, 1.0),
                    )

                original_gray_path = os.path.join(sample_dir, self._analysis_filename("original_gray", "png"))
                if not os.path.exists(original_gray_path):
                    save_image(
                        self._to_grayscale_for_display(original_image.detach().cpu()),
                        original_gray_path,
                        normalize=True,
                        value_range=(-1.0, 1.0),
                    )

                save_image(
                    pseudo_image[0].detach().cpu(),
                    os.path.join(
                        sample_dir,
                        self._analysis_filename(f"reconstruction_event_{event_index:03d}", "png"),
                    ),
                    normalize=True,
                    value_range=(-1.0, 1.0),
                )
                save_image(
                    self._to_grayscale_for_display(pseudo_image.detach().cpu())[0],
                    os.path.join(
                        sample_dir,
                        self._analysis_filename(f"reconstruction_gray_event_{event_index:03d}", "png"),
                    ),
                    normalize=True,
                    value_range=(-1.0, 1.0),
                )

    def _save_latent_probe_data(self, event_index, meta_iteration, indices, stored_latents, labels):
        if not self.analysis_save_probe_data:
            return
        import numpy as np

        probe_dir = os.path.join(self.analysis_dir, "latent_probe_data")
        os.makedirs(probe_dir, exist_ok=True)
        path = os.path.join(
            probe_dir,
            self._analysis_filename(f"latent_probe_event_{event_index:03d}", "npz"),
        )
        np.savez_compressed(
            path,
            latents=stored_latents.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy(),
            buffer_indices=np.asarray(indices, dtype=np.int64),
            event_index=np.asarray([event_index], dtype=np.int64),
            meta_iteration=np.asarray([meta_iteration], dtype=np.int64),
        )

    def _expanded_eval_metrics(self, model, fast_weights, device, candidate_labels=None, event_index=None, meta_iteration=None):
        eval_count = min(self.analysis_expanded_eval_samples, len(self.latents))
        if eval_count <= 0:
            return {}

        indices = torch.randperm(len(self.latents))[:eval_count].tolist()
        stored_latents = torch.stack([self.latents[idx] for idx in indices]).to(device)
        stored_features = torch.stack([self.features[idx] for idx in indices]).to(device)
        stored_images = torch.stack([self.images[idx] for idx in indices]).to(device)
        labels = torch.stack([self.labels[idx] for idx in indices]).to(device).long()
        if event_index is not None and meta_iteration is not None:
            self._save_latent_probe_data(event_index, meta_iteration, indices, stored_latents, labels)

        pseudo_samples = model.srn.fssae.decode(stored_latents)
        outputs = model.forward_features(pseudo_samples, fast_weights=fast_weights)
        logits = outputs["logits"].detach()
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        correct = pred.eq(labels)
        top5_k = min(5, logits.size(1))
        top5_pred = logits.topk(top5_k, dim=1).indices
        top5_correct = top5_pred.eq(labels.view(-1, 1)).any(dim=1)

        task_accuracy = correct.float().mean().item()
        task_top5_accuracy = top5_correct.float().mean().item()
        if candidate_labels is not None:
            keep = torch.cat([candidate_labels.detach().view(-1).to(device).long(), labels]).unique(sorted=True)
        else:
            keep = labels.unique(sorted=True)
        visible_logits = logits.index_select(dim=1, index=keep)
        task_pred = keep[visible_logits.argmax(dim=1)]
        task_accuracy = task_pred.eq(labels).float().mean().item()
        task_top5_k = min(5, visible_logits.size(1))
        task_top5_pred = keep[visible_logits.topk(task_top5_k, dim=1).indices]
        task_top5_accuracy = task_top5_pred.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()

        pseudo_latents = model.srn.fssae.encode(pseudo_samples)
        pseudo_features = model.srn.fssae.encoder(pseudo_samples).flatten(start_dim=1)
        latent_cosine = F.cosine_similarity(stored_latents, pseudo_latents, dim=1)
        feature_cosine = F.cosine_similarity(stored_features, pseudo_features, dim=1)
        pixel_mse = F.mse_loss(pseudo_samples, stored_images, reduction="none").flatten(start_dim=1).mean(dim=1)

        metrics = {
            "expanded_eval_count": eval_count,
            "expanded_pseudo_accuracy": correct.float().mean().item(),
            "expanded_pseudo_top5_accuracy": top5_correct.float().mean().item(),
            "expanded_pseudo_task_accuracy": task_accuracy,
            "expanded_pseudo_task_top5_accuracy": task_top5_accuracy,
            "expanded_pseudo_confidence_mean": probs.max(dim=1).values.mean().item(),
            "expanded_pixel_mse_mean": pixel_mse.mean().item(),
            "expanded_pixel_mse_std": pixel_mse.std(unbiased=True).item() if len(pixel_mse) > 1 else 0.0,
            "expanded_latent_cosine_mean": latent_cosine.mean().item(),
            "expanded_latent_cosine_std": latent_cosine.std(unbiased=True).item() if len(latent_cosine) > 1 else 0.0,
            "expanded_feature_cosine_mean": feature_cosine.mean().item(),
            "expanded_feature_cosine_std": feature_cosine.std(unbiased=True).item() if len(feature_cosine) > 1 else 0.0,
        }
        if self.fidelity_enabled:
            reference_classifier = self._get_reference_classifier(device)
            fidelity, per_sample = compute_fidelity_metrics(
                stored_images,
                pseudo_samples,
                labels,
                reference_classifier,
                dataset=self.fidelity_dataset,
            )
            metrics.update({
                "fidelity_sample_count": fidelity.sample_count,
                "fidelity_pixel_mse_mean": fidelity.pixel_mse_mean,
                "fidelity_pixel_mse_std": fidelity.pixel_mse_std,
                "fidelity_ssim_mean": fidelity.ssim_mean,
                "fidelity_ssim_std": fidelity.ssim_std,
                "fidelity_frozen_feature_cosine_mean": fidelity.frozen_feature_cosine_mean,
                "fidelity_frozen_feature_cosine_std": fidelity.frozen_feature_cosine_std,
                "fidelity_independent_real_accuracy": fidelity.independent_real_accuracy,
                "fidelity_independent_pseudo_accuracy": fidelity.independent_pseudo_accuracy,
                "fidelity_label_preservation_rate": fidelity.label_preservation_rate,
            })
            if event_index is not None and meta_iteration is not None:
                sample_path = os.path.join(
                    self.analysis_dir,
                    self._analysis_filename("reconstruction_fidelity_event_samples", "csv"),
                )
                append_event_samples_csv(
                    sample_path,
                    self.analysis_run_id,
                    event_index,
                    meta_iteration,
                    indices,
                    labels,
                    per_sample,
                )
        return metrics

    def export_final_fidelity(self, model, meta_iteration):
        if not self.final_fidelity_export:
            return None
        if not self.fidelity_enabled:
            raise ValueError("--final-fidelity-export requires --reconstruction-fidelity")
        if not self.analysis_dir:
            raise ValueError("--final-fidelity-export requires --replay-analysis-dir")
        if not self.latents:
            raise RuntimeError("Cannot export final fidelity: replay buffer is empty.")

        os.makedirs(self.analysis_dir, exist_ok=True)
        device = next(model.parameters()).device
        stored_latents = torch.stack(self.latents)
        stored_images = torch.stack(self.images)
        labels = torch.stack(self.labels).long()
        pseudo_parts = []
        with torch.no_grad():
            for start in range(0, len(stored_latents), self.fidelity_eval_batch_size):
                latent_batch = stored_latents[start:start + self.fidelity_eval_batch_size].to(device)
                pseudo_parts.append(model.srn.fssae.decode(latent_batch).detach().cpu())
            pseudo_samples = torch.cat(pseudo_parts)
            fidelity, per_sample = compute_fidelity_metrics_batched(
                stored_images,
                pseudo_samples,
                labels,
                self._get_reference_classifier(device),
                dataset=self.fidelity_dataset,
                batch_size=self.fidelity_eval_batch_size,
            )

        indices = list(range(len(self.latents)))
        sample_path = os.path.join(
            self.analysis_dir,
            self._analysis_filename("reconstruction_fidelity_final_buffer_samples", "csv"),
        )
        append_event_samples_csv(
            sample_path,
            self.analysis_run_id,
            0,
            meta_iteration,
            indices,
            labels,
            per_sample,
        )
        summary_path = os.path.join(
            self.analysis_dir,
            self._analysis_filename("reconstruction_fidelity_final_buffer_summary", "json"),
        )
        write_summary_json(
            summary_path,
            fidelity,
            extra={
                "run_id": self.analysis_run_id,
                "meta_iteration": int(meta_iteration),
                "dataset": self.fidelity_dataset,
                "reference_classifier": os.path.abspath(self.reference_classifier_path),
            },
        )
        summary_csv_path = os.path.join(
            self.analysis_dir,
            self._analysis_filename("reconstruction_fidelity_final_buffer_summary", "csv"),
        )
        write_summary_csv(
            summary_csv_path,
            fidelity,
            extra={
                "run_id": self.analysis_run_id,
                "meta_iteration": int(meta_iteration),
                "dataset": self.fidelity_dataset,
                "reference_classifier": os.path.abspath(self.reference_classifier_path),
            },
        )
        pairs_path = os.path.join(
            self.analysis_dir,
            self._analysis_filename("reconstruction_fidelity_final_buffer_pairs", "pt"),
        )
        torch.save({
            "run_id": self.analysis_run_id,
            "meta_iteration": int(meta_iteration),
            "stored_images_model_space": stored_images.detach().cpu(),
            "stored_images_unit_interval": model_to_unit_interval(stored_images, self.fidelity_dataset).detach().cpu(),
            "stored_latents": stored_latents.detach().cpu(),
            "pseudo_images_model_space": pseudo_samples.detach().cpu(),
            "pseudo_images_unit_interval": model_to_unit_interval(pseudo_samples, self.fidelity_dataset).detach().cpu(),
            "labels": labels.detach().cpu(),
            "buffer_indices": torch.arange(len(labels)),
        }, pairs_path)
        return summary_path

    def record_analysis(self, model, logits, real_query_count, fast_weights=None, candidate_labels=None):
        if not self.analysis_dir:
            return
        if self.last_replay_batch is None:
            return
        if self.analysis_max_events and self.analysis_count >= self.analysis_max_events:
            return

        os.makedirs(self.analysis_dir, exist_ok=True)
        metrics_path = os.path.join(self.analysis_dir, self._analysis_filename("replay_pseudo_usability_events", "csv"))
        batch = self.last_replay_batch
        event_index = self.analysis_count + 1

        with torch.no_grad():
            pseudo_logits = logits[real_query_count : real_query_count + len(batch["labels"])].detach()
            pseudo_probs = F.softmax(pseudo_logits, dim=1)
            pseudo_pred = pseudo_logits.argmax(dim=1)
            pseudo_correct = pseudo_pred.eq(batch["labels"])
            pseudo_task_accuracy = pseudo_correct.float().mean().item()
            if candidate_labels is not None:
                keep = candidate_labels.detach().view(-1).unique(sorted=True).to(pseudo_logits.device).long()
                visible_logits = pseudo_logits.index_select(dim=1, index=keep)
                task_pred = keep[visible_logits.argmax(dim=1)]
                pseudo_task_accuracy = task_pred.eq(batch["labels"]).float().mean().item()

            pseudo_samples = batch["pseudo_samples"].to(pseudo_logits.device)
            stored_images = batch["stored_images"].to(pseudo_logits.device)
            pseudo_latents = model.srn.fssae.encode(pseudo_samples)
            pseudo_features = model.srn.fssae.encoder(pseudo_samples).flatten(start_dim=1)

            stored_latents = batch["stored_latents"].to(pseudo_logits.device)
            stored_features = batch["stored_features"].to(pseudo_logits.device)
            latent_cosine = F.cosine_similarity(stored_latents, pseudo_latents, dim=1)
            feature_cosine = F.cosine_similarity(stored_features, pseudo_features, dim=1)
            pixel_mse = F.mse_loss(pseudo_samples, stored_images, reduction="none").flatten(start_dim=1).mean(dim=1)
            self._append_cumulative_embedding_items(
                batch["labels"],
                stored_latents,
                pseudo_latents,
                stored_features,
                pseudo_features,
                event_index,
            )

            row = {
                "run_id": self.analysis_run_id,
                "meta_iteration": batch["meta_iteration"] + 1,
                "replay_count": len(batch["labels"]),
                "pseudo_accuracy": pseudo_correct.float().mean().item(),
                "pseudo_task_accuracy": pseudo_task_accuracy,
                "pseudo_confidence_mean": pseudo_probs.max(dim=1).values.mean().item(),
                "pixel_mse_mean": pixel_mse.mean().item(),
                "pixel_mse_std": pixel_mse.std(unbiased=True).item() if len(pixel_mse) > 1 else 0.0,
                "latent_cosine_mean": latent_cosine.mean().item(),
                "latent_cosine_std": latent_cosine.std(unbiased=True).item() if len(latent_cosine) > 1 else 0.0,
                "feature_cosine_mean": feature_cosine.mean().item(),
                "feature_cosine_std": feature_cosine.std(unbiased=True).item() if len(feature_cosine) > 1 else 0.0,
            }
            row.update(
                self._expanded_eval_metrics(
                    model,
                    fast_weights,
                    pseudo_logits.device,
                    candidate_labels=candidate_labels,
                    event_index=event_index,
                    meta_iteration=batch["meta_iteration"] + 1,
                )
            )

        self._write_csv_row(metrics_path, row, "analysis_header_written")
        self._save_embedding_plot(model, event_index)
        self._save_cumulative_embedding_plot(event_index)
        if fast_weights is not None:
            self._record_tracked_items(model, fast_weights, event_index)

        self.analysis_count += 1
