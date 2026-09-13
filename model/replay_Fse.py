import math
import os

import torch
from torchvision.utils import save_image

from analysis.reconstruction_fidelity import (
    append_event_samples_csv,
    compute_fidelity_metrics_batched,
    load_reference_classifier,
    model_to_unit_interval,
    write_summary_csv,
    write_summary_json,
)


class FSEReplayBuffer:
    """
    Paper-oriented replay backend for FSEML.

    This implementation follows the manuscript direction more closely than the
    default simple replay:
    - maintain partitioned latent buffers
    - rank current query samples by gradient-norm importance
    - build a Top-P candidate pool and sample without replacement according
      to importance-proportional probabilities
    - trigger replay according to sample-count gap instead of meta-step count
    - replay floor(r * GI) pseudo-samples decoded from stored latents
    """

    mode_name = "fse"

    def __init__(self, args):
        self.enabled = getattr(args, "replay", False)


        self.sampling_enabled = True
        self.buffer_size = max(1, getattr(args, "replay_buffer_size", 1000))
        self.replay_gap = max(1, getattr(args, "replay_gap", 960))
        self.replay_rate = max(0.0, min(1.0, getattr(args, "replay_rate", 0.05)))
        self.replay_content = getattr(args, "replay_content", "decoder").lower()
        if self.replay_content not in {"decoder", "raw"}:
            raise ValueError("replay_content must be 'decoder' or 'raw'.")
        self.top_p = max(1, getattr(args, "replay_top_p", 32))
        replay_class_ids = getattr(args, "replay_class_ids", None)
        self.replay_class_ids = (
            tuple(sorted(int(label) for label in replay_class_ids))
            if replay_class_ids else None
        )
        self.partition_by_label = (
            {label: index for index, label in enumerate(self.replay_class_ids)}
            if self.replay_class_ids else None
        )
        self.num_partitions = (
            len(self.replay_class_ids)
            if self.replay_class_ids else max(1, getattr(args, "replay_partitions", 10))
        )
        self.replay_seed = int(getattr(args, "seed", 9)) + 104729
        self.cpu_generator = torch.Generator(device="cpu")
        self.cpu_generator.manual_seed(self.replay_seed)
        self.device_generators = {}

        self.visualize = getattr(args, "visualize_replay", False)
        self.viz_dir = getattr(args, "replay_viz_dir", "replay_visualizations")
        self.viz_max = max(0, getattr(args, "replay_viz_max", 10))
        self.viz_count = 0
        self.dataset = getattr(args, "dataset", "omniglot")
        self.analysis_dir = getattr(args, "replay_analysis_dir", None)
        self.analysis_run_id = getattr(args, "replay_analysis_run_id", "replay")
        self.fidelity_enabled = getattr(args, "reconstruction_fidelity", False)
        self.reference_classifier_path = getattr(args, "reference_classifier", None)
        self.final_fidelity_export = getattr(args, "final_fidelity_export", False)
        self.fidelity_eval_batch_size = max(1, getattr(args, "fidelity_eval_batch_size", 128))
        self.reference_classifier = None
        if self.fidelity_enabled and not self.reference_classifier_path:
            raise ValueError("--reference-classifier is required with --reconstruction-fidelity")

        self.partition_capacity = max(1, self.buffer_size // self.num_partitions)
        self.partitions = [
            {
                "latents": [],
                "labels": [],
                "images": [],
                "position": 0,
            }
            for _ in range(self.num_partitions)
        ]
        self.samples_since_replay = 0
        self.replay_event_count = 0
        self.total_replayed_samples = 0
        self.last_replay_count = 0

    def _partition_index(self, label):
        if self.partition_by_label is not None:
            try:
                return self.partition_by_label[int(label)]
            except KeyError as exc:
                raise ValueError(f"Replay label {label} is outside configured class ids.") from exc
        return int(label) % self.num_partitions

    def _generator_for(self, device):
        key = str(torch.device(device))
        if key not in self.device_generators:
            generator = torch.Generator(device=torch.device(device))
            generator.manual_seed(self.replay_seed)
            self.device_generators[key] = generator
        return self.device_generators[key]

    def _iter_stored_items(self):
        for partition in self.partitions:
            for latent, label, image in zip(
                partition["latents"], partition["labels"], partition["images"]
            ):
                yield latent, label, image

    def _buffer_length(self):
        return sum(len(partition["latents"]) for partition in self.partitions)

    def status(self):
        partition_occupancy = [len(partition["labels"]) for partition in self.partitions]
        label_occupancy = {}
        for partition in self.partitions:
            for label in partition["labels"]:
                key = str(int(label.item()))
                label_occupancy[key] = label_occupancy.get(key, 0) + 1
        return {
            "mode": self.mode_name,
            "content": self.replay_content,
            "enabled": bool(self.enabled),
            "sampling_enabled": bool(self.sampling_enabled),
            "buffer_length": self._buffer_length(),
            "buffer_capacity": self.buffer_size,
            "replay_events": self.replay_event_count,
            "total_replayed_samples": self.total_replayed_samples,
            "last_replay_count": self.last_replay_count,
            "samples_since_replay": self.samples_since_replay,
            "replay_gap": self.replay_gap,
            "replay_rate": self.replay_rate,
            "top_p": self.top_p,
            "partitions": self.num_partitions,
            "partition_class_ids": list(self.replay_class_ids) if self.replay_class_ids else None,
            "partition_capacity": self.partition_capacity,
            "partition_occupancy": partition_occupancy,
            "label_occupancy": label_occupancy,
        }

    def should_replay(self, meta_iteration):
        del meta_iteration
        if not self.enabled:
            return False
        if not self.sampling_enabled:
            return False
        if self._buffer_length() == 0:
            return False
        return self.samples_since_replay >= self.replay_gap

    def _importance_scores(self, model, x, y):
        scores = []
        params = [param for param in model.parameters() if param.requires_grad]

        for idx in range(len(y)):
            sample_x = x[idx : idx + 1]
            sample_y = y[idx : idx + 1]
            outputs = model.forward_features(sample_x)
            loss = model.classification_loss(outputs["logits"], sample_y)
            grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)

            score = torch.zeros((), device=sample_x.device)
            for grad in grads:
                if grad is not None:
                    score = score + grad.pow(2).sum()
            scores.append(score.detach())

        return torch.stack(scores)

    def _store_item(self, latent_item, label_item, image_item):
        partition = self.partitions[self._partition_index(label_item.item())]
        latent_item = latent_item.clone().cpu()
        label_item = label_item.clone().cpu()


        retain_image = self.replay_content == "raw" or self.fidelity_enabled
        image_item = image_item.clone().cpu() if retain_image else None

        if len(partition["latents"]) < self.partition_capacity:
            partition["latents"].append(latent_item)
            partition["labels"].append(label_item)
            partition["images"].append(image_item)
        else:
            position = partition["position"]
            partition["latents"][position] = latent_item
            partition["labels"][position] = label_item
            partition["images"][position] = image_item
            partition["position"] = (position + 1) % self.partition_capacity

    def store_batch(self, model, x, y):
        if not self.enabled:
            return

        self.samples_since_replay += len(y)
        if len(y) == 0:
            return

        flat_y = y.detach().view(-1)
        importance = self._importance_scores(model, x, flat_y)
        top_k = min(self.top_p, len(flat_y))
        top_scores, top_indices = torch.topk(importance, k=top_k, largest=True, sorted=True)

        if top_scores.sum().item() <= 0:
            probs = torch.full((top_k,), 1.0 / top_k, device=top_scores.device)
        else:
            probs = top_scores / top_scores.sum()

        sample_count = min(top_k, self.partition_capacity)
        selected_relative = torch.multinomial(
            probs,
            num_samples=sample_count,
            replacement=False,
            generator=self._generator_for(probs.device),
        )
        selected_indices = top_indices[selected_relative]

        with torch.no_grad():
            selected_x = x[selected_indices]
            selected_y = flat_y[selected_indices]
            selected_latents = model.forward_srn(selected_x)["latent"].detach()

        for latent_item, label_item, image_item in zip(selected_latents, selected_y, selected_x):
            self._store_item(latent_item, label_item, image_item)

    def sample(self, model, query_batch_size, device, meta_iteration):
        if not self.should_replay(meta_iteration):
            return None, None

        del query_batch_size

        replay_count = int(math.floor(self.replay_rate * self.replay_gap))
        replay_count = max(1, replay_count)

        stored_items = list(self._iter_stored_items())
        replay_count = min(replay_count, len(stored_items))
        if replay_count <= 0:
            return None, None

        indices = torch.randperm(
            len(stored_items),
            generator=self.cpu_generator,
        )[:replay_count].tolist()
        latent_batch = torch.stack([stored_items[idx][0] for idx in indices]).to(device)
        label_batch = torch.stack([stored_items[idx][1] for idx in indices]).to(device)
        if self.replay_content == "raw":
            if any(stored_items[idx][2] is None for idx in indices):
                raise RuntimeError("Raw replay requested but buffer images were not retained.")
            replay_samples = torch.stack([stored_items[idx][2] for idx in indices]).to(device).detach()
        else:
            with torch.no_grad():
                replay_samples = model.srn.fssae.decode(latent_batch).detach()

        self.samples_since_replay = 0
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
            f"replay_fse_step_{meta_iteration + 1:06d}"
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

    def _get_reference_classifier(self, device):
        if self.reference_classifier is None:
            self.reference_classifier = load_reference_classifier(
                self.reference_classifier_path, device
            )
        return self.reference_classifier

    def _analysis_path(self, stem, extension):
        return os.path.join(
            self.analysis_dir,
            f"{stem}_{self.analysis_run_id}.{extension}",
        )

    def export_final_fidelity(self, model, meta_iteration):
        if not self.final_fidelity_export:
            return None
        if not self.fidelity_enabled:
            raise ValueError("--final-fidelity-export requires --reconstruction-fidelity")
        if not self.analysis_dir:
            raise ValueError("--final-fidelity-export requires --replay-analysis-dir")

        stored_items = list(self._iter_stored_items())
        if not stored_items:
            raise RuntimeError("Cannot export final fidelity: replay buffer is empty.")

        os.makedirs(self.analysis_dir, exist_ok=True)
        device = next(model.parameters()).device
        stored_latents = torch.stack([item[0] for item in stored_items]).to(device)
        labels = torch.stack([item[1] for item in stored_items]).long().to(device)
        stored_images = torch.stack([item[2] for item in stored_items]).to(device)
        with torch.no_grad():
            pseudo_images = model.srn.fssae.decode(stored_latents)
            summary, per_sample = compute_fidelity_metrics_batched(
                stored_images,
                pseudo_images,
                labels,
                self._get_reference_classifier(device),
                dataset=self.dataset,
                batch_size=self.fidelity_eval_batch_size,
            )

        extra = {
            "run_id": self.analysis_run_id,
            "dataset": self.dataset,
            "meta_iteration": int(meta_iteration),
            "reference_classifier": os.path.abspath(self.reference_classifier_path),
            "replay_mode": self.mode_name,
        }
        summary_path = self._analysis_path(
            "reconstruction_fidelity_final_buffer_summary", "json"
        )
        write_summary_json(summary_path, summary, extra=extra)
        write_summary_csv(
            self._analysis_path("reconstruction_fidelity_final_buffer_summary", "csv"),
            summary,
            extra=extra,
        )
        append_event_samples_csv(
            self._analysis_path("reconstruction_fidelity_final_buffer_samples", "csv"),
            self.analysis_run_id,
            0,
            meta_iteration,
            range(len(labels)),
            labels,
            per_sample,
        )

        real_unit = model_to_unit_interval(stored_images.detach(), self.dataset).cpu()
        pseudo_unit = model_to_unit_interval(pseudo_images.detach(), self.dataset).cpu()
        torch.save(
            {
                "run_id": self.analysis_run_id,
                "meta_iteration": int(meta_iteration),
                "stored_images_model_space": stored_images.detach().cpu(),
                "stored_images_unit_interval": real_unit,
                "stored_latents": stored_latents.detach().cpu(),
                "pseudo_images_model_space": pseudo_images.detach().cpu(),
                "pseudo_images_unit_interval": pseudo_unit,
                "labels": labels.detach().cpu(),
                "buffer_indices": torch.arange(len(labels)),
            },
            self._analysis_path("reconstruction_fidelity_final_buffer_pairs", "pt"),
        )

        example_count = min(16, len(labels))
        paired = torch.stack(
            [image for index in range(example_count) for image in (real_unit[index], pseudo_unit[index])]
        )
        save_image(
            paired,
            self._analysis_path("reconstruction_fidelity_final_buffer_examples", "png"),
            nrow=2,
            normalize=False,
            padding=2,
        )
        return summary_path
