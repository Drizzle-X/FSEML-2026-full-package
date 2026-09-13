from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.core50_continual import CORe50Metadata, core50_eval_transform, core50_train_transform
from model.meta_learner_fseml import MetaLearnerFSEML


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def replay_checkpoint(replay):
    """Return a device-portable replay snapshot without pickling Generators."""
    state = {
        "format_version": 1,
        "status": replay.status(),
        "replay_seed": replay.replay_seed,
        "cpu_generator_state": replay.cpu_generator.get_state().cpu(),
    }
    if hasattr(replay, "partitions"):
        state["partitions"] = replay.partitions
        state["partition_capacity"] = replay.partition_capacity
    return state


def learner_args(args):
    return SimpleNamespace(
        dataset="core50", in_channels=3, image_size=args.image_size, num_classes=50,
        encoder_channels=args.encoder_channels, latent_dim=args.latent_dim, adapter_dim=args.adapter_dim,
        cpn_hidden_dim=getattr(args, "cpn_hidden_dim", 2304),
        normalization=getattr(args, "normalization", "batch"),
        sparsity_target=0.05, sparsity_weight=5e-3, reconstruction_weight=0.1,
        fisher_reconstruction_weight=0.1, weight_decay_weight=0.0, fda_weight=0.1,
        update_lr=args.inner_lr, update_step=args.update_step, meta_lr=args.learning_rate,
        replay=args.replay, replay_mode="fse", replay_content=args.replay_content,
        replay_buffer_size=args.buffer_size, replay_gap=args.replay_gap, replay_rate=args.replay_rate,
        replay_top_p=32, replay_partitions=args.replay_partitions,
        replay_class_ids=getattr(args, "replay_class_ids", None),
        seed=args.seed, visualize_replay=False,
        replay_analysis_dir=None,
    )


@torch.no_grad()
def evaluate(model, dataset, seen, device, batch_size, workers):
    model.eval(); allowed = torch.tensor(sorted(seen), device=device); correct = total = 0
    class_correct = {label: 0 for label in sorted(seen)}
    class_total = {label: 0 for label in sorted(seen)}
    prediction_counts = {label: 0 for label in sorted(seen)}
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers):
        x, y = x.to(device), y.to(device)
        logits = model.forward_features(x)["logits"].index_select(1, allowed)
        pred = allowed[logits.argmax(1)]; matches = pred.eq(y)
        correct += matches.sum().item(); total += len(y)
        for label in allowed.tolist():
            label_mask = y.eq(label)
            class_total[label] += label_mask.sum().item()
            class_correct[label] += (matches & label_mask).sum().item()
            prediction_counts[label] += pred.eq(label).sum().item()
    return {
        "ACC": correct / max(1, total),
        "per_class_ACC": {
            str(label): class_correct[label] / max(1, class_total[label])
            for label in class_total
        },
        "prediction_histogram": {
            str(label): prediction_counts[label] for label in prediction_counts
        },
        "dominant_prediction_fraction": max(prediction_counts.values(), default=0) / max(1, total),
    }


def evaluate_task_aware(model, test, task_classes, device, batch_size, workers):
    task_results = {}
    class_accuracy = {}
    prediction_histogram = {}
    total_predictions = 0
    for task_index, classes in enumerate(task_classes):
        class_set = set(classes)
        subset = torch.utils.data.Subset(
            test, [i for i, label in enumerate(test.labels) if label in class_set]
        )
        result = evaluate(model, subset, class_set, device, batch_size, workers)
        task_results[str(task_index)] = result["ACC"]
        class_accuracy.update(result["per_class_ACC"])
        for label, count in result["prediction_histogram"].items():
            prediction_histogram[label] = prediction_histogram.get(label, 0) + count
        task_total = sum(result["prediction_histogram"].values())
        total_predictions += task_total
    return {
        "ACC": sum(task_results.values()) / max(1, len(task_results)),
        "per_task_ACC": task_results,
        "per_class_ACC": class_accuracy,
        "prediction_histogram": prediction_histogram,
        "dominant_prediction_fraction": max(prediction_histogram.values(), default=0) / max(1, total_predictions),
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = CORe50Metadata(args.data_root, args.metadata_root, args.scenario, args.run)
    learner = MetaLearnerFSEML(learner_args(args)).to(device)
    test = data.test(core50_eval_transform(args.image_size)); seen = set(); events = []
    task_classes = []
    stream_samples = 0
    stop = data.num_batches if args.max_batches <= 0 else min(data.num_batches, args.max_batches)
    for batch_index in range(stop):
        dataset = data.batch(batch_index, core50_train_transform(args.image_size))
        current_classes = tuple(sorted(set(dataset.labels)))
        task_classes.append(current_classes)
        seen.update(current_classes)
        learner.set_active_classes(seen)
        if hasattr(learner.replay, "sampling_enabled"):



            learner.replay.sampling_enabled = batch_index > 0
        stream_samples += len(dataset)
        epochs = args.initial_epochs if batch_index == 0 else 1
        train_accuracy_sum = train_loss_sum = 0.0
        meta_updates = optimization_samples = 0
        learner.train()
        for _ in range(epochs):


            loader = DataLoader(
                dataset,
                batch_size=args.query_batch_size + args.update_step,
                shuffle=batch_index == 0 or args.shuffle_experiences,
                num_workers=args.workers,
            )
            for x, y in loader:
                if len(x) <= args.update_step: continue
                support_x = x[:args.update_step].unsqueeze(1).to(device)
                support_y = y[:args.update_step].unsqueeze(1).to(device)
                query_x = x[args.update_step:].unsqueeze(0).to(device)
                query_y = y[args.update_step:].unsqueeze(0).to(device)
                train_accuracy, loss, _, _ = learner(support_x, support_y, query_x, query_y)
                train_accuracy_sum += float(train_accuracy)
                train_loss_sum += float(loss.detach())
                meta_updates += 1
                optimization_samples += len(x)
        should_evaluate = batch_index == stop - 1 or batch_index % args.eval_every == 0
        if should_evaluate:
            if args.prediction_class_scope == "task":
                evaluation = evaluate_task_aware(
                    learner.net, test, task_classes, device, args.eval_batch_size, args.workers
                )
            else:
                filtered = torch.utils.data.Subset(test, [i for i, label in enumerate(test.labels) if label in seen])
                evaluation = evaluate(learner.net, filtered, seen, device, args.eval_batch_size, args.workers)
            event = {
                "batch": batch_index,
                "seen_classes": len(seen),
                **evaluation,
                "experience_samples": len(dataset),
                "stream_samples": stream_samples,
                "optimization_samples": optimization_samples,
                "epochs": epochs,
                "meta_updates": meta_updates,
                "train_query_ACC": train_accuracy_sum / max(1, meta_updates),
                "train_loss": train_loss_sum / max(1, meta_updates),
            }
            events.append(event); print(json.dumps(event), flush=True)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    name = f"core50_{args.scenario}_{'fseml_er' if args.replay else 'fseml'}_run{args.run}_seed{args.seed}"
    payload = {"run": name, "events": events, "final": events[-1], "arguments": vars(args), "replay": learner.replay.status()}
    (output / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


    torch.save(
        {"model": learner.net, "replay": replay_checkpoint(learner.replay)},
        output / f"{name}.pt",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-root", default="../data/core50_128x128")
    p.add_argument("--metadata-root", default="vendor/core50_official"); p.add_argument("--scenario", default="nicv2_391")
    p.add_argument("--run", type=int, default=0); p.add_argument("--seed", type=int, default=9); p.add_argument("--output-dir", default="core50_results")
    p.add_argument("--prediction-class-scope", choices=["task", "seen"], default="seen")
    p.add_argument("--shuffle-experiences", action=argparse.BooleanOptionalAction, default=False, help="Shuffle every training experience, matching task-based CIFAR100/Omniglot evaluation.")
    p.add_argument("--image-size", type=int, default=128); p.add_argument("--encoder-channels", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=128); p.add_argument("--adapter-dim", type=int, default=256)
    p.add_argument("--cpn-hidden-dim", type=int, default=2304)
    p.add_argument("--normalization", choices=["batch", "group"], default="batch")
    p.add_argument("--learning-rate", type=float, default=5e-4); p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--update-step", type=int, default=10); p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--initial-epochs", type=int, default=5, help="Epochs over the shuffled multi-class batch 0; later batches remain single-pass.")
    p.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--replay-content", choices=["decoder", "raw"], default="decoder")
    p.add_argument("--buffer-size", type=int, default=1000); p.add_argument("--replay-gap", type=int, default=256); p.add_argument("--replay-rate", type=float, default=0.05)
    p.add_argument("--replay-partitions", type=int, default=50, help="Use one partition per CORe50 class by default.")
    p.add_argument("--eval-every", type=int, default=10); p.add_argument("--eval-batch-size", type=int, default=128); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0); p.add_argument("--cpu", action="store_true"); main(p.parse_args())
