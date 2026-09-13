from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.core50_heldout import CORe50HeldOutData, load_split
from model.meta_learner_fseml import MetaLearnerFSEML
from run_core50_experiment import set_seed


def learner_shell(model, inner_lr, device):
    config = model.config
    args = SimpleNamespace(
        update_lr=inner_lr, meta_lr=1e-4, update_step=10,
        dataset="core50", in_channels=config.in_channels, image_size=config.image_size,
        num_classes=config.num_classes, encoder_channels=config.encoder_channels,
        latent_dim=config.latent_dim, adapter_dim=config.adapter_dim,
        cpn_hidden_dim=getattr(config, "cpn_hidden_dim", model.cpn.hidden.out_features),
        normalization=getattr(config, "normalization", "batch"),
        replay=False, replay_mode="fse", replay_content="decoder",
        replay_buffer_size=1, replay_gap=1, replay_rate=0.0,
        replay_top_p=1, replay_partitions=50, seed=9,
        visualize_replay=False, replay_analysis_dir=None,
    )
    learner = MetaLearnerFSEML(args).to(device)
    learner.net = copy.deepcopy(model).to(device)




    learner.eval()
    return learner


def reset_rows(learner, classes, initialization):
    for label in classes:
        learner.reset_classifer(int(label), initialization=initialization)


def adapt(learner, fast_weights, dataset, steps, device, workers):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=workers)
    for _ in range(steps):
        images, labels = next(iter(loader))
        images, labels = images.to(device), labels.to(device)


        fast_weights = learner.inner_update(images, fast_weights, labels)
    return fast_weights


@torch.no_grad()
def accuracy(model, fast_weights, dataset, allowed_classes, device, batch_size, workers):
    allowed = torch.tensor(sorted(allowed_classes), device=device)
    correct = total = 0
    for images, labels in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    ):
        images, labels = images.to(device), labels.to(device)
        logits = model.forward_features(images, fast_weights=fast_weights)["logits"]
        predictions = allowed[logits.index_select(1, allowed).argmax(1)]
        correct += predictions.eq(labels).sum().item()
        total += len(labels)
    return correct / max(1, total)


def evaluate(checkpoint, data, classes, args, inner_lr, device):


    set_seed(args.seed)
    learner = learner_shell(checkpoint["model"], inner_lr, device)
    learner.set_active_classes(classes)
    tasks = [classes[i:i + args.ways] for i in range(0, len(classes), args.ways)]
    fast_weights = None
    seen_classes = []
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    for task_index, task_classes in enumerate(tasks):
        reset_rows(learner, task_classes, args.classifier_reset_init)
        seen_classes.extend(task_classes)
        learner.set_active_classes(seen_classes)


        if fast_weights is None:
            fast_weights = [parameter for _, parameter in learner.net.cpn_named_parameters()]
        else:
            fast_weights = [parameter.clone() for parameter in fast_weights]
            with torch.no_grad():
                fast_weights[-2][task_classes] = learner.net.cpn.output.weight[task_classes]
                fast_weights[-1][task_classes] = learner.net.cpn.output.bias[task_classes]
        support = data.adaptation_set(
            task_classes, args.support_shots, args.seed + task_index, train=True
        )
        fast_weights = adapt(
            learner, fast_weights, support, args.adaptation_epochs, device, args.workers
        )
        for eval_index in range(task_index + 1):
            eval_classes = tasks[eval_index]
            query = data.official_test_set(eval_classes)
            allowed = eval_classes if args.prediction_class_scope == "task" else classes[: (task_index + 1) * args.ways]
            matrix[task_index, eval_index] = accuracy(
                learner.net, fast_weights, query, allowed, device,
                args.eval_batch_size, args.workers,
            )
    final = matrix[len(tasks) - 1, :len(tasks)]
    return {
        "inner_lr": inner_lr,
        "matrix": matrix.tolist(),
        "final_ACC": float(np.nanmean(final)),
        "learning_ACC": float(np.nanmean(np.diag(matrix))),
        "forgetting": float(np.nanmean([
            np.nanmax(matrix[:len(tasks) - 1, i]) - matrix[len(tasks) - 1, i]
            for i in range(max(0, len(tasks) - 1))
        ])) if len(tasks) > 1 else 0.0,
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split = load_split(args.split)
    key = "meta_validation_classes" if args.phase == "validation" else "meta_test_classes"
    classes = split[key]
    data = CORe50HeldOutData(
        args.data_root, args.metadata_root, run=args.run, image_size=args.image_size
    )
    if args.phase == "validation":
        candidates = args.inner_lrs
    else:
        if not args.selection:
            raise ValueError("--selection is required for phase=test.")
        selection = json.loads(Path(args.selection).read_text())
        candidates = [float(selection["selected_inner_lr"])]
    results = [evaluate(checkpoint, data, classes, args, lr, device) for lr in candidates]
    best = max(results, key=lambda item: item["final_ACC"])
    payload = {
        "phase": args.phase,
        "classes": classes,
        "prediction_class_scope": args.prediction_class_scope,
        "results": results,
        "selected_inner_lr": best["inner_lr"],
        "best": best,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--metadata-root", default="vendor/core50_official")
    p.add_argument("--split", default="core50_heldout_splits/split_seed9.json")
    p.add_argument("--phase", choices=["validation", "test"], required=True)
    p.add_argument("--selection", help="Validation JSON required for phase=test.")
    p.add_argument("--output", required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--ways", type=int, default=5)
    p.add_argument("--support-shots", type=int, default=2)
    p.add_argument(
        "--adaptation-epochs",
        type=int,
        default=10,
        help="Number of full-support inner-loop gradient steps.",
    )
    p.add_argument("--inner-lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--prediction-class-scope", choices=["task", "seen"], default="task")
    p.add_argument(
        "--classifier-reset-init",
        choices=["kaiming", "zero"],
        default="kaiming",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
