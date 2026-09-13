from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from datasets.cifar100_continual import CIFAR100ContinualData, load_cifar100_protocol
from utils.utils import set_seed


def load_model(path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        model = checkpoint["model"]
    else:
        raise ValueError(
            "Expected a saved FSEML model object. The current training script saves "
            "this format through torch.save(maml.net, --model_name)."
        )
    if not hasattr(model, "cpn") or not hasattr(model, "forward_features"):
        raise TypeError("Checkpoint is not an FSEML model with CPN and forward_features().")
    if model.cpn.output.out_features != 100:
        raise ValueError(
            f"CIFAR100 evaluation requires a 100-class checkpoint, got "
            f"{model.cpn.output.out_features} outputs."
        )
    return model.to(device)


def configure_finetuning(
    model: nn.Module, scope: str
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if scope == "output":
        cpn_parameters = list(model.cpn.output.parameters())
        srn_parameters = []
    elif scope == "cpn":
        cpn_parameters = list(model.cpn.parameters())
        srn_parameters = []
    elif scope == "cpn-partial-srn":
        cpn_parameters = list(model.cpn.parameters())
        partial_srn_prefixes = (
            "srn.fssae.encoder.8.",
            "srn.fssae.encoder.9.",
            "srn.fssae.to_latent.",
        )
        srn_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(partial_srn_prefixes)
        ]
        if not srn_parameters:
            raise ValueError("No parameters matched the partial SRN adaptation scope.")
    else:
        raise ValueError("finetune scope must be output, cpn, or cpn-partial-srn.")
    for parameter in [*cpn_parameters, *srn_parameters]:
        parameter.requires_grad_(True)
    return cpn_parameters, srn_parameters


def reset_output_layer(model: nn.Module) -> None:
    torch.nn.init.kaiming_normal_(model.cpn.output.weight)
    if model.cpn.output.bias is not None:
        model.cpn.output.bias.data.zero_()


def class_blocked_indices(dataset: Subset) -> List[int]:
    """Match legacy adaptation: shuffled examples grouped into random class blocks."""
    if not isinstance(dataset, Subset):
        raise TypeError("Class-blocked CIFAR100 adaptation expects a Subset.")
    targets = getattr(dataset.dataset, "targets", None)
    if targets is None:
        raise TypeError("The underlying CIFAR100 dataset must expose targets.")
    by_class: Dict[int, List[int]] = {}
    for local_index, source_index in enumerate(dataset.indices):
        by_class.setdefault(int(targets[source_index]), []).append(local_index)
    class_order = np.random.permutation(sorted(by_class)).tolist()
    ordered_indices: List[int] = []
    for class_id in class_order:
        ordered_indices.extend(np.random.permutation(by_class[int(class_id)]).tolist())
    return ordered_indices


def labels_to_seen_positions(labels: torch.Tensor, seen_classes: torch.Tensor, num_classes: int = 100):
    mapping = torch.full((num_classes,), -1, dtype=torch.long, device=labels.device)
    mapping[seen_classes] = torch.arange(len(seen_classes), device=labels.device)
    positions = mapping[labels]
    if torch.any(positions < 0):
        raise ValueError("A fine-tuning label is outside the currently seen class set.")
    return positions


def seen_class_cross_entropy(logits, labels, seen_classes):
    positions = labels_to_seen_positions(labels, seen_classes, logits.size(1))
    return F.cross_entropy(logits.index_select(1, seen_classes), positions)


def finetune_task(
    model,
    loader,
    current_classes,
    seen_classes,
    device,
    learning_rate,
    srn_learning_rate,
    epochs,
    scope,
    loss_class_scope,
    momentum,
    weight_decay,
    optimizer_name,
    progress_prefix=None,
):
    cpn_parameters, srn_parameters = configure_finetuning(model, scope)
    parameter_groups = [{"params": cpn_parameters, "lr": learning_rate}]
    if srn_parameters:
        parameter_groups.append({"params": srn_parameters, "lr": srn_learning_rate})
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameter_groups, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optimizer_name must be adam or sgd.")

    model.eval()
    for epoch_index in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model.forward_features(images)["logits"]
            if loss_class_scope == "current":
                loss = seen_class_cross_entropy(logits, labels, current_classes)
            elif loss_class_scope == "seen":
                loss = seen_class_cross_entropy(logits, labels, seen_classes)
            elif loss_class_scope == "all":
                loss = F.cross_entropy(logits, labels)
            else:
                raise ValueError("loss_class_scope must be current, seen, or all.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if progress_prefix:
            print(
                f"{progress_prefix} finetune epoch {epoch_index + 1}/{epochs} complete",
                flush=True,
            )


@torch.no_grad()
def evaluate_task(model, loader, task_classes, seen_classes, device, prediction_class_scope):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model.forward_features(images)["logits"]
        if prediction_class_scope == "task":
            predictions = task_classes[logits.index_select(1, task_classes).argmax(dim=1)]
        elif prediction_class_scope == "seen":
            predictions = seen_classes[logits.index_select(1, seen_classes).argmax(dim=1)]
        elif prediction_class_scope == "all":
            predictions = logits.argmax(dim=1)
        else:
            raise ValueError("prediction_class_scope must be task, seen, or all.")
        correct += predictions.eq(labels).sum().item()
        total += len(labels)
    return correct / total


def metrics_from_matrix(matrix: np.ndarray, learned_tasks: int) -> Dict[str, float]:
    k = learned_tasks
    final_row = matrix[k - 1, :k]
    accuracy = float(np.mean(final_row))
    learning_accuracy = float(np.mean(np.diag(matrix[:k, :k])))
    if k == 1:
        forgetting = 0.0
    else:
        forgetting_values = []
        for task_index in range(k - 1):
            previous_best = float(np.max(matrix[task_index:k - 1, task_index]))
            forgetting_values.append(previous_best - float(matrix[k - 1, task_index]))
        forgetting = float(np.mean(forgetting_values))
    return {"ACC": accuracy, "FM": forgetting, "LA": learning_accuracy}


def write_matrix_csv(
    path: str,
    matrix: np.ndarray,
    task_ids: Sequence[int],
    loss_class_scope: str,
    prediction_class_scope: str,
    evaluation_mode: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "after_task", "evaluation_mode", "loss_class_scope", "prediction_class_scope",
            *[f"eval_task_{task_id}" for task_id in task_ids],
        ])
        for row_index, task_id in enumerate(task_ids):
            writer.writerow([
                task_id,
                evaluation_mode,
                loss_class_scope,
                prediction_class_scope,
                *[
                    "" if math.isnan(matrix[row_index, column]) else matrix[row_index, column]
                    for column in range(len(task_ids))
                ],
            ])


def main(args):
    set_seed(args.seed)
    if args.shuffle_finetune:
        args.finetune_order = "shuffle"
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    protocol = load_cifar100_protocol(args.task_protocol)
    if protocol.classes_per_task != args.classes_per_task:
        raise ValueError("--classes-per-task does not match the protocol manifest.")
    data = CIFAR100ContinualData(
        root=args.data_root,
        protocol=protocol,
        image_size=28,
        augment_train=args.data_augmentation,
        download_if_missing=True,
    )
    if args.restrict_to_seen_classes is not None:

        args.loss_class_scope = "seen" if args.restrict_to_seen_classes else "all"
        args.prediction_class_scope = "seen" if args.restrict_to_seen_classes else "all"
    if args.loss_class_scope is None:
        args.loss_class_scope = "seen" if args.evaluation_mode == "legacy-cumulative" else "current"
    if args.prediction_class_scope is None:
        args.prediction_class_scope = "task"
    if args.optimizer is None:
        args.optimizer = "adam" if args.evaluation_mode == "legacy-cumulative" else "sgd"
    if args.reset_output is None:
        args.reset_output = args.evaluation_mode == "legacy-cumulative"
    test_tasks = protocol.meta_test_tasks
    task_ids = [task.task_id for task in test_tasks]
    matrix = np.full((len(test_tasks), len(test_tasks)), np.nan, dtype=np.float64)
    events = []
    seen_classes: List[int] = []
    model = None

    print(
        f"Starting {args.evaluation_mode} evaluation on {device}: "
        f"{len(test_tasks)} meta-test tasks; adaptation loss scope="
        f"{args.loss_class_scope}; reporting prediction scope="
        f"{args.prediction_class_scope}",
        flush=True,
    )

    for learned_index, task in enumerate(test_tasks):
        seen_classes.extend(task.class_ids)
        seen_tensor = torch.tensor(sorted(seen_classes), dtype=torch.long, device=device)
        current_tensor = torch.tensor(sorted(task.class_ids), dtype=torch.long, device=device)
        if args.evaluation_mode == "legacy-cumulative":



            model = load_model(args.model, device)
            if args.reset_output:
                reset_output_layer(model)
            learned_task_ids = [item.task_id for item in test_tasks[:learned_index + 1]]
            train_dataset = data.combined_dataset(learned_task_ids, split="train")
            train_sampler = None
            train_shuffle = args.finetune_order == "shuffle"
            if args.finetune_order == "class-blocked":
                train_sampler = class_blocked_indices(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.finetune_batch_size,
                shuffle=train_shuffle,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            loss_current_tensor = seen_tensor
        else:
            if model is None:
                model = load_model(args.model, device)
                if args.reset_output:
                    reset_output_layer(model)
            train_loader = data.loader(
                task.task_id, "train", args.finetune_batch_size,
                shuffle=args.shuffle_finetune, num_workers=args.num_workers,
            )
            loss_current_tensor = current_tensor
        finetune_task(
            model, train_loader, loss_current_tensor, seen_tensor, device,
            learning_rate=args.finetune_lr,
            srn_learning_rate=args.srn_finetune_lr,
            epochs=args.finetune_epochs,
            scope=args.finetune_scope,
            loss_class_scope=args.loss_class_scope,
            momentum=args.momentum,
            weight_decay=args.finetune_weight_decay,
            optimizer_name=args.optimizer,
            progress_prefix=f"stage {learned_index + 1}/{len(test_tasks)}",
        )
        for eval_index in range(learned_index + 1):
            eval_task = test_tasks[eval_index]
            eval_task_tensor = torch.tensor(
                sorted(eval_task.class_ids), dtype=torch.long, device=device,
            )
            test_loader = data.loader(
                eval_task.task_id, "test", args.eval_batch_size,
                shuffle=False, num_workers=args.num_workers,
            )
            matrix[learned_index, eval_index] = evaluate_task(
                model,
                test_loader,
                eval_task_tensor,
                seen_tensor,
                device,
                args.prediction_class_scope,
            )
        metrics = metrics_from_matrix(matrix, learned_index + 1)
        event = {
            "learned_task_index": learned_index + 1,
            "task_id": task.task_id,
            "task_classes": task.class_ids,
            "seen_classes": sorted(seen_classes),
            "finetune_order": args.finetune_order,
            **metrics,
        }
        events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or f"cifar100_{args.classes_per_task}classes_seed{args.seed}"
    matrix_path = os.path.join(args.output_dir, f"{run_name}_R_matrix.csv")
    result_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    write_matrix_csv(
        matrix_path,
        matrix,
        task_ids,
        args.loss_class_scope,
        args.prediction_class_scope,
        args.evaluation_mode,
    )
    payload = {
        "run_name": run_name,
        "model": os.path.abspath(args.model),
        "task_protocol": os.path.abspath(args.task_protocol),
        "classes_per_task": args.classes_per_task,
        "seed": args.seed,
        "finetune_scope": args.finetune_scope,
        "evaluation_mode": args.evaluation_mode,
        "optimizer": args.optimizer,
        "reset_output": args.reset_output,
        "finetune_lr": args.finetune_lr,
        "srn_finetune_lr": args.srn_finetune_lr,
        "finetune_epochs": args.finetune_epochs,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_order": args.finetune_order,
        "data_augmentation": args.data_augmentation,
        "evaluation_protocol": args.evaluation_mode,
        "loss_class_scope": args.loss_class_scope,
        "prediction_class_scope": args.prediction_class_scope,
        "events": events,
        "final": events[-1],
    }
    with open(result_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(
        json.dumps({"matrix": os.path.abspath(matrix_path), "metrics": os.path.abspath(result_path)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential CIFAR100 meta-test evaluation for ACC/FM/LA.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", default="../data/cifar100")
    parser.add_argument("--task-protocol", required=True)
    parser.add_argument("--classes-per-task", type=int, choices=[5, 10, 20], required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="cifar100_main_results")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument(
        "--evaluation-mode",
        choices=["legacy-cumulative", "sequential"],
        default="legacy-cumulative",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=2e-4,
        help="Validated CIFAR100 default; override explicitly for LR ablations.",
    )
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--finetune-scope",
        choices=["output", "cpn", "cpn-partial-srn"],
        default="cpn",
        help=(
            "cpn-partial-srn additionally adapts the final encoder block and "
            "latent projection while keeping the decoder frozen."
        ),
    )
    parser.add_argument(
        "--srn-finetune-lr",
        type=float,
        default=2e-5,
        help="Learning rate for the partial SRN group when enabled.",
    )
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument(
        "--reset-output", action=argparse.BooleanOptionalAction, default=None,
        help="Defaults to enabled in legacy-cumulative mode, matching Omniglot evaluation.",
    )
    parser.add_argument(
        "--shuffle-finetune", action=argparse.BooleanOptionalAction, default=False,
        help="Deprecated compatibility flag; prefer --finetune-order shuffle.",
    )
    parser.add_argument(
        "--finetune-order",
        choices=["class-blocked", "dataset", "shuffle"],
        default="dataset",
        help=(
            "CIFAR100 default preserves the dataset's mixed order; class-blocked "
            "is retained as an explicit Omniglot-alignment diagnostic."
        ),
    )
    parser.add_argument(
        "--loss-class-scope", choices=["current", "seen", "all"], default=None,
        help="Logit classes included in the fine-tuning cross-entropy loss.",
    )
    parser.add_argument(
        "--prediction-class-scope", choices=["task", "seen", "all"], default=None,
        help=(
            "Candidate classes used when evaluating each task. The FSEML main "
            "protocol defaults to task-aware reporting."
        ),
    )
    parser.add_argument(
        "--restrict-to-seen-classes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--data-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enabled by default for CIFAR100; use --no-data-augmentation for "
            "the deterministic Omniglot-alignment diagnostic."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
