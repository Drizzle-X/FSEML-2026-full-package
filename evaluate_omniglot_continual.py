#!/usr/bin/env python3
"""Leakage-safe sequential Omniglot evaluation for manuscript Table I."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import datasets.datasetfactory as df
from continual_metrics import class_order, metrics_from_matrix


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Expected a serialized torch.nn.Module or {'model': module} checkpoint.")
    return model.to(device)


def output_layer(model):
    if hasattr(model, "cpn") and hasattr(model.cpn, "output"):
        return model.cpn.output
    parameters = list(model.parameters())
    if len(parameters) < 2:
        raise TypeError("Could not identify the checkpoint output layer.")
    return None


def configure_output_only(model) -> List[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = output_layer(model)
    if layer is not None:
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.zero_()
        parameters = list(layer.parameters())
    else:
        parameters = list(model.parameters())[-2:]
        torch.nn.init.kaiming_normal_(parameters[0])
        parameters[1].data.zero_()
    for parameter in parameters:
        parameter.requires_grad_(True)

        parameter.learn = True
    return parameters


def logits_for(model, images):
    if hasattr(model, "forward_features"):
        return model.forward_features(images)["logits"]
    result = model(images)
    return result[-1] if isinstance(result, (tuple, list)) else result


def indices_by_class(dataset) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    for index, target in enumerate(dataset.targets):
        result.setdefault(int(target), []).append(index)
    return result


@torch.no_grad()
def evaluate_class(model, dataset, indices, visible_classes, device, batch_size, num_workers):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    visible = torch.as_tensor(visible_classes, dtype=torch.long, device=device)
    correct = total = 0
    model.eval()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = logits_for(model, images)
        predictions = visible[logits.index_select(1, visible).argmax(dim=1)]
        correct += predictions.eq(labels).sum().item()
        total += len(labels)
    return correct / total if total else 0.0


def adapt_one_class(model, optimizer, dataset, indices, device, epochs, num_workers):
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(logits_for(model, images), labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_stream(model, train_dataset, test_dataset, order, learning_rate, epochs,
               device, eval_batch_size, num_workers):
    parameters = configure_output_only(model)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    train_indices = indices_by_class(train_dataset)
    test_indices = indices_by_class(test_dataset)
    matrix = np.full((len(order), len(order)), np.nan, dtype=np.float64)
    events = []
    for learned_index, class_id in enumerate(order):
        adapt_one_class(model, optimizer, train_dataset, train_indices[class_id], device,
                        epochs, num_workers)
        visible = order[:learned_index + 1]
        for eval_index, eval_class in enumerate(visible):
            matrix[learned_index, eval_index] = evaluate_class(
                model, test_dataset, test_indices[eval_class], visible, device,
                eval_batch_size, num_workers,
            )
        event = {"learned_task_index": learned_index + 1, "class_id": class_id,
                 **metrics_from_matrix(matrix.tolist(), learned_index + 1)}
        events.append(event)
        print(json.dumps(event, sort_keys=True), flush=True)
    return matrix, events


def write_matrix(path: Path, matrix: np.ndarray, order: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task", *[f"eval_task_{i + 1}_class_{c}" for i, c in enumerate(order)]])
        for row_index, class_id in enumerate(order):
            writer.writerow([f"task_{row_index + 1}_class_{class_id}", *[
                "" if math.isnan(value) else value for value in matrix[row_index]
            ]])


def select_learning_rate(args, device, checkpoint):
    train = df.DatasetFactory.get_dataset("omniglot", train=True, background=True,
                                          path=args.dataset_path)
    query = df.DatasetFactory.get_dataset("omniglot", train=False, background=True,
                                          path=args.dataset_path)
    available = int(max(train.targets)) + 1
    order = class_order(available, args.seed)[:args.selection_tasks]
    scores = {}
    for learning_rate in args.lr_candidates:


        set_seed(args.seed)
        matrix, _ = run_stream(copy.deepcopy(checkpoint), train, query, order,
                               learning_rate, args.epochs, device,
                               args.eval_batch_size, args.num_workers)
        scores[str(learning_rate)] = metrics_from_matrix(matrix.tolist())["ACC"]
    selected = min(args.lr_candidates, key=lambda lr: (-scores[str(lr)], lr))
    payload = {
        "schema_version": 1,
        "phase": "selection",
        "data_scope": "omniglot_background_meta_training_classes_only",
        "seed": args.seed,
        "model": os.path.abspath(args.model),
        "selection_tasks": args.selection_tasks,
        "class_order": order,
        "candidate_learning_rates": args.lr_candidates,
        "scores": scores,
        "selected_learning_rate": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())


def run_test(args, device, checkpoint):
    if not args.selection or not args.selection.is_file():
        raise FileNotFoundError("--selection must point to a completed training-side selection JSON.")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("phase") != "selection" or selection.get("seed") != args.seed:
        raise ValueError("Selection metadata does not match the requested test seed.")
    if selection.get("data_scope") != "omniglot_background_meta_training_classes_only":
        raise ValueError("Refusing selection that was not produced from meta-training classes only.")
    if os.path.realpath(selection.get("model", "")) != os.path.realpath(args.model):
        raise ValueError("Selection JSON was produced for a different checkpoint.")
    learning_rate = float(selection["selected_learning_rate"])
    train = df.DatasetFactory.get_dataset("omniglot", train=True, background=False,
                                          path=args.dataset_path)
    test = df.DatasetFactory.get_dataset("omniglot", train=False, background=False,
                                         path=args.dataset_path)
    available = int(max(train.targets)) + 1
    order = class_order(available, args.seed)
    matrix, events = run_stream(checkpoint, train, test, order, learning_rate,
                                args.epochs, device, args.eval_batch_size,
                                args.num_workers)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{args.run_name}_R_matrix.csv"
    metrics_path = output_dir / f"{args.run_name}_metrics.json"
    write_matrix(matrix_path, matrix, order)
    payload = {
        "schema_version": 1,
        "run_name": args.run_name,
        "dataset": "Omniglot-1",
        "seed": args.seed,
        "model": os.path.abspath(args.model),
        "selection": str(args.selection.resolve()),
        "selected_learning_rate": learning_rate,
        "tasks": len(order),
        "classes_per_task": 1,
        "support_samples_per_class": 15,
        "query_samples_per_class": 5,
        "prediction_class_scope": "seen",
        "adaptation_scope": "output-only",
        "class_order": order,
        "events": events,
        "final": events[-1],
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"matrix": str(matrix_path.resolve()), "metrics": str(metrics_path.resolve())}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["select", "test"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", default="../data/omni")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr-candidates", type=float, nargs="+", default=[0.001, 0.0005, 0.0002, 0.0001])
    parser.add_argument("--selection-tasks", type=int, default=50)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--run-name", default="omniglot_fseml_seed9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = load_model(args.model, device)
    if args.phase == "select":
        select_learning_rate(args, device, checkpoint)
    else:
        run_test(args, device, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
