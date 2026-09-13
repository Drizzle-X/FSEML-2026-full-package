from __future__ import annotations

import copy
import random
from collections import defaultdict, deque

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.clinc150_continual import EmbeddingDataset
from model.vector_fseml import VectorFSEML, VectorFSSAEConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, allowed_classes) -> torch.Tensor:
    allowed = torch.as_tensor(sorted(set(allowed_classes)), dtype=torch.long, device=logits.device)
    mapping = torch.full((logits.shape[1],), -1, dtype=torch.long, device=logits.device)
    mapping[allowed] = torch.arange(len(allowed), device=logits.device)
    mapped = mapping[labels]
    if (mapped < 0).any():
        raise ValueError("A target label is absent from the allowed prediction classes")
    return F.cross_entropy(logits.index_select(1, allowed), mapped)


def classifier_accuracy(logits: torch.Tensor, labels: torch.Tensor, allowed_classes) -> float:
    allowed = torch.as_tensor(sorted(set(allowed_classes)), dtype=torch.long, device=logits.device)
    predictions = allowed[logits.index_select(1, allowed).argmax(1)]
    return float(predictions.eq(labels).float().mean().item())


def model_config_from_args(args, input_dim: int, num_classes: int = 150):
    return VectorFSSAEConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        classifier_hidden=args.classifier_hidden,
        num_classes=num_classes,
        sparsity_target=args.sparsity_target,
        sparsity_weight=args.sparsity_weight,
        reconstruction_weight=args.reconstruction_weight,
        fisher_weight=args.fisher_weight,
    )


def model_from_checkpoint(payload: dict, device: torch.device):
    config = VectorFSSAEConfig(**payload["model_config"])
    model = VectorFSEML(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model


class ClassBalancedLatentReplay:
    def __init__(self, capacity: int, class_ids):
        self.capacity = int(capacity)
        self.class_ids = sorted(set(int(value) for value in class_ids))
        self.partition_capacity = max(1, self.capacity // max(1, len(self.class_ids)))
        self.storage = {label: deque(maxlen=self.partition_capacity) for label in self.class_ids}
        self.samples_since_replay = 0
        self.replay_events = 0
        self.total_replayed_samples = 0

    def __len__(self):
        return sum(len(values) for values in self.storage.values())

    def add(self, latent: torch.Tensor, labels: torch.Tensor):
        for value, label in zip(latent.detach().cpu(), labels.detach().cpu().tolist()):
            if int(label) in self.storage:
                self.storage[int(label)].append(value.clone())
        self.samples_since_replay += len(labels)

    def ready(self, gap: int, warmup: int):
        return len(self) > 0 and len(self) >= warmup and self.samples_since_replay >= gap

    def sample(self, count: int, rng: random.Random):
        populated = [label for label, values in self.storage.items() if values]
        if not populated:
            return None
        chosen = []
        for index in range(count):
            label = populated[index % len(populated)]
            values = self.storage[label]
            chosen.append((values[rng.randrange(len(values))], label))
        rng.shuffle(chosen)
        self.replay_events += 1
        self.total_replayed_samples += len(chosen)
        return torch.stack([item[0] for item in chosen]), torch.tensor([item[1] for item in chosen])

    def consume_gap(self, gap: int):
        self.samples_since_replay %= max(1, gap)

    def audit(self):
        return {
            "buffer_length": len(self),
            "buffer_capacity": self.capacity,
            "partition_capacity": self.partition_capacity,
            "label_occupancy": {str(label): len(values) for label, values in self.storage.items() if values},
            "samples_since_replay": self.samples_since_replay,
            "replay_events": self.replay_events,
            "total_replayed_samples": self.total_replayed_samples,
        }


def _evaluate_task(model, feature_path, feature_payload, split, task_intents, allowed, batch_size, device):
    dataset = EmbeddingDataset(feature_path, split, task_intents, payload=feature_payload)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            allowed_tensor = torch.as_tensor(sorted(allowed), dtype=torch.long, device=device)
            predictions = allowed_tensor[logits.index_select(1, allowed_tensor).argmax(1)]
            correct += predictions.eq(y).sum().item()
            total += len(y)
    return correct / max(1, total)


def continual_metrics(matrix: np.ndarray):
    task_count = matrix.shape[0]
    final_acc = float(np.nanmean(matrix[-1]))
    learning_acc = float(np.nanmean(np.diag(matrix)))
    if task_count <= 1:
        forgetting = 0.0
    else:
        forgetting = float(
            np.mean([np.nanmax(matrix[column : task_count - 1, column]) - matrix[-1, column] for column in range(task_count - 1)])
        )
    return {"final_ACC": final_acc, "learning_ACC": learning_acc, "forgetting": forgetting}


def run_continual_evaluation(
    model: VectorFSEML,
    feature_path,
    feature_payload,
    protocol,
    phase: str,
    online_lr: float,
    adaptation_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
    classifier_reset_init: str = "zero",
    prediction_class_scope: str = "task",
):
    if phase == "validation":
        tasks, eval_split = protocol.validation_tasks, "validation"
    elif phase == "test":
        tasks, eval_split = protocol.test_tasks, "test"
    else:
        raise ValueError(f"Unknown held-out phase: {phase}")

    candidate = copy.deepcopy(model).to(device)
    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=float)
    seen: set[int] = set()
    task_events = []
    generator = torch.Generator().manual_seed(seed)

    for task_index, intents in enumerate(tasks):
        current = {protocol.intent_to_id[intent] for intent in intents}
        seen.update(current)
        candidate.reset_classifier_rows(current, initialization=classifier_reset_init)
        optimizer = torch.optim.SGD(candidate.cpn_parameters(), lr=online_lr)
        train_data = EmbeddingDataset(feature_path, "train", intents, payload=feature_payload)
        for _ in range(adaptation_epochs):
            loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                generator=generator,
            )
            candidate.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                adaptation_classes = current if prediction_class_scope == "task" else seen
                loss = masked_loss(candidate(x)["logits"], y, adaptation_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for eval_index in range(task_index + 1):
            prediction_classes = (
                {protocol.intent_to_id[intent] for intent in tasks[eval_index]}
                if prediction_class_scope == "task"
                else seen
            )
            matrix[task_index, eval_index] = _evaluate_task(
                candidate,
                feature_path,
                feature_payload,
                eval_split,
                tasks[eval_index],
                prediction_classes,
                eval_batch_size,
                device,
            )
        partial = continual_metrics(matrix[: task_index + 1, : task_index + 1])
        task_events.append({"task": task_index, "intents": intents, **partial})

    result = {
        "phase": phase,
        "online_lr": online_lr,
        "prediction_class_scope": prediction_class_scope,
        "matrix": matrix.tolist(),
        "task_events": task_events,
        **continual_metrics(matrix),
    }
    return result


def select_validation_candidate(model, online_lrs, **kwargs):
    results = []
    for online_lr in online_lrs:
        set_seed(kwargs["seed"])
        results.append(run_continual_evaluation(model, online_lr=online_lr, phase="validation", **kwargs))
    best = max(results, key=lambda item: (item["final_ACC"], item["learning_ACC"], -item["forgetting"]))
    return {"candidates": results, "selected_online_lr": best["online_lr"], "validation_ACC": best["final_ACC"], "best": best}
