from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CLINCProtocol:
    meta_train_domains: list[str]
    meta_validation_domains: list[str]
    meta_test_domains: list[str]
    validation_tasks: list[list[str]]
    test_tasks: list[list[str]]
    intent_to_id: dict[str, int]
    intent_to_domain: dict[str, str]
    seed: int

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text())
        required = {
            "meta_train_domains",
            "meta_validation_domains",
            "meta_test_domains",
            "validation_tasks",
            "test_tasks",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                f"Legacy or incomplete CLINC150 protocol {path}; missing {missing}. "
                "Regenerate it with prepare_clinc150_protocol.py."
            )
        protocol = cls(**payload)
        protocol.validate()
        return protocol

    def validate(self):
        train_domains = set(self.meta_train_domains)
        test_domains = set(self.meta_test_domains)
        if self.meta_validation_domains:
            raise ValueError("The 6/4 paper protocol does not use separate meta-validation domains")
        if len(train_domains) != 6 or len(test_domains) != 4 or train_domains & test_domains:
            raise ValueError("CLINC150 paper protocol requires disjoint 6/4 train/test domains")
        validation_intents = {intent for task in self.validation_tasks for intent in task}
        test_intents = {intent for task in self.test_tasks for intent in task}
        train_intents = {intent for intent, domain in self.intent_to_domain.items() if domain in train_domains}
        expected_test_intents = {intent for intent, domain in self.intent_to_domain.items() if domain in test_domains}
        if validation_intents != train_intents:
            raise ValueError("Training-side validation tasks must cover the 90 meta-training intents")
        if test_intents != expected_test_intents or validation_intents & test_intents:
            raise ValueError("Meta-test tasks must cover exactly 60 unseen intents without leakage")
        if len(self.validation_tasks) != 18 or len(self.test_tasks) != 12:
            raise ValueError("Expected 18 training-side validation tasks and 12 meta-test tasks")
        if any(len(task) != 5 for task in self.validation_tasks + self.test_tasks):
            raise ValueError("Every held-out continual task must contain exactly five intents")


def _make_tasks(domains: dict[str, list[str]], selected: list[str], rng: random.Random):
    ordered_domains = list(selected)
    rng.shuffle(ordered_domains)
    tasks: list[list[str]] = []
    for domain in ordered_domains:
        values = list(domains[domain])
        rng.shuffle(values)
        tasks.extend([values[start : start + 5] for start in range(0, len(values), 5)])
    return tasks


def build_protocol(domains_path: str | Path, seed: int = 9) -> CLINCProtocol:
    """Build the manuscript's domain-disjoint 6/4 protocol.

    The split itself is fixed across random seeds. ``seed`` controls only the
    domain/task order and the intent order inside each held-out domain.
    """
    domains = json.loads(Path(domains_path).read_text())
    test_domains = ["banking", "credit_cards", "travel", "home"]
    train_domains = [name for name in domains if name not in set(test_domains)]
    if len(domains) != 10 or len(train_domains) != 6:
        raise ValueError(f"Expected the official 10-domain CLINC150 taxonomy, found {len(domains)} domains")
    if any(len(values) != 15 for values in domains.values()):
        raise ValueError("Expected exactly 15 intents in every CLINC150 domain")

    intents = sorted(intent for values in domains.values() for intent in values)
    intent_to_id = {name: index for index, name in enumerate(intents)}
    intent_to_domain = {intent: domain for domain, values in domains.items() for intent in values}
    validation_tasks = _make_tasks(domains, train_domains, random.Random(seed + 1000))
    test_tasks = _make_tasks(domains, test_domains, random.Random(seed + 2000))
    protocol = CLINCProtocol(
        meta_train_domains=train_domains,
        meta_validation_domains=[],
        meta_test_domains=test_domains,
        validation_tasks=validation_tasks,
        test_tasks=test_tasks,
        intent_to_id=intent_to_id,
        intent_to_domain=intent_to_domain,
        seed=seed,
    )
    protocol.validate()
    return protocol


def save_protocol(protocol: CLINCProtocol, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(protocol.__dict__, indent=2, sort_keys=True))


def load_feature_payload(feature_path: str | Path):
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    required = {"embeddings", "rows", "intent_to_id", "encoder", "paper_valid"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Embedding payload is missing fields: {missing}")
    if not payload["paper_valid"]:
        raise ValueError("Hash/smoke embeddings cannot be used for a paper experiment")
    if len(payload["rows"]) != len(payload["embeddings"]):
        raise ValueError("Embedding/metadata row count mismatch")
    return payload


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        feature_path: str | Path,
        split: str,
        intents: list[str] | None = None,
        payload: dict | None = None,
    ):
        payload = payload if payload is not None else load_feature_payload(feature_path)
        allowed = set(intents) if intents else None
        rows = [row for row in payload["rows"] if row["split"] == split and row["intent"] != "oos"]
        if allowed is not None:
            rows = [row for row in rows if row["intent"] in allowed]
        if not rows:
            raise ValueError(f"No CLINC150 rows for split={split!r}, intents={intents!r}")
        self.x = torch.stack([payload["embeddings"][row["index"]] for row in rows]).float()
        self.y = torch.tensor([payload["intent_to_id"][row["intent"]] for row in rows], dtype=torch.long)
        self.intents = [row["intent"] for row in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
