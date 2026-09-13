from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms


CIFAR100_NUM_CLASSES = 100
CIFAR100_TRAIN_SAMPLES_PER_CLASS = 500
CIFAR100_TEST_SAMPLES_PER_CLASS = 100
CIFAR100_MODEL_MEAN = (0.5, 0.5, 0.5)
CIFAR100_MODEL_STD = (0.5, 0.5, 0.5)


def cifar100_train_transform(image_size: int = 28, augment: bool = True):
    operations = []
    if augment:
        operations.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    if image_size != 32:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MODEL_MEAN, CIFAR100_MODEL_STD),
    ])
    return transforms.Compose(operations)


def cifar100_eval_transform(image_size: int = 28):
    operations = []
    if image_size != 32:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MODEL_MEAN, CIFAR100_MODEL_STD),
    ])
    return transforms.Compose(operations)


def cifar100_model_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    """Invert the model-space normalization and return RGB pixels in [0, 1]."""
    mean = images.new_tensor(CIFAR100_MODEL_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR100_MODEL_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@dataclass(frozen=True)
class CIFAR100Task:
    task_id: int
    class_ids: List[int]
    split: str


@dataclass(frozen=True)
class CIFAR100Protocol:
    classes_per_task: int
    seed: int
    meta_train_fraction: float
    class_order: List[int]
    tasks: List[CIFAR100Task]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def meta_train_tasks(self) -> List[CIFAR100Task]:
        return [task for task in self.tasks if task.split == "meta_train"]

    @property
    def meta_test_tasks(self) -> List[CIFAR100Task]:
        return [task for task in self.tasks if task.split == "meta_test"]

    def task(self, task_id: int) -> CIFAR100Task:
        return self.tasks[task_id]

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload.update({
            "dataset": "CIFAR100",
            "num_classes": CIFAR100_NUM_CLASSES,
            "official_train_samples_per_class": CIFAR100_TRAIN_SAMPLES_PER_CLASS,
            "official_test_samples_per_class": CIFAR100_TEST_SAMPLES_PER_CLASS,
            "num_tasks": self.num_tasks,
            "num_meta_train_tasks": len(self.meta_train_tasks),
            "num_meta_test_tasks": len(self.meta_test_tasks),
            "model_image_size": 28,
            "model_normalization_mean": list(CIFAR100_MODEL_MEAN),
            "model_normalization_std": list(CIFAR100_MODEL_STD),
        })
        return payload

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def load_cifar100_protocol(path: str) -> CIFAR100Protocol:
    with open(path) as handle:
        payload = json.load(handle)
    tasks = [CIFAR100Task(**task) for task in payload["tasks"]]
    protocol = CIFAR100Protocol(
        classes_per_task=int(payload["classes_per_task"]),
        seed=int(payload["seed"]),
        meta_train_fraction=float(payload["meta_train_fraction"]),
        class_order=[int(value) for value in payload["class_order"]],
        tasks=tasks,
    )
    expected = build_cifar100_protocol(
        protocol.classes_per_task,
        protocol.seed,
        protocol.meta_train_fraction,
    )
    if protocol != expected:
        raise ValueError(f"CIFAR100 protocol does not match its recorded seed/configuration: {path}")
    return protocol


def build_cifar100_protocol(
    classes_per_task: int,
    seed: int = 9,
    meta_train_fraction: float = 0.60,
) -> CIFAR100Protocol:
    if classes_per_task not in (5, 10, 20):
        raise ValueError("classes_per_task must be one of 5, 10, or 20.")
    if not 0.0 < meta_train_fraction < 1.0:
        raise ValueError("meta_train_fraction must lie strictly between 0 and 1.")

    class_order = np.random.RandomState(seed).permutation(CIFAR100_NUM_CLASSES).tolist()
    class_batches = [
        class_order[start:start + classes_per_task]
        for start in range(0, CIFAR100_NUM_CLASSES, classes_per_task)
    ]
    num_meta_train_tasks = int(round(len(class_batches) * meta_train_fraction))
    tasks = [
        CIFAR100Task(
            task_id=task_id,
            class_ids=class_ids,
            split="meta_train" if task_id < num_meta_train_tasks else "meta_test",
        )
        for task_id, class_ids in enumerate(class_batches)
    ]
    return CIFAR100Protocol(
        classes_per_task=classes_per_task,
        seed=seed,
        meta_train_fraction=meta_train_fraction,
        class_order=class_order,
        tasks=tasks,
    )


def load_cifar100(
    root: str,
    train: bool,
    image_size: int = 28,
    augment: bool = False,
    download_if_missing: bool = True,
) -> datasets.CIFAR100:
    """Load local CIFAR100 first and use torchvision's official URL only if absent.

    ``torchvision.datasets.CIFAR100`` performs MD5 integrity checks for both
    the archive and extracted files.  Trying ``download=False`` first makes
    the local-first behavior explicit and prevents unnecessary network use.
    """
    transform = (
        cifar100_train_transform(image_size, augment=augment)
        if train else cifar100_eval_transform(image_size)
    )
    root = os.path.abspath(os.path.expanduser(root))
    try:
        dataset = datasets.CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=False,
        )
        print(f"CIFAR100 found locally: {root}")
        return dataset
    except RuntimeError as error:
        if not download_if_missing:
            raise RuntimeError(
                f"CIFAR100 was not found or failed integrity checks at {root}."
            ) from error
        print(
            f"CIFAR100 was not found at {root}; downloading it through "
            "torchvision from the official CIFAR source."
        )
        return datasets.CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=True,
        )


def class_subset(dataset: Dataset, class_ids: Iterable[int]) -> Subset:
    keep = set(int(class_id) for class_id in class_ids)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError("CIFAR100 dataset must expose a targets attribute.")
    indices = [index for index, label in enumerate(targets) if int(label) in keep]
    return Subset(dataset, indices)


class BalancedClassSampler(Sampler[int]):
    """Maximize class coverage before repeating any class."""

    def __init__(self, dataset: Subset):
        if not isinstance(dataset, Subset):
            raise TypeError("BalancedClassSampler expects a torch Subset.")
        targets = getattr(dataset.dataset, "targets", None)
        if targets is None:
            raise TypeError("The underlying CIFAR100 dataset must expose targets.")
        self.indices_by_class: Dict[int, List[int]] = {}
        for local_index, source_index in enumerate(dataset.indices):
            label = int(targets[source_index])
            self.indices_by_class.setdefault(label, []).append(local_index)
        sizes = {len(indices) for indices in self.indices_by_class.values()}
        if len(sizes) != 1:
            raise ValueError("Balanced sampling requires equal samples per class.")

    def __iter__(self) -> Iterator[int]:
        shuffled = {
            label: np.random.permutation(indices).tolist()
            for label, indices in self.indices_by_class.items()
        }
        labels = sorted(shuffled)
        samples_per_class = len(next(iter(shuffled.values())))
        for sample_index in range(samples_per_class):


            for label in np.random.permutation(labels):
                yield shuffled[int(label)][sample_index]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.indices_by_class.values())


class CIFAR100ContinualData:
    """Paper-aligned CIFAR100 task views with immutable global class labels."""

    def __init__(
        self,
        root: str,
        protocol: CIFAR100Protocol,
        image_size: int = 28,
        augment_train: bool = True,
        download_if_missing: bool = True,
    ):
        self.protocol = protocol
        self.train_dataset = load_cifar100(
            root, train=True, image_size=image_size,
            augment=augment_train, download_if_missing=download_if_missing,
        )
        self.train_eval_dataset = load_cifar100(
            root, train=True, image_size=image_size,
            augment=False, download_if_missing=download_if_missing,
        )
        self.test_dataset = load_cifar100(
            root, train=False, image_size=image_size,
            augment=False, download_if_missing=download_if_missing,
        )

    def task_dataset(self, task_id: int, split: str = "train") -> Subset:
        task = self.protocol.task(task_id)
        sources = {
            "train": self.train_dataset,
            "train_eval": self.train_eval_dataset,
            "test": self.test_dataset,
        }
        if split not in sources:
            raise ValueError("split must be 'train', 'train_eval', or 'test'.")
        return class_subset(sources[split], task.class_ids)

    def combined_dataset(self, task_ids: Sequence[int], split: str = "train") -> Subset:
        class_ids = [
            class_id
            for task_id in task_ids
            for class_id in self.protocol.task(task_id).class_ids
        ]
        sources = {
            "train": self.train_dataset,
            "train_eval": self.train_eval_dataset,
            "test": self.test_dataset,
        }
        if split not in sources:
            raise ValueError("split must be 'train', 'train_eval', or 'test'.")
        return class_subset(sources[split], class_ids)

    def loader(
        self,
        task_id: int,
        split: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self.task_dataset(task_id, split),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )


class CIFAR100TaskSampler:
    """Adapter exposing the iterator interface expected by the meta-learner."""

    def __init__(
        self,
        continual_data: CIFAR100ContinualData,
        task_ids: Sequence[int],
        num_workers: int = 0,
        support_sampling: str = "random",
    ):
        self.data = continual_data
        self.task_ids = [int(task_id) for task_id in task_ids]
        self.num_workers = num_workers
        self.support_sampling = support_sampling
        self.task_iterators = {}
        self.query_iterators = {}
        combined = self.data.combined_dataset(self.task_ids, split="train")
        self.complete_iterator = DataLoader(
            combined,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_complete_iterator(self):
        return self.complete_iterator

    def get_query_iterator(self, task_ids: Sequence[int]):
        """Return a shuffled query loader restricted to the requested tasks."""
        normalized_ids = tuple(int(task_id) for task_id in task_ids)
        if not normalized_ids:
            raise ValueError("At least one task id is required for a task-scoped query.")
        invalid_ids = [task_id for task_id in normalized_ids if task_id not in self.task_ids]
        if invalid_ids:
            raise ValueError(
                f"Query tasks {invalid_ids} are not part of the selected meta-training split."
            )
        if normalized_ids not in self.query_iterators:
            combined = self.data.combined_dataset(normalized_ids, split="train")
            self.query_iterators[normalized_ids] = DataLoader(
                combined,
                batch_size=64,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        return self.query_iterators[normalized_ids]

    def sample_task(self, tasks, train=True):
        if len(tasks) != 1:
            raise ValueError("CIFAR100TaskSampler.sample_task expects exactly one task id.")
        task_id = int(tasks[0])
        if task_id not in self.task_ids:
            raise ValueError(f"Task {task_id} is not part of the selected meta-training split.")
        key = (task_id, bool(train))
        if key not in self.task_iterators:
            split = "train" if train else "test"
            if train and self.support_sampling == "balanced":
                dataset = self.data.task_dataset(task_id, split=split)
                self.task_iterators[key] = DataLoader(
                    dataset,
                    batch_size=1,
                    sampler=BalancedClassSampler(dataset),
                    num_workers=self.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                self.task_iterators[key] = self.data.loader(
                    task_id,
                    split=split,
                    batch_size=1,
                    shuffle=train,
                    num_workers=self.num_workers,
                )
        return self.task_iterators[key]
