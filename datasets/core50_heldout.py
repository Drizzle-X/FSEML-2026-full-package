from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from datasets.core50_continual import core50_eval_transform, core50_train_transform


TRAIN_SUPPORT_SESSIONS = {1, 2, 4, 5}
TRAIN_QUERY_SESSIONS = {6, 8, 9, 11}
OFFICIAL_TEST_SESSIONS = {3, 7, 10}


def load_split(path):
    payload = json.loads(Path(path).read_text())
    for key in ("meta_train_classes", "meta_validation_classes", "meta_test_classes"):
        payload[key] = [int(label) for label in payload[key]]
    return payload


class IndexedImages(Dataset):
    def __init__(self, image_root, paths, indices, labels, transform=None):
        self.image_root = Path(image_root)
        self.paths = paths
        self.indices = list(indices)
        self.labels = [int(label) for label in labels]
        self.transform = transform or core50_eval_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image = Image.open(self.image_root / self.paths[self.indices[item]]).convert("RGB")
        return self.transform(image), self.labels[item]


class CORe50HeldOutData:
    """Session-aware pools built from the official CORe50 NC train/test split."""

    def __init__(self, image_root, metadata_root, run=0, image_size=128):
        self.image_root = Path(image_root)
        self.image_size = image_size
        root = Path(metadata_root)
        with (root / "paths.pkl").open("rb") as handle:
            self.paths = pickle.load(handle)
        with (root / "LUP.pkl").open("rb") as handle:
            lookup = pickle.load(handle)["nc"][run]
        with (root / "labels.pkl").open("rb") as handle:
            labels = pickle.load(handle)["nc"][run]

        self.train_by_class_session = self._pool(lookup[:-1], labels[:-1])
        self.test_by_class_session = self._pool([lookup[-1]], [labels[-1]])

    def _pool(self, index_batches, label_batches):
        pool = {label: {} for label in range(50)}
        for indices, labels in zip(index_batches, label_batches):
            for index, label in zip(indices, labels):
                session = int(re.match(r"s(\d+)/", self.paths[index]).group(1))
                pool[int(label)].setdefault(session, []).append(int(index))
        return pool

    @staticmethod
    def _sample(pool, classes, sessions, per_class, rng):
        indices, labels = [], []
        for label in classes:
            candidates = [
                index for session in sessions for index in pool[int(label)].get(session, [])
            ]
            if len(candidates) < per_class:
                raise ValueError(f"Class {label} has only {len(candidates)} eligible samples.")
            selected = rng.sample(candidates, per_class)
            indices.extend(selected)
            labels.extend([int(label)] * per_class)
        order = list(range(len(indices)))
        rng.shuffle(order)
        return [indices[i] for i in order], [labels[i] for i in order]

    def episode(self, classes, support_shots, query_shots, rng):
        support = self._sample(
            self.train_by_class_session, classes, TRAIN_SUPPORT_SESSIONS, support_shots, rng
        )
        query = self._sample(
            self.train_by_class_session, classes, TRAIN_QUERY_SESSIONS, query_shots, rng
        )
        return (
            IndexedImages(self.image_root, self.paths, *support, core50_train_transform(self.image_size)),
            IndexedImages(self.image_root, self.paths, *query, core50_train_transform(self.image_size)),
        )

    def adaptation_set(self, classes, shots, seed, train=True):
        rng = random.Random(seed)
        sampled = self._sample(
            self.train_by_class_session,
            classes,
            TRAIN_SUPPORT_SESSIONS | TRAIN_QUERY_SESSIONS,
            shots,
            rng,
        )
        transform = core50_train_transform(self.image_size) if train else core50_eval_transform(self.image_size)
        return IndexedImages(self.image_root, self.paths, *sampled, transform)

    def official_test_set(self, classes):
        indices, labels = [], []
        for label in classes:
            for session in OFFICIAL_TEST_SESSIONS:
                selected = self.test_by_class_session[int(label)].get(session, [])
                indices.extend(selected)
                labels.extend([int(label)] * len(selected))
        return IndexedImages(
            self.image_root, self.paths, indices, labels, core50_eval_transform(self.image_size)
        )
