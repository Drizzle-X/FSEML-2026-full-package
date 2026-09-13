from __future__ import annotations

import pickle
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


NBATCH = {"nc": 9, "nicv2_79": 79, "nicv2_196": 196, "nicv2_391": 391}


class CORe50Metadata:
    """Thin PyTorch view over the official CORe50 LUP/labels protocol."""

    def __init__(self, image_root: str, metadata_root: str, scenario: str, run: int):
        if scenario not in NBATCH:
            raise ValueError(f"Unsupported scenario: {scenario}")
        self.image_root = Path(image_root)
        self.scenario = scenario
        self.run = run
        root = Path(metadata_root)
        with (root / "paths.pkl").open("rb") as handle: self.paths = pickle.load(handle)
        with (root / "LUP.pkl").open("rb") as handle: self.lookup = pickle.load(handle)
        with (root / "labels.pkl").open("rb") as handle: self.labels = pickle.load(handle)
        if run not in range(10): raise ValueError("Official CORe50 run must be in [0, 9].")

    @property
    def num_batches(self): return NBATCH[self.scenario]

    def batch(self, batch_index: int, transform=None):
        return CORe50PathDataset(
            self.image_root, self.paths, self.lookup[self.scenario][self.run][batch_index],
            self.labels[self.scenario][self.run][batch_index], transform,
        )

    def test(self, transform=None):
        return CORe50PathDataset(
            self.image_root, self.paths, self.lookup[self.scenario][self.run][-1],
            self.labels[self.scenario][self.run][-1], transform,
        )


class CORe50PathDataset(Dataset):
    def __init__(self, root, paths, indices, labels, transform=None):
        self.root = Path(root); self.paths = paths; self.indices = list(indices)
        self.labels = [int(value) for value in labels]
        self.transform = transform or core50_eval_transform()
        if len(self.indices) != len(self.labels): raise ValueError("CORe50 index/label mismatch")

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        image = Image.open(self.root / self.paths[self.indices[index]]).convert("RGB")
        return self.transform(image), self.labels[index]


def core50_train_transform(image_size=128):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def core50_eval_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
