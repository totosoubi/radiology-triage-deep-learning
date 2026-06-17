from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from . import CHEST_LABELS


def image_transform(image_size: int, in_channels: int, augment: bool = False) -> transforms.Compose:
    if in_channels == 1:
        norm_mean, norm_std = (0.5,), (0.5,)
        channel_tf = transforms.Grayscale(num_output_channels=1)
    else:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        channel_tf = transforms.Grayscale(num_output_channels=3)
    steps = [transforms.Resize((image_size, image_size)), channel_tf]
    if augment:
        steps.extend(
            [
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    return transforms.Compose(steps)


def load_chestmnist(split: str, image_size: int, in_channels: int, root: str = "data", augment: bool = False) -> Dataset:
    from medmnist import ChestMNIST

    Path(root).mkdir(parents=True, exist_ok=True)
    return ChestMNIST(
        split=split,
        transform=image_transform(image_size, in_channels, augment=augment and split == "train"),
        download=True,
        root=root,
    )


def limit_dataset(dataset: Dataset, max_items: int | None, seed: int) -> Dataset:
    if not max_items or max_items >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_items, replace=False).tolist()
    return Subset(dataset, indices)


class NormalOnlyDataset(Dataset):
    """Keeps ChestMNIST samples without positive labels for AE training."""

    def __init__(self, base: Dataset, max_items: int | None = None):
        self.base = base
        indices: list[int] = []
        for i in range(len(base)):
            _, label = base[i]
            if float(torch.as_tensor(label).sum()) == 0.0:
                indices.append(i)
            if max_items and len(indices) >= max_items:
                break
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, _ = self.base[self.indices[idx]]
        return x, x


@dataclass
class OpenIRecord:
    image_path: Path
    report: str
    labels: np.ndarray


class OpenIManifestDataset(Dataset):
    """Manifest format: image_path, report, then the 14 ChestMNIST label columns."""

    def __init__(self, manifest: str | Path, image_size: int, in_channels: int, max_tokens: int = 96):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.frame = pd.read_csv(self.manifest)
        missing = [c for c in ["image_path", "report", *CHEST_LABELS] if c not in self.frame.columns]
        if missing:
            raise ValueError(f"Missing columns in multimodal manifest: {missing}")
        self.transform = image_transform(image_size, in_channels)
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = self.root / img_path
        image = Image.open(img_path).convert("L")
        labels = row[CHEST_LABELS].astype("float32").to_numpy()
        return self.transform(image), str(row["report"]), torch.from_numpy(labels)


class SyntheticMultimodalDataset(Dataset):
    """Small deterministic image+text dataset used only for smoke tests."""

    def __init__(self, n: int, image_size: int, in_channels: int, seed: int):
        rng = np.random.default_rng(seed)
        self.images = rng.normal(0, 1, size=(n, in_channels, image_size, image_size)).astype("float32")
        labels = rng.binomial(1, 0.18, size=(n, len(CHEST_LABELS))).astype("float32")
        self.labels = labels
        self.reports = []
        for y in labels:
            positives = [name for name, value in zip(CHEST_LABELS, y) if value > 0]
            if positives:
                self.reports.append("findings suggest " + " ".join(positives[:3]))
            else:
                self.reports.append("no acute cardiopulmonary abnormality")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), self.reports[idx], torch.from_numpy(self.labels[idx])


class TextVocab:
    def __init__(self, texts: list[str], max_vocab: int = 2048):
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        words = sorted(counts, key=counts.get, reverse=True)[: max_vocab - 2]
        self.stoi = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.strip(".,;:!?()[]").lower() for t in text.split() if t.strip()]

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.stoi.get(t, 1) for t in self.tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def multimodal_collate(vocab: TextVocab, max_len: int):
    def _collate(batch):
        images, texts, labels = zip(*batch)
        return (
            torch.stack(list(images)),
            torch.stack([vocab.encode(t, max_len) for t in texts]),
            torch.stack(list(labels)).float(),
        )

    return _collate
