# Vendored from this project's lhfm-moving-mnist-video-fast-v2/lhfm_video_fast/data.py.
# Source SHA256: 40ebaf61eedddbcabb79656d675d6fa9285888db315c78728a6f22b0407c56ef
# Intentional behavior change: normalize uint8 clips to [0, 1] with uint8 / 255;
# the official-test manifest records the same normalization. No legacy imports.

"""Deterministic training clips and read-only official Moving MNIST test clips.

The official mnist_test_seq.npy is test data only. Training uses MNIST training
digits with a disjoint held-out digit pool for validation. No implicit downloads.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mnist_training_digits(path):
    """Require the standard 60,000-image MNIST training IDX, raw or gzipped."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < 16 or struct.unpack(">IIII", payload[:16]) != (2051, 60000, 28, 28):
        raise ValueError("expected the 60000-image MNIST training IDX file")
    if len(payload) != 16 + 60000 * 28 * 28:
        raise ValueError("truncated or oversized MNIST IDX")
    return np.frombuffer(payload, np.uint8, offset=16).reshape(60000, 28, 28).copy()


def digit_split(count=60000, validation_count=5000, seed=271100):
    if not 0 < validation_count < count:
        raise ValueError("validation pool must be nonempty and smaller than the digit pool")
    order = np.random.default_rng(seed).permutation(count)
    return order[validation_count:], order[:validation_count]


def normalize_clip(frames):
    if frames.dtype != np.uint8 or frames.ndim != 3:
        raise ValueError("frames must be uint8 [T,H,W]")
    return torch.from_numpy(frames.copy()).float().unsqueeze(1).div_(255)


class GeneratedMovingMNIST(Dataset):
    """Stateless sequence generation: seed + absolute sequence index determines a clip.

    Fresh training epochs must use a new sequence_start (or new absolute indices),
    not repeat the same finite epoch unintentionally. No worker-global RNG is used.
    digit_indices must be the training pool or the held-out validation pool.
    """
    def __init__(self, digits, digit_indices, *, length=60000, sequence_start=0, seed=270829,
                 context_frames=10, future_frames=10, image_size=64, speed=.1,
                 source_id="unverified_digit_array"):
        if (digits.ndim != 3 or digits.shape[1:] != (28, 28) or digits.dtype != np.uint8
                or length < 1 or sequence_start < 0 or seed < 0 or image_size <= 28
                or context_frames < 1 or future_frames < 1 or not 0 < speed <= 1):
            raise ValueError("invalid digit pool or sequence parameters")
        ids = np.asarray(digit_indices)
        if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                or ids.min() < 0 or ids.max() >= len(digits) or len(np.unique(ids)) != len(ids)):
            raise ValueError("digit indices must be a nonempty unique subset")
        self.digits = digits
        self.ids = ids.copy()
        self.length, self.sequence_start, self.seed = length, sequence_start, seed
        self.context_frames, self.future_frames = context_frames, future_frames
        self.image_size, self.speed = image_size, speed
        self.source_id = str(source_id)
        self.source_manifest = None

    @classmethod
    def from_training_idx(cls, path, *, split="train", validation_count=5000,
                          split_seed=271100, **kwargs):
        if split not in ("train", "validation") or "source_id" in kwargs:
            raise ValueError("expected train/validation split with fixed source identity")
        digits = read_mnist_training_digits(path)
        train, validation = digit_split(len(digits), validation_count, split_seed)
        ids = train if split == "train" else validation
        dataset = cls(digits, ids, source_id=f"mnist_training_idx_{split}", **kwargs)
        dataset.source_manifest = {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
            "digit_pool_split": split, "validation_count": validation_count, "split_seed": split_seed,
            "digit_indices_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()}
        return dataset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        sequence_id = self.sequence_start + int(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, sequence_id]))
        count = self.context_frames + self.future_frames
        positions = rng.uniform(0, 1, (2, 2))
        angles = rng.uniform(0, 2 * np.pi, 2)
        velocity = self.speed * np.stack((np.cos(angles), np.sin(angles)), -1)
        digit_ids = rng.choice(self.ids, 2, replace=True)
        frames = np.zeros((count, self.image_size, self.image_size), dtype=np.uint8)
        xy = np.empty((count, 2, 2), dtype=np.int64)
        for frame in range(count):
            positions += velocity
            lower, upper = positions <= 0, positions >= 1
            velocity[lower | upper] *= -1
            positions = np.clip(positions, 0, 1)
            xy[frame] = np.floor(positions * (self.image_size - 28)).astype(np.int64)
            for digit in range(2):
                x, y = xy[frame, digit]
                patch = frames[frame, y:y + 28, x:x + 28]
                np.maximum(patch, self.digits[digit_ids[digit]], out=patch)
        clip = normalize_clip(frames)
        return {"context": clip[:self.context_frames], "future": clip[self.context_frames:],
                "sequence_id": sequence_id, "digit_ids": torch.tensor(digit_ids.copy()),
                "top_left_xy_pixels": torch.tensor(xy), "source": self.source_id}


class FixedMovingMNISTTest(Dataset):
    """Official [20,10000,64,64] uint8 test array, opened without loading it all."""
    def __init__(self, path, *, official=True):
        self.path = Path(path)
        self.frames = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if (self.frames.dtype != np.uint8 or self.frames.ndim != 4
                or self.frames.shape[0] != 20 or self.frames.shape[2:] != (64, 64)
                or self.frames.shape[1] < 1 or (official and self.frames.shape[1] != 10000)):
            raise ValueError("expected official uint8 [20,10000,64,64] test file")
        self.official = official

    def __len__(self):
        return self.frames.shape[1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        clip = normalize_clip(self.frames[:, index])
        return {"context": clip[:10], "future": clip[10:], "sequence_id": int(index),
                "source": "official_test" if self.official else "synthetic_test_fixture"}

    def manifest(self):
        return {"path": str(self.path.resolve()), "sha256": file_sha256(self.path),
                "shape": list(self.frames.shape), "split": "test", "official": self.official,
                "normalization": "uint8 / 255", "conditioning": "frames0:10"}
