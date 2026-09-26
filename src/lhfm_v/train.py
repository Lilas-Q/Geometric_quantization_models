"""Explicit CLI for a future authorized run; importing this module never trains."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader, default_collate

from .data import GeneratedMovingMNIST, FixedMovingMNISTTest, file_sha256, digit_split
from .engine import Trainer
from .execution import Execution
from .metrics import PROTOCOL, metric_tensors
from .model import PhysicalFastPlanMovingMNIST
from .pipeline import DevicePrefetcher
from .recipe import validate_recipe
from .continuation import is_validation_step


def validate_compile_environment(compile_enabled):
    if compile_enabled:
        for name in ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE"):
            if os.environ.get(name, "0") not in ("", "0"):
                raise RuntimeError(f"{name} disables the required compiled execution")


def append_training_log(stream, stats):
    """Keep one buffered file open; periodically expose completed update rows."""
    stream.write(json.dumps(stats, allow_nan=False) + "\n")
    if stats["step"] <= 3 or stats["step"] % 25 == 0:
        stream.flush()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _best_entries(history, count):
    return sorted(history, key=lambda row: (row["mse_raw"], row["step"]))[:count]


def _verify_best_files(root, best):
    for row in best:
        path = root / f"best-{row['step']:06d}-ema.pt"
        if not path.is_file() or file_sha256(path) != row.get("ema_sha256"):
            raise ValueError("committed best EMA missing or changed: " + str(path))


def complete_current_node(trainer, evaluator_model, validation, batch_size, root, config):
    """Replay the latest step's unfinished validation/milestone before advancing.

    The history file is the validation commit marker. All referenced metrics
    and best EMA files are durable before that marker changes; pruning follows
    it. An interrupted uncommitted node is evaluated again from this step's
    complete latest state. An interrupted post-commit prune is safe to repeat.
    """
    root = Path(root)
    history_path = root / "validation-history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if not isinstance(history, list):
        raise ValueError("validation history must be a list")
    steps = set()
    for row in history:
        if (not isinstance(row, dict) or type(row.get("step")) is not int
                or not is_validation_step(row["step"],config)
                or row["step"] > trainer.step or row["step"] in steps
                or row.get("identity") != trainer.identity
                or type(row.get("mse_raw")) not in (int, float)
                or not math.isfinite(row["mse_raw"]) or row["mse_raw"] < 0):
            raise ValueError("validation history identity/step/metric mismatch")
        path = root / f"validation-{row['step']:06d}.json"
        if not path.is_file() or file_sha256(path) != row.get("metrics_sha256"):
            raise ValueError("committed validation metrics missing or changed")
        steps.add(row["step"])
    if any(step < trainer.step and step not in steps for step in config["validation_steps"]):
        raise ValueError("earlier validation node missing; latest cannot reconstruct its old EMA")

    if is_validation_step(trainer.step,config) and trainer.step not in steps:
        metrics = evaluate(evaluator_model, trainer.ema, validation, batch_size,
                           config["evaluation_precision"])
        metrics.update(step=trainer.step, identity=trainer.identity)
        score = metrics["overall"]["mse_raw"]
        if not math.isfinite(score) or score < 0:
            raise FloatingPointError("invalid validation selection metric")
        metrics_path = root / f"validation-{trainer.step:06d}.json"
        write_json(metrics_path, metrics)
        entry = dict(step=trainer.step, mse_raw=score, identity=trainer.identity,
                     metrics_sha256=file_sha256(metrics_path))
        candidate_history = sorted(history + [entry], key=lambda row: row["step"])
        best = _best_entries(candidate_history, config["retention"]["best_ema_count"])
        if any(row["step"] == trainer.step for row in best):
            path = root / f"best-{trainer.step:06d}-ema.pt"
            trainer.save_ema(path)
            _sync_directory(root)
            entry["ema_sha256"] = file_sha256(path)
        _verify_best_files(root, best)
        # This is the transaction commit. No old best is deleted before it.
        write_json(history_path, candidate_history)
        history = candidate_history

    best = _best_entries(history, config["retention"]["best_ema_count"])
    _verify_best_files(root, best)
    keep = {f"best-{row['step']:06d}-ema.pt" for row in best}
    for path in root.glob("best-*-ema.pt"):
        # Delete only eligible names from this run's validation-node set.
        try:
            step = int(path.name[5:-7])
        except ValueError:
            continue
        if is_validation_step(step,config) and path.name not in keep:
            path.unlink()
    _sync_directory(root)

    # Milestones are independent, idempotent artifacts. Their absence at the
    # current latest step is recoverable even after validation was committed.
    for step in config["retention"]["milestone_ema_steps"]:
        path = root / f"milestone-{step:06d}-ema.pt"
        if step < trainer.step and not path.is_file():
            raise ValueError("earlier milestone missing; latest cannot reconstruct its old EMA")
        if step == trainer.step and not path.is_file():
            trainer.save_ema(path)
            _sync_directory(root)
    return history


@torch.no_grad()
def evaluate(model, weights, dataset, batch_size, precision, *, preview_dir=None, preview_count=0):
    model.load_state_dict(weights, strict=True)
    model.eval()
    execution = Execution(model, precision=precision)
    device = next(model.parameters()).device
    totals, count = {}, 0
    previews = []
    for start in range(0, len(dataset), batch_size):
        batch = default_collate([dataset[i] for i in range(start, min(len(dataset), start + batch_size))])
        result = execution(batch["context"].to(device), horizon=10)
        # Float64 metrics run on CPU; no FP64 convolution bottleneck on consumer CUDA.
        values = metric_tensors(result.prediction.cpu(), batch["future"])
        for key, value in values.items():
            totals[key] = totals.get(key, torch.zeros(10, dtype=torch.float64)) + value.sum(0)
        count += len(batch["context"])
        for i in range(min(len(batch["context"]), max(0, preview_count - start))):
            previews.append((int(batch["sequence_id"][i]), batch["context"][i],
                             batch["future"][i], result.prediction[i].cpu()))
    if previews and preview_dir is not None:
        from PIL import Image
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        for index, context, target, prediction in previews:
            rows = [torch.cat(list(x[:, 0]), -1) for x in (context, target, prediction)]
            canvas = torch.cat(rows, -2).clamp(0, 1).mul(255).round().byte().numpy()
            Image.fromarray(canvas).save(Path(preview_dir) / f"sequence-{index:05d}.png")
    return dict(protocol=PROTOCOL, count=count,
                overall={key: float(value.mean()/count) for key, value in totals.items()},
                per_frame={key: (value/count).tolist() for key, value in totals.items()})


def main():
    raise RuntimeError('Use package/run_training.py for validation-controlled continuation')


if __name__ == '__main__':
    main()
