"""Atomic full-state checkpoints for the standalone LHFM-I trainer."""

import hashlib
import os
from pathlib import Path
import random
import tempfile
import torch

FORMAT = "lhfm-i-training-v1"


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def save_checkpoint(path, *, model, optimizer, ema, config, step, dataset_sha256):
    if ema.updates != step:
        raise ValueError("EMA update count must match the training step")
    device = next(model.parameters()).device
    state = dict(format=FORMAT, model_name="LHFM-I", config=config, step=step,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 ema=ema.state_dict(), dataset_sha256=dataset_sha256,
                 source_hashes=source_hashes(), python_rng_state=random.getstate(),
                 torch_rng_state=torch.get_rng_state(),
                 cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 device_type=device.type,
                 runtime={"torch": str(torch.__version__), "cuda": torch.version.cuda})
    validate_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def validate_state(state):
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("unsupported checkpoint format")
    if not isinstance(state.get("step"), int) or state["step"] < 1:
        raise ValueError("invalid training step")
    if state["ema"]["updates"] != state["step"]:
        raise ValueError("EMA update count mismatch")
    if set(state["model"]) != set(state["ema"]["shadow"]):
        raise ValueError("EMA state schema mismatch")
    for weights in (state["model"], state["ema"]["shadow"]):
        if not all(torch.isfinite(value).all().item() for value in weights.values()):
            raise ValueError("nonfinite model or EMA checkpoint")


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate_state(state)
    return state


def restore_training(state, *, model, optimizer, ema, config, dataset_sha256):
    validate_state(state)
    if state["config"] != config or state["dataset_sha256"] != dataset_sha256:
        raise ValueError("configuration or dataset differs from the checkpoint")
    if state["source_hashes"] != source_hashes():
        raise ValueError("training source has changed")
    device = next(model.parameters()).device
    if state["device_type"] != device.type:
        raise ValueError("training resume requires the same device type")
    if state["runtime"] != {"torch": str(torch.__version__), "cuda": torch.version.cuda}:
        raise ValueError("training resume requires the same PyTorch/CUDA runtime")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    ema.load_state_dict(state["ema"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda":
        if len(state["cuda_rng_state_all"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return state["step"]
