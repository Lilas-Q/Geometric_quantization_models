"""Read EMA weights without importing a historical training controller."""

import hashlib
from pathlib import Path
import torch
from .model import MODEL_ID, PhysicalFastPlanMovingMNIST
from .engine import FULL_CHECKPOINT_SCHEMA, EMA_CHECKPOINT_SCHEMA


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def load_ema(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    allowed = (FULL_CHECKPOINT_SCHEMA, EMA_CHECKPOINT_SCHEMA,
               "lhfm_v17_wide_b16_onecycle_full_checkpoint_v1",
               "lhfm_v17_wide_b16_onecycle_ema_v1")
    if (not isinstance(payload, dict) or payload.get("schema") not in allowed
            or payload.get("model_id") != MODEL_ID
            or type(payload.get("step")) is not int
            or not 0 <= payload["step"] <= 1250000):
        raise ValueError("expected an LHFM-V checkpoint at or before 1250000 updates")
    model = PhysicalFastPlanMovingMNIST(**payload["config"]["model_kwargs"])
    if model.config() != payload.get("model_config"):
        raise ValueError("checkpoint model configuration mismatch")
    expected, weights = model.state_dict(), payload["ema"]
    if weights.keys() != expected.keys():
        raise ValueError("EMA key mismatch")
    for name, reference in expected.items():
        value = weights[name]
        if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                or value.dtype != reference.dtype or not torch.isfinite(value).all()):
            raise ValueError("invalid EMA tensor: " + name)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), payload
