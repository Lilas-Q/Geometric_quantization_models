"""Predict future video frames from observed frames using LHFM-V EMA weights."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..checkpoint import load_ema


def load_context(path, frames):
    values = np.load(path, allow_pickle=False)
    if values.ndim == 4:
        values = values[None]
    if values.ndim != 5 or values.shape[0] < 1 or values.shape[1:3] != (frames, 1):
        raise ValueError("input must be [B,T,1,H,W] or [T,1,H,W]")
    if values.dtype == np.uint8:
        values = values.astype(np.float32) / 255.0
    elif not np.issubdtype(values.dtype, np.floating):
        raise ValueError("input must be uint8 or floating point in [0,1]")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("observed image values must be finite and in [0,1]")
    return torch.from_numpy(values.astype(np.float32, copy=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True, help="observed frames only, .npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--horizon", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = load_ema(args.checkpoint, args.device)
    context = load_context(args.context, model.context_frames)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = []
    with torch.inference_mode():
        for batch in context.split(args.batch_size):
            result = model(batch.to(args.device), horizon=args.horizon).prediction.cpu()
            if not torch.isfinite(result).all():
                raise FloatingPointError("nonfinite prediction")
            predictions.append(result)
    prediction = torch.cat(predictions)
    np.save(args.output_dir / "prediction.npy", prediction.numpy(), allow_pickle=False)
    # No prediction clipping in the saved array or rollout. Clip for display only.
    for index, clip in enumerate(prediction):
        strip = torch.cat(list(clip[:, 0]), -1).clamp(0, 1).mul(255).round().byte().numpy()
        Image.fromarray(strip).save(args.output_dir / f"prediction-{index:05d}.png")


if __name__ == "__main__":
    main()
