"""Sample images using an LHFM-I EMA checkpoint and the image-space ODE."""

import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from lhfm_i import build_model, sample_ode
from lhfm_i.checkpoint import read_checkpoint
from lhfm_i.training import configure_execution


def main():
    parser = argparse.ArgumentParser(description="Sample from LHFM-I")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="samples/lhfm-i.png")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error("count and batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new path")
    configure_execution()
    state = read_checkpoint(args.checkpoint)
    config = state["config"]
    sampling = config["sampling"]
    device = torch.device(args.device)
    model = build_model(config).to(device).eval()
    model.load_state_dict(state["ema"]["shadow"], strict=True)
    seed = args.seed if args.seed is not None else sampling["seed"]
    generator = torch.Generator(device=device).manual_seed(seed)
    steps = args.steps if args.steps is not None else sampling["steps"]
    images = []
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        result = sample_ode(model, (count, 3, model.image_size, model.image_size),
                            device=device, generator=generator, steps=steps,
                            method=sampling["method"], bfloat16=device.type == "cuda",
                            maximum_rms=sampling["maximum_rms"])
        images.append(result.images.cpu())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Clip only for rendering, never within the ODE.
    save_image(((torch.cat(images) + 1) / 2).clamp(0, 1), output)
    print(f"Saved {args.count} images; {result.nfe} NFE per image.")


if __name__ == "__main__":
    main()
