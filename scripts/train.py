"""Train LHFM-I on CIFAR-10; no training is launched by installing the package."""

import argparse
import json
from pathlib import Path
import torch
from lhfm_i import build_model
from lhfm_i.checkpoint import read_checkpoint, restore_training, save_checkpoint
from lhfm_i.training import (EMA, configure_execution, image_batch, load_cifar,
                             optimizer_for, set_seed, train_step)


def main():
    parser = argparse.ArgumentParser(description="Train LHFM-I")
    parser.add_argument("--config", default="configs/lhfm_i_cifar10.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/lhfm-i")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Earlier stop; does not change the LR horizon")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("model_name") != "LHFM-I":
        parser.error("expected an LHFM-I configuration")
    training = config["training"]
    stop = args.max_steps if args.max_steps is not None else training["stop_step"]
    if not 1 <= stop <= training["steps"]:
        parser.error("stop must lie within the learning-rate horizon")
    output = Path(args.output_dir)
    checkpoint = output / "latest.pt"
    if args.resume and not checkpoint.is_file():
        parser.error("--resume requires output-dir/latest.pt")
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; use --resume or choose a new directory")
    configure_execution()
    set_seed(training["seed"])
    device = torch.device(args.device)
    model = build_model(config).to(device)
    optimizer = optimizer_for(model, config, device)
    ema = EMA(model, training["ema_decay"])
    pixels, dataset_sha = load_cifar(args.data_dir, download=args.download)
    # The original experiment keeps the uint8 dataset on the training device.
    pixels = pixels.to(device)
    start = 0
    if args.resume:
        start = restore_training(read_checkpoint(checkpoint), model=model,
                                 optimizer=optimizer, ema=ema, config=config,
                                 dataset_sha256=dataset_sha)
    if start >= stop:
        parser.error("checkpoint has already reached the requested stop")
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    print(json.dumps({"model": "LHFM-I", "parameters": sum(p.numel() for p in model.parameters()),
                      "start_step": start, "stop_step": stop, "lr_horizon": training["steps"]}))
    for step in range(start + 1, stop + 1):
        images = image_batch(pixels, config, step, device)
        metrics = train_step(model, images, optimizer, ema, config, step,
                             collect_diagnostics=False)
        if step == 1 or step % training["log_every"] == 0 or step == stop:
            print(json.dumps({"step": step, **{k: float(v) for k, v in metrics.items()}}), flush=True)
        if step % training["checkpoint_every"] == 0 or step == stop:
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, ema=ema,
                            config=config, step=step, dataset_sha256=dataset_sha)


if __name__ == "__main__":
    main()
