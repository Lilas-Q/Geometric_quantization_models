"""Train LHFM-V from scratch, or resume this package's complete checkpoint."""

import argparse
import json
import os
from pathlib import Path
import signal
import torch
from torch.utils.data import DataLoader
from ..model import PhysicalFastPlanMovingMNIST
from ..engine import Trainer
from ..data import GeneratedMovingMNIST, digit_split, file_sha256
from ..pipeline import DevicePrefetcher
from ..checkpoint import source_hashes
from ..recipe import validate_recipe
from ..schedule import validate_schedule_config
from ..continuation import is_validation_step
from ..train import complete_current_node, write_json, append_training_log

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/lhfm_v_moving_mnist.json"


def load_datasets(data_dir, config):
    path = data_dir / "train-images-idx3-ubyte.gz"
    if file_sha256(path) != config["input_sha256"][path.name]:
        raise ValueError("MNIST training digit file hash mismatch")
    train = GeneratedMovingMNIST.from_training_idx(
        path, split="train", length=config["budget_sequence_exposures"],
        seed=config["data_seed"], split_seed=config["split_seed"],
        validation_count=config["heldout_digits"])
    validation = GeneratedMovingMNIST(
        train.digits, digit_split(len(train.digits), config["heldout_digits"], config["split_seed"])[1],
        length=config["validation_count"], seed=config["validation_seed"], source_id="heldout_validation")
    return train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, help="earlier stop; does not shorten the LR schedule")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="eager execution; does not alter the model")
    args = parser.parse_args()
    config = validate_recipe(json.loads(args.config.read_text()))
    validate_schedule_config(config)
    limit = config["stop_steps"] if args.max_steps is None else args.max_steps
    if not 1 <= limit <= 1250000 or args.workers < 0:
        parser.error("steps must be 1..1250000 and workers nonnegative")
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("CUDA with BF16 support is required for the default training recipe")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    train, validation = load_datasets(args.data_dir, config)
    binding = dict(source_hashes=source_hashes(), dataset_sha256=config["input_sha256"],
                   torch=str(torch.__version__), cuda=torch.version.cuda,
                   device=args.device, compile=args.device == "cuda" and not args.no_compile,
                   validation_batch=16, publication="LHFM-V")
    root = args.output_dir
    if args.resume:
        if not (root / "latest.pt").is_file():
            parser.error("--resume requires an existing latest.pt")
    else:
        root.mkdir(parents=True, exist_ok=False)
    # Protect this exact output directory against concurrent writers.
    import fcntl
    with (root / "TRAIN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        trainer = Trainer(PhysicalFastPlanMovingMNIST(**config["model_kwargs"]).to(args.device),
                          config, binding=binding)
        evaluator = PhysicalFastPlanMovingMNIST(**config["model_kwargs"]).to(args.device)
        if args.resume:
            trainer.load(root / "latest.pt")
        if trainer.step > limit:
            parser.error("requested stop precedes the saved checkpoint")
        write_json(root / "config.json", config)
        stopped = False

        def request_stop(*_):
            nonlocal stopped
            stopped = True

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

        def validate_node():
            if is_validation_step(trainer.step, config):
                trainer.save(root / "latest.pt")
                history = complete_current_node(trainer, evaluator, validation, 16, root, config)
                trainer.record_validation(history)
                trainer.save(root / "latest.pt")

        try:
            validate_node()
            trainer.save(root / "latest.pt")
            with (root / "train.jsonl").open("a", buffering=65536) as stream:
                while trainer.step < limit and not stopped:
                    train.length = trainer.sequence_cursor + (limit - trainer.step) * 16
                    loader = DataLoader(train, batch_size=16,
                        sampler=range(trainer.sequence_cursor, len(train)),
                        num_workers=args.workers, pin_memory=args.device == "cuda",
                        generator=torch.Generator().manual_seed(config["seed"] + 1))
                    with DevicePrefetcher(loader, args.device) as batches:
                        for batch in batches:
                            if stopped or trainer.step >= limit:
                                break
                            stats = trainer.update([batch])
                            append_training_log(stream, stats)
                            if stats.get("skipped"):
                                trainer.save(root / "latest.pt")
                                print(json.dumps(stats), flush=True)
                                if stats["stop_required"]:
                                    raise RuntimeError("numerical batch circuit breaker opened; review required")
                                continue
                            if trainer.step <= 3 or trainer.step % 25 == 0:
                                print(json.dumps(stats), flush=True)
                            if trainer.step % 500 == 0:
                                trainer.save(root / "latest.pt")
                            validate_node()
            trainer.save(root / "latest.pt")
            trainer.save_ema(root / "last-ema.pt")
            write_json(root / "STATUS.json", dict(step=trainer.step, sequence_cursor=trainer.sequence_cursor,
                phase="complete" if trainer.step == config["stop_steps"] else "paused",
                publication="LHFM-V", test_evaluated=False))
        finally:
            trainer.save(root / "latest.pt")
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
