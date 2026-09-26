"""Evaluate a validation-selected LHFM-V EMA on the official Moving MNIST test set."""

import argparse
from pathlib import Path
import torch
from ..checkpoint import load_ema
from ..data import FixedMovingMNISTTest, file_sha256
from ..train import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, payload = load_ema(args.checkpoint, args.device)
    expected = payload["config"]["input_sha256"]["mnist_test_seq.npy"]
    if file_sha256(args.test_file) != expected:
        raise ValueError("official Moving MNIST test file hash mismatch")
    dataset = FixedMovingMNISTTest(args.test_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = evaluate(model, model.state_dict(), dataset, 16, "fp32",
                      preview_dir=args.output_dir / "previews", preview_count=6)
    report.update(model="LHFM-V", selected_step=payload["step"],
                  checkpoint_sha256=file_sha256(args.checkpoint),
                  selection_policy="caller supplies a validation-selected EMA; test is evaluation only")
    write_json(args.output_dir / "metrics.json", report)
    print(report["overall"])


if __name__ == "__main__":
    main()
