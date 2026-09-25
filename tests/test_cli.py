import importlib.util
import json
from pathlib import Path
import sys
import torch
from PIL import Image
from lhfm_i.checkpoint import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"lhfm_i_{name}_cli", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_resume_and_sample_cli_on_synthetic_data(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/lhfm_i_cifar10.json").read_text())
    config["model"].update(image_size=8, base_channels=8, multipliers=[1, 2],
                           residual_blocks=1, attention_resolutions=[4], heads=2)
    config["training"].update(batch_size=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    pixels = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
    trainer = load_script("train")
    monkeypatch.setattr(trainer, "load_cifar", lambda *a, **k: (pixels, "synthetic-fixture"))
    base = ["train.py", "--config", str(config_path), "--output-dir", str(output), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", base + ["--max-steps", "1"])
    trainer.main()
    assert read_checkpoint(output / "latest.pt")["step"] == 1
    monkeypatch.setattr(sys, "argv", base + ["--resume", "--max-steps", "2"])
    trainer.main()
    assert read_checkpoint(output / "latest.pt")["step"] == 2
    sampler = load_script("sample")
    preview = tmp_path / "preview.png"
    monkeypatch.setattr(sys, "argv", ["sample.py", "--checkpoint", str(output / "latest.pt"),
                       "--output", str(preview), "--device", "cpu", "--count", "1", "--steps", "2"])
    sampler.main()
    with Image.open(preview) as image:
        assert image.size == (8, 8)
