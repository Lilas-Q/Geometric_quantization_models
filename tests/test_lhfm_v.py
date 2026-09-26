import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate
from lhfm_v import LHFM_V, forecast_loss, frozen_exponential_step, upwind_generator
from lhfm_v.schedule import scheduled_values, TrainingScheduler
from lhfm_v.recipe import scheduled_values as original_values
from lhfm_v.engine import Trainer
from lhfm_v.checkpoint import load_ema
from lhfm_v.data import GeneratedMovingMNIST
from lhfm_v.metrics import metric_tensors
from lhfm_v.optimized_expm import SavedExponential
from lhfm_v.scripts.check_reference import SMALL

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "configs/lhfm_v_moving_mnist.json").read_text())
    cfg["model_kwargs"].update(SMALL)
    cfg["model_kwargs"]["widths"] = list(SMALL["widths"])
    return cfg


def test_model_default_size_and_rollout_gradients():
    model = LHFM_V()
    assert sum(p.numel() for p in model.parameters()) == 18637493
    del model
    model = LHFM_V(**SMALL)
    x = torch.rand(1, 3, 1, 16, 16, requires_grad=True)
    y = torch.rand(1, 10, 1, 16, 16)
    forecast = model(x, supervise_plan=True)
    loss, _ = forecast_loss(forecast, y)
    loss.backward()
    assert forecast.prediction.shape == y.shape
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (x.grad.abs().sum((0, 2, 3, 4)) > 0).all()
    with pytest.raises(ValueError):
        model(x, horizon=11)


def test_transport_sign_zero_and_adjoint():
    torch.manual_seed(19)
    ramp = torch.arange(8).float().expand(1, 1, 8, 8)
    velocity = torch.zeros(1, 2, 8, 8); velocity[:, 0] = 1
    torch.testing.assert_close(upwind_generator(ramp, velocity)[..., 2:-2, 2:-2],
                               -torch.ones(1, 1, 4, 4))
    image = torch.rand(1, 1, 8, 8, requires_grad=True)
    source = torch.rand_like(image, requires_grad=True)
    velocity = (torch.rand(1, 2, 8, 8) * .3).requires_grad_()
    torch.testing.assert_close(frozen_exponential_step(image, source, torch.zeros_like(velocity), .5),
                               image + .5 * source, atol=3e-7, rtol=2e-6)
    expected = frozen_exponential_step(image, source, velocity, .5)
    grad = torch.autograd.grad(expected.sum(), (image, source, velocity))
    actual = SavedExponential(cuda_graph=False)(image, source, velocity, .5)
    actual_grad = torch.autograd.grad(actual.sum(), (image, source, velocity))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_grad, grad):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=2e-5)


def test_lr_schedule_cutoff_and_restore():
    cfg = config()
    assert scheduled_values(400000, cfg) == original_values(400000, cfg)
    assert scheduled_values(400001, cfg)[0] == .5 * original_values(400001, cfg)[0]
    assert scheduled_values(1250000, cfg) == (4e-9, original_values(1250000, cfg)[1])
    with pytest.raises(ValueError):
        scheduled_values(1250002, cfg)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    scheduler = TrainingScheduler(optimizer, cfg)
    scheduler.last_epoch = 400000; scheduler._step_count = 400001; scheduler._apply()
    state = scheduler.state_dict()
    clone = TrainingScheduler(optimizer, cfg)
    clone.load_state_dict(state)
    clone.step()
    assert clone.get_last_lr() == [scheduled_values(400002, cfg)[0]]


def test_training_checkpoint_exact_resume_and_ema_loading(tmp_path):
    cfg = config()
    torch.manual_seed(33)
    trainer = Trainer(LHFM_V(**cfg["model_kwargs"]), cfg, binding={"compile": False})
    def batch(start):
        generator = torch.Generator().manual_seed(200 + start)
        return {"context": torch.rand(16, 3, 1, 16, 16, generator=generator),
                "future": torch.rand(16, 10, 1, 16, 16, generator=generator),
                "sequence_id": torch.arange(start, start + 16)}
    trainer.update([batch(0)])
    trainer.save(tmp_path / "latest.pt")
    model, payload = load_ema(tmp_path / "latest.pt")
    assert payload["step"] == 1 and not model.training
    clone = Trainer(LHFM_V(**cfg["model_kwargs"]), cfg, binding={"compile": False})
    clone.load(tmp_path / "latest.pt")
    expected = trainer.update([batch(16)])
    actual = clone.update([batch(16)])
    assert actual == expected
    for field in ("model", "ema"):
        a = trainer.model.state_dict() if field == "model" else trainer.ema
        b = clone.model.state_dict() if field == "model" else clone.ema
        assert all(torch.equal(a[k], b[k]) for k in a)
    clone.save_ema(tmp_path / "ema.pt")
    assert load_ema(tmp_path / "ema.pt")[1]["step"] == 2
    broken = torch.load(tmp_path / "ema.pt", weights_only=True)
    broken["step"] = 1250001
    torch.save(broken, tmp_path / "experimental.pt")
    with pytest.raises(ValueError):
        load_ema(tmp_path / "experimental.pt")


def test_data_and_metrics():
    digits = np.zeros((3, 28, 28), np.uint8); digits[:, 7:21, 10:18] = 255
    dataset = GeneratedMovingMNIST(digits, np.arange(3), length=3)
    assert torch.equal(dataset[1]["context"], dataset[1]["context"])
    clip = dataset[0]["future"][None]
    metrics = metric_tensors(clip, clip)
    assert not metrics["mse_raw"].any()
    torch.testing.assert_close(metrics["ssim_clipped"], torch.ones(1, 10, dtype=torch.float64))


@pytest.mark.parametrize("name", ["train", "predict", "evaluate", "check_reference"])
def test_video_cli_help(name):
    result = subprocess.run([sys.executable, "-m", "lhfm_v.scripts." + name, "--help"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_video_train_resume_predict_cli(tmp_path, monkeypatch):
    from lhfm_v.scripts import train, predict
    cfg = config()
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    class Synthetic:
        length = 32
        def __len__(self):
            return self.length
        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return dict(context=torch.rand(3, 1, 16, 16, generator=g),
                        future=torch.rand(10, 1, 16, 16, generator=g), sequence_id=i)
    monkeypatch.setattr(train, "load_datasets", lambda *args: (Synthetic(), None))
    out = tmp_path / "training"
    base = ["train", "--config", str(config_file), "--data-dir", str(tmp_path),
            "--output-dir", str(out), "--device", "cpu", "--workers", "0"]
    monkeypatch.setattr(sys, "argv", base + ["--max-steps", "1"])
    train.main()
    monkeypatch.setattr(sys, "argv", base + ["--max-steps", "2", "--resume"])
    train.main()
    assert load_ema(out / "latest.pt")[1]["step"] == 2
    context = tmp_path / "context.npy"
    np.save(context, np.zeros((3, 1, 16, 16), np.uint8))
    prediction = tmp_path / "prediction"
    monkeypatch.setattr(sys, "argv", ["predict", "--checkpoint", str(out / "last-ema.pt"),
                       "--context", str(context), "--output-dir", str(prediction)])
    predict.main()
    assert np.load(prediction / "prediction.npy").shape == (1, 10, 1, 16, 16)
    assert (prediction / "prediction-00000.png").is_file()
