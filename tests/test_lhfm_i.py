import copy
import json
import math
from pathlib import Path
import random
import pytest
import torch
from lhfm_i import LHFM_I, build_model, conditional_path, sample_ode, velocity_loss
from lhfm_i.checkpoint import read_checkpoint, restore_training, save_checkpoint
from lhfm_i.geometry import CosineSpatialDerivative
from lhfm_i.training import EMA, image_batch, learning_rate, optimizer_for, set_seed, train_step

ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    config = json.loads((ROOT / "configs/lhfm_i_cifar10.json").read_text())
    config["model"].update(image_size=8, base_channels=8, multipliers=[1, 2],
                           residual_blocks=1, attention_resolutions=[4], heads=2)
    config["training"].update(batch_size=2)
    return config


def test_paper_parameter_count():
    config = json.loads((ROOT / "configs/lhfm_i_cifar10.json").read_text())
    model = build_model(config)
    assert sum(p.numel() for p in model.parameters()) == 39627909
    assert not any("operators" in key for key in model.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spatial_derivative_orientation_and_full_band(dtype):
    n = 8
    coordinates = (torch.arange(n, dtype=dtype) + .5) / n
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = (torch.cos(7 * math.pi * x) * torch.cos(3 * math.pi * y))[None, None]
    dx, dy = CosineSpatialDerivative(n).spatial_derivatives(image)
    expected_x = -7 * math.pi * torch.sin(7 * math.pi * x) * torch.cos(3 * math.pi * y)
    expected_y = -3 * math.pi * torch.cos(7 * math.pi * x) * torch.sin(3 * math.pi * y)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(dx[0, 0], expected_x, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(dy[0, 0], expected_y, atol=tolerance, rtol=tolerance)


def test_derivative_precision_under_autocast_and_module_cast():
    operators = CosineSpatialDerivative(8).half()
    assert operators.derivative.dtype == torch.float32
    assert operators.derivative_double.dtype == torch.float64
    image = torch.randn(2, 3, 8, 8)
    expected = operators.spatial_derivatives(image)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = operators.spatial_derivatives(image)
    for left, right in zip(expected, actual):
        assert right.dtype == torch.float32
        assert torch.equal(left, right)


def test_fields_velocity_and_joint_gradients():
    model = build_model(tiny_config()).eval()
    with torch.no_grad():
        model.output.weight.normal_(0, .01)
        model.output.bias.normal_(0, .01)
    image = torch.randn(2, 3, 8, 8)
    time = torch.tensor([0., .7])
    v, u, r = model.fields(image, time)
    assert v.shape == r.shape == image.shape
    assert u.shape == (2, 2, 8, 8)
    assert torch.count_nonzero(u[0]) == 0
    assert u[1].abs().max() <= .125 * .7
    dx, dy = model.operators.spatial_derivatives(image)
    torch.testing.assert_close(v, r - dx * u[:, :1] - dy * u[:, 1:])
    batch = conditional_path(image, time=time, noise=torch.zeros_like(image))
    velocity_loss(v, batch).backward()
    assert model.output.weight.grad[:2].abs().sum() > 0
    assert model.output.weight.grad[2:].abs().sum() > 0


def test_linear_conditional_path_and_loss():
    image, noise = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    for time, expected in [(0., noise), (1., image), (.5, (noise + image) / 2)]:
        batch = conditional_path(image, time=time, noise=noise)
        torch.testing.assert_close(batch.image, expected)
        assert torch.equal(batch.target, image - noise)
        assert velocity_loss(batch.target, batch) == 0
        torch.testing.assert_close(velocity_loss(batch.target + 1, batch), torch.tensor(1.))
    with pytest.raises(ValueError):
        conditional_path(image, time=1.1)


@pytest.mark.parametrize("method,nfe", [("euler", 4), ("heun", 8)])
def test_sampler_nfe_no_intermediate_clipping_and_mode_restore(method, nfe):
    class Constant(torch.nn.Module):
        def forward(self, image, time):
            return torch.full_like(image, 2.)
    model = Constant().train()
    result = sample_ode(model, (2, 3, 8, 8), steps=4, method=method,
                        initial=torch.zeros(2, 3, 8, 8))
    assert result.nfe == nfe
    assert model.training
    assert torch.equal(result.images, torch.full_like(result.images, 2.))


def test_sampler_rejects_nonfinite_and_restores_mode():
    class Invalid(torch.nn.Module):
        def forward(self, image, time):
            return torch.full_like(image, float("nan"))
    model = Invalid().train()
    with pytest.raises(FloatingPointError):
        sample_ode(model, (1, 3, 8, 8), steps=1)
    assert model.training


def test_schedule_keeps_paper_horizon():
    config = tiny_config()
    assert learning_rate(2000, config) == .00025
    assert learning_rate(400000, config) == .00002
    assert learning_rate(150000, config) > .00002


def test_checkpoint_resume_is_exact_with_dropout(tmp_path):
    config = tiny_config()
    set_seed(123)
    pixels = torch.randint(0, 256, (12, 3, 8, 8), dtype=torch.uint8)
    model = build_model(config).train()
    optimizer = optimizer_for(model, config, "cpu")
    ema = EMA(model, config["training"]["ema_decay"])
    train_step(model, image_batch(pixels, config, 1, "cpu"), optimizer, ema, config, 1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, ema=ema, config=config,
                    step=1, dataset_sha256="test-data")
    train_step(model, image_batch(pixels, config, 2, "cpu"), optimizer, ema, config, 2)
    expected = copy.deepcopy(model.state_dict())
    expected_ema = copy.deepcopy(ema.shadow)
    expected_random = (random.random(), torch.rand(2))
    replacement = build_model(config).train()
    new_optimizer = optimizer_for(replacement, config, "cpu")
    new_ema = EMA(replacement, config["training"]["ema_decay"])
    state = read_checkpoint(path)
    assert restore_training(state, model=replacement, optimizer=new_optimizer,
                            ema=new_ema, config=config, dataset_sha256="test-data") == 1
    train_step(replacement, image_batch(pixels, config, 2, "cpu"), new_optimizer, new_ema, config, 2)
    assert all(torch.equal(expected[k], v) for k, v in replacement.state_dict().items())
    assert all(torch.equal(expected_ema[k], v) for k, v in new_ema.shadow.items())
    assert random.random() == expected_random[0]
    assert torch.equal(torch.rand(2), expected_random[1])
    with pytest.raises(ValueError, match="configuration or dataset"):
        restore_training(state, model=replacement, optimizer=new_optimizer,
                         ema=new_ema, config=config, dataset_sha256="different-data")
