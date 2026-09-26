import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lhfm_v.metrics import PROTOCOL, metric_tensors
from lhfm_v.ssim import frame_ssim

ROOT = Path(__file__).resolve().parents[1]


def test_ssim_against_300_original_source_scores():
    fixture = json.loads((ROOT / "tests/fixtures/ssim_0193.json").read_text())
    rng = np.random.RandomState(fixture["seed"])
    for case in fixture["cases"]:
        shape = (2, 10, 1, case["height"], case["width"])
        y = rng.uniform(0, 1, shape).astype(np.float32)
        p = (y + rng.normal(0, .15, shape)).astype(np.float32)
        if case["kind"] == "random":
            p = np.clip(p, 0, 1)
        elif case["kind"] == "identical":
            p = y.copy()
        actual = frame_ssim(p, y)
        expected = np.asarray(case["expected"], dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)
        wrapped = metric_tensors(torch.from_numpy(p), torch.from_numpy(y))
        np.testing.assert_array_equal(wrapped["ssim_clipped"].numpy(), expected)
        assert wrapped["ssim_clipped"].dtype == torch.float64


def test_metric_clipping_raw_reductions_and_data_range():
    p = torch.full((2, 3, 1, 9, 13), 2.)
    y = torch.zeros_like(p)
    result = metric_tensors(p, y)
    assert torch.all(result["mse_raw"] == 4)
    assert torch.all(result["mae_raw"] == 2)
    assert torch.all(result["mse_spatial_sum_raw"] == 4 * 9 * 13)
    assert torch.all(result["mae_spatial_sum_raw"] == 2 * 9 * 13)
    assert torch.all(result["mse_clipped"] == 1)
    assert torch.all(result["mae_clipped"] == 1)
    assert torch.all(result["psnr_clipped"] == 0)
    assert torch.all(result["out_of_range_fraction"] == 1)
    expected = np.float32((.01 * 2)**2 / (1 + (.01 * 2)**2))
    np.testing.assert_allclose(result["ssim_clipped"].numpy(), expected, rtol=1e-7)
    same = metric_tensors(y, y)
    assert torch.all(same["ssim_clipped"] == 1)
    assert torch.all(same["psnr_clipped"] == 120)


@pytest.mark.parametrize("case", ["shape", "size", "nan", "target_range"])
def test_metric_validation(case):
    p = torch.zeros(1, 1, 1, 9, 13)
    y = p.clone()
    error = ValueError
    if case == "shape":
        p = p[:, :, 0]
    elif case == "size":
        p, y = p[..., :6, :], y[..., :6, :]
    elif case == "nan":
        p[..., 0, 0] = float("nan")
        error = FloatingPointError
    else:
        y[..., 0, 0] = 2
    with pytest.raises(error):
        metric_tensors(p, y)
    with pytest.raises(error):
        frame_ssim(p.numpy(), y.numpy())


def test_published_metric_protocol_and_values():
    result = json.loads((ROOT / "results/lhfm_v_moving_mnist_1250k.json").read_text())
    assert result["test"]["protocol"] == PROTOCOL
    overall = result["test"]["overall"]
    assert overall["ssim_clipped"] == 0.9582201841366291
    assert abs(np.mean(result["test"]["per_frame"]["ssim_clipped"])
               - overall["ssim_clipped"]) < 1e-15
    assert overall["mse_spatial_sum_raw"] == 18.581171026784503
    assert overall["mae_spatial_sum_raw"] == 62.51341348158763
    assert overall["psnr_clipped"] == 24.674745890517997
