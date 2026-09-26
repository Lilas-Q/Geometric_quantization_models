"""Compare the public LHFM-V package against an archived original package (CPU only)."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
from .. import LHFM_V, forecast_loss
from ..train import write_json
from ..schedule import scheduled_values


SMALL = dict(widths=(4, 8, 16), context_frames=3, latent_channels=8, depth=1,
    expansion=2, spatial_groups=2, controller_hidden=8, field_hidden=8,
    mixing_rank=4, memory_frames=4, memory_key_dim=4, motion_depth=1,
    motion_expansion=2, motion_channels=4, memory_value_dim=4, memory_heads=2,
    coarse_channels=4, coarse_depth=1, plan_channels=4, plan_depth=1,
    plan_expansion=2, correction_hidden=4)


def exact_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, required=True,
                        help="directory containing the archived lhfm_physical/__init__.py")
    parser.add_argument("--reference-lr-policy", type=Path,
                        help="optional original 400k half-LR runtime/lr_policy.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.reference_package.resolve()
    spec = importlib.util.spec_from_file_location("lhfm_physical", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    torch.manual_seed(909)
    old = reference.PhysicalFastPlanMovingMNIST()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(909)
    public = LHFM_V()
    exact_state(old.state_dict(), public.state_dict())
    assert torch.equal(old_rng, torch.get_rng_state())
    assert sum(p.numel() for p in public.parameters()) == 18637493
    x = torch.rand(1, 10, 1, 64, 64)
    with torch.no_grad():
        expected, actual = old(x), public(x)
    for got, want in zip(actual, expected):
        if got is not None:
            torch.testing.assert_close(got, want, atol=0, rtol=0)
    del old, public, actual, expected
    torch.manual_seed(910)
    old = reference.PhysicalFastPlanMovingMNIST(**SMALL)
    public = LHFM_V(**SMALL)
    with torch.no_grad():
        old.correction.source.weight.normal_(std=.002)
        old.correction.transport.weight.normal_(std=.002)
    public.load_state_dict(old.state_dict())
    a = torch.rand(1, 3, 1, 16, 16, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    y = torch.rand(1, 10, 1, 16, 16)
    expected = old(a, supervise_plan=True)
    actual = public(b, supervise_plan=True)
    old_loss = reference.forecast_loss(expected, y)[0]
    new_loss = forecast_loss(actual, y)[0]
    torch.testing.assert_close(old_loss, new_loss, atol=0, rtol=0)
    old_loss.backward(); new_loss.backward()
    for (name, p), (name2, q) in zip(old.named_parameters(), public.named_parameters()):
        assert name == name2 and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    old_ema = {k: v.clone() for k, v in old.state_dict().items()}
    new_ema = {k: v.clone() for k, v in public.state_dict().items()}
    for model, ema in ((old, old_ema), (public, new_ema)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, betas=(.95, .999),
                                      weight_decay=1e-4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                ema[key].lerp_(value, 1 - .999)
    exact_state(old.state_dict(), public.state_dict())
    exact_state(old_ema, new_ema)
    schedule_check = "not requested"
    if args.reference_lr_policy:
        spec = importlib.util.spec_from_file_location("reference_lr_policy", args.reference_lr_policy)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        config = json.loads((root.parent / "training_config.json").read_text())
        for step in (1, 5000, 375000, 399999, 400000, 400001, 400002,
                     600000, 1000000, 1249999, 1250000, 1250001):
            assert scheduled_values(step, config) == policy.scheduled_values(step, config), step
        schedule_check = "exact at 12 boundary/interior indices"
    report = dict(model="LHFM-V", parameters=18637493, device="cpu", torch=str(torch.__version__),
        initialization_state_and_rng="bitwise equal", full_64x64_10_to_10_forward="bitwise equal",
        small_nonzero_correction_loss_and_all_gradients="bitwise equal",
        small_adamw_update_and_ema="bitwise equal", lr_and_beta1=schedule_check,
        new_training_run=False, cuda_tested=False)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
