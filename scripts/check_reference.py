"""Read-only equivalence audit against the preserved experiment implementation."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--reference-src", required=True)
args = parser.parse_args()
sys.path.insert(0, args.reference_src)
from lhfm_final.model import FinalUNet
from image_lagrangian_fm.training import train_step as reference_step
from lhfm_final.training import EMA as ReferenceEMA
from lhfm_i import LHFM_I, sample_ode
from lhfm_i.training import EMA, image_batch, optimizer_for, train_step

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/lhfm_i_cifar10.json").read_text())

def assert_weights_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name

# Verify the production parameter count, schema, initialization, and one full-size forward.
torch.manual_seed(270829)
reference = FinalUNet(variant="bounded", channels_last=False)
reference_rng = torch.get_rng_state().clone()
torch.manual_seed(270829)
released = LHFM_I()
assert torch.equal(torch.get_rng_state(), reference_rng)
assert_weights_equal(reference.state_dict(), released.state_dict())
assert sum(p.numel() for p in released.parameters()) == 39627909
with torch.no_grad():
    reference.output.weight.normal_(0, .01)
    reference.output.bias.normal_(0, .01)
    released.load_state_dict(reference.state_dict())
    x, t = torch.randn(1, 3, 32, 32), torch.tensor([.4])
    reference.eval(); released.eval()
    for expected, actual in zip(reference.fields(x, t), released.fields(x, t)):
        assert torch.equal(expected, actual)
del reference, released

# Nonzero tiny model: compare gradients, one AdamW step, EMA, and sampling.
kwargs = dict(image_size=8, base_channels=8, multipliers=(1,2), residual_blocks=1,
              attention_resolutions=(4,), heads=2, dropout=.1)
torch.manual_seed(123)
reference = FinalUNet(**kwargs, variant="bounded", channels_last=False, transport_modes=4)
released = LHFM_I(**kwargs)
with torch.no_grad():
    reference.output.weight.normal_(0, .01)
    reference.output.bias.normal_(0, .01)
released.load_state_dict(reference.state_dict())
reference.train(); released.train()
x, t = torch.randn(2, 3, 8, 8), torch.tensor([.2, .8])
rng = torch.get_rng_state().clone()
reference(x, t).square().mean().backward()
torch.set_rng_state(rng)
released(x, t).square().mean().backward()
for (name, p), (other, q) in zip(reference.named_parameters(), released.named_parameters()):
    assert name == other
    assert torch.equal(p.grad, q.grad), name

reference_optimizer = optimizer_for(reference, config, "cpu")
release_optimizer = optimizer_for(released, config, "cpu")
reference_ema = ReferenceEMA(reference, .9999)
release_ema = EMA(released, .9999)
rng = torch.get_rng_state().clone()
expected = reference_step(reference, x, reference_optimizer, reference_ema, config, 1)
torch.set_rng_state(rng)
actual = train_step(released, x, release_optimizer, release_ema, config, 1)
assert_weights_equal(reference.state_dict(), released.state_dict())
assert_weights_equal(reference_ema.shadow, release_ema.shadow)
assert expected.keys() == actual.keys()
for key in expected:
    assert expected[key] == actual[key], key
initial = torch.randn(2, 3, 8, 8)
left = sample_ode(reference, initial.shape, initial=initial, steps=3)
right = sample_ode(released, initial.shape, initial=initial, steps=3)
assert torch.equal(left.images, right.images)
assert left.nfe == right.nfe == 6
print(json.dumps(dict(status="PASS", parameters=39627909,
    production_initialization_and_rng="bitwise_equal", production_forward_fields="bitwise_equal",
    tiny_gradients="bitwise_equal", tiny_optimizer_update="bitwise_equal",
    tiny_ema="bitwise_equal", tiny_heun="bitwise_equal",
    device="cpu", torch=str(torch.__version__)), indent=2))
