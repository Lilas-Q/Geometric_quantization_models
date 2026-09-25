"""Self-contained training helpers; step-addressed data/noise, full-state EMA."""

import hashlib
import math
import random
import torch
from .flow import conditional_path, velocity_loss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_execution():
    """Same backend policy in the smoke gate, training, and inference."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False


def step_generator(seed, step, device, stream=0):
    digest = hashlib.sha256(f"image-lagrangian-fm:{seed}:{step}:{stream}".encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


def learning_rate(step, config):
    t = config["training"]
    if not 1 <= step <= t["steps"]:
        raise ValueError("step outside frozen LR horizon")
    if step <= t["warmup_steps"]:
        return t["learning_rate"] * step / t["warmup_steps"]
    progress = (step - t["warmup_steps"]) / (t["steps"] - t["warmup_steps"])
    return t["minimum_learning_rate"] + .5 * (t["learning_rate"] - t["minimum_learning_rate"]) * (1 + math.cos(math.pi * progress))


def optimizer_for(model, config, device):
    t = config["training"]
    return torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], betas=tuple(t["adam_betas"]),
                            weight_decay=t["weight_decay"], fused=torch.device(device).type == "cuda")


class EMA:
    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(list(self.shadow.values()), list(model.state_dict().values()), 1 - decay)

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, state):
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA schema mismatch")
        self.updates = state["updates"]
        for name, tensor in state["shadow"].items():
            self.shadow[name].copy_(tensor)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def load_cifar(data_dir, *, download=False):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=str(data_dir), train=True, download=download)
    if len(dataset.data) != 50000:
        raise ValueError("expected all 50000 official CIFAR-10 training images")
    data = dataset.data.copy(order="C")
    digest = hashlib.sha256(memoryview(data)).hexdigest()
    # Labels are deliberately not returned or used.
    return torch.from_numpy(data).permute(0, 3, 1, 2).contiguous(), digest


def image_batch(pixels, config, step, device):
    t = config["training"]
    generator = step_generator(t["seed"], step, pixels.device, stream=1)
    indices = torch.randint(len(pixels), (t["batch_size"],), device=pixels.device, generator=generator)
    images = pixels[indices].to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
    if t["horizontal_flip"]:
        flip_generator = step_generator(t["seed"], step, device, stream=2)
        flip = torch.rand((len(images), 1, 1, 1), device=device, generator=flip_generator) < .5
        images = torch.where(flip, images.flip(-1), images)
    return images


def train_step(model, images, optimizer, ema, config, step, *, collect_diagnostics=True):
    device = images.device
    generator = step_generator(config["training"]["seed"], step, device, stream=3)
    batch = conditional_path(images, generator=generator)
    rate = learning_rate(step, config)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(batch.image, batch.time)
        loss = velocity_loss(prediction, batch)
    loss.backward()
    # A single joint safety synchronization, before either optimizer or EMA update.
    # Clipping nonfinite gradients is harmless here: we abort without using them.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"],
                                        error_if_nonfinite=False, foreach=device.type == "cuda")
    if not bool(torch.isfinite(loss.detach()) & torch.isfinite(norm)):
        raise FloatingPointError("nonfinite CFM loss or gradient norm; update rejected")
    optimizer.step()
    ema.update(model)
    metrics = {"loss": loss.detach(), "gradient_norm": norm.detach(), "learning_rate": rate}
    if collect_diagnostics:
        metrics.update(target_rms=batch.target.square().mean().sqrt().detach(),
                       velocity_rms=prediction.detach().float().square().mean().sqrt())
    return metrics
