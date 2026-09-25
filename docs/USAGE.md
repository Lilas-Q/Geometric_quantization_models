# LHFM-I usage

Run the commands below from the repository root.

## Install

Python 3.10 or later is required. Install a PyTorch build appropriate for your
hardware, then, from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Model API

```python
import torch
from lhfm_i import LHFM_I, conditional_path, velocity_loss

model = LHFM_I()  # 39,627,909 trainable parameters
data = torch.rand(2, 3, 32, 32) * 2 - 1  # example only; CIFAR-10 uses uint8/127.5 - 1
batch = conditional_path(data)
fields = model.fields(batch.image, batch.time)
loss = velocity_loss(fields.velocity, batch)
loss.backward()

# Standard ODE interface: one image velocity tensor.
velocity = model(batch.image, batch.time)
# fields.transport: [B,2,32,32]; fields.source: [B,3,32,32].
```

## Training

The provided configuration uses the paper's architecture and training recipe:

| Setting | LHFM-I |
| --- | --- |
| Dataset | CIFAR-10, 32 x 32 RGB, unconditional |
| Base channels / multipliers | 128 / (1, 2, 2, 2) |
| Residual blocks per level | 2 down / 3 up |
| Attention | Resolutions 16, 8 and bottleneck 4; 4 heads |
| Dropout | 0.1 |
| Parameters | 39,627,909 |
| Batch size / seed | 256 / 270829 |
| Optimizer | AdamW, betas (0.9, 0.999), zero weight decay, gradient clip 1 |
| Learning rate | 2,000-step warmup to 2.5e-4, cosine decay to 2e-5 over 400k updates |
| Default stopping point | 150k updates; the LR horizon remains 400k |
| EMA | Effective decay min(0.9999, (1 + updates) / (10 + updates)) |
| Precision | BF16 network on CUDA, FP32 image states and spatial derivatives |
| Data sampling | Step-seeded uniform sampling with replacement; random horizontal flip |

```bash
python -m lhfm_i.scripts.train --data-dir data --download --output-dir runs/lhfm-i --device cuda
python -m lhfm_i.scripts.train --data-dir data --output-dir runs/lhfm-i --device cuda --resume
```

`--download` explicitly permits downloading CIFAR-10. Training uses one device
and no gradient accumulation. `--max-steps` can stop earlier without changing
the learning-rate schedule. A complete rolling checkpoint is saved to
`runs/lhfm-i/latest.pt`, including model, optimizer, EMA and RNG state. Resume
checks the configuration, dataset digest, package source and runtime.
CPU mode is provided for small tests, not as the paper's training hardware.

## Sampling

```bash
python -m lhfm_i.scripts.sample --checkpoint runs/lhfm-i/latest.pt \
  --device cuda --count 16 --steps 128 --output samples/lhfm-i.png
```

Sampling uses EMA weights. Heun with 128 steps makes 256 network evaluations
per image. Intermediate image states are not clipped; clipping to [0,1] is
only applied when saving the preview. The preview script is not an FID evaluator.

## Paper result and release scope

The historical CIFAR-10 experiment at 150k updates reported **FID 3.5809**,
**KID x 1000 = 1.524 +/- 0.420**, and **IS = 9.295 +/- 0.087**, using 50,000
generated samples, 50,000 CIFAR-10 training reference images, EMA, and
Heun128 / 256 NFE. These are one-seed results; the uncertainties are KID
subset and IS split standard deviations, not variation across training seeds.

This is a standalone source release reorganized from that implementation.
The backbone, initialization, transport/source arithmetic, loss, sampling,
and update helpers are retained; the command-line runner and checkpoint
format are newly packaged. Historical checkpoints and experiment records are
not renamed or overwritten. The parameter state-dict keys are preserved.

**No pretrained weights or datasets are included.** This publication does
not represent a new training run or a fresh reproduction of the reported
FID. See [PROVENANCE.json](../PROVENANCE.json) and [VALIDATION.md](../VALIDATION.md)
for source provenance and the checks performed for this release.

The executable modules are in `src/lhfm_i/scripts/`. Source/package
reorganization does not change the model arithmetic or sampling settings.
