# LHFM-V

LHFM-V is a deterministic video prediction model: ten observed grayscale
64 × 64 Moving MNIST frames condition ten future frames. It has **18,637,493
parameters**, including its training-only coarse prediction head.

This release contains the implementation, configuration, archived results and
previews of the completed **1,250,000-update** run. It excludes subsequent
experimental extensions. Pretrained weights and datasets are not included.
The reported test results are from that archived run, not a new reproduction
using these standalone scripts.

## Installation

Use Python 3.10–3.12 for the pinned video-evaluation dependencies
(NumPy 1.26.4 and SciPy 1.15.3); release tests used Python 3.12.
From the repository root:

```bash
python -m pip install -e '.[video,test]'
python -m pytest -q
```

See [release checks and limitations](LHFM_V_VALIDATION.md) for validation details.

The existing distribution name remains `lhfm-i`; installing it provides both
`lhfm_i` and `lhfm_v`. The package was checked on Python 3.12 / PyTorch 2.14.0
on CPU. Default training targets a BF16-capable CUDA GPU. The original CUDA
execution code is retained, but this publication has not been requalified on
CUDA. Compiler internals are PyTorch-version-sensitive; `--no-compile` selects
eager training. This is a source release, not a promise of identical training
trajectories across hardware or PyTorch versions.

## Model and notation

```python
import torch
from lhfm_v import LHFM_V, forecast_loss

model = LHFM_V()
context = torch.rand(1, 10, 1, 64, 64)  # example input; values in [0, 1]
with torch.no_grad():
    prediction = model.eval()(context).prediction  # [1, 10, 1, 64, 64]

# During training only:
model.train()
target = torch.rand(1, 10, 1, 64, 64)   # replace with actual future frames
forecast = model(context, supervise_plan=True)
loss, diagnostics = forecast_loss(forecast, target)
```

The history encoder uses widths 64/128/448 and a 384-channel latent state.
An observed-history planner produces ten future features; a recurrent latent
state, multiscale motion modules and a six-slot attention memory produce
state-dependent transport and source fields. It does not use ConvLSTM or
ConvGRU cells. Future frames are used only as training targets, never as inputs
to the rollout. The additional coarse prediction head is training-only.

Let $s$ be video time. The semi-discrete image dynamics are

```math
\frac{dJ_s}{ds}=A(w_s)J_s+r_s^{\mathrm{net}},
\qquad w_s=N u_s.
```

Here $u_s$ uses normalized spatial coordinates and $w_s$ is measured in pixels
per frame. **The reference code calls the pixel-unit field `u`** (including
`Forecast.u_rms`); it corresponds to $w_s$ in this notation, not $u_s$.
`r` denotes the numerical source array, in normalized intensity per frame.
`upwind_generator(J, w)` implements $A(w)J$, with the sign of
$-w\cdot\nabla J$, using a second-order upwind stencil and zero exterior
values. It is distinct from the DCT spatial derivative used by LHFM-I.

Each half-frame step freezes both fields and approximates the affine ODE
solution shown in the main README. The implementation uses a degree-24
shifted exponential series and adaptive internal subdivision (at most 128
pieces). The source is integrated through the same frozen transport operator;
it is not simply added as $h r$. Fields and latent states are subsequently
updated, so the complete predictor is nonlinear. The transport output has no
fixed amplitude bound; the numerical work-budget guard is not an output gate.

## Training recipe

The supervised objective is

```math
\mathcal{L}
= \operatorname{MSE}(\widehat J,J)
+10^{-3}\operatorname{mean}(r^2)
+10^{-4}\operatorname{TV}(w)
+0.05\operatorname{MSE}(\widehat J^{\mathrm{coarse}},J^{\mathrm{coarse}}).
```

Pixel MSE averages over batch, future frames, channels and pixels. Source
energy and transport total variation are averaged over all half-frame steps;
TV is half the sum of the mean absolute differences along the two spatial
axes. Coarse targets use 4 × 4 average pooling. There is no teacher forcing,
CFM noise-to-image objective, or separate Hamiltonian loss.

| Setting | Value |
| --- | --- |
| Updates / batch size | 1,250,000 / 16; no gradient accumulation |
| Optimizer | AdamW; weight decay $10^{-4}$; epsilon $10^{-8}$ |
| Gradient clipping / EMA decay | Global norm 1 / 0.999 |
| Updates 1–400,000 | OneCycle; initial LR $4\times10^{-5}$, peak $10^{-3}$ |
| Updates 400,001–1,250,000 | Half the inherited LR, then cosine decay to $4\times10^{-9}$ |
| Adam first-moment coefficient | Original global-step OneCycle trajectory, 0.95 → 0.85 → 0.95 |
| Training precision | BF16 neural layers; FP32 parameters, image dynamics, optimizer and EMA; TF32 off |
| Evaluation | FP32, validation-selected EMA |
| Seeds | Model/data 270829; digit split 271100; validation 271109 |

The first cosine update uses LR `0.0004989934970631772`. The standalone
scheduler composes the two phases explicitly. Historical full-state recovery
events and discarded attempts are not replayed; consequently a fresh run is
not claimed to be a bitwise reproduction of the archived experiment.
`optimization_policy` in the configuration is authoritative for the combined
schedule; legacy OneCycle identifiers are retained for source provenance.

Training sequences are generated online from MNIST training digits. A fixed
5,000-digit holdout supplies 1,024 validation sequences. Validation is performed
at 5k, 10k, 20k and every 40k thereafter, plus the 1.25M endpoint. The best EMA
is selected by raw validation MSE; the official test set is not used for
selection. Configuration and data checksums are in
[lhfm_v_moving_mnist.json](../configs/lhfm_v_moving_mnist.json).

Place the MNIST training image file `train-images-idx3-ubyte.gz` in your data
directory. Obtain the official `mnist_test_seq.npy` separately for final
evaluation. Scripts verify their hashes and do not download data automatically.

```bash
python -m lhfm_v.scripts.train \
  --config configs/lhfm_v_moving_mnist.json \
  --data-dir data/moving_mnist \
  --output-dir runs/lhfm-v \
  --device cuda
```

To pause earlier, add `--max-steps 1000`; this does **not** shorten the learning
rate schedule. Resume the same output directory with the same command plus
`--resume`. The checkpoint binds the configuration, source hashes, runtime,
optimizer, scheduler, EMA, RNG and sequence cursor. Changing these bindings
requires a separate run. `--device cpu --workers 0` is available for local
checks, not practical full-scale training.

The trainer retains an atomic rolling `latest.pt`, the best three validation
EMA checkpoints, configured milestone EMAs and `last-ema.pt`. It refuses
concurrent writers to the same output directory. Recognized non-finite or
excessive-work batches rejected before the optimizer step advance the data
cursor, but not the optimizer, scheduler or EMA; ten such rejections in a
1,000-attempt window stop the run for review. Validation and test failures are
not skipped. No automatic restart or post-1.25M extension is provided.

## Prediction and evaluation

Save observed frames as a NumPy array with shape `[B, 10, 1, 64, 64]` (or a
single `[10, 1, 64, 64]` clip), either uint8 or floating point in `[0, 1]`.

```bash
python -m lhfm_v.scripts.predict \
  --checkpoint runs/lhfm-v/last-ema.pt \
  --context data/context.npy \
  --output-dir samples/lhfm-v \
  --device cuda
```

The output contains raw `prediction.npy` and PNG strips. Clipping is used only
for display, not for the stored predictions. The loader accepts this package's
EMA/full checkpoints and the original LHFM-V EMA/full checkpoint formats up
to the publication cutoff. Original optimizer checkpoints cannot be resumed
by the new standalone trainer; they can be used for EMA inference.

Choose the EMA with the lowest `mse_raw` in `validation-history.json`, then run
the final test once:

```bash
python -m lhfm_v.scripts.evaluate \
  --checkpoint runs/lhfm-v/best-1250000-ema.pt \
  --test-file data/moving_mnist/mnist_test_seq.npy \
  --output-dir samples/lhfm-v-test \
  --device cuda
```

`best-1250000-ema.pt` is an example matching the archived run; a new run may
select a different step. Do not use test scores to select checkpoints.

## Archived results

| Model | EMA update | Parameters (M) | MSE ↓ | MAE ↓ | SSIM ↑ | PSNR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LHFM-V** | **1,250,000** | **18.637493** | **18.5812** | **62.5134** | **0.9582** | **24.6747** |

One completed run, evaluated on 10,000 official test sequences, with ten
observed and ten predicted frames. Inputs are uint8/255. MSE and MAE above
use raw predictions and sum over the 64 × 64 spatial grid, then average over
frames and sequences. Their pixel-mean values are 0.0045364187 and
0.0152620638. SSIM and PSNR use predictions clipped to `[0, 1]`, with targets
unchanged. SSIM follows the scikit-image 0.19.3 protocol: a 7 × 7 uniform
window, sample covariance, `data_range=2`, and K1 = 0.01, K2 = 0.03.
FP32 SSIM maps are averaged over the interior after excluding a three-pixel
border, with the single-channel mean cast to FP32 and frame/sequence scores
accumulated in FP64. PSNR is averaged per frame. See the
[protocol and source record](../results/lhfm_v_ssim_protocol.json).
Metric definitions and evaluation settings must be aligned before comparing
numbers from other implementations.

The selected EMA's validation MSE is **18.5236**, distinct from test MSE.
[Full test results](../results/lhfm_v_moving_mnist_1250k.json) include per-frame
metrics, clipped errors and the fraction of raw predictions outside `[0, 1]`.
[Validation history](../results/lhfm_v_validation_history.json) retains only
the selected training history through the cutoff.

Across 32 fixed validation trajectories, the RMS of the actual transport
term $A(w)J$ is **0.457097**, versus **0.0793675** for the source, giving a ratio
of **5.75925**. Both terms have intensity-per-frame units. This is not the
ratio of the raw transport field to the source; it is a descriptive diagnostic,
not a causal ablation. See [measurement protocol and results](../results/lhfm_v_transport_source_rms.json).

The six archived test previews are unchanged originals, selected as the first
six sequences, not by visual quality:
[0](../assets/lhfm-v-moving-mnist-1250k/sequence-00000.png),
[1](../assets/lhfm-v-moving-mnist-1250k/sequence-00001.png),
[2](../assets/lhfm-v-moving-mnist-1250k/sequence-00002.png),
[3](../assets/lhfm-v-moving-mnist-1250k/sequence-00003.png),
[4](../assets/lhfm-v-moving-mnist-1250k/sequence-00004.png),
[5](../assets/lhfm-v-moving-mnist-1250k/sequence-00005.png).
Each shows observed frames / future ground truth / predictions from top to
bottom. [Preview hashes](../assets/lhfm-v-moving-mnist-1250k.json) and
[source provenance](LHFM_V_PROVENANCE.json) bind these artifacts to the
1,250,000-step checkpoint.
