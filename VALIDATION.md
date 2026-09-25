# Release validation

Validation date: September 25, 2026. Platform: macOS arm64, CPU, Python
3.12.14, PyTorch 2.14.0, torchvision 0.29.0, pytest 9.1.1.

## Standalone tests

`python -m pytest -q`: **16 passed**.

Checks cover the production parameter count, full-band derivative and coordinate
orientation, derivative precision under autocast, transport/source decomposition
and gradients, conditional path and loss, sampler NFE and nonfinite rejection,
the 400k learning-rate horizon, exact dropout-aware checkpoint resume, and
training/resume/sampling CLI execution on a tiny synthetic dataset.
All three packaged command-line `--help` entry points were tested with
`python -m lhfm_i.scripts.<name>`. The archived preview's hash and README
asset links are also checked.

## Preserved-implementation comparison

`lhfm_i.scripts.check_reference` compares this package with the archived experiment
implementation identified by the source hashes in `PROVENANCE.json`:

- Production-size model: **39,627,909 parameters**, identical parameter
  state-dict keys, bitwise-equal seeded initial weights and post-construction RNG.
- Production-size nonzero-head forward: velocity, transport, and source
  tensors are **bitwise equal** on the tested CPU input.
- Small model with dropout and a nonzero head: gradients, one AdamW update,
  and EMA tensors are **bitwise equal**.
- Small-model Heun sample: endpoints are **bitwise equal**, with matching NFE.

To repeat the archived-source audit, supply its `src` directory:

```bash
python -m lhfm_i.scripts.check_reference --reference-src /path/to/archived/src
```

The archived implementation is not bundled into this repository. The adapter in
the audit script uses its historical identifiers only for compatibility; the
public model name remains LHFM-I.

## Limits

These are CPU code and packaging checks, not a fresh CUDA acceptance run,
CIFAR-10 training run, or reproduction of FID/KID/IS. No pretrained weights,
datasets, private credentials, remote deployment settings, or historical training
states are distributed. A single archived generated-image preview is included;
its bytes match the original evaluation receipt, with provenance in
`assets/lhfm-i-cifar10-150k.json`. Numerical equality is reported only for the checks
above, not asserted across different devices or PyTorch versions.
