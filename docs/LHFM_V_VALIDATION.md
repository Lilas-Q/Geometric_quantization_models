# LHFM-V release checks

Local checks on 2026-09-25 used Python 3.12 and PyTorch 2.14.0 on CPU.
The complete repository test suite passed: **27 tests**, including the
existing LHFM-I tests. No full training run, checkpoint re-evaluation or
CUDA acceptance test was performed for this standalone release.

## Comparison with the archived implementation

The reference source and configuration hashes are recorded in
[LHFM_V_PROVENANCE.json](LHFM_V_PROVENANCE.json). Eighteen reference modules
retain identical bytes. `engine.py` changes only the standalone checkpoint
schemas, explicit two-phase learning-rate scheduler, and eager/compiled
execution selection. The public alias, checkpoint loader and command-line
scripts are new wrappers. Original model identifiers and parameter names
remain unchanged.

The independent original and public package imports produced:

- Identical default initialization, RNG state and 18,637,493 parameters.
- Bitwise-equal full-size 64 × 64, ten-to-ten-frame forward outputs.
- Bitwise-equal losses, input gradients and all parameter gradients in a
  small fixture with nonzero transport/source correction heads.
- Bitwise-equal small-fixture AdamW update and EMA tensors.
- Exactly equal learning rates and Adam beta1 values at twelve indices,
  including both sides of the 400k transition and the 1.25M endpoint.

See the [machine-readable comparison](../results/lhfm_v_release_validation.json).
To repeat this check against your archived reference package:

```bash
python -m lhfm_v.scripts.check_reference \
  --reference-package /path/to/original/lhfm_physical \
  --reference-lr-policy /path/to/original/runtime/lr_policy.py
```

The original reference package is not duplicated in this repository.

## Standalone tests

Tests cover the transport sign, zero-transport affine update, saved-adjoint
gradients, model rollout and history gradients, scheduler cutoff/restoration,
exact complete-state training recovery, EMA loading, deterministic data,
metric sanity checks, CLI help, and a small train/resume/predict sequence.
Synthetic fixtures are not evidence of full-scale training convergence.

Release-asset checks verify all six preview hashes, result/checkpoint identity,
the 1.25M validation cutoff and spatial-sum metric conversion. The archived
test report checksum was verified before export; its overall and per-frame
values are retained without rounding in the JSON. README local links and
preview paths are also checked.

The archived test result and previews remain evidence from the original
completed run. CPU equivalence checks do not establish a CUDA speedup,
bitwise cross-hardware reproduction, or an independent quality improvement.
