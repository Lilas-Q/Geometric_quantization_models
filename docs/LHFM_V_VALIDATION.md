# LHFM-V release checks

Local checks on 2026-09-25 used Python 3.12 and PyTorch 2.14.0 on CPU.
The complete repository test suite passed: **34 tests**, including the
existing LHFM-I tests. No full training run, checkpoint re-evaluation or
CUDA acceptance test was performed for this standalone release.

## Comparison with the archived implementation

The reference source and configuration hashes are recorded in
[LHFM_V_PROVENANCE.json](LHFM_V_PROVENANCE.json). Seventeen reference modules
retain identical bytes. `engine.py` changes only the standalone checkpoint
schemas, explicit two-phase learning-rate scheduler, and eager/compiled
execution selection. The public alias, checkpoint loader and command-line
scripts are new wrappers. `metrics.py` uses the SSIM protocol below; its
other metric reductions retain the reference formulas. Original model
identifiers and parameter names remain unchanged.

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
the 1.25M validation cutoff and spatial-sum metric conversion. Archived
metric-source checksums were verified before export; overall and per-frame
values are retained without rounding in the JSON. README local links and
preview paths are also checked.

## SSIM protocol checks

The evaluator implements the scikit-image 0.19.3 single-channel protocol
used by the referenced PredFormer evaluation code: FP32 images, prediction
clipping, a 7 × 7 uniform window, sample covariance, `data_range=2`, a
three-pixel border crop and equal frame/sequence weighting.
NumPy 1.26.4 and SciPy 1.15.3 are pinned in the video and test extras.
The adapted SSIM routine includes the scikit-image BSD license.

On 300 synthetic frames, including non-square images, clipped predictions,
and identical-image pairs, its scores are exactly equal to the original
scikit-image 0.19.3 source. The regression suite includes these independently
computed reference scores. All eight other metric tensors match the archived
implementation bitwise on a separate fixture. Independently summing the
100,000 archived test-frame SSIM scores yields 0.9582201841366291.
These are metric checks, not a new model evaluation by the standalone runner.
See [protocol and provenance](../results/lhfm_v_ssim_protocol.json).

The archived test result and previews remain evidence from the original
completed run. CPU equivalence checks do not establish a CUDA speedup,
bitwise cross-hardware reproduction, or an independent quality improvement.
