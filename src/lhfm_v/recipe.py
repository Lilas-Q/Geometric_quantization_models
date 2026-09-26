"""Finite full-rollout recipe with one-based optimizer-update learning rates.

Validation does not mutate the configuration or certify CUDA compatibility.
The physical batch is a candidate until a separate GPU qualification succeeds;
changing it must preserve the effective batch and produce a new frozen config.
"""

from __future__ import annotations

import math
import torch
from .model import MODEL_ID


SCHEMA = "lhfm_v17_wide_b16_onecycle_training_config_v1"
EFFECTIVE_BATCH = 16
REFERENCE_EPOCH_SEQUENCES = 10_000
# Select the optimization budget first. Reference epochs are a reporting unit
# for online-generated sequences, not the basis used to choose training length.
OPTIMIZER_STEPS = 1_250_000
SEQUENCE_EXPOSURES = OPTIMIZER_STEPS * EFFECTIVE_BATCH
REFERENCE_EPOCHS = SEQUENCE_EXPOSURES // REFERENCE_EPOCH_SEQUENCES
VALIDATION_STEPS = (5000, 10000, 20000) + tuple(range(40000, OPTIMIZER_STEPS, 40000)) + (OPTIMIZER_STEPS,)
INPUT_HASHES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "mnist_test_seq.npy": "c2a3e8d3939c001ab77bf543f130157424efa486bb4d0974d2cebb773b13f8b3",
}


def _require(config: dict, key: str):
    if key not in config:
        raise ValueError("missing recipe field: " + key)
    return config[key]


def _integer(value, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _equals(config: dict, key: str, expected):
    value = _require(config, key)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{key} must equal {expected!r}")
    return value


def validate_recipe(config: dict) -> dict:
    """Check the fixed 1.25M-update recipe; return the same dict."""
    if not isinstance(config, dict):
        raise ValueError("recipe must be a dictionary")
    _equals(config, "schema", SCHEMA)
    _equals(config, "experiment_id", "lhfm-moving-mnist-physical-fast-plan-v17-wide-b16-onecycle-2000ep")
    _equals(config, "physical_batch_size", 16)
    _equals(config, "gradient_accumulation_steps", 1)
    _equals(config, "effective_batch_size", EFFECTIVE_BATCH)
    _equals(config, "reference_epoch_sequences", REFERENCE_EPOCH_SEQUENCES)
    _equals(config, "budget_reference_epochs", REFERENCE_EPOCHS)
    _equals(config, "budget_sequence_exposures", SEQUENCE_EXPOSURES)
    for key in ('stop_steps','initial_budget_steps'):
        _equals(config,key,OPTIMIZER_STEPS)
    _equals(config,'max_optimizer_steps',OPTIMIZER_STEPS)
    _equals(config,'stop_optimizer_steps',OPTIMIZER_STEPS)
    _equals(config,'budget_scope','fixed_official_onecycle')
    micro = _integer(_require(config, "physical_batch_size"), "physical_batch_size")
    accumulation = _integer(_require(config, "gradient_accumulation_steps"),
                            "gradient_accumulation_steps")
    if micro * accumulation != EFFECTIVE_BATCH:
        raise ValueError("physical batch times accumulation must equal effective batch16")
    if (config["stop_steps"] * config["effective_batch_size"]
            != config["budget_sequence_exposures"]
            or config["budget_reference_epochs"] * config["reference_epoch_sequences"]
            != config["budget_sequence_exposures"]):
        raise ValueError("update budget, sequence exposures and reference epochs disagree")
    if _require(config, "physical_batch_status") not in (
            "candidate_pending_cuda_acceptance", "cuda_qualified"):
        raise ValueError("physical batch needs an explicit qualification status")

    _equals(config, "optimizer", "AdamW")
    peak = _positive(_require(config, "learning_rate"), "learning_rate")
    floor = _positive(_require(config, "min_learning_rate"), "min_learning_rate")
    if floor >= peak:
        raise ValueError("min_learning_rate must be below learning_rate")
    _positive(_require(config, "eps"), "eps")
    _positive(_require(config, "weight_decay"), "weight_decay")
    _positive(_require(config, "gradient_clip_global_norm"), "gradient_clip_global_norm")
    decay = _positive(_require(config, "ema_decay"), "ema_decay")
    if decay >= 1:
        raise ValueError("ema_decay must be below one")
    _equals(config, "ema_update_every_optimizer_steps", 1)
    betas = _require(config, "betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("betas must contain two finite values between zero and one")
    for index, beta in enumerate(betas):
        if _positive(beta, f"betas[{index}]") >= 1:
            raise ValueError("AdamW betas must be below one")
    warmup = _integer(_require(config, "warmup_optimizer_steps"), "warmup_optimizer_steps")
    if warmup >= config["stop_steps"]:
        raise ValueError("warmup must leave at least one cosine update")
    _equals(config, "lr_schedule", "pytorch_onecycle")
    _equals(config, "lr_index_convention", "optimizer_uses_onecycle_current_lr_then_scheduler_steps")
    for key, value in {
        "learning_rate": 0.001, "min_learning_rate": 4e-9, "weight_decay": 0.0001, "gradient_clip_global_norm": 1.0,
        "warmup_optimizer_steps": 375000, "onecycle_pct_start": 0.3,
        "onecycle_div_factor": 25.0, "onecycle_final_div_factor": 10000.0,
        "onecycle_cycle_momentum": True, "onecycle_base_momentum": 0.85,
        "onecycle_max_momentum": 0.95, "onecycle_anneal_strategy": "cos",
        "onecycle_three_phase": False,
    }.items():
        _equals(config, key, value)

    _equals(config, "precision", "bf16")
    for key in ("parameter_precision", "gradient_precision", "geometry_precision",
                "optimizer_state_precision", "ema_precision", "evaluation_precision"):
        _equals(config, key, "fp32")
    _equals(config, "tf32", False)
    _equals(config, "compile", True)
    if _require(config, "compile_backend") not in ("inductor", "aot_eager"):
        raise ValueError("only Inductor or explicitly selected CPU AOT acceptance is supported")
    _equals(config, "compile_mode", "default")
    _equals(config, "compile_fullgraph", True)
    _equals(config, "compile_dynamic", False)
    # Shared weights reduce memory pressure. Benchmark both policies without
    # changing batch, rollout, gradients or the physical update; freeze the
    # selected policy only after CUDA memory and backward qualification.
    if type(_require(config, "activation_checkpoint")) is not bool:
        raise ValueError("activation_checkpoint must be boolean")

    _equals(config, "rollout_frames", 10)
    _equals(config, "substeps", 2)
    _equals(config, "teacher_forcing", False)
    _equals(config, "curriculum", False)
    _equals(config, "loss", "full_rollout_pixel_mse_with_source_energy_velocity_tv_and_coarse_plan")
    _equals(config, "pixel_loss_reduction", "mean_over_batch_future_frames_channels_pixels")
    _equals(config, "source_energy_definition", "mean_r_squared")
    _equals(config, "velocity_tv_definition", "half_of_spatial_dx_abs_mean_plus_spatial_dy_abs_mean")
    _equals(config, "regularizer_reduction", "mean_over_forecast_frames_and_substeps")
    _positive(_require(config, "source_energy_weight"), "source_energy_weight")
    _positive(_require(config, "velocity_tv_weight"), "velocity_tv_weight")

    _equals(config, 'initialization', 'scratch_no_parent_weights')
    _equals(config, 'planning_loss_weight', 0.05)
    _equals(config, 'planning_target', 'avg_pool2d_future_frames_kernel4_stride4')
    _equals(config, 'planning_future_weights', 'equal')
    _equals(config, 'planning_head_at_inference', False)
    _equals(config, 'planning_loss_reduction', 'mean_over_batch_future_frames_channels_coarse_pixels')

    model = _require(config, "model_kwargs")
    if not isinstance(model, dict):
        raise ValueError("model_kwargs must contain the constructor configuration")
    if set(model) != {"widths", "context_frames", "substeps", "channels_last",
                       "latent_channels", "depth", "expansion", "spatial_groups",
                       "controller_hidden", "field_hidden", "mixing_rank", "memory_frames",
                       "memory_key_dim", "motion_depth", "motion_expansion", "motion_channels",
                       "memory_value_dim", "memory_heads", "coarse_channels", "coarse_depth", "plan_channels", "plan_depth",
                       "plan_expansion", "correction_hidden"}:
        raise ValueError("unexpected or missing physical-model constructor options")
    _equals(config, "model_id", MODEL_ID)
    # Small CPU fixtures may use fewer observed frames. A formal launch must
    # separately bind the frozen default model configuration and its hash.
    _integer(_require(model, "context_frames"), "model_kwargs.context_frames", minimum=2)
    _equals(model, "substeps", config["substeps"])
    widths = _require(model, "widths")
    if not isinstance(widths, (list, tuple)) or len(widths) != 3:
        raise ValueError("model widths must contain three positive integers")
    for index, width in enumerate(widths):
        _integer(width, f"model_kwargs.widths[{index}]", minimum=4)
    if type(_require(model, "channels_last")) is not bool:
        raise ValueError("model_kwargs.channels_last must be boolean")
    channels = _integer(_require(model, "latent_channels"), "model_kwargs.latent_channels", minimum=4)
    if channels % 2:
        raise ValueError("latent_channels must be even for rotation blocks")
    _integer(_require(model, "depth"), "model_kwargs.depth")
    _integer(_require(model, "expansion"), "model_kwargs.expansion")

    groups = _integer(_require(model, "spatial_groups"), "model_kwargs.spatial_groups")
    if channels % groups or (channels//groups) % 2:
        raise ValueError("even latent channels per spatial group required")
    for key in ("controller_hidden", "field_hidden", "mixing_rank", "memory_frames",
                "memory_key_dim", "motion_depth", "motion_expansion", "motion_channels",
                "memory_value_dim", "memory_heads", "coarse_channels", "coarse_depth"):
        _integer(_require(model,key), "model_kwargs."+key)
    for key in ('plan_channels','plan_depth','plan_expansion','correction_hidden'):
        _integer(_require(model,key), 'model_kwargs.'+key)
    if model["mixing_rank"] > channels:
        raise ValueError("mixing rank cannot exceed latent channels")
    if model["motion_channels"] > channels:
        raise ValueError("motion bottleneck cannot exceed latent channels")
    if model['memory_key_dim'] % model['memory_heads'] or model['memory_value_dim'] % model['memory_heads']:
        raise ValueError('key and value dimensions must be divisible by heads')
    if model['memory_value_dim'] > channels or model['coarse_channels'] > model['motion_channels']:
        raise ValueError('memory and coarse projections must remain bottlenecks')

    for key in ("seed", "data_seed", "validation_seed", "split_seed"):
        _integer(_require(config, key), key, minimum=0)
    _equals(config, "heldout_digits", 5000)
    _equals(config, "validation_count", 1024)
    _equals(config, "validation_fixed", True)
    _equals(config, "validation_interval", None)
    _equals(config, "validation_steps", list(VALIDATION_STEPS))
    _equals(config, "selection_metric", "validation_pixel_mse_raw")
    _equals(config, "test_count", 10_000)
    _equals(config, "test_policy", "once_after_final_update_using_best_validation_ema")
    _equals(config, "input_sha256", INPUT_HASHES)

    retention = _require(config, "retention")
    if not isinstance(retention, dict):
        raise ValueError("retention must be a dictionary")
    _equals(retention, "latest_full_state", True)
    _equals(retention, "keep_latest_after_finish", True)
    _equals(retention, "best_ema_count", 3)
    _equals(retention, "milestone_ema_steps", [40000, 200000, 400000, 800000, 1000000, OPTIMIZER_STEPS])
    _equals(retention, "preview_clips", 6)
    _equals(retention, "save_all_checkpoints", False)
    _equals(config, 'automatic_restart', False)
    _equals(config, 'automatic_continuation', False)
    _equals(config, 'automatic_extension', False)
    _equals(config, 'continuation', {
        "policy": "fixed_budget", "hard_step_cap": OPTIMIZER_STEPS,
        "stop_on_validation_rebound": False, "stop_on_plateau": False})
    return config


def make_scheduler(optimizer, config):
    """Same OneCycleLR constructor as official MMNIST, defaults made explicit."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["learning_rate"], total_steps=config["stop_steps"],
        pct_start=config["onecycle_pct_start"],
        anneal_strategy=config["onecycle_anneal_strategy"],
        cycle_momentum=config["onecycle_cycle_momentum"],
        base_momentum=config["onecycle_base_momentum"],
        max_momentum=config["onecycle_max_momentum"],
        div_factor=config["onecycle_div_factor"],
        final_div_factor=config["onecycle_final_div_factor"],
        three_phase=config["onecycle_three_phase"])


def scheduled_values(update_index, config):
    """LR/beta1 used by one-based update; T+1 is saved but never executed.

    PyTorch initializes scheduler index zero before update 1 and advances it
    after each optimizer update. After update T, its unused T+1 value is still
    saved exactly for complete optimizer/scheduler restoration.
    """
    _integer(update_index, "update_index")
    total = config["stop_steps"]
    if update_index > total + 1:
        raise ValueError("OneCycle schedule exhausted")
    index = update_index - 1
    boundary = config["onecycle_pct_start"] * total - 1
    initial = config["learning_rate"] / config["onecycle_div_factor"]
    minimum = initial / config["onecycle_final_div_factor"]
    if index <= boundary:
        fraction = index / boundary
        lr_start, lr_end = initial, config["learning_rate"]
        beta_start, beta_end = config["onecycle_max_momentum"], config["onecycle_base_momentum"]
    else:
        fraction = (index - boundary) / (total - 1 - boundary)
        lr_start, lr_end = config["learning_rate"], minimum
        beta_start, beta_end = config["onecycle_base_momentum"], config["onecycle_max_momentum"]
    def cosine(start, end):
        return end + (start - end) / 2.0 * (math.cos(math.pi * fraction) + 1)
    return cosine(lr_start, lr_end), cosine(beta_start, beta_end)


def learning_rate(update_index, config):
    return scheduled_values(update_index, config)[0]
