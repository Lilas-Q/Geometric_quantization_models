"""Finite-budget updates and full-state recovery for this new checkpoint family."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from .batch_guard import BatchGuard, FORWARD_ERRORS, NonfiniteGradient
from .execution import Execution, compiler_context
from .loss import forecast_loss
from .model import MODEL_ID
from .recipe import validate_recipe
from .schedule import scheduled_values, make_scheduler, validate_schedule_config
from .continuation import assess_history, validate_saved_control


FULL_CHECKPOINT_SCHEMA = "lhfm-v-training-v1"
EMA_CHECKPOINT_SCHEMA = "lhfm-v-ema-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Trainer:
    def __init__(self, model, config, *, binding):
        self.config = copy.deepcopy(validate_recipe(config))
        validate_schedule_config(config)
        for key, value in config["model_kwargs"].items():
            if json.loads(json.dumps(getattr(model, key))) != value:
                raise ValueError("model and training configuration disagree")
        self.binding = copy.deepcopy(binding)
        self.identity = canonical_hash(dict(config=self.config, model=model.config(), binding=binding))
        self.model = model
        execution_compile = binding.get("compile", config["compile"])
        if type(execution_compile) is not bool:
            raise ValueError("compile execution option must be boolean")
        self.execution = Execution(model, precision=config["precision"], compile=execution_compile,
                                   backend=config["compile_backend"],
                                   activation_checkpoint=config["activation_checkpoint"], supervise_plan=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                           betas=tuple(config["betas"]), eps=config["eps"],
                                           weight_decay=config["weight_decay"],
                                           fused=next(model.parameters()).device.type == "cuda")
        self.scheduler = make_scheduler(self.optimizer, self.config)
        self.ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self._bind_ema_views()
        self.step = self.sequence_cursor = 0
        self.batch_guard = BatchGuard()
        self.validation_control = []
        self.continuation_decision = assess_history([],self.config)

    def record_validation(self, history):
        if any(row.get('identity') != self.identity for row in history):
            raise ValueError('validation control identity mismatch')
        observations = [dict(step=row['step'],mse_raw=row['mse_raw']) for row in history]
        if observations[:len(self.validation_control)] != self.validation_control:
            raise ValueError('previous continuation observations changed')
        decision = validate_saved_control(observations,self.step,self.config)
        self.validation_control, self.continuation_decision = observations, decision
        return decision

    def _bind_ema_views(self):
        """Cache fixed-state tensor lists instead of rebuilding them every update.

        Only persistent state participates in EMA. Integer/bool buffers are
        copied rather than interpolated; coordinate caches registered with
        persistent=False never enter state_dict or these lists.
        """
        grouped = {}
        self._ema_copy_views = []
        for key, value in self.model.state_dict().items():
            value = value.detach()
            average = self.ema[key]
            if value.is_floating_point() or value.is_complex():
                targets, sources = grouped.setdefault((value.device, value.dtype), ([], []))
                targets.append(average)
                sources.append(value)
            else:
                self._ema_copy_views.append((average, value))
        self._ema_lerp_views = tuple(grouped.values())

    @torch.no_grad()
    def _update_ema(self):
        weight = 1 - self.config["ema_decay"]
        for averages, values in self._ema_lerp_views:
            # Same lerp update as v1, batched into foreach device kernels.
            torch._foreach_lerp_(averages, values, weight)
        for average, value in self._ema_copy_views:
            average.copy_(value)

    def update(self, microbatches):
        if self.batch_guard.stopped:
            raise RuntimeError("numerical batch circuit breaker is open; review required")
        if self.step >= self.continuation_decision["authorized_until"]:
            raise RuntimeError("validation decision required before another training update")
        if len(microbatches) != self.config["gradient_accumulation_steps"]:
            raise ValueError("incorrect number of microbatches")
        physical = self.config["physical_batch_size"]
        for offset, batch in enumerate(microbatches):
            if batch["context"].shape[0] != physical or batch["future"].shape[0] != physical:
                raise ValueError("physical batch changed")
            ids = torch.as_tensor(batch["sequence_id"]).cpu().tolist()
            start = self.sequence_cursor + offset * physical
            if ids != list(range(start, start + physical)):
                raise ValueError("sequence cursor mismatch or duplicated training exposure")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.execution.compiled and next(self.model.parameters()).device.type == 'cuda':
            # One generation covers the complete rollout and backward. Do not
            # mark individual halfsteps or free activations between frames.
            torch.compiler.cudagraph_mark_step_begin()
        rate = self.optimizer.param_groups[0]["lr"]
        beta1 = self.optimizer.param_groups[0]["betas"][0]
        if (rate, beta1) != scheduled_values(self.step + 1, self.config):
            raise ValueError("optimizer and OneCycle state disagree before update")
        rejection = None
        try:
            totals, metric_keys, norm = self._compute_gradients(microbatches)
        except NonfiniteGradient as exc:
            rejection = str(exc)
        except FloatingPointError as exc:
            if str(exc) not in FORWARD_ERRORS:
                raise
            rejection = str(exc)
        # This boundary is before any optimizer, scheduler, EMA or step update.
        # The failed helper frame/traceback is gone before clearing graph tensors.
        if rejection is not None:
            self.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            event = self.batch_guard.reject(self.step, self.sequence_cursor,
                self.config["effective_batch_size"], rejection)
            self.sequence_cursor = event["sequence_cursor"]
            return event
        self.optimizer.step()
        self.scheduler.step()
        self._update_ema()
        self.step += 1
        self.batch_guard.success(self.step)
        self.sequence_cursor += self.config["effective_batch_size"]
        # One batched host transfer for all metrics, including gradient norm.
        # Required pre-optimizer nonfinite-loss/gradient checks remain above.
        scalars = torch.cat((totals, norm.detach().reshape(1).to(torch.float64))).cpu().tolist()
        return dict(zip(metric_keys, scalars[:-1]), step=self.step,
                    sequence_cursor=self.sequence_cursor, learning_rate=rate,
                    gradient_norm=scalars[-1], beta1=beta1)

    def _compute_gradients(self, microbatches):
        totals = None
        metric_keys = None
        device = next(self.model.parameters()).device
        for batch in microbatches:
            context = batch["context"].to(device, non_blocking=True)
            target = batch["future"].to(device, non_blocking=True)
            with compiler_context(self.execution.compiled):
                prediction = self.execution(context, horizon=10)
                loss, values = forecast_loss(prediction, target,
                    source_weight=self.config["source_energy_weight"],
                    velocity_tv_weight=self.config["velocity_tv_weight"],
                    planning_weight=self.config["planning_loss_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite prediction loss; optimizer not advanced")
                (loss / len(microbatches)).backward()
            if metric_keys is None:
                metric_keys = tuple(values)
            elif tuple(values) != metric_keys:
                raise ValueError("loss metric coverage changed between microbatches")
            # Python float(value) used to synchronize once per metric. Keep
            # accumulation on device until the update has completed. Float64
            # preserves the previous Python-float averaging semantics.
            report = torch.stack([values[key].detach() for key in metric_keys]).to(torch.float64)
            report = report / len(microbatches)
            totals = report if totals is None else totals + report
        try:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                self.config["gradient_clip_global_norm"], error_if_nonfinite=True)
        except RuntimeError as exc:
            # PyTorch's documented nonfinite-norm rejection, not arbitrary CUDA errors.
            if str(exc).startswith("The total norm of order") and "non-finite" in str(exc):
                raise NonfiniteGradient("nonfinite gradient norm; optimizer not advanced") from exc
            raise
        return totals, metric_keys, norm

    def save(self, path):
        payload = dict(schema=FULL_CHECKPOINT_SCHEMA, model_id=MODEL_ID,
            identity=self.identity, config=self.config, model_config=self.model.config(),
            binding=self.binding, model=self.model.state_dict(), ema=self.ema,
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            step=self.step, sequence_cursor=self.sequence_cursor,
            batch_guard=self.batch_guard.state_dict(),
            validation_control=self.validation_control,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
                     if next(self.model.parameters()).device.type == "cuda" else [])
        atomic_save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("schema") != FULL_CHECKPOINT_SCHEMA
                or payload.get("model_id") != MODEL_ID or payload.get("identity") != self.identity):
            raise ValueError("checkpoint identity mismatch; other model families "
                             "cannot be resumed as the v17 Wide B16 OneCycle experiment")
        if (payload.get("config") != self.config or payload.get("binding") != self.binding
                or payload.get("model_config") != self.model.config()):
            raise ValueError("checkpoint configuration/binding changed")
        step, cursor = payload["step"], payload["sequence_cursor"]
        control = payload.get("validation_control")
        decision = validate_saved_control(control,step,self.config)
        if (type(step) is not int or type(cursor) is not int or not 0 <= step <= decision["authorized_until"]
                or cursor < 0):
            raise ValueError("checkpoint budget/cursor corruption")
        guard = BatchGuard.load(payload.get("batch_guard"), step, cursor, self.config["effective_batch_size"])
        expected = self.model.state_dict()
        for field in ("model", "ema"):
            tensors = payload.get(field, {})
            if tensors.keys() != expected.keys():
                raise ValueError("checkpoint model/EMA coverage mismatch")
            for key, reference in expected.items():
                tensor = tensors[key]
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape
                        or tensor.dtype != reference.dtype or not torch.isfinite(tensor).all()):
                    raise ValueError("invalid checkpoint model/EMA tensor")
        saved = payload.get("optimizer", {})
        current = self.optimizer.state_dict()
        groups = saved.get("param_groups", [])
        if len(groups) != len(current["param_groups"]):
            raise ValueError("optimizer group coverage mismatch")
        parameter_ids = []
        for group, reference in zip(groups, current["param_groups"]):
            if group.keys() != reference.keys():
                raise ValueError("optimizer group schema changed")
            for key, value in reference.items():
                if key not in ("lr", "betas") and group[key] != value:
                    raise ValueError("optimizer hyperparameters or parameter IDs changed")
            rate, beta1 = scheduled_values(step + 1, self.config)
            if group["lr"] != rate or tuple(group["betas"]) != (beta1, self.config["betas"][1]):
                raise ValueError("optimizer learning rate/beta1 disagrees with saved OneCycle state")
            parameter_ids.extend(group["params"])
        states = saved.get("state", {})
        if set(states) != (set(parameter_ids) if step else set()):
            raise ValueError("optimizer moments missing or duplicated")
        parameters = list(self.model.parameters())
        if step:
            for index, reference in zip(parameter_ids, parameters):
                moment = states[index]
                if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid AdamW moment schema")
                optimizer_step = moment["step"]
                if (not isinstance(optimizer_step, torch.Tensor) or optimizer_step.numel() != 1
                        or not torch.isfinite(optimizer_step).all() or float(optimizer_step) != step):
                    raise ValueError("AdamW moment step mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    value = moment[key]
                    if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                            or value.dtype != torch.float32 or not torch.isfinite(value).all()
                            or (key == "exp_avg_sq" and (value < 0).any())):
                        raise ValueError("invalid AdamW moment tensor")
        scheduler = payload.get("scheduler")
        expected_scheduler = self.scheduler.state_dict()
        if not isinstance(scheduler, dict) or scheduler.keys() != expected_scheduler.keys():
            raise ValueError("OneCycle scheduler state missing or incompatible")
        for key, value in expected_scheduler.items():
            if key not in ("last_epoch", "_step_count", "_last_lr") and scheduler[key] != value:
                raise ValueError("OneCycle scheduler configuration changed")
        if (scheduler["last_epoch"] != step or scheduler["_step_count"] != step + 1
                or scheduler["_last_lr"] != [group["lr"] for group in groups]):
            raise ValueError("OneCycle scheduler progress mismatch")
        rng = payload.get("torch_rng")
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.get_rng_state().shape:
            raise ValueError("invalid Torch RNG state")
        try:
            torch.Generator(device="cpu").set_state(rng)
        except RuntimeError as exc:
            raise ValueError("invalid Torch RNG contents") from exc
        cuda_rng = payload.get("cuda_rng")
        device = next(self.model.parameters()).device
        expected_cuda = torch.cuda.device_count() if device.type == "cuda" else 0
        if not isinstance(cuda_rng, list) or len(cuda_rng) != expected_cuda:
            raise ValueError("CUDA RNG coverage mismatch")
        for index, state in enumerate(cuda_rng):
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("invalid CUDA RNG tensor")
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state)
            except RuntimeError as exc:
                raise ValueError("invalid CUDA RNG contents") from exc
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        device = next(self.model.parameters()).device
        self.ema = {key: value.to(device) for key, value in payload["ema"].items()}
        self._bind_ema_views()
        self.step, self.sequence_cursor = step, cursor
        self.batch_guard = guard
        self.validation_control, self.continuation_decision = control, decision
        torch.set_rng_state(payload["torch_rng"])
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)

    def save_ema(self, path):
        atomic_save(dict(schema=EMA_CHECKPOINT_SCHEMA, model_id=MODEL_ID,
                         identity=self.identity, step=self.step, model_config=self.model.config(),
                         config=self.config, binding=self.binding, ema=self.ema), path)
