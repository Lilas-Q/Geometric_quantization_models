"""The completed 1.25M recipe, excluding later experimental extensions.

Updates 1..400000 use the original OneCycle schedule. Update 400001 halves
the inherited LR; cosine decay ends at 4e-9 on update 1250000. Adam beta1
continues along the original global-step OneCycle trajectory throughout.
"""

import copy
import math
from .recipe import scheduled_values as original_values, make_scheduler as original_scheduler

ANCHOR = 400000
TOTAL = 1250000
FLOOR = 4e-9
POLICY = dict(
    schema="lhfm_v_onecycle_then_half_lr_cosine_v1",
    anchor_step=ANCHOR, last_update=TOTAL, first_cosine_update=ANCHOR + 1,
    initial_lr_factor=0.5, minimum_lr=FLOOR,
    beta1="unchanged_original_onecycle_global_step_trajectory",
)


def validate_schedule_config(config):
    if config.get("optimization_policy") != POLICY:
        raise ValueError("explicit LHFM-V OneCycle/cosine policy required")
    if config["stop_steps"] != TOTAL or config["min_learning_rate"] != FLOOR:
        raise ValueError("LHFM-V publication stops at 1250000 updates")


def scheduled_values(update_index, config):
    old_lr, beta1 = original_values(update_index, config)
    if update_index <= ANCHOR:
        return old_lr, beta1
    initial_lr = 0.5 * original_values(ANCHOR + 1, config)[0]
    fraction = min(1.0, (update_index - (ANCHOR + 1)) / (TOTAL - ANCHOR - 1))
    return FLOOR + (initial_lr - FLOOR) * 0.5 * (1.0 + math.cos(math.pi * fraction)), beta1


class TrainingScheduler:
    def __init__(self, optimizer, config):
        validate_schedule_config(config)
        self.optimizer, self.config = optimizer, config
        self.last_epoch, self._step_count = 0, 1
        self._apply()

    def _apply(self):
        lr, beta1 = scheduled_values(self.last_epoch + 1, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["betas"] = (beta1, group["betas"][1])
        self._last_lr = [lr for _ in self.optimizer.param_groups]

    def step(self):
        if self.last_epoch >= TOTAL:
            raise ValueError("LHFM-V schedule exhausted")
        self.last_epoch += 1
        self._step_count += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return dict(policy=copy.deepcopy(POLICY), last_epoch=self.last_epoch,
                    _step_count=self._step_count, _last_lr=list(self._last_lr))

    def load_state_dict(self, state):
        if (state.keys() != self.state_dict().keys() or state["policy"] != POLICY
                or type(state["last_epoch"]) is not int
                or not 0 <= state["last_epoch"] <= TOTAL
                or state["_step_count"] != state["last_epoch"] + 1
                or state["_last_lr"] != [scheduled_values(state["last_epoch"] + 1, self.config)[0]
                                         for _ in self.optimizer.param_groups]):
            raise ValueError("invalid LHFM-V scheduler state")
        self.last_epoch, self._step_count = state["last_epoch"], state["_step_count"]
        self._last_lr = list(state["_last_lr"])


def make_scheduler(optimizer, config):
    # Retain the original optimizer-group metadata as well as numerical values.
    original_scheduler(optimizer, config)
    return TrainingScheduler(optimizer, config)
