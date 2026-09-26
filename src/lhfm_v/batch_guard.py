"""Training-only rejection of specifically recognized pre-optimizer failures."""
from __future__ import annotations

import copy
from .legacy_batch_guard import BatchGuard as LegacyBatchGuard, POLICY as LEGACY_POLICY


POLICY = {
    "schema": "v17_training_batch_skip_v2",
    "max_consecutive": None,
    "window_attempts": 1000,
    "max_window_skips": 10,
    "scope": "training_only_pre_optimizer",
}
FORWARD_ERRORS = frozenset({
    "nonfinite frozen transport field",
    "frozen transport rate exceeds numerical work budget; no velocity clipping",
    "nonfinite prediction loss; optimizer not advanced",
})


class NonfiniteGradient(FloatingPointError):
    pass


class BatchGuard:
    def __init__(self):
        self.skipped_batches = 0
        self.consecutive = 0
        self.max_consecutive_observed = 0
        self.recent_attempts = []
        self.last_event = None
        self.migration = None

    def _prune(self, attempt):
        self.recent_attempts = [n for n in self.recent_attempts
                                if n > attempt - POLICY["window_attempts"]]

    def success(self, completed_steps):
        self.consecutive = 0
        self._prune(completed_steps + self.skipped_batches)

    def reject(self, completed_steps, cursor, batch_size, reason):
        self.skipped_batches += 1
        self.consecutive += 1
        self.max_consecutive_observed = max(self.max_consecutive_observed, self.consecutive)
        attempt = completed_steps + self.skipped_batches
        self._prune(attempt)
        self.recent_attempts.append(attempt)
        stop = len(self.recent_attempts) >= POLICY["max_window_skips"]
        self.last_event = dict(skipped=True, step=completed_steps, attempt=attempt,
            sequence_id_first=cursor, sequence_id_last=cursor + batch_size - 1,
            sequence_cursor=cursor + batch_size, reason=reason,
            skipped_batches=self.skipped_batches, consecutive=self.consecutive,
            max_consecutive_observed=self.max_consecutive_observed,
            window_skips=len(self.recent_attempts), stop_required=stop,
            optimizer_advanced=False)
        return dict(self.last_event)

    @property
    def stopped(self):
        return self.last_event is not None and self.last_event["stop_required"]

    def state_dict(self):
        return copy.deepcopy(dict(policy=POLICY, skipped_batches=self.skipped_batches,
            consecutive=self.consecutive, recent_attempts=self.recent_attempts,
            max_consecutive_observed=self.max_consecutive_observed,
            migration=self.migration, last_event=self.last_event))

    @classmethod
    def load(cls, state, completed_steps, cursor, batch_size):
        guard = cls()
        if state is None:
            if cursor != completed_steps * batch_size:
                raise ValueError("legacy checkpoint cursor does not match successful steps")
            return guard
        if isinstance(state, dict) and state.get("policy") == LEGACY_POLICY:
            old = LegacyBatchGuard.load(state, completed_steps, cursor, batch_size)
            # V1 cannot pass a streak of three. At its consecutive-stop point,
            # three is therefore the exact historical maximum, not a guess
            # based on the last event. Reject ambiguous legacy imports.
            if old.skipped_batches and old.consecutive != LEGACY_POLICY["max_consecutive"]:
                raise ValueError("legacy migration requires the consecutive-stop checkpoint or an empty history")
            migrated = guard.state_dict()
            for key in ("skipped_batches", "consecutive", "recent_attempts", "last_event"):
                migrated[key] = copy.deepcopy(state[key])
            migrated["max_consecutive_observed"] = old.consecutive
            migrated["migration"] = dict(from_policy=LEGACY_POLICY["schema"],
                at_step=completed_steps, at_cursor=cursor,
                previous_stop_required=old.stopped,
                history_max_basis="v1_consecutive_stop_bound" if old.skipped_batches else "empty_history")
            if migrated["last_event"] is not None:
                migrated["last_event"]["max_consecutive_observed"] = old.consecutive
                migrated["last_event"]["stop_required"] = len(old.recent_attempts) >= POLICY["max_window_skips"]
            return cls.load(migrated, completed_steps, cursor, batch_size)
        if not isinstance(state, dict) or state.keys() != guard.state_dict().keys() or state["policy"] != POLICY:
            raise ValueError("batch guard policy/schema mismatch")
        total, consecutive, recent = state["skipped_batches"], state["consecutive"], state["recent_attempts"]
        maximum = state["max_consecutive_observed"]
        attempt = completed_steps + total if type(total) is int else -1
        if (type(total) is not int or total < 0 or type(consecutive) is not int
                or not 0 <= consecutive <= total
                or type(maximum) is not int or not consecutive <= maximum <= total
                or cursor != (completed_steps + total) * batch_size
                or not isinstance(recent, list)
                or any(type(n) is not int or not max(0, attempt-POLICY["window_attempts"]) < n <= attempt for n in recent)
                or recent != sorted(set(recent)) or len(recent) > min(total, POLICY["max_window_skips"])):
            raise ValueError("batch guard counters/cursor corruption")
        event = state["last_event"]
        if total == 0:
            if event is not None or consecutive or recent or maximum:
                raise ValueError("empty batch guard state corrupted")
        else:
            if (not isinstance(event, dict) or event.get("skipped_batches") != total
                    or type(event.get("step")) is not int or not 0 <= event["step"] <= completed_steps
                    or event.get("attempt") != event["step"] + total
                    or event.get("sequence_cursor") != event["attempt"] * batch_size
                    or event.get("sequence_id_first") != (event["attempt"]-1)*batch_size
                    or event.get("sequence_id_last") != event["sequence_cursor"]-1
                    or event.get("optimizer_advanced") is not False
                    or event.get("skipped") is not True
                    or event.get("max_consecutive_observed") != maximum
                    or type(event.get("stop_required")) is not bool):
                raise ValueError("batch guard last event corrupted")
            if consecutive and (event["attempt"] != attempt or recent[-consecutive:] != list(range(attempt-consecutive+1, attempt+1))):
                raise ValueError("batch guard consecutive history corrupted")
            expected_stop = len(recent) >= POLICY["max_window_skips"]
            if event["stop_required"] != expected_stop:
                raise ValueError("batch guard circuit breaker mismatch")
        guard.skipped_batches, guard.consecutive = total, consecutive
        guard.max_consecutive_observed = maximum
        migration = state["migration"]
        if migration is not None and (not isinstance(migration, dict)
                or migration.get("from_policy") != LEGACY_POLICY["schema"]
                or type(migration.get("at_step")) is not int
                or not 0 <= migration["at_step"] <= completed_steps
                or type(migration.get("at_cursor")) is not int
                or not 0 <= migration["at_cursor"] <= cursor):
            raise ValueError("invalid batch guard migration provenance")
        guard.migration = copy.deepcopy(migration)
        guard.recent_attempts, guard.last_event = list(recent), copy.deepcopy(event)
        return guard
