"""Training-only rejection of specifically recognized pre-optimizer failures."""
from __future__ import annotations

import copy


POLICY = {
    "schema": "v17_training_batch_skip_v1",
    "max_consecutive": 3,
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
        self.recent_attempts = []
        self.last_event = None

    def _prune(self, attempt):
        self.recent_attempts = [n for n in self.recent_attempts
                                if n > attempt - POLICY["window_attempts"]]

    def success(self, completed_steps):
        self.consecutive = 0
        self._prune(completed_steps + self.skipped_batches)

    def reject(self, completed_steps, cursor, batch_size, reason):
        self.skipped_batches += 1
        self.consecutive += 1
        attempt = completed_steps + self.skipped_batches
        self._prune(attempt)
        self.recent_attempts.append(attempt)
        stop = (self.consecutive >= POLICY["max_consecutive"] or
                len(self.recent_attempts) >= POLICY["max_window_skips"])
        self.last_event = dict(skipped=True, step=completed_steps, attempt=attempt,
            sequence_id_first=cursor, sequence_id_last=cursor + batch_size - 1,
            sequence_cursor=cursor + batch_size, reason=reason,
            skipped_batches=self.skipped_batches, consecutive=self.consecutive,
            window_skips=len(self.recent_attempts), stop_required=stop,
            optimizer_advanced=False)
        return dict(self.last_event)

    @property
    def stopped(self):
        return self.last_event is not None and self.last_event["stop_required"]

    def state_dict(self):
        return copy.deepcopy(dict(policy=POLICY, skipped_batches=self.skipped_batches,
            consecutive=self.consecutive, recent_attempts=self.recent_attempts,
            last_event=self.last_event))

    @classmethod
    def load(cls, state, completed_steps, cursor, batch_size):
        guard = cls()
        if state is None:
            if cursor != completed_steps * batch_size:
                raise ValueError("legacy checkpoint cursor does not match successful steps")
            return guard
        if not isinstance(state, dict) or state.keys() != guard.state_dict().keys() or state["policy"] != POLICY:
            raise ValueError("batch guard policy/schema mismatch")
        total, consecutive, recent = state["skipped_batches"], state["consecutive"], state["recent_attempts"]
        attempt = completed_steps + total if type(total) is int else -1
        if (type(total) is not int or total < 0 or type(consecutive) is not int
                or not 0 <= consecutive <= min(total, POLICY["max_consecutive"])
                or cursor != (completed_steps + total) * batch_size
                or not isinstance(recent, list)
                or any(type(n) is not int or not max(0, attempt-POLICY["window_attempts"]) < n <= attempt for n in recent)
                or recent != sorted(set(recent)) or len(recent) > min(total, POLICY["max_window_skips"])):
            raise ValueError("batch guard counters/cursor corruption")
        event = state["last_event"]
        if total == 0:
            if event is not None or consecutive or recent:
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
                    or type(event.get("stop_required")) is not bool):
                raise ValueError("batch guard last event corrupted")
            if consecutive and (event["attempt"] != attempt or recent[-consecutive:] != list(range(attempt-consecutive+1, attempt+1))):
                raise ValueError("batch guard consecutive history corrupted")
            expected_stop = consecutive >= POLICY["max_consecutive"] or len(recent) >= POLICY["max_window_skips"]
            if event["stop_required"] != expected_stop:
                raise ValueError("batch guard circuit breaker mismatch")
        guard.skipped_batches, guard.consecutive = total, consecutive
        guard.recent_attempts, guard.last_event = list(recent), copy.deepcopy(event)
        return guard
