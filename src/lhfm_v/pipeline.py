"""Ordered input transfer with one CUDA batch of lookahead.

Only image tensors move to the accelerator. Absolute sequence IDs stay on the
CPU for the trainer's strict resume-cursor check. Construction is lazy so a
caller can include worker startup and the first transfer in its timing.
"""
from __future__ import annotations

from collections import deque

import torch


class DevicePrefetcher:
    """Transfer the next batch on a CUDA side stream while this batch computes.

    DataLoader remains responsible for parallel CPU generation and pinning.
    CPU mode and ``enabled=False`` use ordinary ordered transfer with no batch
    lookahead. No examples are shuffled, regenerated, or marked consumed here;
    only the trainer's successfully completed update advances its cursor.
    """

    def __init__(self, batches, device, *, enabled=True):
        self.batches = batches
        self.device = torch.device(device)
        self.active = enabled and self.device.type == "cuda"
        self._iterator = None
        self._stream = None
        self._next_batch = None
        self._next_ready = None
        self._sources = deque()
        self._exhausted = False
        self._closed = False

    def __iter__(self):
        return self

    def _move(self, batch):
        result = dict(batch)
        for key in ("context", "future"):
            result[key] = batch[key].to(self.device, non_blocking=True)
        return result

    def _preload(self):
        try:
            host_batch = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            self._next_batch = self._next_ready = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = self._move(host_batch)
            self._next_ready = torch.cuda.Event()
            self._next_ready.record(self._stream)
        # Keep pinned source storage alive until its asynchronous copy finishes.
        self._sources.append((self._next_ready, host_batch))
        while self._sources and self._sources[0][0].query():
            self._sources.popleft()

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._iterator is None:
            self._iterator = iter(self.batches)
            if self.active:
                self._stream = torch.cuda.Stream(device=self.device)
                self._preload()
        if not self.active:
            return self._move(next(self._iterator))
        if self._exhausted:
            raise StopIteration
        batch, ready = self._next_batch, self._next_ready
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(ready)
        for key in ("context", "future"):
            # Prevent CUDA's caching allocator from recycling side-stream
            # storage before the consuming training stream has finished.
            batch[key].record_stream(consumer)
        # Enqueue the lookahead AFTER the consumer's wait. The current update
        # depends on its own H2D copy, not the following batch's transfer.
        self._preload()
        return batch

    def close(self):
        if self._stream is not None:
            self._stream.synchronize()
        self._next_batch = self._next_ready = None
        self._sources.clear()
        self._iterator = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
