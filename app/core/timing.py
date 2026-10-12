"""StageTimer: wall-clock milliseconds per pipeline stage.

Component 6B · Plan: Part 6, step 2.

One timer follows a request through retrieval and generation. Because it lives outside the
pipeline functions, the stages that finished are still known when a later one fails, so a
failed chat is stored with the timings up to the failure.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - started) * 1000, 1)

    def finish(self) -> dict[str, float]:
        """Record `total` (since the timer was created) and return all timings."""
        self.timings_ms["total"] = round((time.perf_counter() - self._started) * 1000, 1)
        return self.timings_ms
