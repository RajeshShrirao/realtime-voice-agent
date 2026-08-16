import contextlib
import os
import time

import state


@contextlib.contextmanager
def measure_time(stats_key):
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000
    record_stat(stats_key, elapsed_ms)


class Silence:
    def __enter__(self):
        self._stdout = os.dup(1)
        self._stderr = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)

    def __exit__(self, *_):
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)
        os.close(self._devnull)
        os.close(self._stdout)
        os.close(self._stderr)


def record_stat(key, value=None, increment=None):
    with state.stats_lock:
        if value is not None:
            state.stats[key].append(value)
        elif increment is not None:
            state.stats[key] += increment
