"""Bounded broadcast delivery and checkpoint-safe memory synchronization."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps


class SerializableRLock:
    def __init__(self):
        self._lock = threading.RLock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._lock = threading.RLock()


def synchronized_memory(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._memory_lock:
            return method(self, *args, **kwargs)

    return wrapped


_pools = {}
_pool_lock = threading.Lock()


def broadcast_map(fn, recipients, workers):
    """A process-wide pool avoids one nested thread pool per posting agent."""
    if workers <= 1:
        return [fn(recipient) for recipient in recipients]
    key = (os.getpid(), workers)
    with _pool_lock:
        if key not in _pools:
            _pools[key] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="post")
        pool = _pools[key]
    # map preserves recipient/message ordering; each caller waits for delivery.
    return list(pool.map(fn, recipients))
