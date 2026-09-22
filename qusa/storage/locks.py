"""Cross-process, per-ticker coordination for local file artifacts."""

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


class TickerBusyError(RuntimeError):
    """Raised when another local process owns a ticker write lock."""


@contextmanager
def ticker_lock(lock_root, ticker, timeout_seconds=5.0, poll_seconds=0.05):
    """Acquire a bounded advisory lock scoped to one ticker."""

    root = Path(lock_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".{ticker.upper()}.lock"
    with path.open("a+") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TickerBusyError(f"Ticker {ticker.upper()} is busy; try again shortly.")
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
