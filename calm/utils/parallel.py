"""Sub-module for process-pool fan-out with prompt Ctrl+C shutdown.

The dataset generator, the grid evaluator, and the heatmap renderer all spread work
across processes and all need Ctrl+C to stop *now*. ``ProcessPoolExecutor``'s default
shutdown waits for in-flight tasks, and an in-flight task here can be a 30-minute
episode -- which reads as a hung terminal. This helper terminates the worker
processes directly, then re-raises so the caller can save partial results.
"""
from __future__ import annotations

import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Tuple


def ignore_sigint() -> None:
    """Pool-initializer helper: let the parent process own Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def imap_unordered(
    fn: Callable[..., Any],
    jobs: Iterable[Any],
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: Tuple[Any, ...] = (),
    star: bool = False,
) -> Iterator[Any]:
    """Yield each job's result in completion order.

    ``star=False`` calls ``fn(job)``; ``star=True`` calls ``fn(*job)`` for jobs that
    are argument tuples.

    ``workers <= 1`` runs everything in this process -- no pool, readable tracebacks,
    and shared-memory progress counters just work. ``initializer`` is only used for
    the pool; single-process callers do their own setup first, because the pool
    initializer also disables Ctrl+C, which the parent must keep.

    On Ctrl+C every worker is terminated before ``KeyboardInterrupt`` propagates.
    """
    # Normalize to argument tuples up front so there is one call shape below.
    args = [tuple(job) if star else (job,) for job in jobs]
    if workers <= 1:
        for job in args:
            yield fn(*job)
        return

    pool = ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs)
    try:
        for future in as_completed([pool.submit(fn, *job) for job in args]):
            yield future.result()
        pool.shutdown()
    except KeyboardInterrupt:
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
