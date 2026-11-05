"""Bounded async boundary for the synchronous retrieval stack.

Keep SQL sessions inside the submitted callable. ContextVars are copied so
request embedding caches and retrieval traces survive the thread boundary.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from weakref import WeakKeyDictionary

from app.core.config import settings


class BlockingRetrievalPool:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="retrieval")
        self._slots = WeakKeyDictionary()

    async def run(self, function, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        # ASGI has one event loop per process. Tests may create successive loops.
        slots = self._slots.setdefault(loop, asyncio.Semaphore(self.max_workers))
        await slots.acquire()
        try:
            job = partial(function, *args, **kwargs)
            future = self._executor.submit(copy_context().run, job)
        except BaseException:
            slots.release()
            raise

        def completed(_):
            # Cancelling an await cannot stop a running HTTP/SQL call. Retain its
            # slot until the actual worker exits, rather than queueing more work.
            try:
                loop.call_soon_threadsafe(slots.release)
            except RuntimeError:
                pass  # The owning event loop has already closed.

        future.add_done_callback(completed)
        return await asyncio.wrap_future(future, loop=loop)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


retrieval_pool = BlockingRetrievalPool(settings.retrieval_max_workers)
run_retrieval = retrieval_pool.run
