"""Thread-safe shared admission with loop-local asyncio semaphores."""
import asyncio
import threading
import time
import weakref
from contextlib import asynccontextmanager


class Admission:
    def __init__(self, concurrency: int, rate: float, burst: int):
        self.concurrency, self.rate, self.burst = concurrency, rate, burst
        self.tokens, self.updated, self.active = float(burst), time.monotonic(), 0
        self.lock = threading.Lock()
        self.semaphores = weakref.WeakKeyDictionary()

    @asynccontextmanager
    async def slot(self):
        loop = asyncio.get_running_loop()
        with self.lock:
            semaphore = self.semaphores.setdefault(loop, asyncio.Semaphore(self.concurrency))
        async with semaphore:
            admitted = False
            try:
                while not admitted:
                    with self.lock:
                        if self.active < self.concurrency:
                            self.active += 1
                            admitted = True
                    if not admitted:
                        await asyncio.sleep(.01)
                yield
            finally:
                if admitted:
                    with self.lock:
                        self.active -= 1

    async def token(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                delay = (1 - self.tokens) / self.rate
            await asyncio.sleep(delay)


_loop = None
_loop_lock = threading.Lock()

def sync_loop():
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="llm-sync-gateway", daemon=True).start()
        return _loop
