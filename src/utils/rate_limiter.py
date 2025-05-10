import threading
import time

class TokenBucket:
    def __init__(self, tokens: float, refill_time: float):
        self.capacity = tokens
        self._tokens = tokens
        self.refill_rate = tokens / refill_time
        self._lock = threading.Lock()
        self._last = time.time()

    def consume(self, amount: float = 1.0):
        with self._lock:
            now = time.time()
            delta = (now - self._last) * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + delta)
            self._last = now
            if self._tokens < amount:
                wait = (amount - self._tokens) / self.refill_rate
                time.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= amount