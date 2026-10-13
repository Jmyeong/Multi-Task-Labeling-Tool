"""Bounded JSON-line reads with deadlines on both Windows and POSIX pipes.

os.read in a daemon thread works on subprocess pipes on both platforms and avoids
select()'s Windows socket-only restriction and buffered-stdin shutdown locks.
"""

import os
from queue import Empty, Full, Queue
import threading


class LineReader:
    def __init__(self, stream, limit=32 * 1024 * 1024):
        self.fd = stream.fileno()
        self.limit = limit
        self.lines = Queue(maxsize=1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True, name="sam-json-lines")
        self.thread.start()

    def _put(self, value):
        while not self.stopped.is_set():
            try:
                self.lines.put(value, timeout=0.1)
                return
            except Full:
                pass

    def _read(self):
        pending = bytearray()
        try:
            while not self.stopped.is_set():
                chunk = os.read(self.fd, 65536)
                if not chunk:
                    if pending:
                        raise RuntimeError("Worker ended with an incomplete JSON line")
                    self._put(None)
                    return
                pending.extend(chunk)
                while b"\n" in pending:
                    end = pending.index(b"\n") + 1
                    if end > self.limit:
                        raise RuntimeError("Worker JSON response is too large")
                    line = bytes(pending[:end])
                    del pending[:end]
                    self._put(line)
                    if self.stopped.is_set():
                        return
                if len(pending) > self.limit:
                    raise RuntimeError("Worker JSON response is too large")
        except (OSError, RuntimeError) as error:
            self._put(error)

    def read(self, timeout):
        try:
            result = self.lines.get(timeout=timeout)
        except Empty as error:
            raise TimeoutError("Timed out waiting for worker response") from error
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=0.5)
