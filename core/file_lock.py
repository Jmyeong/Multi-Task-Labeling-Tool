"""Nonblocking interprocess locks: Windows CRT byte lock or POSIX flock.

Keep the same lock file and POSIX protocol as older Linux versions. Never unlink
or truncate lock files: every writer must contend on the same file/byte.
"""

from contextlib import contextmanager
import errno
import os
from pathlib import Path


@contextmanager
def exclusive_lock(path):
    """Hold an exclusive lock, or raise BlockingIOError immediately if busy."""
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            # CRT locking supports regions beyond EOF; no initialization write
            # is needed, which avoids writing into a region another process owns.
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK) or getattr(
                    error, "winerror", None
                ) in (32, 33):
                    raise BlockingIOError(
                        errno.EAGAIN, "Another process holds the workspace lock"
                    ) from error
                raise
            try:
                yield handle
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
