"""Validated file paths and durable atomic writes shared by all services."""

import json


class ReviewError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def inside(root, relative):
    path = (root / relative).resolve()
    if not relative or not path.is_relative_to(root):
        raise ReviewError("허용되지 않은 파일 경로입니다.")
    return path


def atomic_bytes(path, data):
    temporary = path.with_name(path.name + ".review-tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
