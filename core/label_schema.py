"""Workspace label names, stable IDs, colors and reversible target removal."""

from copy import deepcopy
import hashlib
import json
import re
from datetime import datetime, timezone

from core.annotations import CLASS_NAMES
from core.io import ReviewError, atomic_bytes, json_bytes

FILE_NAME = "label_schema.json"
BOX_COLORS = ["#fc927d", "#76baff", "#d7b7ff", "#ffe180"]
DEFAULT = {
    "version": 1,
    "detection": [
        {"id": i, "name": name, "color": BOX_COLORS[i], "active": True}
        for i, name in enumerate(CLASS_NAMES)
    ],
    "segmentation": [
        {"id": 0, "name": "비도로", "color": "#37455a", "active": True},
        {"id": 1, "name": "도로", "color": "#28da8a", "active": True},
        {"id": 255, "name": "불확실", "color": "#ffb335", "active": True},
    ],
}


def load_schema(root):
    path = root / FILE_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else deepcopy(DEFAULT)


def schema_revision(schema):
    return hashlib.sha256(json_bytes(schema)).hexdigest()


def update_schema(root, payload):
    """Caller must hold the workspace generation lock. No label pixels are rewritten."""
    if not isinstance(payload, dict):
        raise ReviewError("클래스 설정 JSON 객체가 필요합니다.")
    old = load_schema(root)
    if payload.get("revision") != schema_revision(old):
        raise ReviewError("다른 창에서 클래스 설정을 변경했습니다. 다시 열어주세요.", 409)
    result = {"version": 1}
    for kind, limit in [("detection", 4095), ("segmentation", 255)]:
        submitted = payload.get(kind)
        if not isinstance(submitted, list) or len(submitted) > limit + 1:
            raise ReviewError("클래스 목록 크기가 올바르지 않습니다.")
        previous = {item["id"]: item for item in old[kind]}
        seen, items = set(), []
        allocated = set(previous)
        for entry in submitted:
            if not isinstance(entry, dict):
                raise ReviewError("잘못된 클래스 형식입니다.")
            label_id = entry.get("id")
            if label_id is None:
                candidates = range(2, 255) if kind == "segmentation" else range(limit + 1)
                label_id = next((i for i in candidates if i not in allocated), None)
                if label_id is None:
                    raise ReviewError("사용할 수 있는 클래스 ID가 없습니다.")
                allocated.add(label_id)
            elif type(label_id) is not int or label_id not in previous:
                raise ReviewError("기존 ID는 변경할 수 없습니다. 새 클래스는 ID 없이 추가하세요.")
            if label_id in seen:
                raise ReviewError("중복 클래스 ID입니다.")
            seen.add(label_id)
            name, color, active = entry.get("name"), entry.get("color"), entry.get("active", True)
            if (
                not isinstance(name, str)
                or not name.strip()
                or len(name.strip()) > 80
                or any(ord(c) < 32 for c in name)
            ):
                raise ReviewError("클래스 이름은 줄바꿈 없이 1~80자로 입력하세요.")
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ReviewError("색상은 #RRGGBB 형식이어야 합니다.")
            if type(active) is not bool:
                raise ReviewError("사용 여부는 true/false여야 합니다.")
            if kind == "segmentation" and label_id in (0, 255) and not active:
                raise ReviewError("배경(ID 0)과 ignore(ID 255)는 유지해야 합니다.")
            items.append(
                {"id": label_id, "name": name.strip(), "color": color.lower(), "active": active}
            )
        for label_id, entry in previous.items():
            if label_id not in seen:
                if kind == "segmentation" and label_id in (0, 255):
                    raise ReviewError("배경(ID 0)과 ignore(ID 255)는 유지해야 합니다.")
                items.append(dict(entry, active=False))
        names = [i["name"].casefold() for i in items if i["active"]]
        if len(names) != len(set(names)):
            raise ReviewError("사용 중인 클래스 이름이 중복됩니다.")
        result[kind] = sorted(items, key=lambda item: item["id"])
    # Persist previous names/IDs/colors so settings can be recovered separately from labels.
    history = root / "_schema_history"
    history.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    atomic_bytes(history / (stamp + ".json"), json_bytes(old))
    atomic_bytes(root / FILE_NAME, json_bytes(result))
    return result


def public_schema(root):
    schema = load_schema(root)
    return {
        "label_schema": schema,
        "schema_revision": schema_revision(schema),
        "classes": {item["id"]: item["name"] for item in schema["detection"]},
    }


def require_teacher_schema(root):
    """The bundled automatic model retains its trained class mapping."""
    schema = load_schema(root)
    for kind in ("detection", "segmentation"):
        by_id = {item["id"]: item for item in schema[kind]}
        for item in DEFAULT[kind]:
            if item["id"] not in by_id or by_id[item["id"]]["name"] != item["name"]:
                raise ReviewError(
                    "기존 자동 라벨 모델의 클래스 매핑과 다릅니다. 새 출력 폴더를 사용하세요."
                )
