"""Read-only image discovery and separate, persistent annotation workspaces."""

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from core.dataset import annotation_paths, read_manifest
from core.io import ReviewError, atomic_bytes, inside, json_bytes
from core.label_schema import FILE_NAME, load_schema
from core.file_lock import exclusive_lock

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FIELDS = [
    "date",
    "split",
    "sequence",
    "domain",
    "image_file",
    "nir_file",
    "image_sha256",
    "image_stamp",
    "semseg_file",
    "instseg_file",
    "depth_file",
    "filtered_depth_file",
    "completed_depth_file",
    "mask_file",
    "camera_info_file",
    "pcd_file",
]
IGNORE = {
    "nir",
    "infrared",
    "labels",
    "annotations",
    "road",
    "road_probability",
    "segmentation",
    "masks",
    "previews",
    "pseudo_labels",
    "outputs",
    "depth",
    "depth_gt",
    "depth_gt_mask",
    "depth_gt_overlap_filtered",
    "depth_gt_near_completed",
}


def sha256(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def input_rows(dataset_root, output_root=None):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ReviewError(f"입력 폴더가 없습니다: {root}")
    if (root / "manifest.csv").is_file():
        rows = read_manifest(root)
    else:
        output = Path(output_root).resolve() if output_root else None
        images = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
            and not any(part.lower() in IGNORE for part in p.relative_to(root).parts[:-1])
            and not (output and p.resolve().is_relative_to(output))
        )
        rows = []
        for index, file in enumerate(images, 1):
            relative = file.relative_to(root)
            parts = relative.parts
            split = next((p for p in parts[:-1] if p in ("train", "test")), "train")
            sequence = (
                file.parent.parent.relative_to(root).as_posix()
                if file.parent != root and file.parent.name.lower() in ("images", "rgb", "color")
                else file.parent.relative_to(root).as_posix()
            )
            sequence = sequence.replace("/", "__") if sequence != "." else "images"
            nir = ""
            base = (
                file.parent.parent
                if file.parent != root and file.parent.name.lower() in ("images", "rgb", "color")
                else root
            )
            for folder in ("nir", "infrared"):
                candidates = [
                    p
                    for p in (base / folder).glob(file.stem + ".*")
                    if p.suffix.lower() in EXTENSIONS
                ]
                if len(candidates) > 1:
                    raise ReviewError(f"NIR 파일이 여러 개입니다: {file.name}")
                if candidates:
                    nir = candidates[0].relative_to(root).as_posix()
                    break
            row = {k: "" for k in FIELDS}
            row.update(
                date="dataset",
                split=split,
                sequence=sequence,
                domain="images",
                image_file=relative.as_posix(),
                nir_file=nir,
                image_stamp=str(index),
            )
            rows.append(row)
    identities = set()
    for row in rows:
        row.setdefault("date", "dataset")
        row.setdefault("split", "train")
        row.setdefault("sequence", "images")
        for key in FIELDS:
            row.setdefault(key, "")
        if row["split"] not in ("train", "test"):
            raise ReviewError("split은 train 또는 test여야 합니다.")
        file = inside(root, row["image_file"])
        if not file.is_file():
            raise ReviewError(f"RGB 파일이 없습니다: {file}")
        # Keep supplied hashes for existing manifest compatibility; new images
        # receive their actual hashes, never an identity derived from the filename.
        if not row["image_sha256"]:
            row["image_sha256"] = sha256(file)
        if row["nir_file"] and not inside(root, row["nir_file"]).is_file():
            raise ReviewError(f"NIR 파일이 없습니다: {row['nir_file']}")
        if not row["image_stamp"]:
            row["image_stamp"] = str(len(identities) + 1)
        for key in ("date", "sequence"):
            value = row[key]
            if not value or value in (".", "..") or any(c in value for c in "/\\\0"):
                raise ReviewError(f"{key}에는 폴더 경로를 넣을 수 없습니다.")
        identity = annotation_paths(row)["annotation_file"]
        if identity in identities:
            raise ReviewError(f"중복 라벨 출력 경로입니다: {identity}")
        identities.add(identity)
    return rows


def prepare_workspace(dataset_root, output_root, initialize=True, allow_empty=False):
    source = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if source == output or source.is_relative_to(output):
        raise ReviewError("입력과 출력은 서로 다른 폴더로 지정하세요.")
    rows = input_rows(source, output)
    if not rows and not allow_empty:
        raise ReviewError("입력 폴더에 RGB 이미지가 없습니다.")
    state = output / "workspace.json"
    if (
        state.exists()
        and Path(json.loads(state.read_text(encoding="utf-8"))["dataset_root"]).resolve() != source
    ):
        raise ReviewError(
            "이 출력 폴더는 다른 입력 데이터용입니다. 새 출력 폴더를 지정하세요.", 409
        )
    output.mkdir(parents=True, exist_ok=True)
    from contextlib import ExitStack

    with ExitStack() as stack:
        try:
            stack.enter_context(exclusive_lock(output / ".generation.lock"))
        except BlockingIOError as error:
            raise ReviewError("다른 작업이 라벨을 생성하거나 저장 중입니다.", 409) from error
        # Check again under the writer lock before writing workspace files.
        if (
            state.exists()
            and Path(json.loads(state.read_text(encoding="utf-8"))["dataset_root"]).resolve()
            != source
        ):
            raise ReviewError("이 출력 폴더는 다른 입력 데이터용입니다.", 409)
        return _write_workspace(source, output, rows, initialize)


def _write_workspace(source, output, rows, initialize):
    state = output / "workspace.json"
    if not (output / FILE_NAME).exists() and (source / FILE_NAME).is_file():
        atomic_bytes(output / FILE_NAME, json_bytes(load_schema(source)))
    manifest = output / "workspace_manifest.csv"
    fields = list(dict.fromkeys(FIELDS + [key for row in rows for key in row]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if initialize:
        for row in rows:
            paths = annotation_paths(row)
            annotation = inside(output, paths["annotation_file"])
            if annotation.exists():
                continue
            annotation.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(inside(source, row["image_file"])) as image:
                width, height = image.size
            record = dict(
                row,
                **{k: v for k, v in paths.items() if k not in row},
                width=width,
                height=height,
                review_status="pending",
                detections=[],
                detection_count=0,
                road_fraction=0,
                uncertain_fraction=0.0,
                label_source="manual",
                generation_signature="manual-import",
            )
            # Opening an exported reviewed dataset in a separate output copies its
            # existing annotations; all subsequent editing stays in that output.
            for key in (
                "detection_file",
                "road_file",
                "annotation_file",
                "road_probability_file",
                "preview_file",
            ):
                target = inside(output, paths[key]) if paths.get(key) else None
                old = (
                    inside(source, paths[key])
                    if row.get("exported_reviewed") == "1" and paths.get(key)
                    else None
                )
                if target and old and old.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old, target)
            if not annotation.exists():
                label = inside(output, paths["detection_file"])
                label.parent.mkdir(parents=True, exist_ok=True)
                atomic_bytes(label, b"")
                mask = inside(output, paths["road_file"])
                mask.parent.mkdir(parents=True, exist_ok=True)
                png = io.BytesIO()
                Image.fromarray(np.zeros((height, width), np.uint8)).save(png, format="PNG")
                atomic_bytes(mask, png.getvalue())
                atomic_bytes(annotation, json_bytes(record))
    atomic_bytes(manifest, buffer.getvalue().encode("utf-8-sig"))
    atomic_bytes(
        state,
        json_bytes(
            dict(
                version=1,
                dataset_root=str(source),
                output_root=str(output),
                frames=len(rows),
                source_images_modified=False,
            )
        ),
    )
    return manifest, rows
