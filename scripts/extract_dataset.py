"""Export approved snapshots into a new, timestamp-ordered portable dataset."""

from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import shutil
import threading
import time
import uuid
from core.label_schema import load_schema, FILE_NAME


class ExportError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def component(value):
    if isinstance(value, str):
        value = value.strip()
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or any(c in value for c in "/\\\0")
        or any(ord(c) < 32 for c in value)
    ):
        raise ExportError("폴더 이름에는 /, \\, .. 또는 제어 문자를 사용할 수 없습니다.")
    return value.strip()


def stamp(row):
    try:
        value = Decimal(row.get("image_stamp", ""))
        if not value.is_finite():
            raise InvalidOperation
        return value
    except (InvalidOperation, ValueError):
        raise ExportError(f"정렬에 필요한 image_stamp가 없습니다: {row['image_file']}")


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


class DatasetExports:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.thread = None
        self.jobs = {}
        self.history = store.root / "_export_history"
        for path in sorted(self.history.glob("*.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ("queued", "snapshot", "planning", "copying"):
                job.update(
                    status="error",
                    error="서버가 중단되었습니다. 미완료 폴더를 확인하고 새 이름으로 다시 추출하세요.",
                )
                self._persist(job)
            self.jobs[job["id"]] = job

    def _persist(self, job):
        from core.io import atomic_bytes, json_bytes

        self.history.mkdir(exist_ok=True)
        atomic_bytes(self.history / (job["id"] + ".json"), json_bytes(job))

    def get(self, job_id=None):
        with self.lock:
            if job_id is None:
                job = max(self.jobs.values(), key=lambda j: j["created"], default=None)
            else:
                if job_id not in self.jobs:
                    raise ExportError("추출 작업을 찾을 수 없습니다.", 404)
                job = self.jobs[job_id]
            return copy.deepcopy(job)

    def start(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("parent"), str):
            raise ExportError("저장 경로와 새 폴더 이름을 입력하세요.")
        parent = Path(payload["parent"].strip()).expanduser()
        if not parent.is_absolute():
            raise ExportError("저장 경로는 서버의 절대 경로로 입력하세요.")
        parent = parent.resolve()
        target = parent / component(payload.get("name"))
        if target.exists() or target.is_symlink():
            raise ExportError("같은 이름의 폴더/파일이 이미 있습니다. 새 이름을 입력하세요.", 409)
        if (
            target == self.store.dataset_root
            or target.is_relative_to(self.store.root)
            or self.store.root.is_relative_to(target)
        ):
            raise ExportError("라벨 저장 폴더와 겹치는 위치에는 추출할 수 없습니다.")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ExportError("데이터 추출이 이미 진행 중입니다.", 409)
            job = {
                "id": uuid.uuid4().hex,
                "created": time.time(),
                "status": "queued",
                "output": str(target),
                "frames": 0,
                "files_done": 0,
                "files_total": 0,
                "bytes_done": 0,
                "bytes_total": 0,
                "current_file": "",
                "error": "",
            }
            self._persist(job)
            self.jobs[job["id"]] = job
            self.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self.thread.start()
            return copy.deepcopy(job)

    def _update(self, job, **values):
        with self.lock:
            job.update(values)
            self._persist(job)

    def snapshot(self):
        approved = []
        with self.store.lock, self.store.generation_lock():
            self.store.recover()
            self.store.refresh()
            for frame in self.store.frames.values():
                if frame["record"].get("review_status") != "approved":
                    continue
                data, revision = self.store.current(frame)
                record = json.loads(data["annotation_file"])
                if record.get("review_status") == "approved":
                    approved.append(
                        {
                            "label_schema": load_schema(self.store.root),
                            "row": dict(frame["source"]),
                            "record": record,
                            "paths": dict(frame["paths"]),
                            "data": data,
                            "revision": revision,
                        }
                    )
        if not approved:
            raise ExportError("저장된 승인 라벨이 없습니다. 먼저 프레임을 저장·승인하세요.")
        return approved

    def plan(self, snapshots):
        from core.io import inside, json_bytes

        groups = defaultdict(list)
        for item in snapshots:
            row = item["row"]
            split = row["split"]
            if split not in ("train", "test"):
                raise ExportError(f"지원하지 않는 split: {split}")
            domain = component(row.get("domain") or row["sequence"].split("__")[0])
            groups[split, domain].append(item)
        files, rows, frame_rows = {}, [], defaultdict(list)

        def add(relative, source, expected_hash=None):
            if relative in files:
                raise ExportError(f"출력 파일 충돌: {relative}")
            if isinstance(source, Path):
                if not source.is_file():
                    raise ExportError(f"원본 파일이 없습니다: {source}")
                stat = source.stat()
                signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
                size = stat.st_size
            else:
                signature, size = None, len(source)
            files[relative] = {
                "source": source,
                "size": size,
                "signature": signature,
                "expected_hash": expected_hash,
            }

        modalities = {
            "image_file": "images",
            "nir_file": "nir",
            "depth_file": "depth_gt",
            "mask_file": "depth_gt_mask",
            "filtered_depth_file": "depth_gt_overlap_filtered",
            "completed_depth_file": "depth_gt_near_completed",
            "pcd_file": "pcd",
            "camera_info_file": "camera_info",
            "semseg_file": "segmentation/source_class",
            "instseg_file": "segmentation/source_instance",
        }
        for (split, domain), items in sorted(groups.items()):
            items.sort(
                key=lambda i: (
                    stamp(i["row"]),
                    i["row"]["date"],
                    i["row"]["sequence"],
                    i["row"]["image_file"],
                )
            )
            prefix = f"{split}/{domain}"
            for index, item in enumerate(items, 1):
                source, record = item["row"], copy.deepcopy(item["record"])
                row = dict(source)
                stem = f"{index:06d}"
                path_map = {}
                if not source.get("image_file"):
                    raise ExportError(f"RGB 경로가 없는 프레임입니다: {source['image_file']}")
                for key, folder in modalities.items():
                    row[key] = ""
                    if source.get(key):
                        path = inside(self.store.dataset_root, source[key])
                        relative = f"{prefix}/{folder}/{stem}{path.suffix}"
                        add(
                            relative,
                            path,
                            source.get("image_sha256") if key == "image_file" else None,
                        )
                        row[key] = relative
                        path_map[source[key]] = relative
                        path_map[source[key].removeprefix(source["date"] + "/")] = relative
                # Keep any original labels separately; current reviewed binary Road is authoritative.
                row["source_semseg_file"], row["source_instseg_file"] = (
                    row["semseg_file"],
                    row["instseg_file"],
                )
                row["semseg_file"] = row["instseg_file"] = ""
                row.update(
                    detection_file=f"{prefix}/labels/{stem}.txt",
                    road_file=f"{prefix}/segmentation/road/{stem}.png",
                    annotation_file=f"{prefix}/annotations/{stem}.json",
                    road_probability_file="",
                    preview_file="",
                    review_status="approved",
                    label_source="reviewed",
                    exported_reviewed="1",
                    export_index=str(index),
                    domain=domain,
                    source_image_file=source["image_file"],
                    source_sequence=source["sequence"],
                    source_annotation_file=item["paths"]["annotation_file"],
                )
                add(row["detection_file"], item["data"]["detection_file"])
                add(row["road_file"], item["data"]["road_file"])
                probability_rel = item["paths"].get("road_probability_file")
                probability = inside(self.store.root, probability_rel) if probability_rel else None
                if probability and probability.is_file():
                    row["road_probability_file"] = f"{prefix}/road_probability/{stem}.png"
                    add(row["road_probability_file"], probability)
                # Depth masks and per-frame completion metadata are not all listed in manifest.csv.
                if source.get("filtered_depth_file"):
                    path = inside(self.store.dataset_root, source["filtered_depth_file"])
                    for folder in (
                        "depth_gt_overlap_filtered_mask",
                        "depth_gt_overlap_filtered_removed_mask",
                    ):
                        companion = path.parent.parent / folder / (path.stem + ".png")
                        if companion.is_file():
                            add(
                                f"{prefix}/{folder}/{stem}.png",
                                inside(
                                    self.store.dataset_root,
                                    str(companion.relative_to(self.store.dataset_root)),
                                ),
                            )
                if source.get("completed_depth_file"):
                    companion = inside(
                        self.store.dataset_root, source["completed_depth_file"]
                    ).with_suffix(".json")
                    if companion.is_file():
                        companion = inside(
                            self.store.dataset_root,
                            str(companion.relative_to(self.store.dataset_root)),
                        )
                        metadata = json.loads(companion.read_text(encoding="utf-8"))
                        metadata["export_source_paths"] = {
                            k: v for k, v in metadata.items() if k.endswith("_file")
                        }
                        for key, value in list(metadata.items()):
                            if isinstance(value, str) and value in path_map:
                                metadata[key] = path_map[value]
                        add(f"{prefix}/depth_gt_near_completed/{stem}.json", json_bytes(metadata))
                record.update(
                    {
                        k: row[k]
                        for k in (
                            "image_file",
                            "nir_file",
                            "detection_file",
                            "road_file",
                            "annotation_file",
                            "road_probability_file",
                            "preview_file",
                        )
                    }
                )
                record.update(
                    label_source="reviewed", exported_reviewed=True, review_status="approved"
                )
                record["export_source"] = {
                    "dataset_root": str(self.store.dataset_root),
                    "pseudo_root": str(self.store.root),
                    "image_file": source["image_file"],
                    "annotation_file": item["paths"]["annotation_file"],
                    "revision": item["revision"],
                    "sequence": source["sequence"],
                    "image_stamp": source.get("image_stamp", ""),
                }
                add(row["annotation_file"], json_bytes(record))
                rows.append(row)
                frame_rows[prefix].append(row)
        schema = snapshots[0]["label_schema"]
        add(FILE_NAME, json_bytes(schema))
        names = {item["id"]: item["name"] for item in schema["detection"]}
        add(
            "classes.txt",
            (
                "\n".join(names.get(i, "") for i in range(max(names, default=-1) + 1)) + "\n"
            ).encode(),
        )
        fields = list(dict.fromkeys(key for row in rows for key in row))
        add("manifest.csv", csv_bytes(rows, fields))
        for split in ("train", "test"):
            add(split + ".csv", csv_bytes([r for r in rows if r["split"] == split], fields))
        for prefix, group_rows in frame_rows.items():
            add(prefix + "/frames.csv", csv_bytes(group_rows, fields))
        return files, rows

    def _run(self, job):
        from core.io import atomic_bytes, json_bytes

        target = Path(job["output"])
        owned = False
        try:
            self._update(job, status="snapshot")
            snapshots = self.snapshot()
            snapshot_at = datetime.now(timezone.utc).isoformat()
            self._update(job, status="planning", frames=len(snapshots), snapshot_utc=snapshot_at)
            files, rows = self.plan(snapshots)
            total = sum(f["size"] for f in files.values())
            target.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target.parent).free < total + 1024 * 1024:
                raise ExportError("추출할 데이터를 저장할 디스크 공간이 부족합니다.")
            target.mkdir()  # Exclusive reservation; never overwrite any existing folder.
            owned = True
            for split in ("train", "test"):
                (target / split).mkdir()
            marker = target / ".export_incomplete.json"
            atomic_bytes(marker, json_bytes({"job_id": job["id"], "snapshot_utc": snapshot_at}))
            self._update(job, status="copying", files_total=len(files), bytes_total=total)
            hashes, bytes_done, files_done = {}, 0, 0
            last_update = time.monotonic()
            for relative, spec in files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                source = spec["source"]
                digest = hashlib.sha256()
                if isinstance(source, Path):
                    stat = source.stat()
                    if (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != spec[
                        "signature"
                    ]:
                        raise ExportError(f"복사 준비 중 원본 파일이 변경됐습니다: {source}")
                    reader = source.open("rb")
                else:
                    reader = io.BytesIO(source)
                with reader, path.open("xb") as writer:
                    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                        bytes_done += len(chunk)
                        if time.monotonic() - last_update > 0.5:
                            self._update(
                                job,
                                bytes_done=bytes_done,
                                files_done=files_done,
                                current_file=relative,
                            )
                            last_update = time.monotonic()
                if isinstance(source, Path):
                    stat = source.stat()
                    if (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != spec[
                        "signature"
                    ]:
                        raise ExportError(f"복사 도중 원본 파일이 변경됐습니다: {source}")
                if path.stat().st_size != spec["size"]:
                    raise ExportError(f"복사 크기가 일치하지 않습니다: {relative}")
                value = digest.hexdigest()
                if spec["expected_hash"] and value != spec["expected_hash"]:
                    raise ExportError(f"원본 RGB 해시가 manifest와 다릅니다: {relative}")
                hashes[relative] = value
                files_done += 1
            summary = {
                "schema_version": 1,
                "snapshot_utc": snapshot_at,
                "frames": len(rows),
                "splits": dict(Counter(r["split"] for r in rows)),
                "domains": dict(Counter(r["domain"] for r in rows)),
                "ordering": "image_stamp ascending per split/domain; ties: date, sequence, image_file",
                "layout": "<split>/<domain>/<modality>/<000001...>.<extension>",
                "mask_values": {
                    str(item["id"]): item["name"]
                    for item in snapshots[0]["label_schema"]["segmentation"]
                },
                "label_schema_file": FILE_NAME,
                "label_source": "reviewed pseudo labels",
                "path_base": "export root for every CSV",
                "depth_note": "Original NPZ copied unchanged; source=1 measured, source=2 estimated. Evaluate measured pixels only.",
                "files_sha256": hashes,
            }
            atomic_bytes(target / "export_summary.json", json_bytes(summary))
            marker.unlink()
            self._update(
                job,
                status="completed",
                bytes_done=bytes_done,
                files_done=files_done,
                current_file="",
                finished=time.time(),
            )
        except Exception as error:
            if owned:
                try:
                    shutil.rmtree(target)
                except OSError:
                    pass
            self._update(job, status="error", error=str(error), finished=time.time())


def main():
    """CLI counterpart of the GUI export button; export approved frames only."""
    import argparse
    from core.store import ReviewStore

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "dataset")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs/review",
        help="Existing annotation workspace",
    )
    parser.add_argument("--export-parent", type=Path, default=root / "outputs/exports")
    parser.add_argument("--name", required=True, help="New export folder name")
    args = parser.parse_args()
    manifest = args.output_root / "workspace_manifest.csv"
    store = ReviewStore(
        args.dataset_root, args.output_root, manifest=manifest if manifest.exists() else None
    )
    exports = DatasetExports(store)
    job = exports.start(
        {"parent": str(args.export_parent.expanduser().resolve()), "name": args.name}
    )
    while exports.thread.is_alive():
        exports.thread.join(1)
    job = exports.get(job["id"])
    print(json.dumps(job, indent=2, ensure_ascii=False))
    if job["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
