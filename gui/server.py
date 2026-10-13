"""Local HTTP editor for MT-Labeling; no model dependencies for manual editing."""

import argparse
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from core.label_schema import public_schema
from core.io import ReviewError, inside, json_bytes
from core.store import ReviewStore
from core.workspace import prepare_workspace
from sam.service import (
    SamService,
    SamError,
    validate_points,
    DEFAULT_PYTHON,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
)
from scripts.extract_dataset import DatasetExports, ExportError

PROJECT = Path(__file__).resolve().parents[1]


class Workspace:
    def __init__(self, dataset_root, output_root, allow_empty=False):
        self.lock = threading.RLock()
        self.store = None
        self.exports = None
        self.open(dataset_root, output_root, allow_empty=allow_empty)

    def open(self, dataset_root, output_root, allow_empty=False):
        if (
            not isinstance(dataset_root, (str, Path))
            or not isinstance(output_root, (str, Path))
            or not str(dataset_root).strip()
            or not str(output_root).strip()
        ):
            raise ReviewError("입력 폴더와 출력 폴더를 지정하세요.")
        with self.lock:
            if self.exports and self.exports.thread and self.exports.thread.is_alive():
                raise ReviewError("데이터 추출이 진행 중입니다. 완료 후 작업 폴더를 바꾸세요.", 409)
            manifest, rows = prepare_workspace(dataset_root, output_root, allow_empty=allow_empty)
            new = ReviewStore(dataset_root, output_root, manifest=manifest, allow_empty=allow_empty)
            exports = DatasetExports(new)
            self.store, self.exports = new, exports


def make_handler(store=None, sam=None, exports=None, workspace=None):
    if exports is None and workspace is None:
        exports = DatasetExports(store)

    initial_store, initial_exports = store, exports

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(fmt, *args)

        def send_data(
            self, content, content_type="application/json; charset=utf-8", status=200, extra=None
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)

        def json(self, value, status=200):
            self.send_data(json_bytes(value), status=status)

        def do_GET(self):
            store = workspace.store if workspace else initial_store
            exports = workspace.exports if workspace else initial_exports
            try:
                path = urlparse(self.path).path
                if path in (
                    "/",
                    "/app.js",
                    "/workspace.js",
                    "/export.js",
                    "/labels.js",
                    "/style.css",
                ):
                    filename = {
                        "/": "index.html",
                        "/app.js": "app.js",
                        "/workspace.js": "workspace.js",
                        "/export.js": "export.js",
                        "/labels.js": "labels.js",
                        "/style.css": "style.css",
                    }[path]
                    file = PROJECT / "gui/web" / filename
                    return self.send_data(
                        file.read_bytes(), mimetypes.guess_type(str(file))[0] + "; charset=utf-8"
                    )
                if path == "/api/folders":
                    directory = (
                        Path(
                            parse_qs(urlparse(self.path).query).get(
                                "path", [str(store.dataset_root)]
                            )[0]
                        )
                        .expanduser()
                        .resolve()
                    )
                    if not directory.is_dir():
                        raise ReviewError("폴더가 없습니다: " + str(directory))
                    return self.json(
                        {
                            "path": str(directory),
                            "parent": str(directory.parent),
                            "folders": [
                                {"name": p.name, "path": str(p.resolve())}
                                for p in sorted(directory.iterdir(), key=lambda p: p.name.lower())
                                if p.is_dir() and not p.name.startswith(".")
                            ],
                        }
                    )
                if path == "/api/config":
                    return self.json(
                        {
                            "token": store.token,
                            **public_schema(store.root),
                            "root": str(store.root),
                            "export_parent": str(store.root.parent / "exports"),
                            "dataset_root": str(store.dataset_root),
                            "sam": sam.info()
                            if sam
                            else {"available": False, "message": "SAM이 연결되지 않았습니다."},
                        }
                    )
                if path == "/api/dataset-exports/latest":
                    return self.json({"job": exports.get()})
                if path.startswith("/api/dataset-exports/"):
                    return self.json({"job": exports.get(path.rsplit("/", 1)[-1])})
                if path == "/api/frames":
                    if parse_qs(urlparse(self.path).query).get("refresh") == ["1"]:
                        store.refresh()
                    return self.json({"frames": store.listing()})
                if path == "/api/export":
                    return self.send_data(
                        store.export_csv(),
                        "text/csv; charset=utf-8",
                        extra={"Content-Disposition": 'attachment; filename="review_status.csv"'},
                    )
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "frame"]:
                    frame_id = parts[2]
                    if len(parts) == 3:
                        return self.json(store.load(frame_id))
                    if len(parts) == 4 and parts[3] in ("rgb", "nir"):
                        frame = store.frame(frame_id)
                        key = "image_file" if parts[3] == "rgb" else "nir_file"
                        file = inside(store.dataset_root, frame["source"][key])
                        return self.send_data(
                            file.read_bytes(),
                            mimetypes.guess_type(str(file))[0] or "application/octet-stream",
                        )
                raise ReviewError("찾을 수 없는 주소입니다.", 404)
            except (ReviewError, ExportError) as error:
                self.json({"error": str(error)}, error.status)
            except (ValueError, KeyError, OSError) as error:
                self.json({"error": str(error)}, 500)

        def do_POST(self):
            store = workspace.store if workspace else initial_store
            exports = workspace.exports if workspace else initial_exports
            try:
                if not secrets.compare_digest(self.headers.get("X-Review-Token", ""), store.token):
                    raise ReviewError("세션이 만료되었습니다. 페이지를 새로고침하세요.", 403)
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ReviewError("다른 사이트에서 보낸 저장 요청입니다.", 403)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ReviewError("JSON 요청만 지원합니다.", 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ReviewError("저장 요청 크기가 올바르지 않습니다.", 413)
                parts = urlparse(self.path).path.strip("/").split("/")
                if parts == ["api", "label-schema"]:
                    payload = json.loads(self.rfile.read(length))
                    store.save_schema(payload)
                    return self.json(public_schema(store.root))
                if parts == ["api", "workspace"]:
                    if workspace is None:
                        raise ReviewError("이 서버는 작업 폴더 변경을 지원하지 않습니다.")
                    payload = json.loads(self.rfile.read(length))
                    workspace.open(payload.get("dataset_root"), payload.get("output_root"))
                    return self.json(
                        {
                            "frames": len(workspace.store.frames),
                            "dataset_root": str(workspace.store.dataset_root),
                            "output_root": str(workspace.store.root),
                        }
                    )
                if parts == ["api", "dataset-exports"]:
                    return self.json(
                        {"job": exports.start(json.loads(self.rfile.read(length)))}, status=202
                    )
                batch = parts == ["api", "frames", "reject"]
                sam_request = (
                    len(parts) == 4 and parts[:2] == ["api", "frame"] and (parts[3] == "sam")
                )
                if (
                    not batch
                    and (not sam_request)
                    and (len(parts) != 3 or parts[:2] != ["api", "frame"])
                ):
                    raise ReviewError("찾을 수 없는 주소입니다.", 404)
                payload = json.loads(self.rfile.read(length))
                if sam_request:
                    if sam is None:
                        raise SamError("SAM이 연결되지 않았습니다.")
                    with store.lock:
                        frame = store.frame(parts[2])
                        (w, h) = (int(frame["record"]["width"]), int(frame["record"]["height"]))
                        points = validate_points(
                            payload.get("points") if isinstance(payload, dict) else None, w, h
                        )
                        image_path = inside(store.dataset_root, frame["source"]["image_file"])
                    return self.json(sam.predict(image_path, w, h, points))
                self.json(store.reject_many(payload) if batch else store.save(parts[2], payload))
            except SamError as error:
                self.json({"error": str(error)}, error.status)
            except (ReviewError, ExportError) as error:
                self.json({"error": str(error)}, error.status)
            except (ValueError, TypeError, KeyError) as error:
                self.json({"error": str(error)}, 400)
            except OSError as error:
                self.json({"error": f"파일 저장 실패: {error}"}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "dataset")
    parser.add_argument(
        "--output-root",
        "--pseudo-root",
        dest="output_root",
        type=Path,
        default=PROJECT / "outputs/review",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--sam-python", default=DEFAULT_PYTHON)
    parser.add_argument("--sam-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-config", default=DEFAULT_CONFIG)
    parser.add_argument("--sam-device", default="auto")
    args = parser.parse_args()
    workspace = Workspace(args.dataset_root, args.output_root, allow_empty=True)
    sam = SamService(args.sam_python, args.sam_checkpoint, args.sam_config, args.sam_device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sam=sam, workspace=workspace))
    print(
        f"MT-Labeling: http://{args.host}:{server.server_port} | {len(workspace.store.frames)} frames",
        flush=True,
    )
    print(f"Input: {workspace.store.dataset_root}\nOutput: {workspace.store.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sam.close()


if __name__ == "__main__":
    main()
