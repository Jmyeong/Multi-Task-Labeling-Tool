"""Portable raw-image inputs, separate outputs and workspace switching."""

import base64
from core.file_lock import exclusive_lock
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from core.dataset import read_manifest
from core.io import ReviewError
from core.workspace import input_rows, prepare_workspace
from gui.server import Workspace, make_handler


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "input"
        self.source.mkdir()
        self.output = self.base / "labels"
        self.image = self.source / "frame.png"
        Image.fromarray(np.full((32, 48, 3), 90, np.uint8)).save(self.image)
        self.original = self.image.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, store):
        fid = next(iter(store.frames))
        frame = store.load(fid)
        return fid, {
            "revision": frame["revision"],
            "status": "approved",
            "note": "portable",
            "boxes": [{"class_id": 1, "bbox_xyxy": [2, 3, 15, 20]}],
            "mask": base64.b64encode(np.ones((32, 48), np.uint8).tobytes()).decode(),
        }

    def test_rgb_only_save_reopen_export_and_source_unchanged(self):
        workspace = Workspace(self.source, self.output)
        store = workspace.store
        fid, payload = self.payload(store)
        initial = store.load(fid)
        self.assertEqual(set(base64.b64decode(initial["mask"])), {255})
        store.save(fid, payload)
        manifest = self.output / "workspace_manifest.csv"
        self.assertEqual(len(read_manifest(self.source, manifest)), 1)
        self.assertFalse((self.source / "manifest.csv").exists())
        workspace.open(self.source, self.output)
        self.assertEqual(workspace.store.load(fid)["note"], "portable")
        job = workspace.exports.start({"parent": str(self.base / "exports"), "name": "reviewed"})
        workspace.exports.thread.join(10)
        job = workspace.exports.get(job["id"])
        self.assertEqual(job["status"], "completed", job["error"])
        exported = Path(job["output"])
        row = read_manifest(exported)[0]
        self.assertEqual(row["nir_file"], "")
        self.assertEqual((exported / row["image_file"]).read_bytes(), self.original)
        self.assertTrue((exported / row["road_file"]).exists())
        # Reimporting exported labels must also leave that source untouched.
        before = {p: p.read_bytes() for p in exported.rglob("*") if p.is_file()}
        reimport = Workspace(exported, self.base / "reimport")
        newfid = next(iter(reimport.store.frames))
        self.assertEqual(reimport.store.load(newfid)["boxes"][0]["class_id"], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(list(self.source.iterdir()), [self.image])
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_direct_images_folder_root(self):
        images = self.base / "images"
        images.mkdir()
        (images / "a.png").write_bytes(self.original)
        manifest, rows = prepare_workspace(images, self.output)
        self.assertEqual(rows[0]["sequence"], "images")
        self.assertEqual(rows[0]["image_file"], "a.png")
        self.assertEqual(rows[0]["nir_file"], "")
        self.assertTrue(manifest.is_file())

    def test_manifest_only_and_output_subfolder_not_scanned(self):
        manifest, rows = prepare_workspace(self.source, self.source / "labels", initialize=False)
        self.assertEqual(len(rows), 1)
        self.assertFalse(list((self.source / "labels").rglob("annotations/*.json")))
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(self.source / "labels" / "preview.png")
        self.assertEqual(len(input_rows(self.source, self.source / "labels")), 1)
        self.assertTrue(manifest.is_file())

    def test_nir_pair_and_train_test_assignments(self):
        self.image.unlink()
        for split in ("train", "test"):
            for modality in ("images", "nir"):
                folder = self.source / split / "seq" / modality
                folder.mkdir(parents=True)
                Image.fromarray(np.zeros((16, 24, 3), np.uint8)).save(folder / "a.png")
        rows = input_rows(self.source)
        self.assertEqual({r["split"] for r in rows}, {"train", "test"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["nir_file"].endswith("/nir/a.png") for r in rows))

    def test_bad_paths_duplicate_identity_and_wrong_workspace(self):
        with self.assertRaises(ReviewError):
            prepare_workspace(self.source, self.source)
        prepare_workspace(self.source, self.output)
        other = self.base / "other"
        other.mkdir()
        (other / "a.png").write_bytes(self.original)
        with self.assertRaises(ReviewError):
            prepare_workspace(other, self.output)
        (self.source / "frame.jpg").write_bytes(self.original)
        with self.assertRaises(ReviewError):
            input_rows(self.source)
        (self.source / "frame.jpg").unlink()
        outside = self.base / "outside.png"
        outside.write_bytes(self.original)
        try:
            (self.source / "linked.png").symlink_to(outside)
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise  # Windows symlinks require Developer Mode/admin rights.
            return
        with self.assertRaises(ReviewError):
            input_rows(self.source)

    def test_initialization_respects_generation_lock(self):
        prepare_workspace(self.source, self.output)
        before = (self.output / "workspace_manifest.csv").read_bytes()
        with exclusive_lock(self.output / ".generation.lock"):
            with self.assertRaises(ReviewError):
                prepare_workspace(self.source, self.output)
        self.assertEqual((self.output / "workspace_manifest.csv").read_bytes(), before)

    def test_empty_landing_and_http_switch_invalidates_previous_tab(self):
        empty = self.base / "empty"
        empty.mkdir()
        workspace = Workspace(empty, self.base / "empty-labels", allow_empty=True)
        self.assertEqual(len(workspace.store.frames), 0)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workspace=workspace))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"

        def get(path):
            with urlopen(address + path) as response:
                return json.load(response)

        def post(payload, token):
            request = Request(
                address + "/api/workspace",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json", "X-Review-Token": token},
            )
            with urlopen(request) as response:
                return json.load(response)

        try:
            old = get("/api/config")["token"]
            folders = get("/api/folders?path=" + str(self.source))
            self.assertEqual(folders["path"], str(self.source))
            bad = {"dataset_root": str(self.base / "missing"), "output_root": str(self.output)}
            with self.assertRaises(HTTPError):
                post(bad, old)
            self.assertEqual(get("/api/config")["token"], old)
            result = post({"dataset_root": str(self.source), "output_root": str(self.output)}, old)
            self.assertEqual(result["frames"], 1)
            self.assertNotEqual(get("/api/config")["token"], old)
            with self.assertRaises(HTTPError) as error:
                post(bad, old)
            self.assertEqual(error.exception.code, 403)
            with self.assertRaises(HTTPError) as error:
                get("/api/vlm")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == "__main__":
    unittest.main()
