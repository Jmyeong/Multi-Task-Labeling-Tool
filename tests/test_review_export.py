"""Exercise actual copies in temporary datasets, never production data."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from core.dataset import annotation_paths, adapt_rows, read_manifest
from scripts.extract_dataset import DatasetExports, ExportError
from core.store import ReviewStore
from gui.server import make_handler


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "source"
        self.root.mkdir()
        self.pseudo = self.root / "pseudo_labels"
        self.pseudo.mkdir()
        self.rows = []
        specs = [
            ("train", "base", "base__A", "100.000000002", "approved"),
            ("train", "base", "base__B", "100.000000001", "approved"),
            ("train", "hills", "hills__C", "99", "approved"),
            ("test", "base", "base__D", "50", "approved"),
            ("train", "base", "base__E", "1", "pending"),
            ("train", "base", "base__F", "2", "rejected"),
        ]
        for i, (split, domain, seq, stamp, status) in enumerate(specs):
            prefix = f"260916/{split}/{seq}"
            row = {
                "date": "260916",
                "split": split,
                "sequence": seq,
                "domain": domain,
                "image_stamp": stamp,
                "pointcloud_stamp": stamp,
                "image_file": f"{prefix}/images/000001.png",
                "nir_file": f"{prefix}/nir/000001.png",
                "depth_file": f"{prefix}/depth_gt/000001.npz",
                "filtered_depth_file": f"{prefix}/depth_gt_overlap_filtered/000001.npz",
                "completed_depth_file": f"{prefix}/depth_gt_near_completed/000001.npz",
                "mask_file": f"{prefix}/depth_gt_mask/000001.png",
                "camera_info_file": f"{prefix}/projection_camera_info.yaml",
                "semseg_file": "",
                "instseg_file": "",
                "pcd_file": "",
            }
            for field in ("image_file", "nir_file", "mask_file"):
                path = self.root / row[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((16, 24), i, np.uint8)).save(path)
            for field in ("depth_file", "filtered_depth_file", "completed_depth_file"):
                path = self.root / row[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    depth_m=np.full((16, 24), i + 1, np.float32),
                    source=np.full((16, 24), 2, np.uint8),
                )
            (self.root / row["camera_info_file"]).write_text(
                f"camera: {seq}\n", encoding="utf-8", newline="\n"
            )
            completed = self.root / row["completed_depth_file"]
            completed.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "image_file": row["image_file"].removeprefix("260916/"),
                        "source_depth_file": row["filtered_depth_file"].removeprefix("260916/"),
                        "output_file": row["completed_depth_file"].removeprefix("260916/"),
                        "source_note": "estimated",
                    }
                ),
                encoding="utf-8",
                newline="\n",
            )
            row["image_sha256"] = hashlib.sha256(
                (self.root / row["image_file"]).read_bytes()
            ).hexdigest()
            paths = annotation_paths(row)
            for path in paths.values():
                (self.pseudo / path).parent.mkdir(parents=True, exist_ok=True)
            (self.pseudo / paths["detection_file"]).write_text(
                f"{i % 4} 0.5 0.5 0.5 0.5\n", encoding="utf-8", newline="\n"
            )
            mask = np.zeros((16, 24), np.uint8)
            mask[2:8] = 1
            mask[8:10] = 255
            Image.fromarray(mask).save(self.pseudo / paths["road_file"])
            Image.fromarray(np.zeros((16, 24), np.uint16)).save(
                self.pseudo / paths["road_probability_file"]
            )
            record = {
                **row,
                **paths,
                "width": 24,
                "height": 16,
                "review_status": status,
                "label_source": "pseudo",
                "detections": [],
                "detection_count": 1,
                "review": {"note": seq, "save_count": 3},
            }
            (self.pseudo / paths["annotation_file"]).write_text(
                json.dumps(record), encoding="utf-8", newline="\n"
            )
            self.rows.append(row)
        self.write_manifest()
        self.store = ReviewStore(self.root, self.pseudo)
        self.exports = DatasetExports(self.store)

    def write_manifest(self):
        with (self.root / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        if self.exports.thread:
            self.exports.thread.join(10)
        self.temp.cleanup()

    def run_export(self, name="exported"):
        job = self.exports.start({"parent": str(self.base), "name": name})
        self.exports.thread.join(10)
        self.assertFalse(self.exports.thread.is_alive())
        return self.exports.get(job["id"])

    def test_approved_merge_timestamp_order_all_modalities_and_loader(self):
        originals = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        job = self.run_export()
        self.assertEqual(job["status"], "completed", job.get("error"))
        out = Path(job["output"])
        rows = read_manifest(out)
        self.assertEqual(len(rows), 4)
        base = [r for r in rows if r["split"] == "train" and r["domain"] == "base"]
        self.assertEqual([r["source_sequence"] for r in base], ["base__B", "base__A"])
        self.assertEqual(
            [r["image_file"] for r in base],
            ["train/base/images/000001.png", "train/base/images/000002.png"],
        )
        self.assertFalse(any(p.name.startswith("base__") for p in out.rglob("*")))
        lookup = {r["image_file"]: r for r in self.rows}
        for row in rows:
            source = lookup[row["source_image_file"]]
            paths = annotation_paths(source)
            for field in (
                "image_file",
                "nir_file",
                "depth_file",
                "filtered_depth_file",
                "completed_depth_file",
                "mask_file",
                "camera_info_file",
            ):
                self.assertEqual(
                    (out / row[field]).read_bytes(), (self.root / source[field]).read_bytes()
                )
            for field in ("detection_file", "road_file", "road_probability_file"):
                self.assertEqual(
                    (out / row[field]).read_bytes(), (self.pseudo / paths[field]).read_bytes()
                )
            record = json.loads((out / row["annotation_file"]).read_text(encoding="utf-8"))
            self.assertEqual(record["review_status"], "approved")
            self.assertEqual(record["review"]["save_count"], 3)
            meta = json.loads(
                (out / row["completed_depth_file"]).with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["image_file"], row["image_file"])
            self.assertEqual(meta["source_depth_file"], row["filtered_depth_file"])
            with np.load(out / row["completed_depth_file"]) as data:
                self.assertTrue((data["source"] == 2).all())
        for split in ("train", "test"):
            filtered = [r for r in rows if r["split"] == split]
            loaded = adapt_rows(out, filtered, split, "", "reviewed", "completed")
            self.assertEqual(len(loaded), len(filtered))
            self.assertTrue(
                all(Path(r["label"]).is_file() and Path(r["road_mask"]).is_file() for r in loaded)
            )
        reopened = ReviewStore(out, out)
        self.assertEqual(len(reopened.frames), 4)
        self.assertEqual(reopened.load(next(iter(reopened.frames)))["status"], "approved")
        summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
        for rel, digest in summary["files_sha256"].items():
            self.assertEqual(hashlib.sha256((out / rel).read_bytes()).hexdigest(), digest)
        self.assertFalse((out / ".export_incomplete.json").exists())
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)

    def test_paths_existing_dest_and_concurrent_job_are_protected(self):
        (self.base / "taken").mkdir()
        (self.base / "taken" / "keep").write_text("keep", encoding="utf-8", newline="\n")
        for payload in (
            {"parent": str(self.base), "name": "taken"},
            {"parent": "relative", "name": "new"},
            {"parent": str(self.base), "name": "../bad"},
            {"parent": str(self.base), "name": " .. "},
            {"parent": str(self.pseudo), "name": "bad"},
        ):
            with self.assertRaises(ExportError):
                self.exports.start(payload)
        self.assertEqual((self.base / "taken" / "keep").read_text(encoding="utf-8"), "keep")
        entered = threading.Event()
        release = threading.Event()
        snapshot = self.exports.snapshot

        def wait_snapshot():
            entered.set()
            release.wait(5)
            return snapshot()

        with patch.object(self.exports, "snapshot", side_effect=wait_snapshot):
            self.exports.start({"parent": str(self.base), "name": "first"})
            self.assertTrue(entered.wait(3))
            with self.assertRaises(ExportError):
                self.exports.start({"parent": str(self.base), "name": "second"})
            release.set()
            self.exports.thread.join(10)

    def test_missing_timestamp_or_file_fails_without_output(self):
        self.rows[0]["image_stamp"] = ""
        self.write_manifest()
        job = self.run_export()
        self.assertEqual(job["status"], "error")
        self.assertIn("image_stamp", job["error"])
        self.assertFalse(Path(job["output"]).exists())
        self.rows[0]["image_stamp"] = "100"
        self.write_manifest()
        (self.root / self.rows[0]["depth_file"]).unlink()
        job = self.run_export("missing")
        self.assertEqual(job["status"], "error")
        self.assertFalse(Path(job["output"]).exists())

    def test_snapshot_survives_later_gui_changes(self):
        plan = self.exports.plan

        def edit_after_snapshot(snapshots):
            row = self.rows[0]
            path = self.pseudo / annotation_paths(row)["detection_file"]
            old = path.read_bytes()
            path.write_text("", encoding="utf-8", newline="\n")
            result = plan(snapshots)
            self.saved_before = old
            return result

        with patch.object(self.exports, "plan", side_effect=edit_after_snapshot):
            job = self.run_export()
        self.assertEqual(job["status"], "completed", job.get("error"))
        row = next(
            r for r in read_manifest(Path(job["output"])) if r["source_sequence"] == "base__A"
        )
        self.assertEqual(
            (Path(job["output"]) / row["detection_file"]).read_bytes(), self.saved_before
        )

    def test_bad_image_hash_cleans_partial_copy(self):
        self.rows[0]["image_sha256"] = "bad"
        path = self.pseudo / annotation_paths(self.rows[0])["annotation_file"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["image_sha256"] = "bad"
        path.write_text(json.dumps(record), encoding="utf-8", newline="\n")
        self.write_manifest()
        job = self.run_export()
        self.assertEqual(job["status"], "error")
        self.assertIn("해시", job["error"])
        self.assertFalse(Path(job["output"]).exists())

    def test_http_auth_start_status_and_assets(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.store, exports=self.exports)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        data = json.dumps({"parent": str(self.base), "name": "http_export"}).encode()
        try:
            self.assertIsNone(json.load(urlopen(url + "/api/dataset-exports/latest"))["job"])
            with self.assertRaises(HTTPError) as error:
                urlopen(
                    Request(
                        url + "/api/dataset-exports",
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                )
            self.assertEqual(error.exception.code, 403)
            with urlopen(
                Request(
                    url + "/api/dataset-exports",
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "X-Review-Token": self.store.token,
                    },
                )
            ) as response:
                self.assertEqual(response.status, 202)
                job = json.load(response)["job"]
            self.exports.thread.join(10)
            self.assertEqual(
                json.load(urlopen(url + "/api/dataset-exports/" + job["id"]))["job"]["status"],
                "completed",
            )
            self.assertEqual(urlopen(url + "/export.js").status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
