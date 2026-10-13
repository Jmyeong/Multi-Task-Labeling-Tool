"""Dynamic class identities, multiclass masks, conflicts and portable exports."""

import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from core.file_lock import exclusive_lock

import numpy as np
from PIL import Image

from core.io import ReviewError
from core.label_schema import DEFAULT, load_schema, schema_revision, require_teacher_schema
from core.workspace import prepare_workspace
from core.store import ReviewStore
from gui.server import Workspace
from scripts.extract_dataset import DatasetExports


class LabelSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / "images"
        self.source.mkdir()
        Image.fromarray(np.full((20, 30, 3), 80, np.uint8)).save(self.source / "a.png")
        self.output = self.base / "labels"
        manifest, _ = prepare_workspace(self.source, self.output)
        self.store = ReviewStore(self.source, self.output, manifest=manifest)
        self.fid = next(iter(self.store.frames))

    def tearDown(self):
        self.tmp.cleanup()

    def settings(self):
        schema = load_schema(self.output)
        return {**deepcopy(schema), "revision": schema_revision(schema)}

    def add_classes(self):
        settings = self.settings()
        settings["detection"].append(
            {"id": None, "name": "Vehicle", "color": "#e01234", "active": True}
        )
        settings["segmentation"].append(
            {"id": None, "name": "Water", "color": "#1256ef", "active": True}
        )
        return self.store.save_schema(settings)

    def save_custom_labels(self):
        frame = self.store.load(self.fid)
        mask = np.full((20, 30), 255, np.uint8)
        mask[2:10, 3:14] = 2
        return self.store.save(
            self.fid,
            {
                "revision": frame["revision"],
                "status": "approved",
                "boxes": [{"class_id": 4, "bbox_xyxy": [1, 2, 15, 18]}],
                "mask": base64.b64encode(mask.tobytes()).decode(),
            },
        )

    def test_add_color_save_reload_multiclass_and_export_import(self):
        self.add_classes()
        self.save_custom_labels()
        schema = load_schema(self.output)
        self.assertEqual(schema["detection"][-1]["id"], 4)
        self.assertEqual(
            next(c for c in schema["segmentation"] if c["id"] == 2)["color"], "#1256ef"
        )
        workspace = Workspace(self.source, self.output)
        frame = workspace.store.load(self.fid)
        self.assertEqual(frame["boxes"][0]["class_id"], 4)
        self.assertEqual(set(base64.b64decode(frame["mask"])), {2, 255})
        exports = DatasetExports(workspace.store)
        job = exports.start({"parent": str(self.base), "name": "exported"})
        exports.thread.join(10)
        job = exports.get(job["id"])
        self.assertEqual(job["status"], "completed", job["error"])
        target = Path(job["output"])
        self.assertEqual(load_schema(target), schema)
        self.assertEqual(
            (target / "classes.txt").read_text(encoding="utf-8").splitlines()[4], "Vehicle"
        )
        self.assertEqual(
            json.loads((target / "export_summary.json").read_text(encoding="utf-8"))["mask_values"][
                "2"
            ],
            "Water",
        )
        imported = Workspace(target, self.base / "imported")
        self.assertEqual(load_schema(imported.store.root), schema)
        self.assertEqual(
            imported.store.load(next(iter(imported.store.frames)))["boxes"][0]["class_id"], 4
        )

    def test_removal_restoration_never_rewrites_labels_or_reuses_ids(self):
        self.add_classes()
        self.save_custom_labels()
        paths = self.store.frame(self.fid)["paths"]
        before = {
            key: (self.output / paths[key]).read_bytes()
            for key in ["detection_file", "road_file", "annotation_file"]
        }
        settings = self.settings()
        settings["detection"] = [c for c in settings["detection"] if c["id"] != 4]
        settings["segmentation"] = [c for c in settings["segmentation"] if c["id"] != 2]
        self.store.save_schema(settings)
        schema = load_schema(self.output)
        self.assertFalse(next(c for c in schema["detection"] if c["id"] == 4)["active"])
        self.assertFalse(next(c for c in schema["segmentation"] if c["id"] == 2)["active"])
        self.assertEqual(self.store.load(self.fid)["boxes"][0]["class_id"], 4)
        self.assertEqual(before, {key: (self.output / paths[key]).read_bytes() for key in before})
        new = self.add_classes_with_names()
        self.assertEqual(new["detection"][-1]["id"], 5)
        self.assertIn(3, [c["id"] for c in new["segmentation"]])
        restore = self.settings()
        for kind, label_id in [("detection", 4), ("segmentation", 2)]:
            next(c for c in restore[kind] if c["id"] == label_id)["active"] = True
        self.store.save_schema(restore)
        self.assertEqual(before, {key: (self.output / paths[key]).read_bytes() for key in before})

    def add_classes_with_names(self):
        settings = self.settings()
        settings["detection"].append({"id": None, "name": "Obstacle", "color": "#123456"})
        settings["segmentation"].append({"id": None, "name": "Grass", "color": "#654321"})
        return self.store.save_schema(settings)

    def test_invalid_colors_names_reserved_ids_and_duplicates_do_not_write(self):
        cases = []
        for field, value in [
            ("color", "red"),
            ("name", ""),
            ("name", "a\nb"),
            ("active", "true"),
            ("id", 100),
            ("id", True),
        ]:
            settings = self.settings()
            settings["detection"][0][field] = value
            cases.append(settings)
        for label_id in [0, 255]:
            settings = self.settings()
            next(c for c in settings["segmentation"] if c["id"] == label_id)["active"] = False
            cases.append(settings)
        settings = self.settings()
        settings["detection"][0]["name"] = "Person"
        cases.append(settings)
        for settings in cases:
            with self.assertRaises(ReviewError):
                self.store.save_schema(settings)
        self.assertFalse((self.output / "label_schema.json").exists())
        self.assertEqual(load_schema(self.output), DEFAULT)

    def test_settings_and_frame_conflicts_generation_lock_and_failed_write(self):
        old = self.settings()
        frame_revision = self.store.load(self.fid)["revision"]
        self.add_classes()
        with self.assertRaises(ReviewError):
            self.store.save_schema(old)
        self.assertNotEqual(self.store.load(self.fid)["revision"], frame_revision)
        with exclusive_lock(self.output / ".generation.lock"):
            with self.assertRaises(ReviewError):
                self.store.save_schema(self.settings())
        before = (self.output / "label_schema.json").read_bytes()
        with patch("core.label_schema.atomic_bytes", side_effect=OSError("disk write failed")):
            with self.assertRaises(OSError):
                self.store.save_schema(self.settings())
        self.assertEqual((self.output / "label_schema.json").read_bytes(), before)

    def test_teacher_mapping_rejects_renaming_but_allows_added_manual_classes(self):
        self.add_classes()
        require_teacher_schema(self.output)
        settings = self.settings()
        settings["detection"][0]["name"] = "Car"
        self.store.save_schema(settings)
        with self.assertRaises(ReviewError):
            require_teacher_schema(self.output)


if __name__ == "__main__":
    unittest.main()
