"""Pseudo-label contracts without downloading weights or requiring Torch."""

import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.dataset import read_manifest
from core.io import ReviewError
from core.label_schema import DEFAULT
from gui.server import Workspace
from pseudo_labeling.predictions import class_mapping, convert_result
from pseudo_labeling.registry import catalog
from pseudo_labeling.service import PseudoJobs
from scripts.generate_single_task import generate


class Tensor:
    def __init__(self, value):
        self.value = np.array(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def result(task):
    boxes = SimpleNamespace(data=Tensor([[1, 2, 10, 12, 0.8, 0]]))
    masks = SimpleNamespace(data=Tensor(np.ones((1, 16, 24)))) if task == "segmentation" else None
    return SimpleNamespace(orig_shape=(16, 24), boxes=boxes, masks=masks)


class PseudoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.source = self.base / "RGB 한글 input"
        self.source.mkdir()
        self.image = self.source / "0001.png"
        Image.fromarray(np.full((16, 24, 3), 80, np.uint8)).save(self.image)
        self.original = self.image.read_bytes()
        self.weight = self.base / "test checkpoint.pt"
        self.weight.write_bytes(b"stub")

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, task):
        return dict(
            task=task,
            dataset_root=str(self.source),
            output_root=str(self.base / task),
            checkpoint=str(self.weight),
            schema=DEFAULT,
            device="cpu",
            height=352,
            width=640,
            conf=0.2,
            max_det=300,
            limit=0,
            class_mapping={"person": 1},
        )

    def test_class_mapping_does_not_assume_checkpoint_ids(self):
        self.assertEqual(class_mapping({0: "person", 1: "car"}, DEFAULT["detection"]), {0: 1})
        self.assertEqual(
            class_mapping({0: "person"}, DEFAULT["segmentation"], {"person": "도로"}), {0: 1}
        )
        with self.assertRaises(ValueError):
            class_mapping({0: "car"}, DEFAULT["detection"])
        with self.assertRaises(ValueError):
            class_mapping({0: "person"}, DEFAULT["detection"], {"person": 999})
        with self.assertRaises(ValueError):
            class_mapping({0: "person"}, DEFAULT["detection"], {"typo": 1})

    def test_original_resolution_and_overlap_priority(self):
        prediction = result("segmentation")
        mask1 = np.zeros((16, 24))
        mask1[:, :12] = 1
        prediction.boxes.data = Tensor([[0, 0, 12, 16, 0.9, 0], [0, 0, 24, 16, 0.7, 1]])
        prediction.masks.data = Tensor([mask1, np.ones((16, 24))])
        boxes, mask = convert_result(prediction, "segmentation", {0: 1, 1: 2}, DEFAULT)
        self.assertFalse(boxes)
        self.assertTrue(np.all(mask[:, :12] == 1))
        self.assertTrue(np.all(mask[:, 12:] == 2))
        prediction.masks.data = Tensor(np.ones((2, 8, 12)))
        with self.assertRaises(ValueError):
            convert_result(prediction, "segmentation", {0: 1}, DEFAULT)

    def test_generate_review_load_resume_and_no_source_writes(self):
        for task in ("segmentation", "object_detection"):
            config = self.config(task)
            network = SimpleNamespace(names={0: "person"}, predict=lambda **kw: [result(task)])
            with patch(
                "scripts.generate_single_task.importlib.import_module",
                return_value=SimpleNamespace(load_network=lambda _: network),
            ):
                generate(config)
                output = Path(config["output_root"])
                workspace = Workspace(self.source, output)
                frame = workspace.store.load(next(iter(workspace.store.frames)))
                self.assertEqual(frame["status"], "pending")
                self.assertEqual(
                    set(base64.b64decode(frame["mask"])), {1} if task == "segmentation" else {0}
                )
                self.assertEqual(len(frame["boxes"]), 0 if task == "segmentation" else 1)
                before = {
                    p: p.read_bytes()
                    for p in output.rglob("*")
                    if p.is_file() and p.suffix in [".png", ".txt"]
                }
                generate(config)
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                different = dict(config, conf=0.9)
                with self.assertRaises(ValueError):
                    generate(different)
                manual = Workspace(self.source, self.base / "manual")
                with self.assertRaises(ValueError):
                    generate(dict(config, output_root=str(manual.store.root)))
            self.assertEqual(list(self.source.iterdir()), [self.image])
            self.assertEqual(self.image.read_bytes(), self.original)
            self.assertEqual(len(read_manifest(self.source, output / "workspace_manifest.csv")), 1)

    def test_job_dispatch_failure_log_and_protected_output(self):
        workspace = Workspace(self.source, self.base / "manual")
        jobs = PseudoJobs(self.base / "jobs")
        payload = dict(
            task="object_detection",
            checkpoint=str(self.weight),
            python=sys.executable,
            output_root=str(self.base / "generated"),
            device="cpu",
            class_mapping={"person": "Person"},
        )

        class Process:
            def __init__(self, *args, **kwargs):
                self.stdout = io.StringIO("Selected=1 existing=0 pending=1\nGenerated 1/1\n")

            def wait(self):
                return 0

            def poll(self):
                return 0

        with patch("pseudo_labeling.service.subprocess.Popen", Process):
            job = jobs.start(payload, workspace.store)
            jobs.thread.join(10)
        self.assertEqual(jobs.get()["status"], "completed")
        self.assertEqual(jobs.get()["done"], 1)
        self.assertEqual(PseudoJobs(self.base / "jobs").get()["id"], job["id"])
        for output in [workspace.store.root, self.source, self.source / "labels", self.base]:
            with self.assertRaises(ReviewError):
                jobs.start(dict(payload, output_root=str(output)), workspace.store)
        with self.assertRaises(ReviewError):
            jobs.start(dict(payload, device="0; echo bad"), workspace.store)
        with self.assertRaises(ReviewError):
            jobs.start(dict(payload, height=353), workspace.store)
        with self.assertRaises(ReviewError):
            jobs.start(dict(payload, checkpoint="missing.pt"), workspace.store)

        class Failed(Process):
            def wait(self):
                return 1

        with patch("pseudo_labeling.service.subprocess.Popen", Failed):
            jobs.start(payload, workspace.store)
            jobs.thread.join(10)
        self.assertEqual(jobs.get()["status"], "error")
        self.assertIn("exit 1", jobs.get()["error"])

    def test_grounded_bundle_prompts_and_lazy_model_loading(self):
        from pseudo_labeling.model.segmentation.grounded_sam import (
            GroundedSAM,
            bundle_paths,
            validate_prompts,
        )

        bundle = self.base / "grounded_sam.json"
        for name in ["grounding_dino", "sam"]:
            directory = self.base / name
            directory.mkdir()
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"test")
        config = dict(
            architecture="grounded_sam",
            grounding_dino="grounding_dino",
            sam="sam",
            prompts={"road": 1},
        )
        bundle.write_text(json.dumps(config), encoding="utf-8")
        model = GroundedSAM(bundle)
        self.assertIsNone(model.sam)
        self.assertEqual(model.names, {0: "road"})
        model.configure({"road": 1, "water": 2}, 0.3)
        self.assertEqual(model.names, {0: "road", 1: "water"})
        for prompts in [{}, {"": 1}, {"road\nwater": 1}, {"road": []}]:
            with self.assertRaises(ValueError):
                validate_prompts(prompts)
        (self.base / "sam" / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            bundle_paths(bundle)

    def test_grounded_job_validates_semantic_targets(self):
        workspace = Workspace(self.source, self.base / "manual")
        jobs = PseudoJobs(self.base / "jobs")
        bundle = self.base / "grounded_sam.json"
        for name in ["grounding_dino", "sam"]:
            directory = self.base / name
            directory.mkdir()
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"test")
        bundle.write_text(
            json.dumps(
                dict(architecture="grounded_sam", grounding_dino="grounding_dino", sam="sam")
            ),
            encoding="utf-8",
        )
        payload = dict(
            task="segmentation",
            checkpoint=str(bundle),
            python=sys.executable,
            output_root=str(self.base / "generated"),
            class_mapping={"road": 255},
        )
        with self.assertRaises(ReviewError):
            jobs.start(payload, workspace.store)
        with self.assertRaises(ReviewError):
            jobs.start(dict(payload, class_mapping={"road": 999}), workspace.store)
        with self.assertRaises(ReviewError):
            jobs.start(dict(payload, class_mapping={"road": 1}, text_threshold=2), workspace.store)

    def test_grounded_weight_changes_invalidate_resumed_output(self):
        from pseudo_labeling.model.segmentation.grounded_sam import GroundedSAM

        bundle = self.base / "grounded_sam.json"
        for name in ["grounding_dino", "sam"]:
            directory = self.base / name
            directory.mkdir()
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"test")
        bundle.write_text(
            json.dumps(
                dict(architecture="grounded_sam", grounding_dino="grounding_dino", sam="sam")
            ),
            encoding="utf-8",
        )
        network = GroundedSAM(bundle)
        network.predict = lambda **kw: [result("segmentation")]
        config = dict(
            self.config("segmentation"), checkpoint=str(bundle), class_mapping={"road": 1}
        )
        with patch(
            "scripts.generate_single_task.importlib.import_module",
            return_value=SimpleNamespace(load_network=lambda _: network),
        ):
            generate(config)
            generate(config)
            (self.base / "sam" / "model.safetensors").write_bytes(b"changed model")
            with self.assertRaises(ValueError):
                generate(config)

    def test_model_catalog_contains_real_network_and_checkpoint_paths(self):
        from pseudo_labeling.registry import MODEL_ROOT

        for item in catalog():
            self.assertTrue((MODEL_ROOT / item["network"]).is_file())
            self.assertTrue(Path(item["checkpoint_dir"]).is_dir())
