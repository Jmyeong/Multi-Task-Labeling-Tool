"""Text -> Grounding DINO boxes -> original SAM masks, using local HF weights.

Implements the Grounded-SAM workflow without CUDA extension compilation.
Weights are the official IDEA-Research/grounding-dino-tiny and facebook/sam-vit-base.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace


def validate_prompts(prompts):
    if not isinstance(prompts, dict) or not 1 <= len(prompts) <= 32:
        raise ValueError(
            "Grounded-SAM requires 1–32 English text prompts mapped to GUI class names/IDs."
        )
    for text, target in prompts.items():
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 120
            or any(ord(c) < 32 for c in text)
        ):
            raise ValueError("Each text prompt must contain 1–120 printable characters.")
        if type(target) not in (str, int):
            raise ValueError("Each prompt must map to an active GUI segmentation class.")
    return prompts


def bundle_paths(checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("architecture") != "grounded_sam":
        raise ValueError("Select a Grounded-SAM bundle JSON with architecture=grounded_sam.")
    result = {}
    for name in ("grounding_dino", "sam"):
        directory = (path.parent / config[name]).resolve()
        if not directory.is_dir() or not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"Local {name} weights are missing: {directory}. Run scripts.download_example_models."
            )
        if not list(directory.glob("*.safetensors")):
            raise FileNotFoundError(f"No safetensors weights in {directory}")
        result[name] = directory
    return config, result


class GroundedSAM:
    def __init__(self, checkpoint):
        self.bundle, self.paths = bundle_paths(checkpoint)
        self.configure(self.bundle.get("prompts", {"road": 1}), 0.25)
        self.grounder = self.sam = None
        self.device = None

    def configure(self, prompts, text_threshold=0.25):
        self.prompts = validate_prompts(prompts)
        self.names = dict(enumerate(self.prompts))
        self.text_threshold = text_threshold

    def load(self, device):
        import torch

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            SamProcessor,
            SamModel,
        )

        target = "cpu" if str(device) == "cpu" else f"cuda:{str(device).removeprefix('cuda:')}"
        if self.grounder is not None and target == self.device:
            return
        if target != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; choose cpu or install a compatible PyTorch build."
            )
        self.processor = AutoProcessor.from_pretrained(
            self.paths["grounding_dino"], local_files_only=True
        )
        self.grounder = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self.paths["grounding_dino"], local_files_only=True, use_safetensors=True
            )
            .to(target)
            .eval()
        )
        self.sam_processor = SamProcessor.from_pretrained(self.paths["sam"], local_files_only=True)
        self.sam = (
            SamModel.from_pretrained(self.paths["sam"], local_files_only=True, use_safetensors=True)
            .to(target)
            .eval()
        )
        self.device = target

    def predict(self, source, device="cpu", conf=0.3, max_det=100, **unused):
        import torch
        from torchvision.ops import nms

        self.load(device)
        image = source.convert("RGB")
        height, width = image.height, image.width
        detections = []
        with torch.inference_mode():
            # One prompt per pass makes class identity exact, including multi-word prompts.
            for class_id, text in self.names.items():
                inputs = self.processor(
                    images=image, text=text.strip().rstrip(".") + ".", return_tensors="pt"
                ).to(self.device)
                outputs = self.grounder(**inputs)
                detected = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=conf,
                    text_threshold=self.text_threshold,
                    target_sizes=[(height, width)],
                )[0]
                boxes, scores = detected["boxes"].cpu().float(), detected["scores"].cpu().float()
                # A box whose phrase has no surviving tokens must not be assigned
                # a semantic class merely because we know the requested prompt.
                phrase_valid = torch.tensor(
                    [bool(label.strip(" .")) for label in detected["text_labels"]], dtype=torch.bool
                )
                boxes[:, 0::2].clamp_(0, width)
                boxes[:, 1::2].clamp_(0, height)
                valid = (
                    torch.isfinite(boxes).all(dim=1)
                    & torch.isfinite(scores)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                    & phrase_valid
                )
                boxes, scores = boxes[valid], scores[valid]
                keep = nms(boxes, scores, 0.5)[:max_det]
                for i in keep:
                    detections.append(
                        torch.cat((boxes[i], scores[i : i + 1], torch.tensor([float(class_id)])))
                    )
            detections.sort(key=lambda row: float(row[4]), reverse=True)
            data = torch.stack(detections[:max_det]) if detections else torch.empty((0, 6))
            if not len(data):
                return [
                    SimpleNamespace(
                        orig_shape=(height, width), boxes=SimpleNamespace(data=data), masks=None
                    )
                ]
            sam_inputs = self.sam_processor(
                images=image, input_boxes=[data[:, :4].tolist()], return_tensors="pt"
            ).to(self.device)
            outputs = self.sam(**sam_inputs, multimask_output=True)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            quality = outputs.iou_scores[0].cpu()
            best = quality.argmax(dim=-1)
            masks = masks[torch.arange(len(data)), best]
        return [
            SimpleNamespace(
                orig_shape=(height, width),
                boxes=SimpleNamespace(data=data),
                masks=SimpleNamespace(data=masks),
            )
        ]
