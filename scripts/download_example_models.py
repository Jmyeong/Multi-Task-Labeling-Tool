"""Install example official weights locally; model weights are never committed."""

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def download_yolo():
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    target = PROJECT / "pseudo_labeling/model/object_detection/checkpoint/yolo26n.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(target))
    model = YOLO(str(target))
    if model.task != "detect":
        raise ValueError("Downloaded checkpoint is not a detection model")
    print(f"Official YOLO26n ready: {target}", flush=True)


def download_grounded_sam():
    from huggingface_hub import HfApi, snapshot_download
    from core.io import atomic_bytes, json_bytes

    root = PROJECT / "pseudo_labeling/model/segmentation/checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    model_ids = {
        "grounding_dino": "IDEA-Research/grounding-dino-tiny",
        "sam": "facebook/sam-vit-base",
    }
    revisions = {}
    for name, model_id in model_ids.items():
        revision = HfApi().model_info(model_id).sha
        revisions[name] = revision
        print(f"Downloading {model_id}@{revision} -> {root / name}", flush=True)
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=root / name,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE*", "README.md"],
        )
    config = dict(
        architecture="grounded_sam",
        grounding_dino="grounding_dino",
        sam="sam",
        prompts={"road": 1},
        model_ids=model_ids,
        revisions=revisions,
    )
    atomic_bytes(root / "grounded_sam.json", json_bytes(config))
    print(f"Grounded-SAM ready: {root / 'grounded_sam.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", choices=["yolo26", "grounded-sam", "all"], nargs="?", default="all"
    )
    args = parser.parse_args()
    if args.model in ["yolo26", "all"]:
        download_yolo()
    if args.model in ["grounded-sam", "all"]:
        download_grounded_sam()


if __name__ == "__main__":
    main()
