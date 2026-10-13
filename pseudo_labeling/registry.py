"""Fixed model catalog. Listing models never imports Torch or loads a checkpoint."""

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent / "model"
MODELS = {
    "segmentation": {
        "name": "YOLO Segmentation (RGB)",
        "network": "segmentation/network.py",
        "nir": False,
    },
    "object_detection": {
        "name": "YOLO Object Detection (RGB)",
        "network": "object_detection/network.py",
        "nir": False,
    },
    "multi_task": {
        "name": "YOLO26 Multi-task (RGB + NIR)",
        "network": "multi_task/network.py",
        "nir": True,
    },
}


def catalog():
    return [
        dict(
            id=key,
            **value,
            checkpoint_dir=str(MODEL_ROOT / key / "checkpoint"),
            checkpoints=[
                str(p.resolve()) for p in sorted((MODEL_ROOT / key / "checkpoint").glob("*.pt"))
            ],
        )
        for key, value in MODELS.items()
    ]
