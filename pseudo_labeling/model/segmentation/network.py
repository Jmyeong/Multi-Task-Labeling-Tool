"""Ultralytics instance-segmentation network loader, converted to semantic IDs."""


def load_network(checkpoint):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="segment")
    if model.task != "segment":
        raise ValueError("Select a YOLO segmentation checkpoint (*-seg.pt or a trained model).")
    return model
