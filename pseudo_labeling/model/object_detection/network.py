"""Ultralytics detection network loader; model architecture is stored in the .pt."""


def load_network(checkpoint):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect")
    if model.task != "detect":
        raise ValueError("Select a detection checkpoint, not a segmentation/classification model.")
    return model
