"""Map checkpoint classes by name/explicit ID, never by coincidental class number."""

import numpy as np


def class_mapping(names, targets, explicit=None):
    explicit = explicit or {}
    active = [t for t in targets if t["active"] and t["id"] != 255]
    by_id = {t["id"]: t for t in active}
    by_name = {t["name"].casefold(): t["id"] for t in active}
    result = {}
    names = dict(enumerate(names)) if isinstance(names, list) else names
    for source_id, source_name in names.items():
        target = explicit.get(str(source_id), explicit.get(source_name))
        if target is None:
            target = by_name.get(source_name.casefold())
        elif isinstance(target, str) and target.casefold() in by_name:
            target = by_name[target.casefold()]
        elif isinstance(target, str) and target.isdecimal():
            target = int(target)
        if target is not None:
            if type(target) is not int or target not in by_id:
                raise ValueError(
                    f"Invalid target for {source_name}: {target}. Use an active GUI class ID/name."
                )
            result[int(source_id)] = target
    unknown = set(explicit) - {str(i) for i in names} - set(names.values())
    if unknown:
        raise ValueError(f"Unknown checkpoint classes in mapping: {sorted(unknown)}")
    if not result:
        raise ValueError(
            f"No matching classes. Model classes: {names}. Set class_mapping explicitly."
        )
    return result


def convert_result(result, task, mapping, schema):
    """Return canonical boxes/mask at the original RGB size (retina_masks=True)."""
    height, width = result.orig_shape
    mask = np.zeros((height, width), np.uint8)
    boxes = []
    names = {t["id"]: t["name"] for t in schema["detection"]}
    if result.boxes is None:
        return boxes, mask
    detections = result.boxes.data.detach().cpu().numpy()
    masks = None
    if task == "segmentation" and len(detections):
        if result.masks is None:
            raise ValueError("Segmentation checkpoint returned boxes without masks.")
        masks = result.masks.data.detach().cpu().numpy()
        if masks.shape != (len(detections), height, width):
            raise ValueError(f"Expected original-resolution masks, got {masks.shape}.")
    # Low confidence first so the highest confidence owns overlapping pixels.
    for i in sorted(range(len(detections)), key=lambda i: float(detections[i, 4])):
        x1, y1, x2, y2, confidence, source_id = detections[i, :6]
        if not np.isfinite([x1, y1, x2, y2, confidence, source_id]).all():
            raise ValueError("Non-finite model prediction")
        target = mapping.get(int(source_id))
        if target is None:
            continue
        if task == "segmentation":
            mask[masks[i] >= 0.5] = target
        else:
            xyxy = [
                float(np.clip(x1, 0, width)),
                float(np.clip(y1, 0, height)),
                float(np.clip(x2, 0, width)),
                float(np.clip(y2, 0, height)),
            ]
            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                boxes.append(
                    dict(
                        class_id=target,
                        class_name=names[target],
                        bbox_xyxy=xyxy,
                        confidence=float(confidence),
                    )
                )
    return boxes, mask
