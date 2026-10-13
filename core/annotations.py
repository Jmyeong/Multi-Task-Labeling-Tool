"""Class mappings, instance-mask conversion and YOLO box serialization."""

import numpy as np

from PIL import Image

CLASS_NAMES = ["Golfcart", "Person", "Tree", "Undef_obj"]

RGB_TO_SOURCE_CLASS = {
    (96, 96, 192): 2,
    (192, 96, 96): 2,
    (160, 0, 128): 3,
    (128, 0, 160): 3,
    (0, 96, 144): 4,
    (144, 96, 0): 4,
    (172, 233, 115): 7,
    (249, 25, 35): 7,
}

SOURCE_TO_YOLO = {3: 0, 2: 1, 4: 2, 7: 3}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def align_label(path, image_size):
    label = Image.open(path).convert("RGB")
    width, height = image_size
    if label.size == (width, height * 2):
        # Some exported labels contain an identifier view above the actual mask.
        label = label.crop((0, height, width, height * 2))
    if label.size != image_size:
        label = label.resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(label)


def semantic_class_mask(semantic_rgb):
    result = np.full(semantic_rgb.shape[:2], -1, dtype=np.int16)
    for color, source_class in RGB_TO_SOURCE_CLASS.items():
        result[np.all(semantic_rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = source_class
    return result


def majority_source_class(instance_mask, class_mask):
    values = class_mask[instance_mask]
    values = values[values >= 0]
    if values.size == 0:
        return None
    classes, counts = np.unique(values, return_counts=True)
    return int(classes[counts.argmax()])


def boxes_from_instance(instance_rgb, semantic_rgb, min_area):
    class_mask = semantic_class_mask(semantic_rgb)
    object_region = class_mask >= 0
    boxes = []
    encoded = (
        instance_rgb[:, :, 0].astype(np.uint32) << 16
        | instance_rgb[:, :, 1].astype(np.uint32) << 8
        | instance_rgb[:, :, 2].astype(np.uint32)
    )
    for color_value in np.unique(encoded[object_region]):
        color_tuple = (
            int((color_value >> 16) & 255),
            int((color_value >> 8) & 255),
            int(color_value & 255),
        )
        if color_tuple == (0, 0, 0):
            continue
        instance_mask = (encoded == color_value) & object_region
        area = int(instance_mask.sum())
        if area < min_area:
            continue
        source_class = majority_source_class(instance_mask, class_mask)
        if source_class not in SOURCE_TO_YOLO:
            continue
        ys, xs = np.where(instance_mask)
        boxes.append(
            {
                "class_id": SOURCE_TO_YOLO[source_class],
                "class_name": CLASS_NAMES[SOURCE_TO_YOLO[source_class]],
                "source_class_id": source_class,
                "instance_color": list(color_tuple),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ],
                "area": area,
            }
        )
    return sorted(boxes, key=lambda box: (box["class_id"], box["bbox_xyxy"]))


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box["bbox_xyxy"]
    center_x = (x1 + x2) / (2.0 * width)
    center_y = (y1 + y2) / (2.0 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{box['class_id']} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
