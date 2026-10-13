"""Original image geometry for compatible YOLO26 joint checkpoints."""

import cv2

import numpy as np


def normalize_image_size(size):
    """Convert a scalar or two-element size into ``(height, width)``."""
    if isinstance(size, (int, float)):
        return int(size), int(size)
    height, width = size
    return int(height), int(width)


def resolve_image_size(args):
    """Resolve CLI arguments into the network input ``(height, width)``."""
    height = args.height
    width = args.imgsz if args.imgsz is not None else args.width
    height, width = normalize_image_size((height, width))
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {height}x{width}")
    return height, width


def letterbox_all(image, nir, depth, road, boxes, size):
    target_height, target_width = normalize_image_size(size)
    height, width = image.shape[:2]
    scale = min(target_height / height, target_width / width)
    new_width, new_height = round(width * scale), round(height * scale)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    out_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    out_nir = np.zeros((target_height, target_width), dtype=np.uint8)
    out_depth = np.zeros((target_height, target_width), dtype=np.float32)
    out_road = np.full((target_height, target_width), 255, dtype=np.uint8)
    region = np.s_[pad_y : pad_y + new_height, pad_x : pad_x + new_width]
    out_image[region] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # The synchronized NIR sensor stores 848x480 frames while RGB is 1280x720.
    # Resize it into the RGB coordinate system before applying the exact same
    # letterbox transform so corresponding content stays pixel aligned.
    nir = cv2.resize(nir, (width, height), interpolation=cv2.INTER_LINEAR)
    out_nir[region] = cv2.resize(nir, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    out_depth[region] = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    out_road[region] = cv2.resize(road, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    transformed = []
    for class_id, cx, cy, bw, bh in boxes:
        transformed.append(
            (
                class_id,
                (cx * width * scale + pad_x) / target_width,
                (cy * height * scale + pad_y) / target_height,
                bw * width * scale / target_width,
                bh * height * scale / target_height,
            )
        )
    return out_image, out_nir, out_depth, out_road, transformed
