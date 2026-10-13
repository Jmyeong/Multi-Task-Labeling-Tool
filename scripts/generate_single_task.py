"""Generate RGB detection or segmentation labels into a separate workspace."""

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from core.annotations import yolo_line
from core.dataset import annotation_paths
from core.file_lock import exclusive_lock
from core.io import atomic_bytes, inside, json_bytes
from core.index import rebuild_index
from core.workspace import prepare_workspace
from pseudo_labeling.predictions import class_mapping, convert_result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_prediction(root, row, boxes, mask, signature, task):
    paths = annotation_paths(row)
    paths.update(road_probability_file="", preview_file="")
    height, width = mask.shape
    record = {
        **row,
        **paths,
        "width": width,
        "height": height,
        "label_source": "pseudo",
        "review_status": "pending",
        "generation_signature": signature,
        "pseudo_task": task,
        "detections": boxes,
        "detection_count": len(boxes),
        "road_fraction": float((mask == 1).mean()),
        "uncertain_fraction": float((mask == 255).mean()),
        "class_fractions": {str(i): float((mask == i).mean()) for i in np.unique(mask)},
        "review_reasons": ["pseudo_review_required"],
    }
    for key in ("detection_file", "road_file", "annotation_file"):
        inside(root, paths[key]).parent.mkdir(parents=True, exist_ok=True)
    label = "\n".join(yolo_line(box, width, height) for box in boxes)
    atomic_bytes(
        inside(root, paths["detection_file"]), (label + ("\n" if label else "")).encode("utf-8")
    )
    png = io.BytesIO()
    Image.fromarray(mask).save(png, format="PNG")
    atomic_bytes(inside(root, paths["road_file"]), png.getvalue())
    # Completion marker last; interrupted outputs can be regenerated on resume.
    atomic_bytes(inside(root, paths["annotation_file"]), json_bytes(record))
    return record


def generate(config):
    task = config["task"]
    if task not in ("segmentation", "object_detection"):
        raise ValueError("Unsupported single task")
    source, root = Path(config["dataset_root"]).resolve(), Path(config["output_root"]).resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Choose an output separate from the input tree")
    weight = Path(config["checkpoint"]).expanduser().resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    network = importlib.import_module(f"pseudo_labeling.model.{task}.network").load_network(weight)
    if weight.suffix.lower() == ".json":
        network.configure(
            config.get("class_mapping") or network.bundle.get("prompts", {"road": 1}),
            config.get("text_threshold", 0.25),
        )
        config["class_mapping"] = network.prompts
    schema = config["schema"]
    mapping = class_mapping(
        network.names,
        schema["segmentation" if task == "segmentation" else "detection"],
        config.get("class_mapping"),
    )
    print("Model classes: " + json.dumps(network.names, ensure_ascii=False), flush=True)
    print("Class mapping: " + json.dumps(mapping), flush=True)
    missing = {k: v for k, v in network.names.items() if k not in mapping}
    if missing:
        print("Unmapped classes skipped: " + json.dumps(missing, ensure_ascii=False), flush=True)
    identity = {**config, "checkpoint_sha256": file_hash(weight)}
    if weight.suffix.lower() == ".json":
        identity["model_files_sha256"] = {
            f"{name}/{p.name}": file_hash(p)
            for name, directory in network.paths.items()
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in (".json", ".txt", ".safetensors")
        }
    signature = hashlib.sha256(json_bytes(identity)).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    run_file = root / "run.json"
    if run_file.exists():
        if json.loads(run_file.read_text(encoding="utf-8"))["generation_signature"] != signature:
            raise ValueError("Generation mismatch; use a new output folder")
    elif any(p.name != ".generation.lock" for p in root.iterdir()):
        raise ValueError("Output is not empty; existing manual labels are preserved")
    manifest, rows = prepare_workspace(source, root, initialize=False)
    if config.get("limit"):
        rows = rows[: config["limit"]]
    with exclusive_lock(root / ".generation.lock"):
        if (
            run_file.exists()
            and json.loads(run_file.read_text(encoding="utf-8"))["generation_signature"]
            != signature
        ):
            raise ValueError("Output changed while loading the model; existing labels preserved")
        schema_file = root / "label_schema.json"
        if schema_file.exists() and json.loads(schema_file.read_text(encoding="utf-8")) != schema:
            raise ValueError("Output class settings have changed; existing settings preserved")
        atomic_bytes(root / "label_schema.json", json_bytes(schema))
        identity["generation_signature"] = signature
        atomic_bytes(run_file, json_bytes(identity))
        pending = []
        for row in rows:
            path = inside(root, annotation_paths(row)["annotation_file"])
            if not path.exists():
                pending.append(row)
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("generation_signature") != signature:
                raise ValueError("Existing annotation belongs to a different generator")
            for field in ("road_file", "detection_file"):
                if not inside(root, record[field]).is_file():
                    raise ValueError("Existing annotation incomplete; preserved")
        print(
            f"Selected={len(rows)} existing={len(rows) - len(pending)} pending={len(pending)} output={root}",
            flush=True,
        )
        try:
            for index, row in enumerate(pending, 1):
                with Image.open(inside(source, row["image_file"])) as original:
                    rgb = original.convert("RGB")
                prediction = network.predict(
                    source=rgb,
                    device=config["device"],
                    imgsz=[config["height"], config["width"]],
                    conf=config["conf"],
                    max_det=config["max_det"],
                    retina_masks=True,
                    verbose=False,
                )[0]
                boxes, mask = convert_result(prediction, task, mapping, schema)
                write_prediction(root, row, boxes, mask, signature, task)
                print(f"Generated {index}/{len(pending)}", flush=True)
        finally:
            summary = rebuild_index(root)
            summary.update(
                segmentation="Class-ID PNG; 0=background, 255=ignore",
                probability_encoding="not generated",
                pseudo_task=task,
            )
            atomic_bytes(root / "summary.json", json_bytes(summary))
        print("Generation complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    generate(json.loads(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
