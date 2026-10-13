# MT-Labeling

A local browser editor for detection boxes and binary road segmentation. It preserves the manual editing, SAM prompts, review status, backups, and approved-data export workflow of EasyLabeling_Tool. VLM functionality and training code are excluded.

[한국어 사용 안내](docs/GUI_GUIDE.md) · [Code structure](docs/STRUCTURE.md)

## Quick start

Python 3.10+ on Linux or WSL. Manual editing works without PyTorch, SAM or a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_gui.sh
```

Open http://localhost:8766. Put images in `dataset/`, or click **입력·출력 폴더** to choose an existing input folder and a separate annotation output. Both paths refer to the machine running the server. For VS Code Remote SSH, forward port 8766 in the Ports tab.

```bash
INPUT=/path/to/images OUTPUT_DIR=/path/to/annotations PORT=8766 ./run_gui.sh
# CLI flags can also be supplied:
./run_gui.sh --dataset-root /path/to/images --output-root /path/to/annotations
```

`PYTHON=/path/to/python` selects an existing environment. New outputs start with empty boxes and a 255/ignore mask. Reopening an existing output preserves saved edits. Input images and input manifests are never overwritten.

## Layout

```text
MT-Labeling/
├── gui/                    HTTP API and browser editor
├── core/                   Manifest, annotation IO, review transactions and filters
├── sam/                    Optional SAM worker and service
│   └── checkpoint/         User-supplied SAM weights
├── scripts/                Prepare input, generate labels, filter and export
├── pseudo_labeling/
│   ├── model/              Joint YOLO26 model, checkpoint loader and preprocessing
│   └── checkpoint/         User-supplied pseudo-label weights
├── dataset/                User input images (or select an external folder)
├── outputs/                Editable labels, manifests, backups and exports
├── tests/                  Temporary-fixture regression tests
└── docs/                   GUI and architecture guides
```

## Annotation format

Default detection classes are `0 Golfcart`, `1 Person`, `2 Tree`, `3 Undef_obj`. Use **클래스·색상 설정** in the GUI to add/remove targets, rename classes, select colors and restore removed classes independently for boxes and segmentation.
YOLO `.txt` boxes use normalized `class x_center y_center width height`. Segmentation masks are lossless single-channel PNGs storing class IDs. Defaults are `0 background/non-road`, `1 road`, `255 ignore`; additional target IDs are assigned from 2–254. JSON records store frame identity, teacher predictions, review status, revisions, and notes. Class names, IDs, colors and active status are saved in `label_schema.json` in the current output workspace. Removal hides a target from new labeling without deleting existing boxes or pixels. IDs are never reassigned; removed targets can be restored. Background 0 and ignore 255 are reserved and remain available. Color changes affect visualization only. Save/discard current frame edits before applying settings. Settings changes invalidate stale frame revisions in other tabs.

Images may be a plain folder or a nested tree. NIR is optional for manual editing. When present, place same-stem images in a sibling `nir/` or `infrared/` folder. A DY `manifest.csv` can instead specify modalities and train/test metadata. Raw folders without a manifest receive a **separate output manifest** and default to train; an imported manifest preserves train/test assignments.

## Optional SAM

Install SAM 2 in its own environment following the [official installation guide](https://github.com/facebookresearch/sam2#installation), then place the matching checkpoint in `sam/checkpoint/`. SAM 2 requires Python 3.10+, PyTorch 2.5.1+ and torchvision 0.20.1+. No SAM code or weights from Meta are vendored here; `sam/` contains this tool's integration.

```bash
SAM_PYTHON=/path/to/sam-env/bin/python ./run_gui.sh \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam-device cuda:0
```

SAM starts lazily when requested, previews the mask without saving, and applies only when the user chooses to. Manual tools continue to work when SAM is unavailable.

## Automatic labels

Install a compatible PyTorch/torchvision pair for your hardware and `pip install -r requirements-pseudo.txt`. Put **compatible joint-model weights** and the detector template in `pseudo_labeling/checkpoint/`. Standard COCO-only and DAv2-MTL checkpoints are incompatible. No dataset or weights are bundled.

```bash
INPUT=/path/to/rgb-nir-dataset OUTPUT_DIR=/path/to/pseudo-labels \
CHECKPOINT=/path/to/best_joint.pt MODEL_TEMPLATE=/path/to/yolo26s.pt \
DEVICE=0 ./run_pseudo.sh --batch 8

# Review the generated annotations:
INPUT=/path/to/rgb-nir-dataset OUTPUT_DIR=/path/to/pseudo-labels ./run_gui.sh
```

GUI class settings do not retrain the automatic model: it still predicts its original four detection classes and binary road mask. Additional classes can be labeled manually or with SAM. If original model class names/IDs are changed, use a separate default workspace for automatic generation.

Existing RGB/NIR fusion models require matched NIR images. Automatic generation selects frames without semantic/instance labels. Default input is 352×640; the existing model and letterbox geometry are preserved. The generator resumes compatible completed outputs and refuses to overwrite labels from another generation configuration. Use a new output folder rather than the blank manual workspace for first-time generation. Legacy `hsvnet` checkpoints also require their separately installed fusion repository and weights via `--fusion-repo` and `--fusion-checkpoint`.

## Utilities

Run from the repository root; these commands do not start training.

```bash
python -m scripts.prepare_dataset --dataset-root /path/to/images --output-root /path/to/labels
python -m scripts.prepare_dataset --dataset-root /path/to/images --output-root /path/to/labels --manifest-only
python -m scripts.filter_pseudo_boxes --dataset-root /path/to/images --output-root /path/to/labels
python -m scripts.filter_pseudo_roads --dataset-root /path/to/images --output-root /path/to/labels
# Both filters default to dry-run. Add --apply to back up and commit changes.
python -m scripts.extract_dataset --dataset-root /path/to/images --output-root /path/to/labels \
  --export-parent /path/to/exports --name reviewed_v1
```

The GUI **데이터 추출** button provides the same export. Only approved frames are included. RGB and labels are exported; existing NIR, depth and calibration files are copied when supplied. Depth files are copied without resampling or modification. Exports contain `manifest.csv`, `train.csv`, `test.csv`, per-group `frames.csv`, `label_schema.json`, `classes.txt` and a SHA-256 summary. Reopening exported data in a new workspace retains its custom classes and colors. Existing target folders are never replaced.

## Development

```bash
python -m unittest discover -s tests -v
```

Tests use temporary datasets only. Two generator tests require optional pseudo-label dependencies and skip in a GUI-only environment. Datasets, outputs, environments, checkpoints and caches are excluded by `.gitignore`. The original EasyLabeling_Tool is independent of this copy.

Third-party packages retain their upstream licenses. SAM 2: [official repository](https://github.com/facebookresearch/sam2). Ultralytics: [official licensing information](https://www.ultralytics.com/license). No new license is assigned to the copied project code in this restructuring.
