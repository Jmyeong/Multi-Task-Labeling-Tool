# MT-Labeling

A local browser editor for detection boxes and semantic segmentation. It preserves the manual editing, SAM prompts, review status, backups, and approved-data export workflow of EasyLabeling_Tool. VLM functionality and training code are excluded.

[한국어 사용 안내](docs/GUI_GUIDE.md) · [Code structure](docs/STRUCTURE.md) · [Windows setup](docs/WINDOWS.md)

## Quick start

Python 3.10+ on Windows, Linux or WSL. Manual editing works without PyTorch, SAM or a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_gui.sh
```

On Windows CMD use `run_gui.bat` after activating the environment; in PowerShell use `.\.venv\Scripts\python.exe run.py gui`. The common entry point on either OS is `python run.py gui`. See [Windows setup](docs/WINDOWS.md) for native Windows installation and SAM.

Open http://localhost:8766. Put images in `dataset/`, or click **입력·출력 폴더** to choose an existing input folder and a separate annotation output. Both paths refer to the machine running the server. For VS Code Remote SSH, forward port 8766 in the Ports tab.

```bash
INPUT=/path/to/images OUTPUT_DIR=/path/to/annotations PORT=8766 ./run_gui.sh
# CLI flags can also be supplied:
./run_gui.sh --dataset-root /path/to/images --output-root /path/to/annotations
```

`PYTHON=/path/to/python` selects an existing environment. New outputs start with empty boxes and a 0/background mask, with review status pending. Reopening an existing output preserves saved edits, including existing ignore pixels. Input images and input manifests are never overwritten.

## Layout

```text
MT-Labeling/
├── gui/                    HTTP API and browser editor
├── core/                   Manifest, annotation IO, review transactions and filters
├── sam/                    Optional SAM worker and service
│   └── checkpoint/         User-supplied SAM weights
├── scripts/                Prepare input, generate labels, filter and export
├── pseudo_labeling/
│   └── model/
│       ├── segmentation/       RGB YOLO segmentation network + checkpoint/
│       ├── object_detection/   RGB YOLO detection network + checkpoint/
│       └── multi_task/         Joint RGB/NIR model + checkpoint/
├── dataset/                User input images (or select an external folder)
├── outputs/                Editable labels, manifests, backups and exports
├── run.py                  Shared Windows/Linux GUI and pseudo-label launcher
├── run_gui.sh / .bat        OS-specific wrappers for the same launcher
├── run_pseudo.sh / .bat     OS-specific wrappers for the same launcher
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

Click **Pseudo labeling** in the GUI. Select segmentation, object detection or multi-task, choose a local `.pt` checkpoint or Grounded-SAM bundle `.json`, set the inference Python environment, device and a **new empty output folder**, then click **생성 실행**. Progress, logs, cancellation and **완료 결과 열기** are available in the dialog. Generation uses the current input folder, preserves input files and existing editor labels, and creates pending annotations for review. Closing the dialog does not stop the job. Job configuration/status/logs are saved under `outputs/pseudo_jobs/`.

Install the appropriate PyTorch/torchvision build and `python -m pip install -r requirements-pseudo.txt` in the inference environment. `PSEUDO_PYTHON` sets the GUI's default model interpreter; it can also be entered in the dialog. Manual editing still works without model packages.

| Selection | Network / weights | Inputs and output |
|---|---|---|
| segmentation | `segmentation/network.py`, `grounded_sam.py`, `checkpoint/grounded_sam.json` (or YOLO `.pt`) | Grounding DINO + original SAM, English text + RGB → semantic class-ID PNG |
| object_detection | `object_detection/network.py`, `checkpoint/yolo26n.pt` (or trained YOLO `.pt`) | Official COCO-pretrained YOLO26n, RGB → YOLO boxes |
| multi_task | `pseudo_labeling/model/multi_task/network.py`, `yolo26.py`, `checkpoint.py`, `checkpoint/*.pt` | Existing compatible YOLO26 joint RGB/NIR model → boxes + binary road mask; requires detector template |

Checkpoints are user supplied. External checkpoint paths are also accepted. The single-task loaders use the architecture embedded in the Ultralytics `.pt` checkpoint; modifying a loader alone cannot convert incompatible weights. DAv2-MTL and arbitrary raw state dictionaries are not supported by these loaders.

**Class mapping:** single-task models map checkpoint class names to active GUI class names (case-insensitive), never by identical numeric IDs. Supply JSON such as `{"person": "Person"}` or `{"0": 1}` to override. Segmentation uses the segmentation palette; detection uses box classes. Unmapped classes are skipped and logged; zero matched classes is an error. Overlapping masks use the prediction with the highest confidence. Segmentation-only jobs leave boxes empty; detection-only jobs leave the mask at background 0. These are separate outputs, not a merge into existing annotations.

### Official examples

Both examples have a common Windows/Linux installer. Install a matching PyTorch/torchvision pair for your hardware, then:

```bash
python -m pip install -r requirements-grounded-sam.txt
python -m scripts.download_example_models all
# Individual downloads: yolo26 or grounded-sam instead of all
```

The installer places **official Ultralytics YOLO26n** weights in `object_detection/checkpoint/yolo26n.pt`. It downloads **IDEA-Research/grounding-dino-tiny** and **facebook/sam-vit-base** into `segmentation/checkpoint/grounding_dino/` and `segmentation/checkpoint/sam/`, and writes `segmentation/checkpoint/grounded_sam.json`. Downloaded revisions are recorded in the bundle. Weights and local bundles are excluded from Git; no model download occurs during inference. The GUI selects these examples by default when installed.

YOLO26n uses COCO's 80 classes. `person` automatically maps to the default `Person` target; Golfcart/Tree are not COCO classes and are not automatically substituted with car/plant. Other classes require adding a GUI target or an explicit mapping.

For Grounded-SAM, the mapping field becomes **English text prompts → active segmentation class**. The default `{"road": 1}` finds road and assigns its mask to GUI ID 1. Add a class in the GUI first to use prompts such as `{"road": 1, "water": 2}`. Grounding DINO runs each prompt separately, then SAM segments its boxes and chooses the mask with highest predicted IoU. Overlaps use the highest Grounding DINO detection confidence. Confidence controls box filtering; Text threshold controls phrase-token filtering. Grounding DINO/SAM use their official preprocessing and original RGB geometry, so the YOLO height/width fields are hidden. Grounded-SAM does not retrain either model or guarantee correct semantic labels; inspect generated masks before approval.

Grounded-SAM uses Transformers' implementation of the [official Grounded-SAM workflow](https://github.com/IDEA-Research/Grounded-Segment-Anything), following the [Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) and [SAM](https://huggingface.co/docs/transformers/model_doc/sam) APIs. It requires no GroundingDINO CUDA extension build on either OS. Sources and weight terms: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny), [SAM ViT-B](https://huggingface.co/facebook/sam-vit-base).

The joint model retains its trained Golfcart/Person/Tree/Undef_obj and binary road mapping. It requires the default class names/active targets and matched NIR. Standard COCO weights and DAv2-MTL checkpoints are incompatible with the joint loader. Road predictions near the confidence threshold may be marked 255/ignore. The GUI also includes frames with source semantic/instance labels, but never changes their source labels.

The existing joint CLI remains available on Windows and Linux:

```bash
INPUT=/path/to/rgb-nir-dataset OUTPUT_DIR=/path/to/new-pseudo-labels \
CHECKPOINT=/path/to/best_joint.pt MODEL_TEMPLATE=/path/to/yolo26s.pt \
DEVICE=0 ./run_pseudo.sh --batch 8
# Windows: set the same variables and run run_pseudo.bat
# Review: select the generated output folder in the GUI.
```

The joint CLI defaults to unlabeled source frames; `--include-labeled` includes all frames. Generation resumes compatible outputs without replacing completed annotations; a different model/configuration requires a new output. GUI launches always require an empty new folder. Legacy `hsvnet` checkpoints additionally require the separately installed fusion repository and weights via CLI `--fusion-repo` and `--fusion-checkpoint`. Input defaults to 352×640. Small shim modules at the previous `pseudo_labeling/model/*.py` paths retain import compatibility; network code now lives under `multi_task/`.

## Utilities

Run from the repository root; these Python module commands are the same on Windows and Linux and do not start training.

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

CI tests Windows and Ubuntu on Python 3.10–3.12, including native file locks, subprocess-pipe deadlines, UTF-8 paths and the Windows batch wrapper. Tests use temporary datasets only. Two generator tests require optional pseudo-label dependencies and skip in a GUI-only environment. Datasets, outputs, environments, checkpoints and caches are excluded by `.gitignore`. The original EasyLabeling_Tool is independent of this copy.

Third-party packages retain their upstream licenses. SAM 2: [official repository](https://github.com/facebookresearch/sam2). Ultralytics: [official licensing information](https://www.ultralytics.com/license). No new license is assigned to the copied project code in this restructuring.
