# Code structure

The GUI has no PyTorch/Ultralytics import dependency. `run.py` handles shared Windows/Linux launch defaults; `.sh` and `.bat` wrappers delegate to it. SAM is isolated in a lazy subprocess; pseudo labeling runs in a separate subprocess, launched from the GUI or CLI.

```mermaid
flowchart LR
    B[Browser: gui/web] --> H[gui/server.py]
    H --> W[core/workspace.py]
    W --> I[User input: read only]
    W --> O[Separate output manifest]
    H --> R[core/store.py]
    R --> L[Editable labels and backups]
    H --> S[sam/service.py]
    S --> SW[sam/worker.py]
    H --> E[scripts/extract_dataset.py]
    E --> X[Approved data export]
    H --> J[pseudo_labeling/service.py]
    J --> P[Generation subprocess]
    P --> M[model: segmentation / object_detection / multi_task]
    M --> C[Each task: checkpoint/]
    P --> L
```

| Area | Responsibilities |
|---|---|
| `gui/server.py` | API routing, write tokens, workspace switching, local folder browser, optional SAM/export services |
| `gui/web/app.js` | Frame navigation, box/mask editing, undo/redo, review controls, keyboard shortcuts |
| `gui/web/workspace.js` | Input/output selection and server-side folder browser |
| `gui/web/labels.js` | Class settings dialog, color pickers, add/remove/restore and live application |
| `gui/web/export.js` | Export dialog, start/progress/result |
| `core/workspace.py` | Discover inputs, validate identities, create separate manifests and initial annotations |
| `core/store.py` | Label loading, validation, revisions, transactional save/rollback, review history |
| `core/label_schema.py` | Workspace class names, stable IDs, colors, reversible removal, setting revisions and backups |
| `core/file_lock.py` | Common nonblocking process locks: Windows CRT and POSIX flock |
| `core/io.py` | Path containment checks and atomic file writes |
| `core/dataset.py` | Manifest schema, annotation paths, existing training-data adapter |
| `core/annotations.py` | Existing class mapping, legacy instance-mask conversion and YOLO serialization |
| `core/index.py` | Rebuild label/review index files |
| `core/box_filter.py`, `core/road_filter.py` | Pure postprocessing rules |
| `sam/pipe_reader.py` | Threaded raw-pipe JSON-line reader with size limits and deadlines on Windows/Linux |
| `sam/service.py`, `sam/worker.py` | JSON-line worker protocol, prompt validation, lazy image cache and SAM inference |
| `scripts/prepare_dataset.py` | Discover/prepare raw input folders |
| `scripts/generate_pseudo_labels.py` | Batched inference and resumable pseudo labels |
| `scripts/filter_pseudo_*.py` | Dry-run or backed-up filtering of untouched pending labels |
| `scripts/extract_dataset.py` | Approved snapshot export, ordering, integrity checks, CLI |
| `pseudo_labeling/model/multi_task/yolo26.py` | Existing joint YOLO26, RGB/NIR fusion, depth and road heads |
| `pseudo_labeling/model/multi_task/checkpoint.py` | Strict compatible checkpoint loading and detection decoding |
| `pseudo_labeling/model/multi_task/preprocess.py` | Existing 352×640 letterbox geometry |

To change frontend layout, edit `gui/web/`. To change label storage, edit `core/store.py` and validate rollback tests. To add another pseudo-label architecture, implement its structure/loader under `pseudo_labeling/model/` and adapt the generation CLI; do not import training loops into the GUI. Dataset modalities are optional for manual editing/export, but each automatic model may require its own inputs.

The existing YOLO normalized-box format and PNG mask storage are retained. Dynamic class metadata is stored and exported as `label_schema.json`; masks can use additional class IDs. Legacy model helpers in `core/annotations.py` keep their trained class mapping independently of manual GUI settings. No training runs, dataset copies, model weights, VLM backend, external fusion source trees or original repository Git metadata are bundled.

| New component | Responsibilities |
|---|---|
| `gui/web/pseudo.js` | Model/checkpoint/environment selection, class mapping, progress, cancel, open results |
| `pseudo_labeling/service.py` | Validated job launch, background process, UTF-8 logs and persistent job status |
| `pseudo_labeling/registry.py` | Fixed architecture catalog and checkpoint discovery without model imports |
| `pseudo_labeling/predictions.py` | Safe class mapping, original-resolution masks, confidence overlap rule |
| `scripts/generate_single_task.py` | RGB detection/segmentation generator, pending records, resume/signature validation |
| `pseudo_labeling/model/{segmentation,object_detection}/network.py` | Lazy Ultralytics network loaders, task validation |
| `pseudo_labeling/model/multi_task/network.py` | Joint network/loader entry point |

Keep user weights inside each task's `checkpoint/`, or select an external path. Compatibility shims at the old model module paths support older imports. The GUI chooses only catalogued loaders, not arbitrary uploaded Python code.
