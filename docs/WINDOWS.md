# Windows / Linux common execution

Python 3.10+ is required. The manual GUI needs neither WSL nor a GPU. Use the same Python modules on Windows and Linux; `run.py` is the shared entry point. Choose a compatible PyTorch/torchvision environment separately for SAM or automatic labels.

## Windows CMD

Run these commands from the downloaded repository folder:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
run_gui.bat
```

Open http://localhost:8766. Choose input/output folders in the GUI, or specify Windows paths:

```bat
set "INPUT=D:\data\RGB images"
set "OUTPUT_DIR=D:\labels\review"
run_gui.bat
```

An existing interpreter can be selected with `set "PYTHON=C:\path to env\Scripts\python.exe"`. CLI arguments override environment defaults. `PORT`, `HOST`, `SAM_PYTHON`, and `SAM_CHECKPOINT` work as on Linux.

## Windows PowerShell

Activation is optional, so no script execution-policy change is needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py gui
```

To pass paths directly:

```powershell
.\.venv\Scripts\python.exe run.py gui --dataset-root "D:\data\RGB images" --output-root "D:\labels\review"
```

Or use environment variables and the batch wrapper:

```powershell
$env:PYTHON = (Resolve-Path ".venv\Scripts\python.exe").Path
$env:INPUT = "D:\data\RGB images"
$env:OUTPUT_DIR = "D:\labels\review"
.\run_gui.bat
```

The input path is a folder on the computer running the server. If the server runs remotely on Linux, select its Linux folders through the browser, even when the browser runs on Windows. Native Windows runs can select `C:\...` or `D:\...` folders. Spaces and Korean names are supported.

## Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_gui.sh
# Same entry point on either operating system:
python run.py gui --dataset-root /path/to/images --output-root /path/to/labels
```

## Automatic labels

Install the optional model dependencies in the selected environment and supply the compatible joint-model checkpoint/template. The model structures and weights have the same layout on both operating systems.

```bat
python -m pip install -r requirements-pseudo.txt
set "INPUT=D:\data\rgb-nir"
set "OUTPUT_DIR=D:\labels\pseudo"
set "CHECKPOINT=D:\models\best_joint.pt"
set "MODEL_TEMPLATE=D:\models\yolo26s.pt"
set "DEVICE=cpu"
run_pseudo.bat --batch 1 --workers 0
```

`--workers 0` is useful for a first run; normal DataLoader workers are also supported through the guarded common entry point. On Linux use `run_pseudo.sh`; on either OS use `python run.py pseudo` with the same arguments. CUDA support depends on the installed PyTorch build and GPU.

## Optional native Windows SAM

Install matching PyTorch/torchvision first using the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install SAM 2 in a separate environment using the [official repository](https://github.com/facebookresearch/sam2). For a native Windows install without the optional CUDA extension build, SAM's [setup.py](https://github.com/facebookresearch/sam2/blob/main/setup.py) supports `SAM2_BUILD_CUDA=0`:

```bat
set "SAM2_BUILD_CUDA=0"
python -m pip install "git+https://github.com/facebookresearch/sam2.git"
```

This command requires Git. The tool already uses `apply_postprocessing=False`, so its predictor does not need that optional extension. GPU inference still requires an appropriate PyTorch CUDA build; disabling the extension build does not turn GPU inference off.

Place the matching checkpoint in `sam/checkpoint/`, then start the GUI with its SAM interpreter:

```bat
set "SAM_PYTHON=C:\path to sam env\Scripts\python.exe"
run_gui.bat --sam-checkpoint "D:\models\sam2.1_hiera_large.pt" --sam-device cpu
```

Change `--sam-device` to `cuda:0` for a compatible GPU. SAM/Ultralytics dependencies and checkpoints are optional, are not distributed here, and must be installed for the chosen OS.

## Compatibility checks

The GitHub Actions matrix runs on actual Windows and Ubuntu with Python 3.10, 3.11 and 3.12. It exercises GUI startup, save/backup/rollback, labels, exports, process locks, Unicode paths, and SAM worker-pipe contracts without downloading model weights. GPU/model accuracy tests require your own model environment and weights.
