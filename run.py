#!/usr/bin/env python3
"""Common Windows/Linux entry point: python run.py gui|pseudo [options]."""

import argparse
import importlib
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent


def launch_arguments(mode, extra, environment=None):
    """Keep environment-variable defaults identical across every shell wrapper."""
    env = os.environ if environment is None else environment
    result = ["--dataset-root", env.get("INPUT") or str(PROJECT / "dataset")]
    if mode == "gui":
        result += [
            "--output-root",
            env.get("OUTPUT_DIR") or str(PROJECT / "outputs/review"),
            "--host",
            env.get("HOST") or "127.0.0.1",
            "--port",
            env.get("PORT") or "8766",
        ]
    else:
        result += [
            "--output-root",
            env.get("OUTPUT_DIR") or str(PROJECT / "outputs/pseudo_labels"),
            "--checkpoint",
            env.get("CHECKPOINT") or str(PROJECT / "pseudo_labeling/checkpoint/best_joint.pt"),
            "--model-template",
            env.get("MODEL_TEMPLATE") or str(PROJECT / "pseudo_labeling/checkpoint/yolo26s.pt"),
            "--device",
            env.get("DEVICE") or "cpu",
        ]
    return [*result, *extra]  # Explicit CLI arguments override environment defaults.


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gui", "pseudo"])
    args = parser.parse_args(argv[:1])
    # UTF-8 logs even when redirected by Windows CMD or a subprocess.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    os.chdir(PROJECT)
    module_name = "gui.server" if args.mode == "gui" else "scripts.generate_pseudo_labels"
    sys.argv = [module_name, *launch_arguments(args.mode, argv[1:])]
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
