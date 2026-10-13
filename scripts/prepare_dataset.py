"""Create blank review annotations or an input manifest without modifying images."""

import argparse
from pathlib import Path
from core.workspace import prepare_workspace


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "dataset")
    parser.add_argument("--output-root", type=Path, default=root / "outputs/review")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest, rows = prepare_workspace(
        args.dataset_root, args.output_root, initialize=not args.manifest_only
    )
    print(f"{len(rows)} images; manifest: {manifest}; annotations: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
