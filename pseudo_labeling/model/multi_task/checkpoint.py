"""Strict joint-checkpoint loading and native YOLO26 detection decoding."""

from .preprocess import resolve_image_size
from .yolo26 import (
    Yolo26DepthDetRoad,
    build_depth_nir_encoder,
    build_gated_feature_fusion,
    build_pixel_aligned_image_fusion,
    build_scalar_feature_fusion,
    build_yolo26_detector,
    discover_dense_features,
    validate_environment,
)


from pathlib import Path


from types import SimpleNamespace


import torch


def validate_test_environment(args):
    # Reuse the training environment check without exposing irrelevant options.
    environment_args = SimpleNamespace(
        height=args.height,
        width=args.width,
        imgsz=args.imgsz,
        device=args.device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_init=8.0,
        head_warmup_epochs=0,
        input_mode=args.input_mode,
        fusion_repo=args.fusion_repo,
        fusion_checkpoint=args.fusion_checkpoint,
        nir_dropout=0.0,
        nir_mismatch=0.0,
    )
    validate_environment(environment_args)


def load_model(args, device):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"YOLO26 joint checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if not str(checkpoint.get("architecture", "")).startswith("YOLO26"):
        raise ValueError(
            "This is not a train_yolo26_mtl.py checkpoint: "
            f"architecture={checkpoint.get('architecture')!r}"
        )

    checkpoint_mode = checkpoint.get("input_mode") or checkpoint_args.get("input_mode")
    if checkpoint_mode is None:
        checkpoint_mode = "hsvnet"  # checkpoints produced before input_mode was recorded
    if args.input_mode != "auto" and args.input_mode != checkpoint_mode:
        raise ValueError(
            f"Requested input mode {args.input_mode!r} does not match checkpoint "
            f"mode {checkpoint_mode!r}"
        )
    args.input_mode = checkpoint_mode

    image_size = resolve_image_size(args)
    template = args.model_template or checkpoint_args.get("model", "yolo26s.pt")
    build_args = SimpleNamespace(
        model=template,
        dataset_root=args.dataset_root,
        epochs=int(checkpoint_args.get("epochs", 100)),
        height=image_size[0],
        width=image_size[1],
        imgsz=None,
        input_mode=checkpoint_mode,
        fusion_repo=(
            args.fusion_repo
            or checkpoint_args.get(
                "fusion_repo",
                "external/Pixel_aligned_RGB_NIR_Stereo",
            )
        ),
        fusion_checkpoint=(
            args.fusion_checkpoint
            or checkpoint_args.get(
                "fusion_checkpoint",
                "external/Pixel_aligned_RGB_NIR_Stereo/weights/model_image_fusion.pth",
            )
        ),
    )
    # Fill resolved paths before validation so empty CLI overrides remain useful.
    args.fusion_repo = build_args.fusion_repo
    args.fusion_checkpoint = build_args.fusion_checkpoint
    detector = build_yolo26_detector(build_args, device)
    image_fusion = (
        build_pixel_aligned_image_fusion(build_args, device)
        if checkpoint_mode == "hsvnet"
        else None
    )
    feature_fusion = (
        build_scalar_feature_fusion(detector, device)
        if checkpoint_mode == "scalar"
        else (build_gated_feature_fusion(detector, device) if checkpoint_mode == "gated" else None)
    )
    nir_encoder = (
        build_depth_nir_encoder(
            detector, device, quality_gated=checkpoint_mode == "depth_nir_gated"
        )
        if checkpoint_mode in ("depth_nir", "depth_nir_gated")
        else None
    )
    detector.model[-1].max_det = args.max_det
    saved_indices = tuple(checkpoint.get("feature_indices", ()))
    feature_indices, feature_channels = discover_dense_features(
        detector,
        image_size,
        device,
        include_p1=len(saved_indices) != 4,
    )
    if saved_indices and saved_indices != tuple(feature_indices):
        raise ValueError(
            "Checkpoint/template feature mismatch: "
            f"saved={saved_indices}, discovered={feature_indices}"
        )

    min_depth = float(checkpoint_args.get("min_depth", args.min_depth))
    max_depth = float(checkpoint_args.get("max_depth", args.max_depth))
    initial_depth = float(checkpoint_args.get("depth_init", 8.0))
    model = Yolo26DepthDetRoad(
        detector,
        feature_indices,
        feature_channels,
        min_depth=min_depth,
        max_depth=max_depth,
        initial_depth=initial_depth,
        image_fusion=image_fusion,
        feature_fusion=feature_fusion,
        nir_encoder=nir_encoder,
        input_mode=checkpoint_mode,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def decode_yolo26_detections(raw_output, args):
    """Convert native [x1,y1,x2,y2,confidence,class] output to test tuples."""
    prediction = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    if prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError(
            f"Expected YOLO26 end-to-end output shaped [B,N,6], got {tuple(prediction.shape)}"
        )

    decoded = []
    for sample in prediction:
        sample = sample[sample[:, 4] >= args.conf]
        sample = sample[: args.max_det]
        decoded.append(
            [
                (
                    int(row[5].item()),
                    *row[:4].detach().cpu().tolist(),
                    float(row[4].item()),
                )
                for row in sample
            ]
        )
    return decoded
