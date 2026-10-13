"""YOLO26 metric-depth/detection/road model and RGB-NIR fusion structures."""

from copy import deepcopy


import math


from pathlib import Path


import re

import sys


from .preprocess import normalize_image_size, resolve_image_size

import torch

from torch import nn

import torch.nn.functional as F


from ultralytics import YOLO, __version__ as ultralytics_version

from ultralytics.cfg import DEFAULT_CFG_DICT

from ultralytics.nn.tasks import DetectionModel

from ultralytics.utils import IterableSimpleNamespace


CLASS_NAMES = {0: "Golfcart", 1: "Person", 2: "Tree", 3: "Undef_obj"}

MIN_ULTRALYTICS_VERSION = (8, 4, 0)


def version_tuple(version_text):
    numbers = re.findall(r"\d+", version_text.split("+")[0])
    return tuple(int(value) for value in numbers[:3])


def validate_environment(args):
    if version_tuple(ultralytics_version) < MIN_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "YOLO26 requires ultralytics>=8.4.0, but this environment has "
            f"{ultralytics_version}. Use a separate YOLO26 environment."
        )
    image_size = resolve_image_size(args)
    if any(dimension % 32 for dimension in image_size):
        raise ValueError(f"Input height and width must be divisible by 32, got {image_size}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a smoke test")
    if not args.min_depth < args.depth_init < args.max_depth:
        raise ValueError("--depth-init must be strictly between --min-depth and --max-depth")
    if args.head_warmup_epochs < 0:
        raise ValueError("--head-warmup-epochs cannot be negative")
    for name in ("nir_dropout", "nir_mismatch"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.input_mode == "hsvnet":
        for label, path in (
            ("fusion repository", Path(args.fusion_repo)),
            ("fusion checkpoint", Path(args.fusion_checkpoint)),
        ):
            if str(path) not in ("", ".") and not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")


def select_device(device_text):
    if device_text == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{str(device_text).split(',')[0]}")


def build_pixel_aligned_image_fusion(args, device):
    """Load the pretrained HSV channel-attention fusion model from its source repo."""
    repository = str(Path(args.fusion_repo).expanduser().resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from net.image_fusion import HSVNet

    fusion = HSVNet()
    checkpoint = torch.load(
        Path(args.fusion_checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    fusion.load_state_dict(state_dict, strict=True)
    print(f"RGB-NIR image fusion: HSVNet ({args.fusion_checkpoint})")
    return fusion.to(device)


def make_detector_args(source_args, args):
    """Keep checkpoint-compatible detector configuration for inference."""
    if not isinstance(source_args, dict):
        source_args = vars(source_args)
    values = {**DEFAULT_CFG_DICT, **source_args}
    values.update(
        {
            "epochs": args.epochs,
            "imgsz": list(resolve_image_size(args)),
            "data": str(Path(args.dataset_root) / "data.yaml"),
        }
    )
    return IterableSimpleNamespace(**values)


def build_yolo26_detector(args, device):
    """Build a four-class YOLO26 graph and transfer compatible COCO weights."""
    pretrained = YOLO(args.model).model
    yaml = deepcopy(pretrained.yaml)
    if not yaml.get("end2end", False) or int(yaml.get("reg_max", 0)) != 1:
        raise ValueError(
            f"{args.model} is not a native YOLO26 detection model "
            "(expected end2end=True and reg_max=1)"
        )

    input_channels = 4 if args.input_mode == "rgbn" else 3
    detector = DetectionModel(yaml, ch=input_channels, nc=len(CLASS_NAMES), verbose=True)
    detector.names = CLASS_NAMES.copy()
    detector.args = make_detector_args(getattr(pretrained, "args", {}), args)
    detector.load(pretrained, verbose=True)
    if input_channels == 4:
        # Ultralytics transfers the three compatible RGB slices automatically.
        # Initialize the extra NIR slice from their mean instead of leaving it
        # random, preserving the scale of the pretrained first-layer response.
        first_weight = detector.model[0].conv.weight
        with torch.no_grad():
            first_weight[:, 3:4].copy_(first_weight[:, :3].mean(dim=1, keepdim=True))
        print("RGBN early fusion: first convolution initialized RGB + mean(RGB) NIR")
    detector = detector.to(device)

    head = detector.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("The constructed detector lost YOLO26 end-to-end mode")
    print(f"YOLO26 detector: classes={head.nc}, reg_max={head.reg_max}, end2end={head.end2end}")
    return detector


def build_gated_feature_fusion(detector, device):
    fusion = GatedRGBNStem(detector.model[0])
    print("RGB-NIR gated fusion: separate P1 stems, channel+spatial gate, initial gate=sigmoid(-4)")
    return fusion.to(device)


def build_scalar_feature_fusion(detector, device):
    fusion = ScalarRGBNStem(detector.model[0])
    print("RGB-NIR scalar fusion: separate P1 stems, fused=RGB+alpha*NIR, initial alpha=0")
    return fusion.to(device)


def build_depth_nir_encoder(detector, device, quality_gated=False):
    encoder = DepthNIRPyramid(detector.model[0])
    description = (
        "percentile-normalized NIR pyramid with learned quality/spatial gates"
        if quality_gated
        else "legacy NIR pyramid with scalar residual weights"
    )
    print(f"Depth-only NIR fusion: RGB-only detector/Road, {description}")
    return encoder.to(device)


def execute_graph(detector, image, capture_indices=(), injected_outputs=None):
    """Execute an Ultralytics graph while retaining requested intermediate maps."""
    saved_outputs = []
    captured = {}
    x = image
    capture_indices = set(capture_indices)
    injected_outputs = injected_outputs or {}

    for module in detector.model:
        if module.i in injected_outputs:
            x = injected_outputs[module.i]
        else:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = saved_outputs[module.f]
                else:
                    x = [x if index == -1 else saved_outputs[index] for index in module.f]
            x = module(x)
        saved_outputs.append(x if module.i in detector.save else None)
        if module.i in capture_indices:
            captured[module.i] = x
    return x, captured


def discover_dense_features(detector, image_size, device, include_p1=True):
    """Find P1/P2 plus the three feature maps feeding the Detect head."""
    detect_indices = [int(index) for index in detector.model[-1].f]
    if len(detect_indices) != 3:
        raise ValueError(f"Expected three detection scales, received {detect_indices}")

    # Keep every intermediate output only during this one discovery pass.
    outputs = []
    image_height, image_width = normalize_image_size(image_size)
    input_channels = int(detector.model[0].conv.in_channels)
    x = torch.zeros(1, input_channels, image_height, image_width, device=device)
    was_training = detector.training
    detector.eval()
    for module in detector.model:
        if module.f != -1:
            if isinstance(module.f, int):
                x = outputs[module.f]
            else:
                x = [x if index == -1 else outputs[index] for index in module.f]
        x = module(x)
        outputs.append(x)
    detector.train(was_training)

    p3 = outputs[detect_indices[0]]
    expected_p2_size = (p3.shape[-2] * 2, p3.shape[-1] * 2)
    p2_candidates = [
        index
        for index, output in enumerate(outputs[: detect_indices[0]])
        if isinstance(output, torch.Tensor) and tuple(output.shape[-2:]) == expected_p2_size
    ]
    if not p2_candidates:
        raise RuntimeError("Could not discover a P2/4 feature map in the YOLO26 graph")

    p2_index = p2_candidates[-1]
    if include_p1:
        p2 = outputs[p2_index]
        expected_p1_size = (p2.shape[-2] * 2, p2.shape[-1] * 2)
        p1_candidates = [
            index
            for index, output in enumerate(outputs[:p2_index])
            if isinstance(output, torch.Tensor) and tuple(output.shape[-2:]) == expected_p1_size
        ]
        if not p1_candidates:
            raise RuntimeError("Could not discover a P1/2 feature map in the YOLO26 graph")
        feature_indices = (p1_candidates[-1], p2_index, *detect_indices)
    else:
        # Legacy baseline checkpoints used P2-P5 only.
        feature_indices = (p2_index, *detect_indices)
    feature_channels = tuple(int(outputs[index].shape[1]) for index in feature_indices)
    feature_sizes = tuple(tuple(outputs[index].shape[-2:]) for index in feature_indices)
    print(
        f"Dense features: indices={feature_indices}, "
        f"channels={feature_channels}, sizes={feature_sizes}"
    )
    return feature_indices, feature_channels


def group_count(channels):
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(ConvBlock(channels, channels), ConvBlock(channels, channels))

    def forward(self, inputs):
        return inputs + self.blocks(inputs)


class GatedRGBNStem(nn.Module):
    """Fuse independent RGB/NIR P1 features with channel and spatial gates.

    The RGB feature is produced by the detector's pretrained first layer. This
    module owns only the NIR stem and gate branches. Its residual gate starts
    near zero, so the initial network remains close to the RGB-only detector.
    """

    def __init__(self, rgb_stem, initial_gate=-4.0):
        super().__init__()
        rgb_conv = rgb_stem.conv
        out_channels = int(rgb_conv.out_channels)
        self.nir_stem = deepcopy(rgb_stem)
        self.nir_stem.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.nir_stem.conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                self.nir_stem.conv.bias.copy_(rgb_conv.bias)

        hidden_channels = max(8, out_channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        self.spatial_gate = nn.Conv2d(out_channels * 2, 1, 3, padding=1)
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_gate)))

        # All learned gate residuals start at zero; sigmoid(-4) ~= 0.018.
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(self, rgb_feature, nir):
        nir_feature = self.nir_stem(nir)
        if nir_feature.shape != rgb_feature.shape:
            raise RuntimeError(
                f"RGB/NIR stem mismatch: {tuple(rgb_feature.shape)} vs {tuple(nir_feature.shape)}"
            )
        combined = torch.cat([rgb_feature, nir_feature], dim=1)
        gate = torch.sigmoid(
            self.channel_gate(combined) + self.spatial_gate(combined) + self.gate_bias
        )
        return rgb_feature + gate * nir_feature, gate


class ScalarRGBNStem(nn.Module):
    """Lightweight dual stem with one learnable NIR residual coefficient."""

    def __init__(self, rgb_stem):
        super().__init__()
        rgb_conv = rgb_stem.conv
        out_channels = int(rgb_conv.out_channels)
        self.nir_stem = deepcopy(rgb_stem)
        self.nir_stem.conv = nn.Conv2d(
            1,
            out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.nir_stem.conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                self.nir_stem.conv.bias.copy_(rgb_conv.bias)

        # Exactly RGB-only at initialization. Alpha receives a gradient on the
        # first step; the NIR stem starts learning once alpha moves away from 0.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb_feature, nir):
        nir_feature = self.nir_stem(nir)
        if nir_feature.shape != rgb_feature.shape:
            raise RuntimeError(
                f"RGB/NIR stem mismatch: {tuple(rgb_feature.shape)} vs {tuple(nir_feature.shape)}"
            )
        return rgb_feature + self.alpha * nir_feature, self.alpha


class DepthNIRPyramid(nn.Module):
    """Shallow NIR encoder producing P1/2 and P2/4 depth features."""

    def __init__(self, rgb_stem):
        super().__init__()
        rgb_conv = rgb_stem.conv
        p1_channels = int(rgb_conv.out_channels)
        self.p1 = deepcopy(rgb_stem)
        self.p1.conv = nn.Conv2d(
            1,
            p1_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=rgb_conv.bias is not None,
            padding_mode=rgb_conv.padding_mode,
        ).to(device=rgb_conv.weight.device, dtype=rgb_conv.weight.dtype)
        with torch.no_grad():
            self.p1.conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
            if rgb_conv.bias is not None:
                self.p1.conv.bias.copy_(rgb_conv.bias)

        self.p2 = nn.Sequential(
            nn.Conv2d(p1_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(64), 64),
            nn.SiLU(inplace=True),
        )

    def forward(self, nir):
        p1 = self.p1(nir)
        return p1, self.p2(p1)


class LegacyDenseDepthRoadHead26(nn.Module):
    """Original P2-P5 decoder retained for baseline checkpoint evaluation."""

    def __init__(self, feature_channels, min_depth):
        super().__init__()
        p2_channels, p3_channels, p4_channels, p5_channels = feature_channels
        self.min_depth = float(min_depth)

        self.p2 = ConvBlock(p2_channels, 48)
        self.p3 = ConvBlock(p3_channels, 64)
        self.p4 = ConvBlock(p4_channels, 64)
        self.p5 = ConvBlock(p5_channels, 64)
        self.context = ConvBlock(192, 96)
        self.refine = nn.Sequential(ConvBlock(144, 96), ConvBlock(96, 64))
        self.depth_output = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, 1, 1))
        self.road_output = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, features, output_size):
        p2, p3, p4, p5 = features
        p3_size = p3.shape[-2:]
        context = self.context(
            torch.cat(
                [
                    self.p3(p3),
                    F.interpolate(self.p4(p4), p3_size, mode="bilinear", align_corners=False),
                    F.interpolate(self.p5(p5), p3_size, mode="bilinear", align_corners=False),
                ],
                dim=1,
            )
        )
        context = F.interpolate(context, p2.shape[-2:], mode="bilinear", align_corners=False)
        dense = self.refine(torch.cat([self.p2(p2), context], dim=1))

        depth = F.interpolate(
            self.depth_output(dense), output_size, mode="bilinear", align_corners=False
        )[:, 0]
        road = F.interpolate(
            self.road_output(dense), output_size, mode="bilinear", align_corners=False
        )[:, 0]
        return F.softplus(depth) + self.min_depth, road


class DenseDepthRoadHead26(nn.Module):
    """P1-P5 top-down decoder with independent depth and Road refinement."""

    def __init__(
        self,
        feature_channels,
        min_depth,
        max_depth,
        initial_depth,
        use_depth_nir=False,
        quality_gated_nir=False,
    ):
        super().__init__()
        p1_channels, p2_channels, p3_channels, p4_channels, p5_channels = feature_channels
        self.register_buffer("log_min_depth", torch.tensor(math.log(float(min_depth))))
        self.register_buffer(
            "log_depth_range",
            torch.tensor(math.log(float(max_depth)) - math.log(float(min_depth))),
        )

        self.p1 = ConvBlock(p1_channels, 32)
        self.p2 = ConvBlock(p2_channels, 64)
        self.p3 = ConvBlock(p3_channels, 96)
        self.p4 = ConvBlock(p4_channels, 96)
        self.p5 = ConvBlock(p5_channels, 96)
        self.fuse_p4 = nn.Sequential(ConvBlock(192, 96), ResidualBlock(96))
        self.fuse_p3 = nn.Sequential(ConvBlock(192, 96), ResidualBlock(96))
        self.fuse_p2 = nn.Sequential(ConvBlock(160, 64), ResidualBlock(64))

        # Multi-scale context is shared, but task-specific spatial refinement is not.
        self.depth_p2 = nn.Sequential(ResidualBlock(64), ConvBlock(64, 64))
        self.road_p2 = nn.Sequential(ResidualBlock(64), ConvBlock(64, 64))
        self.depth_p1 = nn.Sequential(ConvBlock(96, 64), ResidualBlock(64), ConvBlock(64, 32))
        self.road_p1 = nn.Sequential(ConvBlock(96, 64), ResidualBlock(64), ConvBlock(64, 32))
        self.depth_output = nn.Conv2d(32, 1, 1)
        self.road_output = nn.Conv2d(32, 1, 1)
        self.use_depth_nir = bool(use_depth_nir)
        self.quality_gated_nir = bool(quality_gated_nir)
        if self.use_depth_nir:
            # P1 and P2 residual weights start at zero, exactly reproducing the
            # RGB depth path before learning whether NIR is useful.
            self.depth_nir_alpha = nn.Parameter(torch.zeros(2))
        if self.quality_gated_nir:
            quality_hidden = 16
            self.depth_nir_quality_gate = nn.Sequential(
                nn.Linear(8, quality_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(quality_hidden, 2),
            )
            self.depth_nir_spatial_gate_p1 = nn.Conv2d(64, 1, 3, padding=1)
            self.depth_nir_spatial_gate_p2 = nn.Conv2d(128, 1, 3, padding=1)
            self.depth_nir_gate_bias = nn.Parameter(torch.full((2,), -1.0))
            # Alpha starts at zero, so the new model is exactly RGB-only before
            # training. Gates become trainable as soon as alpha leaves zero.
            nn.init.zeros_(self.depth_nir_quality_gate[-1].weight)
            nn.init.zeros_(self.depth_nir_quality_gate[-1].bias)
            nn.init.zeros_(self.depth_nir_spatial_gate_p1.weight)
            nn.init.zeros_(self.depth_nir_spatial_gate_p1.bias)
            nn.init.zeros_(self.depth_nir_spatial_gate_p2.weight)
            nn.init.zeros_(self.depth_nir_spatial_gate_p2.bias)
        self.last_depth_nir_gate_mean = None

        normalized = (math.log(float(initial_depth)) - math.log(float(min_depth))) / (
            math.log(float(max_depth)) - math.log(float(min_depth))
        )
        initial_bias = math.log(normalized / (1.0 - normalized))
        nn.init.normal_(self.depth_output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.depth_output.bias, initial_bias)

    @staticmethod
    def upsample(inputs, reference):
        return F.interpolate(inputs, reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features, output_size, nir_features=None, modality_quality=None):
        p1, p2, p3, p4, p5 = features
        top_p5 = self.p5(p5)
        top_p4 = self.fuse_p4(torch.cat([self.p4(p4), self.upsample(top_p5, p4)], dim=1))
        top_p3 = self.fuse_p3(torch.cat([self.p3(p3), self.upsample(top_p4, p3)], dim=1))
        top_p2 = self.fuse_p2(torch.cat([self.p2(p2), self.upsample(top_p3, p2)], dim=1))

        rgb_p1 = self.p1(p1)
        depth_top_p2 = top_p2
        depth_rgb_p1 = rgb_p1
        if self.use_depth_nir:
            if nir_features is None:
                raise ValueError("Depth-only NIR head requires P1/P2 NIR features")
            nir_p1, nir_p2 = nir_features
            if nir_p1.shape != rgb_p1.shape or nir_p2.shape != top_p2.shape:
                raise RuntimeError(
                    "Depth NIR feature mismatch: "
                    f"nir_p1={tuple(nir_p1.shape)} rgb_p1={tuple(rgb_p1.shape)} "
                    f"nir_p2={tuple(nir_p2.shape)} rgb_p2={tuple(top_p2.shape)}"
                )
            if self.quality_gated_nir:
                if modality_quality is None or modality_quality.shape[-1] != 8:
                    raise ValueError("Quality-gated NIR fusion requires [B,8] modality statistics")
                quality_logits = self.depth_nir_quality_gate(modality_quality)
                gate_p1 = torch.sigmoid(
                    self.depth_nir_spatial_gate_p1(torch.cat([rgb_p1, nir_p1], dim=1))
                    + quality_logits[:, 0, None, None, None]
                    + self.depth_nir_gate_bias[0]
                )
                gate_p2 = torch.sigmoid(
                    self.depth_nir_spatial_gate_p2(torch.cat([top_p2, nir_p2], dim=1))
                    + quality_logits[:, 1, None, None, None]
                    + self.depth_nir_gate_bias[1]
                )
                depth_rgb_p1 = rgb_p1 + self.depth_nir_alpha[0] * gate_p1 * nir_p1
                depth_top_p2 = top_p2 + self.depth_nir_alpha[1] * gate_p2 * nir_p2
                self.last_depth_nir_gate_mean = torch.stack(
                    [
                        gate_p1.mean(dim=(1, 2, 3)),
                        gate_p2.mean(dim=(1, 2, 3)),
                    ],
                    dim=1,
                ).detach()
            else:
                depth_rgb_p1 = rgb_p1 + self.depth_nir_alpha[0] * nir_p1
                depth_top_p2 = top_p2 + self.depth_nir_alpha[1] * nir_p2

        depth_p1 = self.depth_p1(
            torch.cat(
                [depth_rgb_p1, self.upsample(self.depth_p2(depth_top_p2), p1)],
                dim=1,
            )
        )
        road_p1 = self.road_p1(torch.cat([rgb_p1, self.upsample(self.road_p2(top_p2), p1)], dim=1))
        depth_logit = F.interpolate(
            self.depth_output(depth_p1),
            output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        road = F.interpolate(
            self.road_output(road_p1),
            output_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        log_depth = self.log_min_depth + depth_logit.sigmoid() * self.log_depth_range
        return log_depth.exp(), road


class Yolo26DepthDetRoad(nn.Module):
    def __init__(
        self,
        detector,
        feature_indices,
        feature_channels,
        min_depth,
        max_depth=35.0,
        initial_depth=8.0,
        image_fusion=None,
        feature_fusion=None,
        nir_encoder=None,
        input_mode="rgbn",
    ):
        super().__init__()
        self.detector = detector
        self.feature_indices = tuple(feature_indices)
        self.input_mode = str(input_mode)
        if self.input_mode not in (
            "depth_nir_gated",
            "depth_nir",
            "scalar",
            "gated",
            "rgbn",
            "hsvnet",
        ):
            raise ValueError(f"Unsupported input mode: {self.input_mode}")
        if self.input_mode == "hsvnet" and image_fusion is None:
            raise ValueError("HSVNet input mode requires an image-fusion model")
        if self.input_mode in ("scalar", "gated") and feature_fusion is None:
            raise ValueError(f"{self.input_mode} input mode requires a P1 feature-fusion module")
        if self.input_mode in ("depth_nir", "depth_nir_gated") and nir_encoder is None:
            raise ValueError("Depth-only NIR modes require a shallow NIR encoder")
        self.image_fusion = image_fusion
        self.feature_fusion = feature_fusion
        self.nir_encoder = nir_encoder
        self.last_gate_mean = None
        self.last_depth_nir_alpha = None
        self.last_depth_nir_gate_mean = None
        if len(feature_indices) == 4:
            if self.input_mode in ("depth_nir", "depth_nir_gated"):
                raise ValueError("Depth-only NIR modes require the revised P1-P5 decoder")
            self.architecture_version = "legacy_p2_p5"
            self.dense_head = LegacyDenseDepthRoadHead26(feature_channels, min_depth)
        elif len(feature_indices) == 5:
            self.architecture_version = (
                "v3_p1_p5_depth_nir_quality_gated"
                if self.input_mode == "depth_nir_gated"
                else "v2_p1_p5_depth_only_nir"
                if self.input_mode == "depth_nir"
                else "v1_p1_p5_separate"
            )
            self.dense_head = DenseDepthRoadHead26(
                feature_channels,
                min_depth,
                max_depth,
                initial_depth,
                use_depth_nir=self.input_mode in ("depth_nir", "depth_nir_gated"),
                quality_gated_nir=self.input_mode == "depth_nir_gated",
            )
        else:
            raise ValueError(f"Expected four or five dense features, got {feature_indices}")

    def fuse_inputs(self, image, nir=None):
        if self.input_mode == "rgbn" and nir is None and image.shape[1] == 4:
            # Ultralytics profiling utilities infer four input channels from the
            # detector's first convolution and invoke the wrapper directly.
            return image
        if nir is None:
            # Profilers in Ultralytics invoke the model with a single tensor.
            # A neutral zero-NIR image keeps that interface usable.
            nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
        if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"Expected NIR [B,1,H,W] aligned with RGB, got {tuple(nir.shape)} "
                f"for RGB {tuple(image.shape)}"
            )

        if self.input_mode == "rgbn":
            return torch.cat([image, nir], dim=1)

        # The original attention block uses batch statistics even in eval mode
        # and cannot process B=1 after global pooling. Duplicate only that edge
        # case, then discard the duplicate result.
        single = image.shape[0] == 1
        fusion_rgb = image.repeat(2, 1, 1, 1) if single else image
        fusion_nir = nir.repeat(2, 1, 1, 1) if single else nir
        fused = self.image_fusion(fusion_rgb * 255.0, fusion_nir * 255.0) / 255.0
        if single:
            fused = fused[:1]
        return fused.clamp(0.0, 1.0)

    @staticmethod
    def normalize_nir_percentile(nir):
        """Normalize useful NIR contrast while leaving flat/clipped frames stable."""
        reduced = F.interpolate(nir, size=(44, 80), mode="bilinear", align_corners=False)
        flat = reduced.flatten(2)
        low = torch.quantile(flat, 0.05, dim=2, keepdim=True).unsqueeze(-1)
        high = torch.quantile(flat, 0.95, dim=2, keepdim=True).unsqueeze(-1)
        span = high - low
        normalized = ((nir - low) / span.clamp_min(0.05)).clamp(0.0, 1.0)
        # Near-constant frames contain no robust percentile range. Keeping the
        # raw level lets the quality gate recognize zero or saturated inputs.
        return torch.where(span >= 0.05, normalized, nir)

    @staticmethod
    def modality_statistics(image, nir):
        luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
        nir_values = nir[:, 0]
        statistics = []
        for values in (luminance, nir_values):
            statistics.extend(
                (
                    values.mean(dim=(1, 2)),
                    values.std(dim=(1, 2), unbiased=False),
                    (values < 0.15).float().mean(dim=(1, 2)),
                    (values > 0.85).float().mean(dim=(1, 2)),
                )
            )
        return torch.stack(statistics, dim=1)

    def forward(self, image, nir=None):
        nir_features = None
        modality_quality = None
        if self.input_mode in ("depth_nir", "depth_nir_gated"):
            if nir is None:
                nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
            if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
                raise ValueError(f"Expected aligned NIR [B,1,H,W], got {tuple(nir.shape)}")
            detection, captured = execute_graph(
                self.detector, image, capture_indices=self.feature_indices
            )
            if self.input_mode == "depth_nir_gated":
                modality_quality = self.modality_statistics(image, nir)
                encoded_nir = self.normalize_nir_percentile(nir)
            else:
                encoded_nir = nir
            nir_features = self.nir_encoder(encoded_nir)
            self.last_depth_nir_alpha = self.dense_head.depth_nir_alpha.detach().clone()
            output_size = image.shape[-2:]
        elif self.input_mode in ("scalar", "gated"):
            if nir is None:
                nir = image.new_zeros((image.shape[0], 1, *image.shape[-2:]))
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(image.shape)}")
            if nir.ndim != 4 or nir.shape[1] != 1 or nir.shape[-2:] != image.shape[-2:]:
                raise ValueError(f"Expected aligned NIR [B,1,H,W], got {tuple(nir.shape)}")
            rgb_p1 = self.detector.model[0](image)
            fused_p1, gate = self.feature_fusion(rgb_p1, nir)
            self.last_gate_mean = gate.detach().mean()
            detection, captured = execute_graph(
                self.detector,
                image,
                capture_indices=self.feature_indices,
                injected_outputs={0: fused_p1},
            )
            output_size = image.shape[-2:]
        else:
            fused_image = self.fuse_inputs(image, nir)
            detection, captured = execute_graph(
                self.detector, fused_image, capture_indices=self.feature_indices
            )
            output_size = fused_image.shape[-2:]
        missing = set(self.feature_indices) - set(captured)
        if missing:
            raise RuntimeError(f"Missing YOLO26 dense features: {sorted(missing)}")
        features = tuple(captured[index] for index in self.feature_indices)
        if self.input_mode in ("depth_nir", "depth_nir_gated"):
            depth, road = self.dense_head(features, output_size, nir_features, modality_quality)
            self.last_depth_nir_gate_mean = self.dense_head.last_depth_nir_gate_mean
        else:
            depth, road = self.dense_head(features, output_size)
        return detection, depth, road
