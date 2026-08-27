"""Minimal DepthAnything/DPT implementation for DCCNet.

The DINOv2 encoder architecture is obtained from the official, pinned
facebookresearch repository.  ``pretrained=False`` is deliberate: this call
only obtains the architecture and never requests DINOv2 checkpoint weights.
On a clean machine, the first construction needs internet access for
``torch.hub`` to cache that source repository.  Pass ``backbone=...`` when a
fully local or test-time backbone is required.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything.blocks import FeatureFusionBlock, _make_scratch


# Pinning prevents a future upstream default branch from silently changing the
# geometry prior architecture used by a release/checkpoint.
DINOv2_HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
_VALID_ENCODERS = ("vits", "vitb", "vitl")


def _load_official_dinov2(encoder):
    """Load a DINOv2 architecture from the pinned official torch.hub source."""

    model_name = "dinov2_{:}14".format(encoder)
    kwargs = {"pretrained": False}
    try:
        # torch>=2.0 supports trust_repo and avoids an interactive trust prompt
        # in unattended release environments.
        return torch.hub.load(DINOv2_HUB_REPO, model_name, trust_repo=True, **kwargs)
    except TypeError:  # pragma: no cover - compatibility with old torch.hub
        return torch.hub.load(DINOv2_HUB_REPO, model_name, **kwargs)


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    """DPT refinement head that converts DINO tokens to one depth map."""

    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=(256, 512, 1024, 1024),
        use_clstoken=False,
    ):
        super().__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        )
        self.resize_layers = nn.ModuleList(
            (
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            )
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()) for _ in range(len(self.projects))
            )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

    def forward(self, out_features, patch_h, patch_w):
        projected = []
        for index, feature in enumerate(out_features):
            if self.use_clstoken:
                tokens, cls_token = feature[0], feature[1]
                cls_tokens = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, cls_tokens), dim=-1))
            else:
                # DINOv2 returns a one-element tuple when no class token is
                # requested.  Accept a raw tensor too for local test backbones.
                tokens = feature[0] if isinstance(feature, (tuple, list)) else feature

            tokens = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            projected.append(self.resize_layers[index](self.projects[index](tokens)))

        layer_1, layer_2, layer_3, layer_4 = projected
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.nclass > 1:
            return self.scratch.output_conv(path_1)
        output = self.scratch.output_conv1(path_1)
        output = F.interpolate(output, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(output)


class DPT_DINOv2(nn.Module):
    """DPT head on an official DINOv2 backbone, without bundled torchhub code."""

    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=(256, 512, 1024, 1024),
        use_bn=False,
        use_clstoken=False,
        localhub=None,
        backbone=None,
    ):
        super().__init__()
        if encoder not in _VALID_ENCODERS:
            raise ValueError("encoder must be one of {}".format(_VALID_ENCODERS))
        if localhub:
            warnings.warn(
                "localhub is ignored: DCCNet no longer vendors a torchhub checkout. "
                "Use backbone=... for an explicitly local backbone.",
                DeprecationWarning,
            )

        self.pretrained = backbone if backbone is not None else _load_official_dinov2(encoder)
        try:
            dim = self.pretrained.blocks[0].attn.qkv.in_features
        except (AttributeError, IndexError):
            dim = self.pretrained.embed_dim
        self.depth_head = DPTHead(
            1,
            dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    @staticmethod
    def _resize_to_patch_multiple(x, patch_size=14):
        height, width = x.shape[-2:]
        new_height = ((height + patch_size - 1) // patch_size) * patch_size
        new_width = ((width + patch_size - 1) // patch_size) * patch_size
        if (new_height, new_width) == (height, width):
            return x, height, width
        return F.interpolate(x, size=(new_height, new_width), mode="bilinear", align_corners=False), height, width

    def forward(self, x):
        resized, original_h, original_w = self._resize_to_patch_multiple(x)
        height, width = resized.shape[-2:]
        features = self.pretrained.get_intermediate_layers(resized, 4, return_class_token=True)
        depth = self.depth_head(features, height // 14, width // 14)
        depth = F.interpolate(depth, size=(original_h, original_w), mode="bilinear", align_corners=True)
        return F.relu(depth).squeeze(1)


class DepthAnything(DPT_DINOv2):
    """DepthAnything-compatible wrapper with optional, explicit local weights.

    Constructing this class builds only a DINOv2 *architecture* with
    ``pretrained=False``.  To use an actual DepthAnything checkpoint, pass a
    local ``checkpoint_path`` to :meth:`from_pretrained`; this avoids an
    implicit model-weight download in library code.
    """

    def __init__(self, config=None, **kwargs):
        options = dict(config or {})
        options.update(kwargs)
        super().__init__(**options)

    @classmethod
    def from_pretrained(cls, model_id=None, checkpoint_path=None, map_location="cpu", config=None, **kwargs):
        """Compatibility helper that loads explicit local DepthAnything weights.

        ``model_id`` is accepted to keep the old call signature recognizable,
        but it is not downloaded implicitly.  Supply ``checkpoint_path`` for
        reproducible releases instead.
        """

        if checkpoint_path is None:
            raise ValueError(
                "DepthAnything weights are not downloaded implicitly. Provide checkpoint_path="
                "... (model_id={!r} was not used).".format(model_id)
            )
        model = cls(config=config, **kwargs)
        checkpoint = torch.load(os.fspath(checkpoint_path), map_location=map_location)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        # Most public checkpoints store DPT keys directly; tolerate a DataParallel prefix.
        if isinstance(checkpoint, dict) and all(key.startswith("module.") for key in checkpoint):
            checkpoint = {key[7:]: value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
        return model

    def extract_feature(self, x):
        """Return the last DINO token map resized to the input resolution."""

        resized, original_h, original_w = self._resize_to_patch_multiple(x)
        height, width = resized.shape[-2:]
        features = self.pretrained.get_intermediate_layers(resized, 4, return_class_token=False)
        last_feature = features[-1]
        if isinstance(last_feature, (tuple, list)):
            last_feature = last_feature[0]
        patch_h, patch_w = height // 14, width // 14
        last_feature = last_feature.permute(0, 2, 1).reshape(
            last_feature.shape[0], last_feature.shape[-1], patch_h, patch_w
        )
        return F.interpolate(last_feature, size=(original_h, original_w), mode="bilinear", align_corners=False)
