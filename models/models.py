"""Core DCCNet model definitions.

``DCCNet`` and ``AIFE`` are the paper-facing names.  The released GPD names
remain aliases so existing launch scripts and checkpoints retain their module
paths and principal state-dict keys.
"""

import functools
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from utils.attention_module import AGSA
from utils.test_conv import GELEN


def resize_to_multiple(x, multiple=14):
    """Resize a tensor to the next multiple used by a patch-based backbone."""

    height, width = x.shape[-2:]
    new_height = ((height + multiple - 1) // multiple) * multiple
    new_width = ((width + multiple - 1) // multiple) * multiple
    if (new_height, new_width) == (height, width):
        return x, height, width
    return F.interpolate(x, size=(new_height, new_width), mode="bilinear", align_corners=False), height, width


def normalize_depth_map(depth_map):
    """Normalize each predicted depth map independently to [0, 1]."""

    d_min = depth_map.amin(dim=(2, 3), keepdim=True)
    d_max = depth_map.amax(dim=(2, 3), keepdim=True)
    return (depth_map - d_min) / (d_max - d_min + 1e-8)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        bn=False,
        relu=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DepthHead(nn.Module):
    """Map the three-scale scalar depth prior into learnable geometric features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2, kernel_size=3, padding=1, bn=True),
            ConvBlock(out_channels // 2, out_channels, kernel_size=3, padding=1, bn=True),
        )

    def forward(self, x):
        return self.conv(x)


class FixedUpsample(nn.Module):
    """Deterministic upsampling retained from the released implementation."""

    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor <= 1 or scale_factor % 2:
            raise ValueError("scale_factor must be an even integer greater than one")
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1
        self.weight = nn.Parameter(
            torch.empty((1, 1, kernel_size, kernel_size), dtype=torch.float32)
            .expand(channel, -1, -1, -1)
            .clone()
        )
        self.conv = functools.partial(
            F.conv2d,
            weight=self.weight,
            bias=None,
            padding=scale_factor // 2,
            groups=channel,
        )
        with torch.no_grad():
            self.weight.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, x):
        if x is None:
            return None
        return self.conv(F.interpolate(x, scale_factor=self.scale_factor, mode="nearest"))


class Upsample(nn.Module):
    def __init__(self, channel: int, scale_factor: int, deterministic=True):
        super().__init__()
        self.deterministic = deterministic
        if deterministic:
            self.upsample = FixedUpsample(channel, scale_factor)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.upsample(x)


def upsample(x, scale_factor=2, mode="bilinear"):
    if mode == "nearest":
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)


class _VGGGELANBackbone(nn.Module):
    """Shared VGG16-BN encoder and GELEN decoder construction."""

    def _init_vgg_gelen(self, pretrained=True, deterministic=True, vgg_features=None):
        if vgg_features is None:
            weights = tv_models.VGG16_BN_Weights.DEFAULT if pretrained else None
            vgg_features = tv_models.vgg16_bn(weights=weights).features
        layers = list(vgg_features.children())
        if len(layers) < 43:
            raise ValueError("vgg_features must expose at least the 43 VGG16-BN feature layers")

        # Keep these attribute names: they are part of existing checkpoints.
        self.enc1 = nn.Sequential(*layers[:23])
        self.enc2 = nn.Sequential(*layers[23:33])
        self.enc3 = nn.Sequential(*layers[33:43])
        self.dec3 = nn.Sequential(GELEN(c1=512, c2=512, c3=1024, c4=256, n=1))
        self.dec2 = nn.Sequential(GELEN(c1=1024, c2=256, c3=512, c4=128, n=1))
        self.dec1 = nn.Sequential(GELEN(c1=512, c2=128, c3=256, c4=64, n=1))
        self.upsample1 = Upsample(512, 2, deterministic)
        self.upsample2 = Upsample(256, 2, deterministic)
        self.upsample3 = Upsample(256, 2, deterministic)
        self.upsample4 = Upsample(512, 4, deterministic)
        self.upsample_d = Upsample(1, 4, deterministic)

    def forward_fe(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.dec3(x3)
        y3 = x
        x = self.upsample1(x)
        x = torch.cat((x, x2), dim=1)
        x = self.dec2(x)
        y2 = x
        x = self.upsample2(x)
        x = torch.cat((x, x1), dim=1)
        y1 = self.dec1(x)

        y2 = self.upsample3(y2)
        y3 = self.upsample4(y3)
        return torch.cat((y1, y2, y3), dim=1), x3


class GPDBase(_VGGGELANBackbone):
    """Legacy RGB-only baseline retained for prior experiment scripts."""

    def __init__(self, pretrained=True, den_dropout=0.5, deterministic=True, vgg_features=None):
        super().__init__()
        self.den_dropout = den_dropout
        self._init_vgg_gelen(pretrained, deterministic, vgg_features)
        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, 256, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout),
        )
        self.den_head = nn.Sequential(ConvBlock(256, 1, kernel_size=1, padding=0))

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        return self.upsample_d(self.den_head(self.den_dec(y_cat)))


class GPDMem(GPDBase):
    """Legacy dictionary-memory baseline."""

    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        den_dropout=0.5,
        deterministic=True,
        vgg_features=None,
    ):
        super().__init__(pretrained, den_dropout, deterministic, vgg_features)
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.mem = nn.Parameter(torch.empty(1, mem_dim, mem_size).normal_(0.0, 1.0))
        self.den_dec = nn.Sequential(
            ConvBlock(512 + 256 + 128, mem_dim, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout),
        )
        self.den_head = nn.Sequential(ConvBlock(mem_dim, 1, kernel_size=1, padding=0))

    def forward_mem(self, y_den, m_mask=None):
        batch_size, channels, height, width = y_den.shape
        memory = self.mem.repeat(batch_size, 1, 1)
        memory_key = memory.transpose(1, 2)
        if m_mask is not None:
            y_den = F.dropout2d(y_den * m_mask, self.den_dropout, training=self.training)
        tokens = y_den.reshape(batch_size, channels, -1)
        logits = torch.bmm(memory_key, tokens) / sqrt(channels)
        reconstructed = torch.bmm(memory_key.transpose(1, 2), F.softmax(logits, dim=1))
        return reconstructed.reshape(batch_size, channels, height, width), logits

    def forward(self, x):
        y_cat, _ = self.forward_fe(x)
        y_den, _ = self.forward_mem(self.den_dec(y_cat))
        return self.upsample_d(self.den_head(y_den))


class GPDMemAdd(GPDMem):
    """Legacy dual-view memory baseline."""

    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        den_dropout=0.5,
        err_thrs=0.5,
        deterministic=True,
        vgg_features=None,
    ):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, deterministic, vgg_features)
        self.err_thrs = err_thrs
        self.den_dec = nn.Sequential(ConvBlock(512 + 256 + 128, mem_dim, kernel_size=1, padding=0, bn=True))

    @staticmethod
    def jsd(logits1, logits2):
        return F.mse_loss(F.softmax(logits1, dim=1), F.softmax(logits2, dim=1))

    def forward_train(self, img1, img2):
        y_cat1, _ = self.forward_fe(img1)
        y_cat2, _ = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)
        normalized1 = F.instance_norm(y_den1, eps=1e-5)
        normalized2 = F.instance_norm(y_den2, eps=1e-5)
        mask = (torch.abs(normalized1 - normalized2) < self.err_thrs).detach()
        rec1, logits1 = self.forward_mem(y_den1, mask)
        rec2, logits2 = self.forward_mem(y_den2, mask)
        return self.upsample_d(self.den_head(rec1)), self.upsample_d(self.den_head(rec2)), self.jsd(logits1, logits2)


class _ClassifierMixin(object):
    def _init_classifier(self, cls_dropout, cls_thrs):
        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs
        self.cls_head = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=cls_dropout),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def transform_cls_map_gt(c_gt):
        return upsample(c_gt, scale_factor=4, mode="nearest")

    def transform_cls_map_pred(self, c):
        mask = (c >= self.cls_thrs).to(dtype=c.dtype)
        return upsample(mask, scale_factor=4, mode="nearest")

    def transform_cls_map(self, c, c_gt=None):
        return self.transform_cls_map_gt(c_gt) if c_gt is not None else self.transform_cls_map_pred(c)


class GPDCls(GPDBase, _ClassifierMixin):
    """Legacy RGB classifier/density baseline."""

    def __init__(
        self,
        pretrained=True,
        den_dropout=0.5,
        cls_dropout=0.3,
        cls_thrs=0.5,
        deterministic=True,
        vgg_features=None,
    ):
        super().__init__(pretrained, den_dropout, deterministic, vgg_features)
        self._init_classifier(cls_dropout, cls_thrs)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)
        c = self.cls_head(x3)
        d = self.den_head(self.den_dec(y_cat))
        return self.upsample_d(d * self.transform_cls_map(c, c_gt)), c


class GPDMemCls(GPDMem, _ClassifierMixin):
    """Legacy memory/classifier baseline."""

    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        den_dropout=0.5,
        cls_dropout=0.3,
        cls_thrs=0.5,
        deterministic=True,
        vgg_features=None,
    ):
        super().__init__(pretrained, mem_size, mem_dim, den_dropout, deterministic, vgg_features)
        self._init_classifier(cls_dropout, cls_thrs)

    def forward(self, x, c_gt=None):
        y_cat, x3 = self.forward_fe(x)
        y_den, _ = self.forward_mem(self.den_dec(y_cat))
        c = self.cls_head(x3)
        return self.upsample_d(self.den_head(y_den) * self.transform_cls_map(c, c_gt)), c


class AIFE(nn.Module):
    """Appearance-Invariant Feature Enhancement (paper Eq. 9--11).

    ``mem`` is intentionally retained as the prototype parameter name used by
    released GPD checkpoints.  The three identity-initialized projections are
    the explicit ``W_Q``, ``W_K`` and ``W_V`` terms from Eq. (10).  When an
    older state dict lacks them, loading supplies their identity defaults.
    """

    def __init__(self, mem_size=1024, mem_dim=256, den_dropout=0.5):
        super().__init__()
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.den_dropout = den_dropout  # preserved constructor/state semantics
        self.mem = nn.Parameter(torch.empty(1, mem_dim, mem_size).normal_(0.0, 1.0))
        self.query_proj = nn.Linear(mem_dim, mem_dim, bias=False)
        self.key_proj = nn.Linear(mem_dim, mem_dim, bias=False)
        self.value_proj = nn.Linear(mem_dim, mem_dim, bias=False)
        self._init_identity_projections()

    def _init_identity_projections(self):
        with torch.no_grad():
            nn.init.eye_(self.query_proj.weight)
            nn.init.eye_(self.key_proj.weight)
            nn.init.eye_(self.value_proj.weight)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # A pre-AIFE checkpoint has ``memory.mem`` but no projection matrices.
        # Populate identity matrices so strict loading remains useful.
        for name in ("query_proj", "key_proj", "value_proj"):
            key = prefix + name + ".weight"
            if key not in state_dict:
                state_dict[key] = getattr(self, name).weight.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, features, m_mask=None):
        if features.ndim != 4:
            raise ValueError("AIFE expects [batch, channels, height, width] features")
        batch_size, channels, height, width = features.shape
        if channels != self.mem_dim:
            raise ValueError("AIFE received {} channels, expected {}".format(channels, self.mem_dim))

        if m_mask is None:
            masked_features = features
        else:
            if m_mask.shape != features.shape:
                if m_mask.ndim == 4 and m_mask.shape[1] == 1 and m_mask.shape[0] == batch_size:
                    m_mask = m_mask.expand_as(features)
                else:
                    raise ValueError("AIFE mask must match feature shape or have one channel")
            # The consistency mask is an eligibility signal, not a second
            # learned branch, so gradients do not flow through it.
            masked_features = features * m_mask.detach().to(dtype=features.dtype)

        tokens = masked_features.flatten(2).transpose(1, 2)  # [B, HW, C]
        prototypes = self.mem.squeeze(0).transpose(0, 1)  # [N_p, C]
        queries = self.query_proj(tokens)
        keys = self.key_proj(prototypes)
        values = self.value_proj(prototypes)
        attention_logits = torch.matmul(queries, keys.transpose(0, 1)) / sqrt(channels)
        attention = F.softmax(attention_logits, dim=-1)
        reconstructed = torch.matmul(attention, values).transpose(1, 2).reshape(batch_size, channels, height, width)

        # Legacy callers compute a consistency loss from logits with shape
        # [B, N_p, HW], so retain that return contract.
        return reconstructed, attention_logits.transpose(1, 2)


# Historical public name; it now implements the paper's AIFE module.
MemoryModule = AIFE


class DCCNet(_VGGGELANBackbone, _ClassifierMixin):
    """Decoupling Crowd Counting Network from the paper.

    The depth prior is frozen and kept in evaluation mode even while DCCNet is
    trained.  Project datasets normalize RGB images to [-1, 1]; before the
    prior, this class restores [0, 1] RGB and applies ImageNet mean/std.
    """

    def __init__(
        self,
        pretrained=True,
        mem_size=1024,
        mem_dim=256,
        cls_thrs=0.5,
        err_thrs=0.5,
        den_dropout=0.5,
        cls_dropout=0.3,
        has_err_loss=False,
        deterministic=True,
        depth_anything=None,
        vgg_features=None,
    ):
        super().__init__()
        self.den_dropout = den_dropout
        self.cls_dropout = cls_dropout
        self.cls_thrs = cls_thrs
        self.err_thrs = err_thrs
        self.has_err_loss = has_err_loss
        self.depth_anything = depth_anything
        self._init_vgg_gelen(pretrained, deterministic, vgg_features)

        # ``memory`` and ``cbam_attention`` retain released key prefixes.
        self.memory = AIFE(mem_size, mem_dim, den_dropout)
        self.cem_channels = 512 + 256 + 128
        self.depth_head = DepthHead(in_channels=3, out_channels=128)
        self.fuse_depth_den = ConvBlock(512 + 128, 512, kernel_size=1, padding=0, bn=True)
        self.cbam_attention = AGSA(
            in_planes=512,
            ratio=8,
            spatial_kernel_size=5,
            spatial_dilation=2,
            use_learnable_avg_pool=True,
            use_learnable_topk_max_pool=True,
            top_k_for_max_pool=5,
            use_spatial_attention=True,
            use_channel_attention=True,
        )
        self.den_dec = nn.Sequential(ConvBlock(self.cem_channels, mem_dim, kernel_size=1, padding=0, bn=True))
        self.den_head = nn.Sequential(ConvBlock(mem_dim, 1, kernel_size=1, padding=0))
        self._init_classifier(cls_dropout, cls_thrs)

        # Non-persistent buffers avoid adding noise to legacy checkpoints.
        self.register_buffer("_depth_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_depth_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self._freeze_depth_prior()

    @property
    def agsa(self):
        """Paper-facing AGSA name; state dict remains under cbam_attention.*."""

        return self.cbam_attention

    @property
    def aife(self):
        """Paper-facing AIFE name; state dict remains under memory.*."""

        return self.memory

    def _freeze_depth_prior(self):
        if self.depth_anything is None:
            return
        self.depth_anything.eval()
        for parameter in self.depth_anything.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # ``Module.train`` recursively flips every child module.  Restore the
        # DepthAnything invariant afterwards.
        self._freeze_depth_prior()
        return self

    def normalize_depth_input(self, image):
        """Convert project-normalized RGB [-1, 1] to DepthAnything input RGB."""

        rgb = ((image + 1.0) * 0.5).clamp_(0.0, 1.0)
        return (rgb - self._depth_mean.to(dtype=rgb.dtype)) / self._depth_std.to(dtype=rgb.dtype)

    @staticmethod
    def _coerce_depth_map(depth_output):
        if isinstance(depth_output, (tuple, list)):
            depth_output = depth_output[0]
        if not isinstance(depth_output, torch.Tensor):
            raise TypeError("DepthAnything must return a tensor or a tuple/list whose first item is a tensor")
        if depth_output.ndim == 3:
            return depth_output.unsqueeze(1)
        if depth_output.ndim == 4 and depth_output.shape[1] == 1:
            return depth_output
        raise ValueError("DepthAnything output must have shape [B,H,W] or [B,1,H,W]")

    def get_depth_feat(self, image, target_shape):
        """Extract frozen, normalized multi-scale geometric prior features."""

        if self.depth_anything is None:
            raise RuntimeError("get_depth_feat requires a DepthAnything prior")
        self._freeze_depth_prior()
        depth_input = self.normalize_depth_input(image)
        depth_input, original_h, original_w = resize_to_multiple(depth_input, 14)
        with torch.no_grad():
            depth_map = self._coerce_depth_map(self.depth_anything(depth_input))
        depth_map = depth_map[..., :original_h, :original_w]
        depth_map = normalize_depth_map(depth_map)

        depth_avg = F.avg_pool2d(depth_map, kernel_size=2, stride=2)
        depth_max = F.max_pool2d(depth_map, kernel_size=2, stride=2)
        depth_up = F.interpolate(depth_map, scale_factor=2, mode="bilinear", align_corners=False)
        depth_features = (depth_avg, depth_max, depth_up)
        depth_features = tuple(
            F.interpolate(feature, size=target_shape, mode="bilinear", align_corners=False) for feature in depth_features
        )
        return torch.cat(depth_features, dim=1)

    def _align_geometric_features(self, image, rgb_features):
        depth_features = self.depth_head(self.get_depth_feat(image, target_shape=rgb_features.shape[2:]))
        fused = self.fuse_depth_den(torch.cat((rgb_features, depth_features), dim=1))
        return self.agsa(fused)

    @staticmethod
    def jsd(logits1, logits2):
        """Legacy helper retained for callers of the former GPD API."""

        return F.mse_loss(F.softmax(logits1, dim=1), F.softmax(logits2, dim=1))

    @staticmethod
    def appearance_consistency_loss(reconstruction1, reconstruction2):
        """Paper Eq. (15): consistency between original/augmented AIFE outputs."""

        return F.mse_loss(reconstruction1, reconstruction2)

    def forward_train(self, img1, img2, c_gt=None):
        """Dual-view training forward pass used by the existing trainer."""

        y_cat1, x3_1 = self.forward_fe(img1)
        y_cat2, x3_2 = self.forward_fe(img2)
        y_den1 = self.den_dec(y_cat1)
        y_den2 = self.den_dec(y_cat2)

        norm1 = F.instance_norm(y_den1, eps=1e-5)
        norm2 = F.instance_norm(y_den2, eps=1e-5)
        mask = (torch.abs(norm1 - norm2) < self.err_thrs).detach()
        rec1, _ = self.memory(y_den1, mask)
        rec2, _ = self.memory(y_den2, mask)
        loss_con = self.appearance_consistency_loss(rec1, rec2)
        loss_err = F.l1_loss(norm1, norm2) if self.has_err_loss else y_den1.new_zeros(())

        if self.depth_anything is not None:
            x3_1 = self._align_geometric_features(img1, x3_1)
            x3_2 = self._align_geometric_features(img2, x3_2)
        c1 = self.cls_head(x3_1)
        c2 = self.cls_head(x3_2)

        c_resized1 = self.transform_cls_map_pred(c1)
        c_resized2 = self.transform_cls_map_pred(c2)
        c_err = torch.abs(c_resized1 - c_resized2)
        if c_gt is None:
            c_resized = torch.maximum(c_resized1, c_resized2)
        else:
            c_resized = torch.clamp(self.transform_cls_map_gt(c_gt) + c_err, 0, 1)

        d1 = self.den_head(rec1)
        d2 = self.den_head(rec2)
        return (
            self.upsample_d(d1 * c_resized),
            self.upsample_d(d2 * c_resized),
            c1,
            c2,
            upsample(c_err, scale_factor=4),
            loss_con,
            loss_err,
        )

    def forward(self, x, c_gt=None):
        """Single-stream inference forward pass returning density and aux map."""

        y_cat, x3 = self.forward_fe(x)
        reconstructed, _ = self.memory(self.den_dec(y_cat))
        if self.depth_anything is not None:
            x3 = self._align_geometric_features(x, x3)
        c = self.cls_head(x3)
        density = self.den_head(reconstructed) * self.transform_cls_map(c, c_gt)
        return self.upsample_d(density), c


# Backward-compatible class aliases.  These are identities rather than wrapper
# subclasses so existing type checks and serialized module paths stay simple.
GPD = DCCNet
DGModel_base = GPDBase
DGModel_mem = GPDMem
DGModel_memadd = GPDMemAdd
DGModel_cls = GPDCls
DGModel_memcls = GPDMemCls
DGModel_final = DCCNet


__all__ = (
    "AIFE",
    "AGSA",
    "ConvBlock",
    "DCCNet",
    "DGModel_base",
    "DGModel_cls",
    "DGModel_final",
    "DGModel_mem",
    "DGModel_memadd",
    "DGModel_memcls",
    "GPD",
    "GPDBase",
    "GPDCls",
    "GPDMem",
    "GPDMemAdd",
    "GPDMemCls",
    "MemoryModule",
)
