"""Adaptive Geometric-Semantic Alignment (AGSA).

The implementation follows the two residual re-weighting operations in the
paper: ``F_enh = (1 + W_c) * F_cat`` (Eq. 5) and
``F_out = F_enh * (1 + W_s)`` (Eq. 8).  In particular, attention never
suppresses the identity path by multiplying a feature map only by a sigmoid
weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d

    HAS_DEFORM_CONV = True
except ImportError:  # pragma: no cover - exercised only without torchvision
    DeformConv2d = None
    HAS_DEFORM_CONV = False


class LearnableTopKMaxPool2d(nn.Module):
    """Top-K pooling used to summarize salient crowd activations (Eq. 2).

    The legacy checkpoint contains ``aggregation_weights``.  It is retained
    for compatibility, but plain Top-K averaging is the default because the
    paper specifies Top-K pooling rather than an additional learned pooling
    mechanism.
    """

    def __init__(self, k, in_channels=None, use_learnable_aggregation_weights=False):
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.in_channels = in_channels
        self.use_learnable_aggregation_weights = use_learnable_aggregation_weights
        # Keep the historical parameter name and shape so older AGSA state
        # dictionaries remain loadable.  It is used only when explicitly
        # requested by a caller.
        self.aggregation_weights = nn.Parameter(torch.ones(1, 1, self.k))

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        spatial_size = height * width
        actual_k = min(self.k, spatial_size)
        if actual_k == 0:
            return x.new_zeros(batch_size, num_channels, 1, 1)

        values = torch.topk(x.reshape(batch_size, num_channels, spatial_size), actual_k, dim=2).values
        if self.use_learnable_aggregation_weights:
            weights = F.softmax(self.aggregation_weights[:, :, :actual_k], dim=2)
            pooled = torch.sum(values * weights, dim=2, keepdim=True)
        else:
            pooled = values.mean(dim=2, keepdim=True)
        return pooled.view(batch_size, num_channels, 1, 1)


class LearnableWeightedAdaptiveAvgPool2d(nn.Module):
    """Learnable spatial pooling (L-Pool in Eq. 3)."""

    def __init__(self, in_channels, reduction_channels=None, use_bn=True):
        super().__init__()
        if reduction_channels is None:
            reduction_channels = max(1, in_channels // 8)

        layers = [nn.Conv2d(in_channels, reduction_channels, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(reduction_channels))
        layers.extend((nn.ReLU(inplace=True), nn.Conv2d(reduction_channels, 1, 1, bias=True)))
        self.weight_generator = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        logits = self.weight_generator(x)
        weights = F.softmax(logits.reshape(batch_size, 1, height * width), dim=2).reshape(
            batch_size, 1, height, width
        )
        return torch.sum(x * weights, dim=(2, 3), keepdim=True)


class ModifiedChannelAttention(nn.Module):
    """Learnable channel selection (LCA) with Eq. (5) residual scaling."""

    def __init__(
        self,
        in_planes,
        ratio=16,
        use_learnable_avg_pool=True,
        use_learnable_topk_max_pool=False,
        top_k_for_max_pool=3,
        use_learnable_topk_aggregation=False,
    ):
        super().__init__()
        self.use_learnable_avg_pool = use_learnable_avg_pool
        self.use_learnable_topk_max_pool = use_learnable_topk_max_pool

        if use_learnable_avg_pool:
            self.adaptive_avg_pool = LearnableWeightedAdaptiveAvgPool2d(in_planes)
        else:
            self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        if use_learnable_topk_max_pool:
            self.adaptive_max_pool = LearnableTopKMaxPool2d(
                k=top_k_for_max_pool,
                in_channels=in_planes,
                use_learnable_aggregation_weights=use_learnable_topk_aggregation,
            )
        else:
            self.adaptive_max_pool = nn.AdaptiveMaxPool2d(1)

        reduction_channels = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduction_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def attention_weights(self, x):
        """Return ``W_c`` without applying it, which aids inspection/tests."""

        return self.sigmoid(self.fc(self.adaptive_avg_pool(x)) + self.fc(self.adaptive_max_pool(x)))

    def forward(self, x):
        return x * (1.0 + self.attention_weights(x))


class DeformableSpatialAttention(nn.Module):
    """Deformable spatial attention (DSA) with Eq. (8) residual scaling."""

    def __init__(self, in_channels_for_offset_calc=2, kernel_size=7, dilation=1):
        super().__init__()
        if not HAS_DEFORM_CONV:
            raise ImportError("AGSA requires torchvision.ops.DeformConv2d; install a compatible torchvision build.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation // 2
        self.offset_conv = nn.Conv2d(
            in_channels_for_offset_calc,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # Start from a regular convolution and learn offsets during training.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        self.deform_conv = DeformConv2d(
            in_channels_for_offset_calc,
            1,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def pooled_context(x):
        return torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)

    def attention_weights(self, x):
        """Return ``W_s`` from the TopK/L-Pool-compatible spatial context."""

        pooled = self.pooled_context(x)
        offsets = self.offset_conv(pooled)
        return self.sigmoid(self.deform_conv(pooled, offsets))

    def forward(self, x):
        return x * (1.0 + self.attention_weights(x))


class AGSA(nn.Module):
    """Adaptive Geometric-Semantic Alignment module from DCCNet."""

    def __init__(
        self,
        in_planes,
        ratio=16,
        spatial_kernel_size=7,
        spatial_dilation=1,
        use_learnable_avg_pool=True,
        use_learnable_topk_max_pool=False,
        top_k_for_max_pool=3,
        use_spatial_attention=True,
        use_channel_attention=True,
        use_learnable_topk_aggregation=False,
    ):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention

        if use_channel_attention:
            self.ca = ModifiedChannelAttention(
                in_planes,
                ratio=ratio,
                use_learnable_avg_pool=use_learnable_avg_pool,
                use_learnable_topk_max_pool=use_learnable_topk_max_pool,
                top_k_for_max_pool=top_k_for_max_pool,
                use_learnable_topk_aggregation=use_learnable_topk_aggregation,
            )
        if use_spatial_attention:
            self.sa = DeformableSpatialAttention(
                in_channels_for_offset_calc=2,
                kernel_size=spatial_kernel_size,
                dilation=spatial_dilation,
            )

    def forward(self, x):
        if self.use_channel_attention:
            x = self.ca(x)
        if self.use_spatial_attention:
            x = self.sa(x)
        return x
