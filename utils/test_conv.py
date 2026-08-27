"""Compact GELEN decoder blocks used by DCCNet.

The original release carried a malformed, partially garbled copy of this
module: ``forward_split`` was indented outside ``GELEN`` and made importing
the model fail with an ``IndentationError``.  This file intentionally keeps
the public class names and layer names used by existing checkpoints.
"""

from typing import Tuple

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Return same-shape padding for an odd convolution kernel.

    ``d`` is accepted for compatibility with common YOLO-style helpers.  The
    decoder only uses dilation one, so retaining the historical calculation
    preserves its parameter shapes and behaviour.
    """

    del d
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv-BatchNorm-SiLU block."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two-convolution bottleneck with an optional residual shortcut."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RepConv(nn.Module):
    """Training-time compact RepConv used by the released decoder."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act=act)

    def forward(self, x):
        return self.conv(x)


class RepBottleneck(Bottleneck):
    """Bottleneck variant whose first convolution is a RepConv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k[0], 1)


class C3(nn.Module):
    """CSP bottleneck block with three outer convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepCSP(C3):
    """RepCSP block used in the GELEN branches."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class GELEN(nn.Module):
    """Generalized ELAN decoder block used by the released DCCNet model."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Equivalent forward path using ``split`` instead of ``chunk``."""

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


# Historical import used this name even though the released implementation is
# GELEN.  Keep it as an alias, so state-dict key paths are unchanged.
RepNCSPELAN4 = GELEN
