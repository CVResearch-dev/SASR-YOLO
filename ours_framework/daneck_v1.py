#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deformable Alignment Neck v1 (DANeck-v1)

- P2 neck output (layer18): deformable alignment
- P3 neck output (layer21): deformable alignment
- P4/P5 untouched
- Detect untouched
- classification/regression both consume aligned P2/P3 features
- zero-init final residual projection => exact identity at installation
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.ops import DeformConv2d


class DeformableAlignmentBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.channels = int(channels)

        # 3x3 deformable conv => 2 * 3 * 3 = 18 offsets.
        self.offset = nn.Conv2d(
            channels,
            18,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            channels
        )

        self.act = nn.SiLU(
            inplace=True
        )

        # Final residual projection.
        # Exact zero-init means:
        #
        #   output = x
        #
        # at installation.
        self.out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Start deformable sampling from regular grid.
        nn.init.zeros_(
            self.offset.weight
        )
        nn.init.zeros_(
            self.offset.bias
        )

        # Exact detector identity.
        nn.init.zeros_(
            self.out.weight
        )
        nn.init.zeros_(
            self.out.bias
        )

    def residual(self, x):
        offset = self.offset(x)

        y = self.deform(
            x,
            offset,
        )

        y = self.act(
            self.bn(y)
        )

        return self.out(y)

    def forward(self, x):
        return x + self.residual(x)


class DANeckWrapper(nn.Module):
    """
    Wrap one existing neck block.

        original:
            y = base(x)

        DANeck:
            y = base(x)
            y = align(y)

    Ultralytics graph metadata is copied from base.
    """

    def __init__(
        self,
        base: nn.Module,
        channels: int,
    ):
        super().__init__()

        self.base = base

        self.align = (
            DeformableAlignmentBlock(
                channels
            )
        )

        # Preserve Ultralytics graph metadata.
        for attr in (
            "i",
            "f",
            "type",
            "np",
        ):
            if hasattr(
                base,
                attr,
            ):
                setattr(
                    self,
                    attr,
                    getattr(
                        base,
                        attr,
                    ),
                )

    def forward(self, x):
        y = self.base(x)
        return self.align(y)


def _module_device_dtype(
    module: nn.Module,
):
    p = next(
        module.parameters()
    )

    return (
        p.device,
        p.dtype,
    )


def install_daneck_v1(
    model: nn.Module,
):
    """
    Install:
        layer18 = P2 neck output
        layer21 = P3 neck output
    """

    if getattr(
        model,
        "_daneck_v1_installed",
        False,
    ):
        raise RuntimeError(
            "DANeck-v1 already installed"
        )

    if not hasattr(
        model,
        "model",
    ):
        raise RuntimeError(
            "Ultralytics model.model not found"
        )

    layers = model.model

    if len(layers) <= 21:
        raise RuntimeError(
            "Unexpected YOLOv8s-P2 architecture"
        )

    p2_base = layers[18]
    p3_base = layers[21]

    if isinstance(
        p2_base,
        DANeckWrapper,
    ):
        raise RuntimeError(
            "P2 already wrapped"
        )

    if isinstance(
        p3_base,
        DANeckWrapper,
    ):
        raise RuntimeError(
            "P3 already wrapped"
        )

    p2 = DANeckWrapper(
        p2_base,
        channels=64,
    )

    p3 = DANeckWrapper(
        p3_base,
        channels=128,
    )

    dev2, dtype2 = (
        _module_device_dtype(
            p2_base
        )
    )

    dev3, dtype3 = (
        _module_device_dtype(
            p3_base
        )
    )

    p2 = p2.to(
        device=dev2,
        dtype=dtype2,
    )

    p3 = p3.to(
        device=dev3,
        dtype=dtype3,
    )

    layers[18] = p2
    layers[21] = p3

    model._daneck_v1_installed = True

    return p2, p3


try:
    torch.serialization.add_safe_globals(
        [
            DeformableAlignmentBlock,
            DANeckWrapper,
        ]
    )
except Exception:
    pass


__all__ = [
    "DeformableAlignmentBlock",
    "DANeckWrapper",
    "install_daneck_v1",
]
