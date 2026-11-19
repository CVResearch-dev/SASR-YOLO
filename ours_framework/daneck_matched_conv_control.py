#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ours_framework.cada_v2 import (
    CADAV2DetectionModel,
    CADAV2Trainer,
)


CADA_PARAMS = 10_639_576

P2_ADDED_PARAMS = 51_424
P3_ADDED_PARAMS = 184_736

EXPECTED_ADDED_PARAMS = (
    P2_ADDED_PARAMS
    + P3_ADDED_PARAMS
)

EXPECTED_TOTAL_PARAMS = (
    CADA_PARAMS
    + EXPECTED_ADDED_PARAMS
)


class MatchedConvBlock(nn.Module):
    """
    Capacity/FLOPs-matched ordinary-convolution control for DANeck.

    P2:
        C=64, hidden=80

    P3:
        C=128, hidden=144

    Structure:
        x
        -> 3x3 ordinary Conv
        -> BN
        -> SiLU
        -> 1x1 projection
        -> residual addition

    Final projection is zero initialized so the block is an
    exact identity mapping at installation, matching DANeck's
    residual initialization principle.
    """

    def __init__(
        self,
        channels: int,
        hidden: int,
    ):
        super().__init__()

        self.channels = int(channels)
        self.hidden = int(hidden)

        self.conv = nn.Conv2d(
            self.channels,
            self.hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            self.hidden
        )

        self.act = nn.SiLU(
            inplace=True
        )

        self.out = nn.Conv2d(
            self.hidden,
            self.channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Exact residual identity at installation.
        nn.init.zeros_(
            self.out.weight
        )
        nn.init.zeros_(
            self.out.bias
        )

    def residual(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return self.out(y)

    def forward(self, x):
        return x + self.residual(x)


class MatchedConvWrapper(nn.Module):
    """
    Wrap the same neck locations used by DANeck.
    """

    def __init__(
        self,
        base: nn.Module,
        channels: int,
        hidden: int,
    ):
        super().__init__()

        self.base = base

        self.control = MatchedConvBlock(
            channels=channels,
            hidden=hidden,
        )

        # Preserve Ultralytics graph metadata.
        for attr in (
            "i",
            "f",
            "type",
            "np",
        ):
            if hasattr(base, attr):
                setattr(
                    self,
                    attr,
                    getattr(base, attr),
                )

    def forward(self, x):
        y = self.base(x)
        return self.control(y)


def _device_dtype(module):
    p = next(
        module.parameters()
    )

    return p.device, p.dtype


def install_matched_conv_control(
    model: nn.Module,
):
    """
    Install matched ordinary-convolution blocks at exactly
    the same locations used by formal DANeck:

        node18 = P2
        node21 = P3
    """

    if getattr(
        model,
        "_matched_conv_control_installed",
        False,
    ):
        raise RuntimeError(
            "matched-conv control already installed"
        )

    layers = model.model

    if len(layers) <= 21:
        raise RuntimeError(
            "unexpected YOLOv8s-P2 graph"
        )

    p2_base = layers[18]
    p3_base = layers[21]

    p2 = MatchedConvWrapper(
        base=p2_base,
        channels=64,
        hidden=80,
    )

    p3 = MatchedConvWrapper(
        base=p3_base,
        channels=128,
        hidden=144,
    )

    dev2, dtype2 = _device_dtype(
        p2_base
    )

    dev3, dtype3 = _device_dtype(
        p3_base
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

    model._matched_conv_control_installed = True

    return p2, p3


class MatchedConvCADAV2Trainer(
    CADAV2Trainer
):
    """
    CADA-v2 + ordinary matched convolution at P2/P3.
    """

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADAV2DetectionModel:

        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        before = sum(
            p.numel()
            for p in model.parameters()
        )

        if before != CADA_PARAMS:
            raise RuntimeError(
                "CADA reference params mismatch: "
                f"{before:,} != {CADA_PARAMS:,}"
            )

        install_matched_conv_control(
            model
        )

        after = sum(
            p.numel()
            for p in model.parameters()
        )

        added = after - before

        if added != EXPECTED_ADDED_PARAMS:
            raise RuntimeError(
                "MatchedConv added params mismatch: "
                f"{added:,} != "
                f"{EXPECTED_ADDED_PARAMS:,}"
            )

        if after != EXPECTED_TOTAL_PARAMS:
            raise RuntimeError(
                "MatchedConv total params mismatch: "
                f"{after:,} != "
                f"{EXPECTED_TOTAL_PARAMS:,}"
            )

        if not isinstance(
            model.model[18],
            MatchedConvWrapper,
        ):
            raise RuntimeError(
                "P2 matched-conv missing"
            )

        if not isinstance(
            model.model[21],
            MatchedConvWrapper,
        ):
            raise RuntimeError(
                "P3 matched-conv missing"
            )

        print()
        print("=" * 100)
        print("DANECK MATCHED-CONV CONTROL")
        print("P2 hidden       = 80")
        print("P3 hidden       = 144")
        print(
            "added params    =",
            f"{added:,}",
        )
        print(
            "total params    =",
            f"{after:,}",
        )
        print(
            "DANeck target   = 236,516 added params"
        )
        print(
            "parameter diff  = 356"
        )
        print(
            "P2/P3 placement = PASS"
        )
        print("=" * 100)

        return model


try:
    torch.serialization.add_safe_globals(
        [
            MatchedConvBlock,
            MatchedConvWrapper,
            MatchedConvCADAV2Trainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "MatchedConvBlock",
    "MatchedConvWrapper",
    "MatchedConvCADAV2Trainer",
    "install_matched_conv_control",
]
