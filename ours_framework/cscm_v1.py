#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


EXPECTED_CSCM_V1_PARAMS = 12_480


class CSCMV1DetectWrapper(nn.Module):
    """
    CSCM-v1
    Cross-Scale Context Modulation

    Target:
        Detect input only.

    Inputs:
        P2 = node18, [B,  64,160,160]
        P3 = node21, [B, 128, 80, 80]
        P4 = node24, [B, 256, 40, 40]
        P5 = node27, [B, 512, 20, 20]

    Mechanism:
        P3
         -> 1x1 semantic projection
         -> upsample to P2 resolution
         -> zero-init gate projection
         -> gain = 1 + tanh(logits)
         -> P2' = P2 * gain

    The original Detect module is preserved intact.

    Initial state:
        gate logits = 0
        gain = 1
        P2' = P2

    Therefore the complete detector is exactly
    identical at initialization.
    """

    def __init__(
        self,
        base: Detect,
        p3_channels: int = 128,
        p2_channels: int = 64,
    ):
        super().__init__()

        if not isinstance(
            base,
            Detect,
        ):
            raise TypeError(
                "CSCM-v1 base must be Detect"
            )

        self.base = base

        self.p3_channels = int(
            p3_channels
        )

        self.p2_channels = int(
            p2_channels
        )

        # P3 semantic context projection.
        self.context_proj = Conv(
            self.p3_channels,
            self.p2_channels,
            k=1,
            s=1,
        )

        # Final modulation logits.
        # Zero-init gives gain exactly 1.
        self.gate_out = nn.Conv2d(
            self.p2_channels,
            self.p2_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        nn.init.zeros_(
            self.gate_out.weight
        )

        nn.init.zeros_(
            self.gate_out.bias
        )

        # Preserve Ultralytics graph metadata.
        self.i = getattr(
            base,
            "i",
            -1,
        )

        self.f = getattr(
            base,
            "f",
            -1,
        )

        self.type = (
            "CSCMV1DetectWrapper"
        )

        self.np = sum(
            p.numel()
            for p in self.parameters()
        )


    # --------------------------------------------------------
    # Detect interface required by detection loss.
    # --------------------------------------------------------

    @property
    def stride(self):
        return self.base.stride

    @property
    def nc(self):
        return self.base.nc

    @property
    def nl(self):
        return self.base.nl

    @property
    def reg_max(self):
        return self.base.reg_max

    @property
    def no(self):
        return self.base.no


    def _apply(
        self,
        fn,
        recurse=True,
    ):
        """
        DetectionModel._apply() normally special-cases
        a last-layer Detect instance.

        Since CSCM wraps Detect, explicitly propagate
        tensor transforms to Detect's runtime tensors.
        """

        super()._apply(
            fn,
            recurse=recurse,
        )

        for name in (
            "stride",
            "anchors",
            "strides",
        ):

            if not hasattr(
                self.base,
                name,
            ):
                continue

            value = getattr(
                self.base,
                name,
            )

            if torch.is_tensor(
                value
            ):
                setattr(
                    self.base,
                    name,
                    fn(value),
                )

        return self


    def modulation(
        self,
        p3: torch.Tensor,
        target_hw,
    ):
        """
        Returns:
            gain [B,64,H_P2,W_P2]

        Initial gain is exactly 1.
        """

        context = self.context_proj(
            p3
        )

        context = F.interpolate(
            context,
            size=target_hw,
            mode="nearest",
        )

        logits = self.gate_out(
            context
        )

        gain = (
            1.0
            + torch.tanh(
                logits
            )
        )

        return gain


    def modulated_p2(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ):
        gain = self.modulation(
            p3,
            p2.shape[-2:],
        )

        if gain.shape != p2.shape:
            raise RuntimeError(
                "CSCM gain/P2 mismatch: "
                f"{tuple(gain.shape)} "
                f"vs {tuple(p2.shape)}"
            )

        return (
            p2 * gain
        )


    def forward(
        self,
        x,
    ):
        if not isinstance(
            x,
            (list, tuple),
        ):
            raise TypeError(
                "CSCM Detect input must "
                "be list/tuple"
            )

        if len(x) != 4:
            raise RuntimeError(
                "CSCM expected 4 Detect "
                f"inputs, got {len(x)}"
            )

        p2, p3, p4, p5 = x

        if p2.shape[1] != 64:
            raise RuntimeError(
                "unexpected P2 channels: "
                f"{p2.shape[1]}"
            )

        if p3.shape[1] != 128:
            raise RuntimeError(
                "unexpected P3 channels: "
                f"{p3.shape[1]}"
            )

        p2_mod = self.modulated_p2(
            p2,
            p3,
        )

        detect_inputs = [
            p2_mod,
            p3,
            p4,
            p5,
        ]

        return self.base(
            detect_inputs
        )


def install_cscm_v1(
    model: nn.Module,
) -> nn.Module:

    if getattr(
        model,
        "_cscm_v1_installed",
        False,
    ):
        return model

    if not hasattr(
        model,
        "model",
    ):
        raise RuntimeError(
            "model has no .model graph"
        )

    if len(model.model) <= 28:
        raise RuntimeError(
            "model graph has no node28"
        )

    base = model.model[28]

    if isinstance(
        base,
        CSCMV1DetectWrapper,
    ):
        model._cscm_v1_installed = True
        return model

    if not isinstance(
        base,
        Detect,
    ):
        raise RuntimeError(
            "node28 is not Detect: "
            f"{type(base)}"
        )

    # YOLOv8s-P2 Detect inputs must remain:
    # node18 P2, node21 P3, node24 P4, node27 P5.
    expected_f = [
        18,
        21,
        24,
        27,
    ]

    actual_f = list(
        base.f
    )

    if actual_f != expected_f:
        raise RuntimeError(
            "unexpected Detect.f: "
            f"{actual_f}"
        )

    try:
        ref = next(
            base.parameters()
        )

        device = ref.device
        dtype = ref.dtype

    except StopIteration:
        device = torch.device(
            "cpu"
        )
        dtype = torch.float32

    wrapper = CSCMV1DetectWrapper(
        base=base,
        p3_channels=128,
        p2_channels=64,
    )

    wrapper = wrapper.to(
        device=device,
        dtype=dtype,
    )

    added = sum(
        p.numel()
        for name, p
        in wrapper.named_parameters()
        if not name.startswith(
            "base."
        )
    )

    if added != EXPECTED_CSCM_V1_PARAMS:
        raise RuntimeError(
            "CSCM-v1 added params mismatch: "
            f"{added:,} != "
            f"{EXPECTED_CSCM_V1_PARAMS:,}"
        )

    model.model[28] = wrapper

    model._cscm_v1_installed = True

    return model


try:
    torch.serialization.add_safe_globals(
        [
            CSCMV1DetectWrapper,
        ]
    )
except Exception:
    pass


__all__ = [
    "CSCMV1DetectWrapper",
    "install_cscm_v1",
    "EXPECTED_CSCM_V1_PARAMS",
]
