#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

from ours_framework.cada_v2 import CADAV2DetectionModel
from ours_framework.daneck_v1 import DANeckWrapper
from ours_framework.daneck_cada_v2 import DANeckCADAV2Trainer


BASE_PARAMS = 10_876_092

ADDED_PARAMS = {
    "add": 12_480,
    "concat": 12_352,
}


def nparams(module):
    return sum(p.numel() for p in module.parameters())


class CSCMFusionDetectWrapper(nn.Module):
    """
    Controlled alternatives to formal CSCM-v1.

    Both variants:
      - use P3/node21 as the context source;
      - modify only the P2 Detect input;
      - preserve the original Detect module;
      - use zero-initialized residual output;
      - are identity mappings at installation.

    add:
        P2' = P2 + Out(Up(Proj(P3)))

    concat:
        P2' = P2 + Out(Cat(P2, Up(P3)))
    """

    def __init__(
        self,
        base: Detect,
        variant: str,
    ):
        super().__init__()

        if not isinstance(base, Detect):
            raise TypeError("base must be Detect")

        if variant not in ADDED_PARAMS:
            raise ValueError(variant)

        self.base = base
        self.variant = variant

        if variant == "add":
            # 128 -> 64, including BN.
            self.context_proj = Conv(
                128, 64, k=1, s=1
            )

            # 64 -> 64, zero initialized.
            self.add_out = nn.Conv2d(
                64,
                64,
                kernel_size=1,
                bias=True,
            )

            nn.init.zeros_(self.add_out.weight)
            nn.init.zeros_(self.add_out.bias)

        else:
            # Cat(P2[64], Up(P3)[128]) -> 192 channels.
            #
            # 192 * 64 + 64 = 12,352 parameters.
            self.concat_out = nn.Conv2d(
                192,
                64,
                kernel_size=1,
                bias=True,
            )

            nn.init.zeros_(self.concat_out.weight)
            nn.init.zeros_(self.concat_out.bias)

        # Preserve graph metadata.
        self.i = getattr(base, "i", -1)
        self.f = getattr(base, "f", -1)

        self.type = (
            f"CSCMFusionDetectWrapper[{variant}]"
        )

        self.np = nparams(self)

    # --------------------------------------------------------
    # Detect interface
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

    def _apply(self, fn, recurse=True):
        """
        Preserve device/dtype transformation of Detect's
        runtime stride/anchor tensors.
        """
        super()._apply(fn, recurse=recurse)

        for name in ("stride", "anchors", "strides"):
            if not hasattr(self.base, name):
                continue

            value = getattr(self.base, name)

            if torch.is_tensor(value):
                setattr(
                    self.base,
                    name,
                    fn(value),
                )

        return self

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    def fuse_p2(self, p2, p3):
        if p2.shape[1] != 64:
            raise RuntimeError(
                "Unexpected P2 channels"
            )

        if p3.shape[1] != 128:
            raise RuntimeError(
                "Unexpected P3 channels"
            )

        if self.variant == "add":

            context = self.context_proj(p3)

            context = F.interpolate(
                context,
                size=p2.shape[-2:],
                mode="nearest",
            )

            residual = self.add_out(context)

        elif self.variant == "concat":

            p3_up = F.interpolate(
                p3,
                size=p2.shape[-2:],
                mode="nearest",
            )

            fused = torch.cat(
                [p2, p3_up],
                dim=1,
            )

            residual = self.concat_out(fused)

        else:
            raise RuntimeError(
                f"Unknown variant: {self.variant}"
            )

        if residual.shape != p2.shape:
            raise RuntimeError(
                "Residual/P2 shape mismatch"
            )

        return p2 + residual

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(
                "Detect input must be list/tuple"
            )

        if len(x) != 4:
            raise RuntimeError(
                f"Expected four inputs, got {len(x)}"
            )

        p2, p3, p4, p5 = x

        p2_fused = self.fuse_p2(p2, p3)

        return self.base([
            p2_fused,
            p3,
            p4,
            p5,
        ])


def install_cscm_fusion(
    model: nn.Module,
    variant: str,
):
    if variant not in ADDED_PARAMS:
        raise ValueError(variant)

    if getattr(
        model,
        "_cscm_fusion_installed",
        False,
    ):
        raise RuntimeError(
            "CSCM fusion already installed"
        )

    if len(model.model) <= 28:
        raise RuntimeError(
            "Unexpected model graph"
        )

    base = model.model[28]

    if not isinstance(base, Detect):
        raise RuntimeError(
            "node28 must be original Detect"
        )

    expected_f = [18, 21, 24, 27]

    if list(base.f) != expected_f:
        raise RuntimeError(
            f"Unexpected Detect.f={list(base.f)}"
        )

    ref = next(base.parameters())

    wrapper = CSCMFusionDetectWrapper(
        base=base,
        variant=variant,
    ).to(
        device=ref.device,
        dtype=ref.dtype,
    )

    added = sum(
        p.numel()
        for name, p in wrapper.named_parameters()
        if not name.startswith("base.")
    )

    if added != ADDED_PARAMS[variant]:
        raise RuntimeError(
            f"Added params mismatch: "
            f"{added:,} != {ADDED_PARAMS[variant]:,}"
        )

    model.model[28] = wrapper

    model._cscm_fusion_installed = True
    model._cscm_fusion_variant = variant

    return model


class _CSCMFusionTrainer(DANeckCADAV2Trainer):

    fusion_variant = ""

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

        if not isinstance(
            model,
            CADAV2DetectionModel,
        ):
            raise RuntimeError(
                "Wrong detection model"
            )

        if not isinstance(
            model.model[18],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "DANeck P2 missing"
            )

        if not isinstance(
            model.model[21],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "DANeck P3 missing"
            )

        before = nparams(model)

        if before != BASE_PARAMS:
            raise RuntimeError(
                f"Base params mismatch: {before:,}"
            )

        install_cscm_fusion(
            model,
            self.fusion_variant,
        )

        after = nparams(model)

        expected = (
            BASE_PARAMS
            + ADDED_PARAMS[self.fusion_variant]
        )

        if after != expected:
            raise RuntimeError(
                f"Total params mismatch: "
                f"{after:,} != {expected:,}"
            )

        wrapper = model.model[28]

        if not isinstance(
            wrapper,
            CSCMFusionDetectWrapper,
        ):
            raise RuntimeError(
                "Fusion wrapper missing"
            )

        if wrapper.variant != self.fusion_variant:
            raise RuntimeError(
                "Fusion variant mismatch"
            )

        if self.fusion_variant == "add":
            out = wrapper.add_out
        else:
            out = wrapper.concat_out

        if torch.count_nonzero(out.weight).item() != 0:
            raise RuntimeError(
                "Output weight not zero initialized"
            )

        if torch.count_nonzero(out.bias).item() != 0:
            raise RuntimeError(
                "Output bias not zero initialized"
            )

        print()
        print("=" * 100)
        print("CSCM FUSION CONTROL")
        print("variant        =", self.fusion_variant)
        print("context source = P3/node21")
        print("target         = P2 Detect input")
        print("added params   =", ADDED_PARAMS[self.fusion_variant])
        print("total params   =", after)
        print("DANeck P2/P3   = PASS")
        print("zero-init      = PASS")
        print("=" * 100)

        return model


class CSCMAddTrainer(_CSCMFusionTrainer):
    fusion_variant = "add"


class CSCMConcatTrainer(_CSCMFusionTrainer):
    fusion_variant = "concat"


try:
    torch.serialization.add_safe_globals([
        CSCMFusionDetectWrapper,
        CSCMAddTrainer,
        CSCMConcatTrainer,
    ])
except Exception:
    pass


__all__ = [
    "CSCMFusionDetectWrapper",
    "CSCMAddTrainer",
    "CSCMConcatTrainer",
    "install_cscm_fusion",
    "ADDED_PARAMS",
]
