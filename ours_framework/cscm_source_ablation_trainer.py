#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

from ours_framework.cada_v2 import (
    CADAV2DetectionModel,
)

from ours_framework.daneck_v1 import (
    DANeckWrapper,
)

from ours_framework.daneck_cada_v2 import (
    DANeckCADAV2Trainer,
)

from ours_framework.cscm_v1 import (
    CSCMV1DetectWrapper,
    install_cscm_v1,
)


BASE_PARAMS = 10_876_092

SOURCE_CFG = {
    "p2": {
        "index": 0,
        "channels": 64,
        "added_params": 8_384,
    },
    "p3": {
        "index": 1,
        "channels": 128,
        "added_params": 12_480,
    },
    "p4": {
        "index": 2,
        "channels": 256,
        "added_params": 20_672,
    },
}


def nparams(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


class CSCMSourceDetectWrapper(nn.Module):
    """
    CSCM source-control wrapper.

    Same modulation form as formal CSCM-v1:
        source
          -> 1x1 Conv projection to 64 channels
          -> resize to P2 resolution
          -> zero-init 1x1 gate
          -> gain = 1 + tanh(logits)
          -> P2' = P2 * gain

    Only the context source changes.
    """

    def __init__(
        self,
        base: Detect,
        source: str,
    ):
        super().__init__()

        if not isinstance(base, Detect):
            raise TypeError(
                "base must be Detect"
            )

        if source not in SOURCE_CFG:
            raise ValueError(
                f"unsupported source: {source}"
            )

        cfg = SOURCE_CFG[source]

        self.base = base
        self.source = source
        self.source_index = cfg["index"]
        self.source_channels = cfg["channels"]
        self.p2_channels = 64

        self.context_proj = Conv(
            self.source_channels,
            self.p2_channels,
            k=1,
            s=1,
        )

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
            f"CSCMSourceDetectWrapper[{source}]"
        )

        self.np = sum(
            p.numel()
            for p in self.parameters()
        )

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

            if torch.is_tensor(value):
                setattr(
                    self.base,
                    name,
                    fn(value),
                )

        return self

    def modulation(
        self,
        source_feat,
        target_hw,
    ):
        context = self.context_proj(
            source_feat
        )

        context = F.interpolate(
            context,
            size=target_hw,
            mode="nearest",
        )

        logits = self.gate_out(
            context
        )

        return (
            1.0
            + torch.tanh(logits)
        )

    def forward(self, x):
        if not isinstance(
            x,
            (list, tuple),
        ):
            raise TypeError(
                "Detect input must be list/tuple"
            )

        if len(x) != 4:
            raise RuntimeError(
                f"expected 4 Detect inputs, got {len(x)}"
            )

        p2, p3, p4, p5 = x

        feats = [
            p2,
            p3,
            p4,
            p5,
        ]

        source_feat = feats[
            self.source_index
        ]

        if (
            source_feat.shape[1]
            != self.source_channels
        ):
            raise RuntimeError(
                "source channel mismatch"
            )

        gain = self.modulation(
            source_feat,
            p2.shape[-2:],
        )

        if gain.shape != p2.shape:
            raise RuntimeError(
                "gain/P2 shape mismatch"
            )

        p2_mod = p2 * gain

        return self.base(
            [
                p2_mod,
                p3,
                p4,
                p5,
            ]
        )


def install_cscm_source(
    model: nn.Module,
    source: str,
):
    if source == "p3":
        # The P3 control reuses the complete model's CSCM implementation.
        return install_cscm_v1(
            model
        )

    if source not in (
        "p2",
        "p4",
    ):
        raise ValueError(source)

    if getattr(
        model,
        "_cscm_source_installed",
        False,
    ):
        raise RuntimeError(
            "CSCM source wrapper already installed"
        )

    base = model.model[28]

    if not isinstance(
        base,
        Detect,
    ):
        raise RuntimeError(
            "node28 is not Detect"
        )

    expected_f = [
        18,
        21,
        24,
        27,
    ]

    if list(base.f) != expected_f:
        raise RuntimeError(
            f"unexpected Detect.f={list(base.f)}"
        )

    ref = next(
        base.parameters()
    )

    wrapper = CSCMSourceDetectWrapper(
        base=base,
        source=source,
    ).to(
        device=ref.device,
        dtype=ref.dtype,
    )

    added = sum(
        p.numel()
        for name, p
        in wrapper.named_parameters()
        if not name.startswith("base.")
    )

    expected = SOURCE_CFG[
        source
    ]["added_params"]

    if added != expected:
        raise RuntimeError(
            f"{source} added params mismatch: "
            f"{added:,} != {expected:,}"
        )

    model.model[28] = wrapper

    model._cscm_source_installed = True
    model._cscm_source = source

    return model


class _CSCMSourceTrainer(
    DANeckCADAV2Trainer
):
    source = ""

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
                f"base params mismatch: {before:,}"
            )

        install_cscm_source(
            model,
            self.source,
        )

        after = nparams(model)

        expected = (
            BASE_PARAMS
            + SOURCE_CFG[
                self.source
            ]["added_params"]
        )

        if after != expected:
            raise RuntimeError(
                f"total params mismatch: "
                f"{after:,} != {expected:,}"
            )

        wrapper = model.model[28]

        if (
            torch.count_nonzero(
                wrapper.gate_out.weight
            ).item()
            != 0
        ):
            raise RuntimeError(
                "gate weight not zero-init"
            )

        if (
            torch.count_nonzero(
                wrapper.gate_out.bias
            ).item()
            != 0
        ):
            raise RuntimeError(
                "gate bias not zero-init"
            )

        print()
        print("=" * 96)
        print("CSCM SOURCE ABLATION")
        print("source       =", self.source)
        print("target       = P2 Detect input")
        print(
            "added params =",
            f"{after-before:,}",
        )
        print(
            "total params =",
            f"{after:,}",
        )
        print("zero-init    = PASS")
        print("DANeck P2/P3 = PASS")
        print("=" * 96)

        return model


class CSCMP2SourceTrainer(
    _CSCMSourceTrainer
):
    source = "p2"


class CSCMP4SourceTrainer(
    _CSCMSourceTrainer
):
    source = "p4"


try:
    torch.serialization.add_safe_globals(
        [
            CSCMSourceDetectWrapper,
            CSCMP2SourceTrainer,
            CSCMP4SourceTrainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "CSCMSourceDetectWrapper",
    "CSCMP2SourceTrainer",
    "CSCMP4SourceTrainer",
]
