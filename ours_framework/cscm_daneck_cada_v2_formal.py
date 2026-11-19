#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils import RANK

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
    EXPECTED_CSCM_V1_PARAMS,
    install_cscm_v1,
)


EXPECTED_BASE_PARAMS = 10_876_092
EXPECTED_TOTAL_PARAMS = 10_888_572


def nparams(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


class CSCMV1FormalTrainer(
    DANeckCADAV2Trainer
):
    """
    Complete static CADA + DANeck + CSCM model.

    YOLOv8s-P2
      + CADA-v2
      + DANeck-v1
      + CSCM-v1

    P3/node21 semantic context
      -> modulates P2 Detect input only.

    No hooks.
    No diagnostic forwards.
    No auxiliary loss.
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

        if not isinstance(
            model,
            CADAV2DetectionModel,
        ):
            raise RuntimeError(
                "CSCM wrong model class"
            )

        if not isinstance(
            model.model[18],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "CSCM DANeck node18 missing"
            )

        if not isinstance(
            model.model[21],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "CSCM DANeck node21 missing"
            )

        before = nparams(
            model
        )

        if before != EXPECTED_BASE_PARAMS:
            raise RuntimeError(
                "CSCM base params mismatch: "
                f"{before:,}"
            )

        install_cscm_v1(
            model
        )

        after = nparams(
            model
        )

        added = after - before

        if (
            added
            != EXPECTED_CSCM_V1_PARAMS
        ):
            raise RuntimeError(
                "CSCM added params mismatch: "
                f"{added:,}"
            )

        if (
            after
            != EXPECTED_TOTAL_PARAMS
        ):
            raise RuntimeError(
                "CSCM total params mismatch: "
                f"{after:,}"
            )

        if not isinstance(
            model.model[28],
            CSCMV1DetectWrapper,
        ):
            raise RuntimeError(
                "CSCM node28 missing"
            )

        wrapper = model.model[28]

        if (
            torch.count_nonzero(
                wrapper.gate_out.weight
            ).item()
            != 0
        ):
            raise RuntimeError(
                "CSCM gate weight "
                "not zero-init"
            )

        if (
            torch.count_nonzero(
                wrapper.gate_out.bias
            ).item()
            != 0
        ):
            raise RuntimeError(
                "CSCM gate bias "
                "not zero-init"
            )

        if RANK in (-1, 0):

            print()
            print("=" * 112)
            print(
                "CSCM-v1 COMPLETE MODEL CONSTRUCTION"
            )
            print("=" * 112)

            print(
                "base params      =",
                f"{before:,}",
            )

            print(
                "CSCM params      =",
                f"{added:,}",
            )

            print(
                "candidate params =",
                f"{after:,}",
            )

            print(
                "CSCM node28      = PASS"
            )

            print(
                "context source   = P3/node21"
            )

            print(
                "target           = P2 Detect input"
            )

            print(
                "DANeck node18/21 = PASS"
            )

            print(
                "gate zero-init   = PASS"
            )

            print(
                "training hooks   = NONE"
            )

            print("=" * 112)

        return model


try:
    torch.serialization.add_safe_globals(
        [
            CSCMV1FormalTrainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "CSCMV1FormalTrainer",
]
