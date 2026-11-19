#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer

from ours_framework.daneck_v1 import (
    DANeckWrapper,
    install_daneck_v1,
)

from ours_framework.cscm_v1 import (
    CSCMV1DetectWrapper,
    EXPECTED_CSCM_V1_PARAMS,
    install_cscm_v1,
)

from ours_framework.ablation_75_78_trainers import (
    SafeDANeckFinalMixin,
)


# ============================================================
# UAVDT: nc = 3
#
# YOLOv8s-P2:
#   10,637,756
#
# + DANeck:
#   +236,516
#   =10,874,272
#
# + CSCM:
#   +12,480
#   =10,650,236
# ============================================================

UAVDT_BASE_PARAMS = 10_637_756
UAVDT_DANECK_PARAMS = 10_874_272
UAVDT_CSCM_PARAMS = 10_650_236

DANECK_DELTA = 236_516
CSCM_DELTA = 12_480


def nparams(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


def _check_nc3(model):
    nc = getattr(model, "nc", None)

    if nc is not None and int(nc) != 3:
        raise RuntimeError(
            f"UAVDT requires nc=3, got {nc}"
        )


class UAVDTDANeckOnlyTrainer(
    SafeDANeckFinalMixin,
    DetectionTrainer,
):
    """YOLOv8s-P2 + DANeck for UAVDT."""

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ):
        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        _check_nc3(model)

        before = nparams(model)

        if before != UAVDT_BASE_PARAMS:
            raise RuntimeError(
                "UAVDT DANeck base params mismatch: "
                f"{before:,} != {UAVDT_BASE_PARAMS:,}"
            )

        install_daneck_v1(model)

        after = nparams(model)

        if after - before != DANECK_DELTA:
            raise RuntimeError(
                "UAVDT DANeck delta mismatch: "
                f"{after-before:,}"
            )

        if after != UAVDT_DANECK_PARAMS:
            raise RuntimeError(
                "UAVDT DANeck total params mismatch: "
                f"{after:,} != {UAVDT_DANECK_PARAMS:,}"
            )

        if not isinstance(
            model.model[18],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "DANeck node18 missing"
            )

        if not isinstance(
            model.model[21],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "DANeck node21 missing"
            )

        return model


class UAVDTCSCMOnlyTrainer(
    DetectionTrainer,
):
    """YOLOv8s-P2 + CSCM for UAVDT."""

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ):
        model = super().get_model(
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        _check_nc3(model)

        before = nparams(model)

        if before != UAVDT_BASE_PARAMS:
            raise RuntimeError(
                "UAVDT CSCM base params mismatch: "
                f"{before:,} != {UAVDT_BASE_PARAMS:,}"
            )

        install_cscm_v1(model)

        after = nparams(model)

        if after - before != EXPECTED_CSCM_V1_PARAMS:
            raise RuntimeError(
                "UAVDT CSCM delta mismatch: "
                f"{after-before:,}"
            )

        if after != UAVDT_CSCM_PARAMS:
            raise RuntimeError(
                "UAVDT CSCM total params mismatch: "
                f"{after:,} != {UAVDT_CSCM_PARAMS:,}"
            )

        if not isinstance(
            model.model[28],
            CSCMV1DetectWrapper,
        ):
            raise RuntimeError(
                "CSCM node28 missing"
            )

        return model


__all__ = [
    "UAVDTDANeckOnlyTrainer",
    "UAVDTCSCMOnlyTrainer",
]
