#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import ours_framework.cscm_daneck_cada_v2_formal as formal

from ours_framework.cada_v2 import (
    CADAV2DetectionModel,
)


# UAVDT: nc = 3
#
# baseline:
#   10,637,756
#
# + DANeck:
#   10,874,272
#
# + CSCM:
#   10,886,752

UAVDT_EXPECTED_BASE_PARAMS = 10_874_272
UAVDT_EXPECTED_TOTAL_PARAMS = 10_886_752


class UAVDTCSCMV1FormalTrainer(
    formal.CSCMV1FormalTrainer
):
    """
    UAVDT-specific wrapper around the static
    CADA + DANeck + CSCM trainer.

    The original VisDrone formal implementation
    is left untouched. Only the absolute parameter
    expectations are adjusted for nc=3.
    """

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADAV2DetectionModel:

        nc = None

        if isinstance(
            getattr(self, "data", None),
            dict,
        ):
            nc = self.data.get(
                "nc",
                None,
            )

        if nc is not None and int(nc) != 3:
            raise RuntimeError(
                "UAVDTCSCMV1FormalTrainer "
                f"requires nc=3, got nc={nc}"
            )

        old_base = (
            formal.EXPECTED_BASE_PARAMS
        )

        old_total = (
            formal.EXPECTED_TOTAL_PARAMS
        )

        formal.EXPECTED_BASE_PARAMS = (
            UAVDT_EXPECTED_BASE_PARAMS
        )

        formal.EXPECTED_TOTAL_PARAMS = (
            UAVDT_EXPECTED_TOTAL_PARAMS
        )

        try:
            model = super().get_model(
                cfg=cfg,
                weights=weights,
                verbose=verbose,
            )

        finally:
            formal.EXPECTED_BASE_PARAMS = (
                old_base
            )

            formal.EXPECTED_TOTAL_PARAMS = (
                old_total
            )

        return model


__all__ = [
    "UAVDTCSCMV1FormalTrainer",
]
