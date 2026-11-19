#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ours_framework.cada_v2 import (
    CADAV2DetectionModel,
    CADAV2Trainer,
)

from ours_framework.daneck_cada_v2 import (
    DANeckCADAV2Trainer,
)

from ours_framework.daneck_v1 import (
    DANeckWrapper,
)


EXPECTED_ADDED_PARAMS = {
    "p2": 51_538,
    "p3": 184_978,
}


def _module_device_dtype(module: nn.Module):
    p = next(module.parameters())
    return p.device, p.dtype


def install_daneck_placement(
    model: nn.Module,
    placement: str,
):
    """
    Install DANeck at exactly one pyramid level.

    P2 = layer18, channels=64
    P3 = layer21, channels=128
    """

    placement = str(placement).lower()

    if placement not in ("p2", "p3"):
        raise ValueError(
            f"Unsupported DANeck placement: {placement}"
        )

    if getattr(
        model,
        "_daneck_v1_installed",
        False,
    ):
        raise RuntimeError(
            "DANeck already installed"
        )

    if not hasattr(model, "model"):
        raise RuntimeError(
            "Ultralytics model.model not found"
        )

    layers = model.model

    if len(layers) <= 21:
        raise RuntimeError(
            "Unexpected YOLOv8s-P2 architecture"
        )

    before = sum(
        p.numel()
        for p in model.parameters()
    )

    if placement == "p2":
        idx = 18
        channels = 64
    else:
        idx = 21
        channels = 128

    base = layers[idx]

    if isinstance(base, DANeckWrapper):
        raise RuntimeError(
            f"layer {idx} already wrapped"
        )

    dev, dtype = _module_device_dtype(base)

    wrapper = DANeckWrapper(
        base=base,
        channels=channels,
    ).to(
        device=dev,
        dtype=dtype,
    )

    layers[idx] = wrapper

    after = sum(
        p.numel()
        for p in model.parameters()
    )

    added = after - before
    expected = EXPECTED_ADDED_PARAMS[
        placement
    ]

    if added != expected:
        raise RuntimeError(
            f"{placement} added params mismatch: "
            f"{added:,} != {expected:,}"
        )

    has_p2 = isinstance(
        layers[18],
        DANeckWrapper,
    )

    has_p3 = isinstance(
        layers[21],
        DANeckWrapper,
    )

    if placement == "p2":
        if not has_p2 or has_p3:
            raise RuntimeError(
                "P2-only structural check failed"
            )

    if placement == "p3":
        if has_p2 or not has_p3:
            raise RuntimeError(
                "P3-only structural check failed"
            )

    model._daneck_v1_installed = True
    model._daneck_placement = placement
    model._daneck_v1_added_params = added

    print()
    print("=" * 92)
    print("DANeck PLACEMENT INSTALLATION")
    print("placement    =", placement)
    print("P2 / node18  =", has_p2)
    print("P3 / node21  =", has_p3)
    print("added params =", f"{added:,}")
    print("STRUCTURE    = PASS")
    print("=" * 92)

    return wrapper


class _DANeckSingleLevelCADAV2Trainer(
    DANeckCADAV2Trainer
):
    """
    Inherit DANeck safe finalization, but bypass the
    original P2+P3 installer.

    get_model() starts directly from CADAV2Trainer,
    then installs exactly one DANeck branch.
    """

    placement: str = ""

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADAV2DetectionModel:

        model = CADAV2Trainer.get_model(
            self,
            cfg=cfg,
            weights=weights,
            verbose=verbose,
        )

        install_daneck_placement(
            model,
            self.placement,
        )

        return model


class DANeckP2CADAV2Trainer(
    _DANeckSingleLevelCADAV2Trainer
):
    placement = "p2"


class DANeckP3CADAV2Trainer(
    _DANeckSingleLevelCADAV2Trainer
):
    placement = "p3"


__all__ = [
    "install_daneck_placement",
    "DANeckP2CADAV2Trainer",
    "DANeckP3CADAV2Trainer",
]
