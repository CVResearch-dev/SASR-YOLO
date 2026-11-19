#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import torch

from ours_framework.cada_v2 import (
    CADAV2DetectionModel,
    CADAV2Trainer,
)

from ours_framework.daneck_v1 import (
    DeformableAlignmentBlock,
    DANeckWrapper,
    install_daneck_v1,
)


EXPECTED_ADDED_PARAMS = 236_516


class DANeckCADAV2Trainer(CADAV2Trainer):
    """
    Static CADA with deformable alignment at P2 and P3.

    Order:
      1. construct CADAV2DetectionModel
      2. load official YOLOv8s pretrained weights
      3. install DANeck-v1

    Pretrained weights are loaded before the alignment blocks
    are installed, preserving the CADA initialization order.
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

        if getattr(
            model,
            "_daneck_v1_installed",
            False,
        ):
            raise RuntimeError(
                "DANeck-v1 unexpectedly already installed"
            )

        before = sum(
            p.numel()
            for p in model.parameters()
        )

        p2, p3 = install_daneck_v1(
            model
        )

        after = sum(
            p.numel()
            for p in model.parameters()
        )

        added = after - before

        if added != EXPECTED_ADDED_PARAMS:
            raise RuntimeError(
                "DANeck-v1 added params abnormal: "
                f"{added:,}, "
                f"expected {EXPECTED_ADDED_PARAMS:,}"
            )

        if not isinstance(
            model.model[18],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "P2 DANeck wrapper missing"
            )

        if not isinstance(
            model.model[21],
            DANeckWrapper,
        ):
            raise RuntimeError(
                "P3 DANeck wrapper missing"
            )

        model._daneck_v1_added_params = added

        return model

    def final_eval(self):
        """
        Safe finalization for the deformable-convolution model.

        DANeck-v1 uses torchvision DeformConv2d. The normal
        Ultralytics post-training standalone final validation
        enters AutoBackend(fuse=True), whose model-info/THOP
        profiling path causes a native SIGSEGV with this model.

        Training-time validation and EarlyStopping have already
        completed normally, so here we:
          1. keep last.pt / best.pt optimizer stripping;
          2. skip only the redundant post-training validation.

        Standalone validation must use the separate safe manual-fuse
        evaluation entry point supplied with this release.
        """

        from ultralytics.utils.torch_utils import (
            strip_optimizer,
        )

        ddp = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        rank = (
            torch.distributed.get_rank()
            if ddp
            else 0
        )

        if rank == 0:

            if self.last.exists():
                strip_optimizer(
                    self.last
                )

            if self.best.exists():
                strip_optimizer(
                    self.best
                )

            print()
            print("=" * 96)
            print(
                "DANeck-v1 SAFE FINALIZATION"
            )
            print(
                "last.pt / best.pt optimizer stripping: PASS"
            )
            print(
                "Ultralytics redundant AutoBackend(fuse=True) "
                "final validation: SKIPPED"
            )
            print(
                "Training-time validation / EarlyStopping / "
                "best.pt selection: UNCHANGED"
            )
            print("=" * 96)

        # Rank0 may still be writing stripped checkpoints.
        # Do not allow the other DDP ranks to exit early.
        if ddp:
            torch.distributed.barrier(
                device_ids=[
                    torch.cuda.current_device()
                ]
            )


try:
    torch.serialization.add_safe_globals(
        [
            DeformableAlignmentBlock,
            DANeckWrapper,
            DANeckCADAV2Trainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "DANeckCADAV2Trainer",
]
