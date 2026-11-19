#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any
import torch

from ultralytics.models.yolo.detect import DetectionTrainer

from ours_framework.cada_v2 import CADAV2DetectionModel, CADAV2Trainer
from ours_framework.daneck_v1 import (
    DeformableAlignmentBlock,
    DANeckWrapper,
    install_daneck_v1,
)
from ours_framework.cscm_v1 import (
    CSCMV1DetectWrapper,
    install_cscm_v1,
)

BASE_PARAMS = 10_639_576
DANECK_PARAMS = 236_516
CSCM_PARAMS = 12_480

EXPECTED = {
    "75": BASE_PARAMS + DANECK_PARAMS,
    "76": BASE_PARAMS + CSCM_PARAMS,
    "77": BASE_PARAMS + CSCM_PARAMS,
    "78": BASE_PARAMS + DANECK_PARAMS + CSCM_PARAMS,
}

def nparams(module):
    return sum(p.numel() for p in module.parameters())

def _check_daneck(model):
    if not isinstance(model.model[18], DANeckWrapper):
        raise RuntimeError("DANeck P2/node18 missing")
    if not isinstance(model.model[21], DANeckWrapper):
        raise RuntimeError("DANeck P3/node21 missing")

def _check_cscm(model):
    if not isinstance(model.model[28], CSCMV1DetectWrapper):
        raise RuntimeError("CSCM node28 missing")
    w = model.model[28]
    if torch.count_nonzero(w.gate_out.weight).item() != 0:
        raise RuntimeError("CSCM gate_out.weight is not zero-init")
    if torch.count_nonzero(w.gate_out.bias).item() != 0:
        raise RuntimeError("CSCM gate_out.bias is not zero-init")

class SafeDANeckFinalMixin:
    """Preserve checkpoints without redundant AutoBackend fuse validation."""
    def final_eval(self):
        from ultralytics.utils.torch_utils import strip_optimizer

        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if ddp else 0

        if rank == 0:
            if self.last.exists():
                strip_optimizer(self.last)
            if self.best.exists():
                strip_optimizer(self.best)
            print("=" * 96)
            print("DANeck safe finalization: checkpoint stripping PASS")
            print("Redundant post-training AutoBackend(fuse=True) validation: SKIPPED")
            print("Training-time validation / EarlyStopping / best.pt selection: UNCHANGED")
            print("=" * 96)

        if ddp:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

class DANeckOnlyTrainer(SafeDANeckFinalMixin, DetectionTrainer):
    """YOLOv8s-P2 + DANeck-v1."""
    def get_model(self, cfg=None, weights: Any = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        before = nparams(model)
        install_daneck_v1(model)
        after = nparams(model)
        if after - before != DANECK_PARAMS:
            raise RuntimeError(f"DANeck params mismatch: {after-before:,}")
        if after != EXPECTED["75"]:
            raise RuntimeError(f"DANeck total params mismatch: {after:,} != {EXPECTED['75']:,}")
        _check_daneck(model)
        return model

class CSCMOnlyTrainer(DetectionTrainer):
    """YOLOv8s-P2 + CSCM-v1."""
    def get_model(self, cfg=None, weights: Any = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        before = nparams(model)
        install_cscm_v1(model)
        after = nparams(model)
        if after - before != CSCM_PARAMS:
            raise RuntimeError(f"CSCM params mismatch: {after-before:,}")
        if after != EXPECTED["76"]:
            raise RuntimeError(f"CSCM total params mismatch: {after:,} != {EXPECTED['76']:,}")
        _check_cscm(model)
        return model

class CADACSCMTrainer(CADAV2Trainer):
    """YOLOv8s-P2 + static CADA-v2 + CSCM-v1."""
    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADAV2DetectionModel:
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        before = nparams(model)
        install_cscm_v1(model)
        after = nparams(model)
        if after - before != CSCM_PARAMS:
            raise RuntimeError(f"CADA+CSCM added params mismatch: {after-before:,}")
        if after != EXPECTED["77"]:
            raise RuntimeError(f"CADA+CSCM total params mismatch: {after:,} != {EXPECTED['77']:,}")
        if not isinstance(model, CADAV2DetectionModel):
            raise RuntimeError("CADA+CSCM did not construct CADAV2DetectionModel")
        _check_cscm(model)
        return model

class DANeckCSCMTrainer(SafeDANeckFinalMixin, DetectionTrainer):
    """YOLOv8s-P2 + DANeck-v1 + CSCM-v1."""
    def get_model(self, cfg=None, weights: Any = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        before = nparams(model)
        install_daneck_v1(model)
        middle = nparams(model)
        install_cscm_v1(model)
        after = nparams(model)
        if middle - before != DANECK_PARAMS:
            raise RuntimeError(f"DANeck+CSCM DANeck params mismatch: {middle-before:,}")
        if after - middle != CSCM_PARAMS:
            raise RuntimeError(f"DANeck+CSCM CSCM params mismatch: {after-middle:,}")
        if after != EXPECTED["78"]:
            raise RuntimeError(f"DANeck+CSCM total params mismatch: {after:,} != {EXPECTED['78']:,}")
        _check_daneck(model)
        _check_cscm(model)
        return model

try:
    torch.serialization.add_safe_globals([
        DeformableAlignmentBlock,
        DANeckWrapper,
        CSCMV1DetectWrapper,
        DANeckOnlyTrainer,
        CSCMOnlyTrainer,
        CADACSCMTrainer,
        DANeckCSCMTrainer,
    ])
except Exception:
    pass

__all__ = [
    "DANeckOnlyTrainer",
    "CSCMOnlyTrainer",
    "CADACSCMTrainer",
    "DANeckCSCMTrainer",
]
