#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CADA with difficulty-aware Dynamic TopK and static master-soft targets.

Historical module/class names are retained for checkpoint compatibility.

设计原则：
1. 不修改 Ultralytics 源码；
2. dynamic_variant="off" 时，标签分配与 CADA-v1 严格等价；
3. dynamic_variant="full" 时，仅将固定 TAL TopK 替换为按 GT 难度变化的 TopK；
4. 保留静态分类目标重加权及P2/P3竞争，不包含时间退火；
5. 不增加模型参数，不改变推理图和 NMS。

动态 TopK：
    difficulty = 1 - best_iou
    K = round_half_up(
        topk_min + difficulty * (topk_max - topk_min)
    )

其中 best_iou 使用 GT 内候选预测框与该 GT 的真实 IoU；候选排序仍沿用
当前 Ultralytics TAL 的 task-aligned metric，不改动其分类/定位对齐定义。
"""

from __future__ import annotations

import os
from typing import Any

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner

from ours_framework.cada import (
    CADADetectionLoss,
    CADADetectionModel,
    DEFAULT_ANNEAL_MIN,
    DEFAULT_LEVEL_COUNT,
    DEFAULT_QUALITY_GAMMA,
    DEFAULT_SOFT_TOPK,
    DEFAULT_VARIANT,
)


VALID_DYNAMIC_VARIANTS = {"off", "full"}
DEFAULT_DYNAMIC_VARIANT = "full"
DEFAULT_DYNAMIC_TOPK_MIN = 0  # PRE-ACCEPTANCE PLACEHOLDERS
DEFAULT_DYNAMIC_TOPK_MAX = 0
DEFAULT_FIXED_TOPK = 0


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


class DifficultyAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """仅训练期使用的 Difficulty-aware Dynamic TopK TAL。"""

    def __init__(
        self,
        *,
        dynamic_variant: str = DEFAULT_DYNAMIC_VARIANT,
        topk_min: int = DEFAULT_DYNAMIC_TOPK_MIN,
        topk_max: int = DEFAULT_DYNAMIC_TOPK_MAX,
        fixed_topk: int = DEFAULT_FIXED_TOPK,
        num_classes: int = 80,
        alpha: float = 0.0,
        beta: float = 0.0,
        stride: list[int] | tuple[int, ...] = (8, 16, 32),
        eps: float = 1e-9,
    ) -> None:
        dynamic_variant = str(dynamic_variant).strip().lower()
        if dynamic_variant not in VALID_DYNAMIC_VARIANTS:
            raise ValueError(
                f"不支持的 Dynamic TopK 变体：{dynamic_variant}。"
                f"可选值：{sorted(VALID_DYNAMIC_VARIANTS)}"
            )
        if fixed_topk < 1:
            raise ValueError("fixed_topk 必须大于或等于 1。")
        if topk_min < 1:
            raise ValueError("topk_min 必须大于或等于 1。")
        if topk_max < topk_min:
            raise ValueError("topk_max 必须大于或等于 topk_min。")

        active_topk = fixed_topk if dynamic_variant == "off" else topk_max
        super().__init__(
            topk=int(active_topk),
            topk2=int(active_topk),
            num_classes=int(num_classes),
            alpha=float(alpha),
            beta=float(beta),
            stride=list(stride),
            eps=float(eps),
        )

        self.dynamic_variant = dynamic_variant
        self.dynamic_enabled = dynamic_variant == "full"
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.fixed_topk = int(fixed_topk)

        self._last_batch_state: dict[str, float | str] = self._empty_state()

    def _empty_state(self) -> dict[str, float | str]:
        return {
            "dynamic_variant": self.dynamic_variant,
            "dynamic_gt_count": 0.0,
            "inside_candidates_per_gt": 0.0,
            "best_iou_mean": 0.0,
            "dynamic_topk_mean": 0.0,
            "dynamic_topk_min_observed": 0.0,
            "dynamic_topk_max_observed": 0.0,
            "selected_before_conflict": 0.0,
        }

    def batch_state(self) -> dict[str, float | str]:
        """返回最近一个 batch 的 Dynamic TopK 统计。"""
        return dict(self._last_batch_state)

    @staticmethod
    def _true_iou_matrix(
        pd_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """只在 GT 内候选位置计算真实 IoU，其他位置保持 0。"""
        batch_size, max_gt, anchor_count = valid_mask.shape
        ious = torch.zeros(
            (batch_size, max_gt, anchor_count),
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )

        if not bool(valid_mask.any()):
            return ious

        expanded_pd = pd_bboxes.unsqueeze(1).expand(-1, max_gt, -1, -1)
        expanded_gt = gt_bboxes.unsqueeze(2).expand(-1, -1, anchor_count, -1)

        selected_pd = expanded_pd[valid_mask]
        selected_gt = expanded_gt[valid_mask]
        selected_iou = bbox_iou(
            selected_gt,
            selected_pd,
            xywh=False,
            CIoU=False,
        ).squeeze(-1)

        ious[valid_mask] = selected_iou.clamp(min=0.0, max=1.0)
        return ious

    def _select_dynamic_topk(
        self,
        metrics: torch.Tensor,
        valid_mask: torch.Tensor,
        dynamic_k: torch.Tensor,
    ) -> torch.Tensor:
        """统一取 topk_max，再按每个 GT 的 K 生成不同长度掩码。"""
        anchor_count = metrics.shape[-1]
        max_k = min(self.topk_max, anchor_count)

        masked_metrics = metrics.masked_fill(~valid_mask, -torch.inf)
        topk_values, topk_indices = torch.topk(
            masked_metrics,
            k=max_k,
            dim=-1,
            largest=True,
        )

        ranks = torch.arange(
            max_k,
            device=metrics.device,
            dtype=torch.long,
        ).view(1, 1, max_k)

        rank_mask = ranks < dynamic_k.unsqueeze(-1)
        finite_mask = torch.isfinite(topk_values)
        selected_topk = rank_mask & finite_mask

        selected = torch.zeros_like(metrics, dtype=torch.bool)
        selected.scatter_(
            dim=-1,
            index=topk_indices,
            src=selected_topk,
        )
        selected &= valid_mask
        return selected.to(metrics.dtype)

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """动态模式重写 TopK；关闭时直接走原始 TAL 路径。"""
        if not self.dynamic_enabled:
            self._last_batch_state = self._empty_state()
            return super().get_pos_mask(
                pd_scores,
                pd_bboxes,
                gt_labels,
                gt_bboxes,
                anc_points,
                mask_gt,
            )

        mask_in_gts = self.select_candidates_in_gts(
            anc_points,
            gt_bboxes,
            mask_gt,
        )
        valid_mask = mask_in_gts.bool() & mask_gt.bool()

        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            valid_mask,
        )

        true_ious = self._true_iou_matrix(
            pd_bboxes,
            gt_bboxes,
            valid_mask,
        )

        valid_count = valid_mask.sum(dim=-1).long()
        valid_gt = mask_gt.squeeze(-1).bool() & valid_count.gt(0)

        best_iou = true_ious.amax(dim=-1).clamp(min=0.0, max=1.0)
        difficulty = 1.0 - best_iou

        raw_k = self.topk_min + difficulty * (self.topk_max - self.topk_min)
        # 使用 half-up，避免 torch.round 在 x.5 处采用 bankers rounding。
        dynamic_k = torch.floor(raw_k + 0.5).long()
        dynamic_k = dynamic_k.clamp(min=self.topk_min, max=self.topk_max)
        dynamic_k = torch.minimum(dynamic_k, valid_count)
        dynamic_k = torch.where(valid_gt, dynamic_k, torch.zeros_like(dynamic_k))

        mask_topk = self._select_dynamic_topk(
            metrics=align_metric,
            valid_mask=valid_mask,
            dynamic_k=dynamic_k,
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt

        observed_k = dynamic_k[valid_gt]
        observed_iou = best_iou[valid_gt]
        observed_candidates = valid_count[valid_gt]

        if observed_k.numel() > 0:
            self._last_batch_state = {
                "dynamic_variant": self.dynamic_variant,
                "dynamic_gt_count": float(observed_k.numel()),
                "inside_candidates_per_gt": float(
                    observed_candidates.float().mean().item()
                ),
                "best_iou_mean": float(observed_iou.float().mean().item()),
                "dynamic_topk_mean": float(observed_k.float().mean().item()),
                "dynamic_topk_min_observed": float(observed_k.min().item()),
                "dynamic_topk_max_observed": float(observed_k.max().item()),
                "selected_before_conflict": float(mask_pos.sum().item()),
            }
        else:
            self._last_batch_state = self._empty_state()

        return mask_pos, align_metric, overlaps


class CADAV2DetectionLoss(CADADetectionLoss):
    """CADA-v1 分类重加权 + Difficulty-aware Dynamic TopK。"""

    def __init__(
        self,
        model: DetectionModel,
        variant: str = DEFAULT_VARIANT,
        soft_topk: int = DEFAULT_SOFT_TOPK,
        quality_gamma: float = DEFAULT_QUALITY_GAMMA,
        anneal_min: float = DEFAULT_ANNEAL_MIN,
        level_count: int = DEFAULT_LEVEL_COUNT,
        dynamic_variant: str = DEFAULT_DYNAMIC_VARIANT,
        dynamic_topk_min: int = DEFAULT_DYNAMIC_TOPK_MIN,
        dynamic_topk_max: int = DEFAULT_DYNAMIC_TOPK_MAX,
        fixed_topk: int = DEFAULT_FIXED_TOPK,
    ) -> None:
        super().__init__(
            model=model,
            variant=variant,
            soft_topk=soft_topk,
            quality_gamma=quality_gamma,
            anneal_min=anneal_min,
            level_count=level_count,
        )

        base_assigner = self.assigner
        self.dynamic_variant = str(dynamic_variant).strip().lower()
        self.dynamic_topk_min = int(dynamic_topk_min)
        self.dynamic_topk_max = int(dynamic_topk_max)
        self.fixed_topk = int(fixed_topk)

        self.assigner = DifficultyAwareTaskAlignedAssigner(
            dynamic_variant=self.dynamic_variant,
            topk_min=self.dynamic_topk_min,
            topk_max=self.dynamic_topk_max,
            fixed_topk=self.fixed_topk,
            num_classes=int(base_assigner.num_classes),
            alpha=float(base_assigner.alpha),
            beta=float(base_assigner.beta),
            stride=list(base_assigner.stride),
            eps=float(base_assigner.eps),
        )

        self._dynamic_batches = 0
        self._dynamic_gt_count = 0.0
        self._dynamic_candidate_per_gt_sum = 0.0
        self._dynamic_best_iou_sum = 0.0
        self._dynamic_topk_sum = 0.0
        self._dynamic_topk_min = float("inf")
        self._dynamic_topk_max = float("-inf")
        self._dynamic_selected_sum = 0.0

        self.last_epoch_state.update(
            {
                "dynamic_variant": self.dynamic_variant,
                "dynamic_topk_min": float(self.dynamic_topk_min),
                "dynamic_topk_max": float(self.dynamic_topk_max),
                "fixed_topk": float(self.fixed_topk),
                "dynamic_gt_count": 0.0,
                "inside_candidates_per_gt": 0.0,
                "best_iou_mean": 0.0,
                "dynamic_topk_mean": 0.0,
                "dynamic_topk_min_observed": 0.0,
                "dynamic_topk_max_observed": 0.0,
                "selected_before_conflict": 0.0,
            }
        )

        if RANK in (-1, 0):
            LOGGER.info(
                "[CADA-v2] Difficulty-aware Dynamic TopK："
                f"dynamic_variant={self.dynamic_variant}, "
                f"topk_min={self.dynamic_topk_min}, "
                f"topk_max={self.dynamic_topk_max}, "
                f"fixed_topk={self.fixed_topk}"
            )

    def _accumulate_dynamic_state(self) -> None:
        state = self.assigner.batch_state()
        gt_count = float(state["dynamic_gt_count"])

        self._dynamic_batches += 1
        self._dynamic_gt_count += gt_count
        self._dynamic_selected_sum += float(state["selected_before_conflict"])

        if gt_count > 0:
            self._dynamic_candidate_per_gt_sum += (
                float(state["inside_candidates_per_gt"]) * gt_count
            )
            self._dynamic_best_iou_sum += float(state["best_iou_mean"]) * gt_count
            self._dynamic_topk_sum += float(state["dynamic_topk_mean"]) * gt_count
            self._dynamic_topk_min = min(
                self._dynamic_topk_min,
                float(state["dynamic_topk_min_observed"]),
            )
            self._dynamic_topk_max = max(
                self._dynamic_topk_max,
                float(state["dynamic_topk_max_observed"]),
            )

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = super().loss(preds, batch)
        self._accumulate_dynamic_state()
        return result

    def update(self) -> None:
        """合并 CADA-v1 与 Dynamic TopK 的逐 epoch 统计。"""
        gt_count = max(self._dynamic_gt_count, 1.0)
        batch_count = max(self._dynamic_batches, 1)

        dynamic_summary = {
            "dynamic_variant": self.dynamic_variant,
            "dynamic_topk_min": float(self.dynamic_topk_min),
            "dynamic_topk_max": float(self.dynamic_topk_max),
            "fixed_topk": float(self.fixed_topk),
            "dynamic_gt_count": self._dynamic_gt_count / batch_count,
            "inside_candidates_per_gt": (
                self._dynamic_candidate_per_gt_sum / gt_count
                if self._dynamic_gt_count > 0
                else 0.0
            ),
            "best_iou_mean": (
                self._dynamic_best_iou_sum / gt_count
                if self._dynamic_gt_count > 0
                else 0.0
            ),
            "dynamic_topk_mean": (
                self._dynamic_topk_sum / gt_count
                if self._dynamic_gt_count > 0
                else 0.0
            ),
            "dynamic_topk_min_observed": (
                self._dynamic_topk_min
                if self._dynamic_gt_count > 0
                else 0.0
            ),
            "dynamic_topk_max_observed": (
                self._dynamic_topk_max
                if self._dynamic_gt_count > 0
                else 0.0
            ),
            "selected_before_conflict": self._dynamic_selected_sum / batch_count,
        }

        super().update()
        self.last_epoch_state.update(dynamic_summary)

        epoch = int(float(self.last_epoch_state.get("epoch", 0.0)))
        should_log = epoch == 1 or epoch % 10 == 0 or epoch >= self.total_epochs
        if should_log and RANK in (-1, 0):
            LOGGER.info(
                "[CADA-v2] "
                f"epoch={epoch}/{self.total_epochs}, "
                f"best_iou={dynamic_summary['best_iou_mean']:.4f}, "
                f"dynamic_k={dynamic_summary['dynamic_topk_mean']:.3f}, "
                f"K范围={dynamic_summary['dynamic_topk_min_observed']:.0f}-"
                f"{dynamic_summary['dynamic_topk_max_observed']:.0f}, "
                f"GT/批={dynamic_summary['dynamic_gt_count']:.1f}, "
                f"冲突前正样本/批={dynamic_summary['selected_before_conflict']:.1f}"
            )

        self._dynamic_batches = 0
        self._dynamic_gt_count = 0.0
        self._dynamic_candidate_per_gt_sum = 0.0
        self._dynamic_best_iou_sum = 0.0
        self._dynamic_topk_sum = 0.0
        self._dynamic_topk_min = float("inf")
        self._dynamic_topk_max = float("-inf")
        self._dynamic_selected_sum = 0.0


class CADAV2DetectionModel(CADADetectionModel):
    """YOLOv8 检测模型 + CADA-v2 训练损失。"""

    def __init__(
        self,
        cfg: str | dict = "yolov8.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        cada_variant: str = DEFAULT_VARIANT,
        cada_soft_topk: int = DEFAULT_SOFT_TOPK,
        cada_quality_gamma: float = DEFAULT_QUALITY_GAMMA,
        cada_anneal_min: float = DEFAULT_ANNEAL_MIN,
        cada_level_count: int = DEFAULT_LEVEL_COUNT,
        dynamic_variant: str = DEFAULT_DYNAMIC_VARIANT,
        dynamic_topk_min: int = DEFAULT_DYNAMIC_TOPK_MIN,
        dynamic_topk_max: int = DEFAULT_DYNAMIC_TOPK_MAX,
        fixed_topk: int = DEFAULT_FIXED_TOPK,
    ) -> None:
        self.cada_v2_dynamic_variant = str(dynamic_variant).strip().lower()
        self.cada_v2_dynamic_topk_min = int(dynamic_topk_min)
        self.cada_v2_dynamic_topk_max = int(dynamic_topk_max)
        self.cada_v2_fixed_topk = int(fixed_topk)

        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            cada_variant=cada_variant,
            cada_soft_topk=cada_soft_topk,
            cada_quality_gamma=cada_quality_gamma,
            cada_anneal_min=cada_anneal_min,
            cada_level_count=cada_level_count,
        )

    def init_criterion(self) -> CADAV2DetectionLoss:
        return CADAV2DetectionLoss(
            model=self,
            variant=self.cada_variant,
            soft_topk=self.cada_soft_topk,
            quality_gamma=self.cada_quality_gamma,
            anneal_min=self.cada_anneal_min,
            level_count=self.cada_level_count,
            dynamic_variant=self.cada_v2_dynamic_variant,
            dynamic_topk_min=self.cada_v2_dynamic_topk_min,
            dynamic_topk_max=self.cada_v2_dynamic_topk_max,
            fixed_topk=self.cada_v2_fixed_topk,
        )


class CADAV2Trainer(DetectionTrainer):
    """DDP 安全地创建 CADAV2DetectionModel。"""

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADAV2DetectionModel:
        model = CADAV2DetectionModel(
            cfg=cfg,
            ch=int(self.data.get("channels", 3)),
            nc=int(self.data["nc"]),
            verbose=verbose and RANK == -1,
            cada_variant=_env_str("CADA_VARIANT", DEFAULT_VARIANT).lower(),
            cada_soft_topk=_env_int("CADA_SOFT_TOPK", DEFAULT_SOFT_TOPK),
            cada_quality_gamma=_env_float(
                "CADA_QUALITY_GAMMA", DEFAULT_QUALITY_GAMMA
            ),
            cada_anneal_min=_env_float("CADA_ANNEAL_MIN", DEFAULT_ANNEAL_MIN),
            cada_level_count=_env_int("CADA_LEVEL_COUNT", DEFAULT_LEVEL_COUNT),
            dynamic_variant=_env_str(
                "CADA_V2_DYNAMIC_VARIANT", DEFAULT_DYNAMIC_VARIANT
            ).lower(),
            dynamic_topk_min=_env_int(
                "CADA_V2_TOPK_MIN", DEFAULT_DYNAMIC_TOPK_MIN
            ),
            dynamic_topk_max=_env_int(
                "CADA_V2_TOPK_MAX", DEFAULT_DYNAMIC_TOPK_MAX
            ),
            fixed_topk=_env_int("CADA_V2_FIXED_TOPK", DEFAULT_FIXED_TOPK),
        )

        if weights:
            model.load(weights)
        return model


try:
    torch.serialization.add_safe_globals(
        [
            DifficultyAwareTaskAlignedAssigner,
            CADAV2DetectionLoss,
            CADAV2DetectionModel,
            CADAV2Trainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "DifficultyAwareTaskAlignedAssigner",
    "CADAV2DetectionLoss",
    "CADAV2DetectionModel",
    "CADAV2Trainer",
]
