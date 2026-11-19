#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CADA: static cross-scale classification-target reweighting.

The module/class names are retained for checkpoint compatibility. This release
contains no epoch-dependent classification-target annealing.

设计目标：
1. 保留YOLOv8原始Task-Aligned Assigner的一对多匹配；
2. 仅在P2/P3正样本中，为每个GT选择一个主候选；
3. 主候选之外仅保留少量质量感知软候选；
4. 软候选始终采用静态质量权重，与训练进度无关；
5. 框回归和DFL仍使用原始TAL监督；
6. 不增加模型参数，不改变推理图，不修改NMS。

消融变体：
- off：完全关闭CADA，用于损失等价检查；
- master：仅保留跨尺度主候选（C1）；
- one_to_few：最终无退火主候选＋质量感知软候选。

注意：
CADA只重加权分类target_scores。原始target_scores及其归一化项继续用于
CIoU和DFL，以避免破坏YOLOv8原始定位监督。
"""

from __future__ import annotations

import os
from typing import Any

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


VALID_VARIANTS = {
    "off",
    "master",
    "one_to_few",
}

DEFAULT_VARIANT = "one_to_few"
DEFAULT_SOFT_TOPK = 0  # PRE-ACCEPTANCE PLACEHOLDER, NOT A PAPER SETTING
DEFAULT_QUALITY_GAMMA = 0.0
# Historical field name retained for checkpoint compatibility: this is a STATIC
# soft-candidate weight floor, not a time-dependent annealing parameter.
DEFAULT_ANNEAL_MIN = 0.0
DEFAULT_LEVEL_COUNT = 0


def _read_env_str(
    name: str,
    default: str,
) -> str:
    value = os.getenv(name, default)
    return str(value).strip()


def _read_env_int(
    name: str,
    default: int,
) -> int:
    value = _read_env_str(
        name,
        str(default),
    )
    return int(value)


def _read_env_float(
    name: str,
    default: float,
) -> float:
    value = _read_env_str(
        name,
        str(default),
    )
    return float(value)


class CADADetectionLoss(v8DetectionLoss):
    """
    YOLOv8检测损失＋CADA分类目标重加权。

    原始TAL输出：
        target_scores：质量感知软分类目标；
        fg_mask：正样本掩码；
        target_gt_idx：每个正样本对应的GT编号。

    CADA在P2/P3候选内部按“图像＋GT”分组：
        1. 质量最高者作为主候选，分类权重保持1；
        2. 额外保留soft_topk个软候选；
        3. 其余P2/P3重复正样本的分类目标置0；
        4. P4/P5正样本保持原TAL分类目标；
        5. 框回归和DFL始终使用原target_scores。
    """

    def __init__(
        self,
        model: DetectionModel,
        variant: str = DEFAULT_VARIANT,
        soft_topk: int = DEFAULT_SOFT_TOPK,
        quality_gamma: float = DEFAULT_QUALITY_GAMMA,
        anneal_min: float = DEFAULT_ANNEAL_MIN,
        level_count: int = DEFAULT_LEVEL_COUNT,
    ) -> None:
        super().__init__(
            model=model,
        )

        variant = str(variant).strip().lower()

        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"不支持的CADA变体：{variant}。"
                f"可选值：{sorted(VALID_VARIANTS)}"
            )

        if soft_topk < 0:
            raise ValueError(
                "soft_topk必须大于或等于0。"
            )

        if quality_gamma <= 0:
            raise ValueError(
                "quality_gamma必须大于0。"
            )

        if not 0.0 <= anneal_min <= 1.0:
            raise ValueError(
                "anneal_min必须位于[0, 1]。"
            )

        if level_count < 1:
            raise ValueError(
                "level_count必须大于或等于1。"
            )

        self.variant = variant
        self.soft_topk = int(
            soft_topk
        )
        self.quality_gamma = float(
            quality_gamma
        )
        self.anneal_min = float(
            anneal_min
        )
        self.level_count = int(
            level_count
        )

        model_epochs = getattr(
            model.args,
            "epochs",
            0,  # Placeholder; no training is supported by this package.
        )

        self.total_epochs = max(
            int(model_epochs),
            1,
        )
        self.current_epoch = 0

        self._epoch_batches = 0
        self._epoch_candidate_count = 0.0
        self._epoch_group_count = 0.0
        self._epoch_master_count = 0.0
        self._epoch_soft_count = 0.0
        self._epoch_soft_weight_sum = 0.0
        self._epoch_selected_ratio_sum = 0.0

        self.last_epoch_state: dict[
            str,
            float | str
        ] = {
            "variant": self.variant,
            "epoch": 0.0,
            "distinct_strength": self.distinct_strength(),
            "dense_retention": 1.0 - self.distinct_strength(),
            "candidate_count": 0.0,
            "group_count": 0.0,
            "master_count": 0.0,
            "soft_count": 0.0,
            "mean_soft_weight": 0.0,
            "selected_candidate_ratio": 1.0,
        }

        if RANK in (-1, 0):
            LOGGER.info(
                "[CADA static] 已启用跨尺度静态"
                "差异化标签分配："
                f"variant={self.variant}, "
                f"soft_topk={self.soft_topk}, "
                f"quality_gamma="
                f"{self.quality_gamma:.2f}, "
                f"soft_weight_floor="
                f"{self.anneal_min:.2f}, "
                f"levels=P2-P"
                f"{self.level_count + 1}, "
                f"total_epochs={self.total_epochs}"
            )

    def distinct_strength(
        self,
    ) -> float:
        """Return the static mode strength; reject historical timed modes."""

        if self.variant == "off":
            return 0.0

        if self.variant in {
            "master",
            "one_to_few",
        }:
            return 1.0

        raise ValueError(
            f"Unsupported static CADA mode {self.variant!r}; historical timed "
            "training states are not converted automatically."
        )

    @staticmethod
    def _select_one_per_group(
        values: torch.Tensor,
        inverse_groups: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        """
        每个分组选择一个最大值位置。

        当同组存在完全相同的最大值时，使用候选张量中的最早位置，
        保证每个GT只有一个主候选。
        """

        if values.numel() == 0:
            return torch.empty(
                0,
                dtype=torch.long,
                device=values.device,
            )

        group_max = torch.full(
            (group_count,),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        )

        group_max.scatter_reduce_(
            dim=0,
            index=inverse_groups,
            src=values,
            reduce="amax",
            include_self=True,
        )

        valid = torch.isfinite(
            values
        )

        is_group_max = (
            valid
            & (
                values
                == group_max[
                    inverse_groups
                ]
            )
        )

        positions = torch.arange(
            values.numel(),
            device=values.device,
            dtype=torch.long,
        )

        selected_positions = torch.full(
            (group_count,),
            values.numel(),
            dtype=torch.long,
            device=values.device,
        )

        selected_positions.scatter_reduce_(
            dim=0,
            index=inverse_groups[
                is_group_max
            ],
            src=positions[
                is_group_max
            ],
            reduce="amin",
            include_self=True,
        )

        return selected_positions[
            selected_positions
            < values.numel()
        ]

    def _build_classification_weights(
        self,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        features: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        dict[str, float],
    ]:
        """
        为P2/P3正样本构造CADA分类权重。

        返回：
            anchor_weights：[B, A]；
            diagnostics：当前batch统计信息。
        """

        # AMP safety: fg_mask is a logical mask and must remain bool.
        fg_mask = fg_mask.to(dtype=torch.bool)


        batch_size, anchor_count = (
            fg_mask.shape
        )

        anchor_weights = torch.ones(
            (
                batch_size,
                anchor_count,
            ),
            dtype=target_scores.dtype,
            device=target_scores.device,
        )

        strength = self.distinct_strength()
        dense_retention = 1.0 - strength

        empty_diagnostics = {
            "distinct_strength": strength,
            "dense_retention": dense_retention,
            "candidate_count": 0.0,
            "group_count": 0.0,
            "master_count": 0.0,
            "soft_count": 0.0,
            "mean_soft_weight": 0.0,
            "selected_candidate_ratio": 1.0,
        }

        if self.variant == "off":
            return (
                anchor_weights,
                empty_diagnostics,
            )

        if len(features) < self.level_count:
            raise RuntimeError(
                "检测尺度数量少于CADA要求："
                f"实际={len(features)}, "
                f"要求={self.level_count}。"
            )

        selected_level_anchors = sum(
            int(
                feature.shape[-2]
                * feature.shape[-1]
            )
            for feature in features[
                : self.level_count
            ]
        )

        selected_level_anchors = min(
            selected_level_anchors,
            anchor_count,
        )

        anchor_index = torch.arange(
            anchor_count,
            device=fg_mask.device,
        ).view(
            1,
            -1,
        )

        candidate_mask = (
            fg_mask
            & (
                anchor_index
                < selected_level_anchors
            )
        )

        flat_candidate_index = (
            candidate_mask
            .reshape(-1)
            .nonzero(
                as_tuple=False
            )
            .squeeze(1)
        )

        candidate_count = int(
            flat_candidate_index.numel()
        )

        if candidate_count == 0:
            return (
                anchor_weights,
                empty_diagnostics,
            )

        flat_quality = (
            target_scores
            .amax(dim=-1)
            .reshape(-1)[
                flat_candidate_index
            ]
            .detach()
            .float()
        )

        flat_gt_index = (
            target_gt_idx
            .reshape(-1)[
                flat_candidate_index
            ]
            .long()
        )

        flat_batch_index = (
            flat_candidate_index
            // anchor_count
        ).long()

        gt_stride = (
            int(
                target_gt_idx.max()
                .detach()
                .item()
            )
            + 1
        )

        gt_stride = max(
            gt_stride,
            1,
        )

        raw_group_index = (
            flat_batch_index
            * gt_stride
            + flat_gt_index
        )

        _, inverse_groups = torch.unique(
            raw_group_index,
            sorted=True,
            return_inverse=True,
        )

        group_count = int(
            inverse_groups.max()
            .detach()
            .item()
            + 1
        )

        flat_anchor_weights = (
            anchor_weights.reshape(-1)
        )

        # Static master/soft selection: all remaining P2/P3 classification
        # targets are zero at every epoch. Regression/DFL remain unchanged.
        flat_anchor_weights[
            flat_candidate_index
        ] = dense_retention

        remaining_quality = (
            flat_quality.clone()
        )

        master_quality = torch.zeros(
            group_count,
            dtype=flat_quality.dtype,
            device=flat_quality.device,
        )

        selected_master_count = 0
        selected_soft_count = 0
        soft_weight_sum = 0.0

        selection_rounds = 1

        if self.variant == "one_to_few":
            selection_rounds += (
                self.soft_topk
            )

        for selection_rank in range(
            selection_rounds
        ):
            selected_positions = (
                self._select_one_per_group(
                    values=remaining_quality,
                    inverse_groups=(
                        inverse_groups
                    ),
                    group_count=group_count,
                )
            )

            if (
                selected_positions.numel()
                == 0
            ):
                break

            selected_groups = (
                inverse_groups[
                    selected_positions
                ]
            )

            selected_flat_anchors = (
                flat_candidate_index[
                    selected_positions
                ]
            )

            if selection_rank == 0:
                flat_anchor_weights[
                    selected_flat_anchors
                ] = 1.0

                master_quality[
                    selected_groups
                ] = flat_quality[
                    selected_positions
                ].clamp_min(
                    1e-8
                )

                selected_master_count += int(
                    selected_positions.numel()
                )

            else:
                quality_ratio = (
                    flat_quality[
                        selected_positions
                    ]
                    / master_quality[
                        selected_groups
                    ].clamp_min(
                        1e-8
                    )
                ).clamp(
                    min=0.0,
                    max=1.0,
                )

                terminal_weights = (
                    self.anneal_min
                    + (
                        1.0
                        - self.anneal_min
                    )
                    * quality_ratio.pow(
                        self.quality_gamma
                    )
                )

                soft_weights = terminal_weights.to(
                    flat_anchor_weights.dtype
                )

                flat_anchor_weights[
                    selected_flat_anchors
                ] = soft_weights

                selected_soft_count += int(
                    selected_positions.numel()
                )

                soft_weight_sum += float(
                    soft_weights
                    .detach()
                    .float()
                    .sum()
                    .item()
                )

            remaining_quality[
                selected_positions
            ] = -torch.inf

        selected_count = (
            selected_master_count
            + selected_soft_count
        )

        diagnostics = {
            "distinct_strength": strength,
            "dense_retention": dense_retention,
            "candidate_count": float(
                candidate_count
            ),
            "group_count": float(
                group_count
            ),
            "master_count": float(
                selected_master_count
            ),
            "soft_count": float(
                selected_soft_count
            ),
            "mean_soft_weight": (
                soft_weight_sum
                / max(
                    selected_soft_count,
                    1,
                )
            ),
            "selected_candidate_ratio": (
                selected_count
                / max(
                    candidate_count,
                    1,
                )
            ),
        }

        return (
            anchor_weights,
            diagnostics,
        )

    def _accumulate_diagnostics(
        self,
        diagnostics: dict[str, float],
    ) -> None:
        self._epoch_batches += 1
        self._epoch_candidate_count += (
            diagnostics[
                "candidate_count"
            ]
        )
        self._epoch_group_count += (
            diagnostics[
                "group_count"
            ]
        )
        self._epoch_master_count += (
            diagnostics[
                "master_count"
            ]
        )
        self._epoch_soft_count += (
            diagnostics[
                "soft_count"
            ]
        )
        self._epoch_soft_weight_sum += (
            diagnostics[
                "mean_soft_weight"
            ]
            * diagnostics[
                "soft_count"
            ]
        )
        self._epoch_selected_ratio_sum += (
            diagnostics[
                "selected_candidate_ratio"
            ]
        )

    def state(
        self,
    ) -> dict[str, float | str]:
        """返回最近一个完整epoch的CADA状态。"""

        return dict(
            self.last_epoch_state
        )

    def update(
        self,
    ) -> None:
        """
        每个epoch结束后由Ultralytics BaseTrainer自动调用。

        本地源码中的训练循环会检查criterion是否具有update方法，
        因此不需要修改Trainer主循环。
        """

        batch_count = max(
            self._epoch_batches,
            1,
        )

        mean_soft_weight = (
            self._epoch_soft_weight_sum
            / max(
                self._epoch_soft_count,
                1.0,
            )
        )

        self.last_epoch_state = {
            "variant": self.variant,
            "epoch": float(
                self.current_epoch + 1
            ),
            "distinct_strength": float(
                self.distinct_strength()
            ),
            "dense_retention": float(
                1.0
                - self.distinct_strength()
            ),
            "candidate_count": (
                self._epoch_candidate_count
                / batch_count
            ),
            "group_count": (
                self._epoch_group_count
                / batch_count
            ),
            "master_count": (
                self._epoch_master_count
                / batch_count
            ),
            "soft_count": (
                self._epoch_soft_count
                / batch_count
            ),
            "mean_soft_weight": float(
                mean_soft_weight
            ),
            "selected_candidate_ratio": (
                self._epoch_selected_ratio_sum
                / batch_count
            ),
        }

        should_log = (
            self.current_epoch == 0
            or (
                self.current_epoch + 1
            ) % 10
            == 0
            or (
                self.current_epoch + 1
            )
            >= self.total_epochs
        )

        if (
            should_log
            and RANK in (-1, 0)
        ):
            LOGGER.info(
                "[CADA-v1] "
                f"epoch="
                f"{self.current_epoch + 1}/"
                f"{self.total_epochs}, "
                f"distinct="
                f"{self.last_epoch_state['distinct_strength']:.4f}, "
                f"dense="
                f"{self.last_epoch_state['dense_retention']:.4f}, "
                f"候选/批="
                f"{self.last_epoch_state['candidate_count']:.1f}, "
                f"GT组/批="
                f"{self.last_epoch_state['group_count']:.1f}, "
                f"主候选/批="
                f"{self.last_epoch_state['master_count']:.1f}, "
                f"软候选/批="
                f"{self.last_epoch_state['soft_count']:.1f}, "
                f"软权重均值="
                f"{self.last_epoch_state['mean_soft_weight']:.4f}, "
                f"保留比例="
                f"{self.last_epoch_state['selected_candidate_ratio']:.4f}"
            )

        self.current_epoch += 1

        self._epoch_batches = 0
        self._epoch_candidate_count = 0.0
        self._epoch_group_count = 0.0
        self._epoch_master_count = 0.0
        self._epoch_soft_count = 0.0
        self._epoch_soft_weight_sum = 0.0
        self._epoch_selected_ratio_sum = 0.0

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple:
        """
        复用YOLOv8原始TAL、CIoU和DFL，仅替换分类target_scores。
        """

        loss = torch.zeros(
            3,
            device=self.device,
        )

        pred_distri, pred_scores = (
            preds["boxes"]
            .permute(
                0,
                2,
                1,
            )
            .contiguous(),
            preds["scores"]
            .permute(
                0,
                2,
                1,
            )
            .contiguous(),
        )

        anchor_points, stride_tensor = (
            make_anchors(
                preds["feats"],
                self.stride,
                0.5,
            )
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = (
            torch.tensor(
                preds["feats"][0]
                .shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0]
        )

        targets = torch.cat(
            (
                batch["batch_idx"]
                .view(
                    -1,
                    1,
                ),
                batch["cls"]
                .view(
                    -1,
                    1,
                ),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = self.preprocess(
            targets.to(
                self.device
            ),
            batch_size,
            scale_tensor=imgsz[
                [
                    1,
                    0,
                    1,
                    0,
                ]
            ],
        )

        gt_labels, gt_bboxes = (
            targets.split(
                (
                    1,
                    4,
                ),
                dim=2,
            )
        )

        mask_gt = (
            gt_bboxes
            .sum(
                2,
                keepdim=True,
            )
            .gt_(
                0.0
            )
        )

        pred_bboxes = self.bbox_decode(
            anchor_points,
            pred_distri,
        )

        (
            _,
            target_bboxes,
            target_scores,
            fg_mask,
            target_gt_idx,
        ) = self.assigner(
            pred_scores
            .detach()
            .sigmoid(),
            (
                pred_bboxes.detach()
                * stride_tensor
            ).type(
                gt_bboxes.dtype
            ),
            (
                anchor_points
                * stride_tensor
            ),
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # 原始TAL归一化项仅用于CIoU和DFL。分类损失使用重加权后
        # 的目标质量和单独归一化，避免减少正样本后分类损失整体塌缩。
        target_scores_sum = max(
            target_scores.sum(),
            1,
        )

        (
            anchor_weights,
            diagnostics,
        ) = (
            self._build_classification_weights(
                target_scores=target_scores,
                fg_mask=fg_mask,
                target_gt_idx=(
                    target_gt_idx
                ),
                features=preds[
                    "feats"
                ],
            )
        )

        cada_target_scores = (
            target_scores
            * anchor_weights
            .unsqueeze(-1)
            .to(
                target_scores.dtype
            )
        )

        self._accumulate_diagnostics(
            diagnostics
        )

        cada_target_scores_sum = max(
            cada_target_scores.sum(),
            1,
        )

        loss[1] = (
            self.bce(
                pred_scores,
                cada_target_scores.to(
                    dtype
                ),
            )
            .sum()
            / cada_target_scores_sum
        )

        if fg_mask.sum():
            loss[0], loss[2] = (
                self.bbox_loss(
                    pred_distri,
                    pred_bboxes,
                    anchor_points,
                    target_bboxes
                    / stride_tensor,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                    imgsz,
                    stride_tensor,
                )
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return (
            (
                fg_mask,
                target_gt_idx,
                target_bboxes,
                anchor_points,
                stride_tensor,
            ),
            loss,
            loss.detach(),
        )


class CADADetectionModel(DetectionModel):
    """YOLOv8检测模型＋CADA训练期标签分配。"""

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
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
        )

        self.cada_variant = str(
            cada_variant
        ).strip().lower()
        self.cada_soft_topk = int(
            cada_soft_topk
        )
        self.cada_quality_gamma = float(
            cada_quality_gamma
        )
        self.cada_anneal_min = float(
            cada_anneal_min
        )
        self.cada_level_count = int(
            cada_level_count
        )

    def init_criterion(
        self,
    ) -> CADADetectionLoss:
        return CADADetectionLoss(
            model=self,
            variant=self.cada_variant,
            soft_topk=(
                self.cada_soft_topk
            ),
            quality_gamma=(
                self.cada_quality_gamma
            ),
            anneal_min=(
                self.cada_anneal_min
            ),
            level_count=(
                self.cada_level_count
            ),
        )


class CADATrainer(DetectionTrainer):
    """为DDP安全创建CADADetectionModel。"""

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: Any = None,
        verbose: bool = True,
    ) -> CADADetectionModel:
        variant = _read_env_str(
            "CADA_VARIANT",
            DEFAULT_VARIANT,
        ).lower()

        model = CADADetectionModel(
            cfg=cfg,
            ch=int(
                self.data.get(
                    "channels",
                    3,
                )
            ),
            nc=int(
                self.data["nc"]
            ),
            verbose=(
                verbose
                and RANK == -1
            ),
            cada_variant=variant,
            cada_soft_topk=(
                _read_env_int(
                    "CADA_SOFT_TOPK",
                    DEFAULT_SOFT_TOPK,
                )
            ),
            cada_quality_gamma=(
                _read_env_float(
                    "CADA_QUALITY_GAMMA",
                    DEFAULT_QUALITY_GAMMA,
                )
            ),
            cada_anneal_min=(
                _read_env_float(
                    "CADA_ANNEAL_MIN",
                    DEFAULT_ANNEAL_MIN,
                )
            ),
            cada_level_count=(
                _read_env_int(
                    "CADA_LEVEL_COUNT",
                    DEFAULT_LEVEL_COUNT,
                )
            ),
        )

        if weights:
            model.load(
                weights
            )

        return model


try:
    torch.serialization.add_safe_globals(
        [
            CADADetectionLoss,
            CADADetectionModel,
            CADATrainer,
        ]
    )
except Exception:
    pass


__all__ = [
    "CADADetectionLoss",
    "CADADetectionModel",
    "CADATrainer",
]
