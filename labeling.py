"""Segment labeling helpers and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable

import numpy as np


class SegmentLabel(Enum):
    UNKNOWN = 0
    DRIVABLE = 1
    BLOCKED = 2


LABEL_COLORS = {
    SegmentLabel.UNKNOWN: (0, 0, 0),
    SegmentLabel.DRIVABLE: (0, 200, 0),
    SegmentLabel.BLOCKED: (200, 0, 0),
}


@dataclass
class Metrics:
    drivable_ratio: float
    left_right_ratio: float
    decision: str


class LabelManager:
    """Stores labels per segment and computes ROI metrics."""

    def __init__(self) -> None:
        self.labels: Dict[int, SegmentLabel] = {}
        self.active_label: SegmentLabel = SegmentLabel.DRIVABLE

    def set_active_label(self, label: SegmentLabel) -> None:
        self.active_label = label

    def assign_label(self, segment_id: int, label: SegmentLabel | None = None) -> None:
        target = label or self.active_label
        if target == SegmentLabel.UNKNOWN:
            self.labels.pop(segment_id, None)
        else:
            self.labels[segment_id] = target

    def get_label(self, segment_id: int) -> SegmentLabel:
        return self.labels.get(segment_id, SegmentLabel.UNKNOWN)

    def build_overlay(self, segments: np.ndarray) -> np.ndarray:
        overlay = np.zeros((*segments.shape, 3), dtype=np.uint8)
        for seg_id, label in self.labels.items():
            mask = segments == seg_id
            overlay[mask] = LABEL_COLORS[label]
        return overlay

    def _drivable_ids(self) -> Iterable[int]:
        return (seg_id for seg_id, label in self.labels.items() if label == SegmentLabel.DRIVABLE)

    def compute_metrics(self, segments: np.ndarray, roi_y: int) -> Metrics:
        h, w = segments.shape
        roi_y = int(min(max(0, roi_y), h - 1))
        roi = segments[roi_y:, :]
        if roi.size == 0:
            return Metrics(drivable_ratio=0.0, left_right_ratio=0.0, decision="STOP")

        drivable_mask = np.zeros_like(roi, dtype=bool)
        for seg_id in self._drivable_ids():
            drivable_mask |= roi == seg_id

        drivable_ratio = float(drivable_mask.sum()) / float(roi.size)
        mid = w // 2
        left_drive = float(drivable_mask[:, :mid].sum())
        right_drive = float(drivable_mask[:, mid:].sum())
        right_norm = max(right_drive, 1.0)
        left_right_ratio = left_drive / right_norm

        decision = self._decision(drivable_ratio, left_drive, right_drive)
        return Metrics(drivable_ratio=drivable_ratio, left_right_ratio=left_right_ratio, decision=decision)

    @staticmethod
    def _decision(drivable_ratio: float, left_drive: float, right_drive: float) -> str:
        if drivable_ratio < 0.1:
            return "STOP"
        if drivable_ratio > 0.6:
            return "FORWARD"
        if left_drive > right_drive:
            return "TURN_LEFT"
        if right_drive > left_drive:
            return "TURN_RIGHT"
        return "STOP"
