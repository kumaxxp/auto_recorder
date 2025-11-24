"""Segmentation utilities: fisheye undistortion + SLIC superpixels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.segmentation import find_boundaries, slic
from skimage.util import img_as_float


@dataclass
class SegmentationResult:
    frame: np.ndarray
    segments: np.ndarray
    boundaries: np.ndarray
    mask_overlay: Optional[np.ndarray] = None
    mask_classes: Optional[np.ndarray] = None
    composited_frame: Optional[np.ndarray] = None


@dataclass
class BottomROIMetrics:
    drivable_ratio: float
    left_right_diff: float
    decision: str
    geometry_confidence: float = 1.0
    geometry_active: bool = False


@dataclass
class FrontObstacleMetrics:
    obstacle_ratio: float
    detected: bool


@dataclass
class GeometryAnalysis:
    floor_mask: Optional[np.ndarray]
    wall_mask: Optional[np.ndarray]
    confidence: float
    wall_ratio: float


CITYSCAPES_CLASS_NAME_BY_ID = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "void",
    20: "background",
}

CITYSCAPES_NAME_TO_ID = {name: idx for idx, name in CITYSCAPES_CLASS_NAME_BY_ID.items()}

DRIVABLE_CLASS_NAMES = ("road", "sidewalk")
CITYSCAPES_DRIVABLE_CLASS_IDS = tuple(
    CITYSCAPES_NAME_TO_ID[name]
    for name in DRIVABLE_CLASS_NAMES
    if name in CITYSCAPES_NAME_TO_ID
)

OBSTACLE_CLASS_IDS = tuple(
    CITYSCAPES_NAME_TO_ID[name]
    for name in ("car", "person", "wall", "building", "truck", "bus")
    if name in CITYSCAPES_NAME_TO_ID
)


@dataclass
class FusionStatus:
    bottom: BottomROIMetrics = field(
        default_factory=lambda: BottomROIMetrics(
            drivable_ratio=0.0,
            left_right_diff=0.0,
            decision="STOP",
        )
    )
    front: FrontObstacleMetrics = field(
        default_factory=lambda: FrontObstacleMetrics(
            obstacle_ratio=0.0,
            detected=False,
        )
    )
    decision: str = "INIT"
    warning: str = ""
    timestamp: float = 0.0
    geometry_confidence: float = 0.0
    geometry_active: bool = False


def process_bottom_mask(
    mask_classes: Optional[np.ndarray],
    roi_y: int,
    drivable_class_ids: Tuple[int, ...],
    go_threshold: float = 0.12,
    avoid_threshold: float = 0.01,
    geometry: Optional[GeometryAnalysis] = None,
) -> BottomROIMetrics:
    if mask_classes is None or mask_classes.size == 0:
        return BottomROIMetrics(drivable_ratio=0.0, left_right_diff=0.0, decision="STOP")
    h = mask_classes.shape[0]
    roi_start = int(np.clip(roi_y, 0, h - 1))
    roi = mask_classes[roi_start:, :]
    if roi.size == 0:
        return BottomROIMetrics(drivable_ratio=0.0, left_right_diff=0.0, decision="STOP")
    drivable_mask = np.isin(roi, drivable_class_ids)
    geometry_confidence = 1.0
    geometry_active = False
    if geometry and geometry.floor_mask is not None:
        floor_mask = geometry.floor_mask
        if floor_mask.shape != drivable_mask.shape:
            floor_mask = cv2.resize(
                floor_mask.astype(np.uint8),
                (drivable_mask.shape[1], drivable_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        drivable_mask = np.logical_and(drivable_mask, floor_mask)
        geometry_confidence = geometry.confidence
        geometry_active = True
    drivable_ratio = float(drivable_mask.mean())
    mid = roi.shape[1] // 2
    left_ratio = float(drivable_mask[:, :mid].mean()) if mid > 0 else drivable_ratio
    right_ratio = float(drivable_mask[:, mid:].mean()) if mid > 0 else drivable_ratio
    left_right_diff = left_ratio - right_ratio
    if drivable_ratio > go_threshold:
        decision = "GO"
    elif drivable_ratio > avoid_threshold:
        decision = "AVOID"
    else:
        decision = "STOP"
    return BottomROIMetrics(
        drivable_ratio=drivable_ratio,
        left_right_diff=left_right_diff,
        decision=decision,
        geometry_confidence=geometry_confidence,
        geometry_active=geometry_active,
    )


def process_front_mask(
    mask_classes: Optional[np.ndarray],
    central_width_ratio: float = 0.25,
    obstacle_class_ids: Tuple[int, ...] = OBSTACLE_CLASS_IDS,
    detection_threshold: float = 0.15,
) -> FrontObstacleMetrics:
    if mask_classes is None or mask_classes.size == 0:
        return FrontObstacleMetrics(obstacle_ratio=0.0, detected=False)
    h, w = mask_classes.shape
    roi_width = max(1, int(w * central_width_ratio))
    start_x = max(0, (w - roi_width) // 2)
    end_x = min(w, start_x + roi_width)
    roi = mask_classes[:, start_x:end_x]
    if roi.size == 0 or not obstacle_class_ids:
        return FrontObstacleMetrics(obstacle_ratio=0.0, detected=False)
    obstacle_mask = np.isin(roi, obstacle_class_ids)
    ratio = float(obstacle_mask.mean())
    return FrontObstacleMetrics(obstacle_ratio=ratio, detected=ratio > detection_threshold)


def compute_floor_wall_geometry(
    frame: Optional[np.ndarray],
    roi_y: int,
    floor_threshold: float = 0.08,
    wall_threshold: float = 0.18,
    blur_kernel: int = 5,
) -> GeometryAnalysis:
    if frame is None or frame.size == 0:
        return GeometryAnalysis(floor_mask=None, wall_mask=None, confidence=0.0, wall_ratio=0.0)
    h = frame.shape[0]
    roi_start = int(np.clip(roi_y, 0, h - 1))
    roi = frame[roi_start:, :]
    if roi.size == 0:
        return GeometryAnalysis(floor_mask=None, wall_mask=None, confidence=0.0, wall_ratio=0.0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if blur_kernel >= 3:
        ksize = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mag.size == 0:
        return GeometryAnalysis(floor_mask=None, wall_mask=None, confidence=0.0, wall_ratio=0.0)
    mag_norm = np.zeros_like(mag)
    if float(mag.max()) > 1e-6:
        cv2.normalize(mag, mag_norm, 0.0, 1.0, cv2.NORM_MINMAX)
    floor_mask = mag_norm <= floor_threshold
    wall_mask = mag_norm >= wall_threshold
    confidence = float(floor_mask.mean()) if floor_mask.size else 0.0
    wall_ratio = float(wall_mask.mean()) if wall_mask.size else 0.0
    return GeometryAnalysis(
        floor_mask=floor_mask,
        wall_mask=wall_mask,
        confidence=confidence,
        wall_ratio=wall_ratio,
    )



class SegmentationProcessor:
    """Applies fisheye correction and computes SLIC superpixels."""

    def __init__(
        self,
        downscale_factor: float = 0.5,
        slic_compactness: float = 12.0,
        slic_sigma: float = 1.0,
        segmentation_color_tolerance: int = 25,
        mask_drivable_class_id: int = 1,
    ) -> None:
        self.camera_matrix = np.array(
            [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self.dist_coeffs = np.array([[-0.12], [0.04], [-0.01], [0.0]], dtype=np.float32)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self.downscale_factor = downscale_factor
        self.slic_compactness = slic_compactness
        self.slic_sigma = slic_sigma
        self.segmentation_color_tolerance = segmentation_color_tolerance
        self.mask_drivable_class_id = mask_drivable_class_id
        self._color_lut_cache: Dict[Tuple[Tuple[int, Tuple[int, int, int]], ...], np.ndarray] = {}

    def _ensure_maps(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._frame_size == (h, w):
            return
        identity = np.eye(3, dtype=np.float32)
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            identity,
            self.camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )
        self._frame_size = (h, w)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_maps(frame)
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def run(
        self,
        frame: np.ndarray,
        n_segments: int,
        mask: Optional[np.ndarray] = None,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> SegmentationResult:
        undistorted = self.undistort(frame)
        if mask is not None and color_map:
            target_mask = mask
            if mask.shape[:2] != undistorted.shape[:2]:
                target_mask = cv2.resize(
                    mask,
                    (undistorted.shape[1], undistorted.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            segments = self._mask_to_segments(target_mask, color_map)
            boundaries = find_boundaries(segments, mode="inner")
            return SegmentationResult(
                frame=undistorted,
                segments=segments,
                boundaries=boundaries,
                mask_overlay=target_mask,
                mask_classes=segments,
            )
        proc_frame = undistorted
        scale = np.clip(self.downscale_factor, 0.1, 1.0)
        if scale < 1.0:
            new_w = max(1, int(undistorted.shape[1] * scale))
            new_h = max(1, int(undistorted.shape[0] * scale))
            proc_frame = cv2.resize(undistorted, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        segments_small = slic(
            img_as_float(rgb),
            n_segments=max(10, int(n_segments)),
            compactness=self.slic_compactness,
            sigma=self.slic_sigma,
            start_label=1,
        )
        if proc_frame is not undistorted:
            segments = cv2.resize(
                segments_small.astype(np.int32),
                (undistorted.shape[1], undistorted.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            segments = segments_small
        boundaries = find_boundaries(segments, mode="inner")
        return SegmentationResult(frame=undistorted, segments=segments, boundaries=boundaries)

    def _mask_to_segments(
        self, mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        lut = self._lookup_color_lut(color_map)
        mask_b = mask[..., 0].astype(np.uint32)
        mask_g = mask[..., 1].astype(np.uint32)
        mask_r = mask[..., 2].astype(np.uint32)
        codes = mask_b | (mask_g << 8) | (mask_r << 16)
        segments = lut[codes].astype(np.int32)
        return segments

    def _lookup_color_lut(
        self, color_map: Dict[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        key = tuple(sorted(color_map.items()))
        lut = self._color_lut_cache.get(key)
        if lut is not None:
            return lut
        size = 1 << 24  # 24-bit BGR lookup
        lut = np.zeros(size, dtype=np.uint8)
        for label, (b, g, r) in color_map.items():
            code = (int(b) & 0xFF) | ((int(g) & 0xFF) << 8) | ((int(r) & 0xFF) << 16)
            lut[code] = np.uint8(label)
        self._color_lut_cache[key] = lut
        return lut
