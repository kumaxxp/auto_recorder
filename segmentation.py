"""Segmentation utilities: fisheye undistortion + SLIC superpixels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.segmentation import find_boundaries, slic
from skimage.util import img_as_float


@dataclass
class SegmentationResult:
    frame: np.ndarray
    segments: np.ndarray
    boundaries: np.ndarray


class SegmentationProcessor:
    """Applies fisheye correction and computes SLIC superpixels."""

    def __init__(self) -> None:
        self.camera_matrix = np.array(
            [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self.dist_coeffs = np.array([[-0.12], [0.04], [-0.01], [0.0]], dtype=np.float32)
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None

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

    def run(self, frame: np.ndarray, n_segments: int) -> SegmentationResult:
        undistorted = self.undistort(frame)
        rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        segments = slic(img_as_float(rgb), n_segments=int(n_segments), compactness=10.0, start_label=1)
        boundaries = find_boundaries(segments, mode="inner")
        return SegmentationResult(frame=undistorted, segments=segments, boundaries=boundaries)
