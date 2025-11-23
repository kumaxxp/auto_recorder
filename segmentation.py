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

    def __init__(
        self,
        downscale_factor: float = 0.5,
        slic_compactness: float = 12.0,
        slic_sigma: float = 1.0,
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
