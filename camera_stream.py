"""Camera capture helper for 640x480 USB camera streams."""

from __future__ import annotations

import platform
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:  # Optional JetCam backend for Jetson CSI cameras
    from jetcam.csi_camera import CSICamera  # type: ignore
except ImportError:  # pragma: no cover - JetCam not required off Jetson
    CSICamera = None  # type: ignore


class CameraStream:
    """Continuously grabs frames on a background thread.

    The constructor accepts an OpenCV camera index; if your USB/CSI camera appears
    on a different `/dev/video*` entry, pass that index when instantiating the
    class. The backend chooser mirrors the reference notebook by preferring
    Jetson's ``nvarguscamerasrc`` GStreamer pipeline (tunable via ``sensor_width``/
    ``sensor_height``/``flip_method``), then V4L2 on Linux and DirectShow on
    Windows, falling back to the generic API when needed.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        prefer_jetson_pipeline: bool = True,
        sensor_width: Optional[int] = None,
        sensor_height: Optional[int] = None,
        flip_method: int = 0,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.prefer_jetson_pipeline = prefer_jetson_pipeline
        self.sensor_width = sensor_width or 1280
        self.sensor_height = sensor_height or 720
        self.flip_method = flip_method
        self._using_gstreamer = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._starting = False
        self._jetcam: Optional[CSICamera] = None  # type: ignore[valid-type]
        self._using_jetcam = False

    def _is_jetson(self) -> bool:
        return (
            sys.platform.startswith("linux")
            and platform.machine() == "aarch64"
            and Path("/usr/lib/aarch64-linux-gnu/tegra").exists()
        )

    def _gstreamer_pipeline(self) -> str:
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        ) % (
            self.index,
            self.sensor_width,
            self.sensor_height,
            self.fps,
            self.flip_method,
            self.width,
            self.height,
        )

    def _open_with_gstreamer(self) -> Optional[cv2.VideoCapture]:
        try:
            capture = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        except cv2.error:
            return None
        if capture.isOpened():
            return capture
        capture.release()
        return None

    def _select_backends(self) -> list[Optional[int]]:
        """Choose capture API backends suited for the current platform."""
        if sys.platform.startswith("linux"):
            return [cv2.CAP_V4L2, cv2.CAP_ANY, None]
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY, None]
        return [cv2.CAP_ANY, None]

    def _open_capture(self) -> cv2.VideoCapture:
        errors: list[str] = []
        if self.prefer_jetson_pipeline and self._is_jetson():
            capture = self._open_with_gstreamer()
            if capture:
                self._using_gstreamer = True
                return capture
            errors.append("gstreamer(nvarguscamerasrc)")
        for backend in self._select_backends():
            capture = (
                cv2.VideoCapture(self.index, backend)
                if backend is not None
                else cv2.VideoCapture(self.index)
            )
            if capture.isOpened():
                self._using_gstreamer = False
                return capture
            errors.append(f"backend={backend}")
            capture.release()
        detail = ", ".join(errors) if errors else "no backend tried"
        raise RuntimeError(
            f"Unable to open camera index {self.index} (attempted {detail})"
        )

    def _should_use_jetcam(self) -> bool:
        return (
            self.prefer_jetson_pipeline
            and self._is_jetson()
            and CSICamera is not None
        )

    def start(self) -> None:
        if self._running:
            return
        with self._start_lock:
            if self._running or self._starting:
                return
            self._starting = True
            try:
                if self._should_use_jetcam():
                    self._start_with_jetcam()
                else:
                    self._start_with_opencv()
            finally:
                self._starting = False

    def _start_with_jetcam(self) -> None:
        self._jetcam = CSICamera(  # type: ignore[operator]
            capture_device=self.index,
            capture_width=self.sensor_width,
            capture_height=self.sensor_height,
            capture_fps=self.fps,
            width=self.width,
            height=self.height,
        )
        self._using_jetcam = True
        self._running = True
        self._thread = threading.Thread(target=self._loop_jetcam, daemon=True)
        self._thread.start()

    def _start_with_opencv(self) -> None:
        self._capture = self._open_capture()
        if not self._using_gstreamer:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._running = True
        self._thread = threading.Thread(target=self._loop_cv, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._using_jetcam and self._jetcam:
            try:
                self._jetcam.cap.release()  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - defensive
                pass
            self._jetcam = None
            self._using_jetcam = False
        if self._capture:
            self._capture.release()
            self._capture = None

    def _loop_cv(self) -> None:
        while self._running and self._capture:
            ok, frame = self._capture.read()
            if ok:
                resized = cv2.resize(frame, (self.width, self.height))
                with self._lock:
                    self._frame = resized
            else:
                time.sleep(0.05)

    def _loop_jetcam(self) -> None:
        while self._running and self._jetcam:
            try:
                frame = self._jetcam.read()
            except RuntimeError:
                time.sleep(0.05)
                continue
            resized = cv2.resize(frame, (self.width, self.height))
            with self._lock:
                self._frame = resized

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_running(self) -> bool:
        return self._running or self._starting

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()
