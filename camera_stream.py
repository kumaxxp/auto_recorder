"""Camera capture helper for 640x480 USB camera streams."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Continuously grabs frames on a background thread."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            raise RuntimeError("Unable to open camera index %s" % self.index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture:
            self._capture.release()
            self._capture = None

    def _loop(self) -> None:
        while self._running and self._capture:
            ok, frame = self._capture.read()
            if ok:
                resized = cv2.resize(frame, (self.width, self.height))
                with self._lock:
                    self._frame = resized
            else:
                time.sleep(0.05)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()
