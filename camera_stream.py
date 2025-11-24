"""Camera capture helper for 640x480 USB camera streams."""

from __future__ import annotations

import asyncio
import platform
import shlex
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from fastapi import WebSocket, WebSocketDisconnect

import logging

logger = logging.getLogger(__name__)


class FramePublisher:
    """Stores the latest 3-channel frame (BGR) and serves it over WebSockets."""

    CHANNELS = 3

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._latest_frame: Optional[bytes] = None
        self._width = 0
        self._height = 0
        self._serial = 0

    def publish(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != self.CHANNELS:
            raise ValueError("FramePublisher expects RGB frames with 3 channels")
        contiguous = np.ascontiguousarray(frame)
        payload = contiguous.tobytes()
        with self._lock:
            self._latest_frame = payload
            self._height, self._width = frame.shape[:2]
            self._serial += 1
            self._lock.notify_all()

    def _wait_for_frame(self, last_serial: int, timeout: float = 1.0) -> Optional[Tuple[int, int, int, bytes]]:
        with self._lock:
            got_new = self._lock.wait_for(lambda: self._serial != last_serial and self._latest_frame is not None, timeout=timeout)
            if not got_new or self._latest_frame is None:
                return None
            return self._serial, self._width, self._height, self._latest_frame

    async def next_frame(self, last_serial: int) -> Optional[Tuple[int, int, int, bytes]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._wait_for_frame, last_serial)

    async def stream(self, websocket: WebSocket) -> None:
        await websocket.accept()
        last_serial = 0
        try:
            while True:
                snapshot = await self.next_frame(last_serial)
                if snapshot is None:
                    continue
                serial, width, height, data = snapshot
                last_serial = serial
                header = struct.pack("!III", width, height, self.CHANNELS)
                await websocket.send_bytes(header + data)
        except WebSocketDisconnect:
            return


class MultiFramePublisher:
    """Manages a set of named FramePublisher instances."""

    def __init__(self) -> None:
        self._publishers: Dict[str, FramePublisher] = {}
        self._lock = threading.Lock()

    def get_or_create(self, name: str) -> FramePublisher:
        with self._lock:
            publisher = self._publishers.get(name)
            if publisher is None:
                publisher = FramePublisher()
                self._publishers[name] = publisher
            return publisher

    def publish(self, name: str, frame: np.ndarray) -> None:
        self.get_or_create(name).publish(frame)

    async def stream(self, name: str, websocket: WebSocket) -> None:
        publisher = self.get_or_create(name)
        await publisher.stream(websocket)

try:  # Optional JetCam backend for Jetson CSI cameras
    from jetcam.csi_camera import CSICamera  # type: ignore
except ImportError:  # pragma: no cover - JetCam not required off Jetson
    CSICamera = None  # type: ignore

try:  # Optional GStreamer bindings (DeepStream pipeline)
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
except Exception:  # pragma: no cover - optional dependency
    Gst = None  # type: ignore


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
        use_gstreamer_segmentation: bool = False,
        gstreamer_segmentation_config: Optional[str] = None,
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
        self._gst_backend: Optional[_GStreamerSegmentationBackend] = None
        self._using_gst_segmentation = False
        self._mask: Optional[np.ndarray] = None
        self._class_mask: Optional[np.ndarray] = None
        config_path = (
            Path(gstreamer_segmentation_config).expanduser()
            if gstreamer_segmentation_config
            else None
        )
        self._gst_config = config_path if config_path and config_path.exists() else None
        self._gst_color_map = (
            self._parse_segmentation_colors(self._gst_config)
            if self._gst_config
            else {}
        )
        self._gst_plugins_ready = self._check_gst_plugins() if Gst is not None else False
        self._use_gst_segmentation_flag = (
            use_gstreamer_segmentation and self._gst_config is not None
        )

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

    def _can_use_gst_segmentation(self) -> bool:
        return bool(
            self._use_gst_segmentation_flag and Gst is not None and self._gst_plugins_ready
        )

    def start(self) -> None:
        if self._running:
            return
        with self._start_lock:
            if self._running or self._starting:
                return
            self._starting = True
            try:
                if self._can_use_gst_segmentation():
                    try:
                        self._start_with_gstreamer_segmentation()
                        return
                    except Exception:
                        logger.exception(
                            "Failed to start GStreamer segmentation backend; falling back"
                        )
                if self._should_use_jetcam():
                    self._start_with_jetcam()
                else:
                    self._start_with_opencv()
            finally:
                self._starting = False

    def _start_with_gstreamer_segmentation(self) -> None:
        if not self._gst_config:
            raise RuntimeError("GStreamer segmentation config file not found")
        logger.info("Starting GStreamer segmentation backend with %s", self._gst_config)
        self._gst_backend = _GStreamerSegmentationBackend(
            sensor_id=self.index,
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            width=self.width,
            height=self.height,
            fps=self.fps,
            flip_method=self.flip_method,
            config_path=self._gst_config,
        )
        self._gst_backend.start()
        self._using_gst_segmentation = True
        self._running = True
        self._thread = threading.Thread(
            target=self._loop_gst_segmentation, daemon=True
        )
        self._thread.start()

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
        if self._using_gst_segmentation and self._gst_backend:
            logger.info("Stopping GStreamer segmentation backend")
            self._gst_backend.stop()
            self._gst_backend = None
            self._using_gst_segmentation = False
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

    def _loop_gst_segmentation(self) -> None:
        while self._running and self._gst_backend:
            payload = self._gst_backend.pull_frame(timeout=1.0)
            if payload is None:
                continue
            frame, overlay, class_mask = payload
            resized = cv2.resize(frame, (self.width, self.height))
            overlay_resized = None
            if overlay is not None:
                overlay_resized = cv2.resize(
                    overlay, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )
            class_resized = None
            if class_mask is not None:
                class_resized = cv2.resize(
                    class_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
                )
            with self._lock:
                self._frame = resized
                self._mask = overlay_resized
                self._class_mask = class_resized

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_mask(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._mask is None:
                return None
            return self._mask.copy()

    def read_mask_classes(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._class_mask is None:
                return None
            return self._class_mask.copy()

    def get_segmentation_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        return dict(self._gst_color_map)

    def is_running(self) -> bool:
        return self._running or self._starting

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()

    @staticmethod
    def _parse_segmentation_colors(config_path: Path) -> Dict[int, Tuple[int, int, int]]:
        colors: Dict[int, Tuple[int, int, int]] = {}
        try:
            for raw_line in config_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "overlay-color" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key.startswith("overlay-color"):
                    continue
                try:
                    index = int(key.replace("overlay-color", ""))
                except ValueError:
                    continue
                parts = [p.strip() for p in value.split(";")]
                if len(parts) < 3:
                    continue
                try:
                    r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
                except ValueError:
                    continue
                colors[index] = (b, g, r)  # convert to BGR
        except OSError:
            return {}
        return colors

    @staticmethod
    def _check_gst_plugins() -> bool:
        if Gst is None:
            return False
        required = ("nvarguscamerasrc", "nvvidconv", "nvinfer", "nvsegvisual")
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        if missing:
            logger.warning(
                "Missing required GStreamer/DeepStream plugins: %s", ", ".join(missing)
            )
            return False
        return True


class DualDeepStreamPipelines:
    """Bootstraps two independent DeepStream segmentation pipelines.

    This helper guarantees that bottom/front cameras each receive their own
    ``nvinfer`` instance while sharing the same configuration file. It can be
    used by higher layers to start/stop both feeds together and exposes the
    underlying ``CameraStream`` objects via ``pipeline_bottom`` and
    ``pipeline_front`` attributes.
    """

    def __init__(
        self,
        bottom_index: int,
        front_index: int,
        config_path: Path,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        sensor_width: Optional[int] = None,
        sensor_height: Optional[int] = None,
        flip_method: int = 0,
    ) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"DeepStream config not found: {config_path}")
        shared_kwargs = dict(
            width=width,
            height=height,
            fps=fps,
            prefer_jetson_pipeline=True,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            flip_method=flip_method,
            use_gstreamer_segmentation=True,
            gstreamer_segmentation_config=str(config_path),
        )
        self.pipeline_bottom = CameraStream(index=bottom_index, **shared_kwargs)
        self.pipeline_front = CameraStream(index=front_index, **shared_kwargs)

    def start(self) -> None:
        if not self.pipeline_bottom.is_running():
            self.pipeline_bottom.start()
        if not self.pipeline_front.is_running():
            self.pipeline_front.start()

    def stop(self) -> None:
        self.pipeline_bottom.stop()
        self.pipeline_front.stop()

    def is_running(self) -> bool:
        return self.pipeline_bottom.is_running() and self.pipeline_front.is_running()

class _GStreamerSegmentationBackend:
    """Runs a DeepStream-powered segmentation pipeline and surfaces frames."""

    def __init__(
        self,
        sensor_id: int,
        sensor_width: int,
        sensor_height: int,
        width: int,
        height: int,
        fps: int,
        flip_method: int,
        config_path: Path,
    ) -> None:
        if Gst is None:  # pragma: no cover - optional dependency
            raise RuntimeError("GStreamer bindings are not available")
        self._width = width
        self._height = height
        self._pipeline = self._create_pipeline(
            sensor_id=sensor_id,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            fps=fps,
            flip_method=flip_method,
            config_path=config_path,
        )
        masksink = self._pipeline.get_by_name("masksink")
        classsink = self._pipeline.get_by_name("classsink")
        rgbsink = self._pipeline.get_by_name("rgbsink")
        if not masksink or not rgbsink:
            raise RuntimeError("Failed to find appsinks in GStreamer pipeline")
        for sink in (masksink, rgbsink, classsink):
            if sink is None:
                continue
            sink.set_property("emit-signals", False)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
        self._mask_sink = masksink
        self._class_sink = classsink
        self._rgb_sink = rgbsink

    def _create_pipeline(
        self,
        sensor_id: int,
        sensor_width: int,
        sensor_height: int,
        fps: int,
        flip_method: int,
        config_path: Path,
    ) -> "Gst.Pipeline":
        config = shlex.quote(str(config_path.resolve()))
        # Use class dimensions for processing to ensure consistent sizing
        proc_width = self._width
        proc_height = self._height
        # Use model dimensions for segmentation to ensure full coverage
        seg_width = 512
        seg_height = 256
        
        pipeline_desc = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={sensor_width}, height={sensor_height}, framerate={fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! video/x-raw(memory:NVMM), width={proc_width}, height={proc_height}, format=NV12 ! "
            "tee name=rawtee "
            "rawtee. ! queue ! nvvidconv ! video/x-raw, format=RGBA ! videoconvert ! video/x-raw, format=BGR ! appsink name=rgbsink "
            "rawtee. ! queue ! mux.sink_0 "
            f"nvstreammux name=mux width={seg_width} height={seg_height} batch-size=1 live-source=1 enable-padding=0 ! "
                        f"nvinfer batch-size=1 unique-id=1 config-file-path={config} ! "
            f"nvsegvisual width={seg_width} height={seg_height} ! "
            "nvvidconv ! video/x-raw(memory:NVMM), format=RGBA ! nvvidconv ! video/x-raw, format=RGBA ! appsink name=masksink"
        )
        # Debug: print pipeline string
        # print(f"DEBUG: Pipeline string: {pipeline_desc}")
        pipeline = Gst.parse_launch(pipeline_desc)
        return pipeline

    def start(self) -> None:
        self._pipeline.set_state(Gst.State.PLAYING)

    def pull_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        mask_sample = self._mask_sink.emit("try-pull-sample", int(timeout * Gst.SECOND)) if self._mask_sink else None
        class_sample = (
            self._class_sink.emit("try-pull-sample", int(timeout * Gst.SECOND))
            if self._class_sink
            else None
        )
        rgb_sample = self._rgb_sink.emit("try-pull-sample", int(timeout * Gst.SECOND)) if self._rgb_sink else None

        if rgb_sample is None:
            return None

        frame = self._sample_to_ndarray(rgb_sample, channels=3)
        mask = None
        if mask_sample is not None:
            mask = self._sample_to_ndarray(mask_sample)
        class_map = None
        if class_sample is not None:
            class_map = self._sample_to_ndarray(class_sample, channels=1)
        return frame, mask, class_map

    @staticmethod
    def _sample_to_ndarray(sample, channels: Optional[int] = None) -> np.ndarray:
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        fmt = structure.get_value("format")
        if channels is None:
            if fmt in {"RGBA", "BGRA"}:
                channels = 4
            elif fmt in {"GRAY8", "GRAY16_LE", "GRAY16_BE"}:
                channels = 1
            else:
                channels = 3
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Failed to map GStreamer buffer")
        try:
            if channels == 1:
                array = np.ndarray(
                    shape=(height, width), dtype=np.uint8, buffer=map_info.data
                ).copy()
            else:
                array = np.ndarray(
                    shape=(height, width, channels), dtype=np.uint8, buffer=map_info.data
                ).copy()
        finally:
            buffer.unmap(map_info)
        return array

    def stop(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)
