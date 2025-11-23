"""NiceGUI composition for the segmentation dashboard."""

from __future__ import annotations

import base64
import threading
import time
from typing import Optional

import cv2
import numpy as np
from nicegui import ui

from camera_stream import CameraStream
from labeling import LABEL_COLORS, LabelManager, Metrics, SegmentLabel
from segmentation import SegmentationProcessor, SegmentationResult

CITYSCAPES_DRIVABLE_CLASS_IDS = (3, 4)  # road, sidewalk


PHASE_A_RAW_PREVIEW = False  # Phase A diagnostics toggle; False restores overlay pipeline
RAW_PREVIEW_FRAME_SKIP = 3  # Only used when Phase A diagnostics are active
UI_REFRESH_INTERVAL_SEC = 1.0 / 30.0  # Target ~30 Hz UI refresh for overlay testing
PROCESSING_SLEEP_SEC = 0.0  # Optional throttle for the background worker


class SegmentationDashboard:
    """High-level controller wiring NiceGUI with processing logic."""

    def __init__(
        self,
        camera: CameraStream,
        processor: SegmentationProcessor,
        labels: LabelManager,
    ) -> None:
        self.camera = camera
        self.processor = processor
        self.labels = labels
        self.superpixels = 120
        self.roi_y = 360
        self.frame_height = 480
        self.frame_width = 640
        self._latest_segments: Optional[np.ndarray] = None
        self._image_component: Optional[ui.interactive_image] = None
        self._debug_label: Optional[ui.label] = None
        self._roi_label: Optional[ui.label] = None
        self._mode_label: Optional[ui.label] = None
        self._slider_label: Optional[ui.label] = None
        self._perf_label: Optional[ui.label] = None
        self._result_lock = threading.Lock()
        self._raw_lock = threading.Lock()
        self._pending_result: Optional[SegmentationResult] = None
        self._pending_serial = 0
        self._rendered_serial = 0
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._raw_serial = 0
        self._consumed_raw_serial = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._worker_fps_counter = 0
        self._worker_fps_start = time.time()
        self._worker_fps_value: Optional[float] = None
        self._dragging_roi = False
        self._timer: Optional[ui.timer] = None
        self._drivable_class_ids = CITYSCAPES_DRIVABLE_CLASS_IDS
        self._frame_counter = 0
        self._fps_counter = 0
        self._fps_window_start = time.time()

    def mount(self) -> None:
        with ui.row().classes("w-full items-start gap-6"):
            self._image_component = ui.interactive_image(
                self._blank_src(),
                size=(float(self.frame_width), float(self.frame_height)),
                on_mouse=self._handle_mouse,
            )
            with ui.column().classes("w-72 gap-3"):
                ui.label("Superpixel granularity")
                self._slider_label = ui.label(self._slider_text())
                ui.slider(
                    min=40,
                    max=400,
                    value=self.superpixels,
                    step=20,
                    on_change=self._on_superpixel_change,
                ).props("label-always")
                ui.label("Labeling mode")
                with ui.row().classes("gap-2"):
                    ui.button("DRIVABLE", color="green", on_click=lambda: self._set_mode(SegmentLabel.DRIVABLE))
                    ui.button("BLOCKED", color="red", on_click=lambda: self._set_mode(SegmentLabel.BLOCKED))
                ui.button("UNKNOWN", color="gray", on_click=lambda: self._set_mode(SegmentLabel.UNKNOWN))
                self._mode_label = ui.label(f"Active: {self.labels.active_label.name}")
                self._roi_label = ui.label(f"ROI cut line: {self.roi_y}px (right-drag to move)")
                self._debug_label = ui.label("Metrics pending...")

        self._perf_label = ui.label("FPS pending...")

        self._start_processing_worker()
        self._timer = ui.timer(UI_REFRESH_INTERVAL_SEC, self._update_frame)


    def _on_superpixel_change(self, event) -> None:
        self.superpixels = int(event.value)
        if self._slider_label:
            self._slider_label.text = self._slider_text()

    def _set_mode(self, label: SegmentLabel) -> None:
        self.labels.set_active_label(label)
        if self._mode_label:
            self._mode_label.text = f"Active: {label.name}"

    def _slider_text(self) -> str:
        return f"Superpixels: {self.superpixels}"

    def _handle_mouse(self, event) -> None:
        if event is None or self._latest_segments is None:
            return
        if event.type == "down" and event.button == "right":
            self._dragging_roi = True
        elif event.type == "up" and event.button == "right":
            self._dragging_roi = False
        elif event.type == "move" and self._dragging_roi:
            self._update_roi(event.image_y)
            return

        if event.type == "click" and event.button == "left":
            x = int(np.clip(event.image_x, 0, self._latest_segments.shape[1] - 1))
            y = int(np.clip(event.image_y, 0, self._latest_segments.shape[0] - 1))
            segment_id = int(self._latest_segments[y, x])
            self.labels.assign_label(segment_id)

    def _update_roi(self, y_value: float) -> None:
        self.roi_y = int(np.clip(y_value, 0, self.frame_height - 1))
        if self._roi_label:
            self._roi_label.text = f"ROI cut line: {self.roi_y}px (right-drag to move)"

    def _update_frame(self) -> None:
        if PHASE_A_RAW_PREVIEW:
            frame = self._consume_raw_frame()
            if frame is None:
                return
            self.frame_height, self.frame_width = frame.shape[:2]
            if self._image_component:
                src = self._to_data_url(frame)
                self._image_component.set_source(src)
            self._report_fps("PhaseA raw preview")
            return

        result = self._consume_latest_result()
        if result is None:
            return
        self.frame_height, self.frame_width = result.frame.shape[:2]
        self._latest_segments = result.mask_classes if result.mask_classes is not None else result.segments
        if result.mask_overlay is not None:
            blended = self._compose_mask_overlay(result)
        else:
            blended = self._compose_overlay(result)
        src = self._to_data_url(blended)
        if self._image_component:
            self._image_component.set_source(src)
        if result.mask_classes is not None:
            metrics = self._compute_mask_metrics(result.mask_classes)
        else:
            metrics = self.labels.compute_metrics(result.segments, self.roi_y)
        self._report_fps("PhaseC display")
        if self._debug_label:
            self._debug_label.text = (
                f"drivable_ratio={metrics.drivable_ratio:.2f} | "
                f"left/right ratio={metrics.left_right_ratio:.2f} | decision={metrics.decision}"
            )

    def _compose_overlay(self, result) -> np.ndarray:
        overlay = self.labels.build_overlay(result.segments)
        blended = cv2.addWeighted(result.frame, 0.7, overlay, 0.3, 0.0)
        blended[result.boundaries] = (255, 255, 255)
        y = int(np.clip(self.roi_y, 0, blended.shape[0] - 1))
        cv2.line(blended, (0, y), (blended.shape[1] - 1, y), (255, 255, 0), 2)
        return blended

    def _compose_mask_overlay(self, result: SegmentationResult) -> np.ndarray:
        # DeepStream already outputs a colorized mask; blend it directly for clarity.
        overlay = result.mask_overlay
        if overlay is None:
            return result.frame
        blended = cv2.addWeighted(result.frame, 0.6, overlay, 0.4, 0.0)
        y = int(np.clip(self.roi_y, 0, blended.shape[0] - 1))
        cv2.line(blended, (0, y), (blended.shape[1] - 1, y), (255, 255, 0), 2)
        return blended

    def _compute_mask_metrics(self, mask_classes: np.ndarray) -> Metrics:
        roi = mask_classes[self.roi_y :, :]
        if roi.size == 0:
            return Metrics(0.0, 0.0, "STOP")
        drive_mask = np.isin(roi, self._drivable_class_ids)
        drivable_ratio = float(drive_mask.mean())
        mid = roi.shape[1] // 2
        left_drive = float(drive_mask[:, :mid].sum())
        right_drive = float(drive_mask[:, mid:].sum())
        right_norm = max(right_drive, 1.0)
        left_right_ratio = left_drive / right_norm
        decision = "GO" if drivable_ratio > 0.10 else "STOP"
        return Metrics(drivable_ratio, left_right_ratio, decision)

    @staticmethod
    def _to_data_url(image: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            return ""
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _blank_src() -> str:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return SegmentationDashboard._to_data_url(blank)

    def _report_fps(self, tag: str) -> None:
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_window_start
        if elapsed < 1.0:
            return
        fps = self._fps_counter / elapsed
        print(f"[{tag}] FPS: {fps:.1f}")
        if self._perf_label:
            label = f"{tag} FPS={fps:.1f}"
            if self._worker_fps_value is not None:
                label = f"{label} | proc={self._worker_fps_value:.1f}"
            self._perf_label.text = label
        self._fps_counter = 0
        self._fps_window_start = now

    def _start_processing_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

    def shutdown(self) -> None:
        self._stop_processing_worker()

    def _stop_processing_worker(self) -> None:
        self._worker_stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def _processing_loop(self) -> None:
        while not self._worker_stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if PHASE_A_RAW_PREVIEW:
                self._publish_raw_frame(frame)
                self._update_worker_fps()
                continue
            mask_overlay = self.camera.read_mask()
            mask_classes = getattr(self.camera, "read_mask_classes", lambda: None)()
            if mask_overlay is not None and mask_classes is not None:
                result = self._build_mask_result(frame, mask_overlay, mask_classes)
            else:
                color_map = self.camera.get_segmentation_color_map() if mask_overlay is not None else None
                result = self.processor.run(
                    frame,
                    self.superpixels,
                    mask=mask_overlay,
                    color_map=color_map,
                )
            self._publish_result(result)
            self._update_worker_fps()
            if PROCESSING_SLEEP_SEC > 0:
                time.sleep(PROCESSING_SLEEP_SEC)

    def _publish_result(self, result: SegmentationResult) -> None:
        with self._result_lock:
            self._pending_result = result
            self._pending_serial += 1

    def _consume_latest_result(self) -> Optional[SegmentationResult]:
        with self._result_lock:
            if self._pending_result is None or self._pending_serial == self._rendered_serial:
                return None
            result = self._pending_result
            self._rendered_serial = self._pending_serial
        return result

    def _publish_raw_frame(self, frame: np.ndarray) -> None:
        with self._raw_lock:
            self._latest_raw_frame = frame.copy()
            self._raw_serial += 1

    def _consume_raw_frame(self) -> Optional[np.ndarray]:
        with self._raw_lock:
            if self._latest_raw_frame is None or self._raw_serial == self._consumed_raw_serial:
                return None
            frame = self._latest_raw_frame.copy()
            self._consumed_raw_serial = self._raw_serial
        return frame

    def _update_worker_fps(self) -> None:
        self._worker_fps_counter += 1
        now = time.time()
        elapsed = now - self._worker_fps_start
        if elapsed < 1.0:
            return
        self._worker_fps_value = self._worker_fps_counter / elapsed
        print(f"[PhaseC worker] FPS: {self._worker_fps_value:.1f}")
        self._worker_fps_counter = 0
        self._worker_fps_start = now

    @staticmethod
    def _build_mask_result(frame: np.ndarray, overlay: np.ndarray, mask_classes: np.ndarray) -> SegmentationResult:
        boundaries = np.zeros_like(mask_classes, dtype=bool)
        return SegmentationResult(
            frame=frame,
            segments=mask_classes,
            boundaries=boundaries,
            mask_overlay=overlay,
            mask_classes=mask_classes,
        )
