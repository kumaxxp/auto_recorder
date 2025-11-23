"""NiceGUI composition for the segmentation dashboard."""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np
from nicegui import ui

from camera_stream import CameraStream
from labeling import LABEL_COLORS, LabelManager, SegmentLabel
from segmentation import SegmentationProcessor


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
        self._dragging_roi = False
        self._timer: Optional[ui.timer] = None

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

        self._timer = ui.timer(0.1, self._update_frame)


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
        frame = self.camera.read()
        if frame is None:
            return
        mask = self.camera.read_mask()
        color_map = self.camera.get_segmentation_color_map() if mask is not None else None
        result = self.processor.run(
            frame,
            self.superpixels,
            mask=mask,
            color_map=color_map,
        )
        self._latest_segments = result.segments
        self.frame_height, self.frame_width = result.frame.shape[:2]
        blended = self._compose_overlay(result)
        src = self._to_data_url(blended)
        if self._image_component:
            self._image_component.set_source(src)
        metrics = self.labels.compute_metrics(result.segments, self.roi_y)
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
