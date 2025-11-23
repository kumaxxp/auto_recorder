"""NiceGUI composition for the segmentation dashboard."""

from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np
from nicegui import ui

from camera_stream import CameraStream
from labeling import LABEL_COLORS, LabelManager, Metrics, SegmentLabel
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
        
        # --- Debug logging start ---
        # if mask is not None:
        #     unique_vals = np.unique(mask)
        #     if len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] != 0):
        #         print(f"DEBUG: Mask received. Unique values: {unique_vals}")
        #     # else:
        #     #    print(f"DEBUG: Mask is all zeros (empty).")
        # else:
        #     print("DEBUG: Mask is None")
        # --- Debug logging end ---

        color_map = self.camera.get_segmentation_color_map() if mask is not None else None
        result = self.processor.run(
            frame,
            self.superpixels,
            mask=mask,
            color_map=color_map,
        )
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
        # The mask_overlay from DeepStream is the colored mask.
        # We blend it with the original frame.
        if result.mask_overlay is not None:
            try:
                # Heuristic: The most frequent color is the background.
                # We want to mask everything that is NOT the background.
                
                # Reshape to list of pixels
                pixels = result.mask_overlay.reshape(-1, 3)
                
                # Find unique colors and their counts
                colors, counts = np.unique(pixels, axis=0, return_counts=True)
                
                if len(colors) > 0:
                    # Get the most frequent color (background)
                    bg_color = colors[np.argmax(counts)]
                    
                    # Create a mask for everything that is NOT the background
                    # cv2.inRange is inclusive, so we can't easily do "not equal".
                    # Instead, we create a mask for the background and invert it.
                    lower_bg = np.array(bg_color, dtype=np.uint8)
                    upper_bg = np.array(bg_color, dtype=np.uint8)
                    
                    # Allow small tolerance for compression artifacts
                    tolerance = 5
                    lower_bg = np.clip(lower_bg - tolerance, 0, 255)
                    upper_bg = np.clip(upper_bg + tolerance, 0, 255)
                    
                    bg_mask = cv2.inRange(result.mask_overlay, lower_bg, upper_bg)
                    
                    # Invert mask to get foreground (drivable area + others)
                    fg_mask = cv2.bitwise_not(bg_mask)
                    
                    # Create the output frame
                    blended = result.frame.copy()
                    
                    # Apply green tint to foreground
                    roi = blended[fg_mask > 0]
                    if roi.size > 0:
                        green = np.full_like(roi, (0, 255, 0))
                        blended[fg_mask > 0] = cv2.addWeighted(roi, 0.4, green, 0.6, 0.0)
                    
                    y = int(np.clip(self.roi_y, 0, blended.shape[0] - 1))
                    cv2.line(blended, (0, y), (blended.shape[1] - 1, y), (255, 255, 0), 2)
                    return blended
            except Exception as e:
                print(f"Error in _compose_mask_overlay: {e}")
                return result.frame
            
            return result.frame
            
        # Fallback for SLIC or other modes
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
        drivable_class = self.processor.mask_drivable_class_id
        drive_mask = roi == drivable_class
        drivable_ratio = float(drive_mask.sum()) / float(roi.size)
        mid = roi.shape[1] // 2
        left_drive = float(drive_mask[:, :mid].sum())
        right_drive = float(drive_mask[:, mid:].sum())
        right_norm = max(right_drive, 1.0)
        left_right_ratio = left_drive / right_norm
        if drivable_ratio < 0.1:
            decision = "STOP"
        elif drivable_ratio > 0.6:
            decision = "FORWARD"
        elif left_drive > right_drive:
            decision = "TURN_LEFT"
        elif right_drive > left_drive:
            decision = "TURN_RIGHT"
        else:
            decision = "STOP"
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
