"""NiceGUI composition for the segmentation dashboard."""

from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from nicegui import ui

from camera_stream import CameraStream, MultiFramePublisher
from labeling import LABEL_COLORS, LabelManager, Metrics, SegmentLabel
from segmentation import SegmentationProcessor, SegmentationResult

CITYSCAPES_DRIVABLE_CLASS_IDS = (3, 4)  # road, sidewalk
ROI_Y_RATIO = 0.35
DRIVABLE_STATUS_THRESHOLDS = (0.12, 0.01)
STATUS_COLORS = {"GO": "#16a34a", "AVOID": "#f97316", "STOP": "#dc2626"}


PHASE_A_RAW_PREVIEW = False  # Phase A diagnostics toggle; False restores overlay pipeline
RAW_PREVIEW_FRAME_SKIP = 3  # Only used when Phase A diagnostics are active
UI_REFRESH_INTERVAL_SEC = 1.0 / 60.0  # Faster timer cadence for Phase D
DISPLAY_TARGET_FPS = 40.0  # Target websocket stream rate (frames per second)
PROCESSING_SLEEP_SEC = 0.0  # Optional throttle for the background worker

STREAM_CLIENT_SCRIPT = """
<script>
(function() {
    if (window.__phaseDStreamReady) {
        return;
    }
    window.__phaseDStreamReady = true;
    window.startStreamCanvas = function(canvasId, endpoint) {
        let socket = null;
        let pending = null;
        let ctx = null;
        function draw(buffer) {
            if (!buffer) {
                return;
            }
            const view = new DataView(buffer);
            const width = view.getUint32(0, false);
            const height = view.getUint32(4, false);
            const channels = view.getUint32(8, false) || 3;
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                return;
            }
            if (!ctx) {
                ctx = canvas.getContext('2d');
            }
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
                canvas.style.width = width + 'px';
                canvas.style.height = height + 'px';
            }
            const pixels = new Uint8Array(buffer, 12);
            const rgba = new Uint8ClampedArray(width * height * 4);
            for (let src = 0, dst = 0; src < pixels.length; src += channels, dst += 4) {
                const b = pixels[src] || 0;
                const g = pixels[src + 1] || 0;
                const r = pixels[src + 2] || 0;
                rgba[dst] = r;
                rgba[dst + 1] = g;
                rgba[dst + 2] = b;
                rgba[dst + 3] = 255;
            }
            const image = new ImageData(rgba, width, height);
            ctx.putImageData(image, 0, 0);
        }
        function renderLoop() {
            if (pending) {
                draw(pending);
                pending = null;
            }
            window.requestAnimationFrame(renderLoop);
        }
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
            socket = new WebSocket(protocol + '://' + location.host + endpoint);
            socket.binaryType = 'arraybuffer';
            socket.onmessage = (event) => {
                pending = event.data;
            };
            socket.onclose = () => {
                setTimeout(connect, 1000);
            };
        }
        connect();
        renderLoop();
    };
})();
</script>
"""


class SegmentationDashboard:
    """High-level controller wiring NiceGUI with processing logic."""

    _stream_script_registered = False

    def __init__(
        self,
        camera: CameraStream,
        processor: SegmentationProcessor,
        labels: LabelManager,
        frame_publisher: MultiFramePublisher,
        stream_name: str,
        secondary_stream_name: Optional[str] = None,
        secondary_stream_label: str = "Camera 1 (Front)",
        title: str = "Camera 0 (Bottom)",
    ) -> None:
        self.camera = camera
        self.processor = processor
        self.labels = labels
        self.frame_publisher = frame_publisher
        self.stream_name = stream_name
        self.secondary_stream_name = secondary_stream_name
        self.secondary_stream_label = secondary_stream_label
        self.title = title
        self.superpixels = 120
        self.frame_height = 480
        self.frame_width = 640
        self.roi_y = self._default_roi_for_height(self.frame_height)
        self._roi_manual_override = False
        self._latest_segments: Optional[np.ndarray] = None
        self._interactive_overlay: Optional[ui.interactive_image] = None
        self._debug_label: Optional[ui.label] = None
        self._roi_label: Optional[ui.label] = None
        self._mode_label: Optional[ui.label] = None
        self._slider_label: Optional[ui.label] = None
        self._perf_label: Optional[ui.label] = None
        self._status_label: Optional[ui.label] = None
        self._roi_slider: Optional[ui.slider] = None
        self._canvas_container: Optional[ui.element] = None
        self._result_queue: queue.Queue[SegmentationResult] = queue.Queue(maxsize=1)
        self._raw_lock = threading.Lock()
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
        self._fps_counter = 0
        self._fps_window_start = time.time()
        self._last_display_time = 0.0
        self._canvas_id = f"stream-canvas-{id(self)}"
        self._secondary_canvas_id = (
            f"stream-canvas-secondary-{id(self)}" if self.secondary_stream_name else None
        )
        self._overlay_shape = (self.frame_height, self.frame_width)
        self._layout_mode = "stack"
        self._layout_container: Optional[ui.element] = None
        self._layout_toggle: Optional[ui.toggle] = None

    def mount(self) -> None:
        self._ensure_stream_script()
        with ui.column().classes("w-full gap-4"):
            if self.secondary_stream_name:
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("デュアルカメラビュー").classes("text-xl font-bold")
                    self._layout_toggle = ui.toggle(
                        {"stack": "縦配置", "side": "横配置"},
                        value=self._layout_mode,
                        on_change=self._on_layout_toggle,
                    )
            else:
                ui.label(self.title).classes("text-xl font-bold")

            self._layout_container = ui.element("div").classes(
                "w-full flex flex-wrap gap-6 items-start justify-center"
            )
            container = self._layout_container
            self._apply_layout_mode()
            with container:
                self._build_primary_panel()
                if self.secondary_stream_name:
                    self._build_secondary_panel()

        self._start_processing_worker()
        self._timer = ui.timer(UI_REFRESH_INTERVAL_SEC, self._update_frame)

    def start_processing_only(self) -> None:
        self._start_processing_worker()

    def _build_primary_panel(self) -> None:
        with ui.element("div").classes(
            "flex flex-wrap gap-6 items-start justify-center max-w-screen-xl"
        ):
            with ui.column().classes("items-center gap-3"):
                ui.label(self.title).classes("text-lg font-semibold")
                with ui.element("div").classes("relative inline-block").style(
                    f"width:{self.frame_width}px;height:{self.frame_height}px"
                ) as container:
                    self._canvas_container = container
                    ui.html(
                        f'<canvas id="{self._canvas_id}" width="{self.frame_width}" height="{self.frame_height}" '
                        'class="rounded-lg border border-gray-600 bg-black block"></canvas>',
                        sanitize=False,
                    )
                    self._interactive_overlay = ui.interactive_image(
                        self._blank_src(),
                        size=(float(self.frame_width), float(self.frame_height)),
                        on_mouse=self._handle_mouse,
                    ).classes("absolute inset-0 opacity-0 cursor-crosshair").style("z-index: 5;")
                endpoint = f"/stream/{self.stream_name}"
                ui.run_javascript(f"startStreamCanvas('{self._canvas_id}', '{endpoint}');")
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
                self._roi_label = ui.label("")
                self._refresh_roi_label()
                self._roi_slider = ui.slider(
                    min=0,
                    max=self.frame_height,
                    value=self.roi_y,
                    step=5,
                    on_change=self._on_roi_slider_change,
                )
                self._update_roi_slider_gui()
                ui.button("Reset ROI auto", on_click=self._reset_roi_auto)
                self._debug_label = ui.label("Metrics pending...")
                self._status_label = ui.label("Status: STOP | drivable=0.00").classes(
                    "text-xl font-bold text-red-500"
                )
                self._perf_label = ui.label("FPS pending...")

    def _build_secondary_panel(self) -> None:
        if not self.secondary_stream_name or not self._secondary_canvas_id:
            return
        with ui.column().classes("items-center gap-3"):
            ui.label(self.secondary_stream_label).classes("text-lg font-semibold")
            with ui.element("div").classes("relative inline-block").style(
                f"width:{self.frame_width}px;height:{self.frame_height}px"
            ):
                ui.html(
                    f'<canvas id="{self._secondary_canvas_id}" width="{self.frame_width}" height="{self.frame_height}" '
                    'class="rounded-lg border border-gray-600 bg-black block"></canvas>',
                    sanitize=False,
                )
            endpoint = f"/stream/{self.secondary_stream_name}"
            ui.run_javascript(
                f"startStreamCanvas('{self._secondary_canvas_id}', '{endpoint}');"
            )
            ui.label("表示専用 (ROI操作なし)").classes("text-sm text-gray-400")

    def _on_layout_toggle(self, event) -> None:
        value = getattr(event, "value", None) or "stack"
        if value not in {"stack", "side"}:
            value = "stack"
        self._layout_mode = value
        self._apply_layout_mode()

    def _apply_layout_mode(self) -> None:
        if not self._layout_container:
            return
        direction = "column" if self._layout_mode == "stack" else "row"
        self._layout_container.style(
            "display:flex;flex-wrap:wrap;gap:24px;width:100%;justify-content:center;"
            f"align-items:flex-start;flex-direction:{direction};"
        )

    @classmethod
    def _ensure_stream_script(cls) -> None:
        if cls._stream_script_registered:
            return
        ui.add_head_html(STREAM_CLIENT_SCRIPT)
        cls._stream_script_registered = True


    def _on_superpixel_change(self, event) -> None:
        self.superpixels = int(event.value)
        if self._slider_label:
            self._slider_label.text = self._slider_text()

    def _on_roi_slider_change(self, event) -> None:
        self._roi_manual_override = True
        self.roi_y = int(event.value)
        self._refresh_roi_label()

    def _reset_roi_auto(self) -> None:
        self._roi_manual_override = False
        self._sync_default_roi()

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
        self._roi_manual_override = True
        self._refresh_roi_label()

    def _refresh_roi_label(self) -> None:
        if not self._roi_label:
            return
        mode = "manual" if self._roi_manual_override else "auto"
        self._roi_label.text = f"ROI cut line: {self.roi_y}px ({mode}, use slider or drag)"
        self._update_roi_slider_gui()

    def _default_roi_for_height(self, height: int) -> int:
        if height <= 0:
            return 0
        cut_line = height - int(height * ROI_Y_RATIO)
        return int(np.clip(cut_line, 0, height - 1))

    def _sync_default_roi(self) -> None:
        if self._roi_manual_override:
            return
        target = self._default_roi_for_height(self.frame_height)
        if target != self.roi_y:
            self.roi_y = target
            self._refresh_roi_label()

    def _sync_overlay_placeholder(self) -> None:
        if not self._interactive_overlay:
            return
        if (self.frame_height, self.frame_width) == self._overlay_shape:
            return
        self._overlay_shape = (self.frame_height, self.frame_width)
        blank = self._blank_src(self.frame_width, self.frame_height)
        self._interactive_overlay.set_source(blank)
        self._interactive_overlay.style(
            f"width:{self.frame_width}px;height:{self.frame_height}px"
        )
        if self._canvas_container:
            self._canvas_container.style(
                f"width:{self.frame_width}px;height:{self.frame_height}px"
            )
        self._update_roi_slider_gui()

    def _update_roi_slider_gui(self) -> None:
        if not self._roi_slider:
            return
        try:
            self._roi_slider.props(f"max={self.frame_height}")
            self._roi_slider.value = self.roi_y
        except AttributeError:
            pass

    def _ready_for_display(self) -> bool:
        if DISPLAY_TARGET_FPS <= 0:
            return True
        min_interval = 1.0 / DISPLAY_TARGET_FPS
        now = time.time()
        if (now - self._last_display_time) >= min_interval:
            self._last_display_time = now
            return True
        return False

    def _render_display_frame(self, result: SegmentationResult) -> np.ndarray:
        if result.mask_overlay is not None:
            return self._compose_mask_overlay(result)
        return self._compose_overlay(result)

    def _stream_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None or not self.frame_publisher:
            return
        self.frame_publisher.publish(self.stream_name, frame)
        self._report_fps("PhaseD display")

    def _update_frame(self) -> None:
        if PHASE_A_RAW_PREVIEW:
            frame = self._consume_raw_frame()
            if frame is None:
                return
            self.frame_height, self.frame_width = frame.shape[:2]
            self._sync_overlay_placeholder()
            self._sync_default_roi()
            if self._ready_for_display():
                self._stream_frame(frame)
            return

        result = self._consume_latest_result()
        if result is None:
            return
        self.frame_height, self.frame_width = result.frame.shape[:2]
        self._sync_overlay_placeholder()
        self._sync_default_roi()
        self._latest_segments = result.mask_classes if result.mask_classes is not None else result.segments
        if result.mask_classes is not None:
            metrics = self._compute_mask_metrics(result.mask_classes)
            self._update_status_label(metrics)

    def _compose_overlay(self, result) -> np.ndarray:
        overlay = self.labels.build_overlay(result.segments)
        blended = cv2.addWeighted(result.frame, 0.7, overlay, 0.3, 0.0)
        blended[result.boundaries] = (255, 255, 255)
        y = int(np.clip(self.roi_y, 0, blended.shape[0] - 1))
        cv2.line(blended, (0, y), (blended.shape[1] - 1, y), (255, 255, 0), 2)
        return blended

    def _compose_mask_overlay(self, result: SegmentationResult) -> np.ndarray:
        overlay = result.mask_overlay
        if overlay is None:
            return result.frame
        base = result.frame.astype(np.uint16)
        if overlay.shape[-1] == 4:
            alpha = overlay[:, :, 3:4].astype(np.uint16)
            mask_rgb = overlay[:, :, :3].astype(np.uint16)
        else:
            alpha = np.full(overlay.shape[:2] + (1,), 102, dtype=np.uint16)
            mask_rgb = overlay.astype(np.uint16)
        inv_alpha = 255 - alpha
        blended = ((base * inv_alpha + mask_rgb * alpha) // 255).astype(np.uint8)
        y = int(np.clip(self.roi_y, 0, blended.shape[0] - 1))
        cv2.line(blended, (0, y), (blended.shape[1] - 1, y), (255, 255, 0), 2)
        return blended

    def _compute_mask_metrics(self, mask_classes: np.ndarray) -> Metrics:
        h = mask_classes.shape[0]
        roi_start = int(np.clip(self.roi_y, 0, h - 1))
        roi = mask_classes[roi_start:, :]
        if roi.size == 0:
            return Metrics(0.0, 0.0, "STOP")
        drive_mask = np.isin(roi, self._drivable_class_ids)
        drivable_ratio = float(drive_mask.mean())
        go_thresh, avoid_thresh = DRIVABLE_STATUS_THRESHOLDS
        if roi.shape[1] < 2:
            left_right_balance = 0.0
        else:
            mid = roi.shape[1] // 2
            left = drive_mask[:, :mid]
            right = drive_mask[:, mid:]
            left_ratio = float(left.sum()) / float(max(left.size, 1))
            right_ratio = float(right.sum()) / float(max(right.size, 1))
            left_right_balance = left_ratio - right_ratio
        if drivable_ratio > go_thresh:
            decision = "GO"
        elif drivable_ratio > avoid_thresh:
            decision = "AVOID"
        else:
            decision = "STOP"
        return Metrics(drivable_ratio, left_right_balance, decision)

    def _update_status_label(self, metrics: Metrics) -> None:
        status_text = (
            f"Status: {metrics.decision} | drivable={metrics.drivable_ratio:.2f} "
            f"| L-R={metrics.left_right_ratio:+.2f}"
        )
        if self._status_label:
            self._status_label.text = status_text
            color = STATUS_COLORS.get(metrics.decision, "#e5e7eb")
            self._status_label.style(f"color: {color}")
        steer_hint = "CENTER"
        if metrics.left_right_ratio > 0.05:
            steer_hint = "LEFT"
        elif metrics.left_right_ratio < -0.05:
            steer_hint = "RIGHT"
        if self._debug_label:
            self._debug_label.text = f"ROI y={self.roi_y}px | steer={steer_hint}"
        self._refresh_roi_label()

    @staticmethod
    def _to_data_url(image: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            return ""
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _blank_src(width: int = 640, height: int = 480) -> str:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(blank, "Awaiting stream...", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
            if self._ready_for_display():
                display_frame = self._render_display_frame(result)
                result.composited_frame = display_frame
                self._stream_frame(display_frame)
            else:
                result.composited_frame = None
            self._publish_result(result)
            self._update_worker_fps()
            if PROCESSING_SLEEP_SEC > 0:
                time.sleep(PROCESSING_SLEEP_SEC)

    def _publish_result(self, result: SegmentationResult) -> None:
        try:
            while True:
                self._result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            pass

    def _consume_latest_result(self) -> Optional[SegmentationResult]:
        latest: Optional[SegmentationResult] = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

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
