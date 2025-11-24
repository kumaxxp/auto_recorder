"""Entry point for the NiceGUI segmentation prototype."""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import WebSocket
from nicegui import app, ui

from camera_stream import CameraStream, DualDeepStreamPipelines, MultiFramePublisher
from labeling import LabelManager
from segmentation import (
    CITYSCAPES_DRIVABLE_CLASS_IDS,
    BottomROIMetrics,
    FrontObstacleMetrics,
    FusionStatus,
    SegmentationProcessor,
    GeometryAnalysis,
    compute_floor_wall_geometry,
    process_bottom_mask,
    process_front_mask,
)
from ui import SegmentationDashboard


SEGMENTATION_CONFIG = Path("configs/deepstream_drivable_segmentation.txt")
ENABLE_DUAL_CAMERA = True


class FusionMonitor:
    """Continuously fuses bottom/front ROI telemetry on a worker thread."""

    def __init__(
        self,
        bottom_camera: CameraStream,
        front_camera: Optional[CameraStream],
        roi_getter: Callable[[], int],
        drivable_ids: tuple[int, ...],
        interval: float = 0.05,
        front_hard_stop_ratio: float = 0.4,
        floor_threshold: float = 0.04,
        wall_threshold: float = 0.15,
        geometry_conf_threshold: float = 0.45,
    ) -> None:
        self._bottom_camera = bottom_camera
        self._front_camera = front_camera
        self._roi_getter = roi_getter
        self._drivable_ids = drivable_ids
        self._interval = interval
        self._front_hard_stop_ratio = front_hard_stop_ratio
        self._floor_threshold = floor_threshold
        self._wall_threshold = wall_threshold
        self._geometry_conf_threshold = geometry_conf_threshold
        self._status: FusionStatus = FusionStatus()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def latest(self) -> FusionStatus:
        with self._lock:
            return self._status

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            status = self._compute_status()
            with self._lock:
                self._status = status
            time.sleep(self._interval)

    def _compute_status(self) -> FusionStatus:
        roi = self._safe_roi()
        frame = self._bottom_camera.read()
        geometry = compute_floor_wall_geometry(
            frame,
            roi,
            floor_threshold=self._floor_threshold,
            wall_threshold=self._wall_threshold,
        )
        bottom_mask = self._bottom_camera.read_mask_classes()
        bottom_metrics = process_bottom_mask(
            bottom_mask,
            roi,
            self._drivable_ids,
            geometry=geometry,
        )
        front_mask = (
            self._front_camera.read_mask_classes() if self._front_camera is not None else None
        )
        front_metrics = process_front_mask(front_mask)
        decision, warning = self._fuse_decision(bottom_metrics, front_metrics, geometry)
        return FusionStatus(
            bottom=bottom_metrics,
            front=front_metrics,
            decision=decision,
            warning=warning,
            timestamp=time.time(),
            geometry_confidence=geometry.confidence,
            geometry_active=geometry.floor_mask is not None,
        )

    def _safe_roi(self) -> int:
        try:
            return int(self._roi_getter())
        except Exception:
            return 0

    def _fuse_decision(
        self,
        bottom: BottomROIMetrics,
        front: FrontObstacleMetrics,
        geometry: GeometryAnalysis,
    ) -> tuple[str, str]:
        warning = "Front clear"
        decision = bottom.decision
        if front.detected:
            warning = f"Obstacle {front.obstacle_ratio * 100:.1f}% center"
            if front.obstacle_ratio >= self._front_hard_stop_ratio or decision == "STOP":
                decision = "STOP"
            elif decision == "GO":
                decision = "AVOID"
        elif geometry.floor_mask is not None and geometry.confidence < self._geometry_conf_threshold:
            decision = "AVOID" if decision == "GO" else decision
            warning = f"Low floor confidence {geometry.confidence:.2f}"
        else:
            warning = f"geometry={geometry.confidence:.2f}"
        return decision, warning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NiceGUI segmentation demo")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Auto-shutdown NiceGUI after the given number of seconds (dev testing)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip automatically opening a new browser tab/window",
    )
    return parser.parse_args()

def _create_cameras() -> tuple[CameraStream, Optional[CameraStream], Optional[DualDeepStreamPipelines]]:
    if ENABLE_DUAL_CAMERA and SEGMENTATION_CONFIG.exists():
        try:
            pipelines = DualDeepStreamPipelines(
                bottom_index=0,
                front_index=1,
                config_path=SEGMENTATION_CONFIG,
            )
            return pipelines.pipeline_bottom, pipelines.pipeline_front, pipelines
        except FileNotFoundError:
            pass
    bottom = CameraStream(
        index=0,
        use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
        gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
    )
    front = None
    if ENABLE_DUAL_CAMERA:
        front = CameraStream(
            index=1,
            use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
            gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
        )
    return bottom, front, None


camera0, camera1, dual_pipelines = _create_cameras()
processor0 = SegmentationProcessor()
labels0 = LabelManager()
frame_publishers = MultiFramePublisher()
secondary_stream = "front" if camera1 else None

fusion_monitor: Optional[FusionMonitor] = None


def _fusion_status_provider() -> Optional[FusionStatus]:
    return fusion_monitor.latest() if fusion_monitor else None


dashboard0 = SegmentationDashboard(
    camera0,
    processor0,
    labels0,
    frame_publishers,
    "bottom",
    secondary_stream_name=secondary_stream,
    secondary_stream_label="Camera 1 (Front)",
    title="Camera 0 (Bottom)",
    fusion_status_provider=_fusion_status_provider if camera1 else None,
)

dashboard1 = None
if camera1:
    processor1 = SegmentationProcessor()
    labels1 = LabelManager()
    dashboard1 = SegmentationDashboard(
        camera1,
        processor1,
        labels1,
        frame_publishers,
        "front",
        title="Camera 1 (Front)",
    )


def _ensure_fusion_monitor() -> None:
    global fusion_monitor
    if fusion_monitor or camera1 is None:
        return
    roi_getter = lambda: getattr(dashboard0, "roi_y", 0)
    fusion_monitor = FusionMonitor(
        bottom_camera=camera0,
        front_camera=camera1,
        roi_getter=roi_getter,
        drivable_ids=CITYSCAPES_DRIVABLE_CLASS_IDS,
    )
    fusion_monitor.start()


_shutdown_registered = False


@ui.page("/")
def index_page() -> None:
    """Render the dashboard for incoming requests."""
    with ui.column().classes("w-full items-center"):
        dashboard0.mount()

@app.websocket("/stream/{stream_name}")
async def stream_endpoint(websocket: WebSocket, stream_name: str) -> None:
    await frame_publishers.stream(stream_name, websocket)


def main() -> None:
    args = parse_args()
    global _shutdown_registered
    # Start both cameras
    if dual_pipelines:
        dual_pipelines.start()
    else:
        if not camera0.is_running():
            camera0.start()
        if camera1 and not camera1.is_running():
            camera1.start()
    if dashboard1:
        dashboard1.start_processing_only()
    _ensure_fusion_monitor()
        
    if not _shutdown_registered:
        def shutdown():
            dashboard0.shutdown()
            if dashboard1:
                dashboard1.shutdown()
            if fusion_monitor:
                fusion_monitor.stop()
            if dual_pipelines:
                dual_pipelines.stop()
            else:
                camera0.stop()
                if camera1:
                    camera1.stop()
        app.on_shutdown(shutdown)
        _shutdown_registered = True
    if args.duration is not None and args.duration > 0:
        async def _auto_shutdown() -> None:
            await asyncio.sleep(args.duration)
            app.shutdown()

        app.on_startup(_auto_shutdown)
    ui.run(title="RC Explorer Segmentation", reload=False, show=not args.no_browser)


if __name__ == "__main__":
    main()
