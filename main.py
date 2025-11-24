"""Entry point for the NiceGUI segmentation prototype."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fastapi import WebSocket
from nicegui import app, ui

from camera_stream import CameraStream, MultiFramePublisher
from labeling import LabelManager
from segmentation import SegmentationProcessor
from ui import SegmentationDashboard


SEGMENTATION_CONFIG = Path("configs/deepstream_drivable_segmentation.txt")
ENABLE_DUAL_CAMERA = True


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

# Initialize Camera 0 (Bottom)
camera0 = CameraStream(
    index=0,
    use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
    gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
)
processor0 = SegmentationProcessor()
labels0 = LabelManager()
frame_publishers = MultiFramePublisher()
secondary_stream = "front" if ENABLE_DUAL_CAMERA else None
dashboard0 = SegmentationDashboard(
    camera0,
    processor0,
    labels0,
    frame_publishers,
    "bottom",
    secondary_stream_name=secondary_stream,
    secondary_stream_label="Camera 1 (Front)",
    title="Camera 0 (Bottom)",
)

camera1 = None
dashboard1 = None
if ENABLE_DUAL_CAMERA:
    camera1 = CameraStream(
        index=1,
        use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
        gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
    )
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
    if not camera0.is_running():
        camera0.start()
    if ENABLE_DUAL_CAMERA and camera1 and not camera1.is_running():
        camera1.start()
        if dashboard1:
            dashboard1.start_processing_only()
        
    if not _shutdown_registered:
        def shutdown():
            dashboard0.shutdown()
            camera0.stop()
            if ENABLE_DUAL_CAMERA and camera1 and dashboard1:
                dashboard1.shutdown()
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
