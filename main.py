"""Entry point for the NiceGUI segmentation prototype."""

from __future__ import annotations

from pathlib import Path

from nicegui import app, ui

from camera_stream import CameraStream
from labeling import LabelManager
from segmentation import SegmentationProcessor
from ui import SegmentationDashboard


SEGMENTATION_CONFIG = Path("configs/deepstream_drivable_segmentation.txt")

# Initialize Camera 0 (Bottom)
camera0 = CameraStream(
    index=0,
    use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
    gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
)
processor0 = SegmentationProcessor()
labels0 = LabelManager()
dashboard0 = SegmentationDashboard(camera0, processor0, labels0)

# Initialize Camera 1 (Top)
camera1 = CameraStream(
    index=1,
    use_gstreamer_segmentation=SEGMENTATION_CONFIG.exists(),
    gstreamer_segmentation_config=str(SEGMENTATION_CONFIG),
)
processor1 = SegmentationProcessor()
labels1 = LabelManager()
dashboard1 = SegmentationDashboard(camera1, processor1, labels1)

_shutdown_registered = False


@ui.page("/")
def index_page() -> None:
    """Render the dashboard for incoming requests."""
    with ui.column().classes("w-full items-center"):
        ui.label("Camera 1 (Top)").classes("text-xl font-bold")
        dashboard1.mount()
        ui.separator().classes("my-4")
        ui.label("Camera 0 (Bottom)").classes("text-xl font-bold")
        dashboard0.mount()


def main() -> None:
    global _shutdown_registered
    # Start both cameras
    if not camera0.is_running():
        camera0.start()
    if not camera1.is_running():
        camera1.start()
        
    if not _shutdown_registered:
        def shutdown():
            camera0.stop()
            camera1.stop()
        app.on_shutdown(shutdown)
        _shutdown_registered = True
    ui.run(title="RC Explorer Segmentation", reload=False)


if __name__ == "__main__":
    main()
