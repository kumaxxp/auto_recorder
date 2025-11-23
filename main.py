"""Entry point for the NiceGUI segmentation prototype."""

from __future__ import annotations

from nicegui import app, ui

from camera_stream import CameraStream
from labeling import LabelManager
from segmentation import SegmentationProcessor
from ui import SegmentationDashboard


camera = CameraStream()
processor = SegmentationProcessor()
labels = LabelManager()
dashboard = SegmentationDashboard(camera, processor, labels)
_shutdown_registered = False


@ui.page("/")
def index_page() -> None:
    """Render the dashboard for incoming requests."""
    dashboard.mount()


def main() -> None:
    global _shutdown_registered
    if not camera.is_running():
        camera.start()
    if not _shutdown_registered:
        app.on_shutdown(camera.stop)
        _shutdown_registered = True
    ui.run(title="RC Explorer Segmentation", reload=False)


if __name__ == "__main__":
    main()
