"""Entry point for the NiceGUI segmentation prototype."""

from __future__ import annotations

from nicegui import ui

from camera_stream import CameraStream
from labeling import LabelManager
from segmentation import SegmentationProcessor
from ui import SegmentationDashboard


def main() -> None:
    camera = CameraStream()
    camera.start()
    processor = SegmentationProcessor()
    labels = LabelManager()

    dashboard = SegmentationDashboard(camera, processor, labels)
    dashboard.mount()

    ui.run(title="RC Explorer Segmentation", reload=False)


if __name__ == "__main__":
    main()
