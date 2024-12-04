from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi
    from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewer


class MdiInferencer(QObject):
    def __init__(self, mdi: Mdi):
        super().__init__()

        self._mdi = mdi

    def _active_layered_image(self) -> LayeredImage | None:
        viewer = self._active_layered_image_viewer()
        if viewer is not None:
            return viewer.data

    def _active_layered_image_viewer(self) -> LayeredImageViewer | None:
        layered_image_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        return layered_image_viewer_sub_window and layered_image_viewer_sub_window.layered_image_viewer
