from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from bsmu.vision.widgets.viewers.layered import LayeredDataViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.core.data.layered import LayeredData
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi
    from bsmu.vision.widgets.viewers.layered import LayeredDataViewer


class MdiInferencer(QObject):
    def __init__(self, mdi: Mdi):
        super().__init__()

        self._mdi = mdi

    def _active_layered_data(self) -> LayeredData | None:
        viewer = self._active_layered_data_viewer()
        if viewer is not None:
            return viewer.data
        return None

    def _active_layered_data_viewer(self) -> LayeredDataViewer | None:
        layered_data_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredDataViewerHolder)
        return layered_data_viewer_sub_window and layered_data_viewer_sub_window.layered_data_viewer
