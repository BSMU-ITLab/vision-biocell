from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from bsmu.biocell.infervis.segmenters.tiled import MultipassTiledMdiSegmenter
from bsmu.vision.core.image import MaskDrawMode
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu

if TYPE_CHECKING:
    from bsmu.biocell.plugins.tissue_dnn_segmenter import TissueDnnSegmenterPlugin
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow


class TissueDnnSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'tissue_dnn_segmenter_plugin': 'bsmu.biocell.plugins.tissue_dnn_segmenter.TissueDnnSegmenterPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            tissue_dnn_segmenter_plugin: TissueDnnSegmenterPlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None

        self._mdi_plugin = mdi_plugin

        self._tissue_dnn_segmenter_plugin = tissue_dnn_segmenter_plugin

        self._tissue_segmenter_gui: MultipassTiledMdiSegmenter | None = None

    @property
    def tissue_segmenter_gui(self) -> MultipassTiledMdiSegmenter | None:
        return self._tissue_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window
        mdi = self._mdi_plugin.mdi

        self._tissue_segmenter_gui = MultipassTiledMdiSegmenter(
            self._tissue_dnn_segmenter_plugin.tissue_segmenter,
            mdi,
        )

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Segment Tissue Using DNN'),
            partial(
                self._tissue_segmenter_gui.segment_async,
                mask_layer_name='masks',
                mask_draw_mode=MaskDrawMode.OVERLAY_FOREGROUND,
            ),
        )

    def _disable(self):
        self._tissue_segmenter_gui = None

        self._main_window = None

        raise NotImplementedError()
