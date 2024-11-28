from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMessageBox

from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.image.layered import LayeredImage
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.core.visibility import Visibility
from bsmu.vision.plugins.windows.main import AlgorithmsMenu
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder

if TYPE_CHECKING:
    from bsmu.biocell.plugins.tissue_dnn_segmenter import TissueSegmenter, TissueDnnSegmenterPlugin
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin, PalettePackSettings
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow


class TissueDnnSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'tissue_dnn_segmenter_plugin': 'bsmu.biocell.plugins.tissue_dnn_segmenter.TissueDnnSegmenterPlugin',
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            tissue_dnn_segmenter_plugin: TissueDnnSegmenterPlugin,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None

        self._mdi_plugin = mdi_plugin

        self._tissue_dnn_segmenter_plugin = tissue_dnn_segmenter_plugin

        self._palette_pack_settings_plugin = palette_pack_settings_plugin
        self._palette_pack_settings: PalettePackSettings | None = None

        self._tissue_segmenter_gui: TissueSegmenterGui | None = None

    @property
    def tissue_segmenter_gui(self) -> TissueSegmenterGui | None:
        return self._tissue_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window
        mdi = self._mdi_plugin.mdi

        self._tissue_segmenter_gui = TissueSegmenterGui(
            self._tissue_dnn_segmenter_plugin.tissue_segmenter,
            mdi,
        )

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Segment Tissue Using DNN'),
            partial(self._tissue_segmenter_gui.segment, mask_layer_name='masks'),
        )

    def _disable(self):
        self._tissue_segmenter_gui = None

        self._main_window = None

        self._palette_pack_settings = None

        raise NotImplementedError


class MdiSegmenter(QObject):
    def __init__(self, mdi: Mdi):
        super().__init__()

        self._mdi = mdi

    def _active_layered_image(self) -> LayeredImage | None:
        layered_image_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        return layered_image_viewer_sub_window and layered_image_viewer_sub_window.layered_image_viewer.data


class TissueSegmenterGui(MdiSegmenter):
    def __init__(self, tissue_segmenter: TissueSegmenter, mdi: Mdi):
        super().__init__(mdi)

        self._tissue_segmenter = tissue_segmenter

    @property
    def mask_foreground_class(self) -> int:
        return self._tissue_segmenter.mask_foreground_class

    @property
    def mask_background_class(self) -> int:
        return self._tissue_segmenter.mask_background_class

    def segment(self, mask_layer_name: str):
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        if layered_image.contains_layer(mask_layer_name):
            reply = QMessageBox.question(
                self._mdi,
                self.tr('Non-unique Layer Name'),
                self.tr(
                    'Viewer already contains a layer with such name: {0}. '
                    'Repaint its content?'
                ).format(mask_layer_name),
                defaultButton=QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        image_layer = layered_image.layers[0]
        image = image_layer.image
        mask = self._tissue_segmenter.segment(image.pixels)

        layered_image.add_layer_or_modify_pixels(
            mask_layer_name,
            mask,
            FlatImage,
            self._tissue_segmenter.mask_palette,
            visibility=Visibility(True, 0.5),
        )
