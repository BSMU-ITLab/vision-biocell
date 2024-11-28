from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bsmu.biocell.inference.segmenters.tiled import SegmentationMode
from bsmu.biocell.infervis.segmenters.mdi import MaskDrawMode, MdiSegmenter
from bsmu.biocell.infervis.segmenters.tiled import MultipassTiledMdiSegmenter
from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.image.layered import LayeredImage
from bsmu.vision.core.palette import Palette
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.core.visibility import Visibility
from bsmu.vision.plugins.windows.main import AlgorithmsMenu

if TYPE_CHECKING:
    from typing import Sequence, Callable

    from bsmu.biocell.plugins.pca_segmenter import PcaSegmenter, PcaSegmenterPlugin
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin, PalettePackSettings
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from bsmu.vision.plugins.visualizers.manager import DataVisualizationManagerPlugin, DataVisualizationManager


class PcaSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'pca_segmenter_plugin': 'bsmu.biocell.plugins.pca_segmenter.PcaSegmenterPlugin',
        'data_visualization_manager_plugin':
            'bsmu.vision.plugins.visualizers.manager.DataVisualizationManagerPlugin',
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            pca_segmenter_plugin: PcaSegmenterPlugin,
            data_visualization_manager_plugin: DataVisualizationManagerPlugin,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None

        self._mdi_plugin = mdi_plugin
        self._mdi: Mdi | None = None

        self._pca_segmenter_plugin = pca_segmenter_plugin

        self._data_visualization_manager_plugin = data_visualization_manager_plugin
        self._data_visualization_manager: DataVisualizationManager

        self._palette_pack_settings_plugin = palette_pack_settings_plugin
        self._palette_pack_settings: PalettePackSettings | None = None

        self._pca_gleason_3_segmenter_gui: MultipassTiledMdiSegmenter | None = None
        self._pca_gleason_4_segmenter_gui: MultipassTiledMdiSegmenter | None = None

        self._pca_segmenter_gui: PcaMdiSegmenter | None = None

    @property
    def pca_gleason_3_segmenter_gui(self) -> MultipassTiledMdiSegmenter | None:
        return self._pca_gleason_3_segmenter_gui

    @property
    def pca_gleason_4_segmenter_gui(self) -> MultipassTiledMdiSegmenter | None:
        return self._pca_gleason_4_segmenter_gui

    @property
    def pca_segmenter_gui(self) -> PcaMdiSegmenter | None:
        return self._pca_segmenter_gui

    def _enable(self):
        self._palette_pack_settings = self._palette_pack_settings_plugin.settings

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window
        self._mdi = self._mdi_plugin.mdi

        self._data_visualization_manager = self._data_visualization_manager_plugin.data_visualization_manager
        # self._data_visualization_manager.data_visualized.connect(self._pca_gleason_3_segmenter_gui.on_data_visualized)

        self._pca_gleason_3_segmenter_gui = MultipassTiledMdiSegmenter(
            self._pca_segmenter_plugin.pca_gleason_3_segmenter, self._mdi)
        self._pca_gleason_4_segmenter_gui = MultipassTiledMdiSegmenter(
            self._pca_segmenter_plugin.pca_gleason_4_segmenter, self._mdi)

        self._pca_segmenter_gui = PcaMdiSegmenter(
            self._pca_segmenter_plugin.pca_segmenter,
            [self._pca_gleason_3_segmenter_gui, self._pca_gleason_4_segmenter_gui],
            self._mdi,
        )

        # self._main_window.add_menu_action(AlgorithmsMenu, 'Segment Prostate Tissue', self._segment_prostate_tissue)

        self._add_segmentation_submenu(
            self.tr('Segment Cancer'),
            partial(self._pca_segmenter_gui.segment_async, mask_layer_name='masks'),
        )
        self._add_segmentation_submenu(
            self.tr('Segment Gleason >= 3'),
            partial(self._pca_gleason_3_segmenter_gui.segment_async, mask_layer_name='gleason >= 3'),
        )
        self._add_segmentation_submenu(
            self.tr('Segment Gleason >= 4'),
            partial(self._pca_gleason_4_segmenter_gui.segment_async, mask_layer_name='gleason >= 4'),
        )

    def _disable(self):
        self._pca_segmenter_gui = None
        self._pca_gleason_3_segmenter_gui = None
        self._pca_gleason_4_segmenter_gui = None

        self._data_visualization_manager = None

        self._mdi = None
        self._main_window = None

        self._palette_pack_settings = None

        raise NotImplementedError()

    def _add_segmentation_submenu(self, title: str, method: Callable):
        submenu = self._main_window.add_submenu(AlgorithmsMenu, title)
        for segmentation_mode in SegmentationMode:
            submenu.addAction(segmentation_mode.display_name, partial(method, segmentation_mode=segmentation_mode))

    def _segment_prostate_tissue(self):
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        tissue_layer_name = 'prostate-tissue'

        image = layered_image.layers[0].image.pixels
        tissue_mask = segment_tissue(image)
        print('Tissue mask: ', tissue_mask.dtype, tissue_mask.shape, tissue_mask.min(), tissue_mask.max(), np.unique(tissue_mask))
        layered_image.add_layer_or_modify_pixels(
            tissue_layer_name,
            tissue_mask,
            FlatImage,
            Palette.default_binary(rgb_color=[100, 255, 100]),
            Visibility(True, 0.5),
        )


def segment_tissue(image: np.ndarray) -> np.ndarray:
    var = image - image.mean(-1, dtype=np.int16, keepdims=True)
    var = abs(var).mean(-1, dtype=np.uint16)
    tissue_mask = np.where(var > 2, True, False).astype(np.uint8)
    return tissue_mask


class PcaMdiSegmenter(MdiSegmenter):
    def __init__(
            self, pca_segmenter: PcaSegmenter, class_mdi_segmenters: Sequence[MultipassTiledMdiSegmenter], mdi: Mdi):
        super().__init__(mdi)

        self._pca_segmenter = pca_segmenter
        self._class_mdi_segmenters = class_mdi_segmenters

    def segment_async(
            self,
            mask_layer_name: str,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        image_layer = layered_image.layers[0]
        image = image_layer.image
        on_finished = partial(
            self._on_pca_segmentation_finished,
            layered_image=layered_image,
            mask_layer_name=mask_layer_name,
            mask_draw_mode=mask_draw_mode,
        )
        self._pca_segmenter.segment_async(image, segmentation_mode, on_finished)

    def _on_pca_segmentation_finished(
            self,
            masks: Sequence[np.ndarray],
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        # Apply the passed `mask_draw_mode` only to draw the first mask
        first = 0
        modifiable_mask = self._class_mdi_segmenters[first].update_mask_layer(
            masks[first], layered_image, mask_layer_name, mask_draw_mode)

        # Apply other draw modes for subsequent masks to preserve already drawn masks
        if mask_draw_mode == MaskDrawMode.REDRAW_ALL or mask_draw_mode == MaskDrawMode.OVERLAY_FOREGROUND:
            update_mask_layer = partial(self._update_mask_layer, mask_draw_mode=MaskDrawMode.OVERLAY_FOREGROUND)
        elif mask_draw_mode == MaskDrawMode.FILL_BACKGROUND:
            update_mask_layer = partial(self._update_mask_layer_partially, modifiable_mask=modifiable_mask)
        else:
            raise ValueError(f'Invalid MaskDrawMode: {mask_draw_mode}')

        for class_mdi_segmenter, mask in zip(self._class_mdi_segmenters[1:], masks[1:]):
            update_mask_layer(class_mdi_segmenter, mask, layered_image, mask_layer_name)

    @staticmethod
    def _update_mask_layer(
            class_segmenter_gui: MultipassTiledMdiSegmenter,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode,
    ):
        class_segmenter_gui.update_mask_layer(mask, layered_image, mask_layer_name, mask_draw_mode)

    @staticmethod
    def _update_mask_layer_partially(
            class_segmenter_gui: MultipassTiledMdiSegmenter,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            modifiable_mask: np.ndarray,
    ):
        class_segmenter_gui.update_mask_layer_partially(mask, layered_image, mask_layer_name, modifiable_mask)
