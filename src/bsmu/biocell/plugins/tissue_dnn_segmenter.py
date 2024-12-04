from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.biocell.inference.segmenters.tiled import MultipassTiledSegmenter
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.dnn.inferencer import ImageModelParams as DnnModelParams

if TYPE_CHECKING:
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.storages.task import TaskStoragePlugin


class TissueDnnSegmenterPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    _DNN_MODELS_DIR_NAME = 'dnn-models'
    _DATA_DIRS = (_DNN_MODELS_DIR_NAME,)

    def __init__(
            self,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()

        self._palette_pack_settings_plugin = palette_pack_settings_plugin
        self._task_storage_plugin = task_storage_plugin

        self._tissue_segmenter: MultipassTiledSegmenter | None = None

    @property
    def tissue_segmenter(self) -> MultipassTiledSegmenter | None:
        return self._tissue_segmenter

    def _enable(self):
        tissue_model_params = DnnModelParams.from_config(
            self.config_value('tissue_segmenter_model'), self.data_path(self._DNN_MODELS_DIR_NAME))

        main_palette = self._palette_pack_settings_plugin.settings.main_palette
        task_storage = self._task_storage_plugin.task_storage
        self._tissue_segmenter = MultipassTiledSegmenter(
            tissue_model_params, main_palette, 'non_tissue', task_storage)

    def _disable(self):
        self._tissue_segmenter = None
