from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from bsmu.biocell.inference.segmenters.tiled import (
    MultipassTiledSegmenter, SegmentationMode, MultipassTiledSegmentationProfile,
    MulticlassMultipassTiledSegmentationTask)
from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.dnn.inferencer import ImageModelParams as DnnModelParams

if TYPE_CHECKING:
    from typing import Callable, Sequence
    import numpy as np
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.storages.task import TaskStorage, TaskStoragePlugin


class PcaSegmenterPlugin(Plugin):
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

        self._pca_gleason_3_segmenter: MultipassTiledSegmenter | None = None
        self._pca_gleason_4_segmenter: MultipassTiledSegmenter | None = None

        self._pca_segmenter: PcaSegmenter | None = None

    @property
    def pca_gleason_3_segmenter(self) -> MultipassTiledSegmenter | None:
        return self._pca_gleason_3_segmenter

    @property
    def pca_gleason_4_segmenter(self) -> MultipassTiledSegmenter | None:
        return self._pca_gleason_4_segmenter

    @property
    def pca_segmenter(self) -> PcaSegmenter | None:
        return self._pca_segmenter

    def _enable(self):
        gleason_3_model_params = DnnModelParams.from_config(
            self.config_value('gleason_3_segmenter_model'), self.data_path(self._DNN_MODELS_DIR_NAME))
        gleason_4_model_params = DnnModelParams.from_config(
            self.config_value('gleason_4_segmenter_model'), self.data_path(self._DNN_MODELS_DIR_NAME))

        main_palette = self._palette_pack_settings_plugin.settings.main_palette
        task_storage = self._task_storage_plugin.task_storage
        self._pca_gleason_3_segmenter = MultipassTiledSegmenter(
            gleason_3_model_params, main_palette, 'gleason_3', task_storage)
        self._pca_gleason_4_segmenter = MultipassTiledSegmenter(
            gleason_4_model_params, main_palette, 'gleason_4', task_storage)
        self._pca_segmenter = PcaSegmenter(
            [self._pca_gleason_3_segmenter, self._pca_gleason_4_segmenter],
            task_storage,
        )

    def _disable(self):
        self._pca_segmenter = None
        self._pca_gleason_3_segmenter = None
        self._pca_gleason_4_segmenter = None


class PcaSegmenter(QObject):
    def __init__(self, class_segmenters: Sequence[MultipassTiledSegmenter], task_storage: TaskStorage = None):
        super().__init__()

        self._class_segmenters = class_segmenters
        self._task_storage = task_storage

    def segment_async(
            self,
            image: Image,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            on_finished: Callable[[Sequence[np.ndarray]], None] | None = None,
    ):
        pca_segmentation_task = self.create_segmentation_task(image, segmentation_mode)
        pca_segmentation_task.on_finished = on_finished

        if self._task_storage is not None:
            self._task_storage.add_item(pca_segmentation_task)
        ThreadPool.run_async_task(pca_segmentation_task)

    def create_segmentation_task(
            self,
            image: Image,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY
    ) -> MulticlassMultipassTiledSegmentationTask:

        segmentation_profiles = []
        for class_segmenter in self._class_segmenters:
            segmentation_profiles.append(
                MultipassTiledSegmentationProfile(
                    class_segmenter.segmenter,
                    segmentation_mode,
                    class_segmenter.mask_background_class,
                    class_segmenter.mask_foreground_class,
                )
            )
        pca_segmentation_task_name = f'PCa {segmentation_mode.short_name_with_postfix} [{image.path_name}]'
        return MulticlassMultipassTiledSegmentationTask(image.pixels, segmentation_profiles, pca_segmentation_task_name)

    def combine_class_masks(self, class_masks: Sequence[np.ndarray]) -> np.ndarray:
        combined_mask = class_masks[0].copy()
        # Skip first elements, because the `combined_mask` already contains the first mask
        for class_mask, class_segmenter in zip(class_masks[1:], self._class_segmenters[1:]):
            is_foreground_class = class_mask == class_segmenter.mask_foreground_class
            combined_mask[is_foreground_class] = class_segmenter.mask_foreground_class
        return combined_mask
