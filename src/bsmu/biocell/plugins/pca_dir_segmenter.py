from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from bsmu.biocell.plugins.pca_segmenter import SegmentationMode
from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.config import Config
from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.task import DnnTask
from bsmu.vision.plugins.readers.image.wsi import WholeSlideImageFileReader
from bsmu.vision.plugins.writers.image.common import CommonImageFileWriter

if TYPE_CHECKING:
    from typing import Sequence

    from bsmu.biocell.plugins.pca_segmenter import PcaSegmenter
    from bsmu.vision.plugins.readers.image import ImageFileReader
    from bsmu.vision.plugins.storages.task import TaskStorage


@dataclass
class DirSegmentationConfig(Config):
    image_dir: Path = field(default_factory=Path)
    mask_dir: Path = field(default_factory=Path)
    include_subdirs: bool = True
    overwrite_existing_masks: bool = False
    segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY


class PcaDirSegmenter(QObject):
    def __init__(self, pca_segmenter: PcaSegmenter, task_storage: TaskStorage = None):
        super().__init__()

        self._pca_segmenter = pca_segmenter
        self._task_storage = task_storage

    def segment_async(self, config: DirSegmentationConfig) -> bool:
        if (not config.image_dir.is_dir()) or (config.mask_dir.exists() and not config.mask_dir.is_dir()):
            return False

        wsi_file_reader = WholeSlideImageFileReader()
        pca_dir_segmentation_task_name = (
            self.tr(f'PCa Dir {config.segmentation_mode.short_name_with_postfix} [{config.image_dir.name}]')
        )
        pca_dir_segmentation_task = PcaDirSegmentationTask(
            config, wsi_file_reader, self._pca_segmenter, pca_dir_segmentation_task_name)
        if self._task_storage is not None:
            self._task_storage.add_item(pca_dir_segmentation_task)
        ThreadPool.run_async_task(pca_dir_segmentation_task)
        return True


class PcaDirSegmentationTask(DnnTask):
    def __init__(
            self,
            config: DirSegmentationConfig,
            file_reader: ImageFileReader,
            pca_segmenter: PcaSegmenter,
            name: str = '',
    ):
        super().__init__(name)

        self._config = config
        self._file_reader = file_reader
        self._pca_segmenter = pca_segmenter

        self._finished_subtask_count = 0
        self._relative_image_paths: Sequence[Path] | None = None

    def _run(self):
        return self._segment_dir_files()

    def _segment_dir_files(self):
        self._prepare_relative_image_paths()

        image_file_writer = CommonImageFileWriter()
        for self._finished_subtask_count, relative_image_path in enumerate(self._relative_image_paths):
            image_path = self._config.image_dir / relative_image_path
            file_reading_and_segmentation_task = PcaFileReadingAndSegmentationTask(
                image_path, self._file_reader, self._pca_segmenter, self._config.segmentation_mode)
            file_reading_and_segmentation_task.progress_changed.connect(
                self._on_file_reading_and_segmentation_subtask_progress_changed)
            file_reading_and_segmentation_task.run()
            mask = file_reading_and_segmentation_task.result
            mask_path = self._assemble_mask_path(relative_image_path)
            image_file_writer.write_to_file(FlatImage(mask), mask_path, mkdir=True)

    def _prepare_relative_image_paths(self):
        pattern = '**/*' if self._config.include_subdirs else '*'
        self._relative_image_paths = []
        for image_path in self._config.image_dir.glob(pattern):
            if not (image_path.is_file() and self._file_reader.can_read(image_path)):
                continue

            relative_image_path = image_path.relative_to(self._config.image_dir)
            if not self._config.overwrite_existing_masks:
                mask_path = self._assemble_mask_path(relative_image_path)
                if mask_path.exists():
                    continue

            self._relative_image_paths.append(relative_image_path)

    def _assemble_mask_path(self, relative_image_path: Path) -> Path:
        return self._config.mask_dir / relative_image_path.with_suffix('.png')

    def _on_file_reading_and_segmentation_subtask_progress_changed(self, progress: float):
        self._change_subtask_based_progress(self._finished_subtask_count, len(self._relative_image_paths), progress)


class PcaFileReadingAndSegmentationTask(DnnTask):
    def __init__(
            self,
            image_path: Path,
            image_file_reader: ImageFileReader,
            pca_segmenter: PcaSegmenter,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            name: str = ''
    ):
        super().__init__(name)

        self._image_path = image_path
        self._image_file_reader = image_file_reader
        self._pca_segmenter = pca_segmenter
        self._segmentation_mode = segmentation_mode

    def _run(self):
        return self._read_and_segment()

    def _read_and_segment(self):
        image = self._image_file_reader.read_file(self._image_path)
        pca_segmentation_task = self._pca_segmenter.create_segmentation_task(image, self._segmentation_mode)
        pca_segmentation_task.progress_changed.connect(self.progress_changed)
        pca_segmentation_task.run()
        masks = pca_segmentation_task.result
        return self._pca_segmenter.combine_class_masks(masks)
