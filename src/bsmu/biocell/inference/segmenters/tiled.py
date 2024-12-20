from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from timeit import default_timer as timer
from typing import TYPE_CHECKING

import numpy as np
import skimage.io
import skimage.util
from PySide6.QtCore import QObject

from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.palette import Palette
from bsmu.vision.core.task import DnnTask
from bsmu.vision.dnn.inferencer import ImageModelParams as DnnModelParams
from bsmu.vision.dnn.segmenter import Segmenter as DnnSegmenter

if TYPE_CHECKING:
    from typing import Callable, Sequence
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.storages.task import TaskStorage


class SegmentationMode(Enum):
    HIGH_QUALITY = 1
    FAST = 2

    @property
    def display_name(self) -> str:
        return _SEGMENTATION_MODE_TO_DISPLAY_SHORT_NAME[self].display_name

    @property
    def display_name_with_postfix(self) -> str:
        return f'{self.display_name} Segmentation'

    @property
    def short_name(self) -> str:
        return _SEGMENTATION_MODE_TO_DISPLAY_SHORT_NAME[self].short_name

    @property
    def short_name_with_postfix(self) -> str:
        return f'{self.short_name}-Seg'


@dataclass
class DisplayShortName:
    display_name: str
    short_name: str


_SEGMENTATION_MODE_TO_DISPLAY_SHORT_NAME = {
    SegmentationMode.HIGH_QUALITY: DisplayShortName('High-Quality', 'HQ'),
    SegmentationMode.FAST: DisplayShortName('Fast', 'F'),
}


class MultipassTiledSegmenter(QObject):
    def __init__(
            self,
            model_params: DnnModelParams,
            mask_palette: Palette,
            mask_foreground_class_name: str = 'foreground',
            task_storage: TaskStorage = None,
    ):
        super().__init__()

        self._model_params = model_params
        self._task_storage = task_storage

        self._mask_palette = mask_palette
        self._mask_background_class = self._mask_palette.row_index_by_name('background')
        self._mask_foreground_class = self._mask_palette.row_index_by_name(mask_foreground_class_name)

        self._segmenter = DnnSegmenter(self._model_params)

    @property
    def segmenter(self) -> DnnSegmenter:
        return self._segmenter

    @property
    def mask_palette(self) -> Palette:
        return self._mask_palette

    @property
    def mask_background_class(self) -> int:
        return self._mask_background_class

    @property
    def mask_foreground_class(self) -> int:
        return self._mask_foreground_class

    def segment_async(
            self,
            image: Image,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            on_finished: Callable[[np.ndarray], None] | None = None,
    ):
        segmentation_profile = MultipassTiledSegmentationProfile(
            self._segmenter, segmentation_mode, self._mask_background_class, self._mask_foreground_class)
        segmentation_task_name = (
            f'{self._model_params.output_object_short_name} '
            f'{segmentation_mode.short_name_with_postfix} '
            f'[{image.path_name}]'
        )
        segmentation_task = MultipassTiledSegmentationTask(image.pixels, segmentation_profile, segmentation_task_name)
        segmentation_task.on_finished = on_finished

        if self._task_storage is not None:
            self._task_storage.add_item(segmentation_task)
        ThreadPool.run_async_task(segmentation_task)


class TiledSegmentationTask(DnnTask):
    def __init__(
            self,
            image: np.ndarray,
            segmenter: DnnSegmenter,
            extra_pads: Sequence[float] = (0, 0),
            binarize_mask: bool = True,
            mask_background_class: int = 0,
            mask_foreground_class: int = 1,
            tile_weights: np.ndarray | None = None,
            name: str = '',
    ):
        super().__init__(name)

        self._image = image
        self._segmenter = segmenter
        self._extra_pads = extra_pads
        self._binarize_mask = binarize_mask
        self._mask_background_class = mask_background_class
        self._mask_foreground_class = mask_foreground_class
        self._tile_weights = tile_weights

        self._segmented_tile_count: int = 0
        self._tile_row_count: int | None = None
        self._tile_col_count: int | None = None
        self._total_tile_count: int | None = None

    @property
    def model_params(self) -> DnnModelParams:
        return self._segmenter.model_params

    @property
    def tile_size(self) -> int:
        return self.model_params.input_image_size[0]

    def _run(self) -> tuple[np.ndarray, np.ndarray]:
        return self._segment_tiled()

    def _segment_tiled(self) -> tuple[np.ndarray, np.ndarray]:
        logging.info(f'Segment image using {self.model_params.path.name} model with {self._extra_pads} extra pads')
        segmentation_start = timer()

        image = self._image
        # Remove alpha-channel
        if image.shape[2] == 4:
            image = image[..., :3]

        tile_size = self.tile_size
        padded_image, pads = self._padded_image_to_tile(image, tile_size, extra_pads=self._extra_pads)
        # Create a mask filled with `self._mask_background_class`, because this Task can be cancelled, and then
        # we have to return correct partial mask
        padded_mask = np.full(shape=padded_image.shape[:-1], fill_value=self._mask_background_class, dtype=np.float32)

        tiled_image = self._tiled_image(padded_image, tile_size)

        self._tile_row_count = tiled_image.shape[0]
        self._tile_col_count = tiled_image.shape[1]
        self._total_tile_count = self._tile_row_count * self._tile_col_count

        segment_tiled_args = (tiled_image, tile_size, padded_mask)
        if self._segmenter.model_params.batch_size == 1:
            self._segment_tiled_individually(*segment_tiled_args)
        else:
            self._segment_tiled_in_batches(*segment_tiled_args)

        mask = self._unpad_image(padded_mask, pads)

        weights = None
        if self._tile_weights is not None:
            padded_weights = np.tile(self._tile_weights, reps=(tiled_image.shape[:2]))
            weights = self._unpad_image(padded_weights, pads)

        if self._binarize_mask:
            mask = (mask > self.model_params.mask_binarization_threshold).astype(np.uint8)
            mask *= self._mask_foreground_class

        logging.info(f'Segmentation finished. Elapsed time: {timer() - segmentation_start:.2f}')
        return mask, weights

    def _segment_tiled_individually(self, tiled_image: np.ndarray, tile_size: int, padded_mask: np.ndarray):
        for tile_row in range(self._tile_row_count):
            for tile_col in range(self._tile_col_count):
                # if self._is_cancelled:
                #     return mask, weights

                tile = tiled_image[tile_row, tile_col]
                tile_mask = self._segmenter.segment(tile)

                row = tile_row * tile_size
                col = tile_col * tile_size
                padded_mask[row:(row + tile_size), col:(col + tile_size)] = tile_mask

                self._segmented_tile_count += 1
                self._change_step_progress(self._segmented_tile_count, self._total_tile_count)

    def _segment_tiled_in_batches(self, tiled_image: np.ndarray, tile_size: int, padded_mask: np.ndarray):
        batch_size = self._segmenter.model_params.batch_size

        tile_batch = []
        tile_rc_batch = []

        for tile_row in range(self._tile_row_count):
            for tile_col in range(self._tile_col_count):
                # if self._is_cancelled:
                #     return mask, weights

                tile = tiled_image[tile_row, tile_col]
                tile_batch.append(tile)
                tile_rc_batch.append((tile_row, tile_col))

                if len(tile_batch) == batch_size:
                    self._segment_tile_batch(tile_batch, tile_rc_batch, tile_size, padded_mask)

        # Process any remaining tiles in the last batch
        if tile_batch:
            self._segment_tile_batch(tile_batch, tile_rc_batch, tile_size, padded_mask)

    def _segment_tile_batch(
            self,
            tile_batch: list[np.ndarray],
            tile_rc_batch: list[tuple[int, int]],
            tile_size: int,
            padded_mask: np.ndarray,
    ):
        tile_mask_batch = self._segmenter.segment_batch_without_postresize(tile_batch)

        for tile_mask, (tile_mask_row, tile_mask_col) in zip(tile_mask_batch, tile_rc_batch):
            mask_row = tile_mask_row * tile_size
            mask_col = tile_mask_col * tile_size

            padded_mask[mask_row:(mask_row + tile_size), mask_col:(mask_col + tile_size)] = tile_mask

        self._segmented_tile_count += len(tile_batch)
        self._change_step_progress(self._segmented_tile_count, self._total_tile_count)

        tile_batch.clear()
        tile_rc_batch.clear()

    @staticmethod
    def _padded_image_to_tile(
            image: np.ndarray,
            tile_size: int,
            extra_pads: Sequence[float] = (0, 0),
            pad_value=255
    ) -> tuple[np.ndarray, tuple]:
        """
        Returns a padded |image| so that its dimensions are evenly divisible by the |tile_size| and pads
        :param extra_pads: additionally adds the |tile_size| multiplied by |extra_pads| to get shifted tiles
        """
        rows, cols, channels = image.shape

        pad_rows = (-rows % tile_size) + extra_pads[0] * tile_size
        pad_cols = (-cols % tile_size) + extra_pads[1] * tile_size

        pad_rows_half = pad_rows // 2
        pad_cols_half = pad_cols // 2

        pads = ((pad_rows_half, pad_rows - pad_rows_half), (pad_cols_half, pad_cols - pad_cols_half), (0, 0))
        image = np.pad(image, pads, constant_values=pad_value)
        return image, pads

    @staticmethod
    def _unpad_image(image: np.ndarray, pads: tuple) -> np.ndarray:
        return image[
               pads[0][0]:image.shape[0] - pads[0][1],
               pads[1][0]:image.shape[1] - pads[1][1],
               ]

    @staticmethod
    def _tiled_image(image: np.ndarray, tile_size: int) -> np.ndarray:
        tile_shape = (tile_size, tile_size, image.shape[-1])
        tiled = skimage.util.view_as_blocks(image, tile_shape)
        return tiled.squeeze(axis=2)


class MultipassTiledSegmentationTask(DnnTask):
    def __init__(
            self,
            image: np.ndarray,
            segmentation_profile: MultipassTiledSegmentationProfile,
            name: str = '',
    ):
        super().__init__(name)

        self._image = image
        self._segmentation_profile = segmentation_profile

        self._finished_subtask_count = 0

    def _run(self) -> np.ndarray:
        return self._segment_multipass_tiled()

    def _segment_multipass_tiled(self) -> np.ndarray:
        assert self._segmentation_profile.extra_pads_sequence, '`extra_pads_sequence` should not be empty'

        mask = None
        weighted_mask = None
        weight_sum = None
        for self._finished_subtask_count, extra_pads in enumerate(self._segmentation_profile.extra_pads_sequence):
            tiled_segmentation_task = TiledSegmentationTask(
                self._image,
                self._segmentation_profile.segmenter,
                extra_pads,
                False,
                self._segmentation_profile.mask_background_class,
                self._segmentation_profile.mask_foreground_class,
                self._segmentation_profile.tile_weights,
            )
            tiled_segmentation_task.progress_changed.connect(self._on_segmentation_subtask_progress_changed)
            tiled_segmentation_task.run()
            mask, mask_weights = tiled_segmentation_task.result
            if len(self._segmentation_profile.extra_pads_sequence) > 1:
                if weighted_mask is None:
                    weighted_mask = mask * mask_weights
                    weight_sum = mask_weights
                else:
                    weighted_mask += mask * mask_weights
                    weight_sum += mask_weights

        # `weight_sum` accumulates the sum of weights for each pixel across all masks.
        # When dividing `weighted_mask` by `weight_sum`, we normalize the mask values.
        # It's essential that no element in `weight_sum` is zero to prevent division by zero errors.
        if weighted_mask is not None:
            mask = weighted_mask / weight_sum

        mask = (mask > self._segmentation_profile.mask_binarization_threshold).astype(np.uint8)
        mask *= self._segmentation_profile.mask_foreground_class

        return mask

    def _on_segmentation_subtask_progress_changed(self, progress: float):
        self._change_subtask_based_progress(
            self._finished_subtask_count, len(self._segmentation_profile.extra_pads_sequence), progress)


def _tile_weights(tile_size: int) -> np.ndarray:
    """
    Returns tile weights, where maximum weights (equal to 1) are in the center of the tile,
    and the weights gradually decreases to zero towards the edges of the tile
    E.g.: |tile_size| is equal to 6:
    np.array([[0. , 0. , 0. , 0. , 0. , 0. ],
              [0. , 0.5, 0.5, 0.5, 0.5, 0. ],
              [0. , 0.5, 1. , 1. , 0.5, 0. ],
              [0. , 0.5, 1. , 1. , 0.5, 0. ],
              [0. , 0.5, 0.5, 0.5, 0.5, 0. ],
              [0. , 0. , 0. , 0. , 0. , 0. ]], dtype=float16)
    """
    assert tile_size % 2 == 0, 'Current method version can work only with even tile size'
    max_int_weight = (tile_size // 2) - 1
    int_weights = list(range(max_int_weight + 1))
    int_weights = int_weights + int_weights[::-1]
    row_int_weights = np.expand_dims(int_weights, 0)
    col_int_weights = np.expand_dims(int_weights, 1)
    tile_int_weights = np.minimum(row_int_weights, col_int_weights)
    return (tile_int_weights / max_int_weight).astype(np.float16)


@dataclass
class MultipassTiledSegmentationProfile:
    segmenter: DnnSegmenter
    segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY
    mask_background_class: int = 0
    mask_foreground_class: int = 1
    tile_weights: np.ndarray | None = None

    def __post_init__(self):
        if self.tile_weights is None:
            self.tile_weights = _tile_weights(self.tile_size)

    @property
    def extra_pads_sequence(self) -> Sequence[Sequence[float]]:
        return ([(0, 0), (1, 1), (1, 0), (0, 1)]
                if self.segmentation_mode is SegmentationMode.HIGH_QUALITY
                else [(0, 0)])

    @property
    def tile_size(self) -> int:
        return self.segmenter.model_params.input_image_size[0]

    @property
    def mask_binarization_threshold(self) -> float:
        return self.segmenter.model_params.mask_binarization_threshold


class MulticlassMultipassTiledSegmentationTask(DnnTask):
    def __init__(
            self,
            image: np.ndarray,
            segmentation_profiles: Sequence[MultipassTiledSegmentationProfile],
            name: str = '',
    ):
        super().__init__(name)

        self._image = image
        self._segmentation_profiles = segmentation_profiles

        self._finished_subtask_count = 0

    def _run(self) -> Sequence[np.ndarray]:
        return self._segment_multiclass_multipass_tiled()

    def _segment_multiclass_multipass_tiled(self) -> Sequence[np.ndarray]:
        masks = []
        for self._finished_subtask_count, segmentation_profile in enumerate(self._segmentation_profiles):
            tiled_segmentation_task = MultipassTiledSegmentationTask(self._image, segmentation_profile)
            tiled_segmentation_task.progress_changed.connect(self._on_segmentation_subtask_progress_changed)
            tiled_segmentation_task.run()
            mask = tiled_segmentation_task.result
            masks.append(mask)
        return masks

    def _on_segmentation_subtask_progress_changed(self, progress: float):
        self._change_subtask_based_progress(self._finished_subtask_count, len(self._segmentation_profiles), progress)
