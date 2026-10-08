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
from bsmu.vision.core.data.raster import TILED_MASK_DOWNSAMPLE
from bsmu.vision.core.palette import Palette
from bsmu.vision.core.task import DnnTask
from bsmu.vision.dnn.inferencer import ImageModelConfig as DnnModelConfig
from bsmu.vision.dnn.segmenter import Segmenter as DnnSegmenter

if TYPE_CHECKING:
    from typing import Callable, Sequence
    from bsmu.vision.core.data.raster import Raster
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
            model_config: DnnModelConfig,
            mask_palette: Palette,
            task_storage: TaskStorage = None,
    ):
        super().__init__()

        self._model_config = model_config
        self._task_storage = task_storage

        self._mask_palette = mask_palette
        self._mask_background_class = self._mask_palette.row_index_by_name('background')
        self._mask_foreground_classes = tuple(
            self._mask_palette.row_index_by_name(name)
            for name in model_config.output_class_names
        )

        self._segmenter = DnnSegmenter(self._model_config)

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
    def mask_foreground_classes(self) -> Sequence[int]:
        return self._mask_foreground_classes

    def get_segmentation_task(
            self,
            raster: Raster,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            on_finished: Callable[[Sequence[np.ndarray]], None] | None = None,
            name: str | None = None,
    ):
        if not raster.is_tiled:
            raise ValueError(
                'Segmentation is only supported for tiled (WSI) rasters with known MPP.'
            )

        tile_step_mul = 0.5 if segmentation_mode == SegmentationMode.HIGH_QUALITY else 0.875
        tile_size = self._segmenter.model_config.input_image_size[0]
        segmentation_task = TiledSegmentationTask(
            raster,
            self._segmenter,
            tile_step_mul=tile_step_mul,
            mask_background_class=self._mask_background_class,
            mask_foreground_classes=self._mask_foreground_classes,
            background_mask=None,
            tile_weights=_tile_weights(tile_size),
            name=name,
        )
        segmentation_task.on_finished = on_finished
        return segmentation_task

    def segment_async(
            self,
            raster: Raster,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            on_finished: Callable[[Sequence[np.ndarray]], None] | None = None,
    ):
        segmentation_task_name = (
            f'{self._model_config.output_object_short_name} '
            f'{segmentation_mode.short_name_with_postfix} '
            f'[{raster.path_name}]'
        )
        segmentation_task = self.get_segmentation_task(
            raster, segmentation_mode, on_finished, name=segmentation_task_name)
        if self._task_storage is not None:
            self._task_storage.add_item(segmentation_task)
        ThreadPool.run_async_task(segmentation_task)


class PaddedArray:
    """
    Array with auto padding when slice index is out of bounds.
    Works only with tuple of not None slices. Use [slice,] for single slice.
    """

    def __init__(self, array: np.ndarray, pad: int = 255):
        self.array = array
        self.pad = pad
        self.shape = array.shape

    def get_before_after_slices(self, slices: tuple[slice]):
        slices_before = list()
        slices_after = list()
        has_padding = False
        for dim_shape, dim_slice in zip(self.shape, slices):
            start, stop = dim_slice.start, dim_slice.stop
            start_after, stop_after = 0, stop - start
            if start < 0:
                start_after = -start
                start = 0
                has_padding = True
            if stop > dim_shape:
                stop_after -= stop - dim_shape
                stop = dim_shape
                has_padding = True
            slices_before.append(slice(start, stop))
            slices_after.append(slice(start_after, stop_after))
        if not has_padding:
            return None, None
        return tuple(slices_before), tuple(slices_after)

    def __getitem__(self, slices):
        slices_from, slices_to = self.get_before_after_slices(slices)
        if slices_from is None:
            return self.array[slices]
        new_shape = []
        for arg in slices:
            new_shape.append(arg.stop - arg.start)
        if len(slices) < len(self.shape):
            new_shape.extend(self.shape[len(slices):])
        data = np.full(new_shape, self.pad, dtype=self.array.dtype)
        data[slices_to] = self.array[slices_from]
        return data

    def __setitem__(self, slices, value):
        slices_from, slices_to = self.get_before_after_slices(slices)
        if slices_from is None:
            self.array[slices] = value
        elif isinstance(value, (int, float)):
            self.array[slices_from] = value
        else:
            self.array[slices_from] = value[slices_to]

    def iadd(self, slices, value):
        """
        Inplace slice operation for adding.
        """
        slices_from, slices_to = self.get_before_after_slices(slices)
        if slices_from is None:
            self.array[slices] += value
        elif isinstance(value, (int, float)):
            self.array[slices_from] += value
        else:
            self.array[slices_from] += value[slices_to]


class TiledSegmentationTask(DnnTask):
    def __init__(
            self,
            raster: Raster,
            segmenter: DnnSegmenter,
            tile_step_mul: float = 0.5,
            mask_background_class: int = 0,
            mask_foreground_classes: int | Sequence[int] = 1,
            background_mask: np.ndarray | None = None,
            tile_weights: np.ndarray | None = None,
            name: str = '',
    ):
        super().__init__(name)

        self._raster = raster
        self._background_mask = background_mask
        self._segmenter = segmenter
        self._tile_step_mul = tile_step_mul

        self._mask_background_class = mask_background_class
        self._mask_foreground_classes = np.array(
            (mask_foreground_classes,)
            if isinstance(mask_foreground_classes, int)
            else tuple(mask_foreground_classes),
            dtype=np.uint8,
        )

        if tile_weights is not None:
            tile_weights = tile_weights[:, :, None]
        else:
            tile_weights = 1
        self._tile_weights = tile_weights

        self._segmented_tile_count: int = 0
        self._total_tile_count: int | None = 0

    @property
    def model_config(self) -> DnnModelConfig:
        return self._segmenter.model_config

    @property
    def tile_size(self) -> int:
        return self.model_config.input_image_size[0]

    @property
    def tile_step(self) -> int:
        return int(self.tile_size * self._tile_step_mul)

    def _run(self) -> Sequence[np.ndarray]:
        image = self._read_image()
        # Remove alpha-channel
        if image.shape[2] == 4:
            image = image[..., :3]

        return self._segment_tiled(image)

    def _read_image(self) -> np.ndarray:
        """Read the full image at downsampled resolution (for tiled rasters)."""
        raster = self._raster
        if raster.is_tiled:
            output_w = round(raster.shape[1] / TILED_MASK_DOWNSAMPLE)
            output_h = round(raster.shape[0] / TILED_MASK_DOWNSAMPLE)
            return raster.read_region(output_size=(output_w, output_h))
        else:
            return raster.pixels

    def _segment_tiled(self, image: np.ndarray) -> Sequence[np.ndarray]:
        logging.info(f'Segment image using {self.model_config.path.name} model '
                     f'(batch_size={self.model_config.batch_size})')
        t_total = timer()

        # Create inference session before timing inference
        t_warmup = timer()
        self._segmenter.warmup()
        logging.info(f'Warm-up (session creation): {timer() - t_warmup:.2f}s')

        t_inference_start = timer()

        batch_size = self.model_config.batch_size
        thresholds = np.array(self.model_config.mask_binarization_thresholds)[
            None, None, :]  # unsqueeze to [1, 1, classes]

        padded_image = PaddedArray(image)

        if self._background_mask is not None:
            background_r = round(image.shape[0] / self._background_mask.shape[0])
            padded_background = PaddedArray(self._background_mask, pad=1)
        else:
            background_r = 1
            padded_background = None

        tile_size = self.tile_size
        tile_step = self.tile_step
        image_pad = (tile_size - tile_step) // 2  # for checking: = tile_step
        assert tile_size % background_r == 0
        assert tile_step % background_r == 0
        assert image_pad % background_r == 0
        # Create one mask for every foreground class filled with `self._mask_background_class`,
        # because this Task can be cancelled, and then we have to return correct partial mask
        label_classes = len(self._mask_foreground_classes)
        labeled_mask = np.full(
            tuple(image.shape[:2]) + (label_classes,),
            self._mask_background_class, dtype=np.uint8,
        )
        padded_labeled_mask = PaddedArray(labeled_mask)

        tiles_row_count = (image.shape[0] + image_pad + tile_step - 1) // tile_step
        tiles_column_count = (image.shape[1] + image_pad + tile_step - 1) // tile_step
        self._total_tile_count = tiles_row_count * tiles_column_count

        preds_prev, preds_now, weights_prev, weights_now = (
            PaddedArray(np.empty(
                (tile_size, image.shape[1], label_classes if i < 2 else 1), dtype=np.float32
            ), 0) for i in range(4))

        slices_now = list()
        slices_prev = list()
        for i in range(tiles_row_count):
            i_start = -image_pad + i * tile_step
            i_stop = i_start + tile_size

            # swap current and prev and zero new current
            preds_prev, preds_now = preds_now, preds_prev
            weights_prev, weights_now = weights_now, weights_prev
            preds_now.array.fill(0)
            weights_now.array.fill(0)
            slices_prev = slices_now
            slices_now = list()

            # row of current index
            i_slice = slice(i_start, i_stop)
            i_slice_background = slice(i_start // background_r, i_stop // background_r)

            for j in range(tiles_column_count):
                j_start = -image_pad + j * tile_step
                j_stop = j_start + tile_size

                # if background is None or any not background
                if padded_background is None:
                    slices_now.append(slice(j_start, j_stop))
                elif not padded_background[i_slice_background, j_start // background_r: j_stop // background_r].all():
                    slices_now.append(slice(j_start, j_stop))

            # first predict tiles from previous iteration, batch with current iteration
            if len(slices_prev) > 0:
                to_add = min(batch_size - len(slices_prev), len(slices_now))
                batch = self.get_row_tiles(padded_image, slice(i_start - tile_step, i_stop - tile_step), slices_prev)
                batch.extend(self.get_row_tiles(padded_image, i_slice, slices_now[:to_add]))
                preds = self.run_batch(batch)
                # add previous predictions and weights
                self.add_row_weight_preds(preds_prev, weights_prev, slices_prev, preds[:len(slices_prev)])
                # add current predictions and weights
                self.add_row_weight_preds(preds_now, weights_now, slices_now[:to_add], preds[len(slices_prev):])
                # remove checked tiles
                slices_now = slices_now[to_add:]

            # then predict tiles in this iteration with batching
            batch_start = None
            for batch_start in range(0, len(slices_now), batch_size):
                batch_slices = slices_now[batch_start:batch_start + batch_size]
                batch = self.get_row_tiles(padded_image, i_slice, batch_slices)
                preds = self.run_batch(batch)
                self.add_row_weight_preds(preds_now, weights_now, batch_slices, preds)

            if batch_start is not None:
                slices_now = slices_now[batch_start + batch_size:]

            # convert predictions from previous iteration into labels
            if i > 0:
                preds_removed, preds_moving = np.split(preds_prev.array, [tile_step], axis=0)
                weights_removed, weights_moving = np.split(weights_prev.array, [tile_step], axis=0)
                i_start_removed = i_start - tile_step
                # check whether slice is out of bounds
                if i_start_removed + tile_step > 0:
                    partial_mask = self.pred_to_mask(preds_removed, weights_removed, thresholds)
                    padded_labeled_mask[
                        i_start_removed:i_start_removed + tile_step,
                    ] = partial_mask
                # move intersecting parts from previous to current
                preds_now.array[:len(preds_moving)] += preds_moving
                weights_now.array[:len(weights_moving)] += weights_moving

                self.update_progress(tiles_column_count)

        if len(slices_now) > 0:  # handle leftover batch
            batch = self.get_row_tiles(padded_image, i_slice, batch_slices)
            preds = self.run_batch(batch)
            self.add_row_weight_preds(preds_now, weights_now, batch_slices, preds)

        # handle leftover labeling
        partial_mask = self.pred_to_mask(preds_now.array, weights_now.array, thresholds)
        padded_labeled_mask[
            i_start:i_start + tile_step,
        ] = partial_mask
        self.update_progress(tiles_column_count)

        labeled_mask *= self._mask_foreground_classes

        t_inference = timer() - t_inference_start
        t_total_elapsed = timer() - t_total
        logging.info(
            f'Segmentation finished. '
            f'Tiles: {self._total_tile_count}, '
            f'Inference: {t_inference:.2f}s, '
            f'Total: {t_total_elapsed:.2f}s, '
            f'Per tile: {t_inference / max(self._total_tile_count, 1) * 1000:.1f}ms'
        )

        # Return contiguous copies to avoid performance issues with non-contiguous arrays in downstream operations.
        return [np.ascontiguousarray(m) for m in np.moveaxis(labeled_mask, 2, 0)]

    def get_row_tiles(self, image: PaddedArray, row_slice: slice, slices: list[slice]):
        tiles = []
        for j_slice in slices:
            tiles.append(image[row_slice, j_slice])
        return tiles

    def add_row_tiles(self, image: PaddedArray, row_slice: slice, slices: list[slice], items: list[np.ndarray]):
        for j_slice, el2 in zip(slices, items):
            image.iadd((row_slice, j_slice), el2)

    def add_row_weight_preds(
            self, accumulated_preds: PaddedArray, weights: PaddedArray, slices: list[slice], preds: list[np.ndarray]):
        row_slice = slice(0, accumulated_preds.shape[0])
        self.add_row_tiles(accumulated_preds, row_slice, slices, [el * self._tile_weights for el in preds])
        self.add_row_tiles(weights, row_slice, slices, [self._tile_weights] * len(preds))

    def update_progress(self, count: int):
        self._segmented_tile_count += count
        self._change_step_progress(self._segmented_tile_count, self._total_tile_count)

    def run_batch(self, batch: list[np.ndarray]):
        preds = self._segmenter.segment_batch_without_postresize(batch)
        if len(self._mask_foreground_classes) == 1:
            preds = [el[:, :, None] for el in preds]
        elif self.model_config.channels_axis == 0:
            preds = [np.moveaxis(el, 0, 2) for el in preds]
        return preds

    def pred_to_mask(self, preds: np.ndarray, weights: np.ndarray, thresholds: np.ndarray):
        full_predictions = preds / np.maximum(weights, 1e-9)
        binary_predictions = np.where(weights > 0, full_predictions > thresholds, 0).astype(np.uint8)
        return binary_predictions


class TiledSegmentationTask_OLD(DnnTask):
    def __init__(
            self,
            image: np.ndarray,
            segmenter: DnnSegmenter,
            extra_pads: Sequence[float] = (0, 0),
            binarize_mask: bool = True,
            mask_background_class: int = 0,
            mask_foreground_classes: int | Sequence[int] = 1,
            tile_weights: np.ndarray | None = None,
            name: str = '',
    ):
        super().__init__(name)

        self._image = image
        self._segmenter = segmenter
        self._extra_pads = extra_pads
        self._binarize_mask = binarize_mask
        self._mask_background_class = mask_background_class
        self._mask_foreground_classes = (
            (mask_foreground_classes,)
            if isinstance(mask_foreground_classes, int)
            else tuple(mask_foreground_classes)
        )
        self._tile_weights = tile_weights

        self._segmented_tile_count: int = 0
        self._tile_row_count: int | None = None
        self._tile_col_count: int | None = None
        self._total_tile_count: int | None = None
        self._weights: np.ndarray | None = None

    @property
    def model_config(self) -> DnnModelConfig:
        return self._segmenter.model_config

    @property
    def tile_size(self) -> int:
        return self.model_config.input_image_size[0]

    @property
    def weights(self) -> np.ndarray | None:
        """Tile weights after unpadding. Same for all classes."""
        return self._weights

    def _run(self) -> Sequence[np.ndarray]:
        return self._segment_tiled()

    def _segment_tiled(self) -> Sequence[np.ndarray]:
        logging.info(f'Segment image using {self.model_config.path.name} model with {self._extra_pads} extra pads')
        segmentation_start = timer()

        image = self._image
        # Remove alpha-channel
        if image.shape[2] == 4:
            image = image[..., :3]

        tile_size = self.tile_size
        padded_image, pads = self._padded_image_to_tile(image, tile_size, extra_pads=self._extra_pads)
        # Create one mask for every foreground class filled with `self._mask_background_class`,
        # because this Task can be cancelled, and then we have to return correct partial mask
        num_classes = len(self._mask_foreground_classes)
        padded_masks = [
            np.full(shape=padded_image.shape[:-1], fill_value=self._mask_background_class, dtype=np.float32)
            for _ in range(num_classes)
        ]

        tiled_image = self._tiled_image(padded_image, tile_size)

        self._tile_row_count = tiled_image.shape[0]
        self._tile_col_count = tiled_image.shape[1]
        self._total_tile_count = self._tile_row_count * self._tile_col_count

        segment_tiled_args = (tiled_image, tile_size, padded_masks)
        if self._segmenter.model_config.batch_size == 1:
            self._segment_tiled_individually(*segment_tiled_args)
        else:
            self._segment_tiled_in_batches(*segment_tiled_args)

        masks = []
        for i, padded_mask in enumerate(padded_masks):
            mask = self._unpad_image(padded_mask, pads)
            if self._binarize_mask:
                threshold = self.model_config.mask_binarization_thresholds[i]
                foreground_class = self._mask_foreground_classes[i]
                mask = (mask > threshold).astype(np.uint8)
                mask *= foreground_class
            masks.append(mask)

        if self._tile_weights is not None:
            padded_weights = np.tile(self._tile_weights, reps=(tiled_image.shape[:2]))
            self._weights = self._unpad_image(padded_weights, pads)

        logging.info(f'Segmentation finished. Elapsed time: {timer() - segmentation_start:.2f}')
        return masks

    def _segment_tiled_individually(
            self, tiled_image: np.ndarray, tile_size: int, padded_masks: list[np.ndarray]) -> None:
        for tile_row in range(self._tile_row_count):
            for tile_col in range(self._tile_col_count):
                # if self._is_cancelled:
                #     return

                tile = tiled_image[tile_row, tile_col]
                tile_mask = self._segmenter.segment(tile)

                row = tile_row * tile_size
                col = tile_col * tile_size

                # Break down the multi-channel output into separate 2D masks
                if tile_mask.ndim == 2:
                    padded_masks[0][row:(row + tile_size), col:(col + tile_size)] = tile_mask
                else:
                    channels_axis = self.model_config.channels_axis
                    for class_index in range(len(padded_masks)):
                        tile_channel_mask = np.take(tile_mask, class_index, axis=channels_axis)
                        padded_masks[class_index][row:(row + tile_size), col:(col + tile_size)] = tile_channel_mask

                self._segmented_tile_count += 1
                self._change_step_progress(self._segmented_tile_count, self._total_tile_count)

    def _segment_tiled_in_batches(
            self, tiled_image: np.ndarray, tile_size: int, padded_masks: list[np.ndarray]) -> None:
        batch_size = self._segmenter.model_config.batch_size

        tile_batch = []
        tile_rc_batch = []

        for tile_row in range(self._tile_row_count):
            for tile_col in range(self._tile_col_count):
                # if self._is_cancelled:
                #     return

                tile = tiled_image[tile_row, tile_col]
                tile_batch.append(tile)
                tile_rc_batch.append((tile_row, tile_col))

                if len(tile_batch) == batch_size:
                    self._segment_tile_batch(tile_batch, tile_rc_batch, tile_size, padded_masks)

        # Process any remaining tiles in the last batch
        if tile_batch:
            self._segment_tile_batch(tile_batch, tile_rc_batch, tile_size, padded_masks)

    def _segment_tile_batch(
            self,
            tile_batch: list[np.ndarray],
            tile_rc_batch: list[tuple[int, int]],
            tile_size: int,
            padded_masks: list[np.ndarray],
    ) -> None:
        tile_mask_batch = self._segmenter.segment_batch_without_postresize(tile_batch)

        for tile_mask, (tile_mask_row, tile_mask_col) in zip(tile_mask_batch, tile_rc_batch):
            mask_row = tile_mask_row * tile_size
            mask_col = tile_mask_col * tile_size

            if tile_mask.ndim == 2:
                padded_masks[0][mask_row:(mask_row + tile_size), mask_col:(mask_col + tile_size)] = tile_mask
            else:
                channels_axis = self.model_config.channels_axis
                for class_index in range(len(padded_masks)):
                    tile_channel_mask = np.take(tile_mask, class_index, axis=channels_axis)
                    padded_masks[class_index][
                        mask_row:(mask_row + tile_size), mask_col:(mask_col + tile_size)] = tile_channel_mask

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

    def _run(self) -> Sequence[np.ndarray]:
        return self._segment_multipass_tiled()

    def _segment_multipass_tiled(self) -> Sequence[np.ndarray]:
        assert self._segmentation_profile.extra_pads_sequence, '`extra_pads_sequence` should not be empty'

        masks = None
        weighted_masks = None
        weight_sum = None
        for self._finished_subtask_count, extra_pads in enumerate(self._segmentation_profile.extra_pads_sequence):
            tiled_segmentation_task = TiledSegmentationTask(
                self._image,
                self._segmentation_profile.segmenter,
                extra_pads,
                binarize_mask=False,
                mask_background_class=self._segmentation_profile.mask_background_class,
                mask_foreground_classes=self._segmentation_profile.mask_foreground_classes,
                tile_weights=self._segmentation_profile.tile_weights,
            )
            tiled_segmentation_task.progress_changed.connect(self._on_segmentation_subtask_progress_changed)
            tiled_segmentation_task.run()
            masks = tiled_segmentation_task.result
            mask_weights = tiled_segmentation_task.weights
            if len(self._segmentation_profile.extra_pads_sequence) > 1:
                if weighted_masks is None:
                    weighted_masks = [mask * mask_weights for mask in masks]
                    # Intentionally no .copy() here: tiled_segmentation_task dies at the end of the iteration,
                    # so in-place mutation of its weights array is safe and avoids an extra memory copy.
                    weight_sum = mask_weights
                else:
                    for i, mask in enumerate(masks):
                        weighted_masks[i] += mask * mask_weights
                    weight_sum += mask_weights

        # `weight_sum` accumulates the sum of weights for each pixel across all masks.
        # When dividing `weighted_mask` by `weight_sum`, we normalize the mask values.
        # It's essential that no element in `weight_sum` is zero to prevent division by zero errors.
        if weighted_masks is not None:
            masks = [weighted_mask / weight_sum for weighted_mask in weighted_masks]

        for i, (mask, threshold, foreground_class) in enumerate(
                zip(masks, self._segmentation_profile.mask_binarization_thresholds,
                    self._segmentation_profile.mask_foreground_classes)
        ):
            mask = (mask > threshold).astype(np.uint8)
            mask *= foreground_class
            masks[i] = mask

        return masks

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
    mask_foreground_classes: int | Sequence[int] = 1
    tile_weights: np.ndarray | None = None

    def __post_init__(self):
        self.mask_foreground_classes = (
            (self.mask_foreground_classes,)
            if isinstance(self.mask_foreground_classes, int)
            else tuple(self.mask_foreground_classes)
        )

        if self.tile_weights is None:
            self.tile_weights = _tile_weights(self.tile_size)

    @property
    def extra_pads_sequence(self) -> Sequence[Sequence[float]]:
        return ([(0, 0), (1, 1), (1, 0), (0, 1)]
                if self.segmentation_mode is SegmentationMode.HIGH_QUALITY
                else [(0, 0)])

    @property
    def tile_size(self) -> int:
        return self.segmenter.model_config.input_image_size[0]

    @property
    def mask_binarization_thresholds(self) -> Sequence[float]:
        return self.segmenter.model_config.mask_binarization_thresholds


class MulticlassMultipassTiledSegmentationTask(DnnTask):
    def __init__(
            self,
            raster: Raster,
            tile_segmenters: Sequence[MultipassTiledSegmenter],
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            name: str = '',
    ):
        super().__init__(name)

        self._raster = raster
        self._tile_segmenters = tile_segmenters
        self._segmentation_mode = segmentation_mode

        self._finished_subtask_count = 0

    def _run(self) -> Sequence[Sequence[np.ndarray]]:
        return self._segment_multiclass_multipass_tiled()

    def _segment_multiclass_multipass_tiled(self) -> Sequence[Sequence[np.ndarray]]:
        masks_per_segmenter = []
        for self._finished_subtask_count, tile_segmenter in enumerate(self._tile_segmenters):
            tiled_segmentation_task = tile_segmenter.get_segmentation_task(self._raster, self._segmentation_mode)
            tiled_segmentation_task.progress_changed.connect(self._on_segmentation_subtask_progress_changed)
            tiled_segmentation_task.run()
            class_masks = tiled_segmentation_task.result
            masks_per_segmenter.append(class_masks)
        return masks_per_segmenter

    def _on_segmentation_subtask_progress_changed(self, progress: float):
        self._change_subtask_based_progress(self._finished_subtask_count, len(self._tile_segmenters), progress)
