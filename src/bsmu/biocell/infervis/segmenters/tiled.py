from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bsmu.biocell.inference.segmenters.tiled import SegmentationMode
from bsmu.biocell.infervis.segmenters.mdi import MaskDrawMode
from bsmu.biocell.infervis.segmenters.mdi import MdiSegmenter
from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.image.layered import LayeredImage
from bsmu.vision.core.visibility import Visibility

if TYPE_CHECKING:
    from bsmu.vision.core.data import Data
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi
    from bsmu.vision.widgets.mdi.windows.data import DataViewerSubWindow
    from bsmu.biocell.inference.segmenters.tiled import MultipassTiledSegmenter


class MultipassTiledMdiSegmenter(MdiSegmenter):
    def __init__(self, segmenter: MultipassTiledSegmenter, mdi: Mdi):
        super().__init__(mdi)

        self._segmenter = segmenter

    @property
    def mask_foreground_class(self) -> int:
        return self._segmenter.mask_foreground_class

    @property
    def mask_background_class(self) -> int:
        return self._segmenter.mask_background_class

    def segment_async(
            self,
            mask_layer_name: str,
            segmentation_mode: SegmentationMode = SegmentationMode.HIGH_QUALITY,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        layered_image, image = self._check_duplicate_mask_and_get_active_layered_image(
            mask_layer_name, mask_draw_mode=mask_draw_mode)
        if image is None:
            return

        on_finished = partial(
            self._on_segmentation_finished,
            layered_image=layered_image,
            mask_layer_name=mask_layer_name,
            mask_draw_mode=mask_draw_mode,
        )
        self._segmenter.segment_async(image, segmentation_mode, on_finished)

    def _on_segmentation_finished(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        self.update_mask_layer(mask, layered_image, mask_layer_name, mask_draw_mode)

    def update_mask_layer_partially(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            modifiable_mask: np.ndarray | None,
    ):
        mask_layer = layered_image.layer_by_name(mask_layer_name)
        is_foreground_class = mask == self.mask_foreground_class
        if modifiable_mask is not None:
            is_foreground_class &= modifiable_mask
        mask_layer.image_pixels[is_foreground_class] = self.mask_foreground_class
        mask_layer.image.emit_pixels_modified()

    def update_mask_layer(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ) -> np.ndarray | None:
        mask_layer = layered_image.layer_by_name(mask_layer_name)
        is_modified = None
        if mask_draw_mode == MaskDrawMode.REDRAW_ALL or mask_layer is None or not mask_layer.is_image_pixels_valid:
            layered_image.add_layer_or_modify_pixels(
                mask_layer_name,
                mask,
                FlatImage,
                self._segmenter.mask_palette,
                visibility=Visibility(True, 0.75),
            )
        elif mask_draw_mode == MaskDrawMode.OVERLAY_FOREGROUND:
            is_modified = mask == self.mask_foreground_class
            mask_layer.image_pixels[is_modified] = self.mask_foreground_class
            mask_layer.image.emit_pixels_modified()
        elif mask_draw_mode == MaskDrawMode.FILL_BACKGROUND:
            is_modified = mask_layer.image_pixels == self.mask_background_class
            mask_layer.image_pixels[is_modified] = mask[is_modified]
            mask_layer.image.emit_pixels_modified()
        else:
            raise ValueError(f'Invalid MaskDrawMode: {mask_draw_mode}')
        return is_modified

    def on_data_visualized(self, data: Data, data_viewer_sub_windows: list[DataViewerSubWindow]):
        raise NotImplementedError()

        mask_layer_name = self._segmenter.segmenter.model_params.output_object_name
        if not isinstance(data, LayeredImage) or (len(data.layers) > 1 and data.layers[1].name == mask_layer_name):
            return

        self.segment_async(mask_layer_name, data)
