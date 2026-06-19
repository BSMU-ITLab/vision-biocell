from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.biocell.infervis.mdi import MdiInferencer
from bsmu.vision.core.data.raster import MaskDrawMode
from bsmu.vision.core.layers import RasterLayer

if TYPE_CHECKING:
    from bsmu.vision.core.data.layered import LayeredData
    from bsmu.vision.core.data.raster import Raster


class MdiSegmenter(MdiInferencer):
    def _check_duplicate_mask_and_get_active_layered_data(
            self,
            mask_layer_name: str,
            show_repaint_confirmation: bool = True,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ) -> tuple[LayeredData | None, Raster | None]:

        layered_data_viewer = self._active_layered_data_viewer()
        if layered_data_viewer is None or (layered_data := layered_data_viewer.data) is None:
            return None, None

        if (show_repaint_confirmation and
                not layered_data_viewer.is_confirmed_repaint_duplicate_mask_layer(mask_layer_name, mask_draw_mode)):
            return None, None

        raster_layer = layered_data.layers[0]
        if not isinstance(raster_layer, RasterLayer):
            return layered_data, None

        return layered_data, raster_layer.data
