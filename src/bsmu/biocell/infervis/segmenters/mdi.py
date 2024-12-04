from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.biocell.infervis.mdi import MdiInferencer
from bsmu.vision.core.image import MaskDrawMode

if TYPE_CHECKING:
    from bsmu.vision.core.image import Image
    from bsmu.vision.core.image.layered import LayeredImage


class MdiSegmenter(MdiInferencer):
    def _check_duplicate_mask_and_get_active_layered_image(
            self,
            mask_layer_name: str,
            show_repaint_confirmation: bool = True,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ) -> tuple[LayeredImage | None, Image | None]:

        layered_image_viewer = self._active_layered_image_viewer()
        if layered_image_viewer is None or (layered_image := layered_image_viewer.data) is None:
            return None, None

        if (show_repaint_confirmation and
                not layered_image_viewer.is_confirmed_repaint_duplicate_mask_layer(mask_layer_name, mask_draw_mode)):
            return None, None

        image_layer = layered_image.layers[0]
        return layered_image, image_layer.image
