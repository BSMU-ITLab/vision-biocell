from __future__ import annotations

from enum import Enum

from bsmu.biocell.infervis.mdi import MdiInferencer


class MaskDrawMode(Enum):
    REDRAW_ALL = 1
    """Completely replace the existing mask with the new mask."""

    OVERLAY_FOREGROUND = 2
    """Apply the new mask only where its own pixels are equal to foreground value,
    preserving the existing mask elsewhere."""

    FILL_BACKGROUND = 3
    """Apply the new mask only on the background pixels of the existing mask, leaving other pixels unchanged."""


class MdiSegmenter(MdiInferencer):
    pass
