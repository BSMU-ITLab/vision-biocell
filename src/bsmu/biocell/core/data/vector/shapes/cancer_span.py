from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.vision.core.data.vector.shapes.constrained import SnappedSpan

if TYPE_CHECKING:
    from PySide6.QtCore import QObject, QPointF

    from bsmu.biocell.core.domain.gleason import GleasonGrade
    from bsmu.vision.core.data.vector.shapes import NodeBasedShape


class CancerSpan(SnappedSpan):
    """Represent a cancer span with a specific Gleason grade."""

    def __init__(
            self,
            gleason_grade: GleasonGrade,
            origin: QPointF | None = None,
            parent_shape: NodeBasedShape | None = None,
            inherit_transform: bool = False,
            parent: QObject | None = None,
    ) -> None:
        super().__init__(
            origin=origin,
            parent_shape=parent_shape,
            inherit_transform=inherit_transform,
            parent=parent,
        )

        self._gleason_grade = gleason_grade

    @property
    def gleason_grade(self) -> GleasonGrade:
        return self._gleason_grade
