from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from bsmu.biocell.core.data.vector.shapes.cancer_span import CancerSpan
from bsmu.biocell.core.domain.gleason import GleasonGrade
from bsmu.vision.actors.shape.constrained import SnappedSpanActor, SnappedNodeActor
from bsmu.vision.actors.shape.shape import AntialiasedGraphicsPathItem

if TYPE_CHECKING:
    from PySide6.QtCore import QObject


@dataclass(frozen=True)
class GleasonStyle:
    """Visual style configuration for a Gleason grade."""
    pen_screen_width: float
    node_radius: float
    completed_color: QColor
    selected_color: QColor
    z_value: int


class CancerSpanActor(SnappedSpanActor[CancerSpan, AntialiasedGraphicsPathItem]):
    """Actor for visualizing CancerSpan with Gleason grade-specific styling."""

    _GLEASON_GRADE_TO_STYLE: dict[GleasonGrade, GleasonStyle] = {
        GleasonGrade.G3: GleasonStyle(
            pen_screen_width=13,
            node_radius=7.5,
            completed_color=QColor('#e5e50b'),
            selected_color=QColor(Qt.GlobalColor.yellow),
            z_value=3,
        ),
        GleasonGrade.G4: GleasonStyle(
            pen_screen_width=8,
            node_radius=6.5,
            completed_color=QColor('#e5790c'),
            selected_color=QColor(255, 165, 0),
            z_value=4,
        ),
        GleasonGrade.G5: GleasonStyle(
            pen_screen_width=3,
            node_radius=4.0,
            completed_color=QColor('#e60c0c'),
            selected_color=QColor(Qt.GlobalColor.red),
            z_value=5,
        ),
    }

    def __init__(
        self,
        model: CancerSpan | None = None,
        node_actor_class: type[SnappedNodeActor] = SnappedNodeActor,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(model, node_actor_class=node_actor_class, parent=parent)

    def _model_changed(self) -> None:
        super()._model_changed()

        self._apply_gleason_style()

    def _apply_gleason_style(self) -> None:
        """Apply visual styling based on the model's Gleason grade."""
        if not isinstance(self.model, CancerSpan):
            return

        style = self._GLEASON_GRADE_TO_STYLE.get(self.model.gleason_grade)
        if style is None:
            return

        # TODO: Consider adding public setters in the parent class if possible
        #  to avoid direct assignment to private parent attributes.
        self._draft_color = style.completed_color
        self._completed_color = style.completed_color
        self._subselected_color = style.completed_color
        self._selected_color = style.selected_color
        self._pen_screen_width = style.pen_screen_width
        self._node_radius = style.node_radius

        pen = self.graphics_item.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.graphics_item.setPen(pen)

        self.graphics_item.setZValue(style.z_value)
