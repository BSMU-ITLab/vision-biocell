from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal, QObject, QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QVBoxLayout, QRadioButton, QGroupBox, QHBoxLayout, QLayout)

from bsmu.vision.actors.shape import PolylineActor
from bsmu.vision.actors.shape.constrained import SnappedSpanActor, SnappedNodeActor
from bsmu.vision.actors.shape.shape import AntialiasedGraphicsPathItem  ## temp
from bsmu.vision.core.data.vector.shapes import NodeBasedShape
from bsmu.vision.core.data.vector.shapes import Polyline
from bsmu.vision.core.data.vector.shapes.constrained import SnappedSpan
from bsmu.vision.plugins.tools import (
    ViewerToolSettingsWidget, ViewerTool, ViewerToolSettings, ViewerToolPlugin, CursorConfig)
from bsmu.vision.plugins.tools.layered import LayeredDataViewerTool, LayeredDataViewerToolSettings
from bsmu.vision.plugins.tools.polyline import PolylineTool, POLYLINE_CURSOR_CONFIG
from bsmu.vision.plugins.tools.snapped_span import SnappedSpanTool, SnappedSpanFactory

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.palette.settings import PalettePackSettings, PalettePackSettingsPlugin
    from bsmu.vision.plugins.undo import UndoManager, UndoPlugin
    from bsmu.vision.plugins.windows.main import MainWindowPlugin
    from bsmu.vision.widgets.viewers.layered import LayeredDataViewer


@dataclass(frozen=True)
class PolylineAttachedPointInfo:          ## TEMP
    """A point attached to a specific segment of a polyline."""
    point: QPointF
    polyline: Polyline
    segment_index: int  # Index of the polyline segment this point belongs to


class AnnotationMode(Enum):
    TISSUE = 1
    CANCER = 2


from bsmu.vision.actors.shape.registry import register_shape_actor


class GleasonGrade(Enum):
    G3 = 3
    G4 = 4
    G5 = 5


class CancerSpan(SnappedSpan):
    def __init__(
            self,
            gleason_grade: GleasonGrade,
            origin: QPointF | None = None,
            parent_shape: NodeBasedShape | None = None,
            inherit_transform: bool = False,
            parent: QObject | None = None,
    ):
        super().__init__(origin=origin, parent_shape=parent_shape, inherit_transform=inherit_transform, parent=parent)

        self._gleason_grade = gleason_grade

    @property
    def gleason_grade(self) -> GleasonGrade:
        return self._gleason_grade


class CancerSpanActor(SnappedSpanActor[CancerSpan, AntialiasedGraphicsPathItem]):
    DARK_FACTOR = 110
    _GRADE_STYLE_MAP = {
        GleasonGrade.G3: {
            # "pen": QPen(QColor(Qt.GlobalColor.yellow).darker(DARK_FACTOR), 11, s=Qt.PenStyle.SolidLine, c=Qt.PenCapStyle.RoundCap),
            "pen_width": 13,#11,
            "node_radius": 7.5,
            # "completed_color": QColor(Qt.GlobalColor.yellow).darker(DARK_FACTOR),
            "completed_color": QColor('#e5e50b'),
            "selected_color": QColor(Qt.GlobalColor.yellow),
            'z_value': 3,
        },
        GleasonGrade.G4: {
            # "pen": QPen(QColor(255, 165, 0).darker(DARK_FACTOR), 7, s=Qt.PenStyle.SolidLine, c=Qt.PenCapStyle.RoundCap),
            "pen_width": 8,#7,
            "node_radius": 6.5,
            # "completed_color": QColor(255, 165, 0).darker(DARK_FACTOR),
            "completed_color": QColor('#e5790c'),
            "selected_color": QColor(255, 165, 0),
            'z_value': 4,
        },
        GleasonGrade.G5: {
            # "pen": QPen(QColor(Qt.GlobalColor.red).darker(DARK_FACTOR), 3, s=Qt.PenStyle.SolidLine, c=Qt.PenCapStyle.RoundCap),
            "pen_width": 3,
            "node_radius": 4.0,
            # "completed_color": QColor(Qt.GlobalColor.red).darker(DARK_FACTOR),
            "completed_color": QColor('#e60c0c'),
            "selected_color": QColor(Qt.GlobalColor.red),
            'z_value': 5,
        },
    }

    def __init__(
        self,
        model: CancerSpan | None = None,
        node_actor_class: type[SnappedNodeActor] = SnappedNodeActor,
        parent: QObject | None = None,
    ):
        super().__init__(model, node_actor_class=node_actor_class, parent=parent)

    def _model_changed(self) -> None:
        super()._model_changed()

        self._apply_gleason_style()

    def _apply_gleason_style(self) -> None:
        if not isinstance(self.model, CancerSpan):
            return

        grade = self.model.gleason_grade
        style = self._GRADE_STYLE_MAP.get(grade)
        if not style:
            return

        self._draft_color = style["completed_color"]
        self._completed_color = style["completed_color"]
        self._subselected_color = style["completed_color"]
        self._selected_color = style["selected_color"]
        self._pen_screen_width = style["pen_width"]
        self._node_radius = style["node_radius"]

        pen = self.graphics_item.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        self.graphics_item.setPen(pen)

        self.graphics_item.setZValue(style['z_value'])


register_shape_actor(CancerSpan, CancerSpanActor)



class CancerSpanFactory(SnappedSpanFactory):
    def __init__(self, gleason_grade: GleasonGrade):
        self._gleason_grade = gleason_grade

    def create_span(self, parent_shape: NodeBasedShape, **kwargs) -> CancerSpan:
        return CancerSpan(self._gleason_grade, parent_shape=parent_shape)


class CancerSegmentTool(LayeredDataViewerTool):
    def __init__(self, viewer: LayeredDataViewer, undo_manager: UndoManager, settings: CancerSegmentToolSettings):
        super().__init__(viewer, undo_manager, settings)

        self._polyline_subtool = PolylineTool(viewer, undo_manager, settings)
        self._polyline_span_subtool = SnappedSpanTool(
            CancerSpanFactory(settings.gleason_grade), viewer, undo_manager, settings)

        self._annotation_mode_to_subtool = {
            AnnotationMode.TISSUE: self._polyline_subtool,
            AnnotationMode.CANCER: self._polyline_span_subtool,
        }

        self._active_subtool: ViewerTool | None = None  # Sub tool (using composition)

        self._polyline_span_subtool.span_created.connect(self._print_cancer_percentage)

    def activate(self):
        self.settings.gleason_grade_changed.connect(self._change_style_to_match_gleason_grade)
        self.settings.annotation_mode_changed.connect(self._activate_tool_to_match_annotation_mode)

        self._change_style_to_match_gleason_grade(self.settings.gleason_grade)
        self._activate_tool_to_match_annotation_mode(self.settings.annotation_mode)

    def deactivate(self):
        if self._active_subtool is not None:
            self._active_subtool.deactivate()
            self._active_subtool = None

        self.settings.annotation_mode_changed.disconnect(self._activate_tool_to_match_annotation_mode)
        self.settings.gleason_grade_changed.disconnect(self._change_style_to_match_gleason_grade)

    def _change_style_to_match_gleason_grade(self, gleason_grade: GleasonGrade):
        self._polyline_span_subtool.span_factory = CancerSpanFactory(gleason_grade)

    def _activate_tool_to_match_annotation_mode(self, annotation_mode: AnnotationMode):
        if self._active_subtool is not None:
            self._active_subtool.deactivate()

        self._active_subtool = self._annotation_mode_to_subtool[annotation_mode]
        self._active_subtool.activate()

    def _print_cancer_percentage(self):
        gleason_grade_to_total_length = defaultdict(float)
        polylines = []

        for graphics_item in self.viewer._graphics_scene.items():
            if isinstance(graphics_item, PolylineActor):
                polylines.append(graphics_item.polyline)
            elif isinstance(graphics_item, SnappedSpanActor):
                cancer_span = graphics_item.polyline_span
                gleason_grade_to_total_length[cancer_span.gleason_grade] += cancer_span.length

        total_tissue_length = sum(p.length for p in polylines)
        gleason_grade_to_tissue_percentage = {
            grade: (length / total_tissue_length) * 100
            for grade, length in gleason_grade_to_total_length.items()
        } if total_tissue_length > 0 else {}

        if gleason_grade_to_tissue_percentage:
            print('Gleason Grade Distribution (% of tissue):')
            for grade in GleasonGrade:
                percentage = gleason_grade_to_tissue_percentage.get(grade, 0)
                print(f'{grade.name}: {percentage:.1f}%')
        else:
            print('No tissue area available for percentage calculation')


class CancerSegmentToolSettings(LayeredDataViewerToolSettings):
    annotation_mode_changed = Signal(AnnotationMode)
    gleason_grade_changed = Signal(GleasonGrade)

    def __init__(
            self,
            layers_props: dict,
            palette_pack_settings: PalettePackSettings,
            cursor_config: CursorConfig = POLYLINE_CURSOR_CONFIG,
            action_icon_file_name: str = ':/icons/polyline-action.svg',
    ):
        super().__init__(layers_props, palette_pack_settings, cursor_config, action_icon_file_name)

        self._annotation_mode = AnnotationMode.TISSUE
        self._gleason_grade = GleasonGrade.G3

    @property
    def annotation_mode(self) -> AnnotationMode:
        return self._annotation_mode

    @annotation_mode.setter
    def annotation_mode(self, value: AnnotationMode):
        if self._annotation_mode is not value:
            self._annotation_mode = value
            self.annotation_mode_changed.emit(self._annotation_mode)

    @property
    def gleason_grade(self) -> GleasonGrade:
        return self._gleason_grade

    @gleason_grade.setter
    def gleason_grade(self, value: GleasonGrade):
        if self._gleason_grade != value:
            self._gleason_grade = value
            self.gleason_grade_changed.emit(self._gleason_grade)


class CancerSegmentToolSettingsWidget(ViewerToolSettingsWidget):
    def __init__(self, tool_settings: CancerSegmentToolSettings, parent: QWidget = None):
        super().__init__(tool_settings, parent)

        annotation_mode_group_box_layout = QVBoxLayout()
        self._annotation_radio_button_to_mode = {}

        self._tissue_radio_button = self._add_annotation_radio_button(
            self.tr('Tissue'),
            AnnotationMode.TISSUE,
            annotation_mode_group_box_layout,
            self.tr('Allows drawing a polyline along the center of the tissue sample.'),
        )

        gleason_layout = QHBoxLayout()
        self._gleason_radio_button_to_grade = {}
        self._gleason_grade_to_radio_button = {}
        for gleason_grade in GleasonGrade:
            gleason_radio_button = self._add_annotation_radio_button(
                self.tr(f'Gl {gleason_grade.value}'),
                AnnotationMode.CANCER,
                gleason_layout,
                self._generate_gleason_tooltip(gleason_grade),
            )
            self._gleason_radio_button_to_grade[gleason_radio_button] = gleason_grade
            self._gleason_grade_to_radio_button[gleason_grade] = gleason_radio_button
        self._update_radio_buttons(tool_settings.annotation_mode, tool_settings.gleason_grade)
        tool_settings.annotation_mode_changed.connect(self._on_tool_settings_annotation_mode_changed)
        tool_settings.gleason_grade_changed.connect(self._on_tool_settings_gleason_grade_changed)
        gleason_layout.addStretch()

        annotation_mode_group_box_layout.addLayout(gleason_layout)

        annotation_mode_group_box = QGroupBox(self.tr('Annotation Mode'))
        annotation_mode_group_box.setLayout(annotation_mode_group_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(annotation_mode_group_box)
        layout.addStretch()

        self.setLayout(layout)

    def _add_annotation_radio_button(
            self, text: str, annotation_mode: AnnotationMode, layout: QLayout, tooltip: str = '') -> QRadioButton:
        radio_button = QRadioButton(text)
        radio_button.setToolTip(tooltip)
        radio_button.toggled.connect(partial(self._on_annotation_radio_button_toggled, radio_button))
        self._annotation_radio_button_to_mode[radio_button] = annotation_mode
        layout.addWidget(radio_button)
        return radio_button

    def _on_annotation_radio_button_toggled(self, annotation_radio_button: QRadioButton, checked: bool):
        if checked:
            annotation_mode = self._annotation_radio_button_to_mode[annotation_radio_button]
            self.tool_settings.annotation_mode = annotation_mode
            if annotation_mode is AnnotationMode.CANCER:
                self.tool_settings.gleason_grade = self._gleason_radio_button_to_grade[annotation_radio_button]

    def _generate_gleason_tooltip(self, gleason_grade: GleasonGrade) -> str:
        return self.tr(
            f'Allows drawing segments with Gleason grade {gleason_grade.value} on the drawn tissue polyline.')

    def _on_tool_settings_annotation_mode_changed(self, annotation_mode: AnnotationMode):
        self._update_radio_buttons(annotation_mode, self.tool_settings.gleason_grade)

    def _on_tool_settings_gleason_grade_changed(self, gleason_grade: GleasonGrade):
        self._update_radio_buttons(self.tool_settings.annotation_mode, gleason_grade)

    def _update_radio_buttons(self, annotation_mode: AnnotationMode, gleason_grade: GleasonGrade):
        if annotation_mode is AnnotationMode.TISSUE:
            self._tissue_radio_button.setChecked(True)
        else:
            self._gleason_grade_to_radio_button[gleason_grade].setChecked(True)


class CancerSegmentToolPlugin(ViewerToolPlugin):
    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            undo_plugin: UndoPlugin,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
            tool_cls: type[ViewerTool] = CancerSegmentTool,
            tool_settings_cls: type[ViewerToolSettings] = CancerSegmentToolSettings,
            tool_settings_widget_cls: type[ViewerToolSettingsWidget] = CancerSegmentToolSettingsWidget,
            action_name: str = QObject.tr('Cancer Segment'),
            action_shortcut: Qt.Key = Qt.Key_4,
    ):
        super().__init__(
            main_window_plugin,
            mdi_plugin,
            undo_plugin,
            palette_pack_settings_plugin,
            tool_cls,
            tool_settings_cls,
            tool_settings_widget_cls,
            action_name,
            action_shortcut,
        )
