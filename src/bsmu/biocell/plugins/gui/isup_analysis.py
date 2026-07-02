from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QStyle, QProxyStyle, QApplication, QStyleOptionViewItem, QTextEdit,
)
from PySide6.QtWidgets import QStyledItemDelegate

from bsmu.biocell.analysis.isup import analyze as analyze_isup
from bsmu.biocell.core.domain import GleasonGrade, IsupGradeGroup
from bsmu.vision.core.config import Config
from bsmu.vision.core.data.vector.shapes import Polyline
from bsmu.vision.core.layers import VectorLayer, RasterLayer
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu
from bsmu.vision.widgets.viewers.layered import LayeredDataViewerHolder

if TYPE_CHECKING:
    from PySide6.QtCore import QModelIndex, QRect
    from PySide6.QtGui import QPainter
    from PySide6.QtWidgets import QWidget, QStyleOption

    from bsmu.biocell.core.domain.gleason_analysis import GleasonAnalysisReport
    from bsmu.vision.core.data.raster import Raster
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from bsmu.vision.widgets.viewers.layered import LayeredDataViewer


NO_DATA_MARK = '–'
BACKGROUND_COLOR_ROLE = Qt.ItemDataRole.UserRole + 1

GLEASON_TO_COLOR = {
    GleasonGrade.G3: QColor(255, 255, 0, 200),
    GleasonGrade.G4: QColor(255, 165, 0, 200),
    GleasonGrade.G5: QColor(255, 0, 0, 200),
}


class TableColumn(IntEnum):
    METHOD = 0
    G3 = 1
    G4 = 2
    G5 = 3
    SCORE = 4
    ISUP = 5


GLEASON_TO_COLUMN_INDEX = {
    GleasonGrade.G3: TableColumn.G3,
    GleasonGrade.G4: TableColumn.G4,
    GleasonGrade.G5: TableColumn.G5,
}


@dataclass
class IsupGradeProfile(Config):
    """Risk, prognosis and recommendation profile for a single ISUP grade group."""
    risk: str = ''
    prognosis: str = ''
    recommendation: str = ''


@dataclass
class IsupAnalysisConfig(Config):
    """Configuration for ISUP analysis plugin."""
    isup_grade_to_profile: dict[IsupGradeGroup, IsupGradeProfile] = field(default_factory=dict)


class IsupAnalysisPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
    }

    def __init__(self, main_window_plugin: MainWindowPlugin, mdi_plugin: MdiPlugin) -> None:
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None
        self._mdi_plugin = mdi_plugin
        self._mdi: Mdi | None = None
        self._dialog: IsupAnalysisDialog | None = None

    def _enable_gui(self) -> None:
        # Enable custom header background colors by installing a proxy style that respects
        # the QPalette.Window brush set on header items (which Qt's default styles ignore).
        QApplication.setStyle(ColoredTableHeaderStyle(QApplication.style()))

        self._main_window = self._main_window_plugin.main_window
        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Analyze ISUP Grade Group'),
            self._analyze_isup,
        )
        self._mdi = self._mdi_plugin.mdi

    def _active_layered_data_viewer(self) -> LayeredDataViewer | None:
        layered_data_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredDataViewerHolder)
        if layered_data_viewer_sub_window is None:
            return None
        return layered_data_viewer_sub_window.layered_data_viewer

    def _analyze_isup(self) -> None:
        viewer = self._active_layered_data_viewer()
        if viewer is None:
            return

        if self._dialog is None:
            isup_config = IsupAnalysisConfig.from_dict(self.config.full_data)
            self._dialog = IsupAnalysisDialog(
                IsupDataSource(viewer),
                isup_config,
                self._main_window,
            )

        self._dialog.run_analysis()
        self._dialog.show()
        self._dialog.raise_()


class ColoredTableHeaderStyle(QProxyStyle):
    def drawControl(
            self,
            element: QStyle.ControlElement,
            option: QStyleOption,
            painter: QPainter,
            widget: QWidget | None = None,
    ) -> None:
        # Let the base style draw first (including borders/grid)
        self.baseStyle().drawControl(element, option, painter, widget)

        if element == QStyle.ControlElement.CE_HeaderSection and isinstance(widget, QHeaderView):
            # Now overlay the background color from the palette.
            # The palette's Window brush is set by headerItem.setBackground()
            bg_brush = option.palette.brush(QPalette.ColorRole.Window)
            if bg_brush != option.palette.brush(QPalette.ColorRole.Base):  # Avoid redundant fill
                # Shrink rect slightly to avoid painting over borders
                r = option.rect.adjusted(0, 0, -1, -1)  # Remove 1px from right and bottom
                painter.fillRect(r, bg_brush)


class RowSelectionBorderDelegate(QStyledItemDelegate):
    """Delegate that draws background color and selection border."""

    BORDER_COLOR = QColor(51, 153, 255)
    BORDER_THICKNESS = 2

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if bg_color := index.data(BACKGROUND_COLOR_ROLE):
            painter.fillRect(option.rect, bg_color)

        # Save selection state, then clear it from the copy
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        opt = QStyleOptionViewItem(option)
        opt.state &= ~(QStyle.StateFlag.State_Selected |
                       QStyle.StateFlag.State_HasFocus |
                       QStyle.StateFlag.State_KeyboardFocusChange)
        super().paint(painter, opt, index)

        if is_selected:
            self._draw_selection_borders(painter, option.rect, index)

    def _draw_selection_borders(self, painter: QPainter, rect: QRect, index: QModelIndex) -> None:
        """Draw selection borders around the row."""
        t = self.BORDER_THICKNESS
        color = self.BORDER_COLOR

        # Top and bottom borders (all cells)
        painter.fillRect(rect.left(), rect.top(), rect.width(), t, color)
        painter.fillRect(rect.left(), rect.bottom() - t + 1, rect.width(), t, color)

        # Left border (first cell only)
        if index.column() == 0:
            painter.fillRect(rect.left(), rect.top(), t, rect.height(), color)

        #  Right border (last cell only)
        if index.column() == index.model().columnCount() - 1:
            painter.fillRect(rect.right() - t + 1, rect.top(), t, rect.height(), color)


class IsupDataSource:
    """Fetches polylines and mask from the layered viewer."""

    def __init__(self, viewer: LayeredDataViewer) -> None:
        self._viewer = viewer

    def fetch(self) -> tuple[list[Polyline], Raster | None]:
        """Return selected polylines (or all if none selected) and optional mask."""
        polylines = [
            shape for shape in self._viewer.selection_manager.selected_shapes
            if isinstance(shape, Polyline)
        ]

        if not polylines:
            vector_layer = self._viewer.layer_by_name('vectors')
            if isinstance(vector_layer, VectorLayer):
                polylines = [
                    shape for shape in vector_layer.shapes
                    if isinstance(shape, Polyline)
                ]

        mask_layer = self._viewer.layer_by_name('masks')
        mask = mask_layer.data if isinstance(mask_layer, RasterLayer) else None
        return polylines, mask


class IsupTableBuilder:
    """Builds and populates the ISUP analysis table."""

    _METHODS = ['Linear', 'Linear Through', 'Area']
    _HIDDEN_SCORE_ROW = 1  # Linear Through
    _HEADERS = ['Method', 'Gl 3, %', 'Gl 4, %', 'Gl 5, %', 'Gl Score', 'ISUP Grade']

    def __init__(self, table: QTableWidget) -> None:
        self._table = table

        self._bold_font = QFont()
        self._bold_font.setBold(True)
        self._setup()

    def _setup(self) -> None:
        self._table.setColumnCount(len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        # Remove default selection highlight via palette
        palette = self._table.palette()
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.transparent)
        palette.setColor(QPalette.ColorRole.HighlightedText, palette.color(QPalette.ColorRole.Text))
        self._table.setPalette(palette)

        self._table.setItemDelegate(RowSelectionBorderDelegate(self._table))
        self._table.horizontalHeader().setFont(self._bold_font)

        # Color-code Gleason grade headers
        for gleason, column_index in GLEASON_TO_COLUMN_INDEX.items():
            header_item = self._table.horizontalHeaderItem(column_index)
            if header_item is None:
                header_item = QTableWidgetItem()
                self._table.setHorizontalHeaderItem(column_index, header_item)
            header_item.setBackground(GLEASON_TO_COLOR[gleason])
            header_item.setFont(self._bold_font)

    def populate(self, report: GleasonAnalysisReport) -> None:
        distributions = [report.linear, report.linear_through, report.area]
        self._table.setRowCount(len(self._METHODS))

        for row, (method, dist) in enumerate(zip(self._METHODS, distributions)):
            method_item = QTableWidgetItem(method)
            method_item.setFont(self._bold_font)
            if row == self._HIDDEN_SCORE_ROW:
                method_item.setToolTip(
                    'Alternative method; score and ISUP grade hidden to avoid confusion')
            self._table.setItem(row, TableColumn.METHOD, method_item)

            for grade in GleasonGrade:
                col = GLEASON_TO_COLUMN_INDEX[grade]
                pct = dist.grade_to_percentage.get(grade, 0.0) if dist else 0.0
                text = f'{pct:.2f}' if dist else NO_DATA_MARK
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(BACKGROUND_COLOR_ROLE, GLEASON_TO_COLOR[grade])
                self._table.setItem(row, col, item)

            score_text = (
                NO_DATA_MARK if row == self._HIDDEN_SCORE_ROW
                else str(dist.score) if dist and dist.score else NO_DATA_MARK
            )
            score_item = QTableWidgetItem(score_text)
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, TableColumn.SCORE, score_item)

            grade_text = (
                NO_DATA_MARK if row == self._HIDDEN_SCORE_ROW
                else str(dist.isup_grade_group) if dist and dist.isup_grade_group else NO_DATA_MARK
            )
            grade_item = QTableWidgetItem(grade_text)
            grade_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, TableColumn.ISUP, grade_item)

        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def select_first_row(self) -> None:
        if self._table.rowCount() > 0:
            self._table.selectRow(0)


class IsupProfileFormatter:
    """Renders ISUP grade profile as HTML clinical summary."""

    _TRANSLATION_CONTEXT = 'IsupProfileFormatter'

    def __init__(
        self,
        table: QTableWidget,
        text_edit: QTextEdit,
        config: IsupAnalysisConfig,
    ) -> None:
        self._table = table
        self._text_edit = text_edit
        self._config = config

    def format(self) -> None:
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            self._text_edit.clear()
            return

        row = selected_rows[0].row()
        isup_item = self._table.item(row, TableColumn.ISUP)

        if not isup_item or isup_item.text() == NO_DATA_MARK:
            self._text_edit.setHtml(self._no_isup_html())
            return

        isup_grade = IsupGradeGroup(int(isup_item.text()))
        profile = self._config.isup_grade_to_profile.get(isup_grade)

        if profile is None:
            self._text_edit.setHtml(self._missing_profile_html())
            return

        self._text_edit.setHtml(self._format_html(profile))

    @classmethod
    def _tr(cls, text: str) -> str:
        return QCoreApplication.translate(cls._TRANSLATION_CONTEXT, text)

    def _no_isup_html(self) -> str:
        """When the selected row has no ISUP grade."""
        return f'<i>{self._tr("No clinical data available for display.")}</i>'

    def _missing_profile_html(self) -> str:
        """When ISUP grade exists in the table but has no profile in config."""
        return f'<i>{self._tr("Clinical recommendations are not available for this ISUP grade.")}</i>'

    def _empty_profile_html(self) -> str:
        """When profile is found but all fields are empty."""
        return f'<i>{self._tr("No data.")}</i>'

    def _format_html(self, profile: IsupGradeProfile) -> str:
        sections = {
            self._tr('Risk'): profile.risk,
            self._tr('Prognosis'): profile.prognosis,
            self._tr('Recommendations'): profile.recommendation,
        }
        html_parts = []
        for title, text in sections.items():
            if text:
                html_parts.append(
                    f'<p align="justify" style="margin-top:0px; margin-bottom:8px;">'
                    f'<b>{title}:</b> {text}</p>'
                )
        return ''.join(html_parts) if html_parts else self._empty_profile_html()


class IsupAnalysisDialog(QDialog):
    def __init__(
        self,
        data_source: IsupDataSource,
        isup_config: IsupAnalysisConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._data_source = data_source
        self.setWindowTitle(self.tr('ISUP Grade Group Analysis'))

        layout = QVBoxLayout(self)

        self._table = QTableWidget()
        self._table_builder = IsupTableBuilder(self._table)
        layout.addWidget(self._table, stretch=1)

        self._profile_text = QTextEdit()
        self._profile_text.setReadOnly(True)
        layout.addWidget(self._profile_text, stretch=2)

        self._isup_profile_formatter = IsupProfileFormatter(
            self._table, self._profile_text, isup_config)
        self._table.itemSelectionChanged.connect(self._isup_profile_formatter.format)

        button_layout = QHBoxLayout()
        analyze_button = QPushButton(self.tr('Analyze'))
        analyze_button.clicked.connect(self.run_analysis)
        analyze_button.setDefault(True)

        self._table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._table.setTabKeyNavigation(False)

        close_button = QPushButton(self.tr('Close'))
        close_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(analyze_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        analyze_button.setFocus()

        self.resize(500, 450)

    def run_analysis(self) -> None:
        polylines, mask = self._data_source.fetch()
        report = analyze_isup(polylines, mask)
        self._table_builder.populate(report)
        self._table_builder.select_first_row()
        self._isup_profile_formatter.format()
