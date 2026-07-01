from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QPushButton, QStyle, QProxyStyle, QApplication, QStyleOptionViewItem, QTextEdit
)
from PySide6.QtWidgets import QStyledItemDelegate

from bsmu.biocell.analysis.isup import analyze
from bsmu.biocell.core.domain import GleasonGrade, IsupGradeGroup
from bsmu.biocell.core.domain.gleason_analysis import GleasonAnalysisReport
from bsmu.vision.core.config import Config
from bsmu.vision.core.data.vector.shapes import Polyline
from bsmu.vision.core.layers import VectorLayer, RasterLayer
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu
from bsmu.vision.widgets.viewers.layered import LayeredDataViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi
    from bsmu.vision.widgets.viewers.layered import LayeredDataViewer


@dataclass
class IsupGradeProfile(Config):
    """Clinical summary for a single ISUP grade."""
    risk: str = ''
    prognosis: str = ''
    recommendation: str = ''


@dataclass
class IsupAnalysisConfig(Config):
    """Configuration for ISUP analysis plugin."""
    isup_grade_to_profile: dict[IsupGradeGroup, IsupGradeProfile] = field(default_factory=dict)


class ColoredTableHeaderStyle(QProxyStyle):
    def drawControl(self, element, option, painter, widget=None):
        self.baseStyle().drawControl(element, option, painter, widget)

        if element == QStyle.ControlElement.CE_HeaderSection and isinstance(widget, QHeaderView):
            bg_brush = option.palette.brush(QPalette.ColorRole.Window)
            if bg_brush != option.palette.brush(QPalette.ColorRole.Base):
                r = option.rect.adjusted(0, 0, -1, -1)
                painter.fillRect(r, bg_brush)


class IsupAnalysisPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
    }

    def __init__(self, main_window_plugin: MainWindowPlugin, mdi_plugin: MdiPlugin):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None
        self._mdi_plugin = mdi_plugin
        self._mdi: Mdi | None = None
        self._dialog: IsupAnalysisDialog | None = None

    def _enable_gui(self):
        QApplication.setStyle(ColoredTableHeaderStyle(QApplication.style()))

        self._main_window = self._main_window_plugin.main_window
        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Analyze ISUP Grade Group'),
            self._analyze_isup,
            QKeySequence(),
        )
        self._mdi = self._mdi_plugin.mdi

    def _active_viewer(self) -> LayeredDataViewer | None:
        active_sub_window = self._mdi.activeSubWindow()
        if not isinstance(active_sub_window, LayeredDataViewerHolder):
            QMessageBox.warning(
                self._main_window, 'No Layered Image', 'The active window does not contain a layered image.')
            return None
        return active_sub_window.layered_data_viewer

    def _analyze_isup(self):
        viewer = self._active_viewer()
        if viewer is None:
            return

        if self._dialog is None:
            isup_config = IsupAnalysisConfig.from_dict(self.config.full_data)
            self._dialog = IsupAnalysisDialog(viewer, isup_config, self._main_window)

        self._dialog.refresh()
        self._dialog.show()
        self._dialog.raise_()


BACKGROUND_COLOR_ROLE = Qt.ItemDataRole.UserRole + 1


class RowBorderDelegate(QStyledItemDelegate):
    """Delegate that draws background color and selection border."""

    BORDER_COLOR = QColor(51, 153, 255)
    BORDER_THICKNESS = 2

    def paint(self, painter, option, index):
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

    def _draw_selection_borders(self, painter, rect, index):
        """Draws selection borders around the row."""
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


gleason_to_color = {
    GleasonGrade.G3: QColor(255, 255, 0, 200),
    GleasonGrade.G4: QColor(255, 165, 0, 200),
    GleasonGrade.G5: QColor(255, 0, 0, 200),
}


class IsupAnalysisDialog(QDialog):
    def __init__(self, viewer: LayeredDataViewer, isup_config: IsupAnalysisConfig, parent=None) -> None:
        super().__init__(parent)

        self._viewer = viewer
        self._isup_config = isup_config
        self.setWindowTitle('ISUP Grade Group Analysis')

        layout = QVBoxLayout(self)

        # Table
        self._table = QTableWidget()
        self._setup_table()
        layout.addWidget(self._table, stretch=1)

        # Clinical Summary
        self._summary_text = QTextEdit()
        self._summary_text.setReadOnly(True)
        layout.addWidget(self._summary_text, stretch=2)

        # Buttons
        button_layout = QHBoxLayout()
        refresh_button = QPushButton(self.tr('Refresh'))
        refresh_button.clicked.connect(self.refresh)
        refresh_button.setDefault(True)
        refresh_button.setAutoDefault(True)

        self._table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._table.setTabKeyNavigation(False)

        close_button = QPushButton(self.tr('Close'))
        close_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(refresh_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        refresh_button.setFocus()

        self.resize(600, 450)

    def _setup_table(self):
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(['Method', 'Gl 3', 'Gl 4', 'Gl 5', 'Gleason Score', 'ISUP Grade'])

        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        # Remove default blue selection highlight via palette
        palette = self._table.palette()
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.transparent)
        palette.setColor(QPalette.ColorRole.HighlightedText, palette.color(QPalette.ColorRole.Text))
        self._table.setPalette(palette)

        # Custom delegate for row selection border
        delegate = RowBorderDelegate(self._table)
        self._table.setItemDelegate(delegate)

        self._table.itemSelectionChanged.connect(self._update_clinical_summary)

        header_font = QFont()
        header_font.setBold(True)
        self._table.horizontalHeader().setFont(header_font)

        # Color-code Gleason grade headers
        for grade, color in gleason_to_color.items():
            column_index = grade.value - 2  # G3 -> 1, G4 -> 2, G5 -> 3
            header_item = self._table.horizontalHeaderItem(column_index)
            if header_item is None:
                header_item = QTableWidgetItem()
                self._table.setHorizontalHeaderItem(column_index, header_item)
            header_item.setBackground(color)
            header_item.setFont(header_font)

    def _update_clinical_summary(self):
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            self._summary_text.clear()
            return

        row = selected_rows[0].row()
        isup_item = self._table.item(row, 5)

        if not isup_item or isup_item.text() == '—':
            self._summary_text.setHtml('<i>Нет данных для отображения клинических рекомендаций.</i>')
            return

        isup_grade = IsupGradeGroup(int(isup_item.text()))
        profile = self._isup_config.isup_grade_to_profile.get(isup_grade)

        if profile is None:
            self._summary_text.setHtml('<i>Клинические рекомендации недоступны для данного уровня ISUP.</i>')
            return

        self._summary_text.setHtml(self._format_summary_html(profile))

    @staticmethod
    def _format_summary_html(profile: IsupGradeProfile) -> str:
        sections = {
            'Риск': profile.risk,
            'Прогноз': profile.prognosis,
            'Рекомендации': profile.recommendation,
        }
        html_parts = []
        for title, text in sections.items():
            if text:
                html_parts.append(
                    f'<p align="justify" style="margin-top:0px; margin-bottom:8px;">'
                    f'<b>{title}:</b> {text}</p>'
                )
        return ''.join(html_parts) if html_parts else '<i>Нет данных.</i>'

    def refresh(self):
        # Get polylines
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

        # Get mask
        mask_layer = self._viewer.layer_by_name('masks')
        mask = mask_layer.data if isinstance(mask_layer, RasterLayer) else None

        # Analyze
        report = analyze(polylines, mask)

        # Update table
        self._update_table(report)

    def _update_table(self, report: GleasonAnalysisReport):
        methods = ['Linear', 'Linear Through', 'Area']
        distributions = [report.linear, report.linear_through, report.area]

        self._table.setRowCount(len(methods))

        bold_font = QFont()
        bold_font.setBold(True)

        for row, (method, dist) in enumerate(zip(methods, distributions)):
            # Method name
            method_item = QTableWidgetItem(method)
            method_item.setFont(bold_font)
            self._table.setItem(row, 0, method_item)

            # Percentages
            for grade in GleasonGrade:
                col = grade.value - 2
                pct = dist.grade_to_percentage.get(grade, 0.0) if dist else 0.0
                text = f'{pct:.2f}' if dist else '—'
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(BACKGROUND_COLOR_ROLE, gleason_to_color[grade])
                self._table.setItem(row, col, item)

            # Gleason Score
            score = dist.score if dist else None
            score_text = str(score) if score else '—'
            score_item = QTableWidgetItem(score_text)
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 4, score_item)

            # ISUP Grade Group
            grade_group = dist.isup_grade_group if dist else None
            grade_text = str(grade_group.value) if grade_group else '—'
            grade_item = QTableWidgetItem(grade_text)
            grade_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 5, grade_item)

        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        if self._table.rowCount() > 0:
            self._table.selectRow(0)
        self._update_clinical_summary()
