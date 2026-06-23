from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QPalette
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDockWidget, QHBoxLayout

from bsmu.vision.core.plugins import Plugin

if TYPE_CHECKING:
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from typing import ClassVar, Iterable


class PaletteViewPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
    }

    def __init__(self, main_window_plugin: MainWindowPlugin) -> None:
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None
        self._palette_dock: PaletteDockWidget | None = None

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window
        self._palette_dock = PaletteDockWidget(parent=self._main_window)
        self._main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._palette_dock)

    def _disable(self) -> None:
        if self._palette_dock:
            self._main_window.removeDockWidget(self._palette_dock)
            self._palette_dock.setParent(None)
            self._palette_dock.deleteLater()
            self._palette_dock = None
        self._main_window = None


class ColorSwatch(QWidget):
    """Display a color as a small rounded rectangle."""

    DEFAULT_SIZE = QSize(20, 20)
    CORNER_RADIUS = 3.0
    BORDER_WIDTH = 1.0

    def __init__(
        self,
        color: QColor | str,
        size: QSize = DEFAULT_SIZE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._color = QColor(color)
        self.setFixedSize(size)
        self.setToolTip(self._color.name())

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, value: QColor | str) -> None:
        self._color = QColor(value)
        self.setToolTip(self._color.name())
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Offset by 0.5px for crisp 1px borders on any DPI
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawRoundedRect(rect, self.CORNER_RADIUS, self.CORNER_RADIUS)

        border_color = self.palette().color(QPalette.ColorRole.Mid)
        painter.setPen(QPen(border_color, self.BORDER_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, self.CORNER_RADIUS, self.CORNER_RADIUS)


@dataclass(frozen=True)
class PaletteItem:
    """Represent a single entry in a color palette."""
    color: QColor | str
    label: str


class PaletteDockWidget(QDockWidget):
    """Display a named color palette in a dock widget."""

    GLEASON_PALETTE: ClassVar[tuple[PaletteItem, ...]] = (
        PaletteItem('#FFFF00', 'Gleason 3'),
        PaletteItem('#FFA500', 'Gleason 4'),
        PaletteItem('#FF0000', 'Gleason 5'),
        # PaletteItem('#00FFFF', 'Seminal vesicles'),
        # PaletteItem('#0000FF', 'Nerve trunk'),
        # PaletteItem('#00FF00', 'Non-cancerous'),
        # PaletteItem('#000000', 'Unknown'),
        # PaletteItem('#46BF00', 'Non-tissue'),
    )

    def __init__(
        self,
        items: Iterable[PaletteItem] = GLEASON_PALETTE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__('Palette', parent)

        self._init_ui(items)

    def _init_ui(self, items: Iterable[PaletteItem]) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        for item in items:
            layout.addLayout(self._create_row(item))

        layout.addStretch(1)
        self.setWidget(container)

    @staticmethod
    def _create_row(item: PaletteItem) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(ColorSwatch(item.color))

        label = QLabel(item.label)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label, 1)
        row.addStretch()
        return row
