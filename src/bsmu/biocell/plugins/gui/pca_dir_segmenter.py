from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSize
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from bsmu.biocell.plugins.pca_dir_segmenter import DirSegmentationConfig, PcaDirSegmenter
from bsmu.biocell.plugins.pca_segmenter import SegmentationMode
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu

if TYPE_CHECKING:
    from bsmu.biocell.plugins.pca_segmenter import PcaSegmenter, PcaSegmenterPlugin
    from bsmu.vision.plugins.storages.task import TaskStorage, TaskStoragePlugin
    from bsmu.vision.plugins.windows.main import MainWindow, MainWindowPlugin


class PcaDirSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'pca_segmenter_plugin': 'bsmu.biocell.plugins.pca_segmenter.PcaSegmenterPlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            pca_segmenter_plugin: PcaSegmenterPlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None

        self._pca_segmenter_plugin = pca_segmenter_plugin
        self._task_storage_plugin = task_storage_plugin

        self._pca_dir_segmenter_gui: PcaDirSegmenterGui | None = None

    @property
    def pca_dir_segmenter_gui(self) -> PcaDirSegmenterGui | None:
        return self._pca_dir_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window

        task_storage = self._task_storage_plugin.task_storage
        # TODO: read the DirSegmentationConfig from *.conf.yaml file
        self._pca_dir_segmenter_gui = PcaDirSegmenterGui(
            DirSegmentationConfig(), self._pca_segmenter_plugin.pca_segmenter, task_storage, self._main_window)

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Segment Cancer in Directory...'),
            self._pca_dir_segmenter_gui.segment_async_with_dialog,
        )

    def _disable(self):
        self._pca_dir_segmenter_gui = None

        self._main_window = None

        raise NotImplementedError


class DirSelector(QWidget):
    def __init__(self, title: str, default_dir: Path = None, parent: QWidget = None):
        super().__init__(parent)

        self._title = title
        self._default_dir: Path = default_dir or Path()

        self._dir_line_edit: QLineEdit | None = None
        self._browse_button: QPushButton | None = None
        self._browse_dir_row_layout: QHBoxLayout() | None = None

        self._init_gui()

    @property
    def selected_dir(self) -> Path:
        return Path(self.selected_dir_str)

    @property
    def selected_dir_str(self) -> str:
        return self._dir_line_edit.text()

    def _init_gui(self):
        title_label = QLabel(self._title)

        self._dir_line_edit = QLineEdit(str(self._default_dir.resolve()))
        self._dir_line_edit.setReadOnly(True)

        self._browse_button = QPushButton(self.tr('Browse...'))
        self._browse_button.clicked.connect(self._browse_dir)

        self._browse_dir_row_layout = QHBoxLayout()
        self._browse_dir_row_layout.addWidget(self._dir_line_edit)
        self._browse_dir_row_layout.addWidget(self._browse_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title_label)
        layout.addLayout(self._browse_dir_row_layout)
        layout.addStretch(1)

        self.setLayout(layout)
        self.adjustSize()

    def _browse_dir(self):
        selected_dir_str = QFileDialog.getExistingDirectory(self, self.tr('Select Directory'), self.selected_dir_str)
        if selected_dir_str:
            self._dir_line_edit.setText(selected_dir_str)

    def sizeHint(self) -> QSize:
        text_width = self._dir_line_edit.fontMetrics().boundingRect(self.selected_dir_str).width()
        return QSize(text_width + self._browse_dir_row_layout.spacing() + self._browse_button.width(), 0)


class DirSegmentationConfigDialog(QDialog):
    def __init__(self, config: DirSegmentationConfig, title: str, parent: QWidget = None):
        super().__init__(parent)

        self._config = config

        self.setWindowTitle(title)

        self._image_dir_selector: DirSelector | None = None
        self._mask_dir_selector: DirSelector | None = None
        self._include_subdirs_check_box: QCheckBox | None = None
        self._overwrite_existing_masks_check_box: QCheckBox | None = None
        self._segmentation_mode_combo_box: QComboBox | None = None

        self._init_gui()

    @property
    def config(self) -> DirSegmentationConfig:
        return self._config

    def _init_gui(self):
        self._image_dir_selector = DirSelector(self.tr('Images Directory:'), self._config.image_dir)
        self._mask_dir_selector = DirSelector(self.tr('Masks Directory:'), self._config.mask_dir)

        self._include_subdirs_check_box = QCheckBox(self.tr('Include Subfolders'))
        self._include_subdirs_check_box.setChecked(self._config.include_subdirs)

        self._overwrite_existing_masks_check_box = QCheckBox(self.tr('Overwrite Existing Masks'))
        self._overwrite_existing_masks_check_box.setChecked(self._config.overwrite_existing_masks)
        self._overwrite_existing_masks_check_box.setToolTip(
            self.tr('Enabling this option will overwrite any mask files with the same name as new masks.'))

        self._segmentation_mode_combo_box = QComboBox()
        for mode in SegmentationMode:
            self._segmentation_mode_combo_box.addItem(mode.display_name, mode)
        config_segmentation_mode_index = self._segmentation_mode_combo_box.findData(self._config.segmentation_mode)
        assert config_segmentation_mode_index >= 0, 'Config segmentation mode value was not found in the combo box.'
        self._segmentation_mode_combo_box.setCurrentIndex(config_segmentation_mode_index)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = button_box.button(QDialogButtonBox.Ok)
        ok_button.setText(self.tr('Run Segmentation'))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        form_layout = QFormLayout()
        form_layout.addRow(self._include_subdirs_check_box)
        form_layout.addRow(self._overwrite_existing_masks_check_box)
        form_layout.addRow(self.tr('Segmentation Mode:'), self._segmentation_mode_combo_box)

        layout = QVBoxLayout()
        layout.addWidget(self._image_dir_selector)
        layout.addWidget(self._mask_dir_selector)
        layout.addLayout(form_layout)
        layout.addStretch(1)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def accept(self):
        self._config.image_dir = self._image_dir_selector.selected_dir
        self._config.mask_dir = self._mask_dir_selector.selected_dir
        self._config.include_subdirs = self._include_subdirs_check_box.isChecked()
        self._config.overwrite_existing_masks = self._overwrite_existing_masks_check_box.isChecked()
        self._config.segmentation_mode = self._segmentation_mode_combo_box.currentData()

        super().accept()


class PcaDirSegmenterGui(QObject):
    def __init__(
            self,
            dir_segmentation_config: DirSegmentationConfig,
            pca_segmenter: PcaSegmenter,
            task_storage: TaskStorage = None,
            main_window: MainWindow = None,
    ):
        super().__init__()

        self._dir_segmentation_config = dir_segmentation_config
        self._pca_segmenter = pca_segmenter
        self._task_storage = task_storage
        self._main_window = main_window

    def segment_async_with_dialog(self):
        # Pass `self._main_window` as parent to display correct window icon
        # and to place the dialog in the middle of the parent
        dir_segmentation_config_dialog = DirSegmentationConfigDialog(
            self._dir_segmentation_config, self.tr('Cancer Segmentation Settings'), self._main_window)
        dir_segmentation_config_dialog.accepted.connect(self.segment_async)
        dir_segmentation_config_dialog.open()

    def segment_async(self):
        pca_dir_segmenter = PcaDirSegmenter(self._pca_segmenter, self._task_storage)
        pca_dir_segmenter.segment_async(self._dir_segmentation_config)
