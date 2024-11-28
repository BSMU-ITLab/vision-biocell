from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QMessageBox,
    QSlider, QVBoxLayout, QSpinBox, QWidget
)

from bsmu.biocell.plugins.tissue_threshold_segmenter import (
    TissueSegmentationConfig, TissueSegmenter, GradientCornerValues)
from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.core.visibility import Visibility
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, FileMenu
from bsmu.vision.plugins.writers.image.common import CommonImageFileWriter
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.core.palette import Palette
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.windows.main import MainWindow, MainWindowPlugin


class TissueThresholdSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._main_window: MainWindow | None = None

        self._mdi_plugin = mdi_plugin
        self._palette_pack_settings_plugin = palette_pack_settings_plugin

        self._tissue_segmenter_gui: TissueSegmenterGui | None = None

    @property
    def tissue_segmenter_gui(self) -> TissueSegmenterGui | None:
        return self._tissue_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window

        mdi = self._mdi_plugin.mdi
        main_palette = self._palette_pack_settings_plugin.settings.main_palette
        tissue_segmentation_config = TissueSegmentationConfig.from_dict(self.config_value('tissue_segmenter'))
        self._tissue_segmenter_gui = TissueSegmenterGui(
            tissue_segmentation_config, mdi, main_palette, self._main_window)

        self._main_window.add_menu_action(
            FileMenu,
            self.tr('Save Tissue Mask and Config As...'),
            self._tissue_segmenter_gui.save_tissue_mask_and_config_as,
        )
        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Segment Tissue...'),
            self._tissue_segmenter_gui.segment_with_dialog,
        )

    def _disable(self):
        self._tissue_segmenter_gui = None

        self._main_window = None

        raise NotImplementedError


class GradientCornerEditor(QWidget):
    def __init__(self, config: GradientCornerValues, parent: QWidget = None):
        super().__init__(parent)

        self._config = config

        self._spin_boxes = []

        self._init_gui()

    @property
    def config(self) -> GradientCornerValues:
        return self._config

    def _init_gui(self):
        grid_layout = QGridLayout()

        config_corner_values_iter = iter(self._config)
        for row in range(2):
            for col in range(2):
                config_corner_value = next(config_corner_values_iter)

                spin_box = QDoubleSpinBox()
                spin_box.setRange(0.0, 1.0)
                spin_box.setValue(config_corner_value)
                spin_box.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)

                self._spin_boxes.append(spin_box)
                grid_layout.addWidget(spin_box, row, col)

        self.setLayout(grid_layout)

    def apply_changes(self):
        self._config.update_values(*(spin_box.value() for spin_box in self._spin_boxes))


class TissueSegmentationConfigDialog(QDialog):
    applied = Signal()

    def __init__(self, config: TissueSegmentationConfig, title: str, parent: QWidget = None):
        super().__init__(parent)

        self._config = config

        self.setWindowTitle(title)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self._blur_size_spin_box: QSpinBox | None = None

        self._gradient_corner_editor: GradientCornerEditor | None = None

        self._saturation_threshold_slider: QSlider | None = None
        self._brightness_threshold_slider: QSlider | None = None

        self._saturation_spin_box: QDoubleSpinBox | None = None
        self._brightness_spin_box: QDoubleSpinBox | None = None

        self._remove_small_object_size_spin_box: QSpinBox | None = None
        self._fill_hole_size_spin_box: QSpinBox | None = None

        self._init_gui()

    @property
    def config(self) -> TissueSegmentationConfig:
        return self._config

    def _init_gui(self):
        # self._saturation_threshold_slider = QSlider(Qt.Horizontal)
        # self._saturation_threshold_slider.setValue(self._config.saturation_threshold)
        #
        # self._brightness_threshold_slider = QSlider(Qt.Horizontal)
        # self._brightness_threshold_slider.setValue(self._config.brightness_threshold)

        self._blur_size_spin_box = QSpinBox()
        self._blur_size_spin_box.setRange(1, 99)
        self._blur_size_spin_box.setSingleStep(2)
        self._blur_size_spin_box.setValue(self._config.blur_size)
        self._blur_size_spin_box.setToolTip(self.tr('Enter only odd values.'))

        self._gradient_corner_editor = GradientCornerEditor(self._config.gradient_corner_values)

        self._saturation_spin_box = QDoubleSpinBox()
        self._saturation_spin_box.setRange(0, 1)
        self._saturation_spin_box.setDecimals(3)
        self._saturation_spin_box.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self._saturation_spin_box.setValue(self._config.saturation_threshold)

        self._brightness_spin_box = QDoubleSpinBox()
        self._brightness_spin_box.setRange(0, 1)
        self._brightness_spin_box.setDecimals(3)
        self._brightness_spin_box.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self._brightness_spin_box.setValue(self._config.brightness_threshold)

        self._remove_small_object_size_spin_box = QSpinBox()
        self._remove_small_object_size_spin_box.setRange(0, 9999)
        self._remove_small_object_size_spin_box.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self._remove_small_object_size_spin_box.setValue(self._config.remove_small_object_size)

        self._fill_hole_size_spin_box = QSpinBox()
        self._fill_hole_size_spin_box.setRange(0, 9999)
        self._fill_hole_size_spin_box.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self._fill_hole_size_spin_box.setValue(self._config.fill_hole_size)

        form_layout = QFormLayout()
        form_layout.addRow(self.tr('Blur Size (Odd Value):'), self._blur_size_spin_box)
        form_layout.addRow(self.tr('Gradient Corner Values:'), self._gradient_corner_editor)
        form_layout.addRow(self.tr('Saturation Threshold:'), self._saturation_spin_box)
        form_layout.addRow(self.tr('Brightness Threshold:'), self._brightness_spin_box)
        form_layout.addRow(self.tr('Remove Small Object Size:'), self._remove_small_object_size_spin_box)
        form_layout.addRow(self.tr('Fill Hole Size:'), self._fill_hole_size_spin_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Apply)
        apply_button = button_box.button(QDialogButtonBox.Apply)
        apply_button.pressed.connect(self._apply)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addStretch(1)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _apply(self):
        if self._blur_size_spin_box.value() % 2 == 0:
            QMessageBox.warning(
                self,
                self.tr('Incorrect Blur Size Value'),
                self.tr('The Blur Size Must Be Odd.'),
            )
            return

        # self._config.saturation_threshold = self._saturation_threshold_slider.value()
        # self._config.brightness_threshold = self._brightness_threshold_slider.value()

        self._config.blur_size = self._blur_size_spin_box.value()
        self._gradient_corner_editor.apply_changes()
        self._config.saturation_threshold = self._saturation_spin_box.value()
        self._config.brightness_threshold = self._brightness_spin_box.value()
        self._config.remove_small_object_size = self._remove_small_object_size_spin_box.value()
        self._config.fill_hole_size = self._fill_hole_size_spin_box.value()

        self.applied.emit()


class TissueSegmenterGui(QObject):
    def __init__(
            self,
            tissue_segmentation_config: TissueSegmentationConfig,
            mdi: Mdi,
            mask_palette: Palette,
            main_window: MainWindow = None,
    ):
        super().__init__()

        self._tissue_segmentation_config = tissue_segmentation_config
        self._mdi = mdi
        self._mask_palette = mask_palette
        self._main_window = main_window

        self._tissue_segmentation_config_dialog: TissueSegmentationConfigDialog | None = None
        self._mask_layer_name = 'masks'

    def segment_with_dialog(self):
        config_dialog = self._created_tissue_segmentation_config_dialog
        config_dialog.show()
        config_dialog.raise_()
        config_dialog.activateWindow()

    def segment(self):
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        image_layer = layered_image.layers[0]
        image = image_layer.image
        tissue_segmenter = TissueSegmenter()
        mask = tissue_segmenter.segment(image.pixels, self._tissue_segmentation_config)

        layered_image.add_layer_or_modify_pixels(
            self._mask_layer_name,
            mask,
            FlatImage,
            self._mask_palette,
            visibility=Visibility(True, 0.5),
        )

    def save_tissue_mask_and_config_as(self):
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        save_path_str, selected_filter = QFileDialog.getSaveFileName(
            parent=self._main_window,
            caption=self.tr('Save Tissue Mask and Config'),
            dir=str(layered_image.layers[0].image_path.with_suffix('.png')),
            filter='PNG (*.png)',
        )
        if not save_path_str:
            return

        save_path = Path(save_path_str)
        try:
            CommonImageFileWriter().write_to_file(
                layered_image.layer_by_name(self._mask_layer_name).image, save_path)
            self._tissue_segmentation_config.save_to_yaml(save_path.with_suffix('.conf.yaml'))
        except Exception as e:
            QMessageBox.warning(
                self._main_window,
                self.tr('Save Error'),
                self.tr(f'Cannot save the mask.\n{e}'),
            )

    @property
    def _created_tissue_segmentation_config_dialog(self) -> TissueSegmentationConfigDialog:
        if self._tissue_segmentation_config_dialog is None:
            self._tissue_segmentation_config_dialog = TissueSegmentationConfigDialog(
                self._tissue_segmentation_config, self.tr('Tissue Segmentation Settings'), self._main_window)
            self._tissue_segmentation_config_dialog.applied.connect(self.segment)
            self._tissue_segmentation_config_dialog.destroyed.connect(
                self._on_tissue_segmentation_config_dialog_destroyed)
        return self._tissue_segmentation_config_dialog

    def _on_tissue_segmentation_config_dialog_destroyed(self):
        self._tissue_segmentation_config_dialog = None

    def _active_layered_image(self) -> LayeredImage | None:
        layered_image_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        return layered_image_viewer_sub_window and layered_image_viewer_sub_window.layered_image_viewer.data
