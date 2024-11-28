from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from timeit import default_timer as timer

import cv2 as cv
import numpy as np
from PySide6.QtCore import QObject
from numpy.typing import DTypeLike

from bsmu.vision.core.config import Config


@dataclass
class GradientCornerValues(Config):
    top_left: float = 1.0
    top_right: float = 1.0
    bottom_left: float = 1.0
    bottom_right: float = 1.0

    def __iter__(self):
        # Makes an instance of GradientCornerValues iterable, returning the values of its fields in order.
        return (getattr(self, f.name) for f in fields(self))

    def is_unit_gradient(self) -> bool:
        # Checks if all corner values are almost equal to 1.
        return all(math.isclose(value, 1.0) for value in self)

    def update_values(self, top_left: float, top_right: float, bottom_left: float, bottom_right: float):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right


@dataclass
class TissueSegmentationConfig(Config):
    blur_size: int = 3
    gradient_corner_values: GradientCornerValues = field(default_factory=GradientCornerValues)
    saturation_threshold: float = 0.075
    brightness_threshold: float = 0.03
    remove_small_object_size: int = 500
    fill_hole_size: int = 0


class TissueSegmenter(QObject):
    def __init__(self):
        super().__init__()

    def segment(self, image: np.ndarray, config: TissueSegmentationConfig) -> np.ndarray:
        logging.info(f'Segment Tissue: {config}')

        segmentation_start = timer()

        # Convert image into float, else cv.cvtColor returns np.uint8, and we will lose conversion precision
        image = np.float32(image) / 255
        hsb_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

        if config.blur_size > 1:
            blur_kernel_size = (config.blur_size, config.blur_size)
            cv.GaussianBlur(hsb_image, blur_kernel_size, sigmaX=0, dst=hsb_image)

            # Erode image to compensate for the increased size of objects after the blur
            erode_radius = config.blur_size // 3
            if erode_radius % 2 == 0:
                erode_radius -= 1
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_radius, erode_radius))
            cv.erode(hsb_image, kernel, dst=hsb_image, iterations=1)

        saturation = hsb_image[..., 1]
        brightness = hsb_image[..., 2]

        if not config.gradient_corner_values.is_unit_gradient():
            gradient_application_start = timer()

            saturation *= self._generate_corner_gradient(saturation.shape, *config.gradient_corner_values)

            logging.debug(f'Gradient application time: {timer() - gradient_application_start}')

        _, saturation_thresholded = cv.threshold(saturation, config.saturation_threshold, 1, cv.THRESH_BINARY)
        _, brightness_thresholded = cv.threshold(brightness, config.brightness_threshold, 1, cv.THRESH_BINARY)

        mask = (saturation_thresholded > 0) & (brightness_thresholded > 0)

        if config.remove_small_object_size > 0:
            small_object_removing_start = timer()

            mask = self._mask_to_uint8(mask)
            self._remove_small_objects(mask, config.remove_small_object_size)

            logging.debug(f'Small object removing time: {timer() - small_object_removing_start}')

        if config.fill_hole_size > 0:
            holes_removing_start = timer()

            mask = self._mask_to_uint8(mask)
            self._fill_small_holes(mask, config.fill_hole_size)

            logging.debug(f'Small hole filling time: {timer() - holes_removing_start}')

        mask = self._mask_to_uint8(mask)
        logging.debug(f'Tissue segmentation time: {timer() - segmentation_start}')
        return mask

    @staticmethod
    def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
        return mask if mask.dtype == np.uint8 else mask.astype(np.uint8)

    @staticmethod
    def _remove_small_objects_and_holes_using_contours(
            mask: np.ndarray,
            min_object_size: int,
            min_hole_size: int,
            background_value: int = 0,
            foreground_value: int = 1,
    ):
        # Find contours and hierarchy
        contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        # Create a list to hold the contours to be removed or filled
        small_objects = []
        small_holes = []

        # Find small objects and holes
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            # Check if the contour is an object (parent is -1)
            if hierarchy[0][i][3] == -1:
                if area < min_object_size:
                    small_objects.append(contour)
            # The contour is a hole
            elif area < min_hole_size:
                small_holes.append(contour)

        # Remove small objects
        cv.fillPoly(mask, small_objects, color=(background_value,))
        # Fill small holes
        cv.fillPoly(mask, small_holes, color=(foreground_value,))

    @staticmethod
    def _remove_small_external_objects(mask: np.ndarray, min_object_size: int, background_value: int = 0):
        """ Removes only the extreme outer small objects. """
        small_objects = []
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Find small objects
        for contour in contours:
            if cv.contourArea(contour) < min_object_size:
                small_objects.append(contour)

        cv.fillPoly(mask, small_objects, color=(background_value,))

    @staticmethod
    def _remove_small_objects_using_morphology(mask: np.ndarray, min_object_size: int):
        small_object_removing_kernel_size = (min_object_size, min_object_size)
        small_object_removing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, small_object_removing_kernel_size)
        cv.morphologyEx(mask, cv.MORPH_OPEN, small_object_removing_kernel, dst=mask)

    @staticmethod
    def _remove_small_objects(mask: np.ndarray, min_object_size: int, background_value: int = 0, connectivity: int = 8):
        TissueSegmenter._modify_small_regions(
            mask, min_object_size, background_value, connectivity, remove_small_objects=True)

    @staticmethod
    def _fill_small_holes_using_morphology(mask: np.ndarray, fill_hole_size: int):
        fill_hole_kernel_size = (fill_hole_size, fill_hole_size)
        fill_hole_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, fill_hole_kernel_size)
        cv.morphologyEx(mask, cv.MORPH_CLOSE, fill_hole_kernel, dst=mask)

    @staticmethod
    def _fill_small_holes(mask: np.ndarray, min_hole_size: int, foreground_value: int = 1, connectivity: int = 8):
        TissueSegmenter._modify_small_regions(
            mask, min_hole_size, foreground_value, connectivity, remove_small_objects=False)

    @staticmethod
    def _modify_small_regions(
            mask: np.ndarray,
            min_region_size: int,
            value_to_set: int,
            connectivity: int = 8,
            remove_small_objects: bool = True,
    ):
        """
        :param remove_small_objects: True to remove small objects or False to fill small holes in the mask.
        """
        mask_to_analyze_connected_components = (
            mask if remove_small_objects else 1 - mask  # invert the mask to find holes instead of objects
        )
        label_count, labels, stats, _ = cv.connectedComponentsWithStats(
            mask_to_analyze_connected_components, connectivity=connectivity)
        # Create a mask where labels of small regions are marked as True
        skip_background = 1  # skip the first row, because it contains statistics of background
        small_region_label_mask = stats[skip_background:, cv.CC_STAT_AREA] < min_region_size
        # Set the pixels of small regions to `value_to_set`
        mask[np.isin(labels, np.nonzero(small_region_label_mask)[0] + skip_background)] = value_to_set

    @staticmethod
    def _generate_corner_gradient(
            shape: tuple[int, int],
            top_left: float = 1.0,
            top_right: float = 1.0,
            bottom_left: float = 1.0,
            bottom_right: float = 1.0,
            dtype: DTypeLike = np.float32,
    ) -> np.ndarray:
        rows, cols = shape
        # Create a linear gradient for each corner horizontally
        top_gradient = np.linspace(top_left, top_right, cols, dtype=dtype)
        bottom_gradient = np.linspace(bottom_left, bottom_right, cols, dtype=dtype)

        # Interpolate the values for the rest of the array vertically
        gradient = np.linspace(top_gradient, bottom_gradient, rows, dtype=dtype)

        return gradient
