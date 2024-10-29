from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bsmu.vision.plugins.loaders.image.common import CommonImageFileLoader

if TYPE_CHECKING:
    from typing import Iterator


@dataclass
class Metrics:
    tp: int = 0  # True positives count
    tn: int = 0  # True negatives count
    fp: int = 0  # False positives count
    fn: int = 0  # False negatives count

    def __add__(self, other) -> Metrics:
        if not isinstance(other, Metrics):
            return NotImplemented
        return Metrics(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def __iadd__(self, other) -> Metrics:
        if not isinstance(other, Metrics):
            return NotImplemented
        self.tp += other.tp
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def is_valid(self) -> bool:
        return any([self.tp, self.tn, self.fp, self.fn])

    @property
    def are_masks_empty(self) -> bool:
        """
        :return: True if both the ground truth and prediction masks are empty
        """
        return self.tp == 0 and self.fp == 0 and self.fn == 0

    @property
    def is_gt_mask_empty(self) -> bool:
        """
        :return: True if the ground truth mask is empty
        """
        return self.tp == 0 and self.fn == 0

    @property
    def is_prediction_mask_empty(self) -> bool:
        """
        :return: True if the prediction mask is empty
        """
        return self.tp == 0 and self.fp == 0

    def _calculate_metric(self, dividend: float, divisor: float, zero_division: float = -1) -> float:
        if not self.is_valid:
            raise ValueError('Metrics are not valid. At least one value must be non-zero.')

        if self.are_masks_empty:
            return 1

        return dividend / divisor if divisor > 0 else zero_division

    def f1_score(self, zero_division: float = -1) -> float:
        return self._calculate_metric(2 * self.tp, 2 * self.tp + self.fp + self.fn, zero_division)

    def iou(self, zero_division: float = -1) -> float:
        return self._calculate_metric(self.tp, self.tp + self.fp + self.fn, zero_division)

    def sensitivity(self, zero_division: float = -1) -> float:
        return self._calculate_metric(self.tp, self.tp + self.fn, zero_division)

    def recall(self):
        return self.sensitivity()

    def specificity(self, zero_division: float = -1) -> float:
        return self._calculate_metric(self.tn, self.tn + self.fp, zero_division)

    def precision(self, zero_division: float = -1) -> float:
        return self._calculate_metric(self.tp, self.tp + self.fp, zero_division)


def load_mask(mask_path: Path) -> np.ndarray:
    loader = CommonImageFileLoader()
    mask = loader.load_file(mask_path)
    return mask.pixels


def filter_mask_classes(mask: np.ndarray, binarize: bool = True) -> np.ndarray:
    # mask[~np.isin(mask, [3, 4, 5])] = 0

    mask[(mask < 3) | (mask > 5)] = 0

    if binarize:
        mask[mask != 0] = 1

    return mask


def calculate_mask_metrics(gt_mask: np.ndarray, prediction_mask: np.ndarray) -> Metrics:
    # Calculating metrics using OpenCV works faster than using NumPy logical operations
    # and significantly faster than using sklearn.metrics

    # Calculate True Positives (TP)
    intersection = cv2.bitwise_and(gt_mask, prediction_mask)
    tp = cv2.countNonZero(intersection)

    # Calculate False Positives (FP)
    fp = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(gt_mask), prediction_mask))

    # Calculate False Negatives (FN)
    fn = cv2.countNonZero(cv2.bitwise_and(gt_mask, cv2.bitwise_not(prediction_mask)))

    # Calculate True Negatives (TN)
    union_count = tp + fp + fn
    tn = gt_mask.size - union_count

    return Metrics(tp=tp, tn=tn, fp=fp, fn=fn)


def calculate_mask_metrics_in_dir(
        masks_dir: Path,
        gt_dir_name: str,
        prediction_dir_names: list[str],
) -> Iterator[tuple[str, dict[str, Metrics]]]:

    for gt_mask_path in (masks_dir / gt_dir_name).iterdir():
        if not gt_mask_path.is_file() or gt_mask_path.suffix != '.png':
            continue

        print(f'Processing mask: {gt_mask_path.name}')
        gt_mask = load_mask(gt_mask_path)
        gt_mask = filter_mask_classes(gt_mask)

        prediction_dir_name_to_metrics = {}
        for prediction_dir_name in prediction_dir_names:
            prediction_mask_path = masks_dir / prediction_dir_name / gt_mask_path.name
            if not prediction_mask_path.is_file():
                raise FileNotFoundError(f'Prediction mask file not found: {prediction_mask_path}')

            prediction_mask = load_mask(prediction_mask_path)
            prediction_mask = filter_mask_classes(prediction_mask)

            metrics = calculate_mask_metrics(gt_mask, prediction_mask)
            prediction_dir_name_to_metrics[prediction_dir_name] = metrics

        yield gt_mask_path.name, prediction_dir_name_to_metrics


def save_to_csv_metrics(name_to_metrics: dict[str: Metrics], output_file_path: Path, name_column_name: str):
    with open(output_file_path, mode='w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        # Write header
        writer.writerow([
            name_column_name, 'TP', 'TN', 'FP', 'FN', 'F1-Score', 'IoU', 'Sensitivity', 'Specificity', 'Precision'])
        # Write data
        for name, metrics in name_to_metrics.items():
            metrics = [
                metrics.tp,
                metrics.tn,
                metrics.fp,
                metrics.fn,
                metrics.f1_score(),
                metrics.iou(),
                metrics.sensitivity(),
                metrics.specificity(),
                metrics.precision(),
            ]
            formatted_metrics = [value_to_str_with_comma_decimal_separator(m) for m in metrics]
            writer.writerow([
                name,
                *formatted_metrics,
            ])


def save_to_csv_metrics_per_mask(mask_name_to_metrics: dict[str, Metrics], output_file_path: Path):
    save_to_csv_metrics(mask_name_to_metrics, output_file_path, 'Mask Name')


def save_to_csv_total_metrics(prediction_dir_name_to_total_metrics: dict[str, Metrics], output_file_path: Path):
    save_to_csv_metrics(prediction_dir_name_to_total_metrics, output_file_path, 'Model Name')


def value_to_str_with_comma_decimal_separator(value: float) -> str:
    return f'{value:.4f}'.replace('.', ',')


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--masks-dir', type=Path, help='Path to the directory containing mask directories')
    args = arg_parser.parse_args()

    masks_dir = args.masks_dir
    print(f'Processing masks directory: {masks_dir}')

    gt_dir_name = 'masks'
    prediction_dir_names = [
        d.name for d in masks_dir.iterdir()
        if d.is_dir() and d.name != gt_dir_name and d.name.startswith('masks')
    ]

    masks_predictions_metrics = list(calculate_mask_metrics_in_dir(masks_dir, gt_dir_name, prediction_dir_names))

    metrics_dir = masks_dir / 'metrics'
    for prediction_dir_name in prediction_dir_names:
        mask_name_to_metrics = {
            mask_name: prediction_dir_name_to_metrics[prediction_dir_name]
            for mask_name, prediction_dir_name_to_metrics in masks_predictions_metrics
        }
        save_to_csv_metrics_per_mask(mask_name_to_metrics, metrics_dir / f'{prediction_dir_name}.csv')

    prediction_dir_to_total_metrics = defaultdict(Metrics)
    for _, prediction_dir_name_to_metrics in masks_predictions_metrics:
        for prediction_dir_name, metrics in prediction_dir_name_to_metrics.items():
            prediction_dir_to_total_metrics[prediction_dir_name] += metrics

    total_metrics_output_path = metrics_dir / 'total_metrics.csv'
    save_to_csv_total_metrics(prediction_dir_to_total_metrics, total_metrics_output_path)


if __name__ == '__main__':
    main()
