from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bsmu.biocell.core.converters.gleason_to_pixel import GLEASON_TO_PIXEL_CLASS
from bsmu.biocell.core.data.vector.shapes.cancer_span import CancerSpan
from bsmu.biocell.core.domain import GleasonGrade, PixelClass
from bsmu.biocell.core.domain.gleason_analysis import GleasonAnalysisReport, GleasonGradeDistribution

if TYPE_CHECKING:
    from typing import Sequence

    from bsmu.vision.core.data.raster import Raster
    from bsmu.vision.core.data.vector.shapes import Polyline


def analyze(polylines: Sequence[Polyline], mask: Raster | None = None) -> GleasonAnalysisReport:
    """Analyze completed polylines and optionally mask for ISUP result.

    Only completed polylines and their completed CancerSpan children are considered.
    If mask is provided, area percentages are calculated; otherwise area is None.
    """
    completed_polylines = [p for p in polylines if p.is_completed]
    total_tissue_length = sum(p.length for p in completed_polylines)

    if total_tissue_length > 0.0:
        linear = _calculate_linear(completed_polylines, total_tissue_length)
        linear_through = _calculate_linear_through(completed_polylines, total_tissue_length)
    else:
        linear = None
        linear_through = None

    area = _calculate_area(mask) if mask is not None else None

    return GleasonAnalysisReport(
        linear=linear,
        linear_through=linear_through,
        area=area,
    )


def _calculate_linear(polylines: Sequence[Polyline], total_length: float) -> GleasonGradeDistribution:
    """Calculate linear percentages by merging intervals per polyline."""
    grade_to_total_merged_length: dict[GleasonGrade, float] = {
        grade: 0.0 for grade in GleasonGrade
    }

    for polyline in polylines:
        grade_to_intervals: dict[GleasonGrade, list[tuple[float, float]]] = {
            grade: [] for grade in GleasonGrade
        }

        for child in polyline.child_shapes:
            if isinstance(child, CancerSpan) and child.is_completed:
                grade = child.gleason_grade
                start = min(child.start_node.arc_length, child.end_node.arc_length)
                end = max(child.start_node.arc_length, child.end_node.arc_length)
                grade_to_intervals[grade].append((start, end))

        # Merge intervals locally for the current polyline and add to total
        for grade, intervals in grade_to_intervals.items():
            if intervals:
                merged_intervals = _merge_intervals(intervals)
                merged_length = sum(end - start for start, end in merged_intervals)
                grade_to_total_merged_length[grade] += merged_length

    grade_to_percentage = {
        grade: (length / total_length) * 100.0
        for grade, length in grade_to_total_merged_length.items()
    }
    return GleasonGradeDistribution(grade_to_percentage=grade_to_percentage)


def _calculate_linear_through(polylines: Sequence[Polyline], total_length: float) -> GleasonGradeDistribution:
    """Calculate linear-through percentages using bounding box per polyline."""
    grade_to_total_bounding_length: dict[GleasonGrade, float] = {
        grade: 0.0 for grade in GleasonGrade
    }

    for polyline in polylines:
        grade_to_spans: dict[GleasonGrade, list[CancerSpan]] = {
            grade: [] for grade in GleasonGrade
        }

        for child in polyline.child_shapes:
            if isinstance(child, CancerSpan) and child.is_completed:
                grade_to_spans[child.gleason_grade].append(child)

        # Calculate bounding box locally for the current polyline
        for grade, grade_spans in grade_to_spans.items():
            if grade_spans:
                min_start = min(
                    min(s.start_node.arc_length, s.end_node.arc_length) for s in grade_spans
                )
                max_end = max(
                    max(s.start_node.arc_length, s.end_node.arc_length) for s in grade_spans
                )
                grade_to_total_bounding_length[grade] += (max_end - min_start)

    grade_to_percentage = {
        grade: (length / total_length) * 100.0
        for grade, length in grade_to_total_bounding_length.items()
    }
    return GleasonGradeDistribution(grade_to_percentage=grade_to_percentage)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals and return the merged list."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged_intervals = []

    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end

    merged_intervals.append((current_start, current_end))
    return merged_intervals


def _calculate_area(mask: Raster) -> GleasonGradeDistribution | None:
    """Calculate area percentages from mask."""
    pixels = mask.pixels
    if pixels is None:
        return None

    tissue_area = np.sum(~np.isin(pixels, [PixelClass.BACKGROUND, PixelClass.IGNORE]))
    if tissue_area == 0:
        return None

    grade_to_percentage = {
        grade: (np.sum(pixels == GLEASON_TO_PIXEL_CLASS[grade]) / tissue_area) * 100.0
        for grade in GleasonGrade
    }
    return GleasonGradeDistribution(grade_to_percentage=grade_to_percentage)
