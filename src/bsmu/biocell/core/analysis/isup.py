from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bsmu.biocell.core.converters.gleason_to_pixel import GLEASON_TO_PIXEL_CLASS
from bsmu.biocell.core.data.vector.shapes.cancer_span import CancerSpan
from bsmu.biocell.core.domain import GleasonGrade, GleasonScore, PixelClass

if TYPE_CHECKING:
    from typing import Sequence

    from bsmu.vision.core.data.raster import Raster
    from bsmu.vision.core.data.vector.shapes import Polyline


@dataclass(frozen=True)
class IsupResult:
    """Result of ISUP analysis with percentages for each Gleason grade."""
    linear: dict[GleasonGrade, float] = field(default_factory=dict)
    linear_through: dict[GleasonGrade, float] = field(default_factory=dict)
    area: dict[GleasonGrade, float] | None = None


def analyze(polylines: Sequence[Polyline], mask: Raster | None = None) -> IsupResult:
    """Analyze completed polylines and optionally mask for ISUP result.

    Only completed polylines and their completed CancerSpan children are considered.
    If mask is provided, area percentages are calculated; otherwise area is None.
    """
    completed_polylines = [p for p in polylines if p.is_completed]

    if not completed_polylines:
        return _empty_result()

    total_tissue_length = sum(p.length for p in completed_polylines)

    if total_tissue_length <= 0.0:
        return _empty_result()

    linear = _calculate_linear(completed_polylines, total_tissue_length)
    linear_through = _calculate_linear_through(completed_polylines, total_tissue_length)
    area = _calculate_area(mask) if mask is not None else None

    return IsupResult(
        linear=linear,
        linear_through=linear_through,
        area=area,
    )


def _empty_result() -> IsupResult:
    """Return result with zero percentages for all grades."""
    zero_dict = {grade: 0.0 for grade in GleasonGrade}
    return IsupResult(
        linear=zero_dict.copy(),
        linear_through=zero_dict.copy(),
        area=None,
    )


def _calculate_linear(polylines: Sequence[Polyline], total_length: float) -> dict[GleasonGrade, float]:
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

    return {
        grade: (length / total_length) * 100.0
        for grade, length in grade_to_total_merged_length.items()
    }


def _calculate_linear_through(polylines: Sequence[Polyline], total_length: float) -> dict[GleasonGrade, float]:
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

    return {
        grade: (length / total_length) * 100.0
        for grade, length in grade_to_total_bounding_length.items()
    }


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


def _calculate_area(mask: Raster) -> dict[GleasonGrade, float]:
    """Calculate area percentages from mask.

    Returns dict with percentages for each Gleason grade.
    """
    pixels = mask.pixels
    if pixels is None:
        return {grade: 0.0 for grade in GleasonGrade}

    tissue_area = np.sum(~np.isin(pixels, [PixelClass.BACKGROUND, PixelClass.IGNORE]))
    if tissue_area == 0:
        return {grade: 0.0 for grade in GleasonGrade}

    return {
        grade: (np.sum(pixels == GLEASON_TO_PIXEL_CLASS[grade]) / tissue_area) * 100.0
        for grade in GleasonGrade
    }


def calculate_gleason_score(grade_to_percentage: dict[GleasonGrade, float]) -> GleasonScore | None:
    """Calculate Gleason score from grade percentages.

    Primary grade: highest percentage (if tied, highest Gleason grade wins).
    Secondary grade (by priority):
      1. If only one grade present: Primary (duplicated).
      2. Highest grade present if > Primary (regardless of percentage).
      3. Second highest by percentage if > 5%.
      4. Primary (duplicated) otherwise.

    Returns None if no cancer detected (empty dict or all percentages are 0).
    """
    # Secondary sort key (grade) resolves ties in percentage
    sorted_grades = sorted(
        [(grade, pct) for grade, pct in grade_to_percentage.items() if pct > 0.0],
        key=lambda x: (x[1], x[0]),
        reverse=True,
    )

    if not sorted_grades:
        return None

    primary = sorted_grades[0][0]

    if len(sorted_grades) == 1:
        return GleasonScore(primary=primary, secondary=primary)

    higher_grades = [grade for grade, _ in sorted_grades[1:] if grade > primary]
    if higher_grades:
        return GleasonScore(primary=primary, secondary=max(higher_grades))

    second_grade, second_pct = sorted_grades[1]
    if second_pct > 5.0:
        return GleasonScore(primary=primary, secondary=second_grade)

    return GleasonScore(primary=primary, secondary=primary)
