from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bsmu.biocell.core.domain import GleasonScore

if TYPE_CHECKING:
    from bsmu.biocell.core.domain import GleasonGrade, IsupGradeGroup


@dataclass(frozen=True)
class GleasonGradeDistribution:
    """Distribution of Gleason grade percentages for a single analysis method."""
    grade_to_percentage: dict[GleasonGrade, float] = field(default_factory=dict)

    @property
    def score(self) -> GleasonScore | None:
        """Calculate Gleason score from this distribution.

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
            [(grade, pct) for grade, pct in self.grade_to_percentage.items() if pct > 0.0],
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

    @property
    def isup_grade_group(self) -> IsupGradeGroup | None:
        """Calculate ISUP Grade Group from this distribution.

        Returns None if no cancer detected.
        """
        score = self.score
        return score.isup_grade_group if score is not None else None


@dataclass(frozen=True)
class GleasonAnalysisReport:
    """Complete ISUP analysis report with results from all methods."""
    linear: GleasonGradeDistribution
    linear_through: GleasonGradeDistribution
    area: GleasonGradeDistribution | None = None
