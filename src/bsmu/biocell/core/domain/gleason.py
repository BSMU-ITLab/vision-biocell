from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class GleasonGrade(IntEnum):
    G3 = 3
    G4 = 4
    G5 = 5


class IsupGradeGroup(IntEnum):
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5


@dataclass(frozen=True)
class GleasonScore:
    """Gleason score with primary and secondary grades."""
    primary: GleasonGrade
    secondary: GleasonGrade

    @property
    def total(self) -> int:
        """Sum of primary and secondary grades."""
        return self.primary + self.secondary

    @property
    def isup_grade_group(self) -> IsupGradeGroup:
        """Calculate ISUP Grade Group from this Gleason score.

        Mapping:
          Group 1: 3+3 (sum <= 6)
          Group 2: 3+4
          Group 3: 4+3
          Group 4: 4+4, 3+5, 5+3 (sum = 8)
          Group 5: 4+5, 5+4, 5+5 (sum = 9-10)
        """
        if self.primary == GleasonGrade.G3 and self.secondary == GleasonGrade.G3:
            return IsupGradeGroup.G1

        if self.primary == GleasonGrade.G3 and self.secondary == GleasonGrade.G4:
            return IsupGradeGroup.G2

        if self.primary == GleasonGrade.G4 and self.secondary == GleasonGrade.G3:
            return IsupGradeGroup.G3

        if self.total == 8:
            return IsupGradeGroup.G4

        if self.total == 9 or self.total == 10:
            return IsupGradeGroup.G5

        raise ValueError(f'Unexpected Gleason score: {self}')

    def __str__(self) -> str:
        """Return formatted score like '3+4'."""
        return f'{self.primary}+{self.secondary}'
