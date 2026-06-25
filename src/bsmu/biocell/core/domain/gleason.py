from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class GleasonGrade(IntEnum):
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
        return self.primary.value + self.secondary.value

    def __str__(self) -> str:
        """Return formatted score like '3+4'."""
        return f'{self.primary.value}+{self.secondary.value}'
