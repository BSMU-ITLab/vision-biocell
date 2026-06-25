from enum import IntEnum


class PixelClass(IntEnum):
    """Semantic classes for pixel segmentation in prostate cancer analysis."""
    UNLABELED = 0
    FOREGROUND = 1
    GLEASON_2 = 2
    GLEASON_3 = 3
    GLEASON_4 = 4
    GLEASON_5 = 5
    SEMINAL_VESICLES = 6
    NERVE_TRUNK = 7
    NOT_CANCER = 8        # Benign tissue (explicitly marked)
    IGNORE = 9            # Excluded from training/analysis
    BACKGROUND = 255      # Slide background (non-tissue)
