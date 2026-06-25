from bsmu.biocell.core.domain import GleasonGrade, PixelClass


GLEASON_TO_PIXEL_CLASS = {
    GleasonGrade.G3: PixelClass.GLEASON_3,
    GleasonGrade.G4: PixelClass.GLEASON_4,
    GleasonGrade.G5: PixelClass.GLEASON_5,
}
