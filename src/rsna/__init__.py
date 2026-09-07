"""RSNA knee abnormality detection.

Layout mirrors the pipeline rather than the file types:

    config      one dataclass holding every decision that changes what a pixel is
    data        competition metadata and report-derived label tables
    dicom       DICOM headers -> ordered, side-normalised, physically scaled images
    train       folds, augmentation, the fitting loop
    infer       load a weights package and write a submission

`dicom` is imported by both `train` and `infer`, deliberately: one definition of the
pixel path is the only thing that keeps training and inference from drifting apart.
"""
