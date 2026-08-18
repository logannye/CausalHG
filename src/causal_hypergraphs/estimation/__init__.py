"""Evaluate identified estimands against data, with their certificates discharged.

`identify` answers whether a mechanism-level query is answerable and returns a formula.
This package answers it *from a dataset*, and -- because positivity is a property of the
distribution rather than of the graph -- checks the certificates the formula carries
against the data actually in hand, naming the strata where they fail.
"""
from .dataset import Dataset, DatasetError, Point
from .estimator import (
    DISCHARGEABLE_CODES,
    Estimate,
    EstimationError,
    NotIdentified,
    SupportFailure,
    SupportReport,
    UnsupportedEstimand,
    estimate,
)

__all__ = [
    "DISCHARGEABLE_CODES",
    "Dataset",
    "DatasetError",
    "Estimate",
    "EstimationError",
    "NotIdentified",
    "Point",
    "SupportFailure",
    "SupportReport",
    "UnsupportedEstimand",
    "estimate",
]
