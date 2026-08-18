"""Evaluate identified estimands against data, with their certificates discharged.

`identify` answers whether a mechanism-level query is answerable and returns a formula.
This package answers it *from a dataset*, and -- because positivity is a property of the
distribution rather than of the graph -- checks the certificates the formula carries
against the data actually in hand, naming the strata where they fail.
"""
from .dataset import Dataset, DatasetError, Point
from .empirical import EmpiricalModel
from .estimator import (
    DISCHARGEABLE_CODES,
    METHODS,
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
    "METHODS",
    "Dataset",
    "DatasetError",
    "EmpiricalModel",
    "Estimate",
    "EstimationError",
    "NotIdentified",
    "Point",
    "SupportFailure",
    "SupportReport",
    "UnsupportedEstimand",
    "estimate",
]
