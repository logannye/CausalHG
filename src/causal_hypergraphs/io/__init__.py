"""Data-only persistence and optional graph/table interoperability."""

from .adapters import (
    ConversionResult,
    from_hypernetx,
    from_incidence,
    from_networkx,
    from_xgi,
    records_from_array,
    records_from_dataframe,
    to_incidence,
    to_networkx,
    to_xgi,
)
from .serialization import SCHEMA_VERSION, SerializationError, dumps, from_dict, loads, to_dict

__all__ = [
    "SCHEMA_VERSION",
    "SerializationError",
    "dumps",
    "from_dict",
    "loads",
    "to_dict",
    "ConversionResult",
    "from_hypernetx",
    "from_incidence",
    "from_networkx",
    "from_xgi",
    "records_from_array",
    "records_from_dataframe",
    "to_incidence",
    "to_networkx",
    "to_xgi",
]
