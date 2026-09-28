"""Versioned, data-only persistence for the supported public model objects.

The reader constructs only explicitly registered package dataclasses. It never
imports a module named by a payload or deserializes executable functions.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SCHEMA_VERSION = 1


class SerializationError(ValueError):
    """Unsupported object, malformed payload, or unsupported schema version."""


def _registry() -> dict[str, type]:
    # An allow-list controlled by the library, never by input data. Optional
    # modules become available as their public contracts are implemented.
    modules = (
        "graph.model",
        "graph.incidence",
        "queries",
        "expression.ast",
        "identification.queries",
        "identification.results",
        "identification.pearl_id",
        "identification.t7",
        "identification.shpitser",
    )
    result: dict[str, type] = {}
    for name in modules:
        module_name = f"causal_hypergraphs.{name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            continue
        for cls in vars(module).values():
            if isinstance(cls, type) and is_dataclass(cls) and cls.__module__ == module_name:
                result[f"{name}.{cls.__name__}"] = cls
    return result


def _encode(value: Any, registry: dict[str, type]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("Nonfinite numbers cannot be serialized.")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        names = [name for name, cls in registry.items() if type(value) is cls]
        if not names:
            raise SerializationError(f"Unregistered object type: {type(value).__name__}")
        return {
            "type": names[0],
            "fields": {
                field.name: _encode(getattr(value, field.name), registry)
                for field in fields(value)
                if field.init
            },
        }
    if isinstance(value, Mapping):
        entries = [[_encode(key, registry), _encode(item, registry)] for key, item in value.items()]
        entries.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"mapping": entries}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_encode(item, registry) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return {type(value).__name__: items}
    raise SerializationError(
        f"Cannot persist {type(value).__name__}; executable bindings stay external."
    )


def _decode(value: Any, registry: dict[str, type]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("Nonfinite number in payload.")
        return value
    if not isinstance(value, dict):
        raise SerializationError("Encoded values must be primitives or tagged objects.")
    if set(value) == {"type", "fields"}:
        cls = registry.get(value["type"])
        if cls is None or not isinstance(value["fields"], dict):
            raise SerializationError("Unknown object type or malformed fields.")
        allowed = {field.name for field in fields(cls) if field.init}
        if not set(value["fields"]) <= allowed:
            raise SerializationError(f"Unknown fields for {value['type']}.")
        kwargs = {name: _decode(item, registry) for name, item in value["fields"].items()}
        if value["type"] == "identification.queries.ReplaceMechanism":
            incidence = kwargs.pop("incidence", None)
            if incidence is not None:
                kwargs["replacement"] = incidence
        try:
            return cls(**kwargs)
        except (TypeError, ValueError, KeyError) as error:
            raise SerializationError(f"Invalid {value['type']}: {error}") from error
    if len(value) != 1:
        raise SerializationError("Malformed tagged value.")
    tag, items = next(iter(value.items()))
    if not isinstance(items, list):
        raise SerializationError("Tagged collections require an array.")
    if tag == "mapping":
        decoded = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise SerializationError("Malformed mapping entry.")
            key, item = (_decode(part, registry) for part in pair)
            try:
                if key in decoded:
                    raise SerializationError("Duplicate mapping key.")
                decoded[key] = item
            except TypeError as error:
                raise SerializationError("Unhashable mapping key.") from error
        return decoded
    constructors = {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset}
    if tag not in constructors:
        raise SerializationError(f"Unknown collection tag: {tag}")
    try:
        return constructors[tag](_decode(item, registry) for item in items)
    except TypeError as error:
        raise SerializationError("Invalid collection member.") from error


def to_dict(value: Any) -> dict[str, Any]:
    """Return a schema-versioned document containing supported data objects."""
    try:
        library_version = version("causal-hypergraphs")
    except PackageNotFoundError:
        library_version = "uninstalled"
    return {
        "schema_version": SCHEMA_VERSION,
        "library_version": library_version,
        "payload": _encode(value, _registry()),
    }


def from_dict(document: Mapping[str, Any]) -> Any:
    """Restore a document without executing or resolving external code."""
    if set(document) != {"schema_version", "library_version", "payload"}:
        raise SerializationError("Invalid document envelope.")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise SerializationError(f"Unsupported schema version: {document['schema_version']}")
    if not isinstance(document["library_version"], str):
        raise SerializationError("Invalid library version.")
    try:
        return _decode(document["payload"], _registry())
    except (TypeError, KeyError) as error:
        raise SerializationError("Malformed object payload.") from error


def dumps(value: Any, *, indent: int | None = None) -> str:
    """Serialize supported objects deterministically as JSON."""
    return json.dumps(to_dict(value), sort_keys=True, allow_nan=False, indent=indent)


def loads(text: str) -> Any:
    """Parse a supported JSON document. Unknown versions/types are rejected."""
    try:
        document = json.loads(text)
    except (ValueError, TypeError) as error:
        raise SerializationError("Invalid JSON.") from error
    if not isinstance(document, dict):
        raise SerializationError("Expected a document object.")
    return from_dict(document)
