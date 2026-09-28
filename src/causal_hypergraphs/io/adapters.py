"""Optional graph and table adapters with explicit roles and identity mappings."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from causal_hypergraphs.estimation.dataset import DatasetError, _category
from causal_hypergraphs.graph.incidence import DirectedHyperedge, DirectedHypergraph, _plain


@dataclass(frozen=True)
class ConversionResult:
    graph: DirectedHypergraph
    node_ids: Mapping[str, Any]
    edge_ids: Mapping[str, Any]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", MappingProxyType(dict(self.node_ids)))
        object.__setattr__(self, "edge_ids", MappingProxyType(dict(self.edge_ids)))


def _ids(values: Iterable[Any], codec: Callable[[Any], str] | None) -> tuple[dict, dict]:
    forward, reverse = {}, {}
    for value in values:
        name = codec(value) if codec else value
        if not isinstance(name, str) or not name.strip():
            raise TypeError("External IDs require nonempty strings or an explicit id_codec.")
        if name in reverse and reverse[name] != value:
            raise ValueError(f"Identifier codec collision: {name!r}")
        forward[value], reverse[name] = name, value
    return forward, reverse


def from_incidence(
    records: Iterable[Mapping[str, str]],
    *,
    nodes: Iterable[str] = (),
    edge_ids: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
    edge_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> DirectedHypergraph:
    """Import edge/node/role rows. Explicit IDs preserve isolated nodes/empty edges."""
    vertices = set(nodes)
    edges = {edge: {"inputs": [], "outputs": []} for edge in edge_ids}
    for row in records:
        if set(row) != {"edge", "node", "role"} or row["role"] not in {"input", "output"}:
            raise ValueError("Incidence rows require edge, node, and input/output role.")
        name, node = row["edge"], row["node"]
        vertices.add(node)
        spec = edges.setdefault(name, {"inputs": [], "outputs": []})
        spec[row["role"] + "s"].append(node)
    extra = set(edge_metadata or {}) - set(edges)
    if extra:
        raise ValueError(f"Metadata references unknown edges: {sorted(extra)}")
    return DirectedHypergraph(
        vertices,
        {
            name: DirectedHyperedge(
                name, spec["inputs"], spec["outputs"], (edge_metadata or {}).get(name)
            )
            for name, spec in edges.items()
        },
        metadata,
    )


def to_incidence(graph: DirectedHypergraph) -> dict[str, Any]:
    """Export all IDs plus role rows; isolated nodes and empty edges are explicit."""
    return {
        "nodes": graph.nodes,
        "edge_ids": tuple(graph.edges),
        "records": tuple(
            {"edge": name, "node": node, "role": role}
            for name, edge in graph.edges.items()
            for role, members in (("input", edge.inputs), ("output", edge.outputs))
            for node in members
        ),
        "metadata": _plain(graph.metadata),
        "edge_metadata": {name: _plain(edge.metadata) for name, edge in graph.edges.items()},
    }


def from_networkx(graph: Any, *, id_codec: Callable[[Any], str] | None = None) -> ConversionResult:
    """Import a directed bipartite graph with kind=variable/mechanism on every node."""
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError(
            "Expected a simple directed bipartite graph; parallel mechanisms use nodes."
        )
    variables, mechanisms = [], []
    for node, attrs in graph.nodes(data=True):
        kind = attrs.get("kind")
        if kind not in {"variable", "mechanism"}:
            raise ValueError(f"Node {node!r} needs kind='variable' or 'mechanism'.")
        (variables if kind == "variable" else mechanisms).append(node)
    native = graph.graph.get("causalhg_format") == 1
    if native:
        if set(graph.graph) != {"causalhg_format", "causalhg_metadata"}:
            raise ValueError("Native NetworkX graph attributes must use causalhg_metadata.")
        for node, attrs in graph.nodes(data=True):
            expected = {"kind", "native_id"}
            if attrs["kind"] == "mechanism":
                expected.add("attributes")
            if set(attrs) != expected:
                raise ValueError(
                    f"Native NetworkX node {node!r} has unexpected or missing attributes; "
                    "edit the exported metadata wrapper or remove the native format marker."
                )
    codec = (lambda node: graph.nodes[node]["native_id"]) if native else id_codec
    vn, reverse_v = _ids(variables, codec)
    mn, reverse_m = _ids(mechanisms, codec)
    for source, target, attrs in graph.edges(data=True):
        if not ((source in vn and target in mn) or (source in mn and target in vn)):
            raise ValueError("Every arc must join a variable and a mechanism.")
        if attrs:
            raise ValueError(
                "Arc metadata needs an explicit incidence metadata model; not discarded."
            )
    edges = {}
    for node in mechanisms:
        attrs = graph.nodes[node]
        metadata = attrs.get("attributes", {}) if native else dict(attrs)
        edges[mn[node]] = DirectedHyperedge(
            mn[node],
            (vn[v] for v in graph.predecessors(node)),
            (vn[v] for v in graph.successors(node)),
            metadata,
        )
    metadata = (
        graph.graph["causalhg_metadata"]
        if native
        else {
            "source_attributes": dict(graph.graph),
            "node_attributes": {vn[v]: dict(graph.nodes[v]) for v in variables},
        }
    )
    return ConversionResult(DirectedHypergraph(vn.values(), edges, metadata), reverse_v, reverse_m)


def to_networkx(graph: DirectedHypergraph) -> Any:
    """Export namespaced variable/mechanism nodes without clique expansion."""
    nx = importlib.import_module("networkx")

    result = nx.DiGraph(causalhg_format=1, causalhg_metadata=_plain(graph.metadata))
    for node in graph.nodes:
        result.add_node(("variable", node), kind="variable", native_id=node)
    for name, edge in graph.edges.items():
        result.add_node(
            ("mechanism", name), kind="mechanism", native_id=name, attributes=_plain(edge.metadata)
        )
        result.add_edges_from((("variable", v), ("mechanism", name)) for v in edge.inputs)
        result.add_edges_from((("mechanism", name), ("variable", v)) for v in edge.outputs)
    return result


def from_xgi(
    graph: Any,
    *,
    id_codec: Callable[[Any], str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ConversionResult:
    """Import directed XGI incidence; caller supplies any external graph-level metadata."""
    if not hasattr(graph.edges, "dimembers"):
        raise ValueError("An undirected XGI hypergraph needs explicit causal roles.")
    xgi = importlib.import_module("xgi")
    # XGI's mapping API raises XGIError for an absent graph attribute. Its public
    # HIF conversion exposes the complete attributes without relying on private state.
    graph_attrs = xgi.to_hif_dict(graph)["metadata"]
    native = graph_attrs.get("causalhg_format") == 1
    if native:
        if set(graph_attrs) != {"causalhg_format", "causalhg_metadata"}:
            raise ValueError("Native XGI graph attributes must use causalhg_metadata.")
        if metadata is not None:
            raise ValueError("Native XGI graph metadata must be edited in causalhg_metadata.")
        if any(graph.nodes[node] for node in graph.nodes):
            raise ValueError(
                "Native XGI node attributes are not represented; remove the native format "
                "marker for an external import that retains them."
            )
        if any(set(graph.edges[edge]) != {"causalhg_metadata"} for edge in graph.edges):
            raise ValueError("Native XGI edge attributes must use causalhg_metadata.")
    vn, reverse_v = _ids(graph.nodes, id_codec)
    mn, reverse_m = _ids(graph.edges, id_codec)
    edges = {}
    for name in graph.edges:
        inputs, outputs = graph.edges.dimembers(name)
        attrs = dict(graph.edges[name])
        edges[mn[name]] = DirectedHyperedge(
            mn[name],
            (vn[v] for v in inputs),
            (vn[v] for v in outputs),
            attrs.get("causalhg_metadata", {}) if native else attrs,
        )
    meta = (
        graph_attrs["causalhg_metadata"]
        if native
        else {
            "source_attributes": dict(metadata or {}),
            "node_attributes": {vn[v]: dict(graph.nodes[v]) for v in graph.nodes},
        }
    )
    notes = () if native else ("Graph-level attributes require the explicit metadata argument.",)
    return ConversionResult(
        DirectedHypergraph(vn.values(), edges, meta), reverse_v, reverse_m, notes
    )


def to_xgi(graph: DirectedHypergraph) -> Any:
    """Export a directed XGI graph, preserving isolated nodes and edge metadata."""
    xgi = importlib.import_module("xgi")

    result = xgi.DiHypergraph()
    result.add_nodes_from(graph.nodes)
    for name, edge in graph.edges.items():
        result.add_edge(
            (edge.inputs, edge.outputs), idx=name, causalhg_metadata=_plain(edge.metadata)
        )
    result["causalhg_metadata"] = _plain(graph.metadata)
    result["causalhg_format"] = 1
    return result


def from_hypernetx(
    graph: Any,
    *,
    roles: Mapping[tuple[Any, Any], str],
    id_codec: Callable[[Any], str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ConversionResult:
    """Import membership with explicitly supplied input/output/both incidence roles."""
    incidence = graph.incidence_dict
    vn, reverse_v = _ids(graph.nodes, id_codec)
    mn, reverse_m = _ids(incidence, id_codec)
    expected = {(edge, node) for edge, members in incidence.items() for node in members}
    if set(roles) != expected:
        raise ValueError("Supply one role for every incidence, with no extra entries.")
    edges = {}
    for name, members in incidence.items():
        inputs, outputs = [], []
        for node in members:
            role = roles[(name, node)]
            if role not in {"input", "output", "both"}:
                raise ValueError("Incidence roles must be input, output, or both.")
            if role in {"input", "both"}:
                inputs.append(vn[node])
            if role in {"output", "both"}:
                outputs.append(vn[node])
        edges[mn[name]] = DirectedHyperedge(
            mn[name], inputs, outputs, dict(graph.edges[name].properties)
        )
    meta = {
        "source_attributes": dict(metadata or {}),
        "node_attributes": {vn[node]: dict(graph.nodes[node].properties) for node in graph.nodes},
        "incidence_attributes": {
            mn[edge]: {vn[node]: graph.get_cell_properties(edge, node) for node in members}
            for edge, members in incidence.items()
        },
    }
    notes = (
        "Undirected membership needs explicit roles; graph-level attributes require metadata.",
    )
    return ConversionResult(
        DirectedHypergraph(vn.values(), edges, meta), reverse_v, reverse_m, notes
    )


def records_from_dataframe(frame: Any) -> list[dict[str, Any]]:
    """Adapt a labeled table without silently dropping or imputing missing data."""
    if not frame.columns.is_unique:
        raise ValueError("DataFrame columns must be unique.")
    if any(not isinstance(name, str) or not name for name in frame.columns):
        raise TypeError("DataFrame column names must be nonempty strings.")
    if frame.isna().any().any():
        raise ValueError("Missing data require an explicit model or preprocessing policy.")
    return frame.to_dict(orient="records")


def records_from_array(array: Any, columns: Iterable[str]) -> list[dict[str, Any]]:
    """Import a two-dimensional array with explicit labels and no missing values."""
    names = tuple(columns)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Array columns must be nonempty string identifiers.")
    if len(set(names)) != len(names):
        raise ValueError("Array column names must be unique.")
    if getattr(array, "ndim", 2) != 2:
        raise ValueError("Expected a two-dimensional array.")
    rows = array.tolist() if hasattr(array, "tolist") else list(array)
    result = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(names):
            raise ValueError("Every row must match the declared columns.")
        try:
            for name, value in zip(names, row, strict=True):
                _category(value, f"Array column {name!r}")
        except DatasetError as error:
            raise ValueError(
                "Missing, nonfinite, or nonscalar values require an explicit preprocessing "
                "policy."
            ) from error
        result.append(dict(zip(names, row, strict=True)))
    return result
