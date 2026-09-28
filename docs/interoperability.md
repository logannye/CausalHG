# Graph and table interoperability

All imports preserve directed incidence rather than clique-expanding joint mechanisms.
Only `CausalModelSpec`/`to_mechanism_graph` applies a causal profile. Conversions do not
infer causal direction, mechanism independence, or observational identifiability.

```python
from causal_hypergraphs import CausalModelSpec
from causal_hypergraphs.io import from_incidence, to_incidence

structure = from_incidence(
    [{"edge": "step", "node": "X", "role": "input"},
     {"edge": "step", "node": "Y", "role": "output"}],
    nodes=("X", "Y", "isolated"),
    edge_metadata={"step": {"description": "example mechanism"}},
)
assert from_incidence(**to_incidence(structure)) == structure
model = CausalModelSpec(structure).to_mechanism_graph()
```

Native role records use `edge`, `node`, and `role` (`input`/`output`). Export includes
all node/edge IDs, graph metadata and edge metadata, preserving isolated nodes and empty
edges. Generic incidence can place a node in both roles; causal conversion rejects it.

| Adapter | Identity, metadata and direction contract |
|---|---|
| `from_networkx` / `to_networkx` | Simple directed bipartite graph, every node marked `kind="variable"` or `"mechanism"`. Export uses namespaced nodes, preserving parallel mechanism identity. Arbitrary arc attributes are rejected, never silently dropped. Native round trips preserve metadata and isolates/empty edges. |
| `from_xgi` / `to_xgi` | XGI `DiHypergraph` tail/head incidence. Undirected input is refused. Native round trips preserve metadata/isolation; external graph-level attributes require the explicit `metadata=` argument, disclosed in conversion notes. |
| `from_hypernetx` | Import membership only with `roles={(edge,node): "input"/"output"/"both"}` for every incidence. Node, edge and cell attributes are retained; graph attributes require `metadata=`. Nodes/edges already absent from the source cannot be recovered. |

External graph imports return `ConversionResult(graph, node_ids, edge_ids, notes)`.
ID mappings go from canonical IDs back to source IDs. External IDs must be nonempty
strings or explicitly converted with `id_codec`; collisions are rejected, so integer
`1` and string `"1"` cannot silently become the same node. Node and edge namespaces
are distinct in incidence; the mechanism compiler additionally requires disjoint names.
Metadata must be finite JSON-like data. Unsupported custom objects cause a validation
error and need explicit application-side conversion.

Native NetworkX/XGI exports carry the reserved `causalhg_format=1` marker. Their
metadata wrappers preserve the original graph snapshot. Import rejects additional
attributes attached outside those wrappers rather than silently dropping them. Edit
the wrapper contents, or clear the format marker and use the external-import contract.

`records_from_dataframe(frame)` requires unique string columns and no missing values.
`records_from_array(array, columns)` requires explicit unique labels, two-dimensional
rows, and no missing/nonfinite scalar values, including optional-library missing-value
sentinels. Feed the returned records to `Dataset.from_records`
or `fit_linear_gaussian`. These adapters do not infer sampling units or variable types.

Install `.[interop]` to use the third-party integrations. Core imports and native
incidence/JSON do not load any optional graph/table dependency.
