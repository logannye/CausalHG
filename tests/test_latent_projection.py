"""The projection is the foundation, so it is checked against the theorem it implements.

`latent_project_to_variable_admg` turns a mechanism graph into a Pearl ADMG over the
observed variables. Everything the T7 track does runs on its output, so an error here is
not a wrong answer in one branch -- it is a *confident* wrong answer everywhere, because a
projection that drops confounding produces a simpler graph than reality and identification
algorithms are sound only relative to the graph they are given.

There is no need to invent an oracle for it. The repository already proves what the answer
must be. `THEOREM_T4_T5.md` Proposition T4.0 states that when there are no hidden variables
the districts of the latent-projected graph are exactly

    {out(m) : m in E}  union  {{v} : v exogenous}

and its proof says why: "Bidirected edges arise only by projecting out a mechanism noise
u_m, whose children are exactly out(m); projection therefore yields a complete bidirected
component on out(m)". A mechanism has *one* shared noise -- that is the whole content of
the hypergraph formalism -- so its outputs are confounded with each other and with nothing
else. C4 gives each variable one producing mechanism, so the components do not overlap.

That proposition is a specification, and it is the gate: an implementation that fails it
contradicts the theorem the library is built on.

The four structural cases below are the ways a projection can be wrong, stated separately
so a failure names its own cause rather than only moving the aggregate.
"""
from __future__ import annotations

from causal_hypergraphs import MechanismGraph, latent_project_to_variable_admg
from causal_hypergraphs.examples import frontdoor_hidden_boundary_graph


def _districts(graph: MechanismGraph) -> set[tuple[str, ...]]:
    return set(latent_project_to_variable_admg(graph).districts())


# --- the theorem ------------------------------------------------------------------


def test_the_districts_are_the_mechanism_output_sets() -> None:
    """Proposition T4.0, executed rather than cited.

    Swept over generated models rather than fixtured, because the proposition is a claim
    about every C1-C4 graph and a fixture would only ever exhibit the shapes its author
    thought of.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    multi_output_models = 0
    for seed in range(300):
        model = generate_model(seed, allow_hidden=False)  # V_lat empty: T4.0's hypothesis
        graph = model.graph()
        expected = {
            tuple(sorted(graph.get_mechanism(name).outputs)) for name in graph.mechanisms
        }
        expected |= {(variable,) for variable in graph.exogenous_variables}
        assert _districts(graph) == expected, f"seed {seed}"
        checked += 1
        if any(len(district) > 1 for district in expected):
            multi_output_models += 1

    assert checked == 300
    # Without multi-output mechanisms the proposition is vacuous: every district is a
    # singleton and an implementation emitting no bidirected edges at all would pass.
    assert multi_output_models > 150, multi_output_models


# --- the four structural cases ----------------------------------------------------


def test_a_mechanism_confounds_its_own_outputs() -> None:
    """One mechanism, one noise. Its outputs are dependent given its inputs, and a
    variable-level graph can only say so with a bidirected edge.

    This is the case the hypergraph formalism exists for -- `MINIMAL_EXAMPLE.md` calls the
    exact correlation of two jointly produced outputs "structurally inexpressible in any
    latent-free Pearl SCM". A projection that omits the edge throws away the modelling
    choice the whole library is about.
    """
    graph = MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D")}},
    )
    admg = latent_project_to_variable_admg(graph)

    assert admg.bidirected_edges == (("C", "D"),)
    assert set(admg.directed_edges) == {("A", "C"), ("A", "D")}
    assert _districts(graph) == {("A",), ("C", "D")}


def test_a_latent_mechanism_confounds_its_outputs() -> None:
    """`Mechanism(latent=True)` is the textbook confounder and must not be invisible.

    Nothing about the projection should turn on the `latent` flag here: a mechanism node is
    never observed in data either way, so its noise is latent either way. The flag records
    that the mechanism's *functional form* is unknown, which is a different question.
    """
    graph = MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D"), "latent": True}},
    )

    assert latent_project_to_variable_admg(graph).bidirected_edges == (("C", "D"),)


def test_a_directed_path_through_a_hidden_variable_becomes_a_direct_edge() -> None:
    """`A -> h -> B` with `h` hidden is `A -> B` after projection (Pearl 2009 section 3.7)."""
    graph = MechanismGraph(
        variables={"A", "h", "B"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("h",)},
            "m2": {"inputs": ("h",), "outputs": ("B",)},
        },
        observed_variables={"A", "B"},
    )
    admg = latent_project_to_variable_admg(graph)

    assert admg.nodes == ("A", "B")
    assert admg.directed_edges == (("A", "B"),)
    assert admg.bidirected_edges == ()


def test_a_chain_of_hidden_variables_is_traversed_not_stopped_at() -> None:
    """Reachability through hidden nodes is transitive; one hop is not enough."""
    graph = MechanismGraph(
        variables={"A", "h1", "h2", "B"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("h1",)},
            "m2": {"inputs": ("h1",), "outputs": ("h2",)},
            "m3": {"inputs": ("h2",), "outputs": ("B",)},
        },
        observed_variables={"A", "B"},
    )

    assert latent_project_to_variable_admg(graph).directed_edges == (("A", "B"),)


def test_a_hidden_variable_confounds_the_observed_variables_it_reaches() -> None:
    """The case the existing code already handles, kept as a regression.

    A hidden variable feeding two mechanisms is a common cause of their outputs. This must
    survive the repair, and it must reach *through* further hidden variables, not only one
    hop, which is what distinguishes it from the shape the old implementation matched.
    """
    graph = MechanismGraph(
        variables={"h", "mid", "X", "Y"},
        mechanisms={
            "m_x": {"inputs": ("h",), "outputs": ("X",)},
            "m_mid": {"inputs": ("h",), "outputs": ("mid",)},
            "m_y": {"inputs": ("mid",), "outputs": ("Y",)},
        },
        observed_variables={"X", "Y"},
    )

    assert latent_project_to_variable_admg(graph).bidirected_edges == (("X", "Y"),)


def test_a_latent_source_forks_through_two_hidden_chains() -> None:
    """The case that needs the clique rule and the traversal rule at the same time.

    A latent mechanism produces a hidden `H1`, which reaches `A` and `B` down two separate
    hidden chains. Neither rule finds the confounding alone: the clique is over the
    mechanism's outputs, which are hidden, and the traversal has to run past two more
    hidden nodes before it reaches anything observed.
    """
    graph = MechanismGraph(
        variables={"H1", "H2", "H3", "A", "B"},
        mechanisms={
            "m1": {"inputs": (), "outputs": ("H1",), "latent": True},
            "m2": {"inputs": ("H1",), "outputs": ("H2",)},
            "m3": {"inputs": ("H1",), "outputs": ("H3",)},
            "ma": {"inputs": ("H2",), "outputs": ("A",)},
            "mb": {"inputs": ("H3",), "outputs": ("B",)},
        },
        observed_variables={"A", "B"},
    )

    assert latent_project_to_variable_admg(graph).bidirected_edges == (("A", "B"),)


# --- what must not change ---------------------------------------------------------


def test_the_frontdoor_example_projects_exactly_as_before() -> None:
    """The one hidden-boundary case the library already identified. The repair adds
    edges the old projection missed; it must not perturb one it got right."""
    admg = latent_project_to_variable_admg(frontdoor_hidden_boundary_graph())

    assert admg.nodes == ("X", "Y", "Z")
    assert admg.directed_edges == (("X", "Z"), ("Z", "Y"))
    assert admg.bidirected_edges == (("X", "Y"),)


def test_a_mechanism_with_one_output_still_confounds_nothing() -> None:
    """The rule adds a *clique* over out(m); a singleton clique has no edges.

    Worth pinning: an implementation that added a self-edge, or that confounded a
    mechanism's output with its inputs, would fail here and pass the sweep above only by
    accident.
    """
    graph = MechanismGraph(
        variables={"A", "B", "C"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("C",)},
        },
    )
    admg = latent_project_to_variable_admg(graph)

    assert admg.bidirected_edges == ()
    assert set(admg.directed_edges) == {("A", "B"), ("B", "C")}


def test_a_partly_hidden_output_group_confounds_what_it_reaches() -> None:
    """A mechanism producing one observed and one hidden output confounds the observed one
    with whatever the hidden one reaches -- they share the same noise.

    This is the case that needs both halves of the rule at once, and neither the
    output-clique rule nor the hidden-traversal rule finds it alone.
    """
    graph = MechanismGraph(
        variables={"A", "C", "h", "B"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C", "h")},
            "m2": {"inputs": ("h",), "outputs": ("B",)},
        },
        observed_variables={"A", "B", "C"},
    )

    assert latent_project_to_variable_admg(graph).bidirected_edges == (("B", "C"),)


def test_the_projection_is_a_valid_admg_on_generated_models() -> None:
    """Acyclicity of the directed part is checked by `ADMG.__init__`; a projection that
    invented an edge could make it cyclic, and that must surface as a failure here rather
    than deep inside an identification algorithm."""
    from tests.conformance.generation import generate_model

    for seed in range(100):
        graph = generate_model(seed).graph()
        admg = latent_project_to_variable_admg(graph)
        assert admg.node_set == graph.observed_set
        admg.topological_order()  # raises if the projection introduced a cycle


def test_no_edge_of_the_projection_mentions_a_hidden_variable() -> None:
    """A projection exists to *remove* the hidden variables, so none may survive in an edge.

    `ADMG.__init__` would reject an edge naming an unknown node, so a leak surfaces as a
    constructor error rather than as a wrong graph -- but only if a hidden variable ever
    reaches an edge, which the sweep below checks directly on models that have some.
    """
    from tests.conformance.generation import generate_model

    with_hidden = 0
    for seed in range(150):
        graph = generate_model(seed).graph()
        hidden = graph.variable_set - graph.observed_set
        if not hidden:
            continue
        with_hidden += 1
        admg = latent_project_to_variable_admg(graph)
        mentioned = {name for edge in admg.directed_edges for name in edge}
        mentioned |= {name for edge in admg.bidirected_edges for name in edge}
        assert not (mentioned & hidden), f"seed {seed}: {sorted(mentioned & hidden)}"

    assert with_hidden > 30, with_hidden


def test_the_projection_no_longer_produces_a_confident_wrong_answer() -> None:
    """The end-to-end consequence, kept as a regression.

    Before the repair this graph identified, via T7, an estimand containing
    `P(C | X) * P(D | X)` -- two factors where the truth has one, `P(C,D | X)`, because
    `C` and `D` are jointly produced by a single mechanism and the projection recorded no
    confounding between them. It was `Identified`, it was wrong, and nothing said so.

    With the districts correct the stub backend can no longer pretend to handle it, so the
    answer becomes an honest refusal. Turning a confident wrong answer into a refusal is
    the improvement; identifying it correctly is what a real ID backend is for.
    """
    graph = MechanismGraph(
        variables={"A", "X", "C", "D", "Y", "H"},
        observed_variables={"A", "X", "C", "D", "Y"},
        mechanisms={
            "m_h": {"inputs": (), "outputs": ("H",), "latent": True},
            "m_x": {"inputs": ("A", "H"), "outputs": ("X",)},
            "m_cd": {"inputs": ("X",), "outputs": ("C", "D")},
            "m_y": {"inputs": ("C", "D"), "outputs": ("Y",)},
        },
    )
    admg = latent_project_to_variable_admg(graph)

    assert ("C", "D") in admg.districts()
    assert admg.bidirected_edges == (("C", "D"),)

    from causal_hypergraphs import DeleteMechanism, Identified, identify

    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)
    if isinstance(result, Identified):
        assert "P(C | X) * P(D | X)" not in str(result.expression), result.expression
