from causal_hypergraphs import ADMG, Identified, PearlHedgeWitness, Unidentified, identify_effect


def test_pearl_backend_observational_marginal() -> None:
    graph = ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"})

    assert isinstance(result, Identified)
    assert result.theorem == "Pearl-ID observational marginal"
    assert str(result.expression) == "sum_{X} P(X,Y)"


def test_pearl_backend_markovian_truncated_factorization() -> None:
    graph = ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Identified)
    assert result.theorem == "Pearl-ID Markovian truncated factorization"
    assert str(result.expression) == "P(Y | X)"


def test_pearl_backend_frontdoor_identifies_confounding_case() -> None:
    graph = ADMG(
        nodes={"X", "Z", "Y"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Identified)
    assert result.theorem == "Pearl-ID front-door"
    assert str(result.expression) == "sum_{Z} P(Z | X) * sum_{X} P(X) * P(Y | X,Z)"
    assert result.expression.scope() == frozenset({"X", "Y"})
    assert result.expression.conditioned_on() == frozenset({"X"})
    assert len(result.expression.kernels()) == 3


def test_pearl_backend_refuses_unsupported_hedge_like_case() -> None:
    graph = ADMG(nodes={"X", "Y"}, bidirected_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Unidentified)
    assert isinstance(result.witness, PearlHedgeWitness)
    assert result.witness.districts == (("X", "Y"),)
    assert "does not identify" in result.reason
