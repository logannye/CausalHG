from __future__ import annotations

from .graph import Mechanism, MechanismGraph


def reaction_graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m1": Mechanism(
                "m1",
                inputs=("A", "B"),
                outputs=("C", "D"),
                output_equalities=(("C", "D"),),
            ),
            "m2": Mechanism("m2", inputs=("C", "E"), outputs=("F",)),
        },
    )


def latent_mechanism_graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m_lat": Mechanism("m_lat", outputs=("B", "E"), latent=True),
            "m1": Mechanism(
                "m1",
                inputs=("A", "B"),
                outputs=("C", "D"),
                output_equalities=(("C", "D"),),
            ),
            "m2": Mechanism("m2", inputs=("C", "E"), outputs=("F",)),
        },
    )


def hidden_variable_graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F", "W"},
        observed_variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m_W": Mechanism("m_W", outputs=("W",), latent=True),
            "m1": Mechanism(
                "m1",
                inputs=("A", "B"),
                outputs=("C", "D"),
                output_equalities=(("C", "D"),),
            ),
            "m_2": Mechanism("m_2", inputs=("C", "E", "W"), outputs=("F",)),
        },
    )


def frontdoor_hidden_boundary_graph() -> MechanismGraph:
    """Hidden-boundary graph whose T7 variable projection is X -> Z -> Y, X <-> Y."""

    return MechanismGraph(
        variables={"W", "X", "Y", "Z"},
        observed_variables={"X", "Y", "Z"},
        mechanisms={
            "m_W": Mechanism("m_W", outputs=("W",), latent=True),
            "m_x": Mechanism("m_x", inputs=("W",), outputs=("X",)),
            "m_z": Mechanism("m_z", inputs=("X",), outputs=("Z",)),
            "m_y": Mechanism("m_y", inputs=("W", "Z"), outputs=("Y",)),
        },
    )
