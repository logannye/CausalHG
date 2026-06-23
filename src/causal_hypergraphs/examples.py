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
