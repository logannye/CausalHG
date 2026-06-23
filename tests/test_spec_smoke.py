from pathlib import Path


def test_compiler_spec_documents_core_semantics() -> None:
    spec = Path("SPEC.md").read_text()

    for required in [
        "DeleteMechanism",
        "ReplaceMechanism",
        "T2",
        "T3",
        "T4",
        "T6",
        "T7",
        "Unknown",
        "Unidentified",
        "P0(v)",
    ]:
        assert required in spec
