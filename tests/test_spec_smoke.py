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
        "P0^m(out(m))",
    ]:
        assert required in spec

    # `P0` is a joint policy per mechanism. The superseded per-variable product form is a
    # different operator -- it forces `out(m)` independent -- so leaving it in the spec
    # would document behaviour the compiler no longer has.
    assert "product_{v in out(m)} P0(v)" not in spec
