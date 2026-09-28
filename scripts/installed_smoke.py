"""Run against an installed wheel from outside the checkout, with no PYTHONPATH."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = """
import importlib
import importlib.metadata
import pkgutil
import sys
from pathlib import Path

import causal_hypergraphs as ch
assert "site-packages" in ch.__file__.replace("\\\\", "/"), ch.__file__
assert "numpy" not in sys.modules, "Core import must remain lightweight"
for module in pkgutil.walk_packages(ch.__path__, ch.__name__ + "."):
    importlib.import_module(module.name)
assert "numpy" not in sys.modules, "Module discovery must not import optional dependencies"
assert Path(ch.__file__).with_name("py.typed").exists()

graph = ch.MechanismGraph({"X", "Y"}, {"m": {"inputs": {"X"}, "outputs": {"Y"}}})
query = ch.CausalQuery(("Y",), ch.HardIntervention({"X": 1}))
compiled = ch.compile_query(graph, query)
restored = ch.loads(ch.dumps(compiled))
data = ch.Dataset.from_records([{"X": 0, "Y": 0}, {"X": 1, "Y": 1}])
assert ch.estimate_query(restored, data).values[(1,)] == 1
print("installed smoke passed", importlib.metadata.version("causal-hypergraphs"))
"""


def main() -> None:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run([sys.executable, "-c", CODE], cwd=directory, env=environment, check=True)
        root = Path(__file__).resolve().parents[1]
        for name in ("software_pipeline", "manufacturing", "policy"):
            subprocess.run(
                [sys.executable, str(root / "examples" / f"{name}.py")],
                cwd=directory,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
    print("three installed domain examples passed")


if __name__ == "__main__":
    main()
