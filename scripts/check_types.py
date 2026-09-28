"""Run Pyright using this interpreter's installed dependencies on every platform."""
from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "pyright", "--pythonpath", sys.executable])
    )
