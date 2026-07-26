"""Put ``src`` on the import path so the scripts run from a clean checkout.

Importing this module is equivalent to ``pip install -e .``; the editable
install is still the tidier option and is what ``pyproject.toml`` describes.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Qiskit 0.46 emits deprecation notices for the pulse module on import; they are
# expected and would otherwise bury the progress lines the scripts print.
warnings.filterwarnings("ignore", category=DeprecationWarning)
