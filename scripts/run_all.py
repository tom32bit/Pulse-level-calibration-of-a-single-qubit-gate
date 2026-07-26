"""Run the whole study in order and report timings.

The scripts share state through ``results/*.json``: 01 fixes the pi amplitude,
03 the DRAG coefficient, 04 the fine amplitude, and 05-07 consume all three.
Running them in this order from a clean checkout reproduces every figure.
"""

import _bootstrap  # noqa: F401

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = ["00_validate.py", "01_rabi.py", "02_drag_mechanism.py", "03_drag_calibration.py",
           "04_error_amplification.py", "05_allxy.py", "06_grape.py", "07_speed_limit.py"]

total = 0.0
for name in SCRIPTS:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    start = time.perf_counter()
    done = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE)
    elapsed = time.perf_counter() - start
    total += elapsed
    if done.returncode:
        sys.exit(f"\n{name} failed with exit code {done.returncode}")
    print(f"-- {name} finished in {elapsed:.1f} s")

print(f"\n{'=' * 72}\nall scripts completed in {total:.1f} s")
