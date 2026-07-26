"""Pulse-level calibration of a single-qubit gate on a weakly anharmonic transmon.

Layered so each piece can be read on its own:

``device``      Duffing transmon and its RWA Hamiltonian
``pulses``      lifted-Gaussian and DRAG envelopes, ``qiskit.pulse`` export
``propagate``   piecewise-constant propagators and exact control derivatives
``metrics``     leakage-aware average gate fidelity
``sequences``   gate sequences as concatenated envelopes, AllXY table
``experiments`` amplitude, chevron, DRAG and error-amplification routines
``grape``       gradient-ascent pulse engineering for a target unitary
``dynamics``    independent Qiskit Dynamics cross-check of the propagator
``plotting``    shared figure style
"""

from .device import Transmon
from .pulses import PulseSpec

__all__ = ["Transmon", "PulseSpec"]
__version__ = "1.0.0"
