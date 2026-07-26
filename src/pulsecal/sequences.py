"""Gate sequences built by concatenating pulse envelopes.

Back-to-back gates on one drive line are just one longer waveform, so a
sequence is represented as a list of :class:`~pulsecal.pulses.PulseSpec` and
simulated by concatenating their envelopes.  Axis changes are phase offsets on
the carrier rather than extra pulses, which is exactly how virtual-Z gates work
on hardware [McKay2017]; the identity is a zero-amplitude pulse of the same
duration, matching what a real AllXY sequence plays.
"""

from __future__ import annotations

import numpy as np

from .device import Transmon
from .propagate import gate_propagator
from .pulses import PulseSpec, drag_envelope

#: Axis label -> (fraction of a pi rotation, carrier phase).
AXES = {
    "I": (0.0, 0.0),
    "X": (1.0, 0.0),
    "Y": (1.0, -np.pi / 2),
    "x": (0.5, 0.0),
    "y": (0.5, -np.pi / 2),
    "-x": (-0.5, 0.0),
    "-y": (-0.5, -np.pi / 2),
}

#: The 21 AllXY pairs; the ideal excited-state populations are 0 for the first
#: five, 1/2 for the next twelve and 1 for the last four.  Deviations from that
#: staircase fingerprint amplitude, detuning and DRAG errors separately
#: [Reed2013, Krantz2019].
ALLXY = [
    ("I", "I"), ("X", "X"), ("Y", "Y"), ("X", "Y"), ("Y", "X"),
    ("x", "I"), ("y", "I"), ("x", "y"), ("y", "x"), ("x", "Y"), ("y", "X"),
    ("X", "y"), ("Y", "x"), ("x", "X"), ("X", "x"), ("y", "Y"), ("Y", "y"),
    ("X", "I"), ("Y", "I"), ("x", "x"), ("y", "y"),
]

ALLXY_IDEAL = np.array([0.0] * 5 + [0.5] * 12 + [1.0] * 4)


def gate(pi_pulse: PulseSpec, label: str, half: PulseSpec | None = None) -> PulseSpec:
    """Phase a calibrated pulse into the gate named ``label``.

    A negative fraction is realised as a positive-amplitude pulse with the
    carrier phase advanced by pi, since AWG amplitudes are non-negative.
    Passing ``half`` supplies an independently calibrated pi/2 pulse; without
    it the pi amplitude is simply halved, which is convenient but leaves a
    percent-level error because the rotation angle is not exactly linear in
    amplitude.  Backends store the two calibrations separately for this reason.
    """
    frac, phase = AXES[label]
    base = pi_pulse if abs(frac) == 1.0 else (half or pi_pulse.with_(amp=0.5 * pi_pulse.amp))
    return base.with_(amp=(0.0 if frac == 0 else base.amp),
                      phase=phase + (np.pi if frac < 0 else 0.0))


def envelope(specs: list[PulseSpec], dt: float, detuning: float = 0.0) -> np.ndarray:
    """Concatenated complex envelope of a gate sequence.

    When the drive sits at nu_q + detuning the frame it defines precesses
    against the qubit, so the axis a later pulse rotates about drifts by
    2*pi*detuning*t unless the control software puts it back.  Advancing each
    carrier phase by that amount is precisely the virtual-Z bookkeeping a real
    stack performs [McKay2017]; leaving it out makes any detuned calibration
    point look far worse than it is as soon as a sequence has two gates in it.
    """
    out, t = [], 0.0
    for s in specs:
        out.append(drag_envelope(s.with_(phase=s.phase + 2 * np.pi * detuning * t), dt))
        t += s.duration * dt
    return np.concatenate(out)


def run(dev: Transmon, specs: list[PulseSpec], detuning: float = 0.0) -> np.ndarray:
    """Full propagator of a gate sequence."""
    return gate_propagator(dev, envelope(specs, dev.dt, detuning), detuning)


def allxy_specs(pi_pulse: PulseSpec, half: PulseSpec | None = None) -> list[list[PulseSpec]]:
    """The 21 AllXY sequences expressed as pulse lists."""
    return [[gate(pi_pulse, a, half), gate(pi_pulse, b, half)] for a, b in ALLXY]
