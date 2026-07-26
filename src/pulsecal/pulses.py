"""Pulse envelopes: lifted Gaussians and their DRAG quadrature.

The in-phase envelope is the *lifted* Gaussian used by every superconducting
control stack, i.e. a Gaussian shifted and rescaled so that it starts and
ends exactly at zero (a hard edge would radiate broadband and re-excite the
|1> -> |2> transition):

    g(t) = A * ( exp(-(t-t0)^2 / 2s^2) - c ) / (1 - c),   c = exp(-t0^2 / 2s^2).

DRAG [Motzoi2009, Gambetta2011] adds a quadrature proportional to the
derivative of that envelope,

    eps(t) = g(t) + i * beta * dg/dt,

whose leading-order effect is to cancel the leakage amplitude accumulated on
|2>.  ``beta`` is kept in physical units of ns so that it can be compared
directly with the analytic prediction beta = -1/alpha.

The same waveforms are emitted as a ``qiskit.pulse`` schedule by
:func:`to_schedule` [Alexander2020]; :func:`qiskit_beta` documents the exact
unit conversion to Qiskit's ``Drag`` parametrisation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PulseSpec:
    """A single-qubit DRAG pulse on the drive line.

    ``amp`` and ``beta`` are the two calibration knobs; ``phase`` selects the
    rotation axis in the equatorial plane (a virtual-Z frame update
    [McKay2017]); ``detuning`` is the drive offset from nu_q in GHz.
    """

    duration: int
    sigma: float
    amp: float = 0.0
    beta: float = 0.0
    phase: float = 0.0
    detuning: float = 0.0

    def with_(self, **kwargs) -> "PulseSpec":
        """Copy with fields overridden (keeps calibration code declarative)."""
        return replace(self, **kwargs)


def _lift_factor(duration: int, sigma: float) -> float:
    """Gaussian value at the lifting reference point t = -1 (Qiskit convention)."""
    return float(np.exp(-0.5 * ((1.0 + 0.5 * duration) / sigma) ** 2))


def lifted_gaussian(duration: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit-amplitude lifted Gaussian and its derivative d/dn (per sample).

    Samples sit at t = n + 1/2 in units of dt and the envelope is lifted so
    that it vanishes one sample outside the pulse, matching
    ``qiskit.pulse.library.Gaussian`` exactly.  Returning the derivative from
    the same place keeps the DRAG quadrature consistent with the in-phase
    envelope, which matters because the leakage cancellation is a statement
    about the *actual* derivative of the applied waveform.
    """
    t = np.arange(duration) + 0.5
    t0 = 0.5 * duration
    raw = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    scale = 1.0 / (1.0 - _lift_factor(duration, sigma))
    return (raw - _lift_factor(duration, sigma)) * scale, -((t - t0) / sigma**2) * raw * scale


def drag_envelope(spec: PulseSpec, dt: float) -> np.ndarray:
    """Complex baseband envelope eps(t) = (g + i*beta*dg/dt) * e^{i*phase}."""
    g, dg = lifted_gaussian(spec.duration, spec.sigma)
    return spec.amp * (g + 1j * spec.beta * dg / dt) * np.exp(1j * spec.phase)


def area(spec: PulseSpec, dt: float) -> float:
    """Integral of the in-phase envelope in ns; the rotation angle is r*area."""
    g, _ = lifted_gaussian(spec.duration, spec.sigma)
    return float(spec.amp * g.sum() * dt)


def qiskit_beta(spec: PulseSpec, dt: float) -> float:
    """Convert beta [ns] to the dimensionless beta of ``qiskit.pulse.Drag``.

    Qiskit defines Drag as g_lifted(n) * (1 - i*beta*(n-n0)/sigma^2), i.e. it
    applies the derivative *factor* to the lifted envelope instead of
    differentiating it.  Matching the coefficient of -(n-n0)/sigma^2 gives
    beta_qiskit = beta / dt; the two waveforms then agree up to the lifting
    offset, a sub-percent difference in the quadrature for the sigma/duration
    ratio used here (quantified in ``tests/test_pulses.py``).
    """
    return spec.beta / dt


def to_schedule(spec: PulseSpec, dt: float, parametric: bool = False, channel: int = 0):
    """Emit the pulse as a ``qiskit.pulse`` schedule [Alexander2020].

    ``parametric=True`` uses the backend-native ``Drag`` template that real
    control electronics accept; the default plays the exact sampled waveform
    that the simulator integrates, so schedule and simulation never drift
    apart.  Qiskit is imported lazily to keep the numerical core dependency
    free.
    """
    from qiskit import pulse

    if parametric:
        shape = pulse.Drag(spec.duration, spec.amp, spec.sigma,
                           qiskit_beta(spec, dt), angle=spec.phase, limit_amplitude=False)
    else:
        shape = pulse.Waveform(drag_envelope(spec, dt), limit_amplitude=False)
    with pulse.build(name=f"drag_a{spec.amp:.4f}_b{spec.beta:.4f}") as sched:
        pulse.play(shape, pulse.DriveChannel(channel))
    return sched
