"""Gradient ascent pulse engineering for a target single-qubit unitary.

GRAPE [Khaneja2005] treats the two quadrature waveforms as 2K free numbers and
ascends the gate fidelity directly.  Two departures from the 1985-style recipe
matter in practice and are both taken here:

* the derivatives dU_k/du are computed exactly in the instantaneous eigenbasis
  rather than to first order in dt, which removes the step-size ceiling that
  otherwise stalls quasi-Newton search [Machnes2011];
* the search itself is L-BFGS-B rather than fixed-step ascent, which is where
  most of the speed-up over the original algorithm comes from
  [deFouquieres2011].

Two terms make the result something an AWG could actually play: box bounds on
each quadrature, and a discrete-Laplacian roughness penalty evaluated with
virtual zeros outside the pulse, which simultaneously smooths the waveform and
forces it to start and end at zero.  Optimising the *leaky* three- or
four-level system rather than a qubit is what lets the optimiser trade a
little pulse area for a spectrum that avoids the |1> -> |2> transition, the
mechanism behind measured leakage-limited fast gates [Werninghaus2021].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .device import Transmon
from .metrics import average_gate_fidelity, fidelity_gradient, gate_error, leakage
from .propagate import control_gradient, propagator


@dataclass
class GrapeResult:
    """Outcome of one optimisation run."""

    envelope: np.ndarray          # complex, one sample per AWG point
    error: float                  # 1 - F_avg against the target
    leakage: float                # L1 of the optimised gate
    history: np.ndarray           # gate error after each accepted iterate
    n_iter: int
    success: bool


def _roughness(u: np.ndarray) -> tuple[float, np.ndarray]:
    """Sum of squared first differences with zeros padded outside the pulse.

    Padding is what ties the penalty to the boundary condition: the ends of the
    waveform are pulled to zero by the same term that suppresses sample-to-
    sample jitter, so no separate constraint is needed.
    """
    padded = np.concatenate([[0.0], u, [0.0]])
    d = np.diff(padded)
    return float(d @ d), 2.0 * (d[:-1] - d[1:])


def _envelope(x: np.ndarray) -> np.ndarray:
    """Flat control vector ``[I..., Q...]`` read as one complex envelope."""
    i_q = x.reshape(2, -1)
    return i_q[0] + 1j * i_q[1]


def _objective(x: np.ndarray, dev: Transmon, target: np.ndarray, weight: float):
    """J = 1 - F_avg + weight * roughness, with its exact gradient."""
    i_q = x.reshape(2, -1)
    env = _envelope(x)
    u = propagator(dev, env)
    grad_i, grad_q = control_gradient(dev, env, -fidelity_gradient(u, target))

    rough = [_roughness(q) for q in i_q]
    cost = 1.0 - average_gate_fidelity(u, target) + weight * sum(r[0] for r in rough)
    grad = np.concatenate([grad_i + weight * rough[0][1], grad_q + weight * rough[1][1]])
    return cost, grad


def _seed(rng: np.random.Generator, n: int, max_amp: float) -> np.ndarray:
    """Smooth random start: a few low-order Fourier modes under a sine window.

    Random *samples* would start the optimiser deep in the roughness penalty
    and waste iterations flattening noise, so the seed is band-limited from the
    outset.
    """
    k = np.arange(1, 5)[:, None]
    t = (np.arange(n) + 0.5) / n
    modes = np.sin(np.pi * k * t)
    win = np.sin(np.pi * t)
    return 0.4 * max_amp * (rng.normal(size=(2, 4)) @ modes / np.sqrt(4)) * win


def optimize(dev: Transmon, target: np.ndarray, n_slices: int, max_amp: float = 1.0,
             smoothness: float = 2e-3, maxiter: int = 400, n_restarts: int = 6,
             seed: int = 0, warm_start: np.ndarray | None = None) -> GrapeResult:
    """Optimise both quadratures for ``target``; keep the best of several starts.

    ``warm_start`` (typically the calibrated DRAG envelope) is used as one of
    the starting points, so the optimiser can only improve on analytic control.
    """
    rng = np.random.default_rng(seed)
    starts = [] if warm_start is None else [np.stack([warm_start.real, warm_start.imag])]
    starts += [_seed(rng, n_slices, max_amp) for _ in range(n_restarts)]

    def error_of(x: np.ndarray) -> float:
        return gate_error(propagator(dev, _envelope(x)), target)

    best = None
    for x0 in starts:
        history: list[float] = []
        res = minimize(
            _objective, np.clip(x0, -max_amp, max_amp).ravel(),
            args=(dev, target, smoothness), jac=True, method="L-BFGS-B",
            bounds=[(-max_amp, max_amp)] * (2 * n_slices),
            options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14},
            callback=lambda xk: history.append(error_of(xk)),
        )
        candidate = GrapeResult(_envelope(res.x), error_of(res.x),
                                leakage(propagator(dev, _envelope(res.x))),
                                np.array(history), res.nit, bool(res.success))
        if best is None or candidate.error < best.error:
            best = candidate
    return best
