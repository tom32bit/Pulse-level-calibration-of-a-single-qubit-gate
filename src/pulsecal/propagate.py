"""Piecewise-constant time evolution and its exact control derivatives.

An arbitrary-waveform generator emits a staircase, so the exact propagator of
a sampled pulse is a product of matrix exponentials, one per sample:

    U = U_{K-1} ... U_0,     U_k = exp(-i H_k dt).

Every H_k is Hermitian and tiny (3x3), so all propagators are built at once
from a batched eigendecomposition.  The same eigenbasis yields *exact*
derivatives dU_k/du rather than the first-order approximation of the original
GRAPE paper [Khaneja2005]; using them is what lets a quasi-Newton optimiser
converge at its full rate [Machnes2011, deFouquieres2011].
"""

from __future__ import annotations

import numpy as np

from .device import Transmon


def step_hamiltonians(dev: Transmon, envelope: np.ndarray, detuning=0.0) -> np.ndarray:
    """Per-sample Hamiltonians H_k, shape (..., K, d, d).

    Leading batch axes of ``envelope`` (and matching axes of ``detuning``)
    broadcast, so an entire two-dimensional calibration map is one call.
    """
    h_i, h_q = dev.h_drive
    return (dev.h_static(detuning)[..., None, :, :]
            + envelope.real[..., None, None] * h_i
            + envelope.imag[..., None, None] * h_q)


def step_propagators(h: np.ndarray, dt: float) -> np.ndarray:
    """exp(-i H_k dt) for a stack of Hermitian H_k, via batched eigh."""
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * w * dt)[..., None, :]) @ v.conj().swapaxes(-1, -2)


def compose(us: np.ndarray) -> np.ndarray:
    """Time-ordered product U_{K-1} ... U_0 over the third-from-last axis."""
    total = np.eye(us.shape[-1], dtype=complex)
    for k in range(us.shape[-3]):
        total = us[..., k, :, :] @ total
    return total


def partial_products(us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward partial products used by the gradient.

    ``fwd[k] = U_{k-1}...U_0`` (so ``fwd[0] = I``) and
    ``bwd[k] = U_{K-1}...U_k`` (so ``bwd[K] = I``); both have K+1 entries and
    ``fwd[K] == bwd[0]`` is the full propagator.  ``fwd`` doubles as the state
    trajectory generator.
    """
    k, d = us.shape[0], us.shape[-1]
    fwd = np.empty((k + 1, d, d), dtype=complex)
    bwd = np.empty((k + 1, d, d), dtype=complex)
    fwd[0] = bwd[k] = np.eye(d, dtype=complex)
    for i in range(k):
        fwd[i + 1] = us[i] @ fwd[i]
        bwd[k - 1 - i] = bwd[k - i] @ us[k - 1 - i]
    return fwd, bwd


def propagator(dev: Transmon, envelope: np.ndarray, detuning=0.0) -> np.ndarray:
    """Full propagator(s) of a sampled pulse, in the frame of the drive."""
    return compose(step_propagators(step_hamiltonians(dev, envelope, detuning), dev.dt))


def gate_propagator(dev: Transmon, envelope: np.ndarray, detuning=0.0) -> np.ndarray:
    """Propagator in the frame of the *qubit* rather than the drive.

    A drive parked at nu_q + delta defines a frame that precesses against the
    qubit, so its propagator picks up exp(-i 2 pi delta j T) on level j purely
    as bookkeeping.  Control software removes that by advancing the phase of
    later pulses [McKay2017], so it must be removed before comparing a gate
    with its target -- otherwise the optimiser is charged for a frame choice
    and will spend real pulse parameters cancelling it.
    """
    j = np.arange(dev.n_levels)
    t = envelope.shape[-1] * dev.dt
    phase = np.exp(-2j * np.pi * np.asarray(detuning, dtype=float)[..., None] * j * t)
    return phase[..., :, None] * propagator(dev, envelope, detuning)


def trajectory(dev: Transmon, envelope: np.ndarray, psi0: np.ndarray,
               detuning: float = 0.0) -> np.ndarray:
    """State after every sample, shape (K+1, n_levels), starting from psi0."""
    us = step_propagators(step_hamiltonians(dev, envelope, detuning), dev.dt)
    fwd, _ = partial_products(us)
    return fwd @ psi0


def _deriv_kernel(w: np.ndarray, dt: float) -> np.ndarray:
    """Frechet kernel F_mn = (e^{-i w_m dt} - e^{-i w_n dt}) / (-i dt (w_m - w_n)).

    The limit m -> n is exp(-i w_m dt); the expression is regularised rather
    than branched so it stays vectorised over the sample axis.
    """
    ph = np.exp(-1j * w * dt)
    gap = w[..., :, None] - w[..., None, :]
    small = np.abs(gap) < 1e-10
    safe = np.where(small, 1.0, gap)
    kern = (ph[..., :, None] - ph[..., None, :]) / (-1j * dt * safe)
    return np.where(small, ph[..., :, None] * np.ones_like(kern), kern)


def control_gradient(dev: Transmon, envelope: np.ndarray, cost_grad: np.ndarray,
                     detuning: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact d(cost)/d(eps_I[k]) and d(cost)/d(eps_Q[k]).

    ``cost_grad`` is dJ/dU* for a real scalar J, so that dJ = 2 Re Tr[G' dU].
    Returns one array per quadrature, each of length K.
    """
    h = step_hamiltonians(dev, envelope, detuning)
    w, v = np.linalg.eigh(h)
    us = (v * np.exp(-1j * w * dev.dt)[..., None, :]) @ v.conj().swapaxes(-1, -2)
    fwd, bwd = partial_products(us)

    # M_k = fwd[k] G' bwd[k+1] makes dJ/du_k = 2 Re Tr[M_k dU_k/du_k], and
    # dU_k = V (kern * V' H_ctrl V) V' with kern the Frechet kernel above.
    vt = v.conj().swapaxes(-1, -2)
    b = vt @ (fwd[:-1] @ cost_grad.conj().T @ bwd[1:]) @ v
    kern = -1j * dev.dt * _deriv_kernel(w, dev.dt)

    def quad(h_ctrl: np.ndarray) -> np.ndarray:
        inner = kern * (vt @ h_ctrl @ v)
        return 2.0 * np.real(np.einsum("kij,kji->k", b, inner))

    h_i, h_q = dev.h_drive
    return quad(h_i), quad(h_q)
