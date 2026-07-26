"""Figures of merit for a gate acting on a leaky qubit.

The propagator U is (n_levels x n_levels); the gate that the algorithm sees is
its computational block V = P U P^dagger, which is *sub-unitary* whenever
population escapes to |2>.  For such a trace-non-increasing map the average
gate fidelity generalises to [Nielsen2002, Wood2018]

    F_avg = ( Tr(M'M) + |Tr M|^2 ) / ( d (d+1) ),     M = W' V,  d = 2,

which reduces to the familiar expression when V is unitary and automatically
charges leakage as an error.  The average leakage out of the computational
subspace is L1 = 1 - Tr(V'V)/d.

Because a Z rotation costs nothing on hardware (it is a frame relabelling of
subsequent pulses [McKay2017]), ``z_free=True`` maximises the fidelity over a
virtual-Z, isolating the errors that pulse shaping actually has to fix.
"""

from __future__ import annotations

import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """exp(-i * angle/2 * axis) for a Pauli ``axis``."""
    return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * axis


TARGETS = {
    "X": rotation(SX, np.pi),
    "X90": rotation(SX, np.pi / 2),
    "Y": rotation(SY, np.pi),
    "Y90": rotation(SY, np.pi / 2),
    "H": (SX + SZ) / np.sqrt(2) * -1j,
}


def block(u: np.ndarray, d: int = 2) -> np.ndarray:
    """Computational-subspace block V of a full propagator."""
    return u[:d, :d]


def leakage(u: np.ndarray, d: int = 2) -> float:
    """Average leakage L1 = 1 - Tr(V'V)/d out of the computational subspace."""
    v = block(u, d)
    return float(1.0 - np.real(np.trace(v.conj().T @ v)) / d)


def average_gate_fidelity(u: np.ndarray, target: np.ndarray, z_free: bool = False) -> float:
    """Average gate fidelity of the computational block against ``target``.

    With ``z_free`` the fidelity is maximised analytically over a virtual-Z:
    Tr(W' Rz(phi) V) = e^{-i phi/2} B00 + e^{i phi/2} B11 with B = V W', whose
    modulus peaks at |B00| + |B11|, while Tr(M'M) = Tr(V'V) is phi-independent.
    """
    d = target.shape[0]
    v = block(u, d)
    b = v @ target.conj().T
    tr = np.sum(np.abs(np.diag(b))) if z_free else abs(np.trace(b))
    return float((np.real(np.trace(v.conj().T @ v)) + tr**2) / (d * (d + 1)))


def gate_error(u: np.ndarray, target: np.ndarray, z_free: bool = False) -> float:
    return 1.0 - average_gate_fidelity(u, target, z_free)


def fidelity_gradient(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    """dF_avg/dU* embedded in the full space, for use by :mod:`.propagate`.

    With F = (Tr(V'V) + |Tr M|^2)/(d(d+1)) and M = W'V, the Wirtinger
    derivative is (V + Tr(M) W) / (d(d+1)) on the computational block.
    """
    d = target.shape[0]
    v = block(u, d)
    g = np.zeros_like(u)
    g[:d, :d] = (v + np.trace(target.conj().T @ v) * target) / (d * (d + 1))
    return g


def populations(state: np.ndarray) -> np.ndarray:
    """Level populations |<j|psi>|^2 along the last axis."""
    return np.abs(state) ** 2


def bloch(state: np.ndarray) -> np.ndarray:
    """Bloch coordinates of the {|0>,|1>} block, *without* renormalising.

    Leaving the vector sub-normalised is deliberate: the trajectory then
    visibly contracts inside the sphere exactly when population is parked on
    |2>, which is the effect DRAG is designed to undo.
    """
    c0, c1 = state[..., 0], state[..., 1]
    return np.stack([2 * np.real(c0 * c1.conj()),
                     -2 * np.imag(c0 * c1.conj()),
                     np.abs(c0) ** 2 - np.abs(c1) ** 2], axis=-1)
