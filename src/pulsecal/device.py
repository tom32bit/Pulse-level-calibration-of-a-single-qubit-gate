"""Duffing (weakly anharmonic) transmon model.

A transmon is a Cooper-pair box in the E_J >> E_C regime; its low-lying
spectrum is that of an anharmonic oscillator with a small negative
anharmonicity alpha ~ -E_C [Koch2007, Krantz2019].  Truncating to
``n_levels`` and keeping a single microwave drive line,

    H(t)/hbar = w_q a'a + (alpha/2) a'a'aa + r Re[eps(t) e^{-i w_d t}] (a + a')

with a' = a^dagger.  Moving to the frame rotating at the drive frequency
w_d and dropping counter-rotating terms (RWA) gives the model used in every
experiment here:

    H(t)/hbar = sum_j [ j*Delta + (alpha/2) j(j-1) ] |j><j|
                + (r/2) ( eps(t) a' + eps*(t) a ),        Delta = w_q - w_d.

The RWA form is what makes calibration intuitive: a constant real envelope
eps drives x-rotations at rate r*eps, the quadrature Im[eps] drives
y-rotations, and Delta tilts the rotation axis out of the equatorial plane.

Conventions
-----------
Angular frequencies are rad/ns, ordinary frequencies GHz, times ns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Transmon:
    """Static description of the driven transmon.

    Attributes
    ----------
    qubit_frequency, anharmonicity, drive_strength
        nu_q, alpha and r in GHz.
    n_levels
        Truncation dimension; 3 is the minimum that exposes leakage.
    dt
        Arbitrary-waveform-generator sample period in ns.
    """

    qubit_frequency: float
    anharmonicity: float
    drive_strength: float
    n_levels: int = 3
    dt: float = 1.0 / 4.5

    # --- angular-frequency shorthands (rad/ns) -------------------------------
    @property
    def w_q(self) -> float:
        return TWO_PI * self.qubit_frequency

    @property
    def alpha(self) -> float:
        return TWO_PI * self.anharmonicity

    @property
    def r(self) -> float:
        return TWO_PI * self.drive_strength

    # --- ladder operators ----------------------------------------------------
    @property
    def a(self) -> np.ndarray:
        """Annihilation operator truncated to ``n_levels``."""
        return np.diag(np.sqrt(np.arange(1, self.n_levels)), 1).astype(complex)

    @property
    def n_op(self) -> np.ndarray:
        return np.diag(np.arange(self.n_levels)).astype(complex)

    # --- Hamiltonian pieces (RWA frame) --------------------------------------
    def h_static(self, detuning: float | np.ndarray = 0.0) -> np.ndarray:
        """Drift term for a drive offset by ``detuning`` GHz from nu_q.

        ``detuning`` is the *drive* offset delta = nu_d - nu_q, so the frame
        detuning entering the Hamiltonian is Delta = -2*pi*delta.  Array-valued
        detunings broadcast to a stack of (n_levels, n_levels) drifts, which is
        what makes whole chevron maps a single vectorised call.
        """
        j = np.arange(self.n_levels)
        diag = -TWO_PI * np.asarray(detuning, dtype=float)[..., None] * j \
            + 0.5 * self.alpha * j * (j - 1)
        return (np.eye(self.n_levels) * diag[..., None, :]).astype(complex)

    @property
    def h_drive(self) -> tuple[np.ndarray, np.ndarray]:
        """(H_I, H_Q) such that H_d = eps_I H_I + eps_Q H_Q.

        From (r/2)(eps a' + eps* a) with eps = eps_I + i eps_Q one gets
        H_I = (r/2)(a + a') and H_Q = (r/2) i(a' - a).
        """
        a = self.a
        return 0.5 * self.r * (a + a.conj().T), 0.5j * self.r * (a.conj().T - a)

    # --- reference quantities ------------------------------------------------
    def rotation_angle(self, envelope: np.ndarray) -> float:
        """Ideal two-level rotation angle r * integral(Re eps) for a sampled envelope."""
        return float(self.r * np.sum(envelope.real) * self.dt)

    def drag_beta_analytic(self) -> float:
        """First-order DRAG coefficient beta = -1/alpha in ns [Motzoi2009].

        The correction eps_Q = -d(eps_I)/dt / alpha cancels, to leading order
        in 1/alpha, the |1> -> |2> amplitude accumulated by a resonant drive.
        """
        return -1.0 / self.alpha

    def stark_detuning_analytic(self, envelope: np.ndarray, lam: float = 1.0) -> float:
        """Mean ac-Stark/frame detuning predicted at second order [Gambetta2011].

        delta(t) = (lam^2 - 4) * Omega(t)^2 / (4 alpha) with Omega = r*eps_I;
        returned as the pulse-averaged value in GHz.
        """
        omega = self.r * envelope.real
        return float(np.mean((lam**2 - 4.0) * omega**2 / (4.0 * self.alpha)) / TWO_PI)
