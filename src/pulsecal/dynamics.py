"""Independent propagator from Qiskit Dynamics, used to validate the core.

:mod:`.propagate` integrates an already-rotating, already-truncated model with
a hand-rolled product of matrix exponentials.  Qiskit Dynamics [Puzzuoli2023]
starts instead from the *laboratory-frame* Hamiltonian, applies the frame
transformation and the rotating-wave approximation itself, and integrates with
an adaptive ODE solver.  Agreement between the two therefore tests the frame
algebra, the RWA and the discretisation at once, not just the arithmetic.

Setting ``rwa=False`` keeps the counter-rotating terms and so measures the
size of the RWA error itself, which at these drive strengths is the
Bloch-Siegert-scale shift ~ (Omega / 4 w_d).

Frame convention
----------------
Qiskit's ``Signal`` produces s(t) = Re[f(t) e^{i(2 pi nu t + phi)}], so after
moving to the rotating frame the surviving drive term is (r/2)(f* a' + f a),
the complex conjugate of the (r/2)(eps a' + eps* a) used everywhere else here.
The envelope is conjugated on the way in so that both conventions describe the
same physical waveform.
"""

from __future__ import annotations

import numpy as np

from .device import TWO_PI, Transmon


def propagator(dev: Transmon, envelope: np.ndarray, detuning: float = 0.0,
               rwa: bool = True, atol: float = 1e-12, rtol: float = 1e-10) -> np.ndarray:
    """Propagator of a sampled pulse, in the same frame as :func:`propagate.propagator`.

    The rotating frame is the *harmonic* part 2 pi nu_d a'a only, so the
    anharmonic ladder and the detuning survive in the drift exactly as in the
    analytic model.
    """
    from qiskit_dynamics import DiscreteSignal, Solver

    a, n_op = dev.a, dev.n_op
    nu_d = dev.qubit_frequency + detuning
    static = TWO_PI * (dev.qubit_frequency * n_op
                       + 0.5 * dev.anharmonicity * n_op @ (n_op - np.eye(dev.n_levels)))
    solver = Solver(
        static_hamiltonian=static,
        hamiltonian_operators=[dev.r * (a + a.conj().T)],
        rotating_frame=TWO_PI * nu_d * n_op,
        rwa_cutoff_freq=2.0 * nu_d if rwa else None,
        rwa_carrier_freqs=[nu_d] if rwa else None,
    )
    signal = DiscreteSignal(dt=dev.dt, samples=envelope.conj(), carrier_freq=nu_d)
    result = solver.solve(t_span=[0.0, len(envelope) * dev.dt],
                          y0=np.eye(dev.n_levels, dtype=complex),
                          signals=[signal], atol=atol, rtol=rtol, method="DOP853")
    return np.asarray(result.y[-1])
