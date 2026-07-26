"""Checks on the parts of the code where a silent error would not show up in a figure.

Run with ``pytest -q`` from the project root.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulsecal import experiments as ex, metrics as M, sequences, utils  # noqa: E402
from pulsecal.device import Transmon  # noqa: E402
from pulsecal.grape import _objective  # noqa: E402
from pulsecal.propagate import (control_gradient, gate_propagator,  # noqa: E402
                                partial_products, propagator, step_propagators)
from pulsecal.pulses import (area, drag_envelope, lifted_gaussian,  # noqa: E402
                             qiskit_beta)

CFG = utils.load_config()
DEV = utils.build_device(CFG)
SPEC = utils.default_pulse(CFG)


# --- pulses ------------------------------------------------------------------
def test_lifted_gaussian_matches_qiskit():
    """Our envelope must be the waveform Qiskit would upload, sample for sample."""
    from qiskit.pulse.library import Gaussian

    g, _ = lifted_gaussian(SPEC.duration, SPEC.sigma)
    ref = Gaussian(duration=SPEC.duration, amp=1.0, sigma=SPEC.sigma).get_waveform().samples
    assert np.abs(g - ref.real).max() < 1e-12


def test_drag_quadrature_is_the_derivative():
    """eps_Q must be beta * d(eps_I)/dt for the envelope actually played."""
    spec = SPEC.with_(amp=0.3, beta=0.4)
    env = drag_envelope(spec, DEV.dt)
    numeric = np.gradient(env.real, DEV.dt)
    assert np.abs(env.imag - spec.beta * numeric)[4:-4].max() < 1e-4


def test_qiskit_drag_template_is_close():
    """The parametric template a backend accepts differs only by the lifting offset."""
    from qiskit.pulse.library import Drag

    spec = SPEC.with_(amp=0.25, beta=0.43)
    ref = Drag(spec.duration, spec.amp, spec.sigma, qiskit_beta(spec, DEV.dt),
               limit_amplitude=False).get_waveform().samples
    env = drag_envelope(spec, DEV.dt)
    assert np.abs(env - ref).max() / np.abs(env).max() < 0.05


def test_pi_amplitude_matches_analytic_rotation_angle():
    """r * integral(eps) = pi is the two-level prediction; the fit must land on it."""
    a_pi = 1.0 / (2 * DEV.drive_strength * area(SPEC.with_(amp=1.0), DEV.dt))
    assert DEV.rotation_angle(drag_envelope(SPEC.with_(amp=a_pi), DEV.dt)) == pytest.approx(np.pi)


# --- propagation -------------------------------------------------------------
def test_propagator_is_unitary():
    u = propagator(DEV, drag_envelope(SPEC.with_(amp=0.3, beta=0.4), DEV.dt), 0.004)
    assert np.abs(u.conj().T @ u - np.eye(DEV.n_levels)).max() < 1e-12


def test_partial_products_reconstruct_the_whole():
    us = step_propagators(np.random.default_rng(0).normal(size=(7, 3, 3)) * 0.1, 0.2)
    us = us + us.conj().swapaxes(-1, -2) * 0  # keep dtype complex
    fwd, bwd = partial_products(us)
    assert np.abs(fwd[-1] - bwd[0]).max() < 1e-12
    for k in range(len(us)):
        assert np.abs(bwd[k + 1] @ us[k] @ fwd[k] - fwd[-1]).max() < 1e-11


def test_batched_propagator_matches_loop():
    """The vectorised sweep used for every map must equal one-at-a-time evaluation."""
    amps, dets = np.array([0.1, 0.2, 0.3]), np.array([-0.004, 0.0, 0.004])
    env = np.stack([drag_envelope(SPEC.with_(amp=a), DEV.dt) for a in amps])
    batched = propagator(DEV, env[None, :, :], dets[:, None])
    for i, d in enumerate(dets):
        for j in range(len(amps)):
            assert np.abs(batched[i, j] - propagator(DEV, env[j], d)).max() < 1e-12


def test_gate_propagator_removes_only_the_frame():
    """The qubit-frame correction is diagonal and does nothing on resonance."""
    env = drag_envelope(SPEC.with_(amp=0.25), DEV.dt)
    assert np.abs(gate_propagator(DEV, env, 0.0) - propagator(DEV, env, 0.0)).max() < 1e-14
    ratio = gate_propagator(DEV, env, 0.005) / propagator(DEV, env, 0.005)
    assert np.abs(np.abs(ratio) - 1.0).max() < 1e-12          # a pure phase
    assert np.abs(ratio - ratio[:, :1]).max() < 1e-12          # constant along each row


def test_matches_qiskit_dynamics():
    """Independent solver, laboratory-frame Hamiltonian, adaptive integrator."""
    dynamics = pytest.importorskip("pulsecal.dynamics")
    env = drag_envelope(SPEC.with_(amp=0.25, beta=0.43), DEV.dt)
    for det in (0.0, -0.006):
        gap = np.abs(propagator(DEV, env, det) - dynamics.propagator(DEV, env, det)).max()
        assert gap < 1e-7


# --- metrics -----------------------------------------------------------------
def test_fidelity_of_exact_target_is_one():
    embedded = np.eye(DEV.n_levels, dtype=complex)
    embedded[:2, :2] = M.TARGETS["X"]
    assert M.average_gate_fidelity(embedded, M.TARGETS["X"]) == pytest.approx(1.0)


def test_leakage_counts_population_outside_the_qubit():
    """A gate that parks everything on |2> has L1 = 1 and no fidelity."""
    u = np.zeros((DEV.n_levels, DEV.n_levels), dtype=complex)
    u[2, 0] = u[2, 1] = 1 / np.sqrt(2)
    assert M.leakage(u) == pytest.approx(1.0)


def test_z_free_fidelity_forgives_a_virtual_z():
    embedded = np.eye(DEV.n_levels, dtype=complex)
    embedded[:2, :2] = M.rotation(M.SZ, 0.3) @ M.TARGETS["X"]
    assert M.average_gate_fidelity(embedded, M.TARGETS["X"]) < 0.99
    assert M.average_gate_fidelity(embedded, M.TARGETS["X"], z_free=True) == pytest.approx(1.0)


def test_bloch_contracts_when_population_leaks():
    leaky = np.array([0.6, 0.6, 0.52941], dtype=complex)
    assert np.linalg.norm(M.bloch(leaky)) < 1.0


# --- gradients ---------------------------------------------------------------
def test_analytic_gradient_matches_finite_differences():
    """The exact eigenbasis derivative is what lets L-BFGS converge; check it."""
    rng = np.random.default_rng(7)
    x = rng.normal(scale=0.3, size=2 * 12)
    _, grad = _objective(x, DEV, M.TARGETS["X"], 1e-3)
    numeric = np.empty_like(grad)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = 1e-6
        numeric[i] = (_objective(x + step, DEV, M.TARGETS["X"], 1e-3)[0]
                      - _objective(x - step, DEV, M.TARGETS["X"], 1e-3)[0]) / 2e-6
    assert np.abs(grad - numeric).max() / np.abs(numeric).max() < 1e-6


def test_control_gradient_is_zero_for_a_constant_cost():
    env = drag_envelope(SPEC.with_(amp=0.25), DEV.dt)
    gi, gq = control_gradient(DEV, env, np.zeros((DEV.n_levels, DEV.n_levels), dtype=complex))
    assert np.abs(gi).max() + np.abs(gq).max() == pytest.approx(0.0)


# --- sequences and experiments ----------------------------------------------
def test_identity_gate_plays_a_blank_of_the_right_length():
    idle = sequences.gate(SPEC.with_(amp=0.3), "I")
    env = drag_envelope(idle, DEV.dt)
    assert len(env) == SPEC.duration and np.abs(env).max() == 0.0


@pytest.mark.parametrize("det", [0.0, 0.004])
def test_sequence_composes_the_gates_the_compiler_sees(det):
    """A sequence must equal the product of its gates in the qubit frame.

    That identity is the whole point of tracking the frame: without the phase
    advance the concatenated waveform describes a different circuit, and at a
    detuning of a few MHz the two differ at the percent level.
    """
    sx = SPEC.with_(amp=0.125, beta=0.25)
    g = gate_propagator(DEV, drag_envelope(sx, DEV.dt), det)
    tracked = sequences.run(DEV, [sx, sx], det)
    assert np.abs(tracked - g @ g).max() < 1e-11
    naive = propagator(DEV, np.concatenate([drag_envelope(sx, DEV.dt)] * 2), det)
    assert (np.abs(naive - propagator(DEV, sequences.envelope([sx, sx], DEV.dt, det), det)).max()
            > 1e-2) == bool(det)


def test_allxy_staircase_is_flat_for_a_calibrated_pulse():
    x = ex.best_drag(DEV, SPEC.with_(amp=0.25), "X")
    sx = ex.best_drag(DEV, SPEC.with_(amp=0.125), "X90", detuning=x["detuning"])
    p1 = ex.allxy(DEV, x["spec"], detuning=x["detuning"], half=sx["spec"])
    assert np.abs(p1 - sequences.ALLXY_IDEAL).max() < 5e-3


def test_rabi_fit_recovers_a_known_amplitude():
    amps = np.linspace(0.0, 1.0, 201)
    p1 = ex.rabi_amplitude(DEV, SPEC, amps)[:, 1]
    a_pi = 1.0 / (2 * DEV.drive_strength * area(SPEC.with_(amp=1.0), DEV.dt))
    assert ex.fit_rabi(amps, p1)["a_pi"] == pytest.approx(a_pi, rel=0.02)


def test_sampling_is_unbiased_and_reproducible():
    probs = np.array([[0.2, 0.5, 0.3, 0.0]])
    rng = np.random.default_rng(3)
    draws = np.stack([ex.sample(probs, 4096, rng)[0] for _ in range(200)])
    assert np.abs(draws.mean(axis=0) - probs[0]).max() < 5e-3
    assert np.array_equal(ex.sample(probs, 64, np.random.default_rng(1)),
                          ex.sample(probs, 64, np.random.default_rng(1)))


# --- model ------------------------------------------------------------------
def test_drive_strength_only_enters_through_its_product_with_the_envelope():
    """Doubling r and halving the amplitude must leave every error unchanged."""
    twice = Transmon(**{**CFG["device"], "drive_strength": 2 * DEV.drive_strength})
    a = 1.0 / (2 * DEV.drive_strength * area(SPEC.with_(amp=1.0), DEV.dt))
    u1 = propagator(DEV, drag_envelope(SPEC.with_(amp=a, beta=0.25), DEV.dt))
    u2 = propagator(twice, drag_envelope(SPEC.with_(amp=a / 2, beta=0.25), twice.dt))
    assert np.abs(u1 - u2).max() < 1e-12


def test_four_levels_is_enough():
    """The truncation the study is run at must be converged for a calibrated pulse."""
    errors = []
    for n in (4, 5, 6):
        dev = Transmon(**{**CFG["device"], "n_levels": n})
        a = 1.0 / (2 * dev.drive_strength * area(SPEC.with_(amp=1.0), dev.dt))
        errors.append(ex.best_drag(dev, SPEC.with_(amp=a))["error"])
    assert abs(errors[1] - errors[0]) / errors[1] < 0.05
    assert abs(errors[2] - errors[1]) / errors[2] < 1e-3
