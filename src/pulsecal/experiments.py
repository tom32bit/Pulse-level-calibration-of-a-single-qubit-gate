"""Calibration experiments, run the way they are run on hardware.

Each routine returns what a real experiment returns: level populations
estimated from a finite number of shots, plus the noiseless value for
reference.  Fits are the standard ones used by calibration stacks
[Sheldon2016, Kelly2014, QiskitExperiments2023]:

* :func:`rabi_amplitude` / :func:`fit_rabi` turn a sinusoid in drive amplitude
  into the pi-pulse amplitude;
* :func:`chevron` maps the same sweep against drive frequency, whose
  interference pattern locates nu_q independently of the amplitude;
* :func:`drag_repeat` implements the alternating Rp/Rm sequence whose
  repetition amplifies the DRAG phase error [Chen2016];
* :func:`fine_amplitude` amplifies a residual over-rotation linearly in the
  number of repetitions [Sheldon2016];
* :func:`allxy` runs the 21-sequence diagnostic [Reed2013].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from . import sequences
from .device import Transmon
from .metrics import TARGETS, gate_error, leakage
from .propagate import gate_propagator, propagator
from .pulses import PulseSpec, drag_envelope


def sample(probs: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial readout of level populations; ``shots=0`` returns them exactly.

    Readout is assumed ideal and three-state resolving, as in the experiments
    that quantify leakage directly [Chen2016].
    """
    if not shots:
        return probs
    flat = probs.reshape(-1, probs.shape[-1])
    drawn = np.stack([rng.multinomial(shots, p / p.sum()) for p in flat]) / shots
    return drawn.reshape(probs.shape)


def populations(dev: Transmon, specs, detuning=0.0) -> np.ndarray:
    """Populations after a sequence, starting from |0>.

    ``specs`` may be a single list of pulses or a list of such lists; batched
    envelopes are zero-padded so that unequal sequence lengths broadcast.
    """
    if isinstance(specs, PulseSpec):
        specs = [specs]
    if isinstance(specs[0], PulseSpec):
        env = sequences.envelope(specs, dev.dt, detuning)
    else:
        raw = [sequences.envelope(s, dev.dt, detuning) for s in specs]
        width = max(len(e) for e in raw)
        env = np.stack([np.pad(e, (0, width - len(e))) for e in raw])
    return np.abs(propagator(dev, env, detuning)[..., :, 0]) ** 2


# --- amplitude (Rabi) calibration ------------------------------------------
def rabi_amplitude(dev: Transmon, spec: PulseSpec, amps: np.ndarray,
                   shots: int = 0, rng=None, detuning: float = 0.0) -> np.ndarray:
    """Level populations after one pulse, swept over envelope amplitude."""
    env = np.stack([drag_envelope(spec.with_(amp=a), dev.dt) for a in amps])
    probs = np.abs(propagator(dev, env, detuning)[..., :, 0]) ** 2
    return sample(probs, shots, rng)


def chevron(dev: Transmon, spec: PulseSpec, amps: np.ndarray,
            detunings: np.ndarray) -> np.ndarray:
    """P(|1>) over the (drive detuning, amplitude) plane.

    Off resonance the Bloch vector precesses about a tilted axis at the
    generalised Rabi rate sqrt(Omega^2 + delta^2), so contours of constant
    excitation bend away from resonance: the chevron whose apex marks nu_q.
    """
    env = np.stack([drag_envelope(spec.with_(amp=a), dev.dt) for a in amps])
    grid = propagator(dev, env[None, :, :], detunings[:, None])
    return np.abs(grid[..., 1, 0]) ** 2


def _rabi_model(a, off, amp, freq, phase):
    return off + amp * np.cos(2 * np.pi * freq * a + phase)


def fit_rabi(amps: np.ndarray, p1: np.ndarray) -> dict:
    """Fit P1 = off + amp*cos(2 pi f A + phi) and return the pi amplitude.

    The seed frequency comes from the largest non-DC Fourier component of the
    trace, which makes the fit robust to shot noise and to sweeping over more
    than one full oscillation.
    """
    spec = np.abs(np.fft.rfft(p1 - p1.mean()))
    f0 = np.fft.rfftfreq(len(amps), amps[1] - amps[0])[np.argmax(spec)]
    p0 = [p1.mean(), -0.5 * p1.ptp(), max(f0, 1e-6), 0.0]
    popt, pcov = curve_fit(_rabi_model, amps, p1, p0=p0, maxfev=20000)
    off, amp, freq, phase = popt
    phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
    a_pi = (np.pi - phase) / (2 * np.pi * freq)
    # d(a_pi)/d(freq, phase) propagated from the covariance of the fit.
    jac = np.array([-a_pi / freq, -1.0 / (2 * np.pi * freq)])
    err = float(np.sqrt(jac @ pcov[np.ix_([2, 3], [2, 3])] @ jac))
    return {"a_pi": float(a_pi), "a_pi_err": err, "offset": float(off),
            "contrast": float(2 * abs(amp)), "freq": float(freq), "phase": float(phase)}


# --- DRAG calibration -------------------------------------------------------
def drag_landscape(dev: Transmon, spec: PulseSpec, betas: np.ndarray,
                   detunings: np.ndarray, target: str = "X") -> tuple[np.ndarray, np.ndarray]:
    """Gate error and leakage over the (beta, drive detuning) plane.

    Leakage and phase error are minimised along *different* curves in this
    plane; DRAG works because adding the derivative quadrature together with a
    small frame detuning brings the two into coincidence [Motzoi2009,
    Gambetta2011, Chen2016].
    """
    env = np.stack([drag_envelope(spec.with_(beta=b), dev.dt) for b in betas])
    u = gate_propagator(dev, env[None, :, :], detunings[:, None])
    err = np.array([[gate_error(x, TARGETS[target]) for x in row] for row in u])
    leak = np.array([[leakage(x) for x in row] for row in u])
    return err, leak


def drag_repeat(dev: Transmon, spec: PulseSpec, betas: np.ndarray, n_reps: int = 5,
                shots: int = 0, rng=None, detuning: float = 0.0) -> np.ndarray:
    """P(|1>) after n repetitions of the alternating (X90, X-90) pair.

    Each pair is an identity for a perfectly corrected pulse, so repeating it
    amplifies the residual DRAG phase error while leaving the amplitude
    calibration untouched [Chen2016].
    """
    out = []
    for b in betas:
        p = spec.with_(beta=b)
        seq = [sequences.gate(p, ax) for _ in range(n_reps) for ax in ("x", "-x")]
        out.append(populations(dev, seq, detuning))
    return sample(np.stack(out), shots, rng)[..., 1]


# --- fine amplitude (error amplification) -----------------------------------
def fine_amplitude(dev: Transmon, spec: PulseSpec, scales: np.ndarray, n_max: int,
                   shots: int = 0, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """P(|1>) after X90 followed by n pi pulses, for n = 0..n_max.

    All rotations share the x axis, so the total angle is pi/2 + n(pi + dtheta)
    and a per-gate over-rotation dtheta shows up multiplied by n: the
    calibration precision improves linearly with sequence length
    [Sheldon2016].
    """
    ns = np.arange(n_max + 1)
    out = np.empty((len(scales), len(ns), dev.n_levels))
    for i, s in enumerate(scales):
        pi_pulse = spec.with_(amp=spec.amp * s)
        prefix = [sequences.gate(pi_pulse, "x")]
        out[i] = populations(dev, [prefix + [pi_pulse] * n for n in ns])
    return ns, sample(out, shots, rng)[..., 1]


def fit_overrotation(ns: np.ndarray, p1: np.ndarray) -> float:
    """Least-squares per-gate over-rotation dtheta from P1 = sin^2(theta/2)."""
    def model(n, dtheta, phi0):
        return 0.5 * (1 - np.cos(phi0 + n * (np.pi + dtheta)))
    popt, _ = curve_fit(model, ns, p1, p0=[0.0, np.pi / 2], maxfev=20000)
    return float(popt[0])


# --- reference optimum ------------------------------------------------------
def best_drag(dev: Transmon, spec: PulseSpec, target: str = "X",
              x0: tuple[float, float, float] | None = None,
              detuning: float | None = None) -> dict:
    """Directly minimise the gate error over amplitude, beta and detuning.

    This is the answer the shot-based calibrations are trying to find; keeping
    it separate makes it possible to quote how close a realistic experiment
    gets to the best the pulse shape can do.  Pinning ``detuning`` optimises
    only the two pulse parameters, which is the right thing for a second gate
    on the same drive line: the frequency is one hardware setting shared by
    every gate, while amplitude and beta are stored per gate.
    """
    from scipy.optimize import minimize

    def err(x):
        det = x[2] if detuning is None else detuning
        env = drag_envelope(spec.with_(amp=x[0], beta=x[1]), dev.dt)
        return gate_error(gate_propagator(dev, env, det), TARGETS[target])

    x0 = list(x0 or (spec.amp, dev.drag_beta_analytic() / 2, 0.0))
    free = x0 if detuning is None else x0[:2]
    res = minimize(lambda x: err(list(x) + [0.0]), free, method="Nelder-Mead",
                   options=dict(xatol=1e-11, fatol=1e-16, maxiter=8000))
    det = float(res.x[2]) if detuning is None else float(detuning)
    tuned = spec.with_(amp=res.x[0], beta=res.x[1], detuning=det)
    u = gate_propagator(dev, drag_envelope(tuned, dev.dt), det)
    return {"spec": tuned, "amp": float(res.x[0]), "beta": float(res.x[1]), "detuning": det,
            "error": float(res.fun), "leakage": leakage(u)}


# --- AllXY ------------------------------------------------------------------
def allxy(dev: Transmon, pi_pulse: PulseSpec, shots: int = 0, rng=None,
          detuning: float = 0.0, half: PulseSpec | None = None) -> np.ndarray:
    """P(|1>) for the 21 AllXY sequences [Reed2013]."""
    probs = populations(dev, sequences.allxy_specs(pi_pulse, half), detuning)
    return sample(probs, shots, rng)[..., 1]
