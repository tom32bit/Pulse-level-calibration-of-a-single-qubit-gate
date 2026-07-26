"""Validate the numerical core before trusting anything built on it.

Five independent checks, all written to ``results/validation.json``:

1. the piecewise-constant propagator against Qiskit Dynamics [Puzzuoli2023],
   which builds the frame transformation and the RWA itself from a
   laboratory-frame Hamiltonian;
2. the size of the rotating-wave approximation error, obtained by rerunning
   the same solve with the counter-rotating terms kept, and what that costs a
   *calibrated* gate rather than a raw propagator;
3. convergence in the transmon truncation, which fixes how many levels the
   rest of the study needs;
4. the analytic control derivatives against central differences;
5. the waveform this code integrates against the parametric ``Drag`` template a
   backend would accept, whose definition differs by the lifting offset.

Outputs: ``results/validation.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np

from pulsecal import dynamics, experiments as ex, metrics as M, utils
from pulsecal.device import Transmon
from pulsecal.grape import _objective
from pulsecal.propagate import propagator
from pulsecal.pulses import area, drag_envelope, qiskit_beta

cfg = utils.load_config()
dev = utils.build_device(cfg)
spec = utils.default_pulse(cfg)

# Analytic pi amplitude: the two-level rotation angle is r * integral(eps).
a_pi = 1.0 / (2 * dev.drive_strength * area(spec.with_(amp=1.0), dev.dt))
spec = spec.with_(amp=a_pi, beta=dev.drag_beta_analytic() / 2)
env = drag_envelope(spec, dev.dt)
print(f"analytic pi amplitude {a_pi:.6f}  (gate length {spec.duration * dev.dt:.2f} ns)")

# 1 + 2: solver cross-check and the cost of the RWA.
u_pwc = propagator(dev, env)
u_rwa = dynamics.propagator(dev, env, rwa=True)
u_lab = dynamics.propagator(dev, env, rwa=False)
solver_gap = float(np.abs(u_pwc - u_rwa).max())
rwa_gap = float(np.abs(u_rwa - u_lab).max())
print(f"[1] piecewise-constant vs Qiskit Dynamics : {solver_gap:.2e}")
print(f"[2] RWA vs laboratory frame              : {rwa_gap:.2e}")

# A calibration performed in the laboratory frame absorbs most of that gap into
# its amplitude and frame detuning, so the RWA error is far less damaging to a
# *calibrated* gate than the raw propagator difference suggests.
lab_raw = M.gate_error(u_lab, M.TARGETS["X"])
rwa_raw = M.gate_error(u_pwc, M.TARGETS["X"])
lab_cal = M.gate_error(u_lab, M.TARGETS["X"], z_free=True)
print(f"    gate error, same pulse: RWA {rwa_raw:.3e} | lab {lab_raw:.3e} "
      f"| lab with virtual-Z {lab_cal:.3e}")

# 3: truncation convergence.
truncation = {}
for n in (3, 4, 5, 6):
    d = Transmon(**{**cfg["device"], "n_levels": n})
    a = 1.0 / (2 * d.drive_strength * area(spec.with_(amp=1.0), d.dt))
    best = ex.best_drag(d, spec.with_(amp=a))
    truncation[n] = {"error": best["error"], "leakage": best["leakage"],
                     "beta": best["beta"], "detuning": best["detuning"]}
    print(f"[3] n_levels={n}: calibrated error {best['error']:.4e}, "
          f"beta {best['beta']:.4f} ns, detuning {best['detuning'] * 1e3:+.2f} MHz")

# 3b: is the gap between the fitted and the analytic pi amplitude physics or
# fit error?  Refitting without shot noise, and again on a genuine two-level
# device, separates the two: the two-level case must return the analytic value
# exactly, so anything left over is the higher levels.
amps = np.linspace(0.0, 1.0, 241)
fit_4 = ex.fit_rabi(amps, ex.rabi_amplitude(dev, spec.with_(beta=0.0), amps)[:, 1])["a_pi"]
two = Transmon(**{**cfg["device"], "n_levels": 2})
fit_2 = ex.fit_rabi(amps, ex.rabi_amplitude(two, spec.with_(beta=0.0), amps)[:, 1])["a_pi"]
print(f"[3b] pi amplitude: analytic {a_pi:.6f} | two-level fit {fit_2:.6f} "
      f"({1e2 * (fit_2 / a_pi - 1):+.3f}%) | four-level fit {fit_4:.6f} "
      f"({1e2 * (fit_4 / a_pi - 1):+.3f}%)")

# 4: exact gradients against central differences.
rng = utils.rng(cfg, "validate")
x = rng.normal(scale=0.3, size=2 * 16)
_, grad = _objective(x, dev, M.TARGETS["X"], 1e-3)
h, num = 1e-6, np.empty_like(grad)
for i in range(x.size):
    step = np.zeros_like(x)
    step[i] = h
    num[i] = (_objective(x + step, dev, M.TARGETS["X"], 1e-3)[0]
              - _objective(x - step, dev, M.TARGETS["X"], 1e-3)[0]) / (2 * h)
grad_gap = float(np.abs(grad - num).max() / np.abs(num).max())
print(f"[4] analytic vs finite-difference gradient: {grad_gap:.2e} (relative)")

# 5: the emitted qiskit.pulse template against the simulated waveform.
from qiskit.pulse.library import Drag  # noqa: E402  (kept next to its single use)

template = Drag(spec.duration, spec.amp, spec.sigma, qiskit_beta(spec, dev.dt),
                limit_amplitude=False).get_waveform().samples
template_gap = float(np.abs(env - template).max() / np.abs(env).max())
print(f"[5] parametric Drag template vs waveform  : {template_gap:.2e} (relative)")

utils.save("validation", {
    "pi_amplitude_analytic": a_pi,
    "gate_length_ns": spec.duration * dev.dt,
    "solver_agreement": solver_gap,
    "rwa_propagator_error": rwa_gap,
    "gate_error_rwa": rwa_raw,
    "gate_error_lab": lab_raw,
    "gate_error_lab_zfree": lab_cal,
    "truncation": truncation,
    "pi_amplitude_fit_two_level": fit_2,
    "pi_amplitude_fit_four_level": fit_4,
    "gradient_relative_error": grad_gap,
    "drag_template_relative_error": template_gap,
})
print("\nwrote results/validation.json")
