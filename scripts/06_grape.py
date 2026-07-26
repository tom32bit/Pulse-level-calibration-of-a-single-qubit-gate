"""Numerical optimal control: what the analytic pulse leaves on the table.

DRAG is a one-parameter ansatz.  GRAPE drops the ansatz and optimises both
quadratures sample by sample, ascending the gate fidelity of the *leaky*
system directly [Khaneja2005].  Two details make it converge here rather than
crawl: derivatives dU_k/du taken exactly in the instantaneous eigenbasis
instead of to first order in dt [Machnes2011], and a quasi-Newton search in
place of fixed-step ascent [deFouquieres2011].  A roughness penalty with
virtual zeros outside the pulse keeps the waveform band-limited and makes it
start and end at zero, which is what stops the optimiser from returning
something an AWG cannot play [Werninghaus2021].

The optimiser is given the calibrated DRAG pulse as one of its starting points,
so it can only improve on analytic control.  What it buys is visible in panel
c: the transient excursion onto |2> is just as large -- that is set by
Omega/alpha and no pulse of this length avoids it -- but almost all of it is
returned.  The residual leakage, which is the entire remaining error of the
calibrated DRAG pulse, drops by four orders of magnitude.

Two caveats are checked rather than assumed.  A gate error of 10^-9 is only
meaningful if the model is converged there, so the optimised pulse is
re-simulated on deeper transmon ladders.  And an optimiser told only to
maximise fidelity has no reason to stay tolerant of drift, so panel d measures
what happens when the amplitude or the drive frequency moves afterwards.

Outputs: ``figures/fig06_grape.png``, ``results/grape.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import grape, metrics as M, plotting as pl, utils
from pulsecal.device import Transmon
from pulsecal.propagate import gate_propagator, trajectory
from pulsecal.pulses import drag_envelope

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
gcfg = cfg["grape"]
pl.use_style()

cal = utils.load("drag_calibration")
best_amp = utils.load("fine_amplitude")["a_pi_fine"]
drag_spec = spec.with_(amp=best_amp, beta=cal["beta_optimum_ns"])
drag_env = drag_envelope(drag_spec, dev.dt)
drag_u = gate_propagator(dev, drag_env)
drag_err, drag_leak = M.gate_error(drag_u, M.TARGETS["X"]), M.leakage(drag_u)

# The optimiser records the error after every accepted iterate, and the run is
# repeated from several smooth random starts so the reported optimum is not a
# single lucky descent.
res = grape.optimize(dev, M.TARGETS["X"], n_slices=spec.duration, max_amp=gcfg["max_amp"],
                     smoothness=gcfg["smoothness"], maxiter=gcfg["maxiter"],
                     n_restarts=gcfg["n_restarts"], seed=cfg["measurement"]["seed"],
                     warm_start=drag_env)
cold = grape.optimize(dev, M.TARGETS["X"], n_slices=spec.duration, max_amp=gcfg["max_amp"],
                      smoothness=gcfg["smoothness"], maxiter=gcfg["maxiter"],
                      n_restarts=gcfg["n_restarts"], seed=cfg["measurement"]["seed"])

t = (np.arange(spec.duration) + 0.5) * dev.dt
psi1 = np.zeros(dev.n_levels, complex)
psi1[1] = 1.0
leaked = {name: M.populations(trajectory(dev, env, psi1))[:, 2:].sum(1)
          for name, env in (("drag", drag_env), ("grape", res.envelope))}

# Robustness: an optimiser given no reason to care about calibration drift can
# buy its last decade of fidelity with a pulse that is sharper in parameter
# space, so the comparison has to be made explicitly rather than assumed.
amp_errs = np.linspace(-0.03, 0.03, 61)
det_errs = np.linspace(-0.006, 0.006, 61)
robust = {
    name: (np.array([M.gate_error(gate_propagator(dev, (1 + e) * env), M.TARGETS["X"])
                     for e in amp_errs]),
           np.array([M.gate_error(gate_propagator(dev, env, d), M.TARGETS["X"])
                     for d in det_errs]))
    for name, env in (("drag", drag_env), ("grape", res.envelope))}

# The optimum is only meaningful if it survives the model truncation it was
# found in; four levels was chosen in script 00 for the DRAG pulse, and the
# GRAPE pulse is checked against deeper ladders here.
truncation = {}
for n in (3, 5, 6):
    deep = Transmon(**{**cfg["device"], "n_levels": n})
    truncation[n] = M.gate_error(gate_propagator(deep, res.envelope), M.TARGETS["X"])


# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1))
(ax_wave, ax_conv), (ax_leak, ax_rob) = axes
fig.subplots_adjust(hspace=0.46, wspace=0.30)

for env, color, lw in ((drag_env, pl.DRAG, 1.6), (res.envelope, pl.OPT, 2.0)):
    ax_wave.plot(t, env.real, color=color, lw=lw)
    ax_wave.plot(t, env.imag, color=color, lw=lw, ls=(0, (3, 2)))
ax_wave.axhline(0, color=pl.MUTED, lw=0.7)
ax_wave.set(xlabel="time  (ns)", ylabel="envelope", ylim=(-0.055, 0.335))
ax_wave.set_title("a   the two pulses")
pl.annotate(ax_wave, "DRAG", (t[24], drag_env.real[24]), (7.8, 0.305), color=pl.DRAG)
pl.annotate(ax_wave, "GRAPE", (t[13], res.envelope.real[13]), (0.3, 0.275), color=pl.OPT)
ax_wave.text(0.5, 0.035, "solid: in phase      dashed: quadrature",
             transform=ax_wave.transAxes, ha="center", fontsize=8, color=pl.INK2)

ax_conv.semilogy(np.arange(1, len(res.history) + 1), res.history, color=pl.OPT, lw=1.9)
ax_conv.semilogy(np.arange(1, len(cold.history) + 1), cold.history, color=pl.INK, lw=1.4,
                 ls=(0, (4, 2.5)))
ax_conv.axhline(drag_err, color=pl.DRAG, lw=1.5)
ax_conv.set(xlabel="L-BFGS iteration", ylabel="gate error", ylim=(2e-10, 1.0))
ax_conv.set_title("b   convergence")
pl.annotate(ax_conv, f"calibrated DRAG\n{drag_err:.1e}", (len(res.history) * 0.55, drag_err),
            (len(res.history) * 0.18, drag_err * 12), color=pl.DRAG)
pl.annotate(ax_conv, f"from DRAG\n{res.error:.1e}", (len(res.history), res.error),
            (len(res.history) * 0.42, res.error * 0.06), color=pl.OPT)
pl.annotate(ax_conv, "from random starts",
            (len(cold.history) * 0.45, cold.history[int(len(cold.history) * 0.45) - 1]),
            (len(res.history) * 0.42, drag_err * 0.9e-1), color=pl.INK)

for (name, curve), color in zip(leaked.items(), (pl.DRAG, pl.OPT)):
    ax_leak.semilogy(np.append(0, t), curve, color=color, lw=1.9)
    ax_leak.text(t[-1] + 0.3, curve[-1], f"  {curve[-1]:.0e}", va="center", fontsize=8.5,
                 color=color)
ax_leak.set(xlabel="time  (ns)", ylabel="leaked population", ylim=(1e-11, 2.0),
            xlim=(0, t[-1] + 2.7))
ax_leak.set_title(r"c   what is left on $|2\rangle$")
ax_leak.text(0.035, 0.95, "both make the same excursion;\nonly one puts it all back",
             transform=ax_leak.transAxes, va="top", fontsize=8.5, color=pl.INK2)

for (name, (amp_curve, _)), color in zip(robust.items(), (pl.DRAG, pl.OPT)):
    ax_rob.semilogy(amp_errs * 1e2, amp_curve, color=color, lw=1.9)
ax_rob.axhline(1e-4, color=pl.MUTED, lw=0.9, ls=(0, (4, 3)))
ax_rob.set(xlabel="amplitude miscalibration  (%)", ylabel="gate error", ylim=(1e-9, 2e-2))
ax_rob.set_title("d   robustness to drift")
ax_rob.text(2.95, 1.35e-4, r"$10^{-4}$", ha="right", va="bottom", fontsize=8, color=pl.INK2)
pl.annotate(ax_rob, "DRAG", (-1.9, robust["drag"][0][11]), (-1.4, 4.5e-3), color=pl.DRAG)
pl.annotate(ax_rob, "GRAPE", (0.42, robust["grape"][0][34]), (0.95, 1.2e-6), color=pl.OPT)

inset = ax_rob.inset_axes([0.06, 0.12, 0.34, 0.30])
for (name, (_, det_curve)), color in zip(robust.items(), (pl.DRAG, pl.OPT)):
    inset.semilogy(det_errs * 1e3, det_curve, color=color, lw=1.4)
inset.set(ylim=(1e-9, 2e-2), yticks=[1e-8, 1e-4], xticks=[-5, 0, 5])
inset.set_title("vs detuning (MHz)", fontsize=7.5, color=pl.INK2, pad=2)
inset.tick_params(labelsize=7)
for side in ("top", "right"):
    inset.spines[side].set_visible(False)

pl.despine(*axes.ravel())
pl.caption(fig, f"{spec.duration} samples per quadrature, {gcfg['n_restarts']} random restarts "
                f"plus the DRAG warm start, amplitude bound {gcfg['max_amp']}, roughness weight "
                f"{gcfg['smoothness']:g}. Gate length {spec.duration * dev.dt:.2f} ns is "
                "unchanged; only the shape is free.")
pl.save(fig, utils.FIGURES / "fig06_grape.png")

utils.save("grape", {
    "drag_error": drag_err, "drag_leakage": drag_leak,
    "grape_error": res.error, "grape_leakage": res.leakage,
    "grape_error_cold_start": cold.error,
    "improvement": drag_err / res.error, "iterations": res.n_iter,
    "n_slices": spec.duration, "config": gcfg,
    "envelope_real": res.envelope.real, "envelope_imag": res.envelope.imag,
    "peak_amplitude": float(np.abs(res.envelope).max()),
    "truncation_check": truncation,
    "amplitude_errors": amp_errs, "detuning_errors_ghz": det_errs,
    "robustness": {k: {"amplitude": v[0], "detuning": v[1]} for k, v in robust.items()}})
print(f"calibrated DRAG : error {drag_err:.3e}, leakage {drag_leak:.3e}")
print(f"GRAPE (warm)    : error {res.error:.3e}, leakage {res.leakage:.3e}, "
      f"{res.n_iter} iterations")
print(f"GRAPE (cold)    : error {cold.error:.3e}")
print(f"improvement over analytic control: {drag_err / res.error:.0f}x")
print("truncation check (GRAPE pulse re-simulated): "
      + ", ".join(f"n={n}: {e:.2e}" for n, e in truncation.items()))
for name, (amp_curve, det_curve) in robust.items():
    half = np.interp(1e-4, amp_curve[len(amp_errs) // 2:], amp_errs[len(amp_errs) // 2:]) * 1e2
    print(f"{name:5s}: stays below 1e-4 up to {half:.2f}% amplitude error")
