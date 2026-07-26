"""How short can the gate be? Error against gate length for four control strategies.

Everything so far was done at one pulse length.  Sweeping it turns the study
into the question that actually decides a device's clock rate: a shorter gate
spends less time exposed to decoherence, but its bandwidth grows as 1/T and
starts to overlap the |1> -> |2> transition sitting alpha away, so leakage
grows.  Plotting error against T for controls of increasing sophistication
separates what is a property of the pulse shape from what is a property of the
system [Motzoi2009, Gambetta2011, Werninghaus2021].

Four strategies, each calibrated afresh at every duration:

1. amplitude only -- a lifted Gaussian with the rotation angle set to pi;
2. analytic DRAG at lambda = 1/2, no further tuning;
3. DRAG with amplitude, beta and detuning optimised together;
4. GRAPE over both quadratures at the AWG sample rate.

The gap between curves 1 and 3 is what an ansatz plus calibration buys; the gap
between 3 and 4 is what is left for numerical optimal control.

Two limits close in from the short-duration side and they are not the same
limit.  A Gaussian of peak amplitude A_max runs out of pulse area first, which
ends the *ansatz* rather than the gate.  A rectangle carries the most area any
bounded pulse can, so pi / (r A_max) is where an X gate becomes impossible for
any shape at all -- the honest speed limit for this model, and a factor of two
shorter than where the Gaussian gives up.

Outputs: ``figures/fig07_speed_limit.png``, ``results/speed_limit.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import experiments as ex, grape, metrics as M, plotting as pl, utils
from pulsecal.propagate import gate_propagator
from pulsecal.pulses import PulseSpec, area, drag_envelope

cfg = utils.load_config()
dev = utils.build_device(cfg)
gcfg = cfg["grape"]
pl.use_style()

sigma_ratio = cfg["pulse"]["sigma_ratio"]
durations = np.array([10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96])
rows = []
for n in durations:
    spec = PulseSpec(duration=n, sigma=sigma_ratio * n, amp=1.0)
    # Amplitude that makes the two-level rotation angle exactly pi.
    a_pi = 1.0 / (2 * dev.drive_strength * area(spec, dev.dt))
    spec = spec.with_(amp=a_pi)

    def err(sp, det=0.0):
        return M.gate_error(gate_propagator(dev, drag_envelope(sp, dev.dt), det), M.TARGETS["X"])

    plain = err(spec)
    analytic = err(spec.with_(beta=dev.drag_beta_analytic() / 2))
    tuned = ex.best_drag(dev, spec)
    opt = grape.optimize(dev, M.TARGETS["X"], n_slices=n, max_amp=gcfg["max_amp"],
                         smoothness=gcfg["smoothness"], maxiter=gcfg["maxiter"],
                         n_restarts=3, seed=cfg["measurement"]["seed"],
                         warm_start=drag_envelope(tuned["spec"], dev.dt))
    rows.append({"duration": int(n), "t_gate": n * dev.dt, "a_pi": a_pi,
                 "plain": plain, "analytic_drag": analytic,
                 "tuned_drag": tuned["error"], "tuned_leakage": tuned["leakage"],
                 "grape": opt.error, "grape_leakage": opt.leakage,
                 "grape_peak": float(np.abs(opt.envelope).max())})
    print(f"T = {n * dev.dt:5.2f} ns  A_pi = {a_pi:.3f} | plain {plain:.2e} | "
          f"analytic {analytic:.2e} | tuned {tuned['error']:.2e} | GRAPE {opt.error:.2e}")

t_gate = np.array([r["t_gate"] for r in rows])
series = [("amplitude only", "plain", pl.PLAIN, "-"),
          (r"DRAG, $\lambda=1/2$", "analytic_drag", pl.DRAG, (0, (4, 2.5))),
          ("DRAG, calibrated", "tuned_drag", pl.DRAG, "-"),
          ("GRAPE", "grape", pl.OPT, "-")]

# Two walls, and they are not the same one.  A Gaussian of peak amplitude
# max_amp runs out of area first; a rectangle of the same peak carries the most
# area any bounded pulse can, so pi / (r * max_amp) is where the gate becomes
# impossible for *any* shape.  The gap between them is room that only a
# non-Gaussian pulse can use.
a_pi_curve = np.array([r["a_pi"] for r in rows])
t_gauss = float(np.interp(gcfg["max_amp"], a_pi_curve[::-1], t_gate[::-1]))
t_wall = float(np.pi / (dev.r * gcfg["max_amp"]))

# --- figure ----------------------------------------------------------------
fig, (ax_err, ax_leak) = plt.subplots(1, 2, figsize=(8.4, 4.3))
fig.subplots_adjust(wspace=0.26)

for label, key, color, style in series:
    ax_err.loglog(t_gate, [r[key] for r in rows], color=color, ls=style, lw=1.9,
                  marker="o", ms=4)
ax_err.axvspan(t_wall * 0.5, t_gauss, color=pl.MUTED, alpha=0.13, lw=0)
ax_err.axvline(t_wall, color=pl.INK, lw=1.1, ls=(0, (5, 2.5)))
ax_err.axhline(1e-4, color=pl.MUTED, lw=0.9, ls=(0, (4, 3)))
ax_err.set(xlabel="gate length  (ns)", ylabel="gate error",
           xlim=(t_wall * 0.82, t_gate[-1] * 1.15))
ax_err.set_title("a   error against gate length")
# Direct labels sit beside their own curve rather than in a legend box.
for (label, key, color, _), (x, y) in zip(series, ((15.0, 3.4e-2), (10.0, 1.1e-3),
                                                  (11.0, 1.1e-6), (5.6, 6.0e-10))):
    ax_err.text(x, y, label, fontsize=8.5, color=color, ha="center", va="center")
ax_err.text(t_gauss * 1.07, 6e-11, "Gaussian needs\nmore than full scale", fontsize=8,
            color=pl.INK2, ha="left", va="center")
ax_err.text(t_wall * 0.95, 4.5e-1, "no bounded pulse\nof any shape", fontsize=8, color=pl.INK,
            ha="left", va="top")
ax_err.text(20.5, 1.35e-4, r"$10^{-4}$", fontsize=8, color=pl.INK2, ha="right", va="bottom")

for label, key, color, style in (("DRAG, calibrated", "tuned_leakage", pl.DRAG, "-"),
                                 ("GRAPE", "grape_leakage", pl.OPT, "-")):
    ax_leak.loglog(t_gate, [r[key] for r in rows], color=color, ls=style, lw=1.9,
                   marker="o", ms=4)
for label, color, x, y in (("DRAG, calibrated", pl.DRAG, 12.0, 3.0e-3),
                           ("GRAPE", pl.OPT, 4.4, 6.0e-9)):
    ax_leak.text(x, y, label, fontsize=8.5, color=color, ha="center", va="center")
ax_leak.axvspan(t_wall * 0.5, t_gauss, color=pl.MUTED, alpha=0.13, lw=0)
ax_leak.axvline(t_wall, color=pl.INK, lw=1.1, ls=(0, (5, 2.5)))
ax_leak.set(xlabel="gate length  (ns)", ylabel=r"leakage $L_1$",
            xlim=(t_wall * 0.82, t_gate[-1] * 1.15))
ax_leak.set_title("b   what is still leaking")

for ax in (ax_err, ax_leak):
    ax.set_xticks([1.5, 2.5, 4, 7, 12, 21], ["1.5", "2.5", "4", "7", "12", "21"])
    ax.minorticks_off()
pl.despine(ax_err, ax_leak)
pl.caption(fig, f"Every point is recalibrated from scratch at that duration: amplitude by "
                f"construction, DRAG by direct minimisation, GRAPE from {durations.min()} to "
                f"{durations.max()} samples with 3 restarts plus a DRAG warm start. "
                f"sigma = T/{1 / sigma_ratio:.0f} throughout; shading marks durations where a pi "
                f"pulse would need more than {gcfg['max_amp']:g} of full scale; the dashed "
                f"line is pi/(r A_max) = {t_wall:.2f} ns, unreachable by any bounded shape.")
pl.save(fig, utils.FIGURES / "fig07_speed_limit.png")

utils.save("speed_limit", {"rows": rows, "gaussian_wall_ns": t_gauss,
                           "absolute_wall_ns": t_wall,
                           "sigma_ratio": sigma_ratio, "max_amp": gcfg["max_amp"]})
print(f"\nGaussian reaches full scale at T = {t_gauss:.2f} ns; "
      f"the bound for any shape is {t_wall:.2f} ns")
