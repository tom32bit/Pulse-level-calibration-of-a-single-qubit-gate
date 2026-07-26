"""Fine amplitude calibration by error amplification.

The Rabi fit of script 01 locates A_pi to a few parts in 10^3, which is a
0.1-degree-scale rotation error -- far above the 10^-4 gate errors the pulse
shape can otherwise reach.  The fix is not more shots but a longer sequence.
Prepare the state on the equator with an X90 and then apply the pi pulse under
test n times: every rotation is about the same axis, so the total angle is
pi/2 + n(pi + dtheta) and a per-gate over-rotation dtheta enters multiplied by
n.  The signal grows linearly in n while the projection noise does not, so the
uncertainty on dtheta falls like 1/n [Sheldon2016, Kelly2014,
QiskitExperiments2023].

Panel d is the same statement as a scaling law: sequence length buys precision
far more cheaply than shot count, until the accumulated error wraps and the
fit becomes ambiguous.

Outputs: ``figures/fig04_error_amplification.png``, ``results/fine_amplitude.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import experiments as ex, metrics as M, plotting as pl, sequences, utils
from pulsecal.propagate import trajectory

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
rng = utils.rng(cfg, "fine")
shots = cfg["measurement"]["shots"]
pl.use_style()

rabi = utils.load("rabi")
drag = utils.load("drag_calibration")
spec = spec.with_(amp=rabi["a_pi"], beta=drag["beta_measured_ns"])

n_max = 24
scales = np.array([0.985, 0.995, 1.0, 1.005, 1.015])
shown = np.array([0, 2, 4])                       # the traces drawn in panel a
ns, p1 = ex.fine_amplitude(dev, spec, scales, n_max, shots, rng)
_, p1_exact = ex.fine_amplitude(dev, spec, scales, n_max)
d_theta = np.array([ex.fit_overrotation(ns, p) for p in p1])

# Scaling the amplitude by s scales the rotation angle, so the fitted
# over-rotation is dtheta(s) = pi(s-1) + s*dtheta_0.  The intercept dtheta_0 is
# the residual miscalibration the Rabi fit left behind, and dividing it out
# gives the fine-calibrated amplitude.
slope, intercept = np.polyfit(scales - 1.0, d_theta, 1)
a_pi_fine = spec.amp * np.pi / (np.pi + intercept)

# --- precision versus sequence length ---------------------------------------
# One exact trace, resampled many times, isolates the shot-noise contribution
# to the fitted over-rotation from any systematic in the fit itself.
probe = 1.006
_, exact_probe = ex.fine_amplitude(dev, spec, np.array([probe]), n_max)
n_grid = np.array([3, 4, 6, 8, 12, 16, 20, 24])
trials = 60
spread = []
for n_cut in n_grid:
    draws = [ex.fit_overrotation(ns[:n_cut + 1],
                                 rng.binomial(shots, exact_probe[0, :n_cut + 1]) / shots)
             for _ in range(trials)]
    spread.append(np.std(draws))
spread = np.array(spread)

# The Rabi fit of script 01 in the same units: a fractional amplitude error
# maps to a rotation-angle error of pi times that fraction.
rabi_precision = np.pi * rabi["a_pi_err"] / rabi["a_pi"]

# --- Bloch walk -------------------------------------------------------------
# The continuous path, not just the state after each pulse: an over-rotated
# pulse overshoots the pole a little every time, and the path precesses.
walk_scale = 1.03
walk = spec.with_(amp=spec.amp * walk_scale)
psi0 = np.zeros(dev.n_levels, complex)
psi0[0] = 1.0
walk_seq = [sequences.gate(walk, "x")] + [walk] * 12
walk_bloch = M.bloch(trajectory(dev, sequences.envelope(walk_seq, dev.dt), psi0))

# --- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(8.4, 5.7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0], hspace=0.48, wspace=0.42)
ax_fan = fig.add_subplot(gs[0, 0])
ax_bloch = fig.add_subplot(gs[1, 0], projection="3d")
ax_slope = fig.add_subplot(gs[:, 1])
ax_prec = fig.add_subplot(gs[:, 2])

# The deviation alternates sign with n; rectifying it by (-1)^n turns the
# sawtooth into the fan that the fit actually sees.
rect = (-1.0) ** ns
for k, color in zip(shown, (pl.PLAIN, pl.INK, pl.DRAG)):
    ax_fan.plot(ns, 0.5 + rect * (p1_exact[k] - 0.5), color=color, lw=1.6)
    ax_fan.plot(ns, 0.5 + rect * (p1[k] - 0.5), ".", color=color, ms=4.5)
ax_fan.axhline(0.5, color=pl.MUTED, lw=0.9, ls=":")
ax_fan.set(xlabel=r"number of $\pi$ pulses  $n$", ylabel=r"$0.5+(-1)^n[P(|1\rangle)-0.5]$",
           ylim=(-0.03, 1.03))
ax_fan.set_title(r"a   $X_{90}$ then $n$ $\pi$ pulses")
for k, color, txt, y in ((4, pl.DRAG, "+1.5%", 0.92), (0, pl.PLAIN, "-1.5%", 0.08)):
    pl.annotate(ax_fan, txt, (ns[-1], 0.5 + rect[-1] * (p1_exact[k, -1] - 0.5)), (12.5, y),
                color=color)
pl.annotate(ax_fan, "calibrated", (ns[16], 0.5), (1.5, 0.72), color=pl.INK)

pl.bloch_frame(ax_bloch)
ax_bloch.plot(*walk_bloch.T, color=pl.DRAG, lw=1.1, alpha=0.9)
ax_bloch.scatter(*walk_bloch[-1], s=22, color=pl.DRAG, depthshade=False, zorder=6)
ax_bloch.view_init(elev=16, azim=-62)
ax_bloch.set_title(f"b   the walk at +{100 * (walk_scale - 1):.0f}%", y=0.97, fontsize=9.5)

ax_slope.plot(1e2 * (scales - 1), 1e3 * d_theta, "o", ms=6, color=pl.OPT, zorder=4)
grid = np.linspace(-0.019, 0.019, 50)
ax_slope.plot(1e2 * grid, 1e3 * (slope * grid + intercept), color=pl.INK, lw=1.4, ls=(0, (5, 3)))
ax_slope.axhline(0, color=pl.MUTED, lw=0.8)
ax_slope.axvline(0, color=pl.MUTED, lw=0.8)
ax_slope.plot(0, 1e3 * intercept, "o", ms=7, mfc=pl.SURFACE, mec=pl.INK, mew=1.4, zorder=5)
ax_slope.set(xlabel="applied amplitude error  (%)", ylabel=r"fitted $d\theta$  (mrad)")
ax_slope.set_title("c   slope and intercept")
pl.annotate(ax_slope, f"slope {slope:.3f}\n" + r"($\pi=3.142$)", (1.05, 1e3 * (slope * 0.0105)),
            (-1.85, 33.0), color=pl.INK)
pl.annotate(ax_slope, f"residual\n{1e3 * intercept:+.1f} mrad", (0, 1e3 * intercept),
            (0.25, -40.0), color=pl.OPT)

ax_prec.loglog(n_grid, spread * 1e3, "o-", color=pl.OPT, ms=5, lw=1.6)
ax_prec.loglog(n_grid, spread[0] * 1e3 * n_grid[0] / n_grid, color=pl.INK, lw=1.3,
               ls=(0, (5, 3)))
ax_prec.axhline(rabi_precision * 1e3, color=pl.PLAIN, lw=1.5)
ax_prec.set(xlabel=r"sequence length  $n$", ylabel=r"uncertainty on $d\theta$  (mrad)")
ax_prec.set_xticks([3, 6, 12, 24], ["3", "6", "12", "24"])
ax_prec.minorticks_off()
ax_prec.set_title("d   precision from length")
ax_prec.set_ylim(top=rabi_precision * 2.6e3)
pl.annotate(ax_prec, "Rabi fit (script 01)", (7.5, rabi_precision * 1e3),
            (6.2, rabi_precision * 1.55e3), color=pl.PLAIN)
pl.annotate(ax_prec, r"$\propto 1/n$", (n_grid[5], spread[0] * 1e3 * n_grid[0] / n_grid[5]),
            (9.0, spread[0] * 1e3 * 0.55), color=pl.INK)

pl.despine(ax_fan, ax_slope, ax_prec)
pl.caption(fig, f"{shots} shots per point, {trials} resamplings per length in panel d. "
                f"Panel b starts at |0>, applies X90 and {len(walk_seq) - 1} pi pulses "
                f"miscalibrated by "
                f"+{100 * (walk_scale - 1):.0f}%; the end point walks instead of returning.")
pl.save(fig, utils.FIGURES / "fig04_error_amplification.png")

err_before = M.gate_error(sequences.run(dev, [spec]), M.TARGETS["X"])
err_after = M.gate_error(sequences.run(dev, [spec.with_(amp=a_pi_fine)]), M.TARGETS["X"])

utils.save("fine_amplitude", {
    "scales": scales, "fitted_dtheta": d_theta,
    "slope": float(slope), "intercept_rad": float(intercept),
    "a_pi_rabi": spec.amp, "a_pi_fine": float(a_pi_fine),
    "gate_error_before": err_before, "gate_error_after": err_after,
    "n_grid": n_grid, "dtheta_uncertainty": spread,
    "rabi_dtheta_uncertainty": rabi_precision, "shots": shots, "trials": trials,
    "improvement_over_rabi": float(rabi_precision / spread[-1])})
print(f"slope {slope:.4f} (pi = {np.pi:.4f}), residual over-rotation "
      f"{1e3 * intercept:+.2f} mrad")
print(f"A_pi {spec.amp:.6f} -> {a_pi_fine:.6f};  gate error {err_before:.3e} -> {err_after:.3e}")
print(f"dtheta uncertainty: Rabi fit {1e3 * rabi_precision:.3f} mrad -> "
      f"n={n_grid[-1]} sequence {1e3 * spread[-1]:.3f} mrad "
      f"({rabi_precision / spread[-1]:.0f}x better)")
