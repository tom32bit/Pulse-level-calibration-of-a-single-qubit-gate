"""Calibrating DRAG: two knobs, one valley.

Once the amplitude is set, a DRAG pulse still has two free parameters: the
derivative coefficient beta and the detuning of the drive.  They are not
independent.  Beta cancels the non-adiabatic |2> amplitude but shifts the
phase; a detuning shifts the phase but does nothing about leakage.  The gate
error therefore sees a valley running diagonally across the (beta, detuning)
plane rather than a bowl, so the two have to be tuned together [Motzoi2009,
Gambetta2011, Lucero2010, Chen2016].

One caveat decides the scale of that plane.  A drive parked off nu_q makes the
frame it defines precess against the qubit, and control software already
removes that by advancing the phase of later pulses [McKay2017].  Charging the
pulse for it instead would make the optimiser spend beta on cancelling a
bookkeeping term; with the frame tracked, the valley is only a few MHz wide
and its floor sits essentially at zero detuning, leaving beta to do the work.

Panel b is the part an experiment measures cheaply: [X90, X(-90)] is the
identity for a correctly corrected pulse, so repeating it n times amplifies the
residual error linearly in n while the readout noise stays put, and the minimum
sharpens accordingly [Chen2016, Sheldon2016].  It converges on the analytic
lambda = 1/2 value, not the lambda = 1 that minimises leakage.

Outputs: ``figures/fig03_drag_landscape.png``, ``results/drag_calibration.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from pulsecal import experiments as ex, metrics as M, plotting as pl, utils
from pulsecal.propagate import gate_propagator
from pulsecal.pulses import drag_envelope

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
rng = utils.rng(cfg, "drag")
shots = cfg["measurement"]["shots"]
pl.use_style()

a_pi = utils.load("rabi")["a_pi"]
spec = spec.with_(amp=a_pi)
beta_leak = dev.drag_beta_analytic()

# --- a: exact landscape -----------------------------------------------------
betas = np.linspace(0.0, 0.85, 161)
dets = np.linspace(-0.008, 0.008, 161)                     # GHz
err_map, leak_map = ex.drag_landscape(dev, spec, betas, dets)
best = ex.best_drag(dev, spec, x0=(a_pi, beta_leak / 2, 0.0))
# Valley floors: the best detuning at each beta. Mask the columns whose
# minimum falls on the edge of the scanned window, where it is not a minimum.
def floor(grid):
    idx = grid.argmin(axis=0)
    inside = (idx > 0) & (idx < len(dets) - 1)
    return np.where(inside, dets[idx], np.nan)


err_floor, leak_floor = floor(err_map), floor(leak_map)

# --- b: repeated-pair beta scan, with shots ---------------------------------
betas_m = np.linspace(0.02, 0.60, 61)
reps = (1, 4, 12)
scans = {n: ex.drag_repeat(dev, spec, betas_m, n, shots, rng) for n in reps}

# The signal is quadratic in the beta error near its minimum; fit the lowest
# points of the longest sequence, where the amplification is largest.
p_long = scans[reps[-1]]
window = p_long <= np.sort(p_long)[len(p_long) // 4]
coef, cov = np.polyfit(betas_m[window], p_long[window], 2, cov=True)
beta_meas = -coef[1] / (2 * coef[0])
beta_err = abs(beta_meas) * np.sqrt(cov[1, 1] / coef[1] ** 2 + cov[0, 0] / coef[0] ** 2)

# --- c: the calibration chain -----------------------------------------------
def err_of(amp, beta, det):
    return M.gate_error(gate_propagator(dev, drag_envelope(spec.with_(amp=amp, beta=beta), dev.dt),
                                        det), M.TARGETS["X"])


chain = [("nominal $A$", err_of(cfg["pulse"]["amp_guess"], 0.0, 0.0), pl.PLAIN),
         (r"Rabi $A_\pi$", err_of(a_pi, 0.0, 0.0), pl.PLAIN),
         (r"$+\ \beta=-1/2\alpha$", err_of(a_pi, beta_leak / 2, 0.0), pl.DRAG),
         (r"$+\ $measured $\beta$", err_of(a_pi, beta_meas, 0.0), pl.DRAG),
         (r"$+\ $joint $(\beta,\delta)$", best["error"], pl.INK)]

# --- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(8.5, 5.8))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.85],
                      hspace=0.52, wspace=0.62)
ax_map = fig.add_subplot(gs[:, 0])
ax_scan = fig.add_subplot(gs[0, 1])
ax_chain = fig.add_subplot(gs[1, 1])

mesh = ax_map.pcolormesh(betas, dets * 1e3, err_map, cmap=pl.SEQ_R,
                         norm=LogNorm(best["error"], 3e-2), shading="gouraud", rasterized=True)
ax_map.contour(betas, dets * 1e3, err_map, levels=np.logspace(-4.2, -1.6, 8),
               colors=pl.SURFACE, linewidths=0.5, alpha=0.45)
ax_map.plot(betas, err_floor * 1e3, color=pl.SURFACE, lw=1.6)
ax_map.plot(betas, leak_floor * 1e3, color=pl.DRAG, lw=1.5, ls=(0, (4, 2.5)))
ax_map.plot(best["beta"], best["detuning"] * 1e3, "o", ms=7.5, mfc=pl.SURFACE, mec=pl.INK,
            mew=1.5, zorder=6)
ax_map.plot(beta_meas, 0.0, "s", ms=7.5, mfc="none", mec=pl.OPT, mew=1.9, zorder=6)
pl.annotate(ax_map, "error valley floor", (0.70, err_floor[132] * 1e3), (0.34, -6.3),
            color=pl.INK)
pl.annotate(ax_map, "minimum\nleakage", (0.48, leak_floor[91] * 1e3), (0.60, -5.0), color=pl.DRAG)
pl.annotate(ax_map, f"joint optimum\n{best['error']:.1e}", (best["beta"], best["detuning"] * 1e3),
            (0.36, 6.0), color=pl.INK)
pl.annotate(ax_map, f"panel b\n{err_of(a_pi, beta_meas, 0.0):.1e}", (beta_meas, 0.0),
            (0.03, -5.6), color=pl.OPT)
ax_map.set(xlabel=r"DRAG coefficient  $\beta$  (ns)", ylabel="drive detuning  (MHz)")
ax_map.set_title("a   the gate-error valley")
ax_map.grid(False)
cb = fig.colorbar(mesh, ax=ax_map, pad=0.02, fraction=0.045)
cb.set_label("gate error", fontsize=8.5, color=pl.INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, color=pl.INK2, length=2)

for n, shade in zip(reps, (0.30, 0.60, 1.0)):
    ax_scan.plot(betas_m, scans[n], color=pl.PLAIN, lw=1.7, alpha=shade)
    ax_scan.text(betas_m[-1] + 0.012, scans[n][-1], f"n={n}", fontsize=8.5, va="center",
                 color=pl.PLAIN, alpha=max(shade, 0.55))
fine = np.linspace(betas_m[window].min(), betas_m[window].max(), 100)
ax_scan.plot(fine, np.polyval(coef, fine), color=pl.OPT, lw=1.8)
ax_scan.axvline(beta_meas, color=pl.OPT, lw=1.0, ls=":")
ax_scan.axvline(beta_leak / 2, color=pl.MUTED, lw=0.9, ls=(0, (4, 3)))
ax_scan.set(xlabel=r"$\beta$  (ns)", ylabel=r"$P(|1\rangle)$", xlim=(0, 0.70))
ax_scan.set_title(r"b   $[X_{90},X_{-90}]^n$ sharpens the minimum")
pl.annotate(ax_scan, rf"$\beta = {beta_meas:.4f}$" + "\n" + rf"$\pm\,{beta_err:.4f}$ ns",
            (beta_meas, 0.015), (0.025, 0.24), color=pl.OPT)
ax_scan.text(beta_leak / 2 + 0.014, 0.97, r"$-1/2\alpha$", fontsize=8.5, color=pl.INK2,
             transform=ax_scan.get_xaxis_transform(), va="top")

labels, values, colors = zip(*chain)
ax_chain.barh(np.arange(len(values)), values, color=colors, height=0.6)
for k, v in enumerate(values):
    ax_chain.text(v * 1.4, k, f"{v:.1e}", va="center", fontsize=8, color=pl.INK2)
ax_chain.set_xscale("log")
ax_chain.set_xlim(best["error"] * 0.5, 2.0)
ax_chain.set_yticks(np.arange(len(values)), labels, fontsize=8.5)
ax_chain.invert_yaxis()
ax_chain.set_xlabel("gate error")
ax_chain.set_title("c   what each knob buys")
ax_chain.grid(axis="y", visible=False)

pl.despine(ax_scan, ax_chain)
pl.caption(fig, f"Panel a is {len(betas)}x{len(dets)} exact propagators; panel b uses "
                f"{shots} shots per point. Gate errors are evaluated in the qubit frame, so a "
                "detuning is charged only for what it does during the pulse.")
pl.save(fig, utils.FIGURES / "fig03_drag_landscape.png")

utils.save("drag_calibration", {
    "beta_optimum_ns": best["beta"], "detuning_optimum_ghz": best["detuning"],
    "error_optimum": best["error"], "leakage_optimum": best["leakage"],
    "beta_measured_ns": float(beta_meas), "beta_measured_err_ns": float(beta_err),
    "beta_analytic_phase_ns": beta_leak / 2,
    "error_at_measured_beta": err_of(a_pi, beta_meas, 0.0),
    "repetitions": list(reps), "shots": shots,
    "chain": [{"stage": lab, "error": val} for lab, val, _ in chain]})
print(f"measured beta  : {beta_meas:.4f} +/- {beta_err:.4f} ns "
      f"(analytic -1/2alpha = {beta_leak / 2:.4f})")
print(f"joint optimum  : beta {best['beta']:.4f} ns, detuning {best['detuning'] * 1e3:+.2f} MHz")
print(f"gate error     : {err_of(a_pi, beta_meas, 0.0):.3e} -> {best['error']:.3e}")
