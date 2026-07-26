"""What DRAG does, seen inside the pulse, and why one beta is not enough.

A resonant drive on a weakly anharmonic ladder does not only rotate
|0> <-> |1>: the same tone is only |alpha| away from |1> <-> |2>, so a pulse
whose bandwidth is comparable to alpha drives that transition too.  Population
makes a transient excursion onto |2> and, on returning, leaves behind both a
small residual leakage and a phase error on the qubit.

Adding a quadrature proportional to the derivative of the envelope,
eps_Q = -lambda * eps_I' / alpha, cancels the leading non-adiabatic |2>
amplitude [Motzoi2009].  The two damages it does are not cured by the same
lambda, though: the residual leakage is smallest near lambda = 1, while the
phase error, and hence the total gate error, is smallest near lambda = 1/2
[Gambetta2011].  Both minima appear in panel b, and the split is the reason
real calibrations tune beta against a gate-error-like signal rather than
against leakage, and then clean up what is left with a frame detuning
[Lucero2010, Chen2016].

Panel d makes the cancellation concrete: viewed in the frame of the |1>-|2>
transition, the |2> amplitude of a corrected pulse traces a loop that closes
back on the origin.

Outputs: ``figures/fig02_drag_mechanism.png``, ``results/drag_mechanism.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import metrics as M, plotting as pl, utils
from pulsecal.propagate import propagator, trajectory
from pulsecal.pulses import drag_envelope, lifted_gaussian

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
pl.use_style()

spec = spec.with_(amp=utils.load("rabi")["a_pi"])
beta_leak = dev.drag_beta_analytic()          # lambda = 1, cancels leakage
beta_phase = beta_leak / 2                    # lambda = 1/2, cancels phase error
t = (np.arange(spec.duration) + 0.5) * dev.dt

# --- beta scan --------------------------------------------------------------
betas = np.linspace(0.0, 0.9, 361)
u_scan = propagator(dev, np.stack([drag_envelope(spec.with_(beta=b), dev.dt) for b in betas]))
err_scan = np.array([M.gate_error(u, M.TARGETS["X"]) for u in u_scan])
leak_scan = np.array([M.leakage(u) for u in u_scan])
b_err, b_leak = betas[err_scan.argmin()], betas[leak_scan.argmin()]

# --- two representative pulses ---------------------------------------------
psi1 = np.zeros(dev.n_levels, complex)
psi1[1] = 1.0
runs = []
for beta, color in ((0.0, pl.PLAIN), (beta_leak, pl.DRAG)):
    env = drag_envelope(spec.with_(beta=beta), dev.dt)
    traj = trajectory(dev, env, psi1)
    # |2> amplitude in the frame of the 1->2 transition: a fully corrected
    # pulse returns it to the origin.
    runs.append({"beta": beta, "color": color, "env": env,
                 "phasor": traj[:, 2] * np.exp(1j * dev.alpha * np.append(0, t)),
                 "leaked": M.populations(traj)[:, 2:].sum(1),
                 "error": M.gate_error(propagator(dev, env), M.TARGETS["X"]),
                 "leakage": M.leakage(propagator(dev, env))})

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1))
(ax_env, ax_scan), (ax_leak, ax_ph) = axes
fig.subplots_adjust(hspace=0.44, wspace=0.26)

g, _ = lifted_gaussian(spec.duration, spec.sigma)
ax_env.plot(t, spec.amp * g, color=pl.INK, lw=2.0)
ax_env.plot(t, runs[1]["env"].imag, color=pl.DRAG, lw=2.0)
ax_env.fill_between(t, 0, runs[1]["env"].imag, color=pl.DRAG, alpha=0.13, lw=0)
ax_env.axhline(0, color=pl.MUTED, lw=0.7)
pl.annotate(ax_env, r"in phase  $\varepsilon_I$", (t[15], spec.amp * g[15]), (0.9, 0.212),
            color=pl.INK)
pl.annotate(ax_env, r"quadrature  $\beta\,\dot{\varepsilon}_I$",
            (t[10], runs[1]["env"].imag[10]), (4.6, 0.118), color=pl.DRAG)
ax_env.set(xlabel="time  (ns)", ylabel="envelope")
ax_env.set_title("a   the two quadratures")

ax_scan.semilogy(betas, err_scan, color=pl.INK, lw=2.0)
ax_scan.semilogy(betas, leak_scan, color=pl.DRAG, lw=2.0)
for b_val, lam in ((beta_phase, r"$\lambda=1/2$"), (beta_leak, r"$\lambda=1$")):
    ax_scan.axvline(b_val, color=pl.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax_scan.text(b_val, 1.35e-2, lam, ha="center", fontsize=8.5, color=pl.INK2)
for x, y, c in ((b_err, err_scan.min(), pl.INK), (b_leak, leak_scan.min(), pl.DRAG)):
    ax_scan.plot(x, y, "o", ms=6, mfc=c, mec=pl.SURFACE, mew=1.2, zorder=5)
ax_scan.set_ylim(8e-6, 3e-2)
ax_scan.set(xlabel=r"DRAG coefficient  $\beta$  (ns)", ylabel="error")
ax_scan.set_title(r"b   leakage and gate error want different $\beta$")
pl.annotate(ax_scan, f"gate error\nmin at {b_err:.3f} ns", (b_err, err_scan.min()),
            (0.44, 2.4e-4), color=pl.INK)
pl.annotate(ax_scan, f"leakage $L_1$\nmin at {b_leak:.3f} ns", (b_leak, leak_scan.min()),
            (0.545, 1.15e-5), color=pl.DRAG)

for r in runs:
    ax_leak.semilogy(np.append(0, t), r["leaked"], color=r["color"], lw=2.0)
    ax_leak.text(t[-1] + 0.3, r["leaked"][-1], f"  {r['leaked'][-1]:.0e}", va="center",
                 fontsize=8.5, color=r["color"])
ax_leak.set(xlabel="time  (ns)", ylabel="leaked population", ylim=(1e-6, 3e-1),
            xlim=(0, t[-1] + 2.6))
ax_leak.set_title(r"c   the excursion, starting from $|1\rangle$")
ax_leak.text(0.035, 0.955, r"transient peak is set by $\Omega/\alpha$;"
                           "\n" r"$\beta$ only changes what is left",
             transform=ax_leak.transAxes, va="top", fontsize=8.5, color=pl.INK2)
pl.annotate(ax_leak, r"$\beta=0$", (t[30], runs[0]["leaked"][31]), (7.4, 2.0e-3), color=pl.PLAIN)
pl.annotate(ax_leak, r"$\beta=-1/\alpha$", (t[34], runs[1]["leaked"][35]), (5.2, 1.1e-4),
            color=pl.DRAG)

for r in runs:
    ax_ph.plot(r["phasor"].real * 1e2, r["phasor"].imag * 1e2, color=r["color"], lw=1.7)
ax_ph.plot(0, 0, "+", color=pl.INK, ms=9, mew=1.4)
ax_ph.set_aspect("equal")
ax_ph.set_xlim(-17.5, 27.0)
ax_ph.set(xlabel=r"Re $c_2$  ($\times10^{-2}$)", ylabel=r"Im $c_2$  ($\times10^{-2}$)")
ax_ph.set_title(r"d   the $|2\rangle$ amplitude closing its loop")

zoom = ax_ph.inset_axes([0.635, 0.055, 0.345, 0.345])
span = 1.4 * max(abs(r["phasor"][-1]) for r in runs) * 1e2
for r in runs:
    zoom.plot(r["phasor"].real * 1e2, r["phasor"].imag * 1e2, color=r["color"], lw=1.4)
    zoom.plot(r["phasor"][-1].real * 1e2, r["phasor"][-1].imag * 1e2, "o", ms=5.5,
              mfc=r["color"], mec=pl.SURFACE, mew=1.1, zorder=5)
zoom.plot(0, 0, "+", color=pl.INK, ms=7, mew=1.2)
zoom.set(xlim=(-span, span), ylim=(-span, span), xticks=[], yticks=[])
zoom.set_aspect("equal")
zoom.grid(False)
for side in zoom.spines.values():
    side.set_color(pl.MUTED)
zoom.set_title("end points", fontsize=8, color=pl.INK2, pad=2)

pl.despine(*axes.ravel())
pl.caption(fig, f"All panels use the calibrated pi amplitude A = {spec.amp:.4f} at zero drive "
                f"detuning; alpha = {dev.anharmonicity * 1e3:.0f} MHz gives -1/alpha = "
                f"{beta_leak:.3f} ns. Panel b is noiseless.")
pl.save(fig, utils.FIGURES / "fig02_drag_mechanism.png")

utils.save("drag_mechanism", {
    "beta_analytic_leakage_ns": beta_leak, "beta_analytic_phase_ns": beta_phase,
    "beta_min_error_ns": float(b_err), "beta_min_leakage_ns": float(b_leak),
    "error_at_min": float(err_scan.min()), "leakage_at_min": float(leak_scan.min()),
    "betas": betas, "gate_error": err_scan, "leakage": leak_scan,
    "cases": [{"beta": r["beta"], "error": r["error"], "leakage": r["leakage"],
               "peak_leaked": float(r["leaked"].max()),
               "residual_leaked": float(r["leaked"][-1])} for r in runs]})
print(f"gate error minimised at beta = {b_err:.4f} ns  (lambda/2 theory {beta_phase:.4f})")
print(f"leakage    minimised at beta = {b_leak:.4f} ns  (lambda=1 theory {beta_leak:.4f})")
