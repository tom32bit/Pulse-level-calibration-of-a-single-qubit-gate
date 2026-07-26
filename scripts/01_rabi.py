"""Amplitude and frequency calibration: the Rabi chevron.

Sweeping drive amplitude at fixed pulse length gives a sinusoid whose first
maximum defines the pi-pulse amplitude.  Repeating the sweep against drive
frequency turns that single trace into a chevron: off resonance the Bloch
vector precesses about a tilted axis at the generalised Rabi rate
sqrt(Omega^2 + delta^2), so more amplitude is needed for the same excitation
and the fringes bend outwards.  The apex of the chevron is the first thing a
calibration chain measures [Krantz2019, QiskitExperiments2023].

It does not, however, sit at nu_q.  A strong drive ac-Stark shifts the
transition it is driving, by delta = (lam^2 - 4) Omega^2 / (4 alpha) with
lam = 0 for an uncorrected Gaussian [Gambetta2011], and it is the shifted
frequency that the chevron finds.  That same shift returns in the next script
as the frame-detuning axis of the DRAG calibration.

Outputs: ``figures/fig01_rabi_chevron.png``, ``results/rabi.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import experiments as ex, plotting as pl, utils
from pulsecal.pulses import area, lifted_gaussian

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
rng = utils.rng(cfg, "rabi")
shots = cfg["measurement"]["shots"]
pl.use_style()

t_gate = spec.duration * dev.dt
unit_area = area(spec.with_(amp=1.0), dev.dt)
amps = np.linspace(0.0, 1.0, 241)                  # full DAC range
detunings = np.linspace(-0.12, 0.12, 281)          # GHz

# --- data ------------------------------------------------------------------
grid = ex.chevron(dev, spec, amps, detunings)
cut_amp = ex.rabi_amplitude(dev, spec, amps, shots, rng)[:, 1]
fit = ex.fit_rabi(amps, cut_amp)
a_pi = fit["a_pi"]

cut_det = ex.chevron(dev, spec.with_(amp=a_pi), np.array([a_pi]), detunings)[:, 0]
cut_det_meas = ex.sample(np.stack([1 - cut_det, cut_det], -1), shots, rng)[:, 1]

apex = float(detunings[cut_det.argmax()])
shape = a_pi * lifted_gaussian(spec.duration, spec.sigma)[0]
stark = dev.stark_detuning_analytic(shape, lam=0.0)

# Analytic fringe loci: the generalised rotation angle sqrt(theta^2 +
# (2 pi delta T)^2) reaches m*pi, with theta = r * integral(eps) the on-resonance
# rotation angle.  These are ellipses in (detuning, amplitude) and are what give
# the pattern its name.
fringes = []
for m in (1, 2):
    d_edge = m / (2 * t_gate)
    d_line = np.linspace(-d_edge, d_edge, 400)[1:-1]
    fringes.append((d_line, np.sqrt((m * np.pi) ** 2 - (2 * np.pi * d_line * t_gate) ** 2)
                    / (dev.r * unit_area)))

# --- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(7.9, 5.9))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[2.85, 1.15],
                      hspace=0.34, wspace=0.07)
ax_map = fig.add_subplot(gs[0, 0])
ax_amp = fig.add_subplot(gs[0, 1], sharey=ax_map)
ax_det = fig.add_subplot(gs[1, 0], sharex=ax_map)
ax_key = fig.add_subplot(gs[1, 1])

mesh = ax_map.pcolormesh(detunings * 1e3, amps, grid.T, cmap=pl.SEQ, vmin=0, vmax=1,
                         shading="gouraud", rasterized=True)
for d_line, a_line in fringes:
    ax_map.plot(d_line * 1e3, a_line, color=pl.SURFACE, lw=1.3, ls=(0, (4, 2.6)), alpha=0.9)
ax_map.axvline(0, color=pl.SURFACE, lw=0.7, alpha=0.45)
ax_map.plot(apex * 1e3, a_pi, marker="o", ms=6, mfc=pl.DRAG, mec=pl.SURFACE, mew=1.2, zorder=5)
ax_map.axhline(a_pi, color=pl.DRAG, lw=0.8, ls=":", alpha=0.85)
pl.annotate(ax_map, r"$\sqrt{\Omega^2+\delta^2}\,T=\pi,\;2\pi$",
            (66, 0.60), (16, 0.90), color=pl.SURFACE)
pl.annotate(ax_map, r"$A_\pi=$" + f"{a_pi:.4f}", (apex * 1e3, a_pi), (-116, 0.40), color=pl.INK)
ax_map.set_ylabel("envelope amplitude  (DAC full scale)")
ax_map.set_title("a   Rabi chevron")
ax_map.tick_params(labelbottom=False)
ax_map.grid(False)
ax_map.set_ylim(0, 1)

ax_amp.plot(cut_amp, amps, ".", color=pl.MUTED, ms=3, alpha=0.7)
ax_amp.plot(ex._rabi_model(amps, fit["offset"], -0.5 * fit["contrast"],
                           fit["freq"], fit["phase"]), amps, color=pl.DRAG, lw=1.8)
ax_amp.axhline(a_pi, color=pl.DRAG, lw=0.8, ls=":")
ax_amp.set_xlim(-0.06, 1.12)
ax_amp.set_xlabel(r"$P(|1\rangle)$")
ax_amp.tick_params(labelleft=False)
ax_amp.set_title(f"b   on-resonance cut\n     {shots} shots/point", fontsize=9.5)

ax_det.plot(detunings * 1e3, cut_det_meas, ".", color=pl.MUTED, ms=3, alpha=0.7)
ax_det.plot(detunings * 1e3, cut_det, color=pl.PLAIN, lw=1.8)
ax_det.axvline(0, color=pl.MUTED, lw=0.9, ls=":")
ax_det.axvline(apex * 1e3, color=pl.DRAG, lw=1.1)
pl.annotate(ax_det, f"apex {apex * 1e3:+.1f} MHz\nStark theory {stark * 1e3:+.1f} MHz",
            (apex * 1e3, 0.52), (-116, 0.60))
ax_det.set_xlabel(r"drive detuning  $\nu_d-\nu_q$  (MHz)")
ax_det.set_ylabel(r"$P(|1\rangle)$")
ax_det.set_title(r"c   cut at $A_\pi$: the apex is Stark-shifted off $\nu_q$", fontsize=9.5)
ax_det.set_ylim(-0.05, 1.12)

ax_key.axis("off")
cax = ax_key.inset_axes([0.16, 0.78, 0.70, 0.11])
cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
cb.set_label(r"$P(|1\rangle)$ in panel a", fontsize=8.5, color=pl.INK2, labelpad=3)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8, color=pl.INK2, length=2)
ax_key.text(0.16, 0.48, f"$A_\\pi$ ={a_pi:.4f} $\\pm$ {fit['a_pi_err']:.4f}\n"
                        f"contrast = {fit['contrast']:.3f}\n"
                        f"gate length = {t_gate:.2f} ns",
            transform=ax_key.transAxes, va="top", fontsize=8.5, color=pl.INK2)

pl.despine(ax_amp, ax_det)
pl.caption(fig, f"Transmon: nu_q = {dev.qubit_frequency} GHz, alpha = {dev.anharmonicity * 1e3:.0f} MHz, "
                f"{dev.n_levels} levels; lifted-Gaussian pulse, {t_gate:.2f} ns, sigma = T/4. "
                "Panel a noiseless; b and c sampled.")
pl.save(fig, utils.FIGURES / "fig01_rabi_chevron.png")

utils.save("rabi", {"a_pi": a_pi, "a_pi_err": fit["a_pi_err"], "contrast": fit["contrast"],
                    "a_pi_analytic": 1.0 / (2 * dev.drive_strength * unit_area),
                    "chevron_apex_ghz": apex, "stark_analytic_ghz": stark,
                    "gate_length_ns": t_gate, "shots": shots,
                    "amplitudes": amps, "p1_measured": cut_amp,
                    "detunings_ghz": detunings, "p1_vs_detuning": cut_det})
print(f"A_pi = {a_pi:.6f} +/- {fit['a_pi_err']:.6f} "
      f"(analytic {1.0 / (2 * dev.drive_strength * unit_area):.6f})")
print(f"chevron apex {apex * 1e3:+.2f} MHz, Stark prediction {stark * 1e3:+.2f} MHz")
