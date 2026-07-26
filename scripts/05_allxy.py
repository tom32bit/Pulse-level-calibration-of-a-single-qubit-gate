"""AllXY: reading the error type off a staircase.

Twenty-one two-gate sequences drawn from {I, X90, Y90, X, Y}, ordered so that
an ideal qubit returns P(|1>) = 0 for the first five, 1/2 for the next twelve
and 1 for the last four.  The value of the diagnostic is not the number it
produces but the *shape* of the deviation: each miscalibration distorts the
staircase in its own recognisable way [Reed2013, Krantz2019].

* an amplitude error tilts the middle plateau and lifts the ends;
* a drive detuning breaks the plateau into a zig-zag between the pairs that
  end on x and those that end on y;
* a wrong DRAG coefficient leaves the plateau flat but splits the pairs that
  mix the two axes, which is why entries 16 and 17 (y-Y and Y-y) are the ones
  experimenters watch when tuning beta.

Because the sequences are only two gates long the test is coarse: it catches
percent-level errors quickly and is blind below about 10^-3, which is where
error amplification and randomized benchmarking take over [Kelly2014].

Outputs: ``figures/fig05_allxy.png``, ``results/allxy.json``.
"""

import _bootstrap  # noqa: F401

import numpy as np
from matplotlib import pyplot as plt

from pulsecal import experiments as ex, metrics as M, plotting as pl, sequences, utils

cfg = utils.load_config()
dev, spec = utils.build_device(cfg), utils.default_pulse(cfg)
rng = utils.rng(cfg, "allxy")
shots = cfg["measurement"]["shots"]
pl.use_style()

cal = utils.load("drag_calibration")
a_pi = utils.load("fine_amplitude")["a_pi_fine"]
tuned = spec.with_(amp=a_pi, beta=cal["beta_optimum_ns"])
det = cal["detuning_optimum_ghz"]

# The pi/2 pulse gets its own amplitude and beta at the same drive frequency:
# the rotation angle is not exactly linear in amplitude, so halving A_pi is not
# a calibrated X90, and every AllXY entry containing one says so.
half = ex.best_drag(dev, spec.with_(amp=0.5 * a_pi), "X90",
                    x0=(0.5 * a_pi, tuned.beta, 0.0), detuning=det)
halved = sequences.gate(tuned, "x")
err_half = M.gate_error(sequences.run(dev, [half["spec"]], det), M.TARGETS["X90"])
err_halved = M.gate_error(sequences.run(dev, [halved], det), M.TARGETS["X90"])

cases = [
    ("calibrated", tuned, half["spec"], det, pl.INK),
    (r"amplitude $+4\%$", tuned.with_(amp=a_pi * 1.04), half["spec"], det, pl.PLAIN),
    ("detuning $+8$ MHz", tuned, half["spec"], det + 0.008, pl.DRAG),
    (r"$\beta=0$", tuned.with_(beta=0.0), half["spec"].with_(beta=0.0), det, pl.OPT),
]
labels = [f"{a}-{b}" for a, b in sequences.ALLXY]
ideal = sequences.ALLXY_IDEAL
idx = np.arange(len(ideal))

runs = [(name, ex.allxy(dev, sp, shots, rng, d, half=h), ex.allxy(dev, sp, 0, None, d, half=h),
         color) for name, sp, h, d, color in cases]
rms_halved = float(np.sqrt(np.mean((ex.allxy(dev, tuned, 0, None, det) - ideal) ** 2)))

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.30, wspace=0.10)

for ax, (name, meas, exact, color) in zip(axes.ravel(), runs):
    ax.step(idx, ideal, where="mid", color=pl.MUTED, lw=1.2, ls=(0, (4, 2.5)))
    ax.fill_between(idx, ideal, exact, step="mid", color=color, alpha=0.18, lw=0)
    ax.plot(idx, exact, color=color, lw=1.5, drawstyle="steps-mid")
    ax.plot(idx, meas, "o", ms=3.6, color=color)
    rms = float(np.sqrt(np.mean((exact - ideal) ** 2)))
    ax.set_title(f"{'abcd'[list(axes.ravel()).index(ax)]}   {name}", color=color)
    ax.text(0.985, 0.06, f"rms deviation {rms:.4f}", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=pl.INK2)
    ax.set_ylim(-0.09, 1.09)
    ax.grid(axis="x", visible=False)

for ax in axes[1]:
    ax.set_xticks(idx, labels, rotation=90, fontsize=6.8)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$P(|1\rangle)$")

# Mark the three plateaus once, on the first panel.
for lo, hi, txt in ((0, 4, "0"), (5, 16, "1/2"), (17, 20, "1")):
    axes[0, 0].annotate("", xy=(lo - 0.4, 1.03), xytext=(hi + 0.4, 1.03),
                        arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2",
                                        color=pl.MUTED, lw=0.7))
    axes[0, 0].text((lo + hi) / 2, 0.93, txt, ha="center", fontsize=8, color=pl.INK2)

pl.despine(*axes.ravel())
pl.caption(fig, f"Dashed grey is the ideal staircase; the filled band is the exact deviation and "
                f"the dots are {shots} shots per sequence. X uses A = {a_pi:.4f}, beta = "
                f"{cal['beta_optimum_ns']:.4f} ns; X90 is calibrated separately at A = "
                f"{half['amp']:.4f}, beta = {half['beta']:.4f} ns. Reusing half the pi amplitude "
                f"instead would raise the rms of panel a to {rms_halved:.4f}.")
pl.save(fig, utils.FIGURES / "fig05_allxy.png")

utils.save("allxy", {
    "labels": labels, "ideal": ideal,
    "x90_calibrated": {"amp": half["amp"], "beta": half["beta"], "error": err_half},
    "x90_from_halving": {"amp": halved.amp, "beta": halved.beta, "error": err_halved},
    "rms_calibrated_x90": float(np.sqrt(np.mean((runs[0][2] - ideal) ** 2))),
    "rms_halved_x90": rms_halved,
    "cases": [{"name": name, "exact": exact, "measured": meas,
               "rms": float(np.sqrt(np.mean((exact - ideal) ** 2)))}
              for name, meas, exact, _ in runs]})
for name, _, exact, _ in runs:
    print(f"{name:22s} rms deviation {np.sqrt(np.mean((exact - ideal) ** 2)):.5f}")
print(f"\nX90 calibrated separately : error {err_half:.3e} (A = {half['amp']:.5f}, "
      f"beta = {half['beta']:.4f} ns)")
print(f"X90 from halving A_pi     : error {err_halved:.3e} (A = {halved.amp:.5f})")
print(f"AllXY rms with each        : {np.sqrt(np.mean((runs[0][2] - ideal) ** 2)):.5f} "
      f"vs {rms_halved:.5f}")
