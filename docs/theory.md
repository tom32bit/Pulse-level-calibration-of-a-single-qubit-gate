# Derivations behind the code

Conventions throughout: angular frequencies in rad/ns, ordinary frequencies in
GHz, times in ns, `a` the annihilation operator truncated to `n_levels`, and
`'` for Hermitian conjugation. Citation keys refer to `references.bib`.

## 1. The model

A transmon is a Cooper-pair box biased so that `E_J >> E_C`; expanding its
cosine potential to quartic order gives an oscillator with a small negative
anharmonicity `alpha ~ -E_C` [Koch2007, Blais2021]. Truncated and driven on one
microwave line,

```text
H(t)/hbar = w_q a'a + (alpha/2) a'a'aa + r Re[eps(t) e^{-i w_d t}] (a + a')
```

The drive is a carrier at `w_d` shaped by a slowly varying complex envelope
`eps(t) = eps_I + i eps_Q`, which is what an IQ mixer physically produces.

Moving to the frame rotating at `w_d` and dropping terms oscillating at
`w_q + w_d` (the rotating-wave approximation) gives the model every experiment
in this repository uses:

```text
H(t)/hbar = sum_j [ j*Delta + (alpha/2) j(j-1) ] |j><j| + (r/2)( eps a' + eps* a )
Delta = w_q - w_d
```

Restricted to `{|0>, |1>}` the drive term is `(r/2)(eps_I sx - eps_Q sy)`, so a
real positive envelope rotates about `+x` and the quadrature rotates about
`-y`. `Delta` tilts the axis out of the equatorial plane.

Two consequences are used repeatedly.

* The Hamiltonian depends on `r` and `eps` only through their product, so the
  drive strength sets how much DAC range a calibrated pulse consumes and
  nothing else. `tests/test_pulsecal.py::test_drive_strength_only_enters_through_its_product_with_the_envelope`
  checks this.
* Three levels is the minimum that exposes leakage, and suffices for the
  calibrated DRAG pulse (0.3% from converged). It does not suffice for a pulse
  optimised to `1e-9`, which reports `1e-6` when re-simulated with three levels.
  Four is used throughout (`results/validation.json`, `results/grape.json`).

Discretisation is exact rather than approximate: an AWG emits a staircase, so
the propagator of a sampled pulse is exactly a product of matrix exponentials,
`U = U_{K-1} ... U_0` with `U_k = exp(-i H_k dt)`.

## 2. The pi amplitude

For a resonant drive with `eps_Q = 0`, the two-level rotation angle is the area

```text
theta = r * integral eps_I(t) dt
```

so `A_pi = 1 / (2 r * area_1)` with `area_1` the integral of the unit-amplitude
envelope. The lifted Gaussian used here,

```text
g(t) = A ( exp(-(t-t0)^2 / 2 s^2) - c ) / (1 - c),   c = exp(-(t0+1)^2 / 2 s^2)
```

is shifted so it vanishes one sample outside the pulse, matching
`qiskit.pulse.library.Gaussian` exactly. A hard edge would radiate broadband
and drive the leakage transition directly.

The measured `A_pi` sits about 0.3% above this two-level prediction: the real
system is not two-level, and the Stark shift of section 4 slightly changes the
effective rotation rate.

## 3. The chevron

Off resonance the Bloch vector precesses about a tilted axis at the generalised
Rabi rate `sqrt(Omega^2 + delta^2)`, so a fixed-length pulse reaches its first
population maximum when

```text
sqrt( theta^2 + (2 pi delta T)^2 ) = pi
```

an ellipse in the (detuning, amplitude) plane. Successive fringes are the same
condition at `m*pi`. Those ellipses are the dashed curves in `fig01`, and the
pattern they trace is why the measurement is called a chevron.

## 4. The ac Stark shift

A strong drive shifts the transition it is driving. To second order in
`Omega/alpha` [Gambetta2011],

```text
delta_stark(t) = (lambda^2 - 4) Omega(t)^2 / (4 alpha),   Omega = r eps_I
```

with `lambda` the DRAG coefficient in units of `-1/alpha` (so `lambda = 0` is
an uncorrected Gaussian). Since `alpha < 0`, an uncorrected pulse pushes the
transition *up*, and the chevron apex is displaced accordingly. Measured
`+7.7 MHz` against a pulse-averaged prediction of `+9.1 MHz` (`results/rabi.json`)
-- the right size and sign, with the gap explained by the second-order
expansion being applied at `Omega/|alpha| ~ 0.26`.

## 5. DRAG

Write the drive in the adiabatic basis of the driven three-level system. The
leading non-adiabatic coupling to `|2>` is proportional to the *rate of change*
of the drive, so cancelling it needs a control proportional to that rate. Adding
it in the orthogonal quadrature leaves the `|0> <-> |1>` rotation angle
untouched [Motzoi2009]:

```text
eps_Q(t) = -lambda * d(eps_I)/dt / alpha,   beta = -lambda/alpha  [ns]
```

The two things a leakage excursion damages are not repaired by the same
`lambda`.

* `lambda = 1` minimises the population left on `|2>`.
* `lambda = 1/2` minimises the phase error, and therefore the gate error, once
  the Stark term of section 4 is included [Gambetta2011, Lucero2010].

Both minima appear in `fig02b`: leakage bottoms out at `beta = 0.408 ns` against
a prediction of `-1/alpha = 0.482 ns`, and the gate error at `beta = 0.248 ns`
against `-1/2alpha = 0.241 ns`. The leading-order theory is accurate to a few
percent for the quantity it was derived for and off by 15% for the other, which
is what one should expect from a first-order result used at
`Omega/|alpha| ~ 0.26`.

The relation to Qiskit's parametrisation is a convention detail worth stating.
`qiskit.pulse.library.Drag` is defined as
`g_lifted(n) (1 - i beta_q (n-n0)/sigma^2)`, applying the derivative *factor* to
the lifted envelope rather than differentiating it. Matching the coefficient
gives `beta_q = beta / dt`; the two waveforms then differ only by the lifting
offset, checked at the few-percent level in `test_qiskit_drag_template_is_close`.

## 6. Frames, and what a detuning is allowed to be charged for

A drive parked at `nu_q + delta` defines a frame that precesses against the
qubit. Its propagator therefore carries `exp(-i 2 pi delta j T)` on level `j`
purely as bookkeeping, and control software removes it by advancing the phase of
subsequent pulses -- which is exactly what a virtual-Z gate is [McKay2017].

Two places in the code depend on getting this right.

* `propagate.gate_propagator` strips that phase before any comparison with a
  target. Without it the optimiser is charged for a frame choice and will spend
  real pulse parameters cancelling it: the apparent optimum moves to
  `-5.6 MHz` and the DRAG coefficient inflates to `1.4 ns`, both artefacts.
* `sequences.envelope` advances each pulse's carrier phase by `2 pi delta t`.
  Without it a two-gate sequence at a few MHz of detuning describes a different
  circuit, and the AllXY staircase reports errors that are not there.

With both in place the gate-error valley in the `(beta, detuning)` plane is a
few MHz wide and its floor passes through zero detuning, leaving `beta` to do
the work.

## 7. Fidelity when population can leave

The propagator is `n_levels x n_levels`; the gate an algorithm sees is its
computational block `V = P U P'`, which is sub-unitary whenever population
escapes. For such a trace-non-increasing map the average gate fidelity
generalises to [Nielsen2002, Wood2018]

```text
F_avg = ( Tr(M'M) + |Tr M|^2 ) / ( d(d+1) ),   M = W' V,  d = 2
```

which reduces to the familiar expression for unitary `V` and charges leakage
automatically. The average leakage out of the subspace is
`L1 = 1 - Tr(V'V)/d`.

Setting `z_free=True` maximises `F_avg` over a virtual Z analytically:
`Tr(W' Rz(phi) V) = e^{-i phi/2} B00 + e^{i phi/2} B11` with `B = V W'`, whose
modulus peaks at `|B00| + |B11|`, while `Tr(M'M) = Tr(V'V)` does not depend on
`phi`.

## 8. Error amplification

Prepare on the equator with an `X90`, then apply the pi pulse under test `n`
times. Every rotation shares the `x` axis, so the angles simply add:

```text
theta(n) = pi/2 + n(pi + dtheta),   P(|1>) = sin^2(theta/2)
```

A per-gate over-rotation enters multiplied by `n` while projection noise does
not, so the uncertainty on `dtheta` falls like `1/n` until the accumulated
angle wraps and the fit becomes ambiguous. Scaling the amplitude by `s` scales
the angle, giving

```text
dtheta(s) = pi (s - 1) + s * dtheta_0
```

whose slope is a consistency check on the fit (measured 3.106 against `pi`) and
whose intercept is the residual the Rabi fit left behind (`-6.7 mrad`, inside
that fit's own `9.3 mrad` uncertainty). Dividing it out gives the fine-calibrated
amplitude.

## 9. Exact GRAPE gradients

GRAPE ascends `F_avg` in the `2K` control samples [Khaneja2005]. With
`U = U_{K-1} ... U_0`, the chain rule gives

```text
dJ/du_k = 2 Re Tr[ M_k dU_k/du_k ],   M_k = (U_{k-1}...U_0) G' (U_{K-1}...U_{k+1})
```

where `G = dJ/dU*`. For `F_avg` above, `G = (V + Tr(M) W) / (d(d+1))` on the
computational block.

The original algorithm approximates `dU_k/du ~ -i dt H_ctrl U_k`, valid only for
small `dt`, which caps the usable step size. The exact derivative follows from
the Frechet derivative of the matrix exponential in the instantaneous eigenbasis
[Machnes2011]: with `H_k = V diag(w) V'`,

```text
dU_k/du = V [ F o (V' H_ctrl V) ] V',
F_mn = ( e^{-i w_m dt} - e^{-i w_n dt} ) / ( w_m - w_n ),  F_mm = -i dt e^{-i w_m dt}
```

Because every `H_k` is Hermitian and 4x4, one batched `eigh` supplies both the
propagators and their derivatives. Using exact gradients is what lets L-BFGS-B
[deFouquieres2011] run at its natural rate;
`test_analytic_gradient_matches_finite_differences` holds them to `1e-6`
relative.

The objective adds a discrete-Laplacian roughness penalty evaluated with virtual
zeros outside the pulse. That single term does two jobs: it band-limits the
waveform and it pulls the ends to zero, so no separate boundary constraint is
needed.

## 10. Two speed limits

Sweeping the gate length, two limits close in from the short side and they are
not the same limit.

* A Gaussian of peak amplitude `A_max` runs out of area at `T = 2.51 ns`. That
  ends the *ansatz*, not the gate.
* A rectangle carries the most area any bounded pulse can, so
  `T = pi / (r A_max) = 1.43 ns` is where an X gate becomes impossible for any
  shape at all.

The factor of 1.75 between them is room only a non-Gaussian pulse can use.

The second bound is a two-level statement and is exact as such: the geodesic
from the identity to `Rx(pi)` in SU(2) has length `pi/2`, a path travels it at
half the eigenvalue splitting `sqrt((r|eps|)^2 + Delta^2)`, and a detuning
increases that speed only in a direction that does not help, so the fastest
route is `Delta = 0`, `|eps| = A_max`. Extending it to the full ladder assumes
that routing amplitude through `|2>` cannot produce a subspace-preserving gate
faster, which is physically clear but not proved here.
