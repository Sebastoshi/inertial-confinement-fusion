# Rayleigh–Taylor instability — why ICF is hard

The 1-D models make ignition look like a matter of hitting the right velocity
and areal density. Reality intervenes: the imploding shell is **Rayleigh–Taylor
unstable**. A light fluid pushing a heavy one (ablated plasma pushing the cold
shell; later the hot spot pushing the fuel) amplifies any ripple exponentially,
and left unchecked it shreds the shell before it can stagnate. This is the single
effect a 1-D code *cannot* show, and the one that most limits real implosions.

Two scripts: the mechanics (linear + weakly-nonlinear theory) and a real 2D
simulation that grows a mushroom and validates the theory.

## `rt_mechanics.py` — the design constraints

```bash
python3 rt_mechanics.py
```

![RT mechanics](rt_mechanics.png)

- **Dispersion relation** — classical `γ = √(A g k)` grows without bound at short
  wavelength; the **ablative (Takabe)** form `γ = α√(A g k) − β k V_ablation`
  cuts off short wavelengths and picks a single most-dangerous mode (~41 µm,
  mode ℓ ≈ 150 here).
- **The punchline** — over the ~1.85 ns implosion, a 5 nm surface ripple grows
  **×72** under ablative RT (to 0.36 µm — safely under the 40 µm shell). Without
  ablative stabilization the *same* ripple would grow **×13000** (66 µm) and
  **break the shell up**. Ablation is what holds ICF capsules together.
- **Nonlinear saturation** — exponential growth until ~0.1λ, then the bubble
  rises at a terminal velocity.
- **The design lever** — peak growth rate `∝ 1/V_ablation`: faster ablation gives
  a smoother implosion, at the cost of burning away more shell (the rocket-model
  trade).

## `rt_2d.py` — grow a real mushroom, and validate the rate

```bash
python3 rt_2d.py
```

![RT 2D simulation](rt_2d.png)

A 2D incompressible Boussinesq simulation (vorticity–streamfunction, FFT Poisson
solve, semi-Lagrangian advection): heavy on light, a single-mode ripple, gravity.
The ripple grows into the classic **bubble** (light, rising) and **spike**
(heavy, falling), and the spike flanks roll up via Kelvin–Helmholtz into the
mushroom cap.

Then the validation: it measures the early-time growth rate off the flow and
compares to linear theory —

| | value |
|---|---|
| linear theory `√(A g k)` | 3.28 × 10⁹ s⁻¹ |
| measured from the 2D flow | 3.06 × 10⁹ s⁻¹ |
| **agreement** | **−6.7%** |

The residual is honest and visible in the growth-rate panel: the amplitude starts
*below* the exponential (it begins as `cosh(γt)` because the flow starts from
rest), plus the starting interface has finite thickness. Both suppress the
measured rate slightly. Matching `√(A g k)` ties this sim back to the dispersion
relation the mechanics script is built on — change `LAM` and both track together.

## What this is and isn't

Captures the instability **mechanism** and its linear growth rate. Not an
ICF-accurate mixing layer: incompressible Boussinesq (real fronts are
compressible and high-Atwood), single mode (no mode coupling or turbulent mix),
and 2D. The ablative *stabilization* — the physics that actually saves the
implosion — lives in the mechanics script, not the 2D sim. `NOTES` blocks in both
files list the knobs and the simplifications.
