# Inertial Confinement Fusion — toy simulations

A growing set of small, self-contained Python models for building physical
intuition about **inertial confinement fusion (ICF)**. These are deliberately
*toy* models — a few hundred lines each, runnable on a laptop — not design
codes. The goal is to *feel* how an ICF implosion and its ignition threshold
behave, with every simplification written down.

Each model is quantitatively anchored (real DT fusion reactivity, the rocket
equation, textbook ignition criteria) and cross-checked against known results.

## Models

### [`0-D Hotspot/`](0-D%20Hotspot) — hot-spot ignition

A zero-dimensional DT hot spot balancing **alpha self-heating** against
**bremsstrahlung radiation**, with **alpha confinement** set by the areal
density ρR. Uses the Bosch–Hale fusion reactivity fit (accurate 0.2–100 keV).

It reproduces the two facts that define ICF ignition:
- an ignition **temperature** ≈ **4.35 keV** (where alpha heating overtakes radiation), and
- an ignition **areal density** ρR ≈ **0.3 g/cm²** (so the alphas actually stop).

![hot-spot ignition](0-D%20Hotspot/hotspot_ignition.png)

The right panel is the point: burn-up is negligible, then jumps by orders of
magnitude over a ~1 keV window. That **ignition cliff** is why ICF is a
threshold phenomenon and so hard to hit.

### [`Rocket Implosion/`](Rocket%20Implosion) — thin-shell implosion

A lumped "rocket" model of the capsule implosion: the ablator blows off and, by
reaction, drives the shell inward to a huge velocity; a central gas cushion
decelerates it at stagnation. The output is fed straight into the hot-spot
ignition bar above.

For a NIF-scale toy target it delivers a self-consistent **igniting** design:

| quantity | value | note |
|---|---|---|
| implosion velocity | **345 km/s** | rocket equation predicts 345.4 (cross-check) |
| ablated fraction | 90% | payload = 10% of initial mass |
| kinetic energy | 17.9 kJ | NIF-scale |
| convergence ratio | 31 | R₀ / R_min |
| stagnation ρR | **2.27 g/cm²** | clears the 0.3 ignition bar |
| hot-spot temperature | **5.15 keV** | clears the 4.3 keV ignition bar |

![rocket implosion](Rocket%20Implosion/rocket_implosion.png)

### [`1-D Lagrangian Hydro/`](1-D%20Lagrangian%20Hydro) — resolved implosion

A spherical **1-D Lagrangian hydrodynamics** solve (staggered grid, artificial
viscosity, ideal-gas EOS). Instead of *estimating* stagnation, it *computes* it:
a driven shell implodes, launches a shock into the gas fill, and the shock
converges on the origin to form the hot spot — then rebounds.

It reaches a shock-heated hot spot of **10 keV** and **conserves energy to
≈1%** (the validity check). Being lossless and single-shock, it gets hot but not
dense (ρR stays low) — an honest illustration of what the ablative compression in
the models above actually buys you.

![1-D implosion](1-D%20Lagrangian%20Hydro/lagrangian_implosion.png)

### [`Rayleigh-Taylor/`](Rayleigh-Taylor) — why it's actually hard

The instability the 1-D models can't show. The imploding shell is
**Rayleigh–Taylor unstable**: ripples grow exponentially and can shred it before
stagnation. Two scripts — the **mechanics** (dispersion relation, ablative
stabilization, e-foldings over the implosion) and a **2D simulation** that grows
a real bubble-and-spike mushroom and measures its growth rate.

The headline: a 5 nm ripple grows ×72 under *ablative* RT (shell survives) but
would grow ×13000 classically (shell breaks up) — ablation is what holds the
capsule together. The 2D sim's measured growth rate matches `√(A g k)` to ~7%.

![RT 2D](Rayleigh-Taylor/rt_2d.png)

### [`ML Surrogate/`](ML%20Surrogate) — the data-driven layer

Sample the rocket model across its design space, train a neural-network
**surrogate**, and get the ignition boundary, sensitivity, and inverse design
almost for free — the miniature version of LLNL's "cognitive simulation"
pipeline. The surrogate hits R² ≈ 0.99 on all outputs and classifies ignition at
~97%. The highlight is honest: the inverse-design optimizer first proposes an
over-optimistic design that the real simulator says **fizzles**, so the code does
what real ICF-ML does — verify, resample, retrain (**active learning**) — until
it lands a design **verified** to ignite.

![ML surrogate](ML%20Surrogate/ml_ignition.png)

## Running

```bash
pip install -r requirements.txt
python3 "0-D Hotspot/hotspot_0d.py"
python3 "Rocket Implosion/rocket_implosion.py"
python3 "1-D Lagrangian Hydro/lagrangian_1d.py"
python3 "1-D Lagrangian Hydro/hydro_validation.py"   # verify the solver vs exact solutions
python3 "Rayleigh-Taylor/rt_mechanics.py"
python3 "Rayleigh-Taylor/rt_2d.py"
python3 "ML Surrogate/ml_ignition.py"                # needs scikit-learn
```

Each script prints its headline numbers and saves its figure alongside itself.
Every file has a `NOTES` block at the bottom listing the knobs to play with and
the physics that was deliberately left out.

## What's deliberately missing

These are toy models. The largest omissions, roughly in order of impact:
- radiation transport beyond optically-thin bremsstrahlung
- separate ion and electron temperatures; electron heat conduction
- real (Fermi-degenerate) EOS for the cold dense shell
- pulse shaping, cross-beam effects, laser–plasma instabilities
- fully coupled multi-dimensional radiation-hydro (the RT models are stand-alone)

## Roadmap

- [x] 1-D Lagrangian hydro (watch the shock converge and form the hot spot)
- [x] verify the solver against exact solutions (Sod / Noh / Sedov)
- [x] Rayleigh–Taylor growth on the imploding shell (mechanics + 2D sim)
- [x] ML surrogate + ignition boundary + inverse design (active learning)
- [ ] multi-fidelity ML: transfer-learn from these toys onto MULTI runs
- [ ] couple the implosion output into a time-dependent burn

---

*Toy models for learning, not for target design. Corrections welcome.*
