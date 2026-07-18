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

## Running

```bash
pip install -r requirements.txt
python3 "0-D Hotspot/hotspot_0d.py"
python3 "Rocket Implosion/rocket_implosion.py"
```

Each script prints its headline numbers and saves its figure alongside itself.
Every file has a `NOTES` block at the bottom listing the knobs to play with and
the physics that was deliberately left out.

## What's deliberately missing

These are toy models. The largest omissions, roughly in order of impact:
- **Rayleigh–Taylor instability** — the ablation front is unstable; RT growth is
  the single effect that most limits real implosions. (Next on the roadmap.)
- multi-zone hydrodynamics (both models are lumped / single-shell)
- radiation transport beyond optically-thin bremsstrahlung
- separate ion and electron temperatures; electron heat conduction
- pulse shaping, cross-beam effects, laser–plasma instabilities

## Roadmap

- [ ] Rayleigh–Taylor growth on the imploding shell
- [ ] 1-D Lagrangian radiation-hydro (watch the shock converge and form the hot spot)
- [ ] couple the implosion output into a time-dependent burn

---

*Toy models for learning, not for target design. Corrections welcome.*
