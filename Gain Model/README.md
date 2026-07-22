# End-to-end gain — the number that matters

Every other model in this repo covers one stage of an implosion. This one chains
them into a single forward model — **`design → gain`** — and pins the free coupling
efficiencies so that a NIF-scale design reproduces the **real December 2022
ignition shot** (N221204).

```bash
python3 gain_model.py
```

![gain model](gain_model.png)

## The energy ledger

```
laser energy
  × eta_hohlraum   laser  → X-rays absorbed by the capsule
  × eta_rocket     absorbed → imploding-shell kinetic energy
  → stagnation     KE → hot-spot temperature;  convergence → areal density
  → burn           0-D Hotspot model → burn-up fraction        (reused)
  → fusion yield   burn-up × fuel × 17.6 MeV
gain = yield / laser energy
```

The chasm between the stages is the whole story of why ICF is hard: of 2 MJ of
laser, ~14% is absorbed, ~1% becomes shell kinetic energy — and yet, once the hot
spot ignites, the fusion yield comes back **larger than the laser energy**.

## Calibrated to the real ignition point

The two coupling efficiencies and the stagnation/burn coefficients are set **once**
so the reference design lands on NIF N221204. Everything downstream — the ignition
cliff in gain, the sensitivity of gain to implosion velocity — then falls out of
the 0-D burn:

| quantity | model | NIF N221204 |
|---|---|---|
| laser energy | 2.05 MJ | 2.05 MJ |
| fusion yield | **3.22 MJ** | 3.15 MJ |
| target gain | **1.57** | 1.54 |
| hot-spot temperature | **10.7 keV** | ~10.9 keV |
| burn-up fraction | 4.5 % | ~4 % |

All within ~2%.

## Why it matters

This is the **integrating layer** for the repo. Implosion velocity comes from the
rocket / convergent models, drive symmetry from the hohlraum view factor, and yield
degradation from the asymmetry model — all feeding this one `design → gain` forward
model that the ML can then optimize *end to end*, against the metric fusion actually
cares about.

## What this is and isn't

A reduced, calibrated model — not a radiation-hydrodynamics code. It will not
resolve a mix layer or a real hohlraum drive history. Its value is being a fast,
transparent, end-to-end surrogate that is *anchored to reality*: change the design
and watch gain move, with every simplification written down in the `NOTES`.
Next up: replace the stagnation scalings with the 1-D Lagrangian hydro so the
hot-spot conditions are computed rather than fit.
