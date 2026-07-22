r"""
Cache the deceleration-phase RT growth-factor spectrum vs convergence ratio.

Step 3 needs a *multimode* mix width at stagnation, and it needs it fast enough for
an ML/Monte-Carlo loop. The repo's convergent_rt.py already grows the full Legendre
spectrum through deceleration-phase RT + Bell-Plesset convergence -- but tuning the
gas pressure to a target convergence ratio (tune_P0) costs ~dozens of trajectory
integrations, far too slow to call per design.

So we run it once here across a grid of convergence ratios and store, for each CR,
the per-mode growth factor  GF(ell) = eta_stag(ell) / eta_0  and the stagnation
radius R_min. rt_mix.py interpolates this table in CR, applies the surface-roughness
seed spectrum and the acceleration-phase feedthrough, and sums to a mix width in
microseconds instead of seconds. Growth is linear, so GF is independent of the seed
amplitude -- one solve per CR captures it.

Run:  python3 build_rt_table.py        (writes rt_table.json here)
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rayleigh-Taylor"))
import convergent_rt as crt        # noqa: E402

ELL_MAX = 100                       # modes 1..100 (interface thickness cuts off higher ell)


def gf_spectrum(target_cr):
    """Growth-factor spectrum GF(ell) and stagnation radius at a target CR."""
    P0 = crt.tune_P0(target_cr)
    T, R, V, Acc = crt.trajectory(P0)
    istag = int(np.argmin(R))
    ell = np.arange(1, ELL_MAX + 1)
    eta_stag, _ = crt.integrate_modes(T, R, V, Acc, ell, term="full", eta0=crt.A0)
    GF = np.abs(eta_stag) / crt.A0
    return dict(cr=float(R[0] / R[istag]), target_cr=float(target_cr),
                r_min_m=float(R[istag]), GF=GF.tolist())


def main():
    crs = [24, 27, 30, 33, 35, 38, 41, 44, 47]
    rows = []
    for cr in crs:
        r = gf_spectrum(cr)
        rows.append(r)
        gf = np.array(r["GF"])
        print(f"  CR ~ {r['cr']:5.1f}  (R_min = {r['r_min_m']*1e6:5.1f} um)   "
              f"GF: peak x{gf.max():.1e} at ell={int(np.argmax(gf))+1:3d},  "
              f"ell=10 x{gf[9]:.1f}")

    out = os.path.join(os.path.dirname(__file__), "rt_table.json")
    with open(out, "w") as f:
        json.dump({"ell_max": ELL_MAX, "atwood": crt.ATWOOD,
                   "v_imp_ms": crt.V_IMP, "r0_m": crt.R0, "rows": rows}, f, indent=2)
    print(f"\nWrote {len(rows)} CR points (modes 1..{ELL_MAX}) -> {out}")


if __name__ == "__main__":
    main()
