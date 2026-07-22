r"""
Build a cached map  drive pressure -> stagnation conditions  by actually running
the 1-D Lagrangian hydro at a spread of ablation drive pressures.

This is the slow, one-time step behind the coupled gain model: the hydro is a real
PDE solve (a few seconds per point), far too slow to call inside an ML inner loop.
So we run it once across a grid of drive pressures, record the *computed* stagnation
hot-spot temperature (and convergence / areal density), and dump the table to JSON.
`coupled_gain.py` then fits a smooth T_hs(P_drive) to it and evaluates instantly.

Run:  python3 build_hydro_table.py        (writes hydro_table.json here)
"""

import os
import json
import importlib.util

import numpy as np

# --- load the 1-D Lagrangian hydro as a module ---
_lag_path = os.path.join(os.path.dirname(__file__), "..", "1-D Lagrangian Hydro", "lagrangian_1d.py")
_spec = importlib.util.spec_from_file_location("lagrangian_1d", _lag_path)
lag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lag)


def hydro_point(P_mbar):
    """Run the hydro at drive pressure P_mbar [Mbar]; return computed stagnation."""
    lag.P_MAX = P_mbar * lag.MBAR_PA          # override the module-level drive
    hist, _snaps, _is_fill, _node_idx = lag.run()
    pk = lag.peak_conditions(hist)
    return {
        "P_mbar": float(P_mbar),
        "T_hs": float(pk["Ths"]),             # mass-averaged hot-spot T [keV]  <-- the payoff
        "cr": float(pk["cr"]),                # fuel convergence ratio
        "rhoR": float(pk["rhoR"]),            # hot-spot areal density [g/cm^2]
        "rho_gcc": float(pk["rhoc"] / 1000),  # peak hot-spot density [g/cc]
        "t_stag_ns": float(pk["t"] * 1e9),
    }


def main():
    # spread of ablation drive pressures around the ~150 Mbar NIF-scale point
    pressures = [70, 90, 110, 130, 150, 170, 190, 220, 260]
    rows = []
    for P in pressures:
        r = hydro_point(P)
        rows.append(r)
        print(f"  P = {P:4.0f} Mbar  ->  T_hs = {r['T_hs']:6.2f} keV   "
              f"CR = {r['cr']:5.1f}   rhoR = {r['rhoR']:.4f} g/cm^2   "
              f"t_stag = {r['t_stag_ns']:.2f} ns")

    out = os.path.join(os.path.dirname(__file__), "hydro_table.json")
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nWrote {len(rows)} points -> {out}")


if __name__ == "__main__":
    main()
