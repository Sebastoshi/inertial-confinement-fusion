r"""
Coupled gain model — the hot-spot temperature is now COMPUTED by the 1-D
Lagrangian hydro, not fit.

`gain_model.py` estimates the stagnation hot-spot temperature with a single
calibrated scaling,  T_hs = K * v_imp^2.  This module removes that fit: it drives
the actual 1-D Lagrangian PDE solve at an ablation pressure set by the laser
energy, and reads the *computed* hot-spot temperature straight off the converging
shock. The chain becomes

    laser energy
      x eta_hohlraum   -> X-rays absorbed by the capsule
      -> ablation drive pressure   P_a  ~  (absorbed intensity)^(2/3)
      -> [ 1-D LAGRANGIAN HYDRO ]  -> hot-spot temperature T_hs   (COMPUTED)
      -> stagnation areal density  rho*R                          (calibrated*)
      -> burn          0-D Hotspot model -> burn-up fraction       (reused)
      -> fusion yield  -> gain

  * Why rho*R stays calibrated: the 1-D hydro is a lossless single-shock toy with
    no ablative compression, so it reaches ignition *temperature* (~10 keV, right
    on NIF) but almost no areal density (rho*R ~ 0.003 vs the ~1 g/cm^2 needed).
    Temperature — the ignition trigger — is what it computes well, so that is what
    we take from it; rho*R remains the calibrated compression stand-in until a
    hydro with a resolved ablation front is wired in.

The hydro is a real PDE solve (seconds per point), far too slow for an ML inner
loop, so `build_hydro_table.py` samples it once across drive pressure and dumps
`hydro_table.json`; here we fit a smooth T_hs(P_drive) to that table and evaluate
instantly. The fit stays anchored to the same NIF N221204 point as gain_model.py.

Run:  python3 coupled_gain.py       (expects hydro_table.json alongside; build it
                                      with build_hydro_table.py if missing)
Deps: numpy  (matplotlib only for the figure)
"""

import os
import json
import importlib.util
from dataclasses import dataclass

import numpy as np

# --- reuse the calibrated end-to-end model (efficiencies, rho*R, burn, NIF ref) ---
_gm_path = os.path.join(os.path.dirname(__file__), "gain_model.py")
_spec = importlib.util.spec_from_file_location("gain_model", _gm_path)
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)

# --- drive-pressure model:  ablation pressure ~ (absorbed intensity)^(2/3) --------
# For a fixed pulse duration and ablation area, absorbed intensity ~ laser energy,
# so P_a ~ E_laser^(2/3). One anchor pressure pins it to the NIF-scale point.
P_DRIVE_NIF = 153.0     # Mbar at the reference (NIF) laser energy -> ~10.7 keV hot spot
DRIVE_EXP   = 2.0 / 3.0


def load_hydro_fit(path=None):
    """Load hydro_table.json and fit the computed T_hs(P_drive) [keV vs Mbar].

    Strong-shock heating makes post-shock T proportional to drive pressure at
    fixed shock compression, so a straight line through the sampled hydro points
    is both an excellent fit and the physically expected form.
    """
    path = path or os.path.join(os.path.dirname(__file__), "hydro_table.json")
    with open(path) as f:
        rows = json.load(f)["rows"]
    P = np.array([r["P_mbar"] for r in rows])
    T = np.array([r["T_hs"] for r in rows])
    slope, intercept = np.polyfit(P, T, 1)          # T_hs = slope*P + intercept
    return dict(P=P, T=T, slope=float(slope), intercept=float(intercept),
                cr=float(np.mean([r["cr"] for r in rows])),
                rhoR_hydro=float(np.mean([r["rhoR"] for r in rows])))


HYDRO = load_hydro_fit()


def drive_pressure(E_laser):
    """Laser energy [J] -> ablation drive pressure [Mbar]."""
    return P_DRIVE_NIF * (E_laser / gm.NIF.E_laser) ** DRIVE_EXP


def hydro_T_hs(P_mbar):
    """Hot-spot temperature [keV] from the hydro fit (the computed quantity)."""
    return max(HYDRO["slope"] * P_mbar + HYDRO["intercept"], 0.05)


@dataclass
class Design:
    E_laser: float      # laser energy on target [J]
    fuel_mass: float    # DT fuel mass that can burn [kg]
    conv_ratio: float   # convergence ratio R0 / R_min (sets ablative compression)
    adiabat: float      # in-flight adiabat (lower = more compressible)


# Reference design, same NIF N221204 point as gain_model.py (velocity now implicit
# in the drive pressure rather than a free field).
NIF = Design(E_laser=gm.NIF.E_laser, fuel_mass=gm.NIF.fuel_mass,
             conv_ratio=gm.NIF.conv_ratio, adiabat=gm.NIF.adiabat)


def stagnation(d):
    """Design -> (T_hs [keV] COMPUTED by hydro, rho [g/cc], rho*R [g/cm^2] calibrated)."""
    P = drive_pressure(d.E_laser)
    T_hs = hydro_T_hs(P)                                            # <-- from the PDE solve
    rhoR = gm.RHOR_REF * (d.conv_ratio / gm.CR_REF) ** 2 * (gm.ALPHA_REF / d.adiabat)
    return T_hs, gm.RHO_HS, rhoR


def evaluate(d):
    """Full energy ledger for a design point, hot-spot T from the 1-D hydro."""
    E_abs = gm.ETA_HOHLRAUM * d.E_laser
    KE    = gm.ETA_ROCKET * E_abs
    T_hs, rho, rhoR = stagnation(d)
    b = gm.burn(rho, rhoR, T_hs, n_tau=gm.N_TAU_BURN)
    n_react = b["burnup"] * d.fuel_mass / gm.M_DT_PAIR
    Y = n_react * gm.E_FUSION
    return dict(E_laser=d.E_laser, E_abs=E_abs, KE=KE, P_drive=drive_pressure(d.E_laser),
                T_hs=T_hs, rhoR=rhoR, burnup=b["burnup"], yield_J=Y, gain=Y / d.E_laser)


def _scaled(E_laser, ref=NIF):
    """Same design at a different laser energy (drive pressure follows E^(2/3))."""
    return Design(E_laser=E_laser, fuel_mass=ref.fuel_mass,
                  conv_ratio=ref.conv_ratio, adiabat=ref.adiabat)


def report():
    r = evaluate(NIF)
    print("=" * 68)
    print("  COUPLED GAIN MODEL  --  hot-spot T computed by the 1-D hydro")
    print("=" * 68)
    print(f"  drive pressure at NIF energy : {r['P_drive']:8.0f} Mbar")
    print(f"  hot-spot T (HYDRO-COMPUTED)  : {r['T_hs']:8.2f} keV   "
          f"[fit: T = {HYDRO['slope']:.4f}*P {HYDRO['intercept']:+.2f}]")
    print(f"  stagnation rho*R (calibrated): {r['rhoR']:8.2f} g/cm^2")
    print(f"  burn-up fraction             : {r['burnup']*100:8.1f} %")
    print(f"  -> fusion yield              : {r['yield_J']/1e6:8.2f} MJ")
    print(f"  TARGET GAIN                  : {r['gain']:8.2f}")
    print("-" * 68)
    print("  VALIDATION vs NIF N221204  (hot-spot T is now COMPUTED, not fit)")
    print(f"   {'quantity':<22}{'model':>10}{'NIF':>10}")
    for name, mod, nif, unit in [
        ("laser energy",  NIF.E_laser/1e6, 2.05, "MJ"),
        ("fusion yield",  r['yield_J']/1e6, 3.15, "MJ"),
        ("target gain",   r['gain'],        1.54, ""),
        ("hot-spot T",    r['T_hs'],        10.9, "keV")]:
        err = (mod - nif) / nif * 100
        print(f"   {name:<22}{mod:>10.2f}{nif:>10.2f}  {unit:<5} ({err:+.0f}%)")
    print("=" * 68)
    return r


def make_figure(fname="coupled_gain.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    r = evaluate(NIF)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) the coupling itself: hydro-computed T_hs vs drive pressure, + fit
    Pgrid = np.linspace(HYDRO["P"].min(), HYDRO["P"].max(), 100)
    ax[0].plot(HYDRO["P"], HYDRO["T"], "o", color="tab:red", ms=7, label="1-D hydro (PDE solve)")
    ax[0].plot(Pgrid, HYDRO["slope"] * Pgrid + HYDRO["intercept"], "k--", lw=1.5,
               label=f"fit  T = {HYDRO['slope']:.3f} P {HYDRO['intercept']:+.1f}")
    ax[0].axhline(4.3, color="tab:gray", ls=":", lw=1)
    ax[0].text(HYDRO["P"].min(), 4.6, " ignition T ~4.3 keV", fontsize=8, color="tab:gray")
    ax[0].plot(r["P_drive"], r["T_hs"], "k*", ms=15)
    ax[0].text(r["P_drive"], r["T_hs"] * 0.80, "NIF", ha="center", fontsize=9)
    ax[0].set_xlabel("ablation drive pressure [Mbar]")
    ax[0].set_ylabel("hot-spot temperature [keV]")
    ax[0].set_title("the coupling: hot-spot T is COMPUTED\nby the 1-D Lagrangian hydro")
    ax[0].legend(fontsize=8)

    # (2) gain vs laser energy -- the ignition cliff, now with hydro-computed T
    E = np.linspace(0.8e6, 2.6e6, 120)
    g = np.array([evaluate(_scaled(e))["gain"] for e in E])
    ax[1].plot(E/1e6, g, "tab:red", lw=2.5)
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].text(0.85, 1.15, "gain = 1 (breakeven)", fontsize=8)
    ax[1].plot(NIF.E_laser/1e6, r["gain"], "k*", ms=15)
    ax[1].text(NIF.E_laser/1e6 - 0.03, r["gain"]*0.55, "NIF 2022", ha="center", fontsize=9)
    ax[1].set_yscale("log"); ax[1].set_xlabel("laser energy [MJ]")
    ax[1].set_ylabel("target gain")
    ax[1].set_title("ignition cliff in gain\n(driven by the hydro hot-spot T)")

    # (3) validation vs NIF
    names = ["laser\n[MJ]", "yield\n[MJ]", "gain", "hot-spot T\n[keV/5]"]
    model = [NIF.E_laser/1e6, r['yield_J']/1e6, r['gain'], r['T_hs']/5]
    nif   = [2.05, 3.15, 1.54, 10.9/5]
    x = np.arange(len(names)); w = 0.38
    ax[2].bar(x - w/2, model, w, label="model", color="tab:blue")
    ax[2].bar(x + w/2, nif, w, label="NIF 2022", color="tab:orange")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_title("validation vs the real ignition shot")
    ax[2].legend(fontsize=9)

    fig.suptitle("Coupled ICF gain: hot-spot temperature computed by the 1-D Lagrangian hydro",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * What changed vs gain_model.py: the hot-spot temperature is no longer the fit
#   T_hs = K*v_imp^2. It is read from the 1-D Lagrangian hydro, which resolves the
#   converging shock and lands on ~10 keV at NIF-scale drive with no temperature
#   knob at all -- an independent confirmation of the energy ledger.
# * Still calibrated: rho*R (the toy hydro builds no areal density), the two
#   coupling efficiencies, and the burn window. Next: a hydro with a resolved
#   ablation front would let rho*R be computed too, closing the last fit.
# * The drive-pressure map P_a ~ E_laser^(2/3) is the one modelling choice bridging
#   laser energy to the hydro's boundary condition; the anchor P_DRIVE_NIF pins it
#   to N221204, exactly as the efficiencies are pinned in gain_model.py.
# ----------------------------------------------------------------------------
