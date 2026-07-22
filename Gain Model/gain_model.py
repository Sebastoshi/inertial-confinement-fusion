r"""
End-to-end ICF gain model — the energy ledger from laser to fusion yield,
calibrated to the NIF December 2022 ignition shot (N221204).

Every other model in this repo covers ONE stage. This one chains them into a
single forward model  design -> gain,  and pins the free coupling efficiencies so
that a NIF-scale design reproduces the real ignition point:

    laser energy
      x eta_hohlraum   laser  -> X-rays absorbed by the capsule
      x eta_rocket     absorbed -> imploding-shell kinetic energy
      -> stagnation    KE -> hot-spot temperature;  convergence -> areal density
      -> burn          0-D Hotspot model -> burn-up fraction          [reused]
      -> fusion yield  burn-up x fuel x 17.6 MeV
    gain = yield / laser energy

NIF N221204 (Dec 2022):  2.05 MJ in,  3.15 MJ out,  gain ~1.5,  hot spot ~10-11 keV.
The two efficiencies and the stagnation coefficients are the only free knobs; they
are set once so the reference design lands on that point, and everything else -- the
ignition cliff in gain, the sensitivity to velocity and symmetry -- then follows.

Run:  python3 gain_model.py
Deps: numpy, scipy  (matplotlib only for the figure)
"""

import os
import importlib.util
from dataclasses import dataclass, replace

import numpy as np

# --- reuse the repo's validated 0-D hot-spot burn model ---
_hs_path = os.path.join(os.path.dirname(__file__), "..", "0-D Hotspot", "hotspot_0d.py")
_spec = importlib.util.spec_from_file_location("hotspot_0d", _hs_path)
hotspot_0d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hotspot_0d)
burn = hotspot_0d.burn

# --- physical constants ---
E_FUSION  = 17.6e6 * 1.602176634e-19          # J released per D-T fusion reaction
M_DT_PAIR = (2.014 + 3.016) * 1.66053907e-27  # kg per (D + T) pair

# --- calibration knobs (pinned once to NIF N221204) ---
ETA_HOHLRAUM = 0.14     # laser energy -> capsule-absorbed X-ray energy
ETA_ROCKET   = 0.075    # absorbed energy -> imploding-shell kinetic energy
K_THOTSPOT   = 6.7e-11  # hot-spot T [keV] = K_THOTSPOT * v_imp[m/s]^2
RHOR_REF     = 1.20     # stagnation fuel areal density [g/cm^2] at the reference design
CR_REF       = 35.0     # reference convergence ratio
ALPHA_REF    = 2.0      # reference in-flight adiabat
RHO_HS       = 300.0    # hot-spot mass density used in the burn [g/cm^3]
N_TAU_BURN   = 0.42     # burn confinement window (fraction of the naive inertial
                        # time) -- a marginally-igniting hot spot disassembles early


@dataclass
class Design:
    E_laser: float      # laser energy on target [J]
    fuel_mass: float    # DT fuel mass that can burn [kg]
    v_imp: float        # implosion velocity [m/s]
    conv_ratio: float   # convergence ratio R0 / R_min
    adiabat: float      # in-flight adiabat (lower = more compressible)


# Reference: a NIF-scale igniting design (N221204-like).
NIF = Design(E_laser=2.05e6, fuel_mass=0.21e-6, v_imp=400e3, conv_ratio=35.0, adiabat=2.0)


def stagnation(d):
    """Design -> hot-spot conditions at stagnation: (T_hs [keV], rho [g/cc], rhoR [g/cm^2])."""
    T_hs = K_THOTSPOT * d.v_imp ** 2                                   # KE -> thermal
    rhoR = RHOR_REF * (d.conv_ratio / CR_REF) ** 2 * (ALPHA_REF / d.adiabat)
    return T_hs, RHO_HS, rhoR


def evaluate(d):
    """Full energy ledger for a design point."""
    E_abs = ETA_HOHLRAUM * d.E_laser              # laser -> absorbed
    KE    = ETA_ROCKET * E_abs                    # absorbed -> shell KE
    T_hs, rho, rhoR = stagnation(d)
    b = burn(rho, rhoR, T_hs, n_tau=N_TAU_BURN)   # 0-D ignition/burn
    n_react = b["burnup"] * d.fuel_mass / M_DT_PAIR
    Y = n_react * E_FUSION                        # fusion yield [J]
    return dict(E_laser=d.E_laser, E_abs=E_abs, KE=KE,
                T_hs=T_hs, rhoR=rhoR, burnup=b["burnup"],
                yield_J=Y, gain=Y / d.E_laser)


def _scaled(E_laser, ref=NIF):
    """A design at a different laser energy: velocity scales as sqrt(coupled energy)."""
    v = ref.v_imp * (E_laser / ref.E_laser) ** 0.5
    return replace(ref, E_laser=E_laser, v_imp=v)


def report():
    r = evaluate(NIF)
    print("=" * 66)
    print("  END-TO-END ICF GAIN MODEL  --  calibrated to NIF N221204")
    print("=" * 66)
    print("  ENERGY LEDGER (reference design)")
    print(f"    laser on target        : {NIF.E_laser/1e6:8.2f} MJ")
    print(f"    -> absorbed by capsule : {r['E_abs']/1e3:8.1f} kJ   "
          f"({r['E_abs']/NIF.E_laser*100:.1f}% of laser)")
    print(f"    -> shell kinetic energy: {r['KE']/1e3:8.1f} kJ   "
          f"({r['KE']/NIF.E_laser*100:.2f}% of laser)")
    print(f"    hot-spot temperature   : {r['T_hs']:8.1f} keV")
    print(f"    stagnation rho*R        : {r['rhoR']:8.2f} g/cm^2")
    print(f"    burn-up fraction        : {r['burnup']*100:8.1f} %")
    print(f"    -> fusion yield         : {r['yield_J']/1e6:8.2f} MJ")
    print(f"    TARGET GAIN             : {r['gain']:8.2f}")
    print("-" * 66)
    print("  VALIDATION vs NIF N221204")
    print(f"   {'quantity':<22}{'model':>10}{'NIF':>10}")
    for name, mod, nif, unit in [
        ("laser energy",  NIF.E_laser/1e6, 2.05, "MJ"),
        ("fusion yield",  r['yield_J']/1e6, 3.15, "MJ"),
        ("target gain",   r['gain'],        1.54, ""),
        ("hot-spot T",    r['T_hs'],        10.9, "keV")]:
        err = (mod - nif) / nif * 100
        print(f"   {name:<22}{mod:>10.2f}{nif:>10.2f}  {unit:<5} ({err:+.0f}%)")
    print("=" * 66)
    return r


def make_figure(fname="gain_model.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    r = evaluate(NIF)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) energy ledger -- log-scale bars from laser to yield
    stages = ["laser\n2.05 MJ", "absorbed", "shell KE", "hot spot", "yield"]
    E_hs = 3.0 * r['T_hs'] * hotspot_0d.KEV_J * (NIF.fuel_mass / M_DT_PAIR)  # ~hot-spot internal E
    vals = [NIF.E_laser, r['E_abs'], r['KE'], max(E_hs, 1e2), r['yield_J']]
    cols = ["tab:gray", "tab:orange", "tab:red", "tab:purple", "tab:green"]
    ax[0].bar(stages, np.array(vals)/1e3, color=cols)
    ax[0].set_yscale("log"); ax[0].set_ylabel("energy [kJ]")
    ax[0].set_title("the energy ledger:\nhuge losses in, fusion gain out")
    for i, v in enumerate(vals):
        ax[0].text(i, v/1e3*1.25, f"{v/1e3:.0f}" if v >= 1e3 else f"{v/1e3:.1f}",
                   ha="center", fontsize=8)

    # (2) gain vs laser energy -- the ignition cliff
    E = np.linspace(0.8e6, 2.6e6, 120)
    g = np.array([evaluate(_scaled(e))["gain"] for e in E])
    ax[1].plot(E/1e6, g, "tab:red", lw=2.5)
    ax[1].axhline(1.0, color="k", ls=":", lw=1); ax[1].text(0.85, 1.15, "gain = 1 (breakeven)", fontsize=8)
    ax[1].plot(NIF.E_laser/1e6, r["gain"], "k*", ms=15)
    ax[1].text(NIF.E_laser/1e6 - 0.03, r["gain"]*0.55, "NIF 2022", ha="center", fontsize=9)
    ax[1].set_yscale("log"); ax[1].set_xlabel("laser energy [MJ]"); ax[1].set_ylabel("target gain")
    ax[1].set_title("the ignition cliff in gain:\nyield jumps as the hot spot ignites")

    # (3) validation vs NIF
    names = ["laser\n[MJ]", "yield\n[MJ]", "gain", "hot-spot T\n[keV/5]"]
    model = [NIF.E_laser/1e6, r['yield_J']/1e6, r['gain'], r['T_hs']/5]
    nif   = [2.05, 3.15, 1.54, 10.9/5]
    x = np.arange(len(names)); w = 0.38
    ax[2].bar(x - w/2, model, w, label="model", color="tab:blue")
    ax[2].bar(x + w/2, nif, w, label="NIF 2022", color="tab:orange")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_title("validation vs the real ignition shot"); ax[2].legend(fontsize=9)

    fig.suptitle("End-to-end ICF gain: laser -> implosion -> burn, calibrated to NIF",
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
# * This is a reduced, calibrated model, not a radiation-hydro code. The two
#   coupling efficiencies (eta_hohlraum, eta_rocket) and the stagnation scalings
#   are pinned once to NIF N221204; the ignition cliff in gain, and the strong
#   sensitivity of gain to implosion velocity, then fall out of the 0-D burn.
# * It is the integrating layer for the repo: velocity comes from the rocket /
#   convergent models, drive symmetry from the hohlraum view factor, and yield
#   degradation from the asymmetry model -- all feeding this single design -> gain
#   forward model that the ML can then optimize end to end.
# * Next: replace the stagnation scalings with the 1-D Lagrangian hydro so T_hs and
#   rho*R are computed, not fit; add a real absorbed-fraction / rocket calc.
# ----------------------------------------------------------------------------
