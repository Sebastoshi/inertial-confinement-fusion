"""
Plasma thermalization of CaH molecules -- multi-temperature relaxation.

When cold CaH molecules are introduced into a hot plasma (or any dense bath),
their energy modes do NOT reach equilibrium together. Collisions dump energy
into translation almost immediately, into rotation after ~10 collisions, and
into vibration only after thousands -- so the molecule passes through a long
non-equilibrium window with three distinct "temperatures". This separation of
timescales is the whole story of molecular energy relaxation, and it matters
anywhere molecules meet a plasma (edge/divertor plasmas, molecular beams into
discharges, astrophysical and cold-molecule environments).

Model -- relaxation-time approximation, one temperature per mode relaxing toward
the bath at a mode-specific collision number Z:

    dT_i/dt = (T_bath - T_i) / tau_i ,   tau_i = Z_i / nu_coll

with a gas-kinetic collision frequency nu_coll = n_bath * sigma * v_rel and the
real CaH molecular constants (rotational B_e, vibrational omega_e, bond energy).

Run:  python3 cah_thermalization.py
Deps: numpy, matplotlib
"""

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
KB   = 1.380649e-23
AMU  = 1.66053907e-27
EV_K = 11604.5             # 1 eV in kelvin
CM1_K = 1.438777           # 1 cm^-1 in kelvin (hc/k)

# CaH (X 2Sigma+) spectroscopic constants
OMEGA_E = 1298.0           # vibrational frequency [cm^-1]
B_E     = 4.2296           # rotational constant  [cm^-1]
D0_EV   = 1.70             # dissociation energy  [eV]
M_CAH   = (40.078 + 1.008) * AMU                   # molecular mass [kg]

THETA_VIB = OMEGA_E * CM1_K        # ~1868 K  vibrational temperature
THETA_ROT = B_E * CM1_K            # ~6.1 K   rotational temperature
T_DISS    = D0_EV * EV_K           # ~19700 K dissociation temperature

# Collisional relaxation: number of collisions to relax each mode
Z_TR, Z_ROT, Z_VIB = 3.0, 10.0, 3000.0
SIGMA = 1.0e-19            # gas-kinetic cross section [m^2] (~few Angstrom^2)

# Bath (plasma) and initial molecular state
N_BATH  = 1.0e22          # bath density [m^-3]  (1e16 cm^-3)
T_BATH  = 1.0 * EV_K      # bath temperature [K]  (1 eV)
T0_MOL  = 300.0           # molecules injected cold [K]


def collision_freq(n_bath, T_bath, M=M_CAH):
    v_rel = np.sqrt(8.0 * KB * T_bath / (np.pi * M))
    return n_bath * SIGMA * v_rel

def relaxation_times(n_bath=N_BATH, T_bath=T_BATH):
    nu = collision_freq(n_bath, T_bath)
    return np.array([Z_TR, Z_ROT, Z_VIB]) / nu     # tau_tr, tau_rot, tau_vib


def temperatures(t, taus, T0=T0_MOL, Tb=T_BATH):
    """Analytic relaxation T_i(t) = Tb + (T0 - Tb) exp(-t/tau_i)."""
    return Tb + (T0 - Tb) * np.exp(-t[:, None] / taus[None, :])


def dissociation_fraction(Tb=T_BATH):
    """Rough Boltzmann-tail estimate of molecules above the bond energy."""
    return np.exp(-T_DISS / Tb)


def report(taus):
    print("=" * 62)
    print("  CaH PLASMA THERMALIZATION  --  multi-temperature relaxation")
    print("=" * 62)
    print(f"  bath                     : n={N_BATH*1e-6:.0e} cm^-3, "
          f"T={T_BATH/EV_K:.2f} eV ({T_BATH:.0f} K)")
    print(f"  CaH characteristic temps : theta_rot={THETA_ROT:.1f} K, "
          f"theta_vib={THETA_VIB:.0f} K")
    print(f"  dissociation temperature : {T_DISS:.0f} K ({D0_EV:.2f} eV)")
    print("-" * 62)
    print(f"  translational relax time : {taus[0]*1e6:8.2f} us")
    print(f"  rotational   relax time  : {taus[1]*1e6:8.2f} us")
    print(f"  vibrational  relax time  : {taus[2]*1e6:8.2f} us  "
          f"({taus[2]/taus[0]:.0f}x slower than translation)")
    print("-" * 62)
    print(f"  bath vs modes: rotation fully classical (T_bath >> theta_rot),")
    print(f"                 vibration partly excited (T_bath ~ {T_BATH/THETA_VIB:.1f} theta_vib),")
    print(f"                 T_bath < dissociation -> most molecules survive")
    print(f"  Boltzmann-tail dissociation fraction ~ {dissociation_fraction()*100:.1f} %")
    print("=" * 62)


def make_figure(taus, fname="cah_thermalization.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
    labels = ["translational", "rotational", "vibrational"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    # (1) the three temperatures relaxing to the bath (log time)
    a = ax[0]
    t = np.logspace(-8, -1, 600)
    T = temperatures(t, taus)
    for i in range(3):
        a.semilogx(t * 1e6, T[:, i], color=colors[i], lw=2.2, label=labels[i])
    a.axhline(T_BATH, color="k", ls="--", lw=1)
    a.text(t[0] * 1e6 * 1.3, T_BATH * 0.9, "bath T", fontsize=8)
    a.set_xlabel("time [us]"); a.set_ylabel("mode temperature [K]")
    a.set_title("cold CaH into hot plasma:\nmodes thermalize on separate clocks")
    a.legend(fontsize=9, loc="center left")

    # (2) relaxation times vs plasma density (the density lever)
    b = ax[1]
    n = np.logspace(20, 25, 60)                        # m^-3
    for i, (Z, lab) in enumerate(zip([Z_TR, Z_ROT, Z_VIB], labels)):
        taun = Z / collision_freq(n, T_BATH)
        b.loglog(n * 1e-6, taun * 1e6, color=colors[i], lw=2.2, label=lab)
    b.axvline(N_BATH * 1e-6, color="gray", ls=":", lw=1)
    b.set_xlabel("plasma density [cm$^{-3}$]"); b.set_ylabel("relaxation time [us]")
    b.set_title(r"relaxation time $\propto 1/n$" + "\n(denser plasma = faster thermalization)")
    b.legend(fontsize=9)

    # (3) CaH energy-scale ladder vs the bath
    c = ax[2]
    scales = [(THETA_ROT, "theta_rot (rotation), 6 K", "tab:green", "center"),
              (THETA_VIB, "theta_vib (vibration), 1868 K", "tab:blue", "center"),
              (T_BATH, "bath T (1 eV), 11604 K", "k", "top"),
              (T_DISS, "dissociation, 19728 K", "tab:red", "bottom")]
    for Tv, lab, col, va in scales:
        c.hlines(Tv, 0.1, 0.9, color=col, lw=3)
        c.text(0.95, Tv, f" {lab}", va=va, fontsize=8, color=col)
    c.set_yscale("log"); c.set_ylim(1, 1e5); c.set_xlim(0, 2.2)
    c.set_xticks([])
    c.set_ylabel("temperature [K]")
    c.set_title("which modes are thermally active\n(rotation on, vibration partial, no dissoc.)")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    taus = relaxation_times()
    report(taus)
    make_figure(taus)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The separation of timescales (panel 1) is the point: translation equilibrates
#   in a few collisions, rotation in ~10, vibration in thousands (Z_VIB). During
#   the gap the gas is genuinely multi-temperature -- T_vib >> T_rot ~ T_tr is a
#   real, measurable non-equilibrium state.
#
# * Raise N_BATH (panel 2): every relaxation time scales as 1/n, so a denser
#   plasma thermalizes the molecule proportionally faster -- but the *ordering*
#   never changes.
#
# * Push T_BATH toward the dissociation temperature and the Boltzmann-tail
#   dissociation fraction climbs; at ~1 eV CaH mostly survives (kT < D0).
#
# Interpretation / simplifications: this is the relaxation-time (Landau-Teller-
# style) picture with constant collision numbers Z. Real V-T rates are strongly
# temperature dependent (tau_vib ~ exp(const*T^{-1/3})), rotation-vibration
# coupling and anharmonicity are ignored, the bath is infinite (fixed T_bath),
# and CaH is treated as neutral (if ionized to CaH+, translational thermalization
# would run on the much faster Coulomb-collision clock instead). Swap those in to
# specialise it to a particular experiment.
# ----------------------------------------------------------------------------
