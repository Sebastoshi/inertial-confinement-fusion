"""
Thin-shell "rocket" model of an ICF capsule implosion.

The outer surface of the capsule (the ablator) is heated by the laser/X-ray
drive and blows off. Like a rocket exhaust, the ablated material carries
momentum outward and the reaction pushes the remaining shell inward. The shell
accelerates to a huge implosion velocity, coasts, then stagnates on the central
gas -- converting its kinetic energy into a hot, dense core.

This is a *toy* lumped model: the shell is a single spherical mass m(t) at
radius r(t), driven by an ablation pressure and decelerated by an adiabatic
central gas cushion. It reproduces the three classic phases and the headline
scalings, and it hands you the stagnation (T, rho*R) so you can check it against
the ignition bar from the 0-D hot-spot model.

Phases the run walks through:
  1. Acceleration -- drive on: pressure pushes in, mass ablates (rocket eq.).
  2. Coast        -- drive off: shell flies inward at ~constant velocity.
  3. Stagnation   -- central gas stiffens (P ~ r^-5), decelerates shell to v=0
                     at peak compression. Read off convergence, rho*R, T.

Outputs:
  * console: implosion velocity, ablated fraction, kinetic energy, convergence
    ratio, stagnation rho*R and an estimated hot-spot temperature, each compared
    to the rocket-equation prediction and the ignition requirements.
  * figure: r(t), v(t), m(t), and the r-v trajectory.

Run:  python3 rocket_implosion.py
Deps: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
# Constants / units
# ----------------------------------------------------------------------------
MBAR_PA = 1.0e11            # 1 Mbar in Pascals
GAMMA   = 5.0 / 3.0         # ideal-gas index for the central gas cushion

# ----------------------------------------------------------------------------
# Capsule + drive parameters (NIF-scale toy values -- all knobs to play with)
# ----------------------------------------------------------------------------
R0       = 1.0e-3           # initial outer radius            [m]   (1 mm)
M0       = 3.0e-6           # initial shell mass              [kg]  (~3 mg)
P_DRIVE  = 200.0 * MBAR_PA  # ablation drive pressure         [Pa]  (200 Mbar)
V_EX     = 1.5e5            # ablation exhaust velocity       [m/s] (150 km/s)
M_FLOOR  = 0.10 * M0        # ablation stops here (payload = fuel that survives)

# The drive (and ablation) stay on as long as there is ablator left to burn,
# i.e. while m > M_FLOOR. That makes the acceleration end exactly where the
# rocket equation says it should, v_imp = v_ex * ln(M0/M_FLOOR).

# Central gas cushion (the DT vapor the shell implodes onto). Tuned so the shell
# stagnates at a realistic convergence ratio (~30).
P_GAS0   = 3.0e9            # initial central gas pressure    [Pa]

# Hot-spot ignition uses only a small central fraction of the fuel (that is the
# whole idea of "hot-spot" ignition -- you cannot heat all the fuel to keV).
HS_MASS_FRAC = 0.05         # fraction of the payload that forms the hot spot
ETA_HS       = 0.5          # fraction of shell KE that thermalizes the hot spot
M_ION_DT     = 2.5 * 1.66e-27   # mean DT ion mass [kg]

# Mass ablation rate per unit area is fixed by the rocket relation P = mdot*v_ex
MDOT_A   = P_DRIVE / V_EX   # [kg / m^2 / s]

# Ignition bar from the 0-D hot-spot model (for the closing comparison)
T_IGNITE_KEV  = 4.3        # ideal ignition temperature
RHO_R_IGNITE  = 0.30       # alpha-stopping areal density [g/cm^2]


# ----------------------------------------------------------------------------
# Equations of motion for the shell:  state y = [r, v, m]
#   dr/dt = v
#   m dv/dt = 4 pi r^2 (P_gas - P_abl)      (both pressures act on ~4 pi r^2)
#   dm/dt   = -mdot_a * 4 pi r^2            (only while drive is on & m>floor)
# Sign convention: v < 0 is imploding (r decreasing).
# ----------------------------------------------------------------------------
def p_gas(r):
    """Adiabatic central-gas pressure: P V^gamma = const, V ~ r^3 -> P ~ r^-5."""
    return P_GAS0 * (R0 / r) ** (3.0 * GAMMA)


def rhs(t, y):
    r, v, m = y
    r = max(r, 1e-6 * R0)                    # guard against r -> 0 blow-up
    A = 4.0 * np.pi * r ** 2

    driving = m > M_FLOOR                     # ablator remains -> drive is on
    P_abl = P_DRIVE if driving else 0.0
    dvdt  = A * (p_gas(r) - P_abl) / m        # inward when P_abl dominates
    drdt  = v
    dmdt  = -MDOT_A * A if driving else 0.0
    return [drdt, dvdt, dmdt]


def stagnation_event(t, y):
    """Fires at v = 0 (turning point = peak compression)."""
    return y[1]
stagnation_event.terminal = True
stagnation_event.direction = 1               # only catch v going negative->positive


def run_implosion():
    y0 = [R0, 0.0, M0]
    t_span = (0.0, 40.0e-9)                   # plenty of time to reach stagnation
    sol = solve_ivp(rhs, t_span, y0, method="LSODA",
                    events=stagnation_event, rtol=1e-8, atol=1e-12,
                    dense_output=True, max_step=5e-12)
    return sol


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def diagnostics(sol):
    t = np.linspace(sol.t[0], sol.t[-1], 3000)
    r, v, m = sol.sol(t)

    v_peak_idx = np.argmin(v)                 # most-negative velocity
    v_imp = -v[v_peak_idx]                    # implosion speed (positive) [m/s]
    # time the ablator runs out (mass first reaches the payload floor)
    below = np.where(m <= M_FLOOR * 1.001)[0]
    t_drive_off = t[below[0]] if below.size else None
    m_final = m[-1]
    r_min = r[-1]
    ke = 0.5 * m_final * v_imp ** 2           # coasting kinetic energy [J]
    cr = R0 / r_min                           # convergence ratio

    # rocket-equation cross-check: v = v_ex * ln(m0 / m_final)
    v_rocket = V_EX * np.log(M0 / m_final)

    # stagnation areal density of the compressed shell: rho*R ~ m/(4 pi r^2)
    rho_R_si = m_final / (4.0 * np.pi * r_min ** 2)      # kg/m^2
    rho_R = rho_R_si * 0.1                                # -> g/cm^2

    # hot-spot temperature: a fraction ETA_HS of the shell KE thermalizes a small
    # central hot spot (HS_MASS_FRAC of the fuel). Internal energy of a DT plasma
    # is 3 N k T (ions + electrons, 3/2 kT each), so kT = ETA*KE / (3 N).
    n_ions = HS_MASS_FRAC * m_final / M_ION_DT            # ions in the hot spot
    kT_J = ETA_HS * ke / (3.0 * n_ions)
    T_keV = kT_J / 1.602e-16

    return {"t": t, "r": r, "v": v, "m": m,
            "v_imp": v_imp, "v_rocket": v_rocket, "m_final": m_final,
            "r_min": r_min, "cr": cr, "ke": ke,
            "rho_R": rho_R, "T_keV": T_keV, "t_drive_off": t_drive_off,
            "t_stag": sol.t[-1] if sol.t_events[0].size else None}


def report(d):
    print("=" * 62)
    print("  THIN-SHELL ROCKET IMPLOSION  --  results")
    print("=" * 62)
    print(f"  drive pressure        : {P_DRIVE/MBAR_PA:8.1f} Mbar")
    print(f"  exhaust velocity      : {V_EX/1e3:8.1f} km/s")
    print("-" * 62)
    print(f"  implosion velocity    : {d['v_imp']/1e3:8.1f} km/s")
    print(f"    rocket-eq. predicts : {d['v_rocket']/1e3:8.1f} km/s  (cross-check)")
    print(f"  ablated fraction      : {(1-d['m_final']/M0)*100:8.1f} %")
    print(f"  payload (fuel) left   : {d['m_final']/M0*100:8.1f} %  of initial mass")
    print(f"  shell kinetic energy  : {d['ke']/1e3:8.2f} kJ")
    print(f"  convergence ratio     : {d['cr']:8.1f}  (R0/R_min)")
    print(f"  min radius            : {d['r_min']*1e6:8.1f} um")
    if d["t_stag"]:
        print(f"  time to stagnation    : {d['t_stag']*1e9:8.2f} ns")
    print("-" * 62)
    print("  stagnation vs the ignition bar (from the 0-D hot-spot model):")
    ok_rr = "PASS" if d["rho_R"] >= RHO_R_IGNITE else "fail"
    ok_T  = "PASS" if d["T_keV"] >= T_IGNITE_KEV else "fail"
    print(f"    areal density rho*R : {d['rho_R']:8.2f} g/cm^2  "
          f"(need >= {RHO_R_IGNITE})  [{ok_rr}]")
    print(f"    est. hot-spot temp  : {d['T_keV']:8.2f} keV     "
          f"(need >= {T_IGNITE_KEV})  [{ok_T}]")
    print("=" * 62)


def make_figure(d, fname="rocket_implosion.png"):
    import matplotlib.pyplot as plt
    t_ns = d["t"] * 1e9

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(t_ns, d["r"] * 1e3, "tab:blue", lw=2)
    if d["t_drive_off"]:
        ax[0, 0].axvline(d["t_drive_off"] * 1e9, ls="--", color="gray", lw=1)
        ax[0, 0].text(d["t_drive_off"] * 1e9, R0 * 1e3 * 0.9, " drive off",
                      color="gray", fontsize=9)
    ax[0, 0].set_xlabel("time [ns]"); ax[0, 0].set_ylabel("shell radius [mm]")
    ax[0, 0].set_title("radius: accelerate -> coast -> stagnate")

    ax[0, 1].plot(t_ns, -d["v"] / 1e3, "tab:red", lw=2)
    ax[0, 1].axhline(d["v_imp"] / 1e3, ls=":", color="k", lw=1)
    ax[0, 1].text(1, d["v_imp"] / 1e3 * 1.02, f"peak {d['v_imp']/1e3:.0f} km/s", fontsize=9)
    ax[0, 1].set_xlabel("time [ns]"); ax[0, 1].set_ylabel("implosion speed [km/s]")
    ax[0, 1].set_title("velocity: rocket-driven acceleration")

    ax[1, 0].plot(t_ns, d["m"] / M0 * 100, "tab:green", lw=2)
    ax[1, 0].axhline(M_FLOOR / M0 * 100, ls="--", color="gray", lw=1)
    ax[1, 0].text(1, M_FLOOR / M0 * 100 * 1.15, " payload floor", color="gray", fontsize=9)
    ax[1, 0].set_xlabel("time [ns]"); ax[1, 0].set_ylabel("remaining mass [% of M0]")
    ax[1, 0].set_title("mass: ablated away like rocket propellant")

    ax[1, 1].plot(d["r"] * 1e3, -d["v"] / 1e3, "tab:purple", lw=2)
    ax[1, 1].set_xlabel("shell radius [mm]"); ax[1, 1].set_ylabel("implosion speed [km/s]")
    ax[1, 1].set_title("trajectory in (r, v): implosion loop")
    ax[1, 1].invert_xaxis()                    # radius decreases left->right in time

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    sol = run_implosion()
    d = diagnostics(sol)
    report(d)
    make_figure(d)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The rocket equation  v_imp = v_ex * ln(M0/M_final)  is printed as a cross-
#   check next to the simulated velocity. Raise the drive time T_DRIVE (ablate
#   more mass) and both climb together -- more propellant burned = more speed,
#   at the cost of less payload left. That trade is the heart of the rocket model.
#
# * Hydrodynamic efficiency: shell KE / (drive work). Push P_DRIVE up and watch
#   velocity rise; in real ICF only ~5-15% of absorbed energy ends up as KE.
#
# * P_GAS0 sets how hard the central cushion is. Lower it and the shell converges
#   further (higher CR, higher rho*R) but the bounce gets more violent/stiff.
#
# * Connection to ignition: the run compares stagnation rho*R and T to the bar
#   from the 0-D hot-spot model. Try to find a (P_DRIVE, T_DRIVE, M0) combo that
#   clears BOTH -- that's the design problem ICF actually solves.
#
# Deliberate simplifications:
#   - single lumped shell (no thickness, no shell structure or shocks)
#   - ablation pressure constant during the pulse (real pulses are shaped)
#   - central gas is a lossless adiabatic spring (no radiation, no mix)
#   - the hot-spot temperature estimate is a rough KE-partition, not a solve;
#     it lands in the right ballpark (few keV) but don't trust the digits
#   - no Rayleigh-Taylor instability -- the thing that most limits real implosions
# ----------------------------------------------------------------------------
