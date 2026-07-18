"""
0-D hot-spot ignition model for inertial confinement fusion (ICF).

This is a *toy* model meant to build intuition, not a design code. It captures
the three effects that decide whether a DT hot spot ignites:

  1. Alpha self-heating   -- 3.5 MeV alphas from D+T -> He4 + n redeposit their
                             energy in the hot spot (positive feedback -> runaway).
  2. Bremsstrahlung loss  -- the plasma radiates ~ n_e^2 * sqrt(T) (a sink).
  3. Alpha confinement    -- alphas only deposit locally if the areal density
                             rho*R exceeds their stopping range (~0.3 g/cm^2).

Two things fall out that are the *whole point* of ICF:
  * an ignition TEMPERATURE (~4-5 keV) where alpha heating overtakes radiation, and
  * an ignition AREAL DENSITY (rho*R ~ 0.3 g/cm^2) so the alphas actually stop.

The script produces:
  Part 1 - the static ignition curve in the (T, rho*R) plane.
  Part 2 - time-integrated burns showing thermal runaway ("ignite") vs decay
           ("fizzle"), and a scan of initial temperature that reveals the cliff.

Physics simplifications (all deliberate -- see NOTES at the bottom):
  * single temperature, T_ion = T_electron
  * optically thin (all bremsstrahlung escapes)
  * electron heat conduction neglected
  * isochoric burn over a fixed inertial-confinement window
  * simple rho*R alpha-deposition fit with a tunable half-deposition scale

Run:  python3 hotspot_0d.py
Deps: numpy, scipy, matplotlib
"""

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------------
KB      = 1.380649e-23      # Boltzmann constant [J/K]
KEV_J   = 1.602176634e-16   # 1 keV in Joules
E_ALPHA = 3.5e6 * 1.602176634e-19   # alpha energy, 3.5 MeV [J]
MBAR    = (2.014 + 3.016) / 2 * 1.66053907e-27   # mean DT ion mass [kg]
GAMMA   = 5.0 / 3.0         # ideal-gas adiabatic index

# Bremsstrahlung coefficient in SI with T in keV:
#   P_brem = C_BREM * n_e^2 * sqrt(T_keV)   [W/m^3]
# (derived from the NRL Plasma Formulary expression for hydrogenic DT, Z=1)
C_BREM = 5.34e-37

# Alpha deposition: fraction that stops locally rises with areal density.
# Toy fit f = x / (x + x_half) with x = rho*R [g/cm^2]. Half-deposition at the
# canonical alpha range ~0.3 g/cm^2. (Real deposition is T-dependent; see NOTES.)
RHO_R_HALF = 0.30          # g/cm^2


# ----------------------------------------------------------------------------
# DT fusion reactivity  <sigma v>  -- Bosch & Hale (1992) parametrization
# Accurate for T = 0.2 .. 100 keV. Returns m^3/s. Input T in keV.
# ----------------------------------------------------------------------------
def reactivity_dt(T_keV):
    T = np.asarray(T_keV, dtype=float)
    T = np.clip(T, 0.2, 100.0)           # stay inside the fit's validity range
    BG   = 34.3827
    MRC2 = 1.124656e6
    C1, C2, C3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
    C4, C5, C6, C7 = 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5

    theta = T / (1.0 - (T * (C2 + T * (C4 + T * C6)))
                     / (1.0 + T * (C3 + T * (C5 + T * C7))))
    xi = (BG ** 2 / (4.0 * theta)) ** (1.0 / 3.0)
    sv_cm3 = C1 * theta * np.sqrt(xi / (MRC2 * T ** 3)) * np.exp(-3.0 * xi)
    return sv_cm3 * 1e-6                  # cm^3/s -> m^3/s


def f_alpha(rho_R):
    """Fraction of alpha energy deposited in the hot spot vs areal density [g/cm^2]."""
    return rho_R / (rho_R + RHO_R_HALF)


# ----------------------------------------------------------------------------
# Power densities [W/m^3].  n_i = ion density [m^-3]; n_e = n_i for DT.
# For equimolar DT, n_D = n_T = n_i/2, so the fusion rate is n_i^2/4 * <sv>.
# ----------------------------------------------------------------------------
def p_alpha(n_i, T_keV, rho_R):
    """Alpha self-heating power density."""
    rate = 0.25 * n_i ** 2 * reactivity_dt(T_keV)      # reactions / m^3 / s
    return f_alpha(rho_R) * rate * E_ALPHA


def p_brem(n_i, T_keV):
    """Bremsstrahlung radiative loss power density (optically thin)."""
    return C_BREM * n_i ** 2 * np.sqrt(T_keV)


def sound_speed(T_keV):
    """Ideal-gas sound speed of the DT plasma [m/s] (ions+electrons)."""
    # P = 2 n_i kT, rho = n_i * MBAR  ->  c_s = sqrt(gamma * 2 kT / MBAR)
    return np.sqrt(GAMMA * 2.0 * (T_keV * KEV_J) / MBAR)


# ============================================================================
# PART 1 -- Static ignition curve
# The net power per unit n_i^2 is independent of density, so the ignition
# boundary  P_alpha = P_brem  is a clean curve in the (T, rho*R) plane.
# ============================================================================
def ignition_curve(rho_R_grid):
    """For each rho*R, find the temperature where alpha heating = bremsstrahlung."""
    T_scan = np.linspace(0.5, 50.0, 4000)
    T_ig = np.full_like(rho_R_grid, np.nan)
    for i, rr in enumerate(rho_R_grid):
        # net heating per n_i^2 (density cancels):
        net = 0.25 * reactivity_dt(T_scan) * f_alpha(rr) * E_ALPHA \
              - C_BREM * np.sqrt(T_scan)
        pos = np.where(net > 0)[0]
        if pos.size:
            T_ig[i] = T_scan[pos[0]]     # lowest T that self-heats
    return T_ig


# ============================================================================
# PART 2 -- Dynamic isochoric burn
# State: y = [T_keV, n_i].  Fixed volume & radius; alpha heating, brems loss,
# and fuel depletion. Integrated over one inertial-confinement time tau_c.
# ============================================================================
def make_rhs(rho_R):
    def rhs(t, y):
        T, n_i = y
        T   = max(T, 0.2)
        n_i = max(n_i, 0.0)

        heat = p_alpha(n_i, T, rho_R) - p_brem(n_i, T)   # W/m^3
        # internal energy density u = 3 n_i kT (ions+electrons, 3/2 kT each)
        # d/dt (3 n_i k T) = heat ; with n_i also changing from burn-up:
        rate = 0.25 * n_i ** 2 * reactivity_dt(T)        # reactions/m^3/s
        dn_i = -2.0 * rate                               # each reaction burns 1 D +1 T
        # dT from energy balance, accounting for changing n_i:
        dT = (heat - 3.0 * (T * KEV_J) * dn_i) / (3.0 * n_i * KB) / (KEV_J / KB) \
             if n_i > 0 else 0.0
        # (the KEV_J/KB juggling keeps T in keV; see NOTES)
        return [dT, dn_i]
    return rhs


def burn(rho, rho_R, T0_keV, n_tau=1.0):
    """Integrate one hot spot. rho [g/cm^3], rho_R [g/cm^2], T0 [keV].

    Returns dict with time series and the burn-up fraction.
    """
    from scipy.integrate import solve_ivp

    rho_si = rho * 1000.0                       # g/cm^3 -> kg/m^3
    n_i0   = rho_si / MBAR                       # ion density [m^-3]
    R      = (rho_R * 10.0) / rho_si             # radius [m]  (rho_R g/cm^2 ->kg/m^2)
    tau_c  = n_tau * R / (4.0 * sound_speed(T0_keV))  # inertial confinement time [s]

    sol = solve_ivp(make_rhs(rho_R), (0.0, tau_c), [T0_keV, n_i0],
                    method="LSODA", rtol=1e-6, atol=1e-3,
                    dense_output=True, max_step=tau_c / 200.0)

    t  = np.linspace(0.0, tau_c, 400)
    ys = sol.sol(t)
    burnup = 1.0 - ys[1, -1] / n_i0
    return {"t": t, "T": ys[0], "n_i": ys[1], "tau_c": tau_c,
            "burnup": burnup, "T_final": ys[0, -1]}


# ----------------------------------------------------------------------------
# Driver: make the figures and print the headline numbers.
# ----------------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt

    # ---- Part 1: ignition curve --------------------------------------------
    rr_grid = np.linspace(0.05, 3.0, 300)
    T_ig = ignition_curve(rr_grid)

    # ---- Part 2a: two representative burns (same rho*R, different T0) -------
    rho    = 300.0     # g/cm^3, a compressed hot-spot density
    rho_R  = 1.0       # g/cm^2, comfortably above the alpha-stopping threshold
    fizz = burn(rho, rho_R, T0_keV=4.0)     # below the ignition temperature
    ign  = burn(rho, rho_R, T0_keV=6.0)     # above it -> thermal runaway

    # ---- Part 2b: temperature cliff (scan T0) ------------------------------
    T0_scan = np.linspace(2.0, 10.0, 60)
    burnups = np.array([burn(rho, rho_R, t0)["burnup"] for t0 in T0_scan])

    # ---- console summary ----------------------------------------------------
    print(f"Alpha-stopping half-deposition scale : rho*R = {RHO_R_HALF:.2f} g/cm^2")
    print(f"Ideal ignition temperature (rho*R>>1): "
          f"{ignition_curve(np.array([10.0]))[0]:.2f} keV")
    print(f"Fizzle  (T0=4 keV): T_final={fizz['T_final']:6.2f} keV  "
          f"burn-up={fizz['burnup']*100:5.2f}%  tau_c={fizz['tau_c']*1e12:.1f} ps")
    print(f"Ignite  (T0=6 keV): T_final={ign['T_final']:6.2f} keV  "
          f"burn-up={ign['burnup']*100:5.2f}%  tau_c={ign['tau_c']*1e12:.1f} ps")

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(T_ig, rr_grid, "k-", lw=2)
    ax[0].fill_betweenx(rr_grid, T_ig, 50, color="tab:green", alpha=0.15)
    ax[0].fill_betweenx(rr_grid, 0, T_ig, color="tab:red", alpha=0.10)
    ax[0].text(18, 2.2, "IGNITE\n(self-heating)", color="darkgreen", ha="center")
    ax[0].text(3.0, 0.4, "FIZZLE", color="darkred", ha="center")
    ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 3)
    ax[0].set_xlabel("hot-spot temperature  T  [keV]")
    ax[0].set_ylabel(r"areal density  $\rho R$  [g/cm$^2$]")
    ax[0].set_title("Part 1: ignition curve\n(alpha heating = bremsstrahlung)")

    ax[1].plot(fizz["t"] * 1e12, fizz["T"], "tab:red",  lw=2, label="T0=4 keV -> fizzle")
    ax[1].plot(ign["t"]  * 1e12, ign["T"],  "tab:green", lw=2, label="T0=6 keV -> ignite")
    ax[1].set_xlabel("time  [ps]")
    ax[1].set_ylabel("temperature  [keV]")
    ax[1].set_title(f"Part 2: burn dynamics\n(rho={rho:.0f} g/cc, rho*R={rho_R} g/cm2)")
    ax[1].legend()

    ax[2].plot(T0_scan, burnups * 100, "o-", ms=3, color="tab:blue")
    ax[2].set_xlabel("initial temperature  T0  [keV]")
    ax[2].set_ylabel("burn-up fraction  [%]")
    ax[2].set_title("Part 2: the ignition cliff\n(outcome vs starting temperature)")

    fig.tight_layout()
    out = "hotspot_ignition.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try (the fun part) --------------------------------------
#
# * Move RHO_R_HALF up/down: watch the ignition curve's rho*R floor shift. This
#   is why ICF chases high areal density -- the alphas have to stop.
#
# * In main(), lower `rho_R` toward 0.2 g/cm^2 and the T=6 keV shot fizzles even
#   though it's "hot enough" -- alpha confinement, not temperature, kills it.
#
# * The cliff plot (Part 2b) is the whole story: burn-up is ~flat and tiny, then
#   jumps by orders of magnitude over a ~1 keV window. That steepness is what
#   makes ICF a threshold phenomenon and so hard to hit.
#
# * n_tau scales the confinement window (disassembly is ~R/(3-4 c_s)); raising it
#   lets marginal shots catch up and ignite. This is the "burn width" knob.
#
# Deliberate simplifications, roughly in order of how much they'd move numbers:
#   - single T (real hot spots have T_ion != T_electron during heating)
#   - optically thin brems (some reabsorption at high rho*R softens the loss)
#   - no electron heat conduction (a real loss that raises the ignition threshold)
#   - fixed confinement window (real disassembly shortens tau_c as T runs away,
#     and the surrounding dense shell provides extra inertial tamping)
#   - simple f_alpha(rho*R); the true range scales ~ T^1.5 and the geometry
#     factor differs. Swap in a Bosch-Hale-quality stopping model to sharpen it.
# ----------------------------------------------------------------------------
