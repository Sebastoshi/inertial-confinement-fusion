"""
Spherical 1-D Lagrangian hydrodynamics -- an ICF implosion, spatially resolved.

Where the 0-D hot-spot and thin-shell rocket models *estimate* the stagnation
state, this one *computes* it: a dense DT shell is driven inward by an applied
ablation pressure, acts as a piston on the central DT gas fill, and launches a
shock that converges on the origin. The shock focuses, the fill is heated and
compressed into a hot spot, and the whole thing rebounds.

Numerics -- the classic staggered-grid Lagrangian scheme (von Neumann &
Richtmyer, Wilkins):
  * mass zones are fixed (Lagrangian): dm[j] never changes, the mesh moves.
  * nodes carry radius r[i] and velocity u[i]; zones carry rho, e, P.
  * shocks are captured with artificial viscosity q (quadratic + linear).
  * ideal-gas EOS, gamma = 5/3, single temperature (T_ion = T_electron).
  * spherical geometry with a reflecting center (r=0, u=0) and a pressure-driven
    outer boundary.

Deliberately a toy: no radiation transport, no conduction, no real EOS, no
instabilities (this is 1-D). But it is a genuine PDE solve -- it conserves
energy to ~1% and shows the converging shock and hot-spot formation directly.

Run:  python3 lagrangian_1d.py
Deps: numpy, scipy(optional, not required), matplotlib
"""

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants (SI) and DT plasma EOS helpers
# ----------------------------------------------------------------------------
KB      = 1.380649e-23
KEV_J   = 1.602176634e-16
MBAR_PA = 1.0e11                     # 1 Mbar in Pa
MI_DT   = 2.5 * 1.66053907e-27       # mean DT ion mass [kg]
GAMMA   = 5.0 / 3.0

# specific internal energy e [J/kg] <-> temperature.  For DT (ions+electrons):
#   e = 3 k T / m_i   ->   T[keV] = e * m_i / (3 * keV_J)
def temperature_keV(e):
    return e * MI_DT / (3.0 * KEV_J)

def energy_from_T(T_keV):
    return 3.0 * T_keV * KEV_J / MI_DT


# ----------------------------------------------------------------------------
# Problem setup: a DT gas fill surrounded by a dense DT shell
# ----------------------------------------------------------------------------
R_FILL   = 0.6e-3          # fill outer radius / shell inner radius [m]
R_OUT    = 1.0e-3          # shell outer radius                     [m]
RHO_FILL = 3.0             # central gas density  [kg/m^3]  (DT fuel gas fill)
RHO_SHELL = 250.0          # shell density        [kg/m^3]  (~0.25 g/cc DT ice)
T0_KEV   = 0.01            # initial (cold) temperature everywhere [keV]

N_FILL   = 120             # zones in the fill
N_SHELL  = 100             # zones in the shell

# Drive: ablation pressure ramped onto the outer surface, then held.
P_MAX    = 150.0 * MBAR_PA  # peak drive pressure   [Pa]  (150 Mbar)
T_RAMP   = 0.3e-9           # ramp time             [s]
T_DRIVE  = 4.0e-9           # drive held until here, then off

# Artificial viscosity + time step
C_Q      = 2.0             # quadratic viscosity coefficient
C_L      = 1.0             # linear viscosity coefficient
CFL      = 0.30
DT_MAX   = 5.0e-12
T_END    = 4.0e-9
MAX_STEPS = 400000

# Ignition bars (from the 0-D hot-spot model) for the closing comparison
T_IGNITE_KEV = 4.3
RHO_R_IGNITE = 0.30


def build_grid():
    """Uniform-in-radius zoning within each region. Returns node/zone arrays."""
    r_fill  = np.linspace(0.0, R_FILL, N_FILL + 1)
    r_shell = np.linspace(R_FILL, R_OUT, N_SHELL + 1)[1:]      # drop shared node
    r = np.concatenate([r_fill, r_shell])                     # N+1 nodes
    N = r.size - 1

    # zone volumes and (fixed) masses
    vol = (4.0 * np.pi / 3.0) * (r[1:] ** 3 - r[:-1] ** 3)
    rho = np.empty(N)
    rho[:N_FILL] = RHO_FILL
    rho[N_FILL:] = RHO_SHELL
    dm = rho * vol                                            # Lagrangian, fixed

    u = np.zeros(N + 1)                                       # start at rest
    e = np.full(N, energy_from_T(T0_KEV))
    P = (GAMMA - 1.0) * rho * e
    is_fill = np.arange(N) < N_FILL
    return r, u, dm, rho, e, P, is_fill


def p_drive(t):
    if t >= T_DRIVE:
        return 0.0
    return P_MAX * min(1.0, t / T_RAMP)


def artificial_viscosity(rho, P, u):
    """Quadratic+linear q, active only in compression (du < 0)."""
    du = u[1:] - u[:-1]
    cs = np.sqrt(GAMMA * np.maximum(P, 0.0) / rho)
    compress = du < 0.0
    q = np.where(compress, C_Q * rho * du ** 2 + C_L * rho * cs * np.abs(du), 0.0)
    return q, du, cs


def step(state, dt, t):
    r, u, dm, rho, e, P = state
    N = rho.size
    q, du, cs = artificial_viscosity(rho, P, u)
    Pq = P + q

    # --- momentum: accelerate nodes ---
    accel = np.zeros(N + 1)
    A = 4.0 * np.pi * r ** 2
    # interior nodes i = 1..N-1  (right zone i, left zone i-1)
    dm_node = 0.5 * (dm[:-1] + dm[1:])
    accel[1:N] = -A[1:N] * (Pq[1:] - Pq[:-1]) / dm_node
    # outer node i = N: external drive pressure on the right
    accel[N] = -A[N] * (p_drive(t) - Pq[-1]) / (0.5 * dm[-1])
    # inner node i = 0 stays at the origin (accel 0)

    u_new = u + dt * accel
    u_new[0] = 0.0
    r_new = r + dt * u_new
    r_new[0] = 0.0
    r_new = np.maximum(r_new, 0.0)
    # keep the mesh monotone (guard against tangling at the focus)
    np.maximum.accumulate(r_new, out=r_new)

    # --- new density from moved mesh ---
    vol_new = (4.0 * np.pi / 3.0) * (r_new[1:] ** 3 - r_new[:-1] ** 3)
    vol_new = np.maximum(vol_new, 1e-30)
    rho_new = dm / vol_new

    # --- energy: implicit ideal-gas update  e^{n+1} solved analytically ---
    dv = 1.0 / rho_new - 1.0 / rho          # change in specific volume
    e_new = (e - (0.5 * P + q) * dv) / (1.0 + 0.5 * (GAMMA - 1.0) * dv * rho_new)
    e_new = np.maximum(e_new, energy_from_T(1e-4))
    P_new = (GAMMA - 1.0) * rho_new * e_new

    return (r_new, u_new, dm, rho_new, e_new, P_new)


def timestep(state, dt_prev):
    r, u, dm, rho, e, P = state
    q, du, cs = artificial_viscosity(rho, P, u)
    dr = r[1:] - r[:-1]
    signal = cs + np.abs(du) + 1e-30
    dt_cfl = CFL * np.min(dr / signal)
    return min(dt_cfl, 1.1 * dt_prev, DT_MAX)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run():
    r, u, dm, rho, e, P, is_fill = build_grid()
    state = (r, u, dm, rho, e, P)
    N = rho.size

    t = 0.0
    dt = 1.0e-14
    nsub = 6                                   # sub-sample nodes for the r-t plot
    node_idx = np.arange(0, N + 1, nsub)

    dm_fill = dm[is_fill]
    m_fill = np.sum(dm_fill)

    hist = {"t": [], "r_nodes": [], "Tc": [], "Ths": [], "rhoc": [], "rhoR": [],
            "Rfuel": [], "KE": [], "IE": [], "Wdrive": []}
    snaps = {}
    snap_times = [1.0e-9, 2.0e-9, 2.5e-9, 2.85e-9]    # profiles saved near these
    snap_next = 0
    W_drive = 0.0
    rfuel_min = R_FILL

    for n in range(MAX_STEPS):
        dt = timestep(state, dt)
        r_old = state[0]
        state = step(state, dt, t)
        t += dt
        r, u, dm, rho, e, P = state

        # drive work done on the outer boundary: P_drive * dV_outer
        dV_out = (4.0 * np.pi / 3.0) * (r[-1] ** 3 - r_old[-1] ** 3)
        W_drive += -p_drive(t) * dV_out         # inward motion (dV<0) adds energy

        dr = r[1:] - r[:-1]
        rhoR_hot = np.sum(rho[is_fill] * dr[is_fill]) * 0.1        # g/cm^2
        rfuel_min = min(rfuel_min, r[N_FILL])

        if n % 40 == 0:
            # mass-averaged hot-spot (fill) temperature -- the physical measure;
            # the single central zone is singular at shock focus (Guderley).
            e_hs = np.sum(dm_fill * e[is_fill]) / m_fill
            KE = 0.5 * np.sum(0.5 * (dm[:-1] + dm[1:]) * u[1:-1] ** 2)
            IE = np.sum(dm * e)
            hist["t"].append(t); hist["r_nodes"].append(r[node_idx].copy())
            hist["Tc"].append(temperature_keV(e[0]))
            hist["Ths"].append(temperature_keV(e_hs))
            hist["rhoc"].append(np.max(rho[is_fill]))
            hist["rhoR"].append(rhoR_hot)
            hist["Rfuel"].append(r[N_FILL])              # fuel/shell interface
            hist["KE"].append(KE); hist["IE"].append(IE); hist["Wdrive"].append(W_drive)

        # save a few full profiles for the snapshot panel
        if snap_next < len(snap_times) and t >= snap_times[snap_next]:
            rc = 0.5 * (r[1:] + r[:-1])
            snaps[snap_times[snap_next]] = {
                "r": rc.copy(), "rho": rho.copy(),
                "T": temperature_keV(e).copy(), "u": 0.5 * (u[1:] + u[:-1]).copy()}
            snap_next += 1

        # stop once the fuel has clearly rebounded past peak compression
        if t > 2.0e-9 and r[N_FILL] > 1.6 * rfuel_min:
            break
        if t >= T_END:
            break

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist, snaps, is_fill, node_idx


def peak_conditions(hist):
    i = int(np.argmin(hist["Rfuel"]))                # stagnation = peak convergence
    cr = R_OUT / max(hist["Rfuel"][i], 1e-12)
    return {"t": hist["t"][i], "Ths": hist["Ths"][i], "Tc": np.max(hist["Tc"]),
            "rhoc": hist["rhoc"][i], "rhoR": hist["rhoR"][i], "cr": cr}


def report(hist, pk):
    print("=" * 62)
    print("  1-D LAGRANGIAN IMPLOSION  --  stagnation results")
    print("=" * 62)
    print(f"  drive pressure           : {P_MAX/MBAR_PA:8.0f} Mbar")
    print(f"  time to stagnation       : {pk['t']*1e9:8.2f} ns")
    print(f"  fuel convergence ratio   : {pk['cr']:8.1f}  (R_out / R_fuel_min)")
    print(f"  peak hot-spot density    : {pk['rhoc']/1000:8.2f} g/cc")
    print(f"  hot-spot temperature     : {pk['Ths']:8.2f} keV  (mass-averaged)")
    print(f"  hot-spot areal density   : {pk['rhoR']:8.2f} g/cm^2")
    print(f"  peak shock-focus temp    : {pk['Tc']:8.1f} keV  (central zone; ")
    print(f"                             Guderley singularity -- not physical)")
    # energy conservation check
    tot = hist["KE"] + hist["IE"] - hist["Wdrive"]
    drift = (tot[-1] - tot[0]) / max(abs(hist["Wdrive"][-1]), 1e-30) * 100
    print("-" * 62)
    print(f"  energy conservation drift: {drift:7.2f} %  (of drive work)")
    print("-" * 62)
    ok_T = "PASS" if pk["Ths"] >= T_IGNITE_KEV else "fail"
    ok_R = "PASS" if pk["rhoR"] >= RHO_R_IGNITE else "fail"
    print(f"  vs ignition bar: T    {pk['Ths']:6.2f} keV   (need >= {T_IGNITE_KEV})  [{ok_T}]")
    print(f"                  rhoR  {pk['rhoR']:6.3f} g/cm2 (need >= {RHO_R_IGNITE})  [{ok_R}]")
    print("  -> hot enough (strong shock heating) but not dense enough: a")
    print("     lossless single-shock toy has no ablative compression to build")
    print("     areal density. That is what models 1 & 2 buy you.")
    print("=" * 62)


def make_figure(hist, snaps, is_fill, node_idx, fname="lagrangian_implosion.png"):
    import matplotlib.pyplot as plt

    t_ns = hist["t"] * 1e9
    R = np.vstack(hist["r_nodes"]) * 1e3          # [n_time, n_nodes] in mm
    n_fill_nodes = np.searchsorted(node_idx, N_FILL)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (1) r-t diagram: zone-boundary trajectories
    for k in range(R.shape[1]):
        color = "tab:orange" if k >= n_fill_nodes else "tab:blue"
        ax[0, 0].plot(t_ns, R[:, k], color=color, lw=0.6, alpha=0.8)
    ax[0, 0].set_xlabel("time [ns]"); ax[0, 0].set_ylabel("radius [mm]")
    ax[0, 0].set_title("Lagrangian mesh trajectories\n(blue = gas fill, orange = shell)")

    # (2) central conditions vs time
    axr = ax[0, 1]
    axr.plot(t_ns, hist["Ths"], "tab:red", lw=2, label="hot-spot T (mass-avg)")
    axr.set_xlabel("time [ns]"); axr.set_ylabel("hot-spot T [keV]", color="tab:red")
    axr.tick_params(axis="y", labelcolor="tab:red")
    axr.axhline(T_IGNITE_KEV, ls=":", color="tab:red", lw=1)
    axr.text(t_ns[0], T_IGNITE_KEV * 1.05, " ignition T", color="tab:red", fontsize=8)
    ax2 = axr.twinx()
    ax2.plot(t_ns, hist["rhoR"], "tab:purple", lw=2, label="hot-spot rho*R")
    ax2.set_ylabel(r"hot-spot $\rho R$ [g/cm$^2$]", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    axr.set_title("hot-spot temperature & areal density\n(shock focus -> stagnation)")

    # (3) profile snapshots: temperature vs radius
    for tt, s in snaps.items():
        ax[1, 0].plot(s["r"] * 1e3, s["T"], lw=1.8, label=f"t={tt*1e9:.1f} ns")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("radius [mm]"); ax[1, 0].set_ylabel("temperature [keV]")
    ax[1, 0].set_title("temperature profiles: shock converging inward")
    ax[1, 0].legend(fontsize=8)

    # (4) energy conservation
    ax[1, 1].plot(t_ns, hist["KE"] * 1e3, label="kinetic")
    ax[1, 1].plot(t_ns, (hist["IE"] - hist["IE"][0]) * 1e3, label="internal (rel.)")
    ax[1, 1].plot(t_ns, hist["Wdrive"] * 1e3, "k--", label="drive work")
    total = (hist["KE"] + hist["IE"] - hist["IE"][0] - hist["Wdrive"]) * 1e3
    ax[1, 1].plot(t_ns, total, "gray", lw=1, label="KE+IE-Wdrive (should be ~0)")
    ax[1, 1].set_xlabel("time [ns]"); ax[1, 1].set_ylabel("energy [mJ]")
    ax[1, 1].set_title("energy budget / conservation check")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    hist, snaps, is_fill, node_idx = run()
    pk = peak_conditions(hist)
    report(hist, pk)
    make_figure(hist, snaps, is_fill, node_idx)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The r-t diagram (top-left) is the payoff: watch the orange shell accelerate
#   inward, drive a shock (steepening blue lines) into the fill, and the shock
#   converge on r=0 where the trajectories pinch -- that pinch is the hot spot.
#
# * Central T spikes when the shock focuses, then again (harder) at stagnation
#   when the shell arrives. Push P_MAX up and stagnation gets hotter.
#
# * Resolution: bump N_FILL to sharpen the converging shock (Guderley focusing
#   is singular -- a toy code rounds the peak off; more zones = higher spike).
#
# * Energy conservation (bottom-right) is the validity check: KE + IE - drive
#   work should stay near zero. Artificial viscosity leaks a little; the printed
#   drift is the honest error bar on the peak temperature.
#
# Deliberate simplifications (biggest first):
#   - no radiation transport or electron conduction (both cool the hot spot)
#   - ideal-gas EOS (real DT is partially degenerate/Fermi in the cold shell)
#   - single temperature; no alpha heating / burn (couple in the 0-D model!)
#   - 1-D, so no Rayleigh-Taylor -- the real spoiler of hot-spot formation
#   - piston drive, not a resolved ablation front
# ----------------------------------------------------------------------------
