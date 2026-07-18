"""
Verification of the 1-D Lagrangian hydro scheme against exact solutions.

lagrangian_1d.py conserves energy to ~1% on an ICF implosion -- necessary but
not sufficient. This script runs the *same numerical scheme* (generalized to
planar/spherical geometry and arbitrary initial conditions) on the three
canonical hydrodynamics verification problems and overlays the exact answers:

  * Sod shock tube (planar)  -- exact Riemann solution. Tests a shock, a contact
    discontinuity and a rarefaction fan all at once.
  * Noh problem (spherical)  -- exact analytic solution. A cold uniform inflow
    stagnates into a converging shock; notoriously hard, directly ICF-relevant.
  * Sedov blast (spherical)  -- exact self-similar invariants: the strong-shock
    compression ratio (g+1)/(g-1) and the R_shock ~ t^(2/5) scaling law.

If the scheme reproduces these, the ICF results downstream (and any ML trained
on them) rest on a verified solver rather than a plausible-looking one.

Run:  python3 hydro_validation.py
Deps: numpy, matplotlib
"""

import numpy as np

# ============================================================================
# General 1-D Lagrangian hydro (same scheme as lagrangian_1d.py, generalized)
# ============================================================================
def zone_volume(r, geom):
    if geom == "planar":
        return np.diff(r)
    if geom == "cylindrical":
        return np.pi * np.diff(r ** 2)
    return (4.0 * np.pi / 3.0) * np.diff(r ** 3)      # spherical

def node_area(r, geom):
    if geom == "planar":
        return np.ones_like(r)
    if geom == "cylindrical":
        return 2.0 * np.pi * r
    return 4.0 * np.pi * r ** 2                        # spherical


def solve(r, u, rho, e, gamma, geom, t_end,
          inner="reflect", outer="wall", outer_val=0.0,
          cfl=0.4, cq=1.5, cl=1.0, e_floor=1e-12, monitor_times=None):
    """Advance to t_end. Returns final (r, u, rho, e, P) and optional snapshots."""
    r = r.astype(float).copy(); u = u.astype(float).copy()
    rho = rho.astype(float).copy(); e = e.astype(float).copy()
    N = rho.size
    dm = rho * zone_volume(r, geom)                   # fixed Lagrangian masses
    t = 0.0
    dt = 1e-9
    snaps = {}
    mon = sorted(monitor_times) if monitor_times else []
    mi = 0

    while t < t_end:
        P = (gamma - 1.0) * rho * e
        cs = np.sqrt(np.maximum(gamma * P / rho, 0.0))
        du = u[1:] - u[:-1]
        q = np.where(du < 0.0, cq * rho * du ** 2 + cl * rho * cs * np.abs(du), 0.0)
        Pq = P + q

        dr = np.diff(r)
        dt = cfl * np.min(dr / (cs + np.abs(du) + 1e-30))
        dt = min(dt, 1.1 * dt if t > 0 else dt, t_end - t)

        A = node_area(r, geom)
        accel = np.zeros(N + 1)
        dm_node = 0.5 * (dm[:-1] + dm[1:])
        accel[1:N] = -A[1:N] * (Pq[1:] - Pq[:-1]) / dm_node
        if outer == "free":
            accel[N] = -A[N] * (0.0 - Pq[-1]) / (0.5 * dm[-1])
        # (wall/inflow outer handled by fixing velocity below)

        u_new = u + dt * accel
        u_new[0] = 0.0                                # inner reflecting wall / center
        if outer == "wall":
            u_new[N] = 0.0
        elif outer == "inflow":
            u_new[N] = outer_val

        r_new = r + dt * u_new
        r_new = np.maximum.accumulate(np.maximum(r_new, 0.0))

        vol_new = np.maximum(zone_volume(r_new, geom), 1e-30)
        rho_new = dm / vol_new
        dv = 1.0 / rho_new - 1.0 / rho
        e_new = (e - (0.5 * P + q) * dv) / (1.0 + 0.5 * (gamma - 1.0) * dv * rho_new)
        e_new = np.maximum(e_new, e_floor)

        r, u, rho, e = r_new, u_new, rho_new, e_new
        t += dt
        while mi < len(mon) and t >= mon[mi]:
            snaps[mon[mi]] = (r.copy(), rho.copy())
            mi += 1

    P = (gamma - 1.0) * rho * e
    return (r, u, rho, e, P), snaps


def zone_centers(r):
    return 0.5 * (r[1:] + r[:-1])


# ============================================================================
# Exact Riemann solver (Sod)  -- Toro, "Riemann Solvers and Numerical Methods"
# ============================================================================
def _f_and_df(p, rhoK, pK, aK, g):
    if p > pK:                                        # shock
        A = 2.0 / ((g + 1.0) * rhoK)
        B = (g - 1.0) / (g + 1.0) * pK
        f = (p - pK) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (B + p)) * (1.0 - (p - pK) / (2.0 * (B + p)))
    else:                                             # rarefaction
        f = 2.0 * aK / (g - 1.0) * ((p / pK) ** ((g - 1.0) / (2.0 * g)) - 1.0)
        df = (1.0 / (rhoK * aK)) * (p / pK) ** (-(g + 1.0) / (2.0 * g))
    return f, df


def riemann_exact(x, t, x0, WL, WR, g):
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL = np.sqrt(g * pL / rhoL); aR = np.sqrt(g * pR / rhoR)

    p = 0.5 * (pL + pR)                               # Newton iteration for p*
    for _ in range(100):
        fL, dfL = _f_and_df(p, rhoL, pL, aL, g)
        fR, dfR = _f_and_df(p, rhoR, pR, aR, g)
        f = fL + fR + (uR - uL)
        p_new = max(1e-12, p - f / (dfL + dfR))
        if abs(p_new - p) < 1e-10 * p:
            break
        p = p_new
    pstar = p
    fL, _ = _f_and_df(pstar, rhoL, pL, aL, g)
    fR, _ = _f_and_df(pstar, rhoR, pR, aR, g)
    ustar = 0.5 * (uL + uR) + 0.5 * (fR - fL)

    def sample(S):
        if S <= ustar:                                # left of contact
            if pstar > pL:                            # left shock
                SL = uL - aL * np.sqrt((g + 1) / (2 * g) * pstar / pL + (g - 1) / (2 * g))
                if S <= SL:
                    return rhoL, uL, pL
                rho = rhoL * (pstar / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pL + 1)
                return rho, ustar, pstar
            else:                                     # left rarefaction
                aLs = aL * (pstar / pL) ** ((g - 1) / (2 * g))
                SHL = uL - aL; STL = ustar - aLs
                if S <= SHL:
                    return rhoL, uL, pL
                if S >= STL:
                    return rhoL * (pstar / pL) ** (1 / g), ustar, pstar
                u = 2 / (g + 1) * (aL + (g - 1) / 2 * uL + S)
                a = 2 / (g + 1) * (aL + (g - 1) / 2 * (uL - S))
                return rhoL * (a / aL) ** (2 / (g - 1)), u, pL * (a / aL) ** (2 * g / (g - 1))
        else:                                         # right of contact
            if pstar > pR:                            # right shock
                SR = uR + aR * np.sqrt((g + 1) / (2 * g) * pstar / pR + (g - 1) / (2 * g))
                if S >= SR:
                    return rhoR, uR, pR
                rho = rhoR * (pstar / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * pstar / pR + 1)
                return rho, ustar, pstar
            else:                                     # right rarefaction
                aRs = aR * (pstar / pR) ** ((g - 1) / (2 * g))
                SHR = uR + aR; STR = ustar + aRs
                if S >= SHR:
                    return rhoR, uR, pR
                if S <= STR:
                    return rhoR * (pstar / pR) ** (1 / g), ustar, pstar
                u = 2 / (g + 1) * (-aR + (g - 1) / 2 * uR + S)
                a = 2 / (g + 1) * (aR - (g - 1) / 2 * (uR - S))
                return rhoR * (a / aR) ** (2 / (g - 1)), u, pR * (a / aR) ** (2 * g / (g - 1))

    rho = np.empty_like(x); u = np.empty_like(x); p = np.empty_like(x)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i] = sample((xi - x0) / t)
    return rho, u, p


# ============================================================================
# Test problems
# ============================================================================
def run_sod():
    g, N, t_end = 1.4, 400, 0.2
    r = np.linspace(0.0, 1.0, N + 1)
    rc = zone_centers(r)
    rho = np.where(rc < 0.5, 1.0, 0.125)
    P = np.where(rc < 0.5, 1.0, 0.1)
    u = np.zeros(N + 1)
    e = P / ((g - 1.0) * rho)
    (rf, uf, rhof, ef, Pf), _ = solve(r, u, rho, e, g, "planar", t_end,
                                      inner="reflect", outer="wall")
    xc = zone_centers(rf)
    rho_ex, u_ex, P_ex = riemann_exact(xc, t_end, 0.5,
                                       (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), g)
    l1 = np.mean(np.abs(rhof - rho_ex))
    return dict(name="Sod shock tube (planar)", x=xc, rho=rhof, u=uf, P=Pf,
                rho_ex=rho_ex, u_ex=u_ex, P_ex=P_ex, g=g, t=t_end, l1=l1)


def run_noh():
    g, N, t_end = 5.0 / 3.0, 400, 0.6
    r = np.linspace(0.0, 1.0, N + 1)
    rho = np.ones(N)
    u = -np.ones(N + 1); u[0] = 0.0
    P0 = 1e-6
    e = np.full(N, P0 / ((g - 1.0) * 1.0))
    (rf, uf, rhof, ef, Pf), _ = solve(r, u, rho, e, g, "spherical", t_end,
                                      inner="reflect", outer="inflow", outer_val=-1.0,
                                      cq=2.0, cl=1.0)
    xc = zone_centers(rf)
    r_shock = t_end / 3.0                              # exact shock position
    rho_ex = np.where(xc < r_shock, 64.0, (1.0 + t_end / np.maximum(xc, 1e-9)) ** 2)
    # post-shock plateau (away from the central wall-heating hole)
    plateau = (xc > 0.4 * r_shock) & (xc < 0.9 * r_shock)
    rho_plateau = np.median(rhof[plateau])
    fair = xc > 0.03                                   # exclude the known central artifact
    l1 = np.mean(np.abs(rhof - rho_ex)[fair])
    return dict(name="Noh problem (spherical)", x=xc, rho=rhof, u=uf, P=Pf,
                rho_ex=rho_ex, r_shock=r_shock, rho_plateau=rho_plateau,
                g=g, t=t_end, l1=l1)


def run_sedov():
    g, N, t_end = 1.4, 600, 0.05
    r = np.linspace(0.0, 1.0, N + 1)
    rho = np.ones(N)
    u = np.zeros(N + 1)
    P0 = 1e-6
    e = np.full(N, P0 / ((g - 1.0) * 1.0))
    dm0 = rho[0] * zone_volume(r, "spherical")[0]
    E0 = 1.0
    e[0] = E0 / dm0                                    # point blast in central zone
    mon = list(np.linspace(0.2, 1.0, 9) * t_end)
    (rf, uf, rhof, ef, Pf), snaps = solve(r, u, rho, e, g, "spherical", t_end,
                                          inner="reflect", outer="wall", cq=2.0, cl=1.0,
                                          monitor_times=mon)
    xc = zone_centers(rf)
    # shock radius over time = location of peak density
    ts, Rs = [], []
    for tt in mon:
        rr, dd = snaps[tt]
        c = zone_centers(rr)
        ts.append(tt); Rs.append(c[np.argmax(dd)])
    ts, Rs = np.array(ts), np.array(Rs)
    slope = np.polyfit(np.log(ts), np.log(Rs), 1)[0]   # expect 2/5 = 0.4
    compression = rhof.max() / 1.0                      # expect (g+1)/(g-1) = 6
    return dict(name="Sedov blast (spherical)", x=xc, rho=rhof, P=Pf,
                ts=ts, Rs=Rs, slope=slope, slope_exact=0.4,
                compression=compression, compression_exact=(g + 1) / (g - 1),
                g=g, t=t_end)


def report(sod, noh, sedov):
    print("=" * 62)
    print("  1-D LAGRANGIAN HYDRO  --  verification vs exact solutions")
    print("=" * 62)
    print(f"  Sod shock tube   : L1 density error = {sod['l1']:.4f}")
    print(f"  Noh problem      : post-shock plateau rho = {noh['rho_plateau']:.1f} "
          f"(exact 64), shock at r={noh['r_shock']:.3f}")
    print(f"                     L1(r>0.03) = {noh['l1']:.3f}; central dip is the "
          f"known Noh wall-heating artifact")
    print(f"  Sedov blast      : shock scaling R~t^{sedov['slope']:.3f} "
          f"(exact 0.400)")
    print(f"                     compression {sedov['compression']:.2f} "
          f"(exact {sedov['compression_exact']:.2f})")
    print("=" * 62)


def make_figure(sod, noh, sedov, fname="hydro_validation.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

    # Sod
    a = ax[0]
    a.plot(sod["x"], sod["rho_ex"], "k-", lw=2, label="exact")
    a.plot(sod["x"], sod["rho"], "o", ms=2.5, color="tab:red", label="Lagrangian")
    a.set_title(f"{sod['name']}\nt={sod['t']}, L1={sod['l1']:.3f}")
    a.set_xlabel("x"); a.set_ylabel("density"); a.legend(fontsize=9)

    # Noh
    b = ax[1]
    b.plot(noh["x"], noh["rho_ex"], "k-", lw=2, label="exact")
    b.plot(noh["x"], noh["rho"], "o", ms=2.5, color="tab:blue", label="Lagrangian")
    b.axvline(noh["r_shock"], color="gray", ls=":", lw=1)
    b.annotate("wall-heating\nartifact (known)", xy=(0.01, 12), xytext=(0.11, 30),
               fontsize=8, color="gray",
               arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    b.set_title(f"{noh['name']}\nplateau rho={noh['rho_plateau']:.0f} vs exact 64, "
                f"shock at r={noh['r_shock']:.2f}")
    b.set_xlabel("r"); b.set_ylabel("density"); b.set_ylim(0, 75); b.legend(fontsize=9)

    # Sedov (scaling law)
    c = ax[2]
    c.loglog(sedov["ts"], sedov["Rs"], "o", ms=4, color="tab:green", label="sim shock radius")
    tfit = np.array([sedov["ts"][0], sedov["ts"][-1]])
    Cfit = sedov["Rs"][-1] / sedov["ts"][-1] ** 0.4
    c.loglog(tfit, Cfit * tfit ** 0.4, "k--", lw=1.5, label=r"$R \propto t^{2/5}$ (exact)")
    c.set_title(f"{sedov['name']}\nfit R~t^{sedov['slope']:.3f} (exact 0.4), "
                f"compression {sedov['compression']:.1f} (exact 6)")
    c.set_xlabel("time"); c.set_ylabel("shock radius"); c.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    sod = run_sod()
    noh = run_noh()
    sedov = run_sedov()
    report(sod, noh, sedov)
    make_figure(sod, noh, sedov)


if __name__ == "__main__":
    main()
