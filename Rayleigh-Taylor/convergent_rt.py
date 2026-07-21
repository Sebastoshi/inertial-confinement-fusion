r"""
Deceleration-phase Rayleigh-Taylor in a CONVERGING shell (indirect-drive / hohlraum).

rt_mechanics.py and rt_2d.py cover the ACCELERATION-phase ablation-front RT in a
PLANAR geometry. This script adds the two effects that make ICF spherical and
that a planar model cannot show:

  1. DECELERATION-phase RT.  At stagnation the hot spot (light) decelerates the
     dense shell (heavy).  The effective gravity g_eff = +R_ddot (deceleration)
     turns the hot-spot/fuel interface RT-unstable.  This is the RT that mixes
     cold fuel into the hot spot and quenches ignition -- the second, and more
     ignition-relevant, place RT bites.  Unlike the ablation front there is no
     mass ablation here, so no ablative (Takabe) stabilization saves it.

  2. SPHERICAL CONVERGENCE (Bell-Plesset).  In a converging implosion a mode
     amplitude grows geometrically ~ R0/R even without RT, and the local
     wavenumber k = ell/R sweeps upward as the shell shrinks.

Model: integrate an implosion trajectory R(t) -- the shell coasts inward at
v_imp, then decelerates on an adiabatic hot-spot gas cushion and bounces (the
gas P0 is auto-tuned to a target convergence ratio).  Along that trajectory,
for each Legendre mode ell, integrate the linearized interface-amplitude ODE

    eta'' + 2 (R'/R) eta'  -  ell * A * (R''/R) * eta = 0
            \_________/        \_______________/
             convergence          RT  (g_eff = R'' = deceleration > 0)

Two limits are checked (printed) so the reduced model is anchored:
  * PLANAR limit  (R fixed, constant g, k = ell/R):  recovers the growth rate
    gamma = sqrt(A g k) that rt_2d.py measured to -6.7%.
  * CONVERGENCE-only limit  (R''=0, uniform R'):  gives eta ~ 1/R, the classic
    Bell-Plesset geometric amplification.

Run:  python3 convergent_rt.py
Deps: numpy  (matplotlib only for the optional figure)
"""

import numpy as np

# ----------------------------------------------------------------------------
# Parameters -- deceleration phase of an indirect-drive implosion.
# ----------------------------------------------------------------------------
R0        = 1.0e-3        # initial shell / hot-spot radius        [m]
V_IMP     = 3.45e5        # implosion velocity (matches rocket model) [m/s]
SIGMA     = 1.0           # shell areal mass (~200 g/cc x 5 um)    [kg/m^2]
GAMMA_G   = 5.0 / 3.0     # hot-spot gas adiabatic index
ATWOOD    = 0.7           # hot-spot / shell interface Atwood number
TARGET_CR = 30.0          # target convergence ratio R0/R_min (auto-tunes P0)
A0        = 0.05e-6       # seed perturbation amplitude at end of accel [m]
ELL       = np.arange(1, 121)   # Legendre modes to track

_DT       = 2.0e-13       # trajectory time step [s]
_TMAX     = 6.0e-9        # trajectory integration cap [s]


# ----------------------------------------------------------------------------
# Implosion trajectory: shell (areal mass SIGMA) coasting inward, decelerated
# by an adiabatic hot-spot gas  P_gas = P0 (R0/R)^(3 gamma_g).
# ----------------------------------------------------------------------------
def _accel(R, P0):
    return P0 * (R0 / R) ** (3.0 * GAMMA_G) / SIGMA        # R'' [m/s^2], outward +

def trajectory(P0):
    """RK4 integrate [R, V] from R0, V=-V_IMP until the bounce (V>=0)."""
    n = int(_TMAX / _DT)
    T = np.empty(n); R = np.empty(n); V = np.empty(n); Acc = np.empty(n)
    r, v = R0, -V_IMP
    m = 0
    for i in range(n):
        a = _accel(r, P0)
        T[i], R[i], V[i], Acc[i] = i * _DT, r, v, a
        m = i
        if v >= 0.0 and i > 0:            # past stagnation -> bounce
            break
        # RK4 step on (r, v);  r' = v,  v' = accel(r)
        k1r, k1v = v, _accel(r, P0)
        k2r, k2v = v + 0.5*_DT*k1v, _accel(r + 0.5*_DT*k1r, P0)
        k3r, k3v = v + 0.5*_DT*k2v, _accel(r + 0.5*_DT*k2r, P0)
        k4r, k4v = v + _DT*k3v,     _accel(r + _DT*k3r, P0)
        r += _DT/6.0 * (k1r + 2*k2r + 2*k3r + k4r)
        v += _DT/6.0 * (k1v + 2*k2v + 2*k3v + k4v)
        if r < R0 / 80.0:                 # runaway guard
            break
    return T[:m+1], R[:m+1], V[:m+1], Acc[:m+1]

def tune_P0(target_cr=TARGET_CR):
    """Bisection on P0 so R0/R_min == target_cr (CR decreases as P0 increases)."""
    lo, hi = 1.0e5, 1.0e14
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        _, R, _, _ = trajectory(mid)
        cr = R0 / R.min()
        if cr > target_cr:               # too much convergence -> stiffen gas
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


# ----------------------------------------------------------------------------
# Linear perturbation along the trajectory, vectorized over modes.
#   state per mode: eta, etadot.   term = 'full' | 'rt' | 'conv'
# RK4 in time; trajectory coefficients sampled at grid + midpoints.
# ----------------------------------------------------------------------------
def _rhs(eta, ed, R, V, Acc, ell, term):
    conv = -2.0 * (V / R) * ed if term in ("full", "conv") else 0.0
    rt   =  ell * ATWOOD * (Acc / R) * eta if term in ("full", "rt") else 0.0
    return ed, conv + rt

def integrate_modes(T, R, V, Acc, ell, term="full", eta0=A0, ed0=None):
    ell = np.asarray(ell, float)
    eta = np.full_like(ell, eta0)
    ed  = np.zeros_like(ell) if ed0 is None else np.full_like(ell, 1.0) * ed0
    for i in range(len(T) - 1):
        dt = T[i+1] - T[i]
        Rm, Vm, Am = 0.5*(R[i]+R[i+1]), 0.5*(V[i]+V[i+1]), 0.5*(Acc[i]+Acc[i+1])
        k1e, k1d = _rhs(eta,             ed,             R[i], V[i], Acc[i], ell, term)
        k2e, k2d = _rhs(eta+0.5*dt*k1e,  ed+0.5*dt*k1d,  Rm,   Vm,   Am,     ell, term)
        k3e, k3d = _rhs(eta+0.5*dt*k2e,  ed+0.5*dt*k2d,  Rm,   Vm,   Am,     ell, term)
        k4e, k4d = _rhs(eta+dt*k3e,      ed+dt*k3d,      R[i+1], V[i+1], Acc[i+1], ell, term)
        eta = eta + dt/6.0*(k1e + 2*k2e + 2*k3e + k4e)
        ed  = ed  + dt/6.0*(k1d + 2*k2d + 2*k3d + k4d)
    return eta, ed

def amplitude_history(T, R, V, Acc, ell_scalar, term="full", eta0=A0, ed0=0.0):
    """Full eta(t) for a single mode (for the amplitude panel)."""
    eta, ed = eta0, ed0
    out = np.empty(len(T)); out[0] = eta
    for i in range(len(T) - 1):
        dt = T[i+1] - T[i]
        Rm, Vm, Am = 0.5*(R[i]+R[i+1]), 0.5*(V[i]+V[i+1]), 0.5*(Acc[i]+Acc[i+1])
        k1e, k1d = _rhs(eta,            ed,            R[i], V[i], Acc[i], ell_scalar, term)
        k2e, k2d = _rhs(eta+0.5*dt*k1e, ed+0.5*dt*k1d, Rm,  Vm,   Am,     ell_scalar, term)
        k3e, k3d = _rhs(eta+0.5*dt*k2e, ed+0.5*dt*k2d, Rm,  Vm,   Am,     ell_scalar, term)
        k4e, k4d = _rhs(eta+dt*k3e,     ed+dt*k3d,     R[i+1], V[i+1], Acc[i+1], ell_scalar, term)
        eta = eta + dt/6.0*(k1e + 2*k2e + 2*k3e + k4e)
        ed  = ed  + dt/6.0*(k1d + 2*k2d + 2*k3d + k4d)
        out[i+1] = eta
    return out


# ----------------------------------------------------------------------------
# Validation limits
# ----------------------------------------------------------------------------
def validate_planar(g0=2.0e14, ell_scalar=50.0):
    """Frozen radius, constant deceleration g0: measured gamma vs sqrt(A k g0)."""
    n = 4000; dt = 2.0e-13
    T = np.arange(n) * dt
    R = np.full(n, R0); V = np.zeros(n); Acc = np.full(n, g0)
    k = ell_scalar / R0
    gamma_th = np.sqrt(ATWOOD * g0 * k)
    a = amplitude_history(T, R, V, Acc, ell_scalar, term="rt",
                          eta0=1e-9, ed0=gamma_th * 1e-9)   # exponential seed
    # measure slope in the clean exponential window
    lo, hi = int(0.3*n), int(0.8*n)
    gamma_meas = np.polyfit(T[lo:hi], np.log(a[lo:hi]), 1)[0]
    return gamma_th, gamma_meas

def validate_convergence(P0):
    """Convergence-only, BP self-similar seed: check eta ~ 1/R (eta*R ~ const)."""
    T, R, V, Acc = trajectory(P0)
    ell_scalar = 6.0
    ed0 = -A0 * V[0] / R[0]                    # d/dt (C/R) at t=0
    a = amplitude_history(T, R, V, Acc, ell_scalar, term="conv", eta0=A0, ed0=ed0)
    istag = np.argmin(R)
    prod0, prods = a[0]*R[0], a[istag]*R[istag]
    return prod0, prods, R0/R[istag]


# ----------------------------------------------------------------------------
def report():
    P0 = tune_P0()
    T, R, V, Acc = trajectory(P0)
    istag = int(np.argmin(R))
    CR = R0 / R[istag]
    g_dec = Acc[istag]                          # peak deceleration [m/s^2]
    t_stag = T[istag]

    eta_full, _ = integrate_modes(T, R, V, Acc, ELL, term="full")
    eta_rt,   _ = integrate_modes(T, R, V, Acc, ELL, term="rt")
    eta_cnv,  _ = integrate_modes(T, R, V, Acc, ELL, term="conv",
                                  ed0=None)      # from-rest: convergence alone barely grows
    GF = eta_full / A0
    ell_peak = int(ELL[np.argmax(GF)])
    GF_peak = GF.max()

    # dominant low mode (ell=10) decomposition
    j = np.where(ELL == 10)[0][0]
    gf10, gf10_rt = GF[j], eta_rt[j] / A0

    gpl_th, gpl_meas = validate_planar()
    p0c, psc, crc = validate_convergence(P0)

    a_stag = eta_full[j]                         # ell=10 amplitude at stagnation
    clean = max(0.0, 1.0 - a_stag / R[istag])

    print("=" * 66)
    print("  CONVERGENT / DECELERATION-PHASE RAYLEIGH-TAYLOR")
    print("=" * 66)
    print(f"  implosion velocity        : {V_IMP/1e3:8.0f} km/s")
    print(f"  tuned hot-spot gas P0      : {P0:.2e} Pa   -> CR = {CR:.1f}")
    print(f"  convergence ratio  R0/Rmin : {CR:8.1f}   (Rmin = {R[istag]*1e6:.1f} um)")
    print(f"  stagnation time            : {t_stag*1e9:8.2f} ns")
    print(f"  peak deceleration g_eff    : {g_dec:.2e} m/s^2  "
          f"({g_dec/1.9e14:.1f}x the accel-phase g)")
    print("-" * 66)
    print(f"  Atwood (hot-spot interface): {ATWOOD:.2f}   (no ablative stabilization here)")
    print(f"  most-dangerous mode        : ell = {ell_peak:4d}   -> growth factor x{GF_peak:.1e}")
    print(f"  ell=10 growth factor       : x{gf10:.1f}  (RT-only x{gf10_rt:.1f}; "
          f"convergence multiplies on top)")
    print(f"  ell=10 amplitude @ stag    : {a_stag*1e6:.2f} um  vs Rmin {R[istag]*1e6:.1f} um "
          f"-> clean hot-spot ~ {clean*100:.0f}%")
    print("-" * 66)
    print("  VALIDATION")
    print(f"   planar limit  gamma_theory = sqrt(A g k) : {gpl_th:.3e} 1/s")
    print(f"   planar limit  gamma_measured (solver)    : {gpl_meas:.3e} 1/s "
          f"({(gpl_meas-gpl_th)/gpl_th*100:+.1f}%)")
    print(f"   convergence limit  eta*R start / stag    : {p0c:.3e} / {psc:.3e} "
          f"(ratio {psc/p0c:.2f}, expect ~1 for eta~1/R at CR={crc:.0f})")
    print("=" * 66)
    return dict(P0=P0, T=T, R=R, V=V, Acc=Acc, istag=istag, CR=CR,
                GF=GF, eta_rt=eta_rt, eta_cnv=eta_cnv, ell_peak=ell_peak)


def make_figure(info, fname="convergent_rt.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: matplotlib unavailable -- {e}]")
        return
    T, R, V, Acc = info["T"], info["R"], info["V"], info["Acc"]
    istag = info["istag"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(T*1e9, R*1e6, "tab:blue", lw=2)
    a.axvline(T[istag]*1e9, color="k", ls=":", lw=1)
    a.set_xlabel("time [ns]"); a.set_ylabel("shell radius [um]", color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    a2.plot(T*1e9, Acc/1e14, "tab:red", lw=2)
    a2.set_ylabel("deceleration  R''  [1e14 m/s^2]", color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red")
    a.set_title(f"implosion trajectory (CR = {info['CR']:.0f}):\ncoast -> violent stagnation")

    b = ax[0, 1]
    b.semilogy(ELL, info["GF"], "tab:red", lw=2.5, label="full (RT + convergence)")
    b.semilogy(ELL, info["eta_rt"]/A0, "tab:gray", lw=1.8, ls="--", label="RT only")
    b.axvline(info["ell_peak"], color="k", ls=":", lw=1)
    b.set_xlabel("Legendre mode  ell"); b.set_ylabel("growth factor  eta_stag / eta_0")
    b.set_title("growth-factor spectrum:\nconvergence lifts the whole curve")
    b.legend(fontsize=9)

    c = ax[1, 0]
    for ell_s, col in [(4, "tab:green"), (10, "tab:orange"), (40, "tab:purple")]:
        hist = amplitude_history(T, R, V, Acc, float(ell_s), term="full")
        c.semilogy(T*1e9, np.abs(hist)*1e6, col, lw=2, label=f"ell = {ell_s}")
    c.axvline(T[istag]*1e9, color="k", ls=":", lw=1)
    c.text(T[istag]*1e9*0.6, c.get_ylim()[1]*0.3, "stagnation", fontsize=8)
    c.set_xlabel("time [ns]"); c.set_ylabel("mode amplitude [um]")
    c.set_title("amplitude growth:\nquiet coast, explosive deceleration")
    c.legend(fontsize=9, loc="lower right")

    d = ax[1, 1]
    Rn = R / R0
    d.loglog(1.0/Rn, info["eta_cnv"][5]/A0 * np.ones_like(Rn), alpha=0)  # keep frame
    d.loglog(1.0/Rn, (1.0/Rn), "tab:gray", ls="--", lw=1.8, label="Bell-Plesset  ~ R0/R")
    hist_c = amplitude_history(T, R, V, Acc, 6.0, term="conv",
                               eta0=A0, ed0=-A0*V[0]/R[0])
    d.loglog(1.0/Rn, np.abs(hist_c)/A0, "tab:blue", lw=2.2, label="solver (convergence only)")
    d.set_xlabel("convergence  R0 / R"); d.set_ylabel("amplitude / eta_0")
    d.set_title("convergence validation:\nsolver tracks the R0/R law")
    d.legend(fontsize=9, loc="upper left")

    fig.suptitle("Deceleration-phase RT in a converging shell: "
                 "spherical convergence + hot-spot RT", fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    info = report()
    make_figure(info)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * Deceleration RT has NO ablative stabilization (no mass ablation at the
#   hot-spot interface), so unlike the ablation front the high-ell modes are not
#   cut off here -- they are limited only by the short deceleration time and the
#   finite interface thickness (not modelled). That is why the growth-factor
#   spectrum keeps rising with ell; a real interface-thickness cutoff would roll
#   it over near the density-gradient scale length.
#
# * Convergence (Bell-Plesset) MULTIPLIES the RT growth -- compare the 'full' and
#   'RT only' curves in panel 2. Low modes (ell < ~10) are convergence-dominated;
#   high modes are RT-dominated. This is why ICF cares about BOTH surface finish
#   (high ell) and drive symmetry (low ell -> see hohlraum_asymmetry.py).
#
# * Raise TARGET_CR (a higher-convergence design) and watch both the peak
#   deceleration and the growth factors climb -- convergence buys compression but
#   costs stability. Lower ATWOOD (a graded interface) to see the RT term soften.
#
# Simplifications: reduced linear thin-interface ODE (2*R'/R convergence damping,
# k=ell/R local wavenumber), a lumped adiabatic-gas deceleration (no shock
# structure), single mode per solve (no mode coupling / turbulent mix), and no
# finite-density-gradient cutoff. It captures the deceleration-RT + convergence
# MECHANISM and the two limiting laws, not an ICF-accurate mix width.
# ----------------------------------------------------------------------------
