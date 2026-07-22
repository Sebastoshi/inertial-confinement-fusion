r"""
Hohlraum drive asymmetry -> hot-spot shape and yield.

Surface roughness (rt_mechanics / rt_2d) seeds HIGH-ell RT. In an indirect-drive
hohlraum the dominant failure is the opposite end of the spectrum: LOW-ell drive
asymmetry. The hohlraum's geometry -- its length/diameter, the laser entrance
holes (LEH) at the poles, wall re-emission, and the inner/outer laser-cone power
balance -- imprints a coherent P2/P4 pattern on the X-ray flux that drives the
capsule. That low-mode drive asymmetry, amplified by convergence (Bell-Plesset,
convergent_rt.py), distorts the hot spot into a pancake or sausage and kills
yield long before high-mode RT does.

This script:
  1. Maps a drive flux asymmetry  I(theta) = I0 [1 + sum_l a_l P_l(cos theta)]
     to a per-mode velocity asymmetry, seeds the convergent-RT solve with it, and
     reads the mode amplitudes eta_l at stagnation.
  2. Reconstructs the 2D hot-spot shape  R_hs(theta) = Rmin [1 + sum_l (eta_l/Rmin) P_l].
  3. Scores yield-over-clean (YOC) vs the drive asymmetry.
  4. Exposes performance(...) as a FAST, closed-form forward model (the mode
     transfer factors are precomputed once, so each evaluation is a few flops) --
     the hook an ML surrogate / optimizer samples to search hohlraum geometries.

Run:  python3 hohlraum_asymmetry.py
Deps: numpy, scipy  (matplotlib only for the optional figure)
"""

import importlib.util
import os

import numpy as np
from scipy.special import eval_legendre

from convergent_rt import R0, V_IMP, trajectory, tune_P0, integrate_modes

# Load the repo's 0-D hot-spot ignition model (its folder name has a space and a
# leading digit, so a normal import won't work -- load it by path).
_hs_path = os.path.join(os.path.dirname(__file__), "..", "0-D Hotspot", "hotspot_0d.py")
_spec = importlib.util.spec_from_file_location("hotspot_0d", _hs_path)
hotspot_0d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hotspot_0d)

# Fraction of X-ray flux asymmetry that survives conduction / "cloudy-day"
# smoothing to become a shell velocity asymmetry (illustrative).
COUPLING    = 0.25
T_ACCEL     = 1.85e-9                  # acceleration-phase duration [s] (rt_mechanics)
DRIVE_MODES = np.array([2, 4, 6])      # even low modes a symmetric hohlraum imprints

# Nominal round, igniting hot spot (representative of the converged capsule).
NOMINAL_RHO  = 300.0                   # g/cm^3
NOMINAL_RHOR = 1.2                     # g/cm^2  (above the ~0.3 alpha-stopping floor)
NOMINAL_T0   = 6.0                     # keV     (above the ~4.3 keV ignition temperature)
# How a distorted hot spot degrades the two ignition variables:
K_RKE = 10.0                           # residual kinetic energy -> lower T  (~ rms^2)
K_RR  = 1.5                            # peanut-waist thinning   -> lower rho*R (~ rms)
_BURNUP_ROUND = None                   # cached burn-up of the round hot spot


def _round_burnup():
    global _BURNUP_ROUND
    if _BURNUP_ROUND is None:
        _BURNUP_ROUND = hotspot_0d.burn(NOMINAL_RHO, NOMINAL_RHOR, NOMINAL_T0)["burnup"]
    return _BURNUP_ROUND


# ----------------------------------------------------------------------------
# Forward model.  Because the perturbation ODE is LINEAR with a zero-amplitude,
# velocity-seed initial condition, eta_l(stagnation) = M_l * (seed velocity).
# We precompute the transfer factors M_l once; every performance() call is then
# closed form -- ideal for sampling thousands of designs in an ML loop.
# ----------------------------------------------------------------------------
def build_forward(P0=None, modes=DRIVE_MODES):
    if P0 is None:
        P0 = tune_P0()
    T, R, V, Acc = trajectory(P0)
    Rmin = float(R.min())
    # unit-amplitude, from-rest seed in every mode -> M_l = growth factor GF_l
    M, _ = integrate_modes(T, R, V, Acc, modes.astype(float),
                           term="full", eta0=1.0, ed0=0.0)
    return dict(P0=P0, Rmin=Rmin, modes=np.asarray(modes), M=np.asarray(M))


def hotspot_shape(a_l, fwd, ntheta=400):
    """Return theta, R_hs/Rmin, and eta_l(stagnation) for flux asymmetries a_l."""
    a_l = np.asarray(a_l, float)
    # drive asymmetry -> velocity asymmetry -> ballistic offset built over the
    # drive (seed) -> amplified by the convergent-RT growth factor M_l = GF_l
    seed = COUPLING * a_l * V_IMP * T_ACCEL
    eta = fwd["M"] * seed                            # stagnation amplitude per mode
    theta = np.linspace(0.0, np.pi, ntheta)
    delta = np.zeros_like(theta)
    for l, e in zip(fwd["modes"], eta):
        delta += (e / fwd["Rmin"]) * eval_legendre(int(l), np.cos(theta))
    return theta, 1.0 + delta, eta


def performance(a2, a4=0.0, a6=0.0, fwd=None):
    """Yield-over-clean (YOC) and RMS hot-spot distortion for a drive asymmetry.

    The distortion degrades the hot spot -- residual kinetic energy lowers the
    temperature, and the thin peanut waist lowers the confining areal density --
    and the yield is then the burn-up fraction from the repo's 0-D ignition model.
    The sharp YOC collapse is the ignition CLIFF, not a fitted curve."""
    if fwd is None:
        fwd = build_forward()
    theta, shape, _ = hotspot_shape([a2, a4, a6], fwd)
    w = np.sin(theta)                                 # solid-angle weight
    dev = shape - 1.0
    rms = np.sqrt(np.trapezoid(dev**2 * w, theta) / np.trapezoid(w, theta))
    # distortion -> lower T (residual flow) and lower rho*R (thin waist)
    T_eff    = NOMINAL_T0   * (1.0 - min(K_RKE * rms**2, 0.85))
    rhoR_eff = NOMINAL_RHOR * (1.0 - min(K_RR  * rms,    0.80))
    burnup = hotspot_0d.burn(NOMINAL_RHO, rhoR_eff, T_eff)["burnup"]
    yoc = min(float(burnup / _round_burnup()), 1.0)
    return yoc, float(rms)


# ----------------------------------------------------------------------------
# Reduced hohlraum geometry -> drive asymmetry (view-factor heuristic).
# Illustrative ONLY -- a stand-in for a radiation-transport / view-factor solve.
# It gives the ML phase real geometry knobs to turn.
# ----------------------------------------------------------------------------
def geometry_to_asymmetry(case_to_capsule=3.0, length_to_diameter=1.5,
                          leh_radius_frac=0.5, inner_frac=0.5):
    """geometry -> (a2, a4) via the hohlraum view-factor model.

    Delegates to hohlraum_viewfactor.symmetry(), an actual radiative-exchange
    calculation of the drive pattern (replaces the earlier heuristic).
    """
    from hohlraum_viewfactor import symmetry
    a2, a4, _ = symmetry(case_to_capsule, length_to_diameter, leh_radius_frac, inner_frac)
    return a2, a4


def report():
    fwd = build_forward()
    print("=" * 66)
    print("  HOHLRAUM DRIVE ASYMMETRY  ->  HOT-SPOT SHAPE & YIELD")
    print("=" * 66)
    print(f"  convergence ratio          : {R0/fwd['Rmin']:.0f}   (Rmin = {fwd['Rmin']*1e6:.1f} um)")
    print(f"  flux->velocity coupling    : {COUPLING:.2f}")
    print(f"  mode transfer factors M_l  : " +
          ", ".join(f"ell{l}={m:.2e}" for l, m in zip(fwd["modes"], fwd["M"])))
    print("-" * 66)
    print("  YOC vs P2 drive asymmetry (a4=a6=0):")
    for a2 in [0.0, 0.01, 0.02, 0.05, 0.10]:
        yoc, rms = performance(a2, fwd=fwd)
        bar = "#" * int(round(yoc * 30))
        print(f"    a2 = {a2*100:5.1f}%   RMS distortion {rms*100:5.1f}%   "
              f"YOC {yoc:5.2f}  |{bar}")
    print("-" * 66)
    print("  hohlraum geometry -> asymmetry -> YOC (view-factor model):")
    for name, kw in [("nominal cones", dict(inner_frac=0.50)),
                     ("more inner",    dict(inner_frac=0.35)),
                     ("tuned cones",   dict(inner_frac=0.62))]:
        a2, a4 = geometry_to_asymmetry(**kw)
        yoc, rms = performance(a2, a4, fwd=fwd)
        print(f"    {name:13s}: a2={a2*100:+5.1f}% a4={a4*100:+5.1f}%  ->  YOC {yoc:5.2f}")
    print("=" * 66)
    print("  performance(a2, a4, a6, fwd) is closed-form -> ready for an ML sweep.")
    print("=" * 66)
    return fwd


def make_figure(fwd, fname="hohlraum_asymmetry.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: matplotlib unavailable -- {e}]")
        return
    theta, shape, eta = hotspot_shape([0.05, 0.0, 0.0], fwd)

    fig = plt.figure(figsize=(13, 9))

    # (1) drive flux pattern
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    a2 = 0.05
    flux = 1.0 + a2 * eval_legendre(2, np.cos(theta))
    th_full = np.concatenate([theta, theta + np.pi])
    fl_full = np.concatenate([flux, flux[::-1]])
    ax1.plot(th_full, fl_full, "tab:red", lw=2)
    ax1.set_title("drive flux  I(theta), P2 = 5%", fontsize=10)

    # (2) hot-spot shape (round vs distorted), true 2D
    ax2 = fig.add_subplot(2, 2, 2)
    th_full = np.concatenate([theta, theta + np.pi])
    r_full = np.concatenate([shape, shape[::-1]])
    x = r_full * np.sin(th_full); y = r_full * np.cos(th_full)
    ax2.plot(np.sin(th_full), np.cos(th_full), "gray", ls="--", lw=1, label="round (ideal)")
    ax2.plot(x, y, "tab:purple", lw=2.5, label="with P2 drive asymmetry")
    ax2.set_aspect("equal"); ax2.set_title("hot-spot shape at stagnation", fontsize=10)
    ax2.legend(fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])

    # (3) YOC vs P2 asymmetry
    ax3 = fig.add_subplot(2, 2, 3)
    a2s = np.linspace(0, 0.12, 40)
    yoc0 = [performance(a, fwd=fwd)[0] for a in a2s]
    yoc4 = [performance(a, 0.03, fwd=fwd)[0] for a in a2s]
    ax3.plot(a2s*100, yoc0, "tab:red", lw=2.5, label="P4 = 0")
    ax3.plot(a2s*100, yoc4, "tab:orange", lw=2, ls="--", label="P4 = 3%")
    ax3.set_xlabel("P2 drive asymmetry [%]"); ax3.set_ylabel("yield over clean (YOC)")
    ax3.set_ylim(0, 1.05); ax3.set_title("yield collapses with drive asymmetry", fontsize=10)
    ax3.legend(fontsize=9)

    # (4) geometry sweep: YOC vs inner-cone balance for two LEH sizes
    ax4 = fig.add_subplot(2, 2, 4)
    inner = np.linspace(0.25, 0.80, 40)
    for lrf, col, lab in [(0.50, "tab:blue", "LEH 0.50"), (0.60, "tab:green", "LEH 0.60")]:
        yv = []
        for f in inner:
            a2, a4 = geometry_to_asymmetry(leh_radius_frac=lrf, inner_frac=f)
            yv.append(performance(a2, a4, fwd=fwd)[0])
        ax4.plot(inner, yv, col, lw=2.2, label=lab)
    ax4.set_xlabel("inner-cone power fraction"); ax4.set_ylabel("YOC")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("hohlraum geometry -> YOC\n(view-factor; ML target)", fontsize=10)
    ax4.legend(fontsize=9)

    fig.suptitle("Hohlraum drive asymmetry: low-mode P2/P4 distorts the hot spot "
                 "and kills yield", fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    fwd = report()
    make_figure(fwd)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / ML hook -------------------------------------------------------------
#
# * performance(a2, a4, a6, fwd) is the forward model an ML surrogate learns:
#   a handful of drive-symmetry inputs -> one scalar (YOC). Because the transfer
#   factors are precomputed, it is closed form and vectorizes trivially, so a
#   surrogate/active-learning loop (see the repo's ML Surrogate folder) can sample
#   it thousands of times to search for geometries that maximize YOC.
#
# * geometry_to_asymmetry() is the geometry -> drive-symmetry map. It is a crude
#   view-factor heuristic standing in for a real hohlraum radiation-transport
#   solve. Replace it with a proper view-factor / P_n synthesis and the SAME ML
#   loop then optimizes over ACTUAL hohlraum geometry (case-to-capsule, LEH,
#   cone balance, length/diameter) instead of abstract a_l.
#
# Simplifications: linear transfer (small asymmetry), a fixed flux->velocity
# coupling, only even low modes, and a heuristic YOC(RMS) closure. It captures
# the low-mode drive-asymmetry -> hot-spot-shape -> yield CHAIN, not an
# ICF-accurate yield model.
# ----------------------------------------------------------------------------
