"""
Rayleigh-Taylor (RT) instability mechanics -- why ICF is hard.

A 1-D implosion (the other models in this repo) makes ignition look like a
matter of hitting the right velocity and areal density. Reality intervenes:
the imploding shell is Rayleigh-Taylor UNSTABLE. A light fluid pushing a heavy
one (the ablated plasma pushing the cold shell, and later the hot spot pushing
the fuel) amplifies any ripple exponentially. Left unchecked it shreds the shell
before it can stagnate.

This script is the *mechanics* -- the linear and weakly-nonlinear theory that
sets the ICF design constraints:

  1. Dispersion relation: classical  gamma = sqrt(A g k)  vs the ablative
     (Takabe) form  gamma = alpha*sqrt(A g k) - beta*k*V_ablation, which cuts
     off short wavelengths and picks a single most-dangerous mode.
  2. e-foldings over the implosion: how many times a ripple grows during the
     acceleration phase -- and the punchline, that ablative stabilization is
     what keeps the shell from breaking up.
  3. Nonlinear saturation: exponential growth -> bubble/spike terminal velocity.
  4. The design lever: peak growth rate ~ 1/V_ablation.

Companion: rt_2d.py runs an actual 2D simulation and measures gamma to check
the sqrt(A g k) law used here.

Run:  python3 rt_mechanics.py
Deps: numpy, matplotlib
"""

import numpy as np

# ----------------------------------------------------------------------------
# Parameters -- acceleration-phase ablation front of an ICF capsule.
# g comes from the rocket-implosion model: v_imp ~ 345 km/s reached over the
# ~1.85 ns acceleration, so g ~ v/t ~ 1.9e14 m/s^2.
# ----------------------------------------------------------------------------
G_ACCEL   = 1.9e14         # implosion acceleration          [m/s^2]
T_ACCEL   = 1.85e-9        # acceleration-phase duration     [s]
ATWOOD    = 0.9            # Atwood number at the ablation front  A=(rh-rl)/(rh+rl)
V_ABL     = 5.0e3          # ablation velocity               [m/s]  (5 um/ns)
ALPHA     = 0.9            # Takabe coefficient (buoyancy)
BETA      = 3.0            # Takabe coefficient (ablative stabilization)

R_CAP     = 1.0e-3         # capsule radius [m]  -> mode number  ell = k*R = 2piR/lambda
A0_ROUGH  = 5.0e-9         # initial surface roughness amplitude [m]  (5 nm finish)
D_SHELL   = 40.0e-6        # in-flight shell thickness [m]  (amplitude ~ this => breakup)
C_DRAG    = 3.0            # bubble drag coefficient (nonlinear terminal velocity)


# ----------------------------------------------------------------------------
# Dispersion relations.  k = 2*pi/lambda.
# ----------------------------------------------------------------------------
def k_of(lam):
    return 2.0 * np.pi / lam

# safe wavelength <-> mode-number transforms for the secondary axis (no /0 warn)
def _lam_to_ell(L):
    L = np.where(np.asarray(L, float) > 0, L, np.nan)
    return 2.0 * np.pi * R_CAP / (L * 1e-6)

def _ell_to_lam(l):
    l = np.where(np.asarray(l, float) > 0, l, np.nan)
    return 2.0 * np.pi * R_CAP / l * 1e6

def gamma_classical(lam, g=G_ACCEL, A=ATWOOD):
    return np.sqrt(A * g * k_of(lam))

def gamma_ablative(lam, g=G_ACCEL, A=ATWOOD, Va=V_ABL):
    k = k_of(lam)
    return np.maximum(ALPHA * np.sqrt(A * g * k) - BETA * k * Va, 0.0)

def most_unstable_k(g=G_ACCEL, A=ATWOOD, Va=V_ABL):
    """k that maximizes the ablative growth rate:  k* = alpha^2 A g / (4 beta^2 Va^2)."""
    return ALPHA ** 2 * A * g / (4.0 * BETA ** 2 * Va ** 2)

def peak_gamma(g=G_ACCEL, A=ATWOOD, Va=V_ABL):
    """Closed form:  gamma_peak = alpha^2 A g / (4 beta Va)  (note: ~ 1/Va)."""
    return ALPHA ** 2 * A * g / (4.0 * BETA * Va)


# ----------------------------------------------------------------------------
# Nonlinear single-mode amplitude: exponential growth, then saturation at
# ~0.1*lambda, then linear-in-time growth at the bubble terminal velocity.
# ----------------------------------------------------------------------------
def bubble_terminal_velocity(lam, g=G_ACCEL, A=ATWOOD):
    k = k_of(lam)
    return np.sqrt(2.0 * A * g / ((1.0 + A) * C_DRAG * k))

def amplitude(t, lam, a0=A0_ROUGH):
    gam = gamma_ablative(lam)
    a_sat = 0.1 * lam                      # nonlinear onset
    Vb = bubble_terminal_velocity(lam)
    a_lin = a0 * np.exp(gam * t)
    t_sat = np.log(a_sat / a0) / gam
    return np.where(t < t_sat, a_lin, a_sat + Vb * (t - t_sat))


def report():
    kstar = most_unstable_k()
    lam_star = 2.0 * np.pi / kstar
    ell_star = kstar * R_CAP
    gpk = ALPHA ** 2 * ATWOOD * G_ACCEL / (4.0 * BETA * V_ABL)
    N_abl = gpk * T_ACCEL
    N_cls = gamma_classical(lam_star) * T_ACCEL
    N_breakup = np.log(D_SHELL / A0_ROUGH)

    print("=" * 64)
    print("  RAYLEIGH-TAYLOR MECHANICS  --  ablation-front, acceleration phase")
    print("=" * 64)
    print(f"  acceleration g            : {G_ACCEL:.2e} m/s^2  (from the implosion)")
    print(f"  Atwood number A           : {ATWOOD:.2f}")
    print(f"  ablation velocity V_abl   : {V_ABL/1e3:.1f} um/ns")
    print("-" * 64)
    print(f"  most-unstable wavelength  : {lam_star*1e6:8.1f} um   (mode ell ~ {ell_star:.0f})")
    print(f"  peak growth rate          : {gpk:.2e} 1/s")
    print(f"  peak e-foldings (ablative): {N_abl:8.2f}   -> x{np.exp(N_abl):.0f} amplification")
    print(f"  same mode, classical      : {N_cls:8.2f}   -> x{np.exp(N_cls):.0f} amplification")
    print(f"  breakup threshold         : {N_breakup:8.2f}   (a0 grows to shell thickness)")
    print("-" * 64)
    a_final_abl = A0_ROUGH * np.exp(N_abl)
    a_final_cls = A0_ROUGH * np.exp(N_cls)
    print(f"  ripple {A0_ROUGH*1e9:.0f} nm -> ablative {a_final_abl*1e6:6.2f} um  "
          f"({'SURVIVES' if a_final_abl < D_SHELL else 'BREAKS UP'} vs {D_SHELL*1e6:.0f} um shell)")
    print(f"  ripple {A0_ROUGH*1e9:.0f} nm -> classical {a_final_cls*1e6:6.2f} um  "
          f"({'survives' if a_final_cls < D_SHELL else 'BREAKS UP'} -- no ablative stabilization)")
    print("=" * 64)
    return dict(lam_star=lam_star, ell_star=ell_star, N_abl=N_abl,
                N_cls=N_cls, N_breakup=N_breakup)


def make_figure(info, fname="rt_mechanics.png"):
    import matplotlib.pyplot as plt

    lam = np.logspace(np.log10(2e-6), np.log10(1e-3), 500)     # 2 um .. 1 mm
    g_cls = gamma_classical(lam)
    g_abl = gamma_ablative(lam)
    lam_star = info["lam_star"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (1) dispersion relation
    a = ax[0, 0]
    a.plot(lam * 1e6, g_cls / 1e9, "tab:gray", lw=2, ls="--", label="classical  sqrt(A g k)")
    a.plot(lam * 1e6, g_abl / 1e9, "tab:red", lw=2.5, label="ablative (Takabe)")
    a.axvline(lam_star * 1e6, color="k", ls=":", lw=1)
    a.text(lam_star * 1e6 * 1.15, 2.5, f"most unstable\n{lam_star*1e6:.0f} um", fontsize=8)
    a.set_xscale("log"); a.set_xlabel("wavelength  [um]")
    a.set_ylabel("growth rate  [1/ns]"); a.set_ylim(0, 4.0)
    a.set_title("dispersion relation:\nablation cuts off short wavelengths")
    a.legend(fontsize=9, loc="upper right")
    # top axis: spherical-harmonic mode number ell = 2*pi*R/lambda
    at = a.secondary_xaxis("top", functions=(_lam_to_ell, _ell_to_lam))
    at.set_xlabel("mode number  ell")

    # (2) e-foldings over the implosion -- the design constraint
    b = ax[0, 1]
    N_cls = g_cls * T_ACCEL
    N_abl = g_abl * T_ACCEL
    b.plot(lam * 1e6, N_cls, "tab:gray", lw=2, ls="--", label="classical")
    b.plot(lam * 1e6, N_abl, "tab:red", lw=2.5, label="ablative")
    b.axhline(info["N_breakup"], color="k", lw=1.5)
    b.text(3, info["N_breakup"] + 0.3, "shell break-up threshold", fontsize=8)
    b.fill_between(lam * 1e6, info["N_breakup"], 40, color="tab:red", alpha=0.06)
    b.set_xscale("log"); b.set_xlabel("wavelength  [um]")
    b.set_ylabel("e-foldings over implosion  (gamma * t_accel)")
    b.set_ylim(0, 15)
    b.set_title("growth over the implosion:\nablative stabilization saves the shell")
    b.legend(fontsize=9, loc="upper right")

    # (3) nonlinear saturation of the most-unstable mode
    c = ax[1, 0]
    t = np.linspace(0, 6.0e-9, 400)
    gam = gamma_ablative(lam_star)
    a_lin = A0_ROUGH * np.exp(gam * t)
    a_full = amplitude(t, lam_star)
    c.plot(t * 1e9, a_lin * 1e6, "tab:gray", ls="--", lw=1.8, label="linear (exp) growth")
    c.plot(t * 1e9, a_full * 1e6, "tab:red", lw=2.5, label="with saturation")
    c.axhline(0.1 * lam_star * 1e6, color="tab:blue", ls=":", lw=1)
    c.text(0.15, 0.1 * lam_star * 1e6 * 1.2, "saturation ~ 0.1 lambda", fontsize=8, color="tab:blue")
    c.axvline(T_ACCEL * 1e9, color="k", ls=":", lw=1)
    c.text(T_ACCEL * 1e9 * 1.03, 2e-2, "end of\naccel.", fontsize=8)
    c.set_yscale("log"); c.set_ylim(1e-3, 1e2)
    c.set_xlabel("time  [ns]")
    c.set_ylabel("perturbation amplitude  [um]")
    c.set_title(f"single-mode growth ({lam_star*1e6:.0f} um):\nexponential -> bubble terminal velocity")
    c.legend(fontsize=9, loc="lower right")

    # (4) the design lever: peak growth rate & most-unstable lambda vs ablation
    d = ax[1, 1]
    Va = np.linspace(1e3, 15e3, 200)
    kstar = ALPHA ** 2 * ATWOOD * G_ACCEL / (4.0 * BETA ** 2 * Va ** 2)
    gpk = ALPHA ** 2 * ATWOOD * G_ACCEL / (4.0 * BETA * Va)
    lstar = 2 * np.pi / kstar
    d.plot(Va / 1e3, gpk / 1e9, "tab:red", lw=2.5, label="peak growth rate")
    d.axvline(V_ABL / 1e3, color="k", ls=":", lw=1)
    d.set_xlabel("ablation velocity  [um/ns]")
    d.set_ylabel("peak growth rate  [1/ns]", color="tab:red")
    d.tick_params(axis="y", labelcolor="tab:red")
    d2 = d.twinx()
    d2.plot(Va / 1e3, lstar * 1e6, "tab:purple", lw=2.5, label="most-unstable lambda")
    d2.set_ylabel("most-unstable wavelength [um]", color="tab:purple")
    d2.tick_params(axis="y", labelcolor="tab:purple")
    d.set_title("the design lever:\nmore ablation -> slower growth, longer modes")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    info = report()
    make_figure(info)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The headline result (panel 2): a 5 nm ripple grows by ~x100 under ablative
#   RT -- to sub-micron, safely under the ~40 um shell. Turn ablation OFF
#   (classical curve) and the same ripple would run away past the shell
#   thickness. Ablative stabilization is the reason ICF shells hold together.
#
# * peak_gamma ~ alpha^2 A g / (4 beta V_abl): growth scales with the drive (g)
#   and *inversely* with ablation velocity. Faster ablation = smoother implosion,
#   at the cost of ablating away more of your shell (the rocket-model trade).
#
# * Raise A0_ROUGH (a rougher capsule) or G_ACCEL (a harder drive) until the
#   ablative curve in panel 2 crosses the break-up line -- that is the surface-
#   finish / drive spec a real capsule has to meet.
#
# * The most-unstable mode ell ~ 100-200 here; lower-ell (long-wavelength) modes
#   grow less but feed through the shell to seed the deceleration-phase RT on the
#   hot spot -- the second place RT bites. Same mechanics, different interface.
#
# Simplifications: linearized dispersion (single mode, no mode coupling), a
# thin-interface Takabe fit (real fronts have finite density-gradient scale
# length that adds its own stabilization), constant g, and a heuristic
# saturation/terminal-velocity closure. The 2D sim (rt_2d.py) drops the linear
# assumption and grows a real bubble.
# ----------------------------------------------------------------------------
