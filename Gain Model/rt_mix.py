r"""
Multimode Rayleigh-Taylor mix width -> an ignition-margin penalty.

Step 2's whole-design optimizer railed convergence high, adiabat low, and fuel
high: the reduced model rewarded compression without charging for the instability
it costs. This module supplies that charge. It turns a capsule surface-roughness
*spectrum* into a stagnation MIX WIDTH, and the mix width into a yield multiplier
that collapses when mix eats the hot spot -- the missing instability penalty.

The chain, reusing the repo's RT physics:

  surface roughness spectrum  sigma_surf * s(ell)
    x  acceleration-phase ablative feedthrough  A_acc(ell; adiabat)   [rt_mechanics]
    x  deceleration RT + Bell-Plesset growth    GF(ell; CR)           [convergent_rt]
    =  per-mode amplitude at stagnation  eta(ell)
  mix width   sigma_mix = sqrt( sum_ell eta(ell)^2 )
  mix fraction f_mix = sigma_mix / R_min(CR)          (R_min = R0 / CR)
  penalty     M(f_mix) = (1 - f_mix)^p                (clean hot-spot fraction)

Where the two design knobs bite:
  * CONVERGENCE enters geometrically -- R_min = R0/CR shrinks, so the same mix
    width is a larger fraction of a smaller hot spot (f_mix ~ CR).
  * ADIABAT enters through acceleration feedthrough -- a lower adiabat makes a
    thinner, less ablatively-stabilized shell, so surface ripples feed through and
    seed the deceleration RT more strongly (A_acc rises as adiabat falls).

Calibrated so the NIF-like nominal (CR=35, adiabat=2.0, sigma_surf=SIGMA_SURF_REF)
keeps ~85% of clean yield -- mix-degraded but igniting, as N221204 was -- while an
aggressive high-CR / low-adiabat design falls off the mix cliff.

Deceleration growth is the slow part; build_rt_table.py caches GF(ell; CR) so this
evaluates in microseconds. Run:  python3 rt_mix.py
Deps: numpy  (matplotlib only for the figure)
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rayleigh-Taylor"))
from rt_mechanics import gamma_ablative, G_ACCEL, ATWOOD, T_ACCEL  # noqa: E402

# --- load the cached deceleration-phase growth-factor spectrum GF(ell; CR) ---
_TBL = os.path.join(os.path.dirname(__file__), "rt_table.json")
with open(_TBL) as f:
    _T = json.load(f)
ELL = np.arange(1, _T["ell_max"] + 1)
R0_M = _T["r0_m"]
_CR = np.array([r["cr"] for r in _T["rows"]])
_GF = np.array([r["GF"] for r in _T["rows"]])          # [n_cr, n_ell]

# --- reference design / calibration knobs (pinned so NIF-nominal keeps ~85%) ---
SIGMA_SURF_REF = 2.1e-8     # reference capsule surface roughness (RMS) [m]  (~21 nm, calibrated)
SPEC_SLOPE     = 1.0        # surface power spectrum  s(ell) ~ ell^-SPEC_SLOPE
ELL_CUT        = 35.0       # short-wavelength cutoff: finite hot-spot interface
                            # thickness stabilizes modes shorter than R_min/ELL_CUT.
                            # (convergent_rt omits this, so GF rises to the truncation;
                            #  restoring the cutoff is what makes the mix width physical.)
ALPHA_REF      = 2.0        # reference in-flight adiabat
ACCEL_ADIA_EXP = 2.0        # ablation velocity ~ adiabat^ACCEL_ADIA_EXP (feedthrough lever)
N_ACC_CAP      = 6.0        # cap on acceleration e-foldings (nonlinear saturation)
K_IFAR         = 9.7        # in-flight-aspect-ratio amplification (calibrated): a thinner
                            # shell at high CR / low adiabat feeds RT through exponentially.
                            # A fixed-trajectory linear solve omits this; it is the dominant
                            # (CR, adiabat) sensitivity of the mix, so it is put back here.
P_MIX          = 2.5        # clean-fraction exponent in the yield penalty
CR_REF         = 35.0
V_ABL_REF      = 5.0e3      # ablation velocity at the reference adiabat [m/s]

# Bell-Plesset: a mode amplitude grows geometrically ~ R0/R = CR through convergence.
# The cached fixed-trajectory solve (modes seeded from rest) under-represents this,
# so restore it explicitly -- combined with R_min = R0/CR it makes f_mix ~ CR^2, the
# strong, adiabat-independent convergence penalty that a high-CR design cannot escape.


def _seed_spectrum():
    """Normalized surface-roughness amplitude per mode, sum_ell s^2 = 1."""
    s = ELL.astype(float) ** (-SPEC_SLOPE)
    return s / np.sqrt(np.sum(s ** 2))


_S = _seed_spectrum()
_CUTOFF = np.exp(-(ELL / ELL_CUT) ** 2)     # interface-thickness short-wavelength rolloff


def accel_feedthrough(adiabat):
    """Acceleration-phase ablative amplification A_acc(ell) of the seed ripples.

    Uses the Takabe ablative growth rate over the acceleration phase; a lower
    adiabat lowers the ablation velocity (thinner, less-stabilized shell), so the
    short-wavelength modes grow more before deceleration.
    """
    V_abl = V_ABL_REF * (adiabat / ALPHA_REF) ** ACCEL_ADIA_EXP
    lam = 2.0 * np.pi * R0_M / ELL                       # wavelength of mode ell
    n_e = np.minimum(gamma_ablative(lam, g=G_ACCEL, A=ATWOOD, Va=V_abl) * T_ACCEL, N_ACC_CAP)
    return np.exp(n_e)


def _gf_at(cr):
    """Interpolate the deceleration growth-factor spectrum GF(ell) at convergence cr."""
    cr = float(np.clip(cr, _CR.min(), _CR.max()))
    return np.array([np.interp(cr, _CR, _GF[:, j]) for j in range(_GF.shape[1])])


def ifar_amplification(cr, adiabat):
    """Shell-thinning (in-flight-aspect-ratio) RT amplification, 1.0 at the reference.

    Higher CR and lower adiabat both thin the in-flight shell, raising the aspect
    ratio and feeding perturbations through exponentially -- the dominant (CR,
    adiabat) sensitivity that a fixed-trajectory linear solve does not capture.
    """
    ifar = np.sqrt((cr / CR_REF) * (ALPHA_REF / adiabat))
    return np.exp(K_IFAR * (ifar - 1.0))


def mix_state(cr, adiabat, sigma_surf=SIGMA_SURF_REF):
    """Design -> (mix width [m], mix fraction, per-mode stagnation amplitudes)."""
    bp = cr / CR_REF                                              # Bell-Plesset amplitude ~ R0/R
    eta = (sigma_surf * _S * accel_feedthrough(adiabat) * _gf_at(cr) * _CUTOFF
           * ifar_amplification(cr, adiabat) * bp)                # [n_ell]
    sigma_mix = np.sqrt(np.sum(eta ** 2))
    r_min = R0_M / cr
    return dict(sigma_mix=sigma_mix, f_mix=sigma_mix / r_min, eta=eta, r_min=r_min)


def mix_penalty(cr, adiabat, sigma_surf=SIGMA_SURF_REF):
    """Yield multiplier from multimode mix: 1 = clean, ->0 = mix quenches the hot spot."""
    f = mix_state(cr, adiabat, sigma_surf)["f_mix"]
    return float(max(0.0, 1.0 - min(f, 1.0)) ** P_MIX)


def report():
    print("=" * 66)
    print("  MULTIMODE RT MIX  ->  IGNITION-MARGIN PENALTY")
    print("=" * 66)
    cases = [
        ("NIF-like nominal",   35.0, 2.0),
        ("aggressive (Step 2)", 44.0, 1.6),
        ("conservative",       28.0, 2.4),
    ]
    print(f"   {'design':<22}{'CR':>5}{'adiabat':>9}{'f_mix':>8}{'penalty':>9}")
    for name, cr, ad in cases:
        st = mix_state(cr, ad)
        M = mix_penalty(cr, ad)
        print(f"   {name:<22}{cr:>5.0f}{ad:>9.1f}{st['f_mix']:>8.2f}{M:>9.2f}")
    print("-" * 66)
    st = mix_state(35.0, 2.0)
    dom = int(ELL[np.argmax(st['eta'])])
    print(f"  nominal mix width          : {st['sigma_mix']*1e6:6.2f} um  "
          f"(hot spot R_min = {st['r_min']*1e6:.1f} um)")
    print(f"  dominant mix mode          : ell = {dom}")
    print(f"  -> nominal keeps {mix_penalty(35.0,2.0)*100:.0f}% of clean yield "
          f"(mix-degraded but igniting, as NIF was)")
    print("=" * 66)


def make_figure(fname="rt_mix.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) per-mode amplitude at stagnation, nominal design
    st = mix_state(35.0, 2.0)
    ax[0].semilogy(ELL, sigma_component(st) * 1e6, "tab:red", lw=2)
    ax[0].set_xlabel("Legendre mode  ell"); ax[0].set_ylabel("stagnation amplitude [um]")
    ax[0].set_title("multimode spectrum at stagnation\n(seed x feedthrough x growth)")

    # (2) penalty vs convergence, for three adiabats
    cr = np.linspace(_CR.min(), _CR.max(), 60)
    for ad, col in [(2.4, "tab:green"), (2.0, "tab:blue"), (1.6, "tab:red")]:
        M = [mix_penalty(c, ad) for c in cr]
        ax[1].plot(cr, M, col, lw=2.2, label=f"adiabat {ad}")
    ax[1].plot(35.0, mix_penalty(35.0, 2.0), "k*", ms=14)
    ax[1].text(35.4, mix_penalty(35.0, 2.0), "NIF-like", fontsize=8)
    ax[1].set_xlabel("convergence ratio"); ax[1].set_ylabel("yield multiplier (mix)")
    ax[1].set_title("the instability charge:\ncompression costs ignition margin")
    ax[1].legend(fontsize=8)

    # (3) penalty map over (CR, adiabat)
    C = np.linspace(_CR.min(), _CR.max(), 50)
    A = np.linspace(1.5, 2.6, 50)
    CC, AA = np.meshgrid(C, A)
    Z = np.vectorize(mix_penalty)(CC, AA)
    im = ax[2].contourf(CC, AA, Z, levels=20, cmap="viridis")
    ax[2].plot(35.0, 2.0, "w*", ms=14); ax[2].text(35.6, 2.0, "NIF", color="w", fontsize=8)
    fig.colorbar(im, ax=ax[2], label="yield multiplier")
    ax[2].set_xlabel("convergence ratio"); ax[2].set_ylabel("in-flight adiabat")
    ax[2].set_title("mix penalty landscape")

    fig.suptitle("Multimode RT mix width -> ignition-margin penalty", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def sigma_component(st):
    """Per-mode contribution to the mix width (for plotting)."""
    return np.abs(st["eta"])


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * This closes the loop Step 2 opened: gain now pays for compression. Feeding
#   mix_penalty(CR, adiabat, sigma_surf) into the whole-design objective removes
#   the incentive to rail CR high / adiabat low, and makes surface finish and
#   drive tolerances matter -- the basis for the robust optimization in
#   robust_design_ml.py.
# * Reduced closures: a power-law surface spectrum (real capsules have measured,
#   bumpy spectra), a scalar adiabat->ablation-velocity feedthrough law, quadrature
#   mode summation (no mode coupling / turbulent-mix saturation), and a clean-
#   fraction penalty rather than a resolved mix-layer burn. The deceleration growth
#   itself is the repo's convergent_rt linear solve, cached over CR.
# ----------------------------------------------------------------------------
