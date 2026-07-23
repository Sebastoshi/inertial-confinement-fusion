r"""
Ablator doping -- from the tungsten dopant profile to preheat, stability, and gain.

Models the capsule-ablator innovation behind NIF's April-2025 record (gain > 4): a
*continuous gradient* of tungsten in the high-density-carbon ablator instead of a
discrete *step* layer -- "a dimmer switch instead of an on/off switch" (LLNL). Both
profiles peak near 0.44 atomic-percent W; the gradient just removes the abrupt density
jumps of a step layer.

This is a reduced, mechanism-level model -- NOT a radiation-hydro code -- but it closes
the causal chain the rest of the repo leaves implicit. A real prediction of the optimal
profile needs multigroup (spectral) radiation transport through the graded, W-doped
carbon, real opacity/EOS tables, and resolved multimode ablative RT: a HYDRA-class
effort. Here the three effects the article names each map onto a knob the gain model
already has, calibrated so the two extremes land on the two real NIF shots:

    dopant profile  ->  W column density        ->  M-band X-ray shielding  ->  PREHEAT -> ADIABAT
                    ->  density-gradient length  ->  ablative RT stabilization -> MIX (+ enables higher CR)
    (adiabat, mix, CR)  ->  coupled gain model   ->  target gain

  * a smoother gradient lets you bury MORE tungsten without a hydro penalty, so it
    shields the fuel from hard "M-band" X-rays better -> less preheat -> lower adiabat;
  * it removes the step interface's density jump -> longer density-gradient scale
    length -> the ablative Rayleigh-Taylor front is more stable -> less mix, and the
    implosion can be driven to higher convergence before it breaks up.

Run:  python3 ablator_doping.py
Deps: numpy  (matplotlib only for the figure)
"""
import os
import importlib.util

import numpy as np


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cg = _load("coupled_gain", "coupled_gain.py")
mix = _load("rt_mix", "rt_mix.py")
MIX_REF = mix.mix_penalty(35.0, 2.0, mix.SIGMA_SURF_REF)

# --- ablator / dopant constants ---
C_PEAK   = 0.44        # peak tungsten concentration [atomic %]  (NIF value)
RHO_HDC  = 3.5         # high-density-carbon (diamond) ablator density [g/cc]
A_W, A_C = 183.84, 12.011
X0       = 14.0        # dopant-layer center depth into the ablator [um]

# --- the shot the doping is applied to (held fixed; the profile is the variable) ---
E_MJ, FUEL_UG = 2.05, 216.0

# --- endpoints the model is pinned to (the two real NIF shots) ---
STEP_TARGET = dict(adiabat=2.00, surf_nm=21.0, CR=35.0)   # step layer  -> gain ~1.5 (2022)
GRAD_TARGET = dict(adiabat=1.70, surf_nm=2.0,  CR=44.0)   # full gradient -> gain ~4 (2025)


def massfrac(c_atpct):
    """Atomic-% tungsten in carbon -> tungsten mass fraction."""
    f = np.asarray(c_atpct) / 100.0
    return f * A_W / (f * A_W + (1.0 - f) * A_C)


def profile(x, g):
    """Dopant concentration [at%] vs depth x [um]. g in [0,1]: 0 = sharp step, 1 = gradient.

    A super-Gaussian whose order drops (flat-top/sharp -> smooth) and whose width grows
    as the doping is graded -- so grading both softens the edges and buries more W.
    """
    hw = 3.0 * (1.0 + 1.6 * g)                 # half-width [um] grows with grading
    n = 8.0 * (1.0 - g) + 1.0 * g              # super-Gaussian order 8 (step) -> 1 (gradient)
    return C_PEAK * np.exp(-np.abs((x - X0) / hw) ** (2.0 * n))


def props(g):
    """Profile -> (W column density [g/cm^2], min density-gradient scale length [um])."""
    x = np.linspace(0.0, 30.0, 3000)
    c = profile(x, g)
    rho_w = RHO_HDC * massfrac(c)              # tungsten mass density [g/cc]
    rho = RHO_HDC + rho_w                      # total ablator density [g/cc]
    dx_cm = (x[1] - x[0]) * 1e-4
    sigma_W = np.trapezoid(rho_w, dx=dx_cm)    # column density [g/cm^2]
    drho = np.gradient(rho, dx_cm)             # d(rho)/dx [g/cc/cm]
    L_min = rho.max() / (np.abs(drho).max() + 1e-30)   # steepest-edge scale length [cm]
    return sigma_W, L_min * 1e4                # L in um


# --- 2-point calibration: model structure sets the direction, the two NIF shots pin it ---
_sW0, _L0 = props(0.0)   # step
_sW1, _L1 = props(1.0)   # gradient
_a1 = (STEP_TARGET["adiabat"] - GRAD_TARGET["adiabat"]) / (_sW1 - _sW0)  # adiabat falls with W column
_a0 = STEP_TARGET["adiabat"] + _a1 * _sW0
_s1 = (np.log(GRAD_TARGET["surf_nm"]) - np.log(STEP_TARGET["surf_nm"])) / (_L1 - _L0)  # mix falls with L
_s0 = np.log(STEP_TARGET["surf_nm"]) - _s1 * _L0
_c1 = (GRAD_TARGET["CR"] - STEP_TARGET["CR"]) / (_L1 - _L0)              # CR rises with stability
_c0 = STEP_TARGET["CR"] - _c1 * _L0

adiabat_of = lambda sW: _a0 - _a1 * sW
surf_of    = lambda L: float(np.exp(_s0 + _s1 * L))
CR_of      = lambda L: _c0 + _c1 * L


def gain_of(CR, adiabat, surf_nm, E_MJ=E_MJ, fuel_ug=FUEL_UG):
    g0 = cg.evaluate(cg.Design(E_MJ * 1e6, fuel_ug * 1e-9, CR, adiabat))["gain"]
    mf = min(mix.mix_penalty(CR, adiabat, surf_nm * 1e-9) / MIX_REF, 1.0 / MIX_REF)
    return g0 * mf


def evaluate(g):
    """Grading g in [0,1] -> the derived design and its gain."""
    sW, L = props(g)
    ad, surf, CR = adiabat_of(sW), surf_of(L), CR_of(L)
    return dict(g=g, sigma_W=sW, L=L, adiabat=ad, surf_nm=surf, CR=CR, gain=gain_of(CR, ad, surf))


def report():
    print("=" * 70)
    print("  ABLATOR DOPING  --  tungsten step layer vs continuous gradient")
    print("=" * 70)
    print(f"  ablator: high-density carbon, peak W {C_PEAK} at%, held shot E={E_MJ} MJ, fuel={FUEL_UG:.0f} ug")
    print(f"   {'profile':<12}{'W col [ug/cm2]':>15}{'grad L [um]':>13}{'adiabat':>9}"
          f"{'surf [nm]':>11}{'CR':>6}{'GAIN':>8}")
    for name, g in [("step", 0.0), ("half-graded", 0.5), ("gradient", 1.0)]:
        s = evaluate(g)
        print(f"   {name:<12}{s['sigma_W']*1e6:>15.0f}{s['L']:>13.1f}{s['adiabat']:>9.2f}"
              f"{s['surf_nm']:>11.1f}{s['CR']:>6.1f}{s['gain']:>8.2f}")
    print("-" * 70)
    st, gr = evaluate(0.0), evaluate(1.0)
    print(f"  step layer  -> gain {st['gain']:.2f}   (NIF 2022, ~1.5)")
    print(f"  gradient    -> gain {gr['gain']:.2f}   (NIF 2025 record, >4)")
    print(f"  the gradient buries {gr['sigma_W']/st['sigma_W']:.1f}x the W column and softens the")
    print(f"  steepest edge {gr['L']/st['L']:.0f}x -> lower adiabat, less mix, higher convergence.")
    print("=" * 70)


def make_figure(fname="ablator_doping.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    x = np.linspace(0, 30, 600)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) the two dopant profiles
    ax[0].plot(x, profile(x, 0.0), "tab:red", lw=2.2, label="step layer")
    ax[0].plot(x, profile(x, 1.0), "tab:green", lw=2.2, label="continuous gradient")
    ax[0].fill_between(x, profile(x, 1.0), color="tab:green", alpha=0.10)
    ax[0].set_xlabel("depth into ablator [µm]"); ax[0].set_ylabel("tungsten [atomic %]")
    ax[0].set_title("the dopant profile\nstep vs continuous gradient"); ax[0].legend(fontsize=8)

    # (2) what each profile buys: adiabat and mix vs grading
    gs = np.linspace(0, 1, 40)
    S = [evaluate(g) for g in gs]
    axr = ax[1]
    axr.plot(gs, [s["adiabat"] for s in S], "tab:purple", lw=2.2, label="in-flight adiabat")
    axr.set_xlabel("grading   (0 = step,  1 = gradient)")
    axr.set_ylabel("adiabat", color="tab:purple"); axr.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = axr.twinx()
    ax2.plot(gs, [s["surf_nm"] for s in S], "tab:orange", lw=2.2, label="effective mix seed")
    ax2.set_ylabel("effective surface / mix [nm]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    axr.set_title("preheat -> adiabat, stability -> mix\nboth improve as doping is graded")

    # (3) the payoff: gain vs grading, the two shots marked
    ax[2].plot(gs, [s["gain"] for s in S], "tab:red", lw=2.6)
    ax[2].plot(0.0, evaluate(0.0)["gain"], "ko", ms=8)
    ax[2].plot(1.0, evaluate(1.0)["gain"], "k*", ms=16)
    ax[2].text(0.02, evaluate(0.0)["gain"] + 0.15, "step\nNIF 2022", fontsize=8)
    ax[2].text(0.72, evaluate(1.0)["gain"] - 0.5, "gradient\nNIF 2025", fontsize=8)
    ax[2].axhline(1.0, color="k", ls=":", lw=1)
    ax[2].set_xlabel("grading   (0 = step,  1 = gradient)"); ax[2].set_ylabel("target gain")
    ax[2].set_title("the payoff:\ngraded doping unlocks the gain jump")

    fig.suptitle("Ablator tungsten doping: step layer vs continuous gradient (reduced model)", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * Reduced and calibrated, not first-principles: the profile -> (W column, gradient
#   scale length) step is real geometry; the maps to adiabat / mix / CR are structured
#   (right sign and mechanism) but pinned to the two NIF shots, exactly as coupled_gain
#   pins its efficiencies. So the endpoints are anchored and the in-between is a smooth,
#   physically-ordered interpolation -- not a prediction of the optimal profile.
# * What a real version needs: multigroup radiation transport for the actual M-band
#   absorption through the graded W (hohlraum drive spectrum + W-doped-carbon opacity),
#   a real ablator EOS, and 2-D/3-D multimode ablative RT on the resolved density
#   profile. That is the radiation-hydro problem the toy deliberately stands in for.
# * The CR-enablement is the softest link: better stability genuinely lets a design push
#   convergence higher, but convergence is also set by velocity and pulse shape. Hold CR
#   fixed (edit CR_of to a constant) to see preheat+mix alone, which still lifts the gain.
# ----------------------------------------------------------------------------
