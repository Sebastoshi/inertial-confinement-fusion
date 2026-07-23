r"""
Time-resolved implosion timeline -- one animatable model of an ICF shot.

The repo's other models each give a *number* (a gain, a mix penalty) or a single
stage. This one stitches the stages into a single evolution in time -- the thing you
would actually watch:

    coast  ->  convergence  ->  stagnation  ->  ignition (or fizzle)  ->  disassembly

driven by the same physics the rest of the repo is built on and anchored to the same
NIF calibration, so the two preset shots reproduce their real gains:

    * implosion trajectory R(t)   -- shell coasts inward, decelerates on the hot-spot
      gas cushion, stagnates at R_min = R0/CR, and rebounds (a tuned 2-var ODE);
    * compression      T(t), rho(t), rho*R(t) rise adiabatically as the fuel converges,
      reaching the coupled-model stagnation state (hot-spot T from the 1-D hydro fit);
    * burn             the repo's 0-D hot-spot ODE (Bosch-Hale reactivity, alpha
      heating vs bremsstrahlung) seeded at stagnation -- it runs away (ignites) or
      decays (fizzles); its T(t) is the ignition spike;
    * mix              the multimode RT penalty thins the confining areal density, so
      an aggressive / rough-surface design fizzles where a clean one ignites.

This is the Python reference. A JavaScript port drives the interactive dashboard;
`simulate()` returns everything the animation needs (time series + verdict + gain).

Presets: NIF N221204 (Dec 2022, gain 1.54) and the NIF record shot (Apr 2025, gain
4.13) -- same ~2 MJ laser, a better capsule (more compression, less mix), exactly the
lesson the earlier steps draw.

Run:  python3 implosion_timeline.py
Deps: numpy, scipy  (matplotlib only for the figure)
"""

import os
import importlib.util
from dataclasses import dataclass

import numpy as np


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cg = _load("coupled_gain", "coupled_gain.py")
mix = _load("rt_mix", "rt_mix.py")
gm = cg.gm                      # gain_model (efficiencies, stagnation scalings, burn)
hs = gm.hotspot_0d              # 0-D burn physics (reactivity, p_alpha, p_brem, burn)

R0 = 1.0e-3                     # initial capsule / shell radius [m]
GAMMA = 5.0 / 3.0
V_IMP_NIF = 3.7e5              # implosion velocity at the NIF reference energy [m/s]


@dataclass
class Design:
    E_MJ: float        # laser energy on target [MJ]
    fuel_ug: float     # DT fuel mass [micrograms]  (NIF ~ 210 ug = 0.21e-6 kg)
    CR: float          # convergence ratio R0/R_min
    adiabat: float     # in-flight adiabat
    surf_nm: float     # capsule surface finish (RMS) [nm]  -- HIGH-mode RT seed (mix)
    drive_asym_pct: float = 0.0   # P2 drive (pressure-wave) asymmetry [%] -- LOW-mode RT


# preset shots -- capsule knobs chosen so the coupled physics reproduces the real gains
PRESETS = {
    "NIF 2022 (first ignition, gain 1.5)": Design(2.05, 210.0, 35.0, 2.00, 21.0),
    "NIF 2025 (record, gain 4.1)":         Design(2.08, 225.0, 44.0, 1.70, 2.0),
}

MIX_REF = mix.mix_penalty(35.0, 2.0, mix.SIGMA_SURF_REF)   # nominal mix penalty (~0.85)


def mix_factor(d):
    """Mix penalty relative to the NIF-like nominal (as in robust_design_ml)."""
    return min(mix.mix_penalty(d.CR, d.adiabat, d.surf_nm * 1e-9) / MIX_REF, 1.0 / MIX_REF)


# P2 drive (pressure-wave) asymmetry -> yield-over-clean. Low-mode drive asymmetry
# seeds an l=2 perturbation that spherical convergence + deceleration RT amplify into a
# peanut-shaped hot spot, degrading yield. Calibrated to Rayleigh-Taylor/hohlraum_asymmetry.py:
# 1% P2 -> YOC ~0.72, 2% -> ~0.35, 5% -> ~0.02 (the ignition cliff, not a fitted curve).
ASYM_A0, ASYM_N = 0.016, 2.3


def asym_yoc(drive_asym_pct):
    """Yield-over-clean for a given P2 drive asymmetry [%] (1.0 = perfectly symmetric)."""
    a2 = max(drive_asym_pct, 0.0) / 100.0
    return 1.0 / (1.0 + (a2 / ASYM_A0) ** ASYM_N)


def implosion_velocity(E_MJ):
    """Laser energy -> implosion velocity [m/s] (rocket coupling ~ sqrt(energy))."""
    return V_IMP_NIF * (E_MJ / 2.05) ** 0.5


def _trajectory(v_imp, CR, n=1400):
    """Shell coasts in at v_imp and decelerates on a gas cushion tuned to reach R_min=R0/CR.

    Returns (t, R): the radius history through stagnation and part of the rebound.
    R'' = K (R0/R)^(3 gamma) outward; K bisected so min(R) == R0/CR.
    """
    R_min_target = R0 / CR
    exp = 3.0 * GAMMA

    def run(K, tmax, steps):
        dt = tmax / steps
        r, v = R0, -v_imp
        T = np.empty(steps + 1); R = np.empty(steps + 1)
        rmin = R0
        for i in range(steps + 1):
            T[i], R[i] = i * dt, r
            a = K * (R0 / max(r, 1e-9)) ** exp
            # RK2 (midpoint) on (r, v); r' = v, v' = a(r)
            rm = r + 0.5 * dt * v
            vm = v + 0.5 * dt * a
            am = K * (R0 / max(rm, 1e-9)) ** exp
            r = r + dt * vm
            v = v + dt * am
            rmin = min(rmin, r)
            if r < R0 / 400.0:
                R[i:] = r; T[i:] = np.arange(i, steps + 1) * dt
                break
        return T, R, rmin

    # time to stagnation ~ R0 / v_imp; integrate a bit past for the rebound
    tmax = 2.6 * R0 / v_imp
    lo, hi = 1e6, 1e18
    for _ in range(60):                      # bisection on K to hit the target CR
        K = np.sqrt(lo * hi)
        _, _, rmin = run(K, tmax, 500)
        if rmin < R_min_target:              # too much convergence -> stiffen cushion
            lo = K
        else:
            hi = K
    K = np.sqrt(lo * hi)
    return run(K, tmax, n)[:2]


def simulate(d, npts=600):
    """Run one shot. Returns the full time series + scalar outcomes for the animation."""
    E_J = d.E_MJ * 1e6
    v_imp = implosion_velocity(d.E_MJ)

    # --- stagnation state, from the validated coupled model ---
    T_stag = cg.hydro_T_hs(cg.drive_pressure(E_J))                      # hot-spot T [keV]
    rhoR_stag = gm.RHOR_REF * (d.CR / gm.CR_REF) ** 2 * (gm.ALPHA_REF / d.adiabat)  # g/cm^2
    rho_stag = gm.RHO_HS                                                # g/cc
    mix_pen = mix.mix_penalty(d.CR, d.adiabat, d.surf_nm * 1e-9)        # yield multiplier
    asym = asym_yoc(getattr(d, "drive_asym_pct", 0.0))                  # P2 (low-mode) YOC
    # both instabilities thin the confining hot spot -> a smaller burning core
    rhoR_eff = rhoR_stag * max((mix_pen * asym) ** 0.3, 0.03)

    # --- scalar gain: the validated pipeline (coupled x mix) x drive-asymmetry YOC ---
    gain = cg.evaluate(cg.Design(E_J, d.fuel_ug * 1e-9, d.CR, d.adiabat))["gain"] * mix_factor(d) * asym
    yield_J = gain * E_J

    # --- implosion trajectory R(t) ---
    t_tr, R_tr = _trajectory(v_imp, d.CR)
    istag = int(np.argmin(R_tr))
    t_stag, R_min = t_tr[istag], R_tr[istag]

    # --- burn at stagnation (repo's 0-D hot-spot ODE gives the T(t) ignition/fizzle shape) ---
    b = gm.burn(rho_stag, rhoR_eff, T_stag, n_tau=gm.N_TAU_BURN)
    burnup = b["burnup"]
    tau_c = b["tau_c"]
    phi = 1.0 - b["n_i"] / b["n_i"][0]                                  # burned fraction vs burn-time
    ignites = b["T"].max() > 1.6 * T_stag                              # clear thermal runaway

    # --- assemble a single timeline: compression -> burn spike -> disassembly ---
    # non-uniform grid: dense through stagnation + the (tens-of-ps) burn so the
    # ignition spike is resolved; index-based playback then slows down there for free.
    t_end = t_tr[-1] + tau_c
    n1, n2 = int(npts * 0.47), int(npts * 0.36)
    n3 = npts - n1 - n2
    pre = np.linspace(0.0, t_stag, n1, endpoint=False)          # coast (0 .. <t_stag)
    mid = np.linspace(t_stag, t_stag + tau_c, n2)               # stagnation + burn (dense)
    post = np.linspace(t_stag + tau_c, t_end, n3 + 1)[1:]       # disassembly
    t = np.concatenate([pre, mid, post])
    R = np.interp(t, t_tr, R_tr)
    comp = np.clip(R_min / np.maximum(R, 1e-12), 0.0, 1.0)              # 0 (start) -> 1 (stagnation)
    T = T_stag * comp ** 2                                              # adiabatic compression heating
    rho = rho_stag * comp ** 3
    rhoR = rhoR_stag * comp ** 2
    gain_t = np.zeros_like(t)

    # overlay the burn window [t_stag, t_stag + tau_c]: real T(t) and ramping gain
    win = (t >= t_stag) & (t <= t_stag + tau_c)
    tb = t[win] - t_stag
    T[win] = np.interp(tb, b["t"], b["T"])
    rho[win] = rho_stag; rhoR[win] = rhoR_stag
    gain_t[win] = gain * np.interp(tb, b["t"], phi) / max(phi[-1], 1e-9)
    after = t > t_stag + tau_c
    gain_t[after] = gain
    # post-burn disassembly: expansion cooling as the shell rebounds
    T[after] = max(b["T"][-1], T_stag) * np.clip(R_min / np.maximum(R[after], 1e-12), 0.0, 1.0) ** 2

    clean = mix_pen * asym                                             # combined RT survival
    verdict = "QUENCHED" if clean < 0.25 else ("IGNITES" if ignites and gain >= 1.0 else "MARGINAL")
    return dict(
        t=t, R=R, T=T, rho=rho, rhoR=rhoR, gain_t=gain_t,
        t_stag=t_stag, R_min=R_min, T_stag=T_stag, T_peak=float(T.max()),
        rhoR_stag=rhoR_stag, mix_penalty=mix_pen, asym_yoc=asym,
        drive_asym_pct=float(getattr(d, "drive_asym_pct", 0.0)), burnup=burnup,
        gain=gain, yield_MJ=yield_J / 1e6, ignites=bool(ignites), verdict=verdict,
        v_imp=v_imp, tau_c=tau_c,
    )


def report():
    print("=" * 70)
    print("  IMPLOSION TIMELINE  --  coast -> stagnation -> ignition/fizzle")
    print("=" * 70)
    for name, d in PRESETS.items():
        s = simulate(d)
        print(f"\n  {name}")
        print(f"    inputs : E={d.E_MJ} MJ  fuel={d.fuel_ug} ug  CR={d.CR}  "
              f"adiabat={d.adiabat}  surface={d.surf_nm} nm")
        print(f"    v_imp {s['v_imp']/1e3:.0f} km/s  ->  stagnation at "
              f"{s['t_stag']*1e9:.2f} ns,  R_min {s['R_min']*1e6:.1f} um")
        print(f"    hot-spot T {s['T_stag']:.1f} keV   peak T {s['T_peak']:.1f} keV   "
              f"rho*R {s['rhoR_stag']:.2f} g/cm^2   mix {s['mix_penalty']*100:.0f}%")
        print(f"    burn-up {s['burnup']*100:.1f}%   yield {s['yield_MJ']:.2f} MJ   "
              f"GAIN {s['gain']:.2f}   [{s['verdict']}]")
    print("=" * 70)


def make_figure(fname="implosion_timeline.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"NIF 2022 (first ignition, gain 1.5)": "tab:blue",
              "NIF 2025 (record, gain 4.1)": "tab:red"}
    sims = {name: simulate(d) for name, d in PRESETS.items()}

    for name, s in sims.items():
        c = colors[name]; lab = name.split(" (")[0]
        ax[0].plot(s["t"] * 1e9, s["R"] * 1e6, c, lw=2, label=lab)
        ax[1].plot(s["t"] * 1e9, s["T"], c, lw=2, label=lab)
        ax[2].plot(s["t"] * 1e9, s["gain_t"], c, lw=2, label=f"{lab}  (gain {s['gain']:.1f})")

    ax[0].set_xlabel("time [ns]"); ax[0].set_ylabel("shell radius [µm]")
    ax[0].set_title("implosion trajectory\ncoast → stagnation → rebound"); ax[0].legend(fontsize=8)
    ax[1].axhline(4.3, color="gray", ls=":", lw=1); ax[1].text(0.1, 4.6, "ignition ~4.3 keV", fontsize=8)
    ax[1].set_xlabel("time [ns]"); ax[1].set_ylabel("hot-spot T [keV]")
    ax[1].set_title("hot-spot temperature\ncompression + ignition spike"); ax[1].legend(fontsize=8)
    ax[2].axhline(1.0, color="k", ls=":", lw=1); ax[2].text(0.1, 1.1, "breakeven", fontsize=8)
    ax[2].set_xlabel("time [ns]"); ax[2].set_ylabel("cumulative gain")
    ax[2].set_title("gain accumulates during the burn"); ax[2].legend(fontsize=8)

    fig.suptitle("ICF implosion timeline: two NIF shots, same laser, different capsule", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * Reduced and stitched, not a rad-hydro solve: a tuned gas-cushion trajectory, an
#   adiabatic compression law for T/rho/rho*R, and the repo's 0-D burn spliced in at
#   stagnation -- all anchored to the coupled-gain calibration so the presets land on
#   the real NIF gains. The point is a faithful *shape* to animate, not a design code.
# * The mix penalty enters as a reduction of the confining areal density, so a rough
#   or over-aggressive capsule fizzles in the same burn ODE that ignites a clean one.
# * simulate() is the contract for the JS port: identical inputs -> identical arrays.
# ----------------------------------------------------------------------------
