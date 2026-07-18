"""
KrF excimer laser -- gain-switched oscillator rate-equation model.

Excimer lasers are the deep-UV workhorses of direct-drive ICF (e.g. NRL's Nike /
Electra KrF systems): 248 nm light with excellent beam uniformity and broad
bandwidth for beam smoothing. The gain medium is special -- KrF* is a *bound*
excited state that *dissociates* the instant it emits, so the lower laser level
empties itself and the medium behaves like an ideal 4-level system with a very
short (~ns) upper-state lifetime. That makes it naturally gain-switched: you
pump hard and fast (electron beam) and a short intense pulse follows.

This models a KrF oscillator with the standard two coupled rate equations:

    dN/dt   = P(t) - c*sigma*N*phi - N/tau_up          (upper-state density)
    dphi/dt = c*sigma*N*phi - phi/tau_c + seed         (cavity photon density)

and reports the output pulse, extraction efficiency, and the gain-switching
delay between pump and laser pulse.

Run:  python3 excimer_laser.py
Deps: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
# Physical constants and KrF parameters (CGS-ish: cm, s, cm^-3)
# ----------------------------------------------------------------------------
C_LIGHT = 3.0e10            # speed of light [cm/s]
H_PLANCK = 6.626e-34
LAMBDA = 248e-9            # KrF wavelength [m]
HNU = H_PLANCK * 3.0e8 / LAMBDA           # photon energy [J]  (~5.0 eV)

SIGMA   = 2.5e-16          # stimulated-emission cross section [cm^2]
TAU_UP  = 2.0e-9           # effective upper-state lifetime [s] (quenching-limited)
BETA    = 1.0e-6           # spontaneous fraction seeding the lasing mode

# Cavity
L_CAV   = 100.0            # cavity length [cm]
L_GAIN  = 60.0             # gain-medium length [cm]
AREA    = 1.0             # beam cross-section [cm^2]
T_OC    = 0.30            # output-coupler transmission
LOSS    = 0.04            # parasitic round-trip loss
T_RT    = 2.0 * L_CAV / C_LIGHT                    # round-trip time [s]
TAU_C   = T_RT / (T_OC + LOSS)                     # cavity photon lifetime [s]
FILL    = L_GAIN / L_CAV                            # gain fill factor

# Pump (electron-beam deposition into KrF*): Gaussian in time
PUMP_PEAK = 6.0e24         # peak pump rate [cm^-3 s^-1]
PUMP_T0   = 25.0e-9        # pump centre [s]
PUMP_FWHM = 22.0e-9        # pump FWHM [s]


def pump(t):
    s = PUMP_FWHM / 2.3548
    return PUMP_PEAK * np.exp(-0.5 * ((t - PUMP_T0) / s) ** 2)


def rhs(t, y):
    N, phi = y
    N = max(N, 0.0); phi = max(phi, 0.0)
    stim = C_LIGHT * SIGMA * N * phi
    dN = pump(t) - stim - N / TAU_UP
    dphi = FILL * stim - phi / TAU_C + FILL * BETA * N / TAU_UP
    return [dN, dphi]


def run():
    t_end = 90e-9
    sol = solve_ivp(rhs, (0.0, t_end), [0.0, 1.0], method="LSODA",
                    rtol=1e-8, atol=1e-2, dense_output=True, max_step=2e-11)
    t = np.linspace(0.0, t_end, 4000)
    N, phi = sol.sol(t)
    N = np.maximum(N, 0.0); phi = np.maximum(phi, 0.0)

    # output power through the coupler: photons leaving * photon energy
    Vmode = AREA * L_CAV                                # cm^3
    P_out = phi * Vmode * HNU * (T_OC / T_RT)           # W
    P_pump = pump(t) * (AREA * L_GAIN) * HNU            # W (into upper state)

    E_out = np.trapz(P_out, t)
    E_pump = np.trapz(P_pump, t)
    eta = E_out / E_pump if E_pump > 0 else 0.0
    N_th = 1.0 / (C_LIGHT * SIGMA * TAU_C)              # lasing threshold density

    # pulse metrics
    ipk = int(np.argmax(P_out))
    half = P_out > 0.5 * P_out[ipk]
    fwhm = (t[half][-1] - t[half][0]) if half.any() else 0.0
    delay = t[ipk] - PUMP_T0
    return dict(t=t, N=N, phi=phi, P_out=P_out, P_pump=P_pump,
                E_out=E_out, E_pump=E_pump, eta=eta, N_th=N_th,
                fwhm=fwhm, delay=delay, t_pk=t[ipk], P_pk=P_out[ipk])


def report(d):
    print("=" * 60)
    print("  KrF EXCIMER LASER  --  gain-switched oscillator")
    print("=" * 60)
    print(f"  wavelength / photon energy : 248 nm / {HNU/1.602e-19:.2f} eV")
    print(f"  small-signal threshold N   : {d['N_th']:.2e} cm^-3")
    print(f"  peak upper-state density   : {d['N'].max():.2e} cm^-3 "
          f"({d['N'].max()/d['N_th']:.0f}x threshold)")
    print("-" * 60)
    print(f"  output pulse FWHM          : {d['fwhm']*1e9:6.1f} ns")
    print(f"  gain-switch delay (pk-pump): {d['delay']*1e9:6.1f} ns")
    print(f"  peak output power          : {d['P_pk']/1e6:6.2f} MW")
    print(f"  output energy              : {d['E_out']*1e3:6.2f} mJ")
    print(f"  extraction efficiency      : {100*d['eta']:6.1f} %  (laser out / pumped)")
    print("=" * 60)


def make_figure(d, fname="excimer_laser.png"):
    import matplotlib.pyplot as plt
    t = d["t"] * 1e9
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) pump and upper-state population
    a = ax[0]
    a.plot(t, d["P_pump"] / 1e6, "tab:orange", lw=2, label="e-beam pump")
    a.set_xlabel("time [ns]"); a.set_ylabel("pump power [MW]", color="tab:orange")
    a.tick_params(axis="y", labelcolor="tab:orange")
    a2 = a.twinx()
    a2.plot(t, np.maximum(d["N"], 1e10), "tab:blue", lw=2, label="KrF* density")
    a2.axhline(d["N_th"], color="tab:blue", ls=":", lw=1)
    a2.text(42, d["N_th"] * 1.4, "lasing threshold", color="tab:blue", fontsize=8)
    a2.set_yscale("log"); a2.set_ylim(1e11, 3e15)
    a2.set_ylabel("upper-state N [cm$^{-3}$]", color="tab:blue")
    a2.tick_params(axis="y", labelcolor="tab:blue")
    a.set_title("pump builds inversion; it overshoots,\nfires the pulse, then decays past threshold")

    # (2) output laser pulse vs pump (gain-switch delay)
    b = ax[1]
    b.plot(t, d["P_pump"] / d["P_pump"].max(), "tab:orange", lw=1.5, alpha=0.7,
           label="pump (norm.)")
    b.plot(t, d["P_out"] / d["P_out"].max(), "tab:red", lw=2, label="laser out (norm.)")
    b.axvline(d["t_pk"] * 1e9, color="tab:red", ls=":", lw=1)
    b.annotate(f"gain-switch\ndelay {d['delay']*1e9:.0f} ns",
               xy=(PUMP_T0 * 1e9, 0.5), xytext=(d["t_pk"] * 1e9 + 3, 0.55),
               fontsize=8, arrowprops=dict(arrowstyle="<->", lw=0.8))
    b.set_xlabel("time [ns]"); b.set_ylabel("normalized power")
    b.set_title(f"gain-switched pulse: {d['fwhm']*1e9:.0f} ns FWHM")
    b.legend(fontsize=9)

    # (3) efficiency vs pump energy (scan)
    c = ax[2]
    global PUMP_PEAK
    p0 = PUMP_PEAK
    scales = np.linspace(0.3, 3.0, 16)
    etas, eouts = [], []
    for s in scales:
        PUMP_PEAK = p0 * s
        r = run()
        etas.append(100 * r["eta"]); eouts.append(r["E_out"] * 1e3)
    PUMP_PEAK = p0
    c.plot(np.array(eouts), etas, "o-", ms=3, color="tab:green")
    c.set_xlabel("output energy [mJ]"); c.set_ylabel("extraction efficiency [%]")
    c.set_title("efficiency rises then saturates\nwith harder pumping")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    d = run()
    report(d)
    make_figure(d)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The gain-switch delay (panel 2): the laser pulse lags the pump because the
#   inversion must build past threshold before the photon avalanche fires. Pump
#   harder (raise PUMP_PEAK) and the delay shrinks and the pulse sharpens.
#
# * The short upper-state lifetime (TAU_UP ~ 2 ns, quenching-limited) is the
#   excimer signature -- it forces fast pumping and gives the naturally short
#   pulse. Lengthen it and the laser behaves more like a storage laser.
#
# * KrF for ICF (NRL Nike/Electra): 248 nm couples well to the ablator, and the
#   broad bandwidth enables induced-spatial-incoherence beam smoothing -- key for
#   the direct-drive uniformity these implosion models assume.
#
# Simplifications: single longitudinal mode, spatially lumped cavity (no
# transverse profile or ASE), fixed effective upper-state lifetime (real KrF
# kinetics is a multi-species plasma-chemistry network: e-beam -> Kr+/F- ->
# KrF*), and no saturable absorber or injection seeding.
# ----------------------------------------------------------------------------
