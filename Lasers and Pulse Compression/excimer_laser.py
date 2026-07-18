"""
KrF excimer laser -- Xcimer Energy long-pulse architecture.

Xcimer Energy's inertial-fusion driver is built around large e-beam-pumped KrF
(248 nm) amplifiers -- the "Large Xcimer Amplifier" / Argos module (>100 kJ).
Their key move is counterintuitive: instead of generating the short (~ns) pulse
fusion needs directly, they generate a cheap, low-peak-power MICROSECOND pulse in
the excimer amplifier (they hold the record for the longest KrF laser pulse) and
then compress it ~1000x down to ~3 ns later with stimulated Brillouin scattering
(see sbs_compression.py). Long-pulse generation keeps the optics below damage
threshold and the pulsed-power cheap -- the enabling trick for low $/joule.

This models the long-pulse KrF amplifier stage with the two coupled laser rate
equations (upper-state density + cavity photon density). KrF's bound-free upper
state dissociates the instant it emits, so the lower level self-empties and the
medium is an ideal 4-level system with a short (~2 ns) upper-state lifetime --
which is exactly why it can be run as an efficient quasi-CW long-pulse extractor.

    dN/dt   = P(t) - c*sigma*N*phi - N/tau_up
    dphi/dt = c*sigma*N*phi - phi/tau_c + seed

Run:  python3 excimer_laser.py
Deps: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
# Constants and KrF parameters (cm, s, cm^-3)
# ----------------------------------------------------------------------------
C_LIGHT = 3.0e10
H_PLANCK = 6.626e-34
LAMBDA = 248e-9
HNU = H_PLANCK * 3.0e8 / LAMBDA            # KrF photon energy [J] (~5.0 eV)

SIGMA   = 2.5e-16          # stimulated-emission cross section [cm^2]
TAU_UP  = 2.0e-9           # effective upper-state lifetime [s] (quenching-limited)
BETA    = 1.0e-6           # spontaneous fraction seeding the mode

# Amplifier / cavity geometry (Argos-scale module)
L_CAV   = 200.0            # optical length [cm]
L_GAIN  = 150.0            # gain length [cm]
AREA    = 25000.0         # aperture [cm^2] (~1.6x1.6 m) -> ~100 kJ, Argos-class
T_OC    = 0.30
LOSS    = 0.04
T_RT    = 2.0 * L_CAV / C_LIGHT
TAU_C   = T_RT / (T_OC + LOSS)
FILL    = L_GAIN / L_CAV

# Long e-beam pump: microsecond flat-top (the Xcimer long pulse). Pump level set
# so the extraction fluence stays a few J/cm^2 -- low enough for the optics,
# which is the whole reason for going long-pulse then compressing later.
PUMP_PEAK  = 4.0e22        # peak pump rate [cm^-3 s^-1]
PUMP_START = 50e-9
PUMP_DUR   = 1000e-9       # 1 microsecond flat-top
PUMP_EDGE  = 20e-9

# Xcimer SBS downstream: microsecond pulse compressed ~1000x to ~3 ns
SBS_OUT_NS = 3.0
SBS_RATIO  = 1000.0


def pump(t):
    up = 0.5 * (1 + np.tanh((t - PUMP_START) / PUMP_EDGE))
    dn = 0.5 * (1 + np.tanh((PUMP_START + PUMP_DUR - t) / PUMP_EDGE))
    return PUMP_PEAK * up * dn


def rhs(t, y):
    N, phi = y
    N = max(N, 0.0); phi = max(phi, 0.0)
    stim = C_LIGHT * SIGMA * N * phi
    dN = pump(t) - stim - N / TAU_UP
    dphi = FILL * stim - phi / TAU_C + FILL * BETA * N / TAU_UP
    return [dN, dphi]


def run():
    t_end = PUMP_START + PUMP_DUR + 120e-9
    sol = solve_ivp(rhs, (0.0, t_end), [0.0, 1.0], method="LSODA",
                    rtol=1e-7, atol=1e-2, dense_output=True, max_step=1e-9)
    t = np.linspace(0.0, t_end, 6000)
    N, phi = np.maximum(sol.sol(t), 0.0)

    Vmode = AREA * L_CAV
    P_out = phi * Vmode * HNU * (T_OC / T_RT)           # W
    P_pump = pump(t) * (AREA * L_GAIN) * HNU            # W

    E_out = np.trapz(P_out, t)
    E_pump = np.trapz(P_pump, t)
    eta = E_out / E_pump if E_pump > 0 else 0.0
    N_th = 1.0 / (C_LIGHT * SIGMA * TAU_C)

    lasing = P_out > 0.5 * P_out.max()
    pulse_dur = (t[lasing][-1] - t[lasing][0]) if lasing.any() else 0.0
    fluence = E_out / AREA                               # J/cm^2
    return dict(t=t, N=N, phi=phi, P_out=P_out, P_pump=P_pump,
                E_out=E_out, E_pump=E_pump, eta=eta, N_th=N_th,
                pulse_dur=pulse_dur, fluence=fluence, P_pk=P_out.max())


def report(d):
    print("=" * 62)
    print("  KrF EXCIMER (248 nm)  --  Xcimer long-pulse amplifier")
    print("=" * 62)
    print(f"  photon energy              : {HNU/1.602e-19:.2f} eV @ 248 nm")
    print(f"  aperture                   : {AREA:.0f} cm^2 (~{np.sqrt(AREA):.0f}x{np.sqrt(AREA):.0f} cm)")
    print(f"  peak upper-state density   : {d['N'].max():.2e} cm^-3 "
          f"({d['N'].max()/d['N_th']:.0f}x threshold)")
    print("-" * 62)
    print(f"  output pulse duration      : {d['pulse_dur']*1e9:7.0f} ns  (long pulse)")
    print(f"  extraction fluence         : {d['fluence']:7.2f} J/cm^2")
    print(f"  output energy              : {d['E_out']/1e3:7.1f} kJ  (Argos-class ~100 kJ)")
    print(f"  extraction efficiency      : {100*d['eta']:7.1f} %  (laser out / pumped)")
    print("-" * 62)
    print(f"  -> SBS compressor: {d['pulse_dur']*1e9:.0f} ns / {SBS_RATIO:.0f} "
          f"-> ~{SBS_OUT_NS:.0f} ns for the target")
    print("=" * 62)


def make_figure(d, fname="excimer_laser.png"):
    import matplotlib.pyplot as plt
    t = d["t"] * 1e9
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) pump and inversion (log) over the microsecond pulse
    a = ax[0]
    a.plot(t, d["P_pump"] / 1e9, "tab:orange", lw=2, label="e-beam pump")
    a.set_xlabel("time [ns]"); a.set_ylabel("pump power [GW]", color="tab:orange")
    a.tick_params(axis="y", labelcolor="tab:orange")
    a2 = a.twinx()
    a2.plot(t, np.maximum(d["N"], 1e10), "tab:blue", lw=2)
    a2.axhline(d["N_th"], color="tab:blue", ls=":", lw=1)
    a2.text(t[-1] * 0.5, d["N_th"] * 1.4, "lasing threshold", color="tab:blue", fontsize=8)
    a2.set_yscale("log"); a2.set_ylim(1e11, 3e15)
    a2.set_ylabel("upper-state N [cm$^{-3}$]", color="tab:blue")
    a2.tick_params(axis="y", labelcolor="tab:blue")
    a.set_title("microsecond e-beam pump drives\nquasi-steady KrF lasing")

    # (2) the long output pulse
    b = ax[1]
    b.plot(t, d["P_out"] / 1e9, "tab:red", lw=2)
    b.set_xlabel("time [ns]"); b.set_ylabel("output power [GW]")
    b.set_title(f"long-pulse output: {d['pulse_dur']*1e9:.0f} ns, "
                f"{d['E_out']/1e3:.0f} kJ")

    # (3) the Xcimer pipeline: long pulse -> SBS -> ~3 ns (log time)
    c = ax[2]
    long_ns = d["pulse_dur"] * 1e9
    tt = np.logspace(-1, np.log10(long_ns * 1.5), 500)
    long_pulse = np.where((tt > 5) & (tt < long_ns + 5), 1.0, 0.0)
    comp = np.exp(-0.5 * ((tt - SBS_OUT_NS) / (SBS_OUT_NS / 2.3548)) ** 2)
    c.semilogx(tt, long_pulse, "tab:red", lw=2, label=f"KrF out (~{long_ns:.0f} ns)")
    c.semilogx(tt, comp, "tab:purple", lw=2, label=f"after SBS (~{SBS_OUT_NS:.0f} ns)")
    c.annotate(f"SBS compress\n~{SBS_RATIO:.0f}x", xy=(SBS_OUT_NS, 0.6),
               xytext=(20, 0.55), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    c.set_xlabel("time [ns] (log)"); c.set_ylabel("normalized power")
    c.set_title("the Xcimer pipeline:\ncheap long pulse -> SBS -> fusion pulse")
    c.legend(fontsize=8, loc="upper left")

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
# * The whole point of the long pulse: peak power stays low (panel 2 is in GW,
#   not TW), so the amplifier optics never see damaging intensity -- the pulse is
#   only compressed to fusion-relevant power AFTER the expensive optics, in the
#   SBS gas cell. That is what makes Xcimer's $/joule low.
#
# * KrF at 248 nm is chosen for direct-drive ICF: short wavelength couples
#   efficiently to the ablator and suppresses laser-plasma instabilities. See
#   excimer_mixtures.py for how the other excimer gases compare.
#
# * Argos-scale here (~100 kJ/module); Xcimer's roadmap stacks ~40 modules ->
#   Vulcan at 4-12 MJ. Scale AREA to move along that line.
#
# Simplifications: single-mode lumped cavity (no transverse profile, ASE, or
# amplified-spontaneous background), fixed effective upper-state lifetime (real
# KrF is an e-beam plasma-chemistry network), and the SBS stage is only sketched
# here -- the actual compression physics is in sbs_compression.py.
# ----------------------------------------------------------------------------
