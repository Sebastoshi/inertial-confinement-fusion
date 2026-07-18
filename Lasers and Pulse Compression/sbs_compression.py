"""
Stimulated Brillouin Scattering (SBS) pulse compression cell.

An SBS compressor turns a long, energetic laser pulse into a short, intense one
with no gratings or gain medium -- just a cell of nonlinear liquid or high-
pressure gas. A long pump pulse enters; it stimulates a counter-propagating
acoustic (Brillouin) wave and scatters off it into a backward Stokes pulse. The
trick is transient: the *leading edge* of the Stokes pulse sees fresh, undepleted
pump and is amplified hard, while the tail sees pump the front already drained.
The Stokes pulse therefore steepens and compresses -- down to roughly the phonon
lifetime -- while sweeping up most of the pump energy. This is a standard way to
shorten ns pulses to sub-ns in high-power laser chains.

Model -- transient plane-wave SBS, three coupled envelopes (pump A_p forward,
Stokes A_s backward, acoustic Q):

    dA_p/dt + v dA_p/dz = -g * A_s * Q
    dA_s/dt - v dA_s/dz = +g * A_p * Q          (Stokes travels -z)
    dQ/dt               =  A_p * A_s - Q/tau_B  (phonon lifetime tau_B)

solved by exact counter-propagating advection on a characteristic grid
(dt = dz/v).

Run:  python3 sbs_compression.py
Deps: numpy, matplotlib
"""

import numpy as np

# ----------------------------------------------------------------------------
# Cell + pulse parameters (time in ns, length in cm)
# ----------------------------------------------------------------------------
V_MED   = 24.0            # light speed in the medium c/n [cm/ns]  (n~1.25)
L_CELL  = 100.0           # cell length [cm]  -> single transit ~4.2 ns
NZ      = 400             # spatial cells
TAU_B   = 1.0             # phonon (acoustic) lifetime [ns] -> sets compressed width
GAIN    = 6.0             # SBS coupling strength (tuned above threshold)
SEED    = 0.03            # Stokes seed at the far end (noise / spontaneous)

PUMP_A0    = 1.0          # pump amplitude (fields normalised to this)
PUMP_START = 1.0          # [ns]
PUMP_DUR   = 12.0         # pump flat-top duration [ns]
PUMP_EDGE  = 0.8          # rise/fall time [ns]


def pump_input(t):
    """Flat-top pump with smooth tanh edges."""
    up = 0.5 * (1 + np.tanh((t - PUMP_START) / PUMP_EDGE))
    dn = 0.5 * (1 + np.tanh((PUMP_START + PUMP_DUR - t) / PUMP_EDGE))
    return PUMP_A0 * up * dn


def run(gain=GAIN):
    dz = L_CELL / NZ
    dt = dz / V_MED                                     # exact advection step
    z = np.linspace(0.0, L_CELL, NZ + 1)
    Ap = np.zeros(NZ + 1)     # pump, forward (+z)
    As = np.zeros(NZ + 1)     # Stokes, backward (-z)
    Q = np.zeros(NZ + 1)      # acoustic amplitude

    t_end = PUMP_START + PUMP_DUR + 3 * L_CELL / V_MED
    nsteps = int(t_end / dt)
    hist = {"t": [], "pin": [], "sout": [], "pout": []}
    ST = []                   # Stokes space-time for the diagram

    t = 0.0
    for n in range(nsteps):
        sp = -gain * As * Q          # pump source
        ss = +gain * Ap * Q          # Stokes source
        # exact counter-propagating advection + source
        Ap_new = np.empty_like(Ap); As_new = np.empty_like(As)
        Ap_new[1:] = Ap[:-1] + dt * sp[:-1]
        As_new[:-1] = As[1:] + dt * ss[1:]
        Ap_new[0] = pump_input(t)                       # pump enters at z=0
        As_new[-1] = SEED                               # Stokes seed at z=L
        # acoustic response (local ODE)
        Q = Q + dt * (Ap * As - Q / TAU_B)
        Q = np.maximum(Q, 0.0)
        Ap = np.maximum(Ap_new, 0.0); As = np.maximum(As_new, 0.0)
        t += dt

        hist["t"].append(t)
        hist["pin"].append(pump_input(t))               # pump intensity in
        hist["sout"].append(As[0] ** 2)                 # Stokes intensity out (z=0)
        hist["pout"].append(Ap[-1] ** 2)                # pump transmitted (z=L)
        if n % 6 == 0:
            ST.append(As ** 2)

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist, np.array(ST), z, dt


def metrics(hist):
    t = hist["t"]
    pin = hist["pin"] ** 2                               # pump intensity (A0^2 units)
    sout = hist["sout"]
    def fwhm(sig):
        pk = sig.max()
        if pk <= 0:
            return 0.0
        m = sig > 0.5 * pk
        return t[m][-1] - t[m][0]
    pump_w = fwhm(pin)
    stokes_w = fwhm(sout)
    cr = pump_w / stokes_w if stokes_w > 0 else 0.0
    e_in = np.trapz(pin, t)
    e_out = np.trapz(sout, t)
    eff = e_out / e_in if e_in > 0 else 0.0
    enh = sout.max() / pin.max() if pin.max() > 0 else 0.0
    return dict(pump_w=pump_w, stokes_w=stokes_w, cr=cr, eff=eff, enh=enh)


def report(m):
    print("=" * 60)
    print("  SBS PULSE COMPRESSION CELL")
    print("=" * 60)
    print(f"  cell / medium             : {L_CELL:.0f} cm, transit {L_CELL/V_MED:.1f} ns")
    print(f"  phonon lifetime tau_B     : {TAU_B:.2f} ns")
    print("-" * 60)
    print(f"  pump pulse width (FWHM)   : {m['pump_w']:6.2f} ns")
    print(f"  Stokes pulse width (FWHM) : {m['stokes_w']:6.2f} ns")
    print(f"  compression ratio         : {m['cr']:6.1f} x")
    print(f"  peak intensity enhancement: {m['enh']:6.1f} x")
    print(f"  energy efficiency         : {100*m['eff']:6.1f} %  (Stokes out / pump in)")
    print("=" * 60)


def make_figure(hist, ST, z, fname="sbs_compression.png"):
    import matplotlib.pyplot as plt
    t = hist["t"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) input pump vs compressed Stokes output
    a = ax[0]
    a.plot(t, hist["pin"] ** 2, "tab:orange", lw=2, label="pump in (long)")
    a.plot(t, hist["sout"], "tab:red", lw=2, label="Stokes out (compressed)")
    a.set_xlabel("time [ns]"); a.set_ylabel("intensity [arb.]")
    a.set_title("long pump -> short intense Stokes")
    a.legend(fontsize=9)

    # (2) pump depletion (transmitted pump has a hole)
    b = ax[1]
    b.plot(t, hist["pin"] ** 2, "tab:orange", lw=1.5, alpha=0.6, label="pump in")
    b.plot(t, hist["pout"], "tab:brown", lw=2, label="pump transmitted")
    b.set_xlabel("time [ns]"); b.set_ylabel("intensity [arb.]")
    b.set_title("the Stokes pulse drains the pump")
    b.legend(fontsize=9)

    # (3) Stokes space-time diagram
    c = ax[2]
    extent = [0, z[-1], t[-1], t[0]]
    c.imshow(ST, aspect="auto", extent=extent, cmap="inferno", origin="upper")
    c.set_xlabel("position z [cm]"); c.set_ylabel("time [ns]")
    c.set_title("Stokes intensity (z, t):\nbackward pulse steepening")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    hist, ST, z, dt = run()
    m = metrics(hist)
    report(m)
    make_figure(hist, ST, z)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * The compressed pulse width floors near the phonon lifetime TAU_B -- lower it
#   (a medium with faster acoustic damping) and the Stokes pulse gets shorter and
#   the compression ratio climbs.
#
# * Raise GAIN (longer cell, higher pump intensity, better medium) and conversion
#   sharpens: more pump depletion (panel 2), higher efficiency, bigger peak
#   enhancement -- until it saturates.
#
# * Why this pairs with the excimer/ICF theme: SBS compression (and SBS phase-
#   conjugate mirrors) is a standard tool for shortening and cleaning high-power
#   laser pulses in fusion-class chains.
#
# Simplifications: plane-wave (no focusing/transverse profile), real envelopes
# (no phase / Brillouin frequency shift), a single cell (real compressors often
# use generator+amplifier stages), constant gain and phonon lifetime, and a
# fixed seed rather than a true noise-initiated Stokes.
# ----------------------------------------------------------------------------
