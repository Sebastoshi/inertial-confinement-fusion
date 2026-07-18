# Lasers & pulse compression

Two toy models of hardware used to build and shape high-power laser pulses for
ICF and HED experiments: a **KrF excimer laser** (a deep-UV direct-drive driver)
and a **stimulated-Brillouin-scattering (SBS) pulse-compression cell**.

## `excimer_laser.py` — KrF excimer laser

![excimer laser](excimer_laser.png)

A gain-switched KrF oscillator, modeled with the two coupled laser rate equations
(upper-state density + cavity photon density). Excimer lasers (NRL's Nike /
Electra) are attractive ICF *direct-drive* drivers: 248 nm couples well to the
ablator and their broad bandwidth enables beam smoothing for the implosion
uniformity these models assume. The excimer signature is the **bound-free upper
state** — KrF* dissociates the instant it emits, emptying the lower laser level,
so the medium acts like an ideal 4-level system with a ~2 ns upper-state
lifetime, which forces fast pumping and naturally gives short pulses.

```bash
python3 excimer_laser.py
```

- pump (e-beam) builds the inversion ~150× past threshold; the photon field then
  avalanches and fires the pulse — with a **~10 ns gain-switch delay** between
  pump and laser peak (panel 2)
- ~34 ns output pulse, **~83% extraction efficiency** (laser out / pumped)
- efficiency rises then saturates with harder pumping (panel 3)

## `sbs_compression.py` — SBS pulse compression cell

![SBS compression](sbs_compression.png)

A **transient stimulated-Brillouin-scattering compressor**: a long pump pulse
counter-propagates against the Stokes wave it generates in a nonlinear cell, and
the Stokes leading edge — seeing fresh, undepleted pump — is amplified hard while
the tail starves, so the Stokes pulse steepens and compresses. Modeled with the
three coupled SBS envelope equations (pump, Stokes, acoustic) solved by exact
counter-propagating advection.

```bash
python3 sbs_compression.py
```

- **11 ns pump → 1.3 ns Stokes: ~9× compression**, with the compressed width
  flooring near the phonon lifetime `τ_B`
- **~74% energy efficiency**; panel 2 shows the classic pump-depletion hole as
  the Stokes pulse sweeps up the pump; panel 3 is the (z, t) diagram of the
  backward Stokes pulse steepening

SBS compression (and SBS phase-conjugate mirrors) is a standard, grating-free way
to shorten and clean high-power laser pulses in fusion-class chains.

## Caveats

Both are lumped/plane-wave toys: single-mode cavity and fixed effective
upper-state lifetime for the excimer (real KrF is a multi-species e-beam plasma
chemistry); plane-wave, real-envelope, single-cell SBS with constant gain and a
seeded (not noise-initiated) Stokes. Each file's `NOTES` block lists the knobs
and the omitted physics.
