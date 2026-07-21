r"""
Hohlraum radiation symmetry from a view-factor calculation.

hohlraum_asymmetry.py needed a map  geometry -> drive-flux asymmetry a_l.  The
first cut was a hand-waved heuristic.  This replaces it with an actual reduced
VIEW-FACTOR computation of the X-ray flux the hohlraum wall delivers to the
capsule -- the thing that sets low-mode drive symmetry in indirect drive.

Picture: a gold cylinder (radius R_h, half-length H) with a laser-entrance hole
(LEH, radius R_leh) at each end.  The wall re-emits X-rays (near-uniform surface
brightness from albedo), plus brighter rings where the laser cones strike --
inner cones near the waist, outer cones near the LEH.  The LEH holes emit nothing
(energy escapes), so the poles run cold.  A capsule (radius R_c) sits at the
center.

For each point P on the capsule at polar angle theta, the incident flux is the
radiative-exchange integral over every wall element Q:

    I(theta) = sum_Q  S(Q) * max(cosQ,0) * max(cosP,0) / (pi r^2) * dA_Q

with r=|P-Q|, cosQ the emission angle at the wall, cosP the incidence angle at
the capsule (also the visibility test -- the convex capsule only sees its outward
hemisphere).  I(theta) is then decomposed into Legendre modes:

    I(theta) = I0 [ 1 + sum_n a_n P_n(cos theta) ]

The design knobs (case-to-capsule ratio, length/diameter, LEH size, inner/outer
cone power balance) move a_2, a_4 -- exactly the levers a hohlraum designer (or an
optimizer) turns.  symmetry(...) returns (a2, a4, a6); it is the geometry->a_l
map that hohlraum_asymmetry.performance() and the ML sweep consume.

Run:  python3 hohlraum_viewfactor.py
Deps: numpy, scipy  (matplotlib only for the optional figure)
"""

import numpy as np
from scipy.special import eval_legendre

# capsule radius sets the length unit; only ratios matter for a_n.
R_C = 1.0

# fixed model constants
LASER_FRAC   = 0.55        # fraction of wall emission that is localized laser spots
Z_INNER_FRAC = 0.35        # inner-cone ring axial position on the cylinder (fraction of H)
RING_WIDTH   = 0.12        # Gaussian ring width (fraction of H for z, of R_h for r)

# nominal (baseline) design
NOMINAL = dict(case_to_capsule=3.0, length_to_diameter=1.5,
               leh_radius_frac=0.50, inner_frac=0.50)


def _build_wall(geom, nz=70, nphi=64, nr=24):
    """Return element positions Q(N,3), inward normals n(N,3), areas dA(N),
    and source brightness S(N) for the hohlraum wall (cylinder + end annuli)."""
    R_h = geom["case_to_capsule"] * R_C
    H   = geom["length_to_diameter"] * R_h            # half-length
    R_leh = geom["leh_radius_frac"] * R_h
    inner_frac = np.clip(geom["inner_frac"], 0.0, 1.0)

    phi = (np.arange(nphi) + 0.5) * 2*np.pi/nphi
    cphi, sphi = np.cos(phi), np.sin(phi)

    # --- cylinder wall: uniform re-emission + INNER-cone ring (heats equator) ---
    z = np.linspace(-H, H, nz)
    dz = z[1] - z[0]
    ZZ, PP = np.meshgrid(z, phi, indexing="ij")           # (nz, nphi)
    cph, sph = np.cos(PP), np.sin(PP)
    Qc = np.stack([R_h*cph, R_h*sph, ZZ], axis=-1).reshape(-1, 3)
    nc = np.stack([-cph, -sph, np.zeros_like(cph)], axis=-1).reshape(-1, 3)
    dAc = np.full(Qc.shape[0], R_h * dz * (2*np.pi/nphi))
    sz = RING_WIDTH * H
    zf = ZZ.reshape(-1)
    g_in = np.exp(-((zf - Z_INNER_FRAC*H)**2)/(2*sz**2)) + np.exp(-((zf + Z_INNER_FRAC*H)**2)/(2*sz**2))
    g_in = g_in / np.sum(g_in * dAc)                      # normalize over cylinder area
    Sc = inner_frac * LASER_FRAC * g_in                  # inner-cone power

    # --- end annuli (LEH = hole r<R_leh): re-emission + OUTER-cone ring near the
    #     LEH, which faces the capsule poles (heats poles -> P2 control) ---
    r = np.linspace(R_leh, R_h, nr)
    dr = r[1] - r[0]
    Qe_l, ne_l, dAe_l, re_l = [], [], [], []
    for zc, nz_sign in [(H, -1.0), (-H, +1.0)]:
        RR, PP2 = np.meshgrid(r, phi, indexing="ij")
        cph2, sph2 = np.cos(PP2), np.sin(PP2)
        Qe_l.append(np.stack([RR*cph2, RR*sph2, np.full_like(RR, zc)], axis=-1).reshape(-1, 3))
        ne_l.append(np.tile([0.0, 0.0, nz_sign], (RR.size, 1)))
        dAe_l.append((RR * dr * (2*np.pi/nphi)).reshape(-1))
        re_l.append(RR.reshape(-1))
    Qe = np.concatenate(Qe_l); ne = np.concatenate(ne_l)
    dAe = np.concatenate(dAe_l); re = np.concatenate(re_l)
    sr = RING_WIDTH * R_h
    g_out = np.exp(-((re - (R_leh + 0.5*sr))**2)/(2*sr**2))   # ring just outside the LEH
    g_out = g_out / np.sum(g_out * dAe)
    Se = (1 - inner_frac) * LASER_FRAC * g_out               # outer-cone power

    Q = np.concatenate([Qc, Qe]); n = np.concatenate([nc, ne])
    dA = np.concatenate([dAc, dAe]); S = np.concatenate([Sc, Se])

    # uniform re-emission (constant brightness): total power splits
    # (1-LASER_FRAC) uniform : LASER_FRAC laser
    S = S + (1 - LASER_FRAC) / dA.sum()
    return Q, n, dA, S


def flux_profile(geom, theta):
    """Incident flux I(theta) on the capsule (arb units)."""
    Q, n, dA, S = _build_wall(geom)
    I = np.empty_like(theta)
    for i, th in enumerate(theta):
        P = R_C * np.array([np.sin(th), 0.0, np.cos(th)])
        nP = np.array([np.sin(th), 0.0, np.cos(th)])
        d = P - Q                                        # (N,3), points capsule<-wall
        r2 = np.einsum("ij,ij->i", d, d)
        r  = np.sqrt(r2)
        cosQ = np.einsum("ij,ij->i", d, n) / r           # emission angle at wall (d = P-Q)
        cosP = -(d @ nP) / r                             # incidence angle at capsule
        ker = np.clip(cosQ, 0, None) * np.clip(cosP, 0, None) / (np.pi * r2)
        I[i] = np.sum(S * ker * dA)
    return I


def legendre_modes(theta, I, nmax=6):
    """a_n from I(theta) = I0 [1 + sum a_n P_n(cos theta)]."""
    mu = np.cos(theta); w = np.sin(theta)
    c0 = np.trapezoid(I * w, theta)
    out = {}
    for nn in range(1, nmax+1):
        cn = np.trapezoid(I * eval_legendre(nn, mu) * w, theta)
        out[nn] = (2*nn + 1) * cn / c0
    return out


def symmetry(case_to_capsule=3.0, length_to_diameter=1.5,
             leh_radius_frac=0.50, inner_frac=0.50, ntheta=61):
    """geometry -> (a2, a4, a6) drive-flux asymmetry amplitudes."""
    geom = dict(case_to_capsule=case_to_capsule, length_to_diameter=length_to_diameter,
                leh_radius_frac=leh_radius_frac, inner_frac=inner_frac)
    theta = np.linspace(1e-4, np.pi - 1e-4, ntheta)
    I = flux_profile(geom, theta)
    m = legendre_modes(theta, I)
    return m[2], m[4], m[6]


def report():
    print("=" * 66)
    print("  HOHLRAUM VIEW-FACTOR SYMMETRY")
    print("=" * 66)
    a2, a4, a6 = symmetry(**NOMINAL)
    # odd modes should vanish by top/bottom symmetry (validation)
    theta = np.linspace(1e-4, np.pi-1e-4, 61)
    I = flux_profile(dict(NOMINAL), theta)
    m = legendre_modes(theta, I)
    print(f"  nominal design {NOMINAL}")
    print(f"    a2={a2*100:+6.2f}%   a4={a4*100:+6.2f}%   a6={a6*100:+6.2f}%")
    print(f"    odd-mode check (should ~0): a1={m[1]*100:+.2e}%  a3={m[3]*100:+.2e}%")
    print("-" * 66)
    print("  P2 vs inner/outer cone balance (the symmetry tuning knob):")
    for f in [0.2, 0.35, 0.5, 0.65, 0.8]:
        a2, a4, _ = symmetry(inner_frac=f)
        print(f"    inner_frac = {f:.2f}   a2 = {a2*100:+6.2f}%   a4 = {a4*100:+6.2f}%")
    print("-" * 66)
    print("  P2 vs LEH size (bigger hole -> colder poles -> more negative P2):")
    for lrf in [0.35, 0.50, 0.65]:
        a2, _, _ = symmetry(leh_radius_frac=lrf)
        print(f"    leh_radius_frac = {lrf:.2f}   a2 = {a2*100:+6.2f}%")
    print("-" * 66)
    # find the inner_frac that zeroes P2 at nominal (a symmetric design exists)
    fs = np.linspace(0.1, 0.9, 81)
    a2s = np.array([symmetry(inner_frac=f)[0] for f in fs])
    izero = np.argmin(np.abs(a2s))
    print(f"  symmetric drive (a2~0) at inner_frac = {fs[izero]:.3f}  "
          f"(a2 = {a2s[izero]*100:+.2f}%)")
    print("=" * 66)


def make_figure(fname="hohlraum_viewfactor.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: matplotlib unavailable -- {e}]")
        return
    theta = np.linspace(1e-4, np.pi-1e-4, 121)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    for f, col in [(0.25, "tab:blue"), (0.5, "tab:gray"), (0.75, "tab:red")]:
        I = flux_profile(dict(NOMINAL, inner_frac=f), theta)
        ax[0].plot(np.degrees(theta), I/I.mean(), col, lw=2, label=f"inner_frac={f}")
    ax[0].set_xlabel("polar angle [deg]"); ax[0].set_ylabel("normalized flux I/<I>")
    ax[0].set_title("drive flux vs cone balance"); ax[0].legend(fontsize=8)

    fs = np.linspace(0.1, 0.9, 60)
    a2 = [symmetry(inner_frac=f)[0]*100 for f in fs]
    a4 = [symmetry(inner_frac=f)[1]*100 for f in fs]
    ax[1].plot(fs, a2, "tab:red", lw=2.5, label="P2")
    ax[1].plot(fs, a4, "tab:orange", lw=2, ls="--", label="P4")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("inner-cone power fraction"); ax[1].set_ylabel("asymmetry [%]")
    ax[1].set_title("symmetry tuning"); ax[1].legend(fontsize=9)

    lrf = np.linspace(0.3, 0.7, 40)
    for f, col, lab in [(0.5, "tab:blue", "balanced cones")]:
        a2 = [symmetry(leh_radius_frac=L, inner_frac=f)[0]*100 for L in lrf]
        ax[2].plot(lrf, a2, col, lw=2.5, label=lab)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("LEH radius fraction"); ax[2].set_ylabel("P2 [%]")
    ax[2].set_title("bigger LEH -> colder poles"); ax[2].legend(fontsize=9)

    fig.suptitle("Hohlraum view-factor: geometry sets the drive symmetry", fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES / simplifications -----------------------------------------------------
# * Static, grey, single-bounce view factor: uniform-brightness wall re-emission
#   + Gaussian laser rings + dark LEH holes. No time-dependent symmetry "swing",
#   no multi-bounce radiation transport, no wall motion, no cross-beam energy
#   transfer. It captures how GEOMETRY sets the low-mode drive pattern -- enough
#   for symmetry (a2, a4) to respond correctly to the design knobs, which is what
#   the ML search needs.
# * Validation: odd modes vanish (top/bottom symmetry); inner/outer cone balance
#   tunes P2 through zero; a bigger LEH pushes P2 negative (colder poles).
# ----------------------------------------------------------------------------
