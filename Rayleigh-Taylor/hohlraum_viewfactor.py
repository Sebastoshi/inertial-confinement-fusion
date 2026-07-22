r"""
Hohlraum radiation symmetry from a multi-bounce (radiosity) view-factor solve.

hohlraum_asymmetry.py needs a map  geometry -> drive-flux asymmetry a_l.  This
computes it from the actual radiation transport inside the hohlraum.

Picture: a gold cylinder (radius R_h, half-length H) with a laser-entrance hole
(LEH, radius R_leh) at each end.  Laser cones deposit power on the wall -- inner
cones near the waist, outer cones near the LEH.  The wall is a near-blackbody
that RE-EMITS a fraction (the albedo, ~0.85) of every X-ray that lands on it, so
the radiation bounces many times before it escapes through an LEH or is absorbed
by the capsule.  That multi-bounce is what smooths the drive; a designer trades it
against LEH loss and cone pointing.

The transport is a radiosity problem.  Discretize the wall into axisymmetric
rings i with laser input E_i and total emitted power P_i:

    P_i = E_i + albedo * sum_j VF[j->i] P_j      (emit + reflect what arrives)

VF[j->i] is the ring-to-ring view factor (a radiative-exchange integral).  Solve
the linear system for P, get the equilibrium wall brightness B_i = P_i / A_i, and
integrate that brightness onto the capsule:

    I(theta) = sum(wall)  B * max(cosQ,0) * max(cosP,0) / (pi r^2) * dA
    I(theta) = I0 [ 1 + sum_n a_n P_n(cos theta) ]

symmetry(...) returns (a2, a4, a6).  Set albedo=0 to recover single-bounce (only
the raw laser spots drive the capsule -- very asymmetric); raise it and the cavity
smooths.

Run:  python3 hohlraum_viewfactor.py
Deps: numpy, scipy  (matplotlib only for the optional figure)
"""

import numpy as np
from scipy.special import eval_legendre

R_C = 1.0                   # capsule radius sets the length unit; only ratios matter

ALBEDO       = 0.85         # wall X-ray albedo (re-emission fraction)
Z_INNER_FRAC = 0.35         # inner-cone ring axial position on the cylinder (fraction of H)
RING_WIDTH   = 0.12         # Gaussian ring width (fraction of H for z, of R_h for r)

NOMINAL = dict(case_to_capsule=3.0, length_to_diameter=1.5,
               leh_radius_frac=0.50, inner_frac=0.50)


def _wall(geom, nz=60, nphi=48, nr=20):
    """Axisymmetric rings + their azimuthal element cloud.

    Returns rings = (rep_pos, rep_normal, area, E_laser) and
            elems = (pos, normal, dA, ring_index).
    """
    R_h = geom["case_to_capsule"] * R_C
    H   = geom["length_to_diameter"] * R_h
    R_leh = geom["leh_radius_frac"] * R_h
    inner_frac = float(np.clip(geom["inner_frac"], 0.0, 1.0))
    phi = (np.arange(nphi) + 0.5) * 2*np.pi/nphi
    cph, sph = np.cos(phi), np.sin(phi)

    rep_pos, rep_nrm, area, Elas = [], [], [], []
    e_pos, e_nrm, e_dA, e_ring = [], [], [], []
    ring = 0

    # cylinder rings (inner-cone laser deposited here)
    z = np.linspace(-H, H, nz); dz = z[1] - z[0]
    sz = RING_WIDTH * H
    g_in = np.exp(-((z - Z_INNER_FRAC*H)**2)/(2*sz**2)) + np.exp(-((z + Z_INNER_FRAC*H)**2)/(2*sz**2))
    g_in = g_in / g_in.sum()
    a_cyl = 2*np.pi*R_h*dz
    for k in range(nz):
        rep_pos.append([R_h, 0.0, z[k]]); rep_nrm.append([-1.0, 0.0, 0.0])
        area.append(a_cyl); Elas.append(inner_frac * g_in[k])
        e_pos.append(np.stack([R_h*cph, R_h*sph, np.full(nphi, z[k])], axis=1))
        e_nrm.append(np.stack([-cph, -sph, np.zeros(nphi)], axis=1))
        e_dA.append(np.full(nphi, a_cyl/nphi)); e_ring.append(np.full(nphi, ring, int))
        ring += 1

    # end-cap rings (outer-cone laser near the LEH heats the poles)
    r = np.linspace(R_leh, R_h, nr); dr = r[1] - r[0]
    sr = RING_WIDTH * R_h
    g_out = np.exp(-((r - (R_leh + 0.5*sr))**2)/(2*sr**2)); g_out = g_out / g_out.sum()
    for zc, ns in [(H, -1.0), (-H, 1.0)]:
        for m in range(nr):
            a_ann = 2*np.pi*r[m]*dr
            rep_pos.append([r[m], 0.0, zc]); rep_nrm.append([0.0, 0.0, ns])
            area.append(a_ann); Elas.append(0.5 * (1-inner_frac) * g_out[m])
            e_pos.append(np.stack([r[m]*cph, r[m]*sph, np.full(nphi, zc)], axis=1))
            e_nrm.append(np.tile([0.0, 0.0, ns], (nphi, 1)))
            e_dA.append(np.full(nphi, a_ann/nphi)); e_ring.append(np.full(nphi, ring, int))
            ring += 1

    rings = (np.array(rep_pos), np.array(rep_nrm), np.array(area), np.array(Elas))
    elems = (np.concatenate(e_pos), np.concatenate(e_nrm),
             np.concatenate(e_dA), np.concatenate(e_ring))
    return rings, elems


def _view_factor(rings, elems):
    """VF[j, i] = fraction of ring j's emission that lands on ring i."""
    rep_pos, rep_nrm, _, _ = rings
    e_pos, e_nrm, e_dA, e_ring = elems
    nring = rep_pos.shape[0]
    VF = np.zeros((nring, nring))
    for j in range(nring):
        d = e_pos - rep_pos[j]                       # rep_j -> every element
        r2 = np.einsum("ij,ij->i", d, d); r2 = np.where(r2 < 1e-12, np.inf, r2)
        r = np.sqrt(r2)
        cosQ = (d @ rep_nrm[j]) / r                  # emission at rep_j
        cosB = -np.einsum("ij,ij->i", d, e_nrm) / r  # incidence at element
        ker = np.clip(cosQ, 0, None) * np.clip(cosB, 0, None) / (np.pi * r2) * e_dA
        np.add.at(VF[j], e_ring, ker)                # aggregate onto target rings
    return VF


def _brightness(geom, nz=60, nphi=48, nr=20, albedo=ALBEDO):
    """Solve radiosity for the equilibrium wall brightness B (per ring)."""
    rings, elems = _wall(geom, nz, nphi, nr)
    _, _, area, E = rings
    VF = _view_factor(rings, elems)
    # total power: P = E + albedo * VF^T P
    P = np.linalg.solve(np.eye(len(E)) - albedo * VF.T, E)
    B = P / area                                     # brightness per unit area
    return B, elems


def flux_profile(geom, theta, albedo=ALBEDO):
    """Incident flux I(theta) on the capsule from the equilibrium wall brightness."""
    B, (e_pos, e_nrm, e_dA, e_ring) = _brightness(geom, albedo=albedo)
    S = B[e_ring]                                    # source brightness per element
    I = np.empty_like(theta)
    for k, th in enumerate(theta):
        P = R_C * np.array([np.sin(th), 0.0, np.cos(th)])
        nP = np.array([np.sin(th), 0.0, np.cos(th)])
        d = P - e_pos
        r2 = np.einsum("ij,ij->i", d, d); r = np.sqrt(r2)
        cosQ = np.einsum("ij,ij->i", d, e_nrm) / r
        cosP = -(d @ nP) / r
        ker = np.clip(cosQ, 0, None) * np.clip(cosP, 0, None) / (np.pi * r2) * e_dA
        I[k] = np.sum(S * ker)
    return I


def legendre_modes(theta, I, nmax=6):
    mu = np.cos(theta); w = np.sin(theta)
    c0 = np.trapezoid(I * w, theta)
    return {n: (2*n + 1) * np.trapezoid(I * eval_legendre(n, mu) * w, theta) / c0
            for n in range(1, nmax + 1)}


def symmetry(case_to_capsule=3.0, length_to_diameter=1.5,
             leh_radius_frac=0.50, inner_frac=0.50, albedo=ALBEDO, ntheta=61):
    """geometry -> (a2, a4, a6) drive-flux asymmetry amplitudes."""
    geom = dict(case_to_capsule=case_to_capsule, length_to_diameter=length_to_diameter,
                leh_radius_frac=leh_radius_frac, inner_frac=inner_frac)
    theta = np.linspace(1e-4, np.pi - 1e-4, ntheta)
    m = legendre_modes(theta, flux_profile(geom, theta, albedo=albedo))
    return m[2], m[4], m[6]


def report():
    print("=" * 66)
    print("  HOHLRAUM VIEW-FACTOR SYMMETRY  (multi-bounce radiosity)")
    print("=" * 66)
    a2, a4, a6 = symmetry(**NOMINAL)
    theta = np.linspace(1e-4, np.pi-1e-4, 61)
    m = legendre_modes(theta, flux_profile(dict(NOMINAL), theta))
    print(f"  nominal {NOMINAL}, albedo={ALBEDO}")
    print(f"    a2={a2*100:+6.2f}%   a4={a4*100:+6.2f}%   a6={a6*100:+6.2f}%")
    print(f"    odd-mode check (should ~0): a1={m[1]*100:+.1e}%  a3={m[3]*100:+.1e}%")
    print("-" * 66)
    print("  albedo -> multi-bounce smooths the drive (|a2| at fixed cone imbalance):")
    for alb in [0.0, 0.5, 0.8, 0.9, 0.95]:
        a2 = symmetry(inner_frac=0.30, albedo=alb)[0]
        print(f"    albedo = {alb:.2f}   a2(inner_frac=0.30) = {a2*100:+6.2f}%")
    print("-" * 66)
    print("  P2 vs inner/outer cone balance (the symmetry tuning knob):")
    for f in [0.2, 0.35, 0.5, 0.65, 0.8]:
        a2, a4, _ = symmetry(inner_frac=f)
        print(f"    inner_frac = {f:.2f}   a2 = {a2*100:+6.2f}%   a4 = {a4*100:+6.2f}%")
    print("-" * 66)
    print("  P2 vs LEH size (bigger hole -> colder poles):")
    for lrf in [0.35, 0.50, 0.65]:
        print(f"    leh_radius_frac = {lrf:.2f}   a2 = {symmetry(leh_radius_frac=lrf)[0]*100:+6.2f}%")
    fs = np.linspace(0.1, 0.9, 81)
    a2s = np.array([symmetry(inner_frac=f)[0] for f in fs])
    iz = np.argmin(np.abs(a2s))
    print("-" * 66)
    print(f"  symmetric drive (a2~0) at inner_frac = {fs[iz]:.3f}  (a2 = {a2s[iz]*100:+.2f}%)")
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

    fs = np.linspace(0.1, 0.9, 50)
    ax[1].plot(fs, [symmetry(inner_frac=f)[0]*100 for f in fs], "tab:red", lw=2.5, label="P2")
    ax[1].plot(fs, [symmetry(inner_frac=f)[1]*100 for f in fs], "tab:orange", lw=2, ls="--", label="P4")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("inner-cone power fraction"); ax[1].set_ylabel("asymmetry [%]")
    ax[1].set_title("symmetry tuning"); ax[1].legend(fontsize=9)

    albs = np.linspace(0.0, 0.95, 40)
    for f, col, lab in [(0.30, "tab:blue", "inner_frac=0.30"), (0.70, "tab:green", "inner_frac=0.70")]:
        ax[2].plot(albs, [abs(symmetry(inner_frac=f, albedo=a)[0])*100 for a in albs], col, lw=2.5, label=lab)
    ax[2].set_xlabel("wall albedo"); ax[2].set_ylabel("|P2| [%]")
    ax[2].set_title("multi-bounce smooths the drive"); ax[2].legend(fontsize=9)

    fig.suptitle("Hohlraum radiosity: geometry + albedo set the drive symmetry", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


def main():
    report()
    make_figure()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# NOTES / simplifications -----------------------------------------------------
# * Multi-bounce radiosity: grey, static, diffuse (Lambertian) walls with a single
#   albedo, axisymmetric rings, laser deposited in Gaussian cone rings, dark LEH
#   holes. Captures the cavity SMOOTHING and how geometry + albedo set the low-mode
#   drive -- what the ML search needs. Not modelled: time-dependent symmetry swing,
#   wall motion / LEH closure, spectral (non-grey) transport, cross-beam energy
#   transfer, and the laser-to-x-ray conversion physics.
# * Validation: odd modes vanish (top/bottom symmetry); raising the albedo shrinks
#   |a2| at fixed cone imbalance (the cavity bounces wash out the spot pattern);
#   cone balance still tunes P2 through zero; a bigger LEH pushes P2 negative.
# ----------------------------------------------------------------------------
