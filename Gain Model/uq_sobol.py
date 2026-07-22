r"""
Sensitivity / uncertainty quantification: which tolerance drives the yield scatter?

Step 3 showed a real shot scatters shot-to-shot as its tolerances scatter -- surface
finish, drive symmetry, in-flight adiabat, delivered energy -- and that the robust
design backs off the cliff to keep the ignition margin. This module answers the
engineering question that follows: OF those tolerances, which one actually drives the
variance, and therefore which spec is worth tightening?

It is a variance-based global sensitivity analysis (Sobol indices) of the effective
gain as a function of the four tolerance channels, at a fixed design:

  * first-order index  S_i  -- the fraction of gain variance removed if tolerance i
    alone were pinned to nominal (its "main effect").
  * total index       S_Ti -- the fraction involving tolerance i at all, including its
    interactions with the others. S_Ti >> S_i flags a tolerance that matters mostly
    through interactions (typical near the mix cliff, where adiabat and surface finish
    conspire).

Estimated with Saltelli sampling and the Jansen estimators (no SALib dependency).
The four inputs are the standardized tolerance draws (iid N(0,1)); the same transforms
as robust_design_ml.py map them to energy / adiabat / surface / symmetry. Analyzed at
both Step-3 design points -- the on-cliff deterministic optimum and the backed-off
robust design -- to show how the sensitivity ranking shifts with ignition margin.
Closes with a spec-tightening study: halve each tolerance in turn, and report the
resulting drop in gain std -- the actionable ranking of engineering requirements.

Run:  python3 uq_sobol.py
Deps: numpy, scipy  (matplotlib only for the figure)
"""

import os
import importlib.util

import numpy as np


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rd = _load("robust_design_ml", "robust_design_ml.py")   # reuse models + tolerance constants
cg, mix = rd.cg, rd.mix

TOL_NAMES = ["laser energy", "adiabat", "surface finish", "drive symmetry"]
TOL_SIG   = [rd.TOL_E, rd.TOL_ADIA, rd.TOL_LOGSIG, rd.TOL_SYM]

# Step-3 design points (from the committed robust_design_ml.py run, commit db82298):
# they differ only in convergence ratio -- on the cliff shoulder vs backed off.
DETERMINISTIC = np.array([1.96, 0.30, 42.82, 2.28, 4.08, 1.20, 0.45, 0.34])
ROBUST        = np.array([1.96, 0.30, 41.08, 2.28, 4.08, 1.20, 0.45, 0.34])


def gain_at(x, s0, z, tol=TOL_SIG):
    """Effective gain at design x for a standardized tolerance draw z (4-vector)."""
    E   = x[0] * (1.0 + tol[0] * z[0])
    ad  = max(1.2, x[3] + tol[1] * z[1])
    sig = mix.SIGMA_SURF_REF * np.exp(tol[2] * z[2])
    sym = max(0.0, s0 * (1.0 + tol[3] * z[3]))
    g0  = cg.evaluate(cg.Design(E * 1e6, x[1] * 1e-6, x[2], ad))["gain"]
    return g0 * rd.mix_factor(x[2], ad, sig) * sym


def _eval(x, s0, Z, tol=TOL_SIG):
    return np.array([gain_at(x, s0, z, tol) for z in Z])


def sobol(x, s0, N=1024, seed=0):
    """Saltelli sampling + Jansen estimators -> first-order S and total S_T (k=4)."""
    k = 4
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, k))
    B = rng.standard_normal((N, k))
    fA = _eval(x, s0, A)
    fB = _eval(x, s0, B)
    var = np.var(np.concatenate([fA, fB]))
    S, ST = np.zeros(k), np.zeros(k)
    for i in range(k):
        AB = A.copy(); AB[:, i] = B[:, i]
        fAB = _eval(x, s0, AB)
        S[i]  = np.mean(fB * (fAB - fA)) / var          # Saltelli first-order
        ST[i] = 0.5 * np.mean((fA - fAB) ** 2) / var     # Jansen total
    return dict(S=S, ST=ST, mean=float(np.mean(fA)), std=float(np.sqrt(var)))


def spec_tightening(x, s0, N=4000, seed=1):
    """Halve each tolerance in turn; report the resulting gain std (actionable ranking)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, 4))
    base = _eval(x, s0, Z).std()
    out = {}
    for i in range(4):
        tol = list(TOL_SIG); tol[i] = tol[i] * 0.5
        out[TOL_NAMES[i]] = _eval(x, s0, Z, tol).std()
    return base, out


def _report_design(name, x, s0):
    r = sobol(x, s0)
    print(f"  {name}:  mean gain {r['mean']:.2f}   std {r['std']:.2f}")
    print(f"   {'tolerance':<16}{'S_i':>8}{'S_Ti':>8}   variance driver")
    order = np.argsort(-r["ST"])
    for i in order:
        bar = "#" * int(max(0, r["ST"][i]) * 40)
        print(f"   {TOL_NAMES[i]:<16}{r['S'][i]:>8.2f}{r['ST'][i]:>8.2f}   |{bar}")
    return r


def main():
    print("building convergent-RT forward model (one-time)...")
    rd._FWD = rd.build_forward()
    rd.YOC_REF = rd.performance(*rd.symmetry(*rd.NOMINAL[4:]), fwd=rd._FWD)[0]

    print("=" * 72)
    print("  SOBOL SENSITIVITY OF GAIN TO TOLERANCES  (which spec drives the scatter)")
    print("=" * 72)
    s0_det = rd.sym_factor(DETERMINISTIC)
    s0_rob = rd.sym_factor(ROBUST)
    r_det = _report_design("DETERMINISTIC optimum (on the cliff shoulder, CR 42.8)", DETERMINISTIC, s0_det)
    print("-" * 72)
    r_rob = _report_design("ROBUST design (backed off, CR 41.1)", ROBUST, s0_rob)
    print("=" * 72)

    base, tight = spec_tightening(ROBUST, s0_rob)
    print("  SPEC TIGHTENING (robust design): gain std if each tolerance is HALVED")
    print(f"   baseline std = {base:.3f}")
    order = sorted(tight, key=lambda kname: tight[kname])
    for kname in order:
        drop = (base - tight[kname]) / base * 100
        print(f"   halve {kname:<16} -> std {tight[kname]:.3f}   ({drop:+.0f}% variance-width)")
    top = order[0]
    print("-" * 72)
    print(f"  -> tighten the {top.upper()} spec first: it removes the most yield scatter.")
    print("=" * 72)
    return dict(det=r_det, rob=r_rob, base=base, tight=tight, s0_rob=s0_rob)


def make_figure(info, fname="uq_sobol.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    y = np.arange(len(TOL_NAMES)); w = 0.38

    # (1) Sobol first-order vs total for the robust design
    r = info["rob"]
    ax[0].barh(y - w/2, r["S"], w, color="tab:blue", label="first-order S_i")
    ax[0].barh(y + w/2, r["ST"], w, color="tab:red", label="total S_Ti")
    ax[0].set_yticks(y); ax[0].set_yticklabels(TOL_NAMES)
    ax[0].set_xlabel("Sobol index (fraction of gain variance)")
    ax[0].set_title("what drives the yield scatter\n(robust design)")
    ax[0].legend(fontsize=8, loc="lower right")

    # (2) how the ranking shifts on vs off the cliff (total index)
    det, rob = info["det"]["ST"], info["rob"]["ST"]
    ax[1].barh(y - w/2, det, w, color="tab:orange", label="on cliff (CR 42.8)")
    ax[1].barh(y + w/2, rob, w, color="tab:green", label="robust (CR 41.1)")
    ax[1].set_yticks(y); ax[1].set_yticklabels(TOL_NAMES)
    ax[1].set_xlabel("total Sobol index  S_Ti")
    ax[1].set_title("ranking shifts with ignition margin:\ncliff proximity amplifies adiabat/surface")
    ax[1].legend(fontsize=8, loc="lower right")

    # (3) spec-tightening: gain std if each tolerance is halved
    names = list(info["tight"].keys())
    stds  = [info["tight"][n] for n in names]
    order = np.argsort(stds)
    ax[2].barh([names[i] for i in order], [stds[i] for i in order], color="tab:purple")
    ax[2].axvline(info["base"], color="k", ls="--", lw=1.5, label=f"baseline std {info['base']:.2f}")
    ax[2].set_xlabel("gain std after halving that tolerance")
    ax[2].set_title("spec-tightening payoff:\nwhich requirement to buy first")
    ax[2].legend(fontsize=8, loc="lower right")

    fig.suptitle("UQ: Sobol sensitivity of ICF gain to tolerances -> engineering specs", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


if __name__ == "__main__":
    make_figure(main())

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * This turns Step 3's tolerance model into requirements: the total Sobol index
#   ranks the tolerances by how much gain variance each carries, and the spec-
#   tightening study says which one to tighten for the biggest reduction in scatter.
# * S_Ti > S_i for adiabat / surface finish near the cliff is the signature of the
#   mix penalty: those two only bite when they push the design past the ignition
#   margin together, an interaction a one-at-a-time sensitivity sweep would miss --
#   which is exactly why a variance-based (Sobol) method is the right tool here.
# * Reduced model throughout (see each module's NOTES); the tolerance distributions
#   are Gaussian/lognormal and independent. A real program would use measured spec
#   distributions and correlations, but the machinery -- Saltelli sampling, Sobol
#   decomposition, spec-tightening -- is exactly this.
# ----------------------------------------------------------------------------
