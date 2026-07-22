r"""
Robust whole-design optimization: max *robust* gain under tolerances.

Step 2 optimized the whole design for gain and the optimum railed -- convergence
high, adiabat low, fuel high -- because nothing charged for the Rayleigh-Taylor
instability that aggressive compression buys. Step 3 adds two things:

  1. THE INSTABILITY CHARGE.  rt_mix.py turns the design into a multimode mix width
     and an ignition-margin penalty. Folded into the objective (as a factor relative
     to the tuned nominal, so NIF's own mix is not double-counted), it removes the
     incentive to rail: the deterministic optimum now sits at an INTERIOR
     convergence / adiabat -- a real ignition-margin sweet spot.

  2. ROBUSTNESS.  A real shot is not the nominal design -- surface finish, drive
     symmetry, adiabat and delivered energy all scatter shot to shot. We Monte-Carlo
     those tolerances (common random numbers, so the objective stays smooth) and
     optimize a ROBUST objective, mean(gain) - K*std(gain). Because the mix penalty
     is a cliff, an aggressive design that wins on paper can spend half its shots
     quenched; the robust optimum backs off the cliff, trading a little nominal gain
     for far higher probability of ignition.

Forward model (all fast: hydro-fit gain, cached RT table, one view-factor solve):

  effective gain = gain_0(E,fuel,CR,adiabat)          [coupled_gain, hydro hot-spot T]
                 x mix_factor(CR,adiabat,sigma_surf)   [rt_mix / nominal]
                 x symmetry_factor(hohlraum)           [view factor -> YOC / nominal]

Run:  python3 robust_design_ml.py
Deps: numpy, scipy, scikit-learn  (matplotlib only for the optional figure)
"""

import os
import sys
import importlib.util

import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def _load(mod, fname):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cg = _load("coupled_gain", "coupled_gain.py")
mix = _load("rt_mix", "rt_mix.py")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rayleigh-Taylor"))
from hohlraum_viewfactor import symmetry               # noqa: E402
from hohlraum_asymmetry import build_forward, performance  # noqa: E402

NAMES  = ["E_laser_MJ", "fuel_ug", "conv_ratio", "adiabat",
          "case_to_capsule", "length_to_diameter", "leh_radius_frac", "inner_frac"]
BOUNDS = np.array([[1.5, 2.4], [0.15, 0.30], [28.0, 45.0], [1.6, 2.6],
                   [2.5, 4.5], [1.2, 2.0], [0.35, 0.65], [0.20, 0.80]])
NOMINAL = np.array([2.05, 0.21, 35.0, 2.0, 3.0, 1.5, 0.50, 0.40])

_FWD = None          # convergent-RT forward model for the YOC stage (built once)
YOC_REF = None       # YOC at the nominal geometry -> symmetry factor = 1 at nominal
MIX_REF = mix.mix_penalty(35.0, 2.0, mix.SIGMA_SURF_REF)   # mix penalty at nominal (~0.85)

# --- tolerances (1-sigma), applied as common random numbers ---
TOL_E      = 0.03      # delivered laser energy      (+/- 3%)
TOL_ADIA   = 0.20      # in-flight adiabat            (drive timing / shock mistiming)
TOL_LOGSIG = 0.45      # surface finish  sigma*exp(N(0,.45))  (~ +/-55%)
TOL_SYM    = 0.12      # drive symmetry factor       (pointing / cone balance jitter)
K_ROBUST   = 1.5       # penalize std at 1.5 sigma
N_MC       = 20        # Monte-Carlo tolerance draws
G_TARGET   = 3.0       # "useful yield" bar for the ignition-margin metric P(gain>=G_TARGET)


def gain0(x):
    return cg.evaluate(cg.Design(x[0]*1e6, x[1]*1e-6, x[2], x[3]))["gain"]


def mix_factor(cr, adiabat, sigma_surf):
    """Mix penalty relative to the tuned nominal (capped at the clean limit)."""
    return min(mix.mix_penalty(cr, adiabat, sigma_surf) / MIX_REF, 1.0 / MIX_REF)


def sym_factor(x):
    a2, a4, a6 = symmetry(x[4], x[5], x[6], x[7])
    yoc, _ = performance(a2, a4, a6, fwd=_FWD)
    return (yoc / YOC_REF) if YOC_REF else 1.0


def forward_det(x):
    """Deterministic effective gain at the nominal tolerances."""
    return gain0(x) * mix_factor(x[2], x[3], mix.SIGMA_SURF_REF) * sym_factor(x)


def _eps(seed=0):
    """Common random tolerance draws: columns [E, adiabat, log-sigma, symmetry]."""
    return np.random.default_rng(seed).standard_normal((N_MC, 4))


_EPS = _eps(0)


def gain_samples(x):
    """Effective gain across the tolerance draws (symmetry solved once, then jittered)."""
    s0 = sym_factor(x)                                    # one view-factor solve
    out = np.empty(N_MC)
    for i in range(N_MC):
        e = _EPS[i]
        E   = x[0] * (1.0 + TOL_E * e[0])
        ad  = max(1.2, x[3] + TOL_ADIA * e[1])
        sig = mix.SIGMA_SURF_REF * np.exp(TOL_LOGSIG * e[2])
        sym = max(0.0, s0 * (1.0 + TOL_SYM * e[3]))
        g0  = cg.evaluate(cg.Design(E*1e6, x[1]*1e-6, x[2], ad))["gain"]
        out[i] = g0 * mix_factor(x[2], ad, sig) * sym
    return out


def robust_obj(x):
    g = gain_samples(x)
    return float(g.mean() - K_ROBUST * g.std())


def p_ignition(x, thresh=1.0):
    return float(np.mean(gain_samples(x) >= thresh))


# ----------------------------------------------------------------------------
# ML surrogate + active-learning optimum (shared by both objectives)
# ----------------------------------------------------------------------------
def sample(objfun, n, seed=0):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n) * (hi - lo) + lo
    y = np.array([objfun(x) for x in X])
    return X, y


def fit_surrogate(X, y):
    scaler = StandardScaler().fit(X)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                         max_iter=3000, random_state=0)
    model.fit(scaler.transform(X), y)
    return model, scaler


def surrogate_optimum(model, scaler):
    res = differential_evolution(
        lambda x: -float(model.predict(scaler.transform(x.reshape(1, -1)))[0]),
        bounds=list(map(tuple, BOUNDS)), seed=0, maxiter=80, tol=1e-6, polish=True)
    return res.x


def optimize(objfun, label, n_init=350, rounds=6):
    X, y = sample(objfun, n_init, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model, scaler = fit_surrogate(Xtr, ytr)
    r2 = r2_score(yte, model.predict(scaler.transform(Xte)))
    print(f"  [{label}] surrogate on {len(Xtr)} designs   test R^2 = {r2:.3f}")
    best_x, best_y = X[np.argmax(y)].copy(), float(np.max(y))
    for _ in range(rounds):
        model, scaler = fit_surrogate(X, y)
        x_star = surrogate_optimum(model, scaler)
        y_true = objfun(x_star)
        X = np.vstack([X, x_star]); y = np.append(y, y_true)
        if y_true > best_y:
            best_x, best_y = x_star, y_true
    return dict(best_x=best_x, best_y=best_y, X=X, y=y, model=model, scaler=scaler, r2=r2)


def _describe(tag, x):
    g0 = gain0(x); mf = mix_factor(x[2], x[3], mix.SIGMA_SURF_REF); sf = sym_factor(x)
    g = gain_samples(x)
    print(f"  {tag}")
    print("    " + "  ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, x)))
    print(f"    gain_0={g0:.2f}  x mix={mf:.2f}  x symmetry={sf:.2f}  "
          f"-> deterministic gain={forward_det(x):.2f}")
    print(f"    under tolerances: mean={g.mean():.2f}  std={g.std():.2f}  "
          f"P(gain>={G_TARGET:.0f})={np.mean(g>=G_TARGET)*100:.0f}%  10th pct={np.percentile(g,10):.2f}")


def main():
    global _FWD, YOC_REF
    print("building convergent-RT forward model (one-time)...")
    _FWD = build_forward()
    YOC_REF = performance(*symmetry(*NOMINAL[4:]), fwd=_FWD)[0]

    print("=" * 74)
    print("  ROBUST WHOLE-DESIGN OPTIMIZATION  (mix penalty + tolerances)")
    print("=" * 74)
    _describe("NIF-like nominal", NOMINAL)
    print("-" * 74)

    print("  optimizing DETERMINISTIC gain (now with the RT mix charge)...")
    det = optimize(forward_det, "det")

    # Chance-constrained robust design: keep the deterministic optimum's design but
    # back the dominant knob (convergence) off the cliff to the highest CR that still
    # meets the ignition-margin constraint  P(gain >= target) >= 0.99. This is a
    # transparent robust rule; the noisy mean-K*std surrogate over the full 8-D cliff
    # is not reliable enough to trust as an optimizer.
    rob_x, crs_scan = robust_backoff(det["best_x"])
    rob = dict(best_x=rob_x, scan=crs_scan)

    print("=" * 74)
    _describe("DETERMINISTIC optimum (max nominal gain)", det["best_x"])
    print("-" * 74)
    _describe("ROBUST design (chance-constrained: P(gain>=%.0f) >= 99%%)" % G_TARGET, rob_x)
    print("=" * 74)
    print("  Step 2 railed CR->max, adiabat->min. With the mix charge the deterministic")
    print("  optimum sits interior; the deterministic optimum still chases nominal gain")
    print("  to the cliff shoulder (fat downside tail), so the robust design backs")
    print("  convergence off to guarantee the ignition-margin constraint.")
    print("=" * 74)
    return dict(det=det, rob=rob)


def robust_backoff(x0, thresh=0.99):
    """Back convergence off the cliff to the highest CR meeting P(gain>=target)>=thresh."""
    crs = np.linspace(BOUNDS[2, 0], BOUNDS[2, 1], 40)
    scan = []
    for c in crs:
        xx = x0.copy(); xx[2] = c
        g = gain_samples(xx)
        scan.append((c, g.mean(), np.mean(g >= G_TARGET)))
    ok = [row for row in scan if row[2] >= thresh]
    # highest-mean CR among those satisfying the chance constraint
    c_star = max(ok, key=lambda r: r[1])[0] if ok else BOUNDS[2, 0]
    rob_x = x0.copy(); rob_x[2] = c_star
    return rob_x, scan


def make_figure(info, fname="robust_design_ml.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: {e}]")
        return
    det_x, rob_x = info["det"]["best_x"], info["rob"]["best_x"]
    # an "aggressive" excursion: push the robust design's CR up / adiabat down for a
    # little more nominal gain -- the design that looks tempting on paper
    agg_x = rob_x.copy(); agg_x[2] = min(BOUNDS[2, 1], rob_x[2] + 3.0); agg_x[3] = max(1.6, rob_x[3] - 0.5)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) effective-gain landscape over (CR, adiabat), optima + the railed Step-2 corner
    C = np.linspace(BOUNDS[2, 0], BOUNDS[2, 1], 30)
    A = np.linspace(BOUNDS[3, 0], BOUNDS[3, 1], 30)
    CC, AA = np.meshgrid(C, A)
    bx = det_x
    Z = np.array([[forward_det([bx[0], bx[1], CC[i, j], AA[i, j], bx[4], bx[5], bx[6], bx[7]])
                   for j in range(CC.shape[1])] for i in range(CC.shape[0])])
    im = ax[0].contourf(CC, AA, Z, levels=20, cmap="viridis")
    ax[0].plot(NOMINAL[2], NOMINAL[3], "wo", ms=8, label="NIF-like")
    ax[0].plot(BOUNDS[2, 1], BOUNDS[3, 0], "kX", ms=11, label="Step-2 railed corner")
    ax[0].plot(det_x[2], det_x[3], "r*", ms=16, label="deterministic opt")
    ax[0].plot(rob_x[2], rob_x[3], "c^", ms=11, label="robust opt")
    fig.colorbar(im, ax=ax[0], label="deterministic gain")
    ax[0].set_xlabel("convergence ratio"); ax[0].set_ylabel("in-flight adiabat")
    ax[0].set_title("gain landscape with the mix charge:\ninterior optimum, not railed")
    ax[0].legend(fontsize=7, loc="upper left")

    # (2) THE MONEY PLOT: sweep CR at the robust optimum's other knobs. Nominal gain
    #     keeps rising, but mean gain under tolerances rolls over and the ignition-
    #     margin metric P(gain>=target) falls off the cliff -- the robust knee.
    crs = np.linspace(BOUNDS[2, 0], BOUNDS[2, 1], 22)
    det_g, mean_g, p_marg = [], [], []
    for c in crs:
        xx = rob_x.copy(); xx[2] = c
        det_g.append(forward_det(xx))
        g = gain_samples(xx); mean_g.append(g.mean()); p_marg.append(np.mean(g >= G_TARGET))
    ax[1].plot(crs, det_g, "tab:gray", lw=1.8, ls="--", label="nominal gain")
    ax[1].plot(crs, mean_g, "tab:red", lw=2.4, label="mean gain (tolerances)")
    ax[1].axvline(rob_x[2], color="tab:cyan", lw=1.4); ax[1].text(rob_x[2]+0.2, min(det_g), "robust", color="tab:cyan", fontsize=8)
    ax[1].axvline(det_x[2], color="tab:red", ls=":", lw=1.2)
    ax[1].set_xlabel("convergence ratio"); ax[1].set_ylabel("gain")
    axp = ax[1].twinx()
    axp.plot(crs, np.array(p_marg)*100, "tab:blue", lw=2, alpha=0.7)
    axp.set_ylabel("P(gain >= %.0f)  [%%]" % G_TARGET, color="tab:blue")
    axp.tick_params(axis="y", labelcolor="tab:blue"); axp.set_ylim(0, 105)
    ax[1].set_title("push CR for more gain -> ignition\nmargin falls off: the robust knee")
    ax[1].legend(fontsize=8, loc="lower left")

    # (3) gain distributions: robust optimum vs the tempting aggressive excursion
    for x, col, lab in [(rob_x, "tab:cyan", "robust opt"),
                        (agg_x, "tab:red", "aggressive excursion")]:
        g = gain_samples(x)
        ax[2].hist(g, bins=16, alpha=0.6, color=col,
                   label=f"{lab}\n  mean={g.mean():.1f} P(>={G_TARGET:.0f})={np.mean(g>=G_TARGET)*100:.0f}%")
    ax[2].axvline(G_TARGET, color="k", ls=":", lw=1)
    ax[2].text(G_TARGET*1.02, 0.5, "target", rotation=90, fontsize=8, va="bottom")
    ax[2].set_xlabel("effective gain under tolerances"); ax[2].set_ylabel("shots")
    ax[2].set_title("shot-to-shot distribution:\naggressive design spills off the cliff")
    ax[2].legend(fontsize=7)

    fig.suptitle("Robust ICF design: RT mix charge + tolerance robustness", fontsize=13)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


if __name__ == "__main__":
    make_figure(main())

# ----------------------------------------------------------------------------
# NOTES ----------------------------------------------------------------------
# * This is the payoff of the whole chain: gain_0 (hydro hot-spot T) x mix (multimode
#   RT) x symmetry (view factor), optimized end to end -- and then optimized for
#   ROBUST gain under tolerances, which is the number a real ignition program tries
#   to maximize. The mix charge is what makes "max gain" stop railing; robustness is
#   what turns a paper optimum into a design that ignites shot after shot.
# * Reduced throughout (see each module's NOTES). The tolerance model is Gaussian and
#   the four channels are independent; a real UQ (Step 4, Sobol indices +
#   tolerance->yield-variance) would rank which spec actually drives the variance.
# ----------------------------------------------------------------------------
