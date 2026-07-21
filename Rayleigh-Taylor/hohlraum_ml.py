r"""
ML search over hohlraum geometry for better ICF drive symmetry.

This ties the pieces together into an inverse-design loop:

    hohlraum geometry  --(view factor)-->  drive symmetry a_l
                       --(convergent RT)-->  hot-spot shape  -->  yield-over-clean

    hohlraum_viewfactor.symmetry(...)  ->  a2, a4, a6
    hohlraum_asymmetry.performance(...) ->  YOC

That chain is a fast, deterministic forward model  f(geometry) -> YOC.  Here we
learn a surrogate for it and use the surrogate to search the 4-D geometry space
for designs that maximize YOC -- the same surrogate + active-learning pattern as
the repo's ML Surrogate folder, now over ACTUAL hohlraum geometry.

Design variables (bounds):
    case_to_capsule     [2.5, 4.5]   hohlraum radius / capsule radius
    length_to_diameter  [1.2, 2.0]   hohlraum aspect ratio
    leh_radius_frac     [0.35, 0.65] LEH radius / hohlraum radius
    inner_frac          [0.20, 0.80] inner-vs-outer laser cone power split

Run:  python3 hohlraum_ml.py
Deps: numpy, scipy, scikit-learn  (matplotlib only for the optional figure)
"""

import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from hohlraum_viewfactor import symmetry
from hohlraum_asymmetry import build_forward, performance

NAMES  = ["case_to_capsule", "length_to_diameter", "leh_radius_frac", "inner_frac"]
BOUNDS = np.array([[2.5, 4.5], [1.2, 2.0], [0.35, 0.65], [0.20, 0.80]])
NOMINAL = np.array([3.0, 1.5, 0.50, 0.50])

_FWD = None       # convergent-RT forward model (built once; sets the YOC mapping)


def forward(x):
    """geometry vector -> YOC (the physics forward model)."""
    a2, a4, a6 = symmetry(*x)
    yoc, _ = performance(a2, a4, a6, fwd=_FWD)
    return yoc


def sample(n, seed=0):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = qmc.LatinHypercube(d=4, seed=seed).random(n) * (hi - lo) + lo
    y = np.array([forward(x) for x in X])
    return X, y


def perm_importance(model, scaler, X, y, seed=1):
    rng = np.random.default_rng(seed)
    base = r2_score(y, model.predict(scaler.transform(X)))
    imp = {}
    for j, nm in enumerate(NAMES):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[nm] = base - r2_score(y, model.predict(scaler.transform(Xp)))
    return imp


def fit_surrogate(X, y):
    scaler = StandardScaler().fit(X)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                         max_iter=4000, random_state=0)
    model.fit(scaler.transform(X), y)
    return model, scaler


def surrogate_optimum(model, scaler):
    res = differential_evolution(
        lambda x: -float(model.predict(scaler.transform(x.reshape(1, -1)))[0]),
        bounds=list(map(tuple, BOUNDS)), seed=0, maxiter=60, tol=1e-6, polish=True)
    return res.x


def main():
    global _FWD
    print("building convergent-RT forward model (one-time)...")
    _FWD = build_forward()

    y_nom = forward(NOMINAL)
    print("=" * 68)
    print("  ML INVERSE DESIGN OF HOHLRAUM GEOMETRY  (maximize YOC)")
    print("=" * 68)
    print(f"  nominal design {dict(zip(NAMES, np.round(NOMINAL,3)))}")
    print(f"  nominal YOC : {y_nom:.3f}")
    print("-" * 68)

    # initial design of experiments
    X, y = sample(700, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model, scaler = fit_surrogate(Xtr, ytr)
    r2 = r2_score(yte, model.predict(scaler.transform(Xte)))
    print(f"  surrogate trained on {len(Xtr)} designs   test R^2 = {r2:.3f}")

    imp = perm_importance(model, scaler, Xte, yte)
    print("  permutation importance (which geometry knob controls YOC):")
    for nm, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"    {nm:20s} {v:6.3f}  |{'#'*int(max(0,v)*40)}")
    print("-" * 68)

    # active learning: propose surrogate optimum, verify with physics, refit
    print("  active-learning inverse design:")
    best_x, best_y = NOMINAL.copy(), y_nom
    for rnd in range(1, 7):
        model, scaler = fit_surrogate(X, y)
        x_star = surrogate_optimum(model, scaler)
        y_true = forward(x_star)                       # verify against physics
        X = np.vstack([X, x_star]); y = np.append(y, y_true)
        if y_true > best_y:
            best_x, best_y = x_star, y_true
        y_pred = float(model.predict(scaler.transform(x_star.reshape(1, -1)))[0])
        print(f"    round {rnd}: proposed YOC(surrogate)={y_pred:.3f}  "
              f"verified={y_true:.3f}   best={best_y:.3f}")

    a2, a4, a6 = symmetry(*best_x)
    print("-" * 68)
    print("  BEST DESIGN FOUND")
    for nm, v in zip(NAMES, best_x):
        print(f"    {nm:20s} {v:.3f}")
    print(f"    -> a2={a2*100:+.2f}%  a4={a4*100:+.2f}%   YOC = {best_y:.3f}")
    print(f"    improvement over nominal : {best_y:.3f} vs {y_nom:.3f}  "
          f"(+{(best_y-y_nom)*100:.0f} YOC points)")
    print("=" * 68)
    return dict(X=X, y=y, model=model, scaler=scaler, best_x=best_x,
                best_y=best_y, y_nom=y_nom, imp=imp, r2=r2)


def make_figure(info, fname="hohlraum_ml.png"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figure skipped: matplotlib unavailable -- {e}]")
        return
    X, y = info["X"], info["y"]
    model, scaler = info["model"], info["scaler"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) surrogate accuracy
    yp = model.predict(scaler.transform(X))
    ax[0].scatter(y, yp, s=8, alpha=0.4, color="tab:red")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("true YOC"); ax[0].set_ylabel("surrogate YOC")
    ax[0].set_title(f"surrogate accuracy  (R^2={info['r2']:.3f})")

    # (2) permutation importance
    names = list(info["imp"].keys()); vals = [info["imp"][k] for k in names]
    order = np.argsort(vals)
    ax[1].barh([names[i] for i in order], [vals[i] for i in order], color="tab:purple")
    ax[1].set_xlabel("R^2 drop when shuffled")
    ax[1].set_title("what controls YOC")

    # (3) YOC landscape slice: inner_frac vs leh_radius_frac at best CC/LD
    bx = info["best_x"]
    f = np.linspace(0.2, 0.8, 40); L = np.linspace(0.35, 0.65, 40)
    FF, LL = np.meshgrid(f, L)
    Z = np.array([[forward([bx[0], bx[1], LL[i, j], FF[i, j]])
                   for j in range(FF.shape[1])] for i in range(FF.shape[0])])
    im = ax[2].contourf(FF, LL, Z, levels=20, cmap="viridis")
    ax[2].plot(bx[3], bx[2], "r*", ms=16, label="best")
    ax[2].plot(NOMINAL[3], NOMINAL[2], "wo", ms=8, label="nominal")
    fig.colorbar(im, ax=ax[2], label="YOC")
    ax[2].set_xlabel("inner_frac"); ax[2].set_ylabel("leh_radius_frac")
    ax[2].set_title("YOC landscape (slice)"); ax[2].legend(fontsize=8)

    fig.suptitle("ML inverse design: searching hohlraum geometry for drive symmetry",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


if __name__ == "__main__":
    make_figure(main())


# ----------------------------------------------------------------------------
# NOTES -----------------------------------------------------------------------
# * The whole point: f(geometry)->YOC is cheap and deterministic, so the ML does
#   not replace the physics -- it LEARNS the physics and then searches it. The
#   active-learning loop proposes the surrogate's optimum, verifies it against the
#   real forward model, and folds it back in, so the surrogate sharpens exactly
#   where the optimum lives.
# * Permutation importance answers "which knob matters": here the cone balance
#   (inner_frac) and LEH size dominate P2 -> YOC, matching the view-factor trends.
# * To make this a REAL design tool, upgrade the two physics stages (a proper
#   radiation-transport symmetry solve; a real yield model) -- the ML wrapper is
#   unchanged. That is the value of keeping the forward model swappable.
# ----------------------------------------------------------------------------
