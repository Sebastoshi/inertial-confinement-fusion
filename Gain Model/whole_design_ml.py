r"""
ML inverse design of the WHOLE ICF design -- laser + capsule + hohlraum -> gain.

The repo's earlier ML phase searched hohlraum *geometry* alone for drive symmetry
(YOC). This one optimizes the full design against the metric fusion actually cares
about -- target gain -- by chaining the two forward models the repo now has:

    laser + capsule  --[ coupled_gain: 1-D-hydro hot-spot T + energy ledger ]--> gain_0
    hohlraum geometry --[ view factor -> convergent RT ]--------------------> YOC (<=1)

    effective gain  =  gain_0 (E_laser, fuel, convergence, adiabat)  x  YOC/YOC_nominal

gain_0 is calibrated to the *real* (already-asymmetric) NIF shot, so YOC enters as a
symmetry factor RELATIVE to the tuned nominal hohlraum -- better symmetry than NIF
lifts gain above gain_0, worse drops it -- rather than a second absolute penalty on
top of the anchor (which would double-count NIF's own asymmetry loss).

That composite f(design) -> gain is a fast, deterministic forward model. We learn a
surrogate for it over the 8-D design space and run the same active-learning loop as
the repo's other ML phase (surrogate optimum -> verify against physics -> refit),
now over the entire design rather than four geometry knobs.

Design variables (bounds):
    E_laser        [1.5, 2.4] MJ     laser energy on target
    fuel_mass      [0.15, 0.30] ug   DT fuel that can burn
    conv_ratio     [28, 42]          implosion convergence (sets ablative rho*R)
    adiabat        [1.6, 2.6]        in-flight adiabat (lower = more compressible)
    case_to_capsule    [2.5, 4.5]    hohlraum radius / capsule radius
    length_to_diameter [1.2, 2.0]    hohlraum aspect ratio
    leh_radius_frac    [0.35, 0.65]  laser-entrance-hole radius / hohlraum radius
    inner_frac         [0.20, 0.80]  inner-vs-outer cone power split

Run:  python3 whole_design_ml.py
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

# --- coupled gain model (hot-spot T computed by the 1-D hydro) ---
_cg_path = os.path.join(os.path.dirname(__file__), "coupled_gain.py")
_spec = importlib.util.spec_from_file_location("coupled_gain", _cg_path)
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

# --- hohlraum symmetry -> YOC (from the Rayleigh-Taylor folder) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rayleigh-Taylor"))
from hohlraum_viewfactor import symmetry            # noqa: E402
from hohlraum_asymmetry import build_forward, performance  # noqa: E402

NAMES  = ["E_laser_MJ", "fuel_ug", "conv_ratio", "adiabat",
          "case_to_capsule", "length_to_diameter", "leh_radius_frac", "inner_frac"]
BOUNDS = np.array([[1.5, 2.4], [0.15, 0.30], [28.0, 42.0], [1.6, 2.6],
                   [2.5, 4.5], [1.2, 2.0], [0.35, 0.65], [0.20, 0.80]])
NOMINAL = np.array([2.05, 0.21, 35.0, 2.0, 3.0, 1.5, 0.50, 0.40])   # NIF-like, drive-tuned

_FWD = None       # convergent-RT forward model for the YOC stage (built once)
YOC_REF = None    # YOC at the nominal geometry -- normalizes the symmetry factor to 1


def yoc_of(x):
    """Hohlraum-geometry part of the design vector -> yield-over-clean."""
    a2, a4, a6 = symmetry(x[4], x[5], x[6], x[7])
    yoc, _ = performance(a2, a4, a6, fwd=_FWD)
    return yoc


def forward(x):
    """Whole-design vector -> effective target gain (gain_0 x symmetry factor)."""
    d = cg.Design(E_laser=x[0] * 1e6, fuel_mass=x[1] * 1e-6,
                  conv_ratio=x[2], adiabat=x[3])
    gain0 = cg.evaluate(d)["gain"]
    sym = yoc_of(x) / YOC_REF if YOC_REF else 1.0     # relative to the tuned nominal
    return gain0 * sym


def sample(n, seed=0):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n) * (hi - lo) + lo
    y = np.array([forward(x) for x in X])
    return X, y


def fit_surrogate(X, y):
    scaler = StandardScaler().fit(X)
    model = MLPRegressor(hidden_layer_sizes=(96, 96), activation="relu",
                         max_iter=6000, random_state=0)
    model.fit(scaler.transform(X), y)
    return model, scaler


def perm_importance(model, scaler, X, y, seed=1):
    rng = np.random.default_rng(seed)
    base = r2_score(y, model.predict(scaler.transform(X)))
    imp = {}
    for j, nm in enumerate(NAMES):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[nm] = base - r2_score(y, model.predict(scaler.transform(Xp)))
    return imp


def surrogate_optimum(model, scaler):
    res = differential_evolution(
        lambda x: -float(model.predict(scaler.transform(x.reshape(1, -1)))[0]),
        bounds=list(map(tuple, BOUNDS)), seed=0, maxiter=80, tol=1e-6, polish=True)
    return res.x


def main():
    global _FWD, YOC_REF
    print("building convergent-RT forward model (one-time)...")
    _FWD = build_forward()
    YOC_REF = yoc_of(NOMINAL)                 # symmetry-factor normalization

    y_nom = forward(NOMINAL)
    print("=" * 72)
    print("  ML INVERSE DESIGN OF THE WHOLE ICF DESIGN  (maximize target gain)")
    print("     laser + capsule -> coupled_gain ;  hohlraum -> YOC")
    print("=" * 72)
    print(f"  nominal (NIF-like) gain : {y_nom:.3f}   "
          f"(gain_0={cg.evaluate(cg.Design(NOMINAL[0]*1e6, NOMINAL[1]*1e-6, NOMINAL[2], NOMINAL[3]))['gain']:.3f}, "
          f"symmetry factor=1.00 by construction)")
    print("-" * 72)

    # initial design of experiments over the full 8-D space
    X, y = sample(900, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model, scaler = fit_surrogate(Xtr, ytr)
    r2 = r2_score(yte, model.predict(scaler.transform(Xte)))
    print(f"  surrogate trained on {len(Xtr)} designs   test R^2 = {r2:.3f}")

    imp = perm_importance(model, scaler, Xte, yte)
    print("  permutation importance (which design knob controls gain):")
    for nm, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"    {nm:20s} {v:7.3f}  |{'#'*int(max(0,v)*40)}")
    print("-" * 72)

    # active learning: propose surrogate optimum, verify with physics, refit
    print("  active-learning inverse design:")
    best_x, best_y = NOMINAL.copy(), y_nom
    for rnd in range(1, 9):
        model, scaler = fit_surrogate(X, y)
        x_star = surrogate_optimum(model, scaler)
        y_true = forward(x_star)                       # verify against physics
        X = np.vstack([X, x_star]); y = np.append(y, y_true)
        if y_true > best_y:
            best_x, best_y = x_star, y_true
        y_pred = float(model.predict(scaler.transform(x_star.reshape(1, -1)))[0])
        print(f"    round {rnd}: gain(surrogate)={y_pred:.3f}  "
              f"verified={y_true:.3f}   best={best_y:.3f}")

    print("-" * 72)
    print("  BEST DESIGN FOUND")
    for nm, v in zip(NAMES, best_x):
        print(f"    {nm:20s} {v:.3f}")
    g0_best = cg.evaluate(cg.Design(best_x[0]*1e6, best_x[1]*1e-6, best_x[2], best_x[3]))["gain"]
    sym_best = yoc_of(best_x) / YOC_REF
    print(f"    -> gain_0 = {g0_best:.3f}   x  symmetry factor = {sym_best:.2f}")
    print(f"    -> effective target gain = {best_y:.3f}   "
          f"(nominal {y_nom:.3f}, +{(best_y/y_nom-1)*100:.0f}%)")
    print("=" * 72)
    print("  NOTE: convergence rails high, adiabat rails low, fuel rails high --")
    print("  all raise gain in this reduced model precisely because it has no")
    print("  instability penalty yet. That trade (high convergence / low adiabat")
    print("  = more Rayleigh-Taylor growth) is exactly what Step 3 adds.")
    print("=" * 72)
    return dict(X=X, y=y, model=model, scaler=scaler, best_x=best_x,
                best_y=best_y, y_nom=y_nom, imp=imp, r2=r2)


def make_figure(info, fname="whole_design_ml.png"):
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
    lim = [min(y.min(), yp.min()), max(y.max(), yp.max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("true gain"); ax[0].set_ylabel("surrogate gain")
    ax[0].set_title(f"surrogate accuracy  (R^2={info['r2']:.3f})")

    # (2) permutation importance
    names = list(info["imp"].keys()); vals = [info["imp"][k] for k in names]
    order = np.argsort(vals)
    ax[1].barh([names[i] for i in order], [vals[i] for i in order], color="tab:purple")
    ax[1].set_xlabel("R^2 drop when shuffled")
    ax[1].set_title("what controls gain")

    # (3) gain landscape: laser energy vs convergence at the best hohlraum/capsule
    bx = info["best_x"]
    E = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 40)
    C = np.linspace(BOUNDS[2, 0], BOUNDS[2, 1], 40)
    EE, CC = np.meshgrid(E, C)
    Z = np.array([[forward([EE[i, j], bx[1], CC[i, j], bx[3], bx[4], bx[5], bx[6], bx[7]])
                   for j in range(EE.shape[1])] for i in range(EE.shape[0])])
    im = ax[2].contourf(EE, CC, Z, levels=20, cmap="viridis")
    ax[2].plot(bx[0], bx[2], "r*", ms=16, label="best")
    ax[2].plot(NOMINAL[0], NOMINAL[2], "wo", ms=8, label="NIF-like")
    fig.colorbar(im, ax=ax[2], label="gain")
    ax[2].set_xlabel("laser energy [MJ]"); ax[2].set_ylabel("convergence ratio")
    ax[2].set_title("gain landscape (slice)"); ax[2].legend(fontsize=8)

    fig.suptitle("ML inverse design of the whole ICF design: laser + capsule + hohlraum -> gain",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")


if __name__ == "__main__":
    make_figure(main())

# ----------------------------------------------------------------------------
# NOTES -----------------------------------------------------------------------
# * This closes the repo's arc: every stage-specific model now feeds one
#   design -> gain forward model, and the ML optimizes the whole thing end to end
#   against gain -- with the hot-spot temperature COMPUTED by the 1-D hydro rather
#   than fit (see coupled_gain.py).
# * Several knobs rail to their favorable bounds because the reduced model rewards
#   compression without charging for the instability it costs. Adding that penalty
#   (multimode RT + mix width, robustness to tolerances) is Step 3 -- and it is what
#   turns "max gain" into "max *robust* gain," the number a real program optimizes.
# ----------------------------------------------------------------------------
