r"""
ML inverse design of the interactive explorer's knobs -- what does the ML predict?

The explorer lets you turn six knobs by hand; a brute-force grid says the max-gain
design is gain ~7.6. This runs the repo's ML inverse-design recipe -- a scikit-learn
surrogate trained on Latin-hypercube samples, then an active-learning loop that
proposes the surrogate's optimum, verifies it against the real model, and refits --
over those same six knobs, to see whether the ML rediscovers the optimum on its own.

Objective = the explorer's gain:
    gain(design) = coupled_gain(E, fuel, CR, adiabat)     [hydro hot-spot T + burn]
                 x mix_factor(CR, adiabat, surface)        [high-mode RT]
                 x asym_yoc(drive asymmetry)               [low-mode RT]

Run:  python3 ml_optimum.py
Deps: numpy, scipy, scikit-learn
"""
import os
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


it = _load("implosion_timeline", "implosion_timeline.py")
cg, mix = it.cg, it.mix

NAMES = ["E_laser [MJ]", "fuel [ug]", "convergence", "adiabat", "surface [nm]", "drive asym [%]"]
BOUNDS = np.array([[1.5, 2.4], [150.0, 260.0], [28.0, 46.0], [1.6, 2.6], [1.0, 60.0], [0.0, 6.0]])


def gain(x):
    g0 = cg.evaluate(cg.Design(x[0] * 1e6, x[1] * 1e-9, x[2], x[3]))["gain"]
    mf = min(mix.mix_penalty(x[2], x[3], x[4] * 1e-9) / it.MIX_REF, 1.0 / it.MIX_REF)
    return g0 * mf * it.asym_yoc(x[5])


def sample(n, seed=0):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    X = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n) * (hi - lo) + lo
    return X, np.array([gain(x) for x in X])


def fit(X, y):
    sc = StandardScaler().fit(X)
    m = MLPRegressor(hidden_layer_sizes=(96, 96), activation="relu", max_iter=6000, random_state=0)
    m.fit(sc.transform(X), y)
    return m, sc


def surrogate_opt(m, sc):
    res = differential_evolution(
        lambda x: -float(m.predict(sc.transform(x.reshape(1, -1)))[0]),
        bounds=list(map(tuple, BOUNDS)), seed=0, maxiter=80, tol=1e-7, polish=True)
    return res.x


def perm_importance(m, sc, X, y, seed=1):
    rng = np.random.default_rng(seed)
    base = r2_score(y, m.predict(sc.transform(X)))
    out = {}
    for j, nm in enumerate(NAMES):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        out[nm] = base - r2_score(y, m.predict(sc.transform(Xp)))
    return out


def main():
    print("=" * 66)
    print("  ML INVERSE DESIGN OF THE EXPLORER  --  maximize gain")
    print("=" * 66)
    X, y = sample(1500, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    m, sc = fit(Xtr, ytr)
    r2 = r2_score(yte, m.predict(sc.transform(Xte)))
    print(f"  surrogate trained on {len(Xtr)} designs   test R^2 = {r2:.3f}")

    imp = perm_importance(m, sc, Xte, yte)
    print("  which knob controls gain (permutation importance):")
    for nm, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"    {nm:16s} {v:6.3f}  |{'#'*int(max(0, v) * 40)}")
    print("-" * 66)

    print("  active learning (propose surrogate optimum -> verify -> refit):")
    best_x, best_y = X[np.argmax(y)].copy(), float(np.max(y))
    for rnd in range(1, 15):
        m, sc = fit(X, y)
        xs = surrogate_opt(m, sc)
        ys = gain(xs)
        X = np.vstack([X, xs]); y = np.append(y, ys)
        if ys > best_y:
            best_x, best_y = xs, ys
        print(f"    round {rnd:2d}: surrogate {float(m.predict(sc.transform(xs.reshape(1,-1)))[0]):5.2f}  "
              f"verified {ys:5.2f}   best {best_y:5.2f}")

    print("=" * 66)
    print(f"  ML-PREDICTED OPTIMUM   gain = {best_y:.2f}")
    for nm, v in zip(NAMES, best_x):
        print(f"    {nm:16s} {v:8.2f}")
    grid = np.array([1.50, 260.0, 46.0, 1.70, 1.0, 0.0])           # brute-force max
    print("-" * 66)
    print(f"  brute-force grid optimum: gain = {gain(grid):.2f}  "
          f"(E={grid[0]}, adiabat={grid[3]})")
    print("=" * 66)
    return dict(best_x=best_x, best_y=best_y, imp=imp, r2=r2)


if __name__ == "__main__":
    main()
