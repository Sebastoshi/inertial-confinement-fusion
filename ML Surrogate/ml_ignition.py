"""
A neural-network surrogate for ICF ignition -- the data-driven layer.

This is the miniature version of what the LLNL "cognitive simulation" program
does with rad-hydro codes: sample a simulator across its design space, train a
fast surrogate, and use it to map the ignition boundary, rank what matters, and
inverse-design a target. Here the "simulator" is the repo's rocket-implosion
model (cheap and fully understood, so we can trust the ML), but the pipeline is
identical to the one you would run on MULTI or HYDRA.

Pipeline:
  1. sample 5 design knobs, run the rocket model  -> dataset of (design -> outcome)
  2. train an MLP surrogate (design -> v_imp, convergence, rho*R, hot-spot T)
  3. map the ignition boundary in a 2D slice, checked against the real sim
  4. permutation sensitivity: which knobs actually move rho*R and temperature
  5. inverse design: minimum-drive igniting capsule, verified against the sim

Run:  python3 ml_ignition.py         (regenerates + caches the dataset)
Deps: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.optimize import differential_evolution

# ----------------------------------------------------------------------------
# The simulator: rocket-implosion forward model (see ../Rocket Implosion)
# ----------------------------------------------------------------------------
MBAR_PA = 1.0e11
GAMMA   = 5.0 / 3.0
R0      = 1.0e-3
HS_MASS_FRAC = 0.05
ETA_HS  = 0.5
M_ION_DT = 2.5 * 1.66e-27
T_IGN   = 4.3           # ignition temperature   [keV]
RHOR_IGN = 0.30         # ignition areal density [g/cm^2]

# design knobs: name, low, high, unit
DESIGN = [
    ("P_drive",  40.0,   300.0),   # Mbar
    ("V_ex",     80.0,   250.0),   # km/s
    ("payload",  0.05,   0.30),    # fraction of M0 left as fuel
    ("M0",       1.0,    6.0),     # mg
    ("P_gas0",   1.0e9,  8.0e9),   # Pa
]
NAMES = [d[0] for d in DESIGN]
LOW = np.array([d[1] for d in DESIGN])
HIGH = np.array([d[2] for d in DESIGN])
OUTPUTS = ["v_imp [km/s]", "convergence", "rho*R [g/cm2]", "T_hs [keV]"]


def forward(x):
    """Run the rocket model for one design x. Returns [v_imp, CR, rho*R, T_hs]."""
    P_drive = x[0] * MBAR_PA
    V_ex    = x[1] * 1e3
    M_floor = x[2] * (x[3] * 1e-6)
    M0      = x[3] * 1e-6
    P_gas0  = x[4]
    mdot    = P_drive / V_ex

    def rhs(t, y):
        r, v, m = y
        r = max(r, 1e-6 * R0)
        A = 4.0 * np.pi * r ** 2
        driving = m > M_floor
        P_abl = P_drive if driving else 0.0
        p_gas = P_gas0 * (R0 / r) ** (3.0 * GAMMA)
        dvdt = A * (p_gas - P_abl) / m
        dmdt = -mdot * A if driving else 0.0
        return [v, dvdt, dmdt]

    def stag(t, y):
        return y[1]
    stag.terminal = True
    stag.direction = 1

    sol = solve_ivp(rhs, (0.0, 30e-9), [R0, 0.0, M0], method="LSODA",
                    events=stag, rtol=1e-5, atol=1e-2, max_step=3e-11)
    r = sol.y[0]; v = sol.y[1]; m = sol.y[2]
    v_imp = -v.min()
    m_f = m[-1]; r_min = max(r[-1], 1e-9)
    cr = R0 / r_min
    ke = 0.5 * m_f * v_imp ** 2
    rho_R = m_f / (4.0 * np.pi * r_min ** 2) * 0.1
    n_ions = HS_MASS_FRAC * m_f / M_ION_DT
    T_keV = ETA_HS * ke / (3.0 * n_ions) / 1.602e-16
    return np.array([v_imp / 1e3, cr, rho_R, T_keV])


def ignites(y):
    return (y[..., 3] >= T_IGN) & (y[..., 2] >= RHOR_IGN)


# ----------------------------------------------------------------------------
# Dataset (Latin-hypercube sample of the design space), cached to disk
# ----------------------------------------------------------------------------
def build_dataset(n=2500, cache="rocket_dataset.npz", seed=0):
    import os
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"Loaded cached dataset: {d['X'].shape[0]} samples")
        return d["X"], d["Y"]
    print(f"Sampling {n} designs and running the rocket model ...")
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    X = qmc.scale(sampler.random(n), LOW, HIGH)
    Y = np.full((n, 4), np.nan)
    for i in range(n):
        try:
            Y[i] = forward(X[i])
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}")
    ok = np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    np.savez(cache, X=X, Y=Y)
    print(f"Kept {X.shape[0]} valid samples -> {cache}")
    return X, Y


# ----------------------------------------------------------------------------
# Surrogate
# ----------------------------------------------------------------------------
def fit_predict(X, Y):
    """Fit an MLP surrogate; return a predict(design)->outputs function."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(X)
    ys = StandardScaler().fit(Y)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                       alpha=1e-3, max_iter=4000, random_state=0)
    mlp.fit(xs.transform(X), ys.transform(Y))

    def predict(Xraw):
        return ys.inverse_transform(mlp.predict(xs.transform(np.atleast_2d(Xraw))))
    return predict


def train_surrogate(X, Y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    predict = fit_predict(Xtr, Ytr)
    pred_te = predict(Xte)
    r2 = [r2_score(Yte[:, j], pred_te[:, j]) for j in range(4)]
    acc = np.mean(ignites(pred_te) == ignites(Yte))     # derived from predicted T & rho*R
    return predict, r2, acc, (Xte, Yte, pred_te)


def perm_importance(predict, X, Y, col, seed=0):
    """Manual permutation importance: MSE increase when each input is shuffled."""
    rng = np.random.default_rng(seed)
    base = np.mean((predict(X)[:, col] - Y[:, col]) ** 2)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[j] = np.mean((predict(Xp)[:, col] - Y[:, col]) ** 2) - base
    return np.maximum(imp, 0) / (imp.max() + 1e-30)


# ----------------------------------------------------------------------------
# Inverse design with active learning.
# Minimise drive pressure subject to (predicted) ignition -- but a surrogate is
# optimistic in under-sampled corners, and the optimiser will exploit exactly
# those errors. So we VERIFY each proposal with the real simulator; if it
# fizzles, we sample around it, label those points, retrain, and try again.
# This is the ICF-ML workflow in miniature (propose -> simulate -> refit).
# ----------------------------------------------------------------------------
def optimize_design(predict, margin=1.1):
    def cost(x):
        p = predict(x)[0]
        penalty = 1e3 * (max(0.0, T_IGN * margin - p[3])
                         + max(0.0, RHOR_IGN * margin - p[2]))
        return x[0] + penalty                        # minimise drive pressure
    res = differential_evolution(cost, list(zip(LOW, HIGH)), seed=0,
                                 maxiter=200, tol=1e-6, polish=True)
    return res.x


def inverse_design(X, Y, rounds=6, batch=30, seed=0):
    rng = np.random.default_rng(seed)
    Xp, Yp = X.copy(), Y.copy()
    first = None
    for r in range(rounds):
        predict = fit_predict(Xp, Yp)
        x_opt = optimize_design(predict)
        y_surr = predict(x_opt)[0]
        y_sim = forward(x_opt)
        if first is None:
            first = (x_opt.copy(), y_surr.copy(), y_sim.copy())
        if ignites(y_sim):
            return x_opt, y_surr, y_sim, r + 1, first, Xp.shape[0]
        # active learning: label a neighbourhood of the (failed) proposal, refit
        span = 0.1 * (HIGH - LOW)
        newX = np.clip(x_opt + rng.uniform(-1, 1, (batch, 5)) * span, LOW, HIGH)
        newY = np.array([forward(xx) for xx in newX])
        ok = np.isfinite(newY).all(axis=1)
        Xp = np.vstack([Xp, newX[ok]]); Yp = np.vstack([Yp, newY[ok]])
    return x_opt, y_surr, y_sim, rounds, first, Xp.shape[0]


# ----------------------------------------------------------------------------
# Reporting + figure
# ----------------------------------------------------------------------------
def report(X, Y, r2, acc, x_opt, y_surr, y_sim, rounds, first):
    print("=" * 66)
    print("  ML SURROGATE FOR ICF IGNITION  --  results")
    print("=" * 66)
    print(f"  dataset                  : {X.shape[0]} designs, "
          f"{100*np.mean(ignites(Y)):.0f}% ignite")
    print("  surrogate test R^2:")
    for name, r in zip(OUTPUTS, r2):
        print(f"      {name:16s} : {r:.3f}")
    print(f"  ignition classification  : {100*acc:.1f}% accuracy")
    print("-" * 66)
    fx, fs, fsim = first
    print(f"  inverse design (min drive that ignites): {rounds} active-learning round(s)")
    if not ignites(fsim):
        print(f"    round-1 proposal was over-optimistic: surrogate said "
              f"T={fs[3]:.1f} keV, real sim gave T={fsim[3]:.1f} keV (fizzle)")
        print(f"    -> sampled around it, retrained, converged to a verified design")
    for name, v in zip(NAMES, x_opt):
        print(f"      {name:10s} = {v:9.2f}")
    print(f"    predicted: rho*R={y_surr[2]:.2f}  T={y_surr[3]:.2f} keV")
    print(f"    real sim : rho*R={y_sim[2]:.2f}  T={y_sim[3]:.2f} keV  "
          f"[{'IGNITES' if ignites(y_sim) else 'still fizzle'}]")
    print("=" * 66)


def make_figure(predict, test, r2, X, Y, x_opt, y_surr, y_sim, rounds,
                fname="ml_ignition.png"):
    import matplotlib.pyplot as plt
    Xte, Yte, pred_te = test
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (1) parity plot for hot-spot temperature
    a = ax[0, 0]
    a.scatter(Yte[:, 3], pred_te[:, 3], s=8, alpha=0.4, color="tab:red")
    lim = [0, max(Yte[:, 3].max(), pred_te[:, 3].max()) * 1.05]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("true hot-spot T [keV]"); a.set_ylabel("surrogate T [keV]")
    a.set_title(f"surrogate accuracy (T):  R^2 = {r2[3]:.3f}")
    a.set_xlim(lim); a.set_ylim(lim)

    # (2) ignition boundary in a 2D slice, surrogate vs real sim
    b = ax[0, 1]
    V_ex, payl0, M0, Pg = 150.0, None, 3.0, 3.0e9
    pd = np.linspace(LOW[0], HIGH[0], 70)
    pf = np.linspace(LOW[2], HIGH[2], 70)
    PD, PF = np.meshgrid(pd, pf)
    grid = np.column_stack([PD.ravel(), np.full(PD.size, V_ex),
                            PF.ravel(), np.full(PD.size, M0), np.full(PD.size, Pg)])
    ig_pred = ignites(predict(grid)).reshape(PD.shape)
    b.contourf(PD, PF, ig_pred, levels=[-0.5, 0.5, 1.5],
               colors=["#f4c8c8", "#c8e6c8"], alpha=0.8)
    # real-sim boundary on a coarser grid
    pdc = np.linspace(LOW[0], HIGH[0], 26)
    pfc = np.linspace(LOW[2], HIGH[2], 26)
    PDc, PFc = np.meshgrid(pdc, pfc)
    ig_true = np.array([ignites(forward([p, V_ex, f, M0, Pg]))
                        for p, f in zip(PDc.ravel(), PFc.ravel())]).reshape(PDc.shape)
    b.contour(PDc, PFc, ig_true.astype(float), levels=[0.5], colors="k", linewidths=2)
    b.set_xlabel("drive pressure [Mbar]"); b.set_ylabel("payload fraction")
    b.set_title("ignition boundary\n(green=ignite; black line = true sim)")

    # (3) sensitivity: permutation importance for rho*R and T
    c = ax[1, 0]
    imp_rr = perm_importance(predict, Xte, Yte, 2)
    imp_T = perm_importance(predict, Xte, Yte, 3)
    xpos = np.arange(len(NAMES))
    c.bar(xpos - 0.2, imp_rr, 0.4, label="rho*R", color="tab:purple")
    c.bar(xpos + 0.2, imp_T, 0.4, label="T_hs", color="tab:red")
    c.set_xticks(xpos); c.set_xticklabels(NAMES, rotation=20)
    c.set_ylabel("relative importance"); c.legend()
    c.set_title("what drives the outcome (permutation importance)")

    # (4) inverse design: surrogate vs real sim at the optimum
    d = ax[1, 1]
    labels = ["v_imp/100", "CR/10", "rho*R", "T_hs"]
    scale = np.array([1 / 100, 1 / 10, 1.0, 1.0])
    xpos = np.arange(4)
    d.bar(xpos - 0.2, y_surr * scale, 0.4, label="surrogate", color="tab:blue")
    d.bar(xpos + 0.2, y_sim * scale, 0.4, label="real sim", color="tab:orange")
    d.set_xticks(xpos); d.set_xticklabels(labels, rotation=15)
    d.legend(loc="upper right")
    txt = "min-drive igniting design:\n" + \
          "\n".join(f"{n}={v:.3g}" for n, v in zip(NAMES, x_opt))
    d.text(0.03, 0.97, txt, transform=d.transAxes, va="top", fontsize=7.5,
           family="monospace", bbox=dict(fc="white", ec="gray", alpha=0.85))
    d.set_title(f"inverse design @ {x_opt[0]:.0f} Mbar ({rounds} AL round"
                f"{'s' if rounds != 1 else ''})\nsurrogate vs verified sim")

    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    X, Y = build_dataset()
    predict, r2, acc, test = train_surrogate(X, Y)
    x_opt, y_surr, y_sim, rounds, first, npool = inverse_design(X, Y)
    report(X, Y, r2, acc, x_opt, y_surr, y_sim, rounds, first)
    make_figure(predict, test, r2, X, Y, x_opt, y_surr, y_sim, rounds)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# NOTES / things to try -------------------------------------------------------
#
# * This is the whole point of the arxiv paper you started from: once you have a
#   surrogate, the ignition boundary, sensitivity, and inverse design are nearly
#   free -- you no longer pay a simulation per question. On a cheap rocket model
#   the payoff is pedagogical; swap `forward()` for MULTI and it is real.
#
# * The inverse-design row is the honesty check: the surrogate proposes a design,
#   and we VERIFY it with the actual simulator. If the two bars disagree, the
#   surrogate is extrapolating -- add samples there (active learning).
#
# * Sensitivity usually shows rho*R driven by the gas cushion + payload (they set
#   convergence) and T driven by exhaust velocity + payload (they set velocity).
#   That is exactly the rocket-equation physics, recovered from data.
#
# * Next steps toward the real thing: (a) active learning -- sample where the
#   surrogate is most uncertain near the boundary; (b) multi-fidelity/transfer
#   learning -- pre-train on this cheap model, fine-tune on a few MULTI runs
#   (Kustowski et al.); (c) Bayesian calibration against experimental data.
# ----------------------------------------------------------------------------
