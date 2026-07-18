"""
Scaling the ignition surrogate from the toy rocket model to real ICF data (JAG).

ml_ignition.py trains a surrogate on this repo's rocket model. This script runs
the *same pipeline* on LLNL's open JAG dataset -- 10,000 semi-analytic ICF
implosion simulations -- so the methodology transfers from a toy you fully
understand to real ICF physics with essentially the same code.

JAG maps 5 physics inputs (stopping_mult, radiation_mult, ablation_cv, Vi,
conduction_mult) to 15 scalar diagnostics (yield, temperatures, pressures, ...)
plus four 64x64 X-ray images. Here we surrogate the 5 -> 15 scalar map, report
accuracy, and rank input sensitivity.

The honest difference from the rocket case: JAG is a *fixed dataset*, not a
callable simulator, so we CANNOT verify a surrogate-proposed design by
re-simulating it (the active-learning loop in ml_ignition.py needed that). That
gap is exactly what a live code like MULTI-IFE fills -- see the repo README.

--------------------------------------------------------------------------------
GETTING THE DATA (MIT-licensed, ~66 MB):

    git clone https://github.com/LLNL/macc
    mkdir -p jag_data && tar -xzf macc/data/icf-jag-10k.tar.gz -C jag_data

Then point --data at the folder holding the extracted .npy files:

    python3 jag_surrogate.py --data jag_data

The script auto-detects which .npy is inputs/scalars/images by array shape, so
it does not depend on the exact file names. With no data present it runs a small
SYNTHETIC stand-in so you can smoke-test the pipeline:

    python3 jag_surrogate.py --synthetic
--------------------------------------------------------------------------------

Deps: numpy, scikit-learn, matplotlib
"""

import argparse
import glob
import os
import numpy as np

JAG_INPUTS = ["stopping_mult", "radiation_mult", "ablation_cv", "Vi", "conduction_mult"]
N_IN, N_SCALAR, N_IMG = 5, 15, 16384


# ----------------------------------------------------------------------------
# Data loading: classify each .npy by its trailing dimension (robust to names)
# ----------------------------------------------------------------------------
def load_jag(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    X = Y = None
    for f in files:
        a = np.load(f, mmap_mode="r")
        if a.ndim == 2 and a.shape[1] == N_IN:
            X = np.asarray(a)
        elif a.ndim == 2 and a.shape[1] == N_SCALAR:
            Y = np.asarray(a)
        # images (N_IMG) are ignored here -- scalars are the surrogate target
    if X is None or Y is None:
        raise FileNotFoundError(
            f"Could not find inputs (*,{N_IN}) and scalars (*,{N_SCALAR}) .npy "
            f"in {data_dir!r}. See the header for download instructions.")
    return X, Y


def synthetic_jag(n=4000, seed=0):
    """A labeled stand-in with JAG's shape and smooth structure -- smoke test only."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 1.5, size=(n, N_IN))
    # 15 smooth nonlinear responses + noise (structure so R^2 is meaningful)
    W = rng.normal(size=(N_IN, N_SCALAR))
    base = np.tanh(X @ W) + 0.3 * (X[:, [3]] ** 2)      # Vi^2 term (velocity matters)
    Y = base + 0.05 * rng.normal(size=(n, N_SCALAR))
    return X, Y


# ----------------------------------------------------------------------------
# Surrogate (same recipe as ml_ignition.py)
# ----------------------------------------------------------------------------
def train(X, Y):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                       alpha=1e-3, max_iter=3000, random_state=0)
    mlp.fit(xs.transform(Xtr), ys.transform(Ytr))

    def predict(Xraw):
        return ys.inverse_transform(mlp.predict(xs.transform(np.atleast_2d(Xraw))))

    pred = predict(Xte)
    r2_each = np.array([r2_score(Yte[:, j], pred[:, j]) for j in range(Y.shape[1])])
    return predict, r2_each, (Xte, Yte, pred)


def perm_importance(predict, X, Y, col, seed=0):
    rng = np.random.default_rng(seed)
    base = np.mean((predict(X)[:, col] - Y[:, col]) ** 2)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[j] = np.mean((predict(Xp)[:, col] - Y[:, col]) ** 2) - base
    return np.maximum(imp, 0) / (imp.max() + 1e-30)


def make_figure(r2_each, test, predict, synthetic, fname="jag_surrogate.png"):
    import matplotlib.pyplot as plt
    Xte, Yte, pred = test
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) per-scalar R^2
    ax[0].bar(np.arange(len(r2_each)), r2_each, color="tab:blue")
    ax[0].axhline(np.median(r2_each), color="k", ls="--", lw=1,
                  label=f"median R^2 = {np.median(r2_each):.3f}")
    ax[0].set_xlabel("JAG scalar diagnostic index (0-14)")
    ax[0].set_ylabel("test R^2"); ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=9)
    ax[0].set_title("surrogate accuracy across all 15 diagnostics")

    # (2) parity for the best-predicted scalar
    j = int(np.argmax(r2_each))
    ax[1].scatter(Yte[:, j], pred[:, j], s=8, alpha=0.4, color="tab:red")
    lim = [min(Yte[:, j].min(), pred[:, j].min()), max(Yte[:, j].max(), pred[:, j].max())]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel(f"true scalar[{j}]"); ax[1].set_ylabel(f"surrogate scalar[{j}]")
    ax[1].set_title(f"parity, best scalar (index {j}): R^2 = {r2_each[j]:.3f}")

    # (3) input sensitivity for that scalar
    imp = perm_importance(predict, Xte, Yte, j)
    ax[2].bar(np.arange(N_IN), imp, color="tab:purple")
    ax[2].set_xticks(np.arange(N_IN))
    ax[2].set_xticklabels(JAG_INPUTS, rotation=25, ha="right", fontsize=8)
    ax[2].set_ylabel("relative importance")
    ax[2].set_title(f"what drives scalar[{j}] (permutation importance)")

    tag = "  [SYNTHETIC PLACEHOLDER DATA]" if synthetic else "  [real JAG data]"
    fig.suptitle("JAG ICF surrogate: 5 physics inputs -> 15 diagnostics" + tag,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"Saved figure -> {fname}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jag_data", help="folder with extracted JAG .npy files")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic stand-in")
    args = ap.parse_args()

    synthetic = args.synthetic
    if not synthetic:
        try:
            X, Y = load_jag(args.data)
            print(f"Loaded real JAG data: X{X.shape} -> Y{Y.shape}")
        except FileNotFoundError as e:
            print(f"[no real data] {e}\n-> falling back to synthetic smoke test.\n")
            synthetic = True
    if synthetic:
        X, Y = synthetic_jag()
        print(f"SYNTHETIC stand-in: X{X.shape} -> Y{Y.shape} (not real ICF data)")

    predict, r2_each, test = train(X, Y)
    print("=" * 60)
    print("  JAG SURROGATE  --  5 inputs -> 15 scalar diagnostics")
    print("=" * 60)
    print(f"  samples                 : {X.shape[0]}")
    print(f"  median test R^2         : {np.median(r2_each):.3f}")
    print(f"  best / worst scalar R^2 : {r2_each.max():.3f} / {r2_each.min():.3f}")
    print("=" * 60)
    if synthetic:
        print("NOTE: synthetic placeholder. Download real JAG data for real numbers")
        print("      (see the header). Then: python3 jag_surrogate.py --data jag_data")
    make_figure(r2_each, test, predict, synthetic)


if __name__ == "__main__":
    main()
