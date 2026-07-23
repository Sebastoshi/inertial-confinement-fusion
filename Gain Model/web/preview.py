r"""
Static filmstrip preview of the interactive implosion explorer.

Renders, from the same simulate() the browser animation uses, a row of capsule
cross-sections (coast -> converge -> stagnation -> ignition -> disassembly) plus the
R(t)/T(t)/gain(t) traces, for both NIF presets. This is what the live canvas shows,
frozen for the README / a portfolio thumbnail.

Run:  python3 web/preview.py        (from the Gain Model directory)
"""
import os
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

_it_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "implosion_timeline.py")
_spec = importlib.util.spec_from_file_location("implosion_timeline", _it_path)
it = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(it)

R0 = 1.0e-3
_STOPS = np.array([[12,16,44],[36,70,170],[150,46,150],[240,120,34],[255,205,96],[255,255,255]]) / 255.0


def t_color(T):
    x = np.clip(np.log10(max(T, 0.02) / 0.02) / np.log10(80 / 0.02), 0, 1)
    seg = x * (len(_STOPS) - 1); i = min(int(seg), len(_STOPS) - 2); f = seg - i
    return tuple(_STOPS[i] + (_STOPS[i + 1] - _STOPS[i]) * f)


def key_frames(s):
    t = s["t"]
    picks = [0.40 * s["t_stag"], 0.85 * s["t_stag"], s["t_stag"],
             t[int(np.argmax(s["T"]))], s["t_stag"] + 0.6 * (t[-1] - s["t_stag"])]
    return [int(np.argmin(np.abs(t - p))) for p in picks]


def main():
    presets = list(it.PRESETS.items())
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2.2], hspace=0.32, wspace=0.15)

    for row, (name, d) in enumerate(presets):
        s = it.simulate(d)
        frames = key_frames(s)
        labels = ["coast", "converging", "stagnation", "ignition" if s["ignites"] else "stagnation", "disassembly"]
        for col, (fi, lab) in enumerate(zip(frames, labels)):
            ax = fig.add_subplot(gs[row, col]); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor("#0b0e14")
            ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#556", lw=0.8))
            rr = max(s["R"][fi] / R0, 0.02)
            T = s["T"][fi]
            ax.add_patch(Circle((0, 0), rr * 1.5, color=t_color(T * 0.5), alpha=0.35))  # glow
            ax.add_patch(Circle((0, 0), rr, color=t_color(T)))
            ax.set_title(f"{lab}\n{s['t'][fi]*1e9:.2f} ns · {T:.0f} keV", fontsize=8, color="#333")

        # traces
        axp = fig.add_subplot(gs[row, 5])
        axp.plot(s["t"] * 1e9, s["R"] / s["R"].max(), color="tab:blue", lw=1.8, label="radius")
        axp.plot(s["t"] * 1e9, s["T"] / max(s["T"].max(), 1), color="tab:red", lw=1.8, label="hot-spot T")
        axp.plot(s["t"] * 1e9, s["gain_t"] / max(s["gain"], 1e-6), color="tab:orange", lw=1.8, label="gain")
        axp.set_xlabel("time [ns]", fontsize=8); axp.set_yticks([])
        axp.set_title(f"{name.split(' (')[0]}  —  gain {s['gain']:.2f}  [{s['verdict']}]", fontsize=10)
        axp.legend(fontsize=7, loc="center left")
        axp.tick_params(labelsize=7)

    fig.suptitle("ICF implosion explorer — what the live animation shows (two real NIF shots)", fontsize=13)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.png")
    fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
