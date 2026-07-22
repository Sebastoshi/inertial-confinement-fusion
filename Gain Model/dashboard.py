r"""
Interactive ICF design dashboard -- turn the knobs, watch gain and ignition margin.

A Streamlit front end over the whole chain this repo builds up: move the laser,
capsule, and hohlraum sliders and watch, live,

    coupled_gain (hydro hot-spot T)  x  RT mix penalty  x  hohlraum symmetry
                                 =  effective target gain,

with a verdict (IGNITES / MARGINAL / QUENCHED) and the ignition-margin cliff plotted
against convergence ratio so you can see where the current design sits relative to it.

Run:  pip install streamlit
      streamlit run "Gain Model/dashboard.py"

The compute layer (evaluate_design) has no Streamlit dependency, so it can be imported
and checked headlessly (see tests / the __main__ self-check at the bottom).
"""

import os
import sys
import importlib.util

import numpy as np


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

# normalizations so the NIF-like nominal reads gain_0 (mix & symmetry factors = 1)
MIX_REF = mix.mix_penalty(35.0, 2.0, mix.SIGMA_SURF_REF)
_FWD = None
_YOC_REF = None


def _ensure_fwd():
    global _FWD, _YOC_REF
    if _FWD is None:
        _FWD = build_forward()
        _YOC_REF = performance(*symmetry(3.0, 1.5, 0.50, 0.40), fwd=_FWD)[0]
    return _FWD


def evaluate_design(E_MJ, fuel_ug, cr, adiabat, case_to_capsule, length_to_diameter,
                    leh_radius_frac, inner_frac):
    """Whole design -> gain_0, mix factor, symmetry factor, effective gain, verdict."""
    _ensure_fwd()
    g = cg.evaluate(cg.Design(E_MJ * 1e6, fuel_ug * 1e-6, cr, adiabat))
    mix_pen = mix.mix_penalty(cr, adiabat, mix.SIGMA_SURF_REF)
    mix_fac = min(mix_pen / MIX_REF, 1.0 / MIX_REF)
    yoc, _ = performance(*symmetry(case_to_capsule, length_to_diameter, leh_radius_frac, inner_frac), fwd=_FWD)
    sym_fac = yoc / _YOC_REF
    eff = g["gain"] * mix_fac * sym_fac
    verdict = "QUENCHED" if mix_pen < 0.25 else ("MARGINAL" if eff < 1.0 or mix_pen < 0.6 else "IGNITES")
    return dict(gain0=g["gain"], T_hs=g["T_hs"], mix_penalty=mix_pen, mix_factor=mix_fac,
                symmetry_factor=sym_fac, effective_gain=eff, verdict=verdict)


def main():
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="ICF design dashboard", page_icon="🔆", layout="wide")
    st.title("🔆 Inertial-confinement-fusion design dashboard")
    st.caption("coupled_gain (hydro hot-spot T)  ×  multimode RT mix  ×  hohlraum symmetry  →  target gain")

    with st.sidebar:
        st.header("Laser + capsule")
        E   = st.slider("laser energy [MJ]", 1.5, 2.4, 2.05, 0.01)
        fu  = st.slider("DT fuel mass [µg]", 0.15, 0.30, 0.21, 0.01)
        cr  = st.slider("convergence ratio", 28.0, 45.0, 35.0, 0.5)
        ad  = st.slider("in-flight adiabat", 1.6, 2.6, 2.0, 0.05)
        st.header("Hohlraum")
        cc  = st.slider("case-to-capsule", 2.5, 4.5, 3.0, 0.1)
        ld  = st.slider("length / diameter", 1.2, 2.0, 1.5, 0.05)
        leh = st.slider("LEH radius fraction", 0.35, 0.65, 0.50, 0.01)
        inf = st.slider("inner cone fraction", 0.20, 0.80, 0.40, 0.01)

    with st.spinner("solving hohlraum view factor (first run)…"):
        r = evaluate_design(E, fu, cr, ad, cc, ld, leh, inf)

    color = {"IGNITES": "green", "MARGINAL": "orange", "QUENCHED": "red"}[r["verdict"]]
    st.markdown(f"### Verdict: :{color}[{r['verdict']}]")
    c = st.columns(5)
    c[0].metric("effective gain", f"{r['effective_gain']:.2f}")
    c[1].metric("gain₀ (hydro T)", f"{r['gain0']:.2f}")
    c[2].metric("hot-spot T [keV]", f"{r['T_hs']:.1f}")
    c[3].metric("mix penalty", f"{r['mix_penalty']*100:.0f}%")
    c[4].metric("symmetry factor", f"{r['symmetry_factor']:.2f}")

    left, right = st.columns(2)
    with left:
        crs = np.linspace(28, 45, 40)
        eff = [evaluate_design(E, fu, c_, ad, cc, ld, leh, inf)["effective_gain"] for c_ in crs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(crs, eff, "tab:red", lw=2.4)
        ax.axvline(cr, color="tab:cyan", lw=1.5); ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.plot(cr, r["effective_gain"], "k*", ms=15)
        ax.set_xlabel("convergence ratio"); ax.set_ylabel("effective gain")
        ax.set_title("gain vs convergence — the ignition-margin sweet spot")
        st.pyplot(fig)
    with right:
        C = np.linspace(28, 45, 45); A = np.linspace(1.6, 2.6, 45)
        CC, AA = np.meshgrid(C, A)
        Z = np.vectorize(lambda c_, a_: mix.mix_penalty(c_, a_))(CC, AA)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        im = ax2.contourf(CC, AA, Z, levels=20, cmap="viridis")
        ax2.plot(cr, ad, "r*", ms=15)
        fig2.colorbar(im, ax=ax2, label="mix yield multiplier")
        ax2.set_xlabel("convergence ratio"); ax2.set_ylabel("in-flight adiabat")
        ax2.set_title("RT mix landscape (your design = ★)")
        st.pyplot(fig2)

    st.caption("Reduced, NIF-anchored models — see each module's README/NOTES. "
               "Mix and symmetry enter as factors relative to the NIF-like nominal, so nominal reads gain₀.")


if __name__ == "__main__":
    main()
