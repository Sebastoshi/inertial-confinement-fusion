r"""
Validation tests -- the physics limits every model in this repo is anchored to.

These are not smoke tests; each one pins a claim the READMEs make, so the claims
cannot silently rot:

  * the gain models reproduce the NIF N221204 ignition shot,
  * the 1-D hydro conserves energy and computes a ~10 keV hot spot,
  * the convergent-RT solver recovers its planar (sqrt(A g k)) and Bell-Plesset limits,
  * the multimode mix penalty stays calibrated (nominal ignites, the aggressive
    corner quenches),
  * the hohlraum view factor nulls P2 at the known cone balance and the asymmetry
    model reproduces the yield cliff,
  * the 0-D burn shows an ignition cliff in temperature.

Run:  python3 -m pytest -q          (from the repo root)
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def hydro_run(hydro):
    """Run the 1-D hydro once and share its history + stagnation state."""
    hist, _snaps, _is_fill, _node_idx = hydro.run()
    return hist, hydro.peak_conditions(hist)


# ----------------------------------------------------------------------------
# End-to-end gain models vs the real ignition shot (NIF N221204)
# ----------------------------------------------------------------------------
class TestGainModelNIF:
    def test_yield_gain_temperature(self, gain_model):
        r = gain_model.evaluate(gain_model.NIF)
        assert r["yield_J"] / 1e6 == pytest.approx(3.15, rel=0.10)   # NIF 3.15 MJ
        assert r["gain"] == pytest.approx(1.54, rel=0.12)            # NIF gain 1.54
        assert r["T_hs"] == pytest.approx(10.9, rel=0.10)            # NIF ~10.9 keV
        assert 0.02 < r["burnup"] < 0.09                             # NIF ~4%

    def test_coupled_reproduces_nif(self, coupled):
        r = coupled.evaluate(coupled.NIF)
        assert r["gain"] == pytest.approx(1.54, rel=0.12)
        assert r["yield_J"] / 1e6 == pytest.approx(3.15, rel=0.12)

    def test_hydro_temperature_is_computed_not_fit(self, coupled):
        # the hot-spot T comes from the 1-D hydro fit; at NIF drive it must land ~10-11 keV
        r = coupled.evaluate(coupled.NIF)
        assert 10.0 <= r["T_hs"] <= 11.5

    def test_hydro_T_monotonic_in_drive(self, coupled):
        Ts = [coupled.hydro_T_hs(P) for P in (80, 120, 160, 220)]
        assert all(b > a for a, b in zip(Ts, Ts[1:]))               # strong-shock: T rises with drive


# ----------------------------------------------------------------------------
# 0-D hot-spot burn: ignition cliff in temperature
# ----------------------------------------------------------------------------
class TestBurn:
    def test_reference_burnup(self, hotspot):
        b = hotspot.burn(300.0, 1.20, 10.7, n_tau=0.42)
        assert 0.02 < b["burnup"] < 0.09

    def test_ignition_cliff_in_temperature(self, hotspot):
        cold = hotspot.burn(300.0, 1.20, 3.0, n_tau=0.42)["burnup"]
        hot  = hotspot.burn(300.0, 1.20, 10.7, n_tau=0.42)["burnup"]
        assert cold < 0.2 * hot                                     # below ignition -> little burn


# ----------------------------------------------------------------------------
# 1-D Lagrangian hydro: energy conservation + a hot hot-spot
# ----------------------------------------------------------------------------
class TestHydro:
    def test_energy_conservation(self, hydro_run):
        hist, _pk = hydro_run
        tot = hist["KE"] + hist["IE"] - hist["Wdrive"]
        drift = (tot[-1] - tot[0]) / max(abs(hist["Wdrive"][-1]), 1e-30)
        assert abs(drift) < 0.03                                    # < 3% of drive work

    def test_hot_spot_temperature(self, hydro_run):
        _hist, pk = hydro_run
        assert 6.0 < pk["Ths"] < 16.0                              # shock-heated hot spot ~10 keV


# ----------------------------------------------------------------------------
# Convergent / deceleration-phase RT: the two anchoring limits
# ----------------------------------------------------------------------------
class TestConvergentRT:
    def test_planar_limit_recovers_sqrt_Agk(self, crt):
        gamma_th, gamma_meas = crt.validate_planar()
        assert gamma_meas == pytest.approx(gamma_th, rel=0.08)

    def test_bell_plesset_convergence_limit(self, crt, crt_P0):
        prod0, prods, _cr = crt.validate_convergence(crt_P0)
        assert prods / prod0 == pytest.approx(1.0, abs=0.6)         # eta ~ 1/R  =>  eta*R ~ const


# ----------------------------------------------------------------------------
# Multimode RT mix penalty: stays calibrated
# ----------------------------------------------------------------------------
class TestMix:
    def test_nominal_keeps_about_85_percent(self, rtmix):
        assert rtmix.mix_penalty(35.0, 2.0) == pytest.approx(0.85, abs=0.05)

    def test_aggressive_corner_quenches(self, rtmix):
        assert rtmix.mix_penalty(44.0, 1.6) < 0.10                 # Step-2 corner is off the cliff

    def test_conservative_is_clean(self, rtmix):
        assert rtmix.mix_penalty(28.0, 2.4) > 0.90

    def test_penalty_falls_with_convergence(self, rtmix):
        p = [rtmix.mix_penalty(cr, 2.0) for cr in (32, 36, 40, 44)]
        assert all(b < a for a, b in zip(p, p[1:]))

    def test_higher_adiabat_is_more_stable(self, rtmix):
        # at fixed CR a higher adiabat (thicker shell) suffers less mix
        assert rtmix.mix_penalty(40.0, 2.4) > rtmix.mix_penalty(40.0, 1.8)


# ----------------------------------------------------------------------------
# Hohlraum view factor: P2 nulls at the known cone balance
# ----------------------------------------------------------------------------
class TestViewFactor:
    def test_p2_nulls_near_inner_frac_040(self, viewfactor):
        a2, _a4, _a6 = viewfactor.symmetry(3.0, 1.5, 0.5, 0.40)
        assert abs(a2) < 0.02                                       # symmetric drive at inner_frac~0.40

    def test_cone_balance_tunes_p2_through_zero(self, viewfactor):
        a2_lo = viewfactor.symmetry(3.0, 1.5, 0.5, 0.25)[0]
        a2_hi = viewfactor.symmetry(3.0, 1.5, 0.5, 0.55)[0]
        assert a2_lo > 0 > a2_hi                                    # P2 changes sign across the null


# ----------------------------------------------------------------------------
# Hohlraum asymmetry -> yield: the ignition cliff (YOC collapse)
# ----------------------------------------------------------------------------
class TestAsymmetry:
    def test_yield_cliff(self, asym, hohlraum_fwd):
        yoc_small, _ = asym.performance(0.01, fwd=hohlraum_fwd)
        yoc_large, _ = asym.performance(0.05, fwd=hohlraum_fwd)
        assert yoc_small > yoc_large                                # more asymmetry -> less yield
        assert yoc_large < 0.15                                     # 5% P2 is off the cliff


# ----------------------------------------------------------------------------
# Dashboard compute layer: normalization + verdict logic (headless, no Streamlit)
# ----------------------------------------------------------------------------
class TestDashboard:
    def test_nominal_reads_gain0(self, dashboard):
        r = dashboard.evaluate_design(2.05, 0.21, 35.0, 2.0, 3.0, 1.5, 0.50, 0.40)
        assert r["mix_factor"] == pytest.approx(1.0, abs=1e-9)      # factors normalized to nominal
        assert r["symmetry_factor"] == pytest.approx(1.0, abs=1e-6)
        assert r["effective_gain"] == pytest.approx(r["gain0"], rel=1e-6)
        assert r["verdict"] == "IGNITES"

    def test_aggressive_corner_quenches(self, dashboard):
        r = dashboard.evaluate_design(2.05, 0.30, 45.0, 1.6, 4.0, 1.5, 0.50, 0.40)
        assert r["verdict"] == "QUENCHED"                           # mix kills the Step-2 corner
        assert r["effective_gain"] < 0.5
