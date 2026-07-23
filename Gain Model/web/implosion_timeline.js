// Time-resolved ICF implosion timeline — JavaScript port of implosion_timeline.py.
//
// Same contract as the Python reference: simulate(design) -> animation-ready time
// series (t, R, T, rhoR, gain_t) + scalar outcomes + IGNITES/MARGINAL/QUENCHED
// verdict. Runs entirely in the browser (no backend) to drive the interactive
// dashboard. Physics: Bosch-Hale burn ODE, the coupled-gain pipeline, the multimode
// RT mix penalty, and a tuned gas-cushion trajectory — see the .py for derivations.
//
// Verified against the Python reference by web/verify.mjs.

import { DATA } from "./timeline_data.js";

// ---- physical constants (SI) ----
const KB = 1.380649e-23;
const KEV_J = 1.602176634e-16;
const E_ALPHA = 3.5e6 * 1.602176634e-19;
const MBAR = ((2.014 + 3.016) / 2) * 1.66053907e-27; // mean DT ion mass
const GAMMA = 5.0 / 3.0;
const C_BREM = 5.34e-37;
const RHO_R_HALF = 0.30;

// ---- gain_model constants ----
const E_FUSION = 17.6e6 * 1.602176634e-19;
const M_DT_PAIR = (2.014 + 3.016) * 1.66053907e-27;
const ETA_HOHLRAUM = 0.14, ETA_ROCKET = 0.075;
const RHOR_REF = 1.20, CR_REF = 35.0, ALPHA_REF = 2.0, RHO_HS = 300.0, N_TAU_BURN = 0.42;

// ---- coupled_gain (drive pressure + hydro-fit hot-spot T) ----
const drivePressure = (E_J) => DATA.P_DRIVE_NIF * Math.pow(E_J / DATA.E_LASER_NIF, DATA.DRIVE_EXP);
const hydroThs = (P) => Math.max(DATA.HYDRO_SLOPE * P + DATA.HYDRO_INT, 0.05);

// ---- rt_mix constants ----
const SIGMA_SURF_REF = 2.1e-8, SPEC_SLOPE = 1.0, ELL_CUT = 35.0;
const ACCEL_ADIA_EXP = 2.0, K_IFAR = 9.7, P_MIX = 2.5, V_ABL_REF = 5.0e3;
// rt_mechanics (Takabe ablative growth)
const G_ACCEL = 1.9e14, ATWOOD = 0.9, ALPHA_T = 0.9, BETA_T = 3.0, T_ACCEL = 1.85e-9;

const V_IMP_NIF = 3.7e5;
const R0 = 1.0e-3;

// ---- small numeric helpers ----
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * (i / (n - 1)));
function interp(x, xp, fp) {
  // numpy.interp: linear, clamped to endpoints. xp ascending.
  if (x <= xp[0]) return fp[0];
  const n = xp.length;
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + t * (fp[hi] - fp[lo]);
}
const argmin = (a) => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[k]) k = i; return k; };
const amax = (a) => a.reduce((m, v) => (v > m ? v : m), -Infinity);

// ---- DT fusion reactivity <sigma v> (Bosch & Hale 1992), input keV -> m^3/s ----
function reactivityDT(T_keV) {
  const T = Math.min(Math.max(T_keV, 0.2), 100.0);
  const BG = 34.3827, MRC2 = 1.124656e6;
  const C1 = 1.17302e-9, C2 = 1.51361e-2, C3 = 7.51886e-2;
  const C4 = 4.60643e-3, C5 = 1.35e-2, C6 = -1.0675e-4, C7 = 1.366e-5;
  const theta = T / (1.0 - (T * (C2 + T * (C4 + T * C6))) / (1.0 + T * (C3 + T * (C5 + T * C7))));
  const xi = Math.pow((BG * BG) / (4.0 * theta), 1.0 / 3.0);
  const sv_cm3 = C1 * theta * Math.sqrt(xi / (MRC2 * T * T * T)) * Math.exp(-3.0 * xi);
  return sv_cm3 * 1e-6;
}
const fAlpha = (rhoR) => rhoR / (rhoR + RHO_R_HALF);
const pAlpha = (n_i, T, rhoR) => fAlpha(rhoR) * 0.25 * n_i * n_i * reactivityDT(T) * E_ALPHA;
const pBrem = (n_i, T) => C_BREM * n_i * n_i * Math.sqrt(T);
const soundSpeed = (T_keV) => Math.sqrt((GAMMA * 2.0 * (T_keV * KEV_J)) / MBAR);

// ---- 0-D hot-spot burn: integrate [T, n_i] over one confinement time (RK4) ----
function burn(rho, rhoR, T0_keV, n_tau = 1.0) {
  const rho_si = rho * 1000.0;
  const n_i0 = rho_si / MBAR;
  const Rhs = (rhoR * 10.0) / rho_si;
  const tau_c = (n_tau * Rhs) / (4.0 * soundSpeed(T0_keV));

  const deriv = (T, n_i) => {
    T = Math.max(T, 0.2); n_i = Math.max(n_i, 0.0);
    const heat = pAlpha(n_i, T, rhoR) - pBrem(n_i, T);
    const rate = 0.25 * n_i * n_i * reactivityDT(T);
    const dn = -2.0 * rate;
    const dT = n_i > 0 ? (heat - 3.0 * (T * KEV_J) * dn) / (3.0 * n_i * KB) / (KEV_J / KB) : 0.0;
    return [dT, dn];
  };

  const nOut = 400, sub = 20, steps = nOut * sub, dt = tau_c / steps;
  const t = new Array(nOut + 1), Ts = new Array(nOut + 1), ni = new Array(nOut + 1);
  let T = T0_keV, n = n_i0;
  t[0] = 0; Ts[0] = T; ni[0] = n;
  let out = 1;
  for (let i = 1; i <= steps; i++) {
    const [k1T, k1n] = deriv(T, n);
    const [k2T, k2n] = deriv(T + 0.5 * dt * k1T, n + 0.5 * dt * k1n);
    const [k3T, k3n] = deriv(T + 0.5 * dt * k2T, n + 0.5 * dt * k2n);
    const [k4T, k4n] = deriv(T + dt * k3T, n + dt * k3n);
    T += (dt / 6.0) * (k1T + 2 * k2T + 2 * k3T + k4T);
    n += (dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n);
    T = Math.max(T, 0.2); n = Math.max(n, 0.0);
    if (i % sub === 0) { t[out] = i * dt; Ts[out] = T; ni[out] = n; out++; }
  }
  return { t, T: Ts, n_i: ni, tau_c, burnup: 1.0 - n / n_i0 };
}

function evaluateGain(E_J, fuel_kg, CR, adiabat) {
  const T_hs = hydroThs(drivePressure(E_J));
  const rhoR = RHOR_REF * (CR / CR_REF) ** 2 * (ALPHA_REF / adiabat);
  const b = burn(RHO_HS, rhoR, T_hs, N_TAU_BURN);
  const n_react = (b.burnup * fuel_kg) / M_DT_PAIR;
  const Y = n_react * E_FUSION;
  return { gain: Y / E_J, T_hs, yield_J: Y, burnup: b.burnup };
}

// ---- multimode RT mix penalty ----
const ELL = Array.from({ length: DATA.RT_ELLMAX }, (_, i) => i + 1);
const SEED = (() => {
  const s = ELL.map((l) => Math.pow(l, -SPEC_SLOPE));
  const norm = Math.sqrt(s.reduce((a, v) => a + v * v, 0));
  return s.map((v) => v / norm);
})();
const CUTOFF = ELL.map((l) => Math.exp(-((l / ELL_CUT) ** 2)));

function gammaAblative(lam, Va) {
  const k = (2.0 * Math.PI) / lam;
  return Math.max(ALPHA_T * Math.sqrt(ATWOOD * G_ACCEL * k) - BETA_T * k * Va, 0.0);
}
function accelFeedthrough(adiabat) {
  const Va = V_ABL_REF * Math.pow(adiabat / ALPHA_REF, ACCEL_ADIA_EXP);
  return ELL.map((l) => {
    const lam = (2.0 * Math.PI * DATA.RT_R0) / l;
    const nE = Math.min(gammaAblative(lam, Va) * T_ACCEL, 6.0);
    return Math.exp(nE);
  });
}
function gfAt(cr) {
  const CRs = DATA.RT_CR, GF = DATA.RT_GF;
  const c = Math.min(Math.max(cr, CRs[0]), CRs[CRs.length - 1]);
  return ELL.map((_, j) => interp(c, CRs, GF.map((row) => row[j])));
}
const ifarAmp = (cr, ad) => Math.exp(K_IFAR * (Math.sqrt((cr / CR_REF) * (ALPHA_REF / ad)) - 1.0));

function mixState(cr, adiabat, sigma) {
  const accel = accelFeedthrough(adiabat), gf = gfAt(cr);
  const bp = cr / CR_REF, ifar = ifarAmp(cr, adiabat);
  let s2 = 0;
  for (let j = 0; j < ELL.length; j++) {
    const eta = sigma * SEED[j] * accel[j] * gf[j] * CUTOFF[j] * ifar * bp;
    s2 += eta * eta;
  }
  const sigma_mix = Math.sqrt(s2);
  return { sigma_mix, f_mix: sigma_mix / (DATA.RT_R0 / cr) };
}
const mixPenalty = (cr, ad, sigma = SIGMA_SURF_REF) =>
  Math.pow(Math.max(0.0, 1.0 - Math.min(mixState(cr, ad, sigma).f_mix, 1.0)), P_MIX);

const MIX_REF = mixPenalty(35.0, 2.0, SIGMA_SURF_REF);
const mixFactor = (d) => Math.min(mixPenalty(d.CR, d.adiabat, d.surf_nm * 1e-9) / MIX_REF, 1.0 / MIX_REF);

// ---- implosion trajectory R(t): coast + decelerate on a gas cushion tuned to CR ----
const implosionVelocity = (E_MJ) => V_IMP_NIF * Math.sqrt(E_MJ / 2.05);

function trajectory(v_imp, CR, n = 1400) {
  const R_min_target = R0 / CR, exp = 3.0 * GAMMA;
  const tmax = (2.6 * R0) / v_imp;
  const run = (K, steps) => {
    const dt = tmax / steps;
    const T = new Array(steps + 1), R = new Array(steps + 1);
    let r = R0, v = -v_imp, rmin = R0, broke = false;
    for (let i = 0; i <= steps; i++) {
      T[i] = i * dt; R[i] = r;
      const a = K * Math.pow(R0 / Math.max(r, 1e-9), exp);
      const rm = r + 0.5 * dt * v, vm = v + 0.5 * dt * a;
      const am = K * Math.pow(R0 / Math.max(rm, 1e-9), exp);
      r = r + dt * vm; v = v + dt * am;
      rmin = Math.min(rmin, r);
      if (r < R0 / 400.0) { for (let k = i; k <= steps; k++) { T[k] = k * dt; R[k] = r; } broke = true; break; }
    }
    return { T, R, rmin, broke };
  };
  let lo = 1e6, hi = 1e18;
  for (let it = 0; it < 60; it++) {
    const K = Math.sqrt(lo * hi);
    if (run(K, 500).rmin < R_min_target) lo = K; else hi = K;
  }
  const K = Math.sqrt(lo * hi);
  const { T, R } = run(K, n);
  return { t: T, R };
}

// ---- the simulation ----
export function simulate(d, npts = 600) {
  const E_J = d.E_MJ * 1e6;
  const v_imp = implosionVelocity(d.E_MJ);

  const T_stag = hydroThs(drivePressure(E_J));
  const rhoR_stag = RHOR_REF * (d.CR / CR_REF) ** 2 * (ALPHA_REF / d.adiabat);
  const rho_stag = RHO_HS;
  const mix_pen = mixPenalty(d.CR, d.adiabat, d.surf_nm * 1e-9);
  const rhoR_eff = rhoR_stag * Math.max(Math.pow(mix_pen, 0.3), 0.05);

  const gain = evaluateGain(E_J, d.fuel_ug * 1e-9, d.CR, d.adiabat).gain * mixFactor(d);
  const yield_J = gain * E_J;

  const tr = trajectory(v_imp, d.CR);
  const istag = argmin(tr.R);
  const t_stag = tr.t[istag], R_min = tr.R[istag];

  const b = burn(rho_stag, rhoR_eff, T_stag, N_TAU_BURN);
  const phi = b.n_i.map((v) => 1.0 - v / b.n_i[0]);
  const phiEnd = Math.max(phi[phi.length - 1], 1e-9);
  const ignites = amax(b.T) > 1.6 * T_stag;

  const t_end = tr.t[tr.t.length - 1] + b.tau_c;
  const t = linspace(0, t_end, npts);
  const R = t.map((tt) => interp(tt, tr.t, tr.R));
  const comp = R.map((r) => Math.min(Math.max(R_min / Math.max(r, 1e-12), 0), 1));
  const T = comp.map((c) => T_stag * c * c);
  const rho = comp.map((c) => rho_stag * c * c * c);
  const rhoR = comp.map((c) => rhoR_stag * c * c);
  const gain_t = new Array(npts).fill(0);
  const T_end_burn = Math.max(b.T[b.T.length - 1], T_stag);

  for (let i = 0; i < npts; i++) {
    if (t[i] >= t_stag && t[i] <= t_stag + b.tau_c) {
      const tb = t[i] - t_stag;
      T[i] = interp(tb, b.t, b.T);
      rho[i] = rho_stag; rhoR[i] = rhoR_stag;
      gain_t[i] = (gain * interp(tb, b.t, phi)) / phiEnd;
    } else if (t[i] > t_stag + b.tau_c) {
      gain_t[i] = gain;
      const c = Math.min(Math.max(R_min / Math.max(R[i], 1e-12), 0), 1);
      T[i] = T_end_burn * c * c;
    }
  }

  const verdict = mix_pen < 0.25 ? "QUENCHED" : ignites && gain >= 1.0 ? "IGNITES" : "MARGINAL";
  return {
    t, R, T, rho, rhoR, gain_t,
    t_stag, R_min, T_stag, T_peak: amax(T), rhoR_stag,
    mix_penalty: mix_pen, burnup: b.burnup, gain, yield_MJ: yield_J / 1e6,
    ignites, verdict, v_imp, tau_c: b.tau_c,
  };
}

export const PRESETS = {
  "NIF 2022 (first ignition, gain 1.5)": { E_MJ: 2.05, fuel_ug: 210.0, CR: 35.0, adiabat: 2.0, surf_nm: 21.0 },
  "NIF 2025 (record, gain 4.1)": { E_MJ: 2.08, fuel_ug: 225.0, CR: 44.0, adiabat: 1.7, surf_nm: 2.0 },
};
