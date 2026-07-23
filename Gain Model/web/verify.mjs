// Verify the JS port matches the Python reference (implosion_timeline.py).
// Usage:  node web/verify.mjs        (run from the Gain Model directory)
import { readFileSync } from "node:fs";
import { simulate } from "./implosion_timeline.js";

const ref = JSON.parse(readFileSync(new URL("./reference.json", import.meta.url)));
const relerr = (a, b, floor) => Math.abs(a - b) / Math.max(Math.abs(b), floor);

let worstScalar = 0, worstArray = 0, fail = false;
const SCALAR_TOL = 0.05, ARRAY_TOL = 0.06;

for (const c of ref.cases) {
  const [E_MJ, fuel_ug, CR, adiabat, surf_nm, drive_asym_pct = 0] = c.design;
  const s = simulate({ E_MJ, fuel_ug, CR, adiabat, surf_nm, drive_asym_pct });

  let cs = 0;
  for (const [k, v] of Object.entries(c.scalars)) {
    const e = relerr(s[k], v, k === "gain" || k === "yield_MJ" || k === "burnup" ? 0.02 : 1e-3);
    cs = Math.max(cs, e); worstScalar = Math.max(worstScalar, e);
  }
  let ca = 0;
  for (const k of Object.keys(c.arrays)) {
    const jsArr = c.arrays[k].map((_, i) => s[k][ref.idx[i]]);
    const floor = 0.02 * Math.max(...c.arrays[k].map(Math.abs));
    for (let i = 0; i < jsArr.length; i++) ca = Math.max(ca, relerr(jsArr[i], c.arrays[k][i], Math.max(floor, 1e-9)));
    worstArray = Math.max(worstArray, ca);
  }
  const vmatch = s.verdict === c.verdict && s.ignites === c.ignites;
  const ok = cs < SCALAR_TOL && ca < ARRAY_TOL && vmatch;
  if (!ok) fail = true;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${c.name.padEnd(9)} ` +
      `gain js=${s.gain.toFixed(3)} py=${c.scalars.gain.toFixed(3)}  ` +
      `verdict ${s.verdict}${vmatch ? "" : ` != ${c.verdict}`}  ` +
      `maxScalarErr=${(cs * 100).toFixed(1)}%  maxArrayErr=${(ca * 100).toFixed(1)}%`
  );
}

console.log(
  `\nworst scalar error ${(worstScalar * 100).toFixed(1)}%  (tol ${SCALAR_TOL * 100}%),  ` +
    `worst array error ${(worstArray * 100).toFixed(1)}%  (tol ${ARRAY_TOL * 100}%)`
);
console.log(fail ? "RESULT: MISMATCH" : "RESULT: JS port matches Python reference ✓");
process.exit(fail ? 1 : 0);
