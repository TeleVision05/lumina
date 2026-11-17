/**
 * Rotation of SH9 radiance coefficients. Estimators return lighting in the camera frame
 * (docs/CONVENTIONS.md); in WebXR the scene is in the tracking (world) frame, and the phone
 * turns, so the estimate must be rotated by the camera's orientation before it lights the scene.
 *
 * Band 0 is invariant, band 1 rotates as a vector (Y1 ~ y, Y2 ~ z, Y3 ~ x), and band 2 is rotated
 * exactly by evaluating the rotated basis at five fixed directions and solving the 5x5 system
 * (the "rotate by projection on sample directions" method; exact because band 2 is closed under
 * rotation).
 */
import { Quaternion, Vector3 } from 'three';
import { shBasis, type RGB, type SH9, type Vec3 } from './sh';

const DIRS: Vec3[] = (
  [
    [1, 0.3, 0.2],
    [0.1, 1, 0.4],
    [0.3, 0.2, 1],
    [0.7, -0.6, 0.4],
    [-0.5, 0.6, 0.7],
  ] as const
).map(([x, y, z]) => {
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n] as const;
});

const band2 = (d: Vec3): number[] => shBasis(d).slice(4, 9);

/** Solve M x = b for each column of B (M is n x n, rows given), Gaussian elimination. */
function solve(m: number[][], rhs: number[][]): number[][] {
  const n = m.length;
  const a = m.map((row, i) => [...row, ...(rhs[i] ?? [])]);
  const w = (a[0]?.length ?? 0) - n;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(a[r]?.[c] ?? 0) > Math.abs(a[p]?.[c] ?? 0)) p = r;
    }
    const tmp = a[c];
    a[c] = a[p] ?? [];
    a[p] = tmp ?? [];
    const pivot = a[c]?.[c] ?? 0;
    if (Math.abs(pivot) < 1e-12) throw new Error('singular SH rotation system');
    const rowC = a[c] ?? [];
    for (let k = 0; k < n + w; k++) rowC[k] = (rowC[k] ?? 0) / pivot;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const row = a[r] ?? [];
      const f = row[c] ?? 0;
      for (let k = 0; k < n + w; k++) row[k] = (row[k] ?? 0) - f * (rowC[k] ?? 0);
    }
  }
  return a.map((row) => row.slice(n));
}

const IDENTITY5 = [0, 1, 2, 3, 4].map((i) => [0, 1, 2, 3, 4].map((j) => (i === j ? 1 : 0)));
/** Inverse of A[k][m] = Y2_m(N_k). */
const A_INV = solve(DIRS.map(band2), IDENTITY5);

/**
 * SH of the same environment seen in a frame rotated by `q`: if `q` maps camera-frame directions
 * to world-frame directions, the result is the estimate in the world frame.
 */
export function rotateSH9(sh: SH9, q: Quaternion): SH9 {
  const inv = q.clone().invert();
  const v = new Vector3();
  // B[k][m] = Y2_m(R^T N_k); band-2 coefficients after rotation: A^-1 B c.
  const b = DIRS.map((d) => band2(v.set(d[0], d[1], d[2]).applyQuaternion(inv).toArray()));
  const out: RGB[] = [];
  const c = (i: number, ch: number): number => sh[i]?.[ch] ?? 0;
  const v1 = [0, 1, 2].map((ch) => new Vector3(c(3, ch), c(1, ch), c(2, ch)).applyQuaternion(q));
  out.push([c(0, 0), c(0, 1), c(0, 2)]);
  out.push([v1[0]?.y ?? 0, v1[1]?.y ?? 0, v1[2]?.y ?? 0]);
  out.push([v1[0]?.z ?? 0, v1[1]?.z ?? 0, v1[2]?.z ?? 0]);
  out.push([v1[0]?.x ?? 0, v1[1]?.x ?? 0, v1[2]?.x ?? 0]);
  for (let m = 0; m < 5; m++) {
    const rgb = [0, 1, 2].map((ch) => {
      let s = 0;
      for (let k = 0; k < 5; k++) {
        let bc = 0;
        for (let j = 0; j < 5; j++) bc += (b[k]?.[j] ?? 0) * c(4 + j, ch);
        s += (A_INV[m]?.[k] ?? 0) * bc;
      }
      return s;
    });
    out.push([rgb[0] ?? 0, rgb[1] ?? 0, rgb[2] ?? 0]);
  }
  return out;
}
