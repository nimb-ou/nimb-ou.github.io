/* ============================================================
   Num — random, stats, linear algebra, datasets, models, metrics
   Everything here actually computes; no faked animations.
   ============================================================ */
window.Num = (function () {
  'use strict';

  /* ---------------- RNG (mulberry32, seedable) ---------------- */
  function rng(seed) {
    let a = (seed || 12345) >>> 0;
    const f = function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    f.normal = function (mu, sd) {
      let u = 0, v = 0;
      while (u === 0) u = f(); while (v === 0) v = f();
      return (mu || 0) + (sd == null ? 1 : sd) * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    f.int = (n) => Math.floor(f() * n);
    f.pick = (arr) => arr[Math.floor(f() * arr.length)];
    f.shuffle = function (arr) { const a2 = arr.slice(); for (let i = a2.length - 1; i > 0; i--) { const j = f.int(i + 1); const t = a2[i]; a2[i] = a2[j]; a2[j] = t; } return a2; };
    f.poisson = function (lam) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= f(); } while (p > L); return k - 1; };
    f.gamma = function (k) { // Marsaglia-Tsang, k>0
      if (k < 1) return f.gamma(k + 1) * Math.pow(f(), 1 / k);
      const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) { let x, v; do { x = f.normal(0, 1); v = 1 + c * x; } while (v <= 0);
        v = v * v * v; const u = f();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
    };
    f.beta = function (a, b) { const x = f.gamma(a), y = f.gamma(b); return x / (x + y); };
    f.exp = (lam) => -Math.log(1 - f()) / lam;
    return f;
  }

  /* ---------------- special functions ---------------- */
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (x, mu, sd) => 0.5 * (1 + erf(((x - (mu || 0)) / (sd == null ? 1 : sd)) / Math.SQRT2));
  const normPdf = (x, mu, sd) => { sd = sd == null ? 1 : sd; const z = (x - (mu || 0)) / sd; return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI)); };
  function normPpf(p) { // Acklam
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425;
    let q, r2;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r2 = q * q;
    return (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * q / (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1);
  }
  function lgamma(z) {
    const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1; let x = 0.99999999999980993;
    for (let i = 0; i < 8; i++) x += g[i] / (z + i + 1);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  const lchoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  const binomPmf = (k, n, p) => Math.exp(lchoose(n, k) + k * Math.log(p || 1e-300) + (n - k) * Math.log(1 - p || 1e-300));
  const poisPmf = (k, lam) => Math.exp(-lam + k * Math.log(lam) - lgamma(k + 1));
  const betaPdf = (x, a, b) => (x <= 0 || x >= 1) ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + lgamma(a + b) - lgamma(a) - lgamma(b));
  const gammaPdf = (x, k, th) => x <= 0 ? 0 : Math.exp((k - 1) * Math.log(x) - x / th - lgamma(k) - k * Math.log(th));
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  function softmax(z, temp) {
    const t = temp || 1, m = Math.max.apply(null, z);
    const e = z.map(v => Math.exp((v - m) / t));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  const logsumexp = z => { const m = Math.max.apply(null, z); return m + Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0)); };

  /* ---------------- descriptive stats ---------------- */
  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => sum(a) / a.length;
  function variance(a, sample) { const m = mean(a); return sum(a.map(v => (v - m) * (v - m))) / (a.length - (sample ? 1 : 0)); }
  const sd = (a, s) => Math.sqrt(variance(a, s));
  function quantile(a, q) {
    const s = a.slice().sort((x, y) => x - y);
    const p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return s[lo] + (s[hi] - s[lo]) * (p - lo);
  }
  function cov(a, b) { const ma = mean(a), mb = mean(b); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb); return s / a.length; }
  const corr = (a, b) => cov(a, b) / (sd(a) * sd(b));
  function hist(vals, nbins, lo, hi) {
    lo = lo != null ? lo : Math.min.apply(null, vals);
    hi = hi != null ? hi : Math.max.apply(null, vals);
    const bins = new Array(nbins).fill(0), w = (hi - lo) / nbins || 1;
    vals.forEach(v => { const i = Math.min(nbins - 1, Math.max(0, Math.floor((v - lo) / w))); bins[i]++; });
    return { bins: bins, lo: lo, hi: hi, w: w, centers: bins.map((_, i) => lo + w * (i + .5)) };
  }

  /* ---------------- linear algebra ---------------- */
  const zeros = (n, m) => m == null ? new Float64Array(n) : Array.from({ length: n }, () => new Float64Array(m));
  function matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const C = zeros(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; if (!a) continue; for (let j = 0; j < m; j++) C[i][j] += a * B[p][j]; }
    return C;
  }
  const transpose = A => A[0].map((_, j) => Float64Array.from(A.map(r => r[j])));
  const matvec = (A, x) => A.map(row => { let s = 0; for (let j = 0; j < x.length; j++) s += row[j] * x[j]; return s; });
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = a => Math.sqrt(dot(a, a));
  function solve(Ain, bin) { // Gaussian elimination with partial pivoting
    const n = bin.length;
    const A = Ain.map((r, i) => { const row = Array.from(r); row.push(bin[i]); return row; });
    for (let c = 0; c < n; c++) {
      let p = c; for (let r2 = c + 1; r2 < n; r2++) if (Math.abs(A[r2][c]) > Math.abs(A[p][c])) p = r2;
      const t = A[c]; A[c] = A[p]; A[p] = t;
      if (Math.abs(A[c][c]) < 1e-12) A[c][c] = 1e-12;
      for (let r2 = 0; r2 < n; r2++) {
        if (r2 === c) continue;
        const f = A[r2][c] / A[c][c];
        for (let k = c; k <= n; k++) A[r2][k] -= f * A[c][k];
      }
    }
    return A.map((row, i) => row[n] / row[i]);
  }
  /* symmetric eigendecomposition by cyclic Jacobi; returns {values, vectors(columns)} sorted desc */
  function eigSym(Ain) {
    const n = Ain.length;
    const A = Ain.map(r => Array.from(r));
    let V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-16) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
    return { values: idx.map(i => A[i][i]), vectors: idx.map(i => V.map(row => row[i])) }; // vectors[k] = k-th eigenvector
  }
  /* SVD of small matrix via eig of AᵀA */
  function svd(A) {
    const At = transpose(A);
    const AtA = matmul(At, A);
    const e = eigSym(AtA);
    const sv = e.values.map(v => Math.sqrt(Math.max(0, v)));
    const V = e.vectors;
    const U = V.map((v, k) => {
      const av = matvec(A, v);
      const s = sv[k] || 1e-12;
      return av.map(x => x / s);
    });
    return { U: U, s: sv, V: V }; // U[k], V[k] are k-th singular vectors
  }

  /* ---------------- datasets ---------------- */
  function dataset(kind, n, noise, seed) {
    const R = rng(seed || 7); n = n || 200; noise = noise == null ? .2 : noise;
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      let p, c;
      if (kind === 'blobs') {
        c = i % 2;
        const cx = c ? 1.4 : -1.4, cy = c ? 1.1 : -0.9;
        p = [cx + R.normal(0, .7 + noise), cy + R.normal(0, .7 + noise)];
      } else if (kind === 'moons') {
        c = i % 2;
        const t = R() * Math.PI;
        p = c ? [1 - Math.cos(t) - .5 + R.normal(0, noise * .5), 1 - Math.sin(t) - .4 + R.normal(0, noise * .5)]
              : [Math.cos(t) - .5 + R.normal(0, noise * .5), Math.sin(t) - .1 + R.normal(0, noise * .5)];
        p = [p[0] * 1.6, p[1] * 1.6];
      } else if (kind === 'circles') {
        c = i % 2;
        const t = R() * 2 * Math.PI, rad = (c ? 1.9 : .8) + R.normal(0, noise * .6);
        p = [rad * Math.cos(t), rad * Math.sin(t)];
      } else if (kind === 'xor') {
        p = [R() * 4 - 2, R() * 4 - 2];
        c = (p[0] > 0) === (p[1] > 0) ? 1 : 0;
        p = [p[0] + R.normal(0, noise * .4), p[1] + R.normal(0, noise * .4)];
      } else if (kind === 'spiral') {
        c = i % 2;
        const t = 1.2 * Math.PI * (i / n) + (c ? Math.PI : 0), rad = .25 + 2.1 * (i / n);
        p = [rad * Math.cos(t * 2) + R.normal(0, noise * .4), rad * Math.sin(t * 2) + R.normal(0, noise * .4)];
      } else { // linear
        p = [R() * 4 - 2, R() * 4 - 2];
        c = (p[0] * .9 + p[1] * .6 + R.normal(0, noise)) > 0 ? 1 : 0;
      }
      X.push(p); y.push(c);
    }
    return { X: X, y: y };
  }
  function regressionData(n, kind, noise, seed) {
    const R = rng(seed || 3); const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const x = -3 + 6 * i / (n - 1);
      let f;
      if (kind === 'sine') f = Math.sin(x * 1.4) * 1.6;
      else if (kind === 'step') f = x < -1 ? -1.2 : (x < 1 ? .4 : 1.6);
      else if (kind === 'cubic') f = .18 * x * x * x - .5 * x;
      else f = .9 * x + .3;
      X.push(x); y.push(f + R.normal(0, noise == null ? .35 : noise));
    }
    return { X: X, y: y };
  }

  /* ---------------- models ---------------- */
  /* ridge / OLS with polynomial or raw design matrix */
  function ridgeFit(Phi, y, lam) {
    const p = Phi[0].length;
    const Pt = transpose(Phi);
    const A = matmul(Pt, Phi);
    for (let i = 0; i < p; i++) A[i][i] += (lam || 0);
    const b = matvec(Pt, y);
    return solve(A, b);
  }
  const polyDesign = (xs, deg) => xs.map(x => { const r = [1]; for (let d = 1; d <= deg; d++) r.push(Math.pow(x, d)); return r; });

  /* logistic regression, full-batch gradient descent, returns stepper */
  function logistic(X, y, opts) {
    opts = opts || {};
    const n = X.length, d = X[0].length;
    let w = new Array(d).fill(0), b = 0;
    const lr = opts.lr || .5, l2 = opts.l2 || 0;
    function step(iters) {
      for (let it = 0; it < (iters || 1); it++) {
        const gw = new Array(d).fill(0); let gb = 0;
        for (let i = 0; i < n; i++) {
          const p = sigmoid(dot(w, X[i]) + b);
          const e = p - y[i];
          for (let j = 0; j < d; j++) gw[j] += e * X[i][j];
          gb += e;
        }
        for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
        b -= lr * gb / n;
      }
      return { w: w, b: b };
    }
    const predict = x => sigmoid(dot(w, x) + b);
    function loss() {
      let s = 0;
      for (let i = 0; i < n; i++) { const p = Math.min(1 - 1e-9, Math.max(1e-9, predict(X[i]))); s += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p)); }
      return s / n + .5 * l2 * dot(w, w);
    }
    return { step: step, predict: predict, loss: loss, get w() { return w; }, get b() { return b; }, set: (W, B) => { w = W; b = B; } };
  }

  /* k-means with Lloyd steps exposed */
  function kmeans(X, k, seed, plusplus) {
    const R = rng(seed || 5);
    let C;
    if (plusplus) {
      C = [X[R.int(X.length)].slice()];
      while (C.length < k) {
        const d2 = X.map(p => Math.min.apply(null, C.map(c => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)));
        const tot = sum(d2); let u = R() * tot, i = 0;
        while (u > d2[i] && i < d2.length - 1) { u -= d2[i]; i++; }
        C.push(X[i].slice());
      }
    } else {
      C = R.shuffle(X).slice(0, k).map(p => p.slice());
    }
    let assign = new Array(X.length).fill(0);
    function assignStep() {
      let changed = 0;
      X.forEach((p, i) => {
        let best = 0, bd = Infinity;
        C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; best = j; } });
        if (assign[i] !== best) changed++;
        assign[i] = best;
      });
      return changed;
    }
    function updateStep() {
      C.forEach((c, j) => {
        const pts = X.filter((_, i) => assign[i] === j);
        if (!pts.length) return;
        c[0] = mean(pts.map(p => p[0])); c[1] = mean(pts.map(p => p[1]));
      });
    }
    function wcss() { return sum(X.map((p, i) => (p[0] - C[assign[i]][0]) ** 2 + (p[1] - C[assign[i]][1]) ** 2)); }
    return { centers: C, assign: assign, assignStep: assignStep, updateStep: updateStep, wcss: wcss };
  }

  /* DBSCAN */
  function dbscan(X, eps, minPts) {
    const n = X.length, labels = new Array(n).fill(-2); // -2 unvisited, -1 noise
    const neigh = i => { const out = []; for (let j = 0; j < n; j++) { const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1]; if (dx * dx + dy * dy <= eps * eps) out.push(j); } return out; };
    let c = -1;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -2) continue;
      const N = neigh(i);
      if (N.length < minPts) { labels[i] = -1; continue; }
      c++; labels[i] = c;
      const queue = N.slice();
      for (let qi = 0; qi < queue.length; qi++) {
        const j = queue[qi];
        if (labels[j] === -1) labels[j] = c;
        if (labels[j] !== -2) continue;
        labels[j] = c;
        const N2 = neigh(j);
        if (N2.length >= minPts) N2.forEach(x => { if (queue.indexOf(x) < 0) queue.push(x); });
      }
    }
    return { labels: labels, k: c + 1 };
  }

  /* CART decision tree (classification: gini/entropy; regression: variance) */
  function tree(X, y, opts) {
    opts = opts || {};
    const maxDepth = opts.maxDepth == null ? 3 : opts.maxDepth;
    const minLeaf = opts.minLeaf || 1;
    const crit = opts.criterion || 'gini';
    const reg = !!opts.regression;
    const d = X[0].length;
    function impurity(idx) {
      if (reg) { const vals = idx.map(i => y[i]); return idx.length ? variance(vals) : 0; }
      const c1 = idx.reduce((a, i) => a + y[i], 0) / (idx.length || 1);
      const c0 = 1 - c1;
      if (crit === 'entropy') { const h = p => p > 0 ? -p * Math.log2(p) : 0; return h(c0) + h(c1); }
      return 1 - c0 * c0 - c1 * c1;
    }
    function build(idx, depth) {
      const node = { n: idx.length, depth: depth, imp: impurity(idx) };
      node.value = reg ? mean(idx.map(i => y[i])) : idx.reduce((a, i) => a + y[i], 0) / (idx.length || 1);
      if (depth >= maxDepth || idx.length < 2 * minLeaf || node.imp < 1e-9) return node;
      let best = null;
      for (let f = 0; f < d; f++) {
        const vals = idx.map(i => X[i][f]).sort((a, b) => a - b);
        for (let s = minLeaf; s <= vals.length - minLeaf; s++) {
          if (vals[s] === vals[s - 1]) continue;
          const thr = (vals[s] + vals[s - 1]) / 2;
          const L = idx.filter(i => X[i][f] <= thr), Rr = idx.filter(i => X[i][f] > thr);
          if (L.length < minLeaf || Rr.length < minLeaf) continue;
          const gain = node.imp - (L.length * impurity(L) + Rr.length * impurity(Rr)) / idx.length;
          if (!best || gain > best.gain) best = { gain: gain, f: f, thr: thr, L: L, R: Rr };
        }
      }
      if (!best || best.gain <= 1e-9) return node;
      node.f = best.f; node.thr = best.thr; node.gain = best.gain;
      node.left = build(best.L, depth + 1); node.right = build(best.R, depth + 1);
      return node;
    }
    const root = build(X.map((_, i) => i), 0);
    function predict(x, node) { node = node || root; return node.left ? predict(x, x[node.f] <= node.thr ? node.left : node.right) : node.value; }
    return { root: root, predict: predict };
  }

  /* gradient boosting on 1-D or n-D features, squared error or logloss */
  function boosting(X, y, opts) {
    opts = opts || {};
    const lr = opts.lr == null ? .3 : opts.lr, depth = opts.maxDepth == null ? 2 : opts.maxDepth;
    const logit = !!opts.logistic;
    const base = logit ? 0 : mean(y);
    const trees = [];
    let F = X.map(() => base);
    function addTree() {
      const g = X.map((_, i) => logit ? (y[i] - sigmoid(F[i])) : (y[i] - F[i]));  // negative gradient
      const t = tree(X, g, { maxDepth: depth, regression: true, minLeaf: opts.minLeaf || 1 });
      trees.push(t);
      X.forEach((x, i) => { F[i] += lr * t.predict(x); });
      return t;
    }
    function predictRaw(x) { let v = base; trees.forEach(t => v += lr * t.predict(x)); return v; }
    const predict = x => logit ? sigmoid(predictRaw(x)) : predictRaw(x);
    function loss() {
      if (logit) return mean(X.map((x, i) => { const p = Math.min(1 - 1e-9, Math.max(1e-9, predict(x))); return -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p)); }));
      return mean(X.map((x, i) => (y[i] - predict(x)) ** 2));
    }
    return { addTree: addTree, predict: predict, predictRaw: predictRaw, loss: loss, trees: trees, get F() { return F; } };
  }

  /* MLP with manual backprop; activations relu/tanh/sigmoid; binary cross-entropy or MSE */
  function mlp(sizes, opts) {
    opts = opts || {};
    const R = rng(opts.seed || 11);
    const act = opts.act || 'relu';
    const W = [], B = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan = sizes[l], gain = act === 'relu' ? Math.sqrt(2 / fan) : Math.sqrt(1 / fan);
      W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: fan }, () => R.normal(0, gain))));
      B.push(new Array(sizes[l + 1]).fill(0));
    }
    const f = { relu: z => z > 0 ? z : 0, tanh: Math.tanh, sigmoid: sigmoid, linear: z => z };
    const df = { relu: (z, a) => z > 0 ? 1 : 0, tanh: (z, a) => 1 - a * a, sigmoid: (z, a) => a * (1 - a), linear: () => 1 };
    // Adam state
    const mW = W.map(l => l.map(r => r.map(() => 0))), vW = W.map(l => l.map(r => r.map(() => 0)));
    const mB = B.map(l => l.map(() => 0)), vB = B.map(l => l.map(() => 0));
    let t = 0;

    function forward(x) {
      const zs = [], as = [x];
      for (let l = 0; l < W.length; l++) {
        const z = W[l].map((row, i) => dot(row, as[l]) + B[l][i]);
        const last = l === W.length - 1;
        const a = z.map(v => last ? sigmoid(v) : f[act](v));
        zs.push(z); as.push(a);
      }
      return { zs: zs, as: as };
    }
    const predict = x => forward(x).as[W.length][0];

    function trainBatch(X, y, lr, l2) {
      lr = lr || .03; l2 = l2 || 0; t++;
      const n = X.length;
      const gW = W.map(l => l.map(r => r.map(() => 0)));
      const gB = B.map(l => l.map(() => 0));
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const { zs, as } = forward(X[i]);
        const out = as[W.length][0];
        const p = Math.min(1 - 1e-9, Math.max(1e-9, out));
        loss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
        let delta = [out - y[i]];                     // dL/dz for sigmoid+BCE
        for (let l = W.length - 1; l >= 0; l--) {
          for (let j = 0; j < W[l].length; j++) {
            gB[l][j] += delta[j];
            for (let k = 0; k < W[l][j].length; k++) gW[l][j][k] += delta[j] * as[l][k];
          }
          if (l > 0) {
            const nd = new Array(W[l][0].length).fill(0);
            for (let k = 0; k < nd.length; k++) {
              let s = 0;
              for (let j = 0; j < W[l].length; j++) s += W[l][j][k] * delta[j];
              nd[k] = s * df[act](zs[l - 1][k], as[l][k]);
            }
            delta = nd;
          }
        }
      }
      const b1 = .9, b2 = .999, eps = 1e-8;
      for (let l = 0; l < W.length; l++) {
        for (let j = 0; j < W[l].length; j++) {
          for (let k = 0; k < W[l][j].length; k++) {
            const g = gW[l][j][k] / n + l2 * W[l][j][k];
            mW[l][j][k] = b1 * mW[l][j][k] + (1 - b1) * g;
            vW[l][j][k] = b2 * vW[l][j][k] + (1 - b2) * g * g;
            const mh = mW[l][j][k] / (1 - Math.pow(b1, t)), vh = vW[l][j][k] / (1 - Math.pow(b2, t));
            W[l][j][k] -= lr * mh / (Math.sqrt(vh) + eps);
          }
          const g = gB[l][j] / n;
          mB[l][j] = b1 * mB[l][j] + (1 - b1) * g;
          vB[l][j] = b2 * vB[l][j] + (1 - b2) * g * g;
          const mh = mB[l][j] / (1 - Math.pow(b1, t)), vh = vB[l][j] / (1 - Math.pow(b2, t));
          B[l][j] -= lr * mh / (Math.sqrt(vh) + eps);
        }
      }
      return loss / n;
    }
    return { forward: forward, predict: predict, trainBatch: trainBatch, W: W, B: B, get steps() { return t; } };
  }

  /* kernel SVM via simplified SMO (small n) */
  function svm(X, y01, opts) {
    opts = opts || {};
    const C = opts.C == null ? 1 : opts.C, tol = 1e-3, maxPasses = opts.passes || 8;
    const n = X.length;
    const y = y01.map(v => v ? 1 : -1);
    const kern = opts.kernel || 'rbf', gam = opts.gamma == null ? 1 : opts.gamma, deg = opts.degree || 3;
    function K(a, b) {
      if (kern === 'linear') return dot(a, b);
      if (kern === 'poly') return Math.pow(1 + dot(a, b), deg);
      const dx = a[0] - b[0], dy = a[1] - b[1];
      return Math.exp(-gam * (dx * dx + dy * dy));
    }
    const Km = X.map(a => X.map(b => K(a, b)));
    const alpha = new Array(n).fill(0); let b = 0;
    const R = rng(opts.seed || 3);
    const fx = i => { let s = b; for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * y[j] * Km[i][j]; return s; };
    let passes = 0;
    while (passes < maxPasses) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = fx(i) - y[i];
        if ((y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0)) {
          let j = R.int(n); if (j === i) j = (j + 1) % n;
          const Ej = fx(j) - y[j];
          const ai = alpha[i], aj = alpha[j];
          let L, H;
          if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(C, C + aj - ai); }
          else { L = Math.max(0, ai + aj - C); H = Math.min(C, ai + aj); }
          if (L >= H) continue;
          const eta = 2 * Km[i][j] - Km[i][i] - Km[j][j];
          if (eta >= 0) continue;
          let ajn = aj - y[j] * (Ei - Ej) / eta;
          ajn = Math.min(H, Math.max(L, ajn));
          if (Math.abs(ajn - aj) < 1e-6) continue;
          const ain = ai + y[i] * y[j] * (aj - ajn);
          const b1 = b - Ei - y[i] * (ain - ai) * Km[i][i] - y[j] * (ajn - aj) * Km[i][j];
          const b2 = b - Ej - y[i] * (ain - ai) * Km[i][j] - y[j] * (ajn - aj) * Km[j][j];
          alpha[i] = ain; alpha[j] = ajn;
          b = (ain > 0 && ain < C) ? b1 : (ajn > 0 && ajn < C) ? b2 : (b1 + b2) / 2;
          changed++;
        }
      }
      passes = changed ? 0 : passes + 1;
      if (changed === 0) break;
    }
    const sv = [];
    for (let i = 0; i < n; i++) if (alpha[i] > 1e-6) sv.push(i);
    const decide = x => { let s = b; sv.forEach(i => s += alpha[i] * y[i] * K(X[i], x)); return s; };
    return { decide: decide, sv: sv, alpha: alpha, b: b, margin: 1 / Math.sqrt(Math.max(1e-9, sv.reduce((a, i) => a + sv.reduce((c, j) => c + alpha[i] * alpha[j] * y[i] * y[j] * Km[i][j], 0), 0))) };
  }

  /* PCA on 2-D+ data */
  function pca(X) {
    const d = X[0].length, n = X.length;
    const mu = Array.from({ length: d }, (_, j) => mean(X.map(r => r[j])));
    const Xc = X.map(r => r.map((v, j) => v - mu[j]));
    const S = Array.from({ length: d }, () => new Array(d).fill(0));
    Xc.forEach(r => { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) S[i][j] += r[i] * r[j] / n; });
    const e = eigSym(S);
    return { mean: mu, cov: S, values: e.values, vectors: e.vectors, explained: e.values.map(v => v / sum(e.values)) };
  }

  /* GMM/EM in 1-D (k components) */
  function gmm1d(x, k, seed) {
    const R = rng(seed || 9);
    const mu = Array.from({ length: k }, () => x[R.int(x.length)]);
    const sg = new Array(k).fill(sd(x) || 1);
    const pi = new Array(k).fill(1 / k);
    let resp = [];
    function eStep() {
      resp = x.map(v => {
        const w = mu.map((m, j) => pi[j] * normPdf(v, m, sg[j]) + 1e-300);
        const s = sum(w); return w.map(a => a / s);
      });
      return mean(x.map((v, i) => Math.log(sum(mu.map((m, j) => pi[j] * normPdf(v, m, sg[j]) + 1e-300)))));
    }
    function mStep() {
      for (let j = 0; j < k; j++) {
        const nk = sum(resp.map(r => r[j])) + 1e-12;
        mu[j] = sum(x.map((v, i) => resp[i][j] * v)) / nk;
        sg[j] = Math.max(.08, Math.sqrt(sum(x.map((v, i) => resp[i][j] * (v - mu[j]) ** 2)) / nk));
        pi[j] = nk / x.length;
      }
    }
    return { mu: mu, sg: sg, pi: pi, eStep: eStep, mStep: mStep, get resp() { return resp; } };
  }

  /* ---------------- metrics ---------------- */
  function rocCurve(scores, labels) {
    const pairs = scores.map((s, i) => [s, labels[i]]).sort((a, b) => b[0] - a[0]);
    const P = sum(labels), N = labels.length - P;
    let tp = 0, fp = 0; const pts = [[0, 0]]; const pr = [];
    pairs.forEach(p => {
      if (p[1]) tp++; else fp++;
      pts.push([fp / (N || 1), tp / (P || 1)]);
      pr.push([tp / (P || 1), tp / (tp + fp)]);
    });
    let auc = 0;
    for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2;
    let ap = 0;
    for (let i = 1; i < pr.length; i++) ap += (pr[i][0] - pr[i - 1][0]) * pr[i][1];
    return { roc: pts, pr: pr, auc: auc, ap: ap };
  }
  function confusion(scores, labels, thr) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    scores.forEach((s, i) => {
      const p = s >= thr ? 1 : 0;
      if (p && labels[i]) tp++; else if (p && !labels[i]) fp++; else if (!p && labels[i]) fn++; else tn++;
    });
    const prec = tp / (tp + fp || 1), rec = tp / (tp + fn || 1);
    return { tp, fp, tn, fn, precision: prec, recall: rec, f1: 2 * prec * rec / (prec + rec || 1), accuracy: (tp + tn) / labels.length };
  }
  const brier = (p, y) => mean(p.map((v, i) => (v - y[i]) ** 2));
  function ksStat(scores, labels) {
    const pairs = scores.map((s, i) => [s, labels[i]]).sort((a, b) => a[0] - b[0]);
    const P = sum(labels), N = labels.length - P;
    let cp = 0, cn = 0, ks = 0;
    pairs.forEach(p => { if (p[1]) cp++; else cn++; ks = Math.max(ks, Math.abs(cp / (P || 1) - cn / (N || 1))); });
    return ks;
  }
  function calibrationBins(p, y, nb) {
    nb = nb || 10;
    const bins = Array.from({ length: nb }, () => ({ n: 0, sp: 0, sy: 0 }));
    p.forEach((v, i) => { const b = Math.min(nb - 1, Math.floor(v * nb)); bins[b].n++; bins[b].sp += v; bins[b].sy += y[i]; });
    return bins.map((b, i) => ({ mid: (i + .5) / nb, n: b.n, pred: b.n ? b.sp / b.n : null, obs: b.n ? b.sy / b.n : null }));
  }
  function psi(expected, actual, bins) {
    let s = 0;
    for (let i = 0; i < expected.length; i++) {
      const e = Math.max(1e-6, expected[i]), a = Math.max(1e-6, actual[i]);
      s += (a - e) * Math.log(a / e);
    }
    return s;
  }
  const entropy = p => -sum(p.filter(v => v > 0).map(v => v * Math.log2(v)));
  const klDiv = (p, q) => sum(p.map((v, i) => v > 0 ? v * Math.log(v / Math.max(1e-12, q[i])) : 0));
  const crossEntropy = (p, q) => -sum(p.map((v, i) => v > 0 ? v * Math.log(Math.max(1e-12, q[i])) : 0));

  /* ---------------- text / retrieval ---------------- */
  const tokenizeWords = s => (s.toLowerCase().match(/[a-z0-9']+/g) || []);
  function bm25Index(docs, k1, b) {
    k1 = k1 == null ? 1.2 : k1; b = b == null ? .75 : b;
    const toks = docs.map(tokenizeWords);
    const dl = toks.map(t => t.length), avg = mean(dl);
    const df = {};
    toks.forEach(t => new Set(t).forEach(w => df[w] = (df[w] || 0) + 1));
    const N = docs.length;
    const tf = toks.map(t => { const m = {}; t.forEach(w => m[w] = (m[w] || 0) + 1); return m; });
    return function (query) {
      const q = tokenizeWords(query);
      return docs.map((_, i) => {
        let s = 0;
        q.forEach(w => {
          const f = tf[i][w] || 0; if (!f) return;
          const idf = Math.log(1 + (N - df[w] + .5) / (df[w] + .5));
          s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avg));
        });
        return s;
      });
    };
  }
  /* toy "dense" embedding: hashed bag of character trigrams, L2-normalised */
  function embed(text, dim) {
    dim = dim || 64;
    const v = new Float64Array(dim);
    const s = ' ' + text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ') + ' ';
    for (let i = 0; i < s.length - 2; i++) {
      const g = s.substr(i, 3);
      let h = 2166136261;
      for (let j = 0; j < g.length; j++) { h ^= g.charCodeAt(j); h = Math.imul(h, 16777619); }
      v[Math.abs(h) % dim] += 1;
      v[Math.abs(Math.imul(h, 31)) % dim] += 0.5;
    }
    const nrm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
    for (let i = 0; i < dim; i++) v[i] /= nrm;
    return v;
  }
  const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  function rrf(rankLists, k) {
    k = k == null ? 60 : k;
    const score = {};
    rankLists.forEach(list => list.forEach((id, i) => { score[id] = (score[id] || 0) + 1 / (k + i + 1); }));
    return Object.keys(score).map(id => ({ id: isNaN(+id) ? id : +id, score: score[id] })).sort((a, b) => b.score - a.score);
  }

  /* byte-pair encoding: learn merges from a corpus, then encode */
  function bpeTrain(corpus, numMerges) {
    const words = {};
    tokenizeWords(corpus).forEach(w => { const k = w.split('').join(' ') + ' </w>'; words[k] = (words[k] || 0) + 1; });
    const merges = [];
    for (let m = 0; m < numMerges; m++) {
      const pairs = {};
      Object.keys(words).forEach(w => {
        const sym = w.split(' ');
        for (let i = 0; i < sym.length - 1; i++) { const p = sym[i] + ' ' + sym[i + 1]; pairs[p] = (pairs[p] || 0) + words[w]; }
      });
      let best = null;
      Object.keys(pairs).forEach(p => { if (!best || pairs[p] > pairs[best]) best = p; });
      if (!best || pairs[best] < 2) break;
      merges.push(best);
      const [a, b] = best.split(' ');
      const nw = {};
      Object.keys(words).forEach(w => {
        const rep = w.split(' ');
        const out = [];
        for (let i = 0; i < rep.length; i++) {
          if (rep[i] === a && rep[i + 1] === b) { out.push(a + b); i++; } else out.push(rep[i]);
        }
        nw[out.join(' ')] = (nw[out.join(' ')] || 0) + words[w];
      });
      Object.keys(nw).forEach(k => { });
      for (const k in words) delete words[k];
      Object.assign(words, nw);
    }
    const vocab = new Set();
    Object.keys(words).forEach(w => w.split(' ').forEach(s => vocab.add(s)));
    return { merges: merges, vocab: Array.from(vocab).sort() };
  }
  function bpeEncode(word, merges) {
    let sym = word.toLowerCase().split('').concat(['</w>']);
    merges.forEach(m => {
      const [a, b] = m.split(' ');
      const out = [];
      for (let i = 0; i < sym.length; i++) {
        if (sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]);
      }
      sym = out;
    });
    return sym;
  }

  /* char-level n-gram language model, trained in the browser */
  function charLM(text, order) {
    order = order || 3;
    const counts = new Map();
    const pad = ''.repeat(order);
    const t = pad + text + '';
    for (let i = order; i < t.length; i++) {
      const ctx = t.slice(i - order, i), ch = t[i];
      let m = counts.get(ctx); if (!m) { m = new Map(); counts.set(ctx, m); }
      m.set(ch, (m.get(ch) || 0) + 1);
    }
    function dist(ctx) {
      let o = order;
      while (o > 0) {
        const c = ctx.slice(-o).padStart(o, '');
        const m = counts.get(c);
        if (m && m.size) {
          const tot = Array.from(m.values()).reduce((a, b) => a + b, 0);
          return Array.from(m.entries()).map(([ch, n]) => ({ ch: ch, p: n / tot })).sort((a, b) => b.p - a.p);
        }
        o--;
      }
      return [{ ch: ' ', p: 1 }];
    }
    return { dist: dist, order: order };
  }

  return {
    rng, erf, normCdf, normPdf, normPpf, lgamma, lchoose, binomPmf, poisPmf, betaPdf, gammaPdf,
    sigmoid, softmax, logsumexp, sum, mean, variance, sd, quantile, cov, corr, hist,
    zeros, matmul, transpose, matvec, dot, norm, solve, eigSym, svd,
    dataset, regressionData, ridgeFit, polyDesign, logistic, kmeans, dbscan, tree, boosting, mlp, svm, pca, gmm1d,
    rocCurve, confusion, brier, ksStat, calibrationBins, psi, entropy, klDiv, crossEntropy,
    tokenizeWords, bm25Index, embed, cosine, rrf, bpeTrain, bpeEncode, charLM
  };
})();
