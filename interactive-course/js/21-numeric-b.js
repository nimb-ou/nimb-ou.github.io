/* ============================================================
   Num, part two — the numerics the later tracks need.
   Gaussian processes, MCMC, bootstrap, bandits, ranking metrics,
   ANN search, detection metrics, quantization, DP noise, power
   analysis, contrastive loss. Everything here computes for real.
   ============================================================ */
(function () {
  'use strict';
  const N = window.Num;
  const rng = N.rng;

  /* ---------------- small helpers ---------------- */
  function linspace(a, b, n) { const o = []; for (let i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function argsort(a, desc) { const idx = a.map((_, i) => i); idx.sort((i, j) => desc ? a[j] - a[i] : a[i] - a[j]); return idx; }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

  /* ---------------- Cholesky, for the GP ----------------
     A = L Lᵀ for symmetric positive definite A. Jitter is added by the
     caller: a kernel matrix is PSD in theory and indefinite in float. */
  function cholesky(A) {
    const n = A.length, L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(1e-12, s));
        else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  /* solve L Lᵀ x = b by two triangular sweeps */
  function cholSolve(L, b) {
    const n = L.length, y = new Array(n), x = new Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return x;
  }

  /* ---------------- kernels ---------------- */
  const kernels = {
    rbf: (a, b, p) => p.v * Math.exp(-0.5 * (a - b) * (a - b) / (p.l * p.l)),
    matern32: (a, b, p) => { const r = Math.SQRT2 * Math.sqrt(1.5) * Math.abs(a - b) / p.l; return p.v * (1 + r) * Math.exp(-r); },
    periodic: (a, b, p) => { const s = Math.sin(Math.PI * Math.abs(a - b) / (p.per || 2)); return p.v * Math.exp(-2 * s * s / (p.l * p.l)); },
    linear: (a, b, p) => p.v * (a * b + 1)
  };

  /* ---------------- Gaussian process regression (1-D) ----------------
     Returns posterior mean and standard deviation on a test grid, plus the
     log marginal likelihood — the quantity that actually selects
     hyperparameters, and the one candidates rarely name. */
  function gp(Xtr, ytr, opts) {
    opts = opts || {};
    const p = { l: opts.l == null ? 1 : opts.l, v: opts.v == null ? 1 : opts.v, per: opts.per };
    const kf = kernels[opts.kernel || 'rbf'];
    const sn2 = Math.pow(opts.noise == null ? 0.1 : opts.noise, 2);
    const n = Xtr.length;
    const K = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => kf(Xtr[i], Xtr[j], p) + (i === j ? sn2 + 1e-8 : 0)));
    const L = n ? cholesky(K) : [];
    const alpha = n ? cholSolve(L, ytr) : [];

    function predict(Xs) {
      return Xs.map(xs => {
        const ks = Xtr.map(xi => kf(xi, xs, p));
        let mu = 0;
        for (let i = 0; i < n; i++) mu += ks[i] * alpha[i];
        let vv = kf(xs, xs, p);
        if (n) {
          const v = cholSolve(L, ks);
          for (let i = 0; i < n; i++) vv -= ks[i] * v[i];
        }
        return { mu: mu, sd: Math.sqrt(Math.max(1e-10, vv)) };
      });
    }
    let logML = 0;
    if (n) {
      let quad = 0, logdet = 0;
      for (let i = 0; i < n; i++) { quad += ytr[i] * alpha[i]; logdet += Math.log(L[i][i]); }
      logML = -0.5 * quad - logdet - 0.5 * n * Math.log(2 * Math.PI);
    }
    return { predict: predict, logML: logML, K: K };
  }

  /* expected improvement, the workhorse acquisition of Bayesian optimisation */
  function expectedImprovement(mu, sd, best, xi) {
    xi = xi == null ? 0.01 : xi;
    if (sd < 1e-9) return 0;
    const z = (best - mu - xi) / sd;               // minimisation
    return (best - mu - xi) * N.normCdf(z) + sd * N.normPdf(z);
  }
  function upperConfidence(mu, sd, kappa) { return -(mu - (kappa == null ? 2 : kappa) * sd); }

  /* ---------------- MCMC: random-walk Metropolis ----------------
     logp need only be known up to a constant — the whole point. */
  function metropolis(logp, x0, opts) {
    opts = opts || {};
    const R = rng(opts.seed || 21);
    const step = opts.step == null ? 0.5 : opts.step;
    let x = x0.slice(), lp = logp(x), accepted = 0, total = 0;
    const chain = [x.slice()];
    function draw(nSteps) {
      for (let i = 0; i < (nSteps || 1); i++) {
        const prop = x.map(v => v + R.normal(0, step));
        const lpp = logp(prop);
        total++;
        if (Math.log(R()) < lpp - lp) { x = prop; lp = lpp; accepted++; }
        chain.push(x.slice());
      }
      return x;
    }
    return { draw: draw, chain: chain, get rate() { return total ? accepted / total : 0; }, get x() { return x.slice(); } };
  }

  /* ---------------- bootstrap ---------------- */
  function bootstrap(data, stat, B, seed) {
    const R = rng(seed || 33), n = data.length, out = [];
    for (let b = 0; b < (B || 400); b++) {
      const s = new Array(n);
      for (let i = 0; i < n; i++) s[i] = data[R.int(n)];
      out.push(stat(s));
    }
    out.sort((a, b) => a - b);
    return { samples: out, lo: out[Math.floor(0.025 * out.length)], hi: out[Math.floor(0.975 * out.length)], mean: N.mean(out), se: N.sd(out) };
  }

  /* ---------------- k-nearest neighbours ---------------- */
  function knn(X, y, k) {
    function predict(p) {
      const d = X.map((q, i) => ({ d: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, i: i }));
      d.sort((a, b) => a.d - b.d);
      let s = 0;
      for (let i = 0; i < Math.min(k, d.length); i++) s += y[d[i].i];
      return s / Math.min(k, d.length);
    }
    return { predict: predict };
  }

  /* ---------------- isotonic regression (pool-adjacent-violators) ----------------
     The other calibrator: no functional form, monotone by construction,
     and happy to overfit a small validation set — which is the trade. */
  function isotonic(x, y) {
    const idx = argsort(x);
    const xs = idx.map(i => x[i]), ys = idx.map(i => y[i]);
    const val = [], wt = [], edge = [];
    for (let i = 0; i < ys.length; i++) {
      let v = ys[i], w = 1, e = xs[i];
      while (val.length && val[val.length - 1] > v) {
        const pv = val.pop(), pw = wt.pop(); edge.pop();
        v = (v * w + pv * pw) / (w + pw); w += pw;
      }
      val.push(v); wt.push(w); edge.push(e);
    }
    // expand back to a step function over the sorted x
    const steps = [];
    let k = 0;
    for (let b = 0; b < val.length; b++) {
      const cnt = wt[b];
      steps.push({ from: xs[k], to: xs[Math.min(xs.length - 1, k + cnt - 1)], v: val[b] });
      k += cnt;
    }
    function predict(v) {
      for (let i = 0; i < steps.length; i++) if (v <= steps[i].to) return steps[i].v;
      return steps.length ? steps[steps.length - 1].v : 0;
    }
    return { steps: steps, predict: predict };
  }

  /* ---------------- ranking metrics ---------------- */
  function dcg(rels, k) {
    let s = 0;
    for (let i = 0; i < Math.min(k || rels.length, rels.length); i++) s += (Math.pow(2, rels[i]) - 1) / Math.log2(i + 2);
    return s;
  }
  function ndcg(rels, k) {
    const ideal = rels.slice().sort((a, b) => b - a);
    const id = dcg(ideal, k);
    return id > 0 ? dcg(rels, k) / id : 0;
  }
  function mrr(rels) { for (let i = 0; i < rels.length; i++) if (rels[i] > 0) return 1 / (i + 1); return 0; }
  function apAtK(rels, k) {
    k = k || rels.length;
    let hits = 0, s = 0;
    for (let i = 0; i < Math.min(k, rels.length); i++) if (rels[i] > 0) { hits++; s += hits / (i + 1); }
    const tot = rels.filter(r => r > 0).length;
    return tot ? s / Math.min(tot, k) : 0;
  }
  function recallAtK(rels, k, totalRelevant) {
    let h = 0;
    for (let i = 0; i < Math.min(k, rels.length); i++) if (rels[i] > 0) h++;
    return totalRelevant ? h / totalRelevant : 0;
  }

  /* ---------------- bandits ---------------- */
  function bandit(trueRates, policy, opts) {
    opts = opts || {};
    const R = rng(opts.seed || 44);
    const K = trueRates.length;
    const n = new Array(K).fill(0), s = new Array(K).fill(0);
    let t = 0, reward = 0, regret = 0;
    const best = Math.max.apply(null, trueRates);
    const history = [];
    function choose() {
      if (policy === 'greedy' || policy === 'eps') {
        const eps = policy === 'greedy' ? 0 : (opts.eps == null ? 0.1 : opts.eps);
        if (R() < eps) return R.int(K);
        let bi = 0, bv = -Infinity;
        for (let i = 0; i < K; i++) { const v = n[i] ? s[i] / n[i] : Infinity; if (v > bv) { bv = v; bi = i; } }
        return bi;
      }
      if (policy === 'ucb') {
        let bi = 0, bv = -Infinity;
        for (let i = 0; i < K; i++) {
          const v = n[i] ? s[i] / n[i] + Math.sqrt(2 * Math.log(t + 1) / n[i]) : Infinity;
          if (v > bv) { bv = v; bi = i; }
        }
        return bi;
      }
      // Thompson: one draw from each Beta posterior, pick the argmax
      let bi = 0, bv = -Infinity;
      for (let i = 0; i < K; i++) { const v = R.beta(1 + s[i], 1 + n[i] - s[i]); if (v > bv) { bv = v; bi = i; } }
      return bi;
    }
    function step(m) {
      for (let it = 0; it < (m || 1); it++) {
        const a = choose();
        const r = R() < trueRates[a] ? 1 : 0;
        n[a]++; s[a] += r; reward += r; t++;
        regret += best - trueRates[a];
        history.push({ a: a, r: r, regret: regret });
      }
    }
    return { step: step, get pulls() { return n.slice(); }, get wins() { return s.slice(); }, get regret() { return regret; }, get t() { return t; }, history: history };
  }

  /* ---------------- detection: IoU, NMS, mAP ---------------- */
  function iou(a, b) {  // boxes as [x1,y1,x2,y2]
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return ar > 0 ? inter / ar : 0;
  }
  function nms(boxes, scores, thr) {
    const order = argsort(scores, true);
    const keep = [], suppressed = {};
    for (const i of order) {
      if (suppressed[i]) continue;
      keep.push(i);
      for (const j of order) if (j !== i && !suppressed[j] && iou(boxes[i], boxes[j]) > thr) suppressed[j] = i;
    }
    return { keep: keep, suppressed: suppressed };
  }
  /* average precision at one IoU threshold, the COCO way (greedy matching) */
  function averagePrecision(dets, gts, thr) {
    const order = argsort(dets.map(d => d.score), true);
    const matched = new Array(gts.length).fill(false);
    let tp = 0, fp = 0;
    const pts = [];
    order.forEach(oi => {
      const d = dets[oi];
      let bi = -1, bv = thr;
      gts.forEach((g, gi) => { if (matched[gi]) return; const v = iou(d.box, g); if (v >= bv) { bv = v; bi = gi; } });
      if (bi >= 0) { matched[bi] = true; tp++; } else fp++;
      pts.push({ p: tp / (tp + fp), r: gts.length ? tp / gts.length : 0 });
    });
    // 101-point interpolated precision, as COCO reports it
    let ap = 0;
    for (let i = 0; i <= 100; i++) {
      const rr = i / 100;
      let pmax = 0;
      pts.forEach(pt => { if (pt.r >= rr && pt.p > pmax) pmax = pt.p; });
      ap += pmax / 101;
    }
    return { ap: ap, curve: pts };
  }

  /* ---------------- quantization ---------------- */
  function quantize(values, bits, opts) {
    opts = opts || {};
    const lo = opts.symmetric ? -Math.max.apply(null, values.map(Math.abs)) : Math.min.apply(null, values);
    const hi = opts.symmetric ? Math.max.apply(null, values.map(Math.abs)) : Math.max.apply(null, values);
    const levels = Math.pow(2, bits) - 1;
    const scale = (hi - lo) / levels || 1;
    const zero = opts.symmetric ? Math.round(-lo / scale) : Math.round(-lo / scale);
    const q = values.map(v => clamp(Math.round(v / scale) + zero, 0, levels));
    const deq = q.map(v => (v - zero) * scale);
    let err = 0;
    for (let i = 0; i < values.length; i++) err += (values[i] - deq[i]) ** 2;
    return { q: q, deq: deq, scale: scale, zero: zero, rmse: Math.sqrt(err / values.length), bits: bits };
  }

  /* ---------------- differential privacy ---------------- */
  function laplaceNoise(sensitivity, epsilon, R) {
    const b = sensitivity / epsilon;
    const u = R() - 0.5;
    return -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }
  function gaussianSigma(sensitivity, epsilon, delta) {
    return sensitivity * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
  }

  /* ---------------- experiment design ---------------- */
  /* minimum detectable effect for two proportions, two-sided */
  function mde(p, nPerArm, alpha, power) {
    alpha = alpha == null ? 0.05 : alpha; power = power == null ? 0.8 : power;
    const za = N.normPpf(1 - alpha / 2), zb = N.normPpf(power);
    return (za + zb) * Math.sqrt(2 * p * (1 - p) / nPerArm);
  }
  function sampleSize(p, lift, alpha, power) {
    alpha = alpha == null ? 0.05 : alpha; power = power == null ? 0.8 : power;
    const za = N.normPpf(1 - alpha / 2), zb = N.normPpf(power);
    return Math.ceil(2 * p * (1 - p) * Math.pow((za + zb) / lift, 2));
  }
  /* False-positive rate when you peek k times with a fixed-horizon test.
     Under the null the running difference in totals is a Brownian motion, so
     the z-statistic at look j is B(t_j)/sqrt(t_j) with equally spaced t_j.
     Simulating the increments directly costs k draws per trial instead of
     re-summing every observation — the same answer, ~10,000× less work. */
  function peekingFPR(k, alpha, seed) {
    const R = rng(seed || 91), trials = 4000;
    const crit = N.normPpf(1 - (alpha || 0.05) / 2);
    let bad = 0;
    for (let t = 0; t < trials; t++) {
      let B = 0;
      for (let look = 1; look <= k; look++) {
        B += R.normal(0, 1);                    // unit-variance increment per look
        if (Math.abs(B / Math.sqrt(look)) > crit) { bad++; break; }
      }
    }
    return bad / trials;
  }
  /* CUPED: variance reduction using a pre-experiment covariate */
  function cuped(y, x) {
    const theta = N.cov(x, y) / (N.variance(x) || 1);
    const adj = y.map((v, i) => v - theta * (x[i] - N.mean(x)));
    return { theta: theta, adjusted: adj, varBefore: N.variance(y), varAfter: N.variance(adj) };
  }

  /* ---------------- contrastive loss ---------------- */
  /* InfoNCE over a batch of paired embeddings; returns loss and the
     similarity matrix that the lab draws. */
  function infoNCE(A, B, temp) {
    const n = A.length;
    const S = A.map(a => B.map(b => N.cosine(a, b) / (temp || 0.1)));
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const row = S[i];
      const lse = N.logsumexp(row);
      loss += -(row[i] - lse);
    }
    return { loss: loss / n, sim: S, acc: S.filter((row, i) => argsort(row, true)[0] === i).length / n };
  }

  /* ---------------- approximate nearest neighbours ----------------
     A navigable small-world graph: every node links to its M nearest
     neighbours among those inserted before it, plus a couple of long-range
     links. Greedy search then walks downhill. Small, but the real algorithm's
     shape — and it reports how many distance computations it did, which is
     the number the recall/latency trade is actually about. */
  function nsw(points, M, seed) {
    const R = rng(seed || 55);
    const n = points.length, adj = Array.from({ length: n }, () => []);
    const d2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; };
    for (let i = 1; i < n; i++) {
      const cand = [];
      for (let j = 0; j < i; j++) cand.push({ j: j, d: d2(points[i], points[j]) });
      cand.sort((a, b) => a.d - b.d);
      for (let k = 0; k < Math.min(M, cand.length); k++) { adj[i].push(cand[k].j); adj[cand[k].j].push(i); }
      if (i > 4) { const r = R.int(i); if (adj[i].indexOf(r) < 0) { adj[i].push(r); adj[r].push(i); } }
    }
    function search(q, ef, entry) {
      let cur = entry == null ? 0 : entry;
      let curD = d2(points[cur], q), comps = 1;
      const visited = { [cur]: 1 }, path = [cur];
      let cands = [{ i: cur, d: curD }];
      const best = [{ i: cur, d: curD }];
      let improved = true;
      while (improved) {
        improved = false;
        const frontier = cands.slice(0, Math.max(1, ef));
        cands = [];
        for (const c of frontier) {
          for (const nb of adj[c.i]) {
            if (visited[nb]) continue;
            visited[nb] = 1;
            const dd = d2(points[nb], q); comps++;
            cands.push({ i: nb, d: dd });
            best.push({ i: nb, d: dd });
            if (dd < curD) { curD = dd; cur = nb; path.push(nb); improved = true; }
          }
        }
        cands.sort((a, b) => a.d - b.d);
      }
      best.sort((a, b) => a.d - b.d);
      return { nearest: best.slice(0, ef).map(b => b.i), comps: comps, path: path };
    }
    function exact(q, k) {
      const all = points.map((p, i) => ({ i: i, d: d2(p, q) }));
      all.sort((a, b) => a.d - b.d);
      return all.slice(0, k).map(a => a.i);
    }
    return { adj: adj, search: search, exact: exact };
  }

  /* ---------------- gradient of a trained net w.r.t. its input ----------------
     Central differences: exact enough in 2-D, and honest about what it is. */
  function inputGrad(f, x, h) {
    h = h || 1e-4;
    return x.map((_, i) => {
      const a = x.slice(), b = x.slice();
      a[i] += h; b[i] -= h;
      return (f(a) - f(b)) / (2 * h);
    });
  }

  /* ---------------- speculative decoding arithmetic ----------------
     Expected accepted tokens per verification pass for draft length k and
     per-token acceptance α (Leviathan et al.): (1-α^{k+1})/(1-α). */
  function specSpeedup(alpha, k, costRatio) {
    const acc = alpha === 1 ? k + 1 : (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
    const cost = 1 + k * (costRatio == null ? 0.15 : costRatio);
    return { accepted: acc, cost: cost, speedup: acc / cost };
  }

  Object.assign(N, {
    linspace, argsort, clamp, cholesky, cholSolve, kernels, gp,
    expectedImprovement, upperConfidence, metropolis, bootstrap, knn, isotonic,
    dcg, ndcg, mrr, apAtK, recallAtK, bandit,
    iou, nms, averagePrecision, quantize, laplaceNoise, gaussianSigma,
    mde, sampleSize, peekingFPR, cuped, infoNCE, nsw, inputGrad, specSpeedup
  });
})();
