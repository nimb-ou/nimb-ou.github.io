/* ============================================================
   PART 7 — The ML interview (7.5 – 7.7): the coding round,
   the case round, and the behavioural round + a prep plan.
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 7.5 */
  ML.section({
    id: 'coding-round', track: 'interview', num: '7.5', level: 2,
    title: 'The coding round: implement it from scratch',
    lede: 'Two kinds of coding round exist. One is standard algorithms; the other asks you to implement a piece of machine learning without a library. The second is where ML candidates lose points they did not know were available, and every drill below runs its tests in your browser.',
    prereq: ['python-toolkit'],
    related: ['metrics', 'attention', 'unsupervised'],
    html: `
${H.tldr([
      'The ritual that scores: <b>clarify → state the approach → state the complexity → write it → test it out loud</b>. Skipping the last two is the most common avoidable loss.',
      'The from-scratch classics: <b>softmax (stably), AUC, IoU + NMS, one k-means step, scaled dot-product attention, and a gradient-descent loop</b>. All six are below, with tests.',
      'Numerical stability is a scored dimension. A softmax that overflows, or a variance that can go negative, is a wrong answer even when the algebra is right (§1.15).'
    ])}

<h2><span class="sn">7.5.1</span> The five minutes before you write anything</h2>
${H.steps([
      '<b>Restate the problem and give one example.</b> "So given scores and binary labels, return the AUC — for scores [0.9, 0.1] and labels [1, 0] that is 1.0." This catches misunderstandings while they are free.',
      '<b>Ask about inputs.</b> Can they be empty? Are there ties? What types? Can the arrays be different lengths? Each of these is a test case you will write later.',
      '<b>State the approach in one sentence, then the complexity.</b> "Sort by score, sweep once accumulating TPs and FPs — $O(n\\log n)$ time, $O(n)$ space." If the interviewer wants a different approach, this is where they say so — before you have written thirty lines.',
      '<b>Write it, narrating.</b> Silence is the enemy: the interviewer is scoring your reasoning, and they cannot score what they cannot hear.',
      '<b>Test out loud.</b> The normal case, an edge case, and a degenerate case. <b>Finding your own bug is a positive signal</b>, not a negative one — it is the closest thing in an interview to watching you work.'
    ])}
${H.key('The complexity statement is not a formality. Half of ML coding rounds have a follow-up of the form "now it has to handle a hundred million rows" — and that conversation is impossible if you never established the current cost.')}

<h2><span class="sn">7.5.2</span> Six drills, with tests that run</h2>
<p class="small">Each editor holds a JavaScript function expression. Write the body, press <b>Run tests</b>, and the hidden cases execute against it. The algorithms are identical to their NumPy equivalents — the point is the reasoning, not the language.</p>

${H.drill('d1', 'Numerically stable softmax')}
${H.drill('d2', 'ROC-AUC from scratch')}
${H.drill('d3', 'IoU, then non-maximum suppression')}
${H.drill('d4', 'One Lloyd step of k-means')}
${H.drill('d5', 'Scaled dot-product attention')}
${H.drill('d6', 'Gradient descent for linear regression')}

<h2><span class="sn">7.5.3</span> The follow-ups that always come</h2>
${H.table(['Drill', 'Follow-up you should expect', 'The answer'], [
      ['Softmax', 'Why subtract the max?', '$e^{z}$ overflows past ~709 in float64. Subtracting the max makes the largest exponent $e^0=1$ and leaves the result mathematically identical (§1.15).'],
      ['AUC', 'What about ties in the scores?', 'Tied pairs count as half. The rank-based (Mann–Whitney) formulation handles it naturally by using average ranks.'],
      ['AUC', 'Now for a hundred million rows.', 'The sort dominates at $O(n\\log n)$. Bin the scores into a fixed number of buckets and accumulate — $O(n)$ with a bounded approximation error.'],
      ['NMS', 'What if there are 100k boxes?', 'The naive version is $O(n^2)$. Sort once, then use a spatial index or grid to only test nearby boxes; or use a GPU kernel, which is what detectors actually ship.'],
      ['k-means', 'What if a cluster becomes empty?', 'Reinitialise that centre — to the point furthest from any centre, or a random point. Leaving it empty produces NaN on the next mean.'],
      ['Attention', 'How would you add a causal mask?', 'Add $-\\infty$ (in practice a large negative number) to the upper triangle of the scores <i>before</i> the softmax, so masked positions get exactly zero weight.'],
      ['Gradient descent', 'How do you choose the learning rate?', 'An LR range test (§3.12), or for least squares the stability bound $\\eta < 2/\\lambda_{\\max}(X^\\top X)$ directly (§1.12).']
    ])}

<h2><span class="sn">7.5.4</span> The standard-algorithms half</h2>
${H.table(['Pattern', 'Shows up as', 'Practise until automatic'], [
      'Hash map counting|"top-k frequent", "first unique", dedupe|building the map and the heap in one pass'.split('|'),
      'Two pointers / sliding window|"longest window with at most k", "max sum subarray"|shrinking from the left correctly'.split('|'),
      'Sorting + sweep|intervals, AUC, NMS, calibration bins|writing the comparator without a bug'.split('|'),
      'Heap|top-k, k-way merge, streaming median|when a heap beats a sort ($k \\ll n$)'.split('|'),
      'Binary search on the answer|"minimum capacity such that…"|the predicate, and the invariant'.split('|'),
      'Prefix sums|range queries, cumulative metrics|the off-by-one at index 0'.split('|'),
      'Graph BFS/DFS|dependency order, connected components|iterative DFS, to avoid stack overflow'.split('|'),
      'Dynamic programming|edit distance, longest common subsequence|stating the recurrence before coding'.split('|')
    ])}
${H.pitfall('The most common non-algorithmic failure is <b>not running your own code mentally</b>. Before you say "done", trace one small input through it line by line, out loud. Interviewers overwhelmingly prefer a candidate who finds their own off-by-one to one who declares victory on broken code.')}

${H.iq('Coding-round questions that are really design questions', [
      {
        q: 'Implement train/test split. Now make it correct for a dataset with multiple rows per customer.',
        level: 'core',
        a: `<p>The naive version shuffles indices and slices. The follow-up is the real question: with several rows per customer, a random split puts the <i>same customer</i> in both sets, and the model memorises the customer rather than learning the relationship. That is group leakage (§2.11).</p>
<p>Fix: split on <b>unique customer IDs</b>, then select rows by membership. If the data are also temporal, split by time as well — group and time constraints compose, and getting only one of them is a very common bug.</p>`,
        follow: ['What if you also need class stratification?', 'What about time-series data?', 'How do you check the split is correct?']
      },
      {
        q: 'Write a function that computes a rolling feature over user events. Watch out.',
        level: 'senior',
        a: `<p>The trap is inclusivity of the current event. A "count of transactions in the last 30 days" that <i>includes</i> the current transaction is fine; one that includes anything <i>after</i> the prediction time is time-travel leakage and will look brilliant offline and fail live.</p>
<p>State the invariant before coding: <b>every feature for a row at time $t$ uses only data with timestamp strictly less than $t$</b> (or ≤ $t$ if the event itself is known at decision time — say which, explicitly). Then implement with a sorted sweep and a deque, which is $O(n)$ rather than the $O(n^2)$ recompute.</p>`,
        follow: ['How would you validate this offline?', 'How do you keep training and serving consistent (§5.12)?']
      },
      {
        q: 'You have 10GB of data and 8GB of RAM. Compute the mean and variance of a column.',
        level: 'core',
        a: `<p>Stream it in chunks and use <b>Welford's algorithm</b> (§1.15): one pass, numerically stable, constant memory. The naive $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$ is one pass too, but it suffers catastrophic cancellation for large means and can return a negative variance.</p>
<p>Then the extension: parallel chunks can be combined with the pairwise merge formula for mean and $M_2$, which is how distributed frameworks implement it.</p>`,
        follow: ['How would you compute a median under the same constraint?', 'What about a quantile?', 'How would you parallelise it?']
      }
    ])}`,
    labs: {
      d1: function (host) {
        Labs.codeDrill(host, {
          brief: 'Return the softmax of an array of logits. It must not overflow for large inputs — that is the whole point of the exercise.',
          signature: 'function softmax(z: number[]): number[]',
          starter: 'function (z) {\n  // your code here\n}',
          hint: 'Subtract max(z) from every element before exponentiating. The result is mathematically identical because the constant cancels in the ratio.',
          solution: 'function (z) {\n  const m = Math.max.apply(null, z);\n  const e = z.map(v => Math.exp(v - m));\n  const s = e.reduce((a, b) => a + b, 0);\n  return e.map(v => v / s);\n}',
          tests: [
            {
              name: 'sums to 1 on a simple input',
              run: fn => { const r = fn([2, 1, 0.1]); const s = r.reduce((a, b) => a + b, 0); return Math.abs(s - 1) < 1e-9 || 'sum was ' + s; }
            },
            {
              name: 'matches known values for [2, 1, 0.1]',
              run: fn => { const r = fn([2, 1, 0.1]); return Math.abs(r[0] - 0.65900114) < 1e-6 || 'first element ' + r[0]; }
            },
            {
              name: 'does not overflow for [1000, 1001, 1002]',
              run: fn => { const r = fn([1000, 1001, 1002]); return r.every(v => isFinite(v)) && Math.abs(r[2] - 0.66524096) < 1e-6 || 'got ' + JSON.stringify(r); }
            },
            {
              name: 'handles a single element',
              run: fn => { const r = fn([5]); return Math.abs(r[0] - 1) < 1e-12 || 'got ' + r[0]; }
            },
            {
              name: 'is shift-invariant: softmax(z) == softmax(z + 100)',
              run: fn => { const a = fn([1, 2, 3]), b = fn([101, 102, 103]); return a.every((v, i) => Math.abs(v - b[i]) < 1e-9) || 'shift changed the result'; }
            }
          ]
        });
      },

      d2: function (host) {
        Labs.codeDrill(host, {
          brief: 'Compute ROC-AUC from raw scores and binary labels, without sorting into a library call. Ties should count as half.',
          signature: 'function auc(scores: number[], labels: number[]): number',
          starter: 'function (scores, labels) {\n  // your code here\n}',
          hint: 'AUC = P(random positive ranks above random negative). Either sweep the sorted scores accumulating TP/FP, or use the rank formula: (sum of positive ranks − P(P+1)/2) / (P·N).',
          solution: 'function (scores, labels) {\n  const idx = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b]);\n  // average ranks for ties (1-based)\n  const rank = new Array(scores.length);\n  let i = 0;\n  while (i < idx.length) {\n    let j = i;\n    while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;\n    const avg = (i + j) / 2 + 1;\n    for (let k = i; k <= j; k++) rank[idx[k]] = avg;\n    i = j + 1;\n  }\n  let P = 0, N = 0, sumR = 0;\n  labels.forEach((l, k) => { if (l) { P++; sumR += rank[k]; } else N++; });\n  if (!P || !N) return 0.5;\n  return (sumR - P * (P + 1) / 2) / (P * N);\n}',
          tests: [
            {
              name: 'perfect separation gives 1.0',
              run: fn => { const v = fn([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]); return Math.abs(v - 1) < 1e-9 || 'got ' + v; }
            },
            {
              name: 'perfectly reversed gives 0.0',
              run: fn => { const v = fn([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]); return Math.abs(v) < 1e-9 || 'got ' + v; }
            },
            {
              name: 'a known interleaved case gives 0.75',
              run: fn => { const v = fn([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0]); return Math.abs(v - 0.75) < 1e-9 || 'got ' + v; }
            },
            {
              name: 'all-tied scores give 0.5',
              run: fn => { const v = fn([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]); return Math.abs(v - 0.5) < 1e-9 || 'got ' + v + ' — ties must count as half'; }
            },
            {
              name: 'a larger random case matches the brute-force pair count',
              run: fn => {
                const R = Num.rng(4);
                const s = [], l = [];
                for (let i = 0; i < 60; i++) { l.push(R() < .4 ? 1 : 0); s.push(R() + (l[i] ? .3 : 0)); }
                let c = 0, t = 0;
                for (let i = 0; i < 60; i++) for (let j = 0; j < 60; j++) {
                  if (l[i] === 1 && l[j] === 0) { t++; c += s[i] > s[j] ? 1 : (s[i] === s[j] ? .5 : 0); }
                }
                const want = c / t, got = fn(s, l);
                return Math.abs(got - want) < 1e-9 || 'got ' + got.toFixed(6) + ', expected ' + want.toFixed(6);
              }
            }
          ]
        });
      },

      d3: function (host) {
        Labs.codeDrill(host, {
          brief: 'Return the indices kept by non-maximum suppression. Boxes are [x1, y1, x2, y2]; suppress any box whose IoU with an already-kept, higher-scoring box exceeds the threshold.',
          signature: 'function nms(boxes: number[][], scores: number[], thr: number): number[]',
          starter: 'function (boxes, scores, thr) {\n  // your code here — you will need an IoU helper\n}',
          hint: 'Sort indices by score descending. Walk them; keep a box if it does not overlap any already-kept box above the threshold. IoU = intersection / (areaA + areaB − intersection).',
          solution: 'function (boxes, scores, thr) {\n  function iou(a, b) {\n    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);\n    const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);\n    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);\n    const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;\n    return u > 0 ? inter / u : 0;\n  }\n  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);\n  const keep = [];\n  for (const i of order) {\n    let ok = true;\n    for (const k of keep) if (iou(boxes[i], boxes[k]) > thr) { ok = false; break; }\n    if (ok) keep.push(i);\n  }\n  return keep;\n}',
          tests: [
            {
              name: 'keeps the highest-scoring of two heavily overlapping boxes',
              run: fn => {
                const k = fn([[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8], 0.5);
                return (k.length === 1 && k[0] === 0) || 'kept ' + JSON.stringify(k);
              }
            },
            {
              name: 'keeps both when they do not overlap',
              run: fn => {
                const k = fn([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8], 0.5);
                return k.length === 2 || 'kept ' + JSON.stringify(k);
              }
            },
            {
              name: 'respects the threshold: IoU 0.33 survives thr = 0.5',
              run: fn => {
                // two 10x10 boxes offset by 5 in x: inter 50, union 150, IoU = 1/3
                const k = fn([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], 0.5);
                return k.length === 2 || 'kept ' + JSON.stringify(k) + ' — IoU here is 0.333, below the threshold';
              }
            },
            {
              name: 'and the same pair is suppressed at thr = 0.3',
              run: fn => {
                const k = fn([[0, 0, 10, 10], [5, 0, 15, 10]], [0.9, 0.8], 0.3);
                return k.length === 1 || 'kept ' + JSON.stringify(k);
              }
            },
            {
              name: 'returns indices in descending score order',
              run: fn => {
                const k = fn([[0, 0, 4, 4], [100, 100, 104, 104], [200, 200, 204, 204]], [0.2, 0.9, 0.5], 0.5);
                return JSON.stringify(k) === JSON.stringify([1, 2, 0]) || 'got ' + JSON.stringify(k);
              }
            },
            {
              name: 'handles an empty input',
              run: fn => { const k = fn([], [], 0.5); return (k && k.length === 0) || 'should return an empty array'; }
            }
          ]
        });
      },

      d4: function (host) {
        Labs.codeDrill(host, {
          brief: 'One Lloyd iteration: assign every point to its nearest centre, then move each centre to the mean of its members. Return the new centres. If a cluster is empty, leave that centre where it was.',
          signature: 'function step(points: number[][], centers: number[][]): number[][]',
          starter: 'function (points, centers) {\n  // your code here\n}',
          hint: 'Two loops. First build an assignment array by nearest squared distance; then average the members of each cluster. Guard the empty-cluster case or you will produce NaN.',
          solution: 'function (points, centers) {\n  const k = centers.length, d = centers[0].length;\n  const sums = centers.map(() => new Array(d).fill(0));\n  const counts = new Array(k).fill(0);\n  for (const p of points) {\n    let best = 0, bv = Infinity;\n    for (let c = 0; c < k; c++) {\n      let s = 0;\n      for (let j = 0; j < d; j++) s += (p[j] - centers[c][j]) ** 2;\n      if (s < bv) { bv = s; best = c; }\n    }\n    counts[best]++;\n    for (let j = 0; j < d; j++) sums[best][j] += p[j];\n  }\n  return centers.map((c, i) =>\n    counts[i] === 0 ? c.slice() : sums[i].map(s => s / counts[i]));\n}',
          tests: [
            {
              name: 'two obvious clusters converge in one step',
              run: fn => {
                const out = fn([[0, 0], [0, 1], [10, 10], [10, 11]], [[0, 0], [10, 10]]);
                return (Math.abs(out[0][1] - 0.5) < 1e-9 && Math.abs(out[1][1] - 10.5) < 1e-9)
                  || 'got ' + JSON.stringify(out);
              }
            },
            {
              name: 'an empty cluster keeps its centre (no NaN)',
              run: fn => {
                const out = fn([[0, 0], [0, 1]], [[0, 0], [100, 100]]);
                return out[1].every(v => isFinite(v)) || 'produced ' + JSON.stringify(out[1]) + ' — guard the empty case';
              }
            },
            {
              name: 'a single centre becomes the global mean',
              run: fn => {
                const out = fn([[1, 2], [3, 4], [5, 6]], [[0, 0]]);
                return (Math.abs(out[0][0] - 3) < 1e-9 && Math.abs(out[0][1] - 4) < 1e-9) || 'got ' + JSON.stringify(out[0]);
              }
            },
            {
              name: 'works in three dimensions',
              run: fn => {
                const out = fn([[0, 0, 0], [2, 2, 2]], [[0, 0, 0]]);
                return out[0].every(v => Math.abs(v - 1) < 1e-9) || 'got ' + JSON.stringify(out[0]);
              }
            },
            {
              name: 'does not mutate the input centres',
              run: fn => {
                const c = [[0, 0], [10, 10]];
                fn([[0, 0], [10, 10]], c);
                return (c[0][0] === 0 && c[1][0] === 10) || 'the input centres were mutated';
              }
            }
          ]
        });
      },

      d5: function (host) {
        Labs.codeDrill(host, {
          brief: 'Single-head scaled dot-product attention. Q is (n × d), K is (m × d), V is (m × dv). Return the (n × dv) output. Apply the 1/√d scaling.',
          signature: 'function attention(Q: number[][], K: number[][], V: number[][]): number[][]',
          starter: 'function (Q, K, V) {\n  // your code here\n}',
          hint: 'scores[i][j] = dot(Q[i], K[j]) / sqrt(d); softmax each row (stably!); then output[i] = sum_j w[i][j] * V[j].',
          solution: 'function (Q, K, V) {\n  const d = Q[0].length, dv = V[0].length, s = Math.sqrt(d);\n  return Q.map(q => {\n    const sc = K.map(k => {\n      let t = 0;\n      for (let j = 0; j < d; j++) t += q[j] * k[j];\n      return t / s;\n    });\n    const m = Math.max.apply(null, sc);\n    const e = sc.map(v => Math.exp(v - m));\n    const z = e.reduce((a, b) => a + b, 0);\n    const w = e.map(v => v / z);\n    const out = new Array(dv).fill(0);\n    w.forEach((wi, j) => { for (let c = 0; c < dv; c++) out[c] += wi * V[j][c]; });\n    return out;\n  });\n}',
          tests: [
            {
              name: 'output has shape (n × dv)',
              run: fn => {
                const o = fn([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1]], [[5, 6, 7], [8, 9, 10]]);
                return (o.length === 3 && o[0].length === 3) || 'got shape ' + o.length + ' × ' + (o[0] ? o[0].length : '?');
              }
            },
            {
              name: 'uniform scores give the mean of V',
              run: fn => {
                const o = fn([[0, 0]], [[1, 0], [0, 1]], [[0, 0], [10, 20]]);
                return (Math.abs(o[0][0] - 5) < 1e-9 && Math.abs(o[0][1] - 10) < 1e-9) || 'got ' + JSON.stringify(o[0]);
              }
            },
            {
              name: 'attention weights sum to 1 (output stays inside the convex hull of V)',
              run: fn => {
                const o = fn([[3, 1]], [[1, 0], [0, 1]], [[0, 0], [1, 1]]);
                return (o[0][0] > 0 && o[0][0] < 1) || 'got ' + JSON.stringify(o[0]) + ' — the output must be a convex combination of V rows';
              }
            },
            {
              name: 'the √d scaling is applied',
              run: fn => {
                // d = 4: without scaling the softmax is much sharper
                const Q = [[2, 2, 2, 2]], K = [[1, 1, 1, 1], [0, 0, 0, 0]], V = [[1], [0]];
                const o = fn(Q, K, V)[0][0];
                const want = 1 / (1 + Math.exp(-(8 / 2)));   // scores 8/2=4 and 0
                return Math.abs(o - want) < 1e-6 || 'got ' + o.toFixed(6) + ', expected ' + want.toFixed(6) + ' (did you divide by √d?)';
              }
            },
            {
              name: 'does not overflow with large scores',
              run: fn => {
                const o = fn([[500, 500]], [[500, 500], [0, 0]], [[1], [0]]);
                return isFinite(o[0][0]) || 'produced ' + o[0][0] + ' — subtract the row max before exponentiating';
              }
            }
          ]
        });
      },

      d6: function (host) {
        Labs.codeDrill(host, {
          brief: 'Fit y ≈ Xw + b by gradient descent on mean squared error. Return {w, b} after the given number of steps. X is (n × d), y is length n.',
          signature: 'function fit(X: number[][], y: number[], lr: number, steps: number): {w: number[], b: number}',
          starter: 'function (X, y, lr, steps) {\n  // your code here\n}',
          hint: 'Gradient of (1/n)Σ(xᵢᵀw + b − yᵢ)² is (2/n)Σ rᵢ xᵢ for w and (2/n)Σ rᵢ for b, where rᵢ is the residual. Start from zeros.',
          solution: 'function (X, y, lr, steps) {\n  const n = X.length, d = X[0].length;\n  let w = new Array(d).fill(0), b = 0;\n  for (let t = 0; t < steps; t++) {\n    const gw = new Array(d).fill(0);\n    let gb = 0;\n    for (let i = 0; i < n; i++) {\n      let p = b;\n      for (let j = 0; j < d; j++) p += w[j] * X[i][j];\n      const r = p - y[i];\n      for (let j = 0; j < d; j++) gw[j] += 2 * r * X[i][j] / n;\n      gb += 2 * r / n;\n    }\n    for (let j = 0; j < d; j++) w[j] -= lr * gw[j];\n    b -= lr * gb;\n  }\n  return { w: w, b: b };\n}',
          tests: [
            {
              name: 'recovers y = 2x + 1',
              run: fn => {
                const X = [[0], [1], [2], [3], [4]], y = [1, 3, 5, 7, 9];
                const r = fn(X, y, 0.05, 3000);
                return (Math.abs(r.w[0] - 2) < 1e-2 && Math.abs(r.b - 1) < 1e-2) || 'got w=' + r.w[0].toFixed(4) + ' b=' + r.b.toFixed(4);
              }
            },
            {
              name: 'recovers a two-feature relationship',
              run: fn => {
                const X = [], y = [];
                for (let i = 0; i < 40; i++) { const a = i / 10, c = (i % 7) / 3; X.push([a, c]); y.push(1.5 * a - 0.8 * c + 0.4); }
                const r = fn(X, y, 0.02, 6000);
                return (Math.abs(r.w[0] - 1.5) < 5e-2 && Math.abs(r.w[1] + 0.8) < 5e-2) || 'got w=' + JSON.stringify(r.w.map(v => +v.toFixed(3)));
              }
            },
            {
              name: 'zero steps returns the initial parameters',
              run: fn => {
                const r = fn([[1], [2]], [1, 2], 0.1, 0);
                return (r.w[0] === 0 && r.b === 0) || 'should start from zeros and not move';
              }
            },
            {
              name: 'the loss decreases monotonically at a small learning rate',
              run: fn => {
                const X = [[0], [1], [2], [3]], y = [1, 3, 5, 7];
                const loss = p => X.reduce((a, x, i) => a + Math.pow(x[0] * p.w[0] + p.b - y[i], 2), 0) / X.length;
                const l10 = loss(fn(X, y, 0.01, 10)), l100 = loss(fn(X, y, 0.01, 100));
                return l100 < l10 || 'loss did not decrease: ' + l10.toFixed(5) + ' → ' + l100.toFixed(5);
              }
            }
          ]
        });
      }
    },
    quiz: [
      {
        q: 'In a coding round, the step most often skipped and most heavily weighted is…',
        options: ['writing comments', 'stating the complexity and testing your own code out loud', 'using the optimal data structure', 'finishing quickly'],
        answer: 1,
        why: 'Both are cheap and both are explicitly on the rubric. Finding your own bug is a positive signal.'
      },
      {
        q: 'The stable softmax subtracts the max because…',
        options: ['it makes the result sum to 1', '$e^z$ overflows for $z \\gtrsim 709$, and the shift leaves the ratio unchanged', 'it centres the distribution', 'it is faster'],
        answer: 1,
        why: 'Softmax already sums to 1 without the shift. The shift is purely about representable range.'
      },
      {
        q: 'Naive NMS over $n$ boxes costs…',
        options: ['$O(n)$', '$O(n\\log n)$', '$O(n^2)$ in the worst case', '$O(n^3)$'],
        answer: 2,
        why: 'Every kept box is compared against every remaining one. Spatial indexing or a GPU kernel is what production detectors use.'
      },
      {
        q: 'Splitting train/test randomly on a dataset with many rows per customer causes…',
        options: ['class imbalance', 'group leakage — the same customer appears in both sets', 'temporal leakage', 'nothing, if you stratify'],
        answer: 1,
        why: 'The model memorises the entity. Split on unique IDs, then select rows by membership.'
      }
    ],
    cards: [
      { q: 'The coding-round ritual', a: 'Clarify → approach → complexity → write while narrating → test out loud, including an edge case.' },
      { q: 'AUC in one line of maths', a: '(sum of positive ranks − P(P+1)/2) / (P·N), using average ranks for ties.' },
      { q: 'IoU', a: 'intersection / (areaA + areaB − intersection), with the intersection clamped at zero.' },
      { q: 'Welford’s use case', a: 'Mean and variance in one pass with constant memory, stable where E[X²]−E[X]² is not.' },
      { q: 'Attention in four lines', a: 'scores = QKᵀ/√d → row-wise stable softmax → weights × V. Causal masking adds −∞ above the diagonal, before the softmax.' }
    ]
  });

  /* ------------------------------------------------------------------ 7.6 */
  ML.section({
    id: 'case-round', track: 'interview', num: '7.6', level: 2,
    title: 'The case round: debugging, metric design, and estimation',
    lede: 'A conversation, not a quiz. You are handed an ambiguous situation and watched: do you ask for numbers, do you form hypotheses in a sensible order, and do you notice when the answer is that the metric was wrong all along?',
    prereq: ['production'],
    related: ['production', 'experimentation', 'metrics'],
    html: `
${H.tldr([
      'Debugging cases are scored on <b>the order of your hypotheses</b>, not on getting the answer. Cheapest and most likely first; state what you would check and what each outcome would rule out.',
      'Metric-design cases are scored on whether you can name what the metric would <b>reward perversely</b>, and pair it with a guardrail.',
      'Estimation cases are scored on decomposition and on stating assumptions aloud. The number does not matter; the structure does.'
    ])}

<h2><span class="sn">7.6.1</span> The debugging case</h2>
${H.lab('debug', 'A live debugging case', 'The setup is real: a fraud model’s precision fell overnight. Choose what to investigate. Each action costs time and returns real information; the panel tracks what you have ruled out. There is a correct root cause and several plausible wrong turns.')}

${H.steps([
      '<b>Establish the facts before the hypotheses.</b> When exactly? How much? Which metric, measured how? Is the metric itself broken? A surprising number of "model incidents" are dashboard incidents.',
      '<b>Ask what changed.</b> Deploys, upstream schema changes, a marketing campaign, a holiday, a new integration. <b>Correlate with the timestamp</b> — a sharp step points at a discrete change, a gradual slope at drift (§3.12).',
      '<b>Split the funnel.</b> Is it the input, the model, or the action? Check input distributions, then the score distribution, then the threshold or downstream logic. Each is independently checkable.',
      '<b>Segment.</b> Is it everywhere or in one country, one platform, one customer tier? A localised failure narrows the search enormously and often names the cause outright.',
      '<b>Only then consider the model itself.</b> Genuine concept drift is real but slow; overnight changes are nearly always data or deployment.'
    ])}
${H.key('The ordering above is the answer to the question being asked. An interviewer wants to see you check the cheap, likely, discrete causes before reaching for "the world changed" — and to hear you say what each check would rule <i>out</i>.')}

${H.iq('Debugging cases, with the reasoning that scores', [
      {
        q: 'Your recommendation model’s CTR dropped 15% overnight. Walk me through it.',
        level: 'senior',
        a: `<p><b>Facts first.</b> Which CTR — per impression or per session? Measured where? Did impressions also change, or only clicks? A 15% CTR drop with a 20% impression rise is a mix change, not a model failure.</p>
<p><b>What changed.</b> Any deploy in the window — including upstream services and the front end? A tracking change that stopped logging some clicks looks exactly like a model regression and is far more common.</p>
<p><b>Split the funnel.</b> Input feature distributions (PSI), then the score distribution, then the served slate. If scores are unchanged but CTR fell, the model is fine and something downstream changed.</p>
<p><b>Segment.</b> By platform, country, new vs returning. A drop confined to iOS is a client release; confined to new users is a cold-start path; everywhere and uniform suggests something global like a tracking change.</p>
<p><b>Then the model.</b> Compare against the permanent holdback (§5.12) — if the holdback fell too, it is not the model at all.</p>`,
        follow: ['What if the holdback also dropped?', 'How would you distinguish a tracking bug from a real drop?', 'What is your rollback criterion?'],
        red: 'Jumping straight to "retrain on fresh data". It is slow, it is expensive, and it fixes nothing if the cause is a logging change.'
      },
      {
        q: 'Precision is stable but recall halved. What kinds of cause fit that pattern specifically?',
        level: 'senior',
        a: `<p>The pattern is informative: whatever is alerted is still right, but far less is being alerted. That points away from the model’s ranking quality and toward volume.</p>
<ul>
<li><b>A threshold change</b> — someone raised it, or a recalibration shifted the score distribution while preserving the ranking.</li>
<li><b>A capacity cap</b> — an alert budget or rate limit truncating the queue.</li>
<li><b>The positive class itself changed</b> — a new fraud pattern the model has never seen. Precision on the old pattern stays fine; the new one is invisible.</li>
<li><b>Upstream filtering</b> — a rule layer or an eligibility check now removing candidates before the model sees them.</li>
</ul>
<p>Distinguish them by looking at the <i>score distribution of the missed positives</i>: clustered just below the threshold means a threshold or calibration issue; spread across the low range means the model genuinely cannot see them.</p>`,
        follow: ['How would you confirm the new-pattern hypothesis?', 'What would you ship first while investigating?']
      },
      {
        q: 'Offline AUC improved 3 points; online conversion is flat. Where do you look?',
        level: 'core',
        a: `<p>Five hypotheses, cheapest first (§7.2): underpowered test, offline/online metric mismatch, distribution mismatch between the offline set and live traffic, implementation skew, and system effects such as added latency.</p>
<p>The discriminating checks: compute the online metric on the offline set (does the improvement survive the change of metric?); score a sample of live requests through the offline pipeline and compare feature-by-feature (skew); check the experiment’s confidence interval rather than its p-value (power).</p>
<p><b>And consider that the offline improvement may be real and irrelevant</b> — a better ranking of items nobody scrolls to changes AUC and changes nothing else.</p>`,
        follow: ['How would you have caught this before launching?', 'What offline metric would have predicted the online result better?']
      }
    ])}

<h2><span class="sn">7.6.2</span> Metric design</h2>
${H.table(['Prompt', 'The naive answer', 'What it rewards perversely', 'The pairing that fixes it'], [
      ['Measure a search engine', 'click-through rate', 'clickbait; a bad result clicked and abandoned counts as a success', 'CTR + dwell time + query reformulation rate'],
      ['Measure a feed', 'time spent', 'outrage and doom-scrolling; the metric rises as the product worsens', 'time spent + next-week retention + a user-reported satisfaction survey'],
      ['Measure a support bot', 'containment (no human handoff)', 'refusing to escalate genuine problems', 'containment + <b>resolution</b> + repeat-contact rate'],
      ['Measure a code assistant', 'suggestions accepted', 'short trivial completions; accepted code that is later deleted', 'acceptance + code retained after 7 days + test pass rate'],
      ['Measure a fraud model', 'fraud caught', 'blocking everything', 'fraud loss in currency + false-decline rate'],
      ['Measure a summariser', 'ROUGE against a reference', 'copying the lead paragraph', 'human preference + factual consistency checks (§5.11)']
    ])}
${H.key('The reliable move in a metric-design case: propose the metric, then immediately say <i>"and here is how I would game it"</i>, then add the guardrail that prevents it. That sequence demonstrates the thing being tested — that you know a metric is an incentive, not a measurement.')}
${H.checklist([
      'One <b>primary</b> metric. Not three; you cannot optimise three.',
      'Two or three <b>guardrails</b> that must not degrade — latency, cost, a fairness slice, a retention proxy.',
      'A stated <b>time horizon</b>: engagement measured over a day and over a month can point in opposite directions.',
      'A <b>sensitivity</b> check: is the metric movable at all at your traffic level (§2.25)?',
      'An explicit statement of who is <b>harmed</b> if the metric goes up and the product gets worse.'
    ])}

<h2><span class="sn">7.6.3</span> Estimation</h2>
${H.worked('"How many GPUs to serve ChatGPT-scale traffic?" — the structure, not the answer', `
<p>Decompose, state assumptions aloud, and keep the arithmetic round.</p>
<ol>
<li><b>Users.</b> Say 200M weekly actives, 40% daily → 80M DAU. Assume 5 messages/day → 400M messages/day.</li>
<li><b>Peak.</b> 400M/86,400 ≈ 4,600 messages/second average; peak 3× ≈ <b>14,000/s</b>.</li>
<li><b>Tokens.</b> Say 500 output tokens each → 7M output tokens/second at peak.</li>
<li><b>Per-GPU throughput.</b> A served mid-size model with batching: order 2,000–5,000 output tokens/second per card. Take 3,000.</li>
<li><b>GPUs.</b> 7M / 3,000 ≈ <b>2,300 cards</b> for decode at peak, plus prefill, plus redundancy and regional spread — call it 5,000–10,000.</li>
</ol>
<p>Then <b>state the two assumptions that dominate</b>: tokens per message and per-GPU throughput. A factor of two in either moves the answer by a factor of two, and saying so is the part that scores. Nobody is checking your number against reality; they are checking whether you know which of your assumptions is load-bearing.</p>`)}
${H.table(['Estimation prompt', 'The decomposition that works'], [
      ['Cost of running this feature for a year', 'requests/day × tokens/request × $/token × 365, then add the retrieval and storage line'],
      ['How much training data do we need?', 'from a learning curve: fit accuracy vs log(n) on what you have, extrapolate to the target, state the extrapolation risk'],
      ['How long to label 100k examples?', 'items/hour/annotator × annotators × agreement overhead (double-labelling a sample) + adjudication'],
      ['Storage for a year of logs', 'events/day × bytes/event × 365 × replication factor, then the retention policy that makes it affordable']
    ])}

${H.probe([
      ['A metric moved. What do you check first?', 'Whether the metric is broken — instrumentation, logging, the dashboard’s own query. Then what changed at that timestamp. Model degradation is rarely the fastest explanation for an overnight step.'],
      ['You are asked to design a metric for a summarisation feature. What is your first move?', 'Ask what decision it informs. Then propose the metric, say how I would game it, and pair it with a guardrail — that sequence is the answer being looked for.'],
      ['How do you approach an estimation question?', 'Decompose into factors I can defend, state every assumption out loud, keep the arithmetic round, and finish by naming which assumption the answer is most sensitive to.']
    ])}`,
    labs: {
      debug: function (host) {
        const el = ML.el;
        /* Root cause: an upstream provider changed a categorical encoding, so a
           high-value feature became mostly unknown for one country. */
        const ACTIONS = [
          { k: 'confirm', label: 'Confirm the metric is real (check the dashboard query and the logging path)', cost: 1,
            info: 'The drop is real: precision fell from 61% to 38% between 02:00 and 03:00 UTC. Alert volume is unchanged. The dashboard query has not changed.',
            value: 3, tag: 'facts' },
          { k: 'deploys', label: 'Check for deploys in the window', cost: 1,
            info: 'No deploys to the model service in nine days. One upstream change: the device-intelligence vendor released a new API version at 02:00 UTC.',
            value: 5, tag: 'facts', key: true },
          { k: 'segment', label: 'Segment the drop by country, platform and customer tier', cost: 2,
            info: 'Precision is normal everywhere except Brazil, where it fell from 59% to 11%. Brazil is 22% of volume.',
            value: 5, tag: 'localise', key: true },
          { k: 'psi', label: 'Compute input drift (PSI) per feature', cost: 2,
            info: 'One feature has PSI 0.94: device_risk_band. Its "unknown" category went from 3% to 71% of rows. Everything else is under 0.05.',
            value: 5, tag: 'localise', key: true },
          { k: 'scores', label: 'Compare the score distribution before and after', cost: 1,
            info: 'The score distribution shifted downward and compressed — fewer confident predictions, more mass near the threshold. Consistent with a feature going missing.',
            value: 3, tag: 'localise' },
          { k: 'holdback', label: 'Check the permanent holdback arm', cost: 1,
            info: 'The holdback (rules only) is unaffected. So this is specific to the model path, not a change in the underlying fraud rate.',
            value: 4, tag: 'facts' },
          { k: 'schema', label: 'Diff the upstream payload schema against last week', cost: 2,
            info: 'The vendor renamed the band values from "low"/"med"/"high" to "L1"/"L2"/"L3". Our mapping does not recognise them, so they fall through to "unknown".',
            value: 6, tag: 'root', root: true },
          { k: 'retrain', label: 'Retrain the model on the last 30 days', cost: 8,
            info: 'Six hours later: the retrained model is slightly worse, because the last 30 days include the corrupted feature. This did not help.',
            value: -3, tag: 'wrong' },
          { k: 'threshold', label: 'Lower the decision threshold to restore alert volume', cost: 2,
            info: 'Alert volume is restored but precision falls further, to 29%. You have treated a symptom and made it worse.',
            value: -2, tag: 'wrong' },
          { k: 'labels', label: 'Audit label quality with the review team', cost: 4,
            info: 'Labels look normal; inter-reviewer agreement is unchanged at κ = 0.79. Not the cause.',
            value: 0, tag: 'neutral' },
          { k: 'concept', label: 'Investigate whether fraud patterns have changed', cost: 4,
            info: 'Confirmed fraud in the period looks like the same patterns as last month. Concept drift does not happen between 02:00 and 03:00.',
            value: 1, tag: 'neutral' }
        ];
        let taken = [], time = 0, solved = false;
        const log = el('div');
        const opts = el('div');
        host.appendChild(log);
        host.appendChild(opts);
        const out = Viz.readout(host, [
          { k: 'time', label: 'hours spent', cls: 'warn' },
          { k: 'found', label: 'root cause', cls: 'key' },
          { k: 'score', label: 'investigation quality', cls: 'good' }
        ]);

        function paint() {
          log.innerHTML =
            '<div class="box" style="margin-top:0"><p class="boxtitle" data-icon="!">the page at 08:40</p>' +
            '<p style="margin:0">Overnight, the fraud model’s precision on reviewed alerts fell from about 61% to about 38%. ' +
            'Alert volume is unchanged. Nobody deployed the model. The review team noticed at 08:00 and escalated. ' +
            '<b>You have the on-call pager. What do you do?</b></p></div>' +
            (taken.length
              ? '<div style="border-left:2px solid var(--line);padding-left:14px;margin:14px 0">' +
                taken.map(k => {
                  const a = ACTIONS.find(x => x.k === k);
                  return '<p style="margin:0 0 10px"><b style="font-family:var(--sans);font-size:13.5px">' + a.label + '</b> ' +
                    '<span class="small" style="color:var(--faint)">(' + a.cost + ' h)</span><br>' +
                    '<span style="font-family:var(--sans);font-size:14px;color:' +
                    (a.root ? 'var(--green)' : a.value < 0 ? 'var(--red)' : 'var(--muted)') + '">' + a.info + '</span></p>';
                }).join('') + '</div>'
              : '');
          opts.innerHTML = '';
          if (solved) {
            opts.innerHTML =
              '<div class="box intuition"><p class="boxtitle" data-icon="◎">root cause found</p>' +
              '<p>The vendor renamed its category values; our mapping silently bucketed everything unfamiliar into "unknown", which destroyed the most predictive feature for 22% of traffic.</p>' +
              '<p><b>The fix, in order:</b> (1) mitigate now — route Brazil to the rules-only path, restoring known precision within minutes; (2) fix the mapping and backfill; (3) prevent recurrence — <b>schema validation that fails loudly rather than defaulting to "unknown"</b>, and a PSI alert on the unknown-rate of every categorical feature.</p>' +
              '<p class="small">The general lesson: a silent default is worse than a crash. A pipeline that fails closed would have paged at 02:05 instead of a human noticing at 08:00.</p></div>';
            const row = el('div', { class: 'btnrow' });
            row.appendChild(el('button', { class: 'btn', type: 'button', text: 'Start over', onclick: () => { taken = []; time = 0; solved = false; paint(); } }));
            opts.appendChild(row);
          } else {
            ACTIONS.filter(a => taken.indexOf(a.k) < 0).forEach(a => {
              const b = el('button', { class: 'qopt', type: 'button' }, [
                el('span', { class: 'k', text: a.cost + 'h' }),
                el('span', { text: a.label })
              ]);
              b.addEventListener('click', () => {
                taken.push(a.k); time += a.cost;
                if (a.root) solved = true;
                paint();
              });
              opts.appendChild(b);
            });
          }
          const score = taken.reduce((s, k) => s + ACTIONS.find(x => x.k === k).value, 0);
          out({
            time: time + ' h',
            found: solved ? '✓ found' : 'not yet',
            score: score + (solved && time <= 8 ? ' — efficient' : solved ? ' — found it, slowly' : '')
          });
        }
        paint();
        Viz.note(host, 'The efficient path is four actions and about six hours: confirm the metric, check what changed, segment, compute input drift — at which point the schema diff is obvious. The tempting wrong turns are <b>retrain</b> and <b>lower the threshold</b>: both are expensive, both feel like doing something, and both make the situation worse. In a real interview you will be scored on the <i>order</i> you propose these, so say why each one is next before you take it.');
      }
    },
    quiz: [
      {
        q: 'A metric dropped sharply overnight. The least likely explanation is…',
        options: ['an upstream schema change', 'a deployment', 'genuine concept drift', 'a logging or instrumentation change'],
        answer: 2,
        why: 'Concept drift is real but gradual. Overnight step changes are nearly always data or deployment events.'
      },
      {
        q: 'In a metric-design case, the highest-scoring move after proposing a metric is…',
        options: ['computing its confidence interval', 'saying how you would game it, then adding the guardrail that prevents it', 'listing alternatives', 'estimating the sample size'],
        answer: 1,
        why: 'It demonstrates that you treat a metric as an incentive rather than a measurement — which is what the case is testing.'
      },
      {
        q: 'Precision stable, recall halved. This pattern most suggests…',
        options: ['the model overfitting', 'fewer things being alerted — a threshold, a capacity cap, upstream filtering, or a new unseen positive pattern', 'label noise', 'a calibration problem only'],
        answer: 1,
        why: 'What is alerted is still right, so ranking quality is intact. The volume of positives reaching the alert is what changed.'
      },
      {
        q: 'In an estimation question, the thing that actually scores is…',
        options: ['landing near the true number', 'the decomposition, the stated assumptions, and naming which assumption dominates', 'speed', 'using exact figures'],
        answer: 1,
        why: 'Nobody checks your number. They check whether you know which factor the answer is most sensitive to.'
      }
    ],
    cards: [
      { q: 'Debugging order', a: 'Is the metric real → what changed at that timestamp → split the funnel → segment → only then the model.' },
      { q: 'Metric design in three moves', a: 'Propose it → say how you would game it → add the guardrail. One primary metric, two or three guardrails.' },
      { q: 'Precision stable, recall halved', a: 'Volume of alerts, not ranking quality: threshold, capacity cap, upstream filter, or a new unseen pattern.' },
      { q: 'Estimation structure', a: 'Decompose → state assumptions aloud → round arithmetic → name the dominant assumption.' }
    ]
  });

  /* ------------------------------------------------------------------ 7.7 */
  ML.section({
    id: 'behavioural', track: 'interview', num: '7.7', level: 1,
    title: 'The behavioural round, the questions to ask, and a six-week plan',
    lede: 'The round people prepare for last and lose on most often. It is not a personality test — it is a structured check on whether you own outcomes, work with people who disagree with you, and can describe a failure honestly.',
    related: ['interview-map', 'interview-breadth', 'ml-system-design'],
    html: `
${H.tldr([
      'Use <b>STAR with a number</b>: Situation, Task, Action, Result — and the Result must be quantified. "Improved things" is not a result.',
      'Prepare <b>four stories</b>, not twenty: a hard technical problem, a conflict, a failure you owned, and a time you changed your mind. Almost every behavioural question maps to one of these.',
      'The questions <i>you</i> ask are scored. Ask about how decisions get made and what failure looks like on the team — not about the technology stack, which you can read.'
    ])}

<h2><span class="sn">7.7.1</span> STAR, adapted for machine learning</h2>
${H.table(['Element', 'Generic advice', 'What it means for an ML story'], [
      ['<b>Situation</b>', 'set the context', 'the business problem and why it mattered — <b>not</b> the dataset'],
      ['<b>Task</b>', 'your responsibility', 'be precise about what was yours versus your team’s; interviewers probe this'],
      ['<b>Action</b>', 'what you did', 'the decisions and the trade-offs, not a chronology of the notebook'],
      ['<b>Result</b>', 'the outcome', '<b>a number with a unit</b>, plus how you knew it was real (§2.25)']
    ])}
${H.key('The result is where most answers collapse. "We improved the model" is unscoreable. "Precision at the fixed alert budget went from 31% to 44%, verified in a two-week A/B test, which removed about 900 manual reviews a week" is a hire signal, and it is the same project described properly.')}

${H.iq('The behavioural questions, and what a strong answer contains', [
      {
        q: 'Tell me about a time a model of yours failed in production.',
        level: 'all levels',
        a: `<p>The single most important story to have ready, and the one most people avoid preparing because it is uncomfortable.</p>
<p>Structure: what broke, <b>how you found out</b> (this is the part that reveals your monitoring maturity), what the immediate mitigation was, what the root cause turned out to be, and — the part that scores highest — <b>what you changed so that class of failure could not recur</b>.</p>
<p>Own it in the first person. "I had not thought about the label lag" is far stronger than "the requirements were unclear". Interviewers are not looking for someone who has never failed; they are looking for someone whose failures produced systems.</p>`,
        follow: ['How long until you noticed?', 'What would have caught it sooner?', 'Did it happen again?'],
        red: 'Choosing a failure that was not your fault, or one so small it is obviously a dodge. Both read as evasion.'
      },
      {
        q: 'Tell me about a time you disagreed with a colleague or a manager.',
        level: 'all levels',
        a: `<p>Pick a genuine technical disagreement with a real resolution. The structure that works: what each side wanted, <b>what evidence you gathered</b> to move the argument off opinion, what was decided, and — importantly — whether you were right.</p>
<p>"I was wrong, and here is what changed my mind" is a strong answer, not a weak one. So is "we ran a small experiment to settle it", because it shows you convert disagreement into evidence rather than into escalation.</p>`,
        follow: ['What would you do if the decision had gone the other way?', 'How did the relationship end up?'],
        red: 'A story where you were right, they were unreasonable, and nothing was learned. It reads as someone difficult to work with.'
      },
      {
        q: 'Tell me about the most technically difficult thing you have built.',
        level: 'mid+',
        a: `<p>Choose depth over impressiveness — a hard problem you understand completely beats a glamorous one you touched. Be explicit about <b>what made it hard</b>: was it the scale, the ambiguity, the data quality, the latency budget, the coordination?</p>
<p>Expect the interviewer to drill until they reach the edge of your knowledge. That is the point of the question. When they get there, say so plainly and reason forward from what you do know — the depth at which you stop being confident <i>is</i> the measurement.</p>`,
        follow: ['Why that approach rather than X?', 'What would you do differently now?', 'What was the hardest bug?']
      },
      {
        q: 'Tell me about a time you had to ship something you were not happy with.',
        level: 'senior',
        a: `<p>Tests judgement about trade-offs under constraints, which is most of a senior job. Good answers name the constraint honestly (deadline, data, headcount), the thing you gave up, <b>the risk you explicitly accepted and communicated</b>, and how you bounded it — a kill switch, a limited rollout, a monitoring alert, a scheduled follow-up that actually happened.</p>
<p>The failure mode is claiming you would never ship something imperfect. That reads as either inexperience or inflexibility, and neither is what the question is fishing for.</p>`,
        follow: ['Did you go back and fix it?', 'How did you communicate the risk, and to whom?']
      },
      {
        q: 'Why are you leaving your current role?',
        level: 'all levels',
        a: `<p>Forward-looking and specific. "I want to work on systems at a scale where the serving constraints actually bind, and this team does that" is credible. Vague positivity is not, and criticism of your current employer — even deserved — reliably costs you.</p>
<p>If the real reason is negative (a reorg, a bad manager, a cancelled project), state it once, neutrally and factually, and move immediately to what you are looking for instead. Interviewers respect brevity here far more than spin.</p>`,
        follow: ['What would have made you stay?', 'What are you looking for that you cannot get there?']
      },
      {
        q: 'Where do you see yourself in three years?',
        level: 'all levels',
        a: `<p>Answer honestly about the <i>kind</i> of work, not the title. "Owning a system end to end and mentoring people into it" or "going deep on inference efficiency" are both fine. A title-only answer invites the follow-up "why?" that you then have to improvise.</p>
<p>Connect it to the role in front of you. This is really a question about whether you will still be here and engaged in two years.</p>`,
        follow: ['What kind of work energises you most?', 'Management or individual contributor?']
      }
    ])}

<h2><span class="sn">7.7.2</span> The questions to ask them</h2>
${H.table(['Ask', 'What it tells you', 'What a bad answer sounds like'], [
      ['How does a model get from a notebook to production here, concretely?', 'ML maturity, and how much of your time will be platform work', '"It depends" with no example'],
      ['What is the last model that was rolled back, and why?', 'whether monitoring and rollback actually exist', '"That has not happened" — either very good or nobody is looking'],
      ['How are priorities decided, and by whom?', 'whether you will have any agency', 'a long chain ending at someone you will never meet'],
      ['What does someone who is struggling in this role look like at six months?', 'the real expectations, stated concretely', 'a generic answer means the bar has never been written down'],
      ['What fraction of the team’s time goes to new models versus maintenance?', 'the honest shape of the job', '"almost all new work" is rarely true and worth probing'],
      ['How do you evaluate whether an ML project was worth doing?', 'whether the org measures anything', 'shipped-count as the measure of success'],
      ['What is the biggest technical debt the team carries?', 'candour, and what you would inherit', 'claiming there is none']
    ])}
${H.note('Ask two or three, not seven. Ask them of everyone in the loop and compare — divergent answers about how decisions get made are far more informative than any single answer.')}

<h2><span class="sn">7.7.3</span> The offer conversation</h2>
${H.checklist([
      '<b>Do not give a number first</b> if you can avoid it. "I would rather understand the role and the level first — what range is budgeted for this position?" is normal and expected.',
      '<b>Negotiate the level, not just the money.</b> Level determines the band, the scope and the next promotion; a level bump is worth more than any one-off adjustment.',
      '<b>Know the components.</b> Base, bonus, equity (and its vesting schedule and refresh policy), sign-on. Equity at a private company needs a strike price, a preferred price and a share count before it means anything.',
      '<b>Ask for time.</b> A week is standard and asking for it costs nothing.',
      '<b>Be straightforward about competing offers</b>, and never invent one — it is checkable more often than people think, and the downside is total.',
      '<b>Get it in writing</b> before resigning anything.'
    ])}

${H.lab('plan', 'Your preparation plan', 'Set your target role, the weeks available and the hours you can give it. The output is a week-by-week schedule against the actual sections of this site, weighted for the role you named.')}

${H.key('The one habit worth more than any of the above: <b>practise out loud, against a clock, preferably to another person.</b> Everything on this site is knowledge; the interview measures the retrieval of that knowledge under mild social pressure, and those are different skills that improve separately.')}`,
    labs: {
      plan: function (host) {
        const el = ML.el;
        const TRACKS = {
          mle: { name: 'ML engineer', mix: { coding: 3, breadth: 2, depth: 1, design: 3, applied: 2, behav: 1 } },
          ds: { name: 'Data scientist / applied scientist', mix: { coding: 2, breadth: 3, depth: 2, design: 1, applied: 1, behav: 1, stats: 3 } },
          research: { name: 'Research scientist', mix: { coding: 1, breadth: 2, depth: 4, design: 1, applied: 1, behav: 1 } },
          llm: { name: 'AI / LLM engineer', mix: { coding: 2, breadth: 1, depth: 1, design: 2, applied: 4, behav: 1 } },
          senior: { name: 'Senior / staff (any of the above)', mix: { coding: 1, breadth: 2, depth: 2, design: 4, applied: 2, behav: 3 } }
        };
        const BLOCKS = {
          coding: { label: 'Coding drills', items: [['7.5', 'coding-round'], ['0.5', 'python-toolkit']] },
          breadth: { label: 'ML breadth', items: [['7.2', 'interview-breadth'], ['2.26', 'part2-recall'], ['2.13', 'metrics'], ['2.2', 'bias-variance']] },
          depth: { label: 'Derivations', items: [['7.3', 'interview-depth'], ['0.7', 'matrix-calculus'], ['3.2', 'backprop'], ['4.3', 'attention']] },
          design: { label: 'System design', items: [['7.4', 'ml-system-design'], ['5.12', 'mlops'], ['4.14', 'serving'], ['5.1', 'rag']] },
          applied: { label: 'Applied AI', items: [['5.13', 'decision-ladder'], ['5.1', 'rag'], ['5.11', 'evals'], ['4.12', 'post-training'], ['5.3', 'agents']] },
          stats: { label: 'Statistics', items: [['1.6', 'intervals'], ['2.25', 'experimentation'], ['1.7', 'causal']] },
          behav: { label: 'Stories and mocks', items: [['7.7', 'behavioural'], ['7.1', 'interview-map']] }
        };
        const st = Viz.controls(host, [
          { k: 'role', label: 'target role', type: 'select', value: 'mle', options: Object.keys(TRACKS).map(k => ({ v: k, t: TRACKS[k].name })) },
          { k: 'weeks', label: 'weeks available', min: 1, max: 12, step: 1, value: 6, fmt: v => v + (v === 1 ? ' week' : ' weeks') },
          { k: 'hours', label: 'hours per week', min: 2, max: 25, step: 1, value: 8, fmt: v => v + ' h' }
        ], () => draw());
        const box = el('div');
        host.appendChild(box);

        function draw() {
          const mix = TRACKS[st.role].mix;
          const total = Object.keys(mix).reduce((a, k) => a + mix[k], 0);
          const totalHours = st.weeks * st.hours;
          const alloc = Object.keys(mix).map(k => ({
            k: k, label: BLOCKS[k].label, hours: Math.round(totalHours * mix[k] / total), items: BLOCKS[k].items
          })).sort((a, b) => b.hours - a.hours);

          // spread across weeks: heaviest blocks start first, mocks always in the last third
          const weeks = [];
          for (let w = 0; w < st.weeks; w++) weeks.push([]);
          let cursor = 0;
          alloc.forEach(a => {
            if (a.k === 'behav') return;
            let left = a.hours;
            while (left > 0 && cursor < st.weeks) {
              const room = st.hours - weeks[cursor].reduce((s, x) => s + x.h, 0);
              if (room <= 0) { cursor++; continue; }
              const put = Math.min(room, left);
              weeks[cursor].push({ n: a.label, h: put, items: a.items });
              left -= put;
            }
          });
          // stories and mocks in the final third
          const behav = alloc.find(a => a.k === 'behav');
          const startW = Math.max(0, st.weeks - Math.max(1, Math.round(st.weeks / 3)));
          let bleft = behav.hours;
          for (let w = startW; w < st.weeks && bleft > 0; w++) {
            const put = Math.ceil(bleft / (st.weeks - w));
            weeks[w].push({ n: 'Stories + mock interviews', h: put, items: behav.items });
            bleft -= put;
          }

          box.innerHTML =
            '<p class="boxtitle" style="margin-top:12px">' + TRACKS[st.role].name + ' · ' + totalHours + ' hours total</p>' +
            weeks.map((wk, i) =>
              '<div style="border:1px solid var(--line);border-left:3px solid var(--c' + ((i % 6) + 1) + ');border-radius:10px;padding:11px 14px;margin-bottom:8px">' +
              '<p style="margin:0 0 6px;font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)">week ' + (i + 1) + '</p>' +
              (wk.length
                ? wk.map(b => '<p style="margin:0 0 4px;font-family:var(--sans);font-size:13.5px"><b>' + b.n + '</b> — ' + b.h + ' h · ' +
                  b.items.map(s => '<a href="#/' + s[1] + '">§' + s[0] + '</a>').join(', ') + '</p>').join('')
                : '<p class="small" style="margin:0;color:var(--faint)">buffer — use it to redo whatever you failed</p>') +
              '</div>').join('') +
            '<div class="box practice"><p class="boxtitle" data-icon="⚙">the rules that make this work</p>' +
            '<ul style="margin:0"><li><b>Every session ends out loud.</b> Ten minutes answering the drill questions (§7.2) at speaking pace.</li>' +
            '<li><b>Book mock interviews for the final third now</b>, before you feel ready. You will not feel ready, and the ones you fail are worth more than the ones you pass.</li>' +
            '<li><b>Keep a list of every question you fumbled.</b> That list, not this plan, is your real revision material in the last week.</li>' +
            '<li><b>Do not start a new topic in the final week.</b> Consolidate; the marginal new fact is worth much less than fluency on what you already know.</li></ul></div>';
        }
        draw();
        Viz.note(host, 'The weighting is what matters here, not the calendar. An LLM-engineering loop spends four times as long on applied AI as on derivations; a research loop inverts that exactly. The most common preparation mistake is running the same generic plan regardless of the role — and the second most common is leaving the behavioural stories until the night before, when they are the round most reliably improved by an hour of preparation.');
      }
    },
    quiz: [
      {
        q: 'The weakest part of most STAR answers is…',
        options: ['the situation', 'the result — unquantified and unverified', 'the task', 'the length'],
        answer: 1,
        why: '"We improved the model" is unscoreable. A number with a unit, plus how you knew it was real, is the whole difference.'
      },
      {
        q: 'A strong "tell me about a failure" answer centres on…',
        options: ['a failure that was not your fault', 'how you found out, what you fixed, and what you changed so it could not recur', 'a very small failure', 'how you avoided blame'],
        answer: 1,
        why: 'The detection and the systemic fix are the parts that reveal maturity. Deflection reads as evasion.'
      },
      {
        q: 'In an offer conversation, the most pivotal term to negotiate is usually…',
        options: ['sign-on bonus', 'the level, because it sets the band and the scope', 'base salary', 'start date'],
        answer: 1,
        why: 'Level determines the compensation band, the work you get, and how soon the next promotion is plausible.'
      },
      {
        q: 'Four stories are enough for a behavioural round because…',
        options: ['interviewers only ask four questions', 'almost every question maps to hard problem / conflict / owned failure / changed your mind', 'they are easier to memorise', 'longer answers score better'],
        answer: 1,
        why: 'Prepare four deeply and adapt them, rather than twenty shallowly.'
      }
    ],
    cards: [
      { q: 'STAR, adapted', a: 'Situation = the business problem; Task = what was yours specifically; Action = decisions and trade-offs; Result = a number, plus how you knew it was real.' },
      { q: 'The four stories', a: 'A hard technical problem, a conflict, a failure you owned, and a time you changed your mind.' },
      { q: 'The best question to ask them', a: '"What is the last model that was rolled back, and why?" — it reveals whether monitoring and rollback actually exist.' },
      { q: 'Offer negotiation', a: 'Negotiate the level first; know all four components; ask for a week; never invent a competing offer.' },
      { q: 'The habit that beats all the reading', a: 'Practise out loud, timed, to another person. Knowledge and retrieval-under-pressure improve separately.' }
    ]
  });
})();
