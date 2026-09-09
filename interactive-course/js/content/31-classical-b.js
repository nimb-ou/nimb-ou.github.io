/* ============================================================
   PART 2 — Core & classical ML (2.7 – 2.11)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.7 */
  ML.section({
    id: 'trees', track: 'classical', num: '2.7',
    title: 'Trees, bagging, random forests',
    lede: 'Rests on §1.3’s variance algebra and §1.10’s impurity measures; sets up §2.8 by contrast — bagging attacks variance with independent parallel models, boosting attacks bias with dependent sequential ones.',
    html: `
<h2><span class="sn">2.7.1</span> A single tree</h2>
<p>A decision tree recursively splits to reduce impurity — <b>Gini</b> $1-\\sum_k p_k^2$ or <b>entropy</b> $-\\sum_k p_k\\log p_k$ for classification, variance for regression. Trees capture interactions for free, need no scaling, handle mixed types, and are readable. They are also high-variance: change a few rows and the top split can flip, rewriting every path beneath it.</p>

${H.table(['Criterion', 'Formula', 'Character'], [
      ['Gini', '$1-\\sum_k p_k^2$', 'Slightly faster; the sklearn default; prefers the largest class purity'],
      ['Entropy', '$-\\sum_k p_k \\log_2 p_k$', 'Information gain; marginally more balanced splits, rarely different in practice'],
      ['Variance (regression)', '$\\frac1n\\sum (y-\\bar y)^2$', 'The squared-error criterion — this is what boosting’s trees fit'],
      ['MAE (regression)', '$\\frac1n\\sum|y-\\text{median}|$', 'Robust, slower, no closed-form leaf update']
    ])}

${H.lab('tree', 'Grow a tree, watch it partition the plane', 'A real CART implementation. Increase depth and watch the axis-aligned rectangles multiply — and watch training accuracy hit 100% while the shapes become obviously unreasonable. That is variance you can see.')}

<h2><span class="sn">2.7.2</span> Bagging and random forests</h2>
<p><b>Bagging</b> fits trees on bootstrap resamples and averages. From §1.3, averaging $B$ models with pairwise correlation $\\rho$ leaves variance $\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$: more trees only ever help, but the benefit saturates at $\\rho\\sigma^2$. <b>Random forests</b> attack that floor by sampling a random subset of features at every split, which decorrelates the trees and lowers $\\rho$. Out-of-bag samples — roughly 37% of rows are omitted from each bootstrap ($(1-1/n)^n \\to e^{-1}$) — give a free validation estimate.</p>

${H.lab('forest', 'One tree versus a forest, on the same data', 'Both are trained here. Watch the single tree’s boundary jump when you reseed the data, while the forest’s barely moves — that stability <i>is</i> the variance term shrinking.')}

${H.note('Bagging attacks variance with independent parallel models. Boosting (§2.8) attacks bias with dependent sequential ones. Same ingredient, opposite recipe — interviewers love this contrast.')}

<h2><span class="sn">2.7.3</span> Reading a tree aloud</h2>
<p>Leaves hold predictions; edges hold conditions you can read to a committee: <i>utilisation ≤ 0.62 → enquiries ≤ 2 → p = 0.03</i>. That readability is why a single shallow tree still appears in regulated settings as a challenger or a segmentation device, even when the production model is an ensemble.</p>

${H.probe([
      ['Why is a random forest better than plain bagging?', 'Feature subsampling decorrelates the trees, lowering ρ and therefore the variance floor ρσ².'],
      ['What is out-of-bag error?', 'Each bootstrap omits ~37% of rows; predicting those with the trees that did not see them gives a free validation estimate.'],
      ['Gini or entropy?', 'They rarely disagree materially. Say that, then say the depth cap and min-samples-per-leaf matter far more.']
    ], 'Reporting impurity-based feature importance as if it were unbiased — it favours high-cardinality and continuous features (§2.17).')}`,
    labs: {
      tree: function (host) {
        let data = Num.dataset('moons', 90, .35, 15);
        const st = Viz.controls(host, [
          { k: 'depth', label: 'max depth', min: 1, max: 8, step: 1, value: 3, fmt: v => v },
          { k: 'leaf', label: 'min samples per leaf', min: 1, max: 20, step: 1, value: 1, fmt: v => v },
          { k: 'crit', label: 'criterion', type: 'buttons', value: 'gini', options: [{ v: 'gini', t: 'Gini' }, { v: 'entropy', t: 'entropy' }] },
          { k: 'view', label: 'view', type: 'buttons', value: 'space', options: [{ v: 'space', t: 'partition' }, { v: 'tree', t: 'the tree' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'acc', label: 'training accuracy', cls: 'key' }, { k: 'leaves', label: 'leaves' }, { k: 'depth', label: 'depth reached' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const t = Num.tree(data.X, data.y, { maxDepth: st.depth, minLeaf: st.leaf, criterion: st.crit });
            let leaves = 0, maxd = 0;
            (function walk(n, d) { maxd = Math.max(maxd, d); if (!n.left) { leaves++; return; } walk(n.left, d + 1); walk(n.right, d + 1); })(t.root, 0);
            if (st.view === 'space') {
              const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
              P.clip(() => Labs.boundary(P, (x, y) => t.predict([x, y]), { step: 3 }));
              P.clip(() => Labs.points(P, data.X, data.y));
            } else {
              Labs.drawTree(ctx, t.root, w / 2, 16, w, T, ['x₁', 'x₂'], st.depth);
            }
            const acc = Num.mean(data.X.map((x, i) => ((t.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            out({ acc: (acc * 100).toFixed(1) + '%', leaves: leaves, depth: maxd });
          }
        });
        Viz.buttons(host, [
          { label: 'Moons', on: () => { data = Num.dataset('moons', 90, .35, 15); S.redraw(); } },
          { label: 'XOR', on: () => { data = Num.dataset('xor', 90, .3, 5); S.redraw(); } },
          { label: 'Circles', on: () => { data = Num.dataset('circles', 90, .2, 8); S.redraw(); } },
          { label: 'Reseed', on: () => { data = Num.dataset('moons', 90, .35, Math.floor(Math.random() * 1000)); S.redraw(); } }
        ]);
        Viz.note(host, 'Note the boundaries are always axis-aligned staircases. A diagonal boundary needs many splits to approximate — which is exactly why trees love interactions and dislike rotations, and why PCA before a tree is usually a bad idea.');
      },

      forest: function (host) {
        let seed = 11;
        const st = Viz.controls(host, [
          { k: 'B', label: 'trees in the forest', min: 1, max: 80, step: 1, value: 30, fmt: v => v },
          { k: 'depth', label: 'depth of each tree', min: 1, max: 8, step: 1, value: 6, fmt: v => v },
          { k: 'feat', label: 'features per split (decorrelation)', type: 'buttons', value: '1', options: [{ v: '2', t: 'both (bagging)' }, { v: '1', t: 'random 1 (forest)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'single', label: 'single-tree accuracy' }, { k: 'ens', label: 'ensemble accuracy', cls: 'good' },
          { k: 'oob', label: 'out-of-bag estimate', cls: 'key' }, { k: 'var', label: 'boundary instability' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const data = Num.dataset('moons', 120, .42, seed);
            const R = Num.rng(seed + 1);
            const trees = [], oobPred = data.X.map(() => []);
            for (let b = 0; b < st.B; b++) {
              const idx = data.X.map(() => R.int(data.X.length));
              const inbag = new Set(idx);
              const Xb = idx.map(i => data.X[i]), yb = idx.map(i => data.y[i]);
              const featMask = st.feat === '1' ? (R() < .5 ? 0 : 1) : null;
              const Xm = featMask === null ? Xb : Xb.map(p => featMask === 0 ? [p[0], p[1] * 0] : [p[0] * 0, p[1]]);
              const t = Num.tree(Xm, yb, { maxDepth: st.depth, minLeaf: 2 });
              t._mask = featMask;
              trees.push(t);
              data.X.forEach((p, i) => { if (!inbag.has(i)) oobPred[i].push(predOne(t, p)); });
            }
            function predOne(t, p) {
              const q = t._mask === null ? p : (t._mask === 0 ? [p[0], 0] : [0, p[1]]);
              return t.predict(q);
            }
            const ens = (x, y) => Num.mean(trees.map(t => predOne(t, [x, y])));
            const single = Num.tree(data.X, data.y, { maxDepth: st.depth, minLeaf: 2 });
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => {
              Labs.boundary(P, ens, { step: 4 });
              P.contours((x, y) => single.predict([x, y]), [.5], { color: T.amber, width: 1.6, alpha: .9 });
            });
            P.clip(() => Labs.points(P, data.X, data.y));
            const accS = Num.mean(data.X.map((x, i) => ((single.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const accE = Num.mean(data.X.map((x, i) => ((ens(x[0], x[1]) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const oobOK = oobPred.map((ps, i) => ps.length ? (((Num.mean(ps) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0) : null).filter(v => v !== null);
            out({
              single: (accS * 100).toFixed(1) + '%', ens: (accE * 100).toFixed(1) + '%',
              oob: oobOK.length ? (Num.mean(oobOK) * 100).toFixed(1) + '%' : '—',
              var: st.B < 5 ? 'high' : st.B < 20 ? 'moderate' : 'low'
            });
          }
        });
        Viz.buttons(host, [{ label: 'Reseed the data', primary: true, on: () => { seed = Math.floor(Math.random() * 10000); S.redraw(); } }]);
        Viz.legend(host, [{ c: Viz.theme().amber, t: 'single deep tree’s boundary' }, { c: Viz.theme().text, t: 'ensemble boundary' }]);
        Viz.note(host, 'Press <i>reseed</i> repeatedly with B = 1 and then with B = 50. The amber single-tree line lurches every time; the ensemble surface barely moves. That difference is $\\frac{1-\\rho}{B}\\sigma^2$ shrinking in front of you — and switching feature sampling on lowers ρ, dropping the floor as well.');
      }
    },
    quiz: [
      {
        q: 'Random forests differ from plain bagging mainly by…',
        options: ['using deeper trees', 'sampling a random subset of features at each split to decorrelate the trees', 'weighting misclassified points', 'using a lower learning rate'],
        answer: 1,
        why: 'Feature subsampling lowers ρ, which lowers the variance floor ρσ² that more trees alone cannot touch.'
      },
      {
        q: 'Out-of-bag error is available because…',
        options: ['trees are pruned', 'each bootstrap sample omits about 37% of the rows, which those trees never saw', 'forests use cross-validation internally', 'the trees are shallow'],
        answer: 1,
        why: '$(1-1/n)^n \\to e^{-1} \\approx 0.368$ — a free, nearly unbiased validation set per tree.'
      }
    ],
    cards: [
      { q: 'Bagging vs boosting, in one line', a: 'Bagging: independent parallel models averaged, attacks variance. Boosting: dependent sequential models added, attacks bias.' },
      { q: 'Out-of-bag fraction', a: '≈ 37% ($e^{-1}$) of rows are omitted by each bootstrap — a free validation estimate.' },
      { q: 'Impurity measures', a: 'Gini $1-\\sum p_k^2$, entropy $-\\sum p_k\\log p_k$, variance for regression.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.8 */
  ML.section({
    id: 'boosting', track: 'classical', num: '2.8',
    title: 'Gradient boosting, in depth',
    lede: 'The centrepiece of Part 2 and the tool you will be asked about most; rests on §1.9’s Newton step and §1.5’s loss derivation.',
    html: `
<h2><span class="sn">2.8.1</span> The functional-gradient view</h2>
<p>Build $F_m(x) = F_{m-1}(x) + \\eta f_m(x)$, where each weak learner $f_m$ is fit to the <b>negative gradient of the loss evaluated at the current predictions</b>. That is gradient descent, except the parameter being updated is the <i>function</i> itself. For squared error the negative gradient is the residual, which is why the folk explanation "each tree fits the previous errors" is right but incomplete: for logloss it fits $y - p$, and for other losses something else again.</p>

<h2><span class="sn">2.8.2</span> XGBoost’s second-order objective</h2>
<p>Taylor-expand the loss to second order around the current prediction, with $g_i$ the first and $h_i$ the second derivative:</p>
$$\\text{obj}^{(t)} \\approx \\sum_i \\left[g_i f_t(x_i) + \\tfrac12 h_i f_t(x_i)^2\\right] + \\Omega(f_t), \\qquad \\Omega = \\gamma T + \\tfrac12\\lambda\\|w\\|^2$$
<p>A tree is piecewise constant, so group the sum by leaf. Leaf $j$ with $G_j = \\sum_{i \\in j} g_i$ and $H_j = \\sum_{i \\in j} h_i$ contributes $G_j w_j + \\tfrac12(H_j+\\lambda)w_j^2$ — a scalar quadratic. Minimising gives the <b>optimal leaf weight</b></p>
$$w_j^* = -\\frac{G_j}{H_j+\\lambda}$$
<p>and substituting back, the <b>structure score</b> $-\\tfrac12\\sum_j G_j^2/(H_j+\\lambda) + \\gamma T$. A candidate split's <b>gain</b> is the improvement in that score:</p>
$$\\text{gain} = \\tfrac12\\left[\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma$$

${H.worked('worked split — by hand', `
<p>Logloss, so $g_i = p_i - y_i$ and $h_i = p_i(1-p_i)$. Start from $p = 0.5$ for everyone, so every $g = \\pm 0.5$ and every $h = 0.25$. Four samples with labels [1, 1, 0, 0]; the candidate split sends the two positives left and the two negatives right. Take $\\lambda = 1$, $\\gamma = 0$.</p>
${H.code(`LEFT            RIGHT           PARENT
G_L = −1.0      G_R = +1.0      G = 0
H_L =  0.5      H_R =  0.5      H = 1.0`)}
$$\\text{gain} = \\tfrac12\\left[\\tfrac{1}{1.5}+\\tfrac{1}{1.5}-\\tfrac{0}{2}\\right] = \\tfrac12(0.667+0.667) = 0.667 > 0$$
<p>The split is kept. Leaf weights are $w^* = -G/(H+\\lambda) = \\mp 0.667$ — positive leaf up, negative leaf down — then scaled by the learning rate $\\eta$ before being added to the ensemble. Note the parent term vanished because $G = 0$: <mark>a node whose gradients cancel is exactly a node with nothing left to learn.</mark></p>`)}

${H.lab('gain', 'The split-gain calculator', 'Every quantity in the worked example, live. Move the labels and the regularization and watch the gain and the leaf weights respond — including the moment γ makes a split not worth taking, which is pre-pruning built into the score.')}

${H.lab('boost', 'Boosting, one tree at a time', 'Press <i>add a tree</i> and watch the ensemble crawl toward the target while the residuals shrink. Lower the learning rate and you need more trees for the same fit — that is the trade the first row of the tuning table describes.')}

<h2><span class="sn">2.8.3</span> The three libraries</h2>
<p><b>LightGBM</b> changes three things: histogram binning of continuous features (a large constant-factor speedup), <i>leaf-wise</i> best-first growth instead of level-wise (lower loss per tree, more overfitting risk, so cap <code>num_leaves</code> and depth), and GOSS — keep all large-gradient samples, subsample the small-gradient ones, reweight to stay unbiased. <b>CatBoost</b> changes two: ordered target statistics, which compute categorical encodings using only earlier rows in a permutation to avoid the target leakage of naive mean encoding (§2.11), and oblivious (symmetric) trees, which use the same split at every node of a level — weaker per tree, very fast to score, strongly regularised.</p>

<h2><span class="sn">2.8.4</span> Tuning, in the order that matters</h2>
<p>Interviewers ask "how would you tune it" to find out whether you have actually done it. The answer is an ordered strategy, not a parameter list: fix a low learning rate and let early stopping choose the number of trees, then control complexity, then subsample, then regularise, and only then touch anything else.</p>
${H.table(['Knob', 'Start at', 'What it trades'], [
      ['<code>learning_rate</code>', '0.05', 'Lower is always better for accuracy and always slower. Halve it and roughly double the trees.'],
      ['<code>n_estimators</code>', 'large + early stop', 'Never tune by hand; let a validation window pick it (50-round patience).'],
      ['<code>max_depth</code> / <code>num_leaves</code>', '6 / 31', 'Interaction depth against variance. The single most important complexity knob.'],
      ['<code>min_child_weight</code>', '1 → raise', 'Minimum Hessian mass per leaf. The cleanest fix for noisy, thin leaves.'],
      ['<code>subsample</code>, <code>colsample</code>', '0.8, 0.8', 'Adds decorrelation (§2.7) and speed; below ~0.5 you start losing signal.'],
      ['<code>reg_lambda</code> / <code>gamma</code>', '1 / 0', 'λ shrinks leaf values (the λ in $w^*$); γ is the minimum gain to split at all.'],
      ['<code>scale_pos_weight</code>', '1', 'Recall against calibration. Read §2.12 before touching it.']
    ])}

<h3>Monotonic constraints — say this unprompted in a credit interview</h3>
<p>All three libraries let you force a feature's effect to be monotone (<code>monotone_constraints</code>), so higher utilisation can never <i>lower</i> predicted risk. That buys three things at once: a model that agrees with domain knowledge, a defensible story for a validator, and free regularization — it removes exactly the wiggles that noise creates. Cost is a point or two of AUC, usually less. Pair it with <code>interaction_constraints</code> when a validator needs to see that two features never combine.</p>

<h3>One caution about early stopping</h3>
<p>The validation set that chose your tree count has been used for a decision, so it is no longer an unbiased estimate — that is what the nested scheme in §2.14 exists for. And under an out-of-time split, early stopping on a random validation fold will overshoot: stop on the OOT window instead, or you have tuned the number of trees to a distribution you will not be serving.</p>

${H.probe([
      ['Why second-order?', 'It is a Newton step (§1.9): curvature $h_i$ gives better leaf values and better split scores than the gradient alone, and it makes the gain formula loss-agnostic.'],
      ['LightGBM vs XGBoost?', 'Histogram binning + leaf-wise growth + GOSS make LightGBM faster on large data, at the cost of more tuning to avoid overfitting.'],
      ['Where is pruning?', 'The −γ in the gain: a split must beat γ to be taken, so pre-pruning is built into the score.']
    ], 'Running LightGBM leaf-wise with no <code>num_leaves</code> or depth cap, then blaming the library for overfitting.')}`,
    labs: {
      gain: function (host) {
        const st = Viz.controls(host, [
          { k: 'nL1', label: 'positives sent LEFT', min: 0, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'nL0', label: 'negatives sent LEFT', min: 0, max: 6, step: 1, value: 0, fmt: v => v },
          { k: 'nR1', label: 'positives sent RIGHT', min: 0, max: 6, step: 1, value: 0, fmt: v => v },
          { k: 'nR0', label: 'negatives sent RIGHT', min: 0, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'lam', label: 'λ (reg_lambda)', min: 0, max: 10, step: .25, value: 1, fmt: v => v.toFixed(2) },
          { k: 'gam', label: 'γ (min gain to split)', min: 0, max: 2, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'p0', label: 'current prediction p', min: .05, max: .95, step: .05, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gain', label: 'gain', cls: 'key' }, { k: 'wL', label: 'left leaf w*' },
          { k: 'wR', label: 'right leaf w*' }, { k: 'keep', label: 'split kept?' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const p = st.p0, g1 = p - 1, g0 = p - 0, hh = p * (1 - p);
            const GL = st.nL1 * g1 + st.nL0 * g0, HL = (st.nL1 + st.nL0) * hh;
            const GR = st.nR1 * g1 + st.nR0 * g0, HR = (st.nR1 + st.nR0) * hh;
            const term = (G, H) => (G * G) / (H + st.lam);
            const gain = .5 * (term(GL, HL) + term(GR, HR) - term(GL + GR, HL + HR)) - st.gam;
            const wL = -GL / (HL + st.lam), wR = -GR / (HR + st.lam);
            const box = (x, y, wd, ht, title, lines, col) => {
              Labs.roundRect(ctx, x, y, wd, ht, 8);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = col || T.line; ctx.lineWidth = 1.4; ctx.stroke();
              ctx.fillStyle = col || T.blue; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x + 10, y + 8);
              ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace';
              lines.forEach((L, i) => ctx.fillText(L, x + 10, y + 26 + i * 17));
            };
            const bw = Math.min(150, (w - 60) / 3);
            box(14, 20, bw, 92, 'LEFT', ['G_L = ' + GL.toFixed(2), 'H_L = ' + HL.toFixed(2), 'w* = ' + (isFinite(wL) ? wL.toFixed(3) : '—')], T.blue);
            box(24 + bw, 20, bw, 92, 'RIGHT', ['G_R = ' + GR.toFixed(2), 'H_R = ' + HR.toFixed(2), 'w* = ' + (isFinite(wR) ? wR.toFixed(3) : '—')], T.red);
            box(34 + 2 * bw, 20, bw, 92, 'PARENT', ['G = ' + (GL + GR).toFixed(2), 'H = ' + (HL + HR).toFixed(2), 'score = ' + (-.5 * term(GL + GR, HL + HR)).toFixed(3)], T.muted);
            ctx.fillStyle = T.text; ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('gain = ½[ ' + term(GL, HL).toFixed(3) + ' + ' + term(GR, HR).toFixed(3) + ' − ' + term(GL + GR, HL + HR).toFixed(3) + ' ] − γ = ' + gain.toFixed(4), 14, 132);
            ctx.fillStyle = gain > 0 ? T.green : T.red; ctx.font = 'bold 14px ui-sans-serif';
            ctx.fillText(gain > 0 ? '→ split taken' : '→ split rejected (gain ≤ 0)', 14, 160);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('Leaf values are then scaled by the learning rate η before being added to the ensemble.', 14, 188);
            out({
              gain: gain.toFixed(4), wL: isFinite(wL) ? wL.toFixed(3) : '—',
              wR: isFinite(wR) ? wR.toFixed(3) : '—', keep: gain > 0 ? 'yes' : 'no'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the worked example', primary: true, on: () => { st.$set('nL1', 2); st.$set('nL0', 0); st.$set('nR1', 0); st.$set('nR0', 2); st.$set('lam', 1); st.$set('gam', 0); st.$set('p0', .5); S.redraw(); } },
          { label: 'A useless split (gradients cancel)', on: () => { st.$set('nL1', 2); st.$set('nL0', 2); st.$set('nR1', 2); st.$set('nR0', 2); S.redraw(); } }
        ]);
      },

      boost: function (host) {
        const data = Num.regressionData(60, 'sine', .28, 12);
        const X = data.X.map(x => [x]);
        let model = Num.boosting(X, data.y, { lr: .3, maxDepth: 2 });
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate η', min: .02, max: 1, step: .02, value: .3, fmt: v => v.toFixed(2) },
          { k: 'depth', label: 'depth per tree', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'show', label: 'show', type: 'buttons', value: 'fit', options: [{ v: 'fit', t: 'the fit' }, { v: 'res', t: 'residuals the next tree sees' }] }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'trees', label: 'trees', cls: 'key' }, { k: 'mse', label: 'train MSE' }, { k: 'resid', label: 'mean |residual|' }
        ]);
        function reset() { model = Num.boosting(X, data.y, { lr: st.lr, maxDepth: st.depth }); S.redraw(); }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: st.show === 'fit' ? [-3, 3] : [-1.6, 1.6] })
              .frame({ xlabel: 'x', ylabel: st.show === 'fit' ? 'y' : 'residual y − F(x)' });
            if (st.show === 'fit') {
              P.clip(() => {
                P.dots(data.X.map((x, i) => [x, data.y[i]]), { r: 3.4, color: T.faint, alpha: .8 });
                P.fn(x => Math.sin(x * 1.4) * 1.6, { color: T.green, width: 1.6, dash: [6, 4] });
                P.fn(x => model.predict([x]), { color: T.blue, width: 2.8, n: 400 });
              });
            } else {
              P.clip(() => {
                P.hline(0, { color: T.faint, dash: [3, 3] });
                data.X.forEach((x, i) => {
                  const r = data.y[i] - model.predict([x]);
                  P.line([[x, 0], [x, r]], { color: r > 0 ? T.blue : T.red, width: 1.4 });
                  P.dots([[x, r]], { r: 3, color: r > 0 ? T.blue : T.red });
                });
              });
            }
            out({
              trees: model.trees.length, mse: model.loss().toFixed(4),
              resid: Num.mean(data.X.map((x, i) => Math.abs(data.y[i] - model.predict([x])))).toFixed(4)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Add a tree', primary: true, on: () => { model.addTree(); S.redraw(); } },
          { label: 'Add 10', on: () => { for (let i = 0; i < 10; i++) model.addTree(); S.redraw(); } },
          { label: 'Add 100', on: () => { for (let i = 0; i < 100; i++) model.addTree(); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        Viz.note(host, 'Switch to the residual view and step one tree at a time: each new stump is fitted to what is left over, and the leftovers shrink in a pattern you can watch. At η = 1 the fit lurches and starts chasing noise; at η = 0.05 it creeps but ends up smoother — the accuracy/patience trade in one control.');
      }
    },
    quiz: [
      {
        q: 'The optimal leaf weight in XGBoost is…',
        options: ['the mean of the residuals in the leaf', '$-G/(H+\\lambda)$', '$G/(H-\\lambda)$', 'the median of the labels'],
        answer: 1,
        why: 'Minimising the per-leaf quadratic $Gw + \\frac12(H+\\lambda)w^2$ gives $w^* = -G/(H+\\lambda)$ — a Newton step with a damping λ.'
      },
      {
        q: 'A candidate split has $G_L = -1$, $H_L = 0.5$, $G_R = +1$, $H_R = 0.5$, λ = 1, γ = 0. The gain is…',
        options: ['0', '0.667', '1.333', 'negative'],
        answer: 1,
        why: '½[1/1.5 + 1/1.5 − 0/2] = ½(1.333) = 0.667. The parent term vanishes because the gradients cancel exactly.'
      },
      {
        q: 'You must tune a boosted model with limited compute. The correct order is…',
        options: ['n_estimators, then learning_rate, then everything else', 'fix a low learning rate + early stopping, then depth/leaves, then subsampling, then regularization', 'gamma first, then lambda, then depth', 'tune all parameters jointly with grid search'],
        answer: 1,
        why: 'Tree count should never be tuned by hand; complexity is the dominant knob; regularization and subsampling are refinements.'
      }
    ],
    cards: [
      { q: 'XGBoost leaf weight and gain', a: '$w^*=-G/(H+\\lambda)$; gain $=\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$.' },
      { q: 'Why second-order boosting?', a: 'It is a Newton step in function space: curvature gives better leaf values and a loss-agnostic gain formula.' },
      { q: 'LightGBM’s three changes', a: 'Histogram binning, leaf-wise growth, GOSS. Faster on large data; needs num_leaves/depth caps.' },
      { q: 'CatBoost’s two changes', a: 'Ordered target statistics (prevents encoding leakage) and oblivious symmetric trees.' },
      { q: 'Monotonic constraints — why say it', a: 'Domain agreement + validator story + free regularization, at a cost of a point or two of AUC.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.9 */
  ML.section({
    id: 'unsupervised', track: 'classical', num: '2.9',
    title: 'Unsupervised: k-means, GMM/EM, DBSCAN, anomaly detection',
    lede: 'The EM derivation here is the same bound-tightening argument used for variational objectives generally — and the anomaly half is the part that pays for itself in fraud and AML.',
    html: `
<h2><span class="sn">2.9.1</span> k-means</h2>
<p>Minimises within-cluster sum of squares by <b>Lloyd's alternation</b>: assign each point to the nearest centroid, then move each centroid to its cluster's mean. Each step cannot increase the objective, so it converges — to a local optimum that depends on initialisation (use k-means++). It requires you to choose $k$, and it implicitly assumes isotropic, similarly-sized clusters, because "nearest centroid" carves space into a Voronoi diagram.</p>

${H.lab('kmeans', 'Lloyd’s algorithm, step by step', 'Press <i>assign</i> and <i>update</i> alternately and watch the objective fall monotonically. Then press <i>random restart</i> a few times on the crescents — the local optima are real, and k-means++ visibly reduces how often you land in a bad one.')}

<h2><span class="sn">2.9.2</span> GMM and EM</h2>
<p>Soft clustering with per-component covariances. The likelihood $\\log p(x) = \\log\\sum_z p(x,z)$ is awkward because the sum sits inside the log. Apply Jensen's inequality with any distribution $q(z)$ to get a lower bound — the <b>ELBO</b>:</p>
$$\\log p(x) \\ge \\mathbb{E}_q[\\log p(x,z)] - \\mathbb{E}_q[\\log q(z)] = \\mathcal{L}(q,\\theta)$$
<p>The gap between the two sides is exactly $D_{KL}(q\\,\\|\\,p(z\\mid x))$ (§1.10). So the <b>E-step</b> sets $q = p(z\\mid x)$, which closes the gap and makes the bound tight; the <b>M-step</b> maximises the now-tight bound over $\\theta$. Because each step either tightens or raises the bound, the likelihood cannot decrease. <mark>"Tighten the bound, then maximise it" is the whole of EM in five words.</mark></p>

${H.lab('em', 'EM on a one-dimensional mixture', 'Watch responsibilities (the E-step) and parameter updates (the M-step) alternate, with the log-likelihood printed each round. It never goes down — that is the guarantee the ELBO argument buys.')}

<h2><span class="sn">2.9.3</span> DBSCAN and hierarchical</h2>
<p><b>DBSCAN</b> grows clusters through dense neighbourhoods (<code>eps</code>, <code>minPts</code>), so it finds arbitrarily-shaped clusters and labels outliers as noise instead of forcing them into a group — but it struggles when density varies between clusters (HDBSCAN addresses that). <b>Hierarchical</b> clustering builds a dendrogram you can cut at any level, so no $k$ is needed up front; the linkage choice (single/complete/average/Ward) determines the shape of what you get.</p>

${H.lab('dbscan', 'Where k-means fails and density-based clustering does not', 'The same crescents through both algorithms. k-means slices them in half — it can only produce convex Voronoi cells. DBSCAN follows the data and gets to say "noise".')}

<h2><span class="sn">2.9.4</span> Choosing k without fooling yourself</h2>
<p>The <b>elbow method</b> plots WCSS against $k$ and looks for the bend — quick and frequently ambiguous, because WCSS decreases monotonically by construction. The <b>silhouette</b> compares each point's mean distance to its own cluster against the nearest other cluster, giving a value in $[-1,1]$ with a meaningful zero: near 0 means the point sits on a boundary, negative means it is in the wrong cluster. Prefer it. The <b>gap statistic</b> compares your WCSS against the same clustering on uniform random data, and it is the only one of the three that can tell you $k = 1$ — i.e. that there is no cluster structure at all. Always ask that question first; a great deal of published clustering is structure imposed on noise.</p>

<h2><span class="sn">2.9.5</span> Anomaly detection — the half that pays for itself</h2>
${H.table(['Method', 'Definition of "unusual"', 'Notes'], [
      ['<b>Isolation Forest</b>', 'Few random splits isolate the point', 'Fast, high-dimensional, almost no tuning beyond contamination — the default'],
      ['<b>One-class SVM</b>', 'Outside a boundary fitted around the bulk', 'RBF kernel (§2.6); sensitive to scaling and to ν'],
      ['<b>Local Outlier Factor</b>', 'Local density much lower than neighbours’', 'Catches locally-anomalous points: a normal-looking transaction in an abnormal neighbourhood'],
      ['<b>Autoencoder</b>', 'High reconstruction error', 'Right when the "normal" manifold is genuinely nonlinear and you have a lot of it']
    ])}
<p>The hard part is not the model, it is evaluation without labels. Three practical moves: score, rank, and have investigators review the top-$k$ so you build a labelled set from the highest-value cases; measure <b>precision@k</b> at the review capacity you actually have, since nobody can investigate 5,000 alerts; and check stability over time, because an anomaly detector whose alert volume triples overnight has usually detected a data-pipeline change, not fraud. Once you have a few thousand labels, a supervised model on those labels almost always beats the unsupervised score — <mark>treat anomaly detection as a bootstrap into supervision, not a destination.</mark></p>

${H.probe([
      ['Derive EM in one sentence.', 'Jensen gives an ELBO whose gap is $D_{KL}(q\\|p(z|x))$; the E-step closes the gap, the M-step raises the bound, so the likelihood never decreases.'],
      ['Why does k-means fail on crescents?', 'Nearest-centroid assignment produces convex Voronoi cells; crescents are not convex.'],
      ['How do you choose k?', 'Silhouette over elbow; gap statistic if you need to test k = 1 — that is, whether there is structure at all.']
    ], 'Clustering unscaled features. k-means minimises Euclidean distance, so an income column in pounds will dominate and you have clustered on income alone without noticing.')}`,
    labs: {
      kmeans: function (host) {
        let data = Num.dataset('blobs', 180, .55, 3);
        let km = null, iters = 0, hist = [];
        const st = Viz.controls(host, [
          { k: 'k', label: 'k (clusters)', min: 2, max: 6, step: 1, value: 3, fmt: v => v },
          { k: 'init', label: 'initialisation', type: 'buttons', value: 'pp', options: [{ v: 'rand', t: 'random' }, { v: 'pp', t: 'k-means++' }] }
        ], restart);
        const out = Viz.readout(host, [
          { k: 'wcss', label: 'WCSS (objective)', cls: 'key' }, { k: 'it', label: 'iterations' }, { k: 'sil', label: 'silhouette' }
        ]);
        function restart(seed) {
          km = Num.kmeans(data.X, st.k, typeof seed === 'number' ? seed : Math.floor(Math.random() * 1e5), st.init === 'pp');
          km.assignStep(); iters = 0; hist = [km.wcss()]; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4.5, 4.5], yd: [-3.4, 3.4] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            const cols = Labs.palette(st.k);
            P.clip(() => {
              // Voronoi shading
              P.field((x, y) => {
                let bi = 0, bd = 1e9;
                km.centers.forEach((c, j) => { const d = (x - c[0]) ** 2 + (y - c[1]) ** 2; if (d < bd) { bd = d; bi = j; } });
                return bi / Math.max(1, st.k - 1);
              }, {
                step: 5, lo: 0, hi: 1,
                colors: t => {
                  const idx = Math.round(t * (st.k - 1));
                  const c = cols[idx] || cols[0];
                  const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
                  return [r, g, b, 34];
                }
              });
              data.X.forEach((p, i) => P.dots([p], { r: 3.2, color: cols[km.assign[i]], alpha: .95 }));
              km.centers.forEach((c, j) => {
                P.dots([c], { r: 8, color: cols[j], stroke: true, strokeWidth: 2.5 });
                P.dots([c], { r: 2.5, color: '#fff' });
              });
            });
            // objective inset
            if (hist.length > 1) {
              const ix = w - 150, iy = 16, iw = 130, ih = 48;
              ctx.fillStyle = T.paper; ctx.globalAlpha = .9; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(ix, iy, iw, ih);
              const mx = hist[0], mn = Math.min.apply(null, hist);
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              hist.forEach((v, i) => {
                const X = ix + iw * i / Math.max(1, hist.length - 1);
                const Y = iy + ih - ih * ((v - mn) / ((mx - mn) || 1)) * .8 - 4;
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('WCSS — monotone', ix + 4, iy + 3);
            }
            // silhouette
            let sil = 0;
            const sample = data.X.slice(0, 80);
            sample.forEach((p, i) => {
              const own = km.assign[i];
              const dists = {};
              data.X.forEach((q, j) => {
                if (i === j) return;
                const c = km.assign[j], d = Math.hypot(p[0] - q[0], p[1] - q[1]);
                dists[c] = dists[c] || []; dists[c].push(d);
              });
              const a = dists[own] ? Num.mean(dists[own]) : 0;
              let b = 1e9;
              Object.keys(dists).forEach(c => { if (+c !== own) b = Math.min(b, Num.mean(dists[c])); });
              sil += (b - a) / Math.max(a, b);
            });
            out({ wcss: km.wcss().toFixed(1), it: iters, sil: (sil / sample.length).toFixed(3) });
          }
        });
        Viz.buttons(host, [
          { label: 'Assign step', primary: true, on: () => { km.assignStep(); hist.push(km.wcss()); S.redraw(); } },
          { label: 'Update step', on: () => { km.updateStep(); iters++; hist.push(km.wcss()); S.redraw(); } },
          { label: 'Run to convergence', on: () => { for (let i = 0; i < 40; i++) { const ch = km.assignStep(); km.updateStep(); iters++; hist.push(km.wcss()); if (!ch) break; } S.redraw(); } },
          { label: 'Random restart', on: () => restart() },
          { label: 'Crescents', on: () => { data = Num.dataset('moons', 180, .3, 4); restart(); } },
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 180, .55, 3); restart(); } }
        ]);
        restart(3);
      },

      em: function (host) {
        let xs = [], g = null, ll = [];
        const st = Viz.controls(host, [
          { k: 'k', label: 'components', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'sep', label: 'true separation', min: .5, max: 6, step: .1, value: 3, fmt: v => v.toFixed(1) },
          { k: 'n', label: 'samples', min: 60, max: 800, step: 20, value: 250, fmt: v => v }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'll', label: 'log-likelihood', cls: 'key' }, { k: 'it', label: 'EM rounds' },
          { k: 'mu', label: 'means' }, { k: 'pi', label: 'weights' }
        ]);
        function reset() {
          const R = Num.rng(29);
          xs = [];
          for (let i = 0; i < st.n; i++) xs.push(R() < .45 ? R.normal(-st.sep / 2, .8) : R.normal(st.sep / 2, 1.1));
          g = Num.gmm1d(xs, st.k, 7); ll = []; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const hh = Num.hist(xs, 40, -8, 8);
            const dens = hh.bins.map(c => c / (xs.length * hh.w));
            const mix = x => Num.sum(g.mu.map((m, j) => g.pi[j] * Num.normPdf(x, m, g.sg[j])));
            const P = Viz.plot(ctx, w, h, { xd: [-8, 8], yd: [0, Math.max(Math.max.apply(null, dens), .4) * 1.25] })
              .frame({ xlabel: 'x', ylabel: 'density' });
            P.clip(() => {
              hh.centers.forEach((c, i) => {
                ctx.fillStyle = T.faint; ctx.globalAlpha = .35;
                const x0 = P.x(c - hh.w / 2), x1 = P.x(c + hh.w / 2);
                ctx.fillRect(x0, P.y(dens[i]), Math.max(1, x1 - x0 - 1), P.y(0) - P.y(dens[i]));
                ctx.globalAlpha = 1;
              });
              const cols = Labs.palette(st.k);
              g.mu.forEach((m, j) => P.fn(x => g.pi[j] * Num.normPdf(x, m, g.sg[j]), { color: cols[j], width: 1.8, dash: [5, 3] }));
              P.fn(mix, { color: T.blue, width: 2.8 });
              g.mu.forEach((m, j) => P.vline(m, { color: cols[j], dash: [2, 3], width: 1 }));
            });
            out({
              ll: ll.length ? ll[ll.length - 1].toFixed(4) : '—', it: ll.length,
              mu: '[' + g.mu.map(v => v.toFixed(2)).join(', ') + ']',
              pi: '[' + g.pi.map(v => v.toFixed(2)).join(', ') + ']'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'E-step', primary: true, on: () => { ll.push(g.eStep()); S.redraw(); } },
          { label: 'M-step', on: () => { g.mStep(); S.redraw(); } },
          { label: 'Run 20 rounds', on: () => { for (let i = 0; i < 20; i++) { ll.push(g.eStep()); g.mStep(); } S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Watch the log-likelihood readout after every round: it never decreases. That monotonicity is not luck — it follows directly from the ELBO argument, and it is the property that makes EM safe to run to convergence.');
      },

      dbscan: function (host) {
        let data = Num.dataset('moons', 200, .18, 6);
        const st = Viz.controls(host, [
          { k: 'algo', label: 'algorithm', type: 'buttons', value: 'db', options: [{ v: 'km', t: 'k-means' }, { v: 'db', t: 'DBSCAN' }] },
          { k: 'k', label: 'k (for k-means)', min: 2, max: 6, step: 1, value: 2, fmt: v => v },
          { k: 'eps', label: 'eps (DBSCAN radius)', min: .08, max: 1.2, step: .02, value: .34, fmt: v => v.toFixed(2) },
          { k: 'min', label: 'minPts', min: 2, max: 20, step: 1, value: 5, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'clusters', label: 'clusters found', cls: 'key' }, { k: 'noise', label: 'points labelled noise' }, { k: 'shape', label: 'recovers the crescents?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            let labels, k, noise = 0;
            if (st.algo === 'km') {
              const km = Num.kmeans(data.X, st.k, 5, true);
              for (let i = 0; i < 40; i++) { const ch = km.assignStep(); km.updateStep(); if (!ch) break; }
              km.assignStep();
              labels = km.assign; k = st.k;
              P.clip(() => {
                const cols = Labs.palette(k);
                km.centers.forEach((c, j) => P.dots([c], { r: 8, color: cols[j], stroke: true, strokeWidth: 2.5 }));
              });
            } else {
              const db = Num.dbscan(data.X, st.eps, st.min);
              labels = db.labels; k = db.k; noise = labels.filter(l => l === -1).length;
            }
            const cols = Labs.palette(Math.max(1, k));
            P.clip(() => {
              data.X.forEach((p, i) => {
                const l = labels[i];
                P.dots([p], { r: 3.4, color: l < 0 ? 'transparent' : cols[l % cols.length], stroke: l < 0 ? T.faint : true, strokeWidth: 1.4 });
                if (l < 0) { P.text(p[0], p[1], '×', { color: T.faint, font: '12px ui-monospace', align: 'center' }); }
              });
            });
            out({
              clusters: k, noise: st.algo === 'db' ? noise : '—',
              shape: st.algo === 'db' && k === 2 && noise < data.X.length * .25 ? 'yes' : (st.algo === 'km' ? 'no — sliced in half' : 'tune eps/minPts')
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Crescents', on: () => { data = Num.dataset('moons', 200, .18, 6); S.redraw(); } },
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 200, .5, 2); S.redraw(); } },
          { label: 'Circles', on: () => { data = Num.dataset('circles', 200, .12, 9); S.redraw(); } }
        ]);
        Viz.note(host, '× marks noise. DBSCAN’s ability to <i>refuse to cluster</i> a point is the feature — in fraud work those refusals are frequently the interesting rows.');
      }
    },
    quiz: [
      {
        q: 'EM is guaranteed not to decrease the likelihood because…',
        options: ['it uses gradient descent with a small step', 'the E-step makes the ELBO tight and the M-step raises it', 'the clusters are convex', 'it re-initialises each round'],
        answer: 1,
        why: 'The gap between log p(x) and the ELBO is KL(q ‖ p(z|x)); setting q = p(z|x) closes it, then maximising the bound cannot lower the likelihood.'
      },
      {
        q: 'You need to know whether your data has any cluster structure at all. Which diagnostic answers that?',
        options: ['The elbow method', 'Silhouette score', 'The gap statistic', 'WCSS at k = 10'],
        answer: 2,
        why: 'Only the gap statistic compares against a null of uniform random data, so only it can support k = 1.'
      },
      {
        q: 'An anomaly detector’s alert volume triples overnight. The first hypothesis should be…',
        options: ['a fraud ring', 'a data-pipeline or upstream schema change', 'model drift in the labels', 'the contamination parameter is wrong'],
        answer: 1,
        why: 'Sudden volume changes are almost always input distribution changes. Check the inputs (§2.18 PSI) before opening an investigation.'
      }
    ],
    cards: [
      { q: 'EM in five words', a: '"Tighten the bound, then maximise it."' },
      { q: 'Why k-means fails on crescents', a: 'Nearest-centroid assignment produces convex Voronoi cells; the clusters are not convex.' },
      { q: 'Choosing k', a: 'Silhouette (meaningful zero) over elbow (monotone by construction); gap statistic to test k = 1.' },
      { q: 'Anomaly detection’s hard part', a: 'Evaluation without labels: rank, review top-k, measure precision@k at real capacity, and watch stability over time.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.10 */
  ML.section({
    id: 'pca', track: 'classical', num: '2.10',
    title: 'Dimensionality reduction: PCA, SVD, t-SNE, UMAP',
    lede: 'Rests entirely on §1.8; the same eigen-argument reappears whenever someone asks what a low-rank approximation is.',
    html: `
<h2><span class="sn">2.10.1</span> PCA, derived</h2>
<p>Find the unit direction of maximum projected variance: maximise $w^\\mathsf{T}\\Sigma w$ subject to $\\|w\\| = 1$. Form the Lagrangian $w^\\mathsf{T}\\Sigma w - \\lambda(w^\\mathsf{T}w - 1)$ and set the derivative to zero: $2\\Sigma w - 2\\lambda w = 0$, i.e.</p>
$$\\Sigma w = \\lambda w$$
<p>The principal directions are the <b>eigenvectors of the covariance matrix</b>, and each eigenvalue is the variance captured along its direction. Equivalently PCA is the SVD of the centred data matrix — the numerically stable way to compute it, and the reason "PCA" and "truncated SVD" get used interchangeably. Always centre; scale too unless your features share units.</p>

${H.lab('pca', 'PCA on data you can shape', 'Drag the cloud’s shape and watch the components track it. The explained-variance readout is $\\lambda_i/\\sum_j\\lambda_j$; the reconstruction shows what you throw away by keeping only PC1.')}

<h2><span class="sn">2.10.2</span> What PCA is not for</h2>
${H.table(['Use', 'Verdict'], [
      ['Compressing correlated numeric features before a linear model', 'Good — collinearity handled, variance retained'],
      ['Visualising structure in 2-D', 'Fine, with the caveat that PC1–PC2 may hide the interesting direction'],
      ['Feature selection', 'No — components are mixtures of all features, so interpretability is lost'],
      ['Before a tree ensemble', 'Usually harmful — trees split on axes, and PCA rotates the axes away from the meaningful ones'],
      ['Removing noise', 'Sometimes, if noise is genuinely low-variance — which is an assumption, not a fact']
    ])}

<h2><span class="sn">2.10.3</span> t-SNE and UMAP</h2>
<p>Both are non-linear and optimise the preservation of <b>local</b> neighbourhoods. They are visualisation tools. Cluster sizes, inter-cluster distances and empty space in a t-SNE plot are artefacts of the optimisation and the perplexity setting, <b>not properties of your data</b>. UMAP preserves somewhat more global structure and is faster, but the same warning applies.</p>
${H.flag('Reading global geometry — cluster distances, relative sizes — off a t-SNE or UMAP plot is the classic mistake. Say "the neighbourhoods are meaningful, the distances are not" and move on.')}

${H.lab('tsne', 'The same data, three projections', 'PCA, a t-SNE-style neighbour embedding, and a random projection — all computed here. Re-run the embedding with a different seed and watch the cluster positions and apparent sizes change while the neighbourhood memberships stay stable. That instability is exactly what you must not interpret.')}

${H.probe([
      ['What does PC1 maximise?', 'Projected variance; it is the top eigenvector of the covariance matrix.'],
      ['PCA or SVD?', 'PCA is the SVD of the centred data matrix — same object, and the SVD route is numerically stabler.'],
      ['Would you PCA before boosting?', 'Rarely — trees split on axes and PCA rotates away from the interpretable ones, costing explainability for no accuracy gain.']
    ], 'Reading global geometry — cluster distances, relative sizes — off a t-SNE or UMAP plot.')}`,
    labs: {
      pca: function (host) {
        const st = Viz.controls(host, [
          { k: 'rot', label: 'rotate the cloud', min: 0, max: 180, step: 1, value: 30, fmt: v => v + '°' },
          { k: 'aniso', label: 'anisotropy (λ₁/λ₂)', min: 1, max: 12, step: .2, value: 4, fmt: v => v.toFixed(1) + '×' },
          { k: 'show', label: 'show', type: 'buttons', value: 'pcs', options: [{ v: 'pcs', t: 'components' }, { v: 'proj', t: 'projection onto PC1' }, { v: 'rec', t: 'reconstruction' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'l1', label: 'λ₁', cls: 'key' }, { k: 'l2', label: 'λ₂' },
          { k: 'ev1', label: 'PC1 explains' }, { k: 'err', label: 'error if you drop PC2' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(43), th = st.rot * Math.PI / 180;
            const pts = [];
            for (let i = 0; i < 320; i++) {
              const a = R.normal(0, Math.sqrt(st.aniso)), b = R.normal(0, 1);
              pts.push([a * Math.cos(th) - b * Math.sin(th), a * Math.sin(th) + b * Math.cos(th)]);
            }
            const pc = Num.pca(pts);
            const P = Viz.plot(ctx, w, h, { xd: [-6, 6], yd: [-4.5, 4.5] }).frame({ xlabel: 'feature 1', ylabel: 'feature 2' });
            P.clip(() => {
              if (st.show === 'pcs') {
                P.dots(pts, { r: 2.6, color: T.blue, alpha: .55 });
              } else {
                const v = pc.vectors[0];
                pts.forEach(p => {
                  const c = [p[0] - pc.mean[0], p[1] - pc.mean[1]];
                  const t = c[0] * v[0] + c[1] * v[1];
                  const proj = [pc.mean[0] + t * v[0], pc.mean[1] + t * v[1]];
                  if (st.show === 'proj') {
                    P.line([p, proj], { color: T.faint, width: .8, alpha: .5 });
                    P.dots([p], { r: 2.2, color: T.blue, alpha: .4 });
                    P.dots([proj], { r: 2.6, color: T.red, alpha: .8 });
                  } else {
                    P.dots([proj], { r: 2.6, color: T.red, alpha: .7 });
                  }
                });
              }
              pc.vectors.forEach((v, i) => {
                const s = 2 * Math.sqrt(Math.max(0, pc.values[i]));
                P.arrow(pc.mean[0], pc.mean[1], pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, { color: i ? T.amber : T.green, width: 2.6 });
                P.text(pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, ' PC' + (i + 1) + ' · λ=' + pc.values[i].toFixed(2), { color: i ? T.amber : T.green, font: '11px ui-sans-serif' });
              });
            });
            out({
              l1: pc.values[0].toFixed(3), l2: pc.values[1].toFixed(3),
              ev1: (pc.explained[0] * 100).toFixed(1) + '%',
              err: (pc.explained[1] * 100).toFixed(1) + '% of variance'
            });
          }
        });
        Viz.note(host, 'Set anisotropy to 1 and the components become arbitrary — with equal eigenvalues there is no preferred direction, and the "principal" component is whatever the numerics happened to return. That degeneracy is worth knowing before you interpret loadings.');
      },

      tsne: function (host) {
        let seed = 3;
        const st = Viz.controls(host, [
          { k: 'method', label: 'projection', type: 'buttons', value: 'ne', options: [{ v: 'pca', t: 'PCA' }, { v: 'ne', t: 'neighbour embedding' }, { v: 'rand', t: 'random projection' }] },
          { k: 'perp', label: 'neighbourhood size', min: 3, max: 40, step: 1, value: 12, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kept', label: 'neighbours preserved', cls: 'key' }, { k: 'var', label: 'variance kept (PCA)' }, { k: 'stable', label: 'layout stable across seeds?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            // 5-D data with three genuine clusters
            const R = Num.rng(seed);
            const D = 5, K = 3, n = 150;
            const centers = Array.from({ length: K }, () => Array.from({ length: D }, () => R.normal(0, 2.4)));
            const X = [], lab = [];
            for (let i = 0; i < n; i++) {
              const c = i % K;
              X.push(centers[c].map(v => v + R.normal(0, .8)));
              lab.push(c);
            }
            let Y;
            if (st.method === 'pca') {
              const pc = Num.pca(X);
              Y = X.map(p => {
                const cc = p.map((v, j) => v - pc.mean[j]);
                return [Num.dot(cc, pc.vectors[0]), Num.dot(cc, pc.vectors[1])];
              });
            } else if (st.method === 'rand') {
              const RP = Num.rng(seed + 9);
              const a = Array.from({ length: D }, () => RP.normal(0, 1)), b = Array.from({ length: D }, () => RP.normal(0, 1));
              Y = X.map(p => [Num.dot(p, a) / Math.sqrt(D), Num.dot(p, b) / Math.sqrt(D)]);
            } else {
              // simple force-directed neighbour embedding (t-SNE flavoured, not identical)
              const RE = Num.rng(seed + 21);
              Y = X.map(() => [RE.normal(0, .6), RE.normal(0, .6)]);
              const kNN = X.map((p, i) => X.map((q, j) => [j, Num.dot(p.map((v, t) => v - q[t]), p.map((v, t) => v - q[t]))])
                .sort((u, v) => u[1] - v[1]).slice(1, st.perp + 1).map(u => u[0]));
              for (let it = 0; it < 220; it++) {
                const lr = 0.12 * (1 - it / 260);
                for (let i = 0; i < n; i++) {
                  let fx = 0, fy = 0;
                  kNN[i].forEach(j => { fx -= (Y[i][0] - Y[j][0]); fy -= (Y[i][1] - Y[j][1]); });
                  for (let s = 0; s < 14; s++) {
                    const j = RE.int(n); if (j === i) continue;
                    const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
                    const d2 = dx * dx + dy * dy + .05;
                    fx += 2.2 * dx / d2; fy += 2.2 * dy / d2;
                  }
                  Y[i][0] += lr * fx / st.perp; Y[i][1] += lr * fy / st.perp;
                }
              }
            }
            const xs = Y.map(p => p[0]), ys = Y.map(p => p[1]);
            const pad = .15;
            const xr = [Math.min.apply(null, xs), Math.max.apply(null, xs)], yr = [Math.min.apply(null, ys), Math.max.apply(null, ys)];
            const dx = (xr[1] - xr[0]) * pad || 1, dy = (yr[1] - yr[0]) * pad || 1;
            const P = Viz.plot(ctx, w, h, { xd: [xr[0] - dx, xr[1] + dx], yd: [yr[0] - dy, yr[1] + dy] })
              .frame({ xlabel: 'component 1', ylabel: 'component 2' });
            const cols = Labs.palette(K);
            P.clip(() => Y.forEach((p, i) => P.dots([p], { r: 3.6, color: cols[lab[i]], stroke: true })));
            // neighbour preservation
            const nn = (pts, i, k) => pts.map((q, j) => [j, (pts[i][0] - q[0]) ** 2 + (pts[i][1] - q[1]) ** 2 + (pts[i].length > 2 ? 0 : 0)])
              .sort((u, v) => u[1] - v[1]).slice(1, k + 1).map(u => u[0]);
            let kept = 0;
            for (let i = 0; i < n; i += 3) {
              const hi = X.map((q, j) => [j, Num.sum(X[i].map((v, t) => (v - q[t]) ** 2))]).sort((u, v) => u[1] - v[1]).slice(1, 11).map(u => u[0]);
              const lo = nn(Y, i, 10);
              kept += hi.filter(j => lo.indexOf(j) >= 0).length / 10;
            }
            const pc = Num.pca(X);
            out({
              kept: (100 * kept / Math.ceil(n / 3)).toFixed(0) + '%',
              var: ((pc.explained[0] + pc.explained[1]) * 100).toFixed(0) + '%',
              stable: st.method === 'ne' ? 'no — re-run and see' : 'yes'
            });
          }
        });
        Viz.buttons(host, [{ label: 'Re-run with a new seed', primary: true, on: () => { seed = Math.floor(Math.random() * 1e4); S.redraw(); } }]);
        Viz.note(host, 'Re-run the neighbour embedding a few times: the three clusters stay separated (that is real) while their positions, orientations and apparent sizes move (that is not). PCA’s layout is deterministic but may hide structure that lives outside the top two variance directions.');
      }
    },
    quiz: [
      {
        q: 'PCA’s first component maximises…',
        options: ['the correlation with the target', 'the projected variance, subject to unit norm', 'the number of non-zero loadings', 'the reconstruction error'],
        answer: 1,
        why: 'The Lagrangian gives Σw = λw: the top eigenvector of the covariance, whose eigenvalue is the variance captured.'
      },
      {
        q: 'In a t-SNE plot, two clusters appear far apart and one looks much larger. What can you conclude?',
        options: ['They are genuinely far apart and one is bigger', 'Neither — inter-cluster distance and cluster size are artefacts of the optimisation', 'The perplexity is too low', 'The data is Gaussian'],
        answer: 1,
        why: 'Neighbourhood membership is meaningful; global geometry is not. Re-running with a different seed usually changes both.'
      }
    ],
    cards: [
      { q: 'PCA derivation', a: 'Maximise $w^\\mathsf{T}\\Sigma w$ s.t. $\\|w\\|=1$ → $\\Sigma w=\\lambda w$: eigenvectors of the covariance.' },
      { q: 'Explained variance', a: '$\\lambda_i/\\sum_j\\lambda_j$ — the share of total variance along component $i$.' },
      { q: 't-SNE caveat', a: 'Local neighbourhoods are meaningful; cluster sizes, distances and empty space are optimisation artefacts.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.11 */
  ML.section({
    id: 'features', track: 'classical', num: '2.11',
    title: 'Feature engineering, encoding, missing data, leakage',
    lede: 'The most practically load-bearing section in Part 2, and the one where interviews are actually lost.',
    html: `
<h2><span class="sn">2.11.1</span> Encoding</h2>
<p><b>One-hot</b> is safe but explodes on high cardinality. <b>Target / mean encoding</b> is powerful and leak-prone: the encoding must be fitted <i>inside</i> each training fold, never on the full dataset, or the target sneaks into the features and cross-validation becomes a fiction. CatBoost's ordered statistics (§2.8) are a principled fix. <b>Ordinal</b> encoding is fine for trees, misleading for linear models unless the order is real.</p>

<h2><span class="sn">2.11.2</span> WOE and IV — the credit-risk vocabulary</h2>
<p>Bin a feature, then per bin:</p>
$$\\mathrm{WOE}_i = \\ln\\frac{\\%\\text{good}_i}{\\%\\text{bad}_i}, \\qquad \\mathrm{IV} = \\sum_i (\\%\\text{good}_i - \\%\\text{bad}_i)\\cdot \\mathrm{WOE}_i$$
<p>WOE replaces a category with a monotone log-odds contribution — exactly the scale a logistic scorecard wants (§2.4). IV summarises a feature's total separation. Conventional bands: &lt;0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, &gt;0.3 strong.</p>
${H.flag('Flag when you say this: those bands are industry rules of thumb, not test-derived thresholds — and IV is algebraically the symmetric KL (Jeffreys) divergence between the good and bad distributions, the same object as PSI in §2.18 with different inputs.')}

${H.lab('woe', 'WOE and IV, computed on an editable table', 'Move the counts and watch WOE, the IV contributions and the monotonicity check respond. The suspicious-IV warning fires above 0.5 for the reason the worked box explains.')}

${H.worked('worked WOE and IV — one feature, three bins', `
${H.table(['Bin', 'good', 'bad', '%good', '%bad', 'WOE', 'contrib'], [
      ['util &lt; 0.3', '4,500', '100', '0.50', '0.20', '+0.916', '0.275'],
      ['0.3–0.7', '3,150', '150', '0.35', '0.30', '+0.154', '0.008'],
      ['util &gt; 0.7', '1,350', '250', '0.15', '0.50', '−1.204', '0.421']
    ])}
<p>WOE for the top bin: $\\ln(0.15/0.50) = \\ln 0.30 = -1.204$. Its IV contribution: $(0.15-0.50)\\times(-1.204) = 0.421$ — positive, because both factors flip sign together. Summing: <b>IV = 0.275 + 0.008 + 0.421 = 0.704</b>.</p>
<p>By the conventional bands that is "very strong" — and <b>above 0.5 you should be suspicious rather than pleased</b>. Check that utilisation was measured strictly before the outcome window opened. Note also that the middle bin contributes almost nothing (0.008): it separates nobody, so merging it into a neighbour simplifies the scorecard at no cost. That is what binning decisions actually look like.</p>`)}

<h2><span class="sn">2.11.3</span> How to bin</h2>
<p>Equal-width bins are almost always wrong on skewed financial data (one bin holds 90% of the population). Equal-frequency (quantile) binning is the safe default. Supervised binning — a shallow decision tree on the single feature, or ChiMerge — finds cuts that actually separate the target, then <b>enforce monotonicity</b> by merging adjacent bins that break the trend. Two rules keep you out of trouble: every bin needs a minimum population (5% is the usual floor) and a minimum number of bads, or its WOE is noise; and the binning must be fitted inside the fold like any other learned transformation.</p>

<h2><span class="sn">2.11.4</span> Missing data</h2>
<p>Name the mechanism before choosing a fix. <b>MCAR</b> — missing completely at random, unconditionally: safe to impute. <b>MAR</b> — explained by observed features: model-based imputation is principled. <b>MNAR</b> — missingness depends on the unobserved value itself: the fact of missingness is informative, so add an indicator flag. Trees handle missing natively by learning a default direction; linear models do not. Median imputation plus a missing-indicator column is a defensible baseline that keeps the signal.</p>

<h2><span class="sn">2.11.5</span> Leakage</h2>
<p>Leakage is any feature computed using information that will not exist at prediction time: post-outcome fields, target-derived encodings fitted on everything, IDs correlated with time, aggregates computed over the full period, a "days since last payment" recorded after default. It shows up as an implausibly good validation score.</p>
${H.key('If a result is too good, look for leakage before you celebrate.')}

${H.lab('leak', 'The leakage checklist, as a live audit', 'Five questions, one worked scenario each. Answer them for a feature and the verdict assembles — this is the same interrogation a model validator will run on your work.')}

${H.probe([
      ['How do you prevent target-encoding leakage?', 'Fit the encoding within each training fold only, or use out-of-fold / ordered statistics.'],
      ['A feature has IV 0.5 — good news?', 'Nominally "strong", but suspiciously high IV is a classic leakage signature. Investigate before shipping.'],
      ['Which missingness mechanism needs an indicator?', 'MNAR — the fact of missingness carries information about the unobserved value.']
    ], 'Imputing before splitting. The imputer has then seen the validation rows.')}`,
    labs: {
      woe: function (host) {
        const st = Viz.controls(host, [
          { k: 'g1', label: 'bin 1 · goods', min: 100, max: 6000, step: 50, value: 4500, fmt: v => v.toLocaleString() },
          { k: 'b1', label: 'bin 1 · bads', min: 10, max: 600, step: 10, value: 100, fmt: v => v },
          { k: 'g2', label: 'bin 2 · goods', min: 100, max: 6000, step: 50, value: 3150, fmt: v => v.toLocaleString() },
          { k: 'b2', label: 'bin 2 · bads', min: 10, max: 600, step: 10, value: 150, fmt: v => v },
          { k: 'g3', label: 'bin 3 · goods', min: 100, max: 6000, step: 50, value: 1350, fmt: v => v.toLocaleString() },
          { k: 'b3', label: 'bin 3 · bads', min: 10, max: 600, step: 10, value: 250, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'iv', label: 'information value', cls: 'key' }, { k: 'band', label: 'conventional band' },
          { k: 'mono', label: 'monotone?' }, { k: 'flag', label: 'suspicious?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const goods = [st.g1, st.g2, st.g3], bads = [st.b1, st.b2, st.b3];
            const G = Num.sum(goods), B = Num.sum(bads);
            const rows = goods.map((g, i) => {
              const pg = g / G, pb = bads[i] / B;
              const woe = Math.log(Math.max(1e-9, pg) / Math.max(1e-9, pb));
              return { bin: ['util < 0.3', '0.3 – 0.7', 'util > 0.7'][i], g: g, b: bads[i], pg: pg, pb: pb, woe: woe, contrib: (pg - pb) * woe };
            });
            const IV = Num.sum(rows.map(r => r.contrib));
            // table
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const cols = ['bin', 'good', 'bad', '%good', '%bad', 'WOE', 'contrib'];
            const cw = Math.min(92, (w - 24) / 7);
            ctx.fillStyle = T.blue;
            cols.forEach((c, i) => { ctx.textAlign = i === 0 ? 'left' : 'right'; ctx.fillText(c, 14 + i * cw + (i ? cw - 8 : 0), 20); });
            rows.forEach((r, ri) => {
              const y = 44 + ri * 24;
              const vals = [r.bin, r.g.toLocaleString(), r.b.toLocaleString(), r.pg.toFixed(3), r.pb.toFixed(3), (r.woe >= 0 ? '+' : '') + r.woe.toFixed(3), r.contrib.toFixed(3)];
              vals.forEach((v, i) => {
                ctx.textAlign = i === 0 ? 'left' : 'right';
                ctx.fillStyle = i === 5 ? (r.woe >= 0 ? T.green : T.red) : T.text;
                ctx.fillText(v, 14 + i * cw + (i ? cw - 8 : 0), y);
              });
            });
            // WOE profile
            const by = 130, bh = 96, bw = Math.min(w - 40, 420);
            const mx = Math.max.apply(null, rows.map(r => Math.abs(r.woe))) || 1;
            ctx.strokeStyle = T.line; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(14, by + bh / 2); ctx.lineTo(14 + bw, by + bh / 2); ctx.stroke(); ctx.setLineDash([]);
            rows.forEach((r, i) => {
              const x = 14 + (i + .5) * (bw / 3) - 26;
              const hgt = (r.woe / mx) * (bh / 2 - 6);
              ctx.fillStyle = r.woe >= 0 ? T.blue : T.red;
              ctx.fillRect(x, by + bh / 2 - Math.max(0, hgt), 52, Math.abs(hgt));
              ctx.fillStyle = T.muted; ctx.font = '10px ui-sans-serif'; ctx.textAlign = 'center';
              ctx.fillText(r.bin, x + 26, by + bh + 12);
            });
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('WOE > 0 = better odds (good)', 14 + bw + 10, by + 16);
            ctx.fillText('WOE < 0 = worse odds (bad)', 14 + bw + 10, by + 34);
            const mono = (rows[0].woe > rows[1].woe && rows[1].woe > rows[2].woe) || (rows[0].woe < rows[1].woe && rows[1].woe < rows[2].woe);
            out({
              iv: IV.toFixed(3),
              band: IV < .02 ? 'useless' : IV < .1 ? 'weak' : IV < .3 ? 'medium' : 'strong',
              mono: mono ? 'yes' : 'no — rebin',
              flag: IV > .5 ? 'yes — check for leakage' : 'no'
            });
          }
        });
        Viz.note(host, 'A non-monotone WOE profile usually means the binning is wrong, not that the feature is wrong. And an IV above 0.5 on a behavioural feature is a leakage smell — verify the measurement date before you celebrate.');
      },

      leak: function (host) {
        const items = [
          { q: 'Could this field exist at decision time?', bad: '"days_since_last_payment" recorded after the default event', good: 'Balance as of the application timestamp', why: 'Anything stamped after the outcome window opens is not a feature, it is the answer.' },
          { q: 'Was any encoding fitted outside the fold?', bad: 'Target-mean encoding computed on the full dataset before CV', good: 'Out-of-fold target encoding, or CatBoost ordered statistics', why: 'The encoding carries the label; fitting it on everything lets each row see its own target.' },
          { q: 'Do aggregates respect the cutoff date?', bad: 'Customer mean transaction size over the whole panel', good: 'Mean over the 90 days strictly before the decision', why: 'Point-in-time correctness. The offline store must reconstruct what was known then (§2.18).' },
          { q: 'Is an ID standing in for time?', bad: 'account_id, which increments with signup date', good: 'Explicit tenure, with the ID dropped', why: 'Monotone IDs let the model read the calendar and exploit cohort effects that will not repeat.' },
          { q: 'Does the score look too good?', bad: 'AUC 0.98 on a problem where the industry ships 0.75', good: 'AUC in the plausible band, with an OOT check', why: 'Implausible performance is the loudest leakage signal there is.' }
        ];
        let idx = 0, answers = new Array(items.length).fill(null);
        const box = ML.el('div');
        host.appendChild(box);
        function render() {
          const it = items[idx];
          box.innerHTML = '<p class="boxtitle">leakage checklist · question ' + (idx + 1) + ' of ' + items.length + '</p>' +
            '<p style="font-size:1.05em;font-weight:600;font-family:var(--sans)">' + it.q + '</p>' +
            '<div class="qopts">' +
            '<button class="qopt" data-v="bad"><span class="k">A</span><span>' + it.bad + '</span></button>' +
            '<button class="qopt" data-v="good"><span class="k">B</span><span>' + it.good + '</span></button>' +
            '</div><p class="qwhy" style="display:none">' + it.why + '</p>' +
            '<div class="btnrow"><button class="btn" data-nav="prev">← previous</button><button class="btn primary" data-nav="next">next →</button></div>' +
            '<p class="small" id="leakScore" style="margin-top:10px"></p>';
          box.querySelectorAll('.qopt').forEach(b => b.addEventListener('click', () => {
            answers[idx] = b.getAttribute('data-v');
            box.querySelectorAll('.qopt').forEach(x => {
              x.classList.add('locked');
              if (x.getAttribute('data-v') === 'good') x.classList.add('right');
              else x.classList.add('wrong');
            });
            box.querySelector('.qwhy').style.display = '';
            score();
          }));
          box.querySelector('[data-nav="next"]').addEventListener('click', () => { idx = (idx + 1) % items.length; render(); });
          box.querySelector('[data-nav="prev"]').addEventListener('click', () => { idx = (idx - 1 + items.length) % items.length; render(); });
          score();
        }
        function score() {
          const done = answers.filter(a => a !== null).length;
          const ok = answers.filter(a => a === 'good').length;
          const el2 = box.querySelector('#leakScore');
          if (el2) el2.innerHTML = done ? ('<b>' + ok + ' of ' + done + '</b> answered with the point-in-time-safe option. A single "A" answer is enough to invalidate a validation score.') : 'Pick the option that is safe at decision time.';
        }
        render();
      }
    },
    quiz: [
      {
        q: 'You compute target-mean encoding on the full dataset, then run 5-fold CV. What is the consequence?',
        options: ['Nothing — the encoding is not a model', 'Each validation row influenced its own encoding, so CV is optimistically biased', 'The model will underfit', 'It only matters for high-cardinality features'],
        answer: 1,
        why: 'The encoding carries the label. Fit it inside the fold, or use out-of-fold / ordered statistics.'
      },
      {
        q: 'A behavioural feature scores IV = 0.62. The right response is…',
        options: ['Ship it — anything above 0.3 is strong', 'Investigate for leakage: verify the measurement date precedes the outcome window', 'Bin it more finely', 'Drop it — IV that high is always noise'],
        answer: 1,
        why: 'Suspiciously high IV is a classic leakage signature; conventional bands are rules of thumb, not tests.'
      },
      {
        q: 'Missingness that depends on the unobserved value itself is called…',
        options: ['MCAR', 'MAR', 'MNAR — and the missing-indicator is informative', 'Censoring'],
        answer: 2,
        why: 'MNAR: the fact of missingness carries signal, so add an explicit indicator rather than silently imputing it away.'
      }
    ],
    cards: [
      { q: 'WOE and IV formulas', a: '$\\mathrm{WOE}_i=\\ln(\\%good_i/\\%bad_i)$; $\\mathrm{IV}=\\sum_i(\\%good_i-\\%bad_i)\\mathrm{WOE}_i$. Bands <0.02 / 0.02–0.1 / 0.1–0.3 / >0.3 are conventions.' },
      { q: 'Leakage, defined', a: 'Any feature computed from information unavailable at prediction time. Signature: an implausibly good validation score.' },
      { q: 'Missingness mechanisms', a: 'MCAR (safe to impute), MAR (model-based imputation), MNAR (add an indicator — missingness is informative).' },
      { q: 'Binning rules', a: 'Quantile bins by default; enforce monotone WOE; minimum 5% population and a minimum bad count per bin; fit inside the fold.' }
    ]
  });
})();
