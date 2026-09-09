/* ============================================================
   PART 2 — Core & classical ML (2.1 – 2.6)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.1 */
  ML.section({
    id: 'supervised-setup', track: 'classical', num: '2.1',
    title: 'The supervised setup; loss vs metric; split discipline',
    lede: 'The vocabulary the rest of Part 2 assumes, and the one habit that separates a real evaluation from a comforting one.',
    rests: 'Rests on §1.4 for why a holdout means anything at all.',
    html: `
<h2><span class="sn">2.1.1</span> The setup</h2>
<p>You want $f: \\mathcal{X} \\to \\mathcal{Y}$ minimising expected loss $\\mathbb{E}_{(x,y)\\sim P}[\\ell(f(x), y)]$ — but $P$ is unavailable, so you minimise <b>empirical risk</b> on a sample and lean on concentration (§1.4) to argue the two are close.</p>

<h2><span class="sn">2.1.2</span> Loss versus metric</h2>
${H.key('The loss is what you optimise; the metric is what you report.')}
<p>The loss must be differentiable and well-behaved (logloss); the metric need not be (AUC, KS, Gini, cost per approval). They differ, and <b>the gap between them is where most production disappointment lives</b>: a model trained on logloss and judged on recall-at-3%-approval-rate is being asked two different questions.</p>
${H.table(['', 'Loss', 'Metric'], [
      ['Purpose', 'Guide the optimiser', 'Decide whether to ship'],
      ['Constraints', 'Differentiable, smooth, decomposes over examples', 'Anything the business can read'],
      ['Examples', 'logloss, MSE, hinge, Poisson deviance', 'AUC, KS, precision@k, £ per approval, lift'],
      ['When they diverge', 'You optimise the wrong thing efficiently', 'Fix: reweight, change the threshold (§2.13), or change the loss']
    ])}

<h2><span class="sn">2.1.3</span> The split, and the seal</h2>
<p><b>Train</b> fits parameters. <b>Validation</b> tunes hyperparameters and makes choices. <b>Test</b> is opened once, at the end, and never again. Every decision you make while looking at test — a threshold, a feature, a model family, a random seed — leaks information and converts an unbiased estimate into an optimistic one.</p>
${H.note('The test set is a bank vault, not a scratchpad. If you looked at it twice, say so out loud in the interview — candour here reads as seniority.')}

${H.lab('burn', 'Watch a test set burn', 'Every model here is pure noise — none has any real signal. Select the best one by test score and the reported number climbs anyway. That climb is the optimism you import every time you choose using the sealed segment.')}

<h2><span class="sn">2.1.4</span> Where the split rules change</h2>
${H.table(['Structure in the data', 'What a random split breaks', 'Correct split'], [
      ['Time', 'Trains on the future to predict the past', 'Out-of-time; rolling origin (§2.14)'],
      ['Entities with many rows', 'The model recognises the customer, not the pattern', 'Group k-fold on entity id'],
      ['Rare positives', 'A fold may contain almost no positives', 'Stratified k-fold'],
      ['Nested geography / hierarchy', 'Leaks neighbourhood-level information', 'Split on the coarsest unit you must generalise across']
    ])}

${H.probe([
      ['Loss or metric — which do you optimise?', 'You optimise the loss and report the metric; naming the gap between them is the senior answer.'],
      ['Why is a random split wrong for time series?', 'It trains on the future. Use out-of-time validation and rolling-origin backtests.']
    ], 'Tuning a threshold on the test set and then reporting test performance at that threshold.')}`,
    labs: {
      burn: function (host) {
        const st = Viz.controls(host, [
          { k: 'k', label: 'candidate models tried', min: 1, max: 200, step: 1, value: 30, fmt: v => v },
          { k: 'n', label: 'test-set size', min: 100, max: 5000, step: 100, value: 500, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'best', label: 'best test AUC seen', cls: 'bad' },
          { k: 'truth', label: 'true AUC of that model', cls: 'key' },
          { k: 'opt', label: 'optimism imported' },
          { k: 'fresh', label: 'on a fresh test set' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(97);
            // every model is pure noise: true AUC 0.5. Observed AUC ~ N(0.5, se)
            const se = 0.5 / Math.sqrt(st.n / 4);
            const obs = [], fresh = [];
            for (let i = 0; i < st.k; i++) { obs.push(0.5 + R.normal(0, se)); fresh.push(0.5 + R.normal(0, se)); }
            let bi = 0; obs.forEach((v, i) => { if (v > obs[bi]) bi = i; });
            const P = Viz.plot(ctx, w, h, { xd: [0, Math.max(2, st.k)], yd: [0.5 - 4 * se, 0.5 + 4 * se] })
              .frame({ xlabel: 'candidate model', ylabel: 'measured AUC on the sealed test set', yfmt: v => v.toFixed(3) });
            P.clip(() => {
              obs.forEach((v, i) => P.dots([[i + .5, v]], { r: 3.4, color: i === bi ? T.red : T.blue, alpha: i === bi ? 1 : .6, stroke: i === bi }));
              P.hline(.5, { color: T.green, dash: [5, 4], label: 'the truth: every model is noise, AUC = 0.500' });
              P.dots([[bi + .5, fresh[bi]]], { r: 5, color: T.amber, stroke: true });
              P.text(bi + .5, fresh[bi], '  same model, fresh test set', { color: T.amber, font: '11px ui-sans-serif' });
            });
            out({
              best: obs[bi].toFixed(3), truth: '0.500',
              opt: '+' + ((obs[bi] - .5) * 1000).toFixed(0) + ' bps',
              fresh: fresh[bi].toFixed(3)
            });
          }
        });
        Viz.note(host, 'The best-of-k maximum drifts upward as k grows — with 200 candidates on a 500-row test set you will "find" an AUC near 0.55 in pure noise. This is the same mechanism as multiple testing (§1.6), and it is why the number you report must come from a segment no decision has touched.');
      }
    },
    quiz: [
      {
        q: 'You compare 40 model configurations on the test set and report the best. The reported number is…',
        options: ['unbiased', 'optimistically biased by the selection itself', 'pessimistic', 'unbiased if you used cross-validation for training'],
        answer: 1,
        why: 'Choosing the maximum of 40 noisy estimates biases upward. Select on validation; report on a segment untouched by any decision.'
      },
      {
        q: 'A model is trained on logloss but the business judges recall at a fixed 3% approval rate. The cleanest first response is…',
        options: ['Retrain with recall as the loss', 'Keep the loss, move the operating threshold, and report at the business operating point', 'Resample the data until recall improves', 'Report AUC instead'],
        answer: 1,
        why: 'Recall at a fixed rate is a thresholding decision, not a training objective. Keep the well-behaved loss and choose the operating point deliberately (§2.13).'
      }
    ],
    cards: [
      { q: 'Loss vs metric', a: 'Loss = what you optimise (differentiable); metric = what you report (anything). The gap is where production disappointment lives.' },
      { q: 'The split rule', a: 'Train fits, validation chooses, test is opened once. Every decision made while looking at test imports optimism.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.2 */
  ML.section({
    id: 'bias-variance', track: 'classical', num: '2.2',
    title: 'Bias–variance, over/underfitting, double descent',
    lede: 'The frame for every model choice in this part — and the modern half of the picture that most candidates are a decade out of date on.',
    rests: 'Rests on §1.3.',
    html: `
<h2><span class="sn">2.2.1</span> The decomposition</h2>
<p>For squared error at a point $x$:</p>
$$\\mathbb{E}[(y-\\hat f(x))^2] = \\underbrace{(\\mathbb{E}[\\hat f(x)]-f(x))^2}_{\\text{bias}^2} + \\underbrace{\\mathrm{Var}(\\hat f(x))}_{\\text{variance}} + \\sigma^2$$

<h3>Derivation in three moves</h3>
${H.steps([
      'Write $y = f(x)+\\varepsilon$ with $\\mathbb{E}[\\varepsilon]=0$, $\\mathrm{Var}(\\varepsilon)=\\sigma^2$.',
      'Inside the squared error, add and subtract $\\mathbb{E}[\\hat f]$.',
      'Expand; the cross terms have expectation zero because $\\varepsilon$ is independent of $\\hat f$ and because $\\mathbb{E}[\\hat f - \\mathbb{E}\\hat f] = 0$.'
    ])}
<p>What survives is the three terms above. $\\sigma^2$ is irreducible: no model, however good, gets below the noise floor.</p>

${H.worked('worked number — when a deeper tree loses', `
<p>Current model: bias² = 0.04, variance = 0.02, σ² = 0.01 → expected error <b>0.07</b>.</p>
<p>Deepen the tree: bias² drops to 0.01 (it fits the structure better) but variance rises to 0.06 → expected error <b>0.08</b>. Worse, despite a better fit on paper.</p>
<p>This is the arithmetic behind every "why did my deeper model get worse" question. Bag the deep trees (variance → $\\rho\\sigma^2+(1-\\rho)\\sigma^2/B$, §1.3) and you keep the bias gain while giving back most of the variance — which is the entire idea of a random forest.</p>`)}

${H.lab('bv', 'Bias and variance, measured not asserted', 'Twenty independent training sets, one model family, fitted repeatedly. The grey curves are the individual fits; the blue is their average. Bias is the gap between the blue curve and the truth; variance is the spread of the grey ones. Move the degree and watch the two trade.')}

<h2><span class="sn">2.2.2</span> Double descent</h2>
<p>Push capacity past the <b>interpolation threshold</b> — parameters ≈ samples, where the model can fit the training set exactly — and test error rises to a peak, then <i>falls again</i> as capacity grows further. The classical U-curve is only the left half of the picture. In the over-parameterised regime, implicit regularization from the optimiser (SGD's preference for small-norm solutions) does the job that explicit capacity control used to. It does not refute bias–variance; it tells you the variance term is not monotone in parameter count.</p>

${H.lab('dd', 'The U-curve and its second half', 'Real fits: ridge-regularised random-feature regression at increasing width, evaluated on held-out data. The peak at parameters ≈ samples is real and reproducible; add a little ridge and watch it soften — which is the modern reading of what implicit regularization does.')}

${H.probe([
      ['High variance — what do you do?', 'More data, stronger regularization, a simpler hypothesis class, or bagging. In that order of preference if data is available.'],
      ['Does double descent kill bias–variance?', 'No — it extends it into the over-parameterised regime where implicit regularization dominates.']
    ], 'Assuming more capacity always overfits. That belief is a decade out of date.')}`,
    labs: {
      bv: function (host) {
        const st = Viz.controls(host, [
          { k: 'deg', label: 'polynomial degree', min: 0, max: 12, step: 1, value: 3, fmt: v => v },
          { k: 'n', label: 'points per training set', min: 8, max: 60, step: 2, value: 16, fmt: v => v },
          { k: 'noise', label: 'noise σ', min: .05, max: 1, step: .05, value: .35, fmt: v => v.toFixed(2) },
          { k: 'lam', label: 'ridge λ', min: 0, max: 2, step: .02, value: 0, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'bias', label: 'bias²', cls: 'key' }, { k: 'var', label: 'variance' },
          { k: 'noise', label: 'σ² (irreducible)' }, { k: 'tot', label: 'total expected error' }
        ]);
        const truth = x => Math.sin(1.3 * x) * 1.4 + .25 * x;
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const REP = 20;
            const grid = []; for (let i = 0; i <= 60; i++) grid.push(-3 + 6 * i / 60);
            const curves = [];
            for (let r = 0; r < REP; r++) {
              const R = Num.rng(1000 + r * 37);
              const xs = [], ys = [];
              for (let i = 0; i < st.n; i++) { const x = -3 + 6 * R(); xs.push(x); ys.push(truth(x) + R.normal(0, st.noise)); }
              const beta = Num.ridgeFit(Num.polyDesign(xs, st.deg), ys, st.lam + 1e-8);
              curves.push(grid.map(x => Num.dot(beta, Num.polyDesign([x], st.deg)[0])));
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3.5, 3.5] }).frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              curves.forEach(c => P.line(grid.map((x, i) => [x, c[i]]), { color: T.faint, width: 1, alpha: .45 }));
              const avg = grid.map((_, i) => Num.mean(curves.map(c => c[i])));
              P.line(grid.map((x, i) => [x, avg[i]]), { color: T.blue, width: 2.8 });
              P.fn(truth, { color: T.green, width: 2.2, dash: [6, 4] });
            });
            const avg = grid.map((_, i) => Num.mean(curves.map(c => c[i])));
            const bias2 = Num.mean(grid.map((x, i) => (avg[i] - truth(x)) ** 2));
            const varr = Num.mean(grid.map((_, i) => Num.variance(curves.map(c => c[i]))));
            out({
              bias: bias2.toFixed(3), var: varr.toFixed(3), noise: (st.noise * st.noise).toFixed(3),
              tot: (bias2 + varr + st.noise * st.noise).toFixed(3)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().green, t: 'truth f(x)' }, { c: Viz.theme().blue, t: 'average fit E[f̂]' }, { c: Viz.theme().faint, t: '20 individual fits' }]);
        Viz.note(host, 'Degree 0–1 → the blue average sits far from green (bias) and the grey curves are tight (low variance). Degree 10+ → blue tracks green but grey scatters wildly. Add ridge λ and watch variance collapse while bias creeps up: that is regularization, measured.');
      },

      dd: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'training samples n', min: 15, max: 60, step: 5, value: 30, fmt: v => v },
          { k: 'lam', label: 'ridge λ', min: 0, max: 1, step: .002, value: 0, fmt: v => v.toFixed(3) },
          { k: 'noise', label: 'noise σ', min: .05, max: .8, step: .05, value: .3, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'peak', label: 'peak test error at', cls: 'key' }, { k: 'best', label: 'best width' }, { k: 'final', label: 'error at max width' }
        ]);
        const S = Viz.surface(host, {
          height: 310,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(5);
            const f = x => Math.sin(2.2 * x) + .4 * x;
            const xs = [], ys = [];
            for (let i = 0; i < st.n; i++) { const x = -2 + 4 * R(); xs.push(x); ys.push(f(x) + R.normal(0, st.noise)); }
            const xte = [], yte = [];
            for (let i = 0; i < 300; i++) { const x = -2 + 4 * R(); xte.push(x); yte.push(f(x)); }
            // random-feature model: features = cos(w x + b), width p
            const widths = [];
            for (let p = 1; p <= 120; p += 2) widths.push(p);
            const RW = Num.rng(31);
            const Wf = [], Bf = [];
            for (let j = 0; j < 130; j++) { Wf.push(RW.normal(0, 2.2)); Bf.push(RW() * 6.28); }
            const curveTr = [], curveTe = [];
            widths.forEach(p => {
              const Phi = xs.map(x => Array.from({ length: p }, (_, j) => Math.cos(Wf[j] * x + Bf[j])));
              const beta = Num.ridgeFit(Phi, ys, st.lam + 1e-9);
              const predTr = Phi.map(r => Num.dot(r, beta));
              const PhiTe = xte.map(x => Array.from({ length: p }, (_, j) => Math.cos(Wf[j] * x + Bf[j])));
              const predTe = PhiTe.map(r => Num.dot(r, beta));
              curveTr.push([p, Math.min(4, Num.mean(ys.map((v, i) => (v - predTr[i]) ** 2)))]);
              curveTe.push([p, Math.min(4, Num.mean(yte.map((v, i) => (v - predTe[i]) ** 2)))]);
            });
            const P = Viz.plot(ctx, w, h, { xd: [1, 120], yd: [0, Math.min(3, Math.max.apply(null, curveTe.map(c => c[1]))) * 1.05] })
              .frame({ xlabel: 'model width (number of random features)', ylabel: 'mean squared error' });
            P.clip(() => {
              P.line(curveTr, { color: T.faint, width: 1.6, dash: [5, 4] });
              P.line(curveTe, { color: T.blue, width: 2.6 });
              P.vline(st.n, { color: T.red, label: 'params ≈ n — the interpolation threshold' });
            });
            let peak = curveTe[0], best = curveTe[0];
            curveTe.forEach(c => { if (c[1] > peak[1]) peak = c; if (c[1] < best[1]) best = c; });
            out({ peak: 'width ' + peak[0], best: 'width ' + best[0], final: curveTe[curveTe.length - 1][1].toFixed(3) });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'test error' }, { c: Viz.theme().faint, t: 'train error' }, { c: Viz.theme().red, t: 'interpolation threshold' }]);
        Viz.note(host, 'Set λ = 0 and the spike at width ≈ n is dramatic; add a little ridge and it flattens. That is the whole modern story: at the interpolation threshold the minimum-norm solution is badly behaved, and either explicit regularization or the optimiser’s implicit bias tames it.');
      }
    },
    quiz: [
      {
        q: 'bias² = 0.04, variance = 0.02, σ² = 0.01. You deepen the model: bias² → 0.01, variance → 0.06. What happened to expected error?',
        options: ['Fell from 0.07 to 0.05', 'Rose from 0.07 to 0.08', 'Unchanged', 'Cannot tell without more data'],
        answer: 1,
        why: '0.01 + 0.06 + 0.01 = 0.08. Better fit, worse generalisation — and bagging is the standard way to keep the bias gain while returning the variance.'
      },
      {
        q: 'Test error peaks when parameters ≈ samples and then falls again with more parameters. This is…',
        options: ['a bug in the evaluation', 'double descent — the over-parameterised regime where implicit regularization dominates', 'evidence that bias–variance is wrong', 'label leakage'],
        answer: 1,
        why: 'The classical U-curve is the left half. Beyond interpolation, minimum-norm solutions found by SGD generalise increasingly well.'
      }
    ],
    cards: [
      { q: 'Bias–variance decomposition', a: '$\\mathbb{E}[(y-\\hat f)^2] = \\text{bias}^2 + \\text{variance} + \\sigma^2$; $\\sigma^2$ is the irreducible noise floor.' },
      { q: 'High variance — the fixes in order', a: 'More data → stronger regularization → simpler class → bagging.' },
      { q: 'Double descent', a: 'Test error peaks at the interpolation threshold (params ≈ n) then descends again; variance is not monotone in parameter count.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.3 */
  ML.section({
    id: 'regularization', track: 'classical', num: '2.3',
    title: 'Regularization: L1, L2, elastic net',
    lede: 'Rests on MAP (§1.5); enables the sparse, defensible scorecards of §2.11. Two arguments for why L1 is sparse — one geometric, one analytic, and the analytic one is the better answer.',
    html: `
<h2><span class="sn">2.3.1</span> The three penalties</h2>
<p><b>L2 (ridge)</b> adds $\\lambda\\|w\\|_2^2$: shrinks every coefficient smoothly toward zero, keeps all features, handles collinearity by splitting weight between correlated columns. <b>L1 (lasso)</b> adds $\\lambda\\|w\\|_1$: drives coefficients to <i>exactly</i> zero, performing selection. <b>Elastic net</b> mixes them, which is the right default when features are correlated <i>and</i> you want sparsity — because pure lasso picks one of a correlated group arbitrarily.</p>

<h2><span class="sn">2.3.2</span> Why L1 is sparse — the geometric argument</h2>
<p>In constrained form you minimise the loss subject to a norm ball. The L1 ball is a diamond with vertices <i>on the axes</i>; the L2 ball is smooth. Elliptical loss contours expanding from the unconstrained optimum first touch the diamond at a vertex with overwhelming probability — and a vertex is a point where some coordinates are exactly zero. A smooth ball has no such privileged points, so tangency almost never lands on an axis.</p>

${H.lab('geom', 'Diamond versus circle — drag the optimum', 'Move the unconstrained optimum (the red dot) and watch where the expanding contours first touch the constraint set. On L1 the touch point snaps to a vertex over most of the plane; on L2 it never does.')}

<h2><span class="sn">2.3.3</span> The analytic argument (the better answer)</h2>
<p>L1's subgradient contributes a constant pull $\\lambda\\,\\mathrm{sign}(w)$ that <b>does not vanish as $w \\to 0$</b>, so it can hold a coordinate pinned at zero whenever the loss gradient there is smaller than $\\lambda$. L2's pull is $2\\lambda w$, which vanishes as $w \\to 0$ — it can shrink forever without arriving.</p>
${H.key('Constant pull versus proportional pull: that is the whole story.')}
<p>The soft-thresholding operator makes it concrete — the lasso coordinate update is $w_j \\leftarrow \\mathcal{S}_\\lambda(\\rho_j) = \\mathrm{sign}(\\rho_j)\\max(0, |\\rho_j| - \\lambda)$, and that $\\max(0, \\cdot)$ is where exact zeros come from.</p>

${H.lab('thresh', 'Soft threshold vs proportional shrinkage', 'The update rule each penalty implies, drawn. The flat segment in the middle of the L1 curve is the dead zone where coefficients are set to exactly zero — L2 has no such segment.')}

<h2><span class="sn">2.3.4</span> Practical guidance</h2>
${H.table(['Situation', 'Choice', 'Why'], [
      ['Many correlated features, want stability', 'Ridge', 'Splits weight rather than choosing arbitrarily'],
      ['Need a short, defensible feature list', 'Lasso or elastic net', 'Exact zeros are a selection you can present'],
      ['Correlated groups AND sparsity', 'Elastic net', 'L1 selects, L2 stabilises which member is selected'],
      ['Trees / boosting', 'Neither, mostly', 'Use depth, min_child_weight, subsampling, and the λ/γ inside the split score (§2.8)'],
      ['Neural networks', 'Weight decay (AdamW), dropout, early stopping', 'Same idea, different implementation (§3.5)']
    ])}
${H.flag('Always standardise before penalising. A penalty on raw coefficients punishes features measured in small units and ignores features measured in large ones — the regularizer becomes a function of your unit choices.')}

${H.probe([
      ['Why does L1 select features and L2 not?', 'The diamond’s vertices sit on the axes, and L1’s constant subgradient pins coordinates at zero; L2’s pull vanishes at zero.'],
      ['When is elastic net the right default?', 'Correlated features plus a need for sparsity — lasso alone is unstable across correlated columns.']
    ], 'Claiming L2 gives sparsity, or that lasso is "just better". Lasso is unstable across correlated features; that is exactly what elastic net fixes.')}`,
    labs: {
      geom: function (host) {
        let opt = [1.6, .75];
        let drag = false;
        const st = Viz.controls(host, [
          { k: 'kind', label: 'penalty', type: 'buttons', value: 'l1', options: [{ v: 'l1', t: 'L1 — diamond' }, { v: 'l2', t: 'L2 — circle' }, { v: 'en', t: 'elastic net' }] },
          { k: 't', label: 'budget size', min: .2, max: 2, step: .02, value: .9, fmt: v => v.toFixed(2) },
          { k: 'rho', label: 'correlation of the loss contours', min: -.9, max: .9, step: .05, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'w1', label: 'w₁ at optimum', cls: 'key' }, { k: 'w2', label: 'w₂ at optimum' }, { k: 'zero', label: 'exactly zero?' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-2.4, 2.4], yd: [-1.9, 1.9] }).frame({ xlabel: 'w₁', ylabel: 'w₂' });
            const A = [[1, st.rho], [st.rho, 1]];
            const loss = (a, b) => {
              const d = [a - opt[0], b - opt[1]];
              return d[0] * (A[0][0] * d[0] + A[0][1] * d[1]) + d[1] * (A[1][0] * d[0] + A[1][1] * d[1]);
            };
            const pen = (a, b) => st.kind === 'l1' ? Math.abs(a) + Math.abs(b)
              : st.kind === 'l2' ? Math.sqrt(a * a + b * b)
              : .5 * (Math.abs(a) + Math.abs(b)) + .5 * Math.sqrt(a * a + b * b);
            // constraint boundary
            const bd = [];
            for (let th = 0; th <= 6.3; th += .01) {
              const dx = Math.cos(th), dy = Math.sin(th);
              let lo = 0, hi = 6;
              for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (pen(dx * mid, dy * mid) > st.t) hi = mid; else lo = mid; }
              bd.push([dx * lo, dy * lo]);
            }
            // constrained optimum by scanning the boundary
            let best = bd[0], bl = 1e9;
            bd.forEach(p => { const v = loss(p[0], p[1]); if (v < bl) { bl = v; best = p; } });
            P.clip(() => {
              P.contours(loss, [bl, bl * 1.6, bl * 2.6, bl * 4, bl * 6, bl * 9], { color: T.red, alpha: .5 });
              P.contours(loss, [bl], { color: T.red, width: 2, alpha: .9 });
              P.line(bd.concat([bd[0]]), { color: T.blue, width: 2.4 });
              P.dots([opt], { r: 5, color: T.red, stroke: true });
              P.dots([best], { r: 6, color: T.green, stroke: true });
              P.line([[-2.4, 0], [2.4, 0]], { color: T.faint, width: 1, alpha: .6 });
              P.line([[0, -1.9], [0, 1.9]], { color: T.faint, width: 1, alpha: .6 });
            });
            P.text(opt[0], opt[1], '  unconstrained optimum', { color: T.red, font: '11px ui-sans-serif' });
            P.text(best[0], best[1], '  constrained solution', { color: T.green, font: '11px ui-sans-serif' });
            out({
              w1: best[0].toFixed(3), w2: best[1].toFixed(3),
              zero: (Math.abs(best[0]) < .02 || Math.abs(best[1]) < .02) ? 'yes — a vertex' : 'no'
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          if (e.type === 'down') drag = true;
          if (e.type === 'up') drag = false;
          if (drag && e.down) { opt = [P.ix(e.x), P.iy(e.y)]; S.redraw(); }
        });
        Viz.note(host, 'Drag the red dot around the plane on L1: the green solution snaps to an axis for most positions — that is feature selection happening geometrically. Switch to L2 and it essentially never does.');
      },

      thresh: function (host) {
        const st = Viz.controls(host, [
          { k: 'lam', label: 'λ', min: 0, max: 1.5, step: .02, value: .5, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3] })
              .frame({ xlabel: 'least-squares coefficient ρ (before penalty)', ylabel: 'coefficient after penalty' });
            P.clip(() => {
              P.fn(x => x, { color: T.faint, width: 1.2, dash: [4, 4] });
              P.fn(x => Math.sign(x) * Math.max(0, Math.abs(x) - st.lam), { color: T.blue, width: 2.8 });
              P.fn(x => x / (1 + st.lam), { color: T.red, width: 2.4, dash: [7, 4] });
              if (st.lam > 0) {
                ctx.globalAlpha = .12; ctx.fillStyle = T.blue;
                ctx.fillRect(P.x(-st.lam), P.pad.t, P.x(st.lam) - P.x(-st.lam), P.ph);
                ctx.globalAlpha = 1;
              }
            });
            ctx.fillStyle = T.blue; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('L1 soft threshold — flat dead zone → exact zeros', 60, 14);
            ctx.fillStyle = T.red; ctx.fillText('L2 proportional shrinkage — never reaches zero', 60, 30);
          }
        });
      }
    },
    quiz: [
      {
        q: 'Which statement about lasso is accurate?',
        options: ['It always outperforms ridge', 'Its constant subgradient can pin coefficients at exactly zero; it is unstable across correlated features', 'It is equivalent to ridge with a different λ', 'It requires standardised targets, not features'],
        answer: 1,
        why: 'Constant pull vs proportional pull gives exact zeros; the instability across correlated groups is precisely what elastic net addresses.'
      },
      {
        q: 'You forget to standardise features before applying an L2 penalty. What goes wrong?',
        options: ['Nothing', 'The penalty depends on the units of each feature, punishing small-unit features arbitrarily', 'The model becomes non-convex', 'Coefficients become exactly zero'],
        answer: 1,
        why: 'A coefficient on a feature measured in pounds is numerically tiny compared to one measured in millions; the penalty then reflects unit choices rather than importance.'
      }
    ],
    cards: [
      { q: 'Why is L1 sparse? (analytic)', a: 'Its subgradient is a constant $\\lambda\\,\\mathrm{sign}(w)$ that does not vanish at zero, so it pins coordinates there. L2’s $2\\lambda w$ vanishes.' },
      { q: 'Soft-thresholding operator', a: '$\\mathcal{S}_\\lambda(\\rho)=\\mathrm{sign}(\\rho)\\max(0,|\\rho|-\\lambda)$ — the lasso coordinate update.' },
      { q: 'When elastic net', a: 'Correlated features plus a need for sparsity; L1 alone picks one of a correlated group arbitrarily.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.4 */
  ML.section({
    id: 'linear-logistic', track: 'classical', num: '2.4',
    title: 'Linear regression, logistic regression, GLMs',
    lede: 'The backbone models: they feed calibration (§2.12), scorecards (§2.11), and the whole idea of a link function. Both derivations are two lines and both get asked.',
    html: `
<h2><span class="sn">2.4.1</span> Normal equations, derived</h2>
<p>Minimise $\\|y - Xw\\|^2$. The gradient is $-2X^\\mathsf{T}(y - Xw)$; set it to zero:</p>
$$X^\\mathsf{T}Xw = X^\\mathsf{T}y \\quad\\Longrightarrow\\quad w = (X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$$
<p>Ridge adds $\\lambda I$ inside the inverse, $w = (X^\\mathsf{T}X + \\lambda I)^{-1}X^\\mathsf{T}y$, which does double duty: it regularises <i>and</i> it makes the matrix invertible when columns are collinear or $p > n$. That is worth saying explicitly — it is the cleanest example of one term solving a numerical problem and a statistical one at once. (Every eigenvalue of $X^\\mathsf{T}X$ is raised by exactly λ, so the smallest ones — the unstable directions — are shrunk hardest.)</p>

<h2><span class="sn">2.4.2</span> Logistic gradient, derived</h2>
<p>With $p = \\sigma(w^\\mathsf{T}x)$ and $\\sigma' = \\sigma(1-\\sigma)$, differentiate the cross-entropy for one example:</p>
$$\\frac{\\partial}{\\partial w}\\big[-y\\log p-(1-y)\\log(1-p)\\big] = \\left(-\\frac{y}{p}+\\frac{1-y}{1-p}\\right)p(1-p)\\,x = (p-y)\\,x$$
${H.key('The update is prediction minus label, times the feature.')}
<p>Structurally identical to linear regression's residual update, and the same form you meet again in softmax cross-entropy (§4.9). The objective is convex (§1.9), so the optimum is global and any reasonable optimiser finds it.</p>

${H.lab('logistic', 'Logistic regression, trained live', 'Real gradient descent on data you can edit. Click to add points, watch the boundary move, the loss fall, and the weights converge. Turn on L2 and watch the boundary stiffen.')}

<h2><span class="sn">2.4.3</span> GLMs generalise the recipe</h2>
<p>Choose a distribution from the exponential family and a <b>link function</b> connecting $\\mathbb{E}[y]$ to the linear predictor: logit for Bernoulli, log for Poisson, identity for Gaussian. Coefficient interpretation follows the link — in logistic regression a coefficient is a log-odds change, so $e^{w_j}$ is an <b>odds ratio</b>. That sentence is what a credit interviewer wants, because scorecard points are just scaled log-odds.</p>

${H.worked('worked number — from log-odds to scorecard points', `
<p>A credit score is an affine map of the log-odds, calibrated by two conventions: an <b>anchor</b> (a chosen score at chosen odds) and the <b>PDO</b> — points to double the odds.</p>
$$\\text{factor}=\\frac{\\text{PDO}}{\\ln 2},\\qquad \\text{offset}=\\text{anchor}-\\text{factor}\\cdot\\ln(\\text{anchor odds})$$
<p>With PDO = 20 and 600 points at 50:1 good:bad odds: factor = 20 ÷ 0.693 = <b>28.85</b>; offset = 600 − 28.85 × ln(50) = 600 − 28.85 × 3.912 = <b>487.1</b>.</p>
<p>An applicant whose model log-odds are 2.5 scores 487.1 + 28.85 × 2.5 = <b>559</b>. One with log-odds 4.2 scores 608. Sanity check that proves you understand it: 20 more points must double the odds — 559 → 579 moves log-odds from 2.5 to 3.19, and $e^{0.693} = 2$. ✓</p>
<p>Because the map is affine and WOE-binned features enter linearly (§2.11), each bin's contribution can be printed as whole points — which is the entire reason regulated lending still ships logistic scorecards when gradient boosting scores better: <mark>the model is auditable by arithmetic, not by a SHAP library.</mark></p>`)}

${H.lab('scorecard', 'Scorecard points calculator', 'Move PDO and the anchor and watch the mapping change. The bottom strip is a real applicant’s points breakdown: each bin contributes whole points that a committee can add up by hand.')}

${H.probe([
      ['Logistic gradient?', '$(\\hat p - y)x$. Derive it in two lines using $\\sigma\' = \\sigma(1-\\sigma)$.'],
      ['Why not fit a line to probabilities?', 'Unbounded outputs and the wrong noise model; the logit link and Bernoulli likelihood fix both.'],
      ['What does ridge do to $X^\\mathsf{T}X$?', 'Adds λ to every eigenvalue, guaranteeing invertibility and shrinking low-variance directions hardest.']
    ])}`,
    labs: {
      logistic: function (host) {
        let data = Num.dataset('blobs', 60, .35, 12);
        let model = null, running = false, hist = [];
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate η', min: .05, max: 3, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'l2', label: 'L2 strength', min: 0, max: 1, step: .01, value: 0, fmt: v => v.toFixed(2) },
          { k: 'cls', label: 'class to add on click', type: 'buttons', value: '1', options: [{ v: '0', t: 'class 0' }, { v: '1', t: 'class 1' }] }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'loss', label: 'logloss', cls: 'key' }, { k: 'acc', label: 'train accuracy' },
          { k: 'w', label: 'weights' }, { k: 'it', label: 'iterations' }
        ]);
        function reset() {
          model = Num.logistic(data.X, data.y, { lr: st.lr, l2: st.l2 });
          hist = []; S.redraw();
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            if (model) {
              P.clip(() => Labs.boundary(P, (x, y) => model.predict([x, y]), { step: 4 }));
            }
            P.clip(() => Labs.points(P, data.X, data.y));
            if (hist.length > 2) {
              // inset loss curve
              const ix = P.pad.l + 8, iy = P.pad.t + 8, iw = Math.min(150, P.pw * .34), ih = 52;
              ctx.fillStyle = T.paper; ctx.globalAlpha = .85; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(ix, iy, iw, ih);
              const mx = Math.max.apply(null, hist);
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              hist.forEach((v, i) => {
                const X = ix + iw * i / (hist.length - 1), Y = iy + ih - ih * (v / (mx || 1));
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('logloss', ix + 4, iy + 3);
            }
            if (model) {
              const acc = Num.mean(data.X.map((x, i) => (model.predict(x) > .5 ? 1 : 0) === data.y[i] ? 1 : 0));
              out({
                loss: model.loss().toFixed(4), acc: (acc * 100).toFixed(1) + '%',
                w: '[' + model.w.map(v => v.toFixed(2)).join(', ') + ']', it: hist.length
              });
            }
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P || e.type !== 'down') return;
          data.X.push([P.ix(e.x), P.iy(e.y)]); data.y.push(+st.cls);
          reset();
        });
        Viz.buttons(host, [
          { label: 'Train 200 steps', primary: true, on: () => { for (let i = 0; i < 200; i++) { model.step(1); hist.push(model.loss()); } S.redraw(); } },
          { label: 'Animate', on: () => {
            running = !running;
            const t = setInterval(() => { if (!running) return clearInterval(t); model.step(3); hist.push(model.loss()); S.redraw(); }, 40);
            ML.onCleanup(() => clearInterval(t));
          } },
          { label: 'Reset weights', on: reset },
          { label: 'New data: circles', on: () => { data = Num.dataset('circles', 80, .25, 4); reset(); } },
          { label: 'New data: blobs', on: () => { data = Num.dataset('blobs', 60, .35, 12); reset(); } }
        ]);
        reset();
        Viz.note(host, 'On the circles dataset the boundary can never fit — logistic regression is linear in the features you give it. That failure is the motivation for kernels (§2.6) and for hidden layers (§3.1): both manufacture the features that make it linear again.');
      },

      scorecard: function (host) {
        const st = Viz.controls(host, [
          { k: 'pdo', label: 'PDO — points to double the odds', min: 10, max: 60, step: 1, value: 20, fmt: v => v },
          { k: 'anchor', label: 'anchor score', min: 400, max: 800, step: 5, value: 600, fmt: v => v },
          { k: 'odds', label: 'anchor odds (good:bad)', min: 5, max: 100, step: 1, value: 50, fmt: v => v + ':1' },
          { k: 'lo', label: 'applicant log-odds', min: -2, max: 8, step: .1, value: 2.5, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'factor', label: 'factor', cls: 'key' }, { k: 'offset', label: 'offset' },
          { k: 'score', label: 'applicant score' }, { k: 'check', label: '+PDO doubles odds?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const factor = st.pdo / Math.log(2);
            const offset = st.anchor - factor * Math.log(st.odds);
            const score = lo => offset + factor * lo;
            const P = Viz.plot(ctx, w, h, { xd: [-2, 8], yd: [Math.min(300, score(-2)) - 20, score(8) + 20] })
              .frame({ xlabel: 'model log-odds', ylabel: 'score', yfmt: v => v.toFixed(0) });
            P.clip(() => {
              P.fn(score, { color: T.blue, width: 2.6 });
              P.vline(st.lo, { color: T.text, dash: [4, 4] });
              P.dots([[st.lo, score(st.lo)]], { r: 6, color: T.red, stroke: true });
              P.line([[st.lo, score(st.lo)], [st.lo + Math.log(2), score(st.lo + Math.log(2))]], { color: T.green, width: 3 });
              P.text(st.lo + Math.log(2), score(st.lo + Math.log(2)), '  +' + st.pdo + ' pts = 2× the odds', { color: T.green, font: '11px ui-sans-serif' });
            });
            // points breakdown strip
            const bins = [['utilisation < 0.3', 0.916], ['tenure ≥ 5 yrs', 0.42], ['enquiries ≤ 2', 0.31], ['recent arrears', -0.83]];
            const bx = P.pad.l + 10, by = h - 96;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('a committee-readable breakdown (WOE bins × factor):', bx, by - 12);
            let total = offset;
            bins.forEach((b, i) => {
              const pts = b[1] * factor;
              total += pts;
              ctx.fillStyle = T.text; ctx.fillText(b[0], bx, by + 12 + i * 17);
              ctx.fillStyle = pts >= 0 ? T.green : T.red; ctx.textAlign = 'right';
              ctx.fillText((pts >= 0 ? '+' : '') + pts.toFixed(0), bx + 250, by + 12 + i * 17);
              ctx.textAlign = 'left';
            });
            ctx.fillStyle = T.muted; ctx.fillText('base (offset)', bx, by + 12 + bins.length * 17);
            ctx.textAlign = 'right'; ctx.fillText(offset.toFixed(0), bx + 250, by + 12 + bins.length * 17);
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace';
            ctx.textAlign = 'left'; ctx.fillText('total', bx, by + 14 + (bins.length + 1) * 17);
            ctx.textAlign = 'right'; ctx.fillText(total.toFixed(0), bx + 250, by + 14 + (bins.length + 1) * 17);
            out({
              factor: factor.toFixed(2), offset: offset.toFixed(1), score: score(st.lo).toFixed(0),
              check: (score(st.lo + Math.log(2)) - score(st.lo)).toFixed(1) + ' pts ✓'
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'The logistic regression gradient for one example is…',
        options: ['$(y-\\hat p)x^2$', '$(\\hat p - y)x$', '$\\hat p(1-\\hat p)x$', '$-y\\log\\hat p$'],
        answer: 1,
        why: 'Prediction minus label times the feature — the σ′ = σ(1−σ) factor cancels exactly against the derivative of the log terms.'
      },
      {
        q: 'Ridge adds λI inside the inverse. Besides regularising, what does that achieve?',
        options: ['It centres the features', 'It guarantees invertibility of $X^\\mathsf{T}X$ under collinearity or p > n', 'It makes the problem non-convex', 'It removes the intercept'],
        answer: 1,
        why: 'Every eigenvalue rises by λ, so the matrix cannot be singular — one term solving a numerical and a statistical problem simultaneously.'
      },
      {
        q: 'A scorecard uses PDO = 20 anchored at 600 = 50:1 odds. An applicant scores 620. Their odds are…',
        options: ['25:1', '50:1', '100:1', '200:1'],
        answer: 2,
        why: 'Twenty points is one doubling by construction, so 50:1 → 100:1.'
      }
    ],
    cards: [
      { q: 'Normal equations', a: '$w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge: $(X^\\mathsf{T}X+\\lambda I)^{-1}X^\\mathsf{T}y$.' },
      { q: 'Logistic gradient', a: '$(\\hat p-y)x$ — prediction minus label times feature.' },
      { q: 'Scorecard conversion', a: 'factor = PDO/ln2, offset = anchor − factor·ln(anchor odds); score = offset + factor × log-odds.' },
      { q: 'What is $e^{w_j}$ in logistic regression?', a: 'The odds ratio for a one-unit change in feature $j$.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.5 */
  ML.section({
    id: 'knn-nb', track: 'classical', num: '2.5',
    title: 'k-NN and Naive Bayes',
    lede: 'Naive Bayes rests directly on §1.1; k-NN is the cleanest possible illustration of the §2.2 tradeoff — one hyperparameter moves you from pure variance to pure bias.',
    html: `
<h2><span class="sn">2.5.1</span> k-NN</h2>
<p>No training phase: store the data, predict by majority vote or average over the $k$ nearest points. $k=1$ is a pure-variance model with a jagged boundary and zero training error; large $k$ is a high-bias model with a smooth boundary. It requires feature scaling (distance is meaningless across mixed units) and it collapses in high dimensions, where the <b>curse of dimensionality</b> makes all pairwise distances nearly equal, so "nearest" stops meaning anything.</p>

${H.lab('knn', 'k = 1 versus k = 25, on the same data', 'One hyperparameter, two completely different models. Watch the boundary go from jagged islands around individual points to a single smooth sweep — the left is variance, the right is bias, and §2.2 is that picture.')}

${H.lab('curse', 'The curse of dimensionality, measured', 'Distances between random points in d dimensions, computed here. As d grows the ratio of nearest to farthest approaches 1 — the geometric fact that kills distance-based methods in high dimensions.')}

<h2><span class="sn">2.5.2</span> Naive Bayes</h2>
<p>Predicts $\\arg\\max_c P(c)\\prod_t P(x_t\\mid c)$, assuming features are conditionally independent given the class. The assumption is almost always false, yet the classifier is strong for text: it only needs the <i>ranking</i> of class scores to be right, and errors in the dependent factors often cancel. Laplace smoothing prevents a single unseen token from zeroing an entire product.</p>

${H.table(['Variant', 'Uses', 'Best for'], [
      ['Multinomial', 'token counts', 'text, the default'],
      ['Bernoulli', 'presence / absence', 'very short documents, where a word appearing twice means little'],
      ['Gaussian', 'continuous features, normal within each class', 'a fast baseline; badly wrong on skewed financial features unless transformed']
    ])}
<p>And work in log space always: multiplying a thousand probabilities underflows to zero in float64, so sum $\\log P$ instead.</p>

${H.worked('worked Naive Bayes — three words, two classes', `
<p>Priors $P(\\text{spam})=0.3$, $P(\\text{ham})=0.7$. Word likelihoods "offer" 0.20 / 0.02, "meeting" 0.01 / 0.15, "free" 0.25 / 0.03 (spam / ham). Document = {offer, free}.</p>
${H.code(`spam score = 0.3 × 0.20 × 0.25 = 0.0150
ham  score = 0.7 × 0.02 × 0.03 = 0.00042
P(spam|doc) = 0.0150 / (0.0150 + 0.00042) = 0.973`)}
<p>The independence assumption is plainly false — "free" and "offer" co-occur constantly — and the classification is still right, because the <i>ranking</i> survives the error even though 0.973 is overconfident. That is the whole story of Naive Bayes in one number: <mark>good classifier, bad probability</mark> — so if you need calibrated output, calibrate it (§2.12).</p>`)}

${H.lab('nb', 'Naive Bayes, editable', 'Change the priors and the word likelihoods, toggle Laplace smoothing, and watch both the decision and the confidence move. Add the unseen word and watch the un-smoothed model produce a zero.')}

${H.probe([
      ['Why does Naive Bayes work despite a false assumption?', 'Classification needs only the argmax of the class scores; dependence errors distort magnitudes far more than ranking.'],
      ['k = 1 vs k = 25?', 'Pure variance vs high bias — the same data, one hyperparameter.'],
      ['Why does k-NN fail in high dimensions?', 'Distance concentration: nearest and farthest neighbours become nearly equidistant.']
    ], 'Using k-NN or an RBF kernel on unscaled features — one column in large units then dominates every distance.')}`,
    labs: {
      knn: function (host) {
        const data = Num.dataset('blobs', 70, .55, 8);
        const st = Viz.controls(host, [
          { k: 'k', label: 'k (neighbours)', min: 1, max: 45, step: 2, value: 1, fmt: v => v },
          { k: 'scale', label: 'scale feature 2 by', min: .2, max: 5, step: .1, value: 1, fmt: v => '×' + v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'tr', label: 'training accuracy', cls: 'key' }, { k: 'loo', label: 'leave-one-out accuracy' }, { k: 'char', label: 'model character' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const X = data.X.map(p => [p[0], p[1] * st.scale]);
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2 * Math.max(1, st.scale), 3.2 * Math.max(1, st.scale)] })
              .frame({ xlabel: 'x₁', ylabel: 'x₂ (scaled)' });
            const predict = (a, b) => {
              const d = X.map((p, i) => [(p[0] - a) ** 2 + (p[1] - b) ** 2, data.y[i]]).sort((u, v) => u[0] - v[0]);
              let s = 0; for (let i = 0; i < Math.min(st.k, d.length); i++) s += d[i][1];
              return s / Math.min(st.k, d.length);
            };
            P.clip(() => Labs.boundary(P, predict, { step: 5 }));
            P.clip(() => Labs.points(P, X, data.y));
            const trAcc = Num.mean(X.map((p, i) => (predict(p[0], p[1]) > .5 ? 1 : 0) === data.y[i] ? 1 : 0));
            const loo = Num.mean(X.map((p, i) => {
              const d = X.map((q, j) => j === i ? [1e9, 0] : [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, data.y[j]]).sort((u, v) => u[0] - v[0]);
              let s = 0; for (let t = 0; t < Math.min(st.k, d.length - 1); t++) s += d[t][1];
              return ((s / Math.min(st.k, d.length - 1)) > .5 ? 1 : 0) === data.y[i] ? 1 : 0;
            }));
            out({
              tr: (trAcc * 100).toFixed(1) + '%', loo: (loo * 100).toFixed(1) + '%',
              char: st.k <= 3 ? 'high variance' : st.k >= 21 ? 'high bias' : 'balanced'
            });
          }
        });
        Viz.note(host, 'At k = 1 training accuracy is 100% and leave-one-out is much lower — the textbook signature of memorisation. Now drag the scale slider: nothing about the data changed, but the boundary does, because distance is not unit-free.');
      },

      curse: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'points', min: 50, max: 1000, step: 50, value: 300, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'd2', label: 'near/far ratio @ d=2', cls: 'good' }, { k: 'd10', label: '@ d=10' },
          { k: 'd100', label: '@ d=100', cls: 'bad' }, { k: 'd1000', label: '@ d=1000', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(13);
            const dims = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400, 800];
            const ratios = dims.map(d => {
              const pts = Array.from({ length: st.n }, () => Array.from({ length: d }, () => R.normal(0, 1)));
              const q = Array.from({ length: d }, () => R.normal(0, 1));
              const ds = pts.map(p => { let s = 0; for (let i = 0; i < d; i++) s += (p[i] - q[i]) ** 2; return Math.sqrt(s); });
              return [d, Math.min.apply(null, ds) / Math.max.apply(null, ds)];
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, 800], yd: [0, 1] })
              .frame({ xlabel: 'dimensions d', ylabel: 'nearest distance ÷ farthest distance' });
            P.clip(() => {
              P.line(ratios, { color: T.blue, width: 2.6 });
              P.dots(ratios, { r: 3.4, color: T.blue });
              P.hline(1, { color: T.red, dash: [4, 4], label: 'everything equidistant — "nearest" is meaningless' });
            });
            const get = d => { const r = ratios.find(x => x[0] >= d); return r ? r[1].toFixed(3) : '—'; };
            out({ d2: get(2), d10: get(10), d100: get(100), d1000: get(400) });
          }
        });
        Viz.note(host, 'This is why embeddings are 384–1536 dimensions rather than 100,000, why cosine similarity is preferred to Euclidean at high d, and why the answer to "just use k-NN" is often "on what distance?"');
      },

      nb: function (host) {
        const st = Viz.controls(host, [
          { k: 'prior', label: 'P(spam)', min: .02, max: .9, step: .01, value: .3, fmt: v => v.toFixed(2) },
          { k: 'offer', label: 'P(offer | spam) / P(offer | ham)', min: .5, max: 30, step: .5, value: 10, fmt: v => v.toFixed(1) + '×' },
          { k: 'free', label: 'P(free | spam) / P(free | ham)', min: .5, max: 30, step: .5, value: 8.3, fmt: v => v.toFixed(1) + '×' },
          { k: 'unseen', label: 'include an unseen word', type: 'toggle', value: false },
          { k: 'smooth', label: 'Laplace smoothing', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'spam', label: 'spam score' }, { k: 'ham', label: 'ham score' },
          { k: 'post', label: 'P(spam | doc)', cls: 'key' }, { k: 'call', label: 'decision' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pHamOffer = .02, pHamFree = .03;
            const pSpamOffer = Math.min(.95, pHamOffer * st.offer), pSpamFree = Math.min(.95, pHamFree * st.free);
            let terms = [
              ['prior', st.prior, 1 - st.prior],
              ['"offer"', pSpamOffer, pHamOffer],
              ['"free"', pSpamFree, pHamFree]
            ];
            if (st.unseen) {
              const eps = st.smooth ? 1 / 1000 : 0;
              terms.push(['"quokka" (unseen)', eps, eps]);
            }
            let spam = 1, ham = 1;
            terms.forEach(t => { spam *= t[1]; ham *= t[2]; });
            const post = (spam + ham) > 0 ? spam / (spam + ham) : NaN;
            ctx.font = '12px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const x0 = 14, colS = Math.min(300, w * .48), colH = Math.min(420, w * .72);
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('factor', x0, 22); ctx.textAlign = 'right';
            ctx.fillText('spam', colS, 22); ctx.fillText('ham', colH, 22);
            terms.forEach((t, i) => {
              const y = 46 + i * 24;
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(t[0], x0, y);
              ctx.textAlign = 'right';
              ctx.fillStyle = t[1] === 0 ? T.red : T.text; ctx.fillText(t[1].toFixed(4), colS, y);
              ctx.fillStyle = t[2] === 0 ? T.red : T.text; ctx.fillText(t[2].toFixed(4), colH, y);
            });
            const yT = 46 + terms.length * 24 + 8;
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(x0, yT - 12); ctx.lineTo(colH, yT - 12); ctx.stroke();
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('product', x0, yT + 4);
            ctx.textAlign = 'right';
            ctx.fillStyle = spam === 0 ? T.red : T.blue; ctx.fillText(spam.toExponential(3), colS, yT + 4);
            ctx.fillStyle = ham === 0 ? T.red : T.green; ctx.fillText(ham.toExponential(3), colH, yT + 4);
            // posterior bar
            const by = yT + 34, bw = Math.max(120, Math.min(colH - x0, w - 28));
            if (isFinite(post)) {
              ctx.fillStyle = T.blue; ctx.fillRect(x0, by, bw * post, 22);
              ctx.fillStyle = T.green; ctx.fillRect(x0 + bw * post, by, bw * (1 - post), 22);
              ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'left';
              if (post > .15) ctx.fillText(' spam ' + (post * 100).toFixed(1) + '%', x0 + 4, by + 11);
            } else {
              ctx.fillStyle = T.red; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
              ctx.fillText('Both products are zero — one unseen word destroyed the whole document. This is what Laplace smoothing exists for.', x0, by + 11);
            }
            out({
              spam: spam.toExponential(2), ham: ham.toExponential(2),
              post: isFinite(post) ? (post * 100).toFixed(1) + '%' : 'undefined',
              call: !isFinite(post) ? 'broken' : (post > .5 ? 'SPAM' : 'ham')
            });
          }
        });
        Viz.note(host, 'Turn on the unseen word with smoothing off and both scores collapse to zero — the classifier stops working entirely. That single failure mode is why smoothing is not optional.');
      }
    },
    quiz: [
      {
        q: 'k-NN with k = 1 has 100% training accuracy. What does that tell you?',
        options: ['The model is excellent', 'Nothing — every point is its own nearest neighbour; only held-out accuracy is informative', 'The data is separable', 'k should be lower'],
        answer: 1,
        why: 'It is memorisation by construction. Leave-one-out or a holdout is the only meaningful estimate.'
      },
      {
        q: 'Naive Bayes gives P(spam) = 0.973 on a document where its independence assumption is badly violated. The correct reading is…',
        options: ['The probability is trustworthy', 'The classification is likely right but the probability is overconfident; calibrate if you need probabilities', 'The model has failed', 'You must drop the correlated features'],
        answer: 1,
        why: 'Dependence inflates the product of likelihoods, distorting magnitude far more than ranking. Good classifier, bad probability.'
      }
    ],
    cards: [
      { q: 'k-NN’s two failure modes', a: 'Unscaled features (distance dominated by large-unit columns) and high dimensions (distance concentration).' },
      { q: 'Naive Bayes in one line', a: '$\\arg\\max_c P(c)\\prod_t P(x_t|c)$ with conditional independence; good classifier, bad probability.' },
      { q: 'Why Laplace smoothing?', a: 'One unseen token would otherwise zero the entire product.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.6 */
  ML.section({
    id: 'svm', track: 'classical', num: '2.6',
    title: 'SVMs and kernels',
    lede: 'Uses convexity (§1.9) and PSD (§1.8); the kernel idea returns as the inner-product view of attention (§4.3).',
    html: `
<h2><span class="sn">2.6.1</span> The margin</h2>
<p>Maximise the margin: $\\min \\tfrac{1}{2}\\|w\\|^2$ subject to $y_i(w^\\mathsf{T}x_i + b) \\ge 1$. The soft-margin version admits slack,</p>
$$\\min \\tfrac12\\|w\\|^2 + C\\sum_i \\xi_i$$
<p>which is equivalent to minimising hinge loss with L2 regularization — so <b>$C$ is an inverse regularization strength</b>, not a mysterious dial. Large $C$ = fit the training data hard; small $C$ = a wider, smoother margin that tolerates violations.</p>

<h2><span class="sn">2.6.2</span> The kernel trick</h2>
<p>The dual problem depends on the data only through inner products $x_i^\\mathsf{T}x_j$. Replace them with $K(x_i,x_j)$ and you are working in an implicit, possibly infinite-dimensional feature space <b>without ever computing the mapping</b>. $K$ must be PSD (Mercer's condition, §1.8) for the problem to stay convex. RBF $K = \\exp(-\\gamma\\|x_i-x_j\\|^2)$ is the default; $\\gamma$ controls how local each point's influence is, and it interacts with $C$ — tune them jointly.</p>

${H.lab('svm', 'Margin, slack, kernels — a real SMO solver', 'This trains an actual soft-margin SVM in your browser (simplified SMO). Ringed points are the support vectors; move C and γ and watch which points the solution depends on. On the circles data, switch to RBF and the boundary curves — without ever computing the lifted coordinates.')}

<h2><span class="sn">2.6.3</span> Support vectors</h2>
<p>The points on or inside the margin. The solution depends on them alone — which is why SVMs are memory-efficient at prediction time, and why they are sensitive to a mislabelled point sitting near the boundary. They do not scale to millions of rows (kernel matrices are $O(n^2)$), which is why boosted trees, not SVMs, own tabular practice today.</p>

${H.fig('KERNELS AS SIMILARITY', `
<p style="margin:0 0 8px">Every kernel is a similarity function, and choosing one is a modelling statement about what "close" means:</p>` +
      H.table(['Kernel', 'Form', 'Says'], [
        ['Linear', '$x^\\mathsf{T}z$', 'Similarity is alignment; the boundary is a hyperplane in the original features'],
        ['Polynomial', '$(1+x^\\mathsf{T}z)^d$', 'Interactions up to order $d$ matter'],
        ['RBF / Gaussian', '$\\exp(-\\gamma\\|x-z\\|^2)$', 'Only nearby points are similar; $1/\\sqrt\\gamma$ is the length scale'],
        ['Laplacian', '$\\exp(-\\gamma\\|x-z\\|_1)$', 'Same, with heavier tails — less smooth boundaries']
      ]), 'The RBF kernel corresponds to an infinite-dimensional feature map, which is why it can separate any finite dataset — and why unregularised (large C) RBF SVMs memorise happily.')}

${H.probe([
      ['What are support vectors?', 'The points on or inside the margin; they alone determine the boundary.'],
      ['What is C?', 'An inverse regularization strength — the price of a margin violation, equivalent to hinge loss plus L2.'],
      ['Why must a kernel be PSD?', 'Otherwise it is not an inner product in any feature space and the dual is no longer convex.']
    ], 'Using an RBF kernel on unscaled features — the Euclidean distance is then dominated by whichever column happens to be measured in large units.')}`,
    labs: {
      svm: function (host) {
        let data = Num.dataset('blobs', 44, .5, 21);
        let model = null;
        const st = Viz.controls(host, [
          { k: 'kernel', label: 'kernel', type: 'buttons', value: 'linear', options: [{ v: 'linear', t: 'linear' }, { v: 'rbf', t: 'RBF' }, { v: 'poly', t: 'poly d=3' }] },
          { k: 'C', label: 'C (inverse regularization)', min: -2, max: 2, step: .1, value: 0, fmt: v => Math.pow(10, v).toFixed(2) },
          { k: 'gamma', label: 'γ (RBF width)', min: -1.5, max: 1.2, step: .1, value: 0, fmt: v => Math.pow(10, v).toFixed(2) }
        ], fit);
        const out = Viz.readout(host, [
          { k: 'sv', label: 'support vectors', cls: 'key' }, { k: 'acc', label: 'training accuracy' },
          { k: 'C', label: 'C' }, { k: 'time', label: 'solve time' }
        ]);
        function fit() {
          const t0 = performance.now();
          model = Num.svm(data.X, data.y, {
            kernel: st.kernel, C: Math.pow(10, st.C), gamma: Math.pow(10, st.gamma), degree: 3, passes: 6
          });
          model._ms = (performance.now() - t0).toFixed(0);
          S.redraw();
        }
        const S = Viz.surface(host, {
          height: 350,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'x₁', ylabel: 'x₂' });
            if (model) {
              P.clip(() => {
                Labs.boundary(P, (x, y) => Num.sigmoid(model.decide([x, y]) * 2), { step: 5 });
                P.contours((x, y) => model.decide([x, y]), [-1, 1], { color: T.faint, width: 1.2, alpha: .9 });
                P.contours((x, y) => model.decide([x, y]), [0], { color: T.text, width: 2 });
              });
              P.clip(() => {
                Labs.points(P, data.X, data.y);
                model.sv.forEach(i => P.dots([data.X[i]], { r: 8, color: 'transparent', stroke: T.amber, strokeWidth: 2 }));
              });
              const acc = Num.mean(data.X.map((x, i) => ((model.decide(x) > 0 ? 1 : 0) === data.y[i]) ? 1 : 0));
              out({
                sv: model.sv.length + ' of ' + data.X.length, acc: (acc * 100).toFixed(1) + '%',
                C: Math.pow(10, st.C).toFixed(2), time: model._ms + ' ms'
              });
            }
          }
        });
        Viz.buttons(host, [
          { label: 'Blobs', on: () => { data = Num.dataset('blobs', 44, .5, 21); fit(); } },
          { label: 'Circles (needs RBF)', on: () => { data = Num.dataset('circles', 60, .18, 3); fit(); } },
          { label: 'Moons', on: () => { data = Num.dataset('moons', 56, .3, 9); fit(); } },
          { label: 'XOR', on: () => { data = Num.dataset('xor', 60, .3, 6); fit(); } }
        ]);
        Viz.legend(host, [{ c: Viz.theme().amber, t: 'support vectors (ringed)' }, { c: Viz.theme().text, t: 'decision boundary' }, { c: Viz.theme().faint, t: 'margin ±1' }]);
        fit();
        Viz.note(host, 'Load circles with a linear kernel: hopeless. Switch to RBF: solved. Nothing about the data changed — only the definition of similarity, and the algorithm never computed a single lifted coordinate.');
      }
    },
    quiz: [
      {
        q: 'Increasing C in a soft-margin SVM…',
        options: ['widens the margin and tolerates more violations', 'narrows the margin and penalises violations more — less regularization', 'has no effect with an RBF kernel', 'is equivalent to increasing γ'],
        answer: 1,
        why: 'C is the price of slack. Large C = fit hard = weak regularization; the equivalent view is hinge loss with L2 penalty 1/C.'
      },
      {
        q: 'Why is FlashAttention-style thinking irrelevant here but kernel PSD-ness essential?',
        options: ['PSD guarantees the kernel corresponds to an inner product, keeping the dual convex', 'PSD makes the kernel faster', 'PSD ensures features are scaled', 'It is a convention only'],
        answer: 0,
        why: 'Mercer’s condition is exactly the requirement that some feature map exists whose inner product is the kernel; without it "the kernel trick" is meaningless and the dual can be unbounded.'
      }
    ],
    cards: [
      { q: 'Soft-margin objective', a: '$\\min \\frac12\\|w\\|^2 + C\\sum\\xi_i$ — equivalently hinge loss + L2; $C$ is inverse regularization.' },
      { q: 'The kernel trick', a: 'The dual depends on data only through inner products; replace them with a PSD kernel to work in an implicit feature space.' },
      { q: 'Why don’t SVMs own tabular ML?', a: 'Kernel matrices are $O(n^2)$; boosted trees scale and handle mixed types natively.' }
    ]
  });
})();
