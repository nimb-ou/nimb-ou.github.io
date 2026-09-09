/* ============================================================
   PART 2 — Core & classical ML (2.12 – 2.20)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.12 */
  ML.section({
    id: 'calibration', track: 'classical', num: '2.12',
    title: 'Class imbalance, calibration, and conformal prediction',
    lede: 'A genuinely contested topic. Flag the contest every time — that is what makes the answer credible.',
    html: `
<h2><span class="sn">2.12.1</span> The evidence on resampling</h2>
<p>Van den Goorbergh, van Smeden, Timmerman and Van Calster (<i>JAMIA</i> 29(9):1525–1534, 2022; doi:10.1093/jamia/ocac093) tested random undersampling, random oversampling and SMOTE on clinical prediction models and found they <b>yielded poorly calibrated models</b> — the probability of belonging to the minority class was strongly overestimated — and did <b>not</b> improve area under the ROC curve. Resampling changes the base rate your model believes in, and a model that believes in the wrong base rate produces the wrong probabilities.</p>

${H.lab('resample', 'What resampling does to your probabilities', 'A logistic model trained here on imbalanced data, then retrained on a resampled version. The ranking barely moves; the reliability curve leaves the diagonal immediately. Both effects are computed, not asserted.')}

<h3>Practical stance</h3>
<p>If you need probabilities — credit decisions, expected loss, pricing — <b>leave the base rate intact</b>. Use class weights or, better, choose a threshold from the actual cost matrix (§2.13). If you resample anyway, <mark>recalibrate afterwards</mark>, and say why. If you only need a ranking (who to review first), the calibration harm may be irrelevant — so state the objective <i>before</i> prescribing a fix. That conditional answer is the senior one.</p>

<h2><span class="sn">2.12.2</span> Calibration methods</h2>
${H.table(['Method', 'What it fits', 'Use when'], [
      ['<b>Platt scaling</b>', 'A one-dimensional logistic regression on the scores — two parameters', 'Little calibration data, or a smooth sigmoidal distortion'],
      ['<b>Isotonic regression</b>', 'Any monotone mapping', 'Thousands of held-out rows and a non-sigmoidal distortion; can overfit into steps'],
      ['<b>Beta calibration</b>', 'A three-parameter family generalising Platt', 'When Platt is too rigid but isotonic overfits'],
      ['<b>Temperature scaling</b>', 'One scalar dividing the logits', 'Neural nets, multi-class; preserves the argmax exactly']
    ])}
<p>Measure with a <b>reliability diagram</b> (predicted vs observed frequency per bin), the <b>Brier score</b> (mean squared probability error, which decomposes into calibration and refinement), and expected calibration error. Fit the calibrator on held-out data, never on the training fold.</p>

${H.lab('calib', 'Reliability diagrams and two calibrators', 'A deliberately miscalibrated model, then Platt and isotonic fitted on a held-out split. Watch the Brier score decompose and the curve return to the diagonal — and watch isotonic overfit when you shrink the calibration set.')}

<h2><span class="sn">2.12.3</span> Conformal prediction — the uncertainty method worth knowing in 2026</h2>
<p>Calibration fixes the average; conformal prediction gives a <b>per-prediction</b> guarantee, distribution-free and model-agnostic. Split conformal, in three steps:</p>
${H.steps([
      'Hold out a calibration set the model never trained on.',
      'Compute a nonconformity score for each held-out point — for regression $|y-\\hat y|$; for classification $1-\\hat p_{\\text{true}}$.',
      'Take the $\\lceil (n+1)(1-\\alpha)\\rceil$-th smallest score as the quantile $q$.'
    ])}
<p>Then the prediction set $\\{y : \\text{score}(x,y) \\le q\\}$ contains the truth with probability at least $1-\\alpha$ — <i>marginally</i>, over exchangeable data, <b>with no assumption about the model being right</b>.</p>

${H.worked('worked conformal interval', `
<p>Regression on loss-given-default, 1,000 calibration points, target 90% coverage ($\\alpha = 0.10$). Rank the absolute residuals; take the $\\lceil 1001 \\times 0.90 \\rceil = 901$st smallest. Suppose it is £2,340.</p>
<p>Every future prediction is then reported as $\\hat y \\pm £2{,}340$, and that interval covers the truth 90% of the time <i>whatever the model is</i> — gradient boosting, a neural net, a rule. Two honest caveats to volunteer: the guarantee is <b>marginal, not conditional</b>, so intervals are the same width for easy and hard cases unless you use a normalised or Mondrian variant; and it assumes <b>exchangeability</b>, which time-series drift breaks — for those, adaptive conformal methods update the quantile online.</p>`)}

${H.lab('conformal', 'Split conformal, end to end', 'Residuals ranked, the quantile taken, the interval applied to fresh data — and then the coverage measured on that fresh data. Move α and watch the empirical coverage track the promise. Turn on drift and watch the guarantee break exactly as the theory says it should.')}

<p>Why it matters for decisions: a calibrated point probability tells you the average is right; a conformal set tells you <b>when the model does not know</b>. In credit that is the difference between an automatic decline and a referral to manual review, and "we route wide prediction sets to a human" is a much stronger answer than "we picked a threshold".</p>

${H.probe([
      ['Does SMOTE improve models?', 'It can raise recall at a fixed threshold, but the JAMIA 2022 evidence is that it harms calibration and rarely improves AUC — and simply moving the threshold often matches it.'],
      ['Platt or isotonic?', 'Platt with little calibration data or a smooth distortion; isotonic with thousands of held-out rows and a non-sigmoidal one.'],
      ['What exactly does conformal guarantee?', 'Marginal coverage at level 1−α over exchangeable data, with no assumption that the model is correct.']
    ], 'Reporting accuracy on a 1%-positive problem. Predicting "no" always scores 99%.')}`,
    labs: {
      resample: function (host) {
        const st = Viz.controls(host, [
          { k: 'base', label: 'true positive rate in the data', min: .01, max: .3, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'method', label: 'treatment', type: 'buttons', value: 'over', options: [{ v: 'none', t: 'none' }, { v: 'over', t: 'oversample to 50/50' }, { v: 'under', t: 'undersample' }, { v: 'weight', t: 'class weights' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'auc', label: 'AUC (ranking)', cls: 'key' }, { k: 'brier', label: 'Brier score' },
          { k: 'mean', label: 'mean predicted p' }, { k: 'true', label: 'true base rate' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(61), n = 1500;
            const X = [], y = [];
            for (let i = 0; i < n; i++) {
              const pos = R() < st.base;
              const x = [R.normal(pos ? 1.1 : 0, 1), R.normal(pos ? .7 : 0, 1)];
              X.push(x); y.push(pos ? 1 : 0);
            }
            // build the training set according to the treatment
            let Xtr = X.slice(), ytr = y.slice(), wts = null;
            const posIdx = y.map((v, i) => v ? i : -1).filter(i => i >= 0);
            const negIdx = y.map((v, i) => v ? -1 : i).filter(i => i >= 0);
            if (st.method === 'over') {
              Xtr = []; ytr = [];
              negIdx.forEach(i => { Xtr.push(X[i]); ytr.push(0); });
              for (let i = 0; i < negIdx.length; i++) { const j = posIdx[R.int(posIdx.length)]; Xtr.push(X[j]); ytr.push(1); }
            } else if (st.method === 'under') {
              Xtr = []; ytr = [];
              posIdx.forEach(i => { Xtr.push(X[i]); ytr.push(1); });
              for (let i = 0; i < posIdx.length; i++) { const j = negIdx[R.int(negIdx.length)]; Xtr.push(X[j]); ytr.push(0); }
            }
            const m = Num.logistic(Xtr, ytr, { lr: .6 });
            m.step(400);
            const p = X.map(x => m.predict(x));
            const roc = Num.rocCurve(p, y);
            const bins = Num.calibrationBins(p, y, 10).filter(b => b.n > 8);
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1] })
              .frame({ xlabel: 'predicted probability', ylabel: 'observed frequency' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, width: 1.4, dash: [5, 4] });
              P.line(bins.map(b => [b.pred, b.obs]), { color: st.method === 'none' || st.method === 'weight' ? T.green : T.red, width: 2.6 });
              P.dots(bins.map(b => [b.pred, b.obs]), { r: 4, color: st.method === 'none' || st.method === 'weight' ? T.green : T.red, stroke: true });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText(st.method === 'over' || st.method === 'under'
              ? 'above the diagonal = over-confident on the minority class'
              : 'on the diagonal = calibrated', w - 16, 10);
            out({
              auc: roc.auc.toFixed(4), brier: Num.brier(p, y).toFixed(4),
              mean: (Num.mean(p) * 100).toFixed(1) + '%', true: (Num.mean(y) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'Compare "none" with "oversample": AUC moves in the third or fourth decimal — resampling did not improve the ranking — while mean predicted probability jumps from the true base rate to something far higher. That is the JAMIA finding, reproduced here in your browser.');
      },

      calib: function (host) {
        const st = Viz.controls(host, [
          { k: 'distort', label: 'model miscalibration', min: .3, max: 3, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'method', label: 'calibrator', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'none' }, { v: 'platt', t: 'Platt' }, { v: 'iso', t: 'isotonic' }] },
          { k: 'ncal', label: 'calibration rows', min: 50, max: 3000, step: 50, value: 800, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ece', label: 'expected calibration error', cls: 'key' }, { k: 'brier', label: 'Brier score' },
          { k: 'auc', label: 'AUC (unchanged by monotone maps)' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(83);
            const make = n => {
              const raw = [], lab = [];
              for (let i = 0; i < n; i++) {
                const pTrue = R();
                const yy = R() < pTrue ? 1 : 0;
                raw.push(Math.pow(pTrue, st.distort));     // distorted score
                lab.push(yy);
              }
              return { raw, lab };
            };
            const cal = make(st.ncal), te = make(1500);
            let f = x => x;
            if (st.method === 'platt') {
              const Xc = cal.raw.map(v => [Math.log(Math.max(1e-6, v) / Math.max(1e-6, 1 - v))]);
              const lr = Num.logistic(Xc, cal.lab, { lr: .5 }); lr.step(500);
              f = x => lr.predict([Math.log(Math.max(1e-6, x) / Math.max(1e-6, 1 - x))]);
            } else if (st.method === 'iso') {
              const pairs = cal.raw.map((v, i) => [v, cal.lab[i]]).sort((a, b) => a[0] - b[0]);
              // PAVA
              let blocks = pairs.map(p => ({ sum: p[1], n: 1, x: p[0] }));
              let changed = true;
              while (changed) {
                changed = false;
                for (let i = 0; i < blocks.length - 1; i++) {
                  if (blocks[i].sum / blocks[i].n > blocks[i + 1].sum / blocks[i + 1].n) {
                    blocks[i] = { sum: blocks[i].sum + blocks[i + 1].sum, n: blocks[i].n + blocks[i + 1].n, x: blocks[i + 1].x };
                    blocks.splice(i + 1, 1); changed = true; break;
                  }
                }
              }
              f = x => {
                for (let i = 0; i < blocks.length; i++) if (x <= blocks[i].x) return blocks[i].sum / blocks[i].n;
                return blocks[blocks.length - 1].sum / blocks[blocks.length - 1].n;
              };
            }
            const p = te.raw.map(f);
            const bins = Num.calibrationBins(p, te.lab, 12).filter(b => b.n > 5);
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1] })
              .frame({ xlabel: 'predicted probability', ylabel: 'observed frequency' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, width: 1.4, dash: [5, 4] });
              const rawBins = Num.calibrationBins(te.raw, te.lab, 12).filter(b => b.n > 5);
              P.line(rawBins.map(b => [b.pred, b.obs]), { color: T.red, width: 1.6, alpha: .55 });
              P.line(bins.map(b => [b.pred, b.obs]), { color: T.green, width: 2.6 });
              P.dots(bins.map(b => [b.pred, b.obs]), { r: 4, color: T.green, stroke: true });
            });
            const ece = Num.sum(bins.map(b => (b.n / te.lab.length) * Math.abs(b.pred - b.obs)));
            out({
              ece: ece.toFixed(4), brier: Num.brier(p, te.lab).toFixed(4),
              auc: Num.rocCurve(p, te.lab).auc.toFixed(4)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: 'uncalibrated model' }, { c: Viz.theme().green, t: 'after the chosen calibrator' }]);
        Viz.note(host, 'AUC is invariant to any monotone transformation, so calibration never changes the ranking — it changes what the number <i>means</i>. Shrink the calibration set to 50 rows with isotonic selected and watch it produce a step function that fits noise; Platt, with two parameters, does not.');
      },

      conformal: function (host) {
        const st = Viz.controls(host, [
          { k: 'alpha', label: 'miscoverage α (target = 1−α)', min: .01, max: .3, step: .01, value: .1, fmt: v => (100 * (1 - v)).toFixed(0) + '% coverage' },
          { k: 'ncal', label: 'calibration points', min: 50, max: 2000, step: 50, value: 1000, fmt: v => v.toLocaleString() },
          { k: 'drift', label: 'break exchangeability (drift)', type: 'toggle', value: false },
          { k: 'hetero', label: 'heteroskedastic noise', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'q', label: 'quantile q', cls: 'key' }, { k: 'idx', label: 'rank taken' },
          { k: 'cov', label: 'empirical coverage on fresh data' }, { k: 'width', label: 'interval width' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(11);
            const f = x => 1.4 * Math.sin(x) + .5 * x;
            const noise = x => st.hetero ? .18 + .38 * Math.abs(x) : .55;
            // model = slightly wrong version of f
            const model = x => f(x) * .92 + .12;
            const cal = [];
            for (let i = 0; i < st.ncal; i++) { const x = -3 + 6 * R(); const yv = f(x) + R.normal(0, noise(x)); cal.push({ x: x, y: yv, s: Math.abs(yv - model(x)) }); }
            const scores = cal.map(c => c.s).sort((a, b) => a - b);
            const k = Math.ceil((st.ncal + 1) * (1 - st.alpha));
            const q = scores[Math.min(scores.length - 1, k - 1)];
            // fresh data (optionally drifted)
            const test = [];
            for (let i = 0; i < 260; i++) {
              const x = -3 + 6 * R();
              const shift = st.drift ? .55 : 0;
              const yv = f(x) + shift + R.normal(0, noise(x) * (st.drift ? 1.5 : 1));
              test.push({ x: x, y: yv });
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-4, 4.5] }).frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              const band = [];
              for (let x = -3.2; x <= 3.21; x += .05) band.push([x, model(x) + q]);
              const band2 = [];
              for (let x = 3.2; x >= -3.21; x -= .05) band2.push([x, model(x) - q]);
              ctx.fillStyle = T.blue; ctx.globalAlpha = .13;
              ctx.beginPath();
              band.concat(band2).forEach((p, i) => { const X = P.x(p[0]), Y = P.y(p[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
              ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
              P.fn(model, { color: T.blue, width: 2.2 });
              test.forEach(t => {
                const covered = Math.abs(t.y - model(t.x)) <= q;
                P.dots([[t.x, t.y]], { r: 2.8, color: covered ? T.faint : T.red, alpha: covered ? .7 : 1 });
              });
            });
            const cov = Num.mean(test.map(t => Math.abs(t.y - model(t.x)) <= q ? 1 : 0));
            out({
              q: q.toFixed(3), idx: k + ' of ' + st.ncal,
              cov: (cov * 100).toFixed(1) + '%', width: (2 * q).toFixed(3)
            });
          }
        });
        Viz.note(host, 'With drift off, empirical coverage tracks 1−α closely — no assumption about the model was needed. Turn drift on and coverage collapses: exchangeability is the one thing conformal genuinely requires, and time-series data violates it routinely. Turn on heteroskedastic noise to see the other caveat: the band is a constant width, so it is too wide where the problem is easy and too narrow where it is hard, which is what normalised/Mondrian conformal fixes.');
      }
    },
    quiz: [
      {
        q: 'You apply SMOTE and your model’s mean predicted probability jumps from 5% to 43%. What happened?',
        options: ['The model got better at finding positives', 'The base rate the model believes in changed, so the probabilities are now wrong', 'The AUC must have improved', 'The features were rescaled'],
        answer: 1,
        why: 'Resampling changes the prior. Ranking may be preserved but the probabilities are no longer usable in an expected-loss calculation without recalibration.'
      },
      {
        q: 'Split conformal with 1,000 calibration points and α = 0.10 takes which residual?',
        options: ['the 900th smallest', 'the ⌈1001 × 0.90⌉ = 901st smallest', 'the mean residual × 1.645', 'the 90th percentile of the training residuals'],
        answer: 1,
        why: 'The (n+1)(1−α) order statistic on held-out residuals; using training residuals destroys the guarantee.'
      },
      {
        q: 'Conformal prediction’s coverage guarantee requires…',
        options: ['a correctly specified model', 'Gaussian residuals', 'exchangeable data', 'a large training set'],
        answer: 2,
        why: 'It is distribution-free and model-agnostic; exchangeability is the assumption, and drift is what breaks it.'
      }
    ],
    cards: [
      { q: 'Resampling and calibration', a: 'SMOTE/over/under-sampling change the base rate and harm calibration (JAMIA 2022) without reliably improving AUC. Recalibrate, or move the threshold instead.' },
      { q: 'Split conformal in three steps', a: 'Hold out a calibration set → score nonconformity → take the ⌈(n+1)(1−α)⌉-th smallest as q; the set {y: score ≤ q} covers at 1−α marginally.' },
      { q: 'Platt vs isotonic', a: 'Platt: 2 parameters, little data, smooth distortion. Isotonic: any monotone map, needs thousands of rows, can overfit into steps.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.13 */
  ML.section({
    id: 'metrics', track: 'classical', num: '2.13',
    title: 'Evaluation metrics in depth',
    lede: 'Rests on §2.1’s loss/metric split and §1.6’s interval logic; feeds §5.7’s cost-per-successful-task thinking.',
    html: `
<h2><span class="sn">2.13.1</span> Everything starts from the confusion matrix</h2>
<p>Precision $= TP/(TP+FP)$ answers "of the ones I flagged, how many were real?" Recall $= TP/(TP+FN)$ answers "of the real ones, how many did I catch?" F1 is their harmonic mean, which punishes imbalance between them — a deliberate choice, not a neutral average.</p>

${H.worked('worked confusion matrix under imbalance', `
<p>10,000 cases, 100 of them positive (1%). The model flags 200 and catches 80 of the positives.</p>
${H.table(['', 'ACTUAL +', 'ACTUAL −'], [
      ['<b>PRED +</b>', '<b>TP = 80</b>', 'FP = 120'],
      ['<b>PRED −</b>', 'FN = 20', 'TN = 9,780']
    ])}
<p>Accuracy $=(80+9780)/10000 =$ <b>98.6%</b> — and completely uninformative. Precision $= 80/200 =$ <b>40%</b>. Recall $= 80/100 =$ <b>80%</b>. F1 $= 2(0.4)(0.8)/1.2 =$ <b>0.53</b>. Three of every five investigations are wasted; that is the number the business cares about, and accuracy hides it entirely.</p>`)}

${H.lab('confusion', 'The threshold is the model’s real dial', 'One trained model, one slider. Watch precision, recall, F1, the confusion matrix, and the position on both curves move together. Nothing about the model changes — only where you cut it.')}

<h2><span class="sn">2.13.2</span> ROC and PR</h2>
<p><b>ROC-AUC</b> is the probability that a random positive outranks a random negative. It measures ranking and is insensitive to the base rate — a virtue when comparing models and a trap when judging deployability, because the false-positive axis is scaled by the huge negative class. <b>PR-AUC</b> uses precision instead, so it responds to the base rate and reveals exactly the weakness ROC hides. Under heavy imbalance, report both and operate on PR.</p>

${H.lab('roc', 'Same model, two stories', 'ROC and PR side by side on data whose imbalance you control. Push the positive rate to 1% and watch ROC stay beautiful while PR collapses — the single clearest demonstration of why a 0.95 AUC can be useless.')}

<h3>Credit-specific</h3>
<p><b>Gini</b> $= 2\\cdot\\mathrm{AUC}-1$ (so AUC 0.75 is Gini 0.50). <b>KS</b> is the maximum vertical gap between the cumulative distributions of goods and bads — a single number for separation, and the one a credit committee will ask for. <b>Brier score</b> is the mean squared error of the probabilities: the metric that notices when §2.12 has gone wrong.</p>

<h2><span class="sn">2.13.3</span> The threshold is not 0.5</h2>
${H.worked('worked number — the threshold is a business decision', `
<p>Approving a customer who defaults costs £900 of loss. Declining a customer who would have been good costs £60 of forgone margin. Approve when the expected cost of approving beats the expected cost of declining:</p>
$$p\\cdot900 < (1-p)\\cdot 60 \\;\\Longrightarrow\\; 960p < 60 \\;\\Longrightarrow\\; p < 0.0625$$
<p>So the operating threshold is <b>6.25%</b>, and in general $p^* = C_{FP}/(C_{FP}+C_{FN})$. Nothing about the model changed; the business asymmetry chose the cut. Use 0.5 and you would approve everyone up to a 50% default probability — which is why <mark>"threshold 0.5" is almost never the right answer to a cost-sensitive problem</mark>, and why §2.12's insistence on calibrated probabilities matters: this formula is meaningless if $p$ is inflated.</p>
<p>The senior addition: costs are rarely constant. Loss given default scales with exposure, so the decision is really a per-application expected-value calculation, and the "threshold" becomes a surface rather than a number.</p>`)}

${H.lab('cost', 'Cost-optimal threshold calculator', 'Set the two costs and the model’s scores; the optimal cut, the expected cost curve, and the money left on the table by using 0.5 are all computed.')}

<h2><span class="sn">2.13.4</span> Lift and gains — the table a business will read</h2>
<p>Rank the population by score, cut into deciles, report what share of all bads each decile contains. If the worst decile holds 42% of bads against a 10% share of the population, lift is 4.2× and cumulative gains at 20% of the population might be 63%. This is the same information as an ROC curve, expressed in the units a credit committee thinks in: "review the riskiest fifth and you catch two thirds of the losses." Learn to move between the two representations fluently; a good answer usually contains both.</p>

${H.lab('lift', 'Deciles, lift and cumulative gains', 'The same scores as the ROC lab, in the business’s units. The table updates with the model quality slider — and the gains curve is exactly the ROC curve with the axes relabelled.')}

<h2><span class="sn">2.13.5</span> For regression targets</h2>
<p><b>RMSE</b> penalises large errors quadratically and is right when big misses are disproportionately expensive; <b>MAE</b> treats every pound the same and is robust to outliers; <b>MAPE</b> is scale-free and interpretable but explodes near zero and punishes over-prediction more than under-prediction, so it quietly biases forecasts low. <b>$R^2$</b> is only meaningful relative to a baseline — always name the baseline. And for skewed monetary targets, evaluate in log space or use a Tweedie/gamma loss rather than forcing squared error onto a distribution that is nothing like Gaussian (§1.2).</p>

${H.probe([
      ['AUC is 0.95 and the model is useless — how?', 'With 1% positives, precision at the operating threshold and PR-AUC can both be poor while ROC looks superb.'],
      ['Gini vs AUC?', 'Gini = 2·AUC − 1; same information, credit-industry scaling.'],
      ['Which single metric would you take to a committee?', 'None alone — separation (KS/Gini), calibration (Brier) and cost at the chosen threshold, together.']
    ], 'Reporting accuracy on a 1%-positive problem — predicting "no" always scores 99%.')}`,
    labs: {
      confusion: function (host) {
        const R = Num.rng(7);
        const N = 4000;
        let scores = [], labels = [];
        function gen(sep, base) {
          scores = []; labels = [];
          for (let i = 0; i < N; i++) {
            const pos = R() < base;
            labels.push(pos ? 1 : 0);
            scores.push(Num.sigmoid(R.normal(pos ? sep : -sep, 1.4)));
          }
        }
        const st = Viz.controls(host, [
          { k: 'thr', label: 'decision threshold', min: .01, max: .99, step: .01, value: .5, fmt: v => v.toFixed(2) },
          { k: 'sep', label: 'model quality (class separation)', min: .2, max: 3, step: .05, value: 1.3, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'positive rate', min: .01, max: .5, step: .01, value: .12, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => { gen(st.sep, st.base); S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'prec', label: 'precision', cls: 'key' }, { k: 'rec', label: 'recall' }, { k: 'f1', label: 'F1' },
          { k: 'acc', label: 'accuracy' }, { k: 'auc', label: 'ROC-AUC' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            if (!scores.length) gen(st.sep, st.base);
            const c = Num.confusion(scores, labels, st.thr);
            const roc = Num.rocCurve(scores, labels);
            // left: score distributions
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40, r: Math.max(180, w * .42), t: 14, b: 34 } });
            const hp = Num.hist(scores.filter((_, i) => labels[i]), 28, 0, 1);
            const hn = Num.hist(scores.filter((_, i) => !labels[i]), 28, 0, 1);
            const mx = Math.max(Math.max.apply(null, hp.bins) / Math.max(1, hp.bins.reduce((a, b) => a + b, 0)),
                                Math.max.apply(null, hn.bins) / Math.max(1, hn.bins.reduce((a, b) => a + b, 0)));
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, mx * 1.15], pad: { l: 40, r: Math.max(180, w * .42), t: 14, b: 34 } })
              .frame({ xlabel: 'model score', ylabel: 'share' });
            P2.clip(() => {
              const drawH = (hh, col) => {
                const tot = hh.bins.reduce((a, b) => a + b, 0) || 1;
                hh.centers.forEach((cc, i) => {
                  ctx.fillStyle = col; ctx.globalAlpha = .45;
                  const x0 = P2.x(cc - hh.w / 2), x1 = P2.x(cc + hh.w / 2);
                  ctx.fillRect(x0, P2.y(hh.bins[i] / tot), Math.max(1, x1 - x0 - 1), P2.y(0) - P2.y(hh.bins[i] / tot));
                  ctx.globalAlpha = 1;
                });
              };
              drawH(hn, T.blue); drawH(hp, T.red);
              P2.vline(st.thr, { color: T.text, width: 2, dash: false, label: 'threshold' });
            });
            // right: confusion matrix
            const rx = w - Math.max(170, w * .40) + 10, ry = 30, cw = Math.min(78, (w - rx - 20) / 2), ch = 44;
            const cells = [['TP', c.tp, T.green], ['FN', c.fn, T.amber], ['FP', c.fp, T.red], ['TN', c.tn, T.faint]];
            ctx.font = '10px ui-monospace, monospace';
            cells.forEach((cell, i) => {
              const x = rx + (i % 2) * (cw + 8), yy = ry + Math.floor(i / 2) * (ch + 8);
              // layout: TP FN / FP TN
              ctx.fillStyle = cell[2]; ctx.globalAlpha = .18; ctx.fillRect(x, yy, cw, ch); ctx.globalAlpha = 1;
              ctx.strokeStyle = T.line; ctx.strokeRect(x, yy, cw, ch);
              ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(cell[0], x + 6, yy + 5);
              ctx.fillStyle = T.text; ctx.font = 'bold 15px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
              ctx.fillText(String(cell[1]), x + cw - 6, yy + ch - 5);
              ctx.font = '10px ui-monospace, monospace';
            });
            ctx.fillStyle = T.faint; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('predicted + / actual + and −', rx, ry - 14);
            // mini ROC
            const mrx = rx, mry = ry + 2 * (ch + 8) + 16, ms = Math.min(110, w - rx - 20);
            ctx.strokeStyle = T.line; ctx.strokeRect(mrx, mry, ms, ms);
            ctx.strokeStyle = T.faint; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(mrx, mry + ms); ctx.lineTo(mrx + ms, mry); ctx.stroke(); ctx.setLineDash([]);
            ctx.strokeStyle = T.blue; ctx.lineWidth = 1.8; ctx.beginPath();
            roc.roc.forEach((p, i) => { const X = mrx + p[0] * ms, Y = mry + ms - p[1] * ms; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
            ctx.stroke();
            const fpr = c.fp / (c.fp + c.tn || 1), tpr = c.recall;
            ctx.fillStyle = T.red; ctx.beginPath(); ctx.arc(mrx + fpr * ms, mry + ms - tpr * ms, 4, 0, 6.3); ctx.fill();
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('ROC · your threshold in red', mrx, mry - 4);
            out({
              prec: (c.precision * 100).toFixed(1) + '%', rec: (c.recall * 100).toFixed(1) + '%',
              f1: c.f1.toFixed(3), acc: (c.accuracy * 100).toFixed(1) + '%', auc: roc.auc.toFixed(3)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the worked example (1% positives)', primary: true, on: () => { st.$set('base', .01); st.$set('sep', 1.6); gen(st.sep, st.base); S.redraw(); } }
        ]);
        Viz.note(host, 'Set the positive rate to 1% and slide the threshold to where accuracy is maximised: the model flags almost nothing and scores 99%. Accuracy is not a metric on imbalanced problems, it is a decoy.');
      },

      roc: function (host) {
        const st = Viz.controls(host, [
          { k: 'sep', label: 'model quality', min: .2, max: 3.5, step: .05, value: 1.8, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'positive rate', min: .005, max: .5, step: .005, value: .01, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'auc', label: 'ROC-AUC', cls: 'good' }, { k: 'pr', label: 'PR-AUC', cls: 'bad' },
          { k: 'gini', label: 'Gini' }, { k: 'ks', label: 'KS' }, { k: 'baserate', label: 'PR baseline' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(19), N = 6000;
            const scores = [], labels = [];
            for (let i = 0; i < N; i++) {
              const pos = R() < st.base;
              labels.push(pos ? 1 : 0);
              scores.push(Num.sigmoid(R.normal(pos ? st.sep : -st.sep, 1.3)));
            }
            const r = Num.rocCurve(scores, labels);
            const half = (w - 60) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40, r: w - 40 - half, t: 16, b: 36 } })
              .frame({ xlabel: 'false positive rate', ylabel: 'true positive rate' });
            P1.clip(() => {
              P1.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1.2 });
              P1.line(r.roc, { color: T.blue, width: 2.6 });
              P1.area(r.roc, { color: T.blue, alpha: .1 });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 40 + half + 40, r: 14, t: 16, b: 36 } })
              .frame({ xlabel: 'recall', ylabel: 'precision' });
            P2.clip(() => {
              P2.line(r.pr, { color: T.red, width: 2.6 });
              P2.hline(st.base, { color: T.faint, dash: [4, 4], label: 'base rate' });
            });
            ctx.fillStyle = T.blue; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('ROC · looks excellent', 46, 2);
            ctx.fillStyle = T.red; ctx.fillText('PR · reveals the operating cost', 46 + half + 40, 2);
            out({
              auc: r.auc.toFixed(3), pr: r.ap.toFixed(3), gini: (2 * r.auc - 1).toFixed(3),
              ks: Num.ksStat(scores, labels).toFixed(3), baserate: st.base.toFixed(3)
            });
          }
        });
        Viz.note(host, 'At 1% positives an AUC of 0.95 routinely coexists with a PR-AUC near 0.4. The ROC curve’s x-axis is normalised by a huge negative class, so thousands of false positives barely move it — which is precisely the cost the business feels.');
      },

      cost: function (host) {
        const st = Viz.controls(host, [
          { k: 'cfn', label: 'cost of approving a defaulter (£)', min: 100, max: 3000, step: 50, value: 900, fmt: v => '£' + v },
          { k: 'cfp', label: 'cost of declining a good customer (£)', min: 10, max: 500, step: 10, value: 60, fmt: v => '£' + v },
          { k: 'sep', label: 'model quality', min: .3, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'default rate', min: .01, max: .3, step: .005, value: .06, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'opt', label: 'cost-optimal threshold', cls: 'key' }, { k: 'formula', label: 'C_FP/(C_FP+C_FN)' },
          { k: 'costOpt', label: 'cost at optimum' }, { k: 'cost50', label: 'cost at 0.5' }, { k: 'waste', label: 'money left on the table', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(37), N = 6000;
            const scores = [], labels = [];
            for (let i = 0; i < N; i++) {
              const pos = R() < st.base;
              labels.push(pos ? 1 : 0);
              scores.push(Num.sigmoid(R.normal(pos ? st.sep : -st.sep, 1.2)));
            }
            const costAt = t => {
              const c = Num.confusion(scores, labels, t);
              // "flagged" = declined. FN here = approved defaulter
              return (c.fn * st.cfn + c.fp * st.cfp) / N;
            };
            const pts = [];
            for (let t = .01; t <= .99; t += .01) pts.push([t, costAt(t)]);
            let best = pts[0]; pts.forEach(p => { if (p[1] < best[1]) best = p; });
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, Math.max.apply(null, pts.map(p => p[1])) * 1.1] })
              .frame({ xlabel: 'decision threshold', ylabel: 'expected cost per application (£)', yfmt: v => '£' + v.toFixed(0) });
            P.clip(() => {
              P.line(pts, { color: T.blue, width: 2.6 });
              P.vline(best[0], { color: T.green, width: 2, dash: false, label: 'cost-optimal' });
              P.vline(.5, { color: T.red, label: 'the default 0.5' });
              P.dots([best], { r: 5.5, color: T.green, stroke: true });
              P.dots([[.5, costAt(.5)]], { r: 5, color: T.red, stroke: true });
            });
            const theory = st.cfp / (st.cfp + st.cfn);
            out({
              opt: best[0].toFixed(3), formula: theory.toFixed(4),
              costOpt: '£' + best[1].toFixed(2), cost50: '£' + costAt(.5).toFixed(2),
              waste: '£' + (costAt(.5) - best[1]).toFixed(2) + ' / application'
            });
          }
        });
        Viz.note(host, 'The empirical optimum tracks $p^*=C_{FP}/(C_{FP}+C_{FN})$ whenever the probabilities are calibrated. With £900 versus £60 the cut is 6.25% — a long way from 0.5, and the per-application difference multiplied by your monthly volume is the business case for reading §2.12.');
      },

      lift: function (host) {
        const st = Viz.controls(host, [
          { k: 'sep', label: 'model quality', min: .2, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'base', label: 'bad rate', min: .01, max: .3, step: .005, value: .08, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'lift1', label: 'lift, worst decile', cls: 'key' }, { k: 'gain2', label: 'gains @ 20%' },
          { k: 'gain5', label: 'gains @ 50%' }, { k: 'ks', label: 'KS' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(53), N = 10000;
            const rows = [];
            for (let i = 0; i < N; i++) {
              const bad = R() < st.base;
              rows.push({ s: Num.sigmoid(R.normal(bad ? st.sep : -st.sep, 1.25)), bad: bad ? 1 : 0 });
            }
            rows.sort((a, b) => b.s - a.s);
            const totalBad = Num.sum(rows.map(r => r.bad));
            const deciles = [];
            for (let d = 0; d < 10; d++) {
              const seg = rows.slice(d * N / 10, (d + 1) * N / 10);
              const bads = Num.sum(seg.map(r => r.bad));
              deciles.push({ d: d + 1, bads: bads, share: bads / totalBad, rate: bads / seg.length });
            }
            let cum = 0; const gains = [[0, 0]];
            deciles.forEach((dd, i) => { cum += dd.share; gains.push([(i + 1) / 10, cum]); });
            const half = (w - 60) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [.5, 10.5], yd: [0, Math.max.apply(null, deciles.map(d => d.share)) * 1.2], pad: { l: 42, r: w - 42 - half, t: 20, b: 38 } })
              .frame({ xlabel: 'decile (1 = riskiest)', ylabel: 'share of all bads', yfmt: v => (v * 100).toFixed(0) + '%' });
            P1.clip(() => {
              deciles.forEach(d => {
                const x0 = P1.x(d.d - .38), x1 = P1.x(d.d + .38);
                ctx.fillStyle = d.d <= 2 ? T.red : T.blue;
                ctx.fillRect(x0, P1.y(d.share), x1 - x0, P1.y(0) - P1.y(d.share));
              });
              P1.hline(.1, { color: T.faint, dash: [4, 4], label: 'a random 10%' });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 42 + half + 40, r: 14, t: 20, b: 38 } })
              .frame({ xlabel: 'share of population reviewed', ylabel: 'share of bads caught', xfmt: v => (v * 100).toFixed(0) + '%', yfmt: v => (v * 100).toFixed(0) + '%' });
            P2.clip(() => {
              P2.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1.2 });
              P2.line(gains, { color: T.green, width: 2.6 });
              P2.dots(gains.slice(1), { r: 3.2, color: T.green });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('lift by decile', 48, 4);
            ctx.fillText('cumulative gains', 48 + half + 40, 4);
            out({
              lift1: (deciles[0].share / .1).toFixed(2) + '×',
              gain2: (gains[2][1] * 100).toFixed(0) + '%',
              gain5: (gains[5][1] * 100).toFixed(0) + '%',
              ks: Num.ksStat(rows.map(r => r.s), rows.map(r => r.bad)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'The gains curve is the ROC curve with the axes relabelled into business units. "Review the riskiest fifth and catch two thirds of the losses" and "AUC 0.78" are the same sentence spoken to two different audiences.');
      }
    },
    quiz: [
      {
        q: '10,000 cases, 1% positive. The model flags 200 and catches 80. Precision and recall are…',
        options: ['80% and 40%', '40% and 80%', '98.6% and 80%', '40% and 0.53'],
        answer: 1,
        why: 'Precision = 80/200 = 40%; recall = 80/100 = 80%. Accuracy 98.6% is uninformative here.'
      },
      {
        q: 'Approving a defaulter costs £900; declining a good customer costs £60. The optimal threshold is…',
        options: ['0.5', '0.0625', '0.0667', '0.9375'],
        answer: 1,
        why: '$p^* = C_{FP}/(C_{FP}+C_{FN}) = 60/960 = 0.0625$ — and it is only valid if the probabilities are calibrated.'
      },
      {
        q: 'Under heavy imbalance, which pair should you report?',
        options: ['Accuracy and F1', 'ROC-AUC alone', 'ROC-AUC and PR-AUC, and operate on PR', 'Precision alone'],
        answer: 2,
        why: 'ROC compares models; PR reveals the operating cost the negative class hides.'
      }
    ],
    cards: [
      { q: 'Precision vs recall', a: 'Precision = TP/(TP+FP) — of what I flagged, how much was real. Recall = TP/(TP+FN) — of what was real, how much I caught.' },
      { q: 'Cost-optimal threshold', a: '$p^*=C_{FP}/(C_{FP}+C_{FN})$; £60 vs £900 gives 6.25%.' },
      { q: 'Gini and KS', a: 'Gini = 2·AUC − 1. KS = max gap between cumulative good and bad distributions.' },
      { q: 'When ROC lies', a: 'Under heavy imbalance: the FPR axis is scaled by the huge negative class, so PR-AUC exposes the real cost.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.14 */
  ML.section({
    id: 'validation', track: 'classical', num: '2.14',
    title: 'Validation strategy, including time series and out-of-time',
    lede: 'Rests on §1.4 for why any of this estimates anything at all.',
    html: `
<h2><span class="sn">2.14.1</span> The schemes</h2>
${H.table(['Scheme', 'What it does', 'When it is mandatory'], [
      ['<b>k-fold</b>', 'Averages k train/validate splits, reducing the variance of your estimate', 'Default for i.i.d. tabular data'],
      ['<b>Stratified k-fold</b>', 'Preserves the class ratio in every fold', 'Under imbalance — otherwise a fold may contain almost no positives'],
      ['<b>Group k-fold</b>', 'Keeps all rows of one entity on one side of the split', 'Whenever one customer/account/device contributes multiple rows'],
      ['<b>Nested CV</b>', 'Hyperparameter search in an inner loop, estimation in an outer one', 'The only honest way to report a tuned model’s performance from CV alone'],
      ['<b>Out-of-time</b>', 'Train on the past, validate on a later untouched window', 'Any time-structured data — and expected in model-risk documentation'],
      ['<b>Rolling origin</b>', 'Repeat OOT at several cut points', 'When you need several estimates and a stability read']
    ])}

${H.lab('cv', 'Every validation scheme, drawn', 'Switch between schemes and watch which rows are used for what. The group and time-series views make the failure modes obvious: a random fold splits a customer across train and validate, and a random fold on time trains on the future.')}

<h2><span class="sn">2.14.2</span> Time series changes the rules</h2>
<p>Never shuffle. Use <b>out-of-time</b> validation and rolling-origin or expanding-window backtests to get several such estimates. For a lender this is not a nicety: OOT validation is how you detect that last year's relationships have drifted (§2.18), and it is expected in model-risk documentation.</p>

${H.lab('leak-cv', 'How much does the wrong scheme lie by?', 'The same data with an entity structure and a time trend, evaluated four ways. The gap between the random-k-fold number and the out-of-time number is the size of the lie — computed, not asserted.')}

${H.probe([
      ['Why not k-fold on a time series?', 'It trains on the future to predict the past. Use out-of-time and rolling origin.'],
      ['When do you need group k-fold?', 'Whenever one entity contributes multiple rows — otherwise the model recognises the entity, not the pattern.'],
      ['Why nested CV?', 'The fold that chose your hyperparameters is no longer an unbiased estimate of performance.']
    ])}`,
    labs: {
      cv: function (host) {
        const st = Viz.controls(host, [
          { k: 'scheme', label: 'scheme', type: 'select', value: 'kfold', options: [
            { v: 'holdout', t: 'single holdout' }, { v: 'kfold', t: 'k-fold' }, { v: 'strat', t: 'stratified k-fold' },
            { v: 'group', t: 'group k-fold' }, { v: 'nested', t: 'nested CV' }, { v: 'oot', t: 'out-of-time' }, { v: 'roll', t: 'rolling origin' }] },
          { k: 'k', label: 'folds', min: 2, max: 8, step: 1, value: 5, fmt: v => v }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const N = 40;
            const rowsFor = {
              holdout: 1, kfold: st.k, strat: st.k, group: st.k, nested: st.k, oot: 1, roll: 4
            }[st.scheme];
            const x0 = 60, wpx = w - 90, rh = Math.min(30, (h - 70) / rowsFor);
            const R = Num.rng(3);
            const groups = Array.from({ length: N }, (_, i) => Math.floor(i / 5) % 8);
            const labels = Array.from({ length: N }, () => R() < .2 ? 1 : 0);
            for (let r = 0; r < rowsFor; r++) {
              for (let i = 0; i < N; i++) {
                let kind = 'train';
                if (st.scheme === 'holdout') kind = i >= N * .8 ? 'valid' : 'train';
                else if (st.scheme === 'kfold' || st.scheme === 'strat') kind = (i % st.k === r) ? 'valid' : 'train';
                else if (st.scheme === 'group') kind = (groups[i] % st.k === r) ? 'valid' : 'train';
                else if (st.scheme === 'nested') kind = (i % st.k === r) ? 'valid' : ((i % st.k === (r + 1) % st.k) ? 'inner' : 'train');
                else if (st.scheme === 'oot') kind = i >= N * .75 ? 'valid' : 'train';
                else if (st.scheme === 'roll') {
                  const trainEnd = 12 + r * 6, validEnd = trainEnd + 6;
                  kind = i < trainEnd ? 'train' : (i < validEnd ? 'valid' : 'unseen');
                }
                const col = kind === 'train' ? T.blue : kind === 'valid' ? T.red : kind === 'inner' ? T.amber : T.line;
                ctx.fillStyle = col;
                ctx.globalAlpha = kind === 'unseen' ? .35 : 1;
                ctx.fillRect(x0 + i * (wpx / N), 40 + r * (rh + 4), wpx / N - 1.5, rh);
                ctx.globalAlpha = 1;
                if (st.scheme === 'strat' && labels[i]) {
                  ctx.fillStyle = '#fff'; ctx.globalAlpha = .85;
                  ctx.fillRect(x0 + i * (wpx / N) + 1, 40 + r * (rh + 4) + rh - 4, wpx / N - 3.5, 2.5);
                  ctx.globalAlpha = 1;
                }
                if (st.scheme === 'group' && r === 0) {
                  ctx.fillStyle = T.faint; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                  if (i % 5 === 2) ctx.fillText('g' + groups[i], x0 + (i + .5) * (wpx / N), 36);
                }
              }
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(st.scheme === 'roll' ? 'run ' + (r + 1) : 'fold ' + (r + 1), x0 - 10, 40 + r * (rh + 4) + rh / 2);
            }
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const desc = {
              holdout: 'One split. Cheapest, highest-variance estimate.',
              kfold: 'Every row is validated exactly once; the estimate averages k splits.',
              strat: 'White ticks are positives — every fold keeps the class ratio.',
              group: 'All rows of a group land in the same fold; no entity spans the split.',
              nested: 'Amber = inner loop for hyperparameter search; red = outer loop for the reported estimate.',
              oot: 'Time flows left to right. Validation is strictly later than training.',
              roll: 'Several out-of-time estimates, each from a later origin. Grey = not yet seen at that point in time.'
            }[st.scheme];
            ctx.fillText(desc, x0, 44 + rowsFor * (rh + 4) + 6);
            const leg = [[T.blue, 'train'], [T.red, 'validate'], [T.amber, 'inner tuning'], [T.line, 'not yet seen']];
            leg.forEach((L, i) => {
              ctx.fillStyle = L[0]; ctx.fillRect(x0 + i * 110, h - 24, 12, 12);
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textBaseline = 'middle';
              ctx.fillText(L[1], x0 + i * 110 + 18, h - 18);
            });
          }
        });
      },

      'leak-cv': function (host) {
        const st = Viz.controls(host, [
          { k: 'entity', label: 'rows per customer', min: 1, max: 12, step: 1, value: 6, fmt: v => v },
          { k: 'drift', label: 'strength of the time trend', min: 0, max: 2, step: .05, value: .9, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rand', label: 'random k-fold AUC', cls: 'bad' }, { k: 'group', label: 'group k-fold AUC' },
          { k: 'oot', label: 'out-of-time AUC', cls: 'key' }, { k: 'lie', label: 'size of the lie' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(67);
            const nCust = 120, rows = [];
            for (let c = 0; c < nCust; c++) {
              const trait = R.normal(0, 1);                    // customer-level signal (leaks across folds)
              for (let j = 0; j < st.entity; j++) {
                const t = (c * st.entity + j) / (nCust * st.entity);
                const drift = st.drift * (t - .5);
                const z = 1.1 * trait + drift + R.normal(0, .9);
                rows.push({ x: [trait + R.normal(0, .35), t], y: R() < Num.sigmoid(z) ? 1 : 0, cust: c, t: t });
              }
            }
            function evalScheme(kind) {
              const K = 5, aucs = [];
              for (let k = 0; k < K; k++) {
                let tr, va;
                if (kind === 'rand') { tr = rows.filter((_, i) => i % K !== k); va = rows.filter((_, i) => i % K === k); }
                else if (kind === 'group') { tr = rows.filter(r => r.cust % K !== k); va = rows.filter(r => r.cust % K === k); }
                else { const cut = .55 + .09 * k; tr = rows.filter(r => r.t < cut); va = rows.filter(r => r.t >= cut && r.t < cut + .09); }
                if (!va.length || !tr.length) continue;
                const m = Num.logistic(tr.map(r => r.x), tr.map(r => r.y), { lr: .5 }); m.step(220);
                const p = va.map(r => m.predict(r.x));
                const lab = va.map(r => r.y);
                if (Num.sum(lab) === 0 || Num.sum(lab) === lab.length) continue;
                aucs.push(Num.rocCurve(p, lab).auc);
              }
              return aucs.length ? Num.mean(aucs) : NaN;
            }
            const a = evalScheme('rand'), b = evalScheme('group'), c = evalScheme('oot');
            const bars = [['random k-fold', a, T.red], ['group k-fold', b, T.amber], ['out-of-time', c, T.green]];
            const bx = 120, bw = w - bx - 90;
            bars.forEach((br, i) => {
              const y = 40 + i * 46;
              ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(br[0], bx - 12, y + 12);
              ctx.fillStyle = br[2];
              const frac = Math.max(0, (br[1] - .5) / .45);
              ctx.fillRect(bx, y, bw * frac, 24);
              ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'left';
              ctx.fillText(isFinite(br[1]) ? br[1].toFixed(3) : '—', bx + bw * frac + 8, y + 12);
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('same data, same model, three validation schemes', bx, h - 30);
            out({
              rand: isFinite(a) ? a.toFixed(3) : '—', group: isFinite(b) ? b.toFixed(3) : '—',
              oot: isFinite(c) ? c.toFixed(3) : '—',
              lie: isFinite(a) && isFinite(c) ? '+' + ((a - c) * 1000).toFixed(0) + ' bps' : '—'
            });
          }
        });
        Viz.note(host, 'Raise "rows per customer" and the random-k-fold number inflates: the model has seen the same customer on both sides of the split. Raise the time trend and the out-of-time number falls behind both. The honest number is the one that matches how you will actually deploy.');
      }
    },
    quiz: [
      {
        q: 'Each customer contributes 8 rows and you use random 5-fold CV. What breaks?',
        options: ['Nothing', 'The same customer appears in train and validate, so you measure memorisation of the customer', 'The folds become imbalanced', 'Training becomes slower'],
        answer: 1,
        why: 'Group k-fold on the entity id is the fix; without it the validation score reflects entity recognition rather than pattern learning.'
      },
      {
        q: 'You tuned 60 hyperparameter configurations with 5-fold CV and report the best CV score. That number is…',
        options: ['unbiased', 'optimistic — the folds chose the hyperparameters, so nest the search or hold out a final set', 'pessimistic', 'valid if you used stratification'],
        answer: 1,
        why: 'Selection over 60 noisy estimates inflates the maximum. Nested CV, or an untouched final holdout, restores honesty.'
      }
    ],
    cards: [
      { q: 'When is group k-fold mandatory?', a: 'Whenever one entity contributes multiple rows.' },
      { q: 'Time-series validation', a: 'Out-of-time, plus rolling origin for several estimates. Never shuffle.' },
      { q: 'Why nested CV', a: 'The fold that selected hyperparameters cannot also provide an unbiased performance estimate.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.15 */
  ML.section({
    id: 'hyperparameters', track: 'classical', num: '2.15',
    title: 'Hyperparameter search and stacking',
    lede: 'Why random beats grid, what Bayesian optimisation actually buys, and why banks rarely ship the stack that wins competitions.',
    html: `
<h2><span class="sn">2.15.1</span> Search strategies</h2>
<p>Grid search is exhaustive and exponential in the number of dimensions. <b>Random search wins per unit of compute</b> whenever only a few hyperparameters actually matter — which is usually — because every draw explores a new value of the important axis instead of repeating it. <b>Bayesian optimisation</b> and TPE fit a surrogate model of the objective and sample where improvement is likely, which pays off when each evaluation is expensive. <b>Successive halving and Hyperband</b> add early stopping of bad configurations, and that is often the biggest practical win of all.</p>

${H.lab('search', 'Why random beats grid', 'The yellow band is the region where the one hyperparameter that matters is well set. Grid samples that band three times with the same value of the important axis; random samples it three times with three different values — and the readout counts how often each strategy finds a good configuration.')}

${H.lab('halving', 'Successive halving, running', 'Configurations are trained on increasing budgets, with the worst half cut at each rung. Watch the total compute spent versus a full-budget random search that finds the same winner.')}

<h2><span class="sn">2.15.2</span> Stacking, and why banks rarely ship it</h2>
<p>Train several diverse base models, collect their <b>out-of-fold</b> predictions on the training set, and fit a simple meta-learner (usually regularised logistic regression) on those predictions as features. The out-of-fold discipline is the whole trick: train the meta-learner on in-fold predictions and it learns to trust models that have memorised their own training rows. Blending is the cheap variant — a fixed weighted average with weights tuned on a holdout.</p>
<p>Stacking reliably adds a point or two of AUC because it exploits <b>disagreement</b> between model families, which is why it wins competitions. It also multiplies the number of artefacts to validate, monitor and explain, so in a regulated setting the marginal AUC rarely survives contact with the model-risk process (§2.18) — know how it works, and know when to argue against it.</p>

${H.probe([
      ['Why does random search beat grid?', 'With few important dimensions, random draws sample the important axis at many distinct values instead of repeating a small set.'],
      ['What is the trick in stacking?', 'Out-of-fold predictions as meta-features; anything else teaches the meta-learner to trust memorisation.'],
      ['When would you not stack?', 'When each additional artefact must be independently validated, monitored and explained — the AUC gain rarely pays for that.']
    ])}`,
    labs: {
      search: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'evaluations budgeted', min: 4, max: 64, step: 1, value: 9, fmt: v => v },
          { k: 'band', label: 'width of the good region', min: .05, max: .5, step: .01, value: .18, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gridHit', label: 'grid: distinct good values', cls: 'bad' },
          { k: 'randHit', label: 'random: distinct good values', cls: 'good' },
          { k: 'gridBest', label: 'grid best score' }, { k: 'randBest', label: 'random best score' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(29);
            const side = Math.max(2, Math.round(Math.sqrt(st.n)));
            const grid = [], rand = [];
            for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) grid.push([(i + .5) / side, (j + .5) / side]);
            for (let i = 0; i < st.n; i++) rand.push([R(), R()]);
            const centre = .62;
            const score = p => Math.exp(-Math.pow((p[0] - centre) / st.band, 2)) * (1 + .06 * p[1]);
            const half = (w - 60) / 2;
            const drawPanel = (pts, x0, title, col) => {
              const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: x0, r: w - x0 - half, t: 26, b: 34 } })
                .frame({ xlabel: 'the hyperparameter that matters', ylabel: 'the one that does not' });
              P.clip(() => {
                ctx.fillStyle = 'rgba(230,190,60,.22)';
                ctx.fillRect(P.x(centre - st.band), P.pad.t, P.x(centre + st.band) - P.x(centre - st.band), P.ph);
                P.dots(pts, { r: 4, color: col, stroke: true });
              });
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x0 + 2, 8);
              return P;
            };
            drawPanel(grid, 44, 'grid · ' + grid.length + ' trials', Viz.theme().blue);
            drawPanel(rand, 44 + half + 40, 'random · ' + rand.length + ' trials', Viz.theme().red);
            const distinct = pts => new Set(pts.filter(p => Math.abs(p[0] - centre) < st.band).map(p => p[0].toFixed(3))).size;
            out({
              gridHit: distinct(grid), randHit: distinct(rand),
              gridBest: Math.max.apply(null, grid.map(score)).toFixed(3),
              randBest: Math.max.apply(null, rand.map(score)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Grid’s trials share x-coordinates by construction: with 9 trials it explores 3 distinct values of the axis that matters. Random explores 9. In higher dimensions the gap widens exponentially — which is the whole of Bergstra & Bengio’s result.');
      },

      halving: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'configurations', min: 8, max: 64, step: 4, value: 32, fmt: v => v },
          { k: 'eta', label: 'cut factor η', min: 2, max: 4, step: 1, value: 3, fmt: v => v },
          { k: 'noise', label: 'noise in early estimates', min: 0, max: .3, step: .01, value: .12, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'compute', label: 'compute used (budget units)', cls: 'key' },
          { k: 'full', label: 'full-budget search would cost' },
          { k: 'saving', label: 'saving', cls: 'good' }, { k: 'found', label: 'found the true best?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(101);
            const configs = Array.from({ length: st.n }, (_, i) => ({ id: i, quality: R() }));
            const trueBest = configs.reduce((a, b) => a.quality > b.quality ? a : b);
            let alive = configs.slice(), budget = 1, spent = 0;
            const rungs = [];
            while (alive.length > 1) {
              alive.forEach(c => { c.obs = c.quality + R.normal(0, st.noise / Math.sqrt(budget)); spent += budget; });
              rungs.push({ budget: budget, alive: alive.slice() });
              alive.sort((a, b) => b.obs - a.obs);
              alive = alive.slice(0, Math.max(1, Math.floor(alive.length / st.eta)));
              budget *= st.eta;
            }
            rungs.push({ budget: budget, alive: alive.slice() });
            const maxB = budget;
            const x0 = 56, wpx = w - 90;
            rungs.forEach((r, i) => {
              const y = 34 + i * ((h - 80) / rungs.length);
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText('budget ' + r.budget, x0 - 8, y + 6);
              r.alive.forEach((c, j) => {
                const x = x0 + (j / st.n) * wpx;
                ctx.fillStyle = c.id === trueBest.id ? T.green : T.blue;
                ctx.globalAlpha = c.id === trueBest.id ? 1 : .55;
                ctx.fillRect(x, y, Math.max(2, wpx / st.n - 2), 12);
                ctx.globalAlpha = 1;
              });
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('each row = a rung; survivors move down and get more budget. Green = the truly best configuration.', x0, h - 34);
            out({
              compute: spent.toFixed(0), full: (st.n * maxB).toFixed(0),
              saving: (100 * (1 - spent / (st.n * maxB))).toFixed(0) + '%',
              found: alive[0] && alive[0].id === trueBest.id ? 'yes' : 'no — noise killed it early'
            });
          }
        });
        Viz.note(host, 'Raise the noise and successive halving starts discarding the true best configuration at an early rung on a bad estimate. That is the real trade: enormous compute savings against the risk of an early wrong call, which is exactly what Hyperband hedges by running several bracket sizes.');
      }
    },
    quiz: [
      {
        q: 'Random search beats grid search mainly because…',
        options: ['it is easier to implement', 'with few important dimensions it samples the important axis at many distinct values', 'it converges to the global optimum', 'it uses fewer evaluations'],
        answer: 1,
        why: 'A 3×3 grid tries three values of each axis; nine random draws try nine values of the axis that matters.'
      },
      {
        q: 'In stacking, the meta-learner must be trained on…',
        options: ['the base models’ in-fold predictions', 'out-of-fold predictions', 'the raw features only', 'the test set'],
        answer: 1,
        why: 'In-fold predictions reflect memorisation, and the meta-learner would learn to trust exactly the models that memorised.'
      }
    ],
    cards: [
      { q: 'Random vs grid', a: 'Random explores more distinct values of the few axes that matter; the advantage grows with dimensionality.' },
      { q: 'Successive halving', a: 'Train many configs cheaply, cut the worst fraction each rung, give survivors more budget. Hyperband hedges the bracket size.' },
      { q: 'Stacking’s trick and its cost', a: 'Out-of-fold meta-features exploit disagreement; the cost is many more artefacts to validate, monitor and explain.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.17 */
  ML.section({
    id: 'interpretability', track: 'classical', num: '2.17',
    title: 'Interpretability, including SHAP',
    lede: 'Not optional in regulated lending: an adverse-action notice needs a reason code, and a reason code needs an attribution.',
    html: `
<h2><span class="sn">2.17.1</span> The methods, and what each actually claims</h2>
<p>Impurity-based feature importance is <b>biased toward high-cardinality and continuous features</b>, because they offer more split points to get lucky with. Prefer <b>permutation importance</b> — shuffle one column, measure the drop in a held-out metric — while remembering that correlated features share credit and will both look unimportant when either alone can substitute. <b>PDP</b> and <b>ICE</b> plots show marginal and per-instance effects; <b>LIME</b> fits a local linear surrogate around one prediction.</p>

<h2><span class="sn">2.17.2</span> SHAP</h2>
<p>SHAP assigns each feature its <b>Shapley value</b> — the average marginal contribution over all orderings — which uniquely satisfies local accuracy, missingness and consistency. The efficiency property is what makes it usable in a regulated setting:</p>
$$\\sum_i \\phi_i = f(x) - \\mathbb{E}[f]$$
<p>the attributions add up exactly to the prediction's deviation from the base rate. <b>TreeSHAP</b> computes this exactly and in polynomial time for tree ensembles — your case, if you ship gradient boosting. KernelSHAP is model-agnostic but a sampling approximation, and slow.</p>

${H.lab('shap', 'Exact Shapley values, computed over all orderings', 'Four features, sixteen coalitions, twenty-four orderings — small enough to compute the definition directly rather than approximate it. Change the inputs and watch the waterfall rebalance, always summing exactly to the prediction minus the base rate.')}

<h2><span class="sn">2.17.3</span> Reason codes, mechanically</h2>
<p>A declined applicant is legally entitled to know why, so the pipeline is: compute per-feature SHAP values for that one prediction, keep the features pushing the score adversely, rank them by magnitude, map each to a plain-language reason from a fixed, reviewed dictionary, and return the top three or four. Two details matter and both come up in interviews. First, the mapping must be <b>from bins, not raw features</b> — "utilisation above 70%" is a reason; "utilisation SHAP +0.083" is not. Second, the dictionary is fixed in advance and reviewed, so the same driver always produces the same wording; generating explanation text freely is exactly the wrong place for creativity.</p>

<h2><span class="sn">2.17.4</span> Global versus local, and the honest caveats</h2>
<p>Mean absolute SHAP across a sample gives global importance; a single waterfall explains one decision. A feature can be globally minor and locally decisive — that is a feature, not a bug, and it is the main reason permutation importance and SHAP disagree. <b>Counterfactual explanations</b> answer the question applicants actually ask: what is the smallest change that flips the decision? "Reduce utilisation below 55%" is more useful than any attribution, and also more dangerous — it is a promise about the model, so it must be computed against the deployed model with real feasibility constraints (an applicant cannot change their age).</p>
${H.flag('SHAP explains the model, never the world. If two features are correlated, the credit split between them is an artefact of the sampling scheme, and TreeSHAP’s default handling of correlated features can attribute importance to a feature the tree never really used. Interventional versus conditional expectations give different answers and the difference is not cosmetic — state what your numbers are: attributions of <i>this model’s output</i>, under a stated background distribution.')}

${H.probe([
      ['TreeSHAP vs KernelSHAP?', 'TreeSHAP is exact and fast for tree ensembles; KernelSHAP is model-agnostic sampling and slow.'],
      ['Why do permutation importance and SHAP disagree?', 'Permutation measures a global metric drop with correlated features substituting for each other; SHAP allocates local credit per prediction.'],
      ['What makes SHAP usable for adverse-action notices?', 'Efficiency: the attributions sum exactly to the prediction minus the base value, so the explanation is complete by construction.']
    ], 'Presenting SHAP values for correlated features as independent causal effects. They are attributions of <i>this model’s</i> output, not statements about the world.')}`,
    labs: {
      shap: function (host) {
        const feats = ['utilisation', 'recent enquiries', 'tenure (yrs)', 'income band'];
        const st = Viz.controls(host, [
          { k: 'f0', label: 'utilisation', min: 0, max: 1, step: .01, value: .94, fmt: v => v.toFixed(2) },
          { k: 'f1', label: 'recent enquiries', min: 0, max: 10, step: 1, value: 5, fmt: v => v },
          { k: 'f2', label: 'tenure (years)', min: 0, max: 20, step: 1, value: 9, fmt: v => v },
          { k: 'f3', label: 'income band', min: 1, max: 10, step: 1, value: 6, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'base', label: 'E[f(x)] base rate' }, { k: 'pred', label: 'f(x) prediction', cls: 'key' },
          { k: 'sum', label: 'Σφᵢ (must equal the gap)' }, { k: 'check', label: 'efficiency holds?' }
        ]);
        // a small non-linear model with interactions, plus a background distribution
        const R = Num.rng(71);
        const background = Array.from({ length: 60 }, () => [R(), R.int(11), R.int(21), 1 + R.int(10)]);
        function model(x) {
          const z = -2.1 + 2.9 * x[0] + 0.22 * x[1] - 0.07 * x[2] - 0.14 * x[3] + 0.9 * x[0] * (x[1] > 3 ? 1 : 0);
          return Num.sigmoid(z);
        }
        function shapExact(x) {
          const n = 4;
          const phi = new Array(n).fill(0);
          // v(S) = E_background[f(x_S, X_-S)]
          const cache = {};
          function v(S) {
            const key = S.join(',');
            if (cache[key] !== undefined) return cache[key];
            const val = Num.mean(background.map(b => {
              const z = [0, 1, 2, 3].map(i => S.indexOf(i) >= 0 ? x[i] : b[i]);
              return model(z);
            }));
            cache[key] = val; return val;
          }
          const perms = [];
          (function permute(arr, cur) {
            if (!arr.length) { perms.push(cur); return; }
            arr.forEach((a, i) => permute(arr.filter((_, j) => j !== i), cur.concat([a])));
          })([0, 1, 2, 3], []);
          perms.forEach(p => {
            const S = [];
            p.forEach(i => { const before = v(S.slice()); S.push(i); phi[i] += v(S.slice()) - before; });
          });
          return { phi: phi.map(p => p / perms.length), base: v([]), pred: v([0, 1, 2, 3]) };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const x = [st.f0, st.f1, st.f2, st.f3];
            const { phi, base, pred } = shapExact(x);
            const order = [0, 1, 2, 3].sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
            const x0 = 150, bw = w - x0 - 90;
            const scale = bw / Math.max(.35, Math.abs(pred - base) * 2.4);
            let cum = base;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const zeroX = x0 + bw * .18;
            ctx.fillStyle = T.muted; ctx.textAlign = 'right';
            ctx.fillText('E[f(x)] = ' + base.toFixed(3), x0 - 10, 26);
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(zeroX, 16); ctx.lineTo(zeroX, h - 40); ctx.stroke();
            order.forEach((i, k) => {
              const y = 52 + k * 34;
              const wpx = phi[i] * scale;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText(feats[i] + ' = ' + (i === 0 ? x[i].toFixed(2) : x[i]), x0 - 10, y);
              ctx.fillStyle = phi[i] >= 0 ? T.red : T.blue;
              const startX = zeroX + (cum - base) * scale;
              ctx.fillRect(Math.min(startX, startX + wpx), y - 10, Math.abs(wpx), 20);
              ctx.fillStyle = T.text; ctx.textAlign = phi[i] >= 0 ? 'left' : 'right';
              ctx.fillText((phi[i] >= 0 ? '+' : '') + phi[i].toFixed(3), startX + wpx + (phi[i] >= 0 ? 6 : -6), y);
              cum += phi[i];
            });
            const yf = 52 + 4 * 34 + 10;
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'right';
            ctx.fillText('f(x) = ' + pred.toFixed(3), x0 - 10, yf);
            ctx.fillStyle = T.text;
            ctx.fillRect(zeroX, yf - 9, Math.max(2, (pred - base) * scale), 18);
            out({
              base: base.toFixed(4), pred: pred.toFixed(4),
              sum: Num.sum(phi).toFixed(4),
              check: Math.abs(Num.sum(phi) - (pred - base)) < 1e-6 ? 'yes — exactly' : 'no'
            });
          }
        });
        Viz.note(host, 'These are exact Shapley values: all 24 orderings, evaluated against a 60-row background sample. Note the efficiency check — the attributions sum to the prediction minus the base value to machine precision, which is the property that makes them defensible in an adverse-action notice.');
      }
    },
    quiz: [
      {
        q: 'Impurity-based feature importance is biased toward…',
        options: ['binary features', 'high-cardinality and continuous features', 'the target-correlated features only', 'features with missing values'],
        answer: 1,
        why: 'More candidate split points means more chances to reduce impurity by luck. Permutation importance on held-out data is the safer default.'
      },
      {
        q: 'SHAP’s efficiency property states that…',
        options: 'the attributions are fast to compute|the attributions sum to f(x) − E[f]|correlated features get equal credit|the model must be a tree'.split('|'),
        answer: 1,
        why: 'Local accuracy: the explanation is complete, which is what makes a reason code defensible.'
      },
      {
        q: 'A reason code should be phrased as…',
        options: ['"utilisation SHAP +0.083"', '"utilisation above 70%" — mapped from bins via a fixed reviewed dictionary', 'a free-text LLM explanation', 'the feature name alone'],
        answer: 1,
        why: 'Bins map to human-readable reasons; a fixed dictionary guarantees the same driver always produces the same wording.'
      }
    ],
    cards: [
      { q: 'SHAP efficiency', a: '$\\sum_i\\phi_i = f(x)-\\mathbb{E}[f]$ — attributions sum exactly to the deviation from the base rate.' },
      { q: 'TreeSHAP vs KernelSHAP', a: 'Exact and polynomial-time for tree ensembles vs model-agnostic sampling approximation.' },
      { q: 'The SHAP caveat', a: 'It explains the model under a stated background distribution, not the world; correlated features share credit as an artefact of the scheme.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.18 */
  ML.section({
    id: 'production', track: 'classical', num: '2.18',
    title: 'Production lifecycle, drift, survival, time series',
    lede: 'Where the previous seventeen sections either survive contact with reality or do not.',
    html: `
<h2><span class="sn">2.18.1</span> The feature store, and the failure it prevents</h2>
<p>A feature store exists to guarantee that the transformation applied at training time is byte-identical to the one applied at serving time. Its absence produces <b>training–serving skew</b>, the most common and least visible production failure: the model is fine, the pipeline is not.</p>
<p>Feature stores have two halves and the split is the whole point: an <b>offline store</b> (columnar, historical, point-in-time correct, used for training) and an <b>online store</b> (low-latency key-value, used for serving), fed by <i>one</i> transformation definition. If the two are computed by different code paths you have re-created training–serving skew inside the tool bought to prevent it. Point-in-time correctness is the hard part — the offline join must reconstruct what was known at each historical decision moment, which is §2.11's leakage checklist expressed in infrastructure.</p>

<h2><span class="sn">2.18.2</span> Two drifts, two detectors</h2>
<p><b>Data drift</b> is a change in $P(x)$; <b>concept drift</b> is a change in $P(y\\mid x)$. You can see data drift immediately from inputs alone; concept drift you can only see once labels arrive, which in credit may be months. Monitor inputs with the <b>population stability index</b>:</p>
$$\\mathrm{PSI} = \\sum_i (a_i - e_i)\\ln\\frac{a_i}{e_i}$$
<p>over matched bins of expected and actual distributions, plus KS for continuous features. Conventional bands: &lt;0.1 stable, 0.1–0.25 moderate, &gt;0.25 significant shift.</p>
${H.flag('Flag: that is the same Jeffreys-divergence form as IV in §2.11 with different inputs, and the thresholds are convention, not derived from any type-I error rate.')}

${H.lab('psi', 'PSI, computed on two distributions you control', 'Drag the actual distribution away from the expected one and watch PSI cross the conventional bands. The per-bin contributions show which part of the population moved — which is the diagnostic, not the total.')}

<h2><span class="sn">2.18.3</span> Retraining and rollout</h2>
<p>Trigger retraining on a drift breach, on sustained metric decay, or on a calendar — and write the trigger down in advance so it is a policy, not a judgement call. Roll out through <b>shadow mode</b> (score, log, don't act), then <b>canary</b> (small traffic share), then <b>champion–challenger</b>. For a bank, model risk governance in the SR 11-7 mould wants documented development, independent validation, ongoing monitoring, and an owner. None of that is optional and all of it is interviewable.</p>

${H.fig('WHAT A MODEL VALIDATOR ASKS FOR (SR 11-7 SHAPE)', H.table(['Section', 'What they want'], [
      ['<b>Conceptual soundness</b>', 'Why this model class, why these features, what is the economic rationale for each sign? Monotonic constraints (§2.8) answer half of this before it is asked.'],
      ['<b>Data lineage</b>', 'Where each feature came from, its availability at decision time, the treatment of missing values, and evidence there is no leakage (§2.11).'],
      ['<b>Outcome analysis</b>', 'Performance on a genuine out-of-time window (§2.14), by segment, with calibration and stability — not one aggregate AUC.'],
      ['<b>Benchmarking</b>', 'Against the incumbent and against a simple challenger; a logistic scorecard is the standard yardstick, and if boosting cannot beat it by a defensible margin, ship the scorecard.'],
      ['<b>Ongoing monitoring plan</b>', 'Which metrics, which thresholds, what happens when one breaches, and who owns it — written before deployment.'],
      ['<b>Limitations, stated by you</b>', 'Where the model should not be used. Validators trust a document that names its own weaknesses far more than one that does not.']
    ]))}

<h3>Champion–challenger, properly</h3>
<p>The challenger runs on a <b>randomised</b> slice of live traffic, so you get a causal read on the decision (§1.7) rather than a backtest. Two traps: the challenger changes the population it sees (approving different customers changes who you observe defaulting), which is a selection effect rather than model quality — reject inference is the credit-specific machinery for it; and outcomes arrive months late, so pre-commit to the evaluation window and the success criterion up front or you will be tempted into §1.6's peeking with a twelve-month lag.</p>

<h2><span class="sn">2.18.4</span> Survival analysis, because default is a time-to-event problem</h2>
<p>The standard credit framing — "did this account default within 12 months?" — throws away two things: <i>when</i> it happened, and every account whose window is not yet complete. Those incomplete accounts are <b>censored</b>, not negative, and scoring them as negatives biases you toward optimism. The <b>Kaplan–Meier</b> estimator gives a non-parametric survival curve $S(t) = P(T > t)$ and is the right first plot for any cohort comparison. <b>Cox proportional hazards</b> models the hazard as $h(t\\mid x) = h_0(t)\\exp(\\beta^\\mathsf{T}x)$ — a non-parametric baseline times a parametric covariate effect — so $e^{\\beta_j}$ is a <b>hazard ratio</b>: 1.4 means a 40% higher instantaneous default rate at every time, all else equal. That is close enough to an odds ratio (§2.4) to be committee-friendly.</p>
<p>What you gain: use of partially-observed accounts, an explicit time dimension (so one model prices a 6-month and a 36-month product), and a natural fit for lifetime expected credit loss under IFRS 9. What you must check: the proportional-hazards assumption itself — if a covariate's effect changes over the life of a loan the model is misspecified, and you test that on the Schoenfeld residuals. And know the modern alternative: gradient boosting with a survival objective (Cox partial likelihood, or an accelerated-failure-time loss) usually beats linear Cox on discrimination while keeping the censoring handling.</p>

${H.lab('km', 'Kaplan–Meier and censoring', 'Two cohorts, with censoring you control. Watch what happens to the naive "default rate" when censored accounts are scored as negatives — and watch the KM curve stay honest.')}

<h2><span class="sn">2.18.5</span> Time series and recommenders, briefly</h2>
<p>Check stationarity, difference or detrend if needed, and know the three live options: classical ARIMA/ETS for few, well-behaved series; gradient-boosted lag and calendar features for tabular-style forecasting at scale (usually the strongest per unit of effort); global deep models when you have thousands of related series. Always backtest with rolling origin (§2.14). <b>Recommenders</b>: collaborative filtering and matrix factorisation on implicit feedback, with cold-start handled by content features.</p>

${H.probe([
      ['Data vs concept drift?', 'Inputs shifting versus the input–output relationship shifting; PSI catches the first, metric decay the second — and only after labels arrive.'],
      ['What is your retrain trigger?', 'PSI > 0.25 on a key feature, or sustained metric decay outside a pre-agreed band — written down before deployment.'],
      ['Why survival rather than a 12-month binary?', 'It uses censored accounts properly, models when rather than whether, and gives hazard ratios plus lifetime loss estimates.']
    ], 'Quoting PSI bands as if they were statistical tests. Call them conventions and you sound like someone who has read the maths.')}`,
    labs: {
      psi: function (host) {
        const st = Viz.controls(host, [
          { k: 'shift', label: 'mean shift of the actual distribution', min: -1.5, max: 1.5, step: .02, value: .3, fmt: v => v.toFixed(2) },
          { k: 'spread', label: 'spread change', min: .5, max: 2.5, step: .02, value: 1, fmt: v => '×' + v.toFixed(2) },
          { k: 'bins', label: 'bins', min: 4, max: 20, step: 1, value: 10, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'psi', label: 'PSI', cls: 'key' }, { k: 'band', label: 'conventional band' },
          { k: 'ks', label: 'KS statistic' }, { k: 'worst', label: 'largest single-bin contribution' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const nb = st.bins, lo = -4, hi = 4, wdt = (hi - lo) / nb;
            const e = [], a = [];
            for (let i = 0; i < nb; i++) {
              const x0 = lo + i * wdt, x1 = x0 + wdt;
              e.push(Num.normCdf(x1, 0, 1) - Num.normCdf(x0, 0, 1));
              a.push(Num.normCdf(x1, st.shift, st.spread) - Num.normCdf(x0, st.shift, st.spread));
            }
            const eS = Num.sum(e), aS = Num.sum(a);
            const eN = e.map(v => Math.max(1e-6, v / eS)), aN = a.map(v => Math.max(1e-6, v / aS));
            const contrib = eN.map((v, i) => (aN[i] - v) * Math.log(aN[i] / v));
            const psi = Num.sum(contrib);
            const P = Viz.plot(ctx, w, h, { xd: [lo, hi], yd: [0, Math.max(Math.max.apply(null, eN), Math.max.apply(null, aN)) * 1.35] })
              .frame({ xlabel: 'feature value (binned)', ylabel: 'population share' });
            P.clip(() => {
              eN.forEach((v, i) => {
                const x0 = lo + i * wdt;
                const px0 = P.x(x0 + wdt * .08), px1 = P.x(x0 + wdt * .48);
                ctx.fillStyle = T.blue; ctx.globalAlpha = .75;
                ctx.fillRect(px0, P.y(v), px1 - px0, P.y(0) - P.y(v));
                const qx0 = P.x(x0 + wdt * .52), qx1 = P.x(x0 + wdt * .92);
                ctx.fillStyle = T.red; ctx.fillRect(qx0, P.y(aN[i]), qx1 - qx0, P.y(0) - P.y(aN[i]));
                ctx.globalAlpha = 1;
                if (contrib[i] > psi * .18) {
                  ctx.fillStyle = T.amber; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                  ctx.fillText(contrib[i].toFixed(3), P.x(x0 + wdt / 2), P.y(Math.max(v, aN[i])) - 4);
                }
              });
            });
            const ks = Math.max.apply(null, eN.map((_, i) => {
              const ce = Num.sum(eN.slice(0, i + 1)), ca = Num.sum(aN.slice(0, i + 1));
              return Math.abs(ce - ca);
            }));
            out({
              psi: psi.toFixed(4),
              band: psi < .1 ? 'stable (<0.1)' : psi < .25 ? 'moderate (0.1–0.25)' : 'significant (>0.25)',
              ks: ks.toFixed(3), worst: Math.max.apply(null, contrib).toFixed(4)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'expected (training)' }, { c: Viz.theme().red, t: 'actual (production)' }, { c: Viz.theme().amber, t: 'per-bin PSI contribution' }]);
        Viz.note(host, 'Amber labels mark the bins carrying the PSI. A total above 0.25 with one bin contributing most of it is a different problem from a uniform spread shift — the first is usually a new segment or a pipeline change, the second a genuine population move.');
      },

      km: function (host) {
        const st = Viz.controls(host, [
          { k: 'hazard', label: 'hazard ratio of cohort B', min: .5, max: 3, step: .05, value: 1.6, fmt: v => v.toFixed(2) + '×' },
          { k: 'censor', label: 'censoring rate', min: 0, max: .8, step: .02, value: .45, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'n', label: 'accounts per cohort', min: 50, max: 800, step: 25, value: 300, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'naive', label: 'naive 12-month default rate (censored = good)', cls: 'bad' },
          { k: 'km', label: 'Kaplan–Meier estimate at 12m', cls: 'key' },
          { k: 'bias', label: 'optimism introduced' }, { k: 'hr', label: 'estimated hazard ratio' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            const cohorts = [0, 1].map(c => {
              const base = .055 * (c ? st.hazard : 1);
              const rows = [];
              for (let i = 0; i < st.n; i++) {
                const t = R.exp(base);
                const cen = R() < st.censor ? R() * 12 : 99;
                rows.push({ t: Math.min(t, cen), event: t <= cen && t <= 12 ? 1 : 0, censored: cen < t });
              }
              return rows;
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, 12], yd: [.6, 1] })
              .frame({ xlabel: 'months since origination', ylabel: 'survival S(t) = P(no default)' });
            const curves = cohorts.map((rows, ci) => {
              const sorted = rows.slice().sort((a, b) => a.t - b.t);
              let atRisk = rows.length, S1 = 1;
              const pts = [[0, 1]];
              sorted.forEach(r => {
                if (r.t > 12) return;
                if (r.event) { S1 *= (1 - 1 / atRisk); pts.push([r.t, S1]); }
                atRisk--;
              });
              pts.push([12, S1]);
              P.clip(() => P.line(pts, { color: ci ? T.red : T.blue, width: 2.4 }));
              return { pts: pts, final: S1, rows: rows };
            });
            // censoring ticks
            P.clip(() => cohorts.forEach((rows, ci) => {
              rows.filter(r => r.censored && r.t <= 12).slice(0, 60).forEach(r => {
                const surv = curves[ci].pts.reduce((acc, p) => p[0] <= r.t ? p[1] : acc, 1);
                P.line([[r.t, surv - .006], [r.t, surv + .006]], { color: ci ? T.red : T.blue, width: 1, alpha: .55 });
              });
            }));
            const naive = Num.mean(cohorts[1].map(r => r.event));
            const kmRate = 1 - curves[1].final;
            out({
              naive: (naive * 100).toFixed(2) + '%', km: (kmRate * 100).toFixed(2) + '%',
              bias: '−' + ((kmRate - naive) * 100).toFixed(2) + ' pts',
              hr: (Math.log(curves[1].final) / Math.log(curves[0].final)).toFixed(2) + '×'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'cohort A' }, { c: Viz.theme().red, t: 'cohort B' }, { c: Viz.theme().faint, t: 'tick marks = censored accounts' }]);
        Viz.note(host, 'Raise censoring and the naive rate drifts further below the Kaplan–Meier estimate: accounts that simply have not been observed long enough are being counted as successes. In a 12-month PD model with heavy attrition, that is a systematic optimism you cannot fix by adding features.');
      }
    },
    quiz: [
      {
        q: 'PSI on a key feature jumps to 0.34 while your AUC is unchanged. What has happened?',
        options: ['Concept drift', 'Data drift — the inputs moved; the relationship may still hold', 'Label leakage', 'Nothing — PSI is unreliable'],
        answer: 1,
        why: 'PSI measures $P(x)$. Concept drift is $P(y|x)$ and shows up as metric decay once labels arrive — which in credit can be months later.'
      },
      {
        q: 'Scoring censored accounts as non-defaults in a 12-month PD model…',
        options: ['is standard and unbiased', 'biases the estimated default rate downward — they are censored, not negative', 'only matters for small samples', 'is fixed by stratified sampling'],
        answer: 1,
        why: 'Kaplan–Meier or a Cox/survival-objective model handles censoring properly and recovers the honest rate.'
      },
      {
        q: 'A challenger model is deployed on a randomised traffic slice. Why randomised rather than a backtest?',
        options: ['It is faster', 'You get a causal read on the decision rather than an estimate confounded by the incumbent’s selection', 'It requires less data', 'Regulators mandate it universally'],
        answer: 1,
        why: 'The incumbent policy chose who you observe. Randomisation severs that arrow — §1.7 applied to deployment.'
      }
    ],
    cards: [
      { q: 'PSI formula and bands', a: '$\\sum_i (a_i-e_i)\\ln(a_i/e_i)$; <0.1 stable, 0.1–0.25 moderate, >0.25 shift — conventions, not tests.' },
      { q: 'Data vs concept drift', a: '$P(x)$ vs $P(y|x)$. PSI sees the first immediately; the second needs labels.' },
      { q: 'Rollout order', a: 'Shadow → canary → champion–challenger on randomised traffic, with a pre-agreed evaluation window.' },
      { q: 'Cox model', a: '$h(t|x)=h_0(t)e^{\\beta^\\mathsf{T}x}$; $e^{\\beta_j}$ is a hazard ratio. Check proportional hazards on Schoenfeld residuals.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.19 */
  ML.section({
    id: 'fairness', track: 'classical', num: '2.19',
    title: 'Fairness, bias, and the regulatory frame',
    lede: 'Unavoidable in lending, frequently asked, and the section most candidates have never thought about properly. Rests on §2.13’s confusion matrix and §2.12’s calibration.',
    html: `
<h2><span class="sn">2.19.1</span> Start with the distinction the law makes</h2>
<p><b>Disparate treatment</b> is using a protected attribute (race, sex, age, religion) in the decision — illegal, and easy to avoid. <b>Disparate impact</b> is a neutral-looking rule that nonetheless produces materially different outcomes across groups — which is where every real argument happens, because <mark>"fairness through unawareness" does not work.</mark> Drop ethnicity and postcode becomes its proxy; drop sex and shopping categories, job titles or device type carry it. A model with no protected attribute can be <i>more</i> discriminatory than one that includes it, because including it at least lets you measure and correct.</p>

<h2><span class="sn">2.19.2</span> The metrics, and what each one asserts</h2>
${H.table(['Criterion', 'Equalises', 'The claim it makes'], [
      ['<b>Demographic parity</b>', 'approval rate', 'Outcomes should match regardless of merit. Strong, and often unlawful to engineer directly.'],
      ['<b>Equal opportunity</b>', 'TPR (recall)', 'Among people who would repay, all groups are approved at the same rate. Usually the most defensible.'],
      ['<b>Equalised odds</b>', 'TPR and FPR', 'Errors of both kinds fall equally on every group. Strictly stronger than equal opportunity.'],
      ['<b>Predictive parity</b>', 'precision', 'A flagged applicant means the same thing in every group.'],
      ['<b>Calibration within groups</b>', 'reliability curve', '"20% risk" means 20% in every group. The minimum bar for a probability used in pricing.']
    ])}

${H.lab('fair', 'The impossibility, made concrete', 'Two groups with different base rates. Try to satisfy calibration within groups and equalised odds at the same time — the readout will tell you exactly how far you are from each, and you will not reach zero on both unless you equalise the base rates or make the classifier perfect.')}

${H.worked('worked number — the four-fifths rule, and the impossibility', `
<p><b>Adverse-impact screen.</b> Group A approval rate 42%, group B 31%. The impact ratio is 31/42 = <b>0.738</b>. Below the conventional 0.80 (four-fifths) threshold, so adverse impact is indicated and the rule now needs a business-necessity justification and a search for a less discriminatory alternative. ⚑ Note the four-fifths rule is a US enforcement <i>convention</i> for employment, widely borrowed elsewhere — cite it as a screen, not as law.</p>
<p><b>Now the result that matters.</b> Suppose true default base rates differ between groups — 10% and 20% — for reasons the model did not create. Kleinberg, Mullainathan & Raghavan (2016) and Chouldechova (2017) proved that <mark>you cannot have calibration within groups and equalised odds at the same time</mark> unless base rates are equal or the classifier is perfect.</p>
<p>The intuition is one line of algebra: precision, recall and prevalence are linked, so fixing two of them across groups over-determines the third. Practical consequence — you must <i>choose</i> which criterion to satisfy and write down why. "We optimise equal opportunity and monitor calibration within groups, because our probabilities feed pricing" is a senior answer. "Our model is fair" is not an answer at all.</p>`)}

<h2><span class="sn">2.19.3</span> Mitigation, at three stages</h2>
<p><b>Pre-processing</b> reweights or resamples the training data, or repairs the features (reweighing, disparate-impact remover) — attractive because the model stays untouched, but it fights §2.12's calibration warning. <b>In-processing</b> adds a fairness constraint or penalty to the objective, or trains an adversary that tries to predict the protected attribute from the representation (adversarial debiasing) — the most principled and the hardest to validate. <b>Post-processing</b> adjusts thresholds per group to equalise a chosen metric — mathematically the cleanest and <b>legally the most dangerous</b>, because group-specific cut-offs can constitute disparate treatment. Never ship one without legal sign-off. Alongside all three: the <i>less discriminatory alternative</i> search — retrain with different feature sets and constraints, and document that you looked. Increasingly that documentation is the compliance artefact, not the model.</p>

<h2><span class="sn">2.19.4</span> The regulatory frame, mid-2026</h2>
<p>In the EU, the AI Act classes creditworthiness assessment of natural persons as a <b>high-risk</b> use (Annex III), which pulls in obligations: risk management, data governance and bias examination, technical documentation, event logging, human oversight, and accuracy/robustness/cybersecurity — plus registration and a conformity assessment. The Act entered into force in August 2024; prohibitions and AI-literacy duties applied from February 2025 and general-purpose-model obligations from August 2025. ⚑ <b>Flag when you cite this:</b> the high-risk timetable has been the subject of active amendment and simplification proposals, so state the direction of travel and check the current date before asserting one. In the US the operative machinery is older and unchanged: ECOA and Regulation B require a specific adverse-action reason (§2.17), and the FCRA governs bureau data. In the UK, Consumer Duty adds an outcomes test — you must show the product works for the customer, which is a different question from whether the model discriminates.</p>

${H.probe([
      ['Just remove the protected attribute — problem solved?', 'No. Proxies reconstruct it, and removing it destroys your ability to measure impact. You generally need it for testing while excluding it from the decision.'],
      ['Which fairness metric would you use?', 'Name one, justify it from the decision’s costs, and say what you monitor alongside it — because the impossibility result means you cannot have them all.'],
      ['What is the four-fifths rule?', 'A US enforcement convention: an impact ratio below 0.80 indicates adverse impact. A screen, not a legal standard, and not a statistical test.']
    ], 'Proposing per-group thresholds as an obviously good fix. It is the one mitigation most likely to create legal exposure.')}`,
    labs: {
      fair: function (host) {
        const st = Viz.controls(host, [
          { k: 'baseA', label: 'group A true default rate', min: .02, max: .4, step: .01, value: .10, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'baseB', label: 'group B true default rate', min: .02, max: .4, step: .01, value: .20, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'thrA', label: 'threshold for group A', min: .02, max: .6, step: .01, value: .15, fmt: v => v.toFixed(2) },
          { k: 'thrB', label: 'threshold for group B', min: .02, max: .6, step: .01, value: .15, fmt: v => v.toFixed(2) },
          { k: 'quality', label: 'model quality', min: .3, max: 3, step: .05, value: 1.3, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ratio', label: 'impact ratio (4/5 rule)', cls: 'key' },
          { k: 'tprGap', label: 'TPR gap (equal opportunity)' },
          { k: 'fprGap', label: 'FPR gap' },
          { k: 'calGap', label: 'calibration gap' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(89);
            function group(base, thr) {
              const n = 4000, scores = [], labels = [];
              for (let i = 0; i < n; i++) {
                const bad = R() < base;
                labels.push(bad ? 1 : 0);
                scores.push(Num.sigmoid(R.normal(bad ? st.quality : -st.quality, 1.3) + Math.log(base / (1 - base))));
              }
              const c = Num.confusion(scores, labels, thr);
              const approved = scores.filter(s => s < thr).length / n;
              // calibration: observed default rate among those predicted ~thr
              const nearIdx = scores.map((s, i) => Math.abs(s - thr) < .06 ? i : -1).filter(i => i >= 0);
              const obs = nearIdx.length ? Num.mean(nearIdx.map(i => labels[i])) : NaN;
              return { c: c, approved: approved, scores: scores, labels: labels, obsAtThr: obs, base: base };
            }
            const A = group(st.baseA, st.thrA), B = group(st.baseB, st.thrB);
            // bars
            const metrics = [
              ['approval rate', A.approved, B.approved],
              ['TPR (catch rate)', A.c.recall, B.c.recall],
              ['FPR', A.c.fp / (A.c.fp + A.c.tn || 1), B.c.fp / (B.c.fp + B.c.tn || 1)],
              ['precision', A.c.precision, B.c.precision],
              ['observed risk @ threshold', A.obsAtThr, B.obsAtThr]
            ];
            const x0 = 172, bw = w - x0 - 90;
            ctx.font = '11px ui-sans-serif'; ctx.textBaseline = 'middle';
            metrics.forEach((m, i) => {
              const y = 30 + i * 52;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText(m[0], x0 - 10, y + 14);
              [1, 2].forEach(k => {
                const v = isFinite(m[k]) ? m[k] : 0;
                ctx.fillStyle = k === 1 ? T.blue : T.red;
                ctx.fillRect(x0, y + (k - 1) * 15, bw * Math.max(0, Math.min(1, v)), 12);
                ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
                ctx.fillText(isFinite(m[k]) ? (m[k] * 100).toFixed(1) + '%' : '—', x0 + bw * Math.max(0, Math.min(1, v)) + 6, y + (k - 1) * 15 + 6);
                ctx.font = '11px ui-sans-serif';
              });
            });
            ctx.fillStyle = T.blue; ctx.fillRect(x0, h - 26, 11, 11);
            ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.fillText('group A', x0 + 16, h - 20);
            ctx.fillStyle = T.red; ctx.fillRect(x0 + 90, h - 26, 11, 11);
            ctx.fillStyle = T.muted; ctx.fillText('group B', x0 + 106, h - 20);
            const ratio = Math.min(A.approved, B.approved) / Math.max(A.approved, B.approved);
            out({
              ratio: ratio.toFixed(3) + (ratio < .8 ? ' ✗' : ' ✓'),
              tprGap: ((A.c.recall - B.c.recall) * 100).toFixed(1) + ' pts',
              fprGap: (((A.c.fp / (A.c.fp + A.c.tn || 1)) - (B.c.fp / (B.c.fp + B.c.tn || 1))) * 100).toFixed(1) + ' pts',
              calGap: isFinite(A.obsAtThr) && isFinite(B.obsAtThr) ? ((A.obsAtThr - B.obsAtThr) * 100).toFixed(1) + ' pts' : '—'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Equal thresholds (calibrated, unequal odds)', primary: true, on: () => { st.$set('thrA', .15); st.$set('thrB', .15); S.redraw(); } },
          { label: 'Try to equalise approval rates', on: () => { st.$set('thrA', .12); st.$set('thrB', .22); S.redraw(); } },
          { label: 'Equal base rates (the escape hatch)', on: () => { st.$set('baseA', .15); st.$set('baseB', .15); S.redraw(); } }
        ]);
        Viz.note(host, 'Press the third button: with equal base rates every criterion can be satisfied at once. Move them apart again and the gaps reappear no matter how you set the thresholds. That is the Kleinberg/Chouldechova impossibility, not a limitation of this simulation.');
      }
    },
    quiz: [
      {
        q: 'Group A approval rate 42%, group B 31%. The impact ratio is…',
        options: ['1.35 — no issue', '0.738 — below the four-fifths screen', '0.11 — below the screen', '0.80 exactly'],
        answer: 1,
        why: '31/42 = 0.738 < 0.80, which indicates adverse impact under the conventional screen and triggers a business-necessity justification.'
      },
      {
        q: 'Base rates genuinely differ between groups. You can simultaneously achieve…',
        options: ['calibration within groups and equalised odds', 'neither', 'calibration within groups OR equalised odds, but not both', 'both, with enough data'],
        answer: 2,
        why: 'Kleinberg et al. (2016) and Chouldechova (2017): unequal base rates make the two mutually exclusive unless the classifier is perfect.'
      },
      {
        q: 'Which mitigation carries the greatest legal risk?',
        options: ['Reweighing the training data', 'Adversarial debiasing', 'Per-group decision thresholds', 'Dropping a proxy feature'],
        answer: 2,
        why: 'Group-specific cut-offs use the protected attribute in the decision, which can constitute disparate treatment. Never ship it without legal sign-off.'
      }
    ],
    cards: [
      { q: 'Disparate treatment vs impact', a: 'Using a protected attribute in the decision vs a neutral rule producing materially unequal outcomes.' },
      { q: 'The impossibility result', a: 'With unequal base rates you cannot have calibration within groups and equalised odds simultaneously (Kleinberg 2016; Chouldechova 2017).' },
      { q: 'Four-fifths rule', a: 'Impact ratio < 0.80 indicates adverse impact — a US enforcement convention used as a screen, not a legal standard.' },
      { q: 'Why unawareness fails', a: 'Proxies reconstruct the attribute, and removing it destroys your ability to measure impact.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.20 */
  ML.section({
    id: 'part2-recall', track: 'classical', num: '2.26',
    title: 'Rapid recall — Part 2 in fourteen lines',
    lede: 'The sheet to read the night before. Every line should unpack into a derivation or a worked number you can reproduce.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', 'Normal equations $w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge adds $\\lambda I$.', '<a href="#/linear-logistic">2.4</a>'],
      ['2', 'Logistic gradient $(\\hat p - y)x$.', '<a href="#/linear-logistic">2.4</a>'],
      ['3', 'Error = bias² + variance + σ²; deeper is not always better.', '<a href="#/bias-variance">2.2</a>'],
      ['4', 'L1 is sparse because of vertices on the axes and a constant subgradient.', '<a href="#/regularization">2.3</a>'],
      ['5', 'Bagging variance $\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$ — decorrelate to beat the floor.', '<a href="#/trees">2.7</a>'],
      ['6', 'XGBoost $w^*=-G/(H+\\lambda)$; gain $=\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{G^2}{H+\\lambda}]-\\gamma$.', '<a href="#/boosting">2.8</a>'],
      ['7', 'The worked split gave gain 0.667 and weights ∓0.667.', '<a href="#/boosting">2.8</a>'],
      ['8', 'EM = tighten the ELBO (E), then maximise it (M).', '<a href="#/unsupervised">2.9</a>'],
      ['9', 'PCA = top eigenvectors of the covariance.', '<a href="#/pca">2.10</a>'],
      ['10', 'Gini = 2·AUC − 1; PR-AUC over ROC under imbalance.', '<a href="#/metrics">2.13</a>'],
      ['11', 'Accuracy 98.6% / precision 40% / recall 80% on the worked matrix.', '<a href="#/metrics">2.13</a>'],
      ['12', 'PSI <0.1 / 0.1–0.25 / >0.25; IV <0.02 / 0.1–0.3 / >0.3 — conventions ⚑.', '<a href="#/production">2.18</a>'],
      ['13', 'Threshold from costs: $p^*=C_{FP}/(C_{FP}+C_{FN})$ — 6.25%, not 0.5.', '<a href="#/metrics">2.13</a>'],
      ['14', 'Conformal: rank residuals, take the $(n{+}1)(1-\\alpha)$ quantile — coverage without assumptions.', '<a href="#/calibration">2.12</a>']
    ])}
${H.table(['#', 'Also memorise', 'Section'], [
      ['15', 'SMOTE harms calibration (JAMIA 2022); recalibrate or do not resample.', '<a href="#/calibration">2.12</a>'],
      ['16', 'Time series: out-of-time and rolling origin, never shuffled k-fold.', '<a href="#/validation">2.14</a>'],
      ['17', 'Fairness: four-fifths screen; calibration and equalised odds are incompatible when base rates differ.', '<a href="#/fairness">2.19</a>'],
      ['18', 'SHAP efficiency: $\\sum\\phi_i = f(x)-\\mathbb{E}[f]$; TreeSHAP exact for trees.', '<a href="#/interpretability">2.17</a>']
    ])}

${H.lab('drill2', 'Part 2 drill', 'Eighteen prompts, shuffled. Answer aloud before flipping.')}`,
    labs: {
      drill2: function (host) {
        const cards = [
          ['Normal equations, and what ridge adds.', '$w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge adds $\\lambda I$ inside the inverse — regularises and guarantees invertibility.'],
          ['Logistic gradient.', '$(\\hat p-y)x$ — prediction minus label times feature.'],
          ['Bias–variance decomposition.', 'bias² + variance + irreducible σ².'],
          ['Why is L1 sparse?', 'Diamond vertices sit on the axes; the constant subgradient $\\lambda\\,\\mathrm{sign}(w)$ pins coefficients at zero.'],
          ['Bagging variance formula.', '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$.'],
          ['XGBoost optimal leaf weight.', '$w^*=-G/(H+\\lambda)$.'],
          ['XGBoost gain formula.', '$\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$.'],
          ['EM in five words.', '"Tighten the bound, then maximise."'],
          ['PCA, in one line.', 'Top eigenvectors of the covariance; $\\Sigma w=\\lambda w$ from the Lagrangian.'],
          ['Gini in terms of AUC.', 'Gini = 2·AUC − 1.'],
          ['Cost-optimal threshold.', '$p^*=C_{FP}/(C_{FP}+C_{FN})$; £60 vs £900 gives 6.25%.'],
          ['The worked confusion matrix numbers.', 'Accuracy 98.6%, precision 40%, recall 80%, F1 0.53 at 1% positives.'],
          ['PSI bands and the caveat.', '<0.1 / 0.1–0.25 / >0.25 — conventions, and the same Jeffreys form as IV.'],
          ['WOE and IV.', '$\\ln(\\%good/\\%bad)$ per bin; $\\mathrm{IV}=\\sum(\\%good-\\%bad)\\mathrm{WOE}$.'],
          ['Split conformal in three steps.', 'Calibration set → nonconformity scores → $\\lceil(n{+}1)(1-\\alpha)\\rceil$ quantile.'],
          ['SMOTE’s documented effect.', 'Harms calibration, rarely improves AUC (JAMIA 2022). Recalibrate or move the threshold.'],
          ['When is group k-fold mandatory?', 'Whenever one entity contributes multiple rows.'],
          ['The fairness impossibility.', 'Calibration within groups and equalised odds cannot both hold when base rates differ.'],
          ['SHAP efficiency property.', '$\\sum_i\\phi_i=f(x)-\\mathbb{E}[f]$.'],
          ['Monotonic constraints — the three benefits.', 'Domain agreement, a validator story, and free regularization.']
        ];
        let order = cards.map((_, i) => i).sort(() => Math.random() - .5);
        let i = 0, showA = false;
        const face = ML.el('div', { class: 'card-face', style: 'cursor:pointer;border:1px solid var(--line);border-radius:12px;background:var(--panel)' });
        host.appendChild(face);
        const pos = ML.el('span');
        function draw() {
          const c = cards[order[i]];
          face.innerHTML = showA ? '<div class="a">' + c[1] + '</div>' : '<div><b>' + c[0] + '</b></div>';
          pos.textContent = (i + 1) + ' / ' + cards.length;
          ML.typeset(face);
        }
        face.addEventListener('click', () => { showA = !showA; draw(); });
        const nav = ML.el('div', { class: 'cardnav' });
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '←', onclick: () => { i = (i - 1 + cards.length) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn primary', type: 'button', text: 'Flip', onclick: () => { showA = !showA; draw(); } }));
        nav.appendChild(pos);
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '→', onclick: () => { i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: () => { order = order.sort(() => Math.random() - .5); i = 0; showA = false; draw(); } }));
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which single Part 2 idea is used by the most other sections?',
        options: ['t-SNE', 'The bias–variance decomposition', 'DBSCAN', 'MAPE'],
        answer: 1,
        why: 'It frames model choice (2.2), regularization (2.3), ensembles (2.7–2.8), validation (2.14) and tuning (2.15).'
      }
    ]
  });
})();
