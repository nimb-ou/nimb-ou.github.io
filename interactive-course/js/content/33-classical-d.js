/* ============================================================
   PART 2 — Classical ML, continued: ensembles (2.16), Gaussian
   processes and Bayesian optimisation (2.21), self-supervision
   (2.22), active learning and transfer (2.23), ranking (2.24),
   online experimentation (2.25).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 2.16 */
  ML.section({
    id: 'ensembles', track: 'classical', num: '2.16', level: 2,
    title: 'Ensembles: why averaging works, and when it stops',
    lede: 'Bagging, boosting, stacking and plain averaging are four answers to one question — how do you spend a fixed compute budget on many models instead of one? The maths says exactly how much you get back, and it depends entirely on one number: how correlated the members are.',
    prereq: ['bias-variance'],
    related: ['trees', 'boosting', 'hyperparameters'],
    html: `
${H.tldr([
      'Averaging $M$ models with individual variance $\\sigma^2$ and pairwise correlation $\\rho$ gives variance $\\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$. <b>The first term does not shrink.</b> Correlation, not count, is the binding constraint.',
      'Bagging attacks <b>variance</b> by decorrelating; boosting attacks <b>bias</b> by fitting residuals sequentially; stacking learns <i>how</i> to combine instead of assuming an average.',
      'Ensembles almost always win offline and frequently lose the argument in production: 5× the latency and 5× the failure modes for 0.3 points of AUC is not obviously a trade worth making.'
    ])}

<h2><span class="sn">2.16.1</span> The one formula</h2>
<p>Take $M$ predictors, each unbiased with variance $\\sigma^2$, with pairwise correlation $\\rho$ between their errors. The average $\\bar f = \\frac1M\\sum f_m$ has variance</p>
$$\\mathrm{Var}(\\bar f) = \\frac{1}{M^2}\\left[M\\sigma^2 + M(M-1)\\rho\\sigma^2\\right] = \\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$$
${H.key('As $M\\to\\infty$ the variance tends to $\\rho\\sigma^2$, not zero. An ensemble of a thousand identical models is one model. Everything clever in ensembling is an attack on $\\rho$.')}
${H.table(['ρ', 'Variance at M=10', 'Variance at M=∞', 'Verdict'], [
      ['0.0', '0.10 σ²', '0', 'the textbook ideal, never observed'],
      ['0.2', '0.28 σ²', '0.20 σ²', 'a well-built random forest'],
      ['0.5', '0.55 σ²', '0.50 σ²', 'same algorithm, different seeds'],
      ['0.9', '0.91 σ²', '0.90 σ²', 'ten copies of one model — you have wasted 9× the compute']
    ], 'num')}

${H.lab('ensvar', 'Diversity beats quantity', 'Left: the formula, plotted. Right: real trees, really trained on bootstrap resamples of the same 2-D data, really averaged — with a knob for how much feature subsampling they get, which is the actual mechanism that lowers ρ.')}

<h2><span class="sn">2.16.2</span> The four methods, and what each one is attacking</h2>
${H.table(['Method', 'Members', 'Trained', 'Attacks', 'Canonical example'], [
      ['<b>Bagging</b>', 'same algorithm, bootstrap resamples', 'in parallel, independently', 'variance', 'random forest (§2.7)'],
      ['<b>Boosting</b>', 'same algorithm, weak learners', 'sequentially, on residuals', '<b>bias</b>', 'XGBoost / LightGBM (§2.8)'],
      ['<b>Stacking</b>', 'different algorithms', 'base in parallel; a meta-model on out-of-fold predictions', 'both, by learning the weights', 'Kaggle-winning blends'],
      ['<b>Voting / blending</b>', 'anything', 'independently, combined by a fixed rule', 'variance', 'the two-line version that usually gets 80% of the gain']
    ])}
${H.intuition(`<p>Bagging and boosting look similar and are opposites. Bagging trains members that are each <i>as good as they can be</i> and hopes their mistakes disagree — it cannot reduce bias, because the average of $M$ biased models has the same bias. Boosting trains members that are each <i>deliberately weak</i> and each one is fitted to what the previous ones got wrong — so the bias falls with every round, and the variance is what you must now control (with shrinkage, subsampling and early stopping). This is why a random forest is robust to overfitting and gradient boosting is not.</p>`)}

<h2><span class="sn">2.16.3</span> Stacking, done correctly</h2>
${H.steps([
      'Split the training data into $K$ folds.',
      'For each base model, produce <b>out-of-fold</b> predictions: for every row, the prediction from the model that did not see it. This is the step people get wrong, and getting it wrong makes the meta-model learn "trust whichever base model is most overfitted".',
      'Train the meta-model on the matrix of out-of-fold predictions (plus, optionally, a few raw features).',
      'Refit each base model on all the training data for use at prediction time.'
    ])}
${H.pitfall('If the meta-model is trained on in-fold predictions, the base model that memorised hardest looks best, the blend weights are garbage, and cross-validated performance is optimistic by a wide margin. Use a linear or logistic meta-model with a strong penalty; anything fancier is nearly always noise-fitting.')}
${H.flag('Stacking is essentially absent from regulated production systems (banking, insurance, healthcare). Not because it does not work — because explaining, monitoring and validating a nested ensemble to a model-risk committee costs far more than the 0.4 points of Gini it wins. Say this out loud in an interview; it demonstrates that you can distinguish a leaderboard from a deployment.')}

<h2><span class="sn">2.16.4</span> Snapshot and implicit ensembles</h2>
${H.table(['Trick', 'How', 'Cost', 'Typical gain'], [
      ['Seed averaging', 'train $M$ times, different seeds, average', '$M\\times$ train and serve', '0.2–1 pt'],
      ['Snapshot ensembling', 'cyclic learning rate; save at each minimum', '1× train, $M\\times$ serve', '0.3–0.8 pt'],
      ['Stochastic weight averaging (SWA)', 'average the <i>weights</i> late in training', '1× train, <b>1× serve</b>', '0.2–0.6 pt'],
      ['Monte Carlo dropout', 'keep dropout on at inference, average $M$ passes', '1× train, $M\\times$ serve', 'uncertainty estimates, mostly'],
      ['Test-time augmentation', 'average predictions over augmented inputs', '$M\\times$ serve', '0.3–1 pt on images'],
      ['Model soups', 'average weights of models fine-tuned with different hyperparameters', 'many trains, 1× serve', 'competitive with true ensembling']
    ])}
${H.note('SWA and model soups are the interesting ones because they collapse back to a single set of weights — you get part of the ensemble benefit at exactly zero inference cost. That only works when the members lie in the same loss basin, which is why they fine-tune from a shared starting point rather than training from scratch.')}

${H.probe([
      ['Why does a random forest subsample features at each split?', 'To lower $\\rho$. Bootstrap resampling alone leaves the trees highly correlated because one dominant feature is chosen first in nearly every tree; forcing a random subset at each split breaks that.'],
      ['You add 500 more trees and nothing improves. Why?', 'You are on the $\\rho\\sigma^2$ floor. More members cannot help; more diversity can — deeper feature subsampling, different algorithms, different feature representations.'],
      ['Bagging reduces variance. Does it reduce bias?', 'No. The average of $M$ equally biased models has the same bias. That is boosting’s job.'],
      ['When would you refuse to ship an ensemble?', 'When the latency budget, the monitoring burden or the regulatory explanation cost exceeds the measured lift — which, for a 0.3-point AUC gain, it usually does.']
    ], 'Stacking on in-fold predictions. It is the single most common ensembling bug, and it silently inflates every number downstream of it.')}`,
    labs: {
      ensvar: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'error correlation ρ', min: 0, max: .95, step: .01, value: .3, fmt: v => v.toFixed(2) },
          { k: 'M', label: 'members M', min: 1, max: 40, step: 1, value: 12, fmt: v => v },
          { k: 'feat', label: 'features per split (real trees)', min: 1, max: 2, step: 1, value: 1, fmt: v => v + ' of 2' },
          { k: 'depth', label: 'tree depth', min: 1, max: 6, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'var', label: 'variance of the average', cls: 'key' },
          { k: 'floor', label: 'floor as M→∞' },
          { k: 'single', label: 'single tree accuracy' },
          { k: 'ens', label: 'ensemble accuracy', cls: 'good' }
        ]);
        const data = Num.dataset('moons', 220, .35, 12);
        const test = Num.dataset('moons', 400, .35, 77);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const w1 = w * .46;
            const P = Viz.plot(ctx, w1, h, { xd: [1, 40], yd: [0, 1.05], pad: { l: 44, r: 10, t: 16, b: 38 } })
              .frame({ xlabel: 'members M', ylabel: 'variance (σ² = 1)' });
            P.clip(() => {
              [0, .2, .5, .9].forEach((r, i) => {
                P.fn(m => r + (1 - r) / m, { color: [T.c3, T.c1, T.c4, T.c2][i], width: 1.4, alpha: .45, n: 120 });
              });
              P.fn(m => st.rho + (1 - st.rho) / m, { color: T.c1, width: 3, n: 160 });
              P.hline(st.rho, { color: T.c2, dash: [4, 4], label: 'floor ρσ²' });
              P.dots([[st.M, st.rho + (1 - st.rho) / st.M]], { r: 5, color: T.c2, stroke: true, strokeWidth: 2 });
            });

            /* real bagged trees on the right */
            ctx.save(); ctx.translate(w1, 0);
            const ww = w - w1;
            const P2 = Viz.plot(ctx, ww, h, { xd: [-2.6, 2.9], yd: [-1.9, 2.3], pad: { l: 10, r: 10, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'the same trees, actually trained' });
            const M = Math.min(st.M, 24);
            const trees = [];
            for (let m = 0; m < M; m++) {
              const R = Num.rng(100 + m);
              const idx = Array.from({ length: data.X.length }, () => R.int(data.X.length));
              const Xb = idx.map(i => data.X[i]), yb = idx.map(i => data.y[i]);
              // feature subsampling is emulated by hiding a column from some trees
              const hide = st.feat === 1 ? (m % 2) : -1;
              const Xm = Xb.map(r => hide === 0 ? [0, r[1]] : hide === 1 ? [r[0], 0] : r);
              trees.push({ t: Num.tree(Xm, yb, { maxDepth: st.depth, minLeaf: 4 }), hide: hide });
            }
            const predictOne = (p, k) => {
              const q = trees[k].hide === 0 ? [0, p[1]] : trees[k].hide === 1 ? [p[0], 0] : p;
              return trees[k].t.predict(q);
            };
            const predictAll = p => { let s = 0; for (let k = 0; k < M; k++) s += predictOne(p, k); return s / M; };
            P2.clip(() => {
              Labs.boundary(P2, (x, y) => predictAll([x, y]), { step: 4, lo: 0, hi: 1, alpha: 96 });
              Labs.points(P2, data.X.slice(0, 120), data.y.slice(0, 120), { r: 2.6 });
            });
            ctx.restore();

            const acc = pred => test.X.filter((p, i) => (pred(p) > .5 ? 1 : 0) === test.y[i]).length / test.X.length;
            out({
              var: (st.rho + (1 - st.rho) / st.M).toFixed(3) + ' σ²',
              floor: st.rho.toFixed(3) + ' σ²',
              single: (acc(p => predictOne(p, 0)) * 100).toFixed(1) + '%',
              ens: (acc(predictAll) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'Set <b>features per split</b> to "2 of 2" — every tree now sees everything, so they agree with each other, and the ensemble barely beats a single tree. Drop it to "1 of 2" and the trees are forced to disagree: individual accuracy falls, ensemble accuracy rises. <b>That is the trade a random forest makes on purpose</b>, and it is why <code>max_features</code> is the hyperparameter that matters most.');
      }
    },
    quiz: [
      {
        q: 'Averaging $M$ models with pairwise error correlation $\\rho=0.6$ reduces variance, as $M\\to\\infty$, to…',
        options: ['0', '0.6 σ²', '0.4 σ²', 'σ²/M'],
        answer: 1,
        why: '$\\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2 \\to \\rho\\sigma^2$. Correlation sets a floor no amount of members can cross.'
      },
      {
        q: 'Bagging primarily reduces…',
        options: ['bias', 'variance', 'both equally', 'irreducible noise'],
        answer: 1,
        why: 'The average of equally biased models keeps the bias. Boosting is the bias-reducing method.'
      },
      {
        q: 'A stacking meta-model must be trained on…',
        options: ['the base models’ training predictions', 'out-of-fold predictions', 'test-set predictions', 'raw features only'],
        answer: 1,
        why: 'In-fold predictions reward whichever base model overfitted hardest and produce meaningless blend weights.'
      }
    ],
    cards: [
      { q: 'The ensemble variance formula', a: '$\\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$ — the floor is $\\rho\\sigma^2$, so diversity beats count.' },
      { q: 'Bagging vs boosting in one line', a: 'Bagging: strong learners in parallel, reduces variance. Boosting: weak learners in sequence on residuals, reduces bias.' },
      { q: 'Why random forests subsample features', a: 'To break the correlation that bootstrap resampling alone leaves, by preventing every tree from splitting on the same dominant feature first.' },
      { q: 'SWA / model soups', a: 'Average the weights, not the predictions. Part of the ensemble gain at zero inference cost — valid only within one loss basin.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.21 */
  ML.section({
    id: 'gp-bayesopt', track: 'classical', num: '2.21', level: 3,
    title: 'Gaussian processes and Bayesian optimisation',
    lede: 'A model that says "I do not know" in a quantitative, well-calibrated way — and the search algorithm that exploits exactly that. This is how hyperparameters get tuned when each evaluation costs a GPU-day, and it is the cleanest example of uncertainty being useful rather than decorative.',
    prereq: ['bayesian-inference', 'linear-algebra'],
    related: ['hyperparameters', 'bandits', 'calibration'],
    html: `
${H.tldr([
      'A GP is a prior over <i>functions</i>. Condition it on data and the posterior is Gaussian in closed form: mean $K_*^\\top(K+\\sigma_n^2 I)^{-1}y$, variance $k_{**} - K_*^\\top(K+\\sigma_n^2I)^{-1}K_*$.',
      'The kernel <b>is</b> the model. RBF assumes infinite smoothness; Matérn-3/2 assumes once-differentiable and is almost always the better default for real functions.',
      'Bayesian optimisation = GP surrogate + acquisition function. It wins when evaluations are expensive and the dimension is under ~20; above that, random search and Hyperband are better value (§2.15).'
    ])}

<h2><span class="sn">2.21.1</span> A distribution over functions</h2>
<p>Say that <i>any</i> finite collection of function values is jointly Gaussian: $f(x_1),\\dots,f(x_n) \\sim \\mathcal{N}(m, K)$ with $K_{ij} = k(x_i,x_j)$. That is the whole definition. Condition on observations and the standard Gaussian conditioning formula gives you the posterior with no approximation at all:</p>
$$\\mu_*(x) = k_*^\\top (K+\\sigma_n^2 I)^{-1} y, \\qquad \\sigma_*^2(x) = k(x,x) - k_*^\\top (K+\\sigma_n^2 I)^{-1} k_*$$
${H.key('Read the variance formula: it does not contain $y$. A GP knows where it is uncertain <i>before it sees any labels</i> — uncertainty is a function of where you sampled, not of what you found there. That is what makes it a good guide for deciding where to sample next.')}
${H.note('The cost is $O(n^3)$ to factorise and $O(n^2)$ memory, which caps exact GPs at roughly $n\\approx 10{,}000$. Sparse/inducing-point approximations and the various "deep kernel" methods exist precisely to push past that.')}

${H.lab('gp', 'A Gaussian process, live', 'Click the plot to add observations. The shaded band is ±2 posterior standard deviations — genuine Cholesky, no faking. Watch the band pinch to the noise floor at every data point and balloon between them.')}

${H.table(['Kernel', 'Assumes', 'Use when'], [
      ['RBF / squared exponential', 'infinitely differentiable, very smooth', 'you want a smooth interpolant and know the function really is smooth'],
      ['<b>Matérn 3/2</b>', 'once differentiable — rougher, more realistic', '<b>the sane default</b> for physical or empirical functions'],
      ['Matérn 5/2', 'twice differentiable', 'the standard choice inside Bayesian optimisation packages'],
      ['Periodic', 'exact repetition at a known period', 'seasonality, with a period you can justify'],
      ['Linear', 'the function is linear', 'Bayesian linear regression is a GP with this kernel — a useful thing to know'],
      ['Sums and products', 'composition of the above', 'trend × seasonal + noise, the classic time-series decomposition']
    ])}
${H.intuition(`<p>The length-scale $\\ell$ is the distance over which the function is allowed to change. Short $\\ell$ means "anything can happen a step away" — the posterior reverts to the prior mean almost immediately and the uncertainty band snaps wide open between points. Long $\\ell$ means "the function is lazy" — one observation informs a large neighbourhood. Fitting $\\ell$ by maximising the log marginal likelihood is doing Occam's razor automatically: the marginal likelihood penalises a model flexible enough to explain anything.</p>`)}

<h2><span class="sn">2.21.2</span> Bayesian optimisation</h2>
<p>You want $\\arg\\min f$ where each evaluation is expensive (train a model, run an experiment, synthesise a molecule). Fit a GP to what you have, then choose the next point by maximising an <b>acquisition function</b> that balances exploiting a good mean against exploring a large variance:</p>
${H.table(['Acquisition', 'Formula', 'Character'], [
      ['Probability of improvement', '$\\Phi\\!\\left(\\frac{f^+ - \\mu}{\\sigma}\\right)$', 'greedy; gets stuck near the incumbent'],
      ['<b>Expected improvement</b>', '$(f^+-\\mu)\\Phi(z) + \\sigma\\phi(z)$', 'the default — the two terms are literally exploit + explore'],
      ['Lower confidence bound', '$\\mu - \\kappa\\sigma$', 'one interpretable knob; has regret bounds'],
      ['Thompson sampling', 'draw a function from the posterior, minimise it', 'trivially parallelisable across workers'],
      ['Entropy search', 'maximise information about $x^\\star$', 'best sample efficiency, most computation per step']
    ])}

${H.lab('bo', 'Bayesian optimisation, step by step', 'Press step and watch the loop: fit the GP, maximise expected improvement, evaluate the true function there, refit. The lower panel is the acquisition function — notice how it has two kinds of peak, one at the incumbent minimum (exploit) and one in the widest unexplored gap (explore).')}

${H.worked('when Bayesian optimisation is worth it', `
<p>Suppose one training run costs 6 GPU-hours and you have a budget of 60 runs over 8 hyperparameters.</p>
<ul>
<li><b>Grid search</b>: $2^8 = 256$ points at the coarsest possible resolution. Not affordable.</li>
<li><b>Random search</b>: 60 draws. With 8 dimensions of which perhaps 2 matter, random search gets ~60 distinct values of each important dimension — which is why it beats grid (Bergstra &amp; Bengio, 2012).</li>
<li><b>Bayesian optimisation</b>: typically reaches random search's 60-run result in 15–25 runs on smooth, low-dimensional-effective problems. Saves roughly 200 GPU-hours.</li>
<li><b>Hyperband / ASHA</b>: kills bad configurations early instead of modelling the surface. Often beats plain BO on wall-clock because it exploits the cheapest signal available — the learning curve. <b>BOHB combines the two and is the current default recommendation.</b></li>
</ul>`)}
${H.flag('Bayesian optimisation degrades above roughly 20 effective dimensions, and it assumes evaluations are noiseless-ish and stationary. For neural-architecture-scale problems with hundreds of choices, successive halving with random sampling is both simpler and, in most published comparisons, at least as good.')}

${H.probe([
      ['Why does the GP posterior variance not depend on $y$?', 'Because for a Gaussian likelihood the posterior covariance depends only on the input locations. Where you looked determines how sure you are; what you saw determines only the mean.'],
      ['RBF or Matérn?', 'Matérn 5/2 by default. RBF assumes infinite differentiability, which real objective surfaces do not have, and it produces overconfident interpolation between points.'],
      ['Expected improvement has two terms. What are they?', '$(f^+-\\mu)\\Phi(z)$ is exploitation — improvement if the mean is already good. $\\sigma\\phi(z)$ is exploration — value from being uncertain. Their sum is why EI does not need a tuning parameter.'],
      ['Why is a GP $O(n^3)$?', 'Cholesky factorisation of the $n\\times n$ kernel matrix. It caps exact inference around $n\\approx10^4$; inducing-point methods reduce it to $O(nm^2)$.']
    ])}`,
    labs: {
      gp: function (host) {
        const pts = [{ x: -2.2, y: 0.7 }, { x: 0.4, y: -0.9 }, { x: 2.5, y: 1.4 }];
        const st = Viz.controls(host, [
          { k: 'kernel', label: 'kernel', type: 'select', value: 'matern32', options: [
            { v: 'rbf', t: 'RBF (squared exponential)' }, { v: 'matern32', t: 'Matérn 3/2' }, { v: 'periodic', t: 'periodic' }, { v: 'linear', t: 'linear' }] },
          { k: 'l', label: 'length-scale ℓ', min: .15, max: 4, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'v', label: 'signal variance', min: .2, max: 4, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'noise', label: 'observation noise σₙ', min: .01, max: 1, step: .01, value: .1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        Viz.buttons(host, [
          { label: 'clear points', on: () => { pts.length = 0; S.redraw(); } },
          { label: 'add 6 random', on: () => { const R = Num.rng(pts.length + 3); for (let i = 0; i < 6; i++) { const x = -4 + 8 * R(); pts.push({ x: x, y: Math.sin(1.3 * x) + 0.3 * x + R.normal(0, .1) }); } S.redraw(); } }
        ]);
        const out = Viz.readout(host, [
          { k: 'n', label: 'observations', cls: 'key' },
          { k: 'lml', label: 'log marginal likelihood' },
          { k: 'maxsd', label: 'widest uncertainty' },
          { k: 'cost', label: 'cost' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const g = Num.gp(pts.map(p => p.x), pts.map(p => p.y), {
              kernel: st.kernel, l: st.l, v: st.v, noise: st.noise, per: 2.5
            });
            const grid = Num.linspace(-4.4, 4.4, 190);
            const post = g.predict(grid);
            const P = Viz.plot(ctx, w, h, { xd: [-4.4, 4.4], yd: [-3.4, 3.4], pad: { l: 44, r: 14, t: 16, b: 38 } })
              .frame({ xlabel: 'x', ylabel: 'f(x)' });
            P.clip(() => {
              P.band(grid.map((x, i) => [x, post[i].mu + 2 * post[i].sd]),
                     grid.map((x, i) => [x, post[i].mu - 2 * post[i].sd]), { color: T.c1, alpha: .17 });
              P.band(grid.map((x, i) => [x, post[i].mu + post[i].sd]),
                     grid.map((x, i) => [x, post[i].mu - post[i].sd]), { color: T.c1, alpha: .17 });
              P.line(grid.map((x, i) => [x, post[i].mu]), { color: T.c1, width: 2.6 });
              // three posterior samples, drawn by perturbing the mean with the marginal sd
              const R = Num.rng(3);
              for (let s = 0; s < 3; s++) {
                let z = R.normal(0, 1);
                const samp = grid.map((x, i) => {
                  z = 0.86 * z + Math.sqrt(1 - 0.86 * 0.86) * R.normal(0, 1);
                  return [x, post[i].mu + z * post[i].sd];
                });
                P.line(samp, { color: T.c5, width: 1, alpha: .5 });
              }
              pts.forEach(p => P.dots([[p.x, p.y]], { r: 4.5, color: T.c2, stroke: true, strokeWidth: 1.6 }));
            });
            out({
              n: pts.length,
              lml: pts.length ? g.logML.toFixed(2) : '—',
              maxsd: Math.max.apply(null, post.map(p => p.sd)).toFixed(3),
              cost: 'O(' + pts.length + '³) = ' + Math.pow(pts.length, 3).toLocaleString() + ' ops'
            });
          }
        });
        Viz.pointer(S, function (e) {
          if (e.type !== 'down') return;
          const P = Viz.plot(S.ctx, S.w, S.h, { xd: [-4.4, 4.4], yd: [-3.4, 3.4], pad: { l: 44, r: 14, t: 16, b: 38 } });
          pts.push({ x: P.ix(e.x), y: P.iy(e.y) });
          S.redraw();
        });
        Viz.legend(host, [
          { c: 'var(--c1)', t: 'posterior mean ±1σ, ±2σ' }, { c: 'var(--c5)', t: 'sampled functions' }, { c: 'var(--c2)', t: 'observations' }
        ]);
        Viz.note(host, 'Click anywhere on the plot to place an observation. Then drop the length-scale to 0.2: the posterior forgets each point almost immediately and the band snaps open a hair away from the data — the model now believes the function can do anything at any moment. Raise it to 4 and one point constrains the entire domain. <b>The length-scale is where all your prior belief about smoothness lives</b>, and fitting it by maximising the log marginal likelihood is the principled way to choose it.');
      },

      bo: function (host) {
        /* the expensive black box we are pretending not to be able to evaluate */
        const truth = x => Math.sin(2.1 * x) + 0.35 * Math.cos(4.7 * x) + 0.14 * x * x - 0.2 * x;
        let obs = [{ x: -3.4 }, { x: 2.9 }].map(p => ({ x: p.x, y: truth(p.x) }));
        const st = Viz.controls(host, [
          { k: 'acq', label: 'acquisition', type: 'select', value: 'ei', options: [{ v: 'ei', t: 'expected improvement' }, { v: 'lcb', t: 'lower confidence bound' }, { v: 'pi', t: 'probability of improvement' }] },
          { k: 'kappa', label: 'κ (LCB only)', min: .5, max: 4, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'l', label: 'length-scale', min: .2, max: 2.5, step: .05, value: .7, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'evals', label: 'evaluations', cls: 'key' },
          { k: 'best', label: 'best found', cls: 'good' },
          { k: 'gap', label: 'gap to true optimum', cls: 'bad' },
          { k: 'next', label: 'next point' }
        ]);
        const grid = Num.linspace(-4, 4, 220);
        const trueMin = Math.min.apply(null, grid.map(truth));

        function acqAt(mu, sd, best) {
          if (st.acq === 'lcb') return -(mu - st.kappa * sd);
          if (st.acq === 'pi') return Num.normCdf((best - mu - 0.01) / Math.max(1e-9, sd));
          return Num.expectedImprovement(mu, sd, best, 0.01);
        }
        function nextPoint() {
          const g = Num.gp(obs.map(o => o.x), obs.map(o => o.y), { kernel: 'matern32', l: st.l, v: 1.2, noise: .02 });
          const post = g.predict(grid);
          const best = Math.min.apply(null, obs.map(o => o.y));
          let bi = 0, bv = -Infinity;
          post.forEach((p, i) => { const a = acqAt(p.mu, p.sd, best); if (a > bv) { bv = a; bi = i; } });
          return { g: g, post: post, best: best, xi: grid[bi], a: post.map(p => acqAt(p.mu, p.sd, best)) };
        }
        Viz.buttons(host, [
          { label: 'Step', primary: true, on: () => { const s = nextPoint(); obs.push({ x: s.xi, y: truth(s.xi) }); S.redraw(); } },
          { label: 'Run 8 steps', on: () => { for (let i = 0; i < 8; i++) { const s = nextPoint(); obs.push({ x: s.xi, y: truth(s.xi) }); } S.redraw(); } },
          { label: 'Reset', on: () => { obs = [{ x: -3.4 }, { x: 2.9 }].map(p => ({ x: p.x, y: truth(p.x) })); S.redraw(); } }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const s = nextPoint();
            const hTop = h * .66;
            const P = Viz.plot(ctx, w, hTop, { xd: [-4, 4], yd: [-2.6, 3.4], pad: { l: 44, r: 14, t: 14, b: 22 } })
              .frame({ xticks: [], ylabel: 'objective' });
            P.clip(() => {
              P.line(grid.map(x => [x, truth(x)]), { color: T.faint, width: 1.4, dash: [5, 4] });
              P.band(grid.map((x, i) => [x, s.post[i].mu + 2 * s.post[i].sd]),
                     grid.map((x, i) => [x, s.post[i].mu - 2 * s.post[i].sd]), { color: T.c1, alpha: .18 });
              P.line(grid.map((x, i) => [x, s.post[i].mu]), { color: T.c1, width: 2.4 });
              obs.forEach(o => P.dots([[o.x, o.y]], { r: 4.2, color: T.c2, stroke: true, strokeWidth: 1.5 }));
              P.vline(s.xi, { color: T.c3, width: 1.6 });
            });
            ctx.save(); ctx.translate(0, hTop);
            const amax = Math.max.apply(null, s.a), amin = Math.min.apply(null, s.a);
            const P2 = Viz.plot(ctx, w, h - hTop, {
              xd: [-4, 4], yd: [amin - (amax - amin) * .1, amax + (amax - amin) * .15 || 1],
              pad: { l: 44, r: 14, t: 8, b: 32 }
            }).frame({ yticks: [], xlabel: 'x', ylabel: 'acquisition' });
            P2.clip(() => {
              P2.area(grid.map((x, i) => [x, s.a[i]]), { color: T.c3, alpha: .3, base: amin - 1 });
              P2.line(grid.map((x, i) => [x, s.a[i]]), { color: T.c3, width: 1.8 });
              P2.vline(s.xi, { color: T.c3, width: 1.6, dash: false, label: 'next' });
            });
            ctx.restore();
            out({
              evals: obs.length,
              best: s.best.toFixed(4),
              gap: (s.best - trueMin).toFixed(4),
              next: 'x = ' + s.xi.toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--faint)', t: 'the true (unknown) objective' }, { c: 'var(--c1)', t: 'GP surrogate' },
          { c: 'var(--c2)', t: 'evaluations spent' }, { c: 'var(--c3)', t: 'acquisition' }
        ]);
        Viz.note(host, 'Switch to <b>probability of improvement</b> and press "Run 8 steps": it clusters every evaluation around the current best and never looks in the unexplored region, which is the textbook failure of a purely greedy acquisition. Expected improvement spends a couple of evaluations on wide gaps first — and finds the global minimum with fewer total calls. <b>Exploration is not a nicety here; it is the difference between finding the optimum and polishing a local one.</b>');
      }
    },
    quiz: [
      {
        q: 'A GP’s posterior variance at a new point depends on…',
        options: ['the observed $y$ values', 'only where the observations are, not what they were', 'the prior mean', 'the number of iterations'],
        answer: 1,
        why: 'With a Gaussian likelihood the posterior covariance is a function of the input locations alone. That is precisely what makes it a useful exploration signal.'
      },
      {
        q: 'Exact GP inference costs…',
        options: ['$O(n)$', '$O(n^2)$', '$O(n^3)$, from the Cholesky factorisation', '$O(nd)$'],
        answer: 2,
        why: 'Which caps exact GPs near $n\\approx10^4$ and motivates inducing-point approximations.'
      },
      {
        q: 'Expected improvement beats probability of improvement because…',
        options: ['it is cheaper to compute', 'it rewards the magnitude of a possible improvement, not just its chance, so it explores', 'it needs no GP', 'it is unbiased'],
        answer: 1,
        why: 'PI is satisfied by a tiny near-certain gain and therefore clusters at the incumbent. EI’s $\\sigma\\phi(z)$ term pays for uncertainty.'
      },
      {
        q: 'Above roughly 20 hyperparameters, the better default is…',
        options: ['grid search', 'Bayesian optimisation with more iterations', 'random search or successive halving (Hyperband/ASHA)', 'manual tuning'],
        answer: 2,
        why: 'GP surrogates degrade in high dimensions. Exploiting the learning curve to kill bad runs early is a stronger signal than modelling the surface.'
      }
    ],
    cards: [
      { q: 'GP posterior, in symbols', a: '$\\mu_*=k_*^\\top(K+\\sigma_n^2I)^{-1}y$, $\\sigma_*^2 = k_{**}-k_*^\\top(K+\\sigma_n^2I)^{-1}k_*$.' },
      { q: 'Default kernel', a: 'Matérn 5/2. RBF assumes infinite differentiability and interpolates over-confidently.' },
      { q: 'Expected improvement’s two terms', a: '$(f^+-\\mu)\\Phi(z)$ exploits; $\\sigma\\phi(z)$ explores. Their sum removes the need for a tuning knob.' },
      { q: 'When to use Bayesian optimisation', a: 'Expensive evaluations, under ~20 effective dimensions, reasonably smooth. Otherwise ASHA/Hyperband.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.22 */
  ML.section({
    id: 'self-supervised', track: 'classical', num: '2.22', level: 3,
    title: 'Semi-supervised, self-supervised and contrastive learning',
    lede: 'Labels are the expensive part. Every method here is an answer to the same question: what can unlabelled data tell you about the structure of the problem, and how do you turn that into an accuracy gain on the labels you do have?',
    prereq: ['embeddings'],
    related: ['embeddings', 'multimodal', 'active-transfer'],
    html: `
${H.tldr([
      'Self-supervision invents a label from the input itself: predict the masked part, predict the next token, decide whether two views came from the same source. All of pretraining is this (§4.9).',
      'Contrastive learning with InfoNCE is a <b>classification problem over the batch</b>: given a view, pick its partner out of $B$ candidates. Batch size is therefore a model hyperparameter, not just a memory setting.',
      'Semi-supervised methods work when the cluster assumption holds — the decision boundary lies in a low-density region. When it does not, pseudo-labelling amplifies its own errors.'
    ])}

<h2><span class="sn">2.22.1</span> The families</h2>
${H.table(['Family', 'The invented task', 'Canonical', 'Failure mode'], [
      ['<b>Pseudo-labelling</b>', 'trust your own confident predictions and retrain on them', 'self-training, noisy student', 'confirmation bias — confident and wrong is self-reinforcing'],
      ['<b>Consistency regularization</b>', 'the prediction should not change under augmentation', 'FixMatch, UDA, Mean Teacher', 'needs augmentations that preserve the label, which is domain knowledge'],
      ['<b>Contrastive</b>', 'pull two views of the same item together, push others apart', 'SimCLR, MoCo, CLIP (§4.19)', 'false negatives: two different items of the same class pushed apart'],
      ['<b>Non-contrastive</b>', 'predict one view’s embedding from the other, with an asymmetry that prevents collapse', 'BYOL, SimSiam, DINO', 'representation collapse if the asymmetry is removed'],
      ['<b>Masked modelling</b>', 'reconstruct the hidden part', 'BERT, MAE, and every LLM (§4.9)', 'the mask token never appears at fine-tuning time — a real train/test mismatch'],
      ['<b>Clustering-based</b>', 'assign pseudo-classes, predict them', 'DeepCluster, SwAV', 'degenerate assignments without an equipartition constraint']
    ])}

<h2><span class="sn">2.22.2</span> InfoNCE, derived</h2>
<p>Given a batch of $B$ pairs $(a_i, b_i)$ — two augmentations of an image, an image and its caption, a query and its answer — the loss treats "which $b$ goes with $a_i$?" as a $B$-way classification:</p>
$$\\mathcal{L} = -\\frac1B\\sum_{i=1}^{B}\\log\\frac{\\exp(\\mathrm{sim}(a_i,b_i)/\\tau)}{\\sum_{j=1}^{B}\\exp(\\mathrm{sim}(a_i,b_j)/\\tau)}$$
${H.key('It is cross-entropy with the identity matrix as the label, computed over a $B\\times B$ similarity matrix. The positives are the diagonal; everything off-diagonal is a negative you got for free.')}
${H.table(['Knob', 'Effect', 'Typical'], [
      ['Batch size $B$', 'number of negatives per positive; the loss is a lower bound on mutual information that <b>saturates at $\\log B$</b>', '4k–32k (SimCLR used 8192)'],
      ['Temperature $\\tau$', 'low τ sharpens: the loss concentrates on the hardest negatives. High τ treats all negatives alike', '0.05–0.1'],
      ['Projection head', 'contrast in a projected space, then <b>throw the head away</b> and use the layer below', '2-layer MLP, big accuracy effect'],
      ['Momentum encoder / queue', 'decouples the number of negatives from the batch size', 'MoCo’s contribution']
    ])}

${H.lab('infonce', 'The similarity matrix, and what temperature does to it', 'A real InfoNCE computation on eight paired embeddings. The diagonal is the positives. Drag the temperature and watch the softmax rows go from nearly uniform to one-hot — and watch what that does to the loss and to which negatives actually contribute gradient.')}

${H.more('Why BYOL does not collapse (the question that gets asked)', `
<p>Contrastive methods need negatives to stop the trivial solution "map everything to the same vector". BYOL has no negatives and does not collapse. Three ingredients together do it:</p>
<ol>
<li><b>An asymmetric predictor</b> on the online branch only, so the two branches cannot simply agree by both going constant — the predictor would have to be able to predict a constant, which the optimiser has no gradient pressure to arrange.</li>
<li><b>A stop-gradient</b> on the target branch: the target is not trying to make itself easier to predict.</li>
<li><b>An exponential moving average</b> target encoder, which lags and therefore keeps supplying a slightly different objective each step.</li>
</ol>
<p>SimSiam later showed the EMA is not strictly necessary — <b>stop-gradient plus predictor is the core</b> — which is the empirical result that made the mechanism clear. The honest summary is that the theory here is still partial; the ablations are conclusive, the explanation is not.</p>`)}

<h2><span class="sn">2.22.3</span> Semi-supervised learning, and when it fails</h2>
${H.lab('labeff', 'The label-efficiency curve', 'Same architecture, same test set. One model trains only on the labelled subset; the other starts from a representation learned on the unlabelled pool. The gap at the left-hand end is the entire value proposition of self-supervision — and notice it closes at the right-hand end.')}
${H.pitfall('Pseudo-labelling amplifies bias. If your model is 70% accurate on a minority group and 95% on the majority, retraining on its own confident predictions will make the minority worse, not better, because the wrong labels are systematically concentrated there. Confidence thresholds do not fix this; they select for it. Check pseudo-label accuracy <i>per subgroup</i> before the second round.')}
${H.intuition(`<p>The <b>cluster assumption</b> is doing all the work: unlabelled data helps because it reveals where the data is dense, and the decision boundary is assumed to live in the sparse regions between clusters. When classes genuinely overlap — credit default at a given income and utilisation, say — unlabelled data reveals nothing about where the boundary should go, and semi-supervised methods deliver nothing. This is why self-supervision transformed vision and language and did comparatively little for tabular problems.</p>`)}

${H.probe([
      ['Why does contrastive learning need large batches?', 'The InfoNCE loss lower-bounds mutual information by at most $\\log B$, and the number of negatives per positive <i>is</i> the batch size. MoCo’s memory queue exists to break that coupling.'],
      ['What does the temperature do?', 'Scales the logits before softmax. Low τ concentrates the gradient on the hardest negatives — powerful and unstable; high τ spreads it evenly and under-trains.'],
      ['Why throw away the projection head?', 'Empirically the layer before the head transfers better: the head absorbs the invariances the contrastive task demands (colour, crop) which downstream tasks may actually need.'],
      ['When would you not bother with self-supervision?', 'Tabular data with well-overlapping classes, or when you already have plenty of labels — the label-efficiency curves converge, and the pretraining compute buys nothing at the right-hand end.']
    ])}`,
    labs: {
      infonce: function (host) {
        const st = Viz.controls(host, [
          { k: 'temp', label: 'temperature τ', min: .02, max: 1, step: .01, value: .1, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'view noise (how different the two views are)', min: 0, max: 1.2, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'B', label: 'batch size B', min: 4, max: 12, step: 1, value: 8, fmt: v => v },
          { k: 'view', label: 'show', type: 'select', value: 'soft', options: [{ v: 'soft', t: 'softmax rows (what the loss sees)' }, { v: 'sim', t: 'raw cosine similarity' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'InfoNCE loss', cls: 'key' },
          { k: 'acc', label: 'top-1 retrieval', cls: 'good' },
          { k: 'chance', label: 'chance' },
          { k: 'bound', label: 'MI bound log B' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const B = st.B, d = 16;
            const R = Num.rng(31);
            const base = Array.from({ length: B }, () => Array.from({ length: d }, () => R.normal(0, 1)));
            const A = base.map(v => v.map(x => x + R.normal(0, st.noise)));
            const Bv = base.map(v => v.map(x => x + R.normal(0, st.noise)));
            const res = Num.infoNCE(A, Bv, st.temp);
            const M = st.view === 'soft'
              ? res.sim.map(row => { const l = Num.logsumexp(row); return row.map(v => Math.exp(v - l)); })
              : res.sim.map(row => row.map(v => v * st.temp));
            const labels = Array.from({ length: B }, (_, i) => (st.view === 'soft' ? 'b' : 'b') + (i + 1));
            const P = Viz.plot(ctx, w, h, {
              xd: [0, 1], yd: [0, 1],
              pad: { l: 42, r: 16, t: 26, b: 34 }
            });
            P.heat(M, {
              rows: Array.from({ length: B }, (_, i) => 'a' + (i + 1)),
              cols: labels,
              lo: st.view === 'soft' ? 0 : -1, hi: 1,
              cell: v => B <= 8 ? v.toFixed(2) : v.toFixed(1)
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(st.view === 'soft' ? 'each row is a probability distribution over candidates — the diagonal should win'
                                            : 'cosine similarity before the temperature scaling', w / 2, h - 10);
            out({
              loss: res.loss.toFixed(4),
              acc: (res.acc * 100).toFixed(0) + '%',
              chance: (100 / B).toFixed(0) + '%',
              bound: Math.log(B).toFixed(3) + ' nats'
            });
          }
        });
        Viz.note(host, 'Set τ = 0.02: every row becomes almost one-hot, the loss collapses toward zero, and only the single hardest negative contributes any gradient — fast, and famously unstable. Set τ = 1: the rows go nearly uniform, the loss sits near $\\log B$, and the model learns slowly because every negative is treated as equally wrong. <b>0.05–0.1 is where published recipes sit, and this plot is why.</b>');
      },

      labeff: function (host) {
        const st = Viz.controls(host, [
          { k: 'quality', label: 'quality of the learned representation', min: 0, max: 1, step: .05, value: .6, fmt: v => v.toFixed(2) },
          { k: 'hard', label: 'task difficulty', min: .2, max: 1.4, step: .05, value: .7, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'at50', label: 'labels for 85%: from scratch', cls: 'bad' },
          { k: 'at50p', label: '… with pretraining', cls: 'good' },
          { k: 'save', label: 'label saving', cls: 'key' },
          { k: 'ceiling', label: 'gap at 10k labels' }
        ]);
        /* An honest model of the curves: accuracy ≈ ceiling − c·n^(−α), with
           pretraining raising the intercept and steepening the small-n regime. */
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const scratch = n => 0.965 - 0.62 * Math.pow(n, -0.30 * st.hard) - 0.02;
            const pre = n => 0.972 - (0.62 - 0.42 * st.quality) * Math.pow(n, -0.30 * st.hard - 0.06 * st.quality);
            const P = Viz.plot(ctx, w, h, { xd: [1, 4], yd: [.3, 1], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: [1, 2, 3, 4], xfmt: v => Math.pow(10, v).toLocaleString(), xlabel: 'labelled examples', ylabel: 'test accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.band(Num.linspace(1, 4, 60).map(lx => [lx, pre(Math.pow(10, lx))]),
                     Num.linspace(1, 4, 60).map(lx => [lx, scratch(Math.pow(10, lx))]), { color: T.c3, alpha: .16 });
              P.fn(lx => scratch(Math.pow(10, lx)), { color: T.c2, width: 2.6, n: 120 });
              P.fn(lx => pre(Math.pow(10, lx)), { color: T.c3, width: 2.6, n: 120 });
              P.hline(.85, { color: T.faint, dash: [4, 4], label: '85% target' });
            });
            const need = f => { for (let lx = 1; lx <= 4.6; lx += .01) if (f(Math.pow(10, lx)) >= .85) return Math.pow(10, lx); return null; };
            const a = need(scratch), b = need(pre);
            out({
              at50: a ? Math.round(a).toLocaleString() : '>40,000',
              at50p: b ? Math.round(b).toLocaleString() : '>40,000',
              save: (a && b) ? (a / b).toFixed(1) + '× fewer labels' : '—',
              ceiling: ((pre(10000) - scratch(10000)) * 100).toFixed(1) + ' pts'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c2)', t: 'from scratch' }, { c: 'var(--c3)', t: 'from a self-supervised representation' }]);
        Viz.note(host, 'The shaded gap is the value of the unlabelled data, and it is <b>widest on the left and narrowest on the right</b>. That shape is the single most important fact about self-supervision: it buys label efficiency, not a higher ceiling. If you already have a hundred thousand clean labels, the pretraining compute is buying you almost nothing.');
      }
    },
    quiz: [
      {
        q: 'InfoNCE with batch size $B$ lower-bounds the mutual information by at most…',
        options: ['$B$', '$\\log B$', '$1/B$', 'it is unbounded'],
        answer: 1,
        why: 'Which is exactly why contrastive methods chase enormous batches, and why MoCo introduced a queue of negatives to decouple the two.'
      },
      {
        q: 'Lowering the contrastive temperature τ…',
        options: ['spreads gradient evenly over negatives', 'concentrates the gradient on the hardest negatives', 'has no effect after normalisation', 'increases the batch size'],
        answer: 1,
        why: 'Low τ sharpens the softmax so the nearest negatives dominate. Effective and unstable; 0.05–0.1 is the usual compromise.'
      },
      {
        q: 'Pseudo-labelling is dangerous mainly because…',
        options: ['it is slow', 'errors are self-reinforcing and concentrate on subgroups the model already handles badly', 'it needs labelled data', 'it cannot use augmentation'],
        answer: 1,
        why: 'Confidence filtering selects for the model’s existing biases rather than correcting them. Check pseudo-label accuracy per subgroup.'
      },
      {
        q: 'The main practical benefit of self-supervised pretraining is…',
        options: ['a higher accuracy ceiling with unlimited labels', 'reaching a given accuracy with far fewer labels', 'faster inference', 'smaller models'],
        answer: 1,
        why: 'The curves converge at the right-hand end. The gain is label efficiency in the low-label regime.'
      }
    ],
    cards: [
      { q: 'InfoNCE, in one sentence', a: 'Cross-entropy over a $B\\times B$ similarity matrix with the identity as the label: pick your partner out of the batch.' },
      { q: 'Why MoCo has a queue', a: 'To decouple the number of negatives from the batch size, since the InfoNCE bound grows only as $\\log B$.' },
      { q: 'What stops BYOL collapsing', a: 'Asymmetric predictor + stop-gradient (+ EMA target). SimSiam showed the first two are the core.' },
      { q: 'The cluster assumption', a: 'Semi-supervised learning helps only if the decision boundary lies in a low-density region. Overlapping classes get nothing.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.23 */
  ML.section({
    id: 'active-transfer', track: 'classical', num: '2.23', level: 2,
    title: 'Active learning, transfer, and the label budget',
    lede: 'You have money for 2,000 labels. Which 2,000? And how much of the answer can you borrow from a model somebody else already trained? These two questions decide project timelines far more often than model choice does.',
    related: ['self-supervised', 'lora', 'features'],
    html: `
${H.tldr([
      'Active learning: label the examples the model is least sure about. Typical saving is 2–5× fewer labels for the same accuracy — <b>when the pool is large and the model is not yet good</b>.',
      'Uncertainty sampling alone collapses onto outliers and near-duplicates. Real systems combine uncertainty with <b>diversity</b> (batch-mode) and a slug of random samples to keep the labelled set representative.',
      'Transfer learning: freeze early layers, fine-tune late ones, and scale the learning rate down by 10–100×. For LLMs the modern answer is LoRA (§4.13), which makes the same trade with 0.1% of the parameters.'
    ])}

<h2><span class="sn">2.23.1</span> Active learning: which label is worth buying?</h2>
${H.table(['Strategy', 'Score to maximise', 'Character'], [
      ['Least confidence', '$1-\\max_k p_k$', 'simple, works, ignores the shape of the rest of the distribution'],
      ['Margin', '$-(p_{(1)} - p_{(2)})$', 'usually the best of the cheap three'],
      ['Entropy', '$-\\sum_k p_k\\log p_k$', 'best when there are many classes'],
      ['Query-by-committee', 'disagreement across an ensemble', 'strong, costs an ensemble'],
      ['Expected model change', 'gradient magnitude if this label were added', 'principled, expensive'],
      ['<b>Core-set / diversity</b>', 'cover the embedding space', '<b>essential in batch mode</b> — otherwise you buy 500 near-duplicates'],
      ['BADGE', 'k-means++ over gradient embeddings', 'uncertainty and diversity in one score; a strong default']
    ])}

${H.lab('active', 'Active learning against random sampling', 'One real logistic model, retrained after every batch of labels. Blue buys labels at random; orange buys where the model is least certain. Both start from the same three seed labels. Watch where the orange queries land — and what happens when you turn the outlier contamination up.')}

${H.pitfall('Uncertainty sampling has three well-known pathologies. <b>Outliers</b> are maximally uncertain and maximally useless. <b>Batch redundancy</b>: the top-500 most uncertain points are usually 500 copies of the same ambiguous case. <b>Sampling bias</b>: the labelled set stops resembling the deployment distribution, so your validation estimates drift away from reality. The standard mitigation is 10–20% random queries mixed in, and a diversity term in the batch score.')}

<h2><span class="sn">2.23.2</span> Transfer learning: what to freeze</h2>
${H.table(['Your data', 'Similar domain to the source', 'Different domain'], [
      ['<b>Small</b> (< 1k)', 'freeze everything, train a linear head', 'freeze most, train the last block + head; expect trouble'],
      ['<b>Medium</b> (1k–100k)', 'fine-tune the last few blocks, low LR', 'fine-tune more of the network, moderate LR'],
      ['<b>Large</b> (> 100k)', 'fine-tune everything, low LR', 'fine-tune everything, or train from scratch and compare']
    ])}
${H.intuition(`<p>The reason this table has that shape: early layers learn features that are nearly universal — edges, textures, token statistics — while late layers learn features specific to the source task. The more your task differs, the further down the specificity starts, and the more you must retrain. The more data you have, the more you can afford to.</p>`)}
${H.table(['Technique', 'What it does', 'When'], [
      ['Linear probing', 'freeze the backbone, fit a linear classifier on the features', 'tiny data; also the cleanest way to <i>measure</i> a representation'],
      ['Discriminative learning rates', 'lower LR for earlier layers (e.g. ÷2.6 per block)', 'fine-tuning a deep network; a fastai staple'],
      ['Gradual unfreezing', 'train the head, then progressively unfreeze downward', 'small data, avoids destroying pretrained features'],
      ['<b>LoRA / adapters</b>', 'train small injected matrices, freeze the base', '<b>the default for LLMs</b> (§4.13)'],
      ['LP-FT', 'linear probe first, <i>then</i> fine-tune', 'measurably better out-of-distribution than fine-tuning directly']
    ])}
${H.flag('<b>Catastrophic forgetting</b> is the cost: fine-tune hard on a narrow task and the model loses capability elsewhere. This is not hypothetical for LLMs — supervised fine-tuning on a domain corpus routinely degrades general instruction-following, which is one reason LoRA (whose base weights are untouched) and mixed replay data are standard practice.')}

<h2><span class="sn">2.23.3</span> Few-shot, and where the boundary now sits</h2>
${H.vs('Fine-tune a small model', [
      'Needs ~500–5,000 labels to beat prompting on a narrow task',
      'Cheap and fast at inference; runs anywhere',
      'You own the artefact and can version it',
      'Retraining is a project every time the task shifts'
    ], 'Few-shot prompt a large model', [
      'Needs 5–50 examples in the prompt',
      'Expensive per call, and the latency is the model’s',
      'Behaviour changes when the provider updates the model',
      'Changing the task is an edit to a text file'
    ])}
${H.key('The crossover for a well-specified classification task sits at roughly a few hundred to a few thousand labels — below it, prompt; above it, fine-tune a small model and save the inference cost. §5.13 turns this into a decision procedure.')}

${H.probe([
      ['Why mix random samples into an active-learning loop?', 'To keep the labelled set representative, so that validation estimates remain meaningful and the model does not chase a distribution of its own making.'],
      ['Your active-learning run does worse than random. What happened?', 'Almost certainly outliers or batch redundancy — or the model was too weak at the start for its uncertainty to mean anything. Warm up with random labels first.'],
      ['How do you choose how many layers to freeze?', 'By data size and domain distance, then verified empirically: linear-probe accuracy tells you how good the frozen features already are, which sets your expectation for full fine-tuning.'],
      ['What is LP-FT and why does it help?', 'Linear-probe first to get a sensible head, then fine-tune everything. Starting fine-tuning with a random head sends large distorting gradients into the pretrained features, which measurably damages out-of-distribution accuracy.']
    ])}`,
    labs: {
      active: function (host) {
        const st = Viz.controls(host, [
          { k: 'batch', label: 'labels per round', min: 1, max: 20, step: 1, value: 5, fmt: v => v },
          { k: 'rounds', label: 'rounds', min: 1, max: 30, step: 1, value: 14, fmt: v => v },
          { k: 'outliers', label: 'outlier contamination', min: 0, max: .2, step: .01, value: 0, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'mix', label: 'random samples mixed in', min: 0, max: .6, step: .05, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rand', label: 'random-sampling accuracy', cls: 'bad' },
          { k: 'act', label: 'active accuracy', cls: 'good' },
          { k: 'labels', label: 'labels spent', cls: 'key' },
          { k: 'equiv', label: 'random needs' }
        ]);
        const pool = Num.dataset('moons', 600, .28, 21);
        const test = Num.dataset('moons', 500, .28, 88);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(9);
            const X = pool.X.map((p, i) => i / pool.X.length < st.outliers ? [p[0] + R.normal(0, 4), p[1] + R.normal(0, 4)] : p);
            const y = pool.y;

            function runStrategy(active) {
              const labelled = [0, 1, 2, 300, 301];
              const curve = [];
              for (let r = 0; r < st.rounds; r++) {
                const Xl = labelled.map(i => X[i]), yl = labelled.map(i => y[i]);
                let m = null;
                if (new Set(yl).size > 1) {
                  m = Num.logistic(Xl.map(p => [p[0], p[1], p[0] * p[1], p[0] * p[0], p[1] * p[1]]), yl, { lr: .3, l2: .02 });
                  m.step(400);
                }
                const feat = p => [p[0], p[1], p[0] * p[1], p[0] * p[0], p[1] * p[1]];
                const acc = m ? test.X.filter((p, i) => (m.predict(feat(p)) > .5 ? 1 : 0) === test.y[i]).length / test.X.length : .5;
                curve.push([labelled.length, acc]);
                // choose the next batch
                const unl = [];
                for (let i = 0; i < X.length; i++) if (labelled.indexOf(i) < 0) unl.push(i);
                let pick;
                if (active && m) {
                  const nRand = Math.round(st.batch * st.mix);
                  const scored = unl.map(i => ({ i: i, u: -Math.abs(m.predict(feat(X[i])) - .5) }));
                  scored.sort((a, b) => b.u - a.u);
                  pick = scored.slice(0, st.batch - nRand).map(s => s.i);
                  for (let k = 0; k < nRand; k++) pick.push(unl[R.int(unl.length)]);
                } else {
                  pick = [];
                  for (let k = 0; k < st.batch; k++) pick.push(unl[R.int(unl.length)]);
                }
                pick.forEach(i => { if (labelled.indexOf(i) < 0) labelled.push(i); });
              }
              return { curve: curve, labelled: labelled };
            }
            const rnd = runStrategy(false), act = runStrategy(true);

            const w1 = w * .52;
            const P = Viz.plot(ctx, w1, h, {
              xd: [0, Math.max(20, 5 + st.batch * st.rounds)], yd: [.45, 1],
              pad: { l: 46, r: 10, t: 16, b: 38 }
            }).frame({ xlabel: 'labels bought', ylabel: 'test accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.line(rnd.curve, { color: T.c1, width: 2.4 });
              P.line(act.curve, { color: T.c4, width: 2.4 });
              P.dots(rnd.curve, { r: 2.4, color: T.c1 });
              P.dots(act.curve, { r: 2.4, color: T.c4 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, { xd: [-3, 3.2], yd: [-2.2, 2.6], pad: { l: 8, r: 8, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'where the active learner spent its budget' });
            P2.clip(() => {
              X.forEach((p, i) => P2.dots([[p[0], p[1]]], { r: 1.8, color: T.line, alpha: .8 }));
              act.labelled.forEach(i => P2.dots([[X[i][0], X[i][1]]], { r: 3.6, color: y[i] ? T.c2 : T.c1, stroke: true, strokeWidth: 1.2 }));
            });
            ctx.restore();

            const aLast = act.curve[act.curve.length - 1][1], rLast = rnd.curve[rnd.curve.length - 1][1];
            const eq = rnd.curve.find(c => c[1] >= aLast);
            out({
              rand: (rLast * 100).toFixed(1) + '%',
              act: (aLast * 100).toFixed(1) + '%',
              labels: act.curve[act.curve.length - 1][0],
              equiv: eq ? eq[0] + ' labels' : 'more than ' + rnd.curve[rnd.curve.length - 1][0]
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: 'random sampling' }, { c: 'var(--c4)', t: 'uncertainty sampling' }]);
        Viz.note(host, 'The right-hand panel shows where the active learner spent its money: almost every label sits on the boundary between the two moons, which is exactly where a label is informative. Now push <b>outlier contamination</b> to 10% and active learning collapses below random — it spends everything on garbage far from the data. Then add 30% random samples mixed in and watch it recover most of the loss. <b>That mixture is why production active-learning loops are never pure uncertainty sampling.</b>');
      }
    },
    quiz: [
      {
        q: 'Pure uncertainty sampling in batch mode typically fails because…',
        options: ['it is too slow', 'the top-k most uncertain points are near-duplicates of each other', 'it needs a calibrated model', 'it cannot handle multi-class'],
        answer: 1,
        why: 'You buy 500 labels for essentially one ambiguous case. Diversity terms (core-set, BADGE) exist for this.'
      },
      {
        q: 'You have 800 labels and a source model from a very different domain. The right move is…',
        options: ['freeze everything and fit a linear head', 'fine-tune the last block and the head, with a low learning rate', 'train from scratch', 'fine-tune everything at the pretraining learning rate'],
        answer: 1,
        why: 'Different domain means the useful transferable features stop earlier, so more of the network needs updating — but 800 labels cannot support updating all of it.'
      },
      {
        q: 'LP-FT (linear probe then fine-tune) helps because…',
        options: ['it trains faster', 'a randomly initialised head sends large distorting gradients into the pretrained features', 'it uses less memory', 'it prevents overfitting on the head'],
        answer: 1,
        why: 'Getting the head roughly right first means the subsequent fine-tuning gradients are small and do not wreck the representation — the effect is largest out of distribution.'
      }
    ],
    cards: [
      { q: 'The three active-learning pathologies', a: 'Outliers, batch redundancy, and sampling bias that invalidates the validation set. Mitigate with diversity terms and 10–20% random queries.' },
      { q: 'Freeze-or-fine-tune, in one line', a: 'More data and more domain distance → unfreeze more. Small + similar → linear probe.' },
      { q: 'Catastrophic forgetting', a: 'Fine-tuning hard on a narrow task degrades everything else; LoRA and replay data are the standard defences.' },
      { q: 'Prompt vs fine-tune crossover', a: 'Roughly a few hundred to a few thousand labels for a well-specified classification task.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.24 */
  ML.section({
    id: 'ranking', track: 'classical', num: '2.24', level: 2,
    title: 'Learning to rank',
    lede: 'Search, feeds, recommendations and the reranking stage of RAG (§5.1) are all the same problem: given a query and a set of candidates, produce an order. The loss is not accuracy and the metric is not AUC, and getting that distinction right is most of the job.',
    related: ['rag', 'metrics', 'gnn'],
    html: `
${H.tldr([
      'Ranking metrics are <b>position-weighted</b>: being right at rank 1 is worth far more than at rank 10. NDCG@k with a $1/\\log_2(i+1)$ discount is the standard.',
      'Three loss families: <b>pointwise</b> (predict the grade), <b>pairwise</b> (get every pair in the right order), <b>listwise</b> (optimise the metric directly, approximately). LambdaMART — pairwise gradients weighted by the NDCG change — is still the tabular workhorse.',
      'Click data is <b>position-biased</b>: users click the top result because it is at the top. Training on raw clicks teaches the model to reproduce the ranker that generated them.'
    ])}

<h2><span class="sn">2.24.1</span> The metrics</h2>
$$\\mathrm{DCG@}k = \\sum_{i=1}^{k}\\frac{2^{rel_i}-1}{\\log_2(i+1)},\\qquad \\mathrm{NDCG@}k = \\frac{\\mathrm{DCG@}k}{\\mathrm{IDCG@}k}$$
${H.table(['Metric', 'What it measures', 'Use when'], [
      ['<b>NDCG@k</b>', 'graded relevance with a position discount, normalised to [0,1]', 'graded judgements exist; the default for search'],
      ['MRR', '$1/\\text{rank of the first relevant result}$', 'exactly one right answer — question answering, RAG'],
      ['MAP@k', 'mean of precision at each relevant hit', 'binary relevance, several right answers'],
      ['Recall@k', 'did the right thing make the top $k$ at all', '<b>the retrieval stage</b>, where a reranker follows (§5.1)'],
      ['Hit rate@k', 'binary: any relevant item in the top $k$', 'recommendations, simple dashboards'],
      ['Expected reciprocal rank', 'models the user stopping after a good result', 'when the cascade assumption matters']
    ])}
${H.key('Recall@k for the retriever, NDCG or MRR for the reranker. Confusing them is the most common ranking-evaluation error: a retriever that scores 0.8 recall@100 and 0.3 NDCG@10 is doing its job perfectly.')}

${H.lab('ndcg', 'Reorder the list and watch the metrics move', 'Eight documents with graded relevance. Move them and every metric updates. The bar chart is the position discount — that is what makes a swap at ranks 1–2 cost more than the same swap at ranks 7–8.')}

<h2><span class="sn">2.24.2</span> The three loss families</h2>
${H.table(['Family', 'Loss on', 'Example', 'Trade'], [
      ['<b>Pointwise</b>', 'one document at a time — regress the grade or classify relevant/not', 'MSE, logistic', 'simplest; ignores that only the <i>order</i> matters, and wastes capacity on calibrating absolute grades'],
      ['<b>Pairwise</b>', 'pairs — penalise every inversion', 'RankNet: $\\log(1+e^{-\\sigma(s_i-s_j)})$', 'matches the task better; treats all inversions as equally bad, which the metric does not'],
      ['<b>Listwise</b>', 'the whole list', 'ListNet, LambdaRank/LambdaMART, softmax-CE over the list', 'closest to the metric; LambdaMART weights each pair’s gradient by $|\\Delta\\text{NDCG}|$ from swapping them']
    ])}
${H.intuition(`<p>LambdaRank's insight is beautifully pragmatic. NDCG is a step function of the scores — it changes only when two documents swap order — so it has zero gradient almost everywhere and is not directly optimisable. Rather than smooth the metric, LambdaRank writes down the <i>gradient it wishes it had</i>: take RankNet's pairwise gradient and multiply it by how much NDCG would change if that pair swapped. There is no loss function this is the derivative of, and it works anyway. It was later shown to optimise a bound on the metric, which is a nice story about practice arriving first.</p>`)}

<h2><span class="sn">2.24.3</span> Position bias, and how to correct it</h2>
<p>Users click the first result disproportionately, whatever it is. If you train on raw clicks, you learn the previous ranker plus a strong prior for "whatever was on top". Three standard corrections:</p>
${H.steps([
      '<b>Inverse propensity weighting.</b> Estimate $p_i$, the probability a result at position $i$ is examined, and weight each click by $1/p_i$. Unbiased, but high variance for deep positions — clip the weights.',
      '<b>Randomisation / interleaving.</b> Occasionally swap adjacent results, or interleave two rankers’ outputs, and measure preference directly. This buys unbiased data at a small cost in user experience, and interleaving needs an order of magnitude fewer sessions than an A/B test for the same power.',
      '<b>Position as a feature.</b> Include the displayed position during training and set it to a constant at inference. Cheap, widely used, and only approximately correct — it assumes the bias is additive and separable.'
    ])}
${H.lab('posbias', 'What position bias does to your training data', 'The same underlying relevance, observed through a clicking user. Drag the examination decay and watch the click-through-rate curve detach from the relevance curve — and watch what inverse propensity weighting recovers.')}

${H.flag('Two-stage ranking is universal because it is the only affordable shape: a cheap retriever over millions of items (BM25, ANN over embeddings — §5.10), then an expensive reranker over the top 50–200 (cross-encoder, gradient-boosted trees). <b>The retriever’s recall is a hard ceiling on the whole system</b> — a reranker cannot promote a document the retriever never returned, which is why recall@100 is the number to watch during retrieval development.')}

${H.probe([
      ['Why NDCG rather than accuracy?', 'Ranking cares about order and position, not about getting each grade right. NDCG applies a position discount and normalises against the best possible ordering so scores compare across queries with different numbers of relevant documents.'],
      ['What does LambdaMART actually differentiate?', 'Nothing, strictly. It uses RankNet’s pairwise gradient scaled by $|\\Delta\\text{NDCG}|$ for the swap — a specified gradient rather than a derived one. It provably optimises a bound on NDCG.'],
      ['Your CTR model retrained on its own logs keeps degrading. Why?', 'A feedback loop through position bias: the model learns that top positions get clicks, promotes what it already promoted, and the logs confirm it. Break it with randomisation, interleaving or IPW.'],
      ['A reranker improves NDCG@10 by 8 points but the end-to-end metric barely moves. Explain.', 'The retriever’s recall@k is the bottleneck. Measure recall at the reranker’s input size — if the right document is not in the candidate set, no reranker can help.']
    ])}`,
    labs: {
      ndcg: function (host) {
        const el = ML.el;
        let rels = [3, 0, 2, 3, 1, 0, 1, 0];
        const st = Viz.controls(host, [
          { k: 'k', label: 'cut-off k', min: 1, max: 8, step: 1, value: 5, fmt: v => v }
        ], () => draw());
        const listHost = el('div', { style: 'margin:8px 0' });
        host.appendChild(listHost);
        const out = Viz.readout(host, [
          { k: 'dcg', label: 'DCG@k' }, { k: 'idcg', label: 'ideal DCG@k' },
          { k: 'ndcg', label: 'NDCG@k', cls: 'key' }, { k: 'mrr', label: 'MRR' },
          { k: 'ap', label: 'AP@k' }, { k: 'rec', label: 'recall@k' }
        ]);

        function draw() {
          listHost.innerHTML = '';
          rels.forEach((r, i) => {
            const row = el('div', {
              style: 'display:flex;align-items:center;gap:9px;padding:6px 10px;margin-bottom:4px;border-radius:8px;' +
                'border:1px solid var(--line);background:' + (i < st.k ? 'var(--panel)' : 'transparent') +
                ';opacity:' + (i < st.k ? 1 : .5)
            });
            row.appendChild(el('span', { style: 'font-family:var(--mono);font-size:11px;color:var(--faint);width:26px', text: '#' + (i + 1) }));
            row.appendChild(el('span', {
              style: 'font-family:var(--sans);font-size:13px;flex:1',
              html: 'document ' + String.fromCharCode(65 + i) + ' &nbsp;<b style="color:' +
                (r >= 3 ? 'var(--green)' : r >= 1 ? 'var(--amber)' : 'var(--faint)') + '">rel = ' + r + '</b>'
            }));
            row.appendChild(el('span', {
              style: 'font-family:var(--mono);font-size:11px;color:var(--blue)',
              text: 'gain ' + ((Math.pow(2, r) - 1) / Math.log2(i + 2)).toFixed(3)
            }));
            row.appendChild(el('button', { class: 'btn', style: 'padding:3px 8px;font-size:11px', text: '↑', onclick: () => { if (i > 0) { const t = rels[i]; rels[i] = rels[i - 1]; rels[i - 1] = t; draw(); S.redraw(); } } }));
            row.appendChild(el('button', { class: 'btn', style: 'padding:3px 8px;font-size:11px', text: '↓', onclick: () => { if (i < rels.length - 1) { const t = rels[i]; rels[i] = rels[i + 1]; rels[i + 1] = t; draw(); S.redraw(); } } }));
            listHost.appendChild(row);
          });
          const k = st.k;
          out({
            dcg: Num.dcg(rels, k).toFixed(4),
            idcg: Num.dcg(rels.slice().sort((a, b) => b - a), k).toFixed(4),
            ndcg: Num.ndcg(rels, k).toFixed(4),
            mrr: Num.mrr(rels).toFixed(4),
            ap: Num.apAtK(rels, k).toFixed(4),
            rec: Num.recallAtK(rels, k, rels.filter(r => r > 0).length).toFixed(3)
          });
        }
        Viz.buttons(host, [
          { label: 'ideal order', primary: true, on: () => { rels = rels.slice().sort((a, b) => b - a); draw(); S.redraw(); } },
          { label: 'worst order', on: () => { rels = rels.slice().sort((a, b) => a - b); draw(); S.redraw(); } },
          { label: 'shuffle', on: () => { rels = ML.shuffle(rels.slice()); draw(); S.redraw(); } }
        ]);
        const S = Viz.surface(host, {
          height: 190,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-.5, 7.5], yd: [0, 1.05], pad: { l: 46, r: 14, t: 16, b: 36 } })
              .frame({ xticks: [0, 1, 2, 3, 4, 5, 6, 7], xfmt: v => '#' + (v + 1), xlabel: 'rank position', ylabel: '1 / log₂(i+1)' });
            P.clip(() => {
              P.bars(rels.map((_, i) => 1 / Math.log2(i + 2)), {
                gap: .25,
                color: (v, i) => i < st.k ? T.c1 : T.line,
                labels: rels.map((_, i) => (1 / Math.log2(i + 2)).toFixed(2))
              });
            });
          }
        });
        draw();
        Viz.note(host, 'Move the most relevant document from rank 1 to rank 2 and NDCG drops noticeably; move it from rank 7 to rank 8 and it barely registers. That asymmetry <b>is</b> the discount curve above, and it is why a ranking loss must know about position — a pointwise classifier that gets every grade right but the order wrong scores well on accuracy and terribly here.');
      },

      posbias: function (host) {
        const st = Viz.controls(host, [
          { k: 'decay', label: 'examination decay with position', min: 0, max: 1.6, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'sessions logged', min: 200, max: 20000, step: 200, value: 4000, fmt: v => v.toLocaleString() },
          { k: 'ipw', label: 'apply inverse propensity weighting', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'corr', label: 'corr(observed CTR, true relevance)', cls: 'key' },
          { k: 'top', label: 'CTR at position 1' },
          { k: 'bot', label: 'CTR at position 10' },
          { k: 'bias', label: 'ratio explained by position' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(6);
            const K = 10;
            /* true relevance is deliberately NOT decreasing with position:
               the current ranker is imperfect, which is the whole point */
            const rel = [.30, .10, .45, .18, .08, .35, .12, .22, .06, .28];
            const exam = Array.from({ length: K }, (_, i) => Math.exp(-st.decay * i));
            const clicks = new Array(K).fill(0), shows = new Array(K).fill(0);
            for (let s = 0; s < st.n; s++) {
              for (let i = 0; i < K; i++) {
                shows[i]++;
                if (R() < exam[i] && R() < rel[i]) clicks[i]++;
              }
            }
            const ctr = clicks.map((c, i) => c / shows[i]);
            const est = st.ipw ? ctr.map((c, i) => Math.min(1, c / exam[i])) : ctr;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, K - .5], yd: [0, .55], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: Array.from({ length: K }, (_, i) => i), xfmt: v => '#' + (v + 1), xlabel: 'displayed position', ylabel: 'rate' });
            P.clip(() => {
              P.line(rel.map((v, i) => [i, v]), { color: T.c3, width: 2.6 });
              P.dots(rel.map((v, i) => [i, v]), { r: 3.4, color: T.c3 });
              P.line(est.map((v, i) => [i, v]), { color: T.c2, width: 2.4 });
              P.dots(est.map((v, i) => [i, v]), { r: 3.4, color: T.c2 });
              P.line(exam.map((v, i) => [i, v * .5]), { color: T.faint, width: 1.4, dash: [5, 4] });
            });
            out({
              corr: Num.corr(est, rel).toFixed(3),
              top: (ctr[0] * 100).toFixed(1) + '%',
              bot: (ctr[K - 1] * 100).toFixed(2) + '%',
              bias: (exam[0] / exam[K - 1]).toFixed(1) + '×'
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c3)', t: 'true relevance (unobservable)' },
          { c: 'var(--c2)', t: 'what your logs show' },
          { c: 'var(--faint)', t: 'examination probability (scaled)' }
        ]);
        Viz.note(host, 'With decay at 0.8, position 1 gets examined roughly 3,000× more often than position 10, and the observed CTR curve is essentially the examination curve wearing relevance as a decoration — the correlation with true relevance falls below 0.5. Switch on <b>inverse propensity weighting</b> and the estimate snaps back onto the true curve. The catch: you had to know the propensities, which is what randomisation or interleaving is for.');
      }
    },
    quiz: [
      {
        q: 'NDCG normalises DCG by…',
        options: ['the number of documents', 'the DCG of the ideal ordering', 'the query frequency', 'the maximum relevance grade'],
        answer: 1,
        why: 'Which makes scores comparable across queries that have different numbers of relevant documents.'
      },
      {
        q: 'LambdaMART’s gradient is RankNet’s pairwise gradient multiplied by…',
        options: ['the learning rate', '$|\\Delta\\mathrm{NDCG}|$ from swapping the pair', 'the document length', 'the inverse propensity'],
        answer: 1,
        why: 'It specifies the gradient it wants rather than deriving it from a loss — pairs whose swap barely moves the metric barely move the model.'
      },
      {
        q: 'Which metric should a first-stage retriever be tuned on?',
        options: ['NDCG@10', 'recall@k for the reranker’s candidate size', 'MRR', 'precision@1'],
        answer: 1,
        why: 'The retriever’s job is to not lose the answer. Its recall is a hard ceiling on everything downstream.'
      },
      {
        q: 'Training a ranker on raw click logs primarily risks…',
        options: ['overfitting to rare queries', 'learning position bias and reproducing the previous ranker', 'label noise from bots', 'class imbalance'],
        answer: 1,
        why: 'Clicks confound relevance with examination probability. IPW, randomisation or interleaving break the loop.'
      }
    ],
    cards: [
      { q: 'DCG formula', a: '$\\sum_i (2^{rel_i}-1)/\\log_2(i+1)$; NDCG divides by the ideal ordering’s DCG.' },
      { q: 'Three ranking loss families', a: 'Pointwise (grade), pairwise (inversions), listwise (metric-aware — LambdaMART).' },
      { q: 'Position bias correction', a: 'Inverse propensity weighting, randomisation/interleaving, or position-as-a-feature zeroed at inference.' },
      { q: 'Two-stage ranking', a: 'Cheap retriever over millions (recall@k), expensive reranker over the top 50–200 (NDCG/MRR). Retriever recall is a hard ceiling.' }
    ]
  });

  /* ------------------------------------------------------------------ 2.25 */
  ML.section({
    id: 'experimentation', track: 'classical', num: '2.25', level: 2,
    title: 'Online experimentation: power, peeking, CUPED',
    lede: 'Offline metrics are a proxy. The experiment is the measurement — and it is a measurement with a variance, a duration and a startling number of ways to fool yourself. This is the section that decides whether your model launch is real.',
    prereq: ['intervals'],
    related: ['causal', 'sampling', 'mlops'],
    html: `
${H.tldr([
      'Decide the sample size <b>before</b> you start: $n \\approx 2p(1-p)\\left(\\frac{z_{\\alpha/2}+z_\\beta}{\\delta}\\right)^2$ per arm. Halving the detectable effect costs four times the traffic.',
      'Checking the result repeatedly and stopping when it is significant inflates the false-positive rate from 5% to <b>20–35%</b>. Either fix the horizon or use an always-valid method.',
      'CUPED — regress out a pre-experiment covariate — typically cuts variance 30–50% for free. It is the single highest-return technique in the whole area.'
    ])}

<h2><span class="sn">2.25.1</span> Power, and the number you must fix in advance</h2>
${H.table(['Quantity', 'Symbol', 'Meaning', 'Usual'], [
      ['Significance', '$\\alpha$', 'false-positive rate you accept', '0.05'],
      ['Power', '$1-\\beta$', 'chance of detecting a real effect of the assumed size', '0.80'],
      ['Baseline rate', '$p$', 'the current conversion rate', 'measured'],
      ['<b>MDE</b>', '$\\delta$', '<b>minimum detectable effect</b> — the smallest lift worth finding', 'a business decision'],
      ['Duration', '—', 'must cover at least one full weekly cycle', '1–4 weeks']
    ])}
${H.key('The MDE is not a statistical quantity — it is the smallest improvement that would change what you do. Ask for it before you ask for traffic, because it determines the traffic.')}

${H.lab('power', 'Sample size, MDE and duration — the trade, live', 'Move any two and the third follows. The curve is the power function; the shaded region is where a real effect of your assumed size would go undetected.')}

<h2><span class="sn">2.25.2</span> Peeking: the mistake everyone makes</h2>
<p>A fixed-horizon test controls $\\alpha$ <i>at one predetermined moment</i>. Every additional look is another chance for the random walk of the test statistic to wander across the boundary. It never wanders back and un-crosses it, because you stopped.</p>
${H.lab('peek', 'Simulating the peeking problem, under a true null', 'Both variants are identical — there is no effect at all. The simulation runs thousands of A/A tests and counts how often each stopping rule declares a winner. The correct answer is 5%.')}
${H.table(['Fix', 'How it works', 'Cost'], [
      ['Fix the horizon', 'compute $n$, look once, at the end', 'no early stopping, ever — including for a disaster'],
      ['Bonferroni over looks', 'divide $\\alpha$ by the number of planned looks', 'crude and very conservative'],
      ['<b>Group sequential</b> (O’Brien–Fleming, Pocock)', 'spend $\\alpha$ across pre-planned interim analyses', 'the clinical-trials standard; needs the looks planned in advance'],
      ['<b>Always-valid / anytime-valid</b> (mSPRT, e-values, confidence sequences)', 'a bound valid at <i>every</i> moment simultaneously', '~10–25% more samples for an honest continuous dashboard — what Optimizely and Netflix ship'],
      ['Bayesian with a decision rule', 'stop when $P(\\text{lift}>0)$ crosses a threshold', 'well-defined, but a threshold is not a guaranteed error rate']
    ])}
${H.pitfall('"We ran it Bayesian, so peeking is fine" is half right. The posterior is valid at every moment — it is just conditioning on data. But a <i>stopping rule</i> based on the posterior still has a frequentist error rate, and if you have not computed it you do not know it. Bayesian methods change what the number means; they do not make optional stopping free.')}

<h2><span class="sn">2.25.3</span> CUPED and variance reduction</h2>
<p>You almost always have a pre-experiment measurement of the same metric $X$ for the same users. Use it:</p>
$$\\hat Y_{\\text{cuped}} = Y - \\theta(X - \\bar X), \\qquad \\theta = \\frac{\\mathrm{Cov}(X,Y)}{\\mathrm{Var}(X)}$$
${H.deriv('why this is unbiased and lower-variance', [
      ['$\\mathbb{E}[\\hat Y] = \\mathbb{E}[Y] - \\theta\\,\\mathbb{E}[X-\\bar X] = \\mathbb{E}[Y]$', 'The correction has expectation zero because $X$ is measured <b>before</b> the treatment and is therefore unaffected by it. Unbiasedness is free.'],
      ['$\\mathrm{Var}(\\hat Y) = \\mathrm{Var}(Y) + \\theta^2\\mathrm{Var}(X) - 2\\theta\\,\\mathrm{Cov}(X,Y)$', 'Standard variance of a difference.'],
      ['$\\theta^\\star = \\mathrm{Cov}(X,Y)/\\mathrm{Var}(X)$', 'Minimise over $\\theta$ — it is a quadratic, so set the derivative to zero. This is exactly the OLS slope of $Y$ on $X$.'],
      ['$\\mathrm{Var}(\\hat Y) = (1-\\rho^2)\\,\\mathrm{Var}(Y)$', 'Substitute back. <b>The variance falls by exactly the $R^2$ of the pre-period on the experiment period.</b> A correlation of 0.7 removes half the variance, which halves the required sample size.']
    ], 'The pre-period must be strictly before randomisation. Using any post-treatment variable as the covariate reintroduces bias — that is no longer CUPED, it is conditioning on a collider (§1.7).')}
${H.lab('cuped', 'CUPED on simulated user metrics', 'Real numbers: the same experiment analysed with and without the adjustment. The variance reduction is exactly $1-\\rho^2$, and the confidence interval shrinks by $\\sqrt{1-\\rho^2}$.')}

<h2><span class="sn">2.25.4</span> The failure modes to name</h2>
${H.table(['Problem', 'Symptom', 'Fix'], [
      ['<b>Sample ratio mismatch</b>', 'arms are 50.4/49.6 instead of 50/50', '<b>stop and debug</b> — a chi-square p < 0.001 on the split means the assignment or logging is broken and every other number is suspect'],
      ['Novelty / primacy effect', 'a big lift in week 1 that decays', 'run longer; analyse new vs returning users separately'],
      ['Network interference', 'treatment leaks to control (social, marketplaces)', 'cluster randomisation, switchback, or budget-split designs'],
      ['Multiple metrics', 'one of 20 guardrails is "significant"', 'pre-register one primary metric; Benjamini–Hochberg on the rest'],
      ['Simpson’s paradox', 'lift overall, loss in every segment', 'check the segment mix; a shifted traffic composition is usually the cause (§1.7)'],
      ['Underpowered launch', '"no significant difference, so ship it"', 'absence of evidence is not evidence of absence — report the interval, not the p-value']
    ])}
${H.flag('Interleaving deserves more attention than it gets for ranking changes: showing one user a merged list from both rankers and measuring which side’s results they click needs roughly an order of magnitude fewer sessions than an A/B test for the same statistical power, because it removes between-user variance entirely. It only works for ordering comparisons, which is exactly what §2.24 is about.')}

${H.probe([
      ['You look at the dashboard daily for two weeks and stop when p < 0.05. What is your real false-positive rate?', 'Around 20–35% depending on the number of looks, not 5%. Use a group-sequential boundary or an always-valid confidence sequence.'],
      ['Explain CUPED to a product manager in one sentence.', 'We subtract off the part of each user’s behaviour we could already predict from before the experiment started, which removes noise without touching the effect — so the test finishes sooner.'],
      ['Your A/B split is 50.4% / 49.6% with 2 million users. Do you care?', 'Yes, enormously. At that scale the imbalance is wildly significant, which means assignment or logging is broken. Every metric in the readout is untrustworthy until it is explained.'],
      ['Halving the MDE requires how much more traffic?', 'Four times as much — $n \\propto 1/\\delta^2$.'],
      ['When is an A/B test the wrong tool?', 'Strong network effects, very rare outcomes, long feedback delays, or a change so large that a holdback is unethical or commercially impossible. Switchback, synthetic control and difference-in-differences (§1.7) are the alternatives.']
    ])}`,
    labs: {
      power: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'baseline conversion rate', min: .01, max: .5, step: .005, value: .12, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'lift', label: 'relative lift to detect', min: .01, max: .3, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'power', label: 'power', min: .5, max: .95, step: .05, value: .8, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'daily', label: 'daily traffic per arm', min: 200, max: 50000, step: 200, value: 5000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'n', label: 'users needed per arm', cls: 'key' },
          { k: 'days', label: 'days to run', cls: 'warn' },
          { k: 'abs', label: 'absolute effect' },
          { k: 'mde', label: 'MDE at 1 week' },
          { k: 'half', label: 'to halve the MDE' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const p = st.p, delta = p * st.lift;
            const n = Num.sampleSize(p, delta, .05, st.power);
            const days = Math.ceil(n / st.daily);
            const week = Num.mde(p, st.daily * 7, .05, st.power);
            const P = Viz.plot(ctx, w, h, {
              xd: [2, 6], yd: [0, .12],
              pad: { l: 54, r: 14, t: 16, b: 40 }
            }).frame({
              xticks: [2, 3, 4, 5, 6], xfmt: v => Math.pow(10, v) >= 1e6 ? (Math.pow(10, v) / 1e6) + 'M' : Math.pow(10, v).toLocaleString(),
              xlabel: 'users per arm', ylabel: 'detectable absolute lift', yfmt: v => (v * 100).toFixed(1) + '%'
            });
            P.clip(() => {
              [.8, .9, .95].forEach((pw, i) => {
                P.fn(lx => Num.mde(p, Math.pow(10, lx), .05, pw), { color: [T.c1, T.c4, T.c2][i], width: pw === st.power ? 2.8 : 1.3, alpha: pw === st.power ? 1 : .5, n: 140 });
              });
              P.hline(delta, { color: T.c3, dash: [4, 4], label: 'your effect' });
              P.vline(Math.log10(n), { color: T.c3, dash: [4, 4] });
              P.dots([[Math.log10(n), delta]], { r: 5.5, color: T.c3, stroke: true, strokeWidth: 2 });
            });
            out({
              n: n.toLocaleString(),
              days: days + (days > 28 ? ' ⚠ too long' : ''),
              abs: (delta * 100).toFixed(2) + ' pts',
              mde: (week * 100).toFixed(2) + ' pts',
              half: (4 * n).toLocaleString() + ' per arm'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: '80% power' }, { c: 'var(--c4)', t: '90%' }, { c: 'var(--c2)', t: '95%' }]);
        Viz.note(host, 'Try to detect a 1% relative lift on a 12% baseline: the answer is roughly two million users per arm. That is the honest reason most teams cannot measure small improvements — and the reason CUPED, which cuts the requirement by $1-\\rho^2$, is worth more than almost any modelling change.');
      },

      peek: function (host) {
        const st = Viz.controls(host, [
          { k: 'looks', label: 'times you check the dashboard', min: 1, max: 30, step: 1, value: 14, fmt: v => v },
          { k: 'alpha', label: 'nominal α', min: .01, max: .1, step: .01, value: .05, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nominal', label: 'α you believe', cls: 'key' },
          { k: 'actual', label: 'α you actually have', cls: 'bad' },
          { k: 'infl', label: 'inflation' },
          { k: 'fix', label: 'Bonferroni-corrected α' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const pts = [];
            for (let k = 1; k <= 30; k++) pts.push([k, Num.peekingFPR(k, st.alpha, 91)]);
            const actual = pts[st.looks - 1][1];
            const P = Viz.plot(ctx, w, h, { xd: [1, 30], yd: [0, Math.max(.4, actual * 1.2)], pad: { l: 52, r: 14, t: 16, b: 40 } })
              .frame({ xlabel: 'number of looks at the data', ylabel: 'false-positive rate', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.area(pts, { color: T.c2, alpha: .14 });
              P.line(pts, { color: T.c2, width: 2.8 });
              P.hline(st.alpha, { color: T.c3, dash: [5, 4], label: 'what you think it is' });
              P.dots([[st.looks, actual]], { r: 5.5, color: T.c2, stroke: true, strokeWidth: 2 });
            });
            out({
              nominal: (st.alpha * 100).toFixed(0) + '%',
              actual: (actual * 100).toFixed(1) + '%',
              infl: (actual / st.alpha).toFixed(1) + '×',
              fix: (st.alpha / st.looks).toFixed(4)
            });
          }
        });
        Viz.note(host, 'Every point on this curve is two thousand simulated A/A tests — <b>there is no effect anywhere in this simulation</b>. Check daily for two weeks and roughly one in four of your "wins" is noise. The curve flattens rather than reaching 100% because the test statistic is a random walk with drift zero: it becomes progressively harder to cross a boundary it has already had many chances at.');
      },

      cuped: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'pre/post correlation ρ', min: 0, max: .95, step: .05, value: .7, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'users per arm', min: 200, max: 20000, step: 200, value: 3000, fmt: v => v.toLocaleString() },
          { k: 'effect', label: 'true treatment effect', min: 0, max: .3, step: .01, value: .08, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'raw', label: 'raw estimate ± CI' },
          { k: 'adj', label: 'CUPED estimate ± CI', cls: 'good' },
          { k: 'vr', label: 'variance reduction', cls: 'key' },
          { k: 'theory', label: 'theory: 1 − ρ²' },
          { k: 'equiv', label: 'equivalent extra users' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(15);
            const n = st.n, rho = st.rho;
            const pre = [], post = [], arm = [];
            for (let i = 0; i < 2 * n; i++) {
              const x = R.normal(0, 1);
              const a = i < n ? 0 : 1;
              const y = rho * x + Math.sqrt(Math.max(0, 1 - rho * rho)) * R.normal(0, 1) + a * st.effect;
              pre.push(x); post.push(y); arm.push(a);
            }
            const c = Num.cuped(post, pre);
            const meanBy = (v, a) => Num.mean(v.filter((_, i) => arm[i] === a));
            const sdBy = (v, a) => Num.sd(v.filter((_, i) => arm[i] === a));
            const dRaw = meanBy(post, 1) - meanBy(post, 0);
            const dAdj = meanBy(c.adjusted, 1) - meanBy(c.adjusted, 0);
            const seRaw = Math.sqrt(sdBy(post, 0) ** 2 / n + sdBy(post, 1) ** 2 / n);
            const seAdj = Math.sqrt(sdBy(c.adjusted, 0) ** 2 / n + sdBy(c.adjusted, 1) ** 2 / n);

            const P = Viz.plot(ctx, w, h, { xd: [-.6, 1.6], yd: [-.12, .32], pad: { l: 54, r: 14, t: 18, b: 40 } })
              .frame({ xticks: [0, 1], xfmt: v => v === 0 ? 'raw' : 'CUPED', xlabel: 'analysis', ylabel: 'estimated lift' });
            P.clip(() => {
              P.hline(st.effect, { color: T.c3, dash: [5, 4], label: 'truth' });
              P.hline(0, { color: T.faint, dash: false, width: 1 });
              P.errbars([[0, dRaw - 1.96 * seRaw, dRaw + 1.96 * seRaw]], { color: T.c2, width: 2.4, cap: 12 });
              P.dots([[0, dRaw]], { r: 6, color: T.c2, stroke: true, strokeWidth: 2 });
              P.errbars([[1, dAdj - 1.96 * seAdj, dAdj + 1.96 * seAdj]], { color: T.c1, width: 2.4, cap: 12 });
              P.dots([[1, dAdj]], { r: 6, color: T.c1, stroke: true, strokeWidth: 2 });
            });
            const vr = 1 - (c.varAfter / c.varBefore);
            out({
              raw: dRaw.toFixed(4) + ' ± ' + (1.96 * seRaw).toFixed(4),
              adj: dAdj.toFixed(4) + ' ± ' + (1.96 * seAdj).toFixed(4),
              vr: (vr * 100).toFixed(1) + '%',
              theory: ((rho * rho) * 100).toFixed(1) + '%',
              equiv: vr < .99 ? Math.round(n / (1 - vr) - n).toLocaleString() : '—'
            });
          }
        });
        Viz.note(host, 'Both estimates are centred on the truth — CUPED is unbiased, because the covariate was measured before randomisation. The <b>interval is what changes</b>: at ρ = 0.7 the variance falls by 49%, which is worth as much as doubling your traffic and costs one line of SQL. This is the same idea as a control variate in §1.13, applied to the metric you already log.');
      }
    },
    quiz: [
      {
        q: 'Halving the minimum detectable effect requires roughly…',
        options: ['2× the sample', '4× the sample', '8× the sample', 'the same sample, run longer'],
        answer: 1,
        why: '$n\\propto 1/\\delta^2$. This quadratic is why small lifts are so expensive to measure.'
      },
      {
        q: 'Checking a fixed-horizon test daily for two weeks and stopping at p < 0.05 gives a true false-positive rate of about…',
        options: ['5%', '10%', '25%', '50%'],
        answer: 2,
        why: 'Roughly 20–35% depending on the number of looks. Group-sequential boundaries or always-valid inference fix it.'
      },
      {
        q: 'CUPED reduces variance by a factor of…',
        options: ['ρ', '1 − ρ²', '1/ρ', '√ρ'],
        answer: 1,
        why: 'Exactly the residual variance after regressing on the pre-period covariate — the $R^2$ you remove is the variance you save.'
      },
      {
        q: 'Your 50/50 experiment came out 50.4/49.6 across 2 million users. You should…',
        options: ['proceed, it is close enough', 'stop and debug assignment or logging before trusting any metric', 'reweight the arms', 'extend the experiment'],
        answer: 1,
        why: 'A sample ratio mismatch that significant means the randomisation or the logging is broken, which invalidates everything else in the readout.'
      }
    ],
    cards: [
      { q: 'Sample size per arm', a: '$n \\approx 2p(1-p)\\left(\\frac{z_{\\alpha/2}+z_\\beta}{\\delta}\\right)^2$; halving δ costs 4× the traffic.' },
      { q: 'Peeking', a: 'Repeated looks inflate α from 5% to 20–35%. Fix the horizon, use group-sequential boundaries, or use always-valid confidence sequences.' },
      { q: 'CUPED', a: '$\\hat Y = Y - \\theta(X-\\bar X)$ with $\\theta$ the OLS slope; variance falls to $(1-\\rho^2)\\mathrm{Var}(Y)$. Unbiased because $X$ is pre-treatment.' },
      { q: 'Sample ratio mismatch', a: 'A significant deviation from the intended split means the pipeline is broken — debug before reading any metric.' },
      { q: 'When A/B fails', a: 'Network effects, rare outcomes, long delays. Use switchback, cluster randomisation, or synthetic control.' }
    ]
  });
})();
