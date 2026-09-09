/* ============================================================
   PART 1 — Mathematical & statistical foundations (1.1 – 1.5)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.1 */
  ML.section({
    id: 'bayes', track: 'foundations', num: '1.1',
    title: 'Probability, conditioning, Bayes',
    lede: 'The update rule underneath Naive Bayes, the probabilistic reading of logistic regression, calibration and MAP. Get the base rate wrong here and every downstream number is wrong by the same factor.',
    rests: 'Establishes: §2.5 Naive Bayes · §2.4 logistic regression · §2.12 calibration · §1.5 MAP.',
    html: `
<h2><span class="sn">1.1.1</span> Intuition first</h2>
<p>A conditional probability is not a new kind of quantity. It is the same probability measure <b>renormalised to the world in which the evidence is true</b>: delete every outcome incompatible with $B$, then rescale what remains so it sums to one. Every confusion about Bayes dissolves once you insist on drawing that shrunken world.</p>

<h2><span class="sn">1.1.2</span> Formally</h2>
$$P(A\\mid B) = \\frac{P(B\\mid A)\\,P(A)}{P(B)}, \\qquad P(B) = \\sum_i P(B\\mid A_i)P(A_i)$$
<p>The numerator is the joint; the denominator is that same joint marginalised over every way $B$ could have happened (the law of total probability). Read it left to right: <b>posterior ∝ likelihood × prior</b>.</p>

${H.worked('worked number — the low base rate', `
<p>Prevalence $P(D) = 0.001$. Sensitivity $P(+\\mid D) = 0.99$. Specificity $0.95$, so $P(+\\mid \\neg D) = 0.05$.</p>
$$P(D\\mid +) = \\frac{0.99 \\times 0.001}{0.99\\times0.001 + 0.05\\times0.999} = \\frac{0.00099}{0.05094} = 0.0194$$
<p>A positive result from a "99% accurate" test is correct about <mark>2% of the time</mark>. In 100,000 people: 100 sick, 99 caught — against 99,900 healthy, of whom 4,995 test positive anyway. The false positives outnumber the true ones fifty to one, and no amount of sensitivity fixes that; only a better prior or better specificity does.</p>`)}

${H.lab('bayes', 'The base rate, drawn as people', 'Each cell is one person out of 10,000. Move prevalence and watch the red band swamp the black dot — the posterior is the ratio of black to the whole flagged group. Compare pushing specificity 95% → 99.5% against pushing sensitivity 99% → 100%.')}

<h2><span class="sn">1.1.3</span> Depth signal: work in odds</h2>
<p>Divide the Bayes rule for $D$ by the same rule for $\\neg D$ and the denominator cancels:</p>
$$\\underbrace{\\frac{P(D\\mid+)}{P(\\neg D\\mid+)}}_{\\text{posterior odds}} = \\underbrace{\\frac{P(+\\mid D)}{P(+\\mid \\neg D)}}_{\\text{likelihood ratio}} \\times \\underbrace{\\frac{P(D)}{P(\\neg D)}}_{\\text{prior odds}}$$
<p>Here $\\mathrm{LR} = 0.99/0.05 = 19.8$. Prior odds $1{:}999$, times 19.8, gives $19.8{:}999 \\approx 1{:}50$ — the same 2%, computed in your head. This is the form to reach for at a whiteboard, and it is the same log-odds arithmetic that makes a credit scorecard additive (§2.4).</p>
${H.note('Taking logs turns the multiplication into addition: log-posterior-odds = log-likelihood-ratio + log-prior-odds. That is <i>why</i> a logistic regression is a sum of evidence terms.')}

<h2><span class="sn">1.1.4</span> Where this shows up later</h2>
${H.table(['Downstream', 'The Bayes content'], [
      ['Naive Bayes (§2.5)', 'Posterior ∝ prior × ∏ likelihoods, with conditional independence assumed'],
      ['Logistic regression (§2.4)', 'The model outputs log-posterior-odds directly; the sigmoid inverts the odds transform'],
      ['Calibration (§2.12)', 'A probability is only meaningful if the base rate it implies matches reality — resampling changes the prior and breaks this'],
      ['MAP (§1.5)', 'The prior over parameters is the regularizer'],
      ['Conformal prediction (§2.12)', 'The escape hatch when you refuse to model the likelihood at all']
    ])}

${H.probe([
      ['The test is 99% accurate — why is the posterior only 2%?', 'Because false positives drawn from the huge healthy majority swamp the handful of true positives.'],
      ['What raises it?', 'A higher prior — targeted rather than population screening — or better specificity. Sensitivity barely moves it.'],
      ['Give the odds form.', 'Posterior odds = likelihood ratio × prior odds; here 19.8 × 1/999 ≈ 1/50.']
    ], 'Quoting sensitivity as if it were $P(D\\mid+)$. It is the single most common probability error in interviews.')}`,
    labs: {
      bayes: function (host) {
        const st = Viz.controls(host, [
          { k: 'prev', label: 'prevalence P(D)', min: .0005, max: .35, step: .0005, value: .001, fmt: v => (v * 100).toFixed(2) + '%' },
          { k: 'sens', label: 'sensitivity P(+|D)', min: .5, max: 1, step: .005, value: .99, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'spec', label: 'specificity P(−|¬D)', min: .5, max: 1, step: .005, value: .95, fmt: v => (v * 100).toFixed(1) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'post', label: 'P(D | +) posterior', cls: 'key' },
          { k: 'lr', label: 'likelihood ratio' },
          { k: 'tp', label: 'true positives / 10k' },
          { k: 'fp', label: 'false positives / 10k' },
          { k: 'ppv', label: 'flagged / 10k' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const N = 10000, side = 100;
            const sick = st.prev * N, tp = sick * st.sens, fn = sick - tp;
            const well = N - sick, fp = well * (1 - st.spec), tn = well - fp;
            const post = tp / (tp + fp || 1);
            const gw = Math.min(h - 34, w * .5), cs = gw / side;
            const ox = 8, oy = 14;
            const nTP = Math.round(tp), nFP = Math.round(fp), nFN = Math.round(fn);
            for (let i = 0; i < N; i++) {
              const r = Math.floor(i / side), c = i % side;
              let col;
              if (i < nTP) col = T.dark ? '#f5f7ff' : '#12141c';
              else if (i < nTP + nFP) col = T.red;
              else if (i < nTP + nFP + nFN) col = T.amber;
              else col = T.dark ? 'rgba(255,255,255,.07)' : 'rgba(0,0,0,.06)';
              ctx.fillStyle = col;
              ctx.fillRect(ox + c * cs, oy + r * cs, Math.max(.8, cs - .35), Math.max(.8, cs - .35));
            }
            ctx.strokeStyle = T.line; ctx.strokeRect(ox, oy, gw, gw);
            const lx = ox + gw + 22;
            const rows = [
              [T.dark ? '#f5f7ff' : '#12141c', Math.round(tp) + ' true positives (sick, flagged)'],
              [T.red, Math.round(fp) + ' false positives (healthy, flagged)'],
              [T.amber, Math.round(fn) + ' missed cases (sick, cleared)'],
              [T.dark ? 'rgba(255,255,255,.12)' : 'rgba(0,0,0,.1)', Math.round(tn) + ' correct negatives']
            ];
            ctx.font = '12px ui-sans-serif, system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            rows.forEach((r, i) => {
              ctx.fillStyle = r[0]; ctx.fillRect(lx, oy + 6 + i * 23, 12, 12);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, oy + 6 + i * 23, 12, 12);
              ctx.fillStyle = T.text; ctx.fillText(r[1], lx + 20, oy + 12 + i * 23);
            });
            const by = oy + 118, bw = Math.max(80, w - lx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'bottom';
            ctx.fillText('among the flagged:', lx, by - 6);
            ctx.fillStyle = T.dark ? '#f5f7ff' : '#12141c'; ctx.fillRect(lx, by, bw * post, 24);
            ctx.fillStyle = T.red; ctx.fillRect(lx + bw * post, by, bw * (1 - post), 24);
            ctx.fillStyle = T.text; ctx.textBaseline = 'top'; ctx.font = 'bold 13px ui-monospace, monospace';
            ctx.fillText((post * 100).toFixed(1) + '% actually sick', lx, by + 30);
            ctx.font = '11px ui-sans-serif'; ctx.fillStyle = T.muted;
            ctx.fillText('posterior odds = LR × prior odds', lx, by + 52);
            ctx.fillText('= ' + (st.sens / (1 - st.spec)).toFixed(1) + ' × ' + (st.prev / (1 - st.prev)).toFixed(4) +
              ' = ' + (st.sens / (1 - st.spec) * st.prev / (1 - st.prev)).toFixed(3) + ' : 1', lx, by + 70);
            out({
              post: (post * 100).toFixed(2) + '%',
              lr: (st.sens / (1 - st.spec || 1e-9)).toFixed(1),
              tp: Math.round(tp), fp: Math.round(fp), ppv: Math.round(tp + fp)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Population screen (0.1%)', on: () => { st.$set('prev', .001); S.redraw(); } },
          { label: 'Targeted (10%)', on: () => { st.$set('prev', .10); S.redraw(); } },
          { label: 'Better specificity (99.5%)', on: () => { st.$set('spec', .995); S.redraw(); } },
          { label: 'Perfect sensitivity', on: () => { st.$set('sens', 1); S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Prevalence 1%, sensitivity 90%, specificity 90%. Roughly what is $P(D\\mid+)$?',
        options: ['about 90%', 'about 50%', 'about 8%', 'about 1%'],
        answer: 2,
        why: 'Per 10,000: 100 sick → 90 caught; 9,900 healthy → 990 false positives. 90/1080 ≈ 8.3%. The healthy majority dominates.'
      },
      {
        q: 'Which change raises the posterior most in a low-prevalence screen?',
        options: ['Raising sensitivity from 95% to 99%', 'Raising specificity from 95% to 99%', 'Doubling the sample size', 'Lowering the decision threshold'],
        answer: 1,
        why: 'False positives are (1 − specificity) × the huge healthy population. Cutting them five-fold beats catching a few extra true cases.'
      },
      {
        q: 'Posterior odds equal…',
        options: ['prior odds ÷ likelihood ratio', 'likelihood ratio × prior odds', 'likelihood ratio + prior odds', 'sensitivity × prevalence'],
        answer: 1,
        why: 'Dividing Bayes for D by Bayes for ¬D cancels the evidence term. In logs it becomes addition — which is why scorecards add points.'
      }
    ],
    cards: [
      { q: 'Bayes, in the form to say at a whiteboard', a: 'Posterior odds = likelihood ratio × prior odds. In logs: log-posterior-odds = log LR + log prior odds.' },
      { q: 'A "99% accurate" test on a 0.1% base rate gives what posterior?', a: '≈ 2%. False positives from the healthy majority outnumber true positives ~50:1.' },
      { q: 'What is $P(B)$ in the denominator of Bayes?', a: 'The joint marginalised over every way B could occur: $\\sum_i P(B\\mid A_i)P(A_i)$ — the law of total probability.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.2 */
  ML.section({
    id: 'distributions', track: 'foundations', num: '1.2',
    title: 'The distributions that recur',
    lede: 'Six distributions, and one idea about them: pick the distribution whose support and shape match your target, and its negative log-likelihood is your loss.',
    rests: 'Establishes: the noise models behind every loss in §1.5 and the link functions in §2.4.',
    html: `
${H.key('Pick the distribution whose support and shape match your target, and its negative log-likelihood is your loss.')}
<p>That sentence collapses §1.2, §1.5 and §2.4 into one move. You never memorise a loss again; you name a noise model.</p>

${H.table(['Distribution', 'PMF / PDF', 'Mean', 'Variance', 'The ML use'], [
      ['<b>Bernoulli</b>', '$p^y(1-p)^{1-y}$', '$p$', '$p(1-p)$', 'binary labels → cross-entropy'],
      ['<b>Categorical</b>', '$\\prod_k p_k^{y_k}$', '$p_k$', '$p_k(1-p_k)$', 'softmax targets, next-token loss'],
      ['<b>Gaussian</b>', '$\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-(x-\\mu)^2/2\\sigma^2}$', '$\\mu$', '$\\sigma^2$', 'CLT limit → squared error'],
      ['<b>Poisson</b>', '$e^{-\\lambda}\\lambda^k/k!$', '$\\lambda$', '$\\lambda$', 'counts → Poisson deviance, log link'],
      ['<b>Exponential / Gamma</b>', '$\\lambda e^{-\\lambda x}$', '$1/\\lambda$', '$1/\\lambda^2$', 'waiting times, survival, time-to-default'],
      ['<b>Beta / Dirichlet</b>', '$x^{\\alpha-1}(1-x)^{\\beta-1}/B(\\alpha,\\beta)$', '$\\alpha/(\\alpha+\\beta)$', 'small when $\\alpha+\\beta$ large', 'conjugate priors on rates; smoothing']
    ])}

<p>Two relationships worth having ready. <b>Binomial is a sum of Bernoullis</b> and tends to Gaussian by the CLT (§1.4). <b>Poisson is the Binomial limit</b> when $n \\to \\infty$, $np \\to \\lambda$ — which is why rare-event counts are Poisson, and why Poisson regression, not a linear model, is the default for claim or default counts.</p>

${H.lab('dist', 'The distribution gallery, live', 'Change the parameters and watch the shape, the moments, and the sampled histogram. The dashed red curve on discrete plots is the Gaussian the CLT promises — notice how fast Binomial gets there and how slowly a skewed Poisson does.')}

<h2><span class="sn">1.2.1</span> Choosing by support and shape</h2>
${H.table(['If your target is…', 'Support', 'Distribution', 'Resulting loss'], [
      ['yes / no', '{0,1}', 'Bernoulli', 'binary cross-entropy'],
      ['one of K classes', '{1..K}', 'Categorical', 'softmax cross-entropy'],
      ['a real number, symmetric error', 'ℝ', 'Gaussian', 'squared error'],
      ['a count', '{0,1,2,…}', 'Poisson (negative binomial if over-dispersed)', 'Poisson deviance'],
      ['a duration / time-to-event', '(0,∞)', 'Exponential / Weibull / Gamma', 'survival likelihood (§2.18)'],
      ['money with a mass at zero', '[0,∞)', 'Tweedie', 'Tweedie deviance'],
      ['a probability or rate', '[0,1]', 'Beta', 'beta likelihood / logit-normal']
    ])}
${H.flag('Over-dispersion is the standard trap with counts: Poisson forces variance = mean. Real claim counts are more variable than that, and a negative binomial (a Poisson whose rate is Gamma-distributed) is the honest fix.')}

${H.probe([
      ['Why is cross-entropy the loss for classification?', 'It is the negative log-likelihood of a Bernoulli or Categorical label — not a heuristic.'],
      ['When would you use Poisson regression rather than linear regression?', 'Non-negative counts; the log link keeps predictions positive and the variance scales with the mean.'],
      ['Why is Beta a natural prior on a rate?', 'Support is exactly [0,1] and it is conjugate to Bernoulli/Binomial, so the posterior is Beta again — which makes Thompson sampling one line (§1.6).']
    ])}`,
    labs: {
      dist: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'distribution', type: 'select', value: 'gauss', options: [
            { v: 'bern', t: 'Bernoulli' }, { v: 'binom', t: 'Binomial' }, { v: 'gauss', t: 'Gaussian' },
            { v: 'pois', t: 'Poisson' }, { v: 'expo', t: 'Exponential' }, { v: 'beta', t: 'Beta' }, { v: 'gamma', t: 'Gamma' }] },
          { k: 'a', label: 'parameter 1', min: .05, max: 10, step: .05, value: 5, fmt: v => v.toFixed(2) },
          { k: 'b', label: 'parameter 2', min: .05, max: 10, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'samples drawn', min: 0, max: 4000, step: 50, value: 800, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'p', label: 'parameters', cls: 'key' }, { k: 'm', label: 'mean' }, { k: 'v', label: 'variance' },
          { k: 'sm', label: 'sample mean' }, { k: 'sv', label: 'sample var' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31);
            const d = st.d;
            let pts = [], samples = [], xd, discrete = true, mean = 0, varr = 0, plabel = '';
            if (d === 'bern') {
              const p = Math.min(.99, st.a / 10);
              xd = [-0.6, 1.6]; pts = [[0, 1 - p], [1, p]]; mean = p; varr = p * (1 - p);
              plabel = 'p = ' + p.toFixed(2);
              for (let i = 0; i < st.n; i++) samples.push(R() < p ? 1 : 0);
            } else if (d === 'binom') {
              const n = Math.max(1, Math.round(st.a * 3)), p = Math.min(.99, st.b / 10);
              xd = [-.5, n + .5]; mean = n * p; varr = n * p * (1 - p);
              plabel = 'n = ' + n + ', p = ' + p.toFixed(2);
              for (let k = 0; k <= n; k++) pts.push([k, Num.binomPmf(k, n, p)]);
              for (let i = 0; i < st.n; i++) { let s = 0; for (let j = 0; j < n; j++) s += R() < p ? 1 : 0; samples.push(s); }
            } else if (d === 'pois') {
              const lam = st.a; xd = [-.5, Math.max(10, lam * 3)]; mean = lam; varr = lam;
              plabel = 'λ = ' + lam.toFixed(2);
              for (let k = 0; k <= xd[1]; k++) pts.push([k, Num.poisPmf(k, lam)]);
              for (let i = 0; i < st.n; i++) samples.push(R.poisson(lam));
            } else if (d === 'gauss') {
              discrete = false;
              const mu = st.a - 5, sdv = Math.max(.15, st.b / 2);
              xd = [mu - 4 * sdv, mu + 4 * sdv]; mean = mu; varr = sdv * sdv;
              plabel = 'μ = ' + mu.toFixed(2) + ', σ = ' + sdv.toFixed(2);
              for (let i = 0; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, Num.normPdf(x, mu, sdv)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.normal(mu, sdv));
            } else if (d === 'expo') {
              discrete = false;
              const lam = Math.max(.1, st.a); xd = [0, 6 / lam]; mean = 1 / lam; varr = 1 / (lam * lam);
              plabel = 'λ = ' + lam.toFixed(2);
              for (let i = 0; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, lam * Math.exp(-lam * x)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.exp(lam));
            } else if (d === 'beta') {
              discrete = false;
              const a = Math.max(.1, st.a), b = Math.max(.1, st.b);
              xd = [0, 1]; mean = a / (a + b); varr = a * b / ((a + b) * (a + b) * (a + b + 1));
              plabel = 'α = ' + a.toFixed(2) + ', β = ' + b.toFixed(2);
              for (let i = 1; i < 220; i++) { const x = i / 220; pts.push([x, Num.betaPdf(x, a, b)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.beta(a, b));
            } else {
              discrete = false;
              const k = Math.max(.15, st.a), th = Math.max(.1, st.b / 2);
              xd = [0, Math.max(4, (k * th) * 3)]; mean = k * th; varr = k * th * th;
              plabel = 'k = ' + k.toFixed(2) + ', θ = ' + th.toFixed(2);
              for (let i = 1; i <= 220; i++) { const x = xd[0] + (xd[1] - xd[0]) * i / 220; pts.push([x, Num.gammaPdf(x, k, th)]); }
              for (let i = 0; i < st.n; i++) samples.push(R.gamma(k) * th);
            }
            const maxp = Math.max.apply(null, pts.map(p => p[1])) || 1;
            const P = Viz.plot(ctx, w, h, { xd: xd, yd: [0, maxp * 1.25] })
              .frame({ xlabel: discrete ? 'k' : 'x', ylabel: discrete ? 'probability' : 'density' });
            if (samples.length) {
              const nb = discrete ? Math.round(xd[1] - xd[0]) : 34;
              const hh = Num.hist(samples, Math.max(2, nb), xd[0], xd[1]);
              const scale = discrete ? 1 / samples.length : 1 / (samples.length * hh.w);
              P.clip(() => {
                hh.bins.forEach((c, i) => {
                  const x0 = hh.lo + i * hh.w, den = c * scale;
                  ctx.fillStyle = T.blue; ctx.globalAlpha = .18;
                  ctx.fillRect(P.x(x0), P.y(den), Math.max(1, P.x(x0 + hh.w) - P.x(x0) - 1), P.y(0) - P.y(den));
                  ctx.globalAlpha = 1;
                });
              });
            }
            if (discrete) {
              P.clip(() => pts.forEach(p => {
                P.line([[p[0], 0], [p[0], p[1]]], { color: T.blue, width: 4 });
                P.dots([[p[0], p[1]]], { r: 3.4, color: T.blue });
              }));
              P.clip(() => P.fn(x => Num.normPdf(x, mean, Math.sqrt(varr)), { color: T.red, width: 1.4, dash: [5, 4] }));
            } else {
              P.clip(() => { P.area(pts, { color: T.blue, alpha: .16 }); P.line(pts, { color: T.blue, width: 2.4 }); });
            }
            P.vline(mean, { color: T.amber, label: 'mean' });
            out({
              p: plabel, m: mean.toFixed(3), v: varr.toFixed(3),
              sm: samples.length ? Num.mean(samples).toFixed(3) : '—',
              sv: samples.length ? Num.variance(samples).toFixed(3) : '—'
            });
          }
        });
        Viz.note(host, 'The blue histogram is actual draws, generated here. Sample moments converge to the theoretical ones at the $1/\\sqrt n$ rate §1.4 quantifies.');
      }
    },
    quiz: [
      {
        q: 'Claim counts per policy-year have variance clearly larger than their mean. The right first model is…',
        options: ['linear regression on the raw count', 'Poisson regression', 'negative binomial regression', 'logistic regression'],
        answer: 2,
        why: 'Poisson forces variance = mean. Over-dispersion is exactly what the negative binomial (Poisson with a Gamma-distributed rate) exists for.'
      },
      {
        q: 'Why does the Beta distribution keep appearing as a prior on a probability?',
        options: ['It is maximum-entropy on ℝ', 'Its support is [0,1] and it is conjugate to Bernoulli/Binomial', 'It has infinite variance', 'It is the CLT limit'],
        answer: 1,
        why: 'Support matches the quantity and conjugacy makes the posterior Beta again with updated counts — one line of code, hence Thompson sampling.'
      }
    ],
    cards: [
      { q: 'The distribution-selection rule', a: 'Match support and shape to your target; the negative log-likelihood of that distribution is your loss.' },
      { q: 'Poisson as a limit', a: 'Binomial with $n\\to\\infty$, $np\\to\\lambda$ — rare events in many trials.' },
      { q: 'Mean and variance of a Binomial', a: '$np$ and $np(1-p)$.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.3 */
  ML.section({
    id: 'expectation', track: 'foundations', num: '1.3',
    title: 'Expectation, variance, covariance',
    lede: 'One asymmetry to memorise: linearity of expectation always holds; variances add only under independence. That single cross term is the entire theory of bagging.',
    rests: 'Feeds: §2.2 bias–variance · §2.10 PCA · every ensemble argument in §2.7.',
    html: `
<h2><span class="sn">1.3.1</span> Definitions</h2>
$$\\mathbb{E}[X]=\\sum_x x\\,p(x), \\quad \\mathrm{Var}(X)=\\mathbb{E}[X^2]-\\mathbb{E}[X]^2, \\quad \\mathrm{Cov}(X,Y)=\\mathbb{E}[XY]-\\mathbb{E}[X]\\mathbb{E}[Y]$$
<p>The covariance matrix $\\Sigma$ is symmetric and positive semi-definite (§1.8) — and that PSD fact is what makes PCA an eigenproblem and kernel methods convex.</p>

<h2><span class="sn">1.3.2</span> The asymmetry to memorise</h2>
${H.key('Linearity of expectation always holds; variances add only under independence.')}
$$\\mathrm{Var}(X+Y)=\\mathrm{Var}(X)+\\mathrm{Var}(Y)+2\\,\\mathrm{Cov}(X,Y)$$
<p>That cross term is the entire theory of bagging. Average $B$ identically-distributed models each with variance $\\sigma^2$ and pairwise correlation $\\rho$:</p>
$$\\mathrm{Var}\\!\\left(\\frac{1}{B}\\sum_b f_b\\right) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$
<p>Adding trees only kills the second term. The first is a floor set by how correlated the models are — and <i>decorrelating</i> them (random feature subsets, §2.7) is what attacks it. If you can write that formula, random forests need no further defence.</p>

${H.lab('bag', 'The bagging variance formula, as a picture', 'Move ρ and watch the floor. The gap between curve and floor is what more trees buy; random forests lower the floor itself by lowering ρ.')}

<h2><span class="sn">1.3.3</span> Covariance is a shape</h2>
${H.lab('cov', 'Covariance as an ellipse', 'Two features, one cloud, the one- and two-sigma ellipses. The tilt is the covariance; the arrows are eigenvectors of the sample covariance scaled by $\\sqrt{\\lambda_i}$ — which is all PCA (§2.10) is.')}

<h2><span class="sn">1.3.4</span> Two facts that save you later</h2>
<ul>
<li><b>Correlation is not dependence.</b> $Y = X^2$ with symmetric $X$ has zero correlation and total dependence. Mutual information (§1.10) catches what Pearson misses.</li>
<li><b>Variance of a mean.</b> $\\mathrm{Var}(\\bar X_n) = \\sigma^2/n$ for independent draws, so the standard error falls like $1/\\sqrt n$ — the reason §1.4's bounds carry $\\sqrt n$ and precision is bought in quadratic instalments.</li>
</ul>

${H.probe([
      ['Write the bagging variance formula.', '$\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ — averaging removes only the second term, so decorrelation is the lever.'],
      ['Does zero correlation imply independence?', 'No — only for jointly Gaussian variables. $Y=X^2$ is the counterexample to give.']
    ])}`,
    labs: {
      bag: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'pairwise correlation ρ', min: 0, max: .95, step: .01, value: .4, fmt: v => v.toFixed(2) },
          { k: 'sig', label: 'single-model variance σ²', min: .2, max: 3, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'floor', label: 'floor ρσ²', cls: 'key' }, { k: 'b10', label: 'variance @ B=10' },
          { k: 'b100', label: 'variance @ B=100' }, { k: 'red', label: 'reduction @ B=100' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const f = B => st.rho * st.sig + (1 - st.rho) / B * st.sig;
            const P = Viz.plot(ctx, w, h, { xd: [1, 100], yd: [0, st.sig * 1.05] })
              .frame({ xlabel: 'number of models B', ylabel: 'variance of the average' });
            P.clip(() => {
              P.fn(f, { color: T.blue, width: 2.6, n: 300 });
              P.hline(st.rho * st.sig, { color: T.red, label: 'floor ρσ² — only decorrelation goes below' });
              [1, 5, 10, 25, 50, 100].forEach(B => P.dots([[B, f(B)]], { r: 4, color: T.blue, stroke: true }));
            });
            out({
              floor: (st.rho * st.sig).toFixed(3), b10: f(10).toFixed(3), b100: f(100).toFixed(3),
              red: (100 * (1 - f(100) / st.sig)).toFixed(0) + '%'
            });
          }
        });
      },
      cov: function (host) {
        const st = Viz.controls(host, [
          { k: 'rho', label: 'correlation between features', min: -.95, max: .95, step: .01, value: 0, fmt: v => v.toFixed(2) },
          { k: 'sx', label: 'σ of feature 1', min: .3, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'sy', label: 'σ of feature 2', min: .3, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'points', min: 40, max: 1200, step: 20, value: 400, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cov', label: 'sample covariance', cls: 'key' }, { k: 'cor', label: 'correlation' },
          { k: 'l1', label: 'λ₁' }, { k: 'l2', label: 'λ₂' }, { k: 'ev', label: 'PC1 explains' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(17);
            const pts = [];
            for (let i = 0; i < st.n; i++) {
              const z1 = R.normal(0, 1), z2 = R.normal(0, 1);
              pts.push([st.sx * z1, st.sy * (st.rho * z1 + Math.sqrt(1 - st.rho * st.rho) * z2)]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3.2, 3.2] }).frame({ xlabel: 'feature 1', ylabel: 'feature 2' });
            P.clip(() => P.dots(pts, { r: 2.6, color: T.blue, alpha: .5 }));
            const pc = Num.pca(pts);
            [1, 2].forEach(k => {
              const e = [];
              for (let t = 0; t <= 6.3; t += .05) {
                const a = Math.sqrt(Math.max(0, pc.values[0])) * k * Math.cos(t);
                const b = Math.sqrt(Math.max(0, pc.values[1])) * k * Math.sin(t);
                e.push([pc.mean[0] + a * pc.vectors[0][0] + b * pc.vectors[1][0],
                        pc.mean[1] + a * pc.vectors[0][1] + b * pc.vectors[1][1]]);
              }
              P.clip(() => P.line(e, { color: k === 1 ? T.red : T.faint, width: k === 1 ? 2 : 1, alpha: k === 1 ? 1 : .7 }));
            });
            pc.vectors.forEach((v, i) => {
              const s = Math.sqrt(Math.max(0, pc.values[i]));
              P.arrow(pc.mean[0], pc.mean[1], pc.mean[0] + v[0] * s, pc.mean[1] + v[1] * s, { color: i ? T.amber : T.green, width: 2.4 });
            });
            const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
            out({
              cov: Num.cov(xs, ys).toFixed(3), cor: Num.corr(xs, ys).toFixed(3),
              l1: pc.values[0].toFixed(3), l2: pc.values[1].toFixed(3),
              ev: (pc.explained[0] * 100).toFixed(0) + '%'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: '1σ ellipse' }, { c: Viz.theme().green, t: 'eigenvector 1 (PC1)' }, { c: Viz.theme().amber, t: 'eigenvector 2' }]);
      }
    },
    quiz: [
      {
        q: 'You double a bagged ensemble from 50 to 100 trees and variance barely moves. Why?',
        options: ['the trees are too shallow', 'ρ is high, so you are already at the floor ρσ²', 'the learning rate is too high', 'you need more features'],
        answer: 1,
        why: 'Var = ρσ² + (1−ρ)σ²/B. Once the second term is small, more trees do nothing; decorrelating them is the only remaining lever.'
      },
      {
        q: '$\\mathrm{Var}(X+Y)=\\mathrm{Var}(X)+\\mathrm{Var}(Y)$ when…',
        options: ['always', 'X and Y are uncorrelated', 'X and Y are identically distributed', 'X and Y are positive'],
        answer: 1,
        why: 'The cross term 2Cov(X,Y) vanishes under zero covariance (implied by independence, but weaker than it).'
      }
    ],
    cards: [
      { q: 'Bagging variance formula', a: '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$ — more models remove only the second term.' },
      { q: 'Linearity vs additivity', a: 'Expectation is linear unconditionally; variance adds only when covariance is zero.' },
      { q: 'Ellipse axes of a 2-D cloud', a: 'Eigenvectors of the covariance matrix with lengths $\\sqrt{\\lambda_i}$ — i.e. PCA.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.4 */
  ML.section({
    id: 'concentration', track: 'foundations', num: '1.4',
    title: 'LLN, CLT, concentration — why generalization is possible at all',
    lede: 'The theoretical licence for the entire train/test protocol. A limit is not a guarantee about your 4,000-row test set; concentration is.',
    rests: 'Licenses: §2.1 the split · §2.14 trusting a validation number · §2.13 every reported metric.',
    html: `
<h2><span class="sn">1.4.1</span> Two limits and one guarantee</h2>
<p>The <b>law of large numbers</b> says sample means converge to $\\mu$. The <b>central limit theorem</b> says the error is asymptotically Gaussian: $\\sqrt{n}(\\bar X_n - \\mu) \\to \\mathcal{N}(0, \\sigma^2)$. Both are limits, and a limit is not a promise about the finite test set in front of you. The idea that <i>is</i> a promise is concentration.</p>

<h3>Hoeffding</h3>
<p>For independent variables bounded in $[a,b]$,</p>
$$P(|\\bar X_n - \\mu| \\ge t) \\le 2\\exp\\!\\left(\\frac{-2nt^2}{(b-a)^2}\\right)$$
<p>The probability that your measured accuracy is off by more than $t$ decays <b>exponentially</b> in the number of test rows. Empirical risk is therefore close to true risk with high probability, and the gap shrinks like $1/\\sqrt n$. That is the whole licence: <mark>averaging kills independent noise, and it does so at a rate you can quote.</mark></p>

${H.worked('worked number — how many test rows?', `
<p>0/1 loss, so $b-a=1$. Want 95% confidence that the estimate is within $t = 0.02$:</p>
$$2e^{-2nt^2} = 0.05 \\;\\Rightarrow\\; n = \\frac{\\ln 40}{2(0.0004)} \\approx 4{,}612 \\text{ rows}$$
<p>Want half the width, $t = 0.01$: $n \\approx 18{,}400$ — four times as many. Error falls like $1/\\sqrt n$, so <b>precision is bought in quadratic instalments</b>.</p>`)}

${H.lab('conc', 'The concentration ladder, and the sample-size calculator', 'Each rung assumes more and pays in tightness. Move n and t; the calculator answers the question you are actually asked — how many rows do I need?')}

<h2><span class="sn">1.4.2</span> The ladder</h2>
${H.table(['Inequality', 'Assumes', 'Bound'], [
      ['<b>Markov</b>', 'non-negativity only', '$P(X\\ge a)\\le \\mathbb{E}[X]/a$'],
      ['<b>Chebyshev</b>', '+ finite variance', '$P(|X-\\mu|\\ge t)\\le \\sigma^2/t^2$'],
      ['<b>Hoeffding</b>', '+ bounded range', '$2\\exp(-2nt^2/(b-a)^2)$'],
      ['<b>Bernstein</b>', '+ variance and bound together', '$2\\exp\\!\\left(\\frac{-nt^2}{2\\sigma^2+2Mt/3}\\right)$']
    ])}
${H.note('Hoeffding is the rung where the bound becomes useful for a test set: it needs only boundedness, which 0/1 loss gives you for free.')}

<h2><span class="sn">1.4.3</span> See the limits happen</h2>
${H.lab('clt', 'LLN and CLT, simulated', 'Pick a distribution — including a badly skewed one — and watch the distribution of sample means become Gaussian. The heavy-tailed option shows why "n ≥ 30" is folklore, not a theorem.')}

${H.probe([
      ['Why is a bigger test set more trustworthy?', 'Estimation error falls like $1/\\sqrt n$, and Hoeffding turns that into an explicit exponential tail bound.'],
      ['How many rows do you need?', 'Invert Hoeffding — roughly 4,600 for ±2 points at 95%, 18,000 for ±1.'],
      ['CLT or concentration for a finite sample?', 'Concentration. The CLT is asymptotic; Hoeffding holds at every $n$.']
    ], 'Reporting a metric from a 50-row holdout as if it were a fact. Its confidence interval is roughly ±14 points.')}`,
    labs: {
      conc: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'test rows n', min: 50, max: 20000, step: 50, value: 1000, fmt: v => v.toLocaleString() },
          { k: 't', label: 'tolerance t', min: .005, max: .2, step: .005, value: .02, fmt: v => '±' + (v * 100).toFixed(1) + ' pts' },
          { k: 'sig', label: 'σ² (Chebyshev / Bernstein)', min: .01, max: .25, step: .005, value: .25, fmt: v => v.toFixed(3) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'hoef', label: 'Hoeffding bound', cls: 'key' }, { k: 'cheb', label: 'Chebyshev bound' },
          { k: 'bern', label: 'Bernstein bound' }, { k: 'need', label: 'n for 95% @ this t' }, { k: 'ci', label: 'CLT 95% half-width' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const hoef = n => Math.min(1, 2 * Math.exp(-2 * n * st.t * st.t));
            const cheb = n => Math.min(1, st.sig / (n * st.t * st.t));
            const bern = n => Math.min(1, 2 * Math.exp(-n * st.t * st.t / (2 * st.sig + 2 * st.t / 3)));
            const P = Viz.plot(ctx, w, h, { xd: [50, 20000], yd: [0, 1] })
              .frame({ xlabel: 'test rows n', ylabel: 'P(estimate off by more than t)' });
            P.clip(() => {
              P.fn(cheb, { color: T.amber, width: 2, n: 300 });
              P.fn(bern, { color: T.green, width: 2, n: 300 });
              P.fn(hoef, { color: T.blue, width: 2.6, n: 300 });
              P.hline(.05, { color: T.red, label: '5% risk' });
              P.vline(st.n, { color: T.text, dash: [3, 3] });
              P.dots([[st.n, hoef(st.n)]], { r: 5, color: T.blue, stroke: true });
            });
            out({
              hoef: (hoef(st.n) * 100).toFixed(1) + '%', cheb: (cheb(st.n) * 100).toFixed(1) + '%',
              bern: (bern(st.n) * 100).toFixed(1) + '%',
              need: Math.ceil(Math.log(40) / (2 * st.t * st.t)).toLocaleString(),
              ci: '±' + (1.96 * Math.sqrt(.25 / st.n) * 100).toFixed(2) + ' pts'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'Hoeffding' }, { c: Viz.theme().green, t: 'Bernstein' }, { c: Viz.theme().amber, t: 'Chebyshev' }, { c: Viz.theme().red, t: '5% risk line' }]);
        Viz.note(host, 'Hoeffding is conservative next to the CLT interval — that is the price of a guarantee holding at every finite n rather than in the limit.');
      },

      clt: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'underlying distribution', type: 'select', value: 'bern', options: [
            { v: 'bern', t: 'Bernoulli(0.3) — accuracy' }, { v: 'expo', t: 'Exponential — skewed' },
            { v: 'unif', t: 'Uniform' }, { v: 'pareto', t: 'Heavy-tailed (Pareto)' }] },
          { k: 'n', label: 'sample size n per experiment', min: 1, max: 200, step: 1, value: 30, fmt: v => v },
          { k: 'reps', label: 'experiments repeated', min: 100, max: 8000, step: 100, value: 2000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mu', label: 'true mean' }, { k: 'sm', label: 'mean of means', cls: 'key' },
          { k: 'se', label: 'observed SE' }, { k: 'pred', label: 'σ/√n predicted' }, { k: 'skew', label: 'skew of means' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            let draw, mu, sig;
            if (st.d === 'bern') { draw = () => R() < .3 ? 1 : 0; mu = .3; sig = Math.sqrt(.21); }
            else if (st.d === 'expo') { draw = () => R.exp(1); mu = 1; sig = 1; }
            else if (st.d === 'unif') { draw = () => R(); mu = .5; sig = Math.sqrt(1 / 12); }
            else { draw = () => Math.pow(1 - R(), -1 / 2.5); mu = 2.5 / 1.5; sig = 2.4; }
            const means = [];
            for (let r = 0; r < st.reps; r++) { let s = 0; for (let i = 0; i < st.n; i++) s += draw(); means.push(s / st.n); }
            const lo = Math.max(Math.min.apply(null, means), mu - 5 * sig / Math.sqrt(st.n));
            const hi = Math.min(Math.max.apply(null, means), mu + 5 * sig / Math.sqrt(st.n));
            const hh = Num.hist(means, 40, lo, hi);
            const dens = hh.bins.map(c => c / (means.length * hh.w));
            const P = Viz.plot(ctx, w, h, { xd: [lo, hi], yd: [0, Math.max.apply(null, dens) * 1.2 || 1] })
              .frame({ xlabel: 'sample mean of n draws', ylabel: 'density' });
            P.clip(() => {
              hh.centers.forEach((c, i) => {
                ctx.fillStyle = T.blue; ctx.globalAlpha = .3;
                const x0 = P.x(c - hh.w / 2), x1 = P.x(c + hh.w / 2);
                ctx.fillRect(x0, P.y(dens[i]), Math.max(1, x1 - x0 - 1), P.y(0) - P.y(dens[i]));
                ctx.globalAlpha = 1;
              });
              P.fn(x => Num.normPdf(x, mu, sig / Math.sqrt(st.n)), { color: T.red, width: 2.2 });
              P.vline(mu, { color: T.amber, label: 'true mean' });
            });
            const m2 = Num.mean(means), s2 = Num.sd(means);
            out({
              mu: mu.toFixed(3), sm: m2.toFixed(4), se: s2.toFixed(4),
              pred: (sig / Math.sqrt(st.n)).toFixed(4),
              skew: Num.mean(means.map(v => Math.pow((v - m2) / (s2 || 1), 3))).toFixed(2)
            });
          }
        });
        Viz.note(host, 'Red curve = the Gaussian the CLT predicts. On the heavy-tailed option the match stays poor even at n = 200 — the CLT needs finite variance.');
      }
    },
    quiz: [
      {
        q: 'You want validation accuracy to ±1 point at 95% confidence with 0/1 loss. Roughly how many rows?',
        options: ['about 500', 'about 4,600', 'about 18,000', 'about 100,000'],
        answer: 2,
        why: 'n = ln(40)/(2t²) with t = 0.01 gives ≈ 18,400. Halving the tolerance quadruples the requirement.'
      },
      {
        q: 'Which statement about the CLT is correct?',
        options: ['It guarantees the error on your finite test set', 'It is asymptotic; a finite-sample guarantee needs a concentration inequality', 'It applies to any distribution including infinite-variance ones', 'It requires the data itself to be Gaussian'],
        answer: 1,
        why: 'The CLT is a limit and needs finite variance. Hoeffding/Bernstein hold at every n, which is what licenses a test-set number.'
      }
    ],
    cards: [
      { q: 'Hoeffding’s inequality', a: '$P(|\\bar X_n-\\mu|\\ge t)\\le 2\\exp(-2nt^2/(b-a)^2)$.' },
      { q: 'Rows needed for ±2 points at 95%', a: '≈ 4,600; for ±1 point ≈ 18,000 — error falls like $1/\\sqrt n$.' },
      { q: 'Why does averaging work?', a: 'Independent noise cancels at rate $1/\\sqrt n$, and concentration makes that a quotable high-probability bound.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.5 */
  ML.section({
    id: 'mle-map', track: 'foundations', num: '1.5',
    title: 'MLE, MAP, and the bridge to loss and regularization',
    lede: 'The single most reused idea on this site: it turns every model in Parts 2 and 4 into an optimization problem you can write down from scratch.',
    rests: 'Consumed by: every loss in Part 2 · the softmax objective in §4.9 · the KL penalties in §4.12.',
    html: `
<h2><span class="sn">1.5.1</span> Maximum likelihood</h2>
<p>Maximum likelihood maximises $\\prod_i p(y_i \\mid x_i; \\theta)$. Products of small numbers underflow and do not differentiate pleasantly, so take a logarithm and flip the sign: <mark>the negative log-likelihood <i>is</i> your loss.</mark> Everything else is bookkeeping about which noise model you assumed.</p>

${H.table(['Assume', 'NLL becomes', 'Known as'], [
      ['Gaussian noise', '$\\frac{1}{2\\sigma^2}\\sum_i (y_i-\\hat y_i)^2 + c$', 'squared error / MSE'],
      ['Bernoulli label', '$-\\sum_i [y_i\\log p_i + (1-y_i)\\log(1-p_i)]$', 'binary cross-entropy / logloss'],
      ['Categorical label', '$-\\sum_i \\log p_{i,y_i}$', 'softmax cross-entropy (§4.9)'],
      ['Poisson counts', '$\\sum_i (\\hat\\lambda_i - y_i\\log\\hat\\lambda_i)$', 'Poisson deviance'],
      ['Laplace noise', '$\\sum_i |y_i-\\hat y_i|$', 'MAE — why MAE is robust to outliers']
    ])}
<p>Note the Gaussian case: $\\sigma^2$ enters only as a multiplicative scale, which is why nobody estimates it before fitting.</p>

<h2><span class="sn">1.5.2</span> MAP adds a prior</h2>
$$\\arg\\max_\\theta\\; \\log p(\\text{data}\\mid\\theta) + \\log p(\\theta)$$
<p>A zero-mean Gaussian prior $\\mathcal{N}(0,\\tau^2)$ contributes $-\\frac{1}{2\\tau^2}\\|\\theta\\|_2^2$ — an <b>L2 penalty with $\\lambda = 1/(2\\tau^2)$</b>. A Laplace prior contributes $-\\frac{1}{b}\\|\\theta\\|_1$ — <b>L1</b>. So:</p>
${H.key('Regularization is a prior, and the regularization strength is the prior’s inverse variance.')}
<p>Let $\\tau^2 \\to \\infty$ (a flat prior) and MAP degenerates to MLE — the cleanest one-line answer to "when are MAP and MLE the same?"</p>

${H.lab('assum', 'Assumption → loss → penalty', 'Pick a noise model and a prior; the objective assembles itself. Compare squared error against absolute error at a large residual — that ratio is exactly why one is robust and the other is not.')}

${H.lab('shrink', 'A prior is a spring: watch coefficients shrink', 'Real ridge and lasso fits on correlated data, recomputed as you move λ. Ridge shrinks smoothly and never quite arrives at zero; lasso pins coefficients to exactly zero one at a time.')}

${H.probe([
      ['Where does cross-entropy come from?', 'It is the negative log-likelihood of a Bernoulli (or Categorical) label — not a heuristic.'],
      ['Why does L2 shrink weights?', 'It is a zero-mean Gaussian prior; the log-prior is a quadratic pull toward zero.'],
      ['When is MAP the same as MLE?', 'As the prior variance → ∞, i.e. a flat prior.']
    ])}`,
    labs: {
      assum: function (host) {
        const objBox = ML.el('div', { class: 'box', style: 'margin:14px 0 0' });
        const st = Viz.controls(host, [
          { k: 'noise', label: 'noise model (likelihood)', type: 'select', value: 'gauss', options: [
            { v: 'gauss', t: 'Gaussian → squared error' }, { v: 'lap', t: 'Laplace → absolute error' },
            { v: 'huber', t: 'Huber (robust hybrid)' }, { v: 'pois', t: 'Poisson → deviance' }] },
          { k: 'prior', label: 'prior on weights', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'flat → MLE' }, { v: 'gauss', t: 'Gaussian → L2' }, { v: 'lap', t: 'Laplace → L1' }] },
          { k: 'tau', label: 'prior sd τ (smaller = stronger)', min: .1, max: 4, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => { S.redraw(); obj(); });
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const L = {
              gauss: r => .5 * r * r,
              lap: r => Math.abs(r),
              huber: r => Math.abs(r) < 1 ? .5 * r * r : Math.abs(r) - .5,
              pois: r => Math.exp(r) - r - 1
            };
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [0, 5] })
              .frame({ xlabel: 'residual  (ŷ − y)', ylabel: 'loss contributed' });
            P.clip(() => {
              P.fn(L.gauss, { color: T.faint, width: 1.4, dash: [4, 4] });
              P.fn(L[st.noise], { color: T.blue, width: 2.8 });
              if (st.prior !== 'none') {
                const lam = st.prior === 'gauss' ? 1 / (2 * st.tau * st.tau) : 1 / st.tau;
                P.fn(t => st.prior === 'gauss' ? lam * t * t : lam * Math.abs(t), { color: T.red, width: 2, dash: [6, 3] });
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('dashed grey = squared error, for comparison', w - 16, 8);
            ctx.fillStyle = T.red; ctx.fillText(st.prior === 'none' ? '' : 'dashed red = the penalty term as a function of a weight', w - 16, 24);
          }
        });
        host.appendChild(objBox);
        function obj() {
          const nll = {
            gauss: '\\tfrac{1}{2\\sigma^2}\\sum_i (y_i-\\hat y_i)^2',
            lap: '\\tfrac{1}{b}\\sum_i |y_i-\\hat y_i|',
            huber: '\\sum_i \\rho_\\delta(y_i-\\hat y_i)',
            pois: '\\sum_i (\\hat\\lambda_i - y_i\\log\\hat\\lambda_i)'
          }[st.noise];
          const pen = st.prior === 'none' ? '' : (st.prior === 'gauss'
            ? ' \\;+\\; \\lambda\\|\\theta\\|_2^2,\\quad \\lambda=\\tfrac{1}{2\\tau^2}=' + (1 / (2 * st.tau * st.tau)).toFixed(3)
            : ' \\;+\\; \\lambda\\|\\theta\\|_1,\\quad \\lambda=\\tfrac{1}{b}=' + (1 / st.tau).toFixed(3));
          objBox.innerHTML = '<p class="boxtitle">the objective you are minimising</p>$$' + nll + pen + '$$' +
            '<p class="small" style="margin:0">' + (st.prior === 'none'
              ? 'Flat prior — this is plain maximum likelihood.'
              : (st.prior === 'gauss'
                ? 'Gaussian prior with τ = ' + st.tau.toFixed(2) + ' → ridge. As τ → ∞, λ → 0 and MAP becomes MLE.'
                : 'Laplace prior with scale ' + st.tau.toFixed(2) + ' → lasso; the constant subgradient is what pins coefficients at exactly zero (§2.3).')) + '</p>';
          ML.typeset(objBox);
        }
        obj();
      },

      shrink: function (host) {
        const R = Num.rng(19);
        const n = 90, p = 6;
        const Xr = [], y = [];
        const trueW = [1.8, -1.2, 0.9, 0.0, 0.0, 0.0];
        for (let i = 0; i < n; i++) {
          const base = R.normal(0, 1);
          const x = [base, base * .85 + R.normal(0, .5), R.normal(0, 1), R.normal(0, 1), base * .6 + R.normal(0, .8), R.normal(0, 1)];
          Xr.push(x); y.push(Num.dot(x, trueW) + R.normal(0, 1.1));
        }
        const st = Viz.controls(host, [
          { k: 'kind', label: 'penalty', type: 'buttons', value: 'l2', options: [{ v: 'l2', t: 'ridge (L2)' }, { v: 'l1', t: 'lasso (L1)' }] },
          { k: 'logl', label: 'λ', min: -4, max: 3, step: .05, value: -2, fmt: v => Math.pow(10, v).toFixed(3) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nz', label: 'non-zero coefs', cls: 'key' }, { k: 'norm', label: '‖w‖' }, { k: 'mse', label: 'train MSE' }
        ]);
        function fit(lam, l1) {
          if (!l1) return Num.ridgeFit(Xr, y, lam);
          let w = new Array(p).fill(0);
          for (let it = 0; it < 160; it++) {
            for (let j = 0; j < p; j++) {
              let rho = 0, zz = 0;
              for (let i = 0; i < n; i++) {
                let pred = 0; for (let k = 0; k < p; k++) if (k !== j) pred += Xr[i][k] * w[k];
                rho += Xr[i][j] * (y[i] - pred); zz += Xr[i][j] * Xr[i][j];
              }
              w[j] = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam * n / 2) / (zz || 1);
            }
          }
          return w;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w2, h, T) {
            const lams = [];
            for (let e = -4; e <= 3.01; e += .12) lams.push(Math.pow(10, e));
            const paths = lams.map(l => fit(l, st.kind === 'l1'));
            const P = Viz.plot(ctx, w2, h, { xd: [-4, 3], yd: [-2.2, 2.4] })
              .frame({ xlabel: 'log₁₀ λ', ylabel: 'coefficient value', xfmt: v => v.toFixed(0) });
            const cols = Labs.palette(p);
            P.clip(() => {
              for (let j = 0; j < p; j++) P.line(lams.map((l, i) => [Math.log10(l), paths[i][j]]), { color: cols[j], width: 2 });
              P.hline(0, { color: T.faint, dash: [2, 3] });
              P.vline(st.logl, { color: T.text, dash: [4, 4] });
            });
            const wNow = fit(Math.pow(10, st.logl), st.kind === 'l1');
            wNow.forEach((v, j) => P.dots([[st.logl, v]], { r: 4.6, color: cols[j], stroke: true }));
            const pred = Xr.map(x => Num.dot(x, wNow));
            out({
              nz: wNow.filter(v => Math.abs(v) > 1e-4).length + ' / ' + p,
              norm: Math.sqrt(Num.dot(wNow, wNow)).toFixed(2),
              mse: Num.mean(y.map((v, i) => (v - pred[i]) ** 2)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'True coefficients are [1.8, −1.2, 0.9, 0, 0, 0], with features 1, 2 and 5 correlated. Ridge splits weight between the correlated columns and keeps everything; lasso selects — and <i>which</i> of a correlated group it picks is unstable, which is exactly what elastic net fixes.');
      }
    },
    quiz: [
      {
        q: 'A colleague says "we chose MSE because it is standard". The more precise justification is…',
        options: ['it is fastest to compute', 'it is the negative log-likelihood under Gaussian noise', 'it is robust to outliers', 'it is the only differentiable loss'],
        answer: 1,
        why: 'Squared error is the Gaussian NLL up to a constant and the 1/2σ² scale. Heavy-tailed noise implies Laplace (MAE) or Huber instead.'
      },
      {
        q: 'Ridge with penalty λ‖w‖² corresponds to which prior?',
        options: ['Uniform on [−1,1]', 'Zero-mean Gaussian with variance 1/(2λ)', 'Laplace with scale 1/λ', 'Beta(1,1)'],
        answer: 1,
        why: 'The log of a zero-mean Gaussian prior is −‖w‖²/(2τ²); matching gives λ = 1/(2τ²). A strong penalty is a confident prior that weights are near zero.'
      },
      {
        q: 'Residuals contain several genuine extreme outliers. Taking that seriously implies which loss?',
        options: ['Squared error', 'Absolute error (Laplace) or Huber', 'Cross-entropy', 'Poisson deviance'],
        answer: 1,
        why: 'A heavy-tailed noise model yields a loss growing linearly rather than quadratically, so one point cannot dominate the fit.'
      }
    ],
    cards: [
      { q: 'The MLE→loss bridge', a: 'The NLL of your assumed noise model IS your loss: Gaussian→MSE, Bernoulli→cross-entropy, Poisson→deviance, Laplace→MAE.' },
      { q: 'The MAP→regularizer bridge', a: 'Gaussian prior → L2 with λ=1/(2τ²); Laplace prior → L1. Strength is the prior’s inverse variance.' },
      { q: 'When does MAP equal MLE?', a: 'When the prior variance → ∞ (flat prior).' }
    ]
  });
})();
