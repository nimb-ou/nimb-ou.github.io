/* ============================================================
   PART 1 — Foundations (1.6 – 1.11)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.6 */
  ML.section({
    id: 'intervals', track: 'foundations', num: '1.6',
    title: 'Confidence vs credible intervals; testing; A/B; peeking',
    lede: 'Feeds evaluation in §2.13 and every experiment you will be asked to design on the spot. The peeking simulation below is the fastest way to stop believing a running p-value.',
    rests: 'Rests on §1.4 (why a sample says anything) and §1.2 (the Beta prior that makes Thompson sampling one line).',
    html: `
<h2><span class="sn">1.6.1</span> Two intervals that are not the same object</h2>
<p>A 95% <b>confidence</b> interval is a statement about the <i>procedure</i>: if you repeated the experiment forever, 95% of the intervals it produced would contain the true $\\theta$. It is <b>not</b> a 95% probability that this particular interval does. A 95% <b>credible</b> interval is that direct probability statement — and it is available to you only because you supplied a prior. Frequentist guarantees are about long runs; Bayesian ones are about your beliefs given the data.</p>

${H.lab('coverage', 'Coverage, simulated — what "95%" actually promises', 'Each horizontal line is one experiment’s interval. Red ones miss the true value. Run enough of them and the miss rate converges to 5% — that, and nothing else, is what the number means.')}

<h2><span class="sn">1.6.2</span> Testing, and the p-value</h2>
<p>A hypothesis test fixes a type-I error rate $\\alpha$ <i>in advance</i> and reports $p = P(\\text{data at least this extreme} \\mid H_0)$. The p-value is a property of the data under the null — never the probability that the null is true, and never the probability you are wrong.</p>

<h2><span class="sn">1.6.3</span> Peeking</h2>
<p>Watching a running A/B test and stopping the first time $p < 0.05$ does not give you a 5% error rate. Each look is another chance for the random walk of the test statistic to cross the boundary, and the looks are correlated, so the realised false-positive rate climbs — to roughly <b>20–30% with a handful of looks</b>, and toward 100% under continuous monitoring of an infinitely-running test. The fix is not discipline, it is method: <mark>pre-register the sample size, or use a sequential procedure</mark> — alpha-spending (O'Brien–Fleming), always-valid p-values / confidence sequences, or a Bayesian test with a decision rule. All stay valid under continuous monitoring because they budget error across looks.</p>

${H.lab('peek', 'Peeking, simulated on a null A/A test', 'Both arms are identical — every "win" here is false. Run the simulation and compare the fixed-horizon error rate against the stop-when-significant rate. The counter is the number that ends arguments.')}

${H.worked('worked number — how long must this test run?', `
<p>For two proportions at $\\alpha=0.05$ two-sided and 80% power, sample size per arm is well approximated by</p>
$$n \\approx \\frac{16\\,p(1-p)}{\\delta^2}$$
<p>(the 16 is $2(z_{0.975}+z_{0.80})^2 = 2(1.96+0.84)^2 \\approx 15.7$). Baseline conversion 4%, detecting a 10% relative lift — so $\\delta = 0.004$:</p>
$$n \\approx \\frac{16 \\times 0.04 \\times 0.96}{0.004^2} = \\frac{0.6144}{0.000016} \\approx 38{,}400 \\text{ per arm}$$
<p>Nearly 77,000 users. At 5,000 eligible users a day that is <b>16 days</b> — and you commit to those 16 days before you start. The lesson worth carrying: $n$ scales with $1/\\delta^2$, so <mark>halving the effect you want to detect quadruples the test</mark>. Most "inconclusive" experiments were never powered to conclude anything.</p>`)}

${H.lab('power', 'Sample size and power calculator', 'The same formula, live. Watch the quadratic blow-up as you shrink the effect you want to detect, and watch CUPED move the line for free.')}

<h2><span class="sn">1.6.4</span> Multiple testing is peeking’s sibling</h2>
<p>Twenty metrics at $\\alpha = 0.05$ gives a 64% chance of at least one spurious "win" ($1 - 0.95^{20}$). <b>Bonferroni</b> divides $\\alpha$ by the number of tests — safe, conservative, fine for a handful of pre-registered metrics. <b>Benjamini–Hochberg</b> controls the <i>false discovery rate</i> instead and is the right tool when screening many hypotheses and tolerating a known proportion of false positives among discoveries. Either way, declare one primary metric in advance; everything else is a guardrail or exploratory, and labelling them honestly is what separates an experiment from a fishing trip.</p>

<h2><span class="sn">1.6.5</span> CUPED — the variance reduction worth knowing</h2>
<p>With a pre-experiment covariate $X$ (last month's spend) correlated with the outcome $Y$, analyse $Y_{\\text{adj}} = Y - \\theta(X - \\mathbb{E}[X])$ with $\\theta = \\mathrm{Cov}(Y,X)/\\mathrm{Var}(X)$. The estimator stays unbiased and its variance falls by $(1-\\rho^2)$: at $\\rho = 0.6$ that is a <b>36% variance reduction</b>, so the same power arrives in 36% fewer samples — ten days instead of sixteen, for a line of arithmetic. It is the most impactful technique in experimentation and nearly always available, because pre-period behaviour usually predicts post-period behaviour.</p>

<h2><span class="sn">1.6.6</span> Bandits, when a fixed horizon is the wrong instrument</h2>
<p>An A/B test deliberately wastes traffic on the losing arm for the whole run — that is the price of a clean estimate. A <b>multi-armed bandit</b> instead shifts traffic toward whatever is winning while it learns, trading estimation precision for cumulative reward.</p>
${H.table(['Algorithm', 'Rule', 'Character'], [
      ['ε-greedy', 'explore at fixed rate ε, else exploit', 'trivial; never stops wasting ε'],
      ['UCB', 'play $\\arg\\max_a\\ \\hat\\mu_a + \\sqrt{2\\ln t / n_a}$', 'uncertainty itself is a reason to try something'],
      ['Thompson sampling', 'draw a parameter from each arm’s posterior, play the argmax', 'one line with a Beta prior; usually the best of the three in practice']
    ])}
<p>The formal object is <b>regret</b> — cumulative reward lost against always playing the best arm — and good algorithms achieve regret growing like $\\log T$ rather than $T$. Choose a bandit when there are many arms, when exploration is expensive, or when the environment is non-stationary; choose a fixed-horizon test when you need an unbiased effect size to defend a decision, which in a regulated setting is most of the time. <b>Contextual bandits</b> condition on features — the honest description of most personalisation systems — and note what that implies: the logged data is now confounded by the policy that produced it, so evaluating a new policy offline needs inverse-propensity weighting (§1.7), not a naive average.</p>

${H.lab('bandit', 'Three bandits racing', 'Regret curves, computed live. Watch UCB’s early exploration cost and Thompson’s quiet dominance; then set the arms nearly equal and see every algorithm struggle — that is the case where an A/B test’s unbiased estimate is worth more than the reward you gave up.')}

${H.probe([
      ['Why can’t I stop as soon as it is significant?', 'Repeated correlated looks multiply the chance of a spurious crossing; you need a sequential correction that budgets α across looks.'],
      ['Confidence vs credible, one line each?', 'Coverage of the procedure vs probability of the parameter given a prior.'],
      ['How would you halve the runtime of an underpowered test?', 'CUPED on a pre-period covariate, or accept detecting a larger effect — n scales with 1/δ².']
    ], 'Reporting the observed effect size at an optimal stopping time as unbiased. It is inflated by construction.')}`,
    labs: {
      coverage: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sample size per experiment', min: 5, max: 200, step: 5, value: 30, fmt: v => v },
          { k: 'conf', label: 'confidence level', min: .5, max: .99, step: .01, value: .95, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'reps', label: 'experiments shown', min: 20, max: 200, step: 10, value: 60, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cov', label: 'observed coverage', cls: 'key' }, { k: 'miss', label: 'intervals missing' }, { k: 'w', label: 'mean half-width' }
        ]);
        let seed = 5;
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(seed);
            const mu = 0, sig = 1, z = Num.normPpf(1 - (1 - st.conf) / 2);
            const rows = [];
            for (let r = 0; r < st.reps; r++) {
              const xs = []; for (let i = 0; i < st.n; i++) xs.push(R.normal(mu, sig));
              const m = Num.mean(xs), se = Num.sd(xs, true) / Math.sqrt(st.n);
              rows.push({ m: m, lo: m - z * se, hi: m + z * se, hw: z * se });
            }
            const span = Math.max(1.2, Math.max.apply(null, rows.map(r => Math.abs(r.m) + r.hw)) * 1.05);
            const P = Viz.plot(ctx, w, h, { xd: [-span, span], yd: [0, st.reps], pad: { l: 30, r: 14, t: 12, b: 32 } })
              .frame({ yticks: [], xlabel: 'estimate of the mean (true value = 0)' });
            let miss = 0;
            rows.forEach((r, i) => {
              const hit = r.lo <= mu && r.hi >= mu;
              if (!hit) miss++;
              P.line([[r.lo, i + .5], [r.hi, i + .5]], { color: hit ? T.blue : T.red, width: 2, alpha: hit ? .55 : 1 });
              P.dots([[r.m, i + .5]], { r: 1.9, color: hit ? T.blue : T.red });
            });
            P.vline(0, { color: T.amber, dash: false, width: 1.6, label: 'truth' });
            out({
              cov: (100 * (1 - miss / rows.length)).toFixed(1) + '%',
              miss: miss + ' of ' + rows.length,
              w: Num.mean(rows.map(r => r.hw)).toFixed(3)
            });
          }
        });
        Viz.buttons(host, [{ label: 'Run again', primary: true, on: () => { seed = Math.floor(Math.random() * 1e6); S.redraw(); } }]);
        Viz.note(host, 'Nothing about a single red interval is "wrong" — the guarantee was always about the long-run frequency of the procedure, which is exactly why the phrase "95% probability the true value is in <i>this</i> interval" is a Bayesian statement requiring a prior.');
      },

      peek: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'users per arm (final)', min: 200, max: 8000, step: 200, value: 2000, fmt: v => v.toLocaleString() },
          { k: 'looks', label: 'number of peeks', min: 1, max: 100, step: 1, value: 20, fmt: v => v },
          { k: 'p', label: 'true conversion (both arms)', min: .02, max: .5, step: .01, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'lift', label: 'true lift in B', min: 0, max: .3, step: .005, value: 0, fmt: v => v === 0 ? 'none (A/A)' : '+' + (v * 100).toFixed(1) + '%' }
        ], () => { runs = []; S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'fixed', label: 'fixed-horizon false wins', cls: 'key' },
          { k: 'peeked', label: 'peeking false wins', cls: 'bad' },
          { k: 'runs', label: 'simulations' }
        ]);
        let runs = [], trace = null;
        function simulate(nSim) {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          for (let s = 0; s < nSim; s++) {
            let ca = 0, cb = 0, stopped = false, stopAt = 0;
            const pathP = [];
            const step = Math.max(1, Math.floor(st.n / st.looks));
            for (let i = 1; i <= st.n; i++) {
              ca += R() < st.p ? 1 : 0;
              cb += R() < st.p * (1 + st.lift) ? 1 : 0;
              if (i % step === 0 || i === st.n) {
                const pa = ca / i, pb = cb / i, pp = (ca + cb) / (2 * i);
                const se = Math.sqrt(2 * pp * (1 - pp) / i) || 1e-9;
                const zz = (pb - pa) / se;
                const pv = 2 * (1 - Num.normCdf(Math.abs(zz)));
                pathP.push([i, Math.max(1e-4, pv)]);
                if (!stopped && pv < .05) { stopped = true; stopAt = i; }
              }
            }
            runs.push({ fixed: pathP[pathP.length - 1][1] < .05, peeked: stopped, stopAt: stopAt, path: pathP });
            if (s === nSim - 1) trace = runs[runs.length - 1];
          }
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [0, st.n], yd: [0, .6] })
              .frame({ xlabel: 'users observed per arm', ylabel: 'running p-value' });
            P.clip(() => {
              runs.slice(-40).forEach(r => {
                P.line(r.path, { color: r.peeked ? T.red : T.faint, width: 1, alpha: r.peeked ? .5 : .3 });
              });
              if (trace) {
                P.line(trace.path, { color: T.blue, width: 2.2 });
                if (trace.peeked) {
                  const pt = trace.path.find(p => p[0] >= trace.stopAt);
                  if (pt) { P.dots([pt], { r: 6, color: T.red, stroke: true }); P.text(pt[0], pt[1], '  stopped here — "significant"', { color: T.red, font: '11px ui-sans-serif' }); }
                }
              }
              P.hline(.05, { color: T.amber, label: 'α = 0.05' });
            });
            const f = runs.filter(r => r.fixed).length, pk = runs.filter(r => r.peeked).length;
            out({
              fixed: runs.length ? (100 * f / runs.length).toFixed(1) + '%' : '—',
              peeked: runs.length ? (100 * pk / runs.length).toFixed(1) + '%' : '—',
              runs: runs.length
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Run 1 experiment', primary: true, on: () => { simulate(1); S.redraw(); } },
          { label: 'Run 200', on: () => { simulate(200); S.redraw(); } },
          { label: 'Reset', on: () => { runs = []; trace = null; S.redraw(); } }
        ]);
        Viz.note(host, 'With <b>lift = none</b>, every declared win is false by construction. The fixed-horizon column should sit near 5%; the peeking column climbs with the number of looks — that gap is the entire argument for sequential methods.');
      },

      power: function (host) {
        const st = Viz.controls(host, [
          { k: 'base', label: 'baseline conversion p', min: .005, max: .5, step: .005, value: .04, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'rel', label: 'relative lift to detect', min: .01, max: .5, step: .005, value: .10, fmt: v => (v * 100).toFixed(1) + '%' },
          { k: 'traffic', label: 'eligible users per day', min: 200, max: 40000, step: 200, value: 5000, fmt: v => v.toLocaleString() },
          { k: 'rho', label: 'CUPED covariate correlation ρ', min: 0, max: .9, step: .01, value: 0, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'n', label: 'n per arm', cls: 'key' }, { k: 'tot', label: 'total users' },
          { k: 'days', label: 'days to run' }, { k: 'saved', label: 'days saved by CUPED', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const p = st.base;
            const nFor = (rel, rho) => 16 * p * (1 - p) * (1 - rho * rho) / Math.pow(p * rel, 2);
            const P = Viz.plot(ctx, w, h, { xd: [.01, .5], yd: [0, Math.min(4e5, nFor(.02, 0) * 1.05)] })
              .frame({ xlabel: 'relative lift you want to detect', ylabel: 'n per arm', yfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0), xfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(r => nFor(r, 0), { color: T.blue, width: 2.6, n: 260 });
              if (st.rho > 0) P.fn(r => nFor(r, st.rho), { color: T.green, width: 2.4, n: 260 });
              P.vline(st.rel, { color: T.text, dash: [4, 4] });
              P.dots([[st.rel, nFor(st.rel, st.rho)]], { r: 5.5, color: st.rho > 0 ? T.green : T.blue, stroke: true });
            });
            const n = nFor(st.rel, st.rho), n0 = nFor(st.rel, 0);
            out({
              n: Math.ceil(n).toLocaleString(), tot: Math.ceil(2 * n).toLocaleString(),
              days: Math.ceil(2 * n / st.traffic), saved: st.rho > 0 ? Math.max(0, Math.ceil(2 * n0 / st.traffic) - Math.ceil(2 * n / st.traffic)) : 0
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'plain difference-in-proportions' }, { c: Viz.theme().green, t: 'with CUPED variance reduction (1−ρ²)' }]);
      },

      bandit: function (host) {
        const st = Viz.controls(host, [
          { k: 'k', label: 'arms', min: 2, max: 8, step: 1, value: 4, fmt: v => v },
          { k: 'gap', label: 'gap between best and rest', min: .005, max: .2, step: .005, value: .05, fmt: v => (v * 100).toFixed(1) + ' pts' },
          { k: 'T', label: 'rounds', min: 500, max: 20000, step: 500, value: 5000, fmt: v => v.toLocaleString() },
          { k: 'eps', label: 'ε for ε-greedy', min: .01, max: .5, step: .01, value: .1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'eg', label: 'ε-greedy regret' }, { k: 'ucb', label: 'UCB regret' },
          { k: 'ts', label: 'Thompson regret', cls: 'good' }, { k: 'ab', label: 'A/B test regret', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(77);
            const k = st.k, best = .3 + st.gap;
            const mu = Array.from({ length: k }, (_, i) => i === 0 ? best : .3 - R() * .02);
            function run(policy) {
              const n = new Array(k).fill(0), s = new Array(k).fill(0);
              let regret = 0; const path = [];
              for (let t = 1; t <= st.T; t++) {
                let a;
                if (policy === 'eg') a = R() < st.eps ? R.int(k) : argmax(n.map((c, i) => c ? s[i] / c : 1e9));
                else if (policy === 'ucb') a = argmax(n.map((c, i) => c ? s[i] / c + Math.sqrt(2 * Math.log(t) / c) : 1e9));
                else if (policy === 'ts') a = argmax(n.map((c, i) => R.beta(1 + s[i], 1 + c - s[i])));
                else a = t % k;  // A/B: equal split throughout
                const r = R() < mu[a] ? 1 : 0;
                n[a]++; s[a] += r;
                regret += best - mu[a];
                if (t % Math.ceil(st.T / 180) === 0) path.push([t, regret]);
              }
              return { path: path, regret: regret };
            }
            function argmax(a) { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; }
            const res = { eg: run('eg'), ucb: run('ucb'), ts: run('ts'), ab: run('ab') };
            const maxR = Math.max(res.ab.regret, res.eg.regret) * 1.05;
            const P = Viz.plot(ctx, w, h, { xd: [0, st.T], yd: [0, maxR] })
              .frame({ xlabel: 'rounds', ylabel: 'cumulative regret (reward lost)' });
            P.clip(() => {
              P.line(res.ab.path, { color: T.red, width: 2 });
              P.line(res.eg.path, { color: T.amber, width: 2 });
              P.line(res.ucb.path, { color: T.blue, width: 2 });
              P.line(res.ts.path, { color: T.green, width: 2.6 });
            });
            out({
              eg: res.eg.regret.toFixed(1), ucb: res.ucb.regret.toFixed(1),
              ts: res.ts.regret.toFixed(1), ab: res.ab.regret.toFixed(1)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().green, t: 'Thompson sampling' }, { c: Viz.theme().blue, t: 'UCB' },
          { c: Viz.theme().amber, t: 'ε-greedy' }, { c: Viz.theme().red, t: 'fixed A/B split (regret grows linearly)' }
        ]);
        Viz.note(host, 'The A/B line is straight because an even split keeps paying the full gap on every losing impression for the whole run. That straight line is the <i>price of an unbiased estimate</i> — sometimes worth paying, which is the actual answer to "bandit or A/B?"');
      }
    },
    quiz: [
      {
        q: 'You monitor an A/B test continuously and stop the moment p < 0.05. Your realised false-positive rate is closest to…',
        options: ['5%', '10%', '20–30% with a handful of looks, and worse with continuous monitoring', 'exactly 50%'],
        answer: 2,
        why: 'Each look is another chance to cross; with unlimited looks on a null effect the crossing probability tends to 1. Alpha-spending or always-valid p-values restore validity.'
      },
      {
        q: 'Baseline 4%, you want to detect a 5% relative lift instead of 10%. The required sample size…',
        options: ['halves', 'stays the same', 'doubles', 'quadruples'],
        answer: 3,
        why: 'n ∝ 1/δ². Halving the detectable effect quadruples n — which is why "let’s just detect a smaller effect" is an expensive sentence.'
      },
      {
        q: 'CUPED with a pre-period covariate correlated ρ = 0.7 with the outcome reduces variance by…',
        options: ['70%', '49%', '30%', '7%'],
        answer: 1,
        why: 'Variance falls by (1 − ρ²) = 0.51, i.e. a 49% reduction, so roughly half the samples for the same power.'
      }
    ],
    cards: [
      { q: 'Confidence vs credible', a: 'Coverage of the procedure over repeated experiments vs probability of the parameter given data and a prior.' },
      { q: 'A/B sample size formula', a: '$n \\approx 16p(1-p)/\\delta^2$ per arm at α=0.05, 80% power. Halving δ quadruples n.' },
      { q: 'CUPED', a: '$Y_{adj}=Y-\\theta(X-\\mathbb{E}X)$ with a pre-period covariate; variance falls by $(1-\\rho^2)$, unbiased.' },
      { q: 'Bandit regret', a: 'Cumulative reward lost vs always playing the best arm; good algorithms grow like $\\log T$, an even A/B split grows like $T$.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.7 */
  ML.section({
    id: 'causal', track: 'foundations', num: '1.7',
    title: 'The causal inference a data scientist actually uses',
    lede: 'Every causal question is the same question: what would have happened to these same units under the other treatment? The counterfactual is missing by construction, so all technique is about reconstructing it credibly.',
    rests: 'Directly relevant to credit-policy decisions and to attributing a drift event to a cause (§2.18 lifecycle).',
    html: `
<h2><span class="sn">1.7.1</span> The target</h2>
<p>Usually the <b>average treatment effect</b> $\\mathrm{ATE} = \\mathbb{E}[Y(1) - Y(0)]$: the average difference between the world where everyone is treated and the world where nobody is. You observe one potential outcome per unit, never both.</p>
<p><b>Confounding</b> is a common cause of both treatment and outcome; it makes the naive difference in means a biased estimate of the ATE. Randomisation is the gold standard because it severs the arrow into treatment. When you cannot randomise:</p>

${H.table(['Strategy', 'Idea', 'The assumption an interviewer will make you defend'], [
      ['<b>Propensity scores</b>', '$e(x)=P(T{=}1\\mid x)$; match or inverse-probability-weight so treated and control covariate distributions coincide', 'No unmeasured confounding (conditional ignorability) <i>plus</i> overlap. Check both, and always report covariate balance after weighting.'],
      ['<b>Difference-in-differences</b>', 'Subtract the control group’s pre-period trend from the treated group’s change', '<b>Parallel trends</b> — absent treatment both groups would have moved together. Testable in spirit by inspecting several pre-periods.'],
      ['<b>Instrumental variables</b>', 'Find $Z$ that shifts treatment but affects $Y$ only through $T$', 'Relevance + exclusion. Weak instruments give wildly unstable estimates, so report first-stage strength.'],
      ['<b>Regression discontinuity</b>', 'Compare units just above and just below a cut-off', 'No manipulation of the running variable at the threshold']
    ])}

${H.lab('dag', 'Four small DAGs, and the bias each one creates', 'Simulated data with a known true effect. Toggle what you condition on and watch the estimate move away from the truth — including the case that surprises people: conditioning on a collider <i>creates</i> bias where there was none.')}

${H.worked('worked Simpson’s paradox — the aggregate lies', `
<p>Two underwriting policies, default outcomes by applicant segment:</p>
${H.code(`Prime      A: 81/900 default = 9.0%    B:   8/100 = 8.0%   → B better
Subprime   A: 20/100 default = 20%      B: 171/900 = 19%    → B better
──────────────────────────────────────────────────────────────────────
Overall    A: 101/1000 = 10.1%          B: 179/1000 = 17.9%  → A better (!)`)}
<p>B wins in <i>every</i> segment and loses overall, because B was applied mostly to subprime applicants. The segment is a confounder — it causes both the policy assignment and the outcome — so the aggregate comparison is not a causal one. <mark>The reversal is not a paradox, it is confounding with a memorable name</mark> — and the fix is not "always disaggregate" (that way lies colliders); it is a DAG that tells you which variables to condition on.</p>`)}

${H.lab('simpson', 'Simpson’s paradox, with the mixing under your control', 'Move the share of each policy that went to subprime applicants. Watch the overall comparison flip while the within-segment comparison never moves.')}

<h2><span class="sn">1.7.2</span> Three biases that break credit models specifically</h2>
<ul>
<li><b>Selection bias.</b> You only observe repayment for applicants you approved, so your training data is the population your <i>old</i> policy chose. This is why reject inference exists, and why a model trained on approved-only data systematically misjudges the region near the cut-off.</li>
<li><b>Survivorship.</b> The same problem in time: accounts that closed are gone from the panel, so the survivors look better than the cohort was.</li>
<li><b>Regression to the mean.</b> Pick the worst-performing branches, intervene, and they improve — partly because extreme observations contain extreme noise, which is not repeated. A control group is the only defence, and noticing that you need one is the whole skill.</li>
</ul>

${H.probe([
      ['IV or DiD?', 'IV when confounders are unmeasured but a valid instrument exists; DiD when you have a clean pre/post and a control group with credible parallel trends.'],
      ['What does propensity weighting assume?', 'Conditional ignorability and overlap — and you must show covariate balance after weighting, not just report the estimate.']
    ], 'Controlling for a collider (a common <i>effect</i>) or a mediator. Both induce bias rather than removing it — "control for everything" is not a strategy.')}`,
    labs: {
      dag: function (host) {
        const st = Viz.controls(host, [
          { k: 'g', label: 'graph', type: 'buttons', value: 'conf', options: [
            { v: 'conf', t: 'confounder' }, { v: 'coll', t: 'collider' }, { v: 'med', t: 'mediator' }, { v: 'rand', t: 'randomised' }] },
          { k: 'adj', label: 'condition on the third variable X', type: 'toggle', value: false },
          { k: 'str', label: 'strength of X’s effect', min: 0, max: 2, step: .05, value: 1.2, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true effect of T on Y', cls: 'key' },
          { k: 'naive', label: 'estimate' }, { k: 'bias', label: 'bias', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41), n = 1200, TRUE = 1.0;
            const rows = [];
            for (let i = 0; i < n; i++) {
              let X, Tt, Y;
              if (st.g === 'conf') { X = R.normal(0, 1); Tt = (st.str * X + R.normal(0, 1)) > 0 ? 1 : 0; Y = TRUE * Tt + st.str * X + R.normal(0, 1); }
              else if (st.g === 'coll') { Tt = R() < .5 ? 1 : 0; Y = TRUE * Tt + R.normal(0, 1); X = st.str * Tt + st.str * Y + R.normal(0, 1); }
              else if (st.g === 'med') { Tt = R() < .5 ? 1 : 0; X = st.str * Tt + R.normal(0, 1); Y = 0.4 * Tt + st.str * X + R.normal(0, 1); }
              else { Tt = R() < .5 ? 1 : 0; X = R.normal(0, 1); Y = TRUE * Tt + st.str * X + R.normal(0, 1); }
              rows.push({ X: X, T: Tt, Y: Y });
            }
            const trueEff = st.g === 'med' ? 0.4 + st.str * st.str : TRUE;
            let est;
            if (!st.adj) {
              const t1 = rows.filter(r => r.T === 1), t0 = rows.filter(r => r.T === 0);
              est = Num.mean(t1.map(r => r.Y)) - Num.mean(t0.map(r => r.Y));
            } else {
              const D = rows.map(r => [1, r.T, r.X]);
              const beta = Num.ridgeFit(D, rows.map(r => r.Y), 1e-8);
              est = beta[1];
            }
            // draw DAG
            const cx = w * .27, cy = h * .5;
            const nodes = {
              T: { x: cx - 70, y: cy + 30, l: 'T' }, Y: { x: cx + 70, y: cy + 30, l: 'Y' }, X: { x: cx, y: cy - 55, l: 'X' }
            };
            const edges = {
              conf: [['X', 'T'], ['X', 'Y'], ['T', 'Y']],
              coll: [['T', 'X'], ['Y', 'X'], ['T', 'Y']],
              med: [['T', 'X'], ['X', 'Y'], ['T', 'Y']],
              rand: [['X', 'Y'], ['T', 'Y']]
            }[st.g];
            edges.forEach(([a, b]) => {
              const A = nodes[a], B = nodes[b];
              const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy);
              const ux = dx / L, uy = dy / L;
              ctx.strokeStyle = (a === 'T' && b === 'Y') ? T.text : T.faint;
              ctx.lineWidth = (a === 'T' && b === 'Y') ? 2.2 : 1.5;
              ctx.beginPath(); ctx.moveTo(A.x + ux * 20, A.y + uy * 20); ctx.lineTo(B.x - ux * 22, B.y - uy * 22); ctx.stroke();
              const ang = Math.atan2(dy, dx);
              ctx.fillStyle = ctx.strokeStyle;
              ctx.beginPath();
              ctx.moveTo(B.x - ux * 20, B.y - uy * 20);
              ctx.lineTo(B.x - ux * 20 - 8 * Math.cos(ang - .4), B.y - uy * 20 - 8 * Math.sin(ang - .4));
              ctx.lineTo(B.x - ux * 20 - 8 * Math.cos(ang + .4), B.y - uy * 20 - 8 * Math.sin(ang + .4));
              ctx.closePath(); ctx.fill();
            });
            Object.keys(nodes).forEach(k => {
              const N = nodes[k];
              ctx.beginPath(); ctx.arc(N.x, N.y, 19, 0, 6.3);
              ctx.fillStyle = (k === 'X' && st.adj) ? T.blue : T.panel;
              ctx.fill(); ctx.strokeStyle = (k === 'X' && st.adj) ? T.blue : T.line; ctx.lineWidth = 2; ctx.stroke();
              ctx.fillStyle = (k === 'X' && st.adj) ? '#fff' : T.text;
              ctx.font = 'bold 14px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(N.l, N.x, N.y);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(st.adj ? 'conditioning on X (blue)' : 'not conditioning on X', cx, h - 18);
            // bias bar
            const bx = w * .58, bw = w * .36;
            ctx.textAlign = 'left'; ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = T.muted;
            ctx.fillText('true effect', bx, 40); ctx.fillText('estimate', bx, 84);
            const scale = bw / Math.max(2.4, Math.abs(est) * 1.2, trueEff * 1.2);
            ctx.fillStyle = T.green; ctx.fillRect(bx, 48, Math.max(0, trueEff) * scale, 18);
            ctx.fillStyle = Math.abs(est - trueEff) > .12 ? T.red : T.green;
            ctx.fillRect(bx, 92, Math.max(0, est) * scale, 18);
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace';
            ctx.fillText(trueEff.toFixed(2), bx + Math.max(0, trueEff) * scale + 6, 58);
            ctx.fillText(est.toFixed(2), bx + Math.max(0, est) * scale + 6, 102);
            const msg = {
              conf: st.adj ? 'Correct: adjusting for the confounder removes the backdoor path.' : 'Biased: X opens a backdoor path T ← X → Y.',
              coll: st.adj ? 'Now biased: conditioning on a collider OPENS a path that was closed.' : 'Correct: the collider path is closed unless you condition on it.',
              med: st.adj ? 'This is the direct effect only — the mediated part has been removed.' : 'This is the total effect (direct + mediated).',
              rand: 'Randomisation cuts X → T, so the naive difference is already unbiased; adjusting only reduces variance.'
            }[st.g];
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
            wrapText(ctx, msg, bx, 132, bw, 16);
            out({ truth: trueEff.toFixed(3), naive: est.toFixed(3), bias: (est - trueEff).toFixed(3) });
          }
        });
        function wrapText(ctx, text, x, y, maxW, lh) {
          const words = text.split(' '); let line = '', yy = y;
          words.forEach(word => {
            const test = line + word + ' ';
            if (ctx.measureText(test).width > maxW && line) { ctx.fillText(line, x, yy); line = word + ' '; yy += lh; }
            else line = test;
          });
          ctx.fillText(line, x, yy);
        }
      },

      simpson: function (host) {
        const st = Viz.controls(host, [
          { k: 'mixA', label: 'share of policy A applied to subprime', min: 0, max: 1, step: .01, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'mixB', label: 'share of policy B applied to subprime', min: 0, max: 1, step: .01, value: .9, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'n', label: 'applicants per policy', min: 200, max: 5000, step: 100, value: 1000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'a', label: 'policy A overall' }, { k: 'b', label: 'policy B overall' },
          { k: 'winner', label: 'aggregate says', cls: 'bad' }, { k: 'seg', label: 'every segment says', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            // within-segment rates: B is strictly better in both
            const rate = { prime: { A: .09, B: .08 }, sub: { A: .20, B: .19 } };
            const nA_sub = st.n * st.mixA, nA_pr = st.n - nA_sub;
            const nB_sub = st.n * st.mixB, nB_pr = st.n - nB_sub;
            const defA = nA_pr * rate.prime.A + nA_sub * rate.sub.A;
            const defB = nB_pr * rate.prime.B + nB_sub * rate.sub.B;
            const oA = defA / st.n, oB = defB / st.n;
            const P = Viz.plot(ctx, w, h, { xd: [0, 4], yd: [0, .26], pad: { l: 46, r: 14, t: 16, b: 46 } })
              .frame({ xticks: [], ylabel: 'default rate', yfmt: v => (v * 100).toFixed(0) + '%' });
            const bars = [
              { x: .55, v: rate.prime.A, c: T.blue, l: 'A · prime' }, { x: 1.0, v: rate.prime.B, c: T.red, l: 'B · prime' },
              { x: 1.7, v: rate.sub.A, c: T.blue, l: 'A · sub' }, { x: 2.15, v: rate.sub.B, c: T.red, l: 'B · sub' },
              { x: 2.9, v: oA, c: T.blue, l: 'A · all' }, { x: 3.35, v: oB, c: T.red, l: 'B · all' }
            ];
            bars.forEach(b => {
              const x0 = P.x(b.x - .18), x1 = P.x(b.x + .18);
              ctx.fillStyle = b.c; ctx.fillRect(x0, P.y(b.v), x1 - x0, P.y(0) - P.y(b.v));
              ctx.fillStyle = T.text; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
              ctx.fillText((b.v * 100).toFixed(1) + '%', (x0 + x1) / 2, P.y(b.v) - 4);
              ctx.fillStyle = T.muted; ctx.textBaseline = 'top'; ctx.font = '10px ui-sans-serif';
              ctx.fillText(b.l, (x0 + x1) / 2, P.y(0) + 6);
            });
            ctx.strokeStyle = T.line; ctx.setLineDash([4, 4]);
            [1.35, 2.55].forEach(x => { ctx.beginPath(); ctx.moveTo(P.x(x), P.pad.t); ctx.lineTo(P.x(x), P.y(0)); ctx.stroke(); });
            ctx.setLineDash([]);
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('PRIME SEGMENT', P.x(.78), P.pad.t + 2);
            ctx.fillText('SUBPRIME SEGMENT', P.x(1.93), P.pad.t + 2);
            ctx.fillText('AGGREGATE', P.x(3.12), P.pad.t + 2);
            out({
              a: (oA * 100).toFixed(1) + '%', b: (oB * 100).toFixed(1) + '%',
              winner: oA < oB ? 'A is better' : 'B is better',
              seg: 'B is better'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Reproduce the notebook’s numbers', primary: true, on: () => { st.$set('mixA', .1); st.$set('mixB', .9); st.$set('n', 1000); S.redraw(); } },
          { label: 'Equal mixes (no confounding)', on: () => { st.$set('mixA', .5); st.$set('mixB', .5); S.redraw(); } }
        ]);
        Viz.note(host, 'Set both mixes equal and the paradox vanishes: with the confounder balanced, the aggregate agrees with the segments. That is exactly what randomisation buys you.');
      }
    },
    quiz: [
      {
        q: 'You have a strong pre-period, a treated group, a control group, and unmeasured confounders that are stable over time. The natural estimator is…',
        options: ['propensity score matching', 'difference-in-differences', 'instrumental variables', 'a naive difference in means'],
        answer: 1,
        why: 'DiD differences out any time-invariant confounding; the assumption you must defend is parallel trends.'
      },
      {
        q: 'Conditioning on a collider…',
        options: ['removes confounding bias', 'has no effect', 'creates a spurious association between its causes', 'is always required'],
        answer: 2,
        why: 'Conditioning on a common effect opens a path between its parents. This is why "control for everything" is wrong and why you need a DAG.'
      },
      {
        q: 'A credit model is trained only on approved applicants. What is the problem called and what fixes it?',
        options: ['Overfitting; use regularization', 'Selection bias; reject inference or a randomised holdout above the cut-off', 'Concept drift; retrain', 'Class imbalance; resample'],
        answer: 1,
        why: 'The training population is the one your old policy chose. Reject inference, or better, a small randomised acceptance band that generates unbiased data near the cut-off.'
      }
    ],
    cards: [
      { q: 'ATE and why it is hard', a: '$\\mathbb{E}[Y(1)-Y(0)]$; you never observe both potential outcomes for one unit, so the counterfactual must be reconstructed.' },
      { q: 'Assumptions: propensity / DiD / IV', a: 'Ignorability + overlap; parallel trends; relevance + exclusion.' },
      { q: 'Simpson’s paradox', a: 'A within-segment ordering reverses in the aggregate because segment membership confounds assignment — confounding with a memorable name.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.8 */
  ML.section({
    id: 'linear-algebra', track: 'foundations', num: '1.8',
    title: 'Linear algebra: eigen, SVD, PSD',
    lede: 'Foundation for PCA, kernel validity, attention as a bilinear form, second-order optimisation, and the low-rank claim that LoRA rests on.',
    rests: 'Consumed by §2.10 PCA · §2.6 kernels · §4.3 attention · §4.13 LoRA.',
    html: `
<h2><span class="sn">1.8.1</span> Eigendecomposition</h2>
<p>$A = Q\\Lambda Q^{-1}$ applies to square matrices; when $A$ is symmetric, $Q$ is orthogonal and the eigenvalues are real, so $A = Q\\Lambda Q^\\mathsf{T}$. An eigenvector is a direction the matrix does not rotate — only stretches, by its eigenvalue.</p>

<h2><span class="sn">1.8.2</span> SVD, and why it is the more useful object</h2>
<p>$A = U\\Sigma V^\\mathsf{T}$ exists for <i>any</i> matrix, rectangular included. The singular values are the square roots of the eigenvalues of $A^\\mathsf{T}A$. Geometrically: <mark>every linear map is a rotation, an axis-aligned stretch, and another rotation.</mark></p>
<p>Truncating to the top $r$ singular values gives the best rank-$r$ approximation in Frobenius norm — the <b>Eckart–Young theorem</b>, which is the theorem behind PCA, latent semantic analysis, and the rank-$r$ update in LoRA (§4.13).</p>

${H.lab('svd', 'SVD is rotate · scale · rotate', 'Drag the matrix columns and watch the unit circle become an ellipse in three separate stages. σ₁/σ₂ is the condition number; when σ₂ ≈ 0 the matrix is effectively rank one — the same picture that explains why low-rank adapters work.')}

${H.lab('lowrank', 'Rank-r approximation, on a real matrix', 'A 48×48 image reconstructed from its top r singular values, computed live. Watch the error and the storage cost fall out of the same curve — and notice how few components carry the structure.')}

<h2><span class="sn">1.8.3</span> Positive semi-definiteness</h2>
<p>$A \\succeq 0$ means $x^\\mathsf{T}Ax \\ge 0$ for all $x$, equivalently all eigenvalues $\\ge 0$. Covariance matrices and kernel matrices are PSD, and that is exactly what makes their associated optimizations convex and well-behaved. When someone asks why an RBF kernel is legal, the answer is <b>Mercer's condition</b>: the kernel matrix must be PSD.</p>

${H.table(['Object', 'Why it is PSD', 'What that buys'], [
      ['Covariance $\\Sigma$', 'It is $\\mathbb{E}[(x-\\mu)(x-\\mu)^\\mathsf{T}]$, a sum of outer products', 'PCA is an eigenproblem with real, non-negative eigenvalues'],
      ['Gram / kernel matrix $K$', 'Mercer’s condition', 'The SVM dual is a convex QP with a unique optimum'],
      ['Hessian of a convex function', 'Definition of convexity', 'No bad local minima; Newton’s method is well-posed']
    ])}

${H.probe([
      ['What is the SVD in one sentence?', 'Rotate, scale along axes, rotate — and truncating the singular values gives the best low-rank approximation.'],
      ['Why must a kernel be PSD?', 'It must correspond to an inner product in some feature space; otherwise the dual problem is not convex and the "kernel trick" is meaningless.'],
      ['How does this connect to LoRA?', 'If the fine-tuning update is well approximated by a rank-r matrix, storing $BA$ with $r \\ll d$ loses almost nothing — Eckart–Young is the licence.']
    ])}`,
    labs: {
      svd: function (host) {
        let A = [[1.6, .9], [.4, 1.1]];
        let drag = null;
        const st = Viz.controls(host, [
          { k: 'stage', label: 'stage', type: 'buttons', value: 'all', options: [
            { v: 'in', t: '1 · input' }, { v: 'v', t: '2 · VᵀX rotate' }, { v: 'sv', t: '3 · ΣVᵀX scale' }, { v: 'all', t: '4 · UΣVᵀX' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 's1', label: 'σ₁', cls: 'key' }, { k: 's2', label: 'σ₂' }, { k: 'cond', label: 'condition κ' },
          { k: 'rank', label: 'effective rank' }, { k: 'det', label: 'det' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const M = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]];
            const { U, s, V } = Num.svd(M);
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-2.4, 2.4] }).frame({});
            const circle = [];
            for (let t = 0; t <= 6.3; t += .04) circle.push([Math.cos(t), Math.sin(t)]);
            const applyVt = p => [V[0][0] * p[0] + V[0][1] * p[1], V[1][0] * p[0] + V[1][1] * p[1]];
            const applyS = p => [s[0] * p[0], s[1] * p[1]];
            const applyU = p => [U[0][0] * p[0] + U[1][0] * p[1], U[0][1] * p[0] + U[1][1] * p[1]];
            let shape = circle, label = 'unit circle';
            if (st.stage === 'v') { shape = circle.map(applyVt); label = 'after VᵀX — a rotation, still a circle'; }
            else if (st.stage === 'sv') { shape = circle.map(p => applyS(applyVt(p))); label = 'after ΣVᵀX — stretched along the axes by σ₁, σ₂'; }
            else if (st.stage === 'all') { shape = circle.map(p => applyU(applyS(applyVt(p)))); label = 'after UΣVᵀX = AX — the final ellipse'; }
            P.clip(() => {
              P.line(circle, { color: T.faint, width: 1.2, dash: [4, 3] });
              P.line(shape, { color: T.red, width: 2.6 });
              if (st.stage === 'all' || st.stage === 'in') {
                P.arrow(0, 0, A[0][0], A[1][0], { color: T.blue, width: 2.4 });
                P.arrow(0, 0, A[0][1], A[1][1], { color: T.green, width: 2.4 });
              }
              if (st.stage === 'all') {
                const u1 = [U[0][0], U[0][1]], u2 = [U[1][0], U[1][1]];
                P.arrow(0, 0, u1[0] * s[0], u1[1] * s[0], { color: T.amber, width: 2 });
                P.arrow(0, 0, u2[0] * s[1], u2[1] * s[1], { color: T.amber, width: 1.4 });
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(label, w / 2, 8);
            out({
              s1: s[0].toFixed(3), s2: s[1].toFixed(3),
              cond: (s[0] / (s[1] || 1e-9)).toFixed(1),
              rank: s[1] / (s[0] || 1) < .05 ? '≈ 1' : '2',
              det: (A[0][0] * A[1][1] - A[0][1] * A[1][0]).toFixed(3)
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const x = P.ix(e.x), y = P.iy(e.y);
          if (e.type === 'down') {
            const d1 = (A[0][0] - x) ** 2 + (A[1][0] - y) ** 2, d2 = (A[0][1] - x) ** 2 + (A[1][1] - y) ** 2;
            drag = Math.min(d1, d2) < .3 ? (d1 < d2 ? 0 : 1) : null;
          } else if (e.type === 'move' && e.down && drag !== null) { A[0][drag] = x; A[1][drag] = y; S.redraw(); }
          else if (e.type === 'up') drag = null;
        });
        Viz.buttons(host, [
          { label: 'Make it nearly rank-1', on: () => { A = [[1.5, .75], [1.2, .6]]; S.redraw(); } },
          { label: 'Make it orthogonal', on: () => { A = [[.9, -.44], [.44, .9]]; S.redraw(); } },
          { label: 'Ill-conditioned', on: () => { A = [[2, 1.9], [1.9, 1.85]]; S.redraw(); } }
        ]);
      },

      lowrank: function (host) {
        const N = 48;
        // a structured synthetic "image": low-rank content plus detail
        const M = [];
        for (let i = 0; i < N; i++) {
          const row = new Float64Array(N);
          for (let j = 0; j < N; j++) {
            const x = i / N, y = j / N;
            row[j] = Math.sin(6 * x) * Math.cos(5 * y) + .6 * Math.sin(11 * y) + .5 * (x > .6 && y < .4 ? 1 : 0)
              + .35 * Math.exp(-((x - .3) ** 2 + (y - .7) ** 2) * 30);
          }
          M.push(row);
        }
        const { U, s, V } = Num.svd(M.map(r => Array.from(r)));
        const total = s.reduce((a, b) => a + b * b, 0);
        const st = Viz.controls(host, [
          { k: 'r', label: 'rank r kept', min: 1, max: 24, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'keep', label: 'energy kept', cls: 'key' }, { k: 'err', label: 'relative error' },
          { k: 'nums', label: 'numbers stored' }, { k: 'comp', label: 'compression' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const r = st.r;
            const rec = Array.from({ length: N }, () => new Float64Array(N));
            for (let k = 0; k < r; k++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) rec[i][j] += s[k] * U[k][i] * V[k][j];
            const cell = Math.min((h - 60) / N, (w * .42) / N);
            const draw = (mat, ox, title) => {
              for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
                const v = Math.max(-1.6, Math.min(1.6, mat[i][j]));
                const t = (v + 1.6) / 3.2;
                const g = Math.round(255 * t);
                ctx.fillStyle = 'rgb(' + Math.round(40 + g * .75) + ',' + Math.round(50 + g * .7) + ',' + Math.round(90 + g * .6) + ')';
                ctx.fillRect(ox + j * cell, 30 + i * cell, cell + .6, cell + .6);
              }
              ctx.strokeStyle = T.line; ctx.strokeRect(ox, 30, N * cell, N * cell);
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText(title, ox, 24);
            };
            draw(M, 10, 'original · rank ' + N);
            draw(rec, 20 + N * cell, 'reconstruction · rank ' + r);
            // spectrum
            const sx = 30 + 2 * N * cell, sw = Math.max(60, w - sx - 14);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
            ctx.fillText('singular values', sx, 24);
            const smax = s[0];
            s.slice(0, 24).forEach((v, i) => {
              ctx.fillStyle = i < r ? T.blue : T.line;
              const bh = 6;
              ctx.fillRect(sx, 32 + i * (bh + 2), sw * v / smax, bh);
            });
            let kept = 0; for (let k = 0; k < r; k++) kept += s[k] * s[k];
            const stored = r * (2 * N + 1);
            out({
              keep: (100 * kept / total).toFixed(1) + '%',
              err: (100 * Math.sqrt(Math.max(0, 1 - kept / total))).toFixed(1) + '%',
              nums: stored.toLocaleString() + ' vs ' + (N * N).toLocaleString(),
              comp: (N * N / stored).toFixed(1) + '×'
            });
          }
        });
        Viz.note(host, 'This is Eckart–Young in action: the rank-r truncation is provably the best rank-r approximation in Frobenius norm. LoRA (§4.13) is the bet that a fine-tuning update to a weight matrix has the same shape — a few directions carrying almost everything.');
      }
    },
    quiz: [
      {
        q: 'Which is true of the SVD?',
        options: ['It only exists for symmetric matrices', 'It exists for any matrix and its truncation is the optimal low-rank approximation', 'It requires the matrix to be invertible', 'It produces complex eigenvalues in general'],
        answer: 1,
        why: 'Any real matrix has a real SVD; Eckart–Young says the rank-r truncation minimises Frobenius error among all rank-r matrices.'
      },
      {
        q: 'A kernel matrix that is not positive semi-definite means…',
        options: ['the data is not normalised', 'there is no feature space in which the kernel is an inner product, so the problem is no longer convex', 'the kernel is too smooth', 'the bandwidth γ is too small'],
        answer: 1,
        why: 'Mercer’s condition is exactly the statement that a PSD kernel corresponds to an inner product in some (possibly infinite-dimensional) space.'
      }
    ],
    cards: [
      { q: 'SVD in one sentence', a: 'Rotate · scale · rotate; $A=U\\Sigma V^\\mathsf{T}$, and truncation gives the best low-rank fit (Eckart–Young).' },
      { q: 'PSD definition and consequence', a: '$x^\\mathsf{T}Ax\\ge0$ ⟺ all eigenvalues ≥ 0. Covariance and kernel matrices are PSD, making their optimizations convex.' },
      { q: 'Condition number', a: '$\\kappa=\\sigma_{\\max}/\\sigma_{\\min}$ — how much the map stretches unevenly; it governs both numerical stability and gradient-descent speed.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.9 */
  ML.section({
    id: 'calculus-ml', track: 'foundations', num: '1.9',
    title: 'Calculus for ML: convexity, conditioning, Newton, Lagrange',
    lede: 'Every trainable thing on this site is minimised with the machinery in this section — and most optimisation pain is one number, the condition number.',
    rests: 'Consumed by §2.8 (XGBoost is a Newton step), §3.5 optimisers, §4.11 training at scale.',
    html: `
<h2><span class="sn">1.9.1</span> Gradients, Hessians, and two matrix-calculus facts</h2>
<p>The gradient $\\nabla f$ points uphill; its negation is steepest descent. The Hessian $H$ holds curvature. Two identities recur so often they are worth memorising:</p>
$$\\nabla_x(a^\\mathsf{T}x) = a, \\qquad \\nabla_x(x^\\mathsf{T}Ax) = (A + A^\\mathsf{T})x \\;\\;(= 2Ax \\text{ for symmetric } A)$$
<p>With those you can differentiate the ridge objective in one line (§2.4).</p>

<h2><span class="sn">1.9.2</span> Convexity</h2>
<p>A twice-differentiable $f$ is convex iff $H \\succeq 0$ everywhere. Convex problems have no bad local minima — which is precisely why linear regression, logistic regression and SVMs are considered "solved" and neural networks are not. Gradient descent is $\\theta \\leftarrow \\theta - \\eta\\nabla f$; <b>Newton's method</b> is $\\theta \\leftarrow \\theta - H^{-1}\\nabla f$, which rescales each direction by its curvature and converges in far fewer steps at the cost of a Hessian. <mark>XGBoost's second-order split score (§2.8) is exactly a Newton step in function space</mark> — worth saying out loud in an interview.</p>

<h2><span class="sn">1.9.3</span> Conditioning explains most optimisation pain</h2>
<p>For a quadratic with Hessian eigenvalues $\\lambda_{\\min} \\ldots \\lambda_{\\max}$, the condition number $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ sets both the largest stable step, $\\eta < 2/\\lambda_{\\max}$, and the convergence rate, which degrades like $(\\kappa-1)/(\\kappa+1)$ per step. A ravine — large $\\kappa$ — forces a step size dictated by the steepest direction while progress is needed along the shallowest. Everything that helps is a response to this:</p>
${H.table(['Fix', 'Mechanism'], [
      ['Feature scaling / normalisation', 'Shrinks κ directly by making the curvature more isotropic'],
      ['Momentum', 'Accumulates the consistent shallow direction while cancelling the alternating steep one'],
      ['Adam / RMSProp', 'Per-coordinate scaling approximates dividing by curvature'],
      ['Newton / L-BFGS', 'Removes the problem outright by rescaling with $H^{-1}$ — at a cost nobody pays at scale'],
      ['Batch norm / layer norm', 'Keeps the effective curvature stable across depth (§3.6)']
    ])}

${H.lab('cond', 'One step size, two landscapes', 'Set the condition number and watch gradient descent zig-zag. Then switch to momentum, Adam and Newton on the identical surface — the traces are computed, not drawn.')}

<h2><span class="sn">1.9.4</span> Constrained problems, in one line</h2>
<p>To minimise $f$ subject to $g = 0$, stationarity of the Lagrangian $f - \\lambda g$ says the two gradients must be parallel: you cannot improve $f$ without violating $g$. That single idea derives PCA (§2.10 — maximise variance subject to unit norm) and the SVM dual (§2.6, where the KKT conditions force the multipliers of non-binding constraints to zero, which is <i>why</i> only support vectors carry weight). If you can say "the Lagrange multiplier is the price of the constraint", you have the intuition both derivations need.</p>

${H.probe([
      ['Why is Newton faster in steps but rarely used at scale?', 'It rescales by curvature so κ stops mattering, but forming and inverting $H$ is $O(p^3)$ in the parameter count.'],
      ['What sets the maximum stable learning rate?', '$2/\\lambda_{\\max}$ for a quadratic — the sharpest curvature direction.'],
      ['Where do you meet Lagrange multipliers in ML?', 'PCA, the SVM dual, and any constrained fairness or budget objective.']
    ])}`,
    labs: {
      cond: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 18, fmt: v => v.toFixed(0) },
          { k: 'lr', label: 'learning rate η', min: .005, max: .35, step: .005, value: .08, fmt: v => v.toFixed(3) },
          { k: 'opt', label: 'optimiser', type: 'buttons', value: 'gd', options: [
            { v: 'gd', t: 'GD' }, { v: 'mom', t: 'momentum' }, { v: 'adam', t: 'Adam' }, { v: 'newton', t: 'Newton' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'steps', label: 'steps to converge', cls: 'key' }, { k: 'stable', label: 'max stable η' },
          { k: 'final', label: 'final loss' }, { k: 'rate', label: 'GD rate (κ−1)/(κ+1)' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const a = 1, b = st.kappa;              // f = ½(a x² + b y²)
            const f = (x, y) => .5 * (a * x * x + b * y * y);
            const P = Viz.plot(ctx, w, h, { xd: [-2.6, 2.6], yd: [-1.5, 1.5] })
              .frame({ xlabel: 'shallow direction', ylabel: 'steep direction' });
            P.contours(f, [.05, .2, .5, 1, 2, 4, 8, 16], { color: T.faint, alpha: .55 });
            let x = -2.2, y = 1.2, vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
            const path = [[x, y]];
            let steps = 0;
            for (let t = 1; t <= 220; t++) {
              const gx = a * x, gy = b * y;
              if (st.opt === 'gd') { x -= st.lr * gx; y -= st.lr * gy; }
              else if (st.opt === 'mom') { vx = .9 * vx - st.lr * gx; vy = .9 * vy - st.lr * gy; x += vx; y += vy; }
              else if (st.opt === 'adam') {
                mx = .9 * mx + .1 * gx; my = .9 * my + .1 * gy;
                sx = .999 * sx + .001 * gx * gx; sy = .999 * sy + .001 * gy * gy;
                const mhx = mx / (1 - Math.pow(.9, t)), mhy = my / (1 - Math.pow(.9, t));
                const shx = sx / (1 - Math.pow(.999, t)), shy = sy / (1 - Math.pow(.999, t));
                x -= (st.lr * 4) * mhx / (Math.sqrt(shx) + 1e-8);
                y -= (st.lr * 4) * mhy / (Math.sqrt(shy) + 1e-8);
              } else { x -= gx / a; y -= gy / b; }
              if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 8 || Math.abs(y) > 8) { path.push([Math.sign(x) * 8, Math.sign(y) * 8]); steps = -1; break; }
              path.push([x, y]);
              if (steps === 0 && f(x, y) < 1e-4) steps = t;
            }
            P.clip(() => {
              P.line(path, { color: T.blue, width: 1.8 });
              P.dots(path.filter((_, i) => i % 4 === 0), { r: 2.4, color: T.blue, alpha: .8 });
              P.dots([[0, 0]], { r: 5, color: T.red, stroke: true });
              P.dots([path[0]], { r: 5, color: T.green, stroke: true });
            });
            out({
              steps: steps === -1 ? 'diverged' : (steps || '> 220'),
              stable: (2 / st.kappa).toFixed(3),
              final: f(path[path.length - 1][0], path[path.length - 1][1]).toExponential(1),
              rate: ((st.kappa - 1) / (st.kappa + 1)).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Raise κ with GD selected and watch the zig-zag: the step size is capped by the steep direction while all the progress is needed in the shallow one. Momentum cancels the alternating component; Adam rescales per coordinate; Newton divides by the curvature exactly and lands in one step.');
      }
    },
    quiz: [
      {
        q: 'Gradient descent on a quadratic with κ = 100 is slow. Which explanation is precise?',
        options: ['The loss is non-convex', 'η is capped by 2/λmax while progress is needed along λmin, so the rate degrades like (κ−1)/(κ+1)', 'The gradient is zero', 'The Hessian is singular'],
        answer: 1,
        why: 'That ratio (0.98 for κ=100) is the per-step contraction — ~100× more steps than a well-conditioned problem.'
      },
      {
        q: 'XGBoost’s split gain uses second derivatives because…',
        options: ['it is faster', 'it is a Newton step in function space: curvature gives better leaf values and a loss-agnostic gain formula', 'first derivatives are unavailable', 'it prevents overfitting by itself'],
        answer: 1,
        why: 'Taylor-expanding the loss to second order around the current prediction is precisely a Newton step; §2.8 derives the leaf weight −G/(H+λ) from it.'
      }
    ],
    cards: [
      { q: 'Convexity test', a: 'Twice-differentiable $f$ is convex iff its Hessian is PSD everywhere; then no bad local minima.' },
      { q: 'Condition number’s two consequences', a: 'Max stable step $\\eta<2/\\lambda_{\\max}$ and convergence rate $(\\kappa-1)/(\\kappa+1)$ per step.' },
      { q: 'Lagrange multiplier, in words', a: 'The price of the constraint — at the optimum the gradients of objective and constraint are parallel.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.10 */
  ML.section({
    id: 'information', track: 'foundations', num: '1.10',
    title: 'Information theory: entropy, cross-entropy, KL, perplexity',
    lede: 'Feeds cross-entropy (§1.5, §2.4), tree splits (§2.7), mutual-information feature screening (§2.11), and the KL penalties in RLHF and DPO (§4.12).',
    html: `
<h2><span class="sn">1.10.1</span> Three quantities, one story about coding</h2>
<p><b>Entropy</b> $H(p) = -\\sum p\\log p$ is the average number of bits (or nats) needed to encode draws from $p$ using an optimal code for $p$. <b>Cross-entropy</b> $H(p,q) = -\\sum p\\log q$ is the cost of encoding $p$'s draws using a code built for $q$. <b>KL divergence</b> $D_{KL}(p\\|q) = \\sum p\\log(p/q) \\ge 0$ is the excess — the bits you waste for believing $q$.</p>
<p>KL is not symmetric and not a metric, which is why forward and reverse KL produce <b>mode-covering</b> versus <b>mode-seeking</b> behaviour, and why the direction matters in variational inference.</p>

<h3>The identity to have cold</h3>
$$H(p,q) = H(p) + D_{KL}(p\\|q)$$
<p>Since $H(p)$ does not depend on your parameters, <mark>minimising cross-entropy is minimising KL to the true distribution, up to a constant.</mark></p>

${H.lab('kl', 'Forward vs reverse KL, and why the direction matters', 'A bimodal target p and a single Gaussian q you control. Minimise the forward KL and q spreads to cover both modes; minimise the reverse and it collapses onto one. Press the fit buttons and watch the optimiser find each solution.')}

<h2><span class="sn">1.10.2</span> Mutual information</h2>
<p>$I(X;Y) = H(X) - H(X\\mid Y) = D_{KL}(p(x,y)\\,\\|\\,p(x)p(y))$ measures dependence <i>of any shape</i>, unlike correlation — which is why it catches non-monotonic feature relationships that Pearson misses (§2.11). It is zero exactly under independence.</p>

<h2><span class="sn">1.10.3</span> Perplexity is entropy in disguise</h2>
$$\\mathrm{PPL} = \\exp(H) = \\exp\\!\\left(-\\frac{1}{N}\\sum_i \\log p(x_i)\\right)$$
<p>The exponential of the average negative log-likelihood per token. Read it as an <b>effective branching factor</b>: perplexity 20 means the model is as uncertain as if choosing uniformly among 20 options at each step. Two consequences worth knowing: perplexity is comparable only between models sharing a tokenizer (§4.2), because the denominator changes with the segmentation; and it measures compression, not usefulness — a model can have lower perplexity and be worse at your task, which is why §4.16 insists on task evals.</p>

${H.lab('ppl', 'Entropy, cross-entropy, KL and perplexity on a real distribution', 'Move the predicted distribution away from the truth and watch every quantity respond. The perplexity readout is the same number a language-model training log prints.')}

<h2><span class="sn">1.10.4</span> The log-sum-exp trick</h2>
<p>It appears in every softmax implementation and interviewers like it. Computing $\\log\\sum_i e^{z_i}$ directly overflows the moment any $z_i$ exceeds about 710 in float64 — and logits routinely get large. Subtract the maximum first:</p>
$$\\log\\sum_i e^{z_i} = m + \\log\\sum_i e^{z_i - m}, \\qquad m = \\max_i z_i$$
<p>Every exponent is now $\\le 0$, so nothing overflows and the largest term is exactly 1. The same identity is what makes FlashAttention's <b>online softmax</b> (§4.7) possible — it lets you merge partial sums computed from different tiles by rescaling with the running maximum, which is why the $s \\times s$ matrix never needs to exist.</p>

${H.lab('lse', 'Overflow, and the one-line fix', 'Push the logits up and watch the naive computation return Infinity while the stabilised version keeps returning the right answer. The shift cancels exactly — this is not an approximation.')}

${H.probe([
      ['Cross-entropy vs KL?', '$H(p,q)=H(p)+D_{KL}(p\\|q)$; minimising one minimises the other since $H(p)$ is fixed.'],
      ['Why is KL asymmetric and does it matter?', 'Forward KL is mode-covering (it punishes $q\\approx0$ where $p>0$); reverse KL is mode-seeking. It decides how a variational or distilled model fails.'],
      ['What is perplexity?', 'exp of mean NLL per token — an effective branching factor, comparable only within a fixed tokenizer.']
    ])}`,
    labs: {
      kl: function (host) {
        const st = Viz.controls(host, [
          { k: 'mu', label: 'q mean μ', min: -4, max: 4, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'sd', label: 'q sd σ', min: .2, max: 3.5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'sep', label: 'separation of p’s modes', min: 1, max: 6, step: .1, value: 3.4, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'fwd', label: 'forward KL(p‖q)', cls: 'key' }, { k: 'rev', label: 'reverse KL(q‖p)' },
          { k: 'ce', label: 'cross-entropy H(p,q)' }, { k: 'hp', label: 'entropy H(p)' }
        ]);
        const grid = [];
        for (let x = -9; x <= 9; x += .02) grid.push(x);
        function pdfP(x, sep) { return .5 * Num.normPdf(x, -sep / 2, .8) + .5 * Num.normPdf(x, sep / 2, .8); }
        function kls(mu, sd, sep) {
          let fwd = 0, rev = 0, ce = 0, hp = 0;
          grid.forEach(x => {
            const p = pdfP(x, sep), q = Math.max(1e-12, Num.normPdf(x, mu, sd));
            const dx = .02;
            if (p > 1e-12) { fwd += p * Math.log(p / q) * dx; ce += -p * Math.log(q) * dx; hp += -p * Math.log(p) * dx; }
            rev += q * Math.log(q / Math.max(1e-12, p)) * dx;
          });
          return { fwd, rev, ce, hp };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-7, 7], yd: [0, .42] }).frame({ xlabel: 'x', ylabel: 'density' });
            P.clip(() => {
              const pPts = grid.map(x => [x, pdfP(x, st.sep)]);
              const qPts = grid.map(x => [x, Num.normPdf(x, st.mu, st.sd)]);
              P.area(pPts, { color: T.blue, alpha: .14 });
              P.line(pPts, { color: T.blue, width: 2.4 });
              P.line(qPts, { color: T.red, width: 2.4, dash: [6, 4] });
            });
            const r = kls(st.mu, st.sd, st.sep);
            out({ fwd: r.fwd.toFixed(3), rev: r.rev.toFixed(3), ce: r.ce.toFixed(3), hp: r.hp.toFixed(3) });
          }
        });
        Viz.buttons(host, [
          { label: 'Minimise forward KL(p‖q)', primary: true, on: () => { opt('fwd'); } },
          { label: 'Minimise reverse KL(q‖p)', on: () => { opt('rev'); } }
        ]);
        function opt(which) {
          let bestMu = 0, bestSd = 1, best = 1e9;
          for (let mu = -3; mu <= 3.01; mu += .1) for (let sd = .25; sd <= 3.2; sd += .1) {
            const v = kls(mu, sd, st.sep)[which];
            if (v < best) { best = v; bestMu = mu; bestSd = sd; }
          }
          st.$set('mu', +bestMu.toFixed(2)); st.$set('sd', +bestSd.toFixed(2)); S.redraw();
        }
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'p — the true bimodal distribution' }, { c: Viz.theme().red, t: 'q — your single Gaussian' }]);
        Viz.note(host, 'Forward KL punishes q ≈ 0 wherever p > 0, so it spreads to cover both modes (blurry but safe). Reverse KL punishes q > 0 where p ≈ 0, so it collapses onto one mode (sharp but incomplete). This is the same trade-off that appears in variational inference and in distillation.');
      },

      ppl: function (host) {
        const words = ['the', 'model', 'predicts', 'a', 'token'];
        const st = Viz.controls(host, [
          { k: 'temp', label: 'model sharpness (1/T)', min: .2, max: 5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'shift', label: 'model’s error (mass moved to the wrong token)', min: 0, max: .8, step: .01, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'h', label: 'H(p) bits' }, { k: 'ce', label: 'H(p,q) bits', cls: 'key' },
          { k: 'kl', label: 'KL(p‖q) bits' }, { k: 'ppl', label: 'perplexity' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pTrue = [.5, .2, .15, .1, .05];
            let logits = pTrue.map(v => Math.log(v));
            let q = Num.softmax(logits.map(v => v * st.temp));
            // move mass from the top token to the last one
            const moved = q[0] * st.shift;
            q = q.slice(); q[0] -= moved; q[4] += moved;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, 4.5], yd: [0, .8], pad: { l: 44, r: 14, t: 14, b: 42 } })
              .frame({ xticks: [], ylabel: 'probability' });
            pTrue.forEach((v, i) => {
              const x0 = P.x(i - .34), x1 = P.x(i - .02);
              ctx.fillStyle = T.blue; ctx.fillRect(x0, P.y(v), x1 - x0, P.y(0) - P.y(v));
              const x2 = P.x(i + .02), x3 = P.x(i + .34);
              ctx.fillStyle = T.red; ctx.fillRect(x2, P.y(q[i]), x3 - x2, P.y(0) - P.y(q[i]));
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText('"' + words[i] + '"', P.x(i), P.y(0) + 6);
            });
            const H = Num.entropy(pTrue);
            const CE = -Num.sum(pTrue.map((v, i) => v * Math.log2(Math.max(1e-12, q[i]))));
            out({
              h: H.toFixed(3), ce: CE.toFixed(3), kl: (CE - H).toFixed(3),
              ppl: Math.pow(2, CE).toFixed(2)
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'p — true next-token distribution' }, { c: Viz.theme().red, t: 'q — the model’s prediction' }]);
        Viz.note(host, 'Cross-entropy never drops below the entropy of the truth — that floor is the irreducible uncertainty of language, and it is why a perplexity of 1 is not a target.');
      },

      lse: function (host) {
        const st = Viz.controls(host, [
          { k: 'scale', label: 'logit magnitude', min: 1, max: 1200, step: 1, value: 20, fmt: v => v.toFixed(0) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'naive', label: 'naive Σeᶻ', cls: 'bad' }, { k: 'lse', label: 'stabilised log Σeᶻ', cls: 'good' },
          { k: 'p0', label: 'softmax p₁ (naive)' }, { k: 'p0s', label: 'softmax p₁ (stable)' }
        ]);
        const S = Viz.surface(host, {
          height: 200,
          draw: function (ctx, w, h, T) {
            const z = [1, .6, .2, -.4].map(v => v * st.scale);
            const naiveSum = z.reduce((a, v) => a + Math.exp(v), 0);
            const m = Math.max.apply(null, z);
            const stable = m + Math.log(z.reduce((a, v) => a + Math.exp(v - m), 0));
            const pNaive = Math.exp(z[0]) / naiveSum;
            const pStable = Math.exp(z[0] - stable);
            ctx.font = '13px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
            const lines = [
              ['logits z', '[' + z.map(v => v.toFixed(0)).join(', ') + ']', T.text],
              ['naive   Σ exp(z)', isFinite(naiveSum) ? naiveSum.toExponential(3) : 'Infinity  ← overflow', isFinite(naiveSum) ? T.text : T.red],
              ['stable  m + log Σ exp(z−m)', stable.toFixed(4), T.green],
              ['softmax p₁ naive', isFinite(pNaive) ? pNaive.toFixed(6) : 'NaN  ← the training run dies here', isFinite(pNaive) ? T.text : T.red],
              ['softmax p₁ stable', pStable.toFixed(6), T.green]
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], 12, 18 + i * 30);
              ctx.fillStyle = L[2]; ctx.font = 'bold 13px ui-monospace, monospace';
              ctx.fillText(L[1], Math.min(w - 200, 250), 18 + i * 30);
              ctx.font = '13px ui-monospace, monospace';
            });
            out({
              naive: isFinite(naiveSum) ? naiveSum.toExponential(1) : '∞',
              lse: stable.toFixed(2),
              p0: isFinite(pNaive) ? pNaive.toFixed(4) : 'NaN',
              p0s: pStable.toFixed(4)
            });
          }
        });
        Viz.note(host, 'Push the magnitude past ~710 and float64 gives up. Every framework implements softmax and cross-entropy this way — which is also why you pass <i>logits</i>, not probabilities, to <code>BCEWithLogitsLoss</code> and <code>CrossEntropyLoss</code>.');
      }
    },
    quiz: [
      {
        q: 'Minimising cross-entropy is equivalent to…',
        options: ['maximising entropy of the model', 'minimising KL divergence to the true distribution, up to a constant', 'minimising the L2 norm of the logits', 'maximising mutual information'],
        answer: 1,
        why: 'H(p,q) = H(p) + KL(p‖q), and H(p) does not depend on your parameters.'
      },
      {
        q: 'A model reports perplexity 12 on its own tokenizer and a competitor reports 9 on a different tokenizer. What can you conclude?',
        options: ['The competitor is better', 'Nothing directly — perplexity is only comparable within a fixed tokenizer', 'The competitor has more parameters', 'Both are equally good'],
        answer: 1,
        why: 'The per-token denominator changes with the segmentation. Compare on shared tokenization, or better, compare on task evals.'
      },
      {
        q: 'You fit a single Gaussian to a bimodal target by minimising reverse KL(q‖p). The result will typically…',
        options: ['cover both modes with high variance', 'collapse onto one mode', 'be identical to forward KL', 'fail to converge'],
        answer: 1,
        why: 'Reverse KL punishes putting mass where p is near zero, so q retreats into a single mode — mode-seeking.'
      }
    ],
    cards: [
      { q: 'The cross-entropy identity', a: '$H(p,q)=H(p)+D_{KL}(p\\|q)$ — so minimising cross-entropy minimises KL up to a constant.' },
      { q: 'Forward vs reverse KL', a: 'Forward $D(p\\|q)$ is mode-covering; reverse $D(q\\|p)$ is mode-seeking.' },
      { q: 'Log-sum-exp trick', a: '$\\log\\sum e^{z_i}=m+\\log\\sum e^{z_i-m}$ with $m=\\max z_i$; behind stable softmax and FlashAttention’s online softmax.' },
      { q: 'Perplexity', a: '$\\exp$ of mean NLL per token — an effective branching factor, comparable only within one tokenizer.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.11 */
  ML.section({
    id: 'part1-recall', track: 'foundations', num: '1.16',
    title: 'Rapid recall — Part 1 in twelve lines',
    lede: 'The night-before sheet. If you can produce each line from memory with its derivation sketch, Part 1 is done.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', 'Posterior odds = LR × prior odds.', '<a href="#/bayes">1.1</a>'],
      ['2', 'Low base rate ⇒ a 99%-accurate positive is ~2% right.', '<a href="#/bayes">1.1</a>'],
      ['3', 'Estimation error falls like $1/\\sqrt n$; Hoeffding quantifies the tail.', '<a href="#/concentration">1.4</a>'],
      ['4', 'NLL is the loss; the noise model picks it.', '<a href="#/mle-map">1.5</a>'],
      ['5', 'Gaussian prior → L2; Laplace prior → L1.', '<a href="#/mle-map">1.5</a>'],
      ['6', 'Variances add only under independence — hence bagging.', '<a href="#/expectation">1.3</a>'],
      ['7', 'CI = coverage of the procedure; credible = probability of θ.', '<a href="#/intervals">1.6</a>'],
      ['8', 'Peeking inflates α; use alpha-spending or always-valid p-values.', '<a href="#/intervals">1.6</a>'],
      ['9', 'ATE targets; propensity needs ignorability + overlap; DiD needs parallel trends; IV needs relevance + exclusion.', '<a href="#/causal">1.7</a>'],
      ['10', 'SVD = rotate · scale · rotate; truncation is the best low-rank fit.', '<a href="#/linear-algebra">1.8</a>'],
      ['11', 'Convex ⟺ $H \\succeq 0$; Newton = curvature-rescaled step.', '<a href="#/calculus-ml">1.9</a>'],
      ['12', 'Cross-entropy = KL + constant.', '<a href="#/information">1.10</a>']
    ])}

<h2>The dependency graph, in words</h2>
<p>§1.5 and §1.8 are the load-bearing sections: every model in Parts 2–4 consumes one or both. §1.4 licenses the entire train/test protocol. §1.10 reappears the moment you touch a language model. If revision time is short, revise those four and treat the rest as reference.</p>

${H.lab('drill1', 'Twelve-line drill', 'Shuffled prompts from Part 1. Say the answer out loud before flipping — recall is trained by retrieval failure, not by rereading.')}

<h2>Numbers from Part 1 worth carrying</h2>
${H.table(['Quantity', 'Value'], [
      ['Rows for ±2 points at 95% (0/1 loss)', '≈ 4,600 · ±1 point: ≈ 18,000'],
      ['A/B sample size', '$n \\approx 16p(1-p)/\\delta^2$ per arm'],
      ['Worked A/B example', '4% baseline, +10% relative → 38,400 per arm → 16 days at 5k/day'],
      ['CUPED gain', 'variance × $(1-\\rho^2)$; ρ = 0.6 → 36% reduction'],
      ['20 metrics at α = 0.05', '64% chance of a spurious win'],
      ['Bayes worked example', 'prevalence 0.001, sens 0.99, spec 0.95 → posterior 1.94%']
    ])}`,
    labs: {
      drill1: function (host) {
        const cards = [
          ['State Bayes in odds form.', 'Posterior odds = likelihood ratio × prior odds. In logs, addition.'],
          ['Why is a 99%-accurate test only 2% right at 0.1% prevalence?', 'False positives from the healthy 99.9% swamp the true positives ~50:1.'],
          ['Hoeffding’s bound.', '$P(|\\bar X-\\mu|\\ge t)\\le 2e^{-2nt^2/(b-a)^2}$.'],
          ['Rows needed for ±2 points at 95%?', '≈ 4,600. For ±1 point, ≈ 18,000.'],
          ['Where does cross-entropy come from?', 'The negative log-likelihood of a Bernoulli/Categorical label.'],
          ['L2 corresponds to which prior, with what λ?', 'Zero-mean Gaussian; $\\lambda = 1/(2\\tau^2)$.'],
          ['Bagging variance formula.', '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$.'],
          ['Confidence vs credible interval.', 'Coverage of the procedure vs probability of the parameter given a prior.'],
          ['Why does peeking inflate α?', 'Repeated correlated looks multiply the chance of crossing the boundary.'],
          ['A/B sample size formula.', '$n\\approx 16p(1-p)/\\delta^2$ per arm; $n \\propto 1/\\delta^2$.'],
          ['DiD’s key assumption.', 'Parallel trends absent treatment.'],
          ['What does conditioning on a collider do?', 'Creates bias — it opens a path between the collider’s causes.'],
          ['SVD in one sentence.', 'Rotate · scale · rotate; truncation is the best low-rank approximation (Eckart–Young).'],
          ['Definition of PSD and why it matters.', '$x^\\mathsf{T}Ax\\ge0$; covariance and kernels are PSD, making the optimisation convex.'],
          ['Convexity test and Newton’s step.', '$H\\succeq0$ everywhere; $\\theta\\leftarrow\\theta-H^{-1}\\nabla f$.'],
          ['Condition number’s two consequences.', 'Max stable $\\eta<2/\\lambda_{\\max}$; rate $(\\kappa-1)/(\\kappa+1)$.'],
          ['Cross-entropy in terms of KL.', '$H(p,q)=H(p)+D_{KL}(p\\|q)$.'],
          ['Log-sum-exp trick.', 'Subtract the max: $\\log\\sum e^{z_i}=m+\\log\\sum e^{z_i-m}$.'],
          ['Perplexity, and its one caveat.', 'exp of mean NLL per token; comparable only within one tokenizer.'],
          ['CUPED’s variance reduction.', 'Factor $(1-\\rho^2)$, unbiased, using a pre-period covariate.']
        ];
        let order = cards.map((_, i) => i).sort(() => Math.random() - .5);
        let i = 0, showA = false, right = 0, seen = 0;
        const face = ML.el('div', { class: 'card-face', style: 'cursor:pointer;border:1px solid var(--line);border-radius:12px;background:var(--panel)' });
        host.appendChild(face);
        const pos = ML.el('span');
        function draw() {
          const c = cards[order[i]];
          face.innerHTML = showA ? '<div class="a">' + c[1] + '</div>' : '<div><b>' + c[0] + '</b></div>';
          pos.textContent = (i + 1) + ' / ' + cards.length + (seen ? '  ·  ' + right + '/' + seen + ' recalled' : '');
          ML.typeset(face);
        }
        face.addEventListener('click', () => { showA = !showA; draw(); });
        const nav = ML.el('div', { class: 'cardnav' });
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Flip', onclick: () => { showA = !showA; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'I knew it', onclick: () => { if (showA) { right++; seen++; } i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Missed it', onclick: () => { if (showA) seen++; i = (i + 1) % cards.length; showA = false; draw(); } }));
        nav.appendChild(pos);
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: () => { order = order.sort(() => Math.random() - .5); i = 0; showA = false; right = seen = 0; draw(); } }));
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which pair of Part 1 sections is load-bearing for the largest number of later sections?',
        options: ['1.6 and 1.7', '1.5 and 1.8', '1.2 and 1.3', '1.9 and 1.10'],
        answer: 1,
        why: 'MLE/MAP supplies every loss and every penalty; eigen/SVD supplies PCA, kernels, attention as a bilinear form and low-rank adaptation.'
      }
    ]
  });
})();
