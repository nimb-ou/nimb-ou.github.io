/* ============================================================
   PART 1 — Foundations (1.12 – 1.15): optimisation theory,
   Monte Carlo and the bootstrap, Bayesian inference in practice,
   and floating point.
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 1.12 */
  ML.section({
    id: 'optimization', track: 'foundations', num: '1.12', level: 3,
    title: 'Optimisation: convexity, conditioning, duality, KKT',
    lede: 'Training is optimisation. Almost every training pathology — the loss that will not move, the one that oscillates, the one that needs a schedule — is a statement about curvature, and almost every classical model with a "dual form" is a statement about Lagrange multipliers.',
    prereq: ['calculus-ml'],
    related: ['optimisers', 'svm', 'regularization'],
    html: `
${H.tldr([
      'Convex means every local minimum is global. Logistic regression, ridge, lasso and SVMs are convex; neural networks are not, and it matters less than you would think (§2.2).',
      'The <b>condition number</b> $\\kappa = \\lambda_{\\max}/\\lambda_{\\min}$ of the Hessian sets the speed limit: gradient descent needs $O(\\kappa\\log(1/\\epsilon))$ steps, momentum $O(\\sqrt{\\kappa}\\log(1/\\epsilon))$, Newton $O(\\log\\log(1/\\epsilon))$ but at $O(d^3)$ per step.',
      'Every constrained problem has a dual. <b>Strong duality plus complementary slackness</b> is why the SVM depends only on its support vectors, and why the kernel trick is possible at all (§2.6).'
    ])}

<h2><span class="sn">1.12.1</span> Convexity, and what it buys</h2>
<p>A set is convex if the segment between any two of its points stays inside it. A function is convex if its epigraph — everything above the graph — is a convex set, equivalently if</p>
$$f(\\theta x + (1-\\theta)y) \\le \\theta f(x) + (1-\\theta)f(y)\\quad \\text{for all } \\theta\\in[0,1]$$
<p>and, when $f$ is twice differentiable, if and only if the Hessian $\\nabla^2 f$ is positive semi-definite everywhere. The payoff is a single sentence:</p>
${H.key('For a convex objective, any point with zero gradient is a global minimum, and any local search that goes downhill will find one. There are no bad valleys to get stuck in.')}
${H.table(['Problem', 'Convex?', 'Consequence'], [
      ['Linear/ridge regression', 'yes, strictly (with $\\lambda>0$)', 'unique solution, closed form'],
      ['Lasso', 'yes, not strictly', 'unique fit, possibly non-unique $w$ under collinearity'],
      ['Logistic regression', 'yes', 'unique optimum unless the data are separable, when $\\|w\\|\\to\\infty$ — which is what regularization actually fixes'],
      ['SVM (hinge + L2)', 'yes', 'a quadratic program with a dual; global optimum guaranteed'],
      ['k-means objective', 'no (in the assignments)', 'Lloyd’s algorithm finds a local optimum; restart it (§2.9)'],
      ['Neural networks', 'no', 'many minima — but at scale most are of similar quality, and saddle points, not minima, are the obstacle (§3.5)']
    ])}
${H.more('Why non-convexity turned out not to be fatal', `
<p>The 1990s expectation was that non-convex optimisation would be hopeless. Three things happened instead. First, in high dimensions the critical points you meet are overwhelmingly <b>saddle points</b>, not local minima: a random critical point needs every one of $d$ Hessian eigenvalues to be positive to be a minimum, which is exponentially unlikely, and gradient noise escapes saddles readily. Second, empirically the local minima found by SGD in large networks have similar loss — the landscape is more like a rolling plain than a mountain range with one summit. Third, we stopped caring about the global optimum of the <i>training</i> loss, because that is not the quantity we want minimised anyway (§2.2).</p>
<p>The honest caveat: this is an empirical picture supported by theory in idealised regimes, not a theorem about the networks you actually train.</p>`)}

<h2><span class="sn">1.12.2</span> Conditioning is the speed limit</h2>
<p>Near a minimum, every smooth loss looks like a quadratic $f(w)\\approx \\frac12 (w-w^\\star)^\\top H (w-w^\\star)$. Rotate into the eigenbasis of $H$ and the coordinates decouple: along eigenvector $i$, gradient descent with step $\\eta$ multiplies the error by $(1-\\eta\\lambda_i)$ each iteration. Stability requires $\\eta < 2/\\lambda_{\\max}$; progress along the flattest direction is then at best $1-\\lambda_{\\min}/\\lambda_{\\max} = 1 - 1/\\kappa$ per step.</p>
${H.key('One learning rate must serve every direction at once. That is the entire problem, and $\\kappa$ measures how badly it fails.')}
${H.table(['Method', 'Iterations to accuracy ε', 'Cost per iteration', 'When it is the right answer'], [
      ['Gradient descent', '$O(\\kappa\\log\\frac1\\epsilon)$', '$O(d)$', 'κ is small, or d is enormous'],
      ['+ Momentum / Nesterov', '$O(\\sqrt{\\kappa}\\log\\frac1\\epsilon)$', '$O(d)$', 'almost always — it is free'],
      ['Adam / RMSProp', 'problem-dependent', '$O(d)$', 'badly scaled coordinates, sparse gradients (§3.5)'],
      ['L-BFGS', 'superlinear', '$O(md)$', 'smooth, deterministic, medium $d$ — full-batch classical models'],
      ['Newton', '$O(\\log\\log\\frac1\\epsilon)$', '$O(d^3)$ solve', '$d$ in the hundreds; the gold standard for GLMs (IRLS)']
    ])}
${H.note('This is why feature scaling is not cosmetic. Standardising inputs shrinks $\\kappa$ directly; batch normalisation (§3.6) is the same idea applied to every hidden layer, and Adam approximates a diagonal preconditioner that undoes per-coordinate scale.')}

${H.lab('convrace', 'The convergence race — watch the condition number bite', 'A genuine quadratic bowl whose curvature ratio you control. Every optimiser here runs its real update rule; the trajectories are what they actually do. Push κ to 50 and gradient descent zig-zags across the valley while momentum sails down it.')}

${H.deriv('why momentum earns the square root', [
      ['$w_{t+1} = w_t - \\eta\\nabla f(w_t)$', 'Plain gradient descent. In the eigenbasis, coordinate $i$ obeys $e_{t+1} = (1-\\eta\\lambda_i)e_t$.'],
      ['$\\text{rate} = \\max_i|1-\\eta\\lambda_i|$', 'The slowest coordinate sets the pace. Minimising the max over $i$ gives $\\eta^\\star = 2/(\\lambda_{\\min}+\\lambda_{\\max})$…'],
      ['$\\text{rate}^\\star = \\dfrac{\\kappa-1}{\\kappa+1} \\approx 1-\\dfrac{2}{\\kappa}$', '…and even at the optimal step, progress per iteration is $O(1/\\kappa)$. This is a hard limit for any method that only looks at the current gradient.'],
      ['$v_{t+1} = \\beta v_t + \\nabla f(w_t),\\; w_{t+1}=w_t-\\eta v_{t+1}$', 'Momentum adds state. The per-coordinate recursion becomes second order — a damped oscillator rather than a leaky bucket.'],
      ['$\\text{rate} = \\dfrac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}$', 'With $\\beta$ tuned to $\\left(\\frac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}\\right)^2$, the two roots of that recursion have equal modulus $\\sqrt{\\beta}$ and the rate improves to $O(1/\\sqrt\\kappa)$. At $\\kappa=10^4$ that is 100 iterations instead of 10,000.']
    ], 'Nesterov’s method achieves this rate with a provable lower-bound match — no first-order method can do better on this problem class. That is the real content of "accelerated gradient": not a heuristic that helps, an algorithm that is optimal.')}

<h2><span class="sn">1.12.3</span> Constraints: Lagrange, duality, KKT</h2>
<p>Minimise $f(x)$ subject to $g_i(x)\\le 0$ and $h_j(x)=0$. Fold the constraints into the objective with multipliers:</p>
$$\\mathcal{L}(x,\\lambda,\\nu) = f(x) + \\sum_i \\lambda_i g_i(x) + \\sum_j \\nu_j h_j(x), \\qquad \\lambda_i \\ge 0$$
<p>The <b>dual function</b> $d(\\lambda,\\nu) = \\inf_x \\mathcal{L}$ is concave <i>whatever $f$ and $g$ are</i>, and always lower-bounds the primal optimum (weak duality). When the problem is convex and mildly regular (Slater's condition: some strictly feasible point exists), the bound is tight — <b>strong duality</b> — and the solution satisfies the KKT conditions:</p>
${H.table(['Condition', 'Statement', 'What it means'], [
      ['Stationarity', '$\\nabla f + \\sum\\lambda_i\\nabla g_i + \\sum\\nu_j\\nabla h_j = 0$', 'the objective gradient is cancelled by the active constraints pushing back'],
      ['Primal feasibility', '$g_i(x)\\le 0,\\; h_j(x)=0$', 'the answer is allowed'],
      ['Dual feasibility', '$\\lambda_i \\ge 0$', 'inequality constraints can only push one way'],
      ['<b>Complementary slackness</b>', '$\\lambda_i\\, g_i(x) = 0$', '<b>either the constraint is tight, or its multiplier is zero</b>']
    ])}
${H.key('Complementary slackness is the money condition. It says inactive constraints have zero multiplier — which is exactly why an SVM’s solution depends only on the points touching the margin, and why $\\alpha_i > 0$ identifies the support vectors (§2.6).')}

${H.lab('kkt', 'Lagrange multipliers, made visible', 'Drag the constraint. When it is slack the unconstrained optimum survives and λ = 0; the moment it bites, the solution slides along the boundary and the two gradients become anti-parallel. That anti-parallelism <i>is</i> the stationarity condition.')}

${H.intuition(`<p>The geometric reading of stationarity: at the constrained optimum, the objective's gradient must point straight <i>into</i> the constraint wall. If it had any component along the wall, you could slide that way and improve while staying feasible. "Gradients are parallel" is not a coincidence of the algebra — it is the statement that there is no free direction left.</p>`)}

${H.more('Duality in the three places you will actually meet it', `
<ul>
<li><b>SVM (§2.6).</b> The dual is a quadratic program in $\\alpha$ with only inner products $x_i^\\top x_j$ in it. Replace those with $k(x_i,x_j)$ and you have the kernel trick — impossible in the primal, because the primal contains $w$, which may live in an infinite-dimensional space.</li>
<li><b>Ridge and lasso (§2.3).</b> The penalty form $\\min \\|Xw-y\\|^2 + \\lambda\\|w\\|$ and the constraint form $\\min\\|Xw-y\\|^2$ s.t. $\\|w\\|\\le t$ are Lagrangian duals of each other. Every $\\lambda$ corresponds to some $t$. The famous diamond-vs-circle picture is the constraint form; the thing you implement is the penalty form.</li>
<li><b>Constrained RL and safe fine-tuning (§4.12).</b> "Maximise reward subject to KL $\\le \\delta$ from the reference policy" is solved by dualising into "maximise reward $-\\beta\\,$KL". That is precisely where the $\\beta$ in PPO's penalty and in DPO's objective comes from — it is a Lagrange multiplier, and tuning it is choosing where on the trade-off curve to sit.</li>
</ul>`)}

${H.probe([
      ['What does convexity guarantee, and what does it not?', 'Guarantees: every stationary point is a global minimum, and duality gaps vanish under Slater. Does not guarantee: a unique minimiser (needs strict convexity), fast convergence (that is conditioning), or that the model is any good.'],
      ['Why is momentum $O(\\sqrt\\kappa)$ and plain GD $O(\\kappa)$?', 'GD’s per-coordinate error contracts by $1-\\eta\\lambda_i$ and one $\\eta$ must serve all $\\lambda_i$; momentum makes the recursion second order so both roots can be balanced at modulus $\\sqrt\\beta$.'],
      ['State complementary slackness and give a use.', '$\\lambda_i g_i(x^\\star)=0$: a constraint is either active or has zero multiplier. It identifies SVM support vectors and prunes inactive constraints from the solution.'],
      ['When would you use Newton’s method in 2026?', 'Small-to-medium $d$ with a smooth deterministic objective: GLM fitting via IRLS, calibration fits, hyperparameter inner loops. Never for a large network — the $d\\times d$ solve and the stochastic gradients both kill it.']
    ], 'Saying "neural networks are non-convex so we cannot say anything". You can say a great deal: about conditioning, about saddle points, about why normalisation and Adam help. Non-convexity is not an excuse to stop reasoning.')}`,
    labs: {
      convrace: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 20, fmt: v => v + '×' },
          { k: 'lr', label: 'learning rate (× the stable max)', min: .05, max: 1.0, step: .01, value: .9, fmt: v => v.toFixed(2) },
          { k: 'beta', label: 'momentum β', min: 0, max: .98, step: .01, value: .9, fmt: v => v.toFixed(2) },
          { k: 'show', label: 'view', type: 'select', value: 'both', options: [{ v: 'both', t: 'trajectory + convergence' }, { v: 'traj', t: 'trajectory only' }, { v: 'conv', t: 'convergence only' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'gd', label: 'GD steps to 1e-4', cls: 'bad' },
          { k: 'mom', label: 'momentum steps', cls: 'good' },
          { k: 'nt', label: 'Newton steps', cls: 'key' },
          { k: 'pred', label: 'theory: κ vs √κ' }
        ]);

        function run(method) {
          const k = st.kappa;
          const lmax = k, lmin = 1;
          const eta = st.lr * 2 / lmax;
          let w = [1, 1], v = [0, 0];
          const path = [w.slice()], errs = [];
          const f = p => .5 * (lmin * p[0] * p[0] + lmax * p[1] * p[1]);
          for (let t = 0; t < 400; t++) {
            const g = [lmin * w[0], lmax * w[1]];
            if (method === 'gd') w = [w[0] - eta * g[0], w[1] - eta * g[1]];
            else if (method === 'mom') {
              v = [st.beta * v[0] + g[0], st.beta * v[1] + g[1]];
              w = [w[0] - eta * v[0], w[1] - eta * v[1]];
            } else { // Newton: H⁻¹g exactly solves a quadratic in one step
              w = [w[0] - g[0] / lmin, w[1] - g[1] / lmax];
            }
            if (!isFinite(w[0]) || !isFinite(w[1]) || Math.abs(w[0]) > 1e6) break;
            path.push(w.slice());
            errs.push(Math.max(1e-16, f(w)));
          }
          let hit = errs.findIndex(e => e < 1e-4);
          return { path: path, errs: errs, hit: hit < 0 ? null : hit + 1 };
        }

        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const gd = run('gd'), mo = run('mom'), nt = run('newton');
            const k = st.kappa;
            const both = st.show === 'both';
            const w1 = both ? w * .52 : w, x0 = 0;

            if (st.show !== 'conv') {
              const P = Viz.plot(ctx, w1, h, { xd: [-1.35, 1.35], yd: [-1.2, 1.2], pad: { l: 40, r: 10, t: 16, b: 36 } })
                .frame({ xlabel: 'flat direction (λ = 1)', ylabel: 'steep (λ = κ)' });
              P.clip(() => {
                P.contours((x, y) => .5 * (x * x + k * y * y), [.05, .2, .5, 1, 2], { color: T.faint, alpha: .5 });
                P.line(gd.path, { color: T.c2, width: 1.6 });
                P.dots(gd.path.slice(0, 40), { r: 2, color: T.c2, alpha: .9 });
                P.line(mo.path, { color: T.c3, width: 1.8 });
                P.dots(mo.path.slice(0, 40), { r: 2, color: T.c3, alpha: .9 });
                P.dots(nt.path.slice(0, 3), { r: 4.5, color: T.c1, stroke: true });
                P.dots([[0, 0]], { r: 4, color: T.text });
              });
            }
            if (st.show !== 'traj') {
              const ox = both ? w * .52 : 0, ww = both ? w * .48 : w;
              ctx.save(); ctx.translate(ox, 0);
              const P2 = Viz.plot(ctx, ww, h, {
                xd: [0, 120],
                yd: [-10, 1],
                pad: { l: 44, r: 12, t: 16, b: 36 }
              }).frame({ xlabel: 'iteration', ylabel: 'log₁₀ loss' });
              P2.clip(() => {
                P2.line(gd.errs.map((e, i) => [i, Math.log10(e)]), { color: T.c2, width: 2 });
                P2.line(mo.errs.map((e, i) => [i, Math.log10(e)]), { color: T.c3, width: 2 });
                P2.line(nt.errs.map((e, i) => [i, Math.log10(Math.max(1e-10, e))]), { color: T.c1, width: 2 });
                P2.hline(-4, { color: T.faint, dash: [4, 4], label: '1e-4' });
              });
              ctx.restore();
            }
            out({
              gd: gd.hit == null ? '>400 (or diverged)' : gd.hit,
              mom: mo.hit == null ? '>400' : mo.hit,
              nt: nt.hit == null ? '—' : nt.hit,
              pred: Math.round(k) + ' vs ' + Math.round(Math.sqrt(k))
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c2)', t: 'gradient descent' }, { c: 'var(--c3)', t: 'momentum' }, { c: 'var(--c1)', t: 'Newton' }
        ]);
        Viz.note(host, 'Newton lands in <b>one</b> step, because a quadratic is exactly what Newton’s method assumes — the price is forming and inverting the Hessian, which is $O(d^3)$ and hopeless at $d=10^9$. Now push the learning rate past 1.00 of the stable maximum and gradient descent diverges along the steep direction while still crawling along the flat one: the two failure modes that make one global learning rate so hard to choose.');
      },

      kkt: function (host) {
        const st = Viz.controls(host, [
          { k: 'c', label: 'constraint: x + y ≤ c', min: -1, max: 4, step: .05, value: 1.2, fmt: v => v.toFixed(2) },
          { k: 'cx', label: 'unconstrained optimum x*', min: -1, max: 3, step: .05, value: 2, fmt: v => v.toFixed(2) },
          { k: 'cy', label: 'unconstrained optimum y*', min: -1, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'active', label: 'constraint', cls: 'key' },
          { k: 'lam', label: 'multiplier λ' },
          { k: 'sol', label: 'solution' },
          { k: 'slack', label: 'λ · g(x*)' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const a = st.cx, b = st.cy, c = st.c;
            const f = (x, y) => (x - a) * (x - a) + (y - b) * (y - b);
            // projection onto x+y=c when the constraint binds
            const viol = a + b - c;
            const active = viol > 0;
            const sx = active ? a - viol / 2 : a, sy = active ? b - viol / 2 : b;
            const lam = active ? viol : 0;   // ∇f = -λ∇g ⇒ 2(x-a) = -λ, with x-a = -viol/2

            const P = Viz.plot(ctx, w, h, { xd: [-1.5, 4], yd: [-1.5, 3.6], pad: { l: 42, r: 14, t: 16, b: 36 } })
              .frame({ xlabel: 'x', ylabel: 'y' });
            P.clip(() => {
              // infeasible half-plane
              P.field((x, y) => (x + y > c ? 1 : 0), {
                step: 4, lo: 0, hi: 1,
                colors: t => t > .5 ? [180, 60, 50, 26] : [0, 0, 0, 0]
              });
              P.contours(f, [.15, .6, 1.5, 3, 5, 8], { color: T.faint, alpha: .55 });
              P.line([[-2, c + 2], [5, c - 5]], { color: T.c2, width: 2 });
              P.dots([[a, b]], { r: 4.5, color: T.faint, stroke: true });
              P.text(a, b, ' unconstrained', { dx: 8, dy: -10, color: T.faint, font: '10.5px ui-sans-serif' });
              P.dots([[sx, sy]], { r: 6, color: T.c1, stroke: true, strokeWidth: 2 });
              P.text(sx, sy, ' solution', { dx: 9, dy: 12, color: T.c1, font: '11px ui-sans-serif' });
              if (active) {
                // ∇f at the solution, and λ∇g pointing back
                const gfx = 2 * (sx - a), gfy = 2 * (sy - b);
                const sc = .6;
                P.arrow(sx, sy, sx + gfx * sc, sy + gfy * sc, { color: T.c3, width: 2 });
                P.text(sx + gfx * sc, sy + gfy * sc, ' −∇f', { dx: 6, color: T.c3, font: '11px ui-sans-serif' });
                P.arrow(sx, sy, sx + lam * sc, sy + lam * sc, { color: T.c2, width: 2 });
                P.text(sx + lam * sc, sy + lam * sc, ' λ∇g', { dx: 6, dy: -8, color: T.c2, font: '11px ui-sans-serif' });
              }
            });
            out({
              active: active ? 'ACTIVE (tight)' : 'inactive (slack)',
              lam: lam.toFixed(3),
              sol: '(' + sx.toFixed(2) + ', ' + sy.toFixed(2) + ')',
              slack: (lam * (sx + sy - c)).toFixed(6)
            });
          }
        });
        Viz.note(host, 'Watch the last readout: <b>λ · g(x*) is zero in every configuration</b>. When the constraint is slack, λ = 0; when it binds, g = 0. That product being identically zero <i>is</i> complementary slackness, and it is the reason a fitted SVM can throw away every training point except the handful on the margin.');
      }
    },
    quiz: [
      {
        q: 'The condition number of the Hessian is 10,000. Roughly how much does well-tuned momentum help versus plain gradient descent?',
        options: ['about 2×', 'about 100×', 'about 10,000×', 'not at all'],
        answer: 1,
        why: '$O(\\kappa) \\to O(\\sqrt\\kappa)$: 10,000 → 100 iterations, a 100× reduction.'
      },
      {
        q: 'Complementary slackness says…',
        options: ['all constraints are active at the optimum', 'either a constraint is tight or its multiplier is zero', 'the dual equals the primal', 'the gradient is zero'],
        answer: 1,
        why: '$\\lambda_i g_i(x^\\star)=0$. It is what makes SVM solutions sparse in the training points.'
      },
      {
        q: 'Logistic regression on perfectly separable data has…',
        options: ['no solution because the loss is non-convex', 'a convex loss whose minimiser runs off to infinite weights', 'a unique bounded optimum', 'multiple disconnected minima'],
        answer: 1,
        why: 'Convex but not coercive: the likelihood keeps improving as $\\|w\\|\\to\\infty$. Any regularization restores a finite optimum — which is the practical reason for the default penalty in sklearn.'
      },
      {
        q: 'Why is Newton’s method impractical for deep networks?',
        options: ['it needs convexity', 'the $d\\times d$ Hessian solve is $O(d^3)$ and gradients are stochastic', 'it converges too slowly', 'it cannot handle constraints'],
        answer: 1,
        why: 'Both problems are fatal at scale. Quasi-Newton and diagonal preconditioners (Adam) are the practical compromises.'
      }
    ],
    cards: [
      { q: 'Convexity, in one sentence', a: 'Every local minimum is global, and the Hessian is PSD everywhere. It does not imply uniqueness or speed.' },
      { q: 'Convergence rates by κ', a: 'GD $O(\\kappa)$, momentum/Nesterov $O(\\sqrt\\kappa)$, Newton $O(\\log\\log)$ at $O(d^3)$ per step.' },
      { q: 'The four KKT conditions', a: 'Stationarity, primal feasibility, dual feasibility ($\\lambda\\ge0$), complementary slackness ($\\lambda_i g_i = 0$).' },
      { q: 'Why the SVM has a kernel trick and the primal does not', a: 'The dual contains only inner products $x_i^\\top x_j$, which can be replaced by $k(x_i,x_j)$; the primal contains $w$ itself.' },
      { q: 'Where the β in PPO/DPO comes from', a: 'A Lagrange multiplier: "maximise reward s.t. KL ≤ δ" dualised into "maximise reward − β·KL".' }
    ]
  });

  /* ------------------------------------------------------------------ 1.13 */
  ML.section({
    id: 'sampling', track: 'foundations', num: '1.13', level: 2,
    title: 'Monte Carlo, the bootstrap, and simulation as proof',
    lede: 'When the integral is hard, sample. When the sampling distribution is unknown, resample. Between them these two ideas replace most of the analytic statistics you were told you needed — and they are how you check the analytic answer when you do have one.',
    prereq: ['concentration'],
    related: ['bayesian-inference', 'intervals', 'experimentation'],
    html: `
${H.tldr([
      'Monte Carlo error falls as $\\sigma/\\sqrt{n}$ — <b>independent of dimension</b>. That is why sampling beats quadrature in high dimensions, and why 100× more samples buys only 10× more accuracy.',
      'The bootstrap replaces "what is the sampling distribution of my statistic?" with "resample the data and look". It works for medians, AUCs, ratios and pipeline outputs where no formula exists.',
      'Simulate before you derive. If your closed-form interval and a simulation disagree, the simulation is usually right — and always cheaper to trust.'
    ])}

<h2><span class="sn">1.13.1</span> Monte Carlo integration</h2>
<p>To compute $I = \\mathbb{E}_{x\\sim p}[f(x)] = \\int f(x)p(x)\\,dx$, draw $x_1,\\dots,x_n \\sim p$ and average:</p>
$$\\hat I_n = \\frac1n\\sum_{i=1}^n f(x_i), \\qquad \\mathbb{E}[\\hat I_n] = I, \\qquad \\mathrm{sd}(\\hat I_n) = \\frac{\\sigma_f}{\\sqrt n}$$
<p>Two properties, both consequential. It is <b>unbiased</b> for any $n$. And the error rate $n^{-1/2}$ contains no $d$ — a grid over $d$ dimensions needs $m^d$ points for the same resolution, so sampling wins outright past $d\\approx 4$.</p>
${H.key('$\\sqrt{n}$ is the tax on everything: every extra digit of accuracy costs a hundredfold increase in samples. Variance reduction is therefore not an optimisation, it is the only lever you have.')}

${H.lab('mc', 'Monte Carlo convergence, and the three ways to cheat it', 'Estimate the same integral four ways: plain sampling, antithetic pairs, stratification and importance sampling. All four are unbiased; the error bands are wildly different. The dashed line is the theoretical $\\sigma/\\sqrt n$.')}

${H.table(['Variance reduction', 'Idea', 'Typical gain', 'Where you meet it'], [
      ['Antithetic variates', 'pair each $u$ with $1-u$; the errors cancel if $f$ is monotone', '2–10×', 'pricing, simulation studies'],
      ['Stratification', 'force the samples to cover the space evenly', '2–20×', 'stratified train/test splits are the same idea (§2.14)'],
      ['Importance sampling', 'sample from $q$, reweight by $p/q$; concentrate effort where $f$ is large', '10–1000×, or catastrophic', 'off-policy RL (§6.1), rare-event estimation'],
      ['Control variates', 'subtract a correlated quantity whose mean you know', '2–50×', 'CUPED in A/B testing is exactly this (§2.25)'],
      ['Common random numbers', 'reuse the same seed across arms', 'large', 'comparing two policies or two model versions']
    ])}
${H.pitfall('Importance sampling has an infinite-variance failure mode: if $q$ has lighter tails than $p\\,|f|$, a single sample with an enormous weight dominates the estimate and the sample standard error <i>understates</i> the true error without bound. Always report the effective sample size $\\text{ESS} = (\\sum w_i)^2/\\sum w_i^2$; if it has collapsed to a handful, your estimate is one number wearing a crowd’s clothing.')}

<h2><span class="sn">1.13.2</span> The bootstrap</h2>
<p>You have one sample of size $n$ and a statistic $T$ — a median, an AUC, an F1, the output of a fitted pipeline. You want its sampling distribution and there is no formula. Efron's answer: <b>the empirical distribution is the best estimate of the population, so resample from it.</b></p>
${H.steps([
      'Draw $n$ observations from your data <b>with replacement</b>. This is one bootstrap sample; about 63.2% of the original rows appear at least once ($1-e^{-1}$), which is also where the out-of-bag estimate in random forests comes from (§2.7).',
      'Compute $T^{*}$ on it.',
      'Repeat $B$ times, typically 1,000–10,000. The spread of $\\{T^*_b\\}$ estimates the sampling variability of $T$; the 2.5th and 97.5th percentiles give a 95% interval.'
    ])}
${H.lab('boot', 'Bootstrap a statistic that has no formula', 'The median, the trimmed mean, the 90th percentile, the correlation, the AUC. The histogram is the bootstrap distribution; the shaded band is the percentile interval; the dashed line is the Gaussian interval you would have written down if the statistic had been a mean.')}

${H.table(['Bootstrap flavour', 'When', 'Caveat'], [
      ['Percentile', 'default; simple and adequate', 'slightly under-covers for skewed statistics'],
      ['BCa (bias-corrected, accelerated)', 'skewed statistics, small $n$', 'more computation, better coverage'],
      ['Block bootstrap', '<b>time series</b> — resample contiguous blocks', 'block length must exceed the autocorrelation range'],
      ['Cluster bootstrap', 'grouped data (users, hospitals, sessions)', 'resample <b>clusters</b>, not rows, or you understate variance badly'],
      ['Bayesian bootstrap', 'want a posterior-flavoured version', 'Dirichlet weights instead of counts']
    ])}
${H.pitfall('The bootstrap fails for extremes. The maximum of a sample cannot exceed the maximum you observed, so bootstrapping a max, a min, or a very high quantile with small $n$ produces a degenerate, over-confident distribution. It also fails silently when observations are dependent and you resample rows — the single most common bootstrap error in industry, and the reason per-user metrics must be bootstrapped by user.')}

<h2><span class="sn">1.13.3</span> Simulate first</h2>
${H.intuition(`<p>Any statistical claim you are about to make can be tested in twenty lines: generate data where you <i>know</i> the truth, run your procedure a thousand times, and count how often it is right. Does your 95% interval cover 95% of the time? Does your p-value come out uniform under the null? Does your early-stopping rule inflate the false-positive rate (§2.25 says it does)? This is not a substitute for theory; it is how you find out that you applied the theory to the wrong problem.</p>`)}
${H.code(`# Does my confidence interval actually cover 95% of the time?
import numpy as np
rng, cover = np.random.default_rng(0), 0
for _ in range(2000):
    x  = rng.exponential(scale=2.0, size=30)       # skewed, not normal
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(30)
    lo, hi = m - 1.96*se, m + 1.96*se              # the interval we teach
    cover += (lo <= 2.0 <= hi)
print(cover / 2000)     # ≈ 0.91, not 0.95 — the CLT has not arrived at n=30`)}
${H.flag('That 0.91 is the honest answer for skewed data at $n=30$, and it is why the bootstrap or a log transform is worth the trouble. The textbook "$n>30$ and the CLT applies" rule is a rule of thumb about symmetric distributions, not a theorem.')}

${H.probe([
      ['Why does Monte Carlo beat numerical integration in high dimensions?', 'Its error is $O(n^{-1/2})$ regardless of $d$; a product grid needs $m^d$ points. Past about four dimensions there is no contest.'],
      ['You bootstrap a click-through rate from 100,000 rows and get an implausibly tight interval. What is wrong?', 'Almost certainly the rows are not independent — many rows per user. Bootstrap at the user level; the effective sample size is the number of users, not impressions.'],
      ['How many bootstrap replicates?', '1,000 for a standard error, 10,000 for a percentile interval you intend to quote. The Monte Carlo error of the interval itself falls as $1/\\sqrt B$ and is free to shrink.'],
      ['What is the 0.632 in the .632 bootstrap?', 'The probability a given row appears in a bootstrap sample: $1-(1-1/n)^n \\to 1-e^{-1}=0.632$. The complement is the out-of-bag set.']
    ])}`,
    labs: {
      mc: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'samples (log₁₀)', min: 1, max: 4, step: .1, value: 3, fmt: v => Math.round(Math.pow(10, v)).toLocaleString() },
          { k: 'reps', label: 'independent repeats', min: 5, max: 60, step: 1, value: 25, fmt: v => v },
          { k: 'seed', label: 'seed', min: 1, max: 40, step: 1, value: 7, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true value', cls: 'key' },
          { k: 'plain', label: 'plain MC error' },
          { k: 'anti', label: 'antithetic' },
          { k: 'strat', label: 'stratified' },
          { k: 'imp', label: 'importance' }
        ]);
        /* I = ∫₀¹ e^x dx = e − 1, with an importance density q ∝ 1 + x */
        const TRUTH = Math.E - 1;
        const f = x => Math.exp(x);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const N = Math.max(4, Math.round(Math.pow(10, st.n)));
            const R = Num.rng(st.seed);
            const errs = { plain: [], anti: [], strat: [], imp: [] };
            for (let r = 0; r < st.reps; r++) {
              let sp = 0, sa = 0, ss = 0, si = 0;
              const half = Math.max(1, Math.floor(N / 2));
              for (let i = 0; i < N; i++) sp += f(R());
              for (let i = 0; i < half; i++) { const u = R(); sa += f(u) + f(1 - u); }
              for (let i = 0; i < N; i++) { const u = (i + R()) / N; ss += f(u); }
              for (let i = 0; i < N; i++) {
                // q(x) = (1+x)/1.5 on [0,1]; inverse-CDF sample
                const u = R();
                const x = Math.sqrt(1 + 3 * u) - 1;
                const q = (1 + x) / 1.5;
                si += f(x) / q;
              }
              errs.plain.push(Math.abs(sp / N - TRUTH));
              errs.anti.push(Math.abs(sa / (2 * half) - TRUTH));
              errs.strat.push(Math.abs(ss / N - TRUTH));
              errs.imp.push(Math.abs(si / N - TRUTH));
            }
            const med = a => Num.quantile(a.slice().sort((x, y) => x - y), .5);
            const P = Viz.plot(ctx, w, h, { xd: [-.6, 3.6], yd: [-7, -0.4], pad: { l: 50, r: 14, t: 16, b: 40 } })
              .frame({ xticks: [0, 1, 2, 3], xfmt: v => '10^' + v, xlabel: 'absolute error (repeats jittered)', ylabel: 'log₁₀ |error|' });
            const cols = [T.c1, T.c3, T.c4, T.c5];
            const keys = ['plain', 'anti', 'strat', 'imp'];
            P.clip(() => {
              keys.forEach((k, j) => {
                errs[k].forEach((e, i) => {
                  P.dots([[j + (i / st.reps - .5) * .55, Math.log10(Math.max(1e-9, e))]], { r: 2.6, color: cols[j], alpha: .75 });
                });
                const m = Math.log10(Math.max(1e-9, med(errs[k])));
                ctx.strokeStyle = cols[j]; ctx.lineWidth = 2.4;
                ctx.beginPath(); ctx.moveTo(P.x(j - .32), P.y(m)); ctx.lineTo(P.x(j + .32), P.y(m)); ctx.stroke();
              });
              const sig = Math.sqrt(Num.variance(Array.from({ length: 400 }, (_, i) => f((i + .5) / 400))));
              P.hline(Math.log10(sig / Math.sqrt(N)), { color: T.faint, dash: [5, 4], label: 'σ/√n' });
            });
            ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif';
            ctx.textAlign = 'center';
            ['plain', 'antithetic', 'stratified', 'importance'].forEach((lbl, j) => ctx.fillText(lbl, P.x(j), h - 22));
            out({
              truth: TRUTH.toFixed(6),
              plain: med(errs.plain).toExponential(2),
              anti: med(errs.anti).toExponential(2),
              strat: med(errs.strat).toExponential(2),
              imp: med(errs.imp).toExponential(2)
            });
          }
        });
        Viz.note(host, 'Stratification wins here by a wide margin because the integrand is smooth and one-dimensional — its error falls like $n^{-1}$ or better rather than $n^{-1/2}$. That advantage evaporates in high dimensions, which is precisely why plain Monte Carlo survives: it is the only one of the four whose rate does not care how many dimensions you have.');
      },

      boot: function (host) {
        const st = Viz.controls(host, [
          { k: 'stat', label: 'statistic', type: 'select', value: 'median', options: [
            { v: 'median', t: 'median' }, { v: 'mean', t: 'mean' }, { v: 'p90', t: '90th percentile' },
            { v: 'trim', t: '10% trimmed mean' }, { v: 'max', t: 'maximum (watch it fail)' }
          ] },
          { k: 'n', label: 'sample size n', min: 15, max: 400, step: 5, value: 60, fmt: v => v },
          { k: 'B', label: 'bootstrap replicates B', min: 100, max: 3000, step: 100, value: 1000, fmt: v => v.toLocaleString() },
          { k: 'skew', label: 'population skew', min: 0, max: 1, step: .05, value: .8, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'est', label: 'point estimate', cls: 'key' },
          { k: 'ci', label: '95% percentile CI' },
          { k: 'se', label: 'bootstrap SE' },
          { k: 'gauss', label: 'naive ±1.96·SE' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(11);
            const data = Array.from({ length: st.n }, () =>
              st.skew > 0 ? R.gamma(1 + 6 * (1 - st.skew)) * (2 + 6 * st.skew) : R.normal(10, 3));
            const stats = {
              median: a => Num.quantile(a.slice().sort((x, y) => x - y), .5),
              mean: a => Num.mean(a),
              p90: a => Num.quantile(a.slice().sort((x, y) => x - y), .9),
              trim: a => { const s = a.slice().sort((x, y) => x - y); const k = Math.floor(a.length * .1); return Num.mean(s.slice(k, s.length - k)); },
              max: a => Math.max.apply(null, a)
            };
            const fn = stats[st.stat];
            const bs = Num.bootstrap(data, fn, st.B, 5);
            const point = fn(data);
            const hist = Num.hist(bs.samples, 34);
            const P = Viz.plot(ctx, w, h, {
              xd: [hist.lo, hist.hi],
              yd: [0, Math.max.apply(null, hist.bins) * 1.18],
              pad: { l: 46, r: 14, t: 16, b: 40 }
            }).frame({ xlabel: 'value of the statistic across ' + st.B.toLocaleString() + ' resamples', ylabel: 'count' });
            P.clip(() => {
              P.bars(hist.bins, {
                gap: .1,
                color: (v, i) => (hist.centers[i] >= bs.lo && hist.centers[i] <= bs.hi) ? T.c1 : T.line
              });
              P.vline(point, { color: T.c2, width: 2, dash: false, label: 'estimate' });
              P.vline(bs.lo, { color: T.c1, dash: [4, 3] });
              P.vline(bs.hi, { color: T.c1, dash: [4, 3] });
            });
            out({
              est: point.toFixed(3),
              ci: '[' + bs.lo.toFixed(2) + ', ' + bs.hi.toFixed(2) + ']',
              se: bs.se.toFixed(3),
              gauss: '[' + (point - 1.96 * bs.se).toFixed(2) + ', ' + (point + 1.96 * bs.se).toFixed(2) + ']'
            });
          }
        });
        Viz.note(host, 'Choose <b>maximum</b>: the bootstrap distribution piles up on a few discrete values and the upper tail is truncated at the observed max — the distribution is wrong and the interval is nonsense. That is the bootstrap’s known failure mode for extremes. Then compare the percentile interval with the naive Gaussian one on a skewed statistic: the percentile interval is asymmetric because the sampling distribution is, and the symmetric one silently mis-covers.');
      }
    },
    quiz: [
      {
        q: 'Monte Carlo error scales as…',
        options: ['$1/n$', '$1/\\sqrt n$, independent of dimension', '$1/n^{d}$', '$\\log n / n$'],
        answer: 1,
        why: 'Which is both the curse (100× samples for 10× accuracy) and the blessing (dimension does not appear).'
      },
      {
        q: 'Roughly what fraction of the original rows appears in one bootstrap sample?',
        options: ['50%', '63.2%', '95%', '100%'],
        answer: 1,
        why: '$1-(1-1/n)^n \\to 1-e^{-1} = 0.632$. The other 36.8% is the out-of-bag set used by random forests.'
      },
      {
        q: 'You have 5 million impressions from 40,000 users. To bootstrap CTR you should resample…',
        options: ['impressions', 'users', 'either — it makes no difference', 'days'],
        answer: 1,
        why: 'Impressions within a user are correlated. Resampling rows treats 5M as the sample size and produces an interval that is far too tight.'
      },
      {
        q: 'Importance sampling with a proposal that has lighter tails than the target can…',
        options: ['only be slow', 'have infinite variance, with a sample SE that hides it', 'introduce bias', 'never converge to the right answer'],
        answer: 1,
        why: 'It stays unbiased but the variance can be unbounded, and the empirical standard error understates it. Report the effective sample size.'
      }
    ],
    cards: [
      { q: 'Monte Carlo error rate', a: '$\\sigma/\\sqrt n$, independent of dimension. Ten times more accuracy costs a hundred times more samples.' },
      { q: 'The bootstrap in one line', a: 'Resample your data with replacement $B$ times, recompute the statistic, and read the spread as its sampling distribution.' },
      { q: 'Where the bootstrap fails', a: 'Extremes (max, min, high quantiles), dependent data resampled at the wrong level, and very small $n$.' },
      { q: 'Effective sample size for weights', a: '$\\mathrm{ESS} = (\\sum w_i)^2 / \\sum w_i^2$. A collapsed ESS means one sample is doing all the work.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.14 */
  ML.section({
    id: 'bayesian-inference', track: 'foundations', num: '1.14', level: 3,
    title: 'Bayesian inference in practice: conjugacy, MCMC, variational',
    lede: 'Bayes’ rule (§1.1) is a line of algebra. Doing Bayesian inference on a real model means computing a posterior you cannot write down — and there are exactly three strategies for that, each with a distinct failure mode.',
    prereq: ['bayes', 'mle-map'],
    related: ['gp-bayesopt', 'sampling', 'vae-gan'],
    html: `
${H.tldr([
      'Three routes to a posterior: <b>conjugacy</b> (exact, rare, instant), <b>MCMC</b> (asymptotically exact, slow, diagnosable), <b>variational</b> (fast, biased, scales — and is what a VAE does, §6.3).',
      'The obstacle is always the same: the evidence $p(D) = \\int p(D\\mid\\theta)p(\\theta)d\\theta$. Conjugacy makes it analytic, MCMC avoids it, VI replaces it with a bound.',
      'A Bayesian answer is a <i>distribution</i>. If you are going to collapse it to a point anyway, you have paid for something you did not use — except that the width is usually the part you needed (§2.12, §2.21).'
    ])}

<h2><span class="sn">1.14.1</span> The obstacle, stated once</h2>
$$p(\\theta\\mid D) = \\frac{p(D\\mid\\theta)\\,p(\\theta)}{p(D)},\\qquad p(D)=\\int p(D\\mid\\theta)p(\\theta)\\,d\\theta$$
<p>The numerator is easy: it is a likelihood times a prior, both of which you wrote down. The denominator is an integral over the whole parameter space, and for anything with more than a handful of parameters it has no closed form and no feasible quadrature. Everything below is a way around that integral.</p>

<h2><span class="sn">1.14.2</span> Conjugacy: when the algebra closes</h2>
<p>A prior is <b>conjugate</b> to a likelihood if the posterior is in the same family. Then updating is arithmetic on the parameters, and you can do it in your head.</p>
${H.table(['Likelihood', 'Conjugate prior', 'Posterior', 'Read it as'], [
      ['Bernoulli / Binomial', 'Beta$(\\alpha,\\beta)$', 'Beta$(\\alpha+s,\\ \\beta+n-s)$', 'α−1 pseudo-successes, β−1 pseudo-failures'],
      ['Poisson', 'Gamma$(\\alpha,\\beta)$', 'Gamma$(\\alpha+\\sum x_i,\\ \\beta+n)$', 'α pseudo-events in β pseudo-periods'],
      ['Normal (known σ²)', 'Normal$(\\mu_0,\\tau^2)$', 'Normal, precision-weighted mean', 'precisions add; means average by precision'],
      ['Multinomial', 'Dirichlet$(\\alpha)$', 'Dirichlet$(\\alpha + \\text{counts})$', 'the smoothing in Naive Bayes (§2.5)'],
      ['Normal (unknown σ²)', 'Normal-Inverse-Gamma', 'Normal-Inverse-Gamma', 'the origin of the $t$ distribution']
    ])}
${H.lab('conjugate', 'Beta–Binomial updating, one observation at a time', 'The prior is your belief about a conversion rate before any data. Click to add a success or a failure and watch the posterior narrow. This is the entire mathematics behind Thompson sampling (§6.6) and behind why a 2-out-of-2 conversion rate should not be reported as 100%.')}

${H.worked('the sentence to have ready', `
<p>You observe 7 conversions in 10 visits. With a uniform prior Beta(1,1), the posterior is Beta(8,4): mean $8/12 = 0.667$, mode $7/10 = 0.700$ (that is the MLE), and a 95% credible interval of roughly $[0.39, 0.89]$.</p>
<p><b>The posterior mean is the MLE shrunk toward the prior mean</b>, and the amount of shrinkage is the prior's pseudo-count divided by the total count. With Beta(1,1) that is 2 pseudo-observations against 10 real ones — noticeable. At $n=1000$ it is invisible. That single sentence — <i>the prior is worth $\\alpha+\\beta$ observations and its influence decays as $1/n$</i> — answers most "how much does the prior matter" questions.</p>`)}

<h2><span class="sn">1.14.3</span> MCMC: build a chain whose stationary distribution is the posterior</h2>
<p>Metropolis–Hastings needs only the <i>unnormalised</i> posterior, which is exactly the part you can compute:</p>
${H.steps([
      'Propose $\\theta^{*} \\sim q(\\cdot\\mid\\theta_t)$ — usually a Gaussian step from where you are.',
      'Accept with probability $\\min\\!\\left(1, \\dfrac{p(D\\mid\\theta^*)p(\\theta^*)}{p(D\\mid\\theta_t)p(\\theta_t)}\\right)$ for a symmetric proposal. <b>The intractable $p(D)$ cancels in the ratio.</b> That cancellation is the whole trick.',
      'If accepted, move; otherwise stay put and record the current point again. The chain’s stationary distribution is the posterior.'
    ])}
${H.lab('mcmc', 'Metropolis on a posterior you cannot integrate', 'A genuinely awkward "banana" posterior. Drag the step size and watch the classic trade: too small and the chain crawls with a 90% acceptance rate and enormous autocorrelation; too large and it rejects nearly everything and stands still. The sweet spot is around 25–40% acceptance, which is the standard tuning target.')}

${H.table(['Diagnostic', 'What it detects', 'Threshold people use'], [
      ['Trace plot', 'the chain stuck, drifting, or not mixing', 'should look like a fuzzy caterpillar, not a wandering line'],
      ['Acceptance rate', 'proposal scale wrong', '≈0.234 asymptotically optimal for random-walk MH; 0.6–0.8 for HMC/NUTS'],
      ['$\\hat R$ (Gelman–Rubin)', 'chains from different starts disagreeing', '$\\hat R < 1.01$'],
      ['Effective sample size', 'autocorrelation eating your samples', 'ESS > 400 per parameter'],
      ['Divergences (HMC)', 'geometry the sampler cannot follow', 'any at all is a warning; reparameterise']
    ])}
${H.flag('Modern practice is not hand-rolled Metropolis. It is Hamiltonian Monte Carlo with the No-U-Turn Sampler (Stan, PyMC, NumPyro), which uses gradients of the log posterior to propose distant, high-acceptance moves. The lab below is Metropolis because it is the version you can watch; the version you would ship is NUTS.')}

<h2><span class="sn">1.14.4</span> Variational inference: turn integration into optimisation</h2>
<p>Pick a tractable family $q_\\phi$ and find the member closest to the posterior in KL. Because $\\mathrm{KL}(q\\|p)$ contains the unknown $\\log p(D)$, minimise it by maximising a bound instead:</p>
$$\\log p(D) \\;=\\; \\underbrace{\\mathbb{E}_{q}[\\log p(D,\\theta) - \\log q(\\theta)]}_{\\text{ELBO}} \\;+\\; \\mathrm{KL}(q\\,\\|\\,p(\\theta\\mid D))$$
<p>The KL term is non-negative and unknown; the ELBO is computable. Push the ELBO up and you push the KL down — <b>which is why maximising the ELBO is inference</b>. The same identity, with $\\theta$ renamed to a latent $z$ and $q$ produced by an encoder network, <i>is</i> the variational autoencoder (§6.3).</p>
${H.vs('MCMC', [
      'Asymptotically exact — given enough time it is the posterior',
      'Has honest diagnostics ($\\hat R$, ESS, divergences)',
      'Slow; scales badly in dimension and dataset size',
      'Hard to run inside a training loop'
    ], 'Variational inference', [
      'Fast, batched, differentiable — runs on a GPU',
      'Biased by the family you chose, with no way to measure the gap',
      'Mean-field $q$ <b>systematically under-estimates variance</b> — the mode-seeking direction of reverse KL',
      'The default when inference must be amortised over millions of data points'
    ])}
${H.key('Reverse KL $\\mathrm{KL}(q\\|p)$ is mode-seeking: it pays an infinite price for putting mass where $p$ has none, so $q$ hides inside one mode and comes out too narrow. Forward KL $\\mathrm{KL}(p\\|q)$ is mass-covering and over-disperses. Which one you optimise decides which way you will be wrong (§1.10).')}

${H.probe([
      ['Why does Metropolis–Hastings not need the normalising constant?', 'Only the ratio of unnormalised posteriors enters the acceptance probability, and $p(D)$ cancels.'],
      ['Your MCMC has $\\hat R = 1.3$. What does that mean and what do you do?', 'The chains have not converged to a common distribution. Run longer, reparameterise (non-centred parameterisation for hierarchical models), or check for an unidentified parameter.'],
      ['Why is mean-field VI over-confident?', 'It optimises reverse KL, which penalises $q$ having mass where $p$ has none but not the reverse — so $q$ contracts onto one mode. The posterior variance it reports is a lower bound.'],
      ['When is Bayesian inference actually worth the cost?', 'Small data, hierarchical structure with partial pooling, genuine need for calibrated uncertainty in a decision, or a sequential decision problem where the posterior drives exploration (§6.6). Not for a well-fed classifier where a validation set answers the question.']
    ], 'Claiming "Bayesian methods prevent overfitting". They regularise via the prior and average over uncertainty, but a badly specified model overfits Bayesianly and reports narrow intervals around the wrong answer.')}`,
    labs: {
      conjugate: function (host) {
        const el = ML.el;
        let a = 1, b = 1, s = 0, f = 0;
        const st = Viz.controls(host, [
          { k: 'a0', label: 'prior α', min: .5, max: 20, step: .5, value: 1, fmt: v => v },
          { k: 'b0', label: 'prior β', min: .5, max: 20, step: .5, value: 1, fmt: v => v }
        ], () => S.redraw());
        Viz.buttons(host, [
          { label: '+ success', primary: true, on: () => { s++; S.redraw(); } },
          { label: '+ failure', on: () => { f++; S.redraw(); } },
          { label: '+10 at 70%', on: () => { for (let i = 0; i < 10; i++) (Math.random() < .7 ? s++ : f++); S.redraw(); } },
          { label: 'clear data', on: () => { s = 0; f = 0; S.redraw(); } }
        ]);
        const out = Viz.readout(host, [
          { k: 'data', label: 'data', cls: 'key' },
          { k: 'mle', label: 'MLE (mode)' },
          { k: 'post', label: 'posterior mean' },
          { k: 'ci', label: '95% credible' },
          { k: 'shrink', label: 'shrinkage' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            a = st.a0 + s; b = st.b0 + f;
            const grid = Num.linspace(0.001, 0.999, 240);
            const post = grid.map(x => Num.betaPdf(x, a, b));
            const prior = grid.map(x => Num.betaPdf(x, st.a0, st.b0));
            const mx = Math.max(Math.max.apply(null, post), Math.max.apply(null, prior));
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, mx * 1.12], pad: { l: 44, r: 14, t: 16, b: 38 } })
              .frame({ xlabel: 'conversion rate θ', ylabel: 'density' });
            // credible interval by cumulative trapezoid
            let tot = 0; const cdf = [];
            for (let i = 0; i < grid.length; i++) { tot += post[i]; cdf.push(tot); }
            const qAt = p => { const target = p * tot; for (let i = 0; i < cdf.length; i++) if (cdf[i] >= target) return grid[i]; return 1; };
            const lo = qAt(.025), hi = qAt(.975);
            P.clip(() => {
              P.area(grid.map((x, i) => [x, post[i]]), { color: T.c1, alpha: .16 });
              P.line(grid.map((x, i) => [x, prior[i]]), { color: T.faint, width: 1.8, dash: [5, 4] });
              P.line(grid.map((x, i) => [x, post[i]]), { color: T.c1, width: 2.8 });
              if (s + f > 0) P.vline(s / (s + f), { color: T.c2, dash: false, width: 1.8, label: 'MLE' });
              P.vline(lo, { color: T.c1, dash: [3, 3], width: 1 });
              P.vline(hi, { color: T.c1, dash: [3, 3], width: 1 });
            });
            const mle = (s + f) ? s / (s + f) : NaN;
            const pm = a / (a + b);
            out({
              data: s + ' / ' + (s + f),
              mle: (s + f) ? mle.toFixed(3) : '—',
              post: pm.toFixed(3),
              ci: '[' + lo.toFixed(3) + ', ' + hi.toFixed(3) + ']',
              shrink: (s + f) ? ((st.a0 + st.b0) / (st.a0 + st.b0 + s + f) * 100).toFixed(0) + '% toward prior' : '100%'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--faint)', t: 'prior' }, { c: 'var(--c1)', t: 'posterior' }, { c: 'var(--c2)', t: 'MLE' }]);
        Viz.note(host, 'Click <b>+ success</b> twice from a uniform prior. The MLE says 100%; the posterior mean says 75% with a credible interval from 0.19 to 0.99. Both are correct answers to different questions, and only one of them is safe to put on a dashboard. Now add fifty observations and watch prior and likelihood swap importance — the shrinkage readout is exactly $(\\alpha_0+\\beta_0)/(\\alpha_0+\\beta_0+n)$.');
      },

      mcmc: function (host) {
        const st = Viz.controls(host, [
          { k: 'step', label: 'proposal step size', min: .05, max: 3, step: .05, value: .6, fmt: v => v.toFixed(2) },
          { k: 'n', label: 'iterations', min: 200, max: 8000, step: 200, value: 3000, fmt: v => v.toLocaleString() },
          { k: 'burn', label: 'discard as burn-in', min: 0, max: 1000, step: 50, value: 200, fmt: v => v },
          { k: 'shape', label: 'posterior', type: 'select', value: 'banana', options: [{ v: 'banana', t: 'banana (correlated)' }, { v: 'bimodal', t: 'bimodal (hard)' }, { v: 'gauss', t: 'Gaussian (easy)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'acc', label: 'acceptance rate', cls: 'key' },
          { k: 'verdict', label: 'tuning' },
          { k: 'ess', label: 'rough ESS' },
          { k: 'mean', label: 'posterior mean x' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const logp = st.shape === 'banana'
              ? p => { const x = p[0], y = p[1]; return -0.5 * (x * x / 4 + Math.pow(y - 0.25 * x * x + 1, 2) / 0.25); }
              : st.shape === 'bimodal'
                ? p => { const d1 = Math.pow(p[0] + 1.6, 2) + Math.pow(p[1], 2), d2 = Math.pow(p[0] - 1.6, 2) + Math.pow(p[1], 2);
                         return Num.logsumexp([-0.5 * d1 / 0.16, -0.5 * d2 / 0.16]); }
                : p => -0.5 * (p[0] * p[0] + p[1] * p[1]) / 0.6;

            const chain = Num.metropolis(logp, [0, 0], { step: st.step, seed: 4 });
            chain.draw(st.n);
            const pts = chain.chain.slice(st.burn);
            const xs = pts.map(p => p[0]);

            const w1 = w * .58;
            const P = Viz.plot(ctx, w1, h, { xd: [-4, 4], yd: [-3, 2.5], pad: { l: 40, r: 8, t: 16, b: 36 } })
              .frame({ xlabel: 'θ₁', ylabel: 'θ₂' });
            P.clip(() => {
              P.field((x, y) => Math.exp(logp([x, y])), { step: 4, colors: t => [90, 120, 200, Math.round(70 * t)] });
              P.contours((x, y) => logp([x, y]), [-8, -4, -2, -1], { color: T.faint, alpha: .5 });
              pts.forEach((p, i) => { if (i % Math.max(1, Math.floor(pts.length / 900)) === 0) P.dots([[p[0], p[1]]], { r: 1.7, color: T.c2, alpha: .55 }); });
              P.line(pts.slice(0, 120), { color: T.c2, width: .8, alpha: .55 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, {
              xd: [0, xs.length], yd: [-4, 4], pad: { l: 36, r: 12, t: 16, b: 36 }
            }).frame({ xlabel: 'iteration', ylabel: 'θ₁ trace' });
            P2.clip(() => {
              P2.line(xs.map((v, i) => [i, v]), { color: T.c1, width: .9 });
              P2.hline(Num.mean(xs), { color: T.c2, dash: [4, 3], label: 'mean' });
            });
            ctx.restore();

            // crude ESS via lag-1 autocorrelation
            const m = Num.mean(xs);
            let num = 0, den = 0;
            for (let i = 1; i < xs.length; i++) num += (xs[i] - m) * (xs[i - 1] - m);
            for (let i = 0; i < xs.length; i++) den += (xs[i] - m) * (xs[i] - m);
            const r1 = den ? num / den : 0;
            const ess = Math.max(1, Math.round(xs.length * (1 - r1) / (1 + r1)));
            const acc = chain.rate;
            out({
              acc: (acc * 100).toFixed(1) + '%',
              verdict: acc > .6 ? 'steps too small' : acc < .12 ? 'steps too large' : '≈ well tuned',
              ess: ess.toLocaleString() + ' of ' + xs.length.toLocaleString(),
              mean: Num.mean(xs).toFixed(3)
            });
          }
        });
        Viz.note(host, 'Select <b>bimodal</b> and set a small step. The chain finds one mode, reports a beautifully tight posterior, and never discovers that the other half of the answer exists — with an acceptance rate that looks perfectly healthy. This is why you run several chains from dispersed starts and check $\\hat R$: no single-chain diagnostic can detect a mode you never visited.');
      }
    },
    quiz: [
      {
        q: 'The Metropolis acceptance ratio avoids the evidence $p(D)$ because…',
        options: ['it is assumed to be 1', 'it cancels in the ratio of unnormalised posteriors', 'it is estimated by sampling', 'the prior is conjugate'],
        answer: 1,
        why: 'Only ratios enter, and the same constant sits in numerator and denominator.'
      },
      {
        q: 'You observe 2 conversions in 2 trials with a uniform Beta(1,1) prior. The posterior mean is…',
        options: ['1.00', '0.75', '0.67', '0.50'],
        answer: 1,
        why: 'Beta(3,1), mean $3/4$. The MLE of 1.00 is the mode, and it is exactly the number you should not report.'
      },
      {
        q: 'Mean-field variational inference typically reports posterior variances that are…',
        options: ['too large', 'too small', 'unbiased', 'not computable at all'],
        answer: 1,
        why: 'Reverse KL is mode-seeking, so $q$ contracts inside one mode. The ELBO gives a bound on the evidence, not on the error in the variance.'
      },
      {
        q: 'Which diagnostic would catch a sampler that never visited a second mode?',
        options: ['acceptance rate', 'multiple chains from dispersed starts, compared with $\\hat R$', 'the trace plot of one chain', 'the ELBO'],
        answer: 1,
        why: 'A single chain stuck in one mode can look perfectly healthy. Disagreement between independently started chains is the only reliable signal.'
      }
    ],
    cards: [
      { q: 'Beta–Binomial update', a: 'Beta$(\\alpha,\\beta)$ + $s$ successes in $n$ trials → Beta$(\\alpha+s,\\beta+n-s)$. Prior worth $\\alpha+\\beta$ observations.' },
      { q: 'Why MH does not need $p(D)$', a: 'The acceptance probability is a ratio of unnormalised posteriors; the constant cancels.' },
      { q: 'The ELBO identity', a: '$\\log p(D) = \\text{ELBO} + \\mathrm{KL}(q\\|p(\\theta|D))$. Maximising the ELBO minimises the KL because the left side is fixed.' },
      { q: 'MCMC health checks', a: '$\\hat R<1.01$, ESS > 400/parameter, acceptance ≈0.23 (RWMH) or 0.6–0.8 (NUTS), zero divergences.' },
      { q: 'Reverse vs forward KL', a: 'Reverse (VI) is mode-seeking and under-disperses; forward is mass-covering and over-disperses.' }
    ]
  });

  /* ------------------------------------------------------------------ 1.15 */
  ML.section({
    id: 'numerics', track: 'foundations', num: '1.15', level: 2,
    title: 'Floating point, stability, and the epsilon in Adam',
    lede: 'Real numbers do not exist in a computer. Almost every "mysterious NaN", every loss that becomes infinity at step 4,000, and every model that trains in float32 and breaks in float16 is one of about five well-understood floating-point failures.',
    related: ['optimisers', 'distributed', 'compression'],
    html: `
${H.tldr([
      'A float has finite <b>range</b> (the exponent) and finite <b>precision</b> (the mantissa). fp16 fails on range; bf16 trades precision to keep fp32’s range, which is why it won for training.',
      'Three moves fix most instability: work in <b>log space</b>, subtract the max before <code>exp</code> (<b>log-sum-exp</b>), and never subtract two nearly-equal large numbers.',
      'The $\\varepsilon$ in Adam, the $\\varepsilon$ in LayerNorm and the clipping in cross-entropy are all the same defensive move: stop a denominator or a logarithm from reaching zero.'
    ])}

<h2><span class="sn">1.15.1</span> What a float actually is</h2>
<p>A value is stored as $\\pm\\, m \\times 2^{e}$ with a fixed number of bits for each part. The exponent sets how big and how small you can go; the mantissa sets how many significant digits you keep.</p>
${H.table(['Format', 'Bits (s/e/m)', 'Max', 'Smallest normal', 'Decimal digits', 'Used for'], [
      ['fp64', '1/11/52', '~1.8e308', '~2.2e-308', '~15.9', 'scientific computing, gradient checks'],
      ['fp32', '1/8/23', '~3.4e38', '~1.2e-38', '~7.2', 'the default everywhere until 2018'],
      ['<b>bf16</b>', '1/8/7', '~3.4e38', '~1.2e-38', '~2.4', '<b>training</b> — same range as fp32, no loss scaling needed'],
      ['fp16', '1/5/10', '65,504', '~6.1e-5', '~3.3', 'training with loss scaling; inference'],
      ['fp8 (E4M3)', '1/4/3', '448', '~2e-3', '~1.5', 'inference, and forward passes on H100-class hardware'],
      ['int8', 'integer', '127', '1', 'exact, 256 levels', 'post-training quantization (§3.13)']
    ], 'num')}
${H.key('bf16 has the same exponent as fp32 and only 8 bits of mantissa. It sacrificed precision — which training tolerates, because gradients are noisy anyway — to keep range, which training does not tolerate losing. That single design choice is why bf16 replaced fp16 for pretraining.')}

${H.lab('formats', 'What happens to your number in each format', 'Type a value, or drag the exponent, and see exactly what each format stores, what it rounds to, and where it overflows or flushes to zero. The gradient magnitudes typical of a deep network are marked.')}

<h2><span class="sn">1.15.2</span> The five failures</h2>
${H.table(['Failure', 'Symptom', 'Cause', 'Fix'], [
      ['Overflow', 'inf, then NaN everywhere', '$e^{z}$ with $z=800$; fp16 gradients past 65,504', 'log-sum-exp; loss scaling; bf16'],
      ['Underflow', 'silent zeros, gradient stops', 'product of 500 probabilities; fp16 gradients below 6e-5', 'work in log space; loss scaling'],
      ['Catastrophic cancellation', 'wrong answer, no warning', 'subtracting nearly equal numbers destroys significant digits', 'algebraic rearrangement; Welford; two-pass'],
      ['Division by ~zero', 'inf, NaN, exploding update', 'normalising by a variance that is 0', 'the $\\varepsilon$ in Adam and LayerNorm'],
      ['Non-associativity', 'results differ run to run', '$(a+b)+c \\ne a+(b+c)$ in float; GPU reduction order varies', 'accept it, or force deterministic kernels and pay ~10–20%']
    ])}

${H.deriv('log-sum-exp: the most useful three lines in numerical ML', [
      ['$\\log\\sum_j e^{z_j}$', 'Wanted. With $z_j = 1000$, every term overflows and the answer is <code>inf</code> — even though the true value is about 1000.'],
      ['$= \\log\\sum_j e^{z_j - c}e^{c}$', 'Insert $e^{c}e^{-c}=1$ for any constant $c$. Nothing has changed mathematically.'],
      ['$= c + \\log\\sum_j e^{z_j-c}$', 'Pull the constant out of the sum and the log. Still exact.'],
      ['$c := \\max_j z_j$', 'Choose the max. Now the largest exponent is exactly $e^{0}=1$, everything else is in $(0,1]$, nothing can overflow, and the worst that happens is that tiny terms underflow to zero — where they contribute nothing anyway.']
    ], 'This is inside every softmax, every cross-entropy, every HMM forward pass and every mixture-model E-step you will ever use. Frameworks do it for you <i>only if you use the fused op</i>: <code>cross_entropy(logits, y)</code> is stable, <code>log(softmax(logits))</code> then NLL is not.')}

${H.lab('stability', 'Break it, then fix it', 'The same three computations done naively and stably, on inputs you control. Push the magnitude slider and watch the naive column produce inf, NaN or a negative variance while the stable column keeps working.')}

<h2><span class="sn">1.15.3</span> Cancellation, and the negative variance</h2>
<p>The textbook one-pass variance formula $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$ is algebraically correct and numerically disastrous when the mean is large relative to the spread. With values near $10^9$ and a standard deviation of 1, the two terms agree to fifteen digits — and fp64 has fifteen digits, so their difference is pure rounding noise. It can and does come out <b>negative</b>.</p>
${H.code(`# Welford's online algorithm: one pass, numerically stable, and it
# streams — which is why it is what every metrics library actually uses.
n = 0; mean = 0.0; M2 = 0.0
for x in stream:
    n += 1
    d = x - mean
    mean += d / n
    M2 += d * (x - mean)      # note: the *updated* mean. That is the trick.
var = M2 / (n - 1)`)}
${H.note('The same disease appears in $\\log(1+x)$ for tiny $x$ (use <code>log1p</code>), in $e^{x}-1$ (use <code>expm1</code>), and in the quadratic formula when $b^2 \\gg 4ac$. Every language ships the fixed versions; almost nobody reaches for them until they have been bitten.')}

<h2><span class="sn">1.15.4</span> Where the epsilons live</h2>
${H.table(['Place', 'Typical value', 'What it prevents', 'What happens if you change it'], [
      ['Adam denominator $\\sqrt{\\hat v}+\\varepsilon$', '1e-8', 'division by zero for a coordinate with no gradient history', 'larger ε damps the adaptivity — 1e-3 makes Adam behave more like SGD; some LLM recipes use 1e-6 for stability in bf16'],
      ['LayerNorm / BatchNorm $\\sqrt{\\sigma^2+\\varepsilon}$', '1e-5', 'a constant feature has zero variance', 'too small → inf on constant inputs; too large → the normalisation stops normalising'],
      ['Cross-entropy clipping', '1e-7…1e-12', '$\\log 0 = -\\infty$', 'use the fused logits version and you do not need it at all'],
      ['Softmax temperature floor', '~1e-3', 'divide-by-zero at $T\\to0$', 'greedy decoding should be a branch, not a limit (§4.15)'],
      ['Gradient clipping norm', '1.0', 'a single bad batch destroying the weights', 'the standard defence against loss spikes in LLM pretraining (§4.11)']
    ])}
${H.pitfall('Adam’s $\\varepsilon$ sits <i>outside</i> the square root in the standard formulation: $\\hat m/(\\sqrt{\\hat v}+\\varepsilon)$. Putting it inside, $\\hat m/\\sqrt{\\hat v+\\varepsilon}$, is a different algorithm with different behaviour for small gradients — and both appear in real codebases. If you are debugging a reproduction that will not match, this is a place to look.')}

${H.probe([
      ['Why did bf16 replace fp16 for training?', 'Same 8-bit exponent as fp32, so activations and gradients cannot overflow or flush to zero, and no loss scaling is required. It pays with mantissa bits, which stochastic training barely notices.'],
      ['Write down the stable softmax.', 'Subtract $\\max_j z_j$ from every logit first: $\\mathrm{softmax}(z)_k = e^{z_k-c}/\\sum_j e^{z_j-c}$ with $c=\\max z$. Exact, and nothing can overflow.'],
      ['Your variance came out negative. Explain.', 'One-pass $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$ with a large mean: catastrophic cancellation. Use Welford or a two-pass computation.'],
      ['Why is the same training run not bit-identical on two GPUs?', 'Floating-point addition is not associative, and reduction order depends on the kernel, the block size and the number of devices. Determinism is available and costs throughput.'],
      ['What is loss scaling and why does fp16 need it?', 'Multiply the loss by a large constant before the backward pass so gradients land inside fp16’s representable range, then unscale before the optimiser step. bf16 does not need it because its range already matches fp32.']
    ], 'Treating NaN as a mystery. NaN has exactly a few sources — 0/0, inf−inf, log of a negative, sqrt of a negative, and inf*0. Find the first NaN, not the hundredth: hooks that check each tensor after every op will localise it in one run.')}`,
    labs: {
      formats: function (host) {
        const st = Viz.controls(host, [
          { k: 'exp', label: 'magnitude (10^x)', min: -12, max: 12, step: .25, value: -4, fmt: v => '1e' + v.toFixed(2) },
          { k: 'mant', label: 'mantissa digits', min: 1, max: 9, step: 1, value: 4, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'val', label: 'your value', cls: 'key' },
          { k: 'f32', label: 'fp32' }, { k: 'bf', label: 'bf16' },
          { k: 'f16', label: 'fp16' }, { k: 'f8', label: 'fp8 e4m3' }
        ]);

        function round(v, mantBits, maxV, minNormal) {
          if (!isFinite(v) || Math.abs(v) > maxV) return Infinity * Math.sign(v);
          if (Math.abs(v) < minNormal) return 0;
          const e = Math.floor(Math.log2(Math.abs(v)));
          const q = Math.pow(2, e - mantBits);
          return Math.round(v / q) * q;
        }
        const FMT = {
          f32: { m: 23, max: 3.4e38, min: 1.2e-38 },
          bf: { m: 7, max: 3.4e38, min: 1.2e-38 },
          f16: { m: 10, max: 65504, min: 6.1e-5 },
          f8: { m: 3, max: 448, min: 2e-3 }
        };
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const v = Math.pow(10, st.exp) * (1 + 0.1 * st.mant);
            const P = Viz.plot(ctx, w, h, { xd: [-14, 14], yd: [0, 5], pad: { l: 60, r: 16, t: 20, b: 40 } })
              .frame({ grid: false, yticks: [], xticks: [-12, -8, -4, 0, 4, 8, 12], xfmt: x => '1e' + x, xlabel: 'magnitude' });
            const rows = [['fp32', FMT.f32, T.c1], ['bf16', FMT.bf, T.c3], ['fp16', FMT.f16, T.c4], ['fp8 e4m3', FMT.f8, T.c2]];
            rows.forEach((r, i) => {
              const y = 4 - i;
              const lo = Math.log10(r[1].min), hi = Math.log10(r[1].max);
              ctx.fillStyle = r[2]; ctx.globalAlpha = .22;
              ctx.fillRect(P.x(lo), P.y(y) - 11, P.x(hi) - P.x(lo), 22);
              ctx.globalAlpha = 1;
              ctx.strokeStyle = r[2]; ctx.lineWidth = 1.4;
              ctx.strokeRect(P.x(lo), P.y(y) - 11, P.x(hi) - P.x(lo), 22);
              ctx.fillStyle = T.text; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              ctx.fillText(r[0], P.x(-14) - 6, P.y(y));
            });
            P.vline(st.exp, { color: T.text, width: 2, dash: false, label: 'your value' });
            // the band where deep-net gradients live
            ctx.fillStyle = T.faint; ctx.globalAlpha = .16;
            ctx.fillRect(P.x(-8), P.y(4.7), P.x(-2) - P.x(-8), P.y(0.2) - P.y(4.7));
            ctx.globalAlpha = 1;
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
            ctx.fillText('typical gradient magnitudes', P.x(-5), P.y(4.75));

            const fmtOut = (key) => {
              const r = round(v, FMT[key].m, FMT[key].max, FMT[key].min);
              if (!isFinite(r)) return 'overflow → inf';
              if (r === 0) return 'underflow → 0';
              const relErr = Math.abs(r - v) / v;
              return r.toExponential(3) + '  (' + (relErr * 100).toFixed(relErr > .01 ? 1 : 3) + '% off)';
            };
            out({ val: v.toExponential(6), f32: fmtOut('f32'), bf: fmtOut('bf'), f16: fmtOut('f16'), f8: fmtOut('f8') });
          }
        });
        Viz.note(host, 'Set the magnitude to 1e-6 — a perfectly ordinary gradient. fp32 and bf16 hold it; <b>fp16 flushes it to zero</b>, and the parameter it belonged to stops learning. That single failure is what loss scaling exists to prevent, and what bf16 makes unnecessary. Now go to 1e6 and watch fp16 and fp8 overflow instead.');
      },

      stability: function (host) {
        const st = Viz.controls(host, [
          { k: 'mag', label: 'input magnitude', min: 1, max: 900, step: 1, value: 40, fmt: v => v },
          { k: 'nprob', label: 'probabilities to multiply', min: 10, max: 2000, step: 10, value: 400, fmt: v => v },
          { k: 'offset', label: 'mean offset for the variance', min: 0, max: 10, step: .5, value: 0, fmt: v => '1e' + v }
        ], () => draw());
        const host2 = ML.el('div');
        host.appendChild(host2);

        function draw() {
          const z = [st.mag, st.mag - 1, st.mag - 3, st.mag - 7];
          // 1. softmax
          const naiveDen = z.reduce((a, v) => a + Math.exp(v), 0);
          const naive = z.map(v => Math.exp(v) / naiveDen);
          const c = Math.max.apply(null, z);
          const stableDen = z.reduce((a, v) => a + Math.exp(v - c), 0);
          const stable = z.map(v => Math.exp(v - c) / stableDen);
          // 2. product of probabilities
          let prod = 1, logsum = 0;
          for (let i = 0; i < st.nprob; i++) { prod *= 0.7; logsum += Math.log(0.7); }
          // 3. variance with a large offset
          const R = Num.rng(2);
          const off = Math.pow(10, st.offset);
          const xs = Array.from({ length: 200 }, () => off + R.normal(0, 1));
          let s1 = 0, s2 = 0;
          xs.forEach(x => { s1 += x; s2 += x * x; });
          const onepass = s2 / xs.length - Math.pow(s1 / xs.length, 2);
          const m = s1 / xs.length;
          let ss = 0; xs.forEach(x => { ss += (x - m) * (x - m); });
          const twopass = ss / xs.length;

          const fmt = v => !isFinite(v) ? '<span style="color:var(--red)">' + (v > 0 ? 'inf' : '-inf') + '</span>'
            : isNaN(v) ? '<span style="color:var(--red)">NaN</span>'
              : (v === 0 ? '<span style="color:var(--red)">0 (underflow)</span>' : v.toPrecision(6));

          host2.innerHTML = H.table(
            ['Computation', 'Naive', 'Stable', 'Why'],
            [
              ['softmax, largest probability',
                fmt(naive[0]), fmt(stable[0]),
                '<code>exp(' + st.mag + ')</code> overflows fp64 past 709'],
              ['product of ' + st.nprob + ' × 0.7',
                fmt(prod), 'exp(' + logsum.toFixed(1) + ') — kept in logs',
                'underflows below ~1e-308'],
              ['variance, values near 1e' + st.offset,
                fmt(onepass), fmt(twopass),
                'E[X²] − E[X]² cancels away every significant digit']
            ], 'num');
        }
        draw();
        Viz.note(host, 'Push the magnitude past 709 and the naive softmax returns NaN — <code>inf/inf</code>. The stable one is unmoved, because after subtracting the max the largest term is exactly 1. Push the mean offset to 1e8 and the one-pass variance goes negative: an impossible answer, returned confidently, with no warning of any kind. <b>That silence is what makes cancellation the dangerous one.</b>');
      }
    },
    quiz: [
      {
        q: 'bf16 is preferred to fp16 for training because…',
        options: ['it has more mantissa bits', 'it has the same exponent range as fp32, so gradients neither overflow nor flush to zero', 'it is faster on all hardware', 'it is more accurate'],
        answer: 1,
        why: 'bf16 has *fewer* mantissa bits and is *less* precise. It wins on range, which is the thing training cannot do without.'
      },
      {
        q: 'The log-sum-exp trick works by…',
        options: ['taking logs of the inputs', 'subtracting the maximum before exponentiating', 'clipping large values', 'using float64'],
        answer: 1,
        why: '$\\log\\sum e^{z_j} = c + \\log\\sum e^{z_j-c}$ with $c=\\max z$. Exact, and the largest exponent becomes $e^0=1$.'
      },
      {
        q: 'Your one-pass variance returns −3.2e-7. The cause is…',
        options: ['a bug in the data', 'catastrophic cancellation between $\\mathbb{E}[X^2]$ and $\\mathbb{E}[X]^2$', 'integer overflow', 'the sample being too small'],
        answer: 1,
        why: 'Two nearly-equal large numbers subtracted; every significant digit is lost. Welford or two-pass fixes it.'
      },
      {
        q: 'Adam’s $\\varepsilon = 10^{-8}$ exists to…',
        options: ['add regularization', 'prevent division by zero when the second-moment estimate is tiny', 'set the learning rate floor', 'stabilise the momentum'],
        answer: 1,
        why: 'It is a denominator guard. Raising it toward 1e-3 damps adaptivity and pushes Adam toward SGD behaviour.'
      }
    ],
    cards: [
      { q: 'fp16 vs bf16, in one line', a: 'Same 16 bits: fp16 spends them on precision, bf16 on range. Training needs range, so bf16 won.' },
      { q: 'Stable softmax', a: 'Subtract $\\max_j z_j$ from every logit before exponentiating. Exact, and overflow becomes impossible.' },
      { q: 'Welford’s update', a: '$d = x-\\mu$; $\\mu \\mathrel{+}= d/n$; $M_2 \\mathrel{+}= d\\,(x-\\mu_{\\text{new}})$. One pass, stable, streaming.' },
      { q: 'fp16 representable range', a: 'Max 65,504; smallest normal ≈6e-5. Gradients below that flush to zero — hence loss scaling.' },
      { q: 'Why runs are not bit-reproducible', a: 'Float addition is not associative and GPU reduction order varies with kernel, block size and device count.' }
    ]
  });
})();
