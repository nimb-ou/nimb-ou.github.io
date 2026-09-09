/* ============================================================
   PART 0 — Start here (0.6 – 0.8): notation, matrix calculus,
   and one complete supervised workflow end to end.
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 0.6 */
  ML.section({
    id: 'notation', track: 'start', num: '0.6', level: 1,
    title: 'Reading the notation',
    lede: 'Most people who "can’t do the maths" can do the maths. What stops them is that nobody ever read a formula out loud for them, left to right, saying what each mark is for. This section does that, and then makes you do it.',
    related: ['matrix-calculus', 'probability-basics', 'formulas'],
    html: `
${H.tldr([
      'A formula is a sentence. Sums are <code>for</code> loops, subscripts are indices, $\\mathbb{E}$ is a weighted average, and $\\arg\\min$ returns <i>the input</i>, not the value.',
      'Read every formula in three passes: <b>shapes</b> (what is a scalar, a vector, a matrix?), then <b>the loop</b> (what is being summed, over what index?), then <b>the story</b> (what would make this big or small?).',
      'Notation is not standard across papers. Fix the shapes first and the disagreements stop mattering.'
    ])}

<h2><span class="sn">0.6.1</span> A formula is a sentence</h2>
<p>Take the one you will meet a hundred times on this site, the mean squared error:</p>
$$\\mathcal{L}(w) = \\frac{1}{n}\\sum_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2$$
<p>Out loud: <i>"The loss, which depends on the parameters $w$, is the average over all $n$ examples of the squared gap between the true answer and our prediction."</i> Every mark did a job:</p>
${H.table(['Mark', 'Out loud', 'In code'], [
      ['$\\mathcal{L}(w)$', 'a number that depends on $w$ — the thing we push down', '<code>def loss(w): ...</code>'],
      ['$\\frac{1}{n}\\sum_{i=1}^{n}$', 'average over the examples', '<code>np.mean(...)</code>'],
      ['$i$', 'which example — an index, never a quantity', 'the loop variable'],
      ['$y_i$', 'the true label of example $i$', '<code>y[i]</code>'],
      ['$\\hat{y}_i$', 'the <i>estimate</i> — the hat always means "our guess at"', '<code>yhat[i]</code>'],
      ['$(\\cdot)^2$', 'punish both directions, punish big errors more', '<code>** 2</code>']
    ])}
${H.key('The hat means estimate, the subscript means index, and $\\sum$ with $1/n$ means average. Those three conventions decode most of machine learning.')}

${H.lab('dissect', 'The formula dissector', 'Step through a formula one symbol at a time. Each step highlights a piece, says what it is for in plain English, and evaluates that piece on the toy numbers shown — so the abstract mark and the concrete number are on the screen together.')}

<h2><span class="sn">0.6.2</span> The decoder ring</h2>
<p>Skim this now; come back to it whenever a line stops you.</p>
${H.table(['Symbol', 'Name', 'What it does', 'Read it as'], [
      ['$\\sum_{i=1}^{n} x_i$', 'sum', 'adds a list up', '"add up all the $x$s"'],
      ['$\\prod_{i} p_i$', 'product', 'multiplies a list — usually independent probabilities', '"multiply all the $p$s"'],
      ['$\\mathbb{E}[X]$', 'expectation', 'the long-run average of a random quantity', '"on average, $X$ is…"'],
      ['$\\mathbb{E}_{x\\sim p}[f(x)]$', 'expectation under $p$', 'average of $f$ weighting each $x$ by how likely $p$ says it is', '"average $f$, as $p$ sees the world"'],
      ['$\\mathrm{Var}(X)$', 'variance', 'average squared distance from the mean', '"how spread out"'],
      ['$P(A\\mid B)$', 'conditional', 'probability of $A$ <i>in the world where $B$ happened</i>', '"$A$ given $B$"'],
      ['$\\arg\\min_{w} f(w)$', 'argmin', 'the $w$ that makes $f$ smallest — <b>not</b> the smallest value', '"whichever $w$ wins"'],
      ['$\\nabla_w f$', 'gradient', 'vector of partial derivatives — the uphill direction', '"which way is up, in every parameter at once"'],
      ['$\\partial f/\\partial x$', 'partial derivative', 'slope in $x$ holding everything else still', '"if I nudge only $x$…"'],
      ['$x^\\top y$', 'dot product', 'sum of elementwise products; one number', '"how aligned are these"'],
      ['$A \\odot B$', 'Hadamard', 'elementwise multiply, same shape out', '"multiply cell by cell"'],
      ['$\\|x\\|_2$', 'L2 norm', 'length: $\\sqrt{\\sum x_i^2}$', '"how big"'],
      ['$\\|x\\|_1$', 'L1 norm', '$\\sum|x_i|$ — the one that makes things sparse (§2.3)', '"total absolute size"'],
      ['$\\propto$', 'proportional to', 'equal up to a constant we do not care about', '"same shape, ignore the scale"'],
      ['$x \\sim \\mathcal{N}(\\mu,\\sigma^2)$', 'distributed as', '$x$ is drawn from that distribution', '"$x$ comes from a bell curve"'],
      ['$\\mathbb{1}[\\cdot]$', 'indicator', '1 when the condition holds, 0 otherwise', '"count it if…"'],
      ['$\\theta$', 'theta', 'the generic name for "all the parameters"', '"the knobs"'],
      ['$:=$', 'defined as', 'this is a definition, not a claim', '"let this mean"']
    ], 'plain')}

<h2><span class="sn">0.6.3</span> Shapes come first</h2>
<p>Before you read what a formula <i>says</i>, work out what shape everything is. Nine tenths of confusion is a shape confusion, and the convention on this site — the common one in machine learning — is:</p>
${H.table(['Object', 'Shape', 'Convention here'], [
      ['$x$ (one example)', '$d$', 'a column vector of $d$ features'],
      ['$X$ (the data)', '$n \\times d$', '<b>rows are examples</b>, columns are features'],
      ['$y$ (the labels)', '$n$', 'one number per row of $X$'],
      ['$w$ (the weights)', '$d$', 'one weight per feature'],
      ['$Xw$', '$n$', 'one prediction per example'],
      ['$W$ (a layer, §3.1)', '$d_{\\text{out}} \\times d_{\\text{in}}$', 'so $Wx$ maps in to out']
    ])}
${H.note('Papers differ. Some make $x$ a row and write $xW$; deep learning libraries usually store a batch as $(\\text{batch}, \\text{features})$ and compute $XW^\\top$. Neither is more correct. <b>Write the shapes in the margin and every product either fits or it does not.</b>')}

${H.svg('reading a matrix product', '0 0 640 150', `
<defs><style>
.lbl{font:11px ui-monospace,monospace;fill:var(--muted)}
.big{font:600 13px ui-sans-serif,system-ui;fill:var(--text)}
.bx{fill:var(--panel);stroke:var(--line)}
.hl{fill:color-mix(in oklab,var(--blue) 22%,transparent);stroke:var(--blue)}
.hl2{fill:color-mix(in oklab,var(--c2) 22%,transparent);stroke:var(--c2)}
</style></defs>
<rect class="bx" x="20" y="35" width="150" height="90" rx="6"/>
<rect class="hl" x="20" y="60" width="150" height="18"/>
<text class="big" x="95" y="25" text-anchor="middle">X</text>
<text class="lbl" x="95" y="140" text-anchor="middle">n × d  (rows = examples)</text>
<text class="big" x="196" y="85">×</text>
<rect class="bx" x="222" y="35" width="46" height="90" rx="6"/>
<rect class="hl2" x="222" y="35" width="46" height="90"/>
<text class="big" x="245" y="25" text-anchor="middle">w</text>
<text class="lbl" x="245" y="140" text-anchor="middle">d × 1</text>
<text class="big" x="292" y="85">=</text>
<rect class="bx" x="320" y="35" width="46" height="90" rx="6"/>
<rect class="hl" x="320" y="60" width="46" height="18"/>
<text class="big" x="343" y="25" text-anchor="middle">Xw</text>
<text class="lbl" x="343" y="140" text-anchor="middle">n × 1</text>
<text class="lbl" x="400" y="60">one highlighted row of X,</text>
<text class="lbl" x="400" y="76">dotted with all of w,</text>
<text class="lbl" x="400" y="92">gives one prediction.</text>
<text class="lbl" x="400" y="116">The inner dimensions (d) must match;</text>
<text class="lbl" x="400" y="132">the outer ones (n, 1) survive.</text>
`, 'Every matrix product is this picture. The inner dimensions vanish into a sum; the outer dimensions are the shape of the answer.')}

<h2><span class="sn">0.6.4</span> Three passes, every time</h2>
${H.steps([
      '<b>Shapes.</b> Label every symbol scalar / vector / matrix and write its dimensions. If a product does not conform, you have misread something — stop and fix it before continuing.',
      '<b>The loop.</b> Find the $\\sum$, $\\prod$ or $\\mathbb{E}$. What is the index? What is inside? Anything not indexed is constant with respect to that loop and can be pulled out — this is the single most useful algebraic move in the subject.',
      '<b>The story.</b> Ask: what makes this quantity large? What makes it zero? What happens at the extremes? A formula you can interrogate this way is a formula you will not forget.'
    ])}
${H.worked('pass three on cross-entropy', `
$$\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i\\log \\hat p_i + (1-y_i)\\log(1-\\hat p_i)\\big]$$
<p><b>Zero when?</b> When $\\hat p_i = y_i$ exactly for every $i$ — because $\\log 1 = 0$. <b>Infinite when?</b> When you predict $\\hat p = 0$ for something that happens: $\\log 0 = -\\infty$. That is the whole reason confident wrong answers are punished so brutally, and the reason production code clips probabilities away from 0 and 1 (§1.15). <b>Only one term survives</b> per example, because $y_i$ is 0 or 1 — the bracket is a switch, not a sum.</p>`)}

${H.probe([
      ['What is the difference between $\\min f$ and $\\arg\\min f$?', '$\\min$ is the smallest <i>value</i> of $f$; $\\arg\\min$ is the <i>input</i> that achieves it. Training returns an $\\arg\\min$.'],
      ['What does $\\mathbb{E}_{x\\sim p}[f(x)]$ mean when $p$ is the data distribution?', 'The average of $f$ over the data-generating process — which we approximate by the sample mean over a finite dataset. That gap is the whole of generalization (§1.4).'],
      ['Why does $\\propto$ show up all over Bayes?', 'Because the denominator $P(B)$ does not depend on the hypothesis, so it cannot change which hypothesis wins. Dropping it saves work and never changes the argmax (§1.1).']
    ], 'Saying "sigma" instead of reading the sum. If you cannot say what the index ranges over, you have not read the formula.')}`,
    labs: {
      dissect: function (host) {
        const el = ML.el;
        const FORMULAS = {
          mse: {
            name: 'Mean squared error',
            tex: '\\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat y_i)^2',
            data: { y: [3, 1, 4, 2], yh: [2.5, 1.4, 3.2, 2.9] },
            steps: [
              ['\\hat y_i', 'The predictions — one per example. The hat says "estimate".', d => 'ŷ = [' + d.yh.join(', ') + ']'],
              ['y_i - \\hat y_i', 'The residual: how wrong we were, with a sign.', d => 'r = [' + d.y.map((v, i) => (v - d.yh[i]).toFixed(1)).join(', ') + ']'],
              ['(y_i - \\hat y_i)^2', 'Square it: sign gone, big errors punished quadratically.', d => 'r² = [' + d.y.map((v, i) => ((v - d.yh[i]) ** 2).toFixed(2)).join(', ') + ']'],
              ['\\sum_{i=1}^{n}', 'Add the four squared errors into one number.', d => 'Σ = ' + d.y.reduce((a, v, i) => a + (v - d.yh[i]) ** 2, 0).toFixed(3)],
              ['\\frac{1}{n}\\sum', 'Divide by n so the loss does not grow just because the dataset did.', d => 'MSE = ' + (d.y.reduce((a, v, i) => a + (v - d.yh[i]) ** 2, 0) / d.y.length).toFixed(4)]
            ]
          },
          softmax: {
            name: 'Softmax',
            tex: '\\sigma(z)_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}',
            data: { z: [2.0, 1.0, 0.1] },
            steps: [
              ['z_k', 'The logits — raw scores. Any real number, no constraints.', d => 'z = [' + d.z.join(', ') + ']'],
              ['e^{z_k}', 'Exponentiate: everything becomes positive, and gaps become ratios.', d => 'e^z = [' + d.z.map(v => Math.exp(v).toFixed(3)).join(', ') + ']'],
              ['\\sum_j e^{z_j}', 'The normalising constant — the same number for every class.', d => 'Σ = ' + d.z.reduce((a, v) => a + Math.exp(v), 0).toFixed(4)],
              ['\\frac{e^{z_k}}{\\sum_j e^{z_j}}', 'Divide: now they are positive and sum to 1, so they can be read as probabilities.', d => { const s = d.z.reduce((a, v) => a + Math.exp(v), 0); return 'p = [' + d.z.map(v => (Math.exp(v) / s).toFixed(3)).join(', ') + ']'; }]
            ]
          },
          gd: {
            name: 'The gradient-descent update',
            tex: 'w_{t+1} = w_t - \\eta\\,\\nabla_w \\mathcal{L}(w_t)',
            data: { w: 3.0, eta: 0.3 },
            steps: [
              ['w_t', 'Where the parameters are right now.', d => 'w = ' + d.w.toFixed(3)],
              ['\\nabla_w \\mathcal{L}(w_t)', 'The uphill direction of the loss at that point. Here L = w², so ∇ = 2w.', d => '∇ = ' + (2 * d.w).toFixed(3)],
              ['-\\,\\nabla_w \\mathcal{L}', 'Negate: we want downhill, not uphill. This is the only reason for the minus sign.', d => '−∇ = ' + (-2 * d.w).toFixed(3)],
              ['\\eta\\,\\nabla_w \\mathcal{L}', 'Scale by the learning rate — how far to trust a local slope.', d => 'η∇ = ' + (d.eta * 2 * d.w).toFixed(3)],
              ['w_t - \\eta\\nabla', 'Step. Repeat until the gradient is small (§3.5).', d => "w' = " + (d.w - d.eta * 2 * d.w).toFixed(3)]
            ]
          },
          bayes: {
            name: 'Bayes’ theorem',
            tex: 'P(H\\mid E) = \\frac{P(E\\mid H)\\,P(H)}{P(E)}',
            data: { pH: 0.001, sens: 0.99, fpr: 0.05 },
            steps: [
              ['P(H)', 'The prior: how common the hypothesis is before any evidence.', d => 'P(H) = ' + d.pH],
              ['P(E\\mid H)', 'The likelihood: how well the hypothesis explains what we saw.', d => 'P(E|H) = ' + d.sens],
              ['P(E\\mid H)P(H)', 'Multiply: the weight of the "H is true and we saw E" world.', d => (d.sens * d.pH).toExponential(3)],
              ['P(E)', 'The evidence — every way E could have happened, true positives plus false positives.', d => (d.sens * d.pH + d.fpr * (1 - d.pH)).toFixed(5)],
              ['\\frac{P(E\\mid H)P(H)}{P(E)}', 'Divide: what share of the E-worlds are H-worlds. That share is the posterior.', d => (d.sens * d.pH / (d.sens * d.pH + d.fpr * (1 - d.pH)) * 100).toFixed(2) + '%']
            ]
          },
          attn: {
            name: 'Scaled dot-product attention',
            tex: '\\mathrm{Attn}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V',
            data: {},
            steps: [
              ['Q, K, V', 'Three projections of the same tokens: what I am looking for, what I advertise, what I hand over.', () => 'shapes (n × d_k), (n × d_k), (n × d_v)'],
              ['QK^\\top', 'Every query dotted with every key — an n × n table of "how relevant is j to i".', () => 'shape (n × n)'],
              ['\\frac{QK^\\top}{\\sqrt{d_k}}', 'Divide by √d_k so the scores do not grow with dimension and saturate the softmax (§4.3).', () => 'variance held at ≈1'],
              ['\\mathrm{softmax}(\\cdot)', 'Turn each row into weights that sum to 1 — a distribution over the other tokens.', () => 'each row sums to 1'],
              ['(\\cdot)V', 'Weighted average of the values. That is the entire operation.', () => 'shape (n × d_v)']
            ]
          }
        };

        const st = Viz.controls(host, [{
          k: 'f', label: 'formula', type: 'select', value: 'mse',
          options: Object.keys(FORMULAS).map(k => ({ v: k, t: FORMULAS[k].name }))
        }], () => build());

        const stage = el('div');
        host.appendChild(stage);

        function build() {
          stage.innerHTML = '';
          const F = FORMULAS[st.f];
          const eq = el('div', { style: 'text-align:center;padding:14px 6px;font-size:1.05em' });
          stage.appendChild(eq);
          const piece = el('div', {
            style: 'border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:13px 15px;min-height:96px'
          });
          stage.appendChild(piece);
          const player = el('div');
          stage.appendChild(player);

          function show(i) {
            const s = F.steps[i];
            eq.innerHTML = '$$' + F.tex + '$$';
            ML.typeset(eq);
            piece.innerHTML =
              '<p class="boxtitle" style="margin-bottom:8px">piece ' + (i + 1) + ' of ' + F.steps.length + '</p>' +
              '<p style="margin:0 0 8px;font-size:1.15em">$' + s[0] + '$</p>' +
              '<p style="margin:0 0 8px;font-family:var(--sans);font-size:14px">' + s[1] + '</p>' +
              '<p style="margin:0;font-family:var(--mono);font-size:12.5px;color:var(--blue)">' + s[2](F.data) + '</p>';
            ML.typeset(piece);
          }
          Viz.player(player, {
            frames: F.steps.length, fps: 0.7, repeat: true,
            label: i => 'piece ' + (i + 1) + ' / ' + F.steps.length,
            onFrame: show
          });
        }
        build();
        Viz.note(host, 'Press play and let it walk. The point is the last line of each step: an abstract mark, and the actual number it produced on these four examples, side by side.');
      }
    },
    quiz: [
      {
        q: '$\\arg\\min_w \\mathcal{L}(w)$ returns…',
        options: ['the smallest value the loss reaches', 'the parameters that achieve the smallest loss', 'the gradient at the minimum', 'the learning rate'],
        answer: 1,
        why: '$\\min$ gives the value; $\\arg\\min$ gives the argument. Training reports the second.'
      },
      {
        q: 'In $X w$ with $X$ of shape $n\\times d$, the vector $w$ must have length…',
        options: ['$n$', '$d$', '$n \\times d$', 'either'],
        answer: 1,
        why: 'The inner dimensions must match and then disappear: $(n\\times d)(d\\times 1) \\to (n \\times 1)$, one prediction per example.'
      },
      {
        q: 'Why can Bayes be written with $\\propto$ instead of dividing by $P(E)$?',
        options: ['because $P(E)$ is always 1', 'because $P(E)$ is the same for every hypothesis, so it cannot change which one wins', 'because the prior cancels', 'it cannot — that is an error'],
        answer: 1,
        why: 'The evidence is a normalising constant across hypotheses. Drop it while comparing, restore it when you need a calibrated number.'
      }
    ],
    cards: [
      { q: 'What does a hat mean?', a: 'An estimate: $\\hat y$ is our prediction of $y$, $\\hat\\theta$ our estimate of $\\theta$.' },
      { q: 'The three passes for reading a formula', a: 'Shapes → the loop (what index, what is constant inside it) → the story (what makes it big, small, zero, infinite).' },
      { q: '$\\mathbb{E}_{x\\sim p}[f(x)]$', a: 'The average of $f$ weighting each $x$ by $p(x)$ — "$f$ on average, as $p$ sees the world".' }
    ]
  });

  /* ------------------------------------------------------------------ 0.7 */
  ML.section({
    id: 'matrix-calculus', track: 'start', num: '0.7', level: 2,
    title: 'Matrix calculus without tears',
    lede: 'Backpropagation, the normal equations, ridge regression and the attention gradient are all one skill: differentiating an expression whose parts are vectors and matrices. There are about six results to know, and one rule for checking every one of them.',
    prereq: ['calculus-basics'],
    related: ['backprop', 'linear-algebra', 'optimization'],
    html: `
${H.tldr([
      'Pick the <b>denominator layout</b> and never change it: $\\partial f/\\partial w$ has <i>the same shape as $w$</i>. Then a gradient can always be subtracted from its parameter.',
      'Six results carry almost everything: $\\nabla_w a^\\top w = a$; $\\nabla_w w^\\top A w = (A+A^\\top)w$; $\\nabla_w \\|w\\|^2 = 2w$; $\\nabla_w \\|Xw-y\\|^2 = 2X^\\top(Xw-y)$; $\\partial(\\text{softmax+CE})/\\partial z = \\hat p - y$; and the chain rule as a product of Jacobians.',
      'Never trust a derivation you have not gradient-checked. The check is four lines and it is in the lab below.'
    ])}

<h2><span class="sn">0.7.1</span> Layout: the convention that removes the confusion</h2>
<p>The single thing that makes matrix calculus feel impossible is that two conventions are in circulation and textbooks rarely say which they are using. Choose <b>denominator layout</b>:</p>
${H.key('The derivative of a scalar with respect to $w$ has the same shape as $w$. Full stop.')}
<p>So if $\\mathcal{L}$ is a scalar loss and $w\\in\\mathbb{R}^d$, then $\\nabla_w\\mathcal{L}\\in\\mathbb{R}^d$. If $W\\in\\mathbb{R}^{m\\times n}$, then $\\partial\\mathcal{L}/\\partial W\\in\\mathbb{R}^{m\\times n}$. This is exactly what you need for the update $W \\leftarrow W - \\eta\\,\\partial\\mathcal{L}/\\partial W$ to typecheck — and it is what every deep learning framework returns.</p>
${H.table(['Function', 'Shapes', 'Derivative shape'], [
      ['scalar → scalar', '$f:\\mathbb{R}\\to\\mathbb{R}$', 'scalar'],
      ['vector → scalar', '$f:\\mathbb{R}^d\\to\\mathbb{R}$', '$d$ — <b>the gradient</b>'],
      ['vector → vector', '$f:\\mathbb{R}^n\\to\\mathbb{R}^m$', '$m\\times n$ — <b>the Jacobian</b>'],
      ['matrix → scalar', '$f:\\mathbb{R}^{m\\times n}\\to\\mathbb{R}$', '$m\\times n$'],
      ['vector → scalar, twice', '$f:\\mathbb{R}^d\\to\\mathbb{R}$', '$d\\times d$ — <b>the Hessian</b> (§1.12)']
    ])}
${H.note('A loss is always a scalar. That is not an accident: it is what makes "the gradient" a single object of the same shape as the parameters, and it is why multi-objective training always ends with a weighted sum.')}

<h2><span class="sn">0.7.2</span> The six results</h2>
<p>Each of these can be verified by writing out one component. Do that once for each; then use them freely for the rest of your life.</p>
${H.table(['Expression', 'Derivative w.r.t. $w$', 'Where it shows up'], [
      ['$a^\\top w$', '$a$', 'any linear score; the logit of a linear model'],
      ['$w^\\top A w$', '$(A + A^\\top)w$, and $2Aw$ if $A$ symmetric', 'quadratic forms, Newton steps, PCA'],
      ['$\\|w\\|_2^2 = w^\\top w$', '$2w$', 'L2 regularization → weight decay (§2.3)'],
      ['$\\|Xw - y\\|_2^2$', '$2X^\\top(Xw-y)$', 'linear regression, ridge, the normal equations'],
      ['$\\log\\big(\\sum_j e^{z_j}\\big)$', '$\\mathrm{softmax}(z)$', 'the log-partition function; why softmax appears at all'],
      ['softmax + cross-entropy', '$\\hat p - y$', 'the cleanest gradient in machine learning (§3.2)']
    ])}

${H.deriv('the least-squares gradient, and the normal equations that fall out of it', [
      ['$\\mathcal{L}(w) = \\|Xw-y\\|_2^2$', 'Start. $X$ is $n\\times d$, $w$ is $d$, so $Xw-y$ is $n$ and $\\mathcal{L}$ is a scalar — as a loss must be.'],
      ['$= (Xw-y)^\\top(Xw-y)$', 'Squared L2 norm <i>is</i> a dot product with itself. This is the move that turns a norm into algebra.'],
      ['$= w^\\top X^\\top X w - 2y^\\top Xw + y^\\top y$', 'Expand. The two cross terms are equal because each is a scalar, and a scalar equals its own transpose.'],
      ['$\\nabla_w = 2X^\\top X w - 2X^\\top y$', 'Term by term: rule 2 with $A = X^\\top X$ (symmetric), then rule 1 with $a = X^\\top y$. The constant $y^\\top y$ dies.'],
      ['$= 2X^\\top(Xw - y)$', 'Factor. Read it: <b>the design matrix transposed, applied to the residuals</b> — every feature is credited with its share of the error.'],
      ['$X^\\top X\\,\\hat w = X^\\top y$', 'Set the gradient to zero. These are the normal equations; adding $\\lambda\\|w\\|^2$ to the loss adds $\\lambda I$ here, which is ridge regression (§2.3) and is exactly why ridge fixes a singular $X^\\top X$.']
    ], 'The last line is worth memorising as a sentence: <i>ridge adds a positive constant to the diagonal, so the matrix is always invertible.</i> That single fact answers three separate interview questions — conditioning, collinearity, and why ridge has a closed form while lasso does not.')}

${H.lab('gradcheck', 'Gradient checker — analytic vs numerical', 'The analytic gradient is the formula above, implemented directly. The numerical gradient perturbs one coordinate at a time and takes a central difference. If your derivation is right, they agree to about 1e-8 in double precision. Break the formula with the toggle and watch the agreement collapse.')}

<h2><span class="sn">0.7.3</span> The chain rule is a product of Jacobians</h2>
<p>For $\\mathcal{L} = f(g(h(x)))$, the derivative is a product — and in the vector case, a product of matrices, applied right to left:</p>
$$\\frac{\\partial \\mathcal{L}}{\\partial x} = \\frac{\\partial \\mathcal{L}}{\\partial u}\\cdot\\frac{\\partial u}{\\partial v}\\cdot\\frac{\\partial v}{\\partial x}, \\qquad u = f(\\cdot),\\ v = h(x)$$
<p>Two facts make this practical, and together they <i>are</i> backpropagation (§3.2):</p>
${H.steps([
      '<b>Matrix products are associative, so the order of multiplication is free — but the cost is not.</b> Because the loss is a scalar, the leftmost factor is a row vector. Multiplying <b>left to right</b> keeps a vector at every stage; multiplying right to left builds full matrices. That is the entire difference between reverse-mode (cheap for many-in, one-out) and forward-mode autodiff (§3.11).',
      '<b>Each layer only needs to know how to turn "gradient of the loss w.r.t. my output" into "gradient w.r.t. my input and my parameters".</b> That is a local contract. It is why you can write a new layer without knowing anything about the rest of the network.'
    ])}

${H.svg('why reverse mode is the cheap direction', '0 0 640 190', `
<defs><style>
.n{fill:var(--panel);stroke:var(--line)}
.t{font:11px ui-sans-serif,system-ui;fill:var(--text);text-anchor:middle}
.s{font:10px ui-monospace,monospace;fill:var(--faint);text-anchor:middle}
.fw{stroke:var(--c1);fill:none;stroke-width:1.6;marker-end:url(#a1)}
.bw{stroke:var(--c2);fill:none;stroke-width:1.6;stroke-dasharray:4 3;marker-end:url(#a2)}
.cap{font:11px ui-sans-serif,system-ui}
</style>
<marker id="a1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--c1)"/></marker>
<marker id="a2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--c2)"/></marker>
</defs>
<g>
<rect class="n" x="18" y="52" width="86" height="40" rx="7"/><text class="t" x="61" y="70">x</text><text class="s" x="61" y="84">10⁶ params</text>
<rect class="n" x="158" y="52" width="86" height="40" rx="7"/><text class="t" x="201" y="70">h₁</text><text class="s" x="201" y="84">1024</text>
<rect class="n" x="298" y="52" width="86" height="40" rx="7"/><text class="t" x="341" y="70">h₂</text><text class="s" x="341" y="84">1024</text>
<rect class="n" x="438" y="52" width="86" height="40" rx="7"/><text class="t" x="481" y="70">ℒ</text><text class="s" x="481" y="84">1 scalar</text>
<path class="fw" d="M104 66 H154"/><path class="fw" d="M244 66 H294"/><path class="fw" d="M384 66 H434"/>
<path class="bw" d="M434 82 H388"/><path class="bw" d="M294 82 H248"/><path class="bw" d="M154 82 H108"/>
<text class="cap" x="18" y="128" fill="var(--c1)">forward →  compute activations, cache them</text>
<text class="cap" x="18" y="148" fill="var(--c2)">backward ←  one row vector per layer, never a full Jacobian</text>
<text class="cap" x="18" y="172" fill="var(--muted)">Many inputs, one output ⇒ start from the scalar end. Reverse mode costs about the same as one forward pass.</text>
</g>
`, 'The asymmetry is the whole reason training a billion-parameter model is affordable: the cost of the gradient is a constant multiple of the cost of the prediction, no matter how many parameters there are.')}

${H.more('Deriving the softmax + cross-entropy gradient (the one worth doing by hand once)', `
<p>Let $z$ be the logits, $\\hat p = \\mathrm{softmax}(z)$, and $\\mathcal{L} = -\\sum_k y_k \\log \\hat p_k$ with $y$ one-hot.</p>
${H.deriv('show that $\\partial\\mathcal{L}/\\partial z = \\hat p - y$', [
        ['$\\mathcal{L} = -\\sum_k y_k\\big(z_k - \\log\\sum_j e^{z_j}\\big)$', 'Substitute $\\log\\hat p_k = z_k - \\log\\sum_j e^{z_j}$. Writing softmax in log space first is the trick — it is also the numerically stable implementation (§1.15).'],
        ['$= -\\sum_k y_k z_k + \\log\\sum_j e^{z_j}$', 'The second term is constant in $k$, and $\\sum_k y_k = 1$ because $y$ is one-hot.'],
        ['$\\dfrac{\\partial}{\\partial z_m}\\Big(-\\sum_k y_k z_k\\Big) = -y_m$', 'Only the $k=m$ term survives.'],
        ['$\\dfrac{\\partial}{\\partial z_m}\\log\\sum_j e^{z_j} = \\dfrac{e^{z_m}}{\\sum_j e^{z_j}} = \\hat p_m$', 'The derivative of the log-sum-exp <i>is</i> the softmax. This is result five in the table, and it is why softmax and cross-entropy are married.'],
        ['$\\dfrac{\\partial\\mathcal{L}}{\\partial z_m} = \\hat p_m - y_m$', 'Add. The gradient is literally "how wrong the probability was" — no Jacobian of the softmax ever needs to be formed.']
      ])}
<p>Two consequences that get asked about. First, the softmax Jacobian $\\mathrm{diag}(\\hat p) - \\hat p\\hat p^\\top$ is <i>never materialised</i> in practice, because it cancels against the cross-entropy derivative — which is why frameworks fuse the two into one op. Second, the gradient is bounded in $[-1,1]$, so cross-entropy on logits cannot explode the way a naive squared loss on probabilities can.</p>
`)}

<h2><span class="sn">0.7.4</span> Always gradient-check</h2>
${H.code(`def grad_check(f, w, analytic, eps=1e-5):
    """Central differences. Agreement to ~1e-8 in float64 means the
    derivation is right; ~1e-3 usually means a transpose is missing."""
    num = np.zeros_like(w)
    for i in range(w.size):
        p = w.copy(); p.flat[i] += eps
        m = w.copy(); m.flat[i] -= eps
        num.flat[i] = (f(p) - f(m)) / (2 * eps)
    rel = np.abs(num - analytic) / (np.abs(num) + np.abs(analytic) + 1e-12)
    return rel.max()`)}
${H.pitfall('Use <b>central</b> differences, not forward. The forward difference has error $O(\\varepsilon)$; the central difference has error $O(\\varepsilon^2)$, which buys you four extra digits for one extra function evaluation. And check in float64 — in float32 the rounding noise swamps the signal and every correct gradient looks broken.')}

${H.probe([
      ['What shape is $\\nabla_W \\mathcal{L}$ for a weight matrix $W$?', 'The same shape as $W$. Otherwise you could not subtract it during the update.'],
      ['Why is reverse-mode differentiation the right choice for neural networks?', 'Many parameters in, one scalar out. Reverse mode costs one gradient per <i>output</i>; forward mode costs one per <i>input</i>. With 10⁹ inputs and 1 output the choice is not close (§3.11).'],
      ['Derive $\\nabla_w\\|Xw-y\\|^2$.', '$2X^\\top(Xw-y)$ — expand into a quadratic form, apply $\\nabla w^\\top Aw = 2Aw$ for symmetric $A=X^\\top X$ and $\\nabla a^\\top w = a$, then factor.'],
      ['Your gradient check gives a relative error of 0.3. What is the most likely bug?', 'A missing or extra transpose, or a sum over the wrong axis. Errors around 1e-2 to 1e-1 are structural; errors around 1e-6 in float32 are just precision.']
    ], 'Reciting "the derivative of $w^\\top A w$ is $2Aw$" without the symmetry condition. For general $A$ it is $(A+A^\\top)w$, and interviewers who know the difference are checking whether you do.')}`,
    labs: {
      gradcheck: function (host) {
        const R = Num.rng(17);
        const n = 40, d = 5;
        const X = Array.from({ length: n }, () => Array.from({ length: d }, () => R.normal(0, 1)));
        const wTrue = [1.5, -0.8, 0.3, 2.0, -1.2];
        const y = X.map(r => Num.dot(r, wTrue) + R.normal(0, 0.3));

        const st = Viz.controls(host, [
          { k: 'expr', label: 'expression', type: 'select', value: 'ls', options: [
            { v: 'ls', t: '‖Xw − y‖²  (least squares)' },
            { v: 'ridge', t: '‖Xw − y‖² + λ‖w‖²  (ridge)' },
            { v: 'quad', t: 'wᵀAw  with A non-symmetric' },
            { v: 'logreg', t: 'logistic cross-entropy' }
          ] },
          { k: 'lam', label: 'λ (ridge only)', min: 0, max: 5, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'bug', label: 'introduce the classic bug', type: 'toggle', value: false },
          { k: 'eps', label: 'ε for the difference', min: -9, max: -2, step: 1, value: -5, fmt: v => '1e' + v }
        ], () => S.redraw());

        const out = Viz.readout(host, [
          { k: 'rel', label: 'max relative error', cls: 'key' },
          { k: 'verdict', label: 'verdict' },
          { k: 'gn', label: '‖analytic‖' },
          { k: 'note', label: 'the bug' }
        ]);

        const A = [[2, 1, 0, 0, 0], [0, 3, 1, 0, 0], [0, 0, 1, 2, 0], [0, 0, 0, 4, 1], [1, 0, 0, 0, 2]];

        function value(w) {
          if (st.expr === 'quad') { const Aw = Num.matvec(A, w); return Num.dot(w, Aw); }
          if (st.expr === 'logreg') {
            let L = 0;
            for (let i = 0; i < n; i++) {
              const z = Num.dot(X[i], w), p = Num.sigmoid(z), t = y[i] > 0 ? 1 : 0;
              L += -(t * Math.log(Math.max(1e-12, p)) + (1 - t) * Math.log(Math.max(1e-12, 1 - p)));
            }
            return L / n;
          }
          let s = 0;
          for (let i = 0; i < n; i++) { const r = Num.dot(X[i], w) - y[i]; s += r * r; }
          if (st.expr === 'ridge') s += st.lam * Num.dot(w, w);
          return s;
        }

        function analytic(w) {
          const g = new Array(d).fill(0);
          if (st.expr === 'quad') {
            // correct: (A + Aᵀ)w. bug: 2Aw, which is only right for symmetric A
            const Aw = Num.matvec(A, w), Atw = Num.matvec(Num.transpose(A), w);
            for (let j = 0; j < d; j++) g[j] = st.bug ? 2 * Aw[j] : Aw[j] + Atw[j];
            return g;
          }
          if (st.expr === 'logreg') {
            for (let i = 0; i < n; i++) {
              const p = Num.sigmoid(Num.dot(X[i], w)), t = y[i] > 0 ? 1 : 0;
              // bug: forget to divide by n
              for (let j = 0; j < d; j++) g[j] += (p - t) * X[i][j] / (st.bug ? 1 : n);
            }
            return g;
          }
          for (let i = 0; i < n; i++) {
            const r = Num.dot(X[i], w) - y[i];
            for (let j = 0; j < d; j++) g[j] += 2 * r * X[i][j];
          }
          // bug: forget the ridge term's derivative
          if (st.expr === 'ridge' && !st.bug) for (let j = 0; j < d; j++) g[j] += 2 * st.lam * w[j];
          if (st.expr === 'ls' && st.bug) for (let j = 0; j < d; j++) g[j] *= 0.5;   // dropped the 2
          return g;
        }

        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w2, h, T) {
            const w = [0.6, -0.4, 1.1, 0.2, -0.9];
            const eps = Math.pow(10, st.eps);
            const ana = analytic(w);
            const num = w.map((_, j) => {
              const a = w.slice(), b = w.slice();
              a[j] += eps; b[j] -= eps;
              return (value(a) - value(b)) / (2 * eps);
            });
            let rel = 0;
            for (let j = 0; j < d; j++) {
              const r = Math.abs(num[j] - ana[j]) / (Math.abs(num[j]) + Math.abs(ana[j]) + 1e-12);
              if (r > rel) rel = r;
            }
            const P = Viz.plot(ctx, w2, h, {
              xd: [-0.5, d - 0.5],
              yd: [Math.min(0, Math.min.apply(null, ana.concat(num))) * 1.25, Math.max(0.1, Math.max.apply(null, ana.concat(num))) * 1.25],
              pad: { l: 52, r: 14, t: 16, b: 40 }
            }).frame({ xticks: ana.map((_, j) => j), xlabel: 'coordinate of w', ylabel: '∂L / ∂wⱼ' });
            P.clip(() => {
              ana.forEach((v, j) => {
                const bw = P.pw / d * 0.3;
                ctx.fillStyle = T.c1; ctx.globalAlpha = .85;
                ctx.fillRect(P.x(j) - bw, Math.min(P.y(0), P.y(v)), bw, Math.abs(P.y(v) - P.y(0)));
                ctx.fillStyle = T.c2;
                ctx.fillRect(P.x(j), Math.min(P.y(0), P.y(num[j])), bw, Math.abs(P.y(num[j]) - P.y(0)));
                ctx.globalAlpha = 1;
              });
              P.hline(0, { color: T.faint, dash: false, width: 1 });
            });
            const bugs = {
              ls: 'dropped the factor of 2', ridge: 'forgot the λ term', quad: 'used 2Aw for non-symmetric A', logreg: 'forgot to divide by n'
            };
            out({
              rel: rel < 1e-4 ? rel.toExponential(1) : rel.toFixed(4),
              verdict: rel < 1e-6 ? '✓ matches' : rel < 1e-3 ? '~ precision only' : '✗ wrong',
              gn: Num.norm(ana).toFixed(3),
              note: st.bug ? bugs[st.expr] : 'none'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: 'analytic (your derivation)' }, { c: 'var(--c2)', t: 'numerical (central difference)' }]);
        Viz.note(host, 'Now drag ε. Too large and the truncation error dominates; too small and floating-point cancellation does — the relative error is a U-shape with its floor near 1e-5 for a well-scaled problem. This U is the same phenomenon as §1.15’s catastrophic cancellation, and it is why nobody trains with numerical gradients.');
      }
    },
    quiz: [
      {
        q: 'In denominator layout, $\\partial \\mathcal{L}/\\partial W$ for $W\\in\\mathbb{R}^{m\\times n}$ has shape…',
        options: ['$n\\times m$', '$m\\times n$', '$mn \\times 1$', 'scalar'],
        answer: 1,
        why: 'Same shape as the parameter, so that $W \\leftarrow W - \\eta\\,\\partial\\mathcal{L}/\\partial W$ is well defined.'
      },
      {
        q: '$\\nabla_w\\, w^\\top A w$ equals $2Aw$ only when…',
        options: ['$A$ is invertible', '$A$ is symmetric', '$w$ is a unit vector', 'always'],
        answer: 1,
        why: 'In general it is $(A+A^\\top)w$. Symmetry makes the two terms equal.'
      },
      {
        q: 'The gradient of softmax + cross-entropy with respect to the logits is…',
        options: ['$\\hat p(1-\\hat p)$', '$\\hat p - y$', '$-y/\\hat p$', 'the softmax Jacobian times $y$'],
        answer: 1,
        why: 'The softmax Jacobian cancels against the cross-entropy derivative, leaving prediction minus target. Frameworks fuse the two ops for exactly this reason.'
      },
      {
        q: 'Reverse-mode autodiff is preferred for training because…',
        options: ['it is more accurate', 'the cost scales with the number of outputs, and a loss has exactly one', 'it uses less memory than forward mode', 'it works for non-differentiable functions'],
        answer: 1,
        why: 'Reverse mode costs one sweep per output; forward mode one per input. Many parameters, one scalar loss — reverse wins by a factor of the parameter count. It uses *more* memory, since activations must be cached (§3.11).'
      }
    ],
    cards: [
      { q: 'Denominator layout, in one line', a: 'The derivative of a scalar w.r.t. $w$ has the same shape as $w$ — so it can be subtracted from it.' },
      { q: '$\\nabla_w \\|Xw-y\\|^2$', a: '$2X^\\top(Xw-y)$: the design matrix transposed, applied to the residuals.' },
      { q: 'Normal equations, and what ridge does to them', a: '$X^\\top X\\hat w = X^\\top y$; ridge makes it $(X^\\top X+\\lambda I)\\hat w = X^\\top y$, always invertible.' },
      { q: 'Why fuse softmax and cross-entropy?', a: 'The gradient collapses to $\\hat p - y$; computing them separately materialises a Jacobian that immediately cancels, and loses numerical stability.' },
      { q: 'Gradient check, correctly done', a: 'Central differences, float64, ε ≈ 1e-5, compare relative error. Below 1e-6 is right; above 1e-3 is a bug, usually a transpose.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.8 */
  ML.section({
    id: 'first-model', track: 'start', num: '0.8', level: 1,
    title: 'Your first model, end to end',
    lede: 'Every idea in Part 2 is an answer to a problem that shows up in this one workflow. Build it once here, badly and then properly, and the rest of the site has somewhere to attach.',
    prereq: ['what-is-ml'],
    related: ['supervised-setup', 'metrics', 'features'],
    html: `
${H.tldr([
      'The workflow is: <b>frame → split → baseline → features → fit → evaluate → threshold → ship → watch</b>. Skipping "baseline" and "threshold" is the most common way to waste a month.',
      'The split comes <b>before</b> anything that looks at the data. Scaling, imputation and encoding are all fitted on train and only applied to test — otherwise you are measuring a model that has already seen the answers.',
      'A model is not finished when the loss stops falling. It is finished when a <i>decision threshold</i> has been chosen against a cost, and someone has agreed what "worse" would look like in production.'
    ])}

<h2><span class="sn">0.8.1</span> The nine steps, and what each one is defending against</h2>
${H.table(['Step', 'What you do', 'What goes wrong if you skip it'], [
      ['1. Frame', 'Write the decision the model informs, and the cost of each kind of mistake, in one sentence each.', 'You optimise accuracy on a problem where a false negative costs 40× a false positive (§2.13).'],
      ['2. Split', 'Hold out test data <i>first</i>, by time if the future is what you predict (§2.14).', 'Every later number is optimistic and you find out in production.'],
      ['3. Baseline', 'Predict the majority class, or last week’s value. Record the score.', 'You have no idea whether your 0.84 AUC is good or embarrassing.'],
      ['4. Features', 'Encode, impute, scale — all <b>fitted on train only</b> (§2.11).', 'Leakage. The most common cause of a model that is brilliant offline and useless live.'],
      ['5. Fit', 'Start with logistic regression or gradient boosting. Not a neural network.', 'You spend three weeks on the wrong axis of the problem.'],
      ['6. Evaluate', 'On the metric from step 1, with an interval (§1.6), against the baseline from step 3.', 'You ship noise.'],
      ['7. Threshold', 'Pick the operating point that minimises expected cost, not 0.5 (§2.12).', 'A well-calibrated model making bad decisions.'],
      ['8. Ship', 'Shadow, then canary, then full (§5.12).', 'Your first real user is your test suite.'],
      ['9. Watch', 'Monitor input drift and outcome drift separately (§2.18).', 'Silent decay — the model keeps answering confidently while the world moves.']
    ])}

${H.lab('pipeline', 'The pipeline sandbox — break it on purpose', 'A real logistic regression on generated customer data, retrained every time you flip a switch. Each toggle is a specific mistake from the table above. Watch the gap between what you would have reported and what the model actually does on held-out data.')}

<h2><span class="sn">0.8.2</span> The whole thing in twenty lines</h2>
${H.code(`import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=0)

num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))])

# The Pipeline is not a style preference. It is what makes "fit on train only"
# structurally impossible to get wrong.
pipe = Pipeline([
    ("prep", ColumnTransformer([("n", num, num_cols), ("c", cat, cat_cols)])),
    ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X_tr, y_tr)

p = pipe.predict_proba(X_te)[:, 1]
print("AUC", roc_auc_score(y_te, p), "AP", average_precision_score(y_te, p))
print("baseline AP", y_te.mean())          # always print the baseline`)}
${H.key('Use a Pipeline object, not a sequence of transformations you remember to repeat. Leakage is not usually a conceptual error — it is a bookkeeping error, and the fix is structural.')}

${H.intuition(`<p>Why does fitting the scaler on all the data leak? Because the mean and standard deviation of the test set are facts about the test set. A model whose inputs were centred using the test mean has been told something about the test distribution that it will not know at prediction time. The effect is usually small — a fraction of a point of AUC — which is exactly what makes it dangerous: it is big enough to change which model you pick and small enough to never look suspicious.</p>`)}

<h2><span class="sn">0.8.3</span> The three numbers to report, always</h2>
${H.steps([
      '<b>The baseline.</b> Majority class, or the current rule-based system. Without it, no number means anything.',
      '<b>The metric that matches the cost</b>, with an interval. Average precision if positives are rare; recall at a fixed alert budget if a team has to work the alerts; calibrated probability if a downstream system multiplies by a monetary amount (§2.13).',
      '<b>The decision.</b> At the chosen threshold: how many alerts per day, what fraction are right, and what the expected cost is versus the baseline. This is the only one your stakeholder actually needed.'
    ])}
${H.pitfall('Reporting accuracy on an imbalanced problem. At 1% positives, a model that predicts "no" forever is 99% accurate and worth nothing. If you catch yourself reporting accuracy, ask what the base rate is — and then report average precision or recall at a fixed budget instead.')}

${H.probe([
      ['A colleague scaled the features before splitting. How bad is it?', 'Usually a small optimistic bias, occasionally a large one — but the real problem is that you can no longer trust the comparison between models. Refit inside a Pipeline and re-run.'],
      ['What is your first model on a new tabular problem?', 'A baseline, then logistic regression with sensible encoding, then gradient boosting. Neural networks rarely win on tabular data and cost far more to debug (§2.8).'],
      ['You have 0.91 AUC. Ship it?', 'Not yet. What is the baseline, what is the operating threshold, what does a false positive cost, and is the probability calibrated if anything downstream multiplies by it?']
    ])}`,
    labs: {
      pipeline: function (host) {
        const st = Viz.controls(host, [
          { k: 'leak', label: 'scale using ALL the data (leak)', type: 'toggle', value: false },
          { k: 'target', label: 'keep a feature built from the label', type: 'toggle', value: false },
          { k: 'thr', label: 'decision threshold', min: .05, max: .95, step: .01, value: .5, fmt: v => v.toFixed(2) },
          { k: 'cost', label: 'cost of a miss ÷ cost of a false alarm', min: 1, max: 40, step: 1, value: 10, fmt: v => v + '×' }
        ], () => S.redraw());

        const out = Viz.readout(host, [
          { k: 'auc', label: 'test AUC', cls: 'key' },
          { k: 'ap', label: 'test AP' },
          { k: 'base', label: 'baseline AP' },
          { k: 'rep', label: 'AUC you’d report', cls: 'warn' },
          { k: 'alerts', label: 'alerts / 1000' },
          { k: 'prec', label: 'precision there' },
          { k: 'rec', label: 'recall there' },
          { k: 'exp', label: 'expected cost', cls: 'bad' }
        ]);

        /* a small credit-risk-shaped dataset: 4 honest features, one poisoned */
        function makeData(n, seed) {
          const R = Num.rng(seed);
          const X = [], y = [];
          for (let i = 0; i < n; i++) {
            const income = R.normal(60, 22);
            const util = Math.min(1.4, Math.max(0, R.beta(2, 3) * 1.2));
            const age = 22 + R.gamma(4) * 6;
            const inq = R.poisson(1.2);
            const z = -2.4 - 0.028 * (income - 60) + 2.6 * util + 0.30 * inq - 0.02 * (age - 40);
            const p = Num.sigmoid(z);
            const label = R() < p ? 1 : 0;
            // the poisoned feature: "days since the default letter was posted"
            const leaky = label ? R.normal(9, 3) : R.normal(60, 20);
            X.push([income, util, age, inq, leaky]);
            y.push(label);
          }
          return { X: X, y: y };
        }
        const tr = makeData(700, 3), te = makeData(700, 99);

        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const cols = st.target ? 5 : 4;
            const cut = arr => arr.map(r => r.slice(0, cols));
            const Xtr = cut(tr.X), Xte = cut(te.X);

            // scaling statistics: honest (train only) or leaked (train+test)
            const pool = st.leak ? Xtr.concat(Xte) : Xtr;
            const mu = [], sd = [];
            for (let j = 0; j < cols; j++) {
              const col = pool.map(r => r[j]);
              mu.push(Num.mean(col)); sd.push(Num.sd(col) || 1);
            }
            const scale = M => M.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
            const Ztr = scale(Xtr), Zte = scale(Xte);

            const model = Num.logistic(Ztr, tr.y, { lr: .35, l2: 1e-3 });
            model.step(600);

            const ptr = Ztr.map(r => model.predict(r));
            const pte = Zte.map(r => model.predict(r));
            const rocTe = Num.rocCurve(pte, te.y);
            const rocTr = Num.rocCurve(ptr, tr.y);
            const baseAP = Num.mean(te.y);

            const cm = Num.confusion(pte, te.y, st.thr);
            const missCost = st.cost, faCost = 1;
            const expected = (cm.fn * missCost + cm.fp * faCost) / te.y.length;

            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 48, r: 16, t: 16, b: 40 } })
              .frame({ xlabel: 'false positive rate', ylabel: 'true positive rate' });
            P.clip(() => {
              P.line([[0, 0], [1, 1]], { color: T.faint, dash: [4, 4], width: 1 });
              P.line(rocTr.roc, { color: T.faint, width: 1.6, dash: [5, 3] });
              P.area(rocTe.roc, { color: T.c1, alpha: .12 });
              P.line(rocTe.roc, { color: T.c1, width: 2.6 });
              const fpr = cm.fp / (cm.fp + cm.tn || 1), tpr = cm.tp / (cm.tp + cm.fn || 1);
              P.dots([[fpr, tpr]], { r: 5.5, color: T.c2, stroke: true, strokeWidth: 2 });
              P.text(fpr, tpr, ' operating point', { dx: 8, color: T.c2, font: '11px ui-sans-serif' });
            });

            out({
              auc: rocTe.auc.toFixed(3),
              ap: rocTe.ap.toFixed(3),
              base: baseAP.toFixed(3),
              rep: (st.leak || st.target ? rocTr.auc : rocTe.auc).toFixed(3),
              alerts: Math.round(1000 * (cm.tp + cm.fp) / te.y.length),
              prec: (cm.precision * 100).toFixed(1) + '%',
              rec: (cm.recall * 100).toFixed(1) + '%',
              exp: expected.toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c1)', t: 'held-out test ROC' },
          { c: 'var(--faint)', t: 'training ROC (what leakage lets you believe)' },
          { c: 'var(--c2)', t: 'your chosen operating point' }
        ]);
        Viz.note(host, 'Turn on <b>keep a feature built from the label</b>: AUC jumps toward 1.0 and the model is worthless, because at prediction time nobody has posted the default letter yet. Then leave it off and drag the threshold — notice that the AUC never moves, because AUC is threshold-free, while expected cost changes by a factor of three. <b>AUC ranks; thresholds decide.</b> The cost-optimal threshold sits near $1/(1+\\text{cost ratio})$, so at 10× it is about 0.09, not 0.5.');
      }
    },
    quiz: [
      {
        q: 'You fit a StandardScaler on the full dataset before splitting. This is…',
        options: ['fine, scaling is not learning', 'leakage — the test distribution has informed the transform', 'only a problem for neural networks', 'good practice for stability'],
        answer: 1,
        why: 'The mean and variance are estimated from data you are about to score on. The bias is small, consistent, and just big enough to change which model you pick.'
      },
      {
        q: 'A fraud model has AUC 0.94 and you must choose a threshold. A miss costs 20× a false alarm. Roughly where does the cost-optimal threshold sit?',
        options: ['0.5, always', 'near 0.05', 'near 0.95', 'thresholds do not affect cost'],
        answer: 1,
        why: 'Optimal threshold ≈ cost(FP)/(cost(FP)+cost(FN)) = 1/21 ≈ 0.048. Expensive misses push the threshold down, so you alert more often.'
      },
      {
        q: 'Which number should always accompany a reported metric?',
        options: ['the training loss', 'the baseline that metric is being compared against', 'the number of epochs', 'the random seed'],
        answer: 1,
        why: 'Without a baseline, 0.84 is a number, not a result. The seed matters too, but the baseline is what makes the claim interpretable.'
      }
    ],
    cards: [
      { q: 'The nine steps', a: 'Frame → split → baseline → features → fit → evaluate → threshold → ship → watch.' },
      { q: 'Why use a Pipeline object?', a: 'Leakage is a bookkeeping error; a Pipeline makes "fit on train only" structural rather than remembered.' },
      { q: 'Cost-optimal threshold', a: '$t^\\star \\approx c_{FP}/(c_{FP}+c_{FN})$ — expensive misses drive the threshold down.' },
      { q: 'Why is accuracy the wrong metric at 1% positives?', a: 'Predicting "no" forever scores 99%. Use average precision, or recall at a fixed alert budget.' }
    ]
  });
})();
