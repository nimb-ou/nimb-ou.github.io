/* ============================================================
   PART 0 — Start here
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 0.1 */
  ML.section({
    id: 'what-is-ml', track: 'start', num: '0.1',
    title: 'What machine learning actually is',
    lede: 'A model is a function with adjustable numbers inside it, a loss that scores how wrong the function is, and a procedure that pushes the numbers downhill. Everything else on this site is a variation on those three objects.',
    html: `
<p>Ordinary programming is: you know the rule, you write the rule, the computer applies it. Machine learning is for the case where <b>you can recognise the answer but cannot write the rule</b> — is this transaction fraudulent, what is the next word, which applicant repays. You supply examples; the machine searches a space of candidate rules for one that reproduces your examples and, crucially, keeps working on examples it has never seen.</p>

<p>Three objects, and you will meet them in every single section that follows:</p>

${H.table(['Object', 'Symbol', 'What it is'], [
      ['<b>Model</b>', '$f_\\theta(x)$', 'A function from input $x$ to prediction, with parameters $\\theta$ you are free to choose. A line has two. A frontier LLM has hundreds of billions.'],
      ['<b>Loss</b>', '$\\ell(f_\\theta(x), y)$', 'A number saying how bad one prediction was. Averaged over data it becomes the objective you minimise. It is not chosen by taste — §1.5 shows it falls out of the noise model you assume.'],
      ['<b>Optimiser</b>', '$\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta L$', 'The procedure that changes $\\theta$ to reduce the loss. Gradient descent and its descendants; §1.9 and §3.4.']
    ])}

<h2><span class="sn">0.1.1</span> The only distinction that matters at the start</h2>
<p><b>Supervised learning</b> has labelled pairs $(x, y)$ and learns $x \\mapsto y$: spam or not, price, next token. <b>Unsupervised learning</b> has only $x$ and looks for structure: clusters, low-dimensional directions, densities, anomalies. <b>Reinforcement learning</b> has neither — it has an environment that returns rewards, and learns a policy that maximises reward over time (§6.1).</p>
<p>Almost all commercial value today is supervised, including the part that does not look it: next-token prediction is supervised learning where the label is <i>the next token in the text you already had</i>. That single trick is why the internet is a training set.</p>

${H.key('The goal is never to fit the data you have. It is to fit the data you do not have — and the whole of Part 1 exists to explain why that is even possible.')}

<h2><span class="sn">0.1.2</span> See it happen</h2>
<p>Below is the entire loop, running for real. Drag any point; the line refits by least squares on every frame. Add points in the empty space and watch the fit move. Then turn on <i>train/test split</i> and notice the thing that matters: the line is chosen using the blue points only, and it is judged on the red ones.</p>

${H.lab('fit', 'The whole loop, in one picture', 'Drag points. Shift-click empty space to add one, and click a point to delete it. The line is the exact least-squares solution recomputed each frame — no animation trickery.')}

<h2><span class="sn">0.1.3</span> Why "learning" is a fair word</h2>
<p>Nothing in that demo was told what a line is <i>for</i>. It was given a family of candidate rules (all lines), a way to score a rule (squared error), and a rule for improving the score (the normal equations, §2.4). What emerged was a rule nobody typed. Swap the family for a hundred-layer network and the score for cross-entropy and you have modern deep learning; the loop above does not change.</p>

${H.probe([
      ['What is machine learning in one sentence?', 'Searching a parameterised family of functions for the one that minimises expected loss on data drawn from the same distribution you will be scored on.'],
      ['Why can a model be good on training data and useless in production?', 'It minimised <i>empirical</i> risk, not expected risk; the gap between them is what concentration (§1.4), validation (§2.14) and regularization (§2.3) are all about.']
    ], 'Saying "the model learns patterns". Every interviewer hears "I have not thought about the objective".')}

<h2><span class="sn">0.1.4</span> The map of the rest</h2>
${H.table(['If you want to…', 'Go to'], [
      ['Understand why any of this generalises', '<a href="#/concentration">§1.4 concentration</a>'],
      ['Know where losses come from', '<a href="#/mle-map">§1.5 MLE and MAP</a>'],
      ['Fit and defend a tabular model', '<a href="#/boosting">§2.8 gradient boosting</a> and <a href="#/metrics">§2.13 metrics</a>'],
      ['Understand a transformer', '<a href="#/attention">§4.3 self-attention</a>'],
      ['Build something with an LLM', '<a href="#/rag">§5.1 RAG</a> and <a href="#/decision-ladder">§5.13 the ladder</a>']
    ])}`,
    labs: {
      fit: function (host) {
        const R = Num.rng(4);
        let pts = Array.from({ length: 14 }, (_, i) => {
          const x = -2.6 + 5.2 * (i + R() * .5) / 14;
          return { x: x, y: .85 * x + .4 + R.normal(0, .55), test: i % 4 === 3 };
        });
        const st = Viz.controls(host, [
          { k: 'split', label: 'train / test split', type: 'toggle', value: false },
          { k: 'noise', label: 'add noisy point', type: 'buttons', value: '', options: [{ v: '', t: '—' }] }
        ], () => S.redraw());
        host.querySelector('.controls').lastChild.remove();

        const out = Viz.readout(host, [
          { k: 'w', label: 'slope w' }, { k: 'b', label: 'intercept b' },
          { k: 'tr', label: 'train MSE', cls: 'key' }, { k: 'te', label: 'test MSE' }
        ]);

        let drag = -1;
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-3.4, 3.4] }).frame({ xlabel: 'x — the feature', ylabel: 'y — the label' });
            const tr = pts.filter(p => !(st.split && p.test));
            const te = pts.filter(p => st.split && p.test);
            const X = Num.polyDesign(tr.map(p => p.x), 1);
            const beta = tr.length > 1 ? Num.ridgeFit(X, tr.map(p => p.y), 1e-9) : [0, 0];
            P.clip(() => P.fn(x => beta[0] + beta[1] * x, { color: T.text, width: 2.4 }));
            // residuals
            tr.forEach(p => {
              const yh = beta[0] + beta[1] * p.x;
              P.line([[p.x, p.y], [p.x, yh]], { color: T.faint, width: 1, dash: [3, 3] });
            });
            P.dots(tr.map(p => [p.x, p.y]), { r: 5, color: T.blue, stroke: true });
            P.dots(te.map(p => [p.x, p.y]), { r: 5, color: T.red, stroke: true });
            const mse = arr => arr.length ? Num.mean(arr.map(p => (p.y - (beta[0] + beta[1] * p.x)) ** 2)) : 0;
            out({
              w: beta[1].toFixed(3), b: beta[0].toFixed(3),
              tr: mse(tr).toFixed(3), te: te.length ? mse(te).toFixed(3) : '—'
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const dx = P.ix(e.x), dy = P.iy(e.y);
          if (e.type === 'down') {
            let best = -1, bd = 1e9;
            pts.forEach((p, i) => { const d = (p.x - dx) ** 2 + (p.y - dy) ** 2; if (d < bd) { bd = d; best = i; } });
            if (bd < .06) {
              if (e.shiftKey) { pts.splice(best, 1); drag = -1; }
              else drag = best;
            } else { pts.push({ x: dx, y: dy, test: false }); drag = pts.length - 1; }
            S.redraw();
          } else if (e.type === 'move' && e.down && drag >= 0) {
            pts[drag].x = dx; pts[drag].y = dy; S.redraw();
          } else if (e.type === 'up') drag = -1;
        });
        Viz.buttons(host, [
          { label: 'Reseed data', on: () => { const R2 = Num.rng(Math.floor(Math.random() * 1e6)); pts = Array.from({ length: 14 }, (_, i) => { const x = -2.6 + 5.2 * (i + R2() * .5) / 14; return { x: x, y: .85 * x + .4 + R2.normal(0, .55), test: i % 4 === 3 }; }); S.redraw(); } },
          { label: 'Add an outlier', on: () => { pts.push({ x: 2.6, y: -2.6, test: false }); S.redraw(); } }
        ]);
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'training points (fit uses these)' }, { c: Viz.theme().red, t: 'held-out points (judged on these)' }, { c: Viz.theme().text, t: 'least-squares fit' }]);
      }
    },
    quiz: [
      {
        q: 'A model scores 0.99 on the data it was fitted to and 0.61 on new data. Which of the three objects is the problem?',
        options: ['The optimiser failed to converge', 'The model family is too flexible for the amount of data', 'The loss was computed incorrectly', 'The data was not shuffled'],
        answer: 1,
        why: 'A large train–test gap is the signature of variance: the family could fit the noise, and it did. More data, a simpler family, or regularization — §2.2 and §2.3.'
      },
      {
        q: 'Next-token prediction on internet text is best described as…',
        options: ['unsupervised learning', 'supervised learning where the label is free', 'reinforcement learning', 'semi-supervised learning'],
        answer: 1,
        why: 'The label is the next token, which is already in the corpus. That is why the objective scales: no annotation cost. (It is often called "self-supervised" for exactly this reason.)'
      }
    ],
    cards: [
      { q: 'The three objects of every ML system', a: 'A parameterised model $f_\\theta$, a loss $\\ell$, and an optimiser that reduces the loss.' },
      { q: 'The goal of learning, stated precisely', a: 'Minimise <i>expected</i> loss on unseen data drawn from the same distribution — not empirical loss on the training set.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.2 */
  ML.section({
    id: 'linear-algebra-basics', track: 'start', num: '0.2',
    title: 'Vectors and matrices, from zero',
    lede: 'A vector is a list of numbers and an arrow. A matrix is a function that moves arrows. Once you believe both sentences, PCA, attention and every neural layer stop being notation and become geometry.',
    html: `
<h2><span class="sn">0.2.1</span> A vector is data and direction at once</h2>
<p>An example with three features — utilisation 0.94, tenure 9 years, enquiries 5 — <i>is</i> the vector $x = (0.94,\\, 9,\\, 5)$. Nothing more mystical. Two operations do almost all the work:</p>
<p><b>Addition</b> $a + b = (a_1+b_1, \\ldots)$ places arrows tip to tail. <b>Scaling</b> $\\alpha a$ stretches. Together they define everything linear: a <i>linear combination</i> $\\alpha a + \\beta b$ is the set of places you can reach, and that set is a line, a plane, or a higher-dimensional flat thing through the origin.</p>

<h3>The dot product is the whole of similarity</h3>
$$a \\cdot b = \\sum_i a_i b_i = \\|a\\|\\,\\|b\\|\\cos\\theta$$
<p>Read the second form: the dot product is <b>length times length times alignment</b>. It is zero exactly when the arrows are perpendicular. Divide it by the two lengths and you get cosine similarity — the number a vector database sorts by (§5.1), the number attention computes between a query and a key (§4.3), and the number a linear model computes between weights and features before squashing it (§2.4). One operation, three chapters.</p>
${H.note('If you can say "a score is a dot product, and a dot product is alignment scaled by magnitude", §4.3 will hold no surprises.')}

${H.lab('vec', 'Vector playground — drag the two arrows', 'Drag either arrow head. The projection of <b>a</b> onto <b>b</b> is drawn in grey: it is the shadow <b>a</b> casts on <b>b</b>, and its length is exactly $a\\cdot b / \\|b\\|$.')}

<h2><span class="sn">0.2.2</span> A matrix is a function on space</h2>
<p>$Ax$ takes a vector and returns another. Because $A(\\alpha x + \\beta y) = \\alpha Ax + \\beta Ay$, a matrix is completely determined by what it does to the basis vectors — the columns of $A$ <i>are</i> the images of $(1,0)$ and $(0,1)$. Every linear map, however complicated it looks, does the same three things in sequence: <b>rotate, stretch along axes, rotate again</b>. That is the singular value decomposition, and §1.8 makes it precise.</p>

${H.lab('mat', 'What a matrix does to space', 'Drag the two column arrows — you are editing the matrix directly. The blue grid is the input space; the shape shows where the unit circle and the letter-F land. Watch the determinant go through zero as you make the columns parallel: that is exactly when the map collapses a dimension and becomes non-invertible.')}

<h3>The vocabulary you actually need</h3>
${H.table(['Term', 'Definition', 'Where it bites'], [
      ['Transpose $A^\\mathsf{T}$', 'Flip rows and columns', '$X^\\mathsf{T}X$ in the normal equations (§2.4)'],
      ['Matrix product $AB$', 'Apply $B$, then $A$; entry $(i,j)$ is row $i$ of $A$ dotted with column $j$ of $B$', 'Every layer of every network'],
      ['Identity $I$', 'Does nothing', 'Ridge adds $\\lambda I$ to make a matrix invertible'],
      ['Inverse $A^{-1}$', 'Undoes $A$; exists iff $\\det A \\ne 0$', 'Closed-form least squares; Newton\'s method'],
      ['Rank', 'Number of genuinely independent directions in the output', 'LoRA (§4.13) is the claim that an update has rank 16'],
      ['Norm $\\|x\\|_2$', '$\\sqrt{\\sum x_i^2}$ — length', 'L2 regularization is a penalty on length (§2.3)'],
      ['Symmetric, PSD', '$A = A^\\mathsf{T}$ and $x^\\mathsf{T}Ax \\ge 0$', 'Covariance and kernel matrices; makes optimisation convex (§1.8)']
    ])}

<h2><span class="sn">0.2.3</span> Shapes are the debugging tool</h2>
<p>Most deep-learning bugs are shape bugs, and most shape confusion dissolves with one habit: write the shape after every line. If $X$ is $[n \\times d]$ (n rows, d features) and $W$ is $[d \\times k]$, then $XW$ is $[n \\times k]$ — inner dimensions must match and they vanish. §4.5 does this for an entire transformer block, and that page is nothing but this habit applied twelve times.</p>

${H.code(`import numpy as np
X = np.random.randn(256, 64)      # [n=256, d=64]  a batch of examples
W = np.random.randn(64, 16)       # [d=64,  k=16]  a linear layer
b = np.zeros(16)                  # [k=16]         broadcast over rows
H = X @ W + b                     # [256, 16]      inner 64 cancels
print(H.shape, (X @ W).T.shape)   # (256, 16) (16, 256)`)}

${H.probe([
      ['What does the dot product mean geometrically?', 'Length times length times the cosine of the angle — alignment. Zero means orthogonal.'],
      ['When is a square matrix not invertible, and why do you care?', 'When its determinant is zero, i.e. its columns are linearly dependent — collinear features. It is why $X^\\mathsf{T}X$ can fail to invert and why ridge\'s $\\lambda I$ fixes it (§2.4).']
    ], 'Treating "high-dimensional" as mystical. Every operation you know in 2-D is defined identically in 4096-D; only your ability to draw it fails.')}`,
    labs: {
      vec: function (host) {
        let a = [2.2, 1.4], b = [3.0, -0.8], drag = null;
        const out = Viz.readout(host, [
          { k: 'dot', label: 'a · b', cls: 'key' }, { k: 'na', label: '‖a‖' }, { k: 'nb', label: '‖b‖' },
          { k: 'ang', label: 'angle' }, { k: 'cos', label: 'cosine sim' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-3, 3] }).frame({ grid: true });
            // projection of a onto b
            const bb = Num.dot(b, b) || 1e-9, k = Num.dot(a, b) / bb;
            const proj = [k * b[0], k * b[1]];
            P.line([[proj[0], proj[1]], [a[0], a[1]]], { color: T.faint, width: 1.2, dash: [4, 4] });
            P.arrow(0, 0, proj[0], proj[1], { color: T.faint, width: 5 });
            P.arrow(0, 0, a[0], a[1], { color: T.blue, width: 2.6 });
            P.arrow(0, 0, b[0], b[1], { color: T.red, width: 2.6 });
            P.arrow(0, 0, a[0] + b[0], a[1] + b[1], { color: T.green, width: 1.6 });
            P.line([[a[0], a[1]], [a[0] + b[0], a[1] + b[1]]], { color: T.green, width: 1, dash: [3, 3] });
            P.line([[b[0], b[1]], [a[0] + b[0], a[1] + b[1]]], { color: T.green, width: 1, dash: [3, 3] });
            P.text(a[0], a[1], ' a', { color: T.blue, font: 'bold 13px ui-sans-serif' });
            P.text(b[0], b[1], ' b', { color: T.red, font: 'bold 13px ui-sans-serif' });
            P.text(a[0] + b[0], a[1] + b[1], ' a+b', { color: T.green, font: '12px ui-sans-serif' });
            const d = Num.dot(a, b), na = Num.norm(a), nb = Num.norm(b);
            out({
              dot: d.toFixed(2), na: na.toFixed(2), nb: nb.toFixed(2),
              ang: (Math.acos(Math.max(-1, Math.min(1, d / (na * nb || 1)))) * 180 / Math.PI).toFixed(0) + '°',
              cos: (d / (na * nb || 1)).toFixed(3)
            });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          const x = P.ix(e.x), y = P.iy(e.y);
          if (e.type === 'down') {
            const da = (a[0] - x) ** 2 + (a[1] - y) ** 2, db = (b[0] - x) ** 2 + (b[1] - y) ** 2;
            drag = Math.min(da, db) < .35 ? (da < db ? 'a' : 'b') : null;
          } else if (e.type === 'move' && e.down && drag) {
            if (drag === 'a') a = [x, y]; else b = [x, y];
            S.redraw();
          } else if (e.type === 'up') drag = null;
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'a' }, { c: Viz.theme().red, t: 'b' }, { c: Viz.theme().green, t: 'a + b' }, { c: Viz.theme().faint, t: 'projection of a onto b' }]);
      },

      mat: function (host) {
        let A = [[1.2, 0.6], [0.3, 1.1]]; // columns are A[:,0], A[:,1]
        let drag = null;
        const st = Viz.controls(host, [
          { k: 'shape', label: 'shape', type: 'buttons', value: 'circle', options: [{ v: 'circle', t: 'circle + grid' }, { v: 'f', t: 'letter F' }] },
          { k: 'eig', label: 'show eigenvectors (symmetric part)', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'det', label: 'determinant', cls: 'key' }, { k: 's1', label: 'σ₁ (max stretch)' }, { k: 's2', label: 'σ₂ (min stretch)' }, { k: 'cond', label: 'condition κ' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-2.4, 2.4] }).frame({});
            const ap = (p) => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
            // faint input grid
            for (let g = -2; g <= 2; g++) {
              P.line([[-2.4, g], [2.4, g]], { color: T.line, width: 1, alpha: .5 });
              P.line([[g, -2.4], [g, 2.4]], { color: T.line, width: 1, alpha: .5 });
            }
            // transformed grid
            for (let g = -2; g <= 2; g++) {
              const l1 = [], l2 = [];
              for (let t = -2.4; t <= 2.41; t += .3) { l1.push(ap([t, g])); l2.push(ap([g, t])); }
              P.line(l1, { color: T.blue, width: 1, alpha: .35 });
              P.line(l2, { color: T.blue, width: 1, alpha: .35 });
            }
            if (st.shape === 'circle') {
              const circ = [], img = [];
              for (let t = 0; t <= 6.3; t += .05) { circ.push([Math.cos(t), Math.sin(t)]); img.push(ap([Math.cos(t), Math.sin(t)])); }
              P.line(circ, { color: T.faint, width: 1.4, dash: [4, 3] });
              P.line(img, { color: T.red, width: 2.4 });
            } else {
              const F = [[0, 0], [0, 1.4], [.9, 1.4], [.9, 1.1], [.3, 1.1], [.3, .8], [.75, .8], [.75, .5], [.3, .5], [.3, 0], [0, 0]];
              P.line(F, { color: T.faint, width: 1.4, dash: [4, 3] });
              P.line(F.map(ap), { color: T.red, width: 2.4 });
            }
            // columns
            P.arrow(0, 0, A[0][0], A[1][0], { color: T.blue, width: 2.6 });
            P.arrow(0, 0, A[0][1], A[1][1], { color: T.green, width: 2.6 });
            P.text(A[0][0], A[1][0], ' col 1', { color: T.blue, font: '11px ui-monospace' });
            P.text(A[0][1], A[1][1], ' col 2', { color: T.green, font: '11px ui-monospace' });
            const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
            const sv = Num.svd([[A[0][0], A[0][1]], [A[1][0], A[1][1]]]);
            if (st.eig) {
              const Sym = [[A[0][0], (A[0][1] + A[1][0]) / 2], [(A[0][1] + A[1][0]) / 2, A[1][1]]];
              const e = Num.eigSym(Sym);
              e.vectors.forEach((v, i) => {
                const s = Math.sign(e.values[i]) * Math.min(2.2, Math.abs(e.values[i]));
                P.line([[-v[0] * 2.6, -v[1] * 2.6], [v[0] * 2.6, v[1] * 2.6]], { color: T.amber, width: 1, dash: [6, 4], alpha: .8 });
              });
            }
            out({
              det: det.toFixed(2), s1: sv.s[0].toFixed(2), s2: sv.s[1].toFixed(2),
              cond: (sv.s[0] / (sv.s[1] || 1e-9)).toFixed(1)
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
          } else if (e.type === 'move' && e.down && drag !== null) {
            A[0][drag] = x; A[1][drag] = y; S.redraw();
          } else if (e.type === 'up') drag = null;
        });
        Viz.buttons(host, [
          { label: 'Identity', on: () => { A = [[1, 0], [0, 1]]; S.redraw(); } },
          { label: 'Rotation 30°', on: () => { const c = Math.cos(.52), s = Math.sin(.52); A = [[c, -s], [s, c]]; S.redraw(); } },
          { label: 'Shear', on: () => { A = [[1, 1], [0, 1]]; S.redraw(); } },
          { label: 'Rank-1 (collapse)', on: () => { A = [[1.4, .7], [1.2, .6]]; S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Two feature vectors have cosine similarity 0. What does that tell you?',
        options: ['They are identical', 'They are orthogonal — no linear alignment', 'One is the negative of the other', 'Both have zero length'],
        answer: 1,
        why: 'Cosine zero means the angle is 90°: the dot product carries no alignment. Note this says nothing about non-linear dependence.'
      },
      {
        q: 'A 2×2 matrix has determinant 0. Which statement follows?',
        options: ['It is symmetric', 'Its columns are linearly dependent and it is not invertible', 'It has orthogonal columns', 'Its condition number is 1'],
        answer: 1,
        why: 'Zero determinant = zero volume scaling = the map collapses the plane onto a line; the columns are parallel and no inverse exists. This is collinearity, the thing ridge fixes.'
      }
    ],
    cards: [
      { q: 'Dot product, in words', a: 'Length × length × cosine of the angle. Alignment scaled by magnitude — the score in linear models and in attention.' },
      { q: 'What are the columns of a matrix?', a: 'The images of the basis vectors. A matrix is fully specified by where it sends $(1,0)$ and $(0,1)$.' },
      { q: 'Every linear map is which three operations?', a: 'Rotate, scale along axes, rotate — the SVD $A = U\\Sigma V^\\mathsf{T}$.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.3 */
  ML.section({
    id: 'calculus-basics', track: 'start', num: '0.3',
    title: 'Derivatives, gradients, and the chain rule',
    lede: 'Training is one idea repeated: measure which way the loss goes up, step the other way. The derivative is that measurement, the gradient is its multi-dimensional version, and the chain rule is what makes it computable through a hundred layers.',
    html: `
<h2><span class="sn">0.3.1</span> A derivative is a slope you can act on</h2>
<p>$f'(x)$ is the limit of rise over run as the run shrinks: how much $f$ changes per unit change in $x$, right here. Two readings matter. <b>Locally</b>, $f(x + \\epsilon) \\approx f(x) + \\epsilon f'(x)$ — the derivative is the best linear prediction of what a small step will do. <b>Globally</b>, $f' = 0$ marks the flat points: minima, maxima, saddles.</p>
<p>Optimisation lives entirely in the first reading. If $f'(x) > 0$, moving right increases $f$; to decrease it, move left. Hence <b>gradient descent</b>: $x \\leftarrow x - \\eta f'(x)$, where $\\eta$ is a step size you choose and immediately regret.</p>

${H.lab('deriv', 'Slope, tangent, and one gradient step', 'Drag the point along the curve. The dashed line is the tangent — the linear model the derivative gives you. Press <i>step</i> to take one gradient step and watch it land.')}

<h2><span class="sn">0.3.2</span> Rules you will use daily</h2>
${H.table(['Rule', 'Statement', 'The ML instance'], [
      ['Power', '$\\frac{d}{dx}x^n = nx^{n-1}$', 'Squared error differentiates to $2(\\hat y - y)$'],
      ['Exponential / log', '$\\frac{d}{dx}e^x = e^x$, $\\frac{d}{dx}\\ln x = 1/x$', 'Every log-likelihood'],
      ['Product', '$(uv)\' = u\'v + uv\'$', 'Attention scores, gated units'],
      ['Chain', '$\\frac{d}{dx}f(g(x)) = f\'(g(x))g\'(x)$', '<b>Backpropagation, entirely</b>'],
      ['Sigmoid', '$\\sigma\' = \\sigma(1-\\sigma)$', 'The two-line logistic gradient (§2.4)']
    ])}

<h2><span class="sn">0.3.3</span> Gradients: many knobs at once</h2>
<p>With several parameters, the partial derivative $\\partial f/\\partial \\theta_j$ asks: if I nudge <i>this</i> knob and freeze the others, how does $f$ move? Stack them and you have the gradient</p>
$$\\nabla f(\\theta) = \\left(\\frac{\\partial f}{\\partial\\theta_1}, \\ldots, \\frac{\\partial f}{\\partial\\theta_p}\\right)$$
<p>which points in the direction of <b>steepest increase</b>, so $-\\nabla f$ is the direction of steepest decrease. The second derivative generalises to the <b>Hessian</b> $H$, the matrix of curvatures; it tells you how much the gradient itself changes as you move, which is what §1.9 needs to explain why some problems train easily and others crawl.</p>

<h2><span class="sn">0.3.4</span> The chain rule is the entire mechanism</h2>
<p>A network is a composition: $L(f_3(f_2(f_1(x))))$. The chain rule says the derivative of a composition is the <i>product</i> of the local derivatives. Backpropagation is nothing more than evaluating that product from the outside in, caching each intermediate so no factor is computed twice.</p>

${H.lab('chain', 'A computation graph, with numbers flowing both ways', 'Change the inputs and watch the forward values (blue, left to right) and the gradients (red, right to left). Every red number is a product of the local derivatives on the path back to it — that is all backprop is.')}

${H.key('Every gradient in a network is a product of local derivatives along a path. Multiply enough factors below one and it vanishes; that single sentence explains ReLU, residual connections and normalisation.')}

${H.probe([
      ['Why reverse-mode and not forward-mode differentiation?', 'The loss is one number and the parameters are many. Reverse mode costs about one forward pass per <i>output</i>; forward mode costs one per <i>input</i>. One output, billions of inputs — reverse wins by a factor of billions.'],
      ['What does the Hessian tell you?', 'Curvature. Its eigenvalue ratio (the condition number) sets both the largest stable step and the convergence rate (§1.9).']
    ])}`,
    labs: {
      deriv: function (host) {
        let x = -1.6;
        const st = Viz.controls(host, [
          { k: 'f', label: 'function', type: 'buttons', value: 'quad', options: [{ v: 'quad', t: 'x²/2' }, { v: 'wave', t: 'x²/6 + sin 2x' }, { v: 'quart', t: 'x⁴/8 − x²' }] },
          { k: 'lr', label: 'step size η', min: .02, max: 1.2, step: .02, value: .35, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const F = {
          quad: [x => x * x / 2, x => x],
          wave: [x => x * x / 6 + Math.sin(2 * x), x => x / 3 + 2 * Math.cos(2 * x)],
          quart: [x => x * x * x * x / 8 - x * x, x => x * x * x / 2 - 2 * x]
        };
        const out = Viz.readout(host, [{ k: 'x', label: 'x' }, { k: 'fx', label: 'f(x)' }, { k: 'd', label: "f '(x)", cls: 'key' }, { k: 'nx', label: 'next x' }]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const [f, df] = F[st.f];
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-3, 5] }).frame({ xlabel: 'x' });
            P.clip(() => {
              P.fn(f, { color: T.blue, width: 2.4 });
              const s = df(x), y0 = f(x);
              P.fn(t => y0 + s * (t - x), { color: T.red, width: 1.6, dash: [5, 4] });
              const nx = x - st.lr * s;
              P.arrow(x, y0, nx, f(nx), { color: T.green, width: 2 });
              P.dots([[x, y0]], { r: 6, color: T.red, stroke: true });
              P.dots([[nx, f(nx)]], { r: 5, color: T.green, stroke: true });
            });
            const s = df(x);
            out({ x: x.toFixed(2), fx: f(x).toFixed(3), d: s.toFixed(3), nx: (x - st.lr * s).toFixed(3) });
            S._P = P;
          }
        });
        Viz.pointer(S, function (e) {
          const P = S._P; if (!P) return;
          if (e.down) { x = Math.max(-3.1, Math.min(3.1, P.ix(e.x))); S.redraw(); }
        });
        Viz.buttons(host, [
          { label: 'Take one step', primary: true, on: () => { const [f, df] = F[st.f]; x = x - st.lr * df(x); S.redraw(); } },
          { label: 'Run 30 steps', on: () => { const [f, df] = F[st.f]; let i = 0; const t = setInterval(() => { x = x - st.lr * df(x); S.redraw(); if (++i > 30) clearInterval(t); }, 60); ML.onCleanup(() => clearInterval(t)); } },
          { label: 'Reset', on: () => { x = -1.6; S.redraw(); } }
        ]);
      },

      chain: function (host) {
        const st = Viz.controls(host, [
          { k: 'x', label: 'input x', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'w', label: 'weight w', min: -2, max: 2, step: .1, value: .5, fmt: v => v.toFixed(1) },
          { k: 'b', label: 'bias b', min: -2, max: 2, step: .1, value: 0, fmt: v => v.toFixed(1) },
          { k: 'y', label: 'target y', min: -2, max: 3, step: .1, value: 1, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const z = st.w * st.x + st.b;
            const a = Math.max(0, z);                 // ReLU
            const L = .5 * (a - st.y) * (a - st.y);
            const dLda = a - st.y, daz = z > 0 ? 1 : 0;
            const dLdz = dLda * daz, dLdw = dLdz * st.x, dLdb = dLdz;
            const nodes = [
              { x: .07, l: 'x', v: st.x, g: null },
              { x: .30, l: 'z = wx + b', v: z, g: dLdz },
              { x: .58, l: 'a = ReLU(z)', v: a, g: dLda },
              { x: .87, l: 'L = ½(a−y)²', v: L, g: 1 }
            ];
            const cy = h * .42, bw = Math.min(126, w * .21), bh = 46;
            ctx.font = '11px ui-sans-serif, system-ui';
            nodes.forEach((n, i) => {
              const cx = n.x * w + bw / 2;
              Labs.roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, 9);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.font = '11px ui-monospace, monospace';
              ctx.fillText(n.l, cx, cy - 9);
              ctx.fillStyle = T.blue; ctx.font = 'bold 13px ui-monospace, monospace';
              ctx.fillText(n.v.toFixed(3), cx, cy + 11);
              if (i < nodes.length - 1) {
                const nx = nodes[i + 1].x * w + bw / 2;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(cx + bw / 2, cy - 8); ctx.lineTo(nx - bw / 2 - 4, cy - 8); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 4, cy - 8); ctx.lineTo(nx - bw / 2 - 10, cy - 12); ctx.lineTo(nx - bw / 2 - 10, cy - 4); ctx.closePath(); ctx.fillStyle = T.blue; ctx.fill();
                ctx.strokeStyle = T.red; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 4, cy + 10); ctx.lineTo(cx + bw / 2, cy + 10); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cx + bw / 2, cy + 10); ctx.lineTo(cx + bw / 2 + 6, cy + 6); ctx.lineTo(cx + bw / 2 + 6, cy + 14); ctx.closePath(); ctx.fillStyle = T.red; ctx.fill();
              }
              if (n.g !== null) {
                ctx.fillStyle = T.red; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
                ctx.fillText('∂L/∂· = ' + n.g.toFixed(3), cx, cy + bh / 2 + 16);
              }
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('forward →', 10, 10);
            ctx.fillStyle = T.red; ctx.fillText('← backward', 10, 28);
            ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('∂L/∂w = ∂L/∂z · x = ' + dLdw.toFixed(3) + '     ∂L/∂b = ' + dLdb.toFixed(3), w / 2, h - 26);
          }
        });
      }
    },
    quiz: [
      {
        q: 'Gradient descent updates $\\theta \\leftarrow \\theta - \\eta\\nabla L$. Why the minus sign?',
        options: ['To keep parameters positive', 'The gradient points uphill; we want to go downhill', 'It cancels the learning rate', 'Convention only — plus works too'],
        answer: 1,
        why: '∇L is the direction of steepest <i>increase</i>. Descending means moving against it.'
      },
      {
        q: 'A 40-layer network trains with sigmoid activations and the early layers barely move. The most direct explanation is…',
        options: ['The learning rate is too large', 'Each backward step multiplies by σ′ ≤ 0.25, so the product over depth collapses', 'The loss is non-convex', 'Batch size is too small'],
        answer: 1,
        why: 'Vanishing gradients: the chain rule multiplies local derivatives, and 0.25⁴⁰ is astronomically small. ReLU passes 1 on the active side; residual connections add a path whose local derivative is exactly 1.'
      }
    ],
    cards: [
      { q: 'Chain rule, and why it matters here', a: '$\\frac{d}{dx}f(g(x)) = f\'(g(x))g\'(x)$ — backpropagation is this product accumulated right-to-left with cached activations.' },
      { q: 'Why reverse-mode autodiff?', a: 'One scalar output, many parameters. Reverse mode costs ~one forward pass per output; forward mode costs one per input.' },
      { q: 'Gradient vs Hessian', a: 'Gradient = direction of steepest ascent (first derivatives). Hessian = curvature (second derivatives); its condition number governs how hard optimisation is.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.4 */
  ML.section({
    id: 'probability-basics', track: 'start', num: '0.4',
    title: 'Probability, from the ground',
    lede: 'Joint, marginal, conditional — three views of one table. Get them straight here and Bayes in §1.1 is a one-line consequence rather than a formula to memorise.',
    html: `
<h2><span class="sn">0.4.1</span> The objects</h2>
<p>A <b>random variable</b> is a quantity whose value is uncertain: a coin flip, tomorrow's demand, the next token. A <b>distribution</b> assigns probability across its possible values — non-negative, summing (or integrating) to one. That is the whole of the axioms you need.</p>
<p>With two variables you get one object and two views of it:</p>
<ul>
<li><b>Joint</b> $P(A, B)$ — the probability of both. The full table.</li>
<li><b>Marginal</b> $P(A) = \\sum_b P(A, b)$ — sum the table along a direction and one variable disappears.</li>
<li><b>Conditional</b> $P(A \\mid B) = P(A, B) / P(B)$ — keep only the row where $B$ happened and rescale it so it sums to one.</li>
</ul>
${H.key('A conditional probability is not a new quantity. It is the same table, restricted to a smaller world and renormalised.')}

${H.lab('joint', 'One table, three views', 'Drag the four joint probabilities. The marginals appear on the edges; the conditional strip shows what happens when you delete every outcome incompatible with the evidence and rescale. Watch independence appear exactly when the conditional matches the marginal.')}

<h2><span class="sn">0.4.2</span> Independence, and why it is a strong claim</h2>
<p>$A$ and $B$ are independent iff $P(A, B) = P(A)P(B)$, equivalently $P(A \\mid B) = P(A)$: learning $B$ tells you nothing about $A$. It is rare in real data and extremely convenient in models — Naive Bayes (§2.5) assumes it between features, and is a useful classifier <i>despite the assumption being false</i>, for reasons that section makes precise.</p>

<h2><span class="sn">0.4.3</span> Expectation and variance, in one line each</h2>
$$\\mathbb{E}[X] = \\sum_x x\\,p(x), \\qquad \\mathrm{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2] = \\mathbb{E}[X^2] - \\mathbb{E}[X]^2$$
<p>Expectation is the long-run average — the centre of mass of the distribution. Variance is the average squared distance from that centre; its square root, the standard deviation, is in the same units as $X$ and is the one to quote. §1.3 shows the asymmetry that runs through the entire site: expectation is linear <i>always</i>, variance adds only under independence.</p>

<h2><span class="sn">0.4.4</span> Discrete or continuous</h2>
<p>Discrete variables have a probability <b>mass</b> function: $p(x)$ is a probability. Continuous variables have a <b>density</b>: $p(x)$ is not a probability and can exceed 1 — only $\\int_a^b p(x)dx$ is a probability. This is why a Gaussian's peak height changes with its width, and why likelihoods of continuous data can be greater than one without anything being wrong.</p>

${H.probe([
      ['Define conditional probability without the formula.', 'Restrict attention to the world where the evidence is true, then rescale so the remaining outcomes sum to one.'],
      ['Can a density be greater than 1?', 'Yes — a density is not a probability. A uniform on $[0, 0.5]$ has density 2 everywhere on its support.']
    ], 'Confusing $P(A\\mid B)$ with $P(B\\mid A)$. That single confusion is the base-rate error in §1.1 and the most common probability mistake in interviews.')}`,
    labs: {
      joint: function (host) {
        // joint over A in {a1,a2}, B in {b1,b2}
        let J = [[.30, .12], [.18, .40]];
        const st = Viz.controls(host, [
          { k: 'p11', label: 'P(A=1, B=1)', min: .01, max: .8, step: .01, value: .30, fmt: v => v.toFixed(2) },
          { k: 'p12', label: 'P(A=1, B=2)', min: .01, max: .8, step: .01, value: .12, fmt: v => v.toFixed(2) },
          { k: 'p21', label: 'P(A=2, B=1)', min: .01, max: .8, step: .01, value: .18, fmt: v => v.toFixed(2) },
          { k: 'p22', label: 'P(A=2, B=2)', min: .01, max: .8, step: .01, value: .40, fmt: v => v.toFixed(2) },
          { k: 'cond', label: 'condition on', type: 'buttons', value: 'B1', options: [{ v: 'B1', t: 'B = 1' }, { v: 'B2', t: 'B = 2' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'pa', label: 'P(A=1)' }, { k: 'pb', label: 'P(B=1)' },
          { k: 'cond', label: 'P(A=1 | evidence)', cls: 'key' }, { k: 'ind', label: 'independent?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const raw = [[st.p11, st.p12], [st.p21, st.p22]];
            const tot = raw[0][0] + raw[0][1] + raw[1][0] + raw[1][1];
            J = raw.map(r => r.map(v => v / tot));
            const gx = 60, gy = 46, cell = Math.min(96, (w - 260) / 2);
            const cols = [T.blue, T.red];
            ctx.font = '11px ui-monospace, monospace';
            for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
              const x = gx + j * cell, y = gy + i * cell;
              const p = J[i][j];
              ctx.fillStyle = i === 0 ? 'rgba(90,120,230,' + (0.15 + p) + ')' : 'rgba(220,90,80,' + (0.15 + p) + ')';
              ctx.fillRect(x, y, cell - 4, cell - 4);
              ctx.strokeStyle = T.line; ctx.strokeRect(x, y, cell - 4, cell - 4);
              ctx.fillStyle = T.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(p.toFixed(3), x + cell / 2 - 2, y + cell / 2 - 2);
            }
            ctx.font = '11px ui-monospace, monospace'; ctx.fillStyle = T.muted;
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            ctx.fillText('A = 1', gx - 8, gy + cell / 2); ctx.fillText('A = 2', gx - 8, gy + cell * 1.5);
            ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText('B = 1', gx + cell / 2, gy - 8); ctx.fillText('B = 2', gx + cell * 1.5, gy - 8);
            // marginals
            const pa = [J[0][0] + J[0][1], J[1][0] + J[1][1]];
            const pb = [J[0][0] + J[1][0], J[0][1] + J[1][1]];
            ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = T.text;
            pa.forEach((v, i) => ctx.fillText('P(A=' + (i + 1) + ') = ' + v.toFixed(3), gx + 2 * cell + 6, gy + cell * (i + .5)));
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            pb.forEach((v, j) => ctx.fillText(v.toFixed(3), gx + cell * (j + .5), gy + 2 * cell + 4));
            // conditional strip
            const jb = st.cond === 'B1' ? 0 : 1;
            const denom = pb[jb];
            const cx = gx, cy = gy + 2 * cell + 34;
            const stripW = Math.min(2 * cell - 4, w - gx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('conditional on ' + (jb ? 'B = 2' : 'B = 1') + ' — that column, rescaled to sum to 1', cx, cy - 5);
            const c1 = J[0][jb] / denom;
            ctx.fillStyle = T.blue; ctx.fillRect(cx, cy, stripW * c1, 26);
            ctx.fillStyle = T.red; ctx.fillRect(cx + stripW * c1, cy, stripW * (1 - c1), 26);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            if (c1 > .12) ctx.fillText('A=1 · ' + c1.toFixed(3), cx + stripW * c1 / 2, cy + 13);
            if (1 - c1 > .12) ctx.fillText('A=2 · ' + (1 - c1).toFixed(3), cx + stripW * (c1 + (1 - c1) / 2), cy + 13);
            const indep = Math.abs(J[0][0] - pa[0] * pb[0]) < .004;
            out({ pa: pa[0].toFixed(3), pb: pb[0].toFixed(3), cond: c1.toFixed(3), ind: indep ? 'yes' : 'no' });
          }
        });
        Viz.buttons(host, [
          { label: 'Make them independent', on: () => { const pa = .45, pb = .55; st.$set('p11', +(pa * pb).toFixed(2)); st.$set('p12', +(pa * (1 - pb)).toFixed(2)); st.$set('p21', +((1 - pa) * pb).toFixed(2)); st.$set('p22', +((1 - pa) * (1 - pb)).toFixed(2)); S.redraw(); } },
          { label: 'Strong dependence', on: () => { st.$set('p11', .45); st.$set('p12', .05); st.$set('p21', .05); st.$set('p22', .45); S.redraw(); } }
        ]);
      }
    },
    quiz: [
      {
        q: 'Given the joint table, how do you get the marginal $P(A)$?',
        options: ['Divide by $P(B)$', 'Sum the joint over all values of $B$', 'Multiply by $P(B\\mid A)$', 'Take the maximum over $B$'],
        answer: 1,
        why: 'Marginalisation is summation over the variable you want to eliminate — the law of total probability, which is exactly the denominator of Bayes.'
      },
      {
        q: 'Which is true of a continuous probability density $p(x)$?',
        options: ['It is always ≤ 1', 'It integrates to 1 and may exceed 1 pointwise', 'It equals $P(X = x)$', 'It must be symmetric'],
        answer: 1,
        why: 'Density is probability per unit length. Only integrals over intervals are probabilities; $P(X=x)=0$ for continuous $X$.'
      }
    ],
    cards: [
      { q: 'Conditional probability in one sentence', a: 'The same measure restricted to the world where the evidence holds, renormalised to sum to one.' },
      { q: 'Definition of independence', a: '$P(A,B)=P(A)P(B)$, equivalently $P(A\\mid B)=P(A)$ — the evidence changes nothing.' },
      { q: 'Variance, two forms', a: '$\\mathrm{Var}(X)=\\mathbb{E}[(X-\\mu)^2]=\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$.' }
    ]
  });

  /* ------------------------------------------------------------------ 0.5 */
  ML.section({
    id: 'python-toolkit', track: 'start', num: '0.5',
    title: 'The working toolkit: numpy, pandas, scikit-learn, PyTorch',
    lede: 'The four libraries that carry ninety per cent of practical work, with the idioms that matter and the three mistakes that quietly invalidate results.',
    html: `
<h2><span class="sn">0.5.1</span> numpy — everything is an array with a shape</h2>
${H.code(`import numpy as np

X = np.random.randn(1000, 8)          # 1000 examples, 8 features
w = np.random.randn(8)
y = X @ w + 0.1 * np.random.randn(1000)   # @ is matrix multiply

# vectorise: never loop over rows if a matrix op exists
mu, sd = X.mean(0), X.std(0)          # per-column statistics
Z = (X - mu) / sd                     # broadcasting: (1000,8) - (8,) works

# the least-squares solution, three ways
beta_normal = np.linalg.solve(X.T @ X, X.T @ y)     # fine, and fast
beta_lstsq  = np.linalg.lstsq(X, y, rcond=None)[0]  # numerically safer
beta_ridge  = np.linalg.solve(X.T @ X + 1e-2*np.eye(8), X.T @ y)`)}
<p>Two habits pay for themselves: annotate shapes in comments, and prefer <code>np.linalg.lstsq</code> or a QR/SVD-based solver to explicitly inverting $X^\\mathsf{T}X$ — the inverse is numerically fragile precisely when your features are collinear, which is the case you were worried about anyway.</p>

<h2><span class="sn">0.5.2</span> pandas — tables, and the leakage trap</h2>
${H.code(`import pandas as pd

df = pd.read_parquet("applications.parquet")
df["utilisation"] = df.balance / df.limit
df["age_days"] = (df.decision_date - df.opened_date).dt.days

# GOOD: aggregate with an explicit time boundary
hist = (df[df.decision_date < cutoff]
        .groupby("customer_id")["amount"].mean()
        .rename("mean_amount_before_cutoff"))

# WRONG: this aggregate sees the whole history, including the future
df["mean_amount"] = df.groupby("customer_id")["amount"].transform("mean")`)}
${H.flag('That last line is the single most common leakage bug in tabular work: a group aggregate computed over the full dataset lets each row see its own future. §2.11 has the full checklist.')}

<h2><span class="sn">0.5.3</span> scikit-learn — fit/transform, and why pipelines are not optional</h2>
${H.code(`from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("scale", StandardScaler()),          # fitted INSIDE each fold
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(auc.mean(), auc.std())`)}
<p>The rule the pipeline enforces: <b>every learned transformation — scaling, imputation, target encoding, binning — must be fitted on the training fold only</b>. Scale first and cross-validate second and your validation rows have already influenced the mean and standard deviation; the score you report is optimistic and the mechanism is invisible.</p>

<h2><span class="sn">0.5.4</span> PyTorch — the same three objects, on a GPU</h2>
${H.code(`import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()      # logits in, not probabilities

for xb, yb in loader:
    opt.zero_grad(set_to_none=True)   # gradients accumulate by default
    logits = model(xb).squeeze(-1)
    loss = loss_fn(logits, yb.float())
    loss.backward()                   # reverse-mode autodiff (§0.3)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()`)}
${H.table(['Idiom', 'Why'], [
      ['<code>BCEWithLogitsLoss</code> not <code>Sigmoid</code> + <code>BCELoss</code>', 'Fuses the log-sum-exp; numerically stable at large logits (§1.10)'],
      ['<code>opt.zero_grad()</code> every step', 'PyTorch accumulates gradients; forgetting it silently sums minibatches'],
      ['<code>model.eval()</code> + <code>torch.no_grad()</code> at inference', 'Turns off dropout and batch-norm updates, and stops building the graph'],
      ['<code>clip_grad_norm_</code>', 'The cheapest insurance against a single bad batch producing NaN (§4.11)'],
      ['<code>AdamW</code> not <code>Adam(weight_decay=)</code>', 'Decoupled decay actually decays; see §3.5']
    ])}

<h2><span class="sn">0.5.5</span> The three mistakes</h2>
${H.checklist([
      '<b>Fitting a transformer outside the fold.</b> Scaling, imputation and encoding are learned parameters. Treat them as model parameters, because they are.',
      '<b>Shuffling time-structured data.</b> A random k-fold on a time series trains on the future to predict the past. Use out-of-time validation (§2.14).',
      '<b>Reporting the tuned score.</b> The validation set that chose your hyperparameters is no longer an unbiased estimate. Nest it, or hold out a final untouched set.'
    ])}`,
    quiz: [
      {
        q: 'Why wrap the scaler and the model in a Pipeline before cross-validating?',
        options: ['It is faster', 'So the scaler is re-fitted inside each training fold, preventing the validation rows from influencing it', 'It makes the model more accurate', 'Because sklearn requires it'],
        answer: 1,
        why: 'Any learned transformation fitted on all the data leaks information from the validation fold into training, inflating the score by an amount you cannot see.'
      },
      {
        q: 'You forget to call opt.zero_grad() in a PyTorch loop. What happens?',
        options: ['The model does not train at all', 'Gradients accumulate across batches, so each step uses a stale sum of gradients', 'Learning rate is ignored', 'It raises an exception'],
        answer: 1,
        why: 'PyTorch accumulates into .grad by design (useful for gradient accumulation). Forgetting to clear it makes every step an increasingly large, increasingly wrong sum.'
      }
    ],
    cards: [
      { q: 'The pipeline rule', a: 'Every learned transformation is fitted inside the training fold only — scaling, imputation, encoding, binning.' },
      { q: 'Why BCEWithLogitsLoss over Sigmoid+BCELoss', a: 'It fuses the sigmoid and the log using the log-sum-exp trick, so large logits do not overflow.' }
    ]
  });
})();
