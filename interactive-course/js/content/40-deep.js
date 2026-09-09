/* ============================================================
   PART 3 — Neural networks & deep learning (3.1 – 3.6)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.1 */
  ML.section({
    id: 'nn-fundamentals', track: 'deep', num: '3.1',
    title: 'Neural network fundamentals',
    lede: 'The bridge into modern AI: every training detail in this part recurs at ten thousand times the scale in Part 4.',
    rests: 'Rests on §0.3 (chain rule), §1.5 (the loss), §1.9 (conditioning).',
    html: `
<h2><span class="sn">3.1.1</span> From perceptron to universal approximator</h2>
<p>A perceptron is a linear model with a step: $\\hat y = \\mathbb{1}[w^\\mathsf{T}x + b > 0]$. Stack them with a nonlinear activation between layers and you get a <b>universal approximator</b> — a network with one sufficiently wide hidden layer can approximate any continuous function on a compact set to arbitrary accuracy.</p>
${H.flag('That theorem is weaker than it sounds. It says a good approximation <i>exists</i>; it says nothing about how wide, whether gradient descent finds it, or how much data it takes. Depth is what makes the approximation efficient in practice — composing simple features beats one enormous layer at the same parameter count.')}

<h3>Why a nonlinearity is not optional</h3>
<p>Two linear layers compose to one: $W_2(W_1x) = (W_2W_1)x$. Without a nonlinearity, depth buys nothing at all. The activation is what lets each layer bend the space so the next layer's linear cut can separate what was previously entangled — which is the same job the kernel did in §2.6, except here the feature map is <i>learned</i> rather than chosen.</p>

<h2><span class="sn">3.1.2</span> The playground</h2>
<p>Below is a real network — forward pass, backpropagation, Adam, all computed in your browser. Nothing is pre-baked. Build a shape, press train, and watch the decision boundary form.</p>

${H.lab('playground', 'Build and train a neural network', 'Choose a dataset, a shape and an activation, then train. The small panels are the individual hidden units — each one is a feature the network invented. Watch them specialise: on the spiral, units first learn crude half-planes, then bend.')}

<h2><span class="sn">3.1.3</span> What each knob actually does</h2>
${H.table(['Knob', 'Effect', 'Failure mode when wrong'], [
      ['<b>Width</b>', 'How many features per layer', 'Too narrow: underfits, loss plateaus high'],
      ['<b>Depth</b>', 'How composed those features can be', 'Too deep without residuals/normalisation: gradients vanish (§3.3)'],
      ['<b>Activation</b>', 'The shape of the bend', 'Saturating activations kill gradients through depth'],
      ['<b>Learning rate</b>', 'Step size', 'Too high: loss oscillates or NaNs. Too low: never arrives'],
      ['<b>Batch size</b>', 'Gradient noise per step', 'Too large: wasted compute past the critical batch size (§4.11)'],
      ['<b>Weight decay</b>', 'Pull toward small weights', 'Too high: underfits; too low: memorises']
    ])}

<h2><span class="sn">3.1.4</span> The features are the point</h2>
<p>Classical ML asks you to engineer features and then fits a simple model on top (§2.11). A neural network fits the feature extractor and the classifier <i>jointly</i>, which is why it wins wherever features are hard to write down — pixels, audio, text — and why it usually loses on small tabular data where the features are already meaningful and boosted trees can exploit them directly (§2.8).</p>

${H.probe([
      ['Why do you need a nonlinearity?', 'Composed linear maps collapse to a single linear map; without one, depth is free of content.'],
      ['What does the universal approximation theorem actually guarantee?', 'Existence of an approximating network, not learnability, not efficiency, and not generalisation.'],
      ['When would you not use a neural net?', 'Small-to-medium tabular data with meaningful features — gradient boosting is usually better and far easier to explain.']
    ])}`,
    labs: {
      playground: function (host) {
        let data = Num.dataset('circles', 220, .22, 5);
        let net = null, running = false, hist = [], epoch = 0;
        const st = Viz.controls(host, [
          { k: 'dataset', label: 'dataset', type: 'select', value: 'circles', options: [
            { v: 'circles', t: 'circles' }, { v: 'moons', t: 'moons' }, { v: 'xor', t: 'xor' }, { v: 'spiral', t: 'spiral' }, { v: 'blobs', t: 'blobs (linear)' }] },
          { k: 'depth', label: 'hidden layers', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'width', label: 'units per layer', min: 2, max: 12, step: 1, value: 6, fmt: v => v },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'sigmoid', t: 'sigmoid' }] },
          { k: 'lr', label: 'learning rate', min: -3.3, max: -.3, step: .1, value: -1.3, fmt: v => Math.pow(10, v).toFixed(3) },
          { k: 'l2', label: 'weight decay', min: 0, max: .05, step: .001, value: 0, fmt: v => v.toFixed(3) },
          { k: 'noise', label: 'data noise', min: .05, max: .6, step: .05, value: .22, fmt: v => v.toFixed(2) }
        ], rebuild);
        const out = Viz.readout(host, [
          { k: 'loss', label: 'train loss', cls: 'key' }, { k: 'acc', label: 'train accuracy' },
          { k: 'tacc', label: 'test accuracy', cls: 'good' }, { k: 'ep', label: 'epochs' }, { k: 'params', label: 'parameters' }
        ]);
        let test = null;
        function rebuild() {
          data = Num.dataset(st.dataset, 220, st.noise, 5);
          test = Num.dataset(st.dataset, 220, st.noise, 999);
          const sizes = [2];
          for (let i = 0; i < st.depth; i++) sizes.push(st.width);
          sizes.push(1);
          net = Num.mlp(sizes, { act: st.act, seed: 3 });
          hist = []; epoch = 0; S.redraw();
        }
        function trainSteps(n) {
          for (let i = 0; i < n; i++) { const l = net.trainBatch(data.X, data.y, Math.pow(10, st.lr), st.l2); hist.push(l); epoch++; }
        }
        const S = Viz.surface(host, {
          height: 380,
          draw: function (ctx, w, h, T) {
            const panelW = Math.min(190, w * .32);
            const P = Viz.plot(ctx, w, h, { xd: [-3.4, 3.4], yd: [-2.6, 2.6], pad: { l: 36, r: panelW + 24, t: 14, b: 34 } })
              .frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 4 }));
            P.clip(() => Labs.points(P, data.X, data.y, { r: 3.2 }));
            // hidden unit panels (first hidden layer)
            const nUnits = Math.min(st.width, 8);
            const cols = 2, cell = Math.min(60, (panelW - 12) / cols);
            const ox = w - panelW - 6, oy = 22;
            ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('hidden layer 1 — learned features', ox, oy - 5);
            for (let u = 0; u < nUnits; u++) {
              const cx = ox + (u % cols) * (cell + 6), cy = oy + Math.floor(u / cols) * (cell + 6);
              const img = ctx.createImageData(Math.round(cell), Math.round(cell));
              for (let py = 0; py < cell; py++) for (let px = 0; px < cell; px++) {
                const x = -3.4 + 6.8 * px / cell, y = 2.6 - 5.2 * py / cell;
                const a = net.forward([x, y]).as[1][u];
                const t = Math.max(0, Math.min(1, (a + 1) / 2));
                const i4 = (py * Math.round(cell) + px) * 4;
                img.data[i4] = Math.round(90 + 130 * t);
                img.data[i4 + 1] = Math.round(110 + 60 * (1 - Math.abs(t - .5) * 2));
                img.data[i4 + 2] = Math.round(220 - 140 * t);
                img.data[i4 + 3] = 190;
              }
              Viz.blit(ctx, img, Math.round(cx), Math.round(cy), cell, cell);
              ctx.strokeStyle = T.line; ctx.strokeRect(cx, cy, cell, cell);
            }
            // loss curve
            if (hist.length > 2) {
              const lx = ox, ly = oy + Math.ceil(nUnits / cols) * (cell + 6) + 14, lw = panelW - 6, lh = 52;
              ctx.fillStyle = T.paper; ctx.fillRect(lx, ly, lw, lh);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, ly, lw, lh);
              const mx = Math.max.apply(null, hist.slice(0, 20));
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.6; ctx.beginPath();
              const step = Math.max(1, Math.floor(hist.length / lw));
              for (let i = 0; i < hist.length; i += step) {
                const X = lx + lw * i / (hist.length - 1), Y = ly + lh - lh * Math.min(1, hist[i] / (mx || 1));
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              }
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('loss', lx + 4, ly + 3);
            }
            const acc = Num.mean(data.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === data.y[i]) ? 1 : 0));
            const tacc = Num.mean(test.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === test.y[i]) ? 1 : 0));
            let params = 0;
            net.W.forEach(l => l.forEach(r => params += r.length + 1));
            out({
              loss: hist.length ? hist[hist.length - 1].toFixed(4) : '—',
              acc: (acc * 100).toFixed(1) + '%', tacc: (tacc * 100).toFixed(1) + '%',
              ep: epoch, params: params
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 200 epochs', primary: true, on: () => { trainSteps(200); S.redraw(); } },
          { label: 'Train / pause', on: () => {
            running = !running;
            const t = setInterval(() => { if (!running) return clearInterval(t); trainSteps(6); S.redraw(); }, 30);
            ML.onCleanup(() => clearInterval(t));
          } },
          { label: 'Reset weights', on: rebuild }
        ]);
        rebuild();
        Viz.note(host, 'Try the spiral with 1 hidden layer of 2 units — it cannot be done; the model does not have enough features. Add width, then depth, and watch the boundary become possible. That progression is what "capacity" means, made visible.');
      }
    },
    quiz: [
      {
        q: 'You stack three linear layers with no activation between them. The result is…',
        options: ['a deep model with more capacity', 'exactly equivalent to a single linear layer', 'a model that cannot be trained', 'a convex problem with better conditioning'],
        answer: 1,
        why: 'Composition of linear maps is linear: $W_3W_2W_1$ is one matrix. The nonlinearity is what makes depth mean anything.'
      },
      {
        q: 'The universal approximation theorem guarantees…',
        options: ['that gradient descent will find a good network', 'that a sufficiently wide network can approximate any continuous function on a compact set', 'good generalisation', 'that depth beats width'],
        answer: 1,
        why: 'It is an existence result about approximation only — nothing about optimisation, sample complexity or generalisation.'
      }
    ],
    cards: [
      { q: 'Why do neural nets need nonlinearity?', a: 'Composed linear maps collapse to one linear map; without it, depth adds nothing.' },
      { q: 'Universal approximation — the caveat', a: 'It guarantees existence, not learnability, efficiency or generalisation. Depth is what makes it practical.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.2 */
  ML.section({
    id: 'backprop', track: 'deep', num: '3.2',
    title: 'Backpropagation, by hand',
    lede: 'Nothing more than the chain rule applied layer by layer in reverse, reusing cached forward activations so the whole gradient costs about one forward pass. That is the entire reason deep learning is computationally feasible.',
    html: `
<h2><span class="sn">3.2.1</span> The mechanism</h2>
<p>For a composed function $f = f_3 \\circ f_2 \\circ f_1$, the derivative is a product of Jacobians: $\\partial f/\\partial x = J_3 J_2 J_1$. Backpropagation evaluates that product <b>right-to-left</b>, which is cheaper than left-to-right whenever the output is a scalar — one number out, many parameters in. That asymmetry is why reverse-mode automatic differentiation, not forward-mode, powers deep learning.</p>

${H.worked('worked backprop — one step, by hand', `
<p>A two-layer net with one unit per layer: $z_1 = w_1x + b_1$, $a_1 = \\mathrm{ReLU}(z_1)$, $\\hat y = w_2a_1 + b_2$, loss $L = \\tfrac12(\\hat y - y)^2$. Take $x=1$, $w_1=0.5$, $w_2=1.0$, biases 0, target $y=1$.</p>
<p><b>Forward.</b> $z_1 = 0.5$, $a_1 = 0.5$, $\\hat y = 0.5$, $L = \\tfrac12(0.5-1)^2 = 0.125$.</p>
<p><b>Backward.</b> $\\partial L/\\partial\\hat y = \\hat y - y = -0.5$. Then $\\partial L/\\partial w_2 = (\\hat y - y)a_1 = -0.25$; $\\partial L/\\partial a_1 = (\\hat y-y)w_2 = -0.5$; through ReLU ($z_1 > 0$, so the gate passes) $\\partial L/\\partial z_1 = -0.5$; and $\\partial L/\\partial w_1 = \\partial L/\\partial z_1 \\cdot x = -0.5$.</p>
<p><b>Update</b> at $\\eta = 0.1$: $w_2 \\to 1.025$, $w_1 \\to 0.55$.</p>
<p><b>Re-forward.</b> $a_1 = 0.55$, $\\hat y = 0.564$, $L = 0.095$ — down from 0.125. ✓</p>
<p>Two things this makes concrete. Every gradient is a product of local derivatives along one path — multiply enough factors below 1 and you get the vanishing gradient (which is why the ReLU gate, passing 1 rather than $\\sigma' \\le 0.25$, was such a large practical change). And $a_1$ appears in $\\partial L/\\partial w_2$, which is <i>why</i> forward activations must be cached: that cache is the memory activation checkpointing (§4.11) trades away.</p>`)}

${H.lab('bp', 'Backprop, step by step, with your numbers', 'Every intermediate value and every gradient, recomputed as you move the inputs. Press <i>step</i> to apply the update and watch the loss fall — the same arithmetic as the worked box, under your control.')}

<h2><span class="sn">3.2.2</span> The general algorithm</h2>
${H.steps([
      '<b>Forward:</b> compute and cache $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$ and $a^{(l)} = \\phi(z^{(l)})$ for every layer.',
      '<b>Output error:</b> $\\delta^{(L)} = \\nabla_a L \\odot \\phi\'(z^{(L)})$ — for sigmoid + cross-entropy this simplifies to $\\hat y - y$.',
      '<b>Propagate:</b> $\\delta^{(l)} = (W^{(l+1)\\mathsf{T}}\\delta^{(l+1)}) \\odot \\phi\'(z^{(l)})$.',
      '<b>Gradients:</b> $\\partial L/\\partial W^{(l)} = \\delta^{(l)}a^{(l-1)\\mathsf{T}}$ and $\\partial L/\\partial b^{(l)} = \\delta^{(l)}$.'
    ])}
<p>Note the outer product in step 4: the gradient of a weight matrix is (error at this layer) × (activation from the layer below). That single line is what every framework's <code>backward()</code> implements.</p>

${H.code(`# backprop in numpy, for a 2-layer net — this is the whole idea
z1 = X @ W1 + b1;  a1 = np.maximum(0, z1)      # cache a1
z2 = a1 @ W2 + b2; p  = 1 / (1 + np.exp(-z2))

d2 = (p - y) / len(X)                          # dL/dz2 for BCE+sigmoid
dW2 = a1.T @ d2;        db2 = d2.sum(0)
d1  = (d2 @ W2.T) * (z1 > 0)                   # ReLU gate
dW1 = X.T  @ d1;        db1 = d1.sum(0)`)}

${H.probe([
      ['Why reverse-mode?', 'One scalar loss, many parameters: reverse mode costs ~one forward pass per output, forward mode one per input.'],
      ['Why must activations be cached?', 'The weight gradient is δ times the incoming activation; without the cache you would recompute the forward pass — which is exactly the trade activation checkpointing makes.'],
      ['Where do vanishing gradients come from?', 'The product of local derivatives along a path; factors below 1 compound geometrically with depth.']
    ])}`,
    labs: {
      bp: function (host) {
        let w1 = .5, w2 = 1.0, b1 = 0, b2 = 0, steps = 0, lossHist = [];
        const st = Viz.controls(host, [
          { k: 'x', label: 'input x', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'y', label: 'target y', min: -2, max: 3, step: .1, value: 1, fmt: v => v.toFixed(1) },
          { k: 'lr', label: 'learning rate η', min: .01, max: .8, step: .01, value: .1, fmt: v => v.toFixed(2) },
          { k: 'act', label: 'activation', type: 'buttons', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'sig', t: 'sigmoid' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'loss', cls: 'key' }, { k: 'dw1', label: '∂L/∂w₁' }, { k: 'dw2', label: '∂L/∂w₂' },
          { k: 'w', label: 'weights' }, { k: 'steps', label: 'steps taken' }
        ]);
        function fwd() {
          const z1 = w1 * st.x + b1;
          const a1 = st.act === 'relu' ? Math.max(0, z1) : st.act === 'tanh' ? Math.tanh(z1) : Num.sigmoid(z1);
          const da = st.act === 'relu' ? (z1 > 0 ? 1 : 0) : st.act === 'tanh' ? 1 - a1 * a1 : a1 * (1 - a1);
          const yh = w2 * a1 + b2;
          const L = .5 * (yh - st.y) * (yh - st.y);
          const dL = yh - st.y;
          return { z1, a1, da, yh, L, dL, dw2: dL * a1, db2: dL, dz1: dL * w2 * da, dw1: dL * w2 * da * st.x, db1: dL * w2 * da };
        }
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const f = fwd();
            const nodes = [
              { l: 'x', v: st.x, g: null },
              { l: 'z₁ = w₁x + b₁', v: f.z1, g: f.dz1 },
              { l: 'a₁ = φ(z₁)', v: f.a1, g: f.dL * w2 },
              { l: 'ŷ = w₂a₁ + b₂', v: f.yh, g: f.dL },
              { l: 'L = ½(ŷ−y)²', v: f.L, g: 1 }
            ];
            const bw = Math.min(122, (w - 40) / nodes.length - 8), cy = 74;
            nodes.forEach((n, i) => {
              const cx = 20 + i * (bw + 8) + bw / 2;
              Labs.roundRect(ctx, cx - bw / 2, cy - 24, bw, 48, 8);
              ctx.fillStyle = T.panel; ctx.fill(); ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(n.l, cx, cy - 10);
              ctx.fillStyle = T.blue; ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(n.v.toFixed(3), cx, cy + 10);
              if (n.g !== null) {
                ctx.fillStyle = T.red; ctx.font = '11px ui-monospace, monospace';
                ctx.fillText('∂L/∂· = ' + n.g.toFixed(3), cx, cy + 40);
              }
              if (i < nodes.length - 1) {
                const nx = 20 + (i + 1) * (bw + 8) + bw / 2;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.moveTo(cx + bw / 2 + 1, cy - 8); ctx.lineTo(nx - bw / 2 - 3, cy - 8); ctx.stroke();
                ctx.strokeStyle = T.red;
                ctx.beginPath(); ctx.moveTo(nx - bw / 2 - 3, cy + 8); ctx.lineTo(cx + bw / 2 + 1, cy + 8); ctx.stroke();
              }
            });
            ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('∂L/∂w₂ = (ŷ−y)·a₁ = ' + f.dw2.toFixed(3), 20, cy + 66);
            ctx.fillText('∂L/∂w₁ = (ŷ−y)·w₂·φ′(z₁)·x = ' + f.dw1.toFixed(3), 20, cy + 88);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('every gradient is a product of local derivatives along the path back to it', 20, cy + 112);
            // loss history
            if (lossHist.length > 1) {
              const lx = w - 160, ly = 20, lw = 140, lh = 52;
              ctx.fillStyle = T.paper; ctx.fillRect(lx, ly, lw, lh);
              ctx.strokeStyle = T.line; ctx.strokeRect(lx, ly, lw, lh);
              const mx = Math.max.apply(null, lossHist);
              ctx.strokeStyle = T.green; ctx.lineWidth = 1.8; ctx.beginPath();
              lossHist.forEach((v, i) => {
                const X = lx + lw * i / Math.max(1, lossHist.length - 1), Y = ly + lh - lh * (v / (mx || 1)) * .9 - 3;
                i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
              });
              ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText('loss over steps', lx + 4, ly + 3);
            }
            out({
              loss: f.L.toFixed(4), dw1: f.dw1.toFixed(4), dw2: f.dw2.toFixed(4),
              w: 'w₁=' + w1.toFixed(3) + ' w₂=' + w2.toFixed(3), steps: steps
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Take one step', primary: true, on: () => { const f = fwd(); w1 -= st.lr * f.dw1; w2 -= st.lr * f.dw2; b1 -= st.lr * f.db1; b2 -= st.lr * f.db2; steps++; lossHist.push(fwd().L); S.redraw(); } },
          { label: 'Run 50 steps', on: () => { for (let i = 0; i < 50; i++) { const f = fwd(); w1 -= st.lr * f.dw1; w2 -= st.lr * f.dw2; b1 -= st.lr * f.db1; b2 -= st.lr * f.db2; steps++; lossHist.push(fwd().L); } S.redraw(); } },
          { label: 'Reset to the worked example', on: () => { w1 = .5; w2 = 1; b1 = b2 = 0; steps = 0; lossHist = []; st.$set('x', 1); st.$set('y', 1); st.$set('lr', .1); S.redraw(); } }
        ]);
        Viz.note(host, 'Switch the activation to sigmoid and watch ∂L/∂w₁ shrink: the gate now passes at most 0.25 instead of 1. Two layers hides the effect; forty layers does not, and that single factor is most of the story of why deep networks were hard to train before ReLU.');
      }
    },
    quiz: [
      {
        q: 'In the worked example, $\\partial L/\\partial w_1 = -0.5$. Which chain produces it?',
        options: ['$(\\hat y - y)\\cdot a_1$', '$(\\hat y-y)\\cdot w_2\\cdot \\phi\'(z_1)\\cdot x$', '$(\\hat y - y)\\cdot x^2$', '$w_1 \\cdot x$'],
        answer: 1,
        why: 'Four local derivatives multiplied along the single path from L back to w₁: loss → output weight → activation gate → cached input.'
      },
      {
        q: 'Activation checkpointing trades…',
        options: ['accuracy for speed', 'memory for recomputation — it discards cached activations and recomputes them in the backward pass', 'batch size for depth', 'precision for range'],
        answer: 1,
        why: 'The cache exists because weight gradients need the incoming activations; dropping it saves memory at the cost of a partial extra forward pass.'
      }
    ],
    cards: [
      { q: 'Backprop in one sentence', a: 'The chain rule evaluated right-to-left with cached forward activations, so the full gradient costs about one forward pass.' },
      { q: 'The weight-gradient rule', a: '$\\partial L/\\partial W^{(l)} = \\delta^{(l)}a^{(l-1)\\mathsf{T}}$ — error at this layer times activation from below.' },
      { q: 'Worked example numbers', a: 'Forward L = 0.125; ∂L/∂w₂ = −0.25, ∂L/∂w₁ = −0.5; after one step at η=0.1, L = 0.095.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.3 */
  ML.section({
    id: 'activations', track: 'deep', num: '3.3',
    title: 'Activations and the vanishing gradient',
    lede: 'One plot explains a decade of architecture history: what a function’s derivative does to a product of forty factors.',
    html: `
<h2><span class="sn">3.3.1</span> The candidates</h2>
${H.table(['Activation', 'Definition', 'Derivative range', 'Character'], [
      ['<b>Sigmoid</b>', '$1/(1+e^{-z})$', '(0, 0.25]', 'Saturates both sides; historical; still the output layer for binary probability'],
      ['<b>Tanh</b>', '$\\tanh z$', '(0, 1]', 'Zero-centred sigmoid; better than sigmoid, still saturates'],
      ['<b>ReLU</b>', '$\\max(0,z)$', '{0, 1}', 'The default for a decade; cheap, non-saturating on the positive side; can "die"'],
      ['<b>Leaky ReLU</b>', '$\\max(\\alpha z, z)$', '{α, 1}', 'Fixes dead units at the cost of a hyperparameter'],
      ['<b>GELU</b>', '$z\\,\\Phi(z)$', 'smooth', 'Smooth ReLU; the transformer default'],
      ['<b>SiLU / Swish</b>', '$z\\,\\sigma(z)$', 'smooth', 'Very close to GELU; the gate inside SwiGLU (§4.6)']
    ])}

${H.lab('act', 'Activations and their derivatives, side by side', 'The lower panel is the one that matters: it is the factor each layer contributes to the gradient product. Note how sigmoid never exceeds 0.25 and how ReLU passes exactly 1 wherever it is active.')}

<h2><span class="sn">3.3.2</span> Why depth used to be impossible</h2>
<p>Backpropagation multiplies one such factor per layer. With sigmoid, each factor is at most 0.25, so after 10 layers the gradient is scaled by at most $0.25^{10} \\approx 10^{-6}$; after 40, $10^{-24}$. The early layers receive nothing and never learn. Exploding gradients are the same phenomenon with factors above 1.</p>

${H.lab('vanish', 'The gradient product, across depth', 'Set the activation and the depth, and watch the gradient magnitude reaching layer 1. This is computed by running an actual backward pass through a randomly initialised network — the collapse is not a cartoon.')}

<h2><span class="sn">3.3.3</span> The three fixes, and what each one does</h2>
${H.table(['Fix', 'Mechanism', 'Where you meet it'], [
      ['<b>ReLU family</b>', 'Local derivative is exactly 1 on the active side, so the product does not decay', 'Everywhere since ~2012'],
      ['<b>Residual connections</b>', '$y = x + F(x)$ gives a path whose local derivative is exactly 1, so gradients reach layer 1 unattenuated', 'Every transformer block (§4.5)'],
      ['<b>Normalisation</b>', 'Keeps the distribution each layer sees stable, so activations do not drift into saturation', 'BatchNorm, LayerNorm, RMSNorm (§3.6, §4.6)']
    ])}
${H.key('Every gradient is a product of local derivatives along a path. ReLU makes the factors 1, residuals add a path of pure 1s, and normalisation stops the inputs drifting into the flat regions. Three fixes, one problem.')}

<h3>Dying ReLU</h3>
<p>A ReLU unit whose pre-activation is negative for every input in the data receives zero gradient forever — it is dead. Causes: too large a learning rate pushing weights into a bad region, or a large negative bias. Leaky ReLU, GELU and careful initialisation all reduce it. In practice with modern initialisation and Adam it is rarely the binding problem, but it is worth being able to name.</p>

${H.probe([
      ['Why did ReLU matter so much?', 'Its derivative is exactly 1 on the active side, so the depth-wise product of local derivatives stops decaying.'],
      ['What do residual connections do to gradients?', 'They add a path whose local derivative is 1, so the gradient reaches early layers unattenuated regardless of depth.'],
      ['GELU vs ReLU?', 'GELU is a smooth, probabilistically-motivated ReLU; slightly better in transformers, marginally more expensive.']
    ])}`,
    labs: {
      act: function (host) {
        const st = Viz.controls(host, [
          { k: 'f', label: 'activation', type: 'select', value: 'relu', options: [
            { v: 'sigmoid', t: 'sigmoid' }, { v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' },
            { v: 'leaky', t: 'Leaky ReLU' }, { v: 'gelu', t: 'GELU' }, { v: 'silu', t: 'SiLU / Swish' }] },
          { k: 'cmp', label: 'compare with sigmoid', type: 'toggle', value: true }
        ], () => S.redraw());
        const F = {
          sigmoid: [z => Num.sigmoid(z), z => Num.sigmoid(z) * (1 - Num.sigmoid(z))],
          tanh: [z => Math.tanh(z), z => 1 - Math.tanh(z) ** 2],
          relu: [z => Math.max(0, z), z => z > 0 ? 1 : 0],
          leaky: [z => z > 0 ? z : .1 * z, z => z > 0 ? 1 : .1],
          gelu: [z => z * Num.normCdf(z), z => Num.normCdf(z) + z * Num.normPdf(z)],
          silu: [z => z * Num.sigmoid(z), z => Num.sigmoid(z) * (1 + z * (1 - Num.sigmoid(z)))]
        };
        const out = Viz.readout(host, [
          { k: 'max', label: 'max derivative', cls: 'key' }, { k: 'at0', label: 'derivative at 0' },
          { k: 'sat', label: 'saturates?' }, { k: 'd10', label: 'after 10 layers (max)' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const [f, df] = F[st.f];
            const half = (h - 20) / 2;
            const P1 = Viz.plot(ctx, w, h, { xd: [-5, 5], yd: [-1.2, 2.2], pad: { l: 44, r: 14, t: 12, b: h - half + 6 } })
              .frame({ ylabel: 'φ(z)' });
            P1.clip(() => {
              if (st.cmp) P1.fn(F.sigmoid[0], { color: T.faint, width: 1.4, dash: [5, 4] });
              P1.fn(f, { color: T.blue, width: 2.8 });
            });
            const P2 = Viz.plot(ctx, w, h, { xd: [-5, 5], yd: [0, 1.15], pad: { l: 44, r: 14, t: half + 18, b: 34 } })
              .frame({ xlabel: 'z', ylabel: "φ'(z)" });
            P2.clip(() => {
              if (st.cmp) P2.fn(F.sigmoid[1], { color: T.faint, width: 1.4, dash: [5, 4] });
              P2.fn(df, { color: T.red, width: 2.8 });
              P2.hline(1, { color: T.green, dash: [3, 3], width: 1 });
            });
            const grid = []; for (let z = -6; z <= 6; z += .01) grid.push(df(z));
            const mx = Math.max.apply(null, grid);
            out({
              max: mx.toFixed(3), at0: df(0).toFixed(3),
              sat: (st.f === 'sigmoid' || st.f === 'tanh') ? 'both sides' : (st.f === 'relu' ? 'negative side (dies)' : 'softly'),
              d10: Math.pow(mx, 10).toExponential(2)
            });
          }
        });
        Viz.note(host, 'The green line at 1 in the lower panel is the threshold that matters: a factor above it grows the gradient with depth, below it shrinks. Sigmoid never reaches 0.25 — ten layers of it multiply the gradient by at most one in a million.');
      },

      vanish: function (host) {
        const st = Viz.controls(host, [
          { k: 'depth', label: 'layers', min: 2, max: 40, step: 1, value: 20, fmt: v => v },
          { k: 'act', label: 'activation', type: 'buttons', value: 'sigmoid', options: [{ v: 'sigmoid', t: 'sigmoid' }, { v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'init', label: 'initialisation scale', min: .3, max: 2.5, step: .05, value: 1, fmt: v => '×' + v.toFixed(2) },
          { k: 'resid', label: 'residual connections', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'g1', label: 'gradient norm at layer 1', cls: 'key' },
          { k: 'gl', label: 'at the last layer' }, { k: 'ratio', label: 'ratio' }, { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(17), width = 24;
            const act = { sigmoid: Num.sigmoid, tanh: Math.tanh, relu: z => Math.max(0, z) }[st.act];
            const dact = {
              sigmoid: (z, a) => a * (1 - a), tanh: (z, a) => 1 - a * a, relu: z => z > 0 ? 1 : 0
            }[st.act];
            const gain = (st.act === 'relu' ? Math.sqrt(2 / width) : Math.sqrt(1 / width)) * st.init;
            const W = [];
            for (let l = 0; l < st.depth; l++) W.push(Array.from({ length: width }, () => Array.from({ length: width }, () => R.normal(0, gain))));
            // forward
            let a = Array.from({ length: width }, () => R.normal(0, 1));
            const acts = [a], zs = [];
            for (let l = 0; l < st.depth; l++) {
              const z = W[l].map(row => Num.dot(row, a));
              const na = z.map(act);
              a = st.resid ? na.map((v, i) => v + acts[l][i]) : na;
              zs.push(z); acts.push(a);
            }
            // backward
            let delta = Array.from({ length: width }, () => R.normal(0, 1));
            const norms = [];
            for (let l = st.depth - 1; l >= 0; l--) {
              const nd = new Array(width).fill(0);
              for (let k = 0; k < width; k++) {
                let s = 0;
                for (let j = 0; j < width; j++) s += W[l][j][k] * delta[j];
                nd[k] = s * dact(zs[l][k], acts[l + 1][k]);
                if (st.resid) nd[k] += delta[k];
              }
              delta = nd;
              norms.unshift(Math.sqrt(Num.dot(delta, delta)));
            }
            const logs = norms.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.depth], yd: [Math.min(-12, Math.min.apply(null, logs) - 1), Math.max(3, Math.max.apply(null, logs) + 1)] })
              .frame({ xlabel: 'layer (1 = closest to the input)', ylabel: 'log₁₀ ‖gradient‖' });
            P.clip(() => {
              P.line(norms.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.blue, width: 2.6 });
              P.hline(0, { color: T.faint, dash: [4, 4] });
              P.dots([[1, logs[0]]], { r: 5, color: logs[0] < -6 ? T.red : T.green, stroke: true });
            });
            out({
              g1: norms[0].toExponential(2), gl: norms[norms.length - 1].toExponential(2),
              ratio: (norms[0] / (norms[norms.length - 1] || 1)).toExponential(2),
              verdict: logs[0] < -6 ? 'vanished' : logs[0] > 6 ? 'exploded' : 'healthy'
            });
          }
        });
        Viz.note(host, 'Sigmoid at depth 20 vanishes by construction. Switch to ReLU and it survives; turn on residual connections and even sigmoid survives, because the identity path contributes a factor of exactly 1 at every layer. That is why a 100-layer network is trainable at all.');
      }
    },
    quiz: [
      {
        q: 'The maximum derivative of the sigmoid is 0.25. After 12 sigmoid layers, the gradient is scaled by at most…',
        options: ['0.25', '3', '$0.25^{12} \\approx 6\\times10^{-8}$', '12 × 0.25'],
        answer: 2,
        why: 'The chain rule multiplies one factor per layer; the product decays geometrically. This is the vanishing gradient in one line.'
      },
      {
        q: 'Residual connections help gradients because…',
        options: ['they reduce the number of parameters', 'they add a path whose local derivative is exactly 1', 'they normalise activations', 'they increase the learning rate'],
        answer: 1,
        why: '$\\partial(x+F(x))/\\partial x = 1 + \\partial F/\\partial x$ — the identity term keeps the product from decaying.'
      }
    ],
    cards: [
      { q: 'Why ReLU changed things', a: 'Derivative exactly 1 on the active side, so the depth-wise gradient product stops decaying.' },
      { q: 'Three fixes for vanishing gradients', a: 'ReLU-family activations, residual connections (a path of derivative 1), and normalisation to prevent saturation.' },
      { q: 'Dying ReLU', a: 'A unit negative for all inputs receives zero gradient forever; caused by too-large steps or large negative bias.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.4 */
  ML.section({
    id: 'initialisation', track: 'deep', num: '3.4',
    title: 'Initialisation: why the starting scale decides whether training happens',
    lede: 'Get the variance of the initial weights wrong by a factor of two per layer and a forty-layer network is dead before the first step.',
    html: `
<h2><span class="sn">3.4.1</span> The requirement</h2>
<p>You want the variance of activations — and of gradients — to stay roughly constant as you go through layers. If each layer multiplies activation variance by 1.5, forty layers multiply it by $1.5^{40} \\approx 10^7$ and everything saturates or overflows; by 0.7, and it decays to nothing.</p>
<p>For a layer with $n_{\\text{in}}$ inputs and independent zero-mean weights, $\\mathrm{Var}(z) = n_{\\text{in}}\\mathrm{Var}(w)\\mathrm{Var}(x)$. Setting $\\mathrm{Var}(z) = \\mathrm{Var}(x)$ gives:</p>
${H.table(['Scheme', 'Variance', 'For'], [
      ['<b>Xavier / Glorot</b>', '$2/(n_{in}+n_{out})$', 'Symmetric activations (tanh, sigmoid)'],
      ['<b>He / Kaiming</b>', '$2/n_{in}$', 'ReLU — the factor 2 compensates for zeroing half the inputs'],
      ['<b>LeCun</b>', '$1/n_{in}$', 'SELU and linear layers'],
      ['<b>Orthogonal</b>', 'orthogonal matrix × gain', 'RNNs, where repeated multiplication makes spectral radius critical'],
      ['<b>Zeros</b>', '—', 'Biases only. Zero weights make every unit identical and the network cannot break symmetry.']
    ])}

${H.lab('init', 'Activation variance across depth', 'A real forward pass through 40 layers at the initialisation scale you choose. Watch the variance track 1.0 with the correct scheme and explode or collapse when it is wrong — the y-axis is logarithmic for a reason.')}

<h2><span class="sn">3.4.2</span> The symmetry argument</h2>
<p>Initialise all weights to the same value and every unit in a layer computes the same function, receives the same gradient, and stays identical forever. Randomness is not a heuristic here; it is what makes units differentiate. (Biases can safely start at zero because the incoming weights already differ.)</p>

<h2><span class="sn">3.4.3</span> At scale</h2>
<p>Modern large models add two refinements. Residual branches are often initialised near zero (or scaled by $1/\\sqrt{2L}$) so the network starts close to the identity and depth costs nothing at step 0. And <b>μP</b> (maximal update parameterization, §4.11) rescales initialisation and learning rates by width so that hyperparameters tuned on a small proxy model transfer to a large one — which turns hyperparameter search on a billion-dollar run from impossible into a sweep on something cheap.</p>

${H.probe([
      ['Why He rather than Xavier for ReLU?', 'ReLU zeroes half the inputs, halving the variance; the factor 2 compensates exactly.'],
      ['What happens if you initialise all weights to zero?', 'Every unit computes the same thing and receives the same gradient — symmetry is never broken.'],
      ['What does μP buy you?', 'Hyperparameters found on a small model transfer to a large one, because initialisation and learning rates are scaled by width.']
    ])}`,
    labs: {
      init: function (host) {
        const st = Viz.controls(host, [
          { k: 'scheme', label: 'initialisation', type: 'select', value: 'he', options: [
            { v: 'he', t: 'He (2/n_in)' }, { v: 'xavier', t: 'Xavier (1/n_in)' }, { v: 'small', t: 'too small (0.1/n_in)' }, { v: 'big', t: 'too big (6/n_in)' }] },
          { k: 'act', label: 'activation', type: 'buttons', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }, { v: 'linear', t: 'linear' }] },
          { k: 'depth', label: 'depth', min: 5, max: 60, step: 1, value: 40, fmt: v => v },
          { k: 'width', label: 'width', min: 16, max: 256, step: 16, value: 128, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'v1', label: 'variance at layer 1', cls: 'key' }, { k: 'vl', label: 'at the last layer' },
          { k: 'dead', label: 'dead units at the top' }, { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(37);
            const n = st.width;
            const varW = { he: 2 / n, xavier: 1 / n, small: .1 / n, big: 6 / n }[st.scheme];
            const act = { relu: z => Math.max(0, z), tanh: Math.tanh, linear: z => z }[st.act];
            let a = Array.from({ length: n }, () => R.normal(0, 1));
            const vars = [Num.variance(a)];
            let deadFrac = 0;
            for (let l = 0; l < st.depth; l++) {
              const z = [];
              for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < n; j++) s += R.normal(0, Math.sqrt(varW)) * a[j];
                z.push(s);
              }
              a = z.map(act);
              vars.push(Num.variance(a));
              if (l === st.depth - 1) deadFrac = a.filter(v => Math.abs(v) < 1e-9).length / n;
            }
            const logs = vars.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [0, st.depth], yd: [Math.min(-14, Math.min.apply(null, logs) - 1), Math.max(6, Math.max.apply(null, logs) + 1)] })
              .frame({ xlabel: 'layer', ylabel: 'log₁₀ variance of activations' });
            P.clip(() => {
              P.line(vars.map((v, i) => [i, Math.log10(Math.max(1e-30, v))]), { color: T.blue, width: 2.6 });
              P.hline(0, { color: T.green, dash: [4, 4], label: 'variance = 1, the target' });
            });
            const last = vars[vars.length - 1];
            out({
              v1: vars[1].toExponential(2), vl: last.toExponential(2),
              dead: (deadFrac * 100).toFixed(0) + '%',
              verdict: last < 1e-6 ? 'collapsed — no signal reaches the output' : last > 1e6 ? 'exploded' : 'stable'
            });
          }
        });
        Viz.note(host, 'Select ReLU with Xavier (1/n) instead of He (2/n) and watch the variance halve every layer — a factor of $2^{-40}$ by the top. That single factor of two is the whole content of the He initialisation paper.');
      }
    },
    quiz: [
      {
        q: 'He initialisation uses variance $2/n_{in}$ rather than $1/n_{in}$ because…',
        options: ['ReLU is faster', 'ReLU zeroes roughly half the inputs, halving the variance; the 2 compensates', 'it prevents overfitting', 'gradients are twice as large'],
        answer: 1,
        why: 'Preserving activation variance through a ReLU requires doubling the weight variance.'
      },
      {
        q: 'Initialising every weight to the same non-zero constant means…',
        options: ['faster convergence', 'all units in a layer stay identical forever — symmetry is never broken', 'exploding gradients', 'nothing, as long as biases are random'],
        answer: 1,
        why: 'Identical weights produce identical activations and identical gradients; randomness is what differentiates units.'
      }
    ],
    cards: [
      { q: 'He vs Xavier', a: 'He $2/n_{in}$ for ReLU (compensates for zeroing half the inputs); Xavier $2/(n_{in}+n_{out})$ for symmetric activations.' },
      { q: 'Why not initialise to zero?', a: 'Symmetry: all units compute the same function and receive the same gradient forever.' },
      { q: 'μP', a: 'Width-aware parameterisation so hyperparameters transfer from a small proxy model to a large run.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.5 */
  ML.section({
    id: 'optimisers', track: 'deep', num: '3.5',
    title: 'Optimisers: SGD, momentum, Adam, AdamW, schedules',
    lede: 'Everything here is a response to the conditioning problem in §1.9 — and AdamW versus Adam is the most-asked one-line distinction in the field.',
    html: `
<h2><span class="sn">3.5.1</span> The family</h2>
${H.table(['Optimiser', 'Update', 'What it fixes'], [
      ['<b>SGD</b>', '$\\theta \\leftarrow \\theta - \\eta g$', 'Nothing; the baseline'],
      ['<b>Momentum</b>', '$v \\leftarrow \\beta v - \\eta g;\\; \\theta \\leftarrow \\theta + v$', 'Accumulates the consistent direction, cancels oscillation across ravines'],
      ['<b>RMSProp</b>', '$\\theta \\leftarrow \\theta - \\eta g/\\sqrt{\\hat s}$', 'Per-coordinate scaling — an approximation to dividing by curvature'],
      ['<b>Adam</b>', 'momentum + RMSProp with bias correction', 'Both at once; the workhorse'],
      ['<b>AdamW</b>', 'Adam with decoupled weight decay', 'Makes weight decay actually decay (see below)'],
      ['<b>Muon / Shampoo / second-order</b>', 'preconditioned updates', 'Better conditioning still, at higher cost per step; increasingly used at scale']
    ])}

<h3>Adam, written out</h3>
$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad s_t = \\beta_2 s_{t-1} + (1-\\beta_2)g_t^2$$
$$\\hat m_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat s_t = \\frac{s_t}{1-\\beta_2^t}, \\quad \\theta \\leftarrow \\theta - \\eta\\frac{\\hat m_t}{\\sqrt{\\hat s_t}+\\epsilon}$$
<p>The bias correction matters early: $m_0 = 0$ biases the first estimates toward zero, and dividing by $1-\\beta_1^t$ removes exactly that.</p>

<h2><span class="sn">3.5.2</span> Adam versus AdamW — the answer to have ready</h2>
<p>In plain Adam, L2 regularization added to the loss becomes part of $g$, so it gets divided by the same adaptive denominator $\\sqrt{\\hat s}$. Weights with large historical gradients are therefore decayed <i>less</i> — the opposite of what you wanted, and it stops behaving like weight decay at all. <b>AdamW decouples it</b>: the decay is applied directly to the weights, outside the adaptive step.</p>
$$\\theta \\leftarrow \\theta - \\eta\\left(\\frac{\\hat m}{\\sqrt{\\hat s}+\\epsilon} + \\lambda\\theta\\right)$$
${H.key('AdamW decouples weight decay from the adaptive denominator, so decay actually decays. It generalises better and it is the modern default.')}

${H.lab('opt', 'Four optimisers on the same surface', 'Real trajectories, computed step by step. Raise the condition number and watch SGD zig-zag, momentum smooth it out, and Adam march almost straight down the valley. Then use the ill-conditioned + high-learning-rate combination to make each one fail.')}

<h2><span class="sn">3.5.3</span> Learning-rate schedules</h2>
<p>The learning rate is the single most important hyperparameter, and it should not be constant. <b>Warmup</b> (linear, a few hundred to a few thousand steps) avoids the large, badly-estimated early updates that Adam's second-moment estimate produces before it has data. <b>Cosine decay</b> is the classic follow-on. <b>WSD</b> (warmup–stable–decay) holds the rate flat for most of training and decays sharply only at the end, which matches cosine's final loss while letting you checkpoint mid-run and branch — one stable trunk, several short decay phases for different data mixtures (§4.9, §4.11).</p>

${H.lab('sched', 'Schedules, drawn', 'Compare cosine, WSD, step decay and constant. The annotation marks where the "anneal" phase begins — the part of training the model is most sensitive to what it is shown.')}

${H.probe([
      ['Adam vs AdamW in one line?', 'AdamW decouples weight decay from the adaptive denominator, so decay actually decays; it generalises better and is the default.'],
      ['Why warmup?', 'Adam’s second-moment estimate is unreliable in the first steps; a large step then can move the model somewhere it never recovers from.'],
      ['When would you still use plain SGD with momentum?', 'Vision models trained long with heavy augmentation often generalise slightly better with SGD+momentum than with Adam.']
    ])}`,
    labs: {
      opt: function (host) {
        const st = Viz.controls(host, [
          { k: 'kappa', label: 'condition number κ', min: 1, max: 60, step: 1, value: 20, fmt: v => v },
          { k: 'lr', label: 'learning rate', min: -3, max: -.4, step: .05, value: -1.4, fmt: v => Math.pow(10, v).toFixed(3) },
          { k: 'surface', label: 'surface', type: 'buttons', value: 'quad', options: [{ v: 'quad', t: 'quadratic ravine' }, { v: 'rosen', t: 'Rosenbrock' }, { v: 'saddle', t: 'saddle' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sgd', label: 'SGD final loss' }, { k: 'mom', label: 'momentum' },
          { k: 'adam', label: 'Adam', cls: 'good' }, { k: 'adamw', label: 'AdamW (λ=0.05)' }
        ]);
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const k = st.kappa;
            const surf = {
              quad: { f: (x, y) => .5 * (x * x + k * y * y), g: (x, y) => [x, k * y], start: [-2.3, 1.1], xd: [-2.6, 2.6], yd: [-1.5, 1.5] },
              rosen: { f: (x, y) => Math.pow(1 - x, 2) + 10 * Math.pow(y - x * x, 2), g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], start: [-1.4, 1.6], xd: [-2, 2], yd: [-.6, 2.4] },
              saddle: { f: (x, y) => x * x - .6 * y * y + .12 * y * y * y * y, g: (x, y) => [2 * x, -1.2 * y + .48 * y * y * y], start: [-1.8, .05], xd: [-2.4, 2.4], yd: [-2, 2] }
            }[st.surface];
            const lr = Math.pow(10, st.lr);
            function run(kind) {
              let x = surf.start[0], y = surf.start[1], vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
              const path = [[x, y]];
              for (let t = 1; t <= 300; t++) {
                const [gx, gy] = surf.g(x, y);
                if (kind === 'sgd') { x -= lr * gx; y -= lr * gy; }
                else if (kind === 'mom') { vx = .9 * vx - lr * gx; vy = .9 * vy - lr * gy; x += vx; y += vy; }
                else {
                  mx = .9 * mx + .1 * gx; my = .9 * my + .1 * gy;
                  sx = .999 * sx + .001 * gx * gx; sy = .999 * sy + .001 * gy * gy;
                  const mhx = mx / (1 - Math.pow(.9, t)), mhy = my / (1 - Math.pow(.9, t));
                  const shx = sx / (1 - Math.pow(.999, t)), shy = sy / (1 - Math.pow(.999, t));
                  const step = lr * 3;
                  x -= step * (mhx / (Math.sqrt(shx) + 1e-8) + (kind === 'adamw' ? .05 * x : 0));
                  y -= step * (mhy / (Math.sqrt(shy) + 1e-8) + (kind === 'adamw' ? .05 * y : 0));
                }
                if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 20 || Math.abs(y) > 20) break;
                path.push([x, y]);
              }
              return path;
            }
            const P = Viz.plot(ctx, w, h, { xd: surf.xd, yd: surf.yd })
              .frame({ xlabel: 'θ₁', ylabel: 'θ₂' });
            const base = surf.f(surf.start[0], surf.start[1]);
            P.contours(surf.f, [base * .01, base * .05, base * .15, base * .35, base * .6, base, base * 1.6], { color: T.faint, alpha: .5 });
            const runs = { sgd: run('sgd'), mom: run('mom'), adam: run('adam'), adamw: run('adamw') };
            const cols = { sgd: T.faint, mom: T.amber, adam: T.blue, adamw: T.green };
            P.clip(() => {
              Object.keys(runs).forEach(kk => P.line(runs[kk], { color: cols[kk], width: kk === 'adam' ? 2.6 : 1.8 }));
              P.dots([surf.start], { r: 5, color: T.text, stroke: true });
            });
            const fin = kk => { const p = runs[kk][runs[kk].length - 1]; return surf.f(p[0], p[1]); };
            out({
              sgd: fin('sgd').toExponential(1), mom: fin('mom').toExponential(1),
              adam: fin('adam').toExponential(1), adamw: fin('adamw').toExponential(1)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().faint, t: 'SGD' }, { c: Viz.theme().amber, t: 'momentum' },
          { c: Viz.theme().blue, t: 'Adam' }, { c: Viz.theme().green, t: 'AdamW' }
        ]);
        Viz.note(host, 'On the saddle surface, plain SGD stalls near the origin where the gradient is tiny in the escape direction; momentum carries through. On the ravine, Adam’s per-coordinate scaling is the visible win. Push the learning rate up and every method fails — no optimiser rescues a step size above the stability limit (§1.9).');
      },

      sched: function (host) {
        const st = Viz.controls(host, [
          { k: 'total', label: 'total steps', min: 1000, max: 100000, step: 1000, value: 20000, fmt: v => (v / 1000) + 'k' },
          { k: 'warm', label: 'warmup steps', min: 0, max: 5000, step: 100, value: 1000, fmt: v => v },
          { k: 'peak', label: 'peak LR', min: -4.3, max: -2, step: .1, value: -3.5, fmt: v => Math.pow(10, v).toExponential(1) },
          { k: 'stable', label: 'WSD stable fraction', min: .3, max: .95, step: .05, value: .8, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const peak = Math.pow(10, st.peak);
            const warm = t => Math.min(1, t / Math.max(1, st.warm));
            const cosine = t => peak * warm(t) * (t <= st.warm ? 1 : .5 * (1 + Math.cos(Math.PI * (t - st.warm) / (st.total - st.warm))));
            const wsd = t => {
              const decayStart = st.warm + (st.total - st.warm) * st.stable;
              if (t <= st.warm) return peak * warm(t);
              if (t <= decayStart) return peak;
              return peak * Math.max(0, 1 - (t - decayStart) / (st.total - decayStart));
            };
            const step = t => peak * warm(t) * Math.pow(.1, Math.floor(3 * t / st.total));
            const P = Viz.plot(ctx, w, h, { xd: [0, st.total], yd: [0, peak * 1.15] })
              .frame({ xlabel: 'training step', ylabel: 'learning rate', yfmt: v => v.toExponential(0), xfmt: v => (v / 1000).toFixed(0) + 'k' });
            P.clip(() => {
              P.fn(t => peak, { color: T.faint, width: 1.4, dash: [4, 4] });
              P.fn(step, { color: T.amber, width: 1.8, n: 400 });
              P.fn(cosine, { color: T.blue, width: 2.6, n: 400 });
              P.fn(wsd, { color: T.green, width: 2.6, n: 400 });
              P.vline(st.warm, { color: T.red, label: 'warmup ends' });
              const ds = st.warm + (st.total - st.warm) * st.stable;
              P.vline(ds, { color: T.green, dash: [3, 3], label: 'anneal begins' });
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().blue, t: 'cosine' }, { c: Viz.theme().green, t: 'WSD (warmup–stable–decay)' },
          { c: Viz.theme().amber, t: 'step decay' }, { c: Viz.theme().faint, t: 'constant' }
        ]);
        Viz.note(host, 'Cosine requires committing to a total step count in advance — awkward when you might extend the run. WSD does not: you can stop the stable phase whenever you like, decay from a checkpoint, and branch several short decays from one trunk. That property is why it has become common for large pretraining runs.');
      }
    },
    quiz: [
      {
        q: 'In plain Adam, adding L2 to the loss behaves badly because…',
        options: ['it is applied twice', 'the penalty gradient is divided by the same adaptive denominator, so high-gradient weights decay less', 'it conflicts with momentum', 'it makes the loss non-convex'],
        answer: 1,
        why: 'Decoupling it (AdamW) restores the intended behaviour — decay proportional to the weight, independent of gradient history.'
      },
      {
        q: 'Warmup exists mainly to…',
        options: ['save compute', 'avoid large early steps when Adam’s second-moment estimate is still unreliable', 'increase the batch size', 'prevent overfitting'],
        answer: 1,
        why: 'Early updates with a badly-estimated denominator can be enormous; a linear ramp keeps the first few hundred steps small.'
      },
      {
        q: 'WSD schedules are preferred over cosine at scale because…',
        options: ['they reach lower loss', 'they do not require committing to a total step count and allow branching from one stable trunk', 'they need no warmup', 'they are simpler to implement'],
        answer: 1,
        why: 'Match cosine’s final loss while keeping mid-run checkpoints useful — several decay phases from one trunk for different data mixtures.'
      }
    ],
    cards: [
      { q: 'Adam update', a: '$m,s$ EMAs of gradient and squared gradient, bias-corrected; $\\theta \\leftarrow \\theta - \\eta\\hat m/(\\sqrt{\\hat s}+\\epsilon)$.' },
      { q: 'AdamW vs Adam', a: 'Decoupled weight decay applied outside the adaptive denominator, so decay actually decays. The modern default.' },
      { q: 'WSD schedule', a: 'Warmup, long stable phase, sharp final decay — matches cosine and allows branching mid-run.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.6 */
  ML.section({
    id: 'normalisation', track: 'deep', num: '3.6',
    title: 'Normalisation and regularization for networks',
    lede: 'The stabilisers that make depth trainable, and the four ways to stop a network memorising.',
    html: `
<h2><span class="sn">3.6.1</span> Normalisation, by axis</h2>
<p>Every normalisation computes a mean and a variance over some axes, subtracts and divides, then applies a learned scale and shift. The only difference is <i>which axes</i>.</p>
${H.table(['Method', 'Normalises over', 'Depends on batch?', 'Used in'], [
      ['<b>BatchNorm</b>', 'the batch, per channel', 'Yes — and that is its weakness', 'CNNs at reasonable batch sizes'],
      ['<b>LayerNorm</b>', 'the features of one example', 'No', 'Transformers, RNNs'],
      ['<b>RMSNorm</b>', 'the features, no mean subtraction', 'No', 'Modern LLMs (§4.6) — cheaper, one less reduction'],
      ['<b>GroupNorm</b>', 'groups of channels within one example', 'No', 'Vision at small batch sizes']
    ])}
<p>BatchNorm's batch dependence is the practical issue: it behaves differently at train and inference (running statistics), it breaks with tiny batches, and it interacts badly with sequence models where different positions have different statistics. That is why transformers use LayerNorm and its cheaper cousin RMSNorm.</p>

${H.lab('norm', 'What normalisation does to the distribution each layer sees', 'Activation histograms at increasing depth, with and without normalisation. Without it the distribution drifts and widens until the activation saturates; with it, every layer sees something stable.')}

<h2><span class="sn">3.6.2</span> Regularization, four ways</h2>
${H.table(['Method', 'Mechanism', 'Note'], [
      ['<b>Weight decay</b>', 'Pull toward small weights (a Gaussian prior, §1.5)', 'Use AdamW so it is decoupled (§3.5)'],
      ['<b>Dropout</b>', 'Randomly zero units during training; scale at inference', 'Prevents co-adaptation. Largely replaced by other methods in large transformers, still standard in smaller nets'],
      ['<b>Early stopping</b>', 'Halt at the best validation epoch', 'The cheapest regularizer there is'],
      ['<b>Data augmentation</b>', 'Expand the effective dataset with label-preserving transforms', 'The most effective of all when the invariances are known']
    ])}

${H.lab('dropout', 'Dropout, and what it does to the boundary', 'Train the same network with and without dropout on noisy data. Watch the decision boundary go from jagged and confident to smoother and better-calibrated — and watch the training loss get <i>worse</i> while the test accuracy improves. That gap is the entire point.')}

<h3>Why dropout works, in one line</h3>
<p>Each forward pass trains a randomly-thinned sub-network, and inference averages over them — an ensemble (§2.7) obtained for free. It also prevents units from co-adapting to fix each other's mistakes, which forces each unit to be independently useful.</p>

${H.probe([
      ['LayerNorm or BatchNorm for a transformer?', 'LayerNorm (or RMSNorm) — no batch dependence, and per-position statistics are meaningless in a batch of variable-length sequences.'],
      ['What does dropout actually average over?', 'An exponential family of thinned sub-networks; inference with scaled weights approximates that ensemble.'],
      ['Which regularizer is cheapest?', 'Early stopping — one validation curve and no extra compute.']
    ])}`,
    labs: {
      norm: function (host) {
        const st = Viz.controls(host, [
          { k: 'norm', label: 'normalisation', type: 'buttons', value: 'none', options: [{ v: 'none', t: 'none' }, { v: 'layer', t: 'LayerNorm' }, { v: 'rms', t: 'RMSNorm' }] },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'scale', label: 'weight scale', min: .5, max: 2.5, step: .05, value: 1.6, fmt: v => '×' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sd1', label: 'sd at layer 1', cls: 'key' }, { k: 'sd8', label: 'sd at layer 8' },
          { k: 'sat', label: 'saturated units at layer 8' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(53), n = 256, depth = 8;
            const act = st.act === 'tanh' ? Math.tanh : (z => Math.max(0, z));
            let a = Array.from({ length: n }, () => R.normal(0, 1));
            const layers = [];
            for (let l = 0; l < depth; l++) {
              const z = [];
              for (let i = 0; i < n; i++) {
                let s = 0;
                for (let j = 0; j < 64; j++) s += R.normal(0, st.scale / 8) * a[(j * 7 + i) % n];
                z.push(s);
              }
              let zz = z;
              if (st.norm === 'layer') { const m = Num.mean(z), sd = Num.sd(z) || 1; zz = z.map(v => (v - m) / sd); }
              else if (st.norm === 'rms') { const rms = Math.sqrt(Num.mean(z.map(v => v * v))) || 1; zz = z.map(v => v / rms); }
              a = zz.map(act);
              layers.push(a.slice());
            }
            const colW = (w - 40) / depth;
            layers.forEach((la, l) => {
              const hh = Num.hist(la, 26, -2.2, 2.2);
              const mx = Math.max.apply(null, hh.bins) || 1;
              const x0 = 24 + l * colW;
              hh.bins.forEach((c, i) => {
                const y = 40 + (i / 26) * (h - 90);
                ctx.fillStyle = T.blue; ctx.globalAlpha = .25 + .6 * (c / mx);
                ctx.fillRect(x0, y, Math.max(2, colW * .78 * (c / mx)), (h - 90) / 26 - 1);
                ctx.globalAlpha = 1;
              });
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText('L' + (l + 1), x0, 34);
              ctx.fillStyle = T.faint; ctx.textBaseline = 'top';
              ctx.fillText('σ=' + Num.sd(la).toFixed(2), x0, h - 44);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('distribution of activations at each depth (each column is a histogram)', 24, h - 26);
            const sat = layers[depth - 1].filter(v => Math.abs(v) > .99).length / n;
            out({
              sd1: Num.sd(layers[0]).toFixed(3), sd8: Num.sd(layers[depth - 1]).toFixed(3),
              sat: (sat * 100).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'With weight scale 1.6 and no normalisation, tanh activations pile up at ±1 by layer 4 — every unit saturated, every derivative near zero, nothing learns. LayerNorm and RMSNorm both hold the distribution steady, and RMSNorm does it without computing a mean, which is the cheaper reduction modern LLMs prefer.');
      },

      dropout: function (host) {
        let netA = null, netB = null, trained = 0;
        const st = Viz.controls(host, [
          { k: 'p', label: 'dropout rate', min: 0, max: .6, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'label noise in the data', min: 0, max: .3, step: .02, value: .12, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'width', label: 'width', min: 4, max: 24, step: 2, value: 16, fmt: v => v }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'trA', label: 'train acc · no dropout' }, { k: 'teA', label: 'test acc · no dropout' },
          { k: 'trB', label: 'train acc · dropout' }, { k: 'teB', label: 'test acc · dropout', cls: 'good' }
        ]);
        let data, test;
        function reset() {
          const R = Num.rng(9);
          data = Num.dataset('moons', 120, .28, 4);
          data.y = data.y.map(v => R() < st.noise ? 1 - v : v);
          test = Num.dataset('moons', 300, .28, 77);
          netA = Num.mlp([2, st.width, st.width, 1], { act: 'tanh', seed: 5 });
          netB = Num.mlp([2, st.width, st.width, 1], { act: 'tanh', seed: 5 });
          trained = 0; S.redraw();
        }
        function trainBoth(steps) {
          const R = Num.rng(1234 + trained);
          for (let i = 0; i < steps; i++) {
            netA.trainBatch(data.X, data.y, .05, 0);
            // dropout approximation: train on a random subset of inputs with noise injection
            const mask = data.X.map(x => [x[0] * (R() > st.p ? 1 / (1 - st.p) : 0), x[1] * (R() > st.p ? 1 / (1 - st.p) : 0)]);
            netB.trainBatch(mask, data.y, .05, st.p * .02);
            trained++;
          }
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const half = (w - 50) / 2;
            [[netA, 40, 'no dropout'], [netB, 40 + half + 22, 'dropout p = ' + st.p.toFixed(2)]].forEach(([net, x0, title]) => {
              const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-2.4, 2.4], pad: { l: x0, r: w - x0 - half, t: 26, b: 34 } })
                .frame({ xticks: [], yticks: [] });
              P.clip(() => Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 5 }));
              P.clip(() => Labs.points(P, data.X, data.y, { r: 2.8 }));
              ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
              ctx.fillText(title, x0, 8);
            });
            const accOf = (net, d) => Num.mean(d.X.map((x, i) => ((net.predict(x) > .5 ? 1 : 0) === d.y[i]) ? 1 : 0));
            out({
              trA: (accOf(netA, data) * 100).toFixed(1) + '%', teA: (accOf(netA, test) * 100).toFixed(1) + '%',
              trB: (accOf(netB, data) * 100).toFixed(1) + '%', teB: (accOf(netB, test) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 300 steps', primary: true, on: () => { trainBoth(300); S.redraw(); } },
          { label: 'Train 1500 steps', on: () => { trainBoth(1500); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Train both to 1500 steps with label noise on: the left network reaches higher training accuracy by bending around the mislabelled points, and pays for it on test. The right one refuses to. Higher training loss with better test accuracy is what a working regularizer looks like.');
      }
    },
    quiz: [
      {
        q: 'Transformers use LayerNorm/RMSNorm rather than BatchNorm because…',
        options: ['it is more accurate', 'batch statistics are unstable across variable-length sequences and differ between train and inference', 'BatchNorm is slower', 'LayerNorm has fewer parameters'],
        answer: 1,
        why: 'LayerNorm normalises within one example, so nothing depends on what else is in the batch; RMSNorm additionally drops the mean subtraction.'
      },
      {
        q: 'Your training loss rises when you add dropout but test accuracy improves. This means…',
        options: ['dropout is misconfigured', 'the regularizer is working as intended', 'the learning rate is too high', 'the model is underfitting'],
        answer: 1,
        why: 'Regularization deliberately trades training fit for generalisation. The gap narrowing is the signal you want.'
      }
    ],
    cards: [
      { q: 'LayerNorm vs BatchNorm vs RMSNorm', a: 'Per-example features / per-channel over the batch / per-example features with no mean subtraction (cheapest, modern LLM default).' },
      { q: 'Why dropout works', a: 'Each pass trains a thinned sub-network; inference averages the ensemble and units cannot co-adapt.' },
      { q: 'The four regularizers', a: 'Weight decay (AdamW), dropout, early stopping, data augmentation — augmentation is usually the strongest when invariances are known.' }
    ]
  });
})();
