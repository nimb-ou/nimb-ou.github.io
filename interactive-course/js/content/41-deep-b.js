/* ============================================================
   PART 3 — Deep learning (3.7 – 3.10): CNNs, RNNs, embeddings
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.7 */
  ML.section({
    id: 'cnn', track: 'deep', num: '3.7',
    title: 'Convolutional networks',
    lede: 'The architecture that made an inductive bias explicit: locality and translation invariance. An excellent bet for images, a poor one for long-range syntax — which is exactly the sentence §4.1 builds on.',
    html: `
<h2><span class="sn">3.7.1</span> What a convolution is</h2>
<p>Slide a small learned kernel over the input and take a dot product at every position. Three consequences follow immediately:</p>
<ul>
<li><b>Parameter sharing</b> — the same 3×3 kernel is used everywhere, so a $224\\times224$ image costs 9 weights per filter rather than 50,176.</li>
<li><b>Locality</b> — each output looks only at a neighbourhood, encoding the assumption that nearby pixels are related.</li>
<li><b>Translation equivariance</b> — shift the input and the feature map shifts with it. Pooling then converts equivariance into approximate invariance.</li>
</ul>
${H.key('An architecture is a bet about structure. CNNs bet on locality and translation; that bet is excellent for images and poor for language, which is why attention won the frontier.')}

${H.lab('conv', 'Convolution, kernel by kernel', 'Pick a kernel and watch it applied to the image, one window at a time. The classic edge detectors are not magic — they are the numbers you can read in the grid, and the network learns numbers like these rather than being given them.')}

<h2><span class="sn">3.7.2</span> Shapes, stride, padding, receptive field</h2>
$$H_{\\text{out}} = \\left\\lfloor\\frac{H_{\\text{in}} + 2p - k}{s}\\right\\rfloor + 1$$
<p>with kernel $k$, padding $p$, stride $s$. "Same" padding means $p=(k-1)/2$ with $s=1$, which preserves the spatial size. Parameters in a conv layer: $k^2 \\times C_{\\text{in}} \\times C_{\\text{out}} + C_{\\text{out}}$ — independent of image size, which is the whole efficiency argument.</p>
<p>The <b>receptive field</b> is how much of the input one output unit can see. Stacking $L$ layers of $3\\times3$ convolutions gives a receptive field of $2L+1$; that is why two $3\\times3$s (RF 5, 18 params per channel pair) are preferred to one $5\\times5$ (RF 5, 25 params) — same view, fewer parameters, an extra nonlinearity.</p>

${H.lab('rf', 'Receptive field growth, computed', 'Add layers, change stride and dilation, and watch how much of the input one output neuron sees. The stride column is what makes deep CNNs see globally without quadratic cost — and the analogous question for transformers is §4.4’s sliding-window attention.')}

<h2><span class="sn">3.7.3</span> The architecture lineage, and what each one contributed</h2>
${H.table(['Model', 'Year', 'The idea it added'], [
      ['LeNet-5', '1998', 'Convolution + pooling + fully connected, trained end to end'],
      ['AlexNet', '2012', 'ReLU, dropout, GPUs, augmentation — the moment deep learning became practical'],
      ['VGG', '2014', 'Depth from stacked 3×3 kernels only'],
      ['Inception', '2014', 'Multiple kernel sizes in parallel; 1×1 convolutions as cheap channel mixing'],
      ['ResNet', '2015', '<b>Residual connections</b> — the single most transferable idea in the list, and the reason transformers train at depth (§4.5)'],
      ['DenseNet / EfficientNet', '2017–19', 'Feature reuse; compound scaling of depth, width and resolution'],
      ['ConvNeXt', '2022', 'A CNN modernised with transformer-era training recipes — competitive again, which is a useful corrective to "attention is all you need" as a slogan']
    ])}

<h2><span class="sn">3.7.4</span> Pooling, and its decline</h2>
<p>Max pooling takes the strongest response in a window, giving small translation invariance and downsampling for free. Modern architectures often use strided convolutions instead (learnable downsampling) and global average pooling before the classifier, which has no parameters and generalises better than a large flattened dense layer.</p>

${H.probe([
      ['Why two 3×3 rather than one 5×5?', 'Same receptive field, fewer parameters, one extra nonlinearity.'],
      ['What inductive bias does a CNN encode?', 'Locality and translation equivariance, with parameter sharing making it cheap.'],
      ['Why do transformers beat CNNs on language?', 'Language dependencies are long-range and not translation-structured; attention gives O(1) path length between any two positions (§4.1).']
    ])}`,
    labs: {
      conv: function (host) {
        const N = 28;
        // procedural "image": a shape with edges and texture
        const img = [];
        for (let i = 0; i < N; i++) {
          const row = [];
          for (let j = 0; j < N; j++) {
            const cx = j - N / 2, cy = i - N / 2;
            let v = 0.15;
            if (Math.abs(cx) < 7 && Math.abs(cy) < 9) v = 0.85;             // block
            if (cx * cx + cy * cy < 16) v = 0.35;                            // hole
            if (i > 20) v = 0.6 + 0.25 * Math.sin(j * 1.4);                  // texture
            row.push(v);
          }
          img.push(row);
        }
        const kernels = {
          edgeV: { k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], t: 'vertical edges (Sobel)' },
          edgeH: { k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], t: 'horizontal edges' },
          blur: { k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], t: 'blur (box)' },
          sharp: { k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], t: 'sharpen' },
          lap: { k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], t: 'Laplacian (all edges)' },
          id: { k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], t: 'identity' }
        };
        const st = Viz.controls(host, [
          { k: 'kern', label: 'kernel', type: 'select', value: 'edgeV', options: Object.keys(kernels).map(k => ({ v: k, t: kernels[k].t })) },
          { k: 'stride', label: 'stride', min: 1, max: 3, step: 1, value: 1, fmt: v => v },
          { k: 'relu', label: 'apply ReLU after', type: 'toggle', value: true },
          { k: 'pool', label: 'then 2×2 max pool', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'in', label: 'input shape', cls: 'key' }, { k: 'outsh', label: 'output shape' },
          { k: 'params', label: 'parameters used' }, { k: 'dense', label: 'a dense layer would need' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const K = kernels[st.kern].k;
            const s = st.stride, outN = Math.floor((N - 3) / s) + 1;
            let outImg = [];
            for (let i = 0; i < outN; i++) {
              const row = [];
              for (let j = 0; j < outN; j++) {
                let acc = 0;
                for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) acc += K[a][b] * img[i * s + a][j * s + b];
                row.push(st.relu ? Math.max(0, acc) : acc);
              }
              outImg.push(row);
            }
            if (st.pool) {
              const pN = Math.floor(outN / 2), pooled = [];
              for (let i = 0; i < pN; i++) {
                const row = [];
                for (let j = 0; j < pN; j++) {
                  row.push(Math.max(outImg[2 * i][2 * j], outImg[2 * i][2 * j + 1], outImg[2 * i + 1][2 * j], outImg[2 * i + 1][2 * j + 1]));
                }
                pooled.push(row);
              }
              outImg = pooled;
            }
            const cell = Math.min((h - 70) / N, (w * .28) / N);
            const drawGrid = (m, ox, oy, cs, title, signed) => {
              const flat = m.flat();
              const mx = Math.max.apply(null, flat.map(Math.abs)) || 1;
              m.forEach((row, i) => row.forEach((v, j) => {
                let c;
                if (signed) {
                  const t = v / mx;
                  c = t >= 0 ? 'rgb(' + Math.round(30 + 200 * t) + ',' + Math.round(50 + 120 * t) + ',' + Math.round(90 + 40 * t) + ')'
                             : 'rgb(' + Math.round(30 - 20 * t) + ',' + Math.round(40 - 40 * t) + ',' + Math.round(90 - 140 * t) + ')';
                } else {
                  const g = Math.round(255 * Math.max(0, Math.min(1, v)));
                  c = 'rgb(' + g + ',' + g + ',' + Math.min(255, g + 12) + ')';
                }
                ctx.fillStyle = c;
                ctx.fillRect(ox + j * cs, oy + i * cs, cs + .5, cs + .5);
              }));
              ctx.strokeStyle = T.line; ctx.strokeRect(ox, oy, m[0].length * cs, m.length * cs);
              ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
              ctx.fillText(title, ox, oy - 5);
            };
            drawGrid(img, 14, 30, cell, 'input ' + N + '×' + N, false);
            // kernel
            const kx = 14 + N * cell + 22, kcell = 24;
            ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('kernel 3×3', kx, 25);
            K.forEach((row, i) => row.forEach((v, j) => {
              ctx.fillStyle = v > 0 ? 'rgba(90,160,255,.28)' : v < 0 ? 'rgba(230,90,80,.28)' : T.panel;
              ctx.fillRect(kx + j * kcell, 30 + i * kcell, kcell - 2, kcell - 2);
              ctx.strokeStyle = T.line; ctx.strokeRect(kx + j * kcell, 30 + i * kcell, kcell - 2, kcell - 2);
              ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(Math.abs(v) < .2 && v !== 0 ? v.toFixed(2) : String(v), kx + j * kcell + kcell / 2 - 1, 30 + i * kcell + kcell / 2 - 1);
            }));
            ctx.fillStyle = T.muted; ctx.font = '18px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText('∗', kx + 1.5 * kcell, 30 + 3 * kcell + 22);
            const outCell = Math.min((h - 70) / outImg.length, (w * .28) / outImg.length);
            drawGrid(outImg, kx + 3 * kcell + 30, 30, outCell, 'output ' + outImg.length + '×' + outImg[0].length + (st.relu ? ' (ReLU)' : '') + (st.pool ? ' + pool' : ''), true);
            out({
              in: N + '×' + N, outsh: outImg.length + '×' + outImg[0].length,
              params: '9 + 1 bias',
              dense: (N * N * outN * outN).toLocaleString()
            });
          }
        });
        Viz.note(host, 'The readout compares 10 parameters against the ~600,000 a fully-connected layer of the same output size would need. That ratio — parameter sharing — is why convolutions made image models trainable in 1998 and why they remain the cheapest way to process a grid.');
      },

      rf: function (host) {
        const st = Viz.controls(host, [
          { k: 'layers', label: 'conv layers', min: 1, max: 20, step: 1, value: 6, fmt: v => v },
          { k: 'k', label: 'kernel size', min: 3, max: 7, step: 2, value: 3, fmt: v => v + '×' + v },
          { k: 'stride', label: 'stride (every layer)', min: 1, max: 2, step: 1, value: 1, fmt: v => v },
          { k: 'dil', label: 'dilation growth', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rf', label: 'receptive field', cls: 'key' }, { k: 'params', label: 'params per channel pair' },
          { k: 'need', label: 'layers to cover 224px' }, { k: 'attn', label: 'attention would need' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const rfs = []; let rf = 1, jump = 1;
            for (let l = 1; l <= st.layers; l++) {
              const d = st.dil ? Math.pow(2, l - 1) : 1;
              rf = rf + (st.k - 1) * d * jump;
              jump = jump * st.stride;
              rfs.push(rf);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, Math.max(2, st.layers)], yd: [0, Math.max(30, rfs[rfs.length - 1] * 1.1)] })
              .frame({ xlabel: 'layer', ylabel: 'receptive field (pixels)' });
            P.clip(() => {
              P.line(rfs.map((v, i) => [i + 1, v]), { color: T.blue, width: 2.6 });
              P.dots(rfs.map((v, i) => [i + 1, v]), { r: 3.6, color: T.blue });
              P.hline(224, { color: T.red, dash: [4, 4], label: 'a 224px image' });
            });
            let need = 0, r2 = 1, j2 = 1;
            while (r2 < 224 && need < 500) { const d = st.dil ? Math.pow(2, need) : 1; r2 += (st.k - 1) * d * j2; j2 *= st.stride; need++; }
            out({
              rf: rfs[rfs.length - 1] + ' px',
              params: (st.k * st.k * st.layers).toLocaleString(),
              need: need + ' layers', attn: '1 layer (O(n²) cost)'
            });
          }
        });
        Viz.note(host, 'Turn on dilation growth: the receptive field doubles per layer instead of growing linearly, covering an image in a handful of layers. That is the CNN answer to long-range dependency — and it is exactly the trade attention removes by making the path length 1 at quadratic cost (§4.1).');
      }
    },
    quiz: [
      {
        q: 'Two stacked 3×3 convolutions are usually preferred to one 5×5 because…',
        options: ['they are faster to compute', 'same receptive field, fewer parameters, and an extra nonlinearity', 'they have a larger receptive field', 'they avoid padding'],
        answer: 1,
        why: 'RF 5 either way; 18 vs 25 parameters per channel pair, plus one more activation between them.'
      },
      {
        q: 'A CNN’s inductive bias is…',
        options: ['sequential dependence', 'locality and translation equivariance with parameter sharing', 'permutation invariance', 'no bias at all'],
        answer: 1,
        why: 'That bet is excellent for images and poor for long-range syntax — the observation §4.1 starts from.'
      }
    ],
    cards: [
      { q: 'Conv output shape', a: '$\\lfloor (H+2p-k)/s\\rfloor + 1$; parameters $k^2C_{in}C_{out}+C_{out}$, independent of image size.' },
      { q: 'Receptive field of L stacked 3×3 layers', a: '$2L+1$ with stride 1; dilation makes it grow geometrically.' },
      { q: 'ResNet’s contribution', a: 'Residual connections — a gradient path of derivative 1, which is what makes very deep stacks (including transformers) trainable.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.8 */
  ML.section({
    id: 'rnn', track: 'deep', num: '3.8',
    title: 'Recurrent networks, LSTM, and why they lost',
    lede: 'The right prior for language and the wrong computational shape for a GPU. Understanding both halves is what makes §4.1 land.',
    html: `
<h2><span class="sn">3.8.1</span> The recurrence</h2>
$$h_t = \\phi(W_h h_{t-1} + W_x x_t + b), \\qquad y_t = W_y h_t$$
<p>One hidden state carries everything the model knows about the past. Parameters are shared across time, so the network handles any sequence length — an elegant inductive bias for language, where the past genuinely conditions the present.</p>

<h2><span class="sn">3.8.2</span> Backpropagation through time, and the two failures</h2>
<p>Unroll the recurrence and backpropagate: the gradient from step $T$ back to step $t$ contains the product $\\prod_{k=t}^{T} W_h^\\mathsf{T}\\mathrm{diag}(\\phi')$. If the largest eigenvalue of that repeated factor is below 1 the gradient <b>vanishes</b>; above 1 it <b>explodes</b>. Unlike depth in a feedforward net, here the <i>same</i> matrix is applied repeatedly, so the effect is geometric in sequence length and there is no escaping it by adding more parameters.</p>

${H.lab('bptt', 'Gradient decay through time, measured', 'A real unrolled recurrence, with the gradient magnitude reaching each earlier step computed by backpropagation. Move the spectral radius past 1 and watch explosion; below and watch a 40-step memory disappear by step 10.')}

<h2><span class="sn">3.8.3</span> LSTM and GRU: additive memory</h2>
<p>The LSTM adds a <b>cell state</b> updated additively rather than multiplicatively, guarded by gates:</p>
$$f_t = \\sigma(W_f[h_{t-1},x_t]),\\quad i_t = \\sigma(W_i[\\cdot]),\\quad o_t = \\sigma(W_o[\\cdot])$$
$$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tanh(W_c[\\cdot]), \\qquad h_t = o_t \\odot \\tanh(c_t)$$
<p>The point is $c_t = f_t \\odot c_{t-1} + \\ldots$: when the forget gate is near 1 the cell state passes through nearly unchanged, giving the gradient a near-identity path across time — the same trick as a residual connection, invented for sequences a decade earlier. The GRU merges the input and forget gates into one update gate: fewer parameters, usually similar quality.</p>

${H.lab('lstm', 'The gates, watched on a real sequence', 'A small LSTM-style cell run over a sequence you control. Watch the forget gate hold a value across many steps and the input gate decide when to overwrite — memory as an explicit, learnable decision.')}

<h2><span class="sn">3.8.4</span> Why they lost</h2>
<p>Gating fixed the gradient problem well enough for practical sequence lengths. What it could not fix is the computational shape: <mark>recurrence cannot be parallelised across time, because step $t$ needs step $t-1$.</mark> Training on trillions of tokens requires saturating a GPU with one enormous matrix multiply, and a sequential loop cannot do that. Attention can (§4.1). That is the whole story — the architecture that won did so on parallelism, not on modelling elegance.</p>
${H.note('The wheel turns: state-space models (§4.8) are recurrences again, engineered so the recurrence is associative and therefore parallelisable by a scan. Same prior, different computational shape.')}

${H.probe([
      ['Why do RNN gradients vanish differently from feedforward ones?', 'The same weight matrix is applied at every step, so the effect is geometric in sequence length; you cannot fix it by changing layer widths.'],
      ['What does the LSTM cell state buy?', 'An additive path across time — a near-identity gradient route when the forget gate is open, like a residual connection.'],
      ['Why did transformers replace RNNs?', 'Parallelism across time plus O(1) path length between positions; recurrence forces a sequential loop that cannot saturate a GPU.']
    ])}`,
    labs: {
      bptt: function (host) {
        const st = Viz.controls(host, [
          { k: 'len', label: 'sequence length', min: 5, max: 60, step: 1, value: 40, fmt: v => v },
          { k: 'rho', label: 'spectral radius of Wₕ', min: .5, max: 1.5, step: .01, value: .85, fmt: v => v.toFixed(2) },
          { k: 'act', label: 'activation', type: 'buttons', value: 'tanh', options: [{ v: 'tanh', t: 'tanh' }, { v: 'relu', t: 'ReLU' }] },
          { k: 'lstm', label: 'LSTM-style additive path (forget≈1)', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'g1', label: 'gradient reaching step 1', cls: 'key' }, { k: 'half', label: 'effective memory (10× decay)' },
          { k: 'verdict', label: 'verdict' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31), n = 16;
            const W = Array.from({ length: n }, () => Array.from({ length: n }, () => R.normal(0, st.rho / Math.sqrt(n))));
            const act = st.act === 'tanh' ? Math.tanh : (z => Math.max(0, z));
            const dact = st.act === 'tanh' ? (a => 1 - a * a) : ((a, z) => z > 0 ? 1 : 0);
            let hh = Array.from({ length: n }, () => R.normal(0, .4));
            const hs = [hh], zs = [];
            for (let t = 0; t < st.len; t++) {
              const z = W.map(row => Num.dot(row, hh) + R.normal(0, .1));
              const nh = z.map(act);
              hh = st.lstm ? nh.map((v, i) => 0.92 * hs[t][i] + 0.25 * v) : nh;
              zs.push(z); hs.push(hh);
            }
            let delta = Array.from({ length: n }, () => R.normal(0, 1));
            const norms = [];
            for (let t = st.len - 1; t >= 0; t--) {
              const nd = new Array(n).fill(0);
              for (let k = 0; k < n; k++) {
                let s = 0;
                for (let j = 0; j < n; j++) s += W[j][k] * delta[j] * dact(hs[t + 1][j], zs[t][j]);
                nd[k] = st.lstm ? 0.92 * delta[k] + 0.25 * s : s;
              }
              delta = nd;
              norms.unshift(Math.sqrt(Num.dot(delta, delta)));
            }
            const logs = norms.map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.len], yd: [Math.min(-12, Math.min.apply(null, logs) - .5), Math.max(2, Math.max.apply(null, logs) + .5)] })
              .frame({ xlabel: 'time step (1 = earliest)', ylabel: 'log₁₀ ‖gradient‖ reaching this step' });
            P.clip(() => {
              P.line(norms.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: st.lstm ? T.green : T.blue, width: 2.6 });
              P.hline(logs[logs.length - 1] - 1, { color: T.faint, dash: [3, 3], label: '10× decay from the last step' });
            });
            let halfIdx = st.len;
            for (let i = norms.length - 1; i >= 0; i--) if (norms[i] < norms[norms.length - 1] / 10) { halfIdx = norms.length - i; break; }
            out({
              g1: norms[0].toExponential(2),
              half: halfIdx + ' steps',
              verdict: logs[0] < -8 ? 'vanished — no long-range learning' : logs[0] > 6 ? 'exploded — clip the gradient' : 'usable'
            });
          }
        });
        Viz.note(host, 'Turn on the additive path: the same recurrence now carries gradient across 40 steps. That single change — a near-identity route through time — is the LSTM’s entire contribution, and it is the same mechanism as a residual connection.');
      },

      lstm: function (host) {
        const st = Viz.controls(host, [
          { k: 'seq', label: 'input pattern', type: 'buttons', value: 'pulse', options: [{ v: 'pulse', t: 'one pulse, then silence' }, { v: 'noise', t: 'noisy stream' }, { v: 'switch', t: 'two pulses' }] },
          { k: 'forget', label: 'forget-gate bias', min: -3, max: 4, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'input', label: 'input-gate bias', min: -4, max: 3, step: .1, value: 0, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'retain', label: 'memory retained at t=30', cls: 'key' }, { k: 'f', label: 'typical forget gate' }, { k: 'i', label: 'typical input gate' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(3), Tn = 40;
            const xs = [];
            for (let t = 0; t < Tn; t++) {
              if (st.seq === 'pulse') xs.push(t === 3 ? 1 : 0);
              else if (st.seq === 'switch') xs.push(t === 3 ? 1 : (t === 20 ? -1 : 0));
              else xs.push(R.normal(0, .35) + (t === 3 ? 1 : 0));
            }
            let c = 0, hh = 0;
            const cs = [], fs = [], is = [], os = [];
            xs.forEach(x => {
              const f = Num.sigmoid(st.forget + .5 * hh);
              const i = Num.sigmoid(st.input + 2.2 * x);
              const o = Num.sigmoid(.5 + hh);
              const g = Math.tanh(2.4 * x);
              c = f * c + i * g;
              hh = o * Math.tanh(c);
              cs.push(c); fs.push(f); is.push(i); os.push(o);
            });
            const P = Viz.plot(ctx, w, h, { xd: [0, Tn - 1], yd: [-1.2, 1.2] })
              .frame({ xlabel: 'time step', ylabel: 'value' });
            P.clip(() => {
              xs.forEach((v, t) => P.line([[t, 0], [t, v]], { color: T.faint, width: 3, alpha: .7 }));
              P.line(cs.map((v, t) => [t, v]), { color: T.blue, width: 2.8 });
              P.line(fs.map((v, t) => [t, v]), { color: T.green, width: 1.6, dash: [4, 3] });
              P.line(is.map((v, t) => [t, v]), { color: T.amber, width: 1.6, dash: [4, 3] });
              P.hline(0, { color: T.faint, dash: [2, 3], width: 1 });
            });
            out({
              retain: (Math.abs(cs[30]) / Math.max(1e-9, Math.abs(cs[4]))).toFixed(3),
              f: Num.mean(fs).toFixed(3), i: Num.mean(is).toFixed(3)
            });
          }
        });
        Viz.legend(host, [
          { c: Viz.theme().faint, t: 'input xₜ' }, { c: Viz.theme().blue, t: 'cell state cₜ (the memory)' },
          { c: Viz.theme().green, t: 'forget gate' }, { c: Viz.theme().amber, t: 'input gate' }
        ]);
        Viz.note(host, 'With a high forget-gate bias the cell holds the pulse for the whole sequence; drop the bias to −1 and the memory decays within a few steps. The gate is a learned decision about what to keep, and that is the concept transformers replace with "attend to everything and let the softmax decide".');
      }
    },
    quiz: [
      {
        q: 'The decisive reason transformers replaced RNNs is…',
        options: ['RNNs cannot model long dependencies at all', 'recurrence cannot be parallelised across time, so it cannot saturate a GPU on trillions of tokens', 'RNNs have more parameters', 'attention is more accurate on short sequences'],
        answer: 1,
        why: 'Gating largely fixed the gradient problem; the unfixable issue was the sequential computational shape.'
      },
      {
        q: 'The LSTM cell state helps because…',
        options: ['it is larger than the hidden state', 'its update is additive and gated, giving a near-identity gradient path across time', 'it uses ReLU', 'it removes the need for backpropagation through time'],
        answer: 1,
        why: '$c_t=f_t\\odot c_{t-1}+\\ldots$ — with the forget gate near 1, gradients flow across many steps, exactly like a residual connection.'
      }
    ],
    cards: [
      { q: 'RNN gradient problem', a: 'BPTT multiplies the same $W_h^\\mathsf{T}\\mathrm{diag}(\\phi\')$ per step — geometric vanishing or explosion in sequence length.' },
      { q: 'LSTM in one line', a: 'Additive gated cell state: $c_t=f_t\\odot c_{t-1}+i_t\\odot\\tilde c_t$ — a residual path through time.' },
      { q: 'Why RNNs lost', a: 'Step t needs step t−1, so training cannot be parallelised across time; attention can.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.9 */
  ML.section({
    id: 'embeddings', track: 'deep', num: '3.9',
    title: 'Embeddings and representation learning',
    lede: 'The idea that carries from word2vec to modern retrieval: put meaning in a geometry, then let dot products do the reasoning.',
    html: `
<h2><span class="sn">3.9.1</span> From one-hot to dense</h2>
<p>A vocabulary of 50,000 words as one-hot vectors is 50,000 dimensions where every pair is equidistant — no notion of similarity exists. An <b>embedding</b> is a learned dense vector per token, typically 256–4096 dimensions, in which similar items land near each other. The embedding matrix is just a lookup table $E \\in \\mathbb{R}^{V\\times d}$, and "looking up token $i$" is multiplying by a one-hot vector — which is why it is a linear layer in disguise.</p>

<h2><span class="sn">3.9.2</span> How the geometry gets there</h2>
${H.table(['Method', 'Objective', 'What it captures'], [
      ['<b>word2vec (skip-gram)</b>', 'Predict context words from a centre word, with negative sampling', 'Distributional similarity — "words in similar contexts get similar vectors"'],
      ['<b>GloVe</b>', 'Factorise the log co-occurrence matrix', 'The same signal, as an explicit matrix factorisation'],
      ['<b>Contextual (BERT-style)</b>', 'Masked language modelling', 'One vector <i>per occurrence</i>, so "bank" differs by sentence'],
      ['<b>Sentence embeddings</b>', 'Contrastive: pull matched pairs together, push mismatched apart', 'Whole-passage meaning — the objective behind retrieval models (§5.1)'],
      ['<b>CLIP</b>', 'Contrastive across modalities', 'A shared image–text space (§4.16)']
    ])}

${H.lab('embed', 'An embedding space you can interrogate', 'A small hand-built space with real geometry. Search by cosine similarity, run analogies with vector arithmetic, and watch what happens when you normalise — the same operations a vector database performs at a billion-vector scale.')}

<h2><span class="sn">3.9.3</span> The arithmetic, and its limits</h2>
<p>The famous $\\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen}$ works because the training objective makes certain differences approximately parallel across pairs. It is a real property and an over-sold one: it holds for frequent, well-attested relations and fails routinely elsewhere, and the standard evaluation excludes the query words themselves from the answer set — which flatters the result.</p>

<h2><span class="sn">3.9.4</span> What matters in practice</h2>
<ul>
<li><b>Cosine or dot product?</b> Cosine if you normalise (pure direction); dot product if magnitude carries meaning, such as term importance. Whichever the model was trained with is the one to use (§5.1).</li>
<li><b>Dimensionality</b> is a memory/quality trade: §5.1 sizes an index and shows why Matryoshka embeddings — trained so that a truncated prefix still works — let you pick the trade after training.</li>
<li><b>Tied embeddings</b>: sharing the input embedding with the output projection saves parameters and usually costs nothing (§4.5), which matters most at small scale where the vocabulary dominates the parameter count.</li>
<li><b>Bias is inherited.</b> Embeddings reproduce the associations in their training corpus, including harmful ones; that is a data property, and mitigating it in the geometry is a research problem rather than a config flag.</li>
</ul>

${H.probe([
      ['Why not one-hot?', 'No similarity structure and enormous dimension; an embedding puts related items close together in a small dense space.'],
      ['Cosine or dot product?', 'Match whatever the model was trained with; cosine for normalised direction-only similarity, dot product when magnitude means something.'],
      ['What does "contextual embedding" add?', 'A different vector per occurrence, so polysemy is resolved by the sentence rather than averaged away.']
    ])}`,
    labs: {
      embed: function (host) {
        // hand-built 2-D-ish semantic space, then lifted to 8-D with structure
        const words = {
          king: [0.9, 0.8, 0.9, 0.1], queen: [0.9, 0.8, 0.1, 0.9], man: [0.2, 0.3, 0.9, 0.1], woman: [0.2, 0.3, 0.1, 0.9],
          prince: [0.8, 0.6, 0.85, 0.15], princess: [0.8, 0.6, 0.15, 0.85], boy: [0.15, 0.15, 0.9, 0.1], girl: [0.15, 0.15, 0.1, 0.9],
          london: [0.1, 0.9, 0.5, 0.5], paris: [0.12, 0.92, 0.5, 0.5], england: [0.05, 0.75, 0.5, 0.5], france: [0.07, 0.77, 0.5, 0.5],
          bank_money: [0.6, 0.15, 0.5, 0.5], bank_river: [0.15, 0.2, 0.5, 0.5], loan: [0.62, 0.12, 0.5, 0.5], river: [0.13, 0.22, 0.5, 0.5],
          cat: [0.3, 0.05, 0.5, 0.45], dog: [0.32, 0.06, 0.55, 0.45], kitten: [0.28, 0.04, 0.5, 0.47], puppy: [0.3, 0.05, 0.55, 0.47]
        };
        const names = Object.keys(words);
        const st = Viz.controls(host, [
          { k: 'query', label: 'nearest neighbours of', type: 'select', value: 'king', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'a', label: 'analogy: a', type: 'select', value: 'king', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'b', label: '− b', type: 'select', value: 'man', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'c', label: '+ c', type: 'select', value: 'woman', options: names.map(n => ({ v: n, t: n.replace('_', ' · ') })) },
          { k: 'norm', label: 'use cosine (normalise)', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nn', label: 'nearest neighbour', cls: 'key' }, { k: 'sim', label: 'similarity' },
          { k: 'ana', label: 'analogy result', cls: 'good' }, { k: 'anasim', label: 'analogy similarity' }
        ]);
        function sim(u, v) {
          const d = Num.dot(u, v);
          return st.norm ? d / ((Num.norm(u) * Num.norm(v)) || 1) : d;
        }
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            // project to 2-D with PCA for display
            const mat = names.map(n => words[n]);
            const pc = Num.pca(mat);
            const proj = mat.map(v => {
              const c = v.map((x, i) => x - pc.mean[i]);
              return [Num.dot(c, pc.vectors[0]), Num.dot(c, pc.vectors[1])];
            });
            const xs = proj.map(p => p[0]), ys = proj.map(p => p[1]);
            const pad = .25;
            const P = Viz.plot(ctx, w, h, {
              xd: [Math.min.apply(null, xs) - pad, Math.max.apply(null, xs) + pad],
              yd: [Math.min.apply(null, ys) - pad, Math.max.apply(null, ys) + pad]
            }).frame({ xlabel: 'PC1', ylabel: 'PC2' });
            const qi = names.indexOf(st.query);
            const sims = names.map((n, i) => ({ n: n, s: sim(words[st.query], words[n]), i: i })).filter(x => x.n !== st.query).sort((a, b) => b.s - a.s);
            // analogy
            const target = words[st.a].map((v, i) => v - words[st.b][i] + words[st.c][i]);
            const anas = names.map(n => ({ n: n, s: sim(target, words[n]) }))
              .filter(x => [st.a, st.b, st.c].indexOf(x.n) < 0).sort((a, b) => b.s - a.s);
            P.clip(() => {
              proj.forEach((p, i) => {
                const isQ = i === qi;
                const isTop = sims.slice(0, 3).some(s => s.i === i);
                P.dots([p], { r: isQ ? 6 : isTop ? 4.6 : 3, color: isQ ? T.red : isTop ? T.blue : T.faint, stroke: true });
                P.text(p[0], p[1], ' ' + names[i].replace('_', '·'), { color: isQ ? T.red : isTop ? T.text : T.faint, font: (isQ ? 'bold ' : '') + '10px ui-sans-serif' });
              });
              // analogy arrows
              const ia = names.indexOf(st.a), ib = names.indexOf(st.b), ic = names.indexOf(st.c), ir = names.indexOf(anas[0].n);
              P.arrow(proj[ib][0], proj[ib][1], proj[ia][0], proj[ia][1], { color: T.green, width: 1.8 });
              if (ir >= 0) P.arrow(proj[ic][0], proj[ic][1], proj[ir][0], proj[ir][1], { color: T.amber, width: 1.8 });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText(st.a + ' − ' + st.b + ' + ' + st.c + '  ≈  ' + anas[0].n + '  (' + anas[0].s.toFixed(3) + ')', 46, 8);
            ctx.fillStyle = T.faint;
            ctx.fillText('top matches: ' + sims.slice(0, 3).map(s => s.n + ' ' + s.s.toFixed(2)).join(' · '), 46, 24);
            out({
              nn: sims[0].n.replace('_', ' · '), sim: sims[0].s.toFixed(3),
              ana: anas[0].n.replace('_', ' · '), anasim: anas[0].s.toFixed(3)
            });
          }
        });
        Viz.note(host, 'Note "bank·money" sits next to "loan" while "bank·river" sits next to "river" — polysemy resolved by giving each sense its own vector, which is exactly what a contextual model does automatically. Also note the analogy evaluation excludes the three query words: without that exclusion, the nearest vector to king − man + woman is usually king itself.');
      }
    },
    quiz: [
      {
        q: 'Why are dense embeddings preferred to one-hot vectors?',
        options: ['They are easier to store', 'One-hot vectors make every pair equidistant, so no similarity structure exists', 'They avoid the need for a vocabulary', 'They are always more accurate'],
        answer: 1,
        why: 'Similarity has to live in the geometry; one-hot has none, and the dimension is the vocabulary size.'
      },
      {
        q: 'You switch an embedding model but keep the same vector index. What breaks?',
        options: ['Nothing', 'Everything — the geometry is different, so the whole corpus must be re-embedded and re-indexed', 'Only the reranker', 'Only long documents'],
        answer: 1,
        why: 'Vectors from different models are not comparable. §5.1 treats changing embedder as a migration, not a config flag.'
      }
    ],
    cards: [
      { q: 'Embedding matrix', a: '$E\\in\\mathbb{R}^{V\\times d}$ — a lookup table, equivalently a linear layer applied to a one-hot vector.' },
      { q: 'Cosine vs dot product', a: 'Cosine for normalised direction-only similarity; dot product when magnitude carries meaning. Match the training objective.' },
      { q: 'Contextual embeddings', a: 'One vector per occurrence rather than per type, so polysemy is resolved by context.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.10 */
  ML.section({
    id: 'part3-recall', track: 'deep', num: '3.15',
    title: 'Rapid recall — Part 3',
    lede: 'The neural-network essentials, in the form you would be asked for them.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', 'Composed linear layers collapse; the nonlinearity is what makes depth mean anything.', '<a href="#/nn-fundamentals">3.1</a>'],
      ['2', 'Backprop = chain rule right-to-left with cached activations; full gradient ≈ one forward pass.', '<a href="#/backprop">3.2</a>'],
      ['3', 'Worked backprop: L = 0.125 → 0.095 after one step at η = 0.1.', '<a href="#/backprop">3.2</a>'],
      ['4', 'Sigmoid derivative ≤ 0.25 ⇒ vanishing gradients; ReLU passes 1.', '<a href="#/activations">3.3</a>'],
      ['5', 'Residuals add a path of derivative exactly 1 — why 100-layer stacks train.', '<a href="#/activations">3.3</a>'],
      ['6', 'He init 2/n_in for ReLU; Xavier for symmetric activations; never all-zero.', '<a href="#/initialisation">3.4</a>'],
      ['7', 'AdamW decouples weight decay from the adaptive denominator — the modern default.', '<a href="#/optimisers">3.5</a>'],
      ['8', 'Warmup then cosine, or WSD to allow branching mid-run.', '<a href="#/optimisers">3.5</a>'],
      ['9', 'LayerNorm/RMSNorm for transformers; BatchNorm depends on the batch.', '<a href="#/normalisation">3.6</a>'],
      ['10', 'Dropout = training an ensemble of thinned sub-networks.', '<a href="#/normalisation">3.6</a>'],
      ['11', 'Conv output $\\lfloor(H+2p-k)/s\\rfloor+1$; two 3×3 beat one 5×5.', '<a href="#/cnn">3.7</a>'],
      ['12', 'RNNs lost on parallelism, not on modelling; LSTM’s cell state is a residual through time.', '<a href="#/rnn">3.8</a>'],
      ['13', 'Embeddings put meaning in a geometry; match the trained similarity metric.', '<a href="#/embeddings">3.9</a>']
    ])}

${H.lab('drill3', 'Part 3 drill', 'Thirteen prompts, shuffled.')}`,
    labs: {
      drill3: function (host) {
        const cards = [
          ['Why is a nonlinearity necessary?', 'Composed linear maps are linear; without one, depth adds no capacity.'],
          ['Backprop in one sentence.', 'Chain rule evaluated right-to-left with cached forward activations.'],
          ['Why reverse-mode rather than forward-mode?', 'One scalar output, many parameters — reverse costs one pass per output.'],
          ['Max derivative of sigmoid, and the consequence.', '0.25; ten layers scale the gradient by at most $10^{-6}$.'],
          ['What do residual connections do to the gradient?', 'Add a path with local derivative exactly 1, so it reaches layer 1 unattenuated.'],
          ['He vs Xavier initialisation.', '$2/n_{in}$ for ReLU (half the inputs are zeroed) vs $2/(n_{in}+n_{out})$ for symmetric activations.'],
          ['Adam vs AdamW.', 'AdamW applies weight decay outside the adaptive denominator, so decay actually decays.'],
          ['Why warmup?', 'Adam’s second-moment estimate is unreliable early; large first steps can be unrecoverable.'],
          ['WSD schedule and its advantage.', 'Warmup, stable, sharp decay — no need to fix total steps up front; branchable.'],
          ['LayerNorm vs BatchNorm.', 'Per-example features vs per-channel over the batch; transformers use LayerNorm/RMSNorm.'],
          ['Why does dropout regularise?', 'It trains an ensemble of thinned sub-networks and prevents co-adaptation.'],
          ['Convolution output shape.', '$\\lfloor(H+2p-k)/s\\rfloor+1$.'],
          ['Receptive field of L stacked 3×3 convs.', '$2L+1$ at stride 1.'],
          ['Why did transformers replace RNNs?', 'Recurrence cannot parallelise across time; attention gives O(1) path length and one big matmul.'],
          ['What is the LSTM cell state for?', 'An additive, gated memory — a residual path through time.'],
          ['Cosine or dot product for embeddings?', 'Whichever the model was trained with; cosine after normalisation.']
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
        host.appendChild(nav);
        draw();
      }
    },
    quiz: [
      {
        q: 'Which two ideas from Part 3 reappear unchanged inside every transformer block?',
        options: ['Dropout and BatchNorm', 'Residual connections and normalisation', 'Convolution and pooling', 'Momentum and warmup'],
        answer: 1,
        why: 'A transformer block is attention and an FFN wrapped in exactly those two stabilisers — §4.5 draws it with shapes.'
      }
    ]
  });
})();
