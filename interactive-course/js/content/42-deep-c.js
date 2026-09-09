/* ============================================================
   PART 3 — Deep learning, continued: autodiff (3.11), reading a
   training curve (3.12), compression (3.13), robustness (3.14).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 3.11 */
  ML.section({
    id: 'autodiff', track: 'deep', num: '3.11', level: 2,
    title: 'Automatic differentiation and the computation graph',
    lede: 'Backpropagation (§3.2) is one instance of a general algorithm. Understanding the general version explains why activations dominate your memory bill, what gradient checkpointing actually trades, and why <code>.detach()</code> and <code>torch.no_grad()</code> are different things.',
    prereq: ['backprop', 'matrix-calculus'],
    related: ['backprop', 'distributed', 'optimisers'],
    html: `
${H.tldr([
      'Autodiff is neither symbolic differentiation nor finite differences. It evaluates the chain rule numerically on a graph of primitive operations, and it is <b>exact</b> to floating-point precision.',
      '<b>Forward mode</b> costs one sweep per input; <b>reverse mode</b> one per output. A loss has one output and $10^9$ inputs, so reverse mode wins by a factor of a billion — and pays for it with memory.',
      'Reverse mode must keep every intermediate activation until its gradient is used. That is why activation memory, not parameter memory, is usually what makes a batch not fit — and why checkpointing (recompute instead of store) is the standard escape.'
    ])}

<h2><span class="sn">3.11.1</span> Three ways to get a derivative, only one of which is used</h2>
${H.table(['Method', 'How', 'Error', 'Cost', 'Verdict'], [
      ['Symbolic', 'manipulate the expression algebraically', 'exact', 'expression swell — the formula can grow exponentially', 'fine for a paper, useless for a network'],
      ['Numerical', 'finite differences', '$O(\\varepsilon^2)$ at best, and it never vanishes', '$O(d)$ function evaluations', 'only for checking (§0.7)'],
      ['<b>Automatic</b>', 'chain rule applied numerically, op by op', '<b>exact</b> up to floating point', '$O(1)$ passes for reverse mode', 'what every framework does']
    ])}
${H.key('Autodiff is often mistaken for one of the other two. It is exact like symbolic differentiation and cheap like numerical differentiation, because it never builds the symbolic expression — it just evaluates the derivative at the point it is standing on.')}

<h2><span class="sn">3.11.2</span> The graph, forwards and backwards</h2>
<p>Every operation you write becomes a node. Each node knows two things: how to compute its output from its inputs (forward), and how to turn "the gradient of the loss with respect to my output" into "the gradient with respect to each of my inputs" (backward, the <b>vector-Jacobian product</b>). Nothing else is needed.</p>
${H.lab('graph', 'A computation graph, differentiated in front of you', 'A real reverse-mode engine, ten lines of it, running on the expression shown. Step forward to fill in the values; step backward to watch adjoints accumulate from the loss end. Every number here is computed, not annotated.')}

${H.deriv('the vector-Jacobian product, and why it is the right primitive', [
      ['$\\bar{v} := \\dfrac{\\partial \\mathcal{L}}{\\partial v}$', 'Define the <i>adjoint</i> of every intermediate $v$: how much the loss moves when $v$ moves. It has the same shape as $v$.'],
      ['$\\bar{u} = \\bar{v}^\\top \\dfrac{\\partial v}{\\partial u}$', 'The chain rule for a node $v = f(u)$. This is a row vector times a Jacobian — <b>a vector-Jacobian product</b>, written <code>vjp</code>.'],
      ['$\\bar{u} \\mathrel{+}= \\bar{v}^\\top J_f$', 'Accumulate with <code>+=</code>, because $u$ may feed several nodes and each contributes. Forgetting to accumulate is the classic hand-rolled-autodiff bug.'],
      ['never form $J_f$', 'For $v = Wu$ the Jacobian is $W$ and $\\bar u = W^\\top\\bar v$ — a matrix-vector product, never an explicit Jacobian. For an elementwise $\\sigma$, $\\bar u = \\bar v \\odot \\sigma\'(u)$. <b>Every primitive ships its vjp; none ships a Jacobian.</b>']
    ], 'This is why writing a custom layer means implementing <code>forward</code> and <code>backward</code> and nothing else: the framework only ever asks a node to do a vjp.')}

${H.vs('Forward mode', [
      'Carries a derivative alongside every value (dual numbers)',
      'One sweep gives $\\partial(\\text{all outputs})/\\partial(\\text{one input})$',
      'Cost: $O(n_{\\text{inputs}})$ sweeps',
      'No memory of the forward pass needed',
      'Right for: few inputs, many outputs — Jacobian-vector products, sensitivity analysis'
    ], 'Reverse mode', [
      'Records the graph, then walks it backwards',
      'One sweep gives $\\partial(\\text{one output})/\\partial(\\text{all inputs})$',
      'Cost: $O(n_{\\text{outputs}})$ sweeps ≈ 2× a forward pass',
      '<b>Must store every activation</b> until its gradient is consumed',
      'Right for: training anything with a scalar loss'
    ])}

<h2><span class="sn">3.11.3</span> Memory: the real cost, and checkpointing</h2>
<p>For a transformer, the memory during training splits roughly into four buckets:</p>
${H.table(['Bucket', 'Size (bf16 weights, Adam)', 'Scales with'], [
      ['Parameters', '2 bytes × $N$', 'model size'],
      ['Gradients', '2 bytes × $N$', 'model size'],
      ['Optimizer state (Adam m, v, fp32 master)', '~12 bytes × $N$', 'model size'],
      ['<b>Activations</b>', '$O(L \\cdot B \\cdot S \\cdot d)$', '<b>batch × sequence × depth</b>']
    ])}
${H.worked('why a 7B model needs ~112 GB to train, and 14 GB to serve', `
<p>7B parameters in bf16: weights 14 GB, gradients 14 GB, Adam state (fp32 momentum + variance + master weights) 84 GB → <b>112 GB before a single activation</b>. That is why a 7B model does not fine-tune on one 80 GB card without LoRA (§4.13), ZeRO sharding (§4.11) or 8-bit optimiser states.</p>
<p>Inference needs only the 14 GB of weights plus the KV cache (§4.7). <b>The 8× gap between training and serving memory is entirely optimiser state and gradients</b> — a number worth having ready, because it is the first thing asked when someone says "can we fine-tune this".</p>`)}

${H.lab('ckpt', 'Gradient checkpointing: the memory/compute trade', 'Store every activation, or store only every $k$-th and recompute the rest during the backward pass. The optimum is at $k=\\sqrt{L}$, which turns $O(L)$ memory into $O(\\sqrt{L})$ for about 30% extra compute.')}

${H.table(['Trick', 'Saves', 'Costs', 'Where'], [
      ['Gradient checkpointing', 'activation memory → $O(\\sqrt L)$', '~30% more compute', 'the standard first move when OOM'],
      ['Gradient accumulation', 'peak activation memory', 'nothing but wall-clock', 'simulating a large batch on small hardware'],
      ['Mixed precision (bf16)', 'half the activation and weight bytes', 'needs fp32 master weights', 'universal since 2020 (§1.15)'],
      ['<code>torch.no_grad()</code>', '<b>all</b> activation storage', 'no gradients at all', 'inference and evaluation'],
      ['<code>.detach()</code>', 'the history behind one tensor', 'that path gets no gradient', 'stop-gradient in BYOL, target networks in RL'],
      ['Fused / flash kernels', 'materialising the attention matrix', 'kernel complexity', 'FlashAttention (§4.7)']
    ])}
${H.pitfall('<code>no_grad()</code> and <code>detach()</code> are not interchangeable. <code>no_grad()</code> is a <i>context</i>: nothing inside it records history at all, so no memory is spent. <code>detach()</code> returns a <i>tensor</i> cut off from its history, but the graph that produced it still exists and is still holding memory if anything else references it. Using <code>detach()</code> in an evaluation loop and wondering why memory still climbs is a rite of passage.')}

${H.probe([
      ['Why does reverse-mode autodiff use so much memory?', 'The backward pass needs each node’s inputs to compute its vjp, so every activation must survive from the forward pass until its gradient is consumed. Memory scales with depth × batch × sequence.'],
      ['What exactly does gradient checkpointing trade?', 'Compute for memory: it stores activations only at checkpoint boundaries and recomputes the segments in between during the backward pass. $O(L)\\to O(\\sqrt L)$ memory for roughly one extra forward pass.'],
      ['When is forward mode the right choice?', 'When you want a Jacobian-vector product — directional derivatives, one-input sensitivity, some Hessian-vector product formulations. Rarely for training.'],
      ['A model trains at batch 8 and OOMs at 16, but the parameter count did not change. Why?', 'Activation memory scales with batch size; parameters do not. Checkpointing, accumulation or a shorter sequence are the fixes.']
    ], 'Saying autodiff is "just symbolic differentiation" or "just finite differences". It is neither, and the distinction is exactly what makes it work.')}`,
    labs: {
      graph: function (host) {
        /* a genuine ten-line reverse-mode engine */
        function V(val, children, op, back) {
          return { v: val, g: 0, children: children || [], op: op || 'leaf', back: back || function () {}, name: '' };
        }
        function mul(a, b) { const o = V(a.v * b.v, [a, b], '×'); o.back = () => { a.g += b.v * o.g; b.g += a.v * o.g; }; return o; }
        function add(a, b) { const o = V(a.v + b.v, [a, b], '+'); o.back = () => { a.g += o.g; b.g += o.g; }; return o; }
        function sin(a) { const o = V(Math.sin(a.v), [a], 'sin'); o.back = () => { a.g += Math.cos(a.v) * o.g; }; return o; }
        function sq(a) { const o = V(a.v * a.v, [a], '( )²'); o.back = () => { a.g += 2 * a.v * o.g; }; return o; }

        const st = Viz.controls(host, [
          { k: 'a', label: 'a', min: -3, max: 3, step: .1, value: 2, fmt: v => v.toFixed(1) },
          { k: 'b', label: 'b', min: -3, max: 3, step: .1, value: -1.5, fmt: v => v.toFixed(1) },
          { k: 'c', label: 'c', min: -3, max: 3, step: .1, value: 0.6, fmt: v => v.toFixed(1) }
        ], () => build());

        const stage = ML.el('div');
        host.appendChild(stage);
        let phase = 0;

        function graph() {
          const a = V(st.a); a.name = 'a';
          const b = V(st.b); b.name = 'b';
          const c = V(st.c); c.name = 'c';
          const ab = mul(a, b); ab.name = 'u = a·b';
          const sc = sin(c); sc.name = 'v = sin(c)';
          const s = add(ab, sc); s.name = 'w = u + v';
          const L = sq(s); L.name = 'L = w²';
          return { a, b, c, ab, sc, s, L };
        }

        function backward(L) {
          const order = [], seen = new Set();
          (function visit(n) { if (seen.has(n)) return; seen.add(n); n.children.forEach(visit); order.push(n); })(L);
          order.forEach(n => { n.g = 0; });
          L.g = 1;
          for (let i = order.length - 1; i >= 0; i--) order[i].back();
          return order;
        }

        let S = null;
        function build() {
          stage.innerHTML = '';
          const g = graph();
          backward(g.L);
          const nodes = [
            { id: 'a', x: .02, y: .05, w: .17, h: .2, label: 'a', ref: g.a },
            { id: 'b', x: .02, y: .40, w: .17, h: .2, label: 'b', ref: g.b },
            { id: 'c', x: .02, y: .75, w: .17, h: .2, label: 'c', ref: g.c },
            { id: 'u', x: .28, y: .22, w: .19, h: .2, label: 'u = a·b', ref: g.ab },
            { id: 'v', x: .28, y: .75, w: .19, h: .2, label: 'v = sin c', ref: g.sc },
            { id: 'w', x: .55, y: .45, w: .18, h: .2, label: 'w = u+v', ref: g.s },
            { id: 'L', x: .81, y: .45, w: .17, h: .2, label: 'L = w²', ref: g.L }
          ];
          const edges = [
            { from: 'a', to: 'u' }, { from: 'b', to: 'u' }, { from: 'c', to: 'v' },
            { from: 'u', to: 'w' }, { from: 'v', to: 'w' }, { from: 'w', to: 'L' }
          ];
          const order = ['a', 'b', 'c', 'u', 'v', 'w', 'L'];

          S = Viz.surface(stage, {
            height: 250,
            draw: function (ctx, w2, h, T) {
              const fwd = phase <= 6;
              const step = phase;
              const shown = fwd ? order.slice(0, step + 1) : order.slice();
              const gradShown = fwd ? [] : order.slice().reverse().slice(0, step - 6);
              const nd = nodes.map(n => {
                const on = shown.indexOf(n.id) >= 0;
                const gon = gradShown.indexOf(n.id) >= 0;
                return {
                  id: n.id, x: n.x, y: n.y, w: n.w, h: n.h,
                  label: n.label,
                  sub: on ? (gon ? 'v=' + n.ref.v.toFixed(2) + '  ḡ=' + n.ref.g.toFixed(2) : 'v=' + n.ref.v.toFixed(2)) : '—',
                  fill: gon ? 'color-mix(in oklab, ' + T.c2 + ' 22%, ' + T.panel + ')'
                    : on ? 'color-mix(in oklab, ' + T.c1 + ' 15%, ' + T.panel + ')' : T.panel,
                  accent: gon ? T.c2 : on ? T.c1 : T.line
                };
              });
              const ed = edges.map(e => ({
                from: e.from, to: e.to,
                color: (!fwd && gradShown.indexOf(e.to) >= 0) ? T.c2 : (shown.indexOf(e.to) >= 0 ? T.c1 : T.line),
                width: 1.8
              }));
              Viz.flow(ctx, w2, h, nd, ed, { pad: 12 });
              ctx.fillStyle = T.muted; ctx.font = '11.5px ui-sans-serif'; ctx.textAlign = 'left';
              ctx.fillText(fwd ? '→ forward pass: computing values' : '← backward pass: accumulating adjoints ḡ = ∂L/∂node', 12, 14);
            }
          });
          const out = Viz.readout(stage, [
            { k: 'L', label: 'L', cls: 'key' },
            { k: 'ga', label: '∂L/∂a' }, { k: 'gb', label: '∂L/∂b' }, { k: 'gc', label: '∂L/∂c' },
            { k: 'chk', label: 'numerical check' }
          ]);
          const f = (A, B, C) => Math.pow(A * B + Math.sin(C), 2);
          const h = 1e-5;
          out({
            L: g.L.v.toFixed(4),
            ga: g.a.g.toFixed(4), gb: g.b.g.toFixed(4), gc: g.c.g.toFixed(4),
            chk: ((f(st.a + h, st.b, st.c) - f(st.a - h, st.b, st.c)) / (2 * h)).toFixed(4) + ' (∂a)'
          });
          Viz.player(stage, {
            frames: 13, fps: 1.4, repeat: true,
            label: i => i <= 6 ? 'forward ' + (i + 1) + '/7' : 'backward ' + (i - 6) + '/6',
            onFrame: i => { phase = i; if (S) S.redraw(); }
          });
        }
        build();
        Viz.note(host, 'The <b>numerical check</b> readout recomputes ∂L/∂a by central differences and matches the autodiff answer to four decimals — the same check as §0.7, now verifying an engine rather than a formula. Note that the backward pass touches each node exactly once, in reverse topological order, and that node <code>w</code> pushes its adjoint into <i>both</i> <code>u</code> and <code>v</code> with <code>+=</code>. That accumulation is the whole reason a node can be reused.');
      },

      ckpt: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers L', min: 8, max: 128, step: 4, value: 48, fmt: v => v },
          { k: 'perLayer', label: 'activation memory per layer (MB)', min: 20, max: 800, step: 20, value: 220, fmt: v => v + ' MB' },
          { k: 'k', label: 'checkpoint every k layers', min: 1, max: 24, step: 1, value: 7, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'none', label: 'no checkpointing', cls: 'bad' },
          { k: 'now', label: 'at your k', cls: 'key' },
          { k: 'best', label: 'optimum k = √L', cls: 'good' },
          { k: 'compute', label: 'extra compute' },
          { k: 'save', label: 'memory saved' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const L = st.L, M = st.perLayer;
            // store L/k checkpoints, plus one segment of k activations live at a time
            const mem = k => (L / k + k) * M;
            const kOpt = Math.max(1, Math.round(Math.sqrt(L)));
            const P = Viz.plot(ctx, w, h, {
              xd: [1, Math.min(24, L)], yd: [0, mem(1) * 1.05],
              pad: { l: 62, r: 14, t: 16, b: 40 }
            }).frame({ xlabel: 'checkpoint interval k', ylabel: 'peak activation memory (MB)' });
            P.clip(() => {
              P.area(Num.linspace(1, Math.min(24, L), 90).map(k => [k, mem(k)]), { color: T.c1, alpha: .12 });
              P.fn(mem, { color: T.c1, width: 2.6, n: 160, from: 1, to: Math.min(24, L) });
              P.hline(mem(1), { color: T.c2, dash: [5, 4], label: 'store everything' });
              P.vline(kOpt, { color: T.c3, dash: [4, 4], label: 'k = √L' });
              P.dots([[st.k, mem(st.k)]], { r: 5.5, color: T.c4, stroke: true, strokeWidth: 2 });
            });
            out({
              none: (mem(1) / 1024).toFixed(1) + ' GB',
              now: (mem(st.k) / 1024).toFixed(2) + ' GB',
              best: 'k=' + kOpt + ' → ' + (mem(kOpt) / 1024).toFixed(2) + ' GB',
              compute: '+' + ((1 - 1 / st.k) * 33).toFixed(0) + '% (≈1 extra fwd)',
              save: ((1 - mem(st.k) / mem(1)) * 100).toFixed(1) + '%'
            });
          }
        });
        Viz.note(host, 'The curve is $\\left(\\frac{L}{k}+k\\right)M$ — checkpoints stored plus one live segment — and it is minimised at $k=\\sqrt L$ by AM–GM. At 48 layers that is $k=7$: <b>10.3 GB becomes 3.0 GB, a 71% saving, for roughly one extra forward pass.</b> This single trick is what lets a long-context model train on the hardware you have.');
      }
    },
    quiz: [
      {
        q: 'Reverse-mode autodiff is memory-hungry because…',
        options: ['it stores the symbolic expression', 'every activation must survive until its gradient is consumed', 'it uses float64', 'it recomputes the forward pass'],
        answer: 1,
        why: 'The vjp at each node needs that node’s inputs. Checkpointing trades this memory back for recomputation.'
      },
      {
        q: 'The primitive every autodiff node implements is…',
        options: ['the full Jacobian', 'a vector-Jacobian product', 'a Hessian-vector product', 'a finite difference'],
        answer: 1,
        why: 'For $v = Wu$ the vjp is $W^\\top\\bar v$ — a matrix-vector product. The Jacobian itself is never formed.'
      },
      {
        q: 'Gradient checkpointing with $k=\\sqrt{L}$ turns activation memory from $O(L)$ into…',
        options: ['$O(1)$', '$O(\\sqrt L)$', '$O(\\log L)$', '$O(L/2)$'],
        answer: 1,
        why: '$(L/k + k)$ is minimised at $k=\\sqrt L$, giving $2\\sqrt L$ segments of storage for about one extra forward pass.'
      },
      {
        q: 'Roughly how much memory does Adam add on top of bf16 weights?',
        options: ['none', 'about 2 bytes per parameter', 'about 12 bytes per parameter', 'about 100 bytes per parameter'],
        answer: 2,
        why: 'fp32 momentum + fp32 variance + fp32 master weights ≈ 12 bytes/param, which is why 7B needs ~112 GB to train and 14 GB to serve.'
      }
    ],
    cards: [
      { q: 'Forward vs reverse mode', a: 'Forward: one sweep per input. Reverse: one per output. A scalar loss with many parameters makes reverse mode a billion times cheaper.' },
      { q: 'What a node must implement', a: 'Forward, and a vector-Jacobian product. Never an explicit Jacobian.' },
      { q: 'Training memory for a 7B model', a: '14 GB weights + 14 GB grads + ~84 GB Adam ≈ 112 GB, before activations. Serving needs 14 GB + KV cache.' },
      { q: 'Checkpointing trade', a: '$O(L)\\to O(\\sqrt L)$ activation memory for roughly one extra forward pass (~30% compute).' },
      { q: 'no_grad vs detach', a: '<code>no_grad</code> stops history being recorded at all; <code>detach</code> cuts one tensor loose but the graph behind it may still be held.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.12 */
  ML.section({
    id: 'training-dynamics', track: 'deep', num: '3.12', level: 2,
    title: 'Reading a training curve',
    lede: 'The most valuable diagnostic skill in deep learning, and the one least often taught: looking at a loss curve and naming the cause. Every shape here has one, and interviewers show these plots because a candidate who can read them has actually trained something.',
    prereq: ['optimisers'],
    related: ['optimisers', 'normalisation', 'distributed'],
    html: `
${H.tldr([
      'Before anything else: <b>overfit a batch of ten examples to zero loss</b>. If you cannot, the bug is in the data, the loss or the label alignment — not in the model.',
      'Flat loss = learning rate too low, dead activations, or no gradient path. Exploding loss = learning rate too high, no clipping, or fp16 overflow. Sawtooth = the learning-rate schedule. Step change = a data problem.',
      'Train and validation curves separating is <i>not</i> automatically bad. What matters is whether validation is still improving.'
    ])}

<h2><span class="sn">3.12.1</span> The shapes, and what causes them</h2>
${H.table(['Shape', 'Most likely cause', 'Confirm by', 'Fix'], [
      ['<b>Flat from step 0</b>', 'LR too low, or gradients are not reaching the parameters', 'print grad norms per layer; check <code>requires_grad</code>', 'raise LR; check the graph is connected'],
      ['<b>Loss = NaN at step k</b>', 'LR too high, fp16 overflow, or log(0)', 'find the <i>first</i> NaN with per-op hooks', 'clip gradients; bf16; stabilise the loss (§1.15)'],
      ['<b>Rises then plateaus high</b>', 'LR above the stability threshold', 'LR range test', 'reduce LR ~10×; add warmup'],
      ['<b>Falls, then spikes, then recovers</b>', 'a bad batch, or a loss-scaling event', 'log the batch index of the spike', 'gradient clipping; skip-on-spike; data cleaning'],
      ['<b>Sawtooth</b>', 'cyclic or restarting LR schedule', 'plot the LR alongside', 'nothing — this is intended'],
      ['<b>Staircase down</b>', 'step LR decay', 'plot the LR', 'nothing'],
      ['<b>Train ↓, val ↑</b>', 'overfitting', 'gap widens with epochs', 'regularise, augment, early-stop'],
      ['<b>Val below train</b>', 'dropout/augmentation active in training only, or an easier val split', 'evaluate train in eval mode', 'usually benign — verify the split'],
      ['<b>Sudden step change</b>', 'a data pipeline event: shard boundary, corrupt file, distribution change', 'plot loss against data index, not step', 'audit the shard'],
      ['<b>Loss falls, metric flat</b>', 'optimising the wrong thing', 'compare loss and metric per epoch', 'change the loss or the threshold (§2.13)']
    ])}

${H.lab('diagnose', 'Diagnose the run', 'Ten real pathologies, generated with their actual mechanism, shown one at a time. Name the cause before you reveal the answer. This is the exercise, in miniature, that interviewers run when they show you a plot.')}

<h2><span class="sn">3.12.2</span> The learning-rate range test</h2>
<p>Sweep the learning rate exponentially upward over a few hundred steps and plot loss against LR. You get a characteristic curve: flat, then descending, then a sharp rise. <b>Pick roughly one order of magnitude below the minimum</b> — the steepest-descent region, not the bottom.</p>
${H.lab('lrfind', 'The LR range test, on a network that really trains', 'A genuine MLP, genuinely trained at each learning rate for a short burst. The suggested value is the point of steepest descent, which is the standard heuristic.')}
${H.note('Why one order of magnitude below the minimum? Because at the minimum of this curve you are already close to the instability threshold, and the curve is measured over a short burst with a fixed initialisation. As training progresses the loss surface sharpens, and a rate that was fine at step 100 diverges at step 10,000. Warmup exists for the same reason.')}

<h2><span class="sn">3.12.3</span> The instruments to log, in order of usefulness</h2>
${H.steps([
      '<b>Loss, train and validation, on the same axes.</b> Log-scale the y-axis. Most of the information is in the first 5% of steps.',
      '<b>Gradient norm, globally and per layer.</b> A layer whose grad norm is 1e-9 is not learning; one at 1e3 is about to break everything. The ratio between the largest and smallest layer norms is the single best early-warning signal.',
      '<b>Learning rate.</b> On the same plot as the loss. Half of all "mysterious" curve features are the schedule.',
      '<b>Update-to-weight ratio</b> $\\|\\Delta w\\|/\\|w\\|$ per layer. Healthy is around $10^{-3}$. Much smaller means nothing is moving; much larger means the layer is being rewritten each step.',
      '<b>Activation statistics.</b> Mean and standard deviation per layer, plus the fraction of dead ReLUs. Drift here precedes loss problems by thousands of steps.',
      '<b>Throughput and the loss-vs-tokens curve.</b> Loss against <i>tokens seen</i>, not steps — it is the only comparison that survives a change of batch size.'
    ])}
${H.key('Plot loss against tokens (or examples) seen, never against steps, whenever the batch size might change. Two runs with different batch sizes are not comparable on a step axis, and half of all "this optimiser is better" claims dissolve when replotted.')}

${H.pitfall('The loss spike that "recovers" often has not. A large spike can knock the optimiser into a worse basin, and Adam’s second-moment estimate takes thousands of steps to forget an enormous gradient. Large pretraining runs keep periodic checkpoints precisely so they can rewind past a spike and skip the offending data — this is standard operating procedure at frontier labs, not an exotic measure (§4.11).')}

${H.probe([
      ['Your loss is flat at exactly $\\ln(C)$ for a $C$-class problem. Diagnose it.', 'The model is outputting a uniform distribution — it has learned the prior and nothing else. Either the LR is far too low, the labels are shuffled relative to the inputs, or the head is disconnected. Check the label alignment first; it is the most common cause.'],
      ['Validation loss is <i>below</i> training loss. Is something wrong?', 'Usually not. Training loss is measured with dropout and augmentation on, validation without. Confirm by evaluating the training set in eval mode; if the gap persists, your split is not random.'],
      ['What is the first thing you do on a new model?', 'Overfit ten examples to near-zero loss with regularization off. It takes two minutes and eliminates the entire class of data and plumbing bugs.'],
      ['Loss falls but your metric does not move. What is happening?', 'The loss and the metric are not aligned: often a threshold problem on an imbalanced task, or a proxy loss that does not track the objective. Look at the metric’s decomposition, not the loss.']
    ])}`,
    labs: {
      diagnose: function (host) {
        const el = ML.el;
        const CASES = [
          { name: 'flat', gen: (t) => 2.3 + 0.0002 * Math.sin(t / 9), cause: 'Learning rate far too low (or the head is disconnected)',
            why: 'The loss sits at ln(10) ≈ 2.30 — a uniform distribution over 10 classes. The model has learned the prior and nothing else. Check that gradients are non-zero before you touch the learning rate.' },
          { name: 'diverge', gen: (t) => t < 40 ? 2.3 - 0.01 * t : Math.min(9, 1.9 + 0.09 * (t - 40)), cause: 'Learning rate above the stability threshold',
            why: 'A brief improvement while the parameters are still small, then monotone divergence. Classic too-high LR. Warmup plus a 10× reduction usually fixes it.' },
          { name: 'nan', gen: (t) => t < 62 ? 2.3 * Math.exp(-t / 45) + .3 : NaN, cause: 'NaN — fp16 overflow or log(0)',
            why: 'Everything is healthy until it is not. Find the *first* NaN with per-tensor hooks; the usual culprits are an unclipped gradient, an fp16 overflow, or an unstabilised log/exp (§1.15).' },
          { name: 'spike', gen: (t) => { const b = 2.3 * Math.exp(-t / 40) + .25; return (t > 55 && t < 62) ? b + 2.6 * Math.exp(-Math.abs(t - 58)) : b; }, cause: 'A bad batch — one pathological example or a corrupt shard',
            why: 'A single sharp excursion that recovers. Log the batch index at the spike. Gradient clipping keeps it from being fatal; the data still needs cleaning, and Adam’s second moment takes a long time to forget the event.' },
          { name: 'saw', gen: (t) => 2.3 * Math.exp(-t / 55) + .28 + .16 * Math.abs(((t % 25) / 25) - .5), cause: 'A cyclic (or restarting) learning-rate schedule',
            why: 'Nothing is wrong. Plot the learning rate on the same axes and the mystery evaporates. Half of all confusing curve features are the schedule.' },
          { name: 'overfit', gen: (t) => 2.3 * Math.exp(-t / 22) + .04, gen2: (t) => 2.3 * Math.exp(-t / 30) + .35 + Math.max(0, (t - 45) * .010), cause: 'Overfitting',
            why: 'Training keeps falling; validation bottoms out around step 45 and then climbs. Early stopping at the minimum, more regularization, more data or more augmentation.' },
          { name: 'step', gen: (t) => (t < 55 ? 2.3 * Math.exp(-t / 30) + .3 : 2.3 * Math.exp(-t / 30) + .95), cause: 'A data pipeline event — shard boundary or distribution change',
            why: 'A discontinuity, not a slope change. Optimisation problems bend the curve; data problems step it. Plot loss against data index rather than step and the boundary will be obvious.' },
          { name: 'plateau', gen: (t) => t < 30 ? 2.3 - .02 * t : (t < 75 ? 1.7 : 1.7 - .02 * (t - 75)), cause: 'A saddle or a dead-ReLU plateau; the schedule later escapes it',
            why: 'Long flat region then renewed progress. Check the fraction of dead ReLUs and the per-layer gradient norms during the plateau. Warmup, better initialisation (§3.4) or a nonzero-gradient activation prevent it.' },
          { name: 'noisy', gen: (t) => 2.3 * Math.exp(-t / 40) + .3 + .38 * Math.sin(t * 2.1) * Math.exp(-t / 120), cause: 'Batch size too small (or LR too high for this batch size)',
            why: 'The trend is fine; the variance is enormous. Gradient noise scales as $1/\\sqrt{B}$. Increase the batch, accumulate gradients, or lower the LR — the linear scaling rule ties the two together.' },
          { name: 'gap', gen: (t) => 2.3 * Math.exp(-t / 32) + .18, gen2: (t) => 2.3 * Math.exp(-t / 34) + .21, cause: 'Healthy',
            why: 'Both curves fall together, validation slightly above training, still improving at the end. Keep training — and note that a small persistent gap is normal, not a problem to be fixed.' }
        ];

        let idx = 0, revealed = false, score = 0, attempted = 0;
        const wrap = el('div');
        host.appendChild(wrap);

        function paint() {
          wrap.innerHTML = '';
          const c = CASES[idx];
          const cvHost = el('div');
          wrap.appendChild(cvHost);
          Viz.surface(cvHost, {
            height: 230,
            draw: function (ctx, w, h, T) {
              const ts = Num.linspace(0, 100, 101);
              const tr = ts.map(t => [t, c.gen(t)]).filter(p => isFinite(p[1]));
              const va = c.gen2 ? ts.map(t => [t, c.gen2(t)]).filter(p => isFinite(p[1])) : null;
              const all = tr.concat(va || []).map(p => p[1]);
              const P = Viz.plot(ctx, w, h, {
                xd: [0, 100], yd: [0, Math.min(9.5, Math.max.apply(null, all) * 1.15)],
                pad: { l: 48, r: 14, t: 16, b: 38 }
              }).frame({ xlabel: 'step', ylabel: 'loss' });
              P.clip(() => {
                if (va) P.line(va, { color: T.c2, width: 2.2 });
                P.line(tr, { color: T.c1, width: 2.4 });
                const brk = ts.find(t => !isFinite(c.gen(t)));
                if (brk != null) {
                  P.vline(brk, { color: T.c2, width: 2, dash: [3, 3], label: 'NaN' });
                }
              });
              if (va) {
                ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif'; ctx.textAlign = 'right';
                ctx.fillText('— train', w - 20, 22); ctx.fillStyle = T.c2; ctx.fillText('— validation', w - 20, 36);
              }
            }
          });

          const opts = ML.shuffle(CASES.map(x => x.cause)).slice(0, 3);
          if (opts.indexOf(c.cause) < 0) opts[0] = c.cause;
          const shuffled = ML.shuffle(opts);
          const optHost = el('div', { class: 'qopts', style: 'margin-top:12px' });
          const explain = el('p', { class: 'qwhy', style: 'display:none' });
          shuffled.forEach(o => {
            const b = el('button', { class: 'qopt', type: 'button' }, [el('span', { class: 'k', text: '›' }), el('span', { text: o })]);
            b.addEventListener('click', () => {
              if (revealed) return;
              revealed = true; attempted++;
              const ok = o === c.cause;
              if (ok) score++;
              b.classList.add(ok ? 'right' : 'wrong');
              Array.prototype.forEach.call(optHost.children, x => {
                x.classList.add('locked');
                if (x.textContent.indexOf(c.cause) >= 0) x.classList.add('right');
              });
              explain.innerHTML = '<b>' + c.cause + '.</b> ' + c.why;
              explain.style.display = '';
              tally.textContent = score + ' / ' + attempted + ' correct';
            });
            optHost.appendChild(b);
          });
          wrap.appendChild(optHost);
          wrap.appendChild(explain);
          const row = el('div', { class: 'btnrow' });
          row.appendChild(el('button', {
            class: 'btn primary', type: 'button', text: 'Next curve →',
            onclick: () => { idx = (idx + 1) % CASES.length; revealed = false; paint(); }
          }));
          row.appendChild(tally);
          wrap.appendChild(row);
          ML.typeset(explain);
        }
        const tally = el('span', { class: 'small', style: 'font-family:var(--mono)', text: '0 / 0 correct' });
        paint();
        Viz.note(host, 'Ten curves, each generated by its actual mechanism rather than drawn by hand. Work through all of them; the distinctions that matter are <b>bend versus step</b> (optimisation versus data) and <b>recovers versus does not</b> (transient versus fatal).');
      },

      lrfind: function (host) {
        const st = Viz.controls(host, [
          { k: 'arch', label: 'network', type: 'select', value: 'mid', options: [{ v: 'small', t: '2–8–1 (small)' }, { v: 'mid', t: '2–16–12–1' }, { v: 'deep', t: '2–16–16–16–1 (deep)' }] },
          { k: 'act', label: 'activation', type: 'select', value: 'relu', options: [{ v: 'relu', t: 'ReLU' }, { v: 'tanh', t: 'tanh' }] },
          { k: 'burst', label: 'steps per LR', min: 3, max: 25, step: 1, value: 8, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sugg', label: 'suggested LR', cls: 'good' },
          { k: 'min', label: 'LR at minimum loss', cls: 'key' },
          { k: 'div', label: 'diverges above' },
          { k: 'note', label: 'rule' }
        ]);
        const data = Num.dataset('circles', 200, .22, 5);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const sizes = st.arch === 'small' ? [2, 8, 1] : st.arch === 'mid' ? [2, 16, 12, 1] : [2, 16, 16, 16, 1];
            const pts = [];
            for (let e = -5; e <= 1; e += .1) {
              const lr = Math.pow(10, e);
              const net = Num.mlp(sizes, { seed: 3, act: st.act });
              let L = NaN;
              for (let s = 0; s < st.burst; s++) L = net.trainBatch(data.X, data.y, lr);
              pts.push([e, isFinite(L) ? Math.min(6, L) : 6]);
            }
            const finite = pts.filter(p => p[1] < 5.9);
            let best = finite.reduce((a, b) => b[1] < a[1] ? b : a, finite[0] || [0, 1]);
            // steepest descent: largest negative slope over a small window
            let steep = best, sv = 0;
            for (let i = 3; i < pts.length - 1; i++) {
              const d = pts[i - 3][1] - pts[i][1];
              if (d > sv && pts[i][1] < best[1] * 2.2) { sv = d; steep = pts[i]; }
            }
            const divIdx = pts.findIndex(p => p[1] >= 5.9);
            const P = Viz.plot(ctx, w, h, {
              xd: [-5, 1], yd: [0, 3], pad: { l: 48, r: 14, t: 16, b: 40 }
            }).frame({ xticks: [-5, -4, -3, -2, -1, 0, 1], xfmt: v => '1e' + v, xlabel: 'learning rate', ylabel: 'loss after ' + st.burst + ' steps' });
            P.clip(() => {
              P.line(pts, { color: T.c1, width: 2.6 });
              P.vline(steep[0], { color: T.c3, width: 2, dash: false, label: 'suggested' });
              P.vline(best[0], { color: T.c4, dash: [4, 3], label: 'minimum' });
              if (divIdx > 0) {
                ctx.fillStyle = T.c2; ctx.globalAlpha = .1;
                ctx.fillRect(P.x(pts[divIdx][0]), P.pad.t, P.x(1) - P.x(pts[divIdx][0]), P.ph);
                ctx.globalAlpha = 1;
              }
            });
            out({
              sugg: Math.pow(10, steep[0]).toExponential(1),
              min: Math.pow(10, best[0]).toExponential(1),
              div: divIdx > 0 ? Math.pow(10, pts[divIdx][0]).toExponential(1) : 'not in range',
              note: 'take ~10× below the minimum'
            });
          }
        });
        Viz.note(host, 'Switch the activation to <b>tanh</b> and watch the whole curve shift left — saturating activations tolerate a smaller learning rate than ReLU. Then switch to the <b>deep</b> network: the usable window narrows, which is exactly the problem residual connections and normalisation were introduced to solve (§3.6).');
      }
    },
    quiz: [
      {
        q: 'Your 10-class classifier’s loss is pinned at 2.303. The most likely cause is…',
        options: ['overfitting', 'the model is predicting a uniform distribution — no learning is happening', 'the learning rate is too high', 'the batch size is too large'],
        answer: 1,
        why: '$\\ln 10 = 2.303$. Check label alignment and gradient flow before touching hyperparameters.'
      },
      {
        q: 'A sudden vertical step in the loss (not a bend) usually indicates…',
        options: ['a learning-rate change', 'a data pipeline event such as a shard boundary', 'overfitting', 'vanishing gradients'],
        answer: 1,
        why: 'Optimisation problems bend curves; data problems step them. Plot against data index to localise it.'
      },
      {
        q: 'The LR range test suggests taking a rate…',
        options: ['at the loss minimum', 'about an order of magnitude below the minimum, in the steepest-descent region', 'at the divergence point', 'as high as possible'],
        answer: 1,
        why: 'The minimum is already close to instability, and the loss surface sharpens as training progresses.'
      },
      {
        q: 'Comparing two runs with different batch sizes, you should plot loss against…',
        options: ['optimizer steps', 'tokens or examples seen', 'wall-clock time', 'epochs'],
        answer: 1,
        why: 'Step axes make the larger batch look artificially efficient. Many optimiser comparisons evaporate when replotted against tokens.'
      }
    ],
    cards: [
      { q: 'First thing to do with a new model', a: 'Overfit ten examples to near-zero loss with regularization off. It rules out every data and plumbing bug in two minutes.' },
      { q: 'Loss stuck at ln(C)', a: 'Uniform predictions — the model learned the prior. Check label alignment and gradient flow.' },
      { q: 'Bend vs step', a: 'A bend is an optimisation event; a vertical step is a data event.' },
      { q: 'Update-to-weight ratio', a: '$\\|\\Delta w\\|/\\|w\\| \\approx 10^{-3}$ is healthy. Orders of magnitude off in either direction is the earliest reliable warning.' },
      { q: 'LR range test', a: 'Sweep LR exponentially, plot loss; pick the steepest-descent point, roughly 10× below the minimum.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.13 */
  ML.section({
    id: 'compression', track: 'deep', num: '3.13', level: 2,
    title: 'Quantization, pruning, and distillation',
    lede: 'Three ways to make a trained model smaller and faster. They compose, they have very different risk profiles, and the arithmetic that decides which one you need is the same memory-bandwidth arithmetic as §4.14.',
    prereq: ['numerics'],
    related: ['serving', 'lora', 'numerics'],
    html: `
${H.tldr([
      '<b>Quantization</b> is nearly always the first move: INT8 is roughly free, 4-bit costs 1–3 points of quality and quarters the memory, and LLM inference is memory-bandwidth-bound so fewer bytes means proportionally more tokens per second.',
      '<b>Pruning</b>: unstructured sparsity compresses well but rarely speeds anything up without special kernels; structured pruning (whole heads, channels, layers) is slower to recover but actually runs faster.',
      '<b>Distillation</b> transfers the teacher’s <i>soft</i> predictions. The information is in the wrong answers’ relative probabilities — which is why the temperature matters and why the gradient is scaled by $T^2$.'
    ])}

<h2><span class="sn">3.13.1</span> Quantization</h2>
<p>Map a float tensor to a small integer grid: $q = \\mathrm{round}(x/s) + z$, dequantise as $\\hat x = s(q-z)$. Two decisions determine everything.</p>
${H.table(['Decision', 'Options', 'Trade'], [
      ['Symmetric or affine', '$z=0$ (symmetric) vs a learned zero-point', 'symmetric is cheaper; affine handles skewed activation ranges (post-ReLU)'],
      ['Granularity', 'per-tensor / per-channel / per-group (e.g. 64 or 128 weights)', 'finer granularity costs a few bits of overhead and recovers most of the quality — <b>group-wise is why 4-bit works at all</b>'],
      ['<b>PTQ or QAT</b>', 'quantise after training vs simulate quantization during it', 'PTQ needs ~128 calibration samples and an afternoon; QAT needs a training run and buys 1–2 points back at ≤4 bits'],
      ['What to quantise', 'weights only / weights + activations / + KV cache', 'weight-only is easiest and helps most for LLMs; activation quantization is where outliers bite']
    ])}
${H.lab('quant', 'Quantize a trained network and watch the accuracy fall', 'A real MLP, really trained, whose weights are really quantised at the bit width you choose. Everything below the plot is measured, including the accuracy — this is not a lookup table.')}
${H.flag('The hard part of LLM quantization is <b>activation outliers</b>: a handful of channels carry values 100× the rest, and per-tensor scaling then wastes the whole grid on them. LLM.int8() handles those channels in fp16; SmoothQuant migrates the difficulty from activations into weights; AWQ protects the 1% of weight channels that matter most; GPTQ solves a layerwise reconstruction problem with second-order information. All four exist because of the same outlier phenomenon.')}
${H.worked('why 4-bit quantization buys throughput, not just memory', `
<p>Decoding one token from a 70B model reads every weight once. At bf16 that is 140 GB per token; on a card with 2 TB/s of memory bandwidth the floor is $140/2000 = 70$ ms per token, or about 14 tokens/second — <b>before any computation at all</b>.</p>
<p>Quantise to 4-bit and the read becomes 35 GB, so the floor drops to 17.5 ms, about 57 tokens/second. <b>The speedup is 4×, and it comes from bandwidth, not arithmetic.</b> This is also why batching is nearly free for decode: the weights are read once for the whole batch (§4.14).</p>`)}

<h2><span class="sn">3.13.2</span> Pruning</h2>
${H.table(['Kind', 'What is removed', 'Compression', 'Actual speedup', 'Recovery'], [
      ['Unstructured magnitude', 'individual weights below a threshold', '10–20× on disk', '<b>none</b> on dense hardware', 'easy, retrain a little'],
      ['2:4 semi-structured', '2 of every 4 weights', '2×', '~1.5–1.7× on Ampere+ tensor cores', 'moderate'],
      ['Structured (channels, heads)', 'whole units', '2–4×', '<b>real, proportional</b>', 'harder; needs fine-tuning'],
      ['Layer dropping', 'whole blocks', 'proportional', 'proportional', 'surprisingly viable for deep LLMs — the middle layers are the redundant ones'],
      ['Movement pruning', 'weights moving toward zero during fine-tuning', 'high', 'as above', 'best for transfer settings']
    ])}
${H.pitfall('"We pruned 90% of the weights" almost never means "it runs 10× faster". A dense GEMM does not care that most of its inputs are zero. Unless you are targeting a sparse kernel, 2:4 sparsity on recent NVIDIA hardware, or actually deleting structural units, unstructured pruning buys disk space and nothing else. Say this in an interview and you will separate yourself from a large number of candidates.')}
${H.intuition(`<p>The <b>lottery ticket hypothesis</b> is the interesting theoretical claim here: a randomly initialised dense network contains a sparse subnetwork that, trained <i>from the same initialisation</i>, matches the full network. It is a genuine and reproducible finding at small scale. What it has not delivered is a way to find the ticket without training the dense network first — so it remains a statement about what exists rather than a practical method.</p>`)}

<h2><span class="sn">3.13.3</span> Distillation</h2>
$$\\mathcal{L} = \\alpha\\, T^2\\,\\mathrm{KL}\\!\\left(\\sigma(z_t/T)\\,\\|\\,\\sigma(z_s/T)\\right) + (1-\\alpha)\\,\\mathrm{CE}(y, \\sigma(z_s))$$
${H.deriv('why the $T^2$ is there, and why it is not a fudge', [
      ['$\\dfrac{\\partial}{\\partial z_s}\\mathrm{KL}\\big(\\sigma(z_t/T)\\,\\|\\,\\sigma(z_s/T)\\big) = \\dfrac{1}{T}\\left(\\sigma(z_s/T)-\\sigma(z_t/T)\\right)$', 'The softmax-plus-cross-entropy gradient from §0.7, with the temperature carried through the chain rule. Each logit gradient picks up a factor $1/T$.'],
      ['gradient magnitude $\\sim 1/T^2$', 'The <i>differences</i> between the softened probabilities are themselves $O(1/T)$ for large $T$, so the product shrinks quadratically.'],
      ['multiply by $T^2$', 'Restores the gradient to the same scale as the hard-label term, so that $\\alpha$ actually controls the balance and you can change $T$ without re-tuning the learning rate. <b>It is a normalisation, not a hyperparameter.</b>']
    ])}
${H.key('The value of distillation is the <i>dark knowledge</i> in the wrong answers: a teacher that says "7" with probability 0.9 and "1" with 0.08 has told the student that this 7 looks a bit like a 1. A hard label destroys that. Raising $T$ amplifies exactly those small probabilities, which is why $T$ is typically 2–10.')}
${H.table(['Variant', 'What is matched', 'Note'], [
      ['Response / logit KD', 'output distribution', 'the original; still a strong baseline'],
      ['Feature / hint KD', 'intermediate activations', 'needs a projection when widths differ'],
      ['Attention transfer', 'attention maps', 'effective for transformers'],
      ['Sequence-level KD', 'the teacher’s generated sequences as training data', '<b>the dominant LLM method</b>: generate from the teacher, fine-tune on the output'],
      ['Self-distillation', 'the model’s own earlier checkpoint', 'improves accuracy with no teacher at all, which is odd and reproducible']
    ])}
${H.flag('Distilling from a commercial API to train a competitor is prohibited by most providers’ terms of service and is a live legal question, not merely a technical one. It is also, separately, how a great many open models were actually trained. Mention the constraint if the topic comes up in an interview — awareness of it is part of the job.')}

${H.probe([
      ['You need 4× less memory on a 13B model tomorrow. What do you do?', 'Weight-only 4-bit group quantization (AWQ or GPTQ, group size 128) with a small calibration set. It is hours of work, costs a point or two, and buys ~4× on both memory and decode throughput.'],
      ['Why does INT8 quantization of LLMs fail without special handling?', 'Activation outliers: a few channels have values two orders of magnitude larger, and per-tensor scaling then wastes the grid. LLM.int8(), SmoothQuant and AWQ are three different answers.'],
      ['Why is unstructured pruning disappointing in production?', 'Dense kernels do not exploit scattered zeros. You get disk compression, not latency. Structured pruning or 2:4 sparsity is what actually runs faster.'],
      ['Why multiply the distillation loss by $T^2$?', 'The gradient through a temperature-$T$ softmax scales as $1/T^2$; the factor restores it so that $\\alpha$ balances the two terms independently of $T$.']
    ])}`,
    labs: {
      quant: function (host) {
        const st = Viz.controls(host, [
          { k: 'bits', label: 'weight bits', min: 1, max: 8, step: 1, value: 4, fmt: v => v + '-bit' },
          { k: 'group', label: 'quantization granularity', type: 'select', value: 'row', options: [{ v: 'tensor', t: 'per tensor' }, { v: 'row', t: 'per output channel' }] },
          { k: 'sym', label: 'symmetric', type: 'toggle', value: true },
          { k: 'prune', label: 'also prune smallest weights', min: 0, max: .9, step: .05, value: 0, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'fp', label: 'fp32 accuracy', cls: 'key' },
          { k: 'q', label: 'quantised accuracy', cls: 'good' },
          { k: 'drop', label: 'drop', cls: 'bad' },
          { k: 'size', label: 'model size' },
          { k: 'rmse', label: 'weight RMSE' }
        ]);
        const train = Num.dataset('moons', 260, .22, 4);
        const test = Num.dataset('moons', 400, .22, 71);
        const net = Num.mlp([2, 24, 16, 1], { seed: 5, act: 'relu' });
        for (let i = 0; i < 900; i++) net.trainBatch(train.X, train.y, .05);
        const W0 = net.W.map(l => l.map(r => r.slice()));

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            let sqErr = 0, nW = 0;
            const Wq = W0.map(layer => {
              if (st.group === 'tensor') {
                const flat = layer.flat();
                const kept = st.prune > 0 ? pruneThr(flat, st.prune) : null;
                const q = Num.quantize(flat.map(v => (kept && Math.abs(v) < kept) ? 0 : v), st.bits, { symmetric: st.sym });
                let k = 0;
                return layer.map(row => row.map(() => q.deq[k++]));
              }
              return layer.map(row => {
                const kept = st.prune > 0 ? pruneThr(row, st.prune) : null;
                const q = Num.quantize(row.map(v => (kept && Math.abs(v) < kept) ? 0 : v), st.bits, { symmetric: st.sym });
                return q.deq;
              });
            });
            W0.forEach((l, li) => l.forEach((r, ri) => r.forEach((v, ci) => { sqErr += (v - Wq[li][ri][ci]) ** 2; nW++; })));

            const accOf = (Wset) => {
              W0.forEach((l, li) => l.forEach((r, ri) => r.forEach((_, ci) => { net.W[li][ri][ci] = Wset[li][ri][ci]; })));
              return test.X.filter((p, i) => (net.predict(p) > .5 ? 1 : 0) === test.y[i]).length / test.X.length;
            };
            const accFp = accOf(W0);
            const accQ = accOf(Wq);

            const w1 = w * .5;
            const flatW = W0.flat(2), flatQ = Wq.flat(2);
            const hs = Num.hist(flatW, 40);
            const P = Viz.plot(ctx, w1, h, {
              xd: [hs.lo, hs.hi], yd: [0, Math.max.apply(null, hs.bins) * 1.15],
              pad: { l: 42, r: 8, t: 16, b: 38 }
            }).frame({ xlabel: 'weight value', ylabel: 'count' });
            P.clip(() => {
              P.bars(hs.bins, { gap: .08, color: T.line });
              const hq = Num.hist(flatQ, 40);
              // quantised weights land on a grid: draw them as spikes
              const levels = {};
              flatQ.forEach(v => { levels[v.toFixed(4)] = (levels[v.toFixed(4)] || 0) + 1; });
              Object.keys(levels).forEach(k => {
                const v = +k;
                if (v < hs.lo || v > hs.hi) return;
                ctx.strokeStyle = T.c2; ctx.lineWidth = 1.6; ctx.globalAlpha = .8;
                ctx.beginPath(); ctx.moveTo(P.x(v), P.y(0)); ctx.lineTo(P.x(v), P.y(Math.min(levels[k], hs.bins.reduce((a, b) => Math.max(a, b), 0)))); ctx.stroke();
                ctx.globalAlpha = 1;
              });
            });

            ctx.save(); ctx.translate(w1, 0);
            const P2 = Viz.plot(ctx, w - w1, h, { xd: [-2.6, 2.9], yd: [-1.9, 2.3], pad: { l: 8, r: 8, t: 16, b: 30 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'decision boundary after quantization' });
            accOf(Wq);
            P2.clip(() => {
              Labs.boundary(P2, (x, y) => net.predict([x, y]), { step: 4, lo: 0, hi: 1 });
              Labs.points(P2, test.X.slice(0, 140), test.y.slice(0, 140), { r: 2.6 });
            });
            ctx.restore();
            accOf(W0);

            const nParams = flatW.length;
            out({
              fp: (accFp * 100).toFixed(1) + '%',
              q: (accQ * 100).toFixed(1) + '%',
              drop: ((accFp - accQ) * 100).toFixed(1) + ' pts',
              size: (nParams * st.bits / 8).toFixed(0) + ' B vs ' + (nParams * 4) + ' B (' + (32 / st.bits).toFixed(1) + '×)',
              rmse: Math.sqrt(sqErr / nW).toFixed(4)
            });
          }
        });
        function pruneThr(arr, frac) {
          const s = arr.map(Math.abs).sort((a, b) => a - b);
          return s[Math.floor(frac * s.length)];
        }
        Viz.legend(host, [{ c: 'var(--line)', t: 'original fp32 weights' }, { c: 'var(--c2)', t: 'quantization grid' }]);
        Viz.note(host, 'At 8 bits the grid is dense enough that the two distributions are indistinguishable and accuracy is unchanged. At 3 bits the weights collapse onto eight levels and the decision boundary visibly coarsens. Now switch granularity to <b>per tensor</b> at 4 bits: accuracy falls further, because one scale must serve channels with very different magnitudes. <b>That is the entire argument for group-wise quantization</b>, and it is why every 4-bit LLM format has a group size.');
      }
    },
    quiz: [
      {
        q: '4-bit weight quantization speeds up LLM decoding mainly because…',
        options: ['integer arithmetic is faster', 'decoding is memory-bandwidth-bound and there are 4× fewer bytes to read', 'the model has fewer parameters', 'it enables larger batches'],
        answer: 1,
        why: 'Each decoded token reads every weight once. Bytes read, not FLOPs, set the floor.'
      },
      {
        q: 'Unstructured magnitude pruning to 90% sparsity typically gives what speedup on standard GPU kernels?',
        options: ['10×', '2×', 'essentially none', '4×'],
        answer: 2,
        why: 'A dense GEMM does not skip zeros. You get disk compression; latency needs structured sparsity or 2:4 with the right kernels.'
      },
      {
        q: 'The $T^2$ factor in the distillation loss…',
        options: ['is a tunable hyperparameter', 'compensates for the $1/T^2$ shrinkage of the gradient so $\\alpha$ stays meaningful', 'sharpens the teacher distribution', 'prevents overfitting'],
        answer: 1,
        why: 'It is a normalisation derived from the chain rule, not a knob.'
      },
      {
        q: 'The main obstacle to INT8 quantization of LLM activations is…',
        options: ['insufficient training data', 'a few channels with outlier values 100× the rest', 'lack of hardware support', 'the softmax'],
        answer: 1,
        why: 'Per-tensor scaling wastes the grid on outliers. LLM.int8(), SmoothQuant and AWQ each attack this differently.'
      }
    ],
    cards: [
      { q: 'Quantization formula', a: '$q=\\mathrm{round}(x/s)+z$, $\\hat x = s(q-z)$. Group-wise scales are what make 4-bit viable.' },
      { q: 'Why 4-bit is 4× faster to decode', a: 'Decode is memory-bandwidth-bound: 70B at bf16 reads 140 GB/token, at 4-bit 35 GB/token.' },
      { q: 'Pruning that actually speeds things up', a: 'Structured (heads, channels, layers) or 2:4 semi-structured on supported hardware. Unstructured buys disk only.' },
      { q: 'Dark knowledge', a: 'The relative probabilities the teacher assigns to wrong answers. Temperature amplifies them; $T^2$ rescales the gradient.' },
      { q: 'Modern LLM distillation', a: 'Sequence-level: generate from the teacher, fine-tune the student on the generations. Subject to provider terms of service.' }
    ]
  });

  /* ------------------------------------------------------------------ 3.14 */
  ML.section({
    id: 'robustness', track: 'deep', num: '3.14', level: 3,
    title: 'Adversarial examples and robustness',
    lede: 'A perturbation too small to see flips a confident prediction. Ten years after the discovery this is still not solved, and the honest state of the field — a measurable robustness/accuracy trade-off, and no defence that survives an adaptive attacker — is itself the thing worth knowing.',
    prereq: ['backprop'],
    related: ['safety', 'production', 'fairness'],
    html: `
${H.tldr([
      'FGSM: $x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x \\mathcal{L})$. One gradient step <i>with respect to the input</i>, in the direction that most increases the loss.',
      'Adversarial examples are not bugs in a particular model — they <b>transfer</b> between architectures and training sets, which suggests they exploit genuinely predictive but non-robust features in the data.',
      'Adversarial training is the only defence that has consistently survived. It costs 3–30× the training compute and measurably reduces clean accuracy. Almost everything else has been broken by adaptive attacks.'
    ])}

<h2><span class="sn">3.14.1</span> The attack</h2>
<p>Training moves the weights to lower the loss. An attack moves the <i>input</i> to raise it, subject to a perturbation budget $\\|\\delta\\|_p \\le \\epsilon$:</p>
$$\\max_{\\|\\delta\\|_\\infty \\le \\epsilon} \\mathcal{L}(f_\\theta(x+\\delta), y)$$
${H.table(['Attack', 'Idea', 'Strength', 'Cost'], [
      ['FGSM', 'one signed gradient step', 'weak; useful as a diagnostic', '1 backward pass'],
      ['<b>PGD</b>', 'many small steps, projected back into the ball, random start', '<b>the standard benchmark</b>', '10–100 passes'],
      ['C&amp;W', 'optimisation with a margin objective', 'very strong, finds minimal perturbations', 'expensive'],
      ['AutoAttack', 'an ensemble of four parameter-free attacks', 'the current reporting standard', 'very expensive'],
      ['Square attack', 'query-only, no gradients', 'black-box baseline', 'many queries'],
      ['Transfer attack', 'attack your own surrogate, use it on theirs', 'works without any access at all', 'cheap — and the realistic threat model']
    ])}

${H.lab('fgsm', 'Flip a real classifier with a perturbation you can measure', 'A genuinely trained network on a genuine 2-D problem. The attack computes $\\nabla_x\\mathcal{L}$ by central differences and steps along it. Watch the point cross a boundary it was nowhere near — and watch how much smaller ε needs to be near the boundary than far from it.')}

${H.intuition(`<p>Why does a tiny perturbation matter so much? In high dimensions, a small per-pixel budget is a large total budget: an $\\epsilon = 8/255$ change to every one of 150,528 pixels has L2 norm around 12, which is not small at all. The linear explanation is that $w^\\top\\delta$ can be large when $\\delta$ aligns with $w$ and $d$ is big, even with $\\|\\delta\\|_\\infty$ tiny. Deep networks are locally close to linear, so the attack needs one gradient rather than a search.</p>`)}

<h2><span class="sn">3.14.2</span> Why they transfer, and what that implies</h2>
${H.key('The most credible account (Ilyas et al., 2019): datasets contain features that are genuinely predictive but not robust — real statistical signal that a small perturbation destroys. Models learn them because they work. Adversarial examples are therefore a property of the <i>data</i> as much as of the model, which is why they transfer across architectures.')}
${H.flag('This reframing is well supported but not universally accepted, and the field has a long history of defences that looked sound and were broken within months — usually because they obscured the gradient rather than removing the vulnerability. Any claim of a new defence should be read alongside whether it was evaluated against an <i>adaptive</i> attacker who knows the defence exists.')}

<h2><span class="sn">3.14.3</span> Defences, honestly assessed</h2>
${H.table(['Defence', 'Idea', 'Status'], [
      ['<b>Adversarial training</b>', 'train on PGD examples generated on the fly', '<b>works</b>; 3–30× compute, and a real clean-accuracy cost'],
      ['TRADES', 'explicitly trade clean accuracy against robustness with a tunable term', 'works; makes the trade-off a dial'],
      ['Randomised smoothing', 'add Gaussian noise, take a majority vote', '<b>certified</b> robustness in L2 — a guarantee, at a modest accuracy cost'],
      ['Gradient masking / obfuscation', 'make gradients uninformative', '<b>broken</b> — repeatedly, by attacks that estimate the gradient another way'],
      ['Input preprocessing (JPEG, blur, quantise)', 'destroy the perturbation', 'broken by BPDA (approximate the non-differentiable step)'],
      ['Detection', 'classify inputs as adversarial', 'mostly broken; the attacker can attack the detector too'],
      ['Ensembles', 'attack must fool all members', 'raises the cost, does not solve it — perturbations transfer within the ensemble']
    ])}
${H.worked('the robustness/accuracy trade-off, in numbers', `
<p>On CIFAR-10 at $\\epsilon = 8/255$ (L∞), the picture has been stable for years:</p>
<ul>
<li>A standard model: ~95% clean, <b>~0%</b> robust under AutoAttack.</li>
<li>Adversarially trained: ~85% clean, ~50–60% robust — using extra generated data, the current leaders sit around 70%.</li>
<li>Randomised smoothing gives a <i>certificate</i>: a provable radius, at lower clean accuracy.</li>
</ul>
<p><b>Ten points of clean accuracy for fifty points of robustness</b> is the trade, and whether it is worth paying is a question about your threat model, not about machine learning. For a photo-tagging feature: no. For a content-moderation classifier with a motivated adversary: yes.</p>`)}

${H.more('Natural distribution shift is the more common problem', `
<p>Adversarial robustness gets the attention; ordinary distribution shift causes far more production failures. The taxonomy worth having:</p>
<ul>
<li><b>Covariate shift</b>: $p(x)$ changes, $p(y\\mid x)$ does not. New camera, new user segment. Detectable from inputs alone — this is what PSI monitoring catches (§2.18).</li>
<li><b>Label shift</b>: $p(y)$ changes, $p(x\\mid y)$ does not. Fraud rate triples. Fixable by reweighting if you can estimate the new prior.</li>
<li><b>Concept drift</b>: $p(y\\mid x)$ itself changes. The relationship you learned is no longer true. <b>Only labels can detect this</b>, which is why outcome monitoring cannot be replaced by input monitoring.</li>
</ul>
<p>The practical consequence: a model that is 95% accurate in the lab and 78% in production has almost certainly met covariate shift, not an adversary. Check the input distribution first.</p>`)}

${H.probe([
      ['Write down FGSM.', '$x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L}(f(x),y))$ — one signed gradient step on the input, clipped to the valid range.'],
      ['Why does a defence that "hides the gradient" fail?', 'It removes the attacker’s easiest path, not the vulnerability. Adaptive attacks estimate the gradient by finite differences, transfer from a surrogate, or replace the non-differentiable step with a differentiable approximation (BPDA).'],
      ['What is the cost of adversarial training?', '3–30× training compute (each step needs a PGD inner loop) and roughly ten points of clean accuracy on CIFAR-10 at the standard budget.'],
      ['You see accuracy drop in production. How do you tell adversarial attack from distribution shift?', 'Shift moves the whole input distribution and is visible in PSI/KS on features. An attack leaves the marginal distribution nearly unchanged but produces high-confidence errors concentrated near boundaries. Check input drift first — it is far more likely.']
    ], 'Presenting adversarial robustness as solved, or as a niche curiosity. It is neither: it is unsolved, and it matters exactly as much as your threat model says it does.')}`,
    labs: {
      fgsm: function (host) {
        const st = Viz.controls(host, [
          { k: 'eps', label: 'perturbation budget ε', min: 0, max: 1.2, step: .02, value: .35, fmt: v => v.toFixed(2) },
          { k: 'steps', label: 'attack steps (1 = FGSM, >1 = PGD)', min: 1, max: 20, step: 1, value: 1, fmt: v => v },
          { k: 'px', label: 'point x', min: -2.2, max: 2.6, step: .05, value: -1.0, fmt: v => v.toFixed(2) },
          { k: 'py', label: 'point y', min: -1.6, max: 2.2, step: .05, value: 0.45, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'orig', label: 'original prediction', cls: 'key' },
          { k: 'adv', label: 'after attack', cls: 'bad' },
          { k: 'flip', label: 'flipped?' },
          { k: 'dist', label: 'perturbation ‖δ‖₂' },
          { k: 'need', label: 'ε needed to flip' }
        ]);
        const data = Num.dataset('moons', 240, .2, 8);
        const net = Num.mlp([2, 20, 14, 1], { seed: 6, act: 'tanh' });
        for (let i = 0; i < 1400; i++) net.trainBatch(data.X, data.y, .05);

        function loss(p, y) {
          const q = Math.min(1 - 1e-9, Math.max(1e-9, net.predict(p)));
          return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
        }
        function attack(p, eps, steps) {
          const y = net.predict(p) > .5 ? 1 : 0;
          let cur = p.slice();
          const path = [cur.slice()];
          const alpha = steps > 1 ? eps / steps * 2.2 : eps;
          for (let s = 0; s < steps; s++) {
            const g = Num.inputGrad(q => loss(q, y), cur, 1e-4);
            cur = cur.map((v, i) => v + alpha * Math.sign(g[i]));
            // project back into the L∞ ball around p
            cur = cur.map((v, i) => Math.max(p[i] - eps, Math.min(p[i] + eps, v)));
            path.push(cur.slice());
          }
          return { adv: cur, path: path, y: y };
        }

        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const p = [st.px, st.py];
            const a = attack(p, st.eps, st.steps);
            const p0 = net.predict(p), p1 = net.predict(a.adv);
            const P = Viz.plot(ctx, w, h, { xd: [-2.6, 3.0], yd: [-1.9, 2.4], pad: { l: 40, r: 14, t: 16, b: 34 } })
              .frame({ xlabel: 'x₁', ylabel: 'x₂' });
            P.clip(() => {
              Labs.boundary(P, (x, y) => net.predict([x, y]), { step: 4, lo: 0, hi: 1, alpha: 80 });
              Labs.points(P, data.X, data.y, { r: 2.4 });
              // the L∞ ball
              ctx.strokeStyle = T.text; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
              ctx.strokeRect(P.x(p[0] - st.eps), P.y(p[1] + st.eps),
                P.x(p[0] + st.eps) - P.x(p[0] - st.eps), P.y(p[1] - st.eps) - P.y(p[1] + st.eps));
              ctx.setLineDash([]);
              P.line(a.path, { color: T.c4, width: 1.8 });
              P.dots([p], { r: 6, color: T.text, stroke: true, strokeWidth: 2 });
              P.dots([a.adv], { r: 6, color: T.c2, stroke: true, strokeWidth: 2 });
              P.text(p[0], p[1], ' original', { dx: 9, dy: -11, color: T.text, font: '11px ui-sans-serif' });
              P.text(a.adv[0], a.adv[1], ' adversarial', { dx: 9, dy: 12, color: T.c2, font: '11px ui-sans-serif' });
            });
            // the smallest ε that flips it
            let need = null;
            for (let e = .02; e <= 1.6; e += .02) {
              const r = attack(p, e, Math.max(st.steps, 6));
              if ((net.predict(r.adv) > .5 ? 1 : 0) !== a.y) { need = e; break; }
            }
            out({
              orig: (p0 > .5 ? 'class 1' : 'class 0') + ' @ ' + (Math.max(p0, 1 - p0) * 100).toFixed(1) + '%',
              adv: (p1 > .5 ? 'class 1' : 'class 0') + ' @ ' + (Math.max(p1, 1 - p1) * 100).toFixed(1) + '%',
              flip: ((p1 > .5 ? 1 : 0) !== a.y) ? 'YES — attack succeeded' : 'no',
              dist: Math.hypot(a.adv[0] - p[0], a.adv[1] - p[1]).toFixed(3),
              need: need ? need.toFixed(2) : '> 1.6'
            });
          }
        });
        Viz.note(host, 'Move the point deep into a cluster: the ε needed to flip it grows, because the distance to the boundary is the thing the attack must cross. Move it near the boundary and a budget of 0.05 suffices. <b>Distance to the decision boundary is the local robustness</b>, and adversarial training works by explicitly pushing that distance up for every training point — which is also, unavoidably, why it moves the boundary away from where clean accuracy would have put it.');
      }
    },
    quiz: [
      {
        q: 'FGSM perturbs the input by…',
        options: ['random noise of magnitude ε', '$\\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L})$', 'the negative gradient of the loss', 'the model’s weights'],
        answer: 1,
        why: 'One signed step along the input gradient, which is the direction that increases the loss fastest under an L∞ budget.'
      },
      {
        q: 'Adversarial examples transfer between independently trained models, which suggests…',
        options: ['all models share a bug', 'they exploit genuinely predictive but non-robust features of the data', 'the attacks are random', 'the models were trained on the same seed'],
        answer: 1,
        why: 'The Ilyas et al. account: non-robust features are real signal, so any model that maximises accuracy learns them.'
      },
      {
        q: 'A defence that makes gradients uninformative is…',
        options: ['a strong defence', 'gradient masking — historically broken by adaptive attacks', 'certified robustness', 'equivalent to adversarial training'],
        answer: 1,
        why: 'Obscuring the gradient removes the easiest attack path, not the vulnerability. BPDA and transfer attacks route around it.'
      },
      {
        q: 'Production accuracy dropped 15 points. The most likely cause is…',
        options: ['an adversarial attack', 'covariate shift — the input distribution moved', 'a bug in the loss function', 'label leakage'],
        answer: 1,
        why: 'Ordinary distribution shift causes far more production failures than adversaries. Check PSI/KS on the input features first.'
      }
    ],
    cards: [
      { q: 'FGSM', a: '$x\' = x + \\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L})$. PGD is the iterative, projected, random-start version — the benchmark.' },
      { q: 'Why perturbations transfer', a: 'They exploit non-robust but genuinely predictive features in the data, so independently trained models learn the same vulnerability.' },
      { q: 'The robustness trade-off', a: 'CIFAR-10 at ε=8/255: ~95% clean / ~0% robust, vs ~85% clean / ~55% robust after adversarial training.' },
      { q: 'Three kinds of shift', a: 'Covariate ($p(x)$), label ($p(y)$), concept ($p(y|x)$). Only the last needs labels to detect.' }
    ]
  });
})();
