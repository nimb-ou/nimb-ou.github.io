/* ============================================================
   PART 4 — LLMs & transformers (4.9 – 4.18)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 4.9 */
  ML.section({
    id: 'pretraining', track: 'llm', num: '4.9',
    title: 'Pretraining',
    lede: 'The objective is one line of mathematics; the corpus is the whole job.',
    html: `
<h2><span class="sn">4.9.1</span> The objective</h2>
<p>Next-token prediction: categorical cross-entropy over the vocabulary at every position, which by §1.5 is just the negative log-likelihood of a Categorical label. There is nothing else to it. Everything interesting is in the data.</p>

<h2><span class="sn">4.9.2</span> The corpus funnel</h2>
${H.table(['Stage', 'What it does', 'Why it matters'], [
      ['<b>Deduplication</b>', 'Exact and near-duplicate removal via MinHash or suffix arrays', 'Measurably improves quality per token <i>and</i> sharply reduces verbatim memorisation'],
      ['<b>Quality filtering</b>', 'A classifier trained to recognise reference-quality text', 'A smaller, cleaner corpus reliably beats a larger, dirtier one at equal compute'],
      ['<b>Mixture weights</b>', 'Proportions across code, web, books, maths, multilingual', 'A first-class hyperparameter — code in the mixture improves reasoning on non-code tasks'],
      ['<b>Curriculum / anneal</b>', 'Ordering, and a high-quality phase at the end', 'A further gain for close to free; the decay phase is when the model is most sensitive'],
      ['<b>Decontamination</b>', 'Remove eval-set overlap — <i>before</i> training, not after', 'Otherwise your benchmark measures the benchmark’s age (§4.16)']
    ])}

${H.lab('corpus', 'The corpus funnel, with the knobs that matter', 'Move the filters and watch tokens fall while modelled quality rises. The curve is illustrative of the published direction of travel — the point is the shape: aggressive filtering costs tokens and buys quality, up to the point where you run out of data.')}

<h2><span class="sn">4.9.3</span> Synthetic data is now a first-class ingredient</h2>
<p>Not a fallback. Three mechanisms carry most of the value: <b>distillation from a stronger model</b> (generate answers with a frontier model, train a small one on them — the single cheapest way to move a small model's quality, subject to the teacher's licence terms); <b>self-instruct bootstrapping</b>, where a model writes new instructions and answers from a seed set which are then filtered; and <b>verifier-filtered generation</b>, where you keep only samples that pass a checker — the same idea as RLVR (§4.12) applied to data rather than rewards, and the reason maths and code benefit most.</p>
${H.flag('The caveat to state before anyone asks: training repeatedly on unfiltered self-generated text degrades diversity and eventually quality — "model collapse". Synthetic data works when it is filtered against something real, and fails when it is a closed loop.')}

<h2><span class="sn">4.9.4</span> Continual pretraining</h2>
<p>The third option between pretraining and fine-tuning: take an existing base model and keep training on a domain corpus (legal, medical, one bank's documents) with a low learning rate and a replay mixture of the original data to prevent catastrophic forgetting. It buys real domain knowledge that LoRA cannot (§4.13), and it is the honest answer to "can we make it know our products" when retrieval is genuinely insufficient.</p>

${H.probe([
      ['What is the pretraining objective?', 'Next-token categorical cross-entropy — the NLL of a Categorical label, nothing more.'],
      ['Bigger corpus or cleaner corpus?', 'Cleaner, at equal compute — deduplication and quality filtering both beat raw volume, and dedup also cuts memorisation.'],
      ['When does synthetic data fail?', 'When it is a closed loop; it works when filtered against something real (a verifier, a stronger model, human data).']
    ])}`,
    labs: {
      corpus: function (host) {
        const st = Viz.controls(host, [
          { k: 'raw', label: 'raw crawl (trillions of tokens)', min: 5, max: 200, step: 5, value: 100, fmt: v => v + 'T' },
          { k: 'dedup', label: 'deduplication strength', min: 0, max: 1, step: .05, value: .6, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'quality', label: 'quality-filter strictness', min: 0, max: 1, step: .05, value: .5, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'anneal', label: 'final high-quality anneal', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'tokens', label: 'usable tokens', cls: 'key' }, { k: 'quality', label: 'relative quality per token', cls: 'good' },
          { k: 'memo', label: 'verbatim memorisation risk' }, { k: 'model', label: 'Chinchilla-optimal model size' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const stages = [
              { l: 'raw crawl', f: 1, c: T.line },
              { l: 'language ID + boilerplate strip', f: .55, c: T.faint },
              { l: 'deduplication (MinHash)', f: .55 * (1 - .45 * st.dedup), c: T.blue },
              { l: 'quality classifier', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality), c: T.red },
              { l: 'mixture weighting', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality) * .92, c: T.text },
              { l: st.anneal ? 'final anneal on the best data' : '(no anneal)', f: .55 * (1 - .45 * st.dedup) * (1 - .55 * st.quality) * .92 * (st.anneal ? 1 : 1), c: T.amber }
            ];
            const bx = 20, bw = w - 260;
            stages.forEach((s, i) => {
              const y = 26 + i * 40;
              ctx.fillStyle = s.c;
              ctx.globalAlpha = i === 5 && !st.anneal ? .25 : 1;
              ctx.fillRect(bx, y, bw * s.f, 26);
              ctx.globalAlpha = 1;
              ctx.fillStyle = T.text; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              ctx.fillText(s.l, bx + bw * s.f + 10, y + 13);
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText((st.raw * s.f).toFixed(1) + 'T', bx + 6, y + 13);
            });
            const finalT = st.raw * stages[4].f;
            const quality = 1 + .55 * st.quality + .25 * st.dedup + (st.anneal ? .12 : 0);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('yellow tail = the high-quality anneal phase: a small fraction of tokens, a disproportionate share of the final quality', bx, 26 + 6 * 40 + 6);
            out({
              tokens: finalT.toFixed(1) + 'T',
              quality: '×' + quality.toFixed(2),
              memo: st.dedup > .5 ? 'low' : st.dedup > .2 ? 'moderate' : 'high',
              model: (finalT * 1e12 / 20 / 1e9).toFixed(0) + 'B'
            });
          }
        });
        Viz.note(host, 'The last readout applies Chinchilla’s 20 tokens per parameter (§4.10) to whatever corpus survives your filters — which is the honest way to discover that the model size you can train is set by your <i>data pipeline</i>, not your GPU budget.');
      }
    },
    quiz: [
      {
        q: 'Deduplication improves models mainly by…',
        options: ['reducing training time only', 'improving quality per token and sharply reducing verbatim memorisation', 'increasing vocabulary coverage', 'removing toxic content'],
        answer: 1,
        why: 'Both effects are documented; memorisation reduction matters for privacy and for benchmark integrity.'
      },
      {
        q: 'Training a model repeatedly on its own unfiltered outputs leads to…',
        options: ['steady improvement', 'model collapse — diversity and then quality degrade', 'faster convergence', 'better calibration'],
        answer: 1,
        why: 'Synthetic data works when filtered against something real; a closed loop narrows the distribution.'
      }
    ],
    cards: [
      { q: 'Pretraining objective', a: 'Next-token categorical cross-entropy — the NLL of a Categorical label.' },
      { q: 'Corpus funnel', a: 'Language ID → dedup (MinHash) → quality classifier → mixture weights → high-quality anneal; decontaminate before training.' },
      { q: 'Synthetic data, three mechanisms', a: 'Distillation from a stronger model, self-instruct bootstrapping, verifier-filtered generation. Fails as a closed loop.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.10 */
  ML.section({
    id: 'scaling-laws', track: 'llm', num: '4.10',
    title: 'Scaling laws, Chinchilla, compute- vs inference-optimal',
    lede: 'One formula lets you sanity-check any training claim in seconds — and one distinction explains why everyone now trains past the compute-optimal point.',
    html: `
<h2><span class="sn">4.10.1</span> The accounting</h2>
<p>Loss follows smooth power laws in parameters $N$, data $D$ and compute $C$, with the standard accounting</p>
$$C \\approx 6ND \\text{ FLOPs}$$
<p>for a decoder-only transformer — two for the forward pass, four for the backward. That single formula lets you check a claimed training run in seconds.</p>

<h2><span class="sn">4.10.2</span> Chinchilla</h2>
<p>Hoffmann et al. (2022) found the field had been training models far too large on far too little data: for a fixed compute budget, parameters and tokens should scale <b>together</b>, at roughly <mark>20 tokens per parameter</mark>. Their demonstration was a 70B model on 1.4T tokens beating a 280B model trained on far less.</p>
${H.flag('Flag when you cite this: the ~20:1 headline is robust and replicated, but the exact fitted coefficients are contested — Epoch AI’s replication differs from the original fit. Quote the ratio, not the exponents.')}

${H.lab('chinchilla', 'The iso-loss surface, and where your budget lands', 'Move the compute budget and watch the compute-optimal point slide along the 20-tokens-per-parameter ray. Then move along a single iso-loss contour to see the inference-optimal trade: identical loss, a smaller model, far more data — and a much cheaper deployment.')}

<h2><span class="sn">4.10.3</span> Inference-optimal is a different question</h2>
<p>Chinchilla minimises training loss per training FLOP. If you will serve the model to millions of users, lifetime inference cost dominates training cost, so it pays to "overtrain" a <i>smaller</i> model on far more tokens: Llama-3-8B saw roughly <b>1,875 tokens per parameter</b>, about 90× the Chinchilla ratio. This is now standard practice for anything deployed. The one-line answer to "why train past compute-optimal": <mark>you pay training once and inference forever.</mark></p>

${H.lab('budget', 'Training budget calculator', 'Enter any two of parameters, tokens and compute and read off the third, with GPU-hours and a rough cost. Check it against a published run — the arithmetic is the same one used to plan them.')}

${H.probe([
      ['Chinchilla in one line?', 'For a fixed training budget, scale parameters and tokens together at about 20 tokens per parameter.'],
      ['Then why does everyone train past it?', 'Deployment cost is dominated by inference; a smaller overtrained model is cheaper forever.'],
      ['Compute for a 7B on 2T tokens?', '$6ND \\approx 6\\cdot7\\times10^9\\cdot2\\times10^{12} \\approx 8.4\\times10^{22}$ FLOPs.']
    ])}`,
    labs: {
      chinchilla: function (host) {
        const st = Viz.controls(host, [
          { k: 'logC', label: 'compute budget (FLOPs)', min: 19, max: 26, step: .1, value: 23, fmt: v => '10^' + v.toFixed(1) },
          { k: 'ratio', label: 'tokens per parameter', min: 5, max: 2000, step: 5, value: 20, fmt: v => v + ':1' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'nopt', label: 'compute-optimal N', cls: 'key' }, { k: 'dopt', label: 'compute-optimal D' },
          { k: 'nyour', label: 'N at your ratio' }, { k: 'dyour', label: 'D at your ratio' }, { k: 'infer', label: 'relative inference cost', cls: 'good' }
        ]);
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const C = Math.pow(10, st.logC);
            // Chinchilla-flavoured loss surface: L = E + A/N^a + B/D^b
            const E = 1.69, A = 406.4, a = 0.34, B = 410.7, b = 0.28;
            const loss = (N, D) => E + A / Math.pow(N, a) + B / Math.pow(D, b);
            const P = Viz.plot(ctx, w, h, { xd: [8, 12.2], yd: [10, 14] })
              .frame({ xlabel: 'log₁₀ parameters N', ylabel: 'log₁₀ tokens D', xfmt: v => '10^' + v.toFixed(0), yfmt: v => '10^' + v.toFixed(0) });
            P.contours((ln, ld) => loss(Math.pow(10, ln), Math.pow(10, ld)), [1.9, 2.0, 2.15, 2.3, 2.5, 2.8, 3.2], { color: T.faint, alpha: .6, nx: 70, ny: 60 });
            P.clip(() => {
              // 20 tokens/param ray
              P.fn(ln => Math.log10(20 * Math.pow(10, ln)), { color: T.red, width: 2, dash: [6, 4] });
              // iso-compute curve: D = C/(6N)
              P.fn(ln => Math.log10(C / (6 * Math.pow(10, ln))), { color: T.blue, width: 2.6 });
              // optimal point on this budget at 20:1
              const Nopt = Math.sqrt(C / (6 * 20));
              const Dopt = 20 * Nopt;
              P.dots([[Math.log10(Nopt), Math.log10(Dopt)]], { r: 6, color: T.red, stroke: true });
              P.text(Math.log10(Nopt), Math.log10(Dopt), '  A · compute-optimal', { color: T.red, font: '11px ui-sans-serif' });
              const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
              P.dots([[Math.log10(Nyour), Math.log10(Dyour)]], { r: 6, color: T.green, stroke: true });
              P.text(Math.log10(Nyour), Math.log10(Dyour), '  B · your ratio', { color: T.green, font: '11px ui-sans-serif' });
            });
            const Nopt = Math.sqrt(C / (6 * 20)), Dopt = 20 * Nopt;
            const Nyour = Math.sqrt(C / (6 * st.ratio)), Dyour = st.ratio * Nyour;
            out({
              nopt: (Nopt / 1e9).toFixed(1) + 'B', dopt: (Dopt / 1e12).toFixed(2) + 'T',
              nyour: (Nyour / 1e9).toFixed(1) + 'B', dyour: (Dyour / 1e12).toFixed(2) + 'T',
              infer: '×' + (Nyour / Nopt).toFixed(2)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Chinchilla 20:1', primary: true, on: () => { st.$set('ratio', 20); S.redraw(); } },
          { label: 'Llama-3-8B ≈ 1875:1', on: () => { st.$set('ratio', 1875); S.redraw(); } }
        ]);
        Viz.note(host, 'Both points sit on the same iso-compute curve — the same training budget. Point B is a much smaller model trained on far more tokens: similar loss, and the "relative inference cost" readout is what you pay forever afterwards. That single comparison is why the field moved past compute-optimal.');
      },

      budget: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'parameters (B)', min: .1, max: 700, step: .1, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'D', label: 'training tokens (T)', min: .1, max: 30, step: .1, value: 15, fmt: v => v.toFixed(1) + 'T' },
          { k: 'gpu', label: 'GPU peak (TFLOP/s)', min: 100, max: 2500, step: 10, value: 990, fmt: v => v },
          { k: 'mfu', label: 'MFU achieved', min: .15, max: .65, step: .01, value: .42, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'ngpu', label: 'GPUs', min: 8, max: 32768, step: 8, value: 1024, fmt: v => v.toLocaleString() },
          { k: 'cost', label: '$ per GPU-hour', min: .5, max: 10, step: .1, value: 2.5, fmt: v => '$' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'flops', label: 'total FLOPs (6ND)', cls: 'key' }, { k: 'gpuh', label: 'GPU-hours' },
          { k: 'days', label: 'wall-clock days' }, { k: 'usd', label: 'approximate cost' }, { k: 'ratio', label: 'tokens per parameter' }
        ]);
        const S = Viz.surface(host, {
          height: 240,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, D = st.D * 1e12;
            const flops = 6 * N * D;
            const gpuSec = flops / (st.gpu * 1e12 * st.mfu);
            const gpuH = gpuSec / 3600;
            const days = gpuH / st.ngpu / 24;
            ctx.font = '14px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const lines = [
              ['C = 6ND', '= 6 × ' + st.N.toFixed(1) + 'e9 × ' + st.D.toFixed(1) + 'e12', T.text],
              ['', '= ' + flops.toExponential(2) + ' FLOPs', T.blue],
              ['GPU-seconds', '= C / (peak × MFU) = ' + (gpuSec / 1e6).toFixed(1) + 'M s', T.text],
              ['GPU-hours', '= ' + Math.round(gpuH).toLocaleString(), T.text],
              ['wall clock', '= ' + days.toFixed(1) + ' days on ' + st.ngpu.toLocaleString() + ' GPUs', T.green],
              ['cost', '≈ $' + (gpuH * st.cost / 1e6).toFixed(2) + 'M', T.amber]
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], 18, 20 + i * 30);
              ctx.fillStyle = L[2]; ctx.font = 'bold 14px ui-monospace, monospace';
              ctx.fillText(L[1], 150, 20 + i * 30);
              ctx.font = '14px ui-monospace, monospace';
            });
            out({
              flops: flops.toExponential(2), gpuh: Math.round(gpuH).toLocaleString(),
              days: days.toFixed(1), usd: '$' + (gpuH * st.cost / 1e6).toFixed(2) + 'M',
              ratio: (D / N).toFixed(0) + ':1'
            });
          }
        });
        Viz.note(host, 'Set 7B and 2T tokens: 8.4×10²² FLOPs, the number quoted in the interview box. The MFU slider is where reality lives — 42% is a good large run, and the difference between 42% and 25% is months of wall clock.');
      }
    },
    quiz: [
      {
        q: 'Training compute for a 7B model on 2T tokens is approximately…',
        options: ['$8.4\\times10^{19}$', '$8.4\\times10^{22}$', '$1.4\\times10^{25}$', '$6\\times10^{12}$'],
        answer: 1,
        why: '$6ND = 6 \\times 7\\times10^9 \\times 2\\times10^{12} = 8.4\\times10^{22}$ FLOPs.'
      },
      {
        q: 'Why do deployed models train far past the Chinchilla ratio?',
        options: ['Chinchilla was wrong', 'Inference cost dominates lifetime cost, so a smaller overtrained model is cheaper to serve forever', 'Larger datasets are cheaper', 'It improves calibration'],
        answer: 1,
        why: 'Chinchilla optimises training loss per training FLOP; deployment optimises total lifetime cost.'
      }
    ],
    cards: [
      { q: 'Compute accounting', a: '$C \\approx 6ND$ — 2N forward, 4N backward, times D tokens.' },
      { q: 'Chinchilla', a: '≈20 tokens per parameter at fixed compute (70B on 1.4T beat 280B). ⚑ Exact coefficients contested — quote the ratio.' },
      { q: 'Inference-optimal', a: 'Overtrain a smaller model far past 20:1 (Llama-3-8B ≈ 1,875 tok/param) — pay training once, inference forever.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.11 */
  ML.section({
    id: 'distributed', track: 'llm', num: '4.11',
    title: 'Distributed training, and debugging a diverging run',
    lede: 'Three parallelisms, one memory budget, and an ordered list that interviewers ask for precisely because anyone can name the causes.',
    html: `
<h2><span class="sn">4.11.1</span> Three parallelisms</h2>
${H.table(['Kind', 'Splits', 'Communication', 'Rule of thumb'], [
      ['<b>Data</b>', 'the batch; every GPU holds the whole model', 'one all-reduce of gradients per step', 'Simplest; the default outer layer'],
      ['<b>Tensor</b>', 'individual weight matrices across GPUs', 'heavy, every layer', 'Keep it inside a node — NVLink, not Ethernet'],
      ['<b>Pipeline</b>', 'layers across devices', 'cheap, point-to-point', 'Bubbles unless you micro-batch'],
      ['<b>Expert</b>', 'MoE experts across devices', 'all-to-all', 'Adds a fourth dimension for MoE (§4.8)']
    ])}
<p>Real runs combine all three (3-D parallelism). <b>ZeRO / FSDP</b> shard optimizer states, gradients and parameters across data-parallel ranks, which is what makes large models fit at all: Adam alone needs about <b>12 bytes per parameter</b> of FP32 state, on top of the weights and gradients.</p>

${H.lab('mem', 'The training memory budget', 'Every consumer of memory, computed. Toggle ZeRO stages and activation checkpointing and watch the total fall below your card — this is the calculation that decides whether a run is possible.')}

<h2><span class="sn">4.11.2</span> The stability toolkit</h2>
<p>Mixed precision (BF16 compute, FP32 master weights — BF16 over FP16 because its exponent range makes loss scaling unnecessary), gradient accumulation to simulate a large batch, activation checkpointing to trade recompute for memory (§3.2's cache, given up deliberately), linear warmup then cosine or WSD decay (§3.5), and gradient clipping by global norm.</p>

<h3>Two schedule facts worth knowing</h3>
<p>Cosine requires committing to a total step count in advance, which is awkward when you may extend a run; WSD holds the rate flat and decays sharply only at the end, matching cosine's final loss while letting you checkpoint mid-run and branch — one stable trunk, several short decay phases for different data mixtures. And <b>μP</b> rescales initialisations and learning rates by width so hyperparameters found on a small proxy model <i>transfer</i> to the large one, turning hyperparameter search on a billion-dollar run from impossible into a sweep on something cheap.</p>

<h3>Batch size is not free</h3>
<p>Below the <b>critical batch size</b>, doubling the batch nearly halves the number of steps needed; above it you are paying compute for almost nothing. The critical size grows as the loss falls, which is why large runs ramp batch size during training. When you do increase it, scale the learning rate — square-root for Adam, linear for SGD — or you will silently under-train.</p>

${H.fig('THE LOSS WENT TO NaN — WORK THE LIST IN THIS ORDER', H.steps([
      '<b>Lower the learning rate</b>, and verify warmup is actually running — a restart that skips warmup is the classic cause.',
      '<b>Check the data:</b> NaNs or infs in a shard, unnormalised inputs, a corrupt record. Print the batch that preceded the blow-up.',
      '<b>Add or tighten gradient clipping</b>, and watch the global-norm trace for the spike <i>before</i> the NaN.',
      '<b>Check mixed-precision handling</b> — loss scaling under FP16, or an fp32 softmax/norm that got cast down.',
      '<b>Confirm the LR the scheduler is actually emitting</b> after a resume from checkpoint.',
      '<b>Reduce batch size</b>, or isolate a bad shard / a single misbehaving rank.'
    ]), 'Interviewers ask this to hear an ordered list. Anyone can name the causes; the signal is knowing which to check first and why.')}

${H.lab('nan', 'The spike before the NaN', 'A simulated run where the gradient-norm trace spikes one or two steps before the loss goes vertical. Watch which trace moves first — that is the step whose batch you want to print, and the reason experienced people watch the norm rather than the loss.')}

${H.probe([
      ['Which parallelism goes inside a node?', 'Tensor parallelism — it is the chattiest, so it wants NVLink rather than the network.'],
      ['How much memory does Adam need?', '≈12 bytes per parameter of FP32 optimizer state, plus weights and gradients — hence ZeRO/FSDP sharding.'],
      ['What do you check first when the loss NaNs?', 'The learning rate and whether warmup is running, then the data, then clipping — and watch the gradient-norm trace, not the loss.']
    ])}`,
    labs: {
      mem: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'parameters (B)', min: .5, max: 200, step: .5, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'gpus', label: 'data-parallel ranks', min: 1, max: 512, step: 1, value: 8, fmt: v => v },
          { k: 'zero', label: 'ZeRO stage', type: 'buttons', value: '2', options: [{ v: '0', t: 'none' }, { v: '1', t: 'ZeRO-1' }, { v: '2', t: 'ZeRO-2' }, { v: '3', t: 'ZeRO-3 / FSDP' }] },
          { k: 'ckpt', label: 'activation checkpointing', type: 'toggle', value: true },
          { k: 'batch', label: 'per-GPU batch × seq (k tokens)', min: 1, max: 64, step: 1, value: 8, fmt: v => v + 'k' },
          { k: 'card', label: 'GPU memory (GB)', min: 16, max: 192, step: 8, value: 80, fmt: v => v + ' GB' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'total', label: 'per-GPU memory', cls: 'key' }, { k: 'fits', label: 'fits?' },
          { k: 'opt', label: 'optimizer state' }, { k: 'act', label: 'activations' }
        ]);
        const S = Viz.surface(host, {
          height: 270,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, z = +st.zero, ranks = st.gpus;
            const weights = N * 2 / 1e9;                     // BF16
            const grads = (z >= 2 ? N * 2 / ranks : N * 2) / 1e9;
            const optim = (z >= 1 ? N * 12 / ranks : N * 12) / 1e9;
            const params = (z >= 3 ? weights / ranks : weights);
            const acts = (st.batch * 1000 * Math.sqrt(N / 1e9) * (st.ckpt ? 0.9 : 9)) * 2 / 1e6;
            const parts = [
              ['parameters (BF16)', params, T.blue],
              ['gradients', grads, T.amber],
              ['optimizer state (Adam FP32)', optim, T.red],
              ['activations', acts, T.green]
            ];
            const total = parts.reduce((a, p) => a + p[1], 0);
            const bx = 20, bw = w - 40, by = 40;
            let x = bx;
            parts.forEach(p => {
              const pw = bw * p[1] / Math.max(total, st.card);
              ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 36);
              x += pw;
            });
            // card limit
            const limitX = bx + bw * st.card / Math.max(total, st.card);
            ctx.strokeStyle = T.text; ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(limitX, by - 10); ctx.lineTo(limitX, by + 46); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = T.text; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText(st.card + ' GB card', limitX + 5, by - 12);
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle';
            parts.forEach((p, i) => {
              const yy = by + 70 + i * 24;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.fillText(p[1].toFixed(1) + ' GB', bx + bw, yy);
            });
            ctx.fillStyle = total <= st.card ? T.green : T.red; ctx.font = 'bold 14px ui-monospace, monospace';
            ctx.textAlign = 'left';
            ctx.fillText('total ' + total.toFixed(1) + ' GB per GPU — ' + (total <= st.card ? 'fits' : 'does NOT fit'), bx, by + 70 + 4 * 24 + 8);
            out({
              total: total.toFixed(1) + ' GB', fits: total <= st.card ? 'yes' : 'no',
              opt: optim.toFixed(1) + ' GB', act: acts.toFixed(1) + ' GB'
            });
          }
        });
        Viz.note(host, 'Optimizer state is the biggest single consumer at 12 bytes per parameter — bigger than the weights themselves. That is what ZeRO shards, and why an 8B model that "should" fit in 16 GB needs 128 GB to fine-tune fully (§4.13 turns that number into the case for LoRA).');
      },

      nan: function (host) {
        const st = Viz.controls(host, [
          { k: 'lr', label: 'learning rate', min: -4.5, max: -2.4, step: .05, value: -3.2, fmt: v => Math.pow(10, v).toExponential(1) },
          { k: 'clip', label: 'gradient clipping', type: 'toggle', value: false },
          { k: 'warm', label: 'warmup running', type: 'toggle', value: true },
          { k: 'badshard', label: 'a corrupt shard at step 320', type: 'toggle', value: true }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'die', label: 'run status', cls: 'key' }, { k: 'spike', label: 'gradient-norm spike at' },
          { k: 'nan', label: 'loss goes vertical at' }, { k: 'lead', label: 'warning lead time' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(23);
            const lr = Math.pow(10, st.lr);
            const loss = [], norm = [];
            let l = 11, dead = -1, spike = -1;
            for (let t = 0; t < 500; t++) {
              const warm = st.warm ? Math.min(1, t / 60) : 1;
              let g = 0.9 + R.normal(0, .12) + (lr * warm > 1.2e-3 ? (t / 500) * 2.2 : 0);
              if (st.badshard && t >= 320 && t < 324) g *= 14;
              if (st.clip) g = Math.min(g, 1.0);
              norm.push(g);
              if (dead < 0) {
                l = Math.max(1.6, l - lr * warm * 900 * (1 / (1 + t * .02))) + R.normal(0, .02) + (g > 3 ? g * .35 : 0);
                if (g > 3 && spike < 0) spike = t;
                if (g > 6 || (lr * warm > 2.2e-3 && t > 200 && R() < .02)) { dead = t + 2; }
              }
              loss.push(dead >= 0 && t >= dead ? NaN : l);
            }
            const P = Viz.plot(ctx, w, h, { xd: [0, 500], yd: [0, 12] })
              .frame({ xlabel: 'step', ylabel: 'loss (blue) · gradient global norm (red, ×2)' });
            P.clip(() => {
              P.line(loss.map((v, i) => [i, isFinite(v) ? v : 12]).filter((p, i) => isFinite(loss[i])), { color: T.blue, width: 2.4 });
              P.line(norm.map((v, i) => [i, Math.min(12, v * 2)]), { color: T.red, width: 1.4, alpha: .85 });
              if (spike >= 0) P.vline(spike, { color: T.amber, label: 'norm spikes here' });
              if (dead >= 0) P.vline(dead, { color: T.red, label: 'NaN' });
            });
            out({
              die: dead >= 0 ? 'died at step ' + dead : 'survived 500 steps',
              spike: spike >= 0 ? 'step ' + spike : '—',
              nan: dead >= 0 ? 'step ' + dead : '—',
              lead: (dead >= 0 && spike >= 0) ? (dead - spike) + ' steps' : '—'
            });
          }
        });
        Viz.note(host, 'Turn on the corrupt shard with clipping off: the red norm trace spikes at step 320 and the blue loss only goes vertical a couple of steps later. That gap is your diagnostic — print the batch at the spike, not at the NaN. Turn clipping on and the same shard passes through harmlessly.');
      }
    },
    quiz: [
      {
        q: 'Adam’s optimizer state costs roughly how much memory per parameter?',
        options: ['2 bytes', '4 bytes', '12 bytes', '32 bytes'],
        answer: 2,
        why: 'FP32 master weights plus two moments — about 12 bytes/param, more than the BF16 weights themselves. ZeRO/FSDP shards it.'
      },
      {
        q: 'Your loss NaNs after resuming from a checkpoint. The first thing to check is…',
        options: ['the model architecture', 'whether the scheduler is emitting the intended learning rate — a resume that skips warmup is the classic cause', 'the tokenizer', 'the evaluation set'],
        answer: 1,
        why: 'It is first on the ordered list precisely because it is both common and instantly checkable.'
      },
      {
        q: 'Tensor parallelism should be kept…',
        options: ['across data centres', 'inside a node, on NVLink', 'on the same GPU', 'between pipeline stages'],
        answer: 1,
        why: 'It communicates every layer; crossing a slower interconnect destroys throughput.'
      }
    ],
    cards: [
      { q: 'Three parallelisms', a: 'Data (all-reduce), tensor (chatty — keep in-node), pipeline (cheap, bubbles). Plus expert parallelism for MoE.' },
      { q: 'Adam memory', a: '≈12 bytes/param FP32 state; ZeRO/FSDP shards optimizer state, gradients and parameters.' },
      { q: 'NaN checklist order', a: 'LR/warmup → data → clipping (watch the norm spike) → mixed precision → scheduler after resume → batch size / bad shard.' },
      { q: 'Critical batch size', a: 'Below it doubling batch nearly halves steps; above it you buy little. It grows as loss falls — hence batch ramping.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.12 */
  ML.section({
    id: 'post-training', track: 'llm', num: '4.12',
    title: 'Post-training: SFT → RLHF → DPO → GRPO → RLVR',
    lede: 'Rests on §1.10 (KL) and §1.5 (likelihood). The DPO derivation is the highest-value derivation in Part 4.',
    html: `
<h2><span class="sn">4.12.1</span> SFT</h2>
<p>Supervised next-token prediction on curated demonstrations. It teaches format and instruction-following — you get a model that answers rather than continues, and nothing more.</p>

<h2><span class="sn">4.12.2</span> RLHF, classically</h2>
<p>Collect human preference pairs, fit a reward model to them, then optimise the policy with PPO subject to a KL penalty against the reference policy. The KL term is not decoration: without it the policy drifts into degenerate text that the reward model happens to love — <b>reward hacking</b>. The pipeline works but is heavy: a reward model, a value network, a policy, and a reference, all in memory.</p>

<h2><span class="sn">4.12.3</span> DPO, derived</h2>
<p>The KL-constrained RLHF objective has a closed-form optimum,</p>
$$\\pi^*(y\\mid x) = \\frac{1}{Z(x)}\\pi_{\\text{ref}}(y\\mid x)\\exp\\!\\left(\\tfrac{1}{\\beta}r(x,y)\\right)$$
<p>which you can invert for the reward: $r(x,y) = \\beta\\log\\frac{\\pi(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)} + \\beta\\log Z(x)$. Substitute that into the Bradley–Terry preference model $P(y_w \\succ y_l) = \\sigma(r(x,y_w) - r(x,y_l))$, and because the reward appears only as a <i>difference</i>, the intractable partition function $Z(x)$ — which depends on $x$ alone — <mark>cancels exactly</mark>. What remains is a plain classification loss on preference pairs:</p>
$$\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log\\sigma\\!\\left(\\beta\\log\\frac{\\pi(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta\\log\\frac{\\pi(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$
<p>No reward model, no value network, no sampling loop.</p>

${H.lab('dpo', 'The DPO loss, and what β controls', 'The loss surface over the implicit reward margin, with β as the slider. Watch how β trades preference-fitting against staying near the reference policy — the same knob the KL penalty is in PPO, arriving by a different route.')}

<h3>DPO’s known failure</h3>
<p>Length bias is real and documented: Park et al. (<i>Disentangling Length from Quality in Direct Preference Optimization</i>, arXiv:2403.19159) show DPO exhibits significant length exploitation, and their length-regularized R-DPO improves length-corrected win rates by roughly 15–20 points. If you run DPO, <b>monitor mean response length as a first-class metric</b>.</p>

<h2><span class="sn">4.12.4</span> GRPO</h2>
<p>GRPO (DeepSeek) removes the <i>critic</i> instead of the reward model. Sample a group of $G$ responses to the same prompt, score them all, and use the group-relative advantage $A_i = (r_i - \\mathrm{mean}(r))/\\mathrm{std}(r)$ in place of a learned value baseline. <b>The group is the baseline.</b> Cheaper, more stable, and a natural fit when rewards are cheap to compute.</p>

${H.worked('worked GRPO advantage', `
<p>Sample $G=6$ answers to one maths prompt and check each against the known answer. Rewards: [1, 0, 1, 1, 0, 0]. Mean = 0.5; population standard deviation = 0.5.</p>
<p>Advantages $(r_i-\\bar r)/\\sigma$ = <b>[+1, −1, +1, +1, −1, −1]</b>: the three correct completions are reinforced, the three wrong ones suppressed, with no value network anywhere.</p>
<p>Now the failure case an interviewer will push on: if all six are correct, rewards are [1,1,1,1,1,1], $\\sigma = 0$, and every advantage is zero (or NaN) — <b>the prompt teaches nothing</b>. Same if all six fail. So GRPO's learning signal lives entirely on prompts of <i>intermediate</i> difficulty, which is why curriculum and difficulty filtering matter far more here than in ordinary RLHF, and why implementations drop degenerate groups rather than dividing by zero.</p>`)}

${H.lab('grpo', 'GRPO advantages, and the degenerate group', 'Set the rewards yourself. Watch the advantages, and watch what happens when the group is unanimous — the readout tells you exactly how much signal the prompt carries.')}

<h2><span class="sn">4.12.5</span> RLVR</h2>
<p>Reinforcement learning from <b>verifiable</b> rewards replaces the learned reward model with a rule-based checker: did the maths answer match, did the unit tests pass, did the program compile. There is nothing to hack, because the reward is ground truth. DeepSeek-R1 (arXiv:2501.12948) used a cold-start SFT phase followed by GRPO on verifiable rewards to reach o1-level reasoning, and long chains of thought <i>emerged</i> from the optimisation rather than being demonstrated. <mark>This — not RLHF — is how current reasoning models are trained.</mark> RLAIF and constitutional methods sit alongside, replacing human labels with model-generated preferences under a written constitution.</p>

<h2><span class="sn">4.12.6</span> The PPO objective, in the form people ask for</h2>
<p>With ratio $r_t = \\pi_\\theta(a_t|s_t)/\\pi_{\\text{old}}(a_t|s_t)$ and advantage $A_t$, PPO maximises</p>
$$\\mathbb{E}\\big[\\min(r_tA_t,\\; \\mathrm{clip}(r_t, 1-\\epsilon, 1+\\epsilon)A_t)\\big] - \\beta D_{KL}(\\pi_\\theta\\|\\pi_{\\text{ref}})$$
<p>The clip is the whole trick: once the new policy has moved more than $\\epsilon$ (typically 0.2) in a favourable direction, the gradient is switched off, so a single batch cannot take an enormous step off the reference policy. The KL term does a different job — <b>clipping bounds each update, the KL penalty bounds the total drift</b> from where you started.</p>

${H.lab('ppo', 'The PPO clip, drawn', 'The objective as a function of the probability ratio, for positive and negative advantage. The flat regions are where the gradient is switched off — and the asymmetry between them is deliberate.')}

<h2><span class="sn">4.12.7</span> Reward hacking, with examples</h2>
<p>A learned reward model is a proxy, and optimising a proxy hard enough always finds its seams: answers get longer because length correlated with quality in the labels (§4.12.3's documented DPO case); models hedge on everything because hedging was rarely marked wrong; they mirror the user's stated view because agreement was rated helpful; they format lavishly because raters liked headers. Verifiable rewards remove the proxy for maths and code, but not for taste — so the practical stance is: <b>verify what can be verified, keep a KL leash on the rest, and monitor length, refusal rate and sycophancy as first-class metrics</b> rather than discovering them in production.</p>

${H.probe([
      ['How does DPO get rid of the reward model?', 'Invert the closed-form optimal RLHF policy for the reward; in the Bradley–Terry difference the partition function cancels, leaving a logistic loss on pairs.'],
      ['GRPO vs PPO?', 'GRPO drops the value network and uses the sampled group’s mean and standard deviation as the baseline — cheaper and stable for verifiable-reward reasoning.'],
      ['Why the KL penalty in RLHF?', 'It keeps the policy near the reference so it cannot exploit the reward model’s blind spots.']
    ], 'Presenting DPO as strictly better than RLHF and forgetting its length inflation.')}`,
    labs: {
      dpo: function (host) {
        const st = Viz.controls(host, [
          { k: 'beta', label: 'β (KL strength)', min: .05, max: 2, step: .05, value: .1, fmt: v => v.toFixed(2) },
          { k: 'margin', label: 'current implicit reward margin', min: -4, max: 4, step: .1, value: 0, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'loss', label: 'DPO loss', cls: 'key' }, { k: 'grad', label: 'gradient magnitude' },
          { k: 'pwin', label: 'implied P(preferred wins)' }, { k: 'drift', label: 'β says' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const L = m => -Math.log(Num.sigmoid(st.beta * m));
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [0, 3] })
              .frame({ xlabel: 'log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)', ylabel: 'loss' });
            P.clip(() => {
              [0.05, 0.1, 0.5, 1.0].forEach(b => {
                P.fn(m => -Math.log(Num.sigmoid(b * m)), { color: Math.abs(b - st.beta) < .001 ? T.blue : T.faint, width: Math.abs(b - st.beta) < .001 ? 2.8 : 1.2, alpha: Math.abs(b - st.beta) < .001 ? 1 : .6 });
              });
              P.fn(L, { color: T.blue, width: 2.8 });
              P.vline(st.margin, { color: T.red, dash: [4, 4] });
              P.dots([[st.margin, L(st.margin)]], { r: 5, color: T.red, stroke: true });
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('grey curves: β = 0.05, 0.1, 0.5, 1.0', w - 16, 10);
            const g = st.beta * (1 - Num.sigmoid(st.beta * st.margin));
            out({
              loss: L(st.margin).toFixed(4), grad: g.toFixed(4),
              pwin: Num.sigmoid(st.beta * st.margin).toFixed(3),
              drift: st.beta < .15 ? 'stay close to the reference' : st.beta > .6 ? 'fit preferences hard' : 'balanced'
            });
          }
        });
        Viz.note(host, 'Small β makes the loss nearly flat: the policy is only weakly pushed away from the reference, which is precisely the KL constraint arriving through a different door. Large β fits the preference data aggressively and is where length exploitation and other reward-hacking behaviours appear fastest.');
      },

      grpo: function (host) {
        const st = Viz.controls(host, [
          { k: 'r1', label: 'reward · answer 1', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r2', label: 'answer 2', min: 0, max: 1, step: 1, value: 0, fmt: v => v },
          { k: 'r3', label: 'answer 3', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r4', label: 'answer 4', min: 0, max: 1, step: 1, value: 1, fmt: v => v },
          { k: 'r5', label: 'answer 5', min: 0, max: 1, step: 1, value: 0, fmt: v => v },
          { k: 'r6', label: 'answer 6', min: 0, max: 1, step: 1, value: 0, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mean', label: 'group mean', cls: 'key' }, { k: 'sd', label: 'group σ' },
          { k: 'signal', label: 'learning signal', cls: 'good' }, { k: 'note', label: 'status' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const r = [st.r1, st.r2, st.r3, st.r4, st.r5, st.r6];
            const mean = Num.mean(r), sd = Math.sqrt(Num.mean(r.map(v => (v - mean) ** 2)));
            const A = r.map(v => sd > 1e-9 ? (v - mean) / sd : 0);
            const bx = 140, bw = w - bx - 80, mid = bx + bw / 2;
            ctx.font = '12px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.strokeStyle = T.line; ctx.beginPath(); ctx.moveTo(mid, 26); ctx.lineTo(mid, 26 + 6 * 30); ctx.stroke();
            r.forEach((v, i) => {
              const y = 40 + i * 30;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText('answer ' + (i + 1) + '  r=' + v, bx - 10, y);
              const a = A[i];
              ctx.fillStyle = a > 0 ? T.green : a < 0 ? T.red : T.faint;
              const len = Math.min(bw / 2 - 4, Math.abs(a) * bw / 4);
              ctx.fillRect(a >= 0 ? mid : mid - len, y - 9, Math.max(2, len), 18);
              ctx.fillStyle = T.text; ctx.textAlign = a >= 0 ? 'left' : 'right';
              ctx.fillText('A = ' + (a >= 0 ? '+' : '') + a.toFixed(2), a >= 0 ? mid + len + 6 : mid - len - 6, y);
            });
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('baseline = the group mean', mid, 26 + 6 * 30 + 4);
            const degenerate = sd < 1e-9;
            ctx.fillStyle = degenerate ? T.red : T.green; ctx.font = '12px ui-sans-serif';
            ctx.fillText(degenerate ? 'σ = 0 — every advantage is zero: this prompt teaches nothing and should be dropped'
              : 'correct completions reinforced, incorrect suppressed — no value network anywhere', mid, 26 + 6 * 30 + 22);
            out({
              mean: mean.toFixed(3), sd: sd.toFixed(3),
              signal: degenerate ? 'none' : 'present',
              note: degenerate ? (mean > .5 ? 'too easy' : 'too hard') : 'intermediate difficulty ✓'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'The worked example [1,0,1,1,0,0]', primary: true, on: () => { st.$set('r1', 1); st.$set('r2', 0); st.$set('r3', 1); st.$set('r4', 1); st.$set('r5', 0); st.$set('r6', 0); S.redraw(); } },
          { label: 'All correct (degenerate)', on: () => { [1, 2, 3, 4, 5, 6].forEach(i => st.$set('r' + i, 1)); S.redraw(); } },
          { label: 'All wrong (degenerate)', on: () => { [1, 2, 3, 4, 5, 6].forEach(i => st.$set('r' + i, 0)); S.redraw(); } }
        ]);
      },

      ppo: function (host) {
        const st = Viz.controls(host, [
          { k: 'eps', label: 'clip ε', min: .05, max: .5, step: .01, value: .2, fmt: v => v.toFixed(2) },
          { k: 'adv', label: 'advantage A', min: -2, max: 2, step: .1, value: 1, fmt: v => v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'range', label: 'gradient active range', cls: 'key' }, { k: 'at', label: 'objective at ratio 1' }, { k: 'note', label: 'behaviour' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const A = st.adv, e = st.eps;
            const obj = r => Math.min(r * A, Math.max(1 - e, Math.min(1 + e, r)) * A);
            const P = Viz.plot(ctx, w, h, { xd: [0, 2.2], yd: [Math.min(-2.6, -Math.abs(A) * 1.4), Math.max(2.6, Math.abs(A) * 1.4)] })
              .frame({ xlabel: 'probability ratio r = π_θ / π_old', ylabel: 'clipped objective' });
            P.clip(() => {
              P.fn(r => r * A, { color: T.faint, width: 1.4, dash: [5, 4] });
              P.fn(obj, { color: T.blue, width: 3 });
              P.vline(1, { color: T.faint, dash: [3, 3] });
              P.vline(1 - e, { color: T.red, dash: [3, 3] });
              P.vline(1 + e, { color: T.red, dash: [3, 3] });
              ctx.fillStyle = 'rgba(200,80,70,.10)';
              ctx.fillRect(P.x(1 - e), P.pad.t, P.x(1 + e) - P.x(1 - e), P.ph);
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('shaded = trust region; outside it in the favourable direction the objective is flat and the gradient is zero', P.pad.l + 6, 10);
            out({
              range: A > 0 ? '[0, ' + (1 + e).toFixed(2) + ']' : '[' + (1 - e).toFixed(2) + ', ∞)',
              at: (1 * A).toFixed(2),
              note: A > 0 ? 'improvement capped at 1+ε' : 'punishment capped at 1−ε'
            });
          }
        });
        Viz.note(host, 'Flip the advantage negative and the flat region moves to the other side. The asymmetry is the point: PPO will always let you move <i>away</i> from a bad action, but caps how far one batch can push you toward a good one — which is what stops a single lucky batch destroying the policy.');
      }
    },
    quiz: [
      {
        q: 'In the DPO derivation, the partition function $Z(x)$ cancels because…',
        options: ['it is approximated by sampling', 'the reward enters only as a difference between two completions of the same prompt', 'it is set to 1 by construction', 'the reference policy is frozen'],
        answer: 1,
        why: 'Z(x) depends on x alone, so in the Bradley–Terry difference $r(x,y_w)-r(x,y_l)$ it cancels exactly — no reward model needed.'
      },
      {
        q: 'GRPO rewards for one prompt are [1,1,1,1,1,1]. What is the learning signal?',
        options: ['Strongly positive', 'Zero — σ = 0, so every advantage is zero and the prompt teaches nothing', 'Negative', 'Undefined but usable'],
        answer: 1,
        why: 'The group is the baseline; a unanimous group has no relative information. Implementations drop degenerate groups and curate for intermediate difficulty.'
      },
      {
        q: 'The most commonly cited failure of DPO is…',
        options: ['reward-model overfitting', 'response length inflation (arXiv:2403.19159)', 'catastrophic forgetting', 'mode collapse to one answer'],
        answer: 1,
        why: 'Length exploitation is documented and material; monitor mean response length as a first-class metric.'
      }
    ],
    cards: [
      { q: 'DPO loss', a: '$-\\log\\sigma(\\beta\\log\\frac{\\pi(y_w)}{\\pi_{ref}(y_w)}-\\beta\\log\\frac{\\pi(y_l)}{\\pi_{ref}(y_l)})$ — no reward model; Z(x) cancels.' },
      { q: 'GRPO advantage', a: '$A_i=(r_i-\\mathrm{mean}(r))/\\mathrm{std}(r)$ over a sampled group; the group is the baseline, no critic.' },
      { q: 'RLVR', a: 'Rule-based verifiable rewards (tests pass, answer matches) — nothing to hack; how reasoning models are actually trained.' },
      { q: 'PPO clip vs KL', a: 'Clipping bounds each update; the KL penalty bounds total drift from the reference.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.13 */
  ML.section({
    id: 'lora', track: 'llm', num: '4.13',
    title: 'PEFT: LoRA and QLoRA',
    lede: 'The clearest case of §1.8’s low-rank idea paying rent — and a memory calculation that explains why parameter-efficient fine-tuning won.',
    html: `
<h2><span class="sn">4.13.1</span> LoRA</h2>
<p>Freeze the pretrained $W_0$ and learn a low-rank update $\\Delta W = BA$ with $B \\in \\mathbb{R}^{d\\times r}$, $A \\in \\mathbb{R}^{r\\times d}$ and $r \\ll d$. The forward pass becomes $W_0x + \\tfrac{\\alpha}{r}BAx$. You train ~0.1–1% of the parameters, checkpoints are megabytes rather than gigabytes, adapters hot-swap at serve time, and there is no added latency once merged. The premise is that task adaptation lives in a low-rank subspace — the same Eckart–Young intuition as §1.8.</p>

<h2><span class="sn">4.13.2</span> QLoRA</h2>
<p>Keep the frozen base in 4-bit NF4 and train the adapter in BF16, with double quantization and paged optimizers. Gradients flow <i>through</i> the quantized weights but never update them, which is what puts a 70B fine-tune on a single large GPU.</p>

<h3>2026 defaults worth quoting</h3>
<p><b>$r = 16$ with $\\alpha = 2r$, targeting <i>all</i> linear layers</b> — attention q, k, v, o plus the MLP gate, up and down projections — rising to $r = 32$–$64$ for harder tasks or larger data. Targeting attention only is the older, weaker recipe. <b>DoRA</b> (decomposing magnitude from direction) is a common quality upgrade at the same budget. Prefix- and prompt-tuning still exist but are largely superseded.</p>

${H.worked('worked number — what an 8B QLoRA actually costs', `
<p>Take the 8B model counted in §4.5, $r=16$, $\\alpha=32$, all seven linear projections per layer (q, k, v, o, gate, up, down), 32 layers.</p>
<p><b>Adapter parameters.</b> Each target of shape $m\\times n$ contributes $r(m+n)$. Per layer roughly $16\\times(4096{+}4096)\\times2$ for q/o, $16\\times(4096{+}1024)\\times2$ for k/v, and $16\\times(4096{+}14336)\\times3$ for the MLP ≈ 0.26M + 0.16M + 0.88M ≈ <b>1.3M</b>. Over 32 layers ≈ <b>42M trainable — 0.52% of the model.</b></p>
<p><b>Memory.</b> Frozen base in 4-bit NF4: 8.03B × 0.5 B ≈ <b>4.0 GB</b>. Adapter weights in BF16: 84 MB. Adam state for the adapter only (≈12 B/param): 0.5 GB. Gradients: 84 MB. Activations with checkpointing at 4k sequence and batch 4: a few GB.</p>
<p><b>Total ≈ 9–11 GB</b> — a single 16 GB card. Compare full fine-tuning at BF16: 16 GB of weights plus ~16 bytes/param of Adam state ≈ <b>128 GB</b> before activations. <mark>That factor of twelve, not the accuracy, is why PEFT won.</mark></p>`)}

${H.lab('lora', 'LoRA cost calculator', 'Rank, targets and precision, with the memory total and the full-fine-tune comparison alongside. The preset reproduces the worked example exactly.')}

${H.lab('rank', 'What rank actually buys', 'A real matrix, approximated at increasing rank via SVD, with the reconstruction error and the parameter count plotted together. The knee in the curve is the empirical justification for r = 16 — and you can watch it move when the underlying matrix is genuinely high-rank.')}

<h2><span class="sn">4.13.3</span> Serving adapters</h2>
<p>An adapter can be <b>merged</b> — compute $W_0 + \\tfrac{\\alpha}{r}BA$ once and ship a single weight matrix, giving zero added latency but losing hot-swappability; or <b>kept separate</b>, so one base model in memory serves dozens of tenant-specific adapters at a small per-request cost. Note that merging into a <i>quantized</i> base is lossy: you trained against the 4-bit weights, so merge into the BF16 original and re-quantize, or keep the adapter separate.</p>
${H.flag('And know when LoRA is the wrong tool: it adapts behaviour and format well, but it is a poor way to install substantial new knowledge — that is what continual pretraining (§4.9) or retrieval (§5.1) is for.')}

<h2><span class="sn">4.13.4</span> Model merging — cheaper than multi-task training</h2>
<p>Because fine-tuning moves weights a short distance, you can often <i>add</i> the movements. A <b>task vector</b> is $\\theta_{\\text{finetuned}} - \\theta_{\\text{base}}$; adding two task vectors to the base frequently yields a model competent at both, and subtracting one can remove a behaviour. Naive addition interferes, so refinements exist: <b>TIES</b> trims small-magnitude entries and resolves sign conflicts before merging, <b>DARE</b> randomly drops and rescales, and <b>SLERP</b> interpolates along the hypersphere between two checkpoints rather than in a straight line. Merging needs no data and no gradient steps, which makes it the fastest way to combine capabilities — and its limit is that it only works between models sharing a base that have not drifted far apart.</p>

${H.probe([
      ['What does the rank control?', 'Adapter capacity. Too low underfits the task; too high spends memory for nothing — find the knee empirically, starting at 16.'],
      ['The QLoRA trick?', '4-bit frozen base, 16-bit adapter; gradients pass through the quantized weights without updating them.'],
      ['When is LoRA the wrong tool?', 'When you need new knowledge rather than new behaviour — use continual pretraining or retrieval.']
    ])}`,
    labs: {
      lora: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'base model (B params)', min: .5, max: 180, step: .5, value: 8, fmt: v => v.toFixed(1) + 'B' },
          { k: 'd', label: 'hidden size d', min: 512, max: 16384, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'L', label: 'layers', min: 8, max: 128, step: 1, value: 32, fmt: v => v },
          { k: 'ffn', label: 'FFN width', min: 1024, max: 65536, step: 256, value: 14336, fmt: v => v.toLocaleString() },
          { k: 'r', label: 'rank r', min: 1, max: 128, step: 1, value: 16, fmt: v => v },
          { k: 'targets', label: 'targets', type: 'buttons', value: 'all', options: [{ v: 'attn', t: 'attention only' }, { v: 'all', t: 'all linear layers' }] },
          { k: 'quant', label: 'base precision', type: 'buttons', value: '0.5', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }, { v: '0.5', t: '4-bit NF4' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'trainable', label: 'trainable parameters', cls: 'key' }, { k: 'pct', label: 'share of the model' },
          { k: 'mem', label: 'total memory', cls: 'good' }, { k: 'full', label: 'full fine-tune would need', cls: 'bad' }, { k: 'ckpt', label: 'checkpoint size' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const d = st.d, r = st.r, kvW = Math.round(d / 4);
            const perLayerAttn = r * (d + d) * 2 + r * (d + kvW) * 2;
            const perLayerMlp = r * (d + st.ffn) * 3;
            const perLayer = st.targets === 'all' ? perLayerAttn + perLayerMlp : perLayerAttn;
            const trainable = perLayer * st.L;
            const N = st.N * 1e9;
            const baseGB = N * (+st.quant) / 1e9;
            const adapterGB = trainable * 2 / 1e9;
            const optGB = trainable * 12 / 1e9;
            const gradGB = trainable * 2 / 1e9;
            const actGB = 2.5;
            const total = baseGB + adapterGB + optGB + gradGB + actGB;
            const fullFT = N * 2 / 1e9 + N * 16 / 1e9;
            const parts = [
              ['frozen base (' + (st.quant === '0.5' ? '4-bit NF4' : st.quant === '1' ? 'FP8' : 'BF16') + ')', baseGB, T.faint],
              ['adapter weights', adapterGB, T.blue],
              ['adapter optimizer state', optGB, T.red],
              ['gradients', gradGB, T.amber],
              ['activations (checkpointed)', actGB, T.green]
            ];
            const bx = 20, bw = w - 40, by = 34;
            let x = bx;
            parts.forEach(p => { const pw = bw * p[1] / Math.max(total, fullFT * .35); ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 30); x += pw; });
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle';
            parts.forEach((p, i) => {
              const yy = by + 56 + i * 22;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.fillText(p[1].toFixed(2) + ' GB', bx + bw, yy);
            });
            ctx.fillStyle = T.green; ctx.font = 'bold 14px ui-monospace, monospace'; ctx.textAlign = 'left';
            ctx.fillText('PEFT total ≈ ' + total.toFixed(1) + ' GB', bx, by + 56 + 5 * 22 + 8);
            ctx.fillStyle = T.red;
            ctx.fillText('full fine-tune ≈ ' + fullFT.toFixed(0) + ' GB', bx + 220, by + 56 + 5 * 22 + 8);
            out({
              trainable: (trainable / 1e6).toFixed(1) + 'M', pct: (100 * trainable / N).toFixed(2) + '%',
              mem: total.toFixed(1) + ' GB', full: fullFT.toFixed(0) + ' GB',
              ckpt: (trainable * 2 / 1e6).toFixed(0) + ' MB'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'The worked 8B QLoRA', primary: true, on: () => { st.$set('N', 8); st.$set('d', 4096); st.$set('L', 32); st.$set('ffn', 14336); st.$set('r', 16); st.$set('targets', 'all'); st.$set('quant', '0.5'); S.redraw(); } },
          { label: '70B QLoRA', on: () => { st.$set('N', 70); st.$set('d', 8192); st.$set('L', 80); st.$set('ffn', 28672); S.redraw(); } }
        ]);
      },

      rank: function (host) {
        const N = 40;
        const st = Viz.controls(host, [
          { k: 'r', label: 'rank r', min: 1, max: 30, step: 1, value: 8, fmt: v => v },
          { k: 'struct', label: 'underlying structure', type: 'buttons', value: 'low', options: [{ v: 'low', t: 'genuinely low-rank' }, { v: 'mid', t: 'mixed' }, { v: 'high', t: 'full-rank noise' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'err', label: 'reconstruction error', cls: 'key' }, { k: 'energy', label: 'energy captured' },
          { k: 'params', label: 'parameters stored' }, { k: 'save', label: 'saving vs full matrix' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(29);
            const M = Array.from({ length: N }, () => new Array(N).fill(0));
            if (st.struct === 'high') {
              for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] = R.normal(0, 1);
            } else {
              const k = st.struct === 'low' ? 3 : 10;
              for (let c = 0; c < k; c++) {
                const u = Array.from({ length: N }, () => R.normal(0, 1)), v = Array.from({ length: N }, () => R.normal(0, 1));
                const s = 1 / (c + 1);
                for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] += s * u[i] * v[j];
              }
              if (st.struct === 'mid') for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] += R.normal(0, .12);
            }
            const { U, s, V } = Num.svd(M);
            const total = s.reduce((a, b) => a + b * b, 0);
            const errs = [];
            for (let r = 1; r <= 30; r++) {
              let kept = 0; for (let k = 0; k < r; k++) kept += s[k] * s[k];
              errs.push([r, Math.sqrt(Math.max(0, 1 - kept / total))]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, 30], yd: [0, 1] })
              .frame({ xlabel: 'rank r kept', ylabel: 'relative reconstruction error' });
            P.clip(() => {
              P.line(errs, { color: T.blue, width: 2.6 });
              P.dots(errs, { r: 2.6, color: T.blue });
              P.vline(st.r, { color: T.red, dash: [4, 4] });
              P.dots([errs[st.r - 1]], { r: 5.5, color: T.red, stroke: true });
              P.hline(.1, { color: T.green, dash: [3, 3], label: '10% error' });
            });
            let kept = 0; for (let k = 0; k < st.r; k++) kept += s[k] * s[k];
            out({
              err: (errs[st.r - 1][1] * 100).toFixed(1) + '%',
              energy: (100 * kept / total).toFixed(1) + '%',
              params: (2 * N * st.r).toLocaleString(),
              save: (100 * (1 - 2 * N * st.r / (N * N))).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'On a genuinely low-rank matrix, r = 4 already captures nearly everything and higher ranks buy nothing — that is the LoRA bet. On full-rank noise the error falls linearly and no small rank helps, which is exactly the case where LoRA underperforms and you need full fine-tuning or continual pretraining.');
      }
    },
    quiz: [
      {
        q: 'An 8B QLoRA at r=16 on all linear layers trains roughly…',
        options: ['0.05% of parameters', '0.5% of parameters', '5% of parameters', '20% of parameters'],
        answer: 1,
        why: '≈42M of 8.03B ≈ 0.52%, fitting in about 9–11 GB against ~128 GB for full BF16 fine-tuning.'
      },
      {
        q: 'Merging a LoRA adapter into a 4-bit quantized base is…',
        options: ['exact', 'lossy — you trained against the quantized weights, so merge into the BF16 original and re-quantize', 'impossible', 'faster than keeping it separate but identical in quality'],
        answer: 1,
        why: 'Or keep the adapter separate, which also preserves hot-swapping across tenants.'
      },
      {
        q: 'A task vector is…',
        options: ['the gradient at convergence', '$\\theta_{finetuned} - \\theta_{base}$, which can be added to or subtracted from the base', 'the LoRA B matrix', 'the embedding of the task description'],
        answer: 1,
        why: 'Adding task vectors combines capabilities with no data and no gradient steps; TIES/DARE/SLERP reduce interference.'
      }
    ],
    cards: [
      { q: 'LoRA', a: 'Freeze $W_0$, learn $\\Delta W=BA$ with $r\\ll d$; forward $W_0x+\\frac{\\alpha}{r}BAx$. Defaults r=16, α=2r, all linear layers.' },
      { q: 'QLoRA', a: '4-bit NF4 frozen base + BF16 adapter; gradients flow through the quantized weights without updating them.' },
      { q: 'The 8B QLoRA numbers', a: '≈42M trainable (0.52%), ≈9–11 GB total, vs ≈128 GB for full BF16 fine-tuning.' },
      { q: 'Model merging', a: 'Add task vectors ($\\theta_{ft}-\\theta_{base}$); TIES trims and resolves signs, DARE drops and rescales, SLERP interpolates on the sphere.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.14 */
  ML.section({
    id: 'serving', track: 'llm', num: '4.14',
    title: 'Inference and serving',
    lede: 'The most practically examined section in Part 4. Know the arithmetic, not the vibes.',
    html: `
<h2><span class="sn">4.14.1</span> Two phases with opposite characters</h2>
<p><b>Prefill</b> processes the whole prompt in parallel: large matrix multiplies, compute-bound, excellent GPU utilisation, and it determines <b>time to first token</b>. <b>Decode</b> emits one token at a time; each step re-reads all weights and the whole KV cache from HBM to produce a single token, so it is memory-bandwidth-bound and determines <b>tokens per second</b>. Batching helps decode enormously (weights are read once for the whole batch) and barely helps prefill.</p>

${H.worked('worked latency — the two numbers a user feels', `
<p><b>TTFT is prefill.</b> A 2,000-token prompt on a 70B model costs about $2N\\cdot s = 2\\times70\\times10^9\\times2000 = 2.8\\times10^{14}$ FLOP. At 990 TFLOP/s peak and a realistic 40% utilisation: $2.8\\times10^{14} \\div (0.4\\times990\\times10^{12}) \\approx$ <b>0.7 s</b>. Double the prompt, double the wait — TTFT is linear in prompt length, which is the real cost of "just paste everything" (§5.2).</p>
<p><b>TPOT is decode</b> — 42 ms at batch 1 from §4.7's bandwidth argument, so ~24 tok/s, about 18 words per second: faster than reading. Batch 32 keeps TPOT roughly flat while multiplying throughput 32×, which is why a serving stack optimises <i>throughput per GPU</i> and a chat UX optimises TTFT, and why those two goals fight.</p>
<p><b>Speculative decoding, quantified.</b> With draft acceptance probability $\\alpha$ and $k$ proposed tokens per verification step, expected accepted tokens per big-model pass is $(1-\\alpha^{k+1})/(1-\\alpha)$. At $\\alpha=0.7$, $k=4$: $(1-0.7^5)/0.3 = (1-0.168)/0.3 \\approx$ <b>2.8 tokens per pass</b> — a ~2.8× speedup, with the output distribution provably unchanged. The failure mode: on unpredictable text $\\alpha$ collapses and you pay for the draft model with no benefit, so measure acceptance in production rather than assuming it.</p>`)}

${H.lab('latency', 'TTFT, TPOT and speculative decoding', 'All three calculations, live. Move prompt length, batch size and draft acceptance and watch which number the user actually feels change.')}

<h2><span class="sn">4.14.2</span> Serving systems</h2>
<p><b>Paged attention</b> (vLLM) stores the KV cache in fixed-size pages with a block table, exactly like OS virtual memory: fragmentation disappears, sequences can share prefix pages, and achievable batch size rises sharply. <b>Continuous batching</b> admits and retires requests mid-flight instead of waiting for a whole batch to finish. <b>Prefix caching</b> reuses the prefill of a shared system prompt across requests — often the single cheapest latency win available.</p>

<h2><span class="sn">4.14.3</span> Quantization, by family</h2>
${H.table(['Method', 'What it is', 'Where it belongs'], [
      ['<b>GPTQ</b>', 'Post-training 4-bit weight-only, second-order error compensation', 'GPU serving; largely superseded by AWQ'],
      ['<b>AWQ</b>', 'Activation-aware: protects salient channels', 'The production pick on GPUs via vLLM'],
      ['<b>GGUF</b>', 'llama.cpp’s format and quant family', 'Local and workstation inference, not high-throughput serving'],
      ['<b>FP8</b>', 'Hardware-supported 8-bit on Hopper/Blackwell', 'Near-BF16 quality; the default for large-scale serving'],
      ['<b>KV-cache quantization</b>', 'Quantize the cache, not the weights', 'Halves the 10 GB cache independently of what you did to the weights']
    ])}
<p>Keep <b>weight</b> quantization distinct from <b>KV-cache</b> quantization — they are separate decisions with separate costs. And keep <i>post-training</i> quantization (GPTQ, AWQ, FP8 casting: minutes to hours, a small calibration set, where almost all production quantization happens) distinct from <i>quantization-aware training</i> (simulate low precision during training so weights adapt — better at aggressive sub-4-bit widths, far more expensive). If asked how to run a model in INT4 with minimal loss: AWQ plus a calibration set drawn from <i>your</i> distribution, and an eval to prove it.</p>

<h2><span class="sn">4.14.4</span> What to turn on, in order</h2>
${H.steps([
      '<b>Prefix caching</b> — a shared system prompt is prefilled once instead of per request; on agent workloads with long fixed instructions this is often a 2–5× TTFT win for a configuration flag.',
      '<b>Continuous batching</b>, then <b>paged attention</b> — both defaults in a modern server.',
      '<b>FP8 or INT4 weights</b>, measured against your own eval rather than a leaderboard, because quantization damage is task-dependent and shows up first on long-tail reasoning.',
      '<b>KV-cache quantization</b> if context is long.',
      '<b>Speculative decoding</b> last: the most operationally fiddly and the most workload-sensitive.'
    ])}

<h3>Speculative decoding without a second model</h3>
<p>The draft does not have to be a separate network. <b>Medusa</b> adds extra prediction heads to the model itself so it proposes several future tokens in one pass; <b>EAGLE</b> drafts in feature space rather than token space and achieves higher acceptance; <b>prompt-lookup (n-gram) decoding</b> simply copies candidate continuations from the prompt, which costs nothing and works remarkably well on summarisation, code editing and RAG answers — anywhere the output quotes the input. Try prompt-lookup first: it is a few lines and needs no training. <b>Multi-token prediction</b> attacks the same problem at training time by adding heads that predict $t+2, t+3, \\ldots$, which both improves the base model (a denser learning signal per position) and leaves you with a built-in speculative drafter; DeepSeek V3 is the documented production example.</p>

${H.probe([
      ['Why is decode memory-bound?', 'Every step streams all weights plus the whole KV cache from HBM for one token’s worth of arithmetic.'],
      ['Which quantization in production?', 'AWQ INT4 or FP8 on GPU; GGUF only for local. Quantize the KV cache separately.'],
      ['Size the cache for 70B, 4k, batch 8.', '10 GB — and be able to show the multiplication.']
    ])}`,
    labs: {
      latency: function (host) {
        const st = Viz.controls(host, [
          { k: 'N', label: 'model size (B)', min: 1, max: 700, step: 1, value: 70, fmt: v => v + 'B' },
          { k: 'prompt', label: 'prompt tokens', min: 100, max: 128000, step: 100, value: 2000, fmt: v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v },
          { k: 'batch', label: 'batch size', min: 1, max: 64, step: 1, value: 1, fmt: v => v },
          { k: 'alpha', label: 'speculative acceptance α', min: 0, max: .95, step: .01, value: .7, fmt: v => v.toFixed(2) },
          { k: 'k', label: 'draft tokens k', min: 1, max: 8, step: 1, value: 4, fmt: v => v },
          { k: 'util', label: 'prefill utilisation', min: .1, max: .8, step: .05, value: .4, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ttft', label: 'time to first token', cls: 'key' }, { k: 'tpot', label: 'time per output token' },
          { k: 'tps', label: 'tokens/s (this stream)' }, { k: 'spec', label: 'tokens per verification pass', cls: 'good' }, { k: 'speed', label: 'speculative speedup' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const N = st.N * 1e9, peak = 990e12, bw = 3.35e12;
            const ttft = (2 * N * st.prompt) / (st.util * peak);
            const tpot = (N * 2) / bw;
            const accepted = (1 - Math.pow(st.alpha, st.k + 1)) / (1 - st.alpha);
            const P = Viz.plot(ctx, w, h, { xd: [100, 128000], yd: [0, Math.max(2, (2 * N * 128000) / (st.util * peak)) * 1.05] })
              .frame({ xlabel: 'prompt tokens', ylabel: 'time to first token (s)', xfmt: v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(0) });
            P.clip(() => {
              P.fn(p => (2 * N * p) / (st.util * peak), { color: T.blue, width: 2.6 });
              P.vline(st.prompt, { color: T.red, dash: [4, 4] });
              P.dots([[st.prompt, ttft]], { r: 5, color: T.red, stroke: true });
              P.hline(1, { color: T.amber, dash: [3, 3], label: '1 second — the perceptual threshold' });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
            ctx.fillText('TTFT is linear in prompt length — the real cost of "paste everything"', w - 16, 8);
            out({
              ttft: ttft < 1 ? (ttft * 1000).toFixed(0) + ' ms' : ttft.toFixed(2) + ' s',
              tpot: (tpot * 1000).toFixed(1) + ' ms',
              tps: (1 / tpot).toFixed(0),
              spec: accepted.toFixed(2),
              speed: '×' + accepted.toFixed(2)
            });
          }
        });
        Viz.note(host, 'At α = 0.7 and k = 4 the expected accepted tokens per pass is 2.8 — matching the worked box. Drag α down to 0.3 (unpredictable text) and the speedup nearly vanishes while you still pay for the draft model: measure acceptance, never assume it.');
      }
    },
    quiz: [
      {
        q: 'Time to first token is determined by…',
        options: ['decode bandwidth', 'prefill compute, and it is linear in prompt length', 'the KV cache size', 'the tokenizer'],
        answer: 1,
        why: '≈2N·s FLOPs at your achieved utilisation. Doubling the prompt doubles the wait.'
      },
      {
        q: 'Speculative decoding with α = 0.7 and k = 4 yields roughly how many tokens per big-model pass?',
        options: ['1.4', '2.8', '4.0', '5.0'],
        answer: 1,
        why: '$(1-0.7^5)/0.3 \\approx 2.8$, with the output distribution provably unchanged.'
      },
      {
        q: 'The cheapest serving win to turn on first is usually…',
        options: ['speculative decoding', 'prefix caching of a shared system prompt', 'INT4 quantization', 'a bigger batch'],
        answer: 1,
        why: 'One flag, no quality risk, and on agent workloads with long fixed instructions it is frequently a 2–5× TTFT improvement.'
      }
    ],
    cards: [
      { q: 'Prefill vs decode', a: 'Prefill: parallel, compute-bound, sets TTFT. Decode: one token at a time, bandwidth-bound, sets tokens/s. Batching helps decode.' },
      { q: 'Speculative decoding formula', a: 'Expected accepted tokens $=(1-\\alpha^{k+1})/(1-\\alpha)$; α=0.7, k=4 → 2.8.' },
      { q: 'Serving order', a: 'Prefix caching → continuous batching + paged attention → FP8/INT4 weights → KV quantization → speculative decoding last.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.15 */
  ML.section({
    id: 'decoding', track: 'llm', num: '4.15',
    title: 'Decoding strategies, test-time compute, prompting',
    lede: 'Same distribution, many ways through it — and the axis of scaling that does not require training anything.',
    html: `
<h2><span class="sn">4.15.1</span> The policies</h2>
${H.table(['Strategy', 'Rule', 'Use for'], [
      ['<b>Greedy</b>', 'take the argmax', 'Extraction, classification, anything deterministic'],
      ['<b>Beam search</b>', 'keep the top-b partial sequences', 'Translation; produces bland text in open generation'],
      ['<b>Temperature</b>', 'divide logits by T before softmax', 'T>1 flattens, T<1 sharpens — reshapes before any policy chooses'],
      ['<b>Top-k</b>', 'truncate to the k most likely', 'Simple, but a fixed k is wrong when the distribution is peaked'],
      ['<b>Top-p (nucleus)</b>', 'smallest set whose mass exceeds p', 'Adapts to the shape of the distribution — the common default'],
      ['<b>Min-p</b>', 'threshold relative to the mode', 'Robust at high temperature'],
      ['<b>Repetition / presence penalties</b>', 'down-weight seen tokens', 'Suppressing loops']
    ])}

${H.lab('sample', 'A real language model, sampled live', 'This trains a character-level n-gram model in your browser on the text in the box, then generates from it under the decoding policy you choose. Temperature, top-k and top-p are doing exactly what they do in a frontier model — the model is tiny, the mechanism is identical.')}

<h2><span class="sn">4.15.2</span> Constrained decoding</h2>
<p>Mask the logits against a grammar or JSON Schema so the output is <b>valid by construction</b>. This is the correct answer to "how do you guarantee parseable output" — not "I ask nicely and retry" — and it is what makes tool calling reliable (§5.3).</p>

<h2><span class="sn">4.15.3</span> Test-time compute</h2>
<p>A scaling axis distinct from model size: spend more at inference — longer chains of thought, sample-and-vote (self-consistency), best-of-$n$ against a verifier, tree or beam search over reasoning steps — and accuracy rises. This is the mechanism behind reasoning models, and it is why §4.12's RLVR matters: training the model to <i>use</i> a long chain well is what makes the extra compute pay.</p>

${H.lab('ttc', 'Test-time compute: majority vote and best-of-n', 'Simulated but honest: with per-sample accuracy p and independent errors, watch how majority voting and verifier-checked best-of-n scale with n — and where each stops helping.')}

<h2><span class="sn">4.15.4</span> Prompting that generalises</h2>
<p>Unglamorous and effective: state the task and the output format explicitly, give two or three well-chosen exemplars, put instructions where the model attends (the start <i>and</i> the end of the context), and use explicit step-by-step reasoning only where it earns its tokens — on non-reasoning models it helps mathematics and hurts simple classification, and on reasoning models it is redundant with what the model already does internally.</p>

${H.probe([
      ['Top-k or top-p?', 'Top-p adapts to the distribution’s shape; a fixed k is wrong whenever the distribution is peaked or flat.'],
      ['How do you guarantee valid JSON?', 'Constrained decoding against a schema — mask the logits so invalid tokens cannot be sampled.'],
      ['What is test-time compute?', 'Trading inference compute for accuracy: longer CoT, self-consistency voting, best-of-n against a verifier, search over steps.']
    ])}`,
    labs: {
      sample: function (host) {
        const seed = 'the quick brown fox jumps over the lazy dog. the model predicts the next token from the previous tokens. attention is all you need for sequence modelling. the cat sat on the mat while the dog barked at the moon. machine learning models learn patterns from data and generalise to new data. ';
        let lm = Num.charLM(seed.repeat(6), 4);
        let outText = 'the ';
        const st = Viz.controls(host, [
          { k: 'temp', label: 'temperature T', min: .1, max: 2.5, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'topk', label: 'top-k (0 = off)', min: 0, max: 12, step: 1, value: 0, fmt: v => v || 'off' },
          { k: 'topp', label: 'top-p (1 = off)', min: .1, max: 1, step: .05, value: 1, fmt: v => v === 1 ? 'off' : v.toFixed(2) },
          { k: 'greedy', label: 'greedy (argmax)', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'cands', label: 'candidates after filtering', cls: 'key' }, { k: 'ent', label: 'entropy of the choice' },
          { k: 'top', label: 'most likely next char' }, { k: 'p', label: 'its probability' }
        ]);
        const textBox = ML.el('div', {
          style: 'font-family:var(--mono);font-size:13px;line-height:1.6;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-height:86px;margin:12px 0;white-space:pre-wrap'
        });
        host.appendChild(textBox);
        function nextDist() {
          const ctx2 = outText.slice(-lm.order);
          let d = lm.dist(ctx2);
          let logits = d.map(x => Math.log(Math.max(1e-9, x.p)) / st.temp);
          let probs = Num.softmax(logits);
          let items = d.map((x, i) => ({ ch: x.ch, p: probs[i] })).sort((a, b) => b.p - a.p);
          if (st.topk > 0) items = items.slice(0, st.topk);
          if (st.topp < 1) {
            const keep = []; let acc = 0;
            for (const it of items) { keep.push(it); acc += it.p; if (acc >= st.topp) break; }
            items = keep;
          }
          const z = items.reduce((a, b) => a + b.p, 0) || 1;
          items = items.map(it => ({ ch: it.ch, p: it.p / z }));
          return items;
        }
        function step(n) {
          for (let i = 0; i < n; i++) {
            const items = nextDist();
            let pick;
            if (st.greedy) pick = items[0];
            else {
              let u = Math.random(), acc = 0;
              pick = items[items.length - 1];
              for (const it of items) { acc += it.p; if (u <= acc) { pick = it; break; } }
            }
            outText += pick.ch;
            if (outText.length > 420) outText = outText.slice(-420);
          }
        }
        const S = Viz.surface(host, {
          height: 220,
          draw: function (ctx, w, h, T) {
            const items = nextDist().slice(0, 14);
            const P = Viz.plot(ctx, w, h, { xd: [-.5, Math.max(1, items.length) - .5], yd: [0, 1], pad: { l: 40, r: 14, t: 14, b: 40 } })
              .frame({ xticks: [], ylabel: 'probability' });
            P.clip(() => items.forEach((it, i) => {
              const x0 = P.x(i - .38), x1 = P.x(i + .38);
              ctx.fillStyle = i === 0 ? T.blue : T.faint;
              ctx.fillRect(x0, P.y(it.p), x1 - x0, P.y(0) - P.y(it.p));
              ctx.fillStyle = T.text; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(it.ch === ' ' ? '␣' : it.ch === '\n' ? '⏎' : it.ch, P.x(i), P.y(0) + 6);
            }));
            textBox.textContent = outText;
            out({
              cands: items.length, ent: Num.entropy(items.map(i => i.p)).toFixed(2),
              top: items[0] ? (items[0].ch === ' ' ? '␣' : items[0].ch) : '—',
              p: items[0] ? items[0].p.toFixed(3) : '—'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Generate 60 characters', primary: true, on: () => { step(60); S.redraw(); } },
          { label: 'Generate 1', on: () => { step(1); S.redraw(); } },
          { label: 'Reset', on: () => { outText = 'the '; S.redraw(); } }
        ]);
        Viz.note(host, 'Set temperature to 0.2 and generate: the text becomes repetitive and safe. Set it to 2.0 and it becomes noise. Turn on top-p 0.9 at high temperature and the tail is cut while the head keeps its diversity — which is exactly why nucleus sampling became the default.');
      },

      ttc: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'per-sample accuracy', min: .05, max: .95, step: .01, value: .45, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'verifier', label: 'verifier accuracy (best-of-n)', min: .5, max: 1, step: .01, value: .95, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'corr', label: 'error correlation between samples', min: 0, max: .9, step: .05, value: .2, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'v1', label: 'single sample', cls: 'key' }, { k: 'maj', label: 'majority vote @ 16' },
          { k: 'bon', label: 'best-of-16 with verifier', cls: 'good' }, { k: 'ceil', label: 'ceiling from correlation' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const p = st.p, c = st.corr;
            // effective independent samples shrinks with correlation
            const eff = n => 1 + (n - 1) * (1 - c);
            const majority = n => {
              const m = eff(n);
              let acc = 0;
              for (let k = Math.floor(m / 2) + 1; k <= Math.ceil(m); k++) acc += Num.binomPmf(Math.round(k), Math.round(m), p);
              return Math.min(1, Math.max(p, acc));
            };
            const bestOf = n => {
              const m = eff(n);
              const anyCorrect = 1 - Math.pow(1 - p, m);
              return anyCorrect * st.verifier + (1 - anyCorrect) * (1 - st.verifier) * .2;
            };
            const P = Viz.plot(ctx, w, h, { xd: [1, 64], yd: [0, 1] })
              .frame({ xlabel: 'samples drawn (test-time compute)', ylabel: 'task accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(n => p, { color: T.faint, width: 1.6, dash: [5, 4] });
              P.fn(majority, { color: T.blue, width: 2.6, n: 120 });
              P.fn(bestOf, { color: T.green, width: 2.6, n: 120 });
            });
            out({
              v1: (p * 100).toFixed(1) + '%', maj: (majority(16) * 100).toFixed(1) + '%',
              bon: (bestOf(16) * 100).toFixed(1) + '%',
              ceil: c > .5 ? 'severe — samples repeat the same error' : c > .2 ? 'moderate' : 'mild'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().faint, t: 'single sample' }, { c: Viz.theme().blue, t: 'majority vote (self-consistency)' }, { c: Viz.theme().green, t: 'best-of-n against a verifier' }]);
        Viz.note(host, 'Raise the error correlation and both curves flatten: if every sample makes the same mistake, drawing more of them buys nothing. That is the real limit of test-time compute, and the reason temperature and prompt diversity matter when you use it.');
      }
    },
    quiz: [
      {
        q: 'Top-p (nucleus) sampling is preferred to top-k because…',
        options: ['it is faster', 'it adapts the candidate set to how peaked the distribution is', 'it always produces longer text', 'it removes the need for temperature'],
        answer: 1,
        why: 'A fixed k either truncates a genuinely flat distribution or admits junk from a peaked one.'
      },
      {
        q: 'The correct way to guarantee schema-valid JSON output is…',
        options: ['prompt engineering and retries', 'constrained decoding that masks invalid tokens', 'temperature 0', 'a larger model'],
        answer: 1,
        why: 'Validity by construction — and it is what makes tool calling reliable (§5.3).'
      }
    ],
    cards: [
      { q: 'Temperature vs top-k vs top-p', a: 'Temperature reshapes the distribution; top-k truncates to a fixed count; top-p truncates to a mass, adapting to the shape.' },
      { q: 'Constrained decoding', a: 'Mask logits against a grammar/schema so output is valid by construction — the answer to "guarantee parseable output".' },
      { q: 'Test-time compute', a: 'Longer CoT, self-consistency voting, best-of-n with a verifier, search over steps. Limited by error correlation between samples.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.16 */
  ML.section({
    id: 'llm-eval', track: 'llm', num: '4.16',
    title: 'Evaluation, hallucination, multimodal',
    lede: 'Contamination first, then judges, then the eval you actually build yourself.',
    html: `
<h2><span class="sn">4.16.1</span> Contamination first</h2>
<p>Public benchmarks leak into pretraining corpora, so a headline score is evidence about the benchmark's <i>age</i> as much as the model's ability. Trust a private held-out set built from your own distribution, and check n-gram overlap against training data where you can.</p>

<h2><span class="sn">4.16.2</span> LLM-as-a-judge</h2>
<p>Cheap and correlates decently with human preference, but it has documented biases: <b>position</b> bias (it favours whichever answer is shown first), <b>verbosity</b> bias (it favours longer answers), and self-preference for its own family's outputs. Mitigations, in order of importance: <mark>score both orders and average — the mandatory position swap</mark>; control for length; prefer pairwise comparison to pointwise scoring; give the judge an explicit rubric; and keep a human review loop on a sample.</p>

${H.worked('worked number — is your judge trustworthy?', `
<p>You cannot take a judge's agreement rate at face value, because two raters who both prefer A most of the time agree often by chance. <b>Cohen's κ</b> corrects for that: $\\kappa = (p_o - p_e)/(1 - p_e)$.</p>
<p>On 100 pairwise comparisons the judge and a human agree 82 times, so $p_o = 0.82$. Both pick A about 70% of the time and B 30%, so chance agreement is $p_e = 0.7^2 + 0.3^2 = 0.58$. Then</p>
$$\\kappa = \\frac{0.82-0.58}{1-0.58} = \\frac{0.24}{0.42} = 0.57$$
<p>"82% agreement" sounds strong; κ = 0.57 is <i>moderate</i> — usable for ranking two systems in aggregate, not for adjudicating individual cases. And human–human κ on the same task is the ceiling you should be comparing against, which on genuinely subjective work is often only 0.6–0.7.</p>`)}

${H.lab('kappa', 'Cohen’s κ calculator, and the position-swap test', 'Move the agreement rate and the marginals; κ falls out. The second panel simulates position bias: two judgements of the same pair in both orders, and how often they disagree.')}

<h2><span class="sn">4.16.3</span> Hallucination</h2>
<p>Not a bug in the ordinary sense. The model is trained to produce plausible continuations, it has no mechanism for checking truth, and the decoding objective rewards fluency. Mitigations are therefore <b>structural</b>: ground with retrieval and require citations (§5.1), constrain decoding, train for abstention so "I don't know" is a reachable output, and verify claims externally where the cost of error is high.</p>

<h2><span class="sn">4.16.4</span> Multimodal, briefly</h2>
<p><b>ViT</b> cuts an image into patches and treats them as tokens — attention needs no pixel-specific machinery. <b>CLIP</b> trains an image encoder and a text encoder contrastively so matched pairs are close, which yields zero-shot classification and the embedding space multimodal retrieval uses. A <b>VLM</b> is a vision encoder plus a projector that maps visual features into the LLM's token space, so the language model can attend to image tokens as if they were words.</p>

<h2><span class="sn">4.16.5</span> How to build an eval worth having</h2>
${H.steps([
      'Start from <b>failures, not capabilities</b>: take 100–300 real inputs weighted toward cases that have actually gone wrong.',
      'Label the outcome you want. <b>Freeze it. Version it. Never let it into a prompt.</b>',
      'Add three layers: deterministic checks (does it parse, does it cite, is the number right), pairwise judge comparisons for anything subjective, and a small human-reviewed sample to keep the judge honest.',
      'Report <b>per-slice</b> results — an aggregate that moves from 84% to 86% while collapsing on the one slice that matters is a regression dressed as progress.'
    ])}

${H.probe([
      ['Why swap positions when using a judge?', 'Judges systematically favour the first-shown answer; averaging both orders removes that bias.'],
      ['A benchmark says 90% — do you trust it?', 'Only after ruling out contamination, and never in place of a private eval on your own distribution.'],
      ['Is hallucination fixable by prompting?', 'No — mitigations are structural: retrieval grounding, citations, constrained decoding, trained abstention, external verification.']
    ])}`,
    labs: {
      kappa: function (host) {
        const st = Viz.controls(host, [
          { k: 'agree', label: 'observed agreement p₀', min: .5, max: 1, step: .01, value: .82, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'pa', label: 'both raters pick A this often', min: .3, max: .95, step: .01, value: .7, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'bias', label: 'judge position bias', min: 0, max: .5, step: .01, value: .18, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kappa', label: 'Cohen’s κ', cls: 'key' }, { k: 'pe', label: 'chance agreement p_e' },
          { k: 'verdict', label: 'interpretation' }, { k: 'flip', label: 'verdict flips on swap' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const pe = st.pa * st.pa + (1 - st.pa) * (1 - st.pa);
            const kappa = (st.agree - pe) / (1 - pe);
            ctx.font = '14px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillStyle = T.muted; ctx.fillText('κ = (p₀ − p_e) / (1 − p_e)', 18, 18);
            ctx.fillStyle = T.text; ctx.font = 'bold 16px ui-monospace, monospace';
            ctx.fillText('= (' + st.agree.toFixed(2) + ' − ' + pe.toFixed(2) + ') / (1 − ' + pe.toFixed(2) + ') = ' + kappa.toFixed(3), 18, 42);
            // scale
            const bx = 18, bw = w - 36, by = 84;
            const bands = [[0, .2, 'slight', T.red], [.2, .4, 'fair', T.amber], [.4, .6, 'moderate', T.amber], [.6, .8, 'substantial', T.green], [.8, 1, 'almost perfect', T.green]];
            bands.forEach(b => {
              ctx.fillStyle = b[3]; ctx.globalAlpha = .25;
              ctx.fillRect(bx + bw * b[0], by, bw * (b[1] - b[0]) - 2, 26); ctx.globalAlpha = 1;
              ctx.fillStyle = T.muted; ctx.font = '9px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(b[2], bx + bw * (b[0] + b[1]) / 2, by + 30);
            });
            const kx = bx + bw * Math.max(0, Math.min(1, kappa));
            ctx.strokeStyle = T.text; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(kx, by - 8); ctx.lineTo(kx, by + 30); ctx.stroke();
            ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
            ctx.fillText(kappa.toFixed(2), kx, by - 10);
            // position swap simulation
            const R = Num.rng(13);
            let flips = 0;
            for (let i = 0; i < 200; i++) {
              const trueBetter = R() < .5 ? 'A' : 'B';
              const j1 = R() < .5 + st.bias ? 'first' : 'second';
              const j2 = R() < .5 + st.bias ? 'first' : 'second';
              const v1 = j1 === 'first' ? 'A' : 'B';
              const v2 = j2 === 'first' ? 'B' : 'A';
              if (v1 !== v2) flips++;
            }
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('Position-swap check on 200 simulated pairs: the same comparison judged in both orders', bx, by + 62);
            ctx.fillStyle = flips / 200 > .25 ? T.red : T.green; ctx.font = 'bold 13px ui-monospace, monospace';
            ctx.fillText((flips / 2).toFixed(0) + '% of pairs flip their verdict when the order is swapped — those are ties, not verdicts', bx, by + 84);
            out({
              kappa: kappa.toFixed(3), pe: pe.toFixed(3),
              verdict: kappa < .2 ? 'slight' : kappa < .4 ? 'fair' : kappa < .6 ? 'moderate' : kappa < .8 ? 'substantial' : 'almost perfect',
              flip: (flips / 2).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'Set the defaults and you reproduce the worked example: 82% agreement, κ = 0.57 — moderate, not strong. High raw agreement with a skewed marginal is mostly chance, which is exactly what κ is for.');
      }
    },
    quiz: [
      {
        q: 'Judge and human agree on 82 of 100 comparisons; both pick A 70% of the time. κ is…',
        options: ['0.82', '0.72', '0.57', '0.24'],
        answer: 2,
        why: 'p_e = 0.7² + 0.3² = 0.58; κ = (0.82−0.58)/(1−0.58) = 0.57 — moderate.'
      },
      {
        q: 'The mandatory mitigation for LLM-judge position bias is…',
        options: ['using a bigger judge model', 'scoring both orders and averaging', 'raising the temperature', 'using pointwise scores'],
        answer: 1,
        why: 'If the winner changes with the order, you measured the judge rather than the answers.'
      },
      {
        q: 'Hallucination is best mitigated by…',
        options: ['a stricter system prompt', 'structural measures: retrieval grounding with citations, constrained decoding, trained abstention, external verification', 'lower temperature alone', 'a larger context window'],
        answer: 1,
        why: 'The model has no truth-checking mechanism; the fixes have to come from outside the decoding objective.'
      }
    ],
    cards: [
      { q: 'Cohen’s κ', a: '$(p_o-p_e)/(1-p_e)$; 82% agreement with 70/30 marginals gives κ = 0.57 — moderate.' },
      { q: 'Judge biases and fixes', a: 'Position (swap and average), verbosity (control length), self-preference (cross-family judge + human sample).' },
      { q: 'Building an eval', a: 'Start from real failures, 100–300 items, frozen and versioned, three layers (deterministic, pairwise judge, human sample), reported per slice.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.17 */
  ML.section({
    id: 'safety', track: 'llm', num: '4.17',
    title: 'Safety, red-teaming, and interpretability',
    lede: 'Increasingly asked even in applied roles, because anyone deploying a model owns these questions.',
    html: `
<h2><span class="sn">4.17.1</span> Jailbreaks, by mechanism</h2>
<p>Refusal is a learned behaviour, and every attack family finds a distribution where it was not learned.</p>
${H.table(['Family', 'Mechanism'], [
      ['<b>Persona / role-play</b>', '"You are an author writing a villain" — moves the request into a frame where compliance looked appropriate in training'],
      ['<b>Obfuscation</b>', 'base64, leetspeak, another language, a cipher — the safety-relevant features never fire while the capability still does'],
      ['<b>Many-shot</b>', 'Fill a long context with dozens of fabricated compliant exchanges, exploiting in-context learning against the model’s own policy — a direct cost of long context (§4.4)'],
      ['<b>Crescendo</b>', 'Escalate gradually across turns so no single turn looks refusable'],
      ['<b>Best-of-n</b>', 'Sample the same request many times at high temperature until one attempt slips through']
    ])}

${H.lab('refusal', 'Per-request refusal rates are the wrong unit', 'A 99% refusal rate against an attacker who can retry. The curve is $1-(1-\\epsilon)^n$ and it is unforgiving — this is the calculation to have ready when someone quotes a single-request safety number.')}

<h2><span class="sn">4.17.2</span> The two-sided metric</h2>
<p>Safety training that only minimises harmful compliance produces <b>over-refusal</b>: a model that declines to explain how to kill a process on Linux is unusable, and users route around it. So evaluate both axes — attack success rate on a red-team set <i>and</i> false-refusal rate on a benign-but-adjacent set — and treat movement in either as a regression. Red-teaming itself is now largely automated: an attacker model generates and mutates prompts against a judge (§4.16), which finds far more surface than manual work, with human effort reserved for the novel classes the automation cannot invent.</p>

<h2><span class="sn">4.17.3</span> Chain-of-thought faithfulness — the uncomfortable finding</h2>
<p>The reasoning a model prints is <b>not guaranteed to be the computation it performed</b>. Models can be shown to reach an answer via a cue in the prompt and then produce a plausible chain that never mentions it; they can also be steered to a wrong answer while the chain looks sound. Two consequences for practice: a legible chain is <i>evidence but not proof</i>, so do not use it as an audit trail for a regulated decision; and there is a real tension between training models to produce <i>nice-looking</i> reasoning and keeping that reasoning <i>informative</i> about the underlying process — optimising the visible chain too hard destroys its value as a monitor.</p>

<h2><span class="sn">4.17.4</span> Mechanistic interpretability, and what it actually buys</h2>
<p>Individual neurons are usually <b>polysemantic</b> — they fire for several unrelated concepts — because a network in <b>superposition</b> represents far more features than it has dimensions. <b>Sparse autoencoders</b> attack this by training a wide, sparsely-activating layer to reconstruct the model's activations; its units come out far more monosemantic, giving named, human-readable features ("legal disclaimer language", "the code is in Python"). With features in hand you can do three useful things: <b>audit</b> what a model attended to on a specific input, <b>steer</b> behaviour by amplifying or suppressing a feature at inference, and <b>detect</b> states like deception or refusal-suppression with a probe. The limits are equally worth stating: coverage is partial, features are not stable across training runs, and none of it yet constitutes a compliance-grade explanation — which is why §2.17's SHAP, for all its faults, is still what goes in the model-risk file.</p>

<h2><span class="sn">4.17.5</span> Two more terms to recognise</h2>
<p><b>Sandbagging</b> is a model performing worse when it detects it is being evaluated — which makes held-out, unfamiliar-format evals valuable for reasons beyond contamination. <b>Provenance</b> covers watermarking generated text (a statistical bias in token selection that a detector can find, robust to light editing and not to paraphrase) and content credentials such as C2PA for images; treat both as deterrents and forensic aids, never as reliable detectors, and never build a policy that depends on catching AI-generated text.</p>

${H.probe([
      ['Your model refuses 99% of attacks — is that safe?', 'Not against an attacker who can retry: 100 samples gives them a 63% success rate. Per-request rates are the wrong unit for adversarial settings.'],
      ['Can I use the chain of thought as an audit trail?', 'No — it can be unfaithful to the actual computation. Use it as a debugging aid, not as evidence.'],
      ['What do sparse autoencoders give you?', 'More monosemantic features, enabling auditing, steering and probe-based detection — with partial coverage and no compliance guarantee.']
    ], 'Reporting only harmfulness and never false-refusal. Half a metric hides the cost of your own fix.')}`,
    labs: {
      refusal: function (host) {
        const st = Viz.controls(host, [
          { k: 'rate', label: 'per-request refusal rate', min: .9, max: .9999, step: .0001, value: .99, fmt: v => (v * 100).toFixed(2) + '%' },
          { k: 'n', label: 'attacker attempts', min: 1, max: 1000, step: 1, value: 100, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'succ', label: 'attacker success probability', cls: 'bad' },
          { k: 'need', label: 'attempts for 50% success' },
          { k: 'target', label: 'rate needed for <1% at this n', cls: 'key' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const eps = 1 - st.rate;
            const f = n => 1 - Math.pow(1 - eps, n);
            const P = Viz.plot(ctx, w, h, { xd: [1, 1000], yd: [0, 1] })
              .frame({ xlabel: 'number of attempts', ylabel: 'probability at least one succeeds', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.fn(f, { color: T.red, width: 2.8, n: 400 });
              P.vline(st.n, { color: T.text, dash: [4, 4] });
              P.dots([[st.n, f(st.n)]], { r: 5.5, color: T.red, stroke: true });
              P.hline(.5, { color: T.faint, dash: [3, 3], label: '50%' });
            });
            out({
              succ: (f(st.n) * 100).toFixed(1) + '%',
              need: Math.ceil(Math.log(.5) / Math.log(1 - eps)),
              target: (100 * (1 - (1 - Math.pow(1 - .01, 1 / st.n)))).toFixed(4) + '%'
            });
          }
        });
        Viz.note(host, 'At 99% refusal and 100 attempts the attacker wins 63% of the time. To hold total risk under 1% across 100 attempts you need a per-request refusal rate of about 99.99% — which is the honest way to state a safety requirement in an adversarial setting.');
      }
    },
    quiz: [
      {
        q: 'A model refuses 99% of harmful requests. Over 100 attempts, an attacker succeeds with probability…',
        options: ['1%', '10%', '63%', '99%'],
        answer: 2,
        why: '$1-0.99^{100} \\approx 0.634$. Per-request rates are the wrong unit whenever retries are cheap.'
      },
      {
        q: 'Chain-of-thought output can be used as…',
        options: ['a compliance audit trail', 'a debugging aid — it may be unfaithful to the actual computation', 'proof of correctness', 'a substitute for evaluation'],
        answer: 1,
        why: 'Models can reach answers via cues never mentioned in the chain; legibility is evidence, not proof.'
      },
      {
        q: 'Sparse autoencoders are used to…',
        options: ['compress the model', 'extract more monosemantic features from activations in superposition, enabling auditing and steering', 'quantize weights', 'detect watermarks'],
        answer: 1,
        why: 'Coverage is partial and features are not stable across runs — useful, not yet compliance-grade.'
      }
    ],
    cards: [
      { q: 'Refusal arithmetic', a: '99% per-request refusal → 63% attacker success over 100 tries ($1-0.99^{100}$).' },
      { q: 'The two-sided safety metric', a: 'Attack success rate AND false-refusal rate on benign-adjacent prompts; movement in either is a regression.' },
      { q: 'CoT faithfulness', a: 'The printed chain need not be the computation performed — a debugging aid, not an audit trail.' },
      { q: 'Superposition & SAEs', a: 'Neurons are polysemantic because features outnumber dimensions; sparse autoencoders recover more monosemantic features for audit and steering.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.18 */
  ML.section({
    id: 'part4-recall', track: 'llm', num: '4.23',
    title: 'Rapid recall — Part 4 in fourteen lines',
    lede: 'The transformer sheet. Every line should unpack into a derivation or an arithmetic you can do on a whiteboard.',
    html: `
${H.table(['#', 'The line', 'Section'], [
      ['1', '√dₖ: the dot product’s variance is dₖ; rescale or the softmax saturates.', '<a href="#/attention">4.3</a>'],
      ['2', 'RoPE: $(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative position for free.', '<a href="#/rope">4.4</a>'],
      ['3', 'KV bytes = 2·L·n_kv·d_head·seq·batch·bytes.', '<a href="#/kv-cache">4.7</a>'],
      ['4', 'Cache size: MHA ≫ GQA > MLA ≈ MQA.', '<a href="#/kv-cache">4.7</a>'],
      ['5', 'FlashAttention is exact — an IO optimisation, not sparsity.', '<a href="#/kv-cache">4.7</a>'],
      ['6', 'C ≈ 6ND; Chinchilla ≈ 20 tokens/param (⚑ coefficients contested).', '<a href="#/scaling-laws">4.10</a>'],
      ['7', 'Inference-optimal overtrains: Llama-3-8B ≈ 1,875 tok/param.', '<a href="#/scaling-laws">4.10</a>'],
      ['8', 'DPO cancels Z(x); it also inflates length (arXiv:2403.19159).', '<a href="#/post-training">4.12</a>'],
      ['9', 'GRPO: $A_i=(r_i-\\mathrm{mean})/\\mathrm{std}$, no critic. RLVR: rule-based reward.', '<a href="#/post-training">4.12</a>'],
      ['10', 'LoRA r=16, α=2r, all linear layers; QLoRA = NF4 base + BF16 adapter.', '<a href="#/lora">4.13</a>'],
      ['11', 'Bytes/param: FP32 4 · FP16/BF16 2 · FP8/INT8 1 · INT4 0.5.', '<a href="#/kv-cache">4.7</a>'],
      ['12', 'Prefill compute-bound; decode memory-bandwidth-bound.', '<a href="#/serving">4.14</a>'],
      ['13', '70B / 4k / batch 8 → 10 GB of KV cache.', '<a href="#/kv-cache">4.7</a>'],
      ['14', 'Judge: always swap positions and control for length.', '<a href="#/llm-eval">4.16</a>']
    ])}
${H.table(['#', 'Also', 'Section'], [
      ['15', 'Long context: local + sink tokens + a few global layers; test with RULER, not one needle.', '<a href="#/rope">4.4</a>'],
      ['16', 'CoT can be unfaithful; SAEs give monosemantic features; 99% refusal ≠ safe under retries.', '<a href="#/safety">4.17</a>'],
      ['17', 'Speculative decoding: $(1-\\alpha^{k+1})/(1-\\alpha)$ — α=0.7, k=4 → 2.8 tokens/pass.', '<a href="#/serving">4.14</a>'],
      ['18', 'Llama-3-8B parameter arithmetic: 42.0M attention + 176.2M FFN per layer → 8.03B.', '<a href="#/block">4.5</a>']
    ])}

${H.lab('drill4', 'Part 4 drill', 'Eighteen prompts, shuffled. These are the ones asked most often.')}`,
    labs: {
      drill4: function (host) {
        const cards = [
          ['Why divide attention logits by √dₖ?', 'Var(q·k)=dₖ, so logits scale like √dₖ; large logits saturate the softmax and kill gradients.'],
          ['State the RoPE identity.', '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — the score depends only on the relative offset.'],
          ['KV cache formula.', 'bytes = 2·L·n_kv·d_head·seq·batch·bytes-per-element.'],
          ['Size the cache for 70B, 4k, batch 8, BF16.', '2·80·8·128·4096·8·2 ≈ 10 GB.'],
          ['Is FlashAttention an approximation?', 'No — exact; it tiles into SRAM with an online softmax so the s×s matrix never reaches HBM.'],
          ['Order the KV variants by cache size.', 'MHA ≫ GQA > MLA ≈ MQA.'],
          ['Training compute formula, and for 7B on 2T?', 'C≈6ND ≈ 8.4×10²² FLOPs.'],
          ['Chinchilla in one line, with the caveat.', '≈20 tokens per parameter at fixed compute; ⚑ exact coefficients are contested.'],
          ['Why train past compute-optimal?', 'Inference cost dominates lifetime cost — pay training once, inference forever.'],
          ['How does DPO remove the reward model?', 'Invert the closed-form optimal policy; in the Bradley–Terry difference Z(x) cancels, leaving a logistic loss on pairs.'],
          ['DPO’s documented failure.', 'Length inflation (arXiv:2403.19159); monitor mean response length.'],
          ['GRPO advantage, and its degenerate case.', '$(r_i-\\mathrm{mean})/\\mathrm{std}$; a unanimous group has σ=0 and teaches nothing.'],
          ['What is RLVR and why does it matter?', 'RL from verifiable rewards (tests pass, answer matches) — nothing to hack; how reasoning models are trained.'],
          ['LoRA defaults in 2026.', 'r=16, α=2r, all linear layers; QLoRA = 4-bit NF4 base + BF16 adapter.'],
          ['The 8B QLoRA memory numbers.', '≈42M trainable (0.52%), 9–11 GB vs ≈128 GB full fine-tune.'],
          ['Bytes per parameter by precision.', 'FP32 4 · FP16/BF16 2 · FP8/INT8 1 · INT4 0.5.'],
          ['Prefill vs decode.', 'Prefill compute-bound (sets TTFT); decode memory-bandwidth-bound (sets tokens/s).'],
          ['Speculative decoding expected tokens.', '$(1-\\alpha^{k+1})/(1-\\alpha)$; α=0.7, k=4 → 2.8.'],
          ['Two mandatory judge mitigations.', 'Swap positions and average; control for length.'],
          ['Long-context recipe and its evaluation.', 'RoPE interpolation + local layers + sink tokens + a few global layers; evaluate with RULER-style multi-needle tests.'],
          ['Parameters per transformer layer.', '≈12d²: 4d² attention + 8d² FFN.'],
          ['99% refusal over 100 attempts?', '63% attacker success — per-request rates are the wrong unit.']
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
        q: 'Which single number is most often needed on an LLM whiteboard?',
        options: ['The learning rate', 'The KV cache size for a given model, context and batch', 'The vocabulary size', 'The number of attention heads'],
        answer: 1,
        why: 'It decides how many cards you need, what batch you can serve, and whether long context is feasible — and the formula is short enough to derive live.'
      }
    ]
  });
})();
