/* ============================================================
   PART 4 — LLMs, continued: multimodal models (4.19), reasoning
   and test-time compute (4.20), structured output (4.21),
   speculative decoding (4.22).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 4.19 */
  ML.section({
    id: 'multimodal', track: 'llm', num: '4.19', level: 2,
    title: 'Vision-language and multimodal models',
    lede: 'Once you can turn anything into a sequence of vectors, the transformer does not care what it was. Images become patches, audio becomes spectrogram frames, and the interesting engineering is entirely in how two modalities are made to share a space.',
    prereq: ['attention', 'self-supervised'],
    related: ['self-supervised', 'attention', 'rag'],
    html: `
${H.tldr([
      'A ViT cuts an image into 16×16 patches, projects each to a vector, adds a position embedding, and runs a standard transformer. A 224×224 image becomes 196 tokens.',
      'CLIP trains an image encoder and a text encoder with InfoNCE (§2.22) over a batch of pairs, so that matching image and caption land near each other. That shared space is what makes zero-shot classification possible.',
      'Modern VLMs mostly do <b>early fusion</b>: project visual features into the LLM’s token space and feed them in as tokens. It is the simplest thing that works, and it inherits the LLM’s reasoning for free.'
    ])}

<h2><span class="sn">4.19.1</span> Vision transformers: an image is 196 tokens</h2>
${H.svg('how an image becomes a sequence', '0 0 660 190', `
<defs><style>
.pt{fill:var(--panel);stroke:var(--line)}
.pa{fill:color-mix(in oklab,var(--c1) 26%,transparent);stroke:var(--c1)}
.tx{font:11px ui-sans-serif,system-ui;fill:var(--text)}
.mo{font:10px ui-monospace,monospace;fill:var(--faint)}
.ar{stroke:var(--faint);fill:none;stroke-width:1.4;marker-end:url(#mm)}
</style>
<marker id="mm" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--faint)"/></marker></defs>
<g>
${Array.from({ length: 16 }, (_, i) => {
      const r = Math.floor(i / 4), c = i % 4;
      return '<rect class="' + (i === 5 ? 'pa' : 'pt') + '" x="' + (22 + c * 28) + '" y="' + (34 + r * 28) + '" width="26" height="26"/>';
    }).join('')}
<text class="tx" x="78" y="24" text-anchor="middle">image 224×224</text>
<text class="mo" x="78" y="164" text-anchor="middle">16×16 patches</text>
<path class="ar" d="M146 82 H196"/>
<text class="mo" x="171" y="74" text-anchor="middle">flatten</text>
<rect class="pt" x="206" y="40" width="34" height="86" rx="4"/>
<rect class="pa" x="206" y="68" width="34" height="14"/>
<text class="tx" x="223" y="24" text-anchor="middle">196 × 768</text>
<text class="mo" x="223" y="146" text-anchor="middle">+ pos. emb.</text>
<path class="ar" d="M248 82 H298"/>
<text class="mo" x="273" y="74" text-anchor="middle">linear</text>
<rect class="pt" x="308" y="34" width="120" height="98" rx="8"/>
<text class="tx" x="368" y="78" text-anchor="middle">Transformer</text>
<text class="mo" x="368" y="96" text-anchor="middle">the same one as §4.5</text>
<path class="ar" d="M436 82 H486"/>
<rect class="pt" x="496" y="40" width="34" height="86" rx="4"/>
<rect class="pa" x="496" y="40" width="34" height="14"/>
<text class="mo" x="513" y="146" text-anchor="middle">[CLS] or mean</text>
<path class="ar" d="M538 60 H578"/>
<text class="tx" x="612" y="64" text-anchor="middle">embedding</text>
<text class="mo" x="612" y="80" text-anchor="middle">or classes</text>
</g>
`, 'Nothing here is image-specific except the patch-and-project step. Everything after it is §4.5 verbatim — which is precisely why one architecture absorbed every modality.')}
${H.table(['Design point', 'Choice', 'Consequence'], [
      ['Patch size', '16×16 → 196 tokens; 14×14 → 256; 32×32 → 49', 'attention is $O(n^2)$, so patch size is the accuracy/compute dial'],
      ['Position information', 'learned 1-D, 2-D sin-cos, or RoPE-2D', 'learned embeddings do not extrapolate to other resolutions without interpolation'],
      ['Pooling', '[CLS] token or mean over patches', 'mean pooling is slightly better and has no extra parameters'],
      ['Inductive bias', '<b>none</b> — no locality, no translation equivariance', 'needs far more data than a CNN, or heavy augmentation and distillation (DeiT)']
    ])}
${H.key('A ViT trained on ImageNet-1k alone loses to a ResNet. Trained on 300M images it wins comfortably. That crossover is the clearest demonstration in the field of the trade between inductive bias and data: bias substitutes for data, and stops paying once you have enough.')}

<h2><span class="sn">4.19.2</span> CLIP: one embedding space for two modalities</h2>
<p>Encode $N$ images and their $N$ captions, L2-normalise, and compute the $N\\times N$ matrix of cosine similarities divided by a learned temperature. Train with cross-entropy in both directions — image→text and text→image — against the identity matrix. That is the whole objective (§2.22).</p>
${H.lab('clip', 'A CLIP-style contrastive space, trained here', 'Two small encoders, one for "images" (feature vectors) and one for "captions", trained with a real InfoNCE objective until the diagonal lights up. Step through training and watch the similarity matrix become diagonal.')}
${H.table(['Property', 'Why it follows', 'Practical use'], [
      ['<b>Zero-shot classification</b>', 'encode "a photo of a {class}" for every class, take the nearest', 'no labelled training data for the new task at all'],
      ['Retrieval both ways', 'the space is shared', 'image search by text and text search by image, one index'],
      ['A quality signal for generators', 'CLIP score = similarity of output to prompt', 'used to filter training data for diffusion models (§6.2)'],
      ['<b>Bag-of-words weakness</b>', 'the contrastive objective never forces word order to matter', '"a horse riding an astronaut" is close to its reverse — a known, measurable failure (Winoground)']
    ])}
${H.flag('CLIP’s compositional weakness is not a small caveat. On benchmarks designed to test relations and word order, CLIP-style models score near chance. This is a direct consequence of the objective: nothing in InfoNCE rewards distinguishing two captions with the same words in a different order, because such a pair almost never appears as a hard negative in a random batch.')}

<h2><span class="sn">4.19.3</span> How vision gets into an LLM</h2>
${H.table(['Architecture', 'Mechanism', 'Examples', 'Trade'], [
      ['Dual encoder (late fusion)', 'two towers, similarity at the end', 'CLIP, SigLIP', 'fast retrieval — you can index one side; no cross-modal reasoning'],
      ['Cross-attention', 'text layers attend to frozen visual features', 'Flamingo, Llama-3.2-vision', 'keeps the LLM mostly intact; adds parameters'],
      ['<b>Early fusion / projection</b>', 'an MLP maps visual features into the token space; they enter as tokens', '<b>LLaVA, Qwen-VL, GPT-4o-class</b>', 'simplest, inherits all the LLM’s abilities, costs context'],
      ['Native multimodal', 'one model trained on interleaved modalities from the start', 'Gemini, GPT-4o', 'best quality; only possible if you are pretraining anyway'],
      ['Q-Former / resampler', 'compress many visual tokens into a fixed few', 'BLIP-2', 'controls the context cost of high-resolution images']
    ])}
${H.worked('the context arithmetic of an image', `
<p>A 1024×1024 image at 14×14 patches is $73^2 \\approx 5{,}300$ tokens. Tiled high-resolution encoders (the standard approach for document understanding) can produce 2,000–6,000 tokens <b>per image</b>.</p>
<p>At $2.50 per million input tokens, one high-resolution page image costs about <b>$0.0125</b> to look at once — roughly the same as 5,000 tokens of text, which is about eight pages of prose. For a document pipeline processing 100,000 pages a month that is $1,250 in vision tokens alone, which is why resamplers, adaptive tiling and "read the OCR text instead" all remain live engineering choices rather than settled ones.</p>`)}

${H.probe([
      ['How does a ViT handle a different input resolution than it was trained on?', 'The patch projection is resolution-agnostic, but the position embeddings are not. You interpolate them — which works surprisingly well and is standard practice.'],
      ['Why can CLIP classify without labels?', 'Text and images share one embedding space, so a class name encoded as text is a point you can measure distance to. Classification becomes nearest-neighbour retrieval among class-name embeddings.'],
      ['What is CLIP bad at, and why?', 'Compositionality and word order. The contrastive objective is satisfied by matching the bag of concepts; nothing pushes it to distinguish "A above B" from "B above A".'],
      ['You need to answer questions about scanned invoices. Where do the tokens go?', 'Count them: high-resolution tiling can be thousands of tokens per page. Consider a resampler, adaptive resolution, or extracting text with OCR and sending that — the cheapest thing that works (§5.13).']
    ])}`,
    labs: {
      clip: function (host) {
        const st = Viz.controls(host, [
          { k: 'temp', label: 'learned temperature τ', min: .02, max: .5, step: .01, value: .07, fmt: v => v.toFixed(2) },
          { k: 'dim', label: 'embedding dimension', min: 4, max: 32, step: 2, value: 12, fmt: v => v },
          { k: 'N', label: 'batch size', min: 4, max: 10, step: 1, value: 7, fmt: v => v }
        ], () => reset());

        let ei = null, et = null, steps = 0;
        const outRef = { fn: null };

        function reset() {
          const R = Num.rng(13);
          const N = st.N, D = st.dim, IN = 10;
          // "raw features" for the two modalities of the same N concepts
          const concept = Array.from({ length: N }, () => Array.from({ length: IN }, () => R.normal(0, 1)));
          window.__clipData = {
            img: concept.map(c => c.map(v => v + R.normal(0, .35))),
            txt: concept.map(c => c.map(v => v * .8 + R.normal(0, .45)))
          };
          ei = Array.from({ length: D }, () => Array.from({ length: IN }, () => R.normal(0, .3)));
          et = Array.from({ length: D }, () => Array.from({ length: IN }, () => R.normal(0, .3)));
          steps = 0;
          if (S) S.redraw();
        }

        function encode(W, x) {
          const v = W.map(row => Num.dot(row, x));
          const n = Num.norm(v) || 1;
          return v.map(z => z / n);
        }
        /* one step of real InfoNCE training by finite-difference gradients on
           the two projection matrices — small enough to be honest and exact. */
        function trainStep(lr) {
          const d = window.__clipData;
          const loss = () => Num.infoNCE(d.img.map(x => encode(ei, x)), d.txt.map(x => encode(et, x)), st.temp).loss;
          const base = loss();
          const h = 1e-3;
          [ei, et].forEach(W => {
            for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) {
              const o = W[i][j];
              W[i][j] = o + h;
              const up = loss();
              W[i][j] = o - lr * (up - base) / h;
            }
          });
          steps++;
        }

        Viz.buttons(host, [
          { label: 'Train 5 steps', primary: true, on: () => { for (let i = 0; i < 5; i++) trainStep(.35); S.redraw(); } },
          { label: 'Train 40 steps', on: () => { for (let i = 0; i < 40; i++) trainStep(.35); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        const out = Viz.readout(host, [
          { k: 'step', label: 'training steps', cls: 'key' },
          { k: 'loss', label: 'InfoNCE loss' },
          { k: 'i2t', label: 'image→text top-1', cls: 'good' },
          { k: 'chance', label: 'chance' }
        ]);

        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const d = window.__clipData;
            const A = d.img.map(x => encode(ei, x)), B = d.txt.map(x => encode(et, x));
            const res = Num.infoNCE(A, B, st.temp);
            const M = res.sim.map(row => { const l = Num.logsumexp(row); return row.map(v => Math.exp(v - l)); });
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 56, r: 18, t: 30, b: 40 } });
            P.heat(M, {
              rows: Array.from({ length: st.N }, (_, i) => 'img ' + (i + 1)),
              cols: Array.from({ length: st.N }, (_, i) => 'txt ' + (i + 1)),
              lo: 0, hi: 1
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('softmax over captions for each image — the diagonal is the correct pairing', w / 2, h - 12);
            out({
              step: steps, loss: res.loss.toFixed(4),
              i2t: (res.acc * 100).toFixed(0) + '%',
              chance: (100 / st.N).toFixed(0) + '%'
            });
          }
        });
        reset();
        Viz.note(host, 'At step 0 the matrix is a fog: the two random projections have nothing to do with each other. Train 40 steps and the diagonal emerges — the two encoders have agreed on a shared coordinate system, which is the entire content of "a shared embedding space". <b>Nothing here knows what an image is</b>; the modality is irrelevant to the objective, which is why the same loss trains CLIP, audio–text models and retrieval encoders alike.');
      }
    },
    quiz: [
      {
        q: 'A 224×224 image with 16×16 patches becomes how many tokens?',
        options: ['16', '49', '196', '768'],
        answer: 2,
        why: '$(224/16)^2 = 14^2 = 196$. Attention cost is quadratic in this number, which is why patch size is the main compute dial.'
      },
      {
        q: 'CLIP enables zero-shot classification because…',
        options: ['it was trained on every class', 'text and images share one embedding space, so a class name is a point to measure distance to', 'it uses a softmax over classes', 'it fine-tunes at inference'],
        answer: 1,
        why: 'Classification becomes nearest-neighbour retrieval among encoded class-name prompts.'
      },
      {
        q: 'The best-known weakness of CLIP-style models is…',
        options: ['slow inference', 'poor compositionality — word order and relations barely matter', 'inability to handle colour', 'requiring labels'],
        answer: 1,
        why: 'A direct consequence of InfoNCE over random batches: nothing supplies hard negatives that differ only in word order.'
      },
      {
        q: 'The dominant way vision enters an LLM today is…',
        options: ['a separate classifier whose label is inserted as text', 'projecting visual features into the token embedding space and feeding them as tokens', 'fine-tuning the LLM on image bytes', 'cross-attention only'],
        answer: 1,
        why: 'Early fusion via an MLP projector (LLaVA-style) is simplest and inherits the LLM’s reasoning; it costs context tokens.'
      }
    ],
    cards: [
      { q: 'ViT in one line', a: 'Patchify → linear project → add position → standard transformer. 224×224 at 16×16 = 196 tokens.' },
      { q: 'CLIP objective', a: 'InfoNCE in both directions over an $N\\times N$ image–text similarity matrix with a learned temperature.' },
      { q: 'Inductive bias vs data', a: 'ViT loses to a CNN on ImageNet-1k and wins at 300M images. Bias substitutes for data and stops paying.' },
      { q: 'Image token cost', a: 'High-resolution tiling can be 2k–6k tokens per image — roughly eight pages of prose, per picture.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.20 */
  ML.section({
    id: 'reasoning', track: 'llm', num: '4.20', level: 3,
    title: 'Reasoning models and test-time compute',
    lede: 'The 2024–2026 shift: instead of making the model bigger, let it think for longer at inference. The scaling law moved from parameters to tokens-of-thought, and the economics of serving changed with it.',
    prereq: ['decoding', 'post-training'],
    related: ['decoding', 'post-training', 'llm-eval'],
    html: `
${H.tldr([
      'Chain of thought works because the model can only do a bounded amount of computation per token. Writing intermediate steps buys more forward passes for one problem — <b>it is serialised compute, not introspection</b>.',
      'Self-consistency (sample $k$, take the majority) converts a per-sample accuracy $p$ into something much higher when errors are diverse — and does nothing at all when they are correlated.',
      'RLVR — reinforcement learning from <i>verifiable</i> rewards — is what trains long-form reasoning: the reward is whether the answer is right, checked by a program, so there is no reward model to hack (§4.12).'
    ])}

<h2><span class="sn">4.20.1</span> Why writing things down helps</h2>
${H.intuition(`<p>A transformer performs a fixed amount of computation per generated token: $L$ layers, once. There are problems — multiplying two large numbers, following a long chain of deductions — whose answer requires more sequential steps than $L$. No amount of width fixes that.</p>
<p>Generating intermediate tokens is the workaround. Each token written is another full forward pass, and the written text is the scratchpad that carries state between them. Chain of thought converts a <b>depth</b> limit into a <b>length</b> budget. This is a real complexity-theoretic statement, not a metaphor: transformers with a polynomial-length scratchpad can express strictly more than transformers without one.</p>`)}
${H.key('The model is not "thinking out loud" in the sense of narrating a hidden process. The written tokens <i>are</i> the computation. Which is also why a reasoning trace can be fluent, plausible and completely disconnected from the answer that follows it.')}

<h2><span class="sn">4.20.2</span> The ways of spending inference compute</h2>
${H.table(['Method', 'How', 'Cost', 'When it helps'], [
      ['Chain of thought', 'ask for steps, or train the model to produce them', '~3–10× output tokens', 'multi-step arithmetic, logic, code'],
      ['<b>Self-consistency</b>', 'sample $k$ traces at $T>0$, take the majority answer', '$k\\times$', '<b>verifiable short answers</b>; the classic result is +10–20 points on GSM8K'],
      ['Best-of-$n$ with a verifier', 'sample $n$, score with a reward or process model, keep the best', '$n\\times$ + verifier', 'when a good scorer exists'],
      ['Process reward models (PRM)', 'score every <i>step</i>, not just the answer', 'expensive to label', 'long derivations where one bad step ruins everything'],
      ['Tree search / MCTS', 'branch, evaluate, backtrack', '10–1000×', 'search-shaped problems with a cheap evaluator'],
      ['<b>Long RL-trained reasoning</b>', 'the model learns to produce long traces with backtracking', 'built in at training time', 'o-series, R1-style models; the current frontier'],
      ['Self-refine / critique', 'generate, critique, revise', '2–3×', 'writing and code; <b>weak without an external signal</b>']
    ])}
${H.lab('selfcons', 'Self-consistency: when majority voting works, and when it cannot', 'Real binomial and multinomial simulation. Set the per-sample accuracy and the diversity of the errors, and see what majority voting over $k$ samples actually buys. The failure case is the important one.')}

${H.deriv('why majority voting helps, and the condition it needs', [
      ['$p > 1/2$ for two options', 'Condorcet’s jury theorem: with independent voters each better than chance, the majority’s accuracy tends to 1 as $k\\to\\infty$.'],
      ['$P(\\text{correct}) = \\sum_{j>k/2}\\binom{k}{j}p^j(1-p)^{k-j}$', 'Exactly the binomial tail. At $p=0.6$, $k=15$ gives 0.79; $k=51$ gives 0.93.'],
      ['many wrong answers help', 'With $m$ possible answers, the correct one needs only a <i>plurality</i>. Errors spread over $m-1$ options each get $\\approx(1-p)/(m-1)$, so the bar is far lower than $1/2$ — this is why self-consistency works on arithmetic with $p$ as low as 0.35.'],
      ['<b>correlated errors break it</b>', 'If the model makes the <i>same</i> mistake every time — a misread premise, a systematic misconception — every sample votes for it. The effective $k$ collapses toward 1. This is exactly §2.16’s $\\rho\\sigma^2$ floor in another costume.']
    ])}
${H.pitfall('Self-consistency needs an answer you can compare for equality. It works on arithmetic and multiple choice; it does not work on essays, and "majority vote over generated prose" is not a thing. For open-ended output the analogue is best-of-$n$ with a verifier — and then you have moved the whole problem into the quality of the verifier.')}

<h2><span class="sn">4.20.3</span> The test-time scaling curve</h2>
${H.lab('ttc', 'Trading training compute for inference compute', 'The same accuracy target reached two ways. The crossover point is a real business decision: a bigger model costs once and is paid on every request; more thinking tokens cost nothing up front and are paid on every request too — but only on the hard ones.')}
${H.flag('The honest state of the evidence in 2026: test-time scaling gives large, reproducible gains on <b>verifiable</b> tasks — competition mathematics, code with tests, formal logic. Gains on open-ended tasks with no automatic checker are much smaller and much harder to measure, because the evaluation is itself a model. Be precise about which kind of task you mean; conflating them is the most common overclaim in the area.')}
${H.table(['Consequence', 'Detail'], [
      ['Latency is now variable', 'a hard question may generate 10,000 reasoning tokens; your p99 is set by the tail, not the mean'],
      ['Cost per request is variable', 'budget by expected tokens, and cap the thinking budget explicitly'],
      ['Caching is less effective', 'reasoning traces are unique; only the prompt prefix caches'],
      ['Evaluation changes', 'pass@1 at a fixed token budget is the meaningful comparison, not pass@1 unbounded'],
      ['Distillation works well', 'training a small model on a large model’s traces transfers much of the gain — the R1-distill result']
    ])}

${H.probe([
      ['Why does chain of thought improve accuracy at all?', 'It converts a depth limit into a length budget: each generated token is another forward pass, and the text carries state between them. It is extra serial computation, not introspection.'],
      ['When does self-consistency fail?', 'When errors are correlated — the model makes the same mistake each time — or when answers cannot be compared for equality. Diversity is the resource being spent.'],
      ['What makes RLVR different from RLHF?', 'The reward is computed by a program (a test suite, a numeric check), not a learned reward model. There is nothing to reward-hack, so training can run much longer before it degenerates.'],
      ['A reasoning model is too slow for your product. What do you do?', 'Route: cheap model first, escalate only on low confidence or detected difficulty; cap the thinking budget; distil the reasoning traces into a smaller model; cache prompt prefixes (§5.13).']
    ])}`,
    labs: {
      selfcons: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'per-sample accuracy p', min: .1, max: .95, step: .01, value: .45, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'm', label: 'plausible answers m', min: 2, max: 20, step: 1, value: 8, fmt: v => v },
          { k: 'rho', label: 'error correlation (the same mistake repeated)', min: 0, max: .95, step: .05, value: 0, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'one', label: 'single sample', cls: 'bad' },
          { k: 'k5', label: 'majority of 5' },
          { k: 'k21', label: 'majority of 21', cls: 'good' },
          { k: 'ceil', label: 'ceiling as k→∞', cls: 'key' },
          { k: 'cost', label: 'cost of k=21' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(19);
            function majorityAcc(k) {
              let ok = 0; const trials = 1500;
              for (let t = 0; t < trials; t++) {
                // with probability rho the model commits to one fixed wrong answer for the whole question
                const stuck = R() < st.rho;
                const stuckAns = 1 + R.int(st.m - 1);
                const votes = new Array(st.m).fill(0);
                for (let i = 0; i < k; i++) {
                  if (R() < st.p) votes[0]++;
                  else votes[stuck ? stuckAns : 1 + R.int(st.m - 1)]++;
                }
                let bi = 0;
                for (let a = 1; a < st.m; a++) if (votes[a] > votes[bi]) bi = a;
                if (bi === 0) ok++;
              }
              return ok / trials;
            }
            const ks = [1, 3, 5, 7, 11, 15, 21, 31, 41, 61];
            const pts = ks.map(k => [k, majorityAcc(k)]);
            const P = Viz.plot(ctx, w, h, { xd: [1, 61], yd: [0, 1], pad: { l: 52, r: 14, t: 16, b: 40 } })
              .frame({ xlabel: 'samples k (majority vote)', ylabel: 'accuracy', yfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.area(pts, { color: T.c1, alpha: .14 });
              P.line(pts, { color: T.c1, width: 2.8 });
              P.dots(pts, { r: 3.4, color: T.c1 });
              P.hline(st.p, { color: T.c2, dash: [5, 4], label: 'single sample' });
              P.hline(1 - st.rho * (1 - st.p) / (1 - st.p * st.rho || 1), { color: T.faint, dash: [3, 3] });
            });
            out({
              one: (st.p * 100).toFixed(1) + '%',
              k5: (pts[2][1] * 100).toFixed(1) + '%',
              k21: (pts[6][1] * 100).toFixed(1) + '%',
              ceil: (pts[pts.length - 1][1] * 100).toFixed(1) + '%',
              cost: '21× output tokens'
            });
          }
        });
        Viz.note(host, 'Set p = 0.45 with m = 8: a single sample is wrong more often than right, and majority voting over 21 samples still climbs above 80% — because the 55% of wrong answers are scattered over seven options while the correct one is concentrated. Now raise <b>error correlation</b> to 0.6 and the curve flattens near the single-sample rate. <b>Diversity of errors, not the number of samples, is what you are buying</b> — the same lesson as the ensemble variance formula in §2.16.');
      },

      ttc: function (host) {
        const st = Viz.controls(host, [
          { k: 'base', label: 'base model capability', min: .2, max: .8, step: .02, value: .42, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'qps', label: 'requests per day', min: 100, max: 1000000, step: 100, value: 50000, fmt: v => v.toLocaleString() },
          { k: 'thinkcost', label: 'thinking tokens at full budget', min: 200, max: 20000, step: 200, value: 4000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'small', label: 'small + thinking', cls: 'good' },
          { k: 'big', label: 'big model, no thinking', cls: 'key' },
          { k: 'cs', label: 'cost/day, small + thinking' },
          { k: 'cb', label: 'cost/day, big model' },
          { k: 'verdict', label: 'cheaper option' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            /* Both curves are log-linear in compute, which is what the published
               scaling results look like; the constants are illustrative. */
            const acc = (base, logC) => Math.min(.97, base + .11 * logC);
            const P = Viz.plot(ctx, w, h, { xd: [0, 4], yd: [.2, 1], pad: { l: 52, r: 14, t: 16, b: 42 } })
              .frame({
                xticks: [0, 1, 2, 3, 4], xfmt: v => Math.pow(10, v) + '×',
                xlabel: 'inference compute multiplier', ylabel: 'accuracy', yfmt: v => (v * 100).toFixed(0) + '%'
              });
            P.clip(() => {
              P.fn(lc => acc(st.base, lc), { color: T.c1, width: 2.8, n: 120 });
              P.fn(lc => acc(st.base + .16, lc * .55), { color: T.c4, width: 2.4, n: 120 });
              P.hline(acc(st.base + .16, 0), { color: T.c4, dash: [4, 4], label: 'big model, single pass' });
            });
            const smallAcc = acc(st.base, Math.log10(1 + st.thinkcost / 400));
            const bigAcc = acc(st.base + .16, 0);
            // $ per million tokens: small 0.60 out, big 10.00 out; 400 output tokens baseline
            const costSmall = st.qps * (400 + st.thinkcost) / 1e6 * 0.60;
            const costBig = st.qps * 400 / 1e6 * 10.0;
            out({
              small: (smallAcc * 100).toFixed(1) + '%',
              big: (bigAcc * 100).toFixed(1) + '%',
              cs: '$' + costSmall.toFixed(0),
              cb: '$' + costBig.toFixed(0),
              verdict: costSmall < costBig
                ? 'small + thinking (' + (costBig / costSmall).toFixed(1) + '× cheaper)'
                : 'big model (' + (costSmall / costBig).toFixed(1) + '× cheaper)'
            });
          }
        });
        Viz.legend(host, [{ c: 'var(--c1)', t: 'small model + thinking tokens' }, { c: 'var(--c4)', t: 'large model + thinking tokens' }]);
        Viz.note(host, 'Both curves rise roughly linearly in log-compute — that is the shape the published test-time scaling results have. The decision is economic: <b>at 4,000 thinking tokens a small model costs about a quarter of the large one per request and lands within a few points on accuracy</b>. Push the thinking budget to 20,000 and it stops being cheaper. The right architecture is usually a router: cheap and short by default, escalate only what needs it (§5.13).');
      }
    },
    quiz: [
      {
        q: 'Chain of thought helps primarily because…',
        options: ['the model explains its reasoning to itself', 'each generated token is another forward pass, converting a depth limit into a length budget', 'it reduces temperature', 'it retrieves better context'],
        answer: 1,
        why: 'It is extra serialised computation. The written text is the computation, not a report about it.'
      },
      {
        q: 'Self-consistency fails when…',
        options: ['the temperature is too high', 'errors are correlated, so every sample makes the same mistake', 'k is odd', 'the answer space is large'],
        answer: 1,
        why: 'A large answer space actually *helps* — errors scatter. Correlated errors are the killer, exactly as in the ensemble formula of §2.16.'
      },
      {
        q: 'RLVR differs from RLHF in that the reward is…',
        options: ['larger', 'computed by a program that verifies the answer, so there is no reward model to hack', 'human-annotated', 'given per token'],
        answer: 1,
        why: 'Verifiable rewards let training run far longer without degeneration, which is why they underpin the long-reasoning models.'
      },
      {
        q: 'The clearest evidence for test-time scaling comes from…',
        options: ['open-ended writing quality', 'verifiable tasks: competition maths, code with tests, formal logic', 'chat helpfulness ratings', 'summarisation'],
        answer: 1,
        why: 'Gains on unverifiable tasks are smaller and harder to measure because the evaluation is itself a model.'
      }
    ],
    cards: [
      { q: 'Why CoT works', a: 'A transformer does fixed work per token; generated tokens buy more sequential passes. Depth limit → length budget.' },
      { q: 'Self-consistency condition', a: 'Errors must be diverse. With $m$ answers the correct one needs only a plurality, so it works even at $p<0.5$ — unless errors are correlated.' },
      { q: 'RLVR', a: 'RL from verifiable rewards: a program checks the answer, so there is no reward model to hack.' },
      { q: 'Operational cost of reasoning models', a: 'Variable latency and cost; p99 set by the token tail; cap the thinking budget and route hard cases only.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.21 */
  ML.section({
    id: 'structured-output', track: 'llm', num: '4.21', level: 2,
    title: 'Structured output, constrained decoding, and tool calls',
    lede: 'Getting valid JSON out of a language model is not a prompting problem. It is a decoding problem, and once you see it that way the guarantee becomes total rather than probabilistic.',
    prereq: ['decoding'],
    related: ['decoding', 'agents', 'mcp'],
    html: `
${H.tldr([
      'Constrained decoding masks the logits: at each step, set the probability of every token that could not continue a valid string to $-\\infty$. <b>Invalid output becomes impossible, not unlikely.</b>',
      'Any regular language (JSON with a fixed schema, an enum, a date) compiles to a finite automaton, and the automaton’s current state determines the allowed-token mask. The mask is precomputed per state, so the runtime cost is near zero.',
      'Constraining <i>shape</i> is free. Constraining <i>content</i> is not — a model forced into a schema it does not understand will fill it with confident nonsense.'
    ])}

<h2><span class="sn">4.21.1</span> The three levels of guarantee</h2>
${H.table(['Approach', 'Guarantee', 'Cost', 'When'], [
      ['Ask nicely in the prompt', 'none — 85–98% valid depending on the model', 'free', 'prototypes'],
      ['Ask + retry on parse failure', 'eventually valid, unbounded latency', 'extra calls on the tail', 'acceptable for low volume'],
      ['<b>Constrained decoding</b>', '<b>100% syntactically valid, by construction</b>', 'a mask per step; effectively free', 'production'],
      ['Fine-tune on the format', 'higher natural compliance', 'a training run', 'combine with constraints, do not replace them']
    ])}
${H.key('Retry-until-it-parses has an unbounded tail and costs the most exactly when the model is struggling. Masking the logits removes the failure mode entirely and costs a lookup.')}

${H.lab('constrained', 'Watch a grammar mask the logits, token by token', 'A real finite-state machine for a small JSON schema. Step through generation: at each position you can see the automaton state, which tokens it permits, and the model’s raw preferences being overruled.')}

<h2><span class="sn">4.21.2</span> How it is implemented</h2>
${H.steps([
      '<b>Compile the schema to an automaton.</b> A JSON Schema with fixed keys is a regular language; a full recursive grammar needs a pushdown automaton, which is what the general-purpose libraries build.',
      '<b>Precompute, for every automaton state, the set of vocabulary tokens that can legally follow.</b> This is the expensive step, and it is done once per schema, not once per token. Because the tokenizer splits text arbitrarily, a single token may advance the automaton several states — the index has to account for that.',
      '<b>At each decoding step</b>, look up the mask for the current state, add $-\\infty$ to every disallowed logit, sample, and advance the automaton.',
      '<b>Jump ahead where the automaton is deterministic.</b> If only one continuation is possible — the <code>":"</code> after a key, the closing brace — emit it without calling the model at all. For a rigid schema this can skip 30–50% of the forward passes, making constrained generation <i>faster</i> than unconstrained.'
    ])}
${H.table(['Library / feature', 'Approach', 'Note'], [
      ['Outlines', 'regex/JSON-Schema → FSM, precomputed index', 'the reference implementation of the index idea'],
      ['llguidance / Guidance', 'grammar with fast lexer, jump-ahead', 'very low overhead; used inside several servers'],
      ['XGrammar', 'pushdown automaton with a context-expansion cache', 'the current speed leader; integrated into vLLM and SGLang'],
      ['llama.cpp GBNF', 'BNF grammar files', 'the local-inference standard'],
      ['OpenAI Structured Outputs', 'constrained decoding server-side', 'guaranteed schema conformance on supported models'],
      ['Tool / function calling', 'the tool schema <i>is</i> the grammar', 'the same mechanism with a different name']
    ])}

${H.more('The tokenizer boundary problem, which is where the bugs live', `
<p>Constraints are defined over <i>characters</i>; decoding happens over <i>tokens</i>. A token like <code>": "</code> spans a schema boundary, and a token like <code>true</code> may or may not be a single token depending on the tokenizer. Two consequences:</p>
<ul>
<li>The allowed-token set for a state cannot be computed by simple string matching — you need to know which token strings are valid <i>prefixes</i> of continuations from that state. This is why the libraries build a trie over the vocabulary.</li>
<li><b>Constraining can change the distribution in ways that are not obvious.</b> Masking renormalises over the allowed set, so the model's relative preferences among allowed tokens are preserved — but the joint distribution over complete strings is <i>not</i> the conditional of the original distribution on the valid set. That difference is small in practice and real in theory, and it is why heavily constrained generation sometimes produces oddly stilted content.</li>
</ul>`)}

<h2><span class="sn">4.21.3</span> Design rules that matter more than the library</h2>
${H.checklist([
      '<b>Put a reasoning field first</b> in the schema if you need reasoning. The model cannot think after it has already emitted the answer — token order is causal, so <code>{"reasoning": "...", "answer": ...}</code> is strictly better than the reverse.',
      '<b>Use enums rather than free strings</b> wherever the value set is known. This is where constrained decoding pays for itself twice: valid by construction, and no downstream normalisation.',
      '<b>Avoid deeply nested optional structures.</b> Every optional branch is a place for the model to make a choice it has no information about.',
      '<b>Give fields descriptive names</b>: <code>confidence_0_to_1</code> beats <code>score</code>. The name is the only documentation the model gets.',
      '<b>Validate semantics separately.</b> The grammar guarantees the shape. Whether the date is in the future, the total equals the sum of the lines, or the cited document actually says that — none of it is checkable by a schema.'
    ])}
${H.pitfall('Forcing a schema on a model that cannot do the task produces <i>confidently well-formed</i> nonsense, which is worse than a parse error because it passes every automated check. Structured output raises the floor on format and does nothing whatever for correctness. Always keep a semantic validation layer, and always allow the model an explicit "I cannot determine this" value rather than forcing a guess.')}

${H.probe([
      ['How do you guarantee valid JSON from an LLM?', 'Constrained decoding: compile the schema to an automaton, mask disallowed tokens at every step. Prompting and retries are probabilistic; masking is a guarantee.'],
      ['Why can constrained decoding be faster than unconstrained?', 'Jump-ahead: where the automaton permits exactly one continuation, emit it without a forward pass. Rigid schemas skip a large fraction of the model calls.'],
      ['Where should a reasoning field go in the schema?', 'First. Generation is causal — tokens emitted after the answer cannot influence it.'],
      ['What does structured output <i>not</i> give you?', 'Semantic correctness. The shape is guaranteed; the content is exactly as reliable as the model. Keep an independent validation layer and an explicit "unknown" option.']
    ])}`,
    labs: {
      constrained: function (host) {
        const el = ML.el;
        /* a tiny JSON schema as an explicit automaton over "tokens" */
        const VOCAB = ['{', '}', '"', 'name', 'age', 'city', ':', ',', 'Ada', 'Bob', '31', '42', 'Paris', 'Oslo', ' the', ' is', 'Hello', '\\n'];
        const SEQ = [
          { allow: ['{'], state: 'start — the object must open' },
          { allow: ['"'], state: 'expecting a key' },
          { allow: ['name'], state: 'key: only "name" is valid first (schema order)' },
          { allow: ['"'], state: 'closing the key string' },
          { allow: [':'], state: 'separator — deterministic, can jump ahead' },
          { allow: ['"'], state: 'opening the value string' },
          { allow: ['Ada', 'Bob'], state: 'value: an enum of two names' },
          { allow: ['"'], state: 'closing the value' },
          { allow: [','], state: 'more fields required — deterministic' },
          { allow: ['"'], state: 'expecting the next key' },
          { allow: ['age'], state: 'key: "age" is next in the schema' },
          { allow: ['"'], state: 'closing the key' },
          { allow: [':'], state: 'separator — deterministic' },
          { allow: ['31', '42'], state: 'value: an integer (no quotes allowed here)' },
          { allow: ['}'], state: 'object closes — generation complete' }
        ];
        let step = 0;
        const stage = el('div');
        host.appendChild(stage);
        const st = Viz.controls(host, [
          { k: 'mode', label: 'decoding', type: 'select', value: 'on', options: [{ v: 'on', t: 'constrained (mask the logits)' }, { v: 'off', t: 'unconstrained (hope for the best)' }] },
          { k: 'temp', label: 'model temperature', min: .1, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => render(step));

        const S = Viz.surface(stage, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const sp = SEQ[Math.min(step, SEQ.length - 1)];
            const R = Num.rng(7 + step);
            /* "raw model preferences": deliberately not schema-aware */
            const logits = VOCAB.map((v, i) => R.normal(0, 1.6) + (sp.allow.indexOf(v) >= 0 ? 0.9 : 0));
            const masked = logits.map((l, i) => (st.mode === 'on' && sp.allow.indexOf(VOCAB[i]) < 0) ? -Infinity : l / st.temp);
            const probs = Num.softmax(masked.map(v => isFinite(v) ? v : -1e9));
            const items = VOCAB.map((v, i) => ({ k: v, v: probs[i], c: sp.allow.indexOf(v) >= 0 ? T.c3 : T.c2 }));
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 12, r: 60, t: 24, b: 14 } });
            P.hbars(items, { max: Math.max.apply(null, probs), maxH: 11, fmt: v => v < .001 ? '0' : (v * 100).toFixed(1) + '%' });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left';
            ctx.fillText(st.mode === 'on' ? 'green = permitted by the automaton; everything else is masked to −∞'
                                          : 'unconstrained: the model may pick anything', 12, 14);
          }
        });

        const info = el('div', { style: 'margin-top:10px' });
        stage.appendChild(info);

        function render(i) {
          step = i;
          const sp = SEQ[Math.min(i, SEQ.length - 1)];
          const emitted = SEQ.slice(0, i).map(s => s.allow[0]).join('');
          const deterministic = sp.allow.length === 1;
          info.innerHTML =
            '<div style="font-family:var(--mono);font-size:13px;background:var(--panel);border:1px solid var(--line);' +
            'border-radius:8px;padding:10px 12px;margin-bottom:8px;white-space:pre-wrap">' +
            ML.escapeHtml(emitted) + '<span style="background:var(--c3);color:var(--bg);padding:0 3px;border-radius:3px">▌</span></div>' +
            '<p class="small" style="margin:0"><b>Automaton state:</b> ' + sp.state +
            ' &nbsp;·&nbsp; <b>permitted:</b> <code>' + sp.allow.join('</code> <code>') + '</code>' +
            (deterministic ? ' &nbsp;·&nbsp; <span style="color:var(--green)"><b>only one option → jump ahead, no forward pass needed</b></span>' : '') +
            '</p>';
          S.redraw();
        }
        Viz.player(stage, {
          frames: SEQ.length, fps: 1.2, repeat: true,
          label: i => 'token ' + (i + 1) + ' / ' + SEQ.length,
          onFrame: render
        });
        Viz.note(host, 'Switch to <b>unconstrained</b> and watch the model happily assign 20% of its probability mass to <code>Hello</code> at a position where only <code>{</code> is legal. That is where your 3% JSON failure rate comes from. Count the steps marked <b>jump ahead</b>: six of fifteen positions are fully determined by the schema, so a constrained decoder can skip 40% of the forward passes — which is why it is often <i>faster</i> than unconstrained decoding, not slower.');
      }
    },
    quiz: [
      {
        q: 'Constrained decoding guarantees valid output by…',
        options: ['retrying until it parses', 'setting the logits of tokens that cannot continue a valid string to $-\\infty$', 'fine-tuning on the schema', 'post-processing the output'],
        answer: 1,
        why: 'The invalid continuations are removed from the distribution, so they cannot be sampled at any temperature.'
      },
      {
        q: 'Constrained decoding can be faster than unconstrained because…',
        options: ['the mask is cheaper than a softmax', 'positions where only one token is legal can be emitted without a forward pass', 'it uses a smaller vocabulary', 'it stops earlier'],
        answer: 1,
        why: 'Jump-ahead. For rigid schemas this skips a large fraction of the model calls.'
      },
      {
        q: 'In a JSON schema that needs reasoning, the reasoning field should be…',
        options: ['last, after the answer', 'first, before the answer', 'in a separate call', 'omitted'],
        answer: 1,
        why: 'Generation is causal: tokens produced after the answer cannot influence it.'
      },
      {
        q: 'Structured output guarantees…',
        options: ['correct content', 'valid shape only', 'both', 'neither'],
        answer: 1,
        why: 'A schema cannot check that the date is plausible or the total matches the lines. Keep a semantic validation layer and an explicit "unknown" value.'
      }
    ],
    cards: [
      { q: 'Constrained decoding, in one line', a: 'Compile the schema to an automaton; mask every token that cannot continue a valid string; sample; advance.' },
      { q: 'Jump-ahead', a: 'Where the automaton allows exactly one token, emit it without calling the model — 30–50% of positions for a rigid schema.' },
      { q: 'Schema design rules', a: 'Reasoning field first, enums over free strings, descriptive field names, shallow structure, explicit "unknown".' },
      { q: 'What structured output does not fix', a: 'Semantic correctness. Well-formed nonsense passes every automated check.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.22 */
  ML.section({
    id: 'speculative', track: 'llm', num: '4.22', level: 3,
    title: 'Speculative decoding and the rest of the latency budget',
    lede: 'Decoding is memory-bandwidth-bound: one token per pass over every weight. Speculative decoding breaks that by guessing several tokens cheaply and verifying them all in one pass — with the guarantee that the output distribution is exactly unchanged.',
    prereq: ['serving', 'kv-cache'],
    related: ['serving', 'kv-cache', 'compression'],
    html: `
${H.tldr([
      'A small draft model proposes $k$ tokens; the large model verifies all $k+1$ positions in <b>one</b> forward pass. A rejection-sampling correction makes the output distribution <b>provably identical</b> to sampling from the large model alone.',
      'Expected accepted tokens per verification: $\\frac{1-\\alpha^{k+1}}{1-\\alpha}$ for per-token acceptance $\\alpha$. At $\\alpha=0.8, k=4$ that is 3.36 tokens per pass — roughly a 2–3× speedup after draft cost.',
      'It only helps because decode is bandwidth-bound. Verification of $k+1$ positions costs almost the same as one, because the weights are read once either way.'
    ])}

<h2><span class="sn">4.22.1</span> Why one token per pass is so wasteful</h2>
${H.worked('the arithmetic of a decode step', `
<p>Decoding one token from a 70B model in bf16 reads 140 GB of weights. At 2 TB/s that is 70 ms — a floor of about 14 tokens/second — while performing only 140 GFLOPs, which the same card could do in under a millisecond.</p>
<p><b>The arithmetic intensity of decode is about 1 FLOP per byte; the hardware wants 200+.</b> The GPU is idle more than 99% of the time, waiting for memory. Every serving optimisation in this section and §4.14 is an attempt to get more useful work out of each pass over the weights: batching amortises across requests, quantization shrinks the bytes, and speculation amortises across <i>positions within one request</i>.</p>`)}

<h2><span class="sn">4.22.2</span> The algorithm, and why it is exact</h2>
${H.steps([
      '<b>Draft.</b> A small model (or an $n$-gram lookup, or extra heads on the same model) autoregressively proposes $k$ tokens $x_1..x_k$ with its own distributions $q(\\cdot)$.',
      '<b>Verify.</b> One forward pass of the large model over the prefix plus all $k$ drafted tokens gives $p(\\cdot)$ at every one of those positions simultaneously — because that is what a causal transformer does anyway.',
      '<b>Accept or reject, left to right.</b> Accept $x_i$ with probability $\\min\\left(1, \\frac{p(x_i)}{q(x_i)}\\right)$. On the first rejection, resample that position from the normalised residual $\\frac{\\max(0,\\,p-q)}{\\sum\\max(0,\\,p-q)}$ and discard the rest of the draft.',
      '<b>Bonus token.</b> If all $k$ are accepted, the verification pass already gave you the distribution for position $k+1$ — sample it free. So one pass yields between 1 and $k+1$ tokens.'
    ])}
${H.key('The accept/reject rule is exactly the rejection-sampling construction that makes the composite process sample from $p$. The output distribution is identical to running the large model alone — this is a theorem, not an approximation. Speculative decoding is a pure latency optimisation with zero quality cost, which is why it is on by default nearly everywhere.')}

${H.lab('spec', 'Speculative decoding, simulated token by token', 'Real acceptance sampling: watch drafts get accepted and rejected, and see the running speedup. The upper panel is the theory — expected tokens per pass — and the simulation converges to it.')}

${H.deriv('expected tokens per verification pass', [
      ['$P(\\text{first } i \\text{ accepted}) = \\alpha^i$', 'Assume each drafted token is accepted independently with probability $\\alpha$ — a simplification, since acceptance is correlated in practice, but it matches measurements closely.'],
      ['$\\mathbb{E}[\\text{accepted}] = \\sum_{i=0}^{k}\\alpha^i$', 'Count: you get at least $i$ tokens whenever the first $i$ were accepted, and the $k+1$-th is the free bonus token.'],
      ['$= \\dfrac{1-\\alpha^{k+1}}{1-\\alpha}$', 'Geometric series. At $\\alpha=0.8,k=4$: $(1-0.328)/0.2 = 3.36$ tokens per verification pass.'],
      ['$\\text{speedup} = \\dfrac{\\mathbb{E}[\\text{accepted}]}{1 + k\\,c}$', 'Divide by the cost: one verification pass plus $k$ draft passes at relative cost $c$. With a draft model 15% the size, $c=0.15$: $3.36/1.6 = 2.1\\times$. <b>There is an optimal $k$</b> — beyond it the draft cost grows faster than the acceptance benefit.']
    ])}

<h2><span class="sn">4.22.3</span> The variants, and the rest of the budget</h2>
${H.table(['Technique', 'Mechanism', 'Typical gain', 'Catch'], [
      ['Draft model', 'a small model of the same family', '2–3×', 'needs a well-aligned small model; two models to serve'],
      ['<b>Medusa / EAGLE</b>', 'extra prediction heads on the <i>same</i> model', '2–3.5×', 'no second model; needs training the heads'],
      ['Prompt lookup / n-gram', 'copy from the prompt when the model is quoting', '2–8× <b>on summarisation and editing</b>', 'nothing at all on creative generation'],
      ['Self-speculation', 'skip layers to draft', '~1.5×', 'lower acceptance'],
      ['<b>Continuous batching</b>', 'admit new requests as others finish, per step', '<b>2–4× throughput</b>', 'the single biggest serving win; orthogonal to everything here'],
      ['PagedAttention', 'KV cache in fixed pages, no contiguous reservation', '2–4× more concurrency', 'the vLLM contribution'],
      ['Prefix caching', 'reuse the KV of a shared system prompt', 'huge on repeated prefixes', 'cache invalidation and memory'],
      ['Chunked prefill', 'interleave prefill chunks with decode steps', 'better p99', 'scheduler complexity']
    ])}
${H.pitfall('Speculative decoding improves <b>latency at low batch size</b>. At high batch the GPU is already compute-saturated — the weights are being amortised across many requests — and speculation can <i>reduce</i> total throughput, because rejected draft tokens are wasted computation that displaces real work. The right question is always "am I optimising p50 latency for one user, or tokens per second across a thousand?" They pull in opposite directions.')}
${H.flag('Prompt-lookup decoding deserves more attention than it gets. For summarisation, editing, RAG-grounded answering and code refactoring — anywhere the output substantially quotes the input — a plain n-gram match against the prompt achieves very high acceptance with <i>no draft model and no training</i>. It is about twenty lines of code and it is frequently the largest single latency win available.')}

${H.probe([
      ['Does speculative decoding change the output distribution?', 'No. The accept/reject rule with residual resampling makes the composite process sample exactly from the target model’s distribution. It is provably lossless.'],
      ['Expected tokens per pass at $\\alpha = 0.8$, $k = 4$?', '$(1-0.8^5)/(1-0.8) = 3.36$, including the bonus token.'],
      ['Why does speculation help at all?', 'Decode is memory-bandwidth-bound: verifying $k+1$ positions reads the weights once, the same as verifying one. You are converting spare FLOPs into fewer weight reads.'],
      ['When would you turn it off?', 'At high batch size, where the GPU is already compute-bound and rejected drafts waste capacity. Also when the draft model is poorly aligned, since $\\alpha$ below about 0.6 makes the draft cost dominate.'],
      ['Your p99 latency is bad but p50 is fine. Where do you look?', 'Queueing and prefill, not decode: long prompts blocking the batch, no chunked prefill, and no admission control. Speculative decoding will not fix a scheduling problem.']
    ])}`,
    labs: {
      spec: function (host) {
        const st = Viz.controls(host, [
          { k: 'alpha', label: 'per-token acceptance α', min: .3, max: .98, step: .01, value: .8, fmt: v => v.toFixed(2) },
          { k: 'k', label: 'draft length k', min: 1, max: 12, step: 1, value: 4, fmt: v => v },
          { k: 'cost', label: 'draft cost (fraction of target)', min: .02, max: .5, step: .01, value: .15, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'exp', label: 'tokens per verification', cls: 'key' },
          { k: 'sp', label: 'speedup', cls: 'good' },
          { k: 'best', label: 'optimal k' },
          { k: 'sim', label: 'simulated (500 passes)' },
          { k: 'waste', label: 'drafted tokens wasted', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const hTop = h * .52;
            const P = Viz.plot(ctx, w, hTop, { xd: [1, 12], yd: [0, 4], pad: { l: 48, r: 14, t: 14, b: 26 } })
              .frame({ xlabel: '', ylabel: 'speedup', xticks: [1, 2, 4, 6, 8, 10, 12] });
            let bestK = 1, bestV = 0;
            P.clip(() => {
              [.6, .7, .8, .9].forEach((a, i) => {
                P.fn(kk => Num.specSpeedup(a, Math.round(kk), st.cost).speedup,
                  { color: [T.c1, T.c3, T.c4, T.c5][i], width: Math.abs(a - st.alpha) < .05 ? 3 : 1.3, alpha: Math.abs(a - st.alpha) < .05 ? 1 : .45, n: 60 });
              });
              P.fn(kk => Num.specSpeedup(st.alpha, Math.round(kk), st.cost).speedup, { color: T.c2, width: 2.6, n: 60 });
              for (let kk = 1; kk <= 12; kk++) {
                const v = Num.specSpeedup(st.alpha, kk, st.cost).speedup;
                if (v > bestV) { bestV = v; bestK = kk; }
              }
              P.vline(bestK, { color: T.c3, dash: [4, 3], label: 'optimal k' });
              P.dots([[st.k, Num.specSpeedup(st.alpha, st.k, st.cost).speedup]], { r: 5, color: T.c2, stroke: true, strokeWidth: 2 });
              P.hline(1, { color: T.faint, dash: [3, 3], width: 1 });
            });

            /* the simulation strip */
            ctx.save(); ctx.translate(0, hTop + 6);
            const R = Num.rng(23);
            const rows = 6, perRow = 26;
            const cw = (w - 24) / perRow, chh = (h - hTop - 40) / rows;
            let tokens = 0, passes = 0, wasted = 0;
            let col = 0, row = 0;
            for (let pass = 0; pass < rows * 6 && row < rows; pass++) {
              passes++;
              let acc = 0;
              while (acc < st.k && R() < st.alpha) acc++;
              const got = acc + 1;
              wasted += st.k - acc;
              for (let t = 0; t < st.k + 1; t++) {
                if (col >= perRow) { col = 0; row++; }
                if (row >= rows) break;
                const x = 12 + col * cw, y = row * chh + 4;
                const accepted = t < acc, bonus = t === acc && acc === st.k;
                const rejected = t === acc && acc < st.k;
                ctx.fillStyle = accepted ? T.c3 : bonus ? T.c1 : rejected ? T.c2 : T.line;
                ctx.globalAlpha = (t > acc) ? .25 : 1;
                ctx.fillRect(x, y, cw - 2, chh - 4);
                ctx.globalAlpha = 1;
                col++;
              }
              tokens += got;
              if (col > 0 && col < perRow) { col++; }
            }
            ctx.fillStyle = T.muted; ctx.font = '10.5px ui-sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('green = accepted draft · blue = free bonus token · red = rejection (rest of the draft discarded)', 12, h - hTop - 14);
            ctx.restore();

            const th = Num.specSpeedup(st.alpha, st.k, st.cost);
            let simTok = 0, simPass = 0;
            const R2 = Num.rng(77);
            for (let i = 0; i < 500; i++) { let a2 = 0; while (a2 < st.k && R2() < st.alpha) a2++; simTok += a2 + 1; simPass++; }
            out({
              exp: th.accepted.toFixed(3),
              sp: th.speedup.toFixed(2) + '×',
              best: 'k = ' + bestK + ' (' + bestV.toFixed(2) + '×)',
              sim: (simTok / simPass).toFixed(3) + ' tokens/pass',
              waste: ((1 - (th.accepted - 1) / Math.max(1, st.k)) * 100).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'The simulated tokens-per-pass converges to the closed form $\\frac{1-\\alpha^{k+1}}{1-\\alpha}$ — the theory and the dice agree. Now raise <b>k</b> past the optimum: the speedup curve turns over, because each extra drafted token costs a draft pass but is only reached with probability $\\alpha^k$. <b>At α = 0.7 the optimum is around k = 4; at α = 0.95 it is past 10.</b> This is why serving stacks tune the draft length per workload rather than fixing it.');
      }
    },
    quiz: [
      {
        q: 'Speculative decoding changes the model’s output distribution…',
        options: ['slightly, in exchange for speed', 'not at all — the accept/reject rule makes it provably exact', 'only at high temperature', 'only for the bonus token'],
        answer: 1,
        why: 'Rejection sampling with residual resampling is constructed precisely so the composite samples from the target distribution.'
      },
      {
        q: 'With $\\alpha = 0.8$ and $k = 4$, expected tokens per verification pass is…',
        options: ['4.0', '3.36', '2.4', '5.0'],
        answer: 1,
        why: '$(1-0.8^5)/(1-0.8) = 3.36$, counting the free bonus token.'
      },
      {
        q: 'Speculative decoding helps because decoding is…',
        options: ['compute-bound', 'memory-bandwidth-bound, so verifying k+1 positions costs about the same as one', 'network-bound', 'limited by the softmax'],
        answer: 1,
        why: 'The weights are read once per pass regardless of how many positions are verified.'
      },
      {
        q: 'At very high batch size, speculative decoding…',
        options: ['helps even more', 'can reduce throughput, because rejected drafts waste compute the GPU now needs', 'has no effect', 'becomes lossy'],
        answer: 1,
        why: 'Large batches already amortise the weight reads, so the GPU is compute-bound and wasted draft work displaces real work.'
      }
    ],
    cards: [
      { q: 'Speculative decoding in one line', a: 'Draft $k$ tokens cheaply, verify all of them in one target pass, accept with $\\min(1,p/q)$ and resample the residual on rejection.' },
      { q: 'Expected tokens per pass', a: '$(1-\\alpha^{k+1})/(1-\\alpha)$; divide by $1+kc$ for the speedup, which has an interior optimum in $k$.' },
      { q: 'Why decode is slow', a: 'Arithmetic intensity ≈ 1 FLOP/byte against hardware that wants 200+. 70B at bf16 reads 140 GB per token.' },
      { q: 'Prompt-lookup decoding', a: 'Draft by n-gram matching against the prompt. No model, no training, very high acceptance on summarisation and editing.' },
      { q: 'When speculation hurts', a: 'High batch size: the GPU is compute-bound and rejected drafts displace useful work.' }
    ]
  });
})();
