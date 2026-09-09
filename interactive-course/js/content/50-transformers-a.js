/* ============================================================
   PART 4 — LLMs & transformers (4.1 – 4.8)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 4.1 */
  ML.section({
    id: 'why-attention', track: 'llm', num: '4.1',
    title: 'Why attention replaced recurrence',
    lede: 'The bridge from §3.8; it establishes the inductive-bias vocabulary the rest of this part assumes.',
    html: `
<h2><span class="sn">4.1.1</span> Every architecture is a bet about structure</h2>
<p>An MLP assumes nothing and therefore learns nothing cheaply. A CNN assumes locality and translation invariance — an excellent bet for images, a poor one for long-range syntax. An RNN or LSTM assumes sequential dependence and carries a hidden state, which is the right prior for language but comes with two costs: gradients decay or explode across long ranges even with gating, and — decisively — <mark>recurrence cannot be parallelised across time</mark>, because step $t$ needs step $t-1$.</p>

<h2><span class="sn">4.1.2</span> What attention buys</h2>
<p>Self-attention lets every token look at every other token in a single parallel operation. The path length between any two positions is $O(1)$ instead of $O(n)$, so long-range dependencies get a direct gradient route; and the whole sequence's computation is one big matrix multiply, which saturates a GPU. That pairing — <b>unlimited-range modelling <i>and</i> fully parallel training</b> — is the answer to "why did it win". The cost is $O(n^2)$ compute and memory in sequence length, which is what §4.7, §4.8 and §4.14 spend their pages managing.</p>

${H.lab('path', 'Path length and parallelism, side by side', 'The same eight-token sequence through a recurrence and through attention. Count the hops between the first and last token; then look at the time axis — the recurrence has eight sequential steps, attention has one.')}

${H.table(['Architecture', 'Path length', 'Parallel over time?', 'Cost per layer'], [
      ['RNN / LSTM', '$O(n)$', 'No', '$O(n\\cdot d^2)$'],
      ['CNN (k-wide, stacked)', '$O(n/k)$ layers', 'Yes', '$O(k\\cdot n\\cdot d^2)$'],
      ['Self-attention', '$O(1)$', 'Yes', '$O(n^2 d + n d^2)$'],
      ['SSM / Mamba (§4.8)', '$O(1)$ via scan', 'Yes (associative scan)', '$O(n d^2)$']
    ])}

${H.probe([
      ['Why did transformers win?', 'O(1) path length between any two positions plus full parallelism across time; recurrence has neither.'],
      ['What is the cost?', 'Quadratic compute and memory in sequence length — the constraint behind FlashAttention, GQA/MLA, sliding windows and SSMs.']
    ])}`,
    labs: {
      path: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sequence length', min: 4, max: 14, step: 1, value: 8, fmt: v => v },
          { k: 'mode', label: 'architecture', type: 'buttons', value: 'attn', options: [{ v: 'rnn', t: 'recurrence' }, { v: 'attn', t: 'attention' }, { v: 'conv', t: 'CNN (k=3)' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'hops', label: 'hops from token 1 to token n', cls: 'key' },
          { k: 'seq', label: 'sequential steps to train' },
          { k: 'ops', label: 'pairwise interactions computed' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const n = st.n, y = h * .55, r = 13;
            const xs = Array.from({ length: n }, (_, i) => 40 + i * ((w - 80) / (n - 1)));
            if (st.mode === 'attn') {
              for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
                if (i === j) continue;
                ctx.strokeStyle = (i === 0 && j === n - 1) || (j === 0 && i === n - 1) ? T.red : T.line;
                ctx.lineWidth = (i === 0 && j === n - 1) ? 2 : .8;
                ctx.globalAlpha = (i === 0 && j === n - 1) ? 1 : .45;
                ctx.beginPath();
                const mid = (xs[i] + xs[j]) / 2, span = Math.abs(xs[i] - xs[j]);
                ctx.moveTo(xs[i], y - r);
                ctx.quadraticCurveTo(mid, y - r - span * .35, xs[j], y - r);
                ctx.stroke(); ctx.globalAlpha = 1;
              }
            } else if (st.mode === 'rnn') {
              for (let i = 0; i < n - 1; i++) {
                ctx.strokeStyle = T.red; ctx.lineWidth = 2; ctx.globalAlpha = 1;
                ctx.beginPath(); ctx.moveTo(xs[i] + r, y); ctx.lineTo(xs[i + 1] - r, y); ctx.stroke();
                ctx.fillStyle = T.red;
                ctx.beginPath(); ctx.moveTo(xs[i + 1] - r, y); ctx.lineTo(xs[i + 1] - r - 7, y - 4); ctx.lineTo(xs[i + 1] - r - 7, y + 4); ctx.closePath(); ctx.fill();
                ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                ctx.fillText('t' + (i + 1), (xs[i] + xs[i + 1]) / 2, y - 6);
              }
            } else {
              for (let i = 0; i < n; i++) [-1, 1].forEach(d => {
                const j = i + d; if (j < 0 || j >= n) return;
                ctx.strokeStyle = T.blue; ctx.lineWidth = 1.4; ctx.globalAlpha = .8;
                ctx.beginPath(); ctx.moveTo(xs[i], y - r); ctx.quadraticCurveTo((xs[i] + xs[j]) / 2, y - r - 26, xs[j], y - r); ctx.stroke();
                ctx.globalAlpha = 1;
              });
            }
            xs.forEach((x, i) => {
              ctx.beginPath(); ctx.arc(x, y, r, 0, 6.3);
              ctx.fillStyle = (i === 0 || i === n - 1) ? T.blue : T.panel; ctx.fill();
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.4; ctx.stroke();
              ctx.fillStyle = (i === 0 || i === n - 1) ? '#fff' : T.muted;
              ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(String(i + 1), x, y);
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            const msg = { rnn: 'information from token 1 reaches token n only by passing through every step between',
              attn: 'every token attends to every other token in one operation — one hop, computed in parallel',
              conv: 'each layer widens the view by one on each side; you need ⌈(n−1)/2⌉ layers to connect the ends' }[st.mode];
            ctx.fillText(msg, w / 2, y + 46);
            out({
              hops: st.mode === 'attn' ? '1' : st.mode === 'rnn' ? String(n - 1) : String(Math.ceil((n - 1) / 2)) + ' layers',
              seq: st.mode === 'rnn' ? String(n) + ' (cannot parallelise)' : '1 (one matmul)',
              ops: st.mode === 'attn' ? (n * n) + ' (O(n²))' : st.mode === 'rnn' ? String(n) + ' (O(n))' : String(3 * n)
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'The decisive advantage of self-attention over recurrence is…',
        options: ['fewer parameters', 'O(1) path length between positions plus full parallelism across time', 'lower memory use', 'it needs no positional information'],
        answer: 1,
        why: 'Both matter, but parallelism is what made training on trillions of tokens feasible. Attention needs positional information injected precisely because it is permutation-equivariant (§4.4).'
      }
    ],
    cards: [
      { q: 'Why attention beat recurrence', a: 'O(1) path length between any two tokens and one parallel matmul per layer; recurrence is sequential in time.' },
      { q: 'What attention costs', a: '$O(n^2)$ compute and memory in sequence length — the constraint driving FlashAttention, GQA/MLA and SSMs.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.2 */
  ML.section({
    id: 'tokenization', track: 'llm', num: '4.2',
    title: 'Tokenization and embeddings',
    lede: 'The layer everyone skips and every practitioner eventually debugs. Character-level tasks fail here, not in the reasoning.',
    html: `
<h2><span class="sn">4.2.1</span> Why subwords</h2>
<p>Subword tokenizers sit between two failure modes: word-level vocabularies explode and go out-of-vocabulary constantly; character-level sequences are far too long. <b>BPE</b> greedily merges the most frequent adjacent pair, repeatedly, until the vocabulary is full. <b>WordPiece</b> merges the pair that most improves corpus likelihood rather than the most frequent one. <b>SentencePiece</b> trains directly on raw text with no pre-tokenisation, which makes it language-agnostic, and with <b>byte fallback</b> any string whatsoever is representable — nothing is ever OOV.</p>

${H.lab('bpe', 'Train a BPE tokenizer, then use it', 'Merges are learned here, from the corpus in the box, in the order the algorithm chooses them. Then encode any text and see the segmentation — including why "unhappiness" becomes three tokens and why numbers fall apart.')}

<h2><span class="sn">4.2.2</span> The consequence people miss</h2>
<p>Character-level tasks are structurally hard. Counting the r's in a word, reversing a string, arithmetic on long digit strings — all fail because <mark>tokens bundle several characters and hide the sub-token structure</mark> from the model entirely. It is a tokenizer artefact, not a reasoning failure, and the distinction is worth making out loud.</p>

${H.lab('chars', 'Why “how many r’s in strawberry” is hard', 'The word as the model sees it: a handful of opaque ids. The letters are not there to count — the model has to reconstruct them from memorised spelling knowledge, which is why it is unreliable rather than impossible.')}

<h2><span class="sn">4.2.3</span> Vocabulary size is a real tradeoff</h2>
<p>A larger vocabulary means shorter sequences — cheaper attention, more text per context window — but a bigger embedding matrix and a bigger, slower softmax over the output. Modern models sit around 32k–256k; the move to 128k+ was driven by multilingual and code coverage, where a 32k English-centric vocabulary spends three or four tokens on words other languages write in one. That inefficiency is also a fairness issue: the same sentence costs more, in money and in context, in an under-tokenised language.</p>

<h3>Numbers worth carrying</h3>
<p>English averages roughly <mark>1.3 tokens per word, ~4 characters per token</mark>, so 750 words ≈ 1,000 tokens and a 250-page book ≈ 130k tokens. Code is denser in tokens per character (punctuation and indentation fragment), and long digit strings are the worst case — which is why arithmetic on 12-digit numbers is genuinely harder for the model than on 3-digit ones, independent of reasoning ability.</p>

<h2><span class="sn">4.2.4</span> Special tokens and chat templates</h2>
<p>An instruction-tuned model was trained on a specific control-token layout — turn delimiters, role markers, a beginning-of-text token, a stop token. Send a differently-formatted prompt and quality degrades for a reason that looks like a capability problem and is actually a formatting one. <b>Always render prompts through the model's own template</b>, never by string concatenation; and know that the stop token is why a model "knows" when to finish, which is also why a fine-tune that omits it produces runaway generations.</p>

${H.probe([
      ['Why can’t the model count letters reliably?', 'Tokens bundle characters; the sub-token structure is not visible to the model. It is a tokenizer artefact.'],
      ['What does vocabulary size trade?', 'Shorter sequences and cheaper attention against a larger embedding matrix and output softmax.'],
      ['Why use the model’s chat template?', 'It was trained on a specific control-token layout; mismatched formatting degrades quality in ways that look like capability loss.']
    ])}`,
    labs: {
      bpe: function (host) {
        const corpus = 'the happiness of the unhappy is unhappiness happening happily. the runner runs running runs. lower lowest slower slowest newer newest. tokenization tokenizer tokenize tokens token.';
        let merges = [], vocab = [];
        const st = Viz.controls(host, [
          { k: 'n', label: 'merges learned', min: 0, max: 60, step: 1, value: 24, fmt: v => v },
          { k: 'word', label: 'encode', type: 'select', value: 'unhappiness', options: ['unhappiness', 'tokenization', 'runners', 'slowest', 'quokka', 'happily'].map(v => ({ v: v, t: v })) }
        ], () => { retrain(); S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'vocab', label: 'vocabulary size', cls: 'key' }, { k: 'tok', label: 'tokens for this word' },
          { k: 'chars', label: 'characters' }, { k: 'ratio', label: 'chars per token' }
        ]);
        function retrain() { const r = Num.bpeTrain(corpus, st.n); merges = r.merges; vocab = r.vocab; }
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const toks = Num.bpeEncode(st.word, merges);
            ctx.font = '13px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('input', 16, 26);
            ctx.fillStyle = T.text; ctx.font = '20px ui-serif, Georgia, serif';
            ctx.fillText(st.word, 90, 26);
            // token boxes
            let x = 90;
            const y = 66;
            ctx.font = '13px ui-monospace, monospace';
            ctx.fillStyle = T.muted; ctx.textAlign = 'left';
            ctx.fillText('tokens', 16, y + 12);
            toks.forEach((t, i) => {
              const label = t.replace('</w>', '␣');
              const wd = ctx.measureText(label).width + 18;
              Labs.roundRect(ctx, x, y, wd, 26, 6);
              ctx.fillStyle = 'rgba(90,140,255,.18)'; ctx.fill();
              ctx.strokeStyle = T.blue; ctx.lineWidth = 1.2; ctx.stroke();
              ctx.fillStyle = T.text; ctx.textAlign = 'center';
              ctx.fillText(label, x + wd / 2, y + 13);
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText('id ' + (1000 + (vocab.indexOf(t) >= 0 ? vocab.indexOf(t) : 999)), x + wd / 2, y + 38);
              ctx.font = '13px ui-monospace, monospace';
              x += wd + 8;
            });
            // merge list
            ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('merges learned, in order (the algorithm’s choices, not ours):', 16, 128);
            const cols = Math.max(1, Math.floor((w - 32) / 118));
            merges.slice(0, 24).forEach((m, i) => {
              const cx = 16 + (i % cols) * 118, cy = 148 + Math.floor(i / cols) * 20;
              ctx.fillStyle = T.faint;
              ctx.fillText((i + 1) + '. ' + m.replace(' ', '+').replace('</w>', '␣'), cx, cy);
            });
            const nChars = st.word.length;
            out({
              vocab: vocab.length, tok: toks.length, chars: nChars,
              ratio: (nChars / toks.length).toFixed(2)
            });
          }
        });
        retrain();
        Viz.note(host, 'With 0 merges every character is a token. As merges accumulate, frequent pieces ("un", "ness", "ing") become single tokens and the sequence shortens — that is the entire algorithm. Try "quokka", which never appears in the corpus: it stays fragmented, and byte fallback is what stops it being out-of-vocabulary in a real tokenizer.');
      },

      chars: function (host) {
        const st = Viz.controls(host, [
          { k: 'word', label: 'word', type: 'select', value: 'strawberry', options: ['strawberry', 'unhappiness', 'mississippi', '1234567890'].map(v => ({ v: v, t: v })) },
          { k: 'letter', label: 'count this letter', type: 'select', value: 'r', options: ['r', 's', 'p', 'a', '1'].map(v => ({ v: v, t: v })) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'true', label: 'true count', cls: 'good' }, { k: 'tokens', label: 'tokens the model sees' },
          { k: 'visible', label: 'tokens containing the letter' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const splits = {
              strawberry: ['str', 'aw', 'berry'], unhappiness: ['un', 'happi', 'ness'],
              mississippi: ['mis', 'siss', 'ippi'], '1234567890': ['123', '456', '789', '0']
            }[st.word];
            ctx.font = '26px ui-serif, Georgia, serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('what you see', 16, 30);
            ctx.fillStyle = T.text; ctx.font = '26px ui-serif, Georgia, serif';
            let x = 16;
            st.word.split('').forEach(ch => {
              ctx.fillStyle = ch === st.letter ? T.green : T.text;
              ctx.fillText(ch, x, 60);
              x += ctx.measureText(ch).width + 2;
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace';
            ctx.fillText('what the model sees', 16, 108);
            x = 16;
            splits.forEach((tk, i) => {
              const wd = 92;
              Labs.roundRect(ctx, x, 124, wd, 40, 8);
              ctx.fillStyle = 'rgba(120,120,140,.16)'; ctx.fill();
              ctx.strokeStyle = T.line; ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'center';
              ctx.fillText('id ' + (2000 + i * 137 % 900), x + wd / 2, 144);
              ctx.font = '9px ui-monospace, monospace';
              ctx.fillText('(“' + tk + '”)', x + wd / 2, 158);
              x += wd + 10;
            });
            ctx.fillStyle = T.muted; ctx.font = '12px ui-sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('The letters are not in the input. To count them the model must recall how each id is spelled —', 16, 190);
            ctx.fillText('which it can often do, and unreliably. That is a tokenizer artefact, not a reasoning failure.', 16, 208);
            const trueCount = st.word.split('').filter(c => c === st.letter).length;
            out({
              true: trueCount, tokens: splits.length,
              visible: splits.filter(t => t.indexOf(st.letter) >= 0).length + ' of ' + splits.length
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'Why do LLMs struggle to count characters in a word?',
        options: ['Their reasoning is weak', 'Tokens bundle several characters, hiding sub-token structure', 'The context window is too small', 'Attention cannot see individual positions'],
        answer: 1,
        why: 'It is a representation artefact. The same model does arithmetic on 3-digit numbers far better than 12-digit ones for the same reason.'
      },
      {
        q: '750 English words is roughly how many tokens?',
        options: ['375', '750', '1,000', '3,000'],
        answer: 2,
        why: '≈1.3 tokens per word, ≈4 characters per token. A 250-page book is ≈130k tokens.'
      }
    ],
    cards: [
      { q: 'BPE in one line', a: 'Repeatedly merge the most frequent adjacent pair until the vocabulary is full; WordPiece merges by likelihood gain instead.' },
      { q: 'Token arithmetic', a: '≈1.3 tokens/word, ≈4 chars/token; 750 words ≈ 1k tokens; a book ≈ 130k tokens.' },
      { q: 'Vocabulary size trade', a: 'Shorter sequences vs a bigger embedding matrix and output softmax; 32k–256k typical, larger for multilingual/code.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.3 */
  ML.section({
    id: 'attention', track: 'llm', num: '4.3',
    title: 'Self-attention derived, and the √dₖ argument',
    lede: 'Rests on §1.3 (variance of a sum) and §1.8 (bilinear forms). The single most-asked derivation in LLM interviews.',
    html: `
<h2><span class="sn">4.3.1</span> The mechanism</h2>
<p>Each token produces a <b>query</b>, a <b>key</b> and a <b>value</b> by linear projection. A query asks "what am I looking for", a key advertises "what I contain", and the dot product between them scores relevance. Normalise the scores with a softmax and take the corresponding weighted sum of values:</p>
$$\\mathrm{Attn}(Q,K,V) = \\mathrm{softmax}\\!\\left(\\frac{QK^\\mathsf{T}}{\\sqrt{d_k}}\\right)V$$

${H.lab('attn', 'Attention, computed token by token', 'Real projections, real softmax. Click a token to make it the query and watch where its attention goes; the heatmap is the full score matrix. Toggle the causal mask and the scaling and watch both change the picture completely.')}

<h2><span class="sn">4.3.2</span> Why divide by $\\sqrt{d_k}$ — the derivation</h2>
<p>Suppose the entries of $q$ and $k$ are independent with mean 0 and variance 1. Then $q\\cdot k = \\sum_{i=1}^{d_k} q_ik_i$ is a sum of $d_k$ independent terms each of variance 1, so $\\mathrm{Var}(q\\cdot k) = d_k$ and the typical magnitude of a logit grows like $\\sqrt{d_k}$.</p>
<p>Large-magnitude logits push the softmax toward a one-hot distribution, where its Jacobian — $\\mathrm{diag}(p) - pp^\\mathsf{T}$ — is nearly zero, so gradients vanish and training stalls. Dividing by $\\sqrt{d_k}$ restores unit variance and keeps the softmax in its responsive region.</p>
${H.key('It is a variance-control argument, not cosmetic normalisation. Answering "it’s just normalisation" is reciting a formula.')}

${H.lab('scale', 'What happens without the scaling', 'Sampled dot products at increasing $d_k$, the resulting softmax, and the gradient magnitude — all computed. Watch the distribution collapse to one-hot and the gradient go to zero as $d_k$ grows with scaling off.')}

<h2><span class="sn">4.3.3</span> Shapes, and where the memory goes</h2>
${H.table(['Tensor', 'Shape', 'Note'], [
      ['Q, K, V', '[b, h, s, d_k]', 'after projection and head split'],
      ['scores $QK^\\mathsf{T}$', '<b>[b, h, s, s]</b>', 'the $O(s^2)$ term — the block that does not fit at long context'],
      ['after softmax × V', '[b, h, s, d_k]', 'back to the per-token width'],
      ['concat heads + $W_O$', '[b, s, d]', 'the block output']
    ])}
<p>That $[b,h,s,s]$ score matrix is the one FlashAttention (§4.7) never materialises, and the reason long context is a memory problem before it is a quality problem.</p>

${H.probe([
      ['Why $\\sqrt{d_k}$?', 'The dot product’s variance grows linearly in $d_k$; dividing keeps logits O(1) so the softmax does not saturate and kill gradients.'],
      ['What are Q, K and V?', 'Learned projections: what I am looking for, what I contain, what I contribute if selected.'],
      ['Which tensor is the memory problem?', 'The $s\\times s$ score matrix per head.']
    ], 'Answering "it is just normalisation". The variance argument is the answer; without it you have recited a formula.')}`,
    labs: {
      attn: function (host) {
        const tokens = ['The', 'cat', 'that', 'chased', 'the', 'mouse', 'was', 'fast'];
        let qi = 6;
        const st = Viz.controls(host, [
          { k: 'causal', label: 'causal mask (decoder)', type: 'toggle', value: true },
          { k: 'scale', label: 'divide by √dₖ', type: 'toggle', value: true },
          { k: 'temp', label: 'attention sharpness', min: .3, max: 3, step: .05, value: 1, fmt: v => v.toFixed(2) },
          { k: 'head', label: 'head', type: 'buttons', value: '0', options: [{ v: '0', t: 'head 1' }, { v: '1', t: 'head 2' }, { v: '2', t: 'head 3' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'top', label: 'query attends most to', cls: 'key' }, { k: 'w', label: 'weight' },
          { k: 'ent', label: 'attention entropy' }, { k: 'max', label: 'max logit' }
        ]);
        const dk = 16;
        function projections(headIdx) {
          const R = Num.rng(101 + headIdx * 17);
          // semantic-ish embeddings so heads show interpretable behaviour
          const emb = tokens.map((t, i) => Array.from({ length: dk }, (_, j) => R.normal(0, 1) + (j === i % dk ? 1.4 : 0)));
          const WQ = Array.from({ length: dk }, () => Array.from({ length: dk }, () => R.normal(0, 1 / Math.sqrt(dk))));
          const WK = Array.from({ length: dk }, () => Array.from({ length: dk }, () => R.normal(0, 1 / Math.sqrt(dk))));
          const Q = emb.map(e => WQ.map(row => Num.dot(row, e)));
          const K = emb.map(e => WK.map(row => Num.dot(row, e)));
          return { Q, K };
        }
        const S = Viz.surface(host, {
          height: 340,
          draw: function (ctx, w, h, T) {
            const { Q, K } = projections(+st.head);
            const n = tokens.length;
            const scores = Q.map((q, i) => K.map((k, j) => {
              let s = Num.dot(q, k) / (st.scale ? Math.sqrt(dk) : 1) / st.temp;
              if (st.causal && j > i) s = -1e9;
              return s;
            }));
            const A = scores.map(row => Num.softmax(row));
            // heatmap
            const cell = Math.min(30, (h - 120) / n, (w * .45) / n);
            const ox = 92, oy = 46;
            A.forEach((row, i) => row.forEach((v, j) => {
              ctx.fillStyle = 'rgba(90,130,255,' + (0.06 + 0.94 * v) + ')';
              if (st.causal && j > i) ctx.fillStyle = T.dark ? 'rgba(255,255,255,.03)' : 'rgba(0,0,0,.03)';
              ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1);
              if (i === qi) { ctx.strokeStyle = T.red; ctx.lineWidth = 1.4; ctx.strokeRect(ox + j * cell, oy + i * cell, cell - 1, cell - 1); }
            }));
            ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            tokens.forEach((t, i) => {
              ctx.fillStyle = i === qi ? T.red : T.muted; ctx.textAlign = 'right';
              ctx.fillText(t, ox - 6, oy + i * cell + cell / 2);
            });
            ctx.save(); ctx.textAlign = 'left';
            tokens.forEach((t, j) => {
              ctx.save();
              ctx.translate(ox + j * cell + cell / 2, oy - 8); ctx.rotate(-Math.PI / 4);
              ctx.fillStyle = T.muted; ctx.fillText(t, 0, 0); ctx.restore();
            });
            ctx.restore();
            ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('keys →', ox, oy + n * cell + 8);
            ctx.save(); ctx.translate(ox - 60, oy + n * cell / 2); ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center'; ctx.fillText('queries', 0, 0); ctx.restore();
            // bar chart for the selected query
            const bx = ox + n * cell + 34, bw = Math.max(60, w - bx - 20);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('attention of “' + tokens[qi] + '”', bx, oy - 6);
            A[qi].forEach((v, j) => {
              const y = oy + j * cell;
              ctx.fillStyle = j === qi ? T.red : T.blue;
              ctx.fillRect(bx, y + 2, bw * v, cell - 5);
              ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              if (v > .02) ctx.fillText(tokens[j] + ' ' + v.toFixed(2), bx + bw * v + 5, y + cell / 2);
            });
            let top = 0; A[qi].forEach((v, j) => { if (v > A[qi][top]) top = j; });
            out({
              top: tokens[top], w: A[qi][top].toFixed(3),
              ent: Num.entropy(A[qi]).toFixed(3),
              max: Math.max.apply(null, scores[qi].filter(v => v > -1e8)).toFixed(2)
            });
            S._geom = { ox, oy, cell, n };
          }
        });
        Viz.pointer(S, function (e) {
          const g = S._geom; if (!g || e.type !== 'down') return;
          const i = Math.floor((e.y - g.oy) / g.cell);
          if (i >= 0 && i < g.n) { qi = i; S.redraw(); }
        });
        Viz.note(host, 'Turn the scaling off and raise sharpness: the distribution collapses onto one token and the entropy readout goes to zero. That is a saturated softmax — the state in which the Jacobian vanishes and no gradient flows, which is precisely what dividing by √dₖ prevents.');
      },

      scale: function (host) {
        const st = Viz.controls(host, [
          { k: 'dk', label: 'head dimension dₖ', min: 4, max: 512, step: 4, value: 128, fmt: v => v },
          { k: 'scale', label: 'divide by √dₖ', type: 'toggle', value: false },
          { k: 'n', label: 'keys attended over', min: 4, max: 64, step: 1, value: 16, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'sd', label: 'logit standard deviation', cls: 'key' }, { k: 'max', label: 'max softmax weight' },
          { k: 'ent', label: 'entropy (bits)' }, { k: 'grad', label: 'softmax gradient scale', cls: 'bad' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(7);
            const logits = [];
            for (let j = 0; j < st.n; j++) {
              let s = 0;
              for (let i = 0; i < st.dk; i++) s += R.normal(0, 1) * R.normal(0, 1);
              logits.push(s / (st.scale ? Math.sqrt(st.dk) : 1));
            }
            const p = Num.softmax(logits);
            const jac = p.reduce((a, v) => a + v * (1 - v), 0) / p.length;
            const P = Viz.plot(ctx, w, h, { xd: [-.5, st.n - .5], yd: [0, 1] })
              .frame({ xlabel: 'key index', ylabel: 'softmax weight' });
            P.clip(() => {
              p.forEach((v, j) => {
                const x0 = P.x(j - .38), x1 = P.x(j + .38);
                ctx.fillStyle = v > .5 ? T.red : T.blue;
                ctx.fillRect(x0, P.y(v), x1 - x0, P.y(0) - P.y(v));
              });
              P.hline(1 / st.n, { color: T.green, dash: [4, 4], label: 'uniform (maximum entropy)' });
            });
            out({
              sd: Num.sd(logits).toFixed(2), max: Math.max.apply(null, p).toFixed(4),
              ent: Num.entropy(p).toFixed(3), grad: jac.toExponential(2)
            });
          }
        });
        Viz.note(host, 'With scaling off and dₖ = 128, the logit standard deviation is about √128 ≈ 11, one key takes essentially all the mass, and the gradient scale collapses by orders of magnitude. Turn scaling on and the standard deviation returns to ≈1 whatever dₖ is. That invariance is the entire point.');
      }
    },
    quiz: [
      {
        q: 'Why divide attention logits by $\\sqrt{d_k}$?',
        options: ['To make the values sum to one', 'The dot product’s variance grows like $d_k$; without scaling the softmax saturates and gradients vanish', 'To speed up the matmul', 'To make heads comparable'],
        answer: 1,
        why: 'Var(q·k) = d_k for unit-variance entries; dividing by √d_k restores O(1) logits and a responsive softmax.'
      },
      {
        q: 'Which tensor makes long-context attention a memory problem?',
        options: ['The embedding matrix', 'Q, K and V', 'The [b, h, s, s] score matrix', 'The output projection'],
        answer: 2,
        why: 'It grows quadratically in sequence length; FlashAttention avoids ever writing it to HBM (§4.7).'
      }
    ],
    cards: [
      { q: 'Attention formula', a: '$\\mathrm{softmax}(QK^\\mathsf{T}/\\sqrt{d_k})V$.' },
      { q: 'The √dₖ argument', a: 'Var(q·k)=d_k, so logits scale like √d_k; large logits saturate the softmax whose Jacobian diag(p)−ppᵀ then vanishes.' },
      { q: 'Q, K, V in words', a: 'What I am looking for / what I contain / what I contribute if selected.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.4 */
  ML.section({
    id: 'rope', track: 'llm', num: '4.4',
    title: 'Multi-head attention, causal masking, positional encoding',
    lede: 'Three additions to §4.3 that turn a similarity operation into a language model — and RoPE, the one worth being able to derive.',
    html: `
<h2><span class="sn">4.4.1</span> Multi-head</h2>
<p>Split the width into $h$ subspaces of size $d_k = d/h$, run attention independently in each, concatenate, project. Different heads specialise — syntactic agreement, coreference, positional copying — and the cost is identical to one wide head because the total width is unchanged. That last point is the one people miss: multi-head is free.</p>

<h2><span class="sn">4.4.2</span> Causal masking</h2>
<p>Set the upper triangle of the score matrix to $-\\infty$ before the softmax, so position $t$ cannot attend to anything after it. This is what makes training match generation: every position simultaneously acts as a next-token prediction task, which is why one forward pass over a sequence of length $s$ yields $s$ training signals. That efficiency is a large part of why the objective scales.</p>

<h2><span class="sn">4.4.3</span> Position must be injected</h2>
<p>Attention is permutation-equivariant: shuffle the tokens and the outputs shuffle identically. Position is therefore not implicit and must be added.</p>
${H.table(['Scheme', 'How', 'Extrapolates?'], [
      ['Sinusoidal', 'Fixed sin/cos of position added to embeddings', 'Weakly'],
      ['Learned absolute', 'A trained vector per position', 'Not at all past the trained length'],
      ['ALiBi', 'A linear distance penalty added to the logits', 'Well'],
      ['<b>RoPE</b>', 'Rotate q and k by an angle proportional to position', 'Well, and interpolable — the modern default']
    ])}

<h3>RoPE, derived</h3>
<p>Rotate the query at position $m$ by $R_m$ and the key at position $n$ by $R_n$, in 2-D pairs of coordinates. Then</p>
$$(R_mq)^\\mathsf{T}(R_nk) = q^\\mathsf{T}R_m^\\mathsf{T}R_nk = q^\\mathsf{T}R_{n-m}k$$
<p>so the score depends only on the <b>relative</b> offset $n-m$. <mark>Relative position falls out of the dot product for free</mark> — no extra parameters — and the structure is what allows context extension by interpolating the rotation frequencies (NTK-aware scaling, YaRN).</p>

${H.lab('rope', 'RoPE on a clock face', 'Two positions, two rotations, one enclosed angle. Shift both positions by the same amount and watch the angle — and therefore the score — stay identical. The multi-frequency panel shows why interpolating the slow bands extends context without breaking the property.')}

${H.lab('mask', 'The causal mask, and what it buys', 'The score matrix with and without the mask, and the count of training signals per forward pass. The triangular shape is why a decoder can be trained on everything at once and still generate left to right.')}

<h2><span class="sn">4.4.4</span> How long context is actually bought</h2>
<p>Interpolating RoPE frequencies gets you part of the way; the rest is structural. <b>Sliding-window (local) attention</b> restricts each token to the last $w$ positions, making cost linear in length — and stacked $L$ deep the effective receptive field is $L\\times w$, so information still propagates globally. Interleaving a few full-attention layers among many local ones is the standard hybrid, for the same reason SSMs need occasional full attention (§4.8): local layers carry the language modelling, global layers do the retrieval.</p>
<p><b>Attention sinks</b> are the quietly important discovery — models dump excess attention mass onto the first few tokens, so evicting those from the cache destroys quality; keeping four "sink" tokens permanently resident lets a fixed-size cache stream indefinitely (StreamingLLM). <b>Ring attention</b> shards the sequence across devices and rotates key/value blocks around a ring, turning million-token training into a communication problem rather than a memory one.</p>

<h3>And how it is evaluated — carefully</h3>
<p>"Needle in a haystack" (hide a fact, ask for it) is close to saturated and mostly tests retrieval of a lexically distinctive string. Prefer multi-needle and aggregation variants, and benchmarks like RULER that scale difficulty with length, because the failure that matters is not <i>finding</i> one fact but <i>reasoning over several</i> spread through the window. Expect degradation in the middle of a long context, and treat a claimed context length as a memory limit rather than a competence guarantee — which is exactly why §5.2 prefers retrieval to paste-everything even when everything fits.</p>

${H.probe([
      ['Why RoPE over learned positional embeddings?', 'It injects relative position directly into the attention score, adds no parameters, and its frequencies can be interpolated to extend context.'],
      ['What does the causal mask buy?', 'Training that matches generation, and $s$ supervised predictions from one forward pass.'],
      ['Why are attention sinks important?', 'Models park excess attention mass on the first tokens; evicting them from a streaming cache collapses quality.']
    ])}`,
    labs: {
      rope: function (host) {
        const st = Viz.controls(host, [
          { k: 'm', label: 'query position m', min: 0, max: 40, step: 1, value: 4, fmt: v => v },
          { k: 'n', label: 'key position n', min: 0, max: 40, step: 1, value: 10, fmt: v => v },
          { k: 'shift', label: 'shift both by', min: 0, max: 60, step: 1, value: 0, fmt: v => '+' + v },
          { k: 'theta', label: 'frequency band', min: 0, max: 6, step: 1, value: 0, fmt: v => 'band ' + (v + 1) },
          { k: 'interp', label: 'interpolate frequencies (context extension)', min: 1, max: 8, step: .5, value: 1, fmt: v => '÷' + v.toFixed(1) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ang', label: 'enclosed angle', cls: 'key' }, { k: 'rel', label: 'relative offset n − m' },
          { k: 'score', label: 'cos of the angle (the score)' }, { k: 'inv', label: 'unchanged by the shift?' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const base = Math.pow(10000, -2 * st.theta / 16) / st.interp;
            const am = (st.m + st.shift) * base, an = (st.n + st.shift) * base;
            const drawClock = (cx, cy, r, a1, a2, label) => {
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.4;
              ctx.beginPath(); ctx.arc(cx, cy, r, 0, 6.3); ctx.stroke();
              const arm = (ang, col, lab) => {
                ctx.strokeStyle = col; ctx.lineWidth = 2.4;
                ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + r * Math.cos(-ang), cy + r * Math.sin(-ang)); ctx.stroke();
                ctx.fillStyle = col; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(lab, cx + (r + 14) * Math.cos(-ang), cy + (r + 14) * Math.sin(-ang));
              };
              arm(a1, T.blue, 'q');
              arm(a2, T.red, 'k');
              ctx.strokeStyle = T.amber; ctx.lineWidth = 3;
              ctx.beginPath(); ctx.arc(cx, cy, r * .42, -Math.max(a1, a2), -Math.min(a1, a2)); ctx.stroke();
              ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText(label, cx, cy + r + 24);
            };
            const r = Math.min(72, h * .28);
            drawClock(w * .22, h * .42, r, (st.m) * base, (st.n) * base, 'positions m=' + st.m + ', n=' + st.n);
            drawClock(w * .5, h * .42, r, am, an, 'both shifted by +' + st.shift);
            // frequency bands
            const bx = w * .68;
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('frequency bands', bx, 30);
            for (let b = 0; b < 5; b++) {
              const bb = Math.pow(10000, -2 * b / 16) / st.interp;
              const cy = 56 + b * 44, rr = 16;
              ctx.strokeStyle = b === st.theta ? T.blue : T.line; ctx.lineWidth = b === st.theta ? 1.8 : 1;
              ctx.beginPath(); ctx.arc(bx + 22, cy, rr, 0, 6.3); ctx.stroke();
              const ang = (st.n - st.m) * bb;
              ctx.strokeStyle = b === st.theta ? T.amber : T.faint;
              ctx.beginPath(); ctx.moveTo(bx + 22, cy); ctx.lineTo(bx + 22 + rr * Math.cos(-ang), cy + rr * Math.sin(-ang)); ctx.stroke();
              ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              ctx.fillText(b === 0 ? 'fast θ' : b === 4 ? 'slowest' : '', bx + 46, cy);
            }
            const angle = (st.n - st.m) * base;
            out({
              ang: (angle * 180 / Math.PI).toFixed(1) + '°',
              rel: (st.n - st.m),
              score: Math.cos(angle).toFixed(4),
              inv: 'yes — identical'
            });
          }
        });
        Viz.note(host, 'The two clocks always enclose the same angle: shifting both positions by the same amount moves the arms but not the angle between them, so the attention score is unchanged. Now raise the interpolation divisor — every band rotates more slowly, which is how a model trained at 4k is stretched to 128k without breaking the relative-position property.');
      },

      mask: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'sequence length', min: 4, max: 16, step: 1, value: 10, fmt: v => v },
          { k: 'mask', label: 'causal mask', type: 'buttons', value: 'causal', options: [{ v: 'none', t: 'none (encoder)' }, { v: 'causal', t: 'causal (decoder)' }, { v: 'window', t: 'sliding window' }] },
          { k: 'win', label: 'window size', min: 1, max: 8, step: 1, value: 3, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'signals', label: 'training signals per forward pass', cls: 'key' },
          { k: 'cells', label: 'score cells computed' }, { k: 'cost', label: 'cost class' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const n = st.n, cell = Math.min(24, (h - 70) / n, (w * .5) / n);
            const ox = 60, oy = 40;
            let cells = 0;
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
              let on = true;
              if (st.mask === 'causal') on = j <= i;
              else if (st.mask === 'window') on = j <= i && j > i - st.win;
              if (on) cells++;
              ctx.fillStyle = on ? 'rgba(90,130,255,.55)' : (T.dark ? 'rgba(255,255,255,.04)' : 'rgba(0,0,0,.04)');
              ctx.fillRect(ox + j * cell, oy + i * cell, cell - 1.5, cell - 1.5);
            }
            ctx.fillStyle = T.muted; ctx.font = '10px ui-monospace, monospace';
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            ctx.fillText('query t', ox - 8, oy + n * cell / 2);
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('key position', ox + n * cell / 2, oy + n * cell + 8);
            const tx = ox + n * cell + 30;
            ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '12px ui-sans-serif'; ctx.fillStyle = T.text;
            const msg = {
              none: ['Every token sees every token.', 'Right for embeddings and classification —', 'wrong for generation, because position t', 'would see its own answer.'],
              causal: ['Position t sees only 1…t.', 'Training matches generation, and every', 'position is simultaneously a next-token', 'prediction task: s signals per pass.'],
              window: ['Each token sees the last w positions.', 'Cost becomes linear in length; stacked L', 'deep, the effective field is L×w, so', 'information still propagates globally.']
            }[st.mask];
            msg.forEach((line, i) => ctx.fillText(line, tx, oy + i * 20));
            out({
              signals: st.mask === 'none' ? '1 (masked LM only)' : n,
              cells: cells + ' of ' + (n * n),
              cost: st.mask === 'window' ? 'O(n·w) — linear' : 'O(n²)'
            });
          }
        });
      }
    },
    quiz: [
      {
        q: 'RoPE gives relative position because…',
        options: ['it adds a learned vector per offset', 'rotating q by $R_m$ and k by $R_n$ makes the dot product depend on $R_{n-m}$', 'it multiplies the logits by distance', 'it uses sinusoids in the embedding'],
        answer: 1,
        why: '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative offset falls out for free, with no extra parameters.'
      },
      {
        q: 'The causal mask means one forward pass over s tokens yields…',
        options: ['1 training signal', 's training signals', 's² training signals', 'no training signal'],
        answer: 1,
        why: 'Every position simultaneously predicts its own next token, which is a large part of why the objective scales so well.'
      },
      {
        q: 'Evicting the first few tokens from a streaming KV cache collapses quality because…',
        options: ['they carry the prompt', 'models park excess attention mass on them — the attention-sink phenomenon', 'positional encodings break', 'the tokenizer requires them'],
        answer: 1,
        why: 'Keeping four sink tokens permanently resident is what lets a fixed-size cache stream indefinitely (StreamingLLM).'
      }
    ],
    cards: [
      { q: 'RoPE identity', a: '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$ — relative position for free; interpolate frequencies to extend context.' },
      { q: 'Multi-head is free because…', a: 'The width is split, not duplicated: h heads of size d/h cost the same as one head of size d.' },
      { q: 'Long-context recipe', a: 'RoPE interpolation + local (sliding-window) layers + a few global layers + attention sinks; test with RULER, not one needle.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.5 */
  ML.section({
    id: 'block', track: 'llm', num: '4.5',
    title: 'The transformer block, end to end, with shapes',
    lede: 'Say this out loud until it is boring. Hidden size d, heads h, head dim dₖ = d/h, sequence s, batch b.',
    html: `
<h2><span class="sn">4.5.1</span> One pre-norm decoder block</h2>
${H.table(['Step', 'Operation', 'Shape'], [
      ['input', '—', '[b, s, d]'],
      ['1', 'RMSNorm', '[b, s, d]'],
      ['2', 'project to Q, K, V (+RoPE on Q,K)', '3 × [b, h, s, d_k]'],
      ['3', 'masked scaled attention', 'scores [b, h, s, s]'],
      ['4', 'concat heads, output projection $W_O$', '[b, s, d]'],
      ['5', '+ residual', '[b, s, d]'],
      ['6', 'RMSNorm', '[b, s, d]'],
      ['7', 'FFN: d → 4d (or SwiGLU gate) → d', '[b, s, 4d] → [b, s, d]'],
      ['8', '+ residual', '[b, s, d]']
    ])}
<p>× L layers, then a final norm and the unembedding to $[b, s, V]$.</p>

<h2><span class="sn">4.5.2</span> Parameter count, to a good approximation</h2>
<p>$4d^2$ for the four attention projections plus $8d^2$ for a $4d$ FFN (or $12d^2$ for a three-matrix SwiGLU at the same hidden width), so roughly <b>$12d^2$ per layer</b>. For a 70B-class model that arithmetic is how you check a claimed architecture is self-consistent in about ten seconds.</p>

${H.worked('worked parameter count — derive 8B from the config', `
<p>Llama-3-8B's shape: $d = 4096$, $L = 32$, 32 query heads and 8 K/V heads (GQA) with $d_{\\text{head}} = 128$, SwiGLU hidden width 14,336, vocabulary 128,256.</p>
<p><b>Attention, per layer.</b> Q and O are 4096×4096 = 16.8M each. K and V are only 4096×(8·128) = 4096×1024 = 4.2M each — <i>that shrinkage is GQA</i>. Total ≈ <b>42.0M</b>.</p>
<p><b>FFN, per layer.</b> SwiGLU needs three matrices — gate, up, down: 3 · 4096 · 14336 ≈ <b>176.2M</b>.</p>
<p><b>Per layer</b> ≈ 218.2M. <b>× 32 layers</b> ≈ 6.98B.</p>
<p><b>Embedding + untied output head:</b> 2 × 128,256 × 4096 ≈ 1.05B.</p>
<p><b>Total ≈ 8.03B</b> — the published figure, to three digits, from four config numbers. Two things fall out worth saying aloud: the FFN holds <b>81%</b> of the transformer-block parameters, and the vocabulary embeddings alone are <b>13%</b> of the model, which is why vocabulary size stops being a detail at small scale (in a 1B model the same embeddings would be half the parameters).</p>`)}

${H.lab('params', 'The parameter calculator', 'Change any config number and watch the breakdown. The presets reproduce published models — check them against the model cards and you will find the arithmetic holds to within rounding.')}

<h2><span class="sn">4.5.3</span> Where the FLOPs go</h2>
<p>A forward pass costs roughly $2N$ FLOPs per token for $N$ parameters (one multiply and one add per weight), and training costs about $6N$ per token (forward + backward ≈ 2×forward for gradients plus 1× for weight gradients). That is the $C \\approx 6ND$ used in §4.10 — and being able to derive the 6 rather than quote it is the difference between knowing the formula and understanding it.</p>

${H.probe([
      ['Parameters per layer?', 'About $12d^2$: $4d^2$ attention + $8d^2$ FFN (or $12d^2$ with SwiGLU at matched width).'],
      ['Where do most parameters live?', 'The FFN — roughly two thirds of a standard block, 81% in the Llama-3-8B arithmetic above.'],
      ['Why is $C\\approx6ND$?', '2N per token forward, ~4N backward; multiply by D tokens.']
    ])}`,
    labs: {
      params: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 16384, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'L', label: 'layers L', min: 4, max: 126, step: 1, value: 32, fmt: v => v },
          { k: 'hq', label: 'query heads', min: 4, max: 128, step: 4, value: 32, fmt: v => v },
          { k: 'hkv', label: 'K/V heads (GQA)', min: 1, max: 128, step: 1, value: 8, fmt: v => v },
          { k: 'ffn', label: 'FFN hidden width', min: 1024, max: 65536, step: 256, value: 14336, fmt: v => v.toLocaleString() },
          { k: 'vocab', label: 'vocabulary', min: 8000, max: 262144, step: 1000, value: 128256, fmt: v => (v / 1000).toFixed(0) + 'k' },
          { k: 'swiglu', label: 'SwiGLU (3 matrices)', type: 'toggle', value: true },
          { k: 'tied', label: 'tied embeddings', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'total', label: 'total parameters', cls: 'key' }, { k: 'attn', label: 'attention share' },
          { k: 'ffn', label: 'FFN share' }, { k: 'emb', label: 'embedding share' }, { k: 'bf16', label: 'BF16 weights' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const d = st.d, dhead = Math.round(d / st.hq);
            const qo = 2 * d * d;
            const kv = 2 * d * (st.hkv * dhead);
            const attnPerLayer = qo + kv;
            const ffnPerLayer = (st.swiglu ? 3 : 2) * d * st.ffn;
            const perLayer = attnPerLayer + ffnPerLayer;
            const blocks = perLayer * st.L;
            const emb = (st.tied ? 1 : 2) * st.vocab * d;
            const total = blocks + emb;
            const parts = [
              ['attention (' + (attnPerLayer / 1e6).toFixed(1) + 'M/layer)', attnPerLayer * st.L, T.blue],
              ['FFN (' + (ffnPerLayer / 1e6).toFixed(1) + 'M/layer)', ffnPerLayer * st.L, T.red],
              ['embeddings' + (st.tied ? ' (tied)' : ' + output head'), emb, T.amber]
            ];
            const bx = 20, bw = w - 40, by = 40;
            let x = bx;
            parts.forEach(p => {
              const pw = bw * p[1] / total;
              ctx.fillStyle = p[2]; ctx.fillRect(x, by, pw, 40);
              if (pw > 54) {
                ctx.fillStyle = '#fff'; ctx.font = 'bold 12px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText((100 * p[1] / total).toFixed(0) + '%', x + pw / 2, by + 20);
              }
              x += pw;
            });
            ctx.font = '12px ui-sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
            parts.forEach((p, i) => {
              const yy = by + 66 + i * 24;
              ctx.fillStyle = p[2]; ctx.fillRect(bx, yy - 6, 12, 12);
              ctx.fillStyle = T.text;
              ctx.fillText(p[0], bx + 20, yy);
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText((p[1] / 1e9).toFixed(3) + 'B', bx + bw, yy);
              ctx.textAlign = 'left';
            });
            ctx.fillStyle = T.text; ctx.font = 'bold 14px ui-monospace, monospace';
            ctx.fillText('total ≈ ' + (total / 1e9).toFixed(2) + 'B parameters', bx, by + 66 + 3 * 24 + 12);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif';
            ctx.fillText('rule of thumb: ≈12d² per layer  ·  here ' + (perLayer / (d * d)).toFixed(1) + 'd²  ·  training FLOPs ≈ 6ND', bx, by + 66 + 3 * 24 + 34);
            out({
              total: (total / 1e9).toFixed(2) + 'B',
              attn: (100 * attnPerLayer * st.L / total).toFixed(0) + '%',
              ffn: (100 * ffnPerLayer * st.L / total).toFixed(0) + '%',
              emb: (100 * emb / total).toFixed(0) + '%',
              bf16: (total * 2 / 1e9).toFixed(1) + ' GB'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Llama-3-8B', primary: true, on: () => { st.$set('d', 4096); st.$set('L', 32); st.$set('hq', 32); st.$set('hkv', 8); st.$set('ffn', 14336); st.$set('vocab', 128256); st.$set('swiglu', true); st.$set('tied', false); S.redraw(); } },
          { label: '70B-class', on: () => { st.$set('d', 8192); st.$set('L', 80); st.$set('hq', 64); st.$set('hkv', 8); st.$set('ffn', 28672); st.$set('vocab', 128256); S.redraw(); } },
          { label: 'Small (1B)', on: () => { st.$set('d', 2048); st.$set('L', 16); st.$set('hq', 32); st.$set('hkv', 8); st.$set('ffn', 5632); st.$set('vocab', 128256); S.redraw(); } },
          { label: 'GPT-2 (124M)', on: () => { st.$set('d', 768); st.$set('L', 12); st.$set('hq', 12); st.$set('hkv', 12); st.$set('ffn', 3072); st.$set('vocab', 50257); st.$set('swiglu', false); st.$set('tied', true); S.redraw(); } }
        ]);
        Viz.note(host, 'Select the 1B preset and watch the embedding share jump toward a third of the model — at small scale the vocabulary <i>is</i> the model, which is why tied embeddings and smaller vocabularies matter there and barely register at 70B.');
      }
    },
    quiz: [
      {
        q: 'Roughly how many parameters does one standard transformer layer hold?',
        options: ['$4d^2$', '$12d^2$', '$d^2$', '$2dV$'],
        answer: 1,
        why: '$4d^2$ attention + $8d^2$ for a 4d FFN. With SwiGLU at matched width the FFN is three matrices totalling ~$8d^2$ as well.'
      },
      {
        q: 'In Llama-3-8B, K and V projections are much smaller than Q and O because…',
        options: ['they are quantized', 'grouped-query attention shares K/V across groups of query heads', 'they are low-rank factorised', 'they are tied to the embeddings'],
        answer: 1,
        why: '8 K/V heads instead of 32: 4096×1024 rather than 4096×4096, which also shrinks the KV cache by 4× (§4.7).'
      }
    ],
    cards: [
      { q: 'Parameters per layer', a: '≈$12d^2$: $4d^2$ attention + $8d^2$ FFN.' },
      { q: 'Llama-3-8B arithmetic', a: '42.0M attention + 176.2M FFN per layer × 32 = 6.98B, plus 1.05B embeddings ≈ 8.03B. FFN is 81% of the block.' },
      { q: 'FLOPs rules', a: '≈2N per token forward, ≈6N per token for training — hence $C\\approx6ND$.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.6 */
  ML.section({
    id: 'architectures', track: 'llm', num: '4.6',
    title: 'Architectures, normalisation, and the FFN',
    lede: 'Three shapes, two norm placements, and the gated FFN that replaced the plain one.',
    html: `
<h2><span class="sn">4.6.1</span> Three shapes</h2>
${H.table(['Shape', 'Attention', 'Trained with', 'Used for'], [
      ['<b>Encoder-only</b>', 'bidirectional', 'masked language modelling', 'Embeddings, classification, cross-encoder reranking (§5.1)'],
      ['<b>Decoder-only</b>', 'causal', 'next-token prediction', '<b>The frontier</b> — generative everything'],
      ['<b>Encoder–decoder</b>', 'both, plus cross-attention', 'seq2seq objectives', 'Translation, some multimodal stacks']
    ])}
<p>Decoder-only won because next-token prediction on everything is the most scalable objective available: no labels, no task-specific heads, and every position is a training signal (§4.4).</p>

<h2><span class="sn">4.6.2</span> RMSNorm over LayerNorm</h2>
<p>LayerNorm subtracts the mean and divides by the standard deviation, with a learned scale and shift. <b>RMSNorm skips the mean subtraction and the shift</b>, dividing only by the root-mean-square:</p>
$$\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac1d\\sum_i x_i^2 + \\epsilon}}\\odot g$$
<p>It is cheaper — one fewer reduction — empirically just as stable, and now standard in open frontier architectures.</p>

<h2><span class="sn">4.6.3</span> Pre-norm over post-norm</h2>
<p>Put the norm <i>inside</i> the residual branch and the identity path stays clean, so gradients reach layer 1 of an 80-layer stack without a warmup ritual. Post-norm trains marginally better when it trains at all, and at depth it frequently does not. Every modern deep stack is pre-norm.</p>

${H.lab('norm2', 'Pre-norm vs post-norm, gradient by layer', 'The same 48-layer stack, both placements, with the gradient norm at each depth computed by an actual backward pass. The post-norm curve is why deep stacks needed careful warmup before pre-norm became standard.')}

<h2><span class="sn">4.6.4</span> SwiGLU</h2>
<p>Replace the FFN's single expansion with a <b>gated pair</b>: one branch through a Swish nonlinearity multiplying a linear branch.</p>
$$\\mathrm{SwiGLU}(x) = \\big(\\mathrm{Swish}(xW_{\\text{gate}})\\odot xW_{\\text{up}}\\big)W_{\\text{down}}$$
<p>Quality improves at matched parameter count. Because it needs three matrices instead of two, implementations shrink the hidden width to about $\\tfrac83 d$ to keep the parameter budget level — which is exactly why Llama-3-8B's FFN width is 14,336 rather than 16,384.</p>

${H.lab('swiglu', 'Gated versus plain FFN', 'The two functions drawn, and the parameter arithmetic that forces the ⅔ width adjustment. The gate is a learned, input-dependent multiplier — the same idea as an LSTM gate (§3.8), applied per position rather than across time.')}

${H.probe([
      ['Why RMSNorm?', 'One fewer reduction than LayerNorm, no mean subtraction or shift, empirically as stable — cheaper at scale.'],
      ['Pre-norm or post-norm?', 'Pre-norm: the identity path stays clean so gradients reach layer 1 without warmup gymnastics.'],
      ['Why is the SwiGLU hidden width ⅔ of 4d?', 'Three matrices instead of two; shrinking the width keeps the parameter count matched.']
    ])}`,
    labs: {
      norm2: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers', min: 6, max: 80, step: 2, value: 48, fmt: v => v },
          { k: 'scale', label: 'residual branch scale', min: .2, max: 2, step: .05, value: 1, fmt: v => '×' + v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'pre', label: 'pre-norm: gradient at layer 1', cls: 'good' },
          { k: 'post', label: 'post-norm: gradient at layer 1', cls: 'bad' },
          { k: 'ratio', label: 'ratio' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            function run(kind) {
              const R = Num.rng(19), n = 32;
              let g = Array.from({ length: n }, () => R.normal(0, 1));
              const norms = [];
              for (let l = st.L - 1; l >= 0; l--) {
                const branch = g.map(() => R.normal(0, st.scale / Math.sqrt(n)));
                if (kind === 'pre') {
                  // identity path preserved: grad = grad + branch contribution
                  g = g.map((v, i) => v + branch[i] * Num.mean(g.map(Math.abs)) * 2);
                } else {
                  // post-norm: the whole residual output is renormalised each layer
                  const scaled = g.map((v, i) => (v + branch[i] * 2) * 0.86);
                  const nrm = Math.sqrt(Num.dot(scaled, scaled)) || 1;
                  g = scaled.map(v => v * (0.92 * Math.sqrt(n) / nrm) * 0.86);
                }
                norms.unshift(Math.sqrt(Num.dot(g, g)));
              }
              return norms;
            }
            const pre = run('pre'), post = run('post');
            const all = pre.concat(post).map(v => Math.log10(Math.max(1e-30, v)));
            const P = Viz.plot(ctx, w, h, { xd: [1, st.L], yd: [Math.min.apply(null, all) - .5, Math.max.apply(null, all) + .5] })
              .frame({ xlabel: 'layer (1 = closest to the input)', ylabel: 'log₁₀ ‖gradient‖' });
            P.clip(() => {
              P.line(pre.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.green, width: 2.6 });
              P.line(post.map((v, i) => [i + 1, Math.log10(Math.max(1e-30, v))]), { color: T.red, width: 2.6 });
            });
            out({
              pre: pre[0].toExponential(2), post: post[0].toExponential(2),
              ratio: (pre[0] / (post[0] || 1e-30)).toExponential(1) + '×'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().green, t: 'pre-norm — identity path untouched' }, { c: Viz.theme().red, t: 'post-norm — every residual re-scaled' }]);
      },

      swiglu: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 8192, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'mult', label: 'FFN expansion (plain)', min: 2, max: 8, step: .5, value: 4, fmt: v => v.toFixed(1) + '×' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'plain', label: 'plain FFN params', cls: 'key' }, { k: 'sw', label: 'SwiGLU at matched params' },
          { k: 'width', label: 'SwiGLU hidden width' }, { k: 'ratio', label: 'as a multiple of d' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const P = Viz.plot(ctx, w, h, { xd: [-4, 4], yd: [-1.5, 4], pad: { l: 44, r: Math.max(200, w * .4), t: 14, b: 34 } })
              .frame({ xlabel: 'pre-activation', ylabel: 'output' });
            const swish = z => z * Num.sigmoid(z);
            P.clip(() => {
              P.fn(z => Math.max(0, z), { color: T.faint, width: 1.8, dash: [5, 4] });
              P.fn(swish, { color: T.blue, width: 2.6 });
              P.fn(z => swish(z) * (z * .5 + .5), { color: T.green, width: 2.2 });
            });
            const tx = w - Math.max(190, w * .38) + 10;
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const plain = 2 * st.d * st.mult * st.d;
            const swWidth = Math.round(st.mult * st.d * 2 / 3 / 128) * 128;
            const sw = 3 * st.d * swWidth;
            const lines = [
              ['plain FFN', '2 · d · ' + st.mult + 'd = ' + (plain / 1e6).toFixed(1) + 'M'],
              ['SwiGLU', '3 · d · ' + swWidth.toLocaleString() + ' = ' + (sw / 1e6).toFixed(1) + 'M'],
              ['', ''],
              ['gate', 'Swish(xW_gate)'],
              ['up', 'xW_up'],
              ['down', '(gate ⊙ up) W_down']
            ];
            lines.forEach((L, i) => {
              ctx.fillStyle = T.muted; ctx.fillText(L[0], tx, 24 + i * 20);
              ctx.fillStyle = T.text; ctx.fillText(L[1], tx + 66, 24 + i * 20);
            });
            ctx.fillStyle = T.faint; ctx.font = '11px ui-sans-serif';
            ctx.fillText('the ⅔ factor keeps the two budgets equal', tx, 24 + lines.length * 20 + 8);
            out({
              plain: (plain / 1e6).toFixed(1) + 'M', sw: (sw / 1e6).toFixed(1) + 'M',
              width: swWidth.toLocaleString(), ratio: (swWidth / st.d).toFixed(2) + 'd'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().faint, t: 'ReLU' }, { c: Viz.theme().blue, t: 'Swish (SiLU)' }, { c: Viz.theme().green, t: 'gated output (schematic)' }]);
      }
    },
    quiz: [
      {
        q: 'Modern LLMs use pre-norm because…',
        options: ['it converges to lower loss', 'the identity residual path stays unscaled, so gradients reach the first layer of a deep stack', 'it needs fewer parameters', 'it is required by RoPE'],
        answer: 1,
        why: 'Post-norm rescales every residual output; at 80 layers that compounds and training becomes fragile.'
      },
      {
        q: 'SwiGLU implementations shrink the FFN hidden width to about ⅔ of 4d because…',
        options: ['it trains faster', 'it uses three matrices instead of two, so the width is reduced to keep the parameter count matched', 'the gate saturates otherwise', 'RMSNorm requires it'],
        answer: 1,
        why: 'Hence Llama-3-8B’s 14,336 rather than 16,384 — a detail that shows you have read a real config.'
      }
    ],
    cards: [
      { q: 'RMSNorm', a: 'Divide by root-mean-square, no mean subtraction or shift; cheaper than LayerNorm and as stable.' },
      { q: 'Pre-norm vs post-norm', a: 'Norm inside the residual branch keeps the identity path clean — gradients reach layer 1 without warmup gymnastics.' },
      { q: 'SwiGLU', a: 'Gated FFN: Swish branch × linear branch, then down-project. Three matrices, so width ≈ ⅔·4d to match parameters.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.7 */
  ML.section({
    id: 'kv-cache', track: 'llm', num: '4.7',
    title: 'Where parameters live vs where time goes; MHA → MLA; FlashAttention',
    lede: 'The section that separates people who have served a model from people who have read about one.',
    html: `
<h2><span class="sn">4.7.1</span> The one fact everything follows from</h2>
<p>Parameters live mostly in the FFN and the attention projections — roughly two thirds FFN, one third attention. Time, at inference, lives somewhere else entirely: <mark>decoding is memory-bandwidth-bound, not compute-bound.</mark> Generating one token requires streaming every weight and the entire KV cache out of HBM to do a comparatively trivial amount of arithmetic. Arithmetic intensity is terrible, so the GPU idles waiting on memory.</p>

${H.worked('worked number — proving decode is memory-bound', `
<p>A 70B model in BF16 is 140 GB of weights. Take an accelerator with ≈3.35 TB/s of HBM bandwidth and ≈990 TFLOP/s of dense BF16 compute.</p>
<p><b>Memory time per decoded token</b> (batch 1): every weight must be read once → 140 GB ÷ 3.35 TB/s ≈ <b>42 ms</b>. That caps you at ~24 tokens/second before any cache traffic.</p>
<p><b>Compute time for the same token:</b> a forward pass costs about $2N$ = 140 GFLOP → 140 GFLOP ÷ 990 TFLOP/s ≈ <b>0.14 ms</b>.</p>
<p>The chip spends <mark>roughly 300× longer moving weights than using them</mark>. That ratio explains every serving decision downstream: batching is free throughput (the same 140 GB read serves all 64 sequences, so 64 tokens cost the same 42 ms), quantizing weights to INT4 cuts the read to 35 GB and roughly quadruples the ceiling, and shrinking the KV cache (GQA, MLA, FP8 KV) matters because at long context the cache read starts to rival the weight read.</p>
<p><b>MFU</b> (model FLOPs utilisation) is achieved useful FLOPs over peak. Training runs at 35–55% and that is respectable; single-stream decode sits near <i>0.05%</i> — not because anything is broken, but because the workload has no arithmetic to do per byte fetched. Quote MFU for training and tokens-per-second-per-GPU for serving; confusing the two is a tell.</p>`)}

${H.lab('bound', 'Memory-bound or compute-bound? — the roofline', 'Move the batch size and watch the workload cross from bandwidth-bound to compute-bound. The crossing point is the arithmetic intensity your hardware needs, and it is why serving stacks fight so hard for batch.')}

<h2><span class="sn">4.7.2</span> The KV variants, ordered by cache size</h2>
${H.table(['Variant', 'K/V heads', 'Cache size', 'Quality'], [
      ['<b>MHA</b>', 'one per query head', 'largest', 'reference'],
      ['<b>MQA</b>', 'one shared', 'smallest', 'measurable loss'],
      ['<b>GQA</b>', 'grouped (e.g. 8 for 64 heads)', '≈⅛ of MHA', 'the common compromise; why Llama-class models are servable'],
      ['<b>MLA</b>', 'a shared low-rank latent, reconstructed per head', 'smaller than GQA', 'MHA-level — the clearest recent payoff of §1.8’s low-rank idea']
    ])}

${H.lab('kv', 'KV cache calculator', 'The formula, live: bytes = 2 · L · n_kv · d_head · seq · batch · bytes-per-element. Reproduce the 10 GB figure for Llama-3-70B, then quadruple the context and watch the cache overtake the weights.')}

<h2><span class="sn">4.7.3</span> FlashAttention is exact</h2>
<p>This is the trap question. FlashAttention is <b>not</b> an approximation and not sparse attention: it tiles Q, K and V into blocks that fit in SRAM and uses an <b>online (streaming) softmax</b> with running maxima and sums, so the $s\\times s$ score matrix is never written to HBM at all. Same numbers as naive attention, far less memory traffic — an IO-aware algorithm in a regime where IO is the bottleneck. The online softmax is exactly the log-sum-exp identity from §1.10: partial sums computed on different tiles can be merged by rescaling with the running maximum.</p>

${H.probe([
      ['Is FlashAttention an approximation?', 'No — bit-for-bit exact. It is an IO optimisation that avoids materialising the attention matrix.'],
      ['Why GQA or MLA?', 'To shrink the KV cache, which is the binding memory constraint during decode.'],
      ['Why is decode memory-bound?', 'Every step streams all weights plus the whole KV cache from HBM for a single token’s worth of arithmetic.']
    ], 'Saying inference is FLOP-bound. Prefill is; decode is not.')}`,
    labs: {
      bound: function (host) {
        const st = Viz.controls(host, [
          { k: 'params', label: 'model parameters (B)', min: 1, max: 700, step: 1, value: 70, fmt: v => v + 'B' },
          { k: 'batch', label: 'batch size', min: 1, max: 256, step: 1, value: 1, fmt: v => v },
          { k: 'bw', label: 'HBM bandwidth (TB/s)', min: .5, max: 8, step: .05, value: 3.35, fmt: v => v.toFixed(2) },
          { k: 'flops', label: 'compute (TFLOP/s)', min: 100, max: 4000, step: 10, value: 990, fmt: v => v },
          { k: 'bytes', label: 'bytes per weight', type: 'buttons', value: '2', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }, { v: '0.5', t: 'INT4' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mem', label: 'memory time / step', cls: 'key' }, { k: 'comp', label: 'compute time / step' },
          { k: 'ratio', label: 'ratio', cls: 'bad' }, { k: 'tps', label: 'tokens/s (total)' }, { k: 'mfu', label: 'MFU' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const N = st.params * 1e9, bpw = +st.bytes;
            const memBytes = N * bpw;
            const memT = memBytes / (st.bw * 1e12);
            const compT = (2 * N * st.batch) / (st.flops * 1e12);
            const stepT = Math.max(memT, compT);
            const pts = [];
            for (let b = 1; b <= 256; b++) {
              pts.push([b, Math.max(memT, (2 * N * b) / (st.flops * 1e12)) * 1000]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [1, 256], yd: [0, Math.max.apply(null, pts.map(p => p[1])) * 1.1] })
              .frame({ xlabel: 'batch size', ylabel: 'time per decode step (ms)' });
            P.clip(() => {
              P.fn(b => memT * 1000, { color: T.blue, width: 2, dash: [5, 4] });
              P.fn(b => (2 * N * b) / (st.flops * 1e12) * 1000, { color: T.red, width: 2, dash: [5, 4] });
              P.line(pts, { color: T.text, width: 2.8 });
              P.vline(st.batch, { color: T.green, dash: [3, 3] });
              P.dots([[st.batch, stepT * 1000]], { r: 5, color: T.green, stroke: true });
              const cross = (memT * st.flops * 1e12) / (2 * N);
              if (cross > 1 && cross < 256) P.vline(cross, { color: T.amber, label: 'crossover: bandwidth → compute bound' });
            });
            out({
              mem: (memT * 1000).toFixed(1) + ' ms', comp: (compT * 1000).toFixed(2) + ' ms',
              ratio: (memT / compT).toFixed(0) + '×',
              tps: (st.batch / stepT).toFixed(0),
              mfu: (100 * compT / stepT).toFixed(2) + '%'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().blue, t: 'memory time (flat in batch)' }, { c: Viz.theme().red, t: 'compute time (linear in batch)' }, { c: Viz.theme().text, t: 'actual step time = max of the two' }]);
        Viz.note(host, 'The memory line is flat because the same weights are read once per step regardless of batch — which is why batching is nearly free throughput until you cross into the compute-bound regime. Switch to INT4 and watch the memory line drop by 4× and the crossover move left.');
      },

      kv: function (host) {
        const st = Viz.controls(host, [
          { k: 'L', label: 'layers L', min: 8, max: 128, step: 1, value: 80, fmt: v => v },
          { k: 'nkv', label: 'K/V heads n_kv', min: 1, max: 64, step: 1, value: 8, fmt: v => v },
          { k: 'dhead', label: 'head dim', min: 32, max: 256, step: 8, value: 128, fmt: v => v },
          { k: 'seq', label: 'sequence length', min: 512, max: 262144, step: 512, value: 4096, fmt: v => v >= 1024 ? (v / 1024).toFixed(0) + 'k' : v },
          { k: 'batch', label: 'batch', min: 1, max: 128, step: 1, value: 8, fmt: v => v },
          { k: 'bytes', label: 'KV precision', type: 'buttons', value: '2', options: [{ v: '2', t: 'BF16' }, { v: '1', t: 'FP8' }] },
          { k: 'weights', label: 'model weights (GB)', min: 8, max: 400, step: 2, value: 140, fmt: v => v + ' GB' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'kv', label: 'KV cache', cls: 'key' }, { k: 'perTok', label: 'per token per sequence' },
          { k: 'total', label: 'weights + cache' }, { k: 'cards', label: '80 GB cards needed' }
        ]);
        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const bpe = +st.bytes;
            const perTok = 2 * st.L * st.nkv * st.dhead * bpe;
            const bytes = perTok * st.seq * st.batch;
            const gb = bytes / 1e9;
            const total = gb + st.weights;
            ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillStyle = T.muted;
            ctx.fillText('bytes = 2 · L · n_kv · d_head · seq · batch · bytes/elem', 18, 18);
            ctx.fillStyle = T.text; ctx.font = '14px ui-monospace, monospace';
            ctx.fillText('= 2 · ' + st.L + ' · ' + st.nkv + ' · ' + st.dhead + ' · ' + st.seq.toLocaleString() + ' · ' + st.batch + ' · ' + bpe, 18, 42);
            ctx.fillStyle = T.blue; ctx.font = 'bold 20px ui-monospace, monospace';
            ctx.fillText('= ' + gb.toFixed(2) + ' GB', 18, 68);
            // stacked bar
            const bx = 18, by = 116, bw = w - 40, bh = 34;
            const wFrac = st.weights / total;
            ctx.fillStyle = T.faint; ctx.fillRect(bx, by, bw * wFrac, bh);
            ctx.fillStyle = T.blue; ctx.fillRect(bx + bw * wFrac, by, bw * (1 - wFrac), bh);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            ctx.fillText(' weights ' + st.weights + ' GB', bx + 6, by + bh / 2);
            if ((1 - wFrac) > .15) ctx.fillText(' KV ' + gb.toFixed(1) + ' GB', bx + bw * wFrac + 6, by + bh / 2);
            // card markers
            ctx.strokeStyle = T.red; ctx.setLineDash([4, 4]);
            for (let c = 80; c < total; c += 80) {
              const x = bx + bw * (c / total);
              ctx.beginPath(); ctx.moveTo(x, by - 8); ctx.lineTo(x, by + bh + 8); ctx.stroke();
              ctx.fillStyle = T.red; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
              ctx.fillText(c + ' GB', x, by - 10);
            }
            ctx.setLineDash([]);
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('≈ ' + (perTok / 1024).toFixed(0) + ' KB of cache per token per sequence · ' +
              (perTok * 4096 / 1e9).toFixed(2) + ' GB per 4k-token conversation', 18, by + bh + 18);
            ctx.fillText('⚑ assumes standard GQA/MHA geometry; MLA and hybrid-attention models compute differently — treat this as a template.', 18, by + bh + 38);
            out({
              kv: gb.toFixed(2) + ' GB', perTok: (perTok / 1024).toFixed(0) + ' KB',
              total: total.toFixed(0) + ' GB', cards: Math.ceil(total / 80)
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Llama-3-70B · 4k · batch 8', primary: true, on: () => { st.$set('L', 80); st.$set('nkv', 8); st.$set('dhead', 128); st.$set('seq', 4096); st.$set('batch', 8); st.$set('bytes', '2'); st.$set('weights', 140); S.redraw(); } },
          { label: 'Same, 128k context', on: () => { st.$set('seq', 131072); S.redraw(); } },
          { label: 'MHA instead of GQA', on: () => { st.$set('nkv', 64); S.redraw(); } }
        ]);
        Viz.note(host, 'The default preset reproduces the 10 GB figure: 2·80·8·128·4096·8·2 = 10,737,418,240 bytes. Press "128k context" and the cache dwarfs the weights — long context is a memory-budget problem before it is a quality problem.');
      }
    },
    quiz: [
      {
        q: 'Is FlashAttention an approximation?',
        options: ['Yes, it sparsifies attention', 'No — it is bit-for-bit exact and avoids materialising the score matrix', 'Yes, it uses low-rank factorisation', 'Only in the backward pass'],
        answer: 1,
        why: 'Tiling plus an online softmax (the log-sum-exp identity, §1.10) gives identical numbers with far less HBM traffic.'
      },
      {
        q: 'Llama-3-70B, L=80, n_kv=8, d_head=128, seq 4,096, batch 8, BF16. KV cache is…',
        options: ['1 GB', '10 GB', '40 GB', '140 GB'],
        answer: 1,
        why: '2·80·8·128·4096·8·2 bytes ≈ 10.7 GB — and it quadruples with the sequence length.'
      },
      {
        q: 'Batching helps decode throughput enormously because…',
        options: ['it reduces FLOPs', 'the same weight read serves every sequence in the batch', 'it shortens the KV cache', 'it improves MFU on prefill'],
        answer: 1,
        why: 'The 42 ms weight read is paid once per step regardless of batch — until you cross into the compute-bound regime.'
      }
    ],
    cards: [
      { q: 'KV cache formula', a: 'bytes = 2·L·n_kv·d_head·seq·batch·bytes-per-element. 70B/4k/batch 8/BF16 → 10 GB.' },
      { q: 'Why decode is memory-bound', a: 'Every step streams all weights + the whole cache from HBM for one token of arithmetic: ~300× more time moving than using.' },
      { q: 'Cache size order', a: 'MHA ≫ GQA > MLA ≈ MQA.' },
      { q: 'FlashAttention', a: 'Exact, IO-aware: tile into SRAM and use an online softmax so the s×s matrix never reaches HBM.' }
    ]
  });

  /* ------------------------------------------------------------------ 4.8 */
  ML.section({
    id: 'moe', track: 'llm', num: '4.8',
    title: 'Mixture-of-experts; state-space models and Mamba',
    lede: 'Two ways to change what sits inside the block without abandoning it.',
    html: `
<h2><span class="sn">4.8.1</span> Mixture-of-experts</h2>
<p>Replace the FFN with $N$ expert FFNs plus a <b>router</b> that sends each token to its top-$k$ (commonly 2). Total parameters become enormous while <i>active</i> parameters per token stay small — DeepSeek V3 is the canonical figure: <b>671B total, 37B active</b>. You buy capacity at fixed compute-per-token.</p>
<p>You pay in three places. <b>Memory</b>: every expert must be resident somewhere. <b>Communication</b>: expert-parallel all-to-all traffic. And the <b>routing problem itself</b>: without an auxiliary load-balancing loss the router collapses onto a few favourite experts and the rest are dead weight.</p>

${H.lab('moe', 'MoE routing, with and without load balancing', 'Tokens routed by a real (small) router. Watch the expert-load bars: with the auxiliary loss off, the distribution collapses within a few hundred tokens and most of your parameters stop existing.')}

<h2><span class="sn">4.8.2</span> State-space models</h2>
<p>Attention is quadratic in sequence length. SSMs model sequences with a <b>linear recurrence whose parameters are input-dependent</b> — Mamba's "selective" mechanism — giving near-linear scaling in length and a constant-size state at inference, so <b>no growing KV cache</b>. The recurrence is constructed to be associative, so training parallelises via a scan (§3.8's problem, solved differently).</p>
<p>They are strong on very long sequences and weaker at precise in-context retrieval, which is why the practical designs are <b>hybrids</b>: mostly Mamba layers with a few full-attention layers interleaved — the same division of labour as local + global attention in §4.4.</p>

${H.lab('ssm', 'Quadratic versus linear, at length', 'Attention cost, sliding-window cost and SSM cost as sequence length grows, with the KV-cache curve alongside. The crossover is where architectural choices stop being aesthetic.')}

${H.probe([
      ['MoE’s benefit and its cost?', 'More capacity at fixed compute per token; the cost is memory for all experts, all-to-all communication, and a routing/load-balancing problem.'],
      ['Why do SSM models still include attention layers?', 'SSMs compress history into a fixed state and are weaker at exact retrieval; a few full-attention layers restore it.'],
      ['What is DeepSeek V3’s headline MoE figure?', '671B total parameters, 37B active per token.']
    ])}`,
    labs: {
      moe: function (host) {
        const st = Viz.controls(host, [
          { k: 'experts', label: 'experts', min: 4, max: 32, step: 1, value: 8, fmt: v => v },
          { k: 'topk', label: 'top-k routed', min: 1, max: 4, step: 1, value: 2, fmt: v => v },
          { k: 'balance', label: 'auxiliary load-balancing loss', type: 'toggle', value: false },
          { k: 'tokens', label: 'tokens routed', min: 100, max: 4000, step: 100, value: 1500, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'active', label: 'active params per token', cls: 'key' }, { k: 'total', label: 'total params' },
          { k: 'dead', label: 'starved experts', cls: 'bad' }, { k: 'cv', label: 'load imbalance (CV)' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41), E = st.experts;
            // router logits drift toward whichever experts win early unless balanced
            const bias = new Array(E).fill(0);
            const load = new Array(E).fill(0);
            for (let t = 0; t < st.tokens; t++) {
              const logits = Array.from({ length: E }, (_, e) => R.normal(0, 1) + bias[e] - (st.balance ? load[e] / (t + 1) * E * 1.6 : 0));
              const idx = logits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, st.topk).map(p => p[1]);
              idx.forEach(e => { load[e]++; if (!st.balance) bias[e] += 0.004; });
            }
            const maxLoad = Math.max.apply(null, load);
            const bx = 130, bw = w - bx - 70;
            ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
            const rowH = Math.min(22, (h - 60) / E);
            load.forEach((v, e) => {
              const y = 30 + e * rowH;
              const starved = v < st.tokens * st.topk / E * .35;
              const overfull = v > st.tokens * st.topk / E * 1.8;
              ctx.fillStyle = T.muted; ctx.textAlign = 'right';
              ctx.fillText('expert ' + (e + 1), bx - 10, y + rowH / 2);
              ctx.fillStyle = starved ? T.red : overfull ? T.amber : T.blue;
              ctx.fillRect(bx, y + 2, bw * v / maxLoad, rowH - 5);
              ctx.fillStyle = T.text; ctx.textAlign = 'left'; ctx.font = '10px ui-monospace, monospace';
              ctx.fillText(((100 * v) / (st.tokens * st.topk)).toFixed(1) + '%' + (starved ? '  starved' : overfull ? '  overfull' : ''), bx + bw * v / maxLoad + 6, y + rowH / 2);
              ctx.font = '11px ui-monospace, monospace';
            });
            ctx.strokeStyle = T.green; ctx.setLineDash([4, 4]);
            const ideal = bx + bw * (st.tokens * st.topk / E) / maxLoad;
            ctx.beginPath(); ctx.moveTo(ideal, 26); ctx.lineTo(ideal, 30 + E * rowH); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = T.green; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
            ctx.fillText('perfectly balanced', ideal + 4, 26);
            const mean = Num.mean(load), cv = Num.sd(load) / (mean || 1);
            out({
              active: (st.topk / E * 100).toFixed(0) + '% of the FFN',
              total: E + ' experts',
              dead: load.filter(v => v < st.tokens * st.topk / E * .35).length,
              cv: cv.toFixed(3)
            });
          }
        });
        Viz.note(host, 'Turn the auxiliary loss on and the bars level out. This is not a cosmetic problem: a starved expert is parameters you paid memory for and never use, and a collapsed router turns a 671B model into a much smaller one wearing its coat.');
      },

      ssm: function (host) {
        const st = Viz.controls(host, [
          { k: 'd', label: 'hidden size d', min: 512, max: 8192, step: 128, value: 4096, fmt: v => v.toLocaleString() },
          { k: 'win', label: 'sliding window w', min: 128, max: 8192, step: 128, value: 1024, fmt: v => v.toLocaleString() },
          { k: 'max', label: 'max sequence length', min: 4096, max: 262144, step: 4096, value: 131072, fmt: v => (v / 1024).toFixed(0) + 'k' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'at32', label: 'attention cost @ 32k', cls: 'bad' }, { k: 'win32', label: 'sliding window @ 32k' },
          { k: 'ssm32', label: 'SSM @ 32k', cls: 'good' }, { k: 'cross', label: 'attention exceeds FFN at' }
        ]);
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const d = st.d;
            const attn = s => (2 * s * s * d + 4 * s * d * d) / 1e12;
            const win = s => (2 * s * st.win * d + 4 * s * d * d) / 1e12;
            const ssm = s => (6 * s * d * d) / 1e12;
            const P = Viz.plot(ctx, w, h, { xd: [1024, st.max], yd: [0, attn(st.max) * 1.05] })
              .frame({ xlabel: 'sequence length', ylabel: 'TFLOPs per layer (forward)', xfmt: v => (v / 1024).toFixed(0) + 'k' });
            P.clip(() => {
              P.fn(attn, { color: T.red, width: 2.6, n: 300 });
              P.fn(win, { color: T.amber, width: 2.2, n: 300 });
              P.fn(ssm, { color: T.green, width: 2.4, n: 300 });
              P.vline(32768, { color: T.faint, dash: [3, 3], label: '32k' });
            });
            let cross = null;
            for (let s = 1024; s <= st.max; s += 512) { if (2 * s * s * d > 4 * s * d * d) { cross = s; break; } }
            out({
              at32: attn(32768).toFixed(2) + ' TF', win32: win(32768).toFixed(2) + ' TF',
              ssm32: ssm(32768).toFixed(2) + ' TF',
              cross: cross ? (cross / 1024).toFixed(0) + 'k tokens' : '—'
            });
          }
        });
        Viz.legend(host, [{ c: Viz.theme().red, t: 'full attention O(s²d)' }, { c: Viz.theme().amber, t: 'sliding window O(s·w·d)' }, { c: Viz.theme().green, t: 'SSM / linear O(s·d²)' }]);
        Viz.note(host, 'The crossover readout answers a question people get wrong: below roughly 2d tokens, attention is <i>not</i> the dominant cost — the FFN is. Quadratic attention only takes over past that point, which is why architecture debates about long context are really debates about the regime you serve.');
      }
    },
    quiz: [
      {
        q: 'Without an auxiliary load-balancing loss, an MoE router tends to…',
        options: ['distribute tokens uniformly', 'collapse onto a few favourite experts, leaving the rest as dead weight', 'route by token length', 'become deterministic and optimal'],
        answer: 1,
        why: 'Early winners get more gradient and win more — a rich-get-richer dynamic the auxiliary loss exists to counteract.'
      },
      {
        q: 'Why do Mamba-style models interleave a few full-attention layers?',
        options: ['for positional encoding', 'SSMs compress history into a fixed state and are weaker at precise in-context retrieval', 'to reduce parameters', 'for numerical stability'],
        answer: 1,
        why: 'Local/linear layers carry the language modelling; a few global layers do the retrieval — the same split as local + global attention.'
      }
    ],
    cards: [
      { q: 'MoE trade', a: 'Huge total parameters, small active per token (DeepSeek V3: 671B/37B). Costs: memory for all experts, all-to-all comms, router load balancing.' },
      { q: 'SSM / Mamba', a: 'Input-dependent linear recurrence: near-linear in length, constant state at inference, parallel training via associative scan; hybrids add a few attention layers.' }
    ]
  });
})();
