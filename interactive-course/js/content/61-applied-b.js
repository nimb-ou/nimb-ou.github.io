/* ============================================================
   PART 5 — Applied, continued: chunking (5.9), vector indexes
   (5.10), evals that do not lie (5.11), MLOps (5.12).
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 5.9 */
  ML.section({
    id: 'chunking', track: 'applied', num: '5.9', level: 2,
    title: 'Chunking and the data layer',
    lede: 'Most RAG systems that fail do not fail at retrieval or generation. They fail because the answer was split across two chunks, or buried in a chunk about something else, or lost when a PDF table became a wall of numbers. This is the unglamorous half of §5.1, and it is the half that decides the outcome.',
    prereq: ['rag'],
    related: ['rag', 'vector-search', 'evals'],
    html: `
${H.tldr([
      'Chunk size trades <b>precision</b> (small chunks, the embedding is about one thing) against <b>completeness</b> (large chunks, the whole answer is inside one). There is no universal answer; there is a measurement.',
      'Overlap is the cheapest insurance against splitting an answer in half. 10–20% of the chunk size is the usual setting.',
      '<b>Parse before you chunk.</b> A table flattened into prose, a header lost, a footnote inlined mid-sentence — these destroy more retrieval quality than any embedding-model choice.'
    ])}

<h2><span class="sn">5.9.1</span> The pipeline nobody draws</h2>
${H.svg('the data layer, in order', '0 0 660 130', `
<defs><style>
.bx{fill:var(--panel);stroke:var(--line)}
.bx2{fill:color-mix(in oklab,var(--c1) 12%,var(--panel));stroke:var(--c1)}
.t{font:11px ui-sans-serif,system-ui;fill:var(--text);text-anchor:middle}
.s{font:9.5px ui-monospace,monospace;fill:var(--faint);text-anchor:middle}
.a{stroke:var(--faint);fill:none;stroke-width:1.4;marker-end:url(#h1)}
</style>
<marker id="h1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--faint)"/></marker></defs>
<g>
${['source', 'parse', 'clean', 'chunk', 'enrich', 'embed', 'index'].map((s, i) => {
      const x = 12 + i * 93;
      const sub = ['pdf, html, db', 'layout-aware', 'dedupe, strip', 'size + overlap', 'title, dates', 'model choice', 'ANN + filters'][i];
      const cls = (i === 3 || i === 1) ? 'bx2' : 'bx';
      return '<rect class="' + cls + '" x="' + x + '" y="34" width="80" height="46" rx="8"/>' +
        '<text class="t" x="' + (x + 40) + '" y="56">' + s + '</text>' +
        '<text class="s" x="' + (x + 40) + '" y="70">' + sub + '</text>' +
        (i < 6 ? '<path class="a" d="M' + (x + 82) + ' 57 H' + (x + 91) + '"/>' : '');
    }).join('')}
<text class="s" x="330" y="106">the two highlighted stages account for most of the quality variance, and get the least attention</text>
</g>
`, 'Everything downstream is bounded by what happens here. A brilliant reranker cannot recover an answer that the parser destroyed.')}

<h2><span class="sn">5.9.2</span> Chunking strategies</h2>
${H.table(['Strategy', 'How', 'Good for', 'Cost'], [
      ['Fixed-size', '$n$ tokens with $m$ overlap', 'a baseline you should always measure against', 'splits mid-sentence'],
      ['<b>Recursive character</b>', 'split on paragraphs, then sentences, then words, until it fits', '<b>the sane default</b>', 'none, really'],
      ['Structural / layout', 'split on markdown headings, HTML sections, PDF blocks', 'documentation, contracts, anything with structure', 'needs a good parser'],
      ['Semantic', 'split where consecutive-sentence embedding similarity drops', 'unstructured prose', 'an embedding call per sentence'],
      ['<b>Hierarchical (small-to-big)</b>', 'embed small chunks, <i>return</i> their larger parent', '<b>usually the best quality/effort ratio</b>', 'two stores, slightly more code'],
      ['Late chunking', 'embed the long document once, then pool per chunk', 'preserves cross-chunk context in each vector', 'needs a long-context embedding model'],
      ['Proposition / atomic', 'an LLM rewrites the text into standalone facts', 'dense factual corpora, FAQ generation', 'an LLM call per document, and drift risk']
    ])}
${H.key('Small-to-big is the trick worth knowing: index sentence-level chunks so the embedding is precise, but hand the language model the surrounding section so it has enough context to answer. Precision on retrieval, completeness on generation, and you stop having to choose.')}

${H.lab('chunk', 'Chunk a real document and watch retrieval succeed or fail', 'A document with three planted answers, one of which straddles a natural boundary. Move the chunk size and overlap and watch which answers survive intact — and what that does to measured hit rate.')}

<h2><span class="sn">5.9.3</span> Metadata is half the system</h2>
${H.checklist([
      '<b>Prepend the document title and section path to the chunk text before embedding.</b> A chunk that says "the limit is 4,096" is meaningless; "API Reference → Rate Limits → the limit is 4,096" is retrievable. This one change is frequently worth more than upgrading the embedding model.',
      '<b>Store dates and versions as filterable fields</b>, not as prose. "What is the current policy?" is a filter, not a similarity question.',
      '<b>Keep a stable chunk ID and a pointer back to the source span</b>, so a citation can be verified and a stale chunk can be deleted.',
      '<b>Record the parser and chunker version</b> on every chunk. When you change either, you need to know what to re-index.',
      '<b>Deduplicate.</b> Corpora built from wikis and exports are 20–40% near-duplicate, and duplicates crowd out diversity in the top $k$.'
    ])}
${H.pitfall('Tables are where naive pipelines die. A financial table flattened to "Revenue 2024 2025 1,204 1,455" is retrievable by nobody and misreadable by everyone. The workable options are: extract tables separately and store them as markdown or HTML; generate a one-sentence natural-language summary per table for the embedding while keeping the structured form for the answer; or route table questions to SQL over the structured source instead of to retrieval at all.')}

${H.worked('a chunking budget, worked', `
<p>A 40,000-document corpus averaging 3,000 tokens each = 120M tokens.</p>
<ul>
<li>At 512-token chunks with 15% overlap: ~270,000 chunks. Embedding at $0.02/M tokens ≈ <b>$2.80</b>, once.</li>
<li>Storage at 1,024-dimensional float32: 270k × 4 KB ≈ <b>1.1 GB</b> — and that is why quantised vectors (int8, or binary with rescoring) matter at scale: the same index in int8 is 270 MB.</li>
<li>Re-embedding when you change the model or chunker: the same $2.80 and a few hours. <b>Cheap.</b> Re-parsing 40,000 PDFs correctly: possibly weeks.</li>
</ul>
<p>The lesson in those three numbers: embeddings are nearly free and easy to redo, parsing is expensive and hard to redo. Spend your effort accordingly.</p>`)}

${H.probe([
      ['What chunk size should we use?', 'Whatever your evaluation set says. Start at 400–600 tokens with 10–15% overlap and recursive splitting, then measure recall@k against a labelled question set. Anyone who gives you a number without asking about your documents is guessing.'],
      ['A user asks a question whose answer spans three sections. What breaks?', 'Single-chunk retrieval. You need larger or hierarchical chunks, multi-chunk retrieval with a higher $k$, or query decomposition into sub-questions.'],
      ['Why prepend the section heading to each chunk?', 'The embedding is of the chunk text alone; without context, an anaphoric or elliptical chunk has no retrievable signal. It is one line of code and a large measured gain.'],
      ['When would you not chunk at all?', 'Short documents that fit whole; or when the model’s context is large enough and the corpus small enough to just include everything — which is often cheaper than a retrieval system (§5.2).']
    ])}`,
    labs: {
      chunk: function (host) {
        const DOC = ('# Rate limits\\n' +
          'The API applies per-organisation quotas. Requests are counted per minute and per day. ' +
          'Exceeding a quota returns HTTP 429 with a Retry-After header. ' +
          '## Standard tier\\n' +
          'Standard organisations may issue up to 4096 requests per minute. ' +
          'The daily ceiling is 2 million requests. Bursts above the per-minute limit are rejected immediately rather than queued. ' +
          '## Enterprise tier\\n' +
          'Enterprise agreements raise the per-minute limit to 60000 and remove the daily ceiling entirely. ' +
          'Enterprise customers also receive a dedicated capacity pool that is not shared with other organisations. ' +
          '## Retries\\n' +
          'Clients should retry with exponential backoff starting at one second and doubling, ' +
          'with full jitter, up to a maximum of six attempts. Retrying without backoff will extend the throttling window. ' +
          '## Support\\n' +
          'Quota increases are requested through the console and are reviewed within two business days.').split(/\\s+/);

        const ANSWERS = [
          { q: 'What is the standard per-minute limit?', need: ['4096'] },
          { q: 'What is the enterprise per-minute limit?', need: ['60000'] },
          { q: 'How should clients retry, and how many times?', need: ['backoff', 'six'] }   // deliberately spans a sentence boundary
        ];

        const st = Viz.controls(host, [
          { k: 'size', label: 'chunk size (words)', min: 8, max: 80, step: 2, value: 26, fmt: v => v },
          { k: 'ov', label: 'overlap', min: 0, max: .5, step: .05, value: .1, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'head', label: 'prepend the section heading', type: 'toggle', value: false },
          { k: 'small2big', label: 'small-to-big (return the parent section)', type: 'toggle', value: false }
        ], () => draw());

        const view = ML.el('div');
        host.appendChild(view);
        const out = Viz.readout(host, [
          { k: 'n', label: 'chunks', cls: 'key' },
          { k: 'hit', label: 'answers fully inside one chunk', cls: 'good' },
          { k: 'split', label: 'answers split across chunks', cls: 'bad' },
          { k: 'tok', label: 'tokens sent to the model' }
        ]);

        function chunks() {
          const step = Math.max(1, Math.round(st.size * (1 - st.ov)));
          const out2 = [];
          let heading = '';
          for (let i = 0; i < DOC.length; i += step) {
            const words = DOC.slice(i, i + st.size);
            for (let j = Math.max(0, i); j >= 0; j--) if (DOC[j] && DOC[j][0] === '#') { heading = DOC.slice(j, j + 3).join(' ').replace(/#+/g, '').trim(); break; }
            out2.push({ from: i, to: Math.min(DOC.length, i + st.size), words: words, heading: heading });
            if (i + st.size >= DOC.length) break;
          }
          return out2;
        }

        function draw() {
          const cs = chunks();
          let hit = 0, split = 0;
          const marks = ANSWERS.map(a => {
            const inside = cs.some(c => a.need.every(n => c.words.join(' ').indexOf(n) >= 0));
            if (inside) hit++; else split++;
            return inside;
          });
          view.innerHTML = '<div style="font-family:var(--mono);font-size:11.5px;line-height:1.85;max-height:210px;overflow:auto;' +
            'border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--panel)">' +
            cs.map((c, i) => '<span style="background:color-mix(in oklab,var(--c' + ((i % 6) + 1) + ') 16%,transparent);' +
              'border-left:2px solid var(--c' + ((i % 6) + 1) + ');padding:1px 4px;margin-right:2px;border-radius:3px">' +
              (st.head ? '<i style="color:var(--faint)">[' + ML.escapeHtml(c.heading) + '] </i>' : '') +
              ML.escapeHtml(c.words.join(' ')) + '</span> ').join('') + '</div>' +
            '<div style="margin-top:10px">' + ANSWERS.map((a, i) =>
              '<p class="small" style="margin:0 0 4px"><span style="color:var(--' + (marks[i] ? 'green">✓' : 'red">✗') +
              '</span> ' + a.q + (marks[i] ? '' : ' <i>— the answer is split across a chunk boundary</i>') + '</p>').join('') + '</div>';
          const perChunk = st.small2big ? st.size * 3 : st.size;
          out({
            n: cs.length,
            hit: hit + ' of ' + ANSWERS.length,
            split: split,
            tok: Math.round(perChunk * 4 * 1.35) + ' (top-4)'
          });
        }
        draw();
        Viz.note(host, 'Set the chunk size to 12 words: the embeddings become very precise and the retry answer — which needs both "backoff" and "six" — is torn in half. Push it to 70 and everything fits, but each chunk now covers four topics so the embedding is a blur and precision collapses. Turn on <b>small-to-big</b> and you get both: precise small chunks for matching, the full parent section for answering, at 3× the tokens sent to the model. <b>That token cost is the price of the trick, and it is usually worth paying.</b>');
      }
    },
    quiz: [
      {
        q: 'Small-to-big retrieval means…',
        options: ['start with a small model, escalate', 'embed small chunks for precise matching but return their larger parent for context', 'chunk smaller as the corpus grows', 'retrieve fewer documents over time'],
        answer: 1,
        why: 'It decouples the unit you match on from the unit you answer with, which is the cleanest resolution of the precision/completeness trade.'
      },
      {
        q: 'The single cheapest improvement to chunk retrievability is usually…',
        options: ['a bigger embedding model', 'prepending the document title and section path to each chunk before embedding', 'more overlap', 'a larger k'],
        answer: 1,
        why: 'It gives an otherwise contextless chunk something to match against, and costs one line of code.'
      },
      {
        q: 'Re-embedding a 120M-token corpus costs roughly…',
        options: ['a few dollars', 'a few hundred dollars', 'a few thousand dollars', 'it is not feasible'],
        answer: 0,
        why: 'At ~$0.02 per million tokens, about $2.80. Parsing correctly is the expensive part; embeddings are cheap and easy to redo.'
      }
    ],
    cards: [
      { q: 'Default chunking recipe', a: 'Recursive splitting, 400–600 tokens, 10–15% overlap, heading prepended — then measure recall@k and tune from there.' },
      { q: 'Small-to-big', a: 'Index small chunks, return the parent section. Precision on retrieval, completeness on generation.' },
      { q: 'Why tables break RAG', a: 'Flattening destroys the row/column relationship. Extract separately as markdown, summarise for embedding, or route to SQL.' },
      { q: 'Where to spend effort', a: 'Parsing is expensive and hard to redo; embeddings are cheap and easy to redo. Optimise the parser.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.10 */
  ML.section({
    id: 'vector-search', track: 'applied', num: '5.10', level: 3,
    title: 'Vector databases and approximate nearest neighbours',
    lede: 'Exact nearest-neighbour search over ten million vectors is a linear scan and takes seconds. Every vector database is one answer to the same question: how much recall will you give up for a thousandfold speedup, and how do you keep the filters honest while doing it?',
    prereq: ['rag', 'embeddings'],
    related: ['rag', 'chunking', 'pca'],
    html: `
${H.tldr([
      'ANN indexes trade <b>recall</b> for <b>latency</b>. The right question is never "which index is best" but "what recall do I need at what p99, and how much memory can I spend".',
      '<b>HNSW</b> — a navigable small-world graph with a hierarchy — is the default: excellent recall/latency, high memory, slow to build. <b>IVF-PQ</b> compresses vectors to a few bytes and is what you use when the index will not fit in RAM.',
      'Filtered search is the operationally hard part. Pre-filter and the graph loses connectivity; post-filter and you may return nothing. Every serious system implements <i>filtered traversal</i> instead.'
    ])}

<h2><span class="sn">5.10.1</span> The index families</h2>
${H.table(['Index', 'Idea', 'Recall@10', 'Build', 'Memory', 'Use when'], [
      ['Flat (exact)', 'scan everything', '100%', 'none', '1×', '<100k vectors — genuinely, just do this'],
      ['<b>HNSW</b>', 'layered proximity graph, greedy descent', '95–99%', 'slow', '<b>1.5–2×</b>', 'the default under ~10M vectors'],
      ['IVF-Flat', 'k-means partitions, scan the nearest few', '90–97%', 'fast', '1×', 'large, memory available'],
      ['<b>IVF-PQ</b>', 'partitions + product-quantised codes', '80–95%', 'fast', '<b>0.03–0.1×</b>', 'hundreds of millions of vectors'],
      ['ScaNN', 'anisotropic quantization, score-aware', '95–98%', 'medium', 'low', 'Google-stack, very strong benchmarks'],
      ['DiskANN / Vamana', 'graph on SSD', '95%+', 'slow', 'tiny RAM', 'billions of vectors, cost-constrained'],
      ['Binary + rescore', '1-bit codes, then exact rerank of the top few hundred', '95%+', 'fast', '<b>0.03×</b>', 'a startlingly strong and underused baseline']
    ])}

${H.lab('ann', 'Build a proximity graph and watch greedy search walk it', 'A real navigable small-world graph, built here from the points shown, searched greedily from an entry point. The path is the actual traversal, and the distance-computation count is the actual cost.')}

${H.intuition(`<p>Why does a greedy walk on a graph find near neighbours? Because the graph has two kinds of edge: short ones to genuine neighbours, and a few long ones that act as motorways. The long edges get you into the right region in a handful of hops (the "small world" property — hop count grows like $\\log n$); the short ones do the local refinement. HNSW's hierarchy makes this explicit: upper layers are sparse and long-range, the bottom layer is dense and local, and you descend.</p>`)}
${H.table(['HNSW parameter', 'Meaning', 'Effect'], [
      ['$M$', 'edges per node', 'higher = better recall, more memory, slower build. 16–48 typical'],
      ['<code>efConstruction</code>', 'candidate list size while building', 'higher = better graph, much slower build. 100–500'],
      ['<b><code>efSearch</code></b>', 'candidate list size at query time', '<b>the runtime recall/latency dial</b> — tune this, not the others'],
      ['layers', 'chosen randomly per node', 'geometric distribution; gives the $\\log n$ hop count']
    ])}
${H.key('<code>efSearch</code> is the one knob you tune in production, because it can be changed per query without rebuilding anything. Recall rises and latency rises with it, and the curve is steep at the start and flat at the end — which means there is almost always a setting that gets you 97% recall for a fraction of the cost of 99.5%.')}

<h2><span class="sn">5.10.2</span> Filtered search: the operationally hard part</h2>
${H.vs('Pre-filter', [
      'Restrict to matching IDs, then search',
      'Correct by construction',
      '<b>Destroys graph connectivity</b> when the filter is selective — the walk cannot reach the survivors',
      'Degenerates to a linear scan of the filtered set'
    ], 'Post-filter', [
      'Search, then drop non-matching results',
      'Fast, uses the index as built',
      '<b>May return far fewer than $k$</b>, or nothing at all',
      'Needs over-fetching by an unknown factor'
    ])}
${H.flag('The real answer is <b>filtered traversal</b>: walk the graph normally but only admit matching nodes into the result set, while still traversing through non-matching ones. Qdrant, Weaviate and pgvector-with-iterative-scan all implement a version of this, and each switches to a brute-force scan below some selectivity threshold. When you evaluate a vector database, <i>measure recall with your actual filters applied</i> — unfiltered benchmark numbers are close to meaningless for a real workload.')}

<h2><span class="sn">5.10.3</span> Product quantization, in one page</h2>
${H.steps([
      'Split each $D$-dimensional vector into $m$ sub-vectors of length $D/m$.',
      'Run k-means with 256 centroids on each sub-space, over the training set. Each sub-vector is now representable by <b>one byte</b> — the index of its nearest centroid.',
      'A 1,024-dimensional float32 vector (4,096 bytes) with $m=64$ becomes <b>64 bytes</b>: a 64× compression.',
      'At query time, precompute the distance from the query’s sub-vectors to all 256 centroids per sub-space — a $m\\times256$ lookup table — then every candidate distance is $m$ table lookups and an add. No decompression, and it is fast because it is cache-resident.',
      '<b>Rescore</b> the top few hundred with the full-precision vectors to recover most of the lost recall. This step is what makes aggressive quantization acceptable.'
    ])}
${H.worked('the memory arithmetic that picks your index', `
<p>10 million chunks, 1,024-dimensional embeddings:</p>
<ul>
<li><b>Flat float32</b>: 10M × 4 KB = <b>41 GB</b>. Exact, and needs a large machine.</li>
<li><b>HNSW float32, M=32</b>: 41 GB + graph ≈ <b>60 GB</b>. Fast and accurate, expensive.</li>
<li><b>int8 quantized + HNSW</b>: ~10 GB + graph ≈ <b>15 GB</b>, with ~1% recall loss. Usually the right answer.</li>
<li><b>IVF-PQ, 64 bytes/vector</b>: <b>0.64 GB</b>. Fits anywhere; 85–93% recall, restored to ~97% with rescoring.</li>
<li><b>Binary (1 bit/dim) + rescore</b>: 128 bytes/vector = <b>1.3 GB</b>, with a full-precision rescore of the top 500.</li>
</ul>
<p>The jump from 60 GB to 1.3 GB changes what machine you rent, which is usually a larger cost difference than the recall difference is a quality difference.</p>`)}

${H.probe([
      ['HNSW or IVF-PQ?', 'HNSW under about 10M vectors when memory is available — better recall/latency and no training step. IVF-PQ when the index must be small or the corpus is hundreds of millions.'],
      ['Which parameter do you tune in production?', '<code>efSearch</code>. It is per-query, needs no rebuild, and trades recall against latency along a curve that is steep at the start and flat at the end.'],
      ['Why is filtered vector search hard?', 'Pre-filtering breaks graph connectivity; post-filtering can return fewer than $k$ results. Filtered traversal — walk through non-matching nodes but only collect matching ones — is the standard fix, with a brute-force fallback below some selectivity.'],
      ['How do you measure whether your index is good enough?', 'Compute exact nearest neighbours on a sample of real queries, then measure recall@k of the index against them <i>with your filters applied</i>, alongside p50 and p99 latency.'],
      ['When do you not need a vector database at all?', 'Under ~100k vectors: a numpy matrix multiply is faster than a network round trip. Or when BM25 alone already answers the queries — measure before adding infrastructure.']
    ])}`,
    labs: {
      ann: function (host) {
        const st = Viz.controls(host, [
          { k: 'n', label: 'vectors in the index', min: 40, max: 400, step: 20, value: 160, fmt: v => v },
          { k: 'M', label: 'edges per node M', min: 2, max: 12, step: 1, value: 4, fmt: v => v },
          { k: 'ef', label: 'efSearch', min: 1, max: 30, step: 1, value: 4, fmt: v => v },
          { k: 'qx', label: 'query x', min: -2.4, max: 2.4, step: .05, value: 1.1, fmt: v => v.toFixed(2) },
          { k: 'qy', label: 'query y', min: -2.4, max: 2.4, step: .05, value: -0.8, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'rec', label: 'recall@10', cls: 'good' },
          { k: 'comps', label: 'distance computations', cls: 'key' },
          { k: 'exact', label: 'exact scan would need', cls: 'bad' },
          { k: 'speed', label: 'speedup' },
          { k: 'hops', label: 'greedy hops' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(12);
            const pts = [];
            for (let i = 0; i < st.n; i++) {
              const c = i % 4;
              const cx = [(-1.3), 1.4, -1.1, 1.2][c], cy = [1.2, 1.1, -1.2, -1.3][c];
              pts.push([cx + R.normal(0, .65), cy + R.normal(0, .65)]);
            }
            const idx = Num.nsw(pts, st.M, 3);
            const q = [st.qx, st.qy];
            const res = idx.search(q, st.ef, 0);
            const truth = idx.exact(q, 10);
            const found = res.nearest.slice(0, 10);
            const rec = truth.filter(t => found.indexOf(t) >= 0).length / Math.min(10, truth.length);

            const w1 = w * .58;
            const P = Viz.plot(ctx, w1, h, { xd: [-3.2, 3.2], yd: [-3, 3], pad: { l: 8, r: 8, t: 14, b: 26 } })
              .frame({ grid: false, xticks: [], yticks: [], xlabel: 'the graph, and the walk' });
            P.clip(() => {
              ctx.strokeStyle = T.line; ctx.lineWidth = .6; ctx.globalAlpha = .55;
              idx.adj.forEach((nbrs, i) => nbrs.forEach(j => {
                if (j < i) return;
                ctx.beginPath(); ctx.moveTo(P.x(pts[i][0]), P.y(pts[i][1])); ctx.lineTo(P.x(pts[j][0]), P.y(pts[j][1])); ctx.stroke();
              }));
              ctx.globalAlpha = 1;
              pts.forEach(p => P.dots([[p[0], p[1]]], { r: 2, color: T.faint, alpha: .7 }));
              truth.forEach(i => P.dots([[pts[i][0], pts[i][1]]], { r: 4.2, color: T.c3, alpha: .9 }));
              found.forEach(i => P.dots([[pts[i][0], pts[i][1]]], { r: 2.6, color: T.c1 }));
              P.line(res.path.map(i => pts[i]), { color: T.c4, width: 2 });
              P.dots([pts[res.path[0]]], { r: 5, color: T.c4, stroke: true, strokeWidth: 1.6 });
              P.dots([q], { r: 6.5, color: T.c2, stroke: true, strokeWidth: 2 });
            });

            ctx.save(); ctx.translate(w1, 0);
            const curve = [];
            for (let e = 1; e <= 30; e += 1) {
              const r2 = idx.search(q, e, 0);
              const f2 = r2.nearest.slice(0, 10);
              curve.push([r2.comps, truth.filter(t => f2.indexOf(t) >= 0).length / Math.min(10, truth.length)]);
            }
            const P2 = Viz.plot(ctx, w - w1, h, {
              xd: [0, Math.max(20, st.n)], yd: [0, 1.05], pad: { l: 44, r: 12, t: 14, b: 40 }
            }).frame({ xlabel: 'distance computations', ylabel: 'recall@10', yfmt: v => (v * 100).toFixed(0) + '%' });
            P2.clip(() => {
              P2.line(curve.sort((a, b) => a[0] - b[0]), { color: T.c1, width: 2.4 });
              P2.dots(curve, { r: 2.4, color: T.c1 });
              P2.vline(st.n, { color: T.c2, dash: [4, 3], label: 'exact scan' });
              P2.dots([[res.comps, rec]], { r: 5, color: T.c2, stroke: true, strokeWidth: 2 });
            });
            ctx.restore();
            out({
              rec: (rec * 100).toFixed(0) + '%',
              comps: res.comps,
              exact: st.n,
              speed: (st.n / Math.max(1, res.comps)).toFixed(1) + '×',
              hops: res.path.length - 1
            });
          }
        });
        Viz.legend(host, [
          { c: 'var(--c2)', t: 'query' }, { c: 'var(--c3)', t: 'true 10 nearest' },
          { c: 'var(--c1)', t: 'what the index returned' }, { c: 'var(--c4)', t: 'the greedy walk' }
        ]);
        Viz.note(host, 'The right-hand curve is the whole story of ANN search: <b>recall against work done</b>, with the vertical line marking what an exact scan would cost. Notice its shape — steep, then flat. Going from efSearch 1 to 6 buys most of the recall for a fraction of the scan cost; going from 20 to 30 buys almost nothing. Now drop <b>M</b> to 2: the graph fragments, the walk gets trapped in a local cluster, and recall collapses no matter how large you make efSearch. <b>Build quality bounds search quality</b>, which is why efConstruction is worth its build time.');
      }
    },
    quiz: [
      {
        q: 'The parameter you tune at query time to trade recall against latency in HNSW is…',
        options: ['M', 'efConstruction', 'efSearch', 'the number of layers'],
        answer: 2,
        why: 'It is per-query and needs no rebuild. M and efConstruction are baked into the index at build time.'
      },
      {
        q: 'Product quantization with $m=64$ sub-spaces compresses a 1024-d float32 vector to…',
        options: ['1024 bytes', '256 bytes', '64 bytes', '16 bytes'],
        answer: 2,
        why: 'One byte per sub-space (a 256-centroid codebook index): 4,096 bytes → 64 bytes, a 64× reduction.'
      },
      {
        q: 'Pre-filtering a graph index by a highly selective predicate…',
        options: ['is the correct approach', 'breaks graph connectivity so the walk cannot reach the surviving nodes', 'is always faster', 'improves recall'],
        answer: 1,
        why: 'Which is why real systems use filtered traversal — walk through non-matching nodes, collect only matching ones — with a brute-force fallback.'
      },
      {
        q: 'With 60,000 vectors, the right index is usually…',
        options: ['HNSW', 'IVF-PQ', 'flat exact search', 'DiskANN'],
        answer: 2,
        why: 'A brute-force matrix multiply over 60k vectors takes single-digit milliseconds — less than the network round trip to a vector database.'
      }
    ],
    cards: [
      { q: 'Index selection, in one line', a: '<100k: flat. <10M with RAM: HNSW (int8). Hundreds of millions: IVF-PQ or DiskANN. Always rescore.' },
      { q: 'HNSW parameters', a: 'M (edges), efConstruction (build quality), efSearch (the runtime recall/latency dial).' },
      { q: 'Product quantization', a: 'Split into $m$ sub-vectors, 256-centroid codebook each → 1 byte per sub-vector; distances by table lookup; rescore the top few hundred.' },
      { q: 'Filtered ANN', a: 'Pre-filter breaks connectivity, post-filter under-returns. Filtered traversal with a brute-force fallback is the real answer — benchmark with your filters on.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.11 */
  ML.section({
    id: 'evals', track: 'applied', num: '5.11', level: 2,
    title: 'Building evals that do not lie',
    lede: 'The eval is the only thing standing between you and shipping on vibes. It is also, in most teams, the least rigorous artefact in the codebase — a hundred hand-picked examples, graded by a model with known biases, reported without an interval.',
    prereq: ['llm-eval', 'intervals'],
    related: ['llm-eval', 'experimentation', 'mlops'],
    html: `
${H.tldr([
      'An eval set of 100 examples has a 95% interval of roughly ±10 points on a proportion near 50%. <b>Most reported "improvements" are inside the noise of the eval that measured them.</b>',
      'LLM judges have measurable, reproducible biases: position, verbosity, self-preference, and sensitivity to formatting. Every one of them has a cheap mitigation, and none of them goes away by choosing a better model.',
      'Build the eval from <b>production failures</b>, not from imagination. The examples you can think of are the ones the system already handles.'
    ])}

<h2><span class="sn">5.11.1</span> How big does the eval set need to be?</h2>
<p>An eval score is a proportion estimated from a sample, so it has a standard error of $\\sqrt{p(1-p)/n}$ — and comparing two systems on the same set is a paired comparison, which is considerably more powerful than two independent ones.</p>
${H.lab('evalsize', 'How many examples do you actually need?', 'The interval on a single score, and the number of examples needed to call a difference between two systems. Paired evaluation on the same items is the free win here.')}
${H.key('Use the <b>same examples</b> for both systems and compare per-item. Paired comparison removes the item-difficulty variance, and typically needs 3–5× fewer examples than scoring the two systems on independent samples. It is the cheapest statistical improvement available to you.')}

<h2><span class="sn">5.11.2</span> LLM-as-judge, and its documented biases</h2>
${H.table(['Bias', 'What happens', 'Mitigation', 'Residual'], [
      ['<b>Position</b>', 'the first (or last) option is preferred regardless of content', '<b>run both orders, average</b> — or discard disagreements', 'largely solved'],
      ['<b>Verbosity</b>', 'longer answers score higher at equal quality', 'control for length; penalise it explicitly in the rubric', 'partly solved'],
      ['Self-preference', 'a judge prefers text from its own family', 'use a different family as judge; or a panel', 'reduced, not eliminated'],
      ['Formatting', 'markdown, bullet points and headers raise scores', 'normalise formatting before judging', 'easy'],
      ['Sycophancy', 'agrees with an assertion embedded in the prompt', 'never state the expected answer in the judge prompt', 'easy'],
      ['Score compression', 'everything gets a 4 out of 5', 'pairwise preference instead of absolute scoring', '<b>use pairwise by default</b>'],
      ['Rubric drift', 'the meaning of "good" moves as you edit the prompt', 'version the judge prompt; re-run a fixed calibration set', 'process, not modelling']
    ])}
${H.lab('judge', 'Simulating judge bias, and what mitigation recovers', 'Two systems of known true quality, evaluated by a biased judge. Turn the biases on and off, and turn the mitigations on and off, and see how far the measured win rate drifts from the truth.')}
${H.pitfall('Report <b>agreement with human labels</b> for your judge, on a held-out set, before you trust it. Cohen’s κ above 0.6 is workable; above 0.8 is good. Human–human agreement on the same task is the real ceiling — if two annotators agree only 70% of the time, a judge at 68% is doing fine and the task definition is the problem, not the judge.')}

<h2><span class="sn">5.11.3</span> What to actually build</h2>
${H.steps([
      '<b>A golden set from production failures.</b> Every incident, complaint and thumbs-down becomes a test case with the expected behaviour written down. This set only grows, and it is the most valuable artefact your team owns.',
      '<b>Deterministic assertions wherever possible.</b> Did it call the right tool? Is the JSON valid? Does the citation exist in the retrieved context? Is the number within tolerance? These need no judge, cost nothing, and never drift.',
      '<b>A held-out slice you never look at</b> until a release decision. The set you iterate against will be overfitted; that is not a failure of discipline, it is arithmetic.',
      '<b>Per-slice reporting.</b> Overall averages hide the group where you are failing. Break out by language, query type, document source, user tier.',
      '<b>Adversarial and safety cases</b>, maintained separately with their own bar. These are pass/fail gates, not averages.',
      '<b>Cost and latency in the same table as quality.</b> A 2-point quality gain for 3× the cost is a decision, and it cannot be made if the numbers live in different dashboards.'
    ])}
${H.vs('Offline eval', [
      'Fast, cheap, repeatable, runs in CI',
      'Catches regressions before users see them',
      'Measures a proxy — correlation with the real outcome is an assumption you should test',
      'Overfits as you iterate against it'
    ], 'Online eval', [
      'Measures what you actually care about',
      'Slow, expensive, and needs traffic',
      'Confounded unless randomised (§2.25)',
      'The final arbiter — but too slow to be the only loop'
    ])}
${H.flag('Public benchmark contamination is now severe enough that a strong score on a well-known benchmark is weak evidence about a model released after that benchmark was published. Prefer a private eval built from your own traffic; treat public leaderboards as a coarse filter for which models to bother testing.')}

${H.probe([
      ['You have 200 eval examples and system B scores 4 points higher. Is it better?', 'Probably not distinguishable. Paired on the same 200 items you might detect it if the per-item disagreement rate is low; unpaired you would need roughly 1,000+. Report the interval and the number of items where they differ.'],
      ['How do you control for position bias in an LLM judge?', 'Evaluate each pair in both orders and average, or count only the cases where the judgement is consistent across orders. Report the flip rate — it is a direct measure of how much of your signal is noise.'],
      ['Your judge agrees with humans 68% of the time. Is that bad?', 'Compare it with human–human agreement on the same task. If two humans agree 70% of the time, the task definition is ambiguous and 68% is close to the ceiling. Fix the rubric, not the judge.'],
      ['Where do good eval cases come from?', 'Production failures. The examples you invent are the ones the system already handles; the ones that broke are the ones that carry information.']
    ])}`,
    labs: {
      evalsize: function (host) {
        const st = Viz.controls(host, [
          { k: 'p', label: 'baseline pass rate', min: .1, max: .95, step: .01, value: .72, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'delta', label: 'improvement to detect', min: .01, max: .2, step: .005, value: .04, fmt: v => '+' + (v * 100).toFixed(1) + ' pts' },
          { k: 'disagree', label: 'items where the two systems differ (paired)', min: .05, max: 1, step: .05, value: .25, fmt: v => (v * 100).toFixed(0) + '%' }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'ci', label: '95% CI at n = 100', cls: 'warn' },
          { k: 'ci500', label: '… at n = 500' },
          { k: 'unpaired', label: 'n needed, unpaired', cls: 'bad' },
          { k: 'paired', label: 'n needed, paired', cls: 'good' },
          { k: 'ratio', label: 'paired is cheaper by' }
        ]);
        const S = Viz.surface(host, {
          height: 280,
          draw: function (ctx, w, h, T) {
            const p = st.p;
            const P = Viz.plot(ctx, w, h, { xd: [1.3, 4], yd: [0, .2], pad: { l: 56, r: 14, t: 16, b: 42 } })
              .frame({
                xticks: [1.3, 2, 2.5, 3, 3.5, 4], xfmt: v => Math.round(Math.pow(10, v)).toLocaleString(),
                xlabel: 'eval set size n', ylabel: '95% CI half-width', yfmt: v => '±' + (v * 100).toFixed(0) + ' pts'
              });
            P.clip(() => {
              P.area(Num.linspace(1.3, 4, 80).map(lx => [lx, 1.96 * Math.sqrt(p * (1 - p) / Math.pow(10, lx))]), { color: T.c1, alpha: .12 });
              P.fn(lx => 1.96 * Math.sqrt(p * (1 - p) / Math.pow(10, lx)), { color: T.c1, width: 2.8, n: 120 });
              P.hline(st.delta / 2, { color: T.c2, dash: [5, 4], label: 'half your effect' });
              [100, 500, 2000].forEach(n => {
                P.dots([[Math.log10(n), 1.96 * Math.sqrt(p * (1 - p) / n)]], { r: 4.2, color: T.c4, stroke: true });
                P.text(Math.log10(n), 1.96 * Math.sqrt(p * (1 - p) / n), ' n=' + n, { dx: 7, color: T.c4, font: '10.5px ui-monospace' });
              });
            });
            const unp = Num.sampleSize(p, st.delta, .05, .8);
            // McNemar-style: only discordant pairs carry information
            const pd = st.disagree;
            const paired = Math.ceil(Math.pow(1.96 * Math.sqrt(pd) + 0.84 * Math.sqrt(pd - st.delta * st.delta), 2) / (st.delta * st.delta));
            out({
              ci: '±' + (1.96 * Math.sqrt(p * (1 - p) / 100) * 100).toFixed(1) + ' pts',
              ci500: '±' + (1.96 * Math.sqrt(p * (1 - p) / 500) * 100).toFixed(1) + ' pts',
              unpaired: unp.toLocaleString(),
              paired: paired.toLocaleString(),
              ratio: (unp / Math.max(1, paired)).toFixed(1) + '×'
            });
          }
        });
        Viz.note(host, 'At n = 100 and a pass rate near 70%, the 95% interval is about <b>±9 points</b>. A "3-point improvement" measured on 100 examples is indistinguishable from nothing. Now look at the paired row: when the two systems differ on only 25% of items, a paired test needs a small fraction of the sample — because the items they both get right or both get wrong carry no information about which is better, and pairing removes them from the variance.');
      },

      judge: function (host) {
        const st = Viz.controls(host, [
          { k: 'trueA', label: 'true quality of A', min: .3, max: .9, step: .02, value: .62, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'pos', label: 'position bias', min: 0, max: .3, step: .01, value: .12, fmt: v => '+' + (v * 100).toFixed(0) + ' pts to first' },
          { k: 'verb', label: 'verbosity bias (B is longer)', min: 0, max: .3, step: .01, value: .1, fmt: v => '+' + (v * 100).toFixed(0) + ' pts to B' },
          { k: 'swap', label: 'mitigation: evaluate both orders', type: 'toggle', value: false },
          { k: 'lennorm', label: 'mitigation: normalise length', type: 'toggle', value: false }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'truth', label: 'true win rate for A', cls: 'key' },
          { k: 'meas', label: 'measured win rate', cls: 'bad' },
          { k: 'err', label: 'error' },
          { k: 'flip', label: 'order-flip disagreement' },
          { k: 'verdict', label: 'conclusion you would draw' }
        ]);
        const S = Viz.surface(host, {
          height: 260,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(41);
            const N = 400;
            let winA = 0, flips = 0;
            for (let i = 0; i < N; i++) {
              const judgeOnce = (aFirst) => {
                let pA = st.trueA;
                pA += aFirst ? st.pos : -st.pos;
                if (!st.lennorm) pA -= st.verb;
                return R() < Math.max(0.01, Math.min(.99, pA));
              };
              if (st.swap) {
                const r1 = judgeOnce(true), r2 = judgeOnce(false);
                if (r1 !== r2) flips++;
                if ((r1 ? 1 : 0) + (r2 ? 1 : 0) >= 1 && r1 === r2) winA += r1 ? 1 : 0;
                else winA += 0.5 * ((r1 ? 1 : 0) + (r2 ? 1 : 0));
              } else {
                if (judgeOnce(true)) winA++;
              }
            }
            const measured = winA / N;
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [-.6, 1.6], pad: { l: 40, r: 14, t: 16, b: 40 } })
              .frame({ yticks: [], xlabel: 'win rate for system A', xfmt: v => (v * 100).toFixed(0) + '%' });
            P.clip(() => {
              P.vline(.5, { color: T.faint, dash: [3, 3], width: 1 });
              // bars
              [[st.trueA, T.c3, 'true', 1.0], [measured, T.c2, 'measured', 0.3]].forEach(([v, c, lbl, y]) => {
                ctx.fillStyle = c; ctx.globalAlpha = .8;
                ctx.fillRect(P.x(0), P.y(y) - 16, P.x(v) - P.x(0), 32);
                ctx.globalAlpha = 1;
                ctx.fillStyle = T.text; ctx.font = '11.5px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(lbl + '  ' + (v * 100).toFixed(1) + '%', P.x(v) + 8, P.y(y));
              });
            });
            const wrongCall = (st.trueA > .5) !== (measured > .5);
            out({
              truth: (st.trueA * 100).toFixed(0) + '%',
              meas: (measured * 100).toFixed(1) + '%',
              err: ((measured - st.trueA) * 100).toFixed(1) + ' pts',
              flip: st.swap ? (flips / N * 100).toFixed(0) + '%' : 'not measured',
              verdict: wrongCall ? '✗ you would pick the WRONG system' : (Math.abs(measured - st.trueA) > .05 ? '~ right winner, wrong margin' : '✓ close enough')
            });
          }
        });
        Viz.note(host, 'With position bias at 12 points and verbosity bias at 10, a system that genuinely wins 62% of the time can measure below 50% — <b>you would ship the worse one.</b> Turn on <b>evaluate both orders</b> and the position bias cancels exactly; turn on <b>normalise length</b> and the verbosity bias goes with it. The order-flip disagreement rate is worth reporting on its own: it tells you what fraction of your judgements were decided by presentation rather than content.');
      }
    },
    quiz: [
      {
        q: 'A 100-example eval with a pass rate near 70% has a 95% confidence interval of roughly…',
        options: ['±1 point', '±3 points', '±9 points', '±20 points'],
        answer: 2,
        why: '$1.96\\sqrt{0.7\\cdot0.3/100} \\approx 0.09$. Most reported small improvements are inside this.'
      },
      {
        q: 'The cheapest fix for position bias in an LLM judge is…',
        options: ['a bigger judge model', 'evaluating each pair in both orders and averaging', 'raising the temperature', 'more examples'],
        answer: 1,
        why: 'It cancels the bias exactly and gives you the flip rate as a free diagnostic.'
      },
      {
        q: 'Paired evaluation on the same items needs fewer examples because…',
        options: ['it is more accurate per item', 'item-difficulty variance cancels; only items where the systems differ carry information', 'it uses a better judge', 'it avoids position bias'],
        answer: 1,
        why: 'Items both systems get right or both get wrong contribute nothing to the comparison and are removed from the variance.'
      },
      {
        q: 'The most valuable source of eval cases is…',
        options: ['a public benchmark', 'production failures and complaints', 'synthetic examples from an LLM', 'the training set'],
        answer: 1,
        why: 'The cases you can invent are the ones the system already handles. The ones that broke carry the information.'
      }
    ],
    cards: [
      { q: 'Eval set size', a: '±9 points at n=100 near a 70% pass rate. Pair on the same items to cut the requirement several-fold.' },
      { q: 'The five judge biases', a: 'Position, verbosity, self-preference, formatting, sycophancy. Both-orders + length normalisation + a different model family fixes most of it.' },
      { q: 'Judge quality bar', a: 'Report Cohen’s κ against human labels; compare it against human–human agreement, which is the real ceiling.' },
      { q: 'What belongs in every eval report', a: 'Score, interval, baseline, per-slice breakdown, and cost and latency in the same table.' }
    ]
  });

  /* ------------------------------------------------------------------ 5.12 */
  ML.section({
    id: 'mlops', track: 'applied', num: '5.12', level: 2,
    title: 'MLOps: skew, shadow, canary, rollback',
    lede: 'The model is perhaps a fifth of the system. The rest is the machinery that gets features to it identically in training and serving, releases a new version without betting the business on it, and notices when the world has moved.',
    prereq: ['production'],
    related: ['production', 'evals', 'robustness'],
    html: `
${H.tldr([
      '<b>Training/serving skew</b> is the most common production ML bug and the hardest to see: the same feature computed two ways in two codebases. A feature store, or literally sharing the transformation code, is the structural fix.',
      'Release in stages: <b>shadow</b> (compute, do not act), <b>canary</b> (act for 1%, then 5%, then 25%), <b>full</b>. Each stage answers a different question, and each has an automatic rollback trigger defined <i>before</i> it starts.',
      'Monitor <b>inputs</b> (immediately available), <b>predictions</b> (immediately available) and <b>outcomes</b> (delayed by the label lag) as three separate systems. Only the last can detect concept drift.'
    ])}

<h2><span class="sn">5.12.1</span> Training/serving skew</h2>
${H.table(['Cause', 'Example', 'Fix'], [
      ['Two implementations', 'pandas in training, Java in serving', '<b>share the transformation code</b>, or a feature store that serves both'],
      ['Time travel', 'a training feature computed with data that did not exist yet at decision time', 'point-in-time correct joins; every feature carries an as-of timestamp'],
      ['Different defaults', 'training fills NA with the median, serving with 0', 'the imputer is part of the artefact, not the notebook'],
      ['Schema drift', 'an upstream column changes units or type', 'schema validation on every batch; fail loudly'],
      ['Different aggregation windows', '"last 30 days" means different things in the two systems', 'define windows once, in one place'],
      ['Feedback loops', 'the model’s own decisions become its next training data', 'log counterfactuals; hold out a random control slice permanently']
    ])}
${H.key('If a feature is computed in two places, it is computed two ways. This is not pessimism — it is what happens over a year of independent maintenance on two codebases. The structural fix is one definition, two callers.')}

<h2><span class="sn">5.12.2</span> The release ladder</h2>
${H.lab('rollout', 'A staged rollout with an injected regression', 'A new model version is 3% worse on a metric you monitor. Choose the rollout plan and the alert threshold, and see how many users are affected before the automatic rollback fires. There is a real trade here between caution and speed.')}
${H.table(['Stage', 'Question it answers', 'Traffic', 'Typical duration'], [
      ['<b>Shadow</b>', 'does it run, at what latency, and do its outputs look sane?', '100% computed, 0% acted on', '1–7 days'],
      ['<b>Canary 1%</b>', 'does it break anything catastrophically?', '1%', 'hours to a day'],
      ['Canary 5–25%', 'is the metric moving in the right direction?', '5–25%', 'until powered (§2.25)'],
      ['Full', '—', '100%', '—'],
      ['<b>Permanent holdback</b>', 'what is the model worth, cumulatively?', '0.5–2% never treated', 'forever']
    ])}
${H.note('The permanent holdback is the one most teams skip and most regret skipping. Without it you cannot answer "what is this model actually contributing?" a year later, and you have no clean data to retrain on that is free of the model’s own influence.')}

<h2><span class="sn">5.12.3</span> Monitoring, in three layers</h2>
${H.table(['Layer', 'Signal', 'Latency', 'Detects'], [
      ['<b>Inputs</b>', 'PSI/KS per feature, null rates, cardinality, range violations', 'immediate', 'covariate shift, upstream breakage'],
      ['<b>Predictions</b>', 'score distribution, alert volume, positive rate, confidence histogram', 'immediate', 'a sudden change in behaviour'],
      ['<b>Outcomes</b>', 'AUC, calibration, business metric, per-slice', 'delayed by the label lag', '<b>concept drift</b> — nothing else can'],
      ['System', 'p50/p99 latency, error rate, cost per request, cache hit rate', 'immediate', 'everything the SRE cares about']
    ])}
${H.pitfall('PSI is a rule of thumb, not a test: >0.1 "investigate", >0.25 "significant shift". It is sensitive to binning and to sample size, and it will fire on a harmless seasonal pattern while missing a genuine concept drift that leaves the input marginals unchanged. Alert on it, but never let it be the only thing you watch — <b>input monitoring cannot detect a change in $p(y\\mid x)$.</b>')}
${H.checklist([
      'Every model artefact is versioned with its training data snapshot, code commit and hyperparameters — reproducible from those three alone.',
      'Rollback is a config change, not a redeploy, and it has been tested this quarter.',
      'Alert thresholds are defined <i>before</i> the rollout begins, with an owner and a runbook.',
      'A permanent randomised holdback exists and is excluded from training data.',
      'Retraining is scheduled and its output is gated by the same eval as a manual release — <b>automated retraining without an automated gate is an automated way to ship a bad model.</b>',
      'The feature transformation code is shared between training and serving, or served from one store.'
    ])}

${H.probe([
      ['What is training/serving skew and how do you eliminate it?', 'The same feature computed differently in the two paths. Eliminate it structurally: one implementation with two callers, or a feature store; and add a monitor comparing a sample of serving-time features against recomputed training-time values.'],
      ['Why shadow before canary?', 'Shadow answers the operational questions — does it run, at what latency, are the outputs sane — with zero user risk. Canary answers the quality question and necessarily has user risk. Do them in that order.'],
      ['Input drift is flat but accuracy fell. What happened?', 'Concept drift: $p(y\\mid x)$ changed while $p(x)$ did not. Only labelled outcomes reveal it, which is why outcome monitoring cannot be replaced by input monitoring.'],
      ['Why keep a permanent holdback?', 'To measure the model’s cumulative contribution and to retain a slice of data uncontaminated by the model’s own decisions.'],
      ['Automated retraining ran and the model got worse. What was missing?', 'A gate. Retraining must be blocked by the same eval suite, baseline comparison and canary process as a human release.']
    ])}`,
    labs: {
      rollout: function (host) {
        const st = Viz.controls(host, [
          { k: 'reg', label: 'regression in the new version', min: 0, max: .1, step: .005, value: .03, fmt: v => '−' + (v * 100).toFixed(1) + ' pts' },
          { k: 'plan', label: 'rollout plan', type: 'select', value: 'staged', options: [
            { v: 'big', t: 'big bang — 100% immediately' }, { v: 'fast', t: 'fast: 10% → 100%' },
            { v: 'staged', t: 'staged: 1% → 5% → 25% → 100%' }, { v: 'slow', t: 'cautious: shadow → 1% → 5% → 25% → 50% → 100%' }] },
          { k: 'thresh', label: 'alert threshold', min: .005, max: .06, step: .005, value: .02, fmt: v => (v * 100).toFixed(1) + ' pts' },
          { k: 'traffic', label: 'requests per hour', min: 1000, max: 500000, step: 1000, value: 60000, fmt: v => v.toLocaleString() }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'detect', label: 'detected at hour', cls: 'key' },
          { k: 'affected', label: 'users served a worse model', cls: 'bad' },
          { k: 'pct', label: 'as a share of traffic' },
          { k: 'delay', label: 'time to full rollout if healthy', cls: 'good' },
          { k: 'note', label: 'verdict' }
        ]);
        const PLANS = {
          big: [[0, 1]],
          fast: [[0, .1], [6, 1]],
          staged: [[0, .01], [6, .05], [18, .25], [36, 1]],
          slow: [[0, 0], [12, .01], [24, .05], [42, .25], [66, .5], [90, 1]]
        };
        const S = Viz.surface(host, {
          height: 290,
          draw: function (ctx, w, h, T) {
            const plan = PLANS[st.plan];
            const HOURS = 120;
            const R = Num.rng(3);
            const base = .82;
            const share = t => { let s = 0; plan.forEach(p => { if (t >= p[0]) s = p[1]; }); return s; };
            const series = [], shares = [];
            let detected = null, affected = 0;
            for (let t = 0; t < HOURS; t++) {
              const sh = detected == null ? share(t) : 0;
              shares.push([t, sh]);
              const obs = base - st.reg * sh + R.normal(0, .004 / Math.sqrt(Math.max(.01, st.traffic / 20000)));
              series.push([t, obs]);
              if (detected == null) affected += sh * st.traffic;
              // detection needs the drop to exceed the threshold, and enough traffic on the new arm
              if (detected == null && t > 1 && (base - obs) > st.thresh && sh > 0) detected = t;
            }
            const P = Viz.plot(ctx, w, h, {
              xd: [0, HOURS], yd: [base - .12, base + .02], pad: { l: 56, r: 46, t: 16, b: 40 }
            }).frame({ xlabel: 'hours since release', ylabel: 'observed metric', yfmt: v => (v * 100).toFixed(1) + '%' });
            P.clip(() => {
              // traffic share as a shaded backdrop
              shares.forEach((s, i) => {
                if (!s[1]) return;
                ctx.fillStyle = T.c1; ctx.globalAlpha = .10 + .18 * s[1];
                ctx.fillRect(P.x(s[0]), P.pad.t, Math.max(1, P.x(1) - P.x(0)), P.ph);
                ctx.globalAlpha = 1;
              });
              P.line(series, { color: T.c1, width: 1.8 });
              P.hline(base, { color: T.c3, dash: [4, 4], label: 'old version' });
              P.hline(base - st.thresh, { color: T.c2, dash: [4, 4], label: 'alert threshold' });
              if (detected != null) P.vline(detected, { color: T.c2, width: 2.4, dash: false, label: 'rollback' });
            });
            const full = plan[plan.length - 1][0];
            out({
              detect: detected == null ? 'never (regression too small)' : 'hour ' + detected,
              affected: Math.round(affected).toLocaleString(),
              pct: (affected / (st.traffic * HOURS) * 100).toFixed(2) + '%',
              delay: full + ' h',
              note: detected == null ? '⚠ below the noise floor — needs a longer window or more traffic'
                : (affected < st.traffic * 2 ? '✓ contained' : 'many users exposed')
            });
          }
        });
        Viz.note(host, 'Try <b>big bang</b> with a 3-point regression: it is detected within an hour or two, and by then everyone has had it. Try <b>staged</b>: detection takes longer, because 1% of traffic is a noisy sample — but the number of affected users is two orders of magnitude smaller. <b>That is the actual trade, and it is not "caution versus speed" — it is exposure versus detection latency.</b> Now shrink the regression to 0.5 points and watch it slip past every plan: a small regression on a noisy metric is invisible at canary scale, which is why staged rollout is a containment strategy, not a detection strategy.');
      }
    },
    quiz: [
      {
        q: 'Training/serving skew is best prevented by…',
        options: ['more tests on the serving code', 'one feature-transformation implementation used by both paths', 'retraining more often', 'monitoring accuracy'],
        answer: 1,
        why: 'Two implementations diverge over time no matter how well tested. Share the code or serve from one store.'
      },
      {
        q: 'Shadow deployment means…',
        options: ['deploying to a staging environment', 'running the new model on real traffic but not acting on its outputs', 'deploying to 1% of users', 'A/B testing two models'],
        answer: 1,
        why: 'It answers the operational questions — does it run, latency, output sanity — at zero user risk.'
      },
      {
        q: 'Input drift monitoring cannot detect…',
        options: ['covariate shift', 'concept drift, where $p(y|x)$ changes but $p(x)$ does not', 'schema changes', 'null-rate spikes'],
        answer: 1,
        why: 'Only labelled outcomes reveal a change in the relationship. This is why outcome monitoring is a separate, unavoidable system.'
      },
      {
        q: 'The main benefit of a permanent randomised holdback is…',
        options: ['faster inference', 'measuring the model’s cumulative value and keeping uncontaminated training data', 'reducing cost', 'satisfying auditors'],
        answer: 1,
        why: 'Without it you cannot answer "what is this worth?" a year later, and every row of your training data has been influenced by the model.'
      }
    ],
    cards: [
      { q: 'Training/serving skew', a: 'The same feature computed two ways. Fix structurally: one implementation, two callers, or a feature store.' },
      { q: 'The release ladder', a: 'Shadow → canary 1% → 5–25% → full, plus a permanent holdback. Thresholds and owners defined before the rollout starts.' },
      { q: 'Three monitoring layers', a: 'Inputs (immediate, covariate shift), predictions (immediate, behaviour change), outcomes (delayed, concept drift).' },
      { q: 'PSI thresholds', a: '>0.1 investigate, >0.25 significant. A rule of thumb sensitive to binning — never the only signal.' },
      { q: 'Automated retraining', a: 'Must be gated by the same eval, baseline and canary as a human release, or it is an automated way to ship a bad model.' }
    ]
  });
})();
