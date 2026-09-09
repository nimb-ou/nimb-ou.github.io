# The Interactive AI/ML Notebook

**Live: https://nimb-ou.github.io/ML_projects/**


A live, interactive course in machine learning and AI — **116 sections, 185
interactive labs, 307 quiz questions, 379 recall cards and ~121,000 words**,
built as a companion to *The AI/ML Study Notebook* (edition July 2026), covering
the material that notebook deliberately excluded, plus a full ML-interview
track.

Every model on this site trains in your browser. Nothing is a recorded
animation: the neural network really backpropagates, the SVM really runs SMO,
the Gaussian process really inverts its kernel matrix by Cholesky, the diffusion
denoiser really trains, the bandits really accumulate regret, and the RAG
pipeline really runs BM25, dense retrieval, RRF fusion and reranking over its
corpus. The code drills in §7.5 run your JavaScript against hidden test cases.

## Running it

Any static file server works — there is no build step for development:

```bash
python3 -m http.server 8781      # then open /site/index.html
```

## Building the single-file bundle

`build.py` inlines the CSS, all 31 scripts, and KaTeX (including its woff2
fonts as data URIs), producing a page that makes zero external requests:

```bash
python3 site/build.py            # -> site/dist/index.html   (~2.4 MB)
python3 site/build.py --body     # -> site/dist/artifact.html (body-only)
python3 site/build.py --pages    # -> docs/index.html         (what Pages serves)
```

After changing anything under `site/`, re-run the `--pages` build and commit
`docs/index.html` — that file, not `site/`, is what the public URL serves.

## Testing

`test/smoke.js` boots the whole site in jsdom, renders every section, mounts
every lab, **forces every canvas surface to draw synchronously** (surfaces defer
their first paint to `requestAnimationFrame`, where a throwing `draw()` would
otherwise go unnoticed), and reports any error plus content statistics:

```bash
node site/test/smoke.js          # needs jsdom on NODE_PATH or JSDOM_PATH
```

## Layout

```
index.html            page shell
css/app.css           design system (both themes, token-level)
js/00-core.js         registry, router, nav, progress, quiz, search, TOC rail
js/10-viz.js          canvas plotting, control widgets, animation player, flow diagrams
js/20-numeric.js      RNG, stats, linear algebra, models, metrics, retrieval
js/21-numeric-b.js    GPs, MCMC, bootstrap, bandits, ranking, ANN, DP, detection
js/30-labs-core.js    authoring helpers (H.*), shared lab drawing, code drills
js/content/*.js       the 116 sections, one file per track (or half-track)
vendor/katex/         vendored KaTeX (woff2 only)
build.py              single-file bundler
test/smoke.js         headless render + draw test over every section
```

## Adding a section

```js
ML.section({
  id: 'my-topic', track: 'classical', num: '2.27',
  level: 2,                        // 1 foundational · 2 core · 3 advanced
  title: '…', lede: '…',
  prereq: ['bias-variance'],       // rendered as chips under the title
  related: ['metrics'],            // rendered as a "read next" strip
  html: `…`,                       // $math$ and $$display math$$ both work
  labs: { key: function (host) { /* mount an interactive here */ } },
  quiz: [{ q, options, answer, why }],
  cards: [{ q, a }]                // recall cards feed the drill room
});
```

Sections are ordered by their printed `num`, not by script load order, so a new
file may add sections anywhere in an existing track.

### Authoring helpers

`H.tldr`, `H.intuition`, `H.analogy`, `H.practice`, `H.history`, `H.pitfall`,
`H.worked`, `H.probe`, `H.iq` (interview bank), `H.deriv` (a numbered
derivation with a justification column), `H.more` (collapsed optional depth),
`H.tabs`, `H.vs`, `H.svg`, `H.table`, `H.code`, `H.steps`, `H.checklist`,
`H.drill` (a code drill, mounted with `Labs.codeDrill`).

### Two rendering traps worth remembering

* `putImageData` ignores the canvas devicePixelRatio transform — use
  `Viz.blit`, which routes through an offscreen canvas.
* A CSS grid whose track list spans ~1000 implicit rows breaks paint once the
  page is scrolled. The in-page rail is absolutely positioned with a sticky
  child instead.
