/* ============================================================
   Tracks + home page
   ============================================================ */
(function () {
  'use strict';

  ML.track({ id: 'start', short: 'Part 0', title: 'Start here', blurb: 'What machine learning actually is, plus the exact linear algebra, matrix calculus, notation and probability the rest of the site consumes — built from zero, with things you can drag.' });
  ML.track({ id: 'foundations', short: 'Part 1', title: 'Mathematical & statistical foundations', blurb: 'Bayes, the distributions that recur, concentration, MLE/MAP, intervals and testing, causal inference, eigen/SVD, information theory, optimisation, sampling, Bayesian inference and floating point.' });
  ML.track({ id: 'classical', short: 'Part 2', title: 'Core & classical machine learning', blurb: 'Loss vs metric, bias–variance, regularization, GLMs, SVMs, trees, boosting, clustering, PCA, features, calibration, metrics, validation, ensembles, Gaussian processes, self-supervision, ranking and online experiments.' });
  ML.track({ id: 'deep', short: 'Part 3', title: 'Neural networks & deep learning', blurb: 'Backprop by hand, activations, initialisation, normalisation, optimisers, CNNs, RNNs, autodiff, reading a training curve, compression and adversarial robustness.' });
  ML.track({ id: 'llm', short: 'Part 4', title: 'LLMs & transformers', blurb: 'Attention derived, RoPE, the block with shapes, MoE and SSMs, pretraining, scaling laws, distributed training, SFT→DPO→GRPO→RLVR, LoRA, serving, decoding, evaluation, safety, multimodality, reasoning and speculative decoding.' });
  ML.track({ id: 'applied', short: 'Part 5', title: 'RAG, agents, MCP, production', blurb: 'Retrieval end to end, chunking, vector indexes, RAG vs fine-tune vs long context, agent loops, the 2026 stateless MCP, prompt injection, evals that do not lie, MLOps and cost per successful task.' });
  ML.track({ id: 'frontier', short: 'Part 6', title: 'Beyond the notebook', blurb: 'The deliberate exclusions, restored: reinforcement learning from MDPs up, bandits, diffusion, VAEs and GANs, graph networks, recommenders, time series, detection and segmentation, and privacy-preserving ML.' });
  ML.track({ id: 'interview', short: 'Part 7', title: 'The ML interview', blurb: 'The loop, question by question: breadth drills, whiteboard derivations, a repeatable system-design framework with six worked designs, from-scratch coding drills that run in the browser, the debugging case round, and a six-week plan.' });
  ML.track({ id: 'reference', short: 'Ref', title: 'Reference & drill room', blurb: 'Every number worth memorising, every formula in one place, the full glossary, sources, and a spaced drill over the whole curriculum.' });

  ML.homeHtml = function () {
    return `
<div class="hero">
  <div class="herocanvas" id="homeHero" aria-hidden="true"></div>
  <p class="eyebrow">an interactive course · mathematics → models → frontier systems</p>
  <h1>Learn machine learning<br>the way it is actually used.</h1>
  <p class="lede">Every idea here is built three times: the intuition you can say out loud, the derivation you can reproduce on a whiteboard, and a live thing you can drag until the formula becomes obvious. Nothing is a screenshot — every model on this site trains in your browser while you watch.</p>
  <div class="herostats" id="homeCounts"></div>
  <div class="btnrow" style="margin-top:22px">
    <a class="btn primary" href="#/what-is-ml">Start from zero →</a>
    <a class="btn" href="#/bayes">Jump to the mathematics</a>
    <a class="btn" href="#/attention">Jump to transformers</a>
    <a class="btn" href="#/interview-map">Prepare for interviews</a>
  </div>
</div>

<h2>The nine tracks</h2>
<div class="trackcards" id="trackCards"></div>

<h2>Four ways through</h2>
<div class="pathcards">
  <div class="pathcard">
    <p class="who">complete beginner · ~40 hours</p>
    <h4>Build the foundation</h4>
    <p>Assumes nothing past secondary-school algebra.</p>
    <ol>
      <li>All of <a href="#/what-is-ml">Part 0</a>, in order</li>
      <li><a href="#/bayes">Part 1</a> up to information theory</li>
      <li><a href="#/supervised-setup">Part 2</a> through metrics</li>
      <li><a href="#/nn-fundamentals">Part 3</a> to backprop, then stop and rest</li>
    </ol>
  </div>
  <div class="pathcard">
    <p class="who">interview in 6 weeks · ~25 hours</p>
    <h4>Prepare for the loop</h4>
    <p>Start at the end and work backwards into the gaps.</p>
    <ol>
      <li><a href="#/interview-map">7.1</a> — what each round actually scores</li>
      <li><a href="#/interview-breadth">7.2</a> breadth drills; follow every link you fail</li>
      <li><a href="#/interview-depth">7.3</a> derivations, then <a href="#/coding-round">7.5</a> code drills</li>
      <li><a href="#/ml-system-design">7.4</a> design, <a href="#/case-round">7.6</a> debugging, <a href="#/behavioural">7.7</a> the plan</li>
    </ol>
  </div>
  <div class="pathcard">
    <p class="who">engineer shipping an LLM feature · ~12 hours</p>
    <h4>Ship something that holds up</h4>
    <p>The parts that decide whether it survives real users.</p>
    <ol>
      <li><a href="#/decision-ladder">5.8</a> the decision ladder — read this first</li>
      <li><a href="#/rag">5.1</a>, <a href="#/chunking">5.10</a>, <a href="#/vector-search">5.9</a> retrieval end to end</li>
      <li><a href="#/evals">5.11</a> evals, <a href="#/production-ai">5.7</a> injection and cost</li>
      <li><a href="#/serving">4.14</a> and <a href="#/speculative">4.22</a> for the latency budget</li>
    </ol>
  </div>
  <div class="pathcard">
    <p class="who">already know the basics · ~20 hours</p>
    <h4>Fill the holes you can feel</h4>
    <p>The sections most people skip and later regret.</p>
    <ol>
      <li><a href="#/calibration">2.12</a> calibration and conformal prediction</li>
      <li><a href="#/optimization">1.14</a> duality and KKT, <a href="#/numerics">1.15</a> floating point</li>
      <li><a href="#/training-dynamics">3.11</a> reading a training curve</li>
      <li><a href="#/experimentation">2.25</a> peeking, CUPED and sequential tests</li>
    </ol>
  </div>
</div>

<h2>How this is built</h2>
<div class="grid2">
  <div>
    <h4>Every model here really runs</h4>
    <p>The neural network backpropagates; the SVM runs SMO; the Gaussian process inverts its own kernel matrix by Cholesky; the diffusion denoiser trains and then samples from noise; the retrieval lab runs BM25, dense retrieval, reciprocal-rank fusion and reranking over a corpus you can query. When a number appears in the text, a lab on the same page computes it — so you can move the inputs and watch the claim bend or break.</p>
    <h4>The mathematics is derived, not asserted</h4>
    <p>Derivations are laid out as a ladder: the line of algebra on the left, <i>the reason that line is allowed</i> on the right. That right-hand column is the part interviewers actually ask about.</p>
  </div>
  <div>
    <h4>What the marks mean</h4>
    <p><mark>Highlighted</mark> sentences are the ones to memorise verbatim. <span style="color:var(--red)">✗ Red lines</span> are the mistakes that cost offers. <span style="color:var(--amber)">⚑ Flags</span> are genuinely contested or fast-moving claims — say them with the caveat attached, every time. Boxes marked <span class="pill">optional depth</span> are collapsed by default: skip them on the first pass and they cost you nothing.</p>
    <h4>Navigating</h4>
    <p>Press <kbd>/</kbd> to search everything, <kbd>j</kbd> and <kbd>k</kbd> to move between sections, <kbd>t</kbd> to flip the theme, <kbd>m</kbd> to mark a section done, and <kbd>?</kbd> for the full list. Progress and quiz results are stored in your browser only — nothing is sent anywhere.</p>
  </div>
</div>

<h2>Everything on one page</h2>
<div id="fullIndex" class="small"></div>

<div class="footer">
  <p>Built as a live companion to <i>The AI/ML Study Notebook</i> (edition July 2026) — every section of that notebook is covered here, plus the material it deliberately excluded and a full interview track. Derivations follow the primary sources listed in <a href="#/sources">Sources</a>; contested claims are flagged rather than smoothed over.</p>
</div>`;
  };

  /* A live two-moons classifier training behind the hero. It is the site in
     miniature: real gradient descent, drawn as it happens. */
  ML.homeHero = function (host) {
    const data = Num.dataset('moons', 160, .18, 4);
    const net = Num.mlp([2, 10, 8, 1], { seed: 9, act: 'tanh' });
    const S = Viz.surface(host, {
      aspect: 0.74,
      draw: function (ctx, w, h, T) {
        const P = Viz.plot(ctx, w, h, { xd: [-2.2, 2.6], yd: [-1.6, 2.0], pad: { l: 2, r: 2, t: 2, b: 2 } });
        P.field(function (x, y) { return net.predict([x, y]); }, {
          step: 5, lo: 0, hi: 1,
          colors: function (t) {
            const s = Math.abs(t - .5) * 2;
            const c = t < .5 ? [70, 105, 210] : [205, 80, 60];
            return [c[0], c[1], c[2], Math.round(30 + 90 * s)];
          }
        });
        data.X.forEach((p, i) => P.dots([[p[0], p[1]]], { r: 2.6, color: data.y[i] ? T.c2 : T.c1, alpha: .95 }));
      }
    });
    let steps = 0;
    Viz.loop(function () {
      if (steps > 700) return;
      for (let i = 0; i < 3; i++) net.trainBatch(data.X, data.y, 0.05);
      steps += 3;
      if (steps % 9 === 0) S.redraw();
    });
  };
})();
