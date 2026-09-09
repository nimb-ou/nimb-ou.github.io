/* ============================================================
   PART 7 — The ML interview (7.1 – 7.4)
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 7.1 */
  ML.section({
    id: 'interview-map', track: 'interview', num: '7.1', level: 1,
    title: 'How the loop is actually structured',
    lede: 'Interviews feel arbitrary until you know what each round is scoring. They are not testing whether you know machine learning; they are testing five specific, separable things, and almost everyone who fails does so on a dimension they never realised was being measured.',
    related: ['interview-breadth', 'ml-system-design', 'behavioural'],
    html: `
${H.tldr([
      'A typical loop has five rounds: <b>coding</b>, <b>ML breadth</b>, <b>ML depth</b>, <b>ML system design</b>, <b>behavioural / project deep-dive</b>. Each has a distinct rubric and a distinct failure mode.',
      'Breadth rounds fail on <i>hedging</i>; depth rounds fail on <i>memorised formulas with no derivation</i>; design rounds fail on <i>jumping to the model before the requirements</i>; behavioural rounds fail on <i>no measurable outcome</i>.',
      'The level you are hired at is decided mainly by <b>scope and ambiguity</b> — whether you were given a problem or found one — not by how much you know.'
    ])}

<h2><span class="sn">7.1.1</span> The five rounds, and what each actually scores</h2>
${H.table(['Round', 'The question behind the question', 'Strong signal', 'What sinks people'], [
      ['<b>Coding</b>', 'Can you write correct code under mild pressure, and reason about complexity?', 'Clarify → state the approach → write it → test it → state the complexity, in that order', 'Silent coding; not testing; a data structure that does not match the access pattern'],
      ['<b>ML breadth</b>', 'Have you internalised the fundamentals, or memorised them?', 'Crisp 60–90 second answers with a concrete example and a stated trade-off', '<b>Hedging.</b> "It depends" without saying on what reads as not knowing'],
      ['<b>ML depth</b>', 'Can you derive it, not just cite it?', 'Reaching for a whiteboard unprompted and deriving from a definition', 'Stating a formula you cannot motivate; being unable to say why a term is there'],
      ['<b>System design</b>', 'Can you build something that survives real traffic and real data?', 'Requirements → metrics → data → baseline → model → serving → monitoring → failure modes', '<b>Naming a model in the first two minutes.</b> The requirements are the interview'],
      ['<b>Behavioural / deep-dive</b>', 'What is it like when this goes wrong, and do you own outcomes?', 'One project, told in depth, with a number attached and a mistake admitted', 'No metric; "we" with no "I"; no failure you can describe honestly']
    ])}
${H.key('Every round is really asking one thing: <i>would I want this person debugging a production incident with me at 2am?</i> Depth, honesty about uncertainty, and structure under pressure are what actually get scored.')}

<h2><span class="sn">7.1.2</span> Company archetypes, and how the emphasis shifts</h2>
${H.table(['Archetype', 'Weighted toward', 'Distinctive round', 'Prepare by'], [
      ['Big tech, MLE', 'coding + system design', 'a full ML system design with real scale numbers', '§7.4, and drilling §7.5 to fluency'],
      ['Big tech, research scientist', 'depth + publications', 'derivations and a defence of your own paper', '§7.3, and re-reading your own work critically'],
      ['Applied-science / data science', 'stats + business framing', 'a case round on metric design and experiment analysis', '§7.6, §2.25, §2.13'],
      ['AI-native startup', 'shipping speed and LLM practice', '"build this in an hour" or a take-home', '§5.13, §5.11, §5.1'],
      ['Quant / trading', 'probability and mental maths', 'brainteasers under time pressure', '§1.1–§1.4, timed'],
      ['Enterprise / regulated', 'validation, monitoring, explainability', 'model-risk and governance questions', '§2.19, §2.17, §5.12']
    ])}

<h2><span class="sn">7.1.3</span> What actually decides the level</h2>
${H.table(['Level', 'Scope you owned', 'Ambiguity you absorbed', 'The sentence that demonstrates it'], [
      ['Junior (L3)', 'a well-specified task', 'the problem was handed to you', '"I improved recall by 6 points on the fraud model"'],
      ['Mid (L4)', 'a whole feature or model end to end', 'you chose the approach', '"I owned the model from framing to launch and it cut manual review 30%"'],
      ['Senior (L5)', 'a system, with others contributing', 'you chose the <i>problem</i>', '"I found that our metric was wrong, changed it, and re-planned the roadmap around it"'],
      ['Staff+ (L6+)', 'multiple systems or a technical direction', 'you defined what was worth doing', '"I convinced the org to stop a project, which saved two quarters"']
    ])}
${H.note('This is why "tell me about a project" is the most pivotal question in the loop. The same work described as a task, a feature or a system-with-a-judgement-call maps to three different levels. Be accurate — inflating this is easy to detect on the follow-ups — but do not undersell scope you genuinely owned.')}

${H.lab('readiness', 'Where is your prep actually weak?', 'Rate yourself honestly on each dimension. The output is an ordered plan with the specific sections to read, weighted by how much each dimension moves a hiring decision.')}

<h2><span class="sn">7.1.4</span> The habits that raise every score</h2>
${H.steps([
      '<b>Restate the question before answering.</b> Ten seconds, and it prevents the most expensive failure mode — a brilliant answer to a question nobody asked.',
      '<b>Answer, then elaborate.</b> Give the one-sentence answer first, then the nuance. Interviewers taking notes need the answer; if you build to it for three minutes they will have stopped listening.',
      '<b>Name the trade-off explicitly.</b> "I would use X; the cost is Y; I would revisit it if Z." This one pattern is the difference between mid and senior scoring on almost every rubric.',
      '<b>Say what would change your mind.</b> It signals genuine understanding rather than a memorised position, and it is nearly impossible to fake.',
      '<b>Admit the edge of your knowledge, then reason forward.</b> "I have not implemented that, but from first principles I would expect…" scores far better than a confident wrong answer — which is the single most damaging thing you can do.',
      '<b>Ask before you assume.</b> In design rounds especially, an unasked question is a wrong assumption you will build on for forty minutes.'
    ])}
${H.pitfall('The most common self-inflicted wound is <b>bluffing</b>. Interviewers ask follow-ups precisely to find the edge of your knowledge; discovering that you invented an answer converts one gap into a credibility problem across the whole loop. "I do not know, here is how I would find out" costs almost nothing.')}

${H.iq('Warm-up: the questions that open loops', [
      {
        q: 'Walk me through a machine learning project you are proud of.',
        level: 'all levels',
        a: `<p>Use a fixed structure and keep it to three minutes unless invited to go longer: <b>problem and why it mattered → what made it hard → what you did → the number → what you got wrong</b>.</p>
<p>Anchor on a metric with a business unit attached: not "improved AUC to 0.91" but "raised precision at the fixed alert budget from 31% to 44%, which removed about 900 manual reviews a week". Then volunteer the mistake before they dig for it — it converts an interrogation into a conversation and it is the strongest available signal of seniority.</p>`,
        follow: ['What would you do differently with twice the time?', 'What was your specific contribution versus your team’s?', 'How did you know the improvement was real?', 'What broke after launch?'],
        red: 'Describing only the modelling. The framing, the data and the launch are where the seniority signal lives.'
      },
      {
        q: 'How do you keep up with the field?',
        level: 'all levels',
        a: `<p>Be specific and honest. Two or three concrete sources, one thing you actually read recently, and what you did with it. "I read the Chinchilla paper and it changed how I size training runs" beats a list of newsletters.</p>
<p>The question is really testing whether you filter. Saying you read everything is not credible; saying you follow two areas closely and skim the rest is.</p>`,
        follow: ['What is a recent result that changed your mind?', 'What do you think is over-hyped right now?'],
        red: 'Naming only Twitter/X threads or LinkedIn posts, with no primary sources.'
      },
      {
        q: 'What is the most interesting thing you have read recently that turned out to be wrong?',
        level: 'senior',
        a: `<p>An excellent question to be ready for, because it tests intellectual honesty and whether you evaluate claims rather than collecting them. Good answers involve a result that failed to replicate, a benchmark that turned out to be contaminated (§5.11), or a technique whose gains vanished under a fair comparison — batch size confounds, tuned-baseline effects, or evaluation on a step axis instead of a token axis (§3.12).</p>`,
        follow: ['How would you have designed the experiment to catch that?'],
        red: 'Not having one. Everyone who reads the literature has been wrong about something recently.'
      }
    ])}`,
    labs: {
      readiness: function (host) {
        const DIMS = [
          { k: 'coding', label: 'Coding — arrays, hashing, complexity, from-scratch ML', weight: 1.0, secs: [['7.5', 'coding-round'], ['0.5', 'python-toolkit']] },
          { k: 'breadth', label: 'ML breadth — bias/variance, regularization, metrics, trees', weight: 1.2, secs: [['7.2', 'interview-breadth'], ['2.26', 'part2-recall'], ['2.13', 'metrics']] },
          { k: 'depth', label: 'ML depth — deriving backprop, attention, the normal equations', weight: 1.1, secs: [['7.3', 'interview-depth'], ['0.7', 'matrix-calculus'], ['3.2', 'backprop']] },
          { k: 'stats', label: 'Statistics — intervals, power, causal inference, experiments', weight: 0.9, secs: [['1.6', 'intervals'], ['2.25', 'experimentation'], ['1.7', 'causal']] },
          { k: 'design', label: 'ML system design — requirements, serving, monitoring', weight: 1.3, secs: [['7.4', 'ml-system-design'], ['5.12', 'mlops'], ['4.14', 'serving']] },
          { k: 'llm', label: 'LLMs and applied AI — attention, RAG, agents, cost', weight: 1.1, secs: [['4.3', 'attention'], ['5.1', 'rag'], ['5.13', 'decision-ladder']] },
          { k: 'behav', label: 'Behavioural — project narrative, conflict, ownership', weight: 1.0, secs: [['7.7', 'behavioural']] },
          { k: 'case', label: 'Case / debugging — "the metric dropped overnight"', weight: 0.9, secs: [['7.6', 'case-round'], ['3.12', 'training-dynamics'], ['2.18', 'production']] }
        ];
        const el = ML.el;
        const spec = DIMS.map(d => ({ k: d.k, label: d.label, min: 1, max: 5, step: 1, value: 3, fmt: v => ['—', 'shaky', 'basic', 'solid', 'strong', 'could teach it'][v] }));
        const st = Viz.controls(host, spec.concat([
          { k: 'weeks', label: 'weeks until the loop', min: 1, max: 12, step: 1, value: 6, fmt: v => v + (v === 1 ? ' week' : ' weeks') }
        ]), () => draw());
        const planHost = el('div');
        host.appendChild(planHost);

        const S = Viz.surface(host, {
          height: 250,
          draw: function (ctx, w, h, T) {
            const items = DIMS.map(d => ({ k: d.label.split(' —')[0], v: (5 - st[d.k]) * d.weight, c: (5 - st[d.k]) * d.weight > 2.2 ? T.c2 : (5 - st[d.k]) * d.weight > 1 ? T.c4 : T.c3 }));
            const P = Viz.plot(ctx, w, h, { xd: [0, 1], yd: [0, 1], pad: { l: 12, r: 60, t: 24, b: 12 } });
            P.hbars(items, { max: 5.2, maxH: 16, fmt: v => v.toFixed(1) });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('weighted gap — longer bar = more of your remaining prep time belongs here', 12, 14);
          }
        });

        function draw() {
          const ranked = DIMS.map(d => ({ d: d, gap: (5 - st[d.k]) * d.weight })).sort((a, b) => b.gap - a.gap);
          const total = ranked.reduce((a, r) => a + r.gap, 0) || 1;
          const hours = st.weeks * 8;
          planHost.innerHTML =
            '<p class="boxtitle" style="margin-top:14px">your plan — about ' + hours + ' hours over ' + st.weeks + ' week' + (st.weeks === 1 ? '' : 's') + '</p>' +
            '<ol class="steps">' + ranked.filter(r => r.gap > 0).map(r =>
              '<li><b>' + r.d.label.split(' —')[0] + '</b> — ' + Math.round(hours * r.gap / total) + ' h. ' +
              'Read ' + r.d.secs.map(s => '<a href="#/' + s[1] + '">§' + s[0] + '</a>').join(', ') + '.</li>').join('') +
            '</ol>' +
            (ranked[0].gap < 0.6
              ? '<p class="small" style="color:var(--green)"><b>You are broadly ready.</b> Spend the remaining time on mock interviews out loud — at this point fluency, not knowledge, is the binding constraint.</p>'
              : '<p class="small">Work top-down. Do not start at the bottom of this list because it is more comfortable — the ordering already accounts for how much each dimension moves a hiring decision.</p>');
          S.redraw();
        }
        draw();
        Viz.note(host, 'The weights are not equal: system design and ML breadth carry more because they appear in nearly every loop and discriminate most strongly between candidates. A "shaky" on system design costs more than a "shaky" on statistics unless you are interviewing for a data-science role, where the weights invert.');
      }
    },
    quiz: [
      {
        q: 'The most common failure in an ML system design round is…',
        options: ['not knowing enough architectures', 'proposing a model before establishing requirements and metrics', 'poor coding', 'weak mathematics'],
        answer: 1,
        why: 'The requirements conversation *is* the interview. Naming a model in the first two minutes forfeits most of the rubric.'
      },
      {
        q: 'What most determines the level you are hired at?',
        options: ['years of experience', 'the scope and ambiguity you have owned', 'the number of papers published', 'breadth of frameworks known'],
        answer: 1,
        why: 'Being given a problem, choosing an approach, choosing the problem, or defining what is worth doing — those four map to L3 through L6.'
      },
      {
        q: 'In a breadth round, "it depends" is…',
        options: ['always the right answer', 'only useful if you immediately say what it depends on', 'a sign of seniority', 'best avoided entirely'],
        answer: 1,
        why: 'Unqualified hedging reads as not knowing. "It depends on whether the classes are balanced — if they are, X; if not, Y" reads as understanding.'
      }
    ],
    cards: [
      { q: 'The five rounds', a: 'Coding, ML breadth, ML depth, system design, behavioural — each with a separate rubric and a distinct failure mode.' },
      { q: 'The answer pattern that scores', a: 'One-sentence answer → the trade-off → what would change your mind.' },
      { q: 'The levelling axis', a: 'Scope and ambiguity: given a task / chose the approach / chose the problem / defined what was worth doing.' },
      { q: 'The most damaging interview behaviour', a: 'Bluffing. Follow-ups exist to find the edge of your knowledge, and being caught costs credibility across the whole loop.' }
    ]
  });

  /* ------------------------------------------------------------------ 7.2 */
  ML.section({
    id: 'interview-breadth', track: 'interview', num: '7.2', level: 2,
    title: 'The breadth round: sixty questions with model answers',
    lede: 'Rapid-fire fundamentals. Each of these should take you sixty to ninety seconds: the answer, one concrete example, and the trade-off. Practise them out loud — the gap between knowing and saying is the entire difficulty of this round.',
    related: ['interview-depth', 'part2-recall', 'metrics'],
    html: `
${H.tldr([
      'The format: a crisp answer, a concrete instance, and a stated trade-off. Sixty to ninety seconds. Then stop talking.',
      'Every question below links to the section that derives it. Any question you fumble is a section to re-read — that is how to use this page.',
      'Use the drill at the bottom to test yourself under something resembling real conditions: question first, answer hidden, self-graded.'
    ])}

${H.lab('drill', 'The rapid-fire drill', 'One question at a time, answer hidden. Say your answer out loud <i>before</i> revealing — reading a good answer feels like knowing it, and it is not the same thing. Your score by topic is tracked below.')}

<h2><span class="sn">7.2.1</span> Fundamentals and generalization</h2>
${H.iq('Bias, variance, overfitting, regularization', [
      {
        q: 'Explain the bias–variance trade-off, and where it breaks down.',
        level: 'core',
        a: `<p>Expected test error decomposes into $\\text{bias}^2 + \\text{variance} + \\text{irreducible noise}$. Bias is error from wrong assumptions — a linear model on a curved relationship. Variance is sensitivity to the particular training sample — a deep unpruned tree. Capacity trades one for the other, and classically the sum is U-shaped.</p>
<p><b>Where it breaks down:</b> in the over-parameterised regime the curve descends again — <i>double descent</i> (§2.2). Past the interpolation threshold, adding parameters keeps improving test error because the implicit bias of the optimiser selects a low-norm interpolant among the many that fit. This is why "more parameters means more overfitting" is not a reliable statement about modern networks.</p>`,
        follow: ['Which does bagging reduce? Which does boosting reduce?', 'Where does regularization sit in this decomposition?', 'Can you have high bias and high variance at once?'],
        red: 'Reciting the decomposition without being able to give a concrete high-bias and high-variance model.'
      },
      {
        q: 'L1 versus L2 regularization — why does L1 produce exact zeros?',
        level: 'core',
        a: `<p>L2 adds $\\lambda\\|w\\|_2^2$, whose gradient $2\\lambda w$ shrinks toward zero proportionally and therefore never reaches it. L1 adds $\\lambda\\|w\\|_1$, whose subgradient is $\\lambda\\,\\mathrm{sign}(w)$ — a <b>constant</b> pull regardless of how small $w$ is. Once the data's gradient is smaller than $\\lambda$, the coefficient is pinned at exactly zero.</p>
<p>Geometrically: the L1 constraint region is a diamond with corners on the axes, and the loss contours touch it at a corner. Bayesian reading: L2 is a Gaussian prior, L1 a Laplace prior whose sharp peak at zero puts mass there (§1.5, §2.3).</p>`,
        follow: ['When would you use elastic net?', 'Does L1 give a unique solution under collinearity?', 'How does weight decay relate to L2 in Adam?'],
        red: 'Saying "L1 gives sparsity" with no mechanism. The constant-versus-proportional gradient is the answer.'
      },
      {
        q: 'What is the difference between a parameter and a hyperparameter?',
        level: 'basic',
        a: `<p>Parameters are fitted by the learning algorithm from the training data — weights, split thresholds, cluster centres. Hyperparameters are set before or around training and govern how the fitting happens: learning rate, tree depth, $\\lambda$, $k$.</p>
<p>The practical consequence is where each is chosen: parameters on the training set, hyperparameters on a validation set, and neither on the test set. Selecting hyperparameters on the test set is a subtle and common form of leakage (§2.14).</p>`,
        follow: ['How do you avoid overfitting the validation set when tuning?', 'Is the number of epochs a hyperparameter?']
      },
      {
        q: 'Why does adding features almost always improve training error but not test error?',
        level: 'core',
        a: `<p>Training error is non-increasing in capacity — an extra feature can always be given zero weight, so the optimum cannot get worse. Test error is a different quantity: each additional feature adds estimation variance ($O(d/n)$ for linear models) and an opportunity to fit noise.</p>
<p>The clean framing: adding a feature always adds variance and only sometimes reduces bias. It is worth it when the bias reduction exceeds the variance addition — which is exactly what a validation set measures.</p>`,
        follow: ['How does regularization change that calculation?', 'What about with $n \\gg d$?']
      },
      {
        q: 'What is the curse of dimensionality, concretely?',
        level: 'core',
        a: `<p>In high dimensions, volume concentrates near the boundary and distances concentrate: the ratio between the nearest and farthest neighbour of a point tends to 1, so "nearest neighbour" stops meaning anything. To keep the same sample density you need exponentially more data.</p>
<p>Concretely: to cover a unit hypercube at 10% resolution per axis needs $10^d$ points — 100 in 2-D, $10^{10}$ in 10-D. This is why k-NN and kernel density estimation degrade badly with dimension while linear models and trees, which do not rely on local neighbourhoods, degrade more gracefully (§2.5, §2.10).</p>`,
        follow: ['Why does it not seem to hurt deep networks on images?', 'What is the manifold hypothesis?']
      }
    ])}

<h2><span class="sn">7.2.2</span> Metrics and evaluation</h2>
${H.iq('Choosing, reading, and defending a metric', [
      {
        q: 'When is ROC-AUC misleading, and what would you use instead?',
        level: 'core',
        a: `<p>Under heavy class imbalance. AUC integrates over the false-positive rate, whose denominator is the huge negative class, so a large absolute number of false positives barely moves it. A model can have AUC 0.95 and precision 2% at any usable threshold.</p>
<p>Use <b>average precision</b> (area under precision–recall), which has the positive class in both denominators, or <b>recall at a fixed alert budget</b> — the number the team working the alerts actually experiences. And always state the baseline: AP's chance level is the positive rate, not 0.5 (§2.13).</p>`,
        follow: ['What is the chance level for AP?', 'Why is AUC threshold-independent, and when is that a disadvantage?', 'What does AUC mean probabilistically?'],
        red: 'Not knowing that AUC equals the probability a random positive is ranked above a random negative.'
      },
      {
        q: 'Your model is well calibrated but has poor AUC. What does that mean, and can the reverse happen?',
        level: 'senior',
        a: `<p>Calibration and discrimination are orthogonal. Predicting the base rate for every example is perfectly calibrated and has AUC 0.5 — it is honest and useless. Conversely a model can rank perfectly (AUC 1.0) and be badly calibrated if its probabilities are systematically compressed or inflated; a monotone recalibration (Platt, isotonic) fixes the probabilities without changing the ranking at all.</p>
<p>Which you need depends on the decision: ranking-only tasks need discrimination; anything that multiplies the probability by a monetary amount needs calibration (§2.12).</p>`,
        follow: ['Which calibration method, and why?', 'What does a reliability diagram show?', 'Does isotonic regression change AUC?']
      },
      {
        q: 'How do you choose a decision threshold?',
        level: 'core',
        a: `<p>From costs, not from 0.5. If a false negative costs $c_{FN}$ and a false positive $c_{FP}$, expected cost is minimised at $t^\\star = c_{FP}/(c_{FP}+c_{FN})$ for a calibrated model. At 20:1 that is 0.048, not 0.5.</p>
<p>In practice you often optimise a constrained version instead: maximise recall subject to a fixed alert volume the team can actually work, or subject to a precision floor set by user tolerance. Pick the threshold on validation data, never on test, and re-check it after any recalibration (§0.8, §2.13).</p>`,
        follow: ['What if the costs are unknown?', 'How does the threshold interact with class weighting during training?']
      },
      {
        q: 'F1 versus precision–recall trade-off — when is F1 the wrong summary?',
        level: 'core',
        a: `<p>F1 is the harmonic mean, which implicitly weights precision and recall equally. That is almost never the actual business trade. For screening (a missed cancer is catastrophic) recall dominates; for automated actioning (a wrong block is expensive) precision dominates.</p>
<p>F1 also ignores true negatives entirely and is not comparable across datasets with different base rates. Use $F_\\beta$ if you must have one number and can state $\\beta$; better, report precision and recall at the operating point you will actually use.</p>`,
        follow: ['What does $F_\\beta$ do?', 'Why does F1 ignore true negatives?']
      }
    ])}

<h2><span class="sn">7.2.3</span> Models</h2>
${H.iq('Trees, boosting, SVMs, clustering, linear models', [
      {
        q: 'Random forest versus gradient boosting — when do you reach for each?',
        level: 'core',
        a: `<p><b>Random forest</b>: independent deep trees on bootstrap samples with feature subsampling, averaged. Reduces variance, hard to overfit, trivially parallel, few hyperparameters that matter. My default when I want a strong result in ten minutes with no tuning.</p>
<p><b>Gradient boosting</b>: shallow trees fitted sequentially to the gradient of the loss. Reduces bias, usually 1–3 points better on tabular data, and <i>will</i> overfit without shrinkage, subsampling and early stopping. My default when the accuracy matters and I have time to tune (§2.7, §2.8).</p>`,
        follow: ['What does the learning rate do in boosting, and how does it interact with the number of trees?', 'Why does XGBoost use second-order information?', 'Which handles missing values natively?'],
        red: 'Not knowing that boosting attacks bias and bagging attacks variance.'
      },
      {
        q: 'Explain the kernel trick without the phrase "map to higher dimensions".',
        level: 'senior',
        a: `<p>The SVM's dual objective depends on the training points only through inner products $x_i^\\top x_j$. Any function $k(x_i,x_j)$ that is a valid inner product in <i>some</i> space can be substituted directly, so you get the effect of that space's geometry while only ever computing an $n\\times n$ matrix of kernel values. You never construct a feature vector, which is what makes an infinite-dimensional feature space (RBF) computationally finite.</p>
<p>Mercer's condition is the requirement: the kernel matrix must be positive semi-definite for every finite sample (§2.6, §1.12).</p>`,
        follow: ['Why can the primal not do this?', 'What does the RBF length-scale control?', 'Why is this $O(n^2)$ in memory?']
      },
      {
        q: 'How do you choose $k$ in k-means, and what are you assuming by using it at all?',
        level: 'core',
        a: `<p>Choose by elbow on inertia (weak), silhouette score (better), gap statistic (principled), or — usually best — by what the downstream consumer needs. All of these are diagnostics, not answers; if the clusters have to be actioned by a team, the number of clusters is partly an operational decision.</p>
<p><b>What you are assuming:</b> spherical, similar-sized, similar-density clusters, because k-means minimises within-cluster squared Euclidean distance. Elongated or nested structure breaks it, which is when you reach for GMM (elliptical), DBSCAN (density, arbitrary shape) or spectral clustering (connectivity) (§2.9).</p>`,
        follow: ['Why does k-means++ matter?', 'Why must you scale features first?', 'What does DBSCAN give you that k-means cannot?']
      },
      {
        q: 'Why is logistic regression called regression, and what exactly is it modelling?',
        level: 'basic',
        a: `<p>It is a generalised linear model: it performs a <i>linear regression on the log-odds</i>. $\\log\\frac{p}{1-p} = w^\\top x + b$, so the linear part predicts a real number and the logistic function maps it into $(0,1)$. The coefficients are therefore log-odds ratios — a one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$, which is why they are directly interpretable and why the model survives in regulated settings.</p>
<p>It is fitted by maximum likelihood, which for the Bernoulli likelihood is exactly cross-entropy minimisation (§1.5, §2.4).</p>`,
        follow: ['Why not use squared loss?', 'What happens on perfectly separable data?', 'What is the link function, and what makes a GLM a GLM?']
      },
      {
        q: 'Naive Bayes is "naive". Why does it work anyway?',
        level: 'core',
        a: `<p>It assumes features are conditionally independent given the class, which is essentially never true. It works because <b>classification only needs the argmax to be right, not the probabilities</b>. Correlated features cause the posterior to be badly over-confident — the same evidence is counted several times — but they often do not change which class is largest.</p>
<p>Consequence: excellent classifier, terrible probability estimate. Never feed a Naive Bayes probability into a downstream expected-value calculation without recalibration (§2.5, §2.12).</p>`,
        follow: ['Why is Laplace smoothing necessary?', 'Why is it still competitive for text?']
      }
    ])}

<h2><span class="sn">7.2.4</span> Deep learning and LLMs</h2>
${H.iq('Networks, transformers, and the modern stack', [
      {
        q: 'Why divide by $\\sqrt{d_k}$ in attention?',
        level: 'core',
        a: `<p>If query and key entries are independent with unit variance, their dot product over $d_k$ dimensions has variance $d_k$ and therefore standard deviation $\\sqrt{d_k}$. Feeding scores of that magnitude into a softmax saturates it: one weight approaches 1, the rest approach 0, and the gradient through the softmax vanishes.</p>
<p>Dividing by $\\sqrt{d_k}$ restores unit variance so the softmax stays in its responsive range at initialisation. It is a variance-control argument, identical in spirit to He/Xavier initialisation (§4.3, §3.4).</p>`,
        follow: ['What if you divided by $d_k$ instead?', 'Where else does this variance argument appear?'],
        red: 'Saying "to normalise" without the variance argument. That is the whole content of the question.'
      },
      {
        q: 'What is the KV cache, how big is it, and why does it dominate serving?',
        level: 'core',
        a: `<p>During autoregressive decoding, the keys and values for all previous tokens are needed at every step. Recomputing them is quadratic; caching makes it linear. Size: $2 \\times L \\times n_{kv} \\times d_{head} \\times S \\times B \\times \\text{bytes}$.</p>
<p>For a 70B model at bf16 with 8 KV heads, 80 layers, head dim 128, 8k context: about 10 GB for a single sequence. That is the memory that limits concurrency — not the weights, which are shared. It is why GQA, MLA and KV quantization exist, and why paged attention was worth a paper (§4.7, §4.14).</p>`,
        follow: ['How does GQA reduce it, and what does it cost?', 'Why does the batch size multiply it but not the weights?']
      },
      {
        q: 'Why do transformers need positional encoding when CNNs and RNNs do not?',
        level: 'core',
        a: `<p>Self-attention is permutation-equivariant: shuffle the input tokens and the outputs shuffle identically. It has no notion of order at all. RNNs get order from the recurrence; CNNs get locality from the kernel. Transformers must have it injected.</p>
<p>RoPE is the modern answer because it encodes <i>relative</i> position by rotating queries and keys, so the attention score depends on $m-n$ rather than on $m$ and $n$ separately — which is what makes context extension by frequency scaling (YaRN, NTK-aware) possible at all (§4.4).</p>`,
        follow: ['Why is relative better than absolute?', 'How does RoPE extrapolate beyond its training length?']
      },
      {
        q: 'What actually happens in RLHF, and what is DPO doing differently?',
        level: 'senior',
        a: `<p><b>RLHF:</b> collect pairwise human preferences, fit a reward model under the Bradley–Terry likelihood, then optimise the policy with PPO against that reward plus a KL penalty to the reference model. Three stages, three failure modes, and reward hacking is the persistent one.</p>
<p><b>DPO:</b> observes that the optimal policy under a KL-constrained reward objective has a closed form, and inverts it — so the reward model can be substituted away and you optimise a simple classification loss on preference pairs directly. One stage, no sampling loop, far more stable. The trade is that you cannot generate new samples and score them, so DPO cannot explore beyond the preference data you have (§4.12).</p>`,
        follow: ['Where does the $\\beta$ in DPO come from?', 'What is GRPO and why did it appear?', 'What is RLVR, and why does it not need a reward model?']
      },
      {
        q: 'Batch norm versus layer norm — why did transformers pick layer norm?',
        level: 'core',
        a: `<p>Batch norm normalises each feature across the <i>batch</i>, so its statistics depend on which other examples are present. That is fine for images at large batch, and it fails for sequences: batch statistics are unstable with variable-length inputs, meaningless at batch size 1, and require a separate running-statistics path at inference that is a persistent source of train/test mismatch.</p>
<p>Layer norm normalises across the <i>features</i> of each token independently — no batch dependence, identical behaviour in training and inference, works at batch size 1. RMSNorm drops the mean-centring as well and is now standard, because the centring turns out not to matter and skipping it is faster (§3.6, §4.6).</p>`,
        follow: ['Why does pre-norm train more stably than post-norm?', 'What is RMSNorm dropping, and why is it fine?']
      },
      {
        q: 'When would you not use an LLM?',
        level: 'core',
        a: `<p>When something cheaper works. A regex for extraction with a fixed format; a classifier when you have a few thousand labels and a fixed label set; a SQL query when the answer is in a database; a rules engine when the logic is genuinely a decision table someone can maintain.</p>
<p>Also when the requirements exclude it: hard latency budgets under ~50 ms, strict determinism, full auditability of the decision, or a per-request cost ceiling that inference cannot meet. The decision ladder in §5.13 is the version of this you can walk an interviewer through.</p>`,
        follow: ['At what point does fine-tuning beat prompting?', 'How would you decide between RAG and long context?']
      }
    ])}

<h2><span class="sn">7.2.5</span> Data, statistics, and practice</h2>
${H.iq('The questions that separate the practitioners', [
      {
        q: 'What is data leakage? Give three examples that are easy to miss.',
        level: 'core',
        a: `<p>Any information in the features that would not be available at prediction time, or that comes from the target.</p>
<ol>
<li><b>Temporal:</b> aggregate features computed over the full history, including after the prediction date. The fix is point-in-time correct joins.</li>
<li><b>Preprocessing:</b> scaler, imputer or encoder fitted on train+test. Small but consistent optimism (§0.8).</li>
<li><b>Group:</b> the same customer, patient or session split across train and test — the model memorises the entity rather than learning the relationship. Use GroupKFold.</li>
</ol>
<p>The tell is a suspiciously high validation score, and the discipline is to be more suspicious of good results than of bad ones (§2.11).</p>`,
        follow: ['How would you detect leakage you did not anticipate?', 'What is target encoding leakage and how do you fix it?'],
        red: 'Only knowing the preprocessing case. Temporal and group leakage cause far more production failures.'
      },
      {
        q: 'Your offline metric improved but the online A/B test showed nothing. Give me five hypotheses.',
        level: 'senior',
        a: `<ol>
<li><b>Underpowered test</b> — the effect is real but smaller than the MDE. Check the interval, not the p-value (§2.25).</li>
<li><b>Offline/online metric mismatch</b> — you improved AUC, the business cares about conversions at a fixed alert budget.</li>
<li><b>Distribution mismatch</b> — the offline set was not sampled from live traffic; historical data is filtered by the previous system's decisions.</li>
<li><b>Implementation skew</b> — features computed differently in serving than in training (§5.12).</li>
<li><b>System effects</b> — latency increased, or a downstream component (a ranker, a cache, a rule layer) absorbed the improvement before it reached the user.</li>
</ol>
<p>Debug in that order: it is cheapest first, and the first two explain the majority of cases.</p>`,
        follow: ['How would you distinguish hypothesis 4 from hypothesis 3?', 'What would you have done differently before launching?']
      },
      {
        q: 'How do you handle missing data?',
        level: 'core',
        a: `<p>First diagnose the mechanism, because it determines what is safe. <b>MCAR</b> (missing at random, unrelated to anything): dropping rows is unbiased, just wasteful. <b>MAR</b> (explainable by observed features): imputation conditional on those features is valid. <b>MNAR</b> (missingness depends on the unobserved value itself — income missing because it is high): no imputation is unbiased, and the fact of missingness is itself a feature.</p>
<p>Practically: add a <code>was_missing</code> indicator almost always; use median or model-based imputation for MAR; let LightGBM and XGBoost learn a default direction natively; and fit the imputer inside the pipeline so it never sees the test set (§2.11).</p>`,
        follow: ['Why is mean imputation usually worse than median?', 'What does MICE do?', 'When is missingness the most predictive feature you have?']
      },
      {
        q: 'How would you detect that a deployed model has degraded, before anyone complains?',
        level: 'senior',
        a: `<p>Three independent monitors, because they detect different things (§5.12).</p>
<ol>
<li><b>Input drift</b> — PSI or KS per feature, null rates, cardinality. Available immediately; catches upstream breakage and covariate shift.</li>
<li><b>Prediction drift</b> — the score distribution and positive rate. Immediate; catches a sudden behaviour change even when inputs look normal.</li>
<li><b>Outcome monitoring</b> — AUC, calibration and the business metric on labels as they arrive. Delayed by the label lag, and the <b>only</b> thing that can detect concept drift, where $p(y\\mid x)$ changed but $p(x)$ did not.</li>
</ol>
<p>Plus a permanent randomised holdback so you can always answer "compared to what?".</p>`,
        follow: ['What is the label lag in your last project and how did you handle it?', 'Why can input monitoring not detect concept drift?']
      },
      {
        q: 'Explain p-values to a product manager.',
        level: 'core',
        a: `<p>"If the change genuinely did nothing, how surprising would this result be?" A p-value of 0.03 means data this extreme would appear 3% of the time in a world where the change had no effect. It is <b>not</b> the probability that the change works, and it says nothing about how big the effect is.</p>
<p>Then redirect: "The number you actually want is the confidence interval — the range of effects consistent with what we saw. It tells you both whether we are confident there is an effect and whether the effect is large enough to matter." That redirection is the part that scores (§1.6).</p>`,
        follow: ['What is a confidence interval, in the same register?', 'What does peeking do to the p-value?']
      }
    ])}

${H.key('If you fumbled more than five of these, do not read them again — say them out loud. The failure mode in a breadth round is almost never "did not know"; it is "knew it and could not assemble it into ninety coherent seconds".')}`,
    labs: {
      drill: function (host) {
        const BANK = [
          ['Fundamentals', 'Bias–variance decomposition — name the three terms.', 'Bias² + variance + irreducible noise. Bias = wrong assumptions; variance = sensitivity to the sample; noise = the floor. Double descent breaks the classical U-shape.', 'bias-variance'],
          ['Fundamentals', 'Why does L1 give exact zeros and L2 not?', 'L1’s subgradient is a constant λ·sign(w); L2’s is 2λw, which vanishes as w→0. Constant pull reaches zero; proportional pull never does.', 'regularization'],
          ['Fundamentals', 'What is the curse of dimensionality in one sentence?', 'Distances concentrate and volume moves to the boundary, so local methods need exponentially more data as dimension grows.', 'pca'],
          ['Fundamentals', 'Generative vs discriminative — one example each.', 'Generative models p(x,y) (Naive Bayes, GMM) and can sample; discriminative models p(y|x) (logistic regression, most networks) and usually classifies better with enough data.', 'knn-nb'],
          ['Metrics', 'What does AUC mean probabilistically?', 'The probability that a randomly chosen positive is ranked above a randomly chosen negative.', 'metrics'],
          ['Metrics', 'When does AUC mislead?', 'Under heavy imbalance — the FPR denominator is huge, so many false positives barely move it. Use average precision or recall at a fixed alert budget.', 'metrics'],
          ['Metrics', 'Chance level for average precision?', 'The positive rate. For AUC it is 0.5; for AP it is the base rate, which is why AP must always be reported against it.', 'metrics'],
          ['Metrics', 'Cost-optimal threshold formula?', 't* ≈ c_FP / (c_FP + c_FN) for a calibrated model. Expensive misses drive the threshold down.', 'calibration'],
          ['Metrics', 'Precision, recall, and what each denominator is.', 'Precision = TP/(TP+FP), the share of alerts that are right. Recall = TP/(TP+FN), the share of real cases caught.', 'metrics'],
          ['Metrics', 'Calibration vs discrimination — can you have one without the other?', 'Yes, both ways. Predicting the base rate is calibrated with AUC 0.5; a perfect ranker can be badly calibrated and fixed by isotonic or Platt scaling without changing AUC.', 'calibration'],
          ['Models', 'Bagging vs boosting — what does each reduce?', 'Bagging reduces variance (parallel strong learners); boosting reduces bias (sequential weak learners on residuals).', 'boosting'],
          ['Models', 'Why does a random forest subsample features?', 'To lower the correlation ρ between trees. The ensemble variance floor is ρσ², so diversity — not the number of trees — is the binding constraint.', 'ensembles'],
          ['Models', 'The kernel trick in one sentence.', 'The dual depends on data only through inner products, so replacing them with a PSD kernel gives another space’s geometry without ever constructing its features.', 'svm'],
          ['Models', 'What does the SVM margin maximise, and why does that generalise?', 'The distance to the nearest points. A larger margin means a smaller effective capacity, which bounds generalization error.', 'svm'],
          ['Models', 'Why must you scale features before k-means and SVM?', 'Both use Euclidean distance, so a feature measured in thousands dominates one measured in units. Trees are scale-invariant and do not care.', 'unsupervised'],
          ['Models', 'k-means assumptions?', 'Spherical, similar-sized, similar-density clusters — it minimises within-cluster squared Euclidean distance. Use GMM for elliptical, DBSCAN for arbitrary shapes.', 'unsupervised'],
          ['Models', 'Why is Naive Bayes badly calibrated but a decent classifier?', 'Conditional independence counts correlated evidence repeatedly, producing over-confident posteriors — but the argmax often survives.', 'knn-nb'],
          ['Models', 'What does XGBoost’s second-order expansion buy?', 'A closed-form optimal leaf value −G/(H+λ) and an exact split gain, so splits are chosen by how much they reduce the loss rather than by a heuristic impurity.', 'boosting'],
          ['Deep', 'Why does ReLU beat sigmoid in deep networks?', 'Sigmoid saturates and its derivative maxes at 0.25, so gradients shrink geometrically with depth. ReLU passes gradient 1 on the positive side.', 'activations'],
          ['Deep', 'What problem does batch norm solve, and what replaced it in transformers?', 'It stabilises layer input distributions and permits higher learning rates. Layer norm (then RMSNorm) replaced it because batch statistics are unstable for variable-length sequences.', 'normalisation'],
          ['Deep', 'Why do residual connections help?', 'They give a gradient path with derivative 1 straight to earlier layers, so depth stops multiplying small numbers. It is what makes 100+ layer stacks trainable.', 'cnn'],
          ['Deep', 'Adam in one sentence, and its two bias corrections.', 'Per-parameter step sizes from EMAs of the gradient and its square; both EMAs start at zero, so each is divided by 1−β^t to remove the initial bias.', 'optimisers'],
          ['Deep', 'What does weight decay do that L2 does not, in Adam?', 'AdamW decouples decay from the adaptive denominator. Plain L2 inside Adam is scaled by 1/√v, so heavily-updated parameters get less regularization than intended.', 'optimisers'],
          ['Deep', 'Why two 3×3 convolutions instead of one 5×5?', 'Same receptive field, 18 vs 25 parameters per channel pair, and one extra nonlinearity.', 'cnn'],
          ['Deep', 'Dropout at training and at inference?', 'Randomly zero units during training (scaling the rest by 1/(1−p)); at inference use all units with no scaling. Getting the scaling wrong shifts every activation.', 'normalisation'],
          ['Deep', 'What does gradient clipping prevent?', 'A single large-gradient batch destroying the weights. Standard in LLM pretraining, usually at global norm 1.0.', 'training-dynamics'],
          ['LLM', 'Why divide attention scores by √d_k?', 'Dot products over d_k dimensions have variance d_k; without the scaling the softmax saturates and its gradient vanishes.', 'attention'],
          ['LLM', 'Self-attention complexity, and what FlashAttention changes.', 'O(n²d) time and O(n²) memory naively. FlashAttention keeps the time and makes memory O(n) by tiling and never materialising the score matrix.', 'kv-cache'],
          ['LLM', 'KV cache size formula.', '2 × layers × kv_heads × head_dim × seq_len × batch × bytes. It is what limits concurrency, not the weights.', 'kv-cache'],
          ['LLM', 'What does GQA trade?', 'Fewer KV heads shared across query heads — a large cut in KV cache for a small quality loss. MQA is the extreme case with one KV head.', 'kv-cache'],
          ['LLM', 'Chinchilla, in one line.', 'For a fixed compute budget, scale parameters and tokens roughly equally — about 20 tokens per parameter. Earlier models were badly under-trained.', 'scaling-laws'],
          ['LLM', 'Why is inference-optimal different from compute-optimal?', 'Chinchilla minimises training cost. If you serve billions of requests, a smaller model trained on far more tokens is cheaper overall.', 'scaling-laws'],
          ['LLM', 'RLHF vs DPO — what is removed?', 'The separate reward model and the PPO sampling loop. DPO inverts the closed-form optimal policy so preferences become a direct classification loss.', 'post-training'],
          ['LLM', 'What is LoRA actually training?', 'Two low-rank matrices whose product is added to a frozen weight: W + BA with rank r ≪ d. ~0.1–1% of the parameters, and mergeable at inference.', 'lora'],
          ['LLM', 'Why does temperature 0 not guarantee determinism?', 'Floating-point non-associativity in batched GPU reductions changes tie-breaking, and batch composition varies with other traffic.', 'decoding'],
          ['LLM', 'Speculative decoding — does it change the output distribution?', 'No. The accept/reject rule with residual resampling makes it provably identical to sampling from the target model.', 'speculative'],
          ['Applied', 'Three signs of data leakage.', 'A suspiciously high validation score; a feature with implausible importance; and a gap between offline and online performance.', 'features'],
          ['Applied', 'RAG vs fine-tuning — the one-line rule.', 'RAG for knowledge that changes or must be cited; fine-tuning for form, format and behaviour. They are not substitutes.', 'rag-vs-ft'],
          ['Applied', 'Why is a reranker needed after retrieval?', 'Bi-encoders embed query and document independently; a cross-encoder sees them together and models interaction. Too slow for the corpus, fine for the top 50.', 'rag'],
          ['Applied', 'What is reciprocal rank fusion for?', 'Combining rankings from BM25 and dense retrieval without needing to calibrate their incomparable scores — it fuses ranks, not scores.', 'rag'],
          ['Applied', 'Chunk size trade-off.', 'Small chunks: precise embeddings, split answers. Large: complete answers, blurred embeddings. Small-to-big retrieval resolves it.', 'chunking'],
          ['Applied', 'Prompt injection — why is it not solvable by filtering?', 'Instructions and data share one channel with no privilege boundary. Defence is architectural: least privilege, human approval for side effects, and treating all tool output as untrusted.', 'production-ai'],
          ['Applied', 'When is a vector database unnecessary?', 'Under ~100k vectors, a numpy matrix multiply beats the network round trip. Measure before adding infrastructure.', 'vector-search'],
          ['Stats', 'Confidence interval vs credible interval.', 'A 95% CI comes from a procedure that covers the true value 95% of the time across repetitions; a credible interval says there is 95% posterior probability the parameter is inside, given the prior.', 'intervals'],
          ['Stats', 'What does peeking do to your false-positive rate?', 'Inflates it from 5% to roughly 20–35% over two weeks of daily checks. Fix the horizon or use always-valid inference.', 'experimentation'],
          ['Stats', 'CUPED in one sentence.', 'Subtract θ(X − X̄) using a pre-experiment covariate; unbiased because X precedes treatment, and variance falls by 1−ρ².', 'experimentation'],
          ['Stats', 'Halving the MDE costs how much more traffic?', 'Four times — n scales as 1/δ².', 'experimentation'],
          ['Stats', 'Central limit theorem, and one caveat.', 'Sample means tend to normal with variance σ²/n. The caveat: for skewed distributions n=30 is not enough — coverage can be 91% instead of 95%.', 'concentration'],
          ['Stats', 'Simpson’s paradox — one sentence and one defence.', 'An association can reverse when a confounder is controlled. Defence: check segment mixes and reason about the causal graph before aggregating.', 'causal'],
          ['Stats', 'Why does correlation not imply causation, mechanically?', 'Three alternatives: reverse causation, a common cause (confounding), and selection into the sample (collider bias).', 'causal'],
          ['Stats', 'Bootstrap — what fraction of rows appears in one resample?', '63.2% — that is 1−e⁻¹, and the complement is the out-of-bag set.', 'sampling'],
          ['Production', 'Training/serving skew — the structural fix.', 'One transformation implementation used by both paths, or a feature store. Two codebases always diverge.', 'mlops'],
          ['Production', 'Shadow vs canary.', 'Shadow computes without acting — answers the operational questions at zero risk. Canary acts for a small share — answers the quality question with real risk.', 'mlops'],
          ['Production', 'PSI thresholds and their limitation.', '>0.1 investigate, >0.25 significant. It cannot detect concept drift, where p(y|x) moves but p(x) does not.', 'production'],
          ['Production', 'Why keep a permanent holdback?', 'To measure the model’s cumulative value, and to keep training data uncontaminated by the model’s own decisions.', 'mlops'],
          ['Production', 'Three kinds of distribution shift.', 'Covariate (p(x)), label (p(y)), concept (p(y|x)). Only the last requires labels to detect.', 'robustness'],
          ['Production', 'Your eval improved by 3 points on 100 examples. Trust it?', 'No — the 95% interval at n=100 is about ±9 points. Pair on the same items and report the interval.', 'evals'],
          ['Maths', 'Eigenvector, in one sentence.', 'A direction the matrix only stretches, not rotates: Av = λv. PCA uses the eigenvectors of the covariance matrix.', 'linear-algebra'],
          ['Maths', 'Why is a covariance matrix always PSD?', 'It is XᵀX/n up to centring, and vᵀXᵀXv = ‖Xv‖² ≥ 0 for every v.', 'linear-algebra'],
          ['Maths', 'KL divergence — is it a distance?', 'No: it is asymmetric and violates the triangle inequality. Reverse KL is mode-seeking, forward KL mass-covering.', 'information'],
          ['Maths', 'Why is cross-entropy the loss for classification?', 'It is the negative log-likelihood of a categorical model, so minimising it is maximum likelihood; its gradient w.r.t. the logits is simply p̂ − y.', 'information'],
          ['Maths', 'Gradient of ‖Xw − y‖².', '2Xᵀ(Xw − y). Setting it to zero gives the normal equations XᵀXw = Xᵀy.', 'matrix-calculus']
        ];
        const el = ML.el;
        const topics = Array.from(new Set(BANK.map(b => b[0])));
        const st = Viz.controls(host, [
          { k: 'topic', label: 'topic', type: 'select', value: 'all', options: [{ v: 'all', t: 'everything (' + BANK.length + ')' }].concat(topics.map(t => ({ v: t, t: t + ' (' + BANK.filter(b => b[0] === t).length + ')' }))) }
        ], () => { build(); });

        let pool = [], i = 0, shown = false;
        const score = {};
        topics.forEach(t => { score[t] = { got: 0, seen: 0 }; });

        const card = el('div', {
          style: 'border:1px solid var(--line);border-radius:var(--radius);background:var(--paper);padding:18px;min-height:150px'
        });
        host.appendChild(card);
        const tally = el('div', { class: 'small', style: 'margin-top:10px' });

        function build() {
          pool = ML.shuffle(BANK.filter(b => st.topic === 'all' || b[0] === st.topic).slice());
          i = 0; shown = false; paint();
        }
        function paint() {
          const q = pool[i % pool.length];
          card.innerHTML =
            '<p class="boxtitle" style="margin-bottom:10px">' + q[0] + ' · question ' + ((i % pool.length) + 1) + ' of ' + pool.length + '</p>' +
            '<p style="font-family:var(--sans);font-size:16.5px;font-weight:600;margin:0 0 14px;line-height:1.45">' + q[1] + '</p>' +
            (shown
              ? '<div style="border-left:2px solid var(--green);padding-left:13px"><p class="lab-answer" style="padding:0">model answer</p>' +
                '<p style="margin:0 0 8px">' + q[2] + '</p>' +
                '<p class="small" style="margin:0">Full treatment: <a href="#/' + q[3] + '">' + q[3] + '</a></p></div>'
              : '<p class="small" style="color:var(--faint);margin:0">Say your answer out loud, then reveal.</p>');
          ML.typeset(card);
          drawTally();
        }
        function drawTally() {
          const rows = topics.filter(t => score[t].seen > 0)
            .map(t => t + ' ' + score[t].got + '/' + score[t].seen);
          tally.innerHTML = rows.length
            ? '<b>Self-graded:</b> ' + rows.join(' · ') + ' &nbsp;—&nbsp; overall ' +
              topics.reduce((a, t) => a + score[t].got, 0) + '/' + topics.reduce((a, t) => a + score[t].seen, 0)
            : '<span style="color:var(--faint)">Grade yourself after each reveal; your weakest topic is the one to re-read.</span>';
        }
        const row = el('div', { class: 'btnrow' });
        row.appendChild(el('button', {
          class: 'btn primary', type: 'button', text: 'Reveal answer',
          onclick: () => { shown = true; paint(); }
        }));
        row.appendChild(el('button', {
          class: 'btn', type: 'button', text: '✓ I had it',
          onclick: () => { const q = pool[i % pool.length]; score[q[0]].got++; score[q[0]].seen++; i++; shown = false; paint(); }
        }));
        row.appendChild(el('button', {
          class: 'btn', type: 'button', text: '✗ Missed it',
          onclick: () => { const q = pool[i % pool.length]; score[q[0]].seen++; i++; shown = false; paint(); }
        }));
        row.appendChild(el('button', {
          class: 'btn ghost', type: 'button', text: 'Skip',
          onclick: () => { i++; shown = false; paint(); }
        }));
        host.appendChild(row);
        host.appendChild(tally);
        build();
        Viz.note(host, 'Sixty-three questions across nine topics, each linked to the section that derives it. The rule that makes this work: <b>say the answer out loud before you press reveal.</b> Recognition feels identical to recall and is not the same skill — and the breadth round tests recall, under mild social pressure, at speaking pace.');
      }
    },
    quiz: [
      {
        q: 'A good breadth-round answer is structured as…',
        options: ['a full derivation', 'answer, concrete example, trade-off — in about 60–90 seconds', 'a list of every related method', 'a question back to the interviewer'],
        answer: 1,
        why: 'Interviewers are taking notes against a rubric. Lead with the answer; the nuance lands better once they have it.'
      },
      {
        q: 'AUC 0.95 with 2% precision at every useful threshold indicates…',
        options: ['a bug in the metric', 'severe class imbalance — use average precision instead', 'overfitting', 'a calibration problem'],
        answer: 1,
        why: 'The FPR denominator is dominated by the negative class, so many false positives barely register.'
      },
      {
        q: 'Naive Bayes is a decent classifier but a poor probability estimator because…',
        options: ['it uses a Gaussian likelihood', 'conditional independence counts correlated evidence repeatedly, so posteriors are over-confident', 'it needs smoothing', 'it has too few parameters'],
        answer: 1,
        why: 'The argmax often survives the violation; the magnitudes do not. Recalibrate before using the probability for anything.'
      }
    ],
    cards: [
      { q: 'The 90-second answer template', a: 'Direct answer → one concrete example → the trade-off → (optionally) what would change your mind.' },
      { q: 'AUC, probabilistically', a: 'P(a random positive is ranked above a random negative). Chance is 0.5; for AP chance is the base rate.' },
      { q: 'Three leakage types', a: 'Temporal (future information), preprocessing (fitted on test), group (same entity in both splits).' },
      { q: 'Five reasons offline ≠ online', a: 'Underpowered test, metric mismatch, distribution mismatch, implementation skew, system effects.' }
    ]
  });

  /* ------------------------------------------------------------------ 7.3 */
  ML.section({
    id: 'interview-depth', track: 'interview', num: '7.3', level: 3,
    title: 'The depth round: ten derivations to produce cold',
    lede: 'A depth round is one question asked five times: "why is that true?" The only defence is being able to derive things rather than recall them. These ten come up more than everything else combined.',
    prereq: ['matrix-calculus'],
    related: ['matrix-calculus', 'backprop', 'attention'],
    html: `
${H.tldr([
      'Derive, do not recite. An interviewer who asks "where does the $\\sqrt{d_k}$ come from" is checking whether you can reconstruct it, not whether you remember it.',
      'Every derivation below starts from a definition and takes fewer than eight lines. If yours takes twenty, you have memorised the wrong thing.',
      'Narrate while you write. A silent whiteboard scores nothing; the reasoning is the artefact being evaluated.'
    ])}

${H.lab('deriv', 'The derivation trainer', 'Pick a derivation, try it on paper, then reveal it one line at a time. The right-hand column is the justification for each step — that column is what interviewers actually probe.')}

<h2><span class="sn">7.3.1</span> How to perform a derivation under pressure</h2>
${H.steps([
      '<b>State what you are deriving and from what.</b> "I will get the logistic loss from the Bernoulli likelihood." Ten seconds, and it structures everything after.',
      '<b>Write the definition down first.</b> Most derivations are three algebraic moves away from a definition; the failure is usually starting from a half-remembered intermediate step.',
      '<b>Say why each step is allowed</b> as you write it — "this is a scalar so it equals its transpose", "the max is constant in $k$ so it comes out of the sum". This is the part being scored.',
      '<b>Sanity-check the result.</b> Shapes, limits, signs. "At $p = y$ this is zero, which it should be." A single check demonstrates more understanding than the whole derivation.',
      '<b>Say what it connects to.</b> "…which is why softmax and cross-entropy are fused in every framework." Connection is the senior signal.'
    ])}

<h2><span class="sn">7.3.2</span> The ten, with their traps</h2>
${H.table(['Derivation', 'Start from', 'The step people miss', 'Section'], [
      ['Logistic loss from MLE', 'the Bernoulli likelihood', 'writing it as $p^y(1-p)^{1-y}$ so the log splits into the two-term form', '<a href="#/mle-map">1.5</a>'],
      ['The normal equations', '$\\|Xw-y\\|^2$ as a dot product', 'that the two cross terms are equal because a scalar equals its transpose', '<a href="#/matrix-calculus">0.7</a>'],
      ['Softmax + CE gradient = $\\hat p - y$', '$\\log\\hat p_k = z_k - \\log\\sum e^{z_j}$', 'that $\\partial(\\text{logsumexp})/\\partial z_m$ <i>is</i> the softmax', '<a href="#/matrix-calculus">0.7</a>'],
      ['Backprop for one layer', 'the chain rule as a vjp', 'accumulating with += when a node feeds several others', '<a href="#/backprop">3.2</a>'],
      ['The $\\sqrt{d_k}$ in attention', '$\\mathrm{Var}(q^\\top k) = d_k$ at unit variance', 'connecting saturation of the softmax to a vanishing gradient', '<a href="#/attention">4.3</a>'],
      ['Bias–variance decomposition', '$\\mathbb{E}[(y-\\hat f)^2]$, add and subtract $\\mathbb{E}[\\hat f]$', 'that the cross term vanishes in expectation', '<a href="#/bias-variance">2.2</a>'],
      ['Ridge as MAP', 'a Gaussian prior on $w$', 'that $-\\log$ of a Gaussian prior is $\\lambda\\|w\\|^2$ up to a constant', '<a href="#/mle-map">1.5</a>'],
      ['PCA as variance maximisation', '$\\max_{\\|v\\|=1} v^\\top \\Sigma v$', 'that the Lagrangian gives $\\Sigma v = \\lambda v$ — the eigenproblem falls out', '<a href="#/pca">2.10</a>'],
      ['The XGBoost split gain', 'a second-order Taylor expansion of the loss', 'that the optimal leaf value is $-G/(H+\\lambda)$, so gain is a difference of $G^2/(H+\\lambda)$ terms', '<a href="#/boosting">2.8</a>'],
      ['The ELBO', '$\\log p(x) = \\log\\int p(x,z)dz$', 'multiplying and dividing by $q(z)$ before applying Jensen', '<a href="#/bayesian-inference">1.14</a>']
    ])}

${H.iq('The depth questions that hide a derivation', [
      {
        q: 'Why is the logistic loss what it is? Derive it.',
        level: 'core',
        a: `<p>Start from the model: $y\\mid x \\sim \\mathrm{Bernoulli}(p)$ with $p = \\sigma(w^\\top x)$. The likelihood of one observation is $p^{y}(1-p)^{1-y}$ — a compact way of writing "$p$ if $y=1$, $1-p$ if $y=0$".</p>
<p>Take logs and negate: $-\\big[y\\log p + (1-y)\\log(1-p)\\big]$. Sum over the data and divide by $n$. That is cross-entropy, and it is exactly the negative log-likelihood — so <b>minimising cross-entropy is maximum likelihood estimation</b>, not an arbitrary choice of loss.</p>
<p>Then add the sanity check: it is zero when $p=y$, infinite when you are confidently wrong, and its gradient with respect to the logit is $p-y$.</p>`,
        follow: ['Why not squared loss on the probability?', 'What does adding L2 make this?', 'What happens if the data are separable?'],
        red: 'Presenting cross-entropy as a definition. The interviewer wants the likelihood.'
      },
      {
        q: 'Derive the bias–variance decomposition.',
        level: 'senior',
        a: `<p>Let $y = f(x)+\\varepsilon$ with $\\mathbb{E}[\\varepsilon]=0$, $\\mathrm{Var}(\\varepsilon)=\\sigma^2$, and let $\\hat f$ be our estimator, random through the training sample.</p>
<p>$\\mathbb{E}\\big[(y-\\hat f)^2\\big]$. Insert $\\pm\\mathbb{E}[\\hat f]$ inside the square and expand. Three terms appear: $\\big(f - \\mathbb{E}[\\hat f]\\big)^2$ (bias²), $\\mathbb{E}\\big[(\\hat f - \\mathbb{E}[\\hat f])^2\\big]$ (variance), and $\\sigma^2$ (noise). <b>All the cross terms vanish</b> — one because $\\mathbb{E}[\\varepsilon]=0$ and $\\varepsilon$ is independent of $\\hat f$, the other because $\\mathbb{E}[\\hat f - \\mathbb{E}[\\hat f]] = 0$ by construction.</p>
<p>Note what the expectation is over: the training sample. Bias and variance are properties of the <i>procedure</i>, not of one fitted model, which is why you cannot measure them from a single fit.</p>`,
        follow: ['What is the expectation over, exactly?', 'How does this change for 0-1 loss?', 'Where does double descent fit?']
      },
      {
        q: 'Show that ridge regression is MAP estimation with a Gaussian prior.',
        level: 'senior',
        a: `<p>Posterior $\\propto$ likelihood × prior. With $y\\mid x,w \\sim \\mathcal{N}(w^\\top x, \\sigma^2)$ and $w\\sim\\mathcal{N}(0,\\tau^2 I)$:</p>
<p>$-\\log p(w\\mid D) = \\frac{1}{2\\sigma^2}\\|Xw-y\\|^2 + \\frac{1}{2\\tau^2}\\|w\\|^2 + \\text{const}$.</p>
<p>Multiply by $2\\sigma^2$: this is $\\|Xw-y\\|^2 + \\lambda\\|w\\|^2$ with $\\lambda = \\sigma^2/\\tau^2$. So <b>the regularization strength is the ratio of noise variance to prior variance</b> — a tighter prior or noisier data means more shrinkage, which is exactly the intuition you would want.</p>
<p>Swap the Gaussian prior for a Laplace and the same argument gives lasso.</p>`,
        follow: ['What does λ → 0 and λ → ∞ correspond to?', 'Why does the Laplace prior give sparsity and the Gaussian not?', 'Is MAP the same as the posterior mean here?']
      },
      {
        q: 'Derive PCA as variance maximisation and explain why eigenvectors appear.',
        level: 'senior',
        a: `<p>Find the unit direction $v$ maximising the variance of the projection: $\\max_{\\|v\\|=1} \\mathrm{Var}(Xv) = v^\\top\\Sigma v$ with $\\Sigma$ the covariance matrix.</p>
<p>Form the Lagrangian $v^\\top\\Sigma v - \\lambda(v^\\top v - 1)$ and set the gradient to zero: $2\\Sigma v - 2\\lambda v = 0$, i.e. $\\Sigma v = \\lambda v$. <b>The eigenproblem is not assumed — it is the stationarity condition</b> (§1.12). Left-multiplying by $v^\\top$ gives $v^\\top\\Sigma v = \\lambda$, so the variance captured <i>is</i> the eigenvalue, and the largest eigenvalue wins.</p>
<p>Subsequent components repeat the argument under an orthogonality constraint, which is why the components are orthogonal.</p>`,
        follow: ['Why must you centre the data first?', 'How does this connect to the SVD of X?', 'What if you do not standardise features of different units?']
      },
      {
        q: 'Derive the XGBoost split gain.',
        level: 'senior',
        a: `<p>Expand the loss to second order around the current prediction: $\\sum_i \\big[g_i f(x_i) + \\tfrac12 h_i f(x_i)^2\\big] + \\Omega(f)$ with $g_i,h_i$ the first and second derivatives.</p>
<p>For a tree, $f$ is constant $w_j$ on leaf $j$, so the objective becomes $\\sum_j \\big[G_j w_j + \\tfrac12(H_j+\\lambda)w_j^2\\big] + \\gamma T$ with $G_j=\\sum_{i\\in j} g_i$. This is a quadratic in each $w_j$: the optimum is $w_j^\\star = -G_j/(H_j+\\lambda)$ and the objective value is $-\\tfrac12\\sum_j G_j^2/(H_j+\\lambda)$.</p>
<p>A split's gain is the improvement in that value: $\\tfrac12\\left[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right]-\\gamma$. <b>Splits are chosen by how much they reduce the actual loss</b>, not by an impurity heuristic — that is XGBoost's central idea.</p>`,
        follow: ['What role does γ play?', 'Why does λ in the denominator regularise?', 'What happens with squared loss — what are g and h?']
      }
    ])}

${H.pitfall('The trap in every one of these is starting too late. If you begin at "the loss is $-y\\log p - (1-y)\\log(1-p)$" you have skipped the question. Begin at the probability model, the definition, or the objective — the first line you write is most of what is being assessed.')}`,
    labs: {
      deriv: function (host) {
        const D = {
          logistic: {
            name: 'Logistic loss from maximum likelihood',
            goal: 'show that cross-entropy is the negative log-likelihood of a Bernoulli model',
            steps: [
              ['$y\\mid x \\sim \\mathrm{Bernoulli}(p),\\quad p = \\sigma(w^\\top x)$', 'State the probability model. This is the line people skip, and it is the question.'],
              ['$P(y\\mid x) = p^{y}(1-p)^{1-y}$', 'A compact way to write "p if y=1, 1−p if y=0". The exponents act as switches.'],
              ['$\\mathcal{L} = \\prod_i p_i^{y_i}(1-p_i)^{1-y_i}$', 'The likelihood of the whole dataset, assuming independence.'],
              ['$\\log\\mathcal{L} = \\sum_i y_i\\log p_i + (1-y_i)\\log(1-p_i)$', 'Take logs: products become sums, and the exponents come down.'],
              ['$-\\tfrac1n\\log\\mathcal{L} = -\\tfrac1n\\sum_i\\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big]$', 'Negate and average. This is cross-entropy — so minimising it <i>is</i> maximum likelihood.'],
              ['check: $\\partial\\mathcal{L}/\\partial z = p - y$', 'Sanity check. Zero when the prediction is exact; bounded in [−1,1], so it cannot explode.']
            ]
          },
          normal: {
            name: 'The normal equations',
            goal: 'minimise $\\|Xw-y\\|^2$ and see where ridge comes from',
            steps: [
              ['$\\mathcal{L} = \\|Xw-y\\|_2^2 = (Xw-y)^\\top(Xw-y)$', 'A squared L2 norm is a dot product with itself — the move that turns geometry into algebra.'],
              ['$= w^\\top X^\\top Xw - 2y^\\top Xw + y^\\top y$', 'Expand. The two cross terms are equal because each is a scalar and a scalar equals its transpose.'],
              ['$\\nabla_w = 2X^\\top Xw - 2X^\\top y$', 'Apply $\\nabla(w^\\top Aw)=2Aw$ for symmetric $A$, and $\\nabla(a^\\top w)=a$.'],
              ['$X^\\top X\\hat w = X^\\top y$', 'Set to zero. The normal equations.'],
              ['$(X^\\top X + \\lambda I)\\hat w = X^\\top y$', 'Adding $\\lambda\\|w\\|^2$ to the loss adds $\\lambda I$ here — and a positive diagonal makes the matrix invertible whatever the collinearity.']
            ]
          },
          sqrtdk: {
            name: 'The √dₖ in attention',
            goal: 'explain the scaling from a variance argument',
            steps: [
              ['$q,k \\in \\mathbb{R}^{d_k}$, entries iid with mean 0 and variance 1', 'The standing assumption at initialisation, which is where the argument applies.'],
              ['$q^\\top k = \\sum_{i=1}^{d_k} q_i k_i$', 'A sum of $d_k$ independent zero-mean products.'],
              ['$\\mathrm{Var}(q^\\top k) = d_k$', 'Variances of independent terms add, and each product has variance 1.'],
              ['$\\mathrm{sd} = \\sqrt{d_k}$', 'So at $d_k=128$ the typical score is around ±11 — enormous for a softmax input.'],
              ['softmax saturates $\\Rightarrow$ $\\partial/\\partial z \\to 0$', 'One weight goes to 1, the rest to 0, and the softmax Jacobian $\\mathrm{diag}(p)-pp^\\top$ vanishes. Training stalls.'],
              ['divide by $\\sqrt{d_k}$', 'Restores unit variance, so the softmax starts in its responsive range. The same variance-control logic as He initialisation.']
            ]
          },
          biasvar: {
            name: 'Bias–variance decomposition',
            goal: 'split expected squared error into three terms',
            steps: [
              ['$y = f(x)+\\varepsilon,\\ \\mathbb{E}[\\varepsilon]=0,\\ \\mathrm{Var}(\\varepsilon)=\\sigma^2$', 'The data-generating assumption. $\\hat f$ is random through the training sample.'],
              ['$\\mathbb{E}[(y-\\hat f)^2] = \\mathbb{E}[(f+\\varepsilon-\\hat f)^2]$', 'Substitute.'],
              ['$=\\mathbb{E}[(f-\\hat f)^2] + \\sigma^2$', 'The cross term dies because $\\varepsilon$ has mean zero and is independent of $\\hat f$.'],
              ['$f-\\hat f = (f - \\mathbb{E}\\hat f) + (\\mathbb{E}\\hat f - \\hat f)$', 'Add and subtract $\\mathbb{E}[\\hat f]$ — the whole trick is here.'],
              ['$= \\underbrace{(f-\\mathbb{E}\\hat f)^2}_{\\text{bias}^2} + \\underbrace{\\mathbb{E}[(\\hat f - \\mathbb{E}\\hat f)^2]}_{\\text{variance}} + \\sigma^2$', 'The cross term vanishes because $\\mathbb{E}[\\hat f - \\mathbb{E}\\hat f]=0$ by construction.'],
              ['note the expectation is over training sets', 'Bias and variance describe the <i>procedure</i>, which is why one fitted model cannot exhibit them.']
            ]
          },
          elbo: {
            name: 'The evidence lower bound',
            goal: 'turn an intractable integral into an optimisation problem',
            steps: [
              ['$\\log p(x) = \\log\\int p(x,z)\\,dz$', 'The quantity we cannot compute — the integral is over all latent configurations.'],
              ['$= \\log\\int q(z)\\frac{p(x,z)}{q(z)}dz$', 'Multiply and divide by any distribution $q$ with the right support. Nothing has changed yet.'],
              ['$= \\log\\mathbb{E}_q\\!\\left[\\frac{p(x,z)}{q(z)}\\right]$', 'Recognise the integral as an expectation under $q$.'],
              ['$\\ge \\mathbb{E}_q\\!\\left[\\log\\frac{p(x,z)}{q(z)}\\right]$', 'Jensen’s inequality: $\\log$ is concave, so the log of an expectation is at least the expectation of the log. <b>This is the only inequality used.</b>'],
              ['$= \\mathbb{E}_q[\\log p(x\\mid z)] - \\mathrm{KL}(q\\|p(z))$', 'Split the joint. Reconstruction term minus a regulariser — literally the VAE loss.'],
              ['$\\log p(x) - \\text{ELBO} = \\mathrm{KL}(q\\|p(z\\mid x))$', 'The gap <i>is</i> the KL to the true posterior. Since the left term is fixed, raising the ELBO lowers the KL — which is why maximising a bound performs inference.']
            ]
          },
          xgb: {
            name: 'The XGBoost split gain',
            goal: 'derive the gain formula from a second-order expansion',
            steps: [
              ['$\\mathcal{L}^{(t)} \\approx \\sum_i\\big[g_i f_t(x_i) + \\tfrac12 h_i f_t(x_i)^2\\big] + \\Omega(f_t)$', 'Second-order Taylor expansion of the loss around the current prediction; $g,h$ are the first and second derivatives.'],
              ['$f_t(x) = w_{q(x)}$, constant on each leaf', 'A tree is a piecewise-constant function. Group the sum by leaf.'],
              ['$= \\sum_j\\big[G_j w_j + \\tfrac12(H_j+\\lambda)w_j^2\\big] + \\gamma T$', '$G_j=\\sum_{i\\in I_j}g_i$, $H_j=\\sum_{i\\in I_j}h_i$. Now it is a separate quadratic per leaf.'],
              ['$w_j^\\star = -\\dfrac{G_j}{H_j+\\lambda}$', 'Minimise each quadratic. Note that $\\lambda$ shrinks the leaf value — that is exactly how it regularises.'],
              ['$\\mathcal{L}^\\star = -\\tfrac12\\sum_j\\dfrac{G_j^2}{H_j+\\lambda} + \\gamma T$', 'Substitute back. This is the score of a fixed tree structure.'],
              ['$\\text{gain} = \\tfrac12\\left[\\dfrac{G_L^2}{H_L+\\lambda}+\\dfrac{G_R^2}{H_R+\\lambda}-\\dfrac{(G_L{+}G_R)^2}{H_L{+}H_R{+}\\lambda}\\right]-\\gamma$', 'The improvement from splitting one leaf into two. <b>A split is chosen by how much it reduces the loss, not by Gini or entropy.</b>']
            ]
          }
        };
        const el = ML.el;
        const st = Viz.controls(host, [
          { k: 'which', label: 'derivation', type: 'select', value: 'logistic', options: Object.keys(D).map(k => ({ v: k, t: D[k].name })) }
        ], () => { shown = 0; paint(); });
        let shown = 0;
        const box = el('div');
        host.appendChild(box);

        function paint() {
          const d = D[st.which];
          box.innerHTML =
            '<div class="deriv"><div class="dhead">derivation<span class="goal">' + d.goal + '</span></div>' +
            d.steps.map((s, i) => i < shown
              ? '<div class="dstep"><div class="di">' + (i + 1) + '</div><div class="dm">' + s[0] + '</div><div class="dw">' + s[1] + '</div></div>'
              : '<div class="dstep" style="opacity:.32"><div class="di">' + (i + 1) + '</div><div class="dm" style="font-family:var(--mono);font-size:12px;color:var(--faint)">— hidden —</div><div class="dw"></div></div>'
            ).join('') + '</div>';
          ML.typeset(box);
          prog.textContent = shown + ' / ' + d.steps.length + ' lines revealed';
        }
        const prog = el('span', { class: 'small', style: 'font-family:var(--mono)' });
        const row = el('div', { class: 'btnrow' });
        row.appendChild(el('button', { class: 'btn primary', type: 'button', text: 'Reveal next line', onclick: () => { shown = Math.min(D[st.which].steps.length, shown + 1); paint(); } }));
        row.appendChild(el('button', { class: 'btn', type: 'button', text: 'Reveal all', onclick: () => { shown = D[st.which].steps.length; paint(); } }));
        row.appendChild(el('button', { class: 'btn ghost', type: 'button', text: 'Hide again', onclick: () => { shown = 0; paint(); } }));
        row.appendChild(prog);
        host.appendChild(row);
        paint();
        Viz.note(host, 'Use it in this order: pick a derivation, close the laptop, write it on paper, <i>then</i> reveal line by line and check your justifications against the right-hand column. The line you could not justify is the one an interviewer will ask about — that correlation is not a coincidence, since they are probing the same joints in the argument.');
      }
    },
    quiz: [
      {
        q: 'The cross-entropy loss is best introduced in an interview as…',
        options: ['a standard classification loss', 'the negative log-likelihood of a Bernoulli (or categorical) model', 'the KL divergence to the labels', 'a smooth approximation to 0-1 loss'],
        answer: 1,
        why: 'Starting from the probability model is the answer to "why is the loss what it is". The other framings are true and are consequences.'
      },
      {
        q: 'In the bias–variance derivation, the cross terms vanish because…',
        options: ['they are small', '$\\mathbb{E}[\\varepsilon]=0$ and $\\mathbb{E}[\\hat f - \\mathbb{E}\\hat f]=0$', 'of independence of features', 'of the law of large numbers'],
        answer: 1,
        why: 'Both are exactly zero in expectation, so the decomposition is exact, not approximate.'
      },
      {
        q: 'PCA’s eigenvector equation $\\Sigma v = \\lambda v$ arises as…',
        options: ['an assumption', 'the stationarity condition of maximising $v^\\top\\Sigma v$ subject to $\\|v\\|=1$', 'a numerical convenience', 'the SVD definition'],
        answer: 1,
        why: 'The Lagrangian’s gradient gives it directly, and $v^\\top\\Sigma v = \\lambda$ shows the eigenvalue is the variance captured.'
      },
      {
        q: 'The ELBO derivation uses exactly one inequality. Which?',
        options: ['Cauchy–Schwarz', 'Jensen’s inequality on the concave log', 'the triangle inequality', 'Markov’s inequality'],
        answer: 1,
        why: 'Everything else is an identity; Jensen is what turns the log of an expectation into a bound.'
      }
    ],
    cards: [
      { q: 'How to open a derivation', a: 'State what you are deriving and from what, then write the definition. Starting mid-way forfeits the question.' },
      { q: 'The √dₖ argument in one line', a: 'Var(qᵀk) = dₖ at unit variance, so scores have sd √dₖ; that saturates the softmax and kills its gradient.' },
      { q: 'Ridge as MAP', a: 'Gaussian prior on w gives λ = σ²/τ² — the regularization strength is noise variance over prior variance.' },
      { q: 'XGBoost gain', a: 'Second-order expansion → optimal leaf −G/(H+λ) → gain is the difference of G²/(H+λ) terms, minus γ.' },
      { q: 'The ELBO gap', a: 'log p(x) − ELBO = KL(q‖p(z|x)). The left side is fixed, so raising the ELBO lowers the KL.' }
    ]
  });

  /* ------------------------------------------------------------------ 7.4 */
  ML.section({
    id: 'ml-system-design', track: 'interview', num: '7.4', level: 3,
    title: 'ML system design: a framework and six worked designs',
    lede: 'The round that most distinguishes senior from mid, and the one most people prepare for last. It is not a test of architectures. It is a test of whether you can turn a vague business sentence into a system with numbers attached.',
    prereq: ['production', 'serving'],
    related: ['mlops', 'serving', 'decision-ladder'],
    html: `
${H.tldr([
      'Eight steps, always in this order: <b>requirements → metrics → data → baseline → features/model → serving → monitoring → failure modes</b>. Spend the first ten minutes on the first two.',
      'Quantify everything. QPS, storage, latency budget, model size, cost per thousand requests. A design without numbers is a diagram.',
      'The senior signal is <b>naming what you would not do, and why</b>, and saying which decision you would revisit first if a number moved.'
    ])}

<h2><span class="sn">7.4.1</span> The eight steps</h2>
${H.table(['Step', 'Minutes', 'What to produce', 'The question that unlocks it'], [
      ['1. Requirements', '5–8', 'scale, latency budget, quality bar, constraints', '"Who sees the output, and what do they do with it?"'],
      ['2. Metrics', '4–6', 'one online metric, offline proxies, guardrails', '"What would make us roll this back?"'],
      ['3. Data', '5', 'sources, labels, volume, label lag, biases', '"Where do labels come from and how late are they?"'],
      ['4. Baseline', '2', 'the non-ML answer and what it scores', '"What does the rules engine get today?"'],
      ['5. Features & model', '6–8', 'features, model family, why <i>not</i> the alternatives', '"What is the simplest thing that could work?"'],
      ['6. Serving', '6–8', 'architecture, batch vs online, caching, capacity numbers', '"What is the p99 budget and where does it go?"'],
      ['7. Monitoring', '4', 'the three layers, alert thresholds, retraining trigger', '"How do we find out it broke before a user tells us?"'],
      ['8. Failure modes', '3–4', 'cold start, abuse, degradation path', '"What happens when the model is unavailable?"']
    ])}
${H.key('The single highest-scoring behaviour: after the requirements, write the numbers on the board — QPS, latency budget, corpus size, cost ceiling — and refer back to them when you make each decision. It converts opinions into consequences, and it is what "senior" looks like in this round.')}

${H.lab('capacity', 'The capacity calculator', 'The arithmetic you should be doing on the whiteboard. Put in the product numbers and it computes QPS, peak load, storage, GPU count and monthly cost — with each step shown so you can reproduce it by hand.')}

<h2><span class="sn">7.4.2</span> The numbers to know cold</h2>
${H.table(['Quantity', 'Rule of thumb'], [
      ['QPS from DAU', 'DAU × actions/day ÷ 86,400. <b>Peak is 2–5× average</b>; design for peak.'],
      ['Latency budget split', 'user-perceived 200 ms → ~50 ms retrieval, ~100 ms model, ~50 ms everything else'],
      ['Embedding storage', 'dim × 4 bytes/vector (float32); int8 quarters it; a graph index adds ~50%'],
      ['LLM decode throughput', 'bandwidth ÷ model bytes: 70B bf16 on 2 TB/s ≈ 14 tok/s single-stream'],
      ['Training memory', '~16 bytes/param with Adam in mixed precision, plus activations (§3.11)'],
      ['GPU cost', 'roughly $2–4/hour for an A100-class card, $1.5–3 for the equivalent inference card'],
      ['Feature store read', 'single-digit ms for a point lookup; batch it or it will dominate your budget'],
      ['A cache hit', 'sub-millisecond and free. <b>Always ask what fraction of traffic repeats.</b>']
    ])}

<h2><span class="sn">7.4.3</span> Six designs, in outline</h2>
${H.tabs([
      ['Feed ranking', `
<p><b>Requirements.</b> 100M DAU, 30 sessions/day, 20 items per session; p99 under 150 ms for the ranking call; the feed must never be empty.</p>
<p><b>Metrics.</b> Online: session-level engagement plus a retention guardrail and a diversity guardrail (a purely engagement-optimised feed collapses). Offline: NDCG@10 on logged interactions with position-bias correction (§2.24).</p>
<p><b>Data.</b> Impressions and interactions — implicit feedback, which means position bias and no true negatives. Label lag minutes to hours.</p>
<p><b>Architecture.</b> The universal three-stage funnel: <b>candidate generation</b> (thousands, via two-tower embedding ANN plus heuristics — §5.10), <b>ranking</b> (hundreds, a heavy model with cross-features), <b>re-ranking</b> (tens, diversity, freshness, business rules).</p>
<p><b>Serving.</b> Precompute item embeddings hourly, user embeddings on interaction; retrieval from an ANN index; ranking on CPU or a small GPU with a feature store.</p>
<p><b>What I would not do.</b> One giant model over the whole catalogue — the latency budget forbids it. And no purely engagement-optimised objective, because the feedback loop degrades the corpus.</p>` ],
      ['Fraud detection', `
<p><b>Requirements.</b> 5,000 transactions/second at peak, decision in under 100 ms, and a false positive is a blocked legitimate customer — expensive in a different currency from a missed fraud.</p>
<p><b>Metrics.</b> Online: fraud loss in currency plus the false-decline rate. Offline: recall at a fixed alert budget, plus <b>calibration</b>, because the decision multiplies probability by transaction value (§2.12).</p>
<p><b>Data.</b> Extreme imbalance (0.1–1%), <b>label lag of 30–90 days</b> from chargebacks, and adversaries who adapt. Out-of-time validation is mandatory (§2.14).</p>
<p><b>Model.</b> Gradient boosting on aggregate features (velocity counts over multiple windows, entity graphs), plus rules for known patterns. Not a deep network: tabular, and the explanation requirement is real.</p>
<p><b>Serving.</b> Streaming feature computation, a feature store with point-in-time correctness, and a fallback rules path if the model is unavailable — <b>a payment cannot wait.</b></p>
<p><b>Failure modes.</b> Concept drift as fraudsters adapt; the label lag means you learn about today’s drift in two months, so unsupervised anomaly signals fill the gap.</p>` ],
      ['RAG assistant', `
<p><b>Requirements.</b> 50k employees, 2M documents, answers must cite sources, p95 under 4 s, no data leaves the tenancy.</p>
<p><b>Metrics.</b> Online: task completion and thumbs-up rate. Offline: retrieval recall@20 on a labelled question set, answer groundedness (does every claim appear in the retrieved context), citation accuracy (§5.11).</p>
<p><b>Data.</b> The hard part is parsing and permissions. Every chunk carries an ACL, and <b>filtering must happen inside retrieval</b>, not after (§5.10) — a post-filter that empties the result set is a broken product, and a missing filter is a data breach.</p>
<p><b>Pipeline.</b> Parse → chunk small-to-big (§5.9) → hybrid BM25 + dense retrieval → RRF → cross-encoder rerank top 50 → generate with citations → verify each citation against the context.</p>
<p><b>Serving.</b> Cache embeddings and the system-prompt prefix; stream tokens so perceived latency is the time-to-first-token, not the total.</p>
<p><b>What I would not do.</b> Fine-tune for knowledge — it goes stale and cannot cite (§5.2). And no re-embedding of the whole corpus on every document edit; index incrementally by chunk ID.</p>` ],
      ['Recommendations', `
<p><b>Requirements.</b> 10M users, 1M items, cold start for both, batch and online paths.</p>
<p><b>Metrics.</b> Online: conversion and revenue per session, with a coverage guardrail. Offline: recall@k for the retriever and NDCG for the ranker, evaluated <b>out of time</b>, never on a random split — a random split lets the model see the future.</p>
<p><b>Model.</b> Two-tower for retrieval (user tower and item tower, trained with in-batch negatives — the InfoNCE of §2.22), gradient boosting or a small network for ranking with cross-features.</p>
<p><b>Cold start.</b> Content features for new items; popularity and demographic priors for new users; an explicit exploration budget so new items get impressions at all (§6.6 — this is a bandit).</p>
<p><b>Serving.</b> Item embeddings refreshed nightly, ANN index rebuilt or updated incrementally, user embeddings computed on the fly from recent history.</p>
<p><b>The trap.</b> A feedback loop: recommending only what was clicked collapses the catalogue. Reserve traffic for exploration and monitor coverage as a first-class metric.</p>` ],
      ['ETA prediction', `
<p><b>Requirements.</b> Every request needs a number; accuracy matters asymmetrically — <b>under-promising is far better than over-promising</b>; p99 under 50 ms.</p>
<p><b>Metrics.</b> Online: on-time rate and customer complaints. Offline: <b>quantile loss, not MSE</b> — you want the 80th percentile of arrival time, not the mean, and that is a different estimator (pinball loss).</p>
<p><b>Data.</b> Historical trips, live traffic, weather, driver behaviour. Strong temporal and spatial structure; heavy tails from rare incidents.</p>
<p><b>Model.</b> Gradient boosting with quantile objective, or a small network with a multi-quantile head. Segment by city — one global model underfits local traffic regimes.</p>
<p><b>Serving.</b> Precompute road-segment travel times on a schedule; the per-request model composes them. This is what keeps a 50 ms budget achievable.</p>
<p><b>Monitoring.</b> Calibration of the quantiles, per city and per hour — a model that is well calibrated overall and badly calibrated at rush hour is the failure that generates complaints.</p>` ],
      ['Content moderation', `
<p><b>Requirements.</b> 1M posts/hour, decision in under 200 ms, multilingual, and adversaries actively probing the boundary.</p>
<p><b>Metrics.</b> Online: prevalence of violating content that reached users, plus the appeal-overturn rate (your false-positive proxy). Offline: recall per policy category at a fixed precision, reported <b>per language</b> — an aggregate hides the languages you are failing (§2.19).</p>
<p><b>Architecture.</b> A cascade: a cheap classifier on everything → a heavier multimodal model on the uncertain middle → human review on the top-uncertainty slice. The cascade is what makes the cost arithmetic work at 1M/hour.</p>
<p><b>Data.</b> Labels from human reviewers with imperfect agreement — <b>report inter-annotator κ, because it caps what any model can score</b>. Severe imbalance and continuous policy change.</p>
<p><b>Adversarial.</b> Users probe the boundary deliberately (§3.14). Expect drift by design; retrain frequently; keep a rapid-response rule layer for emerging patterns, because a model retrain is too slow for a live campaign.</p>
<p><b>What I would not do.</b> Fully automate the enforcement action for high-severity categories. Precision is never high enough, and the cost of a wrong permanent ban is asymmetric.</p>` ]
    ])}

${H.iq('Design-round questions and what a strong answer contains', [
      {
        q: 'Design a system to detect fraudulent transactions.',
        level: 'senior',
        a: `<p>Do not start with the model. Start with: what is the transaction volume and peak QPS? What is the latency budget — is this synchronous in the authorisation path or asynchronous? What is the relative cost of a false decline versus a missed fraud? Where do labels come from and how late are they?</p>
<p>Then: metrics (fraud loss in currency, false-decline rate, calibration), data (imbalance, 30–90 day chargeback lag, adversarial drift), baseline (the rules engine — what does it catch today?), model (gradient boosting on velocity and graph features), serving (streaming features, point-in-time correctness, a rules fallback path), monitoring (input drift immediately, outcome drift at the label lag), failure modes (drift, adversarial adaptation, the fallback path).</p>
<p><b>Close with the trade you would revisit first</b>: "if the false-decline cost turned out to be higher than stated, I would move the threshold and add a step-up verification path rather than retrain."</p>`,
        follow: ['How do you validate with a 60-day label lag?', 'How would you handle a fraud ring rather than individual fraudsters?', 'What if the model is unavailable at authorisation time?'],
        red: 'Saying "I would use XGBoost" in the first minute. The requirements conversation is the interview.'
      },
      {
        q: 'How would you serve this model at 10,000 QPS with a 50 ms p99?',
        level: 'senior',
        a: `<p>Work the budget backwards. 50 ms p99 total: network 5, feature fetch 10, model 20, serialisation and overhead 15. Now check each against reality — a feature store point lookup is single-digit ms only if it is one batched call, so batch the fetches.</p>
<p>Capacity: at 10k QPS and 20 ms of model time, one thread serves 50 QPS, so you need ~200 concurrent slots plus headroom — a handful of machines for gradient boosting, a different conversation for a neural model. Add dynamic batching if the model is on a GPU; measure whether batching helps or hurts p99 (it usually trades p50 for throughput).</p>
<p>Then the levers, in order of return: <b>cache</b> (what fraction of requests repeat?), <b>precompute</b> (which features can be computed offline?), <b>shrink</b> (quantize or distil — §3.13), <b>degrade</b> (a cheap fallback when the budget is blown).</p>`,
        follow: ['What if p99 is fine but p999 is terrible?', 'How does batching interact with the latency target?', 'What is your fallback when the model service is down?']
      },
      {
        q: 'Your system needs to work for a brand-new user with no history. What do you do?',
        level: 'core',
        a: `<p>Cold start has three standard answers, and a strong response names all three and says which applies.</p>
<ol>
<li><b>Content-based features</b>: what can you know without history — device, locale, referrer, the item’s own attributes?</li>
<li><b>Population priors</b>: the popularity baseline, optionally segmented. This is also your non-ML baseline, so you should already have measured it.</li>
<li><b>Deliberate exploration</b>: a new user is an explore/exploit problem (§6.6). Reserve some slots for exploration and the cold-start period shortens for everyone.</li>
</ol>
<p>Then add the operational point: define explicitly when a user stops being cold — after $n$ interactions, blending smoothly rather than switching, so the experience does not lurch.</p>`,
        follow: ['How do you blend the two regimes without a discontinuity?', 'What is the equivalent for a new item?']
      }
    ])}`,
    labs: {
      capacity: function (host) {
        const st = Viz.controls(host, [
          { k: 'dau', label: 'daily active users', min: 10000, max: 100000000, step: 10000, value: 2000000, fmt: v => v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : (v / 1000) + 'k' },
          { k: 'per', label: 'requests per user per day', min: 1, max: 200, step: 1, value: 12, fmt: v => v },
          { k: 'peak', label: 'peak multiplier', min: 1.5, max: 8, step: .5, value: 3, fmt: v => v + '×' },
          { k: 'ms', label: 'model time per request (ms)', min: 1, max: 500, step: 1, value: 25, fmt: v => v + ' ms' },
          { k: 'docs', label: 'documents in the corpus', min: 0, max: 50000000, step: 100000, value: 2000000, fmt: v => v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : (v / 1000) + 'k' },
          { k: 'dim', label: 'embedding dimension', min: 128, max: 4096, step: 128, value: 1024, fmt: v => v },
          { k: 'cache', label: 'cache hit rate', min: 0, max: .9, step: .05, value: .3, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'gpu', label: 'cost per compute hour ($)', min: .5, max: 12, step: .5, value: 3, fmt: v => '$' + v.toFixed(1) }
        ], () => draw());
        const box = ML.el('div');
        host.appendChild(box);

        function draw() {
          const daily = st.dau * st.per;
          const qps = daily / 86400;
          const peakQps = qps * st.peak;
          const served = peakQps * (1 - st.cache);
          const concurrency = served * (st.ms / 1000);
          const machines = Math.ceil(concurrency / 8);      // 8 concurrent slots per machine
          const chunks = st.docs * 4;                        // ~4 chunks per document
          const bytesF32 = chunks * st.dim * 4;
          const bytesInt8 = chunks * st.dim;
          const monthly = machines * st.gpu * 24 * 30;
          const perK = monthly / (daily * 30 / 1000);

          box.innerHTML = H.table(
            ['Quantity', 'Value', 'How it was computed'],
            [
              ['Requests per day', daily.toLocaleString(), st.dau.toLocaleString() + ' DAU × ' + st.per],
              ['Average QPS', qps.toFixed(0), 'daily ÷ 86,400'],
              ['<b>Peak QPS</b>', '<b>' + peakQps.toFixed(0) + '</b>', 'average × ' + st.peak + ' — <b>design for this</b>'],
              ['QPS after cache', served.toFixed(0), 'peak × (1 − ' + (st.cache * 100).toFixed(0) + '% hit rate)'],
              ['Required concurrency', concurrency.toFixed(1), 'QPS × ' + st.ms + ' ms — Little’s law'],
              ['<b>Machines (8 slots each)</b>', '<b>' + machines + '</b>', 'concurrency ÷ 8, rounded up; add 30–50% headroom for real deployments'],
              ['Chunks to index', chunks.toLocaleString(), st.docs.toLocaleString() + ' docs × ~4 chunks'],
              ['Vector storage (fp32)', (bytesF32 / 1e9).toFixed(2) + ' GB', 'chunks × ' + st.dim + ' × 4 bytes'],
              ['Vector storage (int8)', (bytesInt8 / 1e9).toFixed(2) + ' GB', 'a quarter of the above; add ~50% for an HNSW graph'],
              ['Monthly compute', '$' + monthly.toLocaleString(undefined, { maximumFractionDigits: 0 }), machines + ' × $' + st.gpu + ' × 720 h'],
              ['<b>Cost per 1k requests</b>', '<b>$' + perK.toFixed(4) + '</b>', 'monthly ÷ monthly requests — <b>the number a PM will ask for</b>']
            ], 'num');
        }
        draw();
        Viz.note(host, 'Two habits this is meant to build. First, <b>always design for peak, not average</b> — a 3× peak multiplier is the difference between a system that works and one that falls over every evening. Second, <b>the cache is the cheapest lever you have</b>: move the hit rate from 0% to 50% and the machine count halves, which no model optimisation will match for effort. Say both of these out loud in the round.');
      }
    },
    quiz: [
      {
        q: 'In an ML system design round, the first ten minutes should be spent on…',
        options: ['the model architecture', 'requirements and metrics', 'the data pipeline', 'the serving stack'],
        answer: 1,
        why: 'Everything downstream depends on scale, latency and what "good" means. Naming a model early forfeits most of the rubric.'
      },
      {
        q: 'With 2M DAU at 12 requests/day and a 3× peak multiplier, peak QPS is roughly…',
        options: ['280', '830', '2,500', '8,300'],
        answer: 1,
        why: '24M/86,400 ≈ 278 average, × 3 ≈ 833. Do this arithmetic on the board.'
      },
      {
        q: 'For an ETA model, the right offline loss is…',
        options: ['MSE', 'quantile (pinball) loss', 'cross-entropy', 'MAE'],
        answer: 1,
        why: 'You want a high quantile of arrival time, not the mean — under-promising and over-delivering is asymmetrically better.'
      },
      {
        q: 'For a RAG system with per-document permissions, filtering must happen…',
        options: ['after retrieval', 'inside retrieval, as part of the index traversal', 'in the LLM prompt', 'at the UI layer'],
        answer: 1,
        why: 'Post-filtering can empty the result set, and a missing filter is a data breach. Filtered traversal is the correct mechanism (§5.10).'
      }
    ],
    cards: [
      { q: 'The eight steps', a: 'Requirements → metrics → data → baseline → features/model → serving → monitoring → failure modes.' },
      { q: 'QPS arithmetic', a: 'DAU × actions ÷ 86,400 for average; multiply by 2–5 for peak; design for peak.' },
      { q: 'The three-stage funnel', a: 'Candidate generation (thousands) → ranking (hundreds) → re-ranking with business rules (tens).' },
      { q: 'Cold start, three answers', a: 'Content features, population priors, and deliberate exploration — then define when a user stops being cold.' },
      { q: 'The senior signal', a: 'Naming what you would not do and why, and stating which decision you would revisit first if a number moved.' }
    ]
  });
})();
