/* ============================================================
   REFERENCE — numbers, formulas, glossary, sources, drill room
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ R.1 */
  ML.section({
    id: 'numbers', track: 'reference', num: 'R.1',
    title: 'Numbers to have cold',
    lede: 'Everything on this site that is a number, in one place. If you memorise one page, memorise this one — these are the quantities that turn a plausible answer into a checkable one.',
    html: `
${H.lab('numsearch', 'Search the numbers', 'Type to filter. Every entry links back to the section that derives it.')}

<div class="grid2">
<div>
${H.box('memory arithmetic', `
<p>Bytes/param: <b>FP32 4</b> · FP16/BF16 <b>2</b> · FP8/INT8 <b>1</b> · INT4 <b>0.5</b>.</p>
<p>Adam training ≈ <b>16 bytes/param</b> (2 weight + 2 grad in FP16, 12 FP32 optimizer state).</p>
<p>Weight VRAM ≈ params × bytes/param, +~20% overhead.</p>
<p>KV per token per layer = $2\\cdot n_{kv}\\cdot d_{head}\\cdot$ bytes; total × L × seq × batch. <b>70B GQA, 4k, batch 8, BF16 → 10 GB.</b></p>`, 'worked')}

${H.box('retrieval', `
<p>RRF <b>k = 60</b> (Cormack et al., SIGIR 2009).</p>
<p>Chunks <b>200–500 tokens</b>, <b>10–20%</b> overlap.</p>
<p>Rerank the top <b>~100</b> candidates, keep <b>5–10</b>.</p>
<p>Binary quantization = <b>32×</b> smaller (128 B/vector); PQ 64 B = <b>64×</b>. Tool retrieval past <b>~30</b> tools.</p>
<p>1M × 1024-d float32 = <b>4.1 GB</b>; HNSW graph at M=32 ≈ <b>170 MB</b>.</p>`, 'worked')}

${H.box('fine-tuning', `
<p>LoRA <b>r = 16</b> default (32–64 for complex), <b>α = 2r</b>, target all linear layers.</p>
<p>QLoRA = <b>4-bit NF4</b> frozen base + <b>BF16</b> adapter.</p>
<p>8B QLoRA fits in <b>~10 GB</b> vs <b>~128 GB</b> for full BF16 fine-tuning; ≈42M trainable (0.52%).</p>`, 'worked')}
</div>
<div>
${H.box('scaling', `
<p>Training compute <b>C ≈ 6ND</b>.</p>
<p>Chinchilla compute-optimal ≈ <b>20 tokens/param</b> (70B on 1.4T). ⚑ exact coefficients contested.</p>
<p>Inference-optimal overtrains far past it: Llama-3-8B ≈ <b>1,875 tok/param</b>.</p>
<p>MoE active ≪ total: DeepSeek V3 <b>37B / 671B</b>.</p>
<p>English ≈ <b>1.3 tokens/word</b>, ~4 chars/token; 750 words ≈ 1,000 tokens.</p>
<p>Parameters per layer ≈ <b>12d²</b>; a forward pass ≈ 2N FLOPs/token.</p>`, 'worked')}

${H.box('credit &amp; monitoring (⚑ conventions)', `
<p>PSI: <b>&lt;0.1</b> stable · <b>0.1–0.25</b> moderate · <b>&gt;0.25</b> shift.</p>
<p>IV: <b>&lt;0.02</b> useless · 0.02–0.1 weak · 0.1–0.3 medium · <b>&gt;0.3</b> strong · &gt;0.5 suspect leakage.</p>
<p><b>Gini = 2·AUC − 1.</b> F1 = harmonic mean of precision and recall.</p>
<p>Cost threshold $p^* = C_{FP}/(C_{FP}+C_{FN})$ — £60 vs £900 gives <b>6.25%</b>.</p>
<p>Fairness screen: impact ratio &lt; <b>0.80</b> (four-fifths). ⚑ convention, not law.</p>
<p>Rows for ±2 points at 95%: <b>≈4,600</b>; ±1 point: <b>≈18,000</b>.</p>
<p>A/B sample size $n \\approx 16p(1-p)/\\delta^2$ per arm; 4% baseline, +10% relative → <b>38,400/arm ≈ 16 days</b> at 5k/day.</p>`, 'worked')}

${H.box('inference &amp; safety', `
<p>Decode is bandwidth-bound: 70B BF16 ≈ <b>42 ms/token</b> at 3.35 TB/s, ~<b>300×</b> more time moving weights than using them.</p>
<p>MFU: training <b>35–55%</b>; single-stream decode ≈ <b>0.05%</b>.</p>
<p>Speculative decoding $(1-\\alpha^{k+1})/(1-\\alpha)$; α=0.7, k=4 → <b>2.8 tokens/pass</b>.</p>
<p>99% per-request refusal → <b>63%</b> attacker success over 100 tries.</p>
<p>Cohen’s κ: 82% agreement with 70/30 marginals → <b>0.57</b> (moderate).</p>`, 'worked')}
</div>
</div>

<div class="grid2">
<div>
${H.box('optimisation &amp; numerics', `
<p>Convergence: GD <b>$O(\\kappa)$</b> · momentum <b>$O(\\sqrt\\kappa)$</b> · Newton <b>$O(\\log\\log)$</b> at $O(d^3)$/step. At κ=10⁴ that is 10,000 vs 100 steps.</p>
<p>Monte Carlo error <b>$\\sigma/\\sqrt n$</b> — 10× accuracy costs <b>100×</b> the samples.</p>
<p>Bootstrap inclusion <b>63.2%</b> = $1-e^{-1}$; the rest is out-of-bag.</p>
<p>fp16 max <b>65,504</b>, smallest normal <b>6.1e-5</b>. bf16 keeps fp32’s range with <b>7</b> mantissa bits.</p>
<p>$e^x$ overflows fp64 past <b>x ≈ 709</b> — the reason for log-sum-exp.</p>`, 'worked')}

${H.box('experiments &amp; evaluation', `
<p>Halving the MDE costs <b>4×</b> the traffic ($n\\propto1/\\delta^2$).</p>
<p>Daily peeking for two weeks: false-positive rate <b>20–35%</b>, not 5%.</p>
<p>CUPED cuts variance by <b>$1-\\rho^2$</b>; ρ=0.7 → 49%, worth doubling traffic.</p>
<p>Eval set of <b>100</b> examples near a 70% pass rate: 95% CI ≈ <b>±9 points</b>.</p>
<p>Cohen’s κ bar for an LLM judge: <b>&gt;0.6</b> workable, <b>&gt;0.8</b> good — against human–human agreement as the ceiling.</p>`, 'worked')}
</div>
<div>
${H.box('training &amp; compression', `
<p>Adam training memory ≈ <b>16 bytes/param</b>: 7B needs <b>~112 GB</b> to train, <b>14 GB</b> to serve.</p>
<p>Checkpointing at <b>$k=\\sqrt L$</b>: 48 layers → 10.3 GB becomes 3.0 GB for ~30% more compute.</p>
<p>4-bit weights → <b>4×</b> decode throughput, because decode is bandwidth-bound.</p>
<p>Unstructured pruning at 90%: <b>~0×</b> speedup on dense kernels. Structured or 2:4 only.</p>
<p>Adversarial robustness, CIFAR-10 at ε=8/255: <b>95% clean / 0% robust</b> → <b>85% / ~55%</b> after adversarial training.</p>`, 'worked')}

${H.box('retrieval, ranking &amp; privacy', `
<p>10M × 1024-d: flat fp32 <b>41 GB</b> · int8+HNSW <b>~15 GB</b> · IVF-PQ 64 B <b>0.64 GB</b>.</p>
<p>Vector DB unnecessary below <b>~100k</b> vectors — a matrix multiply beats the round trip.</p>
<p>Re-embedding 120M tokens ≈ <b>$2.80</b>. Re-parsing 40,000 PDFs: weeks.</p>
<p>Image tokens: 224 px at 16 px patches = <b>196</b>; high-resolution tiling <b>2k–6k per image</b>.</p>
<p>DP: ε <b>1–3</b> is the research target, <b>&gt;10</b> means $e^{10}\\approx$ <b>22,000×</b> — near vacuous.</p>`, 'worked')}
</div>
</div>`,
    labs: {
      numsearch: function (host) {
        const NUMS = [
          ['C ≈ 6ND', 'training compute for a decoder-only transformer', 'scaling-laws'],
          ['20 tokens per parameter', 'Chinchilla compute-optimal ratio ⚑', 'scaling-laws'],
          ['1,875 tokens per parameter', 'Llama-3-8B — inference-optimal overtraining', 'scaling-laws'],
          ['≈12d² per layer', '4d² attention + 8d² FFN', 'block'],
          ['8.03B', 'Llama-3-8B from four config numbers', 'block'],
          ['81%', 'share of transformer-block parameters in the FFN (Llama-3-8B)', 'block'],
          ['10 GB', 'KV cache: 70B, 4k context, batch 8, BF16', 'kv-cache'],
          ['42 ms', 'memory time per decoded token, 70B BF16 at 3.35 TB/s', 'kv-cache'],
          ['300×', 'ratio of weight-movement time to arithmetic time at batch 1', 'kv-cache'],
          ['0.05%', 'MFU of single-stream decode (35–55% for training)', 'kv-cache'],
          ['2.8 tokens', 'speculative decoding at α=0.7, k=4', 'serving'],
          ['0.7 s', 'TTFT for a 2,000-token prompt on 70B at 40% utilisation', 'serving'],
          ['r = 16, α = 2r', 'LoRA defaults, all linear layers', 'lora'],
          ['0.52% / 42M', 'trainable share for an 8B QLoRA at r=16', 'lora'],
          ['9–11 GB vs 128 GB', 'QLoRA against full BF16 fine-tuning for 8B', 'lora'],
          ['671B / 37B', 'DeepSeek V3 total vs active parameters', 'moe'],
          ['1.3 tokens/word', 'English tokenization; ~4 characters per token', 'tokenization'],
          ['O(κ) vs O(√κ)', 'gradient descent vs momentum iterations to convergence', 'optimization'],
          ['σ/√n', 'Monte Carlo error — independent of dimension', 'sampling'],
          ['63.2%', 'share of rows appearing in one bootstrap resample', 'sampling'],
          ['65,504', 'largest representable fp16 value; smallest normal 6.1e-5', 'numerics'],
          ['709', 'where exp overflows in fp64 — the reason for log-sum-exp', 'numerics'],
          ['112 GB', 'memory to train a 7B model with Adam; 14 GB to serve it', 'autodiff'],
          ['k = √L', 'optimal gradient-checkpoint interval; O(L) → O(√L) memory', 'autodiff'],
          ['4×', 'decode speedup from 4-bit weights, because decode is bandwidth-bound', 'compression'],
          ['ρσ²', 'the variance floor of an ensemble — diversity, not count, is the constraint', 'ensembles'],
          ['O(n³)', 'exact Gaussian process inference, from the Cholesky factorisation', 'gp-bayesopt'],
          ['log B', 'the mutual-information ceiling of InfoNCE at batch size B', 'self-supervised'],
          ['4×', 'extra traffic needed to halve the minimum detectable effect', 'experimentation'],
          ['20–35%', 'true false-positive rate after two weeks of daily peeking', 'experimentation'],
          ['1 − ρ²', 'variance reduction from CUPED', 'experimentation'],
          ['±9 points', '95% CI on a 100-example eval near a 70% pass rate', 'evals'],
          ['196', 'tokens for a 224×224 image at 16×16 patches', 'multimodal'],
          ['3.36', 'expected tokens per verification pass at α=0.8, k=4', 'speculative'],
          ['0.5 / 0.50:0.95', 'PASCAL VOC vs COCO IoU thresholds for mAP', 'vision-tasks'],
          ['22,000×', 'the probability ratio permitted at ε = 10 — a near-vacuous privacy bound', 'privacy'],
          ['0.64 GB', '10M 1024-d vectors under IVF-PQ at 64 bytes each (41 GB as fp32)', 'vector-search'],
          ['k = 60', 'RRF constant (Cormack et al. 2009)', 'rag'],
          ['200–500 tokens', 'chunk size, 10–20% overlap', 'rag'],
          ['4.1 GB', '1M × 1024-d float32 vectors', 'rag'],
          ['170 MB', 'HNSW graph links at M=32 for 1M vectors', 'rag'],
          ['32× / 64×', 'binary quantization / PQ 64-byte compression', 'rag'],
          ['~30 tools', 'where agent tool-selection accuracy degrades', 'agents'],
          ['$0.29 vs $0.11', 'cost per successful task: cheap vs accurate model', 'production-ai'],
          ['63%', 'attacker success over 100 tries at 99% refusal', 'safety'],
          ['κ = 0.57', '82% judge agreement with 70/30 marginals', 'llm-eval'],
          ['4,600 rows', '±2 points at 95% confidence (0/1 loss)', 'concentration'],
          ['18,000 rows', '±1 point at 95% confidence', 'concentration'],
          ['38,400 per arm', 'A/B at 4% baseline, +10% relative lift → 16 days', 'intervals'],
          ['(1−ρ²)', 'CUPED variance reduction; ρ=0.6 → 36%', 'intervals'],
          ['64%', 'chance of a spurious win across 20 metrics at α=0.05', 'intervals'],
          ['1.94%', 'Bayes worked example: prevalence 0.001, sens 0.99, spec 0.95', 'bayes'],
          ['0.667', 'XGBoost worked split gain; leaf weights ∓0.667', 'boosting'],
          ['6.25%', 'cost-optimal threshold for £900 vs £60', 'metrics'],
          ['98.6% / 40% / 80%', 'accuracy / precision / recall on the worked confusion matrix', 'metrics'],
          ['Gini = 2·AUC − 1', 'credit-industry scaling of AUC', 'metrics'],
          ['PSI 0.1 / 0.25', 'stable / moderate / significant shift ⚑ convention', 'production'],
          ['IV 0.02 / 0.1 / 0.3', 'useless / weak / medium / strong ⚑ convention', 'features'],
          ['IV = 0.704', 'worked WOE example across three bins', 'features'],
          ['0.80', 'four-fifths adverse-impact screen ⚑ US enforcement convention', 'fairness'],
          ['0.738', 'worked impact ratio: 31% vs 42% approval', 'fairness'],
          ['≈37%', 'rows omitted by each bootstrap sample (out-of-bag)', 'trees'],
          ['16 bytes/param', 'Adam training state (12 bytes FP32 optimizer state)', 'distributed'],
          ['901st smallest', 'conformal quantile for 1,000 calibration points at α=0.10', 'calibration'],
          ['£2,340', 'worked conformal interval half-width', 'calibration']
        ];
        const input = ML.el('input', { type: 'search', placeholder: 'filter — try “cache”, “tokens”, “threshold”, “Chinchilla”…', style: 'margin-bottom:12px' });
        host.appendChild(input);
        const list = ML.el('div');
        host.appendChild(list);
        function render() {
          const q = input.value.toLowerCase().trim();
          const rows = NUMS.filter(n => !q || (n[0] + ' ' + n[1]).toLowerCase().indexOf(q) >= 0);
          list.innerHTML = '<div class="tablewrap"><table class="num"><thead><tr><th>value</th><th>what it is</th><th>section</th></tr></thead><tbody>' +
            rows.map(n => '<tr><td><b>' + n[0] + '</b></td><td style="font-weight:400">' + n[1] + '</td><td><a href="#/' + n[2] + '">' + n[2] + '</a></td></tr>').join('') +
            '</tbody></table></div><p class="small">' + rows.length + ' of ' + NUMS.length + ' numbers</p>';
        }
        input.addEventListener('input', render);
        render();
      }
    },
    cards: [
      { q: 'Bytes per parameter by precision', a: 'FP32 4 · FP16/BF16 2 · FP8/INT8 1 · INT4 0.5; Adam training ≈16 bytes/param.' },
      { q: 'KV cache, 70B / 4k / batch 8 / BF16', a: '10 GB — 2·80·8·128·4096·8·2 bytes.' },
      { q: 'Compute and Chinchilla', a: 'C ≈ 6ND; ≈20 tokens per parameter ⚑; inference-optimal goes far past it.' }
    ]
  });

  /* ------------------------------------------------------------------ R.2 */
  ML.section({
    id: 'formulas', track: 'reference', num: 'R.2',
    title: 'The formula sheet',
    lede: 'Every formula this site derives, with the section that derives it. Cover the right column and reproduce them.',
    html: `
<h2>Foundations</h2>
${H.table(['Name', 'Formula', '§'], [
      ['Bayes (odds form)', 'posterior odds = LR × prior odds', '<a href="#/bayes">1.1</a>'],
      ['Variance of a sum', '$\\mathrm{Var}(X+Y)=\\mathrm{Var}X+\\mathrm{Var}Y+2\\mathrm{Cov}(X,Y)$', '<a href="#/expectation">1.3</a>'],
      ['Bagging variance', '$\\rho\\sigma^2+\\frac{1-\\rho}{B}\\sigma^2$', '<a href="#/expectation">1.3</a>'],
      ['Hoeffding', '$P(|\\bar X_n-\\mu|\\ge t)\\le 2e^{-2nt^2/(b-a)^2}$', '<a href="#/concentration">1.4</a>'],
      ['MAP → penalty', 'Gaussian prior → $\\lambda\\|\\theta\\|_2^2$, $\\lambda=1/(2\\tau^2)$; Laplace → $\\lambda\\|\\theta\\|_1$', '<a href="#/mle-map">1.5</a>'],
      ['A/B sample size', '$n\\approx 16p(1-p)/\\delta^2$ per arm', '<a href="#/intervals">1.6</a>'],
      ['CUPED', '$Y_{adj}=Y-\\theta(X-\\mathbb{E}X)$, variance × $(1-\\rho^2)$', '<a href="#/intervals">1.6</a>'],
      ['SVD', '$A=U\\Sigma V^\\mathsf{T}$; truncation is the best low-rank fit', '<a href="#/linear-algebra">1.8</a>'],
      ['Gradient descent / Newton', '$\\theta\\leftarrow\\theta-\\eta\\nabla f$ · $\\theta\\leftarrow\\theta-H^{-1}\\nabla f$', '<a href="#/calculus-ml">1.9</a>'],
      ['Cross-entropy identity', '$H(p,q)=H(p)+D_{KL}(p\\|q)$', '<a href="#/information">1.10</a>'],
      ['Log-sum-exp', '$\\log\\sum e^{z_i}=m+\\log\\sum e^{z_i-m}$', '<a href="#/information">1.10</a>']
    ])}

<h2>Classical ML</h2>
${H.table(['Name', 'Formula', '§'], [
      ['Bias–variance', '$\\mathbb{E}[(y-\\hat f)^2]=\\text{bias}^2+\\text{variance}+\\sigma^2$', '<a href="#/bias-variance">2.2</a>'],
      ['Soft threshold (lasso)', '$\\mathcal{S}_\\lambda(\\rho)=\\mathrm{sign}(\\rho)\\max(0,|\\rho|-\\lambda)$', '<a href="#/regularization">2.3</a>'],
      ['Normal equations', '$w=(X^\\mathsf{T}X+\\lambda I)^{-1}X^\\mathsf{T}y$', '<a href="#/linear-logistic">2.4</a>'],
      ['Logistic gradient', '$(\\hat p-y)x$', '<a href="#/linear-logistic">2.4</a>'],
      ['Scorecard', 'factor = PDO/ln2; offset = anchor − factor·ln(anchor odds)', '<a href="#/linear-logistic">2.4</a>'],
      ['SVM soft margin', '$\\min\\frac12\\|w\\|^2+C\\sum\\xi_i$', '<a href="#/svm">2.6</a>'],
      ['XGBoost leaf weight', '$w^*=-G/(H+\\lambda)$', '<a href="#/boosting">2.8</a>'],
      ['XGBoost gain', '$\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}]-\\gamma$', '<a href="#/boosting">2.8</a>'],
      ['ELBO', '$\\log p(x)\\ge\\mathbb{E}_q[\\log p(x,z)]-\\mathbb{E}_q[\\log q(z)]$', '<a href="#/unsupervised">2.9</a>'],
      ['PCA', '$\\Sigma w=\\lambda w$ — top eigenvectors of the covariance', '<a href="#/pca">2.10</a>'],
      ['WOE / IV', '$\\ln(\\%good/\\%bad)$; $\\mathrm{IV}=\\sum(\\%good-\\%bad)\\mathrm{WOE}$', '<a href="#/features">2.11</a>'],
      ['Conformal quantile', '$\\lceil (n+1)(1-\\alpha)\\rceil$-th smallest nonconformity score', '<a href="#/calibration">2.12</a>'],
      ['Cost threshold', '$p^*=C_{FP}/(C_{FP}+C_{FN})$', '<a href="#/metrics">2.13</a>'],
      ['PSI', '$\\sum_i(a_i-e_i)\\ln(a_i/e_i)$', '<a href="#/production">2.18</a>'],
      ['SHAP efficiency', '$\\sum_i\\phi_i=f(x)-\\mathbb{E}[f]$', '<a href="#/interpretability">2.17</a>'],
      ['Cox hazard', '$h(t|x)=h_0(t)e^{\\beta^\\mathsf{T}x}$', '<a href="#/production">2.18</a>']
    ])}

<h2>Deep learning and transformers</h2>
${H.table(['Name', 'Formula', '§'], [
      ['Backprop', '$\\delta^{(l)}=(W^{(l+1)\\mathsf{T}}\\delta^{(l+1)})\\odot\\phi\'(z^{(l)})$; $\\partial L/\\partial W^{(l)}=\\delta^{(l)}a^{(l-1)\\mathsf{T}}$', '<a href="#/backprop">3.2</a>'],
      ['He init', '$\\mathrm{Var}(w)=2/n_{in}$', '<a href="#/initialisation">3.4</a>'],
      ['Adam', '$\\theta\\leftarrow\\theta-\\eta\\hat m/(\\sqrt{\\hat s}+\\epsilon)$; AdamW adds $\\lambda\\theta$ outside', '<a href="#/optimisers">3.5</a>'],
      ['RMSNorm', '$x/\\sqrt{\\frac1d\\sum x_i^2+\\epsilon}\\odot g$', '<a href="#/architectures">4.6</a>'],
      ['Attention', '$\\mathrm{softmax}(QK^\\mathsf{T}/\\sqrt{d_k})V$', '<a href="#/attention">4.3</a>'],
      ['RoPE', '$(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$', '<a href="#/rope">4.4</a>'],
      ['KV cache', 'bytes $=2\\cdot L\\cdot n_{kv}\\cdot d_{head}\\cdot$ seq · batch · bytes', '<a href="#/kv-cache">4.7</a>'],
      ['Compute', '$C\\approx6ND$', '<a href="#/scaling-laws">4.10</a>'],
      ['DPO', '$-\\log\\sigma(\\beta\\log\\frac{\\pi(y_w)}{\\pi_{ref}(y_w)}-\\beta\\log\\frac{\\pi(y_l)}{\\pi_{ref}(y_l)})$', '<a href="#/post-training">4.12</a>'],
      ['PPO', '$\\mathbb{E}[\\min(r_tA_t,\\mathrm{clip}(r_t,1\\pm\\epsilon)A_t)]-\\beta D_{KL}$', '<a href="#/post-training">4.12</a>'],
      ['GRPO advantage', '$A_i=(r_i-\\mathrm{mean}(r))/\\mathrm{std}(r)$', '<a href="#/post-training">4.12</a>'],
      ['LoRA', '$W_0x+\\frac{\\alpha}{r}BAx$', '<a href="#/lora">4.13</a>'],
      ['Speculative decoding', '$(1-\\alpha^{k+1})/(1-\\alpha)$ accepted tokens per pass', '<a href="#/serving">4.14</a>'],
      ['Cohen’s κ', '$(p_o-p_e)/(1-p_e)$', '<a href="#/llm-eval">4.16</a>']
    ])}

<h2>Applied and frontier</h2>
${H.table(['Name', 'Formula', '§'], [
      ['RRF', '$\\sum_r 1/(k+\\mathrm{rank}_r(d))$, k = 60', '<a href="#/rag">5.1</a>'],
      ['Refusal under retries', '$1-(1-\\epsilon)^n$', '<a href="#/safety">4.17</a>'],
      ['Bellman optimality', '$V^*(s)=\\max_a[R+\\gamma\\sum P(s\'|s,a)V^*(s\')]$', '<a href="#/rl">6.1</a>'],
      ['Q-learning', '$Q\\mathrel{+}=\\alpha[r+\\gamma\\max_{a\'}Q(s\',a\')-Q(s,a)]$', '<a href="#/rl">6.1</a>'],
      ['Diffusion forward', '$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$', '<a href="#/diffusion">6.2</a>'],
      ['Diffusion loss', '$\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$', '<a href="#/diffusion">6.2</a>'],
      ['VAE ELBO', '$\\mathbb{E}_q[\\log p(x|z)]-D_{KL}(q\\|p)$', '<a href="#/vae-gan">6.3</a>'],
      ['Message passing', '$h_v^{(l+1)}=\\phi(h_v^{(l)},\\bigoplus_{u\\in\\mathcal{N}(v)}\\psi(\\cdot))$', '<a href="#/gnn">6.4</a>']
    ])}

<h2>Optimisation, matrix calculus, numerics</h2>
${H.table(['Name', 'Formula', '§'], [
      ['Gradient of least squares', '$\\nabla_w\\|Xw-y\\|^2 = 2X^\\mathsf{T}(Xw-y)$', '<a href="#/matrix-calculus">0.7</a>'],
      ['Quadratic form', '$\\nabla_w\\,w^\\mathsf{T}Aw = (A+A^\\mathsf{T})w$, and $2Aw$ if $A$ symmetric', '<a href="#/matrix-calculus">0.7</a>'],
      ['Softmax + CE gradient', '$\\partial\\mathcal{L}/\\partial z = \\hat p - y$', '<a href="#/matrix-calculus">0.7</a>'],
      ['Convergence rates', 'GD $O(\\kappa\\log\\frac1\\epsilon)$ · momentum $O(\\sqrt\\kappa\\log\\frac1\\epsilon)$ · Newton $O(\\log\\log\\frac1\\epsilon)$', '<a href="#/optimization">1.12</a>'],
      ['Optimal GD rate', '$(\\kappa-1)/(\\kappa+1)$ per step at $\\eta=2/(\\lambda_{\\min}+\\lambda_{\\max})$', '<a href="#/optimization">1.12</a>'],
      ['KKT complementary slackness', '$\\lambda_i\\,g_i(x^\\star)=0$', '<a href="#/optimization">1.12</a>'],
      ['Monte Carlo error', '$\\sigma/\\sqrt n$, independent of dimension', '<a href="#/sampling">1.13</a>'],
      ['Effective sample size', '$(\\sum w_i)^2/\\sum w_i^2$', '<a href="#/sampling">1.13</a>'],
      ['Bootstrap inclusion', '$1-(1-1/n)^n\\to 1-e^{-1}=0.632$', '<a href="#/sampling">1.13</a>'],
      ['Beta–Binomial update', 'Beta$(\\alpha+s,\\ \\beta+n-s)$; prior worth $\\alpha+\\beta$ observations', '<a href="#/bayesian-inference">1.14</a>'],
      ['Metropolis acceptance', '$\\min\\!\\left(1,\\frac{p(D|\\theta^*)p(\\theta^*)}{p(D|\\theta_t)p(\\theta_t)}\\right)$ — $p(D)$ cancels', '<a href="#/bayesian-inference">1.14</a>'],
      ['ELBO identity', '$\\log p(D)=\\text{ELBO}+D_{KL}(q\\|p(\\theta|D))$', '<a href="#/bayesian-inference">1.14</a>'],
      ['Gaussian mechanism σ', '$\\sigma=\\Delta_2\\sqrt{2\\ln(1.25/\\delta)}/\\varepsilon$', '<a href="#/privacy">6.8</a>'],
      ['Welford update', '$d=x-\\mu$; $\\mu\\mathrel{+}=d/n$; $M_2\\mathrel{+}=d(x-\\mu_{\\text{new}})$', '<a href="#/numerics">1.15</a>']
    ])}

<h2>Evaluation, ranking, experiments, ensembles</h2>
${H.table(['Name', 'Formula', '§'], [
      ['Ensemble variance', '$\\rho\\sigma^2+\\frac{1-\\rho}{M}\\sigma^2$ — the floor is $\\rho\\sigma^2$', '<a href="#/ensembles">2.16</a>'],
      ['GP posterior', '$\\mu_*=k_*^\\mathsf{T}(K+\\sigma_n^2I)^{-1}y$; $\\sigma_*^2=k_{**}-k_*^\\mathsf{T}(K+\\sigma_n^2I)^{-1}k_*$', '<a href="#/gp-bayesopt">2.21</a>'],
      ['Expected improvement', '$(f^+-\\mu)\\Phi(z)+\\sigma\\phi(z)$', '<a href="#/gp-bayesopt">2.21</a>'],
      ['InfoNCE', '$-\\log\\frac{e^{\\mathrm{sim}(a_i,b_i)/\\tau}}{\\sum_j e^{\\mathrm{sim}(a_i,b_j)/\\tau}}$; MI bound $\\le\\log B$', '<a href="#/self-supervised">2.22</a>'],
      ['DCG / NDCG', '$\\sum_i\\frac{2^{rel_i}-1}{\\log_2(i+1)}$, normalised by the ideal ordering', '<a href="#/ranking">2.24</a>'],
      ['Sample size (two proportions)', '$n\\approx 2p(1-p)\\left(\\frac{z_{\\alpha/2}+z_\\beta}{\\delta}\\right)^2$ per arm', '<a href="#/experimentation">2.25</a>'],
      ['MDE', '$(z_{\\alpha/2}+z_\\beta)\\sqrt{2p(1-p)/n}$', '<a href="#/experimentation">2.25</a>'],
      ['IoU / Dice', '$\\frac{|A\\cap B|}{|A\\cup B|}$; $\\text{Dice}=\\frac{2\\,\\text{IoU}}{1+\\text{IoU}}$', '<a href="#/vision-tasks">6.7</a>'],
      ['UCB1', '$\\hat\\mu_k+\\sqrt{2\\ln t/n_k}$', '<a href="#/bandits">6.6</a>'],
      ['Regret', '$R_T=T\\max_k\\mu_k-\\sum_t\\mu_{a_t}$; $O(\\log T)$ for good policies', '<a href="#/bandits">6.6</a>'],
      ['Checkpointing memory', '$(L/k+k)M$, minimised at $k=\\sqrt L$', '<a href="#/autodiff">3.11</a>'],
      ['Quantization', '$q=\\mathrm{round}(x/s)+z$; $\\hat x=s(q-z)$', '<a href="#/compression">3.13</a>'],
      ['Distillation loss', '$\\alpha T^2 D_{KL}(\\sigma(z_t/T)\\|\\sigma(z_s/T))+(1-\\alpha)\\mathrm{CE}$', '<a href="#/compression">3.13</a>'],
      ['FGSM', '$x\'=x+\\epsilon\\,\\mathrm{sign}(\\nabla_x\\mathcal{L})$', '<a href="#/robustness">3.14</a>'],
      ['Differential privacy', '$P(M(D)\\in S)\\le e^{\\varepsilon}P(M(D\')\\in S)+\\delta$', '<a href="#/privacy">6.8</a>'],
      ['ViT token count', '$(H/p)\\times(W/p)$ — 224 px at 16 px patches = 196', '<a href="#/multimodal">4.19</a>']
    ])}`
  });

  /* ------------------------------------------------------------------ R.3 */
  ML.section({
    id: 'glossary', track: 'reference', num: 'R.3',
    title: 'Glossary and index of terms',
    lede: 'Each entry gives the one-sentence definition you would say out loud, and the section that derives it. Use it as an index: if you cannot produce the definition from memory, go back to the section.',
    html: `${H.lab('gloss', 'Search the glossary', 'Filter by term or by definition — the second is often more useful, because it finds the concept when you have forgotten the name.')}`,
    labs: {
      gloss: function (host) {
        const G = [
          ['A2A', 'frameworks', 'Agent-to-agent protocol: capability cards, task delegation, streaming progress.'],
          ['ALiBi', 'rope', 'A linear penalty on attention logits proportional to token distance; extrapolates beyond trained length.'],
          ['ATE', 'causal', 'Average treatment effect, $\\mathbb{E}[Y(1)-Y(0)]$.'],
          ['Attention sink', 'rope', 'The first few tokens absorb excess attention mass; keep them cached or long-context quality collapses.'],
          ['AWQ', 'serving', 'Activation-aware 4-bit weight quantization that protects salient channels; the GPU production pick.'],
          ['Bagging', 'trees', 'Averaging models fit on bootstrap resamples to cut variance.'],
          ['Bandit', 'intervals', 'Shifts traffic toward the winning arm while learning; regret grows like log T for good algorithms.'],
          ['Bias–variance', 'bias-variance', 'Expected squared error = bias² + variance + irreducible noise.'],
          ['BPE', 'tokenization', 'Subword tokenization by repeatedly merging the most frequent adjacent pair.'],
          ['Brier score', 'calibration', 'Mean squared error of predicted probabilities; sensitive to calibration.'],
          ['Calibration', 'calibration', 'Agreement between predicted probabilities and observed frequencies.'],
          ['CatBoost', 'boosting', 'Boosting with ordered target statistics and oblivious trees.'],
          ['Causal mask', 'rope', '−∞ on future positions so a token cannot see ahead.'],
          ['Chinchilla', 'scaling-laws', 'Compute-optimal training scales parameters and tokens together, ≈20 tokens/param. ⚑ coefficients contested.'],
          ['CLIP', 'llm-eval', 'Contrastively aligned image and text encoders sharing one embedding space.'],
          ['ColBERT', 'rag', 'Late interaction: per-token embeddings scored with MaxSim; near cross-encoder precision, far lower latency.'],
          ['Concentration', 'concentration', 'Exponential bounds (Hoeffding) on the deviation of a sample mean; why holdouts work.'],
          ['Concept drift', 'production', 'A change in $P(y|x)$; visible only once labels arrive.'],
          ['Conformal prediction', 'calibration', 'Distribution-free prediction sets with a coverage guarantee, from ranked calibration residuals.'],
          ['Confounding', 'causal', 'A common cause of treatment and outcome that biases naive comparison.'],
          ['Context engineering', 'multi-agent', 'Deciding what occupies the context window each turn, and pruning the rest.'],
          ['Contamination', 'llm-eval', 'Benchmark data leaking into pretraining; why private evals are required.'],
          ['Cox PH', 'production', 'Hazard = baseline × exp(βᵀx); $e^\\beta$ is a hazard ratio, and censoring is handled properly.'],
          ['Cross-encoder', 'rag', 'Scores a query and document jointly in one forward pass; the biggest RAG lever.'],
          ['Cross-entropy', 'information', '$-\\sum p\\log q$; equals $H(p)+D_{KL}(p\\|q)$, so minimising it minimises KL.'],
          ['CUPED', 'intervals', 'Pre-period covariate adjustment; variance falls by $(1-\\rho^2)$, so tests finish sooner.'],
          ['DBSCAN', 'unsupervised', 'Density-based clustering; arbitrary shapes, explicit noise label.'],
          ['DiD', 'causal', 'Difference-in-differences; requires parallel trends.'],
          ['Double descent', 'bias-variance', 'Test error peaks at the interpolation threshold then falls again.'],
          ['DPO', 'post-training', 'Preference optimisation with no reward model; the partition function cancels in the pairwise difference.'],
          ['Elastic net', 'regularization', 'L1 + L2; sparsity without lasso’s instability across correlated features.'],
          ['ELBO', 'unsupervised', 'The Jensen lower bound on log-likelihood that EM tightens then maximises.'],
          ['EM', 'unsupervised', 'E-step sets q = p(z|x) (tightens the bound); M-step maximises it.'],
          ['Equalised odds', 'fairness', 'Equal TPR and FPR across groups; incompatible with within-group calibration when base rates differ.'],
          ['FlashAttention', 'kv-cache', 'Exact attention that tiles into SRAM with an online softmax; an IO optimisation, not an approximation.'],
          ['FP8', 'serving', '1 byte per element; near-BF16 quality on Hopper/Blackwell, for weights or KV cache.'],
          ['Gini (credit)', 'metrics', '2·AUC − 1.'],
          ['GOSS', 'boosting', 'LightGBM’s gradient-based one-side sampling: keep large gradients, subsample small ones.'],
          ['GQA', 'kv-cache', 'Grouped-query attention; heads share K/V in groups to shrink the cache.'],
          ['GRPO', 'post-training', 'Critic-free RL using group-relative advantages $(r_i-\\mathrm{mean})/\\mathrm{std}$.'],
          ['HNSW', 'rag', 'Graph ANN index; best recall per latency when vectors fit in RAM.'],
          ['Hoeffding', 'concentration', '$P(|\\bar X-\\mu|\\ge t)\\le 2e^{-2nt^2/(b-a)^2}$.'],
          ['HyDE', 'rag', 'Embed a hypothetical answer instead of the question, to close the vocabulary gap.'],
          ['IV', 'features', 'Information value; the symmetric KL between good and bad distributions. ⚑ bands are convention.'],
          ['Instrumental variable', 'causal', 'Z affecting Y only through T; identifies effects under unmeasured confounding.'],
          ['Isolation Forest', 'unsupervised', 'Anomaly score from how few random splits isolate a point; the low-tuning default.'],
          ['KL divergence', 'information', '$\\sum p\\log(p/q)\\ge0$; asymmetric — the "extra bits" for believing q.'],
          ['KS statistic', 'metrics', 'Maximum gap between two cumulative distributions; credit’s separation measure.'],
          ['KV cache', 'kv-cache', 'Stored keys and values per token: $2\\cdot L\\cdot n_{kv}\\cdot d_{head}\\cdot$seq·batch·bytes.'],
          ['Leakage', 'features', 'Using information unavailable at prediction time; the top interview-killer.'],
          ['LoRA', 'lora', 'Frozen $W_0$ plus a trainable low-rank $BA$; r=16, α=2r, all linear layers.'],
          ['Log-sum-exp', 'information', 'Subtract the max before exponentiating; behind stable softmax and online softmax.'],
          ['MAP', 'mle-map', 'Maximise likelihood plus log-prior; the prior IS the regularizer.'],
          ['Matryoshka embedding', 'rag', 'Trained so a truncated prefix still retrieves well; memory scales with the prefix you keep.'],
          ['MCP', 'mcp', 'The tool-integration protocol; stateless at the protocol layer since 2026-07-28.'],
          ['MLA', 'kv-cache', 'Multi-head latent attention: a low-rank K/V latent, smaller cache than GQA at MHA quality.'],
          ['MLE', 'mle-map', 'Maximise the likelihood; its negative log IS the loss.'],
          ['MoE', 'moe', 'Many expert FFNs with top-k routing; huge total parameters, small active parameters.'],
          ['MRTR', 'mcp', 'Multi round-trip requests; MCP’s stateless replacement for server-initiated input.'],
          ['MTP', 'serving', 'Multi-token prediction heads: a denser training signal and a built-in speculative drafter.'],
          ['μP', 'distributed', 'Width-aware parameterisation letting hyperparameters transfer from a small proxy to a large run.'],
          ['Normal equations', 'linear-logistic', '$w=(X^\\mathsf{T}X)^{-1}X^\\mathsf{T}y$; ridge adds $\\lambda I$.'],
          ['nDCG', 'rag', 'Rank metric discounting by $1/\\log_2(\\text{rank}+1)$, normalised by the ideal ordering.'],
          ['OOT validation', 'validation', 'Train on the past, validate on a later window; mandatory for time-structured data.'],
          ['Over-smoothing', 'gnn', 'Deep message passing converges all node representations to the same vector.'],
          ['OWASP LLM Top 10', 'production-ai', 'The standard control vocabulary for LLM application security.'],
          ['Paged attention', 'serving', 'KV cache in fixed pages with a block table; kills fragmentation, enables prefix sharing.'],
          ['PCA', 'pca', 'Project onto the top eigenvectors of the covariance matrix.'],
          ['Peeking', 'intervals', 'Stopping an A/B test at significance; inflates α unless a sequential method is used.'],
          ['Platt scaling', 'calibration', 'A logistic fit on scores to recalibrate probabilities.'],
          ['PR-AUC', 'metrics', 'Precision–recall area; the honest metric under heavy imbalance.'],
          ['Prefill / decode', 'serving', 'Prompt processing (compute-bound) vs token-by-token generation (bandwidth-bound).'],
          ['Prompt injection', 'production-ai', 'Untrusted text hijacking instructions; structural, so defences are architectural.'],
          ['Propensity score', 'causal', '$e(x)=P(T=1|x)$, used for matching or inverse-probability weighting.'],
          ['PSD', 'linear-algebra', '$x^\\mathsf{T}Ax\\ge0$; covariance and valid kernels are PSD.'],
          ['PSI', 'production', 'Population stability index for input drift. ⚑ bands are convention.'],
          ['QLoRA', 'lora', '4-bit NF4 frozen base plus a BF16 adapter.'],
          ['ReAct', 'agents', 'The Thought → Action → Observation agent loop.'],
          ['RLVR', 'post-training', 'RL from verifiable rewards (tests pass, answers match); how reasoning models are trained.'],
          ['RMSNorm', 'architectures', 'Normalise by root-mean-square, no mean subtraction; cheaper than LayerNorm.'],
          ['RoPE', 'rope', 'Rotary position embedding; $(R_mq)^\\mathsf{T}(R_nk)=q^\\mathsf{T}R_{n-m}k$.'],
          ['RRF', 'rag', '$\\sum_r 1/(k+\\mathrm{rank}_r)$ with k=60; fuses ranked lists, not scores.'],
          ['SHAP', 'interpretability', 'Shapley attributions summing exactly to $f(x)-\\mathbb{E}[f]$; TreeSHAP is exact for trees.'],
          ['SMOTE', 'calibration', 'Synthetic minority oversampling; harms calibration (JAMIA 2022). ⚑ contested.'],
          ['Sparse autoencoder', 'safety', 'Extracts monosemantic features from activations in superposition; enables auditing and steering.'],
          ['Speculative decoding', 'serving', 'A draft model proposes, the large model verifies in parallel; $(1-\\alpha^{k+1})/(1-\\alpha)$ tokens per pass.'],
          ['SSM / Mamba', 'moe', 'Selective linear recurrence; near-linear in length, constant state at inference.'],
          ['Stacking', 'hyperparameters', 'A meta-learner on out-of-fold base predictions; exploits disagreement between families.'],
          ['SVD', 'linear-algebra', '$A=U\\Sigma V^\\mathsf{T}$: rotate, scale, rotate; truncation is the best low-rank fit.'],
          ['SwiGLU', 'architectures', 'Gated FFN variant beating a plain MLP at matched parameters; width ≈ ⅔·4d.'],
          ['Task vector', 'lora', '$\\theta_{finetuned}-\\theta_{base}$; add or subtract to merge or remove capabilities.'],
          ['Test-time compute', 'decoding', 'Buying accuracy with inference: long CoT, voting, verifier search.'],
          ['Text-to-SQL', 'rag', 'The right retrieval tool for aggregate questions; guard with a read-only role and show the query.'],
          ['Thompson sampling', 'intervals', 'Draw from each arm’s posterior and play the argmax; usually the strongest simple bandit.'],
          ['Tool retrieval', 'agents', 'RAG over tool descriptions, because selection accuracy degrades past a few dozen tools.'],
          ['Training–serving skew', 'production', 'Features computed differently in training and production.'],
          ['ViT', 'llm-eval', 'An image as a sequence of patch tokens.'],
          ['WOE', 'features', 'Weight of evidence, $\\ln(\\%good/\\%bad)$ per bin; the scorecard’s native scale.'],
          ['WSD schedule', 'optimisers', 'Warmup, long stable phase, sharp final decay; matches cosine and allows branching.'],
          ['XGBoost gain', 'boosting', '$\\frac12[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{G^2}{H+\\lambda}]-\\gamma$.'],
          ['ZeRO / FSDP', 'distributed', 'Shard optimizer state, gradients and parameters across data-parallel ranks.'],

          /* --- terms added with Parts 0.6–0.8, 1.12–1.15, 2.16–2.25,
                 3.11–3.14, 4.19–4.22, 5.9–5.12, 6.6–6.8 and Part 7 --- */
          ['Active learning', 'active-transfer', 'Buy labels where the model is least certain; mix in random queries or it chases outliers.'],
          ['Adversarial training', 'robustness', 'Train on PGD examples; the only defence that has survived adaptive attacks, at 3–30× compute and ~10 points of clean accuracy.'],
          ['Always-valid inference', 'experimentation', 'Confidence sequences valid at every moment, so continuous monitoring does not inflate α.'],
          ['Autodiff (reverse mode)', 'autodiff', 'Chain rule evaluated numerically on a graph; one sweep per output, so a scalar loss is cheap and memory-hungry.'],
          ['Bayesian optimisation', 'gp-bayesopt', 'GP surrogate plus an acquisition function; wins when evaluations are expensive and dimension is under ~20.'],
          ['bf16', 'numerics', 'fp32’s exponent with 7 mantissa bits — range preserved, precision traded. Why it replaced fp16 for training.'],
          ['Catastrophic cancellation', 'numerics', 'Subtracting nearly equal large numbers destroys every significant digit; the reason one-pass variance can go negative.'],
          ['Checkpointing (gradient)', 'autodiff', 'Store activations every k layers and recompute the rest; $O(L)\\to O(\\sqrt L)$ memory for ~30% more compute.'],
          ['Complementary slackness', 'optimization', '$\\lambda_i g_i(x^\\star)=0$ — a constraint is either tight or has zero multiplier. Why SVMs are sparse in the data.'],
          ['Condition number', 'optimization', '$\\kappa=\\lambda_{\\max}/\\lambda_{\\min}$ of the Hessian; it sets how many gradient steps convergence needs.'],
          ['Constrained decoding', 'structured-output', 'Mask logits that cannot continue a valid string; makes invalid output impossible rather than unlikely.'],
          ['Contrastive learning', 'self-supervised', 'Pull two views of the same item together and push others apart; InfoNCE over a batch-sized classification.'],
          ['Cross-validation leakage (group)', 'active-transfer', 'The same entity in train and test; split on IDs, not rows.'],
          ['DP-SGD', 'privacy', 'Clip per-example gradients, add calibrated Gaussian noise, account for the budget. The only defence with a proof.'],
          ['Differential privacy', 'privacy', 'The output is nearly as likely with or without any one individual: $P(M(D)\\in S)\\le e^\\varepsilon P(M(D\')\\in S)+\\delta$.'],
          ['Distillation', 'compression', 'Train a student on the teacher’s soft outputs; the $T^2$ factor restores the gradient scale.'],
          ['DETR', 'vision-tasks', 'Set-based detection with Hungarian matching — no anchors and no NMS, at the cost of slow convergence.'],
          ['efSearch', 'vector-search', 'HNSW’s query-time candidate list — the one knob that trades recall against latency without a rebuild.'],
          ['Focal loss', 'vision-tasks', '$(1-p_t)^\\gamma$ down-weights easy negatives so a dense detector is not swamped by background.'],
          ['Gaussian process', 'gp-bayesopt', 'A prior over functions; the posterior is Gaussian in closed form and its variance does not depend on $y$.'],
          ['Group-wise quantization', 'compression', 'One scale per block of 64–128 weights; what makes 4-bit viable at all.'],
          ['Isotonic regression', 'calibration', 'A monotone step-function calibrator fitted by pool-adjacent-violators; more flexible than Platt, easier to overfit.'],
          ['Jump-ahead', 'structured-output', 'Emit deterministic schema tokens without a forward pass; why constrained decoding can be faster than free decoding.'],
          ['Lottery ticket hypothesis', 'compression', 'A dense network contains a sparse subnetwork that trains to the same accuracy from the same init — real, and not yet actionable.'],
          ['Matérn kernel', 'gp-bayesopt', 'A finitely-differentiable GP kernel; the sane default where RBF interpolates over-confidently.'],
          ['MCMC', 'bayesian-inference', 'Build a chain whose stationary distribution is the posterior; the acceptance ratio cancels the intractable evidence.'],
          ['NDCG', 'ranking', 'Position-discounted graded relevance, normalised by the ideal ordering.'],
          ['NMS', 'vision-tasks', 'Keep the top-scoring box, suppress overlaps above an IoU threshold, repeat.'],
          ['NSW / HNSW graph', 'vector-search', 'Short edges for local refinement, long edges as motorways; greedy descent finds neighbours in $O(\\log n)$ hops.'],
          ['Peeking', 'experimentation', 'Checking a fixed-horizon test repeatedly; inflates the false-positive rate from 5% to 20–35%.'],
          ['Position bias', 'ranking', 'Users click the top result because it is on top; training on raw clicks reproduces the previous ranker.'],
          ['Product quantization', 'vector-search', 'Split a vector into sub-spaces, replace each by a 256-centroid codebook index — 64× compression, distances by table lookup.'],
          ['Prompt-lookup decoding', 'speculative', 'Draft by n-gram matching against the prompt; no model, no training, very high acceptance when the output quotes the input.'],
          ['Self-consistency', 'reasoning', 'Sample $k$ traces and take the majority; buys accuracy only when the errors are diverse.'],
          ['Small-to-big retrieval', 'chunking', 'Embed small chunks for precision, return their parent section for completeness.'],
          ['Speculative decoding', 'speculative', 'Draft $k$ tokens cheaply, verify in one pass; provably identical output distribution.'],
          ['Test-time scaling', 'reasoning', 'Trading inference compute for accuracy; large and reproducible on verifiable tasks, much smaller elsewhere.'],
          ['Thompson sampling (bandit)', 'bandits', 'Draw once from each posterior and play the argmax — plays each arm with the probability it is best.'],
          ['UCB1', 'bandits', 'Optimism in the face of uncertainty: $\\hat\\mu_k+\\sqrt{2\\ln t/n_k}$, from a Hoeffding radius.'],
          ['Vector-Jacobian product', 'autodiff', 'The only primitive a layer must implement; a Jacobian is never formed.'],
          ['Welford’s algorithm', 'numerics', 'One-pass, numerically stable mean and variance that also streams.']
        ];
        G.sort((a, b) => a[0].toLowerCase().localeCompare(b[0].toLowerCase()));
        const input = ML.el('input', { type: 'search', placeholder: 'search ' + G.length + ' terms — or search the definitions', style: 'margin-bottom:12px' });
        host.appendChild(input);
        const list = ML.el('div');
        host.appendChild(list);
        function render() {
          const q = input.value.toLowerCase().trim();
          const rows = G.filter(g => !q || (g[0] + ' ' + g[2]).toLowerCase().indexOf(q) >= 0);
          list.innerHTML = '<div class="grid2">' + [0, 1].map(col =>
            '<div>' + rows.filter((_, i) => i % 2 === col).map(g =>
              '<p style="margin:0 0 10px"><b>' + g[0] + '</b> <a href="#/' + g[1] + '" style="font-size:.85em">§</a> — ' + g[2] + '</p>'
            ).join('') + '</div>').join('') + '</div>' +
            '<p class="small">' + rows.length + ' of ' + G.length + ' terms</p>';
          ML.typeset(list);
        }
        input.addEventListener('input', render);
        render();
      }
    }
  });

  /* ------------------------------------------------------------------ R.4 */
  ML.section({
    id: 'sources', track: 'reference', num: 'R.4',
    title: 'Sources',
    lede: 'Primary papers and official documentation were used for every derivation. Secondary explainers informed intuition only and are load-bearing for nothing.',
    html: `
${H.table(['Source', 'What it underwrites'], [
      ['Vaswani et al., <i>Attention Is All You Need</i> (2017)', '<a href="#/attention">4.3</a>–<a href="#/block">4.5</a>'],
      ['Chen &amp; Guestrin, <i>XGBoost</i> (KDD 2016)', '<a href="#/boosting">2.8</a> objective, leaf weight, gain'],
      ['Ke et al., <i>LightGBM</i> (NeurIPS 2017)', '<a href="#/boosting">2.8</a> histogram, leaf-wise, GOSS'],
      ['Prokhorenkova et al., <i>CatBoost</i> (2018)', '<a href="#/boosting">2.8</a> ordered target statistics'],
      ['Lundberg &amp; Lee (2017); Lundberg et al., TreeSHAP (2018)', '<a href="#/interpretability">2.17</a>'],
      ['van den Goorbergh, van Smeden, Timmerman &amp; Van Calster, <i>JAMIA</i> 29(9):1525–1534 (2022), doi:10.1093/jamia/ocac093', '<a href="#/calibration">2.12</a> imbalance and calibration'],
      ['Hoffmann et al., <i>Training Compute-Optimal LLMs</i> (2022), with Epoch AI’s replication', '<a href="#/scaling-laws">4.10</a> ⚑ coefficients contested'],
      ['Su et al., <i>RoFormer</i> (RoPE, 2021); Press et al., <i>ALiBi</i> (2021); Peng et al., <i>YaRN</i> (2023)', '<a href="#/rope">4.4</a>'],
      ['Dao et al., <i>FlashAttention</i> (2022) and FlashAttention-2 (2023)', '<a href="#/kv-cache">4.7</a>'],
      ['Rafailov et al., <i>Direct Preference Optimization</i> (2023)', '<a href="#/post-training">4.12</a> derivation'],
      ['Park et al., <i>Disentangling Length from Quality in DPO</i>, arXiv:2403.19159', '<a href="#/post-training">4.12</a> length bias'],
      ['DeepSeek-AI, <i>DeepSeek-R1</i>, arXiv:2501.12948; DeepSeek-V3 technical report', '<a href="#/post-training">4.12</a> GRPO, RLVR; <a href="#/moe">4.8</a> MoE, MLA'],
      ['Hu et al., <i>LoRA</i> (2021); Dettmers et al., <i>QLoRA</i> (2023)', '<a href="#/lora">4.13</a>'],
      ['Kwon et al., <i>vLLM / PagedAttention</i> (SOSP 2023)', '<a href="#/serving">4.14</a>'],
      ['Lin et al., <i>AWQ</i> (2023); Frantar et al., <i>GPTQ</i> (2022)', '<a href="#/serving">4.14</a>'],
      ['Cormack, Clarke &amp; Büttcher, <i>Reciprocal Rank Fusion</i> (SIGIR 2009)', '<a href="#/rag">5.1</a>, k = 60'],
      ['Khattab &amp; Zaharia, <i>ColBERT</i> (2020)', '<a href="#/rag">5.1</a> late interaction'],
      ['Yao et al., <i>ReAct</i> (2022)', '<a href="#/agents">5.3</a>'],
      ['Model Context Protocol, <i>The 2026-07-28 Specification</i> and its release candidate (21 May 2026), with the official blog and changelog', '<a href="#/mcp">5.5</a>'],
      ['Kleinberg, Mullainathan &amp; Raghavan (2016); Chouldechova (2017)', '<a href="#/fairness">2.19</a> impossibility result'],
      ['Ho et al., <i>Denoising Diffusion Probabilistic Models</i> (2020); Rombach et al., <i>Latent Diffusion</i> (2022)', '<a href="#/diffusion">6.2</a>'],
      ['Kingma &amp; Welling, <i>Auto-Encoding Variational Bayes</i> (2013); Goodfellow et al., <i>GANs</i> (2014)', '<a href="#/vae-gan">6.3</a>'],
      ['Sutton &amp; Barto, <i>Reinforcement Learning</i> (2nd ed.)', '<a href="#/rl">6.1</a>'],
      ['Schulman et al., <i>PPO</i> (2017)', '<a href="#/post-training">4.12</a>, <a href="#/rl">6.1</a>'],
      ['OWASP Top 10 for LLM Applications; OpenTelemetry GenAI semantic conventions', '<a href="#/production-ai">5.7</a>'],
      ['Anthropic model and system cards; Jimenez et al., <i>SWE-bench</i> (2023)', 'currency anchor, <a href="#/llm-eval">4.16</a>'],
      ['Boyd &amp; Vandenberghe, <i>Convex Optimization</i>; Nesterov (1983)', '<a href="#/optimization">1.12</a> duality, KKT, accelerated rates'],
      ['Efron (1979); Efron &amp; Tibshirani, <i>An Introduction to the Bootstrap</i>', '<a href="#/sampling">1.13</a>'],
      ['Gelman et al., <i>Bayesian Data Analysis</i>; Hoffman &amp; Gelman, <i>NUTS</i> (2014)', '<a href="#/bayesian-inference">1.14</a>'],
      ['Goldberg, <i>What Every Computer Scientist Should Know About Floating-Point</i> (1991); Welford (1962)', '<a href="#/numerics">1.15</a>'],
      ['Breiman, <i>Bagging</i> (1996) and <i>Random Forests</i> (2001); Wolpert, <i>Stacked Generalization</i> (1992)', '<a href="#/ensembles">2.16</a>'],
      ['Rasmussen &amp; Williams, <i>Gaussian Processes for Machine Learning</i>; Shahriari et al., <i>Taking the Human Out of the Loop</i> (2016)', '<a href="#/gp-bayesopt">2.21</a>'],
      ['Chen et al., <i>SimCLR</i> (2020); Grill et al., <i>BYOL</i> (2020); Chen &amp; He, <i>SimSiam</i> (2021); Oord et al., <i>CPC/InfoNCE</i> (2018)', '<a href="#/self-supervised">2.22</a>'],
      ['Burges, <i>From RankNet to LambdaRank to LambdaMART</i> (2010); Joachims et al., <i>Unbiased Learning-to-Rank</i> (2017)', '<a href="#/ranking">2.24</a>'],
      ['Deng et al., <i>CUPED</i> (WSDM 2013); Johari et al., <i>Always Valid Inference</i> (2017); Kohavi et al., <i>Trustworthy Online Controlled Experiments</i>', '<a href="#/experimentation">2.25</a>'],
      ['Baydin et al., <i>Automatic Differentiation in Machine Learning: a Survey</i> (2018); Chen et al., <i>Training Deep Nets with Sublinear Memory Cost</i> (2016)', '<a href="#/autodiff">3.11</a>'],
      ['Smith, <i>Cyclical Learning Rates</i> (2017)', '<a href="#/training-dynamics">3.12</a>'],
      ['Hinton et al., <i>Distilling the Knowledge in a Neural Network</i> (2015); Dettmers et al., <i>LLM.int8()</i> (2022); Xiao et al., <i>SmoothQuant</i> (2022); Frankle &amp; Carbin, <i>Lottery Ticket</i> (2019)', '<a href="#/compression">3.13</a>'],
      ['Goodfellow et al., <i>Explaining and Harnessing Adversarial Examples</i> (2015); Madry et al. (2018); Athalye et al., <i>Obfuscated Gradients</i> (2018); Ilyas et al., <i>Adversarial Examples Are Not Bugs</i> (2019)', '<a href="#/robustness">3.14</a>'],
      ['Dosovitskiy et al., <i>An Image is Worth 16×16 Words</i> (2021); Radford et al., <i>CLIP</i> (2021); Liu et al., <i>LLaVA</i> (2023); Thrush et al., <i>Winoground</i> (2022)', '<a href="#/multimodal">4.19</a>'],
      ['Wei et al., <i>Chain-of-Thought</i> (2022); Wang et al., <i>Self-Consistency</i> (2023); Lightman et al., <i>Let’s Verify Step by Step</i> (2023); Snell et al., <i>Scaling LLM Test-Time Compute</i> (2024)', '<a href="#/reasoning">4.20</a>'],
      ['Willard &amp; Louf, <i>Efficient Guided Generation</i> (Outlines, 2023); XGrammar (2024)', '<a href="#/structured-output">4.21</a>'],
      ['Leviathan et al., <i>Fast Inference from Transformers via Speculative Decoding</i> (2023); Chen et al. (2023); Cai et al., <i>Medusa</i> (2024)', '<a href="#/speculative">4.22</a>'],
      ['Malkov &amp; Yashunin, <i>HNSW</i> (2018); Jégou et al., <i>Product Quantization</i> (2011); Subramanya et al., <i>DiskANN</i> (2019)', '<a href="#/vector-search">5.10</a>'],
      ['Zheng et al., <i>Judging LLM-as-a-Judge with MT-Bench</i> (2023)', '<a href="#/evals">5.11</a>'],
      ['Sculley et al., <i>Hidden Technical Debt in Machine Learning Systems</i> (NeurIPS 2015)', '<a href="#/mlops">5.12</a>'],
      ['Lattimore &amp; Szepesvári, <i>Bandit Algorithms</i>; Auer et al., <i>UCB1</i> (2002); Chapelle &amp; Li, <i>An Empirical Evaluation of Thompson Sampling</i> (2011)', '<a href="#/bandits">6.6</a>'],
      ['Ren et al., <i>Faster R-CNN</i> (2015); Lin et al., <i>Focal Loss</i> (2017); Carion et al., <i>DETR</i> (2020); Kirillov et al., <i>SAM</i> (2023)', '<a href="#/vision-tasks">6.7</a>'],
      ['Dwork &amp; Roth, <i>The Algorithmic Foundations of Differential Privacy</i>; Abadi et al., <i>DP-SGD</i> (2016); Zhu et al., <i>Deep Leakage from Gradients</i> (2019); Carlini et al., <i>Extracting Training Data from LLMs</i> (2021)', '<a href="#/privacy">6.8</a>'],
      ['Bergstra &amp; Bengio, <i>Random Search for Hyper-Parameter Optimization</i> (2012); Li et al., <i>Hyperband</i> (2018)', '<a href="#/gp-bayesopt">2.21</a>, <a href="#/hyperparameters">2.15</a>']
    ])}

${H.note('Three claims are genuinely contested and should be flagged every single time you use them: the exact Chinchilla coefficients, "what works" for class imbalance, and the PSI/IV thresholds. Flagging them is not hedging — it is the thing that makes the rest of your answers credible.')}

<h2>On this site’s relationship to its source</h2>
<p>This site is a live companion to <i>The AI/ML Study Notebook</i> (edition July 2026): every section of that notebook is covered here, restructured for interaction, and the material it deliberately excluded is restored in <a href="#/exclusions">Part 6</a>. Where the notebook gives a worked number, the corresponding lab here recomputes it so you can move the inputs — the Bayes screen, the XGBoost split, the KV cache, the QLoRA budget, the conformal quantile, the RRF fusion and the cost-per-successful-task comparison all reproduce their printed values exactly at the default settings.</p>
<p>Every model on this site trains in your browser. Nothing is a recorded animation; if a curve moves, something computed it.</p>`
  });

  /* ------------------------------------------------------------------ R.5 */
  ML.section({
    id: 'drill', track: 'reference', num: 'R.5',
    title: 'The drill room',
    lede: 'Every recall card from every section, shuffled. Say the answer out loud before flipping — retrieval failure is what trains recall, and rereading is what feels like learning without being it.',
    html: `${H.lab('bigdrill', 'The whole curriculum, shuffled', 'Filter by track, shuffle, and keep score. Progress is stored in this browser only.')}`,
    labs: {
      bigdrill: function (host) {
        const all = [];
        ML.sections.forEach(s => (s.cards || []).forEach(c => all.push({ q: c.q, a: c.a, track: s.trackRef.short, num: s.num, id: s.id, title: s.title })));
        let pool = all.slice(), order = [], i = 0, showA = false, right = 0, seen = 0;
        const st = Viz.controls(host, [
          { k: 'track', label: 'track', type: 'select', value: 'all', options: [{ v: 'all', t: 'everything (' + all.length + ' cards)' }].concat(
            ML.tracks.map(t => ({ v: t.short, t: t.short + ' — ' + t.title }))) }
        ], () => { rebuild(); });
        function rebuild() {
          pool = st.track === 'all' ? all.slice() : all.filter(c => c.track === st.track);
          order = pool.map((_, k) => k).sort(() => Math.random() - .5);
          i = 0; showA = false; right = 0; seen = 0; draw();
        }
        const face = ML.el('div', {
          class: 'card-face',
          style: 'cursor:pointer;border:1px solid var(--line);border-radius:12px;background:var(--panel);min-height:130px;flex-direction:column;gap:10px'
        });
        host.appendChild(face);
        const meta = ML.el('p', { class: 'small', style: 'text-align:center;margin:8px 0 0' });
        host.appendChild(meta);
        const bar = ML.el('div', { style: 'height:6px;border-radius:99px;background:var(--panel);overflow:hidden;margin:12px 0 0' });
        const barFill = ML.el('i', { style: 'display:block;height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green))' });
        bar.appendChild(barFill);
        host.appendChild(bar);
        function draw() {
          if (!pool.length) { face.innerHTML = '<div class="a">No cards in this track.</div>'; return; }
          const c = pool[order[i]];
          face.innerHTML = showA
            ? '<div class="a">' + c.a + '</div>'
            : '<div><b>' + c.q + '</b></div>';
          meta.innerHTML = (i + 1) + ' / ' + pool.length + ' · <a href="#/' + c.id + '">' + c.num + ' ' + c.title + '</a>' +
            (seen ? ' · <b>' + right + '/' + seen + '</b> recalled (' + (100 * right / seen).toFixed(0) + '%)' : '');
          barFill.style.width = (100 * (i + 1) / pool.length) + '%';
          ML.typeset(face);
        }
        face.addEventListener('click', () => { showA = !showA; draw(); });
        const nav = ML.el('div', { class: 'cardnav' });
        nav.appendChild(ML.el('button', { class: 'btn primary', type: 'button', text: 'Flip', onclick: () => { showA = !showA; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '✓ knew it', onclick: () => { if (showA) { right++; seen++; } i = (i + 1) % pool.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: '✗ missed', onclick: () => { if (showA) seen++; i = (i + 1) % pool.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'skip →', onclick: () => { i = (i + 1) % pool.length; showA = false; draw(); } }));
        nav.appendChild(ML.el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: rebuild }));
        host.appendChild(nav);
        rebuild();
      }
    }
  });
})();
