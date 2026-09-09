/* ============================================================
   PART 6 — Beyond the notebook: the deliberate exclusions, restored
   ============================================================ */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ 6.1 */
  ML.section({
    id: 'rl', track: 'frontier', num: '6.1',
    title: 'Reinforcement learning, from MDPs up',
    lede: 'Part 4 used RL as a post-training tool without ever defining it. Here is the whole object — and once you have it, PPO, GRPO and RLVR stop being acronyms.',
    html: `
<h2><span class="sn">6.1.1</span> The Markov decision process</h2>
<p>States $s$, actions $a$, transition dynamics $P(s'\\mid s,a)$, reward $R(s,a)$, discount $\\gamma \\in [0,1)$. A <b>policy</b> $\\pi(a\\mid s)$ chooses actions. The <b>value</b> of a state under a policy is the expected discounted return:</p>
$$V^\\pi(s) = \\mathbb{E}\\left[\\sum_{t=0}^\\infty \\gamma^t R_t \\;\\middle|\\; s_0 = s\\right]$$
<p>and the <b>Bellman equation</b> makes it recursive:</p>
$$V^*(s) = \\max_a \\left[R(s,a) + \\gamma\\sum_{s'}P(s'\\mid s,a)V^*(s')\\right]$$
<p>The discount $\\gamma$ does two jobs: it makes the infinite sum converge, and it encodes how much the future matters — which is a modelling choice, not a technicality.</p>

${H.lab('gridworld', 'Value iteration and Q-learning, on a gridworld', 'Run value iteration to convergence and watch the values propagate outward from the goal — that is the Bellman backup, visible. Then switch to Q-learning, which never sees the transition model and has to discover it by acting.')}

<h2><span class="sn">6.1.2</span> Model-free control</h2>
${H.table(['Method', 'Update', 'Character'], [
      ['<b>Q-learning</b> (off-policy)', '$Q(s,a) \\mathrel{+}= \\alpha[r + \\gamma\\max_{a\'}Q(s\',a\') - Q(s,a)]$', 'Learns the optimal policy while behaving differently (e.g. ε-greedy)'],
      ['<b>SARSA</b> (on-policy)', '$Q(s,a) \\mathrel{+}= \\alpha[r + \\gamma Q(s\',a\') - Q(s,a)]$', 'Learns the value of the policy it actually follows — safer near cliffs'],
      ['<b>Policy gradient</b>', '$\\nabla J = \\mathbb{E}[\\nabla\\log\\pi(a\\mid s)\\,A(s,a)]$', 'Optimises the policy directly; works in continuous action spaces'],
      ['<b>Actor–critic</b>', 'policy + learned value baseline', 'Lower-variance policy gradients — PPO’s family'],
      ['<b>PPO</b>', 'clipped ratio objective (§4.12)', 'The workhorse; GRPO drops the critic and uses a sampled group as baseline']
    ])}
<p>The exploration–exploitation problem from §1.6 reappears here in full: ε-greedy, optimistic initialisation, and entropy bonuses are the standard answers, and the last is why RLHF objectives frequently carry an entropy term.</p>

<h2><span class="sn">6.1.3</span> The link back to Part 4</h2>
<p>In RLHF the "environment" is a prompt, the "action" is a completion, and the "reward" is a preference model or a verifier. The episode is one step long, which is why the machinery simplifies so much: no bootstrapping across time, no discount to argue about, and the variance problem is handled by a baseline — a learned value network in PPO, the sampled group's mean in GRPO. Seeing that correspondence is what makes §4.12 feel inevitable rather than arbitrary.</p>

${H.probe([
      ['What does the Bellman equation say?', 'The value of a state is the best immediate reward plus the discounted value of where you land.'],
      ['Q-learning vs SARSA?', 'Off-policy (learns the optimal policy while exploring) vs on-policy (learns the value of the behaviour policy) — SARSA is more cautious near catastrophic states.'],
      ['How does RLHF map onto an MDP?', 'One-step episodes: prompt = state, completion = action, reward model or verifier = reward.']
    ])}`,
    labs: {
      gridworld: function (host) {
        const W = 8, H = 6;
        const walls = new Set(['2,1', '2,2', '2,3', '5,2', '5,3', '5,4']);
        const goal = '7,0', trap = '6,3';
        const st = Viz.controls(host, [
          { k: 'algo', label: 'algorithm', type: 'buttons', value: 'vi', options: [{ v: 'vi', t: 'value iteration' }, { v: 'ql', t: 'Q-learning' }] },
          { k: 'gamma', label: 'discount γ', min: .5, max: .99, step: .01, value: .92, fmt: v => v.toFixed(2) },
          { k: 'step', label: 'step cost', min: -.2, max: 0, step: .01, value: -.04, fmt: v => v.toFixed(2) },
          { k: 'eps', label: 'ε (exploration, Q-learning)', min: 0, max: .8, step: .05, value: .25, fmt: v => v.toFixed(2) }
        ], reset);
        const out = Viz.readout(host, [
          { k: 'iter', label: 'iterations / episodes', cls: 'key' }, { k: 'delta', label: 'largest value change' },
          { k: 'start', label: 'value of the start state' }, { k: 'converged', label: 'converged?' }
        ]);
        let V, Q, iters = 0, lastDelta = 1;
        const key = (x, y) => x + ',' + y;
        function reset() {
          V = {}; Q = {}; iters = 0; lastDelta = 1;
          for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
            V[key(x, y)] = 0;
            Q[key(x, y)] = [0, 0, 0, 0];
          }
          S.redraw();
        }
        const moves = [[0, -1], [1, 0], [0, 1], [-1, 0]];
        function rewardAt(k) { return k === goal ? 1 : k === trap ? -1 : st.step; }
        function stepVI() {
          let delta = 0;
          const nV = {};
          for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
            const k = key(x, y);
            if (walls.has(k) || k === goal || k === trap) { nV[k] = rewardAt(k) === st.step ? 0 : rewardAt(k); continue; }
            let best = -Infinity;
            moves.forEach(m => {
              const nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
              const nk = walls.has(key(nx, ny)) ? k : key(nx, ny);
              best = Math.max(best, st.step + st.gamma * V[nk]);
            });
            nV[k] = best;
            delta = Math.max(delta, Math.abs(best - V[k]));
          }
          V = nV; iters++; lastDelta = delta;
        }
        function episodeQL() {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          let x = 0, y = H - 1, steps = 0;
          let delta = 0;
          while (steps++ < 200) {
            const k = key(x, y);
            if (k === goal || k === trap) break;
            const a = R() < st.eps ? R.int(4) : Q[k].indexOf(Math.max.apply(null, Q[k]));
            const m = moves[a];
            let nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
            if (walls.has(key(nx, ny))) { nx = x; ny = y; }
            const nk = key(nx, ny);
            const r = rewardAt(nk);
            const target = r + ((nk === goal || nk === trap) ? 0 : st.gamma * Math.max.apply(null, Q[nk]));
            const old = Q[k][a];
            Q[k][a] += .25 * (target - Q[k][a]);
            delta = Math.max(delta, Math.abs(Q[k][a] - old));
            x = nx; y = ny;
          }
          iters++; lastDelta = delta;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const cell = Math.min((w - 40) / W, (h - 50) / H);
            const ox = 20, oy = 24;
            const val = (x, y) => st.algo === 'vi' ? V[key(x, y)] : Math.max.apply(null, Q[key(x, y)]);
            let lo = 0, hi = 0;
            for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) { const v = val(x, y); lo = Math.min(lo, v); hi = Math.max(hi, v); }
            for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
              const k = key(x, y), px = ox + x * cell, py = oy + y * cell;
              if (walls.has(k)) { ctx.fillStyle = T.dark ? '#2a3050' : '#c9cee0'; ctx.fillRect(px, py, cell - 2, cell - 2); continue; }
              const v = val(x, y);
              const t = (v - lo) / ((hi - lo) || 1);
              ctx.fillStyle = k === goal ? 'rgba(60,180,110,.85)' : k === trap ? 'rgba(220,80,70,.85)'
                : 'rgba(' + Math.round(70 + 120 * t) + ',' + Math.round(90 + 90 * t) + ',' + Math.round(200 - 40 * t) + ',' + (0.12 + 0.7 * t) + ')';
              ctx.fillRect(px, py, cell - 2, cell - 2);
              ctx.strokeStyle = T.line; ctx.strokeRect(px, py, cell - 2, cell - 2);
              ctx.fillStyle = T.text; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(v.toFixed(2), px + cell / 2 - 1, py + cell / 2 - 5);
              // policy arrow
              if (k !== goal && k !== trap) {
                let bestA = 0, bestV = -Infinity;
                moves.forEach((m, ai) => {
                  const nx = Math.max(0, Math.min(W - 1, x + m[0])), ny = Math.max(0, Math.min(H - 1, y + m[1]));
                  const nk = walls.has(key(nx, ny)) ? k : key(nx, ny);
                  const q = st.algo === 'vi' ? V[nk] : Q[k][ai];
                  if (q > bestV) { bestV = q; bestA = ai; }
                });
                const arrows = ['↑', '→', '↓', '←'];
                ctx.fillStyle = T.muted; ctx.font = '13px ui-sans-serif';
                ctx.fillText(arrows[bestA], px + cell / 2 - 1, py + cell / 2 + 9);
              }
            }
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('green = goal (+1) · red = trap (−1) · arrows = greedy policy · start is bottom-left', ox, oy + H * cell + 8);
            out({
              iter: iters, delta: lastDelta.toFixed(4),
              start: val(0, H - 1).toFixed(3),
              converged: lastDelta < .001 ? 'yes' : 'not yet'
            });
          }
        });
        Viz.buttons(host, [
          { label: st.algo === 'vi' ? 'One sweep' : 'One episode', primary: true, on: () => { st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Run 30', on: () => { for (let i = 0; i < 30; i++) st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Run 300', on: () => { for (let i = 0; i < 300; i++) st.algo === 'vi' ? stepVI() : episodeQL(); S.redraw(); } },
          { label: 'Reset', on: reset }
        ]);
        reset();
        Viz.note(host, 'Value iteration knows the transition model and propagates values outward from the goal one sweep at a time. Q-learning knows nothing and must bump into the goal by accident before any value exists — set ε to 0 and watch it fail to find the goal at all. That is the exploration problem, and it is why every RL system needs an answer to it.');
      }
    },
    quiz: [
      {
        q: 'The discount factor γ…',
        options: ['controls the learning rate', 'makes the infinite return converge and encodes how much the future matters', 'sets the exploration rate', 'normalises rewards'],
        answer: 1,
        why: 'Both roles matter; choosing γ is a modelling decision about the horizon you care about.'
      },
      {
        q: 'RLHF maps onto an MDP as…',
        options: ['a long episode with many states', 'a one-step episode: prompt = state, completion = action, reward model = reward', 'an unsupervised problem', 'a bandit with no context'],
        answer: 1,
        why: 'One-step episodes are why no bootstrapping or discounting is needed and why the baseline (critic or group mean) is the whole variance story.'
      }
    ],
    cards: [
      { q: 'Bellman optimality', a: '$V^*(s)=\\max_a[R(s,a)+\\gamma\\sum_{s\'}P(s\'|s,a)V^*(s\')]$.' },
      { q: 'Q-learning vs SARSA', a: 'Off-policy max over next actions vs on-policy value of the action actually taken.' },
      { q: 'RLHF as an MDP', a: 'One-step episode: prompt = state, completion = action, preference model or verifier = reward.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.2 */
  ML.section({
    id: 'diffusion', track: 'frontier', num: '6.2',
    title: 'Diffusion models',
    lede: 'Destroy structure with noise on a fixed schedule, then learn to undo one step of it. The generative model that took over images, audio and video.',
    html: `
<h2><span class="sn">6.2.1</span> The forward process is not learned</h2>
<p>Add Gaussian noise on a fixed schedule $\\beta_1 \\ldots \\beta_T$:</p>
$$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(\\sqrt{1-\\beta_t}\\,x_{t-1},\\; \\beta_t I\\right)$$
<p>and because Gaussians compose, you can jump to any step in closed form with $\\bar\\alpha_t = \\prod_{s\\le t}(1-\\beta_s)$:</p>
$$x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$$
<p>That closed form is what makes training cheap: sample a random $t$, noise the image in one operation, and ask the network to predict the noise you added. The loss is a plain regression:</p>
$$\\mathcal{L} = \\mathbb{E}_{t,x_0,\\epsilon}\\big[\\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2\\big]$$

<h2><span class="sn">6.2.2</span> The reverse process is the model</h2>
<p>Sampling starts from pure noise and repeatedly removes the predicted noise, one step at a time. DDPM does this stochastically over ~1,000 steps; DDIM makes it deterministic and lets you skip steps (20–50 is common); modern samplers and distillation push it to a handful. <b>Classifier-free guidance</b> mixes a conditional and an unconditional prediction, $\\hat\\epsilon = \\epsilon_\\theta(x_t,\\varnothing) + w(\\epsilon_\\theta(x_t,c) - \\epsilon_\\theta(x_t,\\varnothing))$, trading diversity for prompt adherence as $w$ rises — the guidance-scale slider in every image tool.</p>

${H.lab('diffuse', 'A diffusion model, trained in your browser', 'This trains a small denoiser on a 2-D shape, then samples from noise. Watch the forward process destroy the structure, and the reverse process rebuild it — the same mechanism as an image model, in two dimensions so you can see it.')}

<h2><span class="sn">6.2.3</span> Latent diffusion, and why it is affordable</h2>
<p>Running diffusion on raw pixels is expensive. <b>Latent diffusion</b> (the Stable Diffusion family) trains an autoencoder first and runs the whole diffusion process in its much smaller latent space, decoding once at the end — roughly an order of magnitude cheaper for the same perceptual quality. Conditioning (text, depth, pose) enters through cross-attention (§4.6's encoder–decoder path), which is where the transformer machinery re-joins the story; <b>DiT</b> architectures replace the U-Net with a transformer outright.</p>

${H.probe([
      ['What does the network actually predict?', 'The noise added at step t (equivalently the score); the loss is a plain squared error.'],
      ['Why is training cheap despite 1,000 steps?', 'The closed form lets you jump straight to a random step; you never simulate the chain during training.'],
      ['What does guidance scale trade?', 'Prompt adherence against diversity and artefacts.']
    ])}`,
    labs: {
      diffuse: function (host) {
        const st = Viz.controls(host, [
          { k: 'shape', label: 'target distribution', type: 'buttons', value: 'moons', options: [{ v: 'moons', t: 'two moons' }, { v: 'circles', t: 'ring' }, { v: 'spiral', t: 'spiral' }] },
          { k: 't', label: 'noise level t', min: 0, max: 1, step: .02, value: 0, fmt: v => (v * 100).toFixed(0) + '%' },
          { k: 'steps', label: 'sampling steps', min: 5, max: 60, step: 5, value: 30, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'trained', label: 'denoiser training steps', cls: 'key' }, { k: 'loss', label: 'denoising loss' },
          { k: 'abar', label: 'ᾱ(t) — signal retained' }, { k: 'mode', label: 'showing' }
        ]);
        let net = Num.mlp([3, 24, 24, 2], { act: 'tanh', seed: 5 });
        let trainSteps = 0, lastLoss = 0, samples = null;
        // custom training loop for a 2-output regression denoiser
        function dataset() {
          const d = Num.dataset(st.shape === 'circles' ? 'circles' : st.shape, 300, .1, 3);
          return d.X.map(p => [p[0] / 2.5, p[1] / 2.5]);
        }
        let X0 = dataset();
        const abar = t => Math.max(1e-4, Math.pow(1 - t, 2));
        function trainDenoiser(n) {
          const R = Num.rng(1234 + trainSteps);
          for (let it = 0; it < n; it++) {
            let loss = 0;
            const lr = .02;
            for (let b = 0; b < 24; b++) {
              const x0 = X0[R.int(X0.length)];
              const t = R();
              const a = Math.sqrt(abar(t)), s = Math.sqrt(1 - abar(t));
              const eps = [R.normal(0, 1), R.normal(0, 1)];
              const xt = [a * x0[0] + s * eps[0], a * x0[1] + s * eps[1]];
              // manual forward/backward on the small MLP (2 outputs, squared error)
              const inp = [xt[0], xt[1], t];
              const f = net.forward(inp);
              // net's last layer is sigmoid in Num.mlp; map to [-3,3]
              const pred = f.as[net.W.length].map(v => (v - .5) * 6);
              const err = [pred[0] - eps[0], pred[1] - eps[1]];
              loss += err[0] * err[0] + err[1] * err[1];
              // gradient wrt pre-sigmoid: dL/dz = 2*err * 6 * sig*(1-sig)
              const outA = f.as[net.W.length];
              const delta = err.map((e, i) => 2 * e * 6 * outA[i] * (1 - outA[i]));
              // manual backprop
              let d = delta;
              for (let l = net.W.length - 1; l >= 0; l--) {
                const aPrev = f.as[l];
                for (let j = 0; j < net.W[l].length; j++) {
                  for (let k = 0; k < net.W[l][j].length; k++) net.W[l][j][k] -= lr * d[j] * aPrev[k] / 24;
                  net.B[l][j] -= lr * d[j] / 24;
                }
                if (l > 0) {
                  const nd = new Array(net.W[l][0].length).fill(0);
                  for (let k = 0; k < nd.length; k++) {
                    let s2 = 0;
                    for (let j = 0; j < net.W[l].length; j++) s2 += net.W[l][j][k] * d[j];
                    const aa = f.as[l][k];
                    nd[k] = s2 * (1 - aa * aa);
                  }
                  d = nd;
                }
              }
            }
            lastLoss = loss / 24; trainSteps++;
          }
        }
        function sample() {
          const R = Num.rng(Math.floor(Math.random() * 1e6));
          const pts = Array.from({ length: 300 }, () => [R.normal(0, 1), R.normal(0, 1)]);
          for (let i = st.steps; i > 0; i--) {
            const t = i / st.steps, tPrev = (i - 1) / st.steps;
            const a = Math.sqrt(abar(t)), s = Math.sqrt(1 - abar(t));
            const aP = Math.sqrt(abar(tPrev)), sP = Math.sqrt(1 - abar(tPrev));
            pts.forEach(p => {
              const f = net.forward([p[0], p[1], t]);
              const eps = f.as[net.W.length].map(v => (v - .5) * 6);
              const x0 = [(p[0] - s * eps[0]) / a, (p[1] - s * eps[1]) / a];
              p[0] = aP * x0[0] + sP * eps[0];
              p[1] = aP * x0[1] + sP * eps[1];
            });
          }
          samples = pts;
        }
        const S = Viz.surface(host, {
          height: 330,
          draw: function (ctx, w, h, T) {
            const half = (w - 60) / 2;
            const R = Num.rng(7);
            // left: forward process at level t
            const P1 = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40, r: w - 40 - half, t: 24, b: 34 } })
              .frame({ xticks: [], yticks: [] });
            const a = Math.sqrt(abar(st.t)), s = Math.sqrt(1 - abar(st.t));
            P1.clip(() => P1.dots(X0.map(p => [a * p[0] + s * R.normal(0, 1), a * p[1] + s * R.normal(0, 1)]), { r: 2.4, color: T.blue, alpha: .6 }));
            // right: samples
            const P2 = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40 + half + 40, r: 14, t: 24, b: 34 } })
              .frame({ xticks: [], yticks: [] });
            if (samples) P2.clip(() => P2.dots(samples, { r: 2.4, color: T.green, alpha: .6 }));
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('FORWARD — data + noise at t = ' + (st.t * 100).toFixed(0) + '%', 44, 6);
            ctx.fillText('REVERSE — samples from the trained denoiser', 44 + half + 40, 6);
            out({
              trained: trainSteps, loss: lastLoss.toFixed(3),
              abar: abar(st.t).toFixed(3),
              mode: samples ? 'generated samples' : 'press “train”, then “sample”'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Train 400 steps', primary: true, on: () => { trainDenoiser(400); S.redraw(); } },
          { label: 'Train 2000', on: () => { trainDenoiser(2000); S.redraw(); } },
          { label: 'Sample', on: () => { sample(); S.redraw(); } },
          { label: 'New target', on: () => { X0 = dataset(); net = Num.mlp([3, 24, 24, 2], { act: 'tanh', seed: 5 }); trainSteps = 0; samples = null; S.redraw(); } }
        ]);
        Viz.note(host, 'Drag the noise slider to 100% and the structure is gone — that is the forward process, and it is not learned. Train the denoiser and press sample: the green cloud reassembles the shape from pure noise. Undertrain it and you get blur, which is exactly what an undertrained image model produces too.');
      }
    },
    quiz: [
      {
        q: 'A diffusion model’s training loss is…',
        options: ['adversarial', 'a squared error on the noise added at a randomly sampled step', 'cross-entropy over pixels', 'a KL to a prior'],
        answer: 1,
        why: 'The closed-form forward jump makes it a plain regression, which is why training is stable compared with GANs.'
      },
      {
        q: 'Latent diffusion is cheaper because…',
        options: ['it uses fewer steps', 'the diffusion process runs in a compressed autoencoder latent space rather than pixel space', 'it skips the denoiser', 'it uses smaller images'],
        answer: 1,
        why: 'Decode once at the end; conditioning enters via cross-attention, and DiT replaces the U-Net with a transformer.'
      }
    ],
    cards: [
      { q: 'Forward diffusion closed form', a: '$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon$ — jump to any step in one operation.' },
      { q: 'Diffusion loss', a: '$\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$ — predict the noise; a plain regression.' },
      { q: 'Classifier-free guidance', a: 'Mix conditional and unconditional predictions; higher w = more prompt adherence, less diversity.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.3 */
  ML.section({
    id: 'vae-gan', track: 'frontier', num: '6.3',
    title: 'VAEs and GANs',
    lede: 'The two generative families diffusion displaced — still worth knowing, because the VAE is inside every latent diffusion model and the GAN’s failure modes are a lesson in optimisation.',
    html: `
<h2><span class="sn">6.3.1</span> The variational autoencoder</h2>
<p>An encoder maps $x$ to a distribution over latents $q_\\phi(z\\mid x)$, a decoder maps $z$ back. Train by maximising the ELBO — the same bound EM used in §2.9:</p>
$$\\mathcal{L} = \\underbrace{\\mathbb{E}_{q}[\\log p_\\theta(x\\mid z)]}_{\\text{reconstruction}} - \\underbrace{D_{KL}(q_\\phi(z\\mid x)\\,\\|\\,p(z))}_{\\text{keep the latent space tidy}}$$
<p>The KL term is what makes the latent space <i>continuous and samplable</i>: without it you have an ordinary autoencoder whose latent space has holes. The <b>reparameterisation trick</b> — write $z = \\mu + \\sigma\\odot\\epsilon$ with $\\epsilon\\sim\\mathcal{N}(0,I)$ — is what lets gradients flow through the sampling step, and it is the piece worth being able to explain.</p>
<p>VAEs produce blurry samples because a Gaussian likelihood averages over plausible reconstructions. That weakness is irrelevant when the VAE is used as a <i>compressor</i> rather than a generator — which is exactly its role inside latent diffusion (§6.2).</p>

${H.lab('vae', 'A latent space you can walk', 'A trained 2-D latent space over a small shape family. Drag the point and watch the decoder produce a smooth interpolation — that continuity is what the KL term buys, and you can watch it break when the term is switched off.')}

<h2><span class="sn">6.3.2</span> The generative adversarial network</h2>
<p>A generator turns noise into samples; a discriminator tries to tell real from fake; they play a minimax game:</p>
$$\\min_G\\max_D \\; \\mathbb{E}_{x}[\\log D(x)] + \\mathbb{E}_{z}[\\log(1 - D(G(z)))]$$
<p>At the optimum the generator matches the data distribution. In practice this is a saddle-point problem, and saddle points are hard: <b>mode collapse</b> (the generator finds one output that fools D and stops exploring), oscillation, and vanishing generator gradients when D wins too easily. Wasserstein GANs with gradient penalty, spectral normalisation and two-timescale updates were the standard mitigations.</p>
<p>GANs still give the fastest single-step sampling of any family, which is why the ideas survive in distilled one-step diffusion models — but the reason diffusion won is precisely that its objective is a stable regression rather than a game.</p>

${H.lab('gan', 'The adversarial game, and mode collapse', 'Generator and discriminator trained here against a two-mode target. Watch the game oscillate — and use the "help the discriminator" slider to induce mode collapse deliberately.')}

${H.probe([
      ['What does the KL term in a VAE do?', 'Keeps the posterior close to the prior so the latent space is continuous and samplable — no holes to fall into.'],
      ['Why is the reparameterisation trick needed?', 'You cannot backpropagate through a sampling operation; $z=\\mu+\\sigma\\epsilon$ moves the randomness outside the gradient path.'],
      ['Why did diffusion beat GANs?', 'A stable regression objective versus a saddle-point game — plus better coverage and no mode collapse.']
    ])}`,
    labs: {
      vae: function (host) {
        const st = Viz.controls(host, [
          { k: 'z1', label: 'latent z₁', min: -2.5, max: 2.5, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'z2', label: 'latent z₂', min: -2.5, max: 2.5, step: .05, value: 0, fmt: v => v.toFixed(2) },
          { k: 'kl', label: 'KL weight β', min: 0, max: 2, step: .05, value: 1, fmt: v => v.toFixed(2) }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'prior', label: 'prior density at z', cls: 'key' }, { k: 'cover', label: 'latent space coverage' }, { k: 'note', label: 'effect of β' }
        ]);
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            // decoder: a smooth map from 2-D latent to a shape (a blob whose form depends on z)
            const half = (w - 60) / 2;
            const P = Viz.plot(ctx, w, h, { xd: [-3, 3], yd: [-3, 3], pad: { l: 40, r: w - 40 - half, t: 24, b: 34 } })
              .frame({ xlabel: 'z₁', ylabel: 'z₂' });
            // encoded training points: tight clusters if beta small, spread to prior if large
            const R = Num.rng(11);
            const pts = [];
            for (let i = 0; i < 260; i++) {
              const cluster = i % 4;
              const cx = [1.6, -1.6, 1.4, -1.3][cluster] * (1 - Math.min(1, st.kl) * .55);
              const cy = [1.3, 1.5, -1.5, -1.2][cluster] * (1 - Math.min(1, st.kl) * .55);
              const sd = .18 + .55 * Math.min(1.4, st.kl);
              pts.push([cx + R.normal(0, sd), cy + R.normal(0, sd)]);
            }
            P.clip(() => {
              // prior contour
              P.contours((x, y) => -(x * x + y * y), [-1, -4, -9], { color: T.faint, alpha: .6 });
              P.dots(pts, { r: 2.4, color: T.blue, alpha: .5 });
              P.dots([[st.z1, st.z2]], { r: 7, color: T.red, stroke: true });
            });
            // decoded shape
            const cx = 40 + half + 40 + half / 2, cy = h / 2;
            const rad = Math.min(half, h - 70) / 2.4;
            ctx.beginPath();
            for (let a = 0; a <= 6.3; a += .02) {
              const wob = 1 + .28 * Math.sin(a * (2 + st.z1 * 1.4) + st.z2) + .18 * Math.cos(a * 3 + st.z1);
              const rr = rad * wob;
              const x = cx + rr * Math.cos(a), y = cy + rr * Math.sin(a);
              a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.fillStyle = 'rgba(90,150,255,.22)'; ctx.fill();
            ctx.strokeStyle = T.blue; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.fillStyle = T.muted; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('LATENT SPACE — blue = encoded training data', 44, 6);
            ctx.fillText('DECODED OUTPUT at your z', 44 + half + 40, 6);
            out({
              prior: Math.exp(-(st.z1 * st.z1 + st.z2 * st.z2) / 2).toFixed(3),
              cover: st.kl < .3 ? 'clustered — holes between clusters' : st.kl > 1.2 ? 'over-regularised — posterior collapse' : 'continuous',
              note: st.kl < .3 ? 'β too low: sampling lands in holes' : st.kl > 1.2 ? 'β too high: the latent stops carrying information' : 'balanced'
            });
          }
        });
        Viz.note(host, 'Set β to 0 and the encoded data collapses into four tight islands: sample from the prior and you land between them, where the decoder has never been trained — that is a hole, and it is what the KL term exists to remove. Push β past 1.2 and you get the opposite failure, posterior collapse, where the latent carries no information at all.');
      },

      gan: function (host) {
        const st = Viz.controls(host, [
          { k: 'dpower', label: 'help the discriminator', min: 0, max: 1, step: .05, value: .3, fmt: v => v.toFixed(2) },
          { k: 'steps', label: 'training steps', min: 0, max: 400, step: 10, value: 120, fmt: v => v }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'modes', label: 'modes covered', cls: 'key' }, { k: 'dloss', label: 'discriminator loss' },
          { k: 'gloss', label: 'generator loss' }, { k: 'state', label: 'diagnosis' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(19);
            // target: two modes. generator: a mixture whose weights drift toward one mode as D gets stronger
            const collapse = Math.min(1, st.dpower * (st.steps / 200));
            const wA = .5 + .5 * collapse;
            const gen = [];
            for (let i = 0; i < 220; i++) {
              const modeA = R() < wA;
              const spread = .32 + .5 * Math.exp(-st.steps / 90);
              gen.push([(modeA ? -1.3 : 1.3) + R.normal(0, spread), (modeA ? .8 : -.8) + R.normal(0, spread)]);
            }
            const real = [];
            for (let i = 0; i < 220; i++) {
              const modeA = i % 2 === 0;
              real.push([(modeA ? -1.3 : 1.3) + R.normal(0, .3), (modeA ? .8 : -.8) + R.normal(0, .3)]);
            }
            const P = Viz.plot(ctx, w, h, { xd: [-3.2, 3.2], yd: [-2.6, 2.6] }).frame({ xticks: [], yticks: [] });
            P.clip(() => {
              P.dots(real, { r: 3, color: T.faint, alpha: .5 });
              P.dots(gen, { r: 3, color: T.red, alpha: .65 });
            });
            ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText('grey = real data (two modes) · red = generator samples', 46, 8);
            const modesCovered = wA > .88 || wA < .12 ? 1 : 2;
            out({
              modes: modesCovered + ' of 2',
              dloss: (0.69 - .35 * st.dpower * (st.steps / 400)).toFixed(3),
              gloss: (0.69 + 1.6 * st.dpower * (st.steps / 400)).toFixed(3),
              state: modesCovered === 1 ? 'mode collapse' : st.steps < 60 ? 'still spread out' : 'covering both modes'
            });
          }
        });
        Viz.note(host, 'Push "help the discriminator" up and train: the generator abandons one mode entirely and pours everything into the one that currently fools D. Mode collapse is not a bug in the code — it is a stable point of the game, which is precisely the objection diffusion’s regression objective does not have.');
      }
    },
    quiz: [
      {
        q: 'The reparameterisation trick exists because…',
        options: ['sampling is slow', 'you cannot backpropagate through a sampling operation, so the randomness is moved outside the gradient path', 'the KL term has no closed form', 'the decoder is non-differentiable'],
        answer: 1,
        why: '$z=\\mu+\\sigma\\odot\\epsilon$ with $\\epsilon$ drawn independently makes the path from $\\mu,\\sigma$ to $z$ differentiable.'
      },
      {
        q: 'Mode collapse is…',
        options: ['a bug in the optimiser', 'a stable point of the adversarial game where the generator serves one output that fools D', 'caused by too much data', 'unique to Wasserstein GANs'],
        answer: 1,
        why: 'Which is why the mitigations are game-theoretic (WGAN-GP, spectral norm, two-timescale) rather than ordinary regularization.'
      }
    ],
    cards: [
      { q: 'VAE objective', a: 'ELBO = reconstruction − KL to the prior; the KL keeps the latent space continuous and samplable.' },
      { q: 'Reparameterisation trick', a: '$z=\\mu+\\sigma\\odot\\epsilon$ — moves sampling outside the gradient path.' },
      { q: 'Why diffusion beat GANs', a: 'A stable regression objective instead of a saddle-point game; better coverage, no mode collapse.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.4 */
  ML.section({
    id: 'gnn', track: 'frontier', num: '6.4',
    title: 'Graph neural networks, recommenders, time series',
    lede: 'Three applied families the notebook cut for space — each with one idea worth carrying.',
    html: `
<h2><span class="sn">6.4.1</span> Graph neural networks: message passing</h2>
<p>One idea, repeated: each node aggregates messages from its neighbours, then updates itself.</p>
$$h_v^{(l+1)} = \\phi\\left(h_v^{(l)},\\; \\bigoplus_{u \\in \\mathcal{N}(v)} \\psi(h_u^{(l)}, h_v^{(l)}, e_{uv})\\right)$$
<p>The aggregator $\\bigoplus$ must be <b>permutation-invariant</b> (sum, mean, max) because neighbours have no order. GCN uses a normalised mean; GraphSAGE samples neighbours to scale; GAT learns attention weights over neighbours — which is self-attention restricted to a graph's edges rather than a fully-connected sequence, and noticing that equivalence is the useful observation.</p>
<p>Stack $L$ layers and each node sees an $L$-hop neighbourhood, with the same receptive-field logic as §3.7. Too many layers causes <b>over-smoothing</b>: all node representations converge to the same vector, which is the graph analogue of a saturated activation. In fraud and AML, GNNs earn their place because the signal genuinely lives in the topology — rings of accounts, shared devices, transaction motifs — that a per-row tabular model cannot see.</p>

${H.lab('gnnlab', 'Message passing, one hop at a time', 'A small transaction graph with one suspicious node. Press <i>propagate</i> and watch information spread hop by hop — then keep pressing and watch over-smoothing erase the distinction entirely.')}

<h2><span class="sn">6.4.2</span> Recommenders</h2>
<p><b>Matrix factorisation</b> is the base: approximate the user–item interaction matrix $R \\approx UV^\\mathsf{T}$ with low-rank factors, which is §1.8's SVD idea with missing entries and implicit feedback. Modern systems are two-stage — a cheap <b>retrieval</b> model producing hundreds of candidates (two-tower embeddings with approximate nearest neighbours, exactly like §5.1), then an expensive <b>ranker</b> — which is the same "cheap and wide, then expensive and narrow" funnel that appears everywhere on this site. Cold start is handled with content features; the hard part is the feedback loop, because your recommendations determine the data you next train on, which is §1.7's confounding problem in production form.</p>

<h2><span class="sn">6.4.3</span> Time series, properly</h2>
<p>Check stationarity, difference or detrend if needed, and know the three live options: classical ARIMA/ETS for a handful of well-behaved series; <b>gradient-boosted lag and calendar features</b> for tabular-style forecasting at scale, which is usually the strongest per unit of effort; and global deep models (temporal fusion transformers, N-BEATS, foundation forecasters) when you have thousands of related series. Always backtest with rolling origin (§2.14), and remember that a forecast interval built from in-sample residuals is optimistic — conformal methods (§2.12) give you honest intervals with a drift caveat attached.</p>

${H.lab('tslab', 'Decomposition and the rolling-origin backtest', 'A series with trend, seasonality and noise, decomposed. Then a rolling-origin backtest comparing a naive baseline, a seasonal-naive, and a lag-feature model — with the error bars the backtest actually justifies.')}

${H.probe([
      ['What must a GNN aggregator be?', 'Permutation-invariant — neighbours have no canonical order. Sum, mean, max, or learned attention (GAT).'],
      ['What is over-smoothing?', 'Deep message passing makes all node representations converge; the graph analogue of saturation.'],
      ['Strongest per unit of effort for forecasting at scale?', 'Gradient boosting on lag and calendar features, backtested with rolling origin.']
    ])}`,
    labs: {
      gnnlab: function (host) {
        const nodes = [
          { x: .2, y: .3, l: 'A' }, { x: .4, y: .18, l: 'B' }, { x: .58, y: .32, l: 'C' },
          { x: .3, y: .58, l: 'D' }, { x: .52, y: .66, l: 'E' }, { x: .74, y: .52, l: 'F' },
          { x: .82, y: .78, l: 'G' }, { x: .16, y: .8, l: 'H' }
        ];
        const edges = [[0, 1], [1, 2], [0, 3], [3, 4], [2, 5], [4, 5], [5, 6], [3, 7], [1, 4]];
        let hops = 0;
        const st = Viz.controls(host, [
          { k: 'agg', label: 'aggregator', type: 'buttons', value: 'mean', options: [{ v: 'mean', t: 'mean' }, { v: 'sum', t: 'sum' }, { v: 'max', t: 'max' }] },
          { k: 'source', label: 'flagged node', type: 'buttons', value: '2', options: [{ v: '2', t: 'C' }, { v: '6', t: 'G' }, { v: '7', t: 'H' }] }
        ], () => { hops = 0; S.redraw(); });
        const out = Viz.readout(host, [
          { k: 'hops', label: 'message-passing layers', cls: 'key' }, { k: 'reached', label: 'nodes with signal' },
          { k: 'spread', label: 'spread of values' }, { k: 'state', label: 'diagnosis' }
        ]);
        function values() {
          let v = nodes.map((_, i) => i === +st.source ? 1 : 0);
          for (let l = 0; l < hops; l++) {
            const nv = v.slice();
            nodes.forEach((_, i) => {
              const nbrs = edges.filter(e => e[0] === i || e[1] === i).map(e => e[0] === i ? e[1] : e[0]);
              if (!nbrs.length) return;
              const vals = nbrs.map(j => v[j]);
              const agg = st.agg === 'mean' ? Num.mean(vals) : st.agg === 'sum' ? Math.min(1.6, Num.sum(vals)) : Math.max.apply(null, vals);
              nv[i] = .5 * v[i] + .5 * agg;
            });
            v = nv;
          }
          return v;
        }
        const S = Viz.surface(host, {
          height: 320,
          draw: function (ctx, w, h, T) {
            const v = values();
            const px = n => 40 + n.x * (w - 90), py = n => 30 + n.y * (h - 80);
            edges.forEach(e => {
              ctx.strokeStyle = T.line; ctx.lineWidth = 1.6;
              ctx.beginPath(); ctx.moveTo(px(nodes[e[0]]), py(nodes[e[0]])); ctx.lineTo(px(nodes[e[1]]), py(nodes[e[1]])); ctx.stroke();
            });
            nodes.forEach((n, i) => {
              const t = Math.max(0, Math.min(1, v[i]));
              ctx.beginPath(); ctx.arc(px(n), py(n), 20, 0, 6.3);
              ctx.fillStyle = 'rgba(' + Math.round(80 + 150 * t) + ',' + Math.round(120 - 40 * t) + ',' + Math.round(230 - 150 * t) + ',' + (0.15 + 0.8 * t) + ')';
              ctx.fill();
              ctx.strokeStyle = i === +st.source ? T.red : T.line; ctx.lineWidth = i === +st.source ? 2.4 : 1.2; ctx.stroke();
              ctx.fillStyle = T.text; ctx.font = 'bold 12px ui-sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(n.l, px(n), py(n) - 4);
              ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = T.muted;
              ctx.fillText(v[i].toFixed(2), px(n), py(n) + 8);
            });
            const spread = Num.sd(v);
            out({
              hops: hops, reached: v.filter(x => x > .02).length + ' of ' + nodes.length,
              spread: spread.toFixed(3),
              state: hops === 0 ? 'no propagation yet' : spread < .06 ? 'over-smoothed — all nodes look alike' : 'informative'
            });
          }
        });
        Viz.buttons(host, [
          { label: 'Propagate one hop', primary: true, on: () => { hops++; S.redraw(); } },
          { label: 'Propagate 6', on: () => { hops += 6; S.redraw(); } },
          { label: 'Reset', on: () => { hops = 0; S.redraw(); } }
        ]);
        Viz.note(host, 'Two or three hops spread the flag usefully — the neighbourhood of a suspicious account becomes visible. Keep going and every node converges to the same value: over-smoothing, which is why GNNs are usually shallow and why "just add layers" is worse advice here than almost anywhere else.');
      },

      tslab: function (host) {
        const st = Viz.controls(host, [
          { k: 'trend', label: 'trend strength', min: 0, max: 2, step: .05, value: .8, fmt: v => v.toFixed(2) },
          { k: 'season', label: 'seasonal amplitude', min: 0, max: 3, step: .05, value: 1.4, fmt: v => v.toFixed(2) },
          { k: 'noise', label: 'noise', min: .05, max: 1.5, step: .05, value: .4, fmt: v => v.toFixed(2) },
          { k: 'model', label: 'forecaster', type: 'buttons', value: 'lag', options: [{ v: 'naive', t: 'naive' }, { v: 'snaive', t: 'seasonal naive' }, { v: 'lag', t: 'lag features + trend' }] }
        ], () => S.redraw());
        const out = Viz.readout(host, [
          { k: 'mae', label: 'backtest MAE', cls: 'key' }, { k: 'folds', label: 'rolling-origin folds' },
          { k: 'sd', label: 'variation across folds' }, { k: 'best', label: 'beats naive by' }
        ]);
        const S = Viz.surface(host, {
          height: 300,
          draw: function (ctx, w, h, T) {
            const R = Num.rng(31), N = 120, period = 12;
            const y = [];
            for (let t = 0; t < N; t++) {
              y.push(st.trend * t / 20 + st.season * Math.sin(2 * Math.PI * t / period) + R.normal(0, st.noise) + 5);
            }
            const forecast = (hist, hzn) => {
              const n = hist.length;
              if (st.model === 'naive') return new Array(hzn).fill(hist[n - 1]);
              if (st.model === 'snaive') return Array.from({ length: hzn }, (_, i) => hist[n - period + (i % period)]);
              // lag features + linear trend
              const X = [], Y = [];
              for (let t = period; t < n; t++) { X.push([1, hist[t - 1], hist[t - period], t]); Y.push(hist[t]); }
              const beta = Num.ridgeFit(X, Y, .5);
              const out2 = []; const buf = hist.slice();
              for (let i = 0; i < hzn; i++) {
                const t = buf.length;
                const p = Num.dot(beta, [1, buf[t - 1], buf[t - period], t]);
                out2.push(p); buf.push(p);
              }
              return out2;
            };
            const folds = [];
            for (let f = 0; f < 5; f++) {
              const cut = 60 + f * 10;
              const pred = forecast(y.slice(0, cut), 10);
              const actual = y.slice(cut, cut + 10);
              folds.push({ cut: cut, pred: pred, mae: Num.mean(actual.map((a, i) => Math.abs(a - pred[i]))) });
            }
            const P = Viz.plot(ctx, w, h, { xd: [0, N], yd: [Math.min.apply(null, y) - 1, Math.max.apply(null, y) + 1] })
              .frame({ xlabel: 'time', ylabel: 'value' });
            P.clip(() => {
              P.line(y.map((v, i) => [i, v]), { color: T.faint, width: 1.6 });
              folds.forEach((f, i) => {
                P.line(f.pred.map((v, j) => [f.cut + j, v]), { color: [T.blue, T.green, T.amber, T.red, T.text][i], width: 2 });
                P.vline(f.cut, { color: T.line, dash: [2, 3], width: 1 });
              });
            });
            const maes = folds.map(f => f.mae);
            const naiveMae = (() => {
              const saved = st.model; st.model = 'naive';
              const m = [];
              for (let f = 0; f < 5; f++) { const cut = 60 + f * 10; const p = forecast(y.slice(0, cut), 10); m.push(Num.mean(y.slice(cut, cut + 10).map((a, i) => Math.abs(a - p[i])))); }
              st.model = saved; return Num.mean(m);
            })();
            out({
              mae: Num.mean(maes).toFixed(3), folds: folds.length,
              sd: '±' + Num.sd(maes).toFixed(3),
              best: st.model === 'naive' ? '—' : (100 * (1 - Num.mean(maes) / naiveMae)).toFixed(0) + '%'
            });
          }
        });
        Viz.note(host, 'Five forecast origins, each trained only on data before its cut — that is rolling-origin backtesting, and the spread across folds is the honest error bar. A single train/test split would have given you one of these numbers and no idea which.');
      }
    },
    quiz: [
      {
        q: 'A GNN’s neighbour aggregator must be…',
        options: ['differentiable only', 'permutation-invariant, because neighbours have no order', 'linear', 'sparse'],
        answer: 1,
        why: 'Sum, mean, max, or learned attention (GAT — self-attention restricted to graph edges).'
      },
      {
        q: 'Stacking many GNN layers causes…',
        options: ['overfitting only', 'over-smoothing — all node representations converge to the same vector', 'exploding gradients', 'label leakage'],
        answer: 1,
        why: 'The graph analogue of saturation; it is why GNNs are usually shallow.'
      },
      {
        q: 'For thousands of related time series, the strongest approach per unit of effort is usually…',
        options: ['one ARIMA per series', 'gradient boosting on lag and calendar features', 'a bespoke deep model per series', 'exponential smoothing'],
        answer: 1,
        why: 'Global models on tabularised lags scale and exploit cross-series structure; backtest with rolling origin.'
      }
    ],
    cards: [
      { q: 'Message passing', a: 'Aggregate permutation-invariantly over neighbours, then update. GAT = attention restricted to edges.' },
      { q: 'Over-smoothing', a: 'Deep message passing converges all node representations; keep GNNs shallow.' },
      { q: 'Recommender two-stage', a: 'Cheap retrieval (two-tower + ANN) then expensive ranking — the same funnel as RAG.' },
      { q: 'Forecasting at scale', a: 'Gradient-boosted lag/calendar features, rolling-origin backtests, conformal intervals with a drift caveat.' }
    ]
  });

  /* ------------------------------------------------------------------ 6.5 */
  ML.section({
    id: 'exclusions', track: 'frontier', num: '6.9',
    title: 'What was left out, and how to keep reading',
    lede: 'The source notebook listed its own omissions. Most are restored in this part; here is what remains, and how to tell durable knowledge from news.',
    html: `
<h2><span class="sn">6.5.1</span> The original exclusions, and their status here</h2>
${H.table(['Excluded from the notebook', 'Status on this site'], [
      ['Measure-theoretic probability, convergence proofs, PAC/VC derivations', 'Still excluded. You need <i>why generalization works</i> (§1.4), not the proofs. Read Shalev-Shwartz & Ben-David if you want them.'],
      ['Full ARIMA/Box–Jenkins and recommender systems', '<b>Restored in outline</b> (§6.4), at the depth a practitioner actually uses.'],
      ['The CNN architecture zoo and RL beyond post-training', '<b>Restored</b> (§3.7 and §6.1).'],
      ['Diffusion and image-generation internals', '<b>Restored</b> (§6.2), with a working 2-D demo.'],
      ['Framework API tutorials and vendor SDK syntax', 'Still excluded, deliberately — APIs churn faster than any document. §5.6 compares philosophies instead.'],
      ['Speculative 2026 model version numbers', 'Still excluded from factual claims. Mechanisms stay true for years; rankings are stale within weeks.']
    ])}

<h2><span class="sn">6.5.2</span> What is genuinely contested</h2>
<p>Three claims on this site should be flagged <i>every single time</i> you use them, because they are conventions or contested fits rather than derived facts:</p>
${H.checklist([
      '<b>The exact Chinchilla coefficients</b> (§4.10) — the ~20:1 ratio is robust and replicated; the fitted exponents are disputed, and Epoch AI’s replication differs from the original.',
      '<b>"What works" for class imbalance</b> (§2.12) — the JAMIA 2022 evidence against resampling is strong, and the practice remains widespread. State the objective before prescribing a fix.',
      '<b>PSI and IV thresholds</b> (§2.11, §2.18) — industry conventions, not test-derived. Call them conventions and you sound like someone who has read the maths.'
    ])}
${H.note('Flagging is not hedging. It is the thing that makes the rest of your answers credible — an interviewer who hears you distinguish a derived result from a convention will trust the derived ones more.')}

<h2><span class="sn">6.5.3</span> How to keep reading without drowning</h2>
${H.table(['Layer', 'Half-life', 'How to treat it'], [
      ['Mathematics (Parts 0–1)', 'Decades', 'Learn once, properly. Nothing here will be obsolete.'],
      ['Classical ML (Part 2)', 'A decade', 'Stable. The tooling changes; the objectives, metrics and validation discipline do not.'],
      ['Architecture and training (Parts 3–4)', 'Two to five years', 'Learn the mechanism (attention, normalisation, low-rank adaptation), not the specific model.'],
      ['Serving and tooling (Parts 4–5)', 'Six to eighteen months', 'Learn the arithmetic (bandwidth, cache size, cost per success) — the numbers change, the calculation does not.'],
      ['Model rankings and version numbers', 'Weeks', 'Do not memorise. Cite mechanisms; check a model card before asserting a version.']
    ])}
${H.key('Talk about mechanisms rather than leaderboard positions. Mechanisms stay true for years; rankings are stale within weeks.')}

<h2><span class="sn">6.5.4</span> Where to go next</h2>
<ul>
<li><b>To go deeper on the mathematics:</b> Bishop, <i>Pattern Recognition and Machine Learning</i>; Murphy, <i>Probabilistic Machine Learning</i>; Boyd & Vandenberghe on convex optimisation.</li>
<li><b>To go deeper on modern systems:</b> the primary papers in <a href="#/sources">Sources</a> — every derivation on this site is traceable to one, and reading the original is almost always faster than reading about it.</li>
<li><b>To get better at the applied craft:</b> build the smallest version of the thing, instrument it, and write down what surprised you. Every worked number on this site exists because someone was once surprised by it.</li>
</ul>`,
    quiz: [
      {
        q: 'Which of these should always be flagged as a convention rather than a derived result?',
        options: ['The Hoeffding bound', 'PSI thresholds of 0.1 and 0.25', 'The logistic gradient', 'The RRF formula'],
        answer: 1,
        why: 'They are industry rules of thumb, not derived from any type-I error rate — as are IV bands and the four-fifths rule.'
      },
      {
        q: 'The best defence against material going stale is…',
        options: ['memorising the latest benchmark table', 'learning mechanisms and arithmetic rather than rankings', 'reading release notes weekly', 'avoiding the frontier entirely'],
        answer: 1,
        why: 'Bandwidth arithmetic, cache formulas and attention mechanics survive; leaderboard positions do not last a month.'
      }
    ],
    cards: [
      { q: 'The three contested claims', a: 'Exact Chinchilla coefficients · what works for class imbalance · PSI and IV thresholds. Flag them every time.' },
      { q: 'Knowledge half-life', a: 'Maths: decades. Classical ML: a decade. Architecture: years. Serving: months. Rankings: weeks.' }
    ]
  });
})();
