/* Interactive behaviour for the credit project page.
 *
 * Everything here reads `data/credit.json`, which is exported from the repo by
 * `tools/export_site_data.py`. The threshold explorer recomputes the confusion
 * matrix and expected cost in the browser — that arithmetic is genuinely live —
 * but the probabilities it operates on were produced offline by the real model.
 * The page says so rather than implying a model is running client-side.
 */

import { loadJSON, fmt, showError, onResize } from "./site.js";
import { lineChart, divergingBars, register, escapeHtml, colour } from "./charts.js";

const COST_FN = 5;
const COST_FP = 1;

let data;

function set(key, value) {
  for (const node of document.querySelectorAll(`[data-k="${key}"]`)) node.textContent = value;
}

/* --- headline ------------------------------------------------------------ */

function renderHeadline() {
  const h = data.headline;
  const [lo, hi] = h.roc_auc_ci95 || [NaN, NaN];

  set("auc", fmt.num(h.roc_auc, 3));
  set("auc-ci", `95% CI ${fmt.num(lo, 2)}–${fmt.num(hi, 2)}`);
  set("auc-ci-inline", `[${fmt.num(lo, 2)}, ${fmt.num(hi, 2)}]`);
  set("threshold", fmt.pct(data.model.threshold, 1));
  set("cost", fmt.num(data.at_threshold.cost_per_applicant, 3));
  set("cost-sub", `vs ${fmt.num(h.baseline_decline_all_cost_per_applicant ?? 0.7, 2)} declining everyone`);
  set("brier", fmt.num(h.brier, 3));
  set("calmethod", data.model.calibration ?? "—");
  set("logit-auc", fmt.num(data.baseline_logistic?.roc_auc, 3));

  const best = data.cost_curve.reduce((a, b) =>
    b.cost_per_applicant < a.cost_per_applicant ? b : a);
  set("threshold-gap",
    fmt.num(data.at_threshold.cost_per_applicant - best.cost_per_applicant, 3));

  const r = data.retrieval || {};
  set("rag-model", fmt.num(r.model_auc, 3));
  set("rag-blend", fmt.num(r.model_plus_neighbours_auc, 3));
  set("rag-neighbour", fmt.num(r.neighbour_bad_rate_auc, 3));
  set("rag-self", fmt.int(r.self_retrieval_count ?? 0));
  set("build.stamp", `generated from commit ${data.generated_from_commit ?? "—"}`);
}

/* --- threshold explorer -------------------------------------------------- */

/* Recomputed from the 250 real probabilities on every slider move. We predict
   `is_bad`, so probability >= threshold means decline; a false negative is an
   approved application that went bad — the error weighted 5x. */
function confusion(threshold) {
  let fn = 0, fp = 0, approved = 0;
  for (const a of data.applicants) {
    const declined = a.probability >= threshold;
    if (!declined) approved += 1;
    if (!declined && a.actual_bad === 1) fn += 1;
    if (declined && a.actual_bad === 0) fp += 1;
  }
  const n = data.applicants.length;
  return {
    fn, fp, n,
    approvalRate: approved / n,
    costPerApplicant: (fn * COST_FN + fp * COST_FP) / n,
  };
}

function renderThreshold(threshold) {
  const c = confusion(threshold);
  const declineAll = data.headline.baseline_decline_all_cost_per_applicant ?? 0.7;

  document.getElementById("thr-val").textContent = threshold.toFixed(3);
  document.getElementById("live-cost").textContent = fmt.num(c.costPerApplicant, 3);
  document.getElementById("live-approval").textContent = fmt.pct(c.approvalRate, 1);
  document.getElementById("live-fn").textContent = c.fn;
  document.getElementById("live-fp").textContent = c.fp;

  const versus = document.getElementById("live-cost-vs");
  const delta = c.costPerApplicant - declineAll;
  versus.textContent = delta <= 0
    ? `${fmt.num(-delta, 3)} better than declining everyone`
    : `${fmt.num(delta, 3)} WORSE than declining everyone`;
  versus.style.color = delta <= 0 ? "var(--good)" : "var(--bad)";

  drawCostCurve(threshold);
}

let currentThreshold = 0.21;

function drawCostCurve(threshold) {
  const container = document.getElementById("cost-curve");
  if (!container) return;
  const declineAll = data.headline.baseline_decline_all_cost_per_applicant ?? 0.7;

  const points = data.cost_curve.map((r) => [r.threshold, r.cost_per_applicant]);
  const flat = data.cost_curve.map((r) => [r.threshold, declineAll]);
  const here = confusion(threshold);

  const chart = lineChart(container, {
    height: 260,
    series: [
      { points: flat, colour: colour.text3, width: 1.2, dash: "4 4" },
      { points, colour: colour.accent, width: 2 },
    ],
    xLabel: "decline threshold",
    yLabel: "cost per applicant",
    yFormat: (v) => v.toFixed(2),
    xFormat: (v) => v.toFixed(1),
  });

  if (chart) {
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", chart.x(threshold));
    marker.setAttribute("cy", chart.y(here.costPerApplicant));
    marker.setAttribute("r", 5);
    marker.setAttribute("fill", colour.bad);
    chart.g.appendChild(marker);
  }
}

function initThreshold() {
  const slider = document.getElementById("thr");
  const shipped = data.model.threshold;
  slider.value = Math.round(shipped * 100);
  currentThreshold = shipped;

  slider.addEventListener("input", () => {
    currentThreshold = Number(slider.value) / 100;
    renderThreshold(currentThreshold);
  });

  const presets = {
    shipped: () => data.model.threshold,
    half: () => 0.5,
    theoretical: () => data.model.theoretical_threshold,
    optimal: () => data.cost_curve.reduce((a, b) =>
      b.cost_per_applicant < a.cost_per_applicant ? b : a).threshold,
  };
  for (const button of document.querySelectorAll("[data-set-thr]")) {
    button.addEventListener("click", () => {
      currentThreshold = presets[button.dataset.setThr]();
      slider.value = Math.round(currentThreshold * 100);
      renderThreshold(currentThreshold);
    });
  }

  renderThreshold(shipped);
  register(() => drawCostCurve(currentThreshold));
}

/* --- applicant explorer -------------------------------------------------- */

const FILTERS = {
  all: () => true,
  "declined-bad": (a, t) => a.probability >= t && a.actual_bad === 1,
  "approved-bad": (a, t) => a.probability < t && a.actual_bad === 1,
  "declined-good": (a, t) => a.probability >= t && a.actual_bad === 0,
  "approved-good": (a, t) => a.probability < t && a.actual_bad === 0,
  "confident-wrong": () => true,
};

function applicantOptions() {
  const filter = document.getElementById("applicant-filter").value;
  const t = data.model.threshold;
  let list = data.applicants.filter((a) => FILTERS[filter](a, t));

  if (filter === "confident-wrong") {
    // Largest gap between what the model said and what happened.
    list = [...data.applicants]
      .sort((a, b) =>
        Math.abs(b.probability - b.actual_bad) - Math.abs(a.probability - a.actual_bad))
      .slice(0, 25);
  }

  const select = document.getElementById("applicant");
  select.replaceChildren(...list.map((a) => {
    const option = document.createElement("option");
    option.value = a.id;
    option.textContent = `${a.id} — P(default) ${(a.probability * 100).toFixed(1)}% — ` +
      (a.actual_bad ? "defaulted" : "repaid");
    return option;
  }));

  if (list.length) renderApplicant(list[0].id);
  else select.replaceChildren(new Option("no applicants match", ""));
}

function renderApplicant(id) {
  const a = data.applicants.find((x) => x.id === id);
  if (!a) return;

  const declined = a.probability >= data.model.threshold;
  document.getElementById("a-prob").textContent = fmt.pct(a.probability, 1);

  const decision = document.getElementById("a-decision");
  decision.textContent = declined ? "DECLINE" : "APPROVE";
  decision.style.color = declined ? "var(--bad)" : "var(--good)";

  const actual = document.getElementById("a-actual");
  actual.textContent = a.actual_bad ? "defaulted" : "repaid";
  actual.style.color = a.actual_bad ? "var(--bad)" : "var(--good)";

  const correct = declined === Boolean(a.actual_bad);
  const verdict = document.getElementById("a-verdict");
  verdict.textContent = correct ? "correct" : (declined ? "lost business" : "lost money");
  verdict.style.color = correct ? "var(--good)" : "var(--bad)";

  renderAttributes(a);
  renderShap(a);
  renderReasons(a, declined);
  renderPrecedents(a);
}

function renderAttributes(a) {
  const body = document.querySelector("#a-attributes tbody");
  const entries = Object.entries(a.attributes);
  const half = Math.ceil(entries.length / 2);
  const rows = [];

  for (let i = 0; i < half; i += 1) {
    const [k1, v1] = entries[i];
    const pair = entries[i + half];
    const protectedLeft = data.protected_features.includes(k1);
    const protectedRight = pair && data.protected_features.includes(pair[0]);
    rows.push(`<tr>
      <td class="muted xs">${escapeHtml(k1)}${protectedLeft ? ' <span class="pill">protected</span>' : ""}</td>
      <td class="wrapline">${escapeHtml(v1)}</td>
      <td class="muted xs">${pair ? escapeHtml(pair[0]) + (protectedRight ? ' <span class="pill">protected</span>' : "") : ""}</td>
      <td class="wrapline">${pair ? escapeHtml(pair[1]) : ""}</td>
    </tr>`);
  }
  body.innerHTML = rows.join("");
}

function renderShap(a) {
  const showProtected = document.getElementById("show-protected").checked;
  const items = a.shap.filter((c) => showProtected || !c.protected).slice(0, 8);

  divergingBars(document.getElementById("a-shap"), {
    rows: items.map((c) => ({
      label: c.label,
      detail: c.value,
      value: c.contribution,
    })),
  });

  const hidden = a.shap.filter((c) => c.protected).length;
  document.getElementById("a-shap-note").textContent = showProtected
    ? "Red pushes towards decline, green towards approval. The model may lawfully use age; the explanation shown to an officer leads with it only when you ask."
    : `Red pushes towards decline, green towards approval. ${hidden} protected attributes hidden — the model uses them, the officer-facing explanation does not lead with them.`;
}

function renderReasons(a, declined) {
  const container = document.getElementById("a-reasons");
  if (!declined) {
    container.innerHTML = `<div class="callout">This application was approved. An approval is
      not an adverse action, so no reasons are generated — producing them anyway invites their
      being shown to the applicant as though they were warnings.</div>`;
    return;
  }
  if (!a.reason_codes.length) {
    container.innerHTML = `<div class="callout">No factor cleared the noise threshold. One
      honest reason beats four, three of which are rounding error.</div>`;
    return;
  }
  container.innerHTML = `<div class="callout">
    <p class="mb1"><strong>Your application was declined. The principal reasons were:</strong></p>
    <ol style="margin:0 0 .8rem;padding-left:1.2rem">
      ${a.reason_codes.map((r) => `<li>${escapeHtml(r.statement)}</li>`).join("")}
    </ol>
    <p class="xs mb0">Bounded, ranked, each quoting the applicant's own value, never citing a
    protected attribute. A different artifact from the table above, not a reformatting of it.</p>
  </div>`;
}

function renderPrecedents(a) {
  const container = document.getElementById("a-precedents");
  if (!a.precedents.length) {
    container.innerHTML = `<p class="muted small">No precedents exported for this applicant.</p>`;
    return;
  }
  const bad = a.precedents.filter((p) => p.bad).length;
  container.innerHTML = `
    <p class="small mb1"><b>${bad} of ${a.precedents.length}</b> of the closest historical
    applicants defaulted.</p>
    ${a.precedents.map((p) => `
      <details class="card mb1" style="padding:.7rem .9rem">
        <summary style="cursor:pointer;font-size:.85rem">
          <span class="pill ${p.bad ? "bad" : "good"}">${p.bad ? "defaulted" : "repaid"}</span>
          <span class="mono xs muted" style="margin-left:.4rem">${escapeHtml(p.id)}</span>
          <span class="xs muted" style="margin-left:.4rem">distance ${p.distance.toFixed(3)}</span>
        </summary>
        <p class="xs muted mt1 mb0" style="white-space:normal">${escapeHtml(p.text)}</p>
      </details>`).join("")}`;
}

function initApplicants() {
  document.getElementById("applicant").addEventListener("change", (e) =>
    renderApplicant(e.target.value));
  document.getElementById("applicant-filter").addEventListener("change", applicantOptions);
  document.getElementById("show-protected").addEventListener("change", () =>
    renderApplicant(document.getElementById("applicant").value));
  applicantOptions();
}

/* --- fairness ------------------------------------------------------------ */

function renderFairness(attribute) {
  const block = data.fairness.attributes[attribute];
  const body = document.querySelector("#fair-table tbody");

  body.innerHTML = block.groups.map((g) => `<tr>
    <td class="wrapline">${escapeHtml(g.group)}${g.small ? ' <span class="pill">small</span>' : ""}</td>
    <td class="num">${g.n}</td>
    <td class="num">${fmt.pct(g.approval_rate, 1)}</td>
    <td class="num">${fmt.pct(g.bad_rate, 1)}</td>
    <td class="num">${fmt.pct(g.approval_rate_among_good, 1)}</td>
    <td class="num">${g.roc_auc === null ? "—" : fmt.num(g.roc_auc, 3)}</td>
  </tr>`).join("");

  // Three outcomes, not two. A gap that vanishes once you restrict to
  // applicants who did not default is explained by differing realised risk —
  // labelling it "Finding" in alarm colours would be the same overstatement the
  // permutation test was introduced to avoid.
  const survivesRisk = block.conclusive
    && block.permutation_p_value_among_good !== undefined
    && block.permutation_p_value_among_good !== null
    && block.permutation_p_value_among_good < 0.05;

  const heading = !block.conclusive
    ? "No conclusive gap"
    : survivesRisk
      ? "Finding — not explained by risk"
      : "Gap present, explained by realised risk";

  document.getElementById("fair-verdict").innerHTML = `
    <div class="callout ${survivesRisk ? "warn" : ""}">
      <strong>${heading}.</strong>
      ${escapeHtml(block.verdict)}
      ${block.permutation_p_value !== undefined ? `
        <div class="xs mt1">
          ratio ${fmt.num(block.approval_rate_ratio, 2)} ·
          chance alone gives ${fmt.num(block.null_median_ratio, 2)} ·
          p = ${fmt.num(block.permutation_p_value, 3)} over
          ${fmt.int(block.n_permutations)} permutations
        </div>` : ""}
    </div>`;
}

function initFairness() {
  const select = document.getElementById("fair-attr");
  const names = Object.keys(data.fairness.attributes);
  select.replaceChildren(...names.map((n) => new Option(n, n)));
  select.addEventListener("change", (e) => renderFairness(e.target.value));

  // Open on the attribute where the gap survives the risk restriction — the one
  // real finding — rather than the first alphabetically or the first that merely
  // trips the screen.
  const survives = (n) => {
    const b = data.fairness.attributes[n];
    return b.conclusive && b.permutation_p_value_among_good < 0.05;
  };
  const interesting = names.find(survives)
    || names.find((n) => data.fairness.attributes[n].conclusive)
    || names[0];
  select.value = interesting;
  renderFairness(interesting);

  renderAblation();
}

function renderAblation() {
  const ab = data.fairness.ablation;
  const body = document.querySelector("#ablation-table tbody");
  if (!ab) {
    body.innerHTML = `<tr><td class="muted">Ablation not present in this export.</td></tr>`;
    return;
  }
  const w = ab.with_protected, wo = ab.without_protected;

  const rows = [
    ["ROC-AUC", fmt.num(w.roc_auc, 4), fmt.num(wo.roc_auc, 4), fmt.num(-ab.auc_lost, 4)],
    ["Cost per applicant", fmt.num(w.cost_per_applicant, 4), fmt.num(wo.cost_per_applicant, 4),
      fmt.num(ab.cost_per_applicant_added, 4)],
    ["Decision threshold", fmt.num(w.threshold, 3), fmt.num(wo.threshold, 3), "—"],
  ];
  for (const [name, d] of Object.entries(ab.disparity_after_removal)) {
    if (d.ratio_before === null || Number.isNaN(d.ratio_before)) continue;
    rows.push([
      `${name} — p-value`,
      fmt.num(d.p_before, 3), fmt.num(d.p_after, 3),
      d.still_conclusive ? "still conclusive" : "no longer conclusive",
    ]);
  }

  body.innerHTML = `<tr><th></th><th class="num">with</th><th class="num">without</th><th class="num">change</th></tr>` +
    rows.map(([a, b, c, d]) =>
      `<tr><td class="wrapline">${escapeHtml(a)}</td><td class="num">${b}</td>
       <td class="num">${c}</td><td class="num">${escapeHtml(d)}</td></tr>`).join("");

  document.getElementById("ablation-note").innerHTML =
    `Removing all ${ab.removed.length} protected attributes costs
     <b>${fmt.num(ab.auc_lost, 4)} of AUC</b> — smaller than the width of the AUC confidence
     interval, so not distinguishable from zero on this data — and the measured disparity does
     not survive the removal. Both halves are measured on the same 250 rows, so this is
     suggestive rather than settled. What it is not is unmeasured.`;
}

/* --- boot ---------------------------------------------------------------- */

try {
  data = await loadJSON("data/credit.json");
  renderHeadline();
  initThreshold();
  initApplicants();
  initFairness();
  onResize(() => drawCostCurve(currentThreshold));
} catch (error) {
  showError(document.getElementById("headline-stats"), error);
  console.error(error);
}
