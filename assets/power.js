/* Interactive behaviour for the power project page.
 *
 * Reads `data/power.json`, exported by `tools/export_site_data.py` in the repo.
 * Two resolutions: daily aggregates for the whole 4.5-year out-of-sample period,
 * and full half-hourly detail for a labelled sample of days. Days are labelled
 * with *why* they were sampled, so the page can say "worst forecast day in the
 * sample" rather than presenting a cherry-pick as typical.
 */

import { loadJSON, fmt, showError, onResize } from "./site.js";
import { lineChart, barChart, register, escapeHtml, colour } from "./charts.js";

let data;
let currentDay;
let band = "none";

function set(key, value) {
  for (const node of document.querySelectorAll(`[data-k="${key}"]`)) node.textContent = value;
}

/* --- headline ------------------------------------------------------------ */

function renderHeadline() {
  const h = data.headline;
  set("imp-strong", fmt.pct(h.improvement_vs_strongest, 1));
  set("imp-strong-sub", `vs ${h.strongest_baseline.replaceAll("_", " ")}`);
  set("imp-ablation", fmt.pct(h.improvement_without_weather, 1));
  set("n-pred", fmt.int(h.n_predictions));

  const study = data.study || {};
  set("folds", `${study.initial_train_days ?? "—"}d train, ${study.refit_days ?? "—"}d refit`);

  const strategy = data.strategy?.schedules || {};
  set("pnl-xgb", fmt.money(strategy.xgb?.total));
  set("pnl-naive", fmt.money(strategy.naive?.total));
  set("pnl-oracle", fmt.money(strategy.oracle?.total));

  // `uplift_vs_naive_pct` is a *fraction* (0.538), not a percentage, despite
  // the name — which is exactly what made the first version of this line print
  // "+0.5%" for a 54% uplift. Formatted with fmt.pct rather than trusting the
  // field name.
  const uplift = data.strategy?.uplift_vs_naive_pct;
  const upliftText = uplift === undefined || uplift === null
    ? "—"
    : `+${fmt.pct(uplift, 1)}`;
  set("uplift", upliftText);
  set("uplift-inline", upliftText);
  set("build.stamp", `generated from commit ${data.generated_from_commit ?? "—"}`);
}

/* --- day explorer -------------------------------------------------------- */

function dayOptions() {
  const select = document.getElementById("day");
  select.replaceChildren(...data.sample_days.map((d) => {
    const option = document.createElement("option");
    option.value = d.date;
    option.textContent = `${d.date} — ${d.reason}`;
    return option;
  }));

  // Open on a typical day from the most recent regime.
  //
  // The tempting choice is the worst forecast day, which is also the best
  // trading day and the widest spread — genuinely the most interesting day in
  // the set. But on that day prices reached 2,000 GBP/MWh and the forecast
  // stayed near 100, so as a *default* it shows a visitor the model failing
  // before showing them it working. A default should be representative. The
  // extreme is one labelled click away, and the note below points at it.
  const opener = data.sample_days.find((d) => d.reason === "typical — post_crisis")
    || data.sample_days.find((d) => d.reason.startsWith("typical"))
    || data.sample_days[0];
  select.value = opener.date;
  renderDay(opener.date);
}

function renderDay(date) {
  currentDay = data.sample_days.find((d) => d.date === date);
  if (!currentDay) return;

  document.getElementById("d-reason").textContent = currentDay.reason;
  document.getElementById("d-regime").textContent =
    currentDay.regime.replaceAll("_", " ");
  document.getElementById("d-mae").textContent = fmt.num(currentDay.mae, 1);
  document.getElementById("d-pnl").textContent =
    currentDay.pnl === null ? "no trade" : fmt.money(currentDay.pnl, 0);

  drawDay();
  drawBattery();
}

function drawDay() {
  const container = document.getElementById("day-chart");
  if (!container || !currentDay) return;

  const rows = currentDay.periods;
  const point = (key) => rows
    .filter((r) => r[key] !== null && r[key] !== undefined)
    .map((r) => [r.settlement_period, r[key]]);

  const bands = [];
  const legend = document.getElementById("band-legend");
  if (band === "raw" && rows.some((r) => r.pred_xgb_p10 !== undefined)) {
    bands.push({ lower: point("pred_xgb_p10"), upper: point("pred_xgb_p90"),
      colour: colour.accent, opacity: 0.16 });
  } else if (band === "conformal" && rows.some((r) => r.pred_xgb_p10_conformal !== undefined)) {
    bands.push({ lower: point("pred_xgb_p10_conformal"), upper: point("pred_xgb_p90_conformal"),
      colour: colour.accent, opacity: 0.16 });
  }
  if (legend) legend.hidden = bands.length === 0;

  lineChart(container, {
    height: 300,
    bands,
    series: [
      { points: point("pred_naive"), colour: colour.text3, width: 1.4, dash: "5 4" },
      { points: point("pred_xgb"), colour: colour.accent, width: 2 },
      { points: point("price"), colour: colour.text, width: 2.2 },
    ],
    xLabel: "settlement period (1 = 00:00–00:30 local)",
    yLabel: "GBP/MWh",
    yFormat: (v) => v.toFixed(0),
    xFormat: (v) => fmt.period(Math.round(v)),
    xTicks: [1, 9, 17, 25, 33, 41, 48],
  });
}

function drawBattery() {
  const container = document.getElementById("day-battery");
  if (!container || !currentDay) return;

  if (!currentDay.positions.length) {
    container.innerHTML = `<p class="muted small">The battery stood down this day — the
      forecast spread did not clear the minimum worth cycling for.</p>`;
    return;
  }

  const byPeriod = new Map(currentDay.positions.map((p) => [p.settlement_period, p.position]));
  const bars = currentDay.periods.map((r) => {
    const position = byPeriod.get(r.settlement_period) || 0;
    return {
      label: r.settlement_period % 8 === 1 ? fmt.period(r.settlement_period) : "",
      value: position,
      colour: position < 0 ? colour.good : position > 0 ? colour.bad : colour.border,
      opacity: position === 0 ? 0.35 : 0.9,
    };
  });

  barChart(container, {
    bars, height: 150, yLabel: "MW",
    yFormat: (v) => v.toFixed(1),
  });
}

function initDays() {
  document.getElementById("day").addEventListener("change", (e) => renderDay(e.target.value));

  for (const button of document.querySelectorAll("[data-band]")) {
    button.addEventListener("click", () => {
      band = button.dataset.band;
      for (const other of document.querySelectorAll("[data-band]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      drawDay();
    });
  }

  dayOptions();
  register(() => { drawDay(); drawBattery(); });
}

/* --- tables -------------------------------------------------------------- */

function renderModels() {
  const declared = data.headline.declared_baseline;
  const strongest = data.headline.strongest_baseline;

  const order = ["xgb", ...Object.keys(data.models).filter((k) => k !== "xgb")
    .sort((a, b) => data.models[a].mae - data.models[b].mae)];

  document.querySelector("#models-table tbody").innerHTML = order.map((name) => {
    const m = data.models[name];
    const notes = [];
    if (name === declared) notes.push('<span class="pill">declared baseline</span>');
    if (name === strongest) notes.push('<span class="pill accent">strongest baseline</span>');
    return `<tr class="${name === "xgb" ? "highlight" : ""}">
      <td class="wrapline">${escapeHtml(name.replaceAll("_", " "))} ${notes.join(" ")}</td>
      <td class="num">${fmt.num(m.mae, 2)}</td>
      <td class="num">${fmt.num(m.rmse, 2)}</td>
      <td class="num">${fmt.num(m.bias, 2)}</td>
      <td class="num">${fmt.pct(m.mae_improvement_vs_naive, 1)}</td>
    </tr>`;
  }).join("");

  const dm = data.diebold_mariano || {};
  document.getElementById("dm-note").innerHTML = dm.p_value === undefined ? "" :
    `Diebold–Mariano against the declared baseline: statistic
     <b>${fmt.num(dm.statistic, 1)}</b>, p = ${Number(dm.p_value).toExponential(1)} over
     ${fmt.int(dm.n)} paired forecast errors, with a Newey–West variance because half-hourly
     errors are autocorrelated. The improvement is not a sampling accident — which is a
     separate question from whether it is large enough to matter.`;
}

function renderRegimes() {
  const rows = Object.entries(data.by_regime || {});
  document.querySelector("#regime-table tbody").innerHTML = rows.length
    ? rows.map(([name, r]) => `<tr>
        <td class="wrapline">${escapeHtml(name.replaceAll("_", " "))}</td>
        <td class="num">${fmt.int(r.n)}</td>
        <td class="num">${fmt.num(r.mean_price, 1)}</td>
        <td class="num">${fmt.num(r.price_volatility, 1)}</td>
        <td class="num">${fmt.num(r.mae_xgb, 2)}</td>
        <td class="num">${fmt.num(r.mae_naive, 2)}</td>
        <td class="num">${fmt.pct(r.mae_improvement_vs_naive, 1)}</td>
      </tr>`).join("")
    : `<tr><td colspan="7" class="muted">Regime breakdown not present in this export.</td></tr>`;
}

function renderIntervals() {
  const intervals = data.intervals;
  const body = document.querySelector("#interval-table tbody");
  if (!intervals) {
    body.innerHTML = `<tr><td colspan="6" class="muted">This run was made without quantiles.</td></tr>`;
    return;
  }
  const row = (label, b) => `<tr>
    <td class="wrapline">${label}</td>
    <td class="num">${fmt.pct(b.coverage, 1)}</td>
    <td class="num">${fmt.pct(b.nominal_coverage, 0)}</td>
    <td class="num">${fmt.num(b.mean_width, 1)}</td>
    <td class="num">${fmt.num(b.winkler, 1)}</td>
    <td class="num">${fmt.int(b.crossings)}</td>
  </tr>`;

  body.innerHTML = row("raw quantile fit", intervals.interval)
    + (intervals.interval_conformal
      ? row(`conformalised <span class="pill accent">${intervals.interval_conformal.calibration_days}-day calibration</span>`,
          intervals.interval_conformal)
      : "");
}

/* --- charts -------------------------------------------------------------- */

function drawPeriodChart() {
  const container = document.getElementById("period-chart");
  if (!container || !data.by_period?.length) return;

  lineChart(container, {
    height: 280,
    series: [
      { points: data.by_period.map((r) => [r.period, r.mae_naive]),
        colour: colour.text3, width: 1.6, dash: "5 4" },
      { points: data.by_period.map((r) => [r.period, r.mae_xgb]),
        colour: colour.accent, width: 2.2 },
    ],
    xLabel: "settlement period",
    yLabel: "MAE, GBP/MWh",
    yFormat: (v) => v.toFixed(0),
    xFormat: (v) => fmt.period(Math.round(v)),
    xTicks: [1, 9, 17, 25, 33, 41, 48],
  });
}

function drawPnlChart() {
  const container = document.getElementById("pnl-chart");
  if (!container || !data.daily?.length) return;

  // Index by position rather than by date: the x scale is numeric, and the
  // labels come back from the same array.
  const points = data.daily.map((r, i) => [i, r.cum_pnl]);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) =>
    Math.min(Math.round(f * (data.daily.length - 1)), data.daily.length - 1));

  lineChart(container, {
    height: 280,
    series: [{ points, colour: colour.accent, width: 2 }],
    yLabel: "cumulative GBP",
    yFormat: (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)),
    xTicks: ticks,
    xFormat: (v) => data.daily[Math.round(v)]?.settlement_date?.slice(0, 7) ?? "",
  });
}

/* --- boot ---------------------------------------------------------------- */

try {
  data = await loadJSON("data/power.json");
  renderHeadline();
  initDays();
  renderModels();
  renderRegimes();
  renderIntervals();
  register(drawPeriodChart);
  register(drawPnlChart);
  onResize(() => { drawDay(); drawBattery(); drawPeriodChart(); drawPnlChart(); });
} catch (error) {
  showError(document.getElementById("headline-stats"), error);
  console.error(error);
}
