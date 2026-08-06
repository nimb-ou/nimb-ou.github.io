/* ---------------------------------------------------------------------------
   Minimal SVG charting.

   Hand-rolled rather than pulled from a CDN. Three reasons, in order of how much
   they mattered: the charts here are lines, bars and a shaded band, which is
   perhaps 200 lines of SVG generation against a 60 KB dependency; a charting
   library's default styling fights a design system rather than joining it; and
   a self-contained page cannot break because someone else's CDN did.

   Everything reads colours from CSS custom properties, so charts follow the
   theme toggle without being redrawn — except where a value has to be baked
   into an attribute, which is why `redrawAll` exists.
--------------------------------------------------------------------------- */

const NS = "http://www.w3.org/2000/svg";

export function el(name, attrs = {}, children = []) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, String(v));
  }
  for (const child of [].concat(children)) {
    if (child) node.appendChild(child);
  }
  return node;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Charts that need literal colour values re-register here so the theme toggle
   can rebuild them. A CSS variable inside `fill=` would not work: SVG
   presentation attributes do not resolve var(). */
const registry = [];
export function register(fn) {
  registry.push(fn);
  fn();
}
export function redrawAll() {
  for (const fn of registry) fn();
}

const PALETTE = () => ({
  text: css("--text"),
  text2: css("--text-2"),
  text3: css("--text-3"),
  border: css("--border"),
  borderStrong: css("--border-strong"),
  accent: css("--accent"),
  good: css("--good"),
  bad: css("--bad"),
  surface: css("--surface"),
  surface2: css("--surface-2"),
});

function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

/**
 * Line chart with optional shaded bands and markers.
 *
 * series: [{ points: [[x, y], ...], colour, width, dash, label }]
 * bands:  [{ lower: [[x, y]], upper: [[x, y]], colour, opacity }]
 */
export function lineChart(container, { series = [], bands = [], xLabel, yLabel,
  height = 300, xTicks, yFormat = (v) => v, xFormat = (v) => v, yDomain }) {
  const p = PALETTE();
  const width = Math.max(container.clientWidth || 640, 480);
  const m = { top: 12, right: 14, bottom: 34, left: 52 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const all = [...series.flatMap((s) => s.points),
               ...bands.flatMap((b) => [...b.lower, ...b.upper])];
  if (!all.length) return;

  const xs = all.map((d) => d[0]);
  const ys = all.map((d) => d[1]);
  const xDomain = [Math.min(...xs), Math.max(...xs)];
  const yd = yDomain || [Math.min(...ys), Math.max(...ys)];
  const pad = (yd[1] - yd[0]) * 0.08 || 1;
  const yDom = yDomain || [yd[0] - pad, yd[1] + pad];

  const x = scale(xDomain, [0, iw]);
  const y = scale(yDom, [ih, 0]);

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`, width: "100%", height,
    role: "img", "aria-label": yLabel || "chart",
  });
  const g = el("g", { transform: `translate(${m.left},${m.top})` });

  for (const t of niceTicks(yDom[0], yDom[1], 5)) {
    g.appendChild(el("line", {
      x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: p.border, "stroke-width": 1,
    }));
    const label = el("text", {
      x: -8, y: y(t) + 4, "text-anchor": "end",
      fill: p.text3, "font-size": 11, "font-family": "ui-monospace, monospace",
    });
    label.textContent = yFormat(t);
    g.appendChild(label);
  }

  for (const t of (xTicks || niceTicks(xDomain[0], xDomain[1], 6))) {
    const label = el("text", {
      x: x(t), y: ih + 20, "text-anchor": "middle",
      fill: p.text3, "font-size": 11, "font-family": "ui-monospace, monospace",
    });
    label.textContent = xFormat(t);
    g.appendChild(label);
  }

  for (const band of bands) {
    const up = band.upper.map((d) => `${x(d[0])},${y(d[1])}`);
    const down = [...band.lower].reverse().map((d) => `${x(d[0])},${y(d[1])}`);
    g.appendChild(el("polygon", {
      points: [...up, ...down].join(" "),
      fill: band.colour || p.accent,
      opacity: band.opacity ?? 0.14,
    }));
  }

  for (const s of series) {
    if (!s.points.length) continue;
    g.appendChild(el("polyline", {
      points: s.points.map((d) => `${x(d[0])},${y(d[1])}`).join(" "),
      fill: "none",
      stroke: s.colour || p.accent,
      "stroke-width": s.width || 1.8,
      "stroke-dasharray": s.dash || null,
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
    }));
  }

  if (yLabel) {
    const label = el("text", {
      transform: `translate(${-m.left + 12},${ih / 2}) rotate(-90)`,
      "text-anchor": "middle", fill: p.text3, "font-size": 11,
    });
    label.textContent = yLabel;
    g.appendChild(label);
  }
  if (xLabel) {
    const label = el("text", {
      x: iw / 2, y: ih + 34, "text-anchor": "middle", fill: p.text3, "font-size": 11,
    });
    label.textContent = xLabel;
    g.appendChild(label);
  }

  svg.appendChild(g);
  container.replaceChildren(svg);
  return { x, y, g, svg, inner: { width: iw, height: ih } };
}

/** Vertical bar chart. bars: [{ label, value, colour }] */
export function barChart(container, { bars, height = 260, yFormat = (v) => v,
  yLabel, everyNthLabel = 1, highlight = () => false }) {
  const p = PALETTE();
  const width = Math.max(container.clientWidth || 640, 480);
  const m = { top: 10, right: 12, bottom: 34, left: 50 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  if (!bars.length) return;

  const maxV = Math.max(...bars.map((b) => b.value));
  const minV = Math.min(0, ...bars.map((b) => b.value));
  const y = scale([minV, maxV * 1.05], [ih, 0]);
  const step = iw / bars.length;
  const bw = Math.max(step * 0.72, 1);

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height,
    role: "img", "aria-label": yLabel || "bar chart" });
  const g = el("g", { transform: `translate(${m.left},${m.top})` });

  for (const t of niceTicks(minV, maxV * 1.05, 4)) {
    g.appendChild(el("line", { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: p.border }));
    const label = el("text", { x: -8, y: y(t) + 4, "text-anchor": "end",
      fill: p.text3, "font-size": 11, "font-family": "ui-monospace, monospace" });
    label.textContent = yFormat(t);
    g.appendChild(label);
  }

  bars.forEach((b, i) => {
    const top = y(Math.max(b.value, 0));
    const bottom = y(Math.min(b.value, 0));
    const rect = el("rect", {
      x: i * step + (step - bw) / 2, y: top, width: bw,
      height: Math.max(bottom - top, 1), rx: 2,
      fill: b.colour || (highlight(b, i) ? p.accent : p.text3),
      opacity: b.opacity ?? (highlight(b, i) ? 1 : 0.55),
    });
    rect.appendChild(el("title", {}, []));
    rect.lastChild.textContent = `${b.label}: ${yFormat(b.value)}`;
    g.appendChild(rect);

    if (i % everyNthLabel === 0) {
      const label = el("text", { x: i * step + step / 2, y: ih + 18,
        "text-anchor": "middle", fill: p.text3, "font-size": 10,
        "font-family": "ui-monospace, monospace" });
      label.textContent = b.label;
      g.appendChild(label);
    }
  });

  g.appendChild(el("line", { x1: 0, x2: iw, y1: y(0), y2: y(0), stroke: p.borderStrong }));
  svg.appendChild(g);
  container.replaceChildren(svg);
}

/** Horizontal diverging bars — used for SHAP-style attributions. */
export function divergingBars(container, { rows, valueFormat = (v) => v.toFixed(3) }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const frag = document.createDocumentFragment();

  for (const row of rows) {
    const wrapper = document.createElement("div");
    wrapper.className = "bar-row";

    const name = document.createElement("div");
    name.className = "name";
    name.innerHTML = `<b>${escapeHtml(row.label)}</b>${row.detail ? " — " + escapeHtml(row.detail) : ""}`;
    name.title = `${row.label}${row.detail ? " — " + row.detail : ""}`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const zero = document.createElement("div");
    zero.className = "zero";
    track.appendChild(zero);

    const fill = document.createElement("div");
    const frac = (Math.abs(row.value) / max) * 50;
    fill.className = "bar-fill " + (row.value > 0 ? "up" : "down");
    if (row.value > 0) { fill.style.left = "50%"; fill.style.width = frac + "%"; }
    else { fill.style.right = "50%"; fill.style.width = frac + "%"; }
    track.appendChild(fill);

    const val = document.createElement("div");
    val.className = "val";
    val.textContent = valueFormat(row.value);

    wrapper.append(name, track, val);
    frag.appendChild(wrapper);
  }

  container.replaceChildren(frag);
}

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export const colour = {
  get accent() { return css("--accent"); },
  get good() { return css("--good"); },
  get bad() { return css("--bad"); },
  get text() { return css("--text"); },
  get text2() { return css("--text-2"); },
  get text3() { return css("--text-3"); },
  get border() { return css("--border"); },
};
