/* Shared page chrome: theme toggle, nav highlighting, number formatting. */

import { redrawAll } from "./charts.js";

/* --- theme ---------------------------------------------------------------
   Stored preference wins over the OS setting; absent one, the OS decides and
   the toggle flips to whichever is not currently showing. Applied by an inline
   script in <head> before paint, so there is no flash of the wrong theme — this
   module only handles the click. */

const root = document.documentElement;

function currentTheme() {
  return root.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

export function initTheme() {
  const button = document.querySelector(".theme-toggle");
  if (!button) return;

  const paint = () => {
    const dark = currentTheme() === "dark";
    button.textContent = dark ? "☀" : "☾";
    button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };

  button.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    paint();
    // Charts bake colours into SVG attributes, which var() cannot reach.
    redrawAll();
  });

  paint();
}

/* --- nav ---------------------------------------------------------------- */

export function initNav() {
  const here = location.pathname.split("/").pop() || "index.html";
  for (const link of document.querySelectorAll(".nav a[href]")) {
    if (link.getAttribute("href") === here) link.classList.add("active");
  }
}

/* --- formatting --------------------------------------------------------- */

export const fmt = {
  pct: (v, dp = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(dp)}%`),
  num: (v, dp = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(dp)),
  int: (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-GB")),
  money: (v, dp = 0) => (v === null || v === undefined ? "—" : "£" + Number(v).toLocaleString("en-GB", {
    minimumFractionDigits: dp, maximumFractionDigits: dp,
  })),
  /* Settlement period -> clock time. Period 1 is 00:00-00:30 local. Not valid
     on the two clock-change days a year, which the site does not plot. */
  period: (p) => {
    const minutes = (p - 1) * 30;
    return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
  },
};

/* --- data loading ------------------------------------------------------- */

export async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

export function showError(container, error) {
  container.innerHTML = `<div class="callout warn"><strong>Could not load data.</strong>
    ${error.message}. If you are opening this file directly from disk, the browser blocks
    <code>fetch</code> — serve it with <code>python3 -m http.server</code> instead.</div>`;
}

/* Debounced resize, so charts reflow without redrawing on every pixel. */
export function onResize(fn) {
  let timer;
  addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(fn, 150);
  });
}

initTheme();
initNav();
