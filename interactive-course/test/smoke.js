/* Headless smoke test: boots the site in jsdom, renders every section,
   mounts every lab, and reports any error. Run: node site/test/smoke.js  */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require(process.env.JSDOM_PATH || 'jsdom');

const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');

const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost/' });
const { window } = dom;

/* ---- stub the 2-D canvas so drawing code runs for real without a GPU ---- */
const ctxStub = () => new Proxy({
  canvas: { width: 800, height: 400 },
  measureText: () => ({ width: 40 }),
  createImageData: (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, Math.ceil(w)) * Math.max(1, Math.ceil(h)) * 4), width: w, height: h }),
  getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(1, Math.ceil(w)) * Math.max(1, Math.ceil(h)) * 4), width: w, height: h }),
  createLinearGradient: () => ({ addColorStop() {} }),
  createRadialGradient: () => ({ addColorStop() {} }),
  setTransform() {}, save() {}, restore() {}, beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
  arc() {}, arcTo() {}, rect() {}, fill() {}, stroke() {}, clip() {}, fillRect() {}, strokeRect() {}, clearRect() {},
  fillText() {}, strokeText() {}, translate() {}, rotate() {}, scale() {}, putImageData() {}, drawImage() {}, setLineDash() {},
  quadraticCurveTo() {}, bezierCurveTo() {}, ellipse() {}
}, { get: (t, k) => (k in t ? t[k] : undefined), set: (t, k, v) => { t[k] = v; return true; } });

window.HTMLCanvasElement.prototype.getContext = function () { return ctxStub(); };
window.HTMLCanvasElement.prototype.setPointerCapture = function () {};
window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 0);
window.cancelAnimationFrame = id => clearTimeout(id);
window.matchMedia = window.matchMedia || (q => ({ matches: false, media: q, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {} }));
window.getComputedStyle = ((orig) => function (e) {
  const s = orig.call(window, e);
  return new Proxy(s, { get: (t, k) => k === 'getPropertyValue' ? (name) => ({ '--text': '#111', '--muted': '#666', '--faint': '#999', '--line': '#ddd', '--panel': '#eee', '--paper': '#fff', '--blue': '#2f4bd6', '--red': '#b3261e', '--green': '#186a3b', '--amber': '#a86400' }[name] || '#888') : t[k] });
})(window.getComputedStyle);
window.ResizeObserver = class { observe() {} disconnect() {} unobserve() {} };
window.scrollTo = () => {};
window.confirm = () => false;

const errors = [];
window.addEventListener('error', e => errors.push('window error: ' + (e.error && e.error.stack || e.message)));
const origWarn = console.warn, origErr = console.error;
console.warn = (...a) => { const s = a.join(' '); if (!/katex/i.test(s)) errors.push('warn: ' + s); };
console.error = (...a) => errors.push('error: ' + a.join(' '));

/* ---- load scripts in document order ---- */
const scripts = Array.from(dom.window.document.querySelectorAll('script[src]')).map(s => s.getAttribute('src'));
for (const src of scripts) {
  const file = path.join(ROOT, src);
  if (!fs.existsSync(file)) { errors.push('missing script ' + src); continue; }
  try { window.eval(fs.readFileSync(file, 'utf8')); }
  catch (e) { errors.push('LOAD FAIL ' + src + ': ' + e.stack.split('\n').slice(0, 3).join(' | ')); }
}

const ML = window.ML;
if (!ML) { console.log('FATAL: ML not defined'); process.exit(1); }

/* ---- boot and walk every section ---- */
try { ML.boot(); } catch (e) { errors.push('boot: ' + e.stack); }

let labCount = 0, quizCount = 0, cardCount = 0, wordCount = 0, mathCount = 0;
const perSection = [];
ML.sections.forEach(s => {
  const before = errors.length;
  try { ML.render(s.id); } catch (e) { errors.push('render ' + s.id + ': ' + e.stack.split('\n').slice(0, 4).join(' | ')); }
  /* Surfaces defer their first paint to requestAnimationFrame, so a lab whose
     draw() throws would otherwise fail in a timer nobody is watching. The
     theme event forces every mounted surface to draw synchronously, here. */
  try { window.document.dispatchEvent(new window.CustomEvent('ml:theme', { detail: 'light' })); }
  catch (e) { errors.push('draw ' + s.id + ': ' + (e.stack || e.message).split('\n').slice(0, 4).join(' | ')); }
  const main = window.document.getElementById('main');
  const text = (main.textContent || '');
  const words = text.split(/\s+/).filter(Boolean).length;
  wordCount += words;
  const labs = s.labs ? Object.keys(s.labs).length : 0;
  labCount += labs;
  quizCount += (s.quiz || []).length;
  cardCount += (s.cards || []).length;
  mathCount += ((s.html || '').match(/\$\$?/g) || []).length / 2;
  // did every declared lab find its host?
  (s.labs ? Object.keys(s.labs) : []).forEach(k => {
    if (!main.querySelector('[data-lab="' + k + '"]')) errors.push('lab host missing: ' + s.id + '/' + k);
  });
  // unresolved template artefacts — anchored so the English word "undefined"
  // inside prose does not trip it, but a hole in a template does
  const ARTEFACT = /(>\s*undefined|undefined\s*<|[=:]\s*undefined|undefined(px|%|,\s*\d))|\[object Object\]|NaN%/;
  if (ARTEFACT.test(main.innerHTML)) {
    const m = main.innerHTML.match(/.{0,60}(undefined|\[object Object\]|NaN%).{0,60}/);
    errors.push('suspicious output in ' + s.id + ': …' + (m ? m[0].replace(/\s+/g, ' ') : '') + '…');
  }
  perSection.push({ id: s.id, num: s.num, words, labs, quiz: (s.quiz || []).length, newErrors: errors.length - before });
});

console.warn = origWarn; console.error = origErr;

const bad = perSection.filter(p => p.newErrors > 0);
console.log('sections   :', ML.sections.length, 'in', ML.tracks.length, 'tracks');
console.log('labs       :', labCount);
console.log('quiz Qs    :', quizCount, '| cards:', cardCount);
console.log('formulas   :', Math.round(mathCount));
console.log('words      :', wordCount.toLocaleString());
console.log('thin (<350w):', perSection.filter(p => p.words < 350).map(p => p.num + ' ' + p.id).join(', ') || 'none');
console.log('no labs     :', perSection.filter(p => p.labs === 0).map(p => p.num).join(', ') || 'none');
console.log('errors      :', errors.length);
errors.slice(0, 40).forEach(e => console.log('  - ' + e));
process.exit(errors.length ? 1 : 0);
