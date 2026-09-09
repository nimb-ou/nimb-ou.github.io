/* ============================================================
   Viz — canvas plotting + control widgets used by every lab
   ============================================================ */
window.Viz = (function () {
  'use strict';
  const el = ML.el;

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function theme() {
    return {
      text: cssVar('--text', '#111'),
      muted: cssVar('--muted', '#666'),
      faint: cssVar('--faint', '#999'),
      line: cssVar('--line', '#ddd'),
      panel: cssVar('--panel', '#eee'),
      paper: cssVar('--paper', '#fff'),
      blue: cssVar('--blue', '#2f4bd6'),
      red: cssVar('--red', '#b3261e'),
      green: cssVar('--green', '#186a3b'),
      amber: cssVar('--amber', '#a86400'),
      violet: cssVar('--violet', '#6743d1'),
      teal: cssVar('--teal', '#0d7683'),
      c1: cssVar('--c1', '#2b47c9'), c2: cssVar('--c2', '#d0492f'),
      c3: cssVar('--c3', '#1a7a45'), c4: cssVar('--c4', '#a8730a'),
      c5: cssVar('--c5', '#6743d1'), c6: cssVar('--c6', '#0d7683'),
      c7: cssVar('--c7', '#c2357e'), c8: cssVar('--c8', '#5c6b2a'),
      dark: (function () {
        const attr = document.documentElement.getAttribute('data-theme');
        if (attr === 'dark' || attr === 'light') return attr === 'dark';
        return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
      })()
    };
  }

  /* ---------------- canvas surface ---------------- */
  function surface(host, opts) {
    opts = opts || {};
    const wrap = el('div', { class: 'canvaswrap' });
    const cv = el('canvas');
    wrap.appendChild(cv);
    host.appendChild(wrap);
    const ctx = cv.getContext('2d');
    const S = {
      canvas: cv, ctx: ctx, w: 300, h: opts.height || 260,
      aspect: opts.aspect || null,
      draw: opts.draw || function () {},
      redraw: function () { S.resize(); }
    };

    S.resize = function () {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      const w = Math.max(200, Math.round(rect.width));
      const h = S.aspect ? Math.round(w * S.aspect) : (opts.height || 260);
      S.w = w; S.h = h;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      S.draw(ctx, w, h, theme());
    };

    let ro;
    if (window.ResizeObserver) { ro = new ResizeObserver(ML.debounce(() => S.resize(), 60)); ro.observe(wrap); }
    else window.addEventListener('resize', S.resize);

    const onTheme = () => S.resize();
    document.addEventListener('ml:theme', onTheme);
    ML.onCleanup(() => { if (ro) ro.disconnect(); document.removeEventListener('ml:theme', onTheme); });

    requestAnimationFrame(() => S.resize());
    return S;
  }

  /* ---------------- plot: data <-> pixel mapping ---------------- */
  function plot(ctx, w, h, cfg) {
    cfg = cfg || {};
    const pad = Object.assign({ l: 44, r: 14, t: 14, b: 34 }, cfg.pad || {});
    const xd = cfg.xd || [0, 1], yd = cfg.yd || [0, 1];
    const T = theme();
    const P = {
      pad: pad, xd: xd, yd: yd, ctx: ctx, w: w, h: h, T: T,
      x: v => pad.l + (v - xd[0]) / (xd[1] - xd[0]) * (w - pad.l - pad.r),
      y: v => h - pad.b - (v - yd[0]) / (yd[1] - yd[0]) * (h - pad.t - pad.b),
      ix: px => xd[0] + (px - pad.l) / (w - pad.l - pad.r) * (xd[1] - xd[0]),
      iy: py => yd[0] + (h - pad.b - py) / (h - pad.t - pad.b) * (yd[1] - yd[0]),
      get pw() { return w - pad.l - pad.r; },
      get ph() { return h - pad.t - pad.b; }
    };

    P.frame = function (o) {
      o = o || {};
      ctx.save();
      // grid
      if (o.grid !== false) {
        ctx.strokeStyle = T.line; ctx.lineWidth = 1; ctx.globalAlpha = .55;
        const xt = o.xticks || ticks(xd[0], xd[1], 6), yt = o.yticks || ticks(yd[0], yd[1], 5);
        xt.forEach(t => { ctx.beginPath(); ctx.moveTo(r(P.x(t)), pad.t); ctx.lineTo(r(P.x(t)), h - pad.b); ctx.stroke(); });
        yt.forEach(t => { ctx.beginPath(); ctx.moveTo(pad.l, r(P.y(t))); ctx.lineTo(w - pad.r, r(P.y(t))); ctx.stroke(); });
        ctx.globalAlpha = 1;
      }
      // axes
      ctx.strokeStyle = T.faint; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
      // labels
      ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      (o.xticks || ticks(xd[0], xd[1], 6)).forEach(t => ctx.fillText(fmt(t, o.xfmt), P.x(t), h - pad.b + 6));
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      (o.yticks || ticks(yd[0], yd[1], 5)).forEach(t => ctx.fillText(fmt(t, o.yfmt), pad.l - 7, P.y(t)));
      if (o.xlabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif, system-ui'; ctx.fillText(o.xlabel, pad.l + P.pw / 2, h - 2); }
      if (o.ylabel) { ctx.save(); ctx.translate(11, pad.t + P.ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = T.muted; ctx.font = '11px ui-sans-serif, system-ui'; ctx.fillText(o.ylabel, 0, 0); ctx.restore(); }
      ctx.restore();
      return P;
    };

    P.clip = function (fn) {
      ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, P.pw, P.ph); ctx.clip(); fn(); ctx.restore();
    };

    P.line = function (pts, o) {
      o = o || {};
      if (!pts.length) return P;
      ctx.save();
      ctx.strokeStyle = o.color || T.blue; ctx.lineWidth = o.width || 2;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (o.dash) ctx.setLineDash(o.dash);
      if (o.alpha != null) ctx.globalAlpha = o.alpha;
      ctx.beginPath();
      pts.forEach((p, i) => { const X = P.x(p[0]), Y = P.y(p[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke(); ctx.restore();
      return P;
    };

    P.fn = function (f, o) {
      o = o || {};
      const n = o.n || 220, pts = [];
      const a = o.from != null ? o.from : xd[0], b = o.to != null ? o.to : xd[1];
      for (let i = 0; i <= n; i++) { const x = a + (b - a) * i / n; const y = f(x); if (isFinite(y)) pts.push([x, y]); }
      return P.line(pts, o);
    };

    P.area = function (pts, o) {
      o = o || {};
      if (!pts.length) return P;
      const base = o.base != null ? o.base : yd[0];
      ctx.save();
      ctx.fillStyle = o.color || T.blue; ctx.globalAlpha = o.alpha != null ? o.alpha : .16;
      ctx.beginPath(); ctx.moveTo(P.x(pts[0][0]), P.y(base));
      pts.forEach(p => ctx.lineTo(P.x(p[0]), P.y(p[1])));
      ctx.lineTo(P.x(pts[pts.length - 1][0]), P.y(base)); ctx.closePath(); ctx.fill(); ctx.restore();
      return P;
    };

    P.dots = function (pts, o) {
      o = o || {};
      ctx.save();
      const rad = o.r || 3.2;
      pts.forEach(p => {
        ctx.beginPath(); ctx.arc(P.x(p[0]), P.y(p[1]), rad, 0, 6.2832);
        ctx.fillStyle = (typeof o.color === 'function') ? o.color(p) : (o.color || T.blue);
        ctx.globalAlpha = o.alpha != null ? o.alpha : .9;
        ctx.fill();
        if (o.stroke) { ctx.globalAlpha = 1; ctx.lineWidth = o.strokeWidth || 1.5; ctx.strokeStyle = o.stroke === true ? T.paper : o.stroke; ctx.stroke(); }
      });
      ctx.restore();
      return P;
    };

    P.bars = function (vals, o) {
      o = o || {};
      ctx.save();
      const n = vals.length, gap = o.gap != null ? o.gap : 0.16;
      const bw = P.pw / n * (1 - gap);
      vals.forEach((v, i) => {
        const cx = pad.l + P.pw * (i + .5) / n;
        const y0 = P.y(0), y1 = P.y(v);
        ctx.fillStyle = (typeof o.color === 'function') ? o.color(v, i) : (o.color || T.blue);
        ctx.globalAlpha = o.alpha != null ? o.alpha : 1;
        ctx.fillRect(cx - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0));
        if (o.labels && o.labels[i] != null) {
          ctx.globalAlpha = 1; ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText(o.labels[i], cx, y0 + 5);
        }
      });
      ctx.restore();
      return P;
    };

    P.vline = function (x, o) {
      o = o || {}; ctx.save();
      ctx.strokeStyle = o.color || T.red; ctx.lineWidth = o.width || 1.5;
      if (o.dash !== false) ctx.setLineDash(o.dash || [5, 4]);
      ctx.beginPath(); ctx.moveTo(r(P.x(x)), pad.t); ctx.lineTo(r(P.x(x)), h - pad.b); ctx.stroke();
      if (o.label) { ctx.setLineDash([]); ctx.fillStyle = o.color || T.red; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.label, P.x(x) + 5, pad.t + 3); }
      ctx.restore(); return P;
    };
    P.hline = function (y, o) {
      o = o || {}; ctx.save();
      ctx.strokeStyle = o.color || T.red; ctx.lineWidth = o.width || 1.5;
      if (o.dash !== false) ctx.setLineDash(o.dash || [5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, r(P.y(y))); ctx.lineTo(w - pad.r, r(P.y(y))); ctx.stroke();
      if (o.label) { ctx.setLineDash([]); ctx.fillStyle = o.color || T.red; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.label, w - pad.r - 3, P.y(y) - 3); }
      ctx.restore(); return P;
    };

    P.text = function (x, y, s, o) {
      o = o || {}; ctx.save();
      ctx.fillStyle = o.color || T.muted;
      ctx.font = o.font || '11px ui-sans-serif, system-ui';
      ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'middle';
      ctx.fillText(s, P.x(x) + (o.dx || 0), P.y(y) + (o.dy || 0));
      ctx.restore(); return P;
    };

    P.arrow = function (x0, y0, x1, y1, o) {
      o = o || {}; ctx.save();
      ctx.strokeStyle = o.color || T.blue; ctx.fillStyle = o.color || T.blue; ctx.lineWidth = o.width || 2;
      const X0 = P.x(x0), Y0 = P.y(y0), X1 = P.x(x1), Y1 = P.y(y1);
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
      const a = Math.atan2(Y1 - Y0, X1 - X0), hs = o.head || 8;
      ctx.beginPath(); ctx.moveTo(X1, Y1);
      ctx.lineTo(X1 - hs * Math.cos(a - .4), Y1 - hs * Math.sin(a - .4));
      ctx.lineTo(X1 - hs * Math.cos(a + .4), Y1 - hs * Math.sin(a + .4));
      ctx.closePath(); ctx.fill(); ctx.restore(); return P;
    };

    /* heat/contour field: f(x,y) sampled on a grid.
       Rendered through an offscreen canvas because putImageData ignores the
       DPR transform — drawImage respects it, so the field lands on the axes. */
    P.field = function (f, o) {
      o = o || {};
      const step = o.step || 4;
      const iw = Math.max(1, Math.ceil(P.pw)), ih = Math.max(1, Math.ceil(P.ph));
      const img = ctx.createImageData(iw, ih);
      const cols = o.colors || defaultRamp(T);
      let lo = Infinity, hi = -Infinity;
      const vals = [];
      for (let py = 0; py < P.ph; py += step) {
        const row = [];
        for (let px = 0; px < P.pw; px += step) {
          const v = f(P.ix(px + pad.l), P.iy(py + pad.t));
          row.push(v); if (v < lo) lo = v; if (v > hi) hi = v;
        }
        vals.push(row);
      }
      if (o.lo != null) lo = o.lo; if (o.hi != null) hi = o.hi;
      const span = (hi - lo) || 1;
      for (let py = 0; py < ih; py++) {
        const ry = vals[Math.min(vals.length - 1, Math.floor(py / step))];
        for (let px = 0; px < iw; px++) {
          const v = ry[Math.min(ry.length - 1, Math.floor(px / step))];
          const t = Math.max(0, Math.min(1, (v - lo) / span));
          const c = cols(t);
          const i = (py * iw + px) * 4;
          img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = c[3];
        }
      }
      blit(ctx, img, pad.l, pad.t, iw, ih);
      return P;
    };

    /* shaded band between two curves, e.g. a ±2σ predictive interval */
    P.band = function (upper, lower, o) {
      o = o || {};
      if (!upper.length) return P;
      ctx.save();
      ctx.fillStyle = o.color || T.blue;
      ctx.globalAlpha = o.alpha != null ? o.alpha : .15;
      ctx.beginPath();
      upper.forEach((p, i) => { const X = P.x(p[0]), Y = P.y(p[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      for (let i = lower.length - 1; i >= 0; i--) ctx.lineTo(P.x(lower[i][0]), P.y(lower[i][1]));
      ctx.closePath(); ctx.fill(); ctx.restore();
      return P;
    };

    /* vertical error bars at given x with [lo,hi] */
    P.errbars = function (pts, o) {
      o = o || {}; ctx.save();
      ctx.strokeStyle = o.color || T.muted; ctx.lineWidth = o.width || 1.4;
      const cap = o.cap == null ? 4 : o.cap;
      pts.forEach(p => {
        const X = P.x(p[0]);
        ctx.beginPath(); ctx.moveTo(X, P.y(p[1])); ctx.lineTo(X, P.y(p[2])); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(X - cap, P.y(p[1])); ctx.lineTo(X + cap, P.y(p[1]));
        ctx.moveTo(X - cap, P.y(p[2])); ctx.lineTo(X + cap, P.y(p[2])); ctx.stroke();
      });
      ctx.restore(); return P;
    };

    /* horizontal bars with a label column — for feature importance, SHAP, ablations */
    P.hbars = function (items, o) {
      o = o || {};
      ctx.save();
      const n = items.length, rowH = P.ph / n, bh = Math.min(o.maxH || 22, rowH * .62);
      const vmax = o.max != null ? o.max : Math.max.apply(null, items.map(i => Math.abs(i.v))) || 1;
      const zero = o.signed ? pad.l + P.pw / 2 : pad.l;
      const scale = o.signed ? (P.pw / 2 - 6) / vmax : (P.pw - 6) / vmax;
      items.forEach((it, i) => {
        const cy = pad.t + rowH * (i + .5);
        const len = it.v * scale;
        ctx.fillStyle = it.c || (it.v < 0 ? T.c2 : T.c1);
        ctx.globalAlpha = o.alpha != null ? o.alpha : .85;
        ctx.fillRect(Math.min(zero, zero + len), cy - bh / 2, Math.abs(len), bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = T.text; ctx.font = '11px ui-sans-serif, system-ui';
        ctx.textBaseline = 'middle';
        if (o.signed) { ctx.textAlign = len < 0 ? 'left' : 'right'; ctx.fillText(it.k, zero + (len < 0 ? 6 : -6), cy); }
        else { ctx.textAlign = 'left'; ctx.fillText(it.k, zero + 7, cy); }
        if (o.value !== false) {
          ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
          ctx.textAlign = len < 0 || !o.signed ? 'left' : 'right';
          const tx = o.signed ? (len < 0 ? zero + len - 6 : zero + len + 6) : zero + len + 6;
          ctx.textAlign = o.signed && len >= 0 ? 'left' : (o.signed ? 'right' : 'left');
          ctx.fillText(o.fmt ? o.fmt(it.v) : fmt(it.v), tx, cy);
        }
      });
      if (o.signed) { ctx.strokeStyle = T.faint; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(r(zero), pad.t); ctx.lineTo(r(zero), pad.t + P.ph); ctx.stroke(); }
      ctx.restore();
      return P;
    };

    /* labelled matrix heatmap drawn in pixel space inside the frame */
    P.heat = function (M, o) {
      o = o || {};
      const rows = M.length, cols = M[0].length;
      const cw = P.pw / cols, ch = P.ph / rows;
      let lo = o.lo, hi = o.hi;
      if (lo == null || hi == null) {
        lo = Infinity; hi = -Infinity;
        M.forEach(r2 => r2.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
      }
      const span = (hi - lo) || 1;
      const ramp = o.ramp || function (t) {
        // white/paper -> accent, readable in both themes
        const c = o.hue === 'red' ? [208, 73, 47] : o.hue === 'green' ? [26, 122, 69] : [43, 71, 201];
        return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (0.06 + 0.86 * t).toFixed(3) + ')';
      };
      ctx.save();
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const t = Math.max(0, Math.min(1, (M[i][j] - lo) / span));
        ctx.fillStyle = ramp(t, M[i][j]);
        ctx.fillRect(pad.l + j * cw, pad.t + i * ch, cw + .6, ch + .6);
      }
      if (o.values !== false && cw > 26 && ch > 14) {
        ctx.font = (o.font || '10px ui-monospace, monospace');
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
          const t = (M[i][j] - lo) / span;
          ctx.fillStyle = t > .62 ? (T.dark ? '#0b0d13' : '#fff') : T.text;
          ctx.fillText(o.cell ? o.cell(M[i][j], i, j) : fmt(M[i][j]), pad.l + (j + .5) * cw, pad.t + (i + .5) * ch);
        }
      }
      ctx.strokeStyle = T.line; ctx.lineWidth = 1;
      ctx.strokeRect(pad.l, pad.t, P.pw, P.ph);
      // axis labels around the grid
      ctx.fillStyle = T.faint; ctx.font = '10px ui-monospace, monospace';
      if (o.cols) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; o.cols.forEach((c, j) => ctx.fillText(c, pad.l + (j + .5) * cw, pad.t - 5)); }
      if (o.rows) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; o.rows.forEach((rr, i) => ctx.fillText(rr, pad.l - 6, pad.t + (i + .5) * ch)); }
      ctx.restore();
      return P;
    };

    /* iso-contours by marching squares on a sampled grid */
    P.contours = function (f, levels, o) {
      o = o || {};
      const nx = o.nx || 90, ny = o.ny || 70;
      const g = [];
      for (let j = 0; j <= ny; j++) {
        const row = [];
        for (let i = 0; i <= nx; i++) row.push(f(xd[0] + (xd[1] - xd[0]) * i / nx, yd[0] + (yd[1] - yd[0]) * j / ny));
        g.push(row);
      }
      ctx.save();
      ctx.strokeStyle = o.color || T.faint; ctx.lineWidth = o.width || 1; ctx.globalAlpha = o.alpha != null ? o.alpha : .8;
      levels.forEach(L => {
        ctx.beginPath();
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const x0 = xd[0] + (xd[1] - xd[0]) * i / nx, x1 = xd[0] + (xd[1] - xd[0]) * (i + 1) / nx;
          const y0 = yd[0] + (yd[1] - yd[0]) * j / ny, y1 = yd[0] + (yd[1] - yd[0]) * (j + 1) / ny;
          const v = [g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]];
          const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
          const segs = [];
          for (let k = 0; k < 4; k++) {
            const a = v[k], b = v[(k + 1) % 4];
            if ((a - L) * (b - L) < 0) {
              const t = (L - a) / (b - a);
              const pa = pts[k], pb = pts[(k + 1) % 4];
              segs.push([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])]);
            }
          }
          if (segs.length === 2) { ctx.moveTo(P.x(segs[0][0]), P.y(segs[0][1])); ctx.lineTo(P.x(segs[1][0]), P.y(segs[1][1])); }
        }
        ctx.stroke();
      });
      ctx.restore();
      return P;
    };

    return P;
  }

  /* Draw pixel data at CSS-pixel coordinates.
     putImageData bypasses the canvas transform (including the devicePixelRatio
     scale), so on a retina display it lands at half position and half size.
     Routing through an offscreen canvas lets drawImage honour the transform. */
  const _scratch = document.createElement('canvas');
  function blit(ctx, img, x, y, w, h) {
    _scratch.width = img.width; _scratch.height = img.height;
    const sc = _scratch.getContext('2d');
    if (!sc) return;
    sc.putImageData(img, 0, 0);
    ctx.drawImage(_scratch, x, y, w != null ? w : img.width, h != null ? h : img.height);
  }

  function defaultRamp(T) {
    // blue -> neutral -> red, theme aware
    const dark = T.dark;
    return function (t) {
      const a = dark ? 40 : 30;
      if (t < .5) { const k = t * 2; return [Math.round(60 + 120 * k), Math.round(90 + 100 * k), 230, a + 60]; }
      const k = (t - .5) * 2;
      return [230, Math.round(190 - 90 * k), Math.round(190 - 150 * k), a + 60];
    };
  }

  function ticks(a, b, n) {
    if (!(b > a)) return [a];
    const span = b - a, step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : +v.toFixed(10));
    return out;
  }
  function fmt(v, f) {
    if (f) return f(v);
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e6 || a < 1e-3) return v.toExponential(0).replace('e+', 'e');
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1).replace(/\.0$/, '');
    return v.toFixed(2).replace(/0$/, '').replace(/\.$/, '');
  }
  const r = v => Math.round(v) + .5;

  /* ---------------- controls ---------------- */
  /* spec: [{k:'alpha', label:'α', type:'range', min,max,step,value, fmt}] */
  function controls(host, spec, onChange) {
    const state = {};
    const wrap = el('div', { class: 'controls' });
    const refs = {};
    spec.forEach(c => {
      state[c.k] = c.value;
      if (c.type === 'select') {
        const box = el('div', { class: 'ctrl' });
        box.appendChild(el('label', null, [el('span', { text: c.label })]));
        const sel = el('select');
        c.options.forEach(o => sel.appendChild(el('option', { value: o.v !== undefined ? o.v : o, selected: (o.v !== undefined ? o.v : o) === c.value ? '' : null, text: o.t || o })));
        sel.value = c.value;
        sel.addEventListener('change', () => { state[c.k] = sel.value; onChange(state, c.k); });
        box.appendChild(sel); wrap.appendChild(box); refs[c.k] = sel;
      } else if (c.type === 'toggle') {
        const box = el('div', { class: 'ctrl' });
        const row = el('label', { class: 'switchrow' });
        const cb = el('input', { type: 'checkbox' }); cb.checked = !!c.value;
        cb.addEventListener('change', () => { state[c.k] = cb.checked; onChange(state, c.k); });
        row.appendChild(cb); row.appendChild(el('span', { text: c.label }));
        box.appendChild(row); wrap.appendChild(box); refs[c.k] = cb;
      } else if (c.type === 'buttons') {
        const box = el('div', { class: 'ctrl' });
        box.appendChild(el('label', null, [el('span', { text: c.label })]));
        const seg = el('div', { class: 'seg' });
        c.options.forEach(o => {
          const v = o.v !== undefined ? o.v : o;
          const b = el('button', { type: 'button', class: v === c.value ? 'on' : '', text: o.t || o });
          b.addEventListener('click', () => {
            state[c.k] = v;
            Array.prototype.forEach.call(seg.children, x => x.classList.remove('on'));
            b.classList.add('on');
            onChange(state, c.k);
          });
          seg.appendChild(b);
        });
        box.appendChild(seg); wrap.appendChild(box); refs[c.k] = seg;
      } else {
        const box = el('div', { class: 'ctrl' });
        const val = el('b', { text: (c.fmt ? c.fmt(c.value) : String(c.value)) });
        box.appendChild(el('label', null, [el('span', { html: c.label }), val]));
        const inp = el('input', { type: 'range', min: c.min, max: c.max, step: c.step != null ? c.step : 'any', value: c.value });
        inp.addEventListener('input', () => {
          state[c.k] = +inp.value;
          val.textContent = c.fmt ? c.fmt(+inp.value) : inp.value;
          onChange(state, c.k);
        });
        box.appendChild(inp); wrap.appendChild(box);
        refs[c.k] = inp; refs[c.k + '_val'] = val;
      }
    });
    host.appendChild(wrap);
    state.$set = function (k, v) {
      const ref = refs[k]; if (!ref) return;
      state[k] = v;
      if (ref.type === 'checkbox') ref.checked = v; else if (ref.tagName === 'INPUT' || ref.tagName === 'SELECT') ref.value = v;
      const lv = refs[k + '_val']; const c = spec.find(s => s.k === k);
      if (lv) lv.textContent = c && c.fmt ? c.fmt(v) : String(v);
    };
    state.$refs = refs;
    return state;
  }

  /* readout stats: [{k,label,fmt,cls}] -> update(obj) */
  function readout(host, spec) {
    const wrap = el('div', { class: 'readout' });
    const refs = {};
    spec.forEach(s => {
      const b = el('b', { text: '—' });
      const box = el('div', { class: 'stat' + (s.cls ? ' ' + s.cls : '') }, [el('i', { text: s.label }), b]);
      wrap.appendChild(box); refs[s.k] = { b: b, box: box, spec: s };
    });
    host.appendChild(wrap);
    return function (obj) {
      Object.keys(obj).forEach(k => {
        const rf = refs[k]; if (!rf) return;
        rf.b.innerHTML = rf.spec.fmt ? rf.spec.fmt(obj[k]) : obj[k];
        if (rf.spec.color) { rf.box.className = 'stat ' + rf.spec.color(obj[k]); }
      });
    };
  }

  function buttons(host, list) {
    const row = el('div', { class: 'btnrow' });
    const refs = {};
    list.forEach(b => {
      const n = el('button', { class: 'btn' + (b.primary ? ' primary' : ''), type: 'button', text: b.label, onclick: b.on });
      row.appendChild(n); refs[b.k || b.label] = n;
    });
    host.appendChild(row);
    return refs;
  }

  function legend(host, items) {
    const w = el('div', { class: 'legend' });
    items.forEach(i => w.appendChild(el('span', { html: '<i style="background:' + i.c + '"></i>' + i.t })));
    host.appendChild(w);
    return w;
  }

  function note(host, html) { host.appendChild(el('p', { class: 'lab-note', style: 'padding:0;margin:10px 0 0', html: html })); }

  /* animation loop helper — auto-stops on section change */
  function loop(fn) {
    let id = 0, last = performance.now(), stopped = false;
    function step(t) {
      if (stopped) return;
      const dt = Math.min(64, t - last); last = t;
      fn(dt);
      id = requestAnimationFrame(step);
    }
    id = requestAnimationFrame(step);
    const stop = () => { stopped = true; cancelAnimationFrame(id); };
    ML.onCleanup(stop);
    return stop;
  }

  /* pointer interaction on a surface: cb({x,y,down,type}) in canvas px */
  function pointer(S, cb) {
    const cv = S.canvas;
    let down = false;
    function pos(e) {
      const r = cv.getBoundingClientRect();
      const t = e.touches ? e.touches[0] : e;
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    }
    function on(type) {
      return function (e) {
        if (type === 'down') { down = true; cv.setPointerCapture && e.pointerId !== undefined && cv.setPointerCapture(e.pointerId); }
        if (type === 'up') down = false;
        if (type === 'move' && !down && !cb.hover) return;
        const p = pos(e);
        cb(Object.assign(p, { down: down, type: type }));
        if (down || type === 'down') e.preventDefault();
      };
    }
    cv.addEventListener('pointerdown', on('down'));
    window.addEventListener('pointermove', on('move'));
    window.addEventListener('pointerup', on('up'));
    ML.onCleanup(() => { window.removeEventListener('pointermove', on('move')); window.removeEventListener('pointerup', on('up')); });
  }

  /* ---------------- animation transport ----------------
     A play/pause/step/scrub bar over a finite number of frames. The caller
     owns the drawing; the player only owns the frame index, so scrubbing
     backwards is always exact rather than a re-simulation. */
  function player(host, opts) {
    opts = opts || {};
    const total = opts.frames;
    let i = opts.start || 0, playing = false, acc = 0;
    const fps = opts.fps || 12;

    const row = el('div', { class: 'btnrow' });
    const playBtn = el('button', { class: 'btn primary', type: 'button', text: '▶ Play' });
    const stepB = el('button', { class: 'btn', type: 'button', text: 'Step ▸' });
    const backB = el('button', { class: 'btn', type: 'button', text: '◂ Back' });
    const resetB = el('button', { class: 'btn ghost', type: 'button', text: 'Reset' });
    const label = el('span', { class: 'small', style: 'font-family:var(--mono);margin-left:4px' });
    const slider = el('input', { type: 'range', min: 0, max: String(total - 1), step: '1', value: String(i), style: 'flex:1 1 140px;min-width:120px' });

    function emit() {
      label.textContent = (opts.label ? opts.label(i) : ('frame ' + (i + 1) + ' / ' + total));
      slider.value = String(i);
      opts.onFrame(i);
    }
    function setPlaying(v) {
      playing = v;
      playBtn.textContent = v ? '❚❚ Pause' : '▶ Play';
      playBtn.classList.toggle('primary', !v);
    }
    playBtn.addEventListener('click', () => setPlaying(!playing));
    stepB.addEventListener('click', () => { setPlaying(false); i = Math.min(total - 1, i + 1); emit(); });
    backB.addEventListener('click', () => { setPlaying(false); i = Math.max(0, i - 1); emit(); });
    resetB.addEventListener('click', () => { setPlaying(false); i = 0; emit(); });
    slider.addEventListener('input', () => { setPlaying(false); i = +slider.value; emit(); });

    row.appendChild(playBtn); row.appendChild(backB); row.appendChild(stepB); row.appendChild(resetB);
    row.appendChild(slider); row.appendChild(label);
    host.appendChild(row);

    loop(function (dt) {
      if (!playing) return;
      acc += dt;
      if (acc < 1000 / fps) return;
      acc = 0;
      if (i >= total - 1) { if (opts.repeat) i = 0; else { setPlaying(false); return; } }
      else i++;
      emit();
    });
    emit();
    return { get frame() { return i; }, set: n => { i = Math.max(0, Math.min(total - 1, n)); emit(); }, stop: () => setPlaying(false) };
  }

  /* ---------------- boxes-and-arrows diagram ----------------
     nodes: [{id,x,y,w,h,label,sub,fill,accent}] in 0..1 space,
     edges: [{from,to,label,dash,curve}] */
  function flow(ctx, w, h, nodes, edges, opts) {
    opts = opts || {};
    const T = theme();
    const pad = opts.pad || 10;
    const NX = id => { const n = map[id]; return { x: pad + n.x * (w - 2 * pad), y: pad + n.y * (h - 2 * pad), w: n.w * (w - 2 * pad), h: n.h * (h - 2 * pad) }; };
    const map = {};
    nodes.forEach(n => { map[n.id] = n; });

    ctx.save();
    (edges || []).forEach(e => {
      const a = NX(e.from), b = NX(e.to);
      const ax = a.x + a.w, ay = a.y + a.h / 2;
      const bx = b.x, by = b.y + b.h / 2;
      const vertical = Math.abs(bx - (a.x + a.w)) < 6 && Math.abs(by - ay) > 20;
      ctx.strokeStyle = e.color || T.faint; ctx.lineWidth = e.width || 1.4;
      ctx.setLineDash(e.dash || []);
      ctx.beginPath();
      if (vertical || e.curve === 'v') {
        const sx = a.x + a.w / 2, sy = a.y + a.h, ex = b.x + b.w / 2, ey = b.y;
        ctx.moveTo(sx, sy); ctx.bezierCurveTo(sx, (sy + ey) / 2, ex, (sy + ey) / 2, ex, ey);
        arrowHead(ctx, ex, ey, Math.PI / 2, e.color || T.faint);
      } else {
        ctx.moveTo(ax, ay); ctx.bezierCurveTo((ax + bx) / 2, ay, (ax + bx) / 2, by, bx, by);
        arrowHead(ctx, bx, by, Math.atan2(by - ay, 1), e.color || T.faint);
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (e.label) {
        const mx = vertical ? (a.x + a.w / 2) : (ax + bx) / 2, my = vertical ? (a.y + a.h + b.y) / 2 : (ay + by) / 2;
        ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const tw = ctx.measureText(e.label).width + 8;
        ctx.fillStyle = T.paper; ctx.fillRect(mx - tw / 2, my - 7, tw, 14);
        ctx.fillStyle = e.color || T.muted; ctx.fillText(e.label, mx, my);
      }
    });

    nodes.forEach(n => {
      const b = NX(n.id);
      ctx.fillStyle = n.fill || T.panel;
      ctx.strokeStyle = n.accent || T.line;
      ctx.lineWidth = n.accent ? 1.8 : 1;
      roundRectPath(ctx, b.x, b.y, b.w, b.h, n.r == null ? 8 : n.r);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = n.textColor || T.text;
      ctx.font = (n.bold === false ? '' : '600 ') + (n.font || '11.5px') + ' ui-sans-serif, system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = n.sub ? 'bottom' : 'middle';
      ctx.fillText(n.label, b.x + b.w / 2, b.y + b.h / 2 + (n.sub ? -1 : 0));
      if (n.sub) {
        ctx.font = '9.5px ui-monospace, monospace'; ctx.fillStyle = T.faint; ctx.textBaseline = 'top';
        ctx.fillText(n.sub, b.x + b.w / 2, b.y + b.h / 2 + 3);
      }
    });
    ctx.restore();
  }

  function arrowHead(ctx, x, y, a, color) {
    ctx.save(); ctx.fillStyle = color; ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - 7 * Math.cos(a - .4), y - 7 * Math.sin(a - .4));
    ctx.lineTo(x - 7 * Math.cos(a + .4), y - 7 * Math.sin(a + .4));
    ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  return { surface, plot, controls, readout, buttons, legend, note, loop, pointer, theme, ticks, fmt, blit, player, flow };
})();
