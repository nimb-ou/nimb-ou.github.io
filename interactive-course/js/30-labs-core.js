/* ============================================================
   Content-authoring helpers + shared lab drawing utilities
   ============================================================ */
(function () {
  'use strict';

  const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  /* ---- HTML template helpers used by every content file ---- */
  window.H = {
    lab: function (key, title, note) {
      return '<div class="lab"><div class="lab-head"><span class="tag">interactive</span><h4>' + title + '</h4></div>' +
        '<div class="lab-body" data-lab="' + key + '"></div>' +
        (note ? '<div class="lab-note">' + note + '</div>' : '') + '</div>';
    },

    box: function (title, html, kind, icon) {
      return '<div class="box ' + (kind || '') + '"><p class="boxtitle"' + (icon ? ' data-icon="' + icon + '"' : '') + '>' + title + '</p>' + html + '</div>';
    },

    /* the recurring voices of the site, each with its own colour and glyph */
    worked:    function (title, html) { return H.box(title || 'worked number', html, 'worked', '∑'); },
    intuition: function (html, title) { return H.box(title || 'the intuition', html, 'intuition', '◎'); },
    analogy:   function (html, title) { return H.box(title || 'an analogy', html, 'analogy', '≈'); },
    practice:  function (html, title) { return H.box(title || 'in practice', html, 'practice', '⚙'); },
    history:   function (html, title) { return H.box(title || 'how we got here', html, 'history', '⌛'); },
    pitfall:   function (html, title) { return H.box(title || 'the trap', html, 'pitfall', '⚠'); },
    warn:      function (title, html) { return H.box(title, html, 'warn', '⚠'); },

    tldr: function (points) {
      return H.box('in one breath', '<ul>' + points.map(p => '<li>' + p + '</li>').join('') + '</ul>', 'tldr', '➜');
    },

    probe: function (qas, mistake) {
      let h = '<div class="box probe"><p class="boxtitle" data-icon="?">what interviewers probe</p>';
      qas.forEach(qa => { h += '<p class="qa"><span class="q">Q.</span> ' + qa[0] + ' <span class="q">A.</span> ' + qa[1] + '</p>'; });
      if (mistake) h += '<p class="mistake">' + mistake + '</p>';
      return h + '</div>';
    },

    /* full interview bank: each entry {q, a, level, follow:[], red} */
    iq: function (title, items) {
      let h = '<div class="iq"><div class="iq-head"><span class="tag">interview</span><h4>' + title +
        '</h4><span class="cnt">' + items.length + ' question' + (items.length > 1 ? 's' : '') + '</span></div>';
      items.forEach((it, i) => {
        h += '<details class="iq-item"><summary><span class="qn">' + (i + 1) + '.</span><span>' + it.q + '</span>' +
          (it.level ? '<span class="lvl">' + it.level + '</span>' : '') + '</summary>' +
          '<div class="iq-body"><p class="lab-answer">how to answer</p>' + it.a;
        if (it.follow && it.follow.length) {
          h += '<div class="followups"><b>they will follow up with</b><ul>' + it.follow.map(f => '<li>' + f + '</li>').join('') + '</ul></div>';
        }
        if (it.red) h += '<p class="redflag">' + it.red + '</p>';
        h += '</div></details>';
      });
      return h + '</div>';
    },

    /* a derivation as a ladder: line of algebra on the left, the reason on the right */
    deriv: function (goal, steps, foot) {
      let h = '<div class="deriv"><div class="dhead">derivation<span class="goal">' + goal + '</span></div>';
      steps.forEach((s, i) => {
        h += '<div class="dstep"><div class="di">' + (i + 1) + '</div><div class="dm">' + s[0] + '</div>' +
          '<div class="dw">' + (s[1] || '') + '</div></div>';
      });
      if (foot) h += '<div class="dfoot">' + foot + '</div>';
      return h + '</div>';
    },

    /* optional depth — collapsed by default so the main line stays readable */
    more: function (summary, html) {
      return '<details class="more"><summary>' + summary + '</summary><div class="mbody">' + html + '</div></details>';
    },

    tabs: function (panes) {
      let bar = '', body = '';
      panes.forEach((p, i) => {
        bar += '<button type="button" class="' + (i === 0 ? 'on' : '') + '">' + p[0] + '</button>';
        body += '<div class="tabs-pane"' + (i === 0 ? '' : ' hidden') + '>' + p[1] + '</div>';
      });
      return '<div class="tabs"><div class="tabs-bar">' + bar + '</div>' + body + '</div>';
    },

    vs: function (titleA, a, titleB, b) {
      const li = x => Array.isArray(x) ? '<ul>' + x.map(i => '<li>' + i + '</li>').join('') + '</ul>' : x;
      return '<div class="vs"><div><h5>' + titleA + '</h5>' + li(a) + '</div><div><h5>' + titleB + '</h5>' + li(b) + '</div></div>';
    },

    note: function (html) { return '<p class="margin-note">↳ ' + html + '</p>'; },
    flag: function (html) { return '<p class="flagged">' + html + '</p>'; },
    key: function (html) { return '<p class="keyline"><mark>' + html + '</mark></p>'; },

    fig: function (label, inner, caption) {
      return '<figure><div class="figframe"><p class="figlabel">' + label + '</p>' + inner + '</div>' +
        (caption ? '<figcaption>' + caption + '</figcaption>' : '') + '</figure>';
    },

    /* inline SVG diagram — the frame supplies the border, caption and label */
    svg: function (label, vb, body, caption) {
      return H.fig(label, '<svg viewBox="' + vb + '" xmlns="http://www.w3.org/2000/svg" role="img">' + body + '</svg>', caption);
    },

    table: function (head, rows, cls) {
      let h = '<div class="tablewrap"><table' + (cls ? ' class="' + cls + '"' : '') + '><thead><tr>';
      head.forEach(c => h += '<th>' + c + '</th>');
      h += '</tr></thead><tbody>';
      rows.forEach(r => { h += '<tr>'; r.forEach(c => h += '<td>' + c + '</td>'); h += '</tr>'; });
      return h + '</tbody></table></div>';
    },

    code: function (src) { return '<pre><code>' + esc(src) + '</code></pre>'; },
    cols: function (a, b) { return '<div class="grid2"><div>' + a + '</div><div>' + b + '</div></div>'; },
    steps: function (items) { return '<ol class="steps">' + items.map(i => '<li>' + i + '</li>').join('') + '</ol>'; },
    checklist: function (items) { return '<ul class="checklist">' + items.map(i => '<li>' + i + '</li>').join('') + '</ul>'; },

    /* write-the-function drill; mount with Labs.codeDrill */
    drill: function (key, title) {
      return '<div class="drillbox"><div class="dh"><span class="tag">code drill</span><b>' + title + '</b></div>' +
        '<div class="db" data-lab="' + key + '"></div></div>';
    }
  };

  /* ---- shared drawing helpers for labs ---- */
  window.Labs = {
    /* decision surface for a 2-D classifier: decide(x,y) -> score in [0,1] */
    boundary: function (P, decide, opts) {
      opts = opts || {};
      const T = P.T, mid = opts.mid == null ? .5 : opts.mid;
      const c0 = opts.c0 || [88, 120, 220], c1 = opts.c1 || [220, 90, 80];
      P.field(function (x, y) { return decide(x, y); }, {
        step: opts.step || 3, lo: opts.lo, hi: opts.hi,
        colors: function (t) {
          const s = Math.max(0, Math.min(1, t));
          const a = opts.alpha == null ? 78 : opts.alpha;
          const mixed = s < .5 ? c0 : c1;
          const strength = Math.abs(s - .5) * 2;
          return [mixed[0], mixed[1], mixed[2], Math.round(a * (0.25 + 0.75 * strength))];
        }
      });
      if (opts.contour !== false) P.contours(decide, [mid], { color: T.text, width: 1.6, alpha: .85 });
      return P;
    },

    /* two-class scatter */
    points: function (P, X, y, opts) {
      opts = opts || {};
      const T = P.T;
      const c0 = opts.c0 || T.blue, c1 = opts.c1 || T.red;
      X.forEach((p, i) => {
        P.dots([[p[0], p[1]]], {
          r: opts.r || 3.6,
          color: (opts.color ? opts.color(i) : (y[i] ? c1 : c0)),
          stroke: opts.stroke === undefined ? true : opts.stroke,
          strokeWidth: 1.3, alpha: 1
        });
      });
      return P;
    },

    /* palette for k clusters — the site's chart series, in order */
    palette: function (k) {
      const T = Viz.theme();
      const base = [T.c1, T.c2, T.c3, T.c4, T.c5, T.c6, T.c7, T.c8];
      return Array.from({ length: k }, (_, i) => base[i % base.length]);
    },

    /* draw a small tree diagram (node objects from Num.tree) */
    drawTree: function (ctx, node, x, y, w, T, featNames, depthLimit) {
      const boxW = Math.min(120, w / 2.2), boxH = 26;
      function walk(n, cx, cy, span, d) {
        const leaf = !n.left || d >= (depthLimit == null ? 9 : depthLimit);
        ctx.fillStyle = leaf ? (n.value > .5 ? 'rgba(220,90,80,.22)' : 'rgba(88,120,220,.22)') : T.panel;
        ctx.strokeStyle = T.line; ctx.lineWidth = 1;
        roundRect(ctx, cx - boxW / 2, cy, boxW, boxH, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = T.text; ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const label = leaf ? ('p = ' + n.value.toFixed(2)) : ((featNames ? featNames[n.f] : 'x' + n.f) + ' ≤ ' + n.thr.toFixed(2));
        ctx.fillText(label, cx, cy + boxH / 2);
        if (leaf) return;
        const ny = cy + 58;
        [[n.left, cx - span / 2, 'yes'], [n.right, cx + span / 2, 'no']].forEach(([ch, nx, tag]) => {
          ctx.strokeStyle = T.faint; ctx.beginPath();
          ctx.moveTo(cx, cy + boxH); ctx.lineTo(nx, ny); ctx.stroke();
          ctx.fillStyle = T.faint; ctx.font = '9px ui-monospace, monospace';
          ctx.fillText(tag, (cx + nx) / 2 + (tag === 'yes' ? -10 : 10), (cy + boxH + ny) / 2);
          walk(ch, nx, ny, span / 2, d + 1);
        });
      }
      walk(node, x, y, w / 2.1, 0);
    },

    /* ---- write-a-function drill ----------------------------------------
       spec: { brief, signature, starter, tests:[{name, run(fn)->true|string}],
               solution, hint } */
    codeDrill: function (host, spec) {
      const el = ML.el;
      if (spec.brief) host.appendChild(el('p', { class: 'small', html: spec.brief, style: 'margin-top:0' }));
      if (spec.signature) host.appendChild(el('p', { html: '<code>' + spec.signature + '</code>', style: 'margin:0 0 10px' }));
      const ta = el('textarea', { spellcheck: 'false', text: spec.starter || '' });
      host.appendChild(ta);
      const list = el('ul', { class: 'testlist' });
      const msg = el('p', { class: 'small', style: 'margin:10px 0 0' });

      function run() {
        list.innerHTML = ''; msg.innerHTML = '';
        let fn;
        try { fn = (new Function('"use strict";return (' + ta.value + ');'))(); }
        catch (e) { msg.innerHTML = '<span style="color:var(--red)">Syntax error: ' + ML.escapeHtml(e.message) + '</span>'; return; }
        if (typeof fn !== 'function') { msg.innerHTML = '<span style="color:var(--red)">That did not evaluate to a function. Leave it as an expression — no trailing semicolon after the closing brace.</span>'; return; }
        let pass = 0;
        spec.tests.forEach(t => {
          let ok, detail = '';
          try { const r = t.run(fn); ok = r === true; if (!ok) detail = typeof r === 'string' ? r : 'returned the wrong value'; }
          catch (e) { ok = false; detail = e.message; }
          if (ok) pass++;
          list.appendChild(el('li', { class: ok ? 'pass' : 'fail' }, [
            el('span', { text: ok ? '✓' : '✗' }),
            el('span', { class: 'nm', text: t.name + (detail ? ' — ' + detail : '') })
          ]));
        });
        msg.innerHTML = pass === spec.tests.length
          ? '<b style="color:var(--green)">All ' + pass + ' tests pass.</b> Now say the complexity out loud — that is the next question.'
          : pass + ' of ' + spec.tests.length + ' passing.';
      }

      const row = el('div', { class: 'btnrow' });
      row.appendChild(el('button', { class: 'btn primary', type: 'button', text: 'Run tests', onclick: run }));
      if (spec.hint) row.appendChild(el('button', {
        class: 'btn', type: 'button', text: 'Hint',
        onclick: function () { msg.innerHTML = '<b>Hint.</b> ' + spec.hint; }
      }));
      if (spec.solution) row.appendChild(el('button', {
        class: 'btn', type: 'button', text: 'Show a solution',
        onclick: function () { ta.value = spec.solution; run(); }
      }));
      row.appendChild(el('button', { class: 'btn ghost', type: 'button', text: 'Reset', onclick: function () { ta.value = spec.starter || ''; list.innerHTML = ''; msg.innerHTML = ''; } }));
      host.appendChild(row);
      host.appendChild(list);
      host.appendChild(msg);
    }
  };

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  window.Labs.roundRect = roundRect;
})();
