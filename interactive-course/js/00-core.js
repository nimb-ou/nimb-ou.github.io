/* ============================================================
   Core: registry, router, navigation, progress, quiz, search
   ============================================================ */
window.ML = (function () {
  'use strict';

  const tracks = [];
  const sections = [];
  const byId = Object.create(null);

  /* ---------- registration ---------- */
  function track(t) { tracks.push(Object.assign({ sections: [] }, t)); return t; }

  function section(s) {
    if (byId[s.id]) { console.warn('duplicate section id', s.id); return; }
    const t = tracks.find(x => x.id === s.track);
    if (!t) { console.warn('unknown track', s.track, 'for', s.id); return; }
    s.trackRef = t;
    t.sections.push(s);
    sections.push(s);
    byId[s.id] = s;
    return s;
  }

  /* ---------- storage ---------- */
  const KEY = 'aiml-notebook-v1';
  let store = { done: {}, quiz: {}, theme: null, collapsed: {}, known: {} };
  try { Object.assign(store, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}
  if (!store.known) store.known = {};
  const save = debounce(() => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }, 250);

  function debounce(fn, ms) { let t; return function () { clearTimeout(t); const a = arguments; t = setTimeout(() => fn.apply(null, a), ms); }; }

  /* ---------- dom helpers ---------- */
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));
  function el(tag, attrs, kids) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(k => n.appendChild(typeof k === 'string' ? document.createTextNode(k) : k));
    return n;
  }

  /* ---------- math ---------- */
  function typeset(root) {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false }
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
        throwOnError: false,
        strict: false
      });
    } catch (e) { console.warn('katex', e); }
  }

  /* ---------- navigation tree ---------- */
  function buildNav() {
    const host = $('#navTree');
    host.innerHTML = '';
    tracks.forEach((t, i) => {
      const wrap = el('div', { class: 'navtrack t' + ((i % 8) + 1) + (store.collapsed[t.id] ? ' collapsed' : ''), 'data-track': t.id });
      const btn = el('button', { type: 'button' }, [
        el('span', { class: 'tracknum', text: t.short || ('T' + i) }),
        el('span', { class: 'tname', text: t.title }),
        el('span', { class: 'chev', text: '▾' })
      ]);
      btn.addEventListener('click', () => {
        wrap.classList.toggle('collapsed');
        store.collapsed[t.id] = wrap.classList.contains('collapsed'); save();
      });
      const bar = el('div', { class: 'trackbar' }, [el('i', { 'data-bar': t.id })]);
      const ul = el('ul', { class: 'navlist' });
      t.sections.forEach(s => {
        const a = el('a', { href: '#/' + s.id, 'data-nav': s.id, 'data-find': (s.num + ' ' + s.title).toLowerCase(), class: store.done[s.id] ? 'done' : '' }, [
          el('span', { class: 'n', text: s.num }),
          el('span', { text: s.title })
        ]);
        ul.appendChild(el('li', null, [a]));
      });
      wrap.appendChild(btn); wrap.appendChild(bar); wrap.appendChild(ul);
      host.appendChild(wrap);
    });
    updateProgress();
  }

  function updateProgress() {
    const total = sections.length;
    const done = sections.filter(s => store.done[s.id]).length;
    const pct = total ? Math.round(100 * done / total) : 0;
    const arc = $('#progArc'); if (arc) arc.style.strokeDashoffset = String(100 - pct);
    const p = $('#progPct'); if (p) p.textContent = pct + '%';
    const d = $('#progDone'); if (d) d.textContent = done + ' of ' + total + ' sections';
    $$('#navTree a[data-nav]').forEach(a => a.classList.toggle('done', !!store.done[a.getAttribute('data-nav')]));
    tracks.forEach(t => {
      const bar = $('[data-bar="' + t.id + '"]');
      if (!bar) return;
      const dn = t.sections.filter(s => store.done[s.id]).length;
      bar.style.width = (t.sections.length ? 100 * dn / t.sections.length : 0) + '%';
    });
  }

  function markDone(id, val) { if (val) store.done[id] = 1; else delete store.done[id]; save(); updateProgress(); }
  function isDone(id) { return !!store.done[id]; }

  /* ---------- rendering ---------- */
  let currentCleanup = [];
  function onCleanup(fn) { currentCleanup.push(fn); }

  function render(id) {
    currentCleanup.forEach(fn => { try { fn(); } catch (e) {} });
    currentCleanup = [];
    const main = $('#main');
    main.innerHTML = '';

    if (!id || id === '/' || id === '') { renderHome(main); afterRender(main, null); return; }
    const s = byId[id];
    if (!s) {
      main.appendChild(el('h1', { text: 'Not found' }));
      main.appendChild(el('p', { html: 'No section with id <code>' + escapeHtml(id) + '</code>. <a href="#/">Back to the map</a>.' }));
      afterRender(main, null); return;
    }

    const t = s.trackRef;
    main.appendChild(el('p', { class: 'eyebrow' }, [
      el('a', { href: '#/', text: 'Home' }), el('span', { class: 'sep', text: '/' }),
      el('a', { href: '#/' + (t.sections[0] ? t.sections[0].id : ''), text: t.short + ' · ' + t.title })
    ]));
    main.appendChild(el('h1', { html: '<span class="hnum">' + s.num + '</span>' + s.title }));
    if (s.lede) main.appendChild(el('p', { class: 'lede', html: s.lede }));
    if (s.rests) main.appendChild(el('p', { class: 'rests', html: s.rests }));
    main.appendChild(buildMeta(s));

    const body = el('div', { class: 'secbody', html: s.html || '' });

    // in-page contents, derived from the h2s the author wrote
    const heads = Array.prototype.slice.call(body.querySelectorAll('h2'));
    heads.forEach((h, i) => { if (!h.id) h.id = 'h-' + (i + 1); });
    if (heads.length > 2) main.appendChild(buildInlineToc(heads));

    main.appendChild(body);

    // labs
    if (s.labs) {
      Object.keys(s.labs).forEach(key => {
        const host = body.querySelector('[data-lab="' + key + '"]');
        if (!host) { console.warn('lab host missing', key, 'in', s.id); return; }
        try { s.labs[key](host, { onCleanup: onCleanup }); }
        catch (e) { console.error('lab failed', key, e); host.innerHTML = '<p class="small">This interactive failed to start: ' + escapeHtml(e.message) + '</p>'; }
      });
    }

    if (s.quiz && s.quiz.length) main.appendChild(buildQuiz(s));
    if (s.cards && s.cards.length) main.appendChild(buildCards(s));
    if (s.related && s.related.length) { const r = buildRelated(s); if (r) main.appendChild(r); }

    main.appendChild(buildDoneBar(s));
    main.appendChild(buildSecNav(s));
    if (heads.length > 2) main.appendChild(buildRail(heads));
    afterRender(main, s);
  }

  function readingTime(s) {
    const words = stripTags((s.lede || '') + ' ' + (s.html || '')).split(/\s+/).filter(Boolean).length;
    const labs = s.labs ? Object.keys(s.labs).length : 0;
    return Math.max(2, Math.round(words / 210 + labs * 2));
  }

  const LEVELS = { 1: ['foundational', 'lv1'], 2: ['core', 'lv2'], 3: ['advanced', 'lv3'] };

  function buildMeta(s) {
    const wrap = el('div', { class: 'secmeta' });
    const lv = LEVELS[s.level || 2];
    wrap.appendChild(el('span', { class: 'chip ' + lv[1], text: lv[0] }));
    wrap.appendChild(el('span', { class: 'chip', html: '<b>' + readingTime(s) + '</b> min' }));
    const nLabs = s.labs ? Object.keys(s.labs).length : 0;
    if (nLabs) wrap.appendChild(el('span', { class: 'chip', html: '<b>' + nLabs + '</b> lab' + (nLabs > 1 ? 's' : '') }));
    const nq = (s.quiz || []).length;
    if (nq) wrap.appendChild(el('span', { class: 'chip', html: '<b>' + nq + '</b> quiz Q' + (nq > 1 ? 's' : '') }));
    (s.prereq || []).forEach(pid => {
      const p = byId[pid];
      if (p) wrap.appendChild(el('span', { class: 'chip', html: 'needs <a href="#/' + p.id + '">' + p.num + ' ' + escapeHtml(p.title) + '</a>' }));
    });
    return wrap;
  }

  function buildInlineToc(heads) {
    const d = el('details', { class: 'inlinetoc' });
    d.appendChild(el('summary', { text: 'On this page' }));
    const ol = el('ol');
    heads.forEach(h => {
      const sn = h.querySelector('.sn');
      const label = h.textContent.replace(/^\s*[\d.]+\s*/, '').trim();
      ol.appendChild(el('li', null, [el('a', { href: '#' + h.id, html: (sn ? '<code>' + sn.textContent + '</code> ' : '') + escapeHtml(label) })]));
    });
    d.appendChild(ol);
    return d;
  }

  function buildRail(heads) {
    const rail = el('nav', { class: 'rail', 'aria-label': 'On this page' });
    const inner = el('div', { class: 'railinner' });
    rail.appendChild(inner);
    inner.appendChild(el('h5', { text: 'On this page' }));
    const ol = el('ol');
    heads.forEach(h => {
      const label = h.textContent.replace(/^\s*[\d.]+\s*/, '').trim();
      ol.appendChild(el('li', null, [el('a', { href: '#' + h.id, 'data-spy': h.id, text: label })]));
    });
    inner.appendChild(ol);
    if (window.IntersectionObserver) {
      const links = {};
      Array.prototype.forEach.call(ol.querySelectorAll('a'), a => { links[a.getAttribute('data-spy')] = a; });
      const seen = new Set();
      const io = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) seen.add(e.target.id); else seen.delete(e.target.id); });
        const first = heads.map(h => h.id).find(id => seen.has(id));
        Object.keys(links).forEach(id => links[id].classList.toggle('on', id === first));
      }, { rootMargin: '-70px 0px -65% 0px' });
      heads.forEach(h => io.observe(h));
      onCleanup(() => io.disconnect());
    }
    return rail;
  }

  function buildRelated(s) {
    const items = (s.related || []).map(id => byId[id]).filter(Boolean);
    if (!items.length) return null;
    const wrap = el('div', { class: 'related' });
    wrap.appendChild(el('p', { class: 'boxtitle', text: 'read next' }));
    const ul = el('ul');
    items.forEach(r => ul.appendChild(el('li', null, [
      el('a', { href: '#/' + r.id }, [el('span', { class: 'n', text: r.num }), el('span', { text: r.title })])
    ])));
    wrap.appendChild(ul);
    return wrap;
  }

  function afterRender(main, s) {
    typeset(main);
    highlightCode(main);
    addCopyButtons(main);
    initTabs(main);
    $$('#navTree a[data-nav]').forEach(a => a.classList.toggle('active', !!s && a.getAttribute('data-nav') === s.id));
    const act = $('#navTree a.active');
    if (act && act.scrollIntoView) {
      const sb = $('#sidebar');
      const r = act.getBoundingClientRect(), sr = sb.getBoundingClientRect();
      if (r.top < sr.top + 40 || r.bottom > sr.bottom - 40) act.scrollIntoView({ block: 'center' });
    }
    document.documentElement.scrollTop = 0; document.body.scrollTop = 0;
    main.focus({ preventScroll: true });
  }

  /* code blocks get a copy button, wrapped so the button can be positioned */
  function addCopyButtons(root) {
    $$('pre', root).forEach(pre => {
      if (pre.parentNode && pre.parentNode.classList.contains('codewrap')) return;
      const wrap = el('div', { class: 'codewrap' });
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      const btn = el('button', { class: 'copybtn', type: 'button', text: 'copy' });
      btn.addEventListener('click', () => {
        const text = pre.textContent;
        const done = () => { btn.textContent = 'copied'; btn.classList.add('done'); setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('done'); }, 1400); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => {});
        else { const ta = el('textarea', { text: text }); document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); done(); } catch (e) {} ta.remove(); }
      });
      wrap.appendChild(btn);
    });
  }

  function initTabs(root) {
    $$('.tabs', root).forEach(t => {
      const bar = t.querySelector('.tabs-bar');
      const panes = $$('.tabs-pane', t);
      if (!bar || !panes.length) return;
      const btns = $$('button', bar);
      btns.forEach((b, i) => b.addEventListener('click', () => {
        btns.forEach((x, j) => x.classList.toggle('on', i === j));
        panes.forEach((p, j) => { p.hidden = i !== j; });
        typeset(panes[i]);
      }));
    });
  }

  function buildDoneBar(s) {
    const bar = el('div', { class: 'donebar' });
    const cb = el('input', { type: 'checkbox', id: 'doneCb' });
    cb.checked = isDone(s.id);
    const txt = () => cb.checked ? '<b>Marked complete.</b> Press <kbd>j</kbd> for the next section.' : 'Mark this section complete <span class="small">(or press <kbd>m</kbd>)</span>';
    cb.addEventListener('change', () => { markDone(s.id, cb.checked); lbl.innerHTML = txt(); });
    const lbl = el('label', { for: 'doneCb', html: txt() });
    bar.appendChild(cb); bar.appendChild(lbl);
    return bar;
  }

  function buildSecNav(s) {
    const i = sections.indexOf(s);
    const prev = sections[i - 1], next = sections[i + 1];
    const nav = el('div', { class: 'secnav' });
    if (prev) nav.appendChild(el('a', { href: '#/' + prev.id }, [el('i', { text: '← previous · ' + prev.num }), el('b', { text: prev.title })]));
    if (next) nav.appendChild(el('a', { href: '#/' + next.id, class: 'next' }, [el('i', { text: 'next · ' + next.num + ' →' }), el('b', { text: next.title })]));
    return nav;
  }

  /* ---------- home ---------- */
  function renderHome(main) {
    const home = ML.homeHtml ? ML.homeHtml() : '<h1>The Interactive AI/ML Notebook</h1>';
    main.innerHTML = home;
    const cards = main.querySelector('#trackCards');
    if (cards) {
      tracks.forEach((t, i) => {
        const dn = t.sections.filter(s => store.done[s.id]).length;
        const a = el('a', { class: 'trackcard', href: '#/' + (t.sections[0] ? t.sections[0].id : ''), style: '--tc:var(--c' + ((i % 8) + 1) + ')' }, [
          el('span', { class: 'tnum', text: t.short + ' · ' + t.sections.length + ' sections' }),
          el('h3', { text: t.title }),
          el('p', { text: t.blurb || '' }),
          el('span', { class: 'meta' }, [
            el('span', { text: dn + ' / ' + t.sections.length }),
            el('span', { class: 'bar', html: '<i style="width:' + (t.sections.length ? 100 * dn / t.sections.length : 0) + '%"></i>' })
          ])
        ]);
        cards.appendChild(a);
      });
    }
    const idx = main.querySelector('#fullIndex');
    if (idx) {
      tracks.forEach(t => {
        const h = el('h3', { text: t.short + ' — ' + t.title });
        const ul = el('ul');
        t.sections.forEach(s => ul.appendChild(el('li', null, [
          el('a', { href: '#/' + s.id, html: '<code>' + s.num + '</code> ' + escapeHtml(s.title) }),
          el('span', { class: 'small', text: s.lede ? ' — ' + stripTags(s.lede).slice(0, 96) : '' })
        ])));
        idx.appendChild(h); idx.appendChild(ul);
      });
    }
    const counts = main.querySelector('#homeCounts');
    if (counts) {
      const st = ML.stats();
      counts.innerHTML =
        stat(st.sections, 'sections') + stat(st.labs, 'interactive labs') +
        stat(st.quiz, 'quiz questions') + stat(st.cards, 'recall cards') +
        stat(st.interview, 'interview questions') + stat(st.tracks, 'tracks');
    }
    if (ML.homeHero) { const hh = main.querySelector('#homeHero'); if (hh) { try { ML.homeHero(hh); } catch (e) {} } }
  }
  function stat(v, l) { return '<div class="stat key"><i>' + l + '</i><b>' + v + '</b></div>'; }

  function stats() {
    const labs = sections.reduce((a, s) => a + (s.labs ? Object.keys(s.labs).length : 0), 0);
    const quiz = sections.reduce((a, s) => a + (s.quiz ? s.quiz.length : 0), 0);
    const cards = sections.reduce((a, s) => a + (s.cards ? s.cards.length : 0), 0);
    const interview = sections.reduce((a, s) => a + (((s.html || '').match(/class="iq-item"/g) || []).length), 0);
    return { sections: sections.length, tracks: tracks.length, labs, quiz, cards, interview };
  }

  /* ---------- quiz ---------- */
  function buildQuiz(s) {
    const wrap = el('div', { class: 'quiz' });
    const score = el('span', { class: 'score', text: '' });
    wrap.appendChild(el('div', { class: 'quiz-head' }, [
      el('span', { class: 'tag', text: 'check yourself' }),
      el('h4', { text: 'Quiz — ' + s.num }),
      score
    ]));
    const body = el('div', { class: 'quiz-body' });
    let answered = 0, right = 0;

    function paint() {
      body.innerHTML = '';
      answered = 0; right = 0; score.textContent = '';
      s.quiz.forEach((q, qi) => {
        const item = el('div', { class: 'qitem' });
        item.appendChild(el('p', { class: 'qq', html: (qi + 1) + '. ' + q.q }));
        const opts = el('div', { class: 'qopts' });
        const why = el('p', { class: 'qwhy', html: q.why || '' });
        why.style.display = 'none';
        q.options.forEach((o, oi) => {
          const b = el('button', { class: 'qopt', type: 'button' }, [
            el('span', { class: 'k', text: 'ABCDE'[oi] }),
            el('span', { html: o })
          ]);
          b.addEventListener('click', () => {
            if (item.dataset.locked) return;
            item.dataset.locked = '1';
            answered++;
            const ok = oi === q.answer;
            if (ok) right++;
            b.classList.add(ok ? 'right' : 'wrong');
            if (!ok) { const cb = opts.children[q.answer]; if (cb) cb.classList.add('right'); }
            Array.prototype.forEach.call(opts.children, c => c.classList.add('locked'));
            why.style.display = '';
            typeset(why);
            score.textContent = right + ' / ' + answered + ' correct';
            store.quiz[s.id] = { right: right, answered: answered }; save();
            if (answered === s.quiz.length && right === s.quiz.length) markDone(s.id, true);
          });
          opts.appendChild(b);
        });
        item.appendChild(opts); item.appendChild(why);
        body.appendChild(item);
      });
      typeset(body);
    }
    paint();
    wrap.appendChild(body);
    const foot = el('div', { class: 'quiz-foot' }, [
      el('button', { class: 'btn ghost', type: 'button', text: 'Reset quiz', onclick: paint })
    ]);
    wrap.appendChild(foot);
    return wrap;
  }

  /* ---------- flashcards ---------- */
  function buildCards(s) {
    const wrap = el('div', { class: 'cards' });
    wrap.appendChild(el('p', { class: 'boxtitle', 'data-icon': '⚡', text: 'rapid recall — say the answer out loud, then flip' }));
    let order = s.cards.map((_, i) => i);
    let i = 0, showA = false;
    const face = el('div', { class: 'card-face', tabindex: '0', role: 'button', 'aria-label': 'Flashcard — click to flip' });
    const pos = el('span', { text: '' });
    function draw() {
      const c = s.cards[order[i]];
      face.innerHTML = showA ? '<div class="a">' + c.a + '</div>' : '<div><b>' + c.q + '</b></div>';
      pos.textContent = (i + 1) + ' / ' + s.cards.length;
      typeset(face);
    }
    const flip = () => { showA = !showA; draw(); };
    const go = d => { i = (i + d + order.length) % order.length; showA = false; draw(); };
    const btns = el('div', { class: 'cardnav' }, [
      el('button', { class: 'btn', type: 'button', text: '←', 'aria-label': 'Previous card', onclick: () => go(-1) }),
      el('button', { class: 'btn primary', type: 'button', text: 'Flip', onclick: flip }),
      pos,
      el('button', { class: 'btn', type: 'button', text: 'Shuffle', onclick: () => { order = shuffle(order.slice()); i = 0; showA = false; draw(); } }),
      el('button', { class: 'btn', type: 'button', text: '→', 'aria-label': 'Next card', onclick: () => go(1) })
    ]);
    face.addEventListener('click', flip);
    face.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flip(); } });
    wrap.appendChild(face);
    wrap.appendChild(btns);
    draw();
    return wrap;
  }

  function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; }

  /* ---------- search ---------- */
  let searchIndex = null;
  function buildSearchIndex() {
    searchIndex = sections.map(s => {
      const text = stripTags((s.lede || '') + ' ' + (s.html || '')).replace(/\s+/g, ' ');
      return { id: s.id, num: s.num, title: s.title, track: s.trackRef.short + ' · ' + s.trackRef.title, text: text, low: (s.num + ' ' + s.title + ' ' + text).toLowerCase() };
    });
  }
  function search(q) {
    if (!searchIndex) buildSearchIndex();
    q = q.trim().toLowerCase();
    if (!q) return [];
    const terms = q.split(/\s+/);
    const out = [];
    searchIndex.forEach(d => {
      let score = 0, hit = -1;
      terms.forEach(t => {
        const tl = d.title.toLowerCase().indexOf(t);
        if (tl >= 0) score += 40 - Math.min(tl, 20);
        const bl = d.low.indexOf(t);
        if (bl >= 0) { score += 6; if (hit < 0) hit = d.text.toLowerCase().indexOf(t); }
        else score -= 25;
      });
      if (score > 0) out.push({ d: d, score: score, hit: hit });
    });
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, 22).map(r => ({
      id: r.d.id, num: r.d.num, title: r.d.title, track: r.d.track,
      snip: r.hit >= 0 ? '…' + r.d.text.substr(Math.max(0, r.hit - 45), 130).trim() + '…' : r.d.text.slice(0, 120)
    }));
  }

  function initSearch() {
    const modal = $('#searchModal'), input = $('#searchInput'), results = $('#searchResults');
    input.placeholder = 'Search ' + sections.length + ' sections — try “KL divergence”, “KV cache”, “NDCG”…';
    let sel = 0, items = [];
    function open() { modal.hidden = false; input.value = ''; results.innerHTML = ''; input.focus(); }
    function close() { modal.hidden = true; }
    function mark(text, terms) {
      let h = escapeHtml(text);
      terms.forEach(t => { if (t.length > 1) h = h.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); });
      return h;
    }
    function draw() {
      const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      results.innerHTML = '';
      items.forEach((r, i) => {
        results.appendChild(el('a', { href: '#/' + r.id, class: i === sel ? 'sel' : '', onclick: close }, [
          el('span', { class: 'path', text: r.track + ' · ' + r.num }),
          el('span', { html: mark(r.title, terms) }),
          el('span', { class: 'snip', html: mark(r.snip, terms) })
        ]));
      });
      if (!items.length && input.value.trim()) results.innerHTML = '<div style="padding:14px;color:var(--muted);font-family:var(--sans);font-size:14px">Nothing matched. Try a formula name, an acronym, or a metric.</div>';
    }
    input.addEventListener('input', () => { items = search(input.value); sel = 0; draw(); });
    input.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, items.length - 1); draw(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); draw(); e.preventDefault(); }
      else if (e.key === 'Enter' && items[sel]) { location.hash = '#/' + items[sel].id; close(); }
      else if (e.key === 'Escape') close();
    });
    modal.addEventListener('click', e => { if (e.target === modal) close(); });
    $('#searchBtn').addEventListener('click', open);
    ML.openSearch = open;
    ML.closeSearch = close;
  }

  /* ---------- misc ---------- */
  function stripTags(h) { const d = document.createElement('div'); d.innerHTML = h; return d.textContent || ''; }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  function highlightCode(root) {
    $$('pre code', root).forEach(code => {
      if (code.dataset.hl) return;
      code.dataset.hl = '1';
      let h = escapeHtml(code.textContent);
      h = h.replace(/(#[^\n]*)/g, '<span class="tok-c">$1</span>');
      h = h.replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;)/g, '<span class="tok-s">$1</span>');
      h = h.replace(/\b(import|from|def|return|for|in|if|elif|else|while|class|with|as|lambda|not|and|or|None|True|False|yield|assert|try|except|raise|const|let|function|new|of|break|continue)\b/g, '<span class="tok-k">$1</span>');
      h = h.replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="tok-n">$1</span>');
      code.innerHTML = h;
    });
  }

  /* The OS preference (and, inside a viewer that stamps data-theme, the
     viewer's toggle) governs by default. We only stamp the attribute once the
     reader has explicitly chosen a theme here. */
  function effectiveTheme() {
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'dark' || attr === 'light') return attr;
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
  }

  function toggleTheme() {
    const nxt = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nxt);
    store.theme = nxt; save();
    document.dispatchEvent(new CustomEvent('ml:theme', { detail: nxt }));
  }

  function initTheme() {
    if (store.theme === 'dark' || store.theme === 'light') {
      document.documentElement.setAttribute('data-theme', store.theme);
    }
    $('#themeBtn').addEventListener('click', toggleTheme);
    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const onChange = () => { if (!store.theme) document.dispatchEvent(new CustomEvent('ml:theme', { detail: effectiveTheme() })); };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
    }
  }

  function initChrome() {
    const sidebar = $('#sidebar'), scrim = $('#navScrim');
    const closeNav = () => { sidebar.classList.remove('open'); scrim.classList.remove('on'); };
    $('#navToggle').addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      scrim.classList.toggle('on', open);
    });
    scrim.addEventListener('click', closeNav);
    document.addEventListener('click', e => {
      const a = e.target.closest ? e.target.closest('a[href^="#/"]') : null;
      if (a && window.innerWidth <= 1000) closeNav();
    });

    const filter = $('#navFilter');
    if (filter) filter.addEventListener('input', () => {
      const q = filter.value.trim().toLowerCase();
      $$('#navTree .navtrack').forEach(tr => {
        let any = false;
        $$('a[data-find]', tr).forEach(a => {
          const hit = !q || a.getAttribute('data-find').indexOf(q) >= 0;
          a.parentNode.classList.toggle('hide', !hit);
          if (hit) any = true;
        });
        tr.classList.toggle('hide', !any);
        if (q) tr.classList.remove('collapsed');
        else tr.classList.toggle('collapsed', !!store.collapsed[tr.getAttribute('data-track')]);
      });
    });

    $('#progReset').addEventListener('click', () => {
      if (!confirm('Clear progress for all sections in this browser?')) return;
      store.done = {}; store.quiz = {}; store.known = {}; save(); updateProgress(); render(location.hash.replace(/^#\//, ''));
    });
    window.addEventListener('scroll', () => {
      const h = document.documentElement;
      const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      const f = $('#readbarFill'); if (f) f.style.width = (p * 100).toFixed(1) + '%';
    }, { passive: true });

    const keys = $('#keySheet');
    const showKeys = v => { keys.hidden = !v; };
    $('#keysBtn').addEventListener('click', () => showKeys(keys.hidden));
    keys.addEventListener('click', () => showKeys(false));

    let gPending = false;
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') { ML.closeSearch && ML.closeSearch(); showKeys(false); closeNav(); }
      if (/input|textarea|select/i.test(e.target.tagName)) return;
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ML.openSearch(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '/') { e.preventDefault(); ML.openSearch(); return; }
      if (e.key === '?') { e.preventDefault(); showKeys(keys.hidden); return; }

      if (gPending) {
        gPending = false;
        if (e.key === 'h') { location.hash = '#/'; return; }
        if (e.key === 'd') { location.hash = '#/drill'; return; }
      }
      if (e.key === 'g') { gPending = true; setTimeout(() => { gPending = false; }, 900); return; }
      if (e.key === 't') { toggleTheme(); return; }

      const cur = byId[location.hash.replace(/^#\//, '')];
      if (!cur) return;
      const i = sections.indexOf(cur);
      if (e.key === 'j' && sections[i + 1]) location.hash = '#/' + sections[i + 1].id;
      if (e.key === 'k' && sections[i - 1]) location.hash = '#/' + sections[i - 1].id;
      if (e.key === 'm') { const cb = $('#doneCb'); if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change')); } }
    });
  }

  /* Content files load in whatever order the page lists them, and a later
     file may add a section that belongs in the middle of an earlier track.
     Order the curriculum by its printed number instead of by load order. */
  function orderSections() {
    const key = n => String(n).split('.').map(p => (/^\d+$/.test(p) ? +p : 0));
    const cmp = (a, b) => {
      const A = key(a.num), B = key(b.num);
      for (let i = 0; i < Math.max(A.length, B.length); i++) {
        if ((A[i] || 0) !== (B[i] || 0)) return (A[i] || 0) - (B[i] || 0);
      }
      return 0;
    };
    tracks.forEach(t => t.sections.sort(cmp));
    sections.length = 0;
    tracks.forEach(t => t.sections.forEach(s => sections.push(s)));
  }

  function boot() {
    orderSections();
    initTheme(); initChrome(); initSearch(); buildNav();
    window.addEventListener('hashchange', () => {
      const h = location.hash;
      if (/^#[^/]/.test(h)) return;               // in-page anchor, not a route
      render(h.replace(/^#\//, ''));
    });
    render(location.hash.replace(/^#\//, ''));
  }

  return {
    track, section, boot, render, tracks, sections, byId, stats,
    el, $, $$, typeset, markDone, isDone, escapeHtml, stripTags, onCleanup, search, debounce, shuffle,
    store: store, save: save, readingTime
  };
})();
