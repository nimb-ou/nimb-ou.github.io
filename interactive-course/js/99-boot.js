/* boot after all content files have registered */
(function () {
  'use strict';
  function go() { try { ML.boot(); } catch (e) { console.error(e); document.getElementById('main').innerHTML = '<h1>Failed to start</h1><pre>' + (e && e.stack || e) + '</pre>'; } }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
