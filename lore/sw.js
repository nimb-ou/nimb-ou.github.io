/* LORE Field service worker (generated at build time from src/sw-template.js).
 * - Precaches the app shell (HTML, JS, CSS, fonts, icons, manifest, bundled City Packs) so LORE opens offline.
 * - Navigations: network first (3.5 s timeout) so updates arrive instantly, cached shell when offline.
 * - Same-origin assets: cache first. Cross-origin API calls (Wikipedia, Overpass, Gemini) are never
 *   touched here: stories and places are cached in IndexedDB by the app.
 * All paths are relative to the worker's location, so it works under any base path (e.g. /lore/). */
const VERSION = "361e167307ca";
const PRECACHE = ["./","assets/fraunces-B2S8_hai.woff2","assets/index-BtgqL6BP.js","assets/index-CztMxnyy.css","assets/inter-9GvP5jGv.woff2","assets/jetbrains-mono-B8NcMHdO.woff2","icons/apple-touch-icon.png","icons/favicon.svg","icons/icon-1024.png","icons/icon-192.png","icons/icon-512.png","icons/icon-maskable-512.png","icons/lore-app-icon.svg","icons/lore-symbol-bone.svg","icons/lore-symbol-ink.svg","icons/lore-symbol-terracotta.svg","index.html","manifest.webmanifest"];
const CACHE = `lore-field-${VERSION}`;
const SHELL = new URL("./", self.location).href;

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await cache.addAll(PRECACHE.map((u) => new Request(new URL(u, self.location).href, { cache: "reload" })));
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) if (key.startsWith("lore-field-") && key !== CACHE) await caches.delete(key);
      await self.clients.claim();
    })(),
  );
});

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const response = await Promise.race([
      fetch(request),
      new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 3500)),
    ]);
    if (response && response.ok) cache.put(SHELL, response.clone());
    return response;
  } catch {
    return (await cache.match(SHELL)) || (await cache.match(request, { ignoreSearch: true })) || Response.error();
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request, { ignoreSearch: true });
  if (hit) return hit;
  const response = await fetch(request);
  if (response.ok && response.type === "basic") cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || !request.url.startsWith(new URL("./", self.location).href)) return;
  if (request.mode === "navigate") event.respondWith(networkFirst(request));
  else event.respondWith(cacheFirst(request));
});
