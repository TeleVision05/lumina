// Minimal static server that mimics GitHub Pages for the built site: web/dist served under
// /lumina/, index.html for directories, 404 for everything else (no SPA fallback, no custom
// headers, so no cross-origin isolation, exactly like Pages).
// Usage: node scripts/serve-pages.mjs [port]
import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../dist/', import.meta.url));
const BASE = '/lumina/';
const PORT = Number(process.argv[2] ?? 5181);
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.map': 'application/json',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function resolve(urlPath) {
  if (!urlPath.startsWith(BASE)) return null;
  const rel = normalize(decodeURIComponent(urlPath.slice(BASE.length)));
  if (rel.startsWith('..')) return null;
  let file = join(ROOT, rel);
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
  return existsSync(file) && statSync(file).isFile() ? file : null;
}

createServer((req, res) => {
  const path = (req.url ?? '/').split('?')[0] ?? '/';
  if (path === '/lumina') {
    res.writeHead(301, { Location: BASE }).end();
    return;
  }
  const file = resolve(path);
  if (file === null) {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
    return;
  }
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' });
  createReadStream(file).pipe(res);
}).listen(PORT, () => {
  console.log(`serving ${ROOT} at http://localhost:${PORT}${BASE}`);
});
