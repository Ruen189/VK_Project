/**
 * Static demo server: serves ./demo, opens the browser.
 * Usage: node scripts/demo-server.mjs
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exec } from 'node:child_process';
import { dirname } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const demoRoot = join(root, 'demo');
const PORT = Number(process.env.PORT || 5173);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.map': 'application/json',
  '.wasm': 'application/wasm',
};

if (!existsSync(join(demoRoot, 'dist', 'index.js'))) {
  console.error('demo/dist missing. Run: npm run build');
  process.exit(1);
}

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const rel = decoded === '/' ? '/index.html' : decoded;
  const full = normalize(join(demoRoot, rel));
  if (!full.startsWith(demoRoot + sep) && full !== demoRoot) {
    return null;
  }
  return full;
}

const server = createServer(async (req, res) => {
  try {
    const filePath = safePath(req.url || '/');
    if (!filePath) {
      res.writeHead(403).end('Forbidden');
      return;
    }
    const st = await stat(filePath);
    if (!st.isFile()) {
      res.writeHead(404).end('Not found');
      return;
    }
    const body = await readFile(filePath);
    const type = MIME[extname(filePath).toLowerCase()] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': type, 'Cache-Control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404).end('Not found');
  }
});

server.listen(PORT, () => {
  const url = `http://127.0.0.1:${PORT}/`;
  console.log(`Demo server: ${url}`);
  console.log('Press Ctrl+C to stop.');
  openBrowser(url);
});

function openBrowser(url) {
  const platform = process.platform;
  const cmd =
    platform === 'win32'
      ? `start "" "${url}"`
      : platform === 'darwin'
        ? `open "${url}"`
        : `xdg-open "${url}"`;
  exec(cmd, (err) => {
    if (err) console.log(`Open manually: ${url}`);
  });
}
