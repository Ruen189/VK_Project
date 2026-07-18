import { readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const LIMIT = 10 * 1024 * 1024;
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');

function walk(dir) {
  let total = 0;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) total += walk(p);
    else total += st.size;
  }
  return total;
}

try {
  const bytes = walk(dist);
  const mb = (bytes / (1024 * 1024)).toFixed(3);
  console.log(`dist size: ${bytes} bytes (${mb} MB)`);
  if (bytes > LIMIT) {
    console.error('FAIL: exceeds 10 MB limit');
    process.exit(1);
  }
  console.log('OK: within 10 MB');
} catch (e) {
  console.error('Build dist/ first:', e.message);
  process.exit(1);
}
