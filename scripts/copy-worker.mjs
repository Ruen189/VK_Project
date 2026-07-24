import { mkdirSync, existsSync, readdirSync, copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Avoid fs.cpSync recursive — crashes Node 25 on Windows (exit -1073740791). */
function copyDir(src, dest) {
  mkdirSync(dest, { recursive: true });
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    const from = join(src, entry.name);
    const to = join(dest, entry.name);
    if (entry.isDirectory()) copyDir(from, to);
    else copyFileSync(from, to);
  }
}

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'dist', 'worker.js');
const demoDir = join(root, 'demo', 'dist');
const modelsSrc = join(root, 'models');
const modelsDst = join(root, 'demo', 'models');

if (!existsSync(src)) {
  console.warn('dist/worker.js missing — run tsc first');
  process.exit(0);
}

copyDir(join(root, 'dist'), demoDir);
console.log('Copied dist → demo/dist');

if (existsSync(modelsSrc)) {
  copyDir(modelsSrc, modelsDst);
  console.log('Copied models → demo/models');
}
