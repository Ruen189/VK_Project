import { mkdirSync, existsSync, cpSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'dist', 'worker.js');
const demoDir = join(root, 'demo', 'dist');
const modelsSrc = join(root, 'models');
const modelsDst = join(root, 'demo', 'models');

if (!existsSync(src)) {
  console.warn('dist/worker.js missing — run tsc first');
  process.exit(0);
}

mkdirSync(demoDir, { recursive: true });
cpSync(join(root, 'dist'), demoDir, { recursive: true });
console.log('Copied dist → demo/dist');

if (existsSync(modelsSrc)) {
  mkdirSync(modelsDst, { recursive: true });
  cpSync(modelsSrc, modelsDst, { recursive: true });
  console.log('Copied models → demo/models');
}
