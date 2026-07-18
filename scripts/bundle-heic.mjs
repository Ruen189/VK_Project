import * as esbuild from 'esbuild';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outfile = join(root, 'dist', 'vendor', 'heic2any.js');

mkdirSync(join(root, 'dist', 'vendor'), { recursive: true });

await esbuild.build({
  entryPoints: [join(root, 'scripts', 'heic-entry.mjs')],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  outfile,
  logLevel: 'info',
});

console.log(`Bundled HEIC decoder → ${outfile}`);
