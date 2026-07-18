/**
 * HEIC/HEIF → PNG via lazy-loaded heic2any (bundled to dist/vendor/heic2any.js).
 * Runs on the main thread (heic2any expects DOM); Worker receives PNG/JPEG only.
 */

import { sniffMime } from './pipeline/decode.js';

export type Heic2AnyFn = (opts: {
  blob: Blob;
  toType?: string;
  quality?: number;
}) => Promise<Blob | Blob[]>;

let decoderUrl: string | null = null;
let decoderPromise: Promise<Heic2AnyFn> | null = null;

/** Override vendor module URL (default: ./vendor/heic2any.js next to this module). */
export function setHeicDecoderUrl(url: string): void {
  decoderUrl = url;
  decoderPromise = null;
}

export function isHeicBuffer(buffer: ArrayBuffer, mimeHint?: string): boolean {
  const mime = sniffMime(new Uint8Array(buffer), mimeHint);
  return mime === 'image/heic' || mime === 'image/heif';
}

async function loadHeic2Any(): Promise<Heic2AnyFn> {
  if (!decoderPromise) {
    const url = decoderUrl ?? new URL('./vendor/heic2any.js', import.meta.url).href;
    decoderPromise = import(/* @vite-ignore */ url).then((mod: unknown) => {
      const m = mod as { default?: Heic2AnyFn } | Heic2AnyFn;
      const fn = typeof m === 'function' ? m : m.default;
      if (typeof fn !== 'function') {
        throw new Error(`HEIC decoder export missing at ${url}`);
      }
      return fn;
    });
  }
  return decoderPromise;
}

/** Convert HEIC/HEIF bytes to PNG ArrayBuffer. */
export async function convertHeicToPng(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const heic2any = await loadHeic2Any();
  const input = new Blob([buffer], { type: 'image/heic' });
  const result = await heic2any({ blob: input, toType: 'image/png' });
  const pngBlob = Array.isArray(result) ? result[0] : result;
  if (!pngBlob) {
    throw new Error('HEIC conversion returned empty result');
  }
  return pngBlob.arrayBuffer();
}

/** If buffer is HEIC, convert to PNG; otherwise return as-is. */
export async function ensureDecodableImage(
  buffer: ArrayBuffer,
  mimeHint?: string,
): Promise<{ buffer: ArrayBuffer; mime: string }> {
  if (!isHeicBuffer(buffer, mimeHint)) {
    const mime = sniffMime(new Uint8Array(buffer), mimeHint);
    return { buffer, mime };
  }
  const png = await convertHeicToPng(buffer);
  return { buffer: png, mime: 'image/png' };
}
