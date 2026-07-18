/**
 * Decode image bytes into ImageData.
 * JPG / PNG / BMP: createImageBitmap.
 * HEIC: preferably converted on main thread (see heic.ts); Safari may decode natively.
 */

import { MAX_PIXELS } from '../types.js';

export class DecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DecodeError';
  }
}

const HEIC_BRANDS = new Set(['heic', 'heif', 'mif1', 'msf1', 'hevx', 'heim', 'heis', 'hevm', 'hevs']);

export function sniffMime(bytes: Uint8Array, fallback?: string): string {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    return 'image/png';
  }
  if (bytes.length >= 2 && bytes[0] === 0x42 && bytes[1] === 0x4d) {
    return 'image/bmp';
  }
  if (bytes.length >= 12) {
    const box = String.fromCharCode(bytes[4]!, bytes[5]!, bytes[6]!, bytes[7]!);
    if (box === 'ftyp') {
      const brand = String.fromCharCode(bytes[8]!, bytes[9]!, bytes[10]!, bytes[11]!);
      if (HEIC_BRANDS.has(brand)) return 'image/heic';
      const head = String.fromCharCode(...bytes.slice(8, Math.min(bytes.length, 32)));
      if (/heic|heif|mif1|msf1/i.test(head)) return 'image/heic';
    }
  }
  if (fallback && /heic|heif/i.test(fallback)) return 'image/heic';
  return fallback ?? 'application/octet-stream';
}

export async function decodeImage(
  buffer: ArrayBuffer,
  mimeHint?: string,
): Promise<ImageData> {
  const bytes = new Uint8Array(buffer);
  const mime = sniffMime(bytes, mimeHint);

  if (mime === 'image/heic') {
    try {
      return await bitmapToImageData(buffer, 'image/heic');
    } catch {
      throw new DecodeError(
        'HEIC is not decodable in this Worker. Convert on the main thread (ImageEnhancer does this automatically) or use Safari.',
      );
    }
  }

  return bitmapToImageData(buffer, mime);
}

async function bitmapToImageData(buffer: ArrayBuffer, mime: string): Promise<ImageData> {
  if (typeof createImageBitmap !== 'function') {
    throw new DecodeError('createImageBitmap is not available in this environment');
  }

  const blob = new Blob([buffer], { type: mime === 'application/octet-stream' ? undefined : mime });
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch (e) {
    throw new DecodeError(`Failed to decode image (${mime}): ${(e as Error).message}`);
  }

  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS) {
    bitmap.close();
    throw new DecodeError(
      `Image exceeds ${MAX_PIXELS / 1_000_000} MP limit (${width}×${height})`,
    );
  }

  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement('canvas'), { width, height });

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    bitmap.close();
    throw new DecodeError('2D context unavailable');
  }

  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  return (ctx as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D).getImageData(
    0,
    0,
    width,
    height,
  );
}
