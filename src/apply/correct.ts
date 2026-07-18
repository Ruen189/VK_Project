/**
 * Apply brightness / contrast / saturation to full-resolution ImageData.
 * Processes in tiles and yields so cancel + progress work.
 */

import type { CorrectionParams } from '../types.js';
import { clamp } from '../types.js';

const TILE = 256;

export interface ApplyOptions {
  signal?: AbortSignal;
  onTile?: (done: number, total: number) => void;
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((r) => setTimeout(r, 0));
}

/** Rec.709 luma */
function luma(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export async function applyCorrection(
  src: ImageData,
  params: CorrectionParams,
  options: ApplyOptions = {},
): Promise<ImageData> {
  const { brightness, contrast, saturation } = params;
  const { width, height, data } = src;
  const out = new ImageData(width, height);
  const dst = out.data;

  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const total = tilesX * tilesY;
  let done = 0;

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      if (options.signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }

      const x0 = tx * TILE;
      const y0 = ty * TILE;
      const x1 = Math.min(width, x0 + TILE);
      const y1 = Math.min(height, y0 + TILE);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * width + x) * 4;
          let r = data[i]! / 255;
          let g = data[i + 1]! / 255;
          let b = data[i + 2]! / 255;
          const a = data[i + 3]!;

          r = (r - 0.5) * contrast + 0.5 + brightness;
          g = (g - 0.5) * contrast + 0.5 + brightness;
          b = (b - 0.5) * contrast + 0.5 + brightness;

          const yL = luma(r, g, b);
          r = yL + (r - yL) * saturation;
          g = yL + (g - yL) * saturation;
          b = yL + (b - yL) * saturation;

          dst[i] = Math.round(clamp(r, 0, 1) * 255);
          dst[i + 1] = Math.round(clamp(g, 0, 1) * 255);
          dst[i + 2] = Math.round(clamp(b, 0, 1) * 255);
          dst[i + 3] = a;
        }
      }

      done++;
      options.onTile?.(done, total);
      await yieldToEventLoop();
    }
  }

  return out;
}
