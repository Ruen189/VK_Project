/**
 * Synthetic degrade (PIL ImageEnhance-like factors) for demos / tests.
 * brightness/contrast/saturation: 1 = no change; <1 darker/flatter/desaturated.
 */

import { clamp } from '../types.js';

export interface DegradeFactors {
  brightness: number;
  contrast: number;
  saturation: number;
}

export function randomDegradeFactors(range = { min: 0.7, max: 1.3 }): DegradeFactors {
  const rnd = () => range.min + Math.random() * (range.max - range.min);
  return { brightness: rnd(), contrast: rnd(), saturation: rnd() };
}

function luma(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** In-place-ish: returns new ImageData. */
export function degradeImageData(src: ImageData, factors: DegradeFactors): ImageData {
  const { brightness: bF, contrast: cF, saturation: sF } = factors;
  const out = new ImageData(src.width, src.height);
  const s = src.data;
  const d = out.data;

  // Mean luminance for contrast pivot (approx ImageEnhance.Contrast)
  let sumY = 0;
  const n = src.width * src.height;
  for (let i = 0; i < s.length; i += 4) {
    sumY += luma(s[i]! / 255, s[i + 1]! / 255, s[i + 2]! / 255);
  }
  const meanY = sumY / n;

  for (let i = 0; i < s.length; i += 4) {
    let r = s[i]! / 255;
    let g = s[i + 1]! / 255;
    let b = s[i + 2]! / 255;
    const a = s[i + 3]!;

    r *= bF;
    g *= bF;
    b *= bF;

    r = meanY + (r - meanY) * cF;
    g = meanY + (g - meanY) * cF;
    b = meanY + (b - meanY) * cF;

    const y = luma(r, g, b);
    r = y + (r - y) * sF;
    g = y + (g - y) * sF;
    b = y + (b - y) * sF;

    d[i] = Math.round(clamp(r, 0, 1) * 255);
    d[i + 1] = Math.round(clamp(g, 0, 1) * 255);
    d[i + 2] = Math.round(clamp(b, 0, 1) * 255);
    d[i + 3] = a;
  }
  return out;
}

/** Browser helper: File/Blob → degraded JPEG Blob. */
export async function degradeBlob(
  input: Blob,
  factors: DegradeFactors = randomDegradeFactors(),
): Promise<{ blob: Blob; factors: DegradeFactors }> {
  const bmp = await createImageBitmap(input);
  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(bmp.width, bmp.height)
      : Object.assign(document.createElement('canvas'), { width: bmp.width, height: bmp.height });
  const ctx = canvas.getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null;
  if (!ctx) {
    bmp.close();
    throw new Error('2D context unavailable');
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const src = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const degraded = degradeImageData(src, factors);
  ctx.putImageData(degraded, 0, 0);

  let blob: Blob;
  if ('convertToBlob' in canvas) {
    blob = await (canvas as OffscreenCanvas).convertToBlob({ type: 'image/jpeg', quality: 0.92 });
  } else {
    blob = await new Promise((resolve, reject) => {
      (canvas as HTMLCanvasElement).toBlob(
        (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
        'image/jpeg',
        0.92,
      );
    });
  }
  return { blob, factors };
}
