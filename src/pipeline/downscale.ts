/**
 * Downscale ImageData for ML analysis (letterbox to square).
 * Letterbox + transparent pixels are composited onto mid-gray (0.5)
 * so alpha holes do not read as black and bias brightness upward.
 */

/** Neutral gray fill: RGB 128 ≈ 0.5 in model space. */
const MODEL_BG = '#808080';

export function downscaleForModel(src: ImageData, size = 224): {
  data: Float32Array;
  width: number;
  height: number;
} {
  const scale = Math.min(size / src.width, size / src.height);
  const tw = Math.max(1, Math.round(src.width * scale));
  const th = Math.max(1, Math.round(src.height * scale));

  const canvas =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(size, size)
      : Object.assign(document.createElement('canvas'), { width: size, height: size });

  const ctx = canvas.getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null;
  if (!ctx) {
    throw new Error('2D context unavailable for downscale');
  }

  ctx.fillStyle = MODEL_BG;
  ctx.fillRect(0, 0, size, size);

  const ox = Math.floor((size - tw) / 2);
  const oy = Math.floor((size - th) / 2);

  const tmp =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(src.width, src.height)
      : Object.assign(document.createElement('canvas'), {
          width: src.width,
          height: src.height,
        });
  const tctx = tmp.getContext('2d') as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null;
  if (!tctx) {
    throw new Error('2D context unavailable for temp canvas');
  }
  tctx.putImageData(src, 0, 0);
  ctx.drawImage(tmp as CanvasImageSource, ox, oy, tw, th);

  const img = ctx.getImageData(0, 0, size, size);
  const out = new Float32Array(size * size * 3);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j += 3) {
    out[j] = img.data[i]! / 255;
    out[j + 1] = img.data[i + 1]! / 255;
    out[j + 2] = img.data[i + 2]! / 255;
  }
  return { data: out, width: size, height: size };
}
