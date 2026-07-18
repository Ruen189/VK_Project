/** Encode ImageData to Blob via OffscreenCanvas / canvas. */

export async function encodeImage(
  image: ImageData,
  type: 'image/jpeg' | 'image/png' = 'image/jpeg',
  quality = 0.92,
): Promise<Blob> {
  const { width, height } = image;

  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
    ctx.putImageData(image, 0, 0);
    return canvas.convertToBlob({ type, quality });
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');
  ctx.putImageData(image, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('toBlob failed'))),
      type,
      quality,
    );
  });
}
