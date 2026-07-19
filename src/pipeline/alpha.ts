/** True if any pixel is not fully opaque (JPEG cannot preserve these). */
export function imageHasAlpha(image: ImageData): boolean {
  const d = image.data;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i]! < 255) return true;
  }
  return false;
}
