/** Shared types for the image enhancer module. */

export type TaskId = string;

export type TaskStatus =
  | 'queued'
  | 'decoding'
  | 'analyzing'
  | 'applying'
  | 'encoding'
  | 'done'
  | 'error'
  | 'cancelled';

export interface CorrectionParams {
  /** Delta brightness in [-0.3, 0.3]. */
  brightness: number;
  /** Contrast multiplier in [0.7, 1.4]. */
  contrast: number;
  /** Saturation (colorfulness) multiplier in [0.7, 1.5]. */
  saturation: number;
}

export interface TaskMetrics {
  elapsedMs: number;
  width: number;
  height: number;
  megapixels: number;
  params?: CorrectionParams;
  predictor?: string;
}

export interface TaskInfo {
  id: TaskId;
  status: TaskStatus;
  /** 0..1 */
  progress: number;
  /** Non-fatal condition; processing continues. */
  warning?: string;
  error?: string;
  metrics?: TaskMetrics;
}

export interface SubmitOptions {
  /**
   * Output MIME type.
   * Default: image/png if the decoded image has alpha, otherwise image/jpeg.
   */
  outputType?: 'image/jpeg' | 'image/png';
  /** JPEG quality 0..1. Default: 0.92 */
  quality?: number;
  /** URL to enhance_params.bin (or legacy .json) */
  modelUrl?: string;
}

export type StatusListener = (info: TaskInfo) => void;

export const IMAGE_SIZE_WARNING_MEGAPIXELS = 15;
export const IMAGE_SIZE_WARNING_PIXELS = IMAGE_SIZE_WARNING_MEGAPIXELS * 1_000_000;
/** @deprecated This is now a warning threshold, not a processing limit. */
export const MAX_MEGAPIXELS = IMAGE_SIZE_WARNING_MEGAPIXELS;
/** @deprecated This is now a warning threshold, not a processing limit. */
export const MAX_PIXELS = IMAGE_SIZE_WARNING_PIXELS;

export function imageSizeWarning(width: number, height: number): string | undefined {
  if (width * height <= IMAGE_SIZE_WARNING_PIXELS) return undefined;
  return `Image exceeds ${IMAGE_SIZE_WARNING_MEGAPIXELS} MP (${width}×${height})`;
}

export const PARAM_CLIP = {
  brightness: { min: -0.3, max: 0.3 },
  contrast: { min: 0.7, max: 1.4 },
  saturation: { min: 0.7, max: 1.5 },
} as const;

export function clipParams(p: CorrectionParams): CorrectionParams {
  return {
    brightness: clamp(p.brightness, PARAM_CLIP.brightness.min, PARAM_CLIP.brightness.max),
    contrast: clamp(p.contrast, PARAM_CLIP.contrast.min, PARAM_CLIP.contrast.max),
    saturation: clamp(p.saturation, PARAM_CLIP.saturation.min, PARAM_CLIP.saturation.max),
  };
}

export function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}
