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

export type PredictorMode = 'heuristic' | 'model';

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
  error?: string;
  metrics?: TaskMetrics;
}

export interface SubmitOptions {
  /** Output MIME type. Default: image/jpeg */
  outputType?: 'image/jpeg' | 'image/png';
  /** JPEG quality 0..1. Default: 0.92 */
  quality?: number;
  /** heuristic (default) or trained TinyCNN JSON weights */
  predictorMode?: PredictorMode;
  /** URL to enhance_params.bin (or legacy .json) when predictorMode is "model" */
  modelUrl?: string;
}

export type StatusListener = (info: TaskInfo) => void;

export const MAX_MEGAPIXELS = 15;
export const MAX_PIXELS = MAX_MEGAPIXELS * 1_000_000;

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
