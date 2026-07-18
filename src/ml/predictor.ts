/** Parameter predictors: heuristic + TinyCNN (JSON weights). */

import type { CorrectionParams } from '../types.js';
import { clipParams } from '../types.js';
import { TinyCnnPredictor } from './tiny-cnn.js';

export type PredictorMode = 'heuristic' | 'model';

export interface ParamPredictor {
  readonly name: string;
  predict(rgbFloat: Float32Array, width: number, height: number): Promise<CorrectionParams>;
}

/**
 * Histogram-based heuristic (baseline until ML weights are ready).
 * Pulls mean luminance toward ~0.45, expands low contrast, boosts weak saturation.
 */
export class HeuristicPredictor implements ParamPredictor {
  readonly name = 'heuristic-v1';

  async predict(rgb: Float32Array, _w: number, _h: number): Promise<CorrectionParams> {
    let sumY = 0;
    let sumSat = 0;
    let minY = 1;
    let maxY = 0;
    const n = rgb.length / 3;

    for (let i = 0; i < rgb.length; i += 3) {
      const r = rgb[i]!;
      const g = rgb[i + 1]!;
      const b = rgb[i + 2]!;
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      const sat = mx > 1e-6 ? (mx - mn) / mx : 0;
      sumY += y;
      sumSat += sat;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    const meanY = sumY / n;
    const meanSat = sumSat / n;
    const range = Math.max(1e-3, maxY - minY);

    const brightness = (0.45 - meanY) * 0.6;
    const contrast = Math.min(1.35, Math.max(0.85, 0.55 / range));
    const saturation = meanSat < 0.25 ? 1.25 : meanSat > 0.55 ? 0.9 : 1.05;

    return clipParams({ brightness, contrast, saturation });
  }
}

/**
 * @deprecated Use TinyCnnPredictor via resolvePredictor('model', url).
 * Kept for API compatibility; throws until ORT is wired.
 */
export class OnnxParamPredictor implements ParamPredictor {
  readonly name = 'onnx-pending';

  constructor(private readonly modelUrl: string) {}

  async predict(_rgb: Float32Array, _w: number, _h: number): Promise<CorrectionParams> {
    void this.modelUrl;
    throw new Error(
      'Use predictorMode: "model" with enhance_params.json (export_web_weights.py). ONNX Runtime path is optional later.',
    );
  }
}

const heuristic = new HeuristicPredictor();
const modelCache = new Map<string, TinyCnnPredictor>();

export function resolvePredictor(mode: PredictorMode = 'heuristic', modelUrl?: string): ParamPredictor {
  if (mode === 'heuristic') return heuristic;
  if (!modelUrl) {
    throw new Error('modelUrl is required when predictorMode is "model"');
  }
  let p = modelCache.get(modelUrl);
  if (!p) {
    p = new TinyCnnPredictor(modelUrl);
    modelCache.set(modelUrl, p);
  }
  return p;
}

let defaultPredictor: ParamPredictor = heuristic;

export function getPredictor(): ParamPredictor {
  return defaultPredictor;
}

export function setPredictor(p: ParamPredictor): void {
  defaultPredictor = p;
}

export { TinyCnnPredictor } from './tiny-cnn.js';
