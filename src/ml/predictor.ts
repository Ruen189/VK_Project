/** Parameter predictor: TinyCNN weights. */

import type { CorrectionParams } from '../types.js';
import { TinyCnnPredictor } from './tiny-cnn.js';

export interface ParamPredictor {
  readonly name: string;
  predict(rgbFloat: Float32Array, width: number, height: number): Promise<CorrectionParams>;
}

const modelCache = new Map<string, TinyCnnPredictor>();

/**
 * Resolve TinyCNN weights.
 * Second argument keeps compatibility with an older worker that called
 * `resolvePredictor(mode, modelUrl)` — otherwise a cached worker treats
 * `"heuristic"` as a weights URL and fetch() 404s.
 */
export function resolvePredictor(modelUrlOrMode: string, legacyModelUrl?: string): ParamPredictor {
  const modelUrl =
    modelUrlOrMode === 'heuristic' || modelUrlOrMode === 'model'
      ? legacyModelUrl
      : modelUrlOrMode;
  if (!modelUrl) {
    throw new Error('modelUrl is required');
  }
  let p = modelCache.get(modelUrl);
  if (!p) {
    p = new TinyCnnPredictor(modelUrl);
    modelCache.set(modelUrl, p);
  }
  return p;
}

export { TinyCnnPredictor } from './tiny-cnn.js';
