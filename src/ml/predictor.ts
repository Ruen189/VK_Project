/** Parameter predictor: TinyCNN weights. */

import type { CorrectionParams } from '../types.js';
import { TinyCnnPredictor } from './tiny-cnn.js';

export interface ParamPredictor {
  readonly name: string;
  predict(rgbFloat: Float32Array, width: number, height: number): Promise<CorrectionParams>;
}

const modelCache = new Map<string, TinyCnnPredictor>();

export function resolvePredictor(modelUrl: string): ParamPredictor {
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
