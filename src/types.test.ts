import { describe, expect, it } from 'vitest';
import { clipParams, clamp, PARAM_CLIP } from './types.js';
import { HeuristicPredictor } from './ml/predictor.js';
import { stageProgress } from './pipeline/progress.js';
import { sniffMime } from './pipeline/decode.js';

describe('clamp / clipParams', () => {
  it('clamps values', () => {
    expect(clamp(2, 0, 1)).toBe(1);
    expect(clamp(-1, 0, 1)).toBe(0);
  });

  it('clips correction params', () => {
    const p = clipParams({ brightness: 9, contrast: 0.1, saturation: 99 });
    expect(p.brightness).toBe(PARAM_CLIP.brightness.max);
    expect(p.contrast).toBe(PARAM_CLIP.contrast.min);
    expect(p.saturation).toBe(PARAM_CLIP.saturation.max);
  });
});

describe('stageProgress', () => {
  it('maps stage fractions into global progress', () => {
    expect(stageProgress('decoding', 0)).toBe(0);
    expect(stageProgress('decoding', 1)).toBeCloseTo(0.15);
    expect(stageProgress('done', 1)).toBe(1);
  });
});

describe('sniffMime', () => {
  it('detects JPEG', () => {
    expect(sniffMime(new Uint8Array([0xff, 0xd8, 0xff, 0xe0]))).toBe('image/jpeg');
  });

  it('detects PNG', () => {
    expect(sniffMime(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x0b, 0x0a]))).toBe(
      'image/png',
    );
  });
});

describe('HeuristicPredictor', () => {
  it('returns clipped params for a gray image', async () => {
    const n = 224 * 224;
    const rgb = new Float32Array(n * 3);
    for (let i = 0; i < rgb.length; i++) rgb[i] = 0.3;
    const p = await new HeuristicPredictor().predict(rgb, 224, 224);
    expect(p.brightness).toBeGreaterThanOrEqual(PARAM_CLIP.brightness.min);
    expect(p.brightness).toBeLessThanOrEqual(PARAM_CLIP.brightness.max);
    expect(p.contrast).toBeGreaterThanOrEqual(PARAM_CLIP.contrast.min);
    expect(p.saturation).toBeGreaterThanOrEqual(PARAM_CLIP.saturation.min);
  });
});
