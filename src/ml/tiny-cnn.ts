/**
 * Pure-JS TinyParamNet inference (matches tools/train/train.py TinyParamNet).
 * Loads weights from JSON produced by tools/train/export_web_weights.py
 */

import type { CorrectionParams } from '../types.js';
import { clipParams } from '../types.js';
import type { ParamPredictor } from './predictor.js';

export interface TinyCnnWeights {
  arch: 'TinyParamNet';
  conv1: { w: number[]; b: number[]; outC: number; inC: number; k: number };
  conv2: { w: number[]; b: number[]; outC: number; inC: number; k: number };
  conv3: { w: number[]; b: number[]; outC: number; inC: number; k: number };
  fc1: { w: number[]; b: number[]; out: number; in: number };
  fc2: { w: number[]; b: number[]; out: number; in: number };
}

function relu(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) if (x[i]! < 0) x[i] = 0;
}

/** NCHW conv, stride 2, padding 1, kernel 3. */
function conv2dStride2Pad1(
  input: Float32Array,
  inC: number,
  h: number,
  w: number,
  weight: Float32Array,
  bias: Float32Array,
  outC: number,
): { data: Float32Array; h: number; w: number } {
  const outH = Math.floor((h + 2 - 3) / 2) + 1;
  const outW = Math.floor((w + 2 - 3) / 2) + 1;
  const out = new Float32Array(outC * outH * outW);

  for (let oc = 0; oc < outC; oc++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = bias[oc]!;
        const iy0 = oy * 2 - 1;
        const ix0 = ox * 2 - 1;
        for (let ic = 0; ic < inC; ic++) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const iy = iy0 + ky;
              const ix = ix0 + kx;
              if (iy < 0 || ix < 0 || iy >= h || ix >= w) continue;
              const inIdx = (ic * h + iy) * w + ix;
              const wIdx = ((oc * inC + ic) * 3 + ky) * 3 + kx;
              sum += input[inIdx]! * weight[wIdx]!;
            }
          }
        }
        out[(oc * outH + oy) * outW + ox] = sum;
      }
    }
  }
  return { data: out, h: outH, w: outW };
}

function adaptiveAvgPool1(input: Float32Array, c: number, h: number, w: number): Float32Array {
  const out = new Float32Array(c);
  const area = h * w;
  for (let ch = 0; ch < c; ch++) {
    let s = 0;
    const base = ch * area;
    for (let i = 0; i < area; i++) s += input[base + i]!;
    out[ch] = s / area;
  }
  return out;
}

function linear(input: Float32Array, weight: Float32Array, bias: Float32Array, outN: number, inN: number): Float32Array {
  const out = new Float32Array(outN);
  for (let o = 0; o < outN; o++) {
    let s = bias[o]!;
    const row = o * inN;
    for (let i = 0; i < inN; i++) s += weight[row + i]! * input[i]!;
    out[o] = s;
  }
  return out;
}

/** HWC RGB [0,1] → NCHW */
function hwcToNchw(rgb: Float32Array, h: number, w: number): Float32Array {
  const out = new Float32Array(3 * h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3;
      const p = y * w + x;
      out[0 * h * w + p] = rgb[i]!;
      out[1 * h * w + p] = rgb[i + 1]!;
      out[2 * h * w + p] = rgb[i + 2]!;
    }
  }
  return out;
}

export class TinyCnnPredictor implements ParamPredictor {
  readonly name = 'tiny-cnn';
  private ready: Promise<void>;
  private weights: TinyCnnWeights | null = null;

  constructor(private readonly weightsUrl: string) {
    this.ready = this.load();
  }

  private async load(): Promise<void> {
    const res = await fetch(this.weightsUrl);
    if (!res.ok) {
      throw new Error(
        `Cannot load model weights (${res.status}): ${this.weightsUrl}. Export with tools/train/export_web_weights.py and place under models/ or demo/models/.`,
      );
    }
    this.weights = (await res.json()) as TinyCnnWeights;
    if (this.weights.arch !== 'TinyParamNet') {
      throw new Error(`Unexpected arch: ${String(this.weights.arch)}`);
    }
  }

  async predict(rgb: Float32Array, width: number, height: number): Promise<CorrectionParams> {
    await this.ready;
    const wts = this.weights!;
    let x = hwcToNchw(rgb, height, width);
    let h = height;
    let ww = width;

    const c1 = conv2dStride2Pad1(
      x,
      wts.conv1.inC,
      h,
      ww,
      new Float32Array(wts.conv1.w),
      new Float32Array(wts.conv1.b),
      wts.conv1.outC,
    );
    relu(c1.data);
    x = c1.data;
    h = c1.h;
    ww = c1.w;

    const c2 = conv2dStride2Pad1(
      x,
      wts.conv2.inC,
      h,
      ww,
      new Float32Array(wts.conv2.w),
      new Float32Array(wts.conv2.b),
      wts.conv2.outC,
    );
    relu(c2.data);
    x = c2.data;
    h = c2.h;
    ww = c2.w;

    const c3 = conv2dStride2Pad1(
      x,
      wts.conv3.inC,
      h,
      ww,
      new Float32Array(wts.conv3.w),
      new Float32Array(wts.conv3.b),
      wts.conv3.outC,
    );
    relu(c3.data);

    const pooled = adaptiveAvgPool1(c3.data, wts.conv3.outC, c3.h, c3.w);
    let fc = linear(
      pooled,
      new Float32Array(wts.fc1.w),
      new Float32Array(wts.fc1.b),
      wts.fc1.out,
      wts.fc1.in,
    );
    relu(fc);
    fc = linear(fc, new Float32Array(wts.fc2.w), new Float32Array(wts.fc2.b), wts.fc2.out, wts.fc2.in);

    return clipParams({
      brightness: fc[0]!,
      contrast: fc[1]!,
      saturation: fc[2]!,
    });
  }
}
