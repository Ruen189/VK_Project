/**
 * Pure-JS TinyParamNet inference (matches tools/train/train.py TinyParamNet).
 * Loads weights from TPCB .bin (float16/float32) or legacy JSON.
 */

import type { CorrectionParams } from '../types.js';
import { clipParams } from '../types.js';
import type { ParamPredictor } from './predictor.js';

/** Runtime weights always float32 for math. */
export interface TinyCnnRuntimeWeights {
  conv1: { w: Float32Array; b: Float32Array; outC: number; inC: number };
  conv2: { w: Float32Array; b: Float32Array; outC: number; inC: number };
  conv3: { w: Float32Array; b: Float32Array; outC: number; inC: number };
  fc1: { w: Float32Array; b: Float32Array; out: number; in: number };
  fc2: { w: Float32Array; b: Float32Array; out: number; in: number };
}

function relu(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) if (x[i]! < 0) x[i] = 0;
}

/** IEEE-754 binary16 → float32 */
export function float16ToFloat32(u16: number): number {
  const sign = (u16 >> 15) & 1;
  let exp = (u16 >> 10) & 0x1f;
  let frac = u16 & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // subnormal
    exp = -14;
    while (!(frac & 0x400)) {
      frac <<= 1;
      exp -= 1;
    }
    frac &= 0x3ff;
    const f = (frac / 0x400) * Math.pow(2, exp);
    return sign ? -f : f;
  }
  if (exp === 0x1f) {
    return frac ? NaN : sign ? -Infinity : Infinity;
  }
  const f = (1 + frac / 0x400) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

function readF32LE(view: DataView, offset: number, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getFloat32(offset + i * 4, true);
  return out;
}

function readF16LE(view: DataView, offset: number, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = float16ToFloat32(view.getUint16(offset + i * 2, true));
  }
  return out;
}

function parseTpcb(buf: ArrayBuffer): TinyCnnRuntimeWeights {
  const view = new DataView(buf);
  if (buf.byteLength < 16) throw new Error('TPCB file too small');
  const m0 = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (m0 !== 'TPCB') throw new Error(`Bad magic: ${m0}`);
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`Unsupported TPCB version: ${version}`);
  const dtype = view.getUint8(6); // 0 f32, 1 f16
  const c1 = view.getUint16(8, true);
  const c2 = view.getUint16(10, true);
  const c3 = view.getUint16(12, true);
  const fcH = view.getUint16(14, true);

  const elem = dtype === 1 ? 2 : 4;
  const read = dtype === 1 ? readF16LE : readF32LE;
  let off = 16;

  const take = (n: number): Float32Array => {
    const arr = read(view, off, n);
    off += n * elem;
    return arr;
  };

  // shapes for k=3 convs
  const conv1w = take(c1 * 3 * 3 * 3);
  const conv1b = take(c1);
  const conv2w = take(c2 * c1 * 3 * 3);
  const conv2b = take(c2);
  const conv3w = take(c3 * c2 * 3 * 3);
  const conv3b = take(c3);
  const fc1w = take(fcH * c3);
  const fc1b = take(fcH);
  const fc2w = take(3 * fcH);
  const fc2b = take(3);

  if (off !== buf.byteLength) {
    console.warn(`TPCB: trailing bytes ${buf.byteLength - off}`);
  }

  return {
    conv1: { w: conv1w, b: conv1b, outC: c1, inC: 3 },
    conv2: { w: conv2w, b: conv2b, outC: c2, inC: c1 },
    conv3: { w: conv3w, b: conv3b, outC: c3, inC: c2 },
    fc1: { w: fc1w, b: fc1b, out: fcH, in: c3 },
    fc2: { w: fc2w, b: fc2b, out: 3, in: fcH },
  };
}

function parseJsonWeights(json: unknown): TinyCnnRuntimeWeights {
  const w = json as {
    arch?: string;
    conv1: { w: number[]; b: number[]; outC: number; inC: number };
    conv2: { w: number[]; b: number[]; outC: number; inC: number };
    conv3: { w: number[]; b: number[]; outC: number; inC: number };
    fc1: { w: number[]; b: number[]; out: number; in: number };
    fc2: { w: number[]; b: number[]; out: number; in: number };
  };
  if (w.arch && w.arch !== 'TinyParamNet') {
    throw new Error(`Unexpected arch: ${w.arch}`);
  }
  return {
    conv1: { w: new Float32Array(w.conv1.w), b: new Float32Array(w.conv1.b), outC: w.conv1.outC, inC: w.conv1.inC },
    conv2: { w: new Float32Array(w.conv2.w), b: new Float32Array(w.conv2.b), outC: w.conv2.outC, inC: w.conv2.inC },
    conv3: { w: new Float32Array(w.conv3.w), b: new Float32Array(w.conv3.b), outC: w.conv3.outC, inC: w.conv3.inC },
    fc1: { w: new Float32Array(w.fc1.w), b: new Float32Array(w.fc1.b), out: w.fc1.out, in: w.fc1.in },
    fc2: { w: new Float32Array(w.fc2.w), b: new Float32Array(w.fc2.b), out: w.fc2.out, in: w.fc2.in },
  };
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
  private weights: TinyCnnRuntimeWeights | null = null;

  constructor(private readonly weightsUrl: string) {
    this.ready = this.load();
  }

  private async load(): Promise<void> {
    const res = await fetch(this.weightsUrl);
    if (!res.ok) {
      throw new Error(
        `Cannot load model weights (${res.status}): ${this.weightsUrl}. ` +
          `Export: python export_web_weights.py --out ../../models/enhance_params.bin`,
      );
    }
    const url = this.weightsUrl.split('?')[0]!.toLowerCase();
    if (url.endsWith('.json')) {
      this.weights = parseJsonWeights(await res.json());
    } else {
      this.weights = parseTpcb(await res.arrayBuffer());
    }
  }

  async predict(rgb: Float32Array, width: number, height: number): Promise<CorrectionParams> {
    await this.ready;
    const wts = this.weights!;
    let x = hwcToNchw(rgb, height, width);
    let h = height;
    let ww = width;

    const c1 = conv2dStride2Pad1(x, wts.conv1.inC, h, ww, wts.conv1.w, wts.conv1.b, wts.conv1.outC);
    relu(c1.data);
    x = c1.data;
    h = c1.h;
    ww = c1.w;

    const c2 = conv2dStride2Pad1(x, wts.conv2.inC, h, ww, wts.conv2.w, wts.conv2.b, wts.conv2.outC);
    relu(c2.data);
    x = c2.data;
    h = c2.h;
    ww = c2.w;

    const c3 = conv2dStride2Pad1(x, wts.conv3.inC, h, ww, wts.conv3.w, wts.conv3.b, wts.conv3.outC);
    relu(c3.data);

    const pooled = adaptiveAvgPool1(c3.data, wts.conv3.outC, c3.h, c3.w);
    let fc = linear(pooled, wts.fc1.w, wts.fc1.b, wts.fc1.out, wts.fc1.in);
    relu(fc);
    fc = linear(fc, wts.fc2.w, wts.fc2.b, wts.fc2.out, wts.fc2.in);

    return clipParams({
      brightness: fc[0]!,
      contrast: fc[1]!,
      saturation: fc[2]!,
    });
  }
}
