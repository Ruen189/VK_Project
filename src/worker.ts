/// <reference lib="webworker" />

/**
 * Enhancement pipeline worker.
 * Runs decode → analyze → apply → encode off the UI thread.
 */

import { applyCorrection } from './apply/correct.js';
import { imageHasAlpha } from './pipeline/alpha.js';
import { decodeImage, DecodeError } from './pipeline/decode.js';
import { downscaleForModel } from './pipeline/downscale.js';
import { encodeImage } from './pipeline/encode.js';
import { stageProgress } from './pipeline/progress.js';
import { resolvePredictor } from './ml/predictor.js';
import { makeTaskInfo, type MainToWorker, type WorkerToMain } from './messages.js';
import { imageSizeWarning } from './types.js';
import type { SubmitOptions, TaskId } from './types.js';

declare const self: DedicatedWorkerGlobalScope;

interface ActiveJob {
  controller: AbortController;
}

const jobs = new Map<TaskId, ActiveJob>();

function post(msg: WorkerToMain, transfer?: Transferable[]): void {
  self.postMessage(msg, transfer ?? []);
}

function emitStatus(
  id: TaskId,
  status: Parameters<typeof makeTaskInfo>[1],
  progress: number,
  extra?: Parameters<typeof makeTaskInfo>[3],
): void {
  post({ type: 'status', info: makeTaskInfo(id, status, progress, extra) });
}

async function runJob(
  id: TaskId,
  buffer: ArrayBuffer,
  mime: string | undefined,
  options: SubmitOptions | undefined,
): Promise<void> {
  const controller = new AbortController();
  jobs.set(id, { controller });
  const started = performance.now();
  const quality = options?.quality ?? 0.92;
  let warning: string | undefined;
  const emitJobStatus = (
    status: Parameters<typeof makeTaskInfo>[1],
    progress: number,
    extra?: Parameters<typeof makeTaskInfo>[3],
  ): void => {
    emitStatus(id, status, progress, { ...extra, warning });
  };

  try {
    emitJobStatus('decoding', stageProgress('decoding', 0));
    const image = await decodeImage(buffer, mime);
    warning = imageSizeWarning(image.width, image.height);
    emitJobStatus('decoding', stageProgress('decoding', 1), {
      metrics: {
        elapsedMs: performance.now() - started,
        width: image.width,
        height: image.height,
        megapixels: (image.width * image.height) / 1_000_000,
      },
    });

    if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');

    // Explicit outputType wins; otherwise PNG when alpha so holes stay transparent.
    const outputType =
      options?.outputType ?? (imageHasAlpha(image) ? 'image/png' : 'image/jpeg');

    emitJobStatus('analyzing', stageProgress('analyzing', 0));
    const thumb = downscaleForModel(image, 224);
    emitJobStatus('analyzing', stageProgress('analyzing', 0.4));
    const predictor = resolvePredictor(options?.predictorMode ?? 'heuristic', options?.modelUrl);
    const params = await predictor.predict(thumb.data, thumb.width, thumb.height);
    emitJobStatus('analyzing', stageProgress('analyzing', 1), {
      metrics: {
        elapsedMs: performance.now() - started,
        width: image.width,
        height: image.height,
        megapixels: (image.width * image.height) / 1_000_000,
        params,
        predictor: predictor.name,
      },
    });

    if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');

    emitJobStatus('applying', stageProgress('applying', 0));
    const corrected = await applyCorrection(image, params, {
      signal: controller.signal,
      onTile: (done, total) => {
        emitJobStatus('applying', stageProgress('applying', done / total));
      },
    });

    emitJobStatus('encoding', stageProgress('encoding', 0));
    const blob = await encodeImage(corrected, outputType, quality);
    const outBuf = await blob.arrayBuffer();
    emitJobStatus('encoding', stageProgress('encoding', 1));

    const metrics = {
      elapsedMs: performance.now() - started,
      width: image.width,
      height: image.height,
      megapixels: (image.width * image.height) / 1_000_000,
      params,
      predictor: predictor.name,
    };

    emitJobStatus('done', 1, { metrics });
    post(
      {
        type: 'result',
        id,
        buffer: outBuf,
        mime: blob.type || outputType,
        params,
      },
      [outBuf],
    );
  } catch (e) {
    if ((e as DOMException)?.name === 'AbortError' || controller.signal.aborted) {
      emitJobStatus('cancelled', 0);
      return;
    }
    const message =
      e instanceof DecodeError ? e.message : e instanceof Error ? e.message : String(e);
    emitJobStatus('error', 0, { error: message });
  } finally {
    jobs.delete(id);
  }
}

self.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'ping':
      post({ type: 'pong' });
      break;
    case 'cancel': {
      const job = jobs.get(msg.id);
      if (job) {
        job.controller.abort();
        emitStatus(msg.id, 'cancelled', 0);
      }
      break;
    }
    case 'submit':
      void runJob(msg.id, msg.buffer, msg.mime, msg.options);
      break;
    default:
      break;
  }
};
