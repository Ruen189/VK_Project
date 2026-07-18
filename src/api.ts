/**
 * Public ImageEnhancer API — task queue facade over a Dedicated Worker.
 */

import { ensureDecodableImage, setHeicDecoderUrl } from './heic.js';
import type { MainToWorker, WorkerToMain } from './messages.js';
import type {
  StatusListener,
  SubmitOptions,
  TaskId,
  TaskInfo,
} from './types.js';

export interface ImageEnhancerOptions {
  /** Custom Worker URL. Defaults to new URL('./worker.js', import.meta.url). */
  workerUrl?: string | URL;
  /** Optional URL to bundled heic2any ESM (default: ./vendor/heic2any.js). */
  heicDecoderUrl?: string | URL;
}

interface PendingResult {
  resolve: (blob: Blob) => void;
  reject: (err: Error) => void;
}

let idSeq = 0;
function nextId(): TaskId {
  idSeq += 1;
  return `task_${Date.now().toString(36)}_${idSeq}`;
}

export class ImageEnhancer {
  private worker: Worker;
  private tasks = new Map<TaskId, TaskInfo>();
  private results = new Map<TaskId, Blob>();
  private pending = new Map<TaskId, PendingResult>();
  private listeners = new Set<StatusListener>();
  private disposed = false;

  constructor(options: ImageEnhancerOptions = {}) {
    if (options.heicDecoderUrl) {
      setHeicDecoderUrl(String(options.heicDecoderUrl));
    }
    const url = options.workerUrl ?? new URL('./worker.js', import.meta.url);
    this.worker = new Worker(url, { type: 'module' });
    this.worker.onmessage = (ev: MessageEvent<WorkerToMain>) => this.onWorkerMessage(ev.data);
    this.worker.onerror = (err) => {
      console.error('[ImageEnhancer] worker error', err);
    };
  }

  /** Subscribe to status/progress changes. Returns unsubscribe. */
  on(event: 'status', cb: StatusListener): () => void {
    if (event !== 'status') throw new Error(`Unknown event: ${event}`);
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  /** Enqueue an image for enhancement. Returns task id immediately after transfer. */
  async submit(
    image: Blob | ArrayBuffer | ImageBitmap,
    options?: SubmitOptions,
  ): Promise<TaskId> {
    this.assertAlive();
    const id = nextId();
    const { buffer, mime } = await toArrayBuffer(image);

    const info: TaskInfo = { id, status: 'queued', progress: 0 };
    this.tasks.set(id, info);
    this.emit(info);

    const msg: MainToWorker = { type: 'submit', id, buffer, mime, options };
    this.worker.postMessage(msg, [buffer]);
    return id;
  }

  getStatus(taskId: TaskId): TaskInfo {
    const info = this.tasks.get(taskId);
    if (!info) {
      return { id: taskId, status: 'error', progress: 0, error: 'Unknown task id' };
    }
    return { ...info };
  }

  /** Request cancellation. Returns true if a running/queued task was signalled. */
  cancel(taskId: TaskId): boolean {
    this.assertAlive();
    const info = this.tasks.get(taskId);
    if (!info) return false;
    if (info.status === 'done' || info.status === 'error' || info.status === 'cancelled') {
      return false;
    }
    const msg: MainToWorker = { type: 'cancel', id: taskId };
    this.worker.postMessage(msg);
    return true;
  }

  /** Resolve with output Blob when status is done. */
  getResult(taskId: TaskId): Promise<Blob> {
    const cached = this.results.get(taskId);
    if (cached) return Promise.resolve(cached);

    const info = this.tasks.get(taskId);
    if (!info) return Promise.reject(new Error('Unknown task id'));
    if (info.status === 'error') return Promise.reject(new Error(info.error ?? 'Task failed'));
    if (info.status === 'cancelled') return Promise.reject(new Error('Task cancelled'));

    return new Promise<Blob>((resolve, reject) => {
      this.pending.set(taskId, { resolve, reject });
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.terminate();
    for (const [, p] of this.pending) {
      p.reject(new Error('Enhancer disposed'));
    }
    this.pending.clear();
    this.listeners.clear();
  }

  private onWorkerMessage(msg: WorkerToMain): void {
    switch (msg.type) {
      case 'status': {
        this.tasks.set(msg.info.id, msg.info);
        this.emit(msg.info);
        if (msg.info.status === 'error') {
          this.pending.get(msg.info.id)?.reject(new Error(msg.info.error ?? 'Task failed'));
          this.pending.delete(msg.info.id);
        }
        if (msg.info.status === 'cancelled') {
          this.pending.get(msg.info.id)?.reject(new Error('Task cancelled'));
          this.pending.delete(msg.info.id);
        }
        break;
      }
      case 'result': {
        const blob = new Blob([msg.buffer], { type: msg.mime });
        this.results.set(msg.id, blob);
        this.pending.get(msg.id)?.resolve(blob);
        this.pending.delete(msg.id);
        break;
      }
      default:
        break;
    }
  }

  private emit(info: TaskInfo): void {
    for (const cb of this.listeners) {
      try {
        cb(info);
      } catch (e) {
        console.error('[ImageEnhancer] listener error', e);
      }
    }
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error('ImageEnhancer is disposed');
  }
}

async function toArrayBuffer(
  image: Blob | ArrayBuffer | ImageBitmap,
): Promise<{ buffer: ArrayBuffer; mime?: string }> {
  let buffer: ArrayBuffer;
  let mime: string | undefined;

  if (image instanceof ArrayBuffer) {
    buffer = image.slice(0);
  } else if (typeof Blob !== 'undefined' && image instanceof Blob) {
    buffer = await image.arrayBuffer();
    mime = image.type || undefined;
  } else {
    // ImageBitmap → PNG via canvas
    const bmp = image as ImageBitmap;
    const canvas =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(bmp.width, bmp.height)
        : Object.assign(document.createElement('canvas'), {
            width: bmp.width,
            height: bmp.height,
          });
    const ctx = canvas.getContext('2d') as
      | OffscreenCanvasRenderingContext2D
      | CanvasRenderingContext2D
      | null;
    if (!ctx) throw new Error('Cannot encode ImageBitmap');
    ctx.drawImage(bmp, 0, 0);

    let blob: Blob;
    if ('convertToBlob' in canvas) {
      blob = await (canvas as OffscreenCanvas).convertToBlob({ type: 'image/png' });
    } else {
      blob = await new Promise<Blob>((resolve, reject) => {
        (canvas as HTMLCanvasElement).toBlob(
          (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
          'image/png',
        );
      });
    }
    buffer = await blob.arrayBuffer();
    mime = 'image/png';
  }

  // HEIC/HEIF → PNG on main thread (Chrome/Firefox)
  const decoded = await ensureDecodableImage(buffer, mime);
  return { buffer: decoded.buffer, mime: decoded.mime };
}
