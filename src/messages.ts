/** Worker ↔ main message protocol. */

import type { CorrectionParams, SubmitOptions, TaskId, TaskInfo, TaskStatus } from './types.js';

export type MainToWorker =
  | {
      type: 'submit';
      id: TaskId;
      buffer: ArrayBuffer;
      mime?: string;
      options?: SubmitOptions;
    }
  | { type: 'cancel'; id: TaskId }
  | { type: 'ping' };

export type WorkerToMain =
  | { type: 'status'; info: TaskInfo }
  | { type: 'result'; id: TaskId; buffer: ArrayBuffer; mime: string; params: CorrectionParams }
  | { type: 'pong' };

export function makeTaskInfo(
  id: TaskId,
  status: TaskStatus,
  progress: number,
  extra?: Partial<TaskInfo>,
): TaskInfo {
  return { id, status, progress, ...extra };
}
