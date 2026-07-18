/** Progress ranges per pipeline stage. */

import type { TaskStatus } from '../types.js';

export const PROGRESS: Record<
  Extract<TaskStatus, 'decoding' | 'analyzing' | 'applying' | 'encoding' | 'done'>,
  { start: number; end: number }
> = {
  decoding: { start: 0, end: 0.15 },
  analyzing: { start: 0.15, end: 0.45 },
  applying: { start: 0.45, end: 0.85 },
  encoding: { start: 0.85, end: 1 },
  done: { start: 1, end: 1 },
};

export function stageProgress(
  stage: keyof typeof PROGRESS,
  fractionWithinStage: number,
): number {
  const { start, end } = PROGRESS[stage];
  const t = Math.min(1, Math.max(0, fractionWithinStage));
  return start + (end - start) * t;
}
