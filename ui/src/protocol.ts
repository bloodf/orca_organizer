// Shapes of the messages between the plugin (Python) and the panel. Python is the source of truth:
// shape_arrange.py `_Job.post`, `_Session.sync` and `_Session.on_message`.

export interface Ctx {
  plate: number | null;
  objects: number | null;
  name: string | null;
  problem: string | null;
}

export interface Cause {
  code: string;
  plate?: number;
  plates?: number;
  unit?: string;
  res?: number;
}

export type MsgCode =
  | 'reading'
  | 'progress'
  | 'cancelling'
  | 'cancelled'
  | 'saving'
  | 'failed'
  | 'save_failed'
  | 'start_failed'
  | 'done';

/** Structured twin of the English `text`. Fields depend on `code`. */
export interface JobMsg {
  code: MsgCode;
  stem?: string | null;
  plate?: number;
  action?: 'arrange' | 'fill';
  phase?: string;
  i?: number;
  n?: number | null;
  count?: number | null;
  subject?: string;
  cause?: Cause;
  detail?: string;
  placed?: number;
  added?: number;
  by_object?: Record<string, number>;
  limit?: boolean;
  max?: number;
  tower?: boolean;
  clearance?: number | null;
  keep?: boolean;
  unplaced?: string[];
  open?: 'asked' | 'later' | 'failed' | 'none';
}

export interface StaleRef {
  stem?: string | null;
  plate: number;
  action?: string;
}

/** A message from the plugin. Every field is optional: a poll answers with everything, a job post with part. */
export interface HostMessage {
  sync?: boolean;
  start?: boolean;
  job?: string | null;
  busy?: boolean;
  text?: string;
  error?: boolean;
  done?: boolean;
  path?: string | null;
  msg?: JobMsg | null;
  ctx?: Ctx | null;
  context?: string | null;
  stale?: string | null;
  stale_ref?: StaleRef | null;
  lang?: string | null;
  relay?: string;
}

export type Rotation = 'auto' | 'keep';

export type OutMessage =
  | { action: 'hello' }
  | { action: 'poll' }
  | { action: 'arrange' | 'fill'; spacing: number; rotation: Rotation }
  | { action: 'cancel' | 'open' | 'finish'; job: string | null };

/** The bridge the host injects into the page. */
export interface Bridge {
  postMessage(message: OutMessage): void;
  onMessage(handler: (message: HostMessage) => void): void;
}

declare global {
  // eslint-disable-next-line no-var
  var orca: Bridge | undefined;
}

export function hostBridge(): Bridge {
  const bridge = globalThis.orca;
  if (!bridge || typeof bridge.postMessage !== 'function' || typeof bridge.onMessage !== 'function') {
    throw new Error('The Orca bridge (orca.postMessage / orca.onMessage) is not available in this page.');
  }
  return bridge;
}
