import { ENGLISH, strings, type Lang, type Num, numberFormat } from './messages';
import type { Cause, JobMsg } from './protocol';

export type LineKind = 'main' | 'cause' | 'next' | 'detail' | 'info';

export interface StatusLine {
  key: string;
  kind: LineKind;
  text: string;
}

export interface Progress {
  i: number;
  n: number | null;
  count: number | null;
  subject: string;
}

export interface StatusView {
  lines: StatusLine[];
  error: boolean;
  progress: Progress | null;
}

/** What the last message told us to show. `starting` is local, before the plugin answers. */
export type StatusSource =
  | { kind: 'job'; msg: JobMsg | null; text: string; error: boolean; hasPath: boolean }
  | { kind: 'starting' };

type Args = Record<string, unknown>;

/** Dotted lookup with an English fallback; functions are called with `args`. */
export function translate(lang: Lang, key: string, args: Args = {}): string {
  const walk = (table: unknown): unknown =>
    key.split('.').reduce<unknown>((value, part) => (value == null ? value : (value as Args)[part]), table);
  const found = walk(strings(lang));
  const value = found == null ? walk(ENGLISH) : found;
  return typeof value === 'function' ? (value as (a: Args) => string)(args) : String(value);
}

/** Split a possibly multi-line text into keyed lines of one kind. */
function split(prefix: string, kind: LineKind, text: string): StatusLine[] {
  return text.split('\n').map((line, index) => ({ key: `${prefix}-${index}`, kind, text: line }));
}

function explain(lang: Lang, cause: Cause | undefined, detail: string | undefined) {
  const known = !!cause && cause.code !== 'unexpected' && cause.code in ENGLISH.err;
  const [why, next] = causeTable(lang, known ? cause.code : 'unexpected', cause);
  const extra = known || !detail ? [] : split('detail', 'detail', strings(lang).details + detail);
  return { why, lines: [...split('next', 'next', strings(lang).next + next), ...extra] };
}

function causeTable(lang: Lang, code: string, cause: Cause | undefined): [string, string] {
  const entry = strings(lang).err[code];
  const pair = typeof entry === 'function' ? (entry as (a: Cause) => [string, string])(cause ?? { code }) : entry;
  return [pair[0], pair[1]];
}

export interface DescribeResult {
  lines: StatusLine[];
  progress: Progress | null;
}

/** Status text and progress of a structured job message; null when the message has no translation. */
export function describe(lang: Lang, ref: (m: JobMsg) => string, m: JobMsg): DescribeResult | null {
  const t = strings(lang);
  const num: Num = numberFormat(lang);
  const r = m.plate != null ? ref(m) : '';
  const args = { ref: r };
  switch (m.code) {
    case 'reading':
      return { lines: split('info', 'info', r + ': ' + t.reading), progress: null };
    case 'progress': {
      const phase = m.phase && ENGLISH.phase[m.phase] ? t.phase[m.phase] : null;
      if (!phase) return null;
      return {
        lines: split('info', 'info', r + ': ' + phase),
        progress: { i: m.i ?? 0, n: m.n ?? null, count: m.count ?? null, subject: m.subject ?? '' },
      };
    }
    case 'cancelling':
      return { lines: split('info', 'info', t.cancelling(args)), progress: null };
    case 'cancelled':
      return { lines: split('info', 'info', t.cancelled(args)), progress: null };
    case 'saving':
      return { lines: split('info', 'info', t.saving(args)), progress: null };
    case 'failed': {
      const why = explain(lang, m.cause, m.detail);
      return {
        lines: [
          ...split('main', 'main', t.failed({ ref: r, act: t.act[m.action ?? 'arrange'] })),
          ...split('cause', 'cause', why.why),
          ...why.lines,
        ],
        progress: null,
      };
    }
    case 'save_failed': {
      const [first, ...rest] = t.saveFailed({ ref: r, detail: m.detail ?? '' }).split('\n');
      return { lines: [...split('main', 'main', first), ...split('info', 'info', rest.join('\n'))], progress: null };
    }
    case 'start_failed': {
      const why = explain(lang, m.cause, m.detail);
      return { lines: [...split('main', 'main', t.startFailed({ why: why.why })), ...why.lines], progress: null };
    }
    case 'done': {
      const out: string[] = [t.res.placed({ ref: r, placed: m.placed ?? 0 })];
      if (m.added) {
        const per = Object.entries(m.by_object ?? {})
          .map(([name, count]) => name + ' ×' + num(count))
          .join(', ');
        out.push(t.res.added({ added: m.added, per }));
        if (m.limit) out.push(t.res.limit({ max: m.max ?? 0 }));
      }
      if (m.tower) out.push(t.res.tower);
      if (m.clearance != null) out.push(t.res.clearance({ clearance: m.clearance }));
      if (m.keep) out.push(t.res.keep);
      if (m.unplaced && m.unplaced.length) out.push(t.res.unplaced({ names: m.unplaced.join(', ') }));
      out.push(t.res.foot);
      if (m.open) {
        const open = t.open_[m.open];
        out.push(typeof open === 'function' ? open({ detail: m.detail ?? '' }) : open);
      }
      return { lines: out.map((text, index) => ({ key: `res-${index}`, kind: index === 0 ? 'main' : 'info', text })), progress: null };
    }
  }
  return null;
}

/** The reference of a job in prose: "<project>, plate <n>". */
export function jobRef(lang: Lang, m: { stem?: string | null; plate?: number }): string {
  const t = strings(lang);
  return (m.stem || t.untitled) + ', ' + t.plate + ' ' + m.plate;
}

export const joinLines = (lines: readonly StatusLine[]): string => lines.map((l) => l.text).join('\n');

export { split as splitLines };
