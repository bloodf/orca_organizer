import { computed, reactive } from 'vue';
import { pickLang, strings, type Lang } from './messages';
import type { Bridge, Ctx, HostMessage, Rotation, StaleRef } from './protocol';
import { describe, jobRef, splitLines, translate, type StatusLine, type StatusSource, type StatusView } from './status';

const BUSY_POLL_MS = 1000;
const IDLE_POLL_MS = 3000;
const SILENCE_MS = 10_000;

/** Element ids the panel moves focus to. Components render these ids. */
export type FocusId = 'spacing' | 'go' | 'fill' | 'cancel' | 'open';

export interface OrganizerOptions {
  /** The browser's language; the host's language (from the plugin) wins once it is known. */
  navigatorLanguage?: string | null;
  now?: () => number;
  /** Called with the new theme-probing request on every tick (see hostTheme.ts). */
  onTick?: () => void;
}

/**
 * All state of the panel. The plugin (Python) owns the job; the page only shows its state and sends requests,
 * so a closed, reloaded or second page (Organizer tab and dock) always agrees with it.
 */
export function createOrganizer(bridge: Bridge, options: OrganizerOptions = {}) {
  const now = options.now ?? Date.now;
  const navLang = options.navigatorLanguage ?? (typeof navigator !== 'undefined' ? navigator.language : null);

  const state = reactive({
    hostLang: null as string | null,
    ctx: null as Ctx | null,
    context: '',
    stale: '',
    staleRef: null as StaleRef | null,
    status: null as StatusSource | null,
    offline: false,
    retrying: false,
    busy: false,
    cancelling: false,
    path: null as string | null,
    resultVisible: false,
    pathNote: null as 'copied' | 'copyFail' | null,
    clipboard: false,
    spacing: 2 as number | null,
    rotation: 'auto' as Rotation,
    spacingInvalid: false,
    validation: false, // a local input error: shown over whatever the plugin says, until the input changes
    focus: null as { id: FocusId; seq: number } | null,
  });

  // Plain bookkeeping that never needs to redraw anything.
  let job: string | null = null;
  let done: string | null = null;
  let last = now();
  let retryAt = 0;
  let focusSeq = 0;
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const lang = computed<Lang>(() => pickLang(state.hostLang || navLang));
  const s = computed(() => strings(lang.value));
  const t = (key: string, args?: Record<string, unknown>) => translate(lang.value, key, args);
  const ref = (m: { stem?: string | null; plate?: number }) => jobRef(lang.value, m);

  const loading = computed(() => state.ctx == null && !state.context);
  const ctxProblem = computed(() => (state.ctx && state.ctx.problem ? s.value.ctxProblem({ problem: state.ctx.problem }) : ''));
  const ctxText = computed(() => {
    const c = state.ctx;
    if (c && c.plate) return s.value.ctx({ name: c.name || s.value.untitledProject, plate: c.plate, objects: c.objects });
    if (ctxProblem.value) return ctxProblem.value;
    return state.context || s.value.reading;
  });
  const goLabel = computed(() => (state.ctx && state.ctx.plate ? s.value.go({ plate: state.ctx.plate }) : s.value.goNone));
  const fillLabel = computed(() => (state.ctx && state.ctx.plate ? s.value.fill({ plate: state.ctx.plate }) : s.value.fillNone));
  const staleText = computed(() => (state.staleRef ? s.value.stale({ ref: ref(state.staleRef) }) : state.stale));

  const statusView = computed<StatusView>(() => {
    const line = (text: string, kind: StatusLine['kind'] = 'info') => splitLines('info', kind, text);
    const src = state.status;
    if (src?.kind === 'starting') return { lines: line(s.value.starting), error: false, progress: null };
    if (src?.kind === 'job') {
      const described = src.msg ? describe(lang.value, ref, src.msg) : null;
      if (described) return { lines: described.lines, error: src.error, progress: state.busy ? described.progress : null };
      const text = src.hasPath
        ? src.text
            .split('\n')
            .filter((l) => !l.startsWith('Saved as: '))
            .join('\n')
        : src.text;
      return { lines: line(text), error: src.error, progress: null };
    }
    if (state.ctx && state.ctx.objects === 0) return { lines: line(s.value.emptyProject), error: false, progress: null };
    return { lines: line(s.value.ready), error: false, progress: null };
  });

  const progress = computed(() => statusView.value.progress);
  const hasCount = computed(() => !!progress.value && (progress.value.n != null || progress.value.count != null));
  const percent = computed(() => {
    const p = progress.value;
    return p && p.n != null && p.n > 0 ? Math.round((p.i / p.n) * 100) : null;
  });
  const countText = computed(() => (progress.value && hasCount.value ? s.value.count(progress.value) : ''));

  const activeId = (): string => {
    const a = typeof document !== 'undefined' ? document.activeElement : null;
    return !a || a === document.body ? 'body' : a.id;
  };
  const requestFocus = (id: FocusId) => {
    state.focus = { id, seq: ++focusSeq };
  };

  const detectClipboard = () => {
    state.clipboard = typeof navigator !== 'undefined' && !!navigator.clipboard && typeof navigator.clipboard.writeText === 'function';
  };

  function setBusy(busy: boolean) {
    const was = state.busy;
    const active = activeId(); // read before the controls change state: they may lose focus when disabled
    state.busy = busy;
    if (!busy) state.cancelling = false;
    // Keep focus where the user can still act: Cancel while running, the result (or Arrange) afterwards.
    if (busy && !was && (active === 'go' || active === 'fill')) requestFocus('cancel');
    if (!busy && was && (active === 'cancel' || active === 'body')) requestFocus(state.resultVisible ? 'open' : 'go');
  }

  function onMessage(m: HostMessage) {
    if (!alive) return;
    last = now();
    if (state.offline || state.retrying) {
      state.offline = false;
      state.retrying = false;
    }
    if (m.lang) state.hostLang = m.lang;
    if ('context' in m) state.context = m.context || '';
    if (m.ctx) state.ctx = m.ctx;
    if ('stale' in m) {
      state.stale = m.stale || '';
      state.staleRef = m.stale_ref || null;
    }
    if (m.sync || m.start) {
      if (m.job !== job) state.cancelling = false;
      job = m.job ?? null;
      if (!m.path) {
        state.resultVisible = false;
        state.path = null;
      }
    } else if ('job' in m && m.job !== job) {
      return; // a late message of an older job
    }
    if (m.start) state.validation = false;
    if ('text' in m) {
      state.status = { kind: 'job', msg: m.msg || null, text: m.text ?? '', error: !!m.error, hasPath: !!m.path };
      if (m.msg?.code === 'start_failed') state.cancelling = false;
    }
    if (m.path) {
      done = m.job ?? null;
      if (state.path !== m.path) state.pathNote = null;
      state.path = m.path;
      state.resultVisible = true;
    }
    if ('busy' in m) setBusy(!!m.busy); // after the result is set: focus may land on Open
    detectClipboard();
    if (m.relay) bridge.postMessage({ action: m.relay as 'finish', job: m.job ?? null });
  }

  function run(action: 'arrange' | 'fill') {
    const v = state.spacing;
    if (v == null || !Number.isFinite(v) || v < 0) {
      state.validation = true;
      state.spacingInvalid = true;
      requestFocus('spacing');
      return;
    }
    state.validation = false;
    state.status = { kind: 'starting' };
    state.path = null;
    state.resultVisible = false;
    state.cancelling = false;
    setBusy(true);
    bridge.postMessage({ action, spacing: v, rotation: state.rotation });
  }

  function setSpacing(value: number | null) {
    state.spacing = value;
    state.spacingInvalid = false;
    state.validation = false;
  }

  function cancel() {
    if (!state.busy || state.cancelling) return;
    state.cancelling = true;
    bridge.postMessage({ action: 'cancel', job });
  }

  function open() {
    bridge.postMessage({ action: 'open', job: done });
  }

  /** The WebView may refuse the clipboard: then `select` marks the path for Ctrl/Cmd+C and the note says so. */
  function copyPath(select: () => void) {
    const text = state.path ?? '';
    const fail = () => {
      try {
        select();
      } catch {
        /* selection is a courtesy; the note below still tells the user what to do */
      }
      state.pathNote = 'copyFail';
    };
    try {
      navigator.clipboard.writeText(text).then(() => {
        state.pathNote = 'copied';
      }, fail);
    } catch {
      fail();
    }
  }

  function retry() {
    state.retrying = true;
    retryAt = now();
    bridge.postMessage({ action: 'hello' });
    bridge.postMessage({ action: 'poll' });
  }

  const poll = () => {
    if (!document.hidden) bridge.postMessage({ action: 'poll' });
  };

  function tick() {
    if (!alive) return;
    options.onTick?.();
    if (!document.hidden && now() - last > SILENCE_MS && now() - retryAt > SILENCE_MS) {
      state.offline = true;
      state.retrying = false;
    }
    poll();
    timer = setTimeout(tick, state.busy ? BUSY_POLL_MS : IDLE_POLL_MS);
  }

  const onVisible = () => {
    if (!document.hidden) {
      last = now();
      poll();
    }
  };

  /** Call once the page is on screen: hello (the plugin may have been reloaded) and the polling loop. */
  function start() {
    options.onTick?.();
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', poll);
    bridge.postMessage({ action: 'hello' });
    timer = setTimeout(tick, 1000);
  }

  function stop() {
    alive = false;
    if (timer !== undefined) clearTimeout(timer);
    document.removeEventListener('visibilitychange', onVisible);
    window.removeEventListener('focus', poll);
  }

  bridge.onMessage(onMessage);

  return {
    state, lang, s, t, loading, ctxText, ctxProblem, goLabel, fillLabel, staleText, statusView, progress, hasCount, percent, countText,
    run, setSpacing, cancel, open, copyPath, retry, start, stop, onMessage,
  };
}

export type Organizer = ReturnType<typeof createOrganizer>;
