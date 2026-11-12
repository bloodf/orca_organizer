// Behaviour of the real Vue panel (App.vue and its components) against a fake `orca` bridge.
// Ported from the legacy Node harness (tests/test_panel_js.py, sections 0-13 of the old file) so that every
// state, message and protocol assertion of the imperative page still holds for the Vue one.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobMsg } from '../src/protocol';
import { CTX, mountPanel, REF, type Panel } from './helpers';

let panel: Panel;
beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
});
afterEach(() => {
  panel?.unmount();
  vi.useRealTimers();
});

const sync = (extra: Record<string, unknown> = {}) => ({ sync: true, job: null, busy: false, ...extra });
const prog = (p: Panel, phase: string, i: number, n: number | null, extra: Record<string, unknown> = {}) =>
  p.deliver({ job: 'j1', busy: true, text: 'EN TEXT', msg: { ...REF, code: 'progress', phase, i, n, count: null, subject: '', ...extra } as never });
const doneMsg: JobMsg = {
  stem: 'P', plate: 2, action: 'fill', code: 'done', placed: 1, added: 1, by_object: { block: 1 }, limit: false, max: 300,
  tower: false, clearance: null, keep: false, unplaced: [], open: 'asked',
};

describe('loading, language and context', () => {
  it('loads in the browser language and says hello', async () => {
    panel = await mountPanel();
    expect(panel.take()).toEqual([{ action: 'hello' }]);
    expect(panel.text('#context')).toBe('Reading project…');
    expect(panel.q('[role=progressbar]')).not.toBeNull(); // loading state is visible, not only text
    expect(document.documentElement.lang).toBe('en');
    expect(panel.text('h1')).toBe('Organizer'); // O01: name
    expect(panel.text('#subtitle')).toMatch(/^Shape-Aware Arrange/); // O01: Script name as subtitle
    expect(document.title).toBe('Organizer');
  });

  it('shows the plate in the buttons, the project in the context and the empty-project hint', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ context: 'x', ctx: { plate: 2, objects: 0, name: 'P', problem: null }, stale: '' }));
    expect(panel.text('#go')).toContain('Arrange plate 2');
    expect(panel.text('#fill')).toContain('Fill plate 2');
    expect(panel.text('#context')).toBe('P · plate 2 selected · 0 objects in the project');
    expect(panel.status()).toMatch(/No objects found/);
    expect(panel.q('[role=progressbar]')).toBeNull();
  });

  it('host language wins, pt_BR and pt-BR both work, unsupported falls back to English (X04)', async () => {
    panel = await mountPanel();
    const ctx = (objects: number, name: string | null) => ({ plate: 2, objects, name, problem: null });
    await panel.deliver(sync({ lang: 'pt_BR', ctx: ctx(1, null) }));
    expect(document.documentElement.lang).toBe('pt-BR');
    expect(panel.text('#go')).toContain('Arrumar placa 2');
    expect(panel.text('#fill')).toContain('Preencher placa 2');
    expect(panel.text('#context')).toBe('Projeto sem título · placa 2 selecionada · 1 objeto no projeto');
    expect(panel.text('h1')).toBe('Organizer');
    expect(panel.status()).toBe('Pronto. O resultado é salvo em um arquivo novo; o seu projeto não é alterado.');
    expect(document.title).toBe('Organizer');
    await panel.deliver(sync({ lang: 'pt-BR', ctx: ctx(2, 'P') }));
    expect(document.documentElement.lang).toBe('pt-BR');
    await panel.deliver(sync({ lang: 'fr_FR', ctx: ctx(2, 'P') }));
    expect(document.documentElement.lang).toBe('en');
    await panel.deliver(sync({ lang: null, ctx: ctx(2, 'P') }));
    expect(document.documentElement.lang).toBe('en'); // no host language: the browser's
    await panel.deliver(sync({ lang: 'en_US', ctx: ctx(2, 'P') }));
  });

  it('uses the browser language until the host says otherwise', async () => {
    panel = await mountPanel({ language: 'pt-BR' });
    expect(document.documentElement.lang).toBe('pt-BR');
    expect(panel.text('#context')).toBe('Lendo o projeto…');
    await panel.deliver(sync({ lang: 'en_US', ctx: CTX }));
    expect(document.documentElement.lang).toBe('en');
  });

  it('names an unreadable project and how to recover (recoverable error)', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: { plate: null, objects: null, name: null, problem: 'no project' } }));
    expect(panel.text('#context')).toMatch(/Could not read the project: no project\. Open a project/);
    await panel.deliver(sync({ ctx: CTX }));
    expect(panel.q('#stale')).toBeNull();
    expect(panel.q('#result')).toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
  });
});

describe('spacing validation', () => {
  it('sends nothing for a blank value, shows the error, focuses the field and keeps it through polls', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.take();
    await panel.typeSpacing('');
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.take()).toEqual([]);
    expect(panel.text('#spacing-error')).toBe('Enter a spacing of 0 mm or more.'); // next to the field it explains
    expect(panel.q('#spacing-error')!.getAttribute('role')).toBe('alert');
    expect(panel.q('#spacing')!.getAttribute('aria-describedby')).toContain('spacing-error');
    expect(document.activeElement).toBe(panel.q('#spacing'));
    expect(panel.q('#spacing')!.getAttribute('aria-invalid')).toBe('true');
    await panel.deliver(sync({ job: 'old', text: 'old job text', msg: null }));
    expect(panel.q('#spacing-error')).not.toBeNull(); // a poll must not erase the validation error
    await panel.typeSpacing('2');
    expect(panel.q('#spacing-error')).toBeNull();
    expect(panel.q('#spacing')!.getAttribute('aria-invalid')).not.toBe('true');
    expect(panel.status()).toBe('old job text');
  });

  it('rejects a negative value that was pasted', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.take();
    panel.organizer.setSpacing(-1);
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.take()).toEqual([]);
    expect(panel.q('#spacing-error')).not.toBeNull();
  });

  it('accepts a decimal typed with the language comma', async () => {
    panel = await mountPanel({ language: 'pt-BR' });
    await panel.deliver(sync({ ctx: CTX }));
    panel.take();
    await panel.typeSpacing('2,5');
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: 2.5, rotation: 'auto' }]);
  });
});

describe('a run', () => {
  async function started(): Promise<Panel> {
    const p = await mountPanel();
    await p.deliver(sync({ ctx: CTX }));
    p.take();
    return p;
  }

  it('locks the controls, moves focus to Cancel, shows progress and sends the options', async () => {
    panel = await started();
    await panel.typeSpacing('3');
    panel.organizer.state.rotation = 'keep';
    panel.q<HTMLButtonElement>('#fill')!.focus();
    panel.q('#fill')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'fill', spacing: 3, rotation: 'keep' }]);
    for (const id of ['#go', '#fill', '#spacing']) expect(panel.q<HTMLButtonElement>(id)!.disabled, id).toBe(true);
    expect(panel.q('#cancel')).not.toBeNull();
    expect(panel.q('#bar')).not.toBeNull();
    expect(document.activeElement).toBe(panel.q('#cancel'));
    expect(panel.q('#status')!.getAttribute('aria-busy')).toBe('true');
    expect(panel.isError()).toBe(false);
    expect(panel.status()).toBe('Starting…');
    expect(panel.text('#busy-hint')).toMatch(/run is in progress/); // a disabled control says why
  });

  it('draws determinate progress from structured counts and keeps the live text for phases only', async () => {
    panel = await started();
    panel.q<HTMLButtonElement>('#fill')!.focus();
    panel.q('#fill')!.click();
    panel.take();
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'reading', msg: { stem: 'P', plate: 2, action: 'fill', code: 'reading' }, ctx: CTX });
    expect(panel.status()).toBe('P, plate 2: Reading project…');
    await prog(panel, 'read', 0, null);
    expect(panel.q('#count')).toBeNull();
    expect(panel.q('#bar')!.getAttribute('aria-valuenow')).toBeNull(); // read: indeterminate
    await prog(panel, 'place', 0, 14, { subject: 'block' });
    expect(panel.status()).toBe('P, plate 2: Placing parts');
    expect(panel.q('#bar')!.getAttribute('aria-valuenow')).toBe('0');
    expect(panel.text('#count')).toBe('1 of 14 · block');
    const live = panel.q('#status')!;
    const mutations: MutationRecord[] = [];
    const observer = new MutationObserver((records) => mutations.push(...records));
    observer.observe(live, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['class'] });
    for (let i = 1; i < 14; i++) await prog(panel, 'place', i, 14, { subject: 'block ' + i });
    await Promise.resolve();
    observer.disconnect();
    expect(panel.q('#bar')!.getAttribute('aria-valuenow')).toBe('93'); // 13 of 14
    expect(panel.text('#count')).toBe('14 of 14 · block 13');
    expect(mutations, 'the live region is not rewritten on every poll').toHaveLength(0);
    await prog(panel, 'fill', 1, null, { count: 1 });
    expect(panel.q('#bar')!.getAttribute('aria-valuenow')).toBeNull();
    expect(panel.text('#count')).toBe('1 copy added');
    await prog(panel, 'fill', 5, null, { count: 5 });
    expect(panel.text('#count')).toBe('5 copies added');
    expect(panel.status()).toBe('P, plate 2: Adding copies');
    expect(document.activeElement).toBe(panel.q('#cancel')); // polls do not move focus
    await panel.deliver(sync({ lang: 'pt_BR', job: 'j1', busy: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'fill', code: 'progress', phase: 'place', i: 6, n: 14, count: null, subject: 'b' } }));
    expect(panel.status()).toBe('P, placa 2: Posicionando as peças');
    expect(panel.text('#count')).toBe('7 de 14 · b');
    await panel.deliver({ job: 'old', text: 'LATE', busy: false });
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(true);
    expect(panel.status()).not.toBe('LATE'); // a late message of an older job is ignored
  });

  it('measures and places with a determinate bar, reads and fills with an indeterminate one (O04)', async () => {
    panel = await started();
    panel.q('#go')!.click();
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'reading' } });
    for (const [phase, determinate] of [['read', false], ['footprint', true], ['place', true], ['patterns', true], ['fill', false]] as const) {
      await prog(panel, phase, 1, determinate ? 4 : null, phase === 'fill' ? { count: 3 } : {});
      expect(panel.q('#bar')!.getAttribute('aria-valuenow') !== null, phase).toBe(determinate);
    }
  });

  it('answers a relay with the job id', async () => {
    panel = await started();
    panel.q('#go')!.click();
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'reading' } });
    panel.take();
    await panel.deliver({ job: 'j1', text: 'saving…', busy: true, msg: { stem: 'P', plate: 2, action: 'fill', code: 'saving' }, relay: 'finish' });
    expect(panel.takeRequests()).toEqual([{ action: 'finish', job: 'j1' }]);
  });
});

describe('result', () => {
  async function finished(): Promise<Panel> {
    const p = await mountPanel();
    await p.deliver(sync({ ctx: CTX }));
    p.q<HTMLButtonElement>('#fill')!.focus();
    p.q('#fill')!.click();
    await p.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'fill', code: 'reading' } });
    p.take();
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async () => {} } });
    await p.deliver({ job: 'j1', text: 'done text', busy: false, done: true, path: '/s/P-arranged.3mf', msg: doneMsg });
    return p;
  }

  it('shows the path, Open and Copy, frees the controls and moves focus to Open', async () => {
    panel = await finished();
    expect(panel.text('#path')).toBe('/s/P-arranged.3mf'); // O02: the path is visible
    expect(panel.q('#result')).not.toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#cancel')).toBeNull();
    expect(panel.q('#bar')).toBeNull();
    expect(panel.q('#count')).toBeNull();
    expect(document.activeElement).toBe(panel.q('#open'));
    expect(panel.q('#copy')).not.toBeNull();
    expect(panel.status().split('\n')[0]).toBe('P, plate 2: placed 1 object.');
    expect(panel.status()).toMatch(/Filled the plate with 1 extra copy \(block ×1\)\./);
    expect(panel.status()).toMatch(/Asked OrcaSlicer to open the copy\. If nothing appears, use Open this copy in OrcaSlicer\./);
    expect(panel.status().split('\n').slice(-2).join(' ').replace(/Open this copy/, '')).not.toMatch(/saved|replace|window/i); // neutral wording
    expect(panel.text('#open')).toBe('Open this copy in OrcaSlicer');
    panel.q('#open')!.click();
    expect(panel.takeRequests()).toEqual([{ action: 'open', job: 'j1' }]);
  });

  it('keeps the path once when the plugin sends only English text', async () => {
    panel = await finished();
    await panel.deliver({ job: 'j1', text: 'line one\nSaved as: /s/P-arranged.3mf\nline three', busy: false, path: '/s/P-arranged.3mf', msg: null });
    expect(panel.status()).toBe('line one\nline three');
  });

  it('translates the outcome of opening the copy', async () => {
    panel = await finished();
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', lang: 'pt_BR', msg: { ...doneMsg, placed: 3, added: 0, by_object: {}, open: 'later' } });
    expect(panel.status().split('\n')[0]).toBe('P, placa 2: 3 objetos posicionados.');
    expect(panel.status()).toMatch(/Concluído com o painel fechado\. Use Abrir esta cópia no OrcaSlicer/);
    expect(panel.text('#open')).toBe('Abrir esta cópia no OrcaSlicer');
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, placed: 1, added: 0, by_object: {}, open: 'failed', detail: 'boom' } });
    expect(panel.status()).toMatch(/Não foi possível pedir ao OrcaSlicer que abrisse a cópia \(boom\)/);
  });

  it('copies the path, falls back to selecting it when the clipboard refuses, and hides Copy without the API', async () => {
    panel = await finished();
    let copied: string | null = null;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: (t: string) => ((copied = t), Promise.resolve()) } });
    panel.q('#copy')!.click();
    expect(copied).toBe('/s/P-arranged.3mf');
    await panel.flush();
    expect(panel.text('#path-note')).toBe('Path copied.');
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.reject(new Error('denied')) } });
    panel.q('#copy')!.click();
    await panel.flush();
    await panel.flush();
    expect(panel.text('#path-note')).toMatch(/The path is selected: press Ctrl\+C/);
    const range = getSelection()!.getRangeAt(0); // the fallback really selected the path element's text
    expect(range.startContainer).toBe(panel.q('#path'));
    expect(range.endOffset).toBe(panel.q('#path')!.childNodes.length);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: doneMsg });
    expect(panel.q('#copy')).toBeNull(); // no clipboard API: no Copy button
  });

  it('keeps the copy note through polls of the same result and clears it for a new one', async () => {
    panel = await finished();
    panel.q('#copy')!.click();
    await panel.flush();
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#path-note')).not.toBeNull();
    await panel.deliver(sync({ job: 'j2', busy: false, path: '/s/Q-arranged.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#path-note')).toBeNull();
  });

  it('shows the stale note, structured and translated, and hides it again', async () => {
    panel = await finished();
    const stale = 'The run below is for P, plate 2, not the current selection.';
    const keep = { job: 'j1', busy: false, text: 'done text', path: '/s/P-arranged.3mf', msg: doneMsg };
    await panel.deliver(sync({ ...keep, ctx: { plate: 1, objects: 1, name: 'Q', problem: null }, stale, stale_ref: { stem: 'P', plate: 2, action: 'arrange' } }));
    expect(panel.text('#stale')).toBe(stale);
    expect(panel.q('#result')).not.toBeNull();
    await panel.deliver(sync({ ...keep, lang: 'pt_BR', stale: 'x', stale_ref: { stem: null, plate: 2, action: 'arrange' } }));
    expect(panel.text('#stale')).toBe('A execução abaixo é de sem título, placa 2, não da seleção atual.');
    await panel.deliver(sync({ ...keep, lang: 'en_US', stale: '', stale_ref: null }));
    expect(panel.q('#stale')).toBeNull();
  });

  it('hides an old result on a reload with no finished job', async () => {
    panel = await finished();
    await panel.deliver(sync({ msg: null }));
    expect(panel.q('#result')).toBeNull();
  });
});

describe('errors (O03, B-27)', () => {
  const fail = (p: Panel, cause: unknown, detail: string, lang: string) =>
    p.deliver(sync({ lang, job: 'j2', error: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'failed', cause, detail } as never }));

  it('explains the cause and the next step in both languages, with technical detail only when the cause is unknown', async () => {
    panel = await mountPanel();
    await fail(panel, { code: 'plate_empty', plate: 2 }, 'x', 'pt_BR');
    expect(panel.status()).toBe('P, placa 2: não foi possível arrumar.\nA placa 2 não tem objetos imprimíveis.\nPróximo passo: Selecione outra placa ou adicione peças a esta.');
    expect(panel.isError()).toBe(true);
    expect(panel.q('#status-error')!.getAttribute('role')).toBe('alert');
    await fail(panel, { code: 'unexpected' }, 'zip is corrupt', 'en_US');
    expect(panel.status()).toMatch(/^P, plate 2: could not arrange\.\nSomething unexpected went wrong\.\nNext: Try again\..*\nDetails: zip is corrupt$/s);
    expect(panel.status()).not.toMatch(/error:/i); // no machine prefix
    expect(panel.q('.detail')!.textContent).toBe('Details: zip is corrupt');
    await fail(panel, { code: 'plate_gone', plate: 9, plates: 1 }, 'x', 'en_US');
    expect(panel.status()).toMatch(/does not exist: the project has 1 plate\./);
    expect(panel.status()).not.toMatch(/Details/);
  });

  it('translates failures to start and to save, and shows an older plugin text as it is', async () => {
    panel = await mountPanel();
    await panel.deliver({ text: 'EN', error: true, busy: false, lang: 'pt_BR', msg: { code: 'start_failed', cause: { code: 'bad_rotation' }, detail: 'd' } });
    expect(panel.status()).toBe('Não foi possível iniciar: A rotação deve ser automática ou manter a orientação.\nPróximo passo: Escolha uma das duas opções.');
    await panel.deliver(sync({ lang: 'en_US', job: 'j2', error: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'save_failed', detail: 'disk full' } }));
    expect(panel.status()).toMatch(/could not save the arranged copy: disk full\nNothing was opened/);
    await panel.deliver(sync({ job: 'j2', error: true, text: 'Some message from an older plugin', msg: null }));
    expect(panel.status()).toBe('Some message from an older plugin');
  });

  it('start failure re-enables Cancel and the controls stay usable', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.q('#go')!.click();
    await panel.deliver({ text: 'EN', error: true, busy: false, msg: { code: 'start_failed', cause: { code: 'unexpected' }, detail: 'd' } });
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.isError()).toBe(true);
  });
});

describe('cancel', () => {
  it('sends the job id, shows the pending state and stays pending through polls until the job ends', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    await panel.deliver(sync({ job: 'j3', busy: true, text: 'running', msg: null }));
    panel.take();
    panel.q<HTMLButtonElement>('#cancel')!.focus();
    panel.q('#cancel')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'cancel', job: 'j3' }]);
    expect(panel.q('#cancel')!.getAttribute('aria-disabled')).toBe('true');
    expect(document.activeElement).toBe(panel.q('#cancel'));
    panel.q('#cancel')!.click();
    expect(panel.takeRequests()).toEqual([]);
    expect(panel.text('#cancel')).toContain('Cancelling…');
    await panel.deliver(sync({ job: 'j3', busy: true, text: 'cancelling', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'cancelling' } }));
    expect(panel.q('#cancel')!.getAttribute('aria-disabled'), 'a poll must not re-enable Cancel').toBe('true');
    expect(document.activeElement).toBe(panel.q('#cancel'));
    panel.q('#cancel')!.click();
    expect(panel.takeRequests()).toEqual([]);
    await panel.deliver({ job: 'j3', text: 'cancelled', busy: false, done: true, msg: { stem: 'P', plate: 2, action: 'arrange', code: 'cancelled' } });
    expect(panel.status()).toBe('P, plate 2: cancelled. Nothing was written or opened.');
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#result')).toBeNull();
  });

  it('moves focus back to Arrange when the cancelled run ends with Cancel focused', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    await panel.deliver(sync({ job: 'j3', busy: true, text: 'running', msg: null }));
    panel.q<HTMLButtonElement>('#cancel')!.focus();
    await panel.deliver({ job: 'j3', text: 'cancelled', busy: false, done: true, msg: { stem: 'P', plate: 2, action: 'arrange', code: 'cancelled' } });
    expect(document.activeElement).toBe(panel.q('#go'));
  });
});

describe('polling, offline and retry (B-27)', () => {
  it('asks while visible, is silent while hidden, then shows the offline state with a retry', async () => {
    vi.useFakeTimers();
    panel = await mountPanel();
    await panel.deliver(sync({ job: null, msg: null, ctx: { plate: 1, objects: 2, name: 'P', problem: null } }));
    panel.take();
    await vi.advanceTimersByTimeAsync(1000);
    expect(panel.take()).toEqual([{ action: 'poll' }]);
    panel.setHidden(true);
    await vi.advanceTimersByTimeAsync(3000);
    expect(panel.take(), 'a hidden page must not poll').toEqual([]);
    expect(panel.q('#offline')).toBeNull();
    expect(panel.text('#context')).toBe('P · plate 1 selected · 2 objects in the project');
    panel.setHidden(false);
    await vi.advanceTimersByTimeAsync(11000);
    expect(panel.q('#offline')).not.toBeNull();
    expect(panel.text('#offline-text')).toMatch(/No answer from the plugin/);
    expect(panel.text('#offline-text')).not.toMatch(/reload/i); // the host offers no reload API
    expect(panel.take().length).toBeGreaterThan(0); // it keeps asking by itself
    panel.q<HTMLButtonElement>('#spacing')!.focus();
    panel.q('#retry')!.click();
    await panel.flush();
    expect(panel.take()).toEqual([{ action: 'hello' }, { action: 'poll' }]);
    expect(panel.text('#retry')).toContain('Trying…');
    expect(document.activeElement).toBe(panel.q('#spacing')); // retry does not steal focus
    await panel.deliver(sync({ lang: 'pt_BR', ctx: { plate: 1, objects: 2, name: 'P', problem: null }, msg: null }));
    expect(panel.q('#offline')).toBeNull();
    await panel.deliver(sync({ lang: 'en_US', job: 'j4', busy: true, text: 'running', msg: null }));
    panel.take();
    await vi.advanceTimersByTimeAsync(3000);
    panel.take();
    await vi.advanceTimersByTimeAsync(1000); // busy: every second
    expect(panel.take()).toEqual([{ action: 'poll' }]);
  });

  it('asks right away when the tab becomes visible again, and a sync never moves focus', async () => {
    vi.useFakeTimers();
    panel = await mountPanel();
    await panel.deliver(sync({ msg: null }));
    panel.take();
    panel.q<HTMLInputElement>('#spacing')!.focus();
    document.dispatchEvent(new Event('visibilitychange'));
    expect(panel.take()).toEqual([{ action: 'poll' }]);
    await panel.deliver(sync({ msg: null, ctx: { plate: 3, objects: 2, name: 'P', problem: null } }));
    expect(document.activeElement).toBe(panel.q('#spacing'));
  });

  it('stops polling after the page is unmounted (lifecycle cleanup)', async () => {
    vi.useFakeTimers();
    panel = await mountPanel();
    panel.take();
    panel.unmount();
    await vi.advanceTimersByTimeAsync(10000);
    expect(panel.take()).toEqual([]);
    panel = await mountPanel(); // so afterEach has something to unmount
  });
});

describe('structure and accessibility', () => {
  it('uses PrimeVue controls and labelled, described fields', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    expect(document.querySelectorAll('.p-button').length).toBeGreaterThanOrEqual(2);
    expect(panel.q('.p-inputnumber')).not.toBeNull();
    expect(panel.q('.p-accordion')).not.toBeNull();
    expect(panel.q('#spacing')!.getAttribute('aria-describedby')).toBe('spacing-help');
    expect(document.querySelector('label[for=spacing]')!.textContent).toBe('Spacing between parts (mm)');
    expect(panel.q('#go')!.getAttribute('aria-describedby')).toBe('go-help');
    expect(panel.q('#status')!.getAttribute('role')).toBe('status');
    expect(panel.q('#status')!.getAttribute('aria-live')).toBe('polite');
    const ids = [...document.querySelectorAll('[id]')].map((e) => e.id);
    expect(new Set(ids).size, `duplicate ids: ${ids}`).toBe(ids.length);
  });

  it('opens Advanced and changes the rotation with PrimeVue radio buttons', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.q('#adv')!.click();
    await panel.flush();
    expect(panel.q('fieldset')!.getAttribute('aria-describedby')).toBe('rotation-help');
    expect(panel.text('legend')).toBe('Rotation');
    expect(panel.q<HTMLInputElement>('#rotation-auto')!.checked).toBe(true);
    expect(document.querySelector('label[for=rotation-keep]')!.textContent).toBe("Keep each part's orientation");
    panel.q('#rotation-keep')!.click();
    await panel.flush();
    expect(panel.q<HTMLInputElement>('#rotation-keep')!.checked).toBe(true);
    panel.take();
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: 2, rotation: 'keep' }]);
    expect(panel.q<HTMLInputElement>('#rotation-keep')!.disabled).toBe(true); // locked while the run is busy
  });

  it('renders no raw HTML from messages (project and path names are text)', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: { plate: 1, objects: 1, name: '<img src=x onerror=alert(1)>', problem: null } }));
    expect(panel.q('#context img')).toBeNull();
    expect(panel.text('#context')).toContain('<img src=x onerror=alert(1)>');
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/x/<b>y</b>.3mf', text: 't', msg: null }));
    expect(panel.q('#path b')).toBeNull();
  });
});
