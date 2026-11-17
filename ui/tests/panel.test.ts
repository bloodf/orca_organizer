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
const bar = () => panel.q('#job [role=progressbar]')!;
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
    expect(panel.text('#context')).toBe('Reading…');
    expect(panel.q('#context .oui-status[data-state=connecting]')).not.toBeNull(); // loading state is visible, not only text
    expect(panel.q('#context')!.getAttribute('aria-busy')).toBe('true');
    expect(panel.q('#context')!.getAttribute('title')).toBe('Loading the project');
    const loadingBar = panel.q('#context [role=progressbar]')!; // an accessible progress indicator, as before
    expect(loadingBar.getAttribute('aria-label')).toBe('Loading the project');
    expect(loadingBar.getAttribute('aria-valuenow')).toBeNull(); // indeterminate
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
    expect(panel.text('#context .oui-sr')).toBe('P · plate 2 selected · 0 objects in the project'); // the whole sentence, for screen readers and the tooltip
    expect(panel.q('#context')!.getAttribute('title')).toBe('P · plate 2 selected · 0 objects in the project');
    expect(panel.text('#context [aria-hidden]')).toBe('P · Plate 2 · 0 objects'); // what the bar shows; the dock drops the name
    expect(panel.status()).toMatch(/No objects found/);
    expect(panel.q('#status .oui-banner[data-tone=neutral]')).not.toBeNull();
    expect(panel.q('#context .oui-status')).toBeNull();
    expect(panel.q('#context [role=progressbar]')).toBeNull(); // loaded: the progress indicator is gone
    expect(panel.q('#context')!.getAttribute('aria-busy')).toBe('false');
  });

  it('host language wins, pt_BR and pt-BR both work, unsupported falls back to English (X04)', async () => {
    panel = await mountPanel();
    const ctx = (objects: number, name: string | null) => ({ plate: 2, objects, name, problem: null });
    await panel.deliver(sync({ lang: 'pt_BR', ctx: ctx(1, null) }));
    expect(document.documentElement.lang).toBe('pt-BR');
    expect(panel.text('#go')).toContain('Arrumar placa 2');
    expect(panel.text('#fill')).toContain('Preencher placa 2');
    expect(panel.text('#context .oui-sr')).toBe('Projeto sem título · placa 2 selecionada · 1 objeto no projeto');
    expect(panel.text('#context [aria-hidden]')).toBe('Projeto sem título · Placa 2 · 1 objeto');
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
    expect(panel.text('#context')).toBe('Lendo…');
    await panel.deliver(sync({ lang: 'en_US', ctx: CTX }));
    expect(document.documentElement.lang).toBe('en');
  });

  it('names an unreadable project and how to recover (recoverable error)', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: { plate: null, objects: null, name: null, problem: 'no project' } }));
    expect(panel.text('.context-problem')).toMatch(/Could not read the project: no project\. Open a project/);
    expect(panel.q('.context-problem')!.getAttribute('role')).toBe('alert'); // interrupts, as before
    expect(panel.text('#context .oui-status[data-state=warn]')).toBe('No project'); // the bar carries the page's one state
    await panel.deliver(sync({ ctx: CTX }));
    expect(panel.q('.context-problem')).toBeNull();
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

  // Key by key, as a keyboard does: PrimeVue reads keypress, not input.
  async function keys(text: string) {
    const input = panel.q<HTMLInputElement>('#spacing')!;
    input.focus();
    input.value = '';
    for (const key of text) {
      const end = input.value.replace(/ mm$/, '').length; // the caret stays before the unit
      input.setSelectionRange(end, end);
      const event = new KeyboardEvent('keypress', { key, bubbles: true, cancelable: true });
      if (input.dispatchEvent(event)) input.setRangeText(key, input.selectionStart!, input.selectionEnd!, 'end');
    }
    input.dispatchEvent(new Event('blur')); // PrimeVue commits the value on blur
    await panel.flush();
  }
  async function paste(text: string) {
    const input = panel.q<HTMLInputElement>('#spacing')!;
    input.focus();
    input.select();
    const event = new Event('paste', { bubbles: true, cancelable: true });
    Object.defineProperty(event, 'clipboardData', { value: { getData: () => text } });
    input.dispatchEvent(event);
    await panel.flush();
  }

  it.each(['de', 'es', 'pt-BR', 'en'])('%s: a point is a decimal, typed or pasted', async (language) => {
    panel = await mountPanel({ language });
    await panel.deliver(sync({ ctx: CTX }));
    await keys('2.5');
    expect(panel.organizer.state.spacing).toBe(2.5);
    await paste('2.5');
    expect(panel.organizer.state.spacing).toBe(2.5);
    await paste('3.25 mm');
    expect(panel.organizer.state.spacing).toBe(3.25);
    expect(panel.q<HTMLInputElement>('#spacing')!.value).toBe(language === 'en' ? '3.25 mm' : '3,25 mm'); // shown as the language writes it
  });

  it('de: the comma is still the decimal, typed or pasted', async () => {
    panel = await mountPanel({ language: 'de' });
    await panel.deliver(sync({ ctx: CTX }));
    await keys('2,5');
    expect(panel.organizer.state.spacing).toBe(2.5);
    await paste('1,5');
    expect(panel.organizer.state.spacing).toBe(1.5);
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
    expect(panel.q('#job [role=progressbar]')).not.toBeNull();
    expect(panel.text('#job .oui-status')).toBe('Starting…');
    expect(panel.q('#status')!.classList.contains('oui-sr')).toBe(true); // the job card shows it; the sentence stays for screen readers
    expect(document.activeElement).toBe(panel.q('#cancel'));
    expect(panel.q('#status')!.getAttribute('aria-busy')).toBe('true');
    expect(panel.isError()).toBe(false);
    expect(panel.status()).toBe('Starting…');
    expect(panel.text('#busy-hint')).toMatch(/run is in progress/); // a disabled control says why, once
    expect(panel.q('#go')!.getAttribute('aria-describedby')).toBe('go-help busy-hint');
    expect(panel.q('#fill')!.getAttribute('aria-describedby')).toBe('fill-help busy-hint');
  });

  it('draws determinate progress from structured counts and keeps the live text for phases only', async () => {
    panel = await started();
    panel.q<HTMLButtonElement>('#fill')!.focus();
    panel.q('#fill')!.click();
    panel.take();
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'reading', msg: { stem: 'P', plate: 2, action: 'fill', code: 'reading' }, ctx: CTX });
    expect(panel.status()).toBe('P, plate 2: Reading project…');
    await prog(panel, 'read', 0, null);
    expect(panel.text('.oui-job-figure')).toBe('—'); // no count yet: a dash, not 0
    expect(bar().getAttribute('aria-valuenow')).toBeNull(); // read: indeterminate
    await prog(panel, 'place', 0, 14, { subject: 'block' });
    expect(panel.status()).toBe('P, plate 2: Placing parts');
    expect(bar().getAttribute('aria-valuenow')).toBe('0');
    expect(bar().getAttribute('aria-label')).toBe('Progress');
    expect(panel.text('#job .oui-status')).toBe('Placing parts');
    expect(panel.text('.oui-job-name')).toBe('P');
    expect(panel.text('.oui-job-figure')).toBe('1 of 14');
    expect(panel.text('.oui-job-detail')).toBe('block'); // the bar already says Plate 2
    expect(panel.q('.oui-job-detail')!.getAttribute('title')).toBe('block'); // cut to one line: the whole text is its tooltip
    expect(bar().getAttribute('aria-valuetext')).toBe('1 of 14 · block'); // a screen reader hears the count, not only a percentage
    const live = panel.q('#status')!;
    const mutations: MutationRecord[] = [];
    const observer = new MutationObserver((records) => mutations.push(...records));
    observer.observe(live, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['class'] });
    for (let i = 1; i < 14; i++) await prog(panel, 'place', i, 14, { subject: 'block ' + i });
    await Promise.resolve();
    observer.disconnect();
    expect(bar().getAttribute('aria-valuenow')).toBe('93'); // 13 of 14
    expect(panel.text('.oui-job-figure')).toBe('14 of 14');
    expect(panel.text('.oui-job-detail')).toBe('block 13');
    expect(panel.q('.oui-job-detail')!.getAttribute('title')).toBe('block 13');
    expect(bar().getAttribute('aria-valuetext')).toBe('14 of 14 · block 13');
    expect(mutations, 'the live region is not rewritten on every poll').toHaveLength(0);
    await prog(panel, 'fill', 1, null, { count: 1 });
    expect(bar().getAttribute('aria-valuenow')).toBeNull();
    expect(panel.text('.oui-job-detail')).toBe('1 copy added');
    await prog(panel, 'fill', 5, null, { count: 5 });
    expect(panel.text('.oui-job-figure')).toBe('+5');
    expect(panel.text('.oui-job-detail')).toBe('5 copies added');
    expect(bar().getAttribute('aria-valuetext')).toBe('5 copies added');
    expect(panel.text('#job .oui-status')).toBe('Adding copies');
    expect(panel.status()).toBe('P, plate 2: Adding copies');
    expect(document.activeElement).toBe(panel.q('#cancel')); // polls do not move focus
    await panel.deliver(sync({ lang: 'pt_BR', job: 'j1', busy: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'fill', code: 'progress', phase: 'place', i: 6, n: 14, count: null, subject: 'b' } }));
    expect(panel.status()).toBe('P, placa 2: Posicionando as peças');
    expect(panel.text('.oui-job-figure')).toBe('7 de 14');
    expect(panel.text('.oui-job-detail')).toBe('b');
    // a run for another plate than the one selected names its plate
    await panel.deliver(sync({ job: 'j1', busy: true, text: 'EN', ctx: { ...CTX, plate: 1 }, msg: { stem: 'P', plate: 2, action: 'fill', code: 'progress', phase: 'place', i: 6, n: 14, count: null, subject: 'b' } }));
    expect(panel.text('.oui-job-detail')).toBe('Placa 2 · b');
    expect(panel.q('.oui-job-detail')!.getAttribute('title')).toBe('Placa 2 · b');
    expect(panel.text('#job .oui-status')).toBe('Posicionando as peças');
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
      expect(bar().getAttribute('aria-valuenow') !== null, phase).toBe(determinate);
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
    expect(panel.q('#job')).toBeNull();
    expect(panel.q('[role=progressbar]')).toBeNull();
    expect(panel.q('#status')!.classList.contains('oui-sr')).toBe(false);
    expect(panel.q('#status .oui-banner[data-tone=success]')).not.toBeNull();
    expect(panel.text('#file')).toBe('P-arranged.3mf'); // the file name leads; the whole path is its tooltip and stays selectable below
    expect(panel.q('#file')!.getAttribute('title')).toBe('/s/P-arranged.3mf');
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

  it('says what happened first, warns about parts that did not fit, and keeps the fine print with the saved copy', async () => {
    panel = await finished();
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, placed: 7, added: 12, by_object: { a: 8, b: 4 }, limit: true, tower: true, clearance: 12.5, right: true, unplaced: ['Part A', 'Part B'] } });
    const success = panel.q('#status .oui-banner[data-tone=success]')!;
    expect(success.querySelector('strong')!.textContent).toBe('P, plate 2: placed 7 objects.');
    expect([...success.querySelectorAll('.line')].map((l) => l.textContent)).toEqual(['Filled the plate with 12 extra copies (a ×8, b ×4).', 'Stopped at the 300-copy limit.']);
    expect(panel.text('#status .oui-banner[data-tone=warn] strong')).toBe('Did not fit (moved above the plate): Part A, Part B');
    expect([...document.querySelectorAll('#result #notes p')].map((p) => p.textContent)).toEqual([
      'Kept an estimated prime tower area free, brim included. Its real size is only known after slicing: check it in Orca.',
      'Print sequence is By object: kept 12.5 mm XY clearance between parts. Orca still checks height and print order.',
      'Rotation was limited to 90° steps.',
      'Arranged from the project as it was when you clicked.',
      'Asked OrcaSlicer to open the copy. If nothing appears, use Open this copy in OrcaSlicer.',
    ]);
    expect(panel.q('#open')!.getAttribute('aria-describedby')).toBe('notes'); // focus lands on Open: the fine print is its description
    expect(panel.q('#status #notes')).toBeNull(); // the live region announces the outcome, not the fine print
    // the banners inside the live region do not announce a second time
    for (const banner of document.querySelectorAll('#status .oui-banner')) expect(banner.getAttribute('role')).toBe('none');
  });

  it('moves focus to Cancel when a run started elsewhere hides the control that had it', async () => {
    for (const id of ['#spacing', '#adv', '#copies']) {
      panel = await mountPanel();
      await panel.deliver(sync({ ctx: CTX }));
      panel.q<HTMLDetailsElement>('#adv')!.parentElement!.setAttribute('open', '');
      panel.q<HTMLElement>(id)!.focus();
      expect(document.activeElement, id).toBe(panel.q(id));
      // the run was started from the other surface (the tab or the Script's panel): the plugin says so
      await panel.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'reading' } });
      expect(panel.q('.grid')!.getAttribute('data-view')).toBe('running');
      expect(document.activeElement, id).toBe(panel.q('#cancel'));
      if (id !== '#copies') panel.unmount(); // afterEach unmounts the last
    }
  });

  it('shows one thing at a time in the dock: form, running job or result, with a way back to the form', async () => {
    panel = await mountPanel();
    const view = () => panel.q('.grid')!.getAttribute('data-view');
    await panel.deliver(sync({ ctx: CTX }));
    expect(view()).toBe('form');
    panel.q('#go')!.click();
    await panel.flush();
    expect(view()).toBe('running');
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'reading' } });
    await panel.deliver({ job: 'j1', text: 'd', busy: false, done: true, path: '/s/P-arranged.3mf', msg: doneMsg });
    expect(view()).toBe('result');
    expect(panel.q('.oui-appbar-actions #again')!.getAttribute('aria-label')).toBe('New arrangement'); // in the bar: on screen however long the result is
    panel.q('#again')!.click();
    await panel.flush();
    expect(view()).toBe('form');
    expect(document.activeElement).toBe(panel.q('#spacing')); // the button it left is gone from the dock: focus goes to the form
    expect(panel.q('#result')).not.toBeNull(); // the result stays on the page, under the form
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'd', msg: doneMsg }));
    expect(view(), 'a poll of the same result does not take the form away').toBe('form');
    panel.q('#fill')!.click();
    await panel.flush();
    expect(view()).toBe('running');
    await panel.deliver({ job: 'j2', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'fill', code: 'reading' } });
    await panel.deliver({ job: 'j2', text: 'd', busy: false, done: true, path: '/s/P-arranged-2.3mf', msg: doneMsg });
    expect(view(), 'a new result is shown').toBe('result');
  });

  it('says what to expect before the first run, with the plate illustration (tab view)', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    expect(panel.text('.idle .oui-panel-title')).toBe('Result');
    expect(panel.text('.idle .oui-empty-text')).toMatch(/^Nothing arranged yet/);
    expect(panel.q<HTMLImageElement>('.idle img')!.getAttribute('alt')).toBe(''); // decorative
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.q('.idle')).toBeNull();
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
    expect(panel.text('#stale strong')).toBe(stale);
    expect(panel.q('#stale')!.getAttribute('role')).toBe('status'); // informs, does not interrupt
    expect(panel.q('#result')).not.toBeNull();
    await panel.deliver(sync({ ...keep, lang: 'pt_BR', stale: 'x', stale_ref: { stem: null, plate: 2, action: 'arrange' } }));
    expect(panel.text('#stale strong')).toBe('A execução abaixo é de sem título, placa 2, não da seleção atual.');
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
    expect(panel.status()).toBe('Não foi possível iniciar: A rotação deve ser automática, só passos de 90° ou manter a orientação.\nPróximo passo: Escolha uma das três opções.');
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
    expect(panel.q('#status .oui-banner[data-tone=neutral]')).not.toBeNull(); // not a success, not an error
    expect(panel.q('.grid')!.getAttribute('data-view')).toBe('form');
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
    expect(panel.text('#context .oui-sr')).toBe('P · plate 1 selected · 2 objects in the project');
    panel.setHidden(false);
    await vi.advanceTimersByTimeAsync(11000);
    expect(panel.q('#offline')).not.toBeNull();
    expect(panel.text('#offline strong')).toBe('No answer from the plugin.');
    expect(panel.text('#offline-text')).toMatch(/still enabled in the Plugins window/);
    expect(panel.text('#offline')).not.toMatch(/reload/i); // the host offers no reload API
    expect(panel.q('#offline')!.getAttribute('role')).toBe('alert');
    expect(panel.text('#context .oui-status[data-state=offline]')).toBe('No answer'); // the bar says the same, never the opposite
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
    expect(panel.q('details > summary#adv')).not.toBeNull();
    expect(panel.q('#spacing')!.getAttribute('aria-describedby')).toBe('spacing-help');
    expect(document.querySelector('label[for=spacing]')!.textContent).toBe('Spacing between parts');
    expect(panel.q<HTMLInputElement>('#spacing')!.value).toBe('2 mm'); // the unit is in the field
    expect(panel.q('#go')!.getAttribute('aria-describedby')).toBe('go-help');
    expect(panel.q('#fill')!.getAttribute('aria-describedby')).toBe('fill-help');
    // What each control does is said once, under "How it works"; the controls point at it and show it as a tooltip.
    for (const id of ['spacing-help', 'go-help', 'fill-help', 'subtitle']) expect(panel.q(`details #${id}`), id).not.toBeNull();
    // the safety note is never folded away: not inside a disclosure, in the form the dock shows
    expect(panel.q('#foot')!.closest('details'), 'foot is not in a disclosure').toBeNull();
    expect(panel.q('#foot')!.closest('.pane-form')).not.toBeNull();
    expect(panel.text('#go-help')).toBe("Fits the plate's objects. Adds nothing.");
    expect(panel.q('#go')!.getAttribute('title')).toBe(panel.text('#go-help'));
    expect(panel.text('#fill-help')).toMatch(/up to 300/);
    expect(panel.text('#foot')).toMatch(/Check the plate in Orca before printing/);
    expect(panel.status()).toBe('Ready. The result is saved as a new file; your project is not changed.');
    expect(panel.q('#go svg')).not.toBeNull(); // the actions carry their icons
    expect(panel.q('.oui-appbar-mark svg')).not.toBeNull();
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

  it('offers 90° steps as a third rotation option', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.q('#adv')!.click();
    await panel.flush();
    expect(document.querySelector('label[for=rotation-right]')!.textContent).toBe('Only 90° steps');
    panel.q('#rotation-right')!.click();
    await panel.flush();
    panel.take();
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: 2, rotation: 'right' }]);
    await panel.deliver(sync({ job: 'j1', busy: false, text: 'EN', path: '/s/P.3mf', msg: { ...REF, code: 'done', placed: 2, added: 0, right: true, keep: false } }));
    expect(panel.status()).toContain('Rotation was limited to 90° steps.');
  });

  it('sends a copy limit with Fill only, and none when the field is empty', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.q('#adv')!.click();
    await panel.flush();
    expect(document.querySelector('label[for=copies]')!.textContent).toBe('Most copies to add when filling');
    expect(panel.q('#copies')!.getAttribute('aria-describedby')).toBe('copies-help');
    expect(panel.q<HTMLInputElement>('#copies')!.value).toBe('');
    panel.take();
    panel.q('#fill')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'fill', spacing: 2, rotation: 'auto' }]);
    await panel.deliver(sync({ busy: false }));
    panel.organizer.setCopies(12);
    await panel.flush();
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: 2, rotation: 'auto' }]);
    await panel.deliver(sync({ busy: false }));
    panel.q('#fill')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'fill', spacing: 2, rotation: 'auto', copies: 12 }]);
    expect(panel.q<HTMLInputElement>('#copies')!.disabled).toBe(true);
  });

  it('opens with the remembered settings once, and never over what the user changed', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, prefs: { spacing: 3.5, rotation: 'right', copies: 12 } }));
    expect(panel.organizer.state.spacing).toBe(3.5);
    expect(panel.organizer.state.rotation).toBe('right');
    expect(panel.organizer.state.copies).toBe(12);
    await panel.typeSpacing('1');
    await panel.deliver(sync({ ctx: CTX, prefs: { spacing: 7, rotation: 'keep', copies: 3 } }));
    expect(panel.organizer.state.spacing).toBe(1);
    expect(panel.organizer.state.rotation).toBe('right');
    panel.unmount();
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, prefs: { spacing: -2, rotation: 'sideways', copies: 2.5 } }));
    expect(panel.organizer.state.spacing).toBe(2);
    expect(panel.organizer.state.rotation).toBe('auto');
    expect(panel.organizer.state.copies).toBe(null);
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

describe('requirements', () => {
  const found = [
    { id: 'numpy', name: 'numpy', status: 'ok', version: '2.5.5', detail: null },
    { id: 'pillow', name: 'Pillow', status: 'ok', version: '13.0.0', detail: null },
  ] as const;
  const noPillow = [found[0], { id: 'pillow', name: 'Pillow', status: 'missing', version: null, detail: "ImportError: No module named 'PIL'" }] as const;

  it('lists what Arrange and Fill need, with versions, collapsed while everything is found', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, requirements: [...found] }));
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#req-hint')).toBeNull();
    expect(panel.q<HTMLDetailsElement>('details:has(#req)')!.open).toBe(false);
    expect(panel.text('#req')).toContain('all found');
    expect(panel.text('[data-req=numpy] .oui-status[data-state=ok]')).toBe('Found, version 2.5.5');
    expect(panel.q('[data-req=pillow] a')!.getAttribute('href')).toBe('https://pillow.readthedocs.io/en/stable/installation/');
  });

  it('turns Arrange and Fill off with a one-line reason and says how to fix it', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, requirements: [...noPillow] }));
    panel.take();
    for (const id of ['#go', '#fill']) {
      const button = panel.q<HTMLButtonElement>(id)!;
      expect(button.disabled).toBe(true);
      expect(button.getAttribute('aria-describedby')).toContain('req-hint');
    }
    expect(panel.text('#req-hint')).toBe('Arrange and Fill are off: Pillow could not be loaded. See Requirements below.');
    expect(panel.q<HTMLDetailsElement>('details:has(#req)')!.open).toBe(true);
    expect(panel.text('[data-req=pillow] .oui-status[data-state=error]')).toBe('Not found');
    expect(panel.text('[data-req=pillow]')).toContain('Reinstall the plugin from Orca Cloud');
    expect(panel.text('[data-req=pillow]')).toContain("No module named 'PIL'");
    panel.organizer.run('arrange');
    expect(panel.take()).toEqual([]); // nothing is sent while a requirement is missing

    panel.q<HTMLButtonElement>('#recheck')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'recheck' }]);
    expect(panel.text('#recheck')).toBe('Checking…');
    await panel.deliver(sync({ ctx: CTX, requirements: [...found] }));
    expect(panel.text('#recheck')).toBe('Check again');
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#req-hint')).toBeNull();
  });

  it('says it is still checking while the plugin loads the packages, and leaves Arrange on', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, requirements: found.map((r) => ({ ...r, status: 'unknown', version: null })) }));
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.text('#req')).toContain('Checking…');
    expect(panel.text('[data-req=numpy] .oui-status[data-state=pending]')).toBe('Checking…');
    await panel.deliver(sync({ ctx: CTX, requirements: [...found] }));
    expect(panel.text('#req')).toContain('all found');
  });

  it('explains the plugin refusing a run for a missing requirement, in the page language', async () => {
    panel = await mountPanel({ language: 'de-DE' });
    await panel.deliver(sync({ ctx: CTX, text: 'EN', error: true, msg: { code: 'start_failed', cause: { code: 'missing_requirements', names: 'numpy' }, detail: 'd' } }));
    expect(panel.status()).toContain('Anordnen und Füllen benötigen numpy');
  });
});
