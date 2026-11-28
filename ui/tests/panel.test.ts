// Behaviour of the real Vue panel (App.vue and its components) against a fake `orca` bridge.
// Ported from the legacy Node harness (tests/test_panel_js.py, sections 0-13 of the old file) so that every
// state, message and protocol assertion of the imperative page still holds for the Vue one.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobMsg } from '../src/protocol';
import { clearToasts } from '@orca-plugins/ui-foundation/src/components/toast';
import { CTX, mountPanel, REF, type Panel } from './helpers';

let panel: Panel;
beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
  sessionStorage.clear(); // dismissed banners are remembered per session
});
afterEach(() => {
  panel?.unmount();
  clearToasts();
  vi.useRealTimers();
});

// A closed toast plays its leave transition: it is gone for the user once it has the leave class.
const TOAST = '.oui-toast:not(.oui-toast-leave-active)';
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
    for (const id of ['go', 'fill']) { // nothing to arrange yet: the buttons are off and point at the bar's "Reading…"
      expect(panel.q<HTMLButtonElement>('#' + id)!.disabled, id).toBe(true);
      expect(panel.q('#' + id)!.getAttribute('aria-describedby')).toBe(id + '-help context');
    }
    panel.organizer.run('arrange');
    expect(panel.takeRequests()).toEqual([]);
  });

  it('shows the plate in the buttons, the project in the context and the empty-project hint', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ context: 'x', ctx: { plate: 2, objects: 0, name: 'P', problem: null }, stale: '' }));
    expect(panel.text('#go')).toContain('Arrange plate 2');
    expect(panel.text('#fill')).toContain('Fill plate 2');
    expect(panel.text('#context .oui-sr')).toBe('P · plate 2 selected · 0 objects in the project'); // the whole sentence, for screen readers and the tooltip
    expect(panel.q('#context')!.getAttribute('title')).toBe('P · plate 2 selected · 0 objects in the project');
    expect(panel.text('#context [aria-hidden]')).toBe('P · Plate 2'); // what a tab's bar shows; the dock shows only a state chip
    expect(panel.text('#context .context-name')).toBe('P');
    // Nothing to arrange: a notice that stays, and the two buttons are off and point at it
    expect(panel.text('#empty')).toMatch(/No objects found/);
    expect(panel.q('#empty')!.matches('.oui-banner[data-tone=info]')).toBe(true);
    expect(panel.q('#empty .oui-banner-close')).toBeNull();
    expect(panel.status()).toBe('');
    for (const id of ['go', 'fill']) {
      expect(panel.q<HTMLButtonElement>('#' + id)!.disabled, id).toBe(true);
      expect(panel.q('#' + id)!.getAttribute('aria-describedby')).toBe(id + '-help empty');
    }
    panel.take();
    panel.organizer.run('fill');
    expect(panel.takeRequests()).toEqual([]);
    expect(panel.q('#context .oui-status')).toBeNull();
    expect(panel.q('#context [role=progressbar]')).toBeNull(); // loaded: the progress indicator is gone
    expect(panel.q('#context')!.getAttribute('aria-busy')).toBe('false');
    await panel.deliver(sync({ context: 'x', ctx: CTX, stale: '' }));
    expect(panel.q('#empty')).toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
  });

  it('host language wins, pt_BR and pt-BR both work, unsupported falls back to English (X04)', async () => {
    panel = await mountPanel();
    const ctx = (objects: number, name: string | null) => ({ plate: 2, objects, name, problem: null });
    await panel.deliver(sync({ lang: 'pt_BR', ctx: ctx(1, null) }));
    expect(document.documentElement.lang).toBe('pt-BR');
    expect(panel.text('#go')).toContain('Arrumar placa 2');
    expect(panel.text('#fill')).toContain('Preencher placa 2');
    expect(panel.text('#context .oui-sr')).toBe('Projeto sem título · placa 2 selecionada · 1 objeto no projeto');
    expect(panel.text('#context [aria-hidden]')).toBe('Projeto sem título · Placa 2');
    expect(panel.text('h1')).toBe('Organizer');
    expect(panel.status()).toBe(''); // nothing ran: no "Ready." line; the promise is part of what Arrange says it does
    expect(panel.text('#go-help')).toMatch(/O resultado é um arquivo novo; o seu projeto não é alterado\.$/);
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
    expect(panel.text('.context-problem strong')).toBe('Could not read the project. Open a project in OrcaSlicer, then try again.');
    // what the host said is never part of the sentence: it is behind a closed disclosure
    expect(panel.q<HTMLDetailsElement>('.context-problem details')!.open).toBe(false);
    expect(panel.text('.context-problem summary')).toBe('Technical details');
    expect(panel.text('.context-problem .tech-text')).toBe('no project');
    expect(panel.q('.context-problem')!.getAttribute('role')).toBe('alert'); // interrupts, as before
    expect(panel.text('#context .oui-status[data-state=warn]')).toBe('No project'); // the bar carries the page's one state
    expect(panel.text('#go')).toContain('Arrange current plate');
    for (const id of ['go', 'fill']) {
      expect(panel.q<HTMLButtonElement>('#' + id)!.disabled, id).toBe(true);
      expect(panel.q('#' + id)!.getAttribute('aria-describedby')).toBe(id + '-help context-problem');
    }
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

  it.each(['de', 'es', 'pt-BR', 'pl', 'en'])('%s: a point is a decimal, typed or pasted', async (language) => {
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

  it('en: a pasted comma decimal is read as one, not as a thousands sign', async () => {
    panel = await mountPanel({ language: 'en' });
    await panel.deliver(sync({ ctx: CTX }));
    await paste('1,5');
    expect(panel.organizer.state.spacing).toBe(1.5);
    await paste('1,234.5');
    expect(panel.organizer.state.spacing).toBe(1234.5);
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
    expect(panel.q('#busy-hint')!.classList.contains('oui-sr')).toBe(true); // the job block shows it; no paragraph under the form
    expect(panel.q('#job .oui-job')!.hasAttribute('data-hero')).toBe(false); // ui-foundation 5: a neutral block, no filled hero
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

  it('names what was arranged, offers Open and Copy path, frees the controls and moves focus to Open', async () => {
    panel = await finished();
    expect(panel.q('#path')).toBeNull(); // a generated name in an internal folder: Copy path has it, the card does not show it
    expect(panel.q('#result')).not.toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#cancel')).toBeNull();
    expect(panel.q('#job')).toBeNull();
    expect(panel.q('[role=progressbar]')).toBeNull();
    expect(panel.q('#status')!.classList.contains('oui-sr')).toBe(false);
    expect(panel.q('#status .oui-banner[data-tone=success]')).not.toBeNull();
    expect(panel.text('#file')).toBe('P'); // the project leads, then what was done; the generated file name is the tooltip
    expect(panel.q('#file')!.getAttribute('title')).toBe('P-arranged.3mf');
    expect(panel.text('#saved-what')).toBe('Plate 2 filled'); // no time in this file name
    expect(panel.q('#result')!.getAttribute('aria-labelledby')).toBe('file');
    expect(document.activeElement).toBe(panel.q('#open'));
    expect(panel.q('#copy')).not.toBeNull();
    expect(panel.status().split('\n')[0]).toBe('Placed 1 object.'); // the card names the project and the plate
    expect(panel.status()).toMatch(/Filled the plate with 1 extra copy \(block ×1\)\./);
    // Was it opened? One line under the buttons says the plugin already asked, so the button is the second try.
    expect(panel.text('#open-status')).toBe('OrcaSlicer was asked to open this copy.');
    expect(panel.q('#open-status')!.getAttribute('data-state')).toBe('pending'); // asked, not known to be open: never "ok"
    expect(panel.text('#open')).toBe('Open again');
    expect(panel.q('#open')!.classList.contains('p-button-secondary')).toBe(true);
    expect(panel.q('#open')!.getAttribute('aria-describedby')).toBe('open-status');
    panel.q('#open')!.click();
    expect(panel.takeRequests()).toEqual([{ action: 'open', job: 'j1' }]);
  });

  it('shows the time the copy was saved, from its file name', async () => {
    panel = await finished();
    const path = '/s/P-arranged-20261111-195500-1f3ffe.3mf';
    await panel.deliver({ job: 'j1', busy: false, path, text: 'x', msg: { ...doneMsg, action: 'arrange' } });
    expect(panel.text('#saved-what')).toMatch(/^Plate 2 arranged · 7:55\sPM$/);
    await panel.deliver({ job: 'j1', busy: false, path, text: 'x', lang: 'de', msg: { ...doneMsg, stem: null } });
    expect(panel.text('#file')).toBe('Unbenanntes Projekt');
    expect(panel.text('#saved-what')).toBe('Platte 2 gefüllt · 19:55');
  });

  it('makes Open the step to take when the copy was not opened, and shows the path only when it must be found by hand', async () => {
    panel = await finished();
    const result = (open: string) => panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, open, detail: 'boom' } as never });
    await result('later');
    expect(panel.text('#open')).toBe('Open in OrcaSlicer');
    expect(panel.q('#open')!.classList.contains('p-button-secondary')).toBe(false);
    expect(panel.text('#open-status')).toBe('Finished while the panel was closed. The copy is not open yet.');
    expect(panel.q('#path')).toBeNull();
    await result('dirty');
    expect(panel.text('#open-status')).toBe('OrcaSlicer has unsaved changes: save or discard them, then open the copy.');
    expect(panel.q('#open-status')!.getAttribute('data-state')).toBe('warn');
    expect(panel.text('#open')).toBe('Open in OrcaSlicer');
    expect(panel.q('#path')).toBeNull();
    for (const open of ['failed', 'none']) {
      await result(open);
      expect(panel.q('#open-status')!.getAttribute('data-state'), open).toBe('warn');
      expect(panel.text('#open-status'), open).toMatch(/Open the file by hand\.$/);
      expect(panel.text('#result'), open).not.toContain('boom'); // the launcher's error is in the OrcaSlicer log, not in a sentence
      expect(panel.text('#path'), open).toBe('/s/P-arranged.3mf');
    }
  });

  it('says what happened first, warns about parts that did not fit, and keeps the fine print with the saved copy', async () => {
    panel = await finished();
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, placed: 7, added: 12, by_object: { a: 8, b: 4 }, limit: true, tower: true, clearance: 12.5, right: true, unplaced: ['Part A', 'Part B'] } });
    const success = panel.q('#status .oui-banner[data-tone=success]')!;
    expect(success.querySelector('strong')!.textContent).toBe('Placed 7 objects.');
    expect([...success.querySelectorAll('.line')].map((l) => l.textContent)).toEqual(['Filled the plate with 12 extra copies (a ×8, b ×4).', 'Stopped at the 300-copy limit.']);
    expect(panel.text('#status .oui-banner[data-tone=warn] strong')).toBe('Did not fit (moved above the plate): Part A, Part B');
    // What changes how the plate must be checked is a short list under its own heading, not grey fine print
    expect(panel.text('#result .checks-title')).toBe('Check before printing');
    expect(panel.q('#result .checks ul')!.getAttribute('aria-labelledby')).toBe('checks-title');
    expect([...document.querySelectorAll('#result .checks li')].map((p) => p.textContent)).toEqual([
      'Kept an estimated prime tower area free, brim included. Its real size is only known after slicing: check it in Orca.',
      'Print sequence is By object: kept 12.5 mm XY clearance between parts. Orca still checks height and print order.',
    ]);
    expect([...document.querySelectorAll('#result .note')].map((p) => p.textContent)).toEqual(['Rotation was limited to 90° steps.']);
    expect(panel.text('#result')).not.toMatch(/as it was when you clicked/);
    expect(panel.q('#open')!.getAttribute('aria-describedby')).toBe('open-status'); // focus lands on Open: how opening went is its description
    expect(panel.q('#status .checks')).toBeNull(); // the live region announces the outcome, not the fine print
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
    expect(panel.status().split('\n')[0]).toBe('3 objetos posicionados.');
    expect(panel.text('#saved-what')).toBe('Placa 2 preenchida');
    expect(panel.text('#open-status')).toBe('Concluído com o painel fechado. A cópia ainda não foi aberta.');
    expect(panel.text('#open')).toBe('Abrir no OrcaSlicer');
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, placed: 1, added: 0, by_object: {}, open: 'asked' } });
    expect(panel.text('#open')).toBe('Abrir de novo');
    await panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, placed: 1, added: 0, by_object: {}, open: 'failed', detail: 'boom' } });
    expect(panel.text('#open-status')).toBe('Não foi possível abrir automaticamente. Abra o arquivo manualmente.');
  });

  it('copies the path, shows and selects it when the clipboard refuses, and shows it in place of Copy without the API', async () => {
    panel = await finished();
    let copied: string | null = null;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: (t: string) => ((copied = t), Promise.resolve()) } });
    panel.q('#copy')!.click();
    expect(copied).toBe('/s/P-arranged.3mf');
    await panel.flush();
    expect(panel.text('.oui-toast[data-toast-tone=success] strong')).toBe('Path copied.');
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.reject(new Error('denied')) } });
    panel.q('#copy')!.click();
    await panel.flush();
    await panel.flush();
    expect(panel.text('.oui-toast[data-toast-tone=warn] strong')).toMatch(/The text is selected: press Ctrl\+C/);
    expect(panel.q(`${TOAST}[data-toast-tone=success]`), 'a second copy replaces the first toast').toBeNull();
    expect(panel.text('#path')).toBe('/s/P-arranged.3mf'); // it could not be copied: now it is on screen
    const range = getSelection()!.getRangeAt(0); // and the fallback really selected the path element's text
    expect(range.startContainer).toBe(panel.q('#path'));
    expect(range.endOffset).toBe(panel.q('#path')!.childNodes.length);
    await panel.deliver(sync({ job: 'j2', busy: false, path: '/s/P-arranged-2.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#path'), 'the next copy starts with the path hidden again').toBeNull();
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    await panel.deliver(sync({ job: 'j2', busy: false, path: '/s/P-arranged-2.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#copy')).toBeNull(); // no clipboard API: no Copy button,
    expect(panel.text('#path')).toBe('/s/P-arranged-2.3mf'); // so the path itself is shown
  });

  it('shows the copy toast in the shared host, with a Close button, and leaves after 5 s', async () => {
    panel = await finished();
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.resolve() } });
    vi.useFakeTimers();
    panel.q('#copy')!.click();
    await panel.flush();
    const host = panel.q('.oui-toasts')!;
    expect(host.getAttribute('aria-label')).toBe('Notifications');
    expect(host.querySelectorAll('.oui-toast').length).toBe(1);
    expect(panel.q('.oui-toast')!.getAttribute('role')).toBe('status');
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('.oui-toast'), 'a poll does not remove it').not.toBeNull();
    await vi.advanceTimersByTimeAsync(5100);
    expect(panel.q(TOAST)).toBeNull();
    panel.q('#copy')!.click();
    await panel.flush();
    const close = panel.q<HTMLButtonElement>('.oui-toast .oui-toast-close')!;
    expect(close.getAttribute('aria-label')).toBe('Close');
    close.click();
    await panel.flush();
    expect(panel.q(TOAST)).toBeNull();
  });

  it('labels the toast host in the page language', async () => {
    panel = await mountPanel({ language: 'de-DE' });
    expect(panel.q('.oui-toasts')!.getAttribute('aria-label')).toBe('Benachrichtigungen');
  });

  it('shows the stale note, structured and translated, and hides it again', async () => {
    panel = await finished();
    const stale = 'The run below is for P, plate 2, not the current selection.';
    const keep = { job: 'j1', busy: false, text: 'done text', path: '/s/P-arranged.3mf', msg: doneMsg };
    await panel.deliver(sync({ ...keep, ctx: { plate: 1, objects: 1, name: 'Q', problem: null }, stale, stale_ref: { stem: 'P', plate: 2, action: 'arrange' } }));
    expect(panel.text('#stale strong')).toBe(stale);
    expect(panel.q('#stale')!.getAttribute('role')).toBe('status'); // informs, does not interrupt
    expect(panel.q('#result')).not.toBeNull();
    // One banner, not three: the outcome of the other selection's run is read in its card, and stays in the live region for screen readers
    expect(panel.q('#status')!.classList.contains('oui-sr')).toBe(true);
    expect([...document.querySelectorAll('#result .outcome > *')].map((e) => e.textContent!.trim())).toEqual(['Placed 1 object.', 'Filled the plate with 1 extra copy (block ×1).']);
    expect(panel.q('#result .outcome')!.getAttribute('aria-hidden')).toBe('true');
    await panel.deliver(sync({ ...keep, msg: { ...doneMsg, unplaced: ['Part A'] }, ctx: { plate: 1, objects: 1, name: 'Q', problem: null }, stale, stale_ref: { stem: 'P', plate: 2, action: 'arrange' } }));
    expect(panel.text('#result .outcome .oui-status[data-state=warn]')).toBe('Did not fit (moved above the plate): Part A');
    await panel.deliver(sync({ ...keep, lang: 'pt_BR', stale: 'x', stale_ref: { stem: null, plate: 2, action: 'arrange' } }));
    expect(panel.text('#stale strong')).toBe('A execução abaixo é de sem título, placa 2, não da seleção atual.');
    await panel.deliver(sync({ ...keep, lang: 'en_US', stale: '', stale_ref: null }));
    expect(panel.q('#stale')).toBeNull();
    expect(panel.q('#status')!.classList.contains('oui-sr')).toBe(false);
    expect(panel.q('#result .outcome')).toBeNull();
  });

  it('lets the stale note be dismissed until the run it names changes', async () => {
    panel = await finished();
    const keep = { job: 'j1', busy: false, text: 'done text', path: '/s/P-arranged.3mf', msg: doneMsg, ctx: { plate: 1, objects: 1, name: 'Q', problem: null } };
    const stale = (plate: number) => sync({ ...keep, stale: 'x', stale_ref: { stem: 'P', plate, action: 'arrange' } });
    await panel.deliver(stale(2));
    const close = panel.q<HTMLButtonElement>('#stale .oui-banner-close')!;
    expect(close.getAttribute('aria-label')).toBe('Close');
    close.click();
    await panel.flush();
    expect(panel.q('#stale')).toBeNull();
    await panel.deliver(stale(2)); // the next poll, same condition
    expect(panel.q('#stale')).toBeNull();
    panel.unmount(); // a reload in the same session
    panel = await finished();
    await panel.deliver(stale(2));
    expect(panel.q('#stale')).toBeNull();
    await panel.deliver(stale(3)); // another run: the note is news again
    expect(panel.text('#stale strong')).toBe('The run below is for P, plate 3, not the current selection.');
  });

  it('lets the result and error banners be dismissed per run', async () => {
    panel = await finished();
    const done = panel.q<HTMLButtonElement>('#status .oui-banner .oui-banner-close')!;
    done.click();
    await panel.flush();
    expect(panel.q('#status .oui-banner')).toBeNull();
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#status .oui-banner'), 'a poll of the same result keeps it closed').toBeNull();
    await panel.deliver(sync({ job: 'j2', busy: false, path: '/s/P-arranged-2.3mf', text: 'x', msg: doneMsg }));
    expect(panel.q('#status .oui-banner'), 'the next run shows its outcome').not.toBeNull();
    const failed = { ...REF, code: 'failed', action: 'fill', cause: { code: 'unexpected' }, detail: 'boom' } as JobMsg;
    await panel.deliver(sync({ job: 'j3', busy: false, error: true, text: 'x', msg: failed }));
    panel.q<HTMLButtonElement>('#status-error .oui-banner-close')!.click();
    await panel.flush();
    expect(panel.q('#status-error')).toBeNull();
    await panel.deliver(sync({ job: 'j3', busy: false, error: true, text: 'x', msg: failed }));
    expect(panel.q('#status-error')).toBeNull();
    await panel.deliver(sync({ job: 'j4', busy: false, error: true, text: 'x', msg: failed }));
    expect(panel.q('#status-error')).not.toBeNull();
  });

  it('lets the project problem be dismissed until the problem changes', async () => {
    panel = await mountPanel();
    const problem = (p: string) => sync({ msg: null, ctx: { plate: 0, objects: 0, name: '', problem: p } });
    await panel.deliver(problem('no plate'));
    panel.q<HTMLButtonElement>('.context-problem .oui-banner-close')!.click();
    await panel.flush();
    expect(panel.q('.context-problem')).toBeNull();
    await panel.deliver(problem('no plate'));
    expect(panel.q('.context-problem')).toBeNull();
    await panel.deliver(problem('locked'));
    expect(panel.text('.context-problem .tech-text')).toBe('locked');
  });

  it('shows a closed project problem again once it cleared and came back', async () => {
    panel = await mountPanel();
    const problem = (p: string | null) => sync({ msg: null, ctx: { plate: 0, objects: 0, name: '', problem: p } });
    await panel.deliver(problem('no plate'));
    panel.q<HTMLButtonElement>('.context-problem .oui-banner-close')!.click();
    await panel.flush();
    await panel.deliver(problem(null)); // solved: the dismissal is forgotten
    expect(panel.q('.context-problem')).toBeNull();
    await panel.deliver(problem('no plate')); // the same problem, later
    expect(panel.text('.context-problem .tech-text')).toBe('no plate');
  });

  it('shows a closed stale note again once it cleared and came back', async () => {
    panel = await finished();
    const keep = { job: 'j1', busy: false, text: 'done text', path: '/s/P-arranged.3mf', msg: doneMsg, ctx: { plate: 1, objects: 1, name: 'Q', problem: null } };
    const stale = (on: boolean) => sync({ ...keep, stale: on ? 'x' : '', stale_ref: on ? { stem: 'P', plate: 2, action: 'arrange' } : null });
    await panel.deliver(stale(true));
    panel.q<HTMLButtonElement>('#stale .oui-banner-close')!.click();
    await panel.flush();
    await panel.deliver(stale(false));
    await panel.deliver(stale(true));
    expect(panel.q('#stale')).not.toBeNull();
  });

  it('shows a closed warning again once it left the status and came back', async () => {
    panel = await finished();
    const result = (unplaced: string[]) => panel.deliver({ job: 'j1', busy: false, path: '/s/P-arranged.3mf', text: 'x', msg: { ...doneMsg, unplaced } });
    await result(['Part A']);
    panel.q<HTMLButtonElement>('#status .oui-banner[data-tone=warn] .oui-banner-close')!.click();
    await panel.flush();
    expect(panel.q('#status .oui-banner[data-tone=warn]')).toBeNull();
    await result(['Part A']);
    expect(panel.q('#status .oui-banner[data-tone=warn]'), 'still the same condition').toBeNull();
    await result([]);
    await result(['Part A']);
    expect(panel.q('#status .oui-banner[data-tone=warn]')).not.toBeNull();
  });

  it('shows a closed start failure again when the plugin sends it again, word for word', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    const startFailed = () => panel.deliver({ text: 'EN', error: true, busy: false, msg: { code: 'start_failed', cause: { code: 'unexpected' }, detail: 'd' } });
    await startFailed();
    panel.q<HTMLButtonElement>('#status-error .oui-banner-close')!.click();
    await panel.flush();
    expect(panel.q('#status-error')).toBeNull();
    await panel.deliver(sync({ ctx: CTX })); // a poll keeps it closed
    expect(panel.q('#status-error')).toBeNull();
    await startFailed(); // a second click that fails the same way
    expect(panel.q('#status-error')).not.toBeNull();
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
    expect(panel.status()).toMatch(/^P, plate 2: could not arrange\.\nSomething unexpected went wrong\.\nNext: Try again\..*\nzip is corrupt$/s);
    expect(panel.status()).not.toMatch(/error:/i); // no machine prefix
    // What the plugin raised is word for word behind a closed disclosure, with a Copy button: never a line of the banner
    expect([...document.querySelectorAll('#status-error .line')].map((l) => l.textContent).join(' ')).not.toContain('zip is corrupt');
    expect(panel.q<HTMLDetailsElement>('#status-error details')!.open).toBe(false);
    expect(panel.text('#status-error summary')).toBe('Technical details');
    expect(panel.text('#status-error .tech-text')).toBe('zip is corrupt');
    expect(panel.q('#status-error .tech-copy')).toBeNull(); // no clipboard here
    let copied = '';
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: (t: string) => ((copied = t), Promise.resolve()) } });
    await fail(panel, { code: 'unexpected' }, 'zip is corrupt', 'en_US');
    panel.q('#status-error .tech-copy')!.click();
    await panel.flush();
    expect(copied).toBe('zip is corrupt');
    expect(panel.text('.oui-toast[data-toast-tone=success] strong')).toBe('Details copied.');
    await fail(panel, { code: 'plate_gone', plate: 9, plates: 1 }, 'x', 'en_US');
    expect(panel.status()).toMatch(/does not exist: the project has 1 plate\./);
    expect(panel.q('#status-error details')).toBeNull(); // a known cause explains itself
  });

  it('translates failures to start and to save, and shows an older plugin text as it is', async () => {
    panel = await mountPanel();
    await panel.deliver({ text: 'EN', error: true, busy: false, lang: 'pt_BR', msg: { code: 'start_failed', cause: { code: 'bad_rotation' }, detail: 'd' } });
    expect(panel.status()).toBe('Não foi possível iniciar: A rotação deve ser automática, só passos de 90° ou manter a orientação.\nPróximo passo: Escolha uma das três opções.');
    await panel.deliver(sync({ lang: 'en_US', job: 'j2', error: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'save_failed', detail: 'disk full' } }));
    expect(panel.status()).toBe('P, plate 2: could not save the arranged copy.\nNothing was opened; earlier copies are unchanged.\ndisk full');
    expect(panel.text('#status-error strong')).toBe('P, plate 2: could not save the arranged copy.'); // the system's error is not part of the sentence
    expect(panel.text('#status-error .tech-text')).toBe('disk full');
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

  it('cancels a run pressed before the plugin named the job, once the start reply gives its id', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX }));
    panel.take();
    panel.q('#go')!.click();
    await panel.flush();
    panel.q('#cancel')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: panel.organizer.state.spacing, rotation: 'auto' }]);
    await panel.deliver({ job: 'j1', start: true, busy: true, text: 'r', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'reading' } });
    expect(panel.takeRequests()).toEqual([{ action: 'cancel', job: 'j1' }]);
    expect(panel.text('#cancel')).toContain('Cancelling…');
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
    expect(panel.text('#offline strong')).toBe('Organizer is not answering.');
    expect(panel.text('#offline-text')).toBe('Check that it is enabled under Plugins in OrcaSlicer. Retrying automatically.');
    // Nobody is listening: Arrange and Fill are off, point at the banner, and nothing is sent
    for (const id of ['go', 'fill']) {
      expect(panel.q<HTMLButtonElement>('#' + id)!.disabled, id).toBe(true);
      expect(panel.q('#' + id)!.getAttribute('aria-describedby')).toBe(id + '-help offline');
    }
    expect(panel.text('.pane-form')).not.toMatch(/Ready/);
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
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    await panel.deliver(sync({ lang: 'en_US', job: 'j4', busy: true, text: 'running', msg: null }));
    panel.take();
    await vi.advanceTimersByTimeAsync(3000);
    panel.take();
    await vi.advanceTimersByTimeAsync(1000); // busy: every second
    expect(panel.take()).toEqual([{ action: 'poll' }]);
  });

  it('lets the offline banner be dismissed while the plugin stays silent, and shows it for the next silence', async () => {
    vi.useFakeTimers();
    panel = await mountPanel();
    await panel.deliver(sync({ msg: null, ctx: CTX }));
    await vi.advanceTimersByTimeAsync(14000);
    panel.q<HTMLButtonElement>('#offline .oui-banner-close')!.click();
    await panel.flush();
    expect(panel.q('#offline')).toBeNull();
    await vi.advanceTimersByTimeAsync(6000); // more polls, still no answer
    expect(panel.q('#offline')).toBeNull();
    expect(panel.text('#context .oui-status[data-state=offline]')).toBe('No answer'); // the bar still says so
    await panel.deliver(sync({ msg: null, ctx: CTX })); // back
    expect(sessionStorage.length, 'the dismissal is forgotten once the plugin answers').toBe(0);
    await vi.advanceTimersByTimeAsync(14000); // silent again: a new episode
    expect(panel.q('#offline')).not.toBeNull();
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
    expect(panel.text('#go-help')).toBe("Fits the plate's objects and adds nothing. The result is a new file; your project is not changed.");
    expect(panel.q('#go')!.getAttribute('title')).toBe(panel.text('#go-help'));
    expect(panel.text('#fill-help')).toMatch(/up to 300/);
    expect(panel.text('#foot')).toBe('Brims, supports and skirts are not counted. Check the plate before printing.');
    expect(panel.q('#foot svg')).not.toBeNull(); // a caution with its mark, not grey fine print
    expect(panel.status()).toBe(''); // nothing ran: the page does not say "Ready."
    expect(panel.text('.idle .oui-empty-text')).toMatch(/The result is a new file; your project is not changed\.$/);
    expect(panel.q('#req')).toBeNull(); // nothing is missing: no Requirements section
    for (const id of ['adv', 'help']) expect(panel.q(`#${id} .sum > svg`), id).not.toBeNull(); // both disclosures turn the same chevron
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
    // The explanation is behind the (i) beside the legend, not a paragraph under the radios.
    expect(panel.q('#rotation-help')!.closest('.oui-hint-bubble[role=tooltip]')).not.toBeNull();
    expect(panel.q('fieldset .oui-hint-btn')!.getAttribute('aria-label')).toBe('Rotation');
    expect(panel.text('#rotation-help')).toMatch(/^Any angle tries every rotation/);
    expect(panel.text('legend')).toBe('Rotation');
    expect(panel.q<HTMLInputElement>('#rotation-auto')!.checked).toBe(true);
    expect([...document.querySelectorAll('.choice label')].map((l) => l.textContent)).toEqual(['Any angle', '90° steps only', 'Do not rotate']);
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
    expect(document.querySelector('label[for=rotation-right]')!.textContent).toBe('90° steps only');
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
    expect(panel.q('#copies-help')!.closest('.oui-hint-bubble[role=tooltip]')).not.toBeNull();
    expect(panel.q<HTMLInputElement>('#copies')!.value).toBe('');
    expect(panel.q('#copies')!.getAttribute('placeholder')).toBe('Until full'); // empty has a meaning: the field says it
    expect(panel.q('#copies')!.closest('.p-inputnumber')!.querySelector('.p-inputnumber-button-group, .p-inputnumber-button')).not.toBeNull();
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
    await panel.deliver(sync({ job: 'j1', busy: false, path: '/x/<b>y.3mf', text: 't', msg: null }));
    expect(panel.text('#file')).toBe('<b>y.3mf');
    expect(panel.q('#result b')).toBeNull();
  });
});

describe('requirements', () => {
  const found = [
    { id: 'numpy', name: 'numpy', status: 'ok', version: '2.5.5', detail: null },
    { id: 'pillow', name: 'Pillow', status: 'ok', version: '13.0.0', detail: null },
  ] as const;
  const noPillow = [found[0], { id: 'pillow', name: 'Pillow', status: 'missing', version: null, detail: "ImportError: No module named 'PIL'" }] as const;

  it('says nothing about requirements while everything is found', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, requirements: [...found] }));
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#req-hint')).toBeNull();
    expect(panel.q('#req')).toBeNull();
    expect(panel.text('.pane-form')).not.toMatch(/numpy|Pillow|2\.5\.5/);
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
    expect(panel.q('#req-hint')!.matches('.oui-banner[data-tone=warn]')).toBe(true); // the reason is the page's banner, said once
    expect(panel.q('.pane-form #req-hint')).toBeNull();
    expect(panel.text('#req')).toBe('Requirements');
    expect(panel.text('[data-req=pillow] .oui-status[data-state=warn]')).toBe('Not found');
    expect(panel.text('[data-req=pillow]')).toContain('Needed to measure part outlines.');
    expect(panel.q('[data-req=numpy]')).toBeNull(); // found: not listed
    expect(panel.q('.reqs a')).toBeNull(); // one instruction (reinstall the plugin), no link to a package's own install guide
    expect(panel.text('.reqs')).toContain('Reinstall the plugin from Orca Cloud');
    // the import error is word for word behind a closed disclosure, not a line of the row
    expect(panel.q<HTMLDetailsElement>('[data-req=pillow] details')!.open).toBe(false);
    expect(panel.text('[data-req=pillow] .tech-text')).toBe("ImportError: No module named 'PIL'");
    panel.organizer.run('arrange');
    expect(panel.take()).toEqual([]); // nothing is sent while a requirement is missing

    // one Check again for the section
    const visible = [...document.querySelectorAll<HTMLElement>('.reqs button')].filter((b) => !b.classList.contains('oui-req-check'));
    expect(visible.map((b) => b.id)).toEqual(['recheck']);
    panel.q<HTMLButtonElement>('#recheck')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'recheck' }]);
    expect(panel.text('#recheck')).toBe('Checking…');
    expect(panel.q<HTMLButtonElement>('#recheck')!.disabled).toBe(true);
    // the plugin answers "unknown" while it retries: what was missing stays, and so does the reason
    await panel.deliver(sync({ ctx: CTX, requirements: found.map((r) => ({ ...r, status: 'unknown', version: null })) }));
    expect(panel.text('#recheck')).toBe('Checking…');
    expect(panel.q('[data-req=pillow]')).not.toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(true);
    await panel.deliver(sync({ ctx: CTX, requirements: [...noPillow] })); // still missing
    expect(panel.text('#recheck')).toBe('Check again');
    expect(panel.q<HTMLButtonElement>('#recheck')!.disabled).toBe(false);
    await panel.deliver(sync({ ctx: CTX, requirements: [...found] }));
    expect(panel.q('#req')).toBeNull();
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#req-hint')).toBeNull();
  });

  it('leaves Arrange on and says nothing while the plugin is still loading the packages', async () => {
    panel = await mountPanel();
    await panel.deliver(sync({ ctx: CTX, requirements: found.map((r) => ({ ...r, status: 'unknown', version: null })) }));
    expect(panel.q<HTMLButtonElement>('#go')!.disabled).toBe(false);
    expect(panel.q('#req')).toBeNull();
  });

  it('explains the plugin refusing a run for a missing requirement, in the page language', async () => {
    panel = await mountPanel({ language: 'de-DE' });
    await panel.deliver(sync({ ctx: CTX, text: 'EN', error: true, msg: { code: 'start_failed', cause: { code: 'missing_requirements', names: 'numpy' }, detail: 'd' } }));
    expect(panel.status()).toContain('Anordnen und Füllen benötigen numpy');
  });
});
