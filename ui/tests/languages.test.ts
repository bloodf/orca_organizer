// The real panel in every language. Each state the page can be in is rendered once in English and once in the other
// language, and nothing a person can read or a screen reader can say (text, tooltips, aria labels) may be the same
// English sentence. Then the wording of the languages added since 0.14.0 (es, de, zh-CN, pl) is checked on the page itself.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Lang } from '../src/messages';
import fixtures from './fixtures/english.json';
import { mountPanel, REF, type Panel } from './helpers';

/** The code OrcaSlicer's `app_language()` gives for each language, and the browser tag used before the host answers. */
const HOST: Record<Exclude<Lang, 'en'>, string> = { 'pt-BR': 'pt_BR', es: 'es', de: 'de', 'zh-CN': 'zh_CN', pl: 'pl_PL' };
const TRANSLATED = Object.keys(HOST) as Exclude<Lang, 'en'>[];

let panel: Panel;
beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
});
afterEach(() => {
  panel?.unmount();
  vi.useRealTimers();
});

/** Everything on the page that is read or spoken. */
function readable(): string[] {
  const out = new Set<string>();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) out.add(node.textContent!.trim());
  for (const el of document.body.querySelectorAll('*')) {
    for (const name of ['title', 'aria-label', 'aria-valuetext', 'aria-description', 'placeholder', 'alt']) out.add(el.getAttribute(name)?.trim() ?? '');
  }
  out.delete('');
  return [...out];
}

const ctx = (objects: number | null, name: string | null = 'P') => ({ plate: 2, objects, name, problem: null });
const PATH = '/s/P-arranged.3mf';
const done = {
  ...REF, code: 'done', placed: 2, added: 300, by_object: { a: 150, b: 150 }, limit: true, max: 300, tower: true, clearance: 20, keep: true, right: true, unplaced: ['c', 'd'], open: 'asked',
};

/** Walks one panel through every state and returns what the page says in each. The same walk in any language. */
async function walk(lang: Lang): Promise<string[][]> {
  const host = lang === 'en' ? 'en_US' : HOST[lang];
  const seen: string[][] = [];
  const look = () => void seen.push(readable());
  const sync = (extra: Record<string, unknown>) => panel.deliver({ sync: true, lang: host, job: null, busy: false, context: 'x', stale: '', stale_ref: null, ...extra } as never);
  const running = (msg: Record<string, unknown>) => sync({ job: 'j1', busy: true, text: 'EN', ctx: ctx(3), msg: { ...REF, ...msg } });

  vi.useFakeTimers(); // the page's clock: the walk ends with the plugin falling silent
  panel = await mountPanel({ language: lang === 'en' ? 'en-US' : lang });
  look(); // loading, in the browser's language
  for (const c of [ctx(3), ctx(1, null), ctx(0), { plate: null, objects: null, name: null, problem: 'no project' }]) {
    await sync({ ctx: c });
    look();
  }
  await sync({ ctx: ctx(3), requirements: [{ id: 'numpy', name: 'numpy', status: 'missing', version: null, detail: 'boom' }] });
  look(); // a missing requirement: the reason, the section and how to fix it
  await sync({ ctx: ctx(3), requirements: [] });
  await panel.typeSpacing('');
  panel.q('#go')!.click();
  await panel.flush();
  look(); // the spacing error
  await panel.typeSpacing('2');

  for (const phase of ['read', 'footprint', 'place', 'patterns']) {
    await running({ code: 'progress', phase, i: 6, n: 14, count: null, subject: 'b' });
    look();
  }
  await running({ code: 'progress', phase: 'fill', i: 5, n: null, count: 5, subject: '' });
  look();
  await sync({ job: 'j1', busy: true, text: 'EN', ctx: { ...ctx(3), plate: 1 }, msg: { ...REF, code: 'reading' } }); // a run for another plate names it
  look();
  panel.q('#cancel')!.click();
  await panel.flush();
  look(); // Cancelling…
  await running({ code: 'cancelling' });
  look();

  // Every structured message the plugin can post: failures with each cause, cancelled, saving, results, a stale run.
  for (const f of fixtures.cases as Array<Record<string, any>>) {
    await sync({ job: 'p', error: !!f.error, text: f.msg ? 'EN' : '', msg: f.msg, path: f.path || null, stale: f.stale || '', stale_ref: f.stale_ref || null, ctx: ctx(3) });
    look();
  }

  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.resolve() } });
  await sync({ job: 'j9', path: PATH, text: 'EN', ctx: ctx(3), msg: done });
  look(); // the result with every note, the saved copy and New arrangement in the bar
  panel.q('#copy')!.click();
  await panel.flush();
  look(); // Path copied.
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.reject(new Error('denied')) } });
  panel.q('#copy')!.click();
  await panel.flush();
  await panel.flush();
  look(); // the path is selected instead

  await vi.advanceTimersByTimeAsync(15000);
  await panel.flush();
  look(); // no answer from the plugin
  panel.q('#retry')!.click();
  await panel.flush();
  look(); // Trying…
  vi.useRealTimers();
  panel.unmount();
  return seen;
}

// What the walk itself put on the page (project, parts, paths, raw details), names that are never translated, and units.
const DATA = /\/s\/P-arranged\.3mf|P-arranged\.3mf|no project|no permission|disk full|boom|inch|numpy|Pillow|block|\b[a-dP]\b/g;
const NAMES = /Shape-Aware Arrange|Organizer|OrcaSlicer|Orca Cloud|Orca|\bmm\b|\bXY\b|\b3D\b|Ctrl\+C|⌘C|\bMac\b/g;
const hasWords = (text: string) => /\p{L}/u.test(text.replace(DATA, '').replace(NAMES, ''));

describe('the page in every language', () => {
  let english: string[][];
  beforeEach(async () => {
    english ??= await walk('en');
  });

  it('walks through every state in English', () => {
    expect(english.length).toBeGreaterThan(90);
    expect(english.flat().some((t) => t === 'Organizer is not answering.')).toBe(true);
    expect(english.flat().some((t) => t === 'Technical details')).toBe(true);
    expect(english.flat().some((t) => t === 'Requirements')).toBe(true);
    expect(english.flat().some((t) => t.startsWith('Could not copy automatically'))).toBe(true);
  });

  it.each(TRANSLATED)('%s: no state of the page shows an English sentence', async (lang) => {
    const translated = await walk(lang);
    expect(translated.length).toBe(english.length);
    translated.forEach((texts, state) => {
      const left = texts.filter((t) => english[state].includes(t) && hasWords(t));
      expect(left, `${lang}, state ${state}`).toEqual([]);
      expect(texts.length, `${lang}, state ${state}: as much to read as in English`).toBe(english[state].length);
    });
    expect(document.documentElement.lang).toBe(lang);
  });

  it('zh-CN: only names, units and keyboard keys are in Latin letters', async () => {
    const texts = (await walk('zh-CN')).flat();
    for (const t of texts) expect(t.replace(DATA, '').replace(NAMES, '').replace(/Brim|build/g, ''), t).not.toMatch(/[A-Za-z]/);
  });
});

interface Wording {
  host: string[];
  reading: string;
  go: string;
  fill: string;
  context: string;
  bar: string;
  spacingError: string;
  decimal: string;
  phase: string;
  figure: string;
  otherPlate: string;
  copies: string;
  placed: string;
  added: string;
  unplaced: string;
  open: string;
  title: string;
  opened: string;
  again: string;
  failed: string;
  startFailed: string;
  stale: string;
  offline: string;
}

const WORDING: Record<'es' | 'de' | 'zh-CN' | 'pl', Wording> = {
  es: {
    host: ['es', 'es_ES', 'es-MX'],
    reading: 'Leyendo…',
    go: 'Organizar cama 2',
    fill: 'Llenar cama 2',
    context: 'P · cama 2 seleccionada · 3 objetos en el proyecto',
    bar: 'P · Cama 2',
    spacingError: 'Escribe una separación de 0 mm o más.',
    decimal: '2,5',
    phase: 'P, cama 2: Colocando las piezas',
    figure: '7 de 14',
    otherPlate: 'Cama 2 · b',
    copies: '5 copias añadidas',
    placed: '2 objetos colocados.',
    added: 'Se llenó la cama con 300 copias más (a ×150, b ×150).',
    unplaced: 'No cupieron (se movieron fuera de la cama): c, d',
    open: 'Abrir de nuevo',
    title: 'Cama 2 llena',
    opened: 'Se pidió a OrcaSlicer que abriera esta copia.',
    again: 'Nueva organización',
    failed: 'P, cama 2: no se pudo organizar.\nLa cama 2 no tiene objetos imprimibles.\nSiguiente paso: Selecciona otra cama o añade piezas a esta.',
    startFailed: 'No se pudo iniciar: La rotación debe ser automática, solo pasos de 90° o mantener la orientación.\nSiguiente paso: Elige una de las tres opciones.',
    stale: 'La ejecución de abajo es de sin título, cama 2, no de la selección actual.',
    offline: 'Organizer no responde.',
  },
  de: {
    host: ['de', 'de_DE', 'de-AT'],
    reading: 'Wird gelesen…',
    go: 'Platte 2 anordnen',
    fill: 'Platte 2 füllen',
    context: 'P · Platte 2 ausgewählt · 3 Objekte im Projekt',
    bar: 'P · Platte 2',
    spacingError: 'Geben Sie einen Abstand von 0 mm oder mehr ein.',
    decimal: '2,5',
    phase: 'P, Platte 2: Teile werden platziert',
    figure: '7 von 14',
    otherPlate: 'Platte 2 · b',
    copies: '5 Kopien hinzugefügt',
    placed: '2 Objekte platziert.',
    added: 'Platte mit 300 zusätzlichen Kopien gefüllt (a ×150, b ×150).',
    unplaced: 'Kein Platz (neben die Platte verschoben): c, d',
    open: 'Erneut öffnen',
    title: 'Platte 2 gefüllt',
    opened: 'OrcaSlicer wurde gebeten, diese Kopie zu öffnen.',
    again: 'Neue Anordnung',
    failed: 'P, Platte 2: Anordnen ist fehlgeschlagen.\nPlatte 2 enthält keine druckbaren Objekte.\nNächster Schritt: Wählen Sie eine andere Platte oder legen Sie Teile auf diese.',
    startFailed: 'Start nicht möglich: Die Drehung muss „Automatisch“, „Nur 90°-Schritte“ oder „Ausrichtung beibehalten“ sein.\nNächster Schritt: Wählen Sie eine der drei Optionen.',
    stale: 'Der Durchlauf unten gehört zu unbenannt, Platte 2, nicht zur aktuellen Auswahl.',
    offline: 'Organizer antwortet nicht.',
  },
  'zh-CN': {
    host: ['zh_CN', 'zh-CN', 'zh_TW'],
    reading: '读取中…',
    go: '自动摆放盘 2',
    fill: '铺满盘 2',
    context: 'P · 已选择盘 2 · 项目中有 3 个对象',
    bar: 'P · 盘 2',
    spacingError: '请输入不小于 0 mm 的间距。',
    decimal: '2.5',
    phase: 'P，盘 2：正在摆放零件',
    figure: '7 / 14',
    otherPlate: '盘 2 · b',
    copies: '已添加 5 个副本',
    placed: '已摆放 2 个对象。',
    added: '已用 300 个额外副本铺满该盘（a ×150、b ×150）。',
    unplaced: '放不下（已移到盘外）：c、d',
    open: '再次打开',
    title: '盘 2 已铺满',
    opened: '已请求 OrcaSlicer 打开此副本。',
    again: '重新摆放',
    failed: 'P，盘 2：无法自动摆放。\n盘 2 上没有可打印的对象。\n下一步：请选择其他盘，或向该盘添加零件。',
    startFailed: '无法开始：旋转必须是“自动”“仅 90° 步长”或“保持朝向”。\n下一步：请从三个选项中选择一个。',
    stale: '下方的任务属于“未命名，盘 2”，不属于当前选择。',
    offline: 'Organizer 无响应。',
  },
  pl: {
    host: ['pl_PL', 'pl', 'pl-PL'],
    reading: 'Wczytywanie…',
    go: 'Rozmieść płytę 2',
    fill: 'Wypełnij płytę 2',
    context: 'P · wybrana płyta 2 · 3 obiekty w projekcie',
    bar: 'P · Płyta 2',
    spacingError: 'Wpisz odstęp 0 mm lub większy.',
    decimal: '2,5',
    phase: 'P, płyta 2: Rozmieszczanie części',
    figure: '7 z 14',
    otherPlate: 'Płyta 2 · b',
    copies: 'Dodano 5 kopii',
    placed: 'Rozmieszczono 2 obiekty.',
    added: 'Wypełniono płytę: dodano 300 kopii (a ×150, b ×150).',
    unplaced: 'Nie zmieściły się (przeniesione nad płytę): c, d',
    open: 'Otwórz ponownie',
    title: 'Płyta 2 wypełniona',
    opened: 'Poproszono OrcaSlicer o otwarcie tej kopii.',
    again: 'Nowe rozmieszczenie',
    failed: 'P, płyta 2: nie udało się rozmieścić.\nPłyta 2 nie ma obiektów do druku.\nNastępny krok: Wybierz inną płytę albo dodaj na nią części.',
    startFailed: 'Nie można uruchomić: Obrót musi mieć wartość „Automatyczny”, „Tylko co 90°” lub „Zachowaj orientację”.\nNastępny krok: Wybierz jedną z trzech opcji.',
    stale: 'Poniższa operacja dotyczy: bez nazwy, płyta 2, a nie bieżącego wyboru.',
    offline: 'Organizer nie odpowiada.',
  },
};

describe.each(Object.entries(WORDING) as [keyof typeof WORDING, Wording][])('the page in %s', (lang, w) => {
  const sync = (extra: Record<string, unknown> = {}) => panel.deliver({ sync: true, lang: w.host[0], job: null, busy: false, ...extra } as never);

  it('follows the browser until OrcaSlicer names its language, then OrcaSlicer wins', async () => {
    panel = await mountPanel({ language: lang });
    expect(document.documentElement.lang).toBe(lang);
    expect(panel.text('#context')).toBe(w.reading);
    await panel.deliver({ sync: true, lang: 'en_US', job: null, busy: false, ctx: ctx(3) });
    expect(document.documentElement.lang).toBe('en');
    expect(panel.text('#go')).toContain('Arrange plate 2');
    for (const host of w.host) {
      await panel.deliver({ sync: true, lang: 'en_US', job: null, busy: false, ctx: ctx(3) });
      await panel.deliver({ sync: true, lang: host, job: null, busy: false, ctx: ctx(3) });
      expect(document.documentElement.lang, host).toBe(lang);
      expect(panel.text('#go'), host).toContain(w.go);
    }
  });

  it('names the plate, the project and what the buttons do', async () => {
    panel = await mountPanel();
    await sync({ ctx: ctx(3) });
    expect(panel.text('#go')).toBe(w.go);
    expect(panel.text('#fill')).toBe(w.fill);
    expect(panel.text('#context .oui-sr')).toBe(w.context);
    expect(panel.q('#context')!.getAttribute('title')).toBe(w.context);
    expect(panel.text('#context [aria-hidden]')).toBe(w.bar);
    expect(panel.status()).toBe(''); // nothing ran yet: nothing is said
    expect(panel.text('h1')).toBe('Organizer'); // the plugin's name is not translated
    expect(document.title).toBe('Organizer');
  });

  it('explains a blank spacing and reads a decimal as the language writes it', async () => {
    panel = await mountPanel();
    await sync({ ctx: ctx(3) });
    panel.take();
    await panel.typeSpacing('');
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.text('#spacing-error')).toBe(w.spacingError);
    await panel.typeSpacing(w.decimal);
    expect(panel.q<HTMLInputElement>('#spacing')!.value).toBe(w.decimal + ' mm'); // the unit stays a symbol
    panel.q('#go')!.click();
    await panel.flush();
    expect(panel.takeRequests()).toEqual([{ action: 'arrange', spacing: 2.5, rotation: 'auto' }]);
  });

  it('shows a run, its count and the plate it is for', async () => {
    panel = await mountPanel();
    const place = { ...REF, code: 'progress', phase: 'place', i: 6, n: 14, count: null, subject: 'b' };
    await sync({ job: 'j1', busy: true, text: 'EN', ctx: ctx(3), msg: place });
    expect(panel.status()).toBe(w.phase);
    expect(panel.text('.oui-job-figure')).toBe(w.figure);
    expect(panel.q('#job [role=progressbar]')!.getAttribute('aria-valuetext')).toBe(w.figure + ' · b');
    await sync({ job: 'j1', busy: true, text: 'EN', ctx: { ...ctx(3), plate: 1 }, msg: place });
    expect(panel.text('.oui-job-detail')).toBe(w.otherPlate);
    await sync({ job: 'j1', busy: true, text: 'EN', ctx: ctx(3), msg: { ...REF, code: 'progress', phase: 'fill', i: 5, n: null, count: 5, subject: '' } });
    expect(panel.text('.oui-job-detail')).toBe(w.copies);
    expect(panel.text('.oui-job-figure')).toBe('+5');
  });

  it('tells the result, what did not fit and how to open the copy', async () => {
    panel = await mountPanel();
    await sync({ job: 'j1', path: PATH, text: 'EN', ctx: ctx(3), msg: done });
    const lines = panel.status().split('\n');
    expect(lines[0]).toBe(w.placed);
    expect(lines[1]).toBe(w.added);
    expect(lines).toContain(w.unplaced);
    expect(lines.length).toBe(9);
    expect(lines[8]).toBe(w.opened);
    expect(panel.text('#file')).toBe('P');
    expect(panel.text('#saved-what')).toBe(w.title);
    expect(panel.text('#open')).toBe(w.open); // already asked: the button is the second try
    expect(panel.q('#again')!.getAttribute('aria-label')).toBe(w.again);
    expect(panel.q('#again')!.getAttribute('title')).toBe(w.again);
  });

  it('explains a failure, a failed start, a stale run and a silent plugin', async () => {
    vi.useFakeTimers();
    panel = await mountPanel();
    await sync({ job: 'j2', error: true, text: 'EN', msg: { stem: 'P', plate: 2, action: 'arrange', code: 'failed', cause: { code: 'plate_empty', plate: 2 }, detail: 'x' } });
    expect(panel.status()).toBe(w.failed);
    expect(panel.isError()).toBe(true);
    await panel.deliver({ text: 'EN', error: true, busy: false, lang: w.host[0], msg: { code: 'start_failed', cause: { code: 'bad_rotation' }, detail: 'd' } });
    expect(panel.status()).toBe(w.startFailed);
    await sync({ stale: 'x', stale_ref: { stem: null, plate: 2, action: 'arrange' } });
    expect(panel.text('#stale strong')).toBe(w.stale);
    await vi.advanceTimersByTimeAsync(15000);
    expect(panel.text('#offline strong')).toBe(w.offline);
  });
});
