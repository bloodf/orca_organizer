import { afterEach, describe, expect, it, vi } from 'vitest';
import { ENGLISH, LANGS, localeOf, numberFormat, pickLang, strings, type Lang } from '../src/messages';
import fixtures from './fixtures/english.json';

const OTHERS = LANGS.filter((l) => l !== 'en');
afterEach(() => vi.restoreAllMocks());

// Every language carries the same keys, with the same kind of value (string / function / table).
function shape(a: unknown, b: unknown, path: string): void {
  expect(typeof a, path).toBe(typeof b);
  if (a && typeof a === 'object') {
    expect(Object.keys(a).sort(), path).toEqual(Object.keys(b as object).sort());
    for (const k of Object.keys(a)) shape((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`);
  }
}

type Leaf = { path: string; value: string | ((a: Record<string, unknown>) => string | [string, string]) | [string, string] };
/** Every entry of a table: plain strings, [cause, next step] pairs and the functions that write either. */
function leaves(table: unknown, path = ''): Leaf[] {
  if (typeof table === 'string' || typeof table === 'function' || Array.isArray(table)) return [{ path, value: table as Leaf['value'] }];
  return Object.entries(table as object).flatMap(([k, v]) => leaves(v, path ? `${path}.${k}` : k));
}
const at = (lang: Lang, path: string) => leaves(strings(lang)).find((l) => l.path === path)!.value;
/** [cause, next step] of an error code in a language. */
const cause = (lang: Lang, code: string, args: object = {}): string[] => {
  const entry = strings(lang).err[code];
  return typeof entry === 'function' ? (entry as unknown as (a: object) => string[])(args) : entry;
};
const text = (value: Leaf['value'], args: Record<string, unknown>) => [typeof value === 'function' ? value(args) : value].flat().join('\n');

// One distinct value per argument a message can take, so a translation that drops one is caught. `i` is shown as i + 1.
const ARGS = {
  name: '«name»', plate: 71, objects: 72, problem: '«problem»', ref: '«ref»', i: 73, n: 75, count: 76, subject: '«subject»', act: '«act»',
  why: '«why»', max: 77, unit: '«unit»', plates: 78, res: 79, spacing: 80, limit: 81, clearance: 82, detail: '«detail»', placed: 83,
  added: 84, per: '«per»', names: '«names»',
};
const SHOWN = Object.values({ ...ARGS, i: 74 }).map(String);
// The other shape of each message: no object count, a fill (no total) and no per-object list.
const BARE = { ...ARGS, objects: null, n: null, per: '', subject: '' };

describe('messages', () => {
  it('supports English, Brazilian Portuguese, Spanish, German and Simplified Chinese', () => {
    expect([...LANGS]).toEqual(['en', 'pt-BR', 'es', 'de', 'zh-CN']);
  });

  it.each(OTHERS)('%s has exactly the keys of English: none missing, none extra', (lang) => shape(ENGLISH, strings(lang), lang));

  it.each(OTHERS)('%s uses the same arguments as English in every message', (lang) => {
    for (const { path, value } of leaves(ENGLISH)) {
      if (typeof value !== 'function') continue;
      for (const args of [ARGS, BARE]) {
        const english = text(value, args);
        const translated = text(at(lang, path), args);
        for (const token of SHOWN) expect(translated.includes(token), `${lang} ${path}: ${token} in "${translated}"`).toBe(english.includes(token));
        expect(translated, `${lang} ${path}`).not.toMatch(/undefined|null|NaN|\[object/);
        expect(translated.split('\n').length, `${lang} ${path}: lines`).toBe(english.split('\n').length);
      }
    }
  });

  // What is the same word in both languages, or a name that is not translated.
  const SAME: Partial<Record<Lang, string[]>> = { de: ['details'] };
  it.each(OTHERS)('%s leaves no entry in English', (lang) => {
    for (const { path, value } of leaves(ENGLISH)) {
      // `sep` and `list` are punctuation: Chinese has its own (checked below), the others share English's.
      if (path === 'title' || path === 'sep' || path === 'list' || SAME[lang]?.includes(path)) continue;
      for (const args of [ARGS, BARE]) expect(text(at(lang, path), args), `${lang} ${path}`).not.toBe(text(value, args));
    }
  });

  it('Chinese is written in Chinese: only names, units and keys stay in Latin letters, and punctuation is full-width', () => {
    const names = /Organizer|Shape-Aware Arrange|OrcaSlicer|Orca|Brim|build|Ctrl\+C|⌘C|Mac|\bmm\b|\bXY\b|\b3D\b|«\w+»/g;
    for (const { path, value } of leaves(strings('zh-CN'))) {
      for (const args of [ARGS, BARE]) {
        const out = text(value, args);
        expect(out.replace(names, ''), `${path}: "${out}"`).not.toMatch(/[A-Za-z]/);
        expect(out.replace(names, ''), `${path}: "${out}"`).not.toMatch(/[一-鿿][,.:;!?()]|[,:;!?()][一-鿿]/);
      }
    }
    expect(strings('zh-CN').sep).toBe('：');
    expect(strings('zh-CN').list).toBe('、');
  });

  it('every error code Python can raise is explained in every language', () => {
    for (const lang of LANGS) for (const code of fixtures.error_codes) expect(strings(lang).err[code], `${lang} ${code}`).toBeTruthy();
    expect(Object.keys(ENGLISH.err).filter((c) => c !== 'unexpected').sort()).toEqual([...fixtures.error_codes].sort());
  });

  it('maps host and browser languages by their primary part; anything else is English (X04)', () => {
    for (const l of ['pt_BR', 'pt-BR', 'pt', 'PT_br', 'pt_PT']) expect(pickLang(l), l).toBe('pt-BR');
    for (const l of ['es', 'es_ES', 'es-MX', 'es-419', 'ES']) expect(pickLang(l), l).toBe('es');
    for (const l of ['de', 'de_DE', 'de-CH', 'DE_at']) expect(pickLang(l), l).toBe('de');
    for (const l of ['zh_CN', 'zh-CN', 'zh', 'zh-Hans-CN', 'zh_TW', 'zh-Hant-TW', 'ZH_cn']) expect(pickLang(l), l).toBe('zh-CN');
    for (const l of ['en', 'en_US', 'fr_FR', 'ru_RU', 'ja', 'ko', 'esperanto', 'constructor', '', null, undefined]) expect(pickLang(l), String(l)).toBe('en');
  });

  it('formats numbers by language for presentation only', () => {
    expect(numberFormat('en')(1234.5)).toBe('1,234.5');
    expect(numberFormat('pt-BR')(1234.5)).toBe('1.234,5');
    expect(numberFormat('es')(12345.5)).toBe('12.345,5');
    expect(numberFormat('de')(1234.5)).toBe('1.234,5');
    expect(numberFormat('zh-CN')(1234.5)).toBe('1,234.5');
    expect(strings('pt-BR').ctx({ name: 'P', plate: 1, objects: 1234 })).toBe('P · placa 1 selecionada · 1.234 objetos no projeto');
    expect(strings('es').ctx({ name: 'P', plate: 1, objects: 12345 })).toBe('P · cama 1 seleccionada · 12.345 objetos en el proyecto');
    expect(strings('de').ctx({ name: 'P', plate: 1, objects: 1234 })).toBe('P · Platte 1 ausgewählt · 1.234 Objekte im Projekt');
    expect(strings('zh-CN').ctx({ name: 'P', plate: 1, objects: 1234 })).toBe('P · 已选择盘 1 · 项目中有 1,234 个对象');
    // decimals in running text follow the language too; units stay symbols
    expect(cause('de', 'spacing_large', { spacing: 300, limit: 37.5 })[1]).toBe('Geben Sie einen Abstand unter 37,5 mm ein und versuchen Sie es erneut.');
    expect(cause('es', 'bad_res')[1]).toContain('0,5');
    expect(cause('zh-CN', 'bad_res')[1]).toContain('0.5');
  });

  it('writes Spanish and German numbers as the system region does, when the system speaks that language', () => {
    const system = (...languages: string[]) => vi.spyOn(navigator, 'languages', 'get').mockReturnValue(languages);
    system('en-US', 'en');
    expect(LANGS.map(localeOf)).toEqual(['en-US', 'pt-BR', 'es', 'de', 'zh-CN']);
    system('es-MX', 'es', 'en');
    expect(localeOf('es')).toBe('es-MX'); // 2.5, where Spain and Argentina write 2,5
    expect(new Intl.NumberFormat(localeOf('es')).format(2.5)).toBe('2.5');
    expect(localeOf('de')).toBe('de');
    system('de-CH');
    expect(localeOf('de')).toBe('de-CH');
    expect(localeOf('es')).toBe('es');
    system('pt-PT', 'fr-CA');
    expect(LANGS.map(localeOf)).toEqual(['en-US', 'pt-BR', 'es', 'de', 'zh-CN']); // other languages keep one way of writing
  });

  it('pluralises by each language\'s own rule; Chinese has no plural (O05)', () => {
    const en = strings('en'), pt = strings('pt-BR'), es = strings('es'), de = strings('de'), zh = strings('zh-CN');
    expect(en.ctx({ name: 'P', plate: 1, objects: 1 })).toMatch(/1 object in/);
    expect(en.ctx({ name: 'P', plate: 1, objects: 2 })).toMatch(/2 objects in/);
    expect(pt.ctx({ name: 'P', plate: 1, objects: 1 })).toMatch(/1 objeto no/);
    expect(pt.ctx({ name: 'P', plate: 1, objects: 2 })).toMatch(/2 objetos no/);
    expect(en.count({ i: 0, n: null, count: 1, subject: '' })).toBe('1 copy added');
    expect(en.count({ i: 0, n: null, count: 2, subject: '' })).toBe('2 copies added');
    expect(pt.count({ i: 0, n: null, count: 1, subject: '' })).toBe('1 cópia adicionada');
    expect(pt.count({ i: 0, n: null, count: 2, subject: '' })).toBe('2 cópias adicionadas');
    expect(pt.res.placed({ ref: 'r', placed: 1 })).toBe('r: 1 objeto posicionado.');
    expect(pt.res.placed({ ref: 'r', placed: 2 })).toBe('r: 2 objetos posicionados.');

    expect([0, 1, 2].map((n) => es.ctxShort({ plate: 1, objects: n }))).toEqual(['Cama 1 · 0 objetos', 'Cama 1 · 1 objeto', 'Cama 1 · 2 objetos']);
    expect([1, 2].map((n) => es.count({ i: 0, n: null, count: n, subject: '' }))).toEqual(['1 copia añadida', '2 copias añadidas']);
    expect([1, 2].map((n) => es.res.placed({ ref: 'r', placed: n }))).toEqual(['r: 1 objeto colocado.', 'r: 2 objetos colocados.']);
    expect([1, 5].map((n) => es.res.added({ added: n, per: '' }))).toEqual(['Se llenó la cama con 1 copia más.', 'Se llenó la cama con 5 copias más.']);

    expect([0, 1, 2].map((n) => de.ctxShort({ plate: 1, objects: n }))).toEqual(['Platte 1 · 0 Objekte', 'Platte 1 · 1 Objekt', 'Platte 1 · 2 Objekte']);
    expect([1, 2].map((n) => de.count({ i: 0, n: null, count: n, subject: '' }))).toEqual(['1 Kopie hinzugefügt', '2 Kopien hinzugefügt']);
    expect([1, 2].map((n) => de.res.placed({ ref: 'r', placed: n }))).toEqual(['r: 1 Objekt platziert.', 'r: 2 Objekte platziert.']);
    expect([1, 5].map((n) => de.res.added({ added: n, per: 'a ×5' }))).toEqual(['Platte mit 1 zusätzlichen Kopie gefüllt (a ×5).', 'Platte mit 5 zusätzlichen Kopien gefüllt (a ×5).']);
    const gone = (lang: Lang, plates: number) => cause(lang, 'plate_gone', { plate: 9, plates })[0];
    expect([gone('es', 1), gone('es', 2)]).toEqual(['La cama 9 no existe: el proyecto tiene 1 cama.', 'La cama 9 no existe: el proyecto tiene 2 camas.']);
    expect([gone('de', 1), gone('de', 2)]).toEqual(['Platte 9 gibt es nicht: Das Projekt hat 1 Platte.', 'Platte 9 gibt es nicht: Das Projekt hat 2 Platten.']);

    expect([1, 2].map((n) => zh.ctxShort({ plate: 1, objects: n }))).toEqual(['盘 1 · 1 个对象', '盘 1 · 2 个对象']);
    expect([1, 2].map((n) => zh.count({ i: 0, n: null, count: n, subject: '' }))).toEqual(['已添加 1 个副本', '已添加 2 个副本']);
    expect(zh.res.added({ added: 5, per: 'a ×2、b ×3' })).toBe('已用 5 个额外副本铺满该盘（a ×2、b ×3）。');
    expect([gone('zh-CN', 1), gone('zh-CN', 2)]).toEqual(['盘 9 不存在：项目共有 1 个盘。', '盘 9 不存在：项目共有 2 个盘。']);
  });
});
