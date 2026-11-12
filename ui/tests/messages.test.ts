import { describe, expect, it } from 'vitest';
import { ENGLISH, PORTUGUESE, numberFormat, pickLang, strings } from '../src/messages';
import fixtures from './fixtures/english.json';

// The two languages carry the same keys, with the same kind of value (string / function / table).
function shape(a: unknown, b: unknown, path: string): void {
  expect(typeof a, path).toBe(typeof b);
  if (a && typeof a === 'object') {
    expect(Object.keys(a).sort(), path).toEqual(Object.keys(b as object).sort());
    for (const k of Object.keys(a)) shape((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`);
  }
}

describe('messages', () => {
  it('English and Portuguese have identical keys', () => shape(ENGLISH, PORTUGUESE, 'L'));

  it('every error code Python can raise is explained in both languages', () => {
    for (const code of fixtures.error_codes) {
      expect(ENGLISH.err[code], code).toBeTruthy();
      expect(PORTUGUESE.err[code], code).toBeTruthy();
    }
    expect(Object.keys(ENGLISH.err).filter((c) => c !== 'unexpected').sort()).toEqual([...fixtures.error_codes].sort());
  });

  it('maps host and browser languages: pt_BR, pt-BR and pt are Portuguese, everything else English (X04)', () => {
    for (const l of ['pt_BR', 'pt-BR', 'pt', 'PT_br']) expect(pickLang(l), l).toBe('pt-BR');
    for (const l of ['en_US', 'fr_FR', 'ru_RU', '', null, undefined]) expect(pickLang(l), String(l)).toBe('en');
  });

  it('formats numbers by language for presentation only', () => {
    expect(numberFormat('en')(1234.5)).toBe('1,234.5');
    expect(numberFormat('pt-BR')(1234.5)).toBe('1.234,5');
    expect(strings('pt-BR').ctx({ name: 'P', plate: 1, objects: 1234 })).toBe('P · placa 1 selecionada · 1.234 objetos no projeto');
  });

  it('pluralises one and many (O05)', () => {
    const en = strings('en'), pt = strings('pt-BR');
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
  });
});
