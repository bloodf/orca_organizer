import { afterEach, describe, expect, it } from 'vitest';
import { applyOrcaTheme } from '../shared/src';
import { detectHostTheme, luminance, parseColor } from '../src/hostTheme';

afterEach(() => {
  document.documentElement.removeAttribute('style');
  document.documentElement.className = '';
});

describe('host theme', () => {
  it('parses hex and rgb colours', () => {
    expect(parseColor('#fff')).toEqual([255, 255, 255]);
    expect(parseColor('#1e1e2e')).toEqual([30, 30, 46]);
    expect(parseColor('rgb(10, 20, 30)')).toEqual([10, 20, 30]);
    expect(parseColor('rgba(10 20 30 / .5)')).toEqual([10, 20, 30]);
    expect(parseColor('banana')).toBeNull();
    expect(luminance([0, 0, 0])).toBe(0);
    expect(luminance([255, 255, 255])).toBeCloseTo(1, 5);
  });

  it('a dark host background means dark, a light one light, whatever the OS prefers', () => {
    document.documentElement.style.setProperty('--orca-bg', '#1e1e1e');
    expect(detectHostTheme()).toBe('dark');
    document.documentElement.style.setProperty('--orca-bg', '#fafafa');
    expect(detectHostTheme()).toBe('light');
  });

  it('falls back to the operating system preference without a host variable', () => {
    expect(detectHostTheme()).toBe('light');
    window.matchMedia = ((q: string) => ({ matches: true, media: q }) as MediaQueryList) as typeof window.matchMedia;
    expect(detectHostTheme()).toBe('dark');
    window.matchMedia = ((q: string) => ({ matches: false, media: q }) as MediaQueryList) as typeof window.matchMedia;
  });

  it('applies the shared preset tokens to the document root, also for teleported dialogs', () => {
    applyOrcaTheme('dark');
    expect(document.documentElement.classList.contains('orca-dark')).toBe(true);
    expect(document.documentElement.style.getPropertyValue('--oui-primary')).toBe('#79d8c7');
    applyOrcaTheme('light');
    expect(document.documentElement.classList.contains('orca-dark')).toBe(false);
    expect(document.documentElement.style.getPropertyValue('--oui-primary')).toBe('#00675b');
  });
});
