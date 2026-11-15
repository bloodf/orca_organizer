import { afterEach, describe, expect, it } from 'vitest';
import { applyOrcaTheme, setOrcaAccent } from '../shared/src';
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

  it('applies the shared tokens in the Organizer accent (sky) to the document root, also for teleported dialogs', () => {
    const token = (name: string) => document.documentElement.style.getPropertyValue(name);
    setOrcaAccent('sky');
    applyOrcaTheme('dark');
    expect(document.documentElement.classList.contains('orca-dark')).toBe(true);
    expect(document.documentElement.dataset.orcaUiAccent).toBe('sky');
    expect(token('--oui-primary')).toBe('#72cdf4');
    expect(token('--oui-background')).toBe('#0b1a23'); // the page is tinted by the accent
    expect(token('--oui-band')).toBe('#00628c');
    applyOrcaTheme('light');
    expect(document.documentElement.classList.contains('orca-dark')).toBe(false);
    expect(token('--oui-primary')).toBe('#00628c');
    expect(token('--oui-background')).toBe('#d6e2ea');
    expect(token('--oui-band')).toBe('#00628c'); // the app bar is the same band in both themes
  });
});
