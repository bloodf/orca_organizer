import { afterEach, describe, expect, it } from 'vitest';
import { applyOrcaTheme, setOrcaAccent } from '@orca-plugins/ui-foundation/src';
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
    expect(token('--oui-primary')).toBe('#00675b'); // OrcaSlicer's teal, whatever the plugin
    expect(token('--oui-background')).toBe('#242428'); // OrcaSlicer's grey: the page is not tinted by the accent
    expect(token('--oui-accent-solid')).toBe('#00628c'); // sky: the mark tile
    applyOrcaTheme('light');
    expect(document.documentElement.classList.contains('orca-dark')).toBe(false);
    expect(token('--oui-primary')).toBe('#008172');
    expect(token('--oui-background')).toBe('#f0f0f0');
    expect(token('--oui-accent-solid')).toBe('#00628c'); // the mark tile is the same in both themes
    expect(token('--oui-band')).toBe(''); // 5.0: no accent-filled band
  });
});
