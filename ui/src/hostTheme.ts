import type { OrcaTheme } from '../shared/src';

/** `#rgb`, `#rrggbb` or `rgb()/rgba()` to [r, g, b]; anything else (named colors, hsl) is normalised by a canvas. */
export function parseColor(value: string): [number, number, number] | null {
  const text = value.trim();
  const rgb = /^rgba?\(\s*(\d+)[ ,]+(\d+)[ ,]+(\d+)/i.exec(text);
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
  if (hex) {
    const h = hex[1].length === 3 ? [...hex[1]].map((c) => c + c).join('') : hex[1];
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
  }
  return null;
}

/** WCAG relative luminance, 0 (black) to 1 (white). */
export function luminance([r, g, b]: [number, number, number]): number {
  const [lr, lg, lb] = [r, g, b].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
}

function normalise(value: string): [number, number, number] | null {
  const direct = parseColor(value);
  if (direct) return direct;
  try {
    const context = document.createElement('canvas').getContext('2d');
    if (!context) return null;
    context.fillStyle = '#000';
    context.fillStyle = value;
    return parseColor(context.fillStyle);
  } catch {
    return null;
  }
}

/**
 * The host's light/dark mode. The host's own CSS variable `--orca-bg` is only read, never replaced:
 * a dark background means dark. Without it the operating system preference decides. Not a private Orca API.
 */
export function detectHostTheme(): OrcaTheme {
  try {
    for (const element of [document.documentElement, document.body]) {
      const raw = element ? getComputedStyle(element).getPropertyValue('--orca-bg') : '';
      const color = raw ? normalise(raw) : null;
      if (color) return luminance(color) < 0.4 ? 'dark' : 'light';
    }
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}
