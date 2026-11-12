// jsdom lacks matchMedia and scroll APIs that the theme probe and PrimeVue overlays touch.
// Not tied to a theme: tests that need a dark preference override it.
Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  writable: true,
  value: (query: string) => ({
    matches: false, media: query, onchange: null,
    addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false,
  }),
});
