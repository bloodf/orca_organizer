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

// jsdom never has focus; timed toasts count down only while the page is visible and focused (foundation 4.2.1).
Object.defineProperty(document, 'hasFocus', { configurable: true, writable: true, value: () => true });
