import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [vue(), viteSingleFile()],
  // PrimeVue components use Vue's Options API internally, even though the SFCs here use <script setup>.
  define: { __VUE_OPTIONS_API__: true, __VUE_PROD_DEVTOOLS__: false, __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false },
  build: { target: 'safari14', outDir: 'dist', emptyOutDir: true, modulePreload: { polyfill: false } },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.ts'],
    setupFiles: ['tests/setup.ts'],
    restoreMocks: true,
    coverage: {
      provider: 'v8',
      include: ['src/**'],
      exclude: ['src/main.ts'], // bootstrap only: exercised by scripts/verify-browser.mjs against the production page
      thresholds: { statements: 80, branches: 80, functions: 80, lines: 80 },
    },
    // PrimeVue must share the app's Vue instance, or template refs and hooks of its components never connect.
    server: { deps: { inline: [/primevue/, /@primeuix/, /@primevue/] } },
  },
});
