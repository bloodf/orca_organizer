import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [vue(), viteSingleFile()],
  // PrimeVue components use Vue's Options API internally, even though the SFCs here use <script setup>.
  define: { __VUE_OPTIONS_API__: true, __VUE_PROD_DEVTOOLS__: false, __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: false },
  // The shared components are imported through their index, which also names the ones this page never uses: their styles are
  // side effects, so without this every shared component's CSS would ship. Unused ones are dropped.
  build: { target: 'safari14', outDir: 'dist', emptyOutDir: true, modulePreload: { polyfill: false }, rollupOptions: { treeshake: { moduleSideEffects: (id) => !/\/ui-foundation\/src\/components\/[^/?]+\.vue$/.test(id) } } },
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
