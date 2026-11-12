# Organizer panel (Vue 3 + PrimeVue 4)

The panel of the Organizer tab and of the dock opened by the *Shape-Aware Arrange* Script is one Vue 3 app:
SFCs in `src/`, PrimeVue 4.5.5 controls, the shared Orca preset in `shared/`. Both entries load the same page.
The plugin is still ONE `.py` file: the build embeds the compiled single-file HTML in `../shape_arrange.py`
(`PANEL_HTML`, between the `# >>> PANEL_HTML` / `# <<< PANEL_HTML` markers). Nothing is read from sibling files
at run time, nothing is downloaded, and the page needs no `unsafe-eval` (templates are compiled ahead of time).

| Path | What |
|---|---|
| `src/App.vue`, `src/components/*.vue` | header, notices, form, job progress, status, result |
| `src/store.ts` | all reactive state, polling, focus requests; the plugin owns the job, the page only shows it |
| `src/messages.ts`, `src/status.ts` | pt-BR / en strings, structured `msg` to text lines |
| `src/protocol.ts`, `src/hostTheme.ts` | message types and the host bridge; light/dark detection |
| `shared/` | vendored ui-foundation 2.0.3 (see `shared/manifest.json`). Do not edit; `npm run check:shared` |
| `tests/` | vitest + jsdom component tests (`panel`, `messages`, `english-parity`, `theme`) |
| `scripts/embed.mjs` | embeds `dist/index.html` into `shape_arrange.py` (`--check` for CI) |
| `scripts/verify-browser.mjs` | production page under CSP in headless Chrome with a MOCK host bridge, screenshots |

## Build and regenerate

```sh
cd ui
npm ci --ignore-scripts
npm run build          # vue-tsc, vite build -> dist/index.html, embed into ../shape_arrange.py
npm test               # component tests
npm run check:shared   # shared manifest hashes + palette contrast
npm run check:embedded # CI: rebuild and fail if shape_arrange.py is stale
```

After changing wording in Python (`ERRORS_EN`, `OPEN_EN`, summaries), run
`uv run --with numpy --with pillow python tests/test_panel_js.py --write-fixtures` and update `src/messages.ts`; the
parity test fails until both match. Run `node scripts/verify-browser.mjs --out <dir>` to refresh MOCK screenshots
(always with `--use-mock-keychain --password-store=basic`, already in the script).

`vite.config.ts` must keep `define: { __VUE_OPTIONS_API__: true }`: PrimeVue uses the Options API internally.
`node_modules`, `dist` and `evidence` are not committed; the lockfile is. `shape_arrange.py` is committed with the
embedded page so that releases need only Python.

## Third-party code in the embedded page

Vue 3.5.45 (MIT), PrimeVue 4.5.5 (MIT), @primeuix/themes 1.2.5 (MIT). Licenses: `shared/licenses/`. Dev tools
(Vite, vitest, jsdom, TypeScript) are not shipped.
