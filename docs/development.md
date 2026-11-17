# Development

Building the panel, running the tests and regenerating the listing images. How the arrangement itself works is
in [how-it-works.md](how-it-works.md); releasing is in [release.md](release.md).

## Panel source

The Organizer tab and the Script's dock panel are one Vue 3 + PrimeVue 5 app in `ui/` (see
`ui/README.md`), built on the shared Orca UI foundation 4.2.1 in the Organizer's sky accent. The plugin stays a single `.py` file: `cd ui && npm ci --ignore-scripts && npm run build`
compiles the app into one offline HTML file and embeds it in `shape_arrange.py` (`PANEL_HTML`).
Editing `ui/src` without rebuilding is caught by CI and by `tests/test_panel_js.py`. Python users who
only run or release the plugin need no Node: the built page is committed inside `shape_arrange.py`.

## Test

```sh
uv run --python 3.12 --with numpy --with pillow python tests/test_shape_arrange.py
uv run --python 3.12 --with numpy --with pillow python tests/test_packing.py
uv run --python 3.12 --with numpy --with pillow python tests/test_plugin.py
uv run --python 3.12 --with numpy --with pillow python tests/test_panel_js.py
```

`test_panel_js.py` needs Node and `npm ci` in `ui/`. It checks that the page embedded in
`shape_arrange.py` is exactly the one `ui/` builds, that it is offline and needs no
`unsafe-eval`, that its English wording equals the plugin's own, and runs the Vue component
tests (`ui/tests`: validation, focus, every state and message, polling, protocol).

`test_plugin.py` runs the plugin against a stub `orca` module and checks the
threading (the export and the file launch happen on the UI thread, the
nesting does not) and the run lifecycle: closing and reopening the panel,
cancelling, a failed save and earlier results that must stay. It also covers the Organizer
tab (registration, a host without Pages, one run shared with the Script's panel,
stale ids, reload, unload) against a stub Pages host. None of it runs inside OrcaSlicer.

`test_shape_arrange.py` builds a two-plate Orca-style project in which two combs fit on
plate 2 only if their teeth interlock. It arranges the project, then checks
that every part sits on the bed with no overlap and that plate 1 and every
other file in the 3MF are unchanged.

CI (`.github/workflows/ci.yml`) runs `npm run check:shared`, `npm run typecheck` and `npm run check:embedded`
in `ui/`, then the four test files above.

## Listing and README images

The images in `assets/orca-cloud/`, `assets/social/` and `assets/readme/` are built from two inputs: the
generated artwork kept at its native size in `assets/marketing/source/`, and screenshots of the built page.

```sh
cd ui && npm ci --ignore-scripts && npx vite build && cd ..
node assets/marketing/render-screens.mjs
uv run --python 3.12 --with numpy --with pillow python assets/marketing/compose.py
```

`render-screens.mjs` opens `ui/dist/index.html` in headless Chrome (with `--use-mock-keychain
--password-store=basic`) against a mock host bridge with made-up data, in English and pt-BR, light and dark.
Nothing in it talks to OrcaSlicer or the network. `compose.py` finds the empty square in each piece of artwork
from its pixels, puts the matching screenshot into it, exports the listing sizes and writes
`assets/marketing/asset-manifest.json` (source, operation, size, bytes, hash, date). Every export stays under
2 MB. The screenshots show the panel, not OrcaSlicer's 3D view.

The plugin icon is the listing tile (`assets/orca-cloud/orca_organizer-tile.png`). `assets/icon.png` is the tile at
256x256 and is attached to every release; the Organizer tab's icon is the tile at 40x40 RGBA (white surround transparent), embedded in
`shape_arrange.py` as `ICON_PNG` and written to the storage folder when OrcaSlicer asks for it.
