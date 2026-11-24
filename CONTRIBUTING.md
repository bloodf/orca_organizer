# Contributing

Bug reports and pull requests are welcome at <https://github.com/bloodf/orca_organizer>.

For a bug, say which OrcaSlicer version and operating system you use, what you clicked and what the panel
showed. If a run failed, the project snapshot in `jobs/<id>/` of the plugin's storage folder helps to reproduce it.

## Layout

The plugin is one file, `shape_arrange.py`: the nesting core, the OrcaSlicer entries and the embedded panel.
The panel's source is a Vue 3 + PrimeVue app in `ui/`, compiled into one offline HTML file and embedded in
`shape_arrange.py`. Do not edit the embedded block by hand.

## Build and test

```sh
cd ui && npm ci --ignore-scripts && npm run build && cd ..
uv run --python 3.12 --with numpy --with pillow python tests/test_shape_arrange.py
uv run --python 3.12 --with numpy --with pillow python tests/test_packing.py
uv run --python 3.12 --with numpy --with pillow python tests/test_plugin.py
uv run --python 3.12 --with numpy --with pillow python tests/test_panel_js.py
```

CI runs the same checks and fails when `ui/src` was changed without rebuilding. Python-only changes need no
Node, apart from `tests/test_panel_js.py`.

## Before you open a pull request

- Keep behaviour changes covered by a test in `tests/`.
- When wording changes in Python, update `ui/src/messages.ts` in English and pt-BR; the parity test checks the English.
- A change to `shape_arrange.py` or the panel needs a version bump and a `CHANGELOG.md` entry. Docs and
  listing images do not.
- Commit messages follow Conventional Commits (`fix(packing): ...`, `docs: ...`).

## More

- [docs/how-it-works.md](docs/how-it-works.md): the arrangement, the two entries, the command line, known limits
- [docs/development.md](docs/development.md): panel source, tests, listing and README images
- [docs/release.md](docs/release.md): releasing and the Orca Cloud listing

The code is GPL-3.0-or-later; contributions are accepted under the same license.

## Privacy

Keep the owner's private setup out of source, tests, fixtures, docs, commits,
branch names, issues, pull requests and comments. Do not include machine names,
home or worktree paths, private network details, service ports or hostnames,
thread and task IDs, personal email or provider accounts, quota or credit data,
device identifiers, Wi-Fi names, or secrets.

Use RFC 5737 addresses such as `192.0.2.10`, example domains, `printer-1`,
`<serial>`, and `$HOME/...`. Keep hardware findings and redact the identifying
details. Run `sh scripts/check-privacy.sh` before submitting changes.
