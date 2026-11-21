# Changelog

## 0.16.1 (2026-11-21)

- Shared Orca UI foundation 5.6.2, with its focus and drag fixes for shared parts.

### Fixed

- Pressing **Cancel** right after **Arrange** or **Fill**, before the panel showed the first progress line, did nothing: the run went on and finished. The cancel is now sent as soon as OrcaSlicer names the run.
- When the panel asked for its state before OrcaSlicer had told the plugin where its storage folder is, the saved spacing, rotation and copy limit were ignored until OrcaSlicer restarted. They are now read as soon as the folder is known.

## 0.16.0 (2026-11-21)

- Shared Orca UI foundation 5.6.1 (Vue 3.5.46, lucide-static 1.77.0). The page also stops shipping the styles of shared parts it never shows, so the plugin file is smaller.

### Fixed

- The panel header said "Plate 1 · 3 objects" when only one of the project's three objects was on plate 1. OrcaSlicer does not tell plugins which objects are on a plate, so the header now says "3 objects in the project", like the line under it.
- After **Arrange**, the panel said it asked OrcaSlicer to open the copy even when the open project had unsaved changes, where OrcaSlicer may keep the current project. It now warns about that, in every language.
- Pasting "1,5" as the spacing with OrcaSlicer in English gave 15 mm. A single comma is now read as the decimal sign in every language, so it gives 1.5 mm.

## 0.15.1 (2026-11-19)

- The panel page ships inside the plugin file as plain text instead of a compressed, base64-encoded block, so the whole file reads as what it is. Nothing changes in OrcaSlicer.

## 0.15.0 (2026-11-19)

- New look on the shared Orca UI foundation 5.0.2: the page uses OrcaSlicer's own greys and its teal for **Arrange**, progress and focus, in the tab and in the dock. Organizer's blue is now only on its mark. Arranging and filling work as before.
- What **Rotation** and **Most copies to add when filling** do is behind an (i) beside each label instead of a paragraph under it. When a requirement is missing, the reason Arrange and Fill are off is a notice at the top of the page. While a run is busy the job block says so; the line under the form is gone.
- In the dock, the button that goes back to the form after a result stays on the title bar's line.
- The README and Orca Cloud pictures show the new look, in English and Brazilian Portuguese.

## 0.14.2 (2026-11-17)

- Polish (pl) translation. Thanks to @xtronets for asking (u1control#20). Organizer follows OrcaSlicer when it runs in Polish (Polski, `pl_PL`): every button, tooltip, setting, progress line, result and error is in Polish, numbers are written the Polish way (2,5 mm) and counts use Polish's three plural forms (1 obiekt, 2 obiekty, 5 obiektów).

## 0.14.1 (2026-11-17)

- Notifications appear bottom-left and each one can be closed; warnings and notices on the page can be dismissed. "Path copied" (or how to copy it by hand) is now one of those notifications instead of a line in the result card. A closed notice stays closed while the same thing is true and comes back when it changes: a new run, another problem with the project, the plugin going silent again.
- A closed notice also comes back when its problem goes away and later returns, and a start that fails again shows its error again. Notifications wait while OrcaSlicer is in another tab or app; errors stay until closed.
- The page fills the whole tab at every height, also after the window is resized: no more grey under a short page.
- The plugin loads even when OrcaSlicer could not install numpy or Pillow (offline, or a pip error during install). It used to fail to load with no word of why. Now **Arrange** and **Fill** are turned off with one line saying what is missing, and a new **Requirements** section under the form lists numpy and Pillow: found (with the version) or not found (with the error), what each is used for, how to get it back (reinstall the plugin from Orca Cloud so OrcaSlicer installs its packages again, then restart OrcaSlicer) and a link to its installation guide. **Check again** looks once more without a restart. The section is closed while everything is found.
- Changed: the plugin icon is the new brand tile. The Organizer tab in OrcaSlicer shows it (the tab had no icon before), and it is the `icon.png` attached to each release.

## 0.14.0 (2026-11-16)

- Organizer now speaks Spanish, German and Simplified Chinese, next to English and Brazilian Portuguese. It follows the language OrcaSlicer runs in (Español, Deutsch, 简体中文); Traditional Chinese shows Simplified, and any other language shows English. There is no language setting in the plugin: change OrcaSlicer's language and the tab and the dock follow.
- Everything the panel says is translated: the buttons and their tooltips, the settings under **Advanced**, **How it works**, the progress of a run, the result and its fine print, every error with its next step, "no answer from the plugin", and what a screen reader announces. The Script's own answer, "Panel opened", is in OrcaSlicer's language too (also in Portuguese now).
- The words are OrcaSlicer's own: *cama*, *Platte* and 盘 for the plate, *torre de purga*, *Reinigungsturm* and 擦拭塔 for the prime tower, *Por objeto*, *Nach Objekt* and 逐件 for the print sequence. German addresses you formally (Sie).
- Numbers are written as the language writes them: a spacing of 2,5 mm in Spanish and German, 2.5 mm in Chinese. When you type or paste the spacing, a point is always read as the decimal sign, so 2.5 and 2,5 both mean two and a half millimetres in Spanish and German (and in Portuguese, where a typed point used to be dropped). Where a language is written differently by country, the system's region decides (2.5 in Mexico, 1’234.5 in Switzerland). Counts use each language's plural ("1 objeto", "2 objetos"; Chinese has none), and Chinese uses full-width punctuation.
- Chinese text is drawn with the system's Chinese font, its quotation marks and ellipsis included (they were narrow Latin marks); the font built into the page has Latin letters only.
- With the dock zoomed to 200 %, "Try again" under "No answer from the plugin" wraps onto two lines instead of running off the edge (it did in Portuguese and German).
- The Spanish, German and Chinese wording was written without a native speaker's review. Corrections are welcome as issues.

## 0.13.0 (2026-11-16)

- Fill packs small parts across the whole bed. Round parts used to stop a row short: 24 mm parts on a 220 mm bed filled 8 rows from the front, left a band at the back and dropped one part alone in a corner (61 parts); now 9 full rows fit (68). The distance between copies in the repeating pattern was measured on the 0.5 mm grid, which adds up to about 1 mm to every step between parts, so on a full bed a row or column was lost. It is now measured on pixels up to 8 times finer, and the pattern is tried in every direction, not only the densest few on paper. The spacing you set is still never undercut, also from the bed edge.
- More parts per Fill (2 mm spacing): round parts on a 220 mm bed 22 mm 72 to 85, 15 mm 141 to 168, 28 mm 46 to 52; on a 270 mm bed 22 mm 111 to 126; 20 mm squares on a 256 mm bed 113 to 118, hexagons 156 to 168. L-brackets 47 to 48. With rotation off: U-channels 35 to 42, right triangles 34 to 37, L-brackets 45 to 46. T-parts, triangles and rings with rotation on stay as they were. Arrange uses the same pattern when parts are left over: 68 round 24 mm parts on a 220 mm bed all fit now (64 before).
- Fill takes about 1 to 3 seconds longer on a typical plate, up to about 10 seconds more for very thin parts or a bed of 400 mm. When the usual pattern already holds every copy asked for, the extra step is skipped.
- Checked again for "parts overlap with instances": that was fixed in 0.9.5 (released the day after the report). New tests build objects whose instances are each mirrored, scaled or turned on their own, stacked on each other, spread over two plates and missing from the plate list; Arrange and Fill leave no two parts overlapping, keep each instance's mirror and scale, and OrcaSlicer reopens the result with every part where it was placed.

## 0.12.0 (2026-11-15)

- New look for the Organizer tab and the Script's panel, on the shared Orca UI foundation 4.0.1: a sky-blue bar with the new Organizer mark, the project, plate and object count, and icons on every action. Arranging and filling work as before.
- The docked panel (about 280 x 360) shows one thing at a time: the form, the running job, or the result. **Arrange** and **Fill** both fit without scrolling, and during a run the progress and **Cancel** sit right under the bar instead of below the fold.
- A run shows as a card with its phase, the project, the part being placed and "7 of 14" (or "+5" copies while filling), with **Cancel** under it.
- The result says what happened first (green), warns about parts that did not fit (amber), and then shows the saved copy: the file name, **Open this copy in OrcaSlicer**, **Copy path**, the full path and the fine print of the run. **New arrangement** goes back to the form in the dock.
- In a tab, the form sits in its own card on the left and the job or result on the right; before the first run, an illustration says what will appear there.
- Spacing has the unit in the field and +/- buttons. Rotation and the copy limit stay under **Advanced**; what each control does is under **How it works** and in each control's tooltip.
- Errors, "no answer from the plugin" and "the run below is for another plate" are banners with their own icon and colour.
- New plugin icon with the Organizer mark.
- The plugin file now carries the licence notices of the code and font bundled in its page (Vue, PrimeVue, Lucide icons, Manrope).

## 0.11.0 (2026-11-14)

- New under **Advanced**: *Only 90° steps* rotation. Parts turn only by quarter turns, so parts that are square to the bed stay square to it.
- New under **Advanced**: *Most copies to add when filling*. Fill stops after that many copies (an order of 12, for example) or when the plate is full; empty fills the plate as before. The CLI takes `--copies N`.
- The panel remembers the spacing, rotation and copy limit of the last run, also after OrcaSlicer restarts.
- Parts that do not fit are parked clear of parts parked before (by an earlier run, or from another plate in the same column) instead of on top of them.
- Snapshots of failed runs (`jobs/<id>/` in the plugin's storage folder) are pruned when a run starts: the 10 newest stay, and none older than 30 days.
- A bad rotation step or grid size (CLI or API) gets its own explanation instead of "Something unexpected went wrong".
- Print by object: parts are arranged as their convex hulls, because Orca checks the extruder clearance between convex hulls. Concave parts no longer nest into each other's notches and then fail Orca's "too close" check when slicing.
- A spacing of half the bed's smaller side or more fails at once with a clear message ("A spacing of 2000 mm leaves no room on this bed. Enter a spacing below 128 mm"). Before, nothing could fit and the run worked for minutes. When the plate prints by object and the printer's extruder clearance radius is what set that spacing, the message names the clearance radius instead.
- An invalid printer or prime tower setting in the project (for example a layer height of 0) is reported with the setting's name instead of "Spacing must be a number".
- Arranging plate 2 or later of a plain 3MF without a plate list is refused; Orca puts all of its objects on plate 1, and the parts used to be moved off every plate.
- CLI: `--preview` closes the arranged file after drawing it.

## 0.10.0 (2026-11-13)

- Interface moved to PrimeVue 5 with the shared Orca UI foundation 3.0.2 (Aura in its 4.x sizes, same palette and layout). No change to arranging or filling, and no visible change intended. PrimeVue 5 is under the PrimeUI Community License; the key ships in the shared foundation and is checked offline.
- Needs a newer system WebView, because PrimeVue 5 colours use CSS `light-dark()`: macOS WebKit 17.5+ (Safari 17.5 / macOS 14.5), Windows WebView2, Linux WebKitGTK 2.44+.

## 0.9.5 (2026-11-13)

- Fix parts overlapping on the plate when an object has several instances. The plate's parts were read from the plate list OrcaSlicer saves in the project, which goes stale once instances are added or removed: instances sitting on the plate could be left out, so they were neither arranged nor kept clear of, and instances on another plate could be pulled in. Arrange and Fill now take every part whose position is on the selected plate, the same way OrcaSlicer assigns parts to plates when it opens the result.

## 0.9.4 (2026-11-12)

- Fix "Could not start: plugin.storage() must be called from a plugin callback" when clicking Arrange or Fill. OrcaSlicer 2.5 answers `plugin.storage()` only inside plugin callbacks, not from a panel or tab message; the folder is now read while loading (and when the tab or panel opens) and kept. The test host now refuses storage() outside callbacks.

## 0.9.3 (2026-11-12)

- Fix OrcaSlicer crashing on start with Organizer 0.9.0–0.9.2 installed. The Organizer tab called the host's base `on_load`/`on_unload`; in OrcaSlicer those are C++ virtuals that dispatch back to the plugin's own method, so the call recursed until the stack overflowed (SIGBUS). The tab no longer calls the base hooks, and the test host now behaves like the real one.

## 0.9.2 (2026-11-12)

- Build tools only: shared Orca UI foundation 2.0.3 (Vite 8.3.8, @vitejs/plugin-vue 6.0.11). No change to arranging, filling or the interface.

## 0.9.1 (2026-11-12)

- Fix Orca Cloud publishing of release notes containing semicolons: send metadata as a literal form string so curl does not truncate the JSON. Version 0.9.0 remains available on GitHub; its Orca Cloud upload was rejected before publication.
- Includes the packing improvements and Vue + PrimeVue interface from 0.9.0 without changing their behavior.

## 0.9.0 (2026-11-12)

- Tighter, still safe spacing. The margin around each footprint is now derived instead of stacked: the outline covers the part at every rotation by construction, and the spacing is added exactly once. The requested gap is never undercut; the extra gap left by the 0.5 mm grid drops from about 1.8 mm to about 1 mm at the median (synthetic exact-geometry checks), up to about 2 mm along slanted edges. Fill fits more: 20 mm squares 99 to 113, L-brackets 44 to 47, T-parts 43 to 52 on a 256 mm bed (2 mm spacing).
- A part that fits only at one odd angle in a narrow gap is no longer left over: when the quick coarse check finds no place, every rotation is tried at full precision, not one in 15. A 60 x 15 mm bar in a 22.5 mm corridor at 37 degrees now fits.
- Fill tiles asymmetric parts in pairs, one turned half a turn, when that fits more: right triangles 35 to 51, U-channels 37 to 50, combs 20 to 22 on a 256 mm bed. With rotation turned off, nothing is turned.
- Small parts still go into rings and frames: among equally compact positions, a free hole inside a placed part wins, and when an identical part comes next, the spot that leaves it the most room in the holes (4 washers with 8 small discs: all 8 inside, two per washer, as before).
- When parts are left over, the arrangement is retried once with those parts first, so interlocking parts no longer depend on which one happens to go first.
- Large spacings (print by object, 35 mm extruder clearance) prepare outlines about 35 to 80 times faster per rotation (synthetic CPU timing); the masks are identical.
- Long inner loops (hundreds of rotations, repeating patterns) reach a cancellation checkpoint about every half second.
- Costs: Fill of asymmetric parts takes about 2 to 4 times longer (pairs are tried), and a part that does not fit costs one full check of every rotation.
- Unchanged on purpose: the box-area placement score (a diagonal score made rows into blocks but pushed small parts out of rings, 8 to 3 in the washer test), and the bed-edge rule in print-by-object mode (the clearance radius still applies to the bed edge).

- Keep keyboard focus on Cancel while cancellation is pending; repeated activation sends no extra cancellation requests.

- The panel (Organizer tab and the Script's dock) is rebuilt as a **Vue 3 + PrimeVue 4** app with the shared Orca visual system (one palette, light and dark, pt-BR/en). Same behaviour, protocol and run: Arrange and Fill, spacing, rotation under Advanced, progress, Cancel, the saved path with **Open this copy in OrcaSlicer** and **Copy path**, stale-run note, offline note with **Try again**. Sources are in `ui/`; the compiled page is embedded in `shape_arrange.py`, so the plugin is still one file and the page needs no network. Size: the page grows from 26 KB of hand-written HTML to 411 KB compiled (95 KB gzipped); `shape_arrange.py` grows from about 101 KB to about 215 KB because the page is stored compressed inside it.
- Cancel shows a pending state and stays pending until the run ends (a poll no longer re-enables it). A disabled Arrange/Fill says why. The spacing error appears next to the field. The progress bar and the copy count are real PrimeVue controls; the status uses a polite live region and errors an alert.

- The panel speaks **English and Brazilian Portuguese**, chosen from the host's UI language (`orca.host.app_language()`) with the browser's as fallback. The title is **Organizer** everywhere, with *Shape-Aware Arrange* as its subtitle.
- Failures say what went wrong and what to do next ("Plate 2 has no printable objects. Next: select another plate…") instead of `error: <exception>`; the raw exception is kept as *Details* only for causes the plugin does not know. The plugin sends structured messages (`msg`) next to the English `text`, so older clients keep working.
- The progress bar is determinate where the plugin has a real total (measuring outlines, placing parts, repeating patterns) and indeterminate otherwise; filling shows a copy count. The screen-reader text changes with the phase, not with every part.
- Plurals are correct ("1 object", "2 objects"). When the plugin does not answer, the tab offers **Try again** instead of telling you to reload. **Open this copy in OrcaSlicer** no longer implies what Orca does with the current project; **Copy path** appears when the WebView allows the clipboard and selects the path otherwise.
- Own, contrast-checked button and error colors for light and dark; the error and note boxes no longer use a colored side stripe. Reduced-motion and forced-colors rules.

- New **Organizer** tab (Orca's Pages capability): **Arrange plate N** and **Fill plate N** are one click away in the main window, without opening the Plugins window or running the Script. It shows the project, plate and object count, explains the spacing, keeps rotation off under **Advanced**, shows progress with **Cancel**, and ends with the saved path and **Open this copy in OrcaSlicer**. A result that belongs to another project or plate is marked. Hosts without Pages keep the Script.
- The tab and the Script's panel share one run and one set of files: a click in either while a run is busy shows that run, and stale ids are ignored. The tab asks for state, so a reloaded tab recovers it.
- The Script's panel got the same layout and labels.
- Keyboard focus moves to **Cancel** when a run starts and to **Open this copy in OrcaSlicer** when it ends. Errors are marked with text, not only color.
- Tests: Pages registration and fallback, shared run, reload, unload and stale ids; panel states checked by Vue component tests (`ui/tests`, run from `tests/test_panel_js.py`).

- Rotated prime towers are reserved where Orca puts them: the tower turns about its own corner at the set position, like Orca, instead of about its center. The panel calls the reserved area an estimate.
- One run at a time, with its own project snapshot and a new output file each time (`<project>-arranged-<date>-<time>-<id>.3mf`). Earlier arranged copies are never overwritten. The panel always shows where the copy was saved and has an **Open this copy in OrcaSlicer** button.
- New **Cancel** button: nothing is written or opened after cancelling.
- Closing the panel no longer loses a run: reopening it shows the progress, and a run that finished meanwhile is saved.
- Fill takes turns between objects also when it tiles copies in a repeating pattern, and lists the copies added per object. Two different objects of the same size now get equal shares, instead of nearly all copies going to one.
- 3MF files in units other than millimetres are refused instead of being placed at the wrong scale. Build sections written with a namespace prefix are supported.
- With rotation turned off (`step` 0), the finer retry no longer rotates parts.
- Invalid settings (non-finite numbers, grid size 0 or less, negative spacing, plate below 1, empty bed) are rejected with a clear error before any work. `enable_prime_tower` accepts JSON `true`/`false` as well as `1`/`0`.
- The status area is announced to screen readers, and the panel shows the project and selected plate.

## 0.8.0

- Plates with many copies of the same part try a repeating pattern: the plugin finds how tightly two copies nest and tiles the bed with that pattern, then fits the remaining parts into the gaps. It is used when it places more parts than the normal arrangement. 20 L-brackets (100 x 75 mm) on a 256 mm bed: 18 fit, up from 15. Fill plate with copies uses it too (18 brackets, up from 14).
- Identical copies share one footprint, so plates with many copies prepare faster.

## 0.7.1

- Multi-color plates keep the prime tower's area free (two or more filaments, or color painting, with the prime tower enabled), brim included.
- Fill works on projects whose settings list no plates.
- Releases now publish to Orca Cloud automatically.

## 0.7.0

- Multi-color plates keep the prime tower's area free. The tower is reserved when the prime tower is enabled and the plate prints with two or more filaments (different filament slots or color painting), at its position on that plate, with its brim and an estimated depth.
- Fill works on projects whose settings list no plates.

## 0.6.0

- New **Fill plate with copies** button: arranges the plate, then adds copies of its objects (round-robin) until no more fit, so the whole bed is used.
- Orca Cloud metadata: plugin type, OS-independent file name for releases.

## 0.5.0

- Plates printed one object at a time ("By object") keep the printer's extruder clearance radius between parts.
- Footprints cover the whole height of each part, so nothing is placed under another part's overhang.

## 0.4.0

- Arranging runs in the background, so OrcaSlicer stays responsive. The panel shows progress.

## 0.3.0

- Every part tries all 360 rotations (1° steps).

## 0.2.0

- Automatic mode: spacing is the only setting. The selected plate is arranged, and a finer second pass runs when parts are left over.

## 0.1.0

- First release: arranges a plate by each part's real footprint, so concave parts interlock.
