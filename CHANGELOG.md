# Changelog

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
