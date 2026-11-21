# How Organizer works

What the plugin does with a plate, step by step, and where its limits are. The short version is in the
[README](../README.md); building and testing are in [development.md](development.md).

Organizer is listed in OrcaSlicer's Plugins window and on Orca Cloud as *Shape-Aware Arrange*; its tab in
the main window is called **Organizer**.

## The arrangement

1. Exports a copy of the current project, unsaved changes included.
2. Rasterizes each object's footprint at every candidate rotation.
3. Places parts largest-first. FFT collision search finds the position that
   keeps the layout's bounding box smallest while staying the set spacing
   away from other parts, the bed edge and any excluded areas.
   If parts are left over, or when filling, it also tries a repeating
   pattern for copies of the same part: it finds how tightly two copies nest,
   also as pairs with one copy turned half a turn (when rotation is on),
   tiles the bed with that pattern and fits the rest into the gaps. For the
   most promising rotations the distance between copies is measured again on
   pixels up to 8 times finer than the grid, and the pattern is tried in
   every direction, so rows of small parts reach across the whole bed. The
   result that places more parts wins.
4. Rewrites only the object transforms in the copy, saves it as a new file
   and opens it in OrcaSlicer. Your original project file is never modified,
   and earlier arranged copies are never overwritten.

It works with any model. Collisions are checked on all three axes:

- **X/Y:** each footprint is the whole 3D model projected straight down, so
  an overhang at any height reserves its space. Nothing is placed under
  another part's overhang.
- **Z:** parts keep their height and stay on the bed. They only move in X/Y
  and turn around the vertical axis.
- **Prime tower:** on multi-color plates (two or more filaments, or color
  painting) with the prime tower enabled, the tower's area is kept free,
  brim included, at its position and rotation on that plate. Its depth is
  only known after slicing, so the plugin reserves an estimate: at least as
  deep as it is wide. Check the tower in Orca after slicing.
- **Print by object:** when the plate prints one object at a time, the
  print head drops between finished parts. Parts are then kept at least
  the printer's extruder clearance radius apart in X/Y, even if the spacing
  you set is smaller. Orca checks this clearance between the parts' convex
  hulls, so in this mode each part is arranged as its convex hull and
  concave parts do not interlock. Height and print order are still checked
  by Orca.

The footprint is the model's geometry only: part brims, supports, skirts
and travel moves are not included. The result is an arrangement, not a
print check; slice and review it in Orca before printing.

Orca 2.5 plugins can read the model but cannot move objects, so the plugin
works on a copy and reopens it. Colors, painting, modifiers, per-object
settings and other plates are kept.

## Using it, in detail

1. Select the plate you want to arrange in OrcaSlicer.
2. Click the **Organizer** tab in the main window. It has **Arrange** and
   **Fill** one click away, so you never open the Plugins window or run a
   script. The tab shows the project, the selected plate and its object
   count, and follows your selection. Where the host has no plugin tabs
   (see Known limits), click **Run** on *Shape-Aware Arrange* in
   the **Plugins** window instead (▾ next to **File** on Windows and Linux,
   **File > Plugins** on macOS): it opens the same controls in a panel docked
   on the right.
3. Set the spacing and click **Arrange plate N**. Everything else is
   automatic: every part tries all 360 rotations (1° steps) and keeps the one
   that packs best. All angles are screened on a coarse grid, and the best 8
   are re-checked at full precision. If parts are left over, the plate is retried once on a finer
   0.25 mm grid, which lets tight shapes interlock.
   **Fill plate N** does the same, then keeps adding copies of
   the plate's objects, one object at a time in turn, until nothing more
   fits. The turns also apply when copies of one object are tiled in a
   repeating pattern. Smaller objects keep filling the gaps once bigger ones
   stop fitting. It adds at most 300 copies per run, and the panel lists how
   many copies of each object were added. To add a set number, for example
   12 for an order, enter it in *Most copies to add when filling* under
   **Advanced**; empty fills the plate.
4. Allow the launch prompt the first time; this is how the plugin opens the
   arranged copy. The panel shows how many parts were placed, lists any
   that did not fit (moved into a row above the plate) and always shows
   where the copy was saved, with an **Open this copy in OrcaSlicer** button.

The spacing is the gap kept between parts' top-down outlines. It is never
undercut; because positions sit on a 0.5 mm grid, the real gap is typically
about 1 mm larger, up to about 2 mm along slanted edges. Under
**Advanced**, *Only 90° steps* turns parts only by quarter turns, so parts that
are square to the bed stay square to it (each part may also keep its current
angle), and *Keep each part's orientation* turns rotation off: parts only
move. Both fit fewer parts than automatic rotation.

The panel remembers the spacing, rotation and copy limit of the last run you
started, also after OrcaSlicer restarts (`settings.json` in the plugin's
storage folder).

Each run is saved as `<project>-arranged-<date>-<time>-<id>.3mf` in the
plugin's storage folder, so earlier results stay. The plugin does not delete
them; remove old ones by hand when you no longer need them. A run that fails
keeps its project snapshot in `jobs/<id>/` of that folder for diagnosis; the
plugin keeps the 10 newest of those and removes any older than 30 days when
the next run starts.

**Cancel** stops a run at the next checkpoint; nothing is written or
opened. Only one run works at a time, whichever entry you use: the
Organizer tab and the Script's panel share one run, so a second click
shows the running one instead of starting another. If you switch tabs,
close the panel or reload the page during a run, the run continues; come
back to see its progress. A run that finished while nobody was watching is
saved when you return, and **Open this copy in OrcaSlicer** asks Orca to load it. The tab marks a
result that belongs to another project or plate than the one selected now.

A plate whose parts all fit takes a few seconds. A crowded plate that
overflows takes about 20 seconds (18 snowflakes, 80 mm), because the finer
retry runs too. The work runs in the background, so OrcaSlicer stays usable,
and the panel shows progress.

## Organizer tab and the Script

The plugin registers two entries that drive one run (they share state):

- **Organizer** is a Pages capability: a tab in the main window, built when
  first shown. Orca's Pages API only offers tabs, so the plugin does not add
  a toolbar button or menu item.
- **Shape-Aware Arrange** is the Script (Plugins window > Run). It stays for
  hosts and workflows that use it, and opens the same controls docked.

**Language.** The panel is in English, Brazilian Portuguese (pt-BR), Spanish, German, Simplified Chinese and Polish. It
uses the language OrcaSlicer's UI runs in (`orca.host.app_language()`, from
Orca's Host API) and falls back to the browser's language when the host does
not offer it. Any other language shows English. Part and file names, paths and
the technical details of an unexpected error are shown as they are. A failure
says what went wrong and what to try next, in the panel's language; the raw
error stays as *Details* only when the cause is not one the plugin knows.
"Open this copy in OrcaSlicer" only asks Orca to open the saved file: what Orca
does with its current project or windows was not tested, so the panel does not
say. When the plugin does not answer, the tab says so and offers **Try again**,
which asks the plugin for its state again (the host has no API to reload a tab).

The tab keeps no state: the plugin does. Reloading the tab asks the plugin
for the current run. The plugin never posts to the tab from its worker
thread, because the host does not document that as thread-safe; the tab asks
for progress every second while a run is busy and every three seconds while
idle, and only while it is visible. A host without the Pages capability
loads the plugin without the tab and keeps the Script.

## Command line

The nesting core also runs outside OrcaSlicer on 3MF files in millimetres
(other units are refused):

```sh
uv run --python 3.12 --with numpy --with pillow python shape_arrange.py \
    in.3mf out.3mf --plate 1 --spacing 2 --auto --preview plate.png
```

Add `--fill` to fill the plate with copies, and `--copies N` to add at most N of them.

`--auto` uses the plugin's settings: 1° steps on a 0.5 mm grid, and one
retry on a 0.25 mm grid with 15° steps if parts are left over. Use `--step`
and `--res` instead to choose the rotation step and grid size yourself;
`--step 0` keeps every part's orientation. The CLI writes to the `dst` you
give it, replacing an existing file there.

Pass `--bed 270x270` for plain 3MF files that carry no printer settings.

## Known limits

- Parts are not mirrored or tilted; only rotation about Z is tried.
- Only 3MF files in millimetres are accepted. In the model's `<build>`
  section, element prefixes bound to the 3MF core namespace are supported;
  a layout the plugin cannot edit safely is refused with an error.
- The Organizer tab needs a host with the Pages capability, and the tab
  count is set by the host (Preferences > General > Plugins > Visible
  plugin pages). Orca documents no toolbar, menu or shortcut API for
  plugins, so none is used. How a real host delivers `post_message`, when it
  calls `get_ui()` again after a reload and how it lays out the tab were
  not tested in OrcaSlicer; the plugin recovers by asking for state.
- Six languages exist (en, pt-BR, es, de, zh-CN, pl), and the wording of the last five was not reviewed by a
  native speaker beyond the author. The panel's own colors (buttons, errors)
  were chosen for contrast on the light and dark backgrounds it detects; the
  colors Orca really passes were not measured. The panel is a Vue 3 + PrimeVue app
  (`ui/`); it was exercised in headless Chrome with a mock host bridge, never in
  OrcaSlicer's WebView. Its layout uses CSS container queries and `color-mix()`
  and an embedded Manrope font; how the macOS, Windows and Linux WebViews draw
  them was not checked.
- Disabling or reloading the plugin during a run ends the run: its state
  lives in the plugin process. The project snapshot stays in the plugin's
  storage folder and earlier copies are untouched.
- Saving the arranged copy runs on OrcaSlicer's UI thread, so a very large
  project can pause the window briefly while it is written.
- On Windows the arranged copy may open in a second OrcaSlicer window unless
  single-instance mode is on. If no launcher is found, the panel shows the
  file path to open by hand.
