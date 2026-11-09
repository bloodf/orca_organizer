# Shape-Aware Arrange for OrcaSlicer

OrcaSlicer's built-in Arrange treats every object as its convex hull, so
concave parts like snowflakes, stars, combs and letters end up spaced as if
they were discs or boxes. This plugin arranges a plate using each object's
real top-down footprint, so arms and teeth interlock and more parts fit.

Requires OrcaSlicer 2.5 or later, which has the Python plugin system.

## How it works

1. Exports a copy of the current project, unsaved changes included.
2. Rasterizes each object's footprint at every candidate rotation.
3. Places parts largest-first. FFT collision search finds the position that
   keeps the layout's bounding box smallest while staying the set spacing
   away from other parts, the bed edge and any excluded areas.
4. Rewrites only the object transforms in the copy, then opens it in
   OrcaSlicer. Your original project file is never modified.

Orca 2.5 plugins can read the model but cannot move objects, so the plugin
works on a copy and reopens it. Colors, painting, modifiers, per-object
settings and other plates are kept.

## Install

1. In OrcaSlicer, open **File > Plugins**.
2. Click the arrow next to **Browse plugins**, choose **Install local
   plugin**, and select `shape_arrange.py`.
3. Tick **Activate** and allow the thread permission prompt (the arranging
   runs in the background so OrcaSlicer stays responsive). On first install,
   OrcaSlicer installs `numpy` and `pillow` for the plugin.

## Use

1. Select the plate you want to arrange in OrcaSlicer.
2. Open **File > Plugins** and click **Run** on *Shape-Aware Arrange*. A
   panel docks on the right.
3. Set the spacing and click **Arrange current plate**. Everything else is
   automatic: every part tries all 360 rotations (1° steps) and keeps the one
   that packs best. All angles are screened on a coarse grid, and the best 8
   are re-checked at full precision. If parts are left over, the plate is retried once on a finer
   0.25 mm grid, which lets tight shapes interlock.
4. Allow the launch prompt the first time; this is how the plugin opens the
   arranged copy. The panel shows how many parts were placed and lists any
   that did not fit; those are moved into a row above the plate.

A plate whose parts all fit takes a few seconds. A crowded plate that
overflows takes about 20 seconds (18 snowflakes, 80 mm), because the finer
retry runs too. The work runs in the background, so OrcaSlicer stays usable,
and the panel shows progress.

## Command line

The nesting core also runs outside OrcaSlicer on any 3MF:

```sh
uv run --python 3.12 --with numpy --with pillow python shape_arrange.py \
    in.3mf out.3mf --plate 1 --spacing 2 --auto --preview plate.png
```

`--auto` uses the plugin's settings. Use `--step` and `--res` instead to
choose the rotation step and grid size yourself.

Pass `--bed 270x270` for plain 3MF files that carry no printer settings.

## Test

```sh
uv run --python 3.12 --with numpy --with pillow python tests/test_shape_arrange.py
uv run --python 3.12 --with numpy --with pillow python tests/test_plugin.py
```

`test_plugin.py` runs the plugin against a stub `orca` module and checks the
threading: the export and the file launch happen on the UI thread, the
nesting does not.

`test_shape_arrange.py` builds a two-plate Orca-style project in which two combs fit on
plate 2 only if their teeth interlock. It arranges the project, then checks
that every part sits on the bed with no overlap and that plate 1 and every
other file in the 3MF are unchanged.

## Known limits

- Parts are not mirrored or tilted; only rotation about Z is tried.
- On Windows the arranged copy may open in a second OrcaSlicer window unless
  single-instance mode is on. If no launcher is found, the panel shows the
  file path to open by hand.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
