# Changelog

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
