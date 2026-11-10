# Changelog

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
