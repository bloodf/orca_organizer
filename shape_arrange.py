# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.9.0"
# ///
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Heitor Ramon
"""Shape-aware plate arrangement for OrcaSlicer.

Orca 2.5 plugins cannot move objects, so the plugin exports a 3MF copy of the
project, rewrites the build-item transforms in it and reopens it. The nesting
core works on any 3MF and is also usable from the command line:

    python shape_arrange.py in.3mf out.3mf --plate 1 --spacing 2 --step 30
"""

from __future__ import annotations

import json
import math
import numbers
import re
import sys
import threading
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:  # only present inside OrcaSlicer
    import orca
except ImportError:
    orca = None

CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
PROD = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}"
PLATE_GAP = 0.2  # Orca/Bambu LOGICAL_PART_PLATE_GAP: plates are laid out at 1.2x bed size
DEFAULT_BED = [(0.0, 0.0), (256.0, 0.0), (256.0, 256.0), (0.0, 256.0)]


@dataclass
class Params:
    plate: int = 1          # 1-based plate to arrange
    spacing: float = 2.0    # min gap between parts, mm
    step: float = 30.0      # rotation step, degrees; 0 disables rotation
    res: float = 0.5        # raster cell size, mm
    center: bool = True     # center the finished layout on the plate
    refine: bool = False    # parts left over: retry once on a 2x finer grid, keep the better result
    fill: bool = False      # after arranging, add copies of the plate's objects until nothing fits


class PlanError(ValueError):
    """A ValueError the page can explain in the user's language. `code` and `args` are what the page
    translates; the English message stays the one logged, printed by the CLI and shown by old clients."""

    def __init__(self, message: str, code: str, **args):
        super().__init__(message)
        self.code, self.args_ = code, args


class Phase(str):
    """Progress message that also says, in structured form, which phase it is. It is still the same
    string for the CLI and for any caller that prints it; the panel reads `code` and `subject`
    instead of parsing the wording. `counted` is False when `n` is not a stable total."""

    def __new__(cls, text: str, code: str, subject: str = "", counted: bool = True):
        self = super().__new__(cls, text)
        self.code, self.subject, self.counted = code, subject, counted
        return self


def _number(value, name: str, minimum: float = 0.0, strict: bool = False) -> float:
    """`value` as a finite float >= minimum (> minimum when strict), else ValueError."""
    try:
        v = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v) or v < minimum or (strict and v == minimum):
        bound = f" {'>' if strict else '>='} {minimum:g}" if math.isfinite(minimum) else ""
        raise PlanError(f"{name} must be a finite number{bound}, got {value!r}", "bad_number", name=name,
                        minimum=minimum, strict=strict)
    return v


def _flag(value, name: str) -> bool:
    """Orca config boolean: "1"/"0", "true"/"false", JSON true/false or 0/1 (a one-element list too)."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("1", "true", "0", "false", ""):
        return value.strip().lower() in ("1", "true")
    raise ValueError(f"{name}: unsupported value {value!r}, expected 0/1 or true/false")


def check_params(p: Params) -> Params:
    """Reject parameters the nesting cannot honour, before any file is read."""
    if isinstance(p.plate, bool) or not isinstance(p.plate, numbers.Integral) or p.plate < 1:
        raise ValueError(f"plate must be a whole number >= 1, got {p.plate!r}")
    _number(p.spacing, "spacing")
    _number(p.step, "step")
    _number(p.res, "res", strict=True)
    for name in ("center", "refine", "fill"):
        if not isinstance(getattr(p, name), bool):
            raise ValueError(f"{name} must be True or False, got {getattr(p, name)!r}")
    return p


def check_bed(poly) -> list[tuple[float, float]]:
    """Printable polygon as finite (x, y) mm points with a positive width and depth."""
    pts = [(_number(x, "bed x", -math.inf), _number(y, "bed y", -math.inf)) for x, y in poly]
    xs, ys = zip(*pts) if pts else ((), ())
    if len(pts) < 3 or max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
        raise ValueError(f"bed must be a polygon with positive width and depth, got {poly!r}")
    return pts


@dataclass
class Item:
    index: int                      # position among <build><item> tags
    object_id: str
    transform: np.ndarray           # 4x4, column-vector convention
    parts: list = field(default_factory=list)  # [(verts_world (N,3), tris (M,3))]
    name: str = ""
    printable: bool = True
    filaments: set = field(default_factory=set)  # 1-based filament slots its parts print with
    painted: bool = False                        # has multi-color painting


# ---------------------------------------------------------------- 3MF reading

def _mat_from_3mf(s: str | None) -> np.ndarray:
    m = np.eye(4)
    if s:
        a = [float(v) for v in s.split()]
        m[:3, :3] = np.array(a[:9]).reshape(3, 3).T
        m[:3, 3] = a[9:12]
    return m


def _mat_to_3mf(m: np.ndarray) -> str:
    a = list(m[:3, :3].T.reshape(-1)) + list(m[:3, 3])
    return " ".join(f"{v:.9g}" for v in a)


TRANSFORM_RE = r"""(?<=\s)transform\s*=\s*(["']).*?\1"""
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf"""\s{name}\s*=\s*(["'])(.*?)\1""", tag)
    return m.group(2) if m else None


def _prefixes(text: str, uri: str) -> list[str]:
    """XML prefixes bound to namespace `uri` in `text` ("" = default namespace), in order of appearance."""
    out = []
    for m in re.finditer(r"""\sxmlns(?::([\w.-]+))?\s*=\s*(["'])(.*?)\2""", text):
        if m.group(3) == uri and (m.group(1) or "") not in out:
            out.append(m.group(1) or "")
    return out


SUPPORTED_UNIT = "millimeter"


def _mesh_arrays(mesh: ET.Element):
    verts = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
                      for v in mesh.iter(CORE + "vertex")], dtype=float).reshape(-1, 3)
    tris = np.array([[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))]
                     for t in mesh.iter(CORE + "triangle")], dtype=np.int64).reshape(-1, 3)
    return verts, tris


class Package:
    """Read-only view of a 3MF zip plus the bits needed to rewrite it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        try:
            self.names = set(self.zip.namelist())
            rels = self._text("_rels/.rels") or ""
            m = re.search(r'Target="/?([^"]+\.model)"', rels)
            self.main_name = m.group(1) if m else "3D/3dmodel.model"
            self.main_text = self._text(self.main_name)
            if self.main_text is None:
                raise PlanError(f"{self.path.name}: no 3D model part found", "no_model", file=self.path.name)
            self._trees: dict[str, ET.Element] = {}
            self._spans = None
            self._tree(self.main_name)  # checks the unit before anything else is read
            self.settings = self._load_settings()
        except BaseException:
            self.zip.close()
            raise

    def close(self):
        self.zip.close()

    def _text(self, name: str) -> str | None:
        return self.zip.read(name).decode("utf-8") if name in self.names else None

    def _tree(self, name: str) -> ET.Element:
        if name not in self._trees:
            root = ET.fromstring(self.zip.read(name))
            unit = root.get("unit", SUPPORTED_UNIT)
            if unit != SUPPORTED_UNIT:  # coordinates would be read as mm: refuse instead of misplacing
                raise PlanError(f"{name}: 3MF unit '{unit}' is not supported; only {SUPPORTED_UNIT} files "
                                "can be arranged", "bad_unit", unit=unit)
            self._trees[name] = root
        return self._trees[name]

    def _object(self, file: str, oid: str) -> ET.Element:
        for obj in self._tree(file).iter(CORE + "object"):
            if obj.get("id") == oid:
                return obj
        raise ValueError(f"object {oid} not found in {file}")

    def _load_settings(self) -> dict:
        """Part subtypes, object names and plate membership from Orca/Bambu metadata."""
        out = {"subtypes": {}, "names": {}, "plates": {}, "plate_meta": {}, "filaments": {}}
        text = self._text("Metadata/model_settings.config")
        if not text:
            return out
        root = ET.fromstring(text)
        for obj in root.iter("object"):
            oid = obj.get("id")
            meta = {md.get("key"): md.get("value", "") for md in obj.findall("metadata")}
            if "name" in meta:
                out["names"][oid] = meta["name"]
            out["subtypes"][oid] = {p.get("id"): p.get("subtype", "normal_part")
                                    for p in obj.findall("part")}
            default = meta.get("extruder") or "1"
            out["filaments"][oid] = {
                p.get("id"): next((m.get("value") for m in p.findall("metadata") if m.get("key") == "extruder"),
                                  None) or default
                for p in obj.findall("part")} | {None: default}
        for plate in root.iter("plate"):
            meta = {m.get("key"): m.get("value") for m in plate.findall("metadata")}
            pid = int(meta.get("plater_id", len(out["plates"]) + 1))
            members = set()
            for mi in plate.findall("model_instance"):
                kv = {m.get("key"): m.get("value") for m in mi.findall("metadata")}
                members.add((kv.get("object_id"), int(kv.get("instance_id", 0))))
            out["plates"][pid] = members
            out["plate_meta"][pid] = meta
        return out

    def config(self) -> dict:
        text = self._text("Metadata/project_settings.config")
        return json.loads(text) if text else {}

    def clearance(self, plate: int) -> float:
        """Extra gap needed when the plate prints one object at a time ("By object").

        The nozzle then lowers between finished parts, so parts must sit at least
        the extruder clearance radius apart. 0 for the usual layer-by-layer printing.
        """
        cfg = self.config()
        seq = self.settings["plate_meta"].get(plate, {}).get("print_sequence") or cfg.get("print_sequence", "")
        if str(seq).strip().lower() != "by object":
            return 0.0
        value = cfg.get("extruder_clearance_radius") or cfg.get("extruder_clearance_max_radius") or 0
        if isinstance(value, list):
            value = max(float(v) for v in value) if value else 0
        return float(value)

    def prime_tower(self, plate: int, items: list[Item]) -> np.ndarray | None:
        """Prime tower footprint on `plate` as plate-local triangles (K, 3, 2), or None.

        Orca only builds the tower when it is enabled and the plate prints with 2+
        filaments (different filament slots or color painting). Its real depth is
        only known after slicing, so this reserves an estimate.
        """
        cfg = self.config()
        if not _flag(cfg.get("enable_prime_tower"), "enable_prime_tower"):
            return None
        printing = [it for it in items if it.printable and it.parts]
        slots = set().union(*(it.filaments for it in printing)) if printing else set()
        if len(slots) < 2 and not any(it.painted for it in printing):
            return None

        def per_plate(key):
            v = cfg.get(key)
            v = v if isinstance(v, list) else [v]
            v = [x for x in v if x not in (None, "")]
            return _number(v[plate - 1] if len(v) >= plate else v[0], key, -math.inf) if v else None

        x, y = per_plate("wipe_tower_x"), per_plate("wipe_tower_y")
        if x is None or y is None:
            return None
        width = _number(cfg.get("prime_tower_width") or 35, "prime_tower_width", strict=True)
        brim = _number(cfg.get("prime_tower_brim_width") or 3, "prime_tower_brim_width")
        # ponytail: depth estimate = purge per tool change spread over the tower width,
        # at least square. Orca's sliced depth can differ; spacing + brim cover small misses.
        volume = cfg.get("filament_prime_volume") or cfg.get("prime_volume") or 45
        volume = max(_number(v, "prime volume") for v in volume) if isinstance(volume, list) else \
            _number(volume, "prime volume")
        layer = _number(cfg.get("layer_height") or 0.2, "layer_height", strict=True)
        changes = max(1, len(slots) - 1)
        depth = max(width, changes * volume / (layer * width))
        # Orca (Print::first_layer_wipe_tower_corners): local corners with the brim, rotated
        # about the tower's own origin, then moved to wipe_tower_x/y.
        local = np.array([[-brim, -brim], [width + brim, -brim], [width + brim, depth + brim], [-brim, depth + brim]])
        angle = math.radians(_number(cfg.get("wipe_tower_rotation_angle") or 0, "wipe_tower_rotation_angle",
                                     -math.inf))
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        corners = local @ rot.T + np.array([x, y])
        return np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]])

    def bed(self):
        """(printable polygon, [exclude polygons]) in plate-local mm."""
        text = self._text("Metadata/project_settings.config")
        cfg = json.loads(text) if text else {}

        def poly(key):
            pts = [tuple(float(c) for c in p.split("x")) for p in cfg.get(key, []) or []]
            return pts if len(pts) >= 3 else None

        return poly("printable_area"), [p for p in [poly("bed_exclude_area")] if p]

    def _item_spans(self) -> list[tuple[int, int]]:
        """(start, end) of every <build><item> tag in main_text; comments ignored.

        The tags are found in the text so untouched bytes stay as they are. Names may carry
        any prefix bound to the 3MF core namespace. The result is checked against the XML
        parser; if they disagree (e.g. a prefix redeclared to another namespace) the file is
        refused rather than edited at the wrong place.
        """
        if self._spans is not None:
            return self._spans
        text = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), self.main_text)
        prefixes = _prefixes(text, CORE[1:-1])
        qname = "(" + "|".join(re.escape(f"{p}:") if p else "" for p in prefixes) + ")"
        b = re.search(rf"<{qname}build\b(?:[^>]*?/>|.*?</\1build\s*>)", text, re.S) if prefixes else None
        if not b:
            raise PlanError("3MF has no <build> section in the core namespace", "no_build")
        self._core_prefix = b.group(1)
        spans = [(b.start() + m.start(), b.start() + m.end())
                 for m in re.finditer(rf"<{re.escape(b.group(1))}item\b[^>]*>", b.group(0))]
        build = self._tree(self.main_name).find(CORE + "build")
        parsed = [i.get("objectid") for i in build.findall(CORE + "item")] if build is not None else []
        if [_attr(self.main_text[s:e], "objectid") for s, e in spans] != parsed:
            raise PlanError("3MF <build> section uses an XML layout this plugin cannot edit safely "
                            "(namespace prefixes redeclared inside it?)", "build_layout")
        self._spans = spans
        return spans

    def items(self) -> list[Item]:
        out = []
        for i, (a, b) in enumerate(self._item_spans()):
            tag = self.main_text[a:b]
            oid = _attr(tag, "objectid")
            item = Item(i, oid, _mat_from_3mf(_attr(tag, "transform")),
                        name=self.settings["names"].get(oid, f"object {oid}"),
                        printable=_attr(tag, "printable") != "0")
            self._collect(self.main_name, oid, item.transform, item.parts, oid, item=item)
            out.append(item)
        return out

    def _collect(self, file, oid, mat, parts, top_oid, part_id=None, item=None):
        obj = self._object(file, oid)
        mesh = obj.find(CORE + "mesh")
        if mesh is not None:
            subtype = self.settings["subtypes"].get(top_oid, {}).get(part_id or oid, "normal_part")
            if subtype == "normal_part":
                if item is not None:
                    slots = self.settings["filaments"].get(top_oid, {})
                    item.filaments.add(int(slots.get(part_id or oid, slots.get(None, "1")) or 1))
                    item.painted |= mesh.find(f".//{CORE}triangle[@paint_color]") is not None
                v, t = _mesh_arrays(mesh)
                if len(t):
                    vw = (np.c_[v, np.ones(len(v))] @ mat.T)[:, :3]
                    parts.append((vw, t))
            return
        comps = obj.find(CORE + "components")
        for c in [] if comps is None else comps.findall(CORE + "component"):
            sub_file = (c.get(PROD + "path") or "").lstrip("/") or file
            sub_mat = mat @ _mat_from_3mf(c.get("transform"))
            self._collect(sub_file, c.get("objectid"), sub_mat, parts, top_oid, c.get("objectid"), item)

    def write(self, out: Path, new_transforms: dict[int, np.ndarray], added=(), plate: int = 1,
              overwrite: bool = True):
        """Write a copy with new item transforms; `added` = [(object id, transform)] new instances.

        Goes through a unique temporary file, so a failed write leaves no partial output and
        an existing `out` untouched. With overwrite=False an existing `out` is an error.
        The package is closed afterwards, whatever happens.
        """
        out = Path(out)
        tmp = out.with_name(f"{out.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            if not overwrite and out.exists():
                raise FileExistsError(f"{out} already exists; not overwriting it")
            replaced = {self.main_name: self._add_items(self._replace_transforms(new_transforms), added)}
            settings = self._text("Metadata/model_settings.config")
            if added and settings is not None:
                replaced["Metadata/model_settings.config"] = self._add_plate_instances(settings, added, plate)
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for info in self.zip.infolist():
                    data = replaced[info.filename].encode("utf-8") if info.filename in replaced else self.zip.read(info)
                    zout.writestr(info, data)
            self.zip.close()  # Windows cannot replace a file that is still open
            if not overwrite and out.exists():
                raise FileExistsError(f"{out} already exists; not overwriting it")
            tmp.replace(out)
        finally:
            self.zip.close()
            tmp.unlink(missing_ok=True)

    def _add_items(self, text: str, added) -> str:
        """Append one <build><item> per new instance (Orca reads instances in build order)."""
        if not added:
            return text
        self._item_spans()  # sets the core prefix the build section uses
        core = self._core_prefix
        prod = [p for p in _prefixes(text, PROD[1:-1]) if p]  # production ext requires UUIDs
        uuid_attr = f' {prod[0]}:UUID="{{}}"' if prod else ""
        tags = "".join(f'\n  <{core}item objectid="{oid}"{uuid_attr.format(uuid.uuid4())} '
                       f'transform="{_mat_to_3mf(m)}" printable="1" />' for oid, m in added)
        clean = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)
        ends = list(re.finditer(rf"</{re.escape(core)}build\s*>", clean))
        if ends:
            end = ends[-1].start()
            return text[:end].rstrip() + tags + "\n " + text[end:]
        empty = re.search(rf"<{re.escape(core)}build\b[^>]*?(\s*/>)", clean)  # <build/>
        a, b = empty.span(1)
        return text[:a] + ">" + tags + f"\n </{core}build>" + text[b:]

    def _add_plate_instances(self, text: str, added, plate: int) -> str:
        """Register new instances on `plate` in model_settings.config."""
        counts: dict[str, int] = {}
        for a, b in self._item_spans():
            oid = _attr(self.main_text[a:b], "objectid")
            counts[oid] = counts.get(oid, 0) + 1
        next_id = max([int(v) for v in re.findall(r'key="identify_id" value="(\d+)"', text)] or [0]) + 1
        entries = []
        for oid, _ in added:
            entries.append(f'    <model_instance>\n      <metadata key="object_id" value="{oid}"/>\n'
                           f'      <metadata key="instance_id" value="{counts.get(oid, 0)}"/>\n'
                           f'      <metadata key="identify_id" value="{next_id}"/>\n    </model_instance>\n')
            counts[oid] = counts.get(oid, 0) + 1
            next_id += 1
        for m in re.finditer(r"<plate>.*?</plate>", text, re.S):
            if re.search(rf'key="plater_id" value="{plate}"', m.group(0)):
                end = m.end() - len("</plate>")
                return text[:end] + "".join(entries) + "  " + text[end:]
        if "<plate>" not in text:  # no plate list: Orca puts every item on plate 1
            return text
        raise PlanError(f"plate {plate} not found in model_settings.config", "plate_missing", plate=plate)

    def _replace_transforms(self, new: dict[int, np.ndarray]) -> str:
        text = self.main_text
        for i, (a, b) in reversed(list(enumerate(self._item_spans()))):
            if i not in new:
                continue
            tag, value = text[a:b], f'transform="{_mat_to_3mf(new[i])}"'
            if re.search(TRANSFORM_RE, tag):
                tag = re.sub(TRANSFORM_RE, value, tag)
            else:
                tag = re.sub(r"\s*(/?>)$", f" {value} \\1", tag)
            text = text[:a] + tag + text[b:]
        return text


# ---------------------------------------------------------------- rasterising

def _silhouette(parts) -> np.ndarray:
    """XY triangles whose union is the top-down footprint, (K, 3, 2). Vertical faces add nothing."""
    tris = np.concatenate([v[t][:, :, :2] for v, t in parts])
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    return tris[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-12]


def _slop_polygon(reach: float = math.sqrt(2) + 0.05, run: float = 0.5 + 0.05) -> np.ndarray:
    """Convex polygon (ccw, fine pixels) containing the disc of radius `reach` widened by `run`
    to each side along y: an octagon around the disc, shifted -run and +run, hulled."""
    a = np.radians(22.5 + 45.0 * np.arange(8))  # ccw from 22.5 degrees: 2 upper, 4 sides, 2 lower
    octagon = (reach / math.cos(math.pi / 8)) * np.stack([np.cos(a), np.sin(a)], axis=1)
    return octagon + np.where(octagon[:, 1:] > 0, run, -run) * np.array([0.0, 1.0])


SLOP = _slop_polygon()  # fine pixels; see _base_raster


def _sweeps(p: np.ndarray, q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Segments p->q (E, 2) swept by the convex ccw polygon: (E, len(poly) + 2, 2) convex polygons.
    The arc of `poly` facing the direction of travel, between its two vertices extreme across the
    segment (both included), goes with q; the rest of the turn with p."""
    m = len(poly)
    across = np.stack([-(q - p)[:, 1], (q - p)[:, 0]], axis=1) @ poly.T  # (E, m)
    right, left = np.argmin(across, axis=1), np.argmax(across, axis=1)
    split = ((left - right) % m)[:, None]
    j = np.arange(m + 2)[None, :]
    idx = (right[:, None] + j - (j > split)) % m
    ends = np.where((j <= split)[:, :, None], q[:, None], p[:, None])
    return ends + poly[idx]


def _base_raster(tris: np.ndarray, fine: float, k: int = 1):
    """Square L-mode image of `tris` (coords relative to the part center) with the center
    at the exact image middle, sized so any rotation about it stays inside. The half size is
    a multiple of `k`, so the center sits on a corner of the k x k pooled cells.

    Every pixel whose center is within sqrt(2) of the part is set, so a NEAREST rotation of the
    image still covers the rotated part: a part point x lies in an output pixel whose center is
    within sqrt(2)/2 of x; rotated back that center is within sqrt(2)/2 of the part point, and
    NEAREST samples the source pixel whose center is within 0.5 per axis (sqrt(2)/2) of it. PIL
    fills pixel i over [i, i+1) but samples its scanlines so that centers up to about 0.45 px
    (along y) inside a polygon can stay unset (measured). So each triangle is filled and each
    edge is swept by SLOP: the sqrt(2) disc widened by 0.5 along y, each with 0.05 to spare."""
    reach = float(np.abs(SLOP).max())
    half = int(math.ceil(np.linalg.norm(tris.reshape(-1, 2), axis=1).max() / fine + reach)) + 4
    half += -half % k
    img = Image.new("L", (2 * half, 2 * half), 0)
    draw = ImageDraw.Draw(img)
    px = tris / fine + half  # PIL pixel i spans [i, i+1): the part center is the image middle
    for tri in px.tolist():
        draw.polygon([tuple(q) for q in tri], fill=255)
    edges = np.concatenate([px[:, [0, 1]], px[:, [1, 2]], px[:, [2, 0]]])
    edges = np.where((edges[:, 0, 0] > edges[:, 1, 0])[:, None, None], edges[:, ::-1], edges)
    edges = np.unique(np.round(edges, 6).reshape(-1, 4), axis=0).reshape(-1, 2, 2)  # shared edges once
    for poly in _sweeps(edges[:, 0], edges[:, 1], SLOP).tolist():
        draw.polygon([tuple(q) for q in poly], fill=255)
    return img, half


def _pool(mask: np.ndarray, k: int) -> np.ndarray:
    """k x k max-pool (any). Conservative: a coarse cell is set if any fine cell is."""
    h, w = -(-mask.shape[0] // k) * k, -(-mask.shape[1] // k) * k
    m = np.pad(mask, ((0, h - mask.shape[0]), (0, w - mask.shape[1])))
    return m.reshape(h // k, k, w // k, k).any(axis=(1, 3))


DILATE_FFT_CELLS = 400  # structuring elements with more cells than this dilate by FFT (same mask, faster)


def _disc(r: int) -> np.ndarray:
    """Structuring element: integer offsets within Euclidean distance r of the center."""
    d = np.arange(-r, r + 1)
    return d[:, None] ** 2 + d[None, :] ** 2 <= r * r


def _gap_offsets(R: float) -> np.ndarray:
    """Structuring element of grid-cell offsets at which two unit cells can hold points closer
    than R cells: the gap between cells offset by (dy, dx) is hypot(max(|dy|-1, 0), max(|dx|-1, 0)).
    Dilating a footprint by it keeps every cell of another footprint at least R away. R need not
    be an integer; R = 0 keeps the cell itself only (touching allowed)."""
    if R <= 0:
        return np.ones((1, 1), bool)
    p = math.ceil(R)
    g = np.maximum(np.abs(np.arange(-p, p + 1)) - 1, 0).astype(float)
    return g[:, None] ** 2 + g[None, :] ** 2 < R * R


def _dilate_se(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary dilation of `mask` by the centered, odd-sized structuring element `se`. The output
    grows by the element's half size on every side. Large elements use an FFT correlation; the
    count of overlapping cells is an integer, so thresholding at 0.5 gives the identical mask."""
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    h, w = mask.shape
    if int(se.sum()) <= DILATE_FFT_CELLS or not mask.any():
        out = np.zeros((h + 2 * ph, w + 2 * pw), bool)
        for dy, dx in zip(*np.nonzero(se)):
            out[dy:dy + h, dx:dx + w] |= mask
        return out
    shape = (h + 2 * ph, w + 2 * pw)
    conv = np.fft.irfft2(np.fft.rfft2(mask.astype(np.float64), s=shape) * np.fft.rfft2(se.astype(np.float64), s=shape),
                         s=shape)
    return conv >= 0.5


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return _dilate_se(mask, _disc(r))


@dataclass
class Shape:
    angle: float          # world rotation applied, degrees CCW
    raw: np.ndarray       # footprint cells
    kernel: np.ndarray    # footprint dilated by spacing
    pad: int              # kernel = raw offset by (pad, pad)
    cx: float             # silhouette center within kernel, cells
    cy: float

    @property
    def coarse(self) -> np.ndarray:  # kernel on the 2x coarser screening grid
        if not hasattr(self, "_coarse"):
            self._coarse = _pool(self.kernel, 2)
        return self._coarse


def _z_angle(transform: np.ndarray) -> float:
    """Current rotation about Z, degrees CCW."""
    return math.degrees(math.atan2(transform[1, 0], transform[0, 0]))


def _center(item: Item) -> np.ndarray:
    """Center of the item's XY bounding box, world coordinates."""
    xy = np.concatenate([v[:, :2] for v, _ in item.parts])
    return (xy.min(0) + xy.max(0)) / 2


def _footprint_key(item: Item):
    """Items with equal keys have the same footprints: same object, same transform up to translation."""
    return item.object_id, tuple(np.round(item.transform[:3, :3], 4).ravel())


def _shapes(item: Item, p: Params, tick=None):
    """Candidate footprints of an item, one per rotation.

    The margin is derived, not stacked: the fine raster (res/4) already covers the part after
    a NEAREST rotation (_base_raster), pooling with "any" keeps it a superset, and the kernel adds
    exactly the cell offsets that could bring two parts closer than `spacing` (_gap_offsets).
    """
    center = _center(item)
    tris = _silhouette(item.parts) - center
    k = 4  # supersampling: rotate a fine image, then pool k x k blocks into one grid cell
    base, half = _base_raster(tris, p.res / k, k)
    gap = _gap_offsets(p.spacing / p.res)
    pad = gap.shape[0] // 2
    angles = [0.0]
    if p.step > 0:  # absolute multiples of step (so parts can land axis-aligned), plus "as is"
        current = _z_angle(item.transform)
        angles += sorted({round((a - current) % 360, 6) for a in np.arange(0, 360, p.step)} - {0.0})
    out = []
    for a in angles:
        if tick:
            tick()
        # image rows run +y, so PIL's on-screen CCW is world CW: negate
        img = base.rotate(-float(a), resample=Image.NEAREST, fillcolor=0)
        m = np.array(img.reduce(k)) > 0  # box mean > 0 == any fine pixel set
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        m = m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        c = half / k  # part center in pooled cells
        out.append(Shape(float(a), m, _dilate_se(m, gap), pad, c - cols[0] + pad, c - rows[0] + pad))
    return center, out


def _bed_grid(poly, excludes, res, obstacles=()):
    """Blocked-cell grid of the plate. `obstacles` are (K,3,2) plate-local triangles that stay put."""
    xs, ys = zip(*poly)
    origin = np.array([min(xs), min(ys)])
    nx = int(math.floor((max(xs) - origin[0]) / res))
    ny = int(math.floor((max(ys) - origin[1]) / res))
    img = Image.new("1", (nx, ny), 0)
    draw = ImageDraw.Draw(img)
    to_px = lambda pts: [((x - origin[0]) / res - 0.5, (y - origin[1]) / res - 0.5) for x, y in pts]
    draw.polygon(to_px(poly), fill=1)
    for ex in excludes:
        draw.polygon(to_px(ex), fill=0, outline=0)
    for tris in obstacles:
        for tri in tris.tolist():
            draw.polygon(to_px(tri), fill=0, outline=0)
    blocked = ~np.array(img, dtype=bool)
    return origin, _dilate(blocked, 1)[1:-1, 1:-1]  # +1 cell margin against raster slop


# ---------------------------------------------------------------- nesting

def _hits(occ_f, occ_shape, kernel: np.ndarray) -> np.ndarray:
    """True where the kernel, top-left at [y, x], overlaps an occupied cell. Kernel must fit."""
    ny, nx = occ_shape
    kh, kw = kernel.shape
    kf = np.fft.rfft2(kernel.astype(np.float32), s=occ_shape)
    return np.fft.irfft2(occ_f * np.conj(kf), s=occ_shape)[: ny - kh + 1, : nx - kw + 1] >= 0.5


def _best_position(occ_f, occ_shape, kernel: np.ndarray, bbox, holes_f=None, again=None):
    """Free top-left (x, y) of `kernel` that keeps the layout's bounding box smallest, ties to
    the lowest x + y. `holes_f` (rfft2 of free cells enclosed by placed parts): among the
    positions with that same smallest box, the one with most of the kernel inside such holes
    wins first, so small parts still go into rings and frames. `again` (the _forbidden offsets of
    this shape, when an identical part comes next): then the spot that rules out the fewest
    in-hole spots for that part, so two can share a hole. Remaining ties: lowest x + y."""
    ny, nx = occ_shape
    kh, kw = kernel.shape
    if kh > ny or kw > nx:
        return None
    hits = _hits(occ_f, occ_shape, kernel)
    x0, y0, x1, y1 = bbox or (nx, ny, 0, 0)  # first part: plain footprint area
    xs, ys = np.arange(nx - kw + 1), np.arange(ny - kh + 1)
    w = np.maximum(x1, xs + kw) - np.minimum(x0, xs)
    h = np.maximum(y1, ys + kh) - np.minimum(y0, ys)
    score = np.outer(h, w).astype(np.float32) + 1e-3 * (ys[:, None] + xs[None, :])
    score[hits] = np.inf
    i = int(np.argmin(score))
    y, x = divmod(i, score.shape[1])
    if not np.isfinite(score[y, x]):
        return None
    if holes_f is not None:
        area = np.outer(h, w)
        ties = ~hits & (area == area[y, x])  # same box: only the tie-break may change
        inside = np.fft.irfft2(holes_f * np.conj(np.fft.rfft2(kernel.astype(np.float32), s=occ_shape)),
                               s=occ_shape)[: ny - kh + 1, : nx - kw + 1]
        inside = np.where(ties, np.round(inside), -1.0)
        if inside.max() >= 1:
            top = inside == inside.max()
            if again is not None:  # one step ahead: the spot that takes fewest spots from the next part
                fh, fw = again.shape[0] // 2, again.shape[1] // 2
                spots = np.pad(top.astype(np.float32), ((fh, fh), (fw, fw)))
                blocked = np.fft.irfft2(np.fft.rfft2(spots) * np.conj(np.fft.rfft2(again.astype(np.float32), s=spots.shape)),
                                        s=spots.shape)[: top.shape[0], : top.shape[1]]
                blocked = np.where(top, np.round(blocked), np.inf)
                top = blocked == blocked.min()
            y, x = divmod(int(np.argmin(np.where(top, score, np.inf))), score.shape[1])
    return float(score[y, x]), x, y


CHECK_EVERY = 0.5  # s: longest silence between cancellation checkpoints inside long inner loops


def _checkpoints(progress):
    """`progress` that remembers its last message, with `.tick()`: repeats that message once
    CHECK_EVERY seconds have passed, so loops over hundreds of angles still reach a cancellation
    point without flooding the panel. None stays None."""
    if progress is None or hasattr(progress, "tick"):
        return progress
    last = [None, time.monotonic()]

    def report(i, n, msg):
        last[0], last[1] = (i, n, msg), time.monotonic()
        progress(i, n, msg)

    def tick():
        if last[0] is not None and time.monotonic() - last[1] >= CHECK_EVERY:
            report(*last[0])
    report.tick = tick
    return report


def _tick(progress):
    return getattr(progress, "tick", None)


def nest(items: list[Item], blocked: np.ndarray, p: Params, progress=None):
    """Place items largest-first.

    Returns ({item.index: (shape, x, y, center)}, [unplaced items], [copies]); copies
    (fill mode only) are (source item, shape, x, y, center).
    """
    progress = _checkpoints(progress)
    prepared, cache = [], {}
    for n, it in enumerate(items):
        key = _footprint_key(it)
        if key not in cache:  # identical copies share footprints
            if progress:
                progress(n, len(items), Phase(f"Footprint {it.name}", "footprint", it.name))
            cache[key] = _shapes(it, p, _tick(progress))[1]
        prepared.append((it, _center(it), cache[key]))
    prepared.sort(key=lambda t: -int(t[2][0].raw.sum()))
    placed, failed, bbox, occ = _place(prepared, blocked.astype(float), progress=progress)
    if failed:  # interlocking parts depend on who goes first: one retry with the left-over parts first
        left = {id(it) for it in failed}
        retry = [e for e in prepared if id(e[0]) in left] + [e for e in prepared if id(e[0]) not in left]
        if [_footprint_key(e[0]) for e in retry] != [_footprint_key(e[0]) for e in prepared]:
            alt = _place(retry, blocked.astype(float), progress=progress)
            if len(alt[1]) < len(failed):
                placed, failed, bbox, occ = alt
    copies = []
    if failed or p.fill:  # greedy left parts over, or fill: see if a repeating pattern does better
        alt = _lattice_layout(prepared, blocked, p, progress)
        # originals first: never trade a placed part for copies
        if alt and (len(alt[0]), len(alt[2])) > (len(placed), 0):
            placed, failed, copies, bbox = alt
        elif p.fill and not failed:  # ponytail: greedy fill only as fallback, it is the slow path
            copies = _fill(prepared, occ, bbox, progress)
    if p.center and placed and not copies:
        _center_layout(placed, blocked, bbox)
    return placed, failed, copies


MAX_COPIES = 300  # ponytail: fill cap so tiny parts on a big bed don't run for minutes; raise if needed


def _fill(prepared, occ, bbox, progress=None, limit=MAX_COPIES, pattern=None):
    """Add copies round-robin, one object at a time, until no object fits anywhere.

    An object drops out of the rotation once a copy no longer fits; smaller
    objects keep filling the gaps left by bigger ones. `pattern` =
    (lead item, its center, [(shape, x, y)] spots, fits): on the lead object's
    turns the next free spot of its repeating pattern is used before searching.
    """
    spots = iter(pattern[2] if pattern else ())

    def next_spot(it):
        if pattern and it.object_id == pattern[0].object_id:
            for s, x, y in spots:
                if pattern[3](s, x, y):
                    return pattern[0], pattern[1], (s, x, y)
        return None
    seen, active = set(), []
    for entry in prepared:  # one entry per object, largest first
        if entry[0].object_id not in seen:
            seen.add(entry[0].object_id)
            active.append(entry)
    copies = []
    while active and len(copies) < limit:
        still = []
        for it, center, shapes in active:
            if len(copies) >= limit:
                break
            if progress:
                progress(len(copies), len(copies) + len(active), Phase(f"Filling: {len(copies)} copies added", "fill",
                                                                       counted=False))
            src, src_center, best = next_spot(it) or (it, center, _try_place(shapes, occ, bbox, _tick(progress)))
            if best is None:
                continue
            s, x, y = best
            bbox = _stamp(occ, s, x, y, bbox)
            copies.append((src, s, x, y, src_center))
            still.append((it, center, shapes))
        active = still
    return copies


SCREEN_KEEP = 8  # angles re-checked at full precision after the coarse screen


def _screen(shapes: list[Shape], occ: np.ndarray, bbox, tick=None) -> list[Shape]:
    """Try every angle on a 2x coarser grid (~4x cheaper), keep the best few.

    The current rotation is always kept. If nothing fits coarsely (the coarse grid is
    pessimistic), every angle is tried at full precision: a part that fits only in a narrow
    slot at one odd angle must not be declared "did not fit" by a 15-degree subset.
    """
    if len(shapes) <= SCREEN_KEEP:
        return shapes
    occ_c = _pool(occ > 0, 2)
    occ_cf = np.fft.rfft2(occ_c.astype(np.float32))
    bbox_c = None if bbox is None else (bbox[0] // 2, bbox[1] // 2, -(-bbox[2] // 2), -(-bbox[3] // 2))
    scored = []
    for i, s in enumerate(shapes):
        if tick:
            tick()
        r = _best_position(occ_cf, occ_c.shape, s.coarse, bbox_c)
        if r:
            scored.append((r[0], i))
    if not scored:
        return shapes
    keep = {0} | {i for _, i in sorted(scored)[:SCREEN_KEEP]}
    return [shapes[i] for i in sorted(keep)]


def _try_place(shapes: list[Shape], occ: np.ndarray, bbox, tick=None, holes=None, again=False):
    """Best (shape, x, y) for one part on the current occupancy, or None. `holes`: bool grid
    of cells enclosed by placed parts; `again`: an identical part is placed next (see _best_position)."""
    occ_f = np.fft.rfft2(occ.astype(np.float32))
    holes_f = None
    if holes is not None and holes.any():
        free = holes & ~(occ > 0)
        holes_f = np.fft.rfft2(free.astype(np.float32)) if free.any() else None
    best = None
    for s in _screen(shapes, occ, bbox, tick):
        if tick:
            tick()
        r = _best_position(occ_f, occ.shape, s.kernel, bbox, holes_f,
                           _forbidden(s) if again and holes_f is not None else None)
        # shapes[0] is the current rotation; others must win by 1% (beats raster noise)
        if r and (best is None or r[0] < best[0] * 0.99):
            best = (r[0], s, r[1], r[2])
    return None if best is None else best[1:]


def _stamp(occ: np.ndarray, s: Shape, x: int, y: int, bbox):
    """Mark a placed part as occupied; return the grown layout bbox."""
    h, w = s.raw.shape
    rx, ry = x + s.pad, y + s.pad
    occ[ry:ry + h, rx:rx + w] = np.maximum(occ[ry:ry + h, rx:rx + w], s.raw)
    kh, kw = s.kernel.shape
    return (x, y, x + kw, y + kh) if bbox is None else (
        min(bbox[0], x), min(bbox[1], y), max(bbox[2], x + kw), max(bbox[3], y + kh))


def _enclosed(mask: np.ndarray) -> np.ndarray:
    """Empty cells of `mask` that cannot reach its border: through-holes and closed pockets."""
    free = ~mask
    reach = np.zeros_like(free)
    reach[[0, -1], :] = free[[0, -1], :]
    reach[:, [0, -1]] |= free[:, [0, -1]]
    while True:  # 4-connected flood from the border, one ring of cells per step
        grow = reach.copy()
        grow[1:] |= reach[:-1]
        grow[:-1] |= reach[1:]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= free
        if np.array_equal(grow, reach):
            return free & ~reach
        reach = grow


def _place(prepared, occ, bbox=None, progress=None):
    """One greedy pass in the given order onto `occ` (updated in place).

    Returns (placed, failed, bbox, occupancy).
    """
    placed, failed = {}, []
    holes = np.zeros(occ.shape, bool)  # inside placed parts' rings, frames and closed pockets
    for n, (it, center, shapes) in enumerate(prepared):
        if progress:
            progress(n, len(prepared), Phase(f"Placing {it.name}", "place", it.name))
        again = n + 1 < len(prepared) and prepared[n + 1][2] is shapes  # identical copies share shapes
        best = _try_place(shapes, occ, bbox, _tick(progress), holes, again)
        if best is None:
            failed.append(it)
            continue
        s, x, y = best
        bbox = _stamp(occ, s, x, y, bbox)
        if not hasattr(s, "holes"):
            s.holes = _enclosed(s.raw)
        h, w = s.raw.shape
        holes[y + s.pad:y + s.pad + h, x + s.pad:x + s.pad + w] |= s.holes
        placed[it.index] = (s, x, y, center)
    return placed, failed, bbox, occ


# ---------------------------------------------------------------- lattice

LATTICE_STEP = 15.0  # degrees between the rotations tried for a repeating pattern
LATTICE_DIRS = 36    # first-neighbour directions tried per rotation, over a half turn
LATTICE_TOP = 6      # densest patterns per rotation that get counted on the bed
PAIR_TOP = 2         # twin offsets per rule (smallest pair box, most shared halo) tried per rotation


def _lattice_layout(prepared, blocked, p: Params, progress=None):
    """Alternative to the greedy pass: tile copies of the largest group of identical parts
    in a repeating pattern, then place everything else greedily into the gaps.

    Identical = same _footprint_key. Returns
    (placed, failed, copies, bbox) like the greedy pass, or None.
    """
    groups: dict = {}
    for entry in prepared:
        groups.setdefault(_footprint_key(entry[0]), []).append(entry)
    group = max(groups.values(), key=lambda g: len(g) * int(g[0][2][0].raw.sum()))
    if len(group) < 2 and not p.fill:
        return None
    lead, center, shapes = group[0]
    current = _z_angle(lead.transform)
    grid = p.step * math.ceil(LATTICE_STEP / p.step) if p.step > 0 else 360.0
    tried = [s for i, s in enumerate(shapes)
             if i == 0 or (round(current + s.angle, 6) % 360 < 180 and abs(math.remainder(current + s.angle, grid)) < 1e-6)]
    need = MAX_COPIES + len(group) if p.fill else len(group)
    occ = blocked.astype(float)
    twins = {}
    if p.step > 0:  # a part and its half-turn twin can nest closer than two copies facing the same way
        by_angle = {round(s.angle % 360, 4): s for s in shapes}
        twins = {id(s): by_angle.get(round((s.angle + 180.0) % 360, 4)) for s in tried}
    spots = _lattice(tried, occ, need, progress, twins)
    if not spots:
        return None
    def fits(s, x, y):  # spots are valid by construction; this guard makes overlaps impossible anyway
        kh, kw = s.kernel.shape
        return not (occ[y:y + kh, x:x + kw][s.kernel] > 0).any()

    rest, placed, copies, bbox = list(group), {}, [], None
    for s, x, y in spots[:len(group)]:
        if fits(s, x, y):
            it, c, _ = rest.pop(0)
            placed[it.index] = (s, x, y, c)
            bbox = _stamp(occ, s, x, y, bbox)
    others = rest + [e for e in prepared if all(e is not g for g in group)]
    more, failed, bbox, occ = _place(others, occ, bbox, progress)  # originals before any copies
    placed.update(more)
    if p.fill and not failed:  # same round-robin as the greedy fill; the pattern only guides the lead's turns
        copies = _fill(prepared, occ, bbox, progress, MAX_COPIES, (lead, center, spots[len(group):], fits))
    return placed, failed, copies, bbox


def _lattice(shapes: list[Shape], occ: np.ndarray, need: int, progress=None, twins=None):
    """Best repeating pattern of one shape on the bed: [(shape, x, y)], nearest the origin first.

    `twins` maps id(shape) to its half-turn twin; then patterns of pairs (shape + twin, nested)
    compete too, scored per part. A pair pattern wins only with more parts, or the same parts
    in a strictly smaller cell per part."""
    ny, nx = occ.shape
    occ_f = np.fft.rfft2(occ.astype(np.float32))
    tick = _tick(progress)
    best, spots = (0, 0), []
    for n, s in enumerate(shapes):
        if progress:
            progress(n, len(shapes), Phase("Trying repeating patterns", "patterns"))
        kh, kw = s.kernel.shape
        if kh > ny or kw > nx or 4 * kh * kw > ny * nx:  # big parts: few copies fit, greedy does fine
            continue
        valid = ~_hits(occ_f, occ.shape, s.kernel)
        if valid.any():
            for v1, v2, area in _bases(_forbidden(s)):
                found = _lattice_spots(valid, v1, v2)
                key = (min(len(found), need), -area)
                if key > best:
                    best, spots = key, [(s, x, y) for y, x in found]
        twin = (twins or {}).get(id(s))
        if twin is None or _half_turn_symmetric(s, twin):  # a symmetric part's pairs are its plain pattern
            continue
        for m, (ay, ax), (by, bx) in _pair_motifs(s, twin, occ.shape):
            if tick:
                tick()
            valid = ~_hits(occ_f, occ.shape, m.kernel)
            if not valid.any():
                continue
            for v1, v2, area in _bases(_forbidden(m)):
                found = _lattice_spots(valid, v1, v2)
                key = (min(2 * len(found), need), -area / 2)
                if key > best:
                    best, spots = key, [t for y, x in found for t in ((s, x + ax, y + ay), (twin, x + bx, y + by))]
    return spots


def _half_turn_symmetric(a: Shape, b: Shape) -> bool:
    """`b` (a turned half a turn) has the same footprint as `a`, up to 1% raster noise."""
    return a.raw.shape == b.raw.shape and (a.raw != b.raw).sum() <= 0.01 * a.raw.sum()


def _pair_motifs(a: Shape, b: Shape, bed_shape):
    """Nested pairs of `a` and its twin `b`, as one Shape each: [(motif, a's kernel corner, b's)].

    Candidate offsets of b are the free offsets next to the forbidden ones (the raster no-fit
    boundary): the PAIR_TOP with the smallest pair box and the PAIR_TOP whose spacing halos
    overlap most."""
    ny, nx = bed_shape
    ah, aw = a.kernel.shape
    bh, bw = b.kernel.shape
    if a.pad != b.pad or 4 * (ah * aw + bh * bw) > ny * nx:
        return []
    canvas = np.zeros((ah + 2 * bh, aw + 2 * bw), np.float32)  # b's kernel corner at [bh + dy, bw + dx]
    canvas[bh + a.pad:bh + a.pad + a.raw.shape[0], bw + a.pad:bw + a.pad + a.raw.shape[1]] = a.raw
    forb = _hits(np.fft.rfft2(canvas), canvas.shape, b.kernel)
    near = np.zeros_like(forb)
    near[1:] |= forb[:-1]
    near[:-1] |= forb[1:]
    near[:, 1:] |= forb[:, :-1]
    near[:, :-1] |= forb[:, 1:]
    ys, xs = np.nonzero(~forb & near)
    if not len(ys):
        return []
    dy, dx = ys - bh, xs - bw
    box = (np.maximum(ah, dy + bh) - np.minimum(0, dy)) * (np.maximum(aw, dx + bw) - np.minimum(0, dx))
    canvas[:] = 0
    canvas[bh:bh + ah, bw:bw + aw] = a.kernel
    shared = np.fft.irfft2(np.fft.rfft2(canvas) * np.conj(np.fft.rfft2(b.kernel.astype(np.float32), s=canvas.shape)),
                           s=canvas.shape)[ys, xs]
    pick = list(np.argsort(box, kind="stable")[:PAIR_TOP]) + list(np.argsort(-shared, kind="stable")[:PAIR_TOP])
    out, seen = [], set()
    for i in pick:
        t = (int(dy[i]), int(dx[i]))
        if t in seen:
            continue
        seen.add(t)
        y0, x0 = min(0, t[0]), min(0, t[1])
        H, W = max(ah, t[0] + bh) - y0, max(aw, t[1] + bw) - x0
        if H > ny or W > nx:
            continue
        ay, ax, by, bx = -y0, -x0, t[0] - y0, t[1] - x0
        ker = np.zeros((H, W), bool)
        ker[ay:ay + ah, ax:ax + aw] |= a.kernel
        ker[by:by + bh, bx:bx + bw] |= b.kernel
        raw = np.zeros((H, W), bool)
        p = a.pad
        raw[ay + p:ay + p + a.raw.shape[0], ax + p:ax + p + a.raw.shape[1]] |= a.raw
        raw[by + p:by + p + b.raw.shape[0], bx + p:bx + p + b.raw.shape[1]] |= b.raw
        out.append((Shape(a.angle, raw[p:H - p, p:W - p] if p else raw, ker, p, 0.0, 0.0), (ay, ax), (by, bx)))
    return out


def _forbidden(s: Shape) -> np.ndarray:
    """Offsets at which a second copy would hit the first one.

    Bool (2kh+1, 2kw+1); element [kh + dy, kw + dx] is offset (dy, dx) between the two kernels.
    """
    kh, kw = s.kernel.shape
    h, w = s.raw.shape
    occ = np.zeros((3 * kh, 3 * kw), np.float32)
    occ[kh + s.pad:kh + s.pad + h, kw + s.pad:kw + s.pad + w] = s.raw
    return _hits(np.fft.rfft2(occ), occ.shape, s.kernel)


def _clip(dst_shape, src_shape, oy: int, ox: int):
    """(dst, src) slices that paste src at [oy, ox] into dst, cut to fit; None if disjoint."""
    y0, x0 = max(oy, 0), max(ox, 0)
    y1, x1 = min(oy + src_shape[0], dst_shape[0]), min(ox + src_shape[1], dst_shape[1])
    if y0 >= y1 or x0 >= x1:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))


def _bases(forb: np.ndarray):
    """Densest valid lattice bases ((dy, dx), (dy, dx), cell area) for copies of one shape.

    v1 is the nearest free offset in each of LATTICE_DIRS directions. v2 is the free offset,
    clear of the whole row of copies along v1, that gives the smallest cell area.
    """
    fh, fw = forb.shape
    cy, cx = fh // 2, fw // 2
    near = np.zeros_like(forb)
    near[1:] |= forb[:-1]
    near[:-1] |= forb[1:]
    near[:, 1:] |= forb[:, :-1]
    near[:, :-1] |= forb[:, 1:]
    ys, xs = np.nonzero(~forb & near)
    dy, dx = ys - cy, xs - cx
    half = (dy > 0) | ((dy == 0) & (dx > 0))
    dy, dx = dy[half], dx[half]
    if not len(dy):
        return []
    bins = np.minimum((np.arctan2(dy, dx) / np.pi * LATTICE_DIRS).astype(int), LATTICE_DIRS - 1)
    order = np.lexsort((dy * dy + dx * dx, bins))
    first = order[np.r_[True, bins[order][1:] != bins[order][:-1]]]
    gy, gx = np.mgrid[-(fh - 1):fh, -(fw - 1):fw]  # v2 search box: twice the forbidden box
    out = {}
    for a, b in zip(dy[first].tolist(), dx[first].tolist()):
        row = np.zeros(gy.shape, bool)  # offsets that hit any copy in the row k * v1
        k_max = min(2 * fh // abs(a) if a else 10**9, 2 * fw // abs(b) if b else 10**9)
        for k in range(-k_max, k_max + 1):
            cut = _clip(row.shape, forb.shape, fh - 1 - cy + k * a, fw - 1 - cx + k * b)
            if cut:
                row[cut[0]] |= forb[cut[1]]
        area = np.abs(a * gx - b * gy)
        area[row | (area == 0)] = np.iinfo(area.dtype).max
        for _ in range(3):  # the smallest cell may still clash two rows away; try the next
            i = int(np.argmin(area))
            if area.flat[i] == np.iinfo(area.dtype).max:
                break
            v1, v2 = _reduce((a, b), (int(gy.flat[i]), int(gx.flat[i])))
            if _clear(forb, v1, v2):
                out.setdefault((v1, v2), int(area.flat[i]))
                break
            area.flat[i] = np.iinfo(area.dtype).max
    return [(v1, v2, a) for (v1, v2), a in sorted(out.items(), key=lambda t: t[1])[:LATTICE_TOP]]


def _reduce(u, v):
    """Lagrange-reduce a 2D lattice basis: same lattice, shortest vectors."""
    dot = lambda p, q: p[0] * q[0] + p[1] * q[1]
    while True:
        if dot(u, u) > dot(v, v):
            u, v = v, u
        m = round(dot(u, v) / dot(u, u))
        if m == 0:
            return u, v
        v = (v[0] - m * u[0], v[1] - m * u[1])


def _clear(forb: np.ndarray, v1, v2) -> bool:
    """No two copies of the lattice v1, v2 collide."""
    fh, fw = forb.shape
    cy, cx = fh // 2, fw // 2
    area = abs(v1[0] * v2[1] - v1[1] * v2[0])
    reach = math.hypot(fh, fw)  # lattice vector u has |coef of v1| <= |v2| |u| / area, and vice versa
    ni, nj = int(math.hypot(*v2) * reach / area) + 1, int(math.hypot(*v1) * reach / area) + 1
    i, j = np.mgrid[-ni:ni + 1, -nj:nj + 1]
    vy, vx = i * v1[0] + j * v2[0], i * v1[1] + j * v2[1]
    m = (np.abs(vy) <= cy) & (np.abs(vx) <= cx) & ((vy != 0) | (vx != 0))
    return not forb[vy[m] + cy, vx[m] + cx].any()


def _lattice_spots(valid: np.ndarray, v1, v2) -> list[tuple[int, int]]:
    """Positions [(y, x)] of the lattice v1, v2 that land on valid cells, best shift, origin first."""
    ph, pw = abs(v1[0]) + abs(v2[0]), abs(v1[1]) + abs(v2[1])  # holds one whole lattice cell
    inv = np.linalg.inv(np.array([[v1[0], v2[0]], [v1[1], v2[1]]], float))
    vh, vw = valid.shape
    corners = np.array([[-ph, -pw], [-ph, vw], [vh, -pw], [vh, vw]], float).T
    ij = inv @ corners
    lo, hi = np.floor(ij.min(1)).astype(int), np.ceil(ij.max(1)).astype(int)
    count = np.zeros((ph, pw), np.int32)
    offsets = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            oy, ox = i * v1[0] + j * v2[0], i * v1[1] + j * v2[1]
            cut = _clip(valid.shape, count.shape, oy, ox)
            if cut:
                count[cut[1]] += valid[cut[0]]
                offsets.append((oy, ox))
    py, px = divmod(int(np.argmax(count)), pw)
    found = [(oy + py, ox + px) for oy, ox in offsets
             if 0 <= oy + py < vh and 0 <= ox + px < vw and valid[oy + py, ox + px]]
    return sorted(found, key=lambda t: (t[0] + t[1], t[0]))


def _center_layout(placed, blocked, bbox):
    ny, nx = blocked.shape
    dx = (nx - bbox[2] - bbox[0]) // 2
    dy = (ny - bbox[3] - bbox[1]) // 2
    if not (dx or dy):
        return
    for s, x, y, _ in placed.values():
        h, w = s.kernel.shape
        kx, ky = x + dx, y + dy
        if kx < 0 or ky < 0 or kx + w > nx or ky + h > ny or (blocked[ky:ky + h, kx:kx + w] & s.kernel).any():
            return  # shifted layout would hit an excluded area; keep it where it is
    for k, (s, x, y, c) in list(placed.items()):
        placed[k] = (s, x + dx, y + dy, c)


def _delta(angle_deg: float, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 world transform: rotate about src by angle (CCW), then move src to dst."""
    a = math.radians(angle_deg)
    d = np.eye(4)
    d[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    d[:2, 3] = dst - d[:2, :2] @ src
    return d


# ---------------------------------------------------------------- public API

def plate_origin(plate: int, n_plates: int, bed_w: float, bed_h: float) -> np.ndarray:
    cols = max(1, math.ceil(math.sqrt(n_plates)))
    row, col = divmod(plate - 1, cols)
    return np.array([col * bed_w * (1 + PLATE_GAP), -row * bed_h * (1 + PLATE_GAP)])


def arrange_3mf(src: Path, dst: Path, p: Params = Params(), bed=None, progress=None) -> dict:
    """Arrange one plate of `src` and write the result to `dst`. Returns a report dict."""
    pkg, (new, copies), report = plan_3mf(src, p, bed, progress)
    pkg.write(Path(dst), new, copies, p.plate)
    return {**report, "output": str(dst)}


def plan_3mf(src: Path, p: Params = Params(), bed=None, progress=None):
    """Compute the arrangement without writing.

    Returns (package, ({item index: transform}, [(object id, transform)] copies to add), report).
    The package stays open for writing; it is closed if planning fails. `progress(i, n, msg)`
    is also the cancellation point: whatever it raises aborts the plan.
    """
    check_params(p)
    pkg = Package(src)
    try:
        return pkg, *_plan(pkg, p, bed, progress)
    except BaseException:
        pkg.close()
        raise


def _plan(pkg: Package, p: Params, bed, progress):
    if progress:
        progress(0, 1, Phase("Reading project", "read", counted=False))
    poly, excludes = pkg.bed()
    poly = check_bed(poly or bed or DEFAULT_BED)
    items = pkg.items()

    plates = pkg.settings["plates"]
    if plates:
        if p.plate not in plates:
            raise PlanError(f"plate {p.plate} does not exist (project has {len(plates)})", "plate_gone",
                            plate=p.plate, plates=len(plates))
        seen: dict[str, int] = {}
        members = []
        for it in items:  # instance_id = order among build items of the same object
            k = seen.get(it.object_id, 0)
            seen[it.object_id] = k + 1
            if (it.object_id, k) in plates[p.plate]:
                members.append(it)
        items = members
    movable = [it for it in items if it.printable and it.parts and len(_silhouette(it.parts))]
    if not movable:
        raise PlanError(f"plate {p.plate} has no printable objects", "plate_empty", plate=p.plate)

    clearance = pkg.clearance(p.plate)
    if clearance > p.spacing:
        p = replace(p, spacing=clearance)
    xs, ys = zip(*poly)
    origin = plate_origin(p.plate, max(1, len(plates)), max(xs) - min(xs), max(ys) - min(ys))
    moving = {it.index for it in movable}
    obstacles = [_silhouette(it.parts) - origin for it in items if it.index not in moving and it.parts]
    obstacles = [t for t in obstacles if len(t)]
    tower = pkg.prime_tower(p.plate, items)
    if tower is not None:
        obstacles.append(tower)

    def run(q: Params):
        if min(max(xs) - min(xs), max(ys) - min(ys)) < 2 * q.res:
            raise PlanError(f"bed is smaller than the {q.res:g} mm grid", "bed_small", res=q.res)
        grid_origin, blocked = _bed_grid(poly, excludes, q.res, obstacles)
        placed, failed, copies = nest(movable, blocked, q, progress)
        to_world = lambda s, x, y, center: _delta(
            s.angle, center, origin + grid_origin + np.array([x + s.cx, y + s.cy]) * q.res)
        moves = {idx: to_world(*v) for idx, v in placed.items()}
        added = [(it, to_world(s, x, y, center)) for it, s, x, y, center in copies]
        return moves, failed, added

    moves, failed, added = run(p)
    if failed and p.refine:
        # finer grid; coarser angles keep it affordable, but step 0 (no rotation) stays 0
        moves2, failed2, added2 = run(replace(p, res=p.res / 2, step=max(p.step, 15.0) if p.step > 0 else 0.0))
        if len(failed2) < len(failed):
            moves, failed, added = moves2, failed2, added2

    by_index = {it.index: it for it in movable}
    new = {idx: d @ by_index[idx].transform for idx, d in moves.items()}
    copies = [(it.object_id, d @ it.transform) for it, d in added]
    # parts that did not fit go in a row above the plate grid, where no plate is
    cursor, top = origin[0] + min(xs), max(ys) + 10
    for it in failed:
        xy = np.concatenate([v[:, :2] for v, _ in it.parts])
        lo, hi = xy.min(0), xy.max(0)
        new[it.index] = _delta(0, lo, np.array([cursor, top])) @ it.transform
        cursor += hi[0] - lo[0] + max(p.spacing, 5)
    by_object: dict[str, int] = {}
    for it, _ in added:
        by_object[it.name] = by_object.get(it.name, 0) + 1
    report = {"placed": len(moves), "unplaced": [it.name for it in failed], "spacing": p.spacing,
              "added": len(added), "added_by_object": by_object, "plate": p.plate,
              "prime_tower": tower is not None}
    return (new, copies), report


# ---------------------------------------------------------------- Orca plugin

# >>> PANEL_HTML (generated by ui/scripts/embed.mjs from the Vue sources in ui/; do not edit by hand)
import base64 as _base64
import zlib as _zlib

PANEL_HTML_SHA256 = "f6452c925471883e4631dae24bf565cb726775a2caa3e4eb1db131f1617dc492"
_PANEL_HTML_Z = (
    "eNq0vVuX2zayBvq+f4WV48VNTKOVlpPJzqaM5nLaztV2PG47mYxGW2JTkARbAhUQ6ktE/vezqgoAQXU7M3PWOg92iyCIOwqF"
    "qq+qng4WVWnvdvLR2m435//1FP482hR6JT6T+jNIkMXi/L+ebqUtHpXrwtTSis/2dnn69Wc+WRdbKT67VvJmVxn72aOy0lZq"
    "Kz67UQu7Fgt5rUp5ig/8kdLKqmJzWpfFRooRFGKV3cjzn82q0OoPaZ5+Tgn/9ejR07o0amcfQQvFZ9tqsd/Izx6VpqrryqiV"
    "0ufLvS6tqvQjmUp22Ej7yIqfrz7I0g5LIwsrU73fbNh4WZkU3upH1fKRHNa7jbLpnM8ZsxM9FaOxkXZv9CMpzuUjpR/Z9row"
    "j6w4tFyLyZQbkTJxfmi5wh+DES8E5B3CmFxUC/nMpmdMCDEajZKklzyi5LMkSXvpT9j56MmTpjlKfPq//8N4RYXXtjC2/lXZ"
    "dTqv9PvdorAymzNe+z4Wda1WmpcildxCA7GPQg6VXsjbn5epZWN9fopNgj6XMtV8xFq+8SXsTGUrGN/huqh/vtFvTLWTxt7x"
    "vS9zMyyLzQYf+EI8M6a4G6oa//IltPM2ldDF+aTCIh+9KnbTOd899OpS2umcbx969bywcjrna3gH7YF5EmLuJ3jOV0dvamuU"
    "Xs357Dj9bntVbeb8+iid6pknyWAg+R28TK9TyZpmnUrGkmSdyqFdS+1+loUt14xf3R8oW11i1fwWCrlyw8P4pe/WsMaR/pqf"
    "jhi/eKizVOZ0zm/g7SqVLEnkQIj56+L1PEnk5GwKT6fzJJnPT3aw7X7QNpV8hItMcimFTOf8o7zjRi7h32xZGfwLaZX+RVcL"
    "+Y1cVka+qvba+iR8kIt+DlpZPo2ejrPobVyMe5SLOeNWQif+ZPPR3kq1OIfN1jQp7jmZasZYy7UUn5/+8+bzFX8nrExx4Ru5"
    "2xSwWCWnjYADOmJDW73f7aS5KGqZMsa4keLzf36TTp6d/mPKPl/xF/eKMJLPTx+P5vDty+omfMuVDHlhA+IG7pd/0lXMeBGy"
    "5/NKPz4ooDjtPJvPGa+k3ywD139V04ap8c1wOMTdGaiQOBvrp3K4kXpl12N9csLkRE9TzNfy51ga19yIwQi+c4Uu5FJp6Xco"
    "5jmUlV6q1d4UVxuZDc641PutdE8jfmOUxd+GXxebvcx0y1pexhOGi+vbTVXYVPqpeqTq18Xr1LJcZrblmzg/LNb89X57JU0q"
    "Wfa6eP3wR3vJFxKJ5V42jXAbcbWprorNu7Wqn87387x7zFyGWm6W+Ap++MQbpRfVDSbTz6xXXFRUdmjH4UxY4qGglukCdrhr"
    "P2T49DRgJiNgNrgSq9SwfCVTw7Il/D9Wy1Qx/zkeFIrZiZwKNZHT1o2DbdUyXSFpAQLD/NGCR8pOis/HaT6Y/F86/cs/GazZ"
    "rRSfZ+nk/6Yn7HO+luLzz9I8m/zfZ//857T55z8n/zdlf/ms+W9M++8o7b/pR/PPz//5l8n/Tf+S//Mv//z881XX/VV0Jh7a"
    "cMKFnbH2mys6ZT7/y5zl83kGRAzPyJ1kw2VlXhTlGhY/jKYrFI4ZyrOVbKzdKJ6PkgT29+RsOrRGbVM2FXoy8g+sbRm3bWjj"
    "z10T5/OxGzdmhRzLTS0f+bn71zP2cwpzxsYGqj8R5mT+aM5aX8h1rxCYN4k7jjJrzOwHyLqm4nTNkM4qK7d1We0kLzab6ma5"
    "32zq0kip+ULVsL92qrR7I5V2P3y6kdvKwnDfXRXlR76szFZX18VGIb1V9bbYcV0RY8OjN/BFrfRGacmNLBaV3tzNWTe11zi1"
    "1ODBQDaNhNNl3o3rnSQSgsvfj9dACOt+ukU5GI1p+AZnYVcocQajqLphVicnzIgLmcqJmnIL/+kwXKar9Kpfaa3+kFgl/Diu"
    "kPiIpam2qWVcCS1vHr1X2n6NL1Lj2xmaZZFzo+mW4nQUXhTibFw89R+Mi5MTppbpQE2KaZJcyNRyMymgxewgRTG+MrL4CDtU"
    "Pj3rGgX7V4zcDh6cdX26DX2idsPy50ogZ6WWqWka1TSpETtK30FvIDFsexjKPIxMBnUtU0fQP8q7GviFMD1xumX35sp32dAK"
    "PtB0HbNuqWG8EPZ+KpKvJBkUTTNQSQJ/cFINTKrBIfJVeVpGnA5xL+63Zd3oXEp/UrGDbhoxgWl8BYua/k5htieKF1OhsXZo"
    "FFLGAn/ZMExquJKW6rFJUuCTRV5nrIY1vILjtMCflkuGy6gSxo1rOIOGC7mRcFBDZvfbMl51bb7orVKoz/f6zK3NLc0lrMyw"
    "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"
    "cX47f8n9QzgteuT1MkQVVvgQ5ATQgPmvXp3NZt+WvL6+ZkfgFJUyfsxIdVLMHe7lx+wkrI60ExV5tAxfL91vl+T4OHr1qp4V"
    "Jftany/PRqfs048piz+KTk488lPXxrisigWkVDylVcfG2PjWZwZ6mhaz17slrxhOeWHewhj99SWnaF1DmqU33qqa6SysSn1Z"
    "LJrrKqOo2BLNfCtdC95N4AlpatNVtdkZklt/FDCzUZFEs2M00ro469bYLc7XQL2hYF8m/DHjrMqSnHK2DY/9NtPyIs8yo95g"
    "vK0J0raIfi9pX4YJ2TD3FQNwJjpiCkS0uCXG2sQwbMXf3zQoIzZPRxCcsnlR2XHn5PB/+9vdpsAk/k6OEmFMj9yix5Z+i6m9"
    "WZlB85YZfpqOhK426GmH2O9YbqR/smLeDOILY7o4h/6kdmar02Q2OmW/JKxyx/5HD4gz9j+Z4J+P7+Of+58+8Jwknyd5wtnb"
    "zgmd23Fx/nNKjCkZ1cui4r0TM5Ez8yH8++GHYHvamhOrr1HMcDVNIMDRcNSiLRTSTNP4N61wPH24HN1/8Juj4VKIU5sjNvF+"
    "PtqpB7t2fw+jT/t7gLrvsQvx7+9G5+z0dcJH/4nl94TlmxKxXFM3RfLhDgC+LZY/3o6CT94Oybfq4daU/J+ovjmqb2J/ZVhd"
    "mUeYjtXVgAgGhk/GI7mzYT6hrPUCzJaksqBnX1cql6hHyEMDpWfahuCF/P45HMyUDYFhvNT7vQnpJWDYYHwEB+WBeF/GcKI1"
    "gfyuPxqtdaydxtoE6KctJkAGzBtNgDqGEs2tt0KbMHLq9iLODs1t8UH0G9RkodAcgbQcYfvGGIfuzwFDoGe3CkdrcoOqrYrD"
    "t9ZXZqPiKtqgQBtc/GwTHd48P9fxIHDkeXFuXjO/sC59n1pPX/dfAX+z0xWwCMVg6RF6AjbcaQDLEkgVO47Md3lxfkcreMyy"
    "9OJOEE3G48HADYkRuYGaWNshPoBYk56kXupbxDXsLS6rYZwH/K1MF/RjOIYHvXJ2AEWwgA2xCmSBgWAF/Q76qHBkzUJArZRK"
    "Db30Wq7rL+ANtSlruj0AhULidnSpa8+vO10/X3rrZ/ri8xtx8fldK77JCflFXhAe4u3BDAKDda7lHPiRI8eL1uRz1I6Zl3T/"
    "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"
    "wbhUwQ=="
)
PANEL_HTML = _zlib.decompress(_base64.b64decode(_PANEL_HTML_Z)).decode("utf-8")
# <<< PANEL_HTML


def _open_in_orca(path: Path) -> bool:
    """Ask the running OrcaSlicer to open `path`. Returns False if no launcher was found."""
    import os
    import subprocess

    if sys.platform == "darwin":
        app = next((p for p in Path(sys.prefix).resolve().parents if p.suffix == ".app"), None)
        cmd = ["open", "-a", str(app) if app else "OrcaSlicer", str(path)]
    elif sys.platform.startswith("linux"):
        exe = os.environ.get("APPIMAGE") or os.path.realpath("/proc/self/exe")
        cmd = [exe, str(path)]
    else:
        candidates = [Path(sys.executable)] + [Path(sys.prefix).parent / n
                                               for n in ("orca-slicer.exe", "OrcaSlicer.exe")]
        exe = next((c for c in candidates if c.is_file() and "orca" in c.name.lower()), None)
        if exe is None:
            return False
        cmd = [str(exe), str(path)]
    subprocess.Popen(cmd)
    return True


def _call(obj, name):
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


AUTO = dict(step=1.0, res=0.5, refine=True)  # plugin presets: 360 angles, 0.5 mm grid, 0.25 mm retry


def _current_plate() -> int:
    """1-based plate selected in Orca; falls back to plate 1."""
    for host in (orca.host.model(), orca.host.plater()):
        if hasattr(host, "current_plate_index"):
            return int(_call(host, "current_plate_index")) + 1  # Orca indexes plates from 0
    return 1


class Cancelled(Exception):
    """Raised at a progress checkpoint once the user cancelled the job."""


CANCELLED = "cancelled. Nothing was written or opened."


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


# English cause and next step for the errors the page can explain. The page carries the same
# wording (and pt-BR) keyed by `code`; tests keep the two in step.
ERRORS_EN = {
    "bad_number": ("Spacing must be a number, 0 mm or more.", "Enter a value such as 2 and try again."),
    "bad_rotation": ("Rotation must be automatic or keep orientation.", "Choose one of the two options."),
    "no_model": ("The project file has no 3D model.", "Save the project again in OrcaSlicer and retry."),
    "bad_unit": ("The project uses {unit} as its unit; only millimetres can be arranged.",
                 "Convert the model to millimetres in OrcaSlicer, save it and retry."),
    "no_build": ("The project file has no build section.", "Save the project again in OrcaSlicer and retry."),
    "build_layout": ("The project file's layout cannot be edited safely.",
                     "Save the project again in OrcaSlicer (this rewrites the file) and retry."),
    "plate_missing": ("Plate {plate} is not in the project's plate list.", "Select an existing plate and retry."),
    "plate_gone": ("Plate {plate} does not exist: the project has {plates} {plate_word}.",
                   "Select one of the existing plates and retry."),
    "plate_empty": ("Plate {plate} has no printable objects.",
                    "Select another plate, or add parts to this one."),
    "bed_small": ("The bed is smaller than the {res:g} mm grid used to place parts.",
                  "Check the printer's bed shape in OrcaSlicer."),
}
UNEXPECTED_EN = ("Something unexpected went wrong.", "Try again. If it keeps happening, the details below help to diagnose it.")


def error_args(e: BaseException) -> dict:
    """The structured cause of `e`: {'code', 'args'} for a PlanError, else the generic 'unexpected'."""
    if isinstance(e, PlanError):
        return {"code": e.code, **e.args_}
    return {"code": "unexpected"}


def error_en(cause: dict) -> tuple[str, str]:
    fmt = ERRORS_EN.get(cause.get("code"))
    if not fmt:
        return UNEXPECTED_EN
    args = {k: v for k, v in cause.items() if k != "code"}
    args["plate_word"] = _plural(args.get("plates", 0), "plate", "plates")
    return fmt[0].format(**args), fmt[1]


def failure_text(action: str, label: str, cause: dict, detail: str) -> str:
    why, nxt = error_en(cause)
    lines = [f"{label}: could not {action}.", why, f"Next: {nxt}"]
    if cause.get("code") not in ERRORS_EN:  # known causes explain themselves; others keep the raw error
        lines.append(f"Details: {detail}")
    return "\n".join(lines)


OPEN_EN = {  # what happens after the copy is saved. Neutral: what Orca does with its open project is not known
    "asked": "Asked OrcaSlicer to open the copy. If nothing appears, use Open this copy in OrcaSlicer.",
    "later": "Finished while the panel was closed. Use Open this copy in OrcaSlicer to load it.",
    "failed": "Could not ask OrcaSlicer to open it ({detail}). Use Open this copy in OrcaSlicer or open the file by hand.",
    "none": "No OrcaSlicer launcher found: open the saved copy by hand.",
}


def start_failed_text(cause: dict, detail: str) -> str:
    why, nxt = error_en(cause)
    text = f"Could not start: {why}\nNext: {nxt}"
    return text + (f"\nDetails: {detail}" if cause.get("code") not in ERRORS_EN else "")


def save_failed_text(label: str, detail: str) -> str:
    return (f"{label}: could not save the arranged copy: {detail}\n"
            "Nothing was opened; earlier copies are unchanged.")


def result_msg(report: dict, p: Params) -> dict:
    """What the page needs to write the result in its own language. Names stay as they are."""
    return {"placed": report["placed"], "added": report["added"], "by_object": report.get("added_by_object", {}),
            "limit": report["added"] >= MAX_COPIES and bool(report["added"]), "max": MAX_COPIES,
            "tower": bool(report["prime_tower"]),
            "clearance": report["spacing"] if report["spacing"] > p.spacing else None,
            "keep": p.step == 0, "unplaced": list(report["unplaced"])}


def _summary(report: dict, p: Params, label: str) -> str:
    """Result text for the panel. Says what was checked and what is only an estimate."""
    lines = [f"{label}: placed {report['placed']} {_plural(report['placed'], 'object', 'objects')}."]
    added = report["added"]
    if added:
        per = ", ".join(f"{name} ×{n}" for name, n in report.get("added_by_object", {}).items())
        lines.append(f"Filled the plate with {added} extra cop{'y' if added == 1 else 'ies'}"
                     + (f" ({per})." if per else "."))
        if added >= MAX_COPIES:
            lines.append(f"Stopped at the {MAX_COPIES}-copy limit.")
    if report["prime_tower"]:
        lines.append("Kept an estimated prime tower area free, brim included. Its real size is only "
                     "known after slicing: check it in Orca.")
    if report["spacing"] > p.spacing:
        lines.append(f"Print sequence is By object: kept {report['spacing']:g} mm XY clearance between parts. "
                     "Orca still checks height and print order.")
    if p.step == 0:
        lines.append("Rotation was off: every part kept its orientation.")
    if report["unplaced"]:
        lines.append(f"Did not fit (moved above the plate): {', '.join(report['unplaced'])}")
    lines.append("Arranged from the project as it was when you clicked.")
    return "\n".join(lines)


class _Job:
    """One arrange run, split across threads. Its state lives here, not in the page, so a
    closed, reopened or reloaded panel picks it up again (see _Session).

    UI thread:  _Session.start() exports the project copy (export_3mf_copy is UI-thread only)
    worker:     work() nests the parts; progress goes to the panel via post(), which Orca
                marshals to the UI thread. Every progress call is a cancellation point.
    UI thread:  finish() writes the 3MF and launches it. The page relays the worker's
                "finish" back as a message, so this runs on the UI thread, where file and
                process permission prompts are allowed. A reopened panel finishes a job
                whose relay was lost.

    States: running -> ready -> writing -> done; running/ready -> cancelled; failed.
    Each job has its own snapshot (jobs/<id>/source.3mf) and its own output name, and
    never overwrites an existing file.
    """

    ACTIVE = ("running", "ready", "writing")

    def __init__(self, session, job_id: str, p: Params, label: str, src: Path, dst: Path, project: str = ""):
        self.session, self.id, self.p, self.label, self.src, self.dst = session, job_id, p, label, src, dst
        self.stem = Path(project).stem if project else None  # None: the project is not saved yet
        self.done_msg: dict = {}
        self.msg = None  # structured twin of `text`: what the page translates
        self.state, self.text, self.result, self.path, self.thread = "running", "", None, None, None
        self.summary, self.project = "", project  # project: path of the exported project, "" if unsaved
        self.error = False
        self.seen = session.clock()  # last time a visible panel talked to this job
        self.cancelled = threading.Event()
        self.lock = threading.Lock()  # state changes and their posts, worker vs UI thread

    @property
    def active(self) -> bool:
        return self.state in self.ACTIVE

    @property
    def action(self) -> str:
        return "fill" if self.p.fill else "arrange"

    def _ref(self) -> dict:
        return {"stem": self.stem, "plate": self.p.plate, "action": self.action}

    def post(self, text: str, msg: dict | None = None, **extra):
        self.text = text
        self.msg = {**self._ref(), **msg} if msg else None
        self.session.post(job=self.id, text=text, msg=self.msg, busy=self.active, error=self.error,
                          lang=self.session.lang, **extra)

    def snapshot(self) -> dict:
        with self.lock:
            return {"job": self.id, "text": self.text, "msg": self.msg, "busy": self.active,
                    "error": self.error, "path": str(self.path) if self.path else None}

    # The worker never posts while holding the lock: if the host's post() waited for the
    # UI thread, a UI thread waiting for the lock would deadlock. The UI thread may post
    # under the lock; it cannot block itself.

    def progress(self, i: int, n: int, msg: str):
        if self.cancelled.is_set():
            raise Cancelled()
        phase = getattr(msg, "code", "other")
        counted = getattr(msg, "counted", True) and n > 0
        # counts are only shown where the total is real; the page draws a determinate bar from them
        self.post(f"{self.label}: {msg}" + (f" ({i + 1}/{n})" if counted else ""),
                  {"code": "progress", "phase": phase, "subject": getattr(msg, "subject", ""),
                   "i": i, "n": n if counted else None, "count": i if phase == "fill" else None})

    def _end(self, state: str, text: str, msg: dict | None = None):
        """Final state. The caller must not hold the lock."""
        with self.lock:
            self.state, self.error = state, state == "failed"
        if state == "cancelled":
            self._drop_snapshot()
        self.post(text, msg, done=True)

    def _drop_snapshot(self):
        self.src.unlink(missing_ok=True)
        try:
            self.src.parent.rmdir()
        except OSError:
            pass

    def work(self):
        import traceback

        try:
            pkg, new, report = plan_3mf(self.src, self.p, progress=self.progress)
        except Cancelled:
            self._end("cancelled", f"{self.label}: {CANCELLED}", {"code": "cancelled"})
            return
        except Exception as e:
            traceback.print_exc()  # lands in OrcaSlicer's python log; the snapshot stays for diagnosis
            cause = error_args(e)
            self._end("failed", failure_text(self.action, self.label, cause, str(e)),
                      {"code": "failed", "cause": cause, "detail": str(e)})
            return
        with self.lock:  # cancel() decides under the same lock: exactly one of the two wins
            cancelled = self.cancelled.is_set()
            if not cancelled:
                self.result, self.state = (pkg, new, report), "ready"
        if cancelled:
            pkg.close()
            self._end("cancelled", f"{self.label}: {CANCELLED}", {"code": "cancelled"})
            return
        self.post(f"{self.label}: saving the arranged copy…", {"code": "saving"}, relay="finish")

    def cancel(self):
        """UI thread. A running worker stops at its next progress checkpoint."""
        with self.lock:
            state, result = self.state, self.result
            if state in ("running", "ready"):
                self.cancelled.set()
            if state == "ready":
                self.result = None
            if state == "running":  # posted before the worker can see the flag and end the job
                self.post(f"{self.label}: cancelling…", {"code": "cancelling"})
        if state == "ready":
            result[0].close()
            self._end("cancelled", f"{self.label}: {CANCELLED}", {"code": "cancelled"})

    def finish(self, open_now: bool):
        """Write the result once (later calls do nothing) and, if asked, open it."""
        import traceback

        with self.lock:
            if self.state != "ready":
                return
            self.state = "writing"
            (pkg, new, report), self.result = self.result, None
        try:
            pkg.write(self.dst, *new, self.p.plate, overwrite=False)
        except Exception as e:
            traceback.print_exc()
            self._end("failed", save_failed_text(self.label, str(e)), {"code": "save_failed", "detail": str(e)})
            return
        self.path = self.dst
        self._drop_snapshot()
        self.summary = _summary(report, self.p, self.label) + f"\nSaved as: {self.dst}"
        self.done_msg = {"code": "done", **result_msg(report, self.p)}
        if open_now:
            tail, how = self._launch()
        else:
            tail, how = "\n" + OPEN_EN["later"], {"open": "later"}
        with self.lock:
            self.state = "done"
            self.post(self.summary + tail, {**self.done_msg, **how}, done=True, path=str(self.dst))

    def _launch(self) -> tuple[str, dict]:
        """Ask the host to open the saved copy. Returns the English text and the structured outcome.
        The wording stays neutral: what Orca does with its current project and windows is not known."""
        import traceback

        try:
            opened = _open_in_orca(self.dst)
        except Exception as e:
            traceback.print_exc()
            return "\n" + OPEN_EN["failed"].format(detail=e), {"open": "failed", "detail": str(e)}
        if opened:
            return "\n" + OPEN_EN["asked"], {"open": "asked"}
        return "\n" + OPEN_EN["none"], {"open": "none"}

    def open(self):
        """Open the saved copy again (UI thread)."""
        tail, how = self._launch()
        with self.lock:
            self.post(self.summary + tail, {**self.done_msg, **how}, path=str(self.path))


WATCH_GAP = 5.0  # s. A page that polls every second while a run is busy has watched it; a longer silence has not


class _Session:
    """The plugin's single authoritative job, shared by every entry: the Organizer tab, the
    dock panel opened by the Script, and any reopened or reloaded page. Each entry only
    shows the session's state and sends requests to it; none starts work on its own."""

    def __init__(self, storage, clock=time.monotonic):
        self.storage, self.clock, self.sinks, self.job = storage, clock, [], None
        self.lang = None  # host UI language, e.g. "pt_BR"; read on the UI thread and cached for the worker
        self._warned: set = set()

    def attach(self, sink):
        if sink not in self.sinks:
            self.sinks.append(sink)

    def detach(self, sink):
        if sink in self.sinks:
            self.sinks.remove(sink)

    def post(self, **msg):
        for sink in list(self.sinks):
            try:
                if not sink.is_open():  # panel closed by the user
                    self.detach(sink)
                    continue
                sink.post(msg)
            except Exception:  # a failing page must not stop the job; say so once in the Orca log
                if id(sink) not in self._warned:
                    import traceback

                    self._warned.add(id(sink))
                    traceback.print_exc()

    def on_message(self, data):
        import traceback

        try:
            self._dispatch(data if isinstance(data, dict) else {})
        except Exception as e:  # report every failure in the panel, never crash the host
            if not isinstance(e, ValueError):  # bad input needs no stack trace in the log
                traceback.print_exc()
            cause = error_args(e)
            self.post(text=start_failed_text(cause, str(e)), msg={"code": "start_failed", "cause": cause, "detail": str(e)}, error=True,
                      busy=bool(self.job and self.job.active), lang=self.lang)

    def _dispatch(self, data: dict):
        action, job = data.get("action"), self.job
        mine = job is not None and data.get("job") == job.id  # stale ids are ignored
        if action == "hello":
            if job is not None:
                job.seen = self.clock()
            self.sync()
            if job is not None and job.state == "ready":  # its relay was lost with the old page
                job.finish(open_now=False)
        elif action == "poll":
            # the Organizer tab never receives posts made on the worker thread (the host does not
            # document post_message as thread-safe), so it asks. A ready job is finished here, on
            # the UI thread, and opened only if this page was watching it.
            if job is not None:
                watched = self.clock() - job.seen <= WATCH_GAP
                job.seen = self.clock()
                if job.state == "ready":
                    job.finish(open_now=watched)
            self.sync()
        elif action == "finish" and mine:
            job.finish(open_now=True)
        elif action == "cancel" and mine:
            job.cancel()
        elif action == "open" and mine and job.path:
            job.open()
        elif action in ("arrange", "fill"):
            if job is not None and job.active:
                self.sync()  # one job at a time: show the running one
            else:
                self.start(action, data)

    def _stale(self, info: dict) -> str:
        """Non-empty when the job shown belongs to another project or plate than the selection now."""
        job = self.job
        if job is None or info.get("plate") is None:
            return ""
        if (job.project, job.p.plate) == (info["project"], info["plate"]):
            return ""
        return f"The run below is for {job.label}, not the current selection."

    def sync(self):
        self.lang = _host_language() or self.lang
        info = _context_info()
        stale = self._stale(info)
        msg = {"sync": True, "context": info["text"], "lang": self.lang,
               "ctx": {k: info.get(k) for k in ("plate", "objects", "name", "problem")},
               "stale": stale, "stale_ref": self.job._ref() if stale else None, "job": None, "busy": False}
        if self.job is not None:
            msg.update({k: v for k, v in self.job.snapshot().items() if v is not None})
        self.post(**msg)

    def start(self, action: str, data: dict):
        spacing = _number(data.get("spacing"), "spacing")
        rotation = data.get("rotation", "auto")
        if rotation not in ("auto", "keep"):
            raise PlanError(f"rotation must be 'auto' or 'keep', got {rotation!r}", "bad_rotation")
        preset = dict(AUTO, step=0.0) if rotation == "keep" else AUTO
        p = check_params(Params(plate=_current_plate(), spacing=spacing, fill=action == "fill", **preset))
        plater = orca.host.plater()
        project = _call(plater, "project_path") or ""
        stem = Path(project).stem if project else "untitled"
        job_id = uuid.uuid4().hex[:12]
        storage = Path(self.storage())
        work = storage / "jobs" / job_id
        work.mkdir(parents=True)  # fresh per job: no other job reads or writes here
        src = work / "source.3mf"
        dst = storage / f"{stem}-arranged-{time.strftime('%Y%m%d-%H%M%S')}-{job_id[:6]}.3mf"
        try:
            plater.export_3mf_copy(str(src))
        except BaseException:
            src.unlink(missing_ok=True)
            work.rmdir()
            raise
        job = self.job = _Job(self, job_id, p, f"{stem}, plate {p.plate}", src, dst, project)
        self.lang = _host_language() or self.lang
        info = _context_info()
        job.post(f"{job.label}: reading project…", {"code": "reading"}, start=True, context=info["text"],
                 ctx={k: info.get(k) for k in ("plate", "objects", "name", "problem")})
        job.thread = threading.Thread(target=job.work, daemon=True)
        job.thread.start()


def _host_language():
    """The language OrcaSlicer's UI runs in (`orca.host.app_language()`, e.g. "pt_BR"), or None when the
    host does not offer it yet. The page maps it to a supported language and falls back to the browser's."""
    try:
        lang = _call(orca.host, "app_language")
    except Exception:
        return None
    return lang if isinstance(lang, str) and lang else None


def _context_info() -> dict:
    """Project, selected plate and object count, shown above the controls."""
    try:
        project = _call(orca.host.plater(), "project_path") or ""
        plate = _current_plate()
    except Exception as e:
        return {"project": "", "plate": None, "objects": None, "name": None, "problem": str(e),
                "text": f"Could not read the project: {e}. Open a project in OrcaSlicer, then try again."}
    try:
        objects = len(_call(orca.host.model(), "objects"))
    except Exception:  # the count is a convenience; the host may not offer it
        objects = None
    name = Path(project).stem if project else "Untitled project"
    text = f"{name} · plate {plate} selected"
    if objects is not None:
        text += f" · {objects} {_plural(objects, 'object', 'objects')} in the project"
    return {"project": project, "plate": plate, "objects": objects, "text": text, "problem": None,
            "name": Path(project).stem if project else None}


def _context() -> str:
    return _context_info()["text"]


class _PageSink:
    """Adapter between the session and the Organizer tab.

    The host does not document `post_message` as thread-safe, so only the UI thread posts
    through it. Posts made on the worker are dropped; the page asks for the state with `poll`.
    `post_message` before the page is built is dropped too, which `hello` covers."""

    def __init__(self):
        self.page, self.ui_thread = None, None

    def bind(self, page):
        self.page, self.ui_thread = page, threading.get_ident()

    def unbind(self):
        self.page = None

    def is_open(self) -> bool:
        return self.page is not None

    def post(self, msg: dict):
        if self.page is not None and threading.get_ident() == self.ui_thread:
            self.page.post_message(msg)


def _pages_base():
    """The host's Pages capability base, or None on a host without usable Pages."""
    return getattr(getattr(orca, "pages", None), "PagesPluginCapabilityBase", None)


def _super_hook(obj, name: str):
    """Run the base class hook if it has one. A base without it must not break load/unload."""
    try:
        getattr(super(type(obj), obj), name)()
    except Exception:
        pass


PAGE_NAME = "Organizer"  # tab label; must be unique within the plugin and have no ';'

if orca is not None:
    _panel = None
    _session = _Session(lambda: orca.host.plugin.storage())
    _page_sink = _PageSink()
    _PageBase = _pages_base()

    class ArrangeScript(orca.script.ScriptPluginCapabilityBase):
        def get_name(self):
            return "Shape-Aware Arrange"

        def execute(self):
            global _panel
            if _panel is not None and _panel.is_open():
                _panel.show()
            else:  # a new panel attaches to the same session, so a running job is not lost
                _panel = orca.host.ui.create_dock_panel(
                    html=PANEL_HTML, title="Organizer", width=280, height=360,
                    dock="right", on_message=_session.on_message)
                _session.attach(_panel)
            return orca.ExecutionResult.success("Panel opened")

    if _PageBase is not None:
        class OrganizerPage(_PageBase):
            """The Organizer tab: Arrange and Fill one click away, no Plugins window needed."""

            def get_name(self):
                return PAGE_NAME

            def get_ui(self):  # called when the tab is first shown
                _page_sink.bind(self)
                _session.attach(_page_sink)
                return PANEL_HTML

            def on_message(self, data):
                _page_sink.bind(self)  # the bridge calls on the UI thread
                _session.attach(_page_sink)
                _session.on_message(data)

            def on_load(self):
                _page_sink.bind(self)
                _super_hook(self, "on_load")

            def on_unload(self):
                _session.detach(_page_sink)
                _page_sink.unbind()
                _super_hook(self, "on_unload")
    else:
        OrganizerPage = None  # old host: the Script and its dock panel are the only entry

    @orca.plugin
    class ShapeArrangePlugin(orca.base):
        def register_capabilities(self):
            orca.request_permissions(threading=True)  # nesting runs off the UI thread
            orca.register_capability(ArrangeScript)
            if OrganizerPage is not None:
                try:
                    orca.register_capability(OrganizerPage)
                except Exception:  # a host whose Pages type is unusable keeps the Script working
                    import traceback

                    traceback.print_exc()


# ---------------------------------------------------------------- CLI

def _preview(dst: Path, png: Path, p: Params, bed=None):
    """Top-down PNG of the arranged plate, for checking results by eye."""
    pkg = Package(dst)
    poly = pkg.bed()[0] or bed or DEFAULT_BED
    xs, ys = zip(*poly)
    origin = plate_origin(p.plate, max(1, len(pkg.settings["plates"])), max(xs) - min(xs), max(ys) - min(ys))
    scale = 3
    img = Image.new("RGB", (int((max(xs) - min(xs)) * scale), int((max(ys) - min(ys)) * scale)), (40, 40, 48))
    draw = ImageDraw.Draw(img)
    for it in pkg.items():
        color = tuple(int(c) for c in np.random.default_rng(it.index).integers(90, 255, 3))
        for v, t in it.parts:
            px = (v[:, :2] - origin - (min(xs), min(ys))) * scale
            px[:, 1] = img.height - px[:, 1]
            for tri in px[t].tolist():
                draw.polygon([tuple(q) for q in tri], fill=color)
    img.save(png)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--plate", type=int, default=1)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=30.0)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--fill", action="store_true", help="after arranging, add copies until the plate is full")
    ap.add_argument("--auto", action="store_true", help="plugin presets: 1 deg steps on a 0.5 mm grid; if parts are left over, one retry on a 0.25 mm grid with 15 deg steps")
    ap.add_argument("--bed", help="WxH in mm when the 3MF has no printer settings, e.g. 270x270")
    ap.add_argument("--preview", type=Path, help="write a top-down PNG of the result")
    a = ap.parse_args()
    bed = None
    if a.bed:
        w, h = (float(v) for v in a.bed.lower().split("x"))
        bed = [(0, 0), (w, 0), (w, h), (0, h)]
    params = Params(a.plate, a.spacing, a.step, a.res)
    if a.auto:
        params = Params(a.plate, a.spacing, **AUTO)
    params = replace(params, fill=a.fill)
    print(json.dumps(arrange_3mf(a.src, a.dst, params, bed=bed,
                                 progress=lambda i, n, m: print(f"[{i}/{n}] {m}", file=sys.stderr)), indent=2))
    if a.preview:
        _preview(a.dst, a.preview, params, bed)
