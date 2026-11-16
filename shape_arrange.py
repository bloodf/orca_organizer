# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.14.0"
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
MAX_COPIES = 300  # ponytail: fill cap so tiny parts on a big bed don't run for minutes; raise if needed


@dataclass
class Params:
    plate: int = 1          # 1-based plate to arrange
    spacing: float = 2.0    # min gap between parts, mm
    step: float = 30.0      # rotation step, degrees; 0 disables rotation
    res: float = 0.5        # raster cell size, mm
    center: bool = True     # center the finished layout on the plate
    refine: bool = False    # parts left over: retry once on a 2x finer grid, keep the better result
    fill: bool = False      # after arranging, add copies of the plate's objects until nothing fits
    copies: int = MAX_COPIES  # fill: most copies to add


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
        message = f"{name} must be a finite number{bound}, got {value!r}"
        # the page explains these causes by name; project settings keep the raw text
        if name == "spacing":
            raise PlanError(message, "bad_number", name=name, minimum=minimum, strict=strict)
        if name == "step":
            raise PlanError(message, "bad_step", name=name, minimum=minimum, strict=strict)
        if name == "res":
            raise PlanError(message, "bad_res", name=name, minimum=minimum, strict=strict)
        raise ValueError(message)
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
    if isinstance(p.copies, bool) or not isinstance(p.copies, numbers.Integral) or not 1 <= p.copies <= MAX_COPIES:
        raise PlanError(f"copies must be a whole number from 1 to {MAX_COPIES}, got {p.copies!r}", "bad_copies",
                        max=MAX_COPIES)
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


def _hull_parts(parts) -> list:
    """`parts` replaced by their XY convex hull as one flat part (monotone chain). Orca checks
    print-by-object clearance between convex hulls, so nesting into a concave notch is refused there."""
    pts = np.unique(np.round(np.concatenate([v[:, :2] for v, _ in parts]), 6), axis=0).tolist()  # sorted by x, y

    def chain(seq):
        h = []
        for x, y in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (y - h[-2][1]) - (h[-1][1] - h[-2][1]) * (x - h[-2][0]) <= 0:
                h.pop()
            h.append((x, y))
        return h[:-1]
    ring = chain(pts) + chain(pts[::-1])
    if len(ring) < 3:  # flat as a line: nothing to fill in
        return parts
    verts = np.c_[np.array(ring), np.zeros(len(ring))]
    return [(verts, np.array([(0, i, i + 1) for i in range(1, len(ring) - 1)], dtype=np.int64))]


def _slop_polygon(reach: float = math.sqrt(2) + 0.05, run: float = 0.5 + 0.05) -> np.ndarray:
    """Convex polygon (ccw, fine pixels) containing the disc of radius `reach` widened by `run`
    to each side along y: an octagon around the disc, shifted -run and +run, hulled."""
    a = np.radians(22.5 + 45.0 * np.arange(8))  # ccw from 22.5 degrees: 2 upper, 4 sides, 2 lower
    octagon = (reach / math.cos(math.pi / 8)) * np.stack([np.cos(a), np.sin(a)], axis=1)
    return octagon + np.where(octagon[:, 1:] > 0, run, -run) * np.array([0.0, 1.0])


SLOP = _slop_polygon()  # fine pixels; see _base_raster
# Drawn straight at its angle (no NEAREST rotation after), a pixel whose square the part touches has its center
# within sqrt(2)/2 of the part, so half the reach covers it; the 0.5 widening along y stays (PIL's scanlines).
SLOP_DRAWN = _slop_polygon(math.sqrt(2) / 2 + 0.05)


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


def _base_raster(tris: np.ndarray, fine: float, k: int = 1, slop: np.ndarray = SLOP):
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
    reach = float(np.abs(slop).max())
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
    for poly in _sweeps(edges[:, 0], edges[:, 1], slop).tolist():
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
    fine: tuple | None = field(default=None, repr=False)  # (centered silhouette, res, spacing): see _fine

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
        out.append(Shape(float(a), m, _dilate_se(m, gap), pad, c - cols[0] + pad, c - rows[0] + pad,
                         (tris, p.res, p.spacing)))
    return center, out


def _fine(s: Shape, k: int):
    """The part of `s` drawn straight at its angle on pixels k times finer than a cell, every pixel it
    touches set (SLOP_DRAWN): (mask, cy, cx), the part's center in mask pixels from its corner."""
    tris, res, _ = s.fine
    a = math.radians(s.angle)
    img, half = _base_raster(tris @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]),
                             res / k, slop=SLOP_DRAWN)
    m = np.array(img) > 0
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], half - rows[0], half - cols[0]


def _phase(s: Shape, f: np.ndarray, cy: float, cx: float, k: int, gap: np.ndarray, py: int, px: int):
    """The fine mask `f` (center (cy, cx) in its pixels) with its corner (py, px) fine pixels into a cell,
    0 <= py, px < k, as a Shape on the cell grid: pooled "any" (a superset of the part), center at the
    fraction. Returns (shape, r0, c0): the cell where the pooled mask was cropped."""
    h, w = f.shape
    m = np.zeros((h + k, w + k), bool)
    m[py:py + h, px:px + w] = f
    raw = _pool(m, k)
    rows, cols = np.flatnonzero(raw.any(1)), np.flatnonzero(raw.any(0))
    r0, c0 = int(rows[0]), int(cols[0])
    raw = raw[r0:rows[-1] + 1, c0:cols[-1] + 1]
    return Shape(s.angle, raw, _dilate_se(raw, gap), s.pad, (cx + px) / k - c0 + s.pad, (cy + py) / k - r0 + s.pad), r0, c0


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
            copies = _fill(prepared, occ, bbox, progress, p.copies)
    if p.center and placed and not copies:
        _center_layout(placed, blocked, bbox)
    return placed, failed, copies


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
FINE_TOP = 2         # rotations whose pattern is measured again on fine pixels; every direction is counted there
LATTICE_FINE = 8     # fine pixels per cell side for that, at most
LATTICE_FINE_PX = 480        # ... and at most this many fine pixels across the part with its spacing
LATTICE_FINE_CELLS = 1 << 24  # ... and on the bed (fine pixels in all)


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
    need = p.copies + len(group) if p.fill else len(group)
    occ = blocked.astype(float)
    twins = {}
    if p.step > 0:  # a part and its half-turn twin can nest closer than two copies facing the same way
        by_angle = {round(s.angle % 360, 4): s for s in shapes}
        twins = {id(s): by_angle.get(round((s.angle + 180.0) % 360, 4)) for s in tried}
    spots = _lattice(tried, occ, need, progress, twins)
    if not spots:
        return None
    mine = np.zeros(occ.shape, bool)  # cells of pattern copies: they keep the spacing by the pattern's pitch

    # Copies of one pattern are not checked against each other here: they are spaced by the pattern itself
    # (_bases/_clear on the coarse or fine outline), and every spot comes from that one pattern.
    def fits(s, x, y):  # spots are valid by construction; this guard keeps out every other part and obstacle
        kh, kw = s.kernel.shape
        ok = not ((occ[y:y + kh, x:x + kw] > 0) & ~mine[y:y + kh, x:x + kw])[s.kernel].any()
        if ok:  # the caller stamps it now
            h, w = s.raw.shape
            mine[y + s.pad:y + s.pad + h, x + s.pad:x + s.pad + w] |= s.raw
        return ok

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
        copies = _fill(prepared, occ, bbox, progress, p.copies, (lead, center, spots[len(group):], fits))
    return placed, failed, copies, bbox


def _lattice(shapes: list[Shape], occ: np.ndarray, need: int, progress=None, twins=None):
    """Best repeating pattern of one shape on the bed: [(shape, x, y)], nearest the origin first.

    `twins` maps id(shape) to its half-turn twin; then patterns of pairs (shape + twin, nested)
    compete too, scored per part. A pair pattern wins only with more parts, or the same parts
    in a strictly smaller cell per part."""
    ny, nx = occ.shape
    occ_f = np.fft.rfft2(occ.astype(np.float32))
    tick = _tick(progress)
    best, spots, ranked = (0, 0), [], []
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
                ranked.append((key, n))
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
    # The most promising rotations again, with the pitch measured on fine pixels (_fine_lattice). It must
    # fit more copies to win: with as many, the looser pattern leaves more room for parts placed after it.
    fine, fine_spots = (best[0] + 1, -math.inf), []
    for n in list(dict.fromkeys(n for _, n in sorted(ranked, reverse=True)))[:FINE_TOP]:
        if best[0] >= need or (fine_spots and fine[0] >= need):  # enough copies already
            break
        if shapes[n].fine is not None:
            key, found = _fine_lattice(shapes[n], occ_f, occ.shape, need, tick)
            if key > fine:
                fine, fine_spots = key, found
    return fine_spots or spots


def _fine_lattice(s: Shape, occ_f, occ_shape, need: int, tick=None):
    """Densest repeating pattern of `s`, with the pitch between copies measured on pixels up to
    LATTICE_FINE times smaller than a cell instead of on cells. On cells each copy's outline is
    rounded out to whole cells (about one cell more per pitch, see _shapes): over a plate of small
    parts that adds up to a whole row or column (24 mm discs on a 220 mm bed: 8 rows instead of 9).
    Each copy still goes on the cell grid, as the shape pooled at its fine offset (_phase), so it is
    stamped, and checked against the bed, obstacles and other parts, like any other part. Every
    direction is counted, not only the densest few: on a bounded bed the pattern square to its edges
    often fits more than one a hair denser on paper. Returns ((count, -area in cells), spots)."""
    ny, nx = occ_shape
    k = min(LATTICE_FINE, LATTICE_FINE_PX // max(s.kernel.shape), math.isqrt(LATTICE_FINE_CELLS // ((ny + 2) * (nx + 2))))
    if k < 2:
        return (0, 0), []
    _, res, spacing = s.fine
    f, cy, cx = _fine(s, k)
    gap_f = _gap_offsets(spacing / res * k)
    bases = _bases(_forbidden(Shape(s.angle, f, _dilate_se(f, gap_f), gap_f.shape[0] // 2, 0.0, 0.0)), LATTICE_DIRS)
    if not bases:
        return (0, 0), []
    gap = _gap_offsets(spacing / res)
    valid = np.zeros(((ny + 2) * k, (nx + 2) * k), bool)  # by fine top-left of `f`, one cell of margin
    phases = {}
    for py in range(k):
        for px in range(k):
            ph, r0, c0 = _phase(s, f, cy, cx, k, gap, py, px)
            kh, kw = ph.kernel.shape
            if kh > ny or kw > nx:
                continue
            ok = ~_hits(occ_f, occ_shape, ph.kernel)
            # kernel corner (y, x) of this phase puts f's corner at fine row (y + pad - r0 + 1) * k + py
            y0, x0 = (s.pad - r0 + 1) * k + py, (s.pad - c0 + 1) * k + px
            valid[y0:y0 + ok.shape[0] * k:k, x0:x0 + ok.shape[1] * k:k] = ok
            phases[py, px] = ph, r0, c0
    best, spots = (0, 0), []
    if not valid.any():
        return best, spots
    for v1, v2, area in bases:
        if tick:
            tick()
        found = _lattice_spots(valid, v1, v2)
        key = (min(len(found), need), -area / (k * k))
        if key > best:
            best, spots = key, found
        if best[0] >= need:
            break
    out = []
    for Y, X in spots:
        ph, r0, c0 = phases[Y % k, X % k]
        out.append((ph, X // k - s.pad + c0 - 1, Y // k - s.pad + r0 - 1))
    return best, out


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


def _bases(forb: np.ndarray, top: int = LATTICE_TOP):
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
    return [(v1, v2, a) for (v1, v2), a in sorted(out.items(), key=lambda t: t[1])[:top]]


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


def _plate_of(item: Item, poly, n_plates: int) -> int | None:
    """1-based plate an item sits on, by position, as Orca decides it; None if on no plate.

    Orca (PartPlateList::reload_all_objects) gives each instance to the first plate whose area its
    bounding box touches. The <model_instance> lists in model_settings.config are not used for this:
    Orca keeps only one entry per object when loading and rebuilds the plates from positions, and
    the lists it writes go stale for objects with several instances (wrong plate, instance ids
    that no longer exist), so trusting them left instances on the plate unmoved and in the way.
    """
    if not item.parts:
        return None
    xy = np.concatenate([v[:, :2] for v, _ in item.parts])
    lo, hi = xy.min(0), xy.max(0)
    xs, ys = zip(*poly)
    bed_lo, bed_hi = np.array([min(xs), min(ys)]), np.array([max(xs), max(ys)])
    for plate in range(1, max(1, n_plates) + 1):
        origin = plate_origin(plate, max(1, n_plates), *(bed_hi - bed_lo))
        if (lo <= origin + bed_hi).all() and (hi >= origin + bed_lo).all():
            return plate
    return None


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
    items = everything = pkg.items()

    plates = pkg.settings["plates"]
    if plates:
        if p.plate not in plates:
            raise PlanError(f"plate {p.plate} does not exist (project has {len(plates)})", "plate_gone",
                            plate=p.plate, plates=len(plates))
        items = [it for it in items if _plate_of(it, poly, len(plates)) == p.plate]
    elif p.plate != 1:  # no plate list: Orca puts every item on plate 1
        raise PlanError(f"plate {p.plate} does not exist (project has 1)", "plate_gone", plate=p.plate, plates=1)
    movable = [it for it in items if it.printable and it.parts and len(_silhouette(it.parts))]
    if not movable:
        raise PlanError(f"plate {p.plate} has no printable objects", "plate_empty", plate=p.plate)

    clearance = pkg.clearance(p.plate)
    if clearance > 0:  # by object: Orca measures the clearance between convex hulls, not outlines
        items = [replace(it, parts=_hull_parts(it.parts)) if it.parts else it for it in items]
        movable = [it for it in items if it.index in {m.index for m in movable}]
    from_clearance = clearance > p.spacing
    if from_clearance:
        p = replace(p, spacing=clearance)
    xs, ys = zip(*poly)
    # The spacing is kept from the bed edge too (a part's kernel, outline plus spacing halo, must lie inside
    # the bed grid), so at half the bed's short side no part fits at all, and rasterising the halo is huge.
    limit = min(max(xs) - min(xs), max(ys) - min(ys)) / 2
    if p.spacing >= limit and from_clearance:
        raise PlanError(f"the printer's extruder clearance radius of {p.spacing:g} mm (print by object) leaves no "
                        f"room on the bed (must be below {limit:g} mm)", "clearance_large", clearance=p.spacing,
                        limit=limit)
    if p.spacing >= limit:
        raise PlanError(f"spacing {p.spacing:g} mm leaves no room on the bed (must be below {limit:g} mm)",
                        "spacing_large", spacing=p.spacing, limit=limit)
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
    # parts that did not fit go in a row above the plate grid, where no plate is, clear of every part
    # that stays put (parked by an earlier run, or left over from another plate in this column)
    def xy_box(it):
        xy = np.concatenate([v[:, :2] for v, _ in it.parts])
        return xy.min(0), xy.max(0)
    taken = [xy_box(it) for it in everything if it.index not in moving and it.parts]
    gap, top = max(p.spacing, 5), max(ys) + 10
    for it in failed:
        lo, hi = xy_box(it)
        size, x = hi - lo, origin[0] + min(xs)
        while True:  # slide right past whatever is in the way; x only grows, so this ends
            hit = [b for b in taken if b[0][0] < x + size[0] + gap and x < b[1][0] + gap
                   and b[0][1] < top + size[1] + gap and top < b[1][1] + gap]
            if not hit:
                break
            x = max(b[1][0] for b in hit) + gap
        new[it.index] = _delta(0, lo, np.array([x, top])) @ it.transform
        taken.append((np.array([x, top]), np.array([x, top]) + size))
    by_object: dict[str, int] = {}
    for it, _ in added:
        by_object[it.name] = by_object.get(it.name, 0) + 1
    report = {"placed": len(moves), "unplaced": [it.name for it in failed], "spacing": p.spacing,
              "added": len(added), "added_by_object": by_object, "plate": p.plate,
              "prime_tower": tower is not None}
    return (new, copies), report


# ---------------------------------------------------------------- Orca plugin

# >>> PANEL_HTML (generated by ui/scripts/embed.mjs from the Vue sources in ui/; do not edit by hand)
# Third-party notices for the code and font bundled in PANEL_HTML (Vue, PrimeVue, PrimeUIX, Lucide icons,
# Manrope font). They ship inside this file because the page does.
# ---- Lucide (ui/shared/licenses/Lucide.txt)
# ISC License
#
# Copyright (c) 2026 Lucide Icons and Contributors
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# ---
#
# The following Lucide icons are derived from the Feather project:
#
# airplay, alert-circle, alert-octagon, alert-triangle, aperture, arrow-down-circle, arrow-down-left, arrow-down-right, arrow-down, arrow-left-circle, arrow-left, arrow-right-circle, arrow-right, arrow-up-circle, arrow-up-left, arrow-up-right, arrow-up, at-sign, calendar, cast, check, chevron-down, chevron-left, chevron-right, chevron-up, chevrons-down, chevrons-left, chevrons-right, chevrons-up, circle, clipboard, clock, code, columns, command, compass, corner-down-left, corner-down-right, corner-left-down, corner-left-up, corner-right-down, corner-right-up, corner-up-left, corner-up-right, crosshair, database, divide-circle, divide-square, dollar-sign, download, external-link, feather, frown, hash, headphones, help-circle, info, italic, key, layout, life-buoy, link-2, link, loader, lock, log-in, log-out, maximize, meh, minimize, minimize-2, minus-circle, minus-square, minus, monitor, moon, more-horizontal, more-vertical, move, music, navigation-2, navigation, octagon, pause-circle, percent, plus-circle, plus-square, plus, power, radio, rss, search, server, share, shopping-bag, sidebar, smartphone, smile, square, table-2, tablet, target, terminal, trash-2, trash, triangle, tv, type, upload, x-circle, x-octagon, x-square, x, zoom-in, zoom-out
#
# The MIT License (MIT) (for the icons listed above)
#
# Copyright (c) 2013-present Cole Bemis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---- Manrope (ui/shared/licenses/Manrope.txt)
# Copyright 2018 The Manrope Project Authors (https://github.com/googlefonts/manrope)
#
# This Font Software is licensed under the SIL Open Font License, Version 1.1.
# This license is copied below, and is also available with a FAQ at:
# http://scripts.sil.org/OFL
#
#
# -----------------------------------------------------------
# SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
# -----------------------------------------------------------
#
# PREAMBLE
# The goals of the Open Font License (OFL) are to stimulate worldwide
# development of collaborative font projects, to support the font creation
# efforts of academic and linguistic communities, and to provide a free and
# open framework in which fonts may be shared and improved in partnership
# with others.
#
# The OFL allows the licensed fonts to be used, studied, modified and
# redistributed freely as long as they are not sold by themselves. The
# fonts, including any derivative works, can be bundled, embedded, 
# redistributed and/or sold with any software provided that any reserved
# names are not used by derivative works. The fonts and derivatives,
# however, cannot be released under any other type of license. The
# requirement for fonts to remain under this license does not apply
# to any document created using the fonts or their derivatives.
#
# DEFINITIONS
# "Font Software" refers to the set of files released by the Copyright
# Holder(s) under this license and clearly marked as such. This may
# include source files, build scripts and documentation.
#
# "Reserved Font Name" refers to any names specified as such after the
# copyright statement(s).
#
# "Original Version" refers to the collection of Font Software components as
# distributed by the Copyright Holder(s).
#
# "Modified Version" refers to any derivative made by adding to, deleting,
# or substituting -- in part or in whole -- any of the components of the
# Original Version, by changing formats or by porting the Font Software to a
# new environment.
#
# "Author" refers to any designer, engineer, programmer, technical
# writer or other person who contributed to the Font Software.
#
# PERMISSION & CONDITIONS
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of the Font Software, to use, study, copy, merge, embed, modify,
# redistribute, and sell modified and unmodified copies of the Font
# Software, subject to the following conditions:
#
# 1) Neither the Font Software nor any of its individual components,
# in Original or Modified Versions, may be sold by itself.
#
# 2) Original or Modified Versions of the Font Software may be bundled,
# redistributed and/or sold with any software, provided that each copy
# contains the above copyright notice and this license. These can be
# included either as stand-alone text files, human-readable headers or
# in the appropriate machine-readable metadata fields within text or
# binary files as long as those fields can be easily viewed by the user.
#
# 3) No Modified Version of the Font Software may use the Reserved Font
# Name(s) unless explicit written permission is granted by the corresponding
# Copyright Holder. This restriction only applies to the primary font name as
# presented to the users.
#
# 4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
# Software shall not be used to promote, endorse or advertise any
# Modified Version, except to acknowledge the contribution(s) of the
# Copyright Holder(s) and the Author(s) or with their explicit written
# permission.
#
# 5) The Font Software, modified or unmodified, in part or in whole,
# must be distributed entirely under this license, and must not be
# distributed under any other license. The requirement for fonts to
# remain under this license does not apply to any document created
# using the Font Software.
#
# TERMINATION
# This license becomes null and void if any of the above conditions are
# not met.
#
# DISCLAIMER
# THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
# OTHER DEALINGS IN THE FONT SOFTWARE.
# ---- PrimeUIX (ui/shared/licenses/PrimeUIX.txt)
# # PrimeUI License
#
# License: https://primeui.dev/licenses
#
# This package is part of **PrimeUI**, a family of commercial UI libraries by PrimeTek Informatics.
#
# ## How PrimeUI May Be Used
#
# PrimeUI may be used under one of two licenses. You choose the one that applies to you based on eligibility.
#
# ### Community License (Free)
#
# Free for organizations that meet all of the following criteria:
#
# - Less than $1,000,000 USD in annual gross revenue
# - Fewer than 5 developers
# - Fewer than 10 employees
# - Less than $3,000,000 USD in venture capital or outside funding
#
# Individuals, students, non-profits, and non-commercial open source projects also qualify.
#
# The Community License provides full access to the core libraries and supports up to 4 developers. It requires annual renewal by confirming continued eligibility.
#
# ### Commercial License (Paid)
#
# For organizations that do not qualify for the Community License.
#
# Licensed per developer, perpetual, with one year of updates included. Renewal is optional and extends access to new versions. The Commercial Suite also includes PrimeBlocks, Theme Designer, all PrimeUI Pro components, and Premium Support.
#
# ## License Key
#
# A valid license key is required to use this software. License verification is performed offline, with no telemetry and no remote connection. A missing, invalid, or expired key may cause the software to display a license notice.
#
# ## Full Terms
#
# The complete license terms, eligibility details, and renewal information are available at:
#
# - Community License: <https://primeui.dev/licenses/community>
# - Commercial License: <https://primeui.dev/licenses/commercial>
#
# ## Restrictions
#
# This software is distributed as a compiled package. You may not reverse-engineer, decompile, or extract its source code, redistribute it as a component library or development tool, or remove its license mechanisms. Redistributing the software so that third parties can develop with it requires a separate OEM License. See the full terms for details.
#
# ---
#
# © 2026 PrimeTek Informatics. All rights reserved.
# ---- PrimeVue (ui/shared/licenses/PrimeVue.txt)
# # PrimeUI License
#
# License: https://primeui.dev/licenses
#
# This package is part of **PrimeUI**, a family of commercial UI libraries by PrimeTek Informatics.
#
# ## How PrimeUI May Be Used
#
# PrimeUI may be used under one of two licenses. You choose the one that applies to you based on eligibility.
#
# ### Community License (Free)
#
# Free for organizations that meet all of the following criteria:
#
# - Less than $1,000,000 USD in annual gross revenue
# - Fewer than 5 developers
# - Fewer than 10 employees
# - Less than $3,000,000 USD in venture capital or outside funding
#
# Individuals, students, non-profits, and non-commercial open source projects also qualify.
#
# The Community License provides full access to the core libraries and supports up to 4 developers. It requires annual renewal by confirming continued eligibility.
#
# ### Commercial License (Paid)
#
# For organizations that do not qualify for the Community License.
#
# Licensed per developer, perpetual, with one year of updates included. Renewal is optional and extends access to new versions. The Commercial Suite also includes PrimeBlocks, Theme Designer, all PrimeUI Pro components, and Premium Support.
#
# ## License Key
#
# A valid license key is required to use this software. License verification is performed offline, with no telemetry and no remote connection. A missing, invalid, or expired key may cause the software to display a license notice.
#
# ## Full Terms
#
# The complete license terms, eligibility details, and renewal information are available at:
#
# - Community License: https://primeui.dev/licenses/community
# - Commercial License: https://primeui.dev/licenses/commercial
#
# ## Restrictions
#
# This software is distributed as a compiled package. You may not reverse-engineer, decompile, or extract its source code, redistribute it as a component library or development tool, or remove its license mechanisms. Redistributing the software so that third parties can develop with it requires a separate OEM License. See the full terms for details.
#
# ---
#
# © 2026 PrimeTek Informatics. All rights reserved.
# ---- Vue (ui/shared/licenses/Vue.txt)
# The MIT License (MIT)
#
# Copyright (c) 2018-present, Yuxi (Evan) You
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import base64 as _base64
import zlib as _zlib

PANEL_HTML_SHA256 = "6d2bd5965a6603cc2fe834b5e968633614bbf8c58f7424d05a8d2f92f3406738"
_PANEL_HTML_Z = (
    "eNq0vVuX2zayBvq+f4WV48VNTKOVlpPJzqaM5nJsJ/Ekjj1p56rRltgU1IJNgRoQ6ktE/vezqgoAQXU7M3PWOg92iyCIOwqF"
    "qq+qno5WdWnvdvLRxm6r8/96Cn8eVYW+Ep9I/QkkyGJ1/l9Pt9IWj8pNYRppxSd7uz798hOfrIutFJ9cK3mzq4395FFZayu1"
    "FZ/cqJXdiJW8VqU8xQf+SGllVVGdNmVRSTGBQqyylTx/Y64Krf6Q5umnlPBfjx49bUqjdvYRtFB8sq1X+0p+8qg0ddPURl0p"
    "fb7e69KqWj+SqWSHStpHVry5fC9LOy6NLKxM9b6q2HRdmxTe6kf1+pEcN7tK2XTJl4zZmZ6LydRIuzf6kRTn8pHSj2x3XZhH"
    "Vhw6rsVszo1ImTg/dFzhj9GEFwLyjmFMntcr+cymZ0wIMZlMkmSQPKHksyRJB+lP2PnkyZO2PUp8+r//w3hNhTe2MLb5RdlN"
    "uqz1T7tVYWW2ZLzxfSyaRl1pXopUcgsNxD4KOVZ6JW/frFPLpvr8FJsEfS5lqvmEdbzyJexMbWsY3/GmaN7c6Lem3klj7/je"
    "l1mNy6Kq8IGvxDNjiruxavAvX0M7b1MJXVzOaizy0etiN1/y3UOvLqSdL/n2oVcvCivnS76Bd9AemCchln6Cl/zq6E1jjdJX"
    "S744Tr/bXtbVkl8fpVM9yyQZjSS/g5fpdSpZ225SyViSbFI5thup3c+ysOWG8cv7A2XrC6ya30Ihl254GL/w3Ro3ONJf8tMJ"
    "488f6iyVOV/yG3h7lUqWJHIkxPKH4odlksjZ2RyeTpdJslye7GDbvdI2lXyCi0xyKYVMl/yDvONGruHfYl0b/Atptf5Z1yv5"
    "lVzXRr6u99r6JHyQq2EOWlk+jZ6Os+htXIx7lKsl41ZCJ/5k89HeSrU4h83WtinuOZlqxljHtRSfnv7j5tMr/k5YmeLCN3JX"
    "FbBYJaeNgAM6YWNb/7TbSfO8aGTKGONGik//8VU6e3b6+5x9esVf3ivCSL48fTxZwrff1zfhW65kyAsbEDfwsPyTvmLGi5A9"
    "X9b68UEBxemW2XLJ+Au/V0au+6qh/VJLeDMej3FzBiIkzqb6qRxXUl/ZzVSfnDA50/MU83X8DZbGNTdiNIHvXKEruVZa+g2K"
    "eQ5lrdfqam+Ky0pmozMu9X4r3dOE3xhl8bfh10W1l5nuWMebeL5wbX1d1YVNpZ+pR6r5ofghtSyXme14GeeHtZr/sN9eSpNK"
    "lv1Q/PDwR5Xke4m0spJtK9w+vKrqy6J6t1HN0+V+mfePmcvQyGqNr+CHT7xRelXfYDL9zAbFRUVlh24ajoQVnglqna5gg7v2"
    "Q4aPTwNmMgJmgytxlRqWb2RqWLaC/6dqnSrmP8dzQjE7k3OhZnLeuXGwnVqnV0hZgL4wf7LgibKW4tNpmo9m/5fO//IPBkt2"
    "J8WnWTr7v/kJ+5Rvpfj0kzTPZv/3yT/+MW//8Y/Z/83ZXz5p/xvT/jtK+2/60f7j03/8ZfZ/87/k//jLPz799Krv/iY6Eg9d"
    "OODCxtj6vRUdMp/+Zcny5TIDGoZH5Fqy8bo2L4tyA2sfRtMVCqcM5dlJNtVuFM8nSQLbe3Y2H1ujtimbCz2b+AfWdYzbLrTx"
    "Kmrjcjl1A8eskFNZNfKRn7x/PWVQ0kzP2dRAA06EOVk+WrLOl3I9KAVmTuKeo8waM/shsq6xfTMXfiWNpJ9OoGzTStpDWRVN"
    "k1ne2LsKNpiQoZwkGV2lltHBD9mglZYxrjEJvxArpIOclsc1knVl5bYp653kRVXVN+t9VTWlkVLzlWpgP+9UafdGKu1++HQj"
    "t7WF6b27LMoPfF2bra6vi0oheVfNtthxXRMfxaM38EWjdKW05EYWq1pXd0vWL6U77D91azSSbSvhMFv2A3QpiWThIPnpGQlh"
    "3U83aqPJlGZrdBZ2oRJnMGeqn1V1csKMuIEJVXNu4T8dJsf0ld4OK23UHxKrhB/HFRLbsjb1NrWMK6HlzaOflLZf4ovU+HaG"
    "ZllkFGl1SXE6CS8KcTYtnvoPpsXJCYOFoWbFPEluZGq5mRXQYnaQopheGll8AIogn571jQJ6ISaOYozO+j5dhD5Ru2G7cSWQ"
    "kVPr1LStatvUiB2l76A3kBjIDAxlHkYmg7rWqTtAPsi7BtiTMD1xumX35sp32dB+OdB0HXOKqWG8EPZ+KpLLJBkVbTtSSQJ/"
    "cFINTKrBIfJVedpJjBUxS+63jfbhc+lPRnbQbStmMI2vYVHT3znM9kzxYi401g6NQkpc4C8bhkmNr6SlemySFPhkkbWaqnED"
    "r+D4LvCn5ZLhMqqFceMazrzxSlYS+ALI7H7Dud+3+WawSqE+3+sztza3NJewMsMqb1uV41RKaNo7tZUptrV/ykaTLDViQR8v"
    "/ELIsQp4s6I3q/AGywtDeCl9EdeU8TpkHBmYsXw0yULuC8myBycHWDigXO+QSRiNUgl3jcXieqGaH+VaCDE6Y/yDZ3NzmUkh"
    "gHjCMbMKpyQSRM9YCyEu23a0iZIYy9/BKOcfUjlGXoZlf7t488OYbgFqDfzQS8mfRM3kL6Xny97BrOQvsTf+c9hT+WG2fF3s"
    "0scHIhsdW86zGfBhY6mtUbJJ2Xxs5GpfyjSVfGa5nnPDxHkqZy9ws7OT5SNxvpwLzSXjh4512c6VfCHtQyVj/VjwttjBqfoC"
    "esa6DCYxh1JZdo3nxmhFf57DizDkmeUvqGdiuWTiHJZAvrzAa0/6+CDHK0n3ZlXrPLcdW2aSP+NvpMBDCHjGxpp9aWuTSmAx"
    "D3ajmvFK2qLcyJWQHJ8XRWnVtRSjM/dca+F+yvValraBizE+l5Us9H7XJyxU87bYN3IlRhOXclMY/Ub/uNd9gYvrRfNB7SBh"
    "JJPkWZKkz8ZUa55ill1hQJLwjD7AS61In43xcGzaNvwUszlj492+2eB37HTCsnTYiwfaMWGsuwKSjllSf8I9ij/sdtCPFHdv"
    "nO4GLeoobWfJ7dRnpbZ5dilKctcKOm0kjKrwDNRUPrVTeXLC9EzOx67yrs8Xj//xJ/G7+OvOyGa/faATSTLsxP1OTf5/6JRv"
    "TRcV4Xv0H5fRmb12zNlweoipfDa15s7P6zOsi+OAKl1U1d3hmbBd19U6ZYeTk7DMhUA2FlegkdcX0D/xjNP3rKvX63gsa31+"
    "liSnp9HnZ/j6GZBi+II+7MtC5taxFs+wr1M5dWxMX6H7+BCnHZVD3IUUUZauO2r3da1Wj866rrH17sGROtonOOOWa2yXFWdc"
    "P7jo7FM9tceLzs7HWA1N4AOf4ZSGIj3deLBM/3Jm5ykLy+/oE0+R4lUpP7oqqWp5XJ0M7R6dsWn8sa8EGLjRgEq6rUMEKklG"
    "clA3JftSdjgikkQ88D4srjjbrKdxc5DwOHLXp7Iupol+WqML74eIhj3rOZBneOwhoR+Q2Gf9YCLllCwDwQge5+/5e4lM8i+y"
    "+HAhLX/94OnhFs9a+0NjJXeNa1qf8K5Q1SBxXRVXjfgrPWh5awdvXasGaQ2M+b6Sxqce9cSvsv4IyPqaEnH6hAVC3qe34ovP"
    "e9oY5f/icz9D/vsv/srfS+IhofAkSd/LwO9BCjXTGnV1JQ2IpjpdW2BLBgU/SZJRXPBnT1jbRs9ftu0rVyASNrp0xl9MAgvr"
    "Rj51K9b16An/zbfoD/eDSLh4z634Rk7fExH8Bg72mDiG4gJt/MoX9F5IyG/5YEg+cySlpyi+hUHQ5XYDrIGpnAKZgsl+IXfs"
    "J2B5pv2SeXCt/DYY21pf2Hrn9h09pGzYpCddF2ZgOJ/5ezkuVqt4aYQllQ8fU/fe7PWr9QsFdxmcjPBw+Fn6VeAzpsRIrFwG"
    "N6Q+X9fxH+Aweyv517Lfrq/6fSn99H3JbZ7SMImvIbuQLPMJbyV/Cwn91v4eFu4P8uSkT/rRnfOnpz/Ic3/lBPL5dSBR0Ija"
    "pF/7swEPHzovqaapHOxKGQb4f7mEAxOLoXPrbf/tW0p6G4q108AmULGwUIb7PSrZhvUDq9L2e6lDUXxq2QEliZ0VusMbNbMb"
    "U988kn3n/0BBRbjDC0lrz06tawKsPTu+lqZRtRanE27xpHyGdOR7pT8IkLPuHGGBBH6cICL51Ve9/AoPFr+CuRF0fk7N1I+6"
    "wZpeyN3UhAYIcTrJUyOEAJGQFpLxn0DSyH+B/1lmheHmuH5z1GR+nOBH1wjZSbe/eN84ofsO/PzvDBhOGzTCNRulLO5329Kr"
    "st7u9hbOxfRbmQ7TmM/0wPesvw0H+dJ4gRupb+W3Xv7mV+LnQEf9w+QL1rZpv0gn/8PlmOTBP4dx/lVSpmH6r5KP5Fg1Fxc/"
    "wrXVlffkyyRJRzQUcLS79rTtCIeL+TZPw7Z9Mo2Gj2vxnhuitTIitLg4p152utapHC/oLjpNbbQkztr2Rap5eEuXY1fT5MmX"
    "4Y3QPHx3cuL3CR7KsC+id1wGqv5eaGiT4bh2o739WdcP+E89aQIZ50ruMs2RpdxfZobDwoBfCqSdIJZKktSMXapQ3LGj+0u3"
    "EhlXSZKqkGq4DLl9Dj1u9pcNSGBgJ9CD4SOTJLpfSOzQP4SG/3XanzfR64+cOnx0xroRsGynp3rclFD+tti5P/5M1yAUiwjt"
    "L36jH9wmzix3pZLM1wLH4CsCWYDf7X0HsV8+1fLQLJ8DGS9cLvw7KWbz/qT4J9D07yRxN99Ihotq0rfub/CeBgByIbv5DQhk"
    "hCu7bSMi+Vvoi+O0MuvmUd7jvXyx76fvA1UH6hyxCe+F7Dps+69wyP3+IKMIJdExu7+MeEXPU/jFH5ZszxriYPkbTXjwq6e/"
    "68QP90hh1/G//xkD6xeNb5pvj+M+i3uk1fdlyOxuiyHX+kHeHXGx4r7so7OmKD94BcP7th19I9v2vbv+9Wvf0RwiNEfNwpM1"
    "mm0LbRsJ8Z7dy4pc/e8yfc8dc4cbJU9tGOD3/THW/+znIkoUlmX0JOJE/hhkWEGDYwfHHe6TMN/RYPP+wAl3KZ8y7bdTaCcP"
    "v7BI9zu0UzL+YJfs0bb71310DyQqTt2DZL2+z3MqfjX1ZPdXeXLiljLdBmS4F0h2AO4Nt9MRvwzrylEut7SZhERfCEM4x/4S"
    "z1SfFvbjjygWia6Fj/vTEz5oSjgP8Ht3mPbMX3xq45rC4w+/2l827GDDgXcWEV37EK3FWjt/2vkipnD4SzzQ/K7VjjL6LS1B"
    "GxY+AOICtEXacB0Fgb+1wgk8Qf+uB08mfuqHQdlIEP+NTJL3od/WaQOmtm2l9fJ/4bQKzMnpKRcI/wHCgLlARw+5/i4ZN7j9"
    "LTe47TUk0M5m0cFaWK/B4IoX1IA6agCQgJodYN04fgiHsEENPIrWe7YYMsMS4rDzl0C5zZLVQV3bsF7MpEgbUAiVJDfQBVLN"
    "wI5ckoxj6fec0+0bNu2LSmHYxPkhjT9oW3gytm1HixRQM/pcgCAf1Cyd07w2NwpYkhRm3dOmGq/R7szD/LUbWMaL/tGAotSy"
    "Q1k0clmsVstM5dHr0GyWpT7NwifrVMaFgoaCxGRTLInO92UGarT/5LtG2mU2zAOfdR1utzC/te0V2q8jRAVMkWzbVIE6CRS8"
    "prByyY1l/Dv4hOU2+wZ+5L/g/9brBn616W+YgWWU9qv1v36LVWON7VW0uNSx+qOqIiagJBWX/4Lq/tWmVP1vFvUMLMMfuAEr"
    "Kw6LBaKgFosMNDh8RttsTHXUZt5fffcWr778KAcgDkr3ClQevKx1WVhAvvStrykDG0cv/XDAMs5xkDP8PmhpjiteujdLqDKV"
    "s8lchIpnkzmj6uW1NHeDgdiF7+HVksNudQdFYa72W6ltwzq+VpWV5uEv6R19iggL1/ao40dl6dXHStKrUE7/+b2vX4F48ONF"
    "4Os/6YpefV809uPfw9t/pxmQ7180JWT5eHM82uTBIujlRz9Wuqz2K9kMF9TGf+5fL7nEzISR/FhefEtZ39dKpw+sUJfe8cr3"
    "66PlRTmoTFwVD3VyW+w+2kHk7f0XV/6LXb1bwku4Hgxq73Psmw1V65SZDpXmM259RnqN9YfMP6qrjf0XX2Ae/1mzUWv7QDsx"
    "HVra1Fv5cOfhzUd77yCsD3eRXlInbf0j7N9GrtL7kxa/hKwXtbFy9dD0Ru8wI9aw+gi1Onrf8b2mcXi4ue4ttderhO9RMXqx"
    "PKabEXe3t0PECB4EXAkzs3M48AmeSmgMQ8zXiI4c4IMKT8O9ov1dcTWHQx5xMY9eOboduIMhvNM9XUn71qNj36zTgrEebYbS"
    "PoABrmxa4FPKuGZcjYRY25mdJ8lD8Eag6ZK74cgOcNz6JhRjdxPyR+sDYOFcji+BqhYsk13HUR5yXAom/vuFdHB8FuMFdoh6"
    "wouj7i1C//zloOhP3NWQwsrxqtYSZVIkzLEB2eDRYGvrcEsBe9xP++5BRpImvxF1PNG8FPXMzmEBlH7UPY9SjovdroIjsAiT"
    "1vjjH1nPSuipKy5t8kr4FiD2wXdG0yrD9ek4C25gZWcBGPgkSdJ/42tL3zFiuveipHc1r7jpWwhIFpXuWbbvx3dre3AQ8bw0"
    "GoVQg9GoheYN6DldclrkaSM8oAvkb7wetJOr0FLQ3DWIJRCumyC4iluPyYZxhb3ou/9ZkqQfLfZ4AFQ0BKVQoAGtgfpGI5Bj"
    "RSXLyn4ENkekABlQZVMzZAOnNDpAIIA2BTRTqvCC3rbwdzRhSfJPmwKSk+X4R7ymJ95/yTIVgTld9QDGOIDIiuMFt28LqFn9"
    "cusxVMBAc5BgcYOrfmEBAxlYTR6wRHyxUM3Pe7lk/JouZwCw6WlQhD77odjKJiXSxsaeWRPniLEPJ8rSge5h8KVZMs+o0Xcz"
    "OQ+fLliMhkRWe5GSRFkEvBGber6f1G49M275clMg18Hv4eQksdeX9iPoHCeG9cCMHx08M6B0VHOxQZiosN2VjO+4eC30gqZl"
    "wCEH2ZO/1h6VzJU4KnkaF4YZUZrkixyZ+xkchNTD2Y4yuHLDezV4b4r+DVwxU5Or/EebfW8zlb+y2deWuety2z58AiFE7eFX"
    "GiBoTpJJcM6Vv3c7uoGS0AJEFKJCUhlGLrRyOIWhsXdEMWvxo1xXrm6cj98tQd/o4p0izuvaOkhitgBqC2sJwHfwwfJK2iWQ"
    "0B7WWcOHv9u0DrKxAm7yluV1VtOxMe0RoICoy7+lW5I/ia7TmuUGUmuWfYX/113Hb926eyRvrdSr5tGlfQAg1ux30qSjCdCl"
    "Rt4jtJJ4jRVej28cVnV0tIp8w7+xqaIM31mUG4y+ob+pEq9TxbgGQsMYHxVJ8jtkTpLR7ygf8BMBF2l3amrGR2du2AtvmGBZ"
    "QBNnezqNmjApffv9tPRkFSTmr1MDl3yg1znoYaB+FNugDAA/VCyjFJBL4H5jvOlItBDbZ3j1p2uDoYFSoSkPfBGoRpJoX6+T"
    "WfQ8MZwvHUJb+zp8mT7ZF7RACGFYbX6JIT2yjOuuvtHfETA4Fh9ERwbeub28JbOWcV9X/2nX8Yv/ZCWdRSspIMvPHhzC8LLj"
    "z4ns31p+4w4Ay9/5NADv8JcWLef4C/wb7cMheeiJ+ZtAMV3nwxGN561b4EFmb4obXtAyrQWwurxENUsQNxA2/kjskSQ1rygf"
    "YK2X8LxHoxGqha+Ezl/azOa/2uw36zWhNkmUTYtoKqpcW5yBJh1y4nvGD8R8omKHrH0kBxYTtDt7x3kHdjc/zpK5hDKfrVKJ"
    "x/wKxSZsjjBdn6/rYnHmh1jqFI9cxOdCr90KBkix7IWVOS1nhFtEhT6z8bo+kGzwEB9WbiIU7lWcD82mQBVgvxYMh005Mgp3"
    "jf6poFP6sCmarO7ECyAvvBEWRl+60VfrtCaGTPUY9UdNaryYcpCjuJcD7z8jAbJW/0nHAR4COODUc95xT/yES2z5awBi93MO"
    "0+0wxJz2cIy99GNhcBBwSPq9T0NikYQRH+i3ffSkGEmQVa79iaQDsaBfKpLLRGQf+VW4YvpG1OI1bImj8Rx0rj7qWdELmQmq"
    "7Xhhwxu8yqR0DHbBgKkB1Xh+KFar7INNkQIz3kiLT0ChGafVhglu4TGOKw6TnKC8y6CM3prJcW3cwKJwI4m3B9QV49WBjirJ"
    "cpX1py0MEDUZlh6o7iUuwcELiS/U/RcKzn2N0KSC8QI0CnSoFLCwkMUDgz2LHPPBti2emW37Df1J3VkZmGzqAy5vBWsuK3CR"
    "GyBXXhQA5gx12wovI47SqaBGFFHW0FHfDE7nYkPnonHnouSQlFEKdgHzUg+ClcRhwCJTQzU21GBDyTjHXYegdhU1NEpnjnkz"
    "uekTHV/Ha2F7u4zoQC2QE/enqQwSJrDXoAUSxAO+gaAUc4ZFBB112SIbLzqjaVX5Igd/4JDtGJ8R7e9lOr2I+liOHvaEEecH"
    "PTNz8QaGFYVrUFogle8HpNIRTn+dI8mAOFcPsO75SGbqAY49HyR7Pj23fSpw57nJYhZ377gkBXZCJtcZ1Yz3mtcWKXj23gL7"
    "OALb8x+GSWes42+jpDPK9fVQ4fdq+Pj98PHHwWN/uv+B55PTQkmnTKKTc5nhAwpYlpmbzgkpe14XO/f6QvqMruzoCd+5D59M"
    "V3Jd7CvrE86iA+2r+JT8xjGdP8FkjSb8ueWvLf861uX8HOf3+d5Z/oPlr+J8397Pd8ZvLH9r+fdxvp8iWRGhDMjwXXrCjajU"
    "2G7HQcN9Frgttm1v3PwSOLxGXVYyNnB1W1JFuszCvyycAdUfNr3AqzIcoQiUH3zd4DS+NfUtMH+Q4UlustjeiohQA/x2D865"
    "P76Q5HvHsmOzJNe9voRvbGzgeJQZd0aU+buPZvY3nT7vP+OmyRzBbdSqLAbwvO5JYyiu5wse2RxO9SxS3/0tbgNcL6KLfpCo"
    "Hk1UkrwZZISt5ySNvyG7jNfGr+jayH/tk9xNst9Wvw97dWzvNeja3+O8j2kxRwP0eCgVDTdlefNIanxF6n/9p6gigBKRHr6H"
    "11B7YmOjXlQSrIFMcfMzSV9zmeEhQ+nXIRG1rw+VgeKWR5jxyHQIsAtO9R8XB/cdl18OuUDfCh4xdaEiPPMlnfmSTaXQvqkv"
    "cAQ8VL3vixx0Qg87QY3zMPVYk/BDNFFuHsbulkDeUIgZz9yFSZwPJDa5zH5IjwUfeDEOn6DhXCQymPaV9Zf8vL/awwLN7l/b"
    "zaDVOlbM/RItH09FrKYlZD6yhJCxGloxNNJaaTw4zY3jkVWDW28xQv3+qvtPzCEmX9DTMUL1dPIROwmyeSAufFC/kxCOXPsR"
    "3ip0b5AwwOu3UPHAxOBLliTvnaWKJ8/OKgFJxuhssPKHQH+/8P3cfutHiEA/A+BXBAn+k40SzYgD3NMDSh36dVBoJ312oNVH"
    "hivfCDj7cyMkWLuiQS0HM+ZGgvxD3jxyvJJbJ7UGrz+NHjAYpfbwx1BhpR1GlmtRksRV+23daI8Vsm3baIcVsiAZB1QaWdtE"
    "HALYsIFcSRAvd1DbrVwp8PZT8JWUu6zmtS5l1vBgppBVvNhfgRz7b/VltufABWfrTijyvFPnMnN0o0ZxPv09y1dQ1YRl8Jfx"
    "Lb/iC37N76Afl4LMxWnzp1eo2HI0gN8J0hpmv8Rv0Q79Dmx8UUqQp5ew9u9gdFHFK85/oUbQx9yX6STtB19bL50mWaZaExLF"
    "p+8cQ7GJ0tb5OkWLX0DgMJZdCZxnna9zqMS/zNBzk1qnC9JKIOJuEQHm/oaAOacP09NSi21sGEPVfMZn13Oo6rr/sNRgDAF4"
    "6CSp/Ua44lbgiJ/lk0/PsnpKXYbhhuOAIHG3Amy1+AV5ldo6ozl+myS3Y2+aWaa33qyJbwnt1SSJ9vXoqRap9ywTNVc6NzK+"
    "kReIBoRvnovLnNwdeOEo6jeqtNYsqzX5IoKB2nogfJKk27HDvjuFcVjiW7J7Qdhw29Zte9e26WVuaeLdZR5OqOcoQ89ADvGc"
    "BfF1qaf9ZCTJImUcB57ezix/LoSotRcRXSbJc1BBYdpsnj3n1/Ppc2E5zI7mn8HtTzvFezQ9ktPkEhrNNbmXJeyTZJ/eML4l"
    "qJ9MrxjfRrZmFS6kKr0BniW7IZdSuO9HE75lfCG2zhKJJpIa7zb/lsYGR23NYA197jCxRz4mQC7XaH9j3YIsVedFfgMCzey5"
    "H+isyqv0hjTSiL4iUswy955fkL2v2NJfyrmFdDJxg4LwR/QG1p244BeRlpqo2uTTM69FegoGCYMbA10HUvDA4B0wpDqoZM7Y"
    "ubBDpYnuXSrY01OSuzOoydEWdKnwoMMXA95Betcg5uQEv0MfEvFHO6wahpvJgcOalTsWWNdnfo7ohyPvFoNi+3f1+tGDGka6"
    "soOMit1zELZzeV41L4M7Ju9EDXQLcVVdcA8UpmCte/1KtLFNLklajPQu2HpsQwf7EnZRCZ5oOr4rKjySjNwhA6bGrlSgST6f"
    "4zjgmg5ah4E7JYVWCgPHKHLgGIWOOmgO+EShavt6oxZv6QA0XMH5UYRDvBY61+NrcDlGmMODNKY23xZ6BWdgQ9COn/QGE1Yv"
    "4eUrmKTVHgvOyg5MqzTonZ+DX8JbOyafWW2Lkt5Az7QzrEX50w54R16L5cbaXZN9+un1Xr5vxrW5+hTrPzVyLY3Upfz0/zF7"
    "bdVWnj4+mG5JRl+x4ZsssZokGcGFLPhNAwujAoH6YPzriD6LTbHOwD45mCqTmxkNmnfJFa8Z6egdRhistaj5MEGNU7+vddpw"
    "HLXJGZ/hZ3NStXtscbeBcTe85oqX0a1so3sXZGcwJxNcSMpb3E2Bh64rOcbR8DrsK3SVuEC7umv8fafR1we/BFOKWy3emnqr"
    "GvBB1dTVNR6BlKNnq57r/nZ0odv2NgB9ZG7RQR8ylB6cQwadEtAqvcuVqIiFPgEG7SqMJE3QUz0NKoUTfX5+PgF/XyBqA3WC"
    "Bg1l8VS2bUEGScrb7+ZWmJNJpoXp4f+h3nfeE0FkmxYcj71GfgtaMguNOZ3MpyPdtn3+J2BUeg6ZNcuvdDDMvtLef+ONBiEp"
    "6M94bxTGX+oBzBweDzB64lbTmP2go+l9ERqKe/lOg8xBrcgc8S5UdQljBzVloTttm16HVkUNOD7ZcPXKePWGz4AdmGJ7eyVc"
    "v9xgvmgXTM3TftaM9/SlaZbcntJh2NC+ARLUCmA9471ata0NsHOQcME7PJrAQanSezntR9XwCepnc3M6yRaa697QMNWRFTmH"
    "kza8RNl9ZGIea8b8dYRcQVz7nnj+5xo6e4aIAnAUFdFRcIilnoYP0B3WgQjstZ6p+TQtxirY+uSk2C1YNirGChwdKej7U83y"
    "az0zJydzUWTOcqxwbK4S4IjGA2gkm4+b2ljPs+FCPX0Nyww5mNASYfidjo27z6byqQruOU5O2J2bY9BsBhoD+e+0UEgEppf6"
    "6V3o2mVw4CbFnZ5d6vk0svCMTDqf9IaKX7Yt0I3oXRdTGbIWeQbXwJ6mvCdJEvrk0OwQEUBHpfvhdPymH5xr4DFSxqFAMn17"
    "o9NgHMc/hAfNI54bq6eWvNbkyhV2F7o9Cjs9P51kcEOAd5EQBremNwpawJAtdLQRFtGQXenZAoYsNob9MrIWvT+IePrLMVho"
    "qnzy12zyeT+Snw+MaJ+AbMj3B7fjv9GMwfcdHQThG3EGwyUDyedpeNW2/f5IEhwCGr23LuvXx6fEq4jEvw0nxFvg+L/WJENF"
    "VmyxwPP11aptsaCIWr9FRvetdnzuyCJ7q4cCaSMI6CbOD2a8WCXJVZGeTjyE7hXS4kKsC9d6XuMaqYWkzwYjSKMV8k7leTGV"
    "p6dsW6Rs+grOHB7qmATGsO56DdwCnSeZ8aJ0f1f4N/IHoElRqNbpW4BOof/Vow79VqRvUb8p4WrXgBu1+XS4qXW8qWGUZwWv"
    "ecNLYcGdo5xPAR8FBAXgk+LgeJqs4HvyHZsVHTlEA+O8lU5r0DIiaTislMkKrnRjC5BoeM+kNa+rFYovsx74nTV8W6/UWknT"
    "ZGXHHuKWf9T30EgIdQbZ5GXv/3bJ5HiB7Yx8S0aZyOntMvBtSWJd/imZpy2qwsrGomcWV5Du/UtiScElbrCncx9N+7pDOZ6I"
    "IO7Jcl9bMAb2yhSYI9BoQoPgd5iqUpxNy6dhNZV+Q1aimJXzaZ1XYz+kop6VcyfLHbQ0SfpM0VmZVqJJDx2v+OFoVnpgbgWt"
    "gZN4nyQpsps7ne655l/ymRzLildccev4zZg3+aMnxm8LP95vC+C5r9VKNtzgk3OsE36G99PeXUL01REy3QAwF5ws2lgPd09E"
    "KL4uUu9YcaH8AlqofKHGi9LdF3wVGSibnfcd3Flta8alzA3dT6IrBij5j9OiYo575PGIZJTn/Mr6SxLsNtCkedBqcLQaIJLg"
    "r9qCyRjp7KFyvL8AT4xU9GftAUmwdpbXp015u2T8W41Si690+rOOEFE/6aFlGJIUpJ2xPbw+gk19E66hvVLtXws119W+2WQl"
    "CTcrEF7uaeUpxESR00idJIBbX+5qMENaw2j8XOD6KQX6+9ZlsE/4FkRSa9hmixu4y0p3VSQyF7braOU/ILfuvb0CSkTAP4ET"
    "msBPkqsYLol92om3xXSPoy2CLgSv3DuSHkCmLTAg2D5sdr6PxEpoyzlWYsd/U/BVkoCta7OTugHIPHYWewXyN6Dw8cfe07vN"
    "wQkkXjk6xvfjXhiM5evYacTnjG/j5ydQZ4pMyQ64RY6tYazDpm+EE0fvw1X9Z6D363xNNHzDshUsOsb4JlKL6iH63Fnjw0Va"
    "kdtocGnl7LOW4yXL/6kRYYICWjOT88zd6QwAHIspruoCnFmSe1G88oOKzHvj/LFwuoVGfKMBXeM+ZxG6vE4HquJ/6hgwYX0g"
    "gHHw/hvkeHBOPHwqJolztyfA0bGcBz2W225/08H+eHGN4KPfHBMIAPZ3spIQJ4H/GmWrZHEtn18uGf89SpXaSgOp/eb8uw4a"
    "l4NqnDN3cPWtmu9lca30FT04OYN7rujVzz/UQG2czC4s+hcmxV6DIxRXIrglYPxZ9KIvkN6RSvCxFrOvXds4ypbnXBpgCFbS"
    "OeLkxW4H0Kuv6hp8S/AdKHwaaLVPqTW5mH8J3c0eQ0L089naSnP86jnwDlUlV5RG38MASEqIfuL3x68e/P4ZNZS+in9DCfdf"
    "xmV03BpxgEAY2fKropHvTKEbRXELQBrYZNKAInQPJn+Hpqptk9nOL0M4hLgRMLfHq1AhlAmRJUlSmTQ8pSj8JYtufwfLtUkV"
    "oPea/eU7I2X+bcqis2VUxD4kasJU0VQ1nUCEuRmHdRSQ5yYtvCVKg7+hqPL+60oUJi15zQ1svxL575EQ+yJJSnhRec4htC9J"
    "GpOGJywYJPLxl6O7It3zkiXJPs4aQPAm3fsa4TyARzSAgngPe3uq9DJJoqYEg4S+n0Bf5bgIq0R4ddEgz4SD/OB9fRmuWaQN"
    "cRqaXgwRnSqD/AClJOYYLPeB3mpOxBvPpOgl6QEexW3ie+/HhNN4YweVPq33dtjBHL3SFXeuK/50cmtJmdQAVVczZ7SyRzc3"
    "c7HncvarnlO7YW3Ro+NRXYsqKlqujts0fOE6n7J/9V2XhZ84Bvsk8UrWYD030PSbCCUzO5tPI1/j55HXOYrxguLSaOYPVmjn"
    "L7MX3SEuwAhrIjcRJpwQhyO62QkJ/h7I+B+Ljh02p+ah4BecdB+UHaTf8XWtMIO7y8GRSsVxUxYRraxhCQ4JZRPoZDkkk9V9"
    "Krk/IpLrQCN3QxK5vU8hN8cE8qqnj4sj8nj9AHW864Tll8FMClccv4W1iB6TLwJHI5NkR+Lo/0Ur4udHUW3sbDKfXrhANMRP"
    "OPt9OFtpHTwVYLcG6rIoBRM6fuPOpcG48st+SD202ogGiZzuD0Qn/jbiqm0buj960TvslCTBP+g/lC5ut7PLuTMkuisQEtxb"
    "nsoKMk8pnoBb0Hvc15LueZShAzFGyrqL1HAQmqK3A2qlWqdQAQinY5puRMkLUfFa7D/agUXbQqbrtoV8d227H3SHUHijydTO"
    "focgLeL8AI6WGyCTF6nM66zAxvCbwa5OkuEzqLqxAH+dxfatBCbGSswJm5r8OfaQr+aAIgBreizC9lf6weJBoSmUA6P+O406"
    "1wMeJdD5lE0v0jW22E0Mdg2Imxbnh6Jt0wJlKKCKSHW+ybbUPRsTwNuZmpMiwJE0SHB9qgVmPeoTfiH5Ln+e7qBr9ZxlNXSt"
    "rGod8F9KRBSgxzgeH0Y3UQoMdP8EJaqHKf5NlJzHDz40gI4+HpDufzWzHVe9Cv0m8n1inFbhzsSIDnFTkK6g3KhqZaQT6UUi"
    "nMZ/OBp8+RsoiJHKUkQt+hqYnP4JYInkuKze7motte2RJCHJsw1o+dFsip38uiquMst9KeTCjbSB6PVo8kW4XbuDxiafPYE7"
    "j/YMWGim7lmySNZQmiDlkONQZ/IFdiW09QD+fDyrKGxwDni/6aVJf9PuQGPEOIF5BhBKf6mN65k8+TIK0+LulpG+Ezzlqebb"
    "u5UpYL/keryTeqX01Vem0OUm004jSY8g26UOjf36ddXKcdM8p3hvg574fP1rWAFN83VRVRCb5GOZ/fu4V3FWTitCCPEqSVbx"
    "QvDdvVFRKu5FjY3HoernpzIBveWJ/gxM887+TJ/t3TWBMrsR2kn0a6BKPuyAZifuFya7LEVWE+mqQ/PztB7vQETh5ytJ1MkJ"
    "N8J4vzeVASsj3xPLG3AClFrwoNSzNSjtgUt541uT3RRpzQ/QpgZEpkD3zydseJc1sYTXkLN41xZx+uSBYCt7E5yI3dtqe5M+"
    "sGQJ/hL66w+6waQlyf3pCpjnvcFICMQhhrIGfHxERFZDbzwIBWtSupPJMfzpIGAWXsIy2Q3Q1Ove8AMFgkHhnMr7wIGxWr01"
    "cq1u23Z5vWQny9PlCYhSmtnZ3P2YzE9Osjg+zg5HD1+K2VHmJ/OTEyiEn/GzeYSIMJGkIvhGTVMtHsSjvHAxHxwsGuC0ehxH"
    "ByMV/cY8bChxZRz+ouC11/YTzkMeG2hR1iRJV6CjBkRCBs7O4MuA7YDpujaoHwB3ZkVEmf46eQIBV5x2pgA514+EBlhB+r1l"
    "1KckSd/Kh5bbgIOJq/w8/61Io09YVqCAWtQ5bJms4QeVrbnJdsDhbwkwYvhCrMdGrtHxX+5+HrqMfvFrsR7jYrqwEELpTrxO"
    "ryFc4DVmVxlEQ9madMGBXO/TO+B1b0OcODC3wLdOybod4e5Nku1IiB0O/wJIPL9Kt4wtZtu5U3GmW9C5XPuEHtT0OyDOenDg"
    "bbrlcvyBIYTP4anxPiLHH5IkXczk+IMrAp0pb9IdY2ud7viaT57wWcUX88iBHAR+2wF84Xeb7ogDa1vtuRh/ZZbjdRCt5Zfp"
    "juXXs908W8B/t+mOte1Ijj/kO6eKoEagKQhboQl4iXZlERRMM6Z7qWGDFolE7iL/kgwqELNmzqFOHJ7dXEDioAezZj69TXd+"
    "WHbe78rRmMAxgUXbPMWCq0GxFWMZrP4HSjoe3YoRbLPytI/GKWV8Y3oDtW5qQRB7OoHUgJP7TSH/Sy1ZGLTTSwcIg0V0Hd6Y"
    "4MEQnWEGb+JxRawDvPbYyH/uZWNfrSr53J22HF+UeGWL06cYk8xQZB852LPf18VKGn5nSLLpXqnmOyl3zyp1HVmLXDpKdmHQ"
    "ZH45UCTcDl6ujt5eOPZYeE2RIdH+qgSzwF5Uq3vcFTocVc0LMvIpIs+pEUgqYGVpgm4MOmTRjEfsEWV10TmAbXMJDJhUr8JB"
    "LQ9LkucgWQE+Hjhd4V9Hs/XcHKspsVK47ZIc78aky/12yaFXZYremxQKsotgHxAhnPpxoSiRETRXoYYWoCmg/wLEOYjPF4sb"
    "uYFBQwihOCcoNOX/sQBKU4gd+KXR0Kb+MuJ9sxS9GjpXY+/aqmCZ8tgSXpD4+x0uidTNmjg/gPIA/aw0u2XbYtMJqSvOnXMs"
    "UBvxl0a8M+nycgsBNek3/HzjkvdLxj/Qb/j5zCdDnvcuHX6/pt/Nbsn4D/Tb2Ksl42/9QxkL1L8mPhDWF8yALJ0PM+jJKyOW"
    "4dBolvx7I5YrZSRyxs2yL+THIe/xrUlfgbUmWIEAaBYL+8McKeH0atCQr0xkx4LqklDMBArJZNv+ETFkP8f5vzXp9+g3qbcN"
    "9CsELrdB4anEW922bwuKZOlItcLdO3XhyV55Nyji1yIljoygVRS8jPwbvEPjcPilZPqOwoD565DTXP9kEAbUtrAMcRwgJWKn"
    "KNlzNkWSgOVoETshH44qNmFm520rZ1Al/vDVz2Pd5PFW4wUe6QDuq4O7l7ptrwKQVYs6SdDgAAeL/EMBlTfCeY1qBJpfcUle"
    "pQA/cYSlj+OKQgDnEKTyaY2BKhUEvbWpyZvcOfWEeJXoYhN/ZRh1NABuiyQpIDlchCIXZRrdnCzpzkw+T8aqeaWtJDfEbQvR"
    "BhGdi+dfaCs7Dn4atwyRkveqH4bQxFP+nt9PpuIAi5IjCAV2t7xfYHQkazEMTDj17dT3xtTgmPZwwKd1DwmsBSECwU4axrKe"
    "o0oAKzTzgP7H4eh15inaVSvANPcaa7fihoNkHwg3ChGMp8ihmB6sTqhHE6Me5QwoOV4/5uhxa7zWTt6Gy9n4VzAGuSeL7lQz"
    "ECQAkyJz8xSHiz6AWKoZ5HoIGPP10EEcHBB5DqZNb/W4lG37VgeQxbVJwxNLkvB7XIaYIYVnUHV0rUVVvaa7ZgHezrgSD8aa"
    "PA8ifYBkfuIEJp+gEhrDx1vGv08Z/yN9RViD2c/pEtRkQI0B0ZAyNucqP32SffE56/w13M6ndZLU40UJ3noQDjXxuqoeZ/W9"
    "O+gqxGYRGayT5DuT1uhnqBHYp7Yt2ha1aR/k3bQS0BbsGZjdjBZpw/ImWy4eH2y3ZCcphLQ0+XKxvsQY0B3HwJw5hI+BM1eh"
    "GS9EEsu/+DwDOFxA7D+MBSt7LFgIa+LAY8d9PPPTDW2oxg7hBrCdMYzZBT03YhbenSxPm+Wc8SpS18fnR2+uNboukICM+qv7"
    "vgBkXC92GX03kMAxNLjcV1HZ/zQDDy7dcSRRPbNJ8ikG7f50DGgoCLS8rHWGGPqsAPzOXICNRNiwpFknbjP/qkAnxfB/9ree"
    "G2NkIvCbEcfucejqc3icoXOgx7LCHw6jIyv+eFXYgtLgF3/slLaQgD/548Ja41LwJ39MalwXn7y2DX8MF8PMRR9fN/wxNQtS"
    "4lbyx6au+1R4YPzxpm4sfVtCC7fKPcEv/rjGaI5YuFJwDj1e16Z0qDXKuPYsMagVnTqRdfwx2I2+U+UHygWY8OeapNXS4YT4"
    "Y8TKQIbv/CsAlfyKaBOg5Bi4jEKhgGEyShou4PILV1v08/G7IU89h0UmOwdB1A85oKM4iYfS3mZOJY4X6ExxnIPC6ctrXpSl"
    "bJrnEG0N7DzudjIrec8+AFyIJJqzszlwl48DCKhxxLmHtLGhC4hHkyzAq/QcfT48euKTipD0uU8yIekzn1TPdA8n+tUMXQXB"
    "eTrhWLbPUtAI7tPiOOMTXsQZ92l9nOMzXsc5jC/KHGf8nJuQ8TEcMJjqwIQr8Ru85Wu+w5MrdrG3pOW9RBc9foGTxyTwqO95"
    "pnQtyvFiUTbNa4whjYHQ1mKN57o3l5z+O00Eky1ReYkamTz3rl/5Pt1FH+5grNELjltdXl+KK0bFq6jgsLDqKAr3rwaHPE9h"
    "lAXetjLlm6fwjerf7FO/4VG8TiaHy8dLOPcokuCEKfTkMMnSOnxHnpkOi4yaJOMm6cFKNthAFS/kfsnjIm+6jpcOytnL/cys"
    "BIaXGl6GJU/7r2Rt+yuIuOEHLCD6q9zf34z7UXx0wOE9oD6PZ7ealaB4HPjmvTuC/qFC3HEE4AqGHNwsWW+tTR84Yz+kxZkc"
    "L8bRuMzsXJzx3i/eA9XFUIC/x4eX0wX3UXotxee1TmZGYXmdE4PHFHzcFyRVL0QhwqqFDKg1OS7t7fQxXgnsmDTFz/FESRKt"
    "0mESl3x5CVe6sDIL7mNpZDXfSrupV03WcCK2JXfwz6ziSsNhBUbcWNIqW/MIupztuAdYb3mMVs6uAt56wYO4I7vjq174kV26"
    "T17Ixpr6Lrv1RZCyNLvgK3ojV9lzvvf2adkNN1KvpMmkdL/egWW/XGW2T0A3EnKVacnREux5sYN49KvsHW+kuZYGJOUSumsk"
    "l7e7upHZS670Rhpln+GJqiTv79hZAbHs/Q07e+H87DdZLTthHUDIqnQP5vp4pje9YgN4iyZY4ANudoO+D/EeKqzT0JJGpPD5"
    "Cu+JS7MpBV1OiQcQX6FbTsiMa4bXw5pqX0INF9lC/JKicwwHxaW6QM26SVFUD4lXMn6Bp182ojZuKKAI5GpCLpaZjk0fco99"
    "lyou+UEG89JsdMZjaT88o68OJs4Lt/GgOhkeheywc+WwWyUzKi2hSwqlWZH8UopNWrG8ciPGsmp63xFk0BZYcX74AwRKaPDV"
    "sW6Ne2YN22Qg+HhDvOKKBi76HMLI+TljGfgnSoeT+CZ9afiO8TfpCwPm2G/SN4Zfwd8Phi/g76Xhd/D31vBL+Pu14e/g71vD"
    "rYQfP0AwM/jxzPAL+Pve8Bv4+9pwA14P05d06X05sN8CID8t51XbAoP7cmB0/PCUgbTv4CcFpSIwI5Z+g2ozns0uxFAZ1tRJ"
    "lEXSBoT4K7hg3aMEKbH0NzUA+EY7TShwZ0dWOv2OEwU4kqFVHzaeeMG4ATgOmnaGmbLKyZQMTBcFbReVQn9OPfgKOXwvrAWG"
    "Q00huDxw+O7et0Q3YV/p1KDAANTYxivR8RSOX9ET4+gkJv/YyOp/dzOo4WZQ/WbI4E4vYqNkrfxJt9OpO2Wc4wpvfeo4aJYd"
    "JRxB4Y06lv6a+1hoeOvWBhk4xvIpS/QMlCS/aORbAveJ5t2YKMP2mN6T2qzu2cZHjYqEMga8aHi0NfTY/Qw0yc5CGrTI+Bah"
    "/DEaPRUdrXR95IetulUaPP45H7Hg9c+GZMXdNYdYpcJNYnag5NfSXMkLC74jr5QE/52diPWzHLR+iM1jvIwcpYsmG6mA3QZz"
    "gpHJS8CVlyAO6V+peHQKlZZc8hoWJOP4ZEHDyemUKAjWx0vGIw/shertpl3sxNA33+UCTjIQTkHegqOXI67i2i3WTgVB7b1p"
    "KGwuy9DC2CQJ+uQj4rBkfsvVaqZA6goCTzWfSgAigVgCfnALuKUM/u8FRujzRgliWRrlmNG14nDzxB+edVmpnqFZKR4zP9le"
    "BeZlrwbcy14F9iW8cfzLXgUGJrzyfEqf1zEqexVxKvCljort+R3MNngcciZ7dcya7FXMgaxUzIGslGPXdirwa43yDFupIp60"
    "UQNxtc1lHjzi9gEEcLfmMnb4TxbsPDbiiV5Y1mU2BiuUw3pWKkUSzCuF/EqPcFEOsdH7a7CxLMZJgGPhpp0BwZ4j2e6NGu6L"
    "FvfDJsg8tieezT2ABYEI89gyf3X84cOCGsgTf7Y+/oxOH2QZBlWDsBTD1IDb4ofLxqsD/G/z/ABug6J6diqAtkYyxNkmZ+ZD"
    "M0r9ERGTZENZl2UoZoYBQ48fIJDuZVo92EOFJQIgYJJjeeqnmh8KWI3viqtMAXZ1XZstGk+OJvz4JpcdOv4wuTx0fOA7w8nl"
    "bwqjj5JgN6hKmjdO6HToOu7I2Gwe75VDF++VQ8eDRdtDYzMg7rFzLaQ495OR/txLJiXjBpFaAb0yXB9h22F4Pqj7sPHucklk"
    "DTBszyaNrvEMS11WpxOFGQG9kav4QlpeA0KshLtgJVCRJQ6LvVplG8BtLcKwZJoviIYajjaEhdLS0JR6m8JM8UUwgcU3zg9a"
    "9rhAV9E09z3dUO7mjmIQ/1ayQ8chqP1RaKaCnMBjeAq4WgAf2NiiqkAKMiavxzwkphV9DVcV5On6LOEVSI5pCcRu4se0KHpO"
    "BqoMqd5JRNX1S+aIQqYq4kSJCa5AozxIjNbY/c/75Rd9Pkjs6PRJ0YKZtraTroAN66KUiO5v25/TGIS7j/gKodCgZHSWN2LZ"
    "XF8tM3ycYCwY7+K3ThKb23TPIVoQ/AEfpWimXY37VSAKXoALpb1cFLvdYiEq/luR7iNoUtfxWrsjD8a6Dt7i/IGXsgOoAnYa"
    "guKMwzLiky9gxnAxxVVGZhh9YtwGFrbtYHhVME91I9txs9e/KLtxwxL5WFHThRLVdOCFrLd9Xyiwwg34gYp28EKRdOZaeVCU"
    "JcvklazQ4njZtn3KKcmUgAGHx9fBKlzOlqCRCQlLVAsvHx9ALfxA8suj5Cisi0J1Jbn9RzBT5K4nRtmDXIg0B7jN29byQhhU"
    "Tza2MLaBQUqXxNlkS4zHlCTXCrh2J0T8H8amGEsIXHFu0Yje+KuFM5WEFylGOeTNmBS+mK/AfA1EhCRNF1CjWSkKmWrQibvf"
    "7+AWMJ+OqiSp0aDbpb/EdMYrNLaouORf8MJbZqlZebJ8o0u5xPvH3p2EqIOAQQi/gI/oAbouEUSFITB6nwZ7AE3DqSqa/Uv1"
    "MA7xVt0z0db5pcrsuD8NILhZ5GNY9VL+EE6GIAjUCrCeP3RIuPEk771hkSUuHeeuYjDgAOUw7tyGRPHdFO4M1tO1YATqE4Kz"
    "biCqjssHlWV4gHTKC1f3e191PQRiVOZ4YwNUr3O3TkZEKDNNwUl8XsS3FBA+0bmVQWsLuJ7En9eM1zGeanBQjmTbjgpgokCa"
    "DfAoWp1PwO857j1cC7nN7NjIXVWUMv0Ukh5/imoBDGsyO5uPbf19fQP2Ro1M2YkNknIUOkv+kuLbUBSUqDXP/Q3xgIJvy52f"
    "L4536MzwG2U36MY1c9cSz5EgABvVbw0nxVyJ7EJWeanl3v0g/mHlbzWkZNrF8vktSuQ3Q6nkFSBBFwLdjIB7TKRt5Mmuh5V6"
    "sZhqWwOe4qfX4iVSc+eQnq/4mm/5jm8YOMksu3DJlsJS5t5+LZfpmh98Z3znqFMgvU7XtAqgJEuUJy+zdyotWed0y4BcKHpv"
    "K1uUvoFjo2vxc7ovSA10Ka5hG9wlydVIhG0mB3r7O8ZjO4lOXE5lv+6T/wFChDGoQWlcA89wJ16q9A5jFlyCwcclv+Pk3D1A"
    "EzS6r0iS1GVwtjJ4/b3Ed4L+5PTH3yLoOzBrxb9cRwYBiOz/TaeXkV3EJWvby0EuGIBL/kqnC8avIxCdIhOAMw+y/hPvVUGe"
    "RXH0rgvAfZCVaAClozsIb+MixPL6VK2XDpnn6CL4Detic6uBevudCgTJHgvUSJmKDoxd5OVlY+8qOCUBvQdzAJBccegYiLEQ"
    "UdRf4fhLdWRMd18pXyMbbNzm/V9Qc1lEERpSxUf3lp5ev1CR0f/Bs73Bria6WYPcQ/KDV3WFLA0PlgfgrM/yCizkgG67zY5g"
    "Y+eqJjYy6fXJ5AusPHe+tspkcvbk88HrMrLoMfkblYLLuwocw9f0+ss+5PzqThdbVcKVqvmXS0KHJfFBpWSInCSjC5VWvf4S"
    "QhfFk56qtkXN6ahp21EzftxYBOjnozP0zlIDOTb5qG7bYUtdcZFz9Tdq6HMh3sQUhcujkUbiYWROP0xD/2PmAfdjBgRKcITu"
    "0aCxbT8opDCF67Pug+GMzrr7rf1w1FqSDhOWKlZG08KGMGawIq9B7ju6AR9timUYWCeKFaTSA50akju0heXe3inTHQhc77lk"
    "9DZVRM9dZtTwut8D4ycy/ItyBtiI0PAfBLUG5DAaY6awgggUTO960DExTGSBTC4pH6oklG0Ig/peAfPymjD2w1v1e8X4D0gy"
    "/iQK3nvVb9a396STj7D0QrxWKZs6fvYFCeKbB62ZX1ERgCjr9wbSj3F8QEP8KEwGUBTEr/Qcmq9E6NzkKvsZhPqZg4/Tmaay"
    "gjsEgoiip349lKA7OqLc6V841uHQ0xKUDvOGYnUBGzpsIa8cPwiuf+rzM1jCaQ1kAmlInUT2czQnD9IF8k97302hxySuU2C5"
    "BvQMhMjBA6G3iq0pPCvWDRiRmjHwY1jM6jkctrN6LlbQ5LOBRcO7tAZjfjsX34N8GoKHrggw7AjOR8pw9OgQzSbeEbz5sumN"
    "4EipCrjDfWpx06dGvIQYV1A0wA/ATAGSy5xQlcU8MOSAoDD9I+lii9DawmNDJTIKmbuoQhaKHjIS4ki/WzBLdaJ0w31QgC6N"
    "OlZRPB6PYGkcgqVfR6+UdzQbHJvdWxoY4ZXctrLBIidYk5SpPZrCSoAdFF9NC4L4rBAhzZCGj+peQLJiuZmt5qLK0tId1vh0"
    "f5EAYJp2UJJUgLNG5DPtpQpaeIYq0LoPXwVBAUUJbm2Hh1Z979BqKB6wmTU4FaCDaLgC+xnJRwAbacBhl5dYR97l1EMxh6XD"
    "89QjJ2ujgxRgKEHh5zHkJjha9Nlqr/ijoGqOmfI+ZcB8bAzosa8LCNdwh9IsFlEAT6yAu8AooppWiREBVHUIzoIUm1K6cPJ3"
    "59aKg2W2O6EBUzYu0eVrKceLBknqDmVDXT07mxPrO5I50NCsnk3mKDoUYrlsW/j70nFjgBnomV8KbfvjR269fyh0e6ACWS6E"
    "yn9UmRn3olFeO+2Wg6/3kUEJ7+pxSxVQ9DWIKnf3b7wY0+WwEwTCm1mu58JVDnu/SXE4dJI4T0/kxHCKWNfj+685vsnawf3X"
    "/pv3X4vreFQmyWjHQqxSGVRsiC7UhLUu2ZEZa3nsqPCRFu8QvsCmXymKKlohur1Xl94DPARiSiP7FepRXGGIhDCigm23Qhky"
    "SJJzurDqLnPiZBRJoIqzQEcdgtjOVaqOWxz8pZ6c9MAjqEQLQoJYBERPPXzSOUlasgP4KXAOTPANmRkvMa40UHzsXyE2zpU3"
    "wqqjAqYGwGwFN7PJXNQ8LeBCbqItysKsW+cnditmFV8H3mw4K1vGt5FrPRWjiSN42khK7x76Z+RVsFELFydzuSjtrf/t+OAl"
    "/xYzRgr3lwXLZnBbZnP+k4qcHODVoPfZaZ0XjLepR9J/q1JnYxQ7BSMPmhNuOv6LOvKSgxZmAPs6RjSA2zlYUkPKrwL9g4Fn"
    "iCH4CVhhBdJkv+bUkDZ+q4AWYV4Mh9F1Hf/m+IYGTQd+CUUBnkpift3x7+JmH2mcwDH/6GcFN1SyMZgL1Dt1/J8PdRbLJw4Q"
    "pXnE7fRijs+e9HejxVTm6XeKbN2QULyBNbRY+iP8l9hs0SbJNyRw6vjfsOo+BJDj2pWTdODlsEahG+G81IMt0NgCkyQSkfWw"
    "8LPv6AQD7EQtRtpfp/gv7kIC+H7nZClJUmyQhsSDj9g26RxFPeI0YAxRpgAyNqcsijiOjv+mRFn0VPzXeA/8rgbQGXh0LMej"
    "QoC55RQUAT//9HKxcNT4oHQjjc1qbuS2vpb+ZgznT1Y67frLSoKXv6xyz+9AmeRxg8/rLb5cgXgL36zhl/sGE3bulgQ6j2zL"
    "ARt+oS4r8BO3gbzOdiC7EoZTe0BIpkrnxCFbwDxd0yqCAwtP0IL+1PQnBDwq6bkSo1G4VT/3VgPOYBhMyoLQmNx9uZhWSlwC"
    "5eAbiTCJmqKVOSNmG7krEOL0CXGvANI8qifkP0pPEoRR3U89TsJw7RoYgvR+4eSH9H46hfn5SEmey6YzBOSUa5ixIG5bAz7E"
    "4dX3Dq9eFdmdH3NGhwDh0fdFdvngi1WRyTy/RbtWYOjid6/AwsJ9xJ1nX3DPRln8tlgnk/zmgWzZOvkif/HQi3SdfPF5267R"
    "+wg4ctuZGhC+9/PyC8m6ldfAFvmVSVfcAdvWvOZgZshHlmUrv/H8S//NFZpOSBKK8proT8fvRBxqjFYYLoE6BbdKYp/a3nKF"
    "3O5E5miYBW7f0z4XepMvwwKBaDNREV3HLwdVSu+Gg+pbRfW1LQjNkVKFijrnE8B/Tq5z5RhkArWZi0XsHYNY70EO1oEDrQMY"
    "tHBKyWw3iLTWGzADdnzKFEZJ4rWjxlwKNa1TF1cNfG4dFRYOpOOCNBXUkFWknoLECZxsxXcEmu9wUjsbIlSi5rVTpvbJ28Ju"
    "iJ/Bn9tqCSXT/FmZHpcazxwMCqwQCRrR5ue9fP4S3INVqFxH4T2KXBaX8krptygtB4ijv/66EoPKkjJLvXJZ4YC2cti1hu/9"
    "2ICc30kkttFO3vBeWpldgRq7QQKKrCE2WVTOzIoX4KF1O1YQ64lvki/zHeyIyM/RJpl8kSTv4vWwosUPzfk7gskYNoovkuTH"
    "4LXX8KUDZsHik1AqeOIhOs8bWALbyCYNjr4tQ3NtUrl6Fg6cM8C3WOgWMc+we1wu+IpyuASXz6EtC8bTNQZL+hmO/a96eBhL"
    "kg9Fuib04HHDB/67kd26FrJIFb9i0+v8ahz7blux7CpJRlfj4NwNXJTMmgIUkLDeV8SzgBELDrVri/eP1rbXbbtgSfKbIq+n"
    "6IM/ah2/TpKrsXSV3Rvk4Pw7LKIO3Bh3XMfrxm0FnSRXxIeYP7MehTwGfZcN7bUjASao6XTb1kXvGwpEld7rEpls68izEp64"
    "4eZB3N5U006wYV3YgSUhOG8n2zXYCO8e2OLiLGZEK1FOq15aXvnLWSnkrJqLJn9RpPCLZS/dj+k1XcbLYcFAX40UXmTjEt2u"
    "c/JXpIeR4G/Pjw7ebEVbb90JPd236J0h+F36wnlrjpT5GwoARA9XKERPksfA9I4mjKdXYjNcxwRmpIV8BSoAWCxrXB7oTIEf"
    "uZjnobgzxjEQ0D2uAwwuj9mHIMdf9aES0r04I3HVynE66XastJbm23evv0+STf/gXZZvxwDecMsDckSPjtYmyS6tyPArf5ne"
    "awlf8UB0NEbPZlnTtogf0bwaECV6jwO3Pyf1zB6ErAqwRVu+ATvt/pq0T57AzkQt10iIDf0CulJxp/ui0t0bKHwPUS4wA6kN"
    "IIAb/OAb9xfzfNnfIP4/6He4ElvQVdRiA5c9EHWB0pe0fN7YCdoA174aCWPX7SF+noz5CB3xETDCOuIjYAhGTZIEhmc4Qmm8"
    "7GghAclaR/Tq6k8WoEPZopkyuNo42sHx3m3E2bTpbd6beO82IGa18GcvyrGs2rYo0hJctUMQgDLYCLctMPLAE7ZtGXul+98v"
    "Wb5N9yzT02t4Hx9hjscHqhnveE8xR0IQLgd+2WO57AiFshjHGoOQ43xgH2H66JYCHPLgM/MxcW6NpvpwcUQTcxBDJ4mNz0Qq"
    "vMHJVqzrT0FD78Ip6A5AEw7CruPFPU4ChsrzEo4LJfZln8I2XAvr+D1Md7wZvYso3+4e5dvymIxnG7habBDeUualV7BvWLYJ"
    "jFaewkmJnGENkAP48W7AxM7AScU6ajjLdudnSbJLvvg8Sbb/5rXKoRSE2Hrn1elDpGaLWwprYpxcH4wcIVNJAsdbOA1BdlZ4"
    "EA8oE9wYx03t+IuP8KfDA080YTzs0HdZrsAmcOwB5o4pBYvL7M2AQy1Z1lATStbxNw/sONptkTGO+KEgOTCe9+gLEzFiUCGh"
    "WaQRF5Lxb4u0AehEDdgwdMf0QpLTMLQjMPIKGCDzQu7ShpdgxFAwDjiuKji49aNGyJDpZer2IfJIcoxgn01dgVWRgXsKSVBK"
    "mTacOkLdADzqPZmSjboUdQ86FfACDjzhGw82GeOBMzh2CJckuO5H7Rm0jlfSSaO8/zdDkZItN71PbRJH/Vvlucgdwna8lB+d"
    "td7jWeRhFy71UHdm+eU+03yfGa+UbpyCsgT2350spnASZwhAjXIEWSFZhT5JZ22iwxBRHLfI6ZXzzT9wMk4LYODLM0dXYopl"
    "FcRhmTrnYOLzL7l70fXhAAkGaPlq+lhRpHZu8wcblllRYtxbgB3ydOUxSSDAxh8f55FWqKgsGac6nM5xLRAOBtGLA38bfok1"
    "v0bXdNt058+dHWCNJHilk3hOcGzmGpq5F/4UfQZVQBpyXb8pBBH8WXP92RrO1bjBivUQrprDRblxwDJErhwut1nFt9nez/qK"
    "o6eHNRl2O++C18ZBM8IQVziMFeOjLd5+S3Dp/rGrUs1XoKPxY8ebJLmRbLgk/KjRiN7gpvWJOFrk92+6hcgjzqEb+YKV+XEC"
    "bXiWWbeNDmtUj61RPbYpIEr9qr75sa5tynwymccgEb8AHgxd5dGY5MF1Go6JhwU4VeKw5f5SUJPcI5rlGkgSWQP/htbAD46d"
    "35hBfj3t5xSGUeKMpjGFf/LXL9p2hR5xVsG52+qeTPzJX79Ap8QF1i/HwATwYfwMboUWZFPQgcN/ivMJtkBBYecCuLvYww3r"
    "c63X3meN8K48RAlxfskKr2R8L3DnU+qr9QsI0uxfTvdjJSTfg1tBjITIyziADBPn73S6j9ZQRU63I0I+IOFTryqIUM9T6ekk"
    "l0RwKVAb0nenikSPen+jlCC00IxjtKg3CK3E8Mx8JR+6Uo4m/U1Pxqw0dD6PpiM747Chw+uIJVpH4pitMyFfu4sIySkPO2C0"
    "V8eypd6P6Rpn+7D+s2zdNvkyx5tNktxBRlyqgL4AnHWq+YqxDF7nW/jvwSoz/yGawKb75Ev6FJi8rRP+HH/DOr6W4gFZLJAB"
    "CM8MklShHbzbX29wtNzPtXhd2M14qwChvQLyS0iw6e7perrrr0EWfV/CxR1+4cUdf0yvUznbzbFdjs32bej256v8rhffA8ey"
    "Ztm79FgYvAYThI91g5p+NmyzDAFU+U6sTifU5P1TsU6S/VOxm/a3tz1BzPah9Xvf+j3JVe7AzZ1i7JqUdse9iGBb0/3JSffx"
    "mtZU00fG6T+raX16ynenp0jjztd0b34KjmGJku0wqK18usrBLhguCioMwW7KHN38WK/xitjXy6FbQT2+P9+x0McpA/ULjKHX"
    "wOxPTnq561bs+Ubs+ZWPA46N2IsNtWPfx4j86ATInqUH8RrqliGJ7wmuvOB34oxfit3p5mTCb0HUcSHO+HPnKe6SuSrPpvun"
    "l1jjc6jqzCVvqRv7+D6/R3Xt3bm4ZIeNTLXv29TfASlIKsEDQ+uYEleIv9DkyTvEu12IzXQB3V1QjObns8XpBtzGnGHoB83t"
    "bDEHflcsnPo+it4a15+lz2cKPt2fTLg6Fxf5hVDZLZwl11iMmt9fNPzu5IRG6kbc5rpIn8Plmtp10++Svbg8nUz35zBOp6fB"
    "n8PJnpNpOVwxZ/JkAqgReTJ5usrXns9as0xNcVCFOMvd0oLtvot1P7fg9vbpWdvuR0LczBbzfAtdg1xPWLY4PUXR4TYm9DSs"
    "Th0sq6zwTp4iqXnVA4X3ESlfeSl68gUDv2X33UH7oArBlTAS+94X/hhUrOm9TF98zkp6RSVcyP5tSW7M67QYBoknYMc+hqJs"
    "ZbpH/5ZUfp36Szv5BYhLXBXswp27Aw9PT5JkBRKkim5KMPisioXbowLiTuS+OVlaDeThIHdzr7hjeyonvS6A7+nVJxjmBjwT"
    "9TEiAFjZB4UpOwE8tWMu4TYaWUblGHzYV9XxVe+XF2KdLEIQpbYdUZOncSo4aXSRU/iwezLfpywD701Y4D5lHJi7FBxjTFUO"
    "Wn84sTBiCI1o34hNv86AYqielyCta+9yCbSvTb/IynsilCpadvsIq+6FyrwEdNYrvZK34KhHbmtgNHB1phhR+0nbVuCmz6l/"
    "SbcNbH8THHUgQ3RlfLR2TdpMipzJd15yi0OPfJC9HaFcP7XjyMoFTh5vjrgHpgUuNzBbvT18H13DR/qDJbbmV2J0jXbRC/Q7"
    "AXtZoNfB4SWO5pzuIQsOIEy+T75g14MdGGlVSdoLTFa08bwZY2Ru2W1iZYoNAnPKucRaPs8dTJjQEWF/gjoVrHirSFMPwNFX"
    "bbsCsRTu6Rx5K/SnMCFSW7gQEslnX37etgBwQ7k0sm8l7RqTJFcIXiLFU3DR7iZk+sA49faCbQtRjSPx7CIeteP+Bl8GS1Qz"
    "pYQux0sarucr6S1DvKkUqDa9ktbEFFPBLbNA861XLgDRSLHDQg4G3GmOVgU7PEff4pAr3n5qEHNGDWLOxJ72a1HkGBsik7yk"
    "rV/kW0TeuSqzBiQE/1EF6JQbxzWSHSfJURksIl4yJl4WAolqvCRyuu9I9I02BQtdzUtolPFSISAofCGH4Cun6/4zRfdxeLXD"
    "5X4LPp3gApcp/r6+zAruDqOs5vstBDMGD4bOcSEKzlWRkmSilojfxfCjYHgOhtq9E3eZ1tyJzbIa3YBGMVH8taxPGnwAAoLf"
    "VAqIYncNOI7pSOEep/vgZ5zjnRo6eXefjvJiqOurRTGte81JfXJCPGM9DzrPjl/S+j0OhuNxe5fyTwKIDIPa9A5LPT2By6e7"
    "LVv0XUO7AugliH0olvPsb5GNSb5BPzYdv43n8L0TrvWC3wUObJJswPEUPTgCjaZrZyy7Di9cfHEsLWSh/0BmQZmE7ADl4h99"
    "sHk5CAqUx0MB/gVeFyC44xdSHHbZNaykjeTbbCu5ya4k39rsDd+W2Tu+K7OV5LvLMnvJdXYpeQ3eEp9LfhO8IaokSWeYMhcq"
    "vQBY5MFZTN5KviGRT/ZcOkTZs90uu1LpreTPZRw95+8qJUIk3UFqu9ghoMd4EE5vCbRcXWmyJ/HWaw7h4bIUWtcANKv16S2m"
    "Uhw0XdYrZBL635HfpI0FiIgfxcidxmOVHki4kkncidQ8nfchcD97wm0fHjeLItd/9pl/k4jTv0YYPlnExrMpWM9K9Ik6ELcy"
    "bPvI9nQqcqBV3LNujqQaKpJheN/S4OTkHobY3EM9W2HICx2Giy6GZHNU3FfIpGnRa8GfirO2LQaQus+eUGh3dJ33okAXehDR"
    "XVaonmIcTKKjT0aEwrMFGolATqcLrAr0KDGA600GRVP2UHDRux3GpgOuJ+3fx+65itjBFJktwn6fnc0HcqRAm2AUyWCn7C11"
    "SvIzDNz4SDgLQgihoMPlac7Bd9LTkh0sugznLgSJiW6MUHQx8FF+OpkWT+spa0RxUp+fT7ic6Vkznz8t80I0J5OsFs20fAqp"
    "xRzNCoBbSbEKPStOJ3OIYF7MwRDMFe+Lhlowx7Q4PT0/mzLMV0OUeDAkuu9exgxH6jiwDuJIAoY61snYoU4mt5kBe/HIyVfh"
    "w0KxP1GlS3R87o6zyFtW+DjWwPQkPkqc+sY/3OiiSKNj417cs97r+9DqLi8ANhOnkAtp8olV9EGbL1xJvCn68MwruZbGyBXe"
    "t2KPhqUjFEjDhroYBLVbJ/ltgpGFREhgnChZ9kJ75Pyro9AR6ytgiqviKNneIq98nFxuIXl1nNxAIAy+LtBUhCC8fRe+TyUS"
    "qXVBzdkVQlKwpFkcagHclWMe5Fp2hVgXs2CCfjqZ06lIHnsKMYk89hQu/Bo7bIoTIbl8CirkAolnuitisGwU8qYY+Cw/ht1u"
    "ivOzfFe0rSafOtA6vsF9BSXviuCTJvJl9ePQpskVvijSrwYiaTTm6Q12IkRX9MnP0SfDD64HLc9hVV0DUBFw2OhbJosNZe+K"
    "ob8rTxJJaQLcAvrgF6gPJ38ahUjRWb0EwKbMca74LaTCDRc8tq4X8N7ir3VtMg05H4gqgV4dl8sT2WMCcOKv0L3P7xb/oBez"
    "g8ream4yyT9klq+z0Uh3mYxjmuAkExdEfyCOQ0CNUxy8swwNZijohqfJh2h8wH2j9yIOv4eMB8deJcklECa8ztskucUHB2bL"
    "vtZDGAQhzULsx8hGnQDswXwYnzyUzj96IB09oxCArnH9HYxYQQKf+ksa5SkMOKKMfl+Ax5g44VmUvXHYe3B/dxbJIYpIDmF4"
    "DGgCN+pHAgxqRO9umnpvb7O3uuu9JKZvALADYYAoFqEca/AyVqk/JJhdYcirCMrTCgh1mH+Zgdch2mKjmrZZWvan/TkwFglc"
    "q8sBrwCxNP12RA+KaB8kLiJriovizxePC3rk8Mh/QABmCZEBOW41z17dBERIdAi8KdAvuW934dpt4qtJvitmu2KsQK6DttRz"
    "YTLfZAzjPIiYyA1cqX8vvFdU3OD/L3Fv3t82jqyN/v3Op7Dz+qhJC5K5aKVMc9JZutNJJ5nYyUy3JJs0Bdns0KTCxbEj6Xz2"
    "+6vCQlCSncycc++dJaYAEACBQgGo5akwfLcoch23AvcZkCSDvgfTNcAIB9UohB4t0PXhir1bMNM2F0UPAOeCTmxfCvy7z8Kk"
    "qThmBTNzc2cUYXHZlQwxhEwnxTgJpjVwIJaqDpEowInL6yDUl2c5MiyHyuxSxu4UnJY68/pS8NiKa0J1521U/Rq/DgBnSXc2"
    "A0J8rQ6+CjSm8OGGRauSdA0dQllWYDwBBmjeR4BQWK5BLR2R8geYBbOxZu3FyDFi5Bix5BggSZuDB3PgcZQdbh2FrER3xgHB"
    "p6mDf+Buz7mLYkytMhlat6BVv4gxAdpmDzVGIBLxV50jiCz2s8YbaFv5pfAJ5fKsDC8gKQtH21deiidyz+w56Qr+qbOTutv7"
    "Fm/ZNthSGE0NmxX5JGIM16dUYb6qDFHy4OoUp3BiJaxsowG0pYaZVTi0GlFWFpQhZolyvnTqRkDgAgHSo0AMPZfeA9tE+SsJ"
    "oWshVaW/4FQsfkjOGjYaGaLUlCTkcW5LXSdltTbOAthv93xSwO2Dv/ZJiwMiJApqWC/cm/nJqQLCY7FzSvGKrjuflF/V6y9q"
    "K1kIlpUzwCX3FvXQLMzB4+onEZRHXrJ0B5ksmnFDU6KvIky3evJ5Xj+z0Vja5tD69dEEiQ3Izz3qYMhoBeYkUOPJGCo4UcZl"
    "7cJ2me0YSgTJQtcT1+yNNsJoFa7rp0wgwRQsjV63QggRPumIaMBC7yQ8vBDBziQa2mYpOUYV/HSZuLY1ErZwF6P9rNHYfxsx"
    "fFfwLXXfJ4j0YkOspUYDgi8xL08WMKhoX7gAyNW+cC2ijNLKRWAbJSbXNboqy+5j16RfY1EZdhWVt2NBLvjmT6Cb3DRORFiH"
    "IxTI2jUYM1K44zNkerqTuIOREsIbrD6Uw4By1/zI7jM/hDer4MCMah4omc6dA7nVt15Io/BMGIWrO+WYPxOeO1UQsGkFKaMr"
    "m+WYP5OM/Z1WbwQbuNvg3Q1ylajRAONjEENrbCNWYpNGuo7KeDfxKgRcUPc7kRMJkk/Ew37Kd3t4IxJI72Bh7ItUaFLiBygg"
    "tZ8DKZJlPsWLBBP6ZJwQcPOFs9TTgCGY/hWoOKkgSRRL8S8FXAXtThVwcuGT5GqJl7QfYOn6avUUogUvQUb5V9BsEgHEg3tu"
    "JIzgknqsE7SIY6c6eRoVYnLGZZjgDp85hiI+gxyPlUYRO1gcvKOoN+QCTMxk0HGsIgabr/54X+VW2HLiPYZcC1/8AIxtIDN0"
    "Es1YUfg79lnw5lqgF3ZiVTDo6tC54vIgwHNFJyqMu28AGBnoRIUFce6URP5hDGhRqUBiYhQbkTZq4QBwLysYHl4hL1MMTqfg"
    "Xtl4WcSfFVwe/1G7TcjdOpOPr2ZO5mVC+vFq5hhEyJX4jUQVLAGSsTSeYz+kcoL9VOK5QsJlyO8x7M/lDb/x858lJx/+Rybz"
    "h1nA+8BnqbgSD7w+yv/mrLNyN09h73eXF066JilG7nKBCJCmU5IihIt7H3GsfLRdBLvAkEn4ASuPpLg+33Of6ZcBqmPeBxCP"
    "j7wKyJtgJABUQJpZuJpqWs37oTEkOgbGg4FuwOOYB4IH+233ZJlVyIKZiiyP0SsdAIOALXY9ehW4heYzV/RXb0/Pnr599uLi"
    "9MXZ2YsPpxfg2A+9c6lO3igFT08/bJT5BGUYyElQodi57wMxdK8Chjpc2UESYa1a2TySVyheXJNvbFyW74NG432wWYSbr1Zy"
    "hvoZY9Nas8Oul3BnrN75VUiguDi+aDTeyCvaNpDdWujZSOBic6P3DDclAGvcL+jeT5LVqhC2rIH3kZ1auI2rEsLxTaDBQSJV"
    "Ns3aAYex3xFVAyTtROPikYrQ+gp5mcYOp39m7DNYsPsM/N7gjKBXQX9RPKqsY1chl9/wVIfkibGBIWyIO080MME1yFgGyJrq"
    "JHXvtQD1dBgkGHYcCKxK2zkEzGY2BTyMCEnxoBIwM/ZvAfkGxrxSgMuSE/dkCaNj6Oiz+0+2LxWVTyUbuvUaolUXPMoMx7k0"
    "BJoUAgW08zz7wKKiCHhhzZ/c2c/2W61W68QHq3cqRd1uwHZgbBBmlP38hU2LElo2EAh2iNYv417nefYqiaOEsiY9tXnYNTPx"
    "KEMMVRzVTRI4xRFsTA0a8kugwuVFFVmw2lYrN5KP2YgvOJyvEc4zjCBlMBR89HCSCnRnhhXxWgqzdfIlYPH7ahDQ8LOKAgcG"
    "A68DjzrAcdRoXL8py48jmKoUybC4voB3MouYyMMlshiHLMAhjw1VIEPgGzdYuCzXakiRPzbuEqycJ56wQbB3lY0nifYbAyLC"
    "kqDtxAAhIkAhA0wRK5MDuWDqH1IPgVHzEIAbUdVVBHJEhuE4XX9ka4BpZSuy6vK/Agkyyt/iASBmUb6Ig/u3wQ3oERGux2F/"
    "Viu8oV9cwA8FQkT9eAbV7kshE7iesVAe/xT0HrBj4adAEcj+Q5IUEMdVoLV4vNTMDbKrEmBCcollx7FyXNe1PEa3+xhx4RYh"
    "gz8J8xXYiJxPjNXJxEJ3tOzE9hIeGHghcAnZFZLBb8kmwUSQ34duAyHhAgRT8km4C0kavgo0k4NIHwSub7e77U7XJzQVsCNF"
    "6vI73tcomaVfj/3SbzTYc7vISlDJnt0vKAMWTXVcJqlbpJy3vk/jKLzX/Fvw31uyNHCgxfiZax7HdYkdSFKXph6ki5ehIBrD"
    "QJzRLHX966JYOEdHX79+bX+122l2dWQZhnEE2nES7cw3h8PBERhl4z+/v/FJIL9oloY4YvBNnvjBWHWaukHaaASiJxz1RfML"
    "erOImeNxnroCZUY19WdJzMYLJpBpbfS1AKKptnPhvAGSthGXmkERFP/gGqnD1KjQFoyJFRL/YbOnb0+1LAWJYqGYB+wqFWGp"
    "ZCsL7r5RDnoNZztLyn/ZdZBC2DIfZTNZ+6aMi2gRy8BZEXBnOCRHl2VBNV/k+6QqC3GjVQweiBgkGoUEGCHY9erIPGopnoam"
    "RRyuR9gggfB9RhEFH/zQNjB8qlKq63axVqF9WORa+bsG9MPiwFYJ5EtJs/tTHBIQdmEfa2m8j1yYyXggrQ9SQVgQ5G3MoF2Y"
    "geySRW+jtMxFt4p2HOTMaWjEPBk1MJSGoMxqb5kxxGiDaiMmW8NRh/0fLM4YQKUWufX3R9w4MU0V5/gkZSB+SJn+cX57dXKw"
    "pOtjWKgnPjInSZL+MTzxfHz0HSrihqRpO2TfPeJmu1gnwwaUsCYCOLw9jzLxzczkTE3RMcoITWZ8eSl5+ijaXHqbI0JkVPBx"
    "6qXqEDiFUhP5zlxMIRxp6vqV0NYncQoWO9FNwH6WaaUUv2Va5dRd4o7GxEpMIsCfwzx3mPEQh6YjQjS1b6zJrMywVmfMi7Pw"
    "9oQdf6cELZhfZunNM9Si8EKY+hSv0tvpZ2ktDYY02K6CJe+og2VsVIJWh1t1YOqOKjC9XsOaLFKmxKEZmac6uUldAPTXaicD"
    "1z9Thl1g5y6AAeoQ6XTJ7+mgyv1HoGUZuQDBEsDsk+sUTwJwNXRPlhxNTwkdQDUe9gUup+x5Ta5SFmGbFxdRwSuYejjiiGdH"
    "vU9hwzukfjziN/yZp2CyXMGTo+VHmLNoKjUMP0Y8ievf+ox2soowok0aCFz/YJmsW5jcgrCB/jZBpLVCzOTDr9NHWCtSpP4W"
    "qcQQv2WLTko33SCSmRtuUsicV47JvJNb5LKoFRKdrFHPTa1IkfqgE7t2b1OAOb5yrxuNa7B/umBP5pQs0+Tnyi3AuSVpwp7u"
    "xdMzCB8TxxgrNk2YCe2deKoyT4mo6Sl+rPPMhcr4j6/uvfxRvUOpewmesQp2Etu5LmitZTcjd4BIVXgzJ9TFc+mkiDGYaBWM"
    "Dn9d+g/ABR6Lz/lrN/zvQidwQIFXz4CmVTlK5FLvKwwAChlw94Kr1+g61SIyLkg61clpysxB7wAci3qxE+jkkj1jH69gxFer"
    "rym6s12BL7xe6dO1YmPgYXFcp9otQUO6y5TBNeFf8KKvDy0v/EwpHPO/JRZmM3jGxAhiCs5w5+PzJjZpdaQMfvVHDElWYnTJ"
    "x25rSjztkg/juxRFRhr+JTy1GiBa8+bQxGRc8sm4SrU7Nk4gLbnAszwwJ+2OjCkIEdZbdAifD74KYKyv3OmvU+0SR2S9TWnV"
    "K8bGK5TKd+oEDa8kDGIw1U5ZobVqspMK6zSuWhIGBKBfEpE1+QZ7Dx4+OH46wWdcnvpU3MohTZxAxwWB63Nl7JMq97qQ1hRf"
    "OIhgPtDOF3FUaEeTvHlUi7GLV0XUebyJ8gIjdwHzB6e9FACj8a/LA+TpOi9QtXD377bA3S8KXQqSoAWuKWPwmKBFStp59I3K"
    "DnC/FDUUDf/sLyXNi6fiMPEyC24oo6vdWRSWGdy9nqWqZgPJq6jrMAC/7dXMbTafpYSLE+FAKTIgRA/zNEj2axMMp/HohqZl"
    "gZHjK6zGlBQsnbv9M+V7uHbP2CAi2LIEZs6kC3fa9Gky8wlzY+WLho3ji1uaFDCuNKGZBq6wJACWNXcLvJoxlT+LDtBslifg"
    "3w4OTyOlk8w96pjnkLxpwnIOZrPtyhWtc6pKHPnt+IoWz3iAT+a3T3WCQZI0CFGM+IWcTHyy5yPqseYfLMN0/Ry8L3wQFMoU"
    "vl/7ICV8kTK1Rc6y4+qFsEqpXojhBYR8KZmYfOYaZA52pa7rhqmXoqls6YYpmbkpmbvC5JBdHuPUi3mJGErEZO6GsoQ24y7P"
    "wR1EUIM2ZieGl57EXpg6ccqu03O39ErWmqjcEXU4HLBg4bISjcbR5FLzHDwiQ3DGFcTa0zyHrA70o3ZB80ITwyJiFvt6u0i5"
    "0lVqjgUkqCCzmUJmcxA/nYkWnIXCQV7wueQ3CNbJCsJJp2g3y0JyynuwHAM4+SHKMt8hn6daoTefpxoe0mqq/LRuPeD6QVmk"
    "vmc47JCuCQMBg7RMXYaM8olPIMaxfmhSW1HqK7VpYBKZfk1o9lwINYR0Q29fprN7kP3ntPiVRlfXqi403YwoArwIAqWB4sQb"
    "FxjObeqM8Y/e/iuNEs3f83WUZqJto1iJyn2f65ydxNu45AoMsgJOwfiMh/SCqVzVW1AKnlx/qSn5ta+T36sUXydvU/dI85zz"
    "1Uif5If82D/JD50jJWbH5hdyrXWEBnhoB8fj34CXF9pELzF0dKF4DeyBwYNYuSNf37SaNzAaJLN08x1f13n0uRHgX3neqxQh"
    "duCCL72epaoeIpxSUYiyYAtquGKQtAKp8M/z0fZ/X0J+YJDilM8F1hFBHc7PKWpU9q8YKLoH3s8iPmmqr1a8aKoYQ/DvB6A1"
    "ccHOxr+n0xFGVmm6/giNWjO4cpwhYgUJ3LcpW5+JqKhgQLMbJMHtB0ZPU7zPgFiBwU8luMU9Tadu4GXi6ubA7NLxXxh4Q6a6"
    "fpImEKkZyeVl6h5N8sP96GaRZkWQFAfKrL+Ss442LImuJ3JLTtwTmV3ZLCSeh4ZBtRCErZav6y/lFzJyVqOmyzX6MkURvy97"
    "4wOJb5SuRQn/xve8B6t/AR5t32kBnR2YMPdN6o79f9LLzxGoG35Pv/nEv8n9KfmQ4n1Sml7XNq4PKQ+/JDHgEy7PPhMBkMCo"
    "Yh7FBZo281hXvCy+7WajzI3A/27DVOVNumWrErlv0nEybWYoodpRV7TDYONn5WxSWXQHV285Qv/Zi3+dPf3w4ilafkDC12hW"
    "XIvwl9fI8gBg8Ar9gcB5ikWT+PSQ+Hh4dBdHyWdfUXamqhTOZee3ZQYCXJVe8DXH173kIeb49lT7lBIR4a9HxCajc3LZLAiO"
    "6YmwMwOHnnvKKGVzhRWbFYAjkef7zgWU5xslVKbsRx9T1Qyfu7T6UqgnRlCRkvr6MpFe16g433jFS1LWiurjCpEk+XwJyFUJ"
    "BAiuxv77D+9++fDi9NRHj6XKQa1VhWUPMJzi+7NX795CFNOr2rbCMSThgOVQjhQYudUkCIDk8JqGny/TO9/z08R3fN+RAzMC"
    "+WIEKiEJRYinb1abG+kk8XYOO9z8WJmk7tfLNxYe2oR3RnyOMBnEODKuajiYuDjFjjR1YjJQFqwC7Mp8H4H+mcZHcTpI0H0K"
    "DI9RIwOTkwhlS7qLKUegbVd0s8oy23H85QjXlXa1VnzXaZy/gZpSdSenkers8yXdRLTgIP90/JrfwF6nUwj1AxFcOLcCj189"
    "5XOTSbY6Tkg+df8FB7BR5v2TYXTDOy7NGXJZzhyAf2FZAIbL8vn9irHS3+BcAd4zq/dBnke3dPUsWBRlRvWDI/JH6h6dp4nj"
    "aZ7zQBklpEEl1CPcKVoD04EbVL7/lkKEif0/+A5A9ZHOQhASWjtacHegZGxOBbsgxRh/1iKHAlKzuCTTsQWAI74Dq4VVZevO"
    "C3nEtOAIN8XP/TN1DfKP1H2fpTdRDnOJRkWaTg6YgOfPdLVytX+kFYIevKKT50EBmpavcACvvDzzmikG81reh0VS5Dr741Yv"
    "KlFwIefYTUDdDQYbtZDBmcuBQIVfZ2XqAR7fr25u6CwC3WOWLoIrvAaNHs5iV8iEKVHRqOYCii64IzfbqDLpERm4YLxY39si"
    "GcZnv3q72ueoG2H4JAwSDOTdReRS/NhFgqfBLpOYSLcKTsyUVCPgHqTgkImTVOTMmy68DrJn6Yw+LTQD4tOZJkfJlckmSzY2"
    "ki39ZNjbSjs2LZskuQofJvGhQzcS6he8OPLTu/eZiaJCrmlkRzvvvUDEd1C9DWtwtVIWd647GgS7hVfavqeEyzV1YuqOyDvf"
    "zANOJ0IghLruaXzbwnDBfFup7xmrVQ1rFn8g66cz8ZMpMOnMX6343p4RDABXvYhbsgDDhwMrI2wwNgDXRQHz9yYNZhha+mj8"
    "tPXnlF9Xodw+HDh03cP+nqFpMLQA44DbalbSKiz3BfxkGsuMjes8iHNRAE1eLjCFl9FJ1W9dteXiY4V8GY2WeBzJ/R/Y3bkp"
    "BrjY53hvAGs4Xe7Z+YLGMQ6keHmWBVdXGHRI1AYXbNSW8wS44YZpljHPdVZNkMxgE0Y3iuoY9+rlh6e/vxCF4JIunuMoL7ZK"
    "v33/8Uy2er+gmwXkqVAMgCm/Y+f5UF7pNk8qr37/RZT99Or5i3fix7Onbz89PRW/Tt99/PDsRdWYPMeygURvr33TYeOr+P/W"
    "uSWjLAYQCGwzqTqvnMp51cwyJMoZkleie4nUN52x4JWZ7qhBSxN9uwAylyB3x35YZLFP/Pw6msMNIojh3xtaBP6UpLm7BCbH"
    "dN/wpLBT8FzN6C3Xz9M2/8HNaDVQfcdzlsXEcSwkSJmBdv2M+fBA41BkHwzwsvg1vSfYEZ6Gz5AYxCIpiDEBOshT4BGSYsre"
    "8y/LokgTn1E0bbOf4KRObqLZLKaPFjJJBoTxaBmL0LsglBYFQc7GN3FP6BjVTK/pvT9FP3DJoEB+T/JcKmNwd5Q+29K48ALs"
    "q39PZzk7EIAjBBO+tP3KFW+cTUWEYU1LUFajQoswtINiC+0gc9N8DDby4iyVATyaEMYLyqWixjX0OMxB21oFVEryNclTncR5"
    "xXrKvDKPj6Hj/4q0MGckNstdGVSMLzUo3pY4GSyTLNFqmXn1SY96TJNWsZm7yDny736mnhIAAOKicnC/RgVTZeMXtYUpD+D8"
    "yh9uVtkxgHQDTEfO2IXBRPGDaiXi8zgagZtgPAMyz4HJS2OvvSjJC9CQpPM9bnOCdWwJRG5bYZwGnyGawYaMDCzKW7etYLFA"
    "s0GdBGsCrihymOe5UKyorZ1++oU3yIeEbdyVxw6zhuJlgMPy+oBxqhXVyom6uOFFxbhwCsRko7ukEDhumb44zLDuRkbEndF5"
    "lFAh6CDXMuOKFu++JiKD3RhyciXzKys4cG+vSpKLHUUWPPNV/iIpb2iGQc1u8wqxhvFi74ZtmEsqS4HfY5gm8+iqlL+/ZlEh"
    "nnFPRjspdssi99VyhrWHpm17LP43XFyuchHSHkzzbnNmlCtU+NcsRCt7KZ3vXcOOoV88/I5Yn2tyqXwNN9qEE7ym8UgpyNtx"
    "3cCVMNcSBvYD/mXM/BfoCM3YSVorVlxn6ded5fgxdJYm1MvE9Vh3Nu8OIoNdGQD6e5RrWuKiN0yMsbh0nUMPgT7zFxcPl3s8"
    "4uXeiyxLs+Wa3OUy5uGeiHP4N/EwycTT//HJKSspg+U2GvTE9Q2g7mPXHyrCnGd5pZgUFz/84D0Yw180/+dgtkfzMFhQHni1"
    "VlGFr1E7i7c6A1E24GXnD5Yd9EXZp7zsywfLdrujB3uHC+srsNYKM3CHoftGmI5eh6OK+E9/fvb8xctffn312+s3v7999/4f"
    "H07PPn7657/++DO4DGd0fnUd/fU5vknSxZcsL8rbr3f33wzTsjvdXn8wbF3442I6rTg1XeuawqjO5EDzLei/Oq5r8rHGGdb8"
    "V8ltEEezvcsgp71OmcV7rKgvEKlQzzKP0zTTeoeioqMBANXAgHyMkmLATkBwwEYPd4MErvHd0CSp+zUfwwJm0Xc3aeHB/sHs"
    "BCGDLMnc7Pi4t0pJ1HR7gDk6gItCyx0A9EyzOXWzk5OoYXW7GKQ1AqVa1tDM4+NIb313IIosiMCabO8yKnJl18dJfyE53jyj"
    "9BvVlossuqFl5Pj8wScSs1qmtRZZ6shkn8DWRmcRmC3WikB6i2X4BD64yOsFWJpPZlFwlQU39Uye6JPFbH4b0a+bHVjM5i2W"
    "7pMiyD9fpkE22+hAkH9uYbqPLlZXWbRRQqT65CpIigI7VC+B6S3M8Mk8iulNkARXm52BjBbP8dc6eY4Ibu9yt9ft2j3ymcUq"
    "YBT2PNfb8yiONUOh8qd5Zbj1OUcEJbqBbfo8Z56PBp7BmJfjf73LSaJSd3L0LpeTzFRif+Wsxt9z92kOd+G39aUOmSOIFgud"
    "NfsKyZuj4tjsSvAg/JiuXKZjszt1bavf65ONJfu+dh9hobBHrOviG+3BYTJKlAXGJCLP802X1CbDGFK/MYNvBCSnqZu1isN3"
    "OTpFGLoIy1soWtnXak/Y+NuWMv61CPLP8wqRKpKIVBFDpNqcDJyFZJw1YR7wTzM6LNBDda5lwKRsIcbZZCO2VTGQDMPbNOmI"
    "bn9igoqRFj18l4MHBh948W1AMFW9z/MdeE+t5zmY+eHfpj04xDBKOJXGSHTnBybg8dFfSyEalzU+ryjwfY6XApKsNVVn8FI9"
    "98EUqeY2r2rUg6SbKf2MXGMUQT8j0U+Q8kbTJignm6BGamrBicvGL5q6QSuDTqodApWYElD9P2uvBe21eHvHhmitudVaq97c"
    "B/Xb3+TaX3nt67/l9XmF9QeA0QUARsMuOC6mJ29z8Pfgt3oMcg3Jx7VkITzYNxQlmMJlVDLahm8zRjvpXSgexnTahPW7RRfM"
    "abqVwCC8z7UdEM1Wo/Etx6DKlZzzcSqEA2vrbQ7/wjJpahkMN5qwZk22OCTDA0ns1p5uW/rogeVrHdKpa3W7DXQxx99Nc4of"
    "eHIygE13B5LbJ2UYcUh3DJhtsRZWWNdIAWVU5uPXuivTp1zjxKjq1+rCnY1mCtaK+zLXEn1HV/+pdPUl7yosuI8wNyC+z9zX"
    "OEskcl9rCXmJN1ESuK+1j7kWkS6gVJOU/QyIaejMggh/4z8pscDgURd5IcvLSRfiokijmm+m6WTkm9U1nOrdkJiGoYOJESus"
    "MO5fRMfxzYK9maxd/KCRZB0fYYV1wdaaqYigZx9yDTY6y+z1uuDSV/3s6bpOvtS3vx20gpuh1TU3tkLbrLZCIAw6ts2pa5vi"
    "jAlb8dNcsza3FdvCbaXOUAbITzAwOJzuQN5uouqZTQfMU21Oi/oW+1uNKl4j3bTvCW3f6eRNrhXte1K07/j8vlIyX6mZMOeY"
    "/Tonr3MUhQOWUcEIAF/7Rmj7GyR+AzJ4kzPQphyeMN5K6L7iTzE8ZVUMqOWd81pD5di98xrj0JFvzmtmUVZgVg1I8g+VL+Ln"
    "qWzxX7XvXd45f+WSW907v1c/vqk/CqXYGjGPFLZeha8YRcBho1ZLzlHg9kcBJAatlp6hXvQPlLqBTWc0PTkJ5IxlXuL+xrY5"
    "R8OuFe07pW9F+17pXNH+pvSuaBe17lXwJeoa/nPzBoTYXTVLWEaBFRm3wbadrXatADI9OemzHo/wp2ta/QY+cZHw9m6Ax8Kd"
    "bJk7D1iHyXSFf5rm9Ph4UFEqN/r4hrxks5uRizyGBEBCEfk9B7oCEgQijHRMgFX8MkdHfcZQXuZajj9ipNgAqGqjy0vkEMUu"
    "DgGcDqgJ3wz1apB/hWZf5oBAlQLzWq00aCAmX3KdbGbq3if40WgkDIpP29fMxn//DGljY6rvszjksfsBUnSyvHNicg88r06R"
    "UD2IQyuMy3/UGdKfufYzNP001zqCe3V1+M+IiXlrN77x3/k95AhaSHJ5DZnuvZhZ3a45xMvg3iKNkmJvHkQxne0VYEAVFVEQ"
    "Rzn19Qdu3gfK3vFHrrH/UWX4PkE3i/Ydcge90fhV8JZvjBvTcJO7ji27TyzTJFanS4YWsXpkOCTmgAwGxDI7xOz2iNXpE7Nn"
    "EcuyiNUZsr8Gefy/Zm+q9LwI64co4Nybh6hjGu46RJ3UkuUhymSupuGD9+X38PDxVe2+fLr7jgxmby82rsXPNq/Cz3dcf98/"
    "f7n3afvKexbkn/d+3rzmPg+K4Jetq+0v8LxxkX0ZxXTv9+rymoVCEPkdiSnXfYMBWqWLChU2noVaEhKqe0k4ptNqkBT02Y3i"
    "L+A07L3IoTiTpFRlU4z36cpa9GX+NeKCxTDIqc99dRwubj5YFus9vir2onyPZbf9ERa+yoLw4bIRON7ne1hob0GzKJ219z5Q"
    "IOc8BQpL9z5Tutgrc5CtFNdRvncLUNNpIuqnd4soo7PdLcxSmu8labEXprc0q7/PmwmKPU5coCrLKNlLs71Z+jW5yoIZhQ4E"
    "4pW9jMY0yOlsD/Q9EXgzlNkeg+3JuQO26FYR3Czog/3Ko6skAAMUNgYoTBJvfs3S5AqkHbMyLOTrb3Z/EdQqXryJchgl+crb"
    "VLb3md7vCSE5nYFNa+1N3oMHZ+kmiEHFS2Ufy6Sq7cG3WC9FsbY/Eq6PO0ekCIoy3yuTz0n6NWn7zLgnD93x2DTNnt21OsaA"
    "2D1j2O/1O93BlIzNwXDY6fQ7HZP0DGswNPtWd0rGttEZ2pbd6ZvEHvY6nUHHHg4hfWiZhjHs9m1imX3bGna7Hahm2DOHg77Z"
    "s0nHGJg9a9DpW1B71xgM+8ZwaBPb6NoDu2P1oHqr07V7drffGRBraPd7fbPbh+qtQd/o92zLMond63V6xrDbM6DZntWxB6Yx"
    "MIjVtzuDgW0PO5BuGt3hoGOYxDR7neGw1+1Asz2jb1ldqz8gpm3ZPdPo96C02bF6gwF0lNjdoQHfiLWYQ6tr9Af2YEA6Rm9g"
    "Dix7YENvzJ5l9AeWAYzftO2eaWJyz+wMBgPTsEnPtgeGbZt96IvVGViW1R0YxO70h/1+Z9CDobEHdtceGtBJq9fr9UybjXzH"
    "sCzL6vT7HQIzYJqmjUPQ69idvtEfEMvumFbP6vdt/KTOwOj3rAGxDKM/MIyhDcn9vmF1u8OBTczOsDscGkPDhC+yOkOza5iW"
    "RcxBt9exTcvu4nx0uzA/Q4vYZr9rmQPTxnkaDntGrzMc9IhlDgfDrjGw4ZusbrdjWcZgYBF7OBz2zaHNOjmwzIHdsTtD0u/1"
    "+jADPUgedq3hsDcwBsTq9nrdYWeA02pbpmGbdg+oyTJs2+4Pu1Detu1et28OhiYxDbvT6fYNC9O7g07XGvRx7+31hp0B/yzT"
    "Hlo9pKZ+d2BbPTZPtj2wOuZg2CVmb9A3+0MbKun1erbRt4wuMc3BwOwPh0gE/b4NnTQtYnY7PaPT7dtIA9aw0+/27T6kW9bA"
    "6HYGOGL2sGcOLGsIs9exB9AWtGn2hl1zYPcNA+bJ7lp9e2jgSA5gjo0ufJLZgekdGEhLMFU9u2sQ0zJ6/e7QRFK1Ot0eELnd"
    "J3anY/T7PQsHvm8bnUF3aJnEtIbGoGd3cCFYA8uwDatjmsQ2YZQ6FpKHbXWHfdsYGAaxu0Z32LV6XSRKu9Pt9zr9PnSnZ5l9"
    "w0Ca7Jo9o9cdmH3gBj3DGNgdXE89w7C71sDoELNjW32r2+/3kFaHnW7fHBpDYnZ6fcM2u7hyrH63Y9l2p0MGXdPsDfsWdLJj"
    "GxaQbYfYpmEMLLvfhU/tGjCTPbNHTLtnW92BOYQR7nWHRs/odnswqUZv0O0O4YsGA3s47A/6fWL3B12ja1g4jsPuwLSHXaAk"
    "2xzYRr+D42XalgWrzxwQe2BafcvuGDhNXZgiy+jZsG5sw+4MsVGz3+kP+na/P4QBsAy7N0D+Zg673W4PFiYxu31YUIaJ0wRj"
    "bnQGJlATrDhraMFAWvihRqdrEatv9oZGxzZwJdg9c9C1OlaHdDpWv98zcHytjjXo2L1Ov0O6Q7s35MNu9bu9vt0xgTH1bdsE"
    "0Ea2boyOYZud/pBYw+HQ7ppdNtu2NbStrjXEr+0OLYONgm0PzW5v2DM7xB5aA3tgDw2DTXfX6vWtvkm6vZ41MPqmicy8Y5iD"
    "fs/oEbvT7Ri9YRfXQsc0Bz3bgOIdwzAsWPrIJcxexxxA582h2THtQbfbYWNpDa2OZQJvNo1u1+ozUh0Cc4R5tQx7OLQNHJlO"
    "z7CHttUbQnLPAlICIhh0O32zb9ukO+h3hr0BLuFB1zI71hCJd9DrD61BF9mAYfYNu4ef3+t2O0Bl2D2rZxgGMGAL1ixscj1c"
    "2gNg0p0+MGZjOOh2Tbb1dQ0T1kpnQDqg4DNg6eLSNvpmrz80uwT3tH6nx3ZKs9cxLBvo1+r0gN8MzemUhLi99vtDw7b7Bm6A"
    "Q8vusqVgm52OaQMFwNUASK7L6M8wYU+wOhbpWH3T7Hf7yFZw2SCjJ2Z32O0DD0e6tLvDwdA2TaAEs9/tdU3GoGHtWjD8pA8X"
    "KNNEMgbmaXd6dpd0LLNrD4bdDq6SbsfsdIeW1SW21Tds2zKG02l1Ho5D4USiiABtyxsbq4IYKzp1kmP4SU9OTpJVcXxsW62E"
    "FPiLsl9TZ4y/W7YFSb1OKyFUJBQsYVodk8utFqsGtio2ZE1KDbNw049Tg/cMvall+FfaBzeTphYddaxhZ9jrW8PeytBXBjFW"
    "kVLbPFRFzDx4Rk10XBx17d6gbwxNCwRWMAjQDIlc0xocYsGQRrGmFU2zrx9BBCISbF4tIx1CrlCEckFrbNMaCPcm+nUPrkNw"
    "d9KC9mU5n1MICAjFoQbb0qLWAORa9aQOYRBue7kbhuidR92TMUgHCR2b0ynIJljjZs/YFDFHI9p0MexaXflj9lCKGI6tw2Lq"
    "pmCuwhukzcFhoRPMASH0Vl6zozYCcLrHA0ONPwOvarRldvUpSZRfEFJynJFo6sYhWtSbOhkHJJW/BzoZ5ySeuiX73QeHzOw8"
    "OM/JzI3O03P03mT1WTrEp5Q/sO4bco11zcmCmEOdjK/IhUzoQWu35B5rxwSdXLo351fnt+TOvT6/OL8n41PybOrOQk10uqdP"
    "SfUMjUCcwZEsd0qeifx+VbQPJXVSK3VJ7tig0ql7yh5gdJ+B3mpckGTq5jCjbHxyQGsYBzAU+RgYdElm8AiMZE4W8NgR35uP"
    "ge3gl+Zj4HL4jfm4P90gBWWSxik4F4iR6uhkfEnuqoSB6DtP6Jg6+eqm55fnp4RSNz+/O38GOA7zxs35f88bVwDIsGhcn//3"
    "onFBxmcko/jZt+SefCUQhaVKgwdSAAiRTjZSc7idwxiwBxzBegk2fHLwcFWMX5CISgqCNTkOKHkuU2z4upSSd1UKUEZOSchq"
    "fnEe0POUkoiePz9/R4pGdl40gvOsEZCkEZ0njfg8asT66Na9IvfuBblybwBSg9y4c3LtLviEzEKImUygm0C0AboaB25GYjcC"
    "ZgIOIYTNs/xe6IK+hmmHNPiLXw9/zSmLOwiEwDJNnmliJnMAAb8IzLR4poWZAQi5gVhYps0zbczEyM5APiyzwzM7mDkHjAcg"
    "KJbZ5ZldzLwh15DZ45k9ntnDzCtyAZl9ntnnmX3MvCX3LBpBvMkuex3061Y5Yyw54wb5IvWWCmscHFKSM5rB/YBsZDY7LNuc"
    "qtvFHhN3LkJ30OvQLrkJmQ/Hx7NnmmVYXRDj6eRaytjAJeYq0XZhbi7D9OamTKLinpnZ3dzQLIyCGNCLdHIV7gTqvAjdsbIz"
    "3yr7ku8/hpBdNF0NYtKjL3blMW72dOlgMNr2OLznuygY74mt+D7XlijScJjnNhNYcfszJpEiTOLgUHJD8zy4og6Te7EglBUo"
    "RVhDYSwiip5cclwUGdtdWNMyXuYayJsYRKAodKhVuOtcxPMmuKSxYvpJ96UHGbgvcPwhy+gMVqt91XkCvMzF94aaFOJwFAfm"
    "Ps28oNvMNC6r9BrWI2+OIwKRuEiKoJrpDp2FQbhbDkMnk7PYaNzlKN0F56hmE3QtiYyEWF7mBfqZCbco9F2uWct9TOjdAl09"
    "9or0M018fVQ0ZSSwNYmqcLuJoKXRiHlDnshDz2adBc1uogTAlPf4uMoBGhfN5lRilz3xK79aSDsG5/laZWBZnKWKWRkIMGWl"
    "wgF3MgHXy6abbQTZFu1xcWrExalPfAf/Tib84ch3kqYbsQDaTOB2iUn+5NJXU+c8dV5LTVjq32qJGS+a1VILlvp/aoml73CL"
    "gGd8No+PTWvFfsCudHw8EL8s+NURv+wpTliHJE2OsY8KsmfcRBK8jpnxkMTnf9BWcr0mgZsy4klPzF59Hs7SdG9G6cLXR4km"
    "zo6U+xvmD1iKKtQFCCW6KOsvfX1ZNJucKJYMJkXTmQWP6/prSRdFs0kylexEsX1GPrX2XojWPtN7pOQm72ikVW1fXKDU/+LC"
    "X61+SA0AzlnbFrCsiY0OOQ91KExjgB5BpgNfNM6nbqClgKWivk98qcXZNQy760b5+drX18AA1uIzx7UhHk83h3j68BBzpG/W"
    "v0c7OP3xDk43O/ik9q46Q4VfgdygH5ivE6bHwuy5mo1OX5BvyvxEzYe9wNeJQFXCAi1/tToFvSsvpyk8rhgpn9cCf7hmU46Z"
    "4QOTlcGn90/5at2mDu53zA4co9Maj1YaaLPhhDb+55XhdsufX1QVi6lrgik1f4YR+B80yod3A5YFnWRQ0fqdLQbM6QMwUuQk"
    "yTyedm0lZ8LGV7J/xdI3XWvbG+V4Kk2W1MOOYvcn9p9jvMIW28E2ceeA8J5Y7MQ1h51GIzt2LcvWI9c2GxkJXAgqB/dw4aCb"
    "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"
    "5sotq7Azf6XZV0aB8f0C/q2ZuGZ7f351je0ShcyWHwpiG6MwK6UcXopfyyqIbW4CFsKh+MSNSyAV8l8m7iXDuSN8JdRQqPHc"
    "UO7uyi5BTLEfDDdZQkzht4bwJ7DGNwZzzjAJd484tD2MdPoKDKxw/CYATHL5DO+8APYQWxjRd8E5onq15o1nfJDnv2QAd2NG"
    "ZPCQhHMzS+dWTMMJjZ14niPuz1uJIX/+JxJD2Od8FnvzQgCj1Rw9QII5j1uUiB/RXA1gBE//OwIYJfPWAYzG8z9XACPen2oA"
    "I/7hUwcw4kz0BwtgdAABjP4c0YlwsraMTjSZ/4mjE33/SaITTeYtohNN5581OpGK0+eLTiQ6qeiqy/nn01WX81xXXWh1VX/e"
    "rKsu8lUA17/Pqqv6899dV53PZUjM1zQlR84aTq9eQnzvqh7149goz54vHOkyp7pTmJtNS0jy17GxQ73DloB2FW5QhRuMzJG4"
    "E8Jim9qaMxUSy5i4ytXb3NxSuW2b+6jgrbf8kmpqjdMb9dZqDOILgs/Tm7T4vhweBd5x5wspGyp3XPOGWa/KR6OqQndkqf3G"
    "T/kRKXx6Lg9f25fPD2grt2P+Pm68kqLzvmntQBbewZEu3tdzx3WJv++lFLx5XCn7QC37QFv2AXobRU7l5PmoxqH7aCdv7sQx"
    "4Mjeeu+HcJ5SfzkvP6mX/rtwwr3NvXfsJCX33sAZK+694L+bqO69E2eluPeuyNTxhkfluy4kzZ1cDZNEsN2U91yWzpF0hyVH"
    "wjMoJMKjG8G/N74HJx6/cK9lKu61LJyjyqUWEktyBpvNQdDtLocTQE1xDE7ZVWWyMG1oAJuYmnC3hL9hF00Wn+42xZ2dK/Ob"
    "GDKGPcM7VW5STOf8OkKY36LgPlP8woICoV5YUG+SaGzbH9crVHt3IK7cHYidcBiyGwA+ev7LcY6H1qU/Gg7j3MVfutMyJ9pd"
    "vPjjOi/+WPHij6X3uM7t7eMyRZGInD47ELIqCosyT97WqMq9ooATl6B1dxOAxDC7mddkmfxp+fpB5KTy+gHdJpwSJyoJp7GT"
    "KMIJhFWkCqeVEyjCKSATxzPS8qBO8WXx7sFEuXsw1d09mJrDqbyDMN5sYMc4Ha5K8oWKi3N8TrI7c1Ohp3tGWn9p4HcULPoF"
    "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"
    "4fWEK0G270dG2Hyp2dtQawwiJqbhEDYf8glih4YTXXzVt5hxQ8g8vHtl4r0stsxlubabmeTD3MlDeq7SNApbx/Qcr+IEws/x"
    "PXL+YZXQuJegJFV3zHeP6ikko91hi4vOVCCUAn2UPOxhfcjOShS+InyTdcHuDJY3nSSCfZ2uKItzWmmxGKiUl2HvrNv/MnWv"
    "b5qCjDLQllFGGXAA2/m6YKN5viS7YG7ISaFWlUNbk1WMM/K/TFIoiBTYsQy33+xTNFqlyLr7NXvTYy4/Wws2R7plBWtD3RZw"
    "tBnJeUBoCTPzloXKZt5SE6oQv9kTP4Gt1USZyWKucpcjfcGeP47CXgwsoBQV3L21jD3xYyNOA7NauFdT2g6j1MjruoIb7GMv"
    "MEv1l8DyJs2GNhsRfh+labSoFjppKhSFwW1NfFw2KABlAVQ1sCOfvD0uSTGAc8GGVwIA8676Gce/vw27/BVOzQILNIXWnVel"
    "Qk1fGiiTv0OJojR95Sf+ez9As2VZcnMqbu2abXvTtBCtVS4Th/+6edI/PP8YzTWGRmWEqYuNqpPkyaJOmCtfbmppmyxKlNgL"
    "OU21uzJ2smjND9rKm/hSw3s7NNcYMZavb7P2AxbM6gZM+XLThM2OA6ZDTlPtrgMWzHag4N0HbIfmhIxXKRPQG3FYBCrcOApW"
    "i7AO3ZJs2VmrqdQ4DVb+pErN/JRIX+IjLQ7aJVEu4aY9B6W3KVB0Uc9E8LsoqryCGnW1Xpdm5eriTmu7Vjk029I3Bn+Xzoka"
    "du4dL9jcvWk0XiU9XH0Kp3i5HqmtG4tZeIjCoDS6YUm7lwU4K9V8xa1saRNRbbRCBqGPRtMp+AzXVM6+1hFDlyKiHAmbq8Us"
    "NHhTuoAmQMS0FXRlGseen5RUJO1AMbjS4FRr26ZxqTUyWP2WoLqktslFwFfOLckIGmeJUrj1BNmezKDUhz3lW97O3hKuUsUe"
    "nWwh5eq72lbe5c3tL/GqdezR2zZST+lunfwr7W+1TTXJo5ppqS28ZZK2SKrBK2/MqtFMelG0PcW35OQo4L7vBOJt7D99ihXs"
    "3LlWU0ffxdYThzd1h2lTqmHnXrabMrybe0wY3sw+06VadMtk2ZYFhuu19WlgGknNyrUmcGMGmRzfPWcH1r731FBL79ahNpNC"
    "1622MwIb2X86FIrv1rNWEwG7tvsswAb2mALlclv4f1tKIlZpQ06iRtqycq2J2pjOKMd3T/7H2vfmf7X0bh1qw/+6brXlf2xk"
    "f/4vFN+tZ634H7u2O/9jA3vwf7ncFv4HD8/tJAaofWjLyrUmKoI3UxNA9uR/rH1v/ldL79ahNvyv61Zb/sdG9uf/QvHdetaK"
    "/7Fru/M/NrAH/5fLbeH/7fnhWLWNCeIa6StKtibtluxyKt57zgTewt5zoVh+1461mQ/67rWdEbyh/edEqYJde9hqXvAu7j4z"
    "eCN7zI1qyS2zo00SQ1b1liyGjdTOy7Ym9NYkiEX895wnspW9Z0q5ht072Ga21HWz7XyRje0/YypV7N7TVrNGdnX3eSMb2mPm"
    "6MpumTsckUkN+bUeIKoTTKF9UdlWX6MWRZsJLMD3nDKV1rZPnT9kd9tOnUpzLabQH6LDW44etiK6z0GCpvDeyN6VP3c5bPjf"
    "RIedGXenk4g/IiWq5v7tqO5svq8U3RPNO7NtaxP//47+786u7e3/fzAKlE3tW3Hc0XpeLLcPdndlzpYW9j90n3dmyLbm9z9M"
    "r8sm763Y7WjFLpbbB7u78mFLS/cfus8782FbM/gfptdl0/NW7Ha0JhfL7YPdXfmwpcX5D93nnfmwrTn6D9Prqgl4K34723TL"
    "JffD8a782Nru+7+g7zvzZXuj8B+q9zoT7FYs97CoVsvui+lduXQHq+v/GhrszK27mGT/cFRYBp6/m4qJJfYyemHBvRC8s4UT"
    "m76jffMP0O/dTZ3Y9l0NnZ+y55BVd3+7uxakihE00tLoCqB7sluhlfbM9nn60JZ1Cs3swDifphftve4/MrO0NXLWIHkXDtrH"
    "3P07dm0nxtrLgv0ZOtfoh/6xWauVUVKL4J3Yamdz9O/Uqd0Yancb86fuVoOf9kfmpBY2xSpqd+GhHa3Gn70vO7HOrtbgT9qb"
    "Bufmj8w0LQyAVdTuwjQ7mng/e192YppdTbeftDcNHsEfmWlaWOuqqN2FaXa0x372vuzENLvaWT9pbxrdaD8y27QypOnQuwvr"
    "7Gw6/V16tBMD7W4Q/cR92uJt+pHZqKV1S4/iXVhpD/vm79avnRhqH5vlJ+9ZxRb5CW1BLS1TH8cOqTT5icxCH6c7u9mIdjUt"
    "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"
    "GAXY5EX4irFs3L1jwVxAvh6YDOhc7hBJJIy7Duz0m8ACd3fRZ7+dK5CaiBcewl0pzbMSnwFHXm/8sh1+Ecue4CxBk5udn2UQ"
    "fg1K7EVD1L1jMEAFz9HsQCHBMxH5RWin5kiuNL1L6Blo+cfdMoTXMNENN03Hse/c+HJr3O09FnygwCOcniiwwLEgel87FgZ8"
    "5vXgcI8oiKG7OQJPnD84GsR1QdvwpxwZG7F5RcpGMK8oeB1QXCaNcxuQlPILoMitQJ7VK3lWbwg/F/w1ruMqUw+4Vz0vJq4H"
    "bhJiGRZoEorhAu7wQeg6IG7R+K8C4pfoUef6eWAKyWMQl4488F0LEG72Yz/PfUfPT7mJ54J/YS9k3atXnn+eJxg5siucYOLL"
    "L9Gd6/bcnzzbUxLclExPtMF9kU07ghasH7yXJb4brh+stfbmHbE2rCv2O66zKKXxFfYSPKyNpNAk7wt2FzHEio/jq6E36Yhl"
    "N8UAU2haWN0Bz9hgSSQa/5duJPd+5maLKBvsymyXP029aDmSw4ILVNcFHV7iblgURLbNFdcF0/0EmwnRFoWNftlIXN/YYAbL"
    "tK2we2j4foid/HGO4+wJ0A3+q4QPFO0JAFXo2Z5gUsuu+kwOsGv7f0lT0EHoKplAKpGo20orNHJS+qB3oGz41yXrC+7c4BX8"
    "e+mSKBUKuMl2r/e6zrwECXYt1aVLme5kS6a9RazuiAL3CMS920vGVtLfE9ZcuGslN6sOAWYZVGw4r0dnEN+TmkK3B2CnmyZp"
    "0t5J2jtFVDVZdiGuYFqGvkWzIrtR1RoEJ5soXGCP2YniC1mx3XsEQYOCVL2QFdFrF3wBF2d7b9BOwR2AWJRU05kRJL3NJcYY"
    "TS47iX9CYo0K1jqumi8pMOnkY2l7eJnbRlE38Eo2yyVqyDHLtPdCjlSvoG9grlS7IG0kvVxHhj9/WpcPi9G1/f86YmPvXfi/"
    "gW1zA0yQpKBu9PupU6pW85AUbCut4FWRNLBf7Lg2cKJU3BPk2HcKfFbc6I/yK29QkYwtsQnvVGO7RENnT6ic0qlpSZZtintC"
    "eSOYGb0ulDwpJ+VFgdlQ2IIlt4VNJ/cuLGGDD5iJGbAAZ5UQWZpdEfeEGoxiKZYK4rYenIZpiHtCnnfWHtMZu8kPwN+JCNxc"
    "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"
    "IPfyqLlgq1tWCpg5WVCVQjRnYB70pawg6D67CMQ/8DBlFIakieJG04kEl2oWWJBPQM13ugfcyw9YwADbwZfIdQA6ZgQUqGHg"
    "nn7xzK6cWJxRgphlMAHAt7GuUlkxzp5QJOx/DgB06MVvJwKt2JIKcVTqxvgDbkT0T7D9iMrihEbQPZZlfbpgOQGGYNdedH/O"
    "doBDR4cX3fehoMDCDb5NGesGfG4Qg+Cmqd9hnCXIKWBVDVqG6sb2qRYN6w/WvJJtzxCqf/r4/aPXbh7df+PwwWdPPvjq8NFX"
    "x++/ePLtvcaj3x399uWjt75rfPPVk989PHp05/iDd/+x/0I4PM++efLO30++/jAUkg91evj9/ZOPX2ZdHz54vfHSV4eP3mZv"
    "/7H/q+Nf3208fOXku2+OH7/beOOVwwe3Gm/cP7r5u8atzxuvPWz88Y+H37z65J2/H37/4fHv3zt+/8Un7/z9yfvv/WP/hcZf"
    "Xzj6w+2T+385/uDdJ/tvHj76in319PFrKJSO33/x6LWPGr95eHT74ZNffX9y79PjTx8+eefvjTfun/z628PH7zde+VPjo4/Z"
    "DJ4+fu340e+O/nD78MHdw8fvn9z/+9G7r+OsnZA7Q8rxB+8ePnjl+P0XG/e/OfniztPHrx0+uHX09aPGK386fPTo8NvfN15+"
    "qXHvG/zSC7k3XrrBPn/6+LXGy7dOvr139N+/Yl+xDhtv8g6h55v3j27/FwD0wd8bNz4/euu7wwe3Dh+8evjg4dHdPz99fOPo"
    "9n7jo/chxEgOH3z29PFNHM2Lsh8+etR45c7R3f9o3P7k5Ps3T+689o/9F07uf31892bju5ca9147uvHm0Y23G2+8ffTVjcab"
    "rzdevnX0wd+P3vrm5Pv3Dh9+cHLvu+Nv72GfGFF/8p/vHj/82BdPP3rn1ZNv/+sf+79ikDrR9MbXfzn89nbj9T8ev/myF0tn"
    "WA0G0hufv390+zOGkWjs/If92ye/+azxyieswQ/7Hx69dvP49h8b9/7gJ8/QSvxj/4Uf9m8zgv1h/8Mn77/RePN1huLTEcom"
    "ygLmT35z6+jtzx3AQpHyo3f+1PjdrcbDtyJR8qN3/vTkzjfHH9yLhMhP7j9qvPH24YO7cUFyhkdfhJxNG1b47l+e/P57iI0/"
    "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"
    "VQqBkGPjjd8ev/XHw4cvH734RuOjt5+89d7J/fseATqfMa/nk3fuPfnzu4ePbkGnv3l4/PrnuDHg+wdmgABFv/Lp4cPXT+6/"
    "cvQ28myz+KJrDDUzt7jp40YWwX3nM+UcKfPB4Tevwo7Y+YwpLHTpvsu3el5kEc1Zn8HIRuUo9YKMrq3MPZIy+xTsX8dGsLyA"
    "ISjBP3/Y+OhtBgBTzYzCRLB6MUb49PENtNohSMg9U1k0//yhwsbXf2MOspC94JjWsE978MqT994EakOHGBMDCCYPG4Ir58FD"
    "1xHy5M8vHv/+PeYIYb7tp49vOD4UBkjjpa8a9/7w5L2XmIlz8tdfgb194zfHN38D/syb4HE6vnOvce99YJI4BwpzPQUCgowE"
    "n9x52Hj4xuGDh+AnRW7D/eT73Avm87IfPny5SfwPRNXKVeL64J/857uNh39t/LvDrP91B4Z4dOv4rS8YWD6YWMCPWz5f/63x"
    "0pdP3rkLlOczKJ/8+ltOMw9fbrz+R9DFaO0w4mE+UNbD4aOXnrz3pt9MwjaBEJ/L9YDmr/8GTrwbn4OG+Ojtp49vcpr2x/Ug"
    "IPLCJ8cvfNP4zaOjd75ybQ+22T35/oOTO68xMoXBwvE7oJj7Xx998YJfVDM2Obn/F0Y1Lgc7Kgf252wpmZ30w/7tM918ju3H"
    "Q3ncXXD7kycf/vnow+8ZYo9u/xFiHejsZy6NZ+qfedUa937NIhnhDdGZ80QW8wQGI+5nBYIL05uvwr6OvUVBgzCx6B4KIuRS"
    "X2fMH3P8yauNh288hWDO1/4+ov5VtphOSI+5TYFC0YvK+MmN6/H5o7XuNnA/ebJ/8/DBXT7mazd58O7p4xuTUgENjdduEgjk"
    "camzt1eTDLJd64eDuXjZElwPlrlkZTWvjvvCj4vvZUlRwRrVMoG9sEmsMivaQkm+Tiica4fI9S9MotsWeIaiB3DZXyRPrW2K"
    "JXoNK3z69rJUJVu0armN/B1berUVbnF3RhAQhjyVCZVLWMyGHaeVqUwkOF2bIstQuMCDFL1AJtF5/lnBIpJFJGIpFSrgiv+C"
    "ZYBAzRLiCitITieGpECyk3URu2PhKGJKdTgp5TtJOwJIcgf8hcnHMVMkJ8uQGYPBqcABWo5gE6ejEQnasY0ngction0JxJK2"
    "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"
    "9TB/yt4sdB3hWRc62pPJvb2kUK3382GvM38KTW1s1AqF6aplPv88vVjUjYRKrTUqGOtwMjaprdH1fsPZnmt7QqXeX60ninVh"
    "bU3c2DALepWOAbkieLXWQqa7o5hPd6DnrVzvx+dZpSKVaNs2zVcv5iWTdncKi1vq5TlNv5LLDazMLV6ZXVJzncs5PZfL5Wbh"
    "PzPbw/i/2cWd2fnpLXg0VxqpWu1tE5VBancWR8zhqcrg0mA+t1sYK5QuZ2qjw6OratvljLq60rWyWum5OtO7qI0pV3O5gd2B"
    "eWrvzsn5vHp1d2B2pqWzTd5SesbMTNd0dTOz2ZevjqYluTS2XZ1brY12yvL26PyEnZfn5q4Yi1Kho0dROuyVa7LVa8wuL9vb"
    "VrqdDs1ca0tPS1Mz6kKX2iJZw1qmo1ubGtcmZtquFNsmivmO3FSLtTvb1VarDa5kxvsGtYli23ixbbDYVl9pq2tt21pbr9rW"
    "W23rvdIG33jvOrZbFuubUz1ddnGst+vK1MrKjJYeLVeurUznc1v1K9NjbUMTpcHK3BVbuzovd+8uzS9MVtqWt3Lm0Fh6OT9R"
    "bVuez+UKA6WB5cWhUkkqlXK5oYHB3JX0wPhWbmA2N9xydWFwea4+uzl2bXjTotWxUm77ymp7j7yynOvu7K7qy/XN1W45N9pV"
    "mpnO9Y1qeq6YWbl2ta17pbs0qtZoumtrtUOf3u3ZtBYKPdes6StL+am2zvHRlnp+aGdnWBkfmMiN1bTMcPeQWp+bWhpRL9f7"
    "pma1XPWabhr54V1dna2Wly+XzcXBuY6CpAyPFpY2Zy6vbHeOXxtfpm19w7miujVezizOmC0jw+pK++zARNd47/LQVWW6ayvX"
    "M7hTmLW29LHq0kQ9s3x1Ymlz+kptzrDTU1qv2VfN5XquaaV69265MtI7N7ezu6uri3PmaH4wV6guW1b7lfL41b7h3NWey0at"
    "Tbk80T2hdo8NXNXS7dO7VwxpYuXa/EhxKD3deW2mtNA9NFE0pnO53gqdb5uhnZneGbtjTjWVuUFrrl7R1XK1d3Z1dVkZU8eW"
    "RkeXZsfk+lh19vLK1vjqZHVkTp+6Zm3WLT1fHNW2Bnsrq5nu6QlrcWirPLhoyz2Ly0tXK11XC1td1lZfJrOVm8hfvXx5vlu9"
    "ok0NbI+3rGzPjWwOVKe0uZWJ2d7x+mJxoTPXO2DrnV10YXukdmWxd3O4Y3G7Zix3FKVqZ27i8rw925035+zCyLYtT0pXCjRn"
    "DHeOrCi5cqUttzCVsyrb1uW0Wu+e7JvZ2h6ev2JrxnZ+c2nA3BmrdHbrV6Ym+vJjl2va8Hz7VMfu8uDstDVTK03lcyN90qQ1"
    "qO8O6MNXtaGrXcujJa1lbHnqyuzC6KZt9M7uTHQXJ1eGlZ4p1VgcreWVNiWnmzt630phpbtP0eubXcNdSu+1TXuiT97t3C1O"
    "b3fal9WR8c6FtsuypO0uZhaXlyxltT1TWxkfMhZmO6fTI5dHclfyk9X5nVwu11YbL2amryzNzy62KZ27Vs/UqL0iXVm6VjN3"
    "9InqSveVua6r44u9qtQhL6TH25RCV8/mwLVhmlav9ZmlPruut1d6xydnJzbzZW1moW15dTCXH1QWx/Xh1WL7wHRvLlMbnOmY"
    "7iwqV64Y1ZxiDldma0u0Opuf6Vus12bM3pZue3W5ZWA13Ted3lrV85XqYu/o6JSpzdSuSpX20XGjareY6kBbQZ4cXi6o9rA8"
    "1ZcZL7WvbLZ1Ls+m564tbm8bs1eHR5ZkaWq1u2Mp17M52t0209J3petyX6XYLu8ODY1fnu2wF5XlhcLwvHo5PX+td3JopFDs"
    "mSyWarvtQ0NjuYmO8Urb4ohWsMfkyY7p3Ghuqe1yb8vSUnGsUOiWl1uG+2bSo2UqzYxf6enZWSqV0mOXe0fHLqvDmSXF7lvN"
    "tBRnx6lV0S7Pt1ib8+rcuDGxu017Fqs1fTjTrW4t2PndoU55+nKtrWWrPtlp1cszy8PbCyuTQxV9QJmp7qyuFqsZpdjR1b28"
    "2jWwODS8OFypTxY626zeSa0wWFR7B8s9C7V2vXurt2/SnpJ2aW5psG1pbNPOT1W7x8vK8u7UdK5urmzPl6rygpoujLTP9hql"
    "dHVkZG6Ldq5ICyOq3Lk6s5iW2/PD+libmqGV1c2BgdGppZXqlFzYnV9dzC2v7tYyg7XZat9W+/S1yuLoUG55omVmZHYurxYv"
    "az2ddHtiuljoyJSqO1MTo1W5y5gcayt21hYte77Yq0ndw+WVSkfJMkboprW5MGWVxkuVngF5a2V1MT9Z62yprKxUOnNXdtLS"
    "6MpI9fLqtaFrY8Obateo1Z3bnasZnVP5nbF8Zmh0d66jd6H78szQpLozblR7t3fVlaWFzUk6WZzomtrUhroWzZaOwbo5W1qR"
    "B9WOufn57onFrr7LW4Vc21iX2V6c6rjSa2ldfW2T28NXp5eWrdWctjNeqYznWmxa70uvpmudCzPL6thMy3RXR3qwq6bVV6pW"
    "rXertyc3Z2ZmxwsD3d0rQ33FpY7BKbtl83L3zuacnilPbQ3vbncsXF3uGq9tt1mV3Vm5PT841d1X7q3IffbS2OJmadjsWx2R"
    "+szJ/Mq1esbsnqgtzizmy+XVvs3tztJkzZytzGxdy+VmCtcWjJ6+ibnV5Suaujw9Wp4eHhuomoOjmSXr8oTRu9I3m6+2LW/X"
    "FtTqlNbVOTUx0SVLPVZmdWirY3Q+fU0emdzpreUWt/umKjS/kOttkcbGZ9uVjvbu7txqTq3sLE+bW5Xd0cWeydlNNV3sk0qz"
    "20vttJ7eWqm29S3MFduubk5fa9uZl2amM7oqD9Jts6IV7YVRtTo7N7Kz2tZdaJmrmVs9k3PLRlttXO7pHlarnbvb48Wtke52"
    "vac+O9I2f22uzzLruenRXKedM0eHa2l6uUMavlK3CuXKsG0Wt671TWiZ2hV7aXusc3GqY6sjbVxLp8urxvJU37Q1eXlwZKIj"
    "M1q5UpVnMna6o8Mcpdv06vD2aGF0cHxanSl111om/g9tb76lKLoljv7fT5FZfVYuPZjBJKJRHScugygyKCqIZuevgklAEJBR"
    "jIznuA90X+wuEIeIyqw6fVffrMoUvolv3PPe37wsZlkv7ScstBQSujs4rbbqdIkFgxzcAxt+t5wuN1jP9sO+qajWbpL7WLDt"
    "z4RtX0LXCoJoJz4cK/3xqUjt/lqd96R1wIZ0KIy89Y5Yz0P4MNn3BkB/jncNWO32JBPXoZ7BWBS93A0iC0e2vhOAJrDC8+Fk"
    "F7vyEk1Gx3A/hvnM7ppYX1rDiDga5fYEEnr+2NNNP7J26Zid7Gk2x+FFpiu63UNFVfSindbvJ7LgsAfRGRK0J+vDKTk4TCcU"
    "pKsmCR/gvj6Lgz4LlaTpTrvQUcq4KOmhDDTfFbvIn59wB5PidZwn6yR30Gzi+LOsO8FxIj8s+ta0GOz3+yKBxq4vp6toWB7D"
    "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"
    "CxJzm9gy2kdkFd7UY4367BcwQPnjnYnNFQCc+aHGsOZudcILHHj9K7qt7tU7su0S6ua54b+awDe3t39jxS8o+l2D130Q3u0D"
    "7e/3wex/SoxdAE+d+4E7vB7V5VWDeBdB487M6I61u+fpOjcw8W5R4A+44GsTXeWyPmeAdAZ9mm/F6W2t3HuF+U+/GNTwy7d+"
    "/Lg9z63tT7qBnM2v0trO6/7Ll4/Ff/2xM9j5NYFpYnqvhw3gmsCk3xmTxs3NRxcj4eb1Do74jX1BHbfpagY8vZjA3qXeo4xL"
    "Oe9Gzn7vEOWdNXLx8tbZvetLRZF9DcL0J2a4zRqc8cHLW0doKsJ1xapOcq10fnvriO9MYWd/Ag3nKHCUFpt/DRgu9qzxOUqV"
    "W7ucVHEYL2ChE1659k5yMRMNHqrhPD9XLh0XJ47//m/wO9h+iMKo1a5zOkZTPL633HzYa1HrooN/fXh4sDpuYFrHx/StXRsA"
    "pFbcql0VPTcwK6eAetLad5b3/tXwPryw6k+2lS4ud2y02hcL/M9p403RuDGYoVHHJnwwYktLrXllItdq/x40XghiaFpUdViC"
    "NKk8Aq7MvO/XRZNWlaaZ5rli0H57J2T6lVzpno+nr4Ya05uBRbXNLig8ernazN22Sh1y8+ekXNC+rkVlodBIOioTBa+82YcQ"
    "5QdbxGq131EjVcLP7O2My+LVzinPzQZs8NcVfsZ38NP9e/jp/X9jZu8s6743QMjw3UgPtdi8QZ56jLeDd44SfDfSKuHnCKMW"
    "vr38glSI76wJtXZt1PqXY6T/Z2P8/t6I9LqwYpha96Yvu/O2ucutzsjZhezl+WabZN4GW3mQfbCPIVsvRlj5Tr1eYNOdSKl6"
    "HfpnoVJ42YJ1hNUaZIVBWG+1ayfaHQxG250PO+Wd8Z9Qdr79xCbH+LOt4L9rj2M91GDjvVWOeLPKeT+f/46kAhloBoZg/RqR"
    "MFeBRCOhYN8BXP7d2/xP4HdRQ1bSDw3vI/ytYW96D3uDxpcgbiCuW22y1qtTxayyvkHfK7/p9NF6qOJFWy24/dbuaA1oDf5t"
    "0FoB8vd1avrsBtQbKF15uTXSzD9B489PT2fcXYlfLwkNeL5C+19VfrpUbv/ccv9Cal5A1mtDNlQ4sdkHttX6dk16/e3MvV1k"
    "kFlSVq4M73DpBdW7eSUAiULfrUDnOVFLw71rXERIHxjfS4Odbw2Keb7snw9cEvTW+Zb8ZON/II/PvaocSO/G8ppWgSje1/4X"
    "/NboE+50DRds8pJkhlHJ6G7KhgYPuFfA/1Btm/pA3M7Kx9zm5FQSpzMTd1eg2muXkTTMWkOv3jNrPznLH9q4P9MNB/7xWNfL"
    "eeYYzlvke/svzzcMd9j7A/5rbqpuv+K86gm/ShwvB/ZvLAU/QqV3a3kewP16/pmmtT7M/3UsH8jc+x5dRtUw2TDUYcp257L7"
    "fsVynff516sPw7udpf1iZ73nzNyLkuMXW0f7u62j/S9sHe3//63D/w+3zq/3zN9xJIgJmwaMYjUiOb2XY9dXrtXR3ivPyisW"
    "qYS7lwYaNuV7R7lVrWp8TVyzkmiPP/i1RdFLJY6OksfX8BK5/PG1MpK+wzuf0ifr91PQ+mPfSR+uxWr92GvyGJz93+KGM3DP"
    "MeVqj7cwqvR3d3WuCMM9S3Re4iyorrJ4eXT/JAdyr0Kg5wv31fhkVfiCM1vtjv9UYSrj4Rx35eHsEPTWyepkv/Pw8JDtW43n"
    "d7vJ7lSw+rEu8FA9/rzUc5339vZR2XhlAi4PQ7++b7Z29X66aP+upeqN8HRBmvVb50N3L5UqFHfuw3P26L+1O6ugmaltaGRJ"
    "R0vKwKhcqF+tSrdYVLcCU0GrffuYbV16Q5as2bIeXLP9fK7daleIn44r/xkiboXtD5bMF/yptIoz4b+vhWanmvIf3vEClz1V"
    "XVlzIaiq/ffbBRU1aecF++2sg3u3sj9+nE9NLafdl513u7Re3rc7N6i6A/OycjrI639n9b922T4rksjyKoJ8e7tj8+Tybu8+"
    "pLFb0UbBE/h/YlvXnv+79d/JP1v/bQLtb58634EPT6D7UAU5baU1Kxg0XOC3sx98q3IManeuL8j9C/q9/b3mFuMn8P/8Z+sb"
    "9HWgfd1+f0Xfflyfe2/tf7z/QnzhSCsuoYGBT09P6PO3h4eHKu17TZxVxBBgtR92oRu0Xl7aNVPR7M9vUAfpdM/l0qd/RVqc"
    "WGyQti50X9pJAaTdgXvtC9v5qVrwmwv+qmx9q+8m/1535lvccTva96cm7fL9y4SCCIZdfeH/6+kBQgdI/zkFYeRhgDy2Winw"
    "AGFYG4Trn3/+E3novl319g8IjPT+GQMPOIwh/3SBBwhHkH9qt76M7tauXsc6FMI1tMHvaVw2uR9Y8mb3tyrtbV5F3KmOBHUW"
    "YLVeEPPms54+t9LaYWCRlr719PKfEAS9dO6TrI5c3pdptx/rGTO01HBe7+fw1nGubLVfq941JvlVyJntt1+BjNvBrXyuv193"
    "63Pd6X2UpZZZf7pl1QOZxWFkxWlZu5W0Xr5+rRz2v+r2S7vWvD+lz6OylZ67eb9xq7UN2v/10H1+MbXYe3l88Wu/38tGOF/5"
    "+bCvBlZFsdW+fPlTUuulFcXW1oqTr0alKfyaGI61tx4/VS22X9rnolby8RP3s3VJu83XoWzV++vp9e08+rhiXtJGKlSFRXqI"
    "LTMzrAtjYl0ixT5a39LvHatid+JWHaqh/fwct/j8tsLn6Aju011MhGe3FbQfF2l1z2glprx2ZNJ0pH1d2Esf/qMeW9RqnUNO"
    "tGpi4uUfr9bb13+8xm8vnYp+eGxMb4K39l2j62uj56F9/px++VKFszWtivW5RfZ6uSRXERb4vOKtOt/q46dWTcTP59zH+yqd"
    "tOLJ4h8/PgcV0zUpL95DN017p56XxnAiAYLL1LwWTvnodhpX6IeHh6pyUOv/mp9zxeoZ0NoVfgy/362a+mFc59JWHFdL4j41"
    "Ex+/m/i4FTw/v9ajSN/ajxfnlG9uxaG6FYi7tb/5Wfsd9+kcaEN7Cs5B/m6RitNWUE3GayVy0N5+Two3NZxWUE9a+9XQEuul"
    "Ce/98thMwXnw1cDdYBvWrnnVjwbED4kVAfFDU6F9M2Gpz/rvdWtXc4fHBnAHZzOTL1/4/Pz0rUn5/hx/SDgfz0vojOe/64l1"
    "14NX9zF4cKtIyMFj8BA8P9c4+2w2E1zCJNdpl5jGwTXccSXhe7tAsMR6uRmX/N2cxHeGKJUm8Kczcg2J/G+3Zpm/bOwcIPnv"
    "WzqX+2Uz52BQzRqlT/VxDB4MLUuq34uN17uPXE7DmfppfuKHc0ut8warozHHVQDlb8HDWRX1/Hx1mf5exShoWqk/VVnAnH/T"
    "KnhynZmeRS7fP3T77TJ664/7vn+z6jrfn+oRNwH5Lp1pTKQu46nXuX0HvP6N8VnXDr+b37QhNP7jpWIpf9rTis7/439vmu9i"
    "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"
    "q2H82cHVqG9TqDLancbw9tp6/fqUXox3XyqPyQuhdUn+I7a2P36cubH0odJZ/viRnuFquxIp6Z+fnrRzV96dhwpyV9kNoE5r"
    "47D63LzbhucZqvLOePN3y0+sT+62VaOTc0+vn7nYbzbfr79645fqr768n7+GEnitOfCzL2SFn2oS7Dyqhiu/YZyGwvh8oXca"
    "gqpOqfr5dibgnmtx9JV+vIjNfjIT7WbwX760wndzch7356ena757s+NrSly4zXpwzUSdi3f+NJNQu3P2lzwPP7FaVZ+rlHZn"
    "2jpbsPpaWfNNt0jRrcb+/rHJrtx3H++6+da+0xVk9UGowENjRHs93bXI7akhiT5/hA9B+8eP4L+g9uv7BYM67ocQo1VaBdwu"
    "gVIvIo+3Dyvd+bi2Vyr+7X4/dX663e4mtfNhtTrVrEFXaH4FuzVVVIOFBom+Nmz8Jfnt8fXt919M7FUYENyEATfwVUlD47c7"
    "aGPeYPAd8PzJVMGdD7Nya2N738YNUN4KRPcFLpDvlr23WtViNU7y95P0px0OdX4+7KvPbbWhtHeqNKdq/Re1zsbIVZ3wflLs"
    "d1j5YuXcCc50QqWktVrtRj/5dneOXi5Gry9vtaL5LymC2gPTsRrbkA+tVDa1b53g8pGgdX8w/jjP121nfdjGt5L5ueQNKUAd"
    "o8Ynv5gPx/L9sLKh+EV+FPr+y1v7rZqa0np6b9Vydo2tcOhf1r0RP3rVO/PLl+pO0GDpGt7zQ6vd+fyhwS9fglb7a/KvVD8/"
    "GfXTPd6DPuK90qpjXSdWunT3VpilLb3a0jU/+4/y0dLb5yEcfzWEButW7dx3eFF1+F1nb7SWaQ6rAM58ff+YFbde8goKuL6b"
    "loZTq3c6R6vdabTjfy5ek6YvndL6+8X5ODLYQu/WnKpntTqw289PT2d7mS9fasH7pc72ruNnR4P/Sd9/WuPW/QuXfne3QKsS"
    "oLxeTLxqi6+okzzuO+mjc/WP+uPq/FzeO4vl15Do+l1I9OOdb1lx85VevHOPpu5tnS2rMXZO70ILBFaFcqlaC7q8RZWw7oJJ"
    "DK8hJjSrDrleBbq+xVw3rXdB17fWXdT1yLoEothbZy8Xx7q5udhW47P9h9X4w+VWY/62sM72b5TVuc7ho2+9vb3jcR5q46dq"
    "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"
    "jugRuJqYfV4JB76XxXCi0GOH2wM9vvBxRoP0bV9e8uMTj+7Hk2UKDjYpMc60qNvbpwgSbUEDVOfqYcbyGLEM2J4zOuHLhD3K"
    "Rwsts9kITw3DRcaLamlLVcJBubsaRZFoHlzBVUchC6D4foxS2Dg/aatDwuxO2shTRkhRMjt3RR22vQEtTU/HzXDNb/kCi4jQ"
    "l/HZFEqZgnMizxigg30xmrs9N94syYNWKoIYA0UUmeB4acJblu0LQw21wHwF6L01bgK7qbcE9X45V7dQL5NGJW6h642I9a3h"
    "Bk6ifsoaxVLvehFHuDRTyN6EttWjoiv6Ep96LJclq9NmpaJOOBsd01m5mkoFzcwsJnJ0JszERTk4xTQ5m/q0fxwOqMWY2jgp"
    "NQNzckAUQwjUD4i6tZdclJETZOCD+1MInnKVtzmJmsIqgwfazM0wJEbmuwh3F2g4wk42CIr2VJVAaJB7Znn01gYq0mP9CG8H"
    "8yWS8btxqOq9JQuGDoDxxQ7XGBYojJ3EDDaApOznRj/eHk/iYC5KXjbarlaQuNrAaqAKhJvlc2gtBKfZwIpGuyzSdhJnzqDN"
    "douPFddHfX1ZsuhxkAzMWU+bksHh4Acww9PumPRKRcJd+7AXh7G/Z7EBp5/wGPKVTYZZOLqLdGyT4XAvTNdT3uUOG3j2/1Jw"
    "1lgWAgAMPBAFbiXu7nS4y8fh9Pv2EGmSyfSnKKqQ7waP35UVcF8+hS6PCynBc+6yiu2SpOW4XKdRHIDygR6X6P/u5ouxpjK6"
    "TEltgr5OfZAthV/NHzuhfDY8P0paNmS4JCFd7GBFqEKQT/037BfCfvPdanYR3QxNE+wrwvLUQmYQGM25fAm6Zn23wK7f5TGt"
    "AmBZuyoGRjcM2EfGTmXIuAPLhCPbOIYSlQxc0XAdw6G/TgddVK1BbqQGlOveuBjj0O7vaAfwjgQUXh53omJ6+H2p0T40IoIU"
    "elKvXJs7bskpqrHneZr2usYdTFLh2+X3DSzbTZ2KTZHTp7MlcqXlcY4dJLMTFoTV++WZKlAMzrWumL0g3IeK2Q/wrnTZ0a9B"
    "j9j4uyWFF4VLouXeAqn9G77VGtqzCRvVQ025sl7jwS8J5BSmJUp1W63VSKni+xvgu3uty0X9Yr2HCWI9hlt8Kh2B8z5dQ96a"
    "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"
    "fWrezXKE0GQVaAKWgLsnQjiFXfQlAGkSGmNxQrVrrr6zRoAMN78u+JNhVfJn14cUUw/2Mu04X+DhwfMMjRKH4mVnOvmZueh+"
    "2XMUV68v+43cton6USlcUlWBsVmKlvOPbhiL/j0W9vwetIyFCl1rFjEBeQpDvFB5HgD3UydrNQ+OQJuKzUaLjyQ9/Eyj/EG1"
    "E/5G3Ny4FCSBStc8MBkn5rxajR62UgUymNSjLtX0L4zza1uJ5qY4Cq9kgbK6sXHXzB7KRaC0e5HvU6Mn4kYaXs+s/ehN4m7x"
    "3vd+X+q6MotOpeKMRSk+T5JAsC3JYYGyZTiWNNkOOWBSD5pXoLE6TLtGFV8U7Qd8bnh+Lr+4v6VsKK59uRUyC8BOVf18VvDW"
    "nOQMXUD4QTfx41ES6Zs2Y2O0FgACYpaftzeGLzBGzW1GNPaKDm2LxxnReT6maEWcNOClAxVyQq1Zd8TnQGpl07Aypdqd7i8h"
    "TNITGDOBhxzNVnnjIugZ2oDPSGSrrlTbmqq4pK3zxcUsXcBvpIj5R9p8QJtRSGy31VjCdm8vPwStdgmXQ0LLwMvAIKDYDqLa"
    "2On7CKpXHvqaHYWny1E8XeE2VgxT7F7CbEggis6NJHiYg+S0TFraiBoeys6La3Bd0jbd5cNCe+8VWl7gXnXnpfEDjhYgA/v1"
    "RLtvrf764dI7SRlfk4NGqEQ1yA/DQQ1/MN55LLZfYg/ocaozMWH34C673xkgY2hWEdxIwS0clGxsegMTnZH5K92tszippLXy"
    "1kSoY5W2v3ML525R30YR09uQ++XObpkkxcbLrzJ7fsQnC0lQcLUWgaAWIKm+9qe9iisi3O81G6u6pAxqUiCiUdqCWCfj7LXj"
    "l8wj2WKvQuqqBcQy+PLImA3F9wTJzXUPAljcWG8f7z5kaNcFsL++o19aJwl33b+KIhbs9A5VabUO3NLd2ctDSwe7vskX6COB"
    "e+EpYbeFltQ8gpNg7DbzxeuRqTojlSBLpZNRQlhm/OEz+LzlybTHL78sKheq6946llM4nDS/7x07VYv/8bxYWT2UIzCqu9Wo"
    "cqFoXrIh1SrPU+d1sddOWfIhFZd9qmSmFQaruMKN8gioF7Gm775OPj0s8nwWrCwZTX73NXjgKmfuO41pNZKqUGUzZI9P46fC"
    "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"
    "RN8+eBupravh/vgVKVvtu7Zn7Jmv+l599+J+3GqZ49npZHq2CyeXd2fo9l2dD09g07zfeh2dnJy0+/Xhm3M/32udeqdP15eH"
    "9TPj4fHqaH/08fbW2D/amfuWeRwgb3N4Auz1unK007hTT55vtTlUR452tf9rcnUymO/sXuwf9Rva5mFTO9eHLdToX1/5u+bZ"
    "m/7+69W9qY+f3k5nr9tXx9vTwzPPb/U/5tsfe7PTtmOfOqfbO431dbjV3758bT8fPZ6bbc+fXdcGjflse3/0tL19dDhUT+qn"
    "rdnHaDQ52P81rN/uHmi7b82WdtM+mvunNvJO1j1Lnx6btzubg0dlNr0fnW8Hl5vayfuZ8/yxvnexffW4e648nB1/tLYf98fq"
    "YfDc/ggePu6vp6c3e43dHde8aH9cB1uBd353CLc3N69R/VZ9Cp4C4LzVHq5r+1brVju8G7re5U7j4VE9Uh/sO/9JO/CPbpF9"
    "A7aPR/vN+0v3PPABfP6wna3W09lkc/P1aNeEF33Vn1pnznpT0ebTA7MFHu/ch+b8sr+zY/wynLtH/V4/fzjcQvcP9SfdHSNn"
    "tmNuTu7q6+rb6dGje/l09TzZevfervuzyZE+0psf7qh2dvpw53vg6HQHPu+1funqh4uaT+htPG441472duu8qoMT7ej20now"
    "H9vN9+snYKu1/t76+93V3snu0Lmd1m+n9QPvaubYJ/VD9f74bOvdHY+enPFcvzenz2+3V4/n56MP7/Lgbv3i/sZ6P/HvXpvO"
    "3cW6OX/cVd6nttm/O7UnB76i3PrnVxOtDW+25/f7O/b4zq03Jm9H08Nfk9fj0fPW3k3r8ONae7zxH2/vTvba26/B3kSZtfbP"
    "pnftj7Fvmi3Qf5iC7a3hTftNvdszLsDpzsX5/ty0g7munTVvNp8+mq8D9fhYNQ/0t6vX95px/wbQvfPW337Q2q1j89qondye"
    "DQ+u148vnfbF+duT8qqfX9jOlWse/TLr1nvTOZ9cHm8eXv9yT4YXr8c3F/OmPdaP11ubgTbx1k+P4Nb8HFycnG1azuuVceke"
    "ve6fvdb7h7PL4/3gsKU2Li7n5t70TXs2h8fty4ttY7R3f6s0brbHj/7F9Zl93Ni3At2391/B/tPgfO/5de/XfDof3T0f3tkz"
    "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"
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
ROTATIONS = {"auto": AUTO["step"], "right": 90.0, "keep": 0.0}  # panel option -> rotation step, degrees


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
# wording (and pt-BR, es, de, zh-CN) keyed by `code`; tests keep the two in step.
ERRORS_EN = {
    "bad_number": ("Spacing must be a number, 0 mm or more.", "Enter a value such as 2 and try again."),
    "bad_step": ("The rotation step must be a number, 0° or more.",
                 "Use a step such as 15, or 0 to keep each part's orientation, and try again."),
    "bad_res": ("The grid size must be a number above 0 mm.", "Use a grid size such as 0.5 and try again."),
    "bad_copies": ("The number of copies must be a whole number from 1 to {max}.",
                   "Enter a number of copies such as 10, or leave it empty to fill the plate, and try again."),
    "bad_rotation": ("Rotation must be automatic, only 90° steps or keep orientation.",
                     "Choose one of the three options."),
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
    "spacing_large": ("A spacing of {spacing:g} mm leaves no room on this bed.",
                      "Enter a spacing below {limit:g} mm and try again."),
    "clearance_large": ("The plate prints by object, and the printer's extruder clearance radius of {clearance:g} mm "
                        "leaves no room on this bed.",
                        "Set the extruder clearance radius below {limit:g} mm in the printer settings, or print "
                        "layer by layer, and try again."),
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
    return fmt[0].format(**args), fmt[1].format(**args)


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
            "limit": report["added"] >= p.copies and bool(report["added"]), "max": p.copies,
            "tower": bool(report["prime_tower"]),
            "clearance": report["spacing"] if report["spacing"] > p.spacing else None,
            "keep": p.step == 0, "right": p.step == 90, "unplaced": list(report["unplaced"])}


def _summary(report: dict, p: Params, label: str) -> str:
    """Result text for the panel. Says what was checked and what is only an estimate."""
    lines = [f"{label}: placed {report['placed']} {_plural(report['placed'], 'object', 'objects')}."]
    added = report["added"]
    if added:
        per = ", ".join(f"{name} ×{n}" for name, n in report.get("added_by_object", {}).items())
        lines.append(f"Filled the plate with {added} extra cop{'y' if added == 1 else 'ies'}"
                     + (f" ({per})." if per else "."))
        if added >= p.copies:
            lines.append(f"Stopped at the {p.copies}-copy limit.")
    if report["prime_tower"]:
        lines.append("Kept an estimated prime tower area free, brim included. Its real size is only "
                     "known after slicing: check it in Orca.")
    if report["spacing"] > p.spacing:
        lines.append(f"Print sequence is By object: kept {report['spacing']:g} mm XY clearance between parts. "
                     "Orca still checks height and print order.")
    if p.step == 0:
        lines.append("Rotation was off: every part kept its orientation.")
    if p.step == 90:
        lines.append("Rotation was limited to 90° steps.")
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


JOBS_KEEP = 10  # job folders kept, the new run's included; only failed or interrupted runs leave one behind
JOBS_MAX_AGE = 30 * 24 * 3600.0  # s: older job folders are removed whatever their count

PREFS_FILE = "settings.json"  # in the plugin's storage folder: the panel settings of the last run

WATCH_GAP = 5.0  # s. A page that polls every second while a run is busy has watched it; a longer silence has not


class _Session:
    """The plugin's single authoritative job, shared by every entry: the Organizer tab, the
    dock panel opened by the Script, and any reopened or reloaded page. Each entry only
    shows the session's state and sends requests to it; none starts work on its own."""

    def __init__(self, storage, clock=time.monotonic):
        self.storage, self.clock, self.sinks, self.job = storage, clock, [], None
        self.lang = None  # host UI language, e.g. "pt_BR"; read on the UI thread and cached for the worker
        self._prefs = None  # last run's panel settings, loaded once from settings.json
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

    def prefs(self) -> dict:
        """The panel settings of the last run (spacing, rotation, copies), from storage/settings.json. The
        file is the user's to edit or break: invalid values are dropped, an unreadable file is logged."""
        if self._prefs is None:
            self._prefs = {}
            try:
                raw = json.loads((Path(self.storage()) / PREFS_FILE).read_text(encoding="utf-8"))
            except FileNotFoundError:
                raw = {}
            except Exception:  # storage unknown yet, unreadable or not JSON: start from defaults
                import traceback

                traceback.print_exc()
                raw = {}
            raw = raw if isinstance(raw, dict) else {}
            spacing, copies = raw.get("spacing"), raw.get("copies")
            if isinstance(spacing, (int, float)) and not isinstance(spacing, bool) and math.isfinite(spacing) \
                    and spacing >= 0:
                self._prefs["spacing"] = spacing
            if raw.get("rotation") in ROTATIONS:
                self._prefs["rotation"] = raw["rotation"]
            if isinstance(copies, int) and not isinstance(copies, bool) and 1 <= copies <= MAX_COPIES:
                self._prefs["copies"] = copies
        return dict(self._prefs)

    def _remember(self, action: str, data: dict, p: Params):
        """Keep the settings of a run that starts. Arrange leaves the copy limit as it was; a failed
        write is logged and the run goes on."""
        prefs = {**self.prefs(), "spacing": p.spacing, "rotation": data.get("rotation", "auto")}
        if action == "fill":
            prefs.pop("copies", None)
            if "copies" in data:
                prefs["copies"] = p.copies
        self._prefs = prefs
        tmp = None
        try:
            path = Path(self.storage()) / PREFS_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text(json.dumps(prefs), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            import traceback

            traceback.print_exc()
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def sync(self):
        self.lang = _host_language() or self.lang
        info = _context_info()
        stale = self._stale(info)
        msg = {"sync": True, "context": info["text"], "lang": self.lang,
               "ctx": {k: info.get(k) for k in ("plate", "objects", "name", "problem")},
               "stale": stale, "stale_ref": self.job._ref() if stale else None, "job": None, "busy": False,
               "prefs": self.prefs()}
        if self.job is not None:
            msg.update({k: v for k, v in self.job.snapshot().items() if v is not None})
        self.post(**msg)

    def _prune_jobs(self, jobs: Path):
        """Remove old job folders (snapshots kept for diagnosis): beyond the newest JOBS_KEEP - 1, or older
        than JOBS_MAX_AGE. The running job's folder is never touched. A folder that cannot be removed is
        logged and left; it must not stop the run."""
        import shutil
        import traceback

        active = self.job.src.parent if self.job is not None and self.job.active else None
        try:
            found = [(d.stat().st_mtime, d) for d in jobs.iterdir() if d.is_dir() and d != active]
        except FileNotFoundError:
            return
        found.sort(key=lambda t: t[0], reverse=True)
        cutoff = time.time() - JOBS_MAX_AGE
        for n, (mtime, folder) in enumerate(found):
            if n >= JOBS_KEEP - 1 or mtime < cutoff:
                try:
                    shutil.rmtree(folder)
                except OSError:
                    traceback.print_exc()

    def start(self, action: str, data: dict):
        spacing = _number(data.get("spacing"), "spacing")
        rotation = data.get("rotation", "auto")
        if rotation not in ROTATIONS:
            raise PlanError(f"rotation must be one of {', '.join(ROTATIONS)}, got {rotation!r}", "bad_rotation")
        preset = dict(AUTO, step=ROTATIONS[rotation])
        p = check_params(Params(plate=_current_plate(), spacing=spacing, fill=action == "fill",
                                copies=data.get("copies", MAX_COPIES), **preset))
        self._remember(action, data, p)
        plater = orca.host.plater()
        project = _call(plater, "project_path") or ""
        stem = Path(project).stem if project else "untitled"
        job_id = uuid.uuid4().hex[:12]
        storage = Path(self.storage())
        self._prune_jobs(storage / "jobs")
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


# The one sentence the plugin says outside its page: the Script's result, which Orca shows itself.
# Keyed by the first part of the host's language code; the page (ui/src/messages*.ts) has everything else.
PANEL_OPENED = {"en": "Panel opened", "pt": "Painel aberto", "es": "Panel abierto", "de": "Panel geöffnet",
                "zh": "面板已打开"}


def panel_opened(lang) -> str:
    """`lang` as the host gives it ("pt_BR", "es", "de", "zh_CN", "zh_TW", ...); unknown or missing: English."""
    primary = lang.lower().replace("-", "_").split("_")[0] if isinstance(lang, str) else ""
    return PANEL_OPENED.get(primary, PANEL_OPENED["en"])


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


_storage_dir = {"path": None}


def _remember_storage():
    """Read the plugin's storage folder while inside a plugin callback and keep it.

    OrcaSlicer 2.5 only answers plugin.storage() inside its own plugin callbacks (load, execute, tab hooks);
    from a panel message it raises "plugin.storage() must be called from a plugin callback".
    """
    try:
        _storage_dir["path"] = orca.host.plugin.storage()
    except Exception:
        pass
    return _storage_dir["path"]


def _storage_path():
    if _storage_dir["path"]:
        return _storage_dir["path"]
    path = _remember_storage()
    if not path:
        raise RuntimeError("the plugin's storage folder is not known yet; reopen the Organizer and try again")
    return path


PAGE_NAME = "Organizer"  # tab label; must be unique within the plugin and have no ';'

if orca is not None:
    _panel = None
    _session = _Session(_storage_path)
    _page_sink = _PageSink()
    _PageBase = _pages_base()

    class ArrangeScript(orca.script.ScriptPluginCapabilityBase):
        def get_name(self):
            return "Shape-Aware Arrange"

        def execute(self):
            global _panel
            _remember_storage()
            if _panel is not None and _panel.is_open():
                _panel.show()
            else:  # a new panel attaches to the same session, so a running job is not lost
                _panel = orca.host.ui.create_dock_panel(
                    html=PANEL_HTML, title="Organizer", width=280, height=360,
                    dock="right", on_message=_session.on_message)
                _session.attach(_panel)
            return orca.ExecutionResult.success(panel_opened(_host_language()))

    if _PageBase is not None:
        class OrganizerPage(_PageBase):
            """The Organizer tab: Arrange and Fill one click away, no Plugins window needed."""

            def get_name(self):
                return PAGE_NAME

            def get_ui(self):  # called when the tab is first shown
                _remember_storage()
                _page_sink.bind(self)
                _session.attach(_page_sink)
                return PANEL_HTML

            def on_message(self, data):
                _page_sink.bind(self)  # the bridge calls on the UI thread
                _session.attach(_page_sink)
                _session.on_message(data)

            def on_load(self):
                # Never call the base on_load/on_unload: in OrcaSlicer they are C++ virtuals that dispatch back to
                # this override, so super().on_load() recurses until the stack overflows and Orca crashes on start.
                _page_sink.bind(self)
                _remember_storage()

            def on_unload(self):
                _session.detach(_page_sink)
                _page_sink.unbind()
    else:
        OrganizerPage = None  # old host: the Script and its dock panel are the only entry

    @orca.plugin
    class ShapeArrangePlugin(orca.base):
        def register_capabilities(self):
            orca.request_permissions(threading=True)  # nesting runs off the UI thread
            _remember_storage()
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
    pkg.close()
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
    ap.add_argument("--copies", type=int, default=MAX_COPIES, help=f"with --fill: most copies to add (1 to {MAX_COPIES})")
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
    params = replace(params, fill=a.fill, copies=a.copies)
    print(json.dumps(arrange_3mf(a.src, a.dst, params, bed=bed,
                                 progress=lambda i, n, m: print(f"[{i}/{n}] {m}", file=sys.stderr)), indent=2))
    if a.preview:
        _preview(a.dst, a.preview, params, bed)
