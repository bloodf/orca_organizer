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

PANEL_HTML_SHA256 = "eaf3a0d8c88952c0a1e2a01b250551196a239ea7aeebf08d1bafd3f85b2cf37e"
_PANEL_HTML_Z = (
    "eNq0vVuX2zayBvq+f4WV48VNTKOVlpPJzqaM5nJ8mXgSO5d2nItGW2JTUAsxBWpAqC8R+d/PqioABNXtzMxZ6zzYLYIg7igU"
    "qr6qejpa1aW928lHG7utzv/rKfx5VBX6Snwi9SeQIIvV+X893UpbPCo3hWmkFZ/s7fr0y098si62UnxyreTNrjb2k0dlra3U"
    "Vnxyo1Z2I1byWpXyFB/4I6WVVUV12pRFJcUECrHKVvL8O3NVaPWHNE8/pYT/evToaVMatbOPoIXik2292lfyk0elqZumNupK"
    "6fP1XpdW1fqRTCU7VNI+suK7y99lacelkYWVqd5XFZuua5PCW/2oXj+S42ZXKZsu+ZIxO9NzMZkaafdGP5LiXD5S+pHtrgvz"
    "yIpDx7WYzbkRKRPnh44r/DGa8EJA3jGMyfN6JZ/Z9IwJISaTSZIMkieUfJYk6SD9CTufPHnStkeJT//3fxivqfDGFsY2Pyu7"
    "SZe1/mm3KqzMlow3vo9F06grzUuRSm6hgdhHIcdKr+Ttd+vUsqk+P8UmQZ9LmWo+YR2vfAk7U9saxne8KZrvbvT3pt5JY+/4"
    "3pdZjcuiqvCBr8QzY4q7sWrwL19DO29TCV1czmos8tGbYjdf8t1Dry6knS/59qFXLwor50u+gXfQHpgnIZZ+gpf86uhNY43S"
    "V0u+OE6/217W1ZJfH6VTPcskGY0kv4OX6XUqWdtuUslYkmxSObYbqd3PsrDlhvHL+wNl6wusmt9CIZdueBi/8N0aNzjSX/LT"
    "CePPH+oslTlf8ht4e5VKliRyJMTybfF2mSRydjaHp9NlkiyXJzvYdq+1TSWf4CKTXEoh0yX/IO+4kWv4t1jXBv9CWq3f63ol"
    "v5Lr2sg39V5bn4QPcjXMQSvLp9HTcRa9jYtxj3K1ZNxK6MSfbD7aW6kW57DZ2jbFPSdTzRjruJbi09N/3Hx6xd8JK1Nc+Ebu"
    "qgIWq+S0EXBAJ2xs6592O2meF41MGWPcSPHpP75KZ89Of5uzT6/4y3tFGMmXp48nS/j22/omfMuVDHlhA+IGHpZ/0lfMeBGy"
    "58taPz4ooDjdMlsuGa+l3ywj13/V0IZp8M14PMbdGaiQOJvqp3JcSX1lN1N9csLkTM9TzNfxF1ga19yI0QS+c4Wu5Fpp6Xco"
    "5jmUtV6rq70pLiuZjc641PutdE8TfmOUxd+GXxfVXma6Yx0v4wnDxfWqqgubSj9Vj1TztnibWpbLzHa8ivPDYs3f7reX0qSS"
    "ZW+Ltw9/tJd8JZFY7mXbCrcRr6r6sqjebVTzdLlf5v1j5jI0slrjK/jhE2+UXtU3mEw/s0FxUVHZoZuGM2GNh4JapyvY4a79"
    "kOHj04CZjIDZ4EpcpYblVzI1LFvD/1O1ThXzn+NBoZidyblQMznv3DjYTq3TKyQtQGCYP1rwSNlJ8ek0zUez/0vnf/kHgzW7"
    "leLTLJ393/yEfco3Unz6SZpns//75B//mLf/+Mfs/+bsL5+0/41p/x2l/Tf9aP/x6T/+Mvu/+V/yf/zlH59+etV3/yo6Ew9d"
    "OOHCztj4zRWdMp/+Zcny5TIDIoZn5E6y8bo2L4tyA4sfRtMVCscM5dlKNtVuFM8nSQL7e3Y2H1ujtimbCz2b+AfWdYzbLrRx"
    "EbVxuZy6gWNWyKmsGvnIT96/njIoaabnbGqgASfCnCwfLVnnS7kelAIzJ3HPUWaNmf0QWdfYvpnXfiWNpJ9OIG3TStpDWRVN"
    "k1ne2LsKNpiQoZwkGV2lltHJD9mglZYxrjEJvxBrJISclscd0nVl5bYp653kRVXVN+t9VTWlkVLzlWpgP+9UafdGKu1++HQj"
    "t7WF6b27LMoPfF2bra6vi0ohfVfNtthxXRMjxaM38EWjdKW05EYWq1pXd0vWL6VL7D91azSSbSvhNFv2A3QriWThIPnpGQlh"
    "3U83aqPJlGZrdBZ2oRJnMGeqn1V1csKMeAcTqubcwn86TI7pK70YVtqoPyRWCT+OKyS+ZW3qbWoZV0LLm0c/KW2/xBep8e0M"
    "zbLIKdLqkuJ0El4U4mxaPPUfTIuTEwYLQ82KeZK8k6nlZlZAi9lBimJ6aWTxASiCfHrWNwrohZg4ijE66/v0PPSJ2g3bjSuB"
    "nJxap6ZtVdumRuwofQe9gcRAZmAo8zAyGdS1Tt0B8kHeNcCfhOmJ0y27N1e+y4b2y4Gm65hVTA3jhbD3U5FcJsmoaNuRShL4"
    "g5NqYFINDpGvytNO4qyIW3K/bbQPb6Q/GdlBt62YwTS+gUVNf+cw2zPFi7nQWDs0Cilxgb9sGCY1vpKW6rFJUuCTRd5qqsYN"
    "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"
    "UKWCFhBwK7xdjNVqH2Hq38usrCUfY3JSTzzBxGtIVHD5l6hgaepWKmWmIPmjCybCPD458uCiLO+8uDRL7wTlr96jgxV8CTgG"
    "bx6pDcVLeeqo8yzghMq8VZDL0nw95ZSAemGpzEx1v526qQjgHjAuwL2//SrdTZU5CpfbJwo1bitJd55sxDLvqAcciD86g8Cn"
    "Z9midsCRGT9mZZndNvKusnxG87NswYLK7spk74mgszsORtmOz/iHSKwQjRLjrLbFhrgIr+kfv+DHr9dyAYgHvgS4MOosWyhl"
    "rrLZvZph6CyIJAvvWmvgd9HAX4n2UBt6PRdq16FSwV6x7xi/PmDtsGdlWouKuGjz7Mom9j88vDanVqTzmf3W4TVaPnz8EEEf"
    "dx0/VFO9eF4Ny64KdmbWqlBMIB4kfSS0/4jyt872Mocd7ncRN7zzMOnje9+g/OVSU07pWx5jrRYfUzEyWwl1Dq+aTSyliUjF"
    "pTAOVbUd7xR9wc6MR40jcY/TV7g758J3Vq+8wzYC54bvtqsdLiV4fU3JCm+liPLk7jrBv+KbeFXxQ3vvzr2xsgRRdp7g0cFq"
    "x+YzeVRTVaq2Ful5fj6HPAj8en4OgcFrkscahNQGROiPAhYLYAkOLpVhCGxEwWIZX8azyy1dPzN71BZLAqY42mIJrp+GTvCh"
    "3SaoETxY4c8Nq/9m+YBF/PXujJgW5HK521j+KWUysywvyOfltw3q75f/aFB/tfxOg/q75ZZB/c3yf21Qf7r8pkH9/VI1jr9b"
    "1g3qPy/rBvWfl2ATefXwO1X4zSey3WsocIlvki9Vb7BSpAreAkI7UdkjSkBGUioOF+iQiwF9wVqG+RSJq47S05z3FPzmG29f"
    "1epW65K8nZwtPfAuIc+W3tFKm1ysNtNDHSyJni490DVcrM+f6tr5rA0KlvbhUWd6XrTPZ224dPyy9I4+h/m5pvmu5u/joz65"
    "0Kd6+1w/uq4WwKfl/++2G4/VJlLG377THy44QDIK/idvMAAJn3bZu2pP/lJ6cjS5OP/SmcoTpfrZe00bLig75aYyaoVvkH0E"
    "dUOhv5ZgFSBsdVqtAggUrHa4AYELPKPcqMfUtzvqCTqdg9UbdsTYKLKXd6ysBGT7HksYlTkC74bI2ZbdarEHBzF45K3yv6x9"
    "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"
    "GauFxUnic3cHb63oqYXAuMc7Omh+ZRJWPB0sFg7EO8BPxUzEZ8xoyuxFbi7K0+RlOsGoUHZNUyU/O25AlFoVOV7XxKvCSOLU"
    "JJJ2qBnSa7YKgUV/FAzwQXRRpPA+qr/oJxUuSAwCJP1PrubcZfWYsT+dBtTtLa8wcLzClQ/0CiRtsfhlcOqZO9h2GidnXhDA"
    "2T5nzpDbCdkuVLNt9yH/Wa5FKnckhaJer1sxC0iAi3OOWyyKE0PBGDe3TN4w5RFP8bd4K6wPn+Mp1c5eSiL6YIZWJaYiK1Ve"
    "kbgGqJ7TIXgOmV+gxINQsJEUv5eoDKu8IaspFAF+QqXMK1cppcK/kOZjMBVjUIy2whJjMtPdKd6rAyzlNy8RgafhBqZULER3"
    "CrcwiZq6MloNkQBmBQ1gF5oCN40rdw2lBS/J6XAEW+46xP/rKytElwncBsY/Fe/8KkXR8bc1+bwjefsTxwNr4VB3xT1EkkON"
    "9zhRBmLMZev5HU5EfKm5y0m2fPM2h7rL0dURV74CCiCwB4cjyGwPe6M+/NO4d1qpVelWcXMUEVhrb5BKK76v0w37qSGXO3wq"
    "5nKPZC53dIAEDlzpVqkIfFcNYZVqjne4YU4WlhH0cH+bLCze9s1K0vigyzPDcwyyUBnUIABKC0LHejWsCB3sUZWdTcLxFY64"
    "CQlxtTVxeZCvI3UwMRqEEud5Tl7r88+AMy7g3BE5P1Iig9JCvPJKOJVgKy/D/XUJGeKb6aiSMYTpypZ0Q1SSh/jhx7rUIZdB"
    "1P+oJHyIWKZeW/MukyiYpbQ5dQersuGcfCnpg9VW00pgRuBhFE9sLel7AdXbhkqCD4lFWmxOseGQES0fEPfDjy0P26m0yQv9"
    "CfM/aYw0rW3uJxpYg15cwPCvj/R2GHsTmgjIHAtQqjzC/0VTrw8ndNWkFrXN6JgHSkOyhrPjLtLvwmniyciRYjBnHTYdZUZ+"
    "/m283vm38V/p/Nu4eP5tNFJqvmSn327Znyv256Jw8u1i3ZNvyTSm3uCPOfR2tf6ht5u/2KG3m6ZDb7ejIi+Wnz3bEneTFE+g"
    "8cNnnA0F6vOaLlcdy7opiuDN1zmO97Z6iOfsKx7iOVMO8VzXHuIZrzjEc51L9firH+IZ//GHeN6NnMlI6HM+Jbtkzi/oLq8y"
    "C/me+ZXUbJXxBiF46jLYLuInvsEb+hinxeo3/F2bNXrQEpRstYvAuHEeXdF4GETXjjv2BwMaulUQYRU4bkxhyXdFXcNmJHBj"
    "soEc9rWZIrRmauxhyi+M5NjxJm1/QqOZvKZa3nSa86SUQNsbsPcyIBEWKAkdRNeiiXhXbUIfRdchspblB0QLvREro+iuiNXu"
    "bUjVD5EVT8OPeFoTvSB0Jk7ovx3pb0f6PI4CjE6iCQ1Tj8Xwb0OcS4t14DYMdn5uHyyN8jO3nLYx7QGeOSmnKdUNnqhUSQQQ"
    "hQ+EffU45ElQi++UppI73qpAIFOOH4ok55A8W+QZgEgCbzql4eBs7Ae4KmM4aODASpl1zBpMlddt1o6BpdLPP6rigbfF10uF"
    "tEepbFCRYXlJvM9TkZrmaxyo2Z/FkLrlHeTgSUlcaB8LfIx3d/m9e5DRgkU0xQTnRZMb/jLcsfYGTuj/BxC/w3qhH+quKjmu"
    "iBasDGwViLhDL0ioa5Atb6bHkJwogL/iwoPnXjo2J96N/hu73WGmY7pkznSWgMnx77nTG5eIt5i8ib/M2LUGEZQjEcQ1eSP6"
    "U8szAzpM7w2iPhop5mU0uDWTfhwFwbto2kIS7ndIwgv83PLMNJrWwj+jw7TlsQKSsDSaOkmRKqjQidi7OlZ+/HRWpvGMihu1"
    "uSZ0IPiK/dTVzezdXRbgV+3MT+5JIyP7bQt0dkkbVBQ3p3F3tx9QLxZkqp8M0iB1ZQlvZlwJcv12ZISNl4a1DTX7oGJiGnZh"
    "8SGfIHdoOKjLr+rPwFwSOg/PXhl4LotNc1lu7WYGeTxy8pSeszSNwrVzevZncQLp5/gaOf8wS2jcSlCTqivmT8/qKTSjrbHJ"
    "pc5VIIyC+ix52MLmlJ2VLHxF+GXeBVuzpjdaEsG6rq4oy3NaqbGYqJSXYe/M278bda9vliUZZaBrZhllwAEs55uSjeb3JdkF"
    "d0POChVVDm0OZjGOyL8bpFAQObBhGe6/uUvRaJai6N6t2psWC/lZWXB5pltWsDHVbYFGm7GcJ4SWMCNvWkA28qY1qQrxmz3w"
    "E1haDZSRLMYqDzmqL9jy+1HYikEElKJCuleWsQd+rMdpYFQLtxpK22GU6jmuKzjB3vcCo4S/BJZXaSypcynBl1GaRpNqoc6y"
    "QlEY3Dbkx2WdAlAmQFUTO/LB2+KaFBM4F3x4JQBw76qfsf/bq6jLX+HQLIjAstS646pWaGjLEs7k71CjKFVf+Yl/6Qfotixr"
    "bs7FlU2zbW+YFrK1ymli+983D9rbp5+juqWpURljmnKj1mnyZNKkzJUvN428TSYlTtyJuBq0mwp2MllbHmqRL5PLGtnboLql"
    "GWP5/DZav8OCUVOHKV9ullGzYYfVEVeDdtMOC0YbcPDTO2yD6oSOVzkT0BuxWQQmXD8KZpOwidySbtnYqqlgHAYzf1DlZr5L"
    "VF/iM00OtVOinMINewxG77JE0UU7E8E/xVDlCBrM1WZbmpVryjtd27TKptmKtjH4T2mcwLBx63jB5c0bRv1Z0sLZp7CLl9uR"
    "tbixmImbKAyqxjYsWfeyABelhq+4lC0tIqqVVtgg7NFoOISY4Qbk7GsTM+quiChnwuZmMUsNvuy6gGWASOla0JVhHHt+UjKR"
    "ajuKwZU6p4ptlcWlYmSw9UuC6pS6zl0EfOZccRnB0lGiFF57gKy+zKDUhjvqt7yeO2u4Coo7NHINLdfc1HX1XV7d3TVeFccd"
    "WruO1lOa26T/Suvb2qqW6aOGYVlbeMUgXeNSDY586a0ay1kviq7P8RV3chRov+sA4nXcffgUEWzcuLWGTn0T1x44vKpPGDYl"
    "DBu3cr0hw5t5hwHDq7nLcKkWXTFYVt0Cw+3a5mtglrKalVubwUtvkMnpvePoQOx3Hhpq6c0atM6gqGvWuiMCK7n7cCgU36xl"
    "aw0EbNrmowAruMMQKJdbIf+rriRiSJfcSbSUt6zc2kxdep1RTu8d5R+x31n+1dKbNWgd+a9r1rryj5XcXf4LxTdr2Vryj03b"
    "XP6xgjvIf7ncCvmHCM/VLAaou/CWlVubqQi+nJsAckf5R+x3ln+19GYNWkf+65q1rvxjJXeX/0LxzVq2lvxj0zaXf6zgDvJf"
    "LrdC/lffD8fQLr0gbil/Rcm1WbvidjmV7juOBF7DncdCsfymDVtnPNQ3b90RwSu6+5goIdi0hWuNC97EzUcGr+QOY6NacsXo"
    "WOcSQ4Z6xS2GS7mdl12b0SsvQSzSf8dxImu580gpY9i8geuMlqZmrjteZGV3HzEVFJu3dK1RI5u6+biRFd1h5NSVXTF2OCGD"
    "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"
    "a84ocvaRN8+GzdOxKLZmwfEfGyh1z/XqiuMfibsNpTPgWoDu6uUAyI8PSUDA7Uc9osFywvX2yoUjMfPfXvmoIoNngJXIlbz4"
    "818M5XERYVPdvuRXBEuvHnsoHKYxB60IFDAYxU6DEycfHFWs2EA4NFhFDdpJAIMoa7GKfIOULQxOe1jz1S+h+I9q/WaTDQ55"
    "Wj2CEKznaiV0SHXpj0bEfVU/FKQbHI5jQDiKH9p+2kgKmUMGKAjAGqow2PO27DrmZdYdu55PPDgGitOyEXTFQAI9AMzMic6Q"
    "QFHuuJ7GoAxXnWJ+BqrA2ysXRnvDb698lKQaHMeo2+qXyH8wAUwuqHPfxy3+kxN2KMN0fhPCmemMAmW33eGW7Y+fXC6PYGIw"
    "POhLNSGBeiLmQKCVx6kDKQ8jui5oq+JQINFPe84q1GcMnIw9pLtOkgaORhI+iyVJA3sApX4uYeCQ+JP2WGqygIK2OsUtlN0d"
    "DQrOOwhmCBtX9TxXkBb2hL4TZwlwN58g4PeSIlJbCPrfvfXB+mvn1q++sXb98zsffrV286uND17Y/PbK4Obv1n/z0vpb3w2+"
    "+erO726s37y48eF7f195Phvop2/uvPu3za8/ygT3M52ufX9185OXqOu1668PXvxq7eY79PbvK89t/Pry4MYrm999s3HrvcEb"
    "r6xdPz944+r6ud8Nzl8bvHZj8Ic/rH3z6p13/7b2/Ucbb7+/8cELd979250P3v/7yvODvzy//vsLm1c/3vjwvTsrb67d/Iq+"
    "unvrNZRnGx+8sP7apcHLN9Yv3Ljz3PebVz7b+OzGnXf/Nnjj6uavv1279cHglT8OLn1CM7h767WNm79b//2FteuX1259sHn1"
    "b+vvvY6zlsF7QsrGh++tXX9l44MXBle/2fzi4t1br61dP7/+9c3BK39cu3lz7du3By+9OLjyDX6ZBO8HL56lz+/eem3w0vnN"
    "b6+s/+dz9BV1OHhTdAg9n7u6fuF/AUAf/m1w9tr6W9+tXT+/dv3Vtes31i//6e6ts+sXVgaXPoAwpbZ2/fO7t87haEm8fu3m"
    "zcErF9cv/3lw4dPN79/cvPja31ee37z69cblc4PvXhxceW397JvrZ98ZvPHO+ldnB2++Pnjp/PqHf1t/65vN799fu/Hh5pXv"
    "Nr69gn1ibP7Of7y3ceMTJTK//u6rm9/+r7+vPEeQyrj84OuP1769MHj9DxtvvpRE5Qmr6ZD84NoH6xc+J4zk4++3Vy5svvz5"
    "4JVPqcHtlY/WXzu3ceEPgyu/V8kzsxJ/X3n+9soFItjbKx/d+eCNwZuvE4pHI5QmSkH3Oy+fX3/nmgQsE21ff/ePg9+dH9x4"
    "KxdpX3/3j3cufrPx4ZVcmH3z6s3BG++sXb9cFGgnPCpRdpo2rPDlj++8/T3E1+/eOqdG2NduvkheIGqRCa/T7Nff/YooRBCY"
    "pKj1t6+lg+yb330H6Lh8KX4PxPDchfXLf6Ke7t76cOPD9wZf/5V6Xbv1wfobbw7Onac2f195HjjysxuDs++Lkd/+29r1zzc+"
    "fG8okcYx9bXvP1p/7bmMoFi/8NHgzd9kA+kxPdxe+Qi49tpHm1ffXj+3sn7h3OYnvx3c+AstfQY1t1c+2rzy2eDb3w3OnSfh"
    "snb9dZI1g6vf3Pn2TRAlODCBQg9FVwQcQnN75aPBG59vfHJz8MqnUmY9P3jz9bXrlzY+Ob/562+RnF6Ne9v87teDa7/Bycro"
    "ODFaLjZODEov1ej44Ou/rn3/0eDye/HKyQj54MKnqtpdu34ZJN6tlXgJZZB8cOn84OxXm19fHXz3Qhwlv/Py+c2rb6sRcoKA"
    "nisRclWyr792ThXfm1cublx5l6Te4LUXB2/+hwapun9feW79ravrv/3N4Levbf75/c3vvrl767X13/1l/dXLg4tvrd18ce36"
    "jbVb1zauvPv3leeFbD93fvPq13JGOJc/r6z/4WOS5yTR4qD25vcfDm78JRvSFgh85Y+b334r+U8EtFUBsvHhe0UR7Vi3FcSz"
    "1c+JhDY+fE8NZtPHyqskjL1+4fPBb74dvHm+IIgdv5PgZuLXKKPey8av7976QAaw/77yHEWww+rOUIlgD77+652Vc+uvfkaf"
    "hzIAFg4JgNH4gPQL58QuQQljr13/nIRqPoK9/u4f1798m8QadQLQqZFr0jXDSJW+Gbz5OlEdrnMcshYqS+lcjVgPrn2xcfNz"
    "kDqkqG99QGwCYvyda+vvvLx28yuiTBBUqNDWn/8UGBO7wqHSUeoEC3+9uH7hXKxIAP4Xzw7e/A3qmMskR2PmVwPSggTfvIri"
    "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"
    "fWtCvj9F7wLq5XnG/3n6q5oYNzV4sR/8e7sEf/cf/Hv/6anaFmsbLf+MDF+FnWHc/QvCeynjeD1TsNh4vloy/VWfRDdWT6Xe"
    "+cMeuaDA/+3cDP2XmdWY8H+dUx3vl9nU+HfNGCWP1XL07zUljcvfs0Hhm0LOq6HmrJqf6L7O6a6eYBUAfVRixn/z72s95dPT"
    "xTv/e4mj0eRSFVWeMurfpMSLrz4mtfzs+7tqv55bb/xxW/dvRpXm+2PV4gaD9FyZxh7v3J5qnFs3xOtvtM+4VPhN/yYNo/Ef"
    "z+X5+MOalmeIP/73uvkGfv+uIlJVjzUd+qd9ppeyh/Ni/PbtkmNkxA326bm/GqRJvwl+egJfW9+/V9tlA2TatGNWrZv7ElTT"
    "NuI7/14t/giatfTy2mwKJXv0vdyX2s+f/r//97lt3yVnjJrovkSlbv1u3IdpbN19u8zask5VSXcvNbxqU3BpfVsSx++tjl9j"
    "6ZZSzw8SV99K47Ljg1/CoNZtaL1WbahATW8SVtLSJmH1rcr+Ah36uSKiv4h/iXX3ckUvvUnc1LWEsPlFDuWnKk6FhvOLSNW3"
    "KtYZCPTLl+tzww/fpG10z1Xac6y7SlIRP9ykux2FqqYfFl569lZllzqj89hH1dsf3+rQ77+/S9nscMmbHS65e/lwLT4k388s"
    "dzP7jZu5UyPk1ExFZMRf//Hin3kKo+YpSln6+zn/MQuPVftmq/FHrnbNi7txiQ9c99FDch8nhvfjR9lV9QVKNYj4Q3P/yWur"
    "AqXbq4+QAXc49RE24A5fviGdufpYyddLEE0suQNrYMnyhCyG4RmQsX3VHFzPY6OqchdGz39M7v0gf3oilcQonzpRGXJGf2Qb"
    "8Menp78LE/lQa3XtRym5e7GCOGErwLFqg0yOzUN9Eqi0YpW6/vywMHZ1jEY7UD2fDVc+Q52L0UopMD9bl3yGqusMy9+bm1I+"
    "Q+X1NVadxZszdfnpLCwpn/eB2lTrrPYrQxvX/Ifu5V64M95Uc0dFXc83oD5luitkTl2O1jTjteKYqvbUP/FjKY/QHsGO+wh2"
    "0scSyOHxM9jZXQDStOLOvj934Y8fpbWY13wyiwtKV6tjnZUqK/Uc2ikDOmY5SbCbwCuOlX2vJcfHhuh8tu+bISmhp24ifPlS"
    "/ZwvsXryGtnM1dzp7uVsQfIm5mururnmDFXVbAhVlAvLVVGScqLfZtusjRpO+MeP85d3V4w1C+V8i1Bz78+Dcb4B6LX1kDUC"
    "5Ob34dLEa6YNa/na6qgfNvpN3czg7qUu9ebra+vh+r28Oq3VOf51VuVR9s8zO9/E1uos/zq7syn1B1leeqYOu/ZOOTjEr7Ou"
    "n/90CKos8ksWF5uec/QqoN7szbvr99Zr66F5a3UM4+38qChTvcW2Hv91ZYSScuYmj3Zj2llJIZ4uusXzpWLPrbfU/e6mKlWE"
    "VqcSz1ek4S2b+jbD0s3jSh292HxyrmvI7FRBrTfHB/98fPBr5qgpKKlVrdfSGkGuf/+2/IdLlZN7S4lLS5mnpBK1355+r3rV"
    "z8Yt1u7zsrqtWIkfPl2B+f7juXSZqZf0LypTb2FNC95MgsutW4/g00+9eXs73C979OGnZJWm+5fxSw2NcTuZkjS+UKe7X1Wu"
    "1er451RGA/dz6dpWZ9V8+vzZN85YeXfnx3u/Zrl+/LiEVAe3OrTV6kTvJuc52hv65V/5Nv9f4BOnJNZ9VF4mXQqUAePe/ycE"
    "gq3zBkBdkC/P1Vk1VMq7rcKljvU6tY2SjH/+vLzg/dlnu8MvXz7/cQnNzk9VPz2dD+ePRkcx3tPhuoc7pZDrzbR/MsrJXe9t"
    "/mMlJqmyqZv5ZN4lrfZlWZUA9c+d6PHS9dWYfwO/P1UTr2S7Ospj0tTk6em2Xs35OHjfrS/17XZz9c7/8iW6neZ+6ylq+Bn/"
    "jBT9ELVqt6rkqWGmmjL+fPdo+MNvyufLBmiUVubn+NdL6840ValoZid4Ci73wlHNxhLc3NX2/bxI8SBwDcU/r8bKQLzVqV23"
    "HoIvX4LzzHk78g0dtx+Ce7vjPwT3/murjn4zN5+e28/t3ZUBuGu+ljUsF1JgPN6MdcO2nUWcFdd2kXeWp+bsLK29pWilnVs5"
    "d95IRp+eK7zdUiGolzqvclq+NBqXykKxvMPz8NBuu6+vHbKuhH01pnr8iIP88uXz5ysTeYn75cv7yNdv93lkJ5Xh9S3ffwN6"
    "rhlXsXVNcTv+Y1Da5NVvj0bH+PGjpCoXdvHxM9Rq9AUfdFhZyY/77KnWmMSlkPSqk6n8KEsh451fLiwzeP7xo3qqYPJKTu7L"
    "l9i4OzvStTqfjS9fLtGb0CZJ1eV1fPu9sUdlb1eBH95IOysx9UtaspIXdO2GTS2NdioNyfPPkKyXSK3KBbR+/tcj+OVLuXs3"
    "6a+ROnfGBSytrGbFGv/48Tawhl59H1ohrbaqfC/B1xhnJc69HdN+YpjljtdAglU1qx//9QhdX/7rEQbretbvj5cE115Jjbuk"
    "AsLXq39r5vvuQkhLwnE+UNRVaz6UbPn1UzlNOrX6sop0Zs4f67BW5+IvVOKMN1WqAh6TK/9ZEvWk3tLqre2xeisDS9l+/OXL"
    "57Si8rVXtVvdJFN+aHUa0+hL7tXrY3I2r34u/XTPrNY5+I/I2P34UR/LkvtSMfrjR1JT1lYpW1I/Pz4qdVXerIeSdtdmWRWp"
    "tsvnR6WspJJY1fp5Mx3rniq/1Tvo74YbG5/s3V21sdQ1vhR3tqBt6lGVfj05VaU/v+3Hhid4qY7ktSduuVNVzFjduuaYft17"
    "Gl7j85nzaVirKqSs52vNyj1V8ukLJ3mWo33QI62m8V++3AXv+qYM/vz4ePl+1cdXjWk6pv7c+annwFan9s6tm6sZd2Udy5BW"
    "h7yrbYZdpahOTFe8+7vGO+Oh+VydZG+q9dq6URbo1QIoyUJjtnxZ1ZXM7bFhhj7/DNX844f/X2Dr5e0AgR37HaJtGVYStTMu"
    "71nm8fpuZDvvx/LCv7/ezp/Oh9PrplM770anU/YaeKHiF3Jb8UMVOWi2z5fmAH8Ofn14ef39Fx17EQP4VzHAlWyV4tDo9YbK"
    "7K6094ZoftBVUOddr9xcJHGbx5VAXiN4txHOFO8GQdy4KwerQWS47aSfZjTY+bjZFw/vckIpb3RppWn7yy9S1ebfZZrgtlP+"
    "eLMdn+3KO37NIJRaWuOu1SgoX8Ps7uVn07Urk1RfB1PbG1VG2K+t10oP/aeMQuUObBmNWcq1jLO15M/F2Ib+tpCOf66jf3e7"
    "rrK6u68T890quMYs6pjXvQTsaI81tvyH3WkZrhuU9h2/+B4GrlvWrOxZteH5LlxK7cddbr1/mvbGTqOsnf7lS3mNsr+yNefp"
    "/q7V+fwuwy9f/LvW1/hfvFo/adXT7T56fcaSsnUl7bvspeD7vVQ1qlsCYiNZ2Z4RpMndsVwu1Sl5rz5wagP9vvxV+5qdvMzn"
    "tjVE2Zo3Lbnyb3oN1M9W9zka0d1zVlIY27WTQrMq3VFnabQ6jer95+gVv/vcUY2/Hrn3LYMM+GZC5FWXl8Rg9/nxsTb0+fKl"
    "kuqf0+xuKl67jfw7df8wxbX657P/ze0rd6WXx8vZNq0yVQs78YPXSR6si2feHxc//uLW8TG7XCah3lwmcbzxk8yvbv/LN57+"
    "xK1VvGE0ZvHJDUiGb5TbN1GpWFdXfBTjBhaFuoClKEZ1WYVu3N5WsTPeXFcRGjf3VXjGGVLFMmqfJdO4Oi39YTTwA5nR+HYW"
    "RmO3Rxi14V5udC59+JAar69vDk73ldVWOQTldLp7ruxFUvu59Xsp4KfVal+6Tl5WvfgilYYsv5dWCnS5zGj10ejsyzstWq9l"
    "pCr5Qn0cqXflZTydl3q+P7Dqa6tzUh+p+E5VOze2igv1tfU76d2d1M5CrVjWn8zpmt+H6h6q59gpnsus7r1KMPH8n0oYPrd+"
    "/y+gdJoJk3/9x6dP/xWXpjGfIsN9/K16jC3DSH77pEVBecGkbdr+vx5KMJWXr1+r+wECP3n4jVP80ormt86nuChP7l9Tu/Pp"
    "qxKGrvG1Dul8wl3bdzhFW1bvo8BPOp9+WxpmYHwS6TKl4sdfYyOyd7/XWZdOKg/Nb0mAjTL+6BMX+MFvnU+/EUqsKbqtnAM4"
    "w3eDzici8OPAVeLOp0u6Jr/K4fIYP0BQeLwNir0HqPs2yNMfIPhtkGs+QOjboKP70AXfBnXLsMHbMPjoPiDXeDH0gFxfug/X"
    "2DF8W5EYeYB61zf0tqy499C9yWTwAF/TRYpup3HZKuRd2EPvfSTXvCm+CTu6t7Wo7okL3NsCmqCyhO5NBctL9fKvlYXuA/gJ"
    "Co+fuuHx039CAwiBNQjpfAI/IeHxE3QTCupvEweZEblKUSYvY8F1VBAadAfdYRN1p8TJwz3UjZv3Eq/34R4anN+N8l1LVVv7"
    "qhon24ju7rudT2D1f6j1Wt1o9bW0pnr5sN7l/5JI8ePy0kE/+Rv1A6v/BuDr/ynXwtedohkvzZNnu8VDszZ+jyPtIY3cuwpw"
    "uIwA5MFu1z0DDuvgcMyZGF7heY3mDobT1SMUle98D8OW+BL79X9jh1N7o1xzJhiwHG1xMeAC3MMAEVrg0sjELGKTrXCKJoKx"
    "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"
    "hbI5sVPxsNyPB/Nud33R+rVJrDcdSeeWPtmmlUkvJp7mZex8kMXVpC3vajqGjfXuYEfkuFHxNK2W5M1EJandvnGof+y2zfUy"
    "Wxll6pchMR/p8/GkVckVt2Zlhx0z1c5hkrtsdFHM0KvZgS5m+VFuj08qQo9ul3rd6kmUzRNLdNKj5OKS3jJ4s4mLNXY/2J5T"
    "wnRPG1NlvyrOiFymKQ6FVGvc4WvDWLOv5Hrd/QLbst2erAxUsfEhktI5rXR3/WayPvxQW3xv2xz1Tml5wzZjmaRJ7LRYu8Fl"
    "T1261+okJWU7EPpqY1vtbMlV/dhvVs16Bqd6/ZNYOeyJpcg3c/1eUVhXpmOMGhU3c7037MhNqiqZrC5Xt3R1wXQry23l43Q4"
    "rSfL+kQ+SuuZvtAalfOoZo7JYjHXalZH3daoOl7MmoOYstnMu11wW5rp9Gq1Ka+fKlV6Js9KYy23Ts6Kl0F3yJ8mpXY7l6PN"
    "qTTbXTrSmTmsmfHIxMuzEb7U65tOTJjJrRZzGY12M6pc1XXROGXLZ2w0pvc1AW8PR4PVadbti3j5yFyW0+MU32PmQGRayd48"
    "VTqJS+085Va1S9tsblZ6ZiiPjrsFIazbpRPW2h+1PdObUUt6oSQ32pliMn0qky1V9XEbl8qLFiFn1vRyWaZpwmg09KV6KTGC"
    "sC/T6lJsNjPCcIrX5hOTu7SMZlta0OQoNs7UetNxdi8z9NSctvCGnBQnzT4+1H9FwXvCtPHzB3zx70e0AN8HPHLCemPkSQyL"
    "ZDDM+gaf3MzLoLpofWEFXQVvDupHWv3cGJLoelzwQGs/7w9xRgsrmtmtNfBmF1J0/xgtSIKc2FiIcQz7w8YqXDj44OUnfN/U"
    "ftD0yiiioiFwwGuWX0Zx2BR4kTslWEHjrAd1GUU0JdlTjT1sChKtrQU5jxWcPoOGLpLewYul12B44Gsej2AR2jQUCP0osMbm"
    "Kbj3/HHDadzPvCKL5wSzEUQ2/gb6pNJrLi7Rghy9umB/2g38kP6MXi2cYGg/b+OITpggbzhNMJBxd49vQmKjBZoB7+EmvOOv"
    "aoJEa+dogaE1zl9sTQ8EYsosp4FHmhMKz+uckX8jOKmQOHKrnWAkLNySohgb8JQuLRsCLQq0zrFWa40DzUGRohqCJFw4kVvb"
    "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"
    "fNDSkMT3f/0/XJm9OA=="
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
