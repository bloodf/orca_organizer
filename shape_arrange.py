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

PANEL_HTML_SHA256 = "00eeb9c3b6b0ca2263905d119a2d8bfb4416313975410f1960a432fdf2628402"
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
    "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"
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
    "/ijY1PDrh5jEgUk3BoSjGJZiBI0kS1KZAQoCMIcqDPa8JSWPGZZ5VclHUwhmQHHKCoLuM5BADwAzU0ZnSKAodzSdSHABZ57F"
    "/ARUgY93bhztDX+886mXNDCLUbfdb5D/YAKYJpCnhoFb/LkTduiE6YwihDODuQG+3XaFyopx/OQiGQEnBkOHvvwmJFCPLakQ"
    "aKVuEkDAw4iuC7ZVUVkg0Qh6zjKsTxc4J/YQ7NoL/0/bDnzoA/HC/z5E5FkkwrdfY92AOw7CE9xq9fuSM7i6A5ahunF/qkVD"
    "/gNVr9bbM4Txnz76eP+tq/t339m7/9WTT77de/jtwcevHP5wp/7wD/u/e23/3R/r33/75A8P9h/ePPjkw3/svBwO3bNvnnzw"
    "98PvPg2F60Od7v109/Dz11jXe/ffrr/67d7D99nbf+y8dPDb2/UHbxz++P3Bow/r77yxd/9a/Z27+1f/UL92r/7Wg/qf/rT3"
    "/ZtPPvj73k+fHrz30cHHrzz54O9PPv7oHzsv1//68v4fbxze/ezgkw+f7Fzfe/gt++rpo7dQQh18/Mr+W7fqrz/Yv/HgyUs/"
    "Hd758uDLB08++Hv9nbuHv/1h79HH9Tf+XL/1OZvB00dvHTz8w/4fb+zdv7336OPDu3/f//BtnLUTjmdIOfjkw737bxx8/Er9"
    "7veHX998+uitvfvX9r97WH/jz3sPH+798F79tVfrd77HL71wfP3VK+zzp4/eqr927fCHO/v/+RL7inVYv847hJ6v3t2/8b8B"
    "oE/+Xr9yb//dH/fuX9u7/+be/Qf7t//y9NGV/Rs79VsfQ+CR7N3/6umjqziaF4Hfe/iw/sbN/dv/Xr/xxeFP1w9vvvWPnZcP"
    "7353cPtq/cdX63fe2r9yff/K+/V33t//9kr9+tv1167tf/L3/Xe/P/zpo70Hnxze+fHghzvYJ0bbn/zHhwcPPvfF2vc/ePPw"
    "h//9j52XGKROpL3+3Wd7P9yov/2ng+uveXF2htVgkL1+7+P9G18xjEQj6o93bhy+/lX9jS9Yg8c7n+6/dfXgxp/qd/7oJ8/Q"
    "Svxj5+XHOzcYwT7e+fTJx+/Ur7/NUHw0QtlEWRj9yevX9t+/5wAWip/vf/Dn+h+u1R+8G4md73/w5yc3vz/45E4kcH5492H9"
    "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"
    "tnauXWrXJze7V63ZfPdla/LcfG6ipWNspKmWG9zYGFLG+sezo1UtPdQ1qNamJ+aH1bO13okLWrZyWTeN3NCmrl6olBbOlsy5"
    "gen2vKQMjeTnV6fOLq53jF0eW6AtvUPZgro2VkrPTZlNw0PqYtuF/vHOsZ6FwYvKZOdatntgI3/BWtNHK/PjtfTCxfH51clz"
    "1WnDbp3QeszeSjbbfVkr1ro2S+Xhnunpjc1NXZ2bNkdyA9l8ZcGy2s6Vxi72DmUvdp81qi3K2fGucbVrtP+i1to2uXnOkMYX"
    "L88MFwZbJzsuTxVnuwbHC8ZkNttTpjMtU7Qj3TNlt0+rpjI9YE3XyrpaqvRcuHRpQRlVR+dHRuYvjMq10cqFs4trY5fOV4an"
    "9YnL1mrN0nOFEW1toKd8Kd01OW7NDa6VBuZsuXtuYf5iufNifq3TWutNp9ey47mLZ8/OdKnntIn+9bGmxfXp4dX+yoQ2vTh+"
    "oWesNleY7cj29Nt6RyedXR+unpvrWR1qn1uvGgvtBanSkR0/O2Nf6MqZ03Z+eN2Wz0vn8jRrDHUMLyrZUrklOzuRtcrr1tlW"
    "tdZ1vndqbX1o5pytGeu51fl+c2O03NGln5sY782Nnq1qQzNtE+2bCwMXJq2panEilx3ulc5bA/pmvz50URu82LkwUtSaRhcm"
    "zl2YHVm1jZ4LG+NdhfOLQ0r3hGrMjVRzSouS1c0NvXcxv9jVq+i11c6hTqXn8qo93itvdmwWJtc77LPq8FjHbMtZWdI259Jz"
    "C/OWcqktXV0cGzRmL3RMtg6fHc6ey52vzGxks9mW6lghPXlufubCXIvSsWl1T4zYi9K5+ctVc0Mfryx2nZvuvDg216NK7fJs"
    "61iLku/sXu2/PERb1cu9ZrHXrult5Z6x8xfGV3MlbWq2ZeHSQDY3oMyN6UOXCm39kz3ZdHVgqn2yo6CcO2dUsoo5VL5QnaeV"
    "C7mp3rladcrsaeqyLy009V9q7Z1sXbuk58qVuZ6RkQlTm6pelMptI2NGxW4y1f6WvHx+aCGv2kPyRG96rNi2uNrSsXChdfry"
    "3Pq6ceHi0PC8LE1c6mqfz3avjnS1TDX1nus821sutMmbg4NjZy+023PKwmx+aEY92zpzuef84HC+0H2+UKxutg0OjmbH28fK"
    "LXPDWt4elc+3T2ZHsvMtZ3ua5ucLo/l8l7zQNNQ71TpSotLU2Lnu7o35YrF19GzPyOhZdSg9r9i9l9JNhQtj1CprZ2earNUZ"
    "dXrMGN9cp91zlao+lO5S12bt3OZghzx5ttrStFY732HVSlMLQ+uzi+cHy3q/MlXZuHSpUEkrhfbOroVLnf1zg0NzQ+Xa+XxH"
    "i9VzXssPFNSegVL3bLVN71rr6T1vT0ibNDs/0DI/umrnJipdYyVlYXNiMlszF9dnihV5Vm3ND7dd6DGKrZXh4ek12rEozQ6r"
    "cselqblWuS03pI+2qGlavrTa3z8yMb9YmZDzmzOX5rILlzar6YHqhUrvWtvk5fLcyGB2YbxpavjCdE4tnNW6O+j6+GQh354u"
    "VjYmxkcqcqdxfrSl0FGds+yZQo8mdQ2VFsvtRcsYpqvW6uyEVRwrlrv75bXFS3O589WOpvLiYrkje26jVRpZHK6cvXR58PLo"
    "0KraOWJ1ZTenq0bHRG5jNJceHNmcbu+Z7To7NXhe3RgzKj3rm+ri/OzqeXq+MN45saoNds6ZTe0DNfNCcVEeUNunZ2a6xuc6"
    "e8+u5bMto51mW2Gi/VyPpXX2tpxfH7o4Ob9gXcpqG2Pl8li2yaa13tZLrdWO2akFdXSqabKzvXWgs6rVFitWtWetpzs7baYv"
    "jOX7u7oWB3sL8+0DE3bT6tmujdVpPV2aWBvaXG+fvbjQOVZdb7HKmxfkttzARFdvqacs99rzo3OrxSGz99Kw1Guezy1erqXN"
    "rvHq3NRcrlS61Lu63lE8XzUvlKfWLmezU/nLs0Z37/j0pf+ftj/fchPbEgfh/+sp7KxaXqGLHExCQ2TFjWYQEmKQEAIhZfnL"
    "YBIgECBGITueox+oX6wXg4aItPPe6vX7nGkLzsQZ97z3Wc8Cfz2fOvMxQ0QJOYWVdMLFQ3Uk6hG4LvKVHwkB1hM4DjO1QQpv"
    "KQ+dStDRpPnTMMflYiQcLH2FDwGNYUXERZF+H9/i/uG0nife4TyVB7y496HdSLPFQkGsEvLUCBytljtws58fwZOkLeZw6Juk"
    "VSSHYJetpn4kLunTFuwbwDJPvAG/XMdgzpqD/tiPeueC3Xl0HwkHpUiD0nE5SpMSn0/xXoYn03EOWRNUG8/K1HAO4yzZeccR"
    "F8D5LFMKpicLqIdC8RGCnG28FkbzlJ+QNIfC08MsMhdwBqFoMrUKazMupsaUZOf+wu7nALcsi0XWT4cJA634hOqNzuudOl9h"
    "wSgHD8CW26/mqy3Wt/1waCqqtZ/lPhbshgt+NxTRjYIg2pkLp8pwei5Se7hRl31xEzAhFfITb7PHN8sQPs4O/REwXA56Bqz2"
    "+qI50KG+QVsktdqPImuA7HwnAE1gPcjHs33syis0mZzCwxTmMrtnYkNxAyPCZJLbM4jv+1NPN/3I2qdTZnagmHwAS5mu6HYf"
    "FVTBi/bacJjIvMMcBWeMU56sj+fE6DifkZCumgR8hIf6Ig6GDFQSpjvvQScxY6Okj9LQcl/sI395HjiYGG/iPNkkuYNmM8df"
    "ZL3ZYIDnR2lozYvR4XAoEmjq+nK6jsblKTyAw/WSYgjGGym+l6g9Np3ph6DPSHO+3MzH1lCYsSo9i1mKNqd7NzkBkjXPI8ou"
    "Ek2e4qxJq/GSsqHDycVtFPS12WoiWw6arIXzZOO5C207BS0+FbfqGC+hIa3FQQnNppmtKtlmvKC1AEmF877AvGKmHwkInbsn"
    "AcgU/gzBU33kLk/7Fb1j7X5KzyF80i/X0ZQa7iE9mpwPyWrfW/ETYkuIujqjOaDXM7PZYdOPCQqfS4aElNuzng8nXr/PwL7H"
    "b1Ofn51n/ZlElzNxihTL88wURFw0hHPEMqA76W8lcWcuse1gMWDYmdib8sa0xPq8k3rluRhw+FRjIr9AhamIYcBmIZk9dHi2"
    "fbWXYH2w4Jn5bIUMbObUFzCElfGTqy4XE2krbmTakujecaaF0/lxR5mu5I9JfFeqe9s5rZBQzqQyHM3kqeAqRrBNuchf7mY7"
    "hY6d+XEyNQDYGjhKdDSxcE5IE3CyLqGpvCqwc4/iXXpvLgPFdpBASSAemEscU/puOmNReLD0hNw7LjZrcrDRQLGPKgK/VUeD"
    "s5n0bFkuh2MdEKJ8i6pn/mQletpbGHSUe0Tk9EpHy44AAEd9nAx4Xzn6RTbvz3YiAe5xnh3pE21PMyWOBgUHBztUng6n+3AN"
    "Jj64OHsqWHEe+PPza9d5rzdkTN9aVK5oP9EaNurCoFHr3dQKlQnDZ+EhaJjFL1+a58a1UHETV/etm+A4zq+C45YXdM1KAn0n"
    "Z2xqdlu2Hf+z001rm6PAqgycK8X1Rwlyxepvala/YsiehIdD+U5yefPw71b85cfESjhyLxKoGrmTjdd6tveyyUtWKz24McYV"
    "M2pXettaxtKIib51/2yYy7tpnoX6LZzwrya6G1mxYQXpU9zdh3qtiHUbf7NanKNdHLPDVtGaPIvpQzWMq0mDHDQ6m8ZCo9UV"
    "vTweMysum8hglVHFY8WO70P9ayNcee38bn35YlUGIHiaxq6epdbDZchuq15qijaWMekvW/4jDn3r+RICXtfib5W5xUXj8eXL"
    "g9ZWTD98rZ6+Oque/W5brvNUmctWPuy/LtupJNA7P0ucp9fKMOT1rVPJJLROJaYOPsg62uh5r/fKjH2o177Hf1ZJr3Gtb69c"
    "kZPLjq0L/HZb4d+aDXVx0m6lrXa1043wEGm1U17XseKw+q03QCXVqe0fu5fJqeV23d8urz9tuXU2rCvX9tlW66vWJDXPb93v"
    "jeb+/nToVWeabVUbHdwrSh5r99C7A9Bu22rHt8Yt901dtQ5GG1780lLQebz5Hb40Urx3cQLukn6uoWglk62w8LePbbb5Vf//"
    "mncVUl60E9qdxibsPPwrvQTzv9VLvOvr5b2e2/flq1Wu1G3X4/76fo3r4F1ZZQLSCjer+t1h1y4/wJWfS6aGxgCBNMSqJVN5"
    "BXouwOmuQ93yLzBJL98Z8iSV2+WpfG6lw45rmlbw29NrGmeV/kC6la6sOq1T+rUe11uXfI8/atwxtTSzDhz0K8BmpKcahMU3"
    "d1733lle67Yxq4Sq0fAS6uApacOkGB9xkH+ROFY45lQta3vM0s5jPR9v3Un88P2gxd5lA/iN9qqSGef2a/d77loFEZ6eXqFP"
    "0Cek9wnpXTxT2/jVZOiH1SzVguA65P3332pVWZz51m9Pr5VKPKzkrebTK489Yp+QR8SBUQ19RD9Vf6t24U/tW/4x42v75nz9"
    "WU71N//6s8aqnPMB+gQ/Yhr8OPxU/a0yoa/tW/Wldxmf2jfnZxlVrfpDf2mr+nuugCnx8BrXwey+n55ee4+D127Z/haumTpP"
    "r6NHtAJ3jT356BH7SyW49zj8S63eI/b3tS6fgrGf1+k9on/9EvSxyrvONVW+db/WSp73YPOqTGjN2IIGE70/BZ+qs5PGWWDU"
    "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"
    "NVzVPH9wHy9z9+NHZbhzaLOY/BozqdN1Lv7IbHlJ7VYJXbvaHdu7xFtUIffRSE/PLbT57D62a1EFAror8OVL/XO5Jejl0ApO"
    "boYnD9+jiwXKfcm3Tn01yCVwUIsJ6iJXWqsGIdUOv2+2PRRNvNYfPy45H+5wak/I5ZqW9mKVJ+tyxcpb5ylvha3t79N1iLdG"
    "W5ryrdPVfzrod32zw4fvzVfvct86T7f86m6qTvf0r5uqeNi/b+xy1VWnK/3r5i5WrT9p8jozTdptdqrFIX/ddPP8t0tQN1Fc"
    "m7jajFyK1wkNlrcfbvmdt85T+9bpWtb7/VGDpAa3dp7/eaOA0mrnps9ua2VXix9ernqoy61Nr533YP3hrit1gU63lpXXB/89"
    "ffq+wcrk/gYWD4n9ot7OkN2tkzrv+IbgwjcEDVXUfiht1HK3r7VS1uDx/fefrl1OHx0tqSw2XtJa2H3P9t50cJ+t+9Cnr1J9"
    "xauWPH26hUn7j9fKfaE50r/oTIO72hG82wTXa42eoZe/zOb99Vu/nNGnv1SrtaK/LF9pM6z7zZRmyRU6Pfyqc51ON7jUstrg"
    "K9ep7XRXbdbnz4F1iVz2cHl8DBpa68ePa0rNsTWpnU43/rA5L8Xewa/gRrAF/4ReeC11HuPqBt5K2gtaj8E/YAjqXBDA+BqH"
    "8NKdVQulDvdduPaxOaeuVYHxz5+la/Q192JT9uXL5z+vqfnlqZ6nlwtX/mx1NesjHG5muFtJt95t+xer2twNbguea/lI3Uwz"
    "zBf7Ie0A12NVRQB/7cbP16mv1/wP6NtLvfEqequrPadtT15e7vvVMsbhx2n93lwfluoPwZcv8f02DzovcUvIBJfAvU9xp3Fx"
    "SV9aKqr9xt9jj5Yw/EP7fEWAVmXweyl/uxXsAlO1GmZ2w5fwevHWuEUs4d1lWN8uh5QIQ9/SgstprG11O93GjeYp/PIlvOyc"
    "9yvfwnH3KXx0u8FT+Bi8dZrid3vz5RV4BfgbAfDQ5lY9rA4S9Xy31C25dhFt1tTaVc5Zccv5RUp7D9Aqc6tq67yTiL681tFP"
    "K42c+dYN6135vdWG1CZ21R2JxycA8N/euvOmE+7Npuf5Z5Tjly+fP9+Ix2vZL18+Fr7lPRaxm9YWsPf0/l0I6sS6iasbgNsN"
    "nqnKjqx+eba61o8fFUy5UozPn+FOqyb4yXxVffz5lL081neHJ5Vs9KYuqT3aKtniQ1AdKzt8/fGjfqpDllV03JcvofVwcWnq"
    "dD9bX75ci7epbZV6xpvy7kf7gdrqqw5EdyfkNOrBZxUheQ113BKplXlHrRh5/Wt4zGuhTu2M1zz/8xn68qXC3W39W6Hug3UN"
    "XFV1syaMf/x4n9iEwfyYWke97NTtXpNvJS66m0c3YYLUsit814ZnqnvWPP7zGb69/PczCjX9bN6frxVus+JbD2kdldys/21I"
    "74crGK3AxoWnaLrWZlRE+S2r2ibdRrNYF7qQ5s9NWqd7ddyogj63XaoTntMb9VmB9LRBaA1ie67fqsRKpJ98+fI5q2F84+Bq"
    "1Bd1VBmdbmt4e229fn1OL8a7r5XH5IXQuiT/GVu7Hz8abix9rHSWP36kDVztVCIl/fPzs9Z05d15qCB3ld0C6rQ2DqvPzbtt"
    "2MxQldfgzd8tP7E+ubuHGp00Pb1+5mK/2X6//uqNX6q/+vp+/lpK4HvNgTe+kBV+qkmwZlQtV37DOC2F8flC77QEVZ1S9fOt"
    "IeBeanH0lX68iM1+MhOddvBfvjyE7+akGffn5+drvnuz42tLXLjNenDtRDXFu3+ZSajTbfwlm+En1kPV5yql050/NBasvlbW"
    "fNMtCPlDa3//1GZX7rtPd91869zpCrL6IFTgoTWivZ7uWuT23JJEn/8aPvfHj+C/oc739wsGdd0PUUartAq4XWKlXkQebx9W"
    "uvtxba9U/Nv9fur+dLvdTWr3w2p1q1mDrtD8CnZrqqgGCy0S/d6y8Zfkt6fvb7//YmKvwoDgJgy4ga9KGhq/3UEb8waD74Dn"
    "T6YK7n6YlVsbu/s2boDyViC6L3CBfLfsg/VQLVbrJH8/SX/Z4VD358O++txWG0p7p0pzqtZ/UasxRq7qhPeTYr/Dyhcr527Q"
    "0AmVktZ66LT6ybe7c/R6MXp9fasVzX9LEdQemI7V2oZ8aKWyqX3rBpePBA/3B+PPZr5uO+vDNr6VzJuSN6QAdY3nJmD3T+fD"
    "sXw/rGwofpEfhb7/+tbEmy6t5/dWLY1rbIVD/7bujfjRq96ZX75U180GK9fwXh4fOt3PHxr88iV46HxN/mnpzZNRP93jPegj"
    "3iutOrp6YqUr92CFWfqgV1u65mfF8um/yjZk9ulXQ2ixbtXOfYelqsPvOnujtcwmwDlXX21nxQ+veQUFXN9NS8Op1Tvdk9Xp"
    "ttrxvxavSdPXbmn968X5ODLYQu/WnKxntTqwu8/Pz429zJcvteD9Umd31/HG0eB/0/ef1rh1/8Kl311b8VAJUL5fTLxqi6+o"
    "mzwduumTc/WP+vPq/FzeO4vl1yD8+l0Q/tOdb1lx85WW3rlHk/e2zpbVGjund6EFAqtCuWStBV3dokpYd8EkxtcQE5pVB/mv"
    "Yl3fovyb1rsw/zvrLs5/ZF0CURysxsvFsW5uLrbV+mz/abX+cLnVmr9JVmP/Rlrd6xw++dbb2zse57E2fqqWoNpOD6+1SUfm"
    "vnZ+r2TwsV7jjtvmdfWr90pla/J7ZUgQVycp1p+trl3dBdB5qwrV1TX9OdAfqltMut+b/f7k6m+dbqg/m8mDUnbvTP40vbKe"
    "OjyEelfTa/LyL1Zp7e9TfYHPa+KVr1VTj4dahPD6n1oUvXZ+/2+wcrOI0n/+x6dP/51U1iufYst//q1+TBzLSn/7ZMRhddee"
    "a7vBP5+qEBTfv36t46qHQfr0G68FlaHLb91PSVnx2F8zt/vpqxZFvvW1Sel+Inw38HjNkOp3OgzS7qffJMsOrU8yU9XUguRr"
    "YsXu7vem6crv4Kn9rWCsVZWnP/FhEP7W/fQbqSWGZrraJYG3Aj/sfiLDIAl9Lel+utZr26td4k7JEwxHp/uk5PAEI++TDuYT"
    "jL5P8u0nGHufdPKfEOh9ElKlDd+noSf/qXcrl8BPvdsL8nQrnaD3HUl6T3D/9obdfyvpPyF3jQyf0Fu9WDPdLKlG1fuQ9tT/"
    "WMi37z7fpp38+17UV2aF/v0H2qTqC8hdB6v7xYqvtaXrE/QJjk6fkOj06T/hIdxDDbjX/QR96kWnT/BdKmS+rxzmVuxrZVW9"
    "KoU2RSF4iAyRUVt0pyXp0yOMJO17Fef06REeXt6t6t3IdNf4qltn14ofHpHuJ6j+H+681VcBfa0Mnr7/tN/Vf2msBUl1/1qQ"
    "/hv9g+o/Q+jt/6rOwtedZljf26eD65dP7dn4PYmNpyz2H+pArVUBsAh3O+QSqNWERhPexok6DhK98HCCqR/huHoX+jguERL+"
    "6z8Tj9f7dGF4UxyU6C0hh3xIHHBQhpeEQtu4Q27yFTFmyHDiCMyEnOypcouPEdwVKZcoOIfAcYLEaWdlTxkil4tSSQ49VdBt"
    "4wBmsGbt8dLjLUc04nE5xkdr9Xya7UBQkCcCr56Z6YaM5kvAKSYSK82kiZwA5aAfHhPMnLOcIrDD1DQ3IuANRjqqwziPFxOb"
    "yk96kPGZ2O/vKWQHSJiSLfDhKVxPU8+zAjWQpvFZJNN5Dki5KbhoMlpsldDBQ8U90gAMn5GptI9Fh+wxW471hak2L1fqgmWw"
    "wxkZBDOU4MYU5SVUJE02RLE/hCLsLqVSjrxwRGtzD1NyzezHIoBtJd44Hx3f4oXeXJsawBQA0X18RrYoLZD+uDzuPVsi4bkj"
    "bRWTXodQ5nBhb5gXCY8C4QoNROY8Lg79JIzliBblJV7Ip7E/xU8ey/fDdZwJsYZMMJBWrFPG5SZoI6d415umpE9IPj5eUvTg"
    "bKEx6hhbNO0vzgGqmKann8URBhwAiY+Jeek79PoADnpZMNgtokBFaAFTtvCYQGFBV9X+zrDQfg5uBXdwHprDjbBabsSSXG+I"
    "CTeekJqIwaMc4Wd8fpjPqWKTb0zct/ClBS8NBwhZxh84CwY3mMlKn7O4Lsr4WBUGa/Y40dN1scBF1C5xRDr3xPFuEIxknejv"
    "VqoyLcGeWtjwxJOnh+mcZEpmbJkbqKR5gRfoAsFADBoEq9SUcu0USdrG0zxguMg4j8ULDmeFJYmfZ/OQiCGGHp1Q3XRPxiYp"
    "Sk1ekzM0M0HM0UuJPIIC6+NjwbBZdDNyCQwVjnsQTSUIFIeisMHdo+j47JQ49EVntpT4/snXHG12ZnN0qO5OCwoPVwyhElNO"
    "mlrUGPdxriwyKpvjZpEV2nwhHzclrXBen1l4fWWL+SvGFLFU0jOY9deLIIzLzGQdQyH7R9iTld4OBEdQmurotpCDiFZ4gtyq"
    "mKxanDFaoLMALI1oLObH4hiI3JCXS0JVTFjxFdUZEwrnHex4yyCRwXLADjQOA8xbuYU8LI7waYaW54OYgPGGmIzmAACMwLV/"
    "ODpkNlsCDs9uKJFNVvQaLE7hIB1ApUDKwHG3O1E7bjKa9AaSQjv9/ExaMDThl7ZD0VpPjPdbAQWcwWQHbicYFvtljmgHknH5"
    "fUmBCxrMOYayd7w9rhQVhV4s7B0REN6ZGNHwYGgZaa67cApOjTERlrPe+pzqPIEvqZJm+r3heqsRx3Tqn0CT6CcBDehIMT9i"
    "vYyfFmJI4DSWRgGKDDV+aosaLprLsZqfMwWbT8brPsqYaxffLGlpsTbLvm2wwYpxUKY4aoENZSNN2a32eQ9SGHG9P5hj+BwO"
    "iOHGtQZLZjbN5+tdavbNabZNMxDSszI7bk/6EDPNEt6QpMznRAD4IJUTPg7rK7/sDQF1qhbkluGnnhiK7LjEyIjcL3vJkvNI"
    "dybGbMQfVM6TqqC8LO06YixrsSGGSZKm4kTRQhOyLMgt+5ioETm5IbmymPRSfI/72jQ65czAhygO5/GxPeZtm5HIPN7B8xxC"
    "SSVkxQVOz6XJeOxtemMJO6pLQqCmE3Zb2sRsxnoqaS8wWsAZWZblPlOGunhIUDkmjqQlMGJ5PqydjBvQ6ricrVxZWfZ7Se8E"
    "4duRPe0ROMOwE34f0zsTpHc4v0xH5GlR4p7jUxTpSnsc3K5z4jwdE6Lis7uTbMkiQCyX0pIa+2s3lvnUKVVsczxniLm3JpqC"
    "wWskO3hZke1cIMJBaWOkGOML7IhD3AMpiIuDND6NBiQJhTy1ddZ5tFgLy2LOJdhhDZ+g9YHzYXhrhAKqrsySX5fT4TBNQqmk"
    "hcVyHfBRtoUojtTQMlHcmYCLDq7aY9y2w005X648ET8VJC6JcsoDlMUQFMOQYzvFPHbtOqnd4ETrRFABx2zLkC7F5QBjbZKm"
    "aZp35SPNeuZJlIBcNmgvliiUjRb5OqJ3zGKYkJ4zW+a8SgvO7HwG9cu3zEm4CHCjR+Azhp24yRkll/gOt/3Yzbb1Fyk8HDD8"
    "ykOyFZum25FkCKjrJdxB89zCT4dFCo78SbTzpQFTUAwR9haU7uPo/tRbmHsucycsNiDHJ4RLZG5ECgtdH8hpD42GC3jgDmFp"
    "SQeIsNRp2hp6O644saG7JwRjNJSWizIZS8Nlb5D4tnoCJcoZEHNpMSxUxLfR1UYculyg9MYz5swutCGyHi/1nhcck9BU9meH"
    "NJO15zjaKbQFAPPBGYcMOQ0gA5cTt5YirFlfHYoTTAKPmDLP7PPawyhpy4UTjPZzlJSc7IwIccm7Y8RzThwjDSJgaoLhiLQ2"
    "SrY1TaHcABujYEaxJ1qESptrghQOJQBmLEsNQppOLJO0T0gCmUyacswg2OXmlkPh/ZZUFU9ID4P+ZBwryinv80srdMvDKKdg"
    "MCZoOYqg7ZwjQzMbqPMl1hvGRUrmgLqR0ZhbzaUT7+eRK2vsiPIX3DFJB0N74PS5yY4Bhizt+bCSjUgQ2ecAgmKJqp0DJxbW"
    "+xM3IZQSgUKEMmNko5/RSFxspIM7KrTRAp7ksYAtPHBHDIGlVkI9SgDF84roOQnKlVNO7OnkZDGNyhG81wl9oGEcRIbhNloN"
    "lBAjgPFsWZ7IvB9xWwuF4POOGzLF7niAFpifxd6636ewfrxP59qOGuU5OMJZ+gzNYXtn89I4hE+rCTn2Y/VwLJXQFqeiI8kc"
    "ny42Pr3N0DMCgAvT52xzP+B3/Q0DLCzdkYaZKUuCvj6yA8KYQb1tDBoQMtjjlEieVMmBpGQzyRJqOtTLCaZw3Abb9XKDzAaB"
    "ZxYgtkRFYL6coCUwTT19mJXoah3B3G536LO7vbaiwUIFlrtsnmzl3jCgTJOJcwDGkXRKWcOQHCaFo69kF8stBFfRtRbsUUGa"
    "ekaCLOHlQuBH51i2A9/OAALLSfgkmX1wWgAckivWxAznfhlpkgxJ88NWjgFEHmEc4lO8bTrcmraigUOdD7shshKEHCbgYD7j"
    "pZMarrhZuLct8jBF815PUPXNYpTNpzt9kB8ce4dt9l6gD3VpccQwMSImS3i7CMCphdnzsUhsJi67NRF3D+ukOl3BESdjru5z"
    "EXKEE+zY38IABk/BMxCxjrgDqM1mADOzfKbHRm/K2itGoflC8riVmm5oZ0ajVgoDaHkMk4WTkLJGGVvXoMpstpyRfHjwjXg7"
    "tCVojSncVtYODLzUl3uXGIQ22t/4BBgLND6O/Q2jHCSdoyE2TpJA6eusclSXGrjgV+XZoU46pi7nB8PCeGx/LhN0HiA+ZKux"
    "5OwW4napkuxJ26icvxlhoe/1RyabH9ZwjlizKB0cEDTXBhMsH6mIvdJ0YpSmU+BYWPp+NYrG2QGyBroCAP6Cm4y24mK0kxZ7"
    "plxsrU1gI4cdZe3wheFk5j4ekTi3EviypP2pvCV8tQioZWSW80BOejBcJCwgDEMjNz2R8XguKUWnmLqiScYT4hgLcx6WfU9J"
    "Q3mnrU/9raMwtDGLuBVl4WNtZWwEHjvIUz8tTIqmz9DK32pFsoWDpQefleNqR6tOedqghLtaoom3l0qHlyXaSzfcZHI6UdP4"
    "KAxtoKcS5No0PGHATTeUhRQkTlHE2Mb7kLyGg+EyZw87HE0HOinE6jFQ9ZUN2I6kFqtAJvX9IAEOGCtRc0Ukcj7sleoYYaIV"
    "ZgSIPZAR5LjSMI+R/XzFGIG0og9Q35zB2KoPyHbPGPI4rcKoI09Pp97WR1ekMEYEFViom52tThnwZE+PEWiSwrHANJwiD9qh"
    "RMFJubLMJevS6T5I1eFAIFE5OinLEY/oVuaqSn9XkKu0mGRZxJeiZiXDVBRCFrACgdjNT6xdTnSvFHq7zbq/PJ9End+f7WPM"
    "5fFkLQw3lO+cbSwPmOjMniTlGKu2YPO4jS9L0s7XNioo/mocnJaBUwAQ3MN7CDcZB31Om/HqDuPRhQQPDcPYDia9rW4y4ZCZ"
    "mCUYwpvcGQvQ3NiKo4VeOGEsrJYsv1GV3NXOBOBZiGwoZ3OFLs+qfy6J1U5cnvbRdMGLI1ZaRtrAyCDf0SkWKhROOE62Eq1s"
    "e9wRsklHKBf9lFPgqEyzyX6WnOa2xE/F8zj1AHHMCjVOVxwWXrFrSXEHjBNuDhy1npkS5AhHRWZ5SfPHNoHv8VCjlvZqaZrU"
    "eEKRKbAbTce71YQsenvs7HJTb39eBxnZd0p235v61HFfsD3VgMrj3N4R1LnohRZN+spuPloqu6Q8zSaatcKGLsm6BD3xZDOP"
    "F4u4AIOJqBjDoR8uRGwk0kSf3c2x1dYZipy052YlthnuQGqMb92IcY6iTFA2jjvJoUeKgA6iRrldTUHQJCD7lDvRabpi0jzo"
    "qStnj6kr0joTpzOKe4Fp7eVIyRZqZhnmcJymILpXS2A4EjzVnR1Go31my1vhwClHWFuDunmO98dDZk1AHnSUneXrRIippuEg"
    "K1IdODxfwvSY4oix700ijPDZTUkSB03DVsR2zDPshJL50sBHGbkXGHGKqtOFC4Gq4ahBRnnFzqFIhme37KK3wil0AirKNobC"
    "HVicbMmeDVlF3fSw44nYgbYcnp0x7akhK/vn0VGMMFDgUGepn49yaO1VmRopSwXIcmC/HBfo3Nsn3HLuxBPEHIPgwbP3Wb7Z"
    "xloxASNoAXgL+Xzaz1AeOdNhD5+J22Tm9fgljge666f5dpG6cKKvwDTWSBhXOHgdz+1F/zwtEA0GXUhR0oW1BwTncJwECbKh"
    "YW8nMidxXgjrAxuDE0Xd+uB+jHIwis1HZyjKlmhmujxGGb0BqM9zwBnPTQnSlgXOxD4C0yd0ixB7Qso8kh3PNM9SAp0LB4vT"
    "Zu0fkX6IISVlObpKDtA0U4pDNDwT+SEem4Zp7gw1PByXQ+wUgFkxD1fRabjhxIifZ9vIh5aZvj6XxXJppZAWCI44SiBIGKSs"
    "jeUbWkUtVjtrBVTtmHPFdZlT2mEC8uhtR/4446IT04dWiortJ4ozncSsBCt5v2DZ6ZA3FNfd+7uZlSYaE8ELstRpQAwldM9T"
    "8gnfxxzjwj4Y4n0MxLZrd2urXuwXzH7gEMcVunMOYyFJIT6bHYKVNV+Lq8jjAm2QLLaUhNFmHvSmTC6MIoKISkMKs1MBwOhx"
    "e+oPC2xLyZNtgQEGCnjzaQxOSmy9khWvv9Zti91sHT637AMAm9GG9v2chyihF8d0bxYWvUgsscF8hfH9cWmU8DLb0jgkG2Pv"
    "aIO9hUid0kwlqc2JPGYUa3FllG3253i5VdfD/TnZH2hfOefgRJtxID/dY9jZnuoE4OQ57Qj8olSjocf2w140WwuBjqySXhYf"
    "BlngD6xFgA5xdo/3ZpSYMHvmOLc1obC4wuHccUSLrL/dzs/Dk8tzwnawKMkwirTiiFOiiB91RYn352gAassld2bkZNc7H/zS"
    "H+bBQWEwMd0eLCvXRLonEFtdZwJ4qU3OCTxan0Y79djfutqCjhdbz8nmHCENrCymlXQEZyN1Gw4VV7P8BZDCDoBO0tFAT8B0"
    "F7Nn0ztSWSAPDkvBWZiqzarwsOc7ugpayTaj45x2RloK58fDwCnm84HpqloK+7kDeNPzLI+AHQeuUn2HmQusNwbsiYtY26PL"
    "GhvI4bkh6iejwg3GPTCjz3uItfZBshFysG8IsQXt5n4vM4ajKbFwF2BAx7iL98IowWFmM9rA4Ck+RKFaWHssUsfTKEVYaIGv"
    "LW51Pk5WYDBketw6kQhUNeasVqrrGVA6Ia/CW9jWESha0D0AzFZrHhsyIlVMt+QisPRzcIRCZbcKd6mLmMcdBsnkcB1OAldY"
    "zWwqMdDhbs+YuAaSS0UzisTbIaUWGl66KKfajl2F81gEtz5IDcXhSRpLQBhFeya3UB8KlaNR5AB3PvcJDcUJ6YDDC3xu4jNs"
    "uMOYKc7YI5smVRlYlxMOMhA2X/Tj3YLaWALAp7FQZuGOd6iRnIUGspx6BHqGp7yQIoAvQu4CV3ieB7QE0XGfDz1zVmJGpk6D"
    "oYmez0UsY0cH2Z1tnRjTqlv6wBk8ZxCyYvzpeqD0vZLHlqsDqiaeO8BBpJzRDgADCYKDG90QgJJYjXqT3gxR4jw2WWtEFbgF"
    "HmN5NFmco72xYXpjK1a5hc4FicEwFCGLtndweysnLDYJwhJb+bBDiBUyCxZejxrPplh/aM2Ek+DPw82Uc8RQmo1JNAJAMA/k"
    "XrTUog10UiwI4TbeCQ0Oa2SeyY5izRTZFQMQMIGeOiosW9P2a2qGD9bmcFCi/XC4nZSexMcOxbKkeVB6yzXMnHp92Qg0bzGF"
    "pJxax5mNZ8R0LFT3Ey0hmu+xOQlpLHNGwJB1Tsk5Gm0xzQLU+UgYZ315taQ1FNrg8xlNsGJ+XGWpOxuTx7Gouo4jhhwCxfMx"
    "70Wr2DpDUZn3F+KEpNlQW57IpZ4pA2ItQTISLRVXLsXjqOc7VHRclqPQz8bRoGB1yzMJnjXnZ28mc8sRidtcEWjwpi+M1cLl"
    "T9wh9xNap8MNDZLUXC8CkRDH0cw2sdSXeRdmSD6QZT2OlTO9V7ED6enWdlM6DEmoU4Omj4wY93h1199ysDWxM/YQxkx/y7NQ"
    "3x5b7nwOQHjIHW2LVbZbhSQ37EamGc4zQnG/tCZL9riAjdOK79nIgtxEmEtyUYEaCe5wslRKbkAQ1ADw7fNgCA/FDU1gM31b"
    "yJxV9MLyzI2zkAmd3tJbJ/7xHAwVTtBFGFLdsle4qOT6psHSfWVj9lVnlk+pZVFMbJgaExS2xrFoLWa2yzIL0tEZkZKDsTla"
    "7afT2SnCBbbHnaSCsNEUTzVzNI1OKW97uHfWx8yKkQs0ULmiPyUY2SHH7H5CJcvZSBu4lGatFGcm6LuwCMdSHAuCIk99ijHA"
    "XHFnyJB1nB7jymNBgGlqw3HS9oSR/RnpSEhIcJtC1GaDeOwTcZnow36pFmIfJSmbjNlzAAuyQZqsQPlimCnWAZmTziziTyUh"
    "SNn0vAu3xGBhByt/ji6x4bw33DN8zx2F5HG0oyah645NJpe3tAEMQtgCN57hmG4P4GdrwgrVjetbQnYObMkrd2GWUuykUPv5"
    "IRoP16V7OEwHmZVO96k4RcRgOOb91SDxZOFgH8OImmwwbKBMqY0QgFscouRtmZrHIlHFLB8lphrZJGdArLgeWzB0PJQIxK1Y"
    "YzxStIFIH/WxqdB7WCA35xyUPXTLTXELSQh8Kzq4gTEWsIs5bzK1bHYwSyyPs+NE39n4KhmQyImg18hyMQ7EfjGH17OSGFsT"
    "SkZLPMDHNKBSkixgJr1ZgRGK73Yzca+pZ3OsK1zBRPsdNV4LvJqnYhwseCQgTcQF1fXO3C8ZQdfMsT2VsRERA85+kDlMcAaE"
    "cJvzJCTMJxE16/fKU9Q7+yAe9Q9cJqCposxGSbpP1omgiFLY87KYobCpwYfafnaCRcY/eFLhOfJghYGSNaOw02ZNIeBEH4zH"
    "xizE9mlIMg52nIRkbwAqS8F0jsdpkMDyJjPB2Jag/TAd0rAPLyE7FKke7c1jCo600mLWp2XYN0U54YRZf7shMCbYsHN6ss97"
    "4UnbTSMo0wYKD4/66jIoFZfJdMo9Adu+u94lQ32R2lNxHHAKHpVstJTmWE+J5SUrwqK+S9FDSsKRFh9xfybr+rGcU7w3DgtL"
    "ZdB+op1YdtzXB5Y87ZVOcJqxhw0nFodQFvrGsD8eh5B89Dd7Li6gQZqp0CYFKJ0GBvaGnCBYYq/SZUhm+uYQlhNP6e/7I0oj"
    "9wfEOzrzReTBkbYp9ortIm6JOXOP8A/I1CH6Vt/bbwplzhvZXPMHInNcDLGtHUFOufQ8YT2fUqtxb9uTCXI7mQZ7zGZFHOqN"
    "ub04cokshhTxLFBTXpqeZJM8wYw/EeBI3+x9lRlDe0vhkrUqawhOlaeYJQjGDnyzXFm8DZ7UDSav84Icmb2DcZgQW2+Ps4GF"
    "ouI4X0k5hZG4ZIEbjdTGfHZw+jokqUsMmQgTkZv4oJRYS2IFh5E6dpcYTZihRc/VjekusO0igb3tzhSlpShE6iA5pRywnyXj"
    "Ab8n80Q+HnvzeJ1iS5DfFmgQKDE+DpdImvpkTFts4uHkISZHHmCQo5AgzGAKj8JkJwWut51PZ8i0hwTUhhiAbE5tZsJwx/Ll"
    "jsADyjAPhXVINqGCCxEjetsRS4yy7OSDysaiTaBX2hGQbaeYuiUGy/1iOUFsVz6ygbQ9DQvvHBemSNOrREBFxPMFLU9F+UhM"
    "zJlV+GmmDwd7Y7QU2dLgkkgcsKLvBZQ58DbWckMvV8V0Impna6cQyNb0DidowOOuHaEHzoc4fsPtwHIw6O84+EjhAps6Be73"
    "0CVGOwvJSTRHjUOJIaIROgdnw/Fqd+YY6YjtJst4DA9Zl8QoCeBP8gBw1ssCncgbwWPOAmt7QrbwRzSvD7PAL6SQO4GHaBc4"
    "c8/EUD6lGP+wH6naVDDK5WxtYpG+JCDukIzH7swVI+YkUgyuKtlo7GqnA2MrThD0zzS7R8SjdrL7x/yw2or+hmTl825wJnvH"
    "9IAohd/P1JzwNJ6RaWcqKNniwCvoUoDCSEoPEc0Y2yGoYj1/xRe0vsg5c3hGZzM5s3gzk6OS2a1QDu2n83k8n5O2pS5ShJ4u"
    "9gC6IUqIJTyRLlnsRMZ9OQl5CoDksdg7QUdFXrr4mUTKAyOU3HkEJZgeEsliDiUj6xwyBbuc0iMo7sUrvjRHYHgYVOdIJOSN"
    "xw2VCTcMFnk8jnuuH7jl2J6Neea4NzHfG5X9NaDNKSDdrkdrYHZcTonRiOzPUWaxmS5X6xGwhYl0tymPiiLzk3DuOXEcb87Y"
    "RhrgvqOnosmkygyUBA06h5WSMZmrVH+PjMyUwofnDSKN5dW4d5xHlDhYSl6JAiy7OfUo/WBNTcgT81U6kZfaQDddKwMwSisd"
    "3opHeBGbWEj74pIa0H1bP3Gr82GwN4cHYjJWTqwPadtUTrFgEuvHnj22V7Ju7HBS0bLS4AdCQACRczqNTHx2djZisZxEaSAY"
    "E8IN+oqWLJYqvlAGR2NvGovVYJUTg0Oc4+hquBA2eDQ5oSFf9L2gz6LF1o4VU+KknQuuaGoZqyhMzM/bMpsK7mRGWqSBreS1"
    "AaW464fscaONUzaQ6SLFe7oyPY9AMzyFArEWxUiaE/AaGi2mC/EEhmVoeabpetFoqUObwlqRlD0gJXI9CbgCLmckOtdJac0P"
    "9weR6m9KmouSsD8h+hgfjw7znYHZhOkL+bKvuszednnivFo64l4LjA1IMRYlg7m0xWlWX7PWsg9v3alBENtlssZKFJfJ8UwZ"
    "01bfSumEcV1JIw6yJQfE0Fpv3SA3j4nla7PTIVptRwjtkDwlBzy2sUvKllThmJCAMAJ4KDhw6fJETi0JhiZRXJx6+W6kF9p+"
    "hfJq4gI4iiBHd8Sz+YifER602WvV9aUJucIjQaSJ2QnS14ACSBDvoIMez2drF+U4cIzoyqrclnsg2R3zOJhus8URMxfuZqBb"
    "kirnam8ICQN9s4RXyircRz7o274PDEb9aWCP91vGpLTeDD9bgWG4ymAohRaHzLccztNjK15lfZL3V8VBm+Fs3135fXXH+uKW"
    "23o6I2U+Y2/cULXgM4wloBuRyFHyskW/0qDFQoSgGAGPIZpHzbmAgINMBKOpT5dL1j5lbjGbofTA5IY0d1gQmyjCONJIAWkt"
    "a/jKOR3ks02ygcNLgUayc5WDFA6f7FlRmB09u9ySCTbLgoWGiuAq6knmID1iW2K5GY6swVJWcvXEhiGwQ9CV45ZGFJTHfL42"
    "d5VNAgBY692ygE14JoymayhHVwcTSth8mMNOaA5RTyixYOGnpx27QIcWAPQ3IyPb4kPY0a2hpqzPPjJWVlZ/Wh4T/gxu1N1i"
    "c/L87WFayjOyv5H6JkdFHOP0wPV6H59za5pAITscavwppo8LYUIg60pEvF5IOGfKhrplIO5snPtxmULBDC2ByWpBYQsjFwNM"
    "NY/DaOZtxxgPJMY61dBDsMl7KG156ASbcnQ5PB1ORgaMj/gqXK56eX+/mw5mp5HW04P9CosZTVlYQSwNS8WciehiJE2Xapqi"
    "Q31AS9h5jSfJ+Hg0USbZrBGkB48QeK5nKu8m2WgDLNcht8/44MROY4Lb9wwEHcn62TbGveQ4QOZQAWaMNdCDDFjswLK3GA0Q"
    "cAEvRI3oq8hgny9WZx8z1KDAyt6WSBMacUbrWTjccQv17OXlASIHtkHYrLnOQXO7PQ/RBIDJjF/pmUbkxv6syQm/KjKWG4Z8"
    "7kmikTBgTEOzJMpVwOyh+112lHvLeKCOpvHQWUIrM1yOMp+14nVfBPhTjirwCSyABX3cHyFTTTwZNzHLXrGU5WiDkcysDBQE"
    "Qp/WjmA68D12FpwQMmez8yCY7OGNgsVCDEqDebr1NBTfrXBRMhL0MBOOBxOTA1tmY3gu6/OJue6N9PHAJSYL+2wCHrte77CT"
    "7Bojq58KgDbd7ZMyYvLQmC5dfFWYPZ4l1HkvGSwCEXdcQEkw4Lhn8fGgrMi3HpRtnWQ8zmZIctKsQ+SikbZSYuI0Ri3IJoBN"
    "tiEOEwTi+5iXOwORBdylv96G6sDMwkKF2OOq4HrefjJYA8seRAM9LUyBQ3++6csj2jmOlyFbTgc0GoUyvdb5/mig2ku8H1Ar"
    "V1K8fYJKIxJGz9uhOZ9OT+jQiNM5opLmXFXX2FE8UwITn/n0MMFyq3/EKDegVMYf9yMkhwcwumVX6Gm0m/fJfbFIRGzsFsXK"
    "Ga9JOtdoOlOXbOHMDS11D4vTnBnrqqzkwGmwCQiwmImLk5Dt8rQcAjsgHMyLLOsVKd4vRSLq0dOUtw/gZL+aMBNqAq5n5pBT"
    "wpHvZTGcKNTUYQ9Anyv8Aa1B+m4or7jpmUMP09kqBUfbFJ9mWtTrH1IEiXagAapL9bhgOAxfBUzfmZwHq4Q5yScLLbPFZJAa"
    "hotMpWppS1UcgHJvPYkiwTy6vKtOQgZAB4cpSmLT/Kytjwm9P2sTT5kgRUnv3TV53PVHlDg/n7bjDbfjCizCQ18eLOZQShes"
    "E3nGCB0disnS7bvxdkUctVLhhRgoosgEpysT3jHMkB9rqAXma0DvbwYmsJ97K1Aflkt1B/UzcVIOLHSzFbChNd7CSTRMGaNY"
    "6T0vYnGXogvZm1G2elJ0RV8N5h7DZsn6vF2rqBMuJqd0Ua7nYkHRC4uOHJ0OM0EqR+eYIhZzn/JP4xEpTcmtk5ILMCdGeDGG"
    "QP2IqDt7xUYZMUNGPng4h+A5/38pOGssCwEABh6IArcSd3c63OXjcPp9e4g0yWRivdEczoJjkZwzuztxZEPcfiU7D10k/GtA"
    "0Gys2AEh+hrK9xmSAjV5OX/gmnZ95NR7eYlzwlfApQVw/e7JTFSAu+gdkU4BJ5zcgtrq5zNp13SGU6qjCDKjFI7n2GC683Kh"
    "xJg/m65WqT/XrHe00obSuiblsBvRMfdfBX3onS5tIrPY+fcbZ1jU+U5mhzd0yK75TaawjZOC01r+kRs0humJVyTarzmeniRM"
    "LEdi6Z32S2G7P0VRhXw3ePyurID78il0eVxICZ5zl1VslyQtx+U6jeIAlA/0uET/dzdfjDWV0WVKahP0deqDbCn8av7YCeWz"
    "4flR0rIhwyUJ6WIHK0IVgnzqv2G/EPab71azi+hmaJpgXxGWpxYyg8BozuVL0DXruwV2/S6PaRUAy9pVMTC6YcA+MnYqQ8Yd"
    "WCYc2cYxlKhk4IqG6xgO/XU66KJqDXIjNaBc98bFGId2f0c7gHckoPDyuBMV08PvS432oRERpNCTeuXa3HFLTlGNPc/TtNc1"
    "7mCSCt8uv29g2W7qVGyKnD6dLZErLY9z7CCZnbAgrN4vz1SBYnCudcXsBeE+VMx+gHely45+DXrExt8tKbwoXBIt9xZI7d/w"
    "rdbQnk3YqB5qypX1Gg9+SSCnMC1RqttqrUZKFd/fAN/da10u6hfrPUwQ6zHc4lPpCJz36Rry1vQS0JcDarQDbFCvINa8oyan"
    "iAEfP7T9W89JhkchIV20LWqVDhW0FRHR5YaVKyrOcIPki7E/VnojhqhK6LNAXj5hc1tzAanfYt/kl2TiRtuUtI5quNlkSc+g"
    "KbJ+BoHWBb/8yvjb6TwtKXhXYT2E8CnP0VfFzUtjSxGEjIr3BfhV4utFFznf+8+hROu8dwjwfLOOFw+IxlO+jk/H0hiiEO0K"
    "xbDf9hUPQa1I6m9oHaQTArv55f3P/BhroGOS+uxHURzWSibrFpScQMDihzwRTOBZYKNB80i4gD42cZGlLaL489luYNsvfUor"
    "9NYo3rH6oKJ8gFQ2+sEE7hpmjlu52/BrpwtoHkf1+JWbkoGZ22UVDXLwaJBw5c1V7E9J9F0waBFmQAGRN+Yvqs5m3MlHfaP4"
    "TWz0DxM/XNmpTH9Su+2sbNUh7KSx71aoLD9iHdPlkztDYW53wjwmlgyxhHp8N3/wYVvpDINuzZMqU+M/avOakGGHmTYmVN2S"
    "79+Bz5p4t7IFsgbe0TQCPv5+yrzHq+cgl6buAKdGwOiZOVLdNy52UFK3YxOMAvWjMLxH5twpLn8GKOCgiiiJu3UoYYkvGohI"
    "WONXPOyssZDGUr9nDw84RwJwYommHLKyMZukfEV/+mBT8zyAECjsN9Ld+GwffM4GSDottyKskqMns4+EH3CP28cYjXyGjgws"
    "uqw1GdvxD6WSBYOMI8xjQYIwUDwVSFEKzpHgzUUxB31K/wvuJ7B4P8TrhS3N+BoX6aOauAkeirXyl8WfCqU/Hc8TA9YiYu/N"
    "WTwMvehS71cbIWZK18zf4OtKCwf1yLVvuQotOOGGrg7t7vACpPbKn0fH4FUtyYRuYejIVEM0woVm1gEAO4jqLbHnOVn8XOZ4"
    "xDuNDxnjoTFkZk+xRwPXknwePVPE3UR+rxe9SlzqqPg4amSPfrPXi8HFI3qZRLKT9kUsYYoGQCd2uqYPhX5Wg5mMFe8jQjo8"
    "NVyiZtrDwQSZgI2Hsf+95I1ixgO+OHkwMWkcI55visR/PBnHIEBOJuDKk84kP93uWl/7grxK6pRu89QwfSAX4c1BCP9KTiha"
    "FNPJvLceUYjKy6IxaWLU2EXcNGW+iuwSz+P8PbMXSnssQbOZ+af2+3WY6xGpAJ2l9NC6zHlD9zhdYZJPn5AcdBNkWPnK16oO"
    "AkSMWQgvoLUbm3uHC40EZpa8X44etZpC9KNsqEcKzzM7yQ7Ysm1NOPBRn8iJ6XhoefDeXKjlrFYJlGL5cF4TLbRWfKDCCpe/"
    "mfdxH4LLo769Rrx+ciYtknFmeODGX1F8tNdEmsqz1IMbVqlctxjeIZNNrpgCBKyGF7SRIXwSzXfxAsHMkQsk0t3ve33cQNUx"
    "xUyjKN1qw2sTJvqQyvc6ox9uGa6qTvl7BjD0oqys6PaJ8Zf0KRiZ9xCs95wR699n9yijth0v3OR6iIbEa9MKoKp/t6SX2Afk"
    "MF/2Sd35CIDwaV/gg4GVXCbCoAa7sUC1l3Liki7yqZ0gC3So0M9VUD+kYRfyTBPeB4AGaCwPbehyiwL52a90IV59GPUzi3rP"
    "3HzmvGn/Pgt4NzarRaXqJ5Lrg5IVwWlmEkWCiwXu0ehDNTVwDgUJeffJ8/a5S7p+7vXdqfuQPZ5rCM4RwPu/i4K+K0f9hGTa"
    "uoAdQmQJzigym6yL2f5miIuKKJZPjTsrwtt/s7UfO1lom6Eg6ieo4o3MSzupXh7bBY/HkMGSDhCecF/2PxqfpT7U7ONGo2Fa"
    "cJfbhxRCgJr0b1n9HZ6Muk9NDkjgYABoATrL2GpqrKArA0YlN5UmpDpNnCvdpedQzQAYzRqvpJl3wpZ3QjOLM2iVVT7Pnm4D"
    "zTuN2P6KmP6+BwMeWAnDQOxofIlL0JaheiXiraZyGGTQCoQDc9Xv2uAY7joigMPHhdon07LAZSLxV5W5RwnPBD5HLkOkstmm"
    "vA7oTU1sKrAZRstlac5euB3HPmDYubvhr/6ppu5/ZAjzVv2M28WyPnKlV3mMje4lWV+KdY83JGbVKbrO7nHIdCBUtw/LY/Bm"
    "u3keWNxIxycozgM0g1OuUjTwDeL8PCh9ebXGrAsThpAw+txxWyYVt8f4VKfikGv39spvk0dY3N6hs6nF1Av5TpAUdyy3iPnC"
    "sdg5P3+abHzy7HRV5WHXHaac0azbc8HzJdLeDgz8qS66OoivJfJCpFHnBJPMll5D9/6NVQAHtHkcFVpZr1h0imcrmSaq6eSw"
    "Z78XVMAE0XW/J0HlZ3HXDCHrT//5bi7mNzeAr2yynQ7KpTvKK15euF+hZodCZVxb9JYh3xVCJYXCJv1eQ5hGIdVsYfUe90LC"
    "oaOq9HaNrgMiHz2XJe2BKV9Sa2c0CzLKkMNrqRloM2sjfP5jtWJs4wKmOsGHKZ11dRAqOINpGQdpnMDNumaelGeDxIc8XOmI"
    "bE/HAo+RPJ2K4mFNC+ujfC/6Z5rcaAo6RnBTGiDbimRBaABh3t5XclSRB+i0Le2lXLsTI0HZSKUvQdmJ6jWWGeBc4uxiDjBm"
    "rO1SKFuVpH2KeRlen5mMOmCMzgYi3vI2I0lwtVqxdXAJYnA4DWEmlVf3EAzoH4l7xqA1cpVvuD7lSmlHliffveYGszgZbNpD"
    "5DyB8gad1cfTcS2Quo/v4tutFGLLgc7fWCl7eE7wVD+u+x5XJT+HF+BaptvE/U7q6dze4ORSTSfRSzinIr0W+icmfr+GdLVy"
    "2uu0te5zQzUvmSBUWC25SqFiI7S/AclLjcf0+4QvZZFcetTQjFuTLaywKm/9TkgYLHVx5f7dx/am4PhBAK7bIk5dR6mhPcP2"
    "7zBb3K+0+YHrDo0UNH6fUJhP5cEQrFnhfwi+pOUC2tFQtcqIESbB+A16vKWocpQkrMS9rElhmyFUnc0IWJYNbuSlXRNgmNS8"
    "8LW4+3NLJUO7V/KqxS22KN8giDiUCR1eFY7Fv7DFOC5Bg7fZ6CDURuHURCRJzch91QmWXQXWl/8YNMXXxFYu8jDo5FbPDzWv"
    "A+vDwrXGP+8eRNnGJR8UJLRtDKVxjopnVeMQr0W6U0a+7Lg2FIbrQpgZrx9h2wuwowa05CLKBPzZcQFt6CLJlBs4qaCZtiZ8"
    "C5nipb9S5OU98Ap1CmfyE2X6fh3HNuTKSG2lnIHSyJNuEuW+0S2NT9NhH6wKntxhHCeIyCfCUZcfqqZqqVcOGc+Fj0UfgfWo"
    "oKN0rxQAq94FegxpvJVG1goxWIlULVX5mxgQXmT+5EFJcgMOKqlZqUu0rWw8MlfVwdTPuZipA4O3tsgovrLblttA9F7KRTRR"
    "ADCr3rwtvalMy54uIwO5SuRq7sqxTiMaIvPDB4DKKwe/CfKhgxS1w9qgr2Gpvc9xAfVSKWLTrKCLFlFyhXkdyS4Q6l0Rns0D"
    "s9XQUMNbZ4SrilcfA8oGVOx+V6205M4Qm9a/cN+Afe3bMAWMVW9wZL6uBxT5QBqPoWSP4IipYOpT826WI4Qmq0ATsATcPRHC"
    "KeyiLwFIk9AYixOqXXP1nTUCZLj5dcGfDKuSP7s+pJh6sJdpx/kCDw+eZ2iUOBQvO9PJz8xF98ueo7h6fdlv5LZN1I9K4ZKq"
    "CozNUrScf3TDWPTvsbDn96BlLFToWrOICchTGOKFyvMAuJ86Wat5cATaVGw2Wnwk6eFnGuUPqp3wN+LmxqUgCVS65oHJODHn"
    "1Wr0sJUqkMGkHnWppn9hnF/bSjQ3xVF4JQuU1Y2Nu2b2UC4Cpd2LfJ8aPRE30vB6Zu1HbxJ3i/e+9/tS15VZdCoVZyxK8XmS"
    "BIJtSQ4LlC3DsaTJdsgBk3rQvAKN1WHaNar4omg/4HPD83P5xf0tZUNx7cutkFkAdqrq57OCt+YkZ+gCwg+6iR+PkkjftBkb"
    "o7UAEBCz/Ly9MXyBMWpuM6KxV3RoWzzOiM7zMUUr4qQBLx2okBNqzbojPgdSK5uGlSnV7nR/CWGSnsCYCTzkaLbKGxdBz9AG"
    "fEYiW3Wl2tZUxSVtnS8uZukCfiNFzD/S5gPajEJiu63GErZ7e/khaLVLuBwSWgZeBgYBxXYQ1cZO30dQvfLQ1+woPF2O4ukK"
    "t7FimGL3EmZDAlF0biTBwxwkp2XS0kbU8FB2XlyD65K26S4fFtp7r9DyAveqOy+NH3C0ABnYryfafWv11w+X3knK+JocNEIl"
    "qkF+GA5q+IPxzmOx/RJ7QI9TnYkJuwd32f3OABlDs4rgRgpu4aBkY9MbmOiMzF/pbp3FSSWtlbcmQh2rtP2dWzh3i/o2ipje"
    "htwvd3bLJCk2Xn6V2fMjPllIgoKrtQgEtQBJ9bU/7VVcEeF+r9lY1SVlUJMCEY3SFsQ6GWevHb9kHskWexVSVy0glsGXR8Zs"
    "KL4nSG6uexDA4sZ6+3j3IUO7LoD99R390jpJuOv+VRSxYKd3qEqrdeCW7s5eHlo62PVNvkAfCdwLTwm7LbSk5hGcBGO3mS9e"
    "j0zVGakEWSqdjBLCMuMPn8HnLU+mPX75ZVG5UF331rGcwuGk+X3v2Kla/I/nxcrqoRyBUd2tRpULRfOSDalWeZ46r4u9dsqS"
    "D6m47FMlM60wWMUVbpRHQL2INX33dfLpYZHns2BlyWjyu6/BA1c5c99pTKuRVIUqmyF7fBo/FQ7Vw9AMnAV7lRQZRVrMN922"
    "PKEDgtee4niDm8fFcqCZ/qXru5l7zVB7vwd4H2Kff9m+kSXr5R4oi6Ypm4o9d0J64okxpa9FfnyHFzsuKRIzLLwsuTLZS0Me"
    "Q0tawUnjAneII1FsD43GRtoPbymY63kiG1SiN6K3f53jW8rr6Y3Zk5hIvp5ZlNBe0junwgPHvVpdrJhZb0BnwsYD/XzCyfdb"
    "y9ulEwxtkxp++jSajx1AEVSDGPLZlXlGp0eF/yFg/KpMnlWap/FPfr1clLt8ZrzMcxmWu4pld7ZE+EbVvXfuJ3f8hvfIEsUf"
    "bY30lyndVdUIkV5yQ5lCQp8Ai0Zy8HX8xRnUPaFY4emZn6E/ucTKAZD8Bayb1K+J+T03ijziw+ObVzFsETI7h3jK4b4uQKCh"
    "1PU6epo7El6UYEg64A3U5pi2C0LSKWPoLZtmNMOeIC9gXqXpGTVqBu2shZOyM3kd75MXRCCNScjofirHLZLYBuOxai+Tm2fx"
    "bd6n5EPENFNJ9t0lTqRJ8DV2YzQJWl6zjNlC9DFaz4RmQStJyCejKX0RJ37N313jIEHwfRPUq1oetI4EF+yjtfcCjmwRH95w"
    "Pn+DpQAxQBHzWaPsJjn6yuGcTSrWBwAvN/k1JI/5xw2sXNjX7h5AF1SeqPIDzTKVSeQrKpjISMKA8sJMUDfEL3RTEAshXvYn"
    "O06ZmxOWpAaUhYhYdTCje1nDeGXDLZjM49bONQPJLlFr3fAvPYrWQfdhwpnWZYPqhcalEE0xErI6eJv+t2S4yK1Fvbvufl5V"
    "9Mv3dtWp9bBl0F1n6mPhPD+sbitgf3+wySrHfNyiK6drAFSZ/JQYYDvAojqvsLTUWoZBWewzsA9BQAL2osHmT4bHEAYxNX+F"
    "9fRjSNpXUrAmUX8ma3APa1u79K1R0+cQGvyRtEnARmrp9g9E91AI/CxzynlTyJ9X7+3vd/uvqyXXEHdr79LGKyd092PMi4KX"
    "RmvAGMwJGu4X7Tw6ZItXekkK5hvewYFHUNW4Emct6KV0jymq9mQAVmnaDZQps9w+mKRGkv6lN7XoP4FhNEYSlHJfT3+7p/Ka"
    "O3nGDfIbmVefimsTBB5cOuYs64Eaeh0dCsJoF76JBH0bWErsMvOUSauqbThQ3bEBcaKQCclyLJIsowmxq4DUFFA/RuDioF3I"
    "PZyee+NtFosNbu3izYSh1hJvLMZfmccTUaPGQoUut26v+AzmmUqpHnR0zD6HywAaXZRyjLDLkZZCD8RhXY9j3pDM6PaaSGhX"
    "+KJlWxTzZrQ+x+toBNogqoUpi0pewmOeGdji22tL3tgw/bhHbxfikD7bABV/4EfAoPZahLAWKJSx5DNPagQigGFAxTWjgLrB"
    "kwwllUEd8WyE9tpFm5q7V2CwddLbnUngwxHavOi1I77016rwLq0O7oZpGhZxu4ZhrHtnGGZaOX2Ifn0UJUDX0Per0CqER78K"
    "F1FMEw1r9R4CD1PcLCsr+z26GSogTB4sBmN99nnGeklQG7jK83nAi/66HPR1wh/59MSCTx3xBCXcBHhWvhPjLSrTnOimkKET"
    "GVVChDbgF829anMRGMR0FEnrIC1jE/hhD0+NgQrN1glsOoHX5Y0pVysUwq0LeJOe0jfkZ66wEMyHhpJQgoEhv6o3+ygyFaCH"
    "082+gKrs/S+Fk6B61Bfq6kYGPN1IbGwATXb6cUrIDSFrZ4wDieEibEmFOPLZ3DSnQxrd3lpgF1g23vfaBkerImKc+PBPijo8"
    "MzkkhX9+lJDNaTNOxRc1PwBsULEJJVKEOFJ2jOnX/dDLD9nz7DiWCun6w5MQrlxTpImztVxhZAIzxhwIrCTPQnSBcrV3Qgcm"
    "6rgs5DvpsB5h/ATOfmvt8vv1wmzCcFrWBxRXnQxVYH9hBegfhCIvXdWpxFKfJv+8tO0btKuGs4C8ZIXM6HkSLXlZhE5FEoz6"
    "dTEYPZzQBBMXaomSNo9s4I+HmU8sCRsgSyQ+u5LA7F+1y4HKXJR7EB/wo5erJtARDGRiAaMYaEnMAuwY6cE0keCJoUSwomGd"
    "Jhdupo56Fm9FYofblhY9FA+xdQXa7PzypgqLZAprv8K7cUmBYRhdQEEexa2fe0p43Bc56/SoTWg1i5vXiXMMxomEiXlQ/Rhd"
    "bz1c7a1vKnFCpcQaqp3YV92wwa6+Pj5KC9UQ2du7d8aj34fDXA+k+eEJgyNWTmDakfHhD5n0k/C6yksiNDc+4/JCaQIkr3A5"
    "YXCLV05zuNWU+ycuQlEGp6V26rS5XRy1HiGb3uvqsFVLUglhnL1Tr11Ir0wVmi5TXLeRcI8jUXcnTNQX7kRt1m9hXNjOeOX9"
    "IhbIvyzjiPBUvPLShRtim0U1sBOhg7GfNX5VBI5veUPo0aDP5TbBSOs4pIOO8xkn1Pv5BkKZygHFJZ2e8Wdu4WQealnwqsFp"
    "dWvULtERnF1rCBPxwbpPHEaCFhhMrut14pnL3kUDv1q+x2n9zPP3GdZyVJxDYNurJOzs/Ej/6gPs1wsBZ9iTxbJwfcKwy2OB"
    "rZYeEaVxJQ0GrWWGBD0x8z2KMAvAoTBsd2nkVZowaASsxFyxd5F6K/PBD2C4p4ONCc/NUlO5vP+IvAm7c+2FLlc26gubIvQA"
    "w40tQ4XlvOj37KYnOMGcHxCGY+NR1JpNIwSZwC9GOLTsLMgiw83L3WEGPhjlNKZiPmEQMBWPlnADPMsVfQHj7YSy3igdmGBP"
    "SutxFWWRVYte1nW7u2juRmysbCm4metmB5yQZOk9vmJtjKyW8RTF2szq1HaLZmkdntyF++BACyLUntstHz/mnOSZUTurTEpr"
    "xPgw+4K7NWMRAOejkj2PPrd0BvezmKO2A+41ITvtR5KVqhJJI5lZDj7gLmfJNwRHw6alR8cS1MwZd5Hew4koS/Qqt+SiM1y9"
    "NzXFNNd+1Hgf955TTh03EsmJU9lKtQDGkpaIPqoTnUME42nCRGaw+l/pUxyiAEr83vsn7JPQNfyeVkeQRnOQnfB0cZ5OraSR"
    "dg+2g7wZXkrxMA0ALPbu3frTNN4AfMmX3ZJWad9iF6Lv1icrvxqYhStR9oP7Izpo2WFllrIqNkdPbDzObeK9QeNBUsLjtaxZ"
    "PFUBCuWKr5bZiPPkkI8A2ftrAFYShzTJ3FnO/oFl9NPwaSJY7nThHKozg3NVToJf0lPQcU80UcwMxAfLT2VtJnf4Z+Tvr8Ek"
    "zr1fn3PU4TStVBJpYzf1PMdwUj7G4wlq1GUaj0cAh+muDWxqa6LdjKXIMUShyq4+yLJaUfAoYrPLiNxVBjPrq1xXoQzmagzG"
    "gv+5nNfZH0DFdn03LvYiRBX9hApdKG9opEWrQgw7OZuL2wkDb6glHJPYXuVeMS0jN/RIC6tPi1wUfi7QsQE6YQ3n/JIZGD1H"
    "hP2L00XSefeBB8C4a6saot4H8aJTWhI5HICpQRsUm15rvokgoNkvGoYHoXCBdjm7hfyJ+VKguIP46+1ufUpcFshie7JKijje"
    "Xz4DBn+rIQ20x4z7m3Y0eBddfPZJxfzKFhKWMi6oGNkkpZjsJO2JIQcii80dxngsPfRMNKHWqLN1j5Jf/HpQYYsSh1UFAkra"
    "RVDTsAtbLsi2TPkpVTuqii6uZe7MriSQX89R+KPe4/bGeu7bOrtlnqQoCdGzuq9IInAOaaFbyeA6jF6kk7H0HFecgDfko7uN"
    "eYrC5a9E2PzF0wYTqui0a87q5pwobwCEydgPOSDuzHxDW4Mn0Mtp5VNR6Tv5MRFF6Gp8V2hdbqkuLcA1Lr7sEXDtYufIbydo"
    "4hcfZ3P5zhxvD7Crh/LxiXId8WPV/ZYnWMvugtrmtc43KyYh08kIPpKf0DGdKVsC2EiA2XWuM2fl71Intf08sKk5wYWmKKmr"
    "FcI6k4mhnq9vpoeV4xUnJrjXkPQ4ZWm1SHm0kXtoKORpTolmaV4UYb1y1XIAX5GE915cKA0s2GPFgW2CUuF4khG7xBxNlt7K"
    "2V4JfM/iuh0j/+Pbbbn8pNzJhqv/X3Dio5kn0YhzXxxZDV9s92ROwXTm+dqNnq+7efW3JTE9XD9co6XNYq/BQ0mXj3IBoLzU"
    "5LYEWIxJYgAWvpA6LG0jBAIH0n76VRwF4zCY2bAli4dvzjJ7Od1T8J3WvUo8IPs/88izsnZyeCn3fFdaX3DgkPdL/d47VIU5"
    "N1OraeeZJbeNqrEWjwUjceCN2cMs7sdS2iRtDV1St29o5BH86N8yoSJTLfW9CP4b5w0ZWnLoxpuBIl1wPqupYqYcUOrRoLA6"
    "vSwnCvMjhZmBzc5XX89H5Is9Me5zeQmSNjfC+KN9gJITH6WZ0xfMLuWx+VYqs0MuVnsewDd1B8NL5a7v3cozTv4Utc595Vtv"
    "wx6+vq3el6lXpOV9p0isFsUkohFiBfUQbKo1rQkHnZCLQWkbjF3sB0IhlMhevYmqhv5+wN7v0jqtGEvlzo4tnX63Nz9Aw0+A"
    "pWsLsQjZ3mCgbeczkKxn9xCsOV0z56wfFan2CsgJeKaSaSADFWRG86ApRWXrj1XR8nMlPir5maTwDzBSGWod6NmfrweCNzSB"
    "Ob6BKNrYw4703213TZZMpB1ujZamJdAy72flRxJDF2VMMNoxhKCQ+6EqvzO/1yMzlBWJ7Icyjvp3CsMrs/keZ9xtNT+Opgjv"
    "wSiCGRf9PhbkSmfXbXrCgLK7zPEdimYLKho/iRJrPftKi/hH0WjnkQqBjrOdPS8W40kX7lUS1H6yajCq5xe1z5swkjG1B7IU"
    "9tGnHouFFVq7XoCYqnKVn9nR6VTWZGY09DvIRJy/QkgFM7lwmPEPvaO2QEajxBe6zf1BYSV2a/GSFWdq+kDPMkP3ANyiq7dz"
    "2ODss2HucSqTfamKXKF7gslfZFGXWiHmNzgZ+3d6LV2ETe6xBN2/uKTBpposdX/l5XH64xmOvpLJ5S/AiA9Xw9Xv8c//mZY/"
    "aKucIct95vFgu6cV8kbnJJeWLZVmtkMyoCaOoEzWziMiaoP6xcXVEHAJfpJAz48S+AycsbuorKV8EI6e+hF1k+ju4sNIxS6h"
    "HZWD4Bngs0/zYEqMGv9+qizDv7baaF66Kpj76VzhNJ14Un6KfAY7p1PRcOaD5O4Nsj7fdNnQLfNnTT9WUEBdt8ALxqs1pSwl"
    "4lX75wtP6tpI4SyV0haSNXjrOTOdzDpFJNgaR+kwH8tk0MqIlrS/G+Q/Hbpet4880bIU57xzbYhO1BKwnKheaqiTUOH4Zz8O"
    "m8hb124gWw/6FQu5tMUqNbD1NOS2XBwa8EPLUxGCB8e/QtBiBpHIw5kTklXJY9QVAxGsz6mEke7VOfLb1kgPQxDJqYmc7mPl"
    "if48LWXRyYTFqfx5I7XD+6x6QAr56YqTviS/4jYkAJ88ipxuBDTVu1pdxKUHpt9reACRI3tqo8juDUTwl6gvBqdJWAXUWY7S"
    "hjLp+UiPq0yd2uoAZdu0oMQsy5vH/hSMoLBnEzN04Pvjr76CscbunlQpCfhnNOkijc4YAbVNHKKLWHS/N5MWcxKYDct1deCP"
    "vKqE8XdaeZZ3Kl0SuW+ozLq6HJiIQJKC8UsYeriRzsONRtY780CpiVk6xe1yQjY93JNgIL5a+6UVOlJYDIIA2KSg8RPrsfv9"
    "akuFxsoAc0ShE2M3vMP8kvFdYc9piJ36Njp2ZQ/4+gz5UQkup02W/iYDXJrZ/lGG4mGcyY425Tcgb/dOOL8sGTsWH/yMEU9D"
    "L7pOoRuy7HOyTHx8hUFCdTQwVhYXNGR5toBkNhOLyMBQ6j0/macXgS3ZH412SCq2fEgjM6sUnPxyGnKcefMVMD6I4ZY7w3Jr"
    "6Z60jhL5ik6759QVd0Gh4uw5XGHqKlawoHmpJlbZrrGulee6TCfXFjBJ8UHOa0cIN7fZ88TdQV6iAm5jy6vZu+a5FTZ7uYiP"
    "XF4hJnhML2OL+GwOkA08X8H3MOD5D7MqoRBeIlGl9bZoHJJzaeGfE0celmTybSteeLQK3AFxKhEHzRzNPA5hWzLQfSvYLexJ"
    "Kk4xHYiRdXf3J4gLfLXsg1qq+hnAyCRBzin3ADUSEcbN0y//ocYgSINZXxWppVO1r2/KR087u+vNB0CR0c8q5rkIWlOwJ6Gf"
    "8p1NYzMoTj9RLzyXkx8+wAcq0NoOUuLRlYLWiDLHZvDSnIqmupRtNV/HnwujwC79GPx0iaZqRaa5n5B9IMBk453sjLzCNpRi"
    "ybCN7qyTl9OjK/cJaCl8ph7KTN2Fwh6aN/w7jq1V9sxJkDSIZ9cWQgJHeGUIoLO+npfzt8qOA+tp4nPKvoHDBTpDdq+0W3gs"
    "Qn1eiw92drOrh3Y/nl5UbCIx8kOS6KfzqKs1epXfk5uz0Zb27BHpsJ/yrWW9KvRERjWh1WFpEqKg1Qipk5nQgqweSv6rNx6i"
    "s0WZ3gCADdV/fY0bLppWvbV3MarqMUjZ2bKOtDuG5rfaUvZLPQGHZ7Xlh43FhuNBCbysTW7Ff2heD/64J9Y4VK4HjzYEWRyc"
    "lo1v5Xwz7OYh8fWNWf4uNBpQpCbhURmkisxaQFb9neiAdzQHg0FZuefpmiVziS+7rEm9ayJ6NzDSbDC/g8FamzaffVWYbzdd"
    "sTwZsMlRVww/oduvRzL7uVKQwEkgZf+Vkiwj6P6KtGWacveRFU/NZ7Fm/xglorX9dKlbw6pr+ZiHal+b8Vr3cd+JuoM1g08D"
    "KmYjQhgTrKXucEU9q2gME9TbLhvjVAWiXlyggRvNYkuGPziFdqqz0PA6AmvCoK1DgejNYcyrf25ImE/0/hYZKvX5GpRKaZsE"
    "m+4I5+Qxl1sM6q8fLo9ZZw27k2L5+DNPbGC+uehJX75fGUMQQofLVk2kD0Im3mCoLkn8r0jdIUWaxakZ3jx/1StQLa+hfWyD"
    "iti+Vv/LDeFYhmQ1gG4zqPZpZ5K7Ri9nkzWip6qYP0Qv9vQ0B8jRm8kT8TXit3tHrwRw1cgfk/xCfZ6qc6aa91K4GlAeStF4"
    "o7UFY05azfAUL9QkbUCS+4xTU/9ceJsB2ZXxN6Xf9xkoPdUkMkM8kkitxivPzlQ65woMgujs29HhpgjsylJCwuBt8dtluxba"
    "Ki5NQvuJgxjRmXMxEw5FzuiRP6fbCZia92aD6UKNmiWYb59wrQ873bMHkuZHgcjH3lcgF0r/2KXBQfWAf99zKPjeZMvviZvm"
    "Ad663AfNcM45zAwLbSNEw/JKV1M2kuWKVjbqXEFsy16iZ7k56+/MUCIygZlwk4vAho9xXXkS0aEiTZGIcFLUIqcq4g68UbTR"
    "E6fQ8ovFkrYhmwaNHkq2DIgjB8ERMKgFzCZhBJE92PGLcovv3w1IRZudy5jNSQ1a/pR6UBa7tkqy24oa72lzyt57NDjFNe+3"
    "zui5nmHTZJHWz2oNzh7mzrTAzoQZojyPUBhVWyCin+GqY9x2AFkEmUh6LiM1u/M4p0tD+W5s51pQlXi400QsSRAFZuPm0s+M"
    "SgkCxZKmLCCU139rPQZqQ8Ic/hz8aEdAl+CMxwS+o1QYdAWzCJJ0XNQ/KNhpXySrAU8yK9U8NX/Arc3LYNAzppMkDY1//bk6"
    "l2FthZnhmKpEBWEtGjckENNRE+UgkSXxnKe576wOlUbXXOZQZWDqdzmeou0sfWHz1zASgl15QN3Rg9awiQdNsdQdqWKx0rwO"
    "v3yZM/2bhlPlLFQuM6uMcsNZrnmhv+zJ20+k93F6jPhRC27OC3RW4yCmG6w/ffGzECgtLIJWyht8wJouzHsWLHMY1NaICl9G"
    "jwi8OARrwvuh8i3XW+BjohUTq+w67wxrkrmz6ctym7PyY3+RwW8iYMeu1sxLDmLS5wafqW6iWrjYNW1JojBTXe51kzbryW/I"
    "M1HDDdKbfpwoFXFjzwJlzTeOkRFBIGC+oX51qARs/3zTB/gC1R/CzeU0zw4etetaC3QbdLWn8MrZt5w99CqDQtr6GLJ+pT5l"
    "1h/R1hZOh7CZ3jF/XWm6N48awY5TE1UYOzyVNsX1GxrAoq+1EqUjzzT8sl6KkA6QCKTJ3PQO5jtGyECmHhFxdWL9+a0SJ5yM"
    "FoeZ9J7bFDqp3GtBgwMruY7MSnZpU3hP4kGYbyNrsqrwZoJa+50qYsf3Kc209wkxpxalS+QBysdOKDlUQJWZao0pGXHKR2oG"
    "22PDcEa0tgGF2M6fY9TXj6SZAt/C7tJkZET81hCTIC/Z6kRkhX92cO4d2vcbOyP2NEqhGAoKJX3a6Xo5s1nS00dyT1FibxYG"
    "veGmU5cx3FUbG+4e8zucX7+i5opIkdNF+McfTInziMPixKMNYtIWbM3dCdEGUrwWVtTZKJrMOJ/a3mhm3MQXb1N/hU0Xxpf0"
    "2RM/HjtUtlOREHOapqMv7oD3Vp5E35ABcozGeq9Y0itKOMCHAMdZU30NRvwP3pdwhr+3vP9eLIC72Dn5d9Wsbv6i3xgRjqqv"
    "x7wXQ9ghfil/rNUQyodfKmEelKWqgoCJ0uVjCHfk0kHvYde74gC/xdymkDbFxk9MkyAFlgWPxQtNq3NHDUlW0Laz8bKN6hqt"
    "ggzatr0M95NumvqZGtla1ZnbYt+8WstSW622OMs/d+X0fsXv5CYLdi+66fI6K1+fjq0i8+nUcfrD2CPnrtfubqsYdDCpB8Hj"
    "Tal/9Bp5X9WLGzLix96S82hm5IEjA4DDcEDBjqk3NIiZYKEbkbK/c+k7og/W60egW8f6v20RG/Tcm2k9t1TjBHUO1KkvXzAt"
    "wid9tTYDjXQqLyKU3zeOX3H+bFpKrNA3AA7xfReF5HOr6LdBvj6O7E1b7Yr6xJwZlrg6UsJOSPPbHaILlE8UmH6wqVM9u/2D"
    "v8W2cbSO55GC0Jp9zvXZ7Qf0lr2ViwHWlWrx0DGYrMgJqazvoJMeZmx2xqNchF50ELdiXsU9b1DSOhaDfcPYzze8ECG5ZSe1"
    "c/q99LLQzTI7rRAY/ZovmVeqnka43f6IJRaf8bHTvfb1az1DhbE1y5CLHSMKBpUDNF+OBfhqNAWGpTP+ZOOXZFrFM4ssKJsr"
    "QC+dUlx6/NaTwqPdeY0cdk6bAtLQbH6ucnwCjv4sMzWr3WhFIn7i8MPozDl/FFBve9iNx/AbmFPYwKA+IfFkzabivBARBGQ5"
    "vRFDhR869YicQAvb6Rt6556ptFQUqxfg7mhjx1nIDDUq/loJJMsHEdI74GN8RPIJeQ+sxCXev2bYK6y7a6a8/H2KdmFRveex"
    "7CMmiUH02c1EzB/GIzLz0gGDbjETZAw/eriJ+SY7EARtWyByKdSV8YZKvaoH9jXDUoi6c93o48tgtIQvL44eXT38jo/BdKF6"
    "L6efWFsdpsIeVXjg26FWnvEJ5nGIFpJ4c7vzKGV3Fkf5GjqS6XOCvr1MQ8MgmVuu7JiQ4+BQFJphsvd6e+Ea9vUdvsXjYfdc"
    "ZZg7cyEJmdT6znsXfqt9l6dqEfrp+EcKbhaxVHha2lneKn/JVQARBdAL56aO/lizK+aeeUt1RgueM+HNKXFq0tMkH4QpgeaX"
    "G+tzWnn0horjiTlGTmc+1UTBR1U95SIGABuHb18huXnByWIoqnjWET4GpvzAbUkqcZieoYuUxOXPDYrTvvTJuoEI3YD4wncm"
    "QAX3kUSxQowBb0psMaNAwMc4nYqbqEPBMQPicjvIBxYDNR8mZU7VmIaafPyfwQOvUD/0WhDT3fsGg7WQUPIFapPHbQ6mmW9L"
    "mtp6gd0OFeQMnPffW/bc8PUzW7NNv3OIvE0MtkOlSeKAtPKF9YFub/e8aHEthJwLgCTMJyE6BGzBEFzsZkxZw2VivQRK79En"
    "6I4RWBfefjR+P6zodlARDnmO8kN9VuiVft9iLm+v7vFar1Qxj3QSc1DMq8EBbV/tOV7Jg3bAVq28FTfkBLsqECf4vM6kKs1C"
    "jQQcwXFHMiTmPB5V7coi0ImGYOC3nREKb7MhD0s4gMZB8oIoaaxLBe9x53soLPRtJOuF0K8bbrP4A01JmmTPyE5KN2+LNfHC"
    "KTU7oPhPjWwJGf03Pzs8o96FeFfc1/cTL3922lNCjF0tsaZv2SKFRSIe/7sMAnUHGSA4oQJWwbhdDrfx2XNwRsJRE5dxoZF9"
    "72jGRkDH3vCIFy8J8LRmemnMoT2BXCfoyUdsE9EOpAmxUERT0Z3JxtzIe6sKse9WY/UJ2GH4FXGO8rSF7LQAdrUZhItJCgxF"
    "xJAbF9L3RTjyAhDOloVrwPhXXk0WzeG5JeeakHo7v83pr2ZE5B2YaQcpfefKuwV++yERZrWceK0jm5FCuSQrS+i9cUHMw5jV"
    "Nv4TGQwIyuUFzlnnxWGcTjNKA/LUseST9+6B3QPsMep/A6+/23Dpcci3UMeC8FYzvRaCGqyRWIKoRXpWhWj2u+Fcx2SKDW8v"
    "RL9l+6vqvHsgPn/IM7qYy52PZ5d4zaFdFgusyPvXJ6I4TtBqWF9jcTd7KnsB3BrU77M7P/RsRq+woksphpOvFzULInzSzz/v"
    "2DAI0wuIbLNsEMRHi0myHcfqsQ6U29xYf9WlviAl33YaocSLyR+32hLHZeOc+nrdT14UZuhfkMDpmLoRg4nrOsknKNkNqpxb"
    "0GJy8JzAYhaLD8593OjmeVg6ymqV1tiVPbs32NpP4HeiN7NTpI6D8Sbp8mqWyur3HdfG+ZI0W2fsPJE9xXyjxjvhidLPr9uw"
    "5mOPhdR+tyTcsP7foziZiMZ5kVuCQDQRbrUtK7GvVeIKET72DbX32X+s2EHVF2zfKrQG4DkckTT0Dj3+JJ60eH98p10sw/2c"
    "SZWcyGAKHdsRXzUFX+ur7Ch+syFOTqr1mfpoOkfYwNOX53jvoBMBbcY/2MTjEwx3G+zP+bdvmttVBPJTAoRMA2ndee0QekRm"
    "tCGsw8r0Q4xAvLQycN3PMeg3uvmLGzUag8+K1zuWe7Ok4OSmcQm5/1Le8bwU4CqOjVp7t7cupEJXr2H5VGqrD895WwSKcJLQ"
    "J4gSLc69Gp60O4x445AStlFl8CXrGV8AsVVkDAylXf0o0w2odYJbd3jKvUfqGun+JqWDQmva5L9+DTCxU2iRYAvn0zcuQgvB"
    "KXqEWGKWgmOaZOG+kQ1WU0b48dPe3p1zAt7dqJU6jZInfDzTrTuQPZtmX0p9tsV3gcR2wpV9JqDsecfNodzQziap7upgbwVJ"
    "w3Muib634OxfElJJkQYflXF79DXQTQiujBWH/8AJuYuXOzwRM75FoUdDj27fASG4E8fN/6bNCprL210kkCWBpLnB+aNc9gPE"
    "kcGgsIlmjKpPVJ7ymnu5pyP+19tzLTduQ/uer+B17s5Ya9EmqWZJWU3ci9y7nfEDREISLDazqFjjf78DgAUkQYr2JjcPG4sA"
    "zkE9/QB7j5OX5sts46gOnx8O5kfgUb6/3Wlre3W4uwNk5eZmsrjcv3qYNa6OG/7Z2/kcDd/OPMlT6urJxenepGXMdN0w69e+"
    "vXdygvSjq7PRqT7eudk7q8Hbvj06AEMTNieXY782OTAPhneP/bva7XX96OjI1e4XqqoePD/Y7vhmARp3b893zlw6ez45BP3b"
    "3ZubA+PidO8S3TQeJpfXe/5Ne0dyGsOP02PPGb1cqLPrLXlnQ35+GM8m6Pigj1qLx4+5Z6o1sPMxvXoZ9FXkoaOHafvdOny+"
    "1M6MiaSdbpy9f1zWjocTbefofSS3rocH4zdP2m7ftx20r79p+/W9y4dxq6WPZ2eT6fkenFzdn3t3C3k+7MOm/rD9Nur3++1B"
    "ffhuPcz3W2fO2fPN1XH9HD0+XZ8cjD7e3xsHJ7tz19BPfc/ZGvaBuVGXTnYb93L/5U6ZQ3lkKdcHh5Prvjrf3bs8OBk0lK3j"
    "pnKhDVteY3Bz7e7p5+/a4vDNvq2Pn9/PZm8716c70+Nzx20NPuY7H/uzs7ZlnllnO7uNjQ24Pdi5emu/nDxd6G3Hnd3U1MZ8"
    "tnMwet7ZOTkeyv36WWv2MRpNjg4Oh/W7vSNl773ZUm7bJ3P3zPSc/oZjaNNT/W53S32SZtOH0cWOf7Wl9Bfn1svHxv7lzvXT"
    "3oX0eH760dp5OhjLx/5L+8N//Hi4mZ7d7jf2dm39sv1x42/7zsX9MdzZ2rrx6nfys//sA+u99nhTOzBad8rx/dB2rnYbj0/y"
    "ifxo3rvPypF7cueZt2DndHTQfLiyL3wXwJcP09puPZ9PtrbeTvZ0eDmQ3alxbm00JWU+PdJb4OnefmzOrwa7u+gQWfdP2oN2"
    "8Xi87T081p81e+xZs119a3Jf35Dfz06e7Kvn65fJ9sJ5vxnMJifaSGt+2KPa+dnjveuAk7Nd+LLfOtTkD9trPnvv43HDurGU"
    "9zvrTVb7ysndlfGoP7Wbi5tnYMq1wf7G4v56v783tO6m9btp/ci5nllmv34sP5yeby/s8ejZGs+1B3368n53/XRxMfpwro7u"
    "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"
    "a72/tjQ07f21RSLtcUvP0Ht//B/XEwG7"
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
