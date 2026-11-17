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

PANEL_HTML_SHA256 = "15c4d1bc3de31c612abc0af23cae2e1a24265b66cfc4d02e9f592e22370982c8"
_PANEL_HTML_Z = (
    "eNq0vVuX2zayBvq+f4WV48VNTKOVlpPJzqaM5nJ8mXiS2EnacS4abYlNQS3EFKgBob5E5H8/q6oAEFS3MzNnrfNgtwiCuKNQ"
    "qPqq6uloVZf2bicfbey2Ov+vp/DnUVXoK/GJ1J9AgixW5//1dCtt8ajcFKaRVnyyt+vTLz/xybrYSvHJtZI3u9rYTx6VtbZS"
    "W/HJjVrZjVjJa1XKU3zgj5RWVhXVaVMWlRQTKMQqW8nzt+aq0OoPaZ5+Sgn/9ejR06Y0amcfQQvFJ9t6ta/kJ49KUzdNbdSV"
    "0ufrvS6tqvUjmUp2qKR9ZMXby99lacelkYWVqd5XFZuua5PCW/2oXj+S42ZXKZsu+ZIxO9NzMZkaafdGP5LiXD5S+pHtrgvz"
    "yIpDx7WYzbkRKRPnh44r/DGa8EJA3jGMyfN6JZ/Z9IwJISaTSZIMkieUfJYk6SD9CTufPHnStkeJT//3fxivqfDGFsY2Pyu7"
    "SZe1/mm3KqzMlow3vo9F06grzUuRSm6hgdhHIcdKr+Tt23Vq2VSfn2KToM+lTDWfsI5XvoSdqW0N4zveFM3bG/29qXfS2Du+"
    "92VW47KoKnzgK/HMmOJurBr8y9fQzttUQheXsxqLfPRdsZsv+e6hVxfSzpd8+9CrF4WV8yXfwDtoD8yTEEs/wUt+dfSmsUbp"
    "qyVfHKffbS/rasmvj9KpnmWSjEaS38HL9DqVrG03qWQsSTapHNuN1O5nWdhyw/jl/YGy9QVWzW+hkEs3PIxf+G6NGxzpL/np"
    "hPHnD3WWypwv+Q28vUolSxI5EmL5pnizTBI5O5vD0+kySZbLkx1su9fappJPcJFJLqWQ6ZJ/kHfcyDX8W6xrg38hrdbvdb2S"
    "X8l1beR39V5bn4QPcjXMQSvLp9HTcRa9jYtxj3K1ZNxK6MSfbD7aW6kW57DZ2jbFPSdTzRjr+Dvx6ek/bj694i+FlSmueyN3"
    "VVHK9B2nbYDDOWFjW/+020nzvGhkyhjjWopP//FVOnt2+tucfXrFX9wrQUu+PH08WcK339Y34VtuZMgL2w+377D8k75ixlXI"
    "ni9r/fhggN50y2y5ZLyQfquMXO9VQ9ulxjfj8Rj3ZqBB4myqn8pxJfWV3Uz1yQmTMz1PMV/H32JpXHMjRhP4zhW6kmulpd+f"
    "mOdQ1nqtrvamuKxkNjrjUu+30j1N+I1RFn8bfl1Ue5npjnW8iacLl9arqi5sKv1EPVLNm+JNalkuM9vxMs4PSzV/s99eSpNK"
    "lr0p3jz8USX5XiKprGTbCrcNr6r6sqjebVTzdLlf5v1j5jI0slrjK/jhE2+UXtU3mEw/s0FxUVHZoZuGE2GFR4JapyvY3679"
    "kOHj04CZjIDZ4EpcpYblG5kalq3g/6lap4r5z/GYUMzO5FyomZx3bhxsp9bpFRIWIC/MHyx4oKyl+HSa5qPZ/6Xzv/yDwZrd"
    "SfFpls7+b37CPuVbKT79JM2z2f998o9/zNt//GP2f3P2l0/a/8a0/47S/pt+tP/49B9/mf3f/C/5P/7yj08/veq7v4lOxEMX"
    "zrewM7aS3ztjPv3LkuXLZQYkDE/ItWTjdW1eFuUGFj+MpisUDhnKs5Nsqt0onk+SBHb37Gw+tkZtUzYXejbxD6zrGLddaONV"
    "1MblcuoGjlkhp7Jq5CM/ef96yqCkmZ6zqYEGnAhzsny0ZJ0v5XpQCsycxD1HmTVm9kNkXWP7Zi78ShpJP51A2KaVtIeyKpom"
    "s7yxdxVsMCFDOUkyukoto3MfskErLZAtTMIvxArJIKflcY1UXVm5bcp6J3lRVfXNel9VTWmk1HylGtjPO1XavZFKux8+3cht"
    "bWF67y6L8gNf12ar6+uiUkjdVbMtdlzXxEbx6A180ShdKS25kcWq1tXdkvVL6Q77T90ajWTbSjjLlv0AXUoiWThIfnpGQlj3"
    "043aaDKl2RqdhV2oxBnMmepnVZ2cMCNuYELVnFv4T4fJMX2lt8NKG/WHxCrhx3GFxLWsTb1NLeNKaHnz6Cel7Zf4IjW+naFZ"
    "FvlEWl1SnE7Ci0KcTYun/oNpcXLCYGGoWTFPkhuZWm5mBbSYHaQoppdGFh+AIsinZ32jgF6IiaMYo7O+TxehT9Ru2G5cCeTj"
    "1Do1bavaNjViR+k76A0kBjIDQ5mHkcmgrnXqDpAP8q4B7iRMT5xu2b258l02tF8ONF3HjGJqGC+EvZ+K5DJJRkXbjlSSwB+c"
    "VAOTanCIfFWedhJfRbyS+22jffhc+pORHXTbihlM43ewqOnvHGZ7pngxFxprh0YhJS7wlw3DpMZX0lI9NkkKfLLIWU3VuIFX"
    "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"
    "XsMJEjHSWenEoSYf+u+8eETT05j+NqNJ+iD0JxiF9iT2JlQH+zYxf5vR+FZ4EHT3XMiYs70zP3v71qRJ35tSXdSQbfdco8sS"
    "ScLEXZjyhM5jeyEueToh5w91d+BfueSZ8vvNhJyn5k4SRGliwkgqzZK/T8j5E11A4HLso2v0DMP+VnddAnrloe6OLZc8nJCP"
    "OsWlgkEsowcGJcfLBpPEjBV/KCHmA66ImjIZ4SyXCELvqk6Rc+HHlQmJI9CB3PMv1AMOYdTvQrC7XUYyicLv6S167PwmlQmF"
    "SJrzlnnvXKEtU9YvZZ1pemkaJ4RZKrKMJM+tb4GckLzLbYGZwCE4NFS8y5Z7T7zdLugeCe04DlUGDxEaqeF7t223LMJ1lU2d"
    "f9WLdyqEzwC5gUmky2eOJxNhUqEVxz4yS846yBgbj8m5i9NtWYCYsCHMRz1lV8eDFBHI+DQLUzZsu6KIXRTSHybkPDxv9xYL"
    "R+Kq6MuPuqu5BmlZBnnn6VALYmayarB+oiGMyTpDsVbLigIu3w9VcGQ4xBiC7yfEJZp7D+qUAFgvOSbfTBRW9Ih1SN5PlNl1"
    "v02+VSHyQcV1fHFU/VYaVWIiKAyrHjzBhP3TxDk/d2PaT10yv7H3yK29x60e61hYO9YxiW/sTtYj5+7UA2t+PrDd53ua1Rmf"
    "XJ08tzra3tXJ+MTNej3yI2JUwDqWdux1tA67xbtltSzzaC9oHbX2lbcdrR20jrT9B/KVtqcdXykFNUvDgkeaUg7+BkcttVjH"
    "0qzDn90SsZM9c0870o7NI+0A/m0dlCGgFZ3OB6uDrfi50oq2dqidaNZR0DrAwj3yugQzOdGsY+2whf+PEDsI0ffjPiwr5v0b"
    "cMq7pH/L/uLN7HV0HF4dBvsao4SOlR4SO0L7yn7QvktubPfYJbf4b3yD1mR8C3/KuPc167DfskwLWN4yT1qdVufDfr8Nr+BJ"
    "67Q6Y6vdRwito5knWofTkY43b8vEOtBOWofaYeXDiXaisfc9Eo5LvLYOtL3x4dVhlTVtzdqHHt6rfjrWrL2rQykaFjTm2wP1"
    "udX5kMuThU1lBMRlAg61TqdUcr9U8tjrmPsa/E/KZnvfPGofBnvmwfGxhv8+KMJ02trxldUp0vh7tSn7WufqwLM0K5f88UEd"
    "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"
    "o1deglik/47jRNZy55FSxrB5A9cZLU3NXHe8yMruPmIqKDZv6VqjRjZ183EjK7rDyKkru2LscEIGDeyvjQBRg2AK9QtkK2ON"
    "1ii6nMEC/I5DplLb6qHzp2zuukOnUt0aQ+hP0eAVWw8rCb3LRkJN4TsT+6nyuclmw/8kPmwsuBvtRPwZOVF1968mdWP3faXo"
    "Hcn8ZLFd28X/P6P9m4vr+v7/PxkHyq72lTRu6D0vlrsLdZ8qnGt62P/Ubd5YINd1v/9pWl12ea+kbkMvdrHcXaj7VDlc09P9"
    "p27zxnK4rhv8T9Pqsut5JXUbepOL5e5C3afK4Zoe5z91mzeWw3Xd0X+aVlddwCvp29inWy55Nxo/VR7X9vv+D2j7xnK5vlP4"
    "T9X6OhfsSirv4FGtlr0rpZ8qpRt4Xf/H8GBjad3EJfun48I08PzNTEwscSenFxa8E4Gf7OHEqj/Rv/knaPfmrk6s+1MdnV+y"
    "5XCr7t397rUgVYqgkjWdrgB6R3Er1LK+sH2dNqwrOoVqNhCcL9OK9aPuP7OwrOvkbCDyUyToLu7uP7BpGwnWnTzYX6FxS+PQ"
    "P7doreWUrCXwk8RqY3f0H9SozQRqcx/zl27WkjjtzyxJa/gUq6R9igxt6DX+6m3ZSHQ29QZ/0dYsCW7+zEKzhgOwStqnCM2G"
    "Lt6v3paNhGZT1+0Xbc2SiODPLDRreOuqpH2K0Gzoj/3qbdlIaDb1s37R1iwNo/3MYrOWI62OvE8RnY1dp39IizYSoM0dol+4"
    "TSuiTT+zGK3p3aon8VNE6Q7+zT+sXRsJ1F18ll+8ZRVf5Bf0Ba3pmfo8fkilyi/kFvo8zdnMR7Spa/FzNygoXlvwZaQFKlmD"
    "jk0E4zMTt07UeTOJzZl8oB9aA9qPYn5xAuTIiMGxu0ElSyXqMzOiPijdzS9MeLTehQmP/koXJjwqZqt/WU0j//ErppH/qKSR"
    "f1CbRv7RijTyD/I08o++ehr5R398Gvlfq9e7sDEgLnV5PCpdKVJNF03zK5DYne/RNOH8P3dlxqBpqx9NppChOnXJy5Fe+998"
    "u5p2a9tm2bVehsHtYpHChUxP+1G4u7vF04DhL0wIRLYrici2bZ0VfxUlwI40wksn1FcsW6Rr7O5yhAoauPHJD0dAA/+pfsSM"
    "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"
    "rt/+xbdbn3y0+dH122/8dfjK1a1ffr1+8+3hC38cvvcBzeDOzZc2b/x24/eX1q9dWb/59tbVv268+TLOWgXvCSmb77y5fu2F"
    "zbefHV79auuzy3duvrR+7eLGlzeGL/xx/caN9a9/N3z+ueEnX+GXafB++Nx5+vzOzZeGz1/c+vqTjf/8BX1FHQ5flR1Czxeu"
    "blz6XwDQO38dnv9047Vv1q9dXL/24vq16xtX/nTn5vmNS6vD996GMCVbv/bxnZsXcLQ0Xr9+48bwhcsbV/59eOnDrW9f3br8"
    "0t9Wn9m6+uXmlQvDb54bfvLSxvlXN86/Pnzl9Y0vzg9ffXn4/MWNd/668dpXW9++tX79na1Pvtn8+hPsE2Pzt//jzc3rH2iR"
    "+Y03Xtz6+n/9bfUXBKmKyw+/fH/960vDl/+w+erzaVSesJoNyQ8/fXvj0seEkWL8/dbqpa1ffTx84UNqcGv13Y2XLmxe+sPw"
    "k9/r5Jlbib+tPnNr9RIR7K3Vd2+//crw1ZcJxeMRShOloPvtX13ceP1TBVgu2r7xxh+Hv704vP5aIdK+8cYfb1/+avOdTwph"
    "9q2rN4avvL5+7UpZoJ3wqEXZadqwwlfev/27byG+fufmBT3Cvn7jOfICUYtceJ1mv/HGF0QhksAURW387tNskH3rm28AHVfe"
    "S94DMfzi0saVP1FPd26+s/nOm8Mv/0K9rt98e+OVV4cXLlKbv60+Axz50fXh+bfkyL/76/q1jzffeXMkkSYx9fVv39146Rc5"
    "QbFx6d3hq7/OB9ITeri1+i5w7afvbl393caF1Y1LF7Y++M3w+p9p6XOoubX67tYnHw2//u3wwkUSLuvXXiZZM7z61e2vXwVR"
    "ggMTKPRQdkXAITS3Vt8dvvLx5gc3hi98qGTWM8NXX16/9t7mBxe3fvk1ktOLSW9b3/xy+OmvcbIqOk6MVoiNE4PSSz06Pvzy"
    "L+vfvju88maycipCPrz0oa52169dAYl3czVZQhUkH753cXj+i60vrw6/eTaJkt/+1cWtq7/TI+QEAT3XIuS6ZN946YIuvrc+"
    "ubz5yRsk9YYvPTd89T8YpOr+bfUXG69d3fjNr4e/eWnr39/a+uarOzdf2vjtnzdevDK8/Nr6jefWr11fv/np5idv/G31GSnb"
    "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"
    "+xjhc650dPN03p/tBRw4YBwzUifKYUwXGOznLOTvYXE6nB6CDRC7wOLsyEB58sCenl66iXqvNqR111iUbms/URrW2kK/1urd"
    "tAqlAcNHruXXZ8VPn+rn2g1RsmNbdY2b3Nj+4yo3bo6Ctl4KoO/EjHXKbnNqj6x2N6nsk3yjNIYu1dbvBciVsKWShZbnsUeu"
    "ZahvBJc3NIBuebx8H1jKRu4lAmUmd6LxSs32VjR5+dQID27n4vIs6qtPX2pJZS0l+tqN1PJsedfLs0C9oRT/qp+7oRFphp88"
    "Rt1DoFZqWLt2TauEOcrFhzto1Kzx09+SVtmKqz3Dxq81NrV5RqMpen44pkZU1IBjpUXFQ3kYPwTq51q08tL+3fj0ySitP7Ak"
    "iWw1TYzWpcV2o1yqo9ZmMckvc/4SBa7xdEGWV5Xoa2lrcdF3fPrUUpqEybvSqt6rPlWd323itR9Ly9rS3f3Xcdul/HnvprH1"
    "+FJahby8tkuJhNIuhdT+O0lHA8r3cq/KOARq5ab8Rxn0ElXK9tJrOb5M2CrCb7cB/q2eTxd/7kbW+kc50bXAC5XKf69rGVFQ"
    "/lYToJTpVKaS3UvnVFK77m+X15/m3PglVokrU26jcWurg+rn1+63Wm9/vzhOZWXqaVVZHNyrSR4qT9K7+d/M2nLCN5Yt91ld"
    "dQ5ag1p+yclvP9xcFJ9rGd4bSIG7oJ/rJxq5ZCMq/O19ns33sv4/fruKKC+6CeVOXxO0W3+llXD/U63Em7pe3qu+fRu/HOVS"
    "2XZd7i9vx7jCBEtL+49GtFmm7w67vvqOrPxcLjXUBj1Q6RmVXMouKc+FNt1VqKv8QJIC9Y0VT1x6aMbqUyMbtmxdN/zfHl+S"
    "KC21B9otdmkAapySz1W7Xrvum92j2jmmhqJXEEO/omtacqooWHRz/LXv3eqVboOExZeZBhdQhMe4AVTR3u9A7kXcWO4ws3JU"
    "m1WWtB+q7njtTqLWN0+JnMv4u7XqqhQYZ+ZL91tmGzkenB5fwA/ghx7yoYdcfFgbVGwicIOykyopcAWk/+23Sk8Wpa7x2+NL"
    "qQ8PSmGr/vjCoQ/oh95Dz4JgBX6AP5R/Zb7Qh+Yte//hc/Nmff7Zl/Iv+/yzzMovZw/8AD2gCvQw/FD+lR/Bz81bWdKbDx+a"
    "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"
    "mf9krvzV0RDR4b0K6/XRkHircamus6xu0nh57eZXPqPUQ1wyaM6LX7vrW9IyxefY1kvdC/VG94KF5XEgCsL48Vtwuejh8Vtp"
    "y3/HmXxInozfcb81z7rJwzVapcj9Fj/6tWNw1PAudg3BWbkCB2GpaL5Lc+Uy7Fr2+BKlfnlL0Muj/YPE0r6KK58vp+DG57Dk"
    "SrR96eD5VPIx2kMNU/VQu669dtMq2O0+PDysvVYDfdFuPnfL3e+xivBQPv481nP17fX1vVb8eoy7PFBudZV3hXXxdFFTX2NV"
    "M+OpcWCu37rvqntJVDJAdR2e00f3td2d+E1P7QMtjbtKXPhaiSHxzSiV4Hl54XpeerleCzONS23wgtZbxoOtt5/r1K12yRaS"
    "UenmhUWtoP3O4P4mVXTro5tXiXeJ6uym353mLlOqvA3sQm/K6ffblcj9djcNf6uVxW9G9vv3ejVVDMffiu6bSVoN7+utuHVV"
    "gVVFUaPq36z6N1HbtcYzr+TeJbklS+fC98TIb4jRvbzxDW7MHeG/X6bdlx+W600IQKp36+LeZToyVeX5X61/xX9v/UvvtL98"
    "6H7tvHsC7IcSb7qVVIICv5ERfGncqEvfuHb3+tK7f4G/tr9WsoToCfi//rv1Bfw8Uj7vv36DX79fn/uv7b+9LSG6yCvKM2RD"
    "d5+enuDnLw8PD2XY16urdse4uGa/tKsjZzP3v4DdXhep4yVP/wyVKDZoP2ldThxJN+n02l2o374IJT6UfXpzTJ+rrS9GN+n6"
    "X6vKfIm6dlf5+tSEXcq/dCjQQ9Er0Mg/nh5AeNQbPicA1HsY9R5braTzAKJoG4Cqn7//vfeAvF6NVx56UK//96jzMIDQ3t/t"
    "zgM46PX+rtzq4tyNXTWOFc7MFTfm95LFr7++E9g0K6tVmjBkJfhZudyIWszaeunpN0CQ5LlVHSjcVVK4xtPLf4Mg+NK9DzK6"
    "pHofp91+rHqsPjDc9+Gt4pjaalcHkMYrpUT/2n/5FTm6EYUS5+LrdbY+V5X2wjQx9KrollE1pPSbN6KkqDyrWi+fP5doKJ9V"
    "86VdmZ88Jc+OWrr7l5W6n7jl2Prtfzwgzy+6Ejkvjy9u5TN/mQj1Tc0PXtmwElBc+fTph6DWSyuMjL0RxZ+1Ul/+OdYswzMe"
    "P5Q5tl/adVQjfl/EfW9dwm79dVBb1fx6+vZatz4qj80XnIISfeAhMvRUMy5HYuMC2v1ofEm+do12136KWhUOTvv5OWoRxW2E"
    "a+gZ++kOcObZbvntx1VSXg9dysavFeGairSvA3upw3+91DgJrRrPp1UxMC/lQffz375Fry/dkmd5bMzP/Nd7rAf+mmndtI8f"
    "k0+fSmRxvYKnuIEsvlyCS/gaoigZt+6Xavktyiyi5/rr432SblJKEKLv3z/65TGaUy8OdDdzk27VL431UNzxL13zLbeKR7vb"
    "wAg8PDyUif1KC9781AnL547SLvfe4OvdqC3etauObURROST2U9Px0ZuOj1r+8/O3qhXJa/vx4p/1xS5lI3ZJ4m75j3+Wf9d+"
    "qlGMlCe/xlu9gcYnLb/sjG+lvEt5/T3O7USzWn7Vae1vmhIbL81NCy+PTRfUjS8bbvv7oPJOLX+UTvQQG2EnemgStG92XNVa"
    "/73K7Wrz89gQbr+2tfr0iSjqpy9NyNfn6F1AvTwvuETPf1UT464G3+xH/8EuQen9R//Bf36utsXadsy/INZXYRd4ef+KPF9K"
    "Y14vFCw2Xm4WVn/VJ9GdNVapD/9pj1zR6f/t3Az9l5nVWPV/nVMd75fZ1Lh8zRglT9Vy9B80JY3L34uh45tCLquh5qyan+ih"
    "zqlVT7AKGD8qsey/+A+1FvX5+Yoa8LWUJDW5VEWVZ4v6Nylx7KuPSS3s+/qu2q+X1ht/3Nf9i1Gl+fpUtbjBRr1UprETvLSn"
    "Guf2HfH6N9pnXCv8pn+ThtH4r5fyzPzTmpZniD/+97r57lqAVkWkqh5rOvRP+0wvBRKXxfjlyzXHyIgbTNZLfzUImH4T/PwM"
    "vra/fq22ywZgtWnHvFo3DyXYp23ELf9BLf4ImrX07bXZFEr26Gu5L3VePvw///dLx67EkVW/RQ8lWnb7d+MhTGOr9eU6a8s6"
    "VSW1vtWwr03BpVVwSRy/trt+jfFbimh/krj6Vhq9nR79Ep61bkP7tWpDBbZ6l7AS7TYJq29V9ldI048VEf1F/Gus1rcbqupd"
    "4qauJbTOL3IoP1VxKpSeX0SqvlWxLgClnz7dnht++C5toyCv0l5itSoxRfx4l+5+FKqa/rTw0uO4KrtUcF3GPqre/vhSh379"
    "/V3KZodL3uxwSevbT9fiY/L1wnI3s9+4mzs1ck/NVERG/Plv3/wLT2HUPEUp+H8/53/OwtPVvtlu/KSrXfPqBl3iFtd99Jg8"
    "xInhff9edlV9sVMNbv7Y3Mvy2q7A8lj1CTLg7lJ9gg24ey7fkC6uPlXKgBLcE0taYA14WZ6QxTC8AEV2bmqO23lMqip3ZfT8"
    "p+TBD/LnZ1JJjPKpG5UhF1RKtgGlfH7+d+ErH2sVtP0kJa1vVhAnbAWEVm2Qyal5qE8ClQqvMiy4PCyNfR2j0WNUzxezmo9Q"
    "92pS8wjeLjH7CFXXLJa/dze4fITKa3WsOos3Z+ry00VYUj4fArWOdZHTVZ4TjbqyjNCgBzz2rlfXXSCxmms06iq/wR0q091Q"
    "feoitaZFrxXzVDWt/omfStGE9gR23Sewmz6VWBNPH8Hu/orh5hUt++HSm9+/lwZtXvNpVVyBxNpd6+Kkf1Avod0yoGuW84W+"
    "C7xBbdkPWnJ6aujPR/uhGZ0SHesuwqdP1c/lnq1nrxHT3CyyWt8uFi9vYr62q8t1Lmhazd5QRblyXxVRKef8fbbNMqkRj79/"
    "v3x5dwtas2YuFx01VxM9GpdLil7bj1mj5Wp+H69NvGXacJmv7a7600a/qZsZtL7Vpd59fW0/3r6Xt7u1u6e/zqo81f55ZpfL"
    "4trd1V9nd7H2/kmW156pw269Uw4O8eus6+c/HYIqi/yaxWUlX2tVBdT7vtm6fW+/th+bt3bXMN7Oj1qnWe22pcXGlSdKypmb"
    "PNmN9WklkHi+6kQv9569tN8S+tZdVaoI7XqxV1TiLcf6NsPSE+VGKL3YfB7f1pDZrYLab04S/uUk4dd8UlNQUmsBbqU1Ml3/"
    "4W35j9cqJw+WEpcWPs9JJXK/Pwjf9MEfjXs44JdVdaGyEj9+uGEH/tdL6dVTL+lfVKbezZoWvJkE14vBnsDnH3rz/gK7X/bo"
    "4w/JKg39L+OXahzjfjIlaXylTq1fVa7d7q6bREYDSHTt2XaXaj59/Li+6GlbzdODX7Ne379fAqrzWx3Ybnf9dxOzifWGdPk3"
    "7s3/J/jMKYn1EJVXXZdiZcB48P8OgWD7QvvJJsNrVaiGPnn3FbjUr16gkVHS748fV1csQvtiGfnp08c/rqHZ5anqoOfLAf3J"
    "6NrGewJcd223FHS9me/PRjmr603Nf6pEJVU2dSOfzVbS7lzXUwme/9KNnq6dXg32F/DrnaJNeUqamjw/39erOSMH7/r0W33x"
    "Hq62/E+fovvp7befo4al8S8g1o9Ru/b4Sp4bfqop4s93jYZF/KJchu5jZJQG8Jf4t/v0LrRUqWhlN3gOrlfWkc2GEtxdI/f1"
    "sjjxIHANxb+swsp2vd2tvcoeg0+fgsu0eTvuDf22H4MHu+s/Bg/+a7uOfjcvn186L53stvG3mq9lDcsFpBhPd0PdcG4XKWfF"
    "uF1FnuXBObsIbO8pWWmXV06dN8LR55cKCrhUCOqv3aCald8apUtlUVleL3p87HTc19fuvK6EfTP+evoZE/np08ePNz7yGvfT"
    "p/eRb98e8shOKpvwe9b/Do89Nm6S65rSdv0npbQhrN+ejK7x/XtJTa4c49NHqN2oDH7SYWUlf95nz7XSJC7lpDe1TOXiWcoZ"
    "W365rszg5fv36qlC8Cs5uE+fAqN18fFrdz8anz5dozehTZKqy+v49nuzkMo+sMJlvBN4alXr05KFvAJ/N+xpaWRUKUlefkSL"
    "vUZqV96p9fM/n8BPn8pdu0l/i9RtGVcct7KaFUv8/fvbwBoV9n1oBQLbrvK9Bt9iXPQ4D3ZM+4lhljtdg1ZW1ax+/OcTdHv5"
    "xxMM1vWs35+uCW694hqtpMLo16t/a6a7daWjJeG4nCnqqjUfSnb89qmcJt1ag1lFujDlT3VYu3t1ZSoh0JsqVQFPyY3vLGl6"
    "Um9l9Zb2VL2VgaV4P/706WNaEfna4VurLrkpP7S7jR33Nffq9Sm5GH6/lC7EFxbrEvxHZOy/f69PZslDqRv9/j2pKWu7FC+p"
    "H5+elLoqb9ZDSbprG7KKUtvl85NSVlJJrGr9vJmOdU+V3+rd83fDjY0P9r5V7St1ja/FXSx+m3pUpd9OTFXpL2/7seEFvlWn"
    "8tpJuNyoKiasbl1zUr9tPQ2P8fHC8TQsVRVS1vO1ZuGeKxH1lYO8iNJ+1iNNTiUvcm1Vu9xvL6fHTqfddNCnT63gXf+VwR+f"
    "nq7fb2r7KvOm8+rP3R96F2x3a+fiuktio1W2owxpd+et2g7aVYrqNHWD6281LiaPzefqwHtXrdf2nU4hrRZJSToaU+zryq9E"
    "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"
    "G5JINjNEn2GzOrkZj7SK0Dswl/pOmaS51aF72pVG7dKx2VO17OZ6Ll2rp25elrpyt1QmYzGQ25SGu/y6tewLeVU7jVM0eT6V"
    "attVqdRqsngn3c2ertvtvlGrs+lppUFUDpksMcm3zlpX0tVOTBWZY1uYlpP0Ejsd59t+yRgmic6lJ6+vseqgNFpW+tii175m"
    "S8sahzeNdf5qLK7z8bE7qZKVsiIM8texkTPU/qwJSsnkWE9P8ZWxMij5kFqMUzUxOyWaM1ZRh2VyscRb+EKaaSuiobWmujSh"
    "Su1tLTMfKn1Do8D6Ksm57Kq3TyZ3rYoABhtcO4o9OZbBiPOxIWSp5UxZZM7DTbnM13l5tmTmTH/RzOnzRXrFKJwun8pCcj9L"
    "x/BDt7VUhqvRep+7qIfx5rRvMVsmc1W2qV53MdNUqtUtg3U1W2fwq6JnVvqB40h5LBOHqbzD6Q7Rmg7FhbDMZy7jFSXhqU01"
    "dpmNqp0KK0+P6ekx3VBHJ1nqpJv4vN3LXRRuu5K5MzMXjuvDdLTs97dXddiYxQbziXjpaLNdRp4NYsJ5WcEuR0nYzLrSvqFh"
    "2FTrj/ZEHkxK53mtLHEzJU3uD61js77ftbfrXHWSbV7HxHKiLaezTjVf2hnVPXbK1nrHWf7KaYKQpTaLI1XKsZP8AZ9V+QHV"
    "LQ/6tbMgGWeG6GUmydU1s6PxdhsXGsxhtLuk+PmB0ufyYVNaEPlsWxjzqc60xzbGsfZQzg/6hxW2Y/oDSR4pQqsupMVLRu7v"
    "h+1kc1xXOuxg154MzhmJY9qxbNIg9mqs2wK5c58adHpJUd6N+KHS2tV6u/SmeRq2a0Yzi5OD4VmoHg/EWmDb+eGgxG+r8ylG"
    "TkrcUhuMe1KbrIkGo0m1HVVb0f3qeletn4/n7WzdnEkncbvQVmqrepk0jGm6VMp32rVJvzOpTVeL9igmc9yy34e3pdneoNGY"
    "s9q5WqMW0qI8VfPb5KJ0HfXH7HlW7nbzecqYi4v9tSde6OOWnk4MvLKY4GutyfVi/ELqdOjrZLJfkJWapgn6OVe5YJMpdWjw"
    "eHc8GW3Oi/5QwCsn+rqen+b4ATNGAt1JDpap8llYq5c52DSuXaPNbbTsWJqc9iuC33bLZ6xzOKkHerAg19RKTnLqhaSzQzKb"
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
