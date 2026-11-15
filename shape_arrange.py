# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.12.0"
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

PANEL_HTML_SHA256 = "642e66f969916811c984f0ba4f6f6b44e550eeb95ee16915b82eb587208a30ed"
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
    "fuwOBcW0BZ+GW6CRFnjswhgfGbk2Xuf1KNZH8oFx6E3oOIp54gSIkRGDYXeHRho56iMTQu+U7uYJE560S5jw5M+UMOFJMVr9"
    "L9Uw8i8/Yxj5l0oY+W+1YeSfbAkj/20eRv7JZw8j/+T3DyP/qprehc0BkdTlw7yUUqQaLprmKZBYzvdomXD6X7oyYtCyN44W"
    "S4hQnbrkl7mh/bc+rIbdOrRZdK3vwuB2s0khIdPzcRR2uwc8DBj+woBA5LASiOzQNljxV1EC5EgjTDqhvmLRIl2z2+UVKtVA"
    "xic/nAEO/Kf6ESOeM3EBdV5BZPQQqR5g2hUiu987WkNIkysa++ltBgnKxIOpALFQRPCV/Sp8Y0GF8CP7qX4F6Qqf4G8JFfyk"
    "worzNIAXv0tlJIhaLlkg3v5vFPlq4QVB4XswU78HECi58B3jqrkER/AZ/DZHLP2zH85gRG23THZkApdgkFAd3+lZjZcCdlJH"
    "ABnqaJ2Koc8AFRxvxAN+qBjAMybOeTbnKdvcp15Cn/LZweZMYv99QRALe81qWIOAsNlkEymgbYiHnrFubPn+Kkr0IC5E/nYZ"
    "0AVG0mdglxyOydpRqUqcFU1tIsCO9b3mnFuDqDwyceXocsinURRQL5SgB4Pi+Gvr49GsiRABDXV52zrhPn375s1337oAiqH9"
    "mxAD9aP+c9JIBEYtNokbKuETuQECj17rP8s7TA2IQkzKBiT5rG8CwYlb0wRCZHyxeDWHyXDlT2g15/n6aDlmkweTNJAjpoY9"
    "FzlH4WWWZbl69bydevX8z6RePS+qVy+q6tUPn1G9+kFRr37TqlfPt6hXv+Xq1fPPrl49//3Vq6f52lFaN57NiR/OQYRAHoEE"
    "EqT4IeBpw0R5xmZcaZLJRDSYnpslME40TGrgAozJTIYiuYlvi1+mQckaoyuDaBF7K/ZZPGUZZF4Y87QeClRlWnfy7Cp58c1G"
    "9xaQcvknLvYzIUeexL73AhfQmhZwdR3mP49ZhgdckIZuxz3OH23XNW0VAU9UnhGuPNa1AouMQD4JopS9yTBXa6Ip9JZh4SU4"
    "iCx/hDcY3CqP6ljJDDOwAD2pqZPXCNTiy9WQiV6pmdeMmVwqJQYaKcyy9yAJebac8gjkCXWAcw/zZkVeH9HgYbF9+V0sjfy7"
    "1HqR+oyzq71+NmakxLXGHB6wjB1yKuTfMgLNvNJUEQXGi3nTvzXPBQRLYv7LJK7ca/DcOcp2Ax7zHUfOf+IBOc4kLmdoXgF/"
    "MokrlF5eD9N7iSu1eZZ7Sir0YgfAXotNQK6xs9Q+UmnHxxq9nW0JWB4U3BWUgcXGAPcqHG/cyZQBEcAkrthTucSQZCpsq8pv"
    "851VTjhgfhhBoGjjKIrR4vUVHkztQPKakYtbVf3xGUGITdznJvb69dIPQxrbFwvyFGVTOs+IzJwKe2qWoOu7eZaRH+fOJc/s"
    "4o7I39SnXC16y3WCQhpf54fYcHlbrkkSfMYGXZOMnR9jmSJOLt3U4jrw8JlBubQjLhcGLlmzdF0zCklpZGouU5EtXi7pMsJT"
    "pfE/A9N+ERrGC8MkvxlPYwPaMslbY/2B3tp9nglMrVfIjkPbs3B6Z8SzUOiaRK6Cr4zKkF7qcc9IFZKKSVnoL3/HdoxvDd7r"
    "S4absjdxMQXQjYEc4sKGkbKUYQUYs6ldABkiTX4QWdZKFNnaLClUVosEqAtk8NA0bT4EkWhp0LalEUmWfmi77pY2yKWLNboj"
    "E4jCx79v2iqR9dQtU9JvIKHflnaSTvCD/8Ht7ajMYih/Ko2Lbv04N037b4aLiQzzjg1wCFAV0WJzUuBtZkMwcxKyZ7IuocIE"
    "FqRrMnjtJhk8In9TcSCU6zp8SBOCDd4nTNXnX4kya/OtvZnvVvlrsYUnK54Qy6aW+EmWqc0RXo65DMkYYc7IpYvtuYQPO3FF"
    "1S5xRQ0ucZcpcIRCQfhH3tn3M5VpiJRvpkkuL8ejkZk9nVsxhaMB5+0c9wU/SR3aC2eQdxqTHy5o6tnr1Jsl9mXhAyTeCqBm"
    "9jShkAgxAYyD6JrG7igjydXMXt8sgjCx3XmaLu17966vr63r+1YUz+6d9Pv9e8nVzCUsmvlJn/AA5Cd9cuXT66fRje32O/3O"
    "CfxzydQPApuluMtIGE1oYl9euksvnbtkPbHdl4MT6/HDx486j6xHg8eP7l8M7lsnD08edh5ZD04enj3oDO5bj+4/epA/P7D6"
    "9/v3c/gH1v2T+yedM6s/OHl40smfH5w9un8m4c+sR2f9k7MXg751Cs+DE+vkrH//YtC3TqD+wYl1+uj+Seex9egh1iOfHzx8"
    "/EjCvzhFdGV9F6fWw0eAl2gvf2b4CHiB70Ps3onsz0Po3v0z+fwI0D05E/AvBv3OoG89PHt8+qJEqp8FdXn2tQs4jHEJcL4b"
    "P/QWD0I3G40y8te5M4mFAfsJcMPXOPzlLda7d1VOSmi6WsLmKaDpGjcHaea8Xhg/zcVaaVASms6XfOZ9C/kg31HjwgMRxxYp"
    "ReaCyfAfJZZdLbUMC6/d2GfM6U9TmELwyg85z/4OvPrYOj15eB8G4OHDR48fXjy2zk76j09x7B49PIWBOjk9he8PHpwOTjqC"
    "19iAnbwQvDgYWCeDx48k7w4G1ung5GFHeX589ijndcmrfC4I3hRzRT5zBhG8Ouh3Hlv3B/3TBy8EW4mqBNuJooItxbNgWwEv"
    "2Fqgpjwj6hKede2FmDai6yXSNfFu8tvJ1UnEePfXCu++XW7jXGCTOr79x/58+3fJt5A5MSkxLX/njoMIWXYMdjsUtTSgKbxh"
    "GUZ/FxH7wHowePy488A6HTx+eAJs92hwctZ5YJ2cPHz8sDM4tQZnjx6y57NTeH5w1j/N4U+tR4/uDzoPrLPByQOEx+dT6+QM"
    "2Y7Dn1qnZ/3BwxeDgdV/2H/YGfRfiE8cBV5Vh2HwmNd0wjF4JGvify+U9wyOY55Xgx3j4CgrWdsvGC4PZE0M106xK6fKM3RV"
    "QjNKnBQp8zhvX4Cz9l+cWY/vP37wGDrMv3BSsIoePWSkOnnEKnp0n7UvKCwJzbDiA/PoES/Fm3901hGd4vDQX2z6dPCiNMZN"
    "8yu6PqPjgM2vf6rz6w3j4brZxVm8bmr9ff+p9d+FU6Tn4XKV5sZA5cs3E2aT/VsUTGhcOGeiCz+FebhaTryU2otoQoMfuTaG"
    "WllvPAfXO3ckjqRyEG69R5TKJyv8eSscItl8RhNeeUHtUYFsT5iv6g8tplG8uACfr0icqDFjd7muTDGaFo8T1uVWwZQUxYva"
    "D898GkwqX5lFQWNHm7xjirdqQkMKOsIqzYyVyhBUQMnkHXT0a26AWy4Dn06g/xnhedzVAZxQugSr/dwLJwGEDy0nwuUoOTQr"
    "jWkDJDkCDL7FkfUXCzrxgbVqWsGTABKeM2OcoJljGDzzNaek6TjAP5uNETqQdXoGaapj/nbIiGNzW31KuAHmSBly09xs1lnG"
    "kZN88Pvgl2MHVCK0iNzXhbmzFcMDTv7qsHe7vxrUFAnDdRDOQZ+oo3fgODSH5wOKBijErIrSoJ5o52USHSAxut0U7NdIxtep"
    "l1KjSA2TnWypKFQxAlpN8wmmHGPA+RMtsygcF+FxEInZQI7Z+AdAo1LdIi37wk8NnVikplkAKUhJQk1ihI7B05pLFE0rCi8Q"
    "RDIKZ4aYrKPYn/mhF2D+dTsVu+/MzAgkbP42Cr9ZLNPbqsC4tCzLi2eY+DsZsezOvxYPWY5gOdOesfxa6LeZkSMhaRVYIBpa"
    "Bh1lkCIE81Iq+Jr1VUHVMGhh6BkN5BShDrWOpso7URF4/WDlpHZ6pc4W9ilVmYoqpZGyii0zy/KWQ1MVYHVnUw2EgCVNnPco"
    "smaIH0ryp6Z6BUTUpBByqH4XyEpcGlHOcZHwOqFT5YAtxONcRDS4+qGf+h6bPaTEF0WWUCGTMmdcFod59DE5RM6AXMbtQoG9"
    "mZ3R7BN0JBdvlY5gWyR15PtUsgk/+5fOBHA08d11KLwA+EG4deTjq4SkZiaJYriKjDQ3m4PSGzjRMFxVg3BRiHOlcptDyVZv"
    "kRYuJzvqdcXTcqm6KaJVNFkz4fjn4ZCPbuwv6NUKvTmm/szyQV3/kcEA69WcYeaHlMPyASbQ7+fqFuANntEJHf+/59yPpuyS"
    "ie2z8zwG4ApfafmlKWeqBKpLA19KXluCL+dkVPylc8hqOO2Sr38OqnfuFxmy3wfR+EO5iJK1ty6ndl2Rm+2BzBXUaoKS5Fnh"
    "7WIOedmVUkUt8s/nOet3KqY6tO9euhAjfZ/GRYz0NmVz+nnLJfVinMAdsOKcN2XRrAxHMYGmJtR7jed/MZ57Xm1d+HYJYdOQ"
    "7VAr9y4qwW3yWuvj/De1gFHlW7bAItDX8Wh9f9unoNWWUbPQ6itVEtFqAVrloNWWrEtDq8zzHtcGWxKRQ7cdKGvZ44tDj6nn"
    "DdeZlB4gqEbStWmhnvu2tVd3G2ynViscua1VzpZtWpVXedS0DUtvDEeTnUGbxUNUULOKaIdcltk6K+1l4I3pHG1vCoraWhXQ"
    "Nlwkdmy7NCF4dZemesmiPi29IgMX+sW9cRlOFvusxEqpmybEg1kbxIPZHogHs30QV0o1It4ixbx711s0uaJXuEgDF2bYNDy0"
    "U4tbEOCKSoEf+CPubHrJAu6sFK9xFEsEs2KJYKaWYBc7WAlkT2xZbMkPy3IFvyrOZOxtqUWkIOsBasujDGzmjb7kUn9u6U7+"
    "fTt38u//TO7k3xfdyY+q7uR09vncyeks5/Z0pnMn/36LOzmWYu7k3392d/Lvf3938nAmdpCa3ePPVY9yaWmNQixQPQnITbAG"
    "tVKY2dwyQWjROLnFAxo8zLkzn+JczjULbgphsqnW4ZyUzLCNjr5Hc2MtVlRWvRBA09xfmAkTUsCiKomyGl/SLCPxzMkdnoDw"
    "cNjNMXaJi67UvF3FlymXOf6s6KvJ8eeeYz47BHxrcPdutsHXOEcKFy9huUMWoGijzMlJFUdsFbFDm0ribDbcj63sbkkEg6SX"
    "/dFm41Rp7lkcpNuVPy1vuQxuDY9IwzbwjHTcFL508czMwpnw6vJnOD29maO1coTI8Xo7B/umLLQTP1kG3q3dYet3bxrQG2Vx"
    "j/iWvRPTwIM5p13CWa3imqqm8mKtXuDPwp6f0kVid8Y0TGmcf/xllaT+9BaDasABQQUA6rI74NngrdJI0VtXcQL63zLyiwW0"
    "mjBD2eK31dtpwsUylR0Z11lqwPkWcD+DCK9qd5NI+4J6o0j78jVmkaYKtvOSbrMjxjkf/cYadowPohm5pgAhzSzSGAhiO65b"
    "g39ommyM/tGMbXMKQRXdJPXGH+hk6/yvEx/5NB50BqVpLOyLfcXKtQcWPej9crssgqcev00AiI6jYLUIdRLQe59EwSpVuuCH"
    "CU3ZbgmQiVO0ipa/c8FKYdAKn7m/VmfsBWMDdjqdXudkeaOM1289P5wgmfagA7pAwpKiGRe2JbM7/YoVEfsByEp7IqJXJzeL"
    "1sVOD3q4D+ugj/HOyAKanx1V/F3FrzDOTYgIW+AxzkNQ1izcI6ob5Ua85lHs/xaFqRdsnYANxnotdVrM/W2ttzX4amXsx8Oi"
    "KjvboNGQf3VXPBqmHx+U+y0mXztm3jox9qyGszSXbdslchM1ipOGk+Ck9hSjv187DXJETIUaqrP/fgS6f5SKVGGip/s0iLyU"
    "BcyYe4lRSyGzUw4jtfDimR+WBreN1qofDHGlUbU8NiyqWyr5pBKtsJzUlBOYCHHcsvO/oyzcgsNnkoR1WGyXg590Rt5dpFZM"
    "0UrtTPdrmKXtKLSfbJQh4XA/Xdwo74bBdql5/9OJuruuVdvHp/X6VR4J3R5FX7LF+UV9sS1YiLq217Qnw7VDtrDM8GMRs0Ze"
    "w9txgY9Kh1GK2lt3jKaYU1pAzxVnnS3gLXsXzD5G7+rO2vS9q4Ou6Z0OXM8e44B6cRnrpr1tGi3tzqk6pbjagB9wf9MbdP7C"
    "Zus4CjkOnXudE7OFTa66Na7fDGntFwooNr99hdKpSTlZzL23eUY95p2/dE7ERk8h0j67Tu0AVqn4STeeHLVPRcitaH9kStfv"
    "Ij43sT/O0Ta37SuH2+IM+joG77GYnXXXHDmrgNXzcHDg94IgukYfYMdxDgbVQuDEAmXwh3KGXmQfPAifR9csnFMCISMZzV54"
    "t9GKH+EhpFuuIR+xFpXkwJV68siWW2vJA7Ycahbcwnk7UU/UXc0kgGGmXiwCNeo5ziUMhb+C/bIMpto24QoeV9t4ULi7sE0e"
    "Y7RJeWcclEcvqpBv4d0cOPLiy8K7eQo26PjWMIFAUs/8ZAiXNdntCPuhirAfqgjDkWo0q7gcf8sP2BSnY+7YDfPcq4022M+I"
    "0nrD3T2VCetCRPIpkpW5QI1EqXUuLw3CVnhZ//YAk7Lq7aBBNPYC2gbmJdzjo3EzKDjf11BqQsf+wgtc8L6nU/+mqa/JaroF"
    "gt2UHd82oyOgvuaHDo3Aq4TiZG+I8gk99MNnsYfT5Wt/BtdYGSxjxipBvJtdwPGoyA9nL+vpOPeC6Tc3Sy+cuITfpolijUPO"
    "pTumPtxpnwYR3CZ2KS/kpvEqHLsEa7pgMPDzGYdTG8CHNwo83BZzR5YfjoPVhCYG+lYs/FDfLTZWC++m6XOS0qX+OwRClYtd"
    "w5jM/dk8AM38u/AZrHlNMUTn0fUFCPmmWKbUm0TN8VkVN8bmObtcpc8nW0Hah61F8NfoUaC9TcygZJjAgE7eNwZ2lZCN0V/p"
    "rys/brrmLMKmerMWrm5slFs6uyWzVs5uyexP5OyGfc59xcYz5S6po3EEk15gomejzYZeul99JZ7dEbh9HYQsOnwce7eWn+Bf"
    "g5p4e3gGTZqbTWquw27XoE5onjPfsT7xVScacflqndg+0UWfjL90qBXQcJbOh+tJFKITnc1/DYQH0GV8fDzKMlK+0g3+ZzQj"
    "U9vPsoo32nPuHu+lKV0s004adVgPKZiUevj7fUA7Qnex/uN5yCxnAPqedgQEQXgPCNBhTnAJc26be1e043Uq9DTMDvNAAy83"
    "dPohEdxiTpyDQU4QhRih9LmDiFPV+3ChFdKb1JD+gpFDLSARoWWaJNCO5yBRlGrS+HYdwXaAVZDf7+U+eGY2hdu9wS0MeWIy"
    "WnpZluV8FRT4ijtNfqC3INWBX/ibGU2V+3iMMonwLGyCgWrSbteInRgc7FMaG/X3/4o1fE2Tcewv0yhGFK3cXTODcBTWcpXM"
    "uc9WSGLheNgJ886tUG5Mo9hgGu3gPP1COndx/jxPj49F9+U3cPZkFxzXma2+PU//82QYzDhZjNCEcGDg5feNN54XejbhznKX"
    "6cjMTHtLDxOtv6oPiyvZVtQITdNWcdIjVO8Qu436IUlNM5MEVnxgRScVF9zpZ3TBnSouuEutC24ya3bBXeYuuChwP6sLbjL7"
    "XV1wJREWM2VVu5rhcvCO/ZmxP/OZocDD07qNYE6WoEZ9VrkskZzlI0vxv2JpzWkiCH7LQNkgO+vMSiNGLSG9rSTwx9Q4I72B"
    "5KwQiMs4vLzaM5ZQ36Bzq0mwzEsPU3bAz9c0dYdsKZ7G0cKgpo3vnwiJ42429/5tDO23/ua5GabG0D7bDB5u7p+YxtC+CLzF"
    "kk5MVsPRPSulCUz/Ie+PvDksaYKDutZzcVV/OODLSUmLYK8F7QrY5y1diZbKioZKcwWeo7wGnmWtpl8KDQLDiEh9wjwX8jx0"
    "+iR2WMWpeR5+kZ6Hx8dmfBmOHHoZjsRIxbhYvy86m5ftBtGs4m4uAhRxe5GLpjSXuO+DVeyOtsc3Z2aQZ8wMgd7D75i0C/gT"
    "3w3zJzQk8d8LP1wlr+EkkT3LnS575Ftj9sB30uyB+ajhb6zvYu7FtusSP3m9pGPfC/AFfkfLJDt0YIEHym9f41Fd+e034YS9"
    "Y+3m9TGk8mcIPsV/1gQd4rbJQoggEYioZXicPNJLTKeJJW16MuILhjI4ooESAgZ3KSL4SqkcgDIAizspOs/GBjWHLFqb7eIp"
    "JgQQEFaTYqgXrJOFbrkQ0/+VFyeU6TBlO8qOhdGysmMZyTp7lhNmkx2Lq4aUXXtZsa7sWkHF3rJjBXx+7ViKT8TdSmUEgmOm"
    "xXAZIlBQARzHfxWmmsga5zIOSN0swFAgyiygu84C9dLLkM8CHsLQLKZyqCRykPO9yPs8srzyitsPZKCxnHcZQfhThTULX/nL"
    "AgOyQchfAIv5i9WixCWs4TL3ATvVApc5rWi8y6Ph8zd4B6kwrJVhV1cNJ6TXnedhGljfKm8NhXIyKYNIyWAyNYY6i5mxvfRa"
    "pdLBIMOdAwA9Pnv08PTB/ZNB3zStGGIiJ9QwCcPopbc0qLXwlkaR10V8a5KOMtNk4a7Ekuf8QGff3CwN9/JoTa1fIj80XNfM"
    "Ri5xZy6/LMWWQEd0SSD2zc0ypkmCNOKAcn2UwC/FGw20YA0JfMFfaGD5mixBv2bPGkgmJCTga3zUwDGxIOFe4aMGDpduR2eE"
    "gmLMyKoVJCWJgyHMeIoCjQyhydhbUjYaWquWFVO0bBr3LnuX/xqtM8P8y/HQIv/617+PNv/Xv5LRvRlx//Wvoy5MfB2FNIaO"
    "rYy4mhmrmUxkUOBniH9f4VNgRSqYdWDlajnsefIP5lCyHec0W+HDVHa1yCfEdU0rjf2FYZYgODMThXWRBlVW3YcIxV72s2Lc"
    "MKnPKf2jD7cjao3nXvwkNfomUfteqLLQGc1UuntvBnlvCpKA96Q32JHi1Sms4ghbHHVR2AFrflNPlHTvtAjp15l+zZLS37Z6"
    "NJJwkJPu3r8SmKRuE1cUvyAnFIhcbkklflmG6UjPxJ6p/EbeVZ7PaZDQXQfmoygHGV+cFLxUMibLAMIrDlzzsj8qE4JPHVWM"
    "GqXKNhsgpEKv8tqgoxdbTkzld04v9vz70evj83G5mzXkH7Qnf15Zkfys3mrkWLALcVMG/ga7S8/NLX5iXFh5YWXcRyETXWMG"
    "dXVyYMNf9IX5+Q76Whv5udm4Pfc8dcrLnlSmAIinaqIJTX+gM3rDlL5jBevjNANG7KRO8aW6VrHBgN3FscrBJpFgVBrZDCFr"
    "IDvLEvSU6mAJh5Qi4qxSjTRjOOmW8TrJWFj6a0RjPdl6lY9cjSSuVSuEFa2PMQaTBUU+TLlR8lgGgfSTb71vwcoHrGuH8tSD"
    "H9IuqZcWFEIwzjPrMzSIR4B8iNkBN/vqO+lmc9rv8xi1sBGE2OKx0E7RmOMkNIW30Sot5C6yWLMGJQ/6GMTUFwmtlj4LLQ73"
    "6CeTn/x0/iqmY78oA5XzJ5UrSOyk6qPvhLnLAdB1GApRYaGo4AZCG07K4hJoXAcaOYP+X/7y0kvnsJkzIK6BoDS+RLllGPQ4"
    "Nf8SmfeijGUSKiIvpjXbRqO9UJ7UIh1SuvxLSsTqJ3m8Ugo33iyU5mYDh648nCOLl6qUKVPTiEkotlxsk4BWTsNnYQtcQFpM"
    "bvYdBDEcRYixYia271goBINCjAcYtih8u2TOSS+jVYJ5SDQBuGWiRH2P0NBnSNnCmYUFVIBUQEvYYYYpDwRrQOazUrtvl5WN"
    "cik7o8qzleIvqHdF963hH/R2/+b/QW8rNIMcZNEEV5zXS29M3c0mf/MNOGcX3ny7Wiy9CXsvcs+ViQhNQjufcqh69WNVanp3"
    "cpUq2Gu88jr2GrFC8U8/Zj02aG8TGpeC2XB0Cyb7oqGuJCscnr4vYWoOH7xCeQfTK7PJfVE28VeYpHw04JSC6ZSBsJocKKEB"
    "u22I7yvQ34QTDew34USHIRw3lNS2A6OwgPHhlKEacm0Bjh1AGTwvou9o+zgcui5R0EpfC8x+gODphrbPur6Jc8QxHD1D1Pf4"
    "Fl0QQisxzPMDA46BQ1ALwdXjXCy/McPtnA/dt0xJgKEzfMFA4PUeAwP5ZO0n3O6Cp0S8lPLK8E3iJ3IrLyDkC8OHE/1sDJZX"
    "oGtoUTyQ4/4ioTU1MGYQHxHdlKioEt3uAbX8hI9eOGP6reUFUBxnRxoH/Cek4vkHvTV1rMoTA0iGdupZn+2JSuDF4T4X7KDn"
    "zlDPiyR2wl5K/DLre7hlOE+ufSQcTnjozwd6a67HkOXqSRxH12+Xrq2qQFrRef4+pt6H87wUZvkqlOu1K/iCTlPX9qdG/OVA"
    "5NDUcZKwBaWmOUyPB3Z6fHKuJW6V/yMSmVzxb6q4NwAPshYo/wC+mkWcEyfsDdrik5CkDT7bsXnjvXdt/MXENfutCnDbq9XD"
    "FF3Ol9kZ6mW0shk1vBrwhOIRtP9+lVKDBeXCGsLo2iWeVofzCh166o0/JLhG2Tj/yp0nsMcIzXX6peNzRZgLmHwPLf3wjVQC"
    "9Uow/DVpOWApeBHhZBw7BX4hFWs71P4cNkY0ATm2cgJrUvpCJpqXoBJHK26ekpvtImuM+QIxLTf6Antj+PkunW39mFvF2DTV"
    "dyBssEkHdhs+dxDpk7R3Yh6LR+gcCqiOrJFjXKpTvFVrnQ7b0rU3INCSXcRjoOCRY7H6st/tpl+uGA2WTnHhAH4yZNaWyknY"
    "ZtM3v5gOXdd2++55pbml0iCbmZAjfTA0KpBu31VgiVfZF3nml/2hh5nPG3qVKVOBB/0jHt/wsCRuPViGIE0ww4c3xL4x6vm4"
    "Qybba4oxy0phon3NEsU1zzKg80JheJPMt7D7zJlX2f2d5mUbdl9wdr/ajd0XLdldGY7jk0ZmXzQx+1VrZj8ekPS4zOwqFsqU"
    "mzFmnzEK3O7B7FdaZjePbwsNMuZ6p2H2EqsfD3Zh9mKvWjA7BBr7tBz/t2hBXVvD7L9KKgpltbhCya9qbd+Ek8bKvJumyrwb"
    "UFBVDfUVmNebVdR8c/CB3orcQkXFOTWFoUZVl0GGorJ3IPd74EVRwd1gtlsjNb90+t2ufPrCebzZxJtNWFHm07IyHxYU9zhj"
    "ZphXXpLSnbdCGtWHEQBU18Bfvo+8ePI17pOu/XASXRffmigwvNQzXIzOisLiQGQtwnstsNuFC+UzGmOm+W733uW//zXpjbg/"
    "YmpCjwvmMNUAZp6H0jOrQJZQsf3xnQje2bkI/PGH6pa1yCFo1idNBg5+qyinc03+k4UPHpncJZG/+AJyAyaaorRYNjcRM1KA"
    "UY+fMwzLRuSCODzom3jDqMgXVec36eernBEojm65CXSKVohL13KJS4pXt3i3isKamkOjXlpX0HsZTWqTPy24RSS/+qdd+LTm"
    "/oR68XhexET4lWgQOxdG5I96GqBHojDOOiKty6u1DRe0GhZwO8zYyfYnIkqhnOQ7UVLLiPz2T6ksP8s5V58qpfxyKXkUXySd"
    "eL2Vdohg/iJU/E3Zm1geZebvfEy9BBJFe9xQvgvy5Um3m9/2OBlJB/6h+tYuyeuDQUEa4O1BJ62lt1SP9HOfmfcLsskwNxu4"
    "QtYbyMMavbZUMGx4W6DAyBE1mFb4ZCFJ7jal6KmRScZOUtVTAwdFpDJUZOWou0axe8sPmfO3UyBOqK67Yp8SYPfJwtGetYud"
    "6NxZgMrnO46zODfgT3+zgT+T48Fmg47kUycixnKz8Q4cp89eKOsOLHJGRFLSJ56p1ZGmJIWM5gDsmlzJ4kgrPGGOBR7joaaS"
    "SK3ENsZf+t3u+AtvsxljN/lSWHEurEfXLxkG9NjmDgQzp+JrCGe+UXBFJ/Lg2tKe7pN3DvTLG4p6bRcVxR5/hCFkvf9yjDqK"
    "f5wKA4IxPh6YXzgzsSWZfOn4w0lvYK/gx8qOxJ2sqRNJPdg3j9Nj8ZhXRq5M+faqtvPvzIyN0e50e4daR16gdKpKYuwdO7C1"
    "8GaRcsYoWNJxnBMxaSnHForuu6r4X/aHVJImNI8rZqXUPBYAsWlrPm82VBwax70Q3eX4TU0NMN5H6Q9Ttc5YLVNExi4+KqUg"
    "IIHcg1QOqMUhdi9V63ZdO8XmJeUYPprGU9NWH47zEjCEPriLJVGsT0m4TZCmDUISthCCIWNUvIjfKKM0/he5fkJoz/FZPHkn"
    "FRYDxW2kuLX35GUaeuzLWzGRQ3uD8+hLp3+ODKrUFMmToVI169iJjn22NcuiXi8Dqyw3AZotd+bx8YDEYveNBx6RQ8+jL8J9"
    "0Tg+znIzZFsklHug8WbTz0+UiruGxuFnpwblow3Y+P0aGPn2TTAV7CUKXdF71Up5MBAmB+EJkm8OtJq4eK1YZ/KXlR2GUNwL"
    "HjxCYc9fVk4bJTIVXUQrh+qMQw2ajfBermZIZsK0QbPhRy/nqbzP4Gl2ksRzDjzI9ik1KIwZMezbHqk4RniYxiKtcX6AdcHM"
    "SOF1RWjlsxIxYEmWJwZ84y43xD9X8zXzi2SlrMuUiMuwTPSmdMITrGcmMWLH8DUZnREjma81ZhcV/dqq0YZQxNPW+e6j+zS/"
    "IXIgt71yR3ngOJprlMPyUFDThhvvheMS7aRge2HcFONla3RpErtseVXlC/FmKH7IyxcS5kvxZih+2FQwnG70JMOdp5uN47pC"
    "6NZzX1Q9waGgHHtCsPtTIzpwHDiqj8QRczjmsIkRkdgEeFA71w3nRFHb85Q+6YvjlIpEklpS29O0MRkzZRapErRaF1fbNxjn"
    "TR1F5CdOpBAQb78W1MnCYY3CY4bHVlWx5gemSZaOMeX8C7rDVLUgCd3MNfnlE6PiA28OXZMs0AuTO5Yu0Yf0fMGEa8TIPXfS"
    "tpXN1MrmrLIZr4xjvshFpAknXcrz8Sz/3ZYpVmSVb0rANpeYIbPQITX5vWONvbitAX51PCArMMCrFe1aR29AVnBiZaiVMHMz"
    "x5D5q7Ve96Hb0uxf10XGSu8cTygigAO5Up5NcuskvQl571SW3Cvz/H23e4s2/tWxM7AP3gtNoKDGvINN2urY6Q3+cgvm/saF"
    "csfx9Ji87HbVIa0RJg3y4qZBXhy3Pn+/ITccs9WxM+klO/Vo94NvahLD33LJ0Xf8wiVHf4+rvlH1qi9R5XjV85Oq5u1993Wl"
    "o/VhiIaAIeTFKmwWchBmm0Q5VGzssj86TptL8a3RcemjrbRaX7PcWMlwIKR8qlhxtFesl5p6cX+jbpgFucBTt9blex9bLmtD"
    "YC5VUjw20F3cVfLJpZBCTmwnWISxyiEEv9AOHkwHhZMYoZXm24n6hbG4zyiHNYNDp4K7kvDV5DomD1Kg4Po0WGkuzpPwvIjy"
    "QBp7WfUNzsfKMszd1oTrMUcCIyTUKaO8CCi3qWOEGuUWMJa6bcp02xAkQFznS+PDt0b54dc5P7caqOoovAhg35f7kFY2VOgx"
    "3+1KCHCazz+wW+o8vmXd0QmPhfClI3RavBvfrtAXjlCWC3kPV9wt+YWfpDSkxRyI+Z5YhLNagDPupOD4lyp7Wo1zuMHiFKwS"
    "ulq2KvR2mRcJim6/DaXQQRgLfqC3bRDkDoyiyHbk0IcYwLOMTKRr8Meim8ZXuw3lKn7WLWmnc69uQ72KT/R2+pW8sDkFS1vb"
    "CvEOVN7V7uFVgMJ6WtyR4WL0jLcW100S1fDdmJhzMoN/nzY3J/upxpnlH9XouBUYUyQ0xejSib3O4x2HM/IENN63S/vXOfuJ"
    "zrt/nROQRIn9z3mWkRvIBiqSfJLXhacLfMpTg8oP13Uf3tR9+KbmQx7d8etSWlFmxfwBQ0qJtLCwc4Y3iH8e3ounHk2WXoiZ"
    "R6vZRjNwn82zuZpkXU4Wmpnk8kcjIuuYTm1ho/EhAyn3cBAJSo+A076FLKXw4yIK0zgK7HUYiSUS45DGEVwfhv2LiHrMkJpR"
    "45IhpoSmBpOvlUf/HJk8BMYE8tfmYT5NEWaTzR7PKk4nniEVp8bCDyFF6sIPC2+9G3zr3ahvw+j6UMnDiu1hiBlaObVH1+6K"
    "j9TQRauhP3bz0L7aFK4ymiqVK2shfCpVc8JzFAMZuxSQLEYzVUEKX/MwpdAS+0kwGQRl007M5DyLLOHz1KYW/yXTx0JqWHlB"
    "g6BEQ2kpU8Zyqcg+4e3ewjd0SALHlfyCAMt/IoHKdzGKwBDZXgeKR7fcIwi+4y90kAErd14CHknE4+HK1/hImJqVv4UnsgQq"
    "LNNFmUlXIfIlkE38rCTdzZj32Blx/QnGO7OQ8cUWS2Fgq8CuVoFNrQJ7WpInJS9JBlJZxipxjPIs2QDH3+KDLoZa5v6Vg2vl"
    "Y2lVR84qj4+gvsWpb3FyczXWWqaSYhYXfeoxIaGWjEYMqr0q4Q/UkPfDZwa1jpIgShPiolLJAtSvpXRhwkVub6FyfLjwggDM"
    "IDjUuQtVRioL3uWPRgJSFAVhXhEXYJeVBkYFllOr5kxUQEbkT750sT6XuLy0OzJHGeGpnPum/TfDdeHIgtNGLoG0NjfB0Cgv"
    "BB/orT0oJJ9WIvnLVaH0Vr82qISX8ddlRPtAKoSeVVGudTR+KhpVFixO2lL0eL44VCPEjzLya2xo2kOqrXUCOPXes9BwvQGX"
    "nXN/MqHhoQ2RwKlLWM5ujpkkTxWj9iTi/KmjAQ7Xb8bT2FA6CHwyZCNou1x/cU11VS+AVnFkjCYxHPcSZnc6tEtIZVVmHJnm"
    "KDNH8OpiBj+J2icZS1877JrdwY4DX0oDgAOvSw0gBl7T4kcf+ipOW4deFmk59IUUBcWhx5tQhcEvAFexbBj8IlrNg389kz9f"
    "z0xFFn1kKdAo2g50ok3hHRBuffJ/RMcfTnS8UVnmkwiRj803/0fy/FEkzzcl3sGXNzMzez+zYhpOaOx8PUMb7Xcz558Wi1tr"
    "8JC2C5ok3oy6fPvoihRb/L0u9fos9id5Njd46kHc5sCDo7joOrE7g2lcm/YSEmnxyuuzQUarFBN0Kfn1SmUKKW3VIqwfLKts"
    "OW8Yr6M3jsIUsgvxnFdqVl0/7Ik0ff0txbenpceUnhDAepEUE3tWUtiK3vGqy2lrkdDeUg8785bVlHkS1yBKaDUh6D4I/7JK"
    "Un96K7pfBcCEwck89sMPhfSm2lTFeS5OHL0rGk+D6NrusIl+rklvGNPAg2jl2vSLkijQX23e43IGRm0JBqDJb6pj30LJ+gSp"
    "3vjDDIPB2J009sKE5SxRkjTCS9ZH+a5YsNBqDm5NVrEHP/7LJIWSmEZx10JiAu1V+H0EI+9Nouudm116Yz+97fSt+0l1NvNc"
    "kVq68W88+LZmYvXPKwmn89yxjYktVwlkusNTMlubb1Y7vZKrGWkE8OuTjRZ5Skl+2ILbtdD1rF4Bb+6WjQd9vSs/8SEK/7qN"
    "kC7MDCxvYVT8bVK7TR0Iqqsjmk4TmrarhMHWE8APp5GabFuZw2r1AGblH3VYKdlOi6VqMpvXwVcA8ymngWYftvRv3wGvRbGG"
    "2LuhXjvud+lSJYV704AWVwSWx70yxnoElEdOLTWfdB3hBGiVUttILks2p3ivQTHxF8vCCNc1wwC31Z6sxmOaJC3IzCF3nzqy"
    "YPvZI4q05kJRYBvDie5+xGkkcb3rTNpW0V371nY+afFoP6VE8V1nlSi318SqFG43t6q4tplektlazbBrLw5bUB3Adp9brFT7"
    "iYXwrTkSobexHfbvI84nhuJdJ1NjLXfqUttpVMWg/RzCsrtOICy01+wplmw3dUootpk3jJ1aTRoKWaZaEBnhdp82vFj7ecMK"
    "tOY+Br6NzVgnP+LU4Vjede40V3O3XrWdPRoc2k8fVnjX+cNK7TWBSkXbzaAylm2mEGesdqodHUfhxItv26z5AnYP9S4vuoOC"
    "Jwu1V49kka2KkOz4x1TzcozvrOhtreruPWyt7Olx2UHdkxXsrPDJkvupfNXiLZU+Dcat1L6cAVvNP7C5xl6SthgFAbr77MtL"
    "tp98skxrjpUltjGm7PNHnHk5unedeFtrunP32k47PSbtZ50sv+ukkwX3mnPV0u2mnAbdNjMuZ7tWEy6lN+l2ay1AWfCtYn7F"
    "AtcaG2xe5Lpw1qAx1oz1Rzb8HCU/udEXJdr6dDZq/FCwTdcc5NRQoa21urWduoWFOlk0n8g1nq4li/IB27YGdmCHZFHliG3V"
    "a8erCqIdPQ1Yi3MGRuJF+2Fb7DByi90Gb9sZSgO42tMt5yOLnY9ItnUjmN2BB4PZdh4sNrADDwazNjxYrF7Lg1UQ7QhpwNry"
    "YDBrzYMa0FoeLMFuHbxtPNgAvtupXrv+NhfYwrhb+65Z3reeUjec9pUX8MIpn2b2l1fr1m03nQarelrTiS1vffeJK/UGrYdG"
    "A5Eb9TvSAr+2SmGFLho7AZxz9+AmxpU6Al7oLzzmZbHssQdaLIPn8x3qJdixzjSKr714kmzz4NDMO7hBtiMGWKbTtwanHAU/"
    "3BWDrz7Q22nsLWhS20NlFsfRQnlU3BQK6lCd+1NfdX/Kcj+PqK7OwfY6B9U62/SMUa5dz/bDorlnJWoxT6B9iZj9B79RQBN7"
    "DZeytFEul3G0FNcpL93ck23Zk3feOjk7unAvHbLU+uktWR9Wp++hnVrKfTzx2iWH5TlbgmQvi3ALgAHZjAALLwgK34OZ+j2A"
    "29RuNsIwAiBzfmLuarZb68nmClANjEtgcVA/MJdHUBjUt/AMdI4SytwoC3UpsogDPR/rQJjfYmaSDzOHuxo+9RL6UrgbMi/E"
    "xP77guCA2WsxCvYanTtZ/BVxfQ28aaeRm2GT4CPKoZ5GUUC9UIJBrKrAn4rPLIiy/MpSnSEZtI2wmySZ0q/tYIxGr1gfEJol"
    "ri80iRqBti72XVwkqwfJuIPmdzOg15WvC5u7PlqOOYHxIik5YovBc7iBFI7ZyyzL8guUT2Ywcw6/eg/3ne7NabCkcdLpdVio"
    "pkNxWfLJzOHBm17fLt5HAdxL5VW43W7hk+WnNPbSKAYGxzfuUBflmMeCyrTRnUp1FpujeW5iaMVhMCzsFMdhGUdpBFUMBQ62"
    "bJBgnzNJgl9mxVCCRuq8hPw7ph926PC797/QcQp30PyQwiDTOL1lobdF7C3K4p8BTx70Yf5N/dlKPkMcCP47M216mY6ckNC8"
    "+ZezXHZ9C6iI2FhmTnwD4nVJaqZ2euy6eQ1YigUUwa4dOG6EWLubzQHN01JinBEMmKHS9gt35Xa7RujQS068NHoV+wsf1ueR"
    "KePYilSAIQuqAC2esxZjpUXRWnyezuPouvPmdkm/gWMGw/3qK6XizmKVpB0O7HWW8j2LHyuTyBr8shaLFsbmBZ/UJowi4PRK"
    "CpiKcPkwI9w7EMI7JBBvF8JMJDZ4UUcJdYkLwqJHw4k7IimLtcDnr5d6mhnGjY0wmnh7PkzVeGiFC/0ssrY/pd2uPvshtXht"
    "DuAlImBIhEx+5xtewFV9ms6jSWKvEfNqBBGlLjWgBu8nNQvBHPDtE3H1te6u+xGMC71a4Yyb+jMeJxyvSA63QrAdiRSWDVfl"
    "f5nBv7VYWm0ePUSz5rI1tQLAl1pxSV4s54UnU3uDHsgy8WOK6ihoFf4SGvhuXrwhn19+z2Xns3ay89mfSXY+K8rO5zM1JjaX"
    "lR/obcJDj/I3kNP2OhTSk7WTCIHSBAPVpN2uETuxNfWDlMaGJnaEtoavaTKO/WUaxYiilUtpSAEfWstVMre85TK4xai6Mhlq"
    "3rkXOL4iOGrqDM7TL8phv8/T4+NKSHCQ8SyU3Dqz1bfn6X+eDJ/POFmMEMNrwo38b7zxvNCzH/ii9P9z92a9jStZwuB7/wpm"
    "9m23WKZlUZslZun6k1d5k/e93WZIDEm0uJmkKNFOAd+8z+MAM681A0yjGiiggcZggHns/CffLxmcWLiJsvNWVVcP5lalRQZj"
    "jxMnTpz10X8S56LyyQi93GNKByf90mdFC5YoKsk+5Xdo+Tn42exbELxpnvD5FE3wxeLJ+/Y3PHnfEifvVu7Ju/fJybsVn7x7"
    "f/OTd++//uS9SXn+6KTerlNvt/AWW9YnvXzsJzPG2PVoqWsP5shD8lo3bkGlCDvt1+OtYNpghsW8bzh9qxDfT1RRgqsSchU1"
    "NhyL7RTA3vydU95nhUVjMJ2ftJGdnKYHC0ZySUcizLEHMsDpHvc/oQf4q6LCXdaF6efpyLdNvR+ZuS16Skj7JBElMA7mHUg2"
    "nb4yJuzWsx9E6ZFbwxHeFtIt7KZ9BvBU4jPA9nDCLwB5j0zwY0v8vLlJW9XzW6m4bJCJLFlrz1z/BTr1FvCp6WfCbo8Ge1kw"
    "pu9zM9i+bT0t756+1D0BB2dioydF08vA6icAx6fOM5Y2Tb8vmDKS6lVqAHidMkKm/GFAh5vHVmGJhWdmgeLbLXQl6TaFrXtE"
    "QKbNMSPfDv5j6en791YOlcIqIF6XFgd5PCwcDwvvc9br5BVbjB1EJHqXnQny8QODTlYvMeHkc6EvM3SNsqp8GsnbB4CRKBFD"
    "xz6sBwNEOzvjf92mlgAiAYvbIUzWo/f0lLAZhQ+doUge6NtN1h75WanMad2AVdUncX4WWZceUevS10XrUrCGszlYZCxME9+S"
    "rPcPzP2WS8A/sNmC/7gvlDVi36gIPdv3bTNXAJPo1cfmg8mMoyWS/ESeNeKPJqmisMz2jJ5OhF+TGUY8N6jn2cbET3zzbSfF"
    "VgUfr37G5LK0MGK5VPqHxeGlU/PM6XL4uHks37c1asWdZC2zy5wilDKj4yZ6sjOj1rvL5Dwpe8wa72r+rPfs2edWr58atn5o"
    "FrusQ/ljyoLOMlWrrLJPqkyuYtdfF4h/g2FqspLfZJz6Gwuyif4LyqdsVH9j6UXz2E8r+PMFmNnql+mSpRBMWl0o7tEa75Hy"
    "W7v9qWA2I+5OVv2ZUtCneZfBaW7mDzch6o8HqI+ppqBuECyVNTMnwwPneGykIIl6KJREwQOGVqFUlD+afMWy/ULRWYviJyoj"
    "5BXyTgAqwBU/xlNJSM/ZqFSz70NluWTd/REGxx5/SZOsij8LW0VlP1BF/MnufgzxH3WCAA3vyUL3P1l8WRLkRO4kGLG74E+M"
    "528KInyky0HlZ1ftczXSv2Qb/M2Xng7n803zG0aS0kL+i1aNKg8vXa0PzoelascpiuuDUoxK+LBianf/UZal6r9J0/wlZZdZ"
    "5P9lW+mvtzgcgpYvUm6fi84acxwp/PqXNM+9T/6WhpmEZI36EP6k9Q92EC3/s0hgoeE/6zj66x0oC/35T8RS7/95uDXR4aSp"
    "Q0Z559Oif3anIy+q+bePn9s8cSW/lXr5ZAx/4fEQ9esnOuSZUn7yx3O7hPT1zJ+/oXnmT3EaFnuSmY5PqXbztxHu5m+k3Zfp"
    "VSe7bAyl/OQ/a5aN4c/PsjH8qVle7MlvnOWPFYF/JvuHs7ygBPwzunM60xSSrBxFuiTDLqFMpxK1uZwtC7ps7BGU3Pge+6rE"
    "+w3S2ckGuX9x+hdQz1bsW3czNzVywQ2FuDfur1lET6pc9KUvfV3YLw7F9wOII0K09KwcLb3M6qfKgOaelaO517NniroAsCr1"
    "mJ79wj3IM4W6LGwRdbrDlDpdYl5irZe7EVepY4JSYAv3dAu5YaQvF7lVX65GF/lg/EiVjvq5/0i1LfZRz3I9smxUwPuUl524"
    "rmfZabZMrrRP94+aj3L+hG7d6ye6dYnZ/kn9uvuf0xG5//+Tjsh9WkfkblHK//A3lPI/JKT857lS/vtPpPznsZT//m8u5b//"
    "r5fy/xJhnFxsc/iBjh2JF6pKPLQUje70JOkWDEZ5z0HsyntGyTah+DbE/tkVizGeA9YFgnqJXJzFFfVNXeFPYgFL74TBP/OV"
    "d36RpME22RkVOaJNxXWaz0lALBr7NBue7OMQUBzuaAQWgoA3vySb5IpsEJiG6dPlHIBLPmRDjCl5ccfSuoBsPYgyIA/usKBI"
    "mBcO7ONIYKnYXfi3R+dKqSaSgIcQK2W5VuJvmKO4NhZshwVimUt8CZZEdv4tjbAwKLmBhnAUYzAFBV+wstUvME1EFgjtQx3J"
    "u6WxhH4DLP95UYZIGCb8HKnpcPouki+v5UXv8aGADklkeKpEBceSGpfmFAYUi2vgu0eV1ExsjlQKf2HToD5J1nNS5ch9ztUr"
    "0p/TekXpcEB5mjQ8GhAqJhHQQlSgmOpFMdWbmZ+vSR/XeZ6pt6JQz9eF91QkIaphQUhCNevAHQpkwgExiiYZCogF0cM8/Dxs"
    "ChRvkAiO4u5HFCBE9qOPPx2j5y+JxPM1ubBflSj0yvfvTMGI466sgklCv8Tm4faiR6btiaRIJTNGbRAvcmlFkIPWQ+Ky5FfD"
    "zgf/sby8IpqHVkWf8ypbhDS2wrGCh/8sSkv0vuB2kQOsJDnfG/qSiqh2U05n+ksdq/Puuc9UicRiv/hZnP8SaYvozzRkdrQ/"
    "PXvgc8zhQwDVJ8mGj+Q2rEoqveaqT5IXFSHZpMUc/eeW5hben58pwXI60WkcGnYfopY/3BxnLkE15FfPGhTNJehVJm0uedif"
    "OAmSA+ja1q+4CBVkQsmkVIlUe6KvQSau0ge1A2TDbwTWX6OxwSf43dxUUb9PLtlReK93yiUo0LBUm5tyXVyVy6uqM1MlxhHI"
    "+zYXcz3pz6XHqN/BMNKqIx2mGlS0ubhG3kgiJdDxdAtuuiWhJJSrQrmqkqNGoQFxJc937TFWVBpR1d8GJpsqfaXJ1KL4q6KW"
    "4yQQGvSR81VRCdcu/QECZ8dfCJ1CbgDqABkeHxEovV0UDihM3nLFP6nwiCX/iaxaQilQ5PpY1pwEc3se2C4JyeZHQA06ZnK5"
    "ATpSDcl+JrpSZQk9i7GuI52/pFpXYhYX1/b/6xObG3fhv2K2vWcgQUTJeG4loRM5Tg+Ugif6GnBVkAX0yyQvD1iUqnNJy/2m"
    "Q7HBc2txv7IMJnLH6pK94+RWSQiduWR+UKnnI3/iqXNp9JzWjH6ShjGWQz1VojQUyUGV27KkUxQLS3pmDco5DfbBVolMljUx"
    "1bkUQCu+7huAbsP0MDxXnUs9Vlk5pzIayQ+6P1tAuG3H2UJuhHJJ+jvhJnnwCxHZDBqMkKUy7gzQmOzeSCPtKnFWwP0U80rv"
    "kEnx5+K7gX3BamHJbfkJ8g6QM3zxWxYQLd7m5uMTBCQuDnRLIzGzCxgwNg0H3GpZxbgZUUKt97br2tMLshH1VVkir8d44Cv6"
    "mix1bBMrJWnXAsYjNQFZk+ePGIxenr7RQOv0evOFf2dX9m/AT8UQzniHXnAL4jfop93yHwsoiqIu/gN/evrmFnJmQrLZdYGU"
    "9lq4yPb+FYm9/A1i1mHPb3Nz+j2wDC8UxNavXvF1gt2QBlUHjsEjB7LW11/ety8vi9jrIwcXeAvzr0+quEkFngUxZhnE+/mi"
    "oI4w0kAz3HiOKYlJ4ll7lh59rgANWylzTA6eQXXXj1R3yW4Tn5K6y1sFdSSrkvMsjQuYxAcUJVl8EqWoXrqb0nrVZqZituPS"
    "VWMKI2zKowosFORhcgb9UERlSv3Ae9K9WK2fE7MRYG9u0i4nYkDqy3AmFJL8eG4z6tk+C3+dQZpFBHwX6T2tgx11T80fQXQi"
    "od5XXjPELqWUCuoRw/ggESOUI58od6vVwonNE98W8r9vliCYD1cNx61f88Hb58AnAtzokYb2W6H/zGkqQsbRj5SUk2tRCMlH"
    "SqBmAYgCG8kzhojxsDQARRI4rpvAJZgYbPEiShpIn5+lR4veNaK6FhDmuKAKqiityaJ0hQrQCqmZwqpI1wkC+iR0zhOUYi6a"
    "5QVUiWBqJVHHnGwxWkH4LKmSoK5Cm1EG0q7UkIJnMaPpPXxOHK/VkjRK5og3FUPy6V3Vy+wqfhKkttUTvMGJffncenxUwbpS"
    "ld5nSkUKlQoje+QGJ3fkhuTOlPL8SXpUHQT0/LumqCcVQS6PmkHzRC4LlaA5aqrzpydpm9SYyFaWhQYqC2UBKKTSmrwmFzcq"
    "xtrGWjWRWhZKxtqGUG1HSUJFaASJgoIskIIbQqIc/Boba8liZVmQ6w9qprNmpVgRNoRGcUOowd+1WjYHjKJcvpHLZBTThVGU"
    "hLrQFOQNY61GCj9JV5k8ZlOQG0J9jfyP5NglOfq624d7xXt/pqhyWZX6If11FVUu5fWjHtSNqkB7spNcIboyqlyNLhLkeaao"
    "DVUKyV93RshJN4SfbN1VQa731+SiDFO+VmyuldfKN9V+CZLgTSivlUdyqU9yCGWh2BTKrB+nf8ZYTLkmNNfqQn3hQ1NoCjT9"
    "SRpn51quCZVRPagvTk1JkKuwwpXFTw1BrgT1CDRkGEynlnxfK9/E8CSTodIOtLMdqAvlcqZkNVOygcrFqgD/ItgsVYsbpbpR"
    "KdYaDYH8bafzlEtCI5DL6T6+LQ6lKpSDGpIFOYb8US1vMpqdxmJyXZArS9I3SPqT9JIHU40ETDUITFUITFU4TC3u/+ZIbuQk"
    "y7W89KZQCXKSYbFp+pN08mfuF7kerFXzPjRGxZJMqu4uAFlZKJfY53S5RrEmyPVitdxENaHGVmBDWGy5JsjlYqPWRACXfKlq"
    "RXljrVysNxf701zMX1orF0uljTW5WCsvAnVZaBQbZSTXBJn3o1qUN0j91ZzsC/lLa7JcLDcaa5XiRn1hhkwAxHJJKJfIFJ1l"
    "pwgQPIKdSpturjWFZnGjRv/QtHpxAyB8o3oMaH6xR7JQCWqjHGwLaDpROVSdrXwNKl8jlVcEeREbNAS53qkEFBnvZZFxGU4L"
    "QW6sNdbk5ElTKVYbcNg0BDl5bABaGcn15HlT3KisVfKgqhnkApu8EUHbwQK0NQDdCYvwb5LkssDOneOIJ5/kogGpqD5JF+lL"
    "YA9ZFnapN6C59MbvsnNpK+9ay3LHF8ObhYvhFskSXQxj3htQNu/cLc8ciND3JKuN3uZAJeUdfAAxT0ASBABQVPirSsSTuaJq"
    "wFJ1VYnF1FBU9qBKFp74LgK2CX1Q55LLqjt5plXtPbNqTp+jCqbPUcmT5zkT6RRAbhAT6YQsYrx6bXFK0pwf69HnfD0yiidK"
    "p9NEEMSQLqibzFhYUSM+weMNIYAXp5yyl8jkMePWTXehFUol1ymjU1Up43OhM1kiWlIpO5ZQ0I4qXdB7HXCMrGFETQP30zfI"
    "nUxClEi+QgVR0peZvVKi9Y1QknqO9WqKkoyryaVHtzK15NOjUkM6ZkykTsx29pGP+VYA+SOCA4vuBsZw6U28EGzZoYzn9sGQ"
    "PY/Fgk3HDymfaC7tp7cR/cZ20VGGz0Inci695m0pE/tInUuH7Fsl+Y0ytO4X9tgutBZvMb61qNA73mJ87+kmuIoChjjMBTyw"
    "eVjO+Ka78Z0MS9lO7Bl7MCBa5t14o4BjjexGyU5NxBCHHhCOODxsbqqpr6xb8J09RkxzsoFYsZwdxFnr0Xp+TWY2bAS2jSqv"
    "bQ7G8GRWQA+GZPuSyBYbC5vDPA4/XWxSXpU8t6+wyiQEHsw4K12VqykKu6pKmouGwzRblW3Fa7ic5fQ4s5tun6XHYrHo0v33"
    "mHtLpb3zHN3i1a/J4tNTbJbMb9jlNJ7ImVyGaZOf2PqrmxGOXZRhpLCU/5hdbIanojXelBtKuZzASuzAYsgJMsNFPxKF7VP8"
    "5KjS0XMGMVG/uN+/W8uQkqNKrwSXWAsYCWzIgYH2SK71tCZ6q8+gKutDVHWYqX4pquowVHUXoaqBMfFGHFUB/199kh7SwrAn"
    "6TyNXBxkYWONcnjn0i95eItmATwjJBAODnIYvzRrfLz7QU5jPVsLAbSCpW0NQGY+l9wgi7jO4PMi4soiLOgi/JIJyUrsiGDk"
    "5yR20eahAZfT+8ShffkaTz3gDPKQEdVBi+SbDizfBAohAwAdFwYQ2fcsgHAu6iJKofMG39UP5YMUcy1nl+EUuyy7wYk8aqFV"
    "qnS6jCjIcGjLqnSeYQ5R4FvYPnxTwv6RcBHWNDE18JpBbr9kecQAqNlqIXFxV37CzsLBz7CzpIb08JwZLynuZ4onaJi4YYB5"
    "7Ebt0ldVsjJlWfpCy3cMG+gB3+8xrUJeR9i1+bPnI0LLo/T2fLF7FITmkh3kkBrw3R9NzJ46l7zFokRE1l9WMCJRjSAhUprk"
    "IQHIrmEf6UZCEqUtNkg0PHEizyDCSUvkcEHWvxB5Iaxky57y6QH+tI9dU7eA9HuSzI/GRKRzyzLEmHC4gMsO7d6C+Mtx7aEL"
    "V4rEMxd8SYRPDA90bpbTYRKs9YKiAqz5ZwQbXJ9uiRiIaaf6Rd6NVku1iManurJCVT+LurenW7qPC3EuUZRcVoNFWfSbJ8gf"
    "FU00K5Qk+qhbBblUkpKFlBLXsy34kv4RNfhi936C2vsagTx8hd/MJzIbjMYDlBsh5a10m0NceIxWs2fY/THYMSSEybz+lZUv"
    "rLL5k5ioBwUJwRbZPGnUYmf2N91fC5ItktdL1pUnzOoHPyHMoqRPdi9RrTt6olJR1DwhR2tKRgBImEJekhqaBCQb/QCINUXq"
    "ROK9AM4dAsC88IHEOQUcNcdyEKleFUWFA1C6gMwLEGAiSggFl+tGAg2nZyQw5FxS/0ElZCytOSNbDCT1f/z3/0UV6VGzlQ91"
    "az0QV1N5GQdckpQR6aW2bSTW+xojGpNoT5TUVBqafQW+YjrVsqdfldQ2Sg9XSYF0Cmt9Vb6wgmnYZuR+pCXEa2cMWFbzqvoP"
    "6pzVLrKTvUqOGSdIHFm5jqrM5Ydd8sQqf3bmjn72zNUDevI9B7lXdqa+qj5JQd5ZQ7fImmb7y5Q3wiBzGycF6AEyl3oLeP2S"
    "brrIXzW/Mxscj0denHMuzAs8pvi+zDD47rPkYIu42tlddofOET2yTuXdolki69ZXovpJHjc31R7ShkR7J7qGPS0TtEYZGPVY"
    "yRO2pnsYBGJC7BoGP3exiqSysF0BAp4pBMTqHbOgQGcUt4aG3UPGFWgp224fgfLFFyCxmR1K0bE9nzkE/pKwb0nksK2c7yI1"
    "rGBGFVcjLJy6fST0XF0bYqEATSWrFtYFkhTVJQq6J1i2L6AA6QZc6QWdKq4LDkSRiO1OMLGuuEwDIbEmU+fSdtACniJMYF+3"
    "hpysAaE/T1Pn0jQuHBMkV3GiabuwxrusMqQF6lzaib97hLg6TXeBXQPH6Wr45a4dJ7u2j+jtie4vjTj97FFt3+jrGlg9qXPp"
    "hXUiSucb7QS298AGw5BuNg8pm8MUO8uftb2g9U6mqG87OobJOGA10oSl9R0vzNkFK8h6/5adtK386bnh6zbpsdvPYmOduCig"
    "dejmdWa5l/XzluUb2kuz7LMscCddmukoomshK7DF1ka65edlfY2wr64+SYe8drjI5+S+T2PNtusCp2jPdk01hRLfPcWXDGQN"
    "FYsxH11paFOKWCc3efqMJIrgbnQ8VWyiaR4qnuROLKUPFV7SCVMMeLlgYKNM4G2bLLmizVuBWRClQZp6BZaV469tXaib7FdR"
    "sbV2famKksOyPjIbNBVNfFtleJ4pqBRtx29PfHvODdVUl+ggL+YiumRxtjHGTk6uI4yd+ZMomaztwrvTPwC97yvCuyUWu+85"
    "+4wc7LpGxr0ZwQ+Nm5uCpgxwzefgB3nUAkWcIgDAJl5VhRgSFCwNWVeYgcrKyheWkymIFeH88pTHp4jExynmyoCsOgd1+mbZ"
    "rLuE+2Bbl5Me6PNZVG3e80iD73MQYlFtOTVFdF8G5Jlhw20gUbsFXyyyoR1Haj03hW6hNwQ1daI8vwZ2BBHilL4SjSNKi31V"
    "ugU3qkEydUspSZ6PHaVYk76aurU2cClyXdP0oe57X5WS9NVEs8X0ikTdnysDpjnlTQYDfaaogmmCpu3InrLQGd5XGH5kS0F6"
    "AFMrcROfZJ8OmKGP4ysmq5iS88mhd7DhAFsMNqcqfbWt66xuFQzUEBNnd2ISwBE+6XraFIfb1kiqA+wGhtHUnMpBSET6nIBG"
    "Rk50C2ewEDG6SQEpoAUW94Nxl+MQH2CFnPDFG5MKyYHvui5TB3xW5Hn+LWUagPysW9gaiRLDn3wLdgu6mLcQOVttVIjwrphc"
    "gaFNJp+rtjFFMdz6tVvoiwUVURwIthTUAIGNBDrUfxZZYrdw+SwyAqu6SGCx4ZGkyIVyvFDZzkaL9US3QjRwwjyOho7ExOxH"
    "IWnVn5+P+JBJzQgkZ+ZETs0J6cZnE/LynzMhDC7IvdZTpSuKUryJaZKx7waJ+9QOA5xkt64+6pYUwyfSAoKKospOI3SFtOAk"
    "wa6M4HRM2ybkjId9VWozfIeHEBFCeolq4BRSjPFyNEodhi1wlu0Ce5HbgTH03B/Zeh9TQXO38AtBnYtYkjcrJdFqRK2pqynj"
    "sgSBmDY9ywOvJQhrshxh8R4kLAwTALAER8WHByUUF/tOuSPsKtKUTgK4lzJ1ScZi6S4sAwA7w0J8Lc9SR9VeVIRSop+cVIx+"
    "zTuo6CdyTsmSiWZKt1AplcQlR1JJ+jrx8Bqx8tOt4Vcwz86ZfsdXfpLeSFHSYBhKtgL3kvHpAaQtX08TzX52CR1VOshMaLwC"
    "Czv8OLPDL5I7/C1nh5/85A4fJdqM1n0r4ELIm5g8mfQ4KtAMVerQLJof6VDkEzGQR1Ol6yyZw8cqZSvRxWS526gcPaFyi6BU"
    "kf2oCEfhiTmNwCXJJjyKSsCXjk4ZfQmuZA5eGjI8QLwcxLjJiTDTGIcLdwvODPuq+HSLUjFsU3oNsvvzMB6FbTPGY6XMhKpB"
    "izXLDUiKz89Bv3/q+N737/gbxPswsP+IJfcJ2OSi9YifWu63KD6I9BC07oLCfSA9PqrPz17fdvBBpFIVrJUHDYz7GxX16UmU"
    "zoMWSVeIGsD6FPecbz3k4XpVuh4b+xeWfdhub91dXB+e3xjt6m3bbrfb7XP4czbdJf/Or2fnl6djSLoY7jl+ef3Y3MaT6mDP"
    "2+2a2zfbvfZb/6A/3JeDzm7nwVjfl42Hu9rdg7lxf9a4tg70+3Z7623rEk/eLrRez7h/2zo/W62ua2N948CTa6fOi/zS7Dmd"
    "EtKGB1Pn4iHoVDVt2rk8nvS0i4tD9xr1Kxu6XpncvWp+wz2/vZ1M/VIZ75y9rpdOUffMuKoZq8jfteRK3eoeWcdn64eD9eNB"
    "r9Lurvpv57X1INi+k4+a29bxYP1osL49WA/v1kNrfWqtN4z1hrPeOFyHMvG3ynT1OnzpbtQmg4NG7bB7d3dmlToj8/XutNce"
    "h4enB+s7x8Nt8+JwYt1favW3m8urE3P9dtz2dg5Kt71jZ/32st3ubw23bq93hkM0HLbbO1vb7cPS1tG4vXXe3l29v9q+vQjP"
    "Xw5ed1987BwM29PDh/KGdnfbrlfrjn0bvjzUtXanNjw7bTc7lt0eyHev9+v1u/qwYwS4VBs/VOzTt40X/6q/8eqfHt70uuvV"
    "o85q2NuZzXb1o63j9kFgybv1HSO86N7sGfths3tutZ1X23N7u2+2ce6MbvdH3vX2RaWP9N1O/+blbP9uWj16PbrF683d9sAY"
    "H43k6zNvdW/XuCufbx3Xjhq3O/f6aW3c3tie9c/9sX3g3ByH8u398c3L6WFw4U5KXavhNZ12e+PVGob1t5G517i4mL292cb1"
    "hdfpbbf7zq3vlw9HR/fN3fb9xr4brOv7x/Vjo36wdW+Vyqdvhy46vnu93BvslE6rr2fDq/rO8cA9bbcbJr5cP8NVuXE2qVwY"
    "nn6x7V+Epm2MnMb5w8OtfmAc3HQ6N+cHWnjgnO/fjY8eTpy9C7v76r+Evt0bdKzxdsN8kOunx/71zni0fT3RNq5vb+7N2n1/"
    "XPPHTVket4979/v7l3Xj0OpuTY9W76YXey9bTte6uDs+bxyF14OraruxNbGrNXw13QsOrxsvu5XraeDeVgbIqbaP9y8n5/We"
    "dzHp700n2gk67OO2u1vdu9PbI3O9fdVt++bU3y8ZYf2keTae7l4eTix32nu52fJmB2a1bh92j5u9g/3A2r0sdytvt9vnp/5Z"
    "MOz22ntNdOJv229b9u69tXNfu+0MrdWD2+7h+VXnZeI2zmfH9cHJ3a6+0TXc607Q09f1tu3N7OZd/67e1O3wpbZb0xuvL5Pj"
    "pvZWfRucTquTfWPvqHq1vq8h6+1avr698fWHshzcHe24V+fV09Le/l77sHfiXM7a7fZ6cDSQTw9vLs+v1/Xqm7/R7Uzu0OHN"
    "a+DN7GPnrn54Ubs/um4YqKJdlY7W9X5t42XrdReXjNemN2xOQrtsNo5Ozo9feiPr7Gr99mG73dvWr4/s3YdBeeu00ZaD7bPK"
    "aXWgHx66Tlv3ds3z4AY7572z5nUYnHmN1frk4XZ166HUPC2NH+ye6Vw3Op2uZ50F98gsd45cZ7LqGVvrfe1k97ZvTHa1blM+"
    "GpbvXtart+eli9fr6dQ9v9/du9FQ96FeuWlvvHTq62erzcPaftMclLW3nZ2j/fPK5Fq/vervXhr7pcvXxsnOXn+wcTIYBm/l"
    "nZ2D9nHlyFy/3rP6kwPtpHLa7rRv1vcbqzc3g4N+v67dru42z0qdEUZnR4cbG7Ob4bB0sN/oHOwbu/KNPmk+yKuD8yPsm9b+"
    "5ar/cmlcHLnHb1O8ce0E9q5cN8ZXk97bTlU73Q/WV8fhSdUPR2e3u9Oru5Md097Sz5zZw8PAkfVBpVa/fahtXe/sXu+a4Um/"
    "uu43Tqz+9sBobI82roKyXR83mieTLnrD7Zvt9ZuDl0mv69SPRvrtW/e0HXp308uho10Zpf5e+bzhDkvO3t7FGFfv0NWeoVUf"
    "zq5LWrm3ax+sGzI2H162tjrdmzunq/XfLh+u27cPb4G8HZw7zXH59NW87uy0b49Xz/bOL3rGYN/aqOLp8emgX5GHzqx73HG0"
    "mntysD6oBtf+5HLQsFB9d3RnVoa+u4df/Jerrj88GpobW9r47uG6dxJUV827O7PaPpyVUOduz9l/eN15Pdh9MWodv95+uwjc"
    "arc3O+jJO523i0rjqr5/tnNizI5cpzF9M+5url5O8MnguNZ9sXZq195qZTv0zod32rZRubi8rB9f15r74357/aDmlQfdymHD"
    "t2rN9ZPp7v3pza3/0LZmR6Z51F6d4LBZeigF1auzW+PgbPW0Vilt1wIrvHP8oDFubLQvPPn8qL9Vr9/tNAc3le3uZPVlvz57"
    "ubDlUXe8+zatXN3f1o6C6bpvvp1r5d52t94cNUytObk5uH4Z7nrNhz3U9E56d6+h7NWPg+uz695o9NB8mVaHJ4F3bp6NX9vt"
    "s/7rlbvRPL54uD20jNvTzuh092DL8bY78o2/f+w27prnPWf9dhpcGU7XqlW7x8c1DW348sPOuNK5LL1qeyezRtC+nja7Ju5d"
    "tRur6ODovKxXyvV6+6FtmLPbU29svnWuN07OX4zSoImG59ObMg5L4ztnvXl1MVi/fzl9XZ9dorNT2Ta0bTz1TGswueoYzvnF"
    "3uxhvd5fvQi88cbJxa27HhxpG/Vdw6m+TY8G47162d4Iz/fWL18vmr4Xtk877eqk7XV2gxLer6Ddw9Dvj8zdiTcYvzaPLTk4"
    "nNxMD6rX3cq4UnJfS6XRg3vbbZ76J/vbe8cVuWMeOtqZPClVKl4HT/H97rTT72wfnRpnw3qwenwRTs8mdb/hHZSuTrydavPt"
    "dnB3elWzmsG6ufpw/HJ1evVQqw8Nu6Hd3OGXw8CoWYPG2cmgcV65vymX0dux3blpdN6m/rBxf3dRP7+3Duwd+2R/fP/Svr+w"
    "5ddDs95cbVxsVPvyXbV+rm30SvX+Ht7euXppOnijPDBG1rq2ersR7B6+uPr1VcXbn9lmRz6eDKtarXF+L5e7+/vB8LB0Ujc6"
    "455mOPjF7xwcmjsHwYZ8Oend9Ib1SveuO3ZeUKPhXZ+MDl67o932zvi6t3u61Xw9Pdwu9e60LflVbvTOXKtxUAq3NP20Wpqd"
    "T44cr17ZK128TF8c4+JtY1Q7d+/dwLv3glFlcjgyzibVw42NdvB62cCn06ZpmlOv1NGNa//W2Q1ntrneuL3YOdg6GDdvjLF3"
    "Vz3yD3umVT+4PD0J7093caN7eHS3d+ge7expnRfdm61e4tPA2RlOPXTdaR9pe3fuxc6wZM709rCybqDDq/1rPKp4t923/fux"
    "foYeOuv4xD9/uNtth6XGHnKtsHTYmQzvbib3u2d7yCr73beXaW08Pey9bpUqp/qsuzq5OXkryZ1eU7+YvVztDY6GdX/vtNTe"
    "r4e3Tmen8VLqOftvpnf1Ur062d962Drv3R3uHa9Wq9rk0Lyvu1s77dPL/mU5fHjrBY39cb1+IBvjkwffODl8O6wfXu6Fh+ed"
    "8vTi7VDrnrfP+9035+hgXd+vP1yeD7SL2sPG2cbB0eF5tXPS74S1+snIH4dv043jdgcdOMa00u2c12qr92eXWrXSeBsad1Wv"
    "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"
    "FjEBeQwCPFd5HgD3UycrNfMPXxvzzULzjyRd/EzC7EG1E/4G3Ny4BCSBUtdcMB5G5rwaje63QgVSmNTDNtH0L4iya1uI+qY4"
    "Ci9lgfq1Q+0sqdUXs0Bp9yzfp0aPxI3UvJ7+9qMzibvBO89dv8RxZBYdC8Ue8kJ8njiGYEuSgxxli2AoaLLpM8CkHjQrQWOx"
    "mWYJSz7Pmw/4nOD8HH521rmoKa55uQUyc8BKVP18FvDW7PgMHEBYoZtYeZREurpJ2QitBICAmHl199rwBMaouM0Ih07RoW12"
    "OSM8z8cUfyEn9XhhQ7kcU0vaHtHZk1pR16xMqVare3MAk/QIRozvIke9le4wC3qK1uAzEOmiK+W2JCouact0cRFL5/AbKmL2"
    "kRbv02YYENv9q3/Cdm8v3/uNdgmXTUJzz8tAL6DYDqLa0Or7AKpXFniaFQanw1E8XeIWlvdj5FzCZEggik61JLiYjWS0TP60"
    "ATVclJ1mx+DauKnby4OF5t5LtLjAvWzPS+N7HM1BBvaqkXbeSl27/tJbSRlek4MGqEA1yAuCXg1WGG9dFtsvsQP0KNGZiLA6"
    "cJed7/SRITDLEK4l/xYOSjY2vYaJ1ki9hW6XSRxV8rfwv5FQhzJp1nMLpnZW31oRk9uQu/lOb5kkxdrNriJ9VuKThdjPuUoL"
    "QVDzkURfutNaxAUR7veajEWdEwY1KRDRKG1Gfidj75XtFcwjWWKnQuqi+cTce/LAmDXFdwTJTVUHAlhU/94u2j3I0K4LYNeu"
    "pV9aJwln2b+SImbsdA9VabQW3JLd3otDS3qruskX6EKBe+ExZreZltQshGN/aDfzxauBKVsjkaCfSseDhLDMsOIT+LzFyTTH"
    "ml0/KhPK695allM4nDS/7x1aVYv+8LxIWVyUIzCqvdWwdKBwmtM+0UrXVadltpZWmbM+Eed9LGWmEfpffgUb5RJQJ2J1136t"
    "fLpY6HosWP5kNF7vq3fBRU6ddxySciBVoUwnyBqe2kuEQ3UxNAUnwVokRUaRBvNMpylO6IDgpaM43uCmYf7Z0ESvyfJu5l4x"
    "1N7tPt4F2Oddlmek8XI5B8qiScImYsedkB67YkTpS54d3+FGtkOKxAQLL0suTPrSkMvQkpZz0jDDLWJLFNtBg7GR1sP/FMxx"
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
# wording (and pt-BR) keyed by `code`; tests keep the two in step.
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
            return orca.ExecutionResult.success("Panel opened")

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
