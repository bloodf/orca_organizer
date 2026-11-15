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
import base64 as _base64
import zlib as _zlib

PANEL_HTML_SHA256 = "8d1508d79b2b478b9bc6b27602f6afa7e5676e9b193064c97c57c178fc64691a"
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
    "X8l1beR39V5bn4QPcjXMQSvLp9HTcRa9jYtxj3K1ZNxK6MSfbD7aW6kW57DZ2jbFPSdTzRjruJbi09N/3Hx6xd8JK1Nc+Ebu"
    "qgIWq+S0EXBAJ2xs6592O2meF41MGWPcSPHpP75KZ89Of5uzT6/4y3tFGMmXp48nS/j22/omfMuVDHlhA+IGHpZ/0lfMeBGy"
    "58taPz4ooDjdMlsuGX/h98rIdV81tF9qCW/G4zFuzkCExNlUP5XjSuoru5nqkxMmZ3qeYr6Ov8XSuOZGjCbwnSt0JddKS79B"
    "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"
    "AfVigab6ySQ1XFfm8HrClSDb9yMjbL7U7G2oNQYRE9NwCJsP+QSxQ8OJLr7qW8y4IWQe3r0y8V4WW+ayXNvNTPJh7uQhPVdp"
    "GoWtY3qOV3EC4ef4Hjn/sEpo3EtQkqo75rtH9RSS0e6wxUVnKhBKgT5KHvawPmRnJQpfEb7JumB3BsubThLBvk5XlMU5rbRY"
    "DFTKy7B31u1/mbrXN01BRhloyyijDDiA7XxdsNE8X5JdMDfkpFCryqGtySrGGflfJikURArsWIbbb/YpGq1SZN39mr3pMZef"
    "rQWbI92ygrWhbgs42ozkPCC0hJl5y0JlM2+pCVWI3+yJn8DWaqLMZDFXucuRvmDPH0dhLwYWUIoK7t5axp74sRGngVkt3Ksp"
    "bYdRauR1XcEN9rEXmKX6S2B5k2ZDm40Iv4/SNFpUC500FYrC4LYmPi4bFICyAKoa2JFP3h6XpBjAuWDDKwGAeVf9jOPf34Zd"
    "/gqnZoEFmkLrzqtSoaYvDZTJ36FEUZq+8hP/vR+g2bIsuTkVt3bNtr1pWojWKpeJw3/dPOkfnn+M5hpDozLC1MVG1UnyZFEn"
    "zJUvN7W0TRYlSuyFnKbaXRk7WbTmB23lTXyp4b0dmmuMGMvXt1n7AQtmdQOmfLlpwmbHAdMhp6l21wELZjtQ8O4DtkNzQsar"
    "lAnojTgsAhVuHAWrRViHbkm27KzVVGqcBit/UqVmfkqkL/GRFgftkiiXcNOeg9LbFCi6qGci+F0UVV5Bjbpar0uzcnVxp7Vd"
    "qxyabekbg79L50QNO/eOF2zu3jQar5Ierj6FU7xcj9TWjcUsPERhUBrdsKTdywKclWq+4la2tImoNlohg9BHo+kUfIZrKmdf"
    "64ihSxFRjoTN1WIWGrwpXUATIGLaCroyjWPPT0oqknagGFxpcKq1bdO41BoZrH5LUF1S2+Qi4CvnlmQEjbNEKdx6gmxPZlDq"
    "w57yLW9nbwlXqWKPTraQcvVdbSvv8ub2l3jVOvbobRupp3S3Tv6V9rfapprkUc201BbeMklbJNXglTdm1WgmvSjanuJbcnIU"
    "cN93AvE29p8+xQp27lyrqaPvYuuJw5u6w7Qp1bBzL9tNGd7NPSYMb2af6VItumWybMsCw/Xa+jQwjaRm5VoTuDGDTI7vnrMD"
    "a997aqild+tQm0mh61bbGYGN7D8dCsV361mriYBd230WYAN7TIFyuS38vy0lEau0ISdRI21ZudZEbUxnlOO7J/9j7Xvzv1p6"
    "tw614X9dt9ryPzayP/8Xiu/Ws1b8j13bnf+xgT34v1xuC/+Dh+d2EgPUPrRl5VoTFcGbqQkge/I/1r43/6uld+tQG/7Xdast"
    "/2Mj+/N/ofhuPWvF/9i13fkfG9iD/8vltvD/9vxwrNrGBHGN9BUlW5N2S3Y5Fe89ZwJvYe+5UCy/a8fazAd999rOCN7Q/nOi"
    "VMGuPWw1L3gXd58ZvJE95ka15JbZ0SaJIat6SxbDRmrnZVsTemsSxCL+e84T2creM6Vcw+4dbDNb6rrZdr7IxvafMZUqdu9p"
    "q1kju7r7vJEN7TFzdGW3zB2OyKSG/FoPENUJptC+qGyrr1GLos0EFuB7TplKa9unzh+yu22nTqW5FlPoD9HhLUcPWxHd5yBB"
    "U3hvZO/Kn7scNvxvosPOjLvTScQfkRJVc/92VHc231eK7onmndm2tYn/f0f/d2fX9vb/PxgFyqb2rTjuaD0vltsHu7syZ0sL"
    "+x+6zzszZFvz+x+m12WT91bsdrRiF8vtg91d+bClpfsP3eed+bCtGfwP0+uy6Xkrdjtak4vl9sHurnzY0uL8h+7zznzY1hz9"
    "h+l11QS8Fb+dbbrlkvvheFd+bG33/V/Q9535sr1R+A/Ve50JdiuWe1hUq2X3xfSuXLqD1fV/DQ125tZdTLJ/OCosA8/fTcXE"
    "EnsZvbDgXgje2cKJTd/RvvkH6Pfupk5s+66Gzk/Zc8iqu7/dXQtSxQgaaWl0BdA92a3QSntm+zx9aMs6hWZ2YJxP04v2Xvcf"
    "mVnaGjlrkLwLB+1j7v4du7YTY+1lwf4MnWv0Q//YrNXKKKlF8E5stbM5+nfq1G4MtbuN+VN3q8FP+yNzUgubYhW1u/DQjlbj"
    "z96XnVhnV2vwJ+1Ng3PzR2aaFgbAKmp3YZodTbyfvS87Mc2upttP2psGj+CPzDQtrHVV1O7CNDvaYz97X3Ziml3trJ+0N41u"
    "tB+ZbVoZ0nTo3YV1djad/i492omBdjeIfuI+bfE2/chs1NK6pUfxLqy0h33zd+vXTgy1j83yk/esYov8hLaglpapj2OHVJr8"
    "RGahj9Od3WxEu5oWP3aHgmLagk/DLdBICzx2YYyPjFwbr/N6FOsj+cA49CZ0HMU8cQLEyIjBsLtDI40c9ZEJoXdKd/OECU/a"
    "JUx48mdKmPCkGK3+l2oY+ZefMYz8SyWM/LfaMPJPtoSR/zYPI//ks4eRf/L7h5F/VU3vwuaASOryYV5KKVINF03zFEgs53u0"
    "TDj9L10ZMWjZG0eLJUSoTl3yy9zQ/lsfVsNuHdosutZ3YXC72aSQkOn5OAq73QMeBgx/YUAgclgJRHZoG6z4qygBcqQRJp1Q"
    "X7Foka7Z7fIKlWog45MfzgAH/lP9iBHPmbiAOq8gMnqIVA8w7QqR3e8drSGkyRWN/fQ2gwRl4sFUgFgoIvjKfhW+saBC+JH9"
    "VL+CdIVP8LeECn5SYcV5GsCL36UyEkQtlywQb/83iny18IKg8D2Yqd8DCJRc+I5x1VyCI/gMfpsjlv7ZD2cworZbJjsygUsw"
    "SKiO7/SsxksBO6kjgAx1tE7F0GeACo434gE/VAzgGRPnPJvzlG3u115Cv+azg82ZxP77giAW9prVsAYBYbPJJlJA2xAPPWPd"
    "2PL9VZToQVyI/O0yoAuMpM/ALjkck7WjUpU4K5raRIAd63vNObcGUXlk4srR5ZBfR1FAvVCCHgyK46+tj0ezJkIENNTlbeuE"
    "+/XbN2+++9YFUAzt34QYqB/1n5NGIjBqsUncUAmfyA0QePRa/1neYWpAFGJSNiDJZ30TCE7cmiYQIuOLxas5TIYrf0KrOc/X"
    "R8sxmzyYpIEcMTXsucg5Ci+zLMvVq+ft1Kvnfyb16nlRvXpRVa9++Izq1Q+KevWbVr16vkW9+i1Xr55/dvXq+e+vXn2drx2l"
    "dePZnPjhHEQI5BFIIEGKHwKeNkyUZ2zGlSaZTESD6blZAuNEw6QGLsCYzGQokpv4tvhlGpSsMboyiBaxt2KfxVOWQeaFMU/r"
    "oUBVpnUnz66SF99sdG8BKZd/4mI/E3LkSex7L3ABrWkBV9dh/vOYZXjABWnodtzj/NF2XdNWEfBE5RnhymNdK7DICOSTIErZ"
    "mwxztSaaQm8ZFl6Cg8jyR3iDwa3yqI6VzDADC9CTmjp5jUAtvlwNmeiVmnnNmMmlUmKgkcIsew+SkGfLKY9AnlAHOPcwb1bk"
    "9RENHhbbl9/F0si/S60Xqc84u9rrZ2NGSlxrzOEBy9ghp0L+LSPQzCtNFVFgvJg3/VvzXECwJOa/TOLKvQbPnaNsN+Ax33Hk"
    "/CcekONM4nKG5hXwJ5O4Qunl9TC9l7hSm2e5p6RCL3YA7LXYBOQaO0vtI5V2fKzR29mWgOVBwV1BGVhsDHCvwvHGnUwZEAFM"
    "4oo9lUsMSabCtqr8Nt9Z5YQD5ocRBIo2jqIYLV5f4cHUDiSvGbm4VdUfnxGE2MR9bmKvXy/9MKSxfbEgX6NsSucZkZlTYU/N"
    "EnR9N88y8uPcueSZXdwR+Zv6lKtFb7lOUEjj6/wQGy5vyzVJgs/YoGuSsfNjLFPEyaWbWlwHHj4zKJd2xOXCwCVrlq5rRiEp"
    "jUzNZSqyxcslXUZ4qjT+Z2DaP4SG8cIwyW/G17EBbZnkrbH+QG/tPs8EptYrZMeh7Vk4vTPiWSh0TSJXwW+NypBe6nHPSBWS"
    "iklZ6C9/x3aMbw3e60uGm7I3cTEF0I2BHOLChpGylGEFGLOpXQAZIk1+EFnWShTZ2iwpVFaLBKgLZPDQNG0+BJFoadC2pRFJ"
    "ln5ou+6WNsilizW6IxOIwse/b9oqkfXULVPSbyCh35Z2kk7wg//B7e2ozGIofyqNi279ODdN+2+Gi4kM844NcAhQFdFic1Lg"
    "bWZDMHMSsmeyLqHCBBakazJ47SYZPCJ/U3EglOs6fEgTgg3eJ0zV51+JMmvzrb2Z71b5a7GFJyueEMumlvhJlqnNEV6OuQzJ"
    "GGHOyKWL7bmEDztxRdUucUUNLnGXKXCEQkH4R97Z9zOVaYiUb6ZJLi/Ho5GZfT23YgpHA87bOe4LfpI6tBfOIO80Jj9c0NSz"
    "16k3S+zLwgdIvBVAzexpQiERYgIYB9E1jd1RRpKrmb2+WQRhYrvzNF3a9+5dX19b1/etKJ7dO+n3+/eSq5lLWDTzkz7hAchP"
    "+uTKp9dfRze22+/0OyfwzyVTPwhsluIuI2E0oYl9eekuvXTukvXEdl8OTqzHDx8/6jyyHg0eP7p/MbhvnTw8edh5ZD04eXj2"
    "oDO4bz26/+hB/vzA6t/v38/hH1j3T+6fdM6s/uDk4Uknf35w9uj+mYQ/sx6d9U/OXgz61ik8D06sk7P+/YtB3zqB+gcn1umj"
    "+yedx9ajh1iPfH7w8PEjCf/iFNGV9V2cWg8fAV6ivfyZ4SPgBb4PsXsnsj8PoXv3z+TzI0D35EzAvxj0O4O+9fDs8emLEql+"
    "FtTl2dcu4DDGJcD5bvzQWzwI3Ww0yshf584rYb9+AszwDY5+eYf17l2VkRKarpawdwpousa9QZo5rxfGT3OxVBqUhKbzJZ94"
    "LyEd5DtqXHgg4dgapYhcsBj+o8Sxq6WWX+G1G/uMN/1pCjMIXvkhZ9nfgVUfW6cnD+8D/R8+fPT44cVj6+yk//gUh+7Rw1MY"
    "p5PTU/j+4MHp4KQjWI2N18kLwYqDgXUyePxIsu5gYJ0OTh52lOfHZ49yVpesyqeCYE0xVeQz5w/BqoN+57F1f9A/ffBCcJWo"
    "SnCdKCq4UjwLrhXwgqsFasozoi7hWddeiFkjul4iXRPrJr+dXJ1EjHV/LbPu2+U2xgUuqWPbf+zPtn+XbAt5E5MSz/J37jiI"
    "kGPHYLVDQUsDmsIbll/0dxGwD6wHg8ePOw+s08HjhyfAdY8GJ2edB9bJycPHDzuDU2tw9ughez47hecHZ/3THP7UevTo/qDz"
    "wDobnDxAeHw+tU7OkOs4/Kl1etYfPHwxGFj9h/2HnUH/hfjEUeBVdRgGj3lNJxyDR7Im/vdCec/gOOZ5NdgxDo6SkrX9guHy"
    "QNbEcO0Uu3KqPENXJTSjxEmRMo/z9gU4a//FmfX4/uMHj6HD/AsnBavo0UNGqpNHrKJH91n7gsKS0AwrPjCPHvFSvPlHZx3R"
    "KQ4P/cWmTwcvSmPcNL2i6zM6Dtj0+qcyvd4wFq6bXJzD62bW3/efWf9dOEJ6Hi5XaW4JVL48nTCD7N+iYELjwiETXfgpTMPV"
    "cuKl1F5EExr8yFUxVMl64zn43bkjcR6Vg3DTPaJUPlbhz1vhEMnmA5rwygtqzwlke8J2VX9iMY3ixQU4fEXiOI1Zust1ZYrF"
    "tHiWsC63CnakKF7Ufnjm02BS+crMCRoj2uQd07pV+xlS0BEmaWapVIagAkom76Cj33Dr23IZ+HQC/c8IT+KuDuCE0iWY7Ode"
    "OAkgdmg5Cy5HyaFZaUwbIMkRYPAtjqy/WNCJD6xV0woeA5DwnFniBM0cw+BprzklTccB/tlsjNCBlNMzyFEd87dDRhybG+pT"
    "wq0vR8qQm+Zms84yjpzkg98Hvxw7oBKhReS+KcydrRgecPJXh73b/dWgpsgWroNwDvpEHb0Dx6E5PB9QtD4hZlWUBvVEOy+T"
    "6ACJ0e2mYLxGMr5OvZQaRWqY7FhLRaGKEdBqmk8w5QwDDp9omUXhrAjPgkjMBnLMxj8AGpXqFjnZF35q6MQiNc0CSEFKEmoS"
    "I3QMntNcomhaUXiBIJJRODPEZB3F/swPvQCTr9up2HpnZkYgW/O3Ufh0sUxvqwLj0rIsL55h1u9kxFI7/1o8YTmC1Ux7wPJr"
    "od9mRo6EpFVggWhoFnSUQYoQzEup4GvWVwVVw6CFoWc0kFOEOtQ6mirvREXg8oOVk9rplTpb2KdUZSqqlBbKKrbMJstbDk1V"
    "gNUdTDUQApY0cdijyJohfijJn5rqFRBRk0LIofpdICtxaUQ5x0XC64ROlQO2EI9zEdHg6od+6nts9pASXxRZQoVMypxxWRzm"
    "0cfkEDkDchm3CwX2ZnZGs0/QkVy8VTqCbZHUke9TySb84F96EsC5xHfXoXAB4Kfg1pGPrxKSmpkkiuEqMtLcbA5Kb+A4w3BV"
    "DcJFIc6Vym3eJFtdRVr4m+yo1xWPyqXqpohW0WTNhOOfh0M+urG/oFcrdOWY+jPLB3X9RwYDrFdzgJmfUA7Lp5dAv5+rW4A3"
    "eEAndPz/nnMnmrI/JrbPDvMYgCscpeWXpoSpEqguB3wpc20JvpyQUXGWziGrsbRLjv45qN6zX6THfh9E4w/lIkrK3rqE2nVF"
    "brZHMVdQq4lIkqeEt4sJ5GVXShW1SD6fJ6zfqZjqzb576UKA9H0aFwHS25TN6ectl9SLcQJ3wIhz3pRCszIcxeyZmjjvNW7/"
    "xWDuebV1sdslhE1DtkOtXLqoRLbJa60P8t/UAoaUb9kCCz9fx6P1/W2ff1ZbRk1Bq69UyUKrBWiVgFZbsi4HrTLPe1wbbElE"
    "Dt12oKxljy8OPaaeN9xlUnqAoBpJ16aFeu7b1l7dVbCdWq1w5LZWOVu2aVXe41FzNiy9MZxLdgZtFg9RQc0qoh1yWWbrrLSX"
    "gTemc7S9KShqa1VA23CR2LHt0oTg1V2a6iWL+pz0igxc6Bf3xmU4WeyzEiulbpoQD2ZtEA9meyAezPZBXCnViHiL/PLuXa/Q"
    "5Ipe4RYN3JZh0/DQTi1uQYD7KQV+4I+4s+klC7iwUrzDUSwRzIolgplagt3qYCWQPbFlsSU/LMsV/Kp4krG3pRaRgqwHqC2P"
    "MrCZNzqSS/25pS/59+18yb//M/mSf1/0JT+q+pLT2efzJaeznNvTmc6X/PstvuRYivmSf//Zfcm///19ycOZ2EFqdo8/V93J"
    "paU1CrFA9SQgN8Ea1EphZnPLBKFF4+QW92dwL+eefIpnOdcsuCmEyaZab3NSMsM2evkezY21WFFZ9UIATXNnYSZMSAGLqiTK"
    "ahxJs4zEMyf3dgLCw1k3x9glLvpR83YVR6Zc5vizoqMmx5+7jfnsEPCtwX272QZf4xkp/LuE5Q5ZgKKNMicnVbywVcQObSqJ"
    "s9lwJ7ayryURDJJe9kebjVOluWdxkG5X/rS85TK4NTwiDdvAM9JrUzjSxTMzC2fCpcuf4fT0Zo7WyhEix+vtHOybstBO/GQZ"
    "eLd2h63fvWlAb5TFPeJb9k5MAw/mnHYJZ7WKO6qayou1eoE/C3t+SheJ3RnTMKVx/vGXVZL601uMqAEHBBUAqMvugGODt0oj"
    "RW9dxQnof8vILxbQasIMZYtfVW+nCRfLVHZkXGepAedbwP0MIryq3U0i7QvqjSLty9eYRZoq2M5Lus2OGOd89Btr2DE4iGbk"
    "mqKDNLNIYxSI7bhujfyhabIx9Eczts35A1V0k9Qbf6CTrfO/Tnzk03jQGZSmsbAv9hUr1x5Y9KD3y+2yCJ56/CoBIDqOgtUi"
    "1ElA730SBatU6YIfJjRluyVAJk7RKlr+zgUrhUErfObuWp2xF4wN2Ol0ep2T5Y0yXr/1/HCCZNqDDugACUuKZlzYlszu9CtW"
    "ROwHICvtiYhendwsWhc7PejhPqyDDsY7IwtofnZU8XcVv8I4NyEibIHHOA9BWbNwj6hulBvxmkex/1sUpl6wdQI2GOu11Gkx"
    "97e13tbgq5WxHw+Lquxsg0ZD8tVd8WiYfnxQ7reYfO2YeevE2LMaztJctm2XyE3UKE4aToKT2lOM/n7tNMgRMRVqqM7++xHo"
    "/lEqUoWJnu7TIPJSFi1j7iVGLYXMTjmG1MKLZ35YGtw2Wqt+MMR9RtXy2LCobqnkk0q0wnJSU05gIsRxy87/jrJwCw6fSRLW"
    "YbFdDn7SGXl3kVoxRSu1M92vYZa2o9B+slHGg8P9dHGjvBsG26Xm/U8n6u66Vm0fn9brV3kkdHsUfckW5xf1xbZgIeraXtOe"
    "DNcO2cIyw49FzBp5DW/HBT4qHUYpam/dMZpiTmkBPVecdbaAt+xdMPsYvas7a9P3rg66pnc6cD17jAPqxWWsm/a2abS0O6fq"
    "lOJqA37A/U1v0PkLm63jKOQ4dO51TswWNrnq1rh+M6S1Xyig2Pz2FUqnJuVkMffe5hn1mHf+0jkRGz2FSPvsOrUDWKXiJ914"
    "ctQ+FSG3ov2RKV2/i/jcxP44R9vctq8cbosz6OsYvMdidtZdc+SsAlbPw8GB3wuC6Bp9gB3HORhUC4ETC5TBH8oZepF98CB8"
    "Hl2zWE4JxItkNHvh3UYrfoSHkG65hnzEWlSSA1fqycNabq0lj9ZyqFlwC+ftRD1RdzWTAIaZerGI0qjnOJcwFP4K9ssymGrb"
    "hCt4XG3jEeHuwjZ5gNEm5Z1xUB66qEK+hXdz4MiLLwvv5muwQce3hgkEknrmJ0O4rMluR9gPVYT9UEUYjlSjWcXl+Ft+wKY4"
    "HXPHbpjnXm2owX5GlNYb7u6pTFgXH5JPkazMBWoYSq1zeWkQtsLL+rdHl5RVbwcNorEX0DYwL+EeH42bQcH5voZSEzr2F17g"
    "gvc9nfo3TX1NVtMtEOyi7Pi2GR0B9Q0/dGgEXiUUJ3tDiE/ooR8+iz2cLt/4M7jGymAZM1YJ4t3sAo5HRX44e1lPx7kXTJ/e"
    "LL1w4hJ+myaKNQ45l+6Y+nClfRpEcJnYpbyQm8arcOwSrOmCwcDPZxxObQAf3ijwcFvMHVl+OA5WE5oY6Fux8EN9t9hYLbyb"
    "ps9JSpf67xAFVS52DWMy92fzADTz78JnsOY1BRCdR9cXIOSbAplSbxI1B2dV3Bib5+xylT6fbAVpH7MWwV+jR4H2NjGDkjEC"
    "Azp53xjVVUI2hn6lv678uOmas4iZ6s1auLqxUW7p7JbMWjm7JbM/kbMb9jn3FRvPlLukjsYRTHqBiZ6NNht66X71lXh2R+D2"
    "dRCy0PBx7N1afoJ/DWri7eEZNGluNqm5Drtdgzqhec58x/rEV51oxOWrdWL7RBd6Mv7SoVZAw1k6H64nUYhOdDb/NRAeQJfx"
    "8fEoy0j5Sjf4n9GMTG0/yyreaM+5e7yXpnSxTDtp1GE9pGBS6uHv9wHtCN3F+o/nIbOcAeh72hEQBOE9IECHOcElzLlt7l3R"
    "jtep0NMwO8wDDbzc0OmHRHCLOXEOBjlBFGKE0ucOwk1V78OFVkhvUkP6C0YOtYBEhJZpkkA7noNEUapJ49t1BNsBVkF+v5f7"
    "4JnZFG73Brcw5InJaOllWZbzVVDgK+40+YHeglQHfuFvZjRV7uMxyiTCs7AJBqpJu10jdmJwsE9pbNTf/yvW8A1NxrG/TKMY"
    "UbRyd80MwlFYy1Uy5z5bIYmF42EnzDu3QrkxjWKDabSD8/QL6dzF+fM8PT4W3ZffwNmTXXBcZ7b69jz9z5NhMONkMUITYoGB"
    "l99Tbzwv9GzCneUu05GZmfaWHiZaf1UfFleyragRmqat4qRHqN4hdhv1Q5KaZiYJrPjAik4qLrjTz+iCO1VccJdaF9xk1uyC"
    "u8xdcFHgflYX3GT2u7rgSiIsZsqqdjXD5eAd+zNjf+YzQ4GHp3UbwZwsQY36rHJZIjnLR5bif8XSmtNEEPyWgbJBdtaZlUaM"
    "WkJ6W0ngj6lxRnoDyVkhEJdxeHm1ZyyhvkHnVpNgmZce5uuAn69p6g7ZUjyNo4VBTRvfPxESx91s7v3bGNpv/c1zM0yNoX22"
    "GTzc3D8xjaF9EXiLJZ2YrIaje1ZKE5j+Q94feXNY0gQHda3n4qr+cMCXk5IWwV4L2hWwz1u6Ei2VFQ2V5go8R3kNPMtaTb8U"
    "GgSGEZH6hHku5Hno9EnssIpT8zz8Ij0Pj4/N+DIcOfQyHImRinGxfl90Ni/bDaJZxd1cBCji9iIXTWkucd8Hq9gdbQ9uzswg"
    "z5gZAr2H3zFpF/AnvhvmT2hI4r8XfrhKXsNJInuWO132yLfG7IHvpNkD81HD31jfxdyLbdclfvJ6Sce+F+AL/I6WSXbowAIP"
    "lN++xqO68tun4YS9Y+3m9TGk8mcIPsV/1gQd4rbJQoggEYioZXicPNJLTKeJJW16MuILhjI4ooESAgZ3KSL4SqkcgDIAizsp"
    "Os/GBjWHLFib7eIpJgQQEFaTYqgXrJOFbrkQ0/+VFyeU6TBlO8qOhdGysmMZyTp7lhNmkx2Lq4aUXXtZsa7sWkHF3rJjBXx+"
    "7ViKT8TdSmUEQmOmxXAZIlBQARzHfxWmmsga5zIOSN0swFAgyiygu84C9dLLkM8CHsHQLOZxqGRxkPO9yPs8rLzyitsPZKCx"
    "nHcZQfhThTULX/nLAgOyQchfAIv5i9WixCWs4TL3ATvVApc5rWi8y0Ph8zd4B6kwrJVhV1cNJ6TXnedhGljfKm8NhXIyI4PI"
    "x2AyNYY6i5mxvfRapdLBIMOdAwA9Pnv08PTB/ZNB3zStGAIiJ9QwCcPopbc0qLXwlkaR10Vwa5KOMtNk4a7Ekuf8QGdPb5aG"
    "e3m0ptYvkR8armtmI5e4M5dflmJLoCO6JBB7erOMaZIgjTigXB8l8EvxRgMtWEMCX/AXGli+JkvQb9izBpIJCQn4Gh81cEws"
    "SLhX+KiBw6Xb0RmhoBgzsmoFSUniYAgznp9AI0NoMvaWlI2G1qplxRQtm8a9y97lv0brzDD/cjy0yL/+9e+jzf/1r2R0b0bc"
    "f/3rqAsTX0chjaFjKyOuZsZqJrMYFPgZgt9X+BRYkQpmHVi5Wg57nvyDOZRsxznNVvgwlV0t8glxXdNKY39hmCUIzsxEYV2k"
    "QZVV9yFCsZf9rBg3TOpzSv/ow+2IWuO5Fz9Jjb5J1L4Xqix0RjOV7t6bQd6bgiTgPekNdqR4dQqrOMIWR10UdsCa39QTJd07"
    "LUL6daZfs6T0t60ejSQc5KS7968EJqnbxBXFL8gJBSKXW1KJX5ZhOtIzsWcqv5F3ledzGiR014H5KMpBxhcnBS+VjMkygPCK"
    "A9e87I/KhOBTRxWjRqmyzQYIqdCrvDbo6MWWE1P5ndOLPf9+9Pr4fFzuZg35B+3Jn1dWJD+rtxo5FuxC3JSBv8Hu0nNzi58Y"
    "F1ZeWBn3UchE15hBXZ0c2PAXfWF+voO+1kZ+bjZuzz1PnfKyJ5UpAOJ5mmhC0x/ojN4wpe9Ywfo4zYARO6lTfKmuVWwwYHdx"
    "rHKwSSQYlUY2Q8gaSM2yBD2lOljCIaWIOKtUI80YTrplvE4yFpb+GtFYT7Ze5SNXI4lr1QphRetjjMFkQZEPU26UPJZBIP3k"
    "W+9bsPIB69qhPPXgh7RL6qUFhRCM88z6DA3iESAfYnbAzb76TrrZnPb7PEYtbAQhtngstFM05jgJTeFttEoLiYss1qxByYM+"
    "BjH1RTarpc9Ci8M9+snkJz+dv4rp2C/KQOX8SeUKEjup+ug7Ye5yAHQdhkJUWCgquIHQhpOyuAQa14FGzqD/l7+89NI5bOYM"
    "iGsgKI0vUW4ZBj1Ozb9E5r0oY2mEisiLac220WgvlCe1SIeULv+SErH6SR6vlMKNNwuludnAoSsP58jipSplytQ0YhKKLRfb"
    "JKCV0/BZ2AIXkBaTm30HQQxHEWKsmIntOxYKwaAQ4wGGLQrfLplz0stolWAWEk0AbpklUd8jNPQZUrZwZmEBFSAP0BJ2mGHK"
    "A8EakPas1O7bZWWjXErNqPJspfgL6l3RfWv4B73dv/l/0NsKzSABWTTBFef10htTd7PJ3zwF5+zCm29Xi6U3Ye9F4rkyEaFJ"
    "aOdTDlWvfqxKTe9OrlIFe41XXsdeI1Yo/unHrMcG7W1C41IwG45uwWRfNNSVZIXDc/clTM3hg1co72BuZTa5L8om/gqTlI8G"
    "nFIwnTIQVpMDJTRgtw3xfQX6aTjRwD4NJzoM4bihpLYdGIUFjA+nDNWQawtw7ADK4HkRfUfbx+HQdYmCVvpaYPYDBE83tH3W"
    "9U2cI47h6Bmivse36IIQWolhnh8YcAwcgloIrh7nYvmNGW7nfOi+ZUoCDJ3hCwYCr/cYGMgnaz/hdhc8JeKllFeGbxI/kVt5"
    "ASFfGD6c6GdjsLwCXUOL4oEc9xcJramBMYP4iOimREWV6HYPqOUnfPTCGdNvLS+A4jg70jjgPyETzz/oraljVZ4YQDK0U8/6"
    "bE9UAi8O97lgBz13hnpeJLET9lLil1nfwy3DeXLtI+FwwkN/PtBbcz2GHFdP4ji6frt0bVUF0orO8/cx9T6c56Uwx1ehXK9d"
    "wRd0mrq2PzXiLwcigaaOk4QtKDXNYXo8sNPjk3Mtcav8H5HI5Ip/U8W9AXiQtUD5B/DVLOKcOGFv0BafhCRt8NmOzRvvvWvj"
    "Lyau2W9VgNterR6m6HK+zM5QL6OVzajh1YAnFI+g/ferlBosKBfWEEbXLvG0OpxX6NDX3vhDgmuUjfOv3HkCe4zQXKdfOj5X"
    "hLmAyffQ0g/fSCVQrwTDX5OWA5aCFxFOxrFT4BdSsbZD7c9hY0QTkGMrJ7AmpS9konkJKnG04uYpudkussaYLxDTcqMvsDeG"
    "n+/S2daPuVWMTVN9B8IGm3Rgt+FzB5E+SXsn5rF4hM6hgOrIGjnGpTrFW7XW6bAtXXsDAi3ZRTwGCh45Fqsv+91u+uWK0WDp"
    "FBcO4CdDZm2pnIRtNn3zi+nQdW23755XmlsqDbKZCQnSB0OjAun2XQWWeJV9kWd+2R96mPa8oVeZMhV40D/i8Q0Py+HWg2UI"
    "cgQzfHhD7Bujno87ZLK9phizrBQm2jcsT1zzLAM6LxSGN8l8C7vPnHmV3d9pXrZh9wVn96vd2H3Rkt2V4Tg+aWT2RROzX7Vm"
    "9uMBSY/LzK5ioUy5GWP2GaPA7R7MfqVldvP4ttAgY653GmYvsfrxYBdmL/aqBbNDoLFPy/F/ixbUtTXM/qukolBWiyuU/KrW"
    "9jScNFbm3TRV5t2AgqpqqK/AvN6souabgw/0VuQWKirOqSkMNaq6DDIUlb0Dud8DL4oK7gaz3Rqp+aXT73bl0xfO480m3mzC"
    "ijKflpX5sKC4xxkzw7zykpTuvBXSqD6MAKC6Bv7yfeTFk29wn3Tth5PouvjWRIHhpZ7hYnRWFBYHImsR3muB3S5cKJ/RGNPM"
    "d7v3Lv/9r0lvxP0RUxN6XDCHqQYw8zyUnlkFsoSK7Y/vRPDOzkXgjz9Ut6xFDkGzPmkycPBbRTmda/KfLHzwyOQuifzFF5Ab"
    "MNEUpcWyuYmYkQKMevycYVg2IhfE4UHfxBtGRb6oOr9JP1/ljEBxdMtNoFO0Qly6lktcUry6xbtVFNbUHBr10rqC3stoUpv8"
    "acEtIvnVP+3CpzX3J9SLx/MiJsKvRIPYuTAif9TTAD0ShXHWEWldXq1tuKDVsIDbYcZOtj8RUQrlJN+JklpG5Ld/SmX5Wc65"
    "+lQp5ZdLyaP4IunE6620QwTzF6Hib8rexPIoM3/nY+olkCja44byXZAvT7rd/LbHyUg68A/Vt3ZJXh8MCtIAbw86aS29pXqk"
    "n/vMvF+QTYa52cAVst5AHtbotaWCYcPbAgVGjqjBtMInC0lytylFT41MMnaSqp4aOCgilaEiK0fdNYrdW37InL+dAnFCdd0V"
    "+5QAu08WjvasXexE584CVD7fcZzFuQF/+psN/JkcDzYbdCSfOhExlpuNd+A4ffZCWXdgkTMikpI+8UytjjQlKeQzB2DX5EoW"
    "R1rhCXMs8BgPNZVEaiW2Mf7S73bHX3ibzRi7yZfCinNhPbp+yTCgxzZ3IJg5FV9DOPONgis6kQfXlvZ0n7xzoF/eUNRru6go"
    "9vgjDCHr/Zdj1FH841QYEIzx8cD8wpmJLcnkS8cfTnoDewU/VnYk7mRNnUjqwb55nB6Lx7wycmXKt1e1nX9nZmyMdqfbO9Q6"
    "8gKlU1USY+/Yga2FN4uUM0bBko7jnIhJSzm2UHTfVcX/sj+kkjSheVwxK6XmsQCITVvzebOh4tA47oXoLsdvamqA8T5Kf5iq"
    "dcZqmSIydvFRKQUBCeQepHJALQ6xe6lat+vaKTYvKcfw0TSemrb6cJyXgCH0wV0siWJ9SsJtgjRtEJKwhRAMGaPiRfxGGaXx"
    "v8j1E0J7js/iyTupsBgobiPFrb0nL9PQY1/eiokc2hucR186/XNkUKWmSJ4MlapZx0507LOtWRb1ehlYZbkJ0Gy5M4+PByQW"
    "u2888Igceh59Ee6LxvFxlpsh2yKh3AONN5t+fqJU3DU0Dj87NSgfbcDG79fAyLdvgqlgL1Hoit6rVsqDgTA5CE+QfHOg1cTF"
    "a8U6k7+s7DCE4l7w4BEKe/6yctookanoIlo5VGccatBshPdyNUMyE6YNmg0/ejlP5X0GT7OTJJ5z4EG2T6lBYcyIYd/2SMUx"
    "wsM0FmmN8wOsC2ZGCq8rQiuflYgBS7I8MeAbd7kh/rmar5lfJCtlXaZEXIZlojelE55gPTOJETuGr8nojBjJfK0xu6jo11aN"
    "NoQinrbOdx/dp/kNkQO57ZU7ygPH0VyjHJaHgpo23HgvHJdoJwXbC+OmGC9bo0uT2GXLqypfiDdD8UNevpAwX4o3Q/HDpoLh"
    "dKMnGe483Wwc1xVCt577ouoJDgXl2BOC3Z8a0YHjwFF9JI6YwzGHTYyIxCbAg9q5bjgnitqep/RJXxynVCSS1JLanqaNyZgp"
    "s0iVoNW6uNq+wThv6igiP3EihYB4+7WgThYOaxQeMzy2qoo1PzBNsnSMKedf0B2mqgVJ6GauyS+fGBUfeHPommSBXpjcsXSJ"
    "PqTnCyZcI0buuZO2rWymVjZnlc14ZRzzRS4iTTjpUp6PZ/nvtkyxIqt8UwK2ucQMmYUOqcnvHWvsxW0N8KvjAVmBAV6taNc6"
    "egOyghMrQ62EmZs5hsxfrfW6D92WZv+6LjJWeud4QhEBHMiV8mySWyfpTch7p7LkXpnn77vdW7Txr46dgX3wXmgCBTXmHWzS"
    "VsdOb/CXWzD3Ny6UO46nx+Rlt6sOaY0waZAXNw3y4rj1+fsNueGYrY6dSS/ZqUe7H3xTkxj+lkuOvuMXLjn6e1z1japXfYkq"
    "x6uen1Q1b++7rysdrQ9DNAQMIS9WYbOQgzDbJMqhYmOX/dFx2lyKb42OSx9tpdX6muXGSoYDIeVTxYqjvWK91NSL+xt1wyzI"
    "BZ66tS7f+9hyWRsCc6mS4rGB7uKukk8uhRRyYjvBIoxVDiH4hXbwYDoonMQIrTTfTtQvjMV9RjmsGRw6FdyVhK8m1zF5kAIF"
    "16+DlebiPAnPiygPpLGXVd/gfKwsw9xtTbgecyQwQkKdMsqLgHKbOkaoUW4BY6nbpky3DUECxHW+ND58a5Qffp3zc6uBqo7C"
    "iwD2fbkPaWVDhR7z3a6EAKf5/AO7pc7jW9YdnfBYCF86QqfFu/HtCn3hCGW5kPdwxd2SX/hJSkNazIGY74lFOKsFOONOCo5/"
    "qbKn1TiHGyxOwSqhq2WrQm+XeZGg6PbbUAodhLHgB3rbBkHuwCiKbEcOfYgBPMvIRLoGfyy6aXy121Cu4mfdknY69+o21Kv4"
    "RG+nX8kLm1OwtLWtEO9A5V3tHl4FKKynxR0ZLkbPeGtx3SRRDd+NiTknM/j3aXNzsp9qnFn+UY2OW4ExRUJTjC6d2Os83nE4"
    "I09A4327tH+ds5/ovPvXOQFJlNj/nGcZuYFsoCLJJ3ldeLrApzw1qPxwXffhTd2HpzUf8uiO35TSijIr5g8YUkqkhYWdM7xB"
    "/PPwXjz1aLL0Qsw8Ws02moH7bJ7N1STrcrLQzCSXPxoRWcd0agsbjQ8ZSLmHg0hQegSc9i1kKYUfF1GYxlFgr8NILJEYhzSO"
    "4Pow7F9E1GOG1IwalwwxJTQ1mHytPPrnyOQhMCaQvzYP82mKMJts9nhWcTrxDKk4NRZ+CClSF35YeOvd4FvvRn0bRteHSh5W"
    "bA9DzNDKqT26dld8pIYuWg39sZuH9tWmcJXRVKlcWQvhU6maE56jGMjYpYBkMZqpClL4mocphZbYT4LJICibdmIm51lkCZ+n"
    "NrX4L5k+FlLDygsaBCUaSkuZMpZLRfYJb/cWvqFDEjiu5BcEWP4TCVS+i1EEhsj2OlA8uuUeQfAdf6GDDFi58xLwSCIeD1e+"
    "xkfC1Kz8LTyRJVBhmS7KTLoKkS+BbOJnJeluxrzHzojrTzDemYWML7ZYCgNbBXa1CmxqFdjTkjwpeUkykMoyVoljlGfJBjj+"
    "Fh90MdQy968cXCsfS6s6clZ5fAT1LU59i5Obq7HWMpUUs7joU48JCbVkNGJQ7VUJf6CGvB8+M6h1lARRmhAXlUoWoH4tpQsT"
    "LnJ7C5Xjw4UXBGAGwaHOXagyUlnwLn80EpCiKAjzirgAu6w0MCqwnFo1Z6ICMiJ/8qWL9bnE5aXdkTnKCE/l3DftvxmuC0cW"
    "nDZyCaS1uQmGRnkh+EBv7UEh+bQSyV+uCqW3+rVBJbyMvy4j2gdSIfSsinKto/HXolFlweKkLUWP54tDNUL8KCO/xoamPaTa"
    "WieAU+89Cw3XG3DZOfcnExoe2hAJnLqE5ezmmEnyVDFqTyLOnzoa4HD9ZnwdG0oHgU+GbARtl+svrqmu6gXQKo6M0SSG417C"
    "7E6HdgmprMqMI9McZeYIXl3M4CdR+yRj6WuHXbM72HHgS2kAcOB1qQHEwGta/OhDX8Vp69DLIi2HvpCioDj0eBOqMPgF4CqW"
    "DYNfRKt58K9n8ufrmanIoo8sBRpF24FOtCm8A8KtT/6P6PjDiY43Kst8EiHysfnm/0ieP4rkeVriHXx5MzOz9zMrpuGExs43"
    "M7TRfjdz/mmxuLUGD2m7oEnizajLt4+uSLHF3+tSr89if5Jnc4OnHsRtDjw4iouuE7szmMa1aS8hkRavvD4bZLRKMUGXkl+v"
    "VKaQ0lYtwvrBssqW84bxOnrjKEwhuxDPeaVm1fXDnkjT199SfHtaekzpCQGsF0kxsWclha3oHa+6nLYWCe0t9bAzb1lNmSdx"
    "DaKEVhOC7oPwL6sk9ae3ovtVAEwYnMxjP/xQSG+qTVWc5+LE0bui8TSIru0Om+jnmvSGMQ08iFauTb8oiQL91eY9Lmdg1JZg"
    "AJr8pjr2LZSsT5DqjT/MMBiM3UljL0xYzhIlSSO8ZH2U74oFC63m4NZkFXvw479MUiiJaRR3LSQm0F6F30cw8t4kut652aU3"
    "9tPbTt+6n1RnM88VqaUb/8aDb2smVv+8knA6zx3bmNhylUCmOzwls7X5ZrXTK7makUYAvz7ZaJGnlOSHLbhdC13P6hXw5m7Z"
    "eNDXu/ITH6Lwr9sI6cLMwPIWRsXfJrXb1IGgujqi6TShabtKGGw9AfxwGqnJtpU5rFYPYFb+UYeVku20WKoms3kdfAUwn3Ia"
    "aPZhS//2HfBaFGuIvRvqteN+ly5VUrg3DWhxRWB53CtjrEdAeeTUUvNJ1xFOgFYptY3ksmRzivcaFBN/sSyMcF0zDHBb7clq"
    "PKZJ0oLMHHL3qSMLtp89okhrLhQFtjGc6O5HnEYS17vOpG0V3bVvbeeTFo/2U0oU33VWiXJ7TaxK4XZzq4prm+klma3VDLv2"
    "4rAF1QFs97nFSrWfWAjfmiMRehvbYf8+4nxiKN51MjXWcqcutZ1GVQzazyEsu+sEwkJ7zZ5iyXZTp4Rim3nD2KnVpKGQZaoF"
    "kRFu92nDi7WfN6xAa+5j4NvYjHXyI04djuVd505zNXfrVdvZo8Gh/fRhhXedP6zUXhOoVLTdDCpj2WYKccZqp9rRcRROvPi2"
    "zZovYPdQ7/KiOyh4slB79UgW2aoIyY5/TDUvx/jOit7Wqu7ew9bKnh6XHdQ9WcHOCp8suZ/KVy3eUunTYNxK7csZsNX8A5tr"
    "7CVpi1EQoLvPvrxk+8kny7TmWFliG2PKPn/EmZeje9eJt7WmO3ev7bTTY9J+1snyu046WXCvOVct3W7KadBtM+Nytms14VJ6"
    "k2631gKUBd8q5lcscK2xweZFrgtnDRpjzVh/ZMPPUfKTG31Roq1PZ6PGDwXbdM1BTg0V2lqrW9upW1iok0XziVzj6VqyKB+w"
    "bWtgB3ZIFlWO2Fa9dryqINrR04C1OGdgJF60H7bFDiO32G3wtp2hNICrPd1yPrLY+YhkWzeC2R14MJht58FiAzvwYDBrw4PF"
    "6rU8WAXRjpAGrC0PBrPWPKgBreXBEuzWwdvGgw3gu53qtetvc4EtjLu175rlfespdcNpX3kBL5zyaWZ/ebVu3XbTabCqpzWd"
    "2PLWd5+4Um/Qemg0ELlRvyMt8GurFFboorETwDl3D25iXKkj4IX+wmNeFssee6DFMng+36Fegh3rTKP42osnyTYPDs28gxtk"
    "O2KAZTp9a3DKUfDDXTH46gO9ncbegia1PVRmcRwtlEfFTaGgDtW5P/VV96cs9/OI6uocbK9zUK2zTc8Y5dr1bD8smntWohbz"
    "BNqXiNl/8BsFNLHXcClLG+VyGUdLcZ3y0s092ZY9eeetk7OjC/fSIUutn96S9WF1+h7aqaXcxxOvXXJYnrMlSPayCLcAGJDN"
    "CLDwgqDwPZip3wO4Te1mIwwjADLnJ+auZru1nmyuANXAuAQWB/UDc3kEhUF9C89A5yihzI2yUJciizjQ87EOhPktZib5MHO4"
    "q+HXXkJfCndD5oWY2H9fEBwwey1GwV6jcyeLvyKur4E37TRyM2wSfEQ51NdRFFAvlGAQqyrwp+IzC6Isv7JUZ0gGbSPsJkmm"
    "9Gs7GKPRK9YHhGaJ6wtNokagrYt9FxfJ6kEy7qD53QzodeXrwuauj5ZjTmC8SEqO2GLwHG4ghWP2Msuy/ALlkxnMnMOv3sN9"
    "p3tzGixpnHR6HRaq6VBclnwyc3jwpte3i/dRAPdSeRVut1v4ZPkpjb00ioHB8Y071EU55rGgMm10p1KdxeZonpsYWnEYDAs7"
    "xXFYxlEaQRVDgYMtGyTY50yS4JdZMZSgkTovIf+O6YcdOvzu/S90nMIdND+kMMg0Tm9Z6G0Re4uy+GfAkwd9mH9Tf7aSzxAH"
    "gv/OTJtepiMnJDRv/uUsl13fAioiNpaZE9+AeF2SmqmdHrtuXgOWYgFFsGsHjhsh1u5mc0DztJQYZwQDZqi0/cJdud2uETr0"
    "khMvjV7F/sKH9Xlkyji2IhVgyIIqQIvnrMVYaVG0Fp+n8zi67ry5XdKncMxguF99pVTcWayStMOBvc5SvmfxY2USWYNf1mLR"
    "wti84JPahFEEnF5JAVMRLh9mhHsHQniHBOLtQpiJxAYv6iihLnFBWPRoOHFHJGWxFvj89VJPM8O4sRFGE2/Ph6kaD61woZ9F"
    "1vantNvVZz+kFq/NAbxEBAyJkMnvfMMLuKpP03k0Sew1Yl6NIKLUpQbU4P2kZiGYA759Iq6+1t11P4JxoVcrnHFTf8bjhOMV"
    "yeFWCLYjkcKy4ar8LzP4txZLq82jh2jWXLamVgD4UisuyYvlvPBkam/QA1kmfkxRHQWtwl9CA9/Nizfk88vvuex81k52Pvsz"
    "yc5nRdn5fKbGxOay8gO9TXjoUf4Gctpeh0J6snYSIVCaYKCatNs1Yie2pn6Q0tjQxI7Q1vANTcaxv0yjGFG0cikNKeBDa7lK"
    "5pa3XAa3GFVXJkPNO/cCx1cER02dwXn6RTns93l6fFwJCQ4ynoWSW2e2+vY8/c+T4fMZJ4sRYnhNuJH/1Pv/uXuz3ta17GDw"
    "Pb+Cx7lxxDItiZos8ZSuP3mQ5UGSbXl2HJMStyRanExSlGQfAV+/92MD3a/VDXRQAQoIEDQa6Mecf/L9ksbaAydR9jn3ViqN"
    "ruQei3se1l577TUOxrGZXdJL6dF74pe89MkM3dRrSgMn/cJnVTMmz0vRMaUPaP09+NnqmxC8aRnx+RQs8OXqzfv2N7x53yI3"
    "717qzdv85ObdC2/e5t/85m3+19+8NzHPH63Y13Xs6xa+Qsv6qJePo2jBELuernXtQR15CG79xsnIBGHH/Xq8ZQwLzLCo9w17"
    "YGbC94nMC/BUUhxJDg3HQjsFsDd/Z5R3J7NqDKaxmzawk1M1f8VILupIhDr2UHRwusf8T2g+2pBkeMs6sPwsXfEsQxsEZm6r"
    "nhLiPkl4AYyD2QCiXcefjBG79WQGLzwyazjM21I0EzlxnwEsFfsMsFwU8QuAvwMT/NASP21t4lb17FXKr5tkpEjS2jPVf4FG"
    "vAV8avoZsdsjwV5WjOkHzAx2YJlP64enrXVPwMAZ2+gJwfJSsPoBwPGI84y1XZP8FVNG3LxMDACvY0bIhD8M6HD30syssfBM"
    "bFD4uoWhRN2m0H0PCMi4OWbg28F7zD99+1ZPoVJoA9jr0uokz0aZs1HmfUlHHX1i86GDiMjokiuBMz8w6KTtYhNOthbaOkPX"
    "oKjMlhF/fQAYkRohdBzBflBAtJIr/tftag0gYrC4HcFiPbpPTxGbUchojXj8g3zdJO2Rn6XikrQNWFV+4pfngXXpKbEufV21"
    "LgVrOIuBRcLCNJIXZb1/YO63XgL+gc0W/I/5QtnG9o0S17c8zzJSBTCRUX1sPhgtOF4jyY+U2cb+aKIqCutsz8jthPk1iWmE"
    "a6P0XUufepE8z7JjbFXw8eolTC7zKzMW8/l/WJ1ePDXNnC6Fj5vG8n3bJlbcUdYyfcxJXD4xO2aiJ9pzYr27Ts4Ts8css6Gm"
    "r3rfmn9u9fqpYeuHZrHrBpQ+pyTorFO1Sir7xOqkKnb9dYH4JwxTo438lHHqT1akC/076sdsVH+y9qp57KcN/HYBZrL5dbpk"
    "MQQTVxcKR7TNRiT97LA/FcwmxN3Rpj9TCvq07Do4TS384SFUBpOhMkBEU1DTMZZKmpnj6YFzPDpTkEQ9ZPI85wJDK5PPih8t"
    "vmRaXiZrbwfxE6Wx4mbSbgAiwOU/xlNRSE85qESz70NluWjbgzECxx6/p0vaxG/CVkHdD1QRf3C4H0P8R4PAQMNGsjL8TzZf"
    "FDgxUjoKRvQt+APz+ZuCCJvpelD50V37XI309xyDv/nWk+l8fmh+YiYxLeTftWtEeXjtbn1wP6xVO45RXB/UolTChw0Tu/uP"
    "iqxV/42a5q+pu84i//cdpb/e5jAIWr9JqWPO2tvUcST36+/pnnmf/JmOqYRkm/gQ/qT3D04Qqf+jSGCl4990Hf31LpSV8fwn"
    "Yqn3/zzcGhlw1NQhobzzadXfPOjAi2r66+PHDk/YyM9SL5/M4XdeD8G4fmBAriGkJ3+8tmtIX9f48Reaa/wQp2F1JInl+JRq"
    "N36OcDd+knZfp1cdHbI+EtKTf9Mq66MfX2V99EOrvDqSn1zljxWBf6T4h6u8ogT8I7pzGtUUEswURboowy6iTCdjtbmUIwu6"
    "bPQnKLmxM7YhhecN0unNBqV/sQeX0M5e6Ft3NzU1cMENlZg37o0kosdNrvrSFzZWzotN8P0Q4ohgLT0zRUsvsfuxOqC5Z6Zo"
    "7vWtuSSvAKxMPKYnc5gHeapQl4QtrE53ElOni6xLqPVyN2YqdVRQCmzhvmYqziLQlwvcqq9Xowt8MH6kSkf83H+k2hb6qKel"
    "HmkxIuB9SiuOXdfT4qRYolTcp/tH3Qclf0C37vUT3brIav+gft39j+mI3P//SUfkPq4jcrcq5X/4G0r5HyJS/otUKf/9J1L+"
    "i1DKf/83l/Lf/9dL+X8JME4qtjn5QMcOxwuVBRZaikR3ehI0EyYjvacgduk9oWQbUXwbIe/8isYYTwHrDEa9WC5O44p6hiax"
    "X3wGCe+YwT/3pHf2kCTBNukdFTiijcV1Wi5xQCwS+zQZnuzjEFAM7kgEFoyAd79Eu2SKbBCYhurTpVyAazKSIcaktLhjcV1A"
    "uh9YGZAFd1hRJEwLB/ZxJLBY7C7089G5YqqJOOAhxEpZr5X4E2sUtkaD7dBALEuBbcGayM4/0wkNg5IaaAgFMQZjUPAFSXuD"
    "DNVEpIHQPtSRvFsbS+gnYPm3RRnCYZjQc6Cmw+i7QL68nRa9x4MKGiTh6ckCERwLclibURhQLWyBnR5ZkBOxOWIp7IMug/wk"
    "mM9RlSPnOVWvSHuO6xXFwwGladKwaEBKNoqAVqIChVSvElK9ifXZiPq4TvNMvReEer7OvMciCRENC0wSykkH7lAhEQ6IUjTR"
    "UEA0iB5i4efhUCjhAQngKBx+QAFCZD/y84dj9PyeSDwb0Y3dkILQK9++UQUjhruSCiYR/RKLhdsLflJtT0UIVDJD1AbxItc2"
    "BCVIOzguS3oz9H7wHgvrGyJlSFPkd1pjq5BGdzhU8PCeeWGN3he8LlKAFSene0Nf0xDRbkoZzGCtY3U2POeZKJGY9C965pe/"
    "BNoi2jMJmR2cT9caegxzeBBA9UmwIBO/hmVBJs9c+Ulwgyq4mLBaYvBcP8+8Pz8TeqU71UgYGvocIoY/zBpnKUAr+K+WtCda"
    "CjCoRNpScJE3tSMUB5C19V9RFhpIRJKJaRLJ1lTbhkJMow9aB8CGvwFUbwRTgyz4u7srK4MBfmMH0b3eCZMgQ6JS7e6KFX5L"
    "LGzJ9lwWKEMgLW/JpzrSXwqPwbj9UaBUhwdMFKhId2GLrJNIiq+h2R48dPNcniuUuEJJxjeNROLhCq7nWBMkySSgqrcPPDZZ"
    "2CDJxKB4Q5ILYRLIDAaKvSHJmGkXz4C42WEOJlPwA0AeKrrLZgQ6b5eZYwKSt0zvT8g8IsF7wrsW0QnkmTqWucSx3J6HloMj"
    "snkBTIOKmViogopUVbCesapUQVCe+VDVkaxfVKsrsoqre/v/9YVNDbvwX7Ha7jNQILygP9ej0KnYdh90gqfaNjBVFBPIl2la"
    "GTAolZeCmpqnQbXhc331vNIChuJM5DVnx05tEtM5S8H4oFHXU7ypKy+F8XNcMfpJGIVITunLAiGhcAmi25aknIJQWMIz7VBM"
    "6XAApkp4scypIS8FH3rxNE8HbLuIT8N15KXQp40VUhojgfxg+PMkvm3Y9p7iBBgXp79jXpILfyEem05CEdJUypsBCpO+Gkmc"
    "XSksCpifIF7hHQpJ3pJ/15HHmXUkOHUvQtwBboYcr24CyeLu7j4+QTji7FAzVRwxO4MAYZNgwPW6mQ274QWl/t5wHGt2ic+h"
    "tiUK+PMMDT1J2xaFlmUgKS8cmsB2JAYg2+LyEYHJy9NXEmadPG6+sHz6YP8K3FQEwYwPyPM2w3+FcVp17zGjBDHU+X9gv56+"
    "OpmUlRAs+ljAtd06ytKjf4UjL3+FiHXI9RrMmL4JduGZDF//1c2+TpGzICHVgV/wyGCsvvHL+36vl0XuQLFRhvWw3HiS+V0i"
    "7szwIcMgPM6XGXmMFBX0wvXnkI6YRn6rz8Kjx9Sf4SQlbsnhMyjueoHiLj5s/FNUc3kvI49FWbCfhUkG4eiAvCDyT7wQtEsO"
    "U1yr2kg0TA9cvGlEYIQuedCAqfhpiJwCP1SRqUo/cJ40N1TqZ6RsANi7u2TIkQiQ2jqUCZUEL1zbhHK2R4NfJ3BmVgGui/Ae"
    "18AOhienzyC4kJT+BmsZIpcSQkXpY7N4PxIhlOGeoHS9XkeRwxO+FdLzd/MQyocphqP6r+ng7THg4wFutEA/+y0zeGYkFabi"
    "SCah5MRyEEDykZCnSQAiwIbLTCBePGwNQJEAbuum8ATG5lqsihQH0udn4dEkL42grRV8OcnInMwL2yIvXCkZ6AW3TGCVJ/sE"
    "4XwiGucRQjEVy7IKsoARtRRpY4mPGGlg8SzIAidvQZ9BAdyvUBX8Zz6h5z16jtyupbwwjpYIDxXF8fFT1U+cKnYRxI7VE3zB"
    "hd17rj8+ymBbKQvvc6koLKQipXrEKqN2xKrgzKXC8kl4lG0FqPl3VZLbRU4sjGt+rS0WuKJfG9fk5dOTsI9bjBQriFxVKXAF"
    "Dgik/La4LWZ3ivr2znYpklrg8vr2DldqBElckav6kYqcyOGKO1ykHvzVd7aj1QoiJ1Ye5MRgjWK2yO1w1ewOV4Z/t8vJEjCL"
    "QuFGLOBZzFZmkecqXI0Td/TtMq78JFwlyhg1TqxylW38f7jEIS4x0JwBPCveB3NJFguyMFiQv44ki/m0cVT8il7iyEgOojtE"
    "dkYWS8E7Av+eS3JVFhb4X2eOqUlnAX+SbZc4sTLYFrMiLPl2trZd2C7clAZ5SIIvrrBdGIv5AS7BFbhsjSvQcXR/w1wMsczV"
    "titcZSWjxtU4kv4kTJJrLZa54rjiV1aXJs+JJdjh4mpWlROLfiUADREm0ypHv7cLNyE8iXiqZACN5AAqXKGQqFlK1KwqhWyJ"
    "g/8C2MyXsjv5il7MlqtVDv/biJcp5LmqLxbiY3xbnUqJK/hlReTEEPLH5bTFqLWqq8kVTiyuSd/B6U/CSxpMVSMwVcUwVcQw"
    "VWQwtXr+a2OxmpIsltPSa1zRT0mGzSbpT0L7N54XseJvl9IyquNsXsRNd1aArMAV8jQ7Xq+aLXNiJVsq1JQyV6Y7sMOt9lzm"
    "xEK2Wq4pAJdsq8pZcWe7kK3UVsdTWy2f3y5k8/mdbTFbLqwCdYGrZqsFRSxzIhtHKSvu4PZLKcVXyue3RTFbqFa3i9mdysoK"
    "GQCIhTxXyOMlOk8uESB4BU4q6bq2XeNq2Z0y+YekVbI7AOE7pTNA86sjErmiXx6nYFtA05HGoelk49vQ+DZuvMiJq9igyomV"
    "VtEnyLiZRMYFuC04sbpd3RajN00xW6rCZVPlxOi1AWhlLFai9012p7hdTIOqmp8KbOJOAG3HK9BWBXTHrcK/gZMLHL13zgKO"
    "fJSHBqSi/CRcxt+AfcU0kUN8AS2FN/aUXQp7aa9aWjp8F94k34V7uETwLgw5b0DYvDOfPEugQd+jjDbymAN9lHdwAETdAAng"
    "/V+S4V9ZwG7MJVkFfqojCzSghiTTH7JgoqnnKMA0IT/kpeDQ5trPpKnmM22m+xw0MHsOarafl1SekwGhQUijY6qIMurV1RWJ"
    "833MR49x9fAsngiZThJBCoOHIO9SS2FJDrgEjzeY/l1dccJcwotHLVt3nZVeCJFcIWxOWSZsz5XBJGloQSa8WExA27JwSZ51"
    "wC8yRwExDbxPT8dPMkEhNPKVkuEFbZ3NK6FZ3zAhqaWYrsYIybCZVHJ0L9FKOjkqVIUzykJqhTxnT/EQOwkgfFTgviKHgbJb"
    "+lN3AYbsUMd1BmDFnsZgQYbtLQiXaCkcxU8RyaOH6DTBZSELuRRe006UgTxFXgonNK8YzSPsrPvkETuEzsITxk4WEXiHJ4wd"
    "Pc0AN1HADYelgB90GdZzvclhfMezkvYjR8YaDrGGeSc8J+BUI3lOkisTcMNhBJgdDj92d+VYLh0W5NOfAcccnx9aLeUAMb56"
    "sJ0b0cK6pYBdo8xaW4IhPF4V0IHBxb5EioWGwsYojb1P9hrXlwXXGUi0MUEB72WMjy6LpRh9XZIF1VFGozhPlZ7Ea3iapYw4"
    "cZhun4XHbDbrkOP3mPpGJaNzbc1kzW+L/NNTaJLM3teFOJpIWVyKaKNZdP/l3QDFrgowYkjKe0xuNkVTwR7vilWpUIggJXpd"
    "UdwEheGZH4jBjgh6smXh9DmBl4hP3G/fzHU4yZaFV4xKzBWEBPbjwD57xI960hJ50ycwlfkhpjpJNL8WU7UoproLMNVQn7pj"
    "hqmA+S8/CQ9xQdiTcBHHLbZiIn2bsHeXwi9paIsUATTDRfAN8lO4vqRoeLl7fkpnfUtdAGj5a/sagrx8KTh+Am+dQ+4q3kri"
    "Kxgh/MXrkZTWYaHIj0nrgrNDYi3Hj4lNxrIRrjygDPwjIaaDHnGeBvzeCAbBEwD1FgoPye8kfDAW6ipGIcsG+fKHskGCuNbz"
    "ylCMV5Y831gWtdIr0TddRxIk2LMFWbhIcIYI7K2cHnYm4fgIKAt7Glka+Ezgtl+SDGKA02SzkLh6KD/hZSH/R3hZQlV4eE7M"
    "F1f3EtUjFEzYMYA8coJ+yacsmIm6NH2l5zuKDDSfHfeQUsGfY+RY7LfrKZiQV+Kn88XqExBaCpafQmhAvjeeGn15KbirVbF4"
    "bLCuYkCg6n5EnDRNwwFQXEWeoukRKZS62iFW7kSRMsMAJa2RwflJ10L4A/ORTWvGlgeY0x5yDM0Ewu9JMD6aE5bMrSsQIsJR"
    "EpWdWP0V0ZftWCMH3hOR30zoJWAeMfwgS7OeChNgq1d0FGDLPyPX4O10i0VAVC/Vy7Jh1OuyiXU95c1NovSZ1dymZmoeyoSl"
    "eF5waAsmYc/vthVvnDWUeSYvkJ+amRHzeSFaScozDduMJ2gf0YIvVv8HaL2NAOIhF/4msvBqUAoPMG6Ak/fifY5Q5jHYzL5u"
    "DSZgwRCRI7P2Nze/0MaWT3ykHcWPCLXw2YljFitxvMnxWpFq4bJutK00QdbA/wFBFiF8kkeJ6NuRC5WIoZYRGVpN0H3AwQTy"
    "orTQ1MfFSAbg1RihE4j2fLh2MACzyscC4xIwzBzKQIRKieclBkDxCiKrgIEJ6x9kHKYVCRSclpC+4GtJ/gcZE7Gk5YRc0Rfk"
    "//Hf/xeZJzfNXjrUbfdBVE1kZQxwcVJCnBc7toFIbyPEMwZWnMjLsTRlvgE8xXiqac02pNgxik9XioF0DGltSF9oxThsU2I/"
    "UBBirVPmK215S/4HeUlb5+nFXsK3jO1HbqxUF1XG+rsuemEVPrtyxz965Wo+ufie/dT3OlVclZ8EP+2qIUdkW7W8dXobCz/x"
    "FMcVyP2xFPpJtN4jZy5wVM0ezDpD44H75pTX8gp/KXwsUwR++CzYyMQ+dg7XPaBTpI50UGlPaJpIh7WBdT7xz91dua+oI6y3"
    "E7zBntbJWIMClHYspslZ4yP0fT4icV34P/aqCgSycFoBAJ4JAISaHXM/Q1YU1Ue61Vf0K1BPtpyBAnoXX4DApgYoWdtyPeoJ"
    "+EvEsCVSwjJT8nliUUGtKa7GiOs6A4XrO5o6QlwGuoo2zeU4nBS0xXOay5mWxym+ounwnuc0orHO2RA+IjQ4QdisoheHQWxG"
    "Ji+Ffb8O/ERYwIFmjhhRA/J+liYvhVlYOSRHrsJEw3Jgjw9pY4rqy0vhIN4jffJ147XYQ24SJjuWp5CnEjlNKnbu2SdavUHu"
    "Nlg3yUuhQfsM0tmxeoHDPLTAAKSdLIPrpvC/OumLdO7X3/GKDCxbQzD3Jm2RJKxt73hlic5oRTr6y/TVeGO7Mu3Tl81q23th"
    "VcDZmA2e2Mx1w2rRciNrbZFrWgTem2sL3bJC8NROyT8KaFooBRyx7bFmphY9DVCvJj8JrzGU2HAc4AE1LceQY/ju3ZU8QVfM"
    "kWRStqIjjCxC7Wr4kU5+KwLBXjcamkkW1h9fSK7gTE1pAA32yHpJOnxcUiCRpvC1jzdYUpd138jwwjBOmQIzyva29y7lXfpX"
    "kpG5fd2TecGmRR+pZZmsTD1LpkicKp5kLdtrTD1ryczPZAdrFq+WwjpiYbEJQnZKqVOE7OUTLxi078y7PTgGbe4rzJXFdrjv"
    "KacKX9qaiue9G4APiYYbA6YEbC2X4N14XAcFmyzs7y7akrlwoyUkjOhQqNnJ5uYXWpIqfmXhcnKlx6eAfEcxvskQ7zqDdPJl"
    "WnS4mLFgmb1pH/T0TKIM77q4w/clCKeIFpwcI6h7Pv5NUd0+kJ/tjMdn6dTOAnWdm0w70x+B8jlWid8G64AAKwobWJOI0Fkb"
    "UjvjBC0IhmZKecH1kC1ly8KGoZnbQ4dgzm1VG2meuyHlhQ1Dma+mFwXi1FwaUo0odzocanNJ5gwDFGjH1owGxHA3YPqBhQQe"
    "ASytwAx3omM6puY7ticZtGFCqken3kK6DRwvOK6ysGGZ10mdKZiozkcu5sgigHt7PPS4gQ2zmBFkGzgJFKHJKY2D9AePOQKN"
    "lFZoZ85hI0JsEgNSQAs0mgflG4eBO8C2OOJhN6QDohM/dByq5vcsicv0F8jMB8FYO7M35gWKPtkRbGc0Pm0jUo7aOBOgXT66"
    "AyMLLz5TWaMKYKj+azsz4DOyQnAgWEgQuwI6ExjQ4Jmnie1M75mn1FNplXqi08NJgWPkcKOSgw0264kchWDimC0cTF3hI6sf"
    "BJqVf3w9wjsmtiKQnFgTMbYmeBifLcjLf86CULjAb1ZXFq4ISnGnhoHnfkhhJTqSq49GIoQgqag+xj4BWXsQYChF9dsB8zEA"
    "zC7pHFMrLvJkYUIRHBpBYAehEdRnBFCI4lJUQ22KHlCShwKHj5lzUXw8GFvaABGRcTvzC8aVq2iRdStE8WhAjMlbMRuxCP0X"
    "tyBLg6c1GGq6HkOxEUQMBSM7vgYphbcFoQNXx05YHfRhURNefHhkUr1Hyi9pr2wDQHdiLzuxu+k8qEIIzU+uJkqept1MJAtf"
    "TKJgKHOpnSnm8/yaOygvbExdtI2N9TRztAFW1inLb3vSDxIYMUIZ7DvxQWDOLj69cdT1+2ko8x/dQlsWmokFDXdg5UgfJ470"
    "WcqRbv/gkR6vbvSlzwSIbyEBMu2zk6/qsrBHiqheoP6QTqZAGVUWbpKEDJuckGxE46P1WkE9cgelVlFiVa6DKgxJh7OzZeE2"
    "zLasQH4pBIAT5f4dBWUhp6UR/l2E2ZiCoUYUI2C3BSGWsgMcNUGLlTcG43FtSB45rES2WhNO/YiGcrFAWEEnfp22zkw/ss/P"
    "/mDQtT332zf0FeJ06Mh7RILzBExu3nxET3XnaxDXQ7j36yd+5tUXHh/l52d3YNnoOFCG8rf7ebVUUCtF+emJF+78Ok6XsAg/"
    "N0N9+2tfcVGlJFxP9KNL0zppNPbuLq9PLm70Rum2YTUajcYF/HM+O8T/XVzPL3rdCSRdjpq2V8idGftoWho23cOOsX+z32+8"
    "DY4HoyPRbx22HvTckag/3JXvHoyd+/PqtXms3Tcae297PTR9u1T7ff3+be/ifKuUUyfazrErlrv2i/hS69utvKKOjmf25YPf"
    "KqnqrNU7m/bVy8sT51oZFHc0rTi9e1W9qnNxezudefkCOjh/zeW7SudcvyrrW4p3aIrFitk5Nc/OcyfD3NmwX2x0try3i3LO"
    "9/fvxNPavnk2zJ0Oc/vD3OIutzBzMzNX1XNVO1c9yUGdMK8427pevHR2ytPhcbV80rm7OzfzrbHxetftNyaLk+5x7uBstG9c"
    "nkzN+55aebvpXbWN3O2k4R4c52/7Z3buttdoDPZGe7fXB6ORMho1Ggd7+42T/N7ppLF30Tjcur/av71cXLwcvx6+eMg+HjVm"
    "Jw+FHfXutlEpVWzrdvHyUFEbrfLovNuotUyrMRTvXu9zlbvKqKX7KF+ePBSt7tvOi3c12Hn1uic3/U6udNraWvQP5vND7XTv"
    "rHHsm+Jh5UBfXHZumvrRota5MBv2q+U6/cM3S7+wx7dHY/d6/7I4ULTD1uDm5fzoblY6fT29RbnaYWOoT07H4vW5u9U81O8K"
    "F3tn5dPq7cG91i1PGjv788GFN7GO7ZuzhXh7f3bz0j3xL51pvmNW3ZrdaOy8mqNF5W1sNKuXl/O3N0u/vnRb/f3GwL71vMLJ"
    "+PS+dti43zly/Jx2dFY50yvHe/dmvtB9O3GUs7vXXnN4kO+WXs9HV5WDs6HTbTSqBurlzlFJrJ5Pi5e6q13ue5cLw9LHdvXi"
    "4eFWO9aPb1qtm4tjdXFsXxzdTU4f2nbz0uq8ei8Lz+oPW+Zkv2o8iJXumXd9MBnvX0/Vnevbm3ujfD+YlL1JTRQnjbP+/dFR"
    "r6KfmJ292enW3eyy+bJnd8zLu7OL6unienhValT3plapjK5mTf/kuvpyWLye+c5tcajYpcbZUW96Uem7l9NBczZV28rJADWc"
    "w1LzTmuMjVzjqtPwjJl3lNcXlXbtfDI77J1MTWfWf7nZc+fHRqlinXTOav3jI9887BU6xbfb/Yuud+6POv1Gs6a0vX3rbc86"
    "vDcP7su3rZG5dXzbObm4ar1MnerF/KwybN8dajsd3blu+X0tpzUsd27V7gZ3lZpmLV7Kh2Wt+voyPaupb6W3YXdWmh7pzdPS"
    "Ve5IVcy3a/H69sbTHgqif3d64FxdlLr55lGzcdJv2715o9HI+adDsXty07u4zmmlN2+n05reKSc3r747t87su8rJZfn+9Lqq"
    "K0X1Kn+a0wblnZe910OU119r7qg2XVgFo3ravjh76Y/N86vc7cN+o7+vXZ9ahw/Dwl632hD9/fNitzTUTk4cu6G5h8aFf4Ps"
    "i/557Xrhn7vVrcr04XZr7yFf6+YnD1bfsK+rrVbHNc/9e8UotE4de7rl6nu5gdo+vB3o00O1UxNPR4W7l1zp9iJ/+Xo9mzkX"
    "94fNG1XpPFSKN42dl1Yld75VOykf1YxhQX07ODg9uihOr7Xbq8FhTz/K916r7YPmYLjTHo78t8LBwXHjrHhq5K6b5mB6rLaL"
    "3UarcZM7qm7d3AyPB4OKert1WDvPt8ZIOT892dmZ34xG+eOjauv4SD8Ub7Rp7UHcGl6cIs8wj3pb3ktPvzx1zt5maOfa9q1D"
    "saJPrqb9t4OS2j3yc1uTRbvkLcbnt4ezq7v2gWHtaef2/OFhaIvasFiu3D6U964PDq8PjUV7UMp51bY52B/q1f3xzpVfsCqT"
    "aq097ShvqHGzn7s5fpn2O3bldKzdvnW6jYV7N+uNbPVKzw+ahYuqM8rbzeblBJXulKumrpYezq/zaqF/aB3ndBEZDy97e63O"
    "zZ3dUQdvvYfrxu3Dmy/u+xd2bVLovhrXrYPG7dnWefPisq8Pj8ydEpqddYeDojiy552zlq2WnfZxbljyr71pb1g1lcrh+M4o"
    "jjyniV68l6uONzodGTt76uTu4brf9ktbxt2dUWqczPNK665pHz28HrweH77o5ZZXabxd+k6p058f98WD1ttlsXpVOTo/aOvz"
    "U8euzt70u5urlzZqD8/KnRfzoHztbhX3F+7F6E7d14uXvV7l7LpcO5oMGrnjslsYdoonVc8s13Lt2eF99+bWe2iY81PDOG1s"
    "TdGiln/I+6Wr81v9+HyrWy7m98u+ubizPb86qe40Ll3x4nSwV6ncHdSGN8X9znTr5agyf7m0xHFncvg2K17d35ZP/VnOM94u"
    "1EJ/v1OpjauGWpveHF+/jA7d2kNTqbnt/t3rQnQrZ/71+XV/PH6ovcxKo7bvXhjnk9dG43zweuXs1M4uH25PTP222xp3D4/3"
    "bHe/Jd54R2dO9a520bdztzP/Src7ZrnUOTsrq8qOJz4cTIqtXv5VbbbnVb9xPat1DNS/alS3lOPTi4JWLFQqjYeGbsxvu+7E"
    "eGtd77QvXvT8sKaMLmY3BbTIT+7sXO3qcpi7f+m+5uY95bwrWrq6j2auYQ6nVy3dvrhszh9ylcHWpe9OdtqXt07OP1V3Koe6"
    "XXqbnQ4nzUrB2llcNHO918ua5y4a3VajNG24rUM/j46KyuHJwhuMjcOpO5y81s5M0T+Z3syOS9ed4qSYd17z+fGDc9updb32"
    "0X7zrCi2jBNbPRen+WLRbaEZuj+ctQat/dOufj6q+Ftnl4vZ+bTiVd3j/FXbPSjV3m6Hd92rslnzc8bWw9nLVffqoVwZ6VZV"
    "vblDLye+XjaH1fP2sHpRvL8pFJS3M6t1U229zbxR9f7usnJxbx5bB1b7aHL/0ri/tMTXE6NS26pe7pQG4l2pcqHu9POVQRPt"
    "H1y91Gy0UxjqYzOnbt3u+IcnL452fVV0j+aW0RLPpqOSWq5e3IuFztGRPzrJtyt6a9JXdRu9eK3jE+Pg2N8Re9P+TX9UKXbu"
    "OhP7RalW3ev2+Pi1Mz5sHEyu+4fdvdpr92Q/379T98RXsdo/d8zqcX6xp2rdUn5+MT213Uqxmb98mb3Y+uXbzrh84dw7vnvv"
    "+uPi9GSsn09LJzs7Df+1V0XdWc0wjJmbb2n6tXdrHy7mlpGr3l4eHO8dT2o3+sS9K516J33DrBz3uu3FffcQVTsnp3fNE+f0"
    "oKm2XjR3vtVDXd8+GM1c5brVOFWbd87lwShvzLXGqJjTlZOro2s0Lrq3nbej+4l2rjy0cqjtXTzcHTYW+WpTccxF/qQ1Hd3d"
    "TO8Pz5uKWfA6by+z8mR20n/dyxe72ryzNb1pv+XFVr+mXc5frprD01HFa3bzjaPK4tZuHVRf8n376M1wr15KV+2jvYe9i/7d"
    "SfNsq1RSpyfGfcXZO2h0e4NeYfHw1verR5NK5VjUJ+0HT2+fvJ1UTnrNxclFqzC7fDtROxeNi0HnzT49zmlHlYfexVC9LD/s"
    "nO8cn55clFrtQWtRrrTH3mTxNts5a7SUY1ufFTuti3J56/68p5aK1beRfldyy5XcrH3cPbkq7IyO55VOuXB63Zhrd5fnR72H"
    "i/vrJuo1S68nitXqvg4PVK2nH+43hou7l9F4flWwrqe9hVU7uW51tJuB+eCd2frl8GR403TG3dej1mBLRDvjG/tVLVvdvd5R"
    "7uh2kW9dX83Kb6WDttZ8US/Nm9G4YN64+fZWt3d2vNA17+S0KO5cTjr+5PX8/nZ/517JXVSKN532w11t5011S6Pr60X1sL/V"
    "sf2H4t1be47cvlc6HzRtf7Jnj0uLsTJ93doS7Upj32zrN6/6bNqtnAwv9nIvjfZprX+kvDSPF42iOTsTzWHxulVtvVi3OVfP"
    "nb9N7nLw8mjU67LwEBPeHKs6oiqXK6IbIrMxiWwlZP+DIPpLO2OSJ+HmJvntIHeqezfRYGjAFnZ8PqHLqKl6oIdC2C24pkDf"
    "7I1nnugiRuwzkvxheOff43c+1lpvZ+78GGvStLyxZo7ukUf03ROJwAyJ8gOIknbA+cYK+nHmI8taMSeGt+iFn3Ta8At+W0ZW"
    "+cTqn1OtkfXrLNjIAQ1OyRFerD4WhmnCADxJ6pKS3ASLBzarmXiwh1qsISce61PRtSe6VVF1FvK0VxzMrqB8whHsGVM8k2Wi"
    "bCbLhP8H7Ami8RKosgHjQNhgn6ktU9U2XJno9DEdN5xEfi+FdyIIjO5zHwZDVgiLKKM8/aw9ViDeXLCVdANg76ggPNpUwCAn"
    "ixq2ZPJZkqRr5miX8J/w9zlVxIgkpTPTKU8tcOyVaDNmdJDMC9hrjJFuRYQLCp/5jIV+/LMs9NhY4/YtsfLUQ5sdQK4c32Pw"
    "3gbbClwTvIEyUde48BMnJJ3FUq5WlLKqqpjFghb18JhFDW68xao3EnNRf09X5XHC0tS7I/O/oi2iaA+jvBbVtV57IAfeHDNO"
    "HfjVG1uOByfSm587Vl9HhqTAGQDnldiloCVQiwzJTZ7WQejzpp2Zw97Rs+TxRAt6KbyDzRDb4mw2O4jYb2BXQuu9/8T8/RBB"
    "KrX628CyG2cKmpAyiFotYA9ig9JsmStkC2OxqIC9OvxHzCzpl5/M2KZf4+20HPjP305rDHLeDLBxLCtitsrBf9RelHxBT7EM"
    "jn6N0zKgFu5opS347w0cW+1lQi8Dcim7g02L8V9qdFPLFkOjm1q2vFJJLGWrK7VK2fLHtVhXYjm9TilbXO0pn6wSGxyp8kSt"
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
