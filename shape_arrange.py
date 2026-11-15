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

PANEL_HTML_SHA256 = "da63896b20611ab8ebe7b70562b3046359993e2e8f9e2b4058d54cd9587a3407"
_PANEL_HTML_Z = (
    "eNq0vVuX2zayBvq+f4WV48VNTKPllpPJzqaM5kraTuJJfJm0c3E02hKbgrphU6AGhPoSkf/9rKoCQFDdzsyctc6D3SII4o5C"
    "oeqrqmejVV3au618dGU31el/PYM/j6pCX4rPpP4MEmSxOv2vZxtpi0flVWEaacVnO7s+/uozn6yLjRSfXSt5s62N/exRWWsr"
    "tRWf3aiVvRIrea1KeYwP/JHSyqqiOm7KopJiAoVYZSt5+sZcFlr9Ic2zJ5TwX48ePWtKo7b2EbRQfLapV7tKfvaoNHXT1EZd"
    "Kn263unSqlo/kqlk+0raR1a8ufggSzsujSysTPWuqth0XZsU3upH9fqRHDfbStl0yZeM2Zmei8nUSLsz+pEUp/KR0o9sd12Y"
    "R1bsO67FbM6NSJk43Xdc4Y/RhBcC8o5hTM7qlfzapidMCDGZTJJkkDyh5JMkSQfpT9np5OnTtj1IfPa//8N4TYU3tjC2+VXZ"
    "q3RZ65+3q8LKbMl44/tYNI261LwUqeQWGoh9FHKs9Erevlmnlk316TE2CfpcylTzCet45UvYmtrWML7jq6J5c6Pfmnorjb3j"
    "O19mNS6LqsIHvhJfG1PcjVWDf/ka2nmbSujiclZjkY9eFdv5km8fenUu7XzJNw+9el5YOV/yK3gH7YF5EmLpJ3jJLw/eNNYo"
    "fbnki8P0u81FXS359UE61bNMktFI8jt4mV6nkrXtVSoZS5KrVI7tldTuZ1nY8orxi/sDZetzrJrfQiEXbngYP/fdGjc40l/x"
    "4wnjZw91lsqcL/kNvL1MJUsSORJi+bp4vUwSOTuZw9PxMkmWy6MtbLuX2qaST3CRSS6lkOmSf5R33Mg1/Fusa4N/Ia3Wv+h6"
    "Jb+R69rIV/VOW5+ED3I1zEEry6fR02EWvYmLcY9ytWTcSujEn2w+2lupFqew2do2xT0nU80Y6/g78eT4HzdPLvkLYWWK697I"
    "bVWUMn3HaRvgcE7Y2NY/b7fSnBWNTBljXEvx5B/fpLOvj3+fsyeX/Pm9ErTky+PHkyV8+2N9E77lRoa8sP1w+w7LP+orZlyF"
    "7Pmy1o/3BuhNt8yWS8bf+J0ycp1XDe2WQsKb8XiMWzOQIHEy1c/kuJL60l5N9dERkzM9TzFfxz9iaVxzI0YT+M4VupJrpaXf"
    "nphnX9Z6rS53prioZDY64VLvNtI9TfiNURZ/G35dVDuZ6Y51vI5nC1fWt1Vd2FT6eXqkmtfF69SyXGa2402cH1Zq/nq3uZAm"
    "lSx7Xbx++KNS8koipSxl2wq3Cy+r+qKo3l2p5tlyt8z7x8xlaGS1xlfwwyfeKL2qbzCZfmaD4qKisn03DQfCDk8EtU5XsL1d"
    "+yHDp6cBMxkBs8GVuEwNyzcyNSzbwf9TtU4V85/jKaGYncm5UDM579w42E6t00ukK0BdmD9X8DxZSfFkmuaj2f+l87/8g8GS"
    "XUvxJEtn/zc/Yk/4Voonn6V5Nvu/z/7xj3n7j3/M/m/O/vJZ+9+Y9t9R2n/Tj/YfT/7xl9n/zf+S/+Mv/3jy5LLv/iY6EPdd"
    "ON7CxthKfu+IefKXJcuXywwoGB6QK8nG69q8KMorWPswmq5QOGMoz1qyqXajeDpJEtjcs5P52Bq1Sdlc6NnEP7CuY9x2oY1X"
    "URuXy6kbOGaFnMqqkY/85P3rKYOSZnrOpgYacCTM0fLRknW+lOtBKTBzEvccZdaY2Q+RdY3tm3npV9JI+ukEujatpN2XVdE0"
    "meWNvatggwkZykmS0WVqGR37kA1aaYFqYRJ+IXZIBTktjwUSdWXlpinrreRFVdU3611VNaWRUvOVamA/b1Vpd0Yq7X74dCM3"
    "tYXpvbsoyo98XZuNrq+LSiFxV82m2HJdExfFozfwRaN0pbTkRharWld3S9YvpWvsP3VrNJJtK+EoW/YDdCeJZOEg+ekZCWHd"
    "Tzdqo8mUZmt0EnahEicwZ6qfVXV0xIw4gwlVc27hPx0mx/SVXgwrbdQfEquEH4cVEtOyNvUmtYwroeXNo5+Vtl/hi9T4doZm"
    "WWQTaXVJcTwJLwpxMi2e+Q+mxdERg4WhZsU8Sc5karmZFdBitpeimF4YWXwEiiCfnfSNAnohJo5ijE76Pt2GPlG7YbtxJZCN"
    "U+vUtK1q29SILaVvoTeQGMgMDGUeRiaDutapO0A+yrsGmJMwPXG6ZffmynfZ0H7Z03Qd8ompYbwQ9n4qksskGRVtO1JJAn9w"
    "Ug1MqsEh8lV52klsFbFK7reN9uG59Ccj2+u2FTOYxlewqOnvHGZ7pngxFxprh0YhJS7wlw3DpMaX0lI9NkkKfLLIWE3VuIFX"
    "eHzjT8slw2VUC+PGNZx545WsJPAFkNn9tozXfZvPBqsU6vO9PnFrc0NzCSszrPK2VTlOpYSmvVMbmWJb+6dsNMlSIxb08cIv"
    "hByrgDcrerMKb7C8MIR30hdxTRmvQ8aRgRnLR5Ms5L6VLHtwcoCBA8p1g0zCaJRKuGksFtcL1fwk10KI0QnjX3smN5eZFAKI"
    "Jxwzq3BKIkH0bLUQ4qJtR1dREmP5DYxy/nUqx8jLsOxv529ej+kOoNbAD72T/GnUTP5Oer7sBmYlf4e98Z/Dnsr3s+WrYps+"
    "3hPZ6Nhyns2ADxtLbY2STcrmYyNXu1KmqeQzy/WcGyZOUzl7gZudHS0fidPlXGguGd93rMu2ruRzaR8qGevHgjfFFk7VF9Az"
    "1mUwiTmUyrJrPDdGK/pzBi/CkGeWv6CeieWSiVNYAvnyHC896eO9HK8k3ZpVrfPcdmyZSf5c8jdS4CkETGNjza60tUkl8Jh7"
    "e6Wa8UraorySKyE5Pi+K0qprKUYn7rnWwv2U67UsbQP3YnwuK1no3bZPWKjmbbFr5EqMJi7lpjD6jf5pp/sCF9eL5qPaQsJI"
    "JslzmSTpczmmevMUM20LA6KE565ReK0VkAtPyKZt+99iNmdsvN01V/gpO56wLB125YHGTBjrLoGwY5bUn3OP4g+7LXQmxT0c"
    "p7uRi3pLm1pyO/VZqW2eaYqS3OWCzhwJQys8GzWVz+xUHh0xPZPzsau86/PFk3D4Sfwu/rozstltHuhEkgw7cb9Tk/8fOuVb"
    "00VF+B79x2V0ZqcdizacHmItn8upNXd+Yp9LrI3jkCpdVNXdHtK6rqt1yvZHR2G5C4H8LK5DI6/PoYuwFF0RrKvX63hAa316"
    "kiTHx1EBJ/j6OZJl+MR92peHnK7jM55L7PNUTh1T09fqPt/HaQcFEa8hRZSl6w4af12r1aOTrmtsvX1wxA72C8685RrbZcUJ"
    "1w8uPvtMT+3h4rPzMVZDE/nAZzi1oUhPRB4s07+c2XnKwjI8+MSTp3h1yk+uTqpaHlYnQ7tHJ2waf+wrAXZuNCCZbgsRrUqS"
    "kRzUTcm+lC2OiCRxD7wPKyzONuup3RykPY7w9amsi8mjn9bo+vsxomXPZc+QfI2nIJL9A4L7XPYDilRUsgxEJXjAf+AfJLLN"
    "v8ri47m0/NWDx4lbQGvtT5GV3DaueX3Cu0JVg8R1VVw24q/0oOWtHbx1rRqkNTDuu0oan3q/M36x9SdC1leWiOOnLND1Pr0V"
    "X37Rk8oo/5df+Iny33/5V/5BEmMJhSdJ+kEGJhBSqKXWqMtLaUBe1enaAq8yKPhpkozigj9/yto2ev6qbV+6ApHO0U00/mIS"
    "+Fo3+KlbuK5HT/l736I/3A+i6OIDt+I7Of1AFPE7OOxjUhmKC4TyG1/QByEhv+WDIfncUZaesPgWBumX2xSwDKZyCtQK5vu5"
    "3LKfgQ+a9qvmweXyfjC2tT639dZtP3pI2bBJT7suzMBwPvMPclysVvHSCKsqHz6m7r3Z6Zfr5wouODgZ4WH/i/SrwGdMia9Y"
    "uQxuSH2+ruOv4Wx7K/m3st+1L/vdKf30fcVtntIwiW8hu5As8wlvJX8LCf0G/xEW7mt5dNQn/eSO/ePj1/LU30OBin4bKNW3"
    "dPR8648IPIPo+KSapnKwMWUY4P/leHhiMXR8ve2/fUtJb0Oxdhq4BioWFspwy0cl27B+YFXafi91KJ1PLdujeLGzQnd4zWb2"
    "ytQ3jyKC9wdKL8LFXkhae3ZqXRNg7dnxtTSNqrU4nnCLB+bXSEh+VPqjAOHr1lEWSOCHCSISan3TC7XwfPErmBtBx+jUTP2o"
    "G6zpudxOTWiAEMeTPDVCCJATaSEZ/xnEj/xX+J9lVhhuDus3B03mhwl+dI2QnXT7i/eNE7rvwC//zoDhtEEjXLNR9OJ+ty29"
    "KuvNdmfheEy/l+kwjflMD3zP+ityEDqNF7iR+lZ+74VyfiV+AXTUP0y+ZG2b9ot08j9cjklI/EsY598kZRqm/yb5SI5Vc37+"
    "E9xlXXlPv0qSdERDASe8a0/bjnC4mG/zNGzbp9No+LgWH7ghWisjQouLc+oFqmudyvGCLqjT1EZL4qRt36Sah7d0Y3Y1TZ5+"
    "Fd4IzcN3R0d+n+C5DPsiesdloOofhIY2GY5rN9rbn3f9gP/ckyYQfK7kNtMcOcvdRWY4LAz4pUAECrKqJEnN2KUKxR1Xurtw"
    "K5FxlSSpCqmGy5Db59DjZnfRgFgGdgI9GD4ySaL7hcT2/UNo+F+n/XkTvf7EqcNHJ6wbAed2fKzHTQnlb4qt++PPdA2SsojQ"
    "/uo3+t5t4sxyVyoJgi1wDL4iEBD43d53EPvlUy0PzfI5kPfC5cJ/kGI270+KfwJN/0ESd/OdZLioJn3r/gbvaQAgF3Kd34GU"
    "Rriy2zYiku9DXxyzlVk3j/Ie++WL/TD9EKg6UOeITfggZNdh23+DQ+73B3lFKImO2d1FxC56nsIv/rBke+4QB8tfbMKDXz39"
    "lSd+uEcKu47//c94WL9ofNN8exwDWtwjrb4vQ353UwwZ14/y7oCRFfflIZ01RfnRax0+tO3oO9m2H9wtsF/7juYQoTloFp6s"
    "0WxbaNtIiA/sXlZk7H+X6QfumDvcKHlqwwB/6I+x/mc/F1GisCyjJxEn8scg2ApqHTs47nCfhPmOBpv3B064UvmUab+dQjt5"
    "+IVFut+hnZLxB7tkD7bdv+6jeyD5ceoeJOuVgJ5T8aupJ7u/yaMjt5TpNiDDvUCyPXBvuJ0O+GVYV45yuaXNJCT6QhgiPHYX"
    "eKb6tLAff0IpSXQ7fNyfnvBBU8J5gN+7w7Rn/uJTG9cUHn/41e6iYXsbDryTiOjah2gt1tr5084XMYXDX+KB5netdpTRb2kJ"
    "KrLwARAXoC3ShhspaAGsFU4KCkp5PXgy8VM/DMpG0vnvZJJ8CP22TkUwtW0rrVcKCKdqYE54T7lAIwCoBswFinvI9XfJuMHt"
    "b7nBba8hgXY2iw7Wwnq1Ble8oAbUUQOABNRsD+vG8UM4hA2q5VHe3rPFkBmWEIedvwTKbZasDjrchvXiJkUqgkKoJLmBLpC+"
    "BnbkkkQdS7/nnMLfsGlfVArDJk73afxB28KTsW07WqQApNGnAqT7oHvpnDq2uVHAkqQw65421XiNdmce5q/dwDJe9I8GtKeW"
    "7cuikctitVpmKo9eh2azLPVpFj5ZpzIuFNQWJC2bYkl0vi8z0K39J9810i6zYR74rOtwu4X5rW2v5X4VwSxgimTbpgp0TKD1"
    "NYWVS24s4z/AJyy32XfwI/8V/7deYfCbTd9jBpZR2m/W/3of68sa2+ttcalj9QdVRUxASXov/wXV/ZtNqfr3FpUPLMMfuAEr"
    "K/aLBQKjFosM1Dp8RttsTHXUZt5ffXcWr778IAfAEEr3CvQgvKx1WViAw/StrykDG0cv/XDAMs5xkDP8PqhuDiteujdLqDKV"
    "s8lchIpnkzmj6uW1NHeDgdiG7+HVksNudQdFYS53G6ltwzq+VpWV5uEv6R19irAL1/ao4wdl6dWnStKrUE7/+b2vX4KU8NNF"
    "4Os/6Ype/Vg09tPfw9t/pxmQ7180JWT5dHM8BOXBIujlJz9Wuqx2K9kMF9SV/9y/XnKJmQk2+am8+JayfqiVTh9YoS6945Xv"
    "1yfLi3JQmbgqHurkpth+soPI2/svLv0X23q7hJdwPRjU3ufYNVdUrdNwOqiaz7jxGek11h8y/6Qur+y/+ALz+M+aK7W2D7QT"
    "06GlTb2RD3ce3nyy9w7V+nAX6SV10tY/wf5t5Cq9P2nxS8h6XhsrVw9Nb/QOM2INq09Qq4P3Hd9pGoeHm+veUnu9nvgeFaMX"
    "y0O6GUPf7BBGggcBV8LM7BwOfEKsEkTDEPM1oiMH+KDC03CvfX9XXM7hkEewzKOXjm4H7mCI+HRPl9K+9YDZN+u0YKyHoKG0"
    "D7CBK5sW+JQyrhlXIyHWdmbnSfIQ5hFouuRuOLI9HLe+CcXY3YT80foAfjiX4wugqgXLZNdxlIccloKJ/34hHRyfxXiBHaKe"
    "8OKge4vQP385KPoTdzWksHK8qrVEmRQJc2yAO3iI2No6MFOAI/fTvn2QkaTJb0QdTzQvRT2zc1gApR91z6OU42K7reAILMKk"
    "Nf74R9azEnrqikubvBK+BQiI8J3RtMpwfTrOghtY2VlACz5NkvTf+NrSd4yY7p0o6V3NK276FgK8RaU7lu368d3YHjFEPC+N"
    "RiHUYDRqoXkD6k6XnBZ52giP8gL5G68H7eQqtBQUeA1CC4TrJgiu4tZjsmFcYS/67n+eJOkniz0cABUNQSkUKEJroL7RCORY"
    "UcmyMoJYHpACZECVTc2QDZzS6ACBANoUIE6pwgt628Lf0YQlyT9tCvBOluMf8YqeeP8ly1QEnXTVAzZjDyIrjhfcvi2gbfXL"
    "rQdWAQPNQYLFDQEjLQAjA6vJA8CILxaq+WUnl4xf0+UMUDc9DYogaa+LjWxSIm1s7Jk1cYqw+3CiLB0OHwZfmiXzjBp9N5Pz"
    "8OmCRRfJO2S1FylJlEUAIbGp5/tJ7dYz45YvrwrkOvg98Jwk9vrCfgKx48SwHqfxk8NsBuSOas6vEDsqbHcp4zsuXgu9oGkZ"
    "wMlB9uSvtQclcyUOSp7GhWFGlCb5IkfmfgaHK/UYt4MMrtzwXg3em6J/A1fM1OQq/8lmP9pM5S9t9q1l7rrctg+fQIhbe/iV"
    "Blyak2QSxnPl792ObqAktAARhaiQVIaRC60cTmFo7B1RzFr8JNeVqxvn43dLeDi6eKcI/rq2DqeYLYDawloCRB58sLyUdgkk"
    "tMd61vDh7zatg2ysgJu8ZXmd1XRsTHtYKMDs8u/pluRPouu0ZrmB1Jpl3+D/ddfxW7fuHslbK/WqeXRhHwCNNbutNOloAnSp"
    "kfcIrSReY4XX4xsHYB0drCLf8O9sqijDDxblBqPv6G+qxKtUMa6B0DDGR0WS/A6Zk2T0O8oH/ETARdqdmprx0Ykb9sJbK1gW"
    "IMbZjk6jJkxK334/LT1ZBYn5q9TAJR/odQ56GKgfxTYoA8APFcsoBeQSuN8YbzoSLcRGG1796dpgaKBUaMoDXwSqkSTa1+tk"
    "Fj1PDOdLh3jXvg5fpk/2BS0QVxhWm19iSI8s47qrb/QPhBaOxQfRkYF3bi9vyaxl3NfVf9p1/Pw/WUkn0UoKcPOTB4cwvOz4"
    "GZH9W8tv3AFg+TufBhge/sKiMR1/jn+jfTgkDz0xfxMoput8OKLxvHULPMjsTXHDC1qmtQBWl5eoZgniBgLMH4g9kqTmFeUD"
    "APYSnndoSUK18JXQ+Qub2fw3m723XhNqk0TZtIimosq1xRlo0iEnvmN8T8wnKnbIBEhyYDFBu7NznHdgd/PDLJlLKPPZKpV4"
    "zK9QbMLmiN31+bouFmd+jKVO8chFfC702q1gwBnLXliZ03JGuEVU6Nc2Xtd7kg3u48PKTYTCvYrzodkUqALs14LhsClHRuGu"
    "0T8VdErvr4omqzvxHMgLb4SF0Zdu9NU6rYkhUz1w/VGTGi+mHOQo7uXA+89IgKzVf9JxgIcAODj1nHfcEz/hElv+CtDZ/ZzD"
    "dDtgMac9HEMx/VgYHAQckn7v05BYJGHEB/ptHz0pRhJklWt/IulALOiXiuQyEdlHfhWumL4RtXgFW+JgPAedqw96VvRCZsJv"
    "O17Y8AavMikdg12wampANZ7vi9Uq+2hTpMCMN9LiE1Boxmm1YYJbeIzjisMkJyjvMiijN3FyXBs3sCjcSOLtAXXFeHWgo0qy"
    "XGX9aQsDRE2GpQeqe4lLcPACWRVcLAcvFJz7GqFJBeMFaBToUClgYSGLB1Z5FjnmvW1bPDPb9jv6k7qzMjDZ1Adc3grWXFbg"
    "IjdArrwoAGwc6rYVXkYcpVNBjSiirKGjvhmczsWGzkXjzkXJISmjFOwC5qUeBNOJ/YBFpoZqbKjBhpLFjrsOQe0qamiUzhzz"
    "ZnLTJzq+jtfC9sYa0YFaICfuT1MZJExgxEELJIgHfANBKeasjQhB6rJFhl90RtOq8kUO/sAh2zE+I9rfy3R6EfWhHD3sCSNO"
    "93pm5uINDCsK16C0QCo/DEilI5z+OkeSAXGqHmDd85HM1AMcez5I9nx6bvtU4M5zk8Us7s5xSQqMh0yuM6oZ7zWvLFLw7IMF"
    "9nEE5uivh0knrONvo6QTyvXtUOH3cvj44/Dxp8Fjf7r/geeT00JJp0yik3OZ4QMKWJaZm84JKXteFVv3+lz6jK7s6AnfuQ+f"
    "TldyXewq6xNOogPtm/iU/M4xnT/DZI0m/MzyV5Z/G+tyfonz+3zvLH9t+cs43/f3853wG8vfWv5jnO/nSFZEKAOyhZeecCMq"
    "NTbmcQhxnwVui23bWzy/AA6vUReVjK1e3ZZUkS6z8C8LZ1X1h03P8aoMRyhC5gdfNziNb019C8wfZHiamyw2wiIi1AC/3YNz"
    "7o8vJPnesezQVsl1ry/hOxtbPR5kxp0RZf7hk5n9TafP+8+4aTJHcBu1KosBPK960hiK6/mCRzaHUz2L1Hd/i9sA14vooh8k"
    "qgcTlSQfBxlh6zlJ43tkl/Ha+A1dG/lvfZK7Sfbb6vdhrw6NwAZd+3uc9zEt5miAHg+louGmLG8eSY2vSP2v/xRVBFAi0sP3"
    "8BpqT2yA1ItKgnGQKW5+IelrLjM8ZCj9OiSi9vWhMlDc8ggzHlgSAXbBqf7j4uC+4/LLIRfoW8Ejpi5UhGe+pDNfsqkU2jf1"
    "DY6Ah6r3fZGDTuhhJ6hxHqYeaxJeRxPl5mHsbgnkIIWY8cxdmMTpQGKTy+x1eij4wItx+ASt6SKRwbSvrL/k5/3VHhZodv/a"
    "bgat1rFi7tdo+XgqYjUtIfOJJYSM1dCQoZHWSuPBaW4cDwwb3HqLEer3V91/YhEx+ZKeDhGqx5NPmEqQzQNx4YP6nYRw5NqP"
    "8Fahe4OEAV6/hYoHJgZfsST54AxWPHl2VglIMkYng5U/BPr7he/n9ns/QgT6GQC/Ikjwn2yUaEYc4J4eUOrQr4NCO+mzA60+"
    "Mlz5RsDZnxshwQQWrWw52DY3EuQf8uaR45XcOqk1OAJq9IDBKLWHP4YKK+0wslyLkiSu2m/rRnuskG3bRjuskAXJOKDSyOAm"
    "4hDApA3kSoJ4ub3abORKgQOggq+k3GY1r3Ups4YHM4Ws4sXuEuTYf6svsh0HLjhbd0KRM546l5mjGzWK8+nvSb6CqiYsg7+M"
    "b/glX/Brfgf9uBBkQ06bP71ExZajAfxOkNYw+zV+i8bpd2D4i1KCPL2AtX8Ho4sqXnH6KzWCPua+TCdp3/vaeuk0yTLVmpAo"
    "Pn3rGIqrKG2dr1M0AwYEDmPZpcB51vk6h0r8ywydOal1uiCtBCLuFhFg7m8ImHP6MD0ttdjEhjFUzed8dj2Hqq77D0sNxhCA"
    "h06S2m+ES24FjvhJPnlyktVT6jIMNxwHBIm7FWCyxc/J0dTG2c7x2yS5HXtLzTK99WZNfENoryZJtK9HT7VIvbuZqLnS+Zbx"
    "jTxHNCB8cyYucvKB4IWjqN+o0lqzrNbknggGauOB8EmSbsYO++4UxmGJb8juBWHDbVu37V3bphe5pYl3l3k4oc5Qhp6BHOKM"
    "BfF1qaf9ZCTJImUcB57eziw/E0LU2ouILpLkDFRQmDabZ2f8ej49E5bD7Gj+Odz+tFO8R9MjOU0uodFck3tZwi5JdukN4xuC"
    "+sn0kvFNZG5W4UKq0hvgWbIb8jKF+3404RvGF2LjLJFoIqnxbvNvaGxw1NYM1tAXDhN74HgC5HKN9jfWDchSdV7kNyDQzM78"
    "QGdVXqU3pJFG9BWRYpa59/yczH/Fhv5Szg2kk4kbFIQ/ojew7sQ5P4+01ETVJk9OvBbpGRgkDG4MdB1IwS2D98qQ6qCSOWGn"
    "wg6VJrr3s2CPj0nuzqAmR1vQz8KDXmAMuAzp/YWYoyP8Dh1LxB9tsWoYbiYHXmxW7lhgXZ/5DNEPBy4vBsX27+r1owc1jHRl"
    "BxkVu+czbOvyvGxeBB9N3q8a6BbiqrrgMyhMwVr3+pVoY5tckrQY6V2w9diEDvYlbKMSPNF0fFdUeCQZuUMGTI1dqUCTfD7H"
    "ccA1HbQOAx9LCq0UBt5S5MBbCh110BxwlELV9vVGLd7QAWi4gvOjCId4LXSux9fghYwwh3tpTG2+L/QKzsCGoB0/6ytMWL2A"
    "ly9hklY7LDgrOzCt0qB3PgNXhbd2TI602hYlvYGeaWdfi/KnLfCOvBbLK2u3TfbkyfVOfmjGtbl8gvUfG7mWRupSPvl/zE5b"
    "tZHHj/emW5LRV2z4JkusJklGcCELrtTAwqhAoD7YADuiz2JTrBMwUw4Wy+R7RoPmXXLFa0Y6eocRBmstaj5MUOPU72udNhxH"
    "bXLCZ/jZnFTtHlvcXcG4G15zxcvoVnale79kJzAnE1xIylvcTYGHris5xtHwOuxL9J64QLu6a/x9p9EBCL8AU4pbLd6aeqMa"
    "cEzV1NU1HoGUo2erznR/OzrXbXsbgD4yt+izDxlKD84hg04JaJVQxE1UxEIfAYN2GUaSJuiZngaVwpE+PT2dgBMwELWBOkGD"
    "hrJ4Jtu2IIMk5e13cyvM0STTwvTw/1DvO++YILJNC97IXiG/BS2ZhcYcT+bTkW7bPv9TMCo9hcya5Zc62GZfau/S8UaDkBT0"
    "Z7w3CuMv9ABmDo97GD1xq2nMXutoep+HhuJevtMgc1ArMke8C1VdwNhBTVnoTtum16FVUQMOTzZcvTJeveEzYAem2N5eCdcv"
    "N5gv2gVT86yfNePdf2maJbendBg2tG+ABLUCWM94p1ZtawPsHCRc8A6PJvBZqvROTvtRNXyC+tncHE+yhea6NzRMdWRFzuGk"
    "DS9Rdh+ZmMeaMX8dIacQ174nnv+5hs6eIKIAvEdFdBS8ZKln4QP0kbUnAnutZ2o+TYuxCrY+OSl2C5aNirEC70cK+v5Ms/xa"
    "z8zR0VwUmbMcKxybqwR4p/EAGsnm46Y21vNsuFCPX8EyQw4mtEQYfqdj4+6TqXymgreOoyN25+YYNJuBxkD+Oy0UEoHphX52"
    "F7p2Eby6SXGnZxd6Po0sPCOTzqe9oeJXbQt0I3rXxVSGrEW+hmtgT1M+kCQJJuJrzfYRAXRUuh9Ox2/6wbkGHiNlHAok07c3"
    "Og3GcfxjeNA84rmxemrJK03eXWF3oS+ksNPz40kGNwR4FwlhcGt6o6AFDNlCRxthEQ3ZpZ4tYMhiY9ivImvR+4OIp78cg4Wm"
    "yid/zSZf9CP5xcCI9inIhnx/cDv+G80YfN/RQRC+EScwXDKQfJ6GV23b748kwSGg0Xvrsn57eEq8jEj823BCvAWO/1tNMlRk"
    "xRYLPF9frtoWC4qo9VtkdN9qx+eOLLK3eiiQNoKAbuJ0b8aLVZJcFunxxEPoXiItLsS6cK3nNa6RWkj6bDCCNFoh71SeFlN5"
    "fMw2RcqmL+HM4aGOSWAM667XwC3QoZIZL0r3d4V/I38AmhSFap2+BegUumQ96ND7In2L+k0JV7sGfKvNp8NNreNNDaM8K3jN"
    "G14KCz4e5XwK+CggKACfFHvH02QF35E72azoyEsaGOetdFqDlhFJw36lTFZwpRtbgETDuyuteV2tUHyZ9cDvrOGbeqXWSpom"
    "Kzv2ELf8k76HRkKoM8gmL3qXuEsmxwtsZ+RwMspEfnCXgW9LEuvyT8k8bVEVVjYWHbS4gnTvdBJLCl5ygz2d+2ja1x3K8UQE"
    "cU+W+9qCMbBXpsAcgUYTGgS/w1SV4mRaPgurqfQbshLFrJxP67wa+yEV9aycO1nuoKVJ0meKzsq0Ek2673jF9wez0gNzK2gN"
    "nMS7JEmR3dzqdMc1/4rP5FhWvOKKW8dvxrzJHz0xflv48X5bAM99rVay4QafnH+d8DO8n/buEqKvDpDpBoC54HnRxnq4eyJC"
    "8W2Rem+LC+UX0ELlCzVelO6+4KvIQNnsnPDgzmpbMy5lbuh+El0xQMl/mBYVc9gjj0ckozznbNZfkmC3gSbNg1aD99UAkQQX"
    "1hZMxkhnD5Xj/QV4YqSiv2gPSIK1s7w+bsrbJePfa5RafKPTX3SEiPpZDy3DkKQg7Yzt4fUBbOq7cA3tlWr/Wqi5rnbNVVaS"
    "cLMC4eWOVp5CTBR5ktRJArj15bYGM6Q1jMYvBa6fUqALcF0G+4TvQSS1hm22uIG7rHRXRSJzYbuOVv4D8vTe2yugRAT8Ezih"
    "CfwkuYrhktinrXhbTHc42iLoQvDKvSXpAWTaAAOC7cNm57tIrIS2nGMltvy9gq+SBGxdm63UDUDmsbPYK5C/AYWPP/bO320O"
    "niHxytExvhv3wmAsX8dOI75gfBM/P4U6U2RKtsAtcmwNYx02/Uo4cfQuXNV/AXq/ztdEw69YtoJFxxi/itSieog+d9b4cJFW"
    "5EsaPFs5+6zleMnyf2pEmKCA1szkPHN3OgMAx2KKq7oAD5fkcxSv/KAi8y46fyqcbqER32lA17jPWYQur9OBqvifOgZMWB8b"
    "YBxcAgc5HpwTD5+KSeK87wnwfiznQY/lttvfdLA/Xlwj+Oi9YwIBwP5OVhJCJ/DfomyVLK7l2cWS8d+jVKmtNJDab86/66Bx"
    "2avG+XcH/9+q+VEW10pf0oOTM7jnil798roGauNkdmHRPzcp9hocobgSwS0B419HL/oC6R2pBB9rMfvWtY2jbHnOpQGGYCWd"
    "d05ebLcAvfqmrsG3BN+CwqeBVvuUWpPX+RfQ3ewxJEQ/v15baQ5fnQHvUFVyRWn0PQyApIToJ35/+OrB77+mhtJX8W8o4f7L"
    "uIyOWyP2EBsjW35TNPKdKXSjKJQBSAObTBpQhO7A5G/fVLVtMtv5ZQiHEDcC5vZwFSqEMiGyJEkqk4anFIW/ZNHt72C5NqkC"
    "9F6zu3hnpMy/T1l0toyK2IdETZgqmqqmE4gwN+OwjgLy3KSFt0Rp8DcUVd5/XYnCpCWvuYHtVyL/PRJiVyRJCS8qzzmE9iVJ"
    "Y9LwhAWDRD7+cnRXpDtesiTZxVkDCN6kO18jnAfwiAZQEAJiZ4+VXiZJ1JRgkND3E+irHBdhlQivLhrkmXCQH3yoL8I1i7Qh"
    "TkPTiyGiU2WQH6CUxByD5T7QW82JeOOZFL0kPcCjuE185/2YcBpv7KDSx/XODjuYo1e64s51xZ9Obi0pkxqg6mrmjFZ26OZm"
    "LnZczn7Tc2o3rC16dDyqa1FFRcvVYZuGL1znU/avvuuy8BPHYJckXskarOcGmn4ToWRmJ/Np5ID8NPI6R2FfUFwazfzeCu3c"
    "ZvaiO8QFGGFN5CbChBNif0A3OyHB3wMZ/2PRsRfn1DwUD4OT7oOyg/Q7vq4VZnB32TtSqThuyiKilTUswSGhbAKdLIdksrpP"
    "JXcHRHIdaOR2SCI39ynk1SGBvOzp4+KAPF4/QB3vOmH5RTCTwhXHb2Etohvl88DRyCTZkjj6f9GK+Owg0I2dTebTcxebhvgJ"
    "Z78PZyutg2cC7NZAXRalYELHb9y5NBhXftEPqYdWG9EgkdP9gejE30Zctm1D90cveoedkiT4B92I0sXtdnYxd4ZEdwVCgnvL"
    "U1lB5ikFGXALeof7WtI9jzJ0IMZIWXeeGg5CU/R2QK1U6xQqAOF0TNONKHkhKl6L3Sc7sGhbyHTdtpDvrm13g+4QCm80mdrZ"
    "7xC3RZzuwfFyA2TyPJV5nRXYGH4z2NVJMnwGVTcW4K+z2L6VwMRYiTlhU5OfYQ/5ag4oArCmxyJsf6UfLB4UmkI5MOq/06hz"
    "PeBRAp1P2fQ8XWOL3cRg14C4aXG6L9o2LVCGAqqIVOdX2Ya6Z2MCeDtTc1IEOJIGCa5PtcCsB33CLyTf5mfpFrpWz1lWQ9fK"
    "qtYB/6VERAF6jOPhYXQTpcBA909QonqY4t9EyXn84OMF6OjjAen+VzPbcdWr0G8i3yfGaRXuTIzoEDcF6QrKK1WtjHQivUiE"
    "0/gPR4Mv34OCGKksBdmir4HJ6Z8AlkiOy+rNttZS2x5JEpI824CWH81VsZXfVsVlZrkvhVy4kTYQvR5Nvgy3a3fQ2OTzp3Dn"
    "0Z4BC83UPUsWyRpKE6QcchzqTL7EroS27sGfj2cVhQ3OAe83vTTpe+0ONEaME5hnAKH0l9q4nsnTr6LYLe5uGek7wVOear6/"
    "W5kC9kuux1upV0pffmMKXV5l2mkk6RFku9ShsV+/rlo5bpozCgE36InP17+GFdA03xZVBQFLPpXZv497FWfltCKEEC+TZBUv"
    "BN/dGxWl4l7U2Hgcqn5+KhPQW57oz8A07+TP9NneXRMosxuhnUS/BqrkYxFoduR+YbLLUmQ1ka46ND9P6/EWRBR+vpJEHR1x"
    "I4z3e1MZsDLyPbG8ASdAqQUPSj1bg9IeuJQ3vjXZTZHWfA9takBkCnT/dMKGd1kTS3gN+Y53bRHHTx+IwLIzwYnYva22M+kD"
    "S5bgL6G//qAbTFqS3J+ugHneGQyPQBxiKGvAx0dEZDX0xoNQsCalO5kcw58OomjhJSyT3QBNve4NP1AgGBTOqbwPHBir1Vsj"
    "1+q2bZfXS3a0PF4egSilmZ3M3Y/J/Ogoi4PmbHH08KWYHWR+Oj86gkL4CT+ZR4gIE0kqgm/UNNXiQTzKcxcIwsGiAU6rx3HI"
    "MFLRX5mHDSUujcNfFLz22n7CechDAy3KmiTpCnTUgEjIwNkZfBmwHTBd1wb1A+DOrIgo018nTyEKi9POFCDn+onQACtIv7eM"
    "+pQk6Vv50HIbcDBxlV/k74s0+oRlBQqoRZ3DlskavlfZmptsCxz+hgAjhi/EemzkGh3/5e7nvsvoF78W6zEupnMLcZXuxKv0"
    "GiIIXmN2lUGIlI1JFxzI9S69A173NgSPA3MLfOuUrJsR7t4k2YyE2OLwL4DE88t0w9hitpk7FWe6AZ3LtU/oQU2/A+KsBwfe"
    "phsuxx8ZQvgcnhrvI3L8MUnSxUyOP7oi0JnyVbplbK3TLV/zyVM+q/hiHjmQg2hwW4Av/G7TLXFgbas9F+OvzHK8DqK1/CLd"
    "svx6tp1nC/jvNt2yth3J8cd861QR1Ag0BWErNAEv0a4sgoJpxnQvNWzQIpHIXeRfkkEFYtbMOdSJw7OdC0gc9GDWzKe36dYP"
    "y9b7XTkYEzgmsGibp1hwNSi2YiyD1f9ASYejWzGCbVae9tE4pYxfmd5ArZtaEMQeTyA14OTeK+R/qSULg3Z66QBhsIiuw1cm"
    "eDBEZ5jBm3hcEeiUUjY28p872diXq0qeudOW44sSr2xx+hRIxrWhcD9ysGd/rIuVNPzOkGTTvVLND1Juv67UdWQtcuEo2blB"
    "k/nlQJFwO3i5Onh77thj4TVFhkT7qxLMAntRre5xV+hwVDXPyciniDynRiCpgJWlCbox6JBFMx6xR5TVBekAts0lMGBSvQoH"
    "tTwsSc5AsgJ8PHC6wr+OZuvMHKopsVK47ZIc78aky91myTGOYoremxQKsotgHxAhnPpxodCRETRXoYYWoCmg/wLEOYjPF4sb"
    "eQWDhhBCcUpQaMr/UwGUphBb8EujoU39ZcT7Zil6NXSuxt61VcEy5bElvCDx9ztcEqmbNXG6B+UB+llptsu2xaYTUlecOudY"
    "oDbiL4x4Z9LlxWbJ+HP6DT/fuOTdkvGP9Bt+fu2TIc8Hlw6/X9HvZrtk/DX9NvZyyfhb/1DGAvVviQ+E9QUzIEvnwwx68tKI"
    "ZTg0miX/0YjlShmJnHGz7Av5ach7fG/Sl2CtCVYgAJrFwv4wB0o4vRo05BsT2bGguiQUM4FCMtm2f0QM2S9x/u9N+iP6Tept"
    "A/0KgcttUHgq8Va37duCwls6Uq1w905dzLKX3g2K+K1IiSMjaBVFNCP/Bi/QOBx+GZm+oNhg/jrkNNc/G4QBtS0sQxwHSInY"
    "KUr2nE2RJGA5WsROyIejik2Y2XnbyhlUiT989fNYN3m41XiBRzqA++rg7qVu28sAZNWiThI0OMDBIv9QQOWNcF6jGoHmV1yS"
    "VynATxxg6eNgoxDTOUSufFZj9EoFcXBtavImd049IYglutjEXxmGIg2A2yJJCkgOF6HIRZlGNydLujOTz5Oxal5qK8kNcdtC"
    "CEJE5+L5F9rKDiOixi1DpOS96odxNfGUv+f3k6k46qLkCEKB3S3vFxgdyVoMoxVOfTv1vTE1OKY9HPBZ3UMCa0GIQLCThrGs"
    "56gSwArNPKD/cTh6nXmKdtUKMM29xtqtuOEg2QdikEJQ4ylyKKYHqxPq0cSoRzkDSo7Xjzl63BqvtZO34XI2/hWMQe7JojvV"
    "DAQJwKTI3DzF4aIPIMBqBrkeAsZ8O3QQBwdEnoNp01s9hkC9b3UAWVybNDyxJAm/x2WIGVJ4BlVH11pU1Wu6axbg7Ywr8WAA"
    "ytMg0gdI5mdOYPIZKqExorxl/MeU8T/Sl4Q1mP2SLkFNBtQYEA0pY3Ou8uOn2ZdfsM5fw+18WidJPV6U4K0H4VATr6vqcVY/"
    "uoOuQmwWkcE6SX4waY1+hhqBfWrbom1Rm/ZR3k0rAW3BnoHZzWiRNixvsuXi8d52S3aUQpxLky8X6wsMDN1xjNaZQ/gYOHMV"
    "mvFCVLH8yy8ygMMFxP7DWLCyx4KFsCYOPHbYxxM/3dCGauwQbgDbGcOYndNzI2bh3dHyuFnOGa8idX18fvTmWqPrAgnIqL+6"
    "7wpAxvVil9EPAwkcQ4PLXRWV/U8z8ODSHYYX1TObJE8wkveTMaChIPrystYZYugzBfiduQAbibBhSbNO3Gb+TYFOiuH/7G89"
    "N8bIROC9EYfucejqs3+coXOgx7LCHw6jIyv+eFXYgtLgF3/slLaQgD/548Ja41LwJ39MalwXtLy2DX8MF8PMhSRfN/wxNQtS"
    "4lbyx6au+1R4YPzxVd1Y+raEFm6Ue4Jf/HGNIR6xcKXgHHq8rk3pUGuUce1ZYlArOnUi6/hjsBt9p8qPlAsw4WeapNXS4YT4"
    "Y8TKQIYf/CsAlfyGaBOg5Bi/jEKhgGEyShrO4fILV1v08/G7IU89+0UmOwdB1A85oKOwifvS3mZOJY4X6ExxnIPC6ctrXpSl"
    "bJozCLoGdh53W5mVvGcfAC5EEs3ZyRy4y8cBBNQ44txD2tjQBcSjSRbgVXqOPh8ePfVJRUj6wieZkPS5T6pnuocT/WaGroLg"
    "PJ1wLNtnKWgEd2lxmPEpL+KMu7Q+zPE5r+McxhdlDjN+wU3I+BgOGEx1YMKVeA9v+Zpv8eSKXewtaXkv0UWPX+DkMQk86nue"
    "KV2LcrxYlE3zCgNLYyC0tVjjue7NJaf/ThPBZEtUXqJGJs+961e+S7fRh1sYa/SC41aX15fiilHxKio4LKw6Cs39m8Ehz1MY"
    "ZYG3rUz55il8o/o3u9RveBSvk8nh8vESzj0KKDhhCj05TLK0Dt+RZ6b9IqMmybhJerCSDTZQxQu5X/K4yJuu46WDcvZyPzMr"
    "geGlhpdhydP+K1nb/gYibvgBC4j+Kvf3vXE/ik8OOLwH1Ofh7FazEhSPA9+8dwfQP1SIO44AXMGQg5sl66216QNn7Ie0OJPj"
    "xTgal5mdixPe+8V7oLoYCvD3+PByuuA+dK+loL3WycwoVq9zYvCYIpL7gqTqhShEWLWQAbUmx6W9nT7GK4Edk6b4DE+UJNEq"
    "HSZxyZcXcKULK7PgPpZGVvONtFf1qskaTsS25A7+mVVcaTiswIgbS1plax5Bl7Mt9wDrDY/RytllwFsveBB3ZHd81Qs/sgv3"
    "yXPZWFPfZbe+CFKWZud8RW/kKjvjO2+flt1wI/VKmkxK9+sdWPbLVWb7BHQjIVfZO46GYGfFFmLUr7IXvJHmWhoQlEvorZZc"
    "3m7rRmbPudJX0ij7NR6oRvL+ip0piG/vL9jZG+dmv8kK2Qnr8EFWpTuw1scjven1GsBaNMEAH2CzV+j6EK+hwjoFLSlECp+v"
    "8I64NJtSIOaUWADxDXrlhMy4ZHg9rKn2JdRwjy3Eryn6xnBIXKoLtKxXKUrqIfFSxi/w8MtG1MYriicCuZqQi2WmY9OHvGPf"
    "pYpLvpfBujQbnfBY2A/P6KqDidPC7TuoToZHITvsXDnsVsmMSkvokkJhViS+lOIqrVheuRFjWTW97wcyKAusON3/AfIktPfq"
    "WLfGLbOGXTKQe3wkVnFFAxd9DlHk/JyxDNwTpcNJ/Ji+MHzL+Mf0uQFr7I/pG8Mv4e9Hwxfw98LwO/h7a/gF/P3W8Bfw963h"
    "VsKP14a/g79fG34Ofz8YfgN/X0GMMzj3ntOV9/nAegtg/LSaV20L7O3zgcnxwzMGsr69nxOUicCEWPoNis14MrsQQWVYUydR"
    "EknbD6QvuF7dowTphfT3NID3RhtNGJAgk41Ov+GEAjcytOjDvhNvwC1GkqDSqmfqrXISJQOzRXHcRaXQm1MPvUL+3otqgd1Q"
    "U4g3D/y9u/Ut0UnYNzo1KC4AJbbxKnQ8g+NX9MQ4uojJPzWy+t/dC2q4F1S/FzK40YvYJFkrf85tderOGOe2wtueOv6ZZQcJ"
    "B0B4ow5lv+Y+EhreurVB5o2xdMoSOQMVya8auZbAe6JxNybKsDum92Q2q3uW8VGjIpGMAR8aHmsNPXY/A0mys5AGLTK+RSh9"
    "jEZPRQcrXR75fqNulQZ/f85DLPj8syFZcXfJIUapcJOY7Sn5lTSX8tyC58hLJcF7Zydi7SwHnR8i8xgvIzfposlGKiC3wZhg"
    "ZPISUOUlCEP6VyoenUKlJZe8hgXJOD5Z0G9yOiQKAvXxkvHI/3qheqtpFzkx9M13uYCDDERTkLfg6OOIq7h2i7VTQVB7bxgK"
    "m8sytC82SYIe+Yg4LJnfcrWaKZC5grhTzacSYEgglIAf3AJqKYP/e3ERerxRghiWRjlWdK043Dvxh2dcVqpnZ1aKx6xPtlOB"
    "ddmpAe+yU4F5CW8c97JTgX0JrzyX0ud1bMpORXwKfKmjYntuB7MNHoeMyU4dciY7FTMgKxUzICvlmLWtCtxaozy7VqqII23U"
    "QFhtc5kHf7h9+ADcrbmM3f2T/TqPTXiiF5Z1mY2hCuWwnpVKkQTzSiG70uNblMNr9N4abCyJcfLfWLRpZ0Cw50i2e5OG+4LF"
    "3bAJMo+tiWdzD19BGMI8tstfHX74sJgG8sSfrQ8/o9MHOYZB1SAqxSA14LT44bLx4gD/2zzfg9OgqJ6tCpCtkQxRtsmV+dCI"
    "Un9CwCTZUNJlGQqZYcDQ3weIo3uJVg/1UGGJAASYpFie+qnmdQGr8V1xmSlArq5rs0HTydGEH97jsn3HHyaX+44PPGc4qfxN"
    "YfRBEuwGVUnzxomc9l3HHRmbzeO9su/ivbLveLBne2hsBsQ9dq2FFOd+MtKfe8mkYrxCnFbArgzXR9h2GJwP6t5feWe5JLAG"
    "ELZnk0bXeIalLqvTiMKMgNbIVXwuLa8BH1bCTbASqMYS+8VOrbIrQG0twrBkmi+IhhqOFoSF0tLQlHqLwkzxRTCAxTfOC1r2"
    "uEBH0TT3Pd1Q7t6OQhD/VrJ9xyGk/UFgpoJcwGNwCrhZAB/Y2KKqQAYyJp/HPCSmFX0NNxXk6fos4RXIjWkJxE7ix7Qoek4G"
    "qgyp3kVE1fVL5oBCpiriRIkJrkCfPEiM1tj9z/vlF30+SOzo9EnRfpm2tpOtgAXropSI7W/bX9IYgruL+Aqh0JxkdJI3Ytlc"
    "Xy4zfJxgJBjv4LdOEpvbdMchVhD8AQ+laKRdjftVIApegAOlnVwU2+1iISr+vkh3ETCp63it3ZEHY10HX3H+wEvZHhQBWw0h"
    "ccZhGfHJlzBjuJjiKiMjjD4xbgML23YwvCoYp7qR7bjZ6V+VvXLDEnlYUdOFEtV04IOst3xfKLDBDeiBinbwQpFs5lp5SJQl"
    "u+SVrNDeeNm2fcoxSZSAAYfHV8EmXM6WoI8JCUtUCi8f70Ep/EDy84PkKKiLQmUlOf1HKFPkrCfG2INUiPQGuM3b1vJCGFRO"
    "NrYwtoFBSpfE2WRLjMaUJNcKuHYnQvwfxqYYSQgccW7QhN74q4UzlIQXKcY45M2Y1L2Yr8B8NcSDJD0XUKNZKZRMNWjE3e8X"
    "cAuYT0dVktRozu3Sn2M64xWaWlRc8i954e2y1Kw8Wr7RpVzi/WPnTkLUQMAghF/AR/TwXJcIgsIQFr1Pgz2AhuFUFc3+hXoY"
    "hXir7hlo6/xCZXbcnwYQ2izyMKx6GX8IJkMABGoF2M7vOyTceJL3vrDIDpeOc1cxmG+Aahh3bkOC+G4Kdwbr6VowAfUJwVU3"
    "EFXH5YPCMjxAOuWFq/u9r7oeADEqc7yxAabXOVsnEyKUmKbgIj4v4lsKyJ7o3MqgtQVcT+LPa8brGE01OChHsm1HBTBRIMsG"
    "cBStzqfg9Rz3Hq6F3GZ2bOS2KkqZPoGkx09QKYBBTWYn87Gtf6xvwNqokSk7skFOjiJnyZ9TdBuKgRK15szfEPco9rbcefni"
    "eIfODL9R9gqduGbuWuI5EoRfo/Kt4aSWK5FdyCovs9y5H8Q/rPythlRM21g6v0F5/NVQKHkJONCFQCcj4BwTaRv5setBpV4q"
    "ptrWgJ/46bV4gdTcuaPnK77mG77lVwxcZJZduGRLYSlzb72Wy3TN974zvnPUKZBdp2taBVCSJcqTl9k7lZasc5plwC0Uva+V"
    "DQrfwK3Rtfgl3RWkBLoQ17AN7pLkciTCNpMDrf0d47GVRCcuprJf98n/ACHCCNSgMq6BZ7gTL1R6hxELLsDc44LfcXLtHoAJ"
    "Gp1XJEnqMjhLGbz+XuA7QX9y+uNvEfQdGLXiX64jcwDE9b/X6UVkFXHB2vZikAsG4IK/1OmC8esIQqfIAODEQ6z/xHdVkGdR"
    "FL3rAlAfZCMaIOnoDMJbuAixvD5W66XD5Tm6CF7DutjYaqDcfqcCQbKHAjVSpaL7Yhd3ednYuwpOScDuwRwAIFfsOwZiLMQT"
    "9Vc4/kIdmNLdV8nXyAYbt3n/F5RcFjGEhhTx0b2lp9fPVWTyv/dsb7CqiW7WIPeQfO8VXSFLw4PdAbjqs7wC+zig226zI9TY"
    "OaqJTUx6bTJ5AitPnaetMpmcPP1i8LqM7HlM/kal4PCuArfwNb3+qg84v7rTxUaVcKVq/uWS0GFJfFQpmSEnyehcpVWvvYTA"
    "RfGkp6ptUW86atp21IwfNxbh+fnoBH2z1ECOTT6q23bYUldc5Fr9jRp6XIg3McXg8likkXgYl9MP09D7mHnA+ZgBgRIcoTs0"
    "Z2zbjwopTOH6rPtQOKOT7n5rPx60lqTDhKSKVdG0sCGIGazIa5D7js7AQ5tiGYbViSIFqXRPp4bkDmthubd2ynQHAtd7Dhm9"
    "RRXRc5cZ9bvu98D0icz+opwBNCI0/AchrQE3jKaYKawgggTTux5yTAwT2R+TQ8qHKgllG0KgflDAvLwihP3wVv1BMf4aScaf"
    "xMD7oPrN+vaedPIRll6IVyplU8fPPidBfPOgLfNLKgLwZP3eQPoxjg9oiB6FyQCJguiVnkPzlQidm1xlv4BQP3PgcTrTVFZw"
    "hz8QUezUb4cSdEdHlDv9C8c67HtagtJh3lCkLmBDhy3kleMHwfFPfXoCSzitgUwgDamTyHqO5uRBukDeae87KfSIxHUKLNeA"
    "noEQOfgf9DaxNQVnxboBIVIzBl4Mi1k9h8N2Vs/FCpp8MrBneJHWYMpv5+JHkE9D6NAVwYUdwflEGY4e7aPZxDuCN142vQkc"
    "6VQBdbhLLW761IjnEOEKigbwARgpQHKZE6aymAeGHPATpn8kVWwRWlt4ZKhERiFzF1XIQrFDRkIcqHcLZqlOlG64DwrQpVHH"
    "KorG4/ErjcOv9OvopfJuZoNbs3tLA+O7ktNWNljkBGqSMrUHU1gJsILiq2lBAJ8V4qMZ0vBR3QtIViw3s9VcVFlausMan+4v"
    "EoBL0w5KkgpicyHumfZSBS08QQ1o3QevgpCAogSntsNDq753aDUUDdjMGpwK0EE0XIH1jOQjAI004K7LS6wj33LqoYjD0qF5"
    "6pGTtdFBCiCUoPDzCHIT3Cz6bLVX/FFINcdMeY8yYDw2BuzYtwUEa7hDaRaLKIAnVsBdYAxRTavEiACp2gdXQYpNKV04+btz"
    "asXBLtud0IAoG5fo8LWU40WDJHWLsqGunp3MifUdyRxoaFbPJnMUHQqxXLYt/H3uuDGADPTMLwW2/ekTt94/FDo9UIEsF0Ll"
    "P6nMjHvRKK+ddsuB1/u4oIR29ailCij6GkSV2/s3Xozost8KguDNLNdz4SqHvd+kOBw6SZyfJ3JhOEWk6+H91xzeZO3g/mv/"
    "zfuvxXU8KpNktGUhUqkMKjbEFmpCWpfswIi1PHRT+EiLF4heYNNvFMUUrRDb3qtL7+EdAjGlkf0G9SiuMARCGFHBtluhDBkk"
    "yTldWHWXOXEyiiRQxVmgmw5BbOcqVYctDt5Sj4562BFUogUBQSzCoacePOlcJC3ZHrwUOPcl+IaMjJcYVRooPvavEFfOkTeC"
    "qqMCpgagbAU3s8lc1Dwt4EJuoi3Kwqxb5yV2I2YVXwfebDgrG8Y3kWM9FWOJI3DaSErvHPoX5FWwUQsXJXO5KO2t/+344CX/"
    "HjNGCvcXBctmcFtmc/6zilwc4NWg99hpnQ+Mt6nH0X+vUmdhFLsEI/+ZE246/qs68JGD9mUA+jpENIDTOVhSQ8qvAv2DgWeI"
    "IfgZWGEF0mS/5tSQNn6vgBZhXgyG0XUd/+7whgZNB34JRQGeSmJ+3fEf4mYfaJzALf/oFwU3VLIwmAvUO3X8nw91FssnDhCl"
    "ecTt9GKOz5/2d6PFVObpD4os3ZBQfIQ1tFj6I/zX2GjRJsl3JHDq+N+w6j4AkOPalZN04OWwRqEbwbzUgy3Q2AKTJBJx9bDw"
    "sx/oBAPsRC1G2l+n+K/uQgLofudiKUlSbJCGxL2P1zbpHEU94DRgDFGmADI2pyyKOI6Ov1eiLHoq/lu8B35XA+gMPDqW41Eh"
    "wNhyCoqAX35+sVg4arxXupHGZjU3clNfS38zhvMnK512/UUlwcdfVrnnd6BM8qjBs3qDL1cg3sI3a/jlvsGErbslgc4j23BA"
    "hp+riwq8xF1BXmc5kF0Kw6k9ICRTpXPhkC1gnq5pFcGBhSdoQX9q+hPCHZX0XInRKNyqz7zNgDMXBoOyIDQmZ18uopUSdxLj"
    "00iESdQUq8yZMNvIWYEQx0+JewWI5kE9If9BepIgjOp+6mESBmvXwBCk9wsnL6T30ynIzydK8lw2nSEgp1zDjAVx2xrwIQ6t"
    "vnNo9arI7vyYMzoECI2+K7KLB1+sikzm+S1atQJDF797CfYV7iPu/PqCczbK4rfFOpnkNw9ky9bJl/mbh16k6+TLL9p2jb5H"
    "wI3b1tSA772fl99K1q28BrbIL0264g7YtuY1ByNDPrIsW/mN51/6by7RcEKSUJTXRH86fifiQGO0wnAJ1Ck4VRK71PZ2K+R0"
    "JzJGwyxw+572udCXfBkWCMSaiYroOn4xqFJ6JxxU3yqqr21BaI6UKlTUOY8A/nNynCvHIBOozVwsYt8YxHoPcrAO3GftwZyF"
    "U0pmu0Gctd58GZDjU6YwRhKvHTXmUqhpnbqoauBx66CwcCAdFqSpoIZsIvUUJE7gYiu+I9B8h5PaWRChEjWvnTK1T94U9or4"
    "Gfy5qZZQMs2flelhqfHMwaDACpGgEW1+2cmzF+AcrELlOgrvUeSyuJCXSr9FaTlAHP3115UYVJaUWeqVywoHtJXDrjV858cG"
    "5PxOIrGJdvIV76WV2SWosRskoMgaYpNF5YyseAH+WTdjBZGe+FXyVb6FHRF5ObpKJl8myYt4Paxo8UNz/o5gMoaN4osk+Sn4"
    "7DV86YBZS8bfQaHghofIPG9gBWwigzQ4+TYMbbVJ4+o5OPDMAN9imRtEPMPmcbngK8rhElw+B7YsGE/XGCnpFzj1v+nRYRCS"
    "skjXBB48bPfAeTdyW9dCFqnil2x6nV+OY8dtK5ZdJsnochw8u4F/kllTgP4RlvuKWBawYMGRdm3xztHa9rptFyxJ3ityeYoO"
    "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"
    "24RkL63JWUcub6ZUjNuNl8nB4sIPr+0YjV3l3wqVh5xrSvypWK+r5aGuclH/gqzPX4GP+SsxJn89HuOvwi58Puf/Z3WQ73wi"
    "B8TTiNOmPdGkWtGkv8B1oRy1T3+ldt52Fe4mb6KFqjNo0YzUWjQD2UiyQRWa2Hy757JqYckQSDizdGxmrFhOzvwM/N7IRueK"
    "UCFUiSjdrFjxTrYc6qgjzcZuOQq8fWK0CKM0NjJuCw9+kRzQg3tWtpjSFbMjyjb4k+XabJ29wH6WjpTVwqhONEGtatKiD1Xs"
    "uM1NFI2bKBo3UTRuomioNTQ/vZbvH6HQW7E8r74EqXTBm16LGuMVKJa9pAj+l45XgtbRyg8wRCaGqNcVDw83MUyucQyTsrWJ"
    "XLJkadI4+kYrE2FhnKv9NjQMs7pfryi+CokzWo4e17STc+R286pPrPxPtzizR534MucEO+8qCF9zP0/LPO6oTv+pWL0/jOGq"
    "8kxsClttA0pz2MFEgJV/uN25Eyt5n896R6j3ZBBb2axreoHan4Phe6y8wWrBO5riU5QjhtQ8H6i4bLaaWy01ZHzaWngX1h2G"
    "nZYic9ChpmWDSq+BUunEr38nv3t7WkmiKw9woTJ/3T+STo78NQmHc9oS96MWSWZH+MCF/jp6j5/U2rkOe7D9QrHrSPd0nlKw"
    "ATmK9xJRxQl+7+PTbdv8sw/VJQIm19hz5gCkoPANtwFeTs7usFMry93VU8VCve1O7NCVxZwFFsEQ1ztLiLeFqe0mVBQbf174"
    "+1R4q6PORHtq5m1vIEFEGm8yVU/aJYS+bQfYzqIfhZs/4VmVxBeTsgzuNqtRRg838iirTO7HmdscnezqVUDg6HYQGE2nHsEK"
    "OTeP/TXbbSh7yL6NZt/LrfHNmBOzbfGyFW6y8hwde8uLWiCymlSB+OIpVrz8lMDYszWPStf0mHNVzrhy9IuatNSI43YRTJmO"
    "1qSy1WJbZDSV1sRabyfzx8wJNrghspCdJNHpS7a8Raw0Me2FUKDaYEVwoyuYn4qUq2GIxbZWuSmAF7nGvvfJ+bT48/6qj6ZL"
    "RkOs4YfkiEKgyC6MVFWoJ7WnlO9V81loYuodpS7Uq/rJUMjXFE9a1VrM8F4plquWVQqn1GCoX7AbOq+L2aoFG0Vt1YIfyfFL"
    "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"
    "UsmUm/n5UVgq7Wc4THspnwRXgqJaXB6tVIH4n8GwU47OY5yab6y7FrL+C389h78ewV9H8Ncr+OsB/PUz/PXTXfKYe/7CXpNq"
    "brT2w2NgwFN8vAld+8DDfmJbyMpw8v/+b/H528ZOMvnTD6XPFCwTiq//939WAGTR0+FJlGS0O9oV7YC2S5ujrbDaUtXnfmi+"
    "se5byDqEMVnI+pb9d0j+e4vWUZid8AH9aofYT2BAHn6PE/Z7bSf/7//CD3uV+AFJ8OHz1014Qv8NyL/2cZRmdDYyvF6RutEm"
    "29BfYfSeJ7r4d/pT7p6P8lc7pP3Tnmm3tFPa5//P3tuuuW0jCaP/9yrUfXq15DRES2233WaHUexOPOMZO3FiO9lJj8akJUhi"
    "TJEKSfVHJJ4LOf/ONZxL2Bs7TxU+CJIgRaltJ5vnnceTFskCUCgUCoVCoYq1yNpjjbF2WBuseiRBSp9BwPPUdieTe4vFvdtb"
    "GPk4Sb/2br+b/kTpB7tPPLjOtMazNdt9RuE6jkuWMb3yI1gNnoQpjf0odkmIbpKv4v/5/278RcRreuXNeKQoFxYbGMLO8n/+"
    "X7i545LAKwD8z/8TpP5C+Q41FirAqhUAQEMFYO87nsSJzZAfYA3KpUIn8MO555JVqHz+miYLLx7Lr1mW2essy+AK3mThrBfR"
    "KkzpxDDX/tRI535iHXH5ZoXegh44jvvGe//CT1J3s3FBs3bhsh8CxnSamEx0ngc07VAnf3+uvUJpUHIIdRySdfnioXotcUBm"
    "NLUN0/kScuzhAp6ZWZZL7emCexWsw253sYBzfGqtEmr8PCGrBdwpJNRa+Dd+aEwWJnZ2uXBccCtUZP8Ca3Em0Xi1gGuj4sc3"
    "AYU/5nq5cCj2LHRSC7aiCU1RxL/138CqcG6EINPRJOpuNviAwt/sdmcLIyQpa3u+cC7dXKeAGchVSeIyBxyXuOhy4xKX67XE"
    "LbjVuKMc79k2vFNrHHhJAqNmMXXQcJWVieZ4Em2/HFp5/2Q8pmHqLBckZdeg2SXa12MGz6l0tLykIxI70nQVXi4XI6JaPPHV"
    "scssmPITOkOxL7jhHBGpxdnhpRuF7vFycdkfWWn0Fq6EMR+h4+XCSmCpNgbmCP1heR3gHeWOsnO4bIqcCZdN5wszvqSsEsXR"
    "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"
    "uK7M4fWEK0G270dG2Hyp2dtQawwiJqbhEDYf8glih4YTXXzVt5hxQ8g8vHtl4r0stsxlubabmeTD3MlDeq7SNApbx/Qcr+IE"
    "ws/xPXL+YZXQuJegJFV3zHeP6ikko91hi4vOVCCUAn2UPOxhfcjOShS+InyTdcHuDJY3nSSCfZ2uKItzWmmxGKiUl2HvrNv/"
    "MnWvb5qCjDLQllFGGXAA2/m6YKN5viS7YG7ISaFWlUNbk1WMM/K/TFIoiBTYsQy33+xTNFqlyLr7NXvTYy4/Wws2R7plBWtD"
    "3RZwtBnJeUBoCTPzloXKZt5SE6oQv9kTP4Gt1USZyWKucpcjfcGeP47CXgwsoBQV3L21jD3xYyNOA7NauFdT2g6j1MjruoIb"
    "7GMvMEv1l8DyJs2GNhsRfh+labSoFjppKhSFwW1NfFw2KABlAVQ1sCOfvD0uSTGAc8GGVwIA8676Gce/vw27/BVOzQILNIXW"
    "nVelQk1fGiiTv0OJojR95Sf+ez9As2VZcnMqbu2abXvTtBCtVS4Th/+6edI/PP8YzTWGRmWEqYuNqpPkyaJOmCtfbmppmyxK"
    "lNgLOU21uzJ2smjND9rKm/hSw3s7NNcYMZavb7P2AxbM6gZM+XLThM2OA6ZDTlPtrgMWzHag4N0HbIfmhIxXKRPQG3FYBCrc"
    "OApWi7AO3ZJs2VmrqdQ4DVb+pErN/JRIX+IjLQ7aJVEu4aY9B6W3KVB0Uc9E8LsoqryCGnW1Xpdm5eriTmu7Vjk029I3Bn+X"
    "zokadu4dL9jcvWk0XiU9XH0Kp3i5HqmtG4tZeIjCoDS6YUm7lwU4K9V8xa1saRNRbbRCBqGPRtMp+AzXVM6+1hFDlyKiHAmb"
    "q8UsNHhTuoAmQMS0FXRlGseen5RUJO1AMbjS4FRr26ZxqTUyWP2WoLqktslFwFfOLckIGmeJUrj1BNmezKDUhz3lW97O3hKu"
    "UsUenWwh5eq72lbe5c3tL/GqdezR2zZST+lunfwr7W+1TTXJo5ppqS28ZZK2SKrBK2/MqtFMelG0PcW35OQo4L7vBOJt7D99"
    "ihXs3LlWU0ffxdYThzd1h2lTqmHnXrabMrybe0wY3sw+06VadMtk2ZYFhuu19WlgGknNyrUmcGMGmRzfPWcH1r731FBL79ah"
    "NpNC1622MwIb2X86FIrv1rNWEwG7tvsswAb2mALlclv4f1tKIlZpQ06iRtqycq2J2pjOKMd3T/7H2vfmf7X0bh1qw/+6brXl"
    "f2xkf/4vFN+tZ634H7u2O/9jA3vwf7ncFv4HD8/tJAaofWjLyrUmKoI3UxNA9uR/rH1v/ldL79ahNvyv61Zb/sdG9uf/QvHd"
    "etaK/7Fru/M/NrAH/5fLbeH/7fnhWLWNCeIa6StKtibtluxyKt57zgTewt5zoVh+1461mQ/67rWdEbyh/edEqYJde9hqXvAu"
    "7j4zeCN7zI1qyS2zo00SQ1b1liyGjdTOy7Ym9NYkiEX895wnspW9Z0q5ht072Ga21HWz7XyRje0/YypV7N7TVrNGdnX3eSMb"
    "2mPm6MpumTsckUkN+bUeIKoTTKF9UdlWX6MWRZsJLMD3nDKV1rZPnT9kd9tOnUpzLabQH6LDW44etiK6z0GCpvDeyN6VP3c5"
    "bPjfRIedGXenk4g/IiWq5v7tqO5svq8U3RPNO7NtaxP//47+786u7e3/fzAKlE3tW3Hc0XpeLLcPdndlzpYW9j90n3dmyLbm"
    "9z9Mr8sm763Y7WjFLpbbB7u78mFLS/cfus8782FbM/gfptdl0/NW7Ha0JhfL7YPdXfmwpcX5D93nnfmwrTn6D9Prqgl4K347"
    "23TLJffD8a782Nru+7+g7zvzZXuj8B+q9zoT7FYs97CoVsvui+lduXQHq+v/GhrszK27mGT/cFRYBp6/m4qJJfYyemHBvRC8"
    "s4UTm76jffMP0O/dTZ3Y9l0NnZ+y55BVd3+7uxakihE00tLoCqB7sluhlfbM9nn60JZ1Cs3swDifphftve4/MrO0NXLWIHkX"
    "DtrH3P07dm0nxtrLgv0ZOtfoh/6xWauVUVKL4J3Yamdz9O/Uqd0Yancb86fuVoOf9kfmpBY2xSpqd+GhHa3Gn70vO7HOrtbg"
    "T9qbBufmj8w0LQyAVdTuwjQ7mng/e192YppdTbeftDcNHsEfmWlaWOuqqN2FaXa0x372vuzENLvaWT9pbxrdaD8y27QypOnQ"
    "uwvr7Gw6/V16tBMD7W4Q/cR92uJt+pHZqKV1S4/iXVhpD/vm79avnRhqH5vlJ+9ZxRb5CW1BLS1TH8cOqTT5icxCH6c7u9mI"
    "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"
    "LCkSZoUD+zgSWCJ2F/7t0bkSqokk4CHESlmtlfgb5iiqjQXbYYFYFhJfghWRnX9LIywMSmagIRzGGExAwResbPdzTBORBUL7"
    "UEfybmUsod8Ay39elCEShgk/h2o6nL4L5csbWdF7fCigQxIZnipRwbGkRqU5hQHFohr47lElNRWbI5HCX9g0qE+S9RxXOXKf"
    "M/WK9OekXlEyHFCWJg2PBoTycQS0FBUoonpRRPWm5udr3Md1lmfq7TDU83XuPRFJiGpYEJJQTTtwhwKpcECMoomHAmJB9DAP"
    "Pw+bAkUbJISjqPshBQiR/ejjT8fo+Usi8XyNL+xXJQy98v07UzDiuCutYBLTL7F5uL3wkWl7IilUyYxQG8SLXFkR5KD1kLgs"
    "2dWw88F/LK2uiOahVdHnrMqWIY2tcKTg4T+L0gq9L7hdZAArSc72hr6iIqrdlNGZ/krH6rx77jNVIrHYL34WF7+E2iL6Mw2Z"
    "He5Pzx74HHP4EED1SbLhI7kNq5JKr7nqk+SFRUg2aTlH/7mpubn352dKsHQnOo1Dw+5D1PKHm+MsJKiG/Oppg6KFBL1KpS0k"
    "D/sTJ0ZyAF3b/BXnoYJUKJmEKpFqT/QNyMRV+qB2gGz4DcH6azg2+AS/W1sq6vfJJTsM7/VOuQQ5GpZqa0uuietyaV11ZqrE"
    "OAJZ3xZipif9hfQY9jsYhlp1pMNUg4o2F9XIG4mlBDqebsNNtygUhVJFKFVUctQoNCCu5PmuPcaKSiOq+jvAZFOlrzSZWhR/"
    "VdRSlARCgz5yvioq4dolP0Dg7OgLoVPIDUAdIMPjIwKlt4vcIYXJW674J+UeseQ/kVWLKQWKXB/LWpBgbs8D2yUh2fwQqEHH"
    "TC7VQUeqLtnPRFeqJKFnMdJ1pPMXV+uKzeLy2v5/fWIz4y78V8y29wwkiCgZz804dCLH6YFS8ETfAK4KsoB+mWTlAYtSdSFp"
    "md90KDZ4bi7vV5bBRO5YXbF3nMwqCaGzkMwPKvV85E88dSGNnpOa0U/SMMJyqKdKlIYiOahyW5p0CmNhSc+sQTmjwT7YKpHJ"
    "siamupACaMXXfQPQ7Tw5DM9VF1KPVVbKqIxG8oPuz5YQbstxtpEbolyS/k64SR78QkQ2gwYjZKmMOwM0Jrs30ki7SpQVcD/F"
    "vNI7ZFL8hfhuYF+wmlhym36MvAPkDF/8pgVEi7e19fgEAYnzA93SSMzsHAaMTcMBN5tWPmpGlFDzveW69vSCbER9XZbI6wke"
    "+Iq+IUtt28RKUdqzgPFITUA25MUjBqOXp2800Dq93nzh39mV/RvwUzGEM96lF9yc+A36aTf9xxwKo6iL/8Cfnr65uYyZkGx2"
    "XSClvSbOs71/RWIvf4OYddjzW9ycfh8sw3M5sfmrl3+dYHdOg6oDx+CRA1nz6y/vO5eXeez1kYNzvIXF1ydV3KICz5wYsQyi"
    "/XyRU0cYaaAZbjxHlMQk9qw9S48+V4CGrZQ6JgfPoLrrh6q7ZLeJT3Hd5e2cOpJVyXmWWjlM4gOKkiw+iVJYL91NSb1qM1Ux"
    "23HJqjGFETblYQUWCrIwOYN+KKIypX7gPelepNbPidkQsLe2aJdjMSD1VTgTCkl+NLcp9Wyfhb9OIc08Ar6L9J7UwQ67p2aP"
    "IDyRUO8rrxlil1JKBfWIYXwQixHKkU+Yu9ls4tjmiW4L2d+3ihDMh6uG4+av2eDtc+ATAW70UEP7Ldd/5jQVIePoR0rKydUw"
    "hOQjJVDTAESBjeRpQcR4WBqAIgkc103gEkwMtngRJQmkz8/So0XvGmFdSwizlVMFVZQ2ZFG6QjlohdRMYVWk6wQBfWI65zFK"
    "MRPN8gKqRDC1EqtjQbYYrWD+LKmSoK5Dm2EG0q5Ul4JnMaXpPXyOHa+VojSK54g2FUPyyV3VS+0qfhIkttUTvMGJffncfHxU"
    "wbpSld5nSlmaK2VG9sh1Tu7IdcmdKaXFk/SoOgjo+XdNUU/LglwaNYLGqVwSykFj1FAXT0/SDqkxlq0kC3VUEkoCUEjFDXlD"
    "zm+WjY3NjUostSQUjY1NodIKk4SyUA9iBQVZIAU3hVg5+DU2N+LFSrIg1x7UVGfNcr4sbAr1/KZQhb8b1XQOGEWpdCOXyCim"
    "S6MoCjWhIcibxkaVFH6SrlJ5zIYg14XaBvkfybFHcvR1tw/3ivf+TFHlkir15/TXVVS5mNWPWlAzKgLtyW58hejKqHIlvEiQ"
    "55mi1lVpTv66M0JOunP4SdddEeRaf0POyzDlG/nGRmmjdFPpFyEJ3oTSRmkkF/skh1AS8g2hxPrR/TPGYspVobFRE2pLHxpC"
    "Q6DpT9I4PddyVSiPakFteWqKglyBFS4vf6oLcjmohaAhw2Da1fj7RukmgieZDJV2oJXuQE0olVIlK6mSdVTKVwT4F8JmsZLf"
    "LNaMcr5arwvkbyuZp1QU6oFcSvbxbXkoFaEUVJEsyBHkj6pZk9Fo15eTa4JcXpG+SdKfpJcsmKrHYKpOYKpMYKrMYWp5/zdG"
    "cj0jWa5mpTeEcpCRDItN05+k0z9zv8i1YKOS9aE+yhdlUnVnCchKQqnIPifL1fNVQa7lK6UGqgpVtgKbwnLLVUEu5evVBgK4"
    "5EtVzcubG6V8rbHcn8Zy/uJGKV8sbm7I+WppGahLQj1fLyG5Ksi8H5W8vEnqr2RkX8pf3JDlfKle3yjnN2tLM2QCIJaKQqlI"
    "pugsPUWA4BHsVNp0Y6MhNPKbVfqHptXymwDhm5UTQPPLPZKFclAdZWBbQNOxyqHqdOUbUPkGqbwsyMvYoC7ItXY5oMh4P42M"
    "S3BaCHJ9o74hx0+acr5Sh8OmLsjxYwPQykiuxc+b/GZ5o5wFVY0gE9jkzRDaDpegrQ7oTliGf5MklwR27pyEPPk4Fw1IRfVJ"
    "ukheAnvIsrBLvQEtpDd+l11I21nXWpY7uhjeLF0Mt0mW8GIY8d6AsnnnbnkWQIS+x1lt9DYHKinv4AOIeQKSIACAosJfVSKe"
    "zBVVA5aqq0ospoaisgdVsvDEdxGwTeiDupBcVt3pM61q/5lV030OK5g+hyVPnxdMpJMDuUFEpBOyiPHqteUpSXJ+rEef8/XI"
    "KJ4onU4TQRBDuqBuMWNhRQ35BI83hABennLKXiKTx4xbt9ylViiVXKOMTlWljM+lzqSJaEml7FhCQTuqdEHvdcAxsoYhNQ3c"
    "T98gdzIJUSL5CuVESV9l9kqJ1jdCSeoZ1qsJSjKqJpMe3U7Vkk2PSnXphDGR2hHb2Uc+5lsB5I8IDiy6GxjDpTfx5mDLDmU8"
    "tw+G7FksFmw6/pzyiRbSQXIb0W9sFx2n+Cx0IhfSa9aWMrGP1IV0xL6V498oQ+t+aY/tQWvRFuNbiwq9oy3G955ugqsoYIjD"
    "XMADm4fVjG+6G9/JsJSd2J6xBwOiZd6JNgo41khvlPTUhAxx6AHhiMPD1paa+Mq6Bd/ZY8g0JxuIFcvYQZy1Hq7n13hmw0Zg"
    "26jy2hZgDE9mBfRgSLYvsWyRsbA5zOLw08Um5VXJc/sKq0xC4MGMs9JVuZKgsCuqpLloOEyyVdlWvIbLWUaPU7vp9ll6zOfz"
    "Lt1/j5m3VNo7z9EtXv2GLD49RWbJ/IZdSuKJjMllmDb+ia2/uhXi2GUZRgJL+Y/pxWZ4KlzjLbmulEoxrMQOLIacIDNc9ENR"
    "2AHFT44qHT+nEBP1i/v9u7UKKTmq9EpwibWEkcCGHBhoj+RaT2uit/oUqrI+RFVHqepXoqo2Q1V3IaoaGBNvxFEV8P/VJ+kh"
    "KQx7ks6TyMVBFjY2KId3If2ShbdoFsAzQgzh4CCD8UuzRse7H2Q01rO1OYBWsLKtAcjMF5IbpBHXGXxeRlxphAVdhF8yIWmJ"
    "HRGM/JzELtw8NOBycp84tC9fo6kHnEEeUqI6aJF804HlG0MhZACg48IAIv2eBhDORV1GKXTe4Lv6oXyQYq7V7DKcYJelNziR"
    "Ry21SpVOVxEFKQ5tSZXOU8whCnxL24dvStg/Es7DmsamBl5TyO2XNI8YADVdLSQu78pP2Fk4+Bl2llSXHp5T4yXF/VTxGA0T"
    "NQwwj92wXfqqSlaqLEtfavmOYQM94Ps9olXI6wi7Nn/2fERoeZTcni92j4LQQrKDDFIDvvujidlTF5K3XJSIyPqrCoYkqhHE"
    "REqTLCQA2TXsI92ISaK05QaJhieO5RmEOGmFHC5I+xciL4SVbNlTPj3An/axa+oWkH5PkvnRmIh0blWGCBMOl3DZkd1bEn85"
    "rj104UoRe+aCL4nwieGBzs1qOkyCtV5SVIA1/4xgg+vTLREDMe1UP8+70WyqFtH4VNfWqOpnXvf2dUv3cS7KJYqSy2qwKIt+"
    "6xT5o7yJZrmiRB91KycXi1K8kFLkerY5X9I/ogZf7N5PUHtfQ5CHr/Cb+kRmg9F4gHJDpLydbHOEc4/havYMuz8GO4aYMJnX"
    "v7b2hVW2eBJj9aAgJtgimyeJWuzU/qb7a0myRfJ68bqyhFn94CeEWZT0Se8lqnVHT1QqilrE5GgNyQgACVPIi1NDk4Bkox8A"
    "sSZInVC8F8C5QwCYFz6UOKeAo+ZIDiLVKqKocABKFpB5AQJMRAkh53LdSKDh9JQEhpxL6j+ohIylNadki4Gk/o///r+oIj1q"
    "trOhbqMH4moqL+OAS5JSIr3Etg3Fel8jRGMS7YmimkhDs6/AV0ymWvb0q5LYRsnhKgmQTmCtr8oXVjAJ24zcD7WEeO2MActq"
    "Xlf/QV2w2kV2slfIMeMEsSMr01GVufqwi59Ypc/O3NHPnrl6QE++5yDzys7UV9UnKcg6a+gW2dBsf5XyxjxI3cZJAXqALKTe"
    "El6/pJsu9FfN78wGx+OhF+eMC/MSjym6LzMMvvcsOdgirnb2Vt2hM0SPrFNZt2iWyLr1lah+ksetLbWHtCHR3gmvYU+rBK1h"
    "BkY9lrOErckeBoEYE7vOg5+7WIVSWdiuAAHPFAIi9Y5ZkKMziptDw+4h4wq0lG23j0D54guQ2MwOJe/Yns8cAn+J2bfEcthW"
    "xneRGlYwo4qrERa6bh8JPVfXhljIQVPxqoWCQJLCukRB9wTL9gUUIN2AK72gU8V1wYEoEpHdCSbWFZdJICTWZOpC2gmawFOE"
    "Cezr1pCTNSD052nqQppGhSOC5CpKNG0X1niPVYa0QF1Iu8kW2a2vmyzF73LjKNm1fUQvS3Q7acTHZ48q94ZfN8DICaThrM0w"
    "ne+rF9jNAxvsQE7TeUjZDB5YJ3uSzoLmO5mRvu3oGMa+z2qkCSvrO1yaohNWkPX+Ins23viqTHrsbrNc93ZUFJA24YWnFnNV"
    "t9os39BemeWaZYEb58pMtyHVClmB6bUx0i0/K+tBiFt19Uk65rXDNT0j92sSJ7ZcF/hA+7ZrqgmE9+4pvmQga6hYjLXoSkOb"
    "0rs6uafTZyRR9HWj46liEz3yueJJ7sRS+lDhJZ0wxYCXCwYlygTedsgKK9qiGZg5URokaVNgSDn+xvaFusV+FRVbG9eXqig5"
    "LOsjszBT0cS3VYbFmfpJ3nb81sS3F9wMTXWJhvFyLqIpFmUbY+xk5DrG2Fk8iZLJ2s69O/1D0Oq+IpxZYo/7nrGtyLGta2Tc"
    "WyH80Ki4CWhKAddiAV6OR01Qs8kDAGzhdVWIIEHB0pB1hZmfrK19YTmZ+lceTidPeXwKCXicYJ0MyKpzUKdvls26S3gLtnU5"
    "6YG2nkWV4j2PNPi+ABEV1YVTEyT1ZUCeGa7bAQK0k/PFPBvaSai0c5Pr5HpDUEInqvEbYCUQokXpK9EnopTWV6WTc8MaJFO3"
    "lKLk+dhR8lXpq6lbGwOXos4NTR/qvvdVKUpfTTRbTi9L1Lm5MmB6Ud5kMNBniiqYJujRjuwpC4zhfYXhh5YSpAcwtRI34In3"
    "6ZCZ8Ti+YrKKKbEeH3obGw4wvWBzqtJX27pOa07BQA0xdjLHJgHc3JOuJw1tuOWMpDrATGAYTc2oHERApM8xaGTEQid3BgsR"
    "oZsEkAJaYFE9GO84CuABNsYxT7sRIRAf+J7rMmW/Z0VeZN9BpgFIxzq57ZEoMfzJt2Anp4tZC5Gx1Ua5EO+K8RUY2mTyueIa"
    "UwPDzV87ub6YUxHFgWApQc0L2EigQ/1nkSV2cpfPIiOfKsvkExseSQodJEcLle5suFhPdCuEAyes4XDoSIzNfhhwVv35+YgO"
    "mcSMQHJqTuTEnJBufDYhL/85E8LggtxaPVW6oijFm5gmGfseg5V4T64+6okUgSTSAoJ9Qrp2N8RQSAtOQ/5jCJhd2jghVzzs"
    "q9KYITg8hAAPUisszymgCMVlKIg6DD3gNBcFNh8362L4uD+y9T6mcuNO7heCK5fRIm9WiuPRkBpT1xO2YjECMGlJlgVPKzDU"
    "ZDWG4j2IGQzGVnwFUopOC0oILvedMjvYzaIhvQRwzWTaj4xjcrq0DADdqbXsJM6ms7AIpTQ/OZoYfZp1MtFP5GCSJRPNlE6u"
    "XCyKK86govR14uENYrSnW8OvYG2dMf2Or/wkgZGglMHOk2wE7vTi0xNHW72eJpr97BI6qrSfmtBoBZa29GFqS59kbOnTn9zS"
    "o+WFvgi4EPEtIkAmPb7zNUOVtmkWzQ91ILLJFMijqdJNmpDhg5PSlehivFw7LEfPoMwiKFHkOizCkXRsEkP4iLP5bsMS8KWt"
    "U0ZdjKuYgYiGbOMTLwURMnJCVDTG86XbA2dmfVV8uiepGLUhHQTpDXkcjcK2GeOwXKLMoKOgyZrlBiD55+eg3+86vvf9O/4G"
    "8ToM7D9iyX0CNrdoPeKnpvstjO8h3QfNoyD3GkiPj+rzs9e3HXwYqkQFG3V5s9gYIKQ+PYnSXdAk6QoR4xemuOd86yEP1yrS"
    "9dg4uLDso1Zr++7i+uj8xmhVblt2q9VqncOfs+ke+Xd+PTu/7I4h6WK47/ilwom5gyeVwb631zF3bnZ6rbf+YX94IAftvfaD"
    "UTiQjYe76t2DuXl/Vr+2DvX7Vmv7bfsST94utF7PuH/bPj9brxS0sb556MnVrvMivzR6TruItOHh1Ll4CNoVTZu2L08mPe3i"
    "4si9Rv3ypq6XJ3evml93z29vJ1O/WMK7Z6+FYhd1zoyrqrGO/D1LLteszrF1clY4GhROBr1yq7Puv51XC0GwcycfN3ask0Hh"
    "eFDYGRTmd4W5VZhahbpRqDuF+lEBykTfytP16/lLZ7M6GRzWq0edu7szq9gema933V5rPD/qHhZ2T4Y75sXRxLq/1GpvN5dX"
    "p2bhdtzydg+Lt70Tp3B72Wr1t4fbt9e7wyEaDlut3e2d1lFx+3jc2j5v7a3fX+3cXszPXw5f91587BwOW9Ojh9KmdnfbqlVq"
    "jn07f3moaa12dXjWbTXalt0ayHev94XaXW3YNgJcrI4fynb3bfPFv+pvvvrdo5tep1A5bq/Pe7uz2Z5+vH3SOgwsea+2a8wv"
    "Ojf7xsG80Tm3Ws6r7bm9vTfbOHdGtwcj73rnotxH+l67f/NydnA3rRy/Ht/iQmOvNTDGxyP5+sxb398z7krn2yfV4/rt7r3e"
    "rY5bmzuz/rk/tg+dm5O5fHt/cvPSPQou3EmxY9W9htNqbb5aw3ntbWTu1y8uZm9vtnF94bV7O62+c+v7paPR8X1jr3W/eeAG"
    "Bf3gpHZi1A63761iqft25KKTu9fL/cFusVt5PRte1XZPBm631aqb+LJwhity/WxSvjA8/WLHv5ibtjFy6ucPD7f6oXF4027f"
    "nB9q80Pn/OBufPxw6uxf2J1X/2Xu271B2xrv1M0HudY98a93x6Od64m2eX17c29W7/vjqj9uyPK4ddK7Pzi4rBlHVmd7erx+"
    "N73Yf9l2OtbF3cl5/Xh+PbiqtOrbE7tSxVfT/eDouv6yV76eBu5teYCcSuvk4HJyXut5F5P+/nSinaKjPm65e5X9O701Mgut"
    "q07LN6f+QdGY104bZ+Pp3uXRxHKnvZebbW92aFZq9lHnpNE7PAisvctSp/x2u3Pe9c+CYafX2m+gU3/Hftu29+6t3fvqbXto"
    "rR/edo7Or9ovE7d+PjupDU7v9vTNjuFet4OeXtBbtjezG3f9u1pDt+cv1b2qXn99mZw0tLfK26A7rUwOjP3jylXhQEPW27V8"
    "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"
    "M9KBYA7K978E8Nckzw8BMyPuzUOuY8KaeGceg9287+pAG1nNwj+7wx7a+qfcP3m/y/2Tti4+CtLTeuqpoOfxDPdzNEybxW6B"
    "j9SBRg4sYkQpfCnFX8pP4hO5LbrNwj//fe6xuNFAG4On9/Lie/hcW4i/JFtw+Y0UbgkMBzabzfIWdfslPz0R4gyIoXUs5iGU"
    "Q05VRXKp4DE8i1JJqtB8fvNXB7kePrT8HKf7fMlfL4kk+sKCq0tDeMRwvqZzEtBBsp5IZx5JCKWnJkvj7fMJLZSqVb4z/N83"
    "88Vyo1Tf8gtyKd8oKbmcv54vVqtiQSY/v/tdKV9ZhOLsfEku1X7nruc35Wrpd/p6vrhZKv0ORX25iq0dWUcSeY/Hx/O/+e6c"
    "fU1dyRn050BcGiBPFWFL7FCeVU4taZGBtr+V84kePQmf1FT/vghuHOJJWNqZx/OAbwuYsT7y+6P3+BxGHd+b58R36B3TVBew"
    "YA8eV6GMaOOCCfJTCK1bpNM0ZhppOofJQHgIRRoBTt3YAOv0jd5QFYmou+lvXc1zPu1mHHBhbS3x9/kK+HZzx6qiGsTglQPC"
    "VLc0e5o3YWCnWNPR2tpSUk7NOS4eYNejYaA3vP4ImxD+FrljURVpVuylm4jPFk8jtvG78yaLHdn8FVCfvOUrVrR9u/PIrxdz"
    "GaZ2+SECXkB7ino5Qg7eaE2RiwVmrqwIA933BGRpAiycJ/gjLDgG8rHQmwsY9UeCg1z/Hz0evT6vSnHTDkVlZslCD/tTjC2S"
    "3QszARtVUQ+QI4yx44eZ4hX7trMBoQB4CxLpQw9rAgYvA9A1POsbEw1rAnIx8vLC7QhbsZ46rm75nmBbWKChIQXkC0jwdRNL"
    "xGPBP3oCnvnuRMOu0DcwckEyL/SRJbhI97Cg+99IdRTtCh6ae4Jn51WJ2poo6j5MUtggzAZpx8sLLU0jLg5ApyZPpbS0CJtg"
    "jwzHEhDko+Y8AphiLNclCT4aw0zCEnqSMLF83eBVC2B5T9cqN3EE3xbAMCqvStxQRVFbgjuxwOGCbglciJ8XbuHUG9iuoPtQ"
    "aqBbujeSBNsVqHqCoPsSfKDhEKFfHqwx0gJFbWkB5NHIK5j0RTZlksDNuEZktG17Ci1MbXcM4SDIPHK2KJhCc50jRe2wATFb"
    "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"
    "VP92Vv2STuFQQk2L3o4iFq8PUGg338HXGFp886a63x/lLDJp4nsfeVhlchFVYVNABw8D162BTWJTwg+ilJibZ/nFyFc8sfH/"
    "RioL/YorzGGDRf25r63tz+nTI0t52nJTCdQsn+/RrU86gmMdeNcVK68DA9lSrLy1tUVcdVD+s8W5yySNs4KtkEsMjv0W3HGB"
    "h9WIQfzZjLgxh+/g8zdzQkJO8k/XhrWVlVG+8uc10Xwrq6FnKFsiv0k2o5Xvo4kHv3RThDsiuReo0xP24+ZpTTkKXoSJ7QLf"
    "+dHKU4S8tRWa4zxBTG5WC2kKyGf66wPPmXz0qae1p1S3F3z0+Dne90dMyjw1yYjZPYZ3hoUi4OMh6yzGUNdPjA+HHU7Mr8/8"
    "i/ydCp6kMnsKLPfnv940x1j4OYKiyIyxCf1wzjRwocX34uNjWKOLPXZ/4vPFqFWLJW9tFRfi0xPxksEuQ2wcXbJv8kCY69jL"
    "Wfne/Nlme+l9wY4E8IryBKfSuir8+F/VdT3ni9wviySo4jecdybeKPcYwiz0iLSTe6cXNNYsBAcBxPgkSha9jYN3vIzC5Bu4"
    "2Z8pFlyk6AjEBRkBuRbFClJPsbQg+UaqDy8fXwgCXZE/zJV7j+4/scKsr3DpWFEDfCJ5yM1iRSbyjeTiV4m1teiZOcGJlaUe"
    "0mhZnitH/IR5SqxctAakn5lNw8WDtAx3B77uLnl7fqSpT99SJdnZ5ifONj/3nrkPFf+JE0QM8nEMbqjbQkpOuNjb+OXd4tQE"
    "ptQEeFxMw3u2154bcmJyeo2cl9w9op/3fGx+/w6TQ9UpRBBtq+tWyGMM5aeEaGvPmzIuS9fzZhmXpVt4q0gHhBbG+f4IuS0/"
    "VxTzvn1C3FZdOw52d5CHc+J65F0yIvGOSddCAs9q+nnLnm5t7YLtvGVPJRdSUKAPkW+7J8gaTtAQb23l2FSHn36vTtSt8I34"
    "2YKcCvUqrDe3/dz7yPZ8qEKhR6M/Yw/U9Q/xyko0APjDBR7QHCwwAXnmsv0vcqQ49EUmqo3wG5OtfpFB4D2ixRLOreAT91rG"
    "s4GbVdYf7m8WPjE+plIKFcUUFa4zXOuNdZDmOrQCZOikHHmg+UljfdZ/4tiTDoT+eE3wDtZvFiWjWZQmTYgn3fxSlAaSw5yw"
    "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"
    "Fj/+1d69+9K0zAL/x3/9PwCUVIc="
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
