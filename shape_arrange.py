# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.10.0"
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

PANEL_HTML_SHA256 = "b63fba707f65ab7bf75c07e8b4bb0b3699d57385bcc08e996fc787d7603e7f1a"
_PANEL_HTML_Z = (
    "eNq0vW2X2zaSBvp9f4WV68MlptFKy8lks5TRPE7bSeyJ7STtOMlotBKbgiTYFKgBoX6JqP9+T1UBIKhuZ2buOfeD3RQI4h2F"
    "QtVTVU8Hi7q0d1v5aG031fl/PYU/j6pCr8RnUn8GCbJYnP/X0420xaNyXZhGWvHZzi5Pv/7MJ+tiI8Vn10rebGtjP3tU1tpK"
    "bcVnN2ph12Ihr1UpT/EHf6S0sqqoTpuyqKQYQSFW2UqevzWrQqs/pHn6OSX816NHT5vSqK19BC0Un23qxa6Snz0qTd00tVEr"
    "pc+XO11aVetHMpVsX0n7yIq3Vx9kaYelkYWVqd5VFRsva5PCW/2oXj6Sw2ZbKZvO+ZwxO9FTMRobaXdGP5LiXD5S+pE9XBfm"
    "kRX7A9diMuVGpEyc7w9c4cNgxAsBeYcwJhf1Qj6z6RkTQoxGoyTpJY8o+SxJ0l76E3Y+evKkbY8Sn/7v/zBeU+GNLYxtflV2"
    "nc5r/ct2UViZzRlvfB+LplErzUuRSm6hgdhHIYdKL+Tt22Vq2Vifn2KToM+lTDUfsQOvfAlbU9saxne4Lpq3N/pHU2+lsXd8"
    "58ushmVRVfiDL8QzY4q7oWrwL19CO29TCV2cT2os8tHrYjud8+1Dry6lnc755qFXzwsrp3O+hnfQHpgnIeZ+gud8dfSmsUbp"
    "1ZzPjtPvNld1NefXR+lUzzxJBgPJ7+Blep1K1rbrVDKWJOtUDu1aavdYFrZcM351f6BsfYlV81so5MoND+OXvlvDBkf6a346"
    "Yvzioc5SmdM5v4G3q1SyJJEDIeZvijfzJJGTsyn8Op0nyXx+soVt91LbVPIRLjLJpRQynfOP8o4buYR/s2Vt8C+k1fq9rhfy"
    "G7msjXxd77T1SfhDLvo5aGX5NPp1nEVv4mLcT7mYM24ldOJPNh/trVSLc9hsbZvinpOpZowduJbi89N/3Hy+4u+ElSkufCO3"
    "VQGLVXLaCDigIza09S/brTQXRSNTxhg3Unz+j2/SybPTv0/Z5yv+4l4RRvL56ePRHL79ob4J33IlQ17YgLiB++WfdBUzXoTs"
    "+bzWj/cKKM5hns3njD/3e2Xguq8a2i+1hDfD4RA3ZyBC4mysn8phJfXKrsf65ITJiZ6mmO/A32JpXHMjBiP4zhW6kEulpd+g"
    "mGdf1nqpVjtTXFUyG5xxqXcb6X6N+I1RFp8Nvy6qncz0gR14E88Xrq1vq7qwqfQz9Ug1b4o3qWW5zOyBl3F+WKv5m93mSppU"
    "suxN8ebhjyrJdxJpZSXbVrh9uKrqq6J6t1bN0/lunnc/M5ehkdUSX8GDT7xRelHfYDI9Zr3ioqKy/WEcjoQFnglqmS5gg7v2"
    "Q4ZPTwNmMgJmgyuxSg3L1zI1LFvA/2O1TBXzn+M5oZidyKlQEzk9uHGwB7VMV0hZgL4wf7LgibKU4vNxmg8m/5dO//IPBkt2"
    "K8XnWTr5v+kJ+5xvpPj8szTPJv/32T/+MW3/8Y/J/03ZXz5r/xvT/jtK+296aP/x+T/+Mvm/6V/yf/zlH59/vuq6v46OxP0h"
    "HHBhY2z83ooOmc//Mmf5fJ4BDcMjcinZcFmbF0W5hrUPo+kKhVOG8mwlG2s3iuejJIHtPTmbDq1Rm5RNhZ6M/A92ODBuD6GN"
    "H7smzudjN27MCjmWVSMf+bn71zP2MYU5Y2MD1Z8IczJ/NGcHX8h1rxCYN4k7jjJrzOwHyLqmdo1c+XU0kH4yga6NK2n3ZVU0"
    "TWZ5Y+8q2F5ChnKSZLBKLaNjH7KJj6lljGtMwQ/EAokgp7UxQ5qurNw0Zb2VvKiq+ma5q6qmNFJqvlANbOatKu3OSKXdg083"
    "clNbmNu7q6L8yJe12ej6uqgU0nbVbIot1zUxUTx6A180SldKS25ksah1dTdn3Tq6xu5TrwYD2bYSTrJ5Nz53kugVjpGfnIEQ"
    "1j26QRuMxjRXg7OwBZU4gylT3ZyqkxNmxIVM5URNuYX/dJgb01V61a+0UX9IrBIejisknmVp6k1qGVdCy5tHvyhtv8YXqfHt"
    "DM2yyCXS2pLidBReFOJsXDz1H4yLkxMG60JNimmSXMjUcjMpoMVsL0UxvjKy+AjkQD496xoFxEKMHLkYnHV9ug19onbDXuNK"
    "IBenlqlpW9W2qRFbSt9CbyAx0BgYyjyMTAZ1LVN3enyUdw3wJmF64nTL7s2V77Kh7bKn6TpmE1PDeCHs/VSklUkyKNp2oJIE"
    "/uCkGphUg0Pkq/KEk7gq4pTcs4224aX0xyLb67YVE5jG17Co6e8UZnuieDEVGmuHRiEZLvDJhmFSw5W0VI9NkgJ/WeSrxmrY"
    "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"
    "/cqu5NJ4b1IULBPgb2y9mlTPlYu3lgKW5j/n7Uc4TLTyy3KuIgTldvhWpmqFf8tZEuZoyXoytlWf+A5UOYpXGa+qvnyEOdhy"
    "uOW1GZC16eFES2ZMDtDizchKiIW5KCx2frBfUfPgRF5uvFm2UJKWcm1kyy9pKVccWzLwtbRIZPIAGeg2kcuRpiCkiBezxGkl"
    "C7SK8cL0E9fSJRk1lWU/MheudwE+vU9w5CYGgiXHHfrwRPUtf7fTG8UqeSRRhWQCyDjwXLtSWiwEjgP5ooPtdBLbkd3I90jl"
    "2eX7tByLjwa4yykY5Er9wv3LZLWMSTyqa+lJv9yTtMpjUI/GwSJVHFoneIF1doyKFi1/0SHPbXoYoCnw2CYVZjIkdtpTRpoj"
    "Y6FwlAVXF6myZbPUsplrmQsTcs2VrSdkAJTtBKV2glw7TBqk7BUbt5NiYDASmVFp7qPBJ4195/cx6s9pmraRtZgZ+OL78NeD"
    "p+Onv71evPn5YffFHz+5z44+DF789jx58eCR99ztuS8fHP/x7Oj7D78e2b89P3o6fN7/4Y8XB0/jp/79i3n/de/N4Af3pfv9"
    "mbXwkl9/+Sl86p659mPPs9yn46eLF6fzn3u9+cLr4kMo83pkPf4R8n/49Zc37rOD78M3v/zgvvzt4eDl0Q/950c/nr/44178"
    "dPHT0Dp4On5+dHzx4reHvedHDwfPn3Q7ix/t05Oflz/cf7Bwj+4//P37vvnb+KefXsXTnfcHk6cvDn/ZeRM86h39sDP49eT3"
    "3fMXp09+CXdHR2a0HN3r/j5PXh2P58PHJ92Xu970P6/P7j+0vYNjIzs+eTnOhspZ7vSQ51oYnMRaC5TQGQ4jDGshXiCqchHr"
    "K5jW54GNDwnXCiuoE0SWSSTpBkgSnpkXONLv9NIUdTodPJvNjDDZuf/a2F95gWXCHbRpWThMdOOAihvNyEARBnmybvwUeAm8"
    "WydBEGPdIM24gQ9pTCxoPPRPXZLH9C3s6cYB+QspOa0R4wX2T5YLDRzCadinp0E7YLkeESXQYt4Ix0svMe1AA09qGo4THGug"
    "LxoFMSt4iM3IOqkuKLcU5/OuummWMZZyxvxSyQ38XG43wX6sxXwQbAmKYna5t1IBou2IgXoeXYQ41t8a9w1k/Ad+PYdfj+HX"
    "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"
    "un9woWVuWf4jEpl84V9XcLcPHmQtSP4RfDXzNCdO2O23pSchSRt6mqn5ybt2bfzF1DX7rSpw26tchylrOV9GZ6jW0cpm1PAq"
    "kicUj6D962VKDQbKhSWE0a1LPO0azss16DNv9DbBOcrG8VdsPIE9Rmg+pJ84Pl8IcwWT7aGlH76RykTdQhr+mrTssBS8iHAw"
    "jpycvJCStR1K/wo2RjQBPbZ0Amtc+ELGmpewJI6W3DwlN9t50RjxCWJSrPQFtsbws1062/oxt4qRaarvQNlglQ7sNnzuINIj"
    "affA3BeP0DhUUB1ZIqe4UKZ4q5Y6GbTla7dPoCY7T0dfoSOjYvlJb28v/WTJeLBw8hMHyJMho7aUTsJWq5758WTgurbbcy9K"
    "1S2UCtnIhBDp/YFRSun2XCUt8Ur7Is/8pDfwMPB5TavWylDgoH/E4xseFsStC9MQRAlm9PCK2DfGPR93yKS5pBijrOQG2ucs"
    "UFz9KEOgQ0XgTTJrEPepMyuL+xvNyzbiPufifrOZuM9birvSHfsHtcI+rxP2m9bCvt8n6X5R2FUqlCE3ZcI+ZRy430LYb7TC"
    "bu7f5ypkwvVGI+wFUd/vbyLs+Va1EHYAGnu3Ev9lNKeurRH2ryUXxWI1P0PJr2ppz8NxbWHeXV1h3h0sUNUV6kswr9cvUbPN"
    "wVt6L2IL5RfOqSkMNepyGXQoLvZ25H4PvChKtBvMdmuk5idOb29PPn3snK9W8WoVlhbzaXExH+YW7vGamWFeeklKN94KaZY+"
    "jAGwdA38xXXkxePPcZ9064fj6Db/1kSF4aWe4SI6KyqLHRG1CO+1wG4XLpRPaYyB5vf2Prr631/H3SH3R0xNaHHOHKYawMyL"
    "UHpm5dgSKrY/vhPBOzuXgT96W96y5iUEzfqkzsDBbxVlfK6IfzL3wSOTuyTyFx9DbMBEk5Xm82YmYsYKMOrxc4ZB0YicU4c7"
    "PRNvGOXlouz8Jv18lTMCxdEtM4FO0Apx5VoucUn+6hZvVl5ZU3NgVGvrEnnfRuPK4E9zbhHJrv5pJz6tuT+hXjya5SkRfiUa"
    "wi6EEflJTwP0ROT6Wcekh+JsbcMFrZoJ3A7X7GT7HTEll0/KncipFUR++6eQl5/lXKhPpVx+MZc8is+zTrxu5B0SmL0IFX9T"
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
