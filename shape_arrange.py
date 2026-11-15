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

PANEL_HTML_SHA256 = "3a7761e04496f53503c4110b212e57db100fc72d08183dd73f2aceda90a3b5cd"
_PANEL_HTML_Z = (
    "eNq0veuX2zaSB/p9/wor14dLbKOVlpPJZimjeRw/Ek/i2JN2nhqtxKagFmwK1IBQP0bk/35PVQEgqG5nZu8594PdIgjijUKh"
    "6ldVT0erurR3O/loY7fV+X88hT+PqkJfic+k/gwSZLE6/4+nW2mLR+WmMI204rO9XZ9+/ZlP1sVWis+ulbzZ1cZ+9qistZXa"
    "is9u1MpuxEpeq1Ke4gN/pLSyqqhOm7KopJhAIVbZSp6/NVeFVv+U5unnlPAfjx49bUqjdvYRtFB8tq1X+0p+9qg0ddPURl0p"
    "fb7e69KqWj+SqWSHStpHVry9/CBLOy6NLKxM9b6q2HRdmxTe6kf1+pEcN7tK2XTJl4zZmZ6LydRIuzf6kRTn8pHSj2x3XZhH"
    "Vhw6rsVszo1ImTg/dFzhj9GEFwLyjmFMntcr+cymZ0wIMZlMkmSQPKHksyRJB+lP2PnkyZO2PUp8+j//zXhNhTe2MLb5VdlN"
    "uqz1z7tVYWW2ZLzxfSyaRl1pXopUcgsNxD4KOVZ6JW/frlPLpvr8FJsEfS5lqvmEdbzyJexMbWsY3/GmaN7e6Hem3klj7/je"
    "l1mNy6Kq8IGvxDNjiruxavAvX0M7b1MJXVzOaizy0ZtiN1/y3UOvLqSdL/n2oVcvCivnS76Bd9AemCchln6Cl/zq6E1jjdJX"
    "S744Tr/bXtbVkl8fpVM9yyQZjSS/g5fpdSpZ225SyViSbFI5thup3c+ysOWG8cv7A2XrC6ya30Ihl254GL/w3Ro3ONJf89MJ"
    "488f6iyVOV/yG3h7lUqWJHIkxPLH4sdlksjZ2RyeTpdJslye7GDbvdY2lXyCi0xyKYVMl/yjvONGruHfYl0b/Atptf5F1yv5"
    "jVzXRr6p99r6JHyQq2EOWlk+jZ6Os+htXIx7lKsl41ZCJ/5k89HeSrU4h83WtinuOZlqxljHtRSfn/795vMr/l5YmeLCN3JX"
    "FbBYJaeNgAM6YWNb/7zbSfO8aGTKGONGis///k06e3b6x5x9fsVf3ivCSL48fTxZwrc/1DfhW65kyAsbEDfwsPyTvmLGi5A9"
    "X9b68UEBxemW2XLJ+Au/V0au+6qh/VJLeDMej3FzBiIkzqb6qRxXUl/ZzVSfnDA50/MU83X8LZbGNTdiNIHvXKEruVZa+g2K"
    "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"
    "ERkyLVKK2h6Ker1BaVlAAtyc87LFpjgxlRLj6pbJmPce8RR7i9ew+vB5OYXa2UtJxBCWoWWJKclKmVck1iTSczoEyyGzCxR4"
    "EAo2kvz3ApVhmTekmULh4CfvVJWCu6fCvpBmYzAVY1CMttwWI14Y7hwjfQNLeSx4IsqpiAmfio3odi4uvKhpIL3VsBAoWSkG"
    "SheaAg+NS9HP05yV5GwxhSN3A/z/jcYK0WRiRI74pB4yREUvOv5WE2EwkvHoeTmwFw4NV0RGlxyqjCxPWRLzXraeR5Un4osm"
    "urxs+fptDgVahJa4YlB6SIE9uJhCrE04G/Xhn8qz01KtSreKWPZElKqNaZ+WbF9na/ZTRXRJ+JSPLhnJ6JJoAAkcI9FUBLar"
    "CrdKNerkmTHMMksP+qEzzDKLt0OrFMYyGPBYlbwEmamY1CSQlOaEjvVqWBI6OKMqGpuE4YtO+RIS/Go1fnmAIJw6GKoBXIkz"
    "5OWfjPtHKDPOlbkt0EFSIp3SQpOEmVEJjvJWeL4uU4b4ZjQtYYswXdmVZogSlLEfvtchUl0F0fC9goMUcUTWjneVRMEipdUg"
    "uazKemhcgfrY7/X+p4D8Oo7imd1Jhl5AjZ4KxcRghbtsTrHhkhEtQlr54XsBA6ZgFYf+jNmfOoy0Ts86SDqwGvTyoFr/eE/v"
    "xrE3o4lImZUClCqPiHk494Z+eiehtSqbsWcdKg1ZVWNF+eF7BGIZTx0pBvesw0ZT9f7bfNrq/tt8+he6/4Ztzm4GzKZKzdfs"
    "9ts79mfC/kxzN9+mbW++JfOYeqNPc+lt0v7S291f7NLbXdWlt3fTPC/q755tiWjJ+Rto/PIZZ0OO+qym66ZrWXd5Ebz7ONfx"
    "rsqXeG4/4iWeW+USz2vtJZ75tP4Sz+tMqnEQf9RLPPPpJ7/Ecz51xlOhz/mU7JJ7DgVX3GXmItANGOoS22W8whQ8mAIcF/EA"
    "J2ANfYrTYvkb/tbGsRt1BSVbvXxiPDgHTMtxEN04LoNGcMtJxKrAcWMKW75r6po2I4EvJivIYV+rKcLVjGY9LGDyeOlwtp36"
    "MxotBLRGBpqX8aQQ0s8bsffSIRE2KAllsAtYehR+D28ArgFZyyKW4Aq9slRG0aYFq91bBcQHJ2Phe7ytiVaQrwLDTeZe6JKr"
    "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"
    "df1xIyvaYOTo8jaMHU7IqIL9Wg8Q1QkmV78orNHXqEXWegaL5BsOmVJtzUPnT9nctkOnVF2LIfSnaHDD0UMjoZscJGgyb0zs"
    "Q+VzncOG/yY+rC24a51E/Bk5UTb3N5O6tvm+lHVDMh8stq1N/P8d7V9fXNvb//9kHCia2htpXNN6ns+3CXUPFc6WFvY/dZvX"
    "Fsi25vc/TauLJu9G6ta0YufzbULdQ+WwpaX7T93mteWwrRn8T9Pqoum5kbo1rcn5fJtQ91A5bGlx/lO3eW05bGuO/tO0umwC"
    "bqRvbZtuMedmND5UHlvbff8L2r62XLY3Cv+pWq8zwTZSuYFFtZx3U0ofKqVrWF3/a3iwtrSuY5L903FhHnj+ektMzLGR0Qsz"
    "bkTggy2cWPUD7Zt/gnavb+rEuh9q6PyQLQdwwc3t7tokZYqgkpZGVwQ73EzccrW0F7aP04a2opOrZg3B+TCtaO91/8jC0tbI"
    "WUHkQyRoE3P3J2zaWoK1kQX7IzSu1g/9sUWrlVFSS+CDxGptc/QnatR6ArW+jflDN6vGT/uRJamFTbFM2kNkaE2r8Udvy1qi"
    "s641+IO2psa5+ZGFpoUBsEzaQ4RmTRPvR2/LWkKzrun2g7amxiP4kYWmhbWuTNpDhGZNe+xHb8taQrOunfWDtqbWjfaRxaaV"
    "IU1H3kNEZ23T6Sdp0VoCtL5B9AO3qcHb9JHFqKV1S0/iQ0RpA/vmJ2vXWgK1ic3yg7esZIv8gLaglpapx7FDKlV+ILPQ4zRn"
    "PRvRuqbFx25Q4IfvP7i0QCUt6FhHMB6ZuDZe59UkViP5QD90R3QYMVA9uwMYGTEYdteopFaiHpkReqd0NwMEfdYO3vfZXwne"
    "91ke3vdtGd73l48I7/uLAu/7Ty2877MGeN9/ZvC+zz46vO+zTw/v+50mbgoPycnwsH+ePjSqGUcMysG2v50a2v8Qzr0AuwVh"
    "zeDhxzC4Wy5Ta+olEDMGcdpRyeAvBAQiT0pAZE9sg2V/GWH80TSau8tl7hVDi3TNnR1eoFIMD6EENPCf6kc/fI9f/PA9lHkN"
    "AetC5Dq8cldENr+7fR9K7OGVS0IFiNgtQBbBV/Yr942BCuFH9lP9CtoVPsHfAin4SU0rztMgvfhdyCOTqPmSGdKdA79XvwcT"
    "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"
    "IkxfryF7c1h6DOkJANazJB/YsxTCVrSOF10MW4uM9ub6tBNvXg6ZJ2kNooSWA4JuQvB/Fknqj+9E88sJMGBwMo398H0uvKk2"
    "VHEWixN775rG4yC6sTtsoJ9pwhvGNPAArVwbflEyBdqrjXtcjMCozcESaOKb6sQ3l7M6QKo3fD9BMBi7k8ZemLCYJUqQRnjJ"
    "2ijf5TPmas2SW6NF7MGPv5kklxPDKK6bSQygjTJfRdDz3ii6WbvauTf007tOz9pPyqOZx4rU8o1/4+DbmoHVOysFnM5ix9YG"
    "tlwkEOkOT8lsbbxZ7fBKriekNoFfHWw0L1NK8MMW0q5NXS3qpeT1zbLxoK977Sc+oPDft1HSuZGB+S1ExW/S2m3KwKS6MqLx"
    "OKFpu0JY2moG+OE4UoNtK2NYLR6SWdlHHVVKtNN8rorI5lXpSwmzIadJzT40tG/TDq8ksYLZ65Fe2e8PaVIphHtdh+ZnBBbH"
    "vdTHegKUR84tNZ50FeNE0jKnmlguc9aHeK8gMfFn81wPV1XDEjaVniyGQ5okLdjMU64/dGTG9qNHZGkthSJDk8CJ5j7iMJK0"
    "PnQkNRX00La1HU9aOtoPKZF93VEl8m00sEqZ242tMq1thpcUtlYj7MaLwxZch2Trjy2Wq/3AwvStJRJTN4kdtu8RxxMj8aGD"
    "qbaUBzWp7TAqU9B+DGHedQcQZtpo9ORzths6BRLbjBsmTq0GDYUoUy2YjOnWHzY8W/txwzK0lj6WvEnMWCMfcehwKh86duqL"
    "eVir2o4eDQ3thw/LvO74Ybk2GkCFrO1GUJHKNkOIC1a7pR0dRuHIi+/azPki7QbLuyzrGgs8man98khmaVwIyYY/5jIvo/jB"
    "C73Goh7ewtaLPT0tayz3ZAFrL/hkzs2WfOXsLRd9GopbLfsyAWw1/sDmGntJ2qIXRNL1R1+Ws/3gk3laS6zM0SSYss2POPIy"
    "ch868BpLenDz2g47PSXtR53Mv+6gkxk3GnPl3O2GnIbcNiMuE7tWAy6lt2mztRZSWfCtZH7FDDcaG2yW5SZ31qAx1gz1Rzb8"
    "HCU7udFnJdrydDZq/JCzTVcc5FRwoa21urWduoWFOpnVn8jVnq4ls+IBW1MFa4hDMitLRFPx2v4qJ9H2niZZi3MGxuJZ+26b"
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
