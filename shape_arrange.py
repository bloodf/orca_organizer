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

import functools
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

np = Image = ImageDraw = None  # numpy and Pillow, bound by _load_deps(); OrcaSlicer pip-installs them with the plugin


def _load_deps() -> dict:
    """Import numpy and Pillow into this module; never raises. When OrcaSlicer's install of them failed (offline,
    pip error) the plugin still loads, Arrange and Fill are refused and the page says why (see requirements()).
    Returns {id: version str, or the import error}."""
    global np, Image, ImageDraw
    found = {}
    try:
        import numpy

        np, found["numpy"] = numpy, numpy.__version__
    except Exception as e:  # ImportError, or a broken build raising anything while it loads
        found["numpy"] = e
    try:
        import PIL
        from PIL import Image as _image, ImageDraw as _draw

        Image, ImageDraw, found["pillow"] = _image, _draw, PIL.__version__
    except Exception as e:
        found["pillow"] = e
    return found


_DEPS = _load_deps()

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


@functools.cache
def _slop(drawn: bool = False) -> np.ndarray:
    """SLOP in fine pixels (see _base_raster); built on first use, so the module loads without numpy.
    drawn: drawn straight at its angle (no NEAREST rotation after), a pixel whose square the part touches has its
    center within sqrt(2)/2 of the part, so half the reach covers it; the 0.5 widening along y stays (PIL's scanlines)."""
    return _slop_polygon(math.sqrt(2) / 2 + 0.05) if drawn else _slop_polygon()


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


def _base_raster(tris: np.ndarray, fine: float, k: int = 1, slop: np.ndarray | None = None):
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
    slop = _slop() if slop is None else slop
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
    touches set (_slop(drawn=True)): (mask, cy, cx), the part's center in mask pixels from its corner."""
    tris, res, _ = s.fine
    a = math.radians(s.angle)
    img, half = _base_raster(tris @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]),
                             res / k, slop=_slop(drawn=True))
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

PANEL_HTML_SHA256 = "f6d7d382127ac5821f8e9a96750aaf3889369353484644c60e998a8e339f83a2"
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
    "qgIWq+S0EXBAJ2xs6592O2meF41MGWP8pfj0H1+ls2env83Zp1f8xb0SXvLl6ePJEr78tr4JX3IjQ1bYfrh9h6Wf9NUyrkL2"
    "fFnrxwcD9KZbZssl44X0W2Xkeq8a2i41vhmPx7g3Aw0SZ1P9VI4rqa/sZqpPTpic6XmK+Tr+FkvjmhsxmsB3rtCVXCst/f7E"
    "PIey1mt1tTfFZSWz0RmXer+V7mnCb4yy+Nvw66Lay0x3rONNPF24tF5VdWFT6SfqkWreFG9Sy3KZ2Y6XcX5Yqvmb/fZSmlSy"
    "7E3x5uGPKsn3EkllJdtWuG14VdWXRfVuo5qny/0y7x8zl6GR1RpfwQ+feKP0qr7BZPqZDYqLisoO3TScCCs8EtQ6XcH+du2H"
    "DB+fBsxkBMwGV+IqNSzfyNSwbAX/T9U6Vcx/jseEYnYm50LN5Lxz42A7tU6vkLAAeWH+YMEDZS3Fp9M0H83+L53/5R8MluxO"
    "ik+zdPZ/8xP2Kd9K8eknaZ7N/u+Tf/xj3v7jH7P/m7O/fNL+N6b9d5T23/Sj/cen//jL7P/mf8n/8Zd/fPrpVd/9TXQiHrpw"
    "voWNsfVbKzpjPv3LkuXLZQYkDE/ItWTjdW1eFuUGFj+MpisUDhnKs5Nsqt0onk+SBHb37Gw+tkZtUzYXejbxD6zrGLddaONV"
    "1MblcuoGjlkhp7Jq5CM/ef96yqCkmZ6zqYEGnAhzsny0ZJ0v5XpQCsycxD1HmTVm9kNkXWP7Zi78ShpJP51A2KaVtIeyKpom"
    "s7yxdxVsMCFDOUkyukoto3MfskErLWNcYxJ+IVZIBjktj2uk6srKbVPWO8mLqqpv1vuqakojpeYr1cB+3qnS7o1U2v3w6UZu"
    "awvTe3dZlB/4ujZbXV8XlULqrpptseO6JjaKR2/gi0bpSmnJjSxWta7ulqxfSnfYf+rWaCTbVsJZtuwH6FISycJB8tMzEsK6"
    "n27URpMpzdboLOxCJc5gzlQ/q+rkhBlxAxOq5tzCfzpMjukrvR1W2qg/JFYJP44rJK5lbeptahlXQsubRz8pbb/EF6nx7QzN"
    "ssgn0uqS4nQSXhTibFo89R9Mi5MTBgtDzYp5ktzI1HIzK6DF7CBFMb00svgAFEE+PesbBfRCTBzFGJ31fboIfaJ2w3bjSiAf"
    "p9apaVvVtqkRO0rfQW8gMZAZGMo8jEwGda1Td4B8kHcNcCdheuJ0y+7Nle+yof1yoOk6ZhRTw3gh7P1UJJdJMiradqSSBP7g"
    "pBqYVIND5KvytJP4KuKV3G8b7cPn0p+M7KDbVsxgGr+DRU1/5zDbM8WLudBYOzQKKXGBv2wYJjW+kpbqsUlS4JNFzmqqxg28"
    "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"
    "ziZzESqeTeaMqpfX0twNBmIbvodXSw671R0Uhbnab6W2Dev4WlVWmoe/pHf0KeIuXNujjh+VpVcfK0mvQjn95/e+fg1iwo8X"
    "ga//pCt69W3R2I9/D2//nWZAvn/RlJDl483xGJQHi6CXH/1Y6bLar2QzXFBX/nP/esklZibg5Mfy4lvK+nutdPrACnXpHa98"
    "vz5aXpSDysRV8VAnt8Xuox1E3t5/sfBf7OrdEl7C9WBQe59j32yoWqfidFg1n3HjM9JrrD9k/lFdbey/+ALz+M+ajVrbB9qJ"
    "6dDSpt7KhzsPbz7ae4drfbiL9JI6aesfYf82cpXen7T4JWS9qI2Vq4emN3qHGbGG1Ueo1dH7ju81jcPDzXVvqb1eUXyPitGL"
    "5THdjLFvdogjwYOAK2Fmdg4HPmFWCaNhiPka0ZEDfFDhabhXv78rruZwyCNa5tFrR7cDdzDEfLqnK2m/95DZt+u0YKzHoKG4"
    "D8CBa5sW+JQyrhlXIyF2dmbnSfIQ6BFouuRuOLIDHLe+CcXY3YT80foAgjiX40ugqgXLZNdxlIccl4KJ/34hHRyfxXiBHaKe"
    "8OKoe4vQP385KPoTdz2ksHK8qrVEoRQJc2zAO3iM2M46NFMAJPfTvn2QkaTJb0QdTzQvRT2zc1gApR91z6OU42K3q+AILMKk"
    "Nf74R9azEnrqikubvBK+BYiI8J3RtMpwfTrOghtY2VmACz5JkvTf+NrSd4yY7r0o6V3NK276FgK+RaV7lu378d3YHjJEPC+N"
    "RiHUYDRqoXkD+k6XnBZ52ggP8wIBHK8H7eSq3+wwCIgtEK6bILiKW4/JhnGFvei7/1mSpB8t9ngAVDQEpVCgCa2B+kYjkGNF"
    "JcvKCGN5RAqQAS1saoZs4JRGBwgE0KaAcUoVXtDbFv6OJixJ/m5TwHeyHP+I3+mJ91+yTEXYSVc9gDMOILLieMHt2wLqVr/c"
    "emQVMNAcJFjcEDLSAjIysJo8IIz4YqGa93u5ZPyOLmcAu+lpUIRJe1NsZZMSaWNjz6yJcwTehxNl6ZD4MPjSLJln1Oi7mZyH"
    "TxcsukheIqu9SEmkLAIKiU093096t54Zt3y5KZDr4PfQc5LY61v7EciOE8N6oMaPDrQZoDuqudggeFTY7krGd1y8FnpB0zKg"
    "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"
    "tdaH/M8p2kmnN/LmqNglX9T6sLoFVYZKgkKxFbu3jbjJOLO18E5e79Z1nabjsjJTnqN19Uq3ewjY6yoOrgXM6yy9UseFbcjN"
    "c3SiPFQles8CNpA/Q30gKRjL8QoJ7T+O/xh792vr7tPHp1TmVZfR5Ch1kiigQVMbBT5rOzn2Qwm3+a4wdhdf8MAfnRyL0liw"
    "nu2yWBVjjcquf4n3CqxgCyXFRT0Ga2O8lutHqffSLQdMuFF3vFF3vFF3vFF3vFF3vFF3vFF3vFF3vBJ1xxsNx4+n4biD4uK1"
    "8Md8HjcYJ7UcvfuxPXmOVvEuEQ6v7f0ssM+jTQfbP5WD87xOTIXvGf6+OG1+rW4sWn1WEddFO/VdrGpih4fpphdmL4jszA+P"
    "73a8b7Pih9wwkUR3NMAsEaQDfa0Hf76hZsdWjo4peVCBTrKq3g21wo2PpEOz45t33WFPs35tXajCILvj0GMwi7g9HEpo/aJV"
    "rv82mu0E4Ed5Jd7rYb3JG1jtgRkezODdt9UZKRThSl4KzQVZL7tBZ3rbkm/zgLe3znWnx/JOagi7LsHO/gV3VrPeSZUbFv1l"
    "GJwLlSm2EKd+dkK3BFPtJaeWyP5j1R8o6pV2bHHIil0rRsIE43+VXXvXdoGR2DItxQMBL1FEaW2aq6K1EgE3qlY4LXR49WmU"
    "mOlMHG5WgjLdDSLnnXTMovKpi6iGy3MmhdWltKpuO+k3sOPUX0neFhV5R1H8XEh1eT9S/l315ArYa7S1RW3phspedyr7h5JL"
    "2w/3Y34LhnGfyn6Y4mxbTZXniOI1ZLf42IUHjXZGNUdn8a4uLzs5XGnwyTMFnzx26K/JEt1teX8rP7eNB41NjulL3OFezzRZ"
    "YjvvrlS7huIhxI1Xx7Wr6dXwRyou1iPqQ4JUafXHI6JF3wXiZruC2vnBbmCKjloakckA5y6mMvT0S3bmog0nzJUL/MceWnvs"
    "/0NtPkAkQ2f/qQrp48F2JFTspfz6up29d2lN7Ryd7iENIRPI9VN4BYOySbafYre47o7YihwYiiUxYNkIkRGEprpkUADqT9QF"
    "DLKHjz6VecIuJILp9XV7ROabmVIBYv6EGp+6P++3/hw92Ivkyr6sO9AmBVtSVfgQhyQ523RxVLJFIKQ4zXBcMx8r+yuCInWr"
    "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"
    "WKCpfjJJDdeVObyecCXI9v3ICJsvNXsbao1BxMQ0HMLmQz5B7NBwoouv+hYzbgiZh3evTLyXxZa5LNd2M5N8mDt5SM9VmkZh"
    "65ie41WcQPg5vkfOP6wSGvcSlKTqjvnuUT2FZLQ7bHHRmQqEUqCPkoc9rA/ZWYnCV4Rvsi7YncHyppNEsK/TFWVxTistFgOV"
    "8jLsnXX7X6bu9U1TkFEG2jLKKAMOYDtfF2w0z5dkF8wNOSnUqnJoa7KKcUb+l0kKBZECO5bh9pt9ikarFFl3v2ZveszlZ2vB"
    "5ki3rGBtqNsCjjYjOQ8ILWFm3rJQ2cxbakIV4jd74iewtZooM1nMVe5ypC/Y88dR2IuBBZSigru3lrEnfmzEaWBWC/dqStth"
    "lBp5XVdwg33sBWap/hJY3qTZ0GYjwu+jNI0W1UInTYWiMLitiY/LBgWgLICqBnbkk7fHJSkGcC7Y8EoAYN5VP+P497dhl7/C"
    "qVlggabQuvOqVKjpSwNl8ncoUZSmr/zEf+8HaLYsS25Oxa1ds21vmhaitcpl4vBfN0/6h+cfo7nG0KiMMHWxUXWSPFnUCXPl"
    "y00tbZNFiRJ7IaepdlfGThat+UFbeRNfanhvh+YaI8by9W3WfsCCWd2AKV9umrDZccB0yGmq3XXAgtkOFLz7gO3QnJDxKmUC"
    "eiMOi0CFG0fBahHWoVuSLTtrNZUap8HKn1SpmZ8S6Ut8pMVBuyTKJdy056D0NgWKLuqZCH4XRZVXUKOu1uvSrFxd3Glt1yqH"
    "Zlv6xuDv0jlRw8694wWbuzeNxqukh6tP4RQv1yO1dWMxCw9RGJRGNyxp97IAZ6War7iVLW0iqo1WyCD00Wg6BZ/hmsrZ1zpi"
    "6FJElCNhc7WYhQZvShfQBIiYtoKuTOPY85OSiqQdKAZXGpxqbds0LrVGBqvfElSX1Da5CPjKuSUZQeMsUQq3niDbkxmU+rCn"
    "fMvb2VvCVarYo5MtpFx9V9vKu7y5/SVetY49ettG6indrZN/pf2ttqkmeVQzLbWFt0zSFkk1eOWNWTWaSS+Ktqf4lpwcBdz3"
    "nUC8jf2nT7GCnTvXaurou9h64vCm7jBtSjXs3Mt2U4Z3c48Jw5vZZ7pUi26ZLNuywHC9tj4NTCOpWbnWBG7MIJPju+fswNr3"
    "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"
    "yjeL99+lt/+cf3H51zd7915f+eG75Qfv9956ffHutd5bt5eu/K537U7v6r3eH/+4+N0bj977++KPHy///oPlD19+9N7fH334"
    "wT/nX+r99aWlP1xfuf3p8kfvP5p/e/H+N/TV4wdXcTmXP3x56eonvdfuLV2/9+jFH1dufbH8xb1H7/2999btlV9/v/jgw97r"
    "f+p98hnN4PGDq8v3f7f0h+uLd28uPvhw5fbfl95/E2ct82cIKcsfvb949/XlD1/u3f5u5asbjx9cXbx7benb+73X/7R4//7i"
    "97/vvfpK79Z3+GWcP9N75TJ9/vjB1d6r11a+v7X0jxfpK+qw97boEHq+cnvp+n8BQB/9vXf5ztI7PyzevbZ4943Fu/eWbv75"
    "8YPLS9fne598CJkCbPHul48fXMHR4pSZxfv3e6/fWLr5l971z1d+fHvlxtV/zr+0cvvb5ZtXej+80rt1deny20uX3+299e7S"
    "N5d7b7/Ze/Xa0kd/X3rnu5UfP1i899HKrR+Wv7+FfWJ6zKP/fH/53mdKcszSe2+sfP9f/5x/kSCVqTG9bz9d/P56780/Lr/9"
    "apwYQ1hNZsX07ny4dP1Lwkg2Bebh/PWV177svf45NXg4//HS1SvL1//Yu/UHlTxTK/HP+Zcezl8ngn04//GjD9/qvf0moXh1"
    "hNJEKe/l0WvXlt69IwFLJbwsvfen3u+u9e69k0l2WXrvT49ufLf80a1MpsvK7fu9t95dvHszL9eF8KgkutC0YYVvfvro9z9C"
    "isvjB1fUJJfF+6+QI5ZapDJcaPZL731DFCIITFLU0u/vJPNcVn74AdBx85PoPRDDi9eXbv6Zenr84KPlj97vffs36nXxwYdL"
    "b73du3KN2vxz/iXYkV/c613+QIz8+78v3v1y+aP3+xJplNay+OPHS1dfTDGKpesf997+TTqXJaKHh/Mfw6698/HK7d8vXZlf"
    "un5l5bPf9u79lZY+hZqH8x+v3Pqi9/3veleuEXNZvPsm8Zre7e8eff82sBIcmEChh6IrAg6heTj/ce+tL5c/u997/XPJs17q"
    "vf3m4t1Plj+7tvLr75Gc3oh6W/nh1707v8HJygQV2miZ9BTaoPRSTVDpffu3xR8/7t18P1o5maTSu/65KpUW794EjvdgPlpC"
    "mafS++Ra7/I3K9/e7v3wcpSo8ui1ayu3f68mqRAE9FxJUlE5+9LVKyr7Xrl1Y/nWe8T1eldf6b39nwyOgPxz/sWld24v/fY3"
    "vd9eXfnLBys/fPf4wdWl3/116Y2bvRvvLN5/ZfHuvcUHd5ZvvffP+ZcEb79ybeX2t3JGOJe/zC/98VPi58TRorySlR8/6t37"
    "azqrRCDw9T+tfP+93H8ip0RlIMsfvZ+XVBLJtpyUEvVzIqHlj95X80noY+VVnEmydP3L3m++7719LSePJHonwU2lkCCPej+d"
    "QvL4wYcyh+Sf8y9SEkkw8kygJJH0vv3bo/krS298QZ8HMgYd9IlB0/iA9OtXhNGrZJIs3v2SmGo2iWTpvT8tff17YmvUCUCn"
    "Jo+QrOlHqvRN7+03iepwnaOsESGylM7VpJHena+W738JXIcE9YMPaZsAG3/3ztK7ry3e/4YoExgVCrSllz6HjYld4VDJRJEY"
    "C3+7sXT9SiRIAP5XLvfe/g3KmJvER6PNr+aECBJ8+zayp1RayMrtb1d++F3vlU9hs9x5a/HemxS3JSVm5duPZWciJ2Tx7htL"
    "734HChMK8N5Xf1i89+bD+etROsjD+Y/F1sNXRH607sRdI1H11m9BHZICS30pWP/tb5dv3Fq59Unv1q9TS7X84csP569TDw/n"
    "P17+8r3eW38Brej+td4nL8Owb99efufz5de/WZp/EQj25W/U5Xx04x+PPv4zqFC4h5bvf7Z8/ybx7JUf/7D05qcITCIRpM+y"
    "K3kg1GLpH288eu0twWaRHUXZH6IBblhqoOZ+iFVCIB69dq33ybWlN3/bu/yezAgQXyvKQjbzI7kvo8SPIHq15p5bfZv1SwLZ"
    "lM4BoZ0v+wbhjHBL15WSEAL902wquR0CqKk0j2DtNA+wG9KSKy/T4/GDD3vf/o0aAde//mXv1Q96r3y6dPldYgUZfTmT7BEN"
    "RRtd8OO334wEY066h8q65foSt87L+IARkKNF2R7xBlfyPahN78F877M3BL+LUz1Qk5xfuvkp8ukozWPl9l+Xfv0KPstmedD+"
    "7/34yqMb95fev51lEKCM3XwPgYn5CBtii9+9sfzVfdFg/kGCkSbSOki5JF0ob5z/vpMzzOL3Py6/8zkUnovHwR4eP7g8yFZu"
    "fL78yb1VNLfHD67kAEQ5HMvf/2bpTw96n3zWu/NWBE/vk89WmbSAJs7LiIBKdJYdUU3RiHVbOSgrs97lO0n/N9mYCMLfl35/"
    "J4338mA8dtQhqDiX3yUVePH+pxkVOM0ZszkYtEYRYKqim6vQLl1+N6udxui6/+bi3SuwKvNXHt34bvHuTRINi3fnF+9+SQAo"
    "+RfEaUlsRiKQbd7Llj6/0fvDG1G3WUlOVhLtyRx5ns6wEFIWFxPwd+33i99fg3WIUy0eP/io99aXoEzjvl28/+nS7f9c/urr"
    "xbv3qHlSiueD1bvzKoEOdH/tz6AWYCcgMhHWFJRKTkUuLpb+8HLv/r3HDy5jq8cPrjz69ee9y6/+LMQk8yUEW7l1pffK573r"
    "d5YfvLvyw29Wbn+qggPUDvrOy0807uMHl8GYf/AhiLtXP5DKUXp75iRBJOQcCpFrYCNgrwDLR+/3Lr+3cuPzxbs3I4hUOgOB"
    "dP2KaJk3WpzmkDfUzfd71z9//OBDgb5XvhIKapzeQNvrp42tpCxkB39daIHS+6S6FtSxyH9FWxzcKG//hmyWtJqYyVkQ1icy"
    "YFVVWboyT1sjk5hAfG6VlSebieBdun4Ppk6DfP/n5dcFFNnsg2yeQaSSgh508xPqBFAS+UpQWCs6cMwdhUTJRtnjTpNrkZtn"
    "kOvHW5m/qs4t5c7rXXu998PLi3fv5YftwNf2JLOBxnIw8oEJppI36sqtH4A59Zk6UcbSO98t3r33aP7t3lcvRVPKCoX++QMJ"
    "M/S3V0nCPLo+v/LXF9XoAjC4718VzAStYVUmqL5tYhS9W1dW/vIKafiPH1wFFvH2bZWwUgAmkgV6b/1m+Z0/Lt57denlt3qf"
    "vPvonQ9Wbt+Od4j8jJyzj9679ejP7y/evwazfu3e8pt30H4RZg5pSLDlXv9i8d6bK7dfX3oXmUq/zIBIW+unDwrdLMoJAC+j"
    "omtKNvjR4ndvgOEuPyOJip7n94VFGucEoL6taLQ0qljzOD0gUuaF49SkT0FBl0pMEEf5QUr/+ePeJ+8SALSytAV0UMsxtP/4"
    "wWU0KyC2LxxoFdRP1Qh/79u/kR8vpdBI3R/MybuvP/rgbdgO6LcjPoVgimg/eJzu3ov8NY/+/PLy7z8gfw254B8/uCxdPQRI"
    "75Vverf+8OiDV0gHW/nri2AQXH5t+cpr4Ha9Ao6x5Ru3erc+hF2c5+chD1kijk975NGNe717by3evQfuXGQHaPZ+KJx1SjBg"
    "8d6rfQL2wEvPnWdRqODRf77fu/fX3m+vJoa4f235na8ILAUmitAL1ezbv/Ve+frRezeB8hSN99Gvvxc0c+/V3pt/BGUB1TEi"
    "HnLVUg+L91959MHbqh6HbRIx+YgtAZq//Rv4Gi/fARH2ybuPH1wRNK0G4iFu89Lnyy9913vt/tJ730TKEdnkKz9+tHLjKpEp"
    "DJYOuAPF3P526auX1C1P22Tl9qdENdEOljIR3Ai0lKTIPZy/vqY3UiqnIuouvBrXP3/08Z+XPv6RELt0/Y8QksGYBHlenqh/"
    "Yne9W7+mgEvaYltznrjFYoZBxP2kQAhuf+UNYNb0FhkNwkTBd2REuEsTbBbcRsufv9G799ZjiDl9q/aRdQPTYsqIO3l3gULR"
    "2Uv7KQq7i/mjAIkaRJ88mr+yePemGPPqFRFgf/zg8lGjhprQ1SsMgu3SbY+R9sUfPl75BzhoUoH23iufg2b87d9ohrlh9uUH"
    "93p3fpMg5VScPVfWwexevL78zucR56cFV6UfLdjSHz9dvv5G7EGTsErw49h6BCgI4yuXl67/V9/IerRqmah67/rnpJsBstAt"
    "lRtIhQ197U+963eW/vQKGKxXaUNfXX7ncxB4eVMm95iIpoP2kdfv8v33e5e/6V2+AyJDwHHzJw6WG0kHy+HDt0hFWLw7v/KP"
    "b0lpoJ22TsVi5dYXKrkn2lyZf3R9Hij76iuqUpRUROTiiSC6+PTqa71r76mxcxG1QxaeipoLzx6+orB5x/DYdGcErrXAMqVQ"
    "V7e8K6g4cQ2kUz8toF5hdQvru5gM3E4+C5p02SFn1S7jcCUVJE390mduGIArNnt9Bf3FqjyY5ljewgvSd1ccMNpsireDqJHa"
    "ceC2B0x32pEjaAhDlZuMmw28BJIuozC5yQy4m6LEzsK1WDGk6HD1mSvytmsBMwJmsMBqcQ1X5pc+pdnCFSuR2IVDXcwzLEgS"
    "DnZidxT/Zb7RhRPGyj0U+wFJ0YC/9MU4fomNmiZkZ2I0OHH9hECwj9NxmAHtyMfDoM5iti+NBQYQAF494cNlKoFly65Zy/U4"
    "rVUhbLPAzV46Mcq80GEWXjIji9JBDSIrgNxvZgXwVd1yLL+pMcwQBv8nswINXtSaAC3ABdfjiTsmzA60MVN3TAC9aWIuMo5+"
    "0J2GEaZdb8pPXzCRLitEExLOT5N1c4sK0VuacQ6ygDqDJrc8ysmUxMn2W7adRXgfVKpVg6YlQC0sSJ0Ko8M1I44/jTspGUY/"
    "5gKyAaZMHF386HdjBBUYzhQLqmGpILhyru0n7oo47thd0snwXfqiCNcP5HQDF2bPpgENQJBQFjxZIQiw5Adu22dGHa4hCpqW"
    "D1lyXdEFEojvunBmajq52Syf1UMb7uzgRge2DkMHBREXIV+0TBJqFEWH/yY5C+WKEj7yKwDVWODBzHiHe124H9DmJZZCCAMl"
    "2Eem4TP/YghcDvI7BTOBPYC8iQVwTx28FY0cd7rEACrcfE2uAsRcGKTldrhPX5fYbjdoAvmwOp/mos/MzQ597nWwJT2IsPkY"
    "/AM478pw+fE2YttC2u0mb0JTbnTowpHgZhQmPwVYAVJWI+Wn8F8lRn4Cp48zb4MiH2+iNA8uodnka8wP23BHA0kIf8qSHThu"
    "QCnSkPKNL+HODjh8gOYZUAliznBE+jWkTKKIU0lJ3M9Q5XXYksjGiY1mbmfw03H0I/QX9SY3oAikS+ZBo+SE0ZH+02+jIjvi"
    "azgkwJ2A2qlxdPw69VYtrkO/cmvr0IMY4MxtDGmg+lzHkIqky6+C5H0MATf1Jwjt6RunO/ETOhyA5aTFE7zy03ISKM/E2ve4"
    "oW0ieUCcVG2bua8BSd2Qr5OkLqRmIOlavadB8FX5oRpuh1fdEjsVy3LLZ7jXGB7Mcvg0cCm+k3Xd0IuHRhEuZKCZdzeDFDt1"
    "UMlTSEAlQPIWTDPH5ahbHhamiwPvY+JX9i4GvFjYYOIZKAh0WNpDYZW4hAHnFsKdtjbIXB8ZW+LyhQCBkwRKlGC5jnr3QiTP"
    "WN1zW2JL5hXEkVvWCIDTAzIDoH7uGFUgZIEIeffCtOWY7jSAv8pK4uq0jQYXDNvwUeupdpkV+Nyup69ckMutYDxz5QI3/NBD"
    "moCaIYKJxfcsnLAJr0I3jYPsxCLxVgUjoBb0KrpewcSxhbKTW82G9wuxn0jtyn/lTlwl+O7WV42+pwPuYqzEFQxdcXKH055H"
    "D2AiHq+M8dMC8vJKhUggrnbvgs1NuHeBtDe4QXoab4l1YH+A1MQtm3PrAj1Cwom0TZhdTgRe/JDLDv/Nv2yhFvG31e5aiLkg"
    "vsmUrYH/rrtSjbSwoA4Tq8K5OHqlpZlExEewqDODG7SB7Q2hfFZYaSLmjvxEKF2oSOWNA6fo42FOw5k1aivHwFsUoORL4OK2"
    "7qfgafmwULgdQGl4lsmgonAGDGZU3Y56OQK8iptLSAZLW/PHUAPsMJLo1a1LtTkecLrp2lED5JBlmFjmCgSB7kxPEVYGESv2"
    "GppyP6TEQXdpL8QwSsVYIz1LUYNdj1ZAwTqe8G+6rs/JSCb7M2h6nDMXy7ykLzo4FTNbFJhwBx8cF9m8l24HwMPQMCuFKxP0"
    "qct7YWaokmZvMVDHCOFOVLVuiw/SAC4yCHYKHVzUWYHTkGC5V+NNLS8wgHtPAR6EENCsfpNSL3yxJAn4lOh6PwTgezi6J/H6"
    "E7CQDKinR/qlz+gdTBNYSJVuHOAm8406t7vrHrWAdoTHgV1yMqNhgGISnD7lUZIFPISGlFR8fukLVQeU0+h4PFZhxhP5KFCx"
    "Qc6AyZInydFMl9N42E0lMVFYh9TVAdMd5e4AoZDTv8nbAwKV+pMQ+nkgpmqaJGEU9IAaCChDkUtIxYMbQCEcscldD83ySFHE"
    "1XGdvNIlp4TRCgoXPEUT3SDkp+PtyAJDn5vQJyo8kU2qKxYXwMm9X/rYrw+ewHVVJRlVldLckiTA3BAVnuu2mCus1ypty7Rm"
    "S1prNuyeYYC58fZTaZdftSvQLlhoYqI5Lj+Pin9EU8kUFll1MmM8EKTTr+N+04v2DcLGfB4A6ZGbBR/CloeSSV36kSMRVikP"
    "IgxW+ARtU4dzM3lUf7oJ1Zli3aUKl98aJk1LnjCzFTuApF7yqD6qMmmuloAxEXQfc1ucVLb4MVQ9Dti056KVH7stSuxQHbgx"
    "WQRNo93mcNSTXMFCSxLIBX8jULo4mQ9MfLVj+fGcfckyE7pg5kB+rGWSarmTccOD+62lcDfwbHlsK+bH3Su0Fc1MzF3o2+qZ"
    "/LSRXVJj8bCi3FS8JtNW0EyH5oEiDdl1fA4/ocfLCD2c2Q36H74P3HYbzOVAYTak8gygRwqbxofuD4NbX/HymAk3EPjroQYH"
    "11jVs1qRy6jEDgW+OE4Pypt0F0054IQin6RvW8AyKqyGTMwK+lbVOIE7yIf7U2AvWj7bLflCheIO/QpmKDs4Eb8Qh+XJ2MXx"
    "fdbk4IdFoqctC4flvcTxeFTRkHTq9YpwVqJFikCAQpNUykTUPfEp4peYeazUpU/E77WIpsENWQDvpCm044hM8kpgSLqPbX6p"
    "PPjovAfLCmz1rhviufcpbmaPvo/CPyoXCFzcKeR0cNtd3MvS1w472fAgmuFztppvUylugSEK4BVN0LsQTjzrDvDVbNcH8jm9"
    "RncAFXA45A5qjD22zAx/Km8agNHMofY1BhN2aKRhAR9vqrUojrlqc9sA/gGmBQZ442/9yB0c96AUmzCCJjEhM3FaPZ4SfhmZ"
    "BoZtd6W3JWiiLiG8ghW479L3k2fU3Zwz6kLOZEtKEOi5YXOpSa5SUCItsICl4pZRa0nkSSs2xjlTISOhkD6Kjr/EKfTVakfA"
    "DormkTiAjg4lCDhkI6CAKGF4FiIfFBPHwaHjIrrk0pNUz5t7xnR+15aDeqER/JT+V6sJQRscJIfi62PrkfzMd1l+fxARqU0Z"
    "EG5BIS48fVlFoW/hB9aAZ4kj5cjro1CGepZc/EodJN/3U+Pe3KkZ1iWD4d2vHPYjeW7h8nG4/S57FQBf+CxbvWGf3zYWPsNL"
    "IVxZvgEapkPg+127QTekWHhbMDTsN0h0kTarWXA9ooE31BgMbrKAewg4S9VpWLhJhRriKciroMKWWqihw2egtzrCAmCIu6cM"
    "ZrqiRoMblWhYuDEANRpc5bpjAy4TU8Li+wQO1dv+BQQldmzhLxCHtOi6tEx5hlHPC+GmOWbytmv5ccvawtd0JWVOrxrjLea5"
    "5sLNGcvVmBEsfMEcGKeG5e5aBtipgMcCvsrEyE/j3Xa8Fi58Bl+JW/LQ/KAlLLFRvDUEFqdutTQGohCdkFxUZYBaVh5dqIGe"
    "iFbbTRRlWPgsW5UBR9PkzKKyDO6TVGU4BtMyEGmwEpYD1+PmhM3DFl7w2HdhUtRN9RdOiC3g9V2NvqhWguhuR0KIE85WX4DF"
    "g4uec4sv8BbpI7m1F1apvCDG6lt5oV/phYWvWfx2jdILEgluxCxS5ReOyxcWcQIkFYFLM1N9gbpD4sK76TW8BUysD3UPtaGs"
    "S5x1jBlLdBcNHTVNUHmy+gI0Esom0XqCw2Ti7OoFNyzAi3voLhqoa/DnqPRCBmfI0Ra+aAFHQibl0+2TF+GqFDfiWLRzfMEa"
    "8aaSC7Dz/ABAU74QBRgy0PtwtxsouFEnEBEAJmGp9RZast4CNfnp9RZwheBSuU5utQV67bj9yy3U4CqwphtH6PEiO+a4HbpA"
    "KBGoz9ZcGI1w5ROjQilj4uzV7ZuQFBS1x6A91VXgFLKnfQsyx2gxB29wC/ByQOUqNNtqteGqm4JivhXZwheIeWTUZOBZHQMv"
    "BqpbWMOG6FAgYn2FFnIrLXicai24MQ+Ig/nI0miknGB+zLvSTZSIvtKFka6zkO7AyNRZ4K3V6yyggE20ygvs14yfEtjHr1KB"
    "/ajOws+KKCaLYNHtpxL72UIMQEJ112Jt1/cXbnbgXj6wfOQXedUYDFA/xPvkZmFQ+Qgrxcn9kCzEkCAFNcB/wnNRSh9XtBIg"
    "UsOG6wJbMKDhXQyxBoMLJRhgXcIYCJjFwhdCiJtuNtTPnWasM8VXVCm4gXA/Ckep6UHthVYkXWH7WJ67zioMhxy4lxX1NK5q"
    "ktF16SFK2kQGwKiqwhhVw7qEc0rVYECeTyUYRMt0AQZVHIMC2CcDQOx1uIfNxlsPA8sJYZdQFQbHYBfAPMeboqIiDGssNhuN"
    "6i9EHba5F5Vg8N0Zy2nmVF/IEkYiDcC0qEGsV0fyIk4GcH3SbaCp+jrOCuC+qAQB9339I7fsghn3IdW6vOyAmrFKdkBia/8r"
    "9/IT1Wd4wuwAEn1SP6QaDVI9ih5SmYa8QX9eusATlGkAXQf4VcMziE5Bza9yDy+9zk0WsDtEElK8C2XbyE8XAIkiqSFWCvML"
    "NGR556qFGrLN+xRsOOEtfI06Kipl2YINEORd+KLmWDU3P7XgeJrrdEQVgFa2cEPEifS9VgPuYw1b6cINGZ6eyDQ4TmDKog1S"
    "Lc0fEjIO4hHBAweyBL+PyzaEiboNq6u8Wn/4KPvgOF7z6jSRbTY8KkSXBc4ATUtiRYEt+piLjxHMQW1r/3HVjITjOQZC7vgQ"
    "QbL6l2WI1yanv7gsQ8jMNc2L1TAWpyaM5i1msiRD1mrApc1ZMnE9uGs3SaaaWI9h4UsIxix8Bqw4ladwPBL3ppuU8gEYA2oV"
    "hjEDLpWOW5k4r3UoJum8heNRF/1rL/CdOO2FL5RNbEjjvJVXfYF7AVwlL0AmqlbKLySg1JBb8XxglSSG1bEDMj3aKC2wCRq8"
    "pRZb+DkYS+Y4jFKNBSqx4PaFCr1PQD9YYcHEGieiuoJDzrd1wlSwgFFQdQWO7cWQxXxwc3IgRtMqOyjIsdPIUeoqYENfmU2U"
    "A4BaYqwcxmUVVoEiTozoD4IoqhCjAdZzjayImqGLf5NZERGMqCvRVJSCCqtAquRH9AUVAEsWU4DNYPnJ4d0Q4pfUBThFMgp2"
    "n4oKo+ReWPhCKR4QM+9MqkToR9eAt6UW6CkuBdqKqO66VDAhiH0YprxtHx2mCXrrVy8hq9Tn1kugLYjOaGwOygqQFgyb5OmJ"
    "7lavl5Czark5FaNre4xh/3gGli7I9R2riFm1UsKaE4VSCcbqowmDJ7dEgoMyzqlbwDFIXqShc0MxU7oxHMjB9cTPPgJv1cwL"
    "4uiKA7Xt8ZrlGy1pjyXLJSCXg9wRo4WMrkauD9P11UwMZHAUjUF3T07JBGiyFhNetWCCyUMmLg+HTIy0gTYWm3oelnFK6ZNy"
    "HYwLIfilE4USVi2TkFVuUfP2snp3OjVDavSkxe+EfRtZHnGlBAk1JBsERi3Abd0vR2MdyRm4EQSvSNRMUJ5mszZOCGcA+YSF"
    "JEslbaTSNWjymMmh2FT49/rTN04YnhsCc6I0gmzNBOo2zt9YeDFRIsFc0yUoQ1i21fESaR0LN010zEhFmLI7SAmquU6T1ywk"
    "O/KJu6xuBBZxsgpt2pirZjI8RjG7Y+HLZLUEX3hz4lIJlQg84hxrF0hQ/MnHhRMTQisIEYf3VB2BY7ZHKzl2IvdDalK+bTWU"
    "wggYz0NdFyuapJRdkQCi6tB1q+aGtHwY8UmpzpnaCBiSccMqshTIBbFQjoOMqwvujAw+PyFERJOiygixRhHVROjAzeJoGXfc"
    "2sKXmBnihtnEkBPctCBCklMywMfwBtEeshaM0YmKCB4lh6zhXo/zQ/a4ktpwU4GubEF6SJ3XmqIUwpq9RVUQPN7AOGeqEEKW"
    "SYmCCGtOLzd5ZC1oQOWpotIhNePcYgjkmyRfgG2JAghJMRA5LCvUYzQgMtl+hRD2ULBC1EJwk7UQsrhQayIYSkmE4zLqoRZF"
    "IA4pMk5QqUtknTjuv74ygumunnySkdkiDgXFEaLaCGoqympim52B4k9KhQQSjGtVR4Ae++alULimf3kErObXL3nAAV2eVOC4"
    "SIHgToYoiePWwrZhyloFOShRs1W4z52mscqA3Pf/FSMq+SvZjJO4OkIrUo7iBJb+ChONl5PCYngQ22wbNVfErNCM5Pn6Vaou"
    "gqWWRRDx5ziH5Qz3sMazJ3tNZbLI91FVBNB+9nZG4HDiUaNdGB/X25AR2g4Gdp/UJ7RxHY6ywX/gN+TLwH/g90xT1/SZ5sCe"
    "Y/rERFE73hnhI8/s7ZQaPCgU+Asv6HqxFLhHQHDvMXxeKJb8tm0FhU3jA5MTm4rjgxPFXbt07ujaVGdkFoKc3Bk4PaZrG3Ds"
    "DRUBg8b9CgIB0wcANuCgGypi8Ln4HopRzMex6gW4owK+eeEF/GVyvThrQwL1SCHotrlbZ47RsRpG4Hr/jx7qu6K/SrbhNEJI"
    "K9q1a3yiMj5RLNUtxyyg3paaDXhO/bNW0CzwjfqAXizutOqF4OmnDzmBXTqGXs79aMaVxEFmbh5xa4bN/eP1QlAs2dxpBM0i"
    "pQ+xYE78mOqM8wlclgvRssRTPIpTpLlcIGTz4s7ghRcKATbODF5ApGizLeOS1Qpb+z0DO0Kzzq8MzRW1C52SD91oQbG4E3p2"
    "RoKdAhgO9b3IGC3wYpz5dEwBIxr2hB16hn0ytLlfwFkURTcFrjmaVwQcUhZggRdhXVyH67ucioezPUFUcKoTEcC+Dqz9WAdW"
    "/lInWvc9nTltfw5qDikw7Y9QI9GLGDoBUBUQhxpOoajtV2avBQjJkc7IoU4BSLMY936yU6DO+UjDdquGfapp+SXXqxmw7E/x"
    "F14QlMVLENQ6CnypwZ8a0WUPutLCdXLeF4Om506zfZ7negU8bkAH1D0LriIpwFBq12wTw0dRX0V5UMboGJaNx0IskTYKJ15L"
    "eoQMjtOc6YyM66YRGAMevwj7Gn5S8os+oe3ujMzWbMP3K3qTG6Y+p52Jn7ihNdDigaHPaQdzn57ujMxO8W5lUEu9Ewn1+px2"
    "FoZveryuT2gHOiMnCrOTkyDzEtmfJ91pXWt7btuHMNzFyuzcnObzIGyLpZ71K8HcSKdVKGrOiBRRzSBo+5VNm/DPkus1NonE"
    "wU2xYJFt6O8ShPiCJjfdml+y3E3c2eTjwRr5JTqcN+lzkqA9zSqOPFM4UihqJwu6bemaRALicoNE64YKGJ8XS5YpnhF+5WP6"
    "a04b313QTauja7s7+NuHiusNXYOwmDZVoMaAnKJWLmrqt7iNpvRdCMpM4Wih2ilqAvXQAup5T4H6CnHBo8B2pGZQmBXiX8Ai"
    "/tq1S9fninM09LA2ruOn+kSxWMmOUZZjcCDa9DBCp8jra6Ko7S7obV0709GmCvKDs83uuMTXRN+pHizouvbUoIDnZOGQmC+h"
    "sa1rB9U+91uXlJ6I+nbJhWvr2ulOhGB6Ca2LFTnIhLZtS7GoOTFc0ceGLpGQpn8NyLqifKMF4PMKKvpk1TacKZDJdkV3XDzl"
    "AYk5Hq9zz8MYXgw6ivyitkM720lANKzNdIrFubmidpjuZSqOPDNLjJuXJic7tdrxduC/8ALfWXe9gs2Dca55E3D0rOiM84kR"
    "T/IBZ0672Bk53Ckc6Gjj4/rkpF9z2/yQKZlBZ2AzN7dvH+TbUMQ/2xmZtSAlhTyIA7SaWt31omf6nHY+ZggkaiA4fS5+CAd5"
    "9TntedGZYXb0Oe25+L0ftvQ57Re5bIV3E90MVF2zq89pQfzYiy632WB4ljFgihrHZrW7IX47AEmQ+pzmdAmI6DlNaU7zuiPj"
    "et319AnNSrfBb7UscEY3AXLd4jawTbc7Mosois7x+6JHetC3v1o3jTNbfCigD7sppJn56KmLz/ywiulDeYO1408D7rUAzFY3"
    "udz94GyKdg23b5OGaAIR676NJrtSZiC6+cWBpgXXIWVbdmB1YO/oE1pXLg+/qM9p1TRKLnXz6GgsH1F7FDriFwEH0wKmIYQJ"
    "cnn7AnWqK8WppU9o++SMXTe39d4u7DtF6InccFDa9IyI00AtrTiC23paw6VEDwtzAOi3IRJXz1h8uuKKnCVfOpArNa1Khigm"
    "69qaFzqVUPM4HjCqmDCkOOtfqcMf8lxQpQ1/7aFocUvI2ubI2UKhOPLMaKfglPDAf7GoNcTDfDW0KdoJbfKUixcCFcqlrULF"
    "RkctaEiksUOk29aLu+izXbvA1xhrY5OAHF6a4l1oXdKffrpBDZ8Sf0L4CaJvTrCX143QDgogAuDaqROe2zYaODd8Rqx5V8m0"
    "/LYR1Jr74KMCTOIw71ZdwzPpiT7Fu226DQhJQgyoVcMqVOiuPDUoEkNAb6g8NThXLCoKs6KbFnipZltt7HqvERi7QFWFHwU9"
    "4JcCHcwiHQQAetoKm1qtTZaG1pRnkaKz6f/8yvy/f1X6lbnxFy/8n1+ZG39V2vWLTSXInC0ExV2E9kJQrMhfwvJqyCVoGe2C"
    "8IuLMX61d1MDxiiWLriWUwBkF3duGi9pE1G/Tz/9VFASZ+j8Ahg8Tz9NA5QsH85NBbzgFNeP+XrBKRbnACVdQTjjswhfRQfn"
    "jlQmAgK6JFKs5zTZCN2XOa0wxTpuBk7SnFaQrTw3UdSqYuzCbLt2yGmHlMs+64GPcjZHhHjQhWXiJHZFrBHVH5ZglCm+OTc3"
    "Vyxql8BGBmlcAlby9NMqR7EJuKefjlnfRKluQbJgYbfr2txw5OowvaiNCcCpqxdeqIm5kVX5zGBR2yNa+LLjp6jtLlc2hYgM"
    "2LiKmaZotXVkRZIh0l8QpMHp84qua64zFlbB+++MD0288MKI7+OAs3PauC5oQJ8oKuptVj5qrnNY7Ks9RjsIPV6Z1FznBHhy"
    "5IMO9SDUjWcj7U49DIM63hnQTRtFbXaDBUs5YJkbYs1E24DpDQM49w2VowUv6kFrWQ7qy7xdKW3VNrQsZ6AuDOUBEy3lDZVB"
    "bUPLuJR9vlmz0aqvCA6n+WG9Do4k1mrp2ga/6U4PVMMgcB1/AyDNtHy8v4gggAXRLAexmoDpED3T2kGlKjqmE0bq1CFLUqtD"
    "rWjoeYPrnG7j2uBMz8iJ1lV1XkGCrukEOuh7Aipd0wU06ATSNV2oDHpO5/pEUUOYlT0RGQknGpEhomoRwlLw4YCqFXQj0wEO"
    "VxLnLM9psyaxjsoxYgynjEJyzfd5Hqz4RFGbrJTnEtrx5iG0LZDezne0cSCK3c2iJnSU2EixivFajAkU5+z4S4VIsymqS0Dn"
    "kDTX2QOnVSvO+CBsAT7yzNFCWCxEl/oU57RZq+Y6cioATm2yKB4eLYxNFjU4hlwpb1FWCV7qE3J+iu2UWKk0sNFqTdBeiKaN"
    "VwpFEzeKCvp9XnMd0/C6+nqxEStxCXzI01UKRsoJjCAQa6Hjwv8MOsjeFAmKunZOmNlhq4Uzf1783TYcXXtOEI0K1qnVwNJi"
    "8jTMDnKiqLNfRNxKHMoisp2IaZR3cWxkhz5cRxrQA5s3OJwOdbqRSaieP8JhCkcKTw2SDYxAnfUKXbF2PGbkxHphK3LxUrDh"
    "WtO1amBY4Hx/gZwzyySja1dVrhpZQxhHwE5Jj40NMBLA8m0O2+vHr9r9+ZWEQNd0+UghgD4sKpYdZIhlYZ9DH4BNeN2hed3i"
    "RLFY1MpDw0XpqLAyywDELpiQXEujq0oqN/pEOdvVX1DJS21zlkBcjApiqqy1jEuVowU4eNVHIg1qG0KfDzQ8N4RrRTdUnirn"
    "ob8dVNap9CQs1TlwwcFW0DEXSl9b/rT6r2fLuLTeJWzrmp9CaLwCmR1e6yZ3uN1VdnjYze7wo+vc4U1lzGjdza70PdUjEP2w"
    "KlmBaetam5qYQeTVy9dhoI2pa61uqoWcq5buxCqq3zWj70g+5X5iJD5pRJ9IFq7g9GjBLqrusklqbBeTbjJSA9QkJaGIRt9G"
    "K5P0maFngw65HS3UipH6Opdcvq66fNXuT2fQHr+YZNCXaELR0LtS3kvg2YVZDMYShKiCYyPIwSzOFSN/pwwYZ8hjjKAPbV3b"
    "09XGc7h2Bn0R/54pXOxI5m2ZGKjmygzB7RHzqrTcF+Z9LPpxlJqIBkZTjcKDiQf5SkJi6ye5StxzpAMcLZhZX3Jqw2NLlNHg"
    "2ux0M2SFRotCg9MRwcrD1SlizEHwHiGGhLtU6U3gdop3s35c8tlbGyoBSQiwz0FAnOqmxcO+eBO5LsKDmih6afNdq8Nbtw0P"
    "1qtbKHraHcHnFatlNPimaV5t76waPt+2RTs9ZR846bjPjo7uPnfy9LPPnbFHt5wddUdHR0efg/+cmN6H/3vu9KXnxo5PwaOT"
    "jf3tYGjTkdYeHm6p7/f3HWvtObOnOjpTO1RrHCh3Du47+Ly96UDZfv7c1nPPt7afPzF82jlknR8d3T2ze4yHMyfNatU+P7P7"
    "uRMbt2wyp6zth/zy1uPtC+ULO6rtg4OG2Tg03T75fOfgFtOcPjh2JKyaJ08+6502apu3W9bm8NxFMxj2njt7NpwOBof43hMX"
    "Nw0eN46dsE9ttTcawT6nvHmbc+ywc+TEpmfrm47Uq5tHj20MZp7buqnT2XOufHjHHudIfdPh+qY99U3dc5u6zqZpZ9OwvWm4"
    "vWn42U3wTfxu8/TG090Lx7ZvDeuHhrc+e+zcuRPO4MFm6+K549XRqe6zxw9t2nuksad18tnQOT9mbps5M3bqaGvT2alRf++h"
    "wbPVI+1NZ8dGR2u7G7vPnt7baBiNxujo3t17Rp8d3H14anT3c6P7Np4/tefsye5zFw5d3Hch4O1DjdHpZ58f2m6eOzu6bcu2"
    "tnu2e+H5bebowa2NE8dHdxx03NF6+dzF85u2ndvWOGh3+ODWqec3u8dntl8ITtW2XwyOP3umemzTlsMHN3arey9d2mcd3n1k"
    "9FDHKe/bttfunjx2Zr99oLvj2HPOaPui63vVfTOu/Vy7efZA0z+95+TmmmHtO1g7c+HEgXPTWw5fPHyWb9qxb7RuTx1ulk+f"
    "8Dfu32efG3pu95Gth4fP7j1vHd86Nbp9z6Xac8GUe6h95ki3fPb8kTMXjj/bOemFg8ecYX9He3R0+0Wn0d0202ztHz558tLM"
    "jGufPukfrO4ZrbXPBsHQs83D53fsGz2//YDX2WQdOLLtiL3t0O7zzuDQ8ZlnPePIuYtj++t7B49vuXiicWrb3iN17/jo6HCL"
    "j206wbeUh0+Em0/avnVyT3Cy23LtZnv4ueefP2sdsg+dOXjwzHOHzO6h9nMHzk0dfv5oe/9J99jF4EI3cKv1g87UnuHW8+Vt"
    "x48Ep/dONfecDs3tp8+eOd/aer42tTWY2lEuT40eqZ4/cGBsm/2sc2z39OGN/y9t77nlJrotAP6/T2H3vcurdCQXSShU3zo1"
    "RAkRJIRASH09XSQBIoooZNdzzAPNi80CoVDVdp9zZ8242xZ8iS/uvb8d1XJJ7/FYCJcqJ47YSt6t+tgIz6M+aq1KupjJoz2F"
    "yGWRrJGdFvcxbiLl4kBPl7lBl7nJazPDwhKqT6su5gQAthKwLCizCehXA3688EpKmuVhUup7BU+PTNAfRDOBG+vMpAgpCRaQ"
    "05oQ59misAUdo8canxHRCY+oTUhu0PXUDrvMWpiJq+k+T0bikRvseJVyh4KfyNNCdwEXi9JjNFYNdTB2o2qPUqg7Ouxzbmye"
    "+qfdvOznE59m+ytgYmrhSYbktZK5WxgqVJZMVmJ/DtITGpvpfCwdMQwDCnYHzWeKJMqA2z9lQ2Gaq9pMORTpMeJidTBbohtW"
    "HvkaYq5AFnANdLjHD5QF+odxao/zKoKDEcuL3F53wsUKWG8JTCdcmY2o7Q7G5yMMKogFMu/v3NksiTE3pQKxUKxY1BdjuSoW"
    "6ag7yLfrLr4Fx3PQ20Z6EMuj6VRIw0Wx0QJ4yiZx3k19HDBMnlobfk6ZwhhibVjdA/21CC4Pclkm4oaiFVMTtgNEwYb76QBY"
    "dMczdDIOdrB5Ikl2IiK57K5XBiX5E1A6jHiSNnZDfmcXJ5gkGYxD2ACQ6dDIGZNH5tgUU4DJqKsoO8YwBua6S40X4NSxtAU7"
    "Gw6Pim2DzGQ0ZSY+BSluPt5C3Z3IWlkQTqRutpf8JZtwp9IaynERUdDA91a5fiL75nxSAF2v4vtZ5SzWVLlSeTKIcHcRH7fb"
    "XQy5OwQdrLcoLpOUTAUVb/SBbMSHBrHzR4QzXBVwNPBGYz4XtJOFKQSgMPtcF+IB67jrkzDHqlQtJTs2Vz5o0LA4Smwwpuml"
    "Z/VVbUX7Zn+7kEET1qmIAXzICrZ7HJ8KihoLpnGStjK23p4KiCjEeOzB80MgT0lszXUXtLjU/d0kHPatkpvvDASy46PATWMT"
    "TXgG2PULOcul3SjUBpSjBoidJbS1z/YrIbNZOxjipqduZZ0v+t1AVYM+NjuC2lSl48n2QB4Yau+j02yAnZZF0hf0I6ND5PS0"
    "REarwWRB8v6RTeJRefJVZbXnLX7HocI+JFE57SJElYq2ahI+spSkASej44lnYACDpvBOQGajLETHAF9Sm7myzrZYeGSDgMW6"
    "uVWNwS1Y9FeLtc8sunMUAQm0CCs1zoqRNxpiyxQSWQMfDFRyvFMQQsi7+8nguF9GkCN41KlEVps1yhYlkAUn0YR1QhiMnVFg"
    "jnOFkfc2lY63tDZOeV09VFA64Ap5IeuOsx3vy77NF6kYLLwDhi2MwyoZjrnldj0L/fV86swpBo9TYgop2YRLRupY1GNgXRYr"
    "PxZCtC9wHGpqwwzakh4ylcCDSfPHUYHJ5VgILH2Fjboaw4qwi8CDAbbF/OC4nqdecJrKQ17c++BurNliqcBWBXpqDIxXyx2w"
    "2c8PwFHSFnMo8k3CKtMg3OWrqR+LS/q4BQZGd1mk3pBfrhOgYM3hgPLj/qlkdx49gKNhJdKAdFiOs7TC5lOsn2PplCpAa4Jo"
    "1KzKDCeg8nTnHcZcCBWzXCmZviwgHgImBxB0tslaGM8zfkLQHAJNg1lsLqAcRJB0apXWhiqnxpRg5/7CHhRdblmVi3yQjVIG"
    "XPEp2R+f1jt1vkLDcQEE3S23X81XW3Rg+9HIVFRrPyt8NNyNFvxuJCIbBYa1ExdNldH0VGb2aKMuB+ImZCIy4ifeZo9tlhF0"
    "mAWDcXe0HPYNSO0PRHOogwODtghytR/H1hDe+U4ImN31sKBm+8SVV0g6OUbBFOJyu2+iI3EDwcJkUtgzkB/4U083/djaZ1Nm"
    "FpBMMYSkXFd0e4AIquDFe200SmXeYQ6CQ2GkJ+vUHB8f5jMC1FUThw7QSF8k4YgBK9x0533wKOZsnA4QGlzuy33sL09DBxWT"
    "TVKkm7RwkHzm+Iu8PxsOseIgjax5OQ6CoEzBqevL2TqmqmMUAKP1kmRwxhsrvpeqfTab6UE4YKQ5X23mlDUSZqxKzxKWpM3p"
    "3k2PXcmaFzFpl6kmTzHWpNVkSdpgcHQxGwF8bbaayJaDpGvhNNl47kLbTgGLz8StSmEVOKK1JKzA2TS3VSXfUAtaC+FMOO1L"
    "1Ctn+gEHkbl7FLq5wp9AaKqP3eVxv6J3rD3I6DmITQbVOp6Soz2ox5NTkK72/RU/wbe4qKszmuv2+2Y+CzaDBCexuWRIcLU9"
    "6cVo4g0GDOR7/Dbz+dlpNphJdDUTp3C5PM1MQcREQzjFLAO4k8FWEnfmEt0OF0OGnYn9KW9MK3TAO5lXncohh001JvZLRJiK"
    "KNrdLCSzj4xOtq/2U3QAlDwzn63goc0cBwIKszJ2dNXlYiJtxY1MWxLdP8y0aDo/7EjTlXyKwHaVured4wqO5FyqovFMngqu"
    "YoTbjIv95W62U+jEmR8mU6MLWUNHiQ8mGs1xaQJM1hU4lVcleuqTvEvvzWWo2A4cKinId+cSx1S+m81YBBouPaHwDovNmhhu"
    "NEAcIIrAb9Xx8GSmfVuWqxGld4W42CLqiT9aqZ71FwYdFx4eO/3K0fJDtwvFA4wIeV85+GU+H8x2Ig7sMZ4d6xNtTzMVhoQl"
    "B4U7RJ6OpvtoDaQ+sDh5KlDfPLDn59eeV91rczGmby1qJw0/EWuf5dnhWap8k0LVqoKf+Yew04rJzs9npxuKm7q6b93kDMlN"
    "4am9v7lmLbC4Y02fa/ZaVgH2Z6eXNUrtoVWb/tX6Hh8FDvUtetOwF+r72BP/MK/eMbtvvq969Z3wY2LNT7tnQ9SN3IlSGjHv"
    "e3b2JavlWNwus/UFEqtVCxq2XKtA1dt/VB+YRfotWs2v5rkXW4lhhdlT0ttHeqMo4J4dMTQMQO3isShqxfzp839lD/UorpqD"
    "6/As4TsrQraSxZfHQ24l1dnxdK27+FjfoPeR/rVV9uv8bn35YtV6lliWJa6eZ9bDZcRuK4xsla9+P8vIf9XyH0nkW8+XCGO6"
    "lnyrtRov8rEvXx60tmL24WvN7DVZzeT32nKdp9ocq3bu9OuynVpmsfPz1Hl6rfUvX986NRtB69SCjfADe6J1zv56L/raR3rj"
    "lOfPOuk1aTRCagWx9LJhmwK/3Rb4t/N+ungvavnzdr3RjSiItcZbRc+xkqj+bTZAzQdsNAR7l8lpeDK93y6vP2259cLRVG4M"
    "F63WicM56fz81vt+1iy5PxyNCmCr/FdPwL1Y7bHxm/ITtb9vnV6rEnbf1JVbZbTRqy4thZ3Hm0OOM7PqvQOtu6SfM6taXnbL"
    "bfrtY5ttft3/v+Zd2doXXpZ2J+GLOg//SpLF/G8lWe/6enlv5vZ9+XqVa+Hs9bi/vl/jxu9yXqsUtEzX1zNnDas+gJVf8KWM"
    "IQxqsNXwpfga8lxg012HesJfQNLivaZVWusy0tVzK09wXNO0wt+eXrMkryVOzK10bdlhHbOvzbjeetw77NFgjqmlmY1DzV/B"
    "NSM7NhAsubm5ce+dSGm91tuwUDcaXVyAPaWt+0DjIwbyO1f2K/9wrFe1PWVZ57GZjrfeJHn4HmiJd1l//yzurBnLhf3a+164"
    "VolHx6dX8BP4Ce5/gvsXjy1tdCQi8qN6khrJQRNQ7ftvjWw1yX3rt6fXWn8iqhn05tMrjz6in+BH2IEQDXlEPtV/63ahT+1b"
    "8THja/vmfP1ZTv23+PqzxuqcUwB+gh5RDXocfar/1png1/at/tK7jE/tm/OzjLpW86G/tFX/PdWwFH+o+dTZa+/78em1/zh8"
    "7VXtb+mamfP0On5Eamh3NlccP6J/qQT1H0d/qdV/RP++1uVTEPrzOv1H5K9fAj9Wede5c5Vvva+NVPA91LxKHFrdz/CMiN4f"
    "gk/10cmSPDSaQBIXiiPtnMHfO9d1Z/j+DoKltbJNA8haN1p/qxd+LvJOa/ud08yfKoE3nXmveQ21COoeJxhRGNaosFa8+Uv7"
    "rW/Nn7Tfez9L7XjfKVCmSe1Asna1e0cG/Ax93k/W9VNC1ap0N3oQP5ke+DI9pZaEP+l7C1J+MTfux7lB3u4EV4uzkCDpvBdB"
    "0ffCLeZcJup0zl74XuuirfzqLOr7dhnB+y/133oXPZnzB86lasjPV2cEAl92xztS+kJf1xpetafxy3tD15zh3lWTu7cLf4q6"
    "W6fJF73v89tfl+TsqvmOjj4n/GtB0wUH+3c42PjXOPjfVq65w2TvlEYaig18fXsqItf8BNZGCSACX5Dwt7N8Z9nq+rYH6mtz"
    "sN96p480eoPLhChzDevviPTGpvCGzi44zG29eWofcVV0k2e18q5fgADlz3caWdfzf74MtXpYd4sTXcDA97+j2ppevSPaLs5F"
    "L/LF1tXo7e3fWPALhn7X4HUbRHfbQPvX22DxvyXFLnCnyf1wNbye1OVV5Hfns/BOK+3uXvftvUzT7fxkUaAPqOBr68/ysj5n"
    "eHSGfJpvJdltrdx7GetPvxg24Mu3fvy4PS+t3U+6cdZ0bwq8//LlY8nff+wMdX5NXpqoPhigY6ghL/EPKv/niM1XFfzz6x0Y"
    "8Vt9lMZT7tVYQ7kYKtyl3mOMS7npjZj91pPvbUbK2h7tXV9qguxrGGU/MZZo16C1lHvrTdqKUFOxrpNeK53f3nrsO+OAw0fI"
    "cHa7TWiJ+fdw4aLen5w1JtzGVL4OzXCBCr3oemXvpRfV5vCxHs1Zq/xig/s//wN8AzqPcRQ/nPXNe8ZVofhOO7hRFb/IzL8/"
    "Pj5aPTc0reNT9tbpXLSTG1Vyzw3NWhe/mbPOnaq+f7WbjC739GfbyqRLaNCHzsWA8nPWGsO2VqhmZDTO4B+NxNIya1nrUz50"
    "fg9bI1IhMi2iPithltYGndebvO83RdOHOk0zzXPFsPP2jsH0K57S/SUev+r1KDd9nHqXXRB4/HpVsLztlCYKx88JubBzXYta"
    "o6Blc9QqBdPqpi8iVx/UVhtbl3tapE74mXrmey3wl3b/tdjrCj6TO/Dp/mvw6f2/u8neKWJ+a2HQ1e7hBniaMd7O3Tlw0N1I"
    "64Sf44uG8fb6C0IhuVM+1ToP/r8aI/m/G+M7iNcOrl5YIcqse1WV9Xnb3OXWZ+Ts+uL15abKZt4GS//FBrHeF0ZUm75/v4Cm"
    "O35S/Ur5Z45SdNmCTdCVBmJFYdQaFbad6PRQCOn0PuyUd7qik58rKRl/VS39d/VnrMcGbLzXomGrO8Wl+/n8d9gU8FgzUBgd"
    "nS0Tr9yIlj2xeQdv1Xdv24/QV2oAa+On4yP4bUBvdg96w9ayKmkBrlvvsYfvtY30k/UH+K1295Q9WY91BCnrAeq8dXraxTDq"
    "34asNRx/X6ehzm4wvQXStY+ClpP5F2BcW0I1mLtmvV4SWuh8Bfa/qvx8qdz5uXHIhdC8Gna0REONEdttYFsPf1yTvv92vrpd"
    "+I95WtXWN+8w6QXRu0XN/Ygj360h5zlRy6LANS7sow+33kuDvT9aDPN3JsLpT/b9B+L43Kva783dWL7XZuBP72v/E3prRQl3"
    "YoYLMnlNc8Oo+XM3OUOLBtwr3H+st01zHm5H5WNue3BqdtP5BndXoN5rl5G0N7WWWr2/qf3kKH9o4/5It9fvj6e6Wc7zfeG8"
    "Rb51/vZ4Q1Bvc3++f32Vatqvr13NhF+5jZfz+i8U+z4CpXdreR7A/Xr+laK1Psz/dSwfiNz7Hl1G1d6wIbA3qzq9y+771YXr"
    "vM+/Xg1e3u0s7Rc76/29zL0IOH6xdbR/tXW0/w+2jvb//9ZR/5db59d75l/dR2ATMg0IQRs8Ir7nYTeB4pv4b7VF+hWJ1Jzd"
    "SwPtJeVbz9KvVesaX9M6uNRbL9PvudlYHF/dWUSXQFF/cWrxKXu2fsfDhz+DXvZ4LdaIxr6nT+HZGDhp7wXu2YV3Y/4bxbXo"
    "7q7OFV+4Z27Oa5KHdWzL1yf3Lzwg98oAerlcvVqrvxpdsOZDp+c/14jKeDy7uXw8G4+99fIm2e89Pj7mwUPrtqfTZvdqUP3U"
    "FHisH39e6qXJe3v7KGe8XgEuD5Tf6Ec3fnqeL4K/a6lmHzxfcGbz1vvQ3UulGsOd+/CSP/lvnd4kbGdqFxl52tPSKjRq/zff"
    "rVqsWGp1DLPabvX6Mdu69AavGPOhVs/uvJxrP3RqvE8mtakVljxEnQ96xxf0qTxwZ7I/aBhmYkP3n+5uApctVUdku5BT9fb7"
    "7YKJrq5G6gX77Sx+e7eyP36cD03DoyWr3rtN2izv250tkl6X2la1Gvm++ffQ/OtVrXb2f1VX9uPb290lL9Tv9u69oXJi69rL"
    "/zz8T/qPh/8xu50/PvW+dT88Ae5jHVPiIWsugmF7B/yjNV6urcg6vesLfP+CfOt8a+6KyTPwf/7nwx/g17H2dfftO/L24/o8"
    "eOv81/svJJf7aH1HaEHg8/Mz8vLH4+NjnfbtaiDdtS4mt6+d5krR7s8/wB7c65/LZc//jLUktZgwe7iQfVkv68KdHjToXC6d"
    "n+oFv5mDJ/rDH1Yv64Xfms78kfTcnvbtuU27fP8yoQCMoldHRv/9/AgiY3j0kgEQ/DiGnx4esu4jiKIdAGp+/vEP+LH/dhXZ"
    "P8IQPPhH0n0cQij8D7f7CA5h+B/arS/u3do169j4sbr6pfo9S6o298OFvN39D7Xgtqj9hNZHgjhzrx5eYfPmcCh7ecga6xIp"
    "q3zr+fU/QRB87d0nWb1Qvy/T6Tw1M2bUJvnf7+fw1nFNf+h8r3vXujmpHWXu/vgVyLgd3Nrbw7frbn1pOh3EeWaZzacfrGYg"
    "tbW6lWRVY4P08Pr1a+1t6atuv3Yaoftz9uLqtZF93an7jVuvbdj578f+y6upJd7r06vfWKpfNkJZhwkpH4N6YHXQEO3Ll78k"
    "Pbw+xI1LmPSrUUsJv6aGYwXW06e6xc5r51zUSj9+4n62Lmm3+Yr0h2Z/PX9/O48+qe8uF+8Atc3/Y2KZuWFd7iXWJTDHk/VH"
    "9q1n1bed5KHxs9V5eUkeuOK2wmfXVu7znUOrF/ch7DxJWeKGds2jvHYkbTvSuS7spQ//8Xr2TvBw9hf20NASr//13Xr7+l/f"
    "k7fXXk0+PLVKN+HbvYcF49roeWifP2dfvtTRQ8zGKcTNH/HrJbl2j8UV9dWq90dz/Py6ieTlnPt0X6WX1Vey5MePz2F950r1"
    "i0HTTcjea+al1ZlIu+Flar6XTvXk9lpj+8fHx7py2Mj+2p9zxfq5q3Vq/Bh9u1s1/8O4zqWtJKmXxH1uJz55N/HJQ/jy8r0Z"
    "RfbWebo4/PnDrS+obg3ibu3nP2u/5z6fvaRpz+HZp/otMEz2ENaT8b1mOGhvv6elmxnOQ9hMWue7oaXWaxtN6fWpnYLz4OuB"
    "u+Euauw46x+tmzymVtxNHtsKnZv2SnPWf29au2o6PLWAOzxrmHz5whXnpz/alG8vyYeE8/G8+D17+Vc9se568N19Ch/dOvBM"
    "+BQ+hi8vDc4+a8yEl6g0TdolhEx4jS5T8/feLhAstV5veiX/ak6SOx2UWgr40xm5RqD5t1uzzF82do5H869bOpf7ZTNnF7bt"
    "GmXPzXEMHw0tT+vfi3rXu49cTsOZ+ml/ksdzSw/nDdYEv0nqeDV/hI9nOdTLy9W6/lvtVaNtpflUrfxy/s3qWDVNZnbmuHz7"
    "0O23y+itP+/7/ofV1Pn23Iy4dSN+6UyrHXUZz9lZ2h3w+jfGZ107/G5+s5bQ+I/X+kb5057WdP6f/99N813on4cGSDUz1k7o"
    "386ZWV+lL4fxjz+uLSZW2rovv8xX6yw6bJNfXsC3zrdvDbpsfZG345g35+ax9ovtWulD+KhXf0btWfr+1iKFmjz6VuOl7uun"
    "//v/eu26D9nFY01ytsb83XqM89R5+OO6a+s+NV96+H72kN5+uNaFrIHjt04vPPvxr/lkP6nc5NWqPsensPZkfh5D560ZQ+OX"
    "/K7iWWJ0rtjkNc1fvX9/boDoL8pfSz18vzkgv6vc9rV2aPOLFuqspkzjG+cXhZq8ptTFl/eXL7fnlh6+q9vKHZu6l1IXg9i7"
    "ever0PT0px+vjSObb9cCjMvaJ83bn3+cU7/9/qFmi+Gydxgue/j+07P4lH27kNzt7rfu9s6ZhXkmKhIr/fpf38MLTWGdaYqa"
    "+/pxz/+chDcbvNlpHe81WPPqV6928X+eo6fsMc2s4MePeqrOwRvPAUye2thrb53GGedOf4YspBfrz4iF9IL6rd9z9OeGI2s4"
    "WoJlD+DZoW59i5Xj+OKItnvjNd/uY3bTuSuhFz5nj2FUvryQWmbVT72kTrl4veVap7cvL/+ue9yns4jRfVayh+9OlGZc46Cs"
    "QZDZsX043wQaEU0TP/DysKydIzZ486wg1TxfIgN+hm7BST9DTfjk+vcuMttnqA6X55yrvbvr1lkXJsalWC1zaftzET7VWa1V"
    "/RN8DUZ7cTvVBsY6d/CdL5663s3TzfljxqX/7wJ0/PGtdzOGrj3aNIRUM8zzT/pcsxKMZ7DnP4O9/Ln20PD8GeztenHLsZkX"
    "D+7jZWZ//KhVeoI2iyke4qvXNefiATLSL6m9OqFn13vHvEvs9P68coOM7PjcwqLP7mO7Up1ecV/gy5fm5xJX8yVo2So3nZSH"
    "7/FFOeW+5FunCaZ3cevV4ommyJUSawBMvf/vm22PzNn3+Y8fl5wPUU/b83MJbNiGInyyLkEJ3zpPRcuJbX+frkO8NdpSnG83"
    "F2DvB/2ub3b08P381bvct87TLb+O5lr79/qXTdU33L9v7BIc9ubj62+au+i7/qTJ68yc026zUy8O8eumz89/uwRNE+W1iXd+"
    "H+5FSTfHra0bwtc753ONP9tOz7Ladsr3kpnbQl1c6F+wzqVgg3KCq1Dt7Iyh7llmXbt21XS5lGsSzuSJ/XDLr6u2b51eaL3f"
    "ug0sPRMFtT+1K+mW1Ycqe3bbcTZ8k5er/OwSgvW18x4fPdx1pSnQ6TU8/gZivSes3zdYmwnc4HmQ2i/57XjbvSap8+7CE14u"
    "POGZnGs/lJ3Fibevtezh8PH995+uXc4eHS2tFU1esoZJf39fvy34Z+veK/qrVGtKfNLqeAZXx4L/8VqbXJyhzS86c0a67Qje"
    "7c9rjNJn8OUvs3kfS/eXM/r0l2qNNPeX5d86vdX9Xmp2cws3H37Vt06nR119YbafuTTb6ZFt1ufP1MUx30P79BieCcQfPy4J"
    "zS3znNjp9JIP+7It9Q6ohjcaM/wn+MJrmfOYNP6erUcXsB7Df0Ag2DLXOj330uK1L2QLOoP7Hlw6eD5emlWjls+fpatjQfei"
    "Avfly+c/r6nF5amZoZcLH+HZ6kXWR9xwnttezY97t99frHpXn1Fs+NxwdJpmzqN8sR+yTvd6nupwOK+95Pk6681i/wF+e2l2"
    "XE0h9rTnrO3Jy8t9v9qrfPRhUr+fYwA7+kP45Utyv73DzkvSUl7hxZf/U9I5m+NkLy3Z137i7xFaS8n+oV3W7rNm1drJl/K3"
    "0L4XMK81kK4XvUTX6LluO+in6C6k7S89SNaaxZ3e2ebnKfryJbpsnPcL3+IW9yl6dHvhU/QYvnXOxe925str97XL34iShza3"
    "7mJ9gubPd0vdEpgXZmxDX145s/X9vrjwle8hWa0eVm+ddzzcl9fGb2/jGKeWoNW78nsrv2k0AutI54enbtd/e+sZ1rkX7k0J"
    "6flnxO6XL58/3+jda9kvXz4WvuU9lombNRq791eUu7gUDU+5veycQW0vfJ7Xim/Ny7PVs378qMHJlch9/gx1WsnGTyas7uPP"
    "5+zlsfajWDtsfX28SXga+7uaHfoQ1ufKjl5//GieGpd8NXH55UtqPVwMsDq9z9aXL9fibWpbpZnyc3n3o8ZDo6bWOFm848vm"
    "zeDzmrq9xj9o6epaIaWR5bz+1ZXstVCnMR08P//zGfzypcbabf1boV7tw72l5OtuNrT8jx/vE88uYz+mNh5iO0271+RbiYu4"
    "6dFNmTCz7BrTte7Hmp6dH//5DN1e/vsZAc/9PL8/Xyvc3R6th6wJVWI2/57vAw9XOFoDjss16Ny1NqO+Kdyy6m3SOwtDm0KX"
    "+8LzOa3Tu9qZ1JEg2i41Cc/ZjSRuSKYzLjvjtOfm7axReaPszjN0l/D8Pr/p1+XS0/aslmKkX758zhskcbbmzZuwd3VGp9cq"
    "Gl9717w+Zxdl5dfaPvRCol2S/0ys3Y8f5wto9liLaX/8yM6QuVNz0fTPz8/aeSjvzlMN+uvsFtJnjTZcc+7ebePzDNd5Z7T7"
    "u+Wn1id399Dgo3NPr5+5KKy232++ersiNl99fT//LRHxvWE6nC0/awTXEG/nUbWMiBvKammTzxdKqSXFmpS6n29n0u+l4cBf"
    "Kc8Lp/AnM9FpB//ly0P0bk7O4/78/HzNd2+Ki22JywW7GVw7Uefivb/MJNjpna1Dz8P3rYe6z3VKp2dYD+cd5mtVcxu8hTZ5"
    "aA0Ontrs2lr56a6fb507+cjucpLe3x8uK1MDnlaf+Ao3Gv7jc0tqff6rE+sfP8L/Bjvf3y8l2HM/+Oat02qwefEwfOH/vH3Y"
    "A72Pq369Gbzd77TeTzfi3XT3Pqxjr55P8IonrgC9IbgagNPi5+8tT+OS/Pb0/e33X8z4lTMS3jgjN8BYs4aTtzs4Ft+g+x1Y"
    "/slUQb0Ps3JrI7hv4waCbwWc+wIXmHrLrrXCv1+gy+f7SfrL3gd7Px/21fa43mnaO7nin3Xrv6h11suu60T3k1K8w/cXhe9e"
    "eKZAaom19dBphbVvdyfs9aL/+/rWSN3/ltZoTFEdq1WU+dBKrV781gsvHwkf7k9MdZ6v2876sI1vJfVzyXuw/ssZvPgavJ+I"
    "46X+BV2BPeP57Hb/p204lu9HtULKL/LjyPfrD9RTK7UE3ZUQOpsY19j9b+veyDKi7p355ctD9hiFK9fwXh4fOr3PHxr88iV8"
    "6HxN/xno5yejebrHyOBHjCzVoG33nFrZyg2sKM8eiPpINHfsnf4U663j+/JXQ2jpgbqd+w6v6g6/6+yNCjTPYQq4Jna2lTy8"
    "FjUUcX03qwynkZX1SqvTa1UN/lq8oZpfe5L1rxfn48ggC7lbc6+e1Pq87z4/P591j758aYQYlyq7u36fTTb+N13/aY1b7y+M"
    "g7sgWw91FJ3vF3W5Rnsu7qVPQS97cq6GZn9ejcire6u74hprQ7+LtXG8M9LLrGuMjfJdjA3LulmjS+8M0Il3CuVWq1G+uvEe"
    "qBrHE42kmbw57bDufHW41tWFR2Q1wTx21n0Mj9h6F8QjsO6ieDjWxdGHbZ0Nif60bpZEhdUaxVfWNTyIbrXGh0erVTdcWWd9"
    "Q693neYn03p7e3dBe2x0zepVqjfcw2ujQZO7r53fa5HHn3qDnW7bu9CvpkK1as/vtd7Gn/VZ+1N/tnp2UKudvNWFmuqV/mzr"
    "D3VYtt7384l4KvSanas/m+lDpvfuNCwrvVZWCx50vVfpDWn8FyXA9vepiUj4mnrVa93UY9CwP17/U4vj187v/w3UNi1x9s//"
    "+PTpv9NaWehTYvnPvzWPqWNZ2W+fjCSqw327thv+86l29vH969cm3kEUZk+/8VpY6xX91vuUVjWH4Gvu9j591eLYt76eU3qf"
    "cN8NPV4zpOadjsKs9+k3ybIj65PM1DW1MP2a1oEcfz83XRt5PLW/NRS36vL0Jz4Ko996n34jtNTQTFe7JPBW6Ee9T0QUppGv"
    "pb1P13pte4354TF9gqD4eJ+UBk8Q/D4pMJ8g5H2Sbz9B6Puko/8Eg++T4Dpt9D4NOfpP/Vu5FHrq317gp1vpFLnvSNp/gga3"
    "N/T+W+ngCb5rZPSE3OolmunmaT2q/oe0p8HHQr599/k27ejf96KJmBr59x9ok+ovwHcdrCMFl18bveIn8BMUHz/B8fHTf0Ij"
    "qI8YUL/3CfzUj4+foLtU0HxfOSqsxNequnpdCjkXBaERPILHbdGdlmZPjxCctu+1P9mnR2h0ebfqdyPXXeOrbp1cK3l4hHuf"
    "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"
    "6/8AgCIiwQ=="
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
    "missing_requirements": ("Arrange and Fill need {names}, which could not be loaded.",
                             "Reinstall the plugin from Orca Cloud, restart OrcaSlicer and try again."),
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

REQUIREMENTS = (("numpy", "numpy"), ("pillow", "Pillow"))  # (id, name): both rasterise footprints for Arrange and Fill


def requirements(refresh: bool = False) -> list[dict]:
    """What Arrange and Fill need, for the page: [{id, name, status: "ok" | "missing", version, detail}]. Reports what
    _load_deps() found when the module loaded, so it costs nothing; `refresh` (Check again) retries the imports of
    anything missing (they only succeed once the packages are installed, usually after a restart)."""
    global _DEPS
    if refresh and any(isinstance(v, Exception) for v in _DEPS.values()):
        _DEPS = _load_deps()
    rows = []
    for rid, name in REQUIREMENTS:
        found = _DEPS.get(rid)
        ok = isinstance(found, str)
        rows.append({"id": rid, "name": name, "status": "ok" if ok else "missing",
                     "version": found if ok else None, "detail": None if ok else f"{type(found).__name__}: {found}"})
    return rows


def missing_requirements() -> list[str]:
    return [r["name"] for r in requirements() if r["status"] != "ok"]


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
        elif action == "recheck":
            requirements(refresh=True)
            self.sync()
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
               "prefs": self.prefs(), "requirements": requirements()}
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
        missing = missing_requirements()
        if missing:  # the page disables both actions too; this covers an old or stale page
            names = ", ".join(missing)
            raise PlanError(f"Arrange and Fill need {names}, which could not be loaded", "missing_requirements",
                            names=names)
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
