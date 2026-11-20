# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.15.0"
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

import base64
import functools
import importlib
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

np = Image = ImageDraw = None  # numpy and Pillow, bound by _deps(); OrcaSlicer pip-installs them with the plugin
_DEPS = None  # what _load_deps() found; None until first needed. Never loaded at import or while the plugin registers
_DEPS_LOCK = threading.Lock()
REQUIREMENTS = (("numpy", "numpy"), ("pillow", "Pillow"))  # (id, name): both rasterise footprints for Arrange and Fill


def _load_deps() -> dict:
    """Import numpy and Pillow into this module; never raises. When OrcaSlicer's install of them failed (offline,
    pip error) the plugin still loads, Arrange and Fill are refused and the page says why (see requirements()).
    Returns {id: version str, or the import error}. Call through _deps()."""
    global np, Image, ImageDraw
    importlib.invalidate_caches()  # a retry must see packages installed since the last attempt
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


def _deps() -> dict:
    """Load numpy and Pillow once, on first need (a run, the CLI, the page's first sync's worker)."""
    global _DEPS
    with _DEPS_LOCK:
        if _DEPS is None:
            _DEPS = _load_deps()
        return _DEPS


def _missing_error(names: str) -> Exception:
    return PlanError(f"Arrange and Fill need {names}, which could not be loaded", "missing_requirements", names=names)


def _need_deps():
    """Load numpy and Pillow if needed; raise the explained error naming any that are missing."""
    missing = [name for rid, name in REQUIREMENTS if not isinstance(_deps().get(rid), str)]
    if missing:
        raise _missing_error(", ".join(missing))

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
        _need_deps()
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
PANEL_HTML_SHA256 = "ffee90b577b97f748912b9bf5cd1ead4aa34bfee7efa9727fdb63e9a767fef64"
PANEL_HTML = (
    "<!doctype html>\n"
    "<html lang=\"en\">\n"
    "<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>Organizer</title>\n"
    "  <script type=\"module\" crossorigin>function e(e){let t=Object.create(null);for(let n of e.split(`,`))t[n]=1;return e=>e in t}var t={},n=[],r=()=>{},i=()=>!1,a=e=>e.charCodeAt(0)===111&&e.charCodeAt(1)===110&&(e.charCodeAt(2)>122||e.charCodeAt(2)<97),o=e=>e.startsWith(`onUpdate:`),s=Object.assign,c=(e,t)=>{let n=e.indexOf(t);n>-1&&e.splice(n,1)},l=Object.prototype.hasOwnProperty,u=(e,t)=>l.call(e,t),d=Array.isArray,f=e=>x(e)===`[object Map]`,p=e=>x(e)===`[object Set]`,m=e=>x(e)===`[object Date]`,h=e=>typeof e==`function`,g=e=>typeof e==`string`,_=e=>typeof e==`symbol`,v=e=>typeof e==`object`&&!!e,y=e=>(v(e)||h(e))&&h(e.then)&&h(e.catch),b=Object.prototype.toString,x=e=>b.call(e),S=e=>x(e).slice(8,-1),C=e=>x(e)===`[object Object]`,w=e=>g(e)&&e!==`NaN`&&e[0]!==`-`&&``+parseInt(e,10)===e,ee=e(`,key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted`),te=e=>{let t=Object.create(null);return(n=>t[n]||(t[n]=e(n)))},ne=/-\\w/g,T=te(e=>e.replace(ne,e=>e.slice(1).toUpperCase())),E=/\\B([A-Z])/g,D=te(e=>e.replace(E,`-$1`).toLowerCase()),re=te(e=>e.charAt(0).toUpperCase()+e.slice(1)),ie=te(e=>e?`on${re(e)}`:``),ae=(e,t)=>!Object.is(e,t),oe=(e,...t)=>{for(let n=0;n<e.length;n++)e[n](...t)},O=(e,t,n,r=!1)=>{Object.defineProperty(e,t,{configurable:!0,enumerable:!1,writable:r,value:n})},se=e=>{let t=parseFloat(e);return isNaN(t)?e:t},ce=e=>{let t=g(e)?Number(e):NaN;return isNaN(t)?e:t},le,ue=()=>le||=typeof globalThis<`u`?globalThis:typeof self<`u`?self:typeof window<`u`?window:typeof global<`u`?global:{};function de(e){if(d(e)){let t={};for(let n=0;n<e.length;n++){let r=e[n],i=g(r)?he(r):de(r);if(i)for(let e in i)t[e]=i[e]}return t}if(g(e)||v(e))return e}var fe=/;(?![^(]*\\))/g,pe=/:([^]+)/,me=/\"(?:[^\"\\\\]|\\\\[^])*\"|'(?:[^'\\\\]|\\\\[^])*'|\\\\[^]|\\/\\*[^]*?\\*\\//g;function he(e){let t={};return e.replace(me,e=>e.startsWith(`/*`)?``:e).split(fe).forEach(e=>{if(e){let n=e.split(pe);n.length>1&&(t[n[0].trim()]=n[1].trim())}}),t}function ge(e){let t=``;if(g(e))t=e;else if(d(e))for(let n=0;n<e.length;n++){let r=ge(e[n]);r&&(t+=r+` `)}else if(v(e))for(let n in e)e[n]&&(t+=n+` `);return t.trim()}function _e(e){if(!e)return null;let{class:t,style:n}=e;return t&&!g(t)&&(e.class=ge(t)),n&&(e.style=de(n)),e}var ve=e(`itemscope,allowfullscreen,disablepictureinpicture,disableremoteplayback,formnovalidate,ismap,nomodule,novalidate,playsinline,readonly`);function ye(e){return!!e||e===``}function be(e,t,n){if(e.length!==t.length)return!1;let r=!0;for(let i=0;r&&i<e.length;i++)r=we(e[i],t[i],n);return r}function xe(e,t,n){if(e.size!==t.size)return!1;let r=Array.from(t),i=new Uint8Array(r.length);for(let t of e){let e=-1;for(let a=0;a<r.length;a++)if(!i[a]&&we(t,r[a],n)){e=a;break}if(e<0)return!1;i[e]=1}return!0}function Se(e,t,n){let r=f(e),i=f(t);if(r||i||(r=p(e),i=p(t),r||i))return r&&i?xe(e,t,n):!1;if(Object.keys(e).length!==Object.keys(t).length)return!1;for(let r in e){let i=e.hasOwnProperty(r),a=t.hasOwnProperty(r);if(i&&!a||!i&&a||!we(e[r],t[r],n))return!1}return String(e)===String(t)}function Ce(e,t,n,r){n||=[new Map,new Map];let[i,a]=n;if(i.has(e)||a.has(t))return i.get(e)===t&&a.get(t)===e;i.set(e,t),a.set(t,e);let o=r(e,t,n);return i.delete(e),a.delete(t),o}function we(e,t,n){if(e===t)return!0;let r=m(e),i=m(t);return r||i?r&&i?e.getTime()===t.getTime():!1:(r=_(e),i=_(t),r||i?e===t:(r=d(e),i=d(t),r||i?r&&i?Ce(e,t,n,be):!1:(r=v(e),i=v(t),r||i?!r||!i?!1:Ce(e,t,n,Se):String(e)===String(t))))}var Te=e=>!!(e&&e.__v_isRef===!0),k=e=>g(e)?e:e==null?``:d(e)||v(e)&&(e.toString===b||!h(e.toString))?Te(e)?k(e.value):JSON.stringify(e,Ee,2):String(e),Ee=(e,t)=>Te(t)?Ee(e,t.value):f(t)?{[`Map(${t.size})`]:[...t.entries()].reduce((e,[t,n],r)=>(e[De(t,r)+` =>`]=n,e),{})}:p(t)?{[`Set(${t.size})`]:[...t.values()].map(e=>De(e))}:_(t)?De(t):v(t)&&!d(t)&&!C(t)?String(t):t,De=(e,t=``)=>_(e)?`Symbol(${e.description??t})`:e,Oe,ke=class{constructor(e=!1){this.detached=e,this._active=!0,this._on=0,this.effects=[],this.cleanups=[],this._isPaused=!1,this._warnOnRun=!0,this.__v_skip=!0,!e&&Oe&&(Oe.active?(this.parent=Oe,this.index=(Oe.scopes||(Oe.scopes=[])).push(this)-1):(this._active=!1,this._warnOnRun=!1))}get active(){return this._active}pause(){if(this._active){this._isPaused=!0;let e,t;if(this.scopes){let n=this.scopes.slice();for(e=0,t=n.length;e<t;e++)n[e].pause()}for(e=0,t=this.effects.length;e<t;e++)this.effects[e].pause()}}resume(){if(this._active&&this._isPaused){this._isPaused=!1;let e,t;if(this.scopes){let n=this.scopes.slice();for(e=0,t=n.length;e<t;e++)n[e].resume()}let n=this.effects.slice();for(e=0,t=n.length;e<t;e++)n[e].resume()}}run(e){if(this._active){let t=Oe;try{return Oe=this,e()}finally{Oe=t}}}on(){++this._on===1&&(this.prevScope=Oe,Oe=this)}off(){if(this._on>0&&--this._on===0){if(Oe===this)Oe=this.prevScope;else{let e=Oe;for(;e;){if(e.prevScope===this){e.prevScope=this.prevScope;break}e=e.prevScope}}this.prevScope=void 0}}stop(e){if(this._active){this._active=!1;let t,n;for(t=0,n=this.effects.length;t<n;t++)this.effects[t].stop();for(this.effects.length=0,t=0,n=this.cleanups.length;t<n;t++)this.cleanups[t]();if(this.cleanups.length=0,this.scopes){let e=this.scopes.slice();for(t=0,n=e.length;t<n;t++)e[t].stop(!0);this.scopes.length=0}if(!this.detached&&this.parent&&!e){let e=this.parent.scopes.pop();e&&e!==this&&(this.parent.scopes[this.index]=e,e.index=this.index)}this.parent=void 0}}};function Ae(){return Oe}function je(e,t=!1){Oe&&(Oe.active?Oe.cleanups.push(e):e())}var A,Me=new WeakSet,Ne=class{constructor(e){this.fn=e,this.deps=void 0,this.depsTail=void 0,this.flags=5,this.next=void 0,this.cleanup=void 0,this.scheduler=void 0,Oe&&(Oe.active?Oe.effects.push(this):this.flags&=-2)}pause(){this.flags|=64}resume(){this.flags&64&&(this.flags&=-65,Me.has(this)&&(Me.delete(this),this.trigger()))}notify(){this.flags&2&&!(this.flags&32)||this.flags&8||Le(this)}run(){if(!(this.flags&1))return this.fn();this.flags|=2,Xe(this),Be(this);let e=A,t=Ke;A=this,Ke=!0;try{return this.fn()}finally{Ve(this),A=e,Ke=t,this.flags&=-3}}stop(){if(this.flags&1){for(let e=this.deps;e;e=e.nextDep)We(e);this.deps=this.depsTail=void 0,Xe(this),this.onStop&&this.onStop(),this.flags&=-2}}trigger(){this.flags&64?Me.add(this):this.scheduler?this.scheduler():this.runIfDirty()}runIfDirty(){He(this)&&this.run()}get dirty(){return He(this)}},Pe=0,Fe,Ie;function Le(e,t=!1){e.flags|=8,t?(e.next=Ie,Ie=e):(e.next=Fe,Fe=e)}function Re(){Pe++}function ze(){if(--Pe>0)return;if(Ie){let e=Ie;for(Ie=void 0;e;){let t=e.next;e.next=void 0,e.flags&=-9,e=t}}let e;for(;Fe;){let t=Fe;for(Fe=void 0;t;){let n=t.next;if(t.next=void 0,t.flags&=-9,t.flags&1)try{t.trigger()}catch(t){e||=t}t=n}}if(e)throw e}function Be(e){for(let t=e.deps;t;t=t.nextDep)t.version=-1,t.prevActiveLink=t.dep.activeLink,t.dep.activeLink=t}function Ve(e){let t,n=e.depsTail,r=n;for(;r;){let e=r.prevDep;r.version===-1?(r===n&&(n=e),We(r),Ge(r)):t=r,r.dep.activeLink=r.prevActiveLink,r.prevActiveLink=void 0,r=e}e.deps=t,e.depsTail=n}function He(e){for(let t=e.deps;t;t=t.nextDep)if(t.dep.version!==t.version||t.dep.computed&&(Ue(t.dep.computed)||t.dep.version!==t.version))return!0;return!!e._dirty}function Ue(e){if(e.flags&4&&!(e.flags&16)||(e.flags&=-17,e.globalVersion===Ze)||(e.globalVersion=Ze,!e.isSSR&&e.flags&128&&(!e.deps&&!e._dirty||!He(e))))return;e.flags|=2;let t=e.dep,n=A,r=Ke;A=e,Ke=!0;try{Be(e);let n=e.fn(e._value);(t.version===0||ae(n,e._value))&&(e.flags|=128,e._value=n,t.version++)}catch(e){throw t.version++,e}finally{A=n,Ke=r,Ve(e),e.flags&=-3}}function We(e,t=!1){let{dep:n,prevSub:r,nextSub:i}=e;if(r&&(r.nextSub=i,e.prevSub=void 0),i&&(i.prevSub=r,e.nextSub=void 0),n.subs===e&&(n.subs=r,!r&&n.computed)){n.computed.flags&=-5;for(let e=n.computed.deps;e;e=e.nextDep)We(e,!0)}!t&&!--n.sc&&n.map&&n.map.delete(n.key)}function Ge(e){let{prevDep:t,nextDep:n}=e;t&&(t.nextDep=n,e.prevDep=void 0),n&&(n.prevDep=t,e.nextDep=void 0)}var Ke=!0,qe=[];function Je(){qe.push(Ke),Ke=!1}function Ye(){let e=qe.pop();Ke=e===void 0||e}function Xe(e){let{cleanup:t}=e;if(e.cleanup=void 0,t){let e=A;A=void 0;try{t()}finally{A=e}}}var Ze=0,Qe=class{constructor(e,t){this.sub=e,this.dep=t,this.version=t.version,this.nextDep=this.prevDep=this.nextSub=this.prevSub=this.prevActiveLink=void 0}},$e=class{constructor(e){this.computed=e,this.version=0,this.activeLink=void 0,this.subs=void 0,this.map=void 0,this.key=void 0,this.sc=0,this.__v_skip=!0}track(e){if(!A||!Ke||A===this.computed)return;let t=this.activeLink;if(t===void 0||t.sub!==A)t=this.activeLink=new Qe(A,this),A.deps?(t.prevDep=A.depsTail,A.depsTail.nextDep=t,A.depsTail=t):A.deps=A.depsTail=t,et(t);else if(t.version===-1&&(t.version=this.version,t.nextDep)){let e=t.nextDep;e.prevDep=t.prevDep,t.prevDep&&(t.prevDep.nextDep=e),t.prevDep=A.depsTail,t.nextDep=void 0,A.depsTail.nextDep=t,A.depsTail=t,A.deps===t&&(A.deps=e)}return t}trigger(e){this.version++,Ze++,this.notify(e)}notify(e){Re();try{for(let e=this.subs;e;e=e.prevSub)e.sub.notify()&&e.sub.dep.notify()}finally{ze()}}};function et(e){if(e.dep.sc++,e.sub.flags&4){let t=e.dep.computed;if(t&&!e.dep.subs){t.flags|=20;for(let e=t.deps;e;e=e.nextDep)et(e)}let n=e.dep.subs;n!==e&&(e.prevSub=n,n&&(n.nextSub=e)),e.dep.subs=e}}var tt=new WeakMap,nt=Symbol(``),rt=Symbol(``),it=Symbol(``);function at(e,t,n){if(Ke&&A){let t=tt.get(e);t||tt.set(e,t=new Map);let r=t.get(n);r||(t.set(n,r=new $e),r.map=t,r.key=n),r.track()}}function ot(e,t,n,r,i,a){let o=tt.get(e);if(!o){Ze++;return}let s=e=>{e&&e.trigger()};if(Re(),t===`clear`)o.forEach(s);else{let i=d(e),a=i&&w(n);if(i&&n===`length`){let e=Number(r);o.forEach((t,n)=>{(n===`length`||n===it||!_(n)&&n>=e)&&s(t)})}else switch((n!==void 0||o.has(void 0))&&s(o.get(n)),a&&s(o.get(it)),t){case`add`:i?a&&s(o.get(`length`)):(s(o.get(nt)),f(e)&&s(o.get(rt)));break;case`delete`:i||(s(o.get(nt)),f(e)&&s(o.get(rt)));break;case`set`:f(e)&&s(o.get(nt))}}ze()}function st(e){let t=j(e);return t===e||(at(t,`iterate`,it),qt(e))?t:Kt(e)?Gt(e)?t.map(e=>Zt(Xt(e))):t.map(Zt):t.map(Xt)}function ct(e){return at(e=j(e),`iterate`,it),e}function lt(e,t){return Kt(e)?Zt(Gt(e)?Xt(t):t):Xt(t)}var ut={__proto__:null,[Symbol.iterator](){return dt(this,Symbol.iterator,e=>lt(this,e))},concat(...e){return st(this).concat(...e.map(e=>d(e)?st(e):e))},entries(){return dt(this,`entries`,e=>(e[1]=lt(this,e[1]),e))},every(e,t){return mt(this,`every`,e,t,void 0,arguments)},filter(e,t){return mt(this,`filter`,e,t,e=>e.map(e=>lt(this,e)),arguments)},find(e,t){return mt(this,`find`,e,t,e=>lt(this,e),arguments)},findIndex(e,t){return mt(this,`findIndex`,e,t,void 0,arguments)},findLast(e,t){return mt(this,`findLast`,e,t,e=>lt(this,e),arguments)},findLastIndex(e,t){return mt(this,`findLastIndex`,e,t,void 0,arguments)},forEach(e,t){return mt(this,`forEach`,e,t,void 0,arguments)},includes(...e){return gt(this,`includes`,e)},indexOf(...e){return gt(this,`indexOf`,e)},join(e){return st(this).join(e)},lastIndexOf(...e){return gt(this,`lastIndexOf`,e)},map(e,t){return mt(this,`map`,e,t,void 0,arguments)},pop(){return _t(this,`pop`)},push(...e){return _t(this,`push`,e)},reduce(e,...t){return ht(this,`reduce`,e,t)},reduceRight(e,...t){return ht(this,`reduceRight`,e,t)},shift(){return _t(this,`shift`)},some(e,t){return mt(this,`some`,e,t,void 0,arguments)},splice(...e){return _t(this,`splice`,e)},toReversed(){return st(this).toReversed()},toSorted(e){return st(this).toSorted(e)},toSpliced(...e){return st(this).toSpliced(...e)},unshift(...e){return _t(this,`unshift`,e)},values(){return dt(this,`values`,e=>lt(this,e))}};function dt(e,t,n){let r=ct(e),i=r[t],a=i.call(r);if(r!==e&&!qt(e)){if(a[Symbol.toStringTag]===`Array Iterator`){let e=Object.create(Object.getPrototypeOf(a));return e.next=()=>ft(a.next(),n),i!==pt[t]&&Object.defineProperties(e,{return:{get(){let e=a.return;return typeof e==`function`?e.bind(a):e}},throw:{get(){let e=a.throw;return typeof e==`function`?e.bind(a):e}}}),e}a._next=a.next,a.next=()=>ft(a._next(),n)}return a}function ft(e,t){return e.done||(e.value=t(e.value)),e}var pt=Array.prototype;function mt(e,t,n,r,i,a){let o=ct(e),s=o!==e&&!qt(e),c=o[t];if(c!==pt[t]){let t=c.apply(e,a);return s?Xt(t):t}let l=n;o!==e&&(s?l=function(t,r){return n.call(this,lt(e,t),r,e)}:n.length>2&&(l=function(t,r){return n.call(this,t,r,e)}));let u=c.call(o,l,r);return s&&i?i(u):u}function ht(e,t,n,r){let i=ct(e),a=i!==e&&!qt(e),o=n,s=!1;i!==e&&(a?(s=r.length===0,o=function(t,r,i){return s&&(s=!1,t=lt(e,t)),n.call(this,t,lt(e,r),i,e)}):n.length>3&&(o=function(t,r,i){return n.call(this,t,r,i,e)}));let c=i[t](o,...r);return s?lt(e,c):c}function gt(e,t,n){let r=j(e);at(r,`iterate`,it);let i=r[t](...n);return(i===-1||i===!1)&&Jt(n[0])?(n[0]=j(n[0]),r[t](...n)):i}function _t(e,t,n=[]){Je(),Re();let r=j(e)[t].apply(e,n);return ze(),Ye(),r}var vt=e(`__proto__,__v_isRef,__isVue`),yt=new Set(Object.getOwnPropertyNames(Symbol).filter(e=>e!==`arguments`&&e!==`caller`).map(e=>Symbol[e]).filter(_));function bt(e){_(e)||(e=String(e));let t=j(this);return at(t,`has`,e),t.hasOwnProperty(e)}var xt=class{constructor(e=!1,t=!1){this._isReadonly=e,this._isShallow=t}get(e,t,n){if(t===`__v_skip`)return e.__v_skip;let r=this._isReadonly,i=this._isShallow;if(t===`__v_isReactive`)return!r;if(t===`__v_isReadonly`)return r;if(t===`__v_isShallow`)return i;if(t===`__v_raw`)return n===(r?i?zt:Rt:i?Lt:It).get(e)||Object.getPrototypeOf(e)===Object.getPrototypeOf(n)?e:void 0;let a=d(e);if(!r){let e;if(a&&(e=ut[t]))return e;if(t===`hasOwnProperty`)return bt}let o=Reflect.get(e,t,Qt(e)?e:n);if((_(t)?yt.has(t):vt(t))||(r||at(e,`get`,t),i))return o;if(Qt(o)){let e=a&&w(t)?o:o.value;return r&&v(e)?Ut(e):e}return v(o)?r?Ut(o):Vt(o):o}},St=class extends xt{constructor(e=!1){super(!1,e)}set(e,t,n,r){let i=e[t],a=d(e)&&w(t);if(!this._isShallow){let e=Kt(i);if(!qt(n)&&!Kt(n)&&(i=j(i),n=j(n)),!a&&Qt(i)&&!Qt(n))return e||(i.value=n),!0}let o=a?Number(t)<e.length:u(e,t),s=Reflect.set(e,t,n,Qt(e)?e:r);return e===j(r)&&s&&(o?ae(n,i)&&ot(e,`set`,t,n,i):ot(e,`add`,t,n)),s}deleteProperty(e,t){let n=u(e,t),r=e[t],i=Reflect.deleteProperty(e,t);return i&&n&&ot(e,`delete`,t,void 0,r),i}has(e,t){let n=Reflect.has(e,t);return _(t)&&yt.has(t)||at(e,`has`,t),n}ownKeys(e){return at(e,`iterate`,d(e)?`length`:nt),Reflect.ownKeys(e)}},Ct=class extends xt{constructor(e=!1){super(!0,e)}set(e,t){return!0}deleteProperty(e,t){return!0}},wt=new St,Tt=new Ct,Et=new St(!0),Dt=e=>e,Ot=e=>Reflect.getPrototypeOf(e);function kt(e,t,n){return function(...r){let i=this.__v_raw,a=j(i),o=f(a),c=e===`entries`||e===Symbol.iterator&&o,l=e===`keys`&&o,u=i[e](...r),d=n?Dt:t?Zt:Xt;return!t&&at(a,`iterate`,l?rt:nt),s(Object.create(u),{next(){let{value:e,done:t}=u.next();return t?{value:e,done:t}:{value:c?[d(e[0]),d(e[1])]:d(e),done:t}}})}}function At(e){return function(...t){return e===`delete`?!1:e===`clear`?void 0:this}}function jt(e,t){let n={get(n){let r=this.__v_raw,i=j(r),a=j(n);e||(ae(n,a)&&at(i,`get`,n),at(i,`get`,a));let{has:o}=Ot(i),s=t?Dt:e?Zt:Xt;if(o.call(i,n))return s(r.get(n));if(o.call(i,a))return s(r.get(a));r!==i&&r.get(n)},get size(){let t=this.__v_raw;return!e&&at(j(t),`iterate`,nt),t.size},has(t){let n=this.__v_raw,r=j(n),i=j(t);return e||(ae(t,i)&&at(r,`has`,t),at(r,`has`,i)),t===i?n.has(t):n.has(t)||n.has(i)},forEach(n,r){let i=this,a=i.__v_raw,o=j(a),s=t?Dt:e?Zt:Xt;return!e&&at(o,`iterate`,nt),a.forEach((e,t)=>n.call(r,s(e),s(t),i))}};return s(n,e?{add:At(`add`),set:At(`set`),delete:At(`delete`),clear:At(`clear`)}:{add(e){let n=j(this),r=Ot(n),i=j(e),a=!t&&!qt(e)&&!Kt(e)?i:e;return r.has.call(n,a)||ae(e,a)&&r.has.call(n,e)||ae(i,a)&&r.has.call(n,i)||(n.add(a),ot(n,`add`,a,a)),this},set(e,n){t||qt(n)||Kt(n)||(n=j(n));let r=j(this),{has:i,get:a}=Ot(r),o=i.call(r,e);o||=(e=j(e),i.call(r,e));let s=a.call(r,e);return r.set(e,n),o?ae(n,s)&&ot(r,`set`,e,n,s):ot(r,`add`,e,n),this},delete(e){let t=j(this),{has:n,get:r}=Ot(t),i=n.call(t,e);i||=(e=j(e),n.call(t,e));let a=r?r.call(t,e):void 0,o=t.delete(e);return i&&ot(t,`delete`,e,void 0,a),o},clear(){let e=j(this),t=e.size!==0,n=e.clear();return t&&ot(e,`clear`,void 0,void 0,void 0),n}}),[`keys`,`values`,`entries`,Symbol.iterator].forEach(r=>{n[r]=kt(r,e,t)}),n}function Mt(e,t){let n=jt(e,t);return(r,i,a)=>i===`__v_isReactive`?!e:i===`__v_isReadonly`?e:i===`__v_isShallow`?t:i===`__v_raw`?r:Reflect.get(u(n,i)&&i in r?n:r,i,a)}var Nt={get:Mt(!1,!1)},Pt={get:Mt(!1,!0)},Ft={get:Mt(!0,!1)},It=new WeakMap,Lt=new WeakMap,Rt=new WeakMap,zt=new WeakMap;function Bt(e){switch(e){case`Object`:case`Array`:return 1;case`Map`:case`Set`:case`WeakMap`:case`WeakSet`:return 2;default:return 0}}function Vt(e){return Kt(e)?e:Wt(e,!1,wt,Nt,It)}function Ht(e){return Wt(e,!1,Et,Pt,Lt)}function Ut(e){return Wt(e,!0,Tt,Ft,Rt)}function Wt(e,t,n,r,i){if(!v(e)||e.__v_raw&&!(t&&e.__v_isReactive)||e.__v_skip||!Object.isExtensible(e))return e;let a=i.get(e);if(a)return a;let o=Bt(S(e));if(o===0)return e;let s=new Proxy(e,o===2?r:n);return i.set(e,s),s}function Gt(e){return Kt(e)?Gt(e.__v_raw):!!(e&&e.__v_isReactive)}function Kt(e){return!!(e&&e.__v_isReadonly)}function qt(e){return!!(e&&e.__v_isShallow)}function Jt(e){return e?!!e.__v_raw:!1}function j(e){let t=e&&e.__v_raw;return t?j(t):e}function Yt(e){return!u(e,`__v_skip`)&&Object.isExtensible(e)&&O(e,`__v_skip`,!0),e}var Xt=e=>v(e)?Vt(e):e,Zt=e=>v(e)?Ut(e):e;function Qt(e){return e?e.__v_isRef===!0:!1}function $t(e){return en(e,!1)}function en(e,t){return Qt(e)?e:new tn(e,t)}var tn=class{constructor(e,t){this.dep=new $e,this.__v_isRef=!0,this.__v_isShallow=!1,this._rawValue=t?e:j(e),this._value=t?e:Xt(e),this.__v_isShallow=t}get value(){return this.dep.track(),this._value}set value(e){let t=this._rawValue,n=this.__v_isShallow||qt(e)||Kt(e);e=n?e:j(e),ae(e,t)&&(this._rawValue=e,this._value=n?e:Xt(e),this.dep.trigger())}};function M(e){return Qt(e)?e.value:e}var nn={get:(e,t,n)=>t===`__v_raw`?e:M(Reflect.get(e,t,n)),set:(e,t,n,r)=>{let i=e[t];return Qt(i)&&!Qt(n)?(i.value=n,!0):Reflect.set(e,t,n,r)}};function rn(e){return Gt(e)?e:new Proxy(e,nn)}var an=class{constructor(e,t,n){this.fn=e,this.setter=t,this._value=void 0,this.dep=new $e(this),this.__v_isRef=!0,this.deps=void 0,this.depsTail=void 0,this.flags=16,this.globalVersion=Ze-1,this.next=void 0,this.effect=this,this.__v_isReadonly=!t,this.isSSR=n}notify(){if(this.flags|=16,!(this.flags&8)&&A!==this)return Le(this,!0),!0}get value(){let e=this.dep.track();return Ue(this),e&&(e.version=this.dep.version),this._value}set value(e){this.setter&&this.setter(e)}};function on(e,t,n=!1){let r,i;return h(e)?r=e:(r=e.get,i=e.set),new an(r,i,n)}var sn={},cn=new WeakMap,ln=void 0;function un(e,t=!1,n=ln){if(n){let t=cn.get(n);t||cn.set(n,t=[]),t.push(e)}}function dn(e,n,i=t){let{immediate:a,deep:o,once:s,scheduler:l,augmentJob:u,call:f}=i,p=e=>o?e:qt(e)||o===!1||o===0?fn(e,1):fn(e),m,g,_,v,y=!1,b=!1;if(Qt(e)?(g=()=>e.value,y=qt(e)):Gt(e)?(g=()=>p(e),y=!0):d(e)?(b=!0,y=e.some(e=>Gt(e)||qt(e)),g=()=>e.map(e=>{if(Qt(e))return e.value;if(Gt(e))return p(e);if(h(e))return f?f(e,2):e()})):g=h(e)?n?f?()=>f(e,2):e:()=>{if(_){Je();try{_()}finally{Ye()}}let t=ln;ln=m;try{return f?f(e,3,[v]):e(v)}finally{ln=t}}:r,n&&o){let e=g,t=o===!0?1/0:o;g=()=>fn(e(),t)}let x=Ae(),S=()=>{m.stop(),x&&x.active&&c(x.effects,m)};if(s&&n){let e=n;n=(...t)=>{try{return e(...t)}finally{S()}}}let C=b?Array(e.length).fill(sn):sn,w=e=>{if(m.flags&1&&(m.dirty||e)){if(n){let t=m.run();if(e||o||y||(b?t.some((e,t)=>ae(e,C[t])):ae(t,C))){let e=ln;Je();try{_&&_(),ln=m;let e=[t,C===sn?void 0:b&&C[0]===sn?[]:C,v];C=t,f?f(n,3,e):n(...e)}finally{ln=e,Ye()}}}else m.run()}};return u&&u(w),m=new Ne(g),m.scheduler=l?()=>l(w,!1):w,v=e=>un(e,!1,m),_=m.onStop=()=>{let e=cn.get(m);if(e){if(f)f(e,4);else for(let t of e)t();cn.delete(m)}},n?a?w(!0):C=m.run():l?l(w.bind(null,!0),!0):m.run(),S.pause=m.pause.bind(m),S.resume=m.resume.bind(m),S.stop=S,S}function fn(e,t=1/0,n){if(t<=0||!v(e)||e.__v_skip||(n||=new Map,(n.get(e)||0)>=t))return e;if(n.set(e,t),t--,Qt(e))fn(e.value,t,n);else if(d(e))for(let r=0;r<e.length;r++)fn(e[r],t,n);else if(p(e)||f(e))e.forEach(e=>{fn(e,t,n)});else if(C(e)){for(let r in e)fn(e[r],t,n);for(let r of Object.getOwnPropertySymbols(e))Object.prototype.propertyIsEnumerable.call(e,r)&&fn(e[r],t,n)}return e}function pn(e,t,n,r){try{return r?e(...r):e()}catch(e){hn(e,t,n)}}function mn(e,t,n,r){if(h(e)){let i=pn(e,t,n,r);return i&&y(i)&&i.catch(e=>{hn(e,t,n,void 0,!0)}),i}if(d(e)){let i=[];for(let a=0;a<e.length;a++)i.push(mn(e[a],t,n,r));return i}}function hn(e,n,r,i=!0,a=!1){let o=n?n.vnode:null,{errorHandler:s,throwUnhandledErrorInProduction:c}=n&&n.appContext.config||t;if(n){let t=n.parent,i=n.proxy,o=`https://vuejs.org/error-reference/#runtime-${r}`;for(;t;){let n=t.ec;if(n&&!t.isUnmounted&&(!a||t.scope.active)){for(let t=0;t<n.length;t++)if(n[t](e,i,o)===!1)return}t=t.parent}if(s){Je(),pn(s,null,10,[e,i,o]),Ye();return}}gn(e,r,o,i,c)}function gn(e,t,n,r=!0,i=!1){if(i)throw e;console.error(e)}var _n=[],vn=-1,yn=[],bn=null,xn=0,Sn=Promise.resolve(),Cn=null;function wn(e){let t=Cn||Sn;return e?t.then(this?e.bind(this):e):t}function Tn(e){let t=vn+1,n=_n.length;for(;t<n;){let r=t+n>>>1,i=_n[r],a=Nn(i);a<e||a===e&&i.flags&2?t=r+1:n=r}return t}function En(e){if(!(e.flags&1)){let t=Nn(e),n=_n[_n.length-1];!n||!(e.flags&2)&&t>=Nn(n)?_n.push(e):_n.splice(Tn(t),0,e),e.flags|=1,Dn()}}function Dn(){Cn||=Sn.then(Pn)}function On(e){if(!d(e))bn&&e.id===-1?bn.splice(xn+1,0,e):e.flags&1||(yn.push(e),e.flags|=1);else for(let t=0;t<e.length;t++)yn.push(e[t]);Dn()}function kn(e,t,n,r=vn+1){for(;r<_n.length;r++){let n=_n[r];if(n&&n.flags&2){if(e&&n.id!==e.uid||t!==void 0&&!(n.id>=t))continue;_n.splice(r,1),r=e?r-1:vn,n.flags&4&&(n.flags&=-2),n(),n.flags&4||(n.flags&=-2)}}}function An(e,t,n=0){if(yn.length){let e=yn,r=0;if(t){e=[];for(let i=0;i<yn.length;i++){let a=yn[i];(a.i===void 0?t.has(a):!a.i||a.i.uid<n)?yn[r++]=a:e.push(a)}}let i=[...new Set(e)].sort((e,t)=>Nn(e)-Nn(t));if(yn.length=r,bn){for(let e=0;e<i.length;e++)bn.push(i[e]);return}for(bn=i,xn=0;xn<bn.length;xn++){let e=bn[xn];e.flags&4&&(e.flags&=-2),e.flags&8||e(),e.flags&=-2}bn=null,xn=0}}var jn=!1;function Mn(e,t){if(jn){e();return}let n=t===void 0?void 0:new Set(yn);e(),jn=!0;try{kn(void 0,t),An(void 0,n,t)}finally{jn=!1}}var Nn=e=>e.id==null?e.flags&2?-1:1/0:e.id;function Pn(e){try{for(vn=0;vn<_n.length;vn++){let e=_n[vn];e&&!(e.flags&8)&&(e.flags&4&&(e.flags&=-2),pn(e,e.i,e.i?15:14),e.flags&4||(e.flags&=-2))}}finally{for(;vn<_n.length;vn++){let e=_n[vn];e&&(e.flags&=-2)}vn=-1,_n.length=0,An(e),Cn=null,(_n.length||yn.length)&&Pn(e)}}var Fn=null,In=null;function Ln(e){let t=Fn;return Fn=e,In=e&&e.type.__scopeId||null,t}function N(e,t=Fn,n){if(!t||e._n)return e;let r=(...n)=>{r._d&&ga(-1);let i=Ln(t),a=fa.length,o;try{o=e(...n)}finally{for(let e=fa.length;e>a;e--)ma();Ln(i),r._d&&ga(1)}return o};return r._n=!0,r._c=!0,r._d=!0,r}function Rn(e,n){if(Fn===null)return e;let r=Ya(Fn),i=e.dirs||=[];for(let e=0;e<n.length;e++){let[a,o,s,c=t]=n[e];a&&(h(a)&&(a={mounted:a,updated:a}),a.deep&&fn(o),i.push({dir:a,instance:r,value:o,oldValue:void 0,arg:s,modifiers:c}))}return e}function zn(e,t,n,r){let i=e;if(r===`beforeMount`)e._mount=e;else if(r===`beforeUpdate`){let n=t&&t._mount;n&&(n._latest=e,e._mount=n)}else if(r===`mounted`){let t=e._latest;e._mount=e._latest=void 0,t&&(i=t,t._mount=void 0)}let a=i.dirs,o=t&&t.dirs;for(let c=0;c<a.length;c++){let l=a[c];o?l.oldValue=o[c].value:r===`mounted`&&l.oldValue!==void 0&&(l=s({},l,{oldValue:void 0}));let u=l.dir[r];u&&(Je(),mn(u,n,8,[e.el,l,i,t]),Ye())}}function Bn(e,t){if(Pa){let n=Pa.provides,r=Pa.parent&&Pa.parent.provides;r===n&&(n=Pa.provides=Object.create(r)),n[e]=t}}function Vn(e,t,n=!1){let r=Fa();if(r||_i){let i=_i?_i._context.provides:r?r.parent==null||r.ce?r.vnode.appContext&&r.vnode.appContext.provides:r.parent.provides:void 0;if(i&&e in i)return i[e];if(arguments.length>1)return n&&h(t)?t.call(r&&r.proxy):t}}var Hn=Symbol.for(`v-scx`),Un=()=>Vn(Hn);function Wn(e,t){return Kn(e,null,t)}function Gn(e,t,n){return Kn(e,t,n)}function Kn(e,n,i=t){let{immediate:a,deep:o,flush:c,once:l}=i,u=s({},i),d=n&&a||!n&&c!==`post`,f;if(Va){if(c===`sync`){let e=Un();f=e.__watcherHandles||=[]}else if(!d){let e=()=>{};return e.stop=r,e.resume=r,e.pause=r,e}}let p=Pa;u.call=(e,t,n)=>mn(e,p,t,n);let m=!1;c===`post`?u.scheduler=e=>{e.i=p,Yi(e,p&&p.suspense)}:c!==`sync`&&(m=!0,u.scheduler=(e,t)=>{t?e():En(e)}),u.augmentJob=e=>{n&&(e.flags|=4),m&&(e.flags|=2,p&&(e.id=p.uid,e.i=p))};let h=dn(e,n,u);return Va&&(f?f.push(h):d&&h()),h}function qn(e,t,n){let r=this.proxy,i=g(e)?e.includes(`.`)?Jn(r,e):()=>r[e]:e.bind(r,r),a;h(t)?a=t:(a=t.handler,n=t);let o=Ra(this),s=Kn(i,a.bind(r),n);return o(),s}function Jn(e,t){let n=t.split(`.`);return()=>{let t=e;for(let e=0;e<n.length&&t;e++)t=t[n[e]];return t}}var Yn=Symbol(`_vte`),Xn=e=>e.__isTeleport,Zn=Symbol(`_leaveCb`),Qn=Symbol(`_enterCb`);function $n(){let e={isMounted:!1,isLeaving:!1,isUnmounting:!1,leavingVNodes:new Map};return Dr(()=>{e.isMounted=!0}),Ar(()=>{e.isUnmounting=!0}),e}var er=[Function,Array],tr={mode:String,appear:Boolean,persisted:Boolean,onBeforeEnter:er,onEnter:er,onAfterEnter:er,onEnterCancelled:er,onBeforeLeave:er,onLeave:er,onAfterLeave:er,onLeaveCancelled:er,onBeforeAppear:er,onAppear:er,onAfterAppear:er,onAppearCancelled:er},nr={name:`BaseTransition`,props:tr,setup(e,{slots:t}){let n=Fa(),r=$n();return()=>{let i=t.default&&ur(t.default(),!0),a=i&&i.length?rr(i):n.subTree?H():void 0;if(!a)return;let o=j(e),{mode:s}=o;if(r.isLeaving)return sr(a);let c=cr(a);if(!c)return sr(a);let l=or(c,o,r,n);c.type!==ua&&lr(c,l);let u=n.subTree&&cr(n.subTree);if(u&&u.type!==ua&&!ya(u,c)&&dr(n.subTree)){let e=or(u,o,r,n);if(lr(u,e),s===`out-in`&&c.type!==ua)return r.isLeaving=!0,e.afterLeave=()=>{if(r.isLeaving=!1,!(n.job.flags&8)){if(n.effect.flags&2){let e=()=>{n.job.flags&8||n.update()};e.i=n,En(e)}else n.update()}delete e.afterLeave,u=void 0},sr(a);s===`in-out`&&c.type!==ua?e.delayLeave=(e,t,n)=>{let i=ar(r,u);i[String(u.key)]=u,e[Zn]=()=>{t(),e[Zn]=void 0,delete l.delayedLeave,u=void 0},l.delayedLeave=()=>{n(),delete l.delayedLeave,u=void 0}}:u=void 0}else u&&=void 0;return a}}};function rr(e){let t=e[0];if(e.length>1){for(let n of e)if(n.type!==ua){t=n;break}}return t}var ir=nr;function ar(e,t){let{leavingVNodes:n}=e,r=n.get(t.type);return r||(r=Object.create(null),n.set(t.type,r)),r}function or(e,t,n,r){let{appear:i,mode:a,persisted:o=!1,onBeforeEnter:s,onEnter:c,onAfterEnter:l,onEnterCancelled:u,onBeforeLeave:f,onLeave:p,onAfterLeave:m,onLeaveCancelled:h,onBeforeAppear:g,onAppear:_,onAfterAppear:v,onAppearCancelled:y}=t,b=String(e.key),x=ar(n,e),S=(e,t)=>{e&&mn(e,r,9,t)},C=(e,t)=>{let n=t[1];S(e,t),d(e)?e.every(e=>e.length<=1)&&n():e.length<=1&&n()},w={mode:a,persisted:o,beforeEnter(t){let r=s;if(!n.isMounted){if(i)r=g||s;else return}t[Zn]&&t[Zn](!0);let a=x[b];if(a&&ya(e,a)){let e=a.el[Zn];if(!e){let t=dr(a);e=t&&t.el[Zn]}e&&e()}S(r,[t])},enter(t){if(x[b]===e)return;let r=c,a=l,o=u;if(!n.isMounted){if(i)r=_||c,a=v||l,o=y||u;else return}let s=!1;t[Qn]=e=>{s||(s=!0,S(e?o:a,[t]),w.delayedLeave&&w.delayedLeave(),t[Qn]=void 0)};let d=t[Qn].bind(null,!1);r?C(r,[t,d]):d()},leave(t,r){let i=String(e.key);if(t[Qn]&&t[Qn](!0),n.isUnmounting)return r();S(f,[t]);let a=!1;t[Zn]=n=>{a||(a=!0,r(),S(n?h:m,[t]),t[Zn]=void 0,x[i]===e&&delete x[i])};let o=t[Zn].bind(null,!1);x[i]=e,p?C(p,[t,o]):o()},clone(e){let i=or(e,t,n,r);return i.afterLeave=()=>{w.afterLeave&&w.afterLeave()},i.delayLeave=(e,t,n)=>{w.delayLeave?w.delayLeave(e,t,n):n()},i.delayedLeave=()=>{w.delayedLeave&&w.delayedLeave()},i}};return w}function sr(e){if(yr(e))return e=wa(e),e.children=null,e}function cr(e){if(!yr(e))return Xn(e.type)&&e.children?rr(e.children):e;if(e.component)return e.component.subTree;let{shapeFlag:t,children:n}=e;if(n){if(t&16)return n[0];if(t&32&&h(n.default))return n.default()}}function lr(e,t){if(e.shapeFlag&6&&e.component){e.transition=t;let n=e.component.subTree;lr(Xn(n.type)&&cr(n)||n,t)}else if(e.shapeFlag&128){let n=e.suspense;if(n){let e=n.isHydrating?n.pendingBranch:n.activeBranch;e&&lr(e,t.clone(e))}else e.ssContent.transition=t.clone(e.ssContent),e.ssFallback.transition=t.clone(e.ssFallback)}else if(e.transition=t,e.type===I&&d(e.children)){let n=wi(e.children,!1);n&&lr(n,t)}}function ur(e,t=!1,n){let r=[],i=0;for(let a=0;a<e.length;a++){let o=e[a],s=n==null?o.key:String(n)+String(o.key==null?a:o.key);o.type===I?(o.patchFlag&128&&i++,r=r.concat(ur(o.children,t,s))):(t||o.type!==ua)&&r.push(s==null?o:wa(o,{key:s}))}if(i>1)for(let e=0;e<r.length;e++)r[e].patchFlag=-2;return r}function dr(e){if(e.component)return dr(e.component.subTree);if(e.type===I){let t=d(e.children)&&wi(e.children,!1);return t?dr(t):void 0}if(e.type!==ua)return e}function P(e,t){return h(e)?s({name:e.name},t,{setup:e}):e}function fr(){let e=Fa();return e?(e.appContext.config.idPrefix||`v`)+`-`+e.ids[0]+e.ids[1]++:``}function pr(e){e.ids=[e.ids[0]+e.ids[2]+++`-`,0,0]}function mr(e,t){let n;return!!((n=Object.getOwnPropertyDescriptor(e,t))&&!n.configurable)}var hr=new WeakMap;function gr(e,n,r,a,o=!1){if(d(e)){e.forEach((e,t)=>gr(e,n&&(d(n)?n[t]:n),r,a,o));return}if(vr(a)&&!o){a.shapeFlag&512&&a.type.__asyncResolved&&a.component.subTree.component&&gr(e,n,r,a.component.subTree);return}let s=a.shapeFlag&4?Ya(a.component):a.el,l=o?null:s,{i:f,r:p}=e,m=n&&n.r,_=f.refs===t?f.refs={}:f.refs,v=f.setupState,y=j(v),b=v===t?i:e=>!mr(_,e)&&u(y,e),x=(e,t)=>!(t&&mr(_,t));if(m!=null&&m!==p){if(_r(n),g(m))_[m]=null,b(m)&&(v[m]=null);else if(Qt(m)){let e=n;x(m,e.k)&&(m.value=null),e.k&&(_[e.k]=null)}}if(h(p))pn(p,f,12,[l,_]);else{let t=g(p),n=Qt(p);if(t||n){let i=()=>{if(e.f){let n=t?b(p)?v[p]:_[p]:x(p)||!e.k?p.value:_[e.k];if(o)d(n)&&c(n,s);else if(d(n))n.includes(s)||n.push(s);else if(t)_[p]=[s],b(p)&&(v[p]=_[p]);else{let t=[s];x(p,e.k)&&(p.value=t),e.k&&(_[e.k]=t)}}else t?(_[p]=l,b(p)&&(v[p]=l)):n&&(x(p,e.k)&&(p.value=l),e.k&&(_[e.k]=l))};if(l){let t=()=>{i(),hr.delete(e)};t.id=-1,hr.set(e,t),Yi(t,r)}else _r(e),i()}}}function _r(e){let t=hr.get(e);t&&(t.flags|=8,hr.delete(e))}ue().requestIdleCallback,ue().cancelIdleCallback;var vr=e=>!!e.type.__asyncLoader,yr=e=>e.type.__isKeepAlive;function br(e,t){Sr(e,`a`,t)}function xr(e,t){Sr(e,`da`,t)}function Sr(e,t,n=Pa){let r=e.__wdc||=()=>{let t=n;for(;t;){if(t.isDeactivated)return;t=t.parent}return e()};if(wr(t,r,n),n){let e=n.parent;for(;e&&e.parent;)yr(e.parent.vnode)&&Cr(r,t,n,e),e=e.parent}}function Cr(e,t,n,r){let i=wr(t,e,r,!0),a=wr(`um`,()=>c(r[t],i),n);a.flags=16}function wr(e,t,n=Pa,r=!1){if(n){let i=n[e]||(n[e]=[]),a=t.__weh||=(...r)=>{Je();let i=Ra(n),a=mn(t,n,e,r);return i(),Ye(),a};return r?i.unshift(a):i.push(a),a}}var Tr=e=>(t,n=Pa)=>{Va&&e!==`sp`||wr(e,(...e)=>t(...e),n)},Er=Tr(`bm`),Dr=Tr(`m`),Or=Tr(`bu`),kr=Tr(`u`),Ar=Tr(`bum`),jr=Tr(`um`),Mr=Tr(`sp`),Nr=Tr(`rtg`),Pr=Tr(`rtc`);function Fr(e,t=Pa){wr(`ec`,e,t)}var Ir=`components`,Lr=`directives`;function Rr(e,t){return Hr(Ir,e,!0,t)||e}var zr=Symbol.for(`v-ndc`);function Br(e){return g(e)?Hr(Ir,e,!1)||e:e||zr}function Vr(e){return Hr(Lr,e)}function Hr(e,t,n=!0,r=!1){let i=Fn||Pa;if(i){let n=i.type;if(e===Ir){let e=Xa(n,!1);if(e&&(e===t||e===T(t)||e===re(T(t))))return n}let a=Ur(i[e]||n[e],t)||Ur(i.appContext[e],t);return!a&&r?n:a}}function Ur(e,t){return e&&(e[t]||e[T(t)]||e[re(T(t))])}function Wr(e,t,n,r){let i,a=n&&n[r],o=d(e);if(o||g(e)){let n=o&&Gt(e),r=!1,s=!1;n&&(r=!qt(e),s=Kt(e),e=ct(e)),i=Array(e.length);for(let n=0,o=e.length;n<o;n++)i[n]=t(r?s?Zt(Xt(e[n])):Xt(e[n]):e[n],n,void 0,a&&a[n])}else if(typeof e==`number`){if(!Number.isInteger(e)||e<0)i=[];else{i=Array(e);for(let n=0;n<e;n++)i[n]=t(n+1,n,void 0,a&&a[n])}}else if(v(e)){if(e[Symbol.iterator])i=Array.from(e,(e,n)=>t(e,n,void 0,a&&a[n]));else{let n=Object.keys(e);i=Array(n.length);for(let r=0,o=n.length;r<o;r++){let o=n[r];i[r]=t(e[o],o,r,a&&a[r])}}}else i=[];return n&&(n[r]=i),i}function Gr(e,t){for(let n=0;n<t.length;n++){let r=t[n];if(d(r))for(let t=0;t<r.length;t++)e[r[t].name]=r[t].fn;else r&&(e[r.name]=r.key?(...e)=>{let t=r.fn(...e);return t&&(t.key=r.key),t}:r.fn)}return e}function F(e,t,n,r,i,a){if(n??={},Fn.ce||Fn.parent&&vr(Fn.parent)&&Fn.parent.ce){let e=a!=null&&n.key==null?s({},n,{key:a}):n,i=Object.keys(e).length>0;return t!==\"default\"&&(e.name=t),L(),z(I,null,[V(`slot`,e,r&&r())],i?-2:64)}let o=e[t];o&&o._c&&(o._d=!1);let c=fa.length;L();let l;try{let i=o&&Kr(o(n)),s=n.key||a||i&&i.key;l=z(I,{key:(s&&!_(s)?s:`_${t}`)+(!i&&r?`_fb`:``)},i||(r?r():[]),i&&e._===1?64:-2)}catch(e){for(let e=fa.length;e>c;e--)ma();throw e}finally{o&&o._c&&(o._d=!0)}return!i&&l.scopeId&&(l.slotScopeIds=[l.scopeId+`-s`]),l}function Kr(e){return e.some(e=>!va(e)||!(e.type===ua||e.type===I&&!Kr(e.children)))?e:null}function qr(e,t){let n={};for(let r in e)n[t&&/[A-Z]/.test(r)?`on:${r}`:ie(r)]=e[r];return n}var Jr=e=>e?Ba(e)?Ya(e):Jr(e.parent):null,Yr=s(Object.create(null),{$:e=>e,$el:e=>e.vnode.el,$data:e=>e.data,$props:e=>e.props,$attrs:e=>e.attrs,$slots:e=>e.slots,$refs:e=>e.refs,$parent:e=>Jr(e.parent),$root:e=>Jr(e.root),$host:e=>e.ce,$emit:e=>e.emit,$options:e=>ii(e),$forceUpdate:e=>e.f||=()=>{En(e.update)},$nextTick:e=>e.n||=wn.bind(e.proxy),$watch:e=>qn.bind(e)}),Xr=(e,n)=>e!==t&&!e.__isScriptSetup&&u(e,n),Zr={get({_:e},n){if(n===`__v_skip`)return!0;let{ctx:r,setupState:i,data:a,props:o,accessCache:s,type:c,appContext:l}=e;if(n[0]!==`$`){let e=s[n];if(e!==void 0)switch(e){case 1:return i[n];case 2:return a[n];case 4:return r[n];case 3:return o[n]}else if(Xr(i,n))return s[n]=1,i[n];else if(a!==t&&u(a,n))return s[n]=2,a[n];else if(u(o,n))return s[n]=3,o[n];else if(r!==t&&u(r,n))return s[n]=4,r[n];else $r&&(s[n]=0)}let d=Yr[n],f,p;if(d)return n===`$attrs`&&at(e.attrs,`get`,``),d(e);if((f=c.__cssModules)&&(f=f[n]))return f;if(r!==t&&u(r,n))return s[n]=4,r[n];if(p=l.config.globalProperties,u(p,n))return p[n]},set({_:e},n,r){let{data:i,setupState:a,ctx:o}=e;return Xr(a,n)?(a[n]=r,!0):i!==t&&u(i,n)?(i[n]=r,!0):u(e.props,n)||n[0]===`$`&&n.slice(1)in e?!1:(o[n]=r,!0)},has({_:{data:e,setupState:n,accessCache:r,ctx:i,appContext:a,props:o,type:s}},c){let l;return!!(r[c]||e!==t&&c[0]!==`$`&&u(e,c)||Xr(n,c)||u(o,c)||u(i,c)||u(Yr,c)||u(a.config.globalProperties,c)||(l=s.__cssModules)&&l[c])},defineProperty(e,t,n){return n.get==null?u(n,`value`)&&this.set(e,t,n.value,null):e._.accessCache[t]=0,Reflect.defineProperty(e,t,n)}};function Qr(e){return d(e)?e.reduce((e,t)=>(e[t]=null,e),{}):e}var $r=!0;function ei(e){let t=ii(e),n=e.proxy,i=e.ctx;$r=!1,t.beforeCreate&&ni(t.beforeCreate,e,`bc`);let{data:a,computed:o,methods:s,watch:c,provide:l,inject:u,created:f,beforeMount:p,mounted:m,beforeUpdate:g,updated:_,activated:y,deactivated:b,beforeDestroy:x,beforeUnmount:S,destroyed:C,unmounted:w,render:ee,renderTracked:te,renderTriggered:ne,errorCaptured:T,serverPrefetch:E,expose:D,inheritAttrs:re,components:ie,directives:ae,filters:oe}=t;if(u&&ti(u,i,null),s)for(let e in s){let t=s[e];h(t)&&(i[e]=t.bind(n))}if(a){let t=a.call(n,n);v(t)&&(e.data=Vt(t))}if($r=!0,o)for(let e in o){let t=o[e],a=W({get:h(t)?t.bind(n,n):h(t.get)?t.get.bind(n,n):r,set:!h(t)&&h(t.set)?t.set.bind(n):r});Object.defineProperty(i,e,{enumerable:!0,configurable:!0,get:()=>a.value,set:e=>a.value=e})}if(c)for(let e in c)ri(c[e],i,n,e);if(l){let e=h(l)?l.call(n):l;Reflect.ownKeys(e).forEach(t=>{Bn(t,e[t])})}f&&ni(f,e,`c`);function O(e,t){d(t)?t.forEach(t=>e(t.bind(n))):t&&e(t.bind(n))}if(O(Er,p),O(Dr,m),O(Or,g),O(kr,_),O(br,y),O(xr,b),O(Fr,T),O(Pr,te),O(Nr,ne),O(Ar,S),O(jr,w),O(Mr,E),d(D)){if(D.length){let t=e.exposed||={};D.forEach(e=>{Object.defineProperty(t,e,{get:()=>n[e],set:t=>n[e]=t,enumerable:!0})})}else e.exposed||={}}ee&&e.render===r&&(e.render=ee),re!=null&&(e.inheritAttrs=re),ie&&(e.components=ie),ae&&(e.directives=ae),E&&pr(e)}function ti(e,t,n=r){d(e)&&(e=li(e));for(let n in e){let r=e[n],i;i=v(r)?`default`in r?Vn(r.from||n,r.default,!0):Vn(r.from||n):Vn(r),Qt(i)?Object.defineProperty(t,n,{enumerable:!0,configurable:!0,get:()=>i.value,set:e=>i.value=e}):t[n]=i}}function ni(e,t,n){mn(d(e)?e.map(e=>e.bind(t.proxy)):e.bind(t.proxy),t,n)}function ri(e,t,n,r){let i=r.includes(`.`)?Jn(n,r):()=>n[r];if(g(e)){let n=t[e];h(n)&&Gn(i,n)}else if(h(e))Gn(i,e.bind(n));else if(v(e)){if(d(e))e.forEach(e=>ri(e,t,n,r));else{let r=h(e.handler)?e.handler.bind(n):t[e.handler];h(r)&&Gn(i,r,e)}}}function ii(e){let t=e.type,{mixins:n,extends:r}=t,{mixins:i,optionsCache:a,config:{optionMergeStrategies:o}}=e.appContext,s=a.get(t),c;return s?c=s:!i.length&&!n&&!r?c=t:(c={},i.length&&i.forEach(e=>ai(c,e,o,!0)),ai(c,t,o)),v(t)&&a.set(t,c),c}function ai(e,t,n,r=!1){let{mixins:i,extends:a}=t;a&&ai(e,a,n,!0),i&&i.forEach(t=>ai(e,t,n,!0));for(let i in t)if(!(r&&i===`expose`)){let r=oi[i]||n&&n[i];e[i]=r?r(e[i],t[i]):t[i]}return e}var oi={data:si,props:fi,emits:fi,methods:di,computed:di,beforeCreate:ui,created:ui,beforeMount:ui,mounted:ui,beforeUpdate:ui,updated:ui,beforeDestroy:ui,beforeUnmount:ui,destroyed:ui,unmounted:ui,activated:ui,deactivated:ui,errorCaptured:ui,serverPrefetch:ui,components:di,directives:di,watch:pi,provide:si,inject:ci};function si(e,t){return t?e?function(){return s(h(e)?e.call(this,this):e,h(t)?t.call(this,this):t)}:t:e}function ci(e,t){return di(li(e),li(t))}function li(e){if(d(e)){let t={};for(let n=0;n<e.length;n++)t[e[n]]=e[n];return t}return e}function ui(e,t){return e?[...new Set([].concat(e,t))]:t}function di(e,t){return e?s(Object.create(null),e,t):t}function fi(e,t){return e?d(e)&&d(t)?[...new Set([...e,...t])]:s(Object.create(null),Qr(e),Qr(t??{})):t}function pi(e,t){if(!e)return t;if(!t)return e;let n=s(Object.create(null),e);for(let r in t)n[r]=ui(e[r],t[r]);return n}function mi(){return{app:null,config:{isNativeTag:i,performance:!1,globalProperties:{},optionMergeStrategies:{},errorHandler:void 0,warnHandler:void 0,compilerOptions:{}},mixins:[],components:{},directives:{},provides:Object.create(null),optionsCache:new WeakMap,propsCache:new WeakMap,emitsCache:new WeakMap}}var hi=0;function gi(e,t){return function(n,r=null){h(n)||(n=s({},n)),r!=null&&!v(r)&&(r=null);let i=mi(),a=new WeakSet,o=[],c=!1,l=i.app={_uid:hi++,_component:n,_props:r,_container:null,_context:i,_instance:null,version:$a,get config(){return i.config},set config(e){},use(e,...t){return a.has(e)||(e&&h(e.install)?(a.add(e),e.install(l,...t)):h(e)&&(a.add(e),e(l,...t))),l},mixin(e){return i.mixins.includes(e)||i.mixins.push(e),l},component(e,t){return t?(i.components[e]=t,l):i.components[e]},directive(e,t){return t?(i.directives[e]=t,l):i.directives[e]},mount(a,o,s){if(!c){let u=l._ceVNode||V(n,r);return u.appContext=i,s===!0?s=`svg`:s===!1&&(s=void 0),o&&t?t(u,a):e(u,a,s),c=!0,l._container=a,a.__vue_app__=l,Ya(u.component)}},onUnmount(e){o.push(e)},unmount(){c&&(mn(o,l._instance,16),e(null,l._container),delete l._container.__vue_app__)},provide(e,t){return i.provides[e]=t,l},runWithContext(e){let t=_i;_i=l;try{return e()}finally{_i=t}}};return l}}var _i=null,vi=(e,t)=>t===`modelValue`||t===`model-value`?e.modelModifiers:e[`${t}Modifiers`]||e[`${T(t)}Modifiers`]||e[`${D(t)}Modifiers`];function yi(e,n,...r){if(e.isUnmounted)return;let i=e.vnode.props||t,a=r,o=n.startsWith(`update:`),s=o&&vi(i,n.slice(7));s&&(s.trim&&(a=r.map(e=>g(e)?e.trim():e)),s.number&&(a=a.map(se)));let c,l=i[c=ie(n)]||i[c=ie(T(n))];!l&&o&&(l=i[c=ie(D(n))]),l&&mn(l,e,6,a);let u=i[c+`Once`];if(u){if(!e.emitted)e.emitted={};else if(e.emitted[c])return;e.emitted[c]=!0,mn(u,e,6,a)}}var bi=new WeakMap;function xi(e,t,n=!1){let r=n?bi:t.emitsCache,i=r.get(e);if(i!==void 0)return i;let a=e.emits,o={},c=!1;if(!h(e)){let r=e=>{let n=xi(e,t,!0);n&&(c=!0,s(o,n))};!n&&t.mixins.length&&t.mixins.forEach(r),e.extends&&r(e.extends),e.mixins&&e.mixins.forEach(r)}return!a&&!c?(v(e)&&r.set(e,null),null):(d(a)?a.forEach(e=>o[e]=null):s(o,a),v(e)&&r.set(e,o),o)}function Si(e,t){return!e||!a(t)?!1:(t=t.slice(2),t=t===`Once`?t:t.replace(/Once$/,``),u(e,t[0].toLowerCase()+t.slice(1))||u(e,D(t))||u(e,t))}function Ci(e){let{type:t,vnode:n,proxy:r,withProxy:i,propsOptions:[a],slots:s,attrs:c,emit:l,render:u,renderCache:d,props:f,data:p,setupState:m,ctx:h,inheritAttrs:g}=e,_=Ln(e),v,y;try{if(n.shapeFlag&4){let e=i||r,t=e;v=Ea(u.call(t,e,d,f,m,p,h)),y=c}else{let e=t;v=Ea(e.length>1?e(f,{attrs:c,slots:s,emit:l}):e(f,null)),y=t.props?c:Ti(c)}}catch(t){fa.length=0,hn(t,e,1),v=V(ua)}let b=v;if(y&&g!==!1){let e=Object.keys(y),{shapeFlag:t}=b;e.length&&t&7&&(a&&e.some(o)&&(y=Ei(y,a)),b=wa(b,y,!1,!0))}return n.dirs&&(b=wa(b,null,!1,!0),b.dirs=b.dirs?b.dirs.concat(n.dirs):n.dirs),n.transition&&lr(Xn(b.type)&&cr(b)||b,n.transition),v=b,Ln(_),v}function wi(e,t=!0){let n;for(let t=0;t<e.length;t++){let r=e[t];if(va(r)){if(r.type!==ua||r.children===`v-if`){if(n)return;n=r}}else return}return n}var Ti=e=>{let t;for(let n in e)(n===`class`||n===`style`||a(n))&&((t||={})[n]=e[n]);return t},Ei=(e,t)=>{let n={};for(let r in e)o(r)&&r.slice(9)in t||(n[r]=e[r]);return n};function Di(e,t,n){let{props:r,children:i,component:a}=e,{props:o,children:s,patchFlag:c}=t,l=a.emitsOptions;if(t.dirs||t.transition)return!0;if(n&&c>=0){if(c&1024)return!0;if(c&16)return r?Oi(r,o,l):!!o;if(c&8){let e=t.dynamicProps;for(let t=0;t<e.length;t++){let n=e[t];if(ki(o,r,n)&&!Si(l,n))return!0}}}else return(i||s)&&(!s||!s.$stable)?!0:r===o?!1:r?!o||Oi(r,o,l):!!o;return!1}function Oi(e,t,n){let r=Object.keys(t);if(r.length!==Object.keys(e).length)return!0;for(let i=0;i<r.length;i++){let a=r[i];if(!u(e,a)||ki(t,e,a)&&!Si(n,a))return!0}return!1}function ki(e,t,n){let r=e[n],i=t[n];return n===`style`&&v(r)&&v(i)?!we(r,i):r!==i}function Ai({vnode:e,parent:t,suspense:n},r){for(;t;){let n=t.subTree;if(n.suspense&&n.suspense.activeBranch===e&&(n.suspense.vnode.el=n.el=r,e=n),n===e)(e=t.vnode).el=r,t=t.parent;else break}n&&n.activeBranch===e&&(n.vnode.el=r)}var ji={},Mi=()=>Object.create(ji),Ni=e=>Object.getPrototypeOf(e)===ji;function Pi(e,t,n,r=!1){let i={},a=Mi();e.propsDefaults=Object.create(null),Ii(e,t,i,a);for(let t in e.propsOptions[0])t in i||(i[t]=void 0);e.props=n?r?i:Ht(i):e.type.props?i:a,e.attrs=a}function Fi(e,t,n,r){let{props:i,attrs:a,vnode:{patchFlag:o}}=e,s=j(i),[c]=e.propsOptions,l=!1;if((r||o>0)&&!(o&16)){if(o&8){let n=e.vnode.dynamicProps;for(let r=0;r<n.length;r++){let o=n[r];if(Si(e.emitsOptions,o))continue;let d=t[o];if(c){if(u(a,o))d!==a[o]&&(a[o]=d,l=!0);else{let t=T(o);i[t]=Li(c,s,t,d,e,!1)}}else d!==a[o]&&(a[o]=d,l=!0)}}}else{Ii(e,t,i,a)&&(l=!0);let r;for(let a in s)t&&(u(t,a)||(r=D(a))!==a&&u(t,r))||(c?n&&(n[a]!==void 0||n[r]!==void 0)&&(i[a]=Li(c,s,a,void 0,e,!0)):delete i[a]);if(a!==s)for(let e in a)t&&u(t,e)||(delete a[e],l=!0)}l&&ot(e.attrs,`set`,``)}function Ii(e,n,r,i){let[a,o]=e.propsOptions,s=!1,c;if(n)for(let t in n){if(ee(t))continue;let l=n[t],d;a&&u(a,d=T(t))?!o||!o.includes(d)?r[d]=l:(c||={})[d]=l:Si(e.emitsOptions,t)||t in i&&l===i[t]||(i[t]=l,s=!0)}if(o){let n=j(r),i=c||t;for(let t=0;t<o.length;t++){let s=o[t];r[s]=Li(a,n,s,i[s],e,!u(i,s))}}return s}function Li(e,t,n,r,i,a){let o=e[n];if(o!=null){let e=u(o,`default`);if(e&&r===void 0){let e=o.default;if(o.type!==Function&&!o.skipFactory&&h(e)){let{propsDefaults:a}=i;if(n in a)r=a[n];else{let o=Ra(i);r=a[n]=e.call(null,t),o()}}else r=e;i.ce&&i.ce._setProp(n,r)}o[0]&&(a&&!e?r=!1:o[1]&&(r===``||r===D(n))&&(r=!0))}return r}var Ri=new WeakMap;function zi(e,r,i=!1){let a=i?Ri:r.propsCache,o=a.get(e);if(o)return o;let c=e.props,l={},f=[],p=!1;if(!h(e)){let t=e=>{p=!0;let[t,n]=zi(e,r,!0);s(l,t),n&&f.push(...n)};!i&&r.mixins.length&&r.mixins.forEach(t),e.extends&&t(e.extends),e.mixins&&e.mixins.forEach(t)}if(!c&&!p)return v(e)&&a.set(e,n),n;if(d(c))for(let e=0;e<c.length;e++){let n=T(c[e]);Bi(n)&&(l[n]=t)}else if(c)for(let e in c){let t=T(e);if(Bi(t)){let n=c[e],r=l[t]=d(n)||h(n)?{type:n}:s({},n),i=r.type,a=!1,o=!0;if(d(i))for(let e=0;e<i.length;++e){let t=i[e],n=h(t)&&t.name;if(n===`Boolean`){a=!0;break}n===`String`&&(o=!1)}else a=h(i)&&i.name===`Boolean`;r[0]=a,r[1]=o,(a||u(r,`default`))&&f.push(t)}}let m=[l,f];return v(e)&&a.set(e,m),m}function Bi(e){return e[0]!==`$`&&!ee(e)}var Vi=e=>e===`_`||e===`_ctx`||e===`$stable`,Hi=e=>d(e)?e.map(Ea):[Ea(e)],Ui=(e,t,n)=>{if(t._n)return t;let r=N((...e)=>Hi(t(...e)),n);return r._c=!1,r},Wi=(e,t,n)=>{let r=e._ctx;for(let n in e){if(Vi(n))continue;let i=e[n];if(h(i))t[n]=Ui(n,i,r);else if(i!=null){let e=Hi(i);t[n]=()=>e}}},Gi=(e,t)=>{let n=Hi(t);e.slots.default=()=>n},Ki=(e,t,n)=>{for(let r in t)(n||!Vi(r))&&(e[r]=t[r])},qi=(e,t,n)=>{let r=e.slots=Mi();if(e.vnode.shapeFlag&32){let e=t._;e?(Ki(r,t,n),n&&O(r,`_`,e,!0)):Wi(t,r)}else t&&Gi(e,t)},Ji=(e,n,r)=>{let{vnode:i,slots:a}=e,o=!0,s=t;if(i.shapeFlag&32){let e=n._;e?r&&e===1?o=!1:Ki(a,n,r):(o=!n.$stable,Wi(n,a)),s=n}else n&&(Gi(e,n),s={default:1});if(o)for(let e in a)!Vi(e)&&s[e]==null&&delete a[e]},Yi=ca;function Xi(e){return Zi(e)}function Zi(e,i){let a=ue();a.__VUE__=!0;let{insert:o,remove:s,patchProp:c,createElement:l,createText:u,createComment:d,setText:f,setElementText:p,parentNode:m,nextSibling:h,setScopeId:g=r,insertStaticContent:_}=e,v=(e,t,r,i=null,a=null,o=null,s=void 0,c=null,l=!!t.dynamicChildren)=>{if(e===t)return;e&&!ya(e,t)&&(i=be(e),he(e,a,o,!0),e=null),t.patchFlag===-2&&(l=!1,t.dynamicChildren=null),t.dynamicChildren&&e&&e.dynamicChildren&&e.dynamicChildren.hasOnce&&(t.dynamicChildren===n&&(t.dynamicChildren=[]),t.dynamicChildren.hasOnce=!0);let{type:u,ref:d,shapeFlag:f}=t;switch(u){case la:y(e,t,r,i);break;case ua:b(e,t,r,i);break;case da:e??x(t,r,i,s);break;case I:ie(e,t,r,i,a,o,s,c,l);break;default:f&1?w(e,t,r,i,a,o,s,c,l):f&6?ae(e,t,r,i,a,o,s,c,l):(f&64||f&128)&&u.process(e,t,r,i,a,o,s,c,l,Se)}d!=null&&a?gr(d,e&&e.ref,o,t||e,!t):d==null&&e&&e.ref!=null&&gr(e.ref,null,o,e,!0)},y=(e,t,n,r)=>{if(e==null)o(t.el=u(t.children),n,r);else{let n=t.el=e.el;t.children!==e.children&&f(n,t.children)}},b=(e,t,n,r)=>{e==null?o(t.el=d(t.children||``),n,r):t.el=e.el},x=(e,t,n,r)=>{[e.el,e.anchor]=_(e.children,t,n,r,e.el,e.anchor)},S=({el:e,anchor:t},n,r)=>{let i;for(;e&&e!==t;)i=h(e),o(e,n,r),e=i;o(t,n,r)},C=({el:e,anchor:t})=>{let n;for(;e&&e!==t;)n=h(e),s(e),e=n;s(t)},w=(e,t,n,r,i,a,o,s,c)=>{if(t.type===`svg`?o=`svg`:t.type===`math`&&(o=`mathml`),e==null)te(t,n,r,i,a,o,s,c);else{let n=e.el&&e.el._isVueCE?e.el:null;try{n&&n._beginPatch(),E(e,t,i,a,o,s,c)}finally{n&&n._endPatch()}}},te=(e,t,n,r,i,a,s,u)=>{let d,f,{props:m,shapeFlag:h,transition:g,dirs:_}=e;if(d=e.el=l(e.type,a,m&&m.is,m),h&8?p(d,e.children):h&16&&T(e.children,d,null,r,i,Qi(e,a),s,u),_&&zn(e,null,r,`created`),ne(d,e,e.scopeId,s,r),m){for(let e in m)e!==`value`&&!ee(e)&&c(d,e,null,m[e],a,r);`value`in m&&c(d,`value`,null,m.value,a),(f=m.onVnodeBeforeMount)&&ka(f,r,e)}_&&zn(e,null,r,`beforeMount`);let v=ea(i,g);v?g.beforeEnter(d):g&&!g.persisted&&(d[sa]=!0),o(d,t,n),((f=m&&m.onVnodeMounted)||v||_)&&Yi(()=>{try{f&&ka(f,r,e),v&&g.enter(d),_&&zn(e,null,r,`mounted`)}finally{}},i)},ne=(e,t,n,r,i)=>{if(n&&g(e,n),r)for(let t=0;t<r.length;t++)g(e,r[t]);if(i){let n=i.subTree;if(t===n||oa(n.type)&&(n.ssContent===t||n.ssFallback===t)){let t=i.vnode;ne(e,t,t.scopeId,t.slotScopeIds,i.parent)}}},T=(e,t,n,r,i,a,o,s,c=0)=>{for(let l=c;l<e.length;l++){let c=e[l]=s?Da(e[l]):Ea(e[l]);v(null,c,t,n,r,i,a,o,s)}},E=(e,n,r,i,a,o,s)=>{let l=n.el=e.el,{patchFlag:u,dynamicChildren:d,dirs:f}=n;u|=e.patchFlag&16;let m=e.props||t,h=n.props||t,g;if(r&&$i(r,!1),(g=h.onVnodeBeforeUpdate)&&ka(g,r,n,e),f&&zn(n,e,r,`beforeUpdate`),r&&$i(r,!0),d&&(!e.dynamicChildren||e.dynamicChildren.length!==d.length)&&(u=0,s=!1,d=null),(m.innerHTML&&h.innerHTML==null||m.textContent&&h.textContent==null)&&p(l,``),d?D(e.dynamicChildren,d,l,r,i,Qi(n,a),o):s||de(e,n,l,null,r,i,Qi(n,a),o,!1),u>0){if(u&16)re(l,m,h,r,a);else if(u&2&&m.class!==h.class&&c(l,`class`,null,h.class,a),u&4&&c(l,`style`,m.style,h.style,a),u&8){let e=n.dynamicProps;for(let t=0;t<e.length;t++){let n=e[t],i=m[n],o=h[n];(o!==i||n===`value`)&&c(l,n,i,o,a,r)}}u&1&&e.children!==n.children&&p(l,n.children)}else!s&&d==null&&re(l,m,h,r,a);((g=h.onVnodeUpdated)||f)&&Yi(()=>{g&&ka(g,r,n,e),f&&zn(n,e,r,`updated`)},i)},D=(e,t,n,r,i,a,o)=>{for(let s=0;s<t.length;s++){let c=e[s],l=t[s],u=c.el||aa(c),d=u&&(c.type===I||!ya(c,l)||c.shapeFlag&198)?m(u):n;v(c,l,d,null,r,i,a,o,!0)}},re=(e,n,r,i,a)=>{if(n!==r){if(n!==t)for(let t in n)!ee(t)&&!(t in r)&&c(e,t,n[t],null,a,i);for(let t in r){if(ee(t))continue;let o=r[t],s=n[t];o!==s&&t!==`value`&&c(e,t,s,o,a,i)}`value`in r&&c(e,`value`,n.value,r.value,a)}},ie=(e,t,n,r,i,a,s,c,l)=>{let d=t.el=e?e.el:u(``),f=t.anchor=e?e.anchor:u(``),{patchFlag:p,dynamicChildren:m,slotScopeIds:h}=t;h&&(c=c?c.concat(h):h),e==null?(o(d,n,r),o(f,n,r),T(t.children||[],n,f,i,a,s,c,l)):p>0&&p&64&&m&&e.dynamicChildren&&e.dynamicChildren.length===m.length?(D(e.dynamicChildren,m,n,i,a,s,c),(t.key!=null||i&&t===i.subTree)&&ta(e,t,!0)):de(e,t,n,f,i,a,s,c,l)},ae=(e,t,n,r,i,a,o,s,c)=>{t.slotScopeIds=s,e==null?t.shapeFlag&512?i.ctx.activate(t,n,r,o,c):O(t,n,r,i,a,o,c):se(e,t,c)},O=(e,t,n,r,i,a,o)=>{let s=e.component=Na(e,r,i);if(yr(e)&&(s.ctx.renderer=Se),Ha(s,!1,o),s.asyncDep){if(i&&i.registerDep(s,ce,o,a),!e.el){let r=s.subTree=V(ua);b(null,r,t,n),e.placeholder=r.el}}else ce(s,e,t,n,i,a,o)},se=(e,t,n)=>{let r=t.component=e.component;if(Di(e,t,n)){if(r.asyncDep&&!r.asyncResolved){t.el=e.el,t.placeholder=e.placeholder,le(r,t,n);return}r.next=t,r.update()}else t.el=e.el,t.placeholder=e.placeholder,r.vnode=t},ce=(e,t,n,r,i,a,o)=>{let s=()=>{if(e.isMounted){let{next:t,bu:n,u:r,parent:s,vnode:c}=e;{let n=ra(e);if(n){t&&(t.el=c.el,le(e,t,o)),n.asyncDep.then(()=>{let t=()=>{e.job.flags&8||(i&&i.pendingBranch?Yi(t,i):l())};t.flags=48,Yi(t,i)});return}}let u=t,d;$i(e,!1),t?(t.el=c.el,le(e,t,o)):t=c,n&&oe(n),(d=t.props&&t.props.onVnodeBeforeUpdate)&&ka(d,s,t,c),$i(e,!0);let f=Ci(e),p=e.subTree;e.subTree=f,v(p,f,m(p.el||aa(p)),be(p),e,i,a),t.el=f.el,u===null&&Ai(e,f.el),r&&Yi(r,i),(d=t.props&&t.props.onVnodeUpdated)&&Yi(()=>ka(d,s,t,c),i)}else{let o,{el:s,props:c}=t,{bm:l,m:u,parent:d,root:f,type:p}=e,m=vr(t);if($i(e,!1),l&&oe(l),!m&&(o=c&&c.onVnodeBeforeMount)&&ka(o,d,t),$i(e,!0),s&&we){let t=()=>{e.subTree=Ci(e),we(s,e.subTree,e,i,null)};m&&p.__asyncHydrate?p.__asyncHydrate(s,e,t):t()}else{f.ce&&f.ce._hasShadowRoot()&&f.ce._injectChildStyle(p,e.parent?e.parent.type:void 0);let o=e.subTree=Ci(e);v(null,o,n,r,e,i,a),t.el=o.el}if(u&&Yi(u,i),!m&&(o=c&&c.onVnodeMounted)){let e=t;Yi(()=>ka(o,d,e),i)}(t.shapeFlag&256||d&&vr(d.vnode)&&d.vnode.shapeFlag&256)&&e.a&&Yi(e.a,i),e.isMounted=!0,t=n=r=null}};e.scope.on();let c=e.effect=new Ne(s);e.scope.off();let l=e.update=c.run.bind(c),u=e.job=c.runIfDirty.bind(c);u.i=e,u.id=e.uid,c.scheduler=()=>En(u),$i(e,!0),l()},le=(e,t,n)=>{t.component=e;let r=e.vnode.props;e.vnode=t,e.next=null,Fi(e,t.props,r,n),Ji(e,t.children,n),Je(),kn(e),Ye()},de=(e,t,n,r,i,a,o,s,c=!1)=>{let l=e&&e.children,u=e?e.shapeFlag:0,d=t.children,{patchFlag:f,shapeFlag:m}=t;if(f>0){if(f&128){pe(l,d,n,r,i,a,o,s,c);return}if(f&256){fe(l,d,n,r,i,a,o,s,c);return}}m&8?(u&16&&ye(l,i,a),d!==l&&p(n,d)):u&16?m&16?pe(l,d,n,r,i,a,o,s,c):ye(l,i,a,!0):(u&8&&p(n,``),m&16&&T(d,n,r,i,a,o,s,c))},fe=(e,t,r,i,a,o,s,c,l)=>{e||=n,t||=n;let u=e.length,d=t.length,f=Math.min(u,d),p=0;for(;p<f;p++){let n=t[p]=l?Da(t[p]):Ea(t[p]);v(e[p],n,r,null,a,o,s,c,l)}u>d?ye(e,a,o,!0,!1,f):T(t,r,i,a,o,s,c,l,f)},pe=(e,t,r,i,a,o,s,c,l)=>{let u=0,d=t.length,f=e.length-1,p=d-1;for(;u<=f&&u<=p;){let n=e[u],i=t[u]=l?Da(t[u]):Ea(t[u]);if(ya(n,i))v(n,i,r,null,a,o,s,c,l);else break;u++}for(;u<=f&&u<=p;){let n=e[f],i=t[p]=l?Da(t[p]):Ea(t[p]);if(ya(n,i))v(n,i,r,null,a,o,s,c,l);else break;f--,p--}if(u>f){if(u<=p){let e=p+1,n=e<d?t[e].el:i;for(;u<=p;)v(null,t[u]=l?Da(t[u]):Ea(t[u]),r,n,a,o,s,c,l),u++}}else if(u>p)for(;u<=f;)he(e[u],a,o,!0),u++;else{let m=u,h=u,g=new Map;for(u=h;u<=p;u++){let e=t[u]=l?Da(t[u]):Ea(t[u]);e.key!=null&&g.set(e.key,u)}let _,y=0,b=p-h+1,x=!1,S=0,C=Array(b);for(u=0;u<b;u++)C[u]=0;for(u=m;u<=f;u++){let n=e[u];if(y>=b){he(n,a,o,!0);continue}let i;if(n.key!=null)i=g.get(n.key);else for(_=h;_<=p;_++)if(C[_-h]===0&&ya(n,t[_])){i=_;break}i===void 0?he(n,a,o,!0):(C[i-h]=u+1,i>=S?S=i:x=!0,v(n,t[i],r,null,a,o,s,c,l),y++)}let w=x?na(C):n;for(_=w.length-1,u=b-1;u>=0;u--){let e=h+u,n=t[e],f=t[e+1],p=e+1<d?f.el||aa(f):i;C[u]===0?v(null,n,r,p,a,o,s,c,l):x&&(_<0||u!==w[_]?me(n,r,p,2):_--)}}},me=(e,t,n,r,i=null)=>{let{el:a,type:c,transition:l,children:u,shapeFlag:d}=e;if(d&6)me(e.component.subTree,t,n,r);else if(d&128)e.suspense.move(t,n,r);else if(d&64)c.move(e,t,n,Se);else if(c===I){o(a,t,n);for(let e=0;e<u.length;e++)me(u[e],t,n,r);o(e.anchor,t,n)}else if(c===da)S(e,t,n);else if(r!==2&&d&1&&l){if(r===0)l.persisted&&!a[Zn]?o(a,t,n):(l.beforeEnter(a),o(a,t,n),Yi(()=>l.enter(a),i));else{let{leave:r,delayLeave:i,afterLeave:c}=l,u=()=>{e.ctx.isUnmounted?s(a):o(a,t,n)},d=()=>{let e=a._isLeaving||!!a[Zn];a._isLeaving&&a[Zn](!0),l.persisted&&!e?u():r(a,()=>{u(),c&&c()})};i?i(a,u,d):d()}}else o(a,t,n)},he=(e,t,n,r=!1,i=!1)=>{let{type:a,props:o,ref:s,children:c,dynamicChildren:l,shapeFlag:u,patchFlag:d,dirs:f,cacheIndex:p,memo:m}=e;if((d===-2||l&&l.hasOnce)&&(i=!1),s!=null&&(Je(),gr(s,null,n,e,!0),Ye()),p==null||e.ctx&&e.ctx!==t||(t.renderCache[p]=void 0),u&256&&t.ctx.deactivate(e))return;let h=u&1&&f,g=!vr(e),_;if(g&&(_=o&&o.onVnodeBeforeUnmount)&&ka(_,t,e),u&6)ve(e.component,n,r);else{if(u&128){e.suspense.unmount(n,r);return}h&&zn(e,null,t,`beforeUnmount`),u&64?e.type.remove(e,t,n,Se,r):l&&!l.hasOnce&&(a!==I||d>0&&d&64)?ye(l,t,n,!1,!0):(a===I&&d&384||!i&&u&16)&&ye(c,t,n),r&&ge(e)}let v=m!=null&&p==null;(g&&(_=o&&o.onVnodeUnmounted)||h||v)&&Yi(()=>{_&&ka(_,t,e),h&&zn(e,null,t,`unmounted`),v&&(e.el=null)},n)},ge=e=>{let{type:t,el:n,anchor:r,transition:i}=e,a=t===I;if(a&&!i){_e(n,r);return}if(t===da){C(e),i&&!i.persisted&&i.afterLeave&&i.afterLeave();return}let o=a?dr(e):e,c=()=>{a?m(n)&&_e(n,r):s(n),i&&!i.persisted&&i.afterLeave&&i.afterLeave()};if(o&&o.shapeFlag&1&&i&&!i.persisted){let{leave:e,delayLeave:t}=i,n=o.el,r=()=>e(n,c);t?t(n,c,r):r()}else c()},_e=(e,t)=>{let n;for(;e!==t;)n=h(e),s(e),e=n;s(t)},ve=(e,t,n)=>{let{bum:r,scope:i,job:a,subTree:o,um:s,m:c,a:l}=e;ia(c),ia(l),r&&oe(r),i.stop(),a?(a.flags|=8,he(o,e,t,n)):o&&(o.transition=e.vnode.transition,he(o,e,t,n)),s&&Yi(s,t);let u=()=>{e.isUnmounted=!0};u.flags=16,Yi(u,t)},ye=(e,t,n,r=!1,i=!1,a=0)=>{for(let o=a;o<e.length;o++)he(e[o],t,n,r,i)},be=e=>{if(e.shapeFlag&6)return be(e.component.subTree);if(e.shapeFlag&128)return e.suspense.next();let t=h(e.anchor||e.el),n=t&&t[Yn];return n?h(n):t},xe=(e,t,n)=>{Mn(()=>{e==null?t._vnode&&he(t._vnode,null,null,!0):v(t._vnode||null,e,t,null,null,null,n),t._vnode=e},e&&t._vnode?void 0:e&&e.component?e.component.uid:Ma())},Se={p:v,um:he,m:me,r:ge,mt:O,mc:T,pc:de,pbc:D,n:be,o:e},Ce,we;return i&&([Ce,we]=i(Se)),{render:xe,hydrate:Ce,createApp:gi(xe,Ce)}}function Qi({type:e,props:t},n){return n===`svg`&&e===`foreignObject`||n===`mathml`&&e===`annotation-xml`&&t&&t.encoding&&t.encoding.includes(`html`)?void 0:n}function $i({effect:e,job:t},n){n?(e.flags|=32,t.flags|=4):(e.flags&=-33,t.flags&=-5)}function ea(e,t){return(!e||e&&!e.pendingBranch)&&t&&!t.persisted}function ta(e,t,n=!1){let r=e.children,i=t.children;if(d(r)&&d(i))for(let e=0;e<r.length;e++){let t=r[e],a=i[e];a.shapeFlag&1&&!a.dynamicChildren&&((a.patchFlag<=0||a.patchFlag===32)&&(a=i[e]=Da(i[e]),a.el=t.el),!n&&a.patchFlag!==-2&&ta(t,a)),a.type===la&&(a.patchFlag===-1&&(a=i[e]=Da(a)),a.el=t.el),a.type===ua&&!a.el&&(a.el=t.el)}}function na(e){let t=e.slice(),n=[0],r,i,a,o,s,c=e.length;for(r=0;r<c;r++){let c=e[r];if(c!==0){if(i=n[n.length-1],e[i]<c){t[r]=i,n.push(r);continue}for(a=0,o=n.length-1;a<o;)s=a+o>>1,e[n[s]]<c?a=s+1:o=s;c<e[n[a]]&&(a>0&&(t[r]=n[a-1]),n[a]=r)}}for(a=n.length,o=n[a-1];a-->0;)n[a]=o,o=t[o];return n}function ra(e){let t=e.subTree.component;if(t)return t.asyncDep&&!t.asyncResolved?t:ra(t)}function ia(e){if(e)for(let t=0;t<e.length;t++)e[t].flags|=8}function aa(e){if(e.placeholder)return e.placeholder;let t=e.component;if(t)return aa(t.subTree);let n=e.suspense;return n&&n.activeBranch?aa(n.activeBranch):null}var oa=e=>e.__isSuspense,sa=Symbol(`_deferredEnter`);function ca(e,t){t&&t.pendingBranch?d(e)?t.effects.push(...e):t.effects.push(e):On(e)}var I=Symbol.for(`v-fgt`),la=Symbol.for(`v-txt`),ua=Symbol.for(`v-cmt`),da=Symbol.for(`v-stc`),fa=[],pa=null;function L(e=!1){fa.push(pa=e?null:[])}function ma(){fa.pop(),pa=fa[fa.length-1]||null}var ha=1;function ga(e,t=!1){ha+=e,e<0&&pa&&t&&(pa.hasOnce=!0)}function _a(e){return e.dynamicChildren=ha>0?pa||n:null,ma(),ha>0&&pa&&pa.push(e),e}function R(e,t,n,r,i,a){return _a(B(e,t,n,r,i,a,!0))}function z(e,t,n,r,i){return _a(V(e,t,n,r,i,!0))}function va(e){return e?e.__v_isVNode===!0:!1}function ya(e,t){return e.type===t.type&&e.key===t.key}var ba=({key:e})=>e??null,xa=({ref:e,ref_key:t,ref_for:n})=>(typeof e==`number`&&(e=``+e),e==null?null:g(e)||Qt(e)||h(e)?{i:Fn,r:e,k:t,f:!!n}:e);function B(e,t=null,n=null,r=0,i=null,a=e===I?0:1,o=!1,s=!1){let c={__v_isVNode:!0,__v_skip:!0,type:e,props:t,key:t&&ba(t),ref:t&&xa(t),scopeId:In,slotScopeIds:null,children:n,component:null,suspense:null,ssContent:null,ssFallback:null,dirs:null,transition:null,el:null,anchor:null,target:null,targetStart:null,targetAnchor:null,staticCount:0,shapeFlag:a,patchFlag:r,dynamicProps:i,dynamicChildren:null,appContext:null,ctx:Fn};return s?(Oa(c,n),a&128&&e.normalize(c)):n&&(c.shapeFlag|=g(n)?8:16),ha>0&&!o&&pa&&(c.patchFlag>0||a&6)&&c.patchFlag!==32&&pa.push(c),c}var V=Sa;function Sa(e,t=null,n=null,r=0,i=null,a=!1){if(e&&e!==zr||(e=ua),va(e)){let r=wa(e,t,!0);return n&&Oa(r,n),ha>0&&!a&&pa&&(r.shapeFlag&6?pa[pa.indexOf(e)]=r:pa.push(r)),r.patchFlag=-2,r}if(Za(e)&&(e=e.__vccOpts),t){t=Ca(t);let{class:e,style:n}=t;e&&!g(e)&&(t.class=ge(e)),v(n)&&(Jt(n)&&!d(n)&&(n=s({},n)),t.style=de(n))}let o=g(e)?1:oa(e)?128:Xn(e)?64:v(e)?4:h(e)?2:0;return B(e,t,n,r,i,o,a,!0)}function Ca(e){return e?Jt(e)?s({},e):Ni(e)?s({},e[Ka]||e):e:null}function wa(e,t,n=!1,r=!1){let{props:i,ref:a,patchFlag:o,children:s,transition:c}=e,l=t?U(i||{},t):i,u={__v_isVNode:!0,__v_skip:!0,type:e.type,props:l,key:l&&ba(l),ref:t&&t.ref?n&&a?d(a)?a.concat(xa(t)):[a,xa(t)]:xa(t):a,scopeId:e.scopeId,slotScopeIds:e.slotScopeIds,children:s,target:e.target,targetStart:e.targetStart,targetAnchor:e.targetAnchor,staticCount:e.staticCount,shapeFlag:e.shapeFlag,patchFlag:t&&e.type!==I?o===-1?16:o|16:o,dynamicProps:e.dynamicProps,dynamicChildren:e.dynamicChildren,appContext:e.appContext,dirs:e.dirs,transition:c,component:e.component,suspense:e.suspense,ssContent:e.ssContent&&wa(e.ssContent),ssFallback:e.ssFallback&&wa(e.ssFallback),placeholder:e.placeholder,el:e.el,anchor:e.anchor,ctx:e.ctx,ce:e.ce,cacheIndex:e.cacheIndex};return c&&r&&lr(u,c.clone(u)),u}function Ta(e=` `,t=0){return V(la,null,e,t)}function H(e=``,t=!1){return t?(L(),z(ua,null,e)):V(ua,null,e)}function Ea(e){return e==null||typeof e==`boolean`?V(ua):d(e)?V(I,null,e.slice()):va(e)?Da(e):V(la,null,String(e))}function Da(e){return e.el===null&&e.patchFlag!==-1||e.memo?e:wa(e)}function Oa(e,t){let n=0,{shapeFlag:r}=e;if(t==null)t=null;else if(d(t))n=16;else if(typeof t==`object`){if(r&65){let n=t.default;n&&(n._c&&(n._d=!1),Oa(e,n()),n._c&&(n._d=!0));return}{n=32;let r=t._;!r&&!Ni(t)?t._ctx=Fn:r===3&&Fn&&(Fn.slots._===1?t._=1:(t._=2,e.patchFlag|=1024))}}else if(h(t)){if(r&65){Oa(e,{default:t});return}t={default:t,_ctx:Fn},n=32}else t=String(t),r&64?(n=16,t=[Ta(t)]):n=8;e.children=t,e.shapeFlag|=n}function U(...e){let t={};for(let n=0;n<e.length;n++){let r=e[n];for(let e in r)if(e===`class`)t.class!==r.class&&(t.class=ge([t.class,r.class]));else if(e===`style`)t.style=de([t.style,r.style]);else if(a(e)){let n=t[e],i=r[e];i&&n!==i&&!(d(n)&&n.includes(i))?t[e]=n?[].concat(n,i):i:i==null&&n==null&&!o(e)&&(t[e]=i)}else e!==``&&(t[e]=r[e])}return t}function ka(e,t,n,r=null){mn(e,t,7,[n,r])}var Aa=mi(),ja=0;function Ma(){return ja}function Na(e,n,r){let i=e.type,a=(n?n.appContext:e.appContext)||Aa,o={uid:ja++,vnode:e,type:i,parent:n,appContext:a,root:null,next:null,subTree:null,effect:null,update:null,job:null,scope:new ke(!0),render:null,proxy:null,exposed:null,exposeProxy:null,withProxy:null,provides:n?n.provides:Object.create(a.provides),ids:n?n.ids:[``,0,0],accessCache:null,renderCache:[],components:null,directives:null,propsOptions:zi(i,a),emitsOptions:xi(i,a),emit:null,emitted:null,propsDefaults:t,inheritAttrs:i.inheritAttrs,ctx:t,data:t,props:t,attrs:t,slots:t,refs:t,setupState:t,setupContext:null,suspense:r,suspenseId:r?r.pendingId:0,asyncDep:null,asyncResolved:!1,isMounted:!1,isUnmounted:!1,isDeactivated:!1,bc:null,c:null,bm:null,m:null,bu:null,u:null,um:null,bum:null,da:null,a:null,rtg:null,rtc:null,ec:null,sp:null};return o.ctx={_:o},o.root=n?n.root:o,o.emit=yi.bind(null,o),e.ce&&e.ce(o),o}var Pa=null,Fa=()=>Pa||Fn,Ia,La;{let e=ue(),t=(t,n)=>{let r;return(r=e[t])||(r=e[t]=[]),r.push(n),e=>{r.length>1?r.forEach(t=>t(e)):r[0](e)}};Ia=t(`__VUE_INSTANCE_SETTERS__`,e=>Pa=e),La=t(`__VUE_SSR_SETTERS__`,e=>Va=e)}var Ra=e=>{let t=Pa;return Ia(e),e.scope.on(),()=>{e.scope.off(),Ia(t)}},za=()=>{Pa&&Pa.scope.off(),Ia(null)};function Ba(e){return e.vnode.shapeFlag&4}var Va=!1;function Ha(e,t=!1,n=!1){t&&La(t);let{props:r,children:i}=e.vnode,a=Ba(e);Pi(e,r,a,t),qi(e,i,n||t);let o=a?Ua(e,t):void 0;return t&&La(!1),o}function Ua(e,t){let n=e.type;e.accessCache=Object.create(null),e.proxy=new Proxy(e.ctx,Zr);let{setup:r}=n;if(r){Je();let n=e.setupContext=r.length>1?Ja(e):null,i=Ra(e),a=pn(r,e,0,[e.props,n]),o=y(a);if(Ye(),i(),(o||e.sp)&&!vr(e)&&pr(e),o){if(a.then(za,za),t)return a.then(n=>{La(!0);try{Wa(e,n,t)}finally{La(!1)}}).catch(t=>{hn(t,e,0,void 0,!0),e.ssrRender=(e,t)=>t(`\\x3C!---->`)});e.asyncDep=a}else Wa(e,a,t)}else Ga(e,t)}function Wa(e,t,n){h(t)?e.type.__ssrInlineRender?e.ssrRender=t:e.render=t:v(t)&&(e.setupState=rn(t)),Ga(e,n)}function Ga(e,t,n){let i=e.type;e.render||=i.render||r;{let t=Ra(e);Je();try{ei(e)}finally{Ye(),t()}}}var Ka=Symbol(),qa={get(e,t){return at(e,`get`,``),t===Ka?e:e[t]}};function Ja(e){return{attrs:new Proxy(e.attrs,qa),slots:e.slots,emit:e.emit,expose:t=>{e.exposed=t||{}}}}function Ya(e){return e.exposed?e.exposeProxy||=new Proxy(rn(Yt(e.exposed)),{get(t,n){if(n in t)return t[n];if(n in Yr)return Yr[n](e)},has(e,t){return t in e||t in Yr}}):e.proxy}function Xa(e,t=!0){return h(e)?e.displayName||e.name:e.name||t&&e.__name}function Za(e){return h(e)&&`__vccOpts`in e}var W=(e,t)=>on(e,t,Va);function Qa(e,t,n){try{ga(-1);let r=arguments.length;return r===2?v(t)&&!d(t)?va(t)?V(e,null,[t]):V(e,t):V(e,null,t):(r>3?n=Array.prototype.slice.call(arguments,2):r===3&&va(n)&&(n=[n]),V(e,t,n))}finally{ga(1)}}var $a=`3.5.45`,eo=void 0,to=typeof window<`u`&&window.trustedTypes;if(to)try{eo=to.createPolicy(`vue`,{createHTML:e=>e})}catch{}var no=eo?e=>eo.createHTML(e):e=>e,ro=`http://www.w3.org/2000/svg`,io=`http://www.w3.org/1998/Math/MathML`,ao=typeof document<`u`?document:null,oo=ao&&ao.createElement(`template`),so={insert:(e,t,n)=>{t.insertBefore(e,n||null)},remove:e=>{let t=e.parentNode;t&&t.removeChild(e)},createElement:(e,t,n,r)=>{let i=t===`svg`?ao.createElementNS(ro,e):t===`mathml`?ao.createElementNS(io,e):n?ao.createElement(e,{is:n}):ao.createElement(e);return e===`select`&&r&&r.multiple!=null&&i.setAttribute(`multiple`,r.multiple),i},createText:e=>ao.createTextNode(e),createComment:e=>ao.createComment(e),setText:(e,t)=>{e.nodeValue=t},setElementText:(e,t)=>{e.textContent=t},parentNode:e=>e.parentNode,nextSibling:e=>e.nextSibling,querySelector:e=>ao.querySelector(e),setScopeId(e,t){e.setAttribute(t,``)},insertStaticContent(e,t,n,r,i,a){let o=n?n.previousSibling:t.lastChild;if(i&&(i===a||i.nextSibling))for(;t.insertBefore(i.cloneNode(!0),n),i!==a&&(i=i.nextSibling););else{oo.innerHTML=no(r===`svg`?`<svg>${e}</svg>`:r===`mathml`?`<math>${e}</math>`:e);let i=oo.content;if(r===`svg`||r===`mathml`){let e=i.firstChild;for(;e.firstChild;)i.appendChild(e.firstChild);i.removeChild(e)}t.insertBefore(i,n)}return[o?o.nextSibling:t.firstChild,n?n.previousSibling:t.lastChild]}},co=`transition`,lo=`animation`,uo=Symbol(`_vtc`),fo={name:String,type:String,css:{type:Boolean,default:!0},duration:[String,Number,Object],enterFromClass:String,enterActiveClass:String,enterToClass:String,appearFromClass:String,appearActiveClass:String,appearToClass:String,leaveFromClass:String,leaveActiveClass:String,leaveToClass:String},po=s({},tr,fo),mo=(e=>(e.displayName=`Transition`,e.props=po,e))((e,{slots:t})=>Qa(ir,_o(e),t)),ho=(e,t=[])=>{d(e)?e.forEach(e=>e(...t)):e&&e(...t)},go=e=>e?d(e)?e.some(e=>e.length>1):e.length>1:!1;function _o(e){let t={};for(let n in e)n in fo||(t[n]=e[n]);if(e.css===!1)return t;let{name:n=`v`,type:r,duration:i,enterFromClass:a=`${n}-enter-from`,enterActiveClass:o=`${n}-enter-active`,enterToClass:c=`${n}-enter-to`,appearFromClass:l=a,appearActiveClass:u=o,appearToClass:d=c,leaveFromClass:f=`${n}-leave-from`,leaveActiveClass:p=`${n}-leave-active`,leaveToClass:m=`${n}-leave-to`}=e,h=vo(i),g=h&&h[0],_=h&&h[1],{onBeforeEnter:v,onEnter:y,onEnterCancelled:b,onLeave:x,onLeaveCancelled:S,onBeforeAppear:C=v,onAppear:w=y,onAppearCancelled:ee=b}=t,te=(e,t,n,r)=>{e._enterCancelled=r,xo(e,t?d:c),xo(e,t?u:o),n&&n()},ne=(e,t)=>{e._isLeaving=!1,xo(e,f),xo(e,m),xo(e,p),t&&t()},T=e=>(t,n)=>{let i=e?w:y,o=()=>te(t,e,n);ho(i,[t,o]),So(()=>{xo(t,e?l:a),bo(t,e?d:c),go(i)||wo(t,r,g,o)})};return s(t,{onBeforeEnter(e){ho(v,[e]),bo(e,a),bo(e,o)},onBeforeAppear(e){ho(C,[e]),bo(e,l),bo(e,u)},onEnter:T(!1),onAppear:T(!0),onLeave(e,t){e._isLeaving=!0;let n=()=>ne(e,t);bo(e,f),e._enterCancelled?(bo(e,p),Oo(e)):(Oo(e),bo(e,p)),So(()=>{e._isLeaving&&(xo(e,f),bo(e,m),go(x)||wo(e,r,_,n))}),ho(x,[e,n])},onEnterCancelled(e){te(e,!1,void 0,!0),ho(b,[e])},onAppearCancelled(e){te(e,!0,void 0,!0),ho(ee,[e])},onLeaveCancelled(e){ne(e),ho(S,[e])}})}function vo(e){if(e==null)return null;if(v(e))return[yo(e.enter),yo(e.leave)];{let t=yo(e);return[t,t]}}function yo(e){return ce(e)}function bo(e,t){t.split(/\\s+/).forEach(t=>t&&e.classList.add(t)),(e[uo]||(e[uo]=new Set)).add(t)}function xo(e,t){t.split(/\\s+/).forEach(t=>t&&e.classList.remove(t));let n=e[uo];n&&(n.delete(t),n.size||(e[uo]=void 0))}function So(e){requestAnimationFrame(()=>{requestAnimationFrame(e)})}var Co=0;function wo(e,t,n,r){let i=e._endId=++Co,a=()=>{i===e._endId&&r()};if(n!=null)return setTimeout(a,n);let{type:o,timeout:s,propCount:c}=To(e,t);if(!o)return r();let l=o+`end`,u=0,d=()=>{e.removeEventListener(l,f),a()},f=t=>{t.target===e&&++u>=c&&d()};setTimeout(()=>{u<c&&d()},s+1),e.addEventListener(l,f)}function To(e,t){let n=window.getComputedStyle(e),r=e=>(n[e]||``).split(`, `),i=r(`${co}Delay`),a=r(`${co}Duration`),o=Eo(i,a),s=r(`${lo}Delay`),c=r(`${lo}Duration`),l=Eo(s,c),u=null,d=0,f=0;t===co?o>0&&(u=co,d=o,f=a.length):t===lo?l>0&&(u=lo,d=l,f=c.length):(d=Math.max(o,l),u=d>0?o>l?co:lo:null,f=u?u===co?a.length:c.length:0);let p=u===co&&/\\b(?:transform|all)(?:,|$)/.test(r(`${co}Property`).toString());return{type:u,timeout:d,propCount:f,hasTransform:p}}function Eo(e,t){for(;e.length<t.length;)e=e.concat(e);return Math.max(...t.map((t,n)=>Do(t)+Do(e[n])))}function Do(e){return e===`auto`?0:Number(e.slice(0,-1).replace(`,`,`.`))*1e3}function Oo(e){return(e?e.ownerDocument:document).body.offsetHeight}function ko(e,t,n){let r=e[uo];r&&(t=(t?[t,...r]:[...r]).join(` `)),t==null?e.removeAttribute(`class`):n?e.setAttribute(`class`,t):e.className=t}var Ao=Symbol(`_vod`),jo=Symbol(`_vsh`),Mo={name:`show`,beforeMount(e,{value:t},{transition:n}){e[Ao]=e.style.display===`none`?``:e.style.display,n&&t?n.beforeEnter(e):No(e,t)},mounted(e,{value:t},{transition:n}){n&&t&&n.enter(e)},updated(e,{value:t,oldValue:n},{transition:r}){!t!=!n&&(r?t?(r.beforeEnter(e),No(e,!0),r.enter(e)):r.leave(e,()=>{No(e,!1)}):No(e,t))},beforeUnmount(e,{value:t}){No(e,t)}};function No(e,t){e.style.display=t?e[Ao]:`none`,e[jo]=!t}var Po=Symbol(``),Fo=/(?:^|;)\\s*display\\s*:/;function Io(e,t,n){let r=e.style,i=g(n),a=!1;if(n&&!i){if(t){if(g(t))for(let e of t.split(`;`)){let t=e.slice(0,e.indexOf(`:`)).trim();n[t]??Ro(r,t,``)}else for(let e in t)n[e]??Ro(r,e,``)}for(let i in n){i===`display`&&(a=!0);let o=n[i];o==null?Ro(r,i,``):Ho(e,i,!g(t)&&t?t[i]:void 0,o)||Ro(r,i,o)}}else if(i){if(t!==n){let e=r[Po];e&&(n+=`;`+e),r.cssText=n,a=Fo.test(n)}}else t&&e.removeAttribute(`style`);Ao in e&&(i&&t===n||(e[Ao]=a?r.display:``),e[jo]&&(r.display=`none`))}var Lo=/\\s*!important$/;function Ro(e,t,n){if(d(n))n.forEach(n=>Ro(e,t,n));else if(n??=``,t.startsWith(`--`))Lo.test(n)?e.setProperty(t,n.replace(Lo,``),`important`):e.setProperty(t,n);else{let r=Vo(e,t);Lo.test(n)?e.setProperty(D(r),n.replace(Lo,``),`important`):e[r]=n}}var zo=[`Webkit`,`Moz`,`ms`],Bo={};function Vo(e,t){let n=Bo[t];if(n)return n;let r=T(t);if(r!==`filter`&&r in e)return Bo[t]=r;r=re(r);for(let n=0;n<zo.length;n++){let i=zo[n]+r;if(i in e)return Bo[t]=i}return t}function Ho(e,t,n,r){return e.tagName===`TEXTAREA`&&(t===`width`||t===`height`)&&g(r)&&n===r}var Uo=`http://www.w3.org/1999/xlink`;function Wo(e,t,n,r,i,a=ve(t)){r&&t.startsWith(`xlink:`)?n==null?e.removeAttributeNS(Uo,t.slice(6,t.length)):e.setAttributeNS(Uo,t,n):n==null||a&&!ye(n)?e.removeAttribute(t):e.setAttribute(t,a?``:_(n)?String(n):n)}function Go(e,t,n,r,i){if(t===`innerHTML`||t===`textContent`){n!=null&&(e[t]=t===`innerHTML`?no(n):n);return}let a=e.tagName;if(t===`value`&&a!==`PROGRESS`&&!a.includes(`-`)){let r=a===`OPTION`?e.getAttribute(`value`)||``:e.value,i=n==null?e.type===`checkbox`?`on`:``:String(n);r===i&&`_value`in e||(e.value=i),n??e.removeAttribute(t),e._value=n;return}let o=!1;if(n===``||n==null){let r=typeof e[t];r===`boolean`?n=ye(n):n==null&&r===`string`?(n=``,o=!0):r===`number`&&(n=0,o=!0)}try{e[t]=n}catch{}o&&e.removeAttribute(i||t)}function Ko(e,t,n,r){e.addEventListener(t,n,r)}function qo(e,t,n,r){e.removeEventListener(t,n,r)}var Jo=Symbol(`_vei`);function Yo(e,t,n,r,i=null){let a=e[Jo]||(e[Jo]={}),o=a[t];if(r&&o)o.value=r;else{let[n,s]=Qo(t);r?Ko(e,n,a[t]=ns(r,i),s):o&&(qo(e,n,o,s),a[t]=void 0)}}var Xo=/(Once|Passive|Capture)$/,Zo=/^on:?(?:Once|Passive|Capture)$/;function Qo(e){let t,n;for(;(n=e.match(Xo))&&!Zo.test(e);)t||={},e=e.slice(0,e.length-n[1].length),t[n[1].toLowerCase()]=!0;return[e[2]===`:`?e.slice(3):D(e.slice(2)),t]}var $o=0,es=Promise.resolve(),ts=()=>$o||=(es.then(()=>$o=0),Date.now());function ns(e,t){let n=e=>{if(!e._vts)e._vts=Date.now();else if(e._vts<=n.attached)return;let r=n.value;if(d(r)){let n=e.stopImmediatePropagation;e.stopImmediatePropagation=()=>{n.call(e),e._stopped=!0};let i=r.slice(),a=[e];for(let n=0;n<i.length&&!e._stopped;n++){let e=i[n];e&&mn(e,t,5,a)}}else mn(r,t,5,[e])};return n.value=e,n.attached=ts(),n}var rs=e=>e.charCodeAt(0)===111&&e.charCodeAt(1)===110&&e.charCodeAt(2)>96&&e.charCodeAt(2)<123,is=(e,t,n,r,i,s)=>{let c=i===`svg`;t===`class`?ko(e,r,c):t===`style`?Io(e,n,r):a(t)?o(t)||Yo(e,t,n,r,s):(t[0]===`.`?(t=t.slice(1),1):t[0]===`^`?(t=t.slice(1),0):as(e,t,r,c))?(Go(e,t,r),e.tagName.includes(`-`)||t!==`value`&&t!==`checked`&&t!==`selected`||Wo(e,t,r,c,s,t!==`value`)):e._isVueCE&&(os(e,t)||e._def.__asyncLoader&&(/[A-Z]/.test(t)||!g(r)))?Go(e,T(t),r,s,t):(t===`true-value`?e._trueValue=r:t===`false-value`&&(e._falseValue=r),Wo(e,t,r,c))};function as(e,t,n,r){if(r)return!!(t===`innerHTML`||t===`textContent`||t in e&&rs(t)&&h(n));if(t===`spellcheck`||t===`draggable`||t===`translate`||t===`autocorrect`||t===`sandbox`&&e.tagName===`IFRAME`||t===`form`||t===`list`&&e.tagName===`INPUT`||t===`type`&&e.tagName===`TEXTAREA`)return!1;if(t===`width`||t===`height`){let t=e.tagName;if(t===`IMG`||t===`VIDEO`||t===`CANVAS`||t===`SOURCE`)return!1}return rs(t)&&g(n)?!1:t in e}function os(e,t){let n=e._def.props;if(!n)return!1;let r=T(t);return Array.isArray(n)?n.some(e=>T(e)===r):Object.keys(n).some(e=>T(e)===r)}var ss=new WeakMap,cs=new WeakMap,ls=Symbol(`_moveCb`),us=Symbol(`_enterCb`),ds=new WeakMap,fs=(e=>(delete e.props.mode,e))({name:`TransitionGroup`,props:s({},po,{tag:String,moveClass:String}),setup(e,{slots:t}){let n=Fa(),r=$n(),i,a,o,s,c=()=>(s===void 0&&(s=i.length>0&&xs(i[0].el,n.vnode.el,ms(e))),s);return kr(()=>{if(ds.delete(n),!c()){i=[],o=void 0;return}let t=ms(e);i.forEach(ps);let r=o&&bs(o,n.vnode.el);o=void 0,i.forEach(e=>hs(e,r));let a=i.filter(gs);Oo(n.vnode.el),a.forEach(e=>{let n=e.el,r=n.style;bo(n,t),r.transform=r.webkitTransform=r.transitionDuration=``;let i=n[ls]=e=>{e&&e.target!==n||(!e||e.propertyName.endsWith(`transform`))&&(n.removeEventListener(`transitionend`,i),n[ls]=null,xo(n,t))};n.addEventListener(`transitionend`,i)}),i=[]}),()=>{let l=j(e),u=_o(l),d=l.tag||I;if(i=[],s=void 0,a)for(let e=0;e<a.length;e++){let t=a[e];t.el&&t.el instanceof Element&&!t.el[jo]&&(i.push(t),lr(t,or(t,u,r,n)),ss.set(t,_s(t.el)))}a=t.default?ur(t.default()):[];for(let e=0;e<a.length;e++){let t=a[e];t.key!=null&&lr(t,or(t,u,r,n))}return i.length&&(o=vs(n),ds.set(n,c)),V(d,null,a)}}});function ps(e){let t=e.el;t[ls]&&t[ls](),t[us]&&t[us]()}function ms(e){return e.moveClass||`${e.name||`v`}-move`}function hs(e,t){cs.set(e,_s(e.el,t))}function gs(e){let t=ss.get(e),n=cs.get(e),r=t.left-n.left,i=t.top-n.top;if(r||i){let t=e.el,n=t.style,a=t.getBoundingClientRect(),o=1,s=1;return t.offsetWidth&&(o=a.width/t.offsetWidth),t.offsetHeight&&(s=a.height/t.offsetHeight),Number.isFinite(o)&&o!==0||(o=1),Number.isFinite(s)&&s!==0||(s=1),Math.abs(o-1)<.01&&(o=1),Math.abs(s-1)<.01&&(s=1),n.transform=n.webkitTransform=`translate(${r/o}px,${i/s}px)`,n.transitionDuration=`0s`,e}}function _s(e,t){let n=e.getBoundingClientRect();return{left:n.left-(t?t.left:0),top:n.top-(t?t.top:0)}}function vs(e){let t;for(let n=e.parent;n;n=n.parent){let r=ds.get(n);if(r){let i=ys(n.vnode.el,e.vnode.el);i&&(t||=[]).push(s(_s(i),{item:i,willMove:r}))}}return t}function ys(e,t){let n=e.nodeType===1?e:e.parentNode;for(;t&&t.parentNode!==n;)t=t.parentNode;if(t&&t.nodeType===1)return t}function bs(e,t){let n=e.find(e=>e.willMove()&&e.item.contains(t));return n&&_s(n.item,n)}function xs(e,t,n){let r=e.cloneNode(),i=e[uo];i&&i.forEach(e=>{e.split(/\\s+/).forEach(e=>e&&r.classList.remove(e))}),n.split(/\\s+/).forEach(e=>e&&r.classList.add(e)),r.style.display=`none`;let a=t.nodeType===1?t:t.parentNode;a.appendChild(r);let{hasTransform:o}=To(r);return a.removeChild(r),o}var Ss=[`ctrl`,`shift`,`alt`,`meta`],Cs={stop:e=>e.stopPropagation(),prevent:e=>e.preventDefault(),self:e=>e.target!==e.currentTarget,ctrl:e=>!e.ctrlKey,shift:e=>!e.shiftKey,alt:e=>!e.altKey,meta:e=>!e.metaKey,left:e=>`button`in e&&e.button!==0,middle:e=>`button`in e&&e.button!==1,right:e=>`button`in e&&e.button!==2,exact:(e,t)=>Ss.some(n=>e[`${n}Key`]&&!t.includes(n))},ws=(e,t)=>{if(!e)return e;let n=e._withMods||={},r=t.join(`.`);return n[r]||(n[r]=((n,...r)=>{for(let e=0;e<t.length;e++){let r=Cs[t[e]];if(r&&r(n,t))return}return e(n,...r)}))},Ts={esc:`escape`,space:` `,up:`arrow-up`,left:`arrow-left`,right:`arrow-right`,down:`arrow-down`,delete:`backspace`},Es=(e,t)=>{let n=e._withKeys||={},r=t.join(`.`);return n[r]||(n[r]=(n=>{if(!(`key`in n))return;let r=D(n.key);if(t.some(e=>e===r||Ts[e]===r))return e(n)}))},Ds=s({patchProp:is},so),Os;function ks(){return Os||=Xi(Ds)}var As=((...e)=>{let t=ks().createApp(...e),{mount:n}=t;return t.mount=e=>{let r=Ms(e);if(!r)return;let i=t._component;h(i)||i.render||i.template||(i.template=r.innerHTML),r.nodeType===1&&(r.textContent=``);let a=n(r,!1,js(r));return r instanceof Element&&(r.removeAttribute(`v-cloak`),r.setAttribute(`data-v-app`,``)),a},t});function js(e){if(e instanceof SVGElement)return`svg`;if(typeof MathMLElement==`function`&&e instanceof MathMLElement)return`mathml`}function Ms(e){return g(e)?document.querySelector(e):e}var Ns=Object.defineProperty,Ps=Object.getOwnPropertySymbols,Fs=Object.prototype.hasOwnProperty,Is=Object.prototype.propertyIsEnumerable,Ls=(e,t,n)=>t in e?Ns(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,Rs=(e,t)=>{for(var n in t||={})Fs.call(t,n)&&Ls(e,n,t[n]);if(Ps)for(var n of Ps(t))Is.call(t,n)&&Ls(e,n,t[n]);return e},zs=(e,t,n)=>new Promise((r,i)=>{var a=e=>{try{s(n.next(e))}catch(e){i(e)}},o=e=>{try{s(n.throw(e))}catch(e){i(e)}},s=e=>e.done?r(e.value):Promise.resolve(e.value).then(a,o);s((n=n.apply(e,t)).next())}),G=class extends Error{},Bs=e=>e===` `||e===`\n"
    "`||e===`\\r`||e===`\t`,Vs=e=>e!==void 0&&e>=`0`&&e<=`9`;function Hs(e){if(e===void 0)throw new G(`Bad escape`);if(e>=`0`&&e<=`9`)return e.charCodeAt(0)-48;if(e>=`a`&&e<=`f`)return e.charCodeAt(0)-87;if(e>=`A`&&e<=`F`)return e.charCodeAt(0)-55;throw new G(`Bad escape`)}var Us=(()=>{let e=Object.create(null);for(let t=0;t<64;t++)e[`ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_`[t]]=t;return e})();function Ws(e){if(e.length%4==1)throw Error(`Invalid base64url length`);let t=Math.floor(6*e.length/8),n=new Uint8Array(t),r=0,i=0,a=0;for(let t=0;t<e.length;t++){let o=Us[e[t]];if(o===void 0)throw Error(`Invalid base64url character`);r=r<<6|o,i+=6,i>=8&&(i-=8,n[a++]=r>>i&255)}if(i>0&&r&(1<<i)-1)throw Error(`Invalid base64url trailing bits`);return n}var Gs=Object.freeze({primeui:`primeui`,scheduler:`primeui-pro:scheduler`,texteditor:`primeui-pro:text-editor`,charts:`primeui-pro:charts`,diagram:`primeui-pro:diagram`,pdfviewer:`primeui-pro:pdf-viewer`,taskboard:`primeui-pro:task-board`,datagrid:`primeui-pro:datagrid`,ganttchart:`primeui-pro:gantt-chart`,filemanager:`primeui-pro:file-manager`}),Ks=16,qs=65536,Js=()=>Array(Ks).fill(0);function Ys(e){let t=Js(),n=e;for(let e=0;e<Ks&&n!==0;e++)t[e]=n%qs,n=Math.floor(n/qs);return t}var Xs=Js(),Zs=Ys(1),Qs=(()=>{let e=Js();e[0]=65517;for(let t=1;t<15;t++)e[t]=65535;return e[15]=32767,e})();function $s(e,t){let n=t;for(;n!==0;){let t=38*n;n=0;for(let n=0;n<Ks;n++){let r=e[n]+t;if(t=Math.floor(r/qs),e[n]=r-t*qs,t===0)break}n=t}}function K(e,t){let n=Array(32).fill(0);for(let r=0;r<Ks;r++){let i=e[r];if(i!==0){for(let e=0;e<Ks;e++)n[r+e]=n[r+e]+i*t[e];if(r%4==3){let e=0;for(let t=0;t<32;t++){let r=n[t]+e;e=Math.floor(r/qs),n[t]=r-e*qs}}}}return function(e){for(let t=Ks;t<e.length;t++)e[t-Ks]=e[t-Ks]+38*e[t],e[t]=0;let t=0;for(let n=0;n<Ks;n++){let r=e[n]+t;t=Math.floor(r/qs),e[n]=r-t*qs}let n=e.slice(0,Ks);return $s(n,t),n}(n)}function ec(e){return K(e,e)}function tc(e,t){let n=Js(),r=0;for(let i=0;i<Ks;i++){let a=e[i]+t[i]+r;r=+(a>=qs),n[i]=a-r*qs}return $s(n,r),n}function nc(e,t){let n=Js(),r=0;for(let i=0;i<Ks;i++){let a=e[i]-t[i]-r;r=+(a<0),n[i]=a+r*qs}return $s(n,-r),n}function rc(e){return nc(Xs,e)}function ic(e){for(let t=15;t>=0;t--){if(e[t]>Qs[t])return!0;if(e[t]<Qs[t])return!1}return!0}function ac(e){let t=function(e){let t=e.slice(),n=0;for(let e=0;e<Ks;e++){let r=t[e]+n;n=Math.floor(r/qs),t[e]=r-n*qs}$s(t,n);for(let e=0;e<2&&ic(t);e++){let e=0;for(let n=0;n<Ks;n++){let r=t[n]-Qs[n]-e;e=+(r<0),t[n]=r+e*qs}}return t}(e),n=new Uint8Array(32);for(let e=0;e<Ks;e++)n[2*e]=255&t[e],n[2*e+1]=t[e]>>8&255;return n}function oc(e){let t=ac(e),n=0;for(let e=0;e<32;e++)n|=t[e];return n===0}function sc(e,t){return oc(nc(e,t))}function cc(e,t){let n=e;for(let e=0;e<t;e++)n=ec(n);return n}function lc(e){let t=ec(e),n=K(e,cc(t,2)),r=K(t,n),i=K(n,ec(r)),a=K(cc(i,5),i),o=K(cc(a,10),a),s=K(cc(K(cc(o,20),o),10),a),c=K(cc(s,50),s);return{z11:r,z250:K(cc(K(cc(c,100),c),50),s)}}function uc(e){let{z11:t,z250:n}=lc(e);return K(cc(n,5),t)}var dc=K(rc(Ys(121665)),uc(Ys(121666))),fc=(()=>{let e=new Uint8Array(32);e[0]=251;for(let t=1;t<31;t++)e[t]=255;e[31]=31;let t=Zs,n=Ys(2);for(let r=0;r<32;r++)for(let i=0;i<8;i++)(e[r]>>i&1)==1&&(t=K(t,n)),n=ec(n);return t})();function pc(e,t){let n=K(nc(e.y,e.x),nc(t.y,t.x)),r=K(tc(e.y,e.x),tc(t.y,t.x)),i=K(K(tc(dc,dc),e.t),t.t),a=K(tc(e.z,e.z),t.z),o=nc(r,n),s=nc(a,i),c=tc(a,i),l=tc(r,n);return{x:K(o,s),y:K(c,l),z:K(s,c),t:K(o,l)}}function mc(e){return pc(e,e)}function hc(e,t){let n={x:Xs.slice(),y:Zs.slice(),z:Zs.slice(),t:Xs.slice()},r=!1;for(let i=e.length-1;i>=0;i--)for(let a=7;a>=0;a--)r&&(n=mc(n)),(e[i]>>a&1)==1&&(r?n=pc(n,t):(n={x:t.x.slice(),y:t.y.slice(),z:t.z.slice(),t:t.t.slice()},r=!0));return n}function gc(e){if(e.length!==32)return null;let t=Uint8Array.from(e),n=(t[31]>>7&1)==1;t[31]=127&t[31];let r=function(e){let t=Js();for(let n=0;n<Ks;n++)t[n]=e[2*n]|e[2*n+1]<<8;return t}(t);if(ic(r))return null;let i=ec(r),a=nc(i,Zs),o=tc(K(dc,i),Zs),s=K(ec(o),o),c=K(ec(s),o),l=K(K(a,s),function(e){let{z250:t}=lc(e);return K(cc(t,2),e)}(K(a,c)));return sc(K(ec(l),o),a)||(l=K(l,fc),sc(K(ec(l),o),a))?oc(l)&&n?null:(!(1&~ac(l)[0])!==n&&(l=rc(l)),{x:l,y:r,z:Zs.slice(),t:K(l,r)}):null}var _c=(()=>{let e=gc(ac(K(Ys(4),uc(Ys(5)))));if(!e)throw Error(`[@primeui/license-manager] Ed25519 base point failed to initialise`);return e})();function vc(e){let t=mc(mc(mc(e)));return oc(K(t.x,t.z))&&sc(t.y,t.z)}var yc=new Uint8Array([237,211,245,92,26,99,18,88,214,156,247,162,222,249,222,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16]);function bc(e){for(let t=31;t>=0;t--){if(e[t]<yc[t])return!0;if(e[t]>yc[t])return!1}return!1}var xc=Object.freeze({primeui:`PrimeUI`,scheduler:`Scheduler`,texteditor:`TextEditor`,charts:`Charts`,diagram:`Diagram`,pdfviewer:`PDF Viewer`,taskboard:`Task Board`,datagrid:`DataGrid`,ganttchart:`Gantt`,filemanager:`File Manager`}),Sc=(e,t)=>Object.prototype.hasOwnProperty.call(e,t);function Cc(e){return Sc(xc,e)?xc[e]:`PrimeUI`}function wc(e){return Sc(Gs,e)?Gs[e]:void 0}function Tc(e,t=`PrimeUI`){switch(e){case`active`:return`${t} license is active.`;case`grace`:return`${t} license is in its grace period. Renew soon to keep using this version.`;case`expired`:return`${t} license does not cover this version. Renew at primeui.store, or downgrade to a version released within your updates window.`;case`tampered`:return`${t} license signature is invalid.`;case`wrong-product`:return`License does not cover ${t}.`;case`missing`:return`No license key configured for ${t}.`;case`invalid`:return`${t} license is malformed.`;case`unconfigured`:return`${t} license is not configured.`;default:return`${t} license status unknown.`}}var Ec=[[1116352408,3609767458],[1899447441,602891725],[3049323471,3964484399],[3921009573,2173295548],[961987163,4081628472],[1508970993,3053834265],[2453635748,2937671579],[2870763221,3664609560],[3624381080,2734883394],[310598401,1164996542],[607225278,1323610764],[1426881987,3590304994],[1925078388,4068182383],[2162078206,991336113],[2614888103,633803317],[3248222580,3479774868],[3835390401,2666613458],[4022224774,944711139],[264347078,2341262773],[604807628,2007800933],[770255983,1495990901],[1249150122,1856431235],[1555081692,3175218132],[1996064986,2198950837],[2554220882,3999719339],[2821834349,766784016],[2952996808,2566594879],[3210313671,3203337956],[3336571891,1034457026],[3584528711,2466948901],[113926993,3758326383],[338241895,168717936],[666307205,1188179964],[773529912,1546045734],[1294757372,1522805485],[1396182291,2643833823],[1695183700,2343527390],[1986661051,1014477480],[2177026350,1206759142],[2456956037,344077627],[2730485921,1290863460],[2820302411,3158454273],[3259730800,3505952657],[3345764771,106217008],[3516065817,3606008344],[3600352804,1432725776],[4094571909,1467031594],[275423344,851169720],[430227734,3100823752],[506948616,1363258195],[659060556,3750685593],[883997877,3785050280],[958139571,3318307427],[1322822218,3812723403],[1537002063,2003034995],[1747873779,3602036899],[1955562222,1575990012],[2024104815,1125592928],[2227730452,2716904306],[2361852424,442776044],[2428436474,593698344],[2756734187,3733110249],[3204031479,2999351573],[3329325298,3815920427],[3391569614,3928383900],[3515267271,566280711],[3940187606,3454069534],[4118630271,4000239992],[116418474,1914138554],[174292421,2731055270],[289380356,3203993006],[460393269,320620315],[685471733,587496836],[852142971,1086792851],[1017036298,365543100],[1126000580,2618297676],[1288033470,3409855158],[1501505948,4234509866],[1607167915,987167468],[1816402316,1246189591]],Dc=[[1779033703,4089235720],[3144134277,2227873595],[1013904242,4271175723],[2773480762,1595750129],[1359893119,2917565137],[2600822924,725511199],[528734635,4215389547],[1541459225,327033209]];function Oc(e,t,n){return n===32?[0|t,0|e]:n<32?[e>>>n|t<<32-n,t>>>n|e<<32-n]:[t>>>n-32|e<<64-n,e>>>n-32|t<<64-n]}function kc(e,t,n){return n<32?[e>>>n,t>>>n|e<<32-n]:[0,e>>>n-32]}function Ac(e,t,n,r){let i=(t>>>0)+(r>>>0);return[e+n+(i/4294967296|0)|0,0|i]}function jc(e){let t=e.length,n=Math.floor(t/536870912),r=t<<3>>>0,i=128*Math.ceil((t+17)/128),a=new Uint8Array(i);a.set(e),a[t]=128;let o=new DataView(a.buffer);o.setUint32(i-8,n),o.setUint32(i-4,r);let s=Dc.map(e=>[e[0],e[1]]),c=Array(160);for(let e=0;e<i;e+=128){for(let t=0;t<16;t++)c[2*t]=o.getUint32(e+8*t),c[2*t+1]=o.getUint32(e+8*t+4);for(let e=16;e<80;e++){let t=c[2*(e-15)],n=c[2*(e-15)+1],[r,i]=Oc(t,n,1),[a,o]=Oc(t,n,8),[s,l]=kc(t,n,7),u=r^a^s,d=i^o^l,f=c[2*(e-2)],p=c[2*(e-2)+1],[m,h]=Oc(f,p,19),[g,_]=Oc(f,p,61),[v,y]=kc(f,p,6),b=m^g^v,x=h^_^y,[S,C]=Ac(c[2*(e-16)],c[2*(e-16)+1],u,d);[S,C]=Ac(S,C,c[2*(e-7)],c[2*(e-7)+1]),[S,C]=Ac(S,C,b,x),c[2*e]=S,c[2*e+1]=C}let[t,n]=s[0],[r,i]=s[1],[a,l]=s[2],[u,d]=s[3],[f,p]=s[4],[m,h]=s[5],[g,_]=s[6],[v,y]=s[7];for(let e=0;e<80;e++){let[o,s]=Oc(f,p,14),[b,x]=Oc(f,p,18),[S,C]=Oc(f,p,41),w=o^b^S,ee=s^x^C,te=f&m^~f&g,ne=p&h^~p&_,[T,E]=Ac(v,y,w,ee);[T,E]=Ac(T,E,te,ne),[T,E]=Ac(T,E,Ec[e][0],Ec[e][1]),[T,E]=Ac(T,E,c[2*e],c[2*e+1]);let[D,re]=Oc(t,n,28),[ie,ae]=Oc(t,n,34),[oe,O]=Oc(t,n,39),[se,ce]=Ac(D^ie^oe,re^ae^O,t&r^t&a^r&a,n&i^n&l^i&l);v=g,y=_,g=m,_=h,m=f,h=p,[f,p]=Ac(u,d,T,E),u=a,d=l,a=r,l=i,r=t,i=n,[t,n]=Ac(T,E,se,ce)}s[0]=Ac(s[0][0],s[0][1],t,n),s[1]=Ac(s[1][0],s[1][1],r,i),s[2]=Ac(s[2][0],s[2][1],a,l),s[3]=Ac(s[3][0],s[3][1],u,d),s[4]=Ac(s[4][0],s[4][1],f,p),s[5]=Ac(s[5][0],s[5][1],m,h),s[6]=Ac(s[6][0],s[6][1],g,_),s[7]=Ac(s[7][0],s[7][1],v,y)}let l=new Uint8Array(64),u=new DataView(l.buffer);for(let e=0;e<8;e++)u.setUint32(8*e,s[e][0]>>>0),u.setUint32(8*e+4,s[e][1]>>>0);return l}var Mc=864e5,Nc=Date.UTC(2025,0,1),Pc=Object.assign(Object.create(null),{community:!0,commercial:!0}),Fc=Object.create(null),Ic=[];function Lc(e){let t=``;for(let n=0;n<e.length;n++)t+=(256|e[n]).toString(16).slice(1);return t}function Rc(e,t,n={}){return Rs({valid:e===`active`||e===`grace`,status:e,message:Tc(e,t)},n)}function zc(e){return e.tier===`community`}function Bc(e,t){return zs(this,null,function*(){let n=t.productLabel;if(typeof e!=`string`||e.length>2048||!e.includes(`.`))return Rc(`invalid`,n);let r=e.split(`.`);if(r.length!==2)return Rc(`invalid`,n);let[i,a]=r,o;try{o=function(e){let t=0,n=()=>{for(;t<e.length&&Bs(e[t]);)t++},r=n=>{if(e.substr(t,n.length)!==n)throw new G(`Unexpected token`);t+=n.length},i=()=>{let n=``;for(;;){if(t>=e.length)throw new G(`Unterminated string`);let r=e[t++];if(r===`\"`)return n;if(r<` `)throw new G(`Control character in string`);if(r!==`\\\\`){n+=r;continue}let i=e[t++];switch(i){case`\"`:case`\\\\`:case`/`:n+=i;break;case`b`:n+=`\\b`;break;case`f`:n+=`\\f`;break;case`n`:n+=`\n"
    "`;break;case`r`:n+=`\\r`;break;case`t`:n+=`\t`;break;case`u`:{let r=Hs(e[t])<<12|Hs(e[t+1])<<8|Hs(e[t+2])<<4|Hs(e[t+3]);t+=4,n+=String.fromCharCode(r);break}default:throw new G(`Bad escape`)}}},a=o=>{if(o>16)throw new G(`Too deep`);n();let s=e[t];if(s===void 0)throw new G(`Unexpected end`);if(s===`{`){t++;let r={};if(n(),e[t]===`}`)return t++,r;for(;;){if(n(),e[t]!==`\"`)throw new G(`Expected key`);t++;let s=i();if(s===`__proto__`||Object.prototype.hasOwnProperty.call(r,s))throw new G(`Bad key`);if(n(),e[t]!==`:`)throw new G(`Expected colon`);if(t++,r[s]=a(o+1),n(),e[t]!==`,`){if(e[t]===`}`)return t++,r;throw new G(`Expected , or }`)}t++}}if(s===`[`){t++;let r=[];if(n(),e[t]===`]`)return t++,r;for(;;){if(r.push(a(o+1)),n(),e[t]!==`,`){if(e[t]===`]`)return t++,r;throw new G(`Expected , or ]`)}t++}}if(s===`\"`)return t++,i();if(s===`t`)return r(`true`),!0;if(s===`f`)return r(`false`),!1;if(s===`n`)return r(`null`),null;if(s===`-`||Vs(s))return(()=>{let n=t;if(e[t]===`-`&&t++,e[t]===`0`)t++;else{if(!Vs(e[t]))throw new G(`Bad number`);for(;Vs(e[t]);)t++}if(e[t]===`.`){if(t++,!Vs(e[t]))throw new G(`Bad number`);for(;Vs(e[t]);)t++}if(e[t]===`e`||e[t]===`E`){if(t++,e[t]!==`+`&&e[t]!==`-`||t++,!Vs(e[t]))throw new G(`Bad number`);for(;Vs(e[t]);)t++}return Number(e.slice(n,t))})();throw new G(`Unexpected token`)},o=a(0);if(n(),t!==e.length)throw new G(`Trailing characters`);return o}(function(e){let t=[],n=0;for(;n<e.length;){let r=e[n++];if(r<128){t.push(r);continue}let i,a,o;if(r>=194&&r<=223)i=31&r,a=1,o=128;else if(r>=224&&r<=239)i=15&r,a=2,o=2048;else{if(!(r>=240&&r<=244))throw new G(`Invalid UTF-8`);i=7&r,a=3,o=65536}if(n+a>e.length)throw new G(`Invalid UTF-8`);for(let t=0;t<a;t++){let t=e[n++];if((192&t)!=128)throw new G(`Invalid UTF-8`);i=i<<6|63&t}if(i<o||i>1114111||i>=55296&&i<=57343)throw new G(`Invalid UTF-8`);i>=65536?(i-=65536,t.push(55296|i>>10,56320|1023&i)):t.push(i)}let r=``;for(let e=0;e<t.length;e+=4096)r+=String.fromCharCode.apply(null,t.slice(e,e+4096));return r}(Ws(i)))}catch{return Rc(`invalid`,n)}if(!o||typeof o!=`object`||Array.isArray(o))return Rc(`invalid`,n);let s=o,c=e=>Object.prototype.hasOwnProperty.call(s,e)?s[e]:void 0,l={id:c(`id`),product:c(`product`),tier:c(`tier`),type:c(`type`),iat:c(`iat`),exp:c(`exp`)};if(typeof l.product!=`string`||typeof l.type!=`string`||!Number.isFinite(l.exp)||!Number.isFinite(l.iat)||typeof l.id!=`string`||l.tier!==void 0&&(typeof l.tier!=`string`||!0!==Pc[l.tier])||l.product===`primeui`&&l.tier===void 0)return Rc(`invalid`,n);let u,d,f;try{u=Ws(a),d=new TextEncoder().encode(i)}catch{return Rc(`invalid`,n)}try{f=function(e){if(!/^[0-9a-fA-F]*$/.test(e))throw Error(`Invalid hex character`);let t=new Uint8Array(32);for(let n=0;n<t.length;n++)t[n]=parseInt(e.slice(2*n,2*n+2),16);return t}(`dae75e66b9f59bebf87d4bb29ca6494f37deccfcc2b132b98ee159ee7505373b`)}catch{return Rc(`invalid`,n)}let p=!1;try{p=yield function(e,t,n){return zs(this,null,function*(){if(!function(e,t,n){let r=Lc(n)+`:`+Lc(e)+`:`+Lc(t),i=Fc[r];if(!0===i||!1===i)return i;let a=function(e,t,n,r){if(e.length!==64||n.length!==32)return!1;let i=e.slice(32,64);if(!bc(i))return!1;let a=gc(n);if(!a||vc(a))return!1;let o=e.slice(0,32),s=gc(o);if(!s||vc(s))return!1;let c=new Uint8Array(64+t.length);c.set(o,0),c.set(n,32),c.set(t,64);let l=function(e){let t=new Uint8Array(32);for(let n=e.length-1;n>=0;n--){let r=e[n];for(let e=0;e<32;e++){let n=256*t[e]+r;t[e]=255&n,r=n>>8}for(;r>0||!bc(t);){let e=0;for(let n=0;n<32;n++){let r=t[n]-yc[n]-e;e=+(r<0),t[n]=r+256*e}r-=e,r<0&&(r=0)}}return t}(r(c)),u=hc(i,_c);return f=pc(s,hc(l,a)),sc(K((d=u).x,f.z),K(f.x,d.z))&&sc(K(d.y,f.z),K(f.y,d.z));var d,f}(e,t,n,jc);return Ic.length>=32&&delete Fc[Ic.shift()],Ic.push(r),Fc[r]=!0===a,!0===a}(e,t,n))return!1;let r=yield function(e,t,n){return zs(this,null,function*(){let r=(typeof globalThis<`u`?globalThis:typeof self<`u`?self:typeof window<`u`?window:void 0)?.crypto?.subtle;if(r)try{let i=yield r.importKey(`raw`,n,{name:`Ed25519`},!1,[`verify`]);return yield r.verify({name:`Ed25519`},i,e,t)}catch{return}})}(e,t,n);return r===void 0||!0===r})}(u,d,f)}catch{return Rc(`tampered`,n,{payload:l})}if(!p)return Rc(`tampered`,n,{payload:l});if(!function(e,t){return e.product===t||!(!t.startsWith(`primeui-pro:`)||e.product!==`primeui`||e.tier!==`commercial`)}(l,t.product))return Rc(`wrong-product`,n,{payload:l});let m=1e3*l.exp,h=Date.now(),g=Math.floor((m-h)/Mc),_=function(e){let t=function(e){if(typeof e==`number`)return Number.isFinite(e)?1e3*e:null;if(typeof e!=`string`||e===``)return null;let t=Date.parse(e);return Number.isNaN(t)?null:t}(e);return t!==null&&t>=Nc?t:null}(t.releaseDate);if(_===null&&!zc(l))return Rc(`invalid`,n,{daysUntilExpiry:g,payload:l});if(_!==null&&_>m)return Rc(`expired`,n,{daysUntilExpiry:g,payload:l});if(zc(l)){if(h>m+30*Mc)return Rc(`expired`,n,{daysUntilExpiry:g,payload:l});if(h>m)return Rc(`grace`,n,{daysUntilExpiry:g,payload:l})}return Rc(`active`,n,{daysUntilExpiry:g,payload:l})})}function Vc(e,t){let n=Object.prototype.hasOwnProperty.call(e,t)?e[t]:void 0;return typeof n==`string`?n:void 0}function Hc(e,t){return{valid:!1,status:e,message:Tc(e,t)}}function Uc(e,t){let n={};return{verify(t,r){return zs(this,null,function*(){let i=wc(t),a=Cc(t),o=r?.releaseDate;if(!i)return Hc(`invalid`,a);let s=Vc(e,t),c=Vc(e,`primeui`);if(s){let e=yield Bc(s,Rs({product:i,productLabel:a,releaseDate:o},n));if(e.valid||e.status!==`wrong-product`)return e}return c&&t!==`primeui`&&i.startsWith(`primeui-pro:`)?Bc(c,Rs({product:i,productLabel:a,releaseDate:o},n)):Hc(s?`wrong-product`:`missing`,a)})},has(t){let n=wc(t);return!!n&&(!!Vc(e,t)||t!==`primeui`&&n.startsWith(`primeui-pro:`)&&!!Vc(e,`primeui`))}}}var Wc=null;function Gc(e,t){if(!e)throw Error(`[@primeui/license-manager] registerLicense: keys argument is required.`);return Wc=Uc(e)}function Kc(e,t){if(!Wc){let t=Cc(e);return Promise.resolve({valid:!1,status:`unconfigured`,message:Tc(`unconfigured`,t)})}return Wc.verify(e,t)}function qc(){let e=new Map,t={on(n,r){let i=e.get(n);return i?i.push(r):i=[r],e.set(n,i),t},off(n,r){let i=e.get(n);if(i){let e=i.indexOf(r);e!==-1&&i.splice(e,1)}return t},emit(t,...n){let r=e.get(t);r&&r.forEach(e=>{e(n[0])})},clear(){e.clear()}};return t}var Jc=Object.defineProperty,Yc=Object.getOwnPropertySymbols,Xc=Object.prototype.hasOwnProperty,Zc=Object.prototype.propertyIsEnumerable,Qc=(e,t,n)=>t in e?Jc(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,$c=(e,t)=>{for(var n in t||={})Xc.call(t,n)&&Qc(e,n,t[n]);if(Yc)for(var n of Yc(t))Zc.call(t,n)&&Qc(e,n,t[n]);return e};function el(e){return e==null||e===``||Array.isArray(e)&&e.length===0||!(e instanceof Date)&&typeof e==`object`&&Object.keys(e).length===0}function tl(e,t,n){if(e===t||e!==e&&t!==t)return!0;if(!e||!t||typeof e!=`object`||typeof t!=`object`)return!1;n||=new WeakMap;let r=n.get(e);if(r!=null&&r.has(t))return!0;r||n.set(e,r=new WeakSet),r.add(t);let i=Array.isArray(e),a=Array.isArray(t),o=!0;if(i&&a){if(e.length!==t.length)o=!1;else for(let r=e.length;r--!==0;)if(!tl(e[r],t[r],n)){o=!1;break}}else if(i!==a)o=!1;else{let r=e instanceof Date,i=t instanceof Date;if(r!==i)o=!1;else if(r&&i)o=e.getTime()===t.getTime();else{let r=e instanceof RegExp,i=t instanceof RegExp;if(r!==i)o=!1;else if(r&&i)o=e.toString()===t.toString();else if(e instanceof Map||t instanceof Map){if(!(e instanceof Map&&t instanceof Map)||e.size!==t.size)o=!1;else for(let[r,i]of e)if(!t.has(r)||!tl(i,t.get(r),n)){o=!1;break}}else if(e instanceof Set||t instanceof Set){if(!(e instanceof Set&&t instanceof Set)||e.size!==t.size)o=!1;else for(let n of e)if(!t.has(n)){o=!1;break}}else{let r=Object.keys(e),i=r.length;if(i!==Object.keys(t).length)o=!1;else{for(let e=i;e--!==0;)if(!Object.prototype.hasOwnProperty.call(t,r[e])){o=!1;break}if(o)for(let a=i;a--!==0;){let i=r[a];if(!tl(e[i],t[i],n)){o=!1;break}}}}}}return o||r.delete(t),o}function nl(e,t){return tl(e,t)}function rl(e){return typeof e==`function`&&`call`in e&&`apply`in e}function q(e){return!el(e)}function il(e,t){if(!e||!t)return null;let n=e;try{let e=n[t];if(q(e))return e}catch{}if(Object.keys(n).length){if(rl(t))return t(e);if(t.indexOf(`.`)===-1)return n[t];{let n=t.split(`.`),r=e;for(let e=0,t=n.length;e<t;++e){if(r==null)return null;r=r[n[e]]}return r}}return null}function al(e,t,n){return n?il(e,n)===il(t,n):nl(e,t)}function ol(e,t=!0){return e instanceof Object&&e.constructor===Object&&(t||Object.keys(e).length!==0)}var sl=new Set([`__proto__`,`constructor`,`prototype`]);function cl(e,t,n,r=new WeakSet){let i=$c({},e);Object.keys(i).length===0&&!n.has(t)&&n.set(t,i);let a=!r.has(t);return a&&r.add(t),Object.keys(t).forEach(a=>{if(sl.has(a))return;let o=a,s=t[o];ol(s)&&o in e&&ol(e[o])?i[o]=r.has(s)?n.get(s)??cl({},s,n,r):cl(e[o],s,n,r):ol(s)?i[o]=n.get(s)??cl({},s,n,r):i[o]=s}),a&&r.delete(t),i}function ll(...e){return e.reduce((e,t)=>cl(e,t||{},new WeakMap),{})}function ul(e,...t){return rl(e)?e(...t):e}function dl(e,t=!0){return typeof e==`string`&&(t||e!==``)}function fl(e){return dl(e)?e.replace(/(-|_)/g,``).toLowerCase():e}function pl(e,t=``,n={}){let r=fl(t).split(`.`),i=r.shift();return i?ol(e)||Array.isArray(e)?pl(ul(e[Object.keys(e).find(e=>fl(e)===i)||``],n),r.join(`.`),n):void 0:ul(e,n)}function ml(e,t=!0){return Array.isArray(e)&&(t||e.length!==0)}function hl(e){return q(e)&&!isNaN(e)}function gl(e,t){if(t){t.lastIndex=0;let n=t.test(e);return t.lastIndex=0,n}return!1}function _l(...e){return ll(...e)}function vl(e,t){let n=0;for(;t-1-n>=0&&e[t-1-n]===`\\\\`;)n++;return n%2==1}function yl(e){return e.replace(/[\\r\\n\\t]+/g,``).replace(/ {2,}/g,` `).replace(/ ([{:}]) /g,`$1`).replace(/([;,]) /g,`$1`).replace(/ !/g,`!`).replace(/: /g,`:`)}function bl(e){if(!e)return e;let t=``,n=``,r=0;for(;r<e.length;){let i=e[r];if(i===`/`&&e[r+1]===`*`){let t=e.indexOf(`*/`,r+2);r=t===-1?e.length:t+2}else if(i===`\"`||i===`'`){t+=yl(n),n=``;let a=r+1;for(;a<e.length&&(e[a]!==i||vl(e,a));)a++;t+=e.slice(r,Math.min(a+1,e.length)),r=a+1}else n+=i,r++}return(t+yl(n)).trim()}function xl(e={},t=``){return Object.entries(e).reduce((e,[n,r])=>{let i=t?`${t}.${n}`:n;return ol(r)?e=e.concat(xl(r,i)):e.push(i),e},[])}function Sl(e){return dl(e,!1)?e[0].toUpperCase()+e.slice(1):e}function Cl(e){return dl(e)?e.replace(/(_)/g,`-`).replace(/([a-z])([A-Z])/g,`$1-$2`).toLowerCase():e}function wl(e){return dl(e)?e.replace(/[A-Z]/g,(e,t)=>t===0?e:`.`+e.toLowerCase()).toLowerCase():e}function Tl(...e){let t=[];for(let n=0;n<e.length;n++){let r=e[n];if(!r)continue;let i=typeof r;if(i===`string`||i===`number`)t.push(r);else if(i===`object`){let e=Array.isArray(r)?[Tl(...r)]:Object.entries(r).map(([e,t])=>t?e:void 0);t=e.length?t.concat(e.filter(e=>!!e)):t}}return t.join(` `).trim()}var El,Dl=e=>e.replace(/[.*+?^${}()|[\\]\\\\]/g,`\\\\$&`);function Ol(e){let{prefix:t,from:n=`p`,keep:r=[]}=typeof e==`object`&&e?e:{prefix:e};return{prefix:t&&t!==n?t:void 0,from:n,keep:r}}function kl(e,t,n){let r=`${t}\n"
    "${e}\n"
    "${n.join(`\n"
    "`)}`;El??=new Map;let i=El.get(r);if(!i){let a=n.length?`(?!${Dl(t)}-(?:${n.map(Dl).join(`|`)})(?!\\\\w))`:``,o=`(?!${Dl(`${e}-`)})${a}`,s=Dl(`${t}-`);i={token:RegExp(`(^|\\\\s)${o}${s}(?=[a-z])`,`g`),dot:RegExp(`\\\\.${o}${s}(?=[a-z])`,`g`),attribute:/\\[\\s*class\\s*([~|^$*]?=)\\s*(?:\"([^\"]*)\"|'([^']*)'|([^\\s\"'\\]]+))/gi,value:RegExp(`(^|\\\\s)${o}${s}(?=[a-z]|$)`,`g`)},El.size>32&&El.clear(),El.set(r,i)}return i}function Al(e,...t){let n=Tl(...t),{prefix:r,from:i,keep:a}=Ol(e);return!n||!r?n:n.replace(kl(r,i,a).token,`$1${r}-`)}function jl(e,t){let{prefix:n,from:r,keep:i}=Ol(e);if(!t||!n)return t;let{dot:a,attribute:o,value:s}=kl(n,r,i),c=e=>e.replace(s,`$1${n}-`);return t.replace(a,`.${n}-`).replace(o,(e,t,n,r,i)=>{let a=n??r??i??``,o=n===void 0?r===void 0?``:`'`:`\"`;return e.slice(0,e.length-a.length-o.length*2)+o+c(a)+o})}function Ml(e,t){return e?e.classList?e.classList.contains(t):RegExp(`(^| )`+t+`( |$)`,`gi`).test(e.className):!1}function Nl(e,t){if(e&&t){let n=t=>{Ml(e,t)||(e.classList?e.classList.add(t):e.className+=` `+t)};[t].flat().filter(Boolean).forEach(e=>e.split(` `).forEach(n))}}function Pl(e,t){if(e&&t){let n=t=>{e.classList?e.classList.remove(t):e.className=e.className.replace(RegExp(`(^|\\\\b)`+t.split(` `).join(`|`)+`(\\\\b|$)`,`gi`),` `)};[t].flat().filter(Boolean).forEach(e=>e.split(` `).forEach(n))}}function Fl(e){return e?Math.abs(e.scrollLeft):0}var Il=/expression\\s*\\(|url\\s*\\(\\s*['\"]?\\s*(?:javascript|vbscript):|@import\\s+['\"]?\\s*(?:javascript|vbscript|data):/i,Ll=/url\\s*\\(\\s*['\"]?\\s*(data:[^'\")]*)/gi,Rl=new Set([`href`,`src`,`xlink:href`,`action`,`formaction`]),zl=new Set([`http`,`https`,`mailto`,`tel`,`sms`,`ftp`,`ftps`,`blob`]),Bl=/^data:image\\/(?:png|gif|jpeg|jpg|webp|bmp|avif);base64,[a-z0-9+/=\\s]+$/i;function Vl(e){if(typeof e!=`string`)return!1;if(Il.test(e))return!0;Ll.lastIndex=0;let t;for(;t=Ll.exec(e);)if(!Bl.test(t[1].trim()))return!0;return!1}function Hl(e){let t=``;for(let n of e){let e=n.charCodeAt(0);e<=31||e===127||/\\s/.test(n)||(t+=n)}return t}function Ul(e,t){let n=Hl(e),r=t.toLowerCase();if(n.startsWith(`#`)||n.startsWith(`/`)||n.startsWith(`./`)||n.startsWith(`../`)||n.startsWith(`?`))return!0;let i=(n.match(/^([a-z][a-z0-9+.-]*):/i)?.[1])?.toLowerCase();return i?i===`data`?(r===`src`||r===`xlink:href`)&&Bl.test(e.trim()):zl.has(i):!0}function Wl(e,t){return typeof t==`string`&&Rl.has(e.toLowerCase())&&!Ul(t,e)}function Gl(e,t){return e.toLowerCase()===`srcdoc`&&typeof t==`string`&&/<\\s*script\\b|on\\w+\\s*=|javascript:|data:text\\/html/i.test(t)}function Kl(e){return e.startsWith(`--`)?e:e.replace(/([a-z])([A-Z])/g,`$1-$2`).toLowerCase()}function ql(e,t,n={}){n.clear&&(e.style.cssText=``),t.forEach(t=>{let n=t.indexOf(`:`);if(n<0)return;let r=t.slice(0,n).trim(),i=t.slice(n+1).trim();if(!r||Vl(i))return;let a=``;/!\\s*important$/i.test(i)&&(i=i.replace(/!\\s*important$/i,``).trim(),a=`important`),e.style.setProperty(r,i,a)})}function Jl(e,t){let n=0;for(;t-1-n>=0&&e[t-1-n]===`\\\\`;)n++;return n%2==1}function Yl(e){let t=[],n=0,r=``,i=0;for(let a=0;a<e.length;a++){let o=e[a];r?o===r&&!Jl(e,a)&&(r=``):o===`'`||o===`\"`?r=o:o===`(`?i++:o===`)`?i=Math.max(0,i-1):o===`;`&&i===0&&(t.push(e.slice(n,a)),n=a+1)}return t.push(e.slice(n)),t}function Xl(e,t,n={}){typeof t==`string`?ql(e,Yl(t),n):(n.clear&&(e.style.cssText=``),Object.entries(t).forEach(([t,n])=>{if(n==null||Vl(n))return;let r=String(n),i=``;/!\\s*important$/i.test(r)&&(r=r.replace(/!\\s*important$/i,``).trim(),i=`important`),e.style.setProperty(Kl(t),r,i)}))}function Zl(e,t){if(e instanceof HTMLElement){let n=e.offsetWidth;if(t){let t=getComputedStyle(e);n+=parseFloat(t.marginLeft)+parseFloat(t.marginRight)}return n}return 0}function Ql(e){if(e){let t=e.parentNode;return t&&t instanceof ShadowRoot&&t.host&&(t=t.host),t}return null}function $l(e){return!!(e!=null&&e.nodeName&&Ql(e))}function eu(e){return typeof Element<`u`?e instanceof Element:typeof e==`object`&&!!e&&e.nodeType===1&&typeof e.nodeName==`string`}function tu(e,t,n){if(typeof n!=`function`&&!(typeof n==`object`&&n&&`handleEvent`in n))return;let r=e,i=r._pListeners||=[],a=!1;for(let r=i.length-1;r>=0;r--)i[r][0]===t&&(i[r][1]===n?a=!0:(e.removeEventListener(t,i[r][1]),i.splice(r,1)));a||(e.addEventListener(t,n),i.push([t,n]))}function nu(){if(window.getSelection){let e=window.getSelection()||{};e.empty?e.empty():e.removeAllRanges&&e.rangeCount&&e.rangeCount>0&&e.getRangeAt&&e.getRangeAt(0).getClientRects().length>0&&e.removeAllRanges()}}function ru(e,t={}){if(eu(e)){let n=e?.$attrs,r=(e,t)=>{let i=n!=null&&n[e]?[n[e]]:[];return[t].flat().reduce((t,n)=>{if(n!=null){let i=typeof n;if(i===`string`||i===`number`)t.push(n);else if(i===`object`){let i=Array.isArray(n)?r(e,n):Object.entries(n).map(([t,n])=>e===`style`&&(n||n===0)?`${t.replace(/([a-z])([A-Z])/g,`$1-$2`).toLowerCase()}:${n}`:n?t:void 0);t=i.length?t.concat(i.filter(e=>!!e)):t}}return t},i)},i=t=>{ql(e,r(`style`,t))},a=e;Object.entries(t).forEach(([t,n])=>{if(n!=null){let o=t.match(/^on(.+)/);if(o)tu(e,o[1].toLowerCase(),n);else if(t===`p-bind`||t===`pBind`)ru(e,n);else if(t===`style`)i(n),a.$attrs=a.$attrs||{},a.$attrs[t]=e.style.cssText;else{if(Wl(t,n)||Gl(t,n))return;n=t===`class`?[...new Set(r(`class`,n))].join(` `).trim():n,a.$attrs=a.$attrs||{},a.$attrs[t]=n,e.setAttribute(t,n)}}})}}function iu(e,t={},...n){if(e){let r=document.createElement(e);return ru(r,t),r.append(...n),r}}function au(e){return(e??``).replace(/<\\/(style)/gi,`<\\\\/$1`)}function ou(e){return String(e).replace(/&/g,`&amp;`).replace(/\"/g,`&quot;`).replace(/</g,`&lt;`).replace(/>/g,`&gt;`)}function su(e={}){return Object.keys(e).filter(e=>/^[^\\s\"'<>/=]+$/.test(e)).map(t=>`${t}=\"${ou(e[t])}\"`)}function cu(e,t){if(eu(e)){let n=e.getAttribute(t);return n!==null&&n.trim()!==``&&!isNaN(n)?+n:n===`true`||n===`false`?n===`true`:n}}function lu(e){if(e){let t=e.offsetHeight,n=getComputedStyle(e);return t-=parseFloat(n.paddingTop)+parseFloat(n.paddingBottom)+parseFloat(n.borderTopWidth)+parseFloat(n.borderBottomWidth),t}return 0}function uu(e){if(e){let t=e.getBoundingClientRect();return{top:t.top+(window.pageYOffset||document.documentElement.scrollTop||document.body.scrollTop||0),left:t.left+(window.pageXOffset||Fl(document.documentElement)||Fl(document.body)||0)}}return{top:`auto`,left:`auto`}}function du(e,t){if(e){let n=e.offsetHeight;if(t){let t=getComputedStyle(e);n+=parseFloat(t.marginTop)+parseFloat(t.marginBottom)}return n}return 0}function fu(){if(window.getSelection)return window.getSelection().toString();if(document.getSelection)return document.getSelection().toString()}function pu(e){if(e){let t=e.offsetWidth,n=getComputedStyle(e);return t-=parseFloat(n.paddingLeft)+parseFloat(n.paddingRight)+parseFloat(n.borderLeftWidth)+parseFloat(n.borderRightWidth),t}return 0}function mu(){return!!(typeof window<`u`&&window.document&&window.document.createElement)}function hu(e,t=``,n){if(eu(e)&&n!=null){let r=t.toLowerCase();if(/^on[a-z]/.test(r)){tu(e,r.slice(2),n);return}if(r===`style`){typeof n==`string`?Xl(e,n,{clear:!0}):typeof n==`object`&&Xl(e,n);return}if(Wl(t,n)||Gl(t,n))return;e.setAttribute(t,n)}}function gu(e){return`\"${String(e).replace(/[\"\\\\]/g,`\\\\$&`).replace(/[\\n\\r\\f]/g,e=>`\\\\${e.charCodeAt(0).toString(16)} `)}\"`}var _u={};function vu(e=`pui_id_`){return Object.hasOwn(_u,e)||(_u[e]=0),_u[e]++,`${e}${_u[e]}`}var yu=Object.defineProperty,bu=Object.defineProperties,xu=Object.getOwnPropertyDescriptors,Su=Object.getOwnPropertySymbols,Cu=Object.prototype.hasOwnProperty,wu=Object.prototype.propertyIsEnumerable,Tu=(e,t,n)=>t in e?yu(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,Eu=(e,t)=>{for(var n in t||={})Cu.call(t,n)&&Tu(e,n,t[n]);if(Su)for(var n of Su(t))wu.call(t,n)&&Tu(e,n,t[n]);return e},Du=(e,t)=>bu(e,xu(t)),Ou=(e,t)=>{var n={};for(var r in e)Cu.call(e,r)&&t.indexOf(r)<0&&(n[r]=e[r]);if(e!=null&&Su)for(var r of Su(e))t.indexOf(r)<0&&wu.call(e,r)&&(n[r]=e[r]);return n},ku=qc(),Au=/{([^{}]*)}/g,ju=/(?:^|\\D)(\\d+\\s+[+*/-]\\s+\\d+)/g,Mu=/var\\((?:(?!var\\()[^)])+\\)/g;function Nu(e){return dl(e)?e.replace(/[A-Z]/g,(e,t)=>t===0?e:`.`+e.toLowerCase()).toLowerCase():e}function Pu(e){return ol(e)&&Object.prototype.hasOwnProperty.call(e,`$value`)&&Object.prototype.hasOwnProperty.call(e,`$type`)?e.$value:e}function Fu(e){return/[^\\w-]/.test(e)?e.replace(/ /g,``).replace(/[^\\w]/g,`-`):e}function Iu(e=``,t=``){return Fu(`${dl(e,!1)&&dl(t,!1)?`${e}-`:e}${t}`)}function Lu(e=``,t=``){return`--${Iu(e,t)}`}function Ru(e=``){let t=0;for(let n=0;n<e.length;n++){let r=e.charCodeAt(n);r!==123&&r!==125||t++}return t%2!=0}var zu=e=>e.indexOf(`\\r`)===-1&&e.indexOf(`\\f`)===-1?e:e.replace(/\\r\\n?|\\f/g,`\n"
    "`),Bu=e=>e===32||e===9||e===10||e===12||e===13,Vu=e=>e>=48&&e<=57||(32|e)>=97&&(32|e)<=102,Hu=e=>e>=128||e===45||e===95||e>=48&&e<=57||(32|e)>=97&&(32|e)<=122;function Uu(e){if(e.indexOf(`\\0`)!==-1)return!1;let t=zu(e),n=t.length,r=0,i=0,a=!1,o=0;for(;o<n;){let e=t[o++];if(e===`\\\\`){if(o>=n)return!1;let e=0;for(;e<6&&Vu(t.charCodeAt(o));)o++,e++;e?Bu(t.charCodeAt(o))&&o++:o++,a=!0}else if(e===`\"`||e===`'`)for(;;){if(o>=n)return!1;let r=t[o++];if(r===e)break;if(r===`\\\\`)o++;else if(r===`\n"
    "`||r===`\\r`||r===`\\f`)return!1}else if(e===`(`){if(a)return!1;if((32|t.charCodeAt(o-2))==108&&(32|t.charCodeAt(o-3))==114&&(32|t.charCodeAt(o-4))==117&&!Hu(t.charCodeAt(o-5))&&t[o-5]!==`#`&&t[o-5]!==`@`){let e=o;for(;e<n&&Bu(t.charCodeAt(e));)e++;if(t[e]!==`\"`&&t[e]!==`'`){for(;;){if(e>=n)return!1;let r=t.charCodeAt(e++);if(r===41)break;if(Bu(r)){for(;e<n&&Bu(t.charCodeAt(e));)e++;if(t.charCodeAt(e++)!==41)return!1;break}if(r===34||r===39||r===40||r<=8||r===11||r>=14&&r<=31||r===127)return!1;if(r===92){if(e>=n||t.charCodeAt(e)===10)return!1;let r=0;for(;r<6&&Vu(t.charCodeAt(e));)e++,r++;r?Bu(t.charCodeAt(e))&&e++:e++}}o=e;continue}}if(i++>29)return!1;r<<=1}else if(e===`[`){if(i++>29)return!1;r=r<<1|1}else if(e===`)`||e===`]`){if(i--<1||(1&r)!=+(e===`]`))return!1;r>>=1}else if(e===`/`&&t[o]===`*`){let e=t.indexOf(`*/`,o+1);if(e<0)return!1;o=e+2}else{if(e===`;`||e===`{`||e===`}`||e===`<`)return!1;Hu(t.charCodeAt(o-1))||(a=!1)}}return i===0}function Wu(e,t=``,n=``,r=[],i){if(dl(e)){if(e.length>1e5)return;let t=e.trim();if(Ru(t))return;if(gl(t,Au)){let e=t.replace(Au,e=>`var(${Lu(n,Cl(e.replace(/{|}/g,``).split(`.`).filter(e=>!r.some(t=>gl(e,t))).join(`-`)))}${q(i)?`, ${i}`:``})`);return Uu(e)?gl(e.replace(Mu,`0`),ju)?`calc(${e})`:e:void 0}return Uu(t)?t:void 0}if(hl(e))return e}function Gu(e,t,n){dl(t,!1)&&e.push(`${t}:${n};`)}function Ku(e,t){return e?`${e}{${t}}`:``}var qu,Ju=[`icon`];function Yu(e,t){if(!t||t===`p`||typeof e!=`string`||!e)return e;let n=/^p-[a-z]/.test(t)?`p\\\\-${t.slice(2)}`:t,{css:r,attribute:i}=qu??(()=>{let e=`(?!(?:${Ju.join(`|`)})(?!\\\\w))`,t=new RegExp(String.raw`\\/\\*[\\s\\S]*?(?:\\*\\/|$)|\"(?:[^\"\\\\\\n\\r\\f]|\\\\[\\s\\S])*\"?|'(?:[^'\\\\\\n\\r\\f]|\\\\[\\s\\S])*'?|\\\\[\\s\\S]|(?:^|[^\\w\\\\-])[uU][rR][lL]\\((?!\\s*[\"'])[^)]*\\)?|(\\.p-${e}(?=[a-z]))|(\\[\\s*[cC][lL][aA][sS][sS]\\s*[~^*]=\\s*(?:\"[ \\t]*p-(?:[^\"\\\\\\n\\r\\f]|\\\\[\\s\\S])*\"?|'[ \\t]*p-(?:[^'\\\\\\n\\r\\f]|\\\\[\\s\\S])*'?|p-))`,`g`);return qu??={css:t,attribute:new RegExp(String.raw`(=\\s*[\"']?[ \\t]*)p-${e}`)}})();return e.replace(r,(e,t,r)=>t?`.${n}-`:r?r.replace(i,(e,t)=>`${t}${n}-`):e)}function Xu(e,t){let n=e.indexOf(`dt(`);if(n===-1)return e;let r=[];for(;n!==-1;){let t=1,i=e.indexOf(`dt(`,n+3),a=e.indexOf(`)`,n+3);for(;a!==-1;)if(i!==-1&&i<a)t++,i=e.indexOf(`dt(`,i+3);else{if(--t===0)break;a=e.indexOf(`)`,a+1)}if(a===-1)break;r.push([n,a]),n=e.indexOf(`dt(`,a+1)}if(!r.length)return e;let i=Array(r.length);for(let n=r.length-1;n>=0;n--){let[a,o]=r[n];i[n]=String(t(...Zu(e.slice(a+3,o),t)))}let a=``,o=0;for(let t=0;t<r.length;t++)a+=e.slice(o,r[t][0])+i[t],o=r[t][1]+1;return a+e.slice(o)}function Zu(e,t){let n=[],r=0,i=null,a=0;for(let o=0;o<=e.length;o++){let s=e[o];if(s!==`\"`&&s!==`'`&&s!==\"`\"||e[o-1]===`\\\\`||(i=i===s?null:s),!i&&(s===`(`&&a++,s===`)`&&a--,(s===`,`||o===e.length)&&a===0)){let i=e.slice(r,o).trim();n.push(i.startsWith(`dt(`)?Xu(i,t):Qu(i)),r=o+1}}return n}function Qu(e){let t=e[0];if((t===`\"`||t===`'`||t===\"`\")&&e[e.length-1]===t)return e.slice(1,-1);let n=Number(e);return isNaN(n)?e:n}var $u=`primeui.base`;function ed(e,t){let n=t?.replace(/^@layer\\s*/,``).replace(/;\\s*$/,``).trim();if(e&&n){let t=n.split(`,`).map(e=>e.trim()),r=t.indexOf(e.slice(0,e.lastIndexOf(`.`)));if(r>0)return t.splice(r+1,0,e),`@layer ${t.join(`,`)};`}let r=[e,n].filter(q).join(`,`);return q(r)?`@layer ${r};`:t}var td={variable:{prefix:`p`,selector:`:root,:host`,excludedKeyRegex:/^(primitive|semantic|components|directives|variables|colorscheme|light|dark|common|root|states|extend|css)$/gi},options:{prefix:`p`,darkModeSelector:`system`,cssLayer:!1,cssVariables:!0,scoped:!1}},nd=(e,t,n)=>({name:t,theme:e.theme,params:n,defaults:e.defaults,region:e._region(),set:{layerNames:e.setLayerNames.bind(e)},dt:e._dt}),rd=/^[A-Za-z_][\\w-]*$/,id=e=>{if(e!==void 0&&e!==``&&(typeof e!=`string`||!rd.test(e)))throw TypeError(`[primeuix] classPrefix takes letters, digits, \"_\" or \"-\".`)},J=new class{constructor(e){this.defaults=td,this._theme=void 0,this._layerNames=new Set,this._loadedStyleNames=new Set,this._loadingStyles=new Set,this._tokens={},this._scopedTokenPaths=new Set;for(let e=Object.getPrototypeOf(this);e&&e!==Object.prototype;e=Object.getPrototypeOf(e))for(let t of Object.getOwnPropertyNames(e))t===`constructor`||Object.prototype.hasOwnProperty.call(this,t)||Object.defineProperty(this,t,Du(Eu({},Object.getOwnPropertyDescriptor(e,t)),{enumerable:!0}));this.service=e}static get defaults(){return td}_styleId(){return this.key?this.styleName.bind(this):void 0}_region(){}styleName(e){let{prefix:t=`p`,classPrefix:n=`p`}=this.options,r=this.key??(n===`p`?``:`${t}-${n}`);return r?`${e}@${r}`:e}update(e={}){let{theme:t}=e;if(t){let e=Eu(Eu({},this.defaults.options),t.options);id(e.classPrefix),this._theme=Du(Eu({},t),{options:e}),this._tokens=gd.createTokens(this.preset,this.defaults),this.resetCaches()}}get theme(){return this._theme}get preset(){return this.theme?.preset||{}}get options(){return this.theme?.options||{}}get tokens(){return this._tokens}hasScopedTokenPath(e){return this._scopedTokenPaths.has(e)}getScopedTokenPaths(){return[...this._scopedTokenPaths]}addScopedToken(e){let t=!1;return q(e)&&xl(e).forEach(e=>{let n=wl(e);this._scopedTokenPaths.has(n)||(this._scopedTokenPaths.add(n),t=!0)}),t}clearScopedTokenPaths(){this._scopedTokenPaths.clear()}getTheme(){return this.theme}setTheme(e){this.update({theme:e}),this.service.emit(`theme:change`,e)}getPreset(){return this.preset}setPreset(e){this._theme=Du(Eu({},this.theme),{preset:e}),this._tokens=gd.createTokens(e,this.defaults),this.resetCaches(),this.service.emit(`preset:change`,e),this.service.emit(`theme:change`,this.theme)}getOptions(){return this.options}setOptions(e){id(e?.classPrefix),this._theme=Du(Eu({},this.theme),{options:e}),this.resetStyleCaches(),this.service.emit(`options:change`,e),this.service.emit(`theme:change`,this.theme)}resetStyleCaches(){this.clearLoadedStyleNames(),this.clearLayerNames()}resetCaches(){this.resetStyleCaches(),this.clearScopedTokenPaths()}getLayerNames(){return[...this._layerNames]}setLayerNames(e){this._layerNames.add(e)}clearLayerNames(){this._layerNames.clear()}getLoadedStyleNames(){return this._loadedStyleNames}isStyleNameLoaded(e){return this._loadedStyleNames.has(e)}setLoadedStyleName(e){this._loadedStyleNames.add(e)}deleteLoadedStyleName(e){this._loadedStyleNames.delete(e)}clearLoadedStyleNames(){this._loadedStyleNames.clear()}getTokenValue(e){return gd.getTokenValue(this.tokens,e,this.defaults)}getCommon(e=``,t){return gd.getCommon(nd(this,e,t))}getComponent(e=``,t){return gd.getPresetC(nd(this,e,t))}getDirective(e=``,t){return gd.getPresetD(nd(this,e,t))}getCustomPreset(e=``,t,n,r){let i=Eu({},this._region());return n&&Object.assign(i,{selector:i.within?.call(i,n)??n,darkSelector:i.within?.call(i,n,!0)}),gd.getPreset(Du(Eu({},nd(this,e,r)),{preset:t,options:this.options,region:i,isScopedTokenPaths:!0}))}getLayerOrderCSS(e=``){return gd.getLayerOrder(e,this.options,{names:this.getLayerNames()},this.defaults)}transformCSS(e=``,t,n=`style`,r){return gd.transformCSS(e,t,r,n,this.options,{layerNames:this.setLayerNames.bind(this)},this.defaults,void 0,this._region())}getCommonStyleSheet(e=``,t,n={},r){return gd.getCommonStyleSheet(Du(Eu({},nd(this,e,t)),{props:n,styleId:r??this._styleId()}))}getStyleSheet(e,t,n={},r){return gd.getStyleSheet(Du(Eu({},nd(this,e,t)),{props:n,styleId:r??this._styleId()}))}onStyleMounted(e){this._loadingStyles.add(e)}onStyleUpdated(e){this._loadingStyles.add(e)}onStyleLoaded(e,{name:t}){this._loadingStyles.size&&(this._loadingStyles.delete(t),this.service.emit(`theme:${t}:load`,e),this._loadingStyles.size||this.service.emit(`theme:load`))}}(ku),ad=(e,t,n,r,i,a)=>{if(typeof t!=`string`)return t??e.getTokenValue(n);if(Au.lastIndex=0,!Au.test(t))return t;let o=n.slice(0,n.indexOf(`.`));return Wu(t.replace(Au,t=>{let n=t.slice(1,-1),r=n.indexOf(`.`);if((r===-1?n:n.slice(0,r))!==o)return t;let i=e.getTokenValue(n);return i==null?t:`${i}`}),void 0,r,[i],a)},od=(e,t,n,r,i)=>{let a=((e,t)=>{let n=e.split(`.`),r=``;for(let e=0;e<n.length;e++){let i=Nu(n[e]);t.lastIndex=0,t.test(i)||(r=r?`${r}.${i}`:i)}return r})(t,r),o=e.tokens,s=gd._caches.strict.get(o);s||gd._caches.strict.set(o,s=new Map);let c=typeof i!=`object`||!i,l=c&&i!=null?`${n}|${a}|${i}`:`${n}|${a}`,u=c?s.get(l):void 0;if(!(u!==void 0||c&&s.has(l))){let t=o[a]?.paths,d=t?.find(e=>e.scheme===`none`),f=t?.find(e=>e.scheme===`light`)??d,p=t?.find(e=>e.scheme===`dark`)??d;if(f&&p&&f!==p){let t=ad(e,f.value,a,n,r,i),o=ad(e,p.value,a,n,r,i);u=t===o?t:`light-dark(${t},${o})`}else u=ad(e,(f??p)?.value,a,n,r,i);typeof u!=`string`||Uu(u)||(u=void 0),c&&s.set(l,u)}return e.hasScopedTokenPath(a)?Wu(`{${a}}`,void 0,n,[r],u):u},sd=(e,t,n)=>cd(J,J.getTheme(),e,t,n),cd=(e,t={},n,r,i)=>{var a;if(!n)return``;let o=e.defaults,s=t?.options,c=o?.variable,l=s?.prefix??o?.options?.prefix,u=(a=s?.cssVariables??o?.options?.cssVariables)==null||a;if(i===`value`)return e.getTokenValue(n);if(el(i)&&!u)return od(e,n,l,c.excludedKeyRegex,r);let d=typeof r==`object`&&r||!c.excludedKeyRegex?void 0:dd(c.excludedKeyRegex),f=d&&`${l}\\0${n}\\0${r??``}\\0${r===void 0?0:typeof r==`number`?1:2}`,p=d?.get(f);return p===void 0&&!d?.has(f)&&(p=Wu(gl(n,Au)?n:`{${n}}`,void 0,l,[c.excludedKeyRegex],r),d&&(d.size>=ld&&d.clear(),d.set(f,p))),p},ld=4096,ud=new WeakMap,dd=e=>{let t=ud.get(e);return t||ud.set(e,t=new Map),t},fd=(...e)=>`${sd(...e)??``}`;function pd(e,...t){let n=J.getOptions()?.classPrefix;return e instanceof Array?Yu(Xu(e.reduce((e,n,r)=>e+n+(ul(t[r],{dt:sd})??``),``),fd),n):Yu(ul(e,{dt:sd}),n)}function md(e,t={}){let n=J.defaults.variable,{prefix:r=n.prefix,selector:i=n.selector,excludedKeyRegex:a=n.excludedKeyRegex}=t,o=[],s=[],c=[{node:e,path:r}];for(;c.length;){let{node:e,path:t}=c.pop();for(let n of Object.keys(e)){let i=Pu(e[n]),l=gl(n,a)?Iu(t):Iu(t,Cl(n));if(ol(i))c.push({node:i,path:l});else{let e=Lu(l),t=Wu(i,l,r,[a]);t===void 0&&typeof i==`string`||Gu(s,e,t==null?t:`${t}`);let n=l;r&&n.startsWith(r+`-`)&&(n=n.slice(r.length+1)),o.push(n.replace(/-/g,`.`))}}}let l=s.join(``);return{value:s,tokens:o,declarations:l,css:Ku(i,l)}}var hd=(...e)=>[...new Set(e.flatMap(e=>e??[]))],gd={_caches:{value:new WeakMap,strict:new WeakMap},regex:{rules:{class:{pattern:/^\\.([a-zA-Z][\\w-]*)$/,resolve(e){return{type:`class`,selector:e,matched:this.pattern.test(e.trim())}}},attr:{pattern:/^\\[(.*)\\]$/,resolve(e){return{type:`attr`,selector:`:root${e},${/^(\\[[^\\]]*\\])+$/.test(e.trim())?`:host(${e})`:`:host${e}`}`,matched:this.pattern.test(e.trim())}}},media:{pattern:/^@media (.*)$/,resolve(e){return{type:`media`,selector:e,matched:this.pattern.test(e.trim())}}},system:{pattern:/^system$/,resolve(e){return{type:`system`,selector:`@media (prefers-color-scheme: dark)`,matched:this.pattern.test(e.trim())}}},custom:{resolve:e=>({type:`custom`,selector:e,matched:!0})}},resolve(e){let t=Object.keys(this.rules).filter(e=>e!==`custom`).map(e=>this.rules[e]);return[e].flat().map(e=>t.map(t=>t.resolve(e)).find(e=>e.matched)??this.rules.custom.resolve(e))}},_toVariables:(e,t)=>md(e,{prefix:t?.prefix}),getCommon({name:e=``,theme:t={},params:n,set:r,defaults:i,selector:a,region:o,dt:s}){let{preset:c,options:l}=t,u,d,f,p,m,h,g;if(q(c)){let{primitive:t,semantic:n,extend:_}=c,v=n||{},{colorScheme:y}=v,b=Ou(v,[`colorScheme`]),x=_||{},{colorScheme:S}=x,C=Ou(x,[`colorScheme`]),w=y||{},{dark:ee}=w,te=Ou(w,[`dark`]),ne=S||{},{dark:T}=ne,E=Ou(ne,[`dark`]),D=(e,t)=>{let n=q(t)?this._toVariables({[e]:t},l):{};return[n.declarations??``,n.tokens]},[re,ie]=D(`primitive`,t),[ae,oe]=D(`semantic`,b),[O,se]=D(`light`,te),[ce,le]=D(`dark`,ee),[ue,de]=D(`semantic`,C),[fe,pe]=D(`light`,E),[me,he]=D(`dark`,T);u=this.transformCSS(e,re,`light`,`variable`,l,r,i,a,o),d=ie,f=`${this.transformCSS(e,`${ae}${O}`,`light`,`variable`,l,r,i,a,o)}${this.transformCSS(e,`${ce}`,`dark`,`variable`,l,r,i,a,o)}`,p=hd(oe,se,le),m=`${this.transformCSS(e,`${ue}${fe}color-scheme:light`,`light`,`variable`,l,r,i,a,o)}${this.transformCSS(e,`${me}color-scheme:dark`,`dark`,`variable`,l,r,i,a,o)}`,h=hd(de,pe,he),g=Yu(ul(c.css,{dt:s??sd}),l?.classPrefix)}return{primitive:{css:u,tokens:d},semantic:{css:f,tokens:p},global:{css:m,tokens:h},style:g}},getPreset({name:e=``,preset:t={},options:n,params:r,set:i,defaults:a,selector:o,region:s,isScopedTokenPaths:c,dt:l}){var u;let d,f,p;if(q(t)&&((u=n?.cssVariables)==null||u||c)){let r=e.replace(`-directive`,``),c=t,{colorScheme:u,extend:m,css:h}=c,g=Ou(c,[`colorScheme`,`extend`,`css`]),_=m||{},{colorScheme:v}=_,y=Ou(_,[`colorScheme`]),b=u||{},{dark:x}=b,S=Ou(b,[`dark`]),C=v||{},{dark:w}=C,ee=Ou(C,[`dark`]),te=q(g)?this._toVariables({[r]:Eu(Eu({},g),y)},n):{},ne=q(S)?this._toVariables({[r]:Eu(Eu({},S),ee)},n):{},T=q(x)?this._toVariables({[r]:Eu(Eu({},x),w)},n):{};d=`${this.transformCSS(r,`${te.declarations??``}${ne.declarations??``}`,`light`,`variable`,n,i,a,o,s)}${this.transformCSS(r,T.declarations??``,`dark`,`variable`,n,i,a,o,s)}`,f=hd(te.tokens,ne.tokens,T.tokens),p=Yu(ul(h,{dt:l??sd}),n?.classPrefix)}return{css:d,tokens:f,style:p}},getScopedSelector(e,t){if(t?.scoped&&e)return`[data-styled=\"${e}\"]`},withinRuntime:(e,t)=>e&&t?`${t} ${e}`:e??t,withinRegion(e,t,n={}){let r=e&&!t&&n.within?.call(n,e);return Du(Eu({},n),r?{selector:r,darkSelector:n.within(e,!0)}:{selector:this.withinRuntime(e,t??n.selector),darkSelector:n.darkSelector&&this.withinRuntime(e,n.darkSelector)})},getPresetC({name:e=``,theme:t={},params:n,set:r,defaults:i,selector:a,region:o,dt:s}){let{preset:c,options:l}=t,u=c?.components?.[e],d=this.getScopedSelector(e,l);return this.getPreset({name:e,preset:u,options:l,params:n,set:r,defaults:i,region:this.withinRegion(d,a,o),dt:s})},getPresetD({name:e=``,theme:t={},params:n,set:r,defaults:i,selector:a,region:o,dt:s}){let c=e.replace(`-directive`,``),{preset:l,options:u}=t,d=l?.components?.[c]||l?.directives?.[c],f=this.getScopedSelector(c,u);return this.getPreset({name:c,preset:d,options:u,params:n,set:r,defaults:i,region:this.withinRegion(f,a,o),dt:s})},applyDarkColorScheme(e){let t=e.darkModeSelector;return t!==`none`&&!1!==t},getColorSchemeOption(e,t){return this.applyDarkColorScheme(e)?this.regex.resolve(!0===e.darkModeSelector?t.options.darkModeSelector:e.darkModeSelector??t.options.darkModeSelector):[]},getLayerOrder(e,t={},n,r){let{cssLayer:i}=t;return i?`@layer ${ul(i.order||i.name||`primeui`,n)}`:``},getCommonStyleSheet({name:e=``,theme:t={},params:n,props:r={},set:i,defaults:a,selector:o,region:s,styleId:c=e=>e,dt:l}){let u=this.getCommon({name:e,theme:t,params:n,set:i,defaults:a,selector:o,region:s,dt:l}),d=su(r).join(` `);return Object.entries(u||{}).reduce((e,[t,n])=>{if(ol(n)&&Object.prototype.hasOwnProperty.call(n,`css`)){let r=bl(n.css);e.push(`<style type=\"text/css\" ${this.styleIdAttributes(c(`${t}-variables`))} ${d}>${au(r)}</style>`)}return e},[]).join(``)},getStyleSheet({name:e=``,theme:t={},params:n,props:r={},set:i,defaults:a,selector:o,region:s,styleId:c=e=>e,dt:l}){let u={name:e,theme:t,params:n,set:i,defaults:a,selector:o,region:s,dt:l},d=(e.includes(`-directive`)?this.getPresetD(u):this.getPresetC(u))?.css,f=su(r).join(` `);return d?`<style type=\"text/css\" ${this.styleIdAttributes(c(`${e}-variables`))} ${f}>${au(bl(d))}</style>`:``},styleIdAttributes:e=>`data-primevue-style-id=\"${ou(e)}\"`,createTokens(e={},t,n=``,r=``,i=Object.create(null)){let a=function(e,t,n,r){return e.replace(Au,e=>{let i=e.slice(1,-1),a=this.tokens[i];if(!a)return console.warn(`Token not found for path: ${i}`),`__UNRESOLVED__`;let o=a.computed(t,n,r);if(Array.isArray(o)&&o.length===2){let e=o[0].value,t=o[1].value;return e===t?e??`__UNRESOLVED__`:`light-dark(${e},${t})`}return o?.value??`__UNRESOLVED__`})},o=function(e,t,n,r){if(e.indexOf(`light-dark(`)===-1)return e;let i=[],s=e.length,c=0;for(;c<s;){let l=e.indexOf(`light-dark(`,c);if(l===-1){i.push(e.slice(c));break}i.push(e.slice(c,l));let u=1,d=l+11,f=-1;for(;d<s&&u>0;){let t=e.charCodeAt(d);t===40?u++:t===41?u--:t===44&&u===1&&f===-1&&(f=d),d++}if(u!==0||f===-1){i.push(e.slice(l));break}let p=e.slice(l+11,f).trim(),m=e.slice(f+1,d-1).trim();if(t===`light`||t===`dark`)i.push(o.call(this,t===`light`?p:m,t,n,r));else{let e=a.call(this,o.call(this,p,`light`,n,r),`light`,n,r),t=a.call(this,o.call(this,m,`dark`,n,r),`dark`,n,r);i.push(e===t?e:`light-dark(${e},${t})`)}c=d}return i.join(``)},s=function(e,t={},n=[]){if(n.includes(this.path))return console.warn(`Circular reference detected at ${this.path}`),{colorScheme:e,path:this.path,paths:t,value:void 0};n.push(this.path),t.name=this.path,t.binding||={};let r=this.value;if(typeof this.value==`string`){let i=this.value.trim(),s=i.indexOf(`light-dark(`)!==-1,c=i.indexOf(`{`)!==-1;if(s||c){let c=s?o.call(this,i,e,t,n):i,l=c.indexOf(`{`)===-1?c:a.call(this,c,e,t,n);ju.lastIndex=0,Mu.lastIndex=0,r=ju.test(l.replace(Mu,`0`))?`calc(${l})`:l}}return el(t.binding)&&delete t.binding,n.pop(),{colorScheme:e,path:this.path,paths:t,value:typeof r==`string`&&r.indexOf(`__UNRESOLVED__`)!==-1?void 0:r}},c=(e,n,r)=>{Object.entries(e).forEach(([e,a])=>{let o=gl(e,t.variable.excludedKeyRegex)?n:n?`${n}.${Nu(e)}`:Nu(e),l=r?[r,e].join(`.`):e;if(ol(a))c(a,o,l);else{let e={path:l,value:a,scheme:l.includes(`colorScheme.light`)?`light`:l.includes(`colorScheme.dark`)?`dark`:`none`,computed:s,tokens:i},t=i[o];if(t)t.paths.push(e);else{let t={paths:[e]};t.computed=(e=>(t,n={},r=[])=>{let i=e.paths;if(i.length===1){let e=i[0],a=e.scheme===`none`?t:e.scheme;return e.computed(a,n.binding,r)}if(t&&t!==`none`)for(let e=0;e<i.length;e++){let a=i[e];if(a.scheme===t)return a.computed(t,n.binding,r)}return i.map(e=>e.computed(e.scheme,n[e.scheme],r))})(t),i[o]=t}}})};return c(e,n,r),i},getTokenValue(e,t,n){let r=this._caches.value.get(e);r||this._caches.value.set(e,r=new Map);let i=r.get(t);if(i!==void 0||r.has(t))return i;let a=n.variable.excludedKeyRegex,o=t.split(`.`),s=[];for(let e=0;e<o.length;e++){let t=o[e];a.lastIndex=0,a.test(t.toLowerCase())||s.push(t)}let c=s.join(`.`),l=t.indexOf(`colorScheme.light`)===-1?t.indexOf(`colorScheme.dark`)===-1?void 0:`dark`:`light`,u=e[c];if(!u?.paths)return void r.set(t,void 0);let d=e=>{let t=u.computed(e);return Array.isArray(t)?t.find(t=>t?.colorScheme===e)?.value:t?.value},f;if(l)f=d(l);else{let e=d(`light`),t=d(`dark`);f=e===void 0&&t===void 0?void 0:e===void 0?t:t===void 0||e===t?e:`light-dark(${e},${t})`}return r.set(t,f),f},getSelectorRule(e,t,n,r,i=`:root,:host`,a){if(n===`class`||n===`attr`){if(!t)return Ku(e,r);let i=e.split(/,(?=:host\\()/).flatMap(e=>e.startsWith(`:host`)?[`${e} ${t}`]:[(n===`attr`&&/^:root(\\[[^\\]]*\\])+$/.test(e)?e.slice(5):e)+t,`${e} ${t}`]);return Ku(a?.darkSelector?[...i,a.darkSelector].join(`,`):i.join(`,`),r)}return Ku(e,Ku(t??i,r))},transformCSS(e,t,n,r,i={},a,o,s,c){if(q(t)){let{cssLayer:l}=i;if(r===`style`)t=Yu(t,i.classPrefix),c!=null&&c.scope&&(t=c.scope(t));else{let e=this.getColorSchemeOption(i,o),r=o?.variable?.selector??`:root,:host`,a=s??c?.selector;t=n===`dark`?e.reduce((e,{type:n,selector:i})=>(q(i)&&(e+=i.includes(`[CSS]`)?i.replace(`[CSS]`,()=>t):this.getSelectorRule(i,a,n,t,r,c)),e),``):Ku(a??r,t)}if(l){let n={name:`primeui`,order:`primeui`};ol(l)&&(n.name=ul(l.name,{name:e,type:r})),q(n.name)&&(t=Ku(`@layer ${n.name}`,t),a?.layerNames(n.name))}return t}return``}};function _d(e,...t){return ll(e,...t)}var vd={STARTS_WITH:`startsWith`,CONTAINS:`contains`,NOT_CONTAINS:`notContains`,ENDS_WITH:`endsWith`,EQUALS:`equals`,NOT_EQUALS:`notEquals`,IN:`in`,LESS_THAN:`lt`,LESS_THAN_OR_EQUAL_TO:`lte`,GREATER_THAN:`gt`,GREATER_THAN_OR_EQUAL_TO:`gte`,BETWEEN:`between`,DATE_IS:`dateIs`,DATE_IS_NOT:`dateIsNot`,DATE_BEFORE:`dateBefore`,DATE_AFTER:`dateAfter`},yd={_loadedStyleNames:new Set,getLoadedStyleNames:function(){return this._loadedStyleNames},isStyleNameLoaded:function(e){return this._loadedStyleNames.has(e)},setLoadedStyleName:function(e){this._loadedStyleNames.add(e)},deleteLoadedStyleName:function(e){this._loadedStyleNames.delete(e)},clearLoadedStyleNames:function(){this._loadedStyleNames.clear()}},bd=new WeakMap;function xd(e,t,n){var r,i=bd.get(e);(!e.isStyleNameLoaded(t)||(r=i)!=null&&(r=r.get(t))!=null&&r.some(function(e){var t;return(e==null||(t=e.isLoaded)==null?void 0:t.value)===!1}))&&(i||bd.set(e,i=new Map),i.set(t,[n()].flat()),e.setLoadedStyleName(t))}var Sd=`\n"
    "@layer primeui.base {\n"
    "    :where(:is([data-scope], [data-pc-section])),\n"
    "    :where(:is([data-scope], [data-pc-section]))::before,\n"
    "    :where(:is([data-scope], [data-pc-section]))::after {\n"
    "        box-sizing: border-box;\n"
    "    }\n"
    "\n"
    "    :where(button:is([data-scope], [data-pc-section])),\n"
    "    :where([type='button']:is([data-scope], [data-pc-section])),\n"
    "    :where([type='reset']:is([data-scope], [data-pc-section])),\n"
    "    :where([type='submit']:is([data-scope], [data-pc-section])) {\n"
    "        appearance: button;\n"
    "        background-color: transparent;\n"
    "        background-image: none;\n"
    "        border: 0 solid;\n"
    "        border-radius: 0;\n"
    "        padding: 0;\n"
    "        margin: 0;\n"
    "        font: inherit;\n"
    "        font-feature-settings: inherit;\n"
    "        font-variation-settings: inherit;\n"
    "        letter-spacing: inherit;\n"
    "        color: inherit;\n"
    "        text-align: inherit;\n"
    "        text-transform: none;\n"
    "        cursor: pointer;\n"
    "    }\n"
    "\n"
    "    :where(button:disabled:is([data-scope], [data-pc-section])) {\n"
    "        cursor: default;\n"
    "    }\n"
    "\n"
    "    :where(input:is([data-scope], [data-pc-section])),\n"
    "    :where(select:is([data-scope], [data-pc-section])),\n"
    "    :where(textarea:is([data-scope], [data-pc-section])),\n"
    "    :where(optgroup:is([data-scope], [data-pc-section])) {\n"
    "        appearance: none;\n"
    "        background-color: transparent;\n"
    "        background-image: none;\n"
    "        border: 0 solid;\n"
    "        border-radius: 0;\n"
    "        padding: 0;\n"
    "        margin: 0;\n"
    "        font: inherit;\n"
    "        font-feature-settings: inherit;\n"
    "        font-variation-settings: inherit;\n"
    "        letter-spacing: inherit;\n"
    "        color: inherit;\n"
    "        opacity: 1;\n"
    "    }\n"
    "\n"
    "    :where(textarea:is([data-scope], [data-pc-section])) {\n"
    "        resize: vertical;\n"
    "    }\n"
    "\n"
    "    :where(:is([data-scope], [data-pc-section]))::placeholder {\n"
    "        opacity: 1;\n"
    "        color: color-mix(in oklab, currentcolor 50%, transparent);\n"
    "    }\n"
    "\n"
    "    :where(:is([data-scope], [data-pc-section]))::-webkit-inner-spin-button,\n"
    "    :where(:is([data-scope], [data-pc-section]))::-webkit-outer-spin-button {\n"
    "        height: auto;\n"
    "    }\n"
    "\n"
    "    :where(:is([data-scope], [data-pc-section]))::-webkit-search-decoration {\n"
    "        -webkit-appearance: none;\n"
    "    }\n"
    "\n"
    "    :where(:is([data-scope], [data-pc-section]))::-webkit-datetime-edit {\n"
    "        display: inline-flex;\n"
    "    }\n"
    "\n"
    "    :where(:is([data-scope], [data-pc-section]))::-webkit-datetime-edit-fields-wrapper {\n"
    "        padding: 0;\n"
    "    }\n"
    "\n"
    "    :where(:-moz-ui-invalid:is([data-scope], [data-pc-section])) {\n"
    "        box-shadow: none;\n"
    "    }\n"
    "\n"
    "    :where(ul:is([data-scope], [data-pc-section])),\n"
    "    :where(ol:is([data-scope], [data-pc-section])),\n"
    "    :where(menu:is([data-scope], [data-pc-section])) {\n"
    "        list-style: none;\n"
    "        margin: 0;\n"
    "        padding: 0;\n"
    "    }\n"
    "\n"
    "    :where(table:is([data-scope], [data-pc-section])) {\n"
    "        border-collapse: collapse;\n"
    "        border-color: inherit;\n"
    "        text-indent: 0;\n"
    "    }\n"
    "\n"
    "    :where(caption:is([data-scope], [data-pc-section])),\n"
    "    :where(th:is([data-scope], [data-pc-section])),\n"
    "    :where(td:is([data-scope], [data-pc-section])) {\n"
    "        text-align: inherit;\n"
    "        font-weight: inherit;\n"
    "        padding: 0;\n"
    "    }\n"
    "\n"
    "    :where(h1:is([data-scope], [data-pc-section])),\n"
    "    :where(h2:is([data-scope], [data-pc-section])),\n"
    "    :where(h3:is([data-scope], [data-pc-section])),\n"
    "    :where(h4:is([data-scope], [data-pc-section])),\n"
    "    :where(h5:is([data-scope], [data-pc-section])),\n"
    "    :where(h6:is([data-scope], [data-pc-section])),\n"
    "    :where(p:is([data-scope], [data-pc-section])),\n"
    "    :where(figure:is([data-scope], [data-pc-section])),\n"
    "    :where(blockquote:is([data-scope], [data-pc-section])),\n"
    "    :where(dl:is([data-scope], [data-pc-section])),\n"
    "    :where(dd:is([data-scope], [data-pc-section])) {\n"
    "        font-size: inherit;\n"
    "        font-weight: inherit;\n"
    "        margin: 0;\n"
    "    }\n"
    "\n"
    "    :where(a:is([data-scope], [data-pc-section])) {\n"
    "        color: inherit;\n"
    "        -webkit-text-decoration: inherit;\n"
    "        text-decoration: inherit;\n"
    "    }\n"
    "\n"
    "    :where(fieldset:is([data-scope], [data-pc-section])),\n"
    "    :where(legend:is([data-scope], [data-pc-section])) {\n"
    "        border: 0 solid;\n"
    "        margin: 0;\n"
    "        padding: 0;\n"
    "    }\n"
    "\n"
    "    :where(img:is([data-scope], [data-pc-section])),\n"
    "    :where(svg:is([data-scope], [data-pc-section])),\n"
    "    :where(video:is([data-scope], [data-pc-section])),\n"
    "    :where(canvas:is([data-scope], [data-pc-section])),\n"
    "    :where(iframe:is([data-scope], [data-pc-section])),\n"
    "    :where(embed:is([data-scope], [data-pc-section])),\n"
    "    :where(object:is([data-scope], [data-pc-section])) {\n"
    "        display: block;\n"
    "        vertical-align: middle;\n"
    "    }\n"
    "\n"
    "    :where(img:is([data-scope], [data-pc-section])),\n"
    "    :where(video:is([data-scope], [data-pc-section])) {\n"
    "        max-width: 100%;\n"
    "    }\n"
    "\n"
    "    :where(img:is([data-scope], [data-pc-section]):not([height])),\n"
    "    :where(video:is([data-scope], [data-pc-section]):not([height])) {\n"
    "        height: auto;\n"
    "    }\n"
    "\n"
    "    :where(hr:is([data-scope], [data-pc-section])) {\n"
    "        height: 0;\n"
    "        color: inherit;\n"
    "        border-top-width: 1px;\n"
    "    }\n"
    "\n"
    "    :where(summary:is([data-scope], [data-pc-section])) {\n"
    "        display: list-item;\n"
    "    }\n"
    "\n"
    "    :where(dialog:is([data-scope], [data-pc-section])) {\n"
    "        border: 0 solid;\n"
    "        padding: 0;\n"
    "        margin: 0;\n"
    "        max-width: none;\n"
    "        max-height: none;\n"
    "        background-color: transparent;\n"
    "        color: inherit;\n"
    "    }\n"
    "\n"
    "    :where(:-moz-focusring:is([data-scope], [data-pc-section])) {\n"
    "        outline: auto;\n"
    "    }\n"
    "}\n"
    "`,Cd=`\n"
    "    *,\n"
    "    ::before,\n"
    "    ::after {\n"
    "        box-sizing: border-box;\n"
    "    }\n"
    "\n"
    "    .p-component {\n"
    "        font-family: dt('typography.font.family');\n"
    "        font-feature-settings: inherit;\n"
    "        line-height: dt('typography.line.height');\n"
    "    }\n"
    "\n"
    "    .p-collapsible-enter-active {\n"
    "        animation: p-animate-collapsible-expand 0.2s ease-out;\n"
    "        overflow: hidden;\n"
    "    }\n"
    "\n"
    "    .p-collapsible-leave-active {\n"
    "        animation: p-animate-collapsible-collapse 0.2s ease-out;\n"
    "        overflow: hidden;\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-collapsible-expand {\n"
    "        from {\n"
    "            grid-template-rows: 0fr;\n"
    "        }\n"
    "        to {\n"
    "            grid-template-rows: 1fr;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-collapsible-collapse {\n"
    "        from {\n"
    "            grid-template-rows: 1fr;\n"
    "        }\n"
    "        to {\n"
    "            grid-template-rows: 0fr;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    .p-disabled,\n"
    "    .p-disabled * {\n"
    "        cursor: default;\n"
    "        pointer-events: none;\n"
    "        user-select: none;\n"
    "    }\n"
    "\n"
    "    .p-disabled,\n"
    "    .p-component:disabled {\n"
    "        opacity: dt('disabled.opacity');\n"
    "    }\n"
    "\n"
    "    .pi {\n"
    "        font-size: dt('icon.size');\n"
    "    }\n"
    "\n"
    "    .p-icon {\n"
    "        width: var(--px-icon-size, dt('icon.size'));\n"
    "        height: var(--px-icon-size, dt('icon.size'));\n"
    "        flex-shrink: 0;\n"
    "    }\n"
    "\n"
    "    .p-icon-spin {\n"
    "        -webkit-animation: p-icon-spin 2s infinite linear;\n"
    "        animation: p-icon-spin 2s infinite linear;\n"
    "    }\n"
    "\n"
    "    @-webkit-keyframes p-icon-spin {\n"
    "        0% {\n"
    "            -webkit-transform: rotate(0deg);\n"
    "            transform: rotate(0deg);\n"
    "        }\n"
    "        100% {\n"
    "            -webkit-transform: rotate(359deg);\n"
    "            transform: rotate(359deg);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    @keyframes p-icon-spin {\n"
    "        0% {\n"
    "            -webkit-transform: rotate(0deg);\n"
    "            transform: rotate(0deg);\n"
    "        }\n"
    "        100% {\n"
    "            -webkit-transform: rotate(359deg);\n"
    "            transform: rotate(359deg);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    .p-overlay-mask {\n"
    "        background: var(--px-mask-background, dt('mask.background'));\n"
    "        color: dt('mask.color');\n"
    "        position: fixed;\n"
    "        top: 0;\n"
    "        left: 0;\n"
    "        width: 100%;\n"
    "        height: 100%;\n"
    "    }\n"
    "\n"
    "    .p-overlay-mask-enter-active {\n"
    "        animation: p-animate-overlay-mask-enter dt('mask.transition.duration') forwards;\n"
    "    }\n"
    "\n"
    "    .p-overlay-mask-leave-active {\n"
    "        animation: p-animate-overlay-mask-leave dt('mask.transition.duration') forwards;\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-overlay-mask-enter {\n"
    "        from {\n"
    "            background: transparent;\n"
    "        }\n"
    "        to {\n"
    "            background: var(--px-mask-background, dt('mask.background'));\n"
    "        }\n"
    "    }\n"
    "    @keyframes p-animate-overlay-mask-leave {\n"
    "        from {\n"
    "            background: var(--px-mask-background, dt('mask.background'));\n"
    "        }\n"
    "        to {\n"
    "            background: transparent;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    .p-anchored-overlay-enter-active {\n"
    "        animation: p-animate-anchored-overlay-enter 300ms cubic-bezier(.19,1,.22,1);\n"
    "    }\n"
    "\n"
    "    .p-anchored-overlay-leave-active {\n"
    "        animation: p-animate-anchored-overlay-leave 300ms cubic-bezier(.19,1,.22,1);\n"
    "        pointer-events: none;\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-anchored-overlay-enter {\n"
    "        from {\n"
    "            opacity: 0;\n"
    "            transform: scale(0.93);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-anchored-overlay-leave {\n"
    "        to {\n"
    "            opacity: 0;\n"
    "            transform: scale(0.93);\n"
    "        }\n"
    "    }\n"
    "`;function wd(e){\"@babel/helpers - typeof\";return wd=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},wd(e)}function Td(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function Ed(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?Td(Object(n),!0).forEach(function(t){Dd(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Td(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function Dd(e,t,n){return(t=Od(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Od(e){var t=kd(e,`string`);return wd(t)==`symbol`?t:t+``}function kd(e,t){if(wd(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(wd(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}function Ad(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0,n=Fa();if(n!=null&&n.components&&!n.isMounted){var r=!1,i=function(){return!r&&(r=!0,e())};Dr(i),Ar(i),xr(i)}else t?e():wn(e)}var jd=0,Md=function(){},Nd=`data-primevue-style-id`,Pd=new WeakMap,Fd=!1;function Id(e,t,n){Fd&&e.querySelectorAll(`style[${Nd}=${gu(t)}],style[${Nd}^=${gu(`${t}@`)}]`).forEach(function(e){var t=e.getAttribute(Nd);if(t!==n&&Pd.has(e)){var r;(r=Ld)==null||r.unset(t),e.remove()}})}var Ld;function Rd(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=$t(!1),r=$t(e),i=$t(null),a=mu()?window.document:void 0,o=t.document,s=o===void 0?a:o,c=t.immediate,l=c===void 0||c,u=t.manual,d=u!==void 0&&u,f=t.name,p=f===void 0?`style_${++jd}`:f,m=t.id,h=m===void 0?void 0:m,g=t.media,_=g===void 0?void 0:g,v=t.nonce,y=v===void 0?void 0:v,b=t.first,x=b!==void 0&&b,S=t.onMounted,C=S===void 0?void 0:S,w=t.onUpdated,ee=w===void 0?void 0:w,te=t.onLoad,ne=te===void 0?void 0:te,T=t.props,E=T===void 0?{}:T,D=t.isolation,re=D===void 0?void 0:D,ie=Md,ae,oe=function(e,t){var n=ie=Gn(e,t,{immediate:!0});Ae()&&je(function(){return ie===n&&(ie=Md)})},O=function(t){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};if(s){var o=Ed(Ed({},E),a),c=o.name||p,l=o.id||h,u=o.nonce||y;if(re){if(n.value)return;r.value=t||r.value,ae=c,oe(r,function(e){re.scope.set(c,e,{first:x,media:_,nonce:u}),ee?.(c)}),n.value=!0}else{var d=J.styleName?.call(J,c)??c;Fd||=d!==c,Id(s,c,d),i.value=s.querySelector(`style[data-primevue-style-id=${gu(d)}]`)||s.getElementById(l)||s.createElement(`style`),Pd.set(i.value,x),i.value.isConnected||(r.value=t||e,ru(i.value,{type:`text/css`,id:l,media:_,nonce:u}),x?s.head.prepend(i.value):s.head.appendChild(i.value),hu(i.value,Nd,d),ru(i.value,o),i.value.onload=function(e){return ne?.(e,{name:c})},C?.(c)),n.value||=(ae=d,oe(r,function(e){var t;i.value.textContent!==e&&(i.value.textContent=e),(t=Ld)==null||t.set(i.value),ee?.(c)}),!0)}}};return l&&!d&&(re?O():Ad(O)),{id:h,name:p,el:i,css:r,unload:function(){var e;s&&n.value&&(ie(),re?(re.scope.delete(ae),n.value=!1):((e=Ld)==null||e.unset(ae),$l(i.value)&&s.head.removeChild(i.value),n.value=!1,i.value=null))},load:O,isLoaded:Ut(n)}}function zd(e){\"@babel/helpers - typeof\";return zd=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},zd(e)}var Bd,Vd,Hd,Ud;function Wd(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function Gd(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?Wd(Object(n),!0).forEach(function(t){Kd(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):Wd(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function Kd(e,t,n){return(t=qd(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function qd(e){var t=Jd(e,`string`);return zd(t)==`symbol`?t:t+``}function Jd(e,t){if(zd(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(zd(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}function Yd(e,t){return t||=e.slice(0),Object.freeze(Object.defineProperties(e,{raw:{value:Object.freeze(t)}}))}var Xd=function(e){var t=e.dt;return`\n"
    ".p-hidden-accessible {\n"
    "    border: 0;\n"
    "    clip: rect(0 0 0 0);\n"
    "    height: 1px;\n"
    "    margin: -1px;\n"
    "    opacity: 0;\n"
    "    overflow: hidden;\n"
    "    padding: 0;\n"
    "    pointer-events: none;\n"
    "    position: absolute;\n"
    "    white-space: nowrap;\n"
    "    width: 1px;\n"
    "}\n"
    "\n"
    ".p-overflow-hidden {\n"
    "    overflow: hidden;\n"
    "    padding-right: ${t(`scrollbar.width`)};\n"
    "}\n"
    "`},Zd={},Qd={},$d=function(e){return`${Nd}=\"${ou(J.styleName?.call(J,e)??e)}\"`},Y={name:`base`,css:Xd,style:Cd,classes:Zd,inlineStyles:Qd,load:function(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=(arguments.length>2&&arguments[2]!==void 0?arguments[2]:function(e){return e})(t.raw?e:(t.isolation?.css||pd)(Bd||=Yd([``,``]),e));return q(n)?Rd(bl(n),Gd({name:this.name},t)):{}},loadCSS:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=e.isolation?.theme;return t?this.load(this.css,e,function(e){return t.scopeCSS(e)}):this.load(this.css,e)},loadPreflight:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=e.isolation?.theme;return this.load(Sd,Gd(Gd({},e),{},{name:`preflight-style`,raw:!0}),function(e){return t?t.scopeCSS(e):e})},loadStyle:function(){var e=this,t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,r=t.isolation;return this.load(this.style,t,function(){var i=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``;return(r?.theme||J).transformCSS(t.name||e.name,`${i}${(r?.css||pd)(Vd||=Yd([``,``]),n)}`)})},getCommonTheme:function(e){return(arguments.length>1&&arguments[1]!==void 0?arguments[1]:J).getCommon(this.name,e)},getComponentTheme:function(e){return(arguments.length>1&&arguments[1]!==void 0?arguments[1]:J).getComponent(this.name,e)},getDirectiveTheme:function(e){return(arguments.length>1&&arguments[1]!==void 0?arguments[1]:J).getDirective(this.name,e)},getPresetTheme:function(e,t,n){return(arguments.length>3&&arguments[3]!==void 0?arguments[3]:J).getCustomPreset(this.name,e,t,n)},getLayerOrderThemeCSS:function(){return(arguments.length>0&&arguments[0]!==void 0?arguments[0]:J).getLayerOrderCSS(this.name)},getStyleSheet:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};if(this.css){var n=ul(this.css,{dt:sd})||``,r=bl(pd(Hd||=Yd([``,``,``]),n,e)),i=su(t).join(` `);return q(r)?`<style type=\"text/css\" ${$d(this.name)} ${i}>${au(r)}</style>`:``}return``},getLayerOrderStyleSheet:function(){var e=arguments.length>0&&arguments[0]!==void 0&&arguments[0],t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=ed(e?$u:void 0,J.getLayerOrderCSS(this.name)),r=su(t).join(` `);return q(n)?`<style type=\"text/css\" ${$d(`layer-order`)} ${r}>${au(n)}</style>`:``},getPreflightStyleSheet:function(){var e=su(arguments.length>0&&arguments[0]!==void 0?arguments[0]:{}).join(` `);return`<style type=\"text/css\" ${$d(`preflight-style`)} ${e}>${au(bl(Sd))}</style>`},getCommonThemeStyleSheet:function(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};return J.getCommonStyleSheet(this.name,e,t,function(e){return J.styleName?.call(J,e)??e})},getThemeStyleSheet:function(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=[J.getStyleSheet(this.name,e,t,function(e){return J.styleName?.call(J,e)??e})];if(this.style){var r=this.name===`base`?`global-style`:`${this.name}-style`,i=pd(Ud||=Yd([``,``]),ul(this.style,{dt:sd})),a=bl(J.transformCSS(r,i)),o=su(t).join(` `);q(a)&&n.push(`<style type=\"text/css\" ${$d(r)} ${o}>${au(a)}</style>`)}return n.join(``)},extend:function(e){return Gd(Gd({},this),{},{css:void 0,style:void 0},e)}};function ef(){if(typeof document<`u`&&!document.getElementById(`p-license-host`)){var e=document.createElement(`div`);e.id=`p-license-host`,e.style.cssText=`all:initial;position:fixed;bottom:16px;right:16px;z-index:2147483647;pointer-events:none;`;var t=e.attachShadow({mode:`closed`});t.innerHTML=`<div role=\"alert\" style=\"padding:10px 14px;background:#991b1b;color:#fff;font:600 13px/1.2 system-ui,-apple-system,sans-serif;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.2);\">Invalid PrimeUI License</div>`,document.body.appendChild(e)}}var tf=qc();function nf(e){\"@babel/helpers - typeof\";return nf=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},nf(e)}function rf(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function af(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?rf(Object(n),!0).forEach(function(t){of(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):rf(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function of(e,t,n){return(t=sf(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function sf(e){var t=cf(e,`string`);return nf(t)==`symbol`?t:t+``}function cf(e,t){if(nf(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(nf(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var lf=`2026-11-12`,uf={ripple:!1,inputVariant:null,license:null,locale:{startsWith:`Starts with`,contains:`Contains`,notContains:`Not contains`,endsWith:`Ends with`,equals:`Equals`,notEquals:`Not equals`,noFilter:`No Filter`,lt:`Less than`,lte:`Less than or equal to`,gt:`Greater than`,gte:`Greater than or equal to`,dateIs:`Date is`,dateIsNot:`Date is not`,dateBefore:`Date is before`,dateAfter:`Date is after`,clear:`Clear`,apply:`Apply`,matchAll:`Match All`,matchAny:`Match Any`,addRule:`Add Rule`,removeRule:`Remove Rule`,accept:`Yes`,reject:`No`,choose:`Choose`,upload:`Upload`,cancel:`Cancel`,completed:`Completed`,pending:`Pending`,fileSizeTypes:[`B`,`KB`,`MB`,`GB`,`TB`,`PB`,`EB`,`ZB`,`YB`],dayNames:[`Sunday`,`Monday`,`Tuesday`,`Wednesday`,`Thursday`,`Friday`,`Saturday`],dayNamesShort:[`Sun`,`Mon`,`Tue`,`Wed`,`Thu`,`Fri`,`Sat`],dayNamesMin:[`Su`,`Mo`,`Tu`,`We`,`Th`,`Fr`,`Sa`],monthNames:[`January`,`February`,`March`,`April`,`May`,`June`,`July`,`August`,`September`,`October`,`November`,`December`],monthNamesShort:[`Jan`,`Feb`,`Mar`,`Apr`,`May`,`Jun`,`Jul`,`Aug`,`Sep`,`Oct`,`Nov`,`Dec`],chooseYear:`Choose Year`,chooseMonth:`Choose Month`,chooseDate:`Choose Date`,prevDecade:`Previous Decade`,nextDecade:`Next Decade`,prevYear:`Previous Year`,nextYear:`Next Year`,prevMonth:`Previous Month`,nextMonth:`Next Month`,prevHour:`Previous Hour`,nextHour:`Next Hour`,prevMinute:`Previous Minute`,nextMinute:`Next Minute`,prevSecond:`Previous Second`,nextSecond:`Next Second`,am:`am`,pm:`pm`,today:`Today`,weekHeader:`Wk`,firstDayOfWeek:0,showMonthAfterYear:!1,dateFormat:`mm/dd/yy`,weak:`Weak`,medium:`Medium`,strong:`Strong`,passwordPrompt:`Enter a password`,emptyFilterMessage:`No results found`,searchMessage:`{0} results are available`,selectionMessage:`{0} items selected`,emptySelectionMessage:`No selected item`,emptySearchMessage:`No results found`,fileChosenMessage:`{0} files`,noFileChosenMessage:`No file chosen`,emptyMessage:`No available options`,aria:{trueLabel:`True`,falseLabel:`False`,nullLabel:`Not Selected`,star:`1 star`,stars:`{star} stars`,selectAll:`All items selected`,unselectAll:`All items unselected`,close:`Close`,previous:`Previous`,next:`Next`,navigation:`Navigation`,scrollTop:`Scroll Top`,moveTop:`Move Top`,moveUp:`Move Up`,moveDown:`Move Down`,moveBottom:`Move Bottom`,moveToTarget:`Move to Target`,moveToSource:`Move to Source`,moveAllToTarget:`Move All to Target`,moveAllToSource:`Move All to Source`,pageLabel:`Page {page}`,firstPageLabel:`First Page`,lastPageLabel:`Last Page`,nextPageLabel:`Next Page`,prevPageLabel:`Previous Page`,rowsPerPageLabel:`Rows per page`,jumpToPageDropdownLabel:`Jump to Page Dropdown`,jumpToPageInputLabel:`Jump to Page Input`,selectRow:`Row Selected`,unselectRow:`Row Unselected`,expandRow:`Row Expanded`,collapseRow:`Row Collapsed`,showFilterMenu:`Show Filter Menu`,hideFilterMenu:`Hide Filter Menu`,filter:`Filter`,filterOperator:`Filter Operator`,filterConstraint:`Filter Constraint`,editRow:`Row Edit`,saveEdit:`Save Edit`,cancelEdit:`Cancel Edit`,listView:`List View`,gridView:`Grid View`,slide:`Slide`,slideNumber:`{slideNumber}`,zoomImage:`Zoom Image`,zoomIn:`Zoom In`,zoomOut:`Zoom Out`,rotateRight:`Rotate Right`,rotateLeft:`Rotate Left`,listLabel:`Option List`}},filterMatchModeOptions:{text:[vd.STARTS_WITH,vd.CONTAINS,vd.NOT_CONTAINS,vd.ENDS_WITH,vd.EQUALS,vd.NOT_EQUALS],numeric:[vd.EQUALS,vd.NOT_EQUALS,vd.LESS_THAN,vd.LESS_THAN_OR_EQUAL_TO,vd.GREATER_THAN,vd.GREATER_THAN_OR_EQUAL_TO],date:[vd.DATE_IS,vd.DATE_IS_NOT,vd.DATE_BEFORE,vd.DATE_AFTER]},zIndex:{modal:1100,overlay:1e3,menu:1e3,tooltip:1100},theme:void 0,unstyled:!1,preflight:!1,pt:void 0,ptOptions:{mergeSections:!0,mergeProps:!1},csp:{nonce:void 0}},df=Symbol();function ff(){var e=Vn(df);if(!e)throw Error(`PrimeVue is not installed!`);return e}function pf(e,t){var n=$t(null);t.license&&Gc({primeui:t.license}),Kc(`primeui`,{releaseDate:lf}).then(function(e){n.value=e.valid,e.valid||(console.warn(`[PrimeUI] ${e.message}`),ef())});var r={config:Vt(t),verified:Ut(n)};if(t.isolation){var i;if(!((i=t.isolation)!=null&&i[Symbol.for(`primevue.isolation`)]))throw TypeError(\"[PrimeVue] `isolation` takes the value of `withIsolation()` from `@primevue/core/isolation`.\");r.isolation=t.isolation(e,r)}if(t.shadowRoot&&!t.shadowRoot[Symbol.for(`primevue.shadowroot`)])throw TypeError(\"[PrimeVue] `shadowRoot` takes the value of `withShadowRoot()` from `@primevue/core/shadowroot`.\");return r.shadowRoot=r.isolation?void 0:t.shadowRoot?.call(t),e.config.globalProperties.$primevue=r,e.provide(df,r),r.isolation||hf(),gf(e,r),r}var mf=[];function hf(){ku.clear(),mf.forEach(function(e){return e?.()}),mf=[]}function gf(e,t){var n=$t(!1),r=t.isolation,i=r?.theme||J,a=i.service??ku,o=r?.events||tf,s=function(){if(t.config?.theme!==`none`&&!i.isStyleNameLoaded(`common`)){var e,n=Y.getCommonTheme?.call(Y,void 0,i)||{},a=n.primitive,o=n.semantic,s=n.global,c=n.style,l={nonce:(e=t.config)==null||(e=e.csp)==null?void 0:e.nonce,isolation:r};Y.load(a?.css,af({name:`primitive-variables`,raw:!0},l)),Y.load(o?.css,af({name:`semantic-variables`,raw:!0},l)),Y.load(s?.css,af({name:`global-variables`,raw:!0},l)),Y.loadStyle(af({name:`global-style`},l),c),i.setLoadedStyleName(`common`)}};a.on(`theme:change`,function(){return(r?.base||yd).clearLoadedStyleNames()}),a.on(`theme:change`,function(t){!n.value&&!(r!=null&&r.settling)&&(e.config.globalProperties.$primevue.config.theme=t,n.value=!0)});var c=Gn(t.config,function(e,t){o.emit(`config:change`,{newValue:e,oldValue:t})},{immediate:!0,deep:!0}),l=Gn(function(){return t.config.ripple},function(e,t){o.emit(`config:ripple:change`,{newValue:e,oldValue:t})},{immediate:!0,deep:!0}),u=Gn(function(){return t.config.theme},function(e,r){n.value||i.setTheme(e),t.config.unstyled||s(),n.value=!1,o.emit(`config:theme:change`,{newValue:e,oldValue:r})},{immediate:!0,deep:!1}),d=Gn(function(){return t.config.unstyled},function(e,n){!e&&t.config.theme&&s(),o.emit(`config:unstyled:change`,{newValue:e,oldValue:n})},{immediate:!0,deep:!0});r&&(r.reload=function(){return t.config.unstyled||s()});var f=r?r.stops:mf;f.push(c),f.push(l),f.push(u),f.push(d)}var _f={install:function(e,t){pf(e,_l(uf,t))}},vf=(e,...t)=>_d(e,...t),yf={root:{transitionDuration:`{transition.duration}`},panel:{borderWidth:`0 0 1px 0`,borderColor:`{content.border.color}`},header:{color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{text.color}`,activeHoverColor:`{text.color}`,padding:`1.125rem`,fontWeight:`600`,borderRadius:`0`,borderWidth:`0`,borderColor:`{content.border.color}`,background:`{content.background}`,hoverBackground:`{content.background}`,activeBackground:`{content.background}`,activeHoverBackground:`{content.background}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},toggleIcon:{color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{text.color}`,activeHoverColor:`{text.color}`},first:{topBorderRadius:`{content.border.radius}`,borderWidth:`0`},last:{bottomBorderRadius:`{content.border.radius}`,activeBottomBorderRadius:`0`},fontSize:`{typography.font.size}`},content:{borderWidth:`0`,borderColor:`{content.border.color}`,background:`{content.background}`,color:`{text.color}`,padding:`0 1.125rem 1.125rem 1.125rem`}},bf={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},list:{padding:`{list.padding}`,gap:`{list.gap}`},option:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`,fontWeight:`{list.option.font.weight}`,fontSize:`{list.option.font.size}`},optionGroup:{background:`{list.option.group.background}`,color:`{list.option.group.color}`,fontWeight:`{list.option.group.font.weight}`,padding:`{list.option.group.padding}`,fontSize:`{list.option.group.font.size}`},dropdown:{width:`2.5rem`,sm:{width:`2rem`},lg:{width:`3rem`},borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.border.color}`,activeBorderColor:`{form.field.border.color}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},background:`light-dark({surface.100}, {surface.800})`,hoverBackground:`light-dark({surface.200}, {surface.700})`,activeBackground:`light-dark({surface.300}, {surface.600})`,color:`light-dark({surface.600}, {surface.300})`,hoverColor:`light-dark({surface.700}, {surface.200})`,activeColor:`light-dark({surface.800}, {surface.100})`},chip:{borderRadius:`{border.radius.sm}`,focusBackground:`light-dark({surface.200}, {surface.700})`,focusColor:`light-dark({surface.800}, {surface.0})`},emptyMessage:{padding:`{list.option.padding}`}},xf={root:{width:`2rem`,height:`2rem`,fontSize:`1rem`,background:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,fontWeight:`{typography.font.weight}`},icon:{size:`1rem`},group:{borderColor:`{content.background}`,offset:`-0.75rem`},lg:{width:`3rem`,height:`3rem`,fontSize:`1.5rem`,icon:{size:`1.5rem`},group:{offset:`-1rem`}},xl:{width:`4rem`,height:`4rem`,fontSize:`2rem`,icon:{size:`2rem`},group:{offset:`-1.5rem`}}},Sf={root:{borderRadius:`{border.radius.md}`,padding:`0 0.5rem`,fontSize:`0.75rem`,fontWeight:`700`,minWidth:`1.5rem`,height:`1.5rem`},dot:{size:`0.5rem`},sm:{fontSize:`0.625rem`,minWidth:`1.25rem`,height:`1.25rem`},lg:{fontSize:`0.875rem`,minWidth:`1.75rem`,height:`1.75rem`},xl:{fontSize:`1rem`,minWidth:`2rem`,height:`2rem`},primary:{background:`{primary.color}`,color:`{primary.contrast.color}`},secondary:{background:`light-dark({surface.100}, {surface.800})`,color:`light-dark({surface.600}, {surface.300})`},success:{background:`light-dark({green.500}, {green.400})`,color:`light-dark({surface.0}, {green.950})`},info:{background:`light-dark({sky.500}, {sky.400})`,color:`light-dark({surface.0}, {sky.950})`},warn:{background:`light-dark({orange.500}, {orange.400})`,color:`light-dark({surface.0}, {orange.950})`},danger:{background:`light-dark({red.500}, {red.400})`,color:`light-dark({surface.0}, {red.950})`},contrast:{background:`light-dark({surface.950}, {surface.0})`,color:`light-dark({surface.0}, {surface.950})`}},Cf={primitive:{borderRadius:{none:`0`,xs:`2px`,sm:`4px`,md:`6px`,lg:`8px`,xl:`12px`},emerald:{50:`#ecfdf5`,100:`#d1fae5`,200:`#a7f3d0`,300:`#6ee7b7`,400:`#34d399`,500:`#10b981`,600:`#059669`,700:`#047857`,800:`#065f46`,900:`#064e3b`,950:`#022c22`},green:{50:`#f0fdf4`,100:`#dcfce7`,200:`#bbf7d0`,300:`#86efac`,400:`#4ade80`,500:`#22c55e`,600:`#16a34a`,700:`#15803d`,800:`#166534`,900:`#14532d`,950:`#052e16`},lime:{50:`#f7fee7`,100:`#ecfccb`,200:`#d9f99d`,300:`#bef264`,400:`#a3e635`,500:`#84cc16`,600:`#65a30d`,700:`#4d7c0f`,800:`#3f6212`,900:`#365314`,950:`#1a2e05`},red:{50:`#fef2f2`,100:`#fee2e2`,200:`#fecaca`,300:`#fca5a5`,400:`#f87171`,500:`#ef4444`,600:`#dc2626`,700:`#b91c1c`,800:`#991b1b`,900:`#7f1d1d`,950:`#450a0a`},orange:{50:`#fff7ed`,100:`#ffedd5`,200:`#fed7aa`,300:`#fdba74`,400:`#fb923c`,500:`#f97316`,600:`#ea580c`,700:`#c2410c`,800:`#9a3412`,900:`#7c2d12`,950:`#431407`},amber:{50:`#fffbeb`,100:`#fef3c7`,200:`#fde68a`,300:`#fcd34d`,400:`#fbbf24`,500:`#f59e0b`,600:`#d97706`,700:`#b45309`,800:`#92400e`,900:`#78350f`,950:`#451a03`},yellow:{50:`#fefce8`,100:`#fef9c3`,200:`#fef08a`,300:`#fde047`,400:`#facc15`,500:`#eab308`,600:`#ca8a04`,700:`#a16207`,800:`#854d0e`,900:`#713f12`,950:`#422006`},teal:{50:`#f0fdfa`,100:`#ccfbf1`,200:`#99f6e4`,300:`#5eead4`,400:`#2dd4bf`,500:`#14b8a6`,600:`#0d9488`,700:`#0f766e`,800:`#115e59`,900:`#134e4a`,950:`#042f2e`},cyan:{50:`#ecfeff`,100:`#cffafe`,200:`#a5f3fc`,300:`#67e8f9`,400:`#22d3ee`,500:`#06b6d4`,600:`#0891b2`,700:`#0e7490`,800:`#155e75`,900:`#164e63`,950:`#083344`},sky:{50:`#f0f9ff`,100:`#e0f2fe`,200:`#bae6fd`,300:`#7dd3fc`,400:`#38bdf8`,500:`#0ea5e9`,600:`#0284c7`,700:`#0369a1`,800:`#075985`,900:`#0c4a6e`,950:`#082f49`},blue:{50:`#eff6ff`,100:`#dbeafe`,200:`#bfdbfe`,300:`#93c5fd`,400:`#60a5fa`,500:`#3b82f6`,600:`#2563eb`,700:`#1d4ed8`,800:`#1e40af`,900:`#1e3a8a`,950:`#172554`},indigo:{50:`#eef2ff`,100:`#e0e7ff`,200:`#c7d2fe`,300:`#a5b4fc`,400:`#818cf8`,500:`#6366f1`,600:`#4f46e5`,700:`#4338ca`,800:`#3730a3`,900:`#312e81`,950:`#1e1b4b`},violet:{50:`#f5f3ff`,100:`#ede9fe`,200:`#ddd6fe`,300:`#c4b5fd`,400:`#a78bfa`,500:`#8b5cf6`,600:`#7c3aed`,700:`#6d28d9`,800:`#5b21b6`,900:`#4c1d95`,950:`#2e1065`},purple:{50:`#faf5ff`,100:`#f3e8ff`,200:`#e9d5ff`,300:`#d8b4fe`,400:`#c084fc`,500:`#a855f7`,600:`#9333ea`,700:`#7e22ce`,800:`#6b21a8`,900:`#581c87`,950:`#3b0764`},fuchsia:{50:`#fdf4ff`,100:`#fae8ff`,200:`#f5d0fe`,300:`#f0abfc`,400:`#e879f9`,500:`#d946ef`,600:`#c026d3`,700:`#a21caf`,800:`#86198f`,900:`#701a75`,950:`#4a044e`},pink:{50:`#fdf2f8`,100:`#fce7f3`,200:`#fbcfe8`,300:`#f9a8d4`,400:`#f472b6`,500:`#ec4899`,600:`#db2777`,700:`#be185d`,800:`#9d174d`,900:`#831843`,950:`#500724`},rose:{50:`#fff1f2`,100:`#ffe4e6`,200:`#fecdd3`,300:`#fda4af`,400:`#fb7185`,500:`#f43f5e`,600:`#e11d48`,700:`#be123c`,800:`#9f1239`,900:`#881337`,950:`#4c0519`},slate:{50:`#f8fafc`,100:`#f1f5f9`,200:`#e2e8f0`,300:`#cbd5e1`,400:`#94a3b8`,500:`#64748b`,600:`#475569`,700:`#334155`,800:`#1e293b`,900:`#0f172a`,950:`#020617`},gray:{50:`#f9fafb`,100:`#f3f4f6`,200:`#e5e7eb`,300:`#d1d5db`,400:`#9ca3af`,500:`#6b7280`,600:`#4b5563`,700:`#374151`,800:`#1f2937`,900:`#111827`,950:`#030712`},zinc:{50:`#fafafa`,100:`#f4f4f5`,200:`#e4e4e7`,300:`#d4d4d8`,400:`#a1a1aa`,500:`#71717a`,600:`#52525b`,700:`#3f3f46`,800:`#27272a`,900:`#18181b`,950:`#09090b`},neutral:{50:`#fafafa`,100:`#f5f5f5`,200:`#e5e5e5`,300:`#d4d4d4`,400:`#a3a3a3`,500:`#737373`,600:`#525252`,700:`#404040`,800:`#262626`,900:`#171717`,950:`#0a0a0a`},stone:{50:`#fafaf9`,100:`#f5f5f4`,200:`#e7e5e4`,300:`#d6d3d1`,400:`#a8a29e`,500:`#78716c`,600:`#57534e`,700:`#44403c`,800:`#292524`,900:`#1c1917`,950:`#0c0a09`}},semantic:{transitionDuration:`0.2s`,focusRing:{width:`1px`,style:`solid`,color:`{primary.color}`,offset:`2px`,shadow:`none`},disabledOpacity:`0.6`,iconSize:`1rem`,anchorGutter:`2px`,primary:{50:`{emerald.50}`,100:`{emerald.100}`,200:`{emerald.200}`,300:`{emerald.300}`,400:`{emerald.400}`,500:`{emerald.500}`,600:`{emerald.600}`,700:`{emerald.700}`,800:`{emerald.800}`,900:`{emerald.900}`,950:`{emerald.950}`,color:`light-dark({primary.500}, {primary.400})`,contrastColor:`light-dark(#ffffff, {surface.900})`,hoverColor:`light-dark({primary.600}, {primary.300})`,activeColor:`light-dark({primary.700}, {primary.200})`},formField:{paddingX:`0.75rem`,paddingY:`0.5rem`,sm:{fontSize:`0.875rem`,paddingX:`0.625rem`,paddingY:`0.375rem`},lg:{fontSize:`1.125rem`,paddingX:`0.875rem`,paddingY:`0.625rem`},borderRadius:`{border.radius.md}`,focusRing:{width:`0`,style:`none`,color:`transparent`,offset:`0`,shadow:`none`},transitionDuration:`{transition.duration}`,background:`light-dark({surface.0}, {surface.950})`,disabledBackground:`light-dark({surface.200}, {surface.700})`,filledBackground:`light-dark({surface.50}, {surface.800})`,filledHoverBackground:`light-dark({surface.50}, {surface.800})`,filledFocusBackground:`light-dark({surface.50}, {surface.800})`,borderColor:`light-dark({surface.300}, {surface.600})`,hoverBorderColor:`light-dark({surface.400}, {surface.500})`,focusBorderColor:`{primary.color}`,invalidBorderColor:`light-dark({red.400}, {red.300})`,color:`light-dark({surface.700}, {surface.0})`,disabledColor:`light-dark({surface.500}, {surface.400})`,placeholderColor:`light-dark({surface.500}, {surface.400})`,invalidPlaceholderColor:`light-dark({red.600}, {red.400})`,floatLabelColor:`light-dark({surface.500}, {surface.400})`,floatLabelFocusColor:`light-dark({primary.600}, {primary.color})`,floatLabelActiveColor:`light-dark({surface.500}, {surface.400})`,floatLabelInvalidColor:`{form.field.invalid.placeholder.color}`,iconColor:`{surface.400}`,shadow:`0 0 #0000, 0 0 #0000, 0 1px 2px 0 rgba(18, 18, 23, 0.05)`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},list:{padding:`0.25rem 0.25rem`,gap:`2px`,header:{padding:`0.5rem 1rem 0.25rem 1rem`},option:{padding:`0.5rem 0.75rem`,borderRadius:`{border.radius.sm}`,focusBackground:`light-dark({surface.100}, {surface.800})`,selectedBackground:`{highlight.background}`,selectedFocusBackground:`{highlight.focus.background}`,color:`{text.color}`,focusColor:`{text.hover.color}`,selectedColor:`{highlight.color}`,selectedFocusColor:`{highlight.focus.color}`,icon:{color:`light-dark({surface.400}, {surface.500})`,focusColor:`light-dark({surface.500}, {surface.400})`},fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`,transitionDuration:`0s`,selectedFontWeight:`{typography.font.weight}`},optionGroup:{padding:`0.5rem 0.75rem`,fontWeight:`600`,background:`transparent`,color:`{text.muted.color}`,fontSize:`{typography.font.size}`}},content:{borderRadius:`{border.radius.md}`,background:`light-dark({surface.0}, {surface.900})`,hoverBackground:`light-dark({surface.100}, {surface.800})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`{text.color}`,hoverColor:`{text.hover.color}`},mask:{transitionDuration:`0.3s`,background:`light-dark(rgba(0,0,0,0.4), rgba(0,0,0,0.6))`,color:`{surface.200}`},navigation:{list:{padding:`0.25rem 0.25rem`,gap:`2px`},item:{padding:`0.5rem 0.75rem`,borderRadius:`{border.radius.sm}`,gap:`0.5rem`,focusBackground:`light-dark({surface.100}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.800})`,color:`{text.color}`,focusColor:`{text.hover.color}`,activeColor:`{text.hover.color}`,icon:{color:`light-dark({surface.400}, {surface.500})`,focusColor:`light-dark({surface.500}, {surface.400})`,activeColor:`light-dark({surface.500}, {surface.400})`,size:`{icon.size}`},label:{fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},transitionDuration:`0s`},submenuLabel:{padding:`0.5rem 0.75rem`,fontWeight:`600`,background:`transparent`,color:`{text.muted.color}`,fontSize:`{typography.font.size}`},submenuIcon:{size:`0.875rem`,color:`light-dark({surface.400}, {surface.500})`,focusColor:`light-dark({surface.500}, {surface.400})`,activeColor:`light-dark({surface.500}, {surface.400})`}},overlay:{select:{borderRadius:`{border.radius.md}`,shadow:`0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)`,background:`light-dark({surface.0}, {surface.900})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`{text.color}`},popover:{borderRadius:`{border.radius.md}`,padding:`0.75rem`,shadow:`0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)`,background:`light-dark({surface.0}, {surface.900})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`{text.color}`},modal:{borderRadius:`{border.radius.xl}`,padding:`1.25rem`,shadow:`0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)`,background:`light-dark({surface.0}, {surface.900})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`{text.color}`},navigation:{shadow:`0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)`}},surface:{0:`#ffffff`,50:`light-dark({slate.50}, {zinc.50})`,100:`light-dark({slate.100}, {zinc.100})`,200:`light-dark({slate.200}, {zinc.200})`,300:`light-dark({slate.300}, {zinc.300})`,400:`light-dark({slate.400}, {zinc.400})`,500:`light-dark({slate.500}, {zinc.500})`,600:`light-dark({slate.600}, {zinc.600})`,700:`light-dark({slate.700}, {zinc.700})`,800:`light-dark({slate.800}, {zinc.800})`,900:`light-dark({slate.900}, {zinc.900})`,950:`light-dark({slate.950}, {zinc.950})`},highlight:{background:`light-dark({primary.50}, color-mix(in srgb, {primary.400}, transparent 84%))`,focusBackground:`light-dark({primary.100}, color-mix(in srgb, {primary.400}, transparent 76%))`,color:`light-dark({primary.700}, rgba(255,255,255,.87))`,focusColor:`light-dark({primary.800}, rgba(255,255,255,.87))`},text:{color:`light-dark({surface.700}, {surface.0})`,hoverColor:`light-dark({surface.800}, {surface.0})`,mutedColor:`light-dark({surface.500}, {surface.400})`,hoverMutedColor:`light-dark({surface.600}, {surface.300})`},typography:{lineHeight:`normal`,fontFamily:`inherit`,fontWeight:`normal`,fontSize:`1rem`}}},wf={root:{borderRadius:`{content.border.radius}`}},Tf={root:{padding:`1rem`,background:`{content.background}`,gap:`0.5rem`,transitionDuration:`{transition.duration}`},item:{color:`{text.muted.color}`,hoverColor:`{text.color}`,borderRadius:`{content.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,hoverColor:`{navigation.item.icon.focus.color}`,size:`{navigation.item.icon.size}`},focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},separator:{color:`{navigation.item.icon.color}`}},Ef={root:{borderRadius:`{form.field.border.radius}`,roundedBorderRadius:`2rem`,gap:`0.5rem`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,iconOnlyWidth:`2.5rem`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`,iconOnlyWidth:`2rem`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`,iconOnlyWidth:`3rem`},label:{fontWeight:`500`},raisedShadow:`0 3px 1px -2px rgba(0, 0, 0, 0.2), 0 2px 2px 0 rgba(0, 0, 0, 0.14), 0 1px 5px 0 rgba(0, 0, 0, 0.12)`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,offset:`{focus.ring.offset}`},badgeSize:`1rem`,transitionDuration:`{form.field.transition.duration}`,primary:{background:`{primary.color}`,hoverBackground:`{primary.hover.color}`,activeBackground:`{primary.active.color}`,borderColor:`{primary.color}`,hoverBorderColor:`{primary.hover.color}`,activeBorderColor:`{primary.active.color}`,color:`{primary.contrast.color}`,hoverColor:`{primary.contrast.color}`,activeColor:`{primary.contrast.color}`,focusRing:{color:`{primary.color}`,shadow:`none`}},secondary:{background:`light-dark({surface.100}, {surface.800})`,hoverBackground:`light-dark({surface.200}, {surface.700})`,activeBackground:`light-dark({surface.300}, {surface.600})`,borderColor:`light-dark({surface.100}, {surface.800})`,hoverBorderColor:`light-dark({surface.200}, {surface.700})`,activeBorderColor:`light-dark({surface.300}, {surface.600})`,color:`light-dark({surface.600}, {surface.300})`,hoverColor:`light-dark({surface.700}, {surface.200})`,activeColor:`light-dark({surface.800}, {surface.100})`,focusRing:{color:`light-dark({surface.600}, {surface.300})`,shadow:`none`}},info:{background:`light-dark({sky.500}, {sky.400})`,hoverBackground:`light-dark({sky.600}, {sky.300})`,activeBackground:`light-dark({sky.700}, {sky.200})`,borderColor:`light-dark({sky.500}, {sky.400})`,hoverBorderColor:`light-dark({sky.600}, {sky.300})`,activeBorderColor:`light-dark({sky.700}, {sky.200})`,color:`light-dark(#ffffff, {sky.950})`,hoverColor:`light-dark(#ffffff, {sky.950})`,activeColor:`light-dark(#ffffff, {sky.950})`,focusRing:{color:`light-dark({sky.500}, {sky.400})`,shadow:`none`}},success:{background:`light-dark({green.500}, {green.400})`,hoverBackground:`light-dark({green.600}, {green.300})`,activeBackground:`light-dark({green.700}, {green.200})`,borderColor:`light-dark({green.500}, {green.400})`,hoverBorderColor:`light-dark({green.600}, {green.300})`,activeBorderColor:`light-dark({green.700}, {green.200})`,color:`light-dark(#ffffff, {green.950})`,hoverColor:`light-dark(#ffffff, {green.950})`,activeColor:`light-dark(#ffffff, {green.950})`,focusRing:{color:`light-dark({green.500}, {green.400})`,shadow:`none`}},warn:{background:`light-dark({orange.500}, {orange.400})`,hoverBackground:`light-dark({orange.600}, {orange.300})`,activeBackground:`light-dark({orange.700}, {orange.200})`,borderColor:`light-dark({orange.500}, {orange.400})`,hoverBorderColor:`light-dark({orange.600}, {orange.300})`,activeBorderColor:`light-dark({orange.700}, {orange.200})`,color:`light-dark(#ffffff, {orange.950})`,hoverColor:`light-dark(#ffffff, {orange.950})`,activeColor:`light-dark(#ffffff, {orange.950})`,focusRing:{color:`light-dark({orange.500}, {orange.400})`,shadow:`none`}},help:{background:`light-dark({purple.500}, {purple.400})`,hoverBackground:`light-dark({purple.600}, {purple.300})`,activeBackground:`light-dark({purple.700}, {purple.200})`,borderColor:`light-dark({purple.500}, {purple.400})`,hoverBorderColor:`light-dark({purple.600}, {purple.300})`,activeBorderColor:`light-dark({purple.700}, {purple.200})`,color:`light-dark(#ffffff, {purple.950})`,hoverColor:`light-dark(#ffffff, {purple.950})`,activeColor:`light-dark(#ffffff, {purple.950})`,focusRing:{color:`light-dark({purple.500}, {purple.400})`,shadow:`none`}},danger:{background:`light-dark({red.500}, {red.400})`,hoverBackground:`light-dark({red.600}, {red.300})`,activeBackground:`light-dark({red.700}, {red.200})`,borderColor:`light-dark({red.500}, {red.400})`,hoverBorderColor:`light-dark({red.600}, {red.300})`,activeBorderColor:`light-dark({red.700}, {red.200})`,color:`light-dark(#ffffff, {red.950})`,hoverColor:`light-dark(#ffffff, {red.950})`,activeColor:`light-dark(#ffffff, {red.950})`,focusRing:{color:`light-dark({red.500}, {red.400})`,shadow:`none`}},contrast:{background:`light-dark({surface.950}, {surface.0})`,hoverBackground:`light-dark({surface.900}, {surface.100})`,activeBackground:`light-dark({surface.800}, {surface.200})`,borderColor:`light-dark({surface.950}, {surface.0})`,hoverBorderColor:`light-dark({surface.900}, {surface.100})`,activeBorderColor:`light-dark({surface.800}, {surface.200})`,color:`light-dark({surface.0}, {surface.950})`,hoverColor:`light-dark({surface.0}, {surface.950})`,activeColor:`light-dark({surface.0}, {surface.950})`,focusRing:{color:`light-dark({surface.950}, {surface.0})`,shadow:`none`}},fontSize:`{form.field.font.size}`},outlined:{primary:{hoverBackground:`light-dark({primary.50}, color-mix(in srgb, {primary.color}, transparent 96%))`,activeBackground:`light-dark({primary.100}, color-mix(in srgb, {primary.color}, transparent 84%))`,borderColor:`light-dark({primary.200}, {primary.700})`,color:`{primary.color}`},secondary:{hoverBackground:`light-dark({surface.50}, rgba(255,255,255,0.04))`,activeBackground:`light-dark({surface.100}, rgba(255,255,255,0.16))`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.500}, {surface.400})`},success:{hoverBackground:`light-dark({green.50}, color-mix(in srgb, {green.400}, transparent 96%))`,activeBackground:`light-dark({green.100}, color-mix(in srgb, {green.400}, transparent 84%))`,borderColor:`light-dark({green.200}, {green.700})`,color:`light-dark({green.500}, {green.400})`},info:{hoverBackground:`light-dark({sky.50}, color-mix(in srgb, {sky.400}, transparent 96%))`,activeBackground:`light-dark({sky.100}, color-mix(in srgb, {sky.400}, transparent 84%))`,borderColor:`light-dark({sky.200}, {sky.700})`,color:`light-dark({sky.500}, {sky.400})`},warn:{hoverBackground:`light-dark({orange.50}, color-mix(in srgb, {orange.400}, transparent 96%))`,activeBackground:`light-dark({orange.100}, color-mix(in srgb, {orange.400}, transparent 84%))`,borderColor:`light-dark({orange.200}, {orange.700})`,color:`light-dark({orange.500}, {orange.400})`},help:{hoverBackground:`light-dark({purple.50}, color-mix(in srgb, {purple.400}, transparent 96%))`,activeBackground:`light-dark({purple.100}, color-mix(in srgb, {purple.400}, transparent 84%))`,borderColor:`light-dark({purple.200}, {purple.700})`,color:`light-dark({purple.500}, {purple.400})`},danger:{hoverBackground:`light-dark({red.50}, color-mix(in srgb, {red.400}, transparent 96%))`,activeBackground:`light-dark({red.100}, color-mix(in srgb, {red.400}, transparent 84%))`,borderColor:`light-dark({red.200}, {red.700})`,color:`light-dark({red.500}, {red.400})`},contrast:{hoverBackground:`light-dark({surface.50}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.700})`,borderColor:`light-dark({surface.700}, {surface.500})`,color:`light-dark({surface.950}, {surface.0})`},plain:{hoverBackground:`light-dark({surface.50}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.700})`,borderColor:`light-dark({surface.200}, {surface.600})`,color:`light-dark({surface.700}, {surface.0})`}},text:{primary:{hoverBackground:`light-dark({primary.50}, color-mix(in srgb, {primary.color}, transparent 96%))`,activeBackground:`light-dark({primary.100}, color-mix(in srgb, {primary.color}, transparent 84%))`,color:`{primary.color}`},secondary:{hoverBackground:`light-dark({surface.50}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.700})`,color:`light-dark({surface.500}, {surface.400})`},success:{hoverBackground:`light-dark({green.50}, color-mix(in srgb, {green.400}, transparent 96%))`,activeBackground:`light-dark({green.100}, color-mix(in srgb, {green.400}, transparent 84%))`,color:`light-dark({green.500}, {green.400})`},info:{hoverBackground:`light-dark({sky.50}, color-mix(in srgb, {sky.400}, transparent 96%))`,activeBackground:`light-dark({sky.100}, color-mix(in srgb, {sky.400}, transparent 84%))`,color:`light-dark({sky.500}, {sky.400})`},warn:{hoverBackground:`light-dark({orange.50}, color-mix(in srgb, {orange.400}, transparent 96%))`,activeBackground:`light-dark({orange.100}, color-mix(in srgb, {orange.400}, transparent 84%))`,color:`light-dark({orange.500}, {orange.400})`},help:{hoverBackground:`light-dark({purple.50}, color-mix(in srgb, {purple.400}, transparent 96%))`,activeBackground:`light-dark({purple.100}, color-mix(in srgb, {purple.400}, transparent 84%))`,color:`light-dark({purple.500}, {purple.400})`},danger:{hoverBackground:`light-dark({red.50}, color-mix(in srgb, {red.400}, transparent 96%))`,activeBackground:`light-dark({red.100}, color-mix(in srgb, {red.400}, transparent 84%))`,color:`light-dark({red.500}, {red.400})`},contrast:{hoverBackground:`light-dark({surface.50}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.700})`,color:`light-dark({surface.950}, {surface.0})`},plain:{hoverBackground:`light-dark({surface.50}, {surface.800})`,activeBackground:`light-dark({surface.100}, {surface.700})`,color:`light-dark({surface.700}, {surface.0})`}},link:{color:`{primary.color}`,hoverColor:`{primary.color}`,activeColor:`{primary.color}`}},Df={root:{background:`{content.background}`,borderRadius:`{border.radius.xl}`,color:`{content.color}`,shadow:`0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)`},body:{padding:`1.25rem`,gap:`0.5rem`},caption:{gap:`0.5rem`},title:{fontSize:`1.25rem`,fontWeight:`500`},subtitle:{color:`{text.muted.color}`,fontSize:`1.125rem`,fontWeight:`{typography.font.weight}`}},Of={root:{transitionDuration:`{transition.duration}`},content:{gap:`0.25rem`},indicatorList:{padding:`1rem`,gap:`0.5rem`},indicator:{width:`2rem`,height:`0.5rem`,borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},background:`light-dark({surface.200}, {surface.700})`,hoverBackground:`light-dark({surface.300}, {surface.600})`,activeBackground:`{primary.color}`}},kf={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontWeight:`{form.field.font.weight}`,fontSize:`{form.field.font.size}`},dropdown:{width:`2.5rem`,color:`{form.field.icon.color}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},list:{padding:`{list.padding}`,gap:`{list.gap}`,mobileIndent:`1rem`},option:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`,icon:{color:`{list.option.icon.color}`,focusColor:`{list.option.icon.focus.color}`,size:`0.875rem`},selectedFontWeight:`{list.option.selected.font.weight}`,fontWeight:`{list.option.font.weight}`,fontSize:`{list.option.font.size}`},clearIcon:{color:`{form.field.icon.color}`}},Af={root:{borderRadius:`{border.radius.sm}`,width:`1.25rem`,height:`1.25rem`,background:`{form.field.background}`,checkedBackground:`{primary.color}`,checkedHoverBackground:`{primary.hover.color}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.border.color}`,checkedBorderColor:`{primary.color}`,checkedHoverBorderColor:`{primary.hover.color}`,checkedFocusBorderColor:`{primary.color}`,checkedDisabledBorderColor:`{form.field.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,shadow:`{form.field.shadow}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{width:`1rem`,height:`1rem`},lg:{width:`1.5rem`,height:`1.5rem`}},icon:{size:`0.875rem`,color:`{form.field.color}`,checkedColor:`{primary.contrast.color}`,checkedHoverColor:`{primary.contrast.color}`,disabledColor:`{form.field.disabled.color}`,sm:{size:`0.75rem`},lg:{size:`1rem`}}},jf={root:{borderRadius:`16px`,paddingX:`0.75rem`,paddingY:`0.5rem`,gap:`0.5rem`,transitionDuration:`{transition.duration}`,background:`light-dark({surface.100}, {surface.800})`,color:`light-dark({surface.800}, {surface.0})`,focusBackground:`light-dark({surface.200}, {surface.700})`},image:{width:`2rem`,height:`2rem`},icon:{size:`1rem`,color:`light-dark({surface.800}, {surface.0})`},removeIcon:{size:`1rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},color:`light-dark({surface.800}, {surface.0})`},label:{fontWeight:`{typography.font.weight}`,fontSize:`0.875rem`}},Mf={root:{transitionDuration:`{transition.duration}`},preview:{width:`1.5rem`,height:`1.5rem`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},panel:{shadow:`{overlay.popover.shadow}`,borderRadius:`{overlay.popover.borderRadius}`,background:`light-dark({surface.800}, {surface.900})`,borderColor:`light-dark({surface.900}, {surface.700})`},handle:{color:`{surface.0}`}},Nf={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,height:`28.625rem`},header:{padding:`0.375rem 1.25rem`,background:`{content.background}`,borderColor:`{content.border.color}`},input:{padding:`0.375rem 0`,fontSize:`1.125rem`,fontWeight:`{typography.font.weight}`,color:`{form.field.color}`,placeholderColor:`{form.field.placeholder.color}`},list:{padding:`0.375rem`},empty:{padding:`2.25rem 0`,color:`{content.color}`},footer:{padding:`0.75rem 1.25rem`,background:`{content.background}`,borderColor:`{content.border.color}`}},Pf={root:{borderRadius:`{content.border.radius}`},handle:{background:`{content.background}`,size:`1px`},indicator:{size:`1.75rem`,background:`{content.background}`,borderRadius:`{content.border.radius}`,focusRing:{width:`2px`,style:`solid`,color:`{content.background}`,offset:`2px`},icon:{color:`{text.muted.color}`,size:`{icon.size}`}}},Ff={icon:{size:`2rem`,color:`{overlay.modal.color}`},content:{gap:`1rem`},message:{color:`{content.color}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`}},If={root:{background:`{overlay.popover.background}`,borderColor:`{overlay.popover.border.color}`,color:`{overlay.popover.color}`,borderRadius:`{overlay.popover.border.radius}`,shadow:`{overlay.popover.shadow}`,gutter:`10px`,arrowOffset:`1.25rem`},content:{padding:`{overlay.popover.padding}`,gap:`1rem`},icon:{size:`1.5rem`,color:`{overlay.popover.color}`},footer:{gap:`0.5rem`,padding:`0 {overlay.popover.padding} {overlay.popover.padding} {overlay.popover.padding}`},message:{color:`{content.color}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`}},Lf={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,shadow:`{overlay.navigation.shadow}`,transitionDuration:`{transition.duration}`},list:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`},item:{focusBackground:`{navigation.item.focus.background}`,activeBackground:`{navigation.item.active.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,activeColor:`{navigation.item.active.color}`,padding:`{navigation.item.padding}`,borderRadius:`{navigation.item.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,activeColor:`{navigation.item.icon.active.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},submenu:{mobileIndent:`1rem`},submenuIcon:{size:`{navigation.submenu.icon.size}`,color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`,activeColor:`{navigation.submenu.icon.active.color}`},separator:{borderColor:`{content.border.color}`},submenuLabel:{padding:`{navigation.submenu.label.padding}`,fontWeight:`{navigation.submenu.label.font.weight}`,fontSize:`{navigation.submenu.label.font.size}`,background:`{navigation.submenu.label.background}`,color:`{navigation.submenu.label.color}`}},Rf=`\n"
    "    li.p-autocomplete-option,\n"
    "    div.p-cascadeselect-option-content,\n"
    "    li.p-listbox-option,\n"
    "    li.p-multiselect-option,\n"
    "    li.p-select-option,\n"
    "    li.p-listbox-option,\n"
    "    div.p-tree-node-content,\n"
    "    li.p-datatable-filter-constraint,\n"
    "    .p-datatable .p-datatable-tbody > tr,\n"
    "    .p-treetable .p-treetable-tbody > tr,\n"
    "    div.p-menu-item-content,\n"
    "    div.p-tieredmenu-item-content,\n"
    "    div.p-contextmenu-item-content,\n"
    "    div.p-menubar-item-content,\n"
    "    div.p-megamenu-item-content,\n"
    "    div.p-panelmenu-header-content,\n"
    "    div.p-panelmenu-item-content,\n"
    "    th.p-datatable-header-cell,\n"
    "    th.p-treetable-header-cell,\n"
    "    thead.p-datatable-thead > tr > th,\n"
    "    .p-treetable thead.p-treetable-thead>tr>th {\n"
    "        transition: none;\n"
    "    }\n"
    "`,zf={root:{transitionDuration:`{transition.duration}`,borderColor:`light-dark({content.border.color}, {surface.800})`},header:{background:`{content.background}`,borderColor:`{datatable.border.color}`,color:`{content.color}`,borderWidth:`0 0 1px 0`,padding:`0.75rem 1rem`,sm:{padding:`0.375rem 0.5rem`},lg:{padding:`1rem 1.25rem`}},headerCell:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{content.background}`,borderColor:`{datatable.border.color}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,selectedColor:`{content.color}`,gap:`0.5rem`,padding:`0.75rem 1rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},sm:{padding:`0.375rem 0.5rem`},lg:{padding:`1rem 1.25rem`}},columnTitle:{fontWeight:`600`,fontSize:`{typography.font.size}`},row:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,selectedColor:`{highlight.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},stripedBackground:`light-dark({surface.50}, {surface.950})`},bodyCell:{borderColor:`{datatable.border.color}`,padding:`0.75rem 1rem`,sm:{padding:`0.375rem 0.5rem`},lg:{padding:`1rem 1.25rem`},selectedBorderColor:`light-dark({primary.100}, {primary.900})`,fontWeight:`{typography.font.size}`,fontSize:`{typography.font.size}`},footerCell:{background:`{content.background}`,borderColor:`{datatable.border.color}`,color:`{content.color}`,padding:`0.75rem 1rem`,sm:{padding:`0.375rem 0.5rem`},lg:{padding:`1rem 1.25rem`}},columnFooter:{fontWeight:`600`,fontSize:`{typography.font.size}`},footer:{background:`{content.background}`,borderColor:`{datatable.border.color}`,color:`{content.color}`,borderWidth:`0 0 1px 0`,padding:`0.75rem 1rem`,sm:{padding:`0.375rem 0.5rem`},lg:{padding:`1rem 1.25rem`}},dropPoint:{color:`{primary.color}`},columnResizer:{width:`0.5rem`},resizeIndicator:{width:`1px`,color:`{primary.color}`},sortIcon:{color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,size:`0.875rem`},loadingIcon:{size:`2rem`},rowToggleButton:{hoverBackground:`{content.hover.background}`,selectedHoverBackground:`{content.background}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,selectedHoverColor:`{primary.color}`,size:`1.75rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},filter:{inlineGap:`0.5rem`,overlaySelect:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},overlayPopover:{background:`{overlay.popover.background}`,borderColor:`{overlay.popover.border.color}`,borderRadius:`{overlay.popover.border.radius}`,color:`{overlay.popover.color}`,shadow:`{overlay.popover.shadow}`,padding:`{overlay.popover.padding}`,gap:`0.5rem`},rule:{borderColor:`{content.border.color}`},constraintList:{padding:`{list.padding}`,gap:`{list.gap}`},constraint:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,separator:{borderColor:`{content.border.color}`},padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`}},paginatorTop:{borderColor:`{datatable.border.color}`,borderWidth:`0 0 1px 0`},paginatorBottom:{borderColor:`{datatable.border.color}`,borderWidth:`0 0 1px 0`},css:`\n"
    "    .p-datatable-mask.p-overlay-mask {\n"
    "        --px-mask-background: light-dark(rgba(255,255,255,0.5),rgba(0,0,0,0.3));\n"
    "    }\n"
    "`},Bf={root:{borderColor:`transparent`,borderWidth:`0`,borderRadius:`0`,padding:`0`},header:{background:`{content.background}`,color:`{content.color}`,borderColor:`{content.border.color}`,borderWidth:`0 0 1px 0`,padding:`0.75rem 1rem`,borderRadius:`0`},content:{background:`{content.background}`,color:`{content.color}`,borderColor:`transparent`,borderWidth:`0`,padding:`0`,borderRadius:`0`},footer:{background:`{content.background}`,color:`{content.color}`,borderColor:`{content.border.color}`,borderWidth:`1px 0 0 0`,padding:`0.75rem 1rem`,borderRadius:`0`},paginatorTop:{borderColor:`{content.border.color}`,borderWidth:`0 0 1px 0`},paginatorBottom:{borderColor:`{content.border.color}`,borderWidth:`1px 0 0 0`}},Vf={root:{transitionDuration:`{transition.duration}`},panel:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,shadow:`{overlay.popover.shadow}`,padding:`{overlay.popover.padding}`},header:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,padding:`0 0 0.5rem 0`},title:{gap:`0.5rem`,fontWeight:`500`,fontSize:`{typography.font.size}`},dropdown:{width:`2.5rem`,sm:{width:`2rem`},lg:{width:`3rem`},borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.border.color}`,activeBorderColor:`{form.field.border.color}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},background:`light-dark({surface.100}, {surface.800})`,hoverBackground:`light-dark({surface.200}, {surface.700})`,activeBackground:`light-dark({surface.300}, {surface.600})`,color:`light-dark({surface.600}, {surface.300})`,hoverColor:`light-dark({surface.700}, {surface.200})`,activeColor:`light-dark({surface.800}, {surface.100})`},inputIcon:{color:`{form.field.icon.color}`},selectMonth:{hoverBackground:`{content.hover.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,padding:`0.25rem 0.5rem`,borderRadius:`{content.border.radius}`,fontWeight:`500`,fontSize:`{typography.font.size}`},selectYear:{hoverBackground:`{content.hover.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,padding:`0.25rem 0.5rem`,borderRadius:`{content.border.radius}`,fontWeight:`500`,fontSize:`{typography.font.size}`},group:{borderColor:`{content.border.color}`,gap:`{overlay.popover.padding}`},dayView:{margin:`0.5rem 0 0 0`},weekDay:{padding:`0.25rem`,fontWeight:`500`,color:`{content.color}`,fontSize:`{typography.font.size}`},date:{hoverBackground:`{content.hover.background}`,selectedBackground:`{primary.color}`,rangeSelectedBackground:`{highlight.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,selectedColor:`{primary.contrast.color}`,rangeSelectedColor:`{highlight.color}`,width:`2rem`,height:`2rem`,borderRadius:`50%`,padding:`0.25rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},monthView:{margin:`0.5rem 0 0 0`},month:{padding:`0.375rem`,borderRadius:`{content.border.radius}`},yearView:{margin:`0.5rem 0 0 0`},year:{padding:`0.375rem`,borderRadius:`{content.border.radius}`},buttonbar:{padding:`0.5rem 0 0 0`,borderColor:`{content.border.color}`},timePicker:{padding:`0.5rem 0 0 0`,borderColor:`{content.border.color}`,gap:`0.5rem`,buttonGap:`0.25rem`,color:`{content.color}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},today:{background:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.900}, {surface.0})`}},Hf={root:{background:`{overlay.modal.background}`,borderColor:`{overlay.modal.border.color}`,color:`{overlay.modal.color}`,borderRadius:`{overlay.modal.border.radius}`,shadow:`{overlay.modal.shadow}`},header:{padding:`{overlay.modal.padding}`,gap:`0.5rem`},title:{fontSize:`1.25rem`,fontWeight:`600`},content:{padding:`0 {overlay.modal.padding} {overlay.modal.padding} {overlay.modal.padding}`},footer:{padding:`0 {overlay.modal.padding} {overlay.modal.padding} {overlay.modal.padding}`,gap:`0.5rem`}},Uf={root:{borderColor:`{content.border.color}`},content:{background:`{content.background}`,color:`{text.color}`},horizontal:{margin:`1rem 0`,padding:`0 1rem`,content:{padding:`0 0.5rem`}},vertical:{margin:`0 1rem`,padding:`0.5rem 0`,content:{padding:`0.5rem 0`}}},Wf={root:{background:`rgba(255, 255, 255, 0.1)`,borderColor:`rgba(255, 255, 255, 0.2)`,padding:`0.5rem`,borderRadius:`{border.radius.xl}`},item:{borderRadius:`{content.border.radius}`,padding:`0.5rem`,size:`3rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},Gf={root:{background:`{overlay.modal.background}`,borderColor:`{overlay.modal.border.color}`,color:`{overlay.modal.color}`,shadow:`{overlay.modal.shadow}`},header:{padding:`{overlay.modal.padding}`},title:{fontSize:`1.5rem`,fontWeight:`600`},content:{padding:`0 {overlay.modal.padding} {overlay.modal.padding} {overlay.modal.padding}`},footer:{padding:`{overlay.modal.padding}`}},Kf={toolbar:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`},toolbarItem:{color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{primary.color}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`,padding:`{list.padding}`},overlayOption:{focusBackground:`{list.option.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`},content:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`}},qf={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,color:`{content.color}`,padding:`0 1.125rem 1.125rem 1.125rem`,transitionDuration:`{transition.duration}`},legend:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,borderRadius:`{content.border.radius}`,borderWidth:`1px`,borderColor:`transparent`,padding:`0.5rem 0.75rem`,gap:`0.5rem`,fontWeight:`600`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},fontSize:`{typography.font.size}`},toggleIcon:{color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`},content:{padding:`0`}},Jf={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,transitionDuration:`{transition.duration}`},header:{background:`transparent`,color:`{text.color}`,padding:`1.125rem`,borderColor:`unset`,borderWidth:`0`,borderRadius:`0`,gap:`0.5rem`},content:{highlightBorderColor:`{primary.color}`,padding:`0 1.125rem 1.125rem 1.125rem`,gap:`1rem`},file:{padding:`1rem`,gap:`1rem`,borderColor:`{content.border.color}`,info:{gap:`0.5rem`}},fileList:{gap:`0.5rem`},progressbar:{height:`0.25rem`},basic:{gap:`0.5rem`},fileName:{color:`{text.color}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},fileSize:{color:`{text.muted.color}`,fontWeight:`{typography.font.weight}`,fontSize:`0.875rem`}},Yf={root:{color:`{form.field.float.label.color}`,focusColor:`{form.field.float.label.focus.color}`,activeColor:`{form.field.float.label.active.color}`,invalidColor:`{form.field.float.label.invalid.color}`,transitionDuration:`0.2s`,positionX:`{form.field.padding.x}`,positionY:`{form.field.padding.y}`,fontWeight:`500`,active:{fontSize:`0.75rem`,fontWeight:`400`},fontSize:`{form.field.font.size}`},over:{active:{top:`-1.25rem`}},in:{input:{paddingTop:`1.5rem`,paddingBottom:`{form.field.padding.y}`},active:{top:`{form.field.padding.y}`}},on:{borderRadius:`{border.radius.xs}`,active:{background:`{form.field.background}`,padding:`0 0.125rem`}}},Xf={root:{borderWidth:`1px`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,transitionDuration:`{transition.duration}`},navButton:{background:`rgba(255, 255, 255, 0.1)`,hoverBackground:`rgba(255, 255, 255, 0.2)`,color:`{surface.100}`,hoverColor:`{surface.0}`,size:`3rem`,gutter:`0.5rem`,prev:{borderRadius:`50%`},next:{borderRadius:`50%`},focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},navIcon:{size:`1.5rem`},thumbnailsContent:{background:`{content.background}`,padding:`1rem 0.25rem`},thumbnailNavButton:{size:`2rem`,borderRadius:`{content.border.radius}`,gutter:`0.5rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},hoverBackground:`light-dark({surface.100}, {surface.700})`,color:`light-dark({surface.600}, {surface.400})`,hoverColor:`light-dark({surface.700}, {surface.0})`},thumbnailNavButtonIcon:{size:`1rem`},caption:{background:`rgba(0, 0, 0, 0.5)`,color:`{surface.100}`,padding:`1rem`},indicatorList:{gap:`0.5rem`,padding:`1rem`},indicatorButton:{width:`1rem`,height:`1rem`,activeBackground:`{primary.color}`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},background:`light-dark({surface.200}, {surface.700})`,hoverBackground:`light-dark({surface.300}, {surface.600})`},insetIndicatorList:{background:`rgba(0, 0, 0, 0.5)`},insetIndicatorButton:{background:`rgba(255, 255, 255, 0.4)`,hoverBackground:`rgba(255, 255, 255, 0.6)`,activeBackground:`rgba(255, 255, 255, 0.9)`},closeButton:{size:`3rem`,gutter:`0.5rem`,background:`rgba(255, 255, 255, 0.1)`,hoverBackground:`rgba(255, 255, 255, 0.2)`,color:`{surface.50}`,hoverColor:`{surface.0}`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},closeButtonIcon:{size:`1.5rem`}},Zf={backdrop:{background:`{surface.950}`},header:{padding:`0.875rem 1.125rem`,background:`{surface.950}`},footer:{padding:`0.25rem 0`,background:`{surface.950}`,borderColor:`{surface.800}`},item:{transitionDuration:`0.3s`},action:{size:`2.625rem`,borderRadius:`50%`,color:`{surface.400}`,hoverBackground:`{surface.800}`,hoverColor:`{surface.0}`,disabledOpacity:`{disabled.opacity}`,transitionDuration:`{transition.duration}`,icon:{size:`1.125rem`}},navigation:{background:`color-mix(in srgb, {surface.800}, transparent 40%)`,size:`2.625rem`,borderRadius:`50%`,color:`{surface.400}`,hoverBackground:`{surface.800}`,hoverColor:`{surface.0}`,offset:`0.625rem`,transitionDuration:`{transition.duration}`,icon:{size:`1.125rem`}},thumbnail:{size:`5.75rem`,padding:`0.25rem`,background:`{surface.800}`,borderRadius:`0.25rem`,borderWidth:`3px`,hoverBorderColor:`{surface.700}`,activeBorderColor:`{primary.color}`,activeScale:`0.85`,transitionDuration:`{transition.duration}`},thumbnailContent:{padding:`0.25rem 0`}},Qf={icon:{color:`{form.field.icon.color}`}},$f={root:{color:`{form.field.float.label.color}`,focusColor:`{form.field.float.label.focus.color}`,invalidColor:`{form.field.float.label.invalid.color}`,transitionDuration:`0.2s`,positionX:`{form.field.padding.x}`,top:`{form.field.padding.y}`,fontSize:`0.75rem`,fontWeight:`400`},input:{paddingTop:`1.5rem`,paddingBottom:`{form.field.padding.y}`}},ep={root:{transitionDuration:`{transition.duration}`},preview:{icon:{size:`1.5rem`},mask:{background:`{mask.background}`,color:`{mask.color}`}},toolbar:{position:{left:`auto`,right:`1rem`,top:`1rem`,bottom:`auto`},blur:`8px`,background:`rgba(255,255,255,0.1)`,borderColor:`rgba(255,255,255,0.2)`,borderWidth:`1px`,borderRadius:`30px`,padding:`.5rem`,gap:`0.5rem`},action:{hoverBackground:`rgba(255,255,255,0.1)`,color:`{surface.50}`,hoverColor:`{surface.0}`,size:`3rem`,iconSize:`1.5rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},tp={handle:{size:`15px`,hoverSize:`30px`,background:`rgba(255,255,255,0.3)`,hoverBackground:`rgba(255,255,255,0.3)`,borderColor:`unset`,hoverBorderColor:`unset`,borderWidth:`0`,borderRadius:`50%`,transitionDuration:`{transition.duration}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`rgba(255,255,255,0.3)`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},np={root:{padding:`{form.field.padding.y} {form.field.padding.x}`,borderRadius:`{content.border.radius}`,gap:`0.5rem`},text:{fontWeight:`500`},icon:{size:`1rem`},info:{background:`light-dark(color-mix(in srgb, {blue.50}, transparent 5%), color-mix(in srgb, {blue.500}, transparent 84%))`,borderColor:`light-dark({blue.200}, color-mix(in srgb, {blue.700}, transparent 64%))`,color:`light-dark({blue.600}, {blue.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {blue.500}, transparent 96%)`},success:{background:`light-dark(color-mix(in srgb, {green.50}, transparent 5%), color-mix(in srgb, {green.500}, transparent 84%))`,borderColor:`light-dark({green.200}, color-mix(in srgb, {green.700}, transparent 64%))`,color:`light-dark({green.600}, {green.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {green.500}, transparent 96%)`},warn:{background:`light-dark(color-mix(in srgb,{yellow.50}, transparent 5%), color-mix(in srgb, {yellow.500}, transparent 84%))`,borderColor:`light-dark({yellow.200}, color-mix(in srgb, {yellow.700}, transparent 64%))`,color:`light-dark({yellow.600}, {yellow.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {yellow.500}, transparent 96%)`},error:{background:`light-dark(color-mix(in srgb, {red.50}, transparent 5%), color-mix(in srgb, {red.500}, transparent 84%))`,borderColor:`light-dark({red.200}, color-mix(in srgb, {red.700}, transparent 64%))`,color:`light-dark({red.600}, {red.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {red.500}, transparent 96%)`},secondary:{background:`light-dark({surface.100}, {surface.800})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.600}, {surface.300})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.500}, transparent 96%)`},contrast:{background:`light-dark({surface.900}, {surface.0})`,borderColor:`light-dark({surface.950}, {surface.100})`,color:`light-dark({surface.50}, {surface.950})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.950}, transparent 96%)`}},rp={root:{padding:`{form.field.padding.y} {form.field.padding.x}`,borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},transitionDuration:`{transition.duration}`},display:{hoverBackground:`{content.hover.background}`,hoverColor:`{content.hover.color}`}},ip={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`},chip:{borderRadius:`{border.radius.sm}`,focusBackground:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.800}, {surface.0})`}},ap={root:{borderColor:`{content.border.color}`},area:{borderRadius:`{content.border.radius}`},slider:{borderRadius:`{content.border.radius}`,size:`1.125rem`},handle:{size:`1.125rem`,borderColor:`#ffffff`,borderWidth:`3px`,shadow:`0px 0.5px 0px 0px rgba(0, 0, 0, 0.08), 0px 1px 1px 0px rgba(0, 0, 0, 0.14)`,transitionDuration:`{transition.duration}`,focusRing:{borderWidth:`2px`,borderColor:`#ffffff`,outlineWidth:`2px`,outlineColor:`rgba(255, 255, 255, 0.3)`,outlineOffset:`2px`}},transparencyGrid:{color:`{surface.100}`,background:`#ffffff`,tileSize:`0.625rem`},swatch:{size:`2.625rem`,borderRadius:`{content.border.radius}`}},op={addon:{background:`{form.field.background}`,borderColor:`{form.field.border.color}`,color:`{form.field.icon.color}`,borderRadius:`{form.field.border.radius}`,padding:`0.5rem`,minWidth:`2.5rem`,fontWeight:`{form.field.font.weight}`,fontSize:`{form.field.font.size}`}},sp={root:{transitionDuration:`{transition.duration}`},button:{width:`2.5rem`,borderRadius:`{form.field.border.radius}`,verticalPadding:`{form.field.padding.y}`,background:`transparent`,hoverBackground:`light-dark({surface.100}, {surface.800})`,activeBackground:`light-dark({surface.200}, {surface.700})`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.border.color}`,activeBorderColor:`{form.field.border.color}`,color:`{surface.400}`,hoverColor:`light-dark({surface.500}, {surface.300})`,activeColor:`light-dark({surface.600}, {surface.200})`}},cp={root:{gap:`0.5rem`},input:{width:`2.5rem`,sm:{width:`2rem`},lg:{width:`3rem`}}},lp={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,gap:`0.25rem`},item:{borderRadius:`{form.field.border.radius}`}},up={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontSize:`{form.field.font.size}`,fontWeight:`{form.field.font.weight}`}},dp={root:{transitionDuration:`{transition.duration}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},value:{background:`{primary.color}`},range:{background:`{content.border.color}`},text:{color:`{text.muted.color}`,fontSize:`1.25rem`,fontWeight:`normal`}},fp={root:{gap:`0.375rem`,fontSize:`{typography.font.size}`,fontWeight:`500`,textColor:`{text.color}`,disabledOpacity:`{disabled.opacity}`}},pp={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,borderColor:`{form.field.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,shadow:`{form.field.shadow}`,borderRadius:`{form.field.border.radius}`,transitionDuration:`{form.field.transition.duration}`},list:{padding:`{list.padding}`,gap:`{list.gap}`,header:{padding:`{list.header.padding}`}},option:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`,stripedBackground:`light-dark({surface.50}, {surface.900})`,fontWeight:`{list.option.font.weight}`,fontSize:`{list.option.font.size}`,selectedFontWeight:`{list.option.selected.font.weight}`},optionGroup:{background:`{list.option.group.background}`,color:`{list.option.group.color}`,fontWeight:`{list.option.group.font.weight}`,padding:`{list.option.group.padding}`,fontSize:`{list.option.group.font.size}`},checkmark:{color:`{list.option.color}`,gutterStart:`-0.375rem`,gutterEnd:`0.375rem`},emptyMessage:{padding:`{list.option.padding}`}},mp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,color:`{content.color}`,gap:`0.5rem`,verticalOrientation:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`},horizontalOrientation:{padding:`0.5rem 0.75rem`,gap:`0.5rem`},transitionDuration:`{transition.duration}`},baseItem:{borderRadius:`{content.border.radius}`,padding:`{navigation.item.padding}`},item:{focusBackground:`{navigation.item.focus.background}`,activeBackground:`{navigation.item.active.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,activeColor:`{navigation.item.active.color}`,padding:`{navigation.item.padding}`,borderRadius:`{navigation.item.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,activeColor:`{navigation.item.icon.active.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},overlay:{padding:`0`,background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,color:`{content.color}`,shadow:`{overlay.navigation.shadow}`,gap:`0.5rem`},submenu:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`},submenuLabel:{padding:`{navigation.submenu.label.padding}`,fontWeight:`{navigation.submenu.label.font.weight}`,background:`{navigation.submenu.label.background}`,color:`{navigation.submenu.label.color}`,fontSize:`{navigation.submenu.label.font.size}`},submenuIcon:{size:`{navigation.submenu.icon.size}`,color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`,activeColor:`{navigation.submenu.icon.active.color}`},separator:{borderColor:`{content.border.color}`},mobileButton:{borderRadius:`50%`,size:`1.75rem`,color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,hoverBackground:`{content.hover.background}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},hp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,shadow:`{overlay.navigation.shadow}`,transitionDuration:`{transition.duration}`},list:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`},item:{focusBackground:`{navigation.item.focus.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,padding:`{navigation.item.padding}`,borderRadius:`{navigation.item.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},submenuLabel:{padding:`{navigation.submenu.label.padding}`,fontWeight:`{navigation.submenu.label.font.weight}`,background:`{navigation.submenu.label.background}`,color:`{navigation.submenu.label.color}`,fontSize:`{navigation.submenu.label.font.size}`},separator:{borderColor:`{content.border.color}`},submenuIcon:{size:`{navigation.submenu.icon.size}`,color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`}},gp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,color:`{content.color}`,gap:`0.5rem`,padding:`0.5rem 0.75rem`,transitionDuration:`{transition.duration}`},baseItem:{borderRadius:`{content.border.radius}`,padding:`{navigation.item.padding}`},item:{focusBackground:`{navigation.item.focus.background}`,activeBackground:`{navigation.item.active.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,activeColor:`{navigation.item.active.color}`,padding:`{navigation.item.padding}`,borderRadius:`{navigation.item.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,activeColor:`{navigation.item.icon.active.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},submenu:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`,background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,shadow:`{overlay.navigation.shadow}`,mobileIndent:`1rem`,icon:{size:`{navigation.submenu.icon.size}`,color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`,activeColor:`{navigation.submenu.icon.active.color}`}},separator:{borderColor:`{content.border.color}`},mobileButton:{borderRadius:`50%`,size:`1.75rem`,color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,hoverBackground:`{content.hover.background}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},_p={root:{borderRadius:`{content.border.radius}`,borderWidth:`1px`,transitionDuration:`{transition.duration}`},content:{padding:`0.5rem 0.75rem`,gap:`0.5rem`,sm:{padding:`0.375rem 0.625rem`},lg:{padding:`0.625rem 0.875rem`}},text:{fontSize:`1rem`,fontWeight:`500`,sm:{fontSize:`0.875rem`},lg:{fontSize:`1.125rem`}},icon:{size:`1.125rem`,sm:{size:`1rem`},lg:{size:`1.25rem`}},closeButton:{width:`1.75rem`,height:`1.75rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,offset:`{focus.ring.offset}`}},closeIcon:{size:`1rem`,sm:{size:`0.875rem`},lg:{size:`1.125rem`}},outlined:{root:{borderWidth:`1px`}},simple:{content:{padding:`0`}},info:{background:`light-dark(color-mix(in srgb, {blue.50}, transparent 5%), color-mix(in srgb, {blue.500}, transparent 84%))`,borderColor:`light-dark({blue.200}, color-mix(in srgb, {blue.700}, transparent 64%))`,color:`light-dark({blue.600}, {blue.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {blue.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({blue.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({blue.600}, {blue.500})`,shadow:`none`}},outlined:{color:`light-dark({blue.600}, {blue.500})`,borderColor:`light-dark({blue.600}, {blue.500})`},simple:{color:`light-dark({blue.600}, {blue.500})`}},success:{background:`light-dark(color-mix(in srgb, {green.50}, transparent 5%), color-mix(in srgb, {green.500}, transparent 84%))`,borderColor:`light-dark({green.200}, color-mix(in srgb, {green.700}, transparent 64%))`,color:`light-dark({green.600}, {green.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {green.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({green.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({green.600}, {green.500})`,shadow:`none`}},outlined:{color:`light-dark({green.600}, {green.500})`,borderColor:`light-dark({green.600}, {green.500})`},simple:{color:`light-dark({green.600}, {green.500})`}},warn:{background:`light-dark(color-mix(in srgb,{yellow.50}, transparent 5%), color-mix(in srgb, {yellow.500}, transparent 84%))`,borderColor:`light-dark({yellow.200}, color-mix(in srgb, {yellow.700}, transparent 64%))`,color:`light-dark({yellow.600}, {yellow.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {yellow.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({yellow.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({yellow.600}, {yellow.500})`,shadow:`none`}},outlined:{color:`light-dark({yellow.600}, {yellow.500})`,borderColor:`light-dark({yellow.600}, {yellow.500})`},simple:{color:`light-dark({yellow.600}, {yellow.500})`}},error:{background:`light-dark(color-mix(in srgb, {red.50}, transparent 5%), color-mix(in srgb, {red.500}, transparent 84%))`,borderColor:`light-dark({red.200}, color-mix(in srgb, {red.700}, transparent 64%))`,color:`light-dark({red.600}, {red.500})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {red.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({red.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({red.600}, {red.500})`,shadow:`none`}},outlined:{color:`light-dark({red.600}, {red.500})`,borderColor:`light-dark({red.600}, {red.500})`},simple:{color:`light-dark({red.600}, {red.500})`}},secondary:{background:`light-dark({surface.100}, {surface.800})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.600}, {surface.300})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({surface.200}, {surface.700})`,focusRing:{color:`light-dark({surface.600}, {surface.300})`,shadow:`none`}},outlined:{color:`light-dark({surface.500}, {surface.400})`,borderColor:`light-dark({surface.500}, {surface.400})`},simple:{color:`light-dark({surface.500}, {surface.400})`}},contrast:{background:`light-dark({surface.900}, {surface.0})`,borderColor:`light-dark({surface.950}, {surface.100})`,color:`light-dark({surface.50}, {surface.950})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.950}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({surface.800}, {surface.100})`,focusRing:{color:`light-dark({surface.50}, {surface.950})`,shadow:`none`}},outlined:{color:`light-dark({surface.950}, {surface.0})`,borderColor:`light-dark({surface.950}, {surface.0})`},simple:{color:`light-dark({surface.950}, {surface.0})`}}},vp={root:{borderRadius:`{content.border.radius}`,gap:`1rem`},meters:{background:`{content.border.color}`,size:`0.5rem`},label:{gap:`0.5rem`},labelMarker:{size:`0.5rem`},labelIcon:{size:`1rem`},labelList:{verticalGap:`0.5rem`,horizontalGap:`1rem`},labelText:{fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`}},yp={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontSize:`{form.field.font.size}`,fontWeight:`{form.field.font.weight}`},dropdown:{width:`2.5rem`,color:`{form.field.icon.color}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},list:{padding:`{list.padding}`,gap:`{list.gap}`,header:{padding:`{list.header.padding}`}},option:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`,gap:`0.5rem`,fontSize:`{list.option.font.size}`,fontWeight:`{list.option.font.weight}`,selectedFontWeight:`{list.option.selected.font.weight}`},optionGroup:{background:`{list.option.group.background}`,color:`{list.option.group.color}`,fontWeight:`{list.option.group.font.weight}`,padding:`{list.option.group.padding}`,fontSize:`{list.option.group.font.size}`},chip:{borderRadius:`{border.radius.sm}`},clearIcon:{color:`{form.field.icon.color}`},emptyMessage:{padding:`{list.option.padding}`}},bp={root:{padding:`0.375rem 0.75rem`,gap:`0.25rem`},baseItem:{padding:`{navigation.item.padding}`,borderRadius:`{content.border.radius}`,gap:`{navigation.item.gap}`,fontSize:`{navigation.item.label.font.size}`,fontWeight:`500`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,focusBackground:`{navigation.item.focus.background}`,activeColor:`{navigation.item.active.color}`,activeBackground:`{navigation.item.active.background}`,transitionDuration:`{navigation.item.transition.duration}`}},xp={root:{gap:`1.125rem`},controls:{gap:`0.5rem`}},Sp={root:{gutter:`0.75rem`,transitionDuration:`{transition.duration}`},node:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,selectedColor:`{highlight.color}`,hoverColor:`{content.hover.color}`,padding:`0.75rem 1rem`,toggleablePadding:`0.75rem 1rem 1.25rem 1rem`,borderRadius:`{content.border.radius}`,fontSize:`{typography.font.size}`,fontWeight:`{typography.font.weight}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},nodeToggleButton:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,borderColor:`{content.border.color}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,size:`1.5rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},icon:{size:`0.875rem`}},connector:{color:`{content.border.color}`,borderRadius:`{content.border.radius}`,height:`24px`}},Cp={root:{outline:{width:`2px`,color:`{content.background}`}}},wp={root:{padding:`0.5rem 1rem`,gap:`0.25rem`,borderRadius:`{content.border.radius}`,background:`{content.background}`,color:`{content.color}`,transitionDuration:`{transition.duration}`},navButton:{background:`transparent`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,selectedColor:`{highlight.color}`,width:`2.5rem`,height:`2.5rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},currentPageReport:{color:`{text.muted.color}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},jumpToPageInput:{maxWidth:`2.5rem`}},Tp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`},header:{background:`transparent`,color:`{text.color}`,padding:`1.125rem`,borderColor:`{content.border.color}`,borderWidth:`0`,borderRadius:`0`},toggleableHeader:{padding:`0.375rem 1.125rem`},title:{fontWeight:`600`,fontSize:`{typography.font.size}`},content:{padding:`0 1.125rem 1.125rem 1.125rem`},footer:{padding:`0 1.125rem 1.125rem 1.125rem`}},Ep={root:{gap:`0.5rem`,transitionDuration:`{transition.duration}`},panel:{background:`{content.background}`,borderColor:`{content.border.color}`,borderWidth:`1px`,color:`{content.color}`,padding:`0.25rem 0.25rem`,borderRadius:`{content.border.radius}`,first:{borderWidth:`1px`,topBorderRadius:`{content.border.radius}`},last:{borderWidth:`1px`,bottomBorderRadius:`{content.border.radius}`}},item:{focusBackground:`{navigation.item.focus.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,gap:`0.5rem`,padding:`{navigation.item.padding}`,borderRadius:`{content.border.radius}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},submenu:{indent:`1rem`},submenuIcon:{color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`}},Dp={meter:{background:`{content.border.color}`,borderRadius:`{content.border.radius}`,height:`.75rem`},icon:{color:`{form.field.icon.color}`},overlay:{background:`{overlay.popover.background}`,borderColor:`{overlay.popover.border.color}`,borderRadius:`{overlay.popover.border.radius}`,color:`{overlay.popover.color}`,padding:`{overlay.popover.padding}`,shadow:`{overlay.popover.shadow}`},content:{gap:`0.5rem`},strength:{weakBackground:`light-dark({red.500}, {red.400})`,mediumBackground:`light-dark({amber.500}, {amber.400})`,strongBackground:`light-dark({green.500}, {green.400})`},meterText:{fontSize:`{typography.font.size}`,fontWeight:`{typography.font.weight}`}},Op={root:{gap:`1.125rem`},controls:{gap:`0.5rem`}},kp={root:{background:`{overlay.popover.background}`,borderColor:`{overlay.popover.border.color}`,color:`{overlay.popover.color}`,borderRadius:`{overlay.popover.border.radius}`,shadow:`{overlay.popover.shadow}`,gutter:`10px`,arrowOffset:`1.25rem`},content:{padding:`{overlay.popover.padding}`}},Ap={root:{background:`{content.border.color}`,borderRadius:`{content.border.radius}`,height:`1.25rem`},value:{background:`{primary.color}`},label:{color:`{primary.contrast.color}`,fontSize:`0.75rem`,fontWeight:`600`}},jp={root:{colorOne:`light-dark({red.500}, {red.400})`,colorTwo:`light-dark({blue.500}, {blue.400})`,colorThree:`light-dark({green.500}, {green.400})`,colorFour:`light-dark({yellow.500}, {yellow.400})`}},Mp={root:{width:`1.25rem`,height:`1.25rem`,background:`{form.field.background}`,checkedBackground:`{primary.color}`,checkedHoverBackground:`{primary.hover.color}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.border.color}`,checkedBorderColor:`{primary.color}`,checkedHoverBorderColor:`{primary.hover.color}`,checkedFocusBorderColor:`{primary.color}`,checkedDisabledBorderColor:`{form.field.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,shadow:`{form.field.shadow}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{width:`1rem`,height:`1rem`},lg:{width:`1.5rem`,height:`1.5rem`}},icon:{size:`0.75rem`,checkedColor:`{primary.contrast.color}`,checkedHoverColor:`{primary.contrast.color}`,disabledColor:`{form.field.disabled.color}`,sm:{size:`0.5rem`},lg:{size:`1rem`}}},Np={root:{gap:`0.25rem`,transitionDuration:`{transition.duration}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},icon:{size:`1rem`,color:`{text.muted.color}`,hoverColor:`{primary.color}`,activeColor:`{primary.color}`}},Pp={root:{background:`light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.3))`}},Fp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},viewport:{padding:`1.125rem`},scrollbar:{background:`transparent`,margin:`0.25rem`,size:`0.25rem`,transitionDuration:`{transition.duration}`},handle:{background:`{content.border.color}`},mask:{fadeSize:`40px`}},Ip={root:{transitionDuration:`{transition.duration}`},bar:{size:`9px`,borderRadius:`{border.radius.sm}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},background:`light-dark({surface.100}, {surface.800})`}},Lp={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontSize:`{form.field.font.size}`,fontWeight:`{form.field.font.weight}`},dropdown:{width:`2.5rem`,color:`{form.field.icon.color}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},list:{padding:`{list.padding}`,gap:`{list.gap}`,header:{padding:`{list.header.padding}`}},option:{focusBackground:`{list.option.focus.background}`,selectedBackground:`{list.option.selected.background}`,selectedFocusBackground:`{list.option.selected.focus.background}`,color:`{list.option.color}`,focusColor:`{list.option.focus.color}`,selectedColor:`{list.option.selected.color}`,selectedFocusColor:`{list.option.selected.focus.color}`,padding:`{list.option.padding}`,borderRadius:`{list.option.border.radius}`,fontSize:`{list.option.font.size}`,fontWeight:`{list.option.font.weight}`,selectedFontWeight:`{list.option.selected.font.weight}`},optionGroup:{background:`{list.option.group.background}`,color:`{list.option.group.color}`,fontWeight:`{list.option.group.font.weight}`,padding:`{list.option.group.padding}`,fontSize:`{list.option.group.font.size}`},clearIcon:{color:`{form.field.icon.color}`},checkmark:{color:`{list.option.color}`,gutterStart:`-0.375rem`,gutterEnd:`0.375rem`},emptyMessage:{padding:`{list.option.padding}`}},Rp={root:{borderRadius:`{form.field.border.radius}`,invalidBorderColor:`{form.field.invalid.border.color}`}},zp={root:{borderColor:`{content.border.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},layout:{background:`light-dark({surface.50}, {surface.900})`},header:{padding:`0.625rem`,gap:`0.625rem`},footer:{padding:`0.625rem`,gap:`0.625rem`},content:{gap:`0.125rem`},aside:{padding:`0.625rem`},panel:{background:`{content.background}`,color:`{content.color}`,floatingBorderRadius:`{content.border.radius}`,floatingShadow:`0 1px 2px 0 rgb(0 0 0 / 0.05)`},group:{padding:`0.625rem`},groupLabel:{padding:`0 0.625rem`,height:`2.25rem`,borderRadius:`{content.border.radius}`,fontSize:`0.875rem`,fontWeight:`500`,color:`{text.muted.color}`},groupAction:{top:`1rem`,right:`0.875rem`,size:`1.375rem`,borderRadius:`{content.border.radius}`,color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,focusBackground:`{navigation.item.focus.background}`,icon:{size:`{navigation.item.icon.size}`}},menu:{gap:`{navigation.list.gap}`},menuButton:{padding:`0.25rem 0.75rem`,gap:`{navigation.item.gap}`,height:`2.25rem`,borderRadius:`{navigation.item.border.radius}`,fontSize:`{navigation.item.label.font.size}`,fontWeight:`{navigation.item.label.font.weight}`,color:`{navigation.item.color}`,focusBackground:`{navigation.item.focus.background}`,focusColor:`{navigation.item.focus.color}`,activeBackground:`{navigation.item.active.background}`,activeColor:`{navigation.item.active.color}`,iconOnlyWidth:`2.25rem`,withActionPaddingEnd:`2.25rem`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,size:`{navigation.item.icon.size}`}},menuAction:{top:`0.375rem`,right:`0.25rem`,width:`1.375rem`,borderRadius:`{content.border.radius}`,color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,focusBackground:`{navigation.item.focus.background}`,icon:{size:`{navigation.item.icon.size}`}},menuBadge:{top:`0.375rem`,right:`0.25rem`,height:`1.375rem`,minWidth:`1.375rem`,borderRadius:`0.375rem`,padding:`0 0.25rem`,fontSize:`0.875rem`,fontWeight:`500`,background:`{content.hover.background}`,borderColor:`{content.border.color}`,color:`{text.muted.color}`},menuSub:{paddingBlock:`0.125rem`,gap:`0.125rem`,indentMargin:`1rem`,indentPadding:`0.75rem`,collapsibleIndent:`1.75rem`,collapsibleTopMargin:`0.125rem`,collapsibleBorderRadius:`0.375rem`},menuSubButton:{padding:`{navigation.item.padding}`,gap:`{navigation.item.gap}`,height:`2.25rem`,borderRadius:`{navigation.item.border.radius}`,fontSize:`{navigation.item.label.font.size}`,fontWeight:`{navigation.item.label.font.weight}`,color:`{navigation.item.color}`,focusBackground:`{navigation.item.focus.background}`,focusColor:`{navigation.item.focus.color}`,activeBackground:`{navigation.item.active.background}`,activeColor:`{navigation.item.active.color}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,size:`{navigation.item.icon.size}`}},main:{background:`light-dark({surface.50}, {surface.900})`,floatingBackground:`light-dark({surface.50}, {surface.900})`,insetBackground:`light-dark({surface.0}, {surface.950})`,margin:`0.625rem`,borderRadius:`{content.border.radius}`,shadow:`0 1px 2px 0 rgb(0 0 0 / 0.05)`}},Bp={root:{borderRadius:`{content.border.radius}`,background:`light-dark({surface.200}, rgba(255, 255, 255, 0.06))`,animationBackground:`light-dark(rgba(255,255,255,0.4), rgba(255, 255, 255, 0.04))`}},Vp={root:{transitionDuration:`{transition.duration}`},track:{background:`{content.border.color}`,borderRadius:`{content.border.radius}`,size:`3px`},range:{background:`{primary.color}`},handle:{width:`20px`,height:`20px`,borderRadius:`50%`,background:`{content.border.color}`,hoverBackground:`{content.border.color}`,content:{borderRadius:`50%`,hoverBackground:`{content.background}`,width:`16px`,height:`16px`,shadow:`0px 0.5px 0px 0px rgba(0, 0, 0, 0.08), 0px 1px 1px 0px rgba(0, 0, 0, 0.14)`,background:`light-dark({surface.0}, {surface.950})`},focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},Hp={root:{gap:`0.5rem`,transitionDuration:`{transition.duration}`}},Up={root:{borderRadius:`{form.field.border.radius}`,roundedBorderRadius:`2rem`,raisedShadow:`0 3px 1px -2px rgba(0, 0, 0, 0.2), 0 2px 2px 0 rgba(0, 0, 0, 0.14), 0 1px 5px 0 rgba(0, 0, 0, 0.12)`}},Wp={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,transitionDuration:`{transition.duration}`},gutter:{background:`{content.border.color}`},handle:{size:`24px`,background:`transparent`,borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}}},Gp={root:{transitionDuration:`{transition.duration}`},separator:{background:`{content.border.color}`,activeBackground:`{primary.color}`,margin:`0 0 0 1.625rem`,size:`2px`},step:{padding:`0.5rem`,gap:`1rem`},stepHeader:{padding:`0`,borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},gap:`0.5rem`},stepTitle:{color:`{text.muted.color}`,activeColor:`{primary.color}`,fontWeight:`500`,fontSize:`{typography.font.size}`},stepNumber:{background:`{content.background}`,activeBackground:`{content.background}`,borderColor:`{content.border.color}`,activeBorderColor:`{content.border.color}`,color:`{text.muted.color}`,activeColor:`{primary.color}`,size:`2rem`,fontSize:`1.143rem`,fontWeight:`500`,borderRadius:`50%`,shadow:`0px 0.5px 0px 0px rgba(0, 0, 0, 0.06), 0px 1px 1px 0px rgba(0, 0, 0, 0.12)`},steppanels:{padding:`0.875rem 0.5rem 1.125rem 0.5rem`},steppanel:{background:`{content.background}`,color:`{content.color}`,padding:`0`,indent:`1rem`}},Kp={root:{transitionDuration:`{transition.duration}`},separator:{background:`{content.border.color}`},itemLink:{borderRadius:`{content.border.radius}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},gap:`0.5rem`},itemLabel:{color:`{text.muted.color}`,activeColor:`{primary.color}`,fontWeight:`500`},itemNumber:{background:`{content.background}`,activeBackground:`{content.background}`,borderColor:`{content.border.color}`,activeBorderColor:`{content.border.color}`,color:`{text.muted.color}`,activeColor:`{primary.color}`,size:`2rem`,fontSize:`1.143rem`,fontWeight:`500`,borderRadius:`50%`,shadow:`0px 0.5px 0px 0px rgba(0, 0, 0, 0.06), 0px 1px 1px 0px rgba(0, 0, 0, 0.12)`}},qp={root:{transitionDuration:`{transition.duration}`},tablist:{borderWidth:`0 0 1px 0`,background:`{content.background}`,borderColor:`{content.border.color}`},item:{background:`transparent`,hoverBackground:`transparent`,activeBackground:`transparent`,borderWidth:`0 0 1px 0`,borderColor:`{content.border.color}`,hoverBorderColor:`{content.border.color}`,activeBorderColor:`{primary.color}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{primary.color}`,padding:`1rem 1.125rem`,fontWeight:`600`,margin:`0 0 -1px 0`,gap:`0.5rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},itemIcon:{color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{primary.color}`},activeBar:{height:`1px`,bottom:`-1px`,background:`{primary.color}`}},Jp={root:{transitionDuration:`{transition.duration}`},tablist:{borderWidth:`0 0 1px 0`,background:`{content.background}`,borderColor:`{content.border.color}`},tab:{background:`transparent`,hoverBackground:`transparent`,activeBackground:`transparent`,borderWidth:`0 0 1px 0`,borderColor:`{content.border.color}`,hoverBorderColor:`{content.border.color}`,activeBorderColor:`{primary.color}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{primary.color}`,padding:`1rem 1.125rem`,fontWeight:`600`,margin:`0 0 -1px 0`,gap:`0.5rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},fontSize:`{typography.font.size}`},tabpanel:{background:`{content.background}`,color:`{content.color}`,padding:`0.875rem 1.125rem 1.125rem 1.125rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`inset {focus.ring.shadow}`}},navButton:{background:`{content.background}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,width:`2.5rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},shadow:`light-dark(0px 0px 10px 50px rgba(255, 255, 255, 0.6), 0px 0px 10px 50px color-mix(in srgb, {content.background}, transparent 50%))`},activeBar:{height:`1px`,bottom:`0`,background:`{primary.color}`}},Yp={root:{transitionDuration:`{transition.duration}`},tabList:{background:`{content.background}`,borderColor:`{content.border.color}`},tab:{borderColor:`{content.border.color}`,activeBorderColor:`{primary.color}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,activeColor:`{primary.color}`},tabPanel:{background:`{content.background}`,color:`{content.color}`},navButton:{background:`{content.background}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,shadow:`light-dark(0px 0px 10px 50px rgba(255, 255, 255, 0.6), 0px 0px 10px 50px color-mix(in srgb, {content.background}, transparent 50%))`}},Xp={root:{fontSize:`0.875rem`,fontWeight:`700`,padding:`0.25rem 0.5rem`,gap:`0.25rem`,borderRadius:`{content.border.radius}`,roundedBorderRadius:`{border.radius.xl}`},icon:{size:`0.75rem`},primary:{background:`light-dark({primary.100}, color-mix(in srgb, {primary.500}, transparent 84%))`,color:`light-dark({primary.700}, {primary.300})`},secondary:{background:`light-dark({surface.100}, {surface.800})`,color:`light-dark({surface.600}, {surface.300})`},success:{background:`light-dark({green.100}, color-mix(in srgb, {green.500}, transparent 84%))`,color:`light-dark({green.700}, {green.300})`},info:{background:`light-dark({sky.100}, color-mix(in srgb, {sky.500}, transparent 84%))`,color:`light-dark({sky.700}, {sky.300})`},warn:{background:`light-dark({orange.100}, color-mix(in srgb, {orange.500}, transparent 84%))`,color:`light-dark({orange.700}, {orange.300})`},danger:{background:`light-dark({red.100}, color-mix(in srgb, {red.500}, transparent 84%))`,color:`light-dark({red.700}, {red.300})`},contrast:{background:`light-dark({surface.950}, {surface.0})`,color:`light-dark({surface.0}, {surface.950})`}},Zp={root:{background:`{form.field.background}`,borderColor:`{form.field.border.color}`,color:`{form.field.color}`,height:`18rem`,padding:`{form.field.padding.y} {form.field.padding.x}`,borderRadius:`{form.field.border.radius}`,fontWeight:`{typography.font.weight}`,fontSize:`{typography.font.size}`},prompt:{gap:`0.25rem`},commandResponse:{margin:`2px 0`}},Qp={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontSize:`{form.field.font.size}`,fontWeight:`{form.field.font.weight}`}},$p={root:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{content.color}`,borderRadius:`{content.border.radius}`,shadow:`{overlay.navigation.shadow}`,transitionDuration:`{transition.duration}`},list:{padding:`{navigation.list.padding}`,gap:`{navigation.list.gap}`},item:{focusBackground:`{navigation.item.focus.background}`,activeBackground:`{navigation.item.active.background}`,color:`{navigation.item.color}`,focusColor:`{navigation.item.focus.color}`,activeColor:`{navigation.item.active.color}`,padding:`{navigation.item.padding}`,borderRadius:`{navigation.item.border.radius}`,gap:`{navigation.item.gap}`,icon:{color:`{navigation.item.icon.color}`,focusColor:`{navigation.item.icon.focus.color}`,activeColor:`{navigation.item.icon.active.color}`,size:`{navigation.item.icon.size}`},label:{fontWeight:`{navigation.item.label.font.weight}`,fontSize:`{navigation.item.label.font.size}`}},submenu:{mobileIndent:`1rem`},submenuIcon:{size:`{navigation.submenu.icon.size}`,color:`{navigation.submenu.icon.color}`,focusColor:`{navigation.submenu.icon.focus.color}`,activeColor:`{navigation.submenu.icon.active.color}`},separator:{borderColor:`{content.border.color}`}},em={event:{minHeight:`5rem`},horizontal:{eventContent:{padding:`1rem 0`}},vertical:{eventContent:{padding:`0 1rem`}},eventMarker:{size:`1.125rem`,borderRadius:`50%`,borderWidth:`2px`,background:`{content.background}`,borderColor:`{content.border.color}`,content:{borderRadius:`50%`,size:`0.375rem`,background:`{primary.color}`,insetShadow:`0px 0.5px 0px 0px rgba(0, 0, 0, 0.06), 0px 1px 1px 0px rgba(0, 0, 0, 0.12)`}},eventConnector:{color:`{content.border.color}`,size:`2px`}},tm={root:{width:`25rem`,borderRadius:`{content.border.radius}`,borderWidth:`1px`,transitionDuration:`{transition.duration}`,blur:`10px`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},icon:{size:`1.125rem`,margin:`1px 0 0 0`},content:{padding:`{overlay.popover.padding}`,gap:`0.5rem`},text:{gap:`0.5rem`},summary:{fontWeight:`500`,fontSize:`1rem`},detail:{fontWeight:`500`,fontSize:`0.875rem`},closeButton:{width:`1.75rem`,height:`1.75rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,offset:`{focus.ring.offset}`}},closeIcon:{size:`1rem`},info:{background:`light-dark(color-mix(in srgb, {blue.50}, transparent 5%), color-mix(in srgb, {blue.500}, transparent 84%))`,borderColor:`light-dark({blue.200}, color-mix(in srgb, {blue.700}, transparent 64%))`,color:`light-dark({blue.600}, {blue.500})`,detailColor:`light-dark({surface.700}, {surface.0})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {blue.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({blue.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({blue.600}, {blue.500})`,shadow:`none`}}},success:{background:`light-dark(color-mix(in srgb, {green.50}, transparent 5%), color-mix(in srgb, {green.500}, transparent 84%))`,borderColor:`light-dark({green.200}, color-mix(in srgb, {green.700}, transparent 64%))`,color:`light-dark({green.600}, {green.500})`,detailColor:`light-dark({surface.700}, {surface.0})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {green.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({green.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({green.600}, {green.500})`,shadow:`none`}}},warn:{background:`light-dark(color-mix(in srgb,{yellow.50}, transparent 5%), color-mix(in srgb, {yellow.500}, transparent 84%))`,borderColor:`light-dark({yellow.200}, color-mix(in srgb, {yellow.700}, transparent 64%))`,color:`light-dark({yellow.600}, {yellow.500})`,detailColor:`light-dark({surface.700}, {surface.0})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {yellow.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({yellow.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({yellow.600}, {yellow.500})`,shadow:`none`}}},error:{background:`light-dark(color-mix(in srgb, {red.50}, transparent 5%), color-mix(in srgb, {red.500}, transparent 84%))`,borderColor:`light-dark({red.200}, color-mix(in srgb, {red.700}, transparent 64%))`,color:`light-dark({red.600}, {red.500})`,detailColor:`light-dark({surface.700}, {surface.0})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {red.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({red.100}, rgba(255, 255, 255, 0.05))`,focusRing:{color:`light-dark({red.600}, {red.500})`,shadow:`none`}}},secondary:{background:`light-dark({surface.100}, {surface.800})`,borderColor:`light-dark({surface.200}, {surface.700})`,color:`light-dark({surface.600}, {surface.300})`,detailColor:`light-dark({surface.700}, {surface.0})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.500}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({surface.200}, {surface.700})`,focusRing:{color:`light-dark({surface.600}, {surface.300})`,shadow:`none`}}},contrast:{background:`light-dark({surface.900}, {surface.0})`,borderColor:`light-dark({surface.950}, {surface.100})`,color:`light-dark({surface.50}, {surface.950})`,detailColor:`light-dark({surface.0}, {surface.950})`,shadow:`0px 4px 8px 0px color-mix(in srgb, {surface.950}, transparent 96%)`,closeButton:{hoverBackground:`light-dark({surface.800}, {surface.100})`,focusRing:{color:`light-dark({surface.50}, {surface.950})`,shadow:`none`}}},normal:{background:`{content.background}`,borderColor:`{content.border.color}`,color:`{text.color}`,detailColor:`{text.muted.color}`,shadow:`{overlay.popover.shadow}`,closeButton:{hoverBackground:`{content.hover.background}`,focusRing:{color:`{focus.ring.color}`,shadow:`none`}}}},nm={root:{padding:`0.25rem`,borderRadius:`{content.border.radius}`,gap:`0.5rem`,fontWeight:`500`,disabledBackground:`{form.field.disabled.background}`,disabledBorderColor:`{form.field.disabled.background}`,disabledColor:`{form.field.disabled.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,padding:`0.25rem`},lg:{fontSize:`{form.field.lg.font.size}`,padding:`0.25rem`},background:`light-dark({surface.100}, {surface.950})`,checkedBackground:`light-dark({surface.100}, {surface.950})`,hoverBackground:`light-dark({surface.100}, {surface.950})`,borderColor:`light-dark({surface.100}, {surface.950})`,color:`light-dark({surface.500}, {surface.400})`,hoverColor:`light-dark({surface.700}, {surface.300})`,checkedColor:`light-dark({surface.900}, {surface.0})`,checkedBorderColor:`light-dark({surface.100}, {surface.950})`,fontSize:`{form.field.font.size}`},icon:{disabledColor:`{form.field.disabled.color}`,color:`light-dark({surface.500}, {surface.400})`,hoverColor:`light-dark({surface.700}, {surface.300})`,checkedColor:`light-dark({surface.900}, {surface.0})`},content:{padding:`0.25rem 0.75rem`,borderRadius:`{content.border.radius}`,checkedShadow:`0px 1px 2px 0px rgba(0, 0, 0, 0.02), 0px 1px 2px 0px rgba(0, 0, 0, 0.04)`,sm:{padding:`0.25rem 0.75rem`},lg:{padding:`0.25rem 0.75rem`},checkedBackground:`light-dark({surface.0}, {surface.800})`}},rm={root:{width:`2.5rem`,height:`1.5rem`,borderRadius:`30px`,gap:`0.25rem`,shadow:`{form.field.shadow}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`},borderWidth:`1px`,borderColor:`transparent`,hoverBorderColor:`transparent`,checkedBorderColor:`transparent`,checkedHoverBorderColor:`transparent`,invalidBorderColor:`{form.field.invalid.border.color}`,transitionDuration:`{form.field.transition.duration}`,slideDuration:`0.2s`,background:`light-dark({surface.300}, {surface.700})`,disabledBackground:`light-dark({form.field.disabled.background}, {surface.600})`,hoverBackground:`light-dark({surface.400}, {surface.600})`,checkedBackground:`{primary.color}`,checkedHoverBackground:`{primary.hover.color}`},handle:{borderRadius:`50%`,size:`1rem`,background:`light-dark({surface.0}, {surface.400})`,disabledBackground:`light-dark({form.field.disabled.color}, {surface.900})`,hoverBackground:`light-dark({surface.0}, {surface.300})`,checkedBackground:`light-dark({surface.0}, {surface.900})`,checkedHoverBackground:`light-dark({surface.0}, {surface.900})`,color:`light-dark({text.muted.color}, {surface.900})`,hoverColor:`light-dark({text.color}, {surface.800})`,checkedColor:`{primary.color}`,checkedHoverColor:`{primary.hover.color}`}},im={root:{background:`{content.background}`,borderColor:`{content.border.color}`,borderRadius:`{content.border.radius}`,color:`{content.color}`,gap:`0.5rem`,padding:`0.75rem`}},am={root:{maxWidth:`12.5rem`,gutter:`0.25rem`,shadow:`{overlay.popover.shadow}`,padding:`0.5rem 0.75rem`,borderRadius:`{overlay.popover.border.radius}`,background:`{surface.700}`,color:`{surface.0}`,fontWeight:`{typography.font.weight}`,fontSize:`0.875rem`}},om={root:{background:`{content.background}`,color:`{content.color}`,padding:`1rem`,gap:`2px`,indent:`1rem`,transitionDuration:`{transition.duration}`},node:{padding:`0.25rem 0.5rem`,borderRadius:`{content.border.radius}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,color:`{text.color}`,hoverColor:`{text.hover.color}`,selectedColor:`{highlight.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`},gap:`0.25rem`},nodeIcon:{color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,selectedColor:`{highlight.color}`},nodeToggleButton:{borderRadius:`50%`,size:`1.75rem`,hoverBackground:`{content.hover.background}`,selectedHoverBackground:`{content.background}`,color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,selectedHoverColor:`{primary.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},loadingIcon:{size:`2rem`},filter:{margin:`0 0 0.5rem 0`},css:`\n"
    "    .p-tree-mask.p-overlay-mask {\n"
    "        --px-mask-background: light-dark(rgba(255,255,255,0.5),rgba(0,0,0,0.3));\n"
    "    }\n"
    "`,nodeLabel:{fontWeight:`{typography.font.weight}`,selectedFontWeight:`{list.option.selected.font.weight}`,fontSize:`{typography.font.size}`}},sm={root:{background:`{form.field.background}`,disabledBackground:`{form.field.disabled.background}`,filledBackground:`{form.field.filled.background}`,filledHoverBackground:`{form.field.filled.hover.background}`,filledFocusBackground:`{form.field.filled.focus.background}`,borderColor:`{form.field.border.color}`,hoverBorderColor:`{form.field.hover.border.color}`,focusBorderColor:`{form.field.focus.border.color}`,invalidBorderColor:`{form.field.invalid.border.color}`,color:`{form.field.color}`,disabledColor:`{form.field.disabled.color}`,placeholderColor:`{form.field.placeholder.color}`,invalidPlaceholderColor:`{form.field.invalid.placeholder.color}`,shadow:`{form.field.shadow}`,paddingX:`{form.field.padding.x}`,paddingY:`{form.field.padding.y}`,borderRadius:`{form.field.border.radius}`,focusRing:{width:`{form.field.focus.ring.width}`,style:`{form.field.focus.ring.style}`,color:`{form.field.focus.ring.color}`,offset:`{form.field.focus.ring.offset}`,shadow:`{form.field.focus.ring.shadow}`},transitionDuration:`{form.field.transition.duration}`,sm:{fontSize:`{form.field.sm.font.size}`,paddingX:`{form.field.sm.padding.x}`,paddingY:`{form.field.sm.padding.y}`},lg:{fontSize:`{form.field.lg.font.size}`,paddingX:`{form.field.lg.padding.x}`,paddingY:`{form.field.lg.padding.y}`},fontSize:`{form.field.font.size}`,fontWeight:`{form.field.font.weight}`},dropdown:{width:`2.5rem`,color:`{form.field.icon.color}`},overlay:{background:`{overlay.select.background}`,borderColor:`{overlay.select.border.color}`,borderRadius:`{overlay.select.border.radius}`,color:`{overlay.select.color}`,shadow:`{overlay.select.shadow}`},tree:{padding:`{list.padding}`},emptyMessage:{padding:`{list.option.padding}`},chip:{borderRadius:`{border.radius.sm}`},clearIcon:{color:`{form.field.icon.color}`}},cm={root:{transitionDuration:`{transition.duration}`,borderColor:`light-dark({content.border.color}, {surface.800})`},header:{background:`{content.background}`,borderColor:`{treetable.border.color}`,color:`{content.color}`,borderWidth:`0 0 1px 0`,padding:`0.75rem 1rem`},headerCell:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,borderColor:`{treetable.border.color}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,selectedColor:`{highlight.color}`,gap:`0.5rem`,padding:`0.75rem 1rem`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`}},columnTitle:{fontWeight:`600`,fontSize:`{typography.font.size}`},row:{background:`{content.background}`,hoverBackground:`{content.hover.background}`,selectedBackground:`{highlight.background}`,color:`{content.color}`,hoverColor:`{content.hover.color}`,selectedColor:`{highlight.color}`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`-1px`,shadow:`{focus.ring.shadow}`}},bodyCell:{borderColor:`{treetable.border.color}`,padding:`0.75rem 1rem`,gap:`0.5rem`,selectedBorderColor:`light-dark({primary.100}, {primary.900})`,fontWeight:`{typography.font.size}`,fontSize:`{typography.font.size}`},footerCell:{background:`{content.background}`,borderColor:`{treetable.border.color}`,color:`{content.color}`,padding:`0.75rem 1rem`},columnFooter:{fontWeight:`600`,fontSize:`{typography.font.size}`},footer:{background:`{content.background}`,borderColor:`{treetable.border.color}`,color:`{content.color}`,borderWidth:`0 0 1px 0`,padding:`0.75rem 1rem`},columnResizer:{width:`0.5rem`},resizeIndicator:{width:`1px`,color:`{primary.color}`},sortIcon:{color:`{text.muted.color}`,hoverColor:`{text.hover.muted.color}`,size:`0.875rem`},loadingIcon:{size:`2rem`},nodeToggleButton:{hoverBackground:`{content.hover.background}`,selectedHoverBackground:`{content.background}`,color:`{text.muted.color}`,hoverColor:`{text.color}`,selectedHoverColor:`{primary.color}`,size:`1.75rem`,borderRadius:`50%`,focusRing:{width:`{focus.ring.width}`,style:`{focus.ring.style}`,color:`{focus.ring.color}`,offset:`{focus.ring.offset}`,shadow:`{focus.ring.shadow}`}},paginatorTop:{borderColor:`{content.border.color}`,borderWidth:`0 0 1px 0`},paginatorBottom:{borderColor:`{content.border.color}`,borderWidth:`0 0 1px 0`},css:`\n"
    "    .p-treetable-mask.p-overlay-mask {\n"
    "        --px-mask-background: light-dark(rgba(255,255,255,0.5),rgba(0,0,0,0.3));\n"
    "    }\n"
    "`},lm={loader:{mask:{background:`{content.background}`,color:`{text.muted.color}`},icon:{size:`2rem`}}},um=Object.defineProperty,dm=Object.defineProperties,fm=Object.getOwnPropertyDescriptors,pm=Object.getOwnPropertySymbols,mm=Object.prototype.hasOwnProperty,hm=Object.prototype.propertyIsEnumerable,gm=(e,t,n)=>t in e?um(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,_m=dm(((e,t)=>{for(var n in t||={})mm.call(t,n)&&gm(e,n,t[n]);if(pm)for(var n of pm(t))hm.call(t,n)&&gm(e,n,t[n]);return e})({},Cf),fm({components:{accordion:yf,autocomplete:bf,avatar:xf,badge:Sf,blockui:wf,breadcrumb:Tf,button:Ef,card:Df,carousel:Of,cascadeselect:kf,checkbox:Af,chip:jf,colorpicker:Mf,commandmenu:Nf,compare:Pf,confirmdialog:Ff,confirmpopup:If,contextmenu:Lf,datatable:zf,dataview:Bf,datepicker:Vf,dialog:Hf,divider:Uf,dock:Wf,drawer:Gf,editor:Kf,fieldset:qf,fileupload:Jf,floatlabel:Yf,galleria:Xf,gallery:Zf,iconfield:Qf,iftalabel:$f,image:ep,imagecompare:tp,inlinemessage:np,inplace:rp,inputchips:ip,inputcolor:ap,inputgroup:op,inputnumber:sp,inputotp:cp,inputtags:lp,inputtext:up,knob:dp,label:fp,listbox:pp,megamenu:mp,menu:hp,menubar:gp,message:_p,metergroup:vp,multiselect:yp,navigationmenu:bp,orderlist:xp,organizationchart:Sp,overlaybadge:Cp,paginator:wp,panel:Tp,panelmenu:Ep,password:Dp,picklist:Op,popover:kp,progressbar:Ap,progressspinner:jp,radiobutton:Mp,rating:Np,ripple:Pp,scrollarea:Fp,scrollpanel:Ip,select:Lp,selectbutton:Rp,sidebar:zp,skeleton:Bp,slider:Vp,speeddial:Hp,splitbutton:Up,splitter:Wp,stepper:Gp,steps:Kp,tabmenu:qp,tabs:Jp,tabview:Yp,tag:Xp,terminal:Zp,textarea:Qp,tieredmenu:$p,timeline:em,toast:tm,togglebutton:nm,toggleswitch:rm,toolbar:im,tooltip:am,tree:om,treeselect:sm,treetable:cm,virtualscroller:lm},css:Rf})),vm={light:{text:`#262e30`,muted:`#5c5f61`,background:`#f0f0f0`,surface:`#ffffff`,raised:`#f4f4f4`,inset:`#eeeeee`,control:`#dfdfdf`,controlHover:`#d4d4d4`,border:`#dbdbdb`,controlBorder:`#7c8282`,primary:`#008172`,primaryHover:`#00675b`,onPrimary:`#ffffff`,primarySoft:`#bfe1de`,primaryText:`#00675b`,primaryMark:`#009688`,onState:`#ffffff`,danger:`#b42318`,dangerBg:`#fde4e1`,dangerSolid:`#c4281c`,dangerSolidHover:`#9c1f15`,onDangerSolid:`#ffffff`,warning:`#7a4f00`,warningBg:`#fdefc4`,success:`#0b6b3a`,successBg:`#d9f2e3`,info:`#1b5bb0`,infoBg:`#dbe9fb`,tool1:`#ff997d`,tool2:`#f4cf78`,tool3:`#a99bff`,tool4:`#78cfd5`,onTool:`#14181f`,onSolid:`#ffffff`,teal:`#00675b`,tealSoft:`#dbeae8`,onTeal:`#ffffff`,tealSolid:`#00675b`,sky:`#00628c`,skySoft:`#dbe9ef`,onSky:`#ffffff`,skySolid:`#00628c`,cobalt:`#2b55c5`,cobaltSoft:`#e1e7f7`,onCobalt:`#ffffff`,cobaltSolid:`#2b55c5`,violet:`#6240c8`,violetSoft:`#e9e4f7`,onViolet:`#ffffff`,violetSolid:`#6240c8`,magenta:`#a3247a`,magentaSoft:`#f2e0ec`,onMagenta:`#ffffff`,magentaSolid:`#a3247a`,lime:`#446600`,limeSoft:`#e5eadb`,onLime:`#ffffff`,limeSolid:`#446600`,orange:`#964700`,orangeSoft:`#f0e5db`,onOrange:`#ffffff`,orangeSolid:`#964700`},dark:{text:`#efeff0`,muted:`#b2b3b5`,background:`#242428`,surface:`#2d2d31`,raised:`#36363b`,inset:`#27272a`,control:`#3e3e45`,controlHover:`#4d4d54`,border:`#4a4a51`,controlBorder:`#818183`,primary:`#00675b`,primaryHover:`#008172`,onPrimary:`#ffffff`,primarySoft:`#223c3c`,primaryText:`#5eead4`,primaryMark:`#009688`,onState:`#10141a`,danger:`#ffaaa4`,dangerBg:`#4a2320`,dangerSolid:`#de3428`,dangerSolidHover:`#b42318`,onDangerSolid:`#ffffff`,warning:`#f3cc70`,warningBg:`#40321a`,success:`#7fdbab`,successBg:`#17392a`,info:`#9cc4ff`,infoBg:`#1c3050`,tool1:`#ff997d`,tool2:`#f4cf78`,tool3:`#a99bff`,tool4:`#78cfd5`,onTool:`#14181f`,onSolid:`#ffffff`,teal:`#79d8c7`,tealSoft:`#103d3c`,onTeal:`#1a2a2a`,tealSolid:`#00675b`,sky:`#72cdf4`,skySoft:`#103b51`,onSky:`#192931`,skySolid:`#00628c`,cobalt:`#9db8ff`,cobaltSoft:`#223569`,onCobalt:`#1f2632`,cobaltSolid:`#2b55c5`,violet:`#c0acff`,violetSoft:`#392c6a`,onViolet:`#242432`,violetSolid:`#6240c8`,magenta:`#f5a0d4`,magentaSoft:`#542149`,onMagenta:`#2c222c`,magentaSolid:`#a3247a`,lime:`#b9dc6c`,limeSoft:`#2c3c16`,onLime:`#232b1e`,limeSolid:`#446600`,orange:`#ffb27d`,orangeSoft:`#4f2f16`,onOrange:`#2d2520`,orangeSolid:`#964700`}},X=e=>`var(--oui-${e.replace(/[A-Z]/g,e=>`-`+e.toLowerCase())})`,ym=(e,t)=>`color-mix(in srgb, ${X(e)} ${t}%, transparent)`,bm={light:[`#ffffff`,`#f8f8f8`,`#f4f4f4`,`#eeeeee`,`#dbdbdb`,`#acacac`,`#7c8282`,`#5c5f61`,`#363636`,`#2c2c2e`,`#262e30`,`#1a1a1c`],dark:[`#ffffff`,`#efeff0`,`#e5e5e6`,`#b2b3b5`,`#909090`,`#818183`,`#65656a`,`#4a4a51`,`#3e3e45`,`#36363b`,`#2d2d31`,`#242428`]},xm=Object.fromEntries([0,50,100,200,300,400,500,600,700,800,900,950].map((e,t)=>[e,`light-dark(${bm.light[t]}, ${bm.dark[t]})`])),Sm=(e,t,n)=>({background:X(e),borderColor:X(e),color:X(t),hoverBackground:X(n),hoverBorderColor:X(n),hoverColor:X(t),activeBackground:X(n),activeBorderColor:X(n),activeColor:X(t),focusRing:{color:X(`primaryMark`),shadow:`none`}}),Cm=(e,t)=>({background:X(t),borderColor:ym(e,32),color:X(e),hoverBackground:X(t),hoverBorderColor:X(e),hoverColor:X(e),activeBackground:X(t),activeBorderColor:X(e),activeColor:X(e),focusRing:{color:X(`primaryMark`),shadow:`none`}}),wm={background:X(`control`),borderColor:X(`control`),color:X(`text`),hoverBackground:X(`controlHover`),hoverBorderColor:X(`controlHover`),hoverColor:X(`text`),activeBackground:X(`controlHover`),activeBorderColor:X(`controlHover`),activeColor:X(`text`),focusRing:{color:X(`primaryMark`),shadow:`none`}},Tm=(e,t)=>({background:X(t),borderColor:ym(e,32),color:X(`text`),shadow:`none`,closeButton:{hoverBackground:ym(e,16),focusRing:{color:X(`primaryMark`),shadow:`none`}},outlined:{color:X(e),borderColor:X(e)},simple:{color:X(e)}}),Em=`light-dark(0 12px 32px rgb(16 24 40 / .16), 0 12px 32px rgb(0 0 0 / .5))`,Dm=vf(_m,{semantic:{typography:{fontFamily:`inherit`,lineHeight:`inherit`,fontSize:`13px`},transitionDuration:`0.12s`,disabledOpacity:`0.45`,iconSize:`14px`,focusRing:{width:`2px`,style:`solid`,color:X(`primaryMark`),offset:`2px`,shadow:`none`},surface:xm,primary:{color:X(`primary`),contrastColor:X(`onPrimary`),hoverColor:X(`primaryHover`),activeColor:X(`primaryHover`)},highlight:{background:X(`primarySoft`),focusBackground:X(`primarySoft`),color:X(`text`),focusColor:X(`text`)},text:{color:X(`text`),hoverColor:X(`text`),mutedColor:X(`muted`),hoverMutedColor:X(`text`)},content:{borderRadius:`6px`,background:X(`surface`),hoverBackground:X(`raised`),borderColor:X(`border`),color:X(`text`),hoverColor:X(`text`)},formField:{paddingX:`10px`,paddingY:`6px`,borderRadius:`4px`,fontSize:`13px`,sm:{fontSize:`12px`,paddingX:`8px`,paddingY:`3px`},focusRing:{width:`2px`,style:`solid`,color:X(`primaryMark`),offset:`1px`,shadow:`none`},background:X(`surface`),disabledBackground:X(`raised`),filledBackground:X(`surface`),filledHoverBackground:X(`surface`),filledFocusBackground:X(`surface`),borderColor:X(`controlBorder`),hoverBorderColor:X(`primaryMark`),focusBorderColor:X(`primaryMark`),invalidBorderColor:X(`danger`),color:X(`text`),disabledColor:X(`muted`),placeholderColor:X(`muted`),invalidPlaceholderColor:X(`danger`),iconColor:X(`muted`),shadow:`none`},list:{padding:`4px`,gap:`1px`,option:{padding:`6px 8px`,borderRadius:`4px`,focusBackground:X(`control`),selectedBackground:X(`primarySoft`),selectedFocusBackground:X(`primarySoft`),color:X(`text`),focusColor:X(`text`),selectedColor:X(`text`),selectedFocusColor:X(`text`)}},overlay:{select:{borderRadius:`6px`,shadow:Em,background:X(`surface`),borderColor:X(`border`),color:X(`text`)},popover:{borderRadius:`6px`,padding:`12px`,shadow:Em,background:X(`surface`),borderColor:X(`border`),color:X(`text`)},modal:{borderRadius:`8px`,padding:`20px`,shadow:Em,background:X(`surface`),borderColor:X(`border`),color:X(`text`)}},mask:{background:`light-dark(rgb(16 24 40 / .36), rgb(0 0 0 / .6))`,color:X(`text`)}},components:{button:{root:{borderRadius:`4px`,gap:`6px`,paddingX:`12px`,paddingY:`6px`,iconOnlyWidth:`32px`,fontSize:`13px`,sm:{fontSize:`12px`,paddingX:`8px`,paddingY:`3px`,iconOnlyWidth:`26px`},label:{fontWeight:`600`},primary:Sm(`primary`,`onPrimary`,`primaryHover`),danger:Sm(`dangerSolid`,`onDangerSolid`,`dangerSolidHover`),secondary:wm,info:Cm(`info`,`infoBg`),success:Cm(`success`,`successBg`),warn:Cm(`warning`,`warningBg`)},outlined:{primary:{color:X(`primaryText`),borderColor:ym(`primaryText`,56),hoverBackground:X(`primarySoft`),activeBackground:X(`primarySoft`)},danger:{color:X(`danger`),borderColor:ym(`danger`,56),hoverBackground:X(`dangerBg`),activeBackground:X(`dangerBg`)},secondary:{color:X(`text`),borderColor:X(`controlBorder`),hoverBackground:X(`raised`),activeBackground:X(`raised`)}},text:{primary:{color:X(`primaryText`),hoverBackground:X(`primarySoft`),activeBackground:X(`primarySoft`)},secondary:{color:X(`text`),hoverBackground:X(`control`),activeBackground:X(`controlHover`)},danger:{color:X(`danger`),hoverBackground:X(`dangerBg`),activeBackground:X(`dangerBg`)}}},message:{root:{borderWidth:`1px`,borderRadius:`4px`},content:{padding:`6px 10px`,gap:`8px`},text:{fontSize:`13px`,fontWeight:`500`},icon:{size:`14px`},error:Tm(`danger`,`dangerBg`),warn:Tm(`warning`,`warningBg`),success:Tm(`success`,`successBg`),info:Tm(`info`,`infoBg`),secondary:Tm(`muted`,`raised`)},dialog:{root:{borderRadius:`8px`},header:{padding:`16px 20px 12px`},title:{fontSize:`15px`,fontWeight:`600`},content:{padding:`0 20px 20px`},footer:{padding:`0 20px 16px`,gap:`8px`}},datatable:{root:{borderColor:X(`border`)},headerCell:{background:X(`raised`),hoverBackground:X(`raised`),selectedBackground:X(`raised`),color:X(`muted`),hoverColor:X(`text`),selectedColor:X(`text`),padding:`8px 12px`,sm:{padding:`6px 12px`},gap:`4px`},columnTitle:{fontWeight:`600`,fontSize:`11px`},row:{background:`transparent`,hoverBackground:X(`raised`),selectedBackground:X(`primarySoft`),selectedColor:X(`text`)},bodyCell:{padding:`8px 12px`,sm:{padding:`5px 12px`},selectedBorderColor:X(`border`),fontSize:`13px`,fontWeight:`inherit`},sortIcon:{size:`12px`}},tabs:{tablist:{background:`transparent`,borderWidth:`0`},tab:{padding:`0 12px`,fontWeight:`600`,borderWidth:`0`,margin:`0`,gap:`6px`,color:X(`muted`),hoverColor:X(`text`),activeColor:X(`text`)},tabpanel:{background:`transparent`,padding:`0`},activeBar:{height:`2px`,background:X(`primaryMark`)},navButton:{background:X(`surface`),shadow:`none`}},progressbar:{root:{background:X(`border`),borderRadius:`4px`,height:`8px`},value:{background:X(`primaryMark`)}},toggleswitch:{root:{width:`32px`,height:`18px`,background:X(`controlBorder`),hoverBackground:X(`muted`),shadow:`none`,checkedBackground:X(`primaryMark`),checkedHoverBackground:X(`primaryMark`)},handle:{size:`12px`,background:X(`surface`),hoverBackground:X(`surface`),checkedBackground:X(`onPrimary`),checkedHoverBackground:X(`onPrimary`)}},checkbox:{root:{width:`16px`,height:`16px`,borderRadius:`4px`,checkedBackground:X(`primaryMark`),checkedHoverBackground:X(`primaryMark`),checkedBorderColor:X(`primaryMark`),checkedHoverBorderColor:X(`primaryMark`),checkedFocusBorderColor:X(`primaryMark`)},icon:{size:`11px`}},togglebutton:{root:{padding:`2px`,borderRadius:`4px`,fontWeight:`600`,background:X(`inset`),checkedBackground:X(`inset`),hoverBackground:X(`inset`),borderColor:X(`border`),checkedBorderColor:X(`border`),color:X(`muted`),hoverColor:X(`text`),checkedColor:X(`onPrimary`)},content:{padding:`3px 10px`,borderRadius:`3px`,checkedBackground:X(`primary`),checkedShadow:`none`}},select:{dropdown:{width:`28px`}},inputnumber:{button:{width:`26px`}},tooltip:{root:{background:X(`text`),color:X(`background`),fontSize:`12px`,padding:`4px 8px`}}}}),Om=`eyJpZCI6IjRmYWE0NzViLTk3NjMtNDFlMi1iODgzLTJkZTdjMTI4M2QzNCIsInByb2R1Y3QiOiJwcmltZXVpIiwidGllciI6ImNvbW11bml0eSIsInR5cGUiOiJkZXYiLCJpYXQiOjE3OTQ2MTUxNzAsImV4cCI6MTgyNjE1MTE3MH0.mUdvhWuQBDmiTBEqJ2aj6VVPs8-_C7INSX-YoF1TQ-3Zhq9xNvHXp95Taru5A0qbtPg6b4Gh0O9l8KRwBEdlCg`;function km(e){return{ripple:!1,license:Om,theme:{preset:Dm,options:{darkModeSelector:`.orca-dark`,cssLayer:!1}},...e===`pt-BR`?{locale:{accept:`Confirmar`,reject:`Voltar`,choose:`Selecionar`,upload:`Enviar`,cancel:`Cancelar`,emptyMessage:`Nenhum item encontrado`,emptyFilterMessage:`Nenhum resultado para estes filtros`,emptySearchMessage:`Nenhum resultado encontrado`,searchMessage:`{0} resultados encontrados`,selectionMessage:`{0} itens selecionados`,emptySelectionMessage:`Nenhum item selecionado`,fileSizeTypes:[`B`,`KB`,`MB`,`GB`,`TB`,`PB`,`EB`,`ZB`,`YB`],dayNames:[`domingo`,`segunda-feira`,`terça-feira`,`quarta-feira`,`quinta-feira`,`sexta-feira`,`sábado`],dayNamesShort:[`dom`,`seg`,`ter`,`qua`,`qui`,`sex`,`sáb`],dayNamesMin:[`D`,`S`,`T`,`Q`,`Q`,`S`,`S`],monthNames:[`janeiro`,`fevereiro`,`março`,`abril`,`maio`,`junho`,`julho`,`agosto`,`setembro`,`outubro`,`novembro`,`dezembro`],monthNamesShort:[`jan`,`fev`,`mar`,`abr`,`mai`,`jun`,`jul`,`ago`,`set`,`out`,`nov`,`dez`],dateFormat:`dd/mm/yy`,firstDayOfWeek:0,aria:{close:`Fechar`,previous:`Anterior`,next:`Próximo`,firstPageLabel:`Primeira página`,lastPageLabel:`Última página`,nextPageLabel:`Próxima página`,prevPageLabel:`Página anterior`,selectRow:`Selecionar linha`,unselectRow:`Desmarcar linha`}}}:{}}}var Am={mounted(){if(this.$options.name!==`TabList`||`tabs`in this.$refs)return;let e=this.$refs;Object.defineProperty(e,\"tabs\",{configurable:!0,enumerable:!1,get:()=>e.content})}};function jm(e,t,n){n&&Nm(n),e.use(_f,km(t)),e.mixin(Am)}var Mm=`teal`;function Nm(e,t=document.documentElement){Mm=e;let n=t.dataset.orcaUiTheme;(n===`light`||n===`dark`)&&Pm(n,t)}function Pm(e,t=document.documentElement){t.classList.toggle(`orca-dark`,e===`dark`),t.dataset.orcaUiTheme=e,t.dataset.orcaUiAccent=Mm,t.style.colorScheme=e;let n=vm[e],r={...n,accent:n[Mm],accentSoft:n[Mm+`Soft`],accentSolid:n[Mm+`Solid`],onAccent:n[`on`+Mm[0].toUpperCase()+Mm.slice(1)]};for(let[e,n]of Object.entries(r))t.style.setProperty(`--oui-${e.replace(/[A-Z]/g,e=>`-`+e.toLowerCase())}`,n)}var Fm=[`data-tone`],Im=[`width`,`height`],Lm=[`data-tone`,`width`,`height`],Rm=P({__name:`OuiIcon`,props:{icon:{},size:{},tone:{},tile:{type:Boolean},soft:{type:Boolean}},setup(e){return(t,n)=>e.tile?(L(),R(`span`,{key:0,class:`oui-tile`,\"data-tone\":e.tone??`neutral`,style:de({width:(e.size??16)+12+`px`,height:(e.size??16)+12+`px`}),\"aria-hidden\":`true`},[(L(),R(`svg`,{class:`oui-icon`,width:e.size??16,height:e.size??16,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,\"stroke-width\":`2`,\"stroke-linecap\":`round`,\"stroke-linejoin\":`round`,focusable:`false`},[(L(!0),R(I,null,Wr(e.icon,([e,t],n)=>(L(),z(Br(e),U({key:n},{ref_for:!0},t),null,16))),128))],8,Im))],12,Fm)):(L(),R(`svg`,{key:1,class:`oui-icon`,\"data-tone\":e.tone,width:e.size??16,height:e.size??16,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,\"stroke-width\":`2`,\"stroke-linecap\":`round`,\"stroke-linejoin\":`round`,\"aria-hidden\":`true`,focusable:`false`},[(L(!0),R(I,null,Wr(e.icon,([e,t],n)=>(L(),z(Br(e),U({key:n},{ref_for:!0},t),null,16))),128))],8,Lm))}}),zm={class:`oui-appbar`},Bm={class:`oui-appbar-row`},Vm={class:`oui-appbar-id`},Hm={key:0,class:`oui-appbar-mark`,\"aria-hidden\":`true`},Um={class:`oui-appbar-name`},Wm=[`aria-label`],Gm=[`data-tab`,`id`,`aria-selected`,`tabindex`,`onClick`],Km={key:1,class:`oui-appbar-count oui-num`},qm=[`title`],Jm={class:`oui-sr`},Ym={key:1,class:`oui-appbar-status`},Xm={key:2,class:`oui-appbar-actions`},Zm=P({__name:`OuiAppBar`,props:{name:{},tabs:{},modelValue:{},tabsLabel:{}},emits:[`update:modelValue`],setup(e,{emit:t}){let n=e,r=t;function i(e){let t=n.tabs??[],i=t.findIndex(e=>e.value===n.modelValue),a={ArrowRight:i+1,ArrowLeft:i-1,Home:0,End:t.length-1}[e.key];if(a==null||!t.length)return;e.preventDefault();let o=t[(a+t.length)%t.length];r(`update:modelValue`,o.value);let s=e.currentTarget;requestAnimationFrame(()=>s.querySelector(`[data-tab=\"${CSS.escape(o.value)}\"]`)?.focus())}return(t,n)=>(L(),R(`header`,zm,[B(`div`,Bm,[B(`div`,Vm,[t.$slots.mark?(L(),R(`span`,Hm,[F(t.$slots,`mark`)])):H(``,!0),B(`h1`,Um,k(e.name),1)]),e.tabs?.length?(L(),R(`nav`,{key:0,class:`oui-appbar-tabs`,role:`tablist`,\"aria-label\":e.tabsLabel??e.name,onKeydown:i},[(L(!0),R(I,null,Wr(e.tabs,t=>(L(),R(`button`,U({key:t.value},{ref_for:!0},t.attrs,{type:`button`,role:`tab`,class:`oui-appbar-tab`,\"data-tab\":t.value,id:`oui-tab-`+t.value,\"aria-selected\":t.value===e.modelValue,tabindex:t.value===e.modelValue?0:-1,onClick:e=>r(`update:modelValue`,t.value)}),[t.icon?(L(),z(Rm,{key:0,icon:t.icon,size:15},null,8,[`icon`])):H(``,!0),B(`span`,null,k(t.label),1),t.count==null?H(``,!0):(L(),R(`span`,Km,[n[0]||=B(`span`,{class:`oui-sr`},k(` `),-1),Ta(k(t.count),1)])),t.attention?(L(),R(`span`,{key:2,class:`oui-appbar-attention`,title:t.attention},[B(`span`,Jm,`, `+k(t.attention),1)],8,qm)):H(``,!0)],16,Gm))),128))],40,Wm)):H(``,!0),n[1]||=B(`span`,{class:`oui-spacer`},null,-1),t.$slots.status?(L(),R(`div`,Ym,[F(t.$slots,`status`)])):H(``,!0),t.$slots.actions?(L(),R(`div`,Xm,[F(t.$slots,`actions`)])):H(``,!0)])]))}}),Qm=[[`rect`,{x:3,y:3,width:18,height:18,rx:2}],[`path`,{d:`M3 12h9v9M12 3v9h9`}]],$m=[[`path`,{d:`M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z`}],[`path`,{d:`m3.3 7 8.7 5 8.7-5`}],[`path`,{d:`M12 22V12`}]],eh=[[`path`,{d:`M20 6 9 17l-5-5`}]],th=[[`path`,{d:`m9 18 6-6-6-6`}]],nh=[[`circle`,{cx:`12`,cy:`12`,r:`10`}],[`path`,{d:`M12 6v6l4 2`}]],rh=[[`rect`,{width:`14`,height:`14`,x:`8`,y:`8`,rx:`2`,ry:`2`}],[`path`,{d:`M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2`}]],ih=[[`circle`,{cx:`12`,cy:`12`,r:`10`}],[`path`,{d:`m15 9-6 6`}],[`path`,{d:`m9 9 6 6`}]],ah=[[`path`,{d:`M15 3h6v6`}],[`path`,{d:`M10 14 21 3`}],[`path`,{d:`M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6`}]],oh=[[`path`,{d:`M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z`}],[`path`,{d:`M14 2v5a1 1 0 0 0 1 1h5`}],[`path`,{d:`M10 9H8`}],[`path`,{d:`M16 13H8`}],[`path`,{d:`M16 17H8`}]],sh=[[`rect`,{width:`18`,height:`18`,x:`3`,y:`3`,rx:`2`}],[`path`,{d:`M3 9h18`}],[`path`,{d:`M3 15h18`}],[`path`,{d:`M9 3v18`}],[`path`,{d:`M15 3v18`}]],ch=[[`circle`,{cx:`12`,cy:`12`,r:`10`}],[`path`,{d:`M12 16v-4`}],[`path`,{d:`M12 8h.01`}]],lh=[[`path`,{d:`M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71`}],[`path`,{d:`M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71`}]],uh=[[`path`,{d:`M12 20h.01`}],[`path`,{d:`M8.5 16.429a5 5 0 0 1 7 0`}],[`path`,{d:`M5 12.859a10 10 0 0 1 5.17-2.69`}],[`path`,{d:`M19 12.859a10 10 0 0 0-2.007-1.523`}],[`path`,{d:`M2 8.82a15 15 0 0 1 4.177-2.643`}],[`path`,{d:`M22 8.82a15 15 0 0 0-11.288-3.764`}],[`path`,{d:`m2 2 20 20`}]],dh=[[`path`,{d:`M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8`}],[`path`,{d:`M21 3v5h-5`}],[`path`,{d:`M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16`}],[`path`,{d:`M8 16H3v5`}]],fh=[[`path`,{d:`m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3`}],[`path`,{d:`M12 9v4`}],[`path`,{d:`M12 17h.01`}]],ph=[[`path`,{d:`M18 6 6 18`}],[`path`,{d:`m6 6 12 12`}]],mh=[`id`,`data-dismissible`,`data-tone`,`role`],hh={class:`oui-banner-text`},gh={key:0},_h={key:0,class:`oui-banner-actions`},vh=[`aria-label`,`title`],yh=P({__name:`OuiBanner`,props:{tone:{},title:{},icon:{},id:{},dismissible:{type:Boolean},dismissKey:{},closeLabel:{},show:{type:Boolean,default:!0}},emits:[`dismiss`],setup(e,{emit:t}){let n=e,r=t,i={info:`info`,warn:`warn`,error:`danger`,success:`success`,neutral:`neutral`},a={info:ch,warn:fh,error:ih,success:eh,neutral:ch},o=e=>`oui-banner:`+e;function s(e){if(!e)return!1;try{return sessionStorage.getItem(o(e))===`1`}catch{return!1}}function c(e){if(e)try{sessionStorage.removeItem(o(e))}catch{}}n.show||c(n.dismissKey);let l=$t(n.show&&s(n.dismissKey));Gn(()=>n.dismissKey,(e,t)=>{t&&t!==e&&c(t),l.value=s(e)}),Gn(()=>n.show,e=>{e||(c(n.dismissKey),l.value=!1)});function u(){if(l.value=!0,n.dismissKey)try{sessionStorage.setItem(o(n.dismissKey),`1`)}catch{}r(`dismiss`)}return(t,r)=>e.show&&!l.value?(L(),R(`div`,{key:0,id:e.id,class:`oui-banner`,\"data-dismissible\":e.dismissible||void 0,\"data-tone\":i[n.tone??`info`],role:n.tone===`error`?`alert`:`status`},[V(Rm,{class:`oui-banner-icon`,icon:e.icon??a[n.tone??`info`],size:16,tone:i[n.tone??`info`]},null,8,[`icon`,`tone`]),B(`p`,hh,[B(`strong`,null,k(e.title),1),r[0]||=Ta(),t.$slots.default?(L(),R(`span`,gh,[F(t.$slots,`default`)])):H(``,!0)]),t.$slots.actions?(L(),R(`div`,_h,[F(t.$slots,`actions`)])):H(``,!0),e.dismissible?(L(),R(`button`,{key:1,type:`button`,class:`oui-banner-close`,\"aria-label\":e.closeLabel??`Close`,title:e.closeLabel??`Close`,onClick:u},[V(Rm,{icon:M(ph),size:14},null,8,[`icon`])],8,vh)):H(``,!0)],8,mh)):H(``,!0)}}),bh=[`data-state`,`data-compact`,`role`,`aria-busy`],xh=[`src`],Sh={key:1,class:`oui-empty-mark`},Ch={class:`oui-empty-text`},wh={class:`oui-strong`},Th={key:0,class:`oui-meta`},Eh={key:3,class:`oui-row`},Dh=P({__name:`OuiEmpty`,props:{title:{},text:{},icon:{},tone:{},image:{},state:{},compact:{type:Boolean}},setup(e){let t={empty:$m,error:ih,offline:uh};return(n,r)=>(L(),R(`div`,{class:`oui-empty`,\"data-state\":e.state??`empty`,\"data-compact\":e.compact||void 0,role:e.state===`error`?`alert`:void 0,\"aria-busy\":e.state===`loading`||void 0},[e.image&&e.state!==`loading`?(L(),R(`img`,{key:0,class:`oui-empty-image`,src:e.image,alt:``,width:`144`,height:`144`,draggable:`false`},null,8,xh)):e.state===`loading`?(L(),R(`span`,Sh,[...r[0]||=[B(`span`,{class:`oui-empty-spin`},null,-1)]])):(L(),z(Rm,{key:2,tile:``,tone:e.state===`error`?`danger`:e.state===`offline`?`neutral`:e.tone??`neutral`,icon:e.icon??t[e.state??`empty`],size:e.compact?18:22},null,8,[`tone`,`icon`,`size`])),B(`div`,Ch,[B(`p`,wh,k(e.title),1),e.text||n.$slots.default?(L(),R(`p`,Th,[F(n.$slots,`default`,{},()=>[Ta(k(e.text),1)])])):H(``,!0)]),n.$slots.actions?(L(),R(`div`,Eh,[F(n.$slots,`actions`)])):H(``,!0)],8,bh))}}),Oh={name:`spinner`,meta:{tags:[`spinner`,`loading`,`process`,`wait`,`buffering`]},svg:{xmlns:`http://www.w3.org/2000/svg`,width:20,height:20,viewBox:`0 0 20 20`,fill:`none`},nodes:[[`path`,{d:`M1 10C1 5.02579 5.02579 1 10 1C12.3905 1 14.562 1.9393 16.1738 3.45312C16.4756 3.73669 16.4905 4.21178 16.207 4.51367C15.9235 4.81558 15.4484 4.83039 15.1465 4.54688C13.7983 3.2807 11.9895 2.5 10 2.5C5.85421 2.5 2.5 5.85421 2.5 10C2.5 14.1458 5.85421 17.5 10 17.5C14.1458 17.5 17.5 14.1458 17.5 10C17.5 9.58579 17.8358 9.25 18.25 9.25C18.6642 9.25 19 9.58579 19 10C19 14.9742 14.9742 19 10 19C5.02579 19 1 14.9742 1 10Z`,fill:`currentColor`,key:`p4wko0`}]]};function kh(e){if(e!=null&&e!==``)return typeof e==`number`||/^\\d+(\\.\\d+)?$/.test(e)?`${e}px`:`${e}`}var Ah=([e,t])=>{let{key:n,...r}=t,i={};for(let[e,t]of Object.entries(r))i[Cl(e)]=t;return Qa(e,{key:n,...i})},jh=e=>{let t={size:{type:[Number,String],default:void 0},color:{type:String,default:void 0},spin:{type:Boolean,default:!1}};return{Icon:P({name:e.name.split(`-`).map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(``),props:t,setup(t,{attrs:n}){let r=W(()=>t.size??20),i=W(()=>({...kh(t.size)&&{\"--px-icon-size\":kh(t.size)},...t.color&&{color:t.color}})),a=W(()=>[`p-icon`,`p-icon-${e.name}`,t.spin&&`p-icon-spin`].filter(Boolean));return()=>Qa(`svg`,{...e.svg,width:r.value,height:r.value,\"aria-hidden\":`true`,...n,style:i.value,class:a.value},e.nodes.map(Ah))}}),props:t}},Mh=P({name:`Spinner`,inheritAttrs:!1,__name:`spinner`,setup(e){let{Icon:t}=jh(Oh);return(e,n)=>(L(),z(M(t),_e(Ca(e.$attrs)),null,16))}});function Nh(){return`${arguments.length>0&&arguments[0]!==void 0?arguments[0]:`pc`}${fr().replace(`v-`,``).replaceAll(`-`,`_`)}`}function Ph(e,t){return!t||t===`p`||!e?e:Al({prefix:t,keep:Ju},e)}function Fh(e,t){return!t||t===`p`||!e?e:jl({prefix:t,keep:Ju},e)}var Ih=!1,Lh=function(){try{return!1}catch{return!0}};function Rh(e){var t,n=e==null||(t=e.getOptions)==null||(t=t.call(e))==null?void 0:t.classPrefix;if(!n||n===`p`||e.styleName)return n;!Ih&&Lh()&&(Ih=!0,console.warn(`[PrimeVue] The theme option classPrefix \"${n}\" needs a newer @primeuix/styled; class names keep the \"p-\" prefix.`))}function zh(e){return e?.theme||J}function Bh(e){return zh(e).service??ku}var Vh=new WeakMap;function Hh(e){if(e){var t=Vh.get(e);return t||Vh.set(e,t=W(function(){var t;return(t=e.config)!=null&&t.theme?Rh(zh(e.isolation)):void 0})),t.value}}var Uh=Y.extend({name:`common`});function Wh(e){\"@babel/helpers - typeof\";return Wh=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},Wh(e)}function Gh(e){return Qh(e)||Kh(e)||Yh(e)||Jh()}function Kh(e){if(typeof Symbol<`u`&&e[Symbol.iterator]!=null||e[`@@iterator`]!=null)return Array.from(e)}function qh(e,t){return Qh(e)||Zh(e,t)||Yh(e,t)||Jh()}function Jh(){throw TypeError(`Invalid attempt to destructure non-iterable instance.\n"
    "In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function Yh(e,t){if(e){if(typeof e==`string`)return Xh(e,t);var n={}.toString.call(e).slice(8,-1);return n===`Object`&&e.constructor&&(n=e.constructor.name),n===`Map`||n===`Set`?Array.from(e):n===`Arguments`||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?Xh(e,t):void 0}}function Xh(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function Zh(e,t){var n=e==null?null:typeof Symbol<`u`&&e[Symbol.iterator]||e[`@@iterator`];if(n!=null){var r,i,a,o,s=[],c=!0,l=!1;try{if(a=(n=n.call(e)).next,t===0){if(Object(n)!==n)return;c=!1}else for(;!(c=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);c=!0);}catch(e){l=!0,i=e}finally{try{if(!c&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(l)throw i}}return s}}function Qh(e){if(Array.isArray(e))return e}function $h(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function Z(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?$h(Object(n),!0).forEach(function(t){eg(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):$h(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function eg(e,t,n){return(t=tg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function tg(e){var t=ng(e,`string`);return Wh(t)==`symbol`?t:t+``}function ng(e,t){if(Wh(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(Wh(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var rg={name:`BaseComponent`,props:{pt:{type:Object,default:void 0},ptOptions:{type:Object,default:void 0},unstyled:{type:Boolean,default:void 0},dt:{type:Object,default:void 0}},inject:{$parentInstance:{default:void 0}},watch:{isUnstyled:{immediate:!0,handler:function(e){Bh(this.$isolation).off(`theme:change`,this._loadCoreStyles),e||(this._loadCoreStyles(),this._themeChangeListener(this._loadCoreStyles))}},dt:{immediate:!0,handler:function(e){var t=this;Bh(this.$isolation).off(`theme:change`,this._themeScopedListener),e?(this._loadScopedThemeStyles(e),this._themeScopedListener=function(){return t._loadScopedThemeStyles(e)},this._themeChangeListener(this._themeScopedListener)):this._unloadScopedThemeStyles()}}},scopedStyleEl:void 0,uid:void 0,$attrSelector:void 0,$isolation:void 0,$styleHost:void 0,beforeCreate:function(){var e,t,n,r,i,a,o,s,c,l,u=this.pt?._usept,d=u?(e=this.pt)==null||(e=e.originalValue)==null?void 0:e[this.$.type.name]:void 0;(n=(u?(t=this.pt)==null||(t=t.value)==null?void 0:t[this.$.type.name]:this.pt)||d)==null||(n=n.hooks)==null||(r=n.onBeforeCreate)==null||r.call(n);var f=(i=this.$primevueConfig)==null||(i=i.pt)==null?void 0:i._usept,p=f?(a=this.$primevue)==null||(a=a.config)==null||(a=a.pt)==null?void 0:a.originalValue:void 0;(c=(f?(o=this.$primevue)==null||(o=o.config)==null||(o=o.pt)==null?void 0:o.value:(s=this.$primevue)==null||(s=s.config)==null?void 0:s.pt)||p)==null||(c=c[this.$.type.name])==null||(c=c.hooks)==null||(l=c.onBeforeCreate)==null||l.call(c),this.$isolation=this.$primevue?.isolation,this.$styleHost=this.$isolation||this.$primevue?.shadowRoot,this.$attrSelector=Nh(),this.uid=this.$attrs.id||this.$attrSelector.replace(`pc`,`pv_id_`)},created:function(){this._hook(`onCreated`)},beforeMount:function(){this._loadStyles(),this._hook(`onBeforeMount`)},mounted:function(){var e;(e=this.$styleHost)==null||e.attach(this),this._hook(`onMounted`),this.$primevue&&this.$primevue.verified?.value!==!1||ef()},beforeUpdate:function(){this._hook(`onBeforeUpdate`)},updated:function(){this._hook(`onUpdated`)},beforeUnmount:function(){this._hook(`onBeforeUnmount`)},unmounted:function(){var e;this._removeThemeListeners(),this._unloadScopedThemeStyles(),(e=this.$styleHost)==null||e.detach(this),this._hook(`onUnmounted`)},methods:{_hook:function(e){if(!this.$options.hostName){var t=this._usePT(this._getPT(this.pt,this.$.type.name),this._getOptionValue,`hooks.${e}`),n=this._useDefaultPT(this._getOptionValue,`hooks.${e}`);t?.(),n?.()}},_mergeProps:function(e){var t=[...arguments].slice(1);return rl(e)?e.apply(void 0,t):U.apply(void 0,t)},_load:function(){var e=this,t=function(){xd(e.$isolation?.base||yd,`base`,function(){return[Y.loadCSS(e.$styleOptions),e._loadGlobalStyles()]}),e.$preflight&&xd(e.$isolation?.base||yd,`preflight`,function(){return Y.loadPreflight(e.$styleOptions)}),e._loadThemeStyles()};this.$isolation?this.$isolation.scope.batch(t):t()},_loadStyles:function(){this._load(),this._themeChangeListener(this._load)},_loadCoreStyles:function(){var e=this,t,n=function(){return xd(e.$isolation?.base||yd,e.$style.name,function(){return[Uh.loadCSS(e.$styleOptions),e.$options.style&&e.$style.loadCSS(e.$styleOptions)]})};(t=this.$style)!=null&&t.name&&(this.$isolation?this.$isolation.scope.batch(n):n())},_loadGlobalStyles:function(){var e=this._useGlobalPT(this._getOptionValue,`global.css`,this.$params);return q(e)?Y.load(e,Z({name:`global`},this.$styleOptions)):void 0},_loadThemeStyles:function(){if(!this.isUnstyled){var e=zh(this.$isolation);if(this.$theme!==`none`&&this._loadThemeSheets(e),!e.isStyleNameLoaded(`layer-order`)){var t,n,r=ed(this.$preflight?$u:void 0,(t=this.$style)==null||(n=t.getLayerOrderThemeCSS)==null?void 0:n.call(t,e));Y.load(r,Z({name:`layer-order`,first:!0,raw:!0},this.$styleOptions)),e.setLoadedStyleName(`layer-order`)}}},_loadThemeSheets:function(e){var t;if(!e.isStyleNameLoaded(`common`)){var n,r,i=((n=this.$style)==null||(r=n.getCommonTheme)==null?void 0:r.call(n,void 0,e))||{},a=i.primitive,o=i.semantic,s=i.global,c=i.style;Y.load(a?.css,Z({name:`primitive-variables`,raw:!0},this.$styleOptions)),Y.load(o?.css,Z({name:`semantic-variables`,raw:!0},this.$styleOptions)),Y.load(s?.css,Z({name:`global-variables`,raw:!0},this.$styleOptions)),Y.loadStyle(Z({name:`global-style`},this.$styleOptions),c),e.setLoadedStyleName(`common`)}if(!e.isStyleNameLoaded(this.$style?.name)&&(t=this.$style)!=null&&t.name){var l,u,d,f,p=((l=this.$style)==null||(u=l.getComponentTheme)==null?void 0:u.call(l,void 0,e))||{},m=p.css,h=p.style;(d=this.$style)==null||d.load(m,Z({name:`${this.$style.name}-variables`,raw:!0},this.$styleOptions)),(f=this.$style)==null||f.loadStyle(Z({name:`${this.$style.name}-style`},this.$styleOptions),h),e.setLoadedStyleName(this.$style.name)}},_loadScopedThemeStyles:function(e){var t,n,r,i,a=zh(this.$isolation);((t=this.$theme)==null||(t=t.options)==null?void 0:t.cssVariables)===!1&&(n=this.$style)!=null&&n.name&&a.addScopedToken(eg({},this.$style.name,e))&&(a.deleteLoadedStyleName(this.$style.name),this._loadThemeStyles());var o=(((r=this.$style)==null||(i=r.getPresetTheme)==null?void 0:i.call(r,e,this.$instanceSelector,void 0,a))||{}).css,s=`${this.$attrSelector}-${this.$style.name}`,c=this.$style?.load(o,Z({name:s,raw:!0},this.$styleOptions));this.scopedStyleEl=c?.el,this._scopedStyleName=s},_unloadScopedThemeStyles:function(){var e,t;(e=this.scopedStyleEl)==null||(e=e.value)==null||e.remove(),this._scopedStyleName&&((t=this.$styleHost)==null||t.drop(this._scopedStyleName))},_themeChangeListener:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(){};Bh(this.$isolation).on(`theme:change`,e)},_removeThemeListeners:function(){var e=Bh(this.$isolation);e.off(`theme:change`,this._loadCoreStyles),e.off(`theme:change`,this._load),e.off(`theme:change`,this._themeScopedListener)},_getHostInstance:function(e){return e?this.$options.hostName?e.$.type.name===this.$options.hostName?e:this._getHostInstance(e.$parentInstance):e.$parentInstance:void 0},_getPropValue:function(e){return this[e]||this._getHostInstance(this)?.[e]},_getOptionValue:function(e){return pl(e,arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,arguments.length>2&&arguments[2]!==void 0?arguments[2]:{})},_getPTValue:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,n=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{},r=arguments.length>3&&arguments[3]!==void 0?arguments[3]:!0,i=/./g.test(t)&&!!n[t.split(`.`)[0]],a=this._getPropValue(`ptOptions`)||this.$primevueConfig?.ptOptions||{},o=a.mergeSections,s=o===void 0||o,c=a.mergeProps,l=c!==void 0&&c,u=r?i?this._useGlobalPT(this._getPTClassValue,t,n):this._useDefaultPT(this._getPTClassValue,t,n):void 0,d=i?void 0:this._getPTSelf(e,this._getPTClassValue,t,Z(Z({},n),{},{global:u||{}})),f=this._getPTDatasets(t),p=s||!s&&d?l?this._mergeProps(l,u,d,f):Z(Z(Z({},u),d),f):Z(Z({},d),f),m=t===`root`?d?.[`data-pc-id`]??this.$attrs[`data-pc-id`]:void 0;return m&&p&&m!==this.$attrSelector?Z(Z({},p),{},{\"data-pc-id\":`${m} ${this.$attrSelector}`}):p},_getPTSelf:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=[...arguments].slice(1);return U(this._usePT.apply(this,[this._getPT(e,this.$name)].concat(t)),this._usePT.apply(this,[this.$_attrsPT].concat(t)))},_getPTDatasets:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=`data-pc-`,n=e===`root`&&q(this.pt?.[`data-pc-section`]);return e!==`transition`&&Z(Z({},e===`root`&&Z(Z(eg({},`${t}name`,fl(n?this.pt?.[`data-pc-section`]:this.$.type.name)),n&&eg({},`${t}extend`,fl(this.$.type.name))),{},eg({\"data-pc-id\":this.$attrSelector},this.$attrSelector,``))),{},eg({},`${t}section`,fl(e)))},_getPTClassValue:function(){var e=this._getOptionValue.apply(this,arguments);return dl(e)||ml(e)?{class:e}:e},_getPT:function(e){var t=this,n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,r=arguments.length>2?arguments[2]:void 0,i=function(e){var i=arguments.length>1&&arguments[1]!==void 0&&arguments[1],a=r?r(e):e,o=fl(n),s=fl(t.$name);return(i&&o===s?void 0:a?.[o])??a};return e!=null&&Object.prototype.hasOwnProperty.call(e,`_usept`)?{_usept:e._usept,originalValue:i(e.originalValue),value:i(e.value)}:i(e,!0)},_usePT:function(e,t,n,r){var i=function(e){return t(e,n,r)};if(e!=null&&Object.prototype.hasOwnProperty.call(e,`_usept`)){var a=e._usept||this.$primevueConfig?.ptOptions||{},o=a.mergeSections,s=o===void 0||o,c=a.mergeProps,l=c!==void 0&&c,u=i(e.originalValue),d=i(e.value);return u===void 0&&d===void 0?void 0:dl(d)?d:dl(u)?u:s||!s&&d?l?this._mergeProps(l,u,d):Z(Z({},u),d):d}return i(e)},_useGlobalPT:function(e,t,n){return this._usePT(this.globalPT,e,t,n)},_useDefaultPT:function(e,t,n){return this._usePT(this.defaultPT,e,t,n)},$markInstance:function(e){e.setAttribute(`data-pc-id`,this.$attrSelector),e.setAttribute(this.$attrSelector,``)},ptm:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};return this._getPTValue(this.pt,e,Z(Z({},this.$params),t))},ptmi:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=U(this.$_attrsWithoutPT,this.ptm(e,t));return n!=null&&Object.prototype.hasOwnProperty.call(n,`id`)&&(n.id??=this.$id),n},ptmo:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,n=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};return this._getPTValue(e,t,Z({instance:this},n),!1)},cx:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};return this.isUnstyled?void 0:Ph(this._getOptionValue(this.$style.classes,e,Z(Z({},this.$params),t)),Hh(this.$primevue))},pcn:function(){var e=[...arguments];return Ph(e.length>1?e:e[0],Hh(this.$primevue))},psel:function(e){return Fh(e,Hh(this.$primevue))},sx:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0,n=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};if(t){var r=this._getOptionValue(this.$style.inlineStyles,e,Z(Z({},this.$params),n));return[this._getOptionValue(Uh.inlineStyles,e,Z(Z({},this.$params),n)),r]}}},computed:{globalPT:function(){var e=this;return this._getPT(this.$primevueConfig?.pt,void 0,function(t){return ul(t,{instance:e})})},defaultPT:function(){var e=this;return this._getPT(this.$primevueConfig?.pt,void 0,function(t){return e._getOptionValue(t,e.$name,Z({},e.$params))||ul(t,Z({},e.$params))})},isUnstyled:function(){return this.unstyled===void 0?this.$primevueConfig?.unstyled:this.unstyled},$preflight:function(){return!this.isUnstyled&&this.$primevueConfig?.preflight===!0},$id:function(){return this.$attrs.id||this.uid},$inProps:function(){var e=Object.keys(this.$.vnode?.props||{});return Object.fromEntries(Object.entries(this.$props).filter(function(t){var n=qh(t,1)[0];return e?.includes(n)}))},$theme:function(){return this.$primevueConfig?.theme},$style:function(){return Z(Z({classes:void 0,inlineStyles:void 0,load:function(){},loadCSS:function(){},loadStyle:function(){}},(this._getHostInstance(this)||{}).$style),this.$options.style)},$instanceSelector:function(){return`[data-pc-id~=\"${this.$attrSelector}\"]`},$styleOptions:function(){var e;return{nonce:(e=this.$primevueConfig)==null||(e=e.csp)==null?void 0:e.nonce,isolation:this.$isolation}},$primevueConfig:function(){return this.$primevue?.config},$name:function(){return this.$options.hostName||this.$.type.name},$params:function(){var e=this._getHostInstance(this)||this.$parent;return{instance:this,props:this.$props,state:this.$data,attrs:this.$attrs,parent:{instance:e,props:e?.$props,state:e?.$data,attrs:e?.$attrs}}},$_attrsPT:function(){return Object.entries(this.$attrs||{}).filter(function(e){return qh(e,1)[0]?.startsWith(`pt:`)}).reduce(function(e,t){var n=qh(t,2),r=n[0],i=n[1];return Xh(Gh(r.split(`:`))).slice(1)?.reduce(function(e,t,n,r){return!e[t]&&(e[t]=n===r.length-1?i:{}),e[t]},e),e},{})},$_attrsWithoutPT:function(){return Object.entries(this.$attrs||{}).filter(function(e){var t=qh(e,1)[0];return!(t!=null&&t.startsWith(`pt:`))}).reduce(function(e,t){var n=qh(t,2),r=n[0];return e[r]=n[1],e},{})}}},ig=Y.extend({name:`badge`,style:`\n"
    "    .p-badge {\n"
    "        display: inline-flex;\n"
    "        border-radius: dt('badge.border.radius');\n"
    "        align-items: center;\n"
    "        justify-content: center;\n"
    "        padding: dt('badge.padding');\n"
    "        background: dt('badge.primary.background');\n"
    "        color: dt('badge.primary.color');\n"
    "        font-size: dt('badge.font.size');\n"
    "        font-weight: dt('badge.font.weight');\n"
    "        min-width: dt('badge.min.width');\n"
    "        height: dt('badge.height');\n"
    "    }\n"
    "\n"
    "    .p-badge-dot {\n"
    "        width: dt('badge.dot.size');\n"
    "        min-width: dt('badge.dot.size');\n"
    "        height: dt('badge.dot.size');\n"
    "        border-radius: 50%;\n"
    "        padding: 0;\n"
    "    }\n"
    "\n"
    "    .p-badge-circle {\n"
    "        padding: 0;\n"
    "        border-radius: 50%;\n"
    "    }\n"
    "\n"
    "    .p-badge-secondary {\n"
    "        background: dt('badge.secondary.background');\n"
    "        color: dt('badge.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-success {\n"
    "        background: dt('badge.success.background');\n"
    "        color: dt('badge.success.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-info {\n"
    "        background: dt('badge.info.background');\n"
    "        color: dt('badge.info.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-warn {\n"
    "        background: dt('badge.warn.background');\n"
    "        color: dt('badge.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-danger {\n"
    "        background: dt('badge.danger.background');\n"
    "        color: dt('badge.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-contrast {\n"
    "        background: dt('badge.contrast.background');\n"
    "        color: dt('badge.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-badge-sm {\n"
    "        font-size: dt('badge.sm.font.size');\n"
    "        min-width: dt('badge.sm.min.width');\n"
    "        height: dt('badge.sm.height');\n"
    "    }\n"
    "\n"
    "    .p-badge-lg {\n"
    "        font-size: dt('badge.lg.font.size');\n"
    "        min-width: dt('badge.lg.min.width');\n"
    "        height: dt('badge.lg.height');\n"
    "    }\n"
    "\n"
    "    .p-badge-xl {\n"
    "        font-size: dt('badge.xl.font.size');\n"
    "        min-width: dt('badge.xl.min.width');\n"
    "        height: dt('badge.xl.height');\n"
    "    }\n"
    "`,classes:{root:function(e){var t=e.props,n=e.instance;return[`p-badge p-component`,{\"p-badge-circle\":q(t.value)&&String(t.value).length===1,\"p-badge-dot\":el(t.value)&&!n.$slots.default,\"p-badge-sm\":t.size===`small`,\"p-badge-lg\":t.size===`large`,\"p-badge-xl\":t.size===`xlarge`,\"p-badge-info\":t.severity===`info`,\"p-badge-success\":t.severity===`success`,\"p-badge-warn\":t.severity===`warn`,\"p-badge-danger\":t.severity===`danger`,\"p-badge-secondary\":t.severity===`secondary`,\"p-badge-contrast\":t.severity===`contrast`}]}}}),ag={name:`BaseBadge`,extends:rg,props:{value:{type:[String,Number],default:null},severity:{type:String,default:null},size:{type:String,default:null}},style:ig,provide:function(){return{$pcBadge:this,$parentInstance:this}}};function og(e){\"@babel/helpers - typeof\";return og=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},og(e)}function sg(e,t,n){return(t=cg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function cg(e){var t=lg(e,`string`);return og(t)==`symbol`?t:t+``}function lg(e,t){if(og(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(og(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var ug={name:`Badge`,extends:ag,inheritAttrs:!1,computed:{dataP:function(){return Tl(sg(sg({circle:this.value!=null&&String(this.value).length===1,empty:this.value==null&&!this.$slots.default},this.severity,this.severity),this.size,this.size))}}},dg=[`data-p`];function fg(e,t,n,r,i,a){return L(),R(`span`,U({class:e.cx(`root`),\"data-p\":a.dataP},e.ptmi(`root`)),[F(e.$slots,`default`,{},function(){return[Ta(k(e.value),1)]})],16,dg)}ug.render=fg;function pg(e){\"@babel/helpers - typeof\";return pg=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},pg(e)}function mg(e,t){return yg(e)||vg(e,t)||gg(e,t)||hg()}function hg(){throw TypeError(`Invalid attempt to destructure non-iterable instance.\n"
    "In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function gg(e,t){if(e){if(typeof e==`string`)return _g(e,t);var n={}.toString.call(e).slice(8,-1);return n===`Object`&&e.constructor&&(n=e.constructor.name),n===`Map`||n===`Set`?Array.from(e):n===`Arguments`||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?_g(e,t):void 0}}function _g(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function vg(e,t){var n=e==null?null:typeof Symbol<`u`&&e[Symbol.iterator]||e[`@@iterator`];if(n!=null){var r,i,a,o,s=[],c=!0,l=!1;try{if(a=(n=n.call(e)).next,t!==0)for(;!(c=(r=a.call(n)).done)&&(s.push(r.value),s.length!==t);c=!0);}catch(e){l=!0,i=e}finally{try{if(!c&&n.return!=null&&(o=n.return(),Object(o)!==o))return}finally{if(l)throw i}}return s}}function yg(e){if(Array.isArray(e))return e}function bg(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function Q(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?bg(Object(n),!0).forEach(function(t){xg(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):bg(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function xg(e,t,n){return(t=Sg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Sg(e){var t=Cg(e,`string`);return pg(t)==`symbol`?t:t+``}function Cg(e,t){if(pg(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(pg(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var wg=new WeakSet,$={_getMeta:function(){return[ol(arguments.length<=0?void 0:arguments[0])||arguments.length<=0?void 0:arguments[0],ul(ol(arguments.length<=0?void 0:arguments[0])?arguments.length<=0?void 0:arguments[0]:arguments.length<=1?void 0:arguments[1])]},_getPrimeVue:function(e,t){var n,r;return(e==null||(n=e.instance)==null?void 0:n.$primevue)||(t==null||(r=t.ctx)==null||(r=r.appContext)==null||(r=r.config)==null||(r=r.globalProperties)==null?void 0:r.$primevue)},_getConfig:function(e,t){return $._getPrimeVue(e,t)?.config},_getIsolation:function(e,t){return $._getPrimeVue(e,t)?.isolation},_getOptionValue:pl,_getPTValue:function(){var e,t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=arguments.length>2&&arguments[2]!==void 0?arguments[2]:``,i=arguments.length>3&&arguments[3]!==void 0?arguments[3]:{},a=arguments.length>4&&arguments[4]!==void 0?arguments[4]:!0,o=function(){var e=$._getOptionValue.apply($,arguments);return dl(e)||ml(e)?{class:e}:e},s=((e=t.binding)==null||(e=e.value)==null?void 0:e.ptOptions)||t.$primevueConfig?.ptOptions||{},c=s.mergeSections,l=c===void 0||c,u=s.mergeProps,d=u!==void 0&&u,f=a?$._useDefaultPT(t,t.defaultPT(),o,r,i):void 0,p=$._usePT(t,$._getPT(n,t.$name),o,r,Q(Q({},i),{},{global:f||{}})),m=$._getPTDatasets(t,r);return l||!l&&p?d?$._mergeProps(t,d,f,p,m):Q(Q(Q({},f),p),m):Q(Q({},p),m)},_getPTDatasets:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,n=`data-pc-`;return Q(Q({},t===`root`&&xg({},`${n}name`,fl(e.$name))),{},xg({},`${n}section`,fl(t)))},_getPT:function(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,n=arguments.length>2?arguments[2]:void 0,r=function(e){var r=n?n(e):e,i=fl(t);return r?.[i]??r};return e&&Object.hasOwn(e,`_usept`)?{_usept:e._usept,originalValue:r(e.originalValue),value:r(e.value)}:r(e)},_usePT:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1?arguments[1]:void 0,n=arguments.length>2?arguments[2]:void 0,r=arguments.length>3?arguments[3]:void 0,i=arguments.length>4?arguments[4]:void 0,a=function(e){return n(e,r,i)};if(t&&Object.hasOwn(t,`_usept`)){var o=t._usept||e.$primevueConfig?.ptOptions||{},s=o.mergeSections,c=s===void 0||s,l=o.mergeProps,u=l!==void 0&&l,d=a(t.originalValue),f=a(t.value);return d===void 0&&f===void 0?void 0:dl(f)?f:dl(d)?d:c||!c&&f?u?$._mergeProps(e,u,d,f):Q(Q({},d),f):f}return a(t)},_useDefaultPT:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=arguments.length>2?arguments[2]:void 0,r=arguments.length>3?arguments[3]:void 0,i=arguments.length>4?arguments[4]:void 0;return $._usePT(e,t,n,r,i)},_loadStyles:function(){var e,t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},n=arguments.length>1?arguments[1]:void 0,r=arguments.length>2?arguments[2]:void 0,i=$._getConfig(n,r),a=t.$isolation,o={nonce:i==null||(e=i.csp)==null?void 0:e.nonce,isolation:a};$._loadCoreStyles(t,o),$._loadThemeStyles(t,o),$._loadScopedThemeStyles(t,o),$._removeThemeListeners(t),t.$loadStyles=function(){return $._loadThemeStyles(t,o)},$._themeChangeListener(t.$loadStyles,a)},_loadCoreStyles:function(){var e,t,n,r,i=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},a=arguments.length>1?arguments[1]:void 0;(e=i.$style)!=null&&e.name&&xd((a==null||(t=a.isolation)==null?void 0:t.base)||yd,i.$style.name,function(){return[Y.loadCSS(a),i.$style?.loadCSS(a)]}),(n=i.preflight)!=null&&n.call(i)&&xd((a==null||(r=a.isolation)==null?void 0:r.base)||yd,`preflight`,function(){return Y.loadPreflight(a)})},_loadThemeStyles:function(){var e,t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},n=arguments.length>1?arguments[1]:void 0;if(!(t!=null&&t.isUnstyled())){var r=zh(n?.isolation);if((t==null||(e=t.theme)==null?void 0:e.call(t))!==`none`&&$._loadThemeSheets(t,r,n),!r.isStyleNameLoaded(`layer-order`)){var i,a,o,s=ed((i=t.preflight)!=null&&i.call(t)?$u:void 0,(a=t.$style)==null||(o=a.getLayerOrderThemeCSS)==null?void 0:o.call(a,r));Y.load(s,Q({name:`layer-order`,first:!0,raw:!0},n)),r.setLoadedStyleName(`layer-order`)}}},_loadThemeSheets:function(e,t,n){var r;if(!t.isStyleNameLoaded(`common`)){var i,a,o=((i=e.$style)==null||(a=i.getCommonTheme)==null?void 0:a.call(i,void 0,t))||{},s=o.primitive,c=o.semantic,l=o.global,u=o.style;Y.load(s?.css,Q({name:`primitive-variables`,raw:!0},n)),Y.load(c?.css,Q({name:`semantic-variables`,raw:!0},n)),Y.load(l?.css,Q({name:`global-variables`,raw:!0},n)),Y.loadStyle(Q({name:`global-style`},n),u),t.setLoadedStyleName(`common`)}if(!t.isStyleNameLoaded(e.$style?.name)&&(r=e.$style)!=null&&r.name){var d,f,p,m,h=((d=e.$style)==null||(f=d.getDirectiveTheme)==null?void 0:f.call(d,void 0,t))||{},g=h.css,_=h.style;(p=e.$style)==null||p.load(g,Q({name:`${e.$style.name}-variables`,raw:!0},n)),(m=e.$style)==null||m.loadStyle(Q({name:`${e.$style.name}-style`},n),_),t.setLoadedStyleName(e.$style.name)}},_loadScopedThemeStyles:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},t=arguments.length>1?arguments[1]:void 0,n=e.preset();if(n&&e.$attrSelector){var r,i,a=(((r=e.$style)==null||(i=r.getPresetTheme)==null?void 0:i.call(r,n,`[data-pc-id~=\"${e.$attrSelector}\"]`,void 0,zh(t?.isolation)))||{}).css,o=`${e.$attrSelector}-${e.$style.name}`;e.scopedStyleEl=(e.$style?.load(a,Q({name:o,raw:!0},t))).el,e.scopedStyleName=o}},_themeChangeListener:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:function(){};Bh(arguments.length>1?arguments[1]:void 0).on(`theme:change`,e)},_removeThemeListeners:function(){var e=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};Bh(e.$isolation).off(`theme:change`,e.$loadStyles),e.$loadStyles=void 0},_hook:function(e,t,n,r,i,a){var o,s,c=`on${Sl(t)}`,l=$._getConfig(r,i),u=n?.$instance,d=$._usePT(u,$._getPT(r==null||(o=r.value)==null?void 0:o.pt,e),$._getOptionValue,`hooks.${c}`),f=$._useDefaultPT(u,l==null||(s=l.pt)==null||(s=s.directives)==null?void 0:s[e],$._getOptionValue,`hooks.${c}`),p={el:n,binding:r,vnode:i,prevVnode:a};d?.(u,p),f?.(u,p)},_mergeProps:function(){var e=arguments.length>1?arguments[1]:void 0,t=[...arguments].slice(2);return rl(e)?e.apply(void 0,t):U.apply(void 0,t)},_extend:function(e){var t=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},n=function(n,r,i,a,o){var s,c,l;r._$instances=r._$instances||{};var u=$._getConfig(i,a),d=r._$instances[e]||{},f=el(d)?Q(Q({},t),t?.methods):{};r._$instances[e]=Q(Q({},d),{},{$name:e,$host:r,$binding:i,$modifiers:i?.modifiers,$value:i?.value,$el:d.$el||r||void 0,$style:Q({classes:void 0,inlineStyles:void 0,load:function(){},loadCSS:function(){},loadStyle:function(){}},t?.style),$primevueConfig:u,$isolation:$._getIsolation(i,a),$styleHost:$._getIsolation(i,a)||$._getPrimeVue(i,a)?.shadowRoot,$attrSelector:(s=r.$pd)==null||(s=s[e])==null?void 0:s.attrSelector,defaultPT:function(){return $._getPT(u?.pt,void 0,function(t){var n;return t==null||(n=t.directives)==null?void 0:n[e]})},isUnstyled:function(){var t,n;return((t=r._$instances[e])==null||(t=t.$binding)==null||(t=t.value)==null?void 0:t.unstyled)===void 0?u?.unstyled:(n=r._$instances[e])==null||(n=n.$binding)==null||(n=n.value)==null?void 0:n.unstyled},theme:function(){var t;return(t=r._$instances[e])==null||(t=t.$primevueConfig)==null?void 0:t.theme},preflight:function(){var t,n;return!((t=r._$instances[e])!=null&&t.isUnstyled())&&((n=r._$instances[e])==null||(n=n.$primevueConfig)==null?void 0:n.preflight)===!0},preset:function(){var t;return(t=r._$instances[e])==null||(t=t.$binding)==null||(t=t.value)==null?void 0:t.dt},ptm:function(){var t,n=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,i=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};return $._getPTValue(r._$instances[e],(t=r._$instances[e])==null||(t=t.$binding)==null||(t=t.value)==null?void 0:t.pt,n,Q({},i))},ptmo:function(){var t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:``,i=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};return $._getPTValue(r._$instances[e],t,n,i,!1)},cx:function(){var t,n,i=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{};return(t=r._$instances[e])!=null&&t.isUnstyled()?void 0:r._$instances[e].pcn($._getOptionValue((n=r._$instances[e])==null||(n=n.$style)==null?void 0:n.classes,i,Q({},a)))},pcn:function(){var t=[...arguments];return Ph(t.length>1?t:t[0],Rh(zh(r._$instances[e]?.$isolation)))},psel:function(t){return Fh(t,Rh(zh(r._$instances[e]?.$isolation)))},sx:function(){var t,n=arguments.length>0&&arguments[0]!==void 0?arguments[0]:``,i=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0,a=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};return i?$._getOptionValue((t=r._$instances[e])==null||(t=t.$style)==null?void 0:t.inlineStyles,n,Q({},a)):void 0}},f),r.$instance=r._$instances[e],(c=(l=r.$instance)[n])==null||c.call(l,r,i,a,o),r[`$${e}`]=r.$instance,$._hook(e,n,r,i,a,o),r.$pd||={},r.$pd[e]=Q(Q({},r.$pd?.[e]),{},{name:e,instance:r._$instances[e]})},r=function(t){var n,r,i,a=t._$instances[e],o=a?.watch,s=function(e){var t,n=e.newValue,r=e.oldValue;return o==null||(t=o.config)==null?void 0:t.call(a,n,r)},c=function(e){var t,n=e.newValue,r=e.oldValue;return o==null||(t=o[`config.ripple`])==null?void 0:t.call(a,n,r)},l=a.$isolation?.events||tf;a.$watchersCallback={config:s,\"config.ripple\":c},o==null||(n=o.config)==null||n.call(a,a?.$primevueConfig),l.on(`config:change`,s),o==null||(r=o[`config.ripple`])==null||r.call(a,a==null||(i=a.$primevueConfig)==null?void 0:i.ripple),l.on(`config:ripple:change`,c)},i=function(t){var n=t._$instances[e].$watchersCallback,r=t._$instances[e].$isolation?.events||tf;n&&(r.off(`config:change`,n.config),r.off(`config:ripple:change`,n[`config.ripple`]),t._$instances[e].$watchersCallback=void 0)};return{created:function(t,r,i,a){t.$pd||={},t.$pd[e]={name:e,attrSelector:vu(`pd`)},n(`created`,t,r,i,a)},beforeMount:function(t,i,a,o){$._loadStyles(t.$pd[e]?.instance,i,a),n(`beforeMount`,t,i,a,o),r(t)},mounted:function(t,r,i,a){var o;wg.has(t)||($._loadStyles(t.$pd[e]?.instance,r,i),(o=t.$pd[e])==null||(o=o.instance)==null||(o=o.$styleHost)==null||o.attach(t.$pd[e],t),n(`mounted`,t,r,i,a))},beforeUpdate:function(e,t,r,i){n(`beforeUpdate`,e,t,r,i)},updated:function(t,r,i,a){wg.has(t)||($._loadStyles(t.$pd[e]?.instance,r,i),n(`updated`,t,r,i,a))},beforeUnmount:function(t,r,a,o){wg.add(t),i(t),$._removeThemeListeners(t.$pd[e]?.instance),n(`beforeUnmount`,t,r,a,o)},unmounted:function(t,r,i,a){var o,s,c;(o=t.$pd[e])==null||(o=o.instance)==null||(o=o.scopedStyleEl)==null||(o=o.value)==null||o.remove();var l=(s=t.$pd[e])==null||(s=s.instance)==null?void 0:s.$styleHost;(c=t.$pd[e])!=null&&(c=c.instance)!=null&&c.scopedStyleName&&l?.drop(t.$pd[e].instance.scopedStyleName),l?.detach(t.$pd[e]),n(`unmounted`,t,r,i,a)}}},extend:function(){var e=mg($._getMeta.apply($,arguments),2),t=e[0],n=e[1];return Q({extend:function(){var e=mg($._getMeta.apply($,arguments),2),t=e[0],r=e[1];return $.extend(t,Q(Q(Q({},n),n?.methods),r))}},$._extend(t,n))}},Tg=Y.extend({name:`ripple-directive`,style:`\n"
    "    .p-ink {\n"
    "        display: block;\n"
    "        position: absolute;\n"
    "        background: dt('ripple.background');\n"
    "        border-radius: 100%;\n"
    "        transform: scale(0);\n"
    "        pointer-events: none;\n"
    "    }\n"
    "\n"
    "    .p-ink-active {\n"
    "        animation: ripple 0.4s linear;\n"
    "    }\n"
    "\n"
    "    @keyframes ripple {\n"
    "        100% {\n"
    "            opacity: 0;\n"
    "            transform: scale(2.5);\n"
    "        }\n"
    "    }\n"
    "`,classes:{root:`p-ink`}}),Eg=$.extend({style:Tg});function Dg(e){\"@babel/helpers - typeof\";return Dg=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},Dg(e)}function Og(e){return Mg(e)||jg(e)||Ag(e)||kg()}function kg(){throw TypeError(`Invalid attempt to spread non-iterable instance.\n"
    "In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function Ag(e,t){if(e){if(typeof e==`string`)return Ng(e,t);var n={}.toString.call(e).slice(8,-1);return n===`Object`&&e.constructor&&(n=e.constructor.name),n===`Map`||n===`Set`?Array.from(e):n===`Arguments`||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?Ng(e,t):void 0}}function jg(e){if(typeof Symbol<`u`&&e[Symbol.iterator]!=null||e[`@@iterator`]!=null)return Array.from(e)}function Mg(e){if(Array.isArray(e))return Ng(e)}function Ng(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}function Pg(e,t,n){return(t=Fg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Fg(e){var t=Ig(e,`string`);return Dg(t)==`symbol`?t:t+``}function Ig(e,t){if(Dg(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(Dg(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var Lg=Eg.extend(`ripple`,{watch:{\"config.ripple\":function(e){e?(this.createRipple(this.$host),this.bindEvents(this.$host),this.$host.setAttribute(`data-pd-ripple`,!0),this.$host.style.overflow=`hidden`,this.$host.style.position=`relative`):(this.remove(this.$host),this.$host.removeAttribute(`data-pd-ripple`))}},unmounted:function(e){this.remove(e)},timeout:void 0,methods:{bindEvents:function(e){e.addEventListener(`mousedown`,this.onMouseDown.bind(this))},unbindEvents:function(e){e.removeEventListener(`mousedown`,this.onMouseDown.bind(this))},createRipple:function(e){var t=this.getInk(e);t||(t=iu(`span`,Pg(Pg({role:`presentation`,\"aria-hidden\":!0,\"data-p-ink\":!0,\"data-p-ink-active\":!1,class:!this.isUnstyled()&&this.cx(`root`),onAnimationEnd:this.onAnimationEnd.bind(this),\"data-pc-id\":this.$attrSelector},this.$attrSelector,``),`p-bind`,this.ptm(`root`))),e.appendChild(t),this.$el=t)},remove:function(e){var t=this.getInk(e);t&&(this.$host.style.overflow=``,this.$host.style.position=``,this.unbindEvents(e),t.removeEventListener(`animationend`,this.onAnimationEnd),t.remove())},onMouseDown:function(e){var t=this,n=e.currentTarget,r=this.getInk(n);if(r&&getComputedStyle(r,null).display!==`none`){if(!this.isUnstyled()&&Pl(r,this.pcn(`p-ink-active`)),r.setAttribute(`data-p-ink-active`,`false`),!lu(r)&&!pu(r)){var i=Math.max(Zl(n),du(n));r.style.height=i+`px`,r.style.width=i+`px`}var a=uu(n),o=e.pageX-a.left+document.body.scrollTop-pu(r)/2,s=e.pageY-a.top+document.body.scrollLeft-lu(r)/2;r.style.top=s+`px`,r.style.left=o+`px`,!this.isUnstyled()&&Nl(r,this.pcn(`p-ink-active`)),r.setAttribute(`data-p-ink-active`,`true`),this.timeout=setTimeout(function(){r&&(!t.isUnstyled()&&Pl(r,t.pcn(`p-ink-active`)),r.setAttribute(`data-p-ink-active`,`false`))},401)}},onAnimationEnd:function(e){this.timeout&&clearTimeout(this.timeout),!this.isUnstyled()&&Pl(e.currentTarget,this.pcn(`p-ink-active`)),e.currentTarget.setAttribute(`data-p-ink-active`,`false`)},getInk:function(e){return e&&e.children?Og(e.children).find(function(e){return cu(e,`data-pc-name`)===`ripple`}):void 0}}}),Rg=`\n"
    "    .p-button {\n"
    "        display: inline-flex;\n"
    "        cursor: pointer;\n"
    "        user-select: none;\n"
    "        align-items: center;\n"
    "        justify-content: center;\n"
    "        overflow: hidden;\n"
    "        position: relative;\n"
    "        color: dt('button.primary.color');\n"
    "        background: dt('button.primary.background');\n"
    "        border: 1px solid dt('button.primary.border.color');\n"
    "        padding: dt('button.padding.y') dt('button.padding.x');\n"
    "        font-size: dt('button.font.size');\n"
    "        font-weight: dt('button.label.font.weight');\n"
    "        transition:\n"
    "            background dt('button.transition.duration'),\n"
    "            color dt('button.transition.duration'),\n"
    "            border-color dt('button.transition.duration'),\n"
    "            outline-color dt('button.transition.duration'),\n"
    "            box-shadow dt('button.transition.duration');\n"
    "        border-radius: dt('button.border.radius');\n"
    "        outline-color: transparent;\n"
    "        gap: dt('button.gap');\n"
    "    }\n"
    "\n"
    "    .p-button:disabled {\n"
    "        cursor: default;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-right {\n"
    "        order: 1;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-right:dir(rtl) {\n"
    "        order: -1;\n"
    "    }\n"
    "\n"
    "    .p-button:not(.p-button-vertical) .p-button-icon:not(.p-button-icon-right):dir(rtl) {\n"
    "        order: 1;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-bottom {\n"
    "        order: 2;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-only {\n"
    "        width: dt('button.icon.only.width');\n"
    "        padding-inline-start: 0;\n"
    "        padding-inline-end: 0;\n"
    "        gap: 0;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-only.p-button-rounded {\n"
    "        border-radius: 50%;\n"
    "        height: dt('button.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button-icon-only .p-button-label {\n"
    "        visibility: hidden;\n"
    "        width: 0;\n"
    "    }\n"
    "\n"
    "    .p-button-icon-only::after {\n"
    "        content: \"\\xA0\";\n"
    "        visibility: hidden;\n"
    "        width: 0;\n"
    "    }\n"
    "\n"
    "    .p-button-sm {\n"
    "        font-size: dt('button.sm.font.size');\n"
    "        padding: dt('button.sm.padding.y') dt('button.sm.padding.x');\n"
    "    }\n"
    "\n"
    "    .p-button-sm .p-button-icon {\n"
    "        font-size: dt('button.sm.font.size');\n"
    "    }\n"
    "\n"
    "    .p-button-sm.p-button-icon-only {\n"
    "        width: dt('button.sm.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button-sm.p-button-icon-only.p-button-rounded {\n"
    "        height: dt('button.sm.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button-lg {\n"
    "        font-size: dt('button.lg.font.size');\n"
    "        padding: dt('button.lg.padding.y') dt('button.lg.padding.x');\n"
    "    }\n"
    "\n"
    "    .p-button-lg .p-button-icon {\n"
    "        font-size: dt('button.lg.font.size');\n"
    "    }\n"
    "\n"
    "    .p-button-lg.p-button-icon-only {\n"
    "        width: dt('button.lg.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button-lg.p-button-icon-only.p-button-rounded {\n"
    "        height: dt('button.lg.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button-vertical {\n"
    "        flex-direction: column;\n"
    "    }\n"
    "\n"
    "    .p-button-label {\n"
    "        font-weight: dt('button.label.font.weight');\n"
    "    }\n"
    "\n"
    "    .p-button-fluid {\n"
    "        width: 100%;\n"
    "    }\n"
    "\n"
    "    .p-button-fluid.p-button-icon-only {\n"
    "        width: dt('button.icon.only.width');\n"
    "    }\n"
    "\n"
    "    .p-button:not(:disabled):hover {\n"
    "        background: dt('button.primary.hover.background');\n"
    "        border: 1px solid dt('button.primary.hover.border.color');\n"
    "        color: dt('button.primary.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button:not(:disabled):active {\n"
    "        background: dt('button.primary.active.background');\n"
    "        border: 1px solid dt('button.primary.active.border.color');\n"
    "        color: dt('button.primary.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button:focus-visible {\n"
    "        box-shadow: dt('button.primary.focus.ring.shadow');\n"
    "        outline: dt('button.focus.ring.width') dt('button.focus.ring.style') dt('button.primary.focus.ring.color');\n"
    "        outline-offset: dt('button.focus.ring.offset');\n"
    "    }\n"
    "\n"
    "    .p-button .p-badge {\n"
    "        min-width: dt('button.badge.size');\n"
    "        height: dt('button.badge.size');\n"
    "        line-height: dt('button.badge.size');\n"
    "    }\n"
    "\n"
    "    .p-button-raised {\n"
    "        box-shadow: dt('button.raised.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-rounded {\n"
    "        border-radius: dt('button.rounded.border.radius');\n"
    "    }\n"
    "\n"
    "    .p-button-secondary {\n"
    "        background: dt('button.secondary.background');\n"
    "        border: 1px solid dt('button.secondary.border.color');\n"
    "        color: dt('button.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-secondary:not(:disabled):hover {\n"
    "        background: dt('button.secondary.hover.background');\n"
    "        border: 1px solid dt('button.secondary.hover.border.color');\n"
    "        color: dt('button.secondary.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-secondary:not(:disabled):active {\n"
    "        background: dt('button.secondary.active.background');\n"
    "        border: 1px solid dt('button.secondary.active.border.color');\n"
    "        color: dt('button.secondary.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-secondary:focus-visible {\n"
    "        outline-color: dt('button.secondary.focus.ring.color');\n"
    "        box-shadow: dt('button.secondary.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-success {\n"
    "        background: dt('button.success.background');\n"
    "        border: 1px solid dt('button.success.border.color');\n"
    "        color: dt('button.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-success:not(:disabled):hover {\n"
    "        background: dt('button.success.hover.background');\n"
    "        border: 1px solid dt('button.success.hover.border.color');\n"
    "        color: dt('button.success.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-success:not(:disabled):active {\n"
    "        background: dt('button.success.active.background');\n"
    "        border: 1px solid dt('button.success.active.border.color');\n"
    "        color: dt('button.success.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-success:focus-visible {\n"
    "        outline-color: dt('button.success.focus.ring.color');\n"
    "        box-shadow: dt('button.success.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-info {\n"
    "        background: dt('button.info.background');\n"
    "        border: 1px solid dt('button.info.border.color');\n"
    "        color: dt('button.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-info:not(:disabled):hover {\n"
    "        background: dt('button.info.hover.background');\n"
    "        border: 1px solid dt('button.info.hover.border.color');\n"
    "        color: dt('button.info.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-info:not(:disabled):active {\n"
    "        background: dt('button.info.active.background');\n"
    "        border: 1px solid dt('button.info.active.border.color');\n"
    "        color: dt('button.info.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-info:focus-visible {\n"
    "        outline-color: dt('button.info.focus.ring.color');\n"
    "        box-shadow: dt('button.info.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-warn {\n"
    "        background: dt('button.warn.background');\n"
    "        border: 1px solid dt('button.warn.border.color');\n"
    "        color: dt('button.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-warn:not(:disabled):hover {\n"
    "        background: dt('button.warn.hover.background');\n"
    "        border: 1px solid dt('button.warn.hover.border.color');\n"
    "        color: dt('button.warn.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-warn:not(:disabled):active {\n"
    "        background: dt('button.warn.active.background');\n"
    "        border: 1px solid dt('button.warn.active.border.color');\n"
    "        color: dt('button.warn.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-warn:focus-visible {\n"
    "        outline-color: dt('button.warn.focus.ring.color');\n"
    "        box-shadow: dt('button.warn.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-help {\n"
    "        background: dt('button.help.background');\n"
    "        border: 1px solid dt('button.help.border.color');\n"
    "        color: dt('button.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-help:not(:disabled):hover {\n"
    "        background: dt('button.help.hover.background');\n"
    "        border: 1px solid dt('button.help.hover.border.color');\n"
    "        color: dt('button.help.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-help:not(:disabled):active {\n"
    "        background: dt('button.help.active.background');\n"
    "        border: 1px solid dt('button.help.active.border.color');\n"
    "        color: dt('button.help.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-help:focus-visible {\n"
    "        outline-color: dt('button.help.focus.ring.color');\n"
    "        box-shadow: dt('button.help.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-danger {\n"
    "        background: dt('button.danger.background');\n"
    "        border: 1px solid dt('button.danger.border.color');\n"
    "        color: dt('button.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-danger:not(:disabled):hover {\n"
    "        background: dt('button.danger.hover.background');\n"
    "        border: 1px solid dt('button.danger.hover.border.color');\n"
    "        color: dt('button.danger.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-danger:not(:disabled):active {\n"
    "        background: dt('button.danger.active.background');\n"
    "        border: 1px solid dt('button.danger.active.border.color');\n"
    "        color: dt('button.danger.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-danger:focus-visible {\n"
    "        outline-color: dt('button.danger.focus.ring.color');\n"
    "        box-shadow: dt('button.danger.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-contrast {\n"
    "        background: dt('button.contrast.background');\n"
    "        border: 1px solid dt('button.contrast.border.color');\n"
    "        color: dt('button.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-contrast:not(:disabled):hover {\n"
    "        background: dt('button.contrast.hover.background');\n"
    "        border: 1px solid dt('button.contrast.hover.border.color');\n"
    "        color: dt('button.contrast.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-contrast:not(:disabled):active {\n"
    "        background: dt('button.contrast.active.background');\n"
    "        border: 1px solid dt('button.contrast.active.border.color');\n"
    "        color: dt('button.contrast.active.color');\n"
    "    }\n"
    "\n"
    "    .p-button-contrast:focus-visible {\n"
    "        outline-color: dt('button.contrast.focus.ring.color');\n"
    "        box-shadow: dt('button.contrast.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined {\n"
    "        background: transparent;\n"
    "        border-color: dt('button.outlined.primary.border.color');\n"
    "        color: dt('button.outlined.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined:not(:disabled):hover {\n"
    "        background: dt('button.outlined.primary.hover.background');\n"
    "        border-color: dt('button.outlined.primary.border.color');\n"
    "        color: dt('button.outlined.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined:not(:disabled):active {\n"
    "        background: dt('button.outlined.primary.active.background');\n"
    "        border-color: dt('button.outlined.primary.border.color');\n"
    "        color: dt('button.outlined.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-secondary {\n"
    "        border-color: dt('button.outlined.secondary.border.color');\n"
    "        color: dt('button.outlined.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-secondary:not(:disabled):hover {\n"
    "        background: dt('button.outlined.secondary.hover.background');\n"
    "        border-color: dt('button.outlined.secondary.border.color');\n"
    "        color: dt('button.outlined.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-secondary:not(:disabled):active {\n"
    "        background: dt('button.outlined.secondary.active.background');\n"
    "        border-color: dt('button.outlined.secondary.border.color');\n"
    "        color: dt('button.outlined.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-success {\n"
    "        border-color: dt('button.outlined.success.border.color');\n"
    "        color: dt('button.outlined.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-success:not(:disabled):hover {\n"
    "        background: dt('button.outlined.success.hover.background');\n"
    "        border-color: dt('button.outlined.success.border.color');\n"
    "        color: dt('button.outlined.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-success:not(:disabled):active {\n"
    "        background: dt('button.outlined.success.active.background');\n"
    "        border-color: dt('button.outlined.success.border.color');\n"
    "        color: dt('button.outlined.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-info {\n"
    "        border-color: dt('button.outlined.info.border.color');\n"
    "        color: dt('button.outlined.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-info:not(:disabled):hover {\n"
    "        background: dt('button.outlined.info.hover.background');\n"
    "        border-color: dt('button.outlined.info.border.color');\n"
    "        color: dt('button.outlined.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-info:not(:disabled):active {\n"
    "        background: dt('button.outlined.info.active.background');\n"
    "        border-color: dt('button.outlined.info.border.color');\n"
    "        color: dt('button.outlined.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-warn {\n"
    "        border-color: dt('button.outlined.warn.border.color');\n"
    "        color: dt('button.outlined.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-warn:not(:disabled):hover {\n"
    "        background: dt('button.outlined.warn.hover.background');\n"
    "        border-color: dt('button.outlined.warn.border.color');\n"
    "        color: dt('button.outlined.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-warn:not(:disabled):active {\n"
    "        background: dt('button.outlined.warn.active.background');\n"
    "        border-color: dt('button.outlined.warn.border.color');\n"
    "        color: dt('button.outlined.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-help {\n"
    "        border-color: dt('button.outlined.help.border.color');\n"
    "        color: dt('button.outlined.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-help:not(:disabled):hover {\n"
    "        background: dt('button.outlined.help.hover.background');\n"
    "        border-color: dt('button.outlined.help.border.color');\n"
    "        color: dt('button.outlined.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-help:not(:disabled):active {\n"
    "        background: dt('button.outlined.help.active.background');\n"
    "        border-color: dt('button.outlined.help.border.color');\n"
    "        color: dt('button.outlined.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-danger {\n"
    "        border-color: dt('button.outlined.danger.border.color');\n"
    "        color: dt('button.outlined.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-danger:not(:disabled):hover {\n"
    "        background: dt('button.outlined.danger.hover.background');\n"
    "        border-color: dt('button.outlined.danger.border.color');\n"
    "        color: dt('button.outlined.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-danger:not(:disabled):active {\n"
    "        background: dt('button.outlined.danger.active.background');\n"
    "        border-color: dt('button.outlined.danger.border.color');\n"
    "        color: dt('button.outlined.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-contrast {\n"
    "        border-color: dt('button.outlined.contrast.border.color');\n"
    "        color: dt('button.outlined.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-contrast:not(:disabled):hover {\n"
    "        background: dt('button.outlined.contrast.hover.background');\n"
    "        border-color: dt('button.outlined.contrast.border.color');\n"
    "        color: dt('button.outlined.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-contrast:not(:disabled):active {\n"
    "        background: dt('button.outlined.contrast.active.background');\n"
    "        border-color: dt('button.outlined.contrast.border.color');\n"
    "        color: dt('button.outlined.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-plain {\n"
    "        border-color: dt('button.outlined.plain.border.color');\n"
    "        color: dt('button.outlined.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-plain:not(:disabled):hover {\n"
    "        background: dt('button.outlined.plain.hover.background');\n"
    "        border-color: dt('button.outlined.plain.border.color');\n"
    "        color: dt('button.outlined.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-outlined.p-button-plain:not(:disabled):active {\n"
    "        background: dt('button.outlined.plain.active.background');\n"
    "        border-color: dt('button.outlined.plain.border.color');\n"
    "        color: dt('button.outlined.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text:not(:disabled):hover {\n"
    "        background: dt('button.text.primary.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text:not(:disabled):active {\n"
    "        background: dt('button.text.primary.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.primary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-secondary {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-secondary:not(:disabled):hover {\n"
    "        background: dt('button.text.secondary.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-secondary:not(:disabled):active {\n"
    "        background: dt('button.text.secondary.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.secondary.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-success {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-success:not(:disabled):hover {\n"
    "        background: dt('button.text.success.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-success:not(:disabled):active {\n"
    "        background: dt('button.text.success.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.success.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-info {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-info:not(:disabled):hover {\n"
    "        background: dt('button.text.info.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-info:not(:disabled):active {\n"
    "        background: dt('button.text.info.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.info.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-warn {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-warn:not(:disabled):hover {\n"
    "        background: dt('button.text.warn.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-warn:not(:disabled):active {\n"
    "        background: dt('button.text.warn.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.warn.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-help {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-help:not(:disabled):hover {\n"
    "        background: dt('button.text.help.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-help:not(:disabled):active {\n"
    "        background: dt('button.text.help.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.help.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-danger {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-danger:not(:disabled):hover {\n"
    "        background: dt('button.text.danger.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-danger:not(:disabled):active {\n"
    "        background: dt('button.text.danger.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.danger.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-contrast {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-contrast:not(:disabled):hover {\n"
    "        background: dt('button.text.contrast.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-contrast:not(:disabled):active {\n"
    "        background: dt('button.text.contrast.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.contrast.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-plain {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-plain:not(:disabled):hover {\n"
    "        background: dt('button.text.plain.hover.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-text.p-button-plain:not(:disabled):active {\n"
    "        background: dt('button.text.plain.active.background');\n"
    "        border-color: transparent;\n"
    "        color: dt('button.text.plain.color');\n"
    "    }\n"
    "\n"
    "    .p-button-link {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.link.color');\n"
    "    }\n"
    "\n"
    "    .p-button-link:not(:disabled):hover {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.link.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-button-link:not(:disabled):hover .p-button-label {\n"
    "        text-decoration: underline;\n"
    "    }\n"
    "\n"
    "    .p-button-link:not(:disabled):active {\n"
    "        background: transparent;\n"
    "        border-color: transparent;\n"
    "        color: dt('button.link.active.color');\n"
    "    }\n"
    "`;function zg(e){\"@babel/helpers - typeof\";return zg=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},zg(e)}function Bg(e,t,n){return(t=Vg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function Vg(e){var t=Hg(e,`string`);return zg(t)==`symbol`?t:t+``}function Hg(e,t){if(zg(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(zg(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var Ug=Y.extend({name:`button`,style:Rg,classes:{root:function(e){var t=e.instance,n=e.props;return[`p-button p-component`,Bg(Bg(Bg(Bg(Bg(Bg(Bg(Bg({\"p-button-icon-only\":n.iconOnly||t.hasIcon&&!n.label&&!n.badge,\"p-button-vertical\":(n.iconPos===`top`||n.iconPos===`bottom`)&&n.label,\"p-button-loading\":n.loading,\"p-button-link\":n.link||n.variant===`link`},`p-button-${n.severity}`,n.severity),`p-button-raised`,n.raised),`p-button-rounded`,n.rounded),`p-button-text`,n.text||n.variant===`text`),`p-button-outlined`,n.outlined||n.variant===`outlined`),`p-button-sm`,n.size===`small`),`p-button-lg`,n.size===`large`),`p-button-fluid`,t.hasFluid)]},loadingIcon:`p-button-loading-icon`,icon:function(e){var t=e.props;return[`p-button-icon`,Bg({},`p-button-icon-${t.iconPos}`,t.label)]},label:`p-button-label`}}),Wg={name:`BaseButton`,extends:rg,props:{label:{type:String,default:null},icon:{type:String,default:null},iconPos:{type:String,default:`left`},iconClass:{type:[String,Object],default:null},badge:{type:String,default:null},badgeClass:{type:[String,Object],default:null},badgeSeverity:{type:String,default:`secondary`},loading:{type:Boolean,default:!1},loadingIcon:{type:String,default:void 0},iconOnly:{type:Boolean,default:!1},as:{type:[String,Object],default:`BUTTON`},asChild:{type:Boolean,default:!1},link:{type:Boolean,default:!1},severity:{type:String,default:null},raised:{type:Boolean,default:!1},rounded:{type:Boolean,default:!1},text:{type:Boolean,default:!1},outlined:{type:Boolean,default:!1},size:{type:String,default:null},variant:{type:String,default:null},fluid:{type:Boolean,default:null}},style:Ug,provide:function(){return{$pcButton:this,$parentInstance:this}}};function Gg(e){\"@babel/helpers - typeof\";return Gg=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},Gg(e)}function Kg(e,t,n){return(t=qg(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function qg(e){var t=Jg(e,`string`);return Gg(t)==`symbol`?t:t+``}function Jg(e,t){if(Gg(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(Gg(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var Yg={name:`Button`,extends:Wg,inheritAttrs:!1,inject:{$pcFluid:{default:null}},methods:{getPTOptions:function(e){return(e===`root`?this.ptmi:this.ptm)(e,{context:{disabled:this.disabled}})}},computed:{disabled:function(){return this.$attrs.disabled||this.$attrs.disabled===``||this.loading},defaultAriaLabel:function(){return this.label?this.label+(this.badge?` `+this.badge:``):this.$attrs.ariaLabel},hasIcon:function(){return this.icon||this.$slots.icon},attrs:function(){return U(this.asAttrs,this.a11yAttrs,this.getPTOptions(`root`))},asAttrs:function(){return this.as===`BUTTON`?{type:`button`,disabled:this.disabled}:void 0},a11yAttrs:function(){return{\"aria-label\":this.defaultAriaLabel,\"data-pc-name\":`button`,\"data-p-disabled\":this.disabled,\"data-p-severity\":this.severity}},hasFluid:function(){return el(this.fluid)?!!this.$pcFluid:this.fluid},dataP:function(){return Tl(Kg(Kg(Kg(Kg(Kg(Kg(Kg(Kg(Kg(Kg({},this.size,this.size),`icon-only`,this.iconOnly||this.hasIcon&&!this.label&&!this.badge),`loading`,this.loading),`fluid`,this.hasFluid),`rounded`,this.rounded),`raised`,this.raised),`outlined`,this.outlined||this.variant===`outlined`),`text`,this.text||this.variant===`text`),`link`,this.link||this.variant===`link`),`vertical`,(this.iconPos===`top`||this.iconPos===`bottom`)&&this.label))},dataIconP:function(){return Tl(Kg(Kg({},this.iconPos,this.iconPos),this.size,this.size))},dataLabelP:function(){return Tl(Kg(Kg({},this.size,this.size),`icon-only`,this.iconOnly||this.hasIcon&&!this.label&&!this.badge))}},components:{Spinner:Mh,Badge:ug},directives:{ripple:Lg}},Xg=[`data-p`],Zg=[`data-p`];function Qg(e,t,n,r,i,a){var o=Rr(`Spinner`),s=Rr(`Badge`),c=Vr(`ripple`);return e.asChild?F(e.$slots,`default`,{class:ge(e.cx(`root`)),a11yAttrs:a.a11yAttrs},void 0,void 0,1):Rn((L(),z(Br(e.as),U({key:0,class:e.cx(`root`),\"data-p\":a.dataP},a.attrs),{default:N(function(){return[F(e.$slots,`default`,{},function(){return[e.loading?F(e.$slots,`loadingicon`,U({class:[e.cx(`loadingIcon`),e.cx(`icon`)]},e.ptm(`loadingIcon`)),function(){return[e.loadingIcon?(L(),R(`span`,U({key:0,class:[e.cx(`loadingIcon`),e.cx(`icon`),e.loadingIcon]},e.ptm(`loadingIcon`)),null,16)):(L(),z(o,U({key:1,class:[e.cx(`loadingIcon`),e.cx(`icon`)],spin:``},e.ptm(`loadingIcon`)),null,16,[`class`]))]},void 0,0):F(e.$slots,`icon`,U({class:[e.cx(`icon`)]},e.ptm(`icon`)),function(){return[e.icon?(L(),R(`span`,U({key:0,class:[e.cx(`icon`),e.icon,e.iconClass],\"data-p\":a.dataIconP},e.ptm(`icon`)),null,16,Xg)):H(``,!0)]},void 0,1),e.label?(L(),R(`span`,U({key:2,class:e.cx(`label`)},e.ptm(`label`),{\"data-p\":a.dataLabelP}),k(e.label),17,Zg)):H(``,!0),e.badge?(L(),z(s,{key:3,value:e.badge,class:ge(e.badgeClass),severity:e.badgeSeverity,unstyled:e.unstyled,pt:e.ptm(`pcBadge`)},null,8,[`value`,`class`,`severity`,`unstyled`,`pt`])):H(``,!0)]})]}),_:3},16,[`class`,`data-p`])),[[c]])}Yg.render=Qg;var $g={name:`times`,meta:{tags:[`times`,`close`,`cancel`,`delete`,`remove`]},svg:{xmlns:`http://www.w3.org/2000/svg`,width:20,height:20,viewBox:`0 0 20 20`,fill:`none`},nodes:[[`path`,{d:`M14.4199 4.51962C14.7128 4.22696 15.1876 4.22685 15.4805 4.51962C15.7731 4.81246 15.7731 5.28732 15.4805 5.58016L11.0606 10L15.4805 14.4199C15.773 14.7129 15.7732 15.1877 15.4805 15.4805C15.1877 15.7732 14.7128 15.773 14.4199 15.4805L10 11.0606L5.58014 15.4805C5.2873 15.7731 4.81245 15.7731 4.5196 15.4805C4.22682 15.1876 4.22692 14.7128 4.5196 14.4199L8.93949 10L4.5196 5.58016C4.22676 5.28727 4.22673 4.8125 4.5196 4.51962C4.81248 4.22677 5.28726 4.22678 5.58014 4.51962L10 8.93951L14.4199 4.51962Z`,fill:`currentColor`,key:`ow8ecl`}]]},e_=P({name:`Times`,inheritAttrs:!1,__name:`times`,setup(e){let{Icon:t}=jh($g);return(e,n)=>(L(),z(M(t),_e(Ca(e.$attrs)),null,16))}}),t_={name:`BaseInput`,extends:{name:`BaseEditableHolder`,extends:rg,emits:[`update:modelValue`,`value-change`],props:{modelValue:{type:null,default:void 0},defaultValue:{type:null,default:void 0},name:{type:String,default:void 0},invalid:{type:Boolean,default:void 0},disabled:{type:Boolean,default:!1},formControl:{type:Object,default:void 0}},inject:{$parentInstance:{default:void 0},$pcForm:{default:void 0},$pcFormField:{default:void 0}},data:function(){return{d_value:this.defaultValue===void 0?this.modelValue:this.defaultValue,d_formDefaultApplied:!1}},watch:{modelValue:{deep:!0,handler:function(e){this.d_value=e}},defaultValue:function(e){this.d_value=e},$formName:{immediate:!0,handler:function(e){var t,n;this.formField=((t=this.$pcForm)==null||(n=t.register)==null?void 0:n.call(t,e,this.$formControl))||{}}},$formControl:{immediate:!0,handler:function(e){var t,n;this.formField=((t=this.$pcForm)==null||(n=t.register)==null?void 0:n.call(t,this.$formName,e))||{}}},$formDefaultValue:{immediate:!0,handler:function(e){!this.d_formDefaultApplied&&q(e)&&(this.d_formDefaultApplied=!0,this.d_value!==e&&(this.d_value=e))}},$formValue:{immediate:!1,handler:function(e){var t;(t=this.$pcForm)!=null&&t.getFieldState(this.$formName)&&e!==this.d_value&&(this.d_value=e)}}},formField:{},methods:{writeValue:function(e,t){var n,r;this.controlled&&(this.d_value=e,this.$emit(`update:modelValue`,e)),this.$emit(`value-change`,e),(n=(r=this.formField).onChange)==null||n.call(r,{originalEvent:t,value:e})},findNonEmpty:function(){return[...arguments].find(q)}},computed:{$filled:function(){return q(this.d_value)},$invalid:function(){var e,t,n=this.$formNovalidate?void 0:this.findNonEmpty((e=this.$pcFormField)==null||(e=e.$field)==null?void 0:e.invalid,(t=this.$pcForm)==null||(t=t.getFieldState(this.$formName))==null?void 0:t.invalid);return this.findNonEmpty(this.invalid,n)},$formName:function(){return this.$formNovalidate?void 0:this.name||this.$formControl?.name},$formControl:function(){return this.formControl||this.$pcFormField?.formControl},$formNovalidate:function(){return this.$formControl?.novalidate},$formDefaultValue:function(){var e,t;return this.findNonEmpty(this.d_value,this.$pcFormField?.initialValue,(e=this.$pcForm)==null||(e=e.initialValues)==null?void 0:e[this.$formName],(t=this.$pcForm)==null||(t=t.getFieldState(this.$formName))==null?void 0:t.value)},$formValue:function(){var e,t;return this.findNonEmpty((e=this.$pcFormField)==null||(e=e.$field)==null?void 0:e.value,(t=this.$pcForm)==null||(t=t.getFieldState(this.$formName))==null?void 0:t.value)},controlled:function(){var e=this,t=function(t){return Object.prototype.hasOwnProperty.call(e.$inProps,t)};return t(`modelValue`)||!t(`modelValue`)&&!t(`defaultValue`)}}},props:{size:{type:String,default:null},fluid:{type:Boolean,default:null},variant:{type:String,default:null}},inject:{$parentInstance:{default:void 0},$pcFluid:{default:void 0}},computed:{$variant:function(){return this.variant??this.$primevue.config.inputVariant},$fluid:function(){return this.fluid??!!this.$pcFluid}}},n_={name:`BaseInputText`,extends:t_,style:Y.extend({name:`inputtext`,style:`\n"
    "    .p-inputtext {\n"
    "        font-weight: dt('inputtext.font.weight');\n"
    "        font-size: dt('inputtext.font.size');\n"
    "        color: dt('inputtext.color');\n"
    "        background: dt('inputtext.background');\n"
    "        padding-block: dt('inputtext.padding.y');\n"
    "        padding-inline: dt('inputtext.padding.x');\n"
    "        border: 1px solid dt('inputtext.border.color');\n"
    "        transition:\n"
    "            background dt('inputtext.transition.duration'),\n"
    "            color dt('inputtext.transition.duration'),\n"
    "            border-color dt('inputtext.transition.duration'),\n"
    "            outline-color dt('inputtext.transition.duration'),\n"
    "            box-shadow dt('inputtext.transition.duration');\n"
    "        appearance: none;\n"
    "        border-radius: dt('inputtext.border.radius');\n"
    "        outline-color: transparent;\n"
    "        box-shadow: dt('inputtext.shadow');\n"
    "    }\n"
    "\n"
    "    .p-inputtext:enabled:hover {\n"
    "        border-color: dt('inputtext.hover.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputtext:enabled:focus {\n"
    "        border-color: dt('inputtext.focus.border.color');\n"
    "        box-shadow: dt('inputtext.focus.ring.shadow');\n"
    "        outline: dt('inputtext.focus.ring.width') dt('inputtext.focus.ring.style') dt('inputtext.focus.ring.color');\n"
    "        outline-offset: dt('inputtext.focus.ring.offset');\n"
    "    }\n"
    "\n"
    "    .p-inputtext.p-invalid {\n"
    "        border-color: dt('inputtext.invalid.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputtext.p-variant-filled {\n"
    "        background: dt('inputtext.filled.background');\n"
    "    }\n"
    "\n"
    "    .p-inputtext.p-variant-filled:enabled:hover {\n"
    "        background: dt('inputtext.filled.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-inputtext.p-variant-filled:enabled:focus {\n"
    "        background: dt('inputtext.filled.focus.background');\n"
    "    }\n"
    "\n"
    "    .p-inputtext:disabled {\n"
    "        opacity: 1;\n"
    "        background: dt('inputtext.disabled.background');\n"
    "        color: dt('inputtext.disabled.color');\n"
    "    }\n"
    "\n"
    "    .p-inputtext::placeholder {\n"
    "        color: dt('inputtext.placeholder.color');\n"
    "    }\n"
    "\n"
    "    .p-inputtext.p-invalid::placeholder {\n"
    "        color: dt('inputtext.invalid.placeholder.color');\n"
    "    }\n"
    "\n"
    "    .p-inputtext-sm {\n"
    "        font-size: dt('inputtext.sm.font.size');\n"
    "        padding-block: dt('inputtext.sm.padding.y');\n"
    "        padding-inline: dt('inputtext.sm.padding.x');\n"
    "    }\n"
    "\n"
    "    .p-inputtext-lg {\n"
    "        font-size: dt('inputtext.lg.font.size');\n"
    "        padding-block: dt('inputtext.lg.padding.y');\n"
    "        padding-inline: dt('inputtext.lg.padding.x');\n"
    "    }\n"
    "\n"
    "    .p-inputtext-fluid {\n"
    "        width: 100%;\n"
    "    }\n"
    "`,classes:{root:function(e){var t=e.instance,n=e.props;return[`p-inputtext p-component`,{\"p-filled\":t.$filled,\"p-inputtext-sm p-inputfield-sm\":n.size===`small`,\"p-inputtext-lg p-inputfield-lg\":n.size===`large`,\"p-invalid\":t.$invalid,\"p-variant-filled\":t.$variant===`filled`,\"p-inputtext-fluid\":t.$fluid}]}}}),provide:function(){return{$pcInputText:this,$parentInstance:this}}};function r_(e){\"@babel/helpers - typeof\";return r_=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},r_(e)}function i_(e,t,n){return(t=a_(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a_(e){var t=o_(e,`string`);return r_(t)==`symbol`?t:t+``}function o_(e,t){if(r_(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(r_(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var s_={name:`InputText`,extends:n_,inheritAttrs:!1,methods:{onInput:function(e){this.writeValue(e.target.value,e)}},computed:{attrs:function(){return U(this.ptmi(`root`,{context:{filled:this.$filled,disabled:this.disabled}}),this.formField)},dataP:function(){return Tl(i_({invalid:this.$invalid,fluid:this.$fluid,filled:this.$variant===`filled`},this.size,this.size))}}},c_=[`value`,`name`,`disabled`,`aria-invalid`,`data-p`];function l_(e,t,n,r,i,a){return L(),R(`input`,U({type:`text`,class:e.cx(`root`),value:e.d_value,name:e.name,disabled:e.disabled,\"aria-invalid\":e.$invalid||void 0,\"data-p\":a.dataP,onInput:t[0]||=function(){return a.onInput&&a.onInput.apply(a,arguments)}},a.attrs),null,16,c_)}s_.render=l_;var u_=[`onKeydown`],d_=[`aria-label`,`aria-describedby`,`aria-expanded`],f_=[`id`],p_=8,m_=4,h_=P({__name:`OuiHint`,props:{label:{},text:{},end:{type:Boolean}},setup(e){let t=e,n=$t(!1),r=fr(),i=$t(),a=$t(),o=$t();function s(){let e=i.value?.getBoundingClientRect(),n=a.value;if(!e||!n)return;let r=document.documentElement.clientWidth,s=document.documentElement.clientHeight,c=n.offsetWidth,l=n.offsetHeight,u=Math.max(p_,Math.min(t.end?e.right+p_-c:e.left-p_,r-p_-c)),d=e.bottom+m_+l<=s-p_||e.top-m_-l<p_;o.value={left:u-e.left+`px`,top:(d?e.height+m_:-(l+m_))+`px`}}async function c(){n.value||(n.value=!0,await wn(),s())}function l(){n.value=!1}return(t,s)=>(L(),R(`span`,{ref_key:`root`,ref:i,class:`oui-hint`,onMouseenter:c,onMouseleave:l,onFocusin:c,onFocusout:l,onKeydown:Es(ws(l,[`stop`]),[`esc`])},[B(`button`,{type:`button`,class:`oui-hint-btn`,\"aria-label\":e.label,\"aria-describedby\":M(r),\"aria-expanded\":n.value,onClick:c},[V(Rm,{icon:M(ch),size:14},null,8,[`icon`])],8,d_),Rn(B(`span`,{id:M(r),ref_key:`bubble`,ref:a,class:`oui-hint-bubble`,role:`tooltip`,style:de(o.value)},[F(t.$slots,`default`,{},()=>[Ta(k(e.text),1)])],12,f_),[[Mo,n.value]])],40,u_))}}),g_=[`data-flush`,`data-fill`],__={key:0,class:`oui-panel-head`},v_={class:`oui-panel-title`},y_={key:2,class:`oui-panel-meta oui-meta`},b_={key:3,class:`oui-panel-actions`},x_={class:`oui-panel-body`},S_={key:1,class:`oui-panel-foot`},C_=P({__name:`OuiPanel`,props:{title:{},icon:{},tone:{},meta:{},hint:{},hintLabel:{},flush:{type:Boolean},fill:{type:Boolean}},setup(e){return(t,n)=>(L(),R(`section`,{class:`oui-panel`,\"data-flush\":e.flush||void 0,\"data-fill\":e.fill||void 0},[e.title||t.$slots.title||t.$slots.actions?(L(),R(`header`,__,[e.icon?(L(),z(Rm,{key:0,icon:e.icon,size:16,tone:e.tone??`neutral`,class:`oui-panel-icon`},null,8,[`icon`,`tone`])):H(``,!0),B(`h2`,v_,[F(t.$slots,`title`,{},()=>[Ta(k(e.title),1)])]),e.hint||t.$slots.hint?(L(),z(h_,{key:1,label:e.hintLabel??e.title??``},{default:N(()=>[F(t.$slots,`hint`,{},()=>[Ta(k(e.hint),1)])]),_:3},8,[`label`])):H(``,!0),e.meta||t.$slots.meta?(L(),R(`span`,y_,[F(t.$slots,`meta`,{},()=>[Ta(k(e.meta),1)])])):H(``,!0),t.$slots.actions?(L(),R(`div`,b_,[F(t.$slots,`actions`)])):H(``,!0)])):H(``,!0),B(`div`,x_,[F(t.$slots,`default`)]),t.$slots.footer?(L(),R(`footer`,S_,[F(t.$slots,`footer`)])):H(``,!0)],8,g_))}}),w_=[`data-compact`,`data-stale`],T_=[`data-thumb`],E_={key:0,class:`oui-job-thumb`},D_={class:`oui-job-id`},O_={key:0,class:`oui-job-status`},k_=[`title`],A_={key:1,class:`oui-job-detail oui-num oui-truncate`},j_={class:`oui-job-progress`},M_={class:`oui-job-figure oui-num`},N_={key:2,\"aria-hidden\":`true`},P_=[`aria-label`,`aria-valuenow`,`data-indeterminate`],F_={key:1,class:`oui-job-stats`},I_={key:2,class:`oui-job-actions`},L_=P({__name:`OuiJob`,props:{name:{},progress:{},progressLabel:{},count:{},detail:{},compact:{type:Boolean},hero:{type:Boolean},stale:{type:Boolean}},setup(e){let t=e,n=W(()=>typeof t.progress==`number`&&Number.isFinite(t.progress)),r=W(()=>n.value?Math.max(0,Math.min(100,t.progress)):0);return(t,i)=>(L(),R(`div`,{class:`oui-job`,\"data-compact\":e.compact||void 0,\"data-stale\":e.stale||void 0},[B(`div`,{class:`oui-job-block`,\"data-thumb\":t.$slots.thumb?!0:void 0},[t.$slots.thumb?(L(),R(`div`,E_,[F(t.$slots,`thumb`)])):H(``,!0),B(`div`,D_,[t.$slots.status?(L(),R(`div`,O_,[F(t.$slots,`status`)])):H(``,!0),B(`p`,{class:`oui-job-name`,title:e.name},k(e.name),9,k_),e.detail?(L(),R(`p`,A_,k(e.detail),1)):H(``,!0)]),B(`div`,j_,[B(`div`,M_,[e.count?(L(),R(I,{key:0},[Ta(k(e.count),1)],64)):n.value?(L(),R(I,{key:1},[Ta(k(Math.round(r.value)),1),i[0]||=B(`span`,null,`%`,-1)],64)):(L(),R(`span`,N_,`—`))]),B(`div`,{class:`oui-job-bar`,role:`progressbar`,\"aria-label\":e.progressLabel??e.name,\"aria-valuemin\":`0`,\"aria-valuemax\":`100`,\"aria-valuenow\":n.value?Math.round(r.value):void 0,\"data-indeterminate\":!n.value||void 0},[B(`span`,{style:de(n.value?{width:r.value+`%`}:void 0)},null,4)],8,P_)]),t.$slots.default?(L(),R(`div`,F_,[F(t.$slots,`default`)])):H(``,!0),t.$slots.actions?(L(),R(`div`,I_,[F(t.$slots,`actions`)])):H(``,!0)],8,T_)],8,w_))}}),R_=[`data-state`,`data-variant`],z_={key:1,class:`oui-status-dot`,\"aria-hidden\":`true`},B_={class:`oui-status-label`},V_=P({__name:`OuiStatus`,props:{state:{},label:{},variant:{}},setup(e){let t={warn:fh,error:ih,offline:uh,stale:nh,pending:nh};return(n,r)=>(L(),R(`span`,{class:`oui-status`,\"data-state\":e.state,\"data-variant\":e.variant??`badge`},[t[e.state]?(L(),z(Rm,{key:0,icon:t[e.state],size:13},null,8,[`icon`])):(L(),R(`span`,z_)),B(`span`,B_,[F(n.$slots,`default`,{},()=>[Ta(k(e.label),1)])])],8,R_))}}),H_=[`data-status`],U_={class:`oui-req-id`},W_={class:`oui-req-name`},G_={key:0,class:`oui-meta`},K_={key:1,class:`oui-meta oui-req-detail`},q_={key:2,class:`oui-req-install`},J_={key:0},Y_=[`href`],X_={class:`oui-req-side`},Z_=[`disabled`],Q_=P({__name:`OuiRequirement`,props:{name:{},why:{},status:{},statusLabel:{},detail:{},installText:{},installUrl:{},installLabel:{},checkLabel:{},checking:{type:Boolean},disabled:{type:Boolean}},emits:[`check`,`open`],setup(e,{emit:t}){let n=e,r=t,i={found:`ok`,missing:`warn`,checking:`busy`,unknown:`idle`},a=!!Fa()?.vnode.props?.onOpen;function o(e){n.installUrl&&a&&(e.preventDefault(),r(`open`,n.installUrl))}return(t,n)=>(L(),R(`div`,{class:`oui-req`,\"data-status\":e.status},[B(`div`,U_,[B(`p`,W_,k(e.name),1),e.why?(L(),R(`p`,G_,k(e.why),1)):H(``,!0),e.detail?(L(),R(`p`,K_,k(e.detail),1)):H(``,!0),e.status===`missing`&&(e.installText||e.installUrl)?(L(),R(`p`,q_,[e.installText?(L(),R(`span`,J_,k(e.installText),1)):H(``,!0),e.installUrl?(L(),R(`a`,{key:1,href:e.installUrl,target:`_blank`,rel:`noopener noreferrer`,class:`oui-req-link`,onClick:o},[V(Rm,{icon:M(lh),size:13},null,8,[`icon`]),Ta(k(e.installLabel??e.installUrl),1)],8,Y_)):H(``,!0)])):H(``,!0)]),B(`div`,X_,[V(V_,{state:i[e.status]},{default:N(()=>[Ta(k(e.statusLabel),1)]),_:1},8,[`state`]),B(`button`,{type:`button`,class:`oui-req-check`,disabled:e.disabled||e.checking||e.status===`checking`,onClick:n[0]||=e=>r(`check`)},[V(Rm,{icon:M(dh),size:14},null,8,[`icon`]),B(`span`,null,k(e.checkLabel),1)],8,Z_)])],8,H_))}}),$_={name:`angle-down`,meta:{tags:[`angle-down`,`fall`,`down`,`decrease`,`lower`]},svg:{xmlns:`http://www.w3.org/2000/svg`,width:20,height:20,viewBox:`0 0 20 20`,fill:`none`},nodes:[[`path`,{d:`M12.9697 7.71973C13.2626 7.42684 13.7374 7.42684 14.0303 7.71973C14.3232 8.01262 14.3232 8.48738 14.0303 8.78028L10.5303 12.2803C10.2374 12.5732 9.76262 12.5732 9.46973 12.2803L5.96973 8.78028C5.67684 8.48738 5.67684 8.01262 5.96973 7.71973C6.26262 7.42684 6.73738 7.42684 7.03028 7.71973L10 10.6895L12.9697 7.71973Z`,fill:`currentColor`,key:`r6am4n`}]]},ev=P({name:`AngleDown`,inheritAttrs:!1,__name:`angle-down`,setup(e){let{Icon:t}=jh($_);return(e,n)=>(L(),z(M(t),_e(Ca(e.$attrs)),null,16))}}),tv={name:`angle-up`,meta:{tags:[`angle-up`,`rise`,`lift`,`up`,`increase`]},svg:{xmlns:`http://www.w3.org/2000/svg`,width:20,height:20,viewBox:`0 0 20 20`,fill:`none`},nodes:[[`path`,{d:`M9.52637 7.66796C9.82095 7.42765 10.2557 7.44512 10.5303 7.71972L14.0303 11.2197C14.3232 11.5126 14.3232 11.9874 14.0303 12.2803C13.7374 12.5732 13.2626 12.5732 12.9697 12.2803L10 9.31054L7.03028 12.2803C6.73738 12.5732 6.26262 12.5732 5.96973 12.2803C5.67684 11.9874 5.67684 11.5126 5.96973 11.2197L9.46973 7.71972L9.52637 7.66796Z`,fill:`currentColor`,key:`sz2v2o`}]]},nv=P({name:`AngleUp`,inheritAttrs:!1,__name:`angle-up`,setup(e){let{Icon:t}=jh(tv);return(e,n)=>(L(),z(M(t),_e(Ca(e.$attrs)),null,16))}}),rv=Y.extend({name:`inputnumber`,style:`\n"
    "    .p-inputnumber {\n"
    "        display: inline-flex;\n"
    "        position: relative;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-button {\n"
    "        display: flex;\n"
    "        align-items: center;\n"
    "        justify-content: center;\n"
    "        flex: 0 0 auto;\n"
    "        cursor: pointer;\n"
    "        background: dt('inputnumber.button.background');\n"
    "        color: dt('inputnumber.button.color');\n"
    "        width: dt('inputnumber.button.width');\n"
    "        transition:\n"
    "            background dt('inputnumber.transition.duration'),\n"
    "            color dt('inputnumber.transition.duration'),\n"
    "            border-color dt('inputnumber.transition.duration'),\n"
    "            outline-color dt('inputnumber.transition.duration');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-button:disabled {\n"
    "        cursor: auto;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-button:not(:disabled):hover {\n"
    "        background: dt('inputnumber.button.hover.background');\n"
    "        color: dt('inputnumber.button.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-button:not(:disabled):active {\n"
    "        background: dt('inputnumber.button.active.background');\n"
    "        color: dt('inputnumber.button.active.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-button {\n"
    "        position: relative;\n"
    "        flex: 1 1 auto;\n"
    "        border: 0 none;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-button-group {\n"
    "        display: flex;\n"
    "        flex-direction: column;\n"
    "        position: absolute;\n"
    "        inset-block-start: 1px;\n"
    "        inset-inline-end: 1px;\n"
    "        height: calc(100% - 2px);\n"
    "        z-index: 1;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-increment-button {\n"
    "        padding: 0;\n"
    "        border-start-end-radius: calc(dt('inputnumber.button.border.radius') - 1px);\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-decrement-button {\n"
    "        padding: 0;\n"
    "        border-end-end-radius: calc(dt('inputnumber.button.border.radius') - 1px);\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-input {\n"
    "        padding-inline-end: calc(dt('inputnumber.button.width') + dt('form.field.padding.x'));\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-button {\n"
    "        border: 1px solid dt('inputnumber.button.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-button:hover {\n"
    "        border-color: dt('inputnumber.button.hover.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-button:active {\n"
    "        border-color: dt('inputnumber.button.active.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-increment-button {\n"
    "        order: 3;\n"
    "        border-start-end-radius: dt('inputnumber.button.border.radius');\n"
    "        border-end-end-radius: dt('inputnumber.button.border.radius');\n"
    "        border-inline-start: 0 none;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-input {\n"
    "        order: 2;\n"
    "        border-radius: 0;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-decrement-button {\n"
    "        order: 1;\n"
    "        border-start-start-radius: dt('inputnumber.button.border.radius');\n"
    "        border-end-start-radius: dt('inputnumber.button.border.radius');\n"
    "        border-inline-end: 0 none;\n"
    "    }\n"
    "\n"
    "    .p-floatlabel:has(.p-inputnumber-horizontal) label {\n"
    "        margin-inline-start: dt('inputnumber.button.width');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical {\n"
    "        flex-direction: column;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-button {\n"
    "        border: 1px solid dt('inputnumber.button.border.color');\n"
    "        padding: dt('inputnumber.button.vertical.padding');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-button:hover {\n"
    "        border-color: dt('inputnumber.button.hover.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-button:active {\n"
    "        border-color: dt('inputnumber.button.active.border.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-increment-button {\n"
    "        order: 1;\n"
    "        border-start-start-radius: dt('inputnumber.button.border.radius');\n"
    "        border-start-end-radius: dt('inputnumber.button.border.radius');\n"
    "        width: 100%;\n"
    "        border-block-end: 0 none;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-input {\n"
    "        order: 2;\n"
    "        border-radius: 0;\n"
    "        text-align: center;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-vertical .p-inputnumber-decrement-button {\n"
    "        order: 3;\n"
    "        border-end-start-radius: dt('inputnumber.button.border.radius');\n"
    "        border-end-end-radius: dt('inputnumber.button.border.radius');\n"
    "        width: 100%;\n"
    "        border-block-start: 0 none;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-input {\n"
    "        flex: 1 1 auto;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-fluid {\n"
    "        width: 100%;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-fluid .p-inputnumber-input {\n"
    "        width: 1%;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-fluid.p-inputnumber-vertical .p-inputnumber-input {\n"
    "        width: 100%;\n"
    "    }\n"
    "\n"
    "    .p-inputnumber:has(.p-inputtext-sm) .p-inputnumber-button .p-icon {\n"
    "        font-size: dt('form.field.sm.font.size');\n"
    "        width: dt('form.field.sm.font.size');\n"
    "        height: dt('form.field.sm.font.size');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber:has(.p-inputtext-lg) .p-inputnumber-button .p-icon {\n"
    "        font-size: dt('form.field.lg.font.size');\n"
    "        width: dt('form.field.lg.font.size');\n"
    "        height: dt('form.field.lg.font.size');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-clear-icon {\n"
    "        position: absolute;\n"
    "        top: 50%;\n"
    "        margin-top: calc(-1 * dt('icon.size') / 2);\n"
    "        cursor: pointer;\n"
    "        inset-inline-end: dt('form.field.padding.x');\n"
    "        color: dt('form.field.icon.color');\n"
    "    }\n"
    "\n"
    "    .p-inputnumber:has(.p-inputnumber-clear-icon) .p-inputnumber-input {\n"
    "        padding-inline-end: calc((dt('form.field.padding.x') * 2) + dt('icon.size'));\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked .p-inputnumber-clear-icon {\n"
    "        inset-inline-end: calc(dt('inputnumber.button.width') + dt('form.field.padding.x'));\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-stacked:has(.p-inputnumber-clear-icon) .p-inputnumber-input {\n"
    "        padding-inline-end: calc(dt('inputnumber.button.width') + (dt('form.field.padding.x') * 2) + dt('icon.size'));\n"
    "    }\n"
    "\n"
    "    .p-inputnumber-horizontal .p-inputnumber-clear-icon {\n"
    "        inset-inline-end: calc(dt('inputnumber.button.width') + dt('form.field.padding.x'));\n"
    "    }\n"
    "`,classes:{root:function(e){var t=e.instance,n=e.props;return[`p-inputnumber p-component p-inputwrapper`,{\"p-invalid\":t.$invalid,\"p-inputwrapper-filled\":t.$filled||n.allowEmpty===!1,\"p-inputwrapper-focus\":t.focused,\"p-inputnumber-stacked\":n.showButtons&&n.buttonLayout===`stacked`,\"p-inputnumber-horizontal\":n.showButtons&&n.buttonLayout===`horizontal`,\"p-inputnumber-vertical\":n.showButtons&&n.buttonLayout===`vertical`,\"p-inputnumber-fluid\":t.$fluid}]},pcInputText:`p-inputnumber-input`,clearIcon:`p-inputnumber-clear-icon`,buttonGroup:`p-inputnumber-button-group`,incrementButton:function(e){var t=e.instance,n=e.props;return[`p-inputnumber-button p-inputnumber-increment-button`,{\"p-disabled\":n.showButtons&&n.max!==null&&t.maxBoundry()}]},decrementButton:function(e){var t=e.instance,n=e.props;return[`p-inputnumber-button p-inputnumber-decrement-button`,{\"p-disabled\":n.showButtons&&n.min!==null&&t.minBoundry()}]}}}),iv={name:`BaseInputNumber`,extends:t_,props:{format:{type:Boolean,default:!0},showButtons:{type:Boolean,default:!1},buttonLayout:{type:String,default:`stacked`},incrementButtonClass:{type:String,default:null},decrementButtonClass:{type:String,default:null},incrementIcon:{type:String,default:void 0},decrementIcon:{type:String,default:void 0},locale:{type:String,default:void 0},localeMatcher:{type:String,default:void 0},mode:{type:String,default:`decimal`},prefix:{type:String,default:null},suffix:{type:String,default:null},currency:{type:String,default:void 0},currencyDisplay:{type:String,default:void 0},useGrouping:{type:Boolean,default:!0},minFractionDigits:{type:Number,default:void 0},maxFractionDigits:{type:Number,default:void 0},roundingMode:{type:String,default:`halfExpand`,validator:function(e){return[`ceil`,`floor`,`expand`,`trunc`,`halfCeil`,`halfFloor`,`halfExpand`,`halfTrunc`,`halfEven`].includes(e)}},min:{type:Number,default:null},max:{type:Number,default:null},step:{type:Number,default:1},allowEmpty:{type:Boolean,default:!0},highlightOnFocus:{type:Boolean,default:!1},showClear:{type:Boolean,default:!1},readonly:{type:Boolean,default:!1},placeholder:{type:String,default:null},inputId:{type:String,default:null},inputClass:{type:[String,Object],default:null},inputStyle:{type:Object,default:null},ariaLabelledby:{type:String,default:null},ariaLabel:{type:String,default:null},required:{type:Boolean,default:!1}},style:rv,provide:function(){return{$pcInputNumber:this,$parentInstance:this}}};function av(e){\"@babel/helpers - typeof\";return av=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},av(e)}function ov(e,t){var n=typeof Symbol<`u`&&e[Symbol.iterator]||e[`@@iterator`];if(!n){if(Array.isArray(e)||(n=mv(e))||t){n&&(e=n);var r=0,i=function(){};return{s:i,n:function(){return r>=e.length?{done:!0}:{done:!1,value:e[r++]}},e:function(e){throw e},f:i}}throw TypeError(`Invalid attempt to iterate non-iterable instance.\n"
    "In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}var a,o=!0,s=!1;return{s:function(){n=n.call(e)},n:function(){var e=n.next();return o=e.done,e},e:function(e){s=!0,a=e},f:function(){try{o||n.return==null||n.return()}finally{if(s)throw a}}}}function sv(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function cv(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?sv(Object(n),!0).forEach(function(t){lv(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):sv(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function lv(e,t,n){return(t=uv(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function uv(e){var t=dv(e,`string`);return av(t)==`symbol`?t:t+``}function dv(e,t){if(av(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(av(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}function fv(e){return gv(e)||hv(e)||mv(e)||pv()}function pv(){throw TypeError(`Invalid attempt to spread non-iterable instance.\n"
    "In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function mv(e,t){if(e){if(typeof e==`string`)return _v(e,t);var n={}.toString.call(e).slice(8,-1);return n===`Object`&&e.constructor&&(n=e.constructor.name),n===`Map`||n===`Set`?Array.from(e):n===`Arguments`||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?_v(e,t):void 0}}function hv(e){if(typeof Symbol<`u`&&e[Symbol.iterator]!=null||e[`@@iterator`]!=null)return Array.from(e)}function gv(e){if(Array.isArray(e))return _v(e)}function _v(e,t){(t==null||t>e.length)&&(t=e.length);for(var n=0,r=Array(t);n<t;n++)r[n]=e[n];return r}var vv={name:`InputNumber`,extends:iv,inheritAttrs:!1,emits:[`input`,`focus`,`blur`],inject:{$pcFluid:{default:null}},numberFormat:null,_numeral:null,_decimal:null,_group:null,_minusSign:null,_currency:null,_suffix:null,_prefix:null,_index:null,groupChar:``,isSpecialChar:null,compositionValue:null,compositionStart:null,compositionEnd:null,prefixChar:null,suffixChar:null,timer:null,data:function(){return{focused:!1}},watch:{d_value:{immediate:!0,handler:function(e){var t;(t=this.$refs.clearIcon)!=null&&(t=t.$el)!=null&&t.style&&(this.$refs.clearIcon.$el.style.display=el(e)?`none`:`block`)}},locale:function(e,t){this.updateConstructParser(e,t)},localeMatcher:function(e,t){this.updateConstructParser(e,t)},mode:function(e,t){this.updateConstructParser(e,t)},currency:function(e,t){this.updateConstructParser(e,t)},currencyDisplay:function(e,t){this.updateConstructParser(e,t)},useGrouping:function(e,t){this.updateConstructParser(e,t)},minFractionDigits:function(e,t){this.updateConstructParser(e,t)},maxFractionDigits:function(e,t){this.updateConstructParser(e,t)},suffix:function(e,t){this.updateConstructParser(e,t)},prefix:function(e,t){this.updateConstructParser(e,t)}},created:function(){this.constructParser()},mounted:function(){var e;(e=this.$refs.clearIcon)!=null&&(e=e.$el)!=null&&e.style&&(this.$refs.clearIcon.$el.style.display=this.$filled?`block`:`none`)},methods:{getOptions:function(){return{localeMatcher:this.localeMatcher,style:this.mode,currency:this.currency,currencyDisplay:this.currencyDisplay,useGrouping:this.useGrouping,minimumFractionDigits:this.minFractionDigits,maximumFractionDigits:this.maxFractionDigits,roundingMode:this.roundingMode}},constructParser:function(){this.numberFormat=new Intl.NumberFormat(this.locale,this.getOptions());var e=fv(new Intl.NumberFormat(this.locale,{useGrouping:!1}).format(9876543210)).reverse(),t=new Map(e.map(function(e,t){return[e,t]}));this._numeral=RegExp(`[${e.join(``)}]`,`g`),this._group=this.getGroupingExpression(),this._minusSign=this.getMinusSignExpression(),this._currency=this.getCurrencyExpression(),this._decimal=this.getDecimalExpression(),this._suffix=this.getSuffixExpression(),this._prefix=this.getPrefixExpression(),this._index=function(e){return t.get(e)}},updateConstructParser:function(e,t){e!==t&&this.constructParser()},escapeRegExp:function(e){return e.replace(/[-[\\]{}()*+?.,\\\\^$|#\\s]/g,`\\\\$&`)},getDecimalExpression:function(){var e=new Intl.NumberFormat(this.locale,cv(cv({},this.getOptions()),{},{useGrouping:!1})),t=e.format(1.1);return t===e.format(1)?RegExp(`[]`,`g`):RegExp(`[${t.replace(this._currency,``).trim().replace(this._numeral,``)}]`,`g`)},getGroupingExpression:function(){var e=new Intl.NumberFormat(this.locale,{useGrouping:!0});return this.groupChar=e.format(1e6).trim().replace(this._numeral,``).charAt(0),RegExp(`[${this.groupChar}]`,`g`)},getMinusSignExpression:function(){var e=new Intl.NumberFormat(this.locale,{useGrouping:!1});return RegExp(`[${e.format(-1).trim().replace(this._numeral,``)}]`,`g`)},getCurrencyExpression:function(){if(this.currency){var e=new Intl.NumberFormat(this.locale,{style:`currency`,currency:this.currency,currencyDisplay:this.currencyDisplay,minimumFractionDigits:0,maximumFractionDigits:0,roundingMode:this.roundingMode});return RegExp(`[${e.format(1).replace(/\\s/g,``).replace(this._numeral,``).replace(this._group,``)}]`,`g`)}return RegExp(`[]`,`g`)},getPrefixExpression:function(){if(this.prefix)this.prefixChar=this.prefix;else{var e=new Intl.NumberFormat(this.locale,{style:this.mode,currency:this.currency,currencyDisplay:this.currencyDisplay});this.prefixChar=e.format(1).split(`1`)[0]}return RegExp(`${this.escapeRegExp(this.prefixChar||``)}`,`g`)},getSuffixExpression:function(){if(this.suffix)this.suffixChar=this.suffix;else{var e=new Intl.NumberFormat(this.locale,{style:this.mode,currency:this.currency,currencyDisplay:this.currencyDisplay,minimumFractionDigits:0,maximumFractionDigits:0,roundingMode:this.roundingMode});this.suffixChar=e.format(1).split(`1`)[1]}return RegExp(`${this.escapeRegExp(this.suffixChar||``)}`,`g`)},formatValue:function(e){if(e!=null){if(e===`-`)return e;if(this.format){var t=new Intl.NumberFormat(this.locale,this.getOptions()).format(e);if(this.prefix){if(e<0){var n=new Intl.NumberFormat(this.locale,{useGrouping:!1}).format(-1).trim().replace(this._numeral,``)||`-`;t=t.replace(this._minusSign,``),this.resetRegex(),t=n+this.prefix+t}else t=this.prefix+t}return this.suffix&&(t+=this.suffix),t}return e.toString()}return``},parseValue:function(e){var t=e.replace(this._suffix,``).replace(this._prefix,``).trim().replace(/\\s/g,``).replace(this._currency,``).replace(this._group,``).replace(this._minusSign,`-`).replace(this._decimal,`.`).replace(this._numeral,this._index);if(t){if(t===`-`)return t;var n=+t;return isNaN(n)?null:n}return null},repeat:function(e,t,n){var r=this;if(!this.readonly){var i=t||500;this.clearTimer(),this.timer=setTimeout(function(){r.repeat(e,40,n)},i),this.spin(e,n)}},addWithPrecision:function(e,t){var n=e.toString(),r=t.toString(),i=n.includes(`.`)?n.split(`.`)[1].length:0,a=r.includes(`.`)?r.split(`.`)[1].length:0,o=10**Math.max(i,a);return Math.round((e+t)*o)/o},spin:function(e,t){if(this.$refs.input){var n=this.step*t,r=this.parseValue(this.$refs.input.$el.value)||0,i=this.validateValue(this.addWithPrecision(r,n));this.updateInput(i,null,`spin`),this.updateModel(e,i),this.handleOnInput(e,r,i)}},onUpButtonMouseDown:function(e){this.disabled||(this.$refs.input.$el.focus(),this.repeat(e,null,1),e.preventDefault())},onUpButtonMouseUp:function(){this.disabled||this.clearTimer()},onUpButtonMouseLeave:function(){this.disabled||this.clearTimer()},onUpButtonKeyUp:function(){this.disabled||this.clearTimer()},onUpButtonKeyDown:function(e){(e.code===`Space`||e.code===`Enter`||e.code===`NumpadEnter`)&&this.repeat(e,null,1)},onDownButtonMouseDown:function(e){this.disabled||(this.$refs.input.$el.focus(),this.repeat(e,null,-1),e.preventDefault())},onDownButtonMouseUp:function(){this.disabled||this.clearTimer()},onDownButtonMouseLeave:function(){this.disabled||this.clearTimer()},onDownButtonKeyUp:function(){this.disabled||this.clearTimer()},onDownButtonKeyDown:function(e){(e.code===`Space`||e.code===`Enter`||e.code===`NumpadEnter`)&&this.repeat(e,null,-1)},onUserInput:function(){this.isSpecialChar&&(this.$refs.input.$el.value=this.lastValue),this.isSpecialChar=!1},onInputCompositionStart:function(e){this.compositionValue=e.target.value,this.compositionStart=e.target.selectionStart,this.compositionEnd=e.target.selectionEnd},onInputCompositionEnd:function(e){if(!(this.readonly||this.disabled)){var t=e.data||``;e.target.value=this.compositionValue??``,e.target.setSelectionRange(this.compositionStart,this.compositionEnd);var n=ov(t),r;try{for(n.s();!(r=n.n()).done;){var i=r.value;this.isNumeralChar(i)&&this.insert(e,i,{isDecimalSign:this.isDecimalSign(i),isMinusSign:this.isMinusSign(i)})}}catch(e){n.e(e)}finally{n.f()}}},onInputKeyDown:function(e){if(!this.readonly&&!e.isComposing){if(e.altKey||e.ctrlKey||e.metaKey)this.isSpecialChar=!0,this.lastValue=this.$refs.input.$el.value;else{this.lastValue=e.target.value;var t=e.target.selectionStart,n=e.target.selectionEnd,r=n-t,i=e.target.value,a=null;switch(e.code||e.key){case`ArrowUp`:this.spin(e,1),e.preventDefault();break;case`ArrowDown`:this.spin(e,-1),e.preventDefault();break;case`ArrowLeft`:if(r>1){var o=this.isNumeralChar(i.charAt(t))?t+1:t+2;this.$refs.input.$el.setSelectionRange(o,o)}else this.isNumeralChar(i.charAt(t-1))||e.preventDefault();break;case`ArrowRight`:if(r>1){var s=n-1;this.$refs.input.$el.setSelectionRange(s,s)}else this.isNumeralChar(i.charAt(t))||e.preventDefault();break;case`Tab`:case`Enter`:case`NumpadEnter`:a=this.validateValue(this.parseValue(i)),this.$refs.input.$el.value=this.formatValue(a),this.$refs.input.$el.setAttribute(`aria-valuenow`,a),this.updateModel(e,a);break;case`Backspace`:if(e.preventDefault(),t===n){t>=i.length&&this.suffixChar!==null&&(t=i.length-this.suffixChar.length,this.$refs.input.$el.setSelectionRange(t,t));var c=i.charAt(t-1),l=this.getDecimalCharIndexes(i),u=l.decimalCharIndex,d=l.decimalCharIndexWithoutPrefix;if(this.isNumeralChar(c)){var f=this.getDecimalLength(i);if(this._group.test(c))this._group.lastIndex=0,a=i.slice(0,t-2)+i.slice(t-1);else if(this._decimal.test(c))this._decimal.lastIndex=0,f?this.$refs.input.$el.setSelectionRange(t-1,t-1):a=i.slice(0,t-1)+i.slice(t);else if(u>0&&t>u){var p=this.isDecimalMode()&&(this.minFractionDigits||0)<f?``:`0`;a=i.slice(0,t-1)+p+i.slice(t)}else d===1?(a=i.slice(0,t-1)+`0`+i.slice(t),a=this.parseValue(a)>0?a:``):a=i.slice(0,t-1)+i.slice(t)}this.updateValue(e,a,null,`delete-single`)}else a=this.deleteRange(i,t,n),this.updateValue(e,a,null,`delete-range`);break;case`Delete`:if(e.preventDefault(),t===n){var m=i.charAt(t),h=this.getDecimalCharIndexes(i),g=h.decimalCharIndex,_=h.decimalCharIndexWithoutPrefix;if(this.isNumeralChar(m)){var v=this.getDecimalLength(i);if(this._group.test(m))this._group.lastIndex=0,a=i.slice(0,t)+i.slice(t+2);else if(this._decimal.test(m))this._decimal.lastIndex=0,v?this.$refs.input.$el.setSelectionRange(t+1,t+1):a=i.slice(0,t)+i.slice(t+1);else if(g>0&&t>g){var y=this.isDecimalMode()&&(this.minFractionDigits||0)<v?``:`0`;a=i.slice(0,t)+y+i.slice(t+1)}else _===1?(a=i.slice(0,t)+`0`+i.slice(t+1),a=this.parseValue(a)>0?a:``):a=i.slice(0,t)+i.slice(t+1)}this.updateValue(e,a,null,`delete-back-single`)}else a=this.deleteRange(i,t,n),this.updateValue(e,a,null,`delete-range`);break;case`Home`:e.preventDefault(),q(this.min)&&this.updateModel(e,this.min);break;case`End`:e.preventDefault(),q(this.max)&&this.updateModel(e,this.max)}}}},onInputKeyPress:function(e){if(!this.readonly){var t=e.key,n=this.isDecimalSign(t),r=this.isMinusSign(t);e.code!==`Enter`&&e.preventDefault(),(Number(t)>=0&&Number(t)<=9||r||n)&&this.insert(e,t,{isDecimalSign:n,isMinusSign:r})}},onPaste:function(e){if(!(this.readonly||this.disabled)){e.preventDefault();var t=(e.clipboardData||window.clipboardData).getData(`Text`);if(!(this.inputId===`integeronly`&&/[^\\d-]/.test(t))&&t){var n=this.parseValue(t);n!=null&&this.insert(e,n.toString())}}},onClearClick:function(e){this.updateModel(e,null),this.$refs.input.$el.focus()},allowMinusSign:function(){return this.min===null||this.min<0},isMinusSign:function(e){return this._minusSign.test(e)||e===`-`?(this._minusSign.lastIndex=0,!0):!1},isDecimalSign:function(e){var t;return(t=this.locale)!=null&&t.includes(`fr`)&&[`.`,`,`].includes(e)||this._decimal.test(e)?(this._decimal.lastIndex=0,!0):!1},isDecimalMode:function(){return this.mode===`decimal`},getDecimalCharIndexes:function(e){var t=e.search(this._decimal);this._decimal.lastIndex=0;var n=e.replace(this._prefix,``).trim().replace(/\\s/g,``).replace(this._currency,``).search(this._decimal);return this._decimal.lastIndex=0,{decimalCharIndex:t,decimalCharIndexWithoutPrefix:n}},getCharIndexes:function(e){var t=e.search(this._decimal);this._decimal.lastIndex=0;var n=e.search(this._minusSign);this._minusSign.lastIndex=0;var r=e.search(this._suffix);this._suffix.lastIndex=0;var i=e.search(this._currency);return this._currency.lastIndex=0,{decimalCharIndex:t,minusCharIndex:n,suffixCharIndex:r,currencyCharIndex:i}},insert:function(e,t){var n=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{isDecimalSign:!1,isMinusSign:!1},r=t.search(this._minusSign);if(this._minusSign.lastIndex=0,this.allowMinusSign()||r===-1){var i=this.$refs.input.$el.selectionStart,a=this.$refs.input.$el.selectionEnd,o=this.$refs.input.$el.value.trim(),s=this.getCharIndexes(o),c=s.decimalCharIndex,l=s.minusCharIndex,u=s.suffixCharIndex,d=s.currencyCharIndex,f;if(n.isMinusSign){var p=l===-1,m=(this.prefixChar||``).length,h=m>0&&i===m;(i===0||i===d+1||h)&&(f=o,(p||a!==0)&&(f=this.insertText(o,t,0,a)),this.updateValue(e,f,t,`insert`))}else if(n.isDecimalSign)c>0&&i===c?this.updateValue(e,o,t,`insert`):(c>i&&c<a||c===-1&&this.maxFractionDigits)&&(f=this.insertText(o,t,i,a),this.updateValue(e,f,t,`insert`));else{var g=this.numberFormat.resolvedOptions().maximumFractionDigits,_=i===a?`insert`:`range-insert`;if(c>0&&i>c){if(i+t.length-(c+1)<=g){var v=d>=i?d-1:u>=i?u:o.length;f=o.slice(0,i)+t+o.slice(i+t.length,v)+o.slice(v),this.updateValue(e,f,t,_)}}else f=this.insertText(o,t,i,a),this.updateValue(e,f,t,_)}}},insertText:function(e,t,n,r){if((t===`.`?t:t.split(`.`)).length===2){var i=e.slice(n,r).search(this._decimal);return this._decimal.lastIndex=0,i>0?e.slice(0,n)+this.formatValue(t)+e.slice(r):this.formatValue(t)||e}return r-n===e.length?this.formatValue(t):n===0?t+e.slice(r):r===e.length?e.slice(0,n)+t:e.slice(0,n)+t+e.slice(r)},deleteRange:function(e,t,n){return n-t===e.length?``:t===0?e.slice(n):n===e.length?e.slice(0,t):e.slice(0,t)+e.slice(n)},initCursor:function(){var e=this.$refs.input.$el.selectionStart,t=this.$refs.input.$el.value,n=t.length,r=null,i=(this.prefixChar||``).length;t=t.replace(this._prefix,``),e-=i;var a=t.charAt(e);if(this.isNumeralChar(a))return e+i;for(var o=e-1;o>=0;){if(a=t.charAt(o),this.isNumeralChar(a)){r=o+i;break}o--}if(r!==null)this.$refs.input.$el.setSelectionRange(r+1,r+1);else{for(o=e;o<n;){if(a=t.charAt(o),this.isNumeralChar(a)){r=o+i;break}o++}r!==null&&this.$refs.input.$el.setSelectionRange(r,r)}return r||0},onInputClick:function(){var e=this.$refs.input.$el.value;!this.readonly&&e!==fu()&&this.initCursor()},isNumeralChar:function(e){return e.length===1&&(this._numeral.test(e)||this._decimal.test(e)||this._group.test(e)||this._minusSign.test(e))?(this.resetRegex(),!0):!1},resetRegex:function(){this._numeral.lastIndex=0,this._decimal.lastIndex=0,this._group.lastIndex=0,this._minusSign.lastIndex=0},updateValue:function(e,t,n,r){var i=this.$refs.input.$el.value,a;t!=null&&(a=this.parseValue(t),a=!a&&!this.allowEmpty?0:a,this.updateInput(a,n,r,t),this.handleOnInput(e,i,a))},handleOnInput:function(e,t,n){if(this.isValueChanged(t,n)){var r,i;this.$emit(`input`,{originalEvent:e,value:n,formattedValue:t}),(r=(i=this.formField).onInput)==null||r.call(i,{originalEvent:e,value:n})}},isValueChanged:function(e,t){return t===null&&e!==null||t!=null&&t!==(typeof e==`string`?this.parseValue(e):e)},validateValue:function(e){return e===`-`||e==null?null:this.min!=null&&e<this.min?this.min:this.max!=null&&e>this.max?this.max:e},updateInput:function(e,t,n,r){var i;t||=``;var a=this.$refs.input.$el.value,o=this.formatValue(e),s=a.length;if(o!==r&&(o=this.concatValues(o,r)),s===0){this.$refs.input.$el.value=o,this.$refs.input.$el.setSelectionRange(0,0);var c=this.initCursor()+t.length;this.$refs.input.$el.setSelectionRange(c,c)}else{var l=this.$refs.input.$el.selectionStart,u=this.$refs.input.$el.selectionEnd;this.$refs.input.$el.value=o;var d=o.length;if(n===`range-insert`){var f=this.parseValue((a||``).slice(0,l)),p=(f===null?``:f.toString()).split(``).join(`(${this.groupChar})?`),m=new RegExp(p,`g`);m.test(o);var h=t.split(``).join(`(${this.groupChar})?`),g=new RegExp(h,`g`);g.test(o.slice(m.lastIndex)),u=m.lastIndex+g.lastIndex,this.$refs.input.$el.setSelectionRange(u,u)}else if(d===s)n===`insert`||n===`delete-back-single`?this.$refs.input.$el.setSelectionRange(u+1,u+1):n===`delete-single`?this.$refs.input.$el.setSelectionRange(u-1,u-1):(n===`delete-range`||n===`spin`)&&this.$refs.input.$el.setSelectionRange(u,u);else if(n===`delete-back-single`){var _=a.charAt(u-1),v=a.charAt(u),y=s-d,b=this._group.test(v);b&&y===1?u+=1:!b&&this.isNumeralChar(_)&&(u+=-1*y+1),this._group.lastIndex=0,this.$refs.input.$el.setSelectionRange(u,u)}else if(a===`-`&&n===`insert`){this.$refs.input.$el.setSelectionRange(0,0);var x=this.initCursor()+t.length+1;this.$refs.input.$el.setSelectionRange(x,x)}else u+=d-s,this.$refs.input.$el.setSelectionRange(u,u)}this.$refs.input.$el.setAttribute(`aria-valuenow`,e),(i=this.$refs.clearIcon)!=null&&(i=i.$el)!=null&&i.style&&(this.$refs.clearIcon.$el.style.display=el(o)?`none`:`block`)},concatValues:function(e,t){if(e&&t){var n=t.search(this._decimal);return this._decimal.lastIndex=0,this.suffixChar?n===-1?e:e.replace(this.suffixChar,``).split(this._decimal)[0]+t.replace(this.suffixChar,``).slice(n)+this.suffixChar:n===-1?e:e.split(this._decimal)[0]+t.slice(n)}return e},getDecimalLength:function(e){if(e){var t=e.split(this._decimal);if(t.length===2)return t[1].replace(this._suffix,``).trim().replace(/\\s/g,``).replace(this._currency,``).length}return 0},updateModel:function(e,t){this.writeValue(t,e)},onInputFocus:function(e){this.focused=!0,!this.disabled&&!this.readonly&&this.$refs.input.$el.value!==fu()&&this.highlightOnFocus&&e.target.select(),this.$emit(`focus`,e)},onInputBlur:function(e){var t,n;this.focused=!1;var r=e.target,i=this.validateValue(this.parseValue(r.value));this.$emit(`blur`,{originalEvent:e,value:r.value}),(t=(n=this.formField).onBlur)==null||t.call(n,e),r.value=this.formatValue(i),r.setAttribute(`aria-valuenow`,i),this.updateModel(e,i),!this.disabled&&!this.readonly&&this.highlightOnFocus&&nu()},clearTimer:function(){this.timer&&clearTimeout(this.timer)},maxBoundry:function(){return this.d_value>=this.max},minBoundry:function(){return this.d_value<=this.min}},computed:{upButtonListeners:function(){var e=this;return{mousedown:function(t){return e.onUpButtonMouseDown(t)},mouseup:function(t){return e.onUpButtonMouseUp(t)},mouseleave:function(t){return e.onUpButtonMouseLeave(t)},keydown:function(t){return e.onUpButtonKeyDown(t)},keyup:function(t){return e.onUpButtonKeyUp(t)}}},downButtonListeners:function(){var e=this;return{mousedown:function(t){return e.onDownButtonMouseDown(t)},mouseup:function(t){return e.onDownButtonMouseUp(t)},mouseleave:function(t){return e.onDownButtonMouseLeave(t)},keydown:function(t){return e.onDownButtonKeyDown(t)},keyup:function(t){return e.onDownButtonKeyUp(t)}}},formattedValue:function(){var e=!this.d_value&&!this.allowEmpty?0:this.d_value;return this.formatValue(e)},getFormatter:function(){return this.numberFormat},dataP:function(){return Tl(lv(lv({invalid:this.$invalid,fluid:this.$fluid,filled:this.$variant===`filled`},this.size,this.size),this.buttonLayout,this.showButtons&&this.buttonLayout))}},components:{InputText:s_,AngleUp:nv,AngleDown:ev,Times:e_}},yv=[`data-p`],bv=[`data-p`],xv=[`disabled`,`data-p`],Sv=[`disabled`,`data-p`],Cv=[`disabled`,`data-p`],wv=[`disabled`,`data-p`];function Tv(e,t,n,r,i,a){var o=Rr(`InputText`),s=Rr(`Times`);return L(),R(`span`,U({class:e.cx(`root`)},e.ptmi(`root`),{\"data-p\":a.dataP}),[V(o,{ref:`input`,id:e.inputId,name:e.$formName,formControl:{novalidate:!0},role:`spinbutton`,class:ge([e.cx(`pcInputText`),e.inputClass]),style:de(e.inputStyle),defaultValue:a.formattedValue,\"aria-valuemin\":e.min,\"aria-valuemax\":e.max,\"aria-valuenow\":e.d_value,inputmode:e.mode===`decimal`&&!e.minFractionDigits?`numeric`:`decimal`,disabled:e.disabled,readonly:e.readonly,placeholder:e.placeholder,\"aria-labelledby\":e.ariaLabelledby,\"aria-label\":e.ariaLabel,required:e.required,size:e.size,invalid:e.$invalid,variant:e.variant,onInput:a.onUserInput,onKeydown:a.onInputKeyDown,onKeypress:a.onInputKeyPress,onCompositionstart:a.onInputCompositionStart,onCompositionend:a.onInputCompositionEnd,onPaste:a.onPaste,onClick:a.onInputClick,onFocus:a.onInputFocus,onBlur:a.onInputBlur,pt:e.ptm(`pcInputText`),unstyled:e.unstyled,\"data-p\":a.dataP},null,8,`id.name.class.style.defaultValue.aria-valuemin.aria-valuemax.aria-valuenow.inputmode.disabled.readonly.placeholder.aria-labelledby.aria-label.required.size.invalid.variant.onInput.onKeydown.onKeypress.onCompositionstart.onCompositionend.onPaste.onClick.onFocus.onBlur.pt.unstyled.data-p`.split(`.`)),e.showClear&&e.buttonLayout!==`vertical`?F(e.$slots,`clearicon`,{class:ge(e.cx(`clearIcon`)),clearCallback:a.onClearClick},function(){return[V(s,U({ref:`clearIcon`,class:[e.cx(`clearIcon`)],onClick:a.onClearClick},e.ptm(`clearIcon`)),null,16,[`class`,`onClick`])]},void 0,0):H(``,!0),e.showButtons&&e.buttonLayout===`stacked`?(L(),R(`span`,U({key:1,class:e.cx(`buttonGroup`)},e.ptm(`buttonGroup`),{\"data-p\":a.dataP}),[F(e.$slots,`incrementbutton`,{listeners:a.upButtonListeners},function(){return[B(`button`,U({class:[e.cx(`incrementButton`),e.incrementButtonClass]},qr(a.upButtonListeners,!0),{disabled:e.disabled,tabindex:-1,\"aria-hidden\":`true`,type:`button`},e.ptm(`incrementButton`),{\"data-p\":a.dataP}),[F(e.$slots,`incrementicon`,{},function(){return[(L(),z(Br(e.incrementIcon?`span`:`AngleUp`),U({class:e.incrementIcon},e.ptm(`incrementIcon`),{\"data-pc-section\":`incrementicon`}),null,16,[`class`]))]})],16,xv)]}),F(e.$slots,`decrementbutton`,{listeners:a.downButtonListeners},function(){return[B(`button`,U({class:[e.cx(`decrementButton`),e.decrementButtonClass]},qr(a.downButtonListeners,!0),{disabled:e.disabled,tabindex:-1,\"aria-hidden\":`true`,type:`button`},e.ptm(`decrementButton`),{\"data-p\":a.dataP}),[F(e.$slots,`decrementicon`,{},function(){return[(L(),z(Br(e.decrementIcon?`span`:`AngleDown`),U({class:e.decrementIcon},e.ptm(`decrementIcon`),{\"data-pc-section\":`decrementicon`}),null,16,[`class`]))]})],16,Sv)]})],16,bv)):H(``,!0),F(e.$slots,`incrementbutton`,{listeners:a.upButtonListeners},function(){return[e.showButtons&&e.buttonLayout!==`stacked`?(L(),R(`button`,U({key:0,class:[e.cx(`incrementButton`),e.incrementButtonClass]},qr(a.upButtonListeners,!0),{disabled:e.disabled,tabindex:-1,\"aria-hidden\":`true`,type:`button`},e.ptm(`incrementButton`),{\"data-p\":a.dataP}),[F(e.$slots,`incrementicon`,{},function(){return[(L(),z(Br(e.incrementIcon?`span`:`AngleUp`),U({class:e.incrementIcon},e.ptm(`incrementIcon`),{\"data-pc-section\":`incrementicon`}),null,16,[`class`]))]})],16,Cv)):H(``,!0)]}),F(e.$slots,`decrementbutton`,{listeners:a.downButtonListeners},function(){return[e.showButtons&&e.buttonLayout!==`stacked`?(L(),R(`button`,U({key:0,class:[e.cx(`decrementButton`),e.decrementButtonClass]},qr(a.downButtonListeners,!0),{disabled:e.disabled,tabindex:-1,\"aria-hidden\":`true`,type:`button`},e.ptm(`decrementButton`),{\"data-p\":a.dataP}),[F(e.$slots,`decrementicon`,{},function(){return[(L(),z(Br(e.decrementIcon?`span`:`AngleDown`),U({class:e.decrementIcon},e.ptm(`decrementIcon`),{\"data-pc-section\":`decrementicon`}),null,16,[`class`]))]})],16,wv)):H(``,!0)]})],16,yv)}vv.render=Tv;var Ev={info:5e3,success:5e3,warn:8e3,error:0},Dv=Vt({list:[]}),Ov=new Map,kv=new Map,Av=new Set,jv=1,Mv=typeof document<`u`,Nv=()=>!Mv||document.visibilityState!==`hidden`&&(typeof document.hasFocus!=`function`||document.hasFocus()),Pv=Nv();function Fv(e){let t=Ov.get(e),n=kv.get(e);t&&n&&(clearTimeout(t),Ov.delete(e),n.ms=Math.max(1e3,n.ms-(Date.now()-n.since)))}function Iv(e){let t=kv.get(e);t&&!Ov.has(e)&&!Av.has(e)&&Pv&&(t.since=Date.now(),Ov.set(e,setTimeout(()=>Rv(e),t.ms)))}function Lv(){let e=Nv();if(e!==Pv){Pv=e;for(let e of[...kv.keys()])(Pv?Iv:Fv)(e)}}Mv&&typeof window<`u`&&(document.addEventListener(`visibilitychange`,Lv),window.addEventListener(`focus`,Lv),window.addEventListener(`blur`,Lv),window.addEventListener(`pointerdown`,Lv,!0),window.addEventListener(`keydown`,Lv,!0));function Rv(e){clearTimeout(Ov.get(e)),Ov.delete(e),kv.delete(e),Av.delete(e),Dv.list=Dv.list.filter(t=>t.id!==e)}function zv(e){Av.add(e),Fv(e)}function Bv(e){Av.delete(e),Iv(e)}function Vv(e){let t=e.tone??`info`,n=t===`error`?0:e.timeout??(e.action?Math.max(8e3,Ev[t]):Ev[t]),r=e.key?Dv.list.find(t=>t.key===e.key):void 0;for(r&&Rv(r.id);Dv.list.length>=4;)Rv((Dv.list.find(e=>e.tone!==`error`&&!e.action)??Dv.list[0]).id);let i=jv++;return Dv.list=[...Dv.list,{id:i,tone:t,title:e.title,body:e.body,action:e.action,key:e.key,timeout:n}],n>0&&(kv.set(i,{ms:n,since:Date.now()}),Iv(i)),i}var Hv=Ut(Dv),Uv=[`aria-label`],Wv=[`data-toast-id`,`data-tone`,`data-toast-tone`,`role`,`aria-live`,`onMouseenter`,`onMouseleave`,`onFocusin`,`onFocusout`],Gv={class:`oui-toast-text`},Kv={key:0,class:`oui-toast-body`},qv=[`onClick`],Jv=[`aria-label`,`title`,`onClick`],Yv=P({__name:`OuiToastHost`,props:{closeLabel:{},label:{}},setup(e){let t=e,n={info:`info`,success:`success`,warn:`warn`,error:`danger`},r={info:ch,success:eh,warn:fh,error:ih},i=new Map;function a(e,t){zv(e);let n=t.relatedTarget;n instanceof HTMLElement&&!t.currentTarget.contains(n)&&i.set(e,n)}function o(e,t){let n=t.currentTarget;queueMicrotask(()=>{!n.matches(`:hover`)&&!n.contains(document.activeElement)&&Bv(e)})}function s(e,t){try{t()}finally{Rv(e)}}return Gn(()=>Hv.list.map(e=>e.id),e=>{let t=document.activeElement instanceof HTMLElement?document.activeElement.closest(`[data-toast-id]`):null,n=Number(t?.dataset.toastId);if(t&&!e.includes(n)){let e=i.get(n),t=e?.isConnected?e:document.querySelector(`main, #app`);t&&!t.hasAttribute(`tabindex`)&&!t.matches(`a[href], button, input, select, textarea`)&&t.setAttribute(`tabindex`,`-1`),t?.focus({preventScroll:!0})}for(let t of i.keys())e.includes(t)||i.delete(t)},{flush:`sync`}),(e,i)=>(L(),R(`section`,{class:`oui-toasts`,\"aria-label\":t.label??`Notifications`},[V(fs,{name:`oui-toast`},{default:N(()=>[(L(!0),R(I,null,Wr(M(Hv).list,e=>(L(),R(`div`,{key:e.id,class:`oui-toast`,\"data-toast-id\":e.id,\"data-tone\":n[e.tone],\"data-toast-tone\":e.tone,role:e.tone===`error`?`alert`:`status`,\"aria-live\":e.tone===`error`?`assertive`:`polite`,\"aria-atomic\":`true`,onMouseenter:t=>M(zv)(e.id),onMouseleave:t=>o(e.id,t),onFocusin:t=>a(e.id,t),onFocusout:t=>o(e.id,t)},[V(Rm,{icon:r[e.tone],size:14,tile:``,tone:n[e.tone]},null,8,[`icon`,`tone`]),B(`p`,Gv,[B(`strong`,null,k(e.title),1),e.body?(L(),R(`span`,Kv,k(` `+e.body),1)):H(``,!0)]),e.action?(L(),R(`button`,{key:0,type:`button`,class:`oui-toast-action`,onClick:t=>s(e.id,e.action.run)},k(e.action.label),9,qv)):H(``,!0),B(`button`,{type:`button`,class:`oui-toast-close`,\"aria-label\":t.closeLabel??`Close`,title:t.closeLabel??`Close`,onClick:t=>M(Rv)(e.id)},[V(Rm,{icon:M(ph),size:14},null,8,[`icon`])],8,Jv)],40,Wv))),128))]),_:1})],8,Uv))}}),Xv=Symbol(`organizer`);function Zv(){let e=Vn(Xv);if(!e)throw Error(`Organizer store is not provided`);return e}var Qv=Y.extend({name:`message`,style:`\n"
    "    .p-message {\n"
    "        display: grid;\n"
    "        grid-template-rows: 1fr;\n"
    "        border-radius: dt('message.border.radius');\n"
    "        outline-width: dt('message.border.width');\n"
    "        outline-style: solid;\n"
    "    }\n"
    "\n"
    "    .p-message-content-wrapper {\n"
    "        min-height: 0;\n"
    "    }\n"
    "\n"
    "    .p-message-content {\n"
    "        display: flex;\n"
    "        align-items: center;\n"
    "        padding: dt('message.content.padding');\n"
    "        gap: dt('message.content.gap');\n"
    "    }\n"
    "\n"
    "    .p-message-close-button {\n"
    "        display: flex;\n"
    "        align-items: center;\n"
    "        justify-content: center;\n"
    "        flex-shrink: 0;\n"
    "        margin-inline-start: auto;\n"
    "        overflow: hidden;\n"
    "        position: relative;\n"
    "        width: dt('message.close.button.width');\n"
    "        height: dt('message.close.button.height');\n"
    "        border-radius: dt('message.close.button.border.radius');\n"
    "        background: transparent;\n"
    "        transition:\n"
    "            background dt('message.transition.duration'),\n"
    "            color dt('message.transition.duration'),\n"
    "            outline-color dt('message.transition.duration'),\n"
    "            box-shadow dt('message.transition.duration'),\n"
    "            opacity 0.3s;\n"
    "        outline-color: transparent;\n"
    "        color: inherit;\n"
    "        padding: 0;\n"
    "        border: none;\n"
    "        cursor: pointer;\n"
    "        user-select: none;\n"
    "    }\n"
    "\n"
    "    .p-message-close-button svg,\n"
    "    .p-message-close-button i {\n"
    "        font-size: dt('message.close.icon.size');\n"
    "        width: dt('message.close.icon.size');\n"
    "        height: dt('message.close.icon.size');\n"
    "    }\n"
    "\n"
    "    .p-message-close-button:focus-visible {\n"
    "        outline-width: dt('message.close.button.focus.ring.width');\n"
    "        outline-style: dt('message.close.button.focus.ring.style');\n"
    "        outline-offset: dt('message.close.button.focus.ring.offset');\n"
    "    }\n"
    "\n"
    "    .p-message-info {\n"
    "        background: dt('message.info.background');\n"
    "        outline-color: dt('message.info.border.color');\n"
    "        color: dt('message.info.color');\n"
    "        box-shadow: dt('message.info.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-info .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.info.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.info.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-info .p-message-close-button:hover {\n"
    "        background: dt('message.info.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-info.p-message-outlined {\n"
    "        color: dt('message.info.outlined.color');\n"
    "        outline-color: dt('message.info.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-info.p-message-simple {\n"
    "        color: dt('message.info.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-success {\n"
    "        background: dt('message.success.background');\n"
    "        outline-color: dt('message.success.border.color');\n"
    "        color: dt('message.success.color');\n"
    "        box-shadow: dt('message.success.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-success .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.success.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.success.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-success .p-message-close-button:hover {\n"
    "        background: dt('message.success.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-success.p-message-outlined {\n"
    "        color: dt('message.success.outlined.color');\n"
    "        outline-color: dt('message.success.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-success.p-message-simple {\n"
    "        color: dt('message.success.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-warn {\n"
    "        background: dt('message.warn.background');\n"
    "        outline-color: dt('message.warn.border.color');\n"
    "        color: dt('message.warn.color');\n"
    "        box-shadow: dt('message.warn.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-warn .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.warn.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.warn.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-warn .p-message-close-button:hover {\n"
    "        background: dt('message.warn.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-warn.p-message-outlined {\n"
    "        color: dt('message.warn.outlined.color');\n"
    "        outline-color: dt('message.warn.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-warn.p-message-simple {\n"
    "        color: dt('message.warn.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-error {\n"
    "        background: dt('message.error.background');\n"
    "        outline-color: dt('message.error.border.color');\n"
    "        color: dt('message.error.color');\n"
    "        box-shadow: dt('message.error.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-error .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.error.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.error.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-error .p-message-close-button:hover {\n"
    "        background: dt('message.error.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-error.p-message-outlined {\n"
    "        color: dt('message.error.outlined.color');\n"
    "        outline-color: dt('message.error.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-error.p-message-simple {\n"
    "        color: dt('message.error.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-secondary {\n"
    "        background: dt('message.secondary.background');\n"
    "        outline-color: dt('message.secondary.border.color');\n"
    "        color: dt('message.secondary.color');\n"
    "        box-shadow: dt('message.secondary.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-secondary .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.secondary.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.secondary.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-secondary .p-message-close-button:hover {\n"
    "        background: dt('message.secondary.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-secondary.p-message-outlined {\n"
    "        color: dt('message.secondary.outlined.color');\n"
    "        outline-color: dt('message.secondary.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-secondary.p-message-simple {\n"
    "        color: dt('message.secondary.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-contrast {\n"
    "        background: dt('message.contrast.background');\n"
    "        outline-color: dt('message.contrast.border.color');\n"
    "        color: dt('message.contrast.color');\n"
    "        box-shadow: dt('message.contrast.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-contrast .p-message-close-button:focus-visible {\n"
    "        outline-color: dt('message.contrast.close.button.focus.ring.color');\n"
    "        box-shadow: dt('message.contrast.close.button.focus.ring.shadow');\n"
    "    }\n"
    "\n"
    "    .p-message-contrast .p-message-close-button:hover {\n"
    "        background: dt('message.contrast.close.button.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-message-contrast.p-message-outlined {\n"
    "        color: dt('message.contrast.outlined.color');\n"
    "        outline-color: dt('message.contrast.outlined.border.color');\n"
    "    }\n"
    "\n"
    "    .p-message-contrast.p-message-simple {\n"
    "        color: dt('message.contrast.simple.color');\n"
    "    }\n"
    "\n"
    "    .p-message-text {\n"
    "        font-size: dt('message.text.font.size');\n"
    "        font-weight: dt('message.text.font.weight');\n"
    "    }\n"
    "\n"
    "    .p-message-icon {\n"
    "        display: inline-flex;\n"
    "    }\n"
    "\n"
    "    .p-message-icon,\n"
    "    .p-message-icon svg,\n"
    "    .p-message-icon i {\n"
    "        flex-shrink: 0;\n"
    "        font-size: dt('message.icon.size');\n"
    "        width: dt('message.icon.size');\n"
    "        height: dt('message.icon.size');\n"
    "    }\n"
    "\n"
    "    .p-message-sm .p-message-content {\n"
    "        padding: dt('message.content.sm.padding');\n"
    "    }\n"
    "\n"
    "    .p-message-sm .p-message-text {\n"
    "        font-size: dt('message.text.sm.font.size');\n"
    "    }\n"
    "\n"
    "    .p-message-sm .p-message-icon,\n"
    "    .p-message-sm .p-message-icon svg,\n"
    "    .p-message-sm .p-message-icon i {\n"
    "        font-size: dt('message.icon.sm.size');\n"
    "        width: dt('message.icon.sm.size');\n"
    "        height: dt('message.icon.sm.size');\n"
    "    }\n"
    "\n"
    "    .p-message-sm .p-message-close-button svg,\n"
    "    .p-message-sm .p-message-close-button i {\n"
    "        width: dt('message.close.icon.sm.size');\n"
    "        height: dt('message.close.icon.sm.size');\n"
    "    }\n"
    "\n"
    "    .p-message-lg .p-message-content {\n"
    "        padding: dt('message.content.lg.padding');\n"
    "    }\n"
    "\n"
    "    .p-message-lg .p-message-text {\n"
    "        font-size: dt('message.text.lg.font.size');\n"
    "    }\n"
    "\n"
    "    .p-message-lg .p-message-icon,\n"
    "    .p-message-lg .p-message-icon svg,\n"
    "    .p-message-lg .p-message-icon i {\n"
    "        font-size: dt('message.icon.lg.size');\n"
    "        width: dt('message.icon.lg.size');\n"
    "        height: dt('message.icon.lg.size');\n"
    "    }\n"
    "\n"
    "    .p-message-lg .p-message-close-button svg,\n"
    "    .p-message-lg .p-message-close-button i {\n"
    "        font-size: dt('message.close.icon.lg.size');\n"
    "        width: dt('message.close.icon.lg.size');\n"
    "        height: dt('message.close.icon.lg.size');\n"
    "    }\n"
    "\n"
    "    .p-message-outlined {\n"
    "        background: transparent;\n"
    "        outline-width: dt('message.outlined.border.width');\n"
    "    }\n"
    "\n"
    "    .p-message-simple {\n"
    "        background: transparent;\n"
    "        outline-color: transparent;\n"
    "        box-shadow: none;\n"
    "    }\n"
    "\n"
    "    .p-message-simple .p-message-content {\n"
    "        padding: dt('message.simple.content.padding');\n"
    "    }\n"
    "\n"
    "    .p-message-outlined .p-message-close-button:hover,\n"
    "    .p-message-simple .p-message-close-button:hover {\n"
    "        background: transparent;\n"
    "    }\n"
    "\n"
    "    .p-message-enter-active {\n"
    "        animation: p-animate-message-enter 0.3s ease-out forwards;\n"
    "        overflow: hidden;\n"
    "    }\n"
    "\n"
    "    .p-message-leave-active {\n"
    "        animation: p-animate-message-leave 0.15s ease-in forwards;\n"
    "        overflow: hidden;\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-message-enter {\n"
    "        from {\n"
    "            opacity: 0;\n"
    "            grid-template-rows: 0fr;\n"
    "        }\n"
    "        to {\n"
    "            opacity: 1;\n"
    "            grid-template-rows: 1fr;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    @keyframes p-animate-message-leave {\n"
    "        from {\n"
    "            opacity: 1;\n"
    "            grid-template-rows: 1fr;\n"
    "        }\n"
    "        to {\n"
    "            opacity: 0;\n"
    "            margin: 0;\n"
    "            grid-template-rows: 0fr;\n"
    "        }\n"
    "    }\n"
    "`,classes:{root:function(e){var t=e.props;return[`p-message p-component p-message-`+t.severity,{\"p-message-outlined\":t.variant===`outlined`,\"p-message-simple\":t.variant===`simple`,\"p-message-sm\":t.size===`small`,\"p-message-lg\":t.size===`large`}]},contentWrapper:`p-message-content-wrapper`,content:`p-message-content`,icon:`p-message-icon`,text:`p-message-text`,closeButton:`p-message-close-button`,closeIcon:`p-message-close-icon`}}),$v={name:`BaseMessage`,extends:rg,props:{severity:{type:String,default:`info`},closable:{type:Boolean,default:!1},life:{type:Number,default:null},icon:{type:String,default:void 0},closeIcon:{type:String,default:void 0},closeButtonProps:{type:null,default:null},size:{type:String,default:null},variant:{type:String,default:null}},style:Qv,provide:function(){return{$pcMessage:this,$parentInstance:this}}};function ey(e){\"@babel/helpers - typeof\";return ey=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},ey(e)}function ty(e,t,n){return(t=ny(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function ny(e){var t=ry(e,`string`);return ey(t)==`symbol`?t:t+``}function ry(e,t){if(ey(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(ey(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var iy={name:`Message`,extends:$v,inheritAttrs:!1,emits:[`close`,`life-end`],timeout:null,data:function(){return{visible:!0}},mounted:function(){var e=this;this.life&&setTimeout(function(){e.visible=!1,e.$emit(`life-end`)},this.life)},methods:{close:function(e){this.visible=!1,this.$emit(`close`,e)}},computed:{closeAriaLabel:function(){return this.$primevue.config.locale.aria?this.$primevue.config.locale.aria.close:void 0},dataP:function(){return Tl(ty(ty({outlined:this.variant===`outlined`,simple:this.variant===`simple`},this.severity,this.severity),this.size,this.size))}},directives:{ripple:Lg},components:{Times:e_}};function ay(e){\"@babel/helpers - typeof\";return ay=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},ay(e)}function oy(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),n.push.apply(n,r)}return n}function sy(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]==null?{}:arguments[t];t%2?oy(Object(n),!0).forEach(function(t){cy(e,t,n[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):oy(Object(n)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))})}return e}function cy(e,t,n){return(t=ly(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function ly(e){var t=uy(e,`string`);return ay(t)==`symbol`?t:t+``}function uy(e,t){if(ay(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(ay(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var dy=[`data-p`],fy=[`data-p`],py=[`data-p`],my=[`aria-label`,`data-p`],hy=[`data-p`];function gy(e,t,n,r,i,a){var o=Rr(`Times`),s=Vr(`ripple`);return L(),z(mo,U({name:e.pcn(`p-message`),appear:``},e.ptm(`transition`)),{default:N(function(){return[i.visible?(L(),R(`div`,U({key:0,class:e.cx(`root`),role:`alert`,\"aria-live\":`assertive`,\"aria-atomic\":`true`,\"data-p\":a.dataP},e.ptmi(`root`)),[B(`div`,U({class:e.cx(`contentWrapper`)},e.ptm(`contentWrapper`)),[e.$slots.container?F(e.$slots,`container`,{closeCallback:a.close},void 0,void 0,0):(L(),R(`div`,U({key:1,class:e.cx(`content`),\"data-p\":a.dataP},e.ptm(`content`)),[F(e.$slots,`icon`,{class:ge(e.cx(`icon`))},function(){return[(L(),z(Br(e.icon?`span`:null),U({class:[e.cx(`icon`),e.icon],\"data-p\":a.dataP},e.ptm(`icon`)),null,16,[`class`,`data-p`]))]}),e.$slots.default?(L(),R(`div`,U({key:0,class:e.cx(`text`),\"data-p\":a.dataP},e.ptm(`text`)),[F(e.$slots,`default`)],16,py)):H(``,!0),e.closable?Rn((L(),R(`button`,U({key:1,class:e.cx(`closeButton`),\"aria-label\":a.closeAriaLabel,type:`button`,onClick:t[0]||=function(e){return a.close(e)},\"data-p\":a.dataP},sy(sy({},e.closeButtonProps),e.ptm(`closeButton`))),[F(e.$slots,`closeicon`,{},function(){return[e.closeIcon?(L(),R(`i`,U({key:0,class:[e.cx(`closeIcon`),e.closeIcon],\"data-p\":a.dataP},e.ptm(`closeIcon`)),null,16,hy)):(L(),z(o,U({key:1,class:[e.cx(`closeIcon`),e.closeIcon],\"data-p\":a.dataP},e.ptm(`closeIcon`)),null,16,[`class`,`data-p`]))]})],16,my)),[[s]]):H(``,!0)],16,fy))],16)],16,dy)):H(``,!0)]}),_:3},16,[`name`])}iy.render=gy;var _y=Y.extend({name:`radiobutton`,style:`\n"
    "    .p-radiobutton {\n"
    "        position: relative;\n"
    "        display: inline-flex;\n"
    "        user-select: none;\n"
    "        vertical-align: bottom;\n"
    "        width: dt('radiobutton.width');\n"
    "        height: dt('radiobutton.height');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-input {\n"
    "        cursor: pointer;\n"
    "        appearance: none;\n"
    "        position: absolute;\n"
    "        top: 0;\n"
    "        inset-inline-start: 0;\n"
    "        width: 100%;\n"
    "        height: 100%;\n"
    "        padding: 0;\n"
    "        margin: 0;\n"
    "        opacity: 0;\n"
    "        z-index: 1;\n"
    "        outline: 0 none;\n"
    "        border: 1px solid transparent;\n"
    "        border-radius: 50%;\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-box {\n"
    "        display: flex;\n"
    "        justify-content: center;\n"
    "        align-items: center;\n"
    "        border-radius: 50%;\n"
    "        border: 1px solid dt('radiobutton.border.color');\n"
    "        background: dt('radiobutton.background');\n"
    "        width: dt('radiobutton.width');\n"
    "        height: dt('radiobutton.height');\n"
    "        transition:\n"
    "            background dt('radiobutton.transition.duration'),\n"
    "            color dt('radiobutton.transition.duration'),\n"
    "            border-color dt('radiobutton.transition.duration'),\n"
    "            box-shadow dt('radiobutton.transition.duration'),\n"
    "            outline-color dt('radiobutton.transition.duration');\n"
    "        outline-color: transparent;\n"
    "        box-shadow: dt('radiobutton.shadow');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-icon {\n"
    "        transition-duration: dt('radiobutton.transition.duration');\n"
    "        background: transparent;\n"
    "        font-size: dt('radiobutton.icon.size');\n"
    "        width: dt('radiobutton.icon.size');\n"
    "        height: dt('radiobutton.icon.size');\n"
    "        border-radius: 50%;\n"
    "        backface-visibility: hidden;\n"
    "        transform: translateZ(0) scale(0.1);\n"
    "    }\n"
    "\n"
    "    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:hover) .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.hover.border.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-checked .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.checked.border.color');\n"
    "        background: dt('radiobutton.checked.background');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-checked .p-radiobutton-box .p-radiobutton-icon {\n"
    "        background: dt('radiobutton.icon.checked.color');\n"
    "        transform: translateZ(0) scale(1, 1);\n"
    "        visibility: visible;\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-checked:not(.p-disabled):has(.p-radiobutton-input:hover) .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.checked.hover.border.color');\n"
    "        background: dt('radiobutton.checked.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:hover).p-radiobutton-checked .p-radiobutton-box .p-radiobutton-icon {\n"
    "        background: dt('radiobutton.icon.checked.hover.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton:not(.p-disabled):has(.p-radiobutton-input:focus-visible) .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.focus.border.color');\n"
    "        box-shadow: dt('radiobutton.focus.ring.shadow');\n"
    "        outline: dt('radiobutton.focus.ring.width') dt('radiobutton.focus.ring.style') dt('radiobutton.focus.ring.color');\n"
    "        outline-offset: dt('radiobutton.focus.ring.offset');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-checked:not(.p-disabled):has(.p-radiobutton-input:focus-visible) .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.checked.focus.border.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-invalid > .p-radiobutton-box {\n"
    "        border-color: dt('radiobutton.invalid.border.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-variant-filled .p-radiobutton-box {\n"
    "        background: dt('radiobutton.filled.background');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-variant-filled.p-radiobutton-checked .p-radiobutton-box {\n"
    "        background: dt('radiobutton.checked.background');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-variant-filled:not(.p-disabled):has(.p-radiobutton-input:hover).p-radiobutton-checked .p-radiobutton-box {\n"
    "        background: dt('radiobutton.checked.hover.background');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-disabled {\n"
    "        opacity: 1;\n"
    "    }\n"
    "\n"
    "    .p-radiobutton.p-disabled .p-radiobutton-box {\n"
    "        background: dt('radiobutton.disabled.background');\n"
    "        border-color: dt('radiobutton.checked.disabled.border.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-checked.p-disabled .p-radiobutton-box .p-radiobutton-icon {\n"
    "        background: dt('radiobutton.icon.disabled.color');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-sm,\n"
    "    .p-radiobutton-sm .p-radiobutton-box {\n"
    "        width: dt('radiobutton.sm.width');\n"
    "        height: dt('radiobutton.sm.height');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-sm .p-radiobutton-icon {\n"
    "        font-size: dt('radiobutton.icon.sm.size');\n"
    "        width: dt('radiobutton.icon.sm.size');\n"
    "        height: dt('radiobutton.icon.sm.size');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-lg,\n"
    "    .p-radiobutton-lg .p-radiobutton-box {\n"
    "        width: dt('radiobutton.lg.width');\n"
    "        height: dt('radiobutton.lg.height');\n"
    "    }\n"
    "\n"
    "    .p-radiobutton-lg .p-radiobutton-icon {\n"
    "        font-size: dt('radiobutton.icon.lg.size');\n"
    "        width: dt('radiobutton.icon.lg.size');\n"
    "        height: dt('radiobutton.icon.lg.size');\n"
    "    }\n"
    "`,classes:{root:function(e){var t=e.instance,n=e.props;return[`p-radiobutton p-component`,{\"p-radiobutton-checked\":t.checked,\"p-disabled\":n.disabled,\"p-invalid\":t.$pcRadioButtonGroup?t.$pcRadioButtonGroup.$invalid:t.$invalid,\"p-variant-filled\":t.$variant===`filled`,\"p-radiobutton-sm p-inputfield-sm\":n.size===`small`,\"p-radiobutton-lg p-inputfield-lg\":n.size===`large`}]},box:`p-radiobutton-box`,input:`p-radiobutton-input`,icon:`p-radiobutton-icon`}}),vy={name:`BaseRadioButton`,extends:t_,props:{value:null,binary:Boolean,readonly:{type:Boolean,default:!1},tabindex:{type:Number,default:null},inputId:{type:String,default:null},inputClass:{type:[String,Object],default:null},inputStyle:{type:Object,default:null},ariaLabelledby:{type:String,default:null},ariaLabel:{type:String,default:null}},style:_y,provide:function(){return{$pcRadioButton:this,$parentInstance:this}}};function yy(e){\"@babel/helpers - typeof\";return yy=typeof Symbol==`function`&&typeof Symbol.iterator==`symbol`?function(e){return typeof e}:function(e){return e&&typeof Symbol==`function`&&e.constructor===Symbol&&e!==Symbol.prototype?`symbol`:typeof e},yy(e)}function by(e,t,n){return(t=xy(t))in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function xy(e){var t=Sy(e,`string`);return yy(t)==`symbol`?t:t+``}function Sy(e,t){if(yy(e)!=`object`||!e)return e;var n;if(typeof Symbol<`u`&&(n=e[Symbol.toPrimitive])!==void 0){var r=n.call(e,t);if(yy(r)!=`object`)return r;throw TypeError(`@@toPrimitive must return a primitive value.`)}return(t===`string`?String:Number)(e)}var Cy={name:`RadioButton`,extends:vy,inheritAttrs:!1,emits:[`change`,`focus`,`blur`],inject:{$pcRadioButtonGroup:{default:void 0}},methods:{getPTOptions:function(e){return(e===`root`?this.ptmi:this.ptm)(e,{context:{checked:this.checked,disabled:this.disabled}})},onChange:function(e){if(!this.disabled&&!this.readonly){var t=this.binary?!this.checked:this.value;this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.writeValue(t,e):this.writeValue(t,e),this.$emit(`change`,e)}},onFocus:function(e){this.$emit(`focus`,e)},onBlur:function(e){var t,n;this.$emit(`blur`,e),(t=(n=this.formField).onBlur)==null||t.call(n,e)}},computed:{groupName:function(){return this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.groupName:this.$formName},checked:function(){var e=this.$pcRadioButtonGroup?this.$pcRadioButtonGroup.d_value:this.d_value;return e!=null&&(this.binary?!!e:al(e,this.value))},dataP:function(){return Tl(by({invalid:this.$invalid,checked:this.checked,disabled:this.disabled,filled:this.$variant===`filled`},this.size,this.size))}}},wy=[`data-p-checked`,`data-p-disabled`,`data-p`],Ty=[`id`,`value`,`name`,`checked`,`tabindex`,`disabled`,`readonly`,`aria-labelledby`,`aria-label`,`aria-invalid`],Ey=[`data-p`],Dy=[`data-p`];function Oy(e,t,n,r,i,a){return L(),R(`div`,U({class:e.cx(`root`)},a.getPTOptions(`root`),{\"data-p-checked\":a.checked,\"data-p-disabled\":e.disabled,\"data-p\":a.dataP}),[B(`input`,U({id:e.inputId,type:`radio`,class:[e.cx(`input`),e.inputClass],style:e.inputStyle,value:e.value,name:a.groupName,checked:a.checked,tabindex:e.tabindex,disabled:e.disabled,readonly:e.readonly,\"aria-labelledby\":e.ariaLabelledby,\"aria-label\":e.ariaLabel,\"aria-invalid\":e.invalid||void 0,onFocus:t[0]||=function(){return a.onFocus&&a.onFocus.apply(a,arguments)},onBlur:t[1]||=function(){return a.onBlur&&a.onBlur.apply(a,arguments)},onChange:t[2]||=function(){return a.onChange&&a.onChange.apply(a,arguments)}},a.getPTOptions(`input`)),null,16,Ty),B(`div`,U({class:e.cx(`box`)},a.getPTOptions(`box`),{\"data-p\":a.dataP}),[B(`div`,U({class:e.cx(`icon`)},a.getPTOptions(`icon`),{\"data-p\":a.dataP}),null,16,Dy)],16,Ey)],16,wy)}Cy.render=Oy;function ky(e,t){let n=n=>e(n)+` `+t(n,`Objekt`,`Objekte`);return{title:`Organizer`,sub:`Shape-Aware Arrange: ordnet und füllt die Platte nach dem Umriss jedes Teils.`,spacingLabel:`Abstand zwischen Teilen`,spacingHelp:`Abstand zwischen dem Umriss jedes Teils (von oben gesehen), dem Rand des Druckbetts und ausgenommenen Bereichen. Wird die Platte nach Objekt gedruckt, kann Orcas Extruder-Freiraum ihn vergrößern; das Ergebnis weist darauf hin.`,goHelp:`Ordnet die Objekte der Platte an. Fügt nichts hinzu.`,fillHelp:`Ordnet an und fügt dann reihum Kopien der Objekte der Platte hinzu, bis nichts mehr passt (höchstens 300).`,busyHint:`Ein Durchlauf läuft gerade. Warten Sie, bis er fertig ist, oder brechen Sie ihn ab, um diese Einstellungen zu ändern.`,adv:`Erweitert`,advMeta:`Drehung, Kopien`,help:`So funktioniert es`,result:`Ergebnis`,nothingYet:`Noch nichts angeordnet`,nothingYetText:`Anordnen platziert die Objekte der Platte nach ihrem tatsächlichen Umriss. Füllen fügt danach Kopien hinzu, bis nichts mehr passt.`,again:`Neue Anordnung`,statusOffline:`Keine Antwort`,statusProblem:`Kein Projekt`,statusReading:`Wird gelesen…`,rotationLabel:`Drehung`,optAuto:`Automatisch (1°-Schritte)`,optRight:`Nur 90°-Schritte`,copiesLabel:`Maximale Anzahl Kopien beim Füllen`,copiesHelp:`Füllen endet nach dieser Anzahl von Kopien oder früher, wenn die Platte voll ist. Leer lassen, um die Platte zu füllen (höchstens 300).`,optKeep:`Ausrichtung jedes Teils beibehalten`,rotationHelp:`Automatisch probiert jeden Winkel. Nur 90°-Schritte hält die Kanten von Teilen, die jetzt rechtwinklig zum Druckbett liegen, rechtwinklig. Ausrichtung beibehalten verschiebt die Teile nur. Bei den beiden letzten passen weniger Teile.`,cancel:`Abbrechen`,cancelPending:`Wird abgebrochen…`,saved:`Gespeicherte Kopie`,open:`Kopie in OrcaSlicer öffnen`,copy:`Pfad kopieren`,retry:`Erneut versuchen`,retrying:`Neuer Versuch…`,foot:`Der Abstand gilt für den Umriss der Teile von oben. Rand (Brim), Stützen und Umrandung sind nicht eingerechnet, und der Reinigungsturm ist nur eine Schätzung. Prüfen Sie die Platte vor dem Drucken in Orca.`,barLabel:`Fortschritt`,loadingLabel:`Projekt wird geladen`,go:e=>`Platte `+e.plate+` anordnen`,fill:e=>`Platte `+e.plate+` füllen`,goNone:`Aktuelle Platte anordnen`,fillNone:`Aktuelle Platte füllen`,untitled:`unbenannt`,untitledProject:`Unbenanntes Projekt`,ref:e=>e.name+`, Platte `+e.plate,sep:`: `,list:`, `,ctx:e=>e.name+` · Platte `+e.plate+` ausgewählt`+(e.objects==null?``:` · `+n(e.objects)+` im Projekt`),ctxProblem:e=>`Das Projekt konnte nicht gelesen werden: `+e.problem+`. Öffnen Sie ein Projekt in OrcaSlicer und versuchen Sie es erneut.`,reading:`Projekt wird gelesen…`,ready:`Bereit. Das Ergebnis wird als neue Datei gespeichert; Ihr Projekt bleibt unverändert.`,emptyProject:`Im Projekt wurden keine Objekte gefunden. Legen Sie zuerst Teile auf eine Platte.`,starting:`Wird gestartet…`,spacingErr:`Geben Sie einen Abstand von 0 mm oder mehr ein.`,stale:e=>`Der Durchlauf unten gehört zu `+e.ref+`, nicht zur aktuellen Auswahl.`,offline:`Keine Antwort vom Plugin.`,offlineHelp:`Prüfen Sie, ob es im Plugins-Fenster von OrcaSlicer noch aktiviert ist, und versuchen Sie es erneut. Die Seite fragt von selbst weiter nach.`,phase:{read:`Projekt wird gelesen…`,footprint:`Umrisse der Teile werden gemessen`,place:`Teile werden platziert`,patterns:`Wiederholmuster werden ausprobiert`,fill:`Kopien werden hinzugefügt`},plateLabel:e=>`Platte `+e.plate,ctxShort:e=>`Platte `+e.plate+(e.objects==null?``:` · `+n(e.objects)),countShort:t=>e(t.i+1)+` von `+e(t.n),count:n=>n.n==null?e(n.count??0)+` `+t(n.count??0,`Kopie`,`Kopien`)+` hinzugefügt`:e(n.i+1)+` von `+e(n.n)+(n.subject?` · `+n.subject:``),cancelling:e=>e.ref+`: wird abgebrochen…`,cancelled:e=>e.ref+`: abgebrochen. Es wurde nichts gespeichert oder geöffnet.`,saving:e=>e.ref+`: angeordnete Kopie wird gespeichert…`,act:{arrange:`Anordnen`,fill:`Füllen`},failed:e=>e.ref+`: `+e.act+` ist fehlgeschlagen.`,startFailed:e=>`Start nicht möglich: `+e.why,next:`Nächster Schritt: `,details:`Details: `,err:{bad_number:[`Der Abstand muss eine Zahl von 0 mm oder mehr sein.`,`Geben Sie einen Wert wie 2 ein und versuchen Sie es erneut.`],bad_step:[`Der Drehschritt muss eine Zahl von 0° oder mehr sein.`,`Verwenden Sie einen Schritt wie 15, oder 0, um die Ausrichtung jedes Teils beizubehalten, und versuchen Sie es erneut.`],bad_res:[`Die Rastergröße muss eine Zahl über 0 mm sein.`,`Verwenden Sie eine Rastergröße wie `+e(.5)+` und versuchen Sie es erneut.`],bad_copies:t=>[`Die Anzahl der Kopien muss eine ganze Zahl von 1 bis `+e(t.max)+` sein.`,`Geben Sie eine Anzahl wie 10 ein, oder lassen Sie das Feld leer, um die Platte zu füllen, und versuchen Sie es erneut.`],bad_rotation:[`Die Drehung muss „Automatisch“, „Nur 90°-Schritte“ oder „Ausrichtung beibehalten“ sein.`,`Wählen Sie eine der drei Optionen.`],no_model:[`Die Projektdatei enthält kein 3D-Modell.`,`Speichern Sie das Projekt erneut in OrcaSlicer und versuchen Sie es noch einmal.`],bad_unit:e=>[`Das Projekt verwendet `+e.unit+` als Einheit; angeordnet werden kann nur in Millimetern.`,`Wandeln Sie das Modell in OrcaSlicer in Millimeter um, speichern Sie es und versuchen Sie es noch einmal.`],no_build:[`Die Projektdatei enthält keinen Build-Abschnitt.`,`Speichern Sie das Projekt erneut in OrcaSlicer und versuchen Sie es noch einmal.`],build_layout:[`Der Aufbau der Projektdatei lässt sich nicht sicher bearbeiten.`,`Speichern Sie das Projekt erneut in OrcaSlicer (dabei wird die Datei neu geschrieben) und versuchen Sie es noch einmal.`],plate_missing:e=>[`Platte `+e.plate+` steht nicht in der Plattenliste des Projekts.`,`Wählen Sie eine vorhandene Platte und versuchen Sie es noch einmal.`],plate_gone:e=>[`Platte `+e.plate+` gibt es nicht: Das Projekt hat `+e.plates+` `+t(e.plates,`Platte`,`Platten`)+`.`,`Wählen Sie eine der vorhandenen Platten und versuchen Sie es noch einmal.`],plate_empty:e=>[`Platte `+e.plate+` enthält keine druckbaren Objekte.`,`Wählen Sie eine andere Platte oder legen Sie Teile auf diese.`],bed_small:t=>[`Das Druckbett ist kleiner als das Raster von `+e(t.res)+` mm, mit dem Teile platziert werden.`,`Prüfen Sie die Druckbettform des Druckers in OrcaSlicer.`],spacing_large:t=>[`Ein Abstand von `+e(t.spacing)+` mm lässt auf diesem Druckbett keinen Platz.`,`Geben Sie einen Abstand unter `+e(t.limit)+` mm ein und versuchen Sie es erneut.`],clearance_large:t=>[`Die Platte wird nach Objekt gedruckt, und der Radius des Extruder-Freiraums des Druckers von `+e(t.clearance)+` mm lässt auf diesem Druckbett keinen Platz.`,`Stellen Sie den Radius des Extruder-Freiraums in den Drucker-Einstellungen unter `+e(t.limit)+` mm ein oder drucken Sie „Nach Ebene“, und versuchen Sie es erneut.`],missing_requirements:e=>[`Anordnen und Füllen benötigen `+e.names+`, konnten aber nicht geladen werden.`,`Installieren Sie das Plugin aus Orca Cloud neu, starten Sie OrcaSlicer neu und versuchen Sie es erneut.`],unexpected:[`Etwas Unerwartetes ist schiefgegangen.`,`Versuchen Sie es erneut. Tritt der Fehler weiter auf, helfen die Details unten bei der Diagnose.`]},saveFailed:e=>e.ref+`: Die angeordnete Kopie konnte nicht gespeichert werden: `+e.detail+`\n"
    "Es wurde nichts geöffnet; frühere Kopien sind unverändert.`,res:{placed:n=>n.ref+`: `+e(n.placed)+` `+t(n.placed,`Objekt`,`Objekte`)+` platziert.`,added:n=>`Platte mit `+e(n.added)+` zusätzlichen `+t(n.added,`Kopie`,`Kopien`)+` gefüllt`+(n.per?` (`+n.per+`).`:`.`),limit:t=>`Bei der Grenze von `+e(t.max)+` Kopien angehalten.`,tower:`Eine geschätzte Fläche für den Reinigungsturm wurde freigehalten, Rand (Brim) eingeschlossen. Seine tatsächliche Größe steht erst nach dem Slicen fest: Prüfen Sie sie in Orca.`,clearance:t=>`Die Druckreihenfolge ist „Nach Objekt“: Zwischen den Teilen wurden `+e(t.clearance)+` mm XY-Freiraum eingehalten. Höhe und Reihenfolge der Objekte prüft Orca weiterhin.`,keep:`Die Drehung war deaktiviert: Jedes Teil hat seine Ausrichtung behalten.`,right:`Die Drehung war auf 90°-Schritte begrenzt.`,unplaced:e=>`Kein Platz (neben die Platte verschoben): `+e.names,foot:`Angeordnet wurde der Projektstand zum Zeitpunkt des Klicks.`},open_:{asked:`OrcaSlicer wurde gebeten, die Kopie zu öffnen. Falls nichts erscheint, verwenden Sie „Kopie in OrcaSlicer öffnen“.`,later:`Fertig geworden, während das Panel geschlossen war. Laden Sie die Kopie mit „Kopie in OrcaSlicer öffnen“.`,failed:e=>`OrcaSlicer konnte nicht gebeten werden, die Kopie zu öffnen (`+e.detail+`). Verwenden Sie „Kopie in OrcaSlicer öffnen“ oder öffnen Sie die Datei von Hand.`,none:`Kein Startprogramm für OrcaSlicer gefunden: Öffnen Sie die gespeicherte Kopie von Hand.`},copied:`Pfad kopiert.`,close:`Schließen`,notifications:`Benachrichtigungen`,copyFail:`Automatisches Kopieren nicht möglich. Der Pfad ist markiert: Drücken Sie Strg+C (⌘C auf dem Mac).`,req:`Voraussetzungen`,reqAllFound:`alle gefunden`,reqMissingMeta:e=>`fehlt: `+e.names,reqBlocked:e=>`Anordnen und Füllen sind aus: `+e.names+` konnte nicht geladen werden. Siehe Voraussetzungen unten.`,reqFound:e=>`Gefunden, Version `+e.version,reqMissing:`Nicht gefunden`,reqWhy:{numpy:`Misst den Umriss jedes Teils auf einem Raster (Rasterung der Grundfläche) für Anordnen und Füllen.`,pillow:`Zeichnet den Umriss jedes Teils in dieses Raster (Rasterung der Grundfläche) für Anordnen und Füllen.`},reqFix:`OrcaSlicer installiert es mit dem Plugin. Installieren Sie das Plugin aus Orca Cloud neu, damit OrcaSlicer seine Pakete erneut installiert, und starten Sie OrcaSlicer neu.`,reqGuide:`Installationsanleitung`,reqCheck:`Erneut prüfen`,reqChecking:`Wird geprüft…`}}function Ay(e,t){let n=n=>e(n)+` `+t(n,`objeto`,`objetos`);return{title:`Organizer`,sub:`Shape-Aware Arrange: organiza y llena la cama según el contorno de cada pieza.`,spacingLabel:`Separación entre piezas`,spacingHelp:`Distancia que se mantiene entre el contorno de cada pieza visto desde arriba, el borde de la cama y las áreas excluidas. Si la cama se imprime objeto por objeto, el margen del extrusor de Orca puede aumentarla; el resultado lo indica.`,goHelp:`Organiza los objetos de la cama. No añade nada.`,fillHelp:`Organiza y luego añade copias de los objetos de la cama, por turnos, hasta que no quepa nada más (hasta 300).`,busyHint:`Hay una ejecución en curso. Espera a que termine, o cancélala, para cambiar estos ajustes.`,adv:`Avanzado`,advMeta:`rotación, copias`,help:`Cómo funciona`,result:`Resultado`,nothingYet:`Todavía no hay nada organizado`,nothingYetText:`Organizar acomoda los objetos de la cama según su contorno real. Llenar añade después copias hasta que no quepa nada más.`,again:`Nueva organización`,statusOffline:`Sin respuesta`,statusProblem:`Sin proyecto`,statusReading:`Leyendo…`,rotationLabel:`Rotación`,optAuto:`Automática (pasos de 1°)`,optRight:`Solo pasos de 90°`,copiesLabel:`Máximo de copias al llenar`,copiesHelp:`Llenar se detiene al llegar a este número de copias, o antes si la cama se llena. Déjalo vacío para llenar la cama (hasta 300).`,optKeep:`Mantener la orientación de cada pieza`,rotationHelp:`Automática prueba todos los ángulos. Solo pasos de 90° mantiene a escuadra con la cama los bordes de las piezas que ya lo están. Mantener la orientación solo mueve las piezas. En los dos últimos casos caben menos piezas.`,cancel:`Cancelar`,cancelPending:`Cancelando…`,saved:`Copia guardada`,open:`Abrir esta copia en OrcaSlicer`,copy:`Copiar ruta`,retry:`Reintentar`,retrying:`Reintentando…`,foot:`Las piezas se separan según su contorno visto desde arriba. El borde de adherencia, los soportes y la falda no se incluyen, y la torre de purga es solo una estimación. Revisa la cama en Orca antes de imprimir.`,barLabel:`Progreso`,loadingLabel:`Cargando el proyecto`,go:e=>`Organizar cama `+e.plate,fill:e=>`Llenar cama `+e.plate,goNone:`Organizar la cama actual`,fillNone:`Llenar la cama actual`,untitled:`sin título`,untitledProject:`Proyecto sin título`,ref:e=>e.name+`, cama `+e.plate,sep:`: `,list:`, `,ctx:e=>e.name+` · cama `+e.plate+` seleccionada`+(e.objects==null?``:` · `+n(e.objects)+` en el proyecto`),ctxProblem:e=>`No se pudo leer el proyecto: `+e.problem+`. Abre un proyecto en OrcaSlicer e inténtalo de nuevo.`,reading:`Leyendo el proyecto…`,ready:`Listo. El resultado se guarda en un archivo nuevo; tu proyecto no se modifica.`,emptyProject:`No se encontraron objetos en el proyecto. Primero añade piezas a una cama.`,starting:`Iniciando…`,spacingErr:`Escribe una separación de 0 mm o más.`,stale:e=>`La ejecución de abajo es de `+e.ref+`, no de la selección actual.`,offline:`El plugin no responde.`,offlineHelp:`Comprueba que siga activado en la ventana Plugins de OrcaSlicer e inténtalo de nuevo. La página sigue preguntando por sí sola.`,phase:{read:`Leyendo el proyecto…`,footprint:`Midiendo el contorno de las piezas`,place:`Colocando las piezas`,patterns:`Probando patrones repetidos`,fill:`Añadiendo copias`},plateLabel:e=>`Cama `+e.plate,ctxShort:e=>`Cama `+e.plate+(e.objects==null?``:` · `+n(e.objects)),countShort:t=>e(t.i+1)+` de `+e(t.n),count:n=>n.n==null?e(n.count??0)+` `+t(n.count??0,`copia añadida`,`copias añadidas`):e(n.i+1)+` de `+e(n.n)+(n.subject?` · `+n.subject:``),cancelling:e=>e.ref+`: cancelando…`,cancelled:e=>e.ref+`: cancelado. No se guardó ni se abrió nada.`,saving:e=>e.ref+`: guardando la copia organizada…`,act:{arrange:`organizar`,fill:`llenar`},failed:e=>e.ref+`: no se pudo `+e.act+`.`,startFailed:e=>`No se pudo iniciar: `+e.why,next:`Siguiente paso: `,details:`Detalles: `,err:{bad_number:[`La separación debe ser un número de 0 mm o más.`,`Escribe un valor como 2 e inténtalo de nuevo.`],bad_step:[`El paso de rotación debe ser un número de 0° o más.`,`Usa un paso como 15, o 0 para mantener la orientación de cada pieza, e inténtalo de nuevo.`],bad_res:[`El tamaño de la cuadrícula debe ser un número mayor que 0 mm.`,`Usa un tamaño de cuadrícula como `+e(.5)+` e inténtalo de nuevo.`],bad_copies:t=>[`El número de copias debe ser un número entero de 1 a `+e(t.max)+`.`,`Escribe un número de copias como 10, o déjalo vacío para llenar la cama, e inténtalo de nuevo.`],bad_rotation:[`La rotación debe ser automática, solo pasos de 90° o mantener la orientación.`,`Elige una de las tres opciones.`],no_model:[`El archivo del proyecto no tiene ningún modelo 3D.`,`Guarda el proyecto de nuevo en OrcaSlicer y vuelve a intentarlo.`],bad_unit:e=>[`El proyecto usa `+e.unit+` como unidad; solo se puede organizar en milímetros.`,`Convierte el modelo a milímetros en OrcaSlicer, guárdalo y vuelve a intentarlo.`],no_build:[`El archivo del proyecto no tiene la sección de construcción (build).`,`Guarda el proyecto de nuevo en OrcaSlicer y vuelve a intentarlo.`],build_layout:[`La estructura del archivo del proyecto no se puede editar de forma segura.`,`Guarda el proyecto de nuevo en OrcaSlicer (esto reescribe el archivo) y vuelve a intentarlo.`],plate_missing:e=>[`La cama `+e.plate+` no está en la lista de camas del proyecto.`,`Selecciona una cama existente y vuelve a intentarlo.`],plate_gone:e=>[`La cama `+e.plate+` no existe: el proyecto tiene `+e.plates+` `+t(e.plates,`cama`,`camas`)+`.`,`Selecciona una de las camas existentes y vuelve a intentarlo.`],plate_empty:e=>[`La cama `+e.plate+` no tiene objetos imprimibles.`,`Selecciona otra cama o añade piezas a esta.`],bed_small:t=>[`La cama de impresión es más pequeña que la cuadrícula de `+e(t.res)+` mm que se usa para colocar las piezas.`,`Revisa la forma de la cama de impresión en OrcaSlicer.`],spacing_large:t=>[`Una separación de `+e(t.spacing)+` mm no deja espacio en esta cama.`,`Escribe una separación menor que `+e(t.limit)+` mm e inténtalo de nuevo.`],clearance_large:t=>[`La cama se imprime por objeto, y el radio del margen del extrusor de la impresora, de `+e(t.clearance)+` mm, no deja espacio en esta cama.`,`Define el radio del margen del extrusor por debajo de `+e(t.limit)+` mm en los ajustes de la impresora, o imprime por capa, e inténtalo de nuevo.`],missing_requirements:e=>[`Organizar y Llenar necesitan `+e.names+`, que no se pudieron cargar.`,`Reinstala el plugin desde Orca Cloud, reinicia OrcaSlicer y vuelve a intentarlo.`],unexpected:[`Algo inesperado salió mal.`,`Inténtalo de nuevo. Si sigue ocurriendo, los detalles de abajo ayudan a diagnosticarlo.`]},saveFailed:e=>e.ref+`: no se pudo guardar la copia organizada: `+e.detail+`\n"
    "No se abrió nada; las copias anteriores no cambiaron.`,res:{placed:n=>n.ref+`: `+e(n.placed)+` `+t(n.placed,`objeto colocado`,`objetos colocados`)+`.`,added:n=>`Se llenó la cama con `+e(n.added)+` `+t(n.added,`copia más`,`copias más`)+(n.per?` (`+n.per+`).`:`.`),limit:t=>`Se detuvo en el límite de `+e(t.max)+` copias.`,tower:`Se dejó libre un área estimada para la torre de purga, borde de adherencia incluido. Su tamaño real solo se conoce después de laminar: revísalo en Orca.`,clearance:t=>`La secuencia de impresión es “Por objeto”: se mantuvo un margen XY de `+e(t.clearance)+` mm entre las piezas. Orca aún comprueba la altura y el orden de impresión.`,keep:`Rotación desactivada: cada pieza mantuvo su orientación.`,right:`La rotación se limitó a pasos de 90°.`,unplaced:e=>`No cupieron (se movieron fuera de la cama): `+e.names,foot:`Organizado a partir del proyecto tal como estaba cuando hiciste clic.`},open_:{asked:`Se pidió a OrcaSlicer que abriera la copia. Si no aparece nada, usa “Abrir esta copia en OrcaSlicer”.`,later:`Terminó con el panel cerrado. Usa “Abrir esta copia en OrcaSlicer” para cargarla.`,failed:e=>`No se pudo pedir a OrcaSlicer que abriera la copia (`+e.detail+`). Usa “Abrir esta copia en OrcaSlicer” o abre el archivo manualmente.`,none:`No se encontró ningún lanzador de OrcaSlicer: abre la copia guardada manualmente.`},copied:`Ruta copiada.`,close:`Cerrar`,notifications:`Notificaciones`,copyFail:`No se pudo copiar automáticamente. La ruta está seleccionada: usa Ctrl+C (⌘C en Mac).`,req:`Requisitos`,reqAllFound:`todos encontrados`,reqMissingMeta:e=>`faltan: `+e.names,reqBlocked:e=>`Organizar y Llenar están desactivados: no se pudo cargar `+e.names+`. Consulta Requisitos más abajo.`,reqFound:e=>`Encontrado, versión `+e.version,reqMissing:`No encontrado`,reqWhy:{numpy:`Mide el contorno de cada pieza en una cuadrícula (rasterizado de la huella) para Organizar y Llenar.`,pillow:`Dibuja el contorno de cada pieza en esa cuadrícula (rasterizado de la huella) para Organizar y Llenar.`},reqFix:`OrcaSlicer lo instala junto con el plugin. Reinstala el plugin desde Orca Cloud para que OrcaSlicer vuelva a instalar sus paquetes y reinicia OrcaSlicer.`,reqGuide:`Guía de instalación`,reqCheck:`Comprobar de nuevo`,reqChecking:`Comprobando…`}}var jy=new Intl.PluralRules(`pl-PL`),My=(e,t,n,r)=>{let i=jy.select(e);return i===`one`?t:i===`few`?n:r};function Ny(e,t){let n=My,r=t=>e(t)+` `+n(t,`obiekt`,`obiekty`,`obiektów`);return{title:`Organizer`,sub:`Shape-Aware Arrange: rozmieszcza i wypełnia płytę według konturu każdej części.`,spacingLabel:`Odstęp między częściami`,spacingHelp:`Odstęp zachowywany między konturem każdej części widzianym z góry, krawędzią stołu i obszarami wykluczonymi. Gdy płyta drukuje się obiekt po obiekcie, prześwit ekstrudera w Orca może go zwiększyć; wynik o tym informuje.`,goHelp:`Rozmieszcza obiekty z płyty. Niczego nie dodaje.`,fillHelp:`Rozmieszcza, a potem dodaje po kolei kopie obiektów z płyty, aż nic więcej się nie zmieści (do 300).`,busyHint:`Trwa operacja. Poczekaj na jej zakończenie lub ją anuluj, aby zmienić te ustawienia.`,adv:`Zaawansowane`,advMeta:`obrót, kopie`,help:`Jak to działa`,result:`Wynik`,nothingYet:`Nic jeszcze nie rozmieszczono`,nothingYetText:`Rozmieść układa obiekty z płyty według ich rzeczywistego konturu. Wypełnij dodaje potem kopie, aż nic więcej się nie zmieści.`,again:`Nowe rozmieszczenie`,statusOffline:`Brak odpowiedzi`,statusProblem:`Brak projektu`,statusReading:`Wczytywanie…`,rotationLabel:`Obrót`,optAuto:`Automatyczny (co 1°)`,optRight:`Tylko co 90°`,copiesLabel:`Maksymalna liczba kopii przy wypełnianiu`,copiesHelp:`Wypełnianie kończy się po tylu kopiach albo wcześniej, gdy płyta jest pełna. Zostaw puste, aby wypełnić płytę (do 300).`,optKeep:`Zachowaj orientację każdej części`,rotationHelp:`Automatyczny próbuje każdego kąta. Tylko co 90° zachowuje krawędzie prostopadłe do stołu w częściach, które są tak ustawione teraz. Zachowanie orientacji tylko przesuwa części. W obu przypadkach zmieści się mniej części.`,cancel:`Anuluj`,cancelPending:`Anulowanie…`,saved:`Zapisana kopia`,open:`Otwórz tę kopię w OrcaSlicer`,copy:`Kopiuj ścieżkę`,retry:`Spróbuj ponownie`,retrying:`Ponawianie…`,foot:`Odstępy liczone są według konturu części widzianego z góry. Brim, podpory i skirt nie są uwzględniane, a wieża czyszcząca jest tylko szacunkiem. Przed drukiem sprawdź płytę w Orca.`,barLabel:`Postęp`,loadingLabel:`Wczytywanie projektu`,go:e=>`Rozmieść płytę `+e.plate,fill:e=>`Wypełnij płytę `+e.plate,goNone:`Rozmieść bieżącą płytę`,fillNone:`Wypełnij bieżącą płytę`,untitled:`bez nazwy`,untitledProject:`Projekt bez nazwy`,ref:e=>e.name+`, płyta `+e.plate,sep:`: `,list:`, `,ctx:e=>e.name+` · wybrana płyta `+e.plate+(e.objects==null?``:` · `+r(e.objects)+` w projekcie`),ctxProblem:e=>`Nie można odczytać projektu: `+e.problem+`. Otwórz projekt w OrcaSlicer i spróbuj ponownie.`,reading:`Wczytywanie projektu…`,ready:`Gotowe. Wynik zostanie zapisany jako nowy plik; twój projekt się nie zmieni.`,emptyProject:`W projekcie nie znaleziono obiektów. Najpierw dodaj części na płytę.`,starting:`Uruchamianie…`,spacingErr:`Wpisz odstęp 0 mm lub większy.`,stale:e=>`Poniższa operacja dotyczy: `+e.ref+`, a nie bieżącego wyboru.`,offline:`Brak odpowiedzi od wtyczki.`,offlineHelp:`Sprawdź, czy jest nadal włączona w oknie Wtyczki w OrcaSlicer, i spróbuj ponownie. Strona sama ponawia zapytania.`,phase:{read:`Wczytywanie projektu…`,footprint:`Mierzenie konturów części`,place:`Rozmieszczanie części`,patterns:`Sprawdzanie powtarzalnych wzorów`,fill:`Dodawanie kopii`},plateLabel:e=>`Płyta `+e.plate,ctxShort:e=>`Płyta `+e.plate+(e.objects==null?``:` · `+r(e.objects)),countShort:t=>e(t.i+1)+` z `+e(t.n),count:t=>t.n==null?`Dodano `+e(t.count??0)+` `+n(t.count??0,`kopię`,`kopie`,`kopii`):e(t.i+1)+` z `+e(t.n)+(t.subject?` · `+t.subject:``),cancelling:e=>e.ref+`: anulowanie…`,cancelled:e=>e.ref+`: anulowano. Niczego nie zapisano ani nie otwarto.`,saving:e=>e.ref+`: zapisywanie rozmieszczonej kopii…`,act:{arrange:`rozmieścić`,fill:`wypełnić`},failed:e=>e.ref+`: nie udało się `+e.act+`.`,startFailed:e=>`Nie można uruchomić: `+e.why,next:`Następny krok: `,details:`Szczegóły: `,err:{bad_number:[`Odstęp musi być liczbą równą 0 mm lub większą.`,`Wpisz wartość, np. 2, i spróbuj ponownie.`],bad_step:[`Krok obrotu musi być liczbą równą 0° lub większą.`,`Użyj kroku, np. 15, albo 0, aby zachować orientację każdej części, i spróbuj ponownie.`],bad_res:[`Rozmiar siatki musi być liczbą większą od 0 mm.`,`Użyj rozmiaru siatki, np. `+e(.5)+`, i spróbuj ponownie.`],bad_copies:t=>[`Liczba kopii musi być liczbą całkowitą od 1 do `+e(t.max)+`.`,`Wpisz liczbę kopii, np. 10, albo zostaw pole puste, aby wypełnić płytę, i spróbuj ponownie.`],bad_rotation:[`Obrót musi mieć wartość „Automatyczny”, „Tylko co 90°” lub „Zachowaj orientację”.`,`Wybierz jedną z trzech opcji.`],no_model:[`Plik projektu nie zawiera modelu 3D.`,`Zapisz projekt ponownie w OrcaSlicer i spróbuj jeszcze raz.`],bad_unit:e=>[`Projekt używa jednostki `+e.unit+`; rozmieszczać można tylko w milimetrach.`,`Przelicz model na milimetry w OrcaSlicer, zapisz go i spróbuj jeszcze raz.`],no_build:[`Plik projektu nie zawiera sekcji build.`,`Zapisz projekt ponownie w OrcaSlicer i spróbuj jeszcze raz.`],build_layout:[`Układu pliku projektu nie da się bezpiecznie edytować.`,`Zapisz projekt ponownie w OrcaSlicer (plik zostanie zapisany od nowa) i spróbuj jeszcze raz.`],plate_missing:e=>[`Płyty `+e.plate+` nie ma na liście płyt projektu.`,`Wybierz istniejącą płytę i spróbuj jeszcze raz.`],plate_gone:e=>[`Płyta `+e.plate+` nie istnieje: projekt ma `+e.plates+` `+n(e.plates,`płytę`,`płyty`,`płyt`)+`.`,`Wybierz jedną z istniejących płyt i spróbuj jeszcze raz.`],plate_empty:e=>[`Płyta `+e.plate+` nie ma obiektów do druku.`,`Wybierz inną płytę albo dodaj na nią części.`],bed_small:t=>[`Stół jest mniejszy niż siatka `+e(t.res)+` mm, na której rozmieszczane są części.`,`Sprawdź kształt stołu drukarki w OrcaSlicer.`],spacing_large:t=>[`Przy odstępie `+e(t.spacing)+` mm na tym stole nie zostaje miejsca.`,`Wpisz odstęp mniejszy niż `+e(t.limit)+` mm i spróbuj ponownie.`],clearance_large:t=>[`Płyta drukuje się „Według obiektu”, a promień prześwitu ekstrudera drukarki wynoszący `+e(t.clearance)+` mm nie zostawia miejsca na tym stole.`,`Ustaw w ustawieniach drukarki promień prześwitu ekstrudera mniejszy niż `+e(t.limit)+` mm albo drukuj „Według warstwy” i spróbuj ponownie.`],missing_requirements:e=>[`Rozmieść i Wypełnij wymagają pakietów `+e.names+`, których nie udało się wczytać.`,`Zainstaluj ponownie wtyczkę z Orca Cloud, uruchom ponownie OrcaSlicer i spróbuj jeszcze raz.`],unexpected:[`Wystąpił nieoczekiwany błąd.`,`Spróbuj ponownie. Jeśli to się powtarza, szczegóły poniżej pomogą ustalić przyczynę.`]},saveFailed:e=>e.ref+`: nie udało się zapisać rozmieszczonej kopii: `+e.detail+`\n"
    "Niczego nie otwarto; wcześniejsze kopie się nie zmieniły.`,res:{placed:e=>e.ref+`: rozmieszczono `+r(e.placed)+`.`,added:t=>`Wypełniono płytę: dodano `+e(t.added)+` `+n(t.added,`kopię`,`kopie`,`kopii`)+(t.per?` (`+t.per+`).`:`.`),limit:t=>`Zatrzymano na limicie `+e(t.max)+` kopii.`,tower:`Zostawiono wolny szacowany obszar na wieżę czyszczącą, razem z brimem. Jej rzeczywisty rozmiar będzie znany dopiero po cięciu: sprawdź go w Orca.`,clearance:t=>`Kolejność drukowania to „Według obiektu”: zachowano `+e(t.clearance)+` mm prześwitu XY między częściami. Wysokość i kolejność obiektów nadal sprawdza Orca.`,keep:`Obrót był wyłączony: każda część zachowała swoją orientację.`,right:`Obrót był ograniczony do kroków co 90°.`,unplaced:e=>`Nie zmieściły się (przeniesione nad płytę): `+e.names,foot:`Rozmieszczono projekt w stanie z chwili kliknięcia.`},open_:{asked:`Poproszono OrcaSlicer o otwarcie kopii. Jeśli nic się nie pojawi, użyj „Otwórz tę kopię w OrcaSlicer”.`,later:`Zakończono, gdy panel był zamknięty. Użyj „Otwórz tę kopię w OrcaSlicer”, aby ją wczytać.`,failed:e=>`Nie udało się poprosić OrcaSlicer o otwarcie kopii (`+e.detail+`). Użyj „Otwórz tę kopię w OrcaSlicer” albo otwórz plik ręcznie.`,none:`Nie znaleziono programu uruchamiającego OrcaSlicer: otwórz zapisaną kopię ręcznie.`},copied:`Skopiowano ścieżkę.`,close:`Zamknij`,notifications:`Powiadomienia`,copyFail:`Nie udało się skopiować automatycznie. Ścieżka jest zaznaczona: naciśnij Ctrl+C (⌘C na Macu).`,req:`Wymagania`,reqAllFound:`wszystkie znalezione`,reqMissingMeta:e=>`brakuje: `+e.names,reqBlocked:e=>`Rozmieść i Wypełnij są wyłączone: nie udało się wczytać `+e.names+`. Zobacz Wymagania poniżej.`,reqFound:e=>`Znaleziono, wersja `+e.version,reqMissing:`Nie znaleziono`,reqWhy:{numpy:`Mierzy kontur każdej części na siatce (rasteryzacja obrysu) dla funkcji Rozmieść i Wypełnij.`,pillow:`Rysuje kontur każdej części na tej siatce (rasteryzacja obrysu) dla funkcji Rozmieść i Wypełnij.`},reqFix:`OrcaSlicer instaluje je razem z wtyczką. Zainstaluj ponownie wtyczkę z Orca Cloud, aby OrcaSlicer znów zainstalował jej pakiety, a potem uruchom ponownie OrcaSlicer.`,reqGuide:`Instrukcja instalacji`,reqCheck:`Sprawdź ponownie`,reqChecking:`Sprawdzanie…`}}function Py(e){return{title:`Organizer`,sub:`Shape-Aware Arrange：按每个零件的轮廓摆放并铺满盘。`,spacingLabel:`零件间距`,spacingHelp:`每个零件的俯视轮廓与其他零件、热床边缘及不可打印区域之间保留的间隙。如果该盘逐件打印，Orca 的挤出机避让空间可能会加大间距，结果中会说明。`,goHelp:`摆放盘上的对象，不添加任何内容。`,fillHelp:`先摆放，再轮流添加盘上各对象的副本，直到放不下为止（最多 300 个）。`,busyHint:`任务正在运行。请等待完成或取消后再更改这些设置。`,adv:`高级`,advMeta:`旋转、副本`,help:`工作原理`,result:`结果`,nothingYet:`尚未摆放`,nothingYetText:`“自动摆放”按真实轮廓摆放盘上的对象。“铺满”随后添加副本，直到放不下为止。`,again:`重新摆放`,statusOffline:`无响应`,statusProblem:`无项目`,statusReading:`读取中…`,rotationLabel:`旋转`,optAuto:`自动（步长 1°）`,optRight:`仅 90° 步长`,copiesLabel:`铺满时最多添加的副本数`,copiesHelp:`达到此副本数后停止铺满；盘已满时会提前停止。留空则铺满整个盘（最多 300 个）。`,optKeep:`保持每个零件的朝向`,rotationHelp:`“自动”会尝试所有角度。“仅 90° 步长”让当前边缘与热床对齐的零件保持对齐。“保持朝向”只移动零件。后两种能放下的零件较少。`,cancel:`取消`,cancelPending:`正在取消…`,saved:`已保存的副本`,open:`在 OrcaSlicer 中打开此副本`,copy:`复制路径`,retry:`重试`,retrying:`正在重试…`,foot:`零件间距按俯视轮廓计算，不包含 Brim、支撑和裙边，擦拭塔仅为估算。打印前请在 Orca 中检查该盘。`,barLabel:`进度`,loadingLabel:`正在加载项目`,go:e=>`自动摆放盘 `+e.plate,fill:e=>`铺满盘 `+e.plate,goNone:`自动摆放当前盘`,fillNone:`铺满当前盘`,untitled:`未命名`,untitledProject:`未命名项目`,ref:e=>e.name+`，盘 `+e.plate,sep:`：`,list:`、`,ctx:t=>t.name+` · 已选择盘 `+t.plate+(t.objects==null?``:` · 项目中有 `+e(t.objects)+` 个对象`),ctxProblem:e=>`无法读取项目：`+e.problem+`。请在 OrcaSlicer 中打开项目后重试。`,reading:`正在读取项目…`,ready:`就绪。结果会保存为新文件，不会更改您的项目。`,emptyProject:`项目中没有对象。请先向盘中添加零件。`,starting:`正在启动…`,spacingErr:`请输入不小于 0 mm 的间距。`,stale:e=>`下方的任务属于“`+e.ref+`”，不属于当前选择。`,offline:`插件无响应。`,offlineHelp:`请确认它在 OrcaSlicer 的“插件”窗口中仍处于启用状态，然后重试。页面会自动继续尝试连接。`,phase:{read:`正在读取项目…`,footprint:`正在测量零件轮廓`,place:`正在摆放零件`,patterns:`正在尝试重复排列`,fill:`正在添加副本`},plateLabel:e=>`盘 `+e.plate,ctxShort:t=>`盘 `+t.plate+(t.objects==null?``:` · `+e(t.objects)+` 个对象`),countShort:t=>e(t.i+1)+` / `+e(t.n),count:t=>t.n==null?`已添加 `+e(t.count??0)+` 个副本`:e(t.i+1)+` / `+e(t.n)+(t.subject?` · `+t.subject:``),cancelling:e=>e.ref+`：正在取消…`,cancelled:e=>e.ref+`：已取消。未写入或打开任何内容。`,saving:e=>e.ref+`：正在保存摆放后的副本…`,act:{arrange:`自动摆放`,fill:`铺满`},failed:e=>e.ref+`：无法`+e.act+`。`,startFailed:e=>`无法开始：`+e.why,next:`下一步：`,details:`详情：`,err:{bad_number:[`间距必须是不小于 0 mm 的数字。`,`请输入 2 之类的数值后重试。`],bad_step:[`旋转步长必须是不小于 0° 的数字。`,`请使用 15 之类的步长（0 表示保持每个零件的朝向）后重试。`],bad_res:[`网格大小必须是大于 0 mm 的数字。`,`请使用 `+e(.5)+` 之类的网格大小后重试。`],bad_copies:t=>[`副本数必须是 1 到 `+e(t.max)+` 之间的整数。`,`请输入 10 之类的副本数，或留空以铺满整个盘，然后重试。`],bad_rotation:[`旋转必须是“自动”“仅 90° 步长”或“保持朝向”。`,`请从三个选项中选择一个。`],no_model:[`项目文件中没有 3D 模型。`,`请在 OrcaSlicer 中重新保存项目后重试。`],bad_unit:e=>[`项目使用的单位是 `+e.unit+`；只能摆放以毫米为单位的项目。`,`请在 OrcaSlicer 中将模型转换为毫米，保存后重试。`],no_build:[`项目文件中没有构建（build）部分。`,`请在 OrcaSlicer 中重新保存项目后重试。`],build_layout:[`无法安全地编辑该项目文件的结构。`,`请在 OrcaSlicer 中重新保存项目（这会重写文件）后重试。`],plate_missing:e=>[`盘 `+e.plate+` 不在项目的盘列表中。`,`请选择一个已有的盘后重试。`],plate_gone:e=>[`盘 `+e.plate+` 不存在：项目共有 `+e.plates+` 个盘。`,`请选择一个已有的盘后重试。`],plate_empty:e=>[`盘 `+e.plate+` 上没有可打印的对象。`,`请选择其他盘，或向该盘添加零件。`],bed_small:t=>[`热床小于摆放零件所用的 `+e(t.res)+` mm 网格。`,`请在 OrcaSlicer 中检查打印机的热床形状。`],spacing_large:t=>[e(t.spacing)+` mm 的间距在此热床上放不下任何零件。`,`请输入小于 `+e(t.limit)+` mm 的间距后重试。`],clearance_large:t=>[`该盘逐件打印，而打印机的挤出机避让半径为 `+e(t.clearance)+` mm，在此热床上放不下任何零件。`,`请在打印机设置中将挤出机避让半径设为小于 `+e(t.limit)+` mm，或改为逐层打印，然后重试。`],missing_requirements:e=>[`自动摆放和铺满需要 `+e.names+`，但无法加载。`,`请从 Orca Cloud 重新安装插件，重启 OrcaSlicer 后重试。`],unexpected:[`发生了意外错误。`,`请重试。如果问题仍然出现，下方的详情有助于诊断。`]},saveFailed:e=>e.ref+`：无法保存摆放后的副本：`+e.detail+`\n"
    "未打开任何文件；之前的副本保持不变。`,res:{placed:t=>t.ref+`：已摆放 `+e(t.placed)+` 个对象。`,added:t=>`已用 `+e(t.added)+` 个额外副本铺满该盘`+(t.per?`（`+t.per+`）。`:`。`),limit:t=>`已达到 `+e(t.max)+` 个副本的上限，停止添加。`,tower:`已为擦拭塔预留估算区域（含 Brim）。其实际大小要在切片后才能确定，请在 Orca 中检查。`,clearance:t=>`打印顺序为“逐件”：零件之间保留了 `+e(t.clearance)+` mm 的 XY 间隙。高度和打印顺序仍由 Orca 检查。`,keep:`旋转已关闭：每个零件都保持了原有朝向。`,right:`旋转仅限 90° 步长。`,unplaced:e=>`放不下（已移到盘外）：`+e.names,foot:`按您点击时的项目状态进行摆放。`},open_:{asked:`已请求 OrcaSlicer 打开该副本。如果没有反应，请使用“在 OrcaSlicer 中打开此副本”。`,later:`任务在面板关闭期间完成。请使用“在 OrcaSlicer 中打开此副本”加载它。`,failed:e=>`无法请求 OrcaSlicer 打开该副本（`+e.detail+`）。请使用“在 OrcaSlicer 中打开此副本”，或手动打开文件。`,none:`未找到 OrcaSlicer 启动程序：请手动打开已保存的副本。`},copied:`路径已复制。`,close:`关闭`,notifications:`通知`,copyFail:`无法自动复制。路径已选中：请按 Ctrl+C（Mac 上按 ⌘C）。`,req:`依赖项`,reqAllFound:`全部已找到`,reqMissingMeta:e=>`缺少：`+e.names,reqBlocked:e=>`自动摆放和铺满已停用：无法加载 `+e.names+`。请查看下方的依赖项。`,reqFound:e=>`已找到，版本 `+e.version,reqMissing:`未找到`,reqWhy:{numpy:`在网格上测量每个零件的轮廓（占地栅格化），用于自动摆放和铺满。`,pillow:`将每个零件的轮廓绘制到该网格中（占地栅格化），用于自动摆放和铺满。`},reqFix:`OrcaSlicer 会随插件一起安装它。请从 Orca Cloud 重新安装插件，让 OrcaSlicer 重新安装所需的包，然后重启 OrcaSlicer。`,reqGuide:`安装指南`,reqCheck:`重新检查`,reqChecking:`正在检查…`}}var Fy=(e,t,n)=>e===1?t:n;function Iy(e){return{title:`Organizer`,sub:`Shape-Aware Arrange: fits and fills the plate by each part's outline.`,spacingLabel:`Spacing between parts`,spacingHelp:`Gap kept between each part's top-down outline, the bed edge and excluded areas. When the plate prints one object at a time, Orca's extruder clearance can raise it; the result says so.`,goHelp:`Fits the plate's objects. Adds nothing.`,fillHelp:`Arranges, then adds copies of the plate's objects, taking turns, until nothing more fits (up to 300).`,busyHint:`A run is in progress. Wait for it to finish, or cancel it, to change these.`,adv:`Advanced`,advMeta:`rotation, copies`,help:`How it works`,result:`Result`,nothingYet:`Nothing arranged yet`,nothingYetText:`Arrange fits the plate's objects by their real outline. Fill then adds copies until nothing more fits.`,again:`New arrangement`,statusOffline:`No answer`,statusProblem:`No project`,statusReading:`Reading…`,rotationLabel:`Rotation`,optAuto:`Automatic (1° steps)`,optRight:`Only 90° steps`,copiesLabel:`Most copies to add when filling`,copiesHelp:`Fill stops after this many copies, or sooner when the plate is full. Leave it empty to fill the plate (up to 300).`,optKeep:`Keep each part's orientation`,rotationHelp:`Automatic tries every angle. Only 90° steps keeps edges square to the bed for parts that are square now. Keeping the orientation only moves parts. Both fit fewer parts.`,cancel:`Cancel`,cancelPending:`Cancelling…`,saved:`Saved copy`,open:`Open this copy in OrcaSlicer`,copy:`Copy path`,retry:`Try again`,retrying:`Trying…`,foot:`Parts are spaced by their top-down outline. Brims, supports and skirts are not included, and a prime tower is only an estimate. Check the plate in Orca before printing.`,barLabel:`Progress`,loadingLabel:`Loading the project`,go:e=>`Arrange plate `+e.plate,fill:e=>`Fill plate `+e.plate,goNone:`Arrange current plate`,fillNone:`Fill current plate`,untitled:`untitled`,untitledProject:`Untitled project`,ref:e=>e.name+`, plate `+e.plate,sep:`: `,list:`, `,ctx:t=>t.name+` · plate `+t.plate+` selected`+(t.objects==null?``:` · `+e(t.objects)+` `+Fy(t.objects,`object`,`objects`)+` in the project`),ctxProblem:e=>`Could not read the project: `+e.problem+`. Open a project in OrcaSlicer, then try again.`,reading:`Reading project…`,ready:`Ready. The result is saved as a new file; your project is not changed.`,emptyProject:`No objects found in the project. Add parts to a plate first.`,starting:`Starting…`,spacingErr:`Enter a spacing of 0 mm or more.`,stale:e=>`The run below is for `+e.ref+`, not the current selection.`,offline:`No answer from the plugin.`,offlineHelp:`Check that it is still enabled in the Plugins window of OrcaSlicer, then try again. The page keeps asking by itself.`,phase:{read:`Reading the project…`,footprint:`Measuring part outlines`,place:`Placing parts`,patterns:`Trying repeating patterns`,fill:`Adding copies`},plateLabel:e=>`Plate `+e.plate,ctxShort:t=>`Plate `+t.plate+(t.objects==null?``:` · `+e(t.objects)+` `+Fy(t.objects,`object`,`objects`)),countShort:t=>e(t.i+1)+` of `+e(t.n),count:t=>t.n==null?e(t.count??0)+` `+Fy(t.count??0,`copy`,`copies`)+` added`:e(t.i+1)+` of `+e(t.n)+(t.subject?` · `+t.subject:``),cancelling:e=>e.ref+`: cancelling…`,cancelled:e=>e.ref+`: cancelled. Nothing was written or opened.`,saving:e=>e.ref+`: saving the arranged copy…`,act:{arrange:`arrange`,fill:`fill`},failed:e=>e.ref+`: could not `+e.act+`.`,startFailed:e=>`Could not start: `+e.why,next:`Next: `,details:`Details: `,err:{bad_number:[`Spacing must be a number, 0 mm or more.`,`Enter a value such as 2 and try again.`],bad_step:[`The rotation step must be a number, 0° or more.`,`Use a step such as 15, or 0 to keep each part's orientation, and try again.`],bad_res:[`The grid size must be a number above 0 mm.`,`Use a grid size such as 0.5 and try again.`],bad_copies:t=>[`The number of copies must be a whole number from 1 to `+e(t.max)+`.`,`Enter a number of copies such as 10, or leave it empty to fill the plate, and try again.`],bad_rotation:[`Rotation must be automatic, only 90° steps or keep orientation.`,`Choose one of the three options.`],no_model:[`The project file has no 3D model.`,`Save the project again in OrcaSlicer and retry.`],bad_unit:e=>[`The project uses `+e.unit+` as its unit; only millimetres can be arranged.`,`Convert the model to millimetres in OrcaSlicer, save it and retry.`],no_build:[`The project file has no build section.`,`Save the project again in OrcaSlicer and retry.`],build_layout:[`The project file's layout cannot be edited safely.`,`Save the project again in OrcaSlicer (this rewrites the file) and retry.`],plate_missing:e=>[`Plate `+e.plate+` is not in the project's plate list.`,`Select an existing plate and retry.`],plate_gone:e=>[`Plate `+e.plate+` does not exist: the project has `+e.plates+` `+Fy(e.plates,`plate`,`plates`)+`.`,`Select one of the existing plates and retry.`],plate_empty:e=>[`Plate `+e.plate+` has no printable objects.`,`Select another plate, or add parts to this one.`],bed_small:t=>[`The bed is smaller than the `+e(t.res)+` mm grid used to place parts.`,`Check the printer's bed shape in OrcaSlicer.`],spacing_large:t=>[`A spacing of `+e(t.spacing)+` mm leaves no room on this bed.`,`Enter a spacing below `+e(t.limit)+` mm and try again.`],clearance_large:t=>[`The plate prints by object, and the printer's extruder clearance radius of `+e(t.clearance)+` mm leaves no room on this bed.`,`Set the extruder clearance radius below `+e(t.limit)+` mm in the printer settings, or print layer by layer, and try again.`],missing_requirements:e=>[`Arrange and Fill need `+e.names+`, which could not be loaded.`,`Reinstall the plugin from Orca Cloud, restart OrcaSlicer and try again.`],unexpected:[`Something unexpected went wrong.`,`Try again. If it keeps happening, the details below help to diagnose it.`]},saveFailed:e=>e.ref+`: could not save the arranged copy: `+e.detail+`\n"
    "Nothing was opened; earlier copies are unchanged.`,res:{placed:t=>t.ref+`: placed `+e(t.placed)+` `+Fy(t.placed,`object`,`objects`)+`.`,added:t=>`Filled the plate with `+e(t.added)+` extra `+Fy(t.added,`copy`,`copies`)+(t.per?` (`+t.per+`).`:`.`),limit:t=>`Stopped at the `+e(t.max)+`-copy limit.`,tower:`Kept an estimated prime tower area free, brim included. Its real size is only known after slicing: check it in Orca.`,clearance:t=>`Print sequence is By object: kept `+e(t.clearance)+` mm XY clearance between parts. Orca still checks height and print order.`,keep:`Rotation was off: every part kept its orientation.`,right:`Rotation was limited to 90° steps.`,unplaced:e=>`Did not fit (moved above the plate): `+e.names,foot:`Arranged from the project as it was when you clicked.`},open_:{asked:`Asked OrcaSlicer to open the copy. If nothing appears, use Open this copy in OrcaSlicer.`,later:`Finished while the panel was closed. Use Open this copy in OrcaSlicer to load it.`,failed:e=>`Could not ask OrcaSlicer to open it (`+e.detail+`). Use Open this copy in OrcaSlicer or open the file by hand.`,none:`No OrcaSlicer launcher found: open the saved copy by hand.`},copied:`Path copied.`,close:`Close`,notifications:`Notifications`,copyFail:`Could not copy automatically. The path is selected: press Ctrl+C (⌘C on Mac).`,req:`Requirements`,reqAllFound:`all found`,reqMissingMeta:e=>`missing: `+e.names,reqBlocked:e=>`Arrange and Fill are off: `+e.names+` could not be loaded. See Requirements below.`,reqFound:e=>`Found, version `+e.version,reqMissing:`Not found`,reqWhy:{numpy:`Measures each part's outline on a grid (footprint rasterising) for Arrange and Fill.`,pillow:`Draws each part's outline into that grid (footprint rasterising) for Arrange and Fill.`},reqFix:`OrcaSlicer installs it with the plugin. Reinstall the plugin from Orca Cloud so OrcaSlicer installs its packages again, then restart OrcaSlicer.`,reqGuide:`Installation guide`,reqCheck:`Check again`,reqChecking:`Checking…`}}function Ly(e){return{title:`Organizer`,sub:`Shape-Aware Arrange: encaixa e preenche a placa pelo contorno de cada peça.`,spacingLabel:`Espaçamento entre peças`,spacingHelp:`Folga mantida entre o contorno de cada peça visto de cima, a borda da mesa e as áreas excluídas. Se a placa imprime um objeto por vez, a folga da extrusora do Orca pode aumentá-la; o resultado avisa.`,goHelp:`Encaixa os objetos da placa. Não adiciona nada.`,fillHelp:`Arruma e depois adiciona cópias dos objetos da placa, em rodízio, até não caber mais nada (até 300).`,busyHint:`Uma execução está em andamento. Aguarde o fim, ou cancele, para alterar estes campos.`,adv:`Avançado`,advMeta:`rotação, cópias`,help:`Como funciona`,result:`Resultado`,nothingYet:`Nada arrumado ainda`,nothingYetText:`Arrumar encaixa os objetos da placa pelo contorno real. Preencher depois adiciona cópias até não caber mais nada.`,again:`Nova arrumação`,statusOffline:`Sem resposta`,statusProblem:`Sem projeto`,statusReading:`Lendo…`,rotationLabel:`Rotação`,optAuto:`Automática (passos de 1°)`,optRight:`Só passos de 90°`,copiesLabel:`Máximo de cópias ao preencher`,copiesHelp:`O preenchimento para depois deste número de cópias, ou antes, se a placa encher. Deixe vazio para preencher a placa (até 300).`,optKeep:`Manter a orientação de cada peça`,rotationHelp:`A automática testa todos os ângulos. Só passos de 90° mantém as bordas no esquadro da mesa para as peças que já estão no esquadro. Manter a orientação só move as peças. Nos dois casos cabem menos peças.`,cancel:`Cancelar`,cancelPending:`Cancelando…`,saved:`Cópia salva`,open:`Abrir esta cópia no OrcaSlicer`,copy:`Copiar caminho`,retry:`Tentar novamente`,retrying:`Tentando…`,foot:`As peças são espaçadas pelo contorno visto de cima. Brim, suportes e skirt não entram na conta, e a torre de limpeza (prime tower) é só uma estimativa. Confira a placa no Orca antes de imprimir.`,barLabel:`Progresso`,loadingLabel:`Carregando o projeto`,go:e=>`Arrumar placa `+e.plate,fill:e=>`Preencher placa `+e.plate,goNone:`Arrumar placa atual`,fillNone:`Preencher placa atual`,untitled:`sem título`,untitledProject:`Projeto sem título`,ref:e=>e.name+`, placa `+e.plate,sep:`: `,list:`, `,ctx:t=>t.name+` · placa `+t.plate+` selecionada`+(t.objects==null?``:` · `+e(t.objects)+` `+Fy(t.objects,`objeto`,`objetos`)+` no projeto`),ctxProblem:e=>`Não foi possível ler o projeto: `+e.problem+`. Abra um projeto no OrcaSlicer e tente novamente.`,reading:`Lendo o projeto…`,ready:`Pronto. O resultado é salvo em um arquivo novo; o seu projeto não é alterado.`,emptyProject:`Nenhum objeto encontrado no projeto. Adicione peças a uma placa primeiro.`,starting:`Iniciando…`,spacingErr:`Informe um espaçamento de 0 mm ou mais.`,stale:e=>`A execução abaixo é de `+e.ref+`, não da seleção atual.`,offline:`Sem resposta do plugin.`,offlineHelp:`Confira se ele continua ativado na janela de Plugins do OrcaSlicer e tente novamente. A página continua perguntando sozinha.`,phase:{read:`Lendo o projeto…`,footprint:`Medindo o contorno das peças`,place:`Posicionando as peças`,patterns:`Testando padrões repetidos`,fill:`Adicionando cópias`},plateLabel:e=>`Placa `+e.plate,ctxShort:t=>`Placa `+t.plate+(t.objects==null?``:` · `+e(t.objects)+` `+Fy(t.objects,`objeto`,`objetos`)),countShort:t=>e(t.i+1)+` de `+e(t.n),count:t=>t.n==null?e(t.count??0)+` `+Fy(t.count??0,`cópia adicionada`,`cópias adicionadas`):e(t.i+1)+` de `+e(t.n)+(t.subject?` · `+t.subject:``),cancelling:e=>e.ref+`: cancelando…`,cancelled:e=>e.ref+`: cancelado. Nada foi gravado nem aberto.`,saving:e=>e.ref+`: salvando a cópia arrumada…`,act:{arrange:`arrumar`,fill:`preencher`},failed:e=>e.ref+`: não foi possível `+e.act+`.`,startFailed:e=>`Não foi possível iniciar: `+e.why,next:`Próximo passo: `,details:`Detalhe técnico: `,err:{bad_number:[`O espaçamento deve ser um número de 0 mm ou mais.`,`Digite um valor como 2 e tente novamente.`],bad_step:[`O passo de rotação deve ser um número de 0° ou mais.`,`Use um passo como 15, ou 0 para manter a orientação de cada peça, e tente novamente.`],bad_res:[`O tamanho da grade deve ser um número acima de 0 mm.`,`Use um tamanho de grade como 0,5 e tente novamente.`],bad_copies:t=>[`O número de cópias deve ser um número inteiro de 1 a `+e(t.max)+`.`,`Digite um número de cópias como 10, ou deixe vazio para preencher a placa, e tente novamente.`],bad_rotation:[`A rotação deve ser automática, só passos de 90° ou manter a orientação.`,`Escolha uma das três opções.`],no_model:[`O arquivo do projeto não tem modelo 3D.`,`Salve o projeto de novo no OrcaSlicer e tente novamente.`],bad_unit:e=>[`O projeto usa `+e.unit+` como unidade; só é possível arrumar em milímetros.`,`Converta o modelo para milímetros no OrcaSlicer, salve e tente novamente.`],no_build:[`O arquivo do projeto não tem a seção de montagem (build).`,`Salve o projeto de novo no OrcaSlicer e tente novamente.`],build_layout:[`A estrutura do arquivo do projeto não pode ser editada com segurança.`,`Salve o projeto de novo no OrcaSlicer (isso reescreve o arquivo) e tente novamente.`],plate_missing:e=>[`A placa `+e.plate+` não está na lista de placas do projeto.`,`Selecione uma placa existente e tente novamente.`],plate_gone:e=>[`A placa `+e.plate+` não existe: o projeto tem `+e.plates+` `+Fy(e.plates,`placa`,`placas`)+`.`,`Selecione uma das placas existentes e tente novamente.`],plate_empty:e=>[`A placa `+e.plate+` não tem objetos imprimíveis.`,`Selecione outra placa ou adicione peças a esta.`],bed_small:t=>[`A mesa é menor que a grade de `+e(t.res)+` mm usada para posicionar as peças.`,`Confira o formato da mesa da impressora no OrcaSlicer.`],spacing_large:t=>[`Um espaçamento de `+e(t.spacing)+` mm não deixa espaço nesta mesa.`,`Digite um espaçamento menor que `+e(t.limit)+` mm e tente novamente.`],clearance_large:t=>[`A placa imprime um objeto por vez, e o raio de folga da extrusora da impressora, de `+e(t.clearance)+` mm, não deixa espaço nesta mesa.`,`Defina o raio de folga da extrusora abaixo de `+e(t.limit)+` mm nas configurações da impressora, ou imprima camada por camada, e tente novamente.`],missing_requirements:e=>[`Arrumar e Preencher precisam de `+e.names+`, que não puderam ser carregados.`,`Reinstale o plugin pelo Orca Cloud, reinicie o OrcaSlicer e tente novamente.`],unexpected:[`Algo inesperado deu errado.`,`Tente novamente. Se continuar, o detalhe técnico abaixo ajuda a diagnosticar.`]},saveFailed:e=>e.ref+`: não foi possível salvar a cópia arrumada: `+e.detail+`\n"
    "Nada foi aberto; as cópias anteriores continuam intactas.`,res:{placed:t=>t.ref+`: `+e(t.placed)+` `+Fy(t.placed,`objeto posicionado`,`objetos posicionados`)+`.`,added:t=>`Placa preenchida com `+e(t.added)+` `+Fy(t.added,`cópia extra`,`cópias extras`)+(t.per?` (`+t.per+`).`:`.`),limit:t=>`Parou no limite de `+e(t.max)+` cópias.`,tower:`Área estimada da torre de limpeza (prime tower) mantida livre, brim incluído. O tamanho real só é conhecido depois do fatiamento: confira no Orca.`,clearance:t=>`A sequência de impressão é Por objeto: mantida folga XY de `+e(t.clearance)+` mm entre as peças. O Orca ainda confere altura e ordem de impressão.`,keep:`Rotação desligada: cada peça manteve a orientação.`,right:`A rotação ficou limitada a passos de 90°.`,unplaced:e=>`Não couberam (movidas para fora da placa): `+e.names,foot:`Arrumado a partir do projeto como estava quando você clicou.`},open_:{asked:`Pedi ao OrcaSlicer que abrisse a cópia. Se nada aparecer, use Abrir esta cópia no OrcaSlicer.`,later:`Concluído com o painel fechado. Use Abrir esta cópia no OrcaSlicer para carregá-la.`,failed:e=>`Não foi possível pedir ao OrcaSlicer que abrisse a cópia (`+e.detail+`). Use Abrir esta cópia no OrcaSlicer ou abra o arquivo manualmente.`,none:`Nenhum inicializador do OrcaSlicer encontrado: abra a cópia salva manualmente.`},copied:`Caminho copiado.`,close:`Fechar`,notifications:`Notificações`,copyFail:`Não foi possível copiar automaticamente. O caminho está selecionado: pressione Ctrl+C (⌘C no Mac).`,req:`Requisitos`,reqAllFound:`todos encontrados`,reqMissingMeta:e=>`faltando: `+e.names,reqBlocked:e=>`Arrumar e Preencher estão desativados: `+e.names+` não puderam ser carregados. Veja Requisitos abaixo.`,reqFound:e=>`Encontrado, versão `+e.version,reqMissing:`Não encontrado`,reqWhy:{numpy:`Mede o contorno de cada peça numa grade (rasterização da área ocupada) para Arrumar e Preencher.`,pillow:`Desenha o contorno de cada peça nessa grade (rasterização da área ocupada) para Arrumar e Preencher.`},reqFix:`O OrcaSlicer instala junto com o plugin. Reinstale o plugin pelo Orca Cloud para o OrcaSlicer instalar os pacotes de novo e reinicie o OrcaSlicer.`,reqGuide:`Guia de instalação`,reqCheck:`Verificar de novo`,reqChecking:`Verificando…`}}var Ry=new Map([[`pt`,`pt-BR`],[`es`,`es`],[`de`,`de`],[`zh`,`zh-CN`],[`pl`,`pl`]]),zy=e=>Ry.get((e||``).toLowerCase().split(/[-_]/)[0])??`en`,By={en:`en-US`,\"pt-BR\":`pt-BR`,es:`es`,de:`de`,\"zh-CN\":`zh-CN`,pl:`pl-PL`};function Vy(e){if(e===`es`||e===`de`){let t=(typeof navigator<`u`?navigator.languages??[]:[]).find(t=>t.toLowerCase().startsWith(e+`-`));if(t&&Intl.NumberFormat.supportedLocalesOf(t).length)return t}return By[e]}var Hy=new Map;function Uy(e){let t=Hy.get(e);t||(t=new Intl.NumberFormat(Vy(e),{maximumFractionDigits:2}),Hy.set(e,t));let n=t;return e=>n.format(e)}function Wy(e){let t=new Intl.PluralRules(By[e]);return(e,n,r)=>t.select(e)===`one`?n:r}var Gy={en:Iy,\"pt-BR\":Ly,es:Ay,de:ky,\"zh-CN\":Py,pl:Ny},Ky=new Map;function qy(e){let t=Ky.get(e);return t||(t=Gy[e](Uy(e),Wy(e)),Ky.set(e,t)),t}var Jy=qy(`en`);function Yy(){let e=globalThis.orca;if(!e||typeof e.postMessage!=`function`||typeof e.onMessage!=`function`)throw Error(`The Orca bridge (orca.postMessage / orca.onMessage) is not available in this page.`);return e}var Xy={id:`spacing-label`,for:`spacing`},Zy={class:`actions`},Qy={class:`more`},$y={id:`adv`},eb={class:`sum`},tb={class:`oui-meta`},nb={class:`more-body`},rb={class:`rotation`,\"aria-describedby\":`rotation-help`},ib={id:`rotation-label`},ab={id:`rotation-help`},ob=[`for`],sb={class:`oui-field`},cb={class:`labelled`},lb={for:`copies`},ub={id:`copies-help`},db={class:`more`},fb={id:`help`},pb={class:`sum`},mb={class:`more-body`},hb={id:`subtitle`,class:`oui-meta`},gb={class:`terms`},_b={id:`spacing-help`,class:`oui-meta`},vb={id:`go-help`,class:`oui-meta`},yb={id:`fill-help`,class:`oui-meta`},bb=[`open`],xb={id:`req`},Sb={class:`sum`},Cb={class:`oui-meta`},wb={class:`more-body`},Tb={class:`reqs`},Eb={key:1,id:`busy-hint`,class:`oui-sr`},Db=[`data-i`],Ob={id:`foot`,class:`oui-meta`},kb=P({__name:`ArrangeForm`,setup(e){let{s:t,lang:n,state:r,goLabel:i,fillLabel:a,statusView:o,ready:s,missing:c,blockedText:l,run:u,recheck:d,setSpacing:f,setRotation:p,setCopies:m}=Zv(),h=W(()=>Vy(n.value)),g=W(()=>new Intl.NumberFormat(h.value).formatToParts(1.5).find(e=>e.type===`decimal`)?.value??`.`);function _(e){e.key===`.`&&g.value!==`.`&&(e.preventDefault(),e.stopPropagation(),e.target?.dispatchEvent(new KeyboardEvent(`keypress`,{key:g.value,bubbles:!0,cancelable:!0})))}function v(e){let t=(e.clipboardData?.getData(`text`)??``).replace(/mm/i,``).trim(),n=/^\\d*\\.\\d+$|^\\d+\\.?$/.test(t)?Number(t):Number(t.split(g.value).map(e=>e.replace(/\\D/g,``)).join(`.`));/[.,]/.test(t)&&!t.includes(`-`)&&Number.isFinite(n)&&(e.preventDefault(),e.stopPropagation(),f(n))}let y=W(()=>[{value:`auto`,label:t.value.optAuto},{value:`right`,label:t.value.optRight},{value:`keep`,label:t.value.optKeep}]),b=W(()=>({pcInputText:{root:{\"aria-describedby\":r.validation?`spacing-error spacing-help`:`spacing-help`}}})),x=e=>[e,r.busy&&`busy-hint`,l.value&&`req-hint`].filter(Boolean).join(` `),S={numpy:`https://numpy.org/install/`,pillow:`https://pillow.readthedocs.io/en/stable/installation/`},C=e=>e.status===`ok`?`found`:e.status===`missing`?`missing`:`checking`,w=e=>e.status===`ok`?t.value.reqFound({version:e.version??``}):e.status===`missing`?t.value.reqMissing:t.value.reqChecking,ee=W(()=>r.busy||c.value.length>0),te=W(()=>s.value&&!r.busy?o.value.lines:[]);return(e,n)=>(L(),R(`form`,{class:`form`,novalidate:``,onSubmit:n[2]||=ws(()=>{},[`prevent`])},[B(`div`,{class:`oui-field`,onKeypressCapture:_,onPasteCapture:v},[B(`label`,Xy,k(M(t).spacingLabel),1),V(M(vv),{\"input-id\":`spacing`,\"model-value\":M(r).spacing,min:0,step:.5,\"min-fraction-digits\":0,\"max-fraction-digits\":3,locale:h.value,suffix:` mm`,\"show-buttons\":``,disabled:M(r).busy,invalid:M(r).spacingInvalid,pt:b.value,title:M(t).spacingHelp,fluid:``,\"onUpdate:modelValue\":M(f)},null,8,[`model-value`,`locale`,`disabled`,`invalid`,`pt`,`title`,`onUpdate:modelValue`]),M(r).validation?(L(),z(M(iy),{key:0,id:`spacing-error`,severity:`error`,closable:!1},{default:N(()=>[Ta(k(M(t).spacingErr),1)]),_:1})):H(``,!0)],32),B(`div`,Zy,[V(M(Yg),{id:`go`,label:M(i),disabled:ee.value,\"aria-describedby\":x(`go-help`),title:M(t).goHelp,onClick:n[0]||=e=>M(u)(`arrange`)},{icon:N(()=>[V(M(Rm),{icon:M(Qm),size:14},null,8,[`icon`])]),_:1},8,[`label`,`disabled`,`aria-describedby`,`title`]),V(M(Yg),{id:`fill`,label:M(a),severity:`secondary`,disabled:ee.value,\"aria-describedby\":x(`fill-help`),title:M(t).fillHelp,onClick:n[1]||=e=>M(u)(`fill`)},{icon:N(()=>[V(M(Rm),{icon:M(sh),size:14},null,8,[`icon`])]),_:1},8,[`label`,`disabled`,`aria-describedby`,`title`])]),B(`details`,Qy,[B(`summary`,$y,[B(`span`,eb,[V(M(Rm),{icon:M(th),size:14},null,8,[`icon`]),Ta(k(M(t).adv),1),B(`span`,tb,k(M(t).advMeta),1)])]),B(`div`,nb,[B(`fieldset`,rb,[B(`legend`,ib,k(M(t).rotationLabel),1),V(M(h_),{class:`hint`,label:M(t).rotationLabel},{default:N(()=>[B(`span`,ab,k(M(t).rotationHelp),1)]),_:1},8,[`label`]),(L(!0),R(I,null,Wr(y.value,e=>(L(),R(`div`,{key:e.value,class:`choice`},[V(M(Cy),{\"model-value\":M(r).rotation,\"input-id\":`rotation-`+e.value,name:`rotation`,value:e.value,disabled:M(r).busy,\"onUpdate:modelValue\":M(p)},null,8,[`model-value`,`input-id`,`value`,`disabled`,`onUpdate:modelValue`]),B(`label`,{for:`rotation-`+e.value},k(e.label),9,ob)]))),128))]),B(`div`,sb,[B(`div`,cb,[B(`label`,lb,k(M(t).copiesLabel),1),V(M(h_),{label:M(t).copiesLabel},{default:N(()=>[B(`span`,ub,k(M(t).copiesHelp),1)]),_:1},8,[`label`])]),V(M(vv),{\"input-id\":`copies`,\"model-value\":M(r).copies,min:1,max:M(300),\"max-fraction-digits\":0,\"use-grouping\":!1,disabled:M(r).busy,pt:{pcInputText:{root:{\"aria-describedby\":`copies-help`}}},size:`small`,fluid:``,\"onUpdate:modelValue\":M(m)},null,8,[`model-value`,`max`,`disabled`,`onUpdate:modelValue`])])])]),B(`details`,db,[B(`summary`,fb,[B(`span`,pb,[V(M(Rm),{icon:M(ch),size:14},null,8,[`icon`]),Ta(k(M(t).help),1)])]),B(`div`,mb,[B(`p`,hb,k(M(t).sub),1),B(`dl`,gb,[B(`dt`,null,k(M(t).spacingLabel),1),B(`dd`,_b,k(M(t).spacingHelp),1),B(`dt`,null,k(M(i)),1),B(`dd`,vb,k(M(t).goHelp),1),B(`dt`,null,k(M(a)),1),B(`dd`,yb,k(M(t).fillHelp),1)])])]),M(r).requirements.length?(L(),R(`details`,{key:0,class:`more`,open:M(c).length>0},[B(`summary`,xb,[B(`span`,Sb,[V(M(Rm),{icon:M(th),size:14},null,8,[`icon`]),Ta(k(M(t).req),1),B(`span`,Cb,k(M(c).length?M(t).reqMissingMeta({names:M(c).join(M(t).list)}):M(r).requirements.some(e=>e.status===`unknown`)?M(t).reqChecking:M(t).reqAllFound),1)])]),B(`div`,wb,[B(`div`,Tb,[(L(!0),R(I,null,Wr(M(r).requirements,e=>(L(),z(M(Q_),{key:e.id,\"data-req\":e.id,name:e.name,why:M(t).reqWhy[e.id],status:C(e),\"status-label\":w(e),detail:e.status===`missing`?e.detail??void 0:void 0,\"install-text\":M(t).reqFix,\"install-url\":S[e.id],\"install-label\":M(t).reqGuide,\"check-label\":M(r).rechecking?M(t).reqChecking:M(t).reqCheck,checking:M(r).rechecking,onCheck:M(d)},null,8,[`data-req`,`name`,`why`,`status`,`status-label`,`detail`,`install-text`,`install-url`,`install-label`,`check-label`,`checking`,`onCheck`]))),128))])])],8,bb)):H(``,!0),M(r).busy?(L(),R(`p`,Eb,k(M(t).busyHint),1)):H(``,!0),(L(!0),R(I,null,Wr(te.value,(e,t)=>(L(),R(`p`,{key:e.key,class:`oui-meta`,\"data-i\":t},k(e.text),9,Db))),128)),B(`p`,Ob,k(M(t).foot),1)],32))}}),Ab=(e,t)=>{let n=e.__vccOpts||e;for(let[e,r]of t)n[e]=r;return n},jb=Ab(kb,[[`__scopeId`,`data-v-3616ab9b`]]),Mb=`data:image/webp;base64,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`,Nb=P({__name:`IdlePanel`,setup(e){let{s:t,state:n}=Zv();return(e,r)=>!M(n).busy&&!M(n).resultVisible?(L(),z(M(C_),{key:0,class:`idle`,title:M(t).result,icon:M(oh),tone:`neutral`},{default:N(()=>[V(M(Dh),{image:M(Mb),title:M(t).nothingYet,text:M(t).nothingYetText},null,8,[`image`,`title`,`text`])]),_:1},8,[`title`,`icon`])):H(``,!0)}}),Pb=[`aria-label`],Fb=Ab(P({__name:`JobProgress`,setup(e){let{s:t,state:n,percent:r,jobView:i,countText:a,cancel:o}=Zv(),s=$t(null);return Wn(()=>{let e=s.value?.querySelector(`.oui-job-detail`);e&&e.setAttribute(`title`,i.value.detail);let t=s.value?.querySelector(`[role=progressbar]`);r.value,t&&(a.value?t.setAttribute(`aria-valuetext`,a.value):t.removeAttribute(`aria-valuetext`))},{flush:`post`}),(e,a)=>M(n).busy?(L(),R(`section`,{key:0,id:`job`,ref_key:`root`,ref:s,class:`job`,\"aria-label\":M(t).barLabel},[V(M(L_),{compact:``,name:M(i).name,progress:M(r),\"progress-label\":M(t).barLabel,count:M(i).figure,detail:M(i).detail},{status:N(()=>[V(M(V_),{state:`busy`,label:M(i).phase},null,8,[`label`])]),actions:N(()=>[V(M(Yg),{id:`cancel`,label:M(n).cancelling?M(t).cancelPending:M(t).cancel,severity:`secondary`,\"aria-disabled\":M(n).cancelling,\"aria-busy\":M(n).cancelling,fluid:``,onClick:a[0]||=e=>M(o)()},{icon:N(()=>[V(M(Rm),{icon:M(ph),size:14},null,8,[`icon`])]),_:1},8,[`label`,`aria-disabled`,`aria-busy`])]),_:1},8,[`name`,`progress`,`progress-label`,`count`,`detail`])],8,Pb)):H(``,!0)}}),[[`__scopeId`,`data-v-3908b3ac`]]),Ib=[`title`,`aria-busy`],Lb=[`aria-label`],Rb={class:`oui-sr`},zb={\"aria-hidden\":`true`},Bb={class:`context-name`},Vb=P({__name:`PanelHeader`,setup(e){let{s:t,state:n,ctxText:r,ctxShort:i,ctxProblem:a,projectName:o,loading:s,again:c}=Zv();return(e,l)=>(L(),z(M(Zm),{name:M(t).title},Gr({mark:N(()=>[l[1]||=B(`svg`,{viewBox:`0 0 24 24`,fill:`currentColor`},[B(`path`,{\"fill-rule\":`evenodd`,d:`M5.5 2.2h13a3.3 3.3 0 0 1 3.3 3.3v13a3.3 3.3 0 0 1-3.3 3.3h-13a3.3 3.3 0 0 1-3.3-3.3v-13a3.3 3.3 0 0 1 3.3-3.3zm0 1.5a1.8 1.8 0 0 0-1.8 1.8v13a1.8 1.8 0 0 0 1.8 1.8h13a1.8 1.8 0 0 0 1.8-1.8v-13a1.8 1.8 0 0 0-1.8-1.8z`}),B(`rect`,{x:`4.7`,y:`4.7`,width:`9.3`,height:`9.5`}),B(`rect`,{x:`14.8`,y:`4.7`,width:`4.5`,height:`9.5`}),B(`rect`,{x:`4.7`,y:`15`,width:`4.5`,height:`4.3`}),B(`rect`,{x:`10`,y:`15`,width:`9.3`,height:`4.3`})],-1)]),status:N(()=>[B(`span`,{id:`context`,class:`context oui-truncate`,title:M(s)?M(t).loadingLabel:M(r),\"aria-busy\":M(s)},[M(n).offline?(L(),z(M(V_),{key:0,state:`offline`,label:M(t).statusOffline},null,8,[`label`])):M(s)?(L(),R(I,{key:1},[V(M(V_),{state:`connecting`,label:M(t).statusReading},null,8,[`label`]),B(`span`,{id:`loading`,class:`oui-sr`,role:`progressbar`,\"aria-label\":M(t).loadingLabel},null,8,Lb)],64)):M(a)?(L(),z(M(V_),{key:2,state:`warn`,label:M(t).statusProblem},null,8,[`label`])):M(i)?(L(),R(I,{key:3},[B(`span`,Rb,k(M(r)),1),B(`span`,zb,[B(`span`,Bb,k(M(o))+` · `,1),Ta(k(M(i)),1)])],64)):(L(),R(I,{key:4},[Ta(k(M(r)),1)],64))],8,Ib)]),_:2},[M(n).resultVisible&&!M(n).formOpen&&!M(n).busy?{name:`actions`,fn:N(()=>[V(M(Yg),{id:`again`,class:`again`,\"aria-label\":M(t).again,title:M(t).again,severity:`secondary`,size:`small`,onClick:l[0]||=e=>M(c)()},{icon:N(()=>[V(M(Rm),{icon:M(Qm),size:14},null,8,[`icon`])]),_:1},8,[`aria-label`,`title`])]),key:`0`}:void 0]),1032,[`name`]))}}),Hb={id:`offline-text`},Ub=Ab(P({__name:`PanelNotices`,setup(e){let{s:t,state:n,staleText:r,ctxProblem:i,blockedText:a,retry:o}=Zv();return(e,s)=>(L(),R(I,null,[V(M(yh),{show:M(n).offline,id:`offline`,tone:`error`,title:M(t).offline,dismissible:``,\"dismiss-key\":`offline:${M(n).offlineAt}`,\"close-label\":M(t).close},{actions:N(()=>[V(M(Yg),{id:`retry`,label:M(n).retrying?M(t).retrying:M(t).retry,severity:`secondary`,size:`small`,loading:M(n).retrying,onClick:s[0]||=e=>M(o)()},{icon:N(()=>[V(M(Rm),{icon:M(dh),size:14},null,8,[`icon`])]),_:1},8,[`label`,`loading`])]),default:N(()=>[B(`span`,Hb,k(M(t).offlineHelp),1)]),_:1},8,[`show`,`title`,`dismiss-key`,`close-label`]),V(M(yh),{show:!!M(i),class:`context-problem`,tone:`warn`,role:`alert`,title:M(i),dismissible:``,\"dismiss-key\":`ctx-problem:${M(i)}`,\"close-label\":M(t).close},null,8,[`show`,`title`,`dismiss-key`,`close-label`]),V(M(yh),{show:!!M(a),id:`req-hint`,tone:`warn`,title:M(a),dismissible:``,\"dismiss-key\":`req:${M(a)}`,\"close-label\":M(t).close},null,8,[`show`,`title`,`dismiss-key`,`close-label`]),V(M(yh),{show:!!(M(n).stale||M(n).staleRef),id:`stale`,tone:`warn`,title:M(r),dismissible:``,\"dismiss-key\":`stale:${M(r)}`,\"close-label\":M(t).close},null,8,[`show`,`title`,`dismiss-key`,`close-label`])],64))}}),[[`__scopeId`,`data-v-1e53f866`]]),Wb={key:0,id:`result`,class:`result`,\"aria-labelledby\":`saved-label`},Gb={id:`saved-label`,class:`oui-label`},Kb=[`title`],qb={class:`row`},Jb={key:0,id:`notes`,class:`notes`},Yb=[`data-i`],Xb=Ab(P({__name:`ResultCard`,setup(e){let{s:t,state:n,statusView:r,open:i,copyPath:a}=Zv(),o=$t(null),s=W(()=>(n.path??``).split(/[\\\\/]/).pop()??``),c=W(()=>r.value.lines.map((e,t)=>({...e,index:t})).filter(e=>e.kind===`note`));function l(){let e=o.value,t=getSelection();if(!e||!t)return;let n=document.createRange();n.selectNodeContents(e),t.removeAllRanges(),t.addRange(n)}return(e,r)=>M(n).resultVisible?(L(),R(`section`,Wb,[B(`p`,Gb,k(M(t).saved),1),B(`p`,{id:`file`,class:`file oui-truncate`,title:M(n).path??``},k(s.value),9,Kb),B(`div`,qb,[V(M(Yg),{id:`open`,label:M(t).open,\"aria-describedby\":c.value.length?`notes`:void 0,onClick:r[0]||=e=>M(i)()},{icon:N(()=>[V(M(Rm),{icon:M(ah),size:14},null,8,[`icon`])]),_:1},8,[`label`,`aria-describedby`]),M(n).clipboard?(L(),z(M(Yg),{key:0,id:`copy`,label:M(t).copy,severity:`secondary`,text:``,size:`small`,onClick:r[1]||=e=>M(a)(l)},{icon:N(()=>[V(M(Rm),{icon:M(rh),size:14},null,8,[`icon`])]),_:1},8,[`label`])):H(``,!0)]),B(`code`,{id:`path`,ref_key:`pathEl`,ref:o,class:`path oui-mono`},k(M(n).path),513),c.value.length?(L(),R(`div`,Jb,[(L(!0),R(I,null,Wr(c.value,e=>(L(),R(`p`,{key:e.key,class:`oui-meta`,\"data-i\":e.index},k(e.text),9,Yb))),128))])):H(``,!0)])):H(``,!0)}}),[[`__scopeId`,`data-v-cdbcaf96`]]),Zb=[`aria-busy`],Qb=[`data-i`],$b=[`data-i`],ex=Ab(P({__name:`StatusBlock`,setup(e){let{s:t,state:n,statusView:r,ready:i}=Zv(),a=e=>({head:e[0],rest:e.slice(1)}),o=W(()=>r.value.lines.map((e,t)=>({...e,index:t}))),s=W(()=>r.value.error),c=W(()=>i.value?[]:o.value.filter(e=>e.kind!==`warn`&&e.kind!==`note`)),l=W(()=>o.value.filter(e=>e.kind===`warn`)),u=(e,t=``)=>`status:${e}:${n.job??``}:${t}`;return Gn(W(()=>s.value?[]:l.value.map(e=>u(`warn`,e.text))),(e,t)=>t.filter(t=>!e.includes(t)).forEach(e=>{try{sessionStorage.removeItem(`oui-banner:`+e)}catch{}})),(e,r)=>(L(),R(I,null,[B(`div`,{id:`status`,class:ge([`status`,{\"oui-sr\":M(n).busy}]),role:`status`,\"aria-live\":`polite`,\"aria-atomic\":`true`,\"aria-busy\":M(n).busy},[V(M(yh),{show:!s.value&&c.value.length>0,role:`none`,class:ge({tall:c.value.length>1}),tone:M(n).resultVisible?`success`:`neutral`,title:a(c.value).head?.text??``,\"data-i\":a(c.value).head?.index,dismissible:``,\"dismiss-key\":u(`outcome`,a(c.value).head?.text),\"close-label\":M(t).close},Gr({_:2},[a(c.value).rest.length?{name:`default`,fn:N(()=>[(L(!0),R(I,null,Wr(a(c.value).rest,e=>(L(),R(`span`,{key:e.key,class:ge([`line`,e.kind]),\"data-i\":e.index},k(e.text),11,Qb))),128))]),key:`0`}:void 0]),1032,[`show`,`class`,`tone`,`title`,`data-i`,`dismiss-key`,`close-label`]),(L(!0),R(I,null,Wr(l.value,e=>(L(),z(M(yh),{key:e.key,show:!s.value,role:`none`,tone:`warn`,title:e.text,\"data-i\":e.index,dismissible:``,\"dismiss-key\":u(`warn`,e.text),\"close-label\":M(t).close},null,8,[`show`,`title`,`data-i`,`dismiss-key`,`close-label`]))),128))],10,Zb),V(M(yh),{show:s.value,id:`status-error`,class:ge({tall:o.value.length>1}),tone:`error`,title:a(o.value).head?.text??``,\"data-i\":a(o.value).head?.index,dismissible:``,\"dismiss-key\":`${u(`error`,a(o.value).head?.text)}:${M(n).errorSeq}`,\"close-label\":M(t).close},Gr({_:2},[a(o.value).rest.length?{name:`default`,fn:N(()=>[(L(!0),R(I,null,Wr(a(o.value).rest,e=>(L(),R(`span`,{key:e.key,class:ge([`line`,e.kind]),\"data-i\":e.index},k(e.text),11,$b))),128))]),key:`0`}:void 0]),1032,[`show`,`class`,`title`,`data-i`,`dismiss-key`,`close-label`])],64))}}),[[`__scopeId`,`data-v-4d3fb3d8`]]),tx={class:`oui-page panel`},nx=[`data-view`,`data-result`],rx={class:`pane-side`},ix=P({__name:`App`,props:{organizer:{}},setup(e){let t=e;Bn(Xv,t.organizer);let{s:n,lang:r,state:i,start:a,stop:o}=t.organizer,s=W(()=>i.busy?`running`:i.resultVisible&&!i.formOpen?`result`:`form`),c=ff(),l={...c.config.locale},u={...l,...km(`pt-BR`).locale,aria:{...l.aria,...km(`pt-BR`).locale?.aria}};return Wn(()=>{document.documentElement.lang=r.value,document.title=n.value.title,c.config.locale=r.value===`pt-BR`?u:l}),Gn(()=>i.focus,async e=>{e&&(await wn(),document.getElementById(e.id)?.focus())}),Dr(a),Ar(o),(e,t)=>(L(),R(I,null,[V(Vb),B(`main`,tx,[V(Ub),B(`div`,{class:`grid`,\"data-view\":s.value,\"data-result\":M(i).resultVisible||void 0},[V(jb,{class:`pane-form`}),B(`div`,rx,[V(ex),V(Fb),V(Xb),V(Nb)])],8,nx)]),V(M(Yv),{\"close-label\":M(n).close,label:M(n).notifications},null,8,[`close-label`,`label`])],64))}});function ax(e){let t=e.trim(),n=/^rgba?\\(\\s*(\\d+)[ ,]+(\\d+)[ ,]+(\\d+)/i.exec(t);if(n)return[Number(n[1]),Number(n[2]),Number(n[3])];let r=/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(t);if(r){let e=r[1].length===3?[...r[1]].map(e=>e+e).join(``):r[1];return[0,2,4].map(t=>parseInt(e.slice(t,t+2),16))}return null}function ox([e,t,n]){let[r,i,a]=[e,t,n].map(e=>{let t=e/255;return t<=.03928?t/12.92:((t+.055)/1.055)**2.4});return .2126*r+.7152*i+.0722*a}function sx(e){let t=ax(e);if(t)return t;try{let t=document.createElement(`canvas`).getContext(`2d`);return t?(t.fillStyle=`#000`,t.fillStyle=e,ax(t.fillStyle)):null}catch{return null}}function cx(){try{for(let e of[document.documentElement,document.body]){let t=e?getComputedStyle(e).getPropertyValue(`--orca-bg`):``,n=t?sx(t):null;if(n)return ox(n)<.4?`dark`:`light`}return window.matchMedia&&window.matchMedia(`(prefers-color-scheme: dark)`).matches?`dark`:`light`}catch{return`light`}}function lx(e,t,n={}){let r=e=>t.split(`.`).reduce((e,t)=>e==null?e:e[t],e),i=r(qy(e))??r(Jy);return typeof i==`function`?i(n):String(i)}function ux(e,t,n){return n.split(`\n"
    "`).map((n,r)=>({key:`${e}-${r}`,kind:t,text:n}))}function dx(e,t,n){let r=!!t&&t.code!==`unexpected`&&t.code in Jy.err,[i,a]=fx(e,r?t.code:`unexpected`,t),o=r||!n?[]:ux(`detail`,`detail`,qy(e).details+n);return{why:i,lines:[...ux(`next`,`next`,qy(e).next+a),...o]}}function fx(e,t,n){let r=qy(e).err[t],i=typeof r==`function`?r(n??{code:t}):r;return[i[0],i[1]]}function px(e,t,n){let r=qy(e),i=Uy(e),a=n.plate==null?``:t(n),o={ref:a};switch(n.code){case`reading`:return{lines:ux(`info`,`info`,a+r.sep+r.reading),progress:null};case`progress`:{let e=n.phase&&Jy.phase[n.phase]?r.phase[n.phase]:null;return e?{lines:ux(`info`,`info`,a+r.sep+e),progress:{i:n.i??0,n:n.n??null,count:n.count??null,subject:n.subject??``}}:null}case`cancelling`:return{lines:ux(`info`,`info`,r.cancelling(o)),progress:null};case`cancelled`:return{lines:ux(`info`,`info`,r.cancelled(o)),progress:null};case`saving`:return{lines:ux(`info`,`info`,r.saving(o)),progress:null};case`failed`:{let t=dx(e,n.cause,n.detail);return{lines:[...ux(`main`,`main`,r.failed({ref:a,act:r.act[n.action??`arrange`]})),...ux(`cause`,`cause`,t.why),...t.lines],progress:null}}case`save_failed`:{let[e,...t]=r.saveFailed({ref:a,detail:n.detail??``}).split(`\n"
    "`);return{lines:[...ux(`main`,`main`,e),...ux(`info`,`info`,t.join(`\n"
    "`))],progress:null}}case`start_failed`:{let t=dx(e,n.cause,n.detail);return{lines:[...ux(`main`,`main`,r.startFailed({why:t.why})),...t.lines],progress:null}}case`done`:{let e=[[`main`,r.res.placed({ref:a,placed:n.placed??0})]];if(n.added){let t=Object.entries(n.by_object??{}).map(([e,t])=>e+` ×`+i(t)).join(r.list);e.push([`info`,r.res.added({added:n.added,per:t})]),n.limit&&e.push([`info`,r.res.limit({max:n.max??0})])}if(n.tower&&e.push([`note`,r.res.tower]),n.clearance!=null&&e.push([`note`,r.res.clearance({clearance:n.clearance})]),n.keep&&e.push([`note`,r.res.keep]),n.right&&e.push([`note`,r.res.right]),n.unplaced&&n.unplaced.length&&e.push([`warn`,r.res.unplaced({names:n.unplaced.join(r.list)})]),e.push([`note`,r.res.foot]),n.open){let t=r.open_[n.open];e.push([`note`,typeof t==`function`?t({detail:n.detail??``}):t])}return{lines:e.map(([e,t],n)=>({key:`res-${n}`,kind:e,text:t})),progress:null}}}return null}function mx(e,t){let n=qy(e);return n.ref({name:t.stem||n.untitled,plate:t.plate})}var hx=1e3,gx=3e3,_x=1e4,vx=e=>e.charAt(0).toLocaleUpperCase()+e.slice(1);function yx(e,t={}){let n=t.now??Date.now,r=t.navigatorLanguage??(typeof navigator<`u`?navigator.language:null),i=Vt({hostLang:null,ctx:null,context:``,stale:``,staleRef:null,status:null,offline:!1,offlineAt:0,retrying:!1,busy:!1,cancelling:!1,path:null,resultVisible:!1,formOpen:!1,job:null,errorSeq:0,clipboard:!1,spacing:2,rotation:`auto`,copies:null,spacingInvalid:!1,validation:!1,focus:null,requirements:[],rechecking:!1}),a=null,o=null,s=n(),c=0,l=0,u=!1,d=!0,f,p=W(()=>zy(i.hostLang||r)),m=W(()=>qy(p.value)),h=(e,t)=>lx(p.value,e,t),g=e=>mx(p.value,e),_=W(()=>i.ctx==null&&!i.context),v=W(()=>i.ctx&&i.ctx.problem?m.value.ctxProblem({problem:i.ctx.problem}):``),y=W(()=>{let e=i.ctx;return e&&e.plate?m.value.ctx({name:e.name||m.value.untitledProject,plate:e.plate,objects:e.objects}):v.value?v.value:i.context||m.value.reading}),b=W(()=>i.ctx&&i.ctx.plate?m.value.go({plate:i.ctx.plate}):m.value.goNone),x=W(()=>i.ctx&&i.ctx.plate?m.value.fill({plate:i.ctx.plate}):m.value.fillNone),S=W(()=>i.ctx&&i.ctx.plate?m.value.ctxShort({plate:i.ctx.plate,objects:i.ctx.objects}):``),C=W(()=>i.ctx&&i.ctx.plate?i.ctx.name||m.value.untitledProject:``),w=W(()=>i.requirements.filter(e=>e.status===`missing`).map(e=>e.name)),ee=W(()=>w.value.length?m.value.reqBlocked({names:w.value.join(m.value.list)}):``),te=W(()=>i.staleRef?m.value.stale({ref:g(i.staleRef)}):i.stale),ne=W(()=>{let e=(e,t=`info`)=>ux(`info`,t,e),t=i.status;if(t?.kind===`starting`)return{lines:e(m.value.starting),error:!1,progress:null};if(t?.kind===`job`){let n=t.msg?px(p.value,g,t.msg):null;return n?{lines:n.lines,error:t.error,progress:i.busy?n.progress:null}:{lines:e(t.hasPath?t.text.split(`\n"
    "`).filter(e=>!e.startsWith(`Saved as: `)).join(`\n"
    "`):t.text),error:t.error,progress:null}}return i.ctx&&i.ctx.objects===0?{lines:e(m.value.emptyProject),error:!1,progress:null}:{lines:e(m.value.ready),error:!1,progress:null}}),T=W(()=>i.status==null&&!(i.ctx&&i.ctx.objects===0)),E=W(()=>ne.value.progress),D=W(()=>!!E.value&&(E.value.n!=null||E.value.count!=null)),re=W(()=>{let e=E.value;return e&&e.n!=null&&e.n>0?Math.round(e.i/e.n*100):null}),ie=W(()=>E.value&&D.value?m.value.count(E.value):``),ae=e=>!!S.value&&!i.offline&&!_.value&&!v.value&&i.ctx?.plate===e,oe=W(()=>{let e=i.status,t=e?.kind===`job`?e.msg:null,n=t&&t.plate!=null?g(t)+m.value.sep:``,r=ne.value.lines[0]?.text??``,a=t?.plate??i.ctx?.plate??null,o=E.value;return{phase:vx(n&&r.startsWith(n)?r.slice(n.length):r),name:t?.stem||i.ctx?.name||m.value.untitledProject,detail:[a!=null&&!ae(a)?m.value.plateLabel({plate:a}):``,o?o.n==null?ie.value:o.subject:``].filter(Boolean).join(` · `),figure:o&&o.n!=null?m.value.countShort({i:o.i,n:o.n}):o&&o.count!=null?`+`+Uy(p.value)(o.count):``}}),O=()=>{let e=typeof document<`u`?document.activeElement:null;return!e||e===document.body?`body`:e.id},se=e=>{i.focus={id:e,seq:++l}},ce=()=>{i.clipboard=typeof navigator<`u`&&!!navigator.clipboard&&typeof navigator.clipboard.writeText==`function`};function le(e){let t=i.busy,n=O();i.busy=e,e||(i.cancelling=!1);let r=typeof document<`u`&&!!document.activeElement?.closest(`.pane-form`);e&&!t&&(n===`go`||n===`fill`||r)&&se(`cancel`),!e&&t&&(n===`cancel`||n===`body`)&&se(i.resultVisible?`open`:`go`)}function ue(e){u=!0,typeof e.spacing==`number`&&Number.isFinite(e.spacing)&&e.spacing>=0&&(i.spacing=e.spacing),(e.rotation===`auto`||e.rotation===`right`||e.rotation===`keep`)&&(i.rotation=e.rotation),Number.isInteger(e.copies)&&e.copies>=1&&e.copies<=300&&(i.copies=e.copies)}function de(t){if(d){if(s=n(),(i.offline||i.retrying)&&(i.offline=!1,i.retrying=!1),t.lang&&(i.hostLang=t.lang),`context`in t&&(i.context=t.context||``),t.ctx&&(i.ctx=t.ctx),t.requirements&&(i.requirements=t.requirements,i.rechecking=!1),t.prefs&&!u&&!i.busy&&ue(t.prefs),`stale`in t&&(i.stale=t.stale||``,i.staleRef=t.stale_ref||null),t.sync||t.start)t.job!==a&&(i.cancelling=!1),a=t.job??null,i.job=a,t.path||(i.resultVisible=!1,i.path=null);else if(`job`in t&&t.job!==a)return;t.start&&(i.validation=!1),`text`in t&&(i.status={kind:`job`,msg:t.msg||null,text:t.text??``,error:!!t.error,hasPath:!!t.path},t.msg?.code===`start_failed`&&(i.cancelling=!1),t.error&&!(`job`in t)&&i.errorSeq++),t.path&&(o=t.job??null,i.path!==t.path&&(i.formOpen=!1),i.path=t.path,i.resultVisible=!0),`busy`in t&&le(!!t.busy),ce(),t.relay&&e.postMessage({action:t.relay,job:t.job??null})}}function fe(t){if(w.value.length)return;u=!0;let n=i.spacing;if(n==null||!Number.isFinite(n)||n<0){i.validation=!0,i.spacingInvalid=!0,se(`spacing`);return}i.validation=!1,i.status={kind:`starting`},i.path=null,i.resultVisible=!1,i.formOpen=!1,i.cancelling=!1,le(!0);let r=t===`fill`&&i.copies!=null?{copies:i.copies}:{};e.postMessage({action:t,spacing:n,rotation:i.rotation,...r})}function pe(e){u=!0,i.spacing=e,i.spacingInvalid=!1,i.validation=!1}function me(e){u=!0,i.rotation=e}function he(e){u=!0,i.copies=e}function ge(){i.busy&&!i.cancelling&&(i.cancelling=!0,e.postMessage({action:`cancel`,job:a}))}function _e(){e.postMessage({action:`open`,job:o})}function ve(e){let t=i.path??``,n=()=>{try{e()}catch{}Vv({tone:`warn`,title:m.value.copyFail,key:`copy`})};try{navigator.clipboard.writeText(t).then(()=>{Vv({tone:`success`,title:m.value.copied,key:`copy`})},n)}catch{n()}}function ye(){i.formOpen=!0,se(`spacing`)}function be(){i.rechecking=!0,e.postMessage({action:`recheck`})}function xe(){i.retrying=!0,c=n(),e.postMessage({action:`hello`}),e.postMessage({action:`poll`})}let Se=()=>{document.hidden||e.postMessage({action:`poll`})};function Ce(){d&&(t.onTick?.(),!document.hidden&&n()-s>_x&&n()-c>_x&&(i.offline||(i.offlineAt=n()),i.offline=!0,i.retrying=!1),Se(),f=setTimeout(Ce,i.busy?hx:gx))}let we=()=>{document.hidden||(s=n(),Se())};function Te(){t.onTick?.(),document.addEventListener(`visibilitychange`,we),window.addEventListener(`focus`,Se),e.postMessage({action:`hello`}),f=setTimeout(Ce,1e3)}function k(){d=!1,f!==void 0&&clearTimeout(f),document.removeEventListener(`visibilitychange`,we),window.removeEventListener(`focus`,Se)}return e.onMessage(de),{state:i,lang:p,s:m,t:h,loading:_,ctxText:y,ctxProblem:v,goLabel:b,fillLabel:x,staleText:te,missing:w,blockedText:ee,ctxShort:S,projectName:C,statusView:ne,ready:T,progress:E,hasCount:D,percent:re,countText:ie,jobView:oe,run:fe,setSpacing:pe,setRotation:me,setCopies:he,cancel:ge,open:_e,copyPath:ve,again:ye,recheck:be,retry:xe,start:Te,stop:k,onMessage:de}}document.body.classList.add(`orca-ui`);var bx=null;function xx(){let e=cx();e!==bx&&(bx=e,Pm(e))}xx();var Sx=yx(Yy(),{onTick:xx}),Cx=As(ix,{organizer:Sx});jm(Cx,Sx.lang.value===`pt-BR`?`pt-BR`:`en`,`sky`),Cx.mount(`#app`);</script>\n"
    "  <style rel=\"stylesheet\" crossorigin>:root{--oui-font:\"Manrope\", system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;--oui-mono:ui-monospace, \"SF Mono\", \"Cascadia Mono\", Menlo, Consolas, monospace;--oui-text-xs:11px;--oui-text-sm:12px;--oui-text-md:13px;--oui-text-lg:15px;--oui-text-xl:20px;--oui-text-2xl:28px;--oui-text-3xl:40px;--oui-s1:4px;--oui-s2:8px;--oui-s3:12px;--oui-s4:16px;--oui-s5:20px;--oui-s6:24px;--oui-s8:32px;--oui-radius-sm:3px;--oui-radius:4px;--oui-radius-lg:6px;--oui-radius-xl:8px;--oui-control:32px;--oui-control-sm:26px;--oui-shadow-panel:0 1px 2px #0000000d;--oui-shadow-overlay:0 10px 28px #00000029;--oui-fast:.12s;--oui-base:.18s;--oui-ease:cubic-bezier(.2, 0, 0, 1)}.orca-dark{--oui-shadow-panel:0 0 0 transparent;--oui-shadow-overlay:0 10px 28px #00000073}@font-face{font-family:Manrope;src:url(data:font/woff2;base64,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)format(\"woff2\");font-weight:400 700;font-style:normal;font-display:swap}html{font-family:var(--oui-font);background:var(--oui-background);min-height:100%;font-size:13px}body.orca-ui{color:var(--oui-text);background:var(--oui-background);min-height:100vh;flex-direction:column;min-height:100dvh;margin:0;display:flex}body.orca-ui>#app{flex-direction:column;flex:1 0 auto;min-width:0;display:flex}body.orca-ui>#app>:where(:only-child,.oui-page,main){flex:1 0 auto}:where(body.orca-ui>#app>*){width:100%}.orca-ui{font-family:inherit;font-size:var(--oui-text-md);accent-color:var(--oui-primary-mark);caret-color:var(--oui-text);text-underline-offset:.2em;-webkit-font-smoothing:antialiased;text-rendering:optimizelegibility;line-height:1.4}.orca-ui *,.orca-ui :before,.orca-ui :after{box-sizing:border-box}.orca-ui [hidden]{display:none!important}.orca-ui :where(button,input,select,textarea){font:inherit}.orca-ui :where(h1,h2,h3,h4,p,dl,dd,ul,ol,figure){margin:0}.orca-ui :where(h1){font-size:var(--oui-text-lg);letter-spacing:-.005em;font-weight:650;line-height:1.3}.orca-ui :where(h2,h3){font-size:var(--oui-text-md);font-weight:650;line-height:1.4}.orca-ui a{color:var(--oui-primary-text)}.orca-ui ::selection{color:var(--oui-on-primary);background:var(--oui-primary)}.orca-ui :where(table,output,.oui-num,.oui-number,.p-inputnumber-input){font-variant-numeric:tabular-nums}.orca-ui :where(button,a,input,select,textarea,summary,[tabindex]):focus-visible{outline:2px solid var(--oui-primary-mark);outline-offset:2px}.orca-ui :where(input,select,textarea){scroll-margin-block:80px}.oui-label{font-size:var(--oui-text-xs);letter-spacing:.05em;text-transform:uppercase;color:var(--oui-muted);font-weight:600;line-height:1.3}.oui-meta{font-size:var(--oui-text-sm);color:var(--oui-muted);line-height:1.4}.oui-muted{color:var(--oui-muted)}.oui-strong{font-weight:600}.oui-mono{font-family:var(--oui-mono);font-size:.94em}.oui-truncate{text-overflow:ellipsis;white-space:nowrap;min-width:0;overflow:hidden}.oui-error-detail{white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--oui-mono);font-size:var(--oui-text-sm)}.oui-sr{clip-path:inset(50%);white-space:nowrap;width:1px;height:1px;position:absolute;overflow:hidden}.oui-page{gap:var(--oui-s3);padding:var(--oui-s3) var(--oui-s4) var(--oui-s6);align-content:start;display:grid;container:oui-page/inline-size}.oui-stack{gap:var(--oui-s3);align-content:start;min-width:0;display:grid}.oui-row{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;min-width:0;display:flex}.oui-spacer{flex:1 1 0}.oui-divider{background:var(--oui-border);border:0;height:1px;margin:0}.oui-well{background:var(--oui-inset);border-radius:var(--oui-radius)}.oui-cards{--oui-cards-gap:var(--oui-s2);gap:var(--oui-cards-gap);grid-template-columns:repeat(auto-fit, minmax(clamp(25% - var(--oui-cards-gap), (var(--oui-cards-4,900px) - 100%) * 999, clamp(50% - var(--oui-cards-gap), (var(--oui-cards-2,440px) - 100%) * 999, 100%)), 1fr));display:grid}.oui-field{gap:var(--oui-s1);min-width:0;display:grid}.oui-field>label{font-size:var(--oui-text-sm);color:var(--oui-muted);font-weight:600}.oui-table-wrap{max-width:100%;overflow:auto}:where([data-tone]){--oui-tone:var(--oui-muted);--oui-tone-soft:var(--oui-raised);--oui-tone-on:var(--oui-surface)}[data-tone=primary]{--oui-tone:var(--oui-primary-text);--oui-tone-soft:var(--oui-primary-soft);--oui-tone-on:var(--oui-surface)}[data-tone=accent]{--oui-tone:var(--oui-accent);--oui-tone-soft:var(--oui-accent-soft);--oui-tone-on:var(--oui-on-accent)}[data-tone=teal]{--oui-tone:var(--oui-teal);--oui-tone-soft:var(--oui-teal-soft);--oui-tone-on:var(--oui-on-teal)}[data-tone=sky]{--oui-tone:var(--oui-sky);--oui-tone-soft:var(--oui-sky-soft);--oui-tone-on:var(--oui-on-sky)}[data-tone=cobalt]{--oui-tone:var(--oui-cobalt);--oui-tone-soft:var(--oui-cobalt-soft);--oui-tone-on:var(--oui-on-cobalt)}[data-tone=violet]{--oui-tone:var(--oui-violet);--oui-tone-soft:var(--oui-violet-soft);--oui-tone-on:var(--oui-on-violet)}[data-tone=magenta]{--oui-tone:var(--oui-magenta);--oui-tone-soft:var(--oui-magenta-soft);--oui-tone-on:var(--oui-on-magenta)}[data-tone=lime]{--oui-tone:var(--oui-lime);--oui-tone-soft:var(--oui-lime-soft);--oui-tone-on:var(--oui-on-lime)}[data-tone=orange]{--oui-tone:var(--oui-orange);--oui-tone-soft:var(--oui-orange-soft);--oui-tone-on:var(--oui-on-orange)}[data-tone=success]{--oui-tone:var(--oui-success);--oui-tone-soft:var(--oui-success-bg);--oui-tone-on:var(--oui-on-state)}[data-tone=warn]{--oui-tone:var(--oui-warning);--oui-tone-soft:var(--oui-warning-bg);--oui-tone-on:var(--oui-on-state)}[data-tone=danger]{--oui-tone:var(--oui-danger);--oui-tone-soft:var(--oui-danger-bg);--oui-tone-on:var(--oui-on-state)}[data-tone=info]{--oui-tone:var(--oui-info);--oui-tone-soft:var(--oui-info-bg);--oui-tone-on:var(--oui-on-state)}.oui-hero{gap:var(--oui-s3);min-width:0;padding:var(--oui-s4);border-radius:var(--oui-radius-lg);background:var(--oui-raised);display:grid}.oui-hero[data-tone]{background:var(--oui-tone-soft);box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-tone) 40%, transparent)}.oui-dash{gap:var(--oui-s3);grid-template-columns:minmax(0,1fr);align-items:start;min-width:0;display:grid}.oui-dash>:where(.oui-dash-top){grid-column:1/-1;min-width:0}.oui-dash-main,.oui-dash-side{gap:var(--oui-s3);align-content:start;min-width:0;display:grid}@container oui-page (width>=1000px){.oui-dash{grid-template-columns:minmax(0,2fr) minmax(320px,1fr)}}.oui-dash-pair{gap:var(--oui-s3);grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));align-items:start;min-width:0;display:grid}.oui-split{gap:var(--oui-s3);grid-template-columns:minmax(0,1fr);align-items:start;min-width:0;display:grid}@container oui-page (width>=900px){.oui-split[data-rail]{grid-template-columns:260px minmax(0,1fr)}.oui-split[data-detail]{grid-template-columns:minmax(0,1fr) 320px}}@container oui-page (width>=1300px){.oui-split[data-rail][data-detail]{grid-template-columns:260px minmax(0,1fr) 320px}}.oui-skeleton{border-radius:var(--oui-radius-sm);background:var(--oui-border);height:12px;animation:1.6s ease-in-out infinite oui-pulse;display:block}@keyframes oui-pulse{50%{opacity:.45}}.oui-chip{align-items:center;gap:var(--oui-s1);border-radius:var(--oui-radius-sm);height:20px;font-size:var(--oui-text-xs);letter-spacing:.02em;white-space:nowrap;color:var(--oui-muted);background:var(--oui-raised);box-shadow:inset 0 0 0 1px var(--oui-border);padding:0 6px;font-weight:600;display:inline-flex}.oui-chip[data-tone]{color:var(--oui-tone);background:var(--oui-tone-soft);box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-tone) 30%, transparent)}.oui-chip[data-solid]{color:var(--oui-tone-on);background:var(--oui-tone);box-shadow:none}.oui-slot-badge{border-radius:var(--oui-radius);min-width:26px;height:22px;font-size:var(--oui-text-sm);color:var(--oui-text);background:var(--oui-surface);box-shadow:inset 0 0 0 1px var(--oui-control-border);place-items:center;padding:0 6px;font-weight:700;display:inline-grid}.oui-slot-badge[data-tool]{box-shadow:none;color:var(--oui-on-tool)}.oui-slot-badge[data-tool=\"1\"]{background:var(--oui-tool1)}.oui-slot-badge[data-tool=\"2\"]{background:var(--oui-tool2)}.oui-slot-badge[data-tool=\"3\"]{background:var(--oui-tool3)}.oui-slot-badge[data-tool=\"4\"]{background:var(--oui-tool4)}.orca-ui .p-button{white-space:nowrap;font-weight:400;line-height:18px}.orca-ui .p-button-label{font-weight:var(--p-button-label-font-weight,600)}.orca-ui .p-button:disabled,.orca-ui .p-disabled{cursor:not-allowed}.orca-ui :where(.p-inputtext,.p-select-label){line-height:18px}.orca-ui .p-datatable-column-title{letter-spacing:.05em;text-transform:uppercase}.orca-ui .p-datatable-thead>tr>th{z-index:1;white-space:nowrap;position:sticky;top:0}.orca-ui .p-progressbar-value{border-radius:4px}.orca-ui .p-togglebutton.p-togglebutton-checked:is(:disabled,.p-disabled){color:var(--oui-on-primary)}.orca-ui .p-button-secondary:not(.p-button-text,.p-button-outlined){border-color:var(--oui-control-hover)}.orca-ui .p-datatable-tbody>tr:last-child>td{border-bottom-width:0}.orca-ui .p-datatable-tbody>tr>td{vertical-align:middle}.orca-ui :where(th,td).oui-num,.orca-ui :where(th,td).oui-end{text-align:end}.orca-ui th.oui-num .p-datatable-column-header-content,.orca-ui th.oui-end .p-datatable-column-header-content{justify-content:flex-end}.orca-ui .p-dialog{border:1px solid var(--oui-border)}.orca-ui .p-tablist-tab-list{height:100%}.orca-ui .p-tab{white-space:nowrap;align-items:center;display:inline-flex}@media (prefers-reduced-motion:reduce){.orca-ui *,.orca-ui :before,.orca-ui :after{scroll-behavior:auto!important;transition:none!important;animation-duration:.01ms!important;animation-iteration-count:1!important}}@media (forced-colors:active){.orca-ui :where(button,a,input,select,textarea,summary,[tabindex]):focus-visible{outline:2px solid highlight}}.oui-icon{flex:none;display:block}.oui-icon[data-tone]{color:var(--oui-tone)}.oui-tile{border-radius:var(--oui-radius);color:var(--oui-tone);background:var(--oui-raised);box-shadow:inset 0 0 0 1px var(--oui-border);flex:none;place-items:center;display:inline-grid}.oui-appbar{z-index:10;background:var(--oui-surface);box-shadow:0 1px 0 var(--oui-border);position:sticky;top:0;container:oui-appbar/inline-size}.oui-appbar-row{align-items:stretch;gap:0 var(--oui-s4);min-height:44px;padding:0 var(--oui-s4);display:flex}.oui-appbar-id{align-items:center;gap:var(--oui-s2);flex:none;display:flex}.oui-appbar-mark{border-radius:var(--oui-radius-lg);width:26px;height:26px;color:var(--oui-on-solid);background:var(--oui-accent-solid);place-items:center;display:grid}.oui-appbar-mark :is(svg,img){width:17px;height:17px;display:block}.oui-appbar-name{letter-spacing:-.005em;white-space:nowrap;font-size:14px;font-weight:700}.oui-appbar-tabs{scrollbar-width:none;flex:0 auto;align-items:stretch;min-width:0;display:flex;overflow-x:auto}.oui-appbar-tab{cursor:pointer;color:var(--oui-muted);white-space:nowrap;transition:color var(--oui-fast);background:0 0;border:0;align-items:center;gap:6px;padding:0 10px;font-weight:600;display:inline-flex;position:relative}.oui-appbar-tab:hover{color:var(--oui-text)}.orca-ui .oui-appbar-tab:focus-visible{outline-offset:-3px;border-radius:var(--oui-radius)}.oui-appbar-tab[aria-selected=true]{color:var(--oui-text)}.oui-appbar-tab[aria-selected=true]:after{content:\"\";background:var(--oui-accent);border-radius:2px 2px 0 0;height:2px;position:absolute;top:auto;bottom:0;left:8px;right:8px}.oui-appbar-count{min-width:18px;font-size:var(--oui-text-xs);text-align:center;color:var(--oui-muted);background:var(--oui-raised);box-shadow:inset 0 0 0 1px var(--oui-border);border-radius:9px;padding:0 5px;line-height:18px}.oui-appbar-attention{background:var(--oui-warning);border-radius:50%;flex:none;width:7px;height:7px}.oui-appbar-status{align-items:center;gap:var(--oui-s2);min-width:0;font-size:var(--oui-text-sm);color:var(--oui-muted);flex:0 auto;display:flex;overflow:hidden}.oui-appbar-status>.oui-status{flex:none}.oui-appbar-actions{align-items:center;gap:var(--oui-s2);flex:none;display:flex}@container oui-appbar (width<=760px){.oui-appbar-row{flex-wrap:wrap}.oui-appbar-id,.oui-appbar-status,.oui-appbar-actions{min-height:40px}.oui-appbar-tabs{min-height:36px;box-shadow:inset 0 1px 0 var(--oui-border);flex:1 0 100%;order:5;margin-inline-start:-10px;margin-inline-end:-10px}}.oui-banner{border-radius:var(--oui-radius-lg);background:var(--oui-tone-soft);min-height:36px;box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-tone) 40%, transparent);flex-wrap:wrap;align-items:center;gap:4px 10px;padding:4px 8px 4px 12px;display:flex}.oui-banner[data-tone=info]{background:var(--oui-raised);box-shadow:inset 0 0 0 1px var(--oui-border)}.oui-banner-text{flex:150px;min-width:0;padding-block-start:3px;padding-block-end:3px}.oui-banner-text>strong{margin-inline-end:4px;font-weight:700}.oui-banner-actions{flex-wrap:wrap;gap:6px;margin-inline-start:auto;display:flex}.oui-banner[data-dismissible]{padding-inline-end:40px;position:relative}.oui-banner-close{border-radius:var(--oui-radius);width:28px;height:28px;color:var(--oui-muted);cursor:pointer;background:0 0;border:0;place-items:center;display:inline-grid;position:absolute;top:4px;right:6px}.oui-banner-close:hover{color:var(--oui-text);background:color-mix(in srgb, var(--oui-tone) 14%, transparent)}.oui-empty{align-content:center;justify-items:center;gap:var(--oui-s3);padding:var(--oui-s6) var(--oui-s4);text-align:center;min-height:100%;display:grid}.oui-empty-text{gap:2px;max-width:44ch;display:grid}.oui-empty-mark{border-radius:var(--oui-radius);width:34px;height:34px;color:var(--oui-muted);background:var(--oui-raised);place-items:center;display:grid}.oui-empty-image{object-fit:contain;width:144px;height:144px;display:block}.oui-empty[data-compact] .oui-empty-image{width:56px;height:56px}.oui-empty .oui-strong{font-weight:700}.oui-empty-spin{border:2px solid;border-right-color:#0000;border-radius:50%;width:16px;height:16px;animation:.9s linear infinite oui-turn}.oui-empty[data-compact]{text-align:start;padding:var(--oui-s3) var(--oui-s4);place-items:center start;gap:var(--oui-s3);grid-template-columns:auto minmax(0,1fr) auto}.oui-empty[data-compact] .oui-empty-mark{width:30px;height:30px}.oui-camera{aspect-ratio:16/9;background:var(--oui-inset);border-radius:0 0 var(--oui-radius-lg) var(--oui-radius-lg);min-height:120px;display:grid;position:relative;overflow:hidden}.oui-camera-picture{object-fit:cover;width:100%;height:100%;display:block}.oui-camera>.oui-empty{min-height:0}.oui-camera-overlay{inset:auto var(--oui-s2) var(--oui-s2);gap:var(--oui-s1);pointer-events:none;flex-wrap:wrap;display:flex;position:absolute}.oui-camera-overlay>.oui-chip{color:#fff;box-shadow:none;background:#0000009e}.oui-confirm-body{gap:var(--oui-s2);color:var(--oui-text);display:grid}.oui-control-row{align-items:center;gap:var(--oui-s2) var(--oui-s3);grid-template-columns:minmax(84px,1fr) minmax(0,auto) auto;min-height:44px;display:grid}.oui-control-row-label{align-items:center;gap:var(--oui-s2);min-width:0;display:flex}.oui-control-row-value{color:var(--oui-muted);text-align:end}.oui-control-row-control{justify-content:flex-end;align-items:center;gap:var(--oui-s1);display:flex}.oui-controls{gap:var(--oui-s3) var(--oui-s8);grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));align-items:start;display:grid}.oui-controls-group{min-width:0;display:grid}.oui-controls-group>.oui-label{align-items:center;gap:var(--oui-s1);min-height:22px;display:flex}.oui-controls-group>:where(.oui-control-row,.oui-setpoint)+:where(.oui-control-row,.oui-setpoint){border-top:1px solid var(--oui-border)}.oui-kv{gap:6px var(--oui-s4);grid-template-columns:max-content minmax(0,1fr);display:grid}.oui-kv-row{display:contents}.oui-kv dt{color:var(--oui-muted)}.oui-kv dd{overflow-wrap:anywhere;min-width:0}.oui-kv[data-inline]{gap:var(--oui-s3) var(--oui-s6);flex-wrap:wrap;display:flex}.oui-kv[data-inline] .oui-kv-row{gap:2px;display:grid}.oui-kv[data-inline] dt{font-size:var(--oui-text-xs);letter-spacing:.05em;text-transform:uppercase;font-weight:600}.oui-kv[data-inline] dd{font-weight:600}.oui-thumb{border-radius:var(--oui-radius-sm);color:var(--oui-muted);background:var(--oui-inset);box-shadow:inset 0 0 0 1px var(--oui-border);vertical-align:middle;flex:none;justify-content:center;align-items:center;display:inline-flex;overflow:hidden}.oui-thumb>img{object-fit:contain;flex:none;display:block}.oui-files{min-width:0;height:var(--oui-files-height,auto);min-height:360px;container:oui-files/inline-size}.oui-files-grid{height:100%;min-height:inherit;background:var(--oui-surface);border-radius:var(--oui-radius-lg);box-shadow:inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel);grid-template-columns:232px minmax(0,1fr) 300px;display:grid;position:relative;overflow:hidden}.oui-files-grid[data-flat]{grid-template-columns:minmax(0,1fr) 300px}.oui-files-listfoot{padding:var(--oui-s2) var(--oui-s3);border-top:1px solid var(--oui-border)}.oui-files-rail{border-inline-end:1px solid var(--oui-border);background:var(--oui-raised);flex-direction:column;min-width:0;min-height:0;display:flex}.oui-files-railhead{padding:var(--oui-s3) var(--oui-s3) var(--oui-s1)}.oui-files-tree{min-height:0;padding:0 var(--oui-s1) var(--oui-s2);flex:auto;overflow-y:auto}.oui-files-folder{min-height:30px;border-radius:var(--oui-radius);cursor:pointer;color:var(--oui-text);align-items:center;gap:6px;padding-inline-end:var(--oui-s2);display:flex}.oui-files-folder:hover{background:var(--oui-control)}.oui-files-folder[aria-selected=true]{background:var(--oui-control);box-shadow:inset 2px 0 0 var(--oui-primary-mark);font-weight:650}.orca-ui .oui-files-folder:focus-visible{outline-offset:-2px}.oui-files-grid[data-dragging] .oui-files-folder[data-target]{box-shadow:inset 0 0 0 1px var(--oui-control-border)}.oui-files-grid[data-dragging] .oui-files-folder[data-drop]{background:var(--oui-primary-soft);box-shadow:inset 0 0 0 2px var(--oui-primary-mark)}.oui-files-twisty{width:18px;height:22px;color:var(--oui-muted);border-radius:var(--oui-radius-sm);flex:none;place-items:center;display:grid}.oui-files-twisty[data-branch]:hover{color:var(--oui-text);background:var(--oui-control-hover)}.oui-files-foldername{flex:1 1 0}.oui-files-railfoot{padding:var(--oui-s3);border-top:1px solid var(--oui-border)}.oui-files-list{flex-direction:column;min-width:0;min-height:0;display:flex}.oui-files-toolbar{align-items:center;gap:var(--oui-s2) var(--oui-s3);min-height:48px;padding:var(--oui-s2) var(--oui-s3);flex-wrap:wrap;display:flex}.oui-files-path{flex-wrap:wrap;flex:160px;align-items:center;gap:2px;min-width:0;margin:0;padding:0;list-style:none;display:flex}.oui-files-path>li{align-items:center;gap:2px;min-width:0;display:inline-flex}.oui-files-path strong{padding:0 4px;font-weight:700}.oui-files-crumb{border-radius:var(--oui-radius-sm);color:var(--oui-muted);cursor:pointer;background:0 0;border:0;padding:2px 4px}.oui-files-crumb:hover{color:var(--oui-text);background:var(--oui-control)}.oui-files-tools{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;display:flex}.oui-files-bar{align-items:center;gap:var(--oui-s2);min-height:40px;padding:4px var(--oui-s3);background:var(--oui-raised);border-block:1px solid var(--oui-border);flex-wrap:wrap;display:flex}.oui-files-move{min-width:150px}.oui-files-scroll{flex:auto;min-height:0;overflow:auto}.oui-files-table{border-collapse:collapse;table-layout:fixed;width:100%}.oui-files-table th{z-index:1;height:30px;padding:0 var(--oui-s3);text-align:start;background:var(--oui-surface);box-shadow:inset 0 -1px 0 var(--oui-border);position:sticky;top:0}.oui-files-table th:nth-child(3){width:96px}.oui-files-table th:nth-child(4){width:132px}.oui-files-table th.oui-num,.oui-files-table td.oui-num{text-align:end}.oui-files-sort{cursor:pointer;font-size:var(--oui-text-xs);letter-spacing:.05em;text-transform:uppercase;color:var(--oui-muted);background:0 0;border:0;align-items:center;gap:2px;padding:0;font-weight:600;display:inline-flex}.oui-files-sort:hover{color:var(--oui-text)}.oui-files-table td{height:44px;padding:0 var(--oui-s3);border-bottom:1px solid var(--oui-border);white-space:nowrap;text-overflow:ellipsis;overflow:hidden}.oui-files-table :is(th,td).oui-files-check{width:36px;padding:0 0 0 var(--oui-s3);overflow:visible}.oui-files-check>input{cursor:pointer;width:16px;height:16px;margin:0;display:block}.oui-files-row{cursor:default;-webkit-user-select:none;user-select:none}.oui-files-row:hover{background:var(--oui-raised)}.oui-files-row[aria-selected=true]{background:var(--oui-primary-soft)}.oui-files-row[data-dragged]{opacity:.5}.orca-ui .oui-files-row:focus-visible{outline-offset:-2px}.oui-files-name{align-items:center;gap:var(--oui-s2);min-width:0;display:flex}.oui-files-name>.oui-truncate{font-weight:600}.oui-files-skeleton td{height:44px;padding:0 var(--oui-s3);border-bottom:1px solid var(--oui-border)}.oui-files-detail{align-content:start;gap:var(--oui-s3);min-width:0;min-height:0;padding:var(--oui-s3);border-inline-start:1px solid var(--oui-border);display:grid;overflow-y:auto}.oui-files-preview{aspect-ratio:16/10;border-radius:var(--oui-radius);background:var(--oui-inset);box-shadow:inset 0 0 0 1px var(--oui-border);place-items:center;display:grid;overflow:hidden}.oui-files-preview>img{object-fit:contain;width:100%;height:100%;display:block}.oui-files-detailhead{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;min-width:0;display:flex}.orca-ui .oui-files-detailname{min-width:0;font-size:var(--oui-text-lg);overflow-wrap:anywhere;font-weight:700}.oui-files-detailactions{gap:var(--oui-s2);padding-top:var(--oui-s1);flex-wrap:wrap;display:flex}.oui-files-names{gap:2px;min-width:0;margin:0;padding:0;list-style:none;display:grid}.oui-files-ghost{border-radius:var(--oui-radius);color:var(--oui-on-primary);background:var(--oui-primary);padding:4px 10px;font-weight:650;position:absolute;top:-200px;left:0}@container oui-files (width<=980px){.oui-files-grid{grid-template-rows:minmax(0,1fr) auto;grid-template-columns:208px minmax(0,1fr)}.oui-files-grid[data-flat]{grid-template-columns:minmax(0,1fr)}.oui-files-detail{border-inline-start:0;border-top:1px solid var(--oui-border);grid-column:1/-1;grid-template-columns:160px minmax(0,1fr)}.oui-files-detail>.oui-files-preview{grid-row:span 3}.oui-files-detail>.oui-empty{padding-block:var(--oui-s3);grid-column:1/-1}}@container oui-files (width<=620px){.oui-files-grid{grid-template-rows:auto minmax(0,1fr) auto;grid-template-columns:minmax(0,1fr)}.oui-files-grid[data-flat]{grid-template-rows:minmax(0,1fr) auto}.oui-files-rail{border-inline-end:0;border-bottom:1px solid var(--oui-border);max-height:168px}.oui-files-table th:nth-child(4),.oui-files-table td:nth-child(4){display:none}}.oui-formrow{align-items:center;gap:var(--oui-s1) var(--oui-s4);min-height:48px;padding:var(--oui-s2) var(--oui-s4);grid-template-columns:minmax(140px,2fr) minmax(0,3fr);display:grid}.oui-formrow-label{gap:1px;min-width:0;display:grid}.oui-formrow-label>:first-child{font-weight:600}.oui-formrow-control{justify-content:flex-end;align-items:center;gap:var(--oui-s2);flex-wrap:wrap;min-width:0;display:flex}.oui-formrow-control>:where(.p-inputtext,.p-select,.p-inputnumber,.p-password){flex:160px;min-width:0;max-width:100%}.oui-formrow-error{font-size:var(--oui-text-sm);color:var(--oui-danger);text-align:end;flex:1 0 100%}.oui-formrow[data-stacked]{grid-template-columns:minmax(0,1fr)}.oui-formrow[data-stacked] .oui-formrow-control{justify-content:flex-start}@container oui-form (width<=420px){.oui-formrow{grid-template-columns:minmax(0,1fr)}.oui-formrow-control{justify-content:flex-start}.oui-formrow-error{text-align:start}}.oui-hint{flex:none;display:inline-flex;position:relative}.oui-hint-btn{width:22px;height:22px;color:var(--oui-muted);cursor:help;background:0 0;border:0;border-radius:50%;place-items:center;padding:0;display:inline-grid}.oui-hint-btn:hover,.oui-hint-btn[aria-expanded=true]{color:var(--oui-text);background:var(--oui-control)}.oui-hint-bubble{z-index:20;width:max-content;max-width:min(300px,100vw - 16px);padding:var(--oui-s2) 10px;border-radius:var(--oui-radius-lg);font-size:var(--oui-text-sm);text-align:start;white-space:normal;color:var(--oui-text);background:var(--oui-surface);box-shadow:inset 0 0 0 1px var(--oui-control-border), var(--oui-shadow-overlay);font-weight:400;line-height:1.45;position:absolute;top:calc(100% + 4px);left:-8px}.oui-panel{background:var(--oui-surface);border-radius:var(--oui-radius-lg);min-width:0;box-shadow:inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel);flex-direction:column;display:flex}.oui-panel[data-fill]{height:100%;min-height:0}.oui-panel-head{align-items:center;gap:var(--oui-s2);min-height:40px;padding:4px var(--oui-s2) 4px var(--oui-s3);border-bottom:1px solid var(--oui-border);display:flex}.oui-panel-title{flex:none;max-width:100%;font-weight:700}.oui-panel-meta{text-overflow:ellipsis;white-space:nowrap;flex:1 1 0;min-width:0;overflow:hidden}.oui-panel-actions{align-items:center;gap:var(--oui-s1);margin-inline-start:auto;display:flex}.oui-panel-head>:last-child:not(.oui-panel-actions){margin-inline-end:var(--oui-s1)}.oui-panel-body{min-width:0;padding:var(--oui-s3);flex:auto}.oui-panel[data-flush]>.oui-panel-body{padding:0}.oui-panel[data-fill]>.oui-panel-body{min-height:0;overflow:auto}.oui-panel-foot{align-items:center;gap:var(--oui-s2);padding:var(--oui-s2) var(--oui-s3);border-top:1px solid var(--oui-border);flex-wrap:wrap;display:flex}.oui-formsection{container:oui-form/inline-size}.orca-ui .oui-formsection:focus{outline:none}.oui-formsection-desc{padding:var(--oui-s3) var(--oui-s4) 0;max-width:72ch}.oui-formsection-rows>*+*{border-top:1px solid var(--oui-border)}.oui-job{min-width:0;container:oui-job/inline-size}.oui-job-block{align-items:center;gap:var(--oui-s3) var(--oui-s5);grid-template-columns:minmax(0,1fr);min-width:0;display:grid}.oui-job-block[data-thumb]{grid-template-columns:auto minmax(0,1fr)}.oui-job-block[data-thumb]>:not(.oui-job-thumb,.oui-job-id){grid-column:1/-1}.oui-job-thumb{border-radius:var(--oui-radius);background:var(--oui-inset);width:72px;height:72px;box-shadow:inset 0 0 0 1px var(--oui-border);place-items:center;display:grid;overflow:hidden}.oui-job-thumb>:is(img,svg){object-fit:contain;max-width:100%;max-height:100%;display:block}.oui-job-id{gap:2px;min-width:0;display:grid}.oui-job-status{margin-bottom:2px}.oui-job-name{font-size:var(--oui-text-lg);overflow-wrap:anywhere;-webkit-line-clamp:2;line-clamp:2;-webkit-box-orient:vertical;font-weight:700;line-height:1.3;display:-webkit-box;overflow:hidden}.oui-job-detail{font-size:var(--oui-text-sm);color:var(--oui-muted)}.oui-job-progress{align-items:center;gap:var(--oui-s3);min-width:0;display:flex}.oui-job-figure{min-width:2.4ch;font-size:var(--oui-text-2xl);letter-spacing:-.02em;flex:none;font-weight:700;line-height:1}.oui-job-figure>span{letter-spacing:0;color:var(--oui-muted);margin-inline-start:.08em;font-size:.5em;font-weight:600}.oui-job-bar{background:var(--oui-border);border-radius:4px;flex:1 1 0;min-width:60px;height:8px;overflow:hidden}.oui-job-bar>span{border-radius:inherit;background:var(--oui-primary-mark);height:100%;transition:width .4s linear;display:block}.oui-job-bar[data-indeterminate]>span{width:32%;animation:oui-slide 1.4s var(--oui-ease) infinite}.oui-job-stats{align-items:center;gap:var(--oui-s2) var(--oui-s5);flex-wrap:wrap;display:flex}.oui-job-actions{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;display:flex}@container oui-job (width>=640px){.oui-job:not([data-compact])>.oui-job-block{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.oui-job:not([data-compact])>.oui-job-block[data-thumb]{grid-template-columns:auto minmax(0,1fr) minmax(0,1fr)}.oui-job:not([data-compact])>.oui-job-block>:is(.oui-job-id,.oui-job-progress,.oui-job-stats,.oui-job-actions){grid-column:auto}.oui-job:not([data-compact])>.oui-job-block[data-thumb]>.oui-job-thumb{grid-row:span 2}.oui-job:not([data-compact])>.oui-job-block>.oui-job-actions{justify-content:flex-end}}@container oui-job (width>=1100px){.oui-job:not([data-compact])>.oui-job-block{grid-template-columns:minmax(180px,1fr) minmax(220px,1.1fr) auto auto}.oui-job:not([data-compact])>.oui-job-block[data-thumb]{grid-template-columns:auto minmax(180px,1fr) minmax(220px,1.1fr) auto auto}.oui-job:not([data-compact])>.oui-job-block[data-thumb]>.oui-job-thumb{grid-row:auto}}.oui-job[data-compact] .oui-job-block{gap:var(--oui-s2)}.oui-job[data-compact] .oui-job-name{font-size:var(--oui-text-md);-webkit-line-clamp:1;line-clamp:1}.oui-job[data-compact] .oui-job-figure{font-size:var(--oui-text-xl)}.oui-job[data-compact] .oui-job-thumb{width:44px;height:44px}.oui-job[data-stale] .oui-job-bar>span{background:var(--oui-control-border)}.oui-job[data-stale] .oui-job-figure{color:var(--oui-muted)}@keyframes oui-slide{0%{transform:translate(-100%)}to{transform:translate(320%)}}.oui-level{background:var(--oui-border);border-radius:3px;min-width:32px;height:6px;display:block;overflow:hidden}.oui-level>span{background:var(--oui-success);border-radius:3px;min-width:3px;height:100%;display:block}.oui-level[data-tone=low]>span{background:var(--oui-warning)}.oui-level[data-tone=critical]>span{background:var(--oui-danger)}.oui-level[data-unknown]{background:repeating-linear-gradient(135deg, var(--oui-border) 0 3px, transparent 3px 6px)}.oui-level[data-unknown]>span{display:none}.oui-pagehead{align-items:center;gap:var(--oui-s2) var(--oui-s3);flex-wrap:wrap;min-width:0;min-height:32px;display:flex}.oui-pagehead-lead{flex:none;align-items:center;display:flex}.oui-pagehead-id{align-items:baseline;gap:0 var(--oui-s3);flex-wrap:wrap;flex:200px;min-width:0;display:flex}.orca-ui .oui-pagehead-title{font-size:var(--oui-text-lg);letter-spacing:-.005em;font-weight:700}.oui-pagehead-meta{min-width:0}.oui-pagehead-actions{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;margin-inline-start:auto;display:flex}.oui-readout{align-content:start;gap:2px;min-width:0;display:grid}.oui-readout[data-icon]{column-gap:var(--oui-s2);grid-template-columns:auto minmax(0,1fr)}.oui-readout-icon{grid-row:1/span 2;align-self:center}.oui-readout[data-icon]>.oui-readout-sub{grid-column:2}.oui-readout-value[data-tone]{color:var(--oui-tone)}.oui-readout-value{font-size:var(--oui-text-xl);letter-spacing:-.01em;overflow-wrap:anywhere;font-weight:700;line-height:1.15}.oui-readout[data-size=sm] .oui-readout-value{font-size:var(--oui-text-lg);letter-spacing:0;line-height:1.3}.oui-readout[data-size=lg] .oui-readout-value{font-size:var(--oui-text-2xl);letter-spacing:-.015em}.oui-readout[data-size=hero]{gap:6px}.oui-readout[data-size=hero] .oui-readout-value{font-size:var(--oui-text-3xl);letter-spacing:-.02em;line-height:1.05}.oui-readout-unit{letter-spacing:0;color:var(--oui-muted);margin-inline-start:.18em;font-size:.6em;font-weight:600}.oui-readout-value[data-tone] .oui-readout-unit{color:inherit}.oui-readout[data-size=hero] .oui-readout-value{font-weight:700}.oui-readout[data-size=sm] .oui-readout-unit{font-size:.8em}.oui-readout[data-incomplete] .oui-readout-value{color:var(--oui-muted);text-underline-offset:.16em;padding-bottom:.12em;-webkit-text-decoration:underline dashed;text-decoration:underline dashed;text-decoration-thickness:1px}.oui-readout[data-stale] .oui-readout-value{color:var(--oui-muted)}.oui-status{--oui-status:var(--oui-muted);--oui-status-bg:var(--oui-raised);width:fit-content;min-width:0;max-width:100%;height:22px;font-size:var(--oui-text-sm);white-space:nowrap;color:var(--oui-status);background:var(--oui-status-bg);box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-status) 32%, transparent);border-radius:11px;align-items:center;gap:5px;padding:0 9px 0 7px;font-weight:650;line-height:1;display:inline-flex}.oui-status[data-variant=plain]{height:auto;box-shadow:none;background:0 0;border-radius:0;padding:0;line-height:1.3}.oui-status[data-state=ok]{--oui-status:var(--oui-success);--oui-status-bg:var(--oui-success-bg)}.oui-status:is([data-state=busy],[data-state=pending]){--oui-status:var(--oui-info);--oui-status-bg:var(--oui-info-bg)}.oui-status:is([data-state=warn],[data-state=stale]){--oui-status:var(--oui-warning);--oui-status-bg:var(--oui-warning-bg)}.oui-status[data-state=error]{--oui-status:var(--oui-danger);--oui-status-bg:var(--oui-danger-bg)}.oui-status-label{text-overflow:ellipsis;overflow:hidden}.oui-status-dot{background:currentColor;border-radius:50%;flex:none;width:8px;height:8px}.oui-status[data-state=idle] .oui-status-dot{background:0 0;box-shadow:inset 0 0 0 1.5px}.oui-status[data-state=connecting] .oui-status-dot{border:1.5px solid;background:0 0;border-right-color:#0000;width:10px;height:10px;animation:.9s linear infinite oui-turn}.oui-status[data-state=busy] .oui-status-dot{animation:oui-halo 2s var(--oui-ease) infinite}@keyframes oui-turn{to{transform:rotate(1turn)}}@keyframes oui-halo{0%{box-shadow:0 0 0 0 color-mix(in srgb, currentColor 50%, transparent)}70%,to{box-shadow:0 0 0 5px #0000}}.oui-req{align-items:flex-start;gap:var(--oui-s2) var(--oui-s4);padding:var(--oui-s3) var(--oui-s4);flex-wrap:wrap;display:flex;container:oui-req/inline-size}.oui-req-id{flex:240px;gap:2px;min-width:0;display:grid}.oui-req-name{font-weight:650}.oui-req-detail{font-family:var(--oui-mono);font-size:var(--oui-text-xs);overflow-wrap:anywhere}.oui-req-install{align-items:center;gap:2px var(--oui-s2);flex-wrap:wrap;margin-top:2px;display:flex}.oui-req-link{overflow-wrap:anywhere;align-items:center;gap:4px;font-weight:600;display:inline-flex}.oui-req-side{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;flex:none;margin-inline-start:auto;display:flex}.oui-req-check{height:var(--oui-control-sm);border:1px solid var(--oui-control-border);border-radius:var(--oui-radius);color:var(--oui-text);background:var(--oui-surface);cursor:pointer;align-items:center;gap:6px;padding:0 10px;font-weight:600;display:inline-flex}.oui-req-check:hover:not(:disabled){background:var(--oui-raised)}.oui-req-check:disabled{opacity:.55;cursor:not-allowed}.oui-spark{flex:none;display:block;overflow:visible}.oui-spark-line{fill:none;stroke:var(--oui-tone);stroke-width:2px;stroke-linecap:round;stroke-linejoin:round}.oui-spark-area{fill:color-mix(in srgb, var(--oui-tone) 34%, transparent);stroke:none}.oui-spark-target{fill:none;stroke:var(--oui-tone);stroke-width:1px;stroke-dasharray:2 3}.oui-spark-dot{fill:var(--oui-tone);stroke:var(--oui-surface);stroke-width:1.5px}.oui-spark rect{fill:color-mix(in srgb, var(--oui-tone) 55%, var(--oui-tone-soft))}.oui-spark rect.oui-spark-last{fill:var(--oui-tone)}.oui-spark-none{stroke:var(--oui-control-border);stroke-width:1.5px;stroke-dasharray:3 4;fill:none}.oui-setpoint{container:oui-setpoint/inline-size}.oui-setpoint-row{align-items:center;gap:var(--oui-s2) var(--oui-s3);grid-template-columns:minmax(84px,1fr) minmax(96px,auto) minmax(0,1fr) auto;min-height:44px;display:grid}.oui-setpoint-label{align-items:center;gap:var(--oui-s2);min-width:0;color:var(--oui-text);display:flex}.oui-setpoint-spark{justify-self:start}.oui-setpoint-phase{color:var(--oui-tone);font-weight:600}.oui-setpoint-read{display:grid}.oui-setpoint-now[data-tone=neutral]{color:var(--oui-text)}.oui-setpoint-now{color:var(--oui-tone);font-size:var(--oui-text-xl);letter-spacing:-.01em;font-weight:700;line-height:1.15}.oui-setpoint-unit{margin-inline-start:.2em;font-size:.6em;font-weight:600}.oui-setpoint-control{align-items:center;gap:var(--oui-s1);grid-column:-2/-1;display:flex}@container oui-setpoint (width<=520px){.oui-setpoint-spark{display:none}.oui-setpoint-row{grid-template-columns:minmax(84px,1fr) minmax(96px,1fr) auto}}@container oui-setpoint (width<=400px){.oui-setpoint-row{grid-template-columns:minmax(0,1fr) auto}.oui-setpoint-control{grid-column:1/-1;justify-content:flex-end}}.oui-settings{--oui-settings-offset:44px;height:max(240px, calc(100vh - var(--oui-settings-offset)));height:max(240px, calc(100dvh - var(--oui-settings-offset)));container:oui-settings/inline-size}.oui-settings-grid{gap:0 var(--oui-s4);height:100%;padding:var(--oui-s3) var(--oui-s4) 0;grid-template:\"nav body\"minmax(0,1fr)\"foot foot\"/208px minmax(0,1fr);display:grid}.oui-settings-nav{grid-area:nav;min-height:0;overflow-y:auto}.oui-settings-tabs{align-content:start;gap:1px;display:grid}.oui-settings-link{align-items:center;gap:var(--oui-s2);border-radius:var(--oui-radius);text-align:start;cursor:pointer;width:100%;min-height:34px;transition:background var(--oui-fast), color var(--oui-fast);background:0 0;border:0;padding:0 10px;font-weight:600;display:flex}.orca-ui .oui-settings-link{color:var(--oui-muted)}.orca-ui .oui-settings-link:hover{color:var(--oui-text);background:var(--oui-raised)}.orca-ui .oui-settings-link[aria-selected=true]{color:var(--oui-text);background:var(--oui-control);box-shadow:inset 2px 0 0 var(--oui-primary-mark)}.oui-settings-flag{color:var(--oui-warning);margin-inline-start:auto}.oui-settings-link[data-state=error] .oui-settings-flag{color:var(--oui-danger)}.oui-settings-body{overscroll-behavior:contain;gap:var(--oui-s3);max-width:960px;min-height:0;padding-bottom:var(--oui-s4);grid-area:body;grid-template-columns:minmax(0,1fr);align-content:start;display:grid;overflow-y:auto}.orca-ui .oui-settings-body:focus{outline:none}.oui-settings-body>.oui-stack{display:contents}.oui-settings-foot{z-index:5;align-items:center;gap:var(--oui-s2) var(--oui-s3);margin-inline:calc(-1 * var(--oui-s4));padding:var(--oui-s2) var(--oui-s4);background:var(--oui-surface);border-top:1px solid var(--oui-border);flex-wrap:wrap;grid-area:foot;display:flex;position:sticky;bottom:0;box-shadow:0 -6px 16px #0000001f}@container oui-settings (width<=820px){.oui-settings-grid{grid-template:\"nav\"\"body\"minmax(0,1fr)\"foot\"/minmax(0,1fr)}.oui-settings-nav{overflow:visible}.oui-settings-tabs{gap:var(--oui-s1);padding-bottom:var(--oui-s2);scrollbar-width:none;display:flex;overflow-x:auto}.oui-settings-link{flex:none;width:auto}}.oui-spool{isolation:isolate;flex:none;display:block;position:relative}.oui-spool img{object-fit:contain;-webkit-user-select:none;user-select:none;width:100%;max-width:100%;height:100%;max-height:100%;display:block}.oui-spool-wind{background:var(--oui-spool-paint);-webkit-mask:var(--oui-spool-map) center / 100% 100% no-repeat;-webkit-mask:var(--oui-spool-map) center / 100% 100% no-repeat;mask:var(--oui-spool-map) center / 100% 100% no-repeat;position:absolute;top:0;bottom:0;left:0;right:0}.oui-spool-wind img{mix-blend-mode:hard-light}.oui-spool-unknown{z-index:1;color:var(--oui-text);background:var(--oui-surface);box-shadow:0 0 0 1px color-mix(in srgb, var(--oui-text) 35%, transparent), 0 1px 2px #00000040;border-radius:50%;place-items:center;font-weight:700;line-height:1;display:grid;position:absolute;top:-2px;right:-2px}.oui-spool:after{content:\"\";z-index:-1;background:radial-gradient(closest-side,#00000047,#0000);border-radius:50%;height:8%;position:absolute;bottom:1%;left:14%;right:14%}.orca-dark .oui-spool-body{filter:drop-shadow(0 0 .6px #ffffff8c)drop-shadow(0 0 6px #ffffff1a)}.oui-slot{gap:var(--oui-s2);min-width:0;padding:var(--oui-s2) var(--oui-s1) var(--oui-s2);border-radius:var(--oui-radius-lg);background:var(--oui-raised);box-shadow:inset 0 0 0 1px var(--oui-border);flex-direction:column;display:flex}.oui-slot[data-active]{box-shadow:inset 0 0 0 2px var(--oui-primary-mark)}.oui-slot[data-empty]{box-shadow:none;outline:1.5px dashed var(--oui-control-border);outline-offset:-1.5px;background:0 0}.oui-slot>:not(.oui-slot-strip){padding-inline-start:6px;padding-inline-end:6px}.oui-slot-strip{border:1px solid color-mix(in srgb, var(--oui-text) 34%, transparent);background-clip:padding-box;border-radius:3px;height:6px;margin:-2px 2px 0;display:block}.oui-slot-strip[data-unknown]{background:repeating-linear-gradient(135deg, transparent 0 4px, color-mix(in srgb, var(--oui-control-border) 55%, transparent) 4px 5px)}.oui-slot-head{justify-content:space-between;align-items:center;gap:var(--oui-s2);min-height:22px;display:flex}.oui-slot-main{align-items:center;gap:var(--oui-s3);min-width:0;display:flex}.oui-slot-id{gap:1px;min-width:0;display:grid}.oui-slot-name{-webkit-line-clamp:2;line-clamp:2;overflow-wrap:anywhere;-webkit-box-orient:vertical;font-weight:700;display:-webkit-box;overflow:hidden}.oui-slot[data-empty] .oui-slot-name{color:var(--oui-muted);font-weight:500}.oui-slot-amount{gap:5px;line-height:20px;display:grid}.oui-slot-amount strong{font-size:var(--oui-text-lg);font-weight:700}.oui-slot-extra{gap:var(--oui-s2);display:grid}.oui-slot-actions{gap:var(--oui-s1);padding-top:var(--oui-s1);flex-wrap:wrap;margin-top:auto;display:flex}.oui-swatch{background:var(--oui-swatch);box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-text) 38%, transparent);border-radius:50%;flex:none;display:inline-block}.oui-swatch[data-unknown]{background:repeating-linear-gradient(135deg, transparent 0 2px, color-mix(in srgb, var(--oui-muted) 60%, transparent) 2px 3px)}.oui-toasts{left:16px;bottom:var(--oui-toast-bottom,16px);z-index:1100;gap:var(--oui-s2);pointer-events:none;width:min(380px,100vw - 32px);display:grid;position:fixed}.oui-toast{pointer-events:auto;align-items:flex-start;gap:var(--oui-s2);border-radius:var(--oui-radius-lg);min-height:44px;color:var(--oui-text);background:var(--oui-surface);box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--oui-tone) 50%, transparent), inset 3px 0 0 var(--oui-tone), var(--oui-shadow-overlay);padding:8px 6px 8px 10px;display:flex}.oui-toast>.oui-icon{margin-top:1px}.oui-toast-text{overflow-wrap:anywhere;flex:auto;min-width:0;padding-top:2px}.oui-toast-text>strong{font-weight:700}.oui-toast-action,.oui-toast-close{border-radius:var(--oui-radius);cursor:pointer;background:0 0;border:0;flex:none;place-items:center;min-height:28px;display:inline-grid}.oui-toast-action{color:var(--oui-primary-text);padding:0 8px;font-weight:650}.oui-toast-close{width:28px;color:var(--oui-muted)}.oui-toast-action:hover,.oui-toast-close:hover{color:var(--oui-text);background:var(--oui-raised)}.oui-toast-enter-active,.oui-toast-leave-active{transition:opacity var(--oui-base) var(--oui-ease), transform var(--oui-base) var(--oui-ease)}.oui-toast-enter-from,.oui-toast-leave-to{opacity:0;transform:translate(-12px)}.oui-toast-leave-active{width:100%;position:absolute}.oui-toast-move{transition:transform var(--oui-base) var(--oui-ease)}@media (prefers-reduced-motion:reduce){.oui-toast-enter-active,.oui-toast-leave-active,.oui-toast-move{transition:none}.oui-toast-enter-from,.oui-toast-leave-to{transform:none}}.oui-toolbar{align-items:center;gap:var(--oui-s2);min-width:0;padding:var(--oui-s2) var(--oui-s3);flex-wrap:wrap;display:flex}.oui-toolbar>:where(.p-inputtext,.p-iconfield){flex:180px;min-width:0;max-width:320px}.oui-toolbar-summary{white-space:nowrap;margin-inline-start:var(--oui-s1)}.oui-toolbar-end{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;margin-inline-start:auto;display:flex}.form[data-v-3616ab9b]{gap:var(--oui-s3);grid-template-columns:minmax(0,1fr);align-content:start;min-width:0;display:grid}.form p[data-v-3616ab9b],.form label[data-v-3616ab9b],.form legend[data-v-3616ab9b],.form dt[data-v-3616ab9b],.form dd[data-v-3616ab9b]{overflow-wrap:anywhere}.actions[data-v-3616ab9b]{gap:var(--oui-s2);grid-template-columns:minmax(0,1fr);display:grid}.actions>.p-button[data-v-3616ab9b]{white-space:normal;min-height:40px;font-size:14px}.actions[data-v-3616ab9b] .p-button-label{overflow-wrap:break-word;flex:0 auto}.more>summary[data-v-3616ab9b]{cursor:pointer;border-radius:var(--oui-radius);font-weight:600;list-style:none;display:list-item}.more>summary>.sum[data-v-3616ab9b]{align-items:center;gap:6px;min-width:0;min-height:28px;display:flex}.more>summary[data-v-3616ab9b]::-webkit-details-marker{display:none}.sum>.oui-icon[data-v-3616ab9b]{color:var(--oui-muted);transition:transform var(--oui-fast)}.more[open]>:is(#adv,#req) .sum>.oui-icon[data-v-3616ab9b]{transform:rotate(90deg)}.sum>.oui-meta[data-v-3616ab9b]{text-overflow:ellipsis;white-space:nowrap;font-weight:400;overflow:hidden}.more+.more[data-v-3616ab9b]{margin-top:calc(-1 * var(--oui-s2))}.more-body[data-v-3616ab9b]{gap:var(--oui-s3);padding-top:var(--oui-s2);display:grid}.rotation[data-v-3616ab9b]{border:0;min-width:0;margin:0;padding:0}.rotation legend[data-v-3616ab9b]{float:left;font-size:var(--oui-text-sm);color:var(--oui-muted);margin-inline-end:var(--oui-s1);padding:0;font-weight:600;line-height:22px}.rotation>.choice[data-v-3616ab9b]{clear:both;padding-top:var(--oui-s2)}.labelled[data-v-3616ab9b]{align-items:center;gap:var(--oui-s1);min-width:0;display:flex}.choice[data-v-3616ab9b]{align-items:flex-start;gap:var(--oui-s2);display:flex}.terms[data-v-3616ab9b]{gap:2px;display:grid}.reqs[data-v-3616ab9b]{margin:0 calc(-1 * var(--oui-s4));display:grid}.terms dt[data-v-3616ab9b]{font-weight:600}.terms dd+dt[data-v-3616ab9b]{margin-top:var(--oui-s2)}.job[data-v-3908b3ac] .oui-job-detail{text-overflow:ellipsis;white-space:nowrap;overflow:hidden}.job[data-v-3908b3ac] .p-button{white-space:normal}.job[data-v-3908b3ac]{padding:var(--oui-s3);border-radius:var(--oui-radius-lg);background:var(--oui-surface);box-shadow:inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel);overflow:hidden}.context{display:block}.context>.oui-status{vertical-align:middle}@container oui-appbar (width<=520px){.context-name{display:none}}#again .p-button-label{display:none}#again{padding-inline-start:0;padding-inline-end:0}@container oui-appbar (width>=621px){.oui-appbar-actions:has(>#again){display:none}}@container oui-appbar (width<=620px){.oui-appbar-row:has(#again){gap:0 var(--oui-s2)}.oui-appbar-row:has(#again)>.oui-spacer{display:none}.oui-appbar-row:has(#again)>.oui-appbar-status{flex:1 1 0;justify-content:flex-end}}#offline[data-v-1e53f866]{align-items:flex-start}#offline[data-v-1e53f866] .oui-banner-actions{min-width:0;max-width:100%}#retry[data-v-1e53f866]{white-space:normal}#retry[data-v-1e53f866] .p-button-label{overflow-wrap:break-word;flex:0 auto;min-width:0}.result[data-v-cdbcaf96]{gap:var(--oui-s2);min-width:0;padding:var(--oui-s3);border-radius:var(--oui-radius-lg);background:var(--oui-surface);box-shadow:inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel);display:grid}.file[data-v-cdbcaf96]{font-weight:600}.row[data-v-cdbcaf96]{gap:var(--oui-s1);grid-template-columns:minmax(0,1fr);min-width:0;display:grid}.row>.p-button[data-v-cdbcaf96]{white-space:normal;flex-wrap:wrap}.row[data-v-cdbcaf96] .p-button-label{overflow-wrap:break-word;flex:0 auto;min-width:0}@container oui-page (width<=620px){.result[data-v-cdbcaf96]{padding:var(--oui-s2);gap:6px}}@container oui-page (width<=200px){.result[data-v-cdbcaf96]{padding:var(--oui-s2)}.row>.p-button[data-v-cdbcaf96]{padding-inline:var(--oui-s2)}}.path[data-v-cdbcaf96]{overflow-wrap:anywhere;-webkit-user-select:text;user-select:text;color:var(--oui-muted);display:block}.notes[data-v-cdbcaf96]{gap:var(--oui-s1);padding-top:var(--oui-s2);border-top:1px solid var(--oui-border);display:grid}.result p[data-v-cdbcaf96]{overflow-wrap:anywhere}.result .file[data-v-cdbcaf96]{overflow-wrap:normal}.status[data-v-4d3fb3d8]{gap:var(--oui-s2);min-width:0;display:grid}.status[data-v-4d3fb3d8]:empty{display:none}.line[data-v-4d3fb3d8]{overflow-wrap:anywhere;display:block}.line.detail[data-v-4d3fb3d8]{font-family:var(--oui-mono);font-size:var(--oui-text-sm)}[data-v-4d3fb3d8] .oui-banner-text{overflow-wrap:anywhere}[data-v-4d3fb3d8] .oui-banner.tall,.oui-banner.tall[data-v-4d3fb3d8]{align-items:flex-start}@container oui-page (width<=620px){.status[data-v-4d3fb3d8]{gap:var(--oui-s1)}[data-v-4d3fb3d8] .oui-banner[data-dismissible]{padding-inline-end:8px}[data-v-4d3fb3d8] .oui-banner[data-dismissible]>.oui-banner-text:before{content:\"\";float:right;width:28px;height:18px}}@font-face{font-family:OuiCJKPunct;src:local(PingFang SC),local(PingFangSC-Regular),local(Hiragino Sans GB),local(HiraginoSansGB-W3),local(Microsoft YaHei),local(Noto Sans CJK SC),local(NotoSansCJKsc-Regular),local(Noto Sans SC);unicode-range:U+2018-201F,U+2026,U+3000-303F,U+FF00-FFEF}:root:lang(zh){--oui-font:\"OuiCJKPunct\", \"Manrope\", system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif}.panel{padding:var(--oui-s3)}.panel>*{min-width:0}.grid{gap:var(--oui-s3);grid-template-columns:minmax(0,1fr);align-items:start;min-width:0;display:grid}.pane-side{gap:var(--oui-s2);grid-template-columns:minmax(0,1fr);align-content:start;min-width:0;display:grid}.pane-side:not(:has(>:not(.oui-sr,:empty))){display:none}@container oui-page (width<=620px){.grid:not([data-result])>.pane-side,.grid[data-view=result]>.pane-side{order:-1}.grid[data-view=running]>.pane-form,.grid[data-view=result]>.pane-form,.grid .idle{display:none}.pane-side:not(:has(>:not(.oui-sr,:empty,.idle))){display:none}.pane-side{gap:var(--oui-s1)}}@container oui-page (width>=621px){.grid{gap:var(--oui-s4);grid-template-columns:minmax(280px,360px) minmax(0,560px)}.pane-form{padding:var(--oui-s4);border-radius:var(--oui-radius-lg);background:var(--oui-surface);box-shadow:inset 0 0 0 1px var(--oui-border), var(--oui-shadow-panel)}.grid .result .row{align-items:center;gap:var(--oui-s2);flex-wrap:wrap;display:flex}.grid .result .row>.p-button{white-space:nowrap}}\n"
    "/*$vite$:1*/</style>\n"
    "</head>\n"
    "<body class=\"orca-ui\"><div id=\"app\"></div></body>\n"
    "</html>\n"
)
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
# wording (and pt-BR, es, de, zh-CN, pl) keyed by `code`; tests keep the two in step.
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

_checking = threading.Event()  # set while a worker loads or retries numpy and Pillow


def _check(refresh: bool):
    global _DEPS
    try:
        with _DEPS_LOCK:
            if _DEPS is None or (refresh and any(isinstance(v, Exception) for v in _DEPS.values())):
                _DEPS = _load_deps()
    finally:
        _checking.clear()


def _warm_deps(refresh: bool = False):
    """Load numpy and Pillow on a worker thread, so the UI thread never waits for the imports: once on the page's
    first sync, and again for Check again (`refresh`) while something is missing. The page reads the result on its
    next poll."""
    if _checking.is_set():
        return
    if _DEPS is not None and not (refresh and any(isinstance(v, Exception) for v in _DEPS.values())):
        return
    _checking.set()
    threading.Thread(target=_check, args=(refresh,), daemon=True).start()


def requirements(wait: bool = True) -> list[dict]:
    """What Arrange and Fill need, for the page: [{id, name, status: "ok" | "missing" | "unknown", version, detail}].
    `wait=False` never imports: "unknown" while a worker is still loading or retrying them."""
    found_all = _deps() if wait else None if _checking.is_set() else _DEPS
    rows = []
    for rid, name in REQUIREMENTS:
        found = (found_all or {}).get(rid)
        ok = isinstance(found, str)
        status = "unknown" if found_all is None else "ok" if ok else "missing"
        rows.append({"id": rid, "name": name, "status": status, "version": found if ok else None,
                     "detail": f"{type(found).__name__}: {found}" if status == "missing" else None})
    return rows


def missing_requirements() -> list[str]:
    """Names known to be missing; never imports (a still unknown result lets a run go ahead: Package loads them)."""
    return [r["name"] for r in requirements(wait=False) if r["status"] == "missing"]


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
            _warm_deps(refresh=True)
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
        _warm_deps()  # first sync: after the page loaded, off the UI thread; later polls carry the result
        self.lang = _host_language() or self.lang
        info = _context_info()
        stale = self._stale(info)
        msg = {"sync": True, "context": info["text"], "lang": self.lang,
               "ctx": {k: info.get(k) for k in ("plate", "objects", "name", "problem")},
               "stale": stale, "stale_ref": self.job._ref() if stale else None, "job": None, "busy": False,
               "prefs": self.prefs(), "requirements": requirements(wait=False)}
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
            raise _missing_error(", ".join(missing))
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
                "zh": "面板已打开", "pl": "Otwarto panel"}


def panel_opened(lang) -> str:
    """`lang` as the host gives it ("pt_BR", "es", "de", "zh_CN", "zh_TW", "pl_PL", ...); unknown or missing: English."""
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

# The tab icon: the brand tile (assets/orca-cloud/orca_organizer-tile.png) at 40x40 RGBA, its white surround made
# transparent for the dark tab bar. OrcaSlicer reads a page icon from a file and draws it at 20 px (40 px on a 2x
# screen), so get_icon() writes it to the storage folder.
ICON_FILE = "organizer-icon.png"  # no organizer-icon.svg next to it: the host would load an .svg of the same name first
ICON_PNG = base64.b64decode(
    "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"
    "TkSuQmCC"
)


def _icon_path() -> str:
    """The tab icon file in the storage folder, written when missing or stale; "" when it cannot be written."""
    folder = _remember_storage()
    if not folder:
        return ""
    path = Path(folder) / ICON_FILE
    try:
        if not path.is_file() or path.read_bytes() != ICON_PNG:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ICON_PNG)
    except OSError:
        return ""
    return str(path)

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

            def get_icon(self):
                return _icon_path()

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
