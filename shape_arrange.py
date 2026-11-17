# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.14.1"
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
import base64 as _base64
import zlib as _zlib

PANEL_HTML_SHA256 = "b06a4818995ff7ba05e76e97828683ec738da274236e139661646faa45cac357"
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
    "gqjhzmOhWH7Thq0zHuqbt+6IEBXdfUyUEGzawrXGhWji5iNDVHKHsVEtuWJ0rHOJIUe94hbDpdzOy67N6JWXIBbpv+M4yWq5"
    "80gpY9i8geuMlqZmrjtessruPmIqKDZv6VqjJmvq5uMmq+gOI6eu7IqxIwgZNrC/NgJEDYIp1C+RrYw1WqPocgZL8DsOmUpt"
    "q4fOn7K56w6dSnVrDKE/RYNXbD2sJPQuGwk1he9M7OfK5yabDf+T+LCx4G60E/Fn5ETV3b+a1I3d95WidyTzs8V2bRf//4z2"
    "by6u6/v//2QcKLvaV9K4ofe8WO4u1H2ucK7pYf9Tt3ljgVzX/f6naXXZ5b2Sug292MVyd6Huc+VwTU/3n7rNG8vhum7wP02r"
    "y67nldRt6E0ulrsLdZ8rh2t6nP/Ubd5YDtd1R/9pWl11Aa+kb2Ofbrnk3Wj8XHlc2+/7P6DtG8vl+k7hP1Xr61ywK6m8g0e1"
    "WvaulH6ulG7gdf0fw4ONpXUTl+yfjgsz3/U2MzGxxJ2cXljwTgR+tocTq/5M/+afoN2buzqx7s91dP6eLYdbde/ud68FqVIE"
    "lazpdAXQO4pboZb1he3rtGFd0SlUs4Hg/D6tWD/q/gsLy7pOzgYiP0eC7uLu/gObtpFg3cmD/RUatzQO/UuL1lpOyVoCP0us"
    "NnZH/0GN2kygNvcx/97NWhKn/YUlaQ2fYpW0z5GhDb3GX70tG4nOpt7g37U1S4Kbv7DQrOEArJL2OUKzoYv3q7dlI6HZ1HX7"
    "u7ZmSUTwFxaaNbx1VdI+R2g29Md+9bZsJDSb+ll/19YsDaP9wmKzliOtjrzPEZ2NXad/SIs2EqDNHaK/c5tWRJt+YTFa07tV"
    "T+LniNId/Jt/WLs2Eqi7+Cx/95ZVfJG/oy9oTc/Ul/FDKlX+Tm6hL9OczXxEm7oWv3SD/OK1Bb+PtEAla9CxiWB8YeLWiTpv"
    "JrE5kw/0Q3vIBmEkLk6AHBkROHY3qGSpRH1hRtQHpdP8woQX612Y8OKvdGHCi2K2+o/VNPIPvmIa+QdKGvlfa9PIv1iRRv7X"
    "PI38i6+eRv7FH59G/ln1ehc+BuSlLt+MS1eKVNNFs/wKJH7neziLBf/PaJYxaNYehNMZZKhOKPk41mv/W2xX025t2zy71ovA"
    "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"
    "dUNsgdhHG64c8zJtjzxGtBif3XY9epaPHlAEDz5gYYgPBkmQXvVs5KIwdxMNq8ImUbtoSNjOWseoc3HUufB7nwVRwLr8uUjQ"
    "sapAVUsF9DwaQcgAiA7qQDrEk3geNOV7dTo4o0qWielRDaQpVovoOdFP0jpuIUrn3u3318+dXb/2+tqNz+9+8NXara823n9x"
    "89urg1u/XX/jpfU3vxt889Xd395cv3Vp44N3/77yQjYyh765+85fNr/+MBONk+l07ftrm5+8RF2v3XhtcPqrtVtv09u/rzy/"
    "8esrg5uvbH73zcbtdwevv7J24/zg9WvrZ387OH99cO7m4KOP1r559e47f1n7/sONt97beP/Fu+/85e777/195YXBn15Y/93F"
    "zWsfb3zw7t2VC2u3vqKv7t0+h8u58f6L6+cuD16+uX7x5t3nv9+8+tnGZzfvvvOXwevXNn/97drt9wev/H5w+ROawb3b5zZu"
    "/Xb9dxfXblxZu/3+5rW/rL/7Gs5aRdsQUjY+eHftxisb7784uPbN5heX7t0+t3bj/PrXtwav/H7t1q21b98avHR6cPUb/DKJ"
    "thmcPkOf37t9bvDS+c1vr67/9Xn6ijocXJAdQs9nr61f/C8A6IO/DM5cX3/zu7Ub59duvLp24+b6lT/cu31m/eLK4PL7EFfA"
    "1m58fu/2WRwtCbBZu3Vr8Mql9St/HFz8dPP7C5uXzv195YXNa19vXDk7+O704Oq59TMX1s+8PXj97fWvzgwuvDZ46fz6B39Z"
    "f/Obze/fW7v5webV7za+vYp9YjDN3f98d+PmJ1oozfo7r25++19/X3meIFWBNIOvP1779uLgtY82LryUhNEQVtMxNIPr769f"
    "/Jwwkg+YubNycfPlzwevfEoN7qx8uH7u7MbFjwZXf6eTZ2Yl/r7ywp2Vi0Swd1Y+vPv+64MLrxGKRyOUJkpRMndfPr/+9nUF"
    "WCY8Zv2d3w9+e35w881caMz6O7+/e+mbjQ+u5uJiNq/dGrz+9tqNK0WRMYRHLSyGpg0rfOXju299DwEx926f1UNi1m6dJrMt"
    "tcjEw9Ds19/5iihEEpiiqPW3rqejYja/+w7QceVy/B6I4fmL61f+QD3du/3BxgfvDr7+M/W6dvv99dcvDM6epzZ/X3kBduRn"
    "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"
    "c7Oazyh0NalpIbc7Zj+jyS3Y4Pfugr3PKLj1UEureOCpwadcWAKeD7aQ5srldInnRKauBBmy4B+tyvVm4TySYHbLWdrlh6Bh"
    "oNwtJFfapJiP6OG2lW8v0C3UAQi9lZBWycDTH68NBBdiG4GMNgIFbRBIpv0ZgRTIyeRDvbigl3NY//gBzN3M7NMsLjjXQJpa"
    "7sLPCXkqBBIgNSH3z7fEIvT9KnsS/XM7w06f9XI2d0UovM/w9JT85JekPpuZEOdmr1V4ze1hHnK+FZObEfNIjdnJkWS50mYJ"
    "ygE74r7abBOlATN+/Mi/vLvCNttR+S2V2b2SLTm/YfKt2AozHVj227oO8VZpRoO+3aI6Pg76oW+qXXhNW737+lZs3b6Dq3lB"
    "yMa/rQrwvH9dWX7TLwjb+LfV5bbgH1R5hUyadoMOmBzy11Wnz385BUkV0bWKhxAt93rNn4OW3MUTBS0Ui7eYi9Gjivw2UfkN"
    "B/k5lGdMDiHzquFNY8aAnl1jNOpXc6lrdUlCSrCohdt3UDR7K0KW/Lh0U2VsQiYU23/ciDkfbCq/rWfjTCQpz1dlbn6f7r74"
    "eEIV7rqSZCimWCpBb4+k9mOFwIXmhuFNT32Wb9tbhZKk4gMLZOUskJUSeFlDfqq+uLWWCaOt8mP7rWuX/bLGe8A06dlPdAX3"
    "HPxtwj/L90Hr9wtgW/OJB9dNXGPF/sceuCOl2OYXnUmP4WwED+vzeuFsG3n+CZr3FyP/EqKtn4olpgW/zP9WhJb3aylZzRne"
    "LPyqb8Ui1LuGN86ayastQt3s0+fPvVy/XMieylZKMv74kSckfGeaWCxC7rt1meV6QKrWjeq0/kCead7Xym4SMVUu67Bctv6F"
    "IkgmbitCel7jtS/dDHWa9z3IO5huL14GR8vnz4trBFQ9N+l8evr8/Zoa5k8JhJ5zyUJbhmz5/dmQwhYCErqH9f4sg1WdHrFW"
    "O5HxJNWko3xWC36xdN1P4G6jPeS2r1BPJvsb8nKnIeTbftaT5+f7fmXMvf0OqK/phc78uWA9Pbn3y9sqPrsZLWblVy203GLq"
    "quY/Z4Rg1sRfH2gZbfuNz+fuMy8Dy/08/+2e5hzN8wmmg+xn+3oVsp4NumXf3U/8y6DAwOq+CKX+cC376cnOF87jxGdni96y"
    "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"
    "P6aDq4er1aSercV/qN/loZwPyC81Og0B5kMc64CJz6QG/7YBAoEDaT7tyo+csRnMqNmCxYM3Y5m9GO/R/07zXn48IHmrcWRp"
    "UdkZPBd7tsuNJ9hwwHuFdu8tqsCckyrluPPMnFl6WZuzy4Kh2PP65GImt7KUOv62mi6o29NV8vBXep1HVGTKubpnwXujrCYD"
    "UwqcaNNRpPXPZzEUzJB8SjlqFFbGl+VEYXp+Qapjk/1V1/MR2WyNjPNcbowk9Y0w3mAd4M+OjsLI6Atm5+LYPDOR2D4Tyz3z"
    "4Zu6/f6lMsdzb/kZRm8MG/u+sq2zYBdf3kbrisTNk+K+EyRS8nwU0RAx/ar3N8Ucl5iDTsjBoKTxhzbyfCEXCmQv31hRAm8/"
    "YHe91FbNh0K+02NLxvV2pweo+REwNXUmZiHdawy0rGwC4uVsH4I1xmvi7OWjQsVaACkGz+Rn6EhP+aleP2hCUemysgpafM6P"
    "Dwt+Iin8A/REghobevbn6wD/DQxgim4gDDf2sEJtva22TuORtIKtVpOkABrm/czsiCPoovQRRluGEGRyPxR5PbN7OVJdXpDQ"
    "eij9qNZT6F+JzfYo5W6zXjmaItwHowhmmLX7mJErmRyn7ggdSu8iw3conEwor704jM3l7Eo15B9Zpe3nlwt0lO7sebEYTzpw"
    "p5CgukqKziiul1ceb8BIylQuyFLYR59aJOZmYO5aDmKKwpVeaoWnXZqjkdLQepCxOH25kAhGfOEw4x1aS22+hIaxJ7Sbs0JB"
    "KbZL/pIlZ6haT08SQ3cA3KCLu3NYb++TbuxRIpFdoYhcrrmCwV9kXhVqLmY3OOr7d7oNnQd15rIE3b34T4UNJZ6r7sqK4/SG"
    "Mxg8OZWK1ceID1eCxevwz1sN0+vVRUqR+T6zqLec0wx4vbXjS03nUjWaPu5RA0dQJm2mARHVXvmi/KoJuAC/n0BPj+x7DJyy"
    "uygvhXQQtpZ4IXWT6O7g/UBFDqEepY3gKeCxT/1gcoTq/3yqNMW/ptxo/neVMLdqXG7XrXhSXoJ8OjslY15zxoNkzg2yHl+3"
    "ad/O02eOKyvIoKaZ4AXj5ZJQphzyirV6wpM4FpLbcyk3+c/s3eWcmFZi7TwULJWjNJiPJNJvJESNm/UG+U+DrtfpQlc0Tdk+"
    "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"
    "IP1/JQ=="
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

# The tab icon: the brand tile (assets/orca-cloud/orca_organizer-tile.png) at 40x40 RGBA, its white surround made
# transparent for the dark tab bar. OrcaSlicer reads a page icon from a file and draws it at 20 px (40 px on a 2x
# screen), so get_icon() writes it to the storage folder.
ICON_FILE = "organizer-icon.png"  # no organizer-icon.svg next to it: the host would load an .svg of the same name first
ICON_PNG = _base64.b64decode(
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
