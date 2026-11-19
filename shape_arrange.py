# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# type = "script"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Irregular and concave parts interlock to fit more on the plate, or fill the whole plate with copies. Just set the spacing."
# author = "Heitor Ramon"
# version = "0.14.2"
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

PANEL_HTML_SHA256 = "de242accd2460d3f8e6881e71802ab11ec8660f9f5247380defe9f8e6db2f4d6"
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
    "XIdQrcgD0QpVgmyqnkEl1yRQp0oWUpRUG7JGB9HszMQD8EvmsBGkQlTgJYtqaESYx+uU2F6QdoR4BkZzyW6m/ZGWsekVSSBU"
    "YZMs0gySNiikDUjasJA2JGkjrdgFpI0LaWOSNimkTUjatJA2JWmzQtqMphWGMSPjUFz6+dQyWQT/FPIIKmOoyoqBPnueJ0sK"
    "moXlvOVxsaPBBmE5LzcpVjOYYAqeSh7BS4mQSP8siRnz53JNehBSyfzkylyIKNceTJRCw1xtS25gquh+zKWOm0Wo1a2hiY1B"
    "kJyvoxS/Kt8TWmVHtNem2/iOoxANKXVmtniKqOjOqOqOVGLMGiWaLes/avOMoqxfeAdo//xUfUdU1R0W646kh5viq2KZNqpU"
    "asqCx2EueBxsFDxOqo9BZS0b9ZQVqjGCUlW5aV+3VvmmPLxxRa7qBZFNbcG27jWv+qjm1ayGstEFKbRwb9OT4CYQmOeiLZWO"
    "4PDjVeSmc0oBCrNfapqmoU7hJyjRgg8ljXpR0qeoA/83Bqij9bVRr+VrWhs9wpLyikafaToaf78h2ivkoObKulJZqvcnle+w"
    "Jzuu4VIuK46Ia3pnVT+0KNVkTvzjE9JEW92YvEKDQkxBPbKgCUNyqMJenQpM3kOj3ksZEBm/hL7slsRs282QXRfGKY9FLZGH"
    "3uqduKBTU4tmVa1uaYUL53iD2vHmXVRRx23iM7Y77LdQ2VBvho3HpZpBUCJ4VQG7gOAZguigddePQVI79oKvuGEPFZ3HjXsS"
    "PDLUcHHMvQ9ctiZlGQKnB1eiTqSt/KV9Z1LVTqGm8T15Oxqk0JQvFviUVGWzso66GlVHuBRRx8lBRkzOL6/tYKyhU/AovwA7"
    "bGri/qeTHg7OU0m1JL8K3ehlzCQ9YEr925DPnHcCV5Pj1TmJFsEpRof9r6/3gJECJgqK7RnKIrsR42ukqBlaRSuYhO10fDiO"
    "/cXmgno9aJ6J86Bov2WUZ8IA56fGCAY6qp2LKXgphYJ745s6GfLJdq3rDJJCCoB5yST+nkcEzyWgQejOruAgP+5TuQMRylXL"
    "sSOKFKQqlVRUVy1pSCWZzuZAXXIglWQX5qG65FAqyWjSSKsZjzwgUnKsLjmWSjIZwkRdciKVZGs6VZecSiXZKT5Tl5xJJRl6"
    "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"
    "gf+zh9YO+19fm/YQydDZ/1WF9GFvMxIq9lJ2c93O3r+ypnaGftxBGkImkOun8AoGZZNsP8Fuft0dsBXZMxRLYsCyESIjCE15"
    "yaAA1B+pCxhkDz/5VOYJ25AIptfX7hGZb2ZKBYj5E6p96v683/oz9MNOJFf2Zd2CNinYkrLChzgkydmmi6OSLQIhxUmKVxXz"
    "saK/IihStfr4zJepHNEpxasjarTSoF7TqEFTvcu0sHaBfl+sQYO5zd5WrPnuNIA1dkVj0OYpYZhWuv7pfX04qLn+KTyytz8m"
    "x22OSaPH5p3ISpMC8k+5z4Ci9VABTa4sYpX3UNFuIkM/fQLSQU1vnvnh+9a87Wexh8moisrYV97DtNXbHfrJd2iGHu92cbEX"
    "9Cm8aNPIuKxrCzrIzdfa2y4XcqtoomCAFLC3QQWFntQW+FVR2d7FD0AzyuUakAXLzIo1RIF/2WNTUPLrc0N1dlO8LL4YX9/0"
    "ZQJ5YvNSXChz+0vT2tPL3LVC2fd/N3t3pfbidnP99TcXRdXNDgg2cNGpvbhGroxzgQ1W5H8+5SGS006dCxm1c43mCdkKCUu6"
    "djcFV3iuJPrirAiYH3ZGgt+oyEs5Z1Isq3KsrJjFkpt6DXzEbybU2kYq/fuuVJq69r1eanxzqR6B8NUVKcCn2DY3Cj8z9J1A"
    "r01vsxM4MhQ6HKPd/QwphawlA5DzoOqeiT+CMhyof7biSKLXhnjgJWpDRyg8g/M6NDSE+BxwD/RXjT6wrdv9zUGICpFydogw"
    "VBuFZiJbCnNoGqNxXSbvm9YDcrcBA8ozIOAnB6ExDs5lRF6fGqBgBbYBhFVhsLAvDo4LX/XRT6SAHVuHNqkJizHJDcQ5EFuE"
    "N1D4129AStWLUoZ+2c7Kqa0NUIOZjThhpyVXLkqzlE6dPcs21irX5KFqFUfLVdl0FVZsubRD9zVOVlGYYHBcRTl28pYEk/zz"
    "rSnZrSnZrSnZrSnZrSnZdZiSZejXm+UGsF1U521uhmUrsZ1CW99GZb+Nyv5n+ZlThDAveZv7PEOZbx3JPEPfr+aX+JSopy39"
    "8Al/o6NTksdvMmmhg4qDLSIgJnwkjwBVW1Tr8PdpUqAYa6rsa7SojicL243Ku8RVKGu9ah6XFAgl9SYlGSJIPfwoL5h8Nlv6"
    "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"
    "fSxzbmlh/133eWeG3Nb8/rvpddnkvRG7Ha3YxXoPwe5j+XBLS/fvus878+G2ZvDfTa/LpueN2O1oTS7Wewh2H8uHW1qcf9d9"
    "3pkPtzVH/256XTUBb8RvZ5tuuebDcPxYftza7vs/oO878+X2RuHfVe/rTLAbsXyARbVa96GYfiyX7mB1/R9Dg525dReT7O+O"
    "CvPA83dTMbHGg4xeWPFBCH60hROb/kj75u+g37ubOrHtjzV0fs6eQ1bdh9vda4tUMYJGtjS6QtEHsluhle2Z7cv0YVvWKTSz"
    "A+N8nl5s73X/iZllWyNnA5Ifw0EPMXf/hl3bibEeZMH+Ap1b64f+qVlrK6NkLYIfxVY7m6N/o07txlC725g/d7fW+Gl/Yk7a"
    "wqZYRe1jeGhHq/EX78tOrLOrNfiz9maNc/MnZpotDIBV1D6GaXY08X7xvuzENLuabj9rb9Z4BH9iptnCWldF7WOYZkd77Bfv"
    "y05Ms6ud9bP2Zq0b7Sdmm60MaXXofQzr7Gw6/U16tBMD7W4Q/cx92uBt+onZaEvrVj2KH8NKD7Bv/mb92omhHmKz/Ow9q9gi"
    "P6MtaEvL1KexQypNfiaz0Kfpzm42ol1Ni5+6Q0ExbcHn4RZoZAs8dmGMT4zcNl7nzSg2R/KBcWiP6DCKeeIEiJERg2F3h0bW"
    "ctQnJkS9U7qbJ0x4uV3ChJd/pIQJL4vR6r+vhpH/9QuGkf9VCSP/tDaM/MsNYeSf5mHkX37xMPIvf/sw8j9U07uwOSCSuryY"
    "lFKKVMNF0zwFEsv5Hs0TTv++KyMGzdvDaDaHCNWpS76f6LX/LferYbf2bRZd67swuF+tUkjI9GIYha3WHg8Dhr8wIBDZrwQi"
    "27d1Vv11lAA50giTTqivWLRI12i1OEAFDGR88sMJ4MB/qh8x4jkTFwDzBiKjh0j1ANOuENn99sESQprc0NhP7zNIUCYeDKUQ"
    "C0UEX9mvwjcWVAg/sp/qV5Cu8An+llDBT2pZcZ4G5cXvUh1ZRK2XzBBv/1eKfDXzgqDwPZio3wMIlFz4jnHVXIIj+Bx+GwOW"
    "/tkPJzCitlsmOzKBSzBIaB3f1bMarwXspI4AMtTBMhVDnwEqON6IB/xQMYBnTJzz9YSnbHOfegl9ymcHmzOJnU4IYmEvGYQl"
    "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"
    "qI1ZrnE+mnxuuRhlhkulqVbUbn233FLLkRrft+RMPpwbZVNsfJHKDLksE8uwNVCuTLZ5DWQTG0lQolx4nj1UW1tdnE6SRv2k"
    "yo14x+JTXDo3XMYTvXr9wLdi3YzU2DGcpIGt2Wi25ErzkI2PLWbG8Jlkb9HbaVs6lzOmXaG57wlVqrKbMtV+Kb/vt8oxZsGV"
    "4vt4jhnoSq4Zs0uc2E/FTkOrrRmZZC022h63mjQ6Z4X0UF/qB2NpHISk1RKkgZVqZbPUYT/Ogf4xL8vy0Yg1RGlqzrWqfVLl"
    "aG4+qjRLzV1+Ju2MRaptthhZyTTH/a697FdBrtdqL2otvV2pcY2taJzCY9A/aBX+aNDTBtXmagt9VOFj8kmk+GRUoluT+hQI"
    "SWPeO9eXO3FArxpR0DWHq0WVsmO5Gq0rdqzVsPjFzFpWBzVaSZi98/aY3h1bzL4US/bFUy9szbrnWLzB5MXRaTupbdp8xqz1"
    "Y1Q9Y8+1RiW3jTFa/Swbk21q0q2XVqUhs2jVOuFUirNa8jKjlypUf8yOE/bqzBxy9V0m04xLu+7KlLqtcyvTGtfs1rCROI7O"
    "La43pIZs76y1m1GxnlmNhxtulF5lB9lmuzVMNbpsw05nuoK5s8/HbIdq0E1NOiZ7jWE6HV4OxlwqmTvz0iJlpDPRY7fZb00S"
    "Wb55yvTSifaUOomL0aA+Xg2X0xoY11L7Fq02+vtNhRPHUrVMbezFlhdOk4Q6tca2mm9NGz1xxiors6NJo01rM6vpQn9fb7Dh"
    "OMgKM23PpdV+aVyP1ud2rDGdHNPnVKUr1rbcSJnxQkKZGbFuuD/uNG1JNFvtZDw72vUOu/1gOS9nl3R0mEnOet3VIp89c0aK"
    "n07tXJUJ97TDKrk4d0/AYMzUgK1ph11JE1K2QFv7cDiuZaiy0pVme+lo9TOtzbAU3VLddp6p09ta06aSyrETVzbJaSPX2Krz"
    "qCFFB+fdIgpPHlSxuCZ3V7YhTU4CA+jG4I5lCDYJUbBhhq+ehHZwX7ohhXA0zfgZu6WYiYbISMBXQNnfiBvxjshBTVZAY4FL"
    "ko7URhQI0kRm3wqAl+OkO8JRKOmZCFDSA89jhW6oz1zpQHzvUCQ8Xt4mQplkUCIEKwno2NA5/FrL4X5yhEf+wRieRSlomoNE"
    "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"
    "M+TUosfwrBy14jTYUvauC4Qhq1ftKpWfL86n1iYa7U3rve7i3Gwsy1p/FBaO9XF73BrXp0bYzmbUvZHm+u3OrNfOmRy3HIZ3"
    "2TyTZOJUlzrW+crhxChW1xpmMttKYhMep2fWgMqd1HnD3O2AslDGDf08LJv9Q3h84Hpi0sgPVjNVoNSZuK+F4/FzojHe6kOh"
    "nGquOm2p16D79mQxaDfT8jmRVVrJUqdaqeyMijauL0vHrawO4+JobE+1nZqv0f1denaguYw+DKdX4y573gsS6PZSfbrBhhvh"
    "aHKrnxOrZK1Xlqr2frvjx+V4XxivZlxtrsYsoaOmcoej0U2G1UlSGTbP1aOcMVR9qtWG0xF1nJ6qUoM67drdjDrXrZ5OJ+rp"
    "aG0GTlbnwEX5xEnfpBpmWSqNJao6qtSyZ5DUkwK7SpqZwVlJzjhux5yH+XRYDo+7eqlvS0JtLkezKUvJbgaaskjUeunZKl4t"
    "JeM9ZrHIbFiQzByiq56YPee43LI3GS2Hdnm+LNU71XqZHqbj+UOi2+oe5H6/clwelhwlAWoE4iNWCKvtppQVBk2KbdYnTL9N"
    "McMpVV30svP2vs6Y8+OAGiZ5m0qMz6lhdZNV8lOmlNlMFrOGHU0tjny8vps25Ea/3LSbVcAtY3at2+v2asdEOpqOZZWJyY0P"
    "9Ekb08sdvQvnBlZn16aOHardG5Wpc6uvlvRYs5Y/JRlOPLFL42jT03m5lbS4aFpg7HF5H+21JaraY/l2cpkXS+lkb7+NJs1x"
    "LDrMDXtLStwPBandKMmZodAajbuZk0QLdOvcPiRzi81pUKHUSbO0KDU64waoVCmJ6thHq2L1Ke5oHen+YLpf2rVZZ5dpDnaZ"
    "2SotTZrcMG2OGSveluYDRdVti2sL7Kyc2cd301lqE43mY6bJJFfHqaLVZt1SebVITxegw+YHyZYStVmtOjzsj3tl2Ml1p3Zp"
    "MePiM2m2EKqlWWcn8/qqmdDYdie8ibJyNr2biMdp7riPn1pJ+ywPjai+LNXz/XA4nI/OJXkvlK3WKCx028vKsG1MavPo8aRm"
    "zWzM7pWn4f1mc6psOvV8PZUdz2pC5nAug3is3h3xQqVGp4b6dtVLhoVsfRNd1dNpXbIPCVouN8Xu1q5EB7XoodOs8JsuX4Va"
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
