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

PANEL_HTML_SHA256 = "ffee90b577b97f748912b9bf5cd1ead4aa34bfee7efa9727fdb63e9a767fef64"
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
    "4kHGxl+VUmlJS18nsUu/JLFr2oCkBCLjAEtu8NxUSusEv2VV2G/AKd5sEGOcoKz33z+OeEI4rIRiIAJMdkSAyZdEgFtgsnCy"
    "8M/WuLvkhuggekDSyGfjr91I9PHO/FKdc7PU6LbJCR5VMoILP8Mrg/41VBtx7tjUuUzSuYwSvfwWfhCkoWZJ3l+SwD4VCULy"
    "5QMqAuUPsMtqbJXKmfFXrlnebL7jD5KlpL4ewn0CFnDi4pcbIhVKz3aEqV76Fb1JueFeFQ0MmoZwsNrCSeTCEeZ31b6S+yoG"
    "FwfRELxa033lKyPnIYp5MEUD66C6yDTSdgODgQRJE+afTk8MHDC8NIen0+kJqJQ0xbu+l4gDA9MqaAEBt6Lb5Viv9hGm/rXK"
    "ylryKSYv6olnmHgNiRoc/woVLE3dSqXMlCR/cs5FmKdnJz5clNWdF5dm6Z+h/NV/dLSGLwHH4O0jvaF4VZ06ajwLOKE2bxUa"
    "uDJfTwUloF5YKTNTErRTL5WR7EPOBYT3d1Cle6k2R4vV7olCD2BL070nG7nMO/oBBwKxziEC7LtsWTvgqIxvs7LMbht5l1k+"
    "Z/m7bMmj6+7L5O/J6Lt7DkarPZ/xNyFpISwnBpxtyw1xGV2z33/Gj99s1AKQD2IJCGHUu2yplbnM5vd6hkl4PEce57bWwG+y"
    "gecL46E2SD0XaidQqWSv2PcwWoEwSMSNhGdtWucacTXPrnxi/8PDa3NqZbqY2S8dXq++cPwQpfceP3RTPYjtLodlXwV7M2tV"
    "aCYQD5I+Etp/RPk7Z3uVww/3+4gb3nmY9PG9L1D+7crQTuk7HmOtlhhTOTI7CXUOr5tNrJSJSMWlMPxUtR3vFX3BzoxHjRN5"
    "jyNr3J1z6TtLKu+wrcS5Ebvteo9LCV5fU7rGWymiPHn7TvC8mF7xQ3vv3r2xsgTRdp7w0dF6z+YzeVRTVeq2Fuk0n15BHkTA"
    "nU4hQnpN8liDkNqCCP1RyEMArMDBpTIMgY0oXK6Si2R+saPr52aPxsWKgimOcbEC10+TUHxotylqBI/W+HPL679fPWARf7k/"
    "I2EFvVvtN5Z/yrjMLMsL+nb1ZYP6J6u/Naj/tPpKg/p3qx2D+vvV/9ig/tnqiwb1T1a6cfy7Vd2g/u2qblD/dgU2kZ8efqeK"
    "uvlUtXsJBe7wTfpz1RusFKlCtIDQTkz1iFGQkZSawwU65GIEX7CW4T5F8qqj9TQXPQW/+cbbn2p163Up3k4/rHzwLqGPV/7J"
    "2picr7ezYwKWRH+ufNA1nG+mT4kxnbdBwdI+PunMpkV7Om/DpePHlX/yMcqnhhF4RnCIj2RyTmakPSUn19UC+Gn1/91247Xe"
    "BJq1KaeDv/OHC4+QjMJ/5w0OIBGwLn9X78lzrScnk/Ppp85MnSj1zz5o2nBB2ZkwldErfIHsI6wbCj1fgVWAtNVpteZAoGC1"
    "IwwIPOAZ5VY/pr7aU0/Y6RytX/AjxlaTvbzhZRUg29dYwujMEXg3hBC3nVaLP7iIwaNulf9lH/pcAfh5xdXdSuA9zYWtMQa7"
    "V6lXIrUumAeDuGAzvYJx+0dI6LcrGRzWsblOaCRUQ6ZUEYm/Dn3Py575vSEGXHUHm40BqLzkzB8NWi3+fOpbpk2/l2Ute8gr"
    "6Lmifvj793XYdrUeflHnFe37zJAcahbWhzIQ1ecVj/wjnUK0gKaHgMsjBOfZaaogAcpJJsBumC/x4hDnzE/rtUv0gDE77bda"
    "71dGqU9iBjZwGYAqAQxU8O1udquVgfy73YbOmJVDg7TDY8IOT4Pb2u1ErvUXzg+Mh35UsHnQ/UwAUR3IxH9IhRYG/OVPV5Xy"
    "st4Vg39/VNNtwszUv6djE4ipZA75tDUyHcy0enszezxz0Godft8cpo4LA1VOso6LcFT/N9R//VPZeGVyKtJWqznWDKYCpgHO"
    "+hMm8OBaLfkMto7aILN9g1yrr90mcoB7VjXg367wjPeVHWnWeIiVqXaVuwa04vT4NDkj/rdnbjb5qT/kvwBPCcCnOKYUqHNz"
    "rs6tTRmkjWz1hZtNo2u4zJvfraxQ91C4/Bow2BznwZ6PbbVYu+0xgHLLfFbBZiEgVLt9Zo+q9vLTU79BeRNOeTslIUawtWkU"
    "JnLFzMRbnc6ptdkYVisnh37bkHlaPWdnzRZPOHXVbGmZrk8EW1qAukP2cKoNV+YzYUsrPfDCsezRWj5s5cOpZimwS/IWgbMs"
    "cKgKTySp+/v9Wl1KuLnwZEYT/G48IdR84s4s5uoCfLjgVcrNN6vKdwV+X8Ne+HhFqo+X28XjFZ7lP0a5cbR+tTJSCrar1W6y"
    "3mzFrlxZzuuy0bxbZLcMRELCxpxIq5pOCKBZR+u/jIQEIT04WidbsO3bkspoC9l+cK23+OOKAs4e/XNFAnAUig3YvknoSZvN"
    "bfVuSZQYF6gPbeErJ59Ku67UJvJMAPsoiqxRjoUy4Zqg7OWqYaaFZ4g1FObfANv0Xyv6w8qfhEkM5i1aqONKPoEOmSiKDWue"
    "mRVQVcNYPPVPzpcd/YJZkiBcTqcdkHbLG+Y29Eq6jovCyzXLuWTr/7UKAj3UGhhRosXlD6u9ppaUY/cIIykBkgYwKtOT6fFk"
    "Wkzfzo4DI/Cmx9OTzRHZPDICb3L+aDoVN8HNdMpLkeNHweYbzP1mX+43gfqxwTM0HO2m086MTFa/zCb5m9lk8WoG52ZQg04e"
    "fTPDY/PxlAQbY9pddmD8lZEm2RholTmJn+B7k+jxbFK8xf9D8n+fH898YZ85OZiWs+Nl58s9r5V66AuWHSJMNCX9wnj7OBGl"
    "NhH7h9TwxZcFvDHCPwqQV0llq1IthJybLOZosCwtGr08qFTGibwgIBULk0dPtyv5bVWLrVqxvHlpCCOKmg8bq+BCVUzrjqWC"
    "Wls0aVZC07YD1g5aMuGJwqNA1CB8I9Fd/jQiCPm5U1cCr0l22+loUa/HzRbQPgBOMLz/vJAALp2kNJphgNFG/fKlQ+myWf9u"
    "oaJSqM06ilX+AIoVjzKQo807QEgJ8UcJKoM/VspmIWo7EGQR2ONWGpqok2oF0Jjr8a2jys8io/mkBCUraScQ4jrz8bc1a1vK"
    "e025ImS6+r9GAJOZOCojjkikNQ9dyU4rT+pM3mIA5DfjIL/yhFXw4xV/eBQ+AtTRrGNJs5TNBqxkfN8vODJOQehh0mpx6FUj"
    "bLWidpsWfHdvtaJOh/IsKuxJlBNJqxUhAVSADNLpJFOGPKnw4KgZKyFxB7+twNOYeP9agZEPzf2sbVXek5rdwUoTkk5MjohZ"
    "yvN6KU1d4OFR+Ai2jkkFaYZab9IwDLZoxyKCnUuM1MoYTeozQEoD28jRSsGFdMFOMdSwCOa1OPNBtS2f/3MR3bN8Whyf1K/M"
    "42lxfHSi2Y6MuVltqrDa5EZOQyJDfihjNzT3k2umZucs7RuF5ytHdMnPFGJiqXT4bYualBEa8i5CvDux91AwqQ0FbOeE0XQm"
    "TxJ/kaqI4q3g1FFVkm/HocfhL8q5D9KjBGVe0uocvCUKlIpDCEovz7KSemAEElJ2hwDu85fs/g27ZnfeybkBI54AvNCmgECR"
    "ZRJv4ux2maUsLYvNPMkZgg8VG9lQsQEM5xyiX96yzQIk15t5lH+A126zdAMNbgDqhBUbiIWZzjdxUZCjk+tkS7MlzGdR6yy8"
    "/GM2Z29Vp4v7omS3IY2L4hV8NYCnxEXxXvYA5HtFnC0ZwKpstzSdVwJDgV9W0hL657Eu/qXLKI9uCy+lAv+78FhXPlIAh8lS"
    "j3Uv+JNBaMFKb41DDjYoULpg5Sv1u3uJ/o5kS+clvDcvt4Tmc//kHERWUefzxWwC4h0gwWQO8oE1F9IpzE4m9Hp7cbgO87nS"
    "NAqIG7CQkTA3YpXczQ5Qhfwax/KgjD6w4mDBSkAHpgfz5DopC3rw6OIRwE4/6jzqgkvFD3jQwffWmu8xLH1AVlKD4pdzigkX"
    "OICi3yKpGhhpyCwzsmgu9CgPZSfpNeY389C1okC3N/zNJ/gdpL6OyhtVXHMhryTSr6VY7ucrBIgiY8aHuCm2Gz/0FiMKOKA8"
    "UD7fDWk3fhPKU3kkhsp1+ysh1BG8ClR/e4X0Ips+XRnPVuDY/DdieX7PaEjFtwRwp5OiW7D8YxIzn21hNSbxwTUrD+T0GpUF"
    "13x7gWqZF3MtESr4wO4DXpOcT072OMDyuqGWzHqrijUdudCDSyNV7tO19bFywRKA48pGAzyNhcsQnU7QwwmtVSq7BH4P+Sdw"
    "Q5Bn8gioQqcPLfPFX259pqlAmf9MDmyN1GUXwN5WPo6TuXQG4X0mtdWgZgi8iiRTY1tSp+XrudC7IRVz6LIu+DNIopc9EO9h"
    "zpMovuHGTzBf2FxjXngfMJtX1sjH7EA0BEYjWFT0cn9ZkakK8w/YaRdTtzdR8ba+OHWDwL2LlxuAE6i78WrVCFjH7H97to3m"
    "c+296pxyaI1rLt/gOVtzSZFAZeh59YW+cX+A/fng/JMS7psOU7xFFeWe73jgfQleBdA3e6aTz2YhMyUjFjQtSFnRlljVXcTU"
    "CnlmfAMGcyHlA/x6H01wYtgWKls2s0PQqkvcXw42wr8lbPb35Lyv96L6qvt//4la9+Bbf95L1IKa4WtlARCqzw0WfM2S1kdg"
    "Z21jl3ED+9KXydf+00/bbYXPFpLSq8YmK7vAM9XWbIhq6jU81P8HaBoGWa9yZ61WR4HZtnZEqiisKoJLiZHtTk93S2qLZvd7"
    "68ymeejYJoX6wd/d4U/NVyR/KnZa0z6j+Y74GA408vWvCWASNQy7X/fAizofgSl6D/pF7dOu5916FqcoXKK0uUS33O7lFqBk"
    "UfXXqEbkie0eIYDlO/xysP81zl2e7L73VN4kvvTe0z3trYoyu5VMC1/V41VqS1adRColtlLbikBjCV2rq1HS5THUA34yS2hK"
    "giDF28jbh8tQvgfonTYU56g6n5OKe5bq1lM/7/BLR0KT5m5a8FOcWno/g4nQk7dvhd61Nm5VAcmEZf14DxJtNhbxtkkLZR6l"
    "BfgoylbQj0Ta8uZ6m/WiIJcDm4daw9q1STC9nZsTHiEb3aC1K4aazIpQcSm8vWEaIcB1Id9LuVrhPfNT8vnJlnAlFKdfLw8C"
    "sYPL47CYBb3hh1r9X2tPdP1HMKjnnGvP3UnyHlH4FzwrfGVhyRIFPni53f8SON7D9XRPVgWt9OCmhuoEeKva+/bUv9k8WAG+"
    "C67DxocVoZG84VPuLMbv0xqcY91F9aAMAtZghhzA9PGqBudz+HgltQs1b3yIJVs50dWwzip02JVRUx/pLnqa/AtAu2oVoFwt"
    "5wYLgDIgm8kJaEmzej8E/Gr9OyqtGQgWgxLuSMk23BK5hHIKsHYRBBjSB65y+Dd0R4m0Fq6P7okkU+G0SXFo4v+0MsChgozr"
    "AEmldBLcbMCtCy5qOSDRgM6rgmvItwRxnSkHbcQdCo+VFzGeSrown/zmC6Ecis1mN49HaSgkCAcRoR0EUSQ67udhQhd+3Gol"
    "3EoKOpVuNwCjsRHaOJUQ0pUfB8gyjYW85aKfpLE6rCDu41aLHxoWhMjLSDaJZkF3CRyczkFKKUGxwLkdz5jgKpKlEF/m6qF8"
    "FKiFJAjmdPlQGdijsAj066rVWrZaV4e+v5QdiWB1X3XR/ohGYu4h1hSkLxvp4xX6KGRARth4B6o3YAFTgOYgIdcir/jrxlUQ"
    "LEnQrESM+kqXYv2yMlZABisZVIbiqMG8LehK0QJKSpoXyogEv66McA1Tsg0lXacUMExXxFttaaEJ/eK58QP9oVtdsCjPoTEv"
    "BAxbrQCQnEY4o9IQQuA/AS2qzaiAwZdbWuxn8MVc/EgXfoG366vkLggyVUqm0ZVvRFBEF1nWC+o5RLpQRsq5SBidqfHZx8jY"
    "Ah1xD1eyWDZH88MFjbtNOa+Mbj1XyEO6RW2+2RzuvhMInLb53NjNBPqdt1rh0XqxnZqwduDfHLw/+FMFG2J6WpvCTyqwPHsb"
    "0qU/D3CdXSmWtvQrCenhPMAVdgUWikv/1xVo71PQ3gepB5SRapSxoJPdbs6Aw8xbLWPOYWT8xbzVmisYmTmSIsTlJYQut3Qx"
    "xzBZdDXXgYDpnMtv+dJazSUgsGTTm81qLqB/S8WLAEb7au5z0CjUSxZz/gPHSFNtLOcNtBokY3WvrV1Yx/vAKVFVF/y+MkDH"
    "qWGCpagxxcDtq4WB4L/reekV8y32gYBqhF7N0Rn49xWC+qkCdWC+uXK9kz1UkjO1JhS0jp/KZVCdtGEvFT92dRAAkNRM3Pol"
    "zUBLV8A/sT9ZgzeoB2L88sbLt0IrG9fA42plyq0fd5fZ0iB1qIi6p4/cy8BEFMx26cJHGotI8AKsKjz4FwxBBHJ/hn7yIrIM"
    "by/h7UGsB4WNy/xXK2MBEqNfQfe2gP04gp1So265Tem+hN+tDDws6vs6GGoKFdoCwNJrGBF5G6xLwIBXHVek5hQjVmbCH7FS"
    "i3XAiKXL4fV5FKNCaJyUdHUtcIco35e9jM5ZvIjyiF9iFqCK8V7ihxGO0X9TUbru4sbQvfJHoVoLgsmMkBm9nvtrsZN7oil9"
    "tfHNXU/awlWJ3XnrfAW6nzUuCG+9jEDRkXon59MuOj2CrTBXupCjEyrh99X1GH0OPOl0p4iTUfQ0ZHN+VxG1NrAvMCZsWea1"
    "VidG95hMZ19oC17ZUcaB4JoerU/Ojelkcj6dzY6nM1K5lskmgxCVdtLoh/9CO4lt+LVdvmXzJNL7/E9MOYCOf6HbWOg/GSKu"
    "t9Mb5ClfaEuq+qpBkn0ELsLyooPaxg4//HgHcDYhXz0AMQoPvLVo3WOgHxR0gFl7vxJu39ut3mXO+2tYuChLA4LUrcBQqSdq"
    "Vkrlqigel4UAjSnnY1lMuhSW3aplop3/RPeIuDVijV3emP7GdksvyqzSlYqTPnJxpZGRZxUQZShZD78P8qs115/AuUlpTkGW"
    "kVf606QaukgKMzJQhxZbpf1B6a4SfyyArWNEKLqEiPf0mgNSx0S+IFTRXkmlMtpLKVciexdbP6YfwbRzvYXo8Issf8uJ4n7r"
    "f6SX/v9j78v727aRQP/fT6Fus1npLayS1M2s6k2cpEmbq7F75uWFFAnabChSJSkfdfjd329wESRBipKd1Gnd7CYizgEwGAwG"
    "c7xcd0/RG0vKAS9J5/N31RqH2fwcHUCN82qNs/kFrQG4ZmKczc9QiqHwGXpDeW9QqsHzQ6nYUTYPMXoEpUIsFXs4L960fie6"
    "ekyWLhapewmxK9MMBT1TxL95E/Zlwkucs4XssvQ2Q29ijHz8dv6QBlPxaYigtIfe2BhFNIPPooUWPfTmJUpoMr1joBT30BsH"
    "o4CmEpARhsQ1Rm65hYMeeuNhtCo08aiH3iwxOpFbOCJXCiJ6LMqKYox4PYszDhY5HUmUWYgj6WPkEa+LiurWnUsbTBdeZlZz"
    "O1ltfQdDXQqmuipwpSduN8IogYnpoWUDOGsAx8NZgURxyHaDcFlqjcHaDPIJgOzC2qATDOHEKD/nwD2D8nT7+4StC4rvH5nw"
    "k8k3HlHhW/OT383yfUhyPJ6zyhCNa0fTlzz9JKPSLfM4y1AuIZUoCxeKAmnhhCHkRCYmRMbPiYydE5mIE5lEJTF1gPIEWY/c"
    "7tb0tgN0hlIY8Hve7a7nYc3Va/3hAyNCxFqFc0vWntB/sQi77MzTIiFZc+q0JFzRCdCoYyACTomuIIsWBI/vSQKU4d18WSVM"
    "p9n8HbqABt5VCdNivpYoznk2X6BDKLqQ6M3B/FQqc5bNDxAmtOtAKpTi+e/dYzUdit+a4pX8uIcuehAbyiT+0Oe/dw83Vzrs"
    "AQ3htY7mv3fPN1c676EzXueeq95yMeyPFFeIIih8KlKVWzCkmwYl6i0Yo6Mqza1uPqkVC3mw+VIhxQrFryP2o4dWbEOekM0Y"
    "sM0Yqjcj4JHL95PHttOKbieK9vx5Indlvt+nj85373I1ausN+DnbI7Vd8E+As3++tTJE3zNer8PUX2LOG4C3SeoyukM4TBPv"
    "76e8KJXD556l2Ca5e/eLFG4jhQeSEOUX41wK3kPxfv7uEhefWHgLYK2m9TIzL0iWpwAvQLG/n18me+Wm5M+7d5UNFAuRMFPS"
    "s9WnYYHmzn4/V7Lb77/Bb0FgyF5qKosc5MIGVqRIVjlNXefdNIDM4JQnhy6xyw5hAr40Kw+veVY6ThOJ5XMWiMGsYc7ceVCa"
    "M+fthw/Bfj/XUCRpIFhVz6OD1s3z6PB5dPOud5lHrziPJK7LQzt+f5BTc9kbQVn9UY6dSaXFd+9+oUPQnozy6qIVKiQqul8E"
    "aGp6ZNcGuEuL+wKJ1FqFYV8oQlWyTEXphuI9881bAnfxoZIJZOkuEWqefjZPc1+VucrrOuj6feIo4cMHvw+L9eGDiBQIsipi"
    "ioJUb38bUJc+ycWQ1orz4E931A8z5zlIVHSBd8XbFO+2iEkbO6ItI3eegNld7h7onjoaAGEMChEBJN9KEcivWlowh5Q/EczQ"
    "IgBHgQkoVDJDof+SOSCO3+b/BD+aXzlJ8s8OO07ZBAm3C0nXYT7Khf4wCJ86dy7d7Os7lzaMLvvvV6Ta11b+JkDCEAjJFD38"
    "/pRVvYZlRPAI0fdDIuNMCiSPbUqJ1q57ZjHpoLvu9Qjfirw6bHD3d1sWXF0Wjy3LIui6PWlpyA6rNAMSFeJPdo9sx9M1pizH"
    "nu9yp0jg+gIVtcWIki+37SO+OBlu0mJd4nqEu8+vxpmvWAhR272SdQR9fLXnkhLMG58GZODWvh3Q4I0C3D+z47BrEfA6YZR2"
    "PHApBEHROkS4S032esh69+6HF68fHb589uOjh+/e8VcjmxxM4ECmSyEkD80FJ28R2EPnEZkMYdgLgTToUxq8H+rs414ednGe"
    "7oNXrVLPpac6IlFM4amOPwaxB7pqTTiSIsWkFk3OpdZL0b24hRARznP7D+RwW1rnvwmzkgrmNe0hh0xQQJu99ItOTZ1ej1sH"
    "lzJQ0KOS8PVcB37gP7qOPLCpIh27/03u3l1/rd3LT1bJ7NSlwvehtr/+z39M8lPfX+/t0Z/Du3fX1Pukx7wLdL2520Puf/4D"
    "VlLw7qt9+OCpAQ4EwNDvSuAfhU94Ll2KDO8/OnL3Cj540/zll1na0Cde1lUkK4/nRfdXcPEmy1d4fLCl4nLVlbgRQZXiR1pb"
    "a8kvQLRS/vsenwiKpHUo2cucuSsMfSWSnhSwkHAE8zdvCSaGObXkEt6TnnrTHvixsw7A2wmIi3Ho4I6LU+yk2O3YKad/0ADs"
    "4MJlmz8V8QLkEywY6asEU3DnplU5ICglPMg8r5gSrSY/PCa+ZLhVPWTT7SzpqojE/OFHOJkUeRxpEinObnFPEhNC5Mj5lyyV"
    "2KiBQIMx28m+vJw+exo3iUpEoTZVSXFMGRMcVvzeb0XFmefFz3j+G1OkCcoGzLn1cgAPGYEwPcNBV8wbuCkhSkUdkYRC+n63"
    "1ZpJb83CKXicD7JEC+l88ZfuGB4M5uLl9LIacCn3XInBsJIdOdGcWnyLx9Dqczno+VClk/6dS+LyJrNM8i8K5vH+mxjht3mU"
    "NBOzt0a713O6IGgICvv7kr44siHbiAseJRZDmrI+1yhh+722HNMqoTvcpFcPxE+2/DnQh7hUPrOFTMHlKVFyp6RAgjOlcCYm"
    "OKe6l4ozcg6vGV2uRQc7Xjq6SQ2iB5EflrqIOf1Ge0ssbEvaNPtggkXTcuthcSTbKBQoFROrVxYPiKniFDWd/Iqmkz3332Ay"
    "WDvvWJg6Fs9+uSNB8YRtoSjJoUXhG/4TVBXAIB3CFYKLhpS48eTDcRhWIp9wwrL2rxyuiErZmJYUJSRcWYFp2hUz5YDFQoMK"
    "vB+z2NzMVpnrPJXCHnf8ezzAUS3mo6gYAjSRQ2fRGY8qMw6cEH57zy5QGJvp6ZXcPn34wFAvpfbEjnjPhu4C2buFYk9QildT"
    "hu6HuUwi+N5gJ+d6jt84lKVcM8UvPjekQsxCbXIHu1QHR1ImkVBC8PLlcND7KX0DhJdBEIEIAIk/HMbnwYse+ZEhjzBXPW/u"
    "dotkw+WvNnDau+y1pnfPm2NZLSHX3GFjznP3UzPPpr4+Gs59oerHJsHrIY/e49g96fU6wLlu4rxoqkoDX4e581zm6JQ8qTOX"
    "EnyuiZOKuCf0JSm+fYW6+3PS2v/t9r7qFXQRClbStMfe/htyIeoQlYu35ptu3t/du1/9PwKd+skefHdR1m4EXgf+kyK5JbGw"
    "36279n5B+kgcAvvILiS+zS2ATT//LZEUMt7v1t10f98nZANVlLLplIK7YBBTI4fMGLyDlCQuQTb375WcM6YgrU6RXzSVcbiD"
    "OYfKmol3efYb2pUxLZeElGVVPkjGYlmbbl9Ic/f3SygwT/b3nTz/XkoiU1G83ZdVneiTfihpHGVgCfw7DVWB/0O4JH7kvTk4"
    "PHxr9fbz+BUsCYG/kDS/fhfw1CeqjqDt7vR6xKrb6plkRffBQ3JG9hx746Uig1xARSRX+XcGYXEDorVD+ch10A3ILySEDTCe"
    "OOv10O+sEAl+Ov9u3c2lYjQjI2++9n4/17rnVaTQNlzPMZN8T75zy2FsAxE5jaj1nLrzy8Oj+6+PDt/99PToiWnlu8ZCBy9f"
    "HN1/+uLQtHiELgu9eHn0Lk8Po/RAZD168ZC3gkOXtfHo+x/uPzs0Lfz72g5YfZ4WRukjlvz0hWn5oYWePTo8fHf05P4L0wpS"
    "6fPdy9e02rujl5CFLfTN60f3jx69ZqWP02JKscIxVHjw6OinR49emNYCp2cYhxZ6eP/o0bunh0DzU/w0EQnvXrw84okvopSl"
    "P3j0+OXrRzT9AfaiGLOM+4+PHr2m6fe9FMdWhi5AA6ps3WNya2iFwZMpLkqbLJ9QxfQpr9vWBgpVjaAKjTRbQyGlOVSbBnK7"
    "KKQ0jJJnYYOFVIYWBR3OHOnPXc4yEQ+ZyJ8vhEZn9wvcr5qOpVST3e9xAkiilVAOSaQx70/yKImDhnvcMTp/ae6S4LUJbZu9"
    "P/OTNqVHOLAcX+hZj4TZ+fBhwbVL/Vy71Gdn6puw2+N6RD3qDL5shwZmVADKoTu3/sFoh+xlo3P5j06n0zHPTnCMu6afdNnL"
    "IRD2t6hDv1bOXkL9LL/t9dDWNUxzQfbELjVt2DUMSvhvEZ3vJf4ffnhsdqj35L1FdH6P5Gf/kDtYrNMULKy2HdIbIkT9N63+"
    "77c7N0DEt1eon6wXS79tA9IMga9+Owb1YLNDB3Evnz3beX8cg3iTKtaZHcI70PgyymIkuJzZgcuSlE/PtI7WSaLAd8sZe7Ht"
    "+uvE7Gh5DnORXUijbsULSV4UpmbHD09w7KfF5D0P2+k6xnsJTlM/PE7qyhHeAualqSR1lLGXrGyHgFUpwOankg7C9T078I/D"
    "ukzBjpWnzVnHCbS5ivwwxXED0rp+AtyRu+3a8x7Yu4SqBz9crdPtkZIyV9vXgxmxY2xvXzNapYCDq6vgfwltb7H/qtgfQY30"
    "wuzoKtzaarGlJYsxGGiYnVNwSe7YgartlucF4elPosAtnBplsKUhUn27pX/e9cNO9D6wFwi2EaAEyeqMtH8hGVF6V4Bu7wwv"
    "3vvpnh+GZPr9cI/ueHSFxqJ1WmxMGvgJCQRhdiB4xDXAnWA7dk72XOxEVEdK6oqXUe+/K3UL7DMo8Oxh10+lLl0/WQX2BeBp"
    "4Id4zwvw+XV3t+f5OHCTvbMYBiYjVXlTl/rcW0Z/7K39PT88tQN/a1JOGB0S1a1+GtfBDmR1hzpLHK63hT/wk5S+Bpfpp4Lw"
    "NU8l9eq/9fwRYuxEQWCvEkw2Ovl1T1Gk/qQH2SAQZSVgjk3kGjuciic71NkahRpZFXJYnDHyUMltXpATfXvwT4wd6gx2qDPc"
    "oc5ohzrj7eustq9CIlzg7estgsh5DzGydqjr7kAi3K2xkyAgPfe3RM4SASni5tbMR93256cB2UX5iVdHKGpLFMGjpwnegZMO"
    "8DGomu9EBhXM6tZU2F8e73BtON2h0qnv4mj7ao4dntrJ9vV8LwYZ1db18HLR8oZWPIDJw/K26yhYHbKx8zXjLDOn80vfdQN8"
    "bevXfikkWJf2+d4ZhLMyO7qm/esKwJhhlHbfUCZ2Z9BKjWzBHZ/E2w6dt6htvkox7iONVmKyVkruNVkvl3Z8sTPGEE7MT/FS"
    "1bjr20F0fG1UpeVFV8KQMm94vsfncKfLu2q2VYy5FznrBFREth16tE7hrlFEmuwfFjoAESt8/h+GpkWp5y6SzP5qTyi9l09O"
    "z176wYXZcdPuv9OLVXQc26uTiz7k9Wnev3vbixDINYovQKlpyOvTvH/3qnAS7tpfBHgPg3RrzyY6rrJgJvSX7IBc7dEPXKx3"
    "vrJDt6P1jaSD7QTDzVYSPZzi2AvgOnTiuy4OmyAIsH2Kd4CA3xF2guF/7/EFOUqS5vFJCxlHS+kT/juOfXcvxctVALXj6AwE"
    "SV6cg5DlbEfUoq5erdsWXDEZWwOsXwFgrRbg/mqPy0VROaHzfzZJQQl1ooLXPQwBtZMyhVknIEahcs7qlVvVu9icQmCrkjnB"
    "NuL5fZZa2UC+mjOGuhBYg3gwqe46yJIqMpIKUUz29lbnJJs0hMoNSaSB7/btaoGoZS85if3wfZlfZHARiZRKPiRvwrygkXT8"
    "0PNDP8WECtkSGmxZhSM471FGdBVk2r9KeClY/1ySH0fg07yrufhYmgWC1ZsK5RsAuKHWXQ1GszadVYrVb/K/2dj7qz2g14F9"
    "sbe0k/fy0St4CQntocxenkORHxL7eWJhCzBWQxQj3/Khu4oSn2Kt559jiUuC2L4yO0QC/MoJFe5Z3qgVnro00K3O32rFfEBk"
    "wskI+u6a3m3/3QO7iDM7dpNGELY5gKsVdwJBeZ4phtd4msmYoeQvG06z68EqjsLthkSnq/WQrgrWxnE3zFmOKXbonEQxdsVQ"
    "tkFYdeXOQNOWScdZL3xnb4H/8HHc7eszpKO+YSC9cnBWWtkGYdWV24HQzIY0oXLNwBvXXvAgWi0pTRw7wF2tPxv0tmEUa+bg"
    "sh5NrgqL5DXtjPg6/ef/FvYCB1+d4GCF46Sz16H6///kKpdnwuMdjVM8n1u8CUu44KJZfT/FsZ1GxHSApFj7KtUpFq0jU+lV"
    "4VKbxe5AAVwEjJjP57QMDVPBYBC2mfscBlN0iMiYc7dsRywmDg0fXPRrBiqdTeGamfZTY0hn4jCfaTsxD0diyGk5UnVjhIp+"
    "Hp8CVBqpPQ21Uu6Gkl6rFIvoEVlfHsg4nev30v/a8TGJZZ7IsaHY8EUexHqmSlWXmSmn3kv/ZewfuWxawJ//F1pu0SGP7CFT"
    "EXuTvu1lPXNTaOz9mrDaXbwpfAdEHeiZMkxqgNSxQgDGTbMfEi/aklWtmOCHQg+Oaail85eg69Yj4bbre+Qe21DrSNx5BG6p"
    "+5euUJGbvwdQhC1UvnW7aU/ai6mZ/seSwhC/d4X7C7IxCu5epWCCBD8kAyyK3f+11hZxm4ffsK2XRq+4F5y3PWH0wDdKyOM7"
    "U6uIM7cbSz3y3uJ7lVg9//uf1HBnuU7SDrcc6Qi3Ox1qjiHMn6k9IJuRfRpJzaQBtHoFGnBfmsby9vhav3s33wD6WzGmfTkV"
    "FiucP7apHWLI9RlDydfC3btfhH0/YQ6p+Yx8ATHwKnqpX8SEYnyhITDOyO49jLt+D90nf5/D39SXbLoPMbrPgHYS5cTfXDAp"
    "c+X2MvTCndeYFlvoVdFF52MXIoHkYbEFcj8Glyj939c4vuDq3PeDoEtV3d/cuXzhZvM7l8fg6zF7i6TU/0eTien6/6xe9tZS"
    "7E0x98TORlhFd1+41Krzi/k8vHv3lcs0a9nU3evG82eu8D0U99ch0eYEBc4YL6NTDBqsdGKeufmoXl99sYmN5Z20+wVxS30H"
    "tF6RD/8Si2tkz5frbm//zA/d6KzvRg6pyQwhiSkPT0PJPMrtQmwzAtdIfX9JfPel4CLXkSxFHLSeg3u7cG0HyJ2vpTBYa3DY"
    "QZXfV3Mvb5Ku0Ls7l//5z29uZpkeWkIHLjqZLysmKkt0DO1D3+jd/LiSf4xOoZModDC6mJ9W8sF7HFjZxEmKzucLCTzwsJT2"
    "o5AhPzqYH1ZqH6IzUoY5YQd/S2eVQsSFHJQC/VzwpZTiSpkUo6M5cdCwStCj+VFe4DIzj9BDmIAkCgg3imI8f1hp4SHy8fy5"
    "i4gHuIJ5LT8ofTz/hlp9X4rVIkT63n3c7d29+5ukxCwOeR9gDUEdGVonPnNezuUTirhU3h0picUqbSWaP3K7j1zwG/Soh2xw"
    "uRUxjx8rFMyjvu9++HCC1pAK6/nhwwWxX8HMbphqULOAvzGztE1hlzGH1XjuoAhiJxb5N2rIQtSqweT1kiCDec5caL5DpDNz"
    "nYFzq/1+14EgEKw3Ht6cwO/Ov83DZTGvSN8ip7e/79x77H74MHe/mM8d9NTtgm2tC6rctJGkSKU4iarxrUCIk0vIEljAHeP0"
    "UYBhRh9cPAW7L0ikzhRYepdb96BXVJ+c9Quet9jPvp8cRGFITKdB5T2fPIzitajA/GdyFxMW8l0zUMzT+X7SP8G2C54mVzh0"
    "eQM9k6WDNlXoHpz4QZ6HTvKOXrgwP1LPUQ5qFILu/1zBg4ewPNySxgFUPSDrJZbrw4d518Zzt4oFVGOfdwEDBPsVHAIdB3sn"
    "Rc4c91A3lcl5Ks+ujC3g3SoTFp0B+LeGsxLvv+z2zPtu9yWEiPBd8wQR0FcIB6aPaGzjNRmubAABsOJ7CZzUpCeyObs9FOP9"
    "rsBmZlFhYwlX9Z7Z7WIZYswOICh2JxCgEwf3sE70TCqtU96ewGDqtSNDBNSXiBs5mD8AZ5vlTMsf7e5uf/yV7m5kzOREf+Ci"
    "H130xEU/SEf7T3/lu9w3H+cu91Obu9x3n/Yu99Ofcpf7rnqX+/0T3uV+l/jSb5V3uT823OW+ze9yf3zyu9wfN+Qu94tb9OT2"
    "4YPwGKP1OHZ4McZ/4G49ml7G9hl3q16sk8KlgtKgn4snJ7/DuNxczfpHf7VH37f3bMfBCXkOZtI8oflBBXRO4K/MDviz6mod"
    "8odJ7sRjBddp4VogeyKlLBBUP66XNUrqhaf5c4u9SKJgnbL0sxM/xcTWgagigyI3y5DUbrJ//IO9YQAIbPhszI2A7cV0oHcu"
    "gcty4igIFnbcJ21DoGfQDcnQry6YY39P/r6j4lwsehulXrNUTCTEjiDetDL0C7cxJnGyCZPws8tchR64NNQqTsxfXUT14mnI"
    "JfN7FxX5iGu5T3YrlQ25sqGsbLxVHsLgAaMf22f72OxKtx3i/uzDh5Xb6z4AJvoXt/sGPLOCM1vckyJlh7391y74Lgt76BuX"
    "x4IGY1FiKw0k1LzMKI9ycHhY4aiqE6HJY9GUY9HIRFDDTgFwKrsjSambN+iV+hECb24YqXgUyroRO344sVT1GI8FlvnE9cIn"
    "G4WA5dCF2f2GXNMgSmaGhM07g2mPx46zz8i5oRzrfmG0EGOTjoyga3VUzP/V7oMLd8Vz4qhOwsfqjORe9lCKSpD7u4JsWdx+"
    "OGZL8eHDt72iy+KU3Y4xFaKQSGB3LqGC2DI/lrcMeMzknm+pt0oSNkmxI7s7Tti3PckTpth/BHXlGJIfrVsWorLSs4hC+XF6"
    "zoNcVnqmzhzL3RbYtkrXA7nrgbLrgRi0HCdT6p30UPLASqAokb86GNphKoWhGK1SAFF233lt9Arcf15N6sSJKr97rIOczvJg"
    "SB8+kO2/CLort/uksJfYdoIzCPngnDNVOOf8vRv3Gvxz3nHlmSKuJquOUcH7pvDhUVzMa5nXYjK6GkMA7oz276y53PjbJtQA"
    "OXTtvIUb5s0i7gz2CENqkbmL2dyFZc+ldAfSs6lpxpJ1d+fzpTqGZujLh2WPul8X/lcPCw5Yy9RaNYqr83N88r9VRlIt0hXF"
    "kV7Lt7LT5iNCDt4jvy1FYr0KvG8FfSBF+O1RNAnXOsJ+71s0FAZbRZMHLyAcJ+eD/PnK7f5QPog5uaGMAyc48BCzCLrflmMT"
    "+BDRq7Jbfu/aPRADbvDOfAeuuYBgEUMwW+V3OZT8c7LIFopZE7wfCeNL2D+4grD9Tq8hzHsmHH+SzyHsdXtS5Fb+nESv8F/w"
    "z7I821rtwT04TPAe9dXFd6uoUJJ1u/6pBR6rfXdergteUgBAIPFH+DydW3YQmKBC6tvBPXGDpPp6iyiF+FH6eHV+j97yyM8/"
    "iFHouWnow8lwOhgPJ/dKl1JyJ7Xu8Zu1naa2c3JIjHm7l0uIG2c5QZRg18oggCqxBn9y9PzZ3Pqv65924ijA83/aAY7Tf3YI"
    "uPN/8muwrq3OO/pwdX5PUrv6cjbTF/riHtVI/NLzvHvElcBY0zr6YHX+ld43OjTW1t7aR2CiHeA9moASO0z2Ehz73r2iAwMY"
    "q2SFrHWG0LWxOu/Exwu7qyHyp2/07v3z66fUyLkDAhL8w9POMzrp//3K9U+/tpBYqUXkXhRE/5gFkUu9+e8OhOvjuBJ6reTE"
    "ofcXkhOTMecyodj7C8uFbe+jyIVjr4VcOPI+qVxYhunTyYX5ICW5cOJ9Orlw4uWnuuOp5MKh1ywXdjwhFyYb45PKhUPvT5UL"
    "A0yBN7cMzRjv6fqeblho7c0vYx9ItwnvbuDHhwRUClMTtgNiBx37iECX0rzMnfaZ1iH53Tkj3ve43z7Tyt30ST77TOtFlHZy"
    "537cbZ9pPQpd3gb13Gda3FWf8NpHa2OR/JhQF0jt0J8WClLTeoaTpJOe2CF8Yum7E8W0dieNLHScmtY35IiPWeljKC0nFStQ"
    "93ym9dBOccdPeMKLKBVp4OifplN/fXnGgvnvEz778ixiwmdR/3SmdQD/WDTCi2ndh39YaMr7QWBaz+FX534Q8MTwQiSGFxay"
    "XRd8S5rWfdftwC8L0bdWmvqa/GYZIIZfpab1C06gFDEgtl5EFnJOoijBpnVA/rXQekVEvNYP5F8LOeCHJTCtA/IvdegMr8Iu"
    "LDv7aSE4jgEBrVf0h4U8P8CH/h8Y8Dwx31gPLGR9B389h7++gb+O4K9X8Ncj+OtX+OuXB9Zb5NoX1GPfG+twHbr2BVSL2I+j"
    "NU7or5+wG/LfRyfrmP18HPv0xyEYK8LPvMXDkyhOabO0TdogbYw2Q1ug1aWaz/2Q1CPVSC1SidQhVUgN6y1aRmF6wsH/FnSW"
    "YgIVXsTs53NwgWMh6/4q9gPyDanfrsFftvXtGpDAur8+XgO3aR3iVYpJ1GtkvXTSiP56EZ3yxIfYoT/lrvk4vwVch75pt7RT"
    "uUvaI+2Q9kb7oX3Q5q23DEt+oVhLfnd+IbhLM55DxyKHfPEswHyRAx8khN/pQ+zYwMW+ivGpH62TDk2wUIjPU575Ap+nIgNq"
    "UQBEHQoC1KAZpDxNhNIMKlGcwQXlWRapwJKhxpNoLbcPn7Q8zSDFaSJp3w/XqTwGmsB6YJm0C5YBtQ6xE4WuVIsm0Fo8k9Ti"
    "GfbStOylhVZL01otLZRGrn1hWkcRQfMzjN8/wTZxEfvTe9h4cZI+tC9eej9h/N7UUHISnZExElpEZuoLnRCnx1G8tFPTWi6/"
    "ct2vLkhj9nvTArVIi6jkrJem9Zz8a0Fk5Qi2+CH510IrO0nOoth9FUdLIC2PiDWB3eHpFsLLVXpByfVznCTgCQ0IeIwTCHlD"
    "o6RABF3YD6LApZaJEnaMO/ap7Qc0bh41rfSjsFAYDNOTDs3DvNfDStEXkShDqoiChd6r4AEhOziJElzsFZL5yVTOfxGR7I5D"
    "kllHcq4YU4fFvbIQnMPmZRqv8TO4r5jWUQx0ybODhKc8ht8WgtOZpcAZeSgGDie1aekd+Jd+JaZFzu+MpImw1eR0uR8ElakD"
    "RR5FAZ4MRci1E44uclqsGBLn6EzxmGKwhUL71D+m5i/WC/HbQvSZ8yhamdYh+dk5ilYWggOLJD6Hk0sk/cBTfmAJD6OzkCXB"
    "T5r4gF62aTL94C0e2fExTllWGnXoN889jNaxg/Nc+k1z7wOUheowLaUmSKFCK6wQb2llH/NFfGUf484lJGRsr77KMx/DdwcS"
    "LATO66WsZ7bIgQmWcgitoDmwHFKOIDE0F8yjX+FYKvA6Oks64JNsRQr8tl6ujiLIfxhHKzc6C1m5b9fLFYyHAM/z5PJPgZdU"
    "FSYZHO9eR2ekTwlnOWqJrB8kXKPm7iLrEfkkSMgMy0XWAUuAXXASnXGaE65N6/AkOmM8YwdSLHTiu1gu8cR3cbGEx7hNzmrS"
    "75crevvn6R2ewAsckLu77YepKJInWQicweWDcX2YF/sUwy/TOgTjJ5pIWS6aTNkulgF+OH708ZlpPfOTtAM/LQQm8DTxm9h3"
    "WWIS+HCAHsI/7IveDYAg5F+Zhf6IouVT4qTS+jWKlh3ymyWHPC2kCS/XKUt5CYtKbWhfE7mW9Zp8dF7TyAY06xlYpfIc+KBD"
    "YHhCXbp3YChWlrEpJOwtBAB8yUIoXoIQ0nxz6vYlV+Lo1O1zR+HwW3YcDt/CWzj5IB7BeTH69RaRq6nvkJZVJeBLeAgvfMje"
    "vyFD9gxe/pbLviVnLumQuQJH+U/wCi4+qTNw8UlcgL/N0B8kloUJ8j87MHVd0xCznTN1PEDExx78SKMoSP0VKZGxwHNMtroO"
    "aURX4ADE8wH5EGr6q1RM/RLHx/iQnqMJXNlJwisSHe8LPUNOsjIvqcYuE9dmyPWYoEoWxXmeeCD5Mey6Hrkgf4F79CrMrsFE"
    "8vfjmt+tOn6YpHYQYPeL/MYviQdWBdkWt0G4l/bZLfbu3W+c7iVzVW2K5KyHvnO6uXP9yxgH4KaD8KmBl/XgoTws2GhwXVXc"
    "JyJKxP798KFbDCX1hsku38IzTH9Jz3uIGgVS617Wu0flBJdUEGL+SGw2TkFu6gs1VyKFyBUFmAIAma5u15d1CITvcJ+LJ0AU"
    "ZnFd77yg1Xvb65WFDv98w6f7bcfKi3ZS+z2GGzSTN3Qir2PBVf0pL9LtWdRO1Pof7+orJ4rxV3kj/X/27sV5/zLQJNAGiaHT"
    "p0Lh11GUQihg6VM9HJoP8R1gPI3DyZuqH8+hKFM7IKlHMiIRjkRUnUujzJ2w5/nsSYhY5tBF79OnnlwC2L/DO53HCINwF7xS"
    "dV0PovVIzX/4cOJ1e+jYIzOIYiLfWXokHA7fEVDi8v2ae7FHS09pcsS30n6/C3rlpJF8Wx2XtxW19ZFNR/y5pE6C7Dl4k49P"
    "fQfv779fowhy6TPGhw+pJ4dpo083bCpYE3KUWl/hPt9yyCOieLNB4fyXfvElk03zLyxODvJ7JIQ5BBYS4jMUzSHuMw1Gj5J5"
    "yFYCOZBOns+COaNlXbC0oUAKJfcubH8nWZVc7mNmFSQmx4yze79QlR6bBt20va4U0IMAIwXN5ApOEJwQsYpRuSKHe0O9pFyP"
    "vSs21iLz3a3Uoc+OUBA5LF5AOTKAWJksu2f3o7Br0aPGObFDYCGqyiHxPgkZ8OHDhdvrK2M0EIxsbCztXX4hbBa+6MZSBAWc"
    "gmOtYxKzZfN+4yVIP/NUssbhpNoBUyeOCahoDRX18RKCUjFazsG8DPEZCaxlYhQFLv2ZwuN1wVwKuRivqF5bAJ0o4nNw8Ki8"
    "Nmvungl1d4divQEKMkkFIGJxLkJMZa6aBAaBohJnNj58SLoFi48S9MWVVgIf1wGvkyDHzcBzOArwh73LL+AtrjDCu3cB0hJ4"
    "gmdqhDBsmN57xKq1H+Oi0VE9pGTGGBZ683g/7icpMF1L755HH+qdHmK/AvFrLX65VP7/zptfMi7KLOIPYZ7eBd21R15/MnTq"
    "zXnsnvnXeWQfdAFPBlGUgmiCO1J5yPyomNalwrtKZkEk5RAH5iV9Df6JKElboNutr847moVoOonuZFrADqX0ZRdSqRMcaOSE"
    "CrVo+EboC5+n/SVEO+Nl0AnwvwdyPs+hfjgasp7UVuUP5Zbe141RDJIieAv/ieqjW2NNjOA1fea2RAIfarshIukBXiojEvkA"
    "H2wsRofUttyTdo0SB4evYSYuqZ67dUmS+vDURNXTM4tpaxSySFJGbulk9FIWH3nkeeDC1gJdfriyE/2AYiMkDfAgjY6PA/zU"
    "icJPjgsZldCYl2m0elBY8vKSUo2HrIIIGZHjwFYAeVTLNthyKqpAg4CM8LICsJbcNoKTNYCaNVzegdeGlk7jvtE6fOdUf8BN"
    "fyFoSqEnUFTqE+/Gxc6457sHNYWFZ7wi9vpBQx2aq6pR3RzVagTVVJUfA/5uqExxvFC5uCryPBQXhlKDusIMqmIV0lttFQZL"
    "sQoLN1BbieWXqznVkjyLr5CiMbF4Aovy8BuK4lJuGd5XjRU50KoGJPIjynPyw7H65xIcNLV/npf4RV3iIiufFooFFjtfSXNL"
    "y1VDflWlqpRYVapMlGuK0eya+VJTbhXLIFWq4R64UKtIHlgqC+fYtIHKJUvUrbgY6sJiQRx1mxXUKeXncwACT/NS0MZL+ObI"
    "kVno2F7xxGN7RUZPI0LQs7BAS0gpmq2iIlxuXluHF1BXe9zYn6ir6JjPkVy+QH8OFAVoO2IeWfOqkqLrcuHHdW2XgK0cUIXC"
    "+VqUMEMuVN2nOUNYGlaY9mncAVaMHdWVQvyspmnfkEBVRYyXq5BAVpunnRbLJ78GSlqsCKt6dmjJfI5qRiQ1yMflsmciQcmM"
    "PmWmk2WeRDgCFOTkbkBTrudELBXmbFW70lck2tfMKF9uIMQK6ivjkhRi+TJZx57t4D55HeiIz6mmZT3FjUNV1ShWndCq1VuI"
    "qu6gWHdM67IpUFUYFysMJDgP6mtNirUMGcSGatNiNZ1Uy5Bz4q84Ny1QooAH/WQp2K2dZk+mlC0go3AVNAsuNxC4LEPngvWW"
    "dyBittbsK9/jOvlW3wfU3B/PrtlFdTeeAqUq32o4icqQTy6BSQ5bho4Z4VRfbGR6Ke6cWn8yUhMeMRGD8kQw2lUAoD8qgJDf"
    "adk95zwQrQ8LrQ9LrRuVto2allmXWYYO80tUE1Yu3eLNTOuPSp3z2SiswQSEHEs/ZDdHPnw+ADF0FwBIWDssDei73PyYCVHk"
    "5oxye4a0InLl6aRaeVKuzJcTJryMvHlFBapnCKTBEBqjePKy1Ap25+lhGtuJJCdIiK5WpaX2dHdbApihZE1cK9R3eBxjHPZH"
    "tDb9GG7sTCo8G9GO/NCLGob1/oL3AT9b9gBFefvwelrffhSD1JV3wb5a9sJK845c+Irru4qxy/uBny07gaK8B44Ym7EAapTI"
    "+eZJk+r2gMIcePNL8apUogTwmIWJ0Oc8MS2DyNmWpjWEH0vXtMbwIzg2rSn8OA9MCyxzyJmCYztwzcuRZlpfYsdzvZGFdA2+"
    "XN2z8chCBvmyJ97A1Sw0IF9jjCeLiYWG5GswdAezmYVG5EvXFrOpbqEx+dJGs/F4ZqEJ/RpOpqOJhab0azzyhmMLzdjXEA8W"
    "FpoRWDTDcAyD0EWMQwagp3muNxQAOp6DJxzAxcKb5ABOx9izHQ7g0HbxVOMAGoYzGmEOoD62B0ObA6iPptrA5QDq4/FoMOQA"
    "6sPRwHAFgCMD62Ny61tiDt/EwwARhQ87nuMsOHzuzJvNXA7fAnvGeMjhswd4PBhx+KZDx9HHHL7xyB5oLodv6E4czePwDbyx"
    "Acr+FL7BeDTQhxw+3TawNrIyFGO+vh72DM/g4HkYG9jg4HnYsR2bg+c59sgecfC86USf6Bw87A2HwyEHz3WMsTHm4C1muqM7"
    "HDxqaMbBm3i6q4vpG440W7PhTkT2LYfQ8yag2MUg9LDrjnII3YmdQ+gu7ImYQG8xMwYOh9CbTQb5BGJ7NNUcDqFjDHUth9Ae"
    "DPMJnDiGS74ohAN9qE2sDNlEmUoAuAANaz6F3sARGOi5eDyVptAdDN0cwIVnDAWAoxnWFmIKZ5OJlk/hcDTQZgJAY6hpWAA4"
    "HYxg9fkU6rY2sDJ0gQNwOCMW2cFTCcKZM8in0NMkCF2sDcUm9mzH0QUOYnsx0KYcQsee2tqQQ2jrY0MTm3g6GroShPrAk6bQ"
    "MDQN9kiK7UDewzaHz3G8hadz+GYzb4yHHL4RxrYrlthw3eHCE0RmuJjaYok1dzacTgWR8SbjMRZ7WB/h0Uzs4cEQw25ne3ho"
    "eAY8fTsXdpgTQex5Aj7Psz0siODIG3iOIIITPPVmAj7DHWDM4dPGi7ErNok2nekLQ8CHJ8OZJuAbjfBkJOAbD/F4IOCbDgbD"
    "IZz/7y/y6Zvl4GHNM3LwFjYee4LETFyXAMto9HThelMBHrZHeCbAM6ZDQGMG3mA8s3VBoyej2VSApzlDG6aWg2d4wxlcP+FV"
    "lE2f541z+NwFlqZv4bkL+KLwzQbOyBMbZKzZI0ALCt9gMTU8sbzGaDyATcdItDvE7lRMHx5qtiemDw/sqVhefWKMRkPC1bj+"
    "ccQhBCoozSCewBeF0Jm4Rg6hPVoM8xmc6lMnn8HxYDz2xCk39IZjOCsZkR4MpkBLGZGeDDR7IIi0bmA4HRmEWF8MF1aGTv0o"
    "wClfY0CzHEIXz/I5dF13nEPoDBfSHNqT6SKfw+li5ORzOHEGNhbHyNg1pq4gMqOFoS/EOTx0dHc24hAaWNfGcIys1jHoHTAI"
    "bW+UQ+gN8DSfQzxzSR6F0J0uhgAvhdDRpmRGKYT2dDTyJhzC2WAwwOIgnmDDcMQmHi8M3Z5yCEdT3ZlOOISDhTYZwyp7a+ck"
    "Ac16CqLrDSUQbRlEb+Rq+SR6mr3IlxlPJzNPMDPubDjGnqCDmjF2B4IOGrpji7N4OtZnU4GIE023J2ISh7Y2HAKdWUEEJAGf"
    "4eV02sETL6fTC8cDGs7gm9nTnA56w4kBi8XotDOczsRGdhfGZCI28gLr05HgZWauPoHziMI3HejToaAzI02bGDCFMSj8i5NO"
    "l5gFDw/xWGIWXJgHfo7YQ5gHftJN9Kk4R7zhwMt5Lazr7nAqwUcObQafpxsDQaenU30wEEs8dLSRDoQmgdhfHMCpZ8OyMQB1"
    "bwTLxnDQwFNPMIPOwh1hnQM4G9qDRb6PwSxfHMXDyWiUc6uDwVAfjXJKY8wGgpvRPH1i5AeJoY31CeFWbUGpZ57t5azCwBt6"
    "YgLxCE+AorE9orsjdyHgc+yBLQ668WJiAOvK4FuMRmOBgIPJUB8JSq17xoxMGTv29KkhJlAbaBMduOk//NDJ97CdH8Te0Bt6"
    "gtfCQzwERpbBN3SHQHEZldFt3RZUBjjDic3hGxkjYyQo9QAGPebwGRODzhiFb6pP9Zzbn2kzDehgiNdpnPMKRRBH8EeawhGQ"
    "XAlEiZ+GPwLEAfwpgCjO4qEGfwSIY8rMMhDJ2ASINmNYkxQuWxKAswKAQwHgBI9yZsYduwNX4KA9tY2ZYBYm04k+dgSAk9Fg"
    "iAWAw6GWbxJjZoyM/ELi6DMJQEeztRlcFbkOoVqRB6IVqgTZVD2DSq5JoE6VLKQoqTZkjQ6i2ZmJB+CXzGEjSIWowEsW1dCI"
    "MN+sU2J7QdoR4hkYzSW7mfZHWsamVySBUIVNskgzSNqgkDYgacNC2pCkjbRiF5A2LqSNSdqkkDYhadNC2pSkzQppM5pWGMaM"
    "jENx6edTy2QR/FPII6iMoSorBvrseZ4sKWgWlvOWx8WOBhuE5bzcpFjNYIIpeCp5DC8lQiL9syRmzJ/LNelBSCXzkytzIaJc"
    "ezBRCg1ztS25gami+zGXOm4WoVa3hiY2BkFyvo5S/Kp8T2iVHdFem27jO45CNKTUmdniKaKiO6OqO1KJMWuUaLas/7jNM4qy"
    "fuEdoP3zU/UdUVV3WKw7kh5uiq+KZdqoUqkpCx6HueBxsFHwOKk+BpW1bNRTVqjGCEpV5aZ93Vrlm/LwxhW5qhdENrUF27rX"
    "vOrjmlezGspGF6TQwv1NT4KbQGCei7ZUOoLDj1eRm84pBSjMfqlpmoY6hZ+gRAs+lDTqRUmfog783xigjtbXRr2Wr2lt9AhL"
    "yisafabpaPz9hmivkIOaK+tKZanen1S+w57suIZLuaw4Iq7pnVX90KJUkznxj09IE211Y/IKDQoxBfXIgiYMyaEKe3UqMHkP"
    "jXovZUBk/BL6slsSs203Q3ZdGKc8FrVEHnqrd+KCTk0tmlW1uqUVLpzjDWrHm3dRRR23ic/Y7rDfQmVDvRk2HpdqBkGJ4FUF"
    "7AKCZwiig9ZdPwZJ7dgLvuKGPVR0HjfuSfDIUMPFMfc+cNmalGUInB5ciTqRtvKX9p1JVTuFmsb35O1okEJTvljgU1KVzco6"
    "6mpUHeFSRB0nBxkxOb+8toOxhk7Bo/wC7LCpifufTno4OE8l1ZL8KnSjlzGT9IAp9W9DPnPeCVxNjlfnJFoEpxgd9r++3gNG"
    "CpgoKLZnKIvsRoyvkaJmaBWtYBK20/HhOPYXmwvq9aB5Js6Dov2WUZ4JA5yfGiMY6Kh2LqbgpRQK7o1v6mTIJ9u1rjNICikA"
    "5iWT+HseETyXgAahO7uCg/y4T+UORChXLceOKFKQqlRSUV21pCGVZDqbA3XJgVSSXZiH6pJDqSSjSSOtZjzygEjJsbrkWCrJ"
    "ZAgTdcmJVJKt6VRdciqVZKf4TF1yJpVk6DVTr9BMWiKupCSuDPVaSrkEMkMdgnV7S/+864edJD5elGSSSA7a3pkO/9XrbWB7"
    "eG2KF9u1Pxn/q9drFJvSGSfIbYxGiP+/P530Gg4xXnvaVBu0Fs7TRg5IJZrZpKys0O1F5Kzf+rAlPT3fULVGoTBnIoBdDvET"
    "xqCE4CYuoPzGY3vpg7NKPzzBMfgkkjkZuaAk2Ad11bM6ddUajeAsQ0eiSk7RG/SR5Qt4gQHfxnCZsv47WZm21HSmBj85/Qaf"
    "0ktq+1PkritlCEOrhENZsmRgQ/lEZUnOLN5QOwYV714eBylTz8Y3FBeX9ASvbOreq9UKZBl6VIfTDbYiBEOxW7TqNfIbKEfa"
    "6zA7BHBfhsHFT1X7G2luZKPHZUEYo4YhWbYAQyqkgkT1bCNXD45bABIctwBEKqQAhFkbKTBspIGpdWz7CXYPBYM1AM6wjnsy"
    "BIMlSWdl7mrY4+LbkbqA0bsGY6KmnQZGQe4xlh9dd7LRbKk2X/VewEuoxA7KgjSrRGTrnlkU1mCNHSpLlnrcpPpfpMS1pYqS"
    "ldpi0urXvbMXHxKvwfLgT7L42ngfaoJ2t7uUYtn/IhZqCsRpD2gZo3ax92hGovfiIQ5+Dlohz3uhXwA/jU1I0wBVU51asJoq"
    "VeFymnQxhKlLHUooy9YhgrLwhuVXTk6FkOxuT9S4+rTkWK7WCgNo0YlcbxMWbIKwuV4DiM0VVTA2YYRsXrUZJwqlN2NFoXgz"
    "XtRPVxk3djfRakQNVnRcqNgKOVjZSaHmJvTYDOeGmk2AbqiqhLQJSQq2a5uxpFh8M5oUyzfjSdO8lTEF4iw1yJWIwrhQbKNf"
    "bTCFFR0XKrbCFFZ2Uqi5CVM2w7mhZhOgG6oqIW3CFFayLaYUi2/GlGL5ZkxpmrcypuxmjdmIJSVFo1b4AQUneZ1NmNEEVVOd"
    "WrCaKlXhasKD3BB1MxJIZTdjgFS4efnVk1Ne+KsZyba6rsyUTHK768pUyZdvvK7Uw7qxZhOwmyqrod3OknjjVUVVZ+NFRVWp"
    "3TVFNZVlJFILjoqeddbgDxYsXYW4ovmMafvoQa/ixWeJGX2W2HAWtX73UHXBXlbqDy1JD1tqqvSQWJInFAQIJ631divvI1pf"
    "G/a21FlRNKKPe1d+K91Cb03ceVrcYGrRIuecd0AJWrkeIeoa34QM+WVENFI/S7UXAS4M2Hi9r50cfuHcYWqgav3EqBveNC38"
    "zs4aaEAc1Z2ZX4TaXGpqZ0Tin3eYFFa7fl5qm980NdLVJG+mfoIaLgX8FtCGo68ntjnzuAulpbUbCG1d8xupbM6X583UT1MD"
    "R5yzwBuZ2tpZyi0Htp4iqFo/P+qGN00O51RZA/XTouQUZdbwZGcTkp20JictubyJUjFuO14mA4sLP7yxYzS2lX8rVB4yrinx"
    "l2K9rpeHus5F/RuyPn8HPubvxJj8/XiMvwu78Pmc/5/VQb71iRwQTyNOk/ZEnWpFnf4C14V6qPbpr9TO26zCXedNNFd1Bi2a"
    "gVqLpicbSdaoQhObb/dCVi0sGAIJZ5aOzYwVi8mpn4LfG9noXBEqhCoRJesFK97KlkMddaTe2C1DL3eK0SKM0tjIuC08+EVy"
    "QA/uWdFiSlfMjihb40+Wa7O19gL7WTpSVgujWtEEtapJgz5UvuPe30bRuI2icRtF4zaKhlpD89Nr+f4ZCr0ly/PyS5BKF7zu"
    "tag2XoFi2QuK4H/reCVoGS38AENkYoh6XfLwcBvD5AbHMClam8glC5YmtaOvtTIRFsaZ2m9DzTDL+/Wa4quQOKPF6HF1OzlD"
    "99t51SdW/mcbnNmjVnyZc4Kd9yWEr7ifp2WetFSn/1Ss3p/GcJV5JjaFjbYBhTlsYSLAyj/e7NyJlXzIZ70l1DsyiI1s1g29"
    "QO3OwfA9VtxgleAddfEpihFDKp4PVFw2W82NlhoyPm0svA3rDsNOCpE56FCTokHlbzWUSid+/Vv53dvRShJde4ALlfnr7pF0"
    "MuQvSTics4a4H5VIMlvCBy70l9Epflpp5ybsweYLxbYj3dF5Ss4GZOj5bmGEY3zq47NN2/yzD9UlAiZX2HPmACSn8DW3AV5O"
    "zm6xU0vL3dZTxUy97U7s0JXFnDkWwRBfbC0h3hSmtp1QUWz8ae7vU+GtjjoT7aiZt52BBBHpap2qetKuIPRtOsC2Fv0o3PwJ"
    "z6okvpiUZXC3WbUyeriRR2lpcj/O3Gbo1bZeBQSObgaB0XTqESyXc/PYX5PthrKD7Nuo973cGN+MOTHbFC9b4SYry9BjCBlf"
    "DkRWkSoQXzz5ihefEhh7tuRR6eoec67LGVeGnqpJS4U4bhbBFOloRSpbLrZBRlNqTaz1ZjJ/zJxggxsiC9lxHJ29ZMubx0oT"
    "054LBcoNlgQ3uoL5KUm5aoaYb2uVmwJ4kavte5ecT4s/z677aLpiNMQKfkiOKASKbMNIlYV6UntK+V45n4Umpt5RqkK9sp8M"
    "hXxN8aRVrsUM75ViuXJZpXBKDYb6Bbum86qYrVywVtRWLviRHL80jlghkmseNqlQGnsbVzGfyicLdV9oXiqlzArnhnKrLFsG"
    "XIVPhWKbZrlQuOVEF+oU57rgdaYdP1njYFLVIZ3RYuxm1WoVy7dZMEUNNsGLmh1erLFpjxdL5/zea29u/aPT6XQCv7/as9dp"
    "5ETLVYBTvEdFv4jkuv5pf7Xn2Ilju5hKmFn+HptWlLcCBG4RnRcaIBnLdZD6hdpSZl26qjUKThpjvBdGLlbA4NqpnYIYaM/z"
    "gxTHUCRJY9vnpeQyhY+9FPRYOl930liUhJ5ESfFRKUnBgnnegw1YBIvB7OMYu41FSNJ52lgGMhd23JB/bDc2QK7HpAS9uTWX"
    "qbaSnhTmjDeCg0DKz2dKkY9ttzjtkEImE/46Ucw9ryMtAKR8ncZfpyedS1KBtC2ObLMDxm33SE72Dwv9sYPQpP7yriQoFTFd"
    "fjfelhUSc7MdM8Q8MWlMZ0urejKlbuOJXFRxi+5LItKCrpK4bmZ8TAc4CNqMq+q2iBdTqKwoXyw/xnQV1POKANW9QJabqGHd"
    "C9P8KSRke7ocI0j9ZnCV5XaiYL0Mj3IdvYKr5RZuk2OInPkpEKUm7sA14kA1jMCNWeE09lfbhpnhPkzhLGP7ud22unaqki/o"
    "JntcvWiPO+Memxqur4ybaoGq9FrelrRdkQZ9BNJM9+pjJlzYZbNywcRf6cACZaRXkR+mterafOZeY5iHXAFXtB+TjKcVDV2y"
    "M+sNmaI4fbpJdFgX6UElYJSUQYLIhtE+LYsXCb09io6PA/xgnaaRSqG+DWV9UlupPjpLW++uhT7qNAHKsuGCgGCk/evmvj/R"
    "a4d56YfgMeEbmU9gMqhD7oP/L6DjxjJeCe/6H0dqvKVUeJOQebPUuLUcOCcSa/IY0krykN9Jn22nIyhXvdUJvH6dwK1lSNen"
    "REiero/9EHo/ilatubGao1Rq7UGUptHy6g06SWJaFRnKHsQi6q/22D4hn9KdfG9vdU7S9iTS0CnHJCp6Lhn1UCEw0aDXE1f5"
    "DD0ovVQeKKJFFsdQXoYCo7HVPb2Zn2n1yt6a6SnDLAe/uh5AG6dMmiEFKO05xeubMjJd8GeLKWvaUtst08YNtd0Qsgz9uJsm"
    "D9VvuaHPa7scpLsLyrYclPTEqnV4HKncBLEUa6xofNjm+lRr9yDpXxoVjcuBHDj3qirApcIq79z1pT935S/0WXntvtFesJm2"
    "VTtlf8ZuPY/C9GTLC+cVxHOKAIRbmshuv8PpQH/BdvzXHucxjffZitrSO0oDdXftix+J7unSjo/9MA/hRw9CdIbx+4f2RTWk"
    "pAL2Jg2WTdTZTvGO4pBGkxLiSuPwzxFJ16rJF4CqF2DX63WrRC6q9bmR58C1BadcAklrRN4lJXpVjU/UVo/yAttxYxcXhNxc"
    "oYcFEQUubEVQacZPt7vspv4Sv/Kd9/hqDRUfzyhw38ieE9DH1lNDaeTaLaJ8bO0bc6Z0FvKkWamSqoC2EI6xgs0KlQWF0jrB"
    "WaGlem6eFsv3VUXZu1SyTjbWzsfHWNOUqpiSQmSxm23TlRrV19Z4ccwZ+kEtp2gSCm57uy9GsjyJYv+PKEwhticnJTpTMZfG"
    "y210qtOcw36K49R35IZ4vcq+VzYl7lZZhn5SYr8Q+3Tyv1hEUHm21MUgulOpsxa+cLjWZUu+qdIBfZMY3OCDL8vQN38erbk+"
    "8qGkGDeEYNQWzdB33vwyjaKAnLSf1DQHTjTS79Pdoy42+6X6K7lmQHXvLWKUL3f1unANDx3X+KSwzZnyCcSFWYZ+/5NM11qI"
    "BLnhWPXHlhYC+BiH7rUrPV3h1thyjsqS6qbXgspRz1/qa4WoTAflxt4XN94YQKnievQ5lGcW7I5vb5j1zDZor5LhF1BGxbTK"
    "Eeb5XisMbB0muM17XsnVIJ9eIfFo9jvRkgrIhl+eD/yJyqWfXh1F3fJQT7qlewO0TNUCioNaxdFxjJOEcBe5W0CuQ7ewE98p"
    "14G2XtjLssPEa79PQz+kyCbHjLtavf8idobKdVgQ2WnRwqF42NaUbbI6qalSMvBhnkU2VOL+R3gt1bbS+kZioVVE05u8u7Ei"
    "De7dKlJTCrXMTgt6LZcdaprVLhwM0fXhrabRCtRUc5U78ONaNNaG11fBwrM09opaN4oMFdqvK5UhoMjNd78ky6eglSehwp1Y"
    "5+PK0M+lK33tWXlFbmUbuhvap1zVr939usJ11N+xy+4H4G2odMZJngkKF2Rudit05GN8Wl4nECpnKCTO7VU5N1XFL7RPn1bt"
    "fjOUnqyXi9D2g+SgPd9dVFzN6blo7EW+wLIheUtUKq/DDeXAWr3Bbu27eazy17/tOyp9Dq0uR8VfjeTiuLITJZfJo9qNVeQm"
    "Kv6D1TYn5bIcV+r9PbVwivv5KNx+cpfCMNUJTp8W12bDgpcrtafYw/YUe6xUSFCXnREX8UGU4CJtURPvj36wjBrPlc9I+1ua"
    "UtURkaFfvTlZddAVKh0PsmGO2rPOdFK6kTTVVziPER5n6uuV+BhZSUQI0NX86yBhPJt0UnE/QaolLOPAMGcuHqjAmzYzH9z7"
    "28uV7fjpBWi4cn9wEU3airEqefYQTCDKbauLy6cMSFJQsZHDOQy1f/U4w/QpZ4njtib6vIYpEWcjzxyVPOVJT7uLeqBLl/p+"
    "YVIYqz0AVluhAieTd7XamzoMw6FjB/SqOdqO7RZjPlC8fhni+Qt9L7wAbXaZeufjX3L/hAtr0/UNtbqRXv0umSG8upLbPF/F"
    "74Nae4mGE2V4pdyW5ORrLd6J+AyalwH2UtMCVwwWimV+jcwgFyvRgZJSGVoE69i0puT+qTp4pWigtS+reRmjVy8E5ttyoElO"
    "MU2rr4r0wQ+BeoagCNd23IDMqMCyFN8HPx9eIUPpan7Jvagx1BoJ+kaHRad7w9oOGpmvYjmVdLVKT1uJXcnkbnF8XNc61Izs"
    "SksRCvKQP78paUmnjsy1vY8XFWOI7KMa6Mav3i2prLjmtqNiPhbBmpl8y4zH6F89ddgpVnzbMJWkmlEbzIpkTyqtjmujWZEK"
    "7N4uQJKCVmur885wdd6ZgiXF6nyboUBsLjla3xZzKcXuazWZcvS8XUMN17e7zXzSGmM5jt8OM1o3HjalNGJe6/m8vMBBEJ1t"
    "MZ+iwrYTyirWzygrsM2UsipsTiXQtpzU2kGxWcVxHMVboakIWNdqUvPAcbvFpa1rc5u5hPLjPHjdDrOoHgXf63kw0d1sQ64/"
    "fLratmObERejkSpGnYf+2zRohb5qC5/FI4XBSHPY1KrfkZ1GPBupR5yh+FOd359TXICaS43rJyuiybWV/slmPRN4+Vvd4Ehq"
    "t8HQ/oRgaLexzG5SLLMMOSf+qk1Int2DRmwbFCFD9mo7TXk7xnZ7P+VJ4FPFpJYmzSXxJird0NV6Sl965D+VtFI+6LT+iB1z"
    "miKYqjaFgKurc2JxrtcU0uFFaLdbdwE2o6K0IMYQrVPwXSMXZEmNWvnkEk7LvZQdqDNcJSe2c/FN7LvVwAb0AVQ+NAQ0Kdct"
    "yoXWGUrO7NQ52fTO0KCIGq3ml7brlqX4tUdV21Ngg5R3C3JVsT9Y+uFPRbvy64rYl6FkJ/HoovjWbGwdRITbmbxqZtqKmFHQ"
    "KtxFcWCrgNFqGncDrPUbXoYOmnhx1fVjo5H4WGVbDu+dAm/K4YyJyH57PwgQW3t1G473lgOti+l1y0/epNi45YjnKgO7+inM"
    "MrS+3ey3m/029vYtfbmNvd3Io7fj9cHf7Q6c/I1Vpzu1g3VZab9imUp8ndSpWpddWJAH1w1GKg0+CsIoXtoBMdMpc74DSX+l"
    "0Vymap8BcChtctsotYHjyE9xiC4+bjTeT8Eftqe9O0rXtg35XrWIJ5k0uWBbfhsH/jOIA7+bH36FE/urxGnfNVI8x7FvmO+v"
    "ujUnvsE2LzUtprT9qxYrjlG9IrRkMQaTYiakBkXceggwvbTj9znlVyEl1Xo/TO0Ywi7kBJ2mP4KJKMfzfM4j621AoixDy9XN"
    "snwvWJNwOdzL2MdhyrSbrxzkLndIo263yXZ8uyfNhZ3gpzs5d6mPSncbpO82SN8nDtInfKwUnC/fCHLRKpBmcQOLkINXJiSf"
    "OFLfR4y8t20UwL9paEYapFJYyVXVjkvxQa4cYWUrVZgbq1B+srqNvvuRo+9e4wn91zl0b16c27/nUbFjBNybcbZkGTq+yXek"
    "2mvL7UXl9qLyd40mfsWj+BNfblrxI4oA6QV7rM+N+75lvz8V+/2upMe6g8fFbc6Ser/LSmeMdUE0hU5lIYwmT+/IXthyc8XD"
    "3DSx+pxVfLSVw1YeF1365i4ElK4FSEMFI0hoICm8zmUlNybc7Q2fAuH55mOGkmzCGg5gxV2QNLrSFFU9LDC1Wtescz8Gu9xf"
    "rgLigU/pW/LWfvRa7EeL2NaofUqaoaqnaoVpbdTrFbBvh/FAnPsihmzRSPOqVCvISNa6zq3F7bVb3G6BhLSdq2Lh5jG1wsP6"
    "Zjasj6JKIy7W17o1Vv4YxspbICRr6KoY2WZcrXCyqaFNK6Wq1IiXTfVuDb6vbPC9BRpCK1fFweaxtMI+dRONa1Ep3ohx6hp/"
    "T8v4LdCjeRDNaNFuPK3Qo8ZqZ9hyCZTVGtGluebfz7XADiijjKfZEmUah7MVxpRmcqfpb40tqnqgvbutPER2sL/EKY6TVmq9"
    "iN+i+R2aCkGL2ggk8bkdk+h1qgoKf74knTp45RpS38gilVzB6RsJclLrqOLZ6SoO9jN0cWsnc2snc2snc2snc2sncx12MvWx"
    "2jcGu/4LBUK7NVj4mxssVKJ1bTAtaGmkcGuBULFAaOV1B56KsB0/beUvemsLhEXVR5r0DFh8MpSiSjF9lR00LxrZ/BqNi20U"
    "EerDw1+DWsoVtGla6rLsqHqjZBDKFdVPxhk6L1otSm6OyM0+CpJKPLLDvIoIC7GD5lMYufjagyIqz4o8zP31KbyWCXvexxZx"
    "GKVtxwI5MPfeEFoQOPpXqgId9tTd0beJ+7OdFWrT3fSGxmCKXHxEJk4VS+UaUGsrdNkq0G7yeTosL+hoyFohThSG2CHqRc61"
    "qGlxnQ1jSDUbDgQFYqKvnGkHnZlKp9IyghjqTHXuSTuqcOS1DpbaOkh7mZRcQ3C3Zg9Y10Uwr6zVtZlolu5eYuE/u71xbZE0"
    "nXUMy/rKPsav8SqK02sNqVnb7W/r5eoogl6fUs9dS/u86Gwuy9DRzTIv+VgxZ5upV30Q2kw6yZ9UgzcNSsGbCkHuS1GSN6OJ"
    "Iuh9Q+haVTSoxvIZeqT07LYV9VrZIcjhr1XHV1aW3BzUm0fi2ZK4e36cKIN8ptHqQUvcDOykJk4oRHB50NZP5c0wVVIbQlz9"
    "Jvi3skmCoLiSUnnJ/OajGtY8XM0vyYNau/e0Lfm0PldebRfoaoMEdRWtNnHjlaKtZKil0rVCVF6uKm8rl8jxvSJp5UVyFkGQ"
    "7JKVdBrj8Dg9MS/PsP2+SV1llKtxMAWAJXb99bKujr1c4JjXoh+sXpLGUXjcrDI4kpXnuOYAwaGjkhb4lS6YWYZebi+TeK/m"
    "RK4PhTbhxJYothE9RMhNnQR8suM4OuNOlIWqu+LUr0fJLEP3N/BrV9r0OVStnHQxUutUovFRfRKF0aMyIBzwR1mGflsVIuW9"
    "DHGb7ULKHp1FCr3lkaS3XCh9EmPccnvQGo+jtVrpb1RQ+hMaNc/FUITpglE2XVDFTqx7+SfObkoXvHLsQ1am+mzPCxblVp9K"
    "meBPe9Ivb342hY3xIwtzqCxZnERW/nEVkJqWH/JZ/7h+1xof8j+naCet3sjro2IXfFHr/fIWVBkqCQrFVuxgE3GTcWZj4a28"
    "3i2rOk3HRWWmLEMvyle67UPA3lRxcCVgXmvplToubE1ulqFXykNVovcsYAP509d7koKxHK+Q0P7Hf5K9+4119+njMyrzqspo"
    "MpQ4cRTQoKm1Ap+lHR/7oYTbfFcY24sveOCPVo5FaSxYz3ZZrIqhRmXXT3cLrGALJcVZNQZrbbyWm0epd9ItzzL07Fbd8Vbd"
    "8Vbd8Vbd8Vbd8Vbd8Vbd8Vbd8VrUHW81HD+ehuMWios3wh/z6zrjpIajdze2J8vQH1tFOLyx97PAvojWLWz/VA7OsyoxFb5n"
    "+PviuP61urZo+VlFXBftxHexqoktHqbrXpi9ILJTPzx+0PK+zYofcsNEEt3RALNEkA50tQ78+YqaHVsZOqbkQQU6ySp7N9Ry"
    "Nz6SDs2Wb95Vhz31+rVVoQqD7L5Dj8E04vZwKKb181a5/ttgshWAH+WVeKeH9TpvYJUHZngwg3ffRmekUIQreSk0F2S97Bqd"
    "6U1LvskD3s46160ey1upIWy7BFv7F9xazXorVW5Y9JdhcCFUpthCnPnpCd0STLWXnFoi+89Vf6CoV9ix+SErdq0YCROM/112"
    "7QPbBUZiw7TkDwS8RB6ltW6u8tYKBNwoW+E00OHFp1FipjNxuF4IyvQgiJz30jGLiqcuohouz5kUVpfSyrrtpN/AXiX+QvK2"
    "qMg7ilbPhVSX9yPlP1BProC9Qlsb1JZuqexNp7J/Krm0/XA35jdnGHep7IcJTjfVVHmOyF9DtouPnXvQaGZUM/Rga5eXrRyu"
    "1PjkGYNPHjv0l2SJHjS8vxWf24a92iaH9CXux52eadLYdt5fq3YNxUOIG6+Oa1fRq+GPVFysR9SHBKnSqo9HRIu+DcT1dgWV"
    "84PdwBQdNTQikwHOXYxl6OmX7MxF64+YKxf4P3to7bD/9bVpD5EMnf1fVUgf9jYjoWIvZTfX7eyTK2tqZ+iHHaQhZAK5fgqv"
    "YFA2yfYT7ObX3QFbkT1DsSQGLBshMoLQlJcMCkD9kbqAQfbwT5/KPGEbEsH0+to9IvPNTKkAMX9CtU/dn/dbf4a+2Ynkyr6s"
    "W9AmBVtSVvgQhyQ523RxVLJFIKQ4SfGqYj5W9FcERapWH5/5MpUjOqV4dUSNVhrUaxo1aKp3mRbWLtDvizVoMLfZ24o1350G"
    "sMauaAzaPCUM00rXP72vDwc11z+FR/b2x+S4zTFp9Ni8E1lpUkD+KfcZULQeKqDJlUWs8h4q2k1k6LtPQDqo6c0zP3zfmrf9"
    "LPYwGVVRGfvKe5i2ertDP/kOzdDvu11c7AV9Ci/aNDIu69qCDnLztfa2y4XcKpooGCAF7G1QQaEntQV+VVS2d/ED0IxyuQZk"
    "wTKzYg1R4F/22BSU/PrcUJ3dFC+LL8bXN32ZQJ7YvBQXytz+0rT29DJ3rVD2/fZm767UXtxurr/+5qKoutkBwQYuOrUX18iV"
    "cS6wwYr8z6c8RHLaqXMho3au0TwhWyFhSdfupuAKz5VEX5wVAfPDzkjwGxV5KedMimVVjpUVs1hyU6+Bj/jNhFrbSKV/2ZVK"
    "U9e+10uNby7VIxC+uiIF+BTb5kbhZ4Z+Fui16W12AkeGQodjtLufIaWQtWQAch5U3TPxR1CGA/XPVhxJ9NoQD7xEbegIhWdw"
    "XoeGhhCfA+6B/qrRB7Z1u785CFEhUs4OEYZqo9BMZEthDk1jNK7L5H3TekDuNmBAeQYE/OQgNMbBuYzI61MDFKzANoCwKgwW"
    "9sXBceGrPvqJFLBj69AmNWExJrmBOAdii/AGCv/6DUipelHK0K/bWTm1tQFqMLMRJ+y05MpFaZbSqbNn2cZa5Zo8VK3iaLkq"
    "m67Cii2Xdui+xskqChMMjqsox07ekmCSv781Jbs1Jbs1Jbs1Jbs1JbsOU7IM3blZbgDbRXXe5mZYthLbKbT1bVT226jsf5af"
    "OUUI85K3uc8zlPnWkczBLGx+iU+JetrSD5/wNzo6JXn8JpMWOqg42CICYsJH8ghQtUW1Dn+fJgWKsabKvkaL6niysN2ovEtc"
    "hbLWq+ZxSYFQUm9SkiGC1MOP8oLJZ7OlD2lJLSdD6bLkPGtLJ85XiL+OFsFa+Gv7LNziCAzkVySyMPBnS9dyJY0CGhW+pCm0"
    "XlKxU4O6D6NLLk5tP2gsKcVH/0yDvf/t46/TVW6IPThRxIv8iwdtv42M/smw4zMNp34brTwPxv3RceQzDnH+d44d/tHx4jML"
    "OP4Zxfb+6Ev3GQYE/0xibG9cur9fYG4IShUv7eDa5aIF5YjCxKv0Kjb7Gm+euiZ78upkqW6s5XnJULhUhGDa7q5eCVVZuCvu"
    "9mrnbnKx3FytzQPTjs9bfznHy9s8scjvu9s+lRSDV251+LFdrXCg3r5yK0KkrrqRcNeA20S0ChWYl3pZ76rFmctOqaKL67aH"
    "kcqTevuhbXyy4gKubTbkTZ4vZVynsiegto5XKACyzFbY8qsktoYksa0tBMbTybIBOrphG7Jb7i61V+C4Iu5V+2kvzdGAyGiL"
    "6nefo9t7VTApaWMplP9rs1UbU1XgSXMzO55uOx4jge/ivILWN5LNpvwDJb+uYhfk2htYAKnBsdae7g+LsIy1uvPmigE7JNfl"
    "dc8+euXJZ+MuHO4+dRSuqmOTVrPWQF2384lSqPpk665ntRelChNeM9KDmqqVSlPVGdKIFI2xR+AhaPnnRBqoU+1QB5QTx0SG"
    "bAFwHolS5+Q+D0Rdpua1d55y6Ne643RTJKfC9MlUJR9rjj3bqz/KcXWjLdZso72OFOqWPDAXTKp3CORdq+XeNozUJ4lauyle"
    "bftItTfFdqek/gqLsYMN5U7RepVBt2sPGPEmu8tCP2nnNujaxvykKSLMjfXGG9mwBeW3ZoO5z/UD4nSm4NqE0T3gwZPEtP7R"
    "6XQ6/dVeGmO8B9FL+qs9RvzIZ+eSlID/9vZW5yRtTyJEnWaXW6MeKgTCGfR690iD2T8sgkfPFDpRtQRyR3fbLfTKM5Qsb/XD"
    "b/XDb/XDb/XDb/XDb0ONXFuoEThX60ONbBuvADkn/qriiKgaB2yrQAwZcpbbW07XC6qVV8WqHE8EAtj2KgpTCr418HYGBFUn"
    "FuULJ1PcZYAd4KDVG95Hu8JccdQF1rcIUvv7TuMFnc7XTbkTgdQ8WC9D5pmu4sOjhXFhHJ39qSv+UZfyxizTInIv2OZqh+E1"
    "eFfATTHNdVSpaOAuPrlj5qb7h3TwbUAgGrCkLeW44gavI190GzxmsVN22Qc87sqNo8x0aK8xABoLPBbq5jHJeBq6vkOtQ86k"
    "RxKnzh1GEsXpdQlPynrq9ZfzqhDlxohIthKLlOQ8H0PV/iMJTlb2sR8CnhxFq5YeW2rwVWrrAXVXc8XmKqIZsqs+sXwmQ8Fy"
    "fgkYTIVIyfuruXop2qEYPBz1ejl/ufgNuGYXe36IX8XRCsfpBXLVGT5OkCeyjnH68izkdR7ixIn9VRrFCVrVlDm8WC6iIEFL"
    "kb+KozRKL1a4f2InUkl0oiiyYplPk0fheoljWBh0vJx3MUpR2Jt/nXb8sIP310uScolFKfMLDR72Pf94Lb7PYj/lv0/tYI3N"
    "MOuZ+E36dh6id8u5u+x2oZ3e/Gvg37undtwJoYP0w4f5ZdZbLvuOHQRd6Pru3WPoM0Tpm/Bt757vdVfLXl4p8jqrZTft9U7q"
    "68Q4XcdhB2e97mWGDrwe8pbA0C9XUYjDNDEvbceJYhfuBRcestdpBJkBTrG58JB9aqd2bJ57aEHCphx6aAGRQta+eeahRYxt"
    "14nXy4V55KEFpXqPPOTYsWs+JP9G6wQH5kv4nTi2iykFMt979JVrEZ2b9z16CfrNo6i28h2wr3vucR8QxPbxBfmE92nzlUdn"
    "PV66vh1Ex+ZjkbCKVuuV+dSj+hbnKan6zEOundp0Wf6gHxCR2nxAfmPW4Y8eYu09gV+nPuySHzzkQmSUnzzkxvYZjs1vPIRd"
    "Hw6j7zxELl9Ar36H3wFer2B3md96NDAEtQ/9xUPHdhDg2LfNn/nvC/NXj2wf0oT5vYd8L7VphTse8pdwfcQr+oOPPF0hPwz8"
    "EC/Z/TKEBCJsMmPyc53CXCamz7/I5rXZF40/FrGvkPp1TdhnlK5Mh/1O7ePEDPgH2IGtV+h9GC1Md8WsXr0VsaeGFVyt0BIf"
    "22SylysaG+uE/gsBr4/hJ4X3HfxMcUwBOV2h5TpIfYYUFyuU24uSRhYrRAgqMdw+h49jO/T/IAWcE4gfeLji0gWKoAernHKb"
    "ZysWEe6I/SCNPoKPJDmLAElXCFaftP9yhdjjoPl+BW5JjmOcJDCA+/lnsvLDEMfmbysEt/OIIf1z+IQ4IOaLFYr91SrA5qsV"
    "i2xux9g2H/MvCtHTFTuMzWf8F2vq9QpBgDvo948VSt7jAEPygxXVk4jNH1coWWHsArKaT+Aj8HnlH9gnMHs/rajDZsBZ+jMx"
    "v1uBrzAyDb+Tn4n5LfmX7Idf4Oex+fMKpThe+qEdmL+uiBkgGcL3K5T6OMZ0Q96BryUGbDTxEqURaDan8ANYIAZPyL+TMz91"
    "TswYviMS692nP1N/ZdpLKlWJ6L9sYhL6Rbets0Snfpyu7YDOIo7NYEmP1dde1uuhUzjZCFNMDirT+tIYG3gArh7hyDKtL0fO"
    "yBvrxZfeLz0N/liIiTIghfzHQx1AwhD+MFNZ0/oSk/+oDW4cBab1pevBH5FCuCtIHsIfzhRAwgL+iHIPePrEmRpTwxI+vawv"
    "NW2qT/IU3qKmjSejhYWi8JUoyuFlRQ8jD4BceFh3sUg9olPC67NUsGQmqbPxdAqtHqZ2Ks8B891kfbkYGgN9yhMeHEMZFw+x"
    "zpMOo8CHyXKGxlR3Cqkc+Jmje/oIunlYqMI7A4sockX4cmIPPbjYsBTenecMLWFZZ32pLcaLgS1SSCl35hl4YFGjTOtLfTFa"
    "LDT6SfMXeOYtLIJ5Oul7Npu49Nsga+14kyn9HpjWl/ZstgDY4HsIkE0dzyVjOIrI0utDfap7ZO5Kw0mxHUhTDp9sbdwFtjGZ"
    "7yNaRq7Cm+H1kvcUH8bG1CFfeSMzTDt+L6MBKSGaIJWcaGEHZEcsRiNnxBNYQ1jHE28CDR3wcrwtXo42x2uf+lFANsLYGGrO"
    "lCfw5mZ4SJv7kZfjzfFytDleG863MLVhsgfGcGKLFNagZ2ANO9Dgc1GStyhK0iZ5A4G/BCweDsfkfgyfHLoRtl2ygZ7RMrwl"
    "WoY2w+tRR2aAuePhJE/gcGl4RJt6ycvxxng52hyrDZ7QIDIuo07Yw54nUaeFsRgsRiXqZAyNoTGVqZPhGu5Al6jTYDwYDxY5"
    "dTImxsSwZeo0wAM8HFWoExCnkUydhvbQHulV6jTVp/p0UKROBTKSUydKtDZTJ8MYOAOnTJ1GGNtALzdRJ13Th7otUSfPs217"
    "WKBOQ9sYGFqZOrl4QOZTQZ04hdtMnbyB40xK1GmoDQzdlqnTxHMX9qJInfTJYAZrw6jTzHGG0LSgTroz0Ebap6ROk5k7dSYF"
    "6qRrA3fgSNRJtw0boK6nThPDceGEzKmTrg0WgEucOukzYwZo20SdZu5iKlMdjimj8axAnXTPGA+MjdTJ0WxHpjqkucHMcMZ2"
    "gTrBHoPmNlEnb2RrgJ1F6jQaGvpwVqROhmMYhrOJOi1mrjN2CtTJcAaOPpaokzEwFjreQJ08b2EAlhSo09AzPNqUoE6Ga4wM"
    "rY46ZejnOZ5/bZ3acXdvL1r7e3cucT/G5HLR/erN/b1f3351jKDInvUf3E+jZ9EZjg/sBHd7PZC4XtArc2/+taWw+7pz+XMX"
    "97LOncs0+1fB7gtMJATz9iZHUetLbwp/yC/Gg+XcV85NWV/aDvyBX5yTyrk9RiHH8MtwDMcgdTlvCNit27pjvaXUudA9J9Hk"
    "2Bhh0gQn09aXMw3+wC9OIK0vx6PxCNArJ6Y5+c1JdU7FBYV/m6FzIZrw4mj5KExjHyfdNxoaaUjXNGRoGhpoGhpqkKShsaah"
    "iaahqaahGfx/pL3tL+0Vly68wUgWl9+5XCz75PtN+jaD9Vgs+5ADnz3rba+HDiWZR1cWDMHKFQTDJMFhP9Ne5f3k527Yq6qR"
    "5KkHedWKRyxSqupIWUqWalfs1n7uWtLxYfXKNms9dCDQtDTGtDjGCxCjDIx8nFg1zlQ5TlwaJ1aOM1WPE5fHiXca59lyXhyf"
    "xQ52q7yWUoZoHLgUSzVgS2YjLOXolUUOSi0r5qNcTzU56jLlxreeLnS0K1bwLgsNNltikjb08S5gRusULtxuXkGxMzOU+CDC"
    "kwsBRjxazguutzs62AMNWOi+rj7uGMPOEMKA9vUxDc1XKMCDhPaJRfvD5fzU675bossEL+0w9R1Ten6iz1SP7aUfXIC7/hMc"
    "+ykcZCHmvrpEouQ5iAbIVOksgFupJNfxermyHT+9gHQgriBIY00M1X6biHI0e5JI4OyzUM2c8xcIQ375pAsgGPHzZe6Au9wM"
    "2UbU0lvgpWCJK9tB5qGr+FzMzZB4BDZLe1virzn+P2go4Kg2TGkXcQdQSppQ3nHkGiMSyRcv+ryQxduu8WA2LjlJ+7lrsTlX"
    "UyN6D6qSNPpZR9Eq0IAmzWMiiL3MVa+oC7Bcy2pcDbrKkK2AvyXNMt2QWvnZtKbFRinGXwe+6gp8rZtJhbKrPJkVrdZCZbUG"
    "q6LI4yoa5kXUZ9ADvm6qc6U47orOaaWEQsX0565FL38K1Cjqj0poXFEWlfLq1ELljoA6VSqWVqrkm3QoLBvJslK9MSl/TL0s"
    "1OBjZdrzE16hvVKhDrlW9jWQkZIiSzX9sYr2SEqDTBysIhRsDh8taxG9JVnIhOhf1U1ubGNcc6/LyLWDcp/TQp801vP19Qnh"
    "zssPv8Un5QInMBjTGNv56T/OvfjIrcpvikz6r4w8nGP2uEgYC4SSk1vYOi/D4II9qQ+MayG45WaNMaHCCneuYyK542f94TI/"
    "5ZF0qKPyMc5EU1BekjWROg8L32VJFNkX3MvP2ZKKig6WXQt+WMiisiIoxSRNkMd+W8gS0iarRz2MQTYTVFnIEiIrwL2cl6xl"
    "ZY7oTi0xwIVcNBorD+YioVBx+4USeZwIp0I+y92zjLqeuRSwpts8u+BQqUzKWh1P6tNT1SvPzDhXtWnOr2NKG4ZXf7FrdTdr"
    "Wq4rLQmQJ647LVOPqnG8TE9U/hXGLFIQIzZTerMgc1+mH4qwp7KfT9oDdcR2JOGfPCyy2Y7qNpvYrUe1u5Vs9CPVRheLCLmU"
    "e0A5PnGdBSWtpYM+KUev1mFuDOIUgobBTnMNWzYvo/K8UEKocGNBGyKnVK5nWM7Wx8WFyCSNjALklfOrVnlb3nFNu1HN8Ihs"
    "R3lnacvDiHFO+WQW3WiM8zkmg+fI2k6tWddJ8YoSc2M0yM1jVzN7lTtRrtW7aZSjfJQqrV2ZG2naelwgICtwJvmRnmVUYeEy"
    "D8PZLsglD+EmO/Q2VPhdrpdHiZQ5lnYYoxRM1cVqLAAvBchWBtEzqlfkwrWnLpJbgVssC5ZkdZeq1aS8ggpujkNGyQ3Verts"
    "gjAraoUUfb9QHi8PaFQdbovDWHnLUriXqNwZa9xIVCeZu+KQMHQbwUWeqQRKFhjVgySVyrJco644n5T2ivmsk2Jc39yonNA0"
    "t9Wi4ONNt/3Soa0ziiFrI5Vd1RkNAp0iXSgMljy11y0cz1QsOc+qvakpp616kWukOwXfIiUUURzfA8EllVwrbUKJvKvDEinh"
    "9/WqvSA7+mXlQ3FVPJNvYplQDVPSIj5UMUUit3jCGIWLNHNMSRwX9tDL5dzCF9+ufj14On762+vlLz890l788aP/7Oj94MVv"
    "z9MXDx8Hz33df/nw+I9nR9++//XI/e350dPhc+P7P14cPE2ehg8uFsZr/ZfB9/5L/9szZxmkv/784+qpf+a73wSB4z8dP12+"
    "OF38pOuLZaDhQ6jzeuR88wOUf//rz7/4zw6+Xf3y8/f+y98eDV4efW88P/rh/MUf95Onyx+HzsHT8fOj44sXvz3Snx89Gjx/"
    "ovWXP7inJz+tv3/wcOkfPXj0+7eG/dv4xx9fJdO9dweTpy8Of977JXqsH32/N/j15PfZ+YvTJz+vZqMjO16P7mu/L9JXx+PF"
    "8JsT7eUsmH73+uzBIzc4OLbueevQAflS5/2yi3uXVGf4kqkxfqGjwHcwxGB7uUTpCV6CWWOMQeb4cMkkU4l5CdKD55GLD8ny"
    "g8Z4P4odm0gVLFDUe2Zf4Nj8Qs8y1O/38Xw+t1bp3oPX1v5lEDk20FLbcfAqNa0Dqs1rxxaKMbyJmtaPUZDCt3MSRQk2LdKN"
    "H4WQxrRurUfhqU/K2KGDA9M6IP9CSsEo03qBw5P1sgPxVjo4pMJ6N2KlHhMfC+WyMU7WQWq7UQcClXRwkuKkA+4Y4uj/c/dm"
    "a24bycLg/XkKVE01D9BKQgRrRzWaLZVkW23Jli3J7jabp4AikyQsEKABsBaTmG8eYq7m7r/6H2Aeod9knmS+iFyQ2LiUFrt9"
    "jltFALlERkZGRkbGkvCKb6gXD6fNFdWekmLZZSfLCyZKyUT4bPhRWCjtpzRMtEQgYaRAUS6ujlapgMEEKKyUt/dzmth996lL"
    "3K/hn1fwz5fwz1v45zX88xz++Qn++edTvC2//8abYbVRNPPDCZxZEjpZhCOvPaZ+DLfgKY3//b/zx18WXpyqj36oPCZgZpo/"
    "/ft/XQOQeU9vplGcsu5YV6wD1i5rjrXCaytVX/mh3XefucR9A2Nyifsd/98b/N+AzKIwnYoB/eyF1I9hQGN6Q2P+e+bF//7f"
    "8MO7jv0AX/jw+PMinLK/Af71JlGSMmykdHaNdaNFumC/wuhGvBzRX9lPtXsxyp+9kPXPembdsk5Zn6xH1h/rjPXD+mDNIwpS"
    "+gVEr01tdzR6PJs9vr+HmY+T9Jl3/+34R0rf2x3igW36Em8tbfcLCrbVYIdFb/wIdoMnYUpjP4pdEqKp1uv43//vnT+LeEuv"
    "vQkP++HCZgNTqM3//b/ADNslgVco8O//J0j9mfIdWiw0gE0rBQAMtQB7r3kSJrZCvoc9KOcKWuCHU88li1D5/IwmMy8eyq9Z"
    "ltnLLMvAn+LJzFnOokWY0pFuLP2xnk79xDzg/M0MvRndcxz3Lcsu7a5WLpxGXPDcwIIxHScGY50XAU016uTvL2r9YXRK9qGN"
    "fbIse5GoPiYWmdDU1g3nr5AwCTfwzMiynGv/PONmG8uw1fpmBoYS1FwkVL8ak/foIEKoOfPv/FB/MjNwsK9mjgv2XArv/wZb"
    "cUbRcDEDHyDx43lA4Y+xfDVzKI4sdFITju8JTZHFv/Pfwq5woYfA01GX7K5W+IDM32i1Xs/0kKRGJnt7vam31BwGXpIArk0m"
    "xOmusp/QvHVSC41DK++fDIc0TJ1XM5IyTzTmx/RmyMrzsd3M+nRAYmdpmmZIPKxjh/1XswF/QAsrePHIxSN0/h4sqsQHUO8O"
    "SBQ+EQ24Ueg+ejXrdwZmGr0Ds3xmO/Xo1cxMYIfVLWOQXYB7T0DTPiXhIBprnHAoNweKDUPAntBUEpL7AIstl4SMFr6YOX0X"
    "MNVOQYIbkBfwAuUxl7js3OAOyMtiMVIt8f3Mea0vr65gqdjutwsfjvCwfqN5Yi+ZdJ4x98IlSHgh++vDzg+eWPbTKAqoF8L5"
    "f5yW3oFcmS7muXyiM0MlakIDPf2lbpDvdTeZA+Ncvqf3docgBdkuoAYKuWRfwr9vA0JC2uu5IV2ksReI29YR1bkwqlP0n+31"
    "rBPjkdV95CrnqLpvmUH2gZG2p/5oRMN9ONcv4G6vL6G7mbhkqYDlI4ZYd3mLohPlDfhnPI3umFNh90jrHrFgQOJkm6Rx9J7a"
    "7nARx5h2KwDOuM9et7GDfdvt5q9A9T/05vu2ywTnwoefIz/Mv+CtHvIid+wFiRjRXgfG9IKEiyAgP8Y6xSAURO9Tkg6YERkM"
    "+1f9aQxWKu90nJUwI8uYjq/GUWzvdTKSGqwB68QwDGJ1zwxjQM7Iixn8tbrki5lh2EX8QTMWqWCxOrm/d8TWUctvgu2XMwMt"
    "hH6dOSp1evP5tRe7GXla974dR7duRn6o/eaP3Ix8NXOqK5EXmCEbr18v72qbBLbiZuRH4ENYDa/r3AH5MudMYHHt4sUalhDa"
    "SBAVvWsIjXiHd3CXgT987w7I1xxAqwbAIcgCGrwIF+De/Qv0gopyd0D+XoQwAST9c01jSeqli8TNyD94oW5NIQ+3Rij1U5mT"
    "PpnPnzKRDHkpvl8KfSheI9PgB6Z3Y295FDZMlOinINUu5iAL2nlRdyBYKllCITvNjCXbAymJnTQXDXzguvAldUITWu/1+gPi"
    "O6k59sMRpKO80ymwYlT9OY4Tmnk3BvGc5ZM4jm6/x7XnP7IIPr6k49T22xb5KppRu0OehyM7NQMaTtJp28r61HxP7wfg9eo5"
    "DlDtarUnvgtBC7x3Mc/gMzr2FkGqG7iLR07a171HsvSfxK/BRazXYIJEDHRWO3GoyZf7Wy+e0PQipr8saJI+Cf0ZmqN9EXsz"
    "qoNIlpi/LGh8Lw69utsXhOjsHywv37wxaTL05lQXPWT7A9fosUBWsA0XNjPB59iVh0t+nZH+U90d+Tcuear8/mFG+ql5kARR"
    "mpiwkkr731cz0v9CFyXwBPHeNQaGYX+luy4BXvJUd6eWS97NyHudonRrEMsYgKiIM8wRJhsOvZu6fZWTLorCJI6Aa3H1vFjc"
    "uEqRIwuy7PVYhyQKv6b3qCLym5gcVCJpjhmmLnIFf0sZVstczvTSNE4IkyBkHQmeWz8CuYV41/uiZQJG8ihAeNdt95F4u19g"
    "L7K04zhUIX0imE7D917HbluEsyObOn+tJ85UkI4Bsw5sv8d5/fczMSUoXbGPTMKyjjOGxjPSd3GDLE8/IxUs815PWeJZoAEC"
    "8SIWYcoWXU9UsYsk9vWM9MN+Z7BaObKtCkt8r7uaa5C2ZZC3ng69YMuM0gw2TzSEFVUnwNXySFHB5ZeWShsZLhDWwN9nxCWa"
    "+wj6lAWwX3JGfpkpqBgQ64R8OVP2w6MO+VEtQcK+1TzOuTekMFZEZBvRxxcmY/pyZLhy/1lamXxjKK5N2QLfEYpN/KPUhNg2"
    "Cm0M4Am29O9mTr/vxnSYumR5Zx+Se/uQy0XWmZCHrDMS39ndbED67twDqX45st1Xh5rVnZ7fnL+yutrhzfn03M0GA3KALSrF"
    "upZ25nW1Lss52rbalnl6GLRP20fK267WCdqn2tET+Uo71M5ulIqapWHFU02pB3+D07ZarWtp1slPbgnY2aF5qJ1qZ+apdgz/"
    "to/LJWAU3e4PVhdHQaflUXS0E+1cs06D9jFWHpC0VGZ2rlln2kkb/x9LhFhi6MdDOF4sh3egdXbJ8J79xTyydXCc3JwERxqD"
    "JJ4qMyTubI6UG5sjl9zZ7plL7vHf+A7lTXCv61baPtKsk2HbMi1Aeds8b3fb3R+Ohh14BU9at92dWp0hltC6mnmudTkc/gPG"
    "MrOOtfP2iXZS+XCunWvs/YB4ZVxbx9rh9OTmpIqajmYdwQwfVj+dadbhzYkkDQsG89Wx+tzu/pDTk4VDZQBEZQBOtG63VPOo"
    "VPPM65pHGvxP0mbnyDztnASH5vHZmYb/PimW6Xa0sxurW4Tx1+pQjrTuzbFnaVZO+dPjOmScf3VWfX2iWYcN70/x/YAkdTR1"
    "pt6qIk0dIk0dCpqqrv/zqXVW89o6rnt/rh3e1LyGyWbvB2T4wPVindy0j+o+nE3NjoVNBxUi62jWoXesHXMkn5rHR+bxUXDY"
    "Vt62T83OKf4TANvqwjRbdRNmWYVKx0dtaKt9qBW76JziPwE00+ZtDciiAltX63Y46MWuzsxjzToxj7rnsl1LO9WqWDnWrK55"
    "dnzuwTgFGR2b1mm7a56cV4dwXi3faXfNTue0bZnH3eqC62pn5lnXs441S8BxZFqn2P5RTfFK+U7bsszu2Vn70Dw9qczeDBZJ"
    "F+yDEEWjMopg8/GAi7Cuz9vn2rl5esz+Ye9OzFNYfadHL2ELqkJkaYc3x9OanQC2EKVxaLrceBsab2Pjh5pV5VRnmnXy1eEN"
    "2yjG5Y2iCzuZZp21z9qWugsemkdnsBGeaZa6pQHLm1on6l5onh62D+so/vymdiFYp3IlzCvUdgasWKuuzRm+7mp8T5xNnb7L"
    "zVJTrz3yk5mfJP41rNWC6g8kaXdAptPCKfjag9Aobbydzchkyg//Gbma1ugBeOn80HszLR7siTxt309Lx+GnWFceh3NVIlpU"
    "CUuADIT3JVjIyXGUFY3809f0Hgri/Yc4OZNkCnZXankyYqdMOGTkB2vexqbjNPGdJXP8ZhZ+aDGIRoIuty2UhoXSi1yaCnId"
    "pS2VlRmcqrG54ZQ1NZ7yZvypbIBOZc3hNCMRevjmyLfdRzQ/5SdwyvfH+h7lp+s96yKN77m6VUtokvjghB/F3oRCCK4XKZ3p"
    "kU4NA5TxlpsNvXQ4XYrKWa7tH/KmqQENllqK6Sy6oXljvJksC02YgtVqqIdmPlHsjB44B6nOCrRaSbGAcfFliIdz9S0RQbfS"
    "Vivdcxzaag3BuS/gJzMYfWYQWRNaBt35kq5WegkCWWnPMjIjx+CCXRzJrx1SqFYz9kRisdiBa7kSD7EuSUxRF8So+2bj3+M9"
    "Fs8J4mQ4sqnpj6pLTxx3leUBx3TlcbW6ifyR1imoVv1+KDTnSMgDdq5mL4EQkArdnusFNE5d25W6r/4PeGCt8gCmv8VFy9Sa"
    "vZ5X6YWdak/YvUEFiPJhl7jsMgNPunOXTKfscJjGERjn8lMvXB6kAWo+SMwOs289XTmA8QVfOp1OpsUTGC9VPoFtOMddTbc4"
    "x5HChPTKahCucCzqOaoIRsZWUcjk7K7Xcy9ZEcZC678JRcVCTiVO2St9PjX49BxVlQ5w4L6ZFg7cZ2SmvoBj6vVU6HCBXOR+"
    "g4HPQIhlWw5X614vknt3QO6gThKD8/6baY3uFe0hmKI5I5fFvYp941vVbfEbJ5KMvK3bt2Y09dyMPOffDtVvTCP+rLxdPYfO"
    "8t1K7FLMGD3frcQ2xmK/wQ6EAU+WzLfGG665EmPK2SWOyj6Y5TtBNB5jhK7FNOOxAPUQeUeRVZQxI3gDQgC0gj96PbfwlYOF"
    "tMR+So6BPIFXq2EKgq/I6dxXC/MYq65oLSN9aiJWWi1ebE8pJleFP5vU6SjZXGN9lyTx0OaNEYwnIq7fXOuocOiGeDKxN5kU"
    "r2I4dd8B/dZAXGIUb6akb5omZy39WgUSgw7iyilKpMFA3HrlSr4uuyp1XUYoNcjl8oP6ic+/m19z1lx8Fjhv2i/PNue9cpJ7"
    "1pnd7SornUuFnPFCYYwnIbTVl4z3zl1yOy0xXWrCIlitwiaGO3fJW+STYYXbkmVGYK/uo2qRtcQ0iyU2HK5lw89Lzder08gZ"
    "uZ4yjdq3U4ddwLg8GKALMQ09sEufgDwo30qqAPewiPt83Hrg8u5eL8ZjCvFrYetKbib28m4WhIntTtN0bj9+fHt7a94emlE8"
    "edztdDqP8eaTEWq3I6i02yldX+JpSr2+ZFGJE7t0HtCsziUcFjvd49Nz+Rdea9al1TUPzzvH8HhkHp/Akfj88BwOQaZ1egiK"
    "j6PjQ6t7CafU0+MT7dA8PTw5OcdTK9Q7MruWdQrnI7PbOdWOzGPr8OT00jo2z7uH8PnMOj4+06xj8+jo7AieDzuH5/BsHZ3A"
    "9+Ojk7OzS+vQPD0/O9QOze5Z51SzLPP87PxY68L5uAN/Lo/NMwi/g6/gf+qz1bnEP0emdXR8Jj9Zp6w6/L0UH9nLU6U4L3aJ"
    "f8/N4zNEz6l5dnh8pp2b3WPNOoN/4eeldWaenBx1+fvzvPw5NnEOrZ6fHnXzv+cIwvmlRPw5Qzb7qlmdn8QkFm+dgQu486Pb"
    "9xGemRUToPdTIWHvoda+1UIO6XJJXoN9Ixpr1HFcZnjrrlaP/+dfo0f6v8x/jR4ZvYPHZkqTVKdGzz1Y0mx+59r4w2XGUVOH"
    "3TSDBB3QlF0ugx1nnLGjjWq5ktZbrvj9y0CnxsBJRVja7zw4Lcm2/MzIyM9TOKWIHY1Zi+C21/8GISdvUlg3A3kYE3sEMwdm"
    "ZVmZShFYmk1HOisTO+QSXU9e62yVs7sqE8Nq6m7bNTDIDl53gpnck1TvGCWLHioNegwTbvx1F1yFcPdP+Z6dkiXeUUF0YBxr"
    "7PyIJ4+UmyZ0Owbx+TsdbJHeT3X2zWi1lvsYGhoYbhte7dv5V7Su5SG/Wy3unydCgEOkTI+32nfnXPLmP9B+CAYLmVxNwFWr"
    "JT7h/jQwWQIynSPPEPGFobnvPGGfAda9ZnIz4fwq5ndmnGnFhSu0kvED2lwxOxyfl2NbpScu+qiJHA0n4QnnyBy1WUZeoejF"
    "uPMbwYe5X9ETxPeeRYRkJhm1KkaxyU8z5+ep/u1UDpCqJhav4OB4RfVLT6fmAc6joRhWZOqZ8JupLiyW3IOlF0/QzC3h16t/"
    "7bRa8l2/M9hzHEarPfWt7c6HbnawHMe6IQ283Ju2S1xXvngSBECexL1yjcxVLO2mePIVx/J0tUrR9NpdrfZoj9pPAkilQMf+"
    "nZ2S95TO7b8vMkIVW70vNrTwc1MLwDleTJ09i7ycOqI53VjmWoU9q6g06Kh2jd8jU4NGUhI6VBgB6KmD6odvmWWmobxPWRxs"
    "1Ejg5SVDps2NCV8jkGBRsBdyA0UYBGUWdWB/K/hleLH3YtpqvZzqRqulwxgw0ncSBdQEbYvu9tHS9YcFHWhvp1RDu3iNGYtq"
    "Smfa/sEyzPa1kNJRonlaSG9prP0NnCjowr97jD2PLlgVDegy0QCF0KC2P2/vawyzpmsoM/LrNLeD02jPxN5Xq7/nJZ6qJbC4"
    "mdD4xh/SXu/9Amfmh6kT0lvtR+q9f+XNc7R/JfcSjnznhyngW6cynHi6Wv0wBeUFGnH+qCtzizXEqoGZYqathtiXUgZs7/up"
    "DmCZfhIFaFNhGIJN480wW+4I6Lup80+T3qU0HIm1DZHBo9BVF9qPCPb+367h0Pp4SoM5jROtzbe+fQH6j1OHb4YsbLzjuKIJ"
    "t9UqfDL9lMYQSNpx3ATfuD050hy7Ym/N7JqPtNRmsTvETpLGiyH24rAybPPmMMg49T0Bgy07JDjmfNK/VCf9O3hYrb5mf/7J"
    "/vx9qivlvxYzXQDxL+4CQOuXkDDY4+uM9t2//U28dflrsXCexLF3jxHsCpD9UmAhAraf2FsGHf4owgdPy3QaR7ca+Cg8h2OO"
    "7r5gEVc0uNafzVMtjbQRZUhcxFQLo7CN0F0HVPPDJAUnEPO/XoQaOjhB8WuqiRIEy3sAtRahxJJos0WSalPvhmqeVsGCbmgz"
    "mk6jkekqkPIB8FWTYxQErgRlESmK/YMVvcAY/s4yM9OISSuCd3Hh4QzOgYJqkVcxiapMNq2WHjqFN9yaB+u88ubCFvsNTd1e"
    "YX5sfP9E7DQoE+o9+52/emGEqd6zz1bWyeqwa+g9+zLwZnM6MlgLQlwMjR4fj1i8OU74h6WeCg6d/pUKi6xWC5kDf7qQSQ2c"
    "Dokd7EJPjYvwL+lF+OiREffDgUP74UDgI8674US0ZNX5LtGDf+ytyLpC0LA/hJyqsdWY+MQjEUmc/oAMYSsIHK4bR+M0PXRC"
    "ufGY4M9AYIPsICGwSdNDY89xQmG2NnT2rIwGCdVg4Bd7+tDRY8djjYSGYY6iEOQ8PTHni2Sqc5nJIEJu2HOc1IBmOsYF20FR"
    "fAHYfIdmY4jdHtwvOYh7w1YrNPkWy/mwHjnilW4QDmUEUEYGB1M2Azptg61DP8vEfYAy1d8JRsLIy0/YDFLRkkbzsgeF+eKn"
    "hPf0HhTwFxJh9Sk++HysTQMCzaStlh47sRBXJV/OOdCGZCMIopl7YcCeFOJcmN58HtyjEk1gIlSIEfAgiDl1rIv0L2WR7yJ9"
    "9EiMPhfy0gEn3GVmq28v0j91ewfTnIxAG2GOo/i5N5wWBkYnzAuknw6MzLA3ZVPpNWRi0SnZVBUo1FZhqgeoyecl3dQBuooY"
    "mcSvQjxikLkTgJNOwMMFM7Q097jkqVjI1slb8qQtSvfpJBeKQgBFMvdcvNBTQ5EXUjt95CrSONZiGwVu3nuOy7YdFKblcmG7"
    "w0X91gz8XrCxNAI51IcgAQNDnh7EOhFsCfYb1mOs9Ch6iy8qe+zf/qY0zDZEXtjT5vI9ItVUbqWcfLvrsU3NZod2Q+fHgXgi"
    "lGdPvYReiphRUjk+F0puNpeVE/xciP3riy1CJlg3HPZFsdH63sBVGb1PlwcsOMMLLk7Yy0rJWw8jGfjJO9m1P5vRkQ8q/L0O"
    "DxcQF2TEp1PuZJYLwZDrSneZl+1wCvpcl2CZK1AmXkYxfQOtJwaBe8m6L3B9ha+xkUtsA7yoaEjj2gpwYEVMbISXET60cbET"
    "6PjuzTCa05GAxCC0p0DDPqLjFh8FNZorq2dJIYQ3t5NtxEcdfIbNvi3C+nYxVFGCr/HN80Bcaiz8kfiJugHpEC1eSpTJN0gx"
    "X0WJoCdyTcdRTC9jCrNROl0hAyRSKCFDEpAFczacpz3zapHQeUpGzqKnU/E6PyRTh5pR7IMLJbfPLx6VaZ/Nq4m5sWCpDjhQ"
    "4N+nL3p6Wm0UTt43NY2lNY2JyqvVKG8AJKhpFL1XTvPAvaLwqYII+S0WghKyybGj+9zXEo/VNwt6yc6csi3f8XN4BXC+QNXc"
    "Gfd0r9REXtkDwazUILyrNOgVESvRNnT0cU+PGjuInKjSAbyrdMB9CGw9aWwrcZJiW6Iux/k8Lzp0htXpKXwuT0ngDJumJGBT"
    "MuSrNqfxEqS9/MDPS0rad0o1V6tyVRbM5fsoSnlddXk5oGlj7xf+yMkLJKY/Em2pFXJN2hxix89vrvzRFUbJwKGN1HXHeAGg"
    "AyJqsLFj9C22UF+BVXu1PHKOIlMWTTzN67E4kOh4XFnqF2IJ52iSKKdg2A5CFxQot/+KNeiK+RBIbLWKz+YNjf2xT0c9Rlt7"
    "jrNnrVZ0rMvBvWMeCY3YeKqUgrEwD4Y16GNFFfS9C2f1CCx1wYphH2EjxlhNZlSEDFuw9HwSGnk6WYvuEW1E9zsBD04m6gcS"
    "e4mfCzsoHMaKfuXTKElR86jsrsiYXr/lu9OEpuL3XBB+vmIFJFIhyjxPXFy6Jt7ggCpANssdpdTWG+tdpD1TN0gI/2YZuZrR"
    "eIJydVIjFsBFtzy5DOQNiDyxw+m4R/nxiW9zqWG/K7/JCK6bysTiEEiq7v13I9C/S4bRM6+9hK5W9yPiwi+XVOSE/j9NFH7e"
    "vIGaOMlCjUwoW7BfBpBXRFDEIIMPB6CERcf2VmtNp7JUTc8a6/m1KFLpP5MQFMWMixJb7JWeTRRCzGtUAqSGneoChayJera0"
    "pZQomsqlxaZ5IWGNVNaMKzF65ghWnad303UTJVcPfgCFGG+tqc4gM7ILIbmwT4pGGmAAhdQOiA4NOwT3PVKhmXoE4dpjxRqX"
    "3gQ/m0OwE+Bc2ou9WSLX0C+whBgd6ZT8JJThrJ6bqRuqGLjUy5EyaalwSraUH18E8L9WBf0LGRwDiQcumtHYgO8tSkdTSlMU"
    "5fdA0f9G3LK8hAy0I90NICJQm8e3kjc9JHboSBfbFF8uvYOFkI1L06jIkKhSwDBD30KbCMPlmzclSSgUyVKpYVxwbMY5NlWg"
    "WHgTOA7F3i1zLaxBMUE7TjYoOcbS6OC0UEZMlYni1VQtqviFB8cSiv+Orof1mADReULTS6yDHZZQIARowlFKDWO1WoJNsW/K"
    "4z2JHN8UuR1I4vgmIzQyhPfQo8Ce1wOizVEom2jfwCXvdUDBOXQdBnlDUakh0fuu7SSldhjgO7aCc6CX28Ci9WuNDJtIQcxe"
    "1jS/Sms9tq2jgryZXTE6CMiCjMiYzB1dD+ppYeEEnBaYpqWOHBaMHIIyOcycOeJx6sz5hOuj2l5GDO2zHOMHS6UcMyrYGv36"
    "uLaTcd2s1PWzboqmDVNUbsUQ67UiJVZXrTiN1/JKXU5jqmCen5mjwhW2cludJD8IZIFx/Z6FWr86cgj57uWZ3kgAG72noU4n"
    "kPG5PC6Y21ZL98wR5JWlG9FAyiydiyXs6B05uq7H9YTnOzEQ3muMFFdHdT6juphQcWbk2jVxQBPk6DFyNJAWE0fOuXqYy9o1"
    "lOCSoVNYWYzJSCpN1tIhE7wKKh5n2DNpwHGifAHcOUlGmk4WNSocebIrdFDU06gqFTiAsEONFN1KALRaepFlFI4vqQnxGPXa"
    "mijH1AiCVVHmgfYrSjtZveowLGsOQW9Xe4qrAlXT4AXdQZG6vujaAnWaw4yAZAfIl+riOsOAXv05sEfV053jOE3FbClEql2B"
    "5FvUVYOdcll9LWXCCQsixfRVdbYNUz/p0wFXn1Q6w6Nwz+zTAWtLEWfrWpvDVUSFhCyVhKxaErIGYPZcqdlVa3Zra3YH9jLj"
    "U/L6bQmyD6RqYK/Ow4cTOg8eEImrdQ/Vuoe1dQ8HNl4QPzYfT9j9fWq0Wnt7YT8V9o2ma/Q74LLt5MoHQSC6K29fXKOsnWMq"
    "154pi6AEETmeiVqDNywoJfDvyBGgrVYRGYoSqFcggTOUkLdaQ7Jw4p7fW3OCev32EkyneLwNEkq9fa2qo1qa7zEjx5fbb174"
    "DQ3GOiVN1X8COSQjoQGm6EsmINoLGDjcF48VFL59xiLRJWA5OHeS1WovabVGvYAPLceAzmU6w4bWsf2FQUaGeAP+jPBAZg5e"
    "t0F8XLc36pncfWY+hNhLg15PUYAWPwm9NF+Us1Zr3mrN9hynuqv2eI9zNsL9vJl9EL1mmM6zuhe7mWHPxZoDHH7cJbdB0/RO"
    "V3RoXLuEOoq+qk8TQgdKOQPQmQ+9FGMlrq19cIUYff1WrSL5i5jljzZecPZw5Oy5zAyST3qr9Ysub33yKU7YSnMHEiF4Qs9z"
    "6rmtFp9WtS14xURGkK8yDLZFxoHO1SANfdgVdaRBwlZLaYjZ7WFT1bJIVlC4QFk1FFWj7wfb17wB3psADLqjysTky7ZJPVPc"
    "vdSJlxMiEToK0HxthipN7stDM5uK3houTes4/ta7RQ3H7xa3Bc7JfKfcu799t8XXxHPiHoSTs0EZAMRgkAT+pnzdCDtPv9UC"
    "rp7Im7Ce2Y8GRq/nZQoRsgMLN9KQdo3m1EsUMxBhseCyyznX6C3ZL5uK+7riLZuvly80eQYC+MADJ8Fv9DQkbF0rE8QuUwWi"
    "6kQgsP0Ga58L1b1i11GwDjxHDOKzbZ816Bk5OWrE9CycvNpI/hbzOQr0kdEbwd+F0VvYG/cvuVfh3mWPhCmPj/fx6k5emoqC"
    "5Fm4B5nwCoSVY63ILX7rZkaihmznADxEa0V11BSA24B/vUipru6iNQyJaxby8vU8C2xYZh93g3ggV1nKxalsjEzSE7dNVMg4"
    "BX00mmfBMPzfyThI6PBdn+/PP/rpNFrALPORYLhhI7en3WkdhxhTEvQmoemPej1xUz0ySIh4iP4AZ4pGWmCS7lJoZ3B7RqEX"
    "oh+Q4d3vkZbzewzBwl5Pa29dChov3MlpsobqyVdTvWR6ASthGFaRUJRSBXTgjSNH1qM27XcGDY0mNKg7RKM3Tm2N5PcxFRD2"
    "4gPIEC6XhP3ixhnzQ/BuZpqcpmnL3dT6te29m27bDIkHGc90CSlY7OWksompQmXNitKbdn2h66wxVV7AXVW+/GgGprFkVN35"
    "PkHftIp9uH5FTTI7QkgEGasVQlp+DcAqVpE1pntossO/56JHPbTSsLNQKyP5TWG1h/LVZtkGRmJCNAEa905GDvxGcMuGRQsf"
    "YPDDknGEmBHVyp2fgm7AjRH6jOYo68mtiZcFF43n3He25EoroI/miVFn4s5sy3+Z6imxQJtzkTuK+eEwWIwomHBnwDLYpUTj"
    "KMsYwtIZt1esqYULh3NReR5R1pZ4VzbwyAi/uK++fFPuKsuIvkYbyW4K+GUEKSpP2UucqOJNQ3UoPI4vCnv/J4TyrTmU7g9c"
    "gQthlFyxRuIexWEES1daFzXZKYLSf5jMyyaZJlYnud1oSd2dIf2rTW6c0R63EswILubG8mW9szi35Kd46BuX+pqTdd00Se5K"
    "Q+EvWJQxhFtvTu08IAp7A/NDmA+1siQJa9BW+CVvh/aKrcCz0gY84i/g8FLLU4OX2sWIxRnplRdkvnOjExyuyB5E2IhTFFNB"
    "q2q7RgaOvKPFkOrqSaawmLsGiR2IuUt8B0LSXuQeZV+CjxBX4dqgF5FasV5ds+zIy7kjuDm0WjrzdnAcJ5YxuHs+KM8JfAF/"
    "XgJxloGdl4Xsj4MlpijJkcSHt6en+dV3BW074U0ywn48QAyKEYGl/6TiZHrtjTBiDHIg9780TdPMeRvfakt8hP8b+ckc8pRr"
    "jNG1xwG9u5BfWVa1dszSqmmjVP9vbMBkH0z24b+NvIYX+BP0YJwltga5Q2icf/x5kaT++L4tErlVCohsZ0pP/JXah5JOTS3I"
    "MruZ+Ve1DgskUFMcP6glIQMbhiVQS8NLDFBQKXnL4gKUy7LXaumZH7KEC2rZmR+a+FItOa00OS20lv1XYTbboyhVZrTSxyiq"
    "gl4LTF3BKix1pUp0ctz5U82cduqhZ+FUlQGUK6zroNyYTDestFdPLrLktgSTVyiQTAUCFrxwc/+s3Na98+Jr+4Zgchs7hkLb"
    "9opl13YJ4QQ2dgmFtu0Sy67tksWJ2tgpK7Ztt7z02o5ZWrsk3di1KLht57L8esqaKR3XcqlkVs+oald7MtuW+ySzDQwomGyC"
    "LJjsAFkw2RayYLIBsrtgE2R3wQ6Q3QXbQnYXVCDjoQ1pwtOCVqUHyg5UcE9mCglQHP5dsXfPMXAd9ztc7hfZ6L79iy6cmVot"
    "5sQoX3DJyHEci+wrm8e+TQOl1l45nFleOJlBsghAFfpKzrwgcPOvwUT9GkDyEeXrXaB+vSt/BlaDBSBpoZ/eQyEWana/xFrL"
    "pWSY2X2VKZVLsVC1+0UmUi4kQtjuV/aTSqfig1JYLONyWfHezQaYvNUreJE+ZXIaE94SO54IZ1KeiZpFj+IhoZg/ah49CiRL"
    "CKTIOqsPH8XL5KGo6r5nIm4Rdn/jj2oOVcuD+RChZQecslkOalbVCDjRZKuYJtHkDxTTBMecO0wnVY/v4Wf0+B4qHt9Brcd3"
    "NFnv8R3kHt84tM/q8R1NfnOP74WyVgvr1JtUInPlmlU4mb+uOVe+DfRkAv8tGbtmZ3/uusZOiYJnyw8Fts1CpOYfHV6Lu2UV"
    "2DZXAQvmUHziyiXgCvkvA8+So4kIZjt3B/lqHk90xXdXDqkQMfSdLuwJzOGdzowzDB5udb5ve5jS8jUoWOH6TRQwIIQlrY2Q"
    "WXWt4eEyeUQRSDrJcuCMJka2mJgxDUfgYj3JAZ9vx4bmfyQ2NC+yodmkEMDofoIWIDcTHrdoIn5MJ2oAI3j6zwhgNJlsHcDo"
    "avLHCmDEx1MNYMQ/fOoARpyIfmcBjPYggNEfIzoRLtYtoxNdT/7A0Ym++yTRia4nW0Qnuvu80YlUmD5fdKK7qqz65jPKqm8U"
    "WfWyVladb5BVL/NdYP7ZZdX5by+r3k5kSMw3NCUHzhJur15BdO+qHBUFenn1/MWRJnOqOYWxWm1ZkiwCfYd2e1sWtKvlrGo5"
    "a2AMhE8Ii21q19ypkFjGxFVcb3N1S8XbNrdRQa+33Ek1NYfpneq1GgP7uoSrjbu0+L4cHgXeceMLyRsqPq55x2xU5atRVaA7"
    "MNVx46f8ihQ+vZCXr9vXzy9oK94x82CtS0qd9c3WBmThBxjSxQ+13HFd4j/UKQU9jyt1j9S6R7V1j9DaKHIqN88HDQbdBztZ"
    "cyeODlf2JiRx9cNJs3NeflMv7XfhhnuTee/QSUrmvYEzVMx7wX43Uc17R85CMe9dkLHj9Q7Kvi4kzY1cdYNEcNyUfi5z50Ca"
    "w5IDYRkUEmHRjcW/078DIx6/4NcyFn4tM+eg4tRCYolOSNMctFrz3ghAUwyDU+aqTGaGDR1gF2MDfEv4G+ZoMvt03hQfbFyZ"
    "e2LICPYM7lTxpLgT7ghh7kXBbaa4w4JSQnVYUD1JanTbH9cqtNZ3IK74DsRO2AuZB4CPlv95QJee2fcHvV6cm/hLc1pmRLuL"
    "FX/cZMUfK1b8sbQerzN7+7hEUUQix88OiKyywiLPk94aVb5XZHDCCbrONwFQDKubWU2W0Z+W3Q8iJ5XuB3QTc0qcqMSchk6i"
    "MCdgVpHKnBZOoDCngIwcT0/LkzrGl0Xfg5HiezCu8z0YG72x9EEYrlZwYhz3FiX+QoXjHF+TzGduLOR0T0+bnQZ+Q8ZSv2F/"
    "JrK6yMUXtilIBWVzEKFPI6DUrrcdnJ7YnsRIGc+/4MKq2MiRyOFWeL6yjftbmdt52cVBJcxmSiKDHFRDI6jvqyHGxNfa+GQp"
    "pozLsV4TUamhwwzarA3lpLZHvC3iOQkC+ID59bac3wucgVJAC8oDWtyNdN1TomV4SqqAcrwMCCtlYFwpf0NcqTz+l2fIwj3l"
    "JQT80kOMg8OtgZVQG3he9Y0ycPEa4GIFuN0ChHloOr0ubNPnW4kYmEg1xMutqXXDEKLCr1M9VE49eKpXTnwgSqc1gUAoD8hk"
    "GEooqYNqHKmUxOh2Em8ZS0ooPelIhwihNTPqi57V8FLIN0rBTMARb5vYUhFr0CNxHlsqId9tF1sKPQw+OJQUU0DhhOCspZvD"
    "SSGiHEASrYwcQkKtjSXF9cI+kRH8DCk/5LGkhk6Ux5ICwYHHklrAezWWFA/d9N1WsaTCPOTTsFRvXegopVpQqtYcKSosBYb6"
    "riEwVGiA02O6OQ5U3cTQShCoOJ8TQbWxEgGKH6nI1NH1Uc30jZ0RTN8zH3LB+3zHKc3gmM3gqDyDE2eKyLlypiL807zaxZwh"
    "cpLjENJIbQr8BNjUZ9XWZnUYrjSo4PqqAdd012hOn/UYAXZKEBdJRx4ZYgTDgi9pfm/DQizRD4ivFJKKNwOtujKI2YcgVioT"
    "V4MvRY5brdsuT497QUthkxS6ZgHjJLFEeQgmsFqnASnUxcBKUfaZwxNtN40fGLXowXQGENJNgdepKvcZxUdHhv8pxadVjRPw"
    "vAjRxR03Cg+Wb2CXzFwSFAVtOCiQhRP28uhdZJSrmBa5iilW9tK4VnkWocOxQQ6ag9IOIZjt2CmrvBYkUAJgB4XQ5BAQeyS4"
    "X1lHDLGNNvY3d5Y0sEPCtYB2TNB3y/bJPKY3P+BvL7sY9Ux9AQqsMf/REDC3cf7reUVDuJPugwLrMtObj6dbQglStiaD0rNW"
    "MTL9RWxeSdJInMIT8BW8SloUiQroj4yKZTEIFSgMHYp6AKF3M0jaM3nAZQO9gEu1nFwbAMpM5u1EyQE4NNkxORCz6pODWTSC"
    "UNhxYvs9Uz6QAx5OgkfIJgc0sEfmAYVA9DLfMXeJ++5zuL+lPe7JVvH6WqjJBRhO5b0FQ6uSaqDu+2pVusiAl4XA68WUBjpM"
    "6cF8VFhvfRlGPg8+X4iCUOu7WrxIeasvmlxT2eWjsBAqhH9tXOghhChrdETlURzFvZaeVmivGLbxoHIh0JgAQbqoGrlia6G4"
    "serhmr7ALKPaF7yt6ytU3GErbp2ltHwbxlfrn5iPibuB1rrcFnG5V4vMhmMkBDDciI61oIXKGY878TI56+Go2GGqR2ltbI+0"
    "7pS9tRu+/+EREQ4K4RzK4yUfFQdzECH47ZFRHxvjN7narMfkrrExNmASxu43hcXAjx9ACN4HE8L2SzFXXhWLm/NhqFfkpS1W"
    "rXp2UeJw83gbPqMYz2gIppE2B9NIc+kptVOwoGCJRcvw9FSBuRpeI1XDa6TbtpHc/U7W+l6njjx2pW6/VzO1G5lD3dSmxYga"
    "oZzfPF/WGHRdsl2nypWGjh44ShGjH+ZdD0XEaiFxkrjvHmDWioFaCYR7zNFB85RJ2PF8tFo5YO8APxUxEZ8xoimTF7m4KL3J"
    "y3CCUKHcmqZKfHa8gCiNKnK8nompwkji1ASSdqgZ0lt2CoFDfxSM8EE6Gyi4j+oT/aRCBYlGgGT4wd30XdaPGfvzeUDdwfoO"
    "A8crpHygN0Bpq1U6vvDMAxw7jZNLLwjAt89ZssbthOwXutm3hxD/LOcilRxJoejX61XEAhLg4Zy3LQ7FiaG0GDePTGaY8oin"
    "6Fu8DdKHz9sp9c5eSiCGIIZWKaZCK1VckbimUD2mQ9AcMr1ACQehQCMpfi9BGVZxQzZDKAz8BEtZVlIppUK/kOZrMBVrUKy2"
    "Yta0he7OMa8OoJRnXiKinYYMTKk4iB4UsjCJnnrSWg0bgZaVZqB1wSnw0riSaygtaEkubidw5a6D/b++sUNUmUA2MP6pmPOr"
    "ZEXH39bE845k9ifeDpyFQ90VeYgkhhrzOFFWxFjK0fMcTkR8qcnlJEe++5hD3eXN1QFXTgEFJXAGbycQ2R7uRn34p/HutNKr"
    "Mq0icxQRrdZmkEoruq+LHeepIZY7fCrGco9kLHdUgAQOpHSrdAS6qwazSjXGO2SYk5WlBT3kb5OVxduhWQkaH/R4ZHjegqxU"
    "LmoQKEoLRMdmNawQHdxRlZVNQvE1m3AREuxqa+zyIF5H6mBgNDAlzuOcfKcvP0KbcaHNAxHzIyXSKC3ElFdCqQRXeRner8uS"
    "Ib55W40YwnhlW6ohKsFD/PB9XeiQ6yAavlcCPkQsUq+teddJFCxS2hy6g3XZ4CdfCvpgddSwEhgReBzFM1tLhl5A9Y6hguBD"
    "YJE221NscDKiZQdxP3zf9nCcypi80J8x/ZPGQNM65lGigTToxYUW/vae3o9jb0YTUTJvBSBVHuH/ork3BA9dNahF7TC65rEy"
    "kKzBd9xF+F3wJn4+cSQZLNmEvZ1kRu7/9mw7/7dnfyT/t2dF/7dvJ0rPr5j328/szxP2533B8+39tp5vyTym3ui3cXp7sr3T"
    "2zd/MKe3b5qc3n6eFHGx3vdsT+QmKXqgceczjoYC9HlPrza5ZX1TJMFvPo873uuqE88Xn9GJ5wvFiedFrRPPsw1OPC9yqn72"
    "2Z14nv32TjwvJ87zieDnfEt2yZIn6C6fMgvxnnlKanbK+B5L8NBlcF3EPb5BG/oct8XqN/xdGzV61BaQ7HWKhfHiPLqh8TiI"
    "bh136o9GNHSrRYRU4LgxhSPfDXUNm4HAhckGcNjXZohQmqmRhylPGMlbx0za/oxGC5mmWmY6zXFSCqDtjdh7aZAIB5SEjqJb"
    "MUTMVZvQZ9FtiKhl8QFRQm9slUH00IbV6W0I1Q+WFS/C9+itiVoQfyE89F9P9NcTfRlHAVon0YSGqcds+PfBzqXNJnAfFjv3"
    "2wdJo/zMJad9DHuAPiflMKW6wQOVKoEAovCJkK+ehzwIavGdMlTywKwKBCLl+KEIcg7Bs0WcAbAk8OZzGo4up36ApzLWBg0c"
    "OCmzidkCqTLdZu0aWEv9/KNKHpgtvp4qpDxK5YCKCMtrYj5PhWqa0zhQc7iIIXTLW4jBk5K4MD5m+Bi3WjzvHkS0YBZNMcF9"
    "0eSCvzR3rM3ACfP/Gux32CwMQ91VKccV1oKVha0WIu7YCxLqGmQvWOgxBCeaw1+R8OCVl07NmXen/4TZHUYLHcMlc6SzAEyO"
    "/8id37lEvMXgTfxlxtIaLKAeicCuyZvQf7Q9M6Dj9NEoGqKQYl5Ho3szGcZRELyN5m0E4XEXbDOxwj/bnplG89ryL+k4bQes"
    "ggQsjeZOUoQKOnQi9q4Old98OCrTeEFFRm3OCR0wvmI/dfUyu9ViBn7VyfzgmTQyctSxgGeXuEGFcXMYW61hQL1YgKl+MkgD"
    "1ZUpvBlxpZLbjyMjbL00nG2oOQQWE9OwB4cP+QSxQ8NRXXzV4QLEJcHz0PfKQL8sts1lubSbGeT7iZOH9FykaRRuHdNzuIgT"
    "CD/Hz8j5h0VC43aCnFQ9MX94VE/BGW2NbS51qgIhFNRHycMRNofsrEThK5Zfp12wNWt+pyURnOvqqrI4p5Uei4FKeR32zrz/"
    "b6Pu9d26IKOs6JZRRlnhAI7zTcFG83xJdkHdkKNCbSovbY4WMa7I/zZIoSJiYMc6XH/zkKrRIkXSfVi3d21m8rOx4vpIt6xi"
    "Y6jbAow2QzkPCC3LTLx5obGJN68JVYjf7JGfwNFqpKxksVa5yVF9xbY/jMJ2DCSgVBXUvbGOPfJjPU4Do1q53VDbDqNUz9u6"
    "AQ/2oRcYpfZLxfIujTV9rgX4OkrTaFat1F1XKQqD+4b4uGxSoJQJpaqBHfnibXNOigGcCzq8UgFQ76qfcf47m6DLX+HSLJDA"
    "utC60ypXaBjLGszk75CjKF3f+Il/7Qeotixzbo7FjUOzbW+cFqK1ym1i/193Tzr7Fx+ju7WhURlimmKj1nHyZNbEzJUvd424"
    "TWYlTDwIuJpmdyXsZLY1PdQ2vo4ua2hvh+7WRozl+9tk+wkLJk0Tpny5WwfNjhNWB1xNs7tOWDDZAYMfPmE7dCd4vIqZgN6J"
    "yyIQ4YZRsJiFTeCWeMvOUk2lxXGw8EdVbOa3RPU1PtLmULslyi3csKcg9K4LFF2UM7H4hwiqvIEGcbVZlmb1muJO1w6tcmm2"
    "YWys/IcMTrSw8+h4xfXDG0fDRdLG3adwi5fLkbVtYzUTL1FYqRrZsCTdywqclBq+4lG2dIiodlpBg5BHo/EYbIYbGmdfm5BR"
    "lyKiHAmbi8UsNPi6dAHrCiKkW5WuLOPY85OSiFQ7UaxcaXKqrW2SuNQWWdn6I0F1S90mFwHfOTckI1i7SpTKWy+QzckMSmN4"
    "IH/L+3kwh6s08YBBbsHlmoe6Lb/Lu3s4x6u28YDRbsP1lOE28b/S+ba2q3X8qGFZ1lbesEi3SKrBG1+bVWM96kXV7TG+ISdH"
    "AfaHLiDex8OXT7GBnQe31dKpH+LWC4d39QHLptTCzqPcbsnwYT5gwfBuHrJcqlU3LJZNWWC4XNucBmYtqlm9rRG8NoNMDu8D"
    "Vwe2/uClodbebUDbLIq6YW27IrCThy+HQvXdRrbVQsCh7b4KsIMHLIFyvQ30vyklEWt0TU6itbhl9bZG6tp0Rjm8D6R/bP3B"
    "9K/W3m1A29B/3bC2pX/s5OH0X6i+28i2on8c2u70jx08gP7L9TbQP1h4bkYxlHoIblm9rZGKxddjE4o8kP6x9QfTv1p7twFt"
    "Q/91w9qW/rGTh9N/ofpuI9uK/nFou9M/dvAA+i/X20D/m/PDsWbXJohbi19Rc2vUbsgup8L9wJXAe3jwWijW33Vg26yH+uFt"
    "uyJ4Rw9fE6UGdh3hVuuCD3H3lcE7ecDaqNbcsDq2SWLImt6QxXAttvO6WyN6YxLEIvwPXCeylwevlHILuw9wm9XSNMxt14vs"
    "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"
    "rZ/9zeD89cG5m4OPPlr7+vV77/5l7bsPN357YeP9V+69+5d771/42/LJwR9Prv/u4ua1jzc+eO/e8ltrt76kr+6vnMOZ3Xj/"
    "lfVzlwenb65fvHnv5e82r3668enNe+/+ZfDmtc1ffrO28v7gtd8PLv+JenB/5dzGrd+s/+7i2o0rayvvb177y/p7b2CvFdCG"
    "BmXjg/fWbry28f4rg2tfb35+6f7KubUb59e/ujV47fdrt26tffPbwaunBle/xi8TnM3g1Bn6/P7KucGr5ze/ubr+ny/TV1Th"
    "4C1ZIdR89tr6xf8Agj74y+DM9fW3v127cX7txutrN26uX/nD/ZUz6xeXB5ffB0ABW7vx2f2Vs9hagq9Zu3Vr8Nql9Sv/Nrj4"
    "yeZ3b21eOve35ZOb177auHJ28O2pwdVz62feWj/zzuDNd9a/PDN4643Bq+fXP/jL+ttfb353Ye3mB5tXv9345irWiYCae//+"
    "3sbNP2lQmvV3X9/85j/+tvwyUarANIOvPl775uLgjY823no1gdLQqKaxNIPr769f/IxGJA+dubt8cfP0Z4PXPqECd5c/XD93"
    "duPiR4Orv9PZMzMTf1s+eXf5IjHs3eUP773/5uCtN2iIRw8odZRQMvdOn19/57oiLAOPWX/394PfnB/cfDsHjFl/9/f3Ln29"
    "8cHVHChm89qtwZvvrN24UgSJoXHUIDHUbZjhKx/f++13gIi5v3JWx8Ss3TpFoA4qkUHFUO/X3/2SOEQymOKo9d9eT6NhNr/9"
    "FobjyuX4PTDDyxfXr/yBarq/8sHGB+8Nvvoz1bq28v76m28Nzp6nMn9bPgkr8tObgzMXZMu//cvajc82PnhvKJPG6Je17z5c"
    "P/dyRlCsX/xw8NavspiXmB/uLn8Iq/b6h5vXfrt+dnn94tnNP/16cPOPNPWZobm7/OHm1U8H3/xmcPY8CZe1G2+QrBlc+/re"
    "N2+BKMGGiRR6KKsi4pCau8sfDt78bONPtwavfaJk1snBW2+s3bi88afzm7/8Btnp9bi2zW9/Obj+K+ysQq/QQsuhV2iB0ksd"
    "vzL46s9r3304uPJePHMKwzK4+Im+Sa3duAISb2U5nkKFYhlcPj848+XmV9cG374SA1junT6/ee23OmqFKKDnGm5Fl+zr587q"
    "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"
    "0W0WUHotXV+KlCDHqGmfMuWT2YhLdrZXGO5PtUnbUvQTs5uXjXNLTmfVdr9bYFqNo1KbJPupy6IyGpjDI99nqHqB7pkV9VJW"
    "ayulusosmrwSaS367dG0ubP0/OjczW57y5qY60v6rHlkxJhIqcZZLSzZZbYgqvYuU8uI+cPO6ha4S/qyHZzSVkOqd9LTWIOj"
    "lcssMVvMTXGdTByXnao+HaUH8XqjTrWZnjY5UxQVO3a2iUF7PhnNYmL6Yub6TWtJt+eHo3FWu9oy2x5nVp1ZXqJT3DTeiYls"
    "JrcrH2ogLh0KBl+wbDUp5zu9UXfHCMpwGlusKxRTEWcdtbbeJsuDPJU4VoapQXorttu6RolGTR4d50AbMcPCzD4OjXwka60X"
    "kfI6XhjE92uVkbVZvtnsG8rwuKLlZLOja1bEkMoxluvVFqxk1bh+IdHhk8tdLL0YxceH2emkj1a1+pyj++tsak7lds1sbBgp"
    "tDONgrxNcpdqtdMYpayZuJiytYnUiE8O+V61zm5zvS1/vCSr1RbVTXXk2KyusFaL66UGVJOaxxr5yHy+bbFslltEaoVhvCkA"
    "ethp53LnOc/HW418s9WQaom5aBXWich21AGmrDQmEXM3kcYdvXs5gdxMO6q1RFbaTy3mUk1zg8YxFtnbvbRpC8NF7TRd9qqy"
    "WhaH2nm93moJcZvKZBfrTHlWrc1qst1j0zEz31PYylbKV4Tc9JhUs/t8oWf16Qug5pXYvLWzmL6W7Qji4tIfULaxPE14jZtK"
    "cbaeHOV1Pq7V6+M9SC/paV3i0uvhLM4lmZraikkJIK935XKzP19qfY69TNYzarG+HBOV40gr7JODgzxrVqlFNzKsj8aMtG0o"
    "uTQ4dQdbNpXgtXO/29S4jN5rxbbp48y0Jtu8QmdrwlJO8aZeBztzN+2bfIeXc2Vuv1zPmN4xHZGXSzlNtc9xurmsa431oXpo"
    "1XZSpmlmqcv4qKf7zLnFJKrNyziVn2Ybw2pPOnd0LX+6SMv5dNcDvW03098p1czMiKQqtjHil1xFSo0nk2x3lik09iwVa2WM"
    "5LafaudNJVOI9U611WC+MNeUcu7IcoeKWMAuxNfxY3o6XEitYWSQScUrmaNiLzXzmN/nc9TYSIw6bDmbXVYL23mq0rciu0b2"
    "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"
    "MH891FMV9TMZ+lyhDZbmRNpN6AFFUslQRVUMVaINMuSVc+pDt4zPRjGR0M7BJEMuJpK3STJXTKRukyS+mMjcJp2lYjJ+m5SE"
    "afnbtNRZKqb9fEaimPZfkkU/t5EKAmKki4ms/5YJtmVki8lAJfliyi+n05xoGbBXqbu0QLtOJokvZu/TzlIAJBRrWJWC9TtJ"
    "sIFkAD4Ykfv0jIyni/FQQjuHkto5BI3z4H/cbTb1CHSJtmHGOMyZ97MmC07WLW2YxWgiaTjv0A11MZrIu+8AvrMWI7LPDLiI"
    "QA9Hk2Qojv6fIK4oxOczNGl7ewgh/J+p04oBY5Mr5p+GL5e6/l9w0T9vaRa8OU+yKNlFZxN8MXS2aOlSGPnRhhliJ3W7Tbp+"
    "tLl4odHjqTJyU1cf7qlyCz0mdPjez1LUpDyhPv6vse8x2fqJ3Tep2KS+Ls/UnlqWqdgsMS7P6zwlVFbHabnWqqgNod9qVBq7"
    "qr2maklKHFXF8qkrlCmqXKHqwpRvtsrH2cmeG3J62Wd4Vo5ZCRrsKHvfA8KI1Wt2jSoslpdzexuL9WeNfm95aTVXFW0wjgin"
    "xqQzaU8aMyNi57Lqwchwg0533u/kTY5bjSL7XIFJMQmqR50afPV4ZhSrZ42y2V01uY1MMnNrSOXP6qJp7vdAWSqTpn4ZVczB"
    "MTI5cn0xZRSG67kqUOpcPNQjicQl2Zzs9JFQSbfW3Y7Ub9IDe7ocdloZ+ZLMKe1UuVurVvdGVZs0VuXTTlZHCXE8sWfaXi3U"
    "6cE+Mz/SXFYfRTLrSY+9HAQJ9PrpAd1kI81ILLXTL8l1qt6vSDX7sNvzk0piIEzWc66+UOOW0FXT+ePJ6KUi6jSljFqX2knO"
    "Gqo+0+qj2Zg6zc41qUmd951eVl3oVl+nk41MrD4HZ6t75GJ88qxv002zIpUnElUbV+u5C0jpKYFdp8zs8KKk5hy3Zy6jQiYi"
    "RyY9vTywJaG+kGO5tKXktkNNWSbr/cx8naiVU4k+s1xmtyxIZY+xdV/MXfJcftWfjlcju7JYlRvdWqNCjzKJwjHZa/eO8mBQ"
    "Pa2OK46SADUGiTErRNROS8oJwxbFthpTZtChmNGMqi37uUXn0GDMxWlIjVK8TSUnl/Sots0phRlTzm6ny3nTjqWXJz7R2M+a"
    "cnNQadmtGuBWcbve6/f69VMyE8vEc8rU5CZH+qxN6NWe3kfyQ6u771CnLtXpjyvUpT1Qy3q8VS+cUwwnntmVcbLp2aLSTllc"
    "LCMw9qRyiPU7ElXrs3wntSqI5Uyqf9jFUuYkHhvlR/0VJR5GgtRpluXsSGiPJ73sWaIFun3pHFP55fY8rFLqtFVelpvdSRNU"
    "a5REde2TVbUGFHeyTvRgODus7Pq8u8+2hvvsfJ2Rpi1ulDEnjJXoSIuhouq2xXUEdl7JHhL72Ty9jcUKcdNkUuvTTNHq8165"
    "sl5mZkvQZQvDVFuJ2axWGx0Pp4My6uZ7M7u8nHOJuTRfCrXyvLuXeX3dSmpspxvZxlg5l9lPxdMsfzokzu2UfZFHRkxflRuF"
    "QSQSKcQWknwQKlZ7HBF6nVV11DGm9UXsdFZzZi5u9yuzyGG7PVe33Uahkc5N5nUhe7xUQCLe6I15oVqn0yN9t+6nIkKusY2t"
    "G5mMLtnHJC1XWmJvZ1djw3rs2G1V+W2Pr0Etzok5DfltWSnvL+VCPZHLA9Y8MmLCjDXZWlm12+nFxWR6ZWpcteutbDq/WNPl"
    "g9mUzjGunDWUeoRJngaHTNrqNU8jtUzVM6ampJJ5utfkRzQ14sa15fFizTODRm2RTbW4hUitxvXJcMHZWZ7tKNOWkGqdDrTC"
    "x60CPd9Od8d0fN4aLXYyV0tc1Fw5vxJBbtxqN4+DxdbkslzTWptWLM5YtnVYn5l8huPsxKpSmfWOZSUixarHskQlmKlkp/OR"
    "ZXN5qqxbveZ+pI46NTtT0Sq7cdoYd/cVsT3SO1pPXnb3E+gzvVMXhZE+o3V2pBqGaY4ac1rl4gDERTubGdHlY2VV6dqnRtqk"
    "dpREN7XzsZWT4tUu1aNqfK3H861J5ahvE4NjPFWZq53RkKoPJo1abb9K1yaZw3Jc7lebjc7a5svtdme/rPDDTL1PtWaz2Szb"
    "slVmJBupmV4+VEC/NbIv8kKwurn6sma3p+JsPs6mjfQ5Tq0LfDNdplqtTqO30+tbLlbfUr2xWaichza1F6RqtSJOdlRsvTiW"
    "L81aeTSXOtvzDMxGkfJ4PBlXa9JC1Gc9U7CXmdXhYiW5HWjQ80xikbTkvXWytmJEo2KTFWtmWlK/U+gmRbnSHw3lSe1cyFUq"
    "cbVXXQuLozZc9MenQdfIyIvEOb6Qu1IisWbVfmo55ezewm7m86ahTux6fzheKD3NWser3Qqdso252O5TI4Fa8jWK59WVPRhP"
    "9yPqfKpQk9HM7EWqoFWutlqVGm9m9p2FKJg8pongXK4q3dbaVuv2aJzLdPhKvV6v98TZod7Zc+fRJHKcsfW9PqmmOtrwuNDq"
    "29Ywb1T2Qnt87C3rfaF9ucQYty2uoQ4Vik2XqXar0xCNS6oyprYUL+mitUYtVik11+pN90lr2jHNdWHC9lPi3ujK9F48SWb+"
    "ZMYKUkPbSpNc61RtldX0sMpIVGp3Tg+5XdcSG51MrlI7J7vGrFuo9IcMk5uZ6ZSWHyZyYj4xGdeVZH/M1Osgv992T+eOKu7K"
    "fbaQn4yHtlGb5MfpnCHxy3NsUhVy5cFkmD8tkxKfmq5GebGrzNO1duvSGdL55KI2ZtJ75WCo3Hx3ESqcsdgLAn1W+X4kI8Xa"
    "3WS+S0cqitgdrcG8v+hIy/yokZnEDpn5wOIvi32mOll31UamLh1TlYlgXZJ93e6JteReOHdbk5wWaXIxtVABq7m15ri+vYqs"
    "2FOroO9HoLysc4typS/bkZjV6VRzar1uAK7Cn5NGnGuZZreVU7ZHbt1NJXbrynK+75tyLtuo6fP5+ZjtjYEq2nLhWE3E9HJ9"
    "pmnx9aBbUTkrtxyMM+m8fjIrx8hyNUvp3elgcu5JR02c0Z1CVRp2D4aZy/M5IdttbFuRfKe+lxJzq1CJJXfHSDKVMZb0RRH0"
    "/mJ37jbKczsZV5NVTk+umEtKGw1XE1ksnOjCMNE46v3McB/blvORMW3H09V+bHSZltOCkeraze4ozVQaw6ZmFxI7pszk6Ew3"
    "XlHVtTbNzdVMOVJrj+1z5ZjVumuQiicu226+ddoe5PgwI1n6fpHNVjNZfWcO6G21cDzGClSnfokPEvyW701qauI8bVRqkr6U"
    "D/Zc5UfNkTCZdXvmcCXV11bqkozEhpzU5bldrrfNrlqRIWCESd7iZpM+szh0cmW2HU+v9RgbT+Z2VHVUOS8nQnxirBqWUW3m"
    "GbuRmXe7q8w2fWQrVk7Zc6dYZpwaRQbjRsqONM09k7fs1HShJbrbrZztbHf0tB47LSPjrTUw1rN0XqlyXEs/RhJU0mxWQV6t"
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
