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

PANEL_HTML_SHA256 = "d4a79655277d14f7176da2c1357809785949d7489553cd0629354680fda4ddf2"
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
    "v7m9TeBCpieDMGi1tkQaMPyFCYHIdiUR2bat8+IvwxjYkYR46YT6imeLpEarJRAqaODGJy8YAw3ip/oRM55zdQE4LyEzeoBc"
    "9/HaFZI1v72zgJQmlyzykpsULiiTD4YCxFMRwVf+q/CNJxXCj/yn+hW0K3yCvyVS8JMKK/fTAF7+LpXJQNRy8RTp9n5jKFdT"
    "1/cL3/2x+t2HRMmF75hXjRLswcfw2+jz65+9YAw9atMy21EIKMEkoXVyVy9qohSIk9oDKFA7i0R2fQqkYH8jHfBDpQCe8eKc"
    "52NxZRt96MbsoRgdfMzE9g8TglTYC45hAQrC5oNNXgFtQz70lDdjxfeXYVwPQiHzN+VAp5hJn4OdCTiua/sllDgqltWJABvi"
    "eyMkt4HQbMuEZr0rIB+Goc/cIAPdsor9X4tPZLMmUgUsweWuagR9+O7t2xfPKYBiav9lhIH50fw5XsoEzi0+iJcgEQN5CQRu"
    "vTZ/zs4wLSEUclIuIVKM+mUgOHAbqkCIVEwWz8YwGC69Iaveeb7YmQ344MFLGsgON8OeyDtH4WWaprl59XI98+rlX8m8elk0"
    "rx5XzasnX9G8eqKYV09rzauXK8yrp7l59fKrm1cv/3jz6nU+d5Tmjedj4gUTUCFwj0AMF6R4AdBpw0B5zEdcaZBlF9Hg9dz8"
    "AuO4Rkh1nIDxMpOevNzEs+UvQ2dkgdmVQbXItRX/LJ/SFG5eGIhrPRSoyrDW8ttV8uK3t3VvgSgqPgm1n0o98iDy3Kc4gTbU"
    "gLNrL/95j9/wgBNSj2r0Xv5oU2rYKgGuRJ4SYTw21QKTjCQ+9sOEv0nxrta4ptA7ToUbYyfy+yNcy7pRHtW+ym6YgQnoQQNO"
    "gRG4JaarHle9mWXe0GfZVJlRUKOF+e09yEJxW065B/ILdUByt/Nq5b0+ssLtYv3Zdzk1iu+Z1Yvc55JdbfVPA85KnGuM3ha/"
    "sSMbCvm3lEA1L2tQXPv64/Gy/xbiLiCYEvNfBqHZWkPcnaMsN+AxX3Hk8icfUOIMQoVACwTiySBUGr0CD7d7Cc2seX73VGbQ"
    "yxUAfy0XAbnFzq/2yYx2fGyw2/mSgN+DgquCMrBcGOBaRdCNK5kyIAIYhMo1FSV6xqbCsqr8Nl9Z5YwD4YceBI4u7UXZWwJf"
    "4cGo7UiBGaV4LdRfXhCk2sR1bmwv3sy8IGCRfT0hD1E37UxSkt2cCmtqfkHXo3Gakt/Gzpm42YX2yUP1KTeL3guboHCNr/M6"
    "0qmoixokxmeskBpk4LyPsivisqmbmcIG7j3WmdB2hAplQMmCX9c1ZnApTXY1l6HoFjfXdCkRV6WJP5Zhvw50/alukN/0h5EO"
    "dRnknb74yG7sjrgJTMUrdce27Zo4vFPimqh0DZLNgs/1Spee1dOekiokk4Oy0F7xjq8Y3+mi1WecNmVtQvEKoGsdJYTCgpHx"
    "K8MKMMayegGkhzx5LW9ZK3FkZbWkgKyRCDAXiHVgGLboglDWZK1bU5/EMy+wKV1RBzmjiJH2DWCK6P+OYatMrudumZPeEhZ6"
    "6/Iu4xP8EH9wedsvixjqn0rlslm/jQ3D/k6neJFh3jALuwBNkVpqugXZ5j4EI2chfyaLEilcYaUG+agL7AaxDslDlQbChK0j"
    "ujQmWOEu4aa++EqUUZsv7Y18tSpeyyU8mYsLsWxmyp9kltiC4NlA6JCUM+aInFGsjxLR7YRK1JRQiYESOktAIhQOwn/k3N5N"
    "VaEhmX4zDHJ2Nuj3jfT12IwYbA0478e4Lvgus6HhOq+Yghns2ovEHcf2mXxHB34YI1mwmPQBM/MZXPNO+bV3tJ+S+HJsL66n"
    "fhDbdJIkM/v+/aurK/Nq1wyj8f1up9O5H1+OKeFZzLsdIhKPdzvk0mNXD8Nrm3a0jtaF/ygZeb5v86vtUhKEQxbbZ2d05iYT"
    "ShZDmz6z9sw96/hY2zP3reOD7qm1Zx5a3SNtz+x2D44PNGvftI4OD/jz0T487x119nP4ffPwcNfS9swjq7uH8Pi8b3aPDne7"
    "Gfy+uX/UsQ6eWpbZOegcaFbnqfwkSBCoNE7BscDUFRQcZpjE31PlPYcTlOdosGEC/KnV0UTdTzktexkmTqtWbMq+8gxNzaA5"
    "J7pFzhzn9UtwXv/TI/N493jvGBosvghWcESHB5xV3UOO6HCX1y85nDGaUyU65vBQlBLVHx5pslECHtqLVe9bT0t9/LMUC3Fd"
    "3CnsHlECQ5WGV0ds4NO030/Ju7HzUjrc33IRLi8Hz88LUh+zZD6DNZ7PkgWuYZLUuZjo343llK4zEhjOv4SCeAbXVp4z/dQF"
    "TcznUmVqAM/mj5mFEScujhV8APvFHSQwdEK4ypXikuFiHt/QPvkWysTRgPbJD2OnMKnQcO614cbrm/bUjcD9/WnsLCrf0OxM"
    "yffFb3EShcGYpuSDwNpRscKQpyn5SXzbVb9F4RVNyc/IUMmzF3PvEVRGpXM28RKf2QvpUJPO2JQkYYDvvak7xh/ICvgh2FD0"
    "e4GrS+kILXEW2Cr7pylh4EywkwkJRyO81SGepLJrAhLJrnmt06F3mVlYOWfkCgsp2LaZiT96PVr4KsiC7+Ln7a2YnfC2VlEM"
    "zG8kiPao67MoofI62+2sO7dVYLmAkdhSAvMucAXcWgi2pYBl8583HVNS7THe11iekjga2AIZccEPK3Ustfb2qFSz/GEYueMx"
    "eqXEFYrZ3PMtzIc1FJem4h/G5Mw0zeis07+9dc4eyvdVSQQTR6JvW0a/n1tMj6dETOiJB6RQLig1zOXRTlT9JPqf9mjA5knk"
    "+vAVyvd61B0McAcM5U8YNr3krNzZfe63zfq4Zx3Z3a4yDQM6Kk0rCsA4iT4UwvVpTKDlM0q+H6NZgQPA4AaM3CQSLg5hN2ds"
    "nFHyYUzOHutBrWmtG86/zt66OmJl1wkg7RuF2d4gGXIXbbm4VxD9n0roBRAtYiFH5Mcx11SvMk018ufxRGoqULW0T3ayr8iV"
    "PmHnBd0ycwPmt5EDNCXJeY3a4iCgZjRF3wQCtFsFlSSnJKqrDK52hd2a5rpGsPBJiXte0lsv4WtVb5X1FTeC0DcfT0o6ik9B"
    "jXpLui/z2QLGDr+eqThMZpyW7ZzzoDLwR6Z0trOewG8eXFmvaBBsAGzbCnkoP5flY8LcIYvqNApnG3yXFAEvtitDiyuubMGQ"
    "j+WOOuj4+LL2y+Mb/lRrxWGmjD4x7Li4qVb6Q51OupSwcxDwJBNwLnuV0SPHJAwfwkzoU4U18FjSbUkJL8ppGS28rA5Kkiwd"
    "kkEJc+OQ3BmX2ovFo1JxqTCgYVnFIPIsyurlj5R4pbLifaXmV0IZhOdyuOeWCj5OWBTK33Hi+qAK4uLo/DW84CKUksF5jaEB"
    "35PJfHoBm5fVoh4UnDcVBBU+B5UwBAp5n/fJqE4HAPiQJa7no74J5lOaklm1QtyvYQrMNFNJxXvY+bXNKZlA1bmPV1pvuGoL"
    "wivJHi8YsoRFUy8Aw69PxsvaBE06bwLIFeFlWZV9H15kigxfLlJ4HEdwSkv5zR3/aHLNA7TOOGuarTACXV1+h12+ylyDGJUf"
    "dRguYqsJjo9yMuCecty+oa0W38cxvfixF3gJ03MouMZaYICQCX/Oetn14h3Cf3qBbnU6RC1kdzIzPSHeMlvwV+DaSltvO5N4"
    "+Ap/S5+QG8LCA42b6eSHxTrHTD/LOvPCDwcfKVlswwu4SNUNhhJ/q7UlkKV9Q8ETn5OzXL3D2ClqlkFZF+LwMvp1asRXcfHh"
    "VEQ2LyETQ66CbVbhKb8qm/AJlZnwlKJlBL8MckyG56CDueSpttDoHMH4B9CrBUNHUj47h2kHBVgWfiKmnZRIzYyfUTWTgz3D"
    "sKUAFQtYsgAKE+4N6BGHNNCC87hpm1m2OC3Rv1E0Yjnm4rQxPSf0//u//m9q8JnmYb3UtS/ciPJFBJWCi68Ke0bMLAzbXo9z"
    "UAAhlVMPFFKHFt6519s2tTrFt0F4tW0XhlGxuXZBpAtKa9veEgWLsi2MfR6QMGS6xL7gKw6B+R79G5UbZtK7tYezzORcmbHK"
    "pjGybdw816kTVnfVlHu+7pQbnvOJ7+a8dr0udmpon1zUTTV8iLSHYUIb5ozr89JSHAuICCjypqzW3/AxJzV7tmD2pRrPQkpq"
    "Vst4Ys6f1C2WhQLfmZIZCzBsaGfasICuruiEIqhdQouXgix4LX72ehTdoWAzZmuwfoPdmAMI23G3bBD2KwPv4pyvyfjT9fl6"
    "qyrpAsbRCgJwIwTgtCAA85j2yVWx7yL2iRspb6vvUQGm5NF5s4PlmzoBylZFgISrQZqSF3VrI6weInp8APkoa0rJA6B8ErER"
    "7ZNfq6TF3hBIe4btE/vJtE+el0XvNfs09yI2xSV0ybK4mtxI/808zn9lxkVuUwgK30pXEH98F6kfc5tkwgYfi0+ViLY035Iv"
    "mx9s6iXgp8aClNBwxgIwS/mwIAv4bicpHx+Bw0jkJMRzFiOMaafhR0qmXhxDlRTGDs1JoOiSI/PgYxBeBTb1hri+dZ2trceu"
    "bvTMS3BK8yDJnhkGL2YsyPcRQxiUgZk3vtVyWy0d4NklC5JvuHDqBol0TjVRoY08DKawjqxMLBH7pI7KeSyH5TxWjIirc+mv"
    "eHuuzMrorria3Kjz8SMOcDW5KU7GtZP3N82TN5FkgMtG8JgiBxT5uL1laqNV1C9wxldgS2u1j7xqBaBSf445K+uKpa9FYLTY"
    "KgxJIKQ2sen5he/Cln3EcP8BOodFWhBGbMSiCJZUpeHFd/hhJ8wbfLTDlJy9R+2Geu2ZHk6MZpVGhFJSxwVM+gpX0KA5Ig/O"
    "C6ZRrZn06znU/eaciGnDO5O90E/VnV5VHSrjmCtFcg42EtDI50GOX4aplIJlSqwQ4zAbr0wJWmKmHFvwOxcO+TZnYcAtMOb8"
    "K9LF0DbKXB1kXN2r4WrReEPjMFM0kqHPzoX+PxX6/+W53Apzg7HP2sPwKijth6kfwIGKW2H8acgGEXNxl8wPr1j0h2yJdc3j"
    "g+ND7dA8tI4Pd0+tXbN70D3QDs297sHRnmbtmoe7h3v5857Z2e3s5vB75m53t6sdmR2re4AbQ+J57+hw9yiDPzIPjzrdo6dW"
    "x9yHZ6trdo86u6dWx+wCfqtr7sPG1rF5eIB4sue9g+PDDP7pPpKb4TvdNw8OgS5ZX/7M6ZHwkt4DbF43a88BNG/3KHs+BHK7"
    "RxIed9I65sHR8f7TEquW7SxFB+50L+A7S4/P852lByAM32DvN+0uqfLStMX08vzOW0xPShI7n9XKK7ymkcdl0xuBRwdfeYEQ"
    "2T9AVI/N/e7BLvD/4ODw+OD02Dzqdo73sesOD/ahn7r7+/B9b2/f6mpS1Hh/dZ9KUbQss2sdH2aia1nmvtU90JTn46PDXNQz"
    "URVDQYqmHCrZs5APKapWRzs2d63O/t5TKVUSlZQ6WVRKpXyWUivhpVRL0pRnJD2D5017KkeNbHqJdctEN/6te9kNueg+LYvu"
    "u9kqwQUpaRLbJ3cX29fn6pmPJ8FsnuShu8qXR0MeQf1d6KPXWjkVIg2/+WwIE900HDL/vYidwCVoezCBrSPal4ZsDiKsSCSp"
    "fA5CPK+E42bx0hMVwaXrNwb2Z/XVWraFIwajMJqewgntUO468ND0Mq5UCXEuBv8vyrVC4GcYTRs/PPaYP6x85fF/NVGvw3Me"
    "JqMGvCIHHRlDzkOLlS6ogJLhOTRU2MUPZjPfY0Nof0qu3GQwKXbgkLEZxNhP3GDow2VfSpQ2R81Jclha6tMlkGQHKHiOPetN"
    "p2zogWg11IJx+yQ44aGzkmeOridOFlsbRlPDcUB+bm/1wEnMiI29OGGReNvjzLFFZH1CRLjkjtLlhnF7u0hTQVwmB38MfTl1"
    "wCXCisR9Uxg7KyncEuyvdnur9UlnRqulN0I4Wx2i9t6W47AcXnQohosiZVWSrGamnZRZtIXMaLUSiDZHNoKPhulFbhj8HIpK"
    "QpUi4NUoH2DKoQM4LcLKIgqHO/DwBol4Rw54//vAoxJu0TmgGPU6tQjeTRWkoCUJM4geOHrkFOXFMMPgFEEyQRHCEJFFGHlj"
    "L3D9R7CgtRMZK5casEUaDJ+HAUaJVBUGBBC40XgOjoa4bwKw/ql4JGIHprTaExGfCu02UrIjNa0CC0zDOF5H6aQQwSAOQMg1"
    "b6tCqq6zQtdzHmRDhDnM3Bkp7yQiWK8hctI4vBJnhfiUUCYSZRZSXKWWB1GLmgNDVWBNJ0mWMAKmNHk6Q9E1PeHML+ifBvQK"
    "iMSkMLKnfpfEZrQsJTmnJYOvUzpVCVjBPCFFpIZW2JnyXD56SEkuiiKhQsZlyTgrdnP/S0pINgJyHbcJB+4s7Jxnv0NDcvVW"
    "aQjWRRIne59kYiJO6mVH/+AgwYurQJ7ZE8fWzB0PX8UkMdKMKTpVdKRxe7tVegPnD3SqWhAUlbj0zq84/rnybOcaB0Q3tOuK"
    "Z9sy001RrbLKhgGXee9F70belF3O8ezlyBubHpjr7zkMiF7DiaP8SFGvfNwI+PdbdQnwFk/USBv/9bk49VpOoID189M3HIDK"
    "zCbZFyWDySgMkvYVX6piDpIMyIRPJv+kpsHBEtivdfDwoSGfdg5ZvfyylJknB61PxTNzh3hAHzeMy0XER/OmroQX4HZPQ5Hr"
    "1deOKqQ1pBDHLC8eZpnJ3hUbWUKUFzCHc56f5u8GKZTFSjYvpqaf2bx04UbTu1QubzRdp2zOP3c2Y26EA1gDB0klm07kDr15"
    "bNd3B/+o4itdzNqQp6d4+2qOtumy1QzCZgFfoVayJFVS0edYm2/lXVYD3gG7Zg38vtgmGW1ub+M9sworl5RB99ffjSVIQSct"
    "A6jQKvsvHI1iliypmwMsYST+RvtoTSYK6HU7ypzJXd42N8+XJB9TWoCgNZpunRqapW9VfU252zaqtSKRq2oVYrlOrVniLfWS"
    "5Zk7gINEmrXO5CERNMwitV2elVk5Ku2Z7w7YBH1vCom1WBXQdaRIrtg2qULK6iZVteNp2Raom9njaf3kvnQajqd3mYmVUtfL"
    "CPfH6xDuj+9AuD++C+FKqaWEo8mn0M73CzSr0/lbluTtc3Ne5YZeIe0VpLfiw3DbTkzhQYCEUgV5EI+4smnHUwiQKiZdKpbw"
    "x8US/lgtwdMw8RIonlizXJJvl/UKflWOfvO3pRqRg7wFaC33U/CZL838ktnPayZ/eXi+VvKXh+d/oeQv2OY8d8r780ryl+/O"
    "v17yl+/Oc2l/d16X/OXh+fLkL1iKJ3/Bpn3V5C8Pz//w5C8/ZivImtXjb+eVna3M0xoGWKC6E5C7YOH0AEaDCFcHKzonV+Qr"
    "gXww4ui9kgpGWBbCFcJ1U2N6GFJywy5Ny/H+XF/IGZWjlwpolGf34MqEFKioaqK0IfNDmpJvIXpMHk/m0b55MJmIgxf1KieP"
    "c53zw3kxs4KgX55z45uA73QRX8IX+DWpDOSBbOm5QxEQQbI1YSciJjPTzixjTimgO0+OQKSAJDwMpcpz1xQgrVb203RnM/9G"
    "d0nm2AaZydIsyJPv354b6Y/n8gz2D+c4PD+dO7VeDhEvX+vn4N+UiXboxTPfvbE1Pn+3Rz67Vib3UCzZtYj5Loy52imcY5VJ"
    "JWuQF7G6vjcO2l7CprGtwfEgFuUff53HiTe6wRTYsEFQAQBctgZBA+48CRW7dR7FYP/NQq9YoNYS5iSbIrfsepZwsUxlRSZs"
    "lgZwsQS8m0NEoNrcJbJ+wXqnyPrlG9wiyxCslqW6xY7s57z3l2LYMJt3Tc8tS+e9XESWpm1eTevKVN01VS7N1b2c2vrcynXk"
    "xok7+MiGK8d/k/rIh7GlWaVhLP2LHcXLdQcq2tD62WpdBE9tkfsHCB2E/nwa1GlA9yIO/XmiNMELYpbw1RIQEyXoFS1/F4qV"
    "QacVPotQKG3g+gM4mvQ3ra11Z9dKf/2Gpyuus3X9RnzAMC2YUmr6hS/JbK1T8SJiO4DYzJ+I5DXpzaJ3UWtDC+8iOhgGuTGx"
    "QOZXJxV/V+kr9PMyQqQv8B6OQzDWTFwjqgvlpXRNwsj7LQwS1185AJc462u5s8bYX1X7ug7fWh375aio6s51yJDa8wvQsWT4"
    "iU7ZXWPwrSfMKwfGHdEIkRa6bbVGXsaN4qARLOg27mJ07lbPEj0ih0ID1/m/X4DvXwSRqkzq+T7yQzfhp7smbqw3csjQypc+"
    "TN1o7AWlzl3Haq3vDJmAUPU8LplUVyD5XTVaYTppKCcpkep4zcb/gbpwBQ1fSRM2UbFaD/6uI/LzVWrFFa1g57bfklG6Hofu"
    "phuzC1xwPV1cKG9GwWqtufv7qbrPnatW98/a81e5J+rWKPUl19i/aC62ggqJazWmOwrcesQWphmxLWI06Gt4OyjIUWkzSjF7"
    "m7bRFHfKGtATJVhnBfiarfPHX6J1TXtt9a1rgm5oXR14vXgMfOZGZaqXrW2TcGZr++qQEmYDfsD1TdvS/sFH6yAMBA3afa1r"
    "rOGTqy6NmxdDtf4LBRSrXz1D1ZlJOVuMOy/z9GbKtX9oXbnQU5h0l1VnbQdWufi7LjwFab8XI1eS/YU53byK+NrM/jJb28K3"
    "r2xuyz3oqwiixyK+192w5awCVvfDIYDf9f3wCmOAHcfZsqqFIIgFyuAPZQ+9KD64ET4Jr/jlCzFc8MR59tS9CediCw8haRlD"
    "3mNrIMmBK3jye6hWYsnTq2/XTLiF/Xai7qjTmkEA3czcSF6rVC9xlHASvgX/ZRlM9W3CKTxhtokrXD5HbPIbwZYZ71yC8rsG"
    "KuybutdbTnbwZepePwQfdHSjG8CgzM783QguW7KrCfYClWAvUAmGLdXvqyHHz8UGmxJ0LAK7YZy7jXcDdVKi1L7k7J4qhE0X"
    "OokhkpalQL03qja4vNQJK+Ez/Kuvg8pQrwb1w0GeGGwpzDM4x8ei5aAQfN/AqSEbeFMXUp/MIjbyrpe1NZ6PVkDw07KDm+Xk"
    "SKhvxKbDUuB5zHCwL7mTC1roBY8jno3oG28Mx1g5LBfGKkPc603AcavIC8bPmvk4cf3Ro+uZGwwpEadpwqgmIOeMDpgHGQ5G"
    "fggniikThSCzUTCgBDGdchj4+VjAqRXgw1sFHk6L0b7pBQN/PmSxjrEVUy+obxbvq6l7vexznLBZ/Xe4tiyb7Jb0ycQbT3yw"
    "zF8Ej2HOW3bj1yS8OgUlv+zmMeYOw+W3qSlhjMvH7GyePBmuBFn/kjkEf4MRBbWniTlUdqmPz4YXS69hyyCX3tXGMwstOeYs"
    "Lzn7dL5GqBvv5TWD3T6sF+z24a8U7PahGOz207lyltSpCQTLosBky/q3t+yM/vvf8pn2IexrK8B4swdR5N6YXox/dWbg6WHv"
    "UsfzwImxCCDPjxMYJzx2rEM8NYhGHr5axLZH6u6Kiv7lMNNnwTiZ9BZDSBy71Ult8cuSEUBn0b17fUgBVQrkgvgzlpKR7aVp"
    "JRrtiQiPd5MEMkFrSajxFjJwKbXx94XPNGm7mP/1JOCeMwC9YJqEIAjvAgM0HgQX8+C2iXvJNFer8FM3NB6BBlFuwBeXhHCK"
    "OXa2rJwhCjOCLOYOMuhVz8MFZsCuEz2LFwwdZgKLCCvzJIZ6XAeZoqBJoptFCMsBjiA/3yti8Ix0BKd7/Rvo8tjgvHTTNM3l"
    "6ueCXImgyY/sBrQ6XsHL34xZopzH45yJZWThMhhAk7RaeuREEGCfsEhvPv9XxPANiweRN0vCCEk083DNFNJRmLN5PBExW5D8"
    "TgQeakHeuFeoN0ZhpHOL1jpJ/pkFdwn5PEnu3ZPNz75BsCc/4LhIbfXtSfK3bu/nc8EWPTAgfRNE+T1yB5NCy3ZEsNxZ0jdS"
    "w17Rwrg2XtWDyZWsKqoHhmGrNNUT1BwQu4r7AUkMI80YrMTA7lRDcNnl1wvBZZf5aiW5rAvB/bAiBBdL8RDcD189BPfDHxuC"
    "mzEhuFRmtfASpwOX//H4n+hSV+DhabGOYo5nYEZ9Vb2cEenlPcvwXzm15jyRDI85KO9kZ5GaSci5JbW3GfvegOlHkD9WSlYA"
    "zOUSXp7tuUiob0S2PizzzMULtuHnG5bQHp+KR1E41Zlh4/sHUuPQ29v7/6337Hfe7RMjSPSefXRrHdzudg29Z5/67nTGhgbH"
    "sHPfTFgMw78n2pOdHM54gp26qJfiqv2wJaaTkhXBX0veFajPawplTWVDQ+W5Ai9IXoDM8lqTf0kLAtOIZPaEcSL1eeB0SORw"
    "xIlxEvwzOQnu3TOis6DvsLOgL3sqwsl6cFkINi/7Db6vhpvLBEXCX0TRlUYJvfDnEe2vvo2Uu0EeczcERg+fc23niyexGhZP"
    "6EgSv6deMI/fwE4if85WuvxRLI35g1hJ8wceo4a/Ed/pxI0geb8Xv5mxgef6+AK/o2eSbzrwxAPlt29wq6789lEw5O94vTk+"
    "TlT+DNfyiJ8NSYeEb7KQIkgmIlozPU6e6SVio9jMfHpZxhdMZbDDfCUFDK5SZPKVUjkA5QCmCFJ0fhrozOjxRGg25QnAYcEr"
    "vCbFVC+Ik6duOZXD/6UbxYzbMGU/yoaF0bOyYZlMdO5YTrpNNiyuOlI2bWXFu7Ipgoq/ZUMEYnxtWEoMxM1KpQQS+CXFdBky"
    "UVABHPt/HiQ1mTVOsjwgTaMAU4Eoo4BtOgrUQy89MQpEdkCjePFy5drlbLwXZV/cA6u8Ev6DLNFYLrucIeKpIpqFr+JlQQB5"
    "J+QvQMS86XxakhJecVn6QJwagcuSVnTe5XfXijd4BqnQrZVuV2cNJ2BX2pMg8c3nyltd4Vx2hbK8QNngZgxzgkt9demFyqUt"
    "K8WVAwAdHx0e7O/tdq2OYZiQ7ziKmW4QTtEzdwa3qLgzvSjr8jZKkvRTw+DpruSU57xm40fXM52e7SyY+WvoBTqlRtqnhI6p"
    "OCzFp0BHNkkS9uh6Btn0kUcCMJsfM+Bn8k0NtBSNDPhUvKiBFXNyBvoNf66B5EoiA3yDjzVwXC1kcC/xsQYOp26nzgkFxbiT"
    "tVaRlDQOpjATFwrX6BAWD9wZ471R69UyI4aeTf3+Wfvsl/4i1Y1/3OuZ5Jdf/nvn9v/4Je7fHxP6yy87LRj4dRyqcXSsFMRX"
    "5/qr8+za4YI8Q/L3ipyCKDIprJaZm+Ww5sk/GL1M7ISk2YocJllTi3JCKDXMJPKmulGCEMJMFNFFHlRF9S5MKLaykxbzhmX2"
    "nNI+drCaUHMwcaMHid4xiNr2AspCY2qG0ue3xspbU9AEoiVta0OOV4ewSiMscdRJYQOqxUk9WZJ+1iRUP890GqaUzqrZYykL"
    "rZx193+JYZDSZVJR/IKSUGByuSaV+WUdVsd6rvYM5TfKrvJ8wvyYbdoxX8Q4SMXkpNClsjGe+ZBe0aLGWadfZoQYOqoa1UvI"
    "bm+BkQq/ynNDHb/4dGIov3N+8ec/jl9fXo7LzWxgv7U++3NkRfZzvNXMseAXEq4M/A1+lzbNPX6yX3h56WW8i0Emm8Yd6urg"
    "wIr/2ZHu58+w19bRn7e3tE1PEqc87WXGFABx+iMWs+Q1G7NrbvTdU6i+l6QgiFriFF+qcxXvDFhd3FMl2CAZGMucbLrUNXCX"
    "+gzslGpnyYCUIuEcaY024zTVTeNNmrEw9Teoxma2tSsfhRlJqNmohBWrjwsG1wVFOUyEU/JelgTSi5+7z8HLB6JrB9muh9ik"
    "nTE3KRiE4Jzn3meoELcARRfzDW7+1XOS29v9TkfkqIWFINy6HEnrFJ05TswSeBvOE11d3Zm8Wp2RvQ4mMfVEKbg/FlOLwzn6"
    "4fBHL5m8jNjAK+pAZf9JlQq46EV99JwgDzkAvvYCqSpMVBXCQWjDTllUAo2aQEPH6vzjH9kFdZDXQHJauW5LZ/cS4x+hcT9M"
    "CTSqRLwc1nwZjf7CbKcW+ZCw2T8SIme/TMYrpXDhzVNp3t7CpqtI58jzpSplytzUIxLIJRdfJKCXU/fE3Wd4k6/oFf4dFDFs"
    "Rci+4i62FzwVgs4gxwN0Wxi8m/HgpGfhPMa7EmoScGc3hNS3CB19eqZbhLDwhApwzUv5Rh3j/2fv7ZvkKK584f/vp0jx+Mrd"
    "z9T0zuDriN0et2ZBEuBFgFYSlkGPHlV2V053zVRXNfXSrR5pIgYZIQkkwDYGBMJYXrAxu3rBeEFIA0R4pC8iT8+M/tJXuHHO"
    "yazKeumeHgnvvXHjOhyopyor8+TJkydPnvPLk7l2n+/kNspJozmZzX2+T/CueNAanhb9B2/+adHP8QwvurNwxTnY4Q1h4s0q"
    "8sleAGennjwbtTvcoudlubHLMhGahHb+nkM1OXysMk1vn12ZCh5ovJI6HmjEUp///cdskgbt+UD4mWQ2ktyUyz7tqMvoCtIS"
    "Dg/IzJGDl/q+BoAlmedkd9bFnxOSbGiglkmmky2E1SSFAuHQaUN8niu917UKyu51rSIKIdyQMdt2lFILmBzOOFVDYi1A2AGM"
    "wZk0+bXCPs7OmqahkRUeVJQdgOTppcI+F/VNxRF/DqFnyPru9xGC4FaCUnlmRwnCwC6YhQD1mFHLr7x2cUYO3bNkJMDQlWwl"
    "QIB690GAbOOEHUi/C0aJ5Ffao5JdNuwg3sqrEvGDkg0R/aUGeF7pojOBATmJF3ErcyXMGSRHpGhK5EwJvADVDuTouU2ybyvc"
    "gc9xdoS+I3/CHTdPi365SFTlxQCxQNeGiz7tiTLF08M9o8ShWDrdYlk0/Jo7CRfNZUSf45ZhJujZyDic8NCfBdEvn2jATTyP"
    "+b7Xe75jVnUTqFB1ztR9wRdmkq/wJqLUd5PjfbhPzIVm1Z4r+bumSaK8WpEkKV9QWC7PhhPT1XDi0ZlC5ubl3zO8sjT8R1U8"
    "OQ0IsjFIPgBYzTTNQc2dnB6XnsAIxqFna2oO8bpZxV+krum3rsCrfKgdptlydnw7w3AdrW1GS3xI8UBgCNquR6EoZS+n5oU2"
    "HE916HHeWAhwjari/MvdVwh7DLd8ItxVs6UhLBVMsoeOcfilMC40mSkjHxtjDlgIKCKcjI1aSl6MnLcdav8pbIxEAHosqjkV"
    "K/PGsAoegknsRdI9FW+206LRkAvEXLbRfdibkp3s0mnrR7CKRrmsPwNlg03WYLdhS4DIlBFOPlqeUH9C51BBsbhGSXGmTvVU"
    "r3Vudly+Tk4b0FI1Tce0RkdCRbRraufOcFdEPOjU0gsHyFMpvrUlFwk7eXKq/JO5WdOEu4tncs11tAZpZlq1Wm16tpQraU6Z"
    "WlmD5/ZFvLxrapZXTXNkr5a0qSCT/hlcbngs4YhQTMIy5ECIEumRDdE74p6NO2Rj65p8vGUlNdH24JstZhnwua0JfNlobSHu"
    "zVorL+7HCh6OI+5tKe7d7Yl7e0xx14Zj4tGRwt4eJezdsYV9YtoIJ7LCrlOhTbkmCXuTONB/AGHvFgp7eaKfapCE61iBsGdE"
    "fWJ6O8Ke7tUYwg6Jxv6+Ev+U1xZmtUDYX4q5qIzV9AoVv9Vr2+taIyvjx0dVxo+DgapbqPvBvT7aRE02Bwuir+4WShvOYVk5"
    "anRzGXQoGns74v0eoChytJfId1sKy7tqUzt3xn/9pPZPJ0/6J0+6OWM+zBrzbspw95fIDbOfB6HY9laowPQhBoDp6tidusd9"
    "aw/uk3q2a3m99NMyKgwe8pKJ2VlRWexQtxbhuRbY7cKB8qbwgQRz585/OPL//3/W5FGJRwzL0OOUO0x3gJVn3BiZlWKLq/n+"
    "5E4Ez+zQ/bK5LWtaQtCtb4xycMhTRQmfh9x/0rYBkSkhifLBT+BuwKDgU5H+NnEREyvAqSfjDLNZJ3JKHe6YKuMJo7Rc5MFv"
    "Mc5XixFoQLfEBTqHXogjZsU0TCN9dEt2K62sRXm2NFxb58h7xrOGXv7Ulh6R5Ohf4cJX6O4PBPcbrTQlCldSQNiMciJ/r9GA"
    "YiJS41zEpBPZ1boKB7RGLOBVuA0dItt/J6akvovlTn1ZKIjy9E/mWxnLmdH/yn1lZ7+KQ/Fp1qnHW/IOCUweuBrelJ74cSgz"
    "eWbj1UugUQrDDdmzILse3bkzOe3x6NEYwD+rP61m9PWO6ZQ2wNODtXAov2PzqHjuk3s/pZtKgMCv1WqT03GwpthaSjk2+Bal"
    "wMnhjXCtyMliBAlsSrNTvbLRqAV5O9WpoYrUhsqIavquUe3ekiBz8nQOmOPq667apzjYfaNdK4y1q51oq9YGk8+u1WrtmRL8"
    "M3XyJPxjTUyfPIlA8rmaZ5Q6J0/yHbXaFD3Q1h1Y5EqeERpTBi8X2khzRgi3LkNhsyyNLEm0JhPlhqKjMVtQiadXUi01dtk7"
    "dzZ+wk+ebGA35VKYAxcOJ9fOOAaKqU0ABM1aDmsIMV/P6QorDlxXCqP7xrEa9IvPqnqrJhqKk/JPGELq/a4G2ij2RKgcCKXG"
    "xHT5J7Wm2pJYu2r2rDU5XY3gR1T11JmsuZoX28F2eSKcUH8mlRndcvy0O7Tzx8pLNEbb59sxtDqSDzJRVcPH3lHAtoIni7QY"
    "oxLJWq32qJq0QlILnz7oqmLvmpoVMWvc8kTOrRSWJ1QBv1wteH3ypFBBY3/SRbicPKlZUBjPo0zNhnqdvv5Nmphq+k/tK0hI"
    "EO9BcgFqFcSeDPW6TbMaYvMx54iegsbDclX/YyL5AobQBrhY4PnFVxJupUjDEUoSthBKIH00vAx7pI4qwF8k9okhJms2Lp+8"
    "FiqPgQYbSW/teXyYRkzY8akYryYmp2e8XbWpGRRQrSYvjgxlqjnh17wJm7ZmS97k5BJ4ZaULsDzmztyfmDZ8tfvGgIdXEzPe"
    "T9wHJWNiYilxQ45LhHYO1D95ciqJKKV3DSOHn6IG2dAGbPwaUSnZvimhgr1EqivFqNpYH0wrl4NCgiSbg0JLXD3WvDPJw9wO"
    "QxnuKQSPMtiTh7loY0xMzhYp1EPDnEMjLBuFXs7fkEzKdIRlI0MvM2F8noEX7CQNXtvB4bbP2ILCnBGzU1Vu5IARHK+xCIeA"
    "H2BdKC8Zqcc5pZXMSqSALlm2SvBOQm4Me0a/r1keJMvcuiwMdRiWVG8oLHnB+lLZKPm1kl1wozNSFN/X6tNBRXto1ehDSNNZ"
    "LcLuI3xanhDZEW974x3ljlqt4BjlbHYoRLkKJ95T4ZLCSUF7YdwU42FrhDSpXXZ8VOUn6sms+hEfvojL7FJPZtWPqlACVzR6"
    "scDNhCdP1kxTKd3h0uflIzgCjGOuFLs9V/J21GoQqvdUiNltyLJByTP8MpQHs/PEiDiRN248ZcqYUuGUnEaKraRxo2kNo0HG"
    "LHLFGWtdjLbeYMyM6igSb9U8jYF4+jVlTqaCNZqMlTitqmrNd8plo1MrzUn5BdthTvcgKdvMLMvDJ6UcBr48a5aNNqIwJbC0"
    "gxjSmTYpV4/Y3aqF41bW1CtrUWVNWZmkvJ2oyDJEurS/J5rJ73GFIjKiZFMCvrmg7JKHDrkpzx0X+IvHdcBHE9NGBA54vaLt"
    "1jE5bUQQsSrplZC7WVJIeLWx133oduz2H9ZFEqVjNa4MEaDB6Gp/l41+LZi0jHott+R2yzP1nTv76OOPJmrT1R11ZQmkzJhj"
    "sEmLJmqT0/9vH9z9IxfKbY4nJ325c6c+pEOUyQh9cXyEvpgYO/5+3DguKYsmatZksK0ebT/wLcpGyd7ikKNds1OHHO0HOOrr"
    "5Y/6GroezyM/he7eftB9XSa0PuuiI2AW7sVKbRaSIuSbRD2UbuzI1NGJcPRXcms0kXlZ1VodXnO8sYrTgRjZqGIOaK95Lwvq"
    "xf2NvmFW7AKk7lDI94P4cqkNRXlskmLYoOjgrnafXAhXyKntBGUYywUh5IF2QDDtSEVilFWabCeGL4zpfUY2rRkEnVJwJYXV"
    "lDamTFKg0fq4ExUcnDfcmTTJ07Gzl6ofAT7WlmEJW1PQY0kEZkgYZozKT8C4DWslt8C4BYpj2zYk29YFDeAPw9LY8G6k/rCH"
    "gZ/HGqj8KPyrA/u+BEOa21AhYn7nzrgEgOaTF3RKXea3HBY6kbkQdtWUTYtn48f76Cc1ZSyn7j2MJCx5nx2EwhXpOxCTPbFK"
    "Z9UGMK6VAv6F2p62ABxeojwFUSCizlgfPd9JPnHSsN8RXyFAGD9cEP1xCJQARvXJ1sQhhhiKLy0ZVgwN/r74VoDVHodzOZz1"
    "mLwrglePw70cJnpr/mVQ2JKDma1tjnk7dNkt3MPrBVLraXpHhovRE7I1f9gk0R3fIy/m/MEx+P/f925O+qnnmZUv9ey4uTJl"
    "daEpZpcOqieSfMeHjxmPgcX7fKe6T/5E8O4TxwzQREH1+SbkL+nWjsSXfBpR6i8L/0quBo1fzA170Rn2oj3kRZLdsdVNXytK"
    "XswDmFJKXQsLO2d4gvQn6b3k1aNBh7t482j+ttElgM8mt7mWjRPZy0KXysaRn5U844Qv5qrKR2PDDaQS4aAuKP0BSNqzcEsp"
    "/NjtuaHvOdUTrqeWSMxD6ntwfBj2LyrrMRHVFKUjRJiWmhpcvpUk++fRskyBYcH9tUmaz7JKs0mzh1fS00nekIpTo227cEVq"
    "23ZTT/lxfMqP609dr/eIdg8rtocpZkQuao/Q7hxGatZEr6HdMJPUvoVXuMbZVEW8sqbSpwr9TnhJohPnLgUi09lM9SKpt0ma"
    "UmiJfhp4GYSgaadmcnKLrCHnaVVU5K/4+li4GjY+oGGgRkNtGV8ZK7UivcLTval3CEgC4EpyQIDuP4kLZc9ipAtDZvuiohi6"
    "lYggeI+/ECADXu7kC/jT8GQ+3Pgx/mmQmZU8hb+MDnChE7azQhq5KJfANvUzd+nuEqHH/tEwbQvznVVQ8NUWSxPgSkpcKykx"
    "raTEsxLLZCxLsQDpIlPJSIz2dywGOP4VOehqqOO7f+PBrSRjWcmPXCU7Por7Fcn9imS3NGMrnTDmWEWqPj1MaIhKnI0YTHtd"
    "w+/QU97PPlESlR8EjhcGholGJSWoPxFrF1Iu8fYWKsc/dnPHATcIDnUCoVoycgvekZ+VAtCiqAiTiqQCO5Jr4GhK5PSqpRCl"
    "iFH3Jx8xsT7TMOXX5tHy0SVDXuU8Va4+VTJNCFlI3sRLoBh6N8FsKbsQLIh+dTp1+bSWyT9eFTJPi9cGnfFx/vU4o70TG4S8"
    "kjOui3j8uGpUW7AkazPZ4+XikM8Qf3TJeMkvFbSHXDtRpIBDXqfUcJPTUne2bMsS7iNVyAQuTIPu7JaUxezJUzQ+i6R8FvEA"
    "h2ux9Lhf0joIcjJLI1g1pf1ilvVVPVU0TyMJWkxhYzIgv9Mj1QxRS3lhPFouH10qH4VHVhd+Gnqf4lz6hcNesDvY5sBnrgHA"
    "gS+6GkANfEGL3/vQ52nacujjT8Yc+tQVBemhx5NQqcFPFc5TOWLw02SNHvy5bvwz6pY1XfQ9a4GRqm1HkWrTZAeU25Txf1XH"
    "/3aqo6OLzN9FiXzfcvN/Nc//LpqnnZEdfOh0y0uNbsUXriX8WquLPtpmt3bCdue86o/Fj4wgajRg2wG/e9x3q/8ofmQIyMhc"
    "nVoyjnVrPwtLKGjVIyCQ3a7K4Gf0k591+nlQhMbxbm3aONhVNwxYXgNxr5Aj2NjdrZXKtV07DnZPnlQvKl07sOu2Y4f9gyEP"
    "8XgEjTJkxs7UUmnxAK3jHdpVCVpd6jVkfet1a7u7pXLiJjiEKYUdEbKw1u3KnHiGW+ur35BnHu4tSLtcoc/y/AuWr7SDWpzu"
    "Y1r8CJ9MlvbwUFRcr1cqT8KFZW5DlMtafuK9WuOpBnd0u0B2SZR37txRT373utB9qqmWVA7EBPCxoeVSAaY+Bw0YYaUdpNrd"
    "A6m2oVlB3KCsRbVet3yi160JRH3ha+bNHalUKv0uJe8vHy2Xet3Zvd3qoS4m+1462I2vq6CTHjJ5ecx8blnoqFcTu2QmQ9tA"
    "zIhp7OmWDXlOJF9chh5GlqHIwMgi8n5C0DRQEtXL0MLSf6kKauKCDD2RkoVYajIy0df/qOt/HOtWYO7U5L/qBoOwtius2NaO"
    "Wk3PYb2ALda7QCZ8fCid4vox9Tqpfm+6xLwmZJAJx4WkBDDTTYAZwoYHJ7Y5O1UVGEbwonB2Fk7Y4/ezsViDDmh24eaBKv1j"
    "+HTiaTbpiGtRNxZEHyGVcIBdZgxHqfJ37nyuW/IrtlWeUV9JlHztf8yUn+uWSqnKRG0X0bwjphOdRkRbeVY1fWTqaBkrhX7a"
    "tePdiQnlzVPcBkGWv40TtlW1Dai3GhqhHToC+x46wqh7FjiU4B+DWqmq5sBFXaUzXpJPVXfpqOECLrnUpyloGyfaAZwjgCla"
    "TaboEo4LnOa2MXH4M93a82HpWLdsPAuuzMSzYB419sde09DjQThpxx5OoDj5De/kE/APmoasxu4K3AejVx6vDE7+RIc+/ok6"
    "0XaT314UmkeNJ7o1uUaZXmSrVsBhs2T8tFtLLfdaCeCXuWTsA9LjHbhxINM3w0Qm67t0Y7Fb2186cewYpVF/LrIPQYVPeUFo"
    "qkvXGo4XCHmX0JLhyB9LoO2ijibdhivXMCnfahkz5Q+TFjMT/mvK9cy0QAn5JhyvoI95K/7wxTZ94bRk6bC1BCmjaIVL1AKn"
    "6C7OVRRBF/G4DqRgPoSBzxk3vqPAm2NPHXpm314HF3EIS8hDCyEVBYRXyG0XruBAuAFqdlfPg0/NqYZSX8+8FIlIPGM3fC/k"
    "wQIuAyd2wL19kA05KJl0ZTnAYHa4SVOJwsZrwSVx5Z07UcEs6Un1ZSTb758ItbtgUDMuqSD4ky42/Iycy5DVF6eybZUNUdsl"
    "B6y40SGMmi0uXUHhCMKSeSQ1ZY6aZUoT79bUcclZNPMCAQnAACVhEUaAEoyos2sAs6Sl0Ua97mJS2FnMQOK6ohEKa1ZUY1pe"
    "ioTfJzAKXFfR5rZrsP+HdzomLeRofmgBZGW9Iv/DZFT4kZYv5o4ajOxWg6GP0mAUjjfwGnDuC47wpUxQOq7TMCenIR/XrDyP"
    "eEIe0zzY8D3HweSvS2ptD5k3x2y5rpc1BgCW31bLCQTmTsw5UdCqmkHfbYC5iRHu2q7YO0aGaew0TJRCYKY96mEFf8zOms96"
    "oT1nNzh6QM0lCJVEgSFvUoi/N5eM+PaDZ1GewNyGlftA6adk9R72S8+UnumWSa2LhCrL7prGCVLYtpXTV6Z0NCtpeYSKPRJr"
    "2UeqsC2CX0dTRekdvaGoDP3W1lGTO3SCJQh5GCVMsLvxl6nSASCuQGdXzY7n2KFQX/DQa9uNeF+jK/RqWNv1TGkBrDCcU7p2"
    "h3cePgckcqzo4THPPoY1TC+OQ/FE2zgBe4uqH7MAgx3T/8MIbYhDmbR0xhyKnfRHTNoqmbgsHS0bj5fMjmk80TXAURWEvuc2"
    "TSq8UJIrbpnS2MACkvG4/rRrLJRMZk7QWyiobWbgI2miZHelcpFKbQTzSxaXcqvczWFtV0B8UBVX/AiOeizExhAJcNn4J2Nf"
    "yo+jeeFObNEqqqvcxEhWuNlZczcVIbuk+J1G8zOl55QYpEfvmVLUKquRyw7R0fJR4x+NA93yUUivuL9bLpeN6Uf/sQx8PVad"
    "XsLXz3YhHlw2Hu/Ky+VKpuc3uWsvwgKSLH8/S/YTP3NLj3fpQLeQV4PJm3yeU1+yIPR8weyAuV7I5KV+VhJ+FWggPdWtvVCh"
    "e1RKUjG0RRBw2DDIbMrqymf5XLvJWULvqqzp21Zyuzj8NQn3CMHCPOl7vaDKpue0S8vreHXQpM8tOwroGnJZeYVeVeiVfmW5"
    "F4V4YbR233vmG3lTdP4T6gcLPEdRmdxjLeuYhBUabruVdzBrnWzb7qS6Nn5qi8+LeDPniOMJTdyxm3ihUjuosoZIX+Yur7RO"
    "905Wre671jvY5J3isk3eyV/hHtMKsq1u/X04guejILTn+qr7+QJQ12TQ8m13IeYeMpX7TdtVl4BTXJXxKPS00esKf87xelVG"
    "LgmNTTKCV2Vo+9ldkbwrEA/sb+Y68eQDNbRDv6AC+icjxDf15VBZhoheE5OTVlnoczegOzSTAviQ+hg/S3+YajUpXrEiH5f7"
    "H5aN1JcNz/H87X6kJtADfVz3YOS55fW23WyHN+ywz6YqPwrysxmJGcI3+U5eBlUwsaayo1iFC8s0+WnQWUMmHRnJiyiAm9fR"
    "TNS/2WJ6Bd2mMbKArc3AOc8NJ3EhKZApWE4wCq4L0lBpLyw9XNRzxUd3q4p27yS6mBx9QRihpFMzA7+v4C1tW2ntcerAokV1"
    "eHNzgQjHq4TKDmcAbFq1rupzWK8eilWSl0VUSSnNf0X6Al/r3w0rnyuYTLmC0vRii/496IAPJXEIs7dH+tBxf5gu4f583AFN"
    "rwjwZcEYFxOg/Sm5ZWmtDmOcKprn1FYsj78sEqctSQzsdic1wsOaoYJb1S49PGOwWZbc/tSJPxx/9qhPxpZC9cFWAqe6+z1O"
    "o5jWh51JW1X0sH0bdz4V0jH+lFKfb3dWqe8eaGLlPh5vbuVpHWd6xcI21gwDx+kYXIdi259b9NX4EwvLjy2RWHorscP+fY/z"
    "iUh82Mk0spaH6tK40yhPwfhzCL/d7gTCjx5o9qS/HG/qZEgcZ96QOI01adApOAaTsdz2p438bPx5Qx+MLX1UfCsxo05+j1NH"
    "Uvmwc2d0NQ/Xq3FnTwEN408f+ni784e+eqAJlPl0vBmUpXKcKSQFazzTTjQ81+J+f5w1X5V9APMu+XQbBl780fjmUfzJloZQ"
    "3PHv08xLKH5oQ2/Lqh6+h2Mbe8W0bMPciyvYtsEXf/lgJl/+8zGNvgKKxzL7EgEca/6Bz9XnQTjGKKii2599yZfjT774m7El"
    "Nv5iK8GM+/w9zryE3IedeFvW9NDdG3faFVMy/qyLv9/upIs/fKA5l/96vClXQO44My4Ru7EmHGADtvbWQqkKvMu5X/GDXoEP"
    "Nvmkl4o1FDhrGsUhGxlHSSI3xZ8ahfUV+ajxRco3PSSQM4QL43qrx/ZTj+GhDtqjI3Ijo2tBOxtg26qBbYhD0M5LxFbVF45X"
    "vkjh6BUUGyPOQCxujz9s7W2MXHt7g7dVDGVEcb2nW8RH2tsOkWzVDaf5EDLoNLeWwXQD25BBpzmODKarL5TBfJHCESooNq4M"
    "Os2xZbCg6FAZzJTdcvC2ksERxbcX1Ruvv6M/2EJwt+x7wfK+ZZR6RLQvu4CnonwFsz+7Wo/d9qhosG6njYrYyta3P3Fju6EQ"
    "oTGCySPtO2MM+sY1CnN8KfATQJx7ksCeWi3ctducUBadSfpDpL/B+DwTPMCOsTnP73HfCrZCcBTMO4DTbZMC/IZNVaZ/LEmw"
    "3e1S8M8Loj/n87YIhvZQm8W+19b+1GAKKXNoGPxpSoc/LSU4D29YndNb1zmdr3OcnhHnxuvZg1ExumcZbhES6EGZuPTfJPZP"
    "BNUTkCSk8NYFxLdL3NsRM0GydSbjHCwsEUcT8qSJLiBI+saJR/LTF5CEWn4Y9dg0HsnO2UxJepgu14YyoJuxQJs7Tuq909Tf"
    "O4A9N5eOYlo70DmHCa5WNYci2UxVtKCMaSCC0Uyv06YBBoP+FM8lGKh06Fhfqi5NF8lCP20UFaFzdEtl4/luTUINH+eBeEbB"
    "DQmFGFSfbqkDCWoUqoT2pHygKp2KPH2whE3CmUVZ6nHPcwR342KQO9mx59RrQqnHb+nqbWRDYSN0qGZJ69fWxYhH+6kPWBpB"
    "oekm0SIorIveq8Qmw4ssSYDmU11DojvzaZBO/KDTkAzGxEbGD2gx+KnE/uPDpaWlBGR6GM8xPfLPdUDC/kNLOB3hB2yS0fGz"
    "RxR69HB8ypAgqzXtUGB8Vo1eVexQ+Dz0fBBwfGLOFt26I3MTLxVmG87UmW4Ol+Ag9KMGtiJRtJQGWdLQ8b3QgypmFQ3VuEHj"
    "cPoo15PddGr7Ulh7ugt3IdkuE7PP1edFI4ScKLYrYJCFH/bpKiiVC1pQPm6QyR1TMP/m7GYU/w15CeXvpXJVHAmP1lxDJM0/"
    "3U1010tAisrVXE6YX4L80TE3w2o4YZpJDfgVJbjEru2omR5SbZ48uUPEiefpdhk8nKHzVp4tdGviiGRe6O337bYN6/PRcnyv"
    "irqa3qUkf9DiDLXoay2q1vwZgikf6neEhCr/8z9rFbN2FIRMFuasEz+n+0xMlbKyJJOHUPZqmhdyUuNZSaDpX2IFk1Muz3cN"
    "iQ6Ekx0B3P8CaQ+DKpzqRcy3CcpiUriWeTQ5f4bzl4e8YIZJZyOMJmZzc0M9P3cqwRzd9GTPiZ07tROkWllRkbXVgC6VkTEm"
    "qCxzkMEDSB0nwpZnqYNb+YyWWl16gkfZT1FOJRfEp4+pVEzDcq/9AMZFdCOccXN2U95bhSl7ZrcsQTuSWFmOSN32ZBf+f0It"
    "rVWZzbJgzaU1NVdALrUqaZtazlN/lQszugFbLNsXaI6CVWF3oIG9zXTGtiQZW6I7XxhPd77wf5LufCGtO3/e1e9okroSTz/R"
    "VRjySVOEz/VcpT2pnUAplFFl5Mntkl/z1YHeglyGhTXsEUHDtztwgAxIrCRaegkyHFU6UdCq8E7H6eMtL+qMnZt07kUcX3VZ"
    "R1ibngl/kr2GaiacmMhdUQU6nlKbn1iq6k9nwv/+6OzPu5ItcApux1QZMsTt5Y1Wqmf/KhclOBG8VK5u0cOgcJmy4dI4Y6tP"
    "4WxeVaepmKDh6+BW3HfhMuElLQdxzOB/za+8P/gvXHl/oK28ol+08r6wxcqLX9HK+8J/+cr7wv/6lTfs65ko3dRffuovu589"
    "sxy/4nrBRLt6/WGpJmViSSOo/cwvmaSw03kmF0ttD9KCyGyQnYZbSvYnZtmArRL3q2aSyCQ5pwD5z7QDkvnkJLZaaWdTxyIz"
    "SVv0xJYy0SSdYEwdXNTOKRYfT8xn7kvnyCzjGUBFgN50esuo5VHLvigbR1R2FnVyWfjpHHbqKR5H9QKh5anDv+OUcElmuCLe"
    "pLO8qV1peVgntSLZ7EOF+fRsyl63ZSoiLY8MXT6aS+7WUGmZGp57dDh59tB0eUqcMWeMEbNXitUYghNSMsehTdP7XGodrN6k"
    "hDR+v5xK0Kc2zrMH3NKQjEOZAUp2t0CKfrZSjntsQGZOhcYHKY9MHT15slZgpcgKMAtwvpMvdksvdksnliTV+ha7nCQs1KjL"
    "cgJfjkgwJOvFlEKKF/awxEtxUVOxEf8aIRjaF4l0cBgPKYheluPfb1NDBBHFwu4Ds44ER49qx37hhdsv4w/6K+xn8mMdq/5o"
    "ieoGrWoeLS/9S5ztyOvjEhf0c6dL4TScp8Qic8JUe6e73kcc9xseAR9xZgv+p3JzTuL5xiqre2HotQsDMBpVo48P6gVbQyL5"
    "WplJzD2gQxSGnT2j1Qn9NZluJLzh9cBzolB7F3qdlFsV7hwJM0cup3I9np6a+u/57qWfFh2nK/DjFrl8Fycpq5juWpabuSqb"
    "yvROHdGb7hyn07vD4jyp85g/VqQWc73uHd/61OuWB1tHHosdRlBxn7KiMwxqlQX7pL4pBHZ9v0K8jYOpeiXbOpy6zQ8lox/i"
    "+9QZ1W1+nT8eu2UFDx7AzFY/DEuWUjBpuFBC0aSiqLpdsrcMzGbC3XrVW4GCtiw7TE4LC4+chLyxMMcbYjLJkpY/Zo7dg2Tt"
    "sqcQiXqxNFVmATi0SlOV6VHMr7peWKp0JlVCxXIVsssVrQAUwC2P1lO6pBdMVEL2jQTL6XU3WgISTT5Mk7KKB9JW8bcjoIhj"
    "kjta4kcRgUKjKMmRv8XgTxtsWiuti5HcC47Rn/9SEVE9HS4q447a1jDSh5kG/+VDT93ZetJsoycpFPJDjRqBh4eO1oj1YSjs"
    "OGVxjfhKWgkjK6Zz96OKDIX/6kfzh3w77ET+w02l729wlAQNH6RCmiudSXmRAdv1MM2r2xC207CMkEzSnTZbtD5iBtH34yqB"
    "XMMPtBx9fwtKjp6/o5Y68ffTrRrB+lGHDHhny08fmOj4Vo/i3cd4kyepZLvWyxZ9eMjlIaZrDIKCtlH8eDRvh5i+QXv8HVrQ"
    "HsvTkKckw44trfb29gz39jZt92G4ap1kp2kUP34gLjvN8bnsNMficp6SbXJ5NBB4nOIjuZwDAY+DnVNZQg23AEinO+w0MJ2J"
    "sLmCKYs5+OgngNzUHHukmsw3eC5XNij9g07jANTzeHLXy2zh0/hKKPhI3Q71SFbRY5X5u92MR3LzpUP6fg7utUSUnluA0suM"
    "fuobQO65Bci9une8auYE1qQbvLJv1I1mElCXlS2E0zX6OpxO40uCejlwTEHqZKAU3MJ12+V+P8bLxdd8DYfRxXcCjILS0b1r"
    "o6BtyZ1pstQRWYwCvEeLiuNVarI4FcuUSt8xNqr5uOQY2LqgPxpbp3F7THyd0x8LI+L0/w/CiGCfkyB51M9F+a3+f12U3+on"
    "+m2uMMrv9EdH+eeSKD927b80yu/0/5dH+TuxxinUNo3+cIydvItA3TdAdwocNWwXOlM9UaDYqycyIFsN+NYU4f5Dz3Uwv3GB"
    "WJdQ9WJcnMBpECivql/lkjBOoIP/eFg9oTaS+FatUfHFKKl7hpeW8ILm3diX7HXZo68kVnJHN4KiAp7doTepgGxwUarE0xUs"
    "gENeZK+8rhbdg53GAsrxQDCgumwwByQsup569M3UqbukxfZvi05BE8FO78DdncNRidvgUVKbvPxVXgy6ZKghKARvbq8ReS1n"
    "4cW3Ir7zPiUFO0Q1xFkeCwAgF0ZgJKP+sLtttyHLD3brLV4L3E5gOsq+i+PLk0W3ybbgA7xlAbtnGhQ4Nszkay3HulaDmj3x"
    "1QvxOp96ov6QbDCPGs0U5OhYMa6om8EVpa+nLULSqNtpeUVXQLlbahOrlydWb4Y/j+h3LhXdlPR4SV1r+3zpROpmW0JYoElo"
    "Zi8Ugw8y19NKi0a/mlZe6i5I3OiqXJ5MkFiOEvJjCxBumqefY98Z+zA3wz6iDyxmkaefJ09KgJHSXVmAiYYv8dT17/FPifbk"
    "RgzJTFRbeGR6eEVQgurBe0KLq5HrQ3jk0eEVURmqin4XVZaXNDnCCcCj1cf06EXSCruLAmHFx8W3cw2piNBNBcQ0hl70pcg7"
    "1icQSVP+2+6Xlzp9hRbp9pPZ2O/rt264tV2i5JYnID19WHINE6zDhdBUPxKY3wnK457kP4d7SepV82CLd8TkYz3uC/aY7+OQ"
    "MM+3XBEyCFfPra44TsgsW7D9DlzLzFzeaDFLtNnzbd8OAjYvLBGwQ8J2goppBOBec5u0gTAfq4Odb7HFnh00WsLFYsKNiz0l"
    "nE5BqeLaWakLMMe6cFlTBKIl3LKBRQ/Ax1Bsjx81FuoiDAOknUdBU7heuy1c4bLHhS9sqL7CDtu+lesScYw1hQW1hAZb4K7L"
    "nvMbPGB7j4d+BN66J3xh+zxqM7vlAkim6a9+ufqR8N0ZZkE5vynqrh2wnrCDkFnc59Eca9luxTSaHvX2OWIuNC8HiVnCV6Rw"
    "t8KeWF1phsy1G60wgI8Xo4qJC1GqAu6qAWpCS67LfGG3ojZ72uvYyES/qAGsz2B1SHxPLbRFy2cdHgQhK7VWv2y04AKqgP1o"
    "aqpcMY16FPSfsuF83F7bZXsiv9FyoFPO6sfRHLDL55aosMPcD4XLDtqC6hY+mwMMUZPBlRjMAwrqvsDhPWgL5B+vGyxqAycC"
    "wfaCJ0U4TuQ2hcsWI7b6MQg/cI5b3aq51+8J2MmF+PczIuRVc48vWpHbNGSXTaOFDDrosbnIXYD5Ygs/ZCIwDV8EeChOjZBp"
    "uF7Yst3mCyKsms96jZZiB0wBmgB6Gbx83nzMxTcug3OXi1j5kHFEkbJbvmizkIfB6seNloPSJwUbR9lxhJsMIHwhx27UIAFD"
    "mtx2q+azIhKMSIrgRgu66eO5uTkMXplPC9uF92HPA67R2/2+V3dEm96y/b5H+oJeHhAcgUsmTpCmcEQg3LvLfzANuMYH+Cnn"
    "tWS8aXid8LEo9Kom/BeOIweNFitN//X65MFGy7fDUJSx0AH0u5nPRj77pyntLRy/7NgikPU+w4/DRetA9SJvOYoddWG3Fb/U"
    "FzQXFBNBUYZSNYE4+aoGUBmyFpTBOX91pSV8g/WE6+o6oOs5Dshqhe0Twmfo+HOVfKpCixEpROEWTRWvEz4tBCi0KPBh2CK3"
    "mVJgdWHXRYs7IXRDsVTqQI1/Hd+ro2zBty47bLsLwqmwHPNYa/VjqZuf5gDEwt6SijXw8bwIF2FX22iFPdtdcOwmW4zaiZ5k"
    "ji2aUFgvUmE6/RrNoPCCRssWdWoUW2Ju5FfY48JmFo0U/ONAuzBNkIvAa7spfPqgYhoN8P3g8iBVgnq0X7iaAPJ6U9R9Dwqg"
    "EAa8K6yq+aQIOqjJ/VDQ2ALrhVs18Q9mk9Y+6NgN4bPVL+fmXCk2/aq5f45bbAHK+TgIIvT7oBVcEYXYv4jowRdICswyn/2M"
    "XiEdc+AMNveAkMmVq2k7IUiGj0yQS5eleszUulWhpar0uG+3ywY7GK6uhIuC9PjzbZ+7FnA8sF2Lpj0TNtwKJhotV4QGFoNK"
    "DwjbtZuR2wzCyG+D0MIgMJzuBxut1Y/DxchtVth+f3VlTurblKT7uGyiFAhX8Qt0PfflRHzC88OAxMw0HA/VgnwldQbrSR3B"
    "4T5Mo+lVRW2XKZuAy3LwaPqEybjUmbSEDSs1p2Z303sWbvUxH1sII7B7k4VRr6a4TFxJ5KK5Y1XNyK0Ll7vgdFcPsQONsGo+"
    "r96JINGEcEE8XhIG5v6EabAssUYAU7zKTAOvHjUNZhqN8Lj+EfvrV6yIE2CP9FY/bjmhOVESFXJRBfLok2lW8UNzwk3egWln"
    "t2PqytCSUuHAyT08Jp0teNAVKTlSebOe8C3hVokO+nDCrLDVd3BWoGhoC0Fm7oDAxXOCysLKDnOlYuImBmdIViKSVUNwq181"
    "0fIKK2xPykLCGe4EzIVVDK4otMGsUzN7hv205cdk1R1h18Eg7Qqf7AIgQLQ7YT8ezZ/GbGK9CDrNFnBGqMW5KeYi14JJuA90"
    "HnZnMRJ+EMpZClYNfkFDV8FV0Q/1BREfiJC0EZmwe30fVFI9YaZwY70AE3+Ktdu09uAaLtAYDEIOm00YQeFrVlWEarwpWqtf"
    "+iEsNzBuvpgDQaSBXYx8xqXcu6Cpe7zlwOJTuOyzrtdm+52oia3KMrTiaNrBYF4dhtZWZYPJJwSYYz72QJMIFwwlvhDaXVyg"
    "0LYbJSVsjy3YQTDc2JzPmyHWFwinHoSM7DlctCum0WnxQFRPgMiMkChQvR0fzVHSskJTsyTrrCng4BKogY7DG6Jqpt7Glptp"
    "dGCYfTeA4RWW8FueA15g4auyPArUWiy1lyltCVkA7bSmQBPOXDJwopOaLFJzMHkPtjw/LFaCY2qEstGAU9RUUQgbwLBiT0yD"
    "pgDmmhPwwJWlqm5tl1txZYWiBOnHIzecnZ2KN43xE8OUi6nspAlFUj2sQgWZttyKW54ouZUgQvpmFcXqQdUErYWruwNTCdUk"
    "SnRV6oDMKi/LgsNEK6qVqrC9AU1xZRxreoNmWlPQsg9aIuDdbLuJiS8NCCVocTVICQeXO5eb4tj2V5Kg7NElY47bWXJhVHkj"
    "BO0dhGxOtJymCGCON4WrFMsT8WfmQfhbTvD26pdN2CpQJb1W33Bx7/Es7CFQOqX9B0uQJUJuOwFYI/SD4Y2i1RN1bh1zMSxR"
    "PZIyVdpREJCae1HZxxkNFZCKyim1w8Deni3Yo7hkjFwcjhpAQBCKjmwedgzSoCgk4a/XCyj4mfB7YNnrVMjOIyHTP5Y7y6nY"
    "UB9hei9GypA1xiHeFxCRAf11gAPb5VY/S/3qSh0IABYOJztTBdAOc6fyY5hJY9BCWx6Y7USR3NxA16VCSqhqcndR4+w07iJJ"
    "K7T5cWiweIBVpcjYKXgkmUtbIbIiecCeEI7FHAF7qGGbo/H4K/c/kslyS0kdubv8W21DdHf5kgGPchugu8uXiEQsX7hngSKq"
    "v4fB9NJ7DJ9avrAZ+epgah41XO9Y27OEI+mSS5GFBopwQ9pzgWnBfrRn8hkoCcuveVCqjoRRahGjbo9jWuHqKmy3zR3Fpci1"
    "cbk4kjL1ulLCQtQRUAbMSycAL0pL2OGMpuLUWoXuLNgn2C57xnYcuy1C8q+Yh7lrCSehnHqVoTj1GYvaBgtSXRbBWJ1yvWP1"
    "yHasLdkLLjsoOPlYPWi0XDsM/15shlaOObwPqUyksozm6jwif45OoLP6MbjJAlu5i/AnuLYE9+tgz7gPQGTJ4nVh0xoE84mM"
    "YVdEDBcN34Z5Wh6rL2hJHGvbQSDXvCNFm60gFC213Niu5rZyYUsj0JcqKQ4KJ07X81sgM7GxvA3imrB1G0ZZE8x8Ib1eVabL"
    "fIuHSdFAWi/qT0NWZqofaL0MnfQJ/a7q+TZ6gPuOoV1ICzFDb3IdUC9qJ1JIFVDjx9wkvRtvU5L9CTq3UGiFdQyBVnJR4Jr3"
    "G22OBQfq9VErgATSCqSZiL7A3WW7bbC2HaJHgNpJfJukOYDerCshbgvCw4nvHbA6KdkGUuVG6RhivIhccCXreySiSBYkqtRc"
    "i3utu62kfgBuLRaZKqruCNMUUu0OIDpk3WPYLw1HninV6d6TrHY4W4sDB7GfBs9WIXdyAYQgzbSECXG7D8CGgyHtCnGUhLsF"
    "ATjzXUXDZNoBP4pzHq2b5D2CtmBtBk7srQtX4Gq9BXOlgjrmi5ci28cb1AOaULGHHWpQ3t26cFe/DG2gC2YaOFkC2BGTxwMC"
    "CbCPVH4PcEhpootYM8dBh1+ij3GbCxs8FFa22/EiC1SuwWiPT0X1na+ItupV5IrjHbybvXrE3Bv2eMCedwUk3QwF+JhgWqL/"
    "dK4pmrBAK2uxeOt8CO1c4PUTouXgrhQ3zDyaM1hLODAfcS6S6S99B3X0xPpsj82brofKYgk9p0/kdykgzvm9UMaPlGytdF8S"
    "bTwmzP+W34vJrdeM8rULZaOiYzPjwwEr+wTu0y3aqyYbqJJboRfJVpX+LgpzTpiJ3sKQkUUVKg0NKo7qxFdQfjEKwFUqwzLU"
    "AL4s3AjjHthBz51b6Qh/1mQl2OZ2hD9hlitm1ayYZQOnCugK83E5Dk/6AuzxZIJLI1zyBNhPlmoF7jPvQbLKvWjGw9IP9IWC"
    "PeHA3k8kHuaMB5gGYM4XdjPe4WieZnIjw/bTA0u+Ag4ZV6TjUuxJuTch2wDdYnGkF6eAy+ZEEFZTXuUgcbeDW1+pLmTAHrVM"
    "QFhSuHOe0xQ4BZS2oLG7u3ypyl5MIr8qgKGceMVa8ecvJKFY7J3kIXtq9csWWSMHtGb1WGgH6A9p2tOMovjsAoZu9K1IjwO3"
    "Y29Xlf1LvKFEaySgbVNq4xGPpU8Rr2x9oMfTQZy6aIKIgNRGrpwJ4BOgAB3INCu5uL7pPnyMxEBMoVxNdKKMTDym2f4oGZod"
    "KyPtUZu9KOywA2FSXB+ehvwjQcVcwlDKseoJHgC0xdQ0INXVFHURqvgSKQwI1lKUpcKe4I4TBy2RSGG7oRHvWOIFY0Ss5u7y"
    "pYppgDnlV80nKIrcFD04AOMaDBzoAIwgPc5d4TBNuIHJFbaPW5qxQk2BBhin3cSbo3c+oxSRB1IhFrMCtEOiJssV9rPtcYBW"
    "WVVZbHfh1gCUyVPctSoQoVY+X4ZOpI7vNX3ebpOu0GpW7u9qyvcPdTZz8TStgSUKt1qpyFlYkZlsq+ZB0B9i9SPwi7leaM/Z"
    "DU54U/NxAQoE5wbqKhWEg4UoFfAE8ZMRuYwPrMJgN4Ytg+Zoc3+BZuIef3UlNj8Ohn5zYjcr3T3/3m4ylESbPcMbZVxfXqqa"
    "P/N8HgWBwLgYRfdeesxxnsDzTSaHCJJiD757hswTBBmAHID/LtTnmS9eetzxEPwF7wuNFlztOJyt12yWXJBGN1agMy3BMtTS"
    "uk5dIYqhySclvQZGJm1aXSpd+q31oWo+K5vS+ne41a+ecKM2BEShXKiHLXWvmQqJtNX2oUT/gjajxS1yrTlancokckW8APe+"
    "7UCicfNFkDTEvxS3aLtk5gYP2+IS8ss+nlJhdmwL4i5T7XtkgIRt11S0ONSg1U8rwn6+ANZUvNeP2yS7eLiBSYP8ZATnKZTd"
    "SrOJu46wQ4Rc+OKl3QA3jKPWHVqQkzd6xIpWO3BnLyUA/fpoEBlEGkLPVD+CBwWRYSHO+gxGhDOHswZvcxaI5upNlwmHAbrc"
    "812PWYI1uMVZxxaLPIciOyg63OcNe/ULAHqEvqByQQZGtsfGQyU2Zy9FggWCtbkb2uCjoI+GNgg5CUJ4FliCcd+369yA0njk"
    "EooqyvvgC2Wrl33BAyaON5zItngA81brHLPbmOOVEfdYx/PlT6wUEgGhoeMwAXuyAKPxJFqdSAAFiGvkvsNn4ANCLnHLY47H"
    "bNey0dZK0GSSy44XyGYCjeQKe9ZjfPVzDgY6t3gGSJaMUCSacUHQ+JxqKazUwD6BQHiBwVo8CInlrgf/dDi2xNqrlwNWorc5"
    "LNlTvM8ilzMxLxqRGlnK8AThnY7wOaNKQ+G3bRdClAwiQ6t/crgDFHAfianbAEEPgEAOWYlEoABjj3W5u8gtT8OLgcsZGzNk"
    "HxVibPfqF20EjTVsz+UJWuyA4n0aLnbIs3h39QqHHrd4n/qr5D1dmHBjitM+4w2v7VnDhkvNjSBKRNUX3KmwfTCFfDVClgg6"
    "0eqfAjVUo8ZAw4tFopvQiYzIgcYO2oAjDDoRxLpzoDF42/G9vmiAfsiAxvaJvnAtrwgvdkAxPosYW70MacZYqcMDYsT0X6+n"
    "EGMHPcdj8VuwnLOQsdXLx+02zWhiBndI4fhpqJjkIEaMSS1QwSawFURIMHf1Zlv4Wl0gdRzxIUFqimP9FbZn9U/z3PFYlzdW"
    "r3gkk9R0XDg9AWKE2DOI18LgCvN8G+Y7zYKUYioEiUmWdfxI1DkLPcsLUJpWL7vNyPFAHeV4lqhC6Gkj4hbMHjhuJ8mEClDd"
    "SWkMpIpFmerDe+DQ6mW3woaSHkCz7Uh09QoqbK+LtQOZqzed0G57AWsgcQ0Ou5q2cL24dIIN243/4iCmoWHyhZI0CQrbDePF"
    "mhH3LW5xBQh7rO7bqB44DSgomWTJVZgw/NZnfhTyGBEGO+wQ1bCOBYufqtZpw7Uv4VcAKw8Igls0l/MLTYXt1RYaboG/BNYw"
    "A5kWeB3PB/GDlYfNccdCnQOLjNtwoj7Yf/gq9Hwfa+hEfhPGmEYDVWwQ2m0aogo7ILp2kCzFkh1Sxi21dtl+Cgy2H/YUIvCy"
    "SLDdHFSJ5cEqlWgFiQRLVB62lIAgYhSYnJCZ1wr/lXyviOWNMOKOjv/al55rqkCC/gpsl4WrV8LI8QrwX/slzSxVLAcAyxA4"
    "Hvwr/REGXR3RwOXF4ttBfwk3xd4cAuxZFIdOBPYBAFe1wjnU12N1H3wjcYH0dGAgVeHqn9wQdBoYDJHoejrQS2p4vQ0d57UP"
    "pBsFOrFZAiEnJUMDnXG/0bK7HtU9w8IoIYYEu+1ZuInM4buoo8Klayl9hFTS+plmEeAebVTjcrGUM5PjbECjSEd2/dS1G3ai"
    "TTRQ117IyF0X+FmgmaCWkIgJtbwmYK59KYsGJnSdz4PqhN86lMuTi76UCihN0qsjufY6rEObDzQDgo7nWiIL49rtteVSgFav"
    "3cR5YHeB+QIVfBcVFlfQLmVtjhx0to+zzurlpg19t5sRuM1EMyLNh+ZfsHoFdAzPQreGyYiG3HrGtmxVRrfIk1Ujhm7t9uB+"
    "BCibehkDt2AmEEk89D1XBMwXHRHalhconM5jIAXUnjT6ckCt3ekZnoJppd99DyAtkoQHxmjROoaibcNCZyqrRz4JzLIO05Kt"
    "PQxKq5FecIvxWbKQ5eGOQ8361S+Ya8NfvO7b8AdtQQogWaQkaKDlWh0b1LwAjaVe+mqYpclXiMZyEx0ZA7MKUFiaKrVRK/hZ"
    "DNZBuxmBxSPQvMoBsBxHDEFg7cuqkDrYCYBP0I3OtFrRFBCcQvd8BnsHQF8V6+kU5gp0Bw/wdbzrGdosgK/iVp8POK4R8DU2"
    "iCgrNkXmbXssw9UYTSNBq/Y6LORtvvq5UoZol65eaUQOLyS1zfuej2oOGKXRqlWj14HUJzirkSTpCKu9Tn4nUEgQ5rXFQtOM"
    "p8I7meHLV0ecnQLOWlvuIrZiZ4Kk2seLhpsn+waDDMP07sAbOqzYD8du0ioodXToi4B5HTBmYLudgkztdeIl3tKWAJiCtAFx"
    "bRfNYvzAYz/ag/F7aSJoH6hOZmyUPutGwunCRkYZ6I6XB0vt1WqKAq7DpJDzkWtb3JohZuCkB89LrFSg0bbtrF5pi9D3cFrs"
    "9lyMAqEfSRLP9UJpQg3WjFYv+xaM11CaNSzUlnxDY6GRzDWZcYQelLCa8vfFyjQgah/uISC7SeRzpG4YpTEjhWWHGDyDSyfJ"
    "rRH5fHvklcCrw3wh5DRK2i0PJ70A/7SP541xV25npYUEljwnDdbGqa5ZkwCriE332IRk4jggpWAl2IKWBO40jBCsqZpiCQ35"
    "CLQTVGTSP4FCOmXIlJOVuhSTG2xFrwZuGkIwEaeMb7ljrDvoedOJ8EJfVpA3xWFLXgBjUi3KnagIyDMYkDOxI16KxOrnZOnm"
    "1ossrEl5gWH6k7cQbUlfd06YhpnsiElUNWdchooxcE3P57cKRdAmNHfngQvwGAWffBS0ORm68wBXCS2ABcCcYStEIZBpX95d"
    "rfup++h4htQJOBmG+KwdLvnj+dxI+pqJ1RtbdncP5jLauskONos7KquQBeRpkn7gPIVeqq8N3hm1sA6HKCWeib5yy7qiIQI7"
    "5Bl8knTIklFpC9izNsBn4pPgUWSIlCHt88gzlISZ4GAmmaJjKe4UAOkxpwl2rEBnOuzEuQMmOCIZARVVsOk7aMvNnteIfB/3"
    "TOSHsqRxm2xpeT+yOGRKsghnBLYFUTEUbaRZ4WTt+0W2fgZeRDa5tn2YIa0mNz1ggNmej0hNFQ7w3AfGFcmQDWkKSwuBxY+U"
    "tk0ARgfJHbz6Raw3GvHRmBhrlEIXUZdBpSXbN/yrPCa26CD6sCNaMoXDwAaxEbebxhhR3QmwCL+bB0pt6QnCWJZ0EVpST+bc"
    "iUaRg5I8kDZs+A5GsQUOwYrYqGp4rtfQghU4IduQzwfuKumuXglA/MQQ/BBumRoRNZZbEO4uX9ofK6y7yx9WVcAPuBK5Sn/8"
    "/IWheklGBXVXNflCwTptxF4VhzPuoOWDShFBKGlyYuzQAc3yDqQThlf1QKMiMIiyVraEC6Xsdwg0wJivfgEZyXSLPYsVetZj"
    "jahDGqaETrQu/TEXCV9f0wrQQs/FcStsxQ9tP23YhdwhoxmUdh2XXdint+wGgsUbjt3I44YOwmJv2Ui6prpAI8JkFiRnKKCo"
    "d1yPccgN2KA4pYHL9t3lS6M9+HeXP0yQQocwUrj6Bc4/6ABigxrC99Et8fyYFarQIuhpdG/NFXoIOsKy/a07l4MBjU2GB3Xp"
    "Zi9IT8QdWIlEjPxJeUXR3ULbK4eCn37a21elOmPiVLwkXXWC+TkQKfosnmB+dgNL/Rze51n5J+0LdayPxrYGxVq0HSk1Cg5H"
    "CMBIu1z3lldRFnaHvhODfISbwvccgDU6sAGqkIb2UHRM+Ywt+T4H7+FOyN0R+J6CFZ+iYdpE94LUCkfyo5sDFbbbc9EtzhJ6"
    "ybDFJTWL8NkbU43oOdJ8Q0E+ntbLAoyPNQr+gJ75tDld8hF1Q2qB1EcLjhzzMk2PPEc0jM8eux7N89ENiuDhGyyE+CBIguyq"
    "+cjFxdxNLKwKG8fsoiZhOmsVo83F0ebC730WRAHr8JciQduqAlMtBeh5MgLIACwdVIEMiCd4HnTle3XaOKNJlsH0qALSFbuE"
    "2SaP92uu6LGfuqFT2e9EPncORI4ISmbHmdy/zywbB/s1lcGtXNuFoB+7drxfoZu/4MZYmXHLxquDXQEZPfH3nOiZs27V1/Kz"
    "7k5Bhw72Db9G7m2yctxSCDaTTehs+tGPf61+0XswLJHvLbZtESw2FjmzWa/fEXdOuTZnnTun+uHti6wnrDunoiag68LIj9gC"
    "v7NiiXnWWLx98c4HDTsHKXrOCsLbFzusbd++aC3244K8bWdQRarkIm+0vF6/x91+/BU1J9q59iAv9aLN3X6bLbLm6hd+32AL"
    "Pu/BZ/bt0ywIvTunImYzrx4scp+3oVMLTtRY9Nx+266wJ60+dY7DQYuFaB5CO7cvMuIj63j0qwFn9zv+orjzQc8OmVgI6JQH"
    "Zz0S57Z3Z0Wwpge5tW5fXAgW+7fPzLBe37UXmMfCfpvZLux7o3mhgYoOaOyWY8gWiZ5+hT1rNxYBLeQCQM6zOH6aoIq0jw2w"
    "KLxQtGU5oHvBc4RNOFIWS0Vcu8H4nRVARzKgtyHmqdvQFNSKvC1ZXh5QdMjvcQbJcHljnlfYfq+xKBb4PHM5mxfzbJEveHde"
    "aSwKqMmJ6mz+9mnG3ciJ5g3G632s3bVvn2Fwli0IeQ/+5ApL9CLnPe4GXo+7QsMTeXV/9QvIFEbHBQhL9C98gYFXa9Hmd05p"
    "WKLDwPRM2im7weaRVQK7mIi55xagiIizdz64fYZFC3dOIegoMzrxTIATiv6iaCz2e2ClNT01NyrssJw+88mowBBhJ8bgv4Yn"
    "8no6zcDbHJ7ocZ8vMM/qeD1bWIt2DlGE7zuETY/yeagai/0Q5pwtimBFzyH/81mo+o1Ft89KDS8HJzrUdxZgSShEEvGFoN/m"
    "DkAV7cZinSNLbJhf/UTnuHaUxhUdTt7A2RmQsj5xrQMzzImwGjjwwJ26x3oNmK2uLeYN1kxm+bwIQoYV8Qp70QMJZB3wY5B4"
    "quZvn4l1njYNYljRi6ik+Hy8u5i/fTGnnIakniKmdfzVL+qgbugrkJvbp0NeYSnOSW2I5ZRagxCxF4Reh1t3ToFmUFqul2jX"
    "RstgMOF9wYLbp1nIF+Rk81xE+vHFCpN9AG7GvbCBkQseqrog6vFEsbPDzKtHOEYdbi0Am2NFgYPQBlbrC0GcfApnfw5eBI+9"
    "ROIkvOhF3rEDCKLjWCp40XNhb/ULf5HBcMALWIkK8EUAbo/mGbQv7qws3L4Yo4wOSnazjud6PZw/Cdhov+fynh1TIndrtBz1"
    "UUSBa8DH7PKXXYZgGOU6VGFwJAgQnFbH8/vMZsGCjbknqKqot9h0bl+0QJ5B9iAdwJ0VYHgf5vjt0w0pqzQgwSJvRO6CLdoA"
    "u1gUFi5XtmizoOPznnXnZrJG57NM7fewM1lckTbrNdUgcUWaElQ1F2CLEh2XL6QQRlpNdejk7dON26dVeR1mlFRWVC5BG9XF"
    "InP5Yq9fjDWiTEZJmRzQSKqC7UKNev26D6KZ/XwUVsFPg4x6ks8NWxQgjODwjHdnBVz6FgwNB97LcclhjNSc6KgUPqkcAiAW"
    "aYnX8UVFA68DjJ70wJEFKxjYL4ugJXFlosnZZ/N8AeJQvT7rOPbCDANS5mNKUiuZa+cARocTJlAxlzti0fZcLzFUKuxZPg+O"
    "lh4tnslUc2NrNAUvet6PGi3eTqaxjjA63LGDReZJCxOj/2CbKFMtBS3a77n2nZVgMbFymOWB0u5XdWQRR9KVnMLE7/Xrnh/p"
    "uKLMmsw8i/WgpgU7Cy46KCexAfOf5j3syB3Wu3Pq9mlQQGBpegvQ5mGqIjXgRuGIw4Ed+DIAZ2mHtBwMYh+GMwcoGioUKUSR"
    "8Mm2IxUIJmWy3EkskWaYQkntfQwnov7S+47XC7m/yB2332ix3qIHtSq0yR7P4j254ndsuyD3U3Yup3M/PeBUHQErWsyiisLa"
    "rjBGFSHBrqc8oSl8kas9MUxax0z6IeS/NoKKCtqagNhWGlMUjoMp4ulVdkjOJ1nIS+855HQHZLaND7wQjlh7xcAiLC3FR7ev"
    "xTwNXQG2yPeUCXH7jBrwxAAbAjGyBYssfueUR3pmJM4oUagRqAevbd8+k0v3xFEruH224HsLKajRQTC2m6tf3DnVL0Ybxbvb"
    "KLBZvX/7DBm0t08zkGH39umcrrl9GrNSoD5CZsKyaDC3U2GPFk/iNNzoad9bYF7d98JoVKt/vZ5v9Pk7K/157GREDQLeCE3l"
    "Kbk1I4Pw9pmRVu1IKglwhAoA/DY2DxfsAjpjukAlxhgjpM+nbyP5MVEaY4tGNq5Di/bpG4s8AQ1+59SC17NDImGaWV4OVkSD"
    "RF+Q2SmpATQR8m1Rbh88SOYxag8xmmcJqoj2WURv2xa3zyQyoqVqwg3E3eUPMVdTarsA7mwY97vLvy3aoZAX3zzch0x7i3Ac"
    "EIRlkYWwgW0B0mjezsCM9jt2sm2UWqGHjncsEUlcEZrtiS2iujjUKFGbcdiI5DBFyoSL7qz0exzJ9AIQpARcNKN7qm6fUbOc"
    "bOUeoIUgfZJPGQ9NMJhhHIlmsCBUgX5mFV2kfkBAdyixGphoOHcCsQC7KSz4fTAojRR6Hr0SEVpfUZoAi5NerIvFji0ai/BM"
    "WP0QJ/bYlJSg5gLDz7PA7OPlEaQWZUYil0kK3gKqGQ5PMcfG/RpNlbgvupzaQQh7y9R2YEsCtOxHWRuAWpe1imrMCh2BE8gV"
    "O0ECxfsQ+tVXP+L8R9lZlZANdg11byuq9YxHxWS3uebNAxc+eC1T3HJdjU2op8iCdsFkBd0Xb9FzsKCDISx2ZH/ifj5Y7DMw"
    "h0kb83xGI9ipo59hXp+RcrusOQNiC5ctBIshv3MqVF4L6AAcN08J4BDYz37wEEkz3h4C+eHobQ1CUMoouyDE85CWQMwHuC/O"
    "bAfSHc2jXYao7kK4z/4CV/Ld5d8eln4DGrkIdTecLPPA/HklcSxHumc5YUzf9WCxbPSHBNjjboJ9L/uZYgRBeGGt6mlOV0g6"
    "otrYgpateERihr3W+9vjfhD2YK0axsXh6B/Na2BrrtRev82bHLzKHb5gC5wG6VxFII59zBuXthR7cktNSpBiQxo1cmt2+yJb"
    "TAGDpOWYlBtLYacQQof7QXj7dMe+cwqIQo+5jfGNOuzuLDk9spu3fxF3PnBsFnrKyUnbJIMFiVkKxe07KwK+a3tN8CtBt9D0"
    "8Bdh0+rCPnkEVijDJNL0t88UmvBZyJC2V5BbgxnN7RosChl+yDgE7pzqZ4BDOkUp37zclsUwohgQFGq+Jygn9V0VdV28/9Jg"
    "Qa76c/jOC/ZYMSgoHAoKepGH/mK/Da3g8tW2G3YGD4QVJnAgcjMjnT3PcfvozfNQAiguBRWh/+/2Rc0BePu0AeIkILZVBzhd"
    "G4RiXgs39JW5zOrSNbzoQq0WZvmA80OsAeEFO6omfkJwVRQjgZ72HDHvkrkJkxk3jgBSKtZhVbVjSPa7GcWkKZOfv1AUAwQX"
    "U+AteHKaL2gEJKsceUIC6TJQpBMUSFrM9f6dU6zXV86SfpW2Lcp/ffuMIvXOKc6Cngf6QzOME1yQXh8keHHR+wscxZ0T+ju8"
    "YmiQFrqBmYlCX+qQuyRAx7vLLSWoBfigAynBT9x6ygZjjVbPdmy24NgLLo4qz8OB9nsQHMAaND3l0fxsKDdKrFwgABXPzo43"
    "z3u2gXY3qvEt3O4pUNCLMvDnuZ6MtyAqCBm5yNtIMQQ1nx+/ctpNwVBpqnsuvb3XdVcH+w66b0TX8zih8QmiZc5Tjlcwkf3b"
    "F9HGTnBCaYemTBMU0TIC/sl56S7UwUKqSmln3z6tCEiqT7BCB+Edzbok0pGAhl5EZs/nUEP7vZ7NLQ81ME+BhjJsDGT9t88Q"
    "eog2nLAi3XmfmpORiUW+6HL0TVYhhZkNan8+jR9yOeCHohhAdBhXbyQghR/qBYt92OJpvBOFEKK6z8GyGoEhGmI3YPgmVhCi"
    "OtQ8SEOJXvTqvLHIYrrjBTeLInoxHnO48sYP5vmIREG6iBSAiIQfYx7yiAc8KMrDhlBQnv4iOqm9ut8PojKzAGEfubj9LOaF"
    "hiA60A8wtji8rRBD0w/Z3ogcQdD8vIjXOWmEnYbw5NgmGqgJreJFF5EO8ntU+ohNIIOxn4AlRph2uQxBfrQAvVaQonlbBxTF"
    "mxs9vKjhiTR3dyZDUK+fXBO7DaTO/ZX318+fW7/25tqNz+598OXarS833n9l85urg1u/Xv/lq+tvfzv4+st7v765fuvyxgfv"
    "/W35VBaWQ9/ce/cvm199mMHhZCpd++7a5h9fparXbrwxOP3l2q136O3fll/e+MWVwc3XNr/9emPlvcGbr63duDB489r6uV8P"
    "LlwfnL85+Oijta9fv/fuX9a++3DjNxc33n/l3rt/uff+xb8tnxr84dT6by9tXvtk44P37i2/tXbrS/rq/sp5HNmN919ZP//x"
    "4MzN9Us377383ebVP2386ea9d/8yePPa5i++WVt5f/Da7wYf/5F6cH/l/MatX6//9tLajStrK+9vXvvL+ntvYK8V0IaYsvHB"
    "e2s3Xtt4/5XBta83P798f+X82o0L61/dGrz2u7Vbt9a++c3g1dODq1/jlwnOZnD6LH1+f+X84NULm99cXf/Pl+krqnDwlqwQ"
    "aj53bf3SfwBBH/xlcPb6+tvfrt24sHbj9bUbN9ev/P7+ytn1S8uDj98HQAFbu/HZ/ZVz2FqCr1m7dWvw2uX1K/82uPTp5ndv"
    "bV4+/7flU5vXvtq4cm7w7enB1fPrZ99aP/vO4M131r88O3jrjcGrF9Y/+Mv6219vfndx7eYHm1e/3fjmKtaJgJp7//7exs0/"
    "alCa9Xdf3/zmP/62/DJRqsA0g68+Wfvm0uCNjzbeejWB0hBX01iawfX31y99RhzJQ2fuLl/aPPPZ4LVPqcDd5Q/Xz5/buPTR"
    "4OpvdfHMjMTflk/dXb5EAnt3+cN77785eOsNYvFohlJHCSVz78yF9XeuK8Iy8Jj1d383+PWFwc23c8CY9Xd/d+/y1xsfXM2B"
    "Yjav3Rq8+c7ajStFkBjiowaJoW7DCF/55N5vvgNEzP2VczomZu3WaQJ1UIkMKoZ6v/7ulyQhUsCURK3/5noaDbP57bfAjisf"
    "x+9BGF6+tH7l91TT/ZUPNj54b/DVn6nWtZX31998a3DuApX52/IpmJF/ujk4e1G2/Ju/rN34bOOD94YKaYx+Wfvuw/XzL2cU"
    "xfqlDwdv/TKLeYnl4e7yhzBrr3+4ee036+eW1y+d2/zjrwY3/0BDn2HN3eUPN6/+afDNrwfnLpByWbvxBumawbWv733zFqgS"
    "bJhIoYeyKiIOqbm7/OHgzc82/nhr8NqnSmedGrz1xtqNjzf+eGHzF9+gOL0e17b57S8G13+JnVXoFZpoOfQKTVB6qeNXBl/9"
    "ee27DwdX3otHTmFYBpc+1ReptRtXQOOtLMdDqFAsg48vDM5+ufnVtcG3r8QAlntnLmxe+42OWiEK6LmGW9E1+/r5c7r63rx6"
    "eePqu6T1BudPD976d8So/G355fW3r63/6peDX53f/LeLm99+fX/l/Pqv/7D++pXB5bfXbp1eu3FzbeX6xtV3/7Z8Sur2cxc2"
    "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"
    "p2qgCW8b1L3jM+DyzCYeSyOJ3pgtovSCKDNbED0BVvvCMgbIpKjpXqqPFLVF06XFaNoaziQ6NadVmqbpIfwzOFXRv+H0PBz3"
    "9zBptK1pZiLakcvASm1qRrUnl2dllr5wTW5bjx8b1cZKitbj0mqRXqzk7HKQmypNcUnTpUtpDKzLiGdZaXkpDQfhVJTfi9mm"
    "EU/3tV18l2e1Rozht82TNlodGymePzXGHYvlR6OWPmW4ZFYUk9biwJs5fTifWyczlgCVwSEa6zO9gTRJS2HGrCrxZEbptZXO"
    "INraRDsbNkn3wuZlmI4ej+VFvJ0vK51NtL2JljdRexG1lehJieakaE6L5lpRWMb/ljyFp/aul01bm2Yu3eotFgMl1hDkw6LP"
    "0nu71W9GK51tWR61LGU55jOX2XjSlaPzPW1UmrE529Gi8zFNc6VtaT6tbLfMdkvTlVKZbsVK7T1dGtLV8HJSno/s4a55qO5M"
    "oDW39Km1SmT5xZzOpDKaOrd3qwxPN9LbQZ/ONxSV3sQXh2U0s8hsG9IRxNL7VVLtX7I7c8JlD2a/NWN70VS7EbbZyvlcFdul"
    "Dt08KvFqpiLZo96sJtXtfG+o0NpBNXS2elGloSbM64IxLY+SHCNWG9xsN6gvTqn2oT0H0XyV3kj7thCfDoxwrSotEsNSJ93O"
    "zStLsZ/e09nymRuae7WpzTp2fL7szHb91nGkW7GekjPyGk1nD8rWzlwEuZYbjc6XiypNR0aDLdOcNjfNREtoL/NVepmt68eo"
    "WO9kOlKmWVoqsUT/0tKZzuIwrm0qsX7qMNhOMpXORu/TdE4G4+gApOK5gZUcSYY4KpsjW1YlQcsNV6u52JSas0ZjNmzydlMb"
    "1hf79qqr1UZq72DubFNlNw1lX87Jq3im3zGnlb1Qnlp8djqfLeX0ktunzX0+Ht/THXZZr48zUkvplU7t8OI0qu1KWk8ZLTrD"
    "XNuebiYpOley1FQaTE61Y2ua21WT09NRnyc3jJaiO/WxNcywxsjiaieL7zItDtB6NVVbiLQgR+lJjzblk1mPSXammx/sT9Vx"
    "y1L0E7ublYxzU05l1Favk2eb9aNSHSd6ycu8POybg+O2x9K1PNM1y+qlpFaXSmWZnje2Srg577WGk8bO0nPDcyez6S6qYrYn"
    "6dPGkRWjIq0aZzW/4BaZvKjau3Q1LeYOO6uT5y+py6Z/Sll1qdZOTaJ1nlEu0/h0PjPFVSJ+XLQr+mSY6sdq9RrdYrva+EzT"
    "dPTY3sT7rdl4OI2KqYuZ7TWsBdOaHY7GWe1oi0xrlF62pzmJSfKTWDsqcunsrnSogph0yBvbvGWrCTnX7g47O1ZQBpPofFWm"
    "2bI4bavV1SZR6ufo+LE8SPZTG7HV0jVaNKry8DgD2pAd5Kf2cWDkwhlrNQ+XVrF8P7ZfqaysTXONRs9QBsclIycabV2zwoZU"
    "inJ8tzrnJKvK9/Lx9jax2EVT82FsdJieTvpwWa3NeKa3yiRndHbXyEQH4XwrXc/LmwR/qVTa9WHSmorzCVcdS/XY+JDrVmrc"
    "JtvdbI+XRKXSpDvJthyd1hTOavLdZJ9u0LNoPReezTZNjsvw83A1P4g1BMAM2q1s9jzbbmPNeq7RrEvV+Ey08qt4eDNsA1NW"
    "6uOwuRtLo7beuZxAdqod1Wo8I+0nFnuppPh+/RgN7+1uyrSFwbx6miy6FVktiQPtvFpttLi4SaYz81W6NK1Up1XZ7nKpqJnr"
    "Klx5I+XKQnZyTKiZfS7ftXrMBdCzcnTW3FlsT8u0BXF+6fVp21icxluNn0gxrpYY5vRtTKvVRnuQWjCTmsSnVoNpjE+wVbUZ"
    "leJAXu1KpUZvttB6PHcZr6b0fHU5xsvHoZbfJ/oHedqo0PNOeFAbjlhpU1eyKXDq9DdcMr7Vzr1OQ+PTercZ3aSOU9Mab3IK"
    "k6kKCzm5NfUa2Jm7Sc/ctrdytsTvF6sp2z2mwvJiIafo1jnGNBY1rb46VA7N6k5KN8wMfRkd9VSPPTfZeKVxGSVzk0x9UOlK"
    "57au5U4XaTGb7Lqgu+mkezulkp4a4WTZNobbBV+WkqPxONOZpvP1PUdHm2kjseklWzlTSeej3VN12Z/NzRWtnNuy3KbDFrDz"
    "sVXsmJoM5lJzEO6nk7Fy+qjYC8085va5LD0y4sM2V8pkFpX8ZpYs96zwrp4570ZqXOjtq5dTcrKcp9vHU9SUL0M+wZZ7mbyQ"
    "k/m8NWtOd9uqkV/VmLzRZRcHO25kOsfpYMoKwiq/O6W23aMxlAf7A00PuMNEz+Y7o9W8pUjzfkPoV5slzSg34jOz3tFzi/yQ"
    "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"
    "h7lhb0mLh6EgtRslOTMUWqNxN3OWGIFpXdrHZG6xOQ8qtDpplhalRmfcAJUqLdEd+2RVrD7Nn6wT0x9MD0u7NuvsM83BPjNb"
    "paVJkx+mzTFrxdvSfKCoum3xbYGblTOH+H46S22i0XzMNNnk6jRVtNqsWyqvFunpAnS4/CDZUqI2p1WHx8PpoAw7ue7ULi1m"
    "fHwmzRZCtTTr7OWtvmomNK7dCW+inJxN7yfiaZo7HeLnVtK+yEMjqi9L9Xw/HA7no3NJPghlqzUKC932sjJsG5PaPHo6q1kz"
    "G7N75Wn4sNmcK5tOPV9PZcezmpA5XsogHqt3R1uhUmNSQ3236iXDQra+ia7q6bQu2ccEI5ebYndnV6KDWvTYaVa2m+62CoU7"
    "J/Y02G5KSml/KeVr8WwOcOaRFeNmtMFVS6rdSs0vJtst0aOKXWtmUrn5iikdzIZ0jvKljKHUwmzi1D+kU1a3cRqqJbqWNjUl"
    "mcgx3cZ2yNBDflRdHC/WLN2vV+eZZJOfi/RyVBsP5ryd2XJtZdIUks3TgVG2MSvPzDaT3TEVmzWH853MV+MXNVvKLUWQHTVb"
    "jWN/vjH5DN+wVqYVjbGWbR1WZzaX5nk7viyXp91jSQlL0cqxJNFxdiLZqVx40Vicyqtmt7EfqsN21U6XtfJulDJGnX1ZbA31"
    "ttaVF539GDokb9dEYahPGZ0bqoZhmsP6jFH5GAAx0c6kh0zpWF6WO/apnjLpHS0xDe18bGalWKVDd+nqttrdbpvj8lHfxPvH"
    "WLI8U9vDAV3rj+vV6n6Zqo7Th8Wo1Ks06u2VvS21Wu39orwdpGs9ujmdTqeZpq2yQ9lITvXSoQx6zaF9keeC1cnWFlW7NRGn"
    "s1EmZaTOMXqV3zZSJbrZbNe7O7224aO1Dd0dmfnyeWDTe0GqVMrieEdHV/Nj6dKoloYzqb05T8F0GC6NRuNRpSrNRX3aNQV7"
    "kV4eLlaC34E6M0vH5wlL3lsnayOGNTo6XnJmuin12vlOQpTLveFAHlfP+Wy5HFO7lZUwP2qDeW906neMtDyPn2NzuSPF4ytO"
    "7SUXE97uzu1GLmca6tiu9QajudLVrFWs0ikzSduYia0ePRToxbZKb7fq0u6PJvshfT6V6fFwanbDFdAsVZrNcnVrpvftuSiY"
    "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"
    "RJH69QCid76940ZLfAPWZlm2DzhXGeOVar3pMH1Yp9Y0YlQksNlQACBut+tsrLDOKcqOAtZG6/UqynQBt1dzTwhpt9fX1wCb"
    "FZlKe7RgqjuDAwtvQcNdo6zNe2N46vz5Er1YDNzv3+IX/769vMfvA/qCIsnuGwZBhQoEJd/iJzffjKi6lnzhFceK3hx0fGD9"
    "lF1du3pccA/s75eHOF/eWcCpkh292ZUqunx8edcVoyQniGEI+uOIVQmF+MHLn/H7pscHTT85UzPtFJzoNctfRrGX30VNCEq8"
    "YgvJk7ucqXm6cVON38vvOrAlxXiD3k99jhpekfQRvVj6mQ0v+voGF6AC8Fwzhu4rvCt/Ce7jzZcFW/j+ZhraocTJisa//oj6"
    "ZAFJeNWBYrx8XsH+eWxwD+mfL58Jzmhof57HMT1hiiELtuKmxv16fEs6//IOuOhp3NLt+Fu2ogP78PLOAVu4L06mJwbiGbxg"
    "R884l0xRdAT37Qci6O8lX2BVxS0luHXTdOXoUV1guArQFOAIfNLaFqLmUZFpuYquhIImSMeni99jqKd5+4Gde1n45+v5zzdW"
    "EE1bSH0AoivYn6wZRH2OILOmzQt2iTWDC4R/Jw9F//V5mjDDNIT/o+iWabvAcC8VTxPgua5pvCqG5bmvjqAJnPsadQDYAniJ"
    "R/003HdNZfhVRl5l9FXGXq1XXnvl+VdPezW1V1GRPFt4+TwtxoymCezM2dOkl3dNcF3BLp1esy/9gCBc0K/2PIFDNwOJ3uOJ"
    "6HuAKlooj2GmJgd85i2mFN63ZAwV07irbRqX1ZfJADLAJd1wAasJr6bnRpMU7yvD089/sIL9+sMqxVOY/Ez+PvZ7D2wFGG5U"
    "U7AV7s0FrKcBO/rt5C0HkL0gXh1Pjyh8/bcLWMXgheCvl7f4ge1S/DC3JnyanhsN4Fv0rKtjagpfyNh7x0qnjYVYwR0hOQvS"
    "4WxT00rJwirFb2q/VaG4vaeUtOgJ7vzZDpy7hfUjXlZxafxibXTIvHlW9LA2cIT32ynUPVe4XTJQ1jKM6gouyKfF0V/yoN8v"
    "wFPZZ3aLpIbj2qYhfd7QdmxtGmbOmRcVvaRY6Q8SE/SklWt7Bgdc4TOmOHq4V9RM/03QNMVyFOfdlxVXiIcyOm59G1hXR8a5"
    "QcKSEpiCbZt2iRdcoGifaQCWLZRiEKdm8a83YBziBfH+HPX3o3wcHPuT0xSrZEXvmCuGI7jfceiPl6wuHE8fK3g/zYEVvFum"
    "o8QHJWAdU/NcIbt30XH3KQErRYeDvrxbgI9fkr/6mtoYDnb1i3h5B5oiGfEr0dH77vHD6ucjWLIV/j0qAooh2G8nvGXFiFdO"
    "NA6nFQE4NYOcLOBZZ32EKIFkm/5n0kpxBd15i45WwX6/AY28JGJKPHF3y+FKgkjoi4bdPgkGcOG4WHllr/DRSZcpJMVn3st7"
    "8v83KD1JqePGU0q+oGnZMOL5P4E4vet9KU4+HBcOB2zeOT5zHf9duu/09ZdzrZf3aABLrqBbGnCFUiJjOW+2YAnA/R6JQSVR"
    "cV8LumLoIPjOaUC3viP4H4VSIQvca+H77WfslYQgK3gplAqRuPRS+GeBJMnXQgIKh34BFPKKYVmg4h8vrwVYtF9eMpaGqAga"
    "f7vI4JcvFlTc6uMLfv2AR2ZyufiYjFffpw6C0kWMvGzWtOz57+gV8ZJrGsJfL8cZjn7cI7uUlRxTdK+WiuLc1DCN9EB4dvSW"
    "+cvPC64/E8H0r2yM5yMyH+exSvw5H3NK2EgjdwWg5aCOih7hjcq/RhpDSWN01EMOQkd92E9HfaKPEYw0Ns5kgZY3uEnhI5xJ"
    "ja/RHiGlMe8VUxPyMCeFjzAnNb7GfISUxqwDSTBckIP6WPoI97HK18hPsNLYNUUXclBHRY/wRuVfI42hpDGaNjCkPJxJ4SOs"
    "SY2v8R4hXa1lj+MEx8lbz0npwzWdVCmx0uNV7QL3GrMPbCMHbVSkGNIjtMcqv46Wj4bAzkGcFD7Cm9T4dbSKIZo5SKOiRyij"
    "8icR/jj+ZIHBf16Oneu6UeEJWHwc5Bcf+W1+hQtPvrL63EFI+vQPMf4PEd7Tks/pM4a8p4+gc23yPSXjvP0DgiD0/cTbDNc2"
    "tRs4LPqeoacmROVYqQz+euwKqYP09aogOZUVTcjZq9fDc5nO267fz2IGFRdkz3TojOKieV5By9aHv9R7S3m4c9D8sEoJipIj"
    "cKbBA/vwS+STJyk2aXT8jFeeR/dmmO73N15xIuGJf3mTI1npAQ1pdE/RiCHPE3P5FO3EtPQOFaCbrkIQBH0JOeFAidFMBrzp"
    "R4AKUCEyT2ROkqeUZME2M7SmtECbodBhXykUsV3rXhKOsKU2ULbGct4aEZJzV+I9Xkg6BFtBIR6Ykq4E3xWj4NgS+1q4BvFS"
    "wKA/XguxpcMCtmC4Jy1HVqwnFDz4yz5GIvtRI0MgK3j/NVtMZGDN0MhzdIDMkTpJ5A+G6V6ZPE0nVCBONKesO6c5O+rXF/01"
    "GrX01N0Zk6Mhf//vTiiaP6EJabERLpu2kmk8IO+KssiKnNpsp/31mTFbN/cy1+asatrUd9mmKUXw2DKh5bpCKVXhlYCglwxQ"
    "Z072mi47ffzkPNsx7YjnlYCmmb7A31keT6bUaJlGdtXEBplQ8PL5sDfRgCdKaKLrl1zF1YTPX7I55gB0ZQHwH6794cqfYSk+"
    "g97grM1ytlQ5rsKph3fXtNIG+B9WybJNyRYchwV2aQ80T/i83tTYTa9cU5I0IRnem58lThY4VeDfFOd7auQvA/7yyAqe19Xo"
    "UujDtd804LjJZdKHy3+ebztc19RPnPgLEFG7vWC7Cge0Uszf3nSF5zXh3sQuv7r8S8q2nlssGHxiEE3gCUZqCbnyCULmcojm"
    "MD7AYsvb620zITm6vmj2ufUcVxEPZwNebHO7IiOeAKCZ0nHQIuvYnTX+yPyupxqwmuK40f9L0R+fqWvO24pZez/jCMninv+r"
    "C7wCCt8tWxAF2ynZAu9xAl/SzXjlJj9fPn/lWux4McAKMtgrph0bei7XXu/xRkv2xfWF2DswFB1EBSXes+M/3n5AsO5kVlFc"
    "IalT4kzPcN/g1M3auVeiaUediRe+8wY4V9mnO/Pfu2mRFUnWovn6mRwCCmcaiXU16vN5JuJbk0uNL0+wnxfJ+guT6fvvnDIX"
    "Ai0NcMLD1XMRmyJyTgec6P7m2ZuAApbFAvvCVaFsHvp+JURGOyoWQSEIEV6hAmYFBRhLfcRuTPQJlnsjffL9kYUdur4pSF/6"
    "40h0N3GWXq7r3Rvcj7gUPg8VHMlst/Bz4ejAVj+fEHoTXo1ClyuV+O/bKXugdJ7ViwdL5LI2UuQVopPJ2Uuvii6dHAuQFCHx"
    "3xkb4wjCAPrd8V36AcHZcmrKD+RWjqycbNRHwJEJwnOeu1NJKx4PbeXxTjr6bqSR49D17Uv23dUVZR+nCyTXc16v187VTxBf"
    "dTsXLnMFywWsc+TM0c+kE/FWv2z6nCFAzvc5pdOWie6pboYms1elIDHz31Bykv0sU4mxnJYikrEUz+stY5ZTR8llraa2nggc"
    "9+U1kd1vP79narRv0I1aFS+fnIEhrnY8jNwqLPgDheUyGEc1P8uicFv135H/wFEKFvg/XdvLOCie2LtXcOPj8/OysiNR+rGy"
    "mBK5jqNxTzxavRlI8mqwcCvI0kZSq9mNBKrIdyOhqnphFNGfd1YENFtH4zw7UsWoeAXcKWhputM1b0jHI8eyvF2VNSeF+/H9"
    "bcveze5+mk2lj42s3Ruzpqvz5H/Pp2ThgrjwPR7/f/1ZIaIryc/bg/L6ljmLsd7eKacZ0nFBHDWJjAvs2EstuTeMtyee2ZcS"
    "cddJweDjzz+f4Jip4cKiJkkbFhhGbK/68lD9yqyQho9dL97/hN3o9qY/n1sVYqHitAujD9VMoSLpemL4S2YBP0kjN2a3xO/n"
    "OA3oBfjxezQLqHUP9ePoIXM/Z1jeaX1snT7lUn0+sePnVnoKYCK48oqjK04szP91XpRpqqC044ktaCDSJK4I4zTT+VoqP4o9"
    "KW4W/51jVrs5JvNOrOeE9QzHmUiYjtDbJ75636McI/Sd7+wTCxfGsu1jpiEqtl6K7ASf97wsG/O9pCnolnv4vPanOQ7GSUt/"
    "xDWzXYOIG9egjKPv1gM5h7JkM13EqZNPBIZxcm6bJ6T6s0iPpUR6LH9JPbTUPiXSx6R9PKuIRgwy3SUlurj+NNltZMgTFfft"
    "eOicfL2wVD+SHxkqQQwq2bmcqVuAc49HbhpHAhAnLvDw83kQ1yvk+eqdeU4CzrEU42S9OWv4ZxEhapG+B8mQHY5dS1GSSJUn"
    "W8bbD9IpRPsU2AXFEBVDcYX4CHY928jtcdrylTiOPeHbdrMXknb3WyHbWypipCcPKeg18kIqXKT7L+YkXsrZ2ufRDcm09ed8"
    "2+CXfC0cq6YOuCtGctMmu4fHzsERv39Fog6eu4uKdqbPVUL20WweK+43J+UXbT7eRMU+GVezXUZP9Y/XxrlWx/f/vF/gDeqP"
    "h4b5G9fn6KcFHMc3bf4Uc0DcyhHX3mHXSGMH1V/zSTs5Qlxbha9FySskR8NV5J4p8H99PlwWyZp/0Lzw/HTFO+9W5I5angVu"
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
    "iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAJLUlEQVR42pWYa6wdVRXHf2vvPXPOvbfcPm7pAxraUiwBbIAPoKAh"
    "CoEAEmiIrRAwMTxEDSKoUTA+YiIKVkgoEkGD0RhSvDWSaNOohADGxBpQA0ho5FFKSqG1LbeX+zhzZvZefpg558yZOfcCk8w5M7P3"
    "7L1mPf7rv5YAsGncsn2zP/W2n5709qz/+mySXawhHI+oBRFQ8kNAtXvXGdL8B4T8Xopx7fsD1d770p2kztpgrd3XjKOd6xc27911"
    "142vdGSSzsWy6++5dkLNfWnQJZqloKG3iHZ+OouWLrVy37koTe1/X/uGusNiEBcRGw4vcfqVtx766iNsGrcCsPS6LVe9Q7TNp21E"
    "vc9nIz2VlL+40ET1kFy7ff8f7FA0BBVrXdxkzPqrDvz8tt/K+pu3rt0zOft8qjIiwSsipquBAXJ0BSirUkpCzydYeU75I8rrKUHF"
    "SMOaqZNXjZ5u0w0X/LhFdI74dkCM7TfhB9SC9Px0gI3r7lH9mPxeRH3wrtFMZlpDLs38RvVeRYzpLSY9vylrrOZ3pXn9O9d9lrnl"
    "zQXT3oAYo1lbE+RylwWORQMYqSyYXxsRTEf2jj+aecxbXAcVgioighGDDBA4hECo+nOhRTQQglnhvGrdbQqtiDH4dopvtStfPcAH"
    "q0NDMSaOciFmEghVKyg0Y2wcEUJJeyXf9EFxhSiI9m8sAqGVsO74pVz44TVYkZ7CahFUcnaBzAf+9Nxr7N1/mKFjhrni/NNZPNzM"
    "NdoxpipPvLiXV988jGnGNWAQOkq69i4NlZgwIoQkZcPq5Tz+rat5/eAEaQhYYyq+pf3up5CFQDOOWLFwhPO++yvuv+FS1q9Ywv7D"
    "k0SRLcyvOGM4cfliLvrhNp5//QCmGXWVTPEBzghORJCK+EYEP9vm6o+dxjOvvcVlt/8SRodz8Nb5orhQ8WzC01s+zy2XfZQ1S0fZ"
    "cPvDJEenwdlcy8bA5DSP3fk5PnPuqTz30huYobjuj0hu4rpr9cw5Md3CDMfEo0OkPlSitg5FzgipKpOzCQ1reGc6QYxgR0eQwgDO"
    "GtpZxjtTrcJyUgfewg/dwKxQmC4UixkRBMF04aAU1RWwFRGMNYhIYUpBBHwIdGM5BEJQrCmlRRmMR24+1I+cIYRAdniSTIEQKuqW"
    "yj+kYmBqhswHYmeLgGDge93nWjorgrqqStHixUbEky/uZfyWjWz95mZSBJH8tNKBTSnWk17kh8BQ5Dh3/SruePRJzl53HB3+o+Wg"
    "6lxKyWVqXEQwuWNLn+29Kma4yc5n/8unfrKdtLtiQIMn9Z7ZNGM2aTOTtJlJEmaShKlWwkyScmhqlovufIS/vbyPBc2Y4AvzV9Sj"
    "qsUzGQyxqrgegStjoKDeMzLS5OPrV7FsdKSLYQP5QgmmgoI1wlknruSZNw6iBTbWfEwGAUIPs7QAbFf3p3yDdHKWmz59Hl+84Ex+"
    "MP4UptnoM9FcfEBRbFC+v/kT/OzxZzk628YZQxr8IGZRyWIywAd1DvYSlDXHLuSP/3qZB8f/CmOj4MPAwOjbwAi8O8NFZ6xjxcIR"
    "shB6XEDmSJeqJd7Zj6uujytrSVoR0syzYLiBWzhCNJpv1pc2Omo0PVZjjSGzhjiypFmYg89Iv42rOFgKb0dNuF7IixEWLxgiyzw+"
    "TVFfYS2dd0oCpsZACCxoxlhr6q5QutGaRbUEilqYuMrpBAIgQzHju3az/ZaN/P3eL6ACplRvNCOHkdyZk9R3oSQoxM7irOEv/3md"
    "809bjYZOoBQbS41e1RVciOO0StVVCRqQZsyuF/Zwzvd+zYUb1lYYurLn0CTtJAUjrD12Ec5ZNIQuAP9+126Wj40yFLseCajVNn36"
    "rNEtENwgVzcihHbKutXLePTmKzDG5KmqwMjIWQ5OTLH5nt9x73UXc/aJK0nSDGsEH5TYWTZ95BTu3rGLti8FiekPf52LefT7oPbq"
    "iAK0jTH46RbXf/J0JmZaXHnnNtzoCF5DXlO12vzhjqt44MZLOWP1ci780TZaWcAUuVXaGX/+zjVcfuZJTEwnOZecqxqcg3SUgFp6"
    "Wu1LjtCwllcPTPDu1GzOjn3AWEOYnOGVgxMsHW6w99BR3j40CY24oFICkzPs+d9RGpHNKVTVz7SUUsswU61nRHJaIjK4wspCYDh2"
    "iDU4YzDW5OzEGRrO4kNOCCRyWCsYZ4isQSJLw1lCqBfp5WJM0Ur6kxoem/4I78/LPmiNC3XWz0LoI9TlooyCVnXpl0j3Sb5FQTyQ"
    "XpuknEVKBML0c8D+8jEAqfeoKqGg6qqgIYeWyApZ5lHvCZqPBwUNgSTz2IIHtjOPdscVHxT1nlaaFf4pg9k5lNhM5yzBTtsHTli6"
    "EBs5DD3yGkWW1WOjTCYZKxaNMDLcwEjOpq1AHEecsOQYjsy2WbZwhNVjx2BRGtYQW0NkhEYz5pTjxphup/V6uySyGyR5UEWGG/zm"
    "6efZfNbJvHj/l/BBc/gJOZF9t5Vy04M72HrDJbxw943MtNpYmxPcOHJMTCfcv+MfLB4Z4qlvX8PE1CzOuYJdK5E1TCcpv3ji38hI"
    "s1KP9ABd7Gfv1jCgGSMihHbG0FDMySuX9CGXKux+6wjJdII0HKcdv5TImi4tDQov7T9M2sq186HjxnJe2CklFQLK7v1HaLdSTByh"
    "le6FFvVNIaAM7EmICCEESNL+2kMEIoexBg2KttO6D8Wu25EISZrnwKqrldYYBIdWBGcKlYvUc7ZqblYZagyoqTRnxAKmGQ/AWO2S"
    "TtuI5kgY2iO0Aw4j4CKR/amwsqAIUm0S6aCEVALfTpegr9lEfy8xzNu+G9jeUxWDM+w3DWfGxUWCapjnhfd+Nm9Fz/zNz0qGQUMQ"
    "G0nD2e3mlFWLtjQ0PawmsqC+v9H4Pjd8X8LKez/L061XcSYK7SNrFo1uyVvA199z6TueHT6oSMgyROz76dj2NY3eq7ta6xxQKda0"
    "6NpZGzeaLDP+yn0P3fqYYdO4PfTw13aOmWxjw5r9RE2nYkWRwacUJ6b0zJTOzpziuvsulbUq4+KEqGljaw8stf7qNx+69bFuE73T"
    "6T/7G/etem2SL8+0/SVBdZWGYGrN8xoz0XqnVMtdUxk8p+TLxhgVY/Y1ndu5NvYP/HPrrW+waZNl+3b/f1B8wIlrFXlgAAAAAElF"
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
