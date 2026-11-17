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

np = Image = ImageDraw = None  # numpy and Pillow, bound by _deps(); OrcaSlicer pip-installs them with the plugin
_DEPS = None  # what _load_deps() found; None until first needed. Never loaded at import or while the plugin registers
_DEPS_LOCK = threading.Lock()


def _load_deps() -> dict:
    """Import numpy and Pillow into this module; never raises. When OrcaSlicer's install of them failed (offline,
    pip error) the plugin still loads, Arrange and Fill are refused and the page says why (see requirements()).
    Returns {id: version str, or the import error}. Call through _deps()."""
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


def _deps() -> dict:
    """Load numpy and Pillow once, on first need (a run, the CLI, the page's first sync's worker)."""
    global _DEPS
    with _DEPS_LOCK:
        if _DEPS is None:
            _DEPS = _load_deps()
        return _DEPS

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
        _deps()
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

PANEL_HTML_SHA256 = "c3f60ffb6ea2788a7708b988b22fa2c25f2e6e88a9242eb68d8bd55c3c7cb101"
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
    "guO7wJ+WS4bLqBbGjWs488YrWUlgDCCz+w3nft/mm8Eqhfp8r8/c2tzSXMLKDKu8bVWOUymhae/UVqbY1v4pG02y1IgFfbzw"
    "CyHHKuDNit6swhssLwzhpfRFXFPG65BxZGDG8tEkC7kvJMsenBzg4IByvUMmYTRKJVw1FovrhWp+lGshxOiM8Q+ey81lJoUA"
    "4gnHzCqckkgQPV8thLhs29EmSmIsfwejnH9I5Rh5GZb9/eLtmzFdAtQa+KGXkj+JmslfBr7sHcxK/hJ74z+HPZUfZsvvil36"
    "+EBko2PLeTYDPmwstTVKNimbj41c7UuZppLPLNdzbpg4T+XsBW52drJ8JM6Xc6G5ZPzQsS7buZIvpH2oZKwfC94WOzhVX0DP"
    "WJfBJOZQKsuu8dwYrejPc3gRhjyz/AX1TCyXTJzDEsiXF3jrSR8f5Hgl6dqsap3ntmPLTPK3kn+QAk8hYBoba/alrU0qgcc8"
    "2I1qxitpi3IjV0JyfF4UpVXXUozO3HOthfsp12tZ2gYuxvhcVrLQ+12fsFDN98W+kSsxmriUm8Lot/rHve4LXFwvmg9qBwkj"
    "mSRvZZKkb+WY6s1TzLQrDMgS3rpG4b1WQC48IZu27X+L2Zyx8W7fbPBTdjphWTrsygONmTDWXQFhxyypP+cexR92O+hMins4"
    "TncjF/WWNrXkduqzUts80xQludsFnTkShlZ4Nmoqn9qpPDlheibnY1d51+eLJ+H4k/hd/HVnZLPfPtCJJBl24n6nJv8/dMq3"
    "pouK8D36j8vozF47Fm04Pe5OKqfW3PmJfSuxNo5DqnRRVXcHSOu6rtYpO5ychOUuBPKzuA6NvL6ALsJSdEWwrl6v4wGt9flZ"
    "kpyeRgWc4eu3SJbhE/dpXx5yuo7PeCuxz1M5dUxNX6v7/BCnHRVEvIYUUZauO2r8da1Wj866rrH17sERO9ovOPOWa2yXFWdc"
    "P7j47FM9tceLz87HWA1N5AOf4dSGIj0RebBM/3Jm5ykLy/DoE0+e4tUpP7o6qWp5XJ0M7R6dsWn8sa8E2LnRgGS6LUS0KklG"
    "clA3JftSdjgikuQ98D6ssDjbrKd2cxD3OMLXp7IuJo9+WqPr77OIlr2VPUPyO56CSPaPCO5b2Q8oUlHJMpCV4AH/jH8nkW3+"
    "WRYfLqTlbx48TtwCWmt/iqzkrnHN6xPeFaoaJK6r4qoRf6UHLW/t4K1r1SCtgXHfV9L41Pud8YutPxGyvrJEnD5hga736a34"
    "4vOeVEb5v/jcT5T//ou/8u8kMZZQeJKk38nABEIKtdQadXUlDYirOl1b4FUGBT9JklFc8GdPWNtGz1+27beuQKRzdBONv5gE"
    "vtYNfuoWruvRE/6Lb9FX7gdRdPGMW/GNnD4jivgNHPYxqQzFBUL53hf0TEjIb/lgSD5zlKUnLL6FQfrlNgUsg6mcArWC+X4h"
    "d+xn4IOm/ap5cLn8MhjbWl/Yeue2Hz2kbNikJ10XZmA4n/l3clysVvHSCKsqHz6m7r3Z69frFwouODgZ4eHwtfSrwGdMia9Y"
    "uQxuSH2+ruPfw9n2SvLXst+13/a7U/rp+5LbPKVhEq8hu5As8wmvJH8FCf0G/xEW7vfy5KRP+sMd+6en38tzfw8FKvo6UKrX"
    "dPS89kcEnkF0fFJNUznYmDIM8P9yPDyxGDq+XvXfvqKkV6FYOw1cAxULC2W45aOSbVg/sCptv5c6FM+nlh1QvNhZoTu8ZjO7"
    "MfXNo4jgfYXSi3CxF5LWnp1a1wRYe3Z8LU2jai1OJ9zigfkMCcm3Sn8QIHzdOcoCCfw4QURCrfe9UAvPF7+CuRF0jE7N1I+6"
    "wZpeyN3UhAYIcTrJUyOEADmRFpLxn0H8yP8G/7PMCsPNcf3mqMn8OMGPrhGyk25/8b5xQvcd+PrfGTCcNmiEazaKXtzvtqVX"
    "Zb3d7S0cj+lPMh2mMZ/pge9Zf0UOQqfxAjdS38qfvFDOr8TPgY76h8kXrG3TfpFO/ofLMQmJ34dx/k1SpmH6b5KP5Fg1Fxc/"
    "wl3WlffkyyRJRzQUcMK79rTtCIeL+TZPw7Z9Mo2Gj2vxjBuitTIitLg4p16gutapHC/ogjpNbbQkztq2AN1deE1XZlfV5MmX"
    "4Y3QPHx4cuI3Ch7MsDGid1wGsv5MaGiU4bh4o839WdeP+M89bQLJ50ruMs2RtdxfZobDyoBfCmSgIKxKktSMXapQ3LGl+0u3"
    "FBlXSZKqkGq4DLl9Dj1u9pcNyGVgK9CD4SOTJLpfSezQP4SG/3XaHzjR648cO3x0xroRsG6np3rclFD+tti5P/5Q1yAqiyjt"
    "3/xOP7hdnFnuSiVJsAWWwVcEEgK/3fsOYr98quWhWT4HMl+4Xvg/pZjN+6Pi70DU/ymJvflGMlxVk751v8J7GgDIhWznNyCm"
    "Ea7sto2o5C+hL47byqybR3mP//LFPps+C2QdyHPEJzwTsuuw7b/BKffDg8wilETn7P4y4hc9U+FXf1iyPXuIg+VvNuHBr57+"
    "zhM/3KOFXccf/xkT6xeNb5pvj+NAi3u01fdlyPBuiyHn+kHeHXGy4r5ApLOmKD94tcOzth19I9v2mbsG9mvfER2iNEfNwqM1"
    "mm0LbRsJ8Yzdy4qc/Q8yfcYdd4cbJU9tGOBn/TnW/+znIkoUlmX0JOJEjiLWoNexg/MO90mY72iweX/ihDuVT5n22ym0k4df"
    "WKT7HdopGX+wS/Zo2/3rProHEiCn7kGyXgvoWRW/mnqy+5s8OXFLma4DMlwMJDsA+4bb6YhhhnXlKJdb2kxCoi+EIcZjf4mH"
    "qk8L+/EPFJNE10MUf9PxCR80JZwH+L07TXvuLz62cU3h+Ydf7S8bdrDhxDuLiK59iNZirZ0/7nwRUzj9JR5oftdqRxn9lpag"
    "IwsfAHFBfI4NV1JUA1jhxKCgljeDJxU/9cNQ2Eg8/41Mkme+39Y6HcHUtq21XisgnK6BOek95QKVAOAaMBdo7iHXY8m4we1v"
    "ucFtryGBdjaLDtbaer0GV7ygBtRRA4AE1OwA68YxRDiEDerlUeDe88WQGZYQh52/BMptlqwOStyG9fImRTqCQqgkuYEukMIG"
    "duSSZB1Lv+ecxt+waV9UCsMmzg9p/EHbwpOybTtapACl0ecCxPugfOmcPra5UcCSpDDrnjbVeI92Zx7mr93AMl70jwrUp5Yd"
    "yqKRy2K1WmYqj16HZrMsDSXAJ+tURoUa0FuQuGyKJdH5vsxAufaffNdIu8yGeeCzrsPtFua3sb2a+/cIZwFTJNs2LUDJBGpf"
    "U1i55Moy/k/4hOU2+wZ+5H/D/63XGPxm018wA8so7Tfrf/0SK8xK2ytucalj9UdVRUxARYov/wXV/ZtNqfpfLGofWIY/cAPu"
    "rTgsFgiNWiwy0OvwGW2zMdVRm3l/911ZvPvyoxyAQ6jcK1CE8LLWZWEBD9O3vqEMbBy99MMByzjHQc7w+6C7Oa546d4socpU"
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
    "j898maoRnTIcH1GjlRb1mlYNmvpdpoO1C/T7YgMazF32tmLN96cBrLFLGoO2TwnDtMr1Tx/q41HD9U/hkb37MTntckwaAzbv"
    "RFaalpB/zn0GlK2HSmhyaRGrvIfKdhM5uv8JSAc1vXnmh+8687afxR4moyorY196D9NWb3boJ9+hOXq538XFXtGn8LJNI+Oy"
    "rizoIDdf6267XMqto4mCAVLA3gUVFHpSO+BXTWV7Hz8A7ShXaECWLDNr1hAl/uWATUHFr8811dnN8Lr8Ynx105cL5EnMC3Gh"
    "LOwvTetAr3LXCmXfd9d7d2X26mZz/fk3F0XV7Q4ItnDRmb26Qq6Mc4EtVuR/POUhktNekwsZtXON9gnZCQkrunbXBVd4riT6"
    "4qwImB/2JoLfqMlLOWdSLqtyrKyYxYqbeg18xG8n1NpWKn1nXypNXfteLTW+vlSPQPjqkhTgU2yba4WfOfpVoNe2t9kZHBkK"
    "HY7J/n6GlELWigHIWVB3z8QfQRkOND9bcSTRG0M88BKNoSMUnsF5HRoaQnyOuAf6y0Yf2NXt/vYgRKVIOXtEGGqMQjOTLYU5"
    "NK3RuC7Sd23rAbm7gAHlGRDwk4PQGgfnIiKvTy1QsAK7AMKqMFjYFwfHha/m6CdSwI6dQ5s0hMWYFQbiHIgdwhso/Ou3IKXq"
    "RSlHz3ezcupqA9RiZiNO2HnFlYvSLKXXZM+yi7XKFXmoipNoHVdNV2HF1ms7dF/jNI7CFIPjKsqxk7ckmOQXN6ZkN6ZkN6Zk"
    "N6ZkN6ZkV2FKlqNX18sNYLeozrvcDKtWYnuFtr6Jyn4Tlf2P8jOnCGFe8Tb3eYYy3zmSeY4exssL/J6op6398DF/o6NTUsRv"
    "MmmhezUHW0RATPhIHgGqsajW4+/TpEA51lTV12hZHU8Wthu1d4nLUNZm1TwuKRBK6m1KMkSQevhRXjD5bHb0IS2p5eToSdV5"
    "1o5OnC8Rfx2tgo3w1/ZZuMURGMivSGRh4M+OruUqGgU0KnxFU2izpmKnFnUfRpdcnNl+0FpSio/+mQZ7/8vHX6er3BJ7cKaI"
    "F/knD9p+Exn9k2HHZxpO/SZaeRGM+6PjyGcc4vyvHDv8o+PFZxZw/DOK7f3Rl+4zDAj+mcTY3rp0f73A3BCUKlnbwZXLRUvK"
    "EaWJV+lVbPc13j51bfbk9clS3Vir85KjZ6oQTLvd1WuhKkt3xf1e7dxtLpbbq3V5YNrzeetP53h5lycW+X1316eScvDKnQ4/"
    "tqsVDtS7V+5EiNRVtxLuBnDbiFapAvNSL+tddThz2SlVdnHd9TBSeVLvPrStT1ZcwLXLhrzO86WM61T1BNTV8QoFQJbZClt+"
    "lcTWkCS2jYXAeDpdt0BHN2xLdsfdpfYK/Lom7lX7aa/M0YjIaMvqd5+j23tVMClpYymU/xuzVRtTVeBxezN7nm57HiOB7+Ki"
    "gjY00u2m/CMlv65iF+TaW1gAqcGp1p3uj8uwTLWm8+aSATsk1+VNzz567cln6y4c7z91FK66Y5NOs9ZCXXfziVKq+njnrheN"
    "F6UaE94w0nsNVWuV5qozpBUpWmOP5Dn6/Q+KNNCk2qEOKCeOiRzdFQAXkSh1Tu6LQNRVat5456mGfm06TrdFcipNn0xVirEW"
    "2LO7+qMcV/f7HdZsq72OFOqWPDCXTKr3COTdqOXeNYzUJ4lauy1ebfdItdfFdqei/gqLsYcN5V7RepVBtxsPGPEmu89CP+7m"
    "NujKxvy4LSLMtfXGG9mwBeW3ZoO5z/UD4nSm5NqE0T3gwdPUtP7W6/V6w/ggSzA+gOglw/iAET/y2bsgJeB/BwfxGUk7kAhR"
    "r93l1mSASoFwRoPBLdJg/jeL4NEzhU5UI4Hc0912B73yHD2+0Q+/0Q+/0Q+/0Q+/0Q+/CTVydaFG4FxtDjWya7wC5Jz4cc0R"
    "UT0O2E6BGHL03R6W082CauVVsS7HE4EAdr2KwpSCbw28mwFB3YlF9cLJFHcZYPdw0OkN76NdYS456hLrWwap+32n9YJO5+u6"
    "3IlAah5s1iHzTFfz4dHBuDCJTv/QFf+oS3ltlmkVuedsc3XD8Aa8K+GmmOYmqlQ2cBef3DFz2/1DOvi2IBANWNKVclxygzeR"
    "L7oNHrLYKfvsAx535dpRZjq01xgATQQeC3XzhGQ8CV3fodYhp9IjidPkDiONkuyqhCdVPfXmy3ldiHJtRCQ7iUUqcp6PoWr/"
    "kQQnsX3sh4AnR1Hc0WNLA75Kbd2l7mou2VxNNEN21SeWz+Toh3h5ARhMhUjpu8u5einboRg8HPWjePly9StwzS72/BC/SqIY"
    "J9k5+q86w8cp+k1kHePs5WnI69zHqZP4cRYlKXraUObwfL2KghT9JPLjJMqi7DzGwxM7lUqiHxVFYpb5JH0QbtY4gYVBP8fL"
    "PkYZCgfLf2c9P+zh249iknKBRSnzCw0e9j3/eCO+TxM/47/f28EGm2E+MPGb7O0yRN/Gy//G/T60M1j+G/j3/ns76YXQQfbh"
    "w/IiH/wUDx07CPrQ9ddf/wx9hih7E74d3PK9/tN4UFSKvN7TuJ8NBj8210lwtknCHs4H/YscJd4A/RYDQ7+OoxCHWWpe2I4T"
    "JS7cC75ykb3JIsgMcIZN7CH7vZ3ZiZl5aEXCpoQeWkGkkI1v2h5aJdh2nWSzXpmRh1aU6qUecuzENR3yb7RJcWAG8Dt1bBdT"
    "CmRuPPrKtYrOTNejlyDPo6gW+w7Y18Ue9wFBbB/X5BPep80Tj856snZ9O4iOzWOREEfxJjZ/8ai+xVlGqr73kGtnNl2WFf2A"
    "iNTmGfmNWYeHHmLt3YNf733YJaceciEyypGH3MQ+xYn5wEPY9eEwuu8hcvkCevUSfgd4E8PuMt95NDAEtQ+946FjOwhw4tvm"
    "r/z3ufncI9uHNGG+8JDvZTat8MpD/hqujw/ZDz7yJx7yw8AP8ZrdL59BAhE2ma/Jz00Gc5mav/Mvsnnvsi8af+x79hVSv66P"
    "2WeUxeZ37HdmH6fmD/wD7MAeeehdGK3M/3rM6vU3j9hTwwo+9dAaH9tksn/yaGysH+m/EPD6Z/hJ4f0WfmY4oYB85aH1Jsh8"
    "hhQ4RoW9KGkkixEhqMRwO4SPYzv0fycFnBOIH5jEXLpAEdSPC8pt2jGLCBexH6TRFD7S9DQCJI0RrD5pP4gRexw0NzG4JTlO"
    "cJrCANziM439MMSJ6cUIbucRQ/oYPiEOiLmOUeLHcYDNk5hFNrcTbJvH/ItC9EvMDmPzPf/FmjqPEQS4g35XMUrf4QBD8llM"
    "9SQS8zBGaYyxC8hq3oOPwOeVT9knMHtHMXXYDDhLf6bm/Rh8hZFpeEl+puY78i/ZD3fg57H5a4wynKz90A7M5zExAyRDeBGj"
    "zMcJphvyFXytMWCj+TBGWQSazU/gB7BADJ5n/Ds99TPnxHwN3xGJ9f47/Zn5sXk3plKV7+m/bGIe0y+6bb+L0Xs/yTZ2QGcR"
    "J+YPMT1Wz718MEBfwclGmGJyUJnWl/pMd4yphciRZVpfjleT2XRmoSg8zOwMm9aXHvkfpNy14fwTCcxnkfXlamyM9DlPuHsM"
    "ZVw8xjpPOowCH2o6Y2OuO6VUwmSZ1pcLR/f0CfRyv1SFdwaWQIQ1/nJmjz1g6FkK785zxpawKLO+1FbT1cgWKaSUu/AMPLKo"
    "MaL1pb6arFYa/aT5K7zwVhaZcZ30vVjMXPptwPfY8WZz+j0yrS/txWIFsMH3GCCbO55LxnAURQH0MNbnugfP5qF7WAMNUn+o"
    "DQtS71dnFhKflKGGpOeVRcuwDd1q2nQ2WdHPw8jL6NBsPKdJfMo1bWKMFwRcWo/PNRRiay03JbPbX7pTbOAJ64NKuCotvCZR"
    "MEzrS2xgG7s08UlI+NYvsesZ3pi1S6gXjHbqrFyNJoKBeRIVeWNnNprBer47p2AZc4d8FQNcYI+kFOMb45lGUJlU4sCl786L"
    "0fFm6oOjndXHlr47Lw3N00haaWRT2qY0MNudkzTFuPT5AvjHlR1AfWM1mTgTnsBGh3U882Y8kQ/QMMbjBYYB3uO1OZC0IBtm"
    "ucnySFdYw6J8fbA0vRjvBC9g2WiyGLLn6d5ctC9G7TpzoAE0uTrwyXyKF7aF3vtRQFqZGmPNmfMEPvAFHsPAaSIf+BiPRjZZ"
    "2e95bQ4wLcgGXm6yNHCsuRhWjvVWGzhNLwa+wDNvwpP5wD2dDpy1z4fmTJ0JDJwmVwc+XUxnixXYoR/jMLOBjoyMMeA2S2FD"
    "9wysYUek8rHPDd2dGjD256IBDjQrykZfabY8/JnrAD3gXdbGzzKKCXDxBBfpYgbGngbzwvvgg3Q1R3OL9OoczJyJNx9bKPDX"
    "0Ol4PCUyGvjkCz/BtruiSXzoo+nE0MiyP6P1OLBQiA1abqqM6S42XIf1URsupBZjnWEbz2iiGKjmGR5vVyy07qwcnSbWllmb"
    "jSZwbNJAVtaXi+mYeMikCXyRNTyBYdJEPtDZfLSgA33Ja4tTmCSwwZabLC/wxPVguKy32oBpejFkB89hyDRZDHrk6V7Rvhg2"
    "dqYwbJpcW9vFdDHxiE9DiHHN+AyMPd2T+IzVeOU4C5nP0DV9rNttfIbn2bY9LvEZY9sYGVqVz3AXhltJ5VPLT9TtfIY3cpxZ"
    "hc8YayNDt2U+Y+a5K3tV5jP02Whh2ILPWDjOGJoWfIbujLSJ9vnzGbOFO3dmJT5D10buyCnzGbaOx+5c4jN027BhfrrwGdpK"
    "t3WnymfoI2PGWxA4rNuj2cgu8xnaVNd1vcJnGMZ4MXbUfMbMtRdkPQnLMDMcF867gs/QtdFqopf4jIXjOt6s4DP0hbEY6V34"
    "DBicMarwGfrImI6MMp+h26PpeFriM7SprukVPsMwxrOJo+QzZrY9WbkSn7FwV/PihGejM4zRZLqo8hkr23HpDUDwGbpHgezK"
    "Z+i6PocFq/IZum2MAV8qfIYxGRmTGp+h2Zqnr2p8xkgbj2ZaE58xXy0cR5f4DEezneKEZwMfLQxnalf5DHfkTOjABZ9hjI0x"
    "DLwrn6HP9YmxqvMZxtjQRm6NzxiNDHdS4zM0T3P1VY3PGI9Gq1kjn7FYLOYw8ILP8Ca25o6rfMZkbOhwDajwGd585QATI/MZ"
    "hmMYhrMLn2FoumFoCj7D8HTHUPAZ45ExHRs1PkOfaCt9XOczJouRMZk28hk2XoxWmuAzVgvXmTolPsNwRg5sI5nPcDCeLuYS"
    "n2GMjJWOu/EZcIjpepXP0D1jpq8qfIaxGs1g+8t8hubomrao8BmjxXhuuGo+YzG1V/OpxGd43sqA06TEZ4w9w9OnVT7D85wF"
    "HKgSn2G4xsTQuvMZuqdPdb3OZxiuYeirGp8x1gwPhlzmM/SxhmHQFT5jMhlhw23iM1bGwp678KLx4xIv/229t5P+wUG08Q++"
    "usDDBBOBX/+bN3cOfn77zTGCIgfWv/Awi55Fpzi5Z6e4PxjAKyheL5mw2VLYYn918WMfD/LeVxdZ/o+SLfbAQtmaC1TeFDyE"
    "9aU39WbegvwyPHKBsb50PTzCJM0ZObY7gl8LoMrk12w1nwPOWV9O9IkLSAr87lgbk1/GxHBGU/il24ZmkFZ0XZ/oK+st5bNK"
    "3eMF3EJJV547dTH8sqcrDQ4A68v5bGHA7cn6cqbP3DnpYGxPCLcKnY7cMQHYcEejEe1+bCxGGu1ex8YYfmlY1/Wx9TZH4Zo/"
    "F3hJtH4QZomP0/4bDU00pGsaMjQNjTQNjTVI0tBU09BM09Bc09AC/ptob4drO+YS/zcYyU/YX11k6yH5fpO9zWE9svUQcuBz"
    "YL0dDFCyLt4h+vJjDaxc6bGWJDjsZzao6TT82A8HddXOIvVeUbXmpZKUqgc3kJKl2jVb8h/7FntjtAZVG/IB8gWKVsaXlceH"
    "132MRkYxRqwaY6YcI66MESvHmKnHiKtjxDuP0V4vy2PjhYGGWbVxiqaQMS/GK3VQH3elQcUcVKsr0hVzUmlXNT+1BhQZO80X"
    "ivZFCQuuZNUG290ikDb06c4gbjKQfLtFYcV2zFHqw1uaXAjQIV0vSzEwejoY5o5YDN2+Pu0Z494Y4nEP9SmNkVsqwKN1D4lr"
    "GWe9/NbtfxujixSv7TDzHVPSA6H6Ig/ttR+cQ9ycE5z4GZy3IeZOM0Wi5MKPRqpWKQ+Cf8e0ULZ+GduOn51D+nhCXbiyJsZq"
    "B4rESonpBqRwRbVUGM7VAAxZ/YhOPmJqh2a4LqJgKJrg7lYEOkbhq/YtQJiHRjRmuTkSmlhm66auOgiuFahireRtVSRyL4zV"
    "smX4WSKRQIhE8sWLPi9l8bYb3IjOK55Kf+xbbM7VtIdGhbaqG8Cin4qRqqAHJaVk/ZC8hl4U+s/UD2eh6jytRz6fUkQr4W5F"
    "vVs3pFZ+ZCMsGqXYfllc1RW4Wp5F4m3VGqhsTeRprBmVSFXV5iO1Ag/ryMcLVBbJkZnPhqNDRtBG4q8w6Pixb1FBlQIHytYa"
    "Er7WTDOkvCYjDLkjIEG1ipVlqXgCHws/AmQNqZa2lD+lPo2qiFdQuPJMC7xXqIrWqEBhAnUF5KKiNVpPf6iiMZKGPnt7VREE"
    "NoXpuoLUu+7+XDyzq3opDFuNq+10Hbl2UO2yRBdMy9Da+ixoYNdOc1RXsiqrb5UO+xEc9qUDflp4zJNblfV32Es7N3WrU0aC"
    "19My/SuNm1NV2Dgvw+Ccqa+NjCuhq9VmjSkhtgrX6VMScZgf6YnMB0tnN6qe1kyIDuUlqTipc7/0XZWZk23BPerZayrU9tdA"
    "Kb3IQhaVakMpJniGPPbbQpaQi1sD6s0TsplI3UKWEK4D8hXsYhvH0ngNmEw78PxbufciFpNTI5rVrllGU8/8naKh2yK75LSw"
    "SsE6nUTqba/qlWfmnGlqm+urmE7V0K62h+YFu9SiAGHiFkoy3ai7oJEpicqL0ZTF42NkZk6vDWT2q5RDEVxc9qY9JlWpu9NI"
    "wkB5WGSbRU3bTOzTqHGfki0eqba4WErIpVwDKjCKawYqqSwhgoX1U3GCweQYxPcSLZEVhixsYibViaE0UOEtijZEDqhCnb+a"
    "rU/LK5FLio8l0GtHV6ONlLzp2jakmtUR2Y7yVtKVexHjnPPJLHurmhZzTAY/ZtjazXpI10nxmq1Qa9Dl7WNXs3m1W09hPLNt"
    "lJNilCrjGJkRadt7/Lov20mkxWme51Qv8KKIdt0tljSPlCrHzTBU+F2tVwRjlpmVbhhTuaqLS7MyJHIJ+AJMdaxao34JFrS9"
    "KVhqiUmsioxkjdK6YwJ59RS3DA4V29Ylpcqy6zTKthXxAOvD6HDONlybuPspCV12kRMUmXU/TxX5TIM3p3KpPC+0yMuTQAmh"
    "mISp+u5WOYF0eW7LbHVxQ2gRP5RxXHn3Vo6bZ540X+obLxx1d2/K+0jrHiq5o6rMsOIoGokjX4GnygHG1WU9rGwNfuusm5gz"
    "fJf11cWN51S+UORCm1i5t/hQxRSJ3DK1LN8HmS9j4ut2gIL10sLnT+Of7z2ZPvn19fqnHx5oL37/3n929G704tfn2Yv7D4Pn"
    "vu6/vH/8+7Ojp+9+PnJ/fX70ZPzc+Pb3F/eepE/Cu+cr47X+0+hb/6X/9NRZB9nPP34fP/FPffdREDj+k+mT9Yv3qx90fbUO"
    "NHwIdV5PnEffQfl3P//4k//s3tP4px+/9V/++mD08uhb4/nRd2cvfr+TPll/P3buPZk+Pzo+f/HrA/350YPR88facP2d+/7k"
    "h823d++v/aO7D357ati/Tr///lU6P/jl3uzJi8MfD36KHupH3x6Mfj75bXH24v3jH+PF5MhONpM72m+r7NXxdDV+dKK9XATz"
    "/74+vfvADe4dW7e8TeiAkKS3Wffx4IKamVwwzfcvdBT4DoawncEaZSd4DZbwCQYpmbNm4pXUvIBL8PPIxYdk+cHIaBgljk0u"
    "xxbodj+zz3FifqHnORoOh3i5XFpxdnD3tXX7IogcG0iR7Tg4zkzrHjUAsRMLJRie7Ezr+yjI4Ns5iaIUmxbpxo9CSGOGGtaD"
    "8L1PytihgwPTukf+hZSSHb/1Aocnm3UPQnT1cEhFy27ESj0kbnmqZROcboLMdqMexLbq4TTDaQ88+CRRyioeYjtxTporyj2l"
    "5bIXWl4UTKWSKTfz86OwVNrPcJj2Uj4JrgRFtbg8WqkC8T+DYaccncc4Nd9Ydy1k/Rf+eg5/PYK/juCvV/DXA/jrZ/jrp7vk"
    "Mff8hb0m1dxo7YfHwICn+HgTuvaBh/3EtpCV4eT//d/i87eNnWTypx9KnylYJhRf/+//rADIoqfDkyjJaHe0K9oBbZc2R1th"
    "taWqz/3QfGPdt5B1CGOykPUt+++Q/PcWraMwO+ED+tUOsZ/AgDz8Hifs99pO/t//hR/2KvEDkuDD56+b8IT+G5B/7eMozehs"
    "ZHi9InWjTbahv8LoPU908e/0p9w9H+Wvdvj/s/e2a27bSMLo/70KdZ9eLTkN0VLbbbfZYRSnY894xk6c2E520qMxaQmSGFOk"
    "QlL9EYnnQs6/cw3nEvbGzlOFD4IkSFFq28nmeefxpEWyABQKhUKhUKhi7bOWWbOsUdYma5G1xxpj7bA2WPVIgpQ+g4Dnqe1O"
    "JvcWi3u3tzDycZJ+491+N/2J0g92n3hwnWmNZ2u2+4zCdRyXLGN65UewGjwJUxr7UeySEN0kX8X/8//d+IuI1/TKm/FIUS4s"
    "NjCEneX//L9wc8clgVcA+J//J0j9hfIdaixUgFUrAICGCsDedzyJE5shP8AalEuFTuCHc88lq1D5/A1NFl48ll+zLLPXWZbB"
    "FbzJwlkvolWY0olhrv2pkc79xDri8s0KvQU9cBz3jff+hZ+k7mbjgmbtwmU/BIzpNDGZ6DwPaNqhTv7+XHuF0qDkEOo4JOvy"
    "xUP1WuKAzGhqG6bzJeTYwwU8M7Msl9rTBfcqWIfd7mIB5/jUWiXU+HlCVgu4U0iotfBv/NCYLEzs7HLhuOBWqMj+BdbiTKLx"
    "agHXRsWPpwGFP+Z6uXAo9ix0Ugu2oglNUcS/9d/AqnBuhCDT0STqbjb4gMLf7HZnCyMkKWt7vnAu3VyngBnIVUniMgccl7jo"
    "cuMSl+u1xC241bijHO/ZNrxTaxx4SQKjZjF10HCVlYnmeBJtvxxaef9kPKZh6iwXJGXXoNkl2tdjBs+pdLS8pCMSO9J0FV4u"
    "FyOiWjzx1bHLLJjyEzpDsS+44RwRqcXZ4aUbhe7xcnHZH1lp9BauhDEfoePlwkpgqTYG5gj9YXkd4B3ljrJzuGyKnAmXTecL"
    "M76krBLF0eiY5nU4WJyORMFLStJRNO1wZqbcgyY0zXv/vvR6v42OjXxYN3xQN2xINzigGzZ8m8JgbgA/8+ieldIkNajZ7eJE"
    "Y/7V7mZjxJd05KSmgkaoQSM2TTEYCU3lHHP38LVySchY9d3CYfeRgWPwtiJo4jnvXS0Mc80m+4+h8W6B13oPqJnO4+i68xTM"
    "X4b7nSjZSdIoph0/6YRR2lnGEdxKnbjKvV5o83bhrEHY2C6/nujCLUsPtvAzWJ3kW5dHL0AjesTtY9cenP2771fTKYUb9eD4"
    "lFzN7PXNIggT252n6dK+d+/6+tq6vm9F8ezeSb/fv5dczVzCVPCTvthqnfQJ3CT8Orph199P4B8LWyc3kRAnIbEvL92ll85d"
    "sp7Y7stBZ9C/GHROrf7J6aPH8i+87gwuBifW/cf9U3h8YJ0+POkMrMf3H9/vDB5ag0f3zzr3rQen9wcnF4OH1oNHpw87961H"
    "9x8+fAzfH0C5B9bJYPDoDJ5P+o86D6zTwf2Hjy4Gp9bjk/vw+WxwenrWGZxaDx6cPYDn+/37j+F58OAhfD998PDs7GJw33r0"
    "+Ox+5751ctZ/1BkMrMdnj087J9YpoHlinV6cWmfgc4qv4P/q86B/gX8eWIMHp2fy0+ARKw5/L8RH9vKRAs7BLvDvY+v0DMnz"
    "yDq7f3rWeWydnHYGZ/Bf+HkxOLMePnxwwt8/zuEfYxWPodbHjx6c5H8fIwqPLyThHzNis6+dQf9nMYjjVRxjXs8A1tEP9NZ2"
    "lw+uP0R9NxuNlBXmPe4L/KlBD5xwFQR8irp8vevA1fto2qGO47J9nbvZ3Pv3vybHxr+sf02OzWE+u4fu0ZpmyxvXxh8u8vzN"
    "wjFQusBF+oCma0AlhG1CnDkp8Z11tkUIxabpX/qBQc2Rk4oJ9b1nUJLX5WdmRl4vwAsTJ23qMGMBONhQ+/JbxJy8TmHejMiE"
    "Tr1VkNpXkT/p9DO+22SwDKYCAlOTQ3wdRQH1QgkC2x6O1hqtdK8MNsspahYWXvQ13J5rooshxcV+7sVPUqNvluR8LqJN65fI"
    "Dw0XrKpxtEzslCQ0XS2NlKy9NI0TiFeAfY2dnwxQIVgUmeHwpG8Sn78z1pZlvV8Y7JvZ7a4PMVgF2FN68OrQzr/i5o0HIel2"
    "+VmGCEoCd3c9Xuulu8QaQDyxqkAGQ2chtrwFtOp2xSd4ckcWC4lqcOKZQjJCdd97hotSCpClVnI14/IqtjAygxBa4vEQVNre"
    "3J9MaHgI1sIVdQH1kLuA+BwOdQPbY08ZoRZKNByEm4WJHl6ctFlGLhaOGDf3tZDD3AT7BOl9MCDv3pXENxsSigPBBj/NnNcL"
    "43YhO0iZT+gLwyS/GS/BNe4dNS48g1pHOI6mSWDegW8boJTPzGtYgVgl7tHai2eo+yRWQMNZOv+y3+3Kd5f90YHjMF4dqm9t"
    "dzl2s6P1NDZMuUi6Vz2XuK588SQIgD2J+841MzeTCLxB9UvgcJBuNinu7N3N5oAOqT0JILgTnfo3dko+ULq0P6wyQs28hqdb"
    "apjW1QCS45uFczAg3y0cUZ1hrtP4VlQ2yMZeOp6Lx76qNn9AoQaVpCR0qIOSbbMxUodiTBKm+JvK+5RF5qAmfzlkxLS5hvkK"
    "kUQVIOT6L3SCMq0EtndCXobnB98sut3vFobZ7RrQB4w9kkQBteAMzHAvcSP144qOOm/mtINmlw7bi3SUxjqHR+swO+yElE6S"
    "jtcJ6TWNO1+BHklX/s09bHlyzop0gC+TDpAQKuwcLnuHHUZZyzWVEXmimIE6dGhh65vN33OIX1QIBLcSGl/5YzocrlY4Mi8X"
    "TkivOz9R78NLb5mT/Vu5lnDiOy8XQG+DSkUo3WxeLkCLQ83+J0MZWywhZg2MFNs5mWJdShmyww8LA9Cy/CQK0NfRNIWYNkHP"
    "Z9MdEX21cP5p0ZuUhhMxtyFWSRS66kR7hmgffvUedqH35jRY0jjp9PjSdyhQf7Zw+GLIFEfHcUUVbrdb+AQp4jFNuuO4Cb5x"
    "h7KnOXXF2prZmo+0VGexOaROksarMbbCdVm2eHMcZOScocDBlg0S7HM+6M/VQX8LD5vNC/bna/bnt4WhwL8QI11A8Qt3Bahd"
    "logwOuDzjF66X30l3rr8tZg4T+LYu0X//QJmPxREiMDtb+wtww5/FPGDpzVT1cEExtX158wrreOlKRjUOmnUmVBGxFVMO2EU"
    "9hC79wHt+GGSgo3R+o/nYQc3MwD+nnYEBEF4D7DuRKixJJ3FKkk7c++KdrxOhQqG2VnQdB5NLFfBlHeAz5qcoqBwJaiLSFXs"
    "RwZ6jlGFnHVmpRHTVoTs4srDGekN5IRDWcU0qjLbdLtG6BTe4BpuEizz0luKrf5rmrrDwvjY+P6JWGlQJzSG9lt/89wMU2No"
    "n20GDzf3T0xjaF8E3mJJJyarQaiLoTnk/RGTN6cJ/7A2UiGh0y8pX/dAqoJw4E/nMsyS0yexg00YqXkefpGeh8fHZnwZjhx6"
    "GY4EPeK8Gc5Ea1acrxJD+I/diq0rDA3rQ8i5GmuNiU88EpHEuRyRMSwFgXMwOIdVzJ8anmOETigXHgvMZQQWyD4yAhs0IzQP"
    "HCcU5qexczDIaJDQDnT8/MAYO0bseKyS0DStSRSCnmck1nKVzA2uM5lE6A0HDuy3ARfznK2gqL4Abr5DsylEkwlu1xzFg3G3"
    "G1p8ieVy2Igc8cowCccyAiwjk6Mpq/GnRsC3zH6W8SFIlKF+KwQJYy8/YSNIRU0dmsP+VBgvvkv4QG8TWF8kwfRBx/h4NAYm"
    "g2rSbteInVioq1Iu5xJoS/gzRNHKjXywJoU4Fpa3XAa3RkhiU1AizDv3M9BBMHPqDM7TL8oq33l6fCx6nyt56Ygz7jqz1bfn"
    "6X+eDH9a5GxEDvqmNY3ip954XujYX7mR8TIdmZlpb4vvNqyJDWdQsq0ocKit4qRHqM6kmm5rAC2RZibpqzDPX6UlVWoY/0AD"
    "KsaMq29xzYPDkdbh5PIwckrz/8g1UudXQEUK91y9MFJT0RdSOz12FW3813yhwMX7wHHZsoPKtJwubHU41y/NIO+FGEsj0EN9"
    "8KcYmXL3IOaJEEuw3rAWY6VF0Vp8Xlljv/pKqZgtiBzY6yzleyQqrINyMPLlbsgWNZtt2k2Dbwf+Lo1nX3sJvRCetS7fw62X"
    "Kd+fs7Gs7OCXQu1vBluFTLGu2ewLsElza3ASjoeb6yPmx/KcqxP2ugJ57aFziJ+8lU37iwWd+BCU4qDPnTnigo74y4KfYeRK"
    "METfNFx2iDuegy+eSxDmHRgTL6KYvobaE5NQ2PFovhgmL4CVXGAdYFqnIY21BWDDipTYii9jfKjjfCfU8d3rcbSkE4GJSehQ"
    "wYZ9RGs+7wU16wure0mhhNfXk22lhw4/02bfVqG+XnTrTPA1vnkacF4gK38ifqJtQJ63i5eSZPINcszfokTwE3lPp1FML2IK"
    "o1HaXaEAJFIpIWMSkBU7y1qmQ+vdKqHLlEyc1dCg4nW+SaYOtaLYhxO64EdUKopbZXrJxtXCaJ0wVUccKTg+MlZDI61WCjvv"
    "K01lqaYyUXizmeQVgAY1j6IPym4epFcUfq0QQn6LhaKEYnLqGD4/ysNt9dWKXrA9p6zLd/wcX4GcL0i1dKZDwytVkRf2QDEr"
    "VQjvKhV6RcJKso0dYzo0otoGIieqNADvKg1EjMi2kdTWlThJsS5RltN8mYOOnXF1eAqfy0MSOOO6IQnYkIz5rM15vITpMN/w"
    "c0jJ+06p5GZTLsr8436IopSXVaeXA5Y29n7lT5wcILH8iahLLZBb0pZwOX159c6fvEMnLOzaRJ13TBYAOcBhi/UdPZXZRH0J"
    "x9JVeJQcRaEsqvg6L8euy+C5dmWqn4spnJNJkpxaXpqC0gUA5fpfsgpdMR6CiN1u8dm6orE/9elkyHjrwHEOBpvNs4khO/d2"
    "OSlJIV1XGBT0ZYW/GsjHQBXyvQ0XegKWmmBg2EZYSzFWMqaL6IqiwBYiPR+EWplOGskNOeNryP1W4IODifaBxF7j58IKCpux"
    "otvCPEpStDwqqysKpldv+Oo0o6n4vRSMn89YgYk0iKLsIS5OXQtPcMAUIKv9hikuau215c7ToWWYJIT/Zhl5t6DxDPXqRKMW"
    "XFqWJXcuI3kCIhTjH8ZwtkT59okvc6lpvy2/yQjOm8rAYhdIqq794QTs71JgDK33XkI3GzohLvxySUVPuPynhcrP69dQEgdZ"
    "mJEJZRP2rwGE/xEcMcrgwxEYYdFvotttaFRCaVrusJZfCZBK+5nEoKhmnJfE4rD0bKESYr1HI0Bq2qkhSMiq0IulllqiqCrX"
    "FuvGhYQarayeVqL3yMOacXq1aBooOXvwAxjEeG11ZUaZmZ0LzYV9UizSgAMYpHYgdGjaIXgikArP6AmEc4+B1U69GX62xuAn"
    "wKW0F3uLRM6hX2EKMT4yKPlZGMNZOTdTF1TRcWmXI2XWUvGUYinfvgjkn1QV/XPpe4XMAwfN6GzA1xaloTmlKaryB2Dofy1O"
    "WV5ATPyJ4QbgcNrj7tPypIfEzrOVIZYpPl2Gr1ZCNy4No6JDokkBvVi/gzoRh4vXr0uaUCjCt1PTPOfUjHNqqkgx7znYDsXe"
    "NRgFtCQm6NDCOiX7WOod7BbKhKkKUeadoiMVP/DgVEL13zGMUE8JUJ1nEMgLymCDJRIIBZpwklLT3GzWGfFATxbbexI5viWC"
    "XJDE8S3GaGQM76FFQT1vCEybk1BW0buCQ973AU3cZgryiqJSRaL1XetJSvUwxHesBcfAKNeBoPq5RsZ1rCBGL6sbX6W2IVvW"
    "0UBeL64YHwRkRSZkSpaOYQR6Xlg5AecFZmnRscOKsUNQZoeFs0Q6zp0lH3Bjom1lwsi+yCl+tFbgmFNBa/IbU20jU92o6Npp"
    "GqJ5zRCVazHFfK1oidVZK3bjWllpyGFMFcrzPXNUOMJWTquT5EdBLNMBlRytfjp2CPnq5VneRCAbfaCh8dcF5KAo9wvGtts1"
    "PGsCke7pVjKQskjnagnbekeOYRixnvF8JwbGe4UXEXRc5zOuiwkVe0ZuXRMbNMGOHmNHE3kxceSYq5u5rKfhBJeMncLMYkJG"
    "cmnSyIdM8SqYeJzx0KIBp4nyBWjnJBmp21loTDhyZ1dooGinUU0qsAFhmxqpupUQ6HaNosgobF9SC677GNqSqMdoFMGqKrOn"
    "/4pST6Y3HYZlyyHY7bS7uCpSmgrP6Q6G1GbQRgCd5TAjoNkB8aW5WOcYMNTvA4dU3d05jlMHZkslUm0KNN+irdq0K69ynXDG"
    "HHGZvUrn2zD3k0s64uaTSmO4FR5al3TE6lLUWV1tfx8blFRYaKCy0EDLQoOR7brVkidqyRNtyZORvc74kLx6U8LsjlwN4tXZ"
    "vzuhs3eHSFwte18te19b9v7IxgPie9a9GTu/T81u9+AgvEyFf6Plmpf90Yhwc2iBQQxXnr64Ztk6x0yuQ0uCoAYROZ6FVoPX"
    "7M4TyO/IEahtNhEZCwi0K5DAGUvMu90xWTnx0B827KBevbkA1ym2hYKTQbvJ1FGF5mvMxPHl8psDv6bB1KCkrvjPoIdkJDTJ"
    "OiNrpiDaK+g4nBdPFRK++YZdT0jAc3DpJJvNQdLtToYB71pOAYPrdKYNtWP9K5NMTPFmnbEHsnDwuA2uX7rDydC6dOEKRG85"
    "7vkTdzQcKgbQ4idhl+aTctHtLrvdxYHjVFfVIW9xyXp4mFdzCKrXAoOZVtdiNzPtpZhzQMOPO+W2WJreGooNjVuX0EZxqdrT"
    "hNKBWs4IbOZjL8WrOI2lj94hRV+9UYtI+SJG+aP113VJ6sjRc5kbJB/0bvdXQ5765EOcsJnmKunIYIeeBxd0u10+rGpd8Iqp"
    "jKBfZUAWl/w6NrgZpKYNu2KONEnY7SoVMb89rKoKi2wFwAXO0nCUxt4Pvq95Bbw1gRg0R5WByadtnXmmuHqpAy8HRBL0H2N0"
    "X/snmjTXzEGZZjYVrdUcmuokfuvVQiPxT4rLApdkvlNu3W/fbPE18Zx4GIOjGBgDgBlMksDflM8b4efpd7sg1RN5Eja0LqOR"
    "ORx6mcKEbMOyJWmg8Fhw2eGcaw7X7JdNxXld8ZTNN8oHmjwTIHxgGnQGv8FxBjR0mNfKALHDVEEonQqE+f1iMztXr1fs2gvW"
    "gOeITny25VNDnomTk0YMz8rJi03kbzGe/xgbE3M4gb8rc7iyt65fcq3CtcueCFceH8/j1ZW8NBQFzbNwDjLjBQiDY7XIJb51"
    "NRNRQtZztPDiD1pVHS0FcG3Af79KqaGuohqBxC0LObxeZoEPy+LjLhB7SpW1nJzKwsg0PXHaRIWOU7BHo3sWdMP/g/SDhA5f"
    "9fn6/JOfzqMVjDLvCXO7yv1pd5rHIXH9CdySNULLnwyH4qR6YpIQ6RD9CfYUtbzANN21sM7g8oxK78EADstv/oi8nJ9jCBH2"
    "ZqE9dSlYvHAlp0kD15NvF0bJ9QJmwjisEqGopQrs4DaO7NmQ2nDBt6ZSSOWqWY/wNo62RPLHGIqD/p3YEA6XhP/i1hFj+VmZ"
    "Jadu2PJrapfa+l4t2lZD4lHG44FiIqP1rLKIqUqlZkYZdau+sHVqXJX/OobbgnL60QxcY8mkuvJ9grZplfpw/IqWZLaFkAQy"
    "NxvEtPwakFW8IjWue+iyw7/nqoceW+nYWSiVkfyksNpC+Wiz7AMjKSGqAIt7PyNHfi26ZceilQ84+GHJOUKMiOrlzndBV3CN"
    "cYjpsFHXk0uTJgVG6SqtwD5aJqbOxZ35lv+wMFIyAGvOeX5RzA/HwWqCMQEyEBnsUKK2l2UKIXTG/RU1pXDicCkq9yPK3BLv"
    "yg4eGeEH99WXr8tNZRkxGqyR7KSAH0aQovGUvcSBKp40VLviXubK3v/tHGrNHIcjV9BCOCVXvJH4jeIwgqkrvYvq/BTB6D9O"
    "lmWXTAuLk9xvtGTuzpD/1Sq3juiQewlmBCdzLXzZ7iz2LfkuHtrGqd6ws9YNk5SuNBT3BYs6hrjWm3M7STDFH3sD40PYHWpl"
    "ShJWoa3IS14PHRZrgWelDnjEXyDhpZVHQxftZERwxnrlCZmv3HgJDmfk0EpSL05RTQWrqu2aGVzknazG1FB3MoXJfGKS2AlB"
    "YfCd8HIg5/WPC+P5woiFCdcGu4i0ig111bItL5eOcM2h2zXYbQfHcWK+ZPcGQx+M54RivhxqEpoRNKaXleyPQyVmKMmJxLt3"
    "YKT50XeFbDvRTQrCy3iEFBQ9yjLyz+olU8xxLjMS/Een0+lYyx6+7azxEf438ZMlBHPvMEHXmwb05lx+ZUH7ejGL2teZpMZ/"
    "YQUW+2CxD/9l5iW8wJ/hDcZFYncgig2N84+/rJLUn972RJzACoAIpqe0xF+pbSjR+lRAFsHGyr+qZVggAQ04flAhIcAfhiVQ"
    "oeElBiioQF6zuABlWPZahV74YY+FFFBgF35o4UsVcl6pcl6oLfuPwmj2JlGqjGiljUlURV2LjA6wiosOqsQnp/3/1IxpX4/9"
    "2I/HgcqS5QJNDZQrk6GZlfr07CIh2zJMXqDAMhUMWATp7e0zuNatc/DGtiFC9daGAahtqwjb2CSEE9jaJAC1bRJhG5tkMb63"
    "NsrA2jbLoRsbFkmCtjYtANs2LuGbOWuhNKyVUslCL6i0sz1ZtJU+yWKLAApm2zALZjtgFszaYhbMtmB2E2zD7CbYAbOboC1m"
    "N0EFM5eIvQWLOlvVHijbUME5mSU0QLH5d8XaveyN83uH68OiGD20fzXEZaZul11ilC+4ZuQ4zoAcKovHof1srJQ6CK2jJIhS"
    "aW3OgZPFoc1IhXclF14QuPnXYKZ+DbwY9I/DfCjUrzflzyBqEABiYvrpLQCxwPuHJdFahpLR+w9VoVSGgncKCJvvZSCRR+Cw"
    "sp5UGhUfFGAxjcuw4r2bjTA28H8XbpF+zfQ0prwl9t8X4jIpOwfi0aN4SCh2HzWPHgWaJYREZo3pw0dxmDwUle57xhXFf2Lz"
    "EDmuqhavj5ZjxJZtcMpuOWhZVSPg/NwupsnPf6aYJj8XI4d8X73xffQZb3wfKTe+6Vx34/vnLTe+sRS78f3zZ7/x/fPvf+M7"
    "nedztTBP/3tRicyVW1ZhZ/5Ks6+MAuP7BfxbM3HN9v786hrbJQqZLT8UxDbGYFZKObwUv5ZVENvcBCyEQ/GJG5dAKuS/TNxL"
    "hnNH+EqogVDjuaHc3ZVdgphiPxhusoSIwm8N4U9gjW8M5pxhEu4ecWh7GOf0FRhY4fhNAJjk8hneeQHsIbIwou+Cc0T1as0b"
    "z/ggz3/JAO7GjMjgIQnnZpbOrZiGExo78TxH3J+3EkP+/E8khrDP+Sz25oUARqs5eoAEcx63KBE/orkawAie/ncEMErmrQMY"
    "jed/rgBGvD/VAEb8w6cOYMSZ6A8WwOgAAhj9OaIT4WRtGZ1oMv8TRyf6/pNEJ5rMW0Qnms4/a3QiFafPF51IdFLRVZfzz6er"
    "Lue5rrrQ6qr+vFlXXeSrAK5/n1VX9ee/u646n8uQmK9pSo6cNZxevYTo3lU96sexUZ49XzjSZU51pzA3m5aQ5K9jY4d6hy0B"
    "7SrcoAo3GJkjcSeExTa1NWcqJJYxcZWrt7m5pXLbNvdRwVtv+SXV1BqnN+qt1RjEF4Sepzdp8X05PAq8484XUjZU7rjmDbNe"
    "lY9GVYXuyFL7jZ/yI1L49FwevrYvnx/QVm7H/H3ceCVF533T2oEsvIMjXbyv547rEn/fSyl487hS9oFa9oG27AP0Noqcysnz"
    "UY1D99FO3tyJY8CRvfXeD+E8pf5yXn5SL/134YR7m3vv2ElK7r2BM1bce8F/N1HdeyfOSnHvXZGp4w2PynddSJo7uRomiWC7"
    "Ke+5LJ0j6Q5LjoRnUEiERzeCf298D048fuFey1Tca1k4R5VLLSSW5Aw2m4Og210OJ4Ca4hicsqvKZGHa0AA2MTXhbgl/wy6a"
    "LD7dbYo7O1fmNzFkBHuGd6rcpJjO+XWEML9FwX2m+IUFBUK9sKDeJNHYtj+uV6j27kBcuTsQO+EwZDcAfPT8l+McD61LfzQc"
    "xrmLv3SnZU60u3jxx3Ve/LHixR9L73Gd29vHZYoiETl9diBkVRQWZZ68rVGVe0UBJy5B6+4mAIlhdjOvyTL50/L1g8hJ5fUD"
    "uk04JU5UEk5jJ1GEEwirSBVOKydQhFNAJo5npOVBneLL4t2DiXL3YKq7ezA1h1N5B2G82cCOcTpcleQLFRfn+Jxkd+amQk/3"
    "jLT+0sDvKFj0C/ZnYqvzXH1hi4I0UNYHEfo0Cop2vu1w6YmtSYyVcf8LV1gVHzkSOdwLz1eWcb+Vu52XnR9VwmymJDLJUTU0"
    "gvq+GmJMfNXGJ0sh8P1RTnVNRKWaBjOoUxvKSa2PeC3iOQkGuMP4ei3H9xxHoBTQgvKAFuHEMDwlWoanpAoox8uAsFImxpXy"
    "t8SVyuN/eaYEHiovIeCXEWIcHO4NrITawP2qb5aRixuQixXkdgsQ5qHrdFPYps83EzEwkeqIl3tTG6YpVIUnCyNUdj24q1d2"
    "fKBKp5pAIJQHZDJNJZTUUTWOVEpivHYSt4wlJYyez1YGRAjVjKgvWlbDS6HcKAUzgYt4bWJLRaxCj8R5bKmEfN8uthTeMLhz"
    "KClmgMIBwVFLt4eTQkI5QCRa6TmEhGqMJcXtwj6REfxMqT/ksaTGTpTHkgLFgceSWsF7NZYUD930fatYUmEe8mlcKtcUOkop"
    "FpSK1UeKCkuBob6vCQwVmnDpMd0eB0o3MLQSBCrOx0RwbaxEgOJbKjJ3DGOiGb6pM4Hh+8aPKSZd143glI3gpDyCM2eOxHnn"
    "zEX4p2W1iSUj5CynIaSR2hb4CahpLKq1LXQUrlSo0PpdDa3prtGcPus2AvyUIC6SgTIyxAiGhbuk+bkNC7FE7xBfKSSV2wy0"
    "epVBjD4EsVKFuBp8KXLcatleeXjcc1oKm6TwNQsYJ5klykMwgdc6DUihLAZWirLPHJ6o3TDeMWrR3nwGGNJtgdepqveZxUdH"
    "hv8pxadVnRNwvwjRxSHR59E6hlUyc0lQVLRho0BWTjjMo3eRSW5iWuUmplhZS2Ot8SzCC8cmOaoPSjuGYLZTp2zyWpFACYAd"
    "FEKTQ0DsiZB+ZRsxxDba2t7SWdPADgm3Atoxwbtbto+Jj3/E3152PhlaxgoMWFP+oyZgbu3462VFTbiTk70C6zLXm49nW0IN"
    "UtYmg9KzWjEy/XlsvZOskTiFJ5AreJS0KjIV8B+ZFGExCBWYzpxnaAcQdjeTpEOLB1w28RZwqZSTWwPAmMluO1FyBBea7Jgc"
    "iVH1ydEimkAo7Dix/aElH8gRDyfBI2STIxrYE+uIQiD6zaYQwt/+/nNcf0uH/CZb5dbXSk0uwGgqzy0YWZVUA7rvm03pIANe"
    "FgKvF1MaGDCkR8tJYb5dyjDyefD5QhQE7d3V4kHKG2NVdzWVHT4KD6FC+NfaiR5CiLLai6g8iqM41zLSCu8VwzYeVQ4EahMg"
    "yCuqZm7YWinXWI2woS1wy6i2BW91bYXKddjKtc5SWr4t/dPeT8z7xK+Baq/cFml5oCVmzTYSAhhuJUcjaqGyx+OXeJmetT8p"
    "dhjqSaqN7ZHqdtmtr+H7d4+IcFQI51DuL/moNFiCCsFPj0x9bIzf5WhTT8ldY2NsoST03a8Li4Ef78AI3p0Zof1UzI1XRXBr"
    "OQ6Nir7UYtaqexclDjePt+EzjvHMmmAaaX0wjTTXnlI7BQ8Klli0jM9QVZir4TVSNbxG2raO5OYPMtcP+jr22JW7/aFmaLcK"
    "B93QpsWIGqEc3zxf1hRsXbJepyqVxo4ROAqIeRnmTY9FxGqhcZL40j3CrBUjtRAo95ijg+Ypk7Dh5WSzccDfAX4qaiI+Y0RT"
    "pi9ydVHeJi/jCUqFcmqaKvHZ8QCi1KvI8YYWpgojiaMJJO1QK6TXbBcCm/4omOCDGKJIoX2kT/STChMkOgGS8Z2buXRZO1bs"
    "L5cBdUfNDQaOV0j5QK+A0zab55NzzzrCvtM4ufCCAO72OWtWuZ2Qw0Izh/YY4p/lUqSSIykU7XrDilpAAtyc87rFpjgxlRrj"
    "+p7JDFMe8RR7i7dF+/B5PaXW2UuJxBjU0CrHVHilSisSa4D0lA7BcsjsAiUahIKMpPi9hGVYpQ3ZjqFw8BMiZV1JpZQK+0Ka"
    "z8FUzEEx2wpbjKPAcJeYVwdIyjMvEVFPTQamVGxEjwpZmERLQ+mthpVAzUo1ULuQFHhoXMk1lBasJOfzORy5G+D/b2xtEE0m"
    "kA2Mfyrm/Cp50fG3mnjekcz+xOuBvXBouCIPkaRQbR4nykDMtew9z+FExBdNLifZ8937HBour06HXDkFFEDgCM7nENkezkZ9"
    "+E/t2WmlVWVYReYoImrVZpBKK7av8x3HqSaWO3wqxnKPZCx3NIAEDqR0qzQEtqsat0o1xjtkmJOFpQc95G+ThcXbsVUJGh8M"
    "eWR4XoMsVAY1CYDSAtOxUQ0rTAdnVGVjkzR8zbkKCX61Gr88iNeROhgYDVyJ8zgn3xvrj1BnXKjzSMT8SIl0Sgsx5ZUwKsFR"
    "Xobn6xIyxDezeSViCJOVPWmGqAQP8cMPutAh74No/EEJ+BCxSL12x3ufRMEqpfWhO1iTNffkS0EfBn01rARGBJ5G8cLuJGMv"
    "oEbfVFHwIbBIj60pNlwyouUL4n74oedhP5U+eaG/YPanDkOt07ceJB3QBr24UMNXH+jtNPYWNBGQeS2AqfII/4uW3hhu6KpB"
    "LbTdOLFOlY5kNXfHXcTfhdvE7+aOZIM1G7DZPDPz+29X7e6/Xf2Z7r9dFe+/3c6Vli/Y7bfX7M8N+/O+cPPtfdubb8kypt7k"
    "97n0dtP+0tv1n+zS23XdpbfX8yItmu+eHYjcJMUbaPzyGSdDAfu8pYtt17Kuiyx4/Xmu472pXuJ5+hkv8TxVLvF8o73Ec7Xl"
    "Es83OVdfffZLPFe//yWe7+bOu7mQ53xJdsmaJ+gu7zIL8Z55Smq2y/gBIXjoMjgu4je+wRr6FJfF6jf8rY0aPekJTA76RWA8"
    "OI+uaDwNomvHnfuTCQ3dKojQChw3prDlu6KuaTMUuDJZgw77Wo8RajMafZjyhJG8dsyk7S9otJJpqmWm05wmpQDa3oS9lw6J"
    "sEFJ6CS6Fl3EXLUJ/Sa6DpG0LD4gaui1tTKM9q1YHd6aUP3gWfE8/IC3NdEK8lsgbui/mRtv5sY6jgL0TqIJDVOP+fAfgp9L"
    "jw3gIUx2fm8fNI3yM9ecDjHsAd45KYcpNUweqFQJBBCFT4R+9TTkQVCL75Sukj2zKhCIlOOHIsg5BM8WcQbAk8BbLmk4uZj7"
    "Ae7KWB00cGCnzAamBVFluk3tHGjkfv5RZQ/MFq/nCqmPUtmhIsHykpjPU+Ga+jQO1BqvYgjd8gZi8KQkLvSPOT7G3S7PuwcR"
    "LZhHU0xwXbS44i/dHbUZOGH85+C/w0ZhHBquyjmu8BasTGwViLhTL0ioa5KDnwIjhuBEf4e/IuHBSy+dWwvvxngJV7vJP/CC"
    "93nMic4CMDn+sbu8cYl4i8Gb+MuMpTX4K5aOwK/Jm9H/7nlWQKfp8SQao5JivY8mt1YyjqMgeBMte4jCvROS8AL/7HlWGi21"
    "8C/oNO39xApIxNJo6SRFrKBBJ2LvdKRc3J2UabyiIqM2l4QOOF+xn4Z6mN3tMge/6mDeeSTNjDzoD0Bml6RBRXBzHLvdcUC9"
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
    "79VvFu+/R2//Of/S8q9v9u69sfLDd8sPPui9/cbi3au9t28vXf5d7+qd3pV7vT/+cfG7Nx+9//fFHz9Z/v2Hyx+98uj9vz/6"
    "6MN/zr/c++vLS3+4tnL70+WPP3g0/87i/W/oq8cPruByLn/0ytKVG73X7y1du/fopR9Xbn2x/MW9R+//vff27ZVff7/44KPe"
    "G3/q3fiMZvD4wZXl+79b+sO1xbs3Fx98tHL770sfvIWzlvkzhJTljz9YvPvG8kev9G5/t/LV9ccPrizevbr07f3eG39avH9/"
    "8fvf9157tXfrO/wyzp/pvXqJPn/84Ervtasr399a+sdL9BV12HtHdAg9X769dO2/AKCP/967dGfp3R8W715dvPvm4t17Szf/"
    "/PjBpaVr870bH0GmAFu8++XjB5dxtDhlZvH+/d4b15du/qV37fOVH99ZuX7ln/Mvr9z+dvnm5d4Pr/ZuXVm69M7Spfd6b7+3"
    "9M2l3jtv9V67uvTx35fe/W7lxw8X7328cuuH5e9vYZ+YHvPoPz9YvveZkhyz9P6bK9//1z/nXyJIZWpM79tPF7+/1nvrj8vv"
    "vBYnxhBWk1kxvTsfLV37kjCSTYF5OH9t5fUve298Tg0ezn+ydOXy8rU/9m79QSXP1Er8c/7lh/PXiGAfzn/y6KO3e++8RShe"
    "HaE0Ucp7efT61aX37kjAUgkvS+//qfe7q71772aSXZbe/9Oj698tf3wrk+mycvt+7+33Fu/ezMt1ITwqiS40bVjhm58++v2P"
    "kOLy+MFlNcll8f6r5IilFqkMF5r90vvfEIUIApMUtfT7O8k8l5UffgB03LwRvQdieOna0s0/U0+PH3y8/PEHvW//Rr0uPvho"
    "6e13epevUpt/zr8MO/KLe71LH4qRf//3xbtfLn/8QV8ijdJaFn/8ZOnKSylGsXTtk947v0nnskT08HD+E9i1dz5Zuf37pcvz"
    "S9cur3z22969v9LSp1DzcP6TlVtf9L7/Xe/yVWIui3ffIl7Tu/3do+/fAVaCAxMo9FB0RcAhNA/nP+m9/eXyZ/d7b3wuedbL"
    "vXfeWrx7Y/mzqyu//h7J6c2ot5Ufft278xucrExQoY2WSU+hDUov1QSV3rd/W/zxk97ND6KVk0kqvWufq1Jp8e5N4HgP5qMl"
    "lHkqvRtXe5e+Wfn2du+HV6JElUevX125/Xs1SYUgoOdKkorK2ZeuXFbZ98qt68u33ieu17vyau+d/2RwBOSf8y8tvXt76be/"
    "6f32yspfPlz54bvHD64s/e6vS2/e7F1/d/H+q4t37y0+uLN86/1/zr8sePvlqyu3v5Uzwrn8ZX7pj58SPyeOFuWVrPz4ce/e"
    "X9NZJQKBb/xp5fvv5f4TOSUqA1n++IO8pJJItuWklKifEwktf/yBmk9CHyuv4kySpWtf9n7zfe+dqzl5JNE7CW4qhQR51Afp"
    "FJLHDz6SOST/nH+JkkiCkWcCJYmk9+3fHs1fXnrzC/o8kDHooE8MmsYHpF+7LIxeJZNk8e6XxFSzSSRL7/9p6evfE1ujTgA6"
    "NXmEZE0/UqVveu+8RVSH6xxljQiRpXSuJo307ny1fP9L4DokqB98RNsE2Ph7d5bee33x/jdEmcCoUKAtvfw5bEzsCodKJorE"
    "WPjb9aVrlyNBAvC/eqn3zm9QxtwkPhptfjUnRJDgO7eRPaXSQlZuf7vyw+96r34Km+XO24v33qK4LSkxK99+IjsTOSGLd99c"
    "eu87UJhQgPe++sPivbcezl+L0kEezn8ith6+IvKjdSfuGomqt38L6pAUWOpLwfpvf7t8/dbKrRu9W79OLdXyR688nL9GPTyc"
    "/2T5y/d7b/8FtKL7V3s3XoFh37m9/O7ny298szT/EhDsK9+oy/no+j8effJnUKFwDy3f/2z5/k3i2Ss//mHprU8RmEQiSJ9l"
    "V/JAqMXSP9589Prbgs0iO4qyP0QD3LDUQM39EKuEQDx6/WrvxtWlt37bu/S+zAgQXyvKQjbzI7kvo8SPIHq15p5bfZv1SwLZ"
    "lM4BoZ0v+wbhjHBL15WSEAL902wquR0CqKk0j2DtNA+wG9KSKy/T4/GDj3rf/o0aAde/9mXvtQ97r366dOk9YgUZfTmT7BEN"
    "RRtd8ON33ooEY066h8q65foSt87L+IARkKNF2R7xBlfyPahN78F877M3Bb+LUz1Qk5xfuvkp8ukozWPl9l+Xfv0qPstmedD+"
    "7/346qPr95c+uJ1lEKCM3XwfgYn5CBtii9+9ufzVfdFg/kGCkSbSOki5JF0ob5z/vpMzzOL3Py6/+zkUnovHwR4eP7g0yFau"
    "f758494qmtvjB5dzAKIcjuXvf7P0pwe9G5/17rwdwdO78dkqkxbQxHkZEVCJzrIjqikasW4rB2Vl1rt0J+n/JhsTQfj70u/v"
    "pPFeHozHjjoEFefSe6QCL97/NKMCpzljNgeD1igCTFV0cxXapUvvZbXTGF3331q8exlWZf7yo+vfLd69SaJh8e784t0vCQAl"
    "/4I4LYnNSASyzXvZ0ufXe394M+o2K8nJSqI9mSPP0xkWQsriYgL+rv5+8fursA5xqsXjBx/33v4SlGnct4v3P126/Z/LX329"
    "ePceNU9K8XywendeI9CB7q/+GdQC7AREJsKaglLJqcjFxdIfXundv/f4wSVs9fjB5Ue//rx36bWfhZhkvoRgK7cu9179vHft"
    "zvKD91Z++M3K7U9VcIDaQd955YnGffzgEhjzDz4Ccffah1I5Sm/PnCSIhJxDIXIVbATsFWD5+IPepfdXrn++ePdmBJFKZyCQ"
    "rl0WLfNGi9Mc8oa6+UHv2uePH3wk0PfqV0JBjdMbaHv9tLGVlIXs4G8ILVB6n1TXgjoW+a9oi4Mb5Z3fkM2SVhMzOQvC+kQG"
    "rKoqS5fnaWtkEhOIz62y8mQzEbxL1+7B1GmQ7/+8/IaAIpt9kM0ziFRS0INu3qBOACWRrwSFtaIDx9xRSJRslD3uNLkWuXkG"
    "uX68lfkr6txS7rze1Td6P7yyePdeftgOfG1PMhtoLAcjH5hgKnmjrtz6AZhTn6kTZSy9+93i3XuP5t/pffVyNKWsUOifP5Aw"
    "Q397hSTMo2vzK399SY0uAIP7/jXBTNAaVmWC6tsmRtG7dXnlL6+Shv/4wRVgEe/cVgkrBWAiWaD39m+W3/3j4r3Xll55u3fj"
    "vUfvfrhy+3a8Q+Rn5Jx99P6tR3/+YPH+VZj16/eW37qD9oswc0hDgi33xheL995auf3G0nvIVPplBkTaWj99UOhmUU4AeBkV"
    "XVOywY8Xv3sTDHf5GUlU9Dx/ICzSOCcA9W1Fo6VRxZrH6QGRMi8cpyZ9Cgq6VGKCOMoPUvrPn/RuvEcA0MrSFtBBLcfQ/uMH"
    "l9CsgNi+cKBVUD9VI/y9b/9GfryUQiN1fzAn777x6MN3YDug3474FIIpov3gcbp7L/LXPPrzK8u//5D8NeSCf/zgknT1ECC9"
    "V7/p3frDow9fJR1s5a8vgUFw6fXly6+D2/UyOMaWr9/q3foIdnGen4c8ZIk4Pu2RR9fv9e69vXj3HrhzkR2g2fuRcNYpwYDF"
    "e6/1CdgDLz17jkWhgkf/+UHv3l97v72SGOL+1eV3vyKwFJgoQi9Us2//1nv160fv3wTKUzTeR7/+XtDMvdd6b/0RlAVUx4h4"
    "yFVLPSzef/XRh++oehy2ScTkI7YEaP72b+BrvHQHRNiN9x4/uCxoWg3EQ9zm5c+XX/6u9/r9pfe/iZQjsslXfvx45foVIlMY"
    "LB1wB4q5/e3SVy+rW562ycrtT4lqoh0sZSK4EWgpSZF7OH9tTW+kVE5F1F14Na59/uiTPy998iMhdunaHyEkgzEJ8rw8Uf/E"
    "7nq3fk0Bl7TFtuY8cYvFDIOI+0mBENz+8pvArOktMhqEiYLvyIhwlybYLLiNlj9/s3fv7ccQc/pW7SPrBqbFlBF38u4ChaKz"
    "l/ZTFHYX80cBEjWIPnk0f3nx7k0x5pXLIsD++MGlI0YNNaErlxkE26XbHiPtiz98svIPcNCkAu29Vz8Hzfjbv9EMc8Psyw/u"
    "9e78JkHKqTh7rqyD2b10bfndzyPOTwuuSj9asKU/frp87c3YgyZhleDHsfUIUBDGly8tXfuvvpH1aNUyUfXetc9JNwNkoVsq"
    "N5AKG/rqn3rX7iz96VUwWK/Qhr6y/O7nIPDypkzuMRFNB+0jr9/l+x/0Ln3Tu3QHRIaA4+ZPHCw3kg6Ww0dvk4qweHd+5R/f"
    "ktJAO22disXKrS9Uck+0uTz/6No8UPaVV1WlKKmIyMUTQXTx6ZXXe1ffV2PnImqHLDwVNReePXxFYfOO4bHpzghca4FlSqGu"
    "bnlXUHHiGkgnf1pAvcLqFtZ3MRm4nXwWNOmyQ86qXcbhSipImvqlz9wwAFds9voK+otVeTDNsbyFF6TvrnjWaLMp3g6iRmrH"
    "gdseMN1pR46gIQxVbjJuNvASSLqMwuQmM+BuihI7A9dixZCiw9VnrsjbrgXMCJjBAqvFNVyZX/qUZgtXrERiFw51Mc+wIEk4"
    "2IndUfyX+UYXThgr91DsByRFA/7SF+P4JTZqmpCdidHgxPUTAsE+TsdhBrQjHw+DOovZvjQWGEAAePWED5epBJYtu2Yt1+O0"
    "VoWwzQI3e+nEKPNCh1l4yYwsSgc1iKwAcr+ZFcBXdcux/KbGMEMY/J/MCjR4UWsCtAAXXI8n7pgwO9DGTN0xAfSmibnIOPoB"
    "dxpGmHa9KT99wUS6rBBNSDg/TdbNLSpEb2nGOcgC6gya3PIoJ1MSJ9tv2XYW4X1QqVYNmpYAtbAgdSqMDteMOP407qRkGP2o"
    "C8gGmDJxdPGj340RVGA4UyyohqWC4Mq5tp+4K+KYY3dJJ8N36YsiXD+Q0w1cmD2bBjQAQUJZ8GSFIMCSH7htnxl1uIYoaFo+"
    "ZMl1RRdIIL7rwpmp6eRms3xWD224s4MbHdg6DB0URFyEfNEySahRFB3+m+QslCtK+MivAFRjgQcz4x3udeF+QJuXWAohDJRg"
    "H5mGz/wLIXA5yO8UzAT2APImFsA9dfBWNHLc6RIDqHDzNbkKEHNhkJbb4T59XWK73aAJ5MPqfJqLPjM3O/S518GW9CDC5mPw"
    "D+C8K8Plx9qIbQtpt5u8CU250aELR4KbUZj8JGAFSFmNlJ/Ef5UY+XGcPs68DYp8vInSPLiEZpOvMT9swx0NJCH8KUt24LgB"
    "pUhDyje+hDs74PABmmdAJYg5wxHp15AyiSJOJSVxP0OV12FLIhsnNpq5ncFPx9EP01/Um9yAIpAumQeNkhNGR/pPv42K7Iiv"
    "4ZAAdwJqp8bR8evUW7W4Dv3Kra1DD2KAM7cxpIHqcx1DKpIuvwqS9zEE3NSfILSnb5zuxE/ocACWkxZP8MpPy0mgPBNr3+OG"
    "tonkAXFStW3mvgYkdUO+TpK6kJqBpGv1ngbBV+WHargdXnVL7GQsyy2f4V5jeDDL4dPApfhO1nVDLx4aRbiQgWbe3QxS7NRB"
    "JU8hAZUAyVswzRyXo255WJguDryPiV/ZuxjwYmGDiWegINBhaQ+FVeISBpxbCHfa2iBzfWRsicsXAgROEihRguU66t0LkTxj"
    "dc9tiS2ZVxBHblkjAE4PyAyA+rljVIGQBSLk3QvTlmO60wD+KiuJq9M2GlwwbMNHrafaZVbgc7uevnJBLreC8cyVC9zwQw9p"
    "AmqGCCYW37Nw3Ca8Ct00DrITi8RbFYyAWtCr6HoFE8cWyk5uNRveL8R+PLUr/5U7cZXgu1tfNfqeDriLsRJXMHTFyR1Oex49"
    "gIl4vDLGTwvIyysVIoG42r0LNjfh3gXS3uAG6Wm8JdaB/QFSE7dszq0L9AgJJ9I2YXY5EXjxQy47/Df/soVaxN9Wu2sh5oL4"
    "JlO2Bv677ko10sKCOkysCufi6JWWZhIRH8Gizgxu0Aa2N4TyWWGliZg78hOhdKEilTcOnKKPhzkFZ9aorRwDb1GAki+Bi9u6"
    "n4Kn5cNC4XYApeFZJoOKwhkwmFF1O+rlCPAqbi4hGSxtzR9DDbDDSKJXty7V5njA6aZrRw2QQ5ZhYpkrEAS6Mz1FWBlErNhr"
    "aMr9kBIH3aW9EMMoFWON9CxFDXY9WgEF63jCv+m6PicjmezPoOlxzlws85K+6OBkzGxRYMIdfHBcZPNeuh0AD0PDrBSuTNCn"
    "Lu+FmaFKmr3FQB0jhDtR1botPkgDuMgg2Cl0cFFnBU5DguVejTe1vMAA7j0FeBBCQLP6TUq98MWSJOBTouv9EIDv4eiexOtP"
    "wEIyoJ4e6Zc+o3cwTWAhVbpxgJvMN+rc7q571ALaER4HdsnJjIYBiklw+pRHSRbwEBpSUvH5pS9UHVBOo+PxWIUZT+SjQMUG"
    "OQMmS54kRzNdTuNhN5XERGEdUlcHTHeUuwOEQk7/Jm8PCFTqT0Lo54GYqmmShFHQA2ogoAxFLiEVD24AhXDEJnc9NMsjRRFX"
    "x3XySpecFEYrKFzwFE10g5CfjrcjCwx9bkKfqPBENqmuWFwAJ/d+6WO/PngC11WVZFRVSnNLkgBzQ1R4rttirrBeq7Qt05ot"
    "aa3ZsHuGAebG20+mXX7VrkC7YKGJiea4/Dwq/hFNJVNYZNXJjPFAkE6/jvtNL9o3CBvzeQCkR24WfAhbHkomdelHjkRYpTyI"
    "MFjhE7RNHc7N5FH96SZUZ4p1lypcfmuYNC15wsxW7ACSesmj+qjKpLlaAsZE0H3MbXFS2eLHUPU4YNOei1Z+7LYosYN14MZk"
    "ETSNdpvDUU9yBQstSSAX/I1A6eJkPjDx1Y7lx3P2JctM6IKZA/mxlkmq5U7GDQ/ut5bC3cCz5bGtmB93r9BWNDMxd6Fvq2fy"
    "00Z2SY3Fw4pyU/GaTFtBMx2aB4o0ZNfxOfyEHi8j9HBmN+h/+D5w220wlwOF2ZDKM4AeKWwaH7o/BG59xctjJtxA4K+HGhxc"
    "Y1XPakUuoxI7GPjiOD0ob9JdNOWAE4p8kr5tAcuosBoyMSvoW1XjOO4gH+5Pgb1o+Wy35AsVijv0K5ih7OBE/EIclidjF8f3"
    "WZODHxaJnrYsHJb3EsfjUUVD0qnXK8JZiRYpAgEKTVIpE1H3xKeIX2LmsVKXPhG/1yKaBjdkAbyTptCOIzLJK4Eh6T62+aXy"
    "4KPzHiwrsNW7bojn3qe4mT36Pgr/qFwgcHGnkNPBbXdxL0tfO+xkw4Nohs/Zar5NpbgFhiiAVzRB70I48aw7wFezXR/I59Qa"
    "3QFUwOGQO6gx9tgyM/ypvGkARjOH2tcYTNihkYYFfLyp1qI46qrNbQP4B5gWGOCNv/Ujd3Dcg1JswgiaxITMxGn1eEr4ZWQa"
    "GLbdld6WoIm6hPAKVuC+S99PnlF3c86oCzmTLSlBoOeGzaUmuUpBibTAApaKW0atJZEnrdgY50yFjIRC+ig6/hKn0FerHQE7"
    "KJpH4gA6OpQg4JCNgAKihOFZiHxQTBwHh46L6JJLT1I9b+4Z0/ldWw7qhUbwU/pfrSYEbXCQHIqvj61H8jPfZfn9QUSkNmVA"
    "uAWFuPD0ZRWFvoUfWAOeJY6UI6+PQhnqWXLxK3WQfN9PjXtzp2ZYFw2Gd79y2I/kuYXLx+H2u+xVAHzhs2z1hn1+21j4DC+F"
    "cGX5BmiYDoHvd+0G3ZBi4W3B0LDfINFF2qxmwfWIBt5QYzC4yQLuIeAsVadh4SYVaoinIK+CCltqoYYOn4He6ggLgCHunjKY"
    "6YoaDW5UomHh+gDUaHCV644NuExMCYvvEzhUb/sXEJTY0YW/QBzSouvSMuUZRj0vhJvmmMnbruXHLWsLX9OVlDm9aoy3mOea"
    "CzdnLFdjRrDwBXNgnBqWu2sZYKcCHgv4KhMjP4V32/FauPAZfCVuyUPzg5awxEbx1hBYnLrV0hiIQnRCclGVAWpZeXShBnoi"
    "Wm03UZRh4bNsVQYcTZMzi8oyuE9SleEoTMtApMFKWA5cj5sTNg9beMFj34VJUTfVXzgutoDXdzX6oloJorsdCSFOOFt9ARYP"
    "LnrOLb7AW6SP5NZeWKXyghirb+WFfqUXFr5m8ds1Si9IJLgRs0iVXzgmX1jECZBUBC7NTPUF6g6JC++m1/AWMLE+1D3UhrIu"
    "ctYxZizRXTR01DRB5cnqC9BIKJtE6wkOk4mzqxfcsAAv7qG7aKCuwZ+j0gsZnCFHW/iiBRwJmZRPt09egKtS3Ihj0c7xBWvE"
    "m0rOw87zAwBN+UIUYMhA78PdbqDgRp1ARACYhKXWW2jJegvU5KfXW8AVgkvlOrnVFui14/Yvt1CDq8Cabhyhx4vsmON26AKh"
    "RKA+W3NhNMKVT4wKpYyJs1e3b0JSUNQeg/ZUV4FTyJ72Lcgco8UcvMEtwMsBlavQbKvVhqtuCor5VmQLXyDmkVGTgWd1DLwY"
    "qG5hDRuiQ4GI9RVayK204HGqteDGPCAO5iNLo5Fygvkx70o3USL6ShdGus5CugMjU2eBt1avs4ACNtEqL7BfM35KYB+/SgX2"
    "ozoLPyuimCyCRbefSuxnCzEACdVdi7Vd31+42YF7+cDykV/kVWMwQP0Q75ObhUHlI6wUJ/dDshBDghTUAP9xz0UpfUzRSoBI"
    "DRuuC2zBgIZ3IcQaDC6UYIB1CWMgYBYLXwghbrrZUD93mrHOFF9RpeAGwv0oHKWmB7UXWpF0he1jee46qzAcdOBeVtTTuKpJ"
    "RtelhyhpExkAo6oKY1QN6yLOKVWDAXk+lWAQLdMFGFRxDApgnwwAsdfhHjYbbz0MLCeEXUJVGByDnQfzHG+KioowrLHYbDSq"
    "vxB12OZeVILBd2csp5lTfSFLGIk0ANOiBrFeHcmLOBnA9Um3gabq6zgrgPuiEgTc9/WP3LILZtyHVOvysgNqxirZAYmt/a/c"
    "y09Un+EJswNI9En9kGo0SPUoekhlGvIG/XnpAk9QpgF0HeBXDc8gOgU1v8o9vPQ6N1nA7hBJSPEulG0jP10AJIqkhlgpzC/Q"
    "kOWdqxZqyDbvU7DhuLfwNeqoqJRlCzZAkHfhi5pj1dz81IJjaa7TEVUAWtnCDREn0vdaDbiPNWylCzdkeHoi0+AYgSmLNki1"
    "NH9IyDiIRwQPHMgS/D4u2xAm6jasrvJq/eGj7INjeM2r00S22fCoEF0WOAM0LYkVBbboYy4+RjAHta39x1UzEo7lGAi540ME"
    "yepfliFem5z+4rIMITPXNC9Ww1icmjCat5jJkgxZqwGXNmfJxPXgrt0kmWpiPYaFLyEYs/AZsOJUnsKxSNybblLKB2AMqFUY"
    "xgy4VDpuZeK81qGYpPMWjkVd9K+9wHfitBe+UDaxIY3zVl71Be4FcJW8AJmoWim/kIBSQ27F84FVkhhWxw7I9GijtMAmaPCW"
    "Wmzh52AsmeMwSjUWqMSC2xcq9D4B/WCFBRNrnIjqCg4539YJU8ECRkHVFTi2F0MW88HNyYEYTavsoCDHTiNHqauADX1lNlEO"
    "AGqJsXIYl1VYBYo4MaI/CKKoQowGWM81siJqhi7+TWZFRDCirkRTUQoqrAKpkh/RF1QALFlMATaD5SeHd0OIX1IX4BTJKNh9"
    "KiqMknth4QuleEDMvDOpEqEfXQPellqgp7gUaCuiuutSwYQg9mGY8rZ9dJgm6K1fvYSsUp9bL4G2IDqjsTkoK0BaMGySpye6"
    "W71eQs6q5eZUjK7tMYb94xlYuiDXd6wiZtVKCWtOFEolGKuPJgye3BIJDso4p24BxyB5kYbODcVM6cZwIAfXEz/7CLxVMy+I"
    "oysO1LbHa5ZvtKQ9liyXgFwOckeMFjK6Grk+TNdXMzGQwVE0Bt09OSUToMlaTHjVggkmD5m4PBwyMdIG2lhs6nlYximlT8p1"
    "MM6H4JdOFEpYtUxCVrlFzdvL6t3p1Ayp0ZMWvxP2bWR5xJUSJNSQbBAYtQC3db8cjXUkZ+BGELwiUTNBeZrN2jgunAHkExaS"
    "LJW0kUrXoMljJodiU+Hf60/fOG54bgjMidIIsjUTqNs4f2PhpUSJBHNNl6AMYdlWx0ukdSzcNNExIxVhyu4gJajmOk1es5Ds"
    "yCfusroRWMTJKrRpY66ayfAYxeyOhS+T1RJ84c2JSyVUIvCIc6xdIEHxJx8TTkwIrSBEHN5TdQSO2R6t5NiJ3A+pSfm21VAK"
    "I2A8D3VdrGiSUnZFAoiqQ9etmhvS8mHEJ6U6Z2ojYEjGDavIUiAXxEI5DjKuLrgzMvj8hBARTYoqI8QaRVQToQM3i6Nl3HFr"
    "C19iZogbZhNDjnPTgghJTskAH8MbRHvIWjBGJyoieJQcsoZ7Pc4P2eNKasNNBbqyBekhdV5rilIIa/YWVUHweAPjnKlCCFkm"
    "JQoirDm93OSRtaABlaeKSofUjHOLIZBvknwBtiUKICTFQOSwrFCP0YDIZPsVQthDwQpRC8FN1kLI4kKtiWAoJRGOyaiHWhSB"
    "OKTIOEGlLpF14rj/+soIprt68klGZos4FBRHiGojqKkoq4ltdhqKPykVEkgwrlUdAXrsm5dC4Zr+5RGwml+/5AEHdHlSgeMi"
    "BYI7GaIkjlsL24YpaxXkoETNVuE+d5rGKgNy3/9XjKjkr2QzTuLqCK1IOYoTWPorTDReTgqL4UFss23UXBGzQjOS5+tXqboI"
    "lloWQcSf4xyW09zDGs+e7DWVySLfR1URQPvZ2xmBw4lHjHZhfFxvQ0ZoOxjYfUKf0MZ1OMoG/4HfkC8D/4HfM01d02eaA3uO"
    "6hMTRe1YZ4SPPLO3U2rwoFDgL76o68VS4B4Gwb3H8HmhWPLbthUUNo0PTE5sKo4PThR37dK5o2tTnZFZCHJyZ+DUmK5twLE3"
    "VAQMGvcrCARMHwDYgINuqIjB5+J7KEYxH8eqF+COCvjmxRfxl8n14qwNCdQjhaDb5m6dOUbHahiB6/0/eqjviv4q2YbTCCGt"
    "aNeu8YnK+ESxVLccs4B6W2o24Dn1z1hBs8A36gN6sbjTqheCp58+6AR26Sh6OfejGVcSB5m5editGTb3j9ULQbFkc6cRNIuU"
    "PsSCOfFjqjPOJ3BZzkfLEk/xCE6R5nKekM2LO4MXXywE2DgzeAGRos22jItWK2zt9wzsCM06vzI0V9TOd0o+dKMFxeJO6NkZ"
    "CXYKYDjU9yJjtMCLcebTUQWMaNjjdugZ9onQ5n4BZ1EU3RS45mheEXBIWYAFXoR1cR2u73IqHs72OFHByU5EAPs6sPZjHVj5"
    "i51o3fd05rT9Oag5qMC0P0KNRC9i6DhAVUAcajiForZfmb0WICSHOyMHOwUgzWLc+4lOgTrnIw3brRr2yabll1yvZsCyP8Vf"
    "fFFQFi9BUOsI8KUGf2pElz3oSgvXyXlfDJqeO832eZ7rFfC4AR1Q9yy4iqQAQ6lds00MH0V9FeVBGaNjWDYeC7FE2iiceC3p"
    "ETI4TnOmMzKum0ZgDHj8Auxr+EnJL/qEtrszMluzDd+v6E1umPqcdjp+4obWQIsHhj6nHch9eqozMjvFu5VBLfVOJNTrc9oZ"
    "GL7p8bo+oT3bGTlemJ2cBJmXyP484U7rWttz2z6E4S5UZufmNJ8HYVss9axfCeZGOq1CUXNGpIhqBkHbr2zahH+WXK+xSSQO"
    "booFi2xDf5cgxBc0uenW/JLlbuLOJh8P1sgv0eG8SZ+TBO1pVnHkmcLhQlE7UdBtS9ckEhCXGyRaN1TA+LxQskzxjPArH9Nf"
    "c9r47oJuWh1d293B3z5UXG/oGoTFtKkCNQbkFLVyUVO/xW00pe9CUGYKRwrVTlETqIcWUM97CtRXiAseAbYjNYPCrBD/Ahbx"
    "165duj5XnKOhh7VxHT/VJ4rFSnpc4ZvIG7wsB+dAzenxhbKRO0i2ryHZV5vyZtK9SeGW191EUdtd0Nu6drqjTRXkF2ea3XG5"
    "LhO5KE1O7UThoMApLVVb1w6o/e23Liq9EIXLL6HxqU60iPQSWhcrBwq6rj01WJzQtm1R/tScGLioF0OXiE1vNg32UEX5RgvA"
    "wRZU9MmqbThToADYFd1x8UgJZAF5vM49DwOG8RxQvyhqO7QznQRow9pMp1icmytqh+gSqOLIM7MkJXhpcrJTqx1rB/6LL/Kd"
    "ddcr2DwY55o3Aefcis44nxjxJNNx5rQLnZFDncKzHW18XJ+c9Gtumx80JefpDGzjxuYt1W3bUJ94rjMya0H+C7krB2hJtbrr"
    "Rc/0Oe1czH1IrkEk/Gz8EE4N63PaC6Izw+zoc9rz8Xs/bOlz2i9yeRjvJroZqLpmV5/TgvixF92ks8HwLGPAFAWVzWp3Q/x2"
    "ADIu9TnN6RIQ0XOa0pzmdUfG9brr6ROalW6D32pZ4IxuAuS6xW3g0W53ZBZRFF0a4Ise6UHf/mrdNM5s8aGAPuymkGbmo6cu"
    "PvPDKuYq5Q3Wjj8NuNcCMFvd5HL3g7Mp2jXcvk0aogmEx/s2muxKAYXo5hcGmhbcvZRt2YHVgb2jT2hduTz8gj6nVdMoudjN"
    "o6OxfETtUeiIXwAcTAuYhhAmSBzuC9TJrpTdlj6h7ZMzdt3c1nu7sO8UCSsS0UFD1DPyVAMduOIIrutpDZeySixMOKDfhsiS"
    "PW3x6YorEqR86a2u1LQqWb2YGWxrXuhUQs3jeJqpYsKQ4mKBSh3+kIeQKm34aw+FpltCsDdHzhQKxZFnRjsFp4S3CxSLWkM8"
    "zNd5m6KdUF1Punj7UKFc2ir0efQKgzpG5gGE1W29uIs+27ULHJux6jcJyOGlKd6F1iX96acb1PAp8SfEuiDU5wR7ed0I7aAA"
    "sgDuuDruuW2jgXPDZ8Sad5VMy28bQa25Dz4qwCQO8W7VNTyTnuhTvNumq4eQJMSAWjWsQjnwylODIgsFlJTKU4NzxaKinSuK"
    "cIGXarbVxq73GoGxC/Ri+FHQA34x0MEG00EAoFuvsKnV2mRpaLp5FmlVm/7Pr8z/+1elX5kbf/Hi//mVufFXpV2/2FSCNN1C"
    "UNxFaC8ExYr8Jcy8hlyCltEuCCe8GONXezc1YIxi6bxrOQVAdnHnpvGSNhH1+/TTTwUlcWDPL4B19fTTNEDJ8uGQVsALTnH9"
    "mK8XnGJxDlDSFYQzPovwVXTwJEmVIiCgSyKfe06TjdBXmtMK87njZuCRzWkFqdFzE0WtKsYuzLZrB512SInzsx44RGdzRIgH"
    "XVgmTmJXxBpRpWIJRpnim3Nzc8WidhEMcpDGJWAlTz+tchSbgHv66Zj1TZTqFmQmFna7rs0NR64O04vamACcunrxxZqYG5mw"
    "zwwWtT2ihS87fora7nJlUwj/gEGt2ISKCl1HViQZIv0FESGcPq/ouuY6Y2EVQg3O+NDEiy+O+D4OODunjeuCBvSJoqJLZ+Wj"
    "5jqHxL7aY7SD0OOVSc11joPbSD7oUA9C3XguUvPUkzeo7J0GHbVR1GY3WLCUA5a5IdZMtA2YSzGAc99QOVLwoh60luWgcs7b"
    "ldJWbUPLcgbqwiofMNEs31AZ1Da0jIvZ55s1G10IFcHhND+s18FrxVotXdvgN93pgWoYBK7jbwCkmZaPlyURBLAgmuUgVhMw"
    "HaRnWjuoVEXHdJxJnTqkZGp1KEwNPW9wnVNtXBuc6Wk50bpqOyhI0DWdQAd9T0Cla7qABj1OuqYLlUHP6VyfKGoIs7InImPh"
    "eCOyelQtQlgfPpyGtYJuZI7ASU7inOU5bdYk1lE5SozhpFFIrvk+z4MVnyhqk5XyXEI73jyEBgbS27mONg5EsbtZ1ISOEpsq"
    "VjFeizGB4pwdf7EQaTZFdQno0JPmOnvgaGzFGR+ELcBHnjlSCIuF6Aah4pw2a9VcR04FwKlNFsXDI4WxyaIGZ54r5S3KKsFL"
    "fULOTzGgEiuVBjZarQnaC9G08f6iaOJGUUG/z2uuYxpeV18vNmIlLoEPeZRLwUg5gREEYi10nP+fQQcZnSIbUtfOCps+bLVw"
    "5i+Iv9uGo2vPC6JRwTq5GlhaTJ6G2UFOFHX2i4hbiRNgRLYTMY3yLo6N7NCHu08DemDzBoejqE43MgnVw044TOFwAaxTMIYR"
    "qDNeoSvWjseMnFgvbEUuXgo2XGu6Vg0MC5zvL5BzZplkdMerylUjawiDFtgp6bGxAUYCWL7NYXv9+FW7P7+SEOiaLh8pBNCH"
    "RcWygwyxLOxz6AywCa87NK9bnCgWi1p5aLgovRVWZhmA2AUTkmtpdFVJ5UafKAfJ+gsqeYNuzhKIW1hBTJW1lnGxcqQAp7z6"
    "SKRBbUPo84GG54Zwh+mGylPlPPS3g8o6lZ6EpToH/j7YCjomXulry59W//VsGRfXu4RtXfNTCI1XILPDa93kDre7yg4Pu9kd"
    "fmSdO7ypjBmtu9mVTqh6BKIfViUrMG1da1MTM4hciPk6DLQxda3VTbWQc9XSnVhF9btm9B3Jp9xPjMQnjegTycIVnB4p2EXV"
    "bzZJje1i0l9GaoCaESUU0ejbaGWSPjP0bNCJuiOFWjFSX+eSy9dVl6/a/ekM2uMXkgz6Ik0oGnpXyiMKPLswi5FfghBVcGwE"
    "CZ/FuWIlO3ffbXGysGL/ZejgbSZ6cVfaSxq5TWV4O0NfYzT90Na1PV1tPIftZ2CIBMBM4UJHcn/LxLA6V1AEfpOY2aUVB+Ef"
    "iHUHHKUmAO8/E3yQr2UkeEeSLcU9R0rEkYKZ9XynOAa2RCEPvtFON0OXaPUoRDwdUbw8Cp6i5hwE7xFyTPhbld4Ebqd4N+sI"
    "pgiDtaESkIgBAx8kzMluWr7si3eh6yI8qMqimzffN2uUq2VjS5V8s8e6I/i8YrWMBt80zavtnVXD59u2aKem7GdPOO5zo6O7"
    "z5449dzzp+3RLWdG3dHR0dHn4T/Hp/fh/54/dfH5sWNT8OhEY387GNp0uLWHh1vq+/19R1t7Tu+pjs7UDtYaz5Y7B/YdeMHe"
    "9GzZfuHs1rMvtLafOz58yjlonRsd3T2ze4yHMyfMatU+N7P7+eMbt2wyp6ztB/3y1mPt8+XzO6rtA4OG2Tg43T7xQufAFtOc"
    "PjB2OKyaJ048550yapu3W9bm8OwFMxj2nj9zJpwOBof43uMXNg0eM44et09utTcawT6nvHmbc/SQc/j4pufqmw7Xq5tHj24M"
    "Zp7fuqnT2XO2fGjHHudwfdOh+qY99U3ds5u6zqZpZ9OwvWm4vWn4uU3wTfxu8/TGU93zR7dvDesHh7c+d/Ts2ePO4IFm68LZ"
    "Y9XRqe5zxw5u2nu4sad14rnQOTdmbps5PXbySGvTmalRf+/BwTPVw+1NZ8ZGR2u7G7vPnNrbaBiNxujo3t17Rp8b3H1oanT3"
    "86P7Np47uefMie7z5w9e2Hc+4O2DjdHp514Y2m6ePTO6bcu2tnume/6Fbeboga2N48dGdxxw3NF6+eyFc5u2nd3WOGB3+ODW"
    "qRc2u8dmtp8PTta2XwiOPXe6enTTlkMHNnarey9e3Gcd2n149GDHKe/bttfunjh6er/9bHfH0eed0fYF1/eq+2Zc+/l288yz"
    "Tf/UnhOba4a170Dt9Pnjz56d3nLowqEzfNOOfaN1e+pQs3zquL9x/z777NDzuw9vPTR8Zu8569jWqdHtey7Wng+m3IPt04e7"
    "5TPnDp8+f+y5zgkvHDzqDPs72qOj2y84je62mWZr//CJExdnZlz71An/QHXPaK19JgiGnmseOrdj3+i57c96nU3Ws4e3Hba3"
    "Hdx9zhkcOjbznGccPnthbH997+CxLReON05u23u47h0bHR1u8bFNx/mW8vDxcPMJ27dO7AlOdFuu3WwPP//CC2esg/bB0wcO"
    "nH7+oNk92H7+2bNTh1440t5/wj16ITjfDdxq/YAztWe49UJ527HDwam9U809p0Jz+6kzp8+1tp6rTW0Npnb8v7S955ab6LYA"
    "+P8+hd33Lq/SkVwkoVB969QQJUSQEAIh9fV0kQSIKKKQXc8xDzQvNguEQlXbfc6dNeNuW/Alvrj3/naEIA/j9M1kIg38WSjg"
    "JdtVyyW9x2MhXKqcOGIrebfqYyM8j/qotSrpYiaP9hQil0WyRnZa3Me4iZSLAz1d5gZd5iavzQwLS6g+rbqYEwDYSsCyoMwm"
    "oF8N+PHCKylplodJqe8VPD0yQX8QzQRurDOTIqQkWEBOa0KcZ4vCFnSMHmt8RkQnPKI2IblB11M77DJrYSaupvs8GYlHbrDj"
    "VcodCn4iTwvdBVwsSo/RWDXUwdiNqj1Koe7osM+5sXnqn3bzsp9PfJrtr4CJqYUnGZLXSuZuYahQWTJZif05SE9obKbzsXTE"
    "MAwo2B00nymSKANu/5QNhWmuajPlUKTHiIvVwWyJblh55GuIuQJZwDXQ4R4/UBboH8apPc6rCA5GLC9ye90JFytgvSUwnXBl"
    "NqK2OxifjzCoIBbIvL9zZ7MkxtyUCsRCsWJRX4zlqliko+4g3667+BYcz0FvG+lBLI+mUyENF8VGC+Apm8R5N/VxwDB5am34"
    "OWUKY4i1YXUP9NciuDzIZZmIG4pWTE3YDhAFG+6nA2DRHc/QyTjYweaJJNmJiOSyu14ZlORPQOkw4kna2A35nV2cYJJkMA5h"
    "A0CmQyNnTB6ZY1NMASajrqLsGMMYmOsuNV6AU8fSFuxsODwqtg0yk9GUmfgUpLj5eAt1dyJrZUE4kbrZXvKXbMKdSmsox0VE"
    "QQPfW+X6ieyb80kBdL2K72eVs1hT5UrlySDC3UV83G53MeTuEHSw3qK4TFIyFVS80QeyER8axM4fEc5wVcDRwBuN+VzQTham"
    "EIDC7HNdiAes465PwhyrUrWU7Nhc+aBBw+IoscGYppee1Ve1Fe2b/e1CBk1YpyIG8CEr2O5xfCooaiyYxknayth6eyogohDj"
    "sQfPD4E8JbE1113Q4lL3d5Nw2LdKbr4zEMiOjwI3jU004Rlg1y/kLJd2o1AbUI4aIHaW0NY+26+EzGbtYIibnrqVdb7odwNV"
    "DfrY7AhqU5WOJ9sDeWCovY9OswF2WhZJX9CPjA6R09MSGa0GkwXJ+0c2iUflyVeV1Z63+B2HCvuQROW0ixBVKtqqSfjIUpIG"
    "nIyOJ56BAQyawjsBmY2yEB0DfElt5so622LhkQ0CFuvmVjUGt2DRXy3WPrPozlEEJNAirNQ4K0beaIgtU0hkDXwwUMnxTkEI"
    "Ie/uJ4PjfhlBjuBRpxJZbdYoW5RAFpxEE9YJYTB2RoE5zhVG3ttUOt7S2jjldfVQQemAK+SFrDvOdrwv+zZfpGKw8A4YtjAO"
    "q2Q45pbb9Sz01/OpM6cYPE6JKaRkEy4ZqWNRj4F1Waz8WAjRvsBxqKkNM2hLeshUAg8mzR9HBSaXYyGw9BU26moMK8IuAg8G"
    "2Bbzg+N6nnrBaSoPeXHvg7uxZoulAlsV6KkxMF4td8BmPz8AR0lbzKHINwmrTINwl6+mfiwu6eMWGBjdZZF6Q365ToCCNYcD"
    "yo/7p5LdefQAjoaVSAPSYTnO0gqbT7F+jqVTqgCtCaJRsyoznIDK0513GHMhVMxypWT6soB4CJgcQNDZJmthPM/4CUFzCDQN"
    "ZrG5gHIQQdKpVVobqpwaU4Kd+wt7UHS5ZVUu8kE2Shlwxadkf3xa79T5Cg3HBRB0t9x+NV9t0YHtRyNTUa39rPDRcDda8LuR"
    "iGwUGNZOXDRVRtNTmdmjjbociJuQiciIn3ibPbZZRtBhFgzG3dFy2DcgtT8QzaEODgzaIsjVfhxbQ3jnOyFgdtfDgprtE1de"
    "IenkGAVTiMvtvomOxA0EC5NJYc9AfuBPPd30Y2ufTZlZQDLFEJJyXdHtASKoghfvtdEolXmHOQgOhZGerFNzfHyYzwhQV00c"
    "OkAjfZGEIwascNOd98GjmLNxOkBocLkv97G/PA0dVEw2SZFu0sJB8pnjL/L+bDjEioM0sublOAiCMgWnri9n65iqjlEAjNZL"
    "ksEZb6z4Xqr22WymB+GAkeZ8tZlT1kiYsSo9S1iSNqd7Nz12JWtexKRdppo8xViTVpMlaYPB0cVsBPC12WoiWw6SroXTZOO5"
    "C207BSw+E7cqhVXgiNaSsAJn09xWlXxDLWgthDPhtC9Rr5zpBxxE5u5R6OYKfwKhqT52l8f9it6x9iCj5yA2GVTreEqO9qAe"
    "T05Butr3V/wE3+Kirs5ortvvm/ks2AwSnMTmkiHB1fakF6OJNxgwkO/x28znZ6fZYCbR1UycwuXyNDMFERMN4RSzDOBOBltJ"
    "3JlLdDtcDBl2JvanvDGt0AHvZF51KoccNtWY2C8RYSqiaHezkMw+MjrZvtpP0QFQ8sx8toKHNnMcCCjMytjRVZeLibQVNzJt"
    "SXT/MNOi6fywI01X8ikC21Xq3naOKziSc6mKxjN5KriKEW4zLvaXu9lOoRNnfphMjS5kDR0lPphoNMelCTBZV+BUXpXoqU/y"
    "Lr03l6FiO3CopCDfnUscU/luNmMRaLj0hMI7LDZrYrjRAHGAKAK/VcfDk5n2bVmuRpTeFeJii6gn/miletZfGHRceHjs9CtH"
    "yw/dLhQPMCLkfeXgl/l8MNuJOLDHeHasT7Q9zVQYEpYcFO4QeTqa7qM1kPrA4uSpQH3zwJ6fX3teda97xpi+tahdSvxELn4W"
    "iIdnsfRNjFUrNn7mH8JOK2c7P59dhChu6uq+dRNUJDf1rPb+5pq1xOOOt32u2Wt5DdifnV7WqOCHVm2oWCuMfJRY1LfoTcOf"
    "qO9jT/zDvHrHLb956urVd8KPiTVD7p6PUTdyJ4tp5MTv+eGXrJblcbvM1hdIrNZNaPh6rRpWb/9R/2AW6bfYOr+a515sJYYV"
    "Zk9Jbx/pjaaBe3Yb0XAQtYt/pajVE0if/yt7qEdx1XNch2cR4VltsxVNvjweciupzm6ya03Lx/oGvY/0r61qYud368sXq9YK"
    "xbIscfU8sx4uI3ZbaWarxvX7Wcj+q5b/SCLfer7EQ9O15Futg3kRsH358qC1FbMPX2tmr8lqJr/Xlus81cZjtSuqX5ft1EKP"
    "nZ+nztNrrS36+tap2Qhap5aMhB/YE60r+dd72dk+0hsXQn/WSa9Jo1JSa5illw3bFPjttsC/nffTxddSy+C3641uREGsNb41"
    "eo6VRPVvswFqRmKjz9i7TE7Dk+n9dnn9acutz5CmcmNmabUuJ85J5+e33vczG+z+cDS6hK0WYT0B93K5x8bLy0+UB791eq1O"
    "2X1TV26V0cbaurQUdh5v7kPOzKr37r7ukn7OrGqZ4S236bePbbb5df//mnfli194WdqdiDDqPPwrURjzvxWFvevr5b2Z2/fl"
    "61WupbvX4/76fo0bL9F5rZPQcm1fz5w1rPoAVn7OlxoZQxjUYKvhS/E15LnAprsO9YS/gKTFe1WttFaGpKvnViDhuKZphb89"
    "vWZJXousmFvp2g7FOmZfm3G99bh32KPBHFNLMxv3n7+Ca0Z2bCBYcnPK4967vNJ6rW9koW40ujgse0pbZ4fGRwzkd67sV/7h"
    "WK9qe8qyzmMzHW+9SfLwPdAS77L+/lleWnOmC/u1971wrRKPjk+v4CfwE9z/BPcv/mXaWE5E5Ef1JDWihyb82/ffGuFskvvW"
    "b0+vtQJGVHP4zadXHn1EP8GPsAMhGvKIfKr/1u1Cn9q34mPG1/bN+fqznPpv8fVnjdU5pwD8BD2iGvQ4+lT/rTPBr+1b/aV3"
    "GZ/aN+dnGXWt5kN/aav+e6phKf5Q86mz197349Nr/3H42qva39I1M+fpdfyI1NDubFw5fkT/UgnqP47+Uqv/iP59rcunIPTn"
    "dfqPyF+/BH6s8q5z5yrfel8bseJ7qHkVWbTKo+EZEb0/BJ/qo5MleWg0YS8uFEfaSh3eOdo7w/d3ECyttXUaQNY6/fpbLfZz"
    "kXfK3+9cfP5Um7zpzHsdbqhFUPc4wYjCsEaFH5TL33kC/Un7vfez1I73nQZmmtTuLmvHwHdkwM/Q5/1kXT8lVBfl8FqO9vI3"
    "uvGlloQ/6XsLUn4xN+7HuUHe7iRfi7OQIOm8l2HR99Ix5lwm6nTOPgNf66KtAOwsK/x2GcH7L/XfehdFm/MHzqVqyM9XZwQC"
    "X3bHO1L6Ql/XKmK1X/TLe0PXnOHeVRW8twt/irpbF88XxfHz21+X5OxY+o6OPif8a0HTBQf7dzjY+Nc4+N/WzrnDZO+0ThqK"
    "DXx9eyoi1/wE1qYNIAJfkPC3s3xn2SoLtwfqa3Ow33qnjzR6g8uEKHMN6++I9MYC8obOLjjMbX2Pah9xVXSTZ7Xyrl+AAOXP"
    "dypd1/N/vgy1ilx3ixNdwMD3v6Paml69I9ourlAv8sXWMert7d9Y8AuGftfgdRtEd9tA+9fbYPG/JcUucKfJ/XA1vJ7U5VXk"
    "d+dh8U6t7e5e9+29TNPt/GRRoA+o4GvrffOyPmd4dIZ8mm8l2W2t3HsZ60+/GDbgy7d+/Lg9L63dT7pxVpVvCrz/8uVjyd9/"
    "7Ax1fk1emqg+GKBjqCEv8Q82A+f40lcd/vPrHRjxW4WWxq/v1dpDuVg63KXeY4xLuemNmP3Wk++NTsraeu5dX2qC7GsYZT+x"
    "tmjXoLXre+tN2opQU7Guk14rnd/eeuw764LDR8hwdhJOaIn593DhYh+QnFUu3Mawvw4kcYEKveh6Ze+lF93o8LEezVkt/WIx"
    "/D//A3wDOo9xFD+cFdZ7xlUj+U69uNE1v8jMvz8+Plo9NzSt41P21ulc1JsbTQnPDc1aT6KZs86drr9/tfKMLvf0Z9vKpEsg"
    "04fOxdzzc9aa7rY2s2ZkNK7rH43E0jJrWStkPnR+D1uTVyEyLaI+K2GW1uan15u87zdF04c6TTPNc8Ww8/aOwfQrntL9JR6/"
    "KgYpN4WeepddEHj8etXQvO2UJmbIzwm5sHNdi1qjoGVz1CoF0+qmLyJXH/ReG2OZe1qkTviZfud7NfKXdv+12OsKPpM78On+"
    "a/Dp/b+7yd5pcn5rYdDVcOIGeJox3s7dOczR3UjrhJ/ji4bx9voLQiG5017VOg/+vxoj+b8b4zuI1w6uXlghyqx7VZX1edvc"
    "5dZn5Oyo4/Xlpgtn3gZL/8Wasd4XRlQb6n+/gKY7flL9SvlnjlJ02YJNiJgGYkVh1Foltp3o9FAI6fQ+7JR3yqaTnyspGX/V"
    "Tf139WesxwZsvNeiYas7xaX7+fx32BTwWDNQGB2dTRuv3IiWPbF5B2/Vd2/bj9BXagBr41XkI/htQG92D3rD1jQraQGuW++x"
    "h++1RfeT9Qf4rXZOlT1Zj3W8K+sB6rx1etrFsurfhqw1HH9fp6HObjC9BdK1R4WWk/kXYFybUjWYu2a9XhJa6HwF9r+q/Hyp"
    "3Pm5dcmF0LxahrREQ40R221gWw9/XJO+/3a+ul34j3la1eY77zDpBdG7Rc39iCPfrSHnOVHLosA1LuyjD7feS4O9P1oM83LZ"
    "Pk9/NTZOf7LvPxDH517VXnruxvK9Nlp/el/7n9BbK0q4EzNckMlrmhtGzZ+7yRlaNOBe4f5jvW2a83A7Kh9z24NTs5vON7i7"
    "AvVeu4ykvam11Or9Te0nR/lDG/dHur1+fzzVzXKe7wvnLfKt87fHG4J6m/vz/eurVNN+fe1qJvzKbbyc13+h2PcRKL1by/MA"
    "7tfzrxSt9WH+r2P5QOTe9+gyqvaGDYG9WdXpXXbfry5c533+9Wox825nab/YWe/vZe5FwPGLraP9q62j/X+wdbT//7eO+r/c"
    "Or/eM//qPgKbkGlACNrgEfE9D7sJa99Eq6tN2q9IpObsXhpoLynfepZ+rVrX+JrWobDeepl+z83G4vjqfCO6hLX6iwuOT9mz"
    "9TsePvwZ9LLHa7FGNPY9fQrP1sRJey9wzw7HG/vhKK5Fd3d1rvjCPXNzXpM8rCNxvj65f+EBuVcG0Mvl6tWaDdbogjUfOj3/"
    "uUZUxuPZKefj2frsrZc3yX7v8fExDx5aJ0OdNrtXg+qnpsBj/fjzUi9N3tvbRznj9QpweaD8Rj+68Sr0fBH8XUs1++D5gjOb"
    "t96H7l4q1Rju3IeX/Ml/6/QmYTtTu8jI056WVqFRe+v5btVixVKrI67Vhq/Xj9nWpTd4xZgPtXp25+Vc+6FT430yqW21sOQh"
    "6nzQO76gT+WBO5P9QcMwExu6/3R3E7hsqTp+3IWcqrffbxdMdHWMUi/Yb2fx27uV/fHjfGgaHi1Z9d5t0mZ53+6MmfS61Laq"
    "1cj3zb+H5l+varWz/6u6sh/f3u4ueaF+t3fvLZ0TW9de/ufhf9J/PPyP2e388an3rfvhCXAf6wgYD1lzEQzbO+AfrfVzbYbW"
    "6V1f4PsX5FvnW3NXTJ6B//M/H/4Av461r7tv35G3H9fnwVvnv95/IbncR+s7QgsCn5+fkZc/Hh8f67RvVwvrrnWx2X3tNFeK"
    "dn/+AfbgXv9cLnv+Z6wlqcWE2cOF7Mt6WRfu9KBB53Lp/FQv+M2ePNEf/rB6WS/81nTmj6Tn9rRvz23a5fuXCQVgFL26Xfrv"
    "50cQGcOjlwyA4Mcx/PTwkHUfQRTtAFDz849/wI/9t6vI/hGG4ME/ku7jEELhf7jdR3AIw//Qbn1x79auWcfG69bVi9bvWVK1"
    "uR8u5O3uf6gFt0Xt1bQ+EsSZe/XwCps390jZy0PWmKdIWeVbz6//CYLga+8+yeqF+n2ZTuepmTGjtun/fj+Ht45r+kPne927"
    "1k9K7dZz98evQMbt4NbuIr5dd+tL0+kgzjPLbD79YDUDqc3drSSrGiOmh9evX2vfUF91+7XTCN2fsxdXr630607db9x6bcPO"
    "fz/2X15NLfFen179xtT9shHKOqhJ+RjUA6tDnGhfvvwl6eH1IW58yqRfjVpK+DU1HCuwnj7VLXZeO+eiVvrxE/ezdUm7zVek"
    "PzT76/n723n0SX13ubgXqJ0GPCaWmRvW5V5iXcKIPFl/ZN96Vn3bSR4ar2Cdl5fkgStuK3x2xOU+37nfenEfws6TlCVuaNc8"
    "ymtH0rYjnevCXvrwH69n9wYPZ+9mDw0t8fpf3623r//1PXl77dXkw1OrdBO+3btoMK6Nnof2+XP25Usd68RsvErcvCe/XpJr"
    "Z15cUV+ten80x8+vm0hezrlP91V6WX0lS378+BzWd65Uv1hE3YTsvWZeWp2JtBtepuZ76VRPbq+11n98fKwrh43sr/05V6yf"
    "u1qnxo/Rt7tV8z+M61zaSpJ6SdznduKTdxOfPIQvL9+bUWRvnaeLx6A/3PqC6tYg7tZ+/rP2e+7z2aeb9hyePcDfwthkD2E9"
    "Gd9rhoP29ntaupnhPITNpHW+G1pqvbaxn16f2ik4D74euBvuosYQtP7RusljasXd5LGt0LlprzRn/femtaumw1MLuMOzhsmX"
    "L1xxfvqjTfn2knxIOB/Pi5e2l3/VE+uuB9/dp/DRrcPkhE/hY/jy0uDss8ZMeImh06RdAt6E11g4NX/v7QLBUuv1plfyr+Yk"
    "udNBqaWAP52Ra7ycf7s1y/xlY+foOf+6pXO5XzZzdrjbrlH23BzH8NHQ8rT+vah3vfvI5TScqZ/2J3k8t/Rw3mBNqJ6kjq7z"
    "R/h4lkO9vFzN87/VbjnaVppP1cov59+sjqzTZGZnjsu3D91+u4ze+vO+739YTZ1vz82IW6fnl8602lGX8Zxdu90Br39jfNa1"
    "w+/mN2sJjf94rW+UP+1pTef/+f/dNN8FKnpogFQzY+2E/u2cmfVV+nIY//jj2mJipa2z9ct8ta6twzb55QV863z71qDL1nN6"
    "O455c24eay/erpU+hI969WfUnqXvby1SqMmjbzVe6r5++r//r9eu+5BdXN4kZ3PO363HOE+dhz+uu7buU/Olh+9nf+7th2td"
    "yBo4fuv0wnPUgZpP9pPKTV6t6nN8Cmu/6+cxdN6aMTRe1O8qniVG54pNXtP81Vf55waI/qL8tdTD95u79LvKbV9rjzi/aKHO"
    "aso0znV+UajJa0pdPI9/+XJ7bunhu7qt3LGpeyl1sai9q3e/Ck1Pf/rx2jiy+XYtwLisfdK8/fnHOfXb7x9qthgue4fhsofv"
    "Pz2LT9m3C8nd7n7rbu+cWZhnoiKx0q//9T280BTWmaaoua8f9/zPSXizwZud1nNfgzWvjvnqgATnOXrKHtPMCn78qKfqHGry"
    "HG7lqY0U99ZpXIfu9GfIQnqx/oxYSC+o3/o9R39uOLKGoyVY9gCe3f/Wt1g5ji9uc7s3XvPtPmY3nbsSeuFz9hhG5csLqWVW"
    "/dRL6pSLj16uddH78vLvOvN9OosY3Wcle/juRGnGNR7OGgSZHduH802gEdE00Q4vD8vau2KDN88KUs3zJY7hZ+gWSvUz1AR7"
    "rn/v4sh9hurgfs652ru7bp11YWJcitUyl7Y/F+FTndWa5T/B19C5F79VbRivcwffOfOp691c5Zw/Zlz6/y6cyB/fejdj6Nol"
    "TkNINcM8/6TPNSvBeAZ7/jPYy59rFw/Pn8Herhe3HJt58eA+Xmb2x49apSdos5jiIb66bXMuLiQj/ZLaqxN6dr13zLvETu/P"
    "KzfIyI7PLSz67D62K9XpFfcFvnxpfi5RQF+Clq1y00l5+B5flFPuS751mtB/F79gLZ5oilwpsQbA1Pv/vtn2yJw9tf/4ccn5"
    "EKO1PT+XMIxt4MQn6xJC8a3zVLSc2Pb36TrEW6Mtxfl28yH2ftDv+mZHD9/PX73Lfes83fLr2LO1g7B/2VR9w/37xi6hbG9O"
    "wv6muYu+60+avM7MOe02O/XiEL9u+vz8t0vQNFFem3jnPOFelPRX96933usa77udnmW17ZTvJTO3hbo4/L9gnUvBBuUEV6Ha"
    "2ZtD3bPMunbtqulyKdcknMkT++GWX1dt3zq90Hq/dRtYeiYKaodsV9Itqw9V9uy242z4Ji9X+dklYOxr5z0+erjrSlOg02t4"
    "/A3Eek9Yv2+wNhO4wfMgtV/y2/G2e01S592FJ7xceMIzOdd+KDuLE29fa9nD4eP77z9du5w9OlpaK5q8ZA2T/v6+flvwz9a9"
    "D/dXqdaU+KTV0Reungn/47U2uThDm1905ox02xG825/XiKrP4MtfZvM+8u8vZ/TpL9Uaae4vy791eqv7vdTs5hZuPvyqb51O"
    "j7o602w/c2m20yPbrM+fqYtnv4f26TE8E4g/flwSmlvmObHT6SUf9mVb6h1QDW80ZvhP8IXXMucxabxTW48uYD2G/4BAsGWu"
    "dXrupcVrX8gWdAb3Pbh08Hy8NKtGLZ8/S1fPhO5FBe7Ll89/XlOLy1MzQy8XPsKz1Yusj7jhPLe9mh/3br+/WPWuPqPY8Lnh"
    "6DTNnEf5Yj9kne71PNXBe157yfN11pvF/gP89tLsuJpC7GnPWduTl5f7frVX+ejDpH4/Ryx29Ifwy5fkfnuHnZekpbzCS+SB"
    "p6RzNsfJXlqyr/3E3yO0lpL9Q7us3WfNqrWTL+VvgYgvYF5rIF0veomusX7ddtBP0V0A3l+6oKw1izu9s83PU/TlS3TZOO8X"
    "vsUt7lP06PbCp+gxfOuci9/tzJfX7muXvxElD21u3cX6BM2f75a6JTAvzNiGvrxyZuv7fXHhK99Dslo9rN4673i4L6+N49/G"
    "MU4tQat35fdWftNoBNZx2Q9P3a7/9tYzrHMv3JsS0vPPiN0vXz5/vtG717JfvnwsfMt7LBM3azR2768od1E0Gp5ye9k5g9pe"
    "+DyvFd+al2erZ/34UYOTK5H7/BnqtJKNn0xY3cefz9nLY+2Isfb4+vp4k/A09nc1O/QhrM+VHb3++NE8NT79auLyy5fUergY"
    "YHV6n60vX67F29S2SjPl5/LuR42HRk2t8dJ4x5fNm8HnNXV7jdbQ0tW1Qkojy3n9qy/aa6FOYzp4fv7nM/jlS4212/q3Qr3a"
    "G3xLydfdbGj5Hz/eJ559zn5MbVzMdpp2r8m3Ehdx06ObMmFm2TWma/2XNT07P/7zGbq9/PczAp77eX5/vla4uz1aD1kTWMVs"
    "/j3fBx6ucLQGHJdr0LlrbUZ9U7hl1dukdxaGNoUu94Xnc1qnd7UzqeNWtF1qEp6zG0nckExnXHbGac/N21mj8kbZnWfoLuH5"
    "fX7Tr8ulp+1ZLcVIv3z5nDdI4mzNmzdB+uqMTq9VNL72rnl9zi7Kyq+1feiFRLsk/5lYux8/zhfQ7LEW0/74kZ0hc6fmoumf"
    "n5+181Denaca9NfZLaTPGm245ty928bnGa7zzmj3d8tPrU/u7qHBR+eeXj9zUVhtv9989XZFbL76+n7+WyLie8N0OFt+1giu"
    "Id7Oo2oZETeU1dImny+UUkuKNSl1P9/OpN9Lw4G/Up4XTuFPZqLTDv7Ll4fo3Zycx/35+fma794UF9sSlwt2M7h2os7Fe3+Z"
    "SbDTO1uHnofvWw91n+uUTs+wHs47zNeq5jZ4C8Ty0BocPLXZtbXy010/3zp38pHd5SS9vz9cVqYGPK0+8RVuNPzH55bU+vxX"
    "L9g/foT/DXa+v19KsOd+cO5bp9Vg8+Ki+ML/efuwB3ofV/16M3i732m9n27Eu+nufVjHXj2f4BVPXAF6Q3A1AKfFz99bnsYl"
    "+e3p+9vvv5jxK2ckvHFGboCxZg0nb3dwLL5B9zuw/JOpgnofZuXWRnDfxg0E3wo49wUuMPWWXWuFf79Al8/3k/SXvQ/2fj7s"
    "q+1xvdO0d3LFP+vWf1HrrJdd14nuJ6V4h+8vCt+98EyB1BJr66HTCmvf7k7Y60X/9/Wtkbr/La3RmKI6Vqso86GVWr34rRde"
    "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"
    "0fzQygxPd1K19aDa3ouHe6bzzojDZm2pDsag0aSW3eGu0X3J9XZibivmV7ts1cil5vnWy3GlDtOvQi3/mnnmuseGVh6nuEqX"
    "WzEZ5eM5R3/ks8/0avHWfK7XRHo/mK46eteU7g96c16bT8eT/Wjfb/XmwuOzuVjoCt+QO68HhXpdSmxNlzMU2B9ErZvrjTpA"
    "qs6F+VBfVqp1dlaejI8HrtzZTEa5CTcdHnOgrI+PC2qXqfTe0pwx2NOTrP667a3345J44LjGeL1KaUeDYtjux8cC+bSfK/Xe"
    "S0cYV7J8hpnOmuWB+Mav1lQ5O1zXnif1pvg+yMB+7dDSeX0qsuXxepqZ7u6raTgZVfZVMKaH/VKeL6fhcwnQTK+3PrRfOqMd"
    "23ljzcaquRfmq4ZBGUyaq7Xey+ustBNFSU53TbVcqwlitdNYvIvLUq/cSMF+XV1UwFyGmXV7aabWFbkyH4zrg1S/m65Wqzo/"
    "PHAcV5mMVH3ZOwB2sJoMtD3VmNReQb3/3OtVpNZ7uS302NG63S2bvXyJ0tj58f3N0BbTFrfrPtKle3oyWu7WwlulLmQP4+Pe"
    "kLkUKB23nemszgmGUB1t8xvlddLmG9Ka4t/vG5tjO/U2X/Ol6mZBZ7vzynJlULn8MK8JL+KKf0mX26NlNisud431tlmG686w"
    "aQwO9H5ehxlxlFst6vV6fpaeb5TR/iXb0BqTXuct3RTGH91aZXHcbNhK7XmvS+K7aWiP8zqQ79NU7Zkd0vXpgNlDeqEw3crr"
    "ulvn9s/ldqU2Y5nHtwzT4udZg531unpZbG74w+tK7aeXk01jtyp130vbt6amZ2fHfen4smvkFbmhNErP7P09zM1KnVV+Wvto"
    "iXlN3/VSHLvflSqLSalUe5vT9XQjuzsuFutq5XWeHpSrTHmTyTL9fG2vN2RDq99rEr99FwfPj9wHtduOFq2S2Xlk6oemMj3e"
    "v7RL3Y9yixo334/Z0kdlSb+Z0/zRHB9HvW2j/8KWn1WxnT/2zJyptYZvsPT42DPSA3piTkygbFLjXqoiZQfM23Cuap1ndvxB"
    "1+ixPNQnTFWvDQy5D0rvi0pm1FFbpg7g9CgrueykuX58XNXKImzPaH0rNZX7DMXst1UxCz6G6jiz78yen4VXQRl+8CO+NX7L"
    "GaNxesKrS0PZPYuP62H6nt40ah9qZ9KdrnMHbdOb7dY1fsFnjuoi1WyMh7oGao1nOH3JvvL0UTUyE2OzXLJKT2E2A2VFc3Wm"
    "NuhIY/Ejnzn0JkCmU7OX+8Ow+1Ivz5XBNj3Ypqtad6fI9fQbPXpv5g7qcjFRlnt+JG6nm0H3o9VaHLVOdXjfHvWlQ10frjLK"
    "sH0v7j/K1GEri7NhQ15XdYoa6K3umsnDfmk/qjzLy6GaZteb2vbtdb16X0xzL/3s27HHfPT1j8Gw/pIvrcyXNbXLVprbYf64"
    "1EUxC2bjLSjl5v38hh6+CG3QeG63KntRNvc808z0HyfHzIqj399pscpvuqtDShhtgDFSNrPSmMln38WekKoPmvNq7/69o+Tb"
    "rc2EWvGttqx0VbH2KqalQ0ZprTvvj2+9V7U+b6/e++19Rl7y7/fZR5NZa/eNGsztW6Bdbz5KyqordNTaqtJcpWdvu857xXzL"
    "0my7sxdfthtmKs7f8512SVi8jAYU2y8tP/R2rym/sxXJ5HW5sgKVCdd6ma5eXvfb/WI4fRvKO2kx1ida7eXQr5qDdKmUr79X"
    "+q16vzKYjN+798py+dFqIW9pttmuVkdzff9SAWN5/DzQ8ovHcenYbfXm++Fzo5HPA3MkjdfHpnTgtgtu0Dfp8rhPT/W3ZfNe"
    "GMv1Onfs99djtlzRddHY58oHqj8Am6pAN3r97mw/bnVEurzjjtPRbkRvKLMrcvXH9kfqeS9OtcMIzqrHhvm+nOnZntzfrSeM"
    "sGg876n6ZqdtuPaYnYKJ8rjUDiyX7bDZ3HNFHzRoqTypM3J2AabTMgCMUavpU/X4zAnCpgzUqfj+nhV6I7r6MTThsW68N6QJ"
    "SPfvB9lqezTIbWQOjMxRna7Jj+LwvUP39J9x9CgxMO5+4Bf/fsSL+H3AHRQWS6OQpqhYlqKsb/jJzYKMqovWF17QVfTmoL4D"
    "6ufSkETf44JboN25D3HagHXhCPH7lZ/4uVL7fdITp4iK5mmBHqeMF2eAWy809MqXp8j9GC9KQFsIcoH6PAPyEiDIS6gJhgex"
    "H0FS4uNFwKGnXpNBAlRNkIB2iBc5oMFwsUUfBmLKPNTQs8RJZT7XoVF4YKBUTO7gbC0YSQu3pCjGEj0SC2RDAKIAdMhbrTWI"
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


_warming = threading.Event()


def _warm_deps():
    """Load numpy and Pillow on a worker thread (once), so the page's first sync never waits for the imports."""
    if _DEPS is None and not _warming.is_set():
        _warming.set()
        threading.Thread(target=_deps, daemon=True).start()


def requirements(refresh: bool = False, wait: bool = True) -> list[dict]:
    """What Arrange and Fill need, for the page: [{id, name, status: "ok" | "missing" | "unknown", version, detail}].
    The imports run once (_deps()); `wait=False` reports "unknown" until they did instead of running them here.
    `refresh` (Check again) retries the imports of anything missing (they succeed once the packages are installed,
    usually after a restart)."""
    global _DEPS
    with _DEPS_LOCK:
        if refresh and _DEPS is not None and any(isinstance(v, Exception) for v in _DEPS.values()):
            _DEPS = _load_deps()
    found_all = _deps() if wait else _DEPS
    rows = []
    for rid, name in REQUIREMENTS:
        found = (found_all or {}).get(rid)
        ok = isinstance(found, str)
        status = "unknown" if found_all is None else "ok" if ok else "missing"
        rows.append({"id": rid, "name": name, "status": status, "version": found if ok else None,
                     "detail": f"{type(found).__name__}: {found}" if status == "missing" else None})
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
            requirements(refresh=True, wait=False)
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
