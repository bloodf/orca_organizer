# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pillow"]
#
# [tool.orcaslicer.plugin]
# name = "Shape-Aware Arrange"
# description = "Arranges a plate using each object's real footprint, not its bounding box or convex hull. Concave parts (snowflakes, stars, combs, letters) interlock to fit more on the plate."
# author = "Heitor Ramon"
# version = "0.1.0"
# ///
"""Shape-aware plate arrangement for OrcaSlicer.

Orca 2.5 plugins cannot move objects, so the plugin exports a 3MF copy of the
project, rewrites the build-item transforms in it and reopens it. The nesting
core works on any 3MF and is also usable from the command line:

    python shape_arrange.py in.3mf out.3mf --plate 1 --spacing 2 --step 30
"""

from __future__ import annotations

import json
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
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


@dataclass
class Params:
    plate: int = 1          # 1-based plate to arrange
    spacing: float = 2.0    # min gap between parts, mm
    step: float = 30.0      # rotation step, degrees; 0 disables rotation
    res: float = 0.5        # raster cell size, mm
    center: bool = True     # center the finished layout on the plate


@dataclass
class Item:
    index: int                      # position among <build><item> tags
    object_id: str
    transform: np.ndarray           # 4x4, column-vector convention
    parts: list = field(default_factory=list)  # [(verts_world (N,3), tris (M,3))]
    name: str = ""
    printable: bool = True


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


TRANSFORM_RE = r"""\btransform=(["']).*?\1"""


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf"""\s{name}=(["'])(.*?)\1""", tag)
    return m.group(2) if m else None


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
        self.names = set(self.zip.namelist())
        rels = self._text("_rels/.rels") or ""
        m = re.search(r'Target="/?([^"]+\.model)"', rels)
        self.main_name = m.group(1) if m else "3D/3dmodel.model"
        self.main_text = self._text(self.main_name)
        if self.main_text is None:
            raise ValueError(f"{self.path.name}: no 3D model part found")
        self._trees: dict[str, ET.Element] = {}
        self.settings = self._load_settings()

    def _text(self, name: str) -> str | None:
        return self.zip.read(name).decode("utf-8") if name in self.names else None

    def _tree(self, name: str) -> ET.Element:
        if name not in self._trees:
            self._trees[name] = ET.fromstring(self.zip.read(name))
        return self._trees[name]

    def _object(self, file: str, oid: str) -> ET.Element:
        for obj in self._tree(file).iter(CORE + "object"):
            if obj.get("id") == oid:
                return obj
        raise ValueError(f"object {oid} not found in {file}")

    def _load_settings(self) -> dict:
        """Part subtypes, object names and plate membership from Orca/Bambu metadata."""
        out = {"subtypes": {}, "names": {}, "plates": {}}
        text = self._text("Metadata/model_settings.config")
        if not text:
            return out
        root = ET.fromstring(text)
        for obj in root.iter("object"):
            oid = obj.get("id")
            for md in obj.findall("metadata"):
                if md.get("key") == "name":
                    out["names"][oid] = md.get("value", "")
            out["subtypes"][oid] = {p.get("id"): p.get("subtype", "normal_part")
                                    for p in obj.findall("part")}
        for plate in root.iter("plate"):
            meta = {m.get("key"): m.get("value") for m in plate.findall("metadata")}
            pid = int(meta.get("plater_id", len(out["plates"]) + 1))
            members = set()
            for mi in plate.findall("model_instance"):
                kv = {m.get("key"): m.get("value") for m in mi.findall("metadata")}
                members.add((kv.get("object_id"), int(kv.get("instance_id", 0))))
            out["plates"][pid] = members
        return out

    def bed(self):
        """(printable polygon, [exclude polygons]) in plate-local mm."""
        text = self._text("Metadata/project_settings.config")
        cfg = json.loads(text) if text else {}

        def poly(key):
            pts = [tuple(float(c) for c in p.split("x")) for p in cfg.get(key, []) or []]
            return pts if len(pts) >= 3 else None

        return poly("printable_area"), [p for p in [poly("bed_exclude_area")] if p]

    def _item_spans(self) -> list[tuple[int, int]]:
        """(start, end) of every <build><item> tag in main_text; comments ignored."""
        b = re.search(r"<build\b.*?</build>", self.main_text, re.S)
        if not b:
            raise ValueError("3MF has no <build> section")
        text = re.sub(r"<!--.*?-->", lambda m: " " * len(m.group(0)), b.group(0), flags=re.S)
        return [(b.start() + m.start(), b.start() + m.end()) for m in re.finditer(r"<item\b[^>]*>", text)]

    def items(self) -> list[Item]:
        out = []
        for i, (a, b) in enumerate(self._item_spans()):
            tag = self.main_text[a:b]
            oid = _attr(tag, "objectid")
            item = Item(i, oid, _mat_from_3mf(_attr(tag, "transform")),
                        name=self.settings["names"].get(oid, f"object {oid}"),
                        printable=_attr(tag, "printable") != "0")
            self._collect(self.main_name, oid, item.transform, item.parts, oid)
            out.append(item)
        return out

    def _collect(self, file, oid, mat, parts, top_oid, part_id=None):
        obj = self._object(file, oid)
        mesh = obj.find(CORE + "mesh")
        if mesh is not None:
            subtype = self.settings["subtypes"].get(top_oid, {}).get(part_id or oid, "normal_part")
            if subtype == "normal_part":
                v, t = _mesh_arrays(mesh)
                if len(t):
                    vw = (np.c_[v, np.ones(len(v))] @ mat.T)[:, :3]
                    parts.append((vw, t))
            return
        comps = obj.find(CORE + "components")
        for c in [] if comps is None else comps.findall(CORE + "component"):
            sub_file = (c.get(PROD + "path") or "").lstrip("/") or file
            sub_mat = mat @ _mat_from_3mf(c.get("transform"))
            self._collect(sub_file, c.get("objectid"), sub_mat, parts, top_oid, c.get("objectid"))

    def write(self, out: Path, new_transforms: dict[int, np.ndarray]):
        text = self._replace_transforms(new_transforms)
        tmp = Path(out).with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in self.zip.infolist():
                data = text.encode("utf-8") if info.filename == self.main_name else self.zip.read(info)
                zout.writestr(info, data)
        self.zip.close()  # Windows cannot replace a file that is still open
        tmp.replace(out)

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


def _rasterize(tris: np.ndarray, res: float):
    """Mask of cells touched by `tris` (coords relative to the part center) and the center in cells."""
    lo = np.floor(tris.reshape(-1, 2).min(0) / res) - 1
    px = tris / res - lo - 0.5  # PIL samples pixel centers at integer coordinates
    w, h = (np.ceil(tris.reshape(-1, 2).max(0) / res) - lo + 2).astype(int)
    img = Image.new("1", (int(w), int(h)), 0)
    draw = ImageDraw.Draw(img)
    for tri in px.tolist():
        draw.polygon([tuple(q) for q in tri], fill=1)
    return np.array(img, dtype=bool), -lo


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros((h + 2 * r, w + 2 * r), bool)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out[r + dy:r + dy + h, r + dx:r + dx + w] |= mask
    return out


@dataclass
class Shape:
    angle: float          # world rotation applied, degrees CCW
    raw: np.ndarray       # footprint cells
    kernel: np.ndarray    # footprint dilated by spacing
    pad: int              # kernel = raw offset by (pad, pad)
    cx: float             # silhouette center within kernel, cells
    cy: float


def _shapes(item: Item, p: Params):
    """Candidate footprints of an item, one per rotation."""
    xy = np.concatenate([v[:, :2] for v, _ in item.parts])
    center = (xy.min(0) + xy.max(0)) / 2
    tris = _silhouette(item.parts) - center
    r = int(math.ceil(p.spacing / p.res))
    angles = [0.0]
    if p.step > 0:  # absolute multiples of step (so parts can land axis-aligned), plus "as is"
        current = math.degrees(math.atan2(item.transform[1, 0], item.transform[0, 0]))
        angles += sorted({round((a - current) % 360, 6) for a in np.arange(0, 360, p.step)} - {0.0})
    out = []
    for a in angles:
        c, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
        m, (cx, cy) = _rasterize(tris @ np.array([[c, s_], [-s_, c]]), p.res)
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        m = _dilate(m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], 1)  # +1 cell: PIL fill under-covers
        out.append(Shape(float(a), m, _dilate(m, r), r, cx - cols[0] + 1 + r, cy - rows[0] + 1 + r))
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

def _best_position(occ_f, occ_shape, shape: Shape, bbox):
    ny, nx = occ_shape
    kh, kw = shape.kernel.shape
    if kh > ny or kw > nx:
        return None
    kf = np.fft.rfft2(shape.kernel.astype(float), s=occ_shape)
    hits = np.fft.irfft2(occ_f * np.conj(kf), s=occ_shape)[: ny - kh + 1, : nx - kw + 1]
    ys, xs = np.nonzero(hits < 0.5)
    if not len(xs):
        return None
    x0, y0, x1, y1 = bbox or (nx, ny, 0, 0)  # first part: plain footprint area
    w = np.maximum(x1, xs + kw) - np.minimum(x0, xs)
    h = np.maximum(y1, ys + kh) - np.minimum(y0, ys)
    score = (w * h).astype(float) + 1e-3 * (xs + ys)
    i = int(np.argmin(score))
    return score[i], int(xs[i]), int(ys[i])


def nest(items: list[Item], blocked: np.ndarray, p: Params, progress=None):
    """Place items on the grid. Returns ({item.index: (shape, x, y, center)}, [unplaced items])."""
    prepared = []
    for n, it in enumerate(items):
        if progress:
            progress(n, len(items) * 2, f"Footprint {it.name}")
        center, shapes = _shapes(it, p)
        prepared.append((it, center, shapes))
    prepared.sort(key=lambda t: -int(t[2][0].raw.sum()))

    occ = blocked.astype(float)
    bbox, placed, failed = None, {}, []
    for n, (it, center, shapes) in enumerate(prepared):
        if progress:
            progress(len(items) + n, len(items) * 2, f"Placing {it.name}")
        occ_f = np.fft.rfft2(occ)
        best = None
        for s in shapes:
            r = _best_position(occ_f, occ.shape, s, bbox)
            # shapes[0] is the current rotation; others must win by 1% (beats raster noise)
            if r and (best is None or r[0] < best[0] * 0.99):
                best = (r[0], s, r[1], r[2])
        if best is None:
            failed.append(it)
            continue
        _, s, x, y = best
        h, w = s.raw.shape
        rx, ry = x + s.pad, y + s.pad
        occ[ry:ry + h, rx:rx + w] = np.maximum(occ[ry:ry + h, rx:rx + w], s.raw)
        kh, kw = s.kernel.shape
        bbox = (x, y, x + kw, y + kh) if bbox is None else (
            min(bbox[0], x), min(bbox[1], y), max(bbox[2], x + kw), max(bbox[3], y + kh))
        placed[it.index] = (s, x, y, center)

    if p.center and placed:
        _center_layout(placed, blocked, bbox)
    return placed, failed


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


def arrange_3mf(src: Path, dst: Path, p: Params = Params(), bed=None, progress=None) -> dict:
    """Arrange one plate of `src` and write the result to `dst`. Returns a report dict."""
    pkg = Package(src)
    poly, excludes = pkg.bed()
    poly = poly or bed or DEFAULT_BED
    items = pkg.items()

    plates = pkg.settings["plates"]
    if plates:
        if p.plate not in plates:
            raise ValueError(f"plate {p.plate} does not exist (project has {len(plates)})")
        seen: dict[str, int] = {}
        members = []
        for it in items:  # instance_id = order among build items of the same object
            k = seen.get(it.object_id, 0)
            seen[it.object_id] = k + 1
            if (it.object_id, k) in plates[p.plate]:
                members.append(it)
        items = members
    movable = [it for it in items if it.printable and it.parts and len(_silhouette(it.parts))]
    if not movable:
        raise ValueError(f"plate {p.plate} has no printable objects")

    xs, ys = zip(*poly)
    origin = plate_origin(p.plate, max(1, len(plates)), max(xs) - min(xs), max(ys) - min(ys))
    moving = {it.index for it in movable}
    obstacles = [_silhouette(it.parts) - origin for it in items if it.index not in moving and it.parts]
    grid_origin, blocked = _bed_grid(poly, excludes, p.res, [t for t in obstacles if len(t)])
    placed, failed = nest(movable, blocked, p, progress)

    by_index = {it.index: it for it in movable}
    new = {}
    for idx, (s, x, y, center) in placed.items():
        dst_xy = origin + grid_origin + np.array([x + s.cx, y + s.cy]) * p.res
        new[idx] = _delta(s.angle, center, dst_xy) @ by_index[idx].transform
    # parts that did not fit go in a row above the plate grid, where no plate is
    cursor, top = origin[0] + min(xs), max(ys) + 10
    for it in failed:
        xy = np.concatenate([v[:, :2] for v, _ in it.parts])
        lo, hi = xy.min(0), xy.max(0)
        new[it.index] = _delta(0, lo, np.array([cursor, top])) @ it.transform
        cursor += hi[0] - lo[0] + max(p.spacing, 5)
    pkg.write(Path(dst), new)
    return {"placed": len(placed), "unplaced": [it.name for it in failed], "output": str(dst)}


# ---------------------------------------------------------------- Orca plugin

PANEL_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{background:var(--orca-bg);color:var(--orca-fg);font:13px var(--orca-font);margin:12px}
label{display:flex;justify-content:space-between;align-items:center;margin:8px 0;gap:8px}
input,select{width:90px;background:var(--orca-bg);color:var(--orca-fg);border:1px solid var(--orca-border);padding:3px}
button{width:100%;margin-top:12px;padding:8px;background:var(--orca-accent);color:var(--orca-accent-fg);border:0;border-radius:4px;cursor:pointer}
button:disabled{opacity:.5}#status{margin-top:12px;color:var(--orca-muted);white-space:pre-wrap}
</style></head><body>
<label>Plate <input id="plate" type="number" min="1" value="1"></label>
<label>Spacing (mm) <input id="spacing" type="number" min="0" step="0.5" value="2"></label>
<label>Rotation step <select id="step">
<option value="0">No rotation</option><option value="90">90&deg;</option><option value="45">45&deg;</option>
<option value="30" selected>30&deg;</option><option value="15">15&deg;</option><option value="10">10&deg;</option></select></label>
<label>Precision <select id="res">
<option value="1">Fast (1 mm)</option><option value="0.5" selected>Normal (0.5 mm)</option><option value="0.25">Fine (0.25 mm)</option></select></label>
<button id="go">Arrange plate</button><div id="status">The arranged project opens as a new file. Your original is not changed.</div>
<script>
const $=id=>document.getElementById(id);
$('go').onclick=()=>{$('go').disabled=true;$('status').textContent='Arranging…';
 orca.postMessage({action:'arrange',plate:+$('plate').value,spacing:+$('spacing').value,step:+$('step').value,res:+$('res').value});};
orca.onMessage(m=>{$('status').textContent=m.text;if(m.done)$('go').disabled=false;});
</script></body></html>"""


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


def _run_from_panel(panel, storage: Path, data: dict):
    # Runs on the UI thread (dock-panel callback), which export_3mf_copy requires.
    # ponytail: UI blocks for a few seconds while nesting; move to a worker thread if big plates hurt.
    try:
        p = Params(plate=max(1, int(data.get("plate", 1))),
                   spacing=max(0.0, float(data.get("spacing", 2))),
                   step=max(0.0, float(data.get("step", 30))),
                   res=min(2.0, max(0.1, float(data.get("res", 0.5)))))
        plater = orca.host.plater()
        project = _call(plater, "project_path") or ""
        stem = Path(project).stem if project else "untitled"
        storage.mkdir(parents=True, exist_ok=True)
        src, dst = storage / "_source.3mf", storage / f"{stem}-arranged.3mf"
        plater.export_3mf_copy(str(src))

        report = arrange_3mf(src, dst, p)
        text = f"Placed {report['placed']} object(s)."
        if report["unplaced"]:
            text += f"\nDid not fit (moved above the plate): {', '.join(report['unplaced'])}"
        if _open_in_orca(dst):
            text += "\nOpening the arranged copy…"
        else:
            text += f"\nOpen this file: {dst}"
        panel.post({"text": text, "done": True})
    except Exception as e:  # report every failure in the panel, never crash the host
        import traceback

        traceback.print_exc()  # lands in OrcaSlicer's python log
        panel.post({"text": f"Error: {e}", "done": True})


if orca is not None:
    _panel = None

    class ArrangeScript(orca.script.ScriptPluginCapabilityBase):
        def get_name(self):
            return "Shape-Aware Arrange"

        def execute(self):
            global _panel
            storage = Path(orca.host.plugin.storage())
            if _panel is not None and _panel.is_open():
                _panel.show()
            else:
                _panel = orca.host.ui.create_dock_panel(
                    html=PANEL_HTML, title="Shape-Aware Arrange", width=280, height=320,
                    dock="right", on_message=lambda data: _run_from_panel(_panel, storage, data))
            return orca.ExecutionResult.success("Panel opened")

    @orca.plugin
    class ShapeArrangePlugin(orca.base):
        def register_capabilities(self):
            orca.register_capability(ArrangeScript)


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
    ap.add_argument("--bed", help="WxH in mm when the 3MF has no printer settings, e.g. 270x270")
    ap.add_argument("--preview", type=Path, help="write a top-down PNG of the result")
    a = ap.parse_args()
    bed = None
    if a.bed:
        w, h = (float(v) for v in a.bed.lower().split("x"))
        bed = [(0, 0), (w, 0), (w, h), (0, h)]
    params = Params(a.plate, a.spacing, a.step, a.res)
    print(json.dumps(arrange_3mf(a.src, a.dst, params, bed=bed,
                                 progress=lambda i, n, m: print(f"[{i}/{n}] {m}", file=sys.stderr)), indent=2))
    if a.preview:
        _preview(a.dst, a.preview, params, bed)
