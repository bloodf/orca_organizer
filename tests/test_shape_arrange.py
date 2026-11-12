"""End-to-end check: build an Orca-style 3MF, arrange it, verify the geometry of the result.

Run: uv run --python 3.12 --with numpy --with pillow python tests/test_shape_arrange.py
"""

import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shape_arrange as sa  # noqa: E402

BED_W, BED_H = 75.0, 56.0


def box(x0, y0, x1, y1, h=2.0):
    v = [(x, y, z) for z in (0, h) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v, f


def comb():
    """60 x 38 mm comb: 8 mm spine, five 4 mm teeth with 8 mm gaps. Concave, interlocks with itself at 180 deg."""
    boxes = [box(0, 0, 60, 8)] + [box(2 + 12 * i, 8, 6 + 12 * i, 38) for i in range(5)]
    verts, faces = [], []
    for v, f in boxes:
        faces += [(a + len(verts), b + len(verts), c + len(verts)) for a, b, c in f]
        verts += v
    return verts, faces


def mesh_xml(oid, mesh):
    v, f = mesh
    vs = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in v)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
    return f'<object id="{oid}" type="model"><mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh></object>'


def tf(angle_deg, x, y):
    m = np.eye(4)
    a = math.radians(angle_deg)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    m[:2, 3] = (x, y)
    return sa._mat_to_3mf(m)


def make_project(path: Path):
    """Two plates. Plate 1: one block (must not move). Plate 2: two combs + one small block."""
    ns = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
          'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"')
    sub = {  # external object files, like Orca/Bambu write them
        "3D/Objects/object_1.model": mesh_xml(1, comb()),
        "3D/Objects/object_3.model": mesh_xml(3, box(0, 0, 20, 20)),
        "3D/Objects/object_5.model": mesh_xml(5, box(0, 0, 10, 10)),
    }
    objects = "".join(
        f'<object id="{oid}" p:UUID="0000000{oid}-0000-0000-0000-000000000000" type="model"><components>'
        f'<component p:path="/{f}" objectid="{sid}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object>'
        for oid, sid, f in [(2, 1, "3D/Objects/object_1.model"), (4, 3, "3D/Objects/object_3.model"),
                            (6, 5, "3D/Objects/object_5.model")])
    px = BED_W * 1.2  # plate 2 origin x
    items = "".join([
        f'<item objectid="4" p:UUID="a1" transform="{tf(0, 20, 20)}" printable="1"/>',          # plate 1
        f'<item objectid="2" p:UUID="a2" transform="{tf(30, px + 5, 5)}" printable="1"/>',      # plate 2 comb #0
        f'<item objectid="2" p:UUID="a3" transform="{tf(-10, px + 10, 30)}" printable="1"/>',   # plate 2 comb #1
        f'<item objectid="6" p:UUID="a4" transform="{tf(0, px + 60, 40)}" printable="1"/>',     # plate 2 small block
    ])
    main = f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns}><resources>{objects}</resources><build p:UUID="b">{items}</build></model>'

    def plate(pid, members):
        mi = "".join(f'<model_instance><metadata key="object_id" value="{o}"/><metadata key="instance_id" value="{i}"/></model_instance>'
                     for o, i in members)
        return f'<plate><metadata key="plater_id" value="{pid}"/>{mi}</plate>'

    settings = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>'
                '<object id="2"><metadata key="name" value="comb"/><part id="1" subtype="normal_part"/></object>'
                '<object id="4"><metadata key="name" value="block"/><part id="3" subtype="normal_part"/></object>'
                '<object id="6"><metadata key="name" value="small"/><part id="5" subtype="normal_part"/></object>'
                + plate(1, [(4, 0)]) + plate(2, [(2, 0), (2, 1), (6, 0)]) + '</config>')
    project = {"printable_area": ["0x0", f"{BED_W:g}x0", f"{BED_W:g}x{BED_H:g}", f"0x{BED_H:g}"],
               "bed_exclude_area": ["0x0"]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("_rels/.rels", '<Relationships><Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", main)
        for name, body in sub.items():
            z.writestr(name, f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" {ns}><resources>{body}</resources><build/></model>')
        z.writestr("Metadata/model_settings.config", settings)
        z.writestr("Metadata/project_settings.config", json.dumps(project))


def footprint(item, origin, res=0.1):
    """Exact-ish top-down footprint of an item on the plate grid (no spacing, fine raster)."""
    img = Image.new("1", (int(BED_W / res) + 1, int(BED_H / res) + 1), 0)
    draw = ImageDraw.Draw(img)
    for v, t in item.parts:
        px = (v[:, :2] - origin) / res
        for tri in px[t].tolist():
            draw.polygon([tuple(p) for p in tri], fill=1)
    return np.array(img, dtype=bool), px


def test_interlocking_combs_fit_on_plate_2():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        make_project(src)
        report = sa.arrange_3mf(src, dst, sa.Params(plate=2, spacing=1.0, step=90, res=0.25))
        assert report["unplaced"] == [], report
        assert report["placed"] == 3

        before, after = sa.Package(src), sa.Package(dst)
        b_items, a_items = before.items(), after.items()

        # plate 1 block untouched, byte-for-byte
        assert np.allclose(b_items[0].transform, a_items[0].transform)
        # every other zip entry untouched
        for name in before.names - {before.main_name}:
            assert before.zip.read(name) == after.zip.read(name), name

        origin = sa.plate_origin(2, 2, BED_W, BED_H)
        masks = []
        for it in a_items[1:]:
            xy = np.concatenate([v[:, :2] for v, _ in it.parts]) - origin
            assert xy.min() >= -1e-6 and xy[:, 0].max() <= BED_W and xy[:, 1].max() <= BED_H, (it.name, xy.min(0), xy.max(0))
            z = np.concatenate([v[:, 2] for v, _ in it.parts])
            assert abs(z.min()) < 1e-6, "parts must stay on the bed"
            masks.append(footprint(it, origin)[0])
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert not (masks[i] & masks[j]).any(), f"items {i} and {j} overlap"

        # the two combs only fit side by side if their teeth interlock: bounding boxes must overlap
        boxes = [np.r_[np.concatenate([v[:, :2] for v, _ in it.parts]).min(0),
                       np.concatenate([v[:, :2] for v, _ in it.parts]).max(0)] for it in a_items[1:3]]
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = boxes
        assert min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0), "combs did not interlock"


def star(r_out=9.0, r_in=3.5, points=6, h=2.0):
    """Concave star prism: top/bottom fans are enough for a footprint."""
    ring = [((r_out if k % 2 == 0 else r_in) * math.cos(math.pi * k / points),
             (r_out if k % 2 == 0 else r_in) * math.sin(math.pi * k / points)) for k in range(2 * points)]
    n = len(ring)
    verts = [(0, 0, 0), (0, 0, h)] + [(x, y, 0) for x, y in ring] + [(x, y, h) for x, y in ring]
    faces = [(0, 2 + (k + 1) % n, 2 + k) for k in range(n)] + [(1, 2 + n + k, 2 + n + (k + 1) % n) for k in range(n)]
    return verts, faces


def test_crowded_plate_spacing_zero_with_obstacle():
    """60 stars + a non-printable block on a 100 mm bed, spacing 0: no overlaps, overflow leaves the plate."""
    bed = 100.0
    rng = np.random.default_rng(7)
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
    objects = mesh_xml(1, star()) + mesh_xml(2, box(-15, -15, 15, 15))
    items = f'<item objectid="2" transform="{tf(0, 50, 50)}" printable="0"/>' + "".join(
        f'<item objectid="1" transform="{tf(rng.uniform(0, 360), rng.uniform(0, bed), rng.uniform(0, bed))}"/>'
        for _ in range(60))
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        with zipfile.ZipFile(src, "w") as z:
            z.writestr("3D/3dmodel.model", f'<model unit="millimeter" {ns}><resources>{objects}</resources><build>{items}</build></model>')
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=0, step=15, res=0.5),
                                bed=[(0, 0), (bed, 0), (bed, bed), (0, bed)])
        assert report["placed"] + len(report["unplaced"]) == 60 and report["unplaced"], report

        res = 0.05
        on_plate = []
        for it in sa.Package(dst).items():
            xy = np.concatenate([v[:, :2] for v, _ in it.parts])
            inside = xy.min() >= 0 and xy.max() <= bed
            off = xy[:, 1].min() >= bed  # overflow row sits above the plate
            assert inside or off, (it.name, xy.min(0), xy.max(0))
            if inside:
                img = Image.new("1", (int(bed / res), int(bed / res)), 0)
                draw = ImageDraw.Draw(img)
                for v, t in it.parts:
                    for tri in (v[:, :2] / res)[t].tolist():
                        draw.polygon([tuple(q) for q in tri], fill=1)
                on_plate.append(np.array(img, dtype=bool))
        total = np.sum(on_plate, axis=0)
        assert (total > 1).sum() <= 4, f"overlap of {(total > 1).sum()} cells at {res} mm"


def test_auto_mode_refines_to_fit_tight_interlock():
    """Plugin presets: 0.5 mm can't interlock the combs, the automatic 0.25 mm retry can.

    1.2 mm spacing: since 0.9.0 the derived margin lets 0.5 mm interlock them at 1 mm, so the
    gap is set where only the finer grid still fits (1.1 to 1.25 mm do; 1.3 mm fits neither)."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        make_project(src)
        coarse = sa.arrange_3mf(src, dst, sa.Params(plate=2, spacing=1.2, step=15, res=0.5))
        auto = sa.arrange_3mf(src, dst, sa.Params(plate=2, spacing=1.2, **sa.AUTO))
        assert coarse["unplaced"] and not auto["unplaced"], (coarse, auto)


def merge(*meshes):
    verts, faces = [], []
    for v, f in meshes:
        faces += [(a + len(verts), b + len(verts), c + len(verts)) for a, b, c in f]
        verts += v
    return verts, faces


def box3(x0, y0, z0, x1, y1, z1):
    v, f = box(x0, y0, x1, y1, z1 - z0)
    return [(x, y, z + z0) for x, y, z in v], f


def simple_project(path: Path, objects: list, items: list, settings: dict, model_settings: str = ""):
    """Plain 3MF: objects = [mesh], items = [(object index, transform string)]."""
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
    res = "".join(mesh_xml(i + 1, m) for i, m in enumerate(objects))
    build = "".join(f'<item objectid="{o + 1}" transform="{t}"/>' for o, t in items)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", f'<model unit="millimeter" {ns}><resources>{res}</resources><build>{build}</build></model>')
        z.writestr("Metadata/project_settings.config", json.dumps(settings))
        if model_settings:
            z.writestr("Metadata/model_settings.config", model_settings)


def projections(path: Path, bed: float, res=0.05):
    """Full top-down projection (every height) of each item, plus its XY bounds."""
    out = []
    for it in sa.Package(path).items():
        img = Image.new("1", (int(bed / res), int(bed / res)), 0)
        draw = ImageDraw.Draw(img)
        for v, t in it.parts:
            for tri in (v[:, :2] / res)[t].tolist():
                draw.polygon([tuple(q) for q in tri], fill=1)
        xy = np.concatenate([v[:, :2] for v, _ in it.parts])
        out.append((np.array(img, dtype=bool), xy.min(0), xy.max(0)))
    return out


def test_overhangs_block_space_at_every_height():
    """A mushroom (4 mm stem, 30 mm cap at z=20) must keep blocks out from under its cap."""
    bed = 60.0
    mushroom = merge(box3(13, 13, 0, 17, 17, 20), box3(0, 0, 20, 30, 30, 22))
    block = box(0, 0, 8, 8)
    settings = {"printable_area": ["0x0", "60x0", "60x60", "0x60"]}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [mushroom, block], [(0, tf(0, 5, 5))] + [(1, tf(0, 20, 20))] * 6, settings)
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=0.5, step=90, res=0.25))
        assert report["placed"] >= 4, report
        masks = []
        for m, lo, hi in projections(dst, bed):
            if lo[1] >= bed:  # overflow row above the plate
                continue
            assert lo.min() >= -1e-6 and hi.max() <= bed + 1e-6, f"part off the bed: {lo}, {hi}"
            masks.append(m)
        total = np.sum(masks, axis=0)
        assert (total > 1).sum() == 0, "a part was placed under the mushroom cap"


def test_by_object_sequence_keeps_extruder_clearance():
    bed = 120.0
    settings = {"printable_area": ["0x0", "120x0", "120x120", "0x120"],
                "print_sequence": "by object", "extruder_clearance_radius": "20"}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 10, 10)], [(0, tf(0, 50, 50))] * 3, settings)
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=2, step=90, res=0.5))
        assert report["placed"] == 3 and report["spacing"] == 20, report
        boxes = [(lo, hi) for _, lo, hi in projections(dst, bed, res=0.5)]
        for i in range(3):
            for j in range(i + 1, 3):
                (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
                gap = np.maximum(np.maximum(blo - ahi, alo - bhi), 0)
                assert np.hypot(*gap) >= 20 - 0.5, f"parts {i},{j} only {np.hypot(*gap):.1f} mm apart"


def test_fill_plate_adds_registered_copies_without_overlap():
    """Fill plate 1 (one 20 mm block) of the two-plate project; plate 2 must not change."""
    import re

    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        make_project(src)
        report = sa.arrange_3mf(src, dst, sa.Params(plate=1, spacing=2, step=90, res=0.5, fill=True))
        # 75 x 56 bed, 20 mm blocks with 2 mm gaps and edge margin: 3 x 2 = 6 in total
        assert report["placed"] == 1 and report["added"] == 5, report

        out = sa.Package(dst)
        items = out.items()
        assert len(items) == 4 + 5
        new_tags = re.findall(r"<item\b[^>]*>", out.main_text)[4:]
        assert all('objectid="4"' in t and "p:UUID=" in t for t in new_tags), new_tags
        assert len({re.search(r'p:UUID="([^"]+)"', t).group(1) for t in new_tags}) == 5

        plates = out.settings["plates"]
        assert plates[1] == {("4", k) for k in range(6)}, plates[1]
        assert plates[2] == {("2", 0), ("2", 1), ("6", 0)}
        ids = re.findall(r'key="identify_id" value="(\d+)"', out.zip.read("Metadata/model_settings.config").decode())
        assert len(ids) == len(set(ids)), "identify_id must stay unique"

        before = sa.Package(src).items()
        for b, a in zip(before[1:4], items[1:4]):  # plate 2 untouched
            assert np.allclose(b.transform, a.transform)

        origin = sa.plate_origin(1, 2, BED_W, BED_H)
        masks = []
        for it in [items[0]] + items[4:]:
            xy = np.concatenate([v[:, :2] for v, _ in it.parts]) - origin
            assert xy.min() >= -1e-6 and xy[:, 0].max() <= BED_W and xy[:, 1].max() <= BED_H, xy
            masks.append(footprint(it, origin)[0])
        assert (np.sum(masks, axis=0) > 1).sum() == 0, "copies overlap"


def test_identical_brackets_tile_in_a_repeating_pattern():
    """20 L-brackets (100 x 75 mm, 11 mm arms) on a 256 mm bed. Greedy placement alone fits 13 here;
    tiling them as a repeating pattern fits 18. No two may overlap."""
    bracket = merge(box(0, 0, 100, 11), box(0, 11, 11, 75))
    bed = {"printable_area": ["0x0", "256x0", "256x256", "0x256"]}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [bracket], [(0, tf(0, 5 + 3 * i, 5 + 3 * i)) for i in range(20)], bed)
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=2, step=15, res=0.5))
        assert report["placed"] >= 18, report
        masks = [m for m, lo, hi in projections(dst, 256.0, 0.1) if lo[1] < 256]
        assert len(masks) == report["placed"]
        assert (np.sum(masks, axis=0) > 1).sum() == 0, "brackets overlap"


def test_pattern_never_adds_unasked_copies_or_drops_originals():
    """Repeating patterns must not add copies when Fill is off, nor push an original off the plate when filling."""
    bed = {"printable_area": ["0x0", "100x0", "100x100", "0x100"]}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        # five 10 mm blocks and a bar too long for the bed: arrange leaves the bar over, adds nothing
        simple_project(src, [box(0, 0, 10, 10), box(0, 0, 160, 2)],
                       [(0, tf(0, 5 + 15 * i, 5)) for i in range(5)] + [(1, tf(0, 0, 50))], bed)
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=2, step=15, res=0.5))
        assert report["placed"] == 5 and report["added"] == 0, report

        # thirty 10 mm blocks and one 50 mm block: fill must keep the big block on the plate
        simple_project(src, [box(0, 0, 10, 10), box(0, 0, 50, 50)],
                       [(0, tf(0, 2 * i, 2 * i)) for i in range(30)] + [(1, tf(0, 20, 20))], bed)
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=2, step=15, res=0.5, fill=True))
        assert report["placed"] == 31 and report["unplaced"] == [], report
        masks = [m for m, lo, hi in projections(dst, 100.0, 0.1) if lo[1] < 100]
        assert len(masks) == 31 + report["added"]
        assert (np.sum(masks, axis=0) > 1).sum() == 0, "parts overlap"


TOWER = {"printable_area": ["0x0", "100x0", "100x100", "0x100"], "enable_prime_tower": "1",
         "wipe_tower_x": ["60"], "wipe_tower_y": ["60"], "prime_tower_width": "30",
         "prime_tower_brim_width": "3", "filament_prime_volume": ["45", "45"], "layer_height": "0.2"}


def filament_settings(*slots):
    objs = "".join(f'<object id="{i + 1}"><metadata key="name" value="o{i + 1}"/>'
                   f'<metadata key="extruder" value="{e}"/></object>' for i, e in enumerate(slots))
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<config>{objs}</config>'


def tower_overlap(path: Path) -> int:
    """Cells (0.25 mm) where any part sits on the reserved tower area incl. brim."""
    res, x0, x1 = 0.25, 60 - 3, 60 + 30 + 3
    hits = 0
    for m, lo, hi in projections(path, 100.0, res):
        if lo[1] >= 100:  # overflow row
            continue
        hits += int(m[int(x0 / res):int(x1 / res), int(x0 / res):int(x1 / res)].sum())
    return hits


def test_prime_tower_is_kept_free_with_two_filaments():
    blocks = [(0, tf(0, 5 + 12 * (i % 8), 5 + 12 * (i // 8))) for i in range(12)] + \
             [(1, tf(0, 5 + 12 * (i % 8), 50 + 12 * (i // 8))) for i in range(12)]
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 10, 10), box(0, 0, 10, 10)], blocks, TOWER, filament_settings(1, 2))
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=1, step=90, res=0.5, fill=True))
        assert report["prime_tower"], report
        assert tower_overlap(dst) == 0, "a part was placed on the prime tower"


def test_single_filament_plate_has_no_tower():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 10, 10)], [(0, tf(0, 5, 5))], TOWER, filament_settings(1))
        report = sa.arrange_3mf(src, dst, sa.Params(spacing=1, step=90, res=0.5, fill=True))
        assert not report["prime_tower"], report
        assert tower_overlap(dst) > 0, "with one filament the tower area should be usable"


def test_painted_part_needs_a_tower():
    star_mesh = box(0, 0, 10, 10)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "in.3mf")
        simple_project(src, [star_mesh], [(0, tf(0, 5, 5))], TOWER, filament_settings(1))
        with zipfile.ZipFile(src) as z:
            files = {n: z.read(n) for n in z.namelist()}
        files["3D/3dmodel.model"] = files["3D/3dmodel.model"].replace(b'<triangle ', b'<triangle paint_color="0C" ', 1)
        with zipfile.ZipFile(src, "w") as z:
            for n, b in files.items():
                z.writestr(n, b)
        pkg = sa.Package(src)
        assert pkg.prime_tower(1, pkg.items()) is not None


def test_missing_plate_is_an_error():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "in.3mf")
        make_project(src)
        try:
            sa.arrange_3mf(src, Path(d, "out.3mf"), sa.Params(plate=3))
        except ValueError as e:
            assert "plate 3" in str(e)
        else:
            raise AssertionError("expected ValueError")


def expect_error(fn, *needles, exc=ValueError):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return e
    raise AssertionError(f"expected {exc.__name__} mentioning {needles}")


def rewrite_entry(path: Path, name: str, edit):
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files[name] = edit(files[name])
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)


def upstream_tower_corners(x, y, width, depth, brim, angle_deg):
    """Orca's first_layer_wipe_tower_corners: local corners (brim included), rotate about the
    tower's local origin, then translate to (x, y). Written out by hand, independent of the plugin."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    local = [(-brim, -brim), (width + brim, -brim), (width + brim, depth + brim), (-brim, depth + brim)]
    return [(px * c - py * s + x, px * s + py * c + y) for px, py in local]


def polygon_mask(poly, bed, res):
    img = Image.new("1", (int(bed / res), int(bed / res)), 0)
    ImageDraw.Draw(img).polygon([(px / res, py / res) for px, py in poly], fill=1)
    return np.array(img, dtype=bool)


def test_rotated_prime_tower_matches_orca_corners():
    """C1: the reserve rotates about the tower's local origin (its x/y), like Orca, at 0, 90 and an oblique angle;
    and Fill never puts a part on that independently computed area."""
    for angle, x, y in ((0, 60, 60), (90, 60, 60), (30, 40, 20)):
        cfg = {**TOWER, "wipe_tower_x": [str(x)], "wipe_tower_y": [str(y)], "wipe_tower_rotation_angle": str(angle)}
        expected = upstream_tower_corners(x, y, 30, 30, 3, angle)
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
            simple_project(src, [box(0, 0, 10, 10), box(0, 0, 10, 10)], [(0, tf(0, 0, 0)), (1, tf(0, 20, 0))],
                           cfg, filament_settings(1, 2))
            pkg = sa.Package(src)
            tris = pkg.prime_tower(1, pkg.items())
            pkg.close()
            got = {tuple(np.round(p, 6)) for p in tris.reshape(-1, 2).tolist()}
            want = {tuple(np.round(p, 6)) for p in expected}
            assert got == want, (angle, sorted(got), sorted(want))

            report = sa.arrange_3mf(src, dst, sa.Params(spacing=1, step=90, res=0.5, fill=True))
            assert report["prime_tower"] and report["added"] > 0, report
            res = 0.25
            reserve = polygon_mask(expected, 100.0, res)
            overlap = sum(int((m & reserve).sum()) for m, lo, hi in projections(dst, 100.0, res) if lo[1] < 100)
            assert overlap == 0, f"{overlap} cells of parts on the {angle} deg tower"


def test_fill_takes_turns_between_models_on_the_pattern_path():
    """C4: two models with equal geometry but distinct identities get equal shares of the copies,
    also when the repeating-pattern (lattice) layout wins."""
    cfg = {"printable_area": ["0x0", "60x0", "60x60", "0x60"]}
    used = []
    real = sa._lattice_layout

    def spy(*a, **k):
        out = real(*a, **k)
        used.append(out)
        return out

    sa._lattice_layout = spy
    try:
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
            simple_project(src, [box(0, 0, 10, 10), box(0, 0, 10, 10)], [(0, tf(0, 0, 0)), (1, tf(0, 20, 0))], cfg)
            report = sa.arrange_3mf(src, dst, sa.Params(step=90, spacing=1, res=0.5, fill=True))
            out = sa.Package(dst)
            counts = {}
            for it in out.items():
                counts[it.object_id] = counts.get(it.object_id, 0) + 1
            out.close()
            assert used and used[-1] is not None and len(used[-1][2]) == report["added"], "pattern path not taken"
            assert report["placed"] == 2 and report["unplaced"] == [], report
            assert report["added"] >= 10 and abs(counts["1"] - counts["2"]) <= 1, (counts, report)
            assert sum(report["added_by_object"].values()) == report["added"], report
            masks = [m for m, lo, hi in projections(dst, 60.0, 0.1) if lo[1] < 60]
            assert len(masks) == 2 + report["added"]
            assert (np.sum(masks, axis=0) > 1).sum() == 0, "copies overlap"
    finally:
        sa._lattice_layout = real


def test_non_millimetre_unit_is_rejected_before_writing():
    """C5: unit="inch" (main model or a referenced model file) fails with a clear error and no output."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 1, 1)], [(0, tf(0, 0, 0))], {})
        rewrite_entry(src, "3D/3dmodel.model", lambda b: b.replace(b'unit="millimeter"', b'unit="inch"'))
        bed = [(0, 0), (60, 0), (60, 60), (0, 60)]
        expect_error(lambda: sa.arrange_3mf(src, dst, sa.Params(step=0), bed=bed), "inch", "millimeter")
        assert not dst.exists()

        make_project(src)
        rewrite_entry(src, "3D/Objects/object_5.model", lambda b: b.replace(b'unit="millimeter"', b'unit="micron"'))
        expect_error(lambda: sa.arrange_3mf(src, dst, sa.Params(plate=2, step=90)), "micron")
        assert not dst.exists()


CORE_URI = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def test_prefixed_build_section_is_arranged_and_filled():
    """C6: <m:build>/<m:item> bound to the core namespace is edited in place; other bytes are kept."""
    cfg = {"printable_area": ["0x0", "60x0", "60x60", "0x60"]}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 10, 10)], [(0, tf(0, 70, 70)), (0, tf(0, 80, 80))], cfg)
        rewrite_entry(src, "3D/3dmodel.model", lambda b: b
                      .replace(b"<build>", f'<m:build xmlns:m="{CORE_URI}">'.encode())
                      .replace(b"</build>", b"</m:build>")
                      .replace(b"<item objectid=", b"<m:item objectid = "))
        before = sa.Package(src).main_text
        report = sa.arrange_3mf(src, dst, sa.Params(step=90, spacing=1, res=0.5, fill=True))
        assert report["placed"] == 2 and report["added"] > 0, report
        out = sa.Package(dst)
        text = out.main_text
        assert text[:text.index("<m:build")] == before[:before.index("<m:build")], "bytes before build changed"
        root = __import__("xml.etree.ElementTree", fromlist=["x"]).fromstring(text)
        items = root.find(f"{{{CORE_URI}}}build").findall(f"{{{CORE_URI}}}item")
        assert len(items) == 2 + report["added"] == len(out.items())
        for it in out.items():
            xy = np.concatenate([v[:, :2] for v, _ in it.parts])
            assert xy.min() >= -1e-6 and xy.max() <= 60 + 1e-6, xy
        out.close()


def test_ambiguous_prefixed_build_is_refused():
    """C6: when the text scan and the XML parser disagree about the build items, refuse instead of guessing."""
    cfg = {"printable_area": ["0x0", "60x0", "60x60", "0x60"]}
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 10, 10)], [(0, tf(0, 5, 5))], cfg)
        rewrite_entry(src, "3D/3dmodel.model", lambda b: b
                      .replace(b"<build>", f'<m:build xmlns:m="{CORE_URI}">'.encode()).replace(b"</build>", b"</m:build>")
                      .replace(b"<item ", b'<m:item xmlns:m="urn:vendor" objectid="1"/><m:item ', 1))
        expect_error(lambda: sa.arrange_3mf(src, dst, sa.Params(step=90)), "build")
        assert not dst.exists()


def test_refine_keeps_rotation_disabled():
    """C7: step=0 means no rotation, also in the finer retry. A 16 x 5 part on a 10 x 20 bed stays unrotated."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        simple_project(src, [box(0, 0, 16, 5)], [(0, tf(0, 0, 0))], {"printable_area": ["0x0", "10x0", "10x20", "0x20"]})
        report = sa.arrange_3mf(src, dst, sa.Params(step=0, spacing=0, refine=True))
        out = sa.Package(dst)
        angle = sa._z_angle(out.items()[0].transform)
        out.close()
        assert abs(angle) < 1e-9, angle
        assert report["unplaced"], report


def test_invalid_parameters_and_config_are_rejected_early():
    """C8: bad numbers fail as ValueError before any rasterising; config booleans are normalised."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "in.3mf")
        simple_project(src, [box(0, 0, 10, 10)], [(0, tf(0, 0, 0))], {"printable_area": ["0x0", "60x0", "60x60", "0x60"]})
        for bad, needle in ((dict(res=0), "res"), (dict(res=-1), "res"), (dict(res=float("nan")), "res"),
                            (dict(spacing=float("inf")), "spacing"), (dict(spacing=-1), "spacing"),
                            (dict(step=float("nan")), "step"), (dict(plate=0), "plate"), (dict(plate=True), "plate"),
                            (dict(fill="yes"), "fill"), (dict(refine=1), "refine")):
            expect_error(lambda: sa.plan_3mf(src, sa.Params(**bad)), needle)
        for bed in ([(0, 0), (0, 0), (0, 0)], [(0, 0), (float("inf"), 0), (10, 10)], [(0, 0), (10, 0)]):
            plain = Path(d, "plain.3mf")
            simple_project(plain, [box(0, 0, 10, 10)], [(0, tf(0, 0, 0))], {})
            expect_error(lambda: sa.plan_3mf(plain, sa.Params(step=0), bed=bed), "bed")

        for value, on in (("1", True), ("true", True), (True, True), (1, True), (["1"], True),
                          ("0", False), ("false", False), (False, False), (0, False), (None, False)):
            simple_project(src, [box(0, 0, 10, 10)] * 2, [(0, tf(0, 0, 0)), (1, tf(0, 20, 0))],
                           {**TOWER, "enable_prime_tower": value}, filament_settings(1, 2))
            pkg = sa.Package(src)
            assert (pkg.prime_tower(1, pkg.items()) is not None) == on, value
            pkg.close()
        simple_project(src, [box(0, 0, 10, 10)] * 2, [(0, tf(0, 0, 0)), (1, tf(0, 20, 0))],
                       {**TOWER, "enable_prime_tower": "maybe"}, filament_settings(1, 2))
        pkg = sa.Package(src)
        expect_error(lambda: pkg.prime_tower(1, pkg.items()), "enable_prime_tower")
        pkg.close()


def test_write_failure_keeps_earlier_output_and_leaves_no_partial_file():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        make_project(src)
        dst.write_bytes(b"earlier result")
        pkg, (new, copies), _ = sa.plan_3mf(src, sa.Params(plate=2, spacing=1, step=90, res=0.25))
        real = zipfile.ZipFile.writestr
        calls = []

        def boom(self, info, data, *a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return real(self, info, data, *a, **k)

        zipfile.ZipFile.writestr = boom
        try:
            expect_error(lambda: pkg.write(dst, new, copies, 2), "disk full", exc=OSError)
        finally:
            zipfile.ZipFile.writestr = real
        assert dst.read_bytes() == b"earlier result"
        assert sorted(p.name for p in Path(d).iterdir()) == ["in.3mf", "out.3mf"], list(Path(d).iterdir())
        pkg2, (new, copies), _ = sa.plan_3mf(src, sa.Params(plate=2, spacing=1, step=90, res=0.25))
        expect_error(lambda: pkg2.write(dst, new, copies, 2, overwrite=False), "exists", exc=FileExistsError)
        assert dst.read_bytes() == b"earlier result"


if __name__ == "__main__":
    test_rotated_prime_tower_matches_orca_corners()
    test_fill_takes_turns_between_models_on_the_pattern_path()
    test_non_millimetre_unit_is_rejected_before_writing()
    test_prefixed_build_section_is_arranged_and_filled()
    test_ambiguous_prefixed_build_is_refused()
    test_refine_keeps_rotation_disabled()
    test_invalid_parameters_and_config_are_rejected_early()
    test_write_failure_keeps_earlier_output_and_leaves_no_partial_file()
    test_interlocking_combs_fit_on_plate_2()
    test_crowded_plate_spacing_zero_with_obstacle()
    test_auto_mode_refines_to_fit_tight_interlock()
    test_overhangs_block_space_at_every_height()
    test_by_object_sequence_keeps_extruder_clearance()
    test_fill_plate_adds_registered_copies_without_overlap()
    test_identical_brackets_tile_in_a_repeating_pattern()
    test_pattern_never_adds_unasked_copies_or_drops_originals()
    test_prime_tower_is_kept_free_with_two_filaments()
    test_single_filament_plate_has_no_tower()
    test_painted_part_needs_a_tower()
    test_missing_plate_is_an_error()
    print("ok")
