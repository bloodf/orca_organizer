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
    """Plugin presets: 0.5 mm can't interlock the combs, the automatic 0.25 mm retry can."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.3mf"), Path(d, "out.3mf")
        make_project(src)
        coarse = sa.arrange_3mf(src, dst, sa.Params(plate=2, spacing=1.0, step=15, res=0.5))
        auto = sa.arrange_3mf(src, dst, sa.Params(plate=2, spacing=1.0, **sa.AUTO))
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


if __name__ == "__main__":
    test_interlocking_combs_fit_on_plate_2()
    test_crowded_plate_spacing_zero_with_obstacle()
    test_auto_mode_refines_to_fit_tight_interlock()
    test_overhangs_block_space_at_every_height()
    test_by_object_sequence_keeps_extruder_clearance()
    test_fill_plate_adds_registered_copies_without_overlap()
    test_prime_tower_is_kept_free_with_two_filaments()
    test_single_filament_plate_has_no_tower()
    test_painted_part_needs_a_tower()
    test_missing_plate_is_an_error()
    print("ok")
