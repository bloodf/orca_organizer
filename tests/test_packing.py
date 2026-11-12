"""Packing geometry checked with exact polygons (no raster in the checks).

Synthetic convex pieces in mm are nested with the plugin's own core (`nest` on a `_bed_grid`), the
resulting transforms are applied to the exact pieces, and distances are measured analytically:
no overlaps, every gap >= the spacing, nothing closer than the spacing to the bed edge or to an
obstacle. Only numpy and Pillow, like the plugin.

Run: uv run --python 3.12 --with numpy --with pillow python tests/test_packing.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shape_arrange as sa  # noqa: E402

BED = 256.0
EPS = 1e-6


# ---------------------------------------------------------------- fixtures (convex pieces, mm)

def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ngon(r, n, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def ring(r_out, r_in, n=64):
    """Washer as n convex quads: its hole is real free space in the footprint."""
    out = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        out.append([(r_in * math.cos(a0), r_in * math.sin(a0)), (r_out * math.cos(a0), r_out * math.sin(a0)),
                    (r_out * math.cos(a1), r_out * math.sin(a1)), (r_in * math.cos(a1), r_in * math.sin(a1))])
    return out


SHAPES = {
    "sq20": [rect(0, 0, 20, 20)],
    "bar60x15": [rect(0, 0, 60, 15)],
    "L40": [rect(0, 0, 40, 12), rect(0, 12, 12, 40)],
    "tri40": [[(0, 0), (40, 0), (0, 40)]],
    "sliver": [[(0, 0), (30, 0), (30, 0.2)]],           # thinner than one fine pixel
    "washer50": ring(25, 17),
    "disc12": [ngon(6, 48)],
}


def prism(piece, h=2.0):
    n = len(piece)
    v = [(x, y, 0.0) for x, y in piece] + [(x, y, h) for x, y in piece]
    t = []
    for i in range(1, n - 1):
        t += [(0, i + 1, i), (n, n + i, n + i + 1)]
    for i in range(n):
        j = (i + 1) % n
        t += [(i, j, n + j), (i, n + j, n + i)]
    return np.array(v, float), np.array(t, int)


def make_item(index, name, at=(60.0, 60.0), angle=0.0):
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    tr = np.eye(4)
    tr[:2, :2], tr[:2, 3] = rot, at
    parts, pieces = [], []
    for piece in SHAPES[name]:
        v, t = prism(piece)
        v[:, :2] = v[:, :2] @ rot.T + np.array(at)
        parts.append((v, t))
        pieces.append(v[:len(piece), :2].copy())
    return sa.Item(index=index, object_id=name, transform=tr, parts=parts, name=f"{name}#{index}"), pieces


def make_items(spec):
    items, exact = [], {}
    for name, count, *angle in spec:
        for _ in range(count):
            it, pieces = make_item(len(items), name, angle=angle[0] if angle else 0.0)
            items.append(it)
            exact[it.index] = pieces
    return items, exact


def run_nest(spec, p, obstacles=(), bed=BED):
    """nest() as _plan runs it, returning exact world pieces: {index: [piece arrays]}, copies, failed, angles."""
    items, exact = make_items(spec)
    poly = [(0.0, 0.0), (bed, 0.0), (bed, bed), (0.0, bed)]
    tris = [np.array([[q[0], q[i], q[i + 1]] for i in range(1, len(q) - 1)], float) for q in obstacles]
    origin, blocked = sa._bed_grid(poly, [], p.res, tris)
    placed, failed, copies = sa.nest(items, blocked, p)

    def move(pieces, s, x, y, center):
        d = sa._delta(s.angle, center, origin + np.array([x + s.cx, y + s.cy]) * p.res)
        return [q @ d[:2, :2].T + d[:2, 3] for q in pieces]
    out = {idx: move(exact[idx], *v) for idx, v in placed.items()}
    added = [move(exact[it.index], s, x, y, c) for it, s, x, y, c in copies]
    angles = [v[0].angle for v in placed.values()] + [s.angle for _, s, _, _, _ in copies]
    return out, added, failed, angles


# ---------------------------------------------------------------- exact distances (convex pieces)

def _seg_dist(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-18), 0.0, 1.0)
    return np.hypot(*(pts - (a + t[:, None] * ab)).T)


def _overlap(P, Q):
    """Separating axis test; True when the interiors intersect (touching is not overlap)."""
    for poly in (P, Q):
        e = np.roll(poly, -1, axis=0) - poly
        for nx, ny in np.stack([-e[:, 1], e[:, 0]], axis=1):
            n = np.array([nx, ny]) / (math.hypot(nx, ny) or 1.0)
            p, q = P @ n, Q @ n
            if p.max() <= q.min() + EPS or q.max() <= p.min() + EPS:
                return False
    return True


def gap(P, Q):
    """Exact distance between two convex polygons; -1 if they overlap."""
    if _overlap(P, Q):
        return -1.0
    d = math.inf
    for A, B in ((P, Q), (Q, P)):
        for i in range(len(B)):
            d = min(d, float(_seg_dist(A, B[i], B[(i + 1) % len(B)]).min()))
    return d


def part_gap(a, b):
    return min(gap(p, q) for p in a for q in b)


def check_layout(parts, spacing, obstacles=(), bed=BED):
    """Asserts the rules and returns (min gap between parts, min gap to bed edges)."""
    min_pair, min_edge = math.inf, math.inf
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            g = part_gap(parts[i], parts[j])
            assert g >= spacing - EPS, f"parts {i} and {j}: gap {g:.4f} < spacing {spacing}"
            min_pair = min(min_pair, g)
        pts = np.concatenate(parts[i])
        edge = float(min(pts.min(), bed - pts.max()))
        assert edge >= spacing - EPS, f"part {i}: {edge:.4f} from the bed edge, spacing {spacing}"
        min_edge = min(min_edge, edge)
        for k, ob in enumerate(obstacles):
            g = part_gap(parts[i], [np.array(ob, float)])
            assert g >= spacing - EPS, f"part {i}: gap {g:.4f} to obstacle {k} < spacing {spacing}"
    return min_pair, min_edge


# ---------------------------------------------------------------- tests

def test_footprint_covers_the_exact_rotated_part():
    """Every point of the exact rotated part lies in a set cell of its footprint (raw) at every angle,
    also for a sliver thinner than one supersampled pixel."""
    rng = np.random.default_rng(20261112)
    for name in ("tri40", "L40", "sliver"):
        for res in (0.5, 0.25):
            item, pieces = make_item(0, name, at=(33.3, 71.7), angle=13.0)
            p = sa.Params(spacing=0.0, step=7.0, res=res)
            center, shapes = sa._shapes(item, p)
            pts = []
            for q in pieces:  # dense samples: vertices, edges and interior (barycentric)
                q = np.asarray(q)
                w = rng.dirichlet(np.ones(len(q)), 4000)
                pts += [q, w @ q]
                for i in range(len(q)):
                    t = np.linspace(0, 1, 200)[:, None]
                    pts.append(q[i] + t * (q[(i + 1) % len(q)] - q[i]))
            pts = np.concatenate(pts) - center
            for s in shapes:
                a = math.radians(s.angle)
                rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                xy = pts @ rot.T / res
                col = np.floor(xy[:, 0] + s.cx - s.pad).astype(int)
                row = np.floor(xy[:, 1] + s.cy - s.pad).astype(int)
                h, w = s.raw.shape
                assert col.min() >= 0 and row.min() >= 0 and col.max() < w and row.max() < h, (name, res, s.angle)
                missed = int((~s.raw[row, col]).sum())
                assert missed == 0, f"{name} res {res} angle {s.angle}: {missed} points outside the footprint"


def test_fft_dilation_is_identical_to_the_shift_loop():
    rng = np.random.default_rng(5)
    for trial in range(6):
        mask = rng.random((37 + 11 * trial, 53 - 5 * trial)) < (0.02 + 0.1 * trial)
        for se in (sa._disc(13), sa._disc(31), sa._gap_offsets(17.5), sa._gap_offsets(40.0)):
            assert se.sum() > sa.DILATE_FFT_CELLS
            fast = sa._dilate_se(mask, se)
            slow = np.zeros_like(fast)
            h, w = mask.shape
            for dy, dx in zip(*np.nonzero(se)):
                slow[dy:dy + h, dx:dx + w] |= mask
            assert fast.dtype == bool and np.array_equal(fast, slow), (trial, se.shape)
    assert np.array_equal(sa._dilate(np.ones((1, 1), bool), 3), sa._disc(3))


def test_gap_offsets_are_exact_for_unit_cells():
    """An offset is forbidden exactly when two unit squares at that offset are closer than R."""
    for R in (0.0, 0.3, 1.0, 2.5, 4.0, 5.3):
        se = sa._gap_offsets(R)
        p = se.shape[0] // 2
        for dy in range(-p - 1, p + 2):
            for dx in range(-p - 1, p + 2):
                a = np.array(rect(0, 0, 1, 1), float)
                b = a + (dx, dy)
                d = max(gap(a, b), 0.0) if (dx, dy) != (0, 0) else 0.0
                inside = abs(dy) <= p and abs(dx) <= p and se[dy + p, dx + p]
                assert inside == ((dx, dy) == (0, 0) or d < R - EPS), (R, dy, dx, d)


def test_exact_spacing_is_kept_with_less_excess():
    """Rotation off: the requested gap is never undercut. On the 0.5 mm grid the derived margin
    wastes at most 1 mm on axis-aligned squares (stacked margins: 1.5 mm) and 1.25 mm at the median
    over these cases (stacked margins: about 1.8 mm)."""
    excess, square = [], []
    for res in (0.5, 0.25):
        for spacing in (0.0, 1.0, 2.0, 6.0):
            for name, angle in (("sq20", 0.0), ("tri40", 45.0), ("L40", 61.3)):
                p = sa.Params(spacing=spacing, step=0.0, res=res, center=False)
                parts, _, failed, angles = run_nest([(name, 4, angle)], p)
                assert not failed and len(parts) == 4, (name, res, spacing)
                assert all(a == 0.0 for a in angles), "rotation off must not rotate"
                pair, _ = check_layout(list(parts.values()), spacing)
                if res == 0.5:
                    excess.append(pair - spacing)
                    if name == "sq20":
                        square.append(pair - spacing)
    assert max(square) <= 1.0 + EPS, f"axis-aligned excess {max(square):.3f} mm"
    assert float(np.median(excess)) <= 1.25, f"median excess {np.median(excess):.3f} mm"


def test_narrow_corridor_at_an_odd_angle_fits():
    """A 60 x 15 mm bar fits only in a 22.5 mm corridor at about 37 degrees. The coarse screen finds
    nothing there; the full-precision fallback must try every angle, not 1 in 15 (0.8.0 left the
    bar over here, and with a 15-degree fallback it is left over even in a 24 mm corridor)."""
    a = math.radians(37.0)
    u, v, c = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]), np.array([128.0, 128.0])
    walls, big = [], 400.0
    for n, t, d in ((u, v, 36.0), (-u, v, 36.0), (v, u, 11.25), (-v, u, 11.25)):
        p0 = c + n * d
        walls.append([tuple(p0 - t * big), tuple(p0 + t * big), tuple(p0 + t * big + n * big), tuple(p0 - t * big + n * big)])
    p = sa.Params(spacing=2.0, **sa.AUTO)
    parts, _, failed, angles = run_nest([("bar60x15", 1)], p, walls)
    assert not failed and len(parts) == 1, "bar left over although it fits"
    assert abs(math.remainder(angles[0] - 37.0, 180.0)) <= 2.0, angles
    check_layout(list(parts.values()), 2.0, walls)


def test_small_parts_still_go_into_rings():
    """4 washers (hole 34 mm) and 8 discs of 12 mm: all 8 discs go into the holes, two per washer,
    as in 0.8.0. The tighter margin alone frees spots between the washers that would win the plain
    tie-break; the hole preference keeps the discs in the rings."""
    p = sa.Params(spacing=2.0, **sa.AUTO)
    parts, _, failed, _ = run_nest([("washer50", 4), ("disc12", 8)], p)
    assert not failed and len(parts) == 12
    check_layout(list(parts.values()), 2.0)
    centers = {i: np.concatenate(q).mean(axis=0) for i, q in parts.items()}
    rings = [centers[i] for i in range(4)]
    inside = [i for i in range(4, 12) if min(np.hypot(*(centers[i] - c)) for c in rings) + 6.0 <= 17.0 + EPS]
    assert len(inside) == 8, f"only {len(inside)} of 8 discs inside the rings"


def test_half_turn_pairs_fill_more_triangles():
    """Fill with right triangles: pairs (one turned half a turn) tile far better than one
    orientation. 0.8.0 fit 35 (1 + 34 copies); at least 50 must fit now, all at the spacing."""
    p = sa.Params(spacing=2.0, fill=True, **sa.AUTO)
    parts, copies, failed, _ = run_nest([("tri40", 1)], p)
    assert not failed and len(parts) + len(copies) >= 50, len(parts) + len(copies)
    check_layout(list(parts.values()) + copies, 2.0)


def test_fill_with_rotation_off_never_turns_parts():
    p = sa.Params(spacing=2.0, fill=True, **dict(sa.AUTO, step=0.0))
    parts, copies, failed, angles = run_nest([("tri40", 1)], p)
    assert not failed and len(copies) >= 30, len(copies)
    assert all(a == 0.0 for a in angles), "rotation off: no half-turn twins"
    check_layout(list(parts.values()) + copies, 2.0)


def test_screen_falls_back_to_every_angle():
    item, _ = make_item(0, "bar60x15")
    _, shapes = sa._shapes(item, sa.Params(spacing=2.0, step=1.0, res=0.5))
    occ = np.ones((80, 80))  # nothing fits on the coarse grid (nor anywhere)
    assert sa._screen(shapes, occ, None) == shapes
    free = np.zeros((300, 300))
    assert len(sa._screen(shapes, free, None)) <= sa.SCREEN_KEEP + 1


def test_long_loops_reach_cancellation_checkpoints():
    """Measuring 360 outlines and screening 360 angles must reach the progress callback (the
    cancellation point) between its own messages, and an exception there aborts the run."""
    class Stop(Exception):
        pass
    old = sa.CHECK_EVERY
    sa.CHECK_EVERY = 0.0
    try:
        calls = []
        items, _ = make_items([("L40", 1)])
        _, blocked = sa._bed_grid([(0, 0), (120, 0), (120, 120), (0, 120)], [], 0.5)
        sa.nest(items, blocked, sa.Params(spacing=2.0, step=1.0, res=0.5), lambda i, n, m: calls.append(m.code))
        assert calls.count("footprint") > 300 and calls.count("place") > 300, (len(calls), set(calls))

        def stop(i, n, m):
            calls.append(m.code)
            if len(calls) > 50:
                raise Stop()
        calls.clear()
        try:
            sa.nest(items, blocked, sa.Params(spacing=2.0, step=1.0, res=0.5), stop)
        except Stop:
            pass
        else:
            raise AssertionError("cancellation did not abort")
        assert len(calls) == 51
    finally:
        sa.CHECK_EVERY = old


if __name__ == "__main__":
    test_gap_offsets_are_exact_for_unit_cells()
    test_fft_dilation_is_identical_to_the_shift_loop()
    test_screen_falls_back_to_every_angle()
    test_footprint_covers_the_exact_rotated_part()
    test_long_loops_reach_cancellation_checkpoints()
    test_exact_spacing_is_kept_with_less_excess()
    test_narrow_corridor_at_an_odd_angle_fits()
    test_small_parts_still_go_into_rings()
    test_half_turn_pairs_fill_more_triangles()
    test_fill_with_rotation_off_never_turns_parts()
    print("ok")
