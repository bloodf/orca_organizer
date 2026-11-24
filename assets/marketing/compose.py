"""Puts the real screenshots into the marketing artwork and exports the listing and README sizes.

    node assets/marketing/render-screens.mjs
    uv run --python 3.12 --with numpy --with pillow python assets/marketing/compose.py

Each generated image in source/ has an empty rounded square. Its interior is found from the pixels (a flood
fill from inside the square, stopped by the frame), the screenshot is scaled to cover it and pasted through
that mask, so the frame and everything outside it stay as generated. Writes asset-manifest.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = HERE / "source"
DATE = "2026-11-24"
ID = "orca_organizer"
LANGS = ("en", "pt-BR")

# (source artwork, screenshot, export path, export size)
JOBS = [(f"{ID}-workshop-tile.png", None, f"assets/orca-cloud/{ID}-tile.png", (512, 512))]
# the listing is English only; pt-BR is exported only where README.pt-BR.md shows it (gallery and README images)
JOBS += [
    (f"{ID}-workshop-cover-en.png", "screen-filled-dark-en.png", f"assets/orca-cloud/{ID}-main-en.png", (1600, 900)),
    (f"{ID}-workshop-main-square-en.png", "screen-form-dark-en-small.png", f"assets/orca-cloud/{ID}-main-square-en.png", (900, 900)),
    (f"{ID}-workshop-social-preview-en.png", "screen-form-dark-en-medium.png", f"assets/social/{ID}-social-preview-en.png", (1280, 640)),
]
for lang in LANGS:
    JOBS += [
        (f"{ID}-workshop-feature-{name}-{lang}.png", f"screen-{state}-dark-{lang}.png", f"assets/orca-cloud/{ID}-feature-{name}-{lang}.png", (1600, 900))
        for name, state in (("01-shape", "outlines"), ("02-fill", "filling"), ("03-spacing", "advanced"), ("04-review", "arranged"))
    ]
    JOBS += [
        (f"{ID}-workshop-hero-{theme}-{lang}.png", f"screen-form-{theme}-{lang}-medium.png", f"assets/readme/{ID}-hero-{theme}-{lang}.png", (1600, 900))
        for theme in ("dark", "light")
    ]
# plain screenshots for the READMEs: (screenshot, export path)
SCREENS = [
    (f"screen-{state}-{theme}-{lang}-wide.png", f"assets/readme/{ID}-screen-{name}-{theme}-{lang}.png")
    for state, name in (("filled", "result"), ("advanced", "advanced")) for theme in ("dark", "light") for lang in LANGS
]


def square_mask(art):
    """Mask of the empty screenshot square's interior and its bounding box."""
    grey = art.convert("L")
    w, h = grey.size
    best = None
    for fx, fy in ((0.74, 0.47), (0.77, 0.56), (0.80, 0.50)):  # inside the square in landscape and 1:1 layouts
        seed = (int(fx * w), int(fy * h))
        light = grey.getpixel(seed) > 200
        flat = grey.point(lambda p: 255 if (p > 225 if light else p < 75) else 0)
        if flat.getpixel(seed) != 255:
            continue
        ImageDraw.floodfill(flat, seed, 128)
        mask = flat.point(lambda p: 255 if p == 128 else 0)
        box = mask.getbbox()
        filled = mask.histogram()[255] / ((box[2] - box[0]) * (box[3] - box[1]))
        if best is None or filled > best[2]:
            best = (mask, box, filled)
    assert best and best[2] > 0.99, "no empty screenshot square found"
    return best[0], best[1]


def composite(art, shot):
    """Returns the artwork with `shot` in its empty square, and the square's box.

    Well inside the square the screenshot replaces the placeholder. In the few pixels where the generated frame
    fades into the placeholder, the placeholder's share of each pixel (read from its brightness between the
    placeholder and the frame beside it) is swapped for the screenshot, so the frame keeps its own soft edge.
    """
    mask, (x0, y0, x1, y1) = square_mask(art)
    bw, bh = x1 - x0, y1 - y0
    scale = max(bw / shot.width, bh / shot.height)
    fitted = shot.resize((max(bw, round(shot.width * scale)), max(bh, round(shot.height * scale))), Image.LANCZOS)
    left = (fitted.width - bw) // 2  # keep the app bar: centred across, anchored to the top
    layer = Image.new("RGB", art.size)
    # the edge band reaches 2 px past the box: repeat the screenshot's edge pixels there
    layer.paste(fitted.crop((left, 0, left + bw, bh)).resize((bw + 4, bh + 4), Image.NEAREST), (x0 - 2, y0 - 2))
    layer.paste(fitted.crop((left, 0, left + bw, bh)), (x0, y0))

    px = np.asarray(art, dtype=np.float32)
    new = np.asarray(layer, dtype=np.float32)
    core = np.asarray(mask.filter(ImageFilter.MinFilter(5))) > 0
    band = (np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0) & ~core
    grey = art.convert("L")
    lum = np.asarray(grey, dtype=np.float32)
    placeholder = np.median(px[core], axis=0)
    base = float(np.median(lum[core]))
    light = base > 128
    # brightness of the frame next to each pixel: the extreme within 3 px
    frame = np.asarray(grey.filter(ImageFilter.MinFilter(7) if light else ImageFilter.MaxFilter(7)), dtype=np.float32)
    span = np.abs(frame - base)
    share = np.clip(1 - np.abs(lum - base) / np.maximum(span, 1), 0, 1)  # how much of the pixel is placeholder
    share[span < 25] = (np.asarray(mask) > 0)[span < 25]  # no frame nearby: inside or outside, nothing between
    out = px.copy()
    out[core] = new[core]
    out[band] = np.clip(px[band] + share[band, None] * (new[band] - placeholder), 0, 255)
    return Image.fromarray(out.round().astype(np.uint8)), (x0, y0, x1, y1)


def describe(path):
    data = path.read_bytes()
    with Image.open(path) as im:
        size = list(im.size)
    return {"size": size, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    exports = []
    for source, screen, target, size in JOBS:
        art = Image.open(SRC / source).convert("RGB")
        operation = []
        if screen:
            art, box = composite(art, Image.open(SRC / "screens" / screen).convert("RGB"))
            operation.append(f"screenshot scaled to cover the empty square {list(box)} and blended under the frame's inner edge")
        ratio = size[0] / size[1]
        if abs(art.width / art.height - ratio) > 0.002:
            width = round(art.height * ratio)
            left = (art.width - width) // 2
            art = art.crop((left, 0, left + width, art.height))
            operation.append(f"centre crop to {art.width}x{art.height}")
        operation.append(f"scale {art.width}x{art.height} to {size[0]}x{size[1]} (Lanczos)")
        path = ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        art.resize(size, Image.LANCZOS).save(path, optimize=True)
        info = describe(path)
        assert info["bytes"] < 2_000_000, f"{target} is {info['bytes']} bytes"
        exports.append({"file": target, "source": f"assets/marketing/source/{source}",
                        "screenshot": f"assets/marketing/source/screens/{screen}" if screen else None,
                        "operation": "; ".join(operation), **info, "date": DATE})
    for screen, target in SCREENS:
        path = ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(SRC / "screens" / screen).convert("RGB").save(path, optimize=True)
        exports.append({"file": target, "source": f"assets/marketing/source/screens/{screen}", "screenshot": None,
                        "operation": "copy (PNG optimised)", **describe(path), "date": DATE})
    used = sorted({e["source"] for e in exports} | {e["screenshot"] for e in exports if e["screenshot"]})
    sources = [{"file": f, **describe(ROOT / f)} for f in used]
    manifest = {
        "note": "Generated marketing artwork (2026-11-16 set, native pixels in source/) with screenshots of the built page "
                "(mock host, made-up data, never OrcaSlicer itself) from render-screens.mjs. Exports are written by compose.py.",
        "exports": exports,
        "sources": sources,
    }
    (HERE / "asset-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for e in exports:
        print(f"{e['file']}  {e['size'][0]}x{e['size'][1]}  {e['bytes']}")
    print(len(exports), "exports, largest", max(e["bytes"] for e in exports), "bytes")


if __name__ == "__main__":
    main()
