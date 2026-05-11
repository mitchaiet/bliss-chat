#!/usr/bin/env python3
"""
Render the Bliss Chat toolbar icon strip — five 24x24 icons in XP-Luna
style, packed left-to-right into one BMP with magenta (255,0,255) used as
the COMCTL32 transparency key.

Order MUST match the TBBUTTON array in xpchat.c:
    0 New Chat  1 Save  2 Stop  3 Settings  4 Help

Run:  python3 assets/make_toolbar_icons.py
Out:  assets/toolbar_icons.bmp
"""
from PIL import Image, ImageDraw
from pathlib import Path
import math

OUT = Path(__file__).resolve().parent / "toolbar_icons.bmp"
SIZE = 24
N = 5
MAGENTA = (255, 0, 255)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient_fill(img, box, top_rgb, bot_rgb):
    """Vertical gradient inside `box` = (x0, y0, x1, y1)."""
    px = img.load()
    x0, y0, x1, y1 = box
    h = max(1, y1 - y0)
    for y in range(y0, y1):
        t = (y - y0) / (h - 1) if h > 1 else 0
        col = lerp(top_rgb, bot_rgb, t)
        for x in range(x0, x1):
            if 0 <= x < img.width and 0 <= y < img.height:
                px[x, y] = (*col, 255)


def new_chat(img: Image.Image):
    """White page with the corner folded + green '+' badge."""
    d = ImageDraw.Draw(img)
    # Page body
    d.rectangle([5, 3, 18, 21], fill=(255, 255, 255), outline=(110, 130, 170))
    # Folded corner
    d.polygon([(15, 3), (18, 3), (18, 6)], fill=(230, 235, 245), outline=(110, 130, 170))
    # Page lines
    for y in (10, 13, 16):
        d.line([(7, y), (16, y)], fill=(180, 195, 215))
    # Green "+" badge in lower-right
    cx, cy, r = 18, 18, 5
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 190, 80), outline=(70, 130, 50))
    d.line([(cx - 2, cy), (cx + 2, cy)], fill=(255, 255, 255), width=2)
    d.line([(cx, cy - 2), (cx, cy + 2)], fill=(255, 255, 255), width=2)


def save(img: Image.Image):
    """Floppy disk in XP blue."""
    d = ImageDraw.Draw(img)
    # Body
    gradient_fill(img, (3, 4, 21, 21), (90, 140, 210), (50, 95, 165))
    d.rectangle([3, 4, 20, 20], outline=(30, 60, 110))
    # Metal shutter (top)
    d.rectangle([7, 4, 17, 11], fill=(210, 215, 225), outline=(120, 130, 145))
    d.rectangle([9, 6, 12, 10], fill=(70, 90, 120))   # slot
    # Label (bottom)
    d.rectangle([6, 13, 18, 19], fill=(245, 245, 235), outline=(150, 150, 130))
    for y in (15, 17):
        d.line([(8, y), (16, y)], fill=(200, 200, 170))


def stop(img: Image.Image):
    """Red octagon with white inner square — readable at 24px."""
    d = ImageDraw.Draw(img)
    # Octagon points
    cx, cy = 12, 12
    r = 9
    pts = []
    for i in range(8):
        a = math.pi / 8 + i * (math.pi / 4)
        pts.append((cx + int(r * math.cos(a)), cy + int(r * math.sin(a))))
    d.polygon(pts, fill=(210, 50, 50), outline=(140, 20, 20))
    # White inner square gives "stop" feel
    d.rectangle([8, 8, 16, 16], fill=(255, 255, 255))


def settings(img: Image.Image):
    """Gear — central hub + six teeth + hole."""
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12
    # Teeth (rectangles rotated around center)
    teeth_color = (110, 110, 110)
    teeth_outline = (60, 60, 60)
    for i in range(6):
        a = i * math.pi / 3
        # rectangle 4x8 centered at (cx + 9*cos, cy + 9*sin)
        tx, ty = cx + 8 * math.cos(a), cy + 8 * math.sin(a)
        # Build a rotated rect by drawing a polygon
        wx, wy = math.cos(a) * 3, math.sin(a) * 3
        hx, hy = -math.sin(a) * 2, math.cos(a) * 2
        poly = [
            (tx + wx + hx, ty + wy + hy),
            (tx - wx + hx, ty - wy + hy),
            (tx - wx - hx, ty - wy - hy),
            (tx + wx - hx, ty + wy - hy),
        ]
        poly = [(int(round(x)), int(round(y))) for x, y in poly]
        d.polygon(poly, fill=teeth_color, outline=teeth_outline)
    # Hub
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(160, 160, 160), outline=teeth_outline)
    # Hole
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(60, 60, 60))


def help_icon(img: Image.Image):
    """Blue circle with white '?'."""
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12
    gradient_fill(img, (3, 3, 21, 21), (100, 170, 230), (50, 110, 190))
    d.ellipse([3, 3, 20, 20], outline=(20, 50, 110))
    # Question mark — drawn as primitive shapes (no font dependency for the
    # toolbar bitmap; this stays sharp at 24px).
    # Curve
    d.arc([cx - 4, cy - 7, cx + 4, cy + 1], start=200, end=340, fill=(255, 255, 255), width=2)
    # Stem
    d.line([(cx, cy - 1), (cx, cy + 3)], fill=(255, 255, 255), width=2)
    # Dot
    d.rectangle([cx - 1, cy + 5, cx + 1, cy + 7], fill=(255, 255, 255))


RENDERERS = [new_chat, save, stop, settings, help_icon]


def main():
    strip = Image.new("RGB", (SIZE * N, SIZE), MAGENTA)
    for i, fn in enumerate(RENDERERS):
        cell = Image.new("RGBA", (SIZE, SIZE), (*MAGENTA, 255))
        fn(cell)
        # Composite onto RGB (drop alpha)
        rgb = Image.new("RGB", (SIZE, SIZE), MAGENTA)
        rgb.paste(cell, (0, 0), cell.split()[3] if cell.mode == "RGBA" else None)
        strip.paste(rgb, (i * SIZE, 0))
    strip.save(OUT, "BMP")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {SIZE*N}x{SIZE})")


if __name__ == "__main__":
    main()
