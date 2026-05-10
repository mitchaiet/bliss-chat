#!/usr/bin/env python3
"""
Generate the bliss-chat application icon.

Renders a stylized Windows XP "Bliss" wallpaper (rolling green hills + blue
sky) with a small white chat-bubble overlaid in the upper-right. The result
is written as a multi-resolution Windows .ico file (16, 24, 32, 48, 64,
128, 256 px).

Run:  python3 assets/make_icon.py
Out:  assets/bliss_chat.ico
"""

from PIL import Image, ImageDraw, ImageFilter
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "bliss_chat.ico"
SIZES = [16, 24, 32, 48, 64, 128, 256]


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render(size: int) -> Image.Image:
    """Render a single-resolution icon. Bliss-style hill + sky + chat bubble."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()

    # Bliss-ish sky values, eyeballed off the iconic XP wallpaper.
    sky_top    = (110, 178, 230)   # pale cyan-blue
    sky_mid    = (165, 207, 240)   # near horizon
    sky_bottom = (210, 232, 248)   # whitish where hills meet sky

    def hill_y(x, phase, amp, base):
        return base - amp * (0.5 + 0.5 * math.cos(math.pi * (x + phase)))

    for y in range(size):
        for x in range(size):
            u = x / max(1, size - 1)
            v = y / max(1, size - 1)

            # sky gradient (top to where hills start ~55%)
            if v < 0.55:
                t = v / 0.55
                col = lerp(sky_top, sky_mid, t)
            else:
                t = (v - 0.55) / 0.45
                col = lerp(sky_mid, sky_bottom, t * 0.6)

            # back hill — darker, lower amplitude
            back_y = hill_y(u, 0.3, 0.18, 0.78)
            if v > back_y:
                d = (v - back_y) / max(0.01, 1 - back_y)
                back_top   = (115, 175,  80)
                back_floor = (78,  138,  56)
                col = lerp(back_top, back_floor, min(1.0, d * 1.6))

            # front hill — brighter, larger amplitude
            front_y = hill_y(u, 0.0, 0.30, 0.72)
            if v > front_y:
                d = (v - front_y) / max(0.01, 1 - front_y)
                front_top   = (130, 195,  95)
                front_floor = (85,  150,  62)
                col = lerp(front_top, front_floor, min(1.0, d * 1.4))

            px[x, y] = (*col, 255)

    # Soft cloud streak in the upper-left
    if size >= 32:
        cloud = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cloud)
        cw = max(8, size // 3)
        ch = max(2, size // 24)
        cy = max(2, size // 8)
        cd.ellipse([size // 8, cy, size // 8 + cw, cy + ch * 2],
                   fill=(255, 255, 255, 110))
        cd.ellipse([size // 6, cy + ch, size // 6 + cw // 2, cy + ch * 2],
                   fill=(255, 255, 255, 80))
        cloud = cloud.filter(ImageFilter.GaussianBlur(radius=max(1, size / 64)))
        img = Image.alpha_composite(img, cloud)

    # Chat bubble overlay (upper-right)
    bubble = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bubble)

    bw = int(size * 0.50)
    bh = int(size * 0.30)
    bx = size - bw - max(1, size // 24)
    by = max(1, size // 16)
    radius = max(2, bh // 3)

    # Drop shadow
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [bx + 1, by + 2, bx + bw + 1, by + bh + 2],
        radius=radius, fill=(0, 0, 0, 90),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, size / 48)))
    img = Image.alpha_composite(img, shadow)

    # The bubble body
    bd.rounded_rectangle(
        [bx, by, bx + bw, by + bh],
        radius=radius, fill=(255, 255, 255, 240),
        outline=(70, 110, 170, 220), width=max(1, size // 64),
    )

    # Tail (pointing down-left toward the scene)
    tx = bx + bw // 4
    ty = by + bh - 1
    tail_h = max(2, size // 8)
    bd.polygon(
        [(tx, ty),
         (tx + max(2, size // 12), ty + tail_h),
         (tx + max(4, size // 6),  ty)],
        fill=(255, 255, 255, 240),
        outline=(70, 110, 170, 220),
    )

    # Three "typing" dots
    if size >= 24:
        dot_r = max(1, size // 32)
        gap = max(2, size // 16)
        total_w = dot_r * 6 + gap * 2
        dx0 = bx + (bw - total_w) // 2 + dot_r
        dy0 = by + bh // 2
        for i in range(3):
            cx = dx0 + i * (dot_r * 2 + gap)
            bd.ellipse([cx - dot_r, dy0 - dot_r, cx + dot_r, dy0 + dot_r],
                       fill=(70, 110, 170, 230))

    img = Image.alpha_composite(img, bubble)

    # Rounded corners so it tiles cleanly as a desktop / Start menu icon
    if size >= 24:
        mask = Image.new("L", (size, size), 0)
        md = ImageDraw.Draw(mask)
        corner = max(2, size // 12)
        md.rounded_rectangle([0, 0, size - 1, size - 1], radius=corner, fill=255)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask=mask)
        img = out

    return img


def main():
    # Render the largest size at full detail, then let Pillow downsample
    # for the smaller variants. This produces sharper results than
    # rendering each size independently (small sizes lose detail in the
    # per-pixel hill gradient).
    master = render(max(SIZES))
    master.save(
        OUT,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(SIZES)} sizes)")


if __name__ == "__main__":
    main()
