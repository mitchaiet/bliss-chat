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


def write_ico_bmp(images, out_path):
    """
    Hand-write a multi-size .ico with each entry as BITMAPINFOHEADER + 32bpp
    BGRA pixel data + 1bpp AND mask.

    Pillow's ICO writer uses PNG-encoded entries which Windows XP's icon
    loader does not understand (PNG-in-ICO support was added in Vista).
    """
    import struct

    n = len(images)
    icondir = struct.pack("<HHH", 0, 1, n)  # reserved=0, type=1=icon, count=n

    entries = []
    blobs = []
    offset = 6 + 16 * n  # after ICONDIR + n * ICONDIRENTRY

    for img in images:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size

        # 32bpp pixel rows, bottom-up, BGRA byte order.
        pixels = img.load()
        pixel_rows = []
        for y in range(h - 1, -1, -1):
            row = bytearray()
            for x in range(w):
                r, g, b, a = pixels[x, y]
                row += bytes((b, g, r, a))
            pixel_rows.append(bytes(row))
        pixel_data = b"".join(pixel_rows)

        # AND mask: 1bpp, bottom-up, rows padded to multiple of 4 bytes.
        # 0 = opaque, 1 = transparent. Even with 32bpp + alpha, the mask must
        # be present, and XP can use it for cursor-style hit-testing.
        bytes_per_mask_row = ((w + 31) // 32) * 4
        mask_rows = []
        for y in range(h - 1, -1, -1):
            bits = bytearray(bytes_per_mask_row)
            for x in range(w):
                _, _, _, a = pixels[x, y]
                if a < 128:
                    bits[x >> 3] |= 0x80 >> (x & 7)
            mask_rows.append(bytes(bits))
        mask_data = b"".join(mask_rows)

        # BITMAPINFOHEADER: 40 bytes. Note: biHeight is doubled (XOR + AND).
        bih = struct.pack(
            "<IiiHHIIiiII",
            40,         # biSize
            w,          # biWidth
            h * 2,      # biHeight = image + mask
            1,          # biPlanes
            32,         # biBitCount
            0,          # biCompression = BI_RGB
            len(pixel_data) + len(mask_data),  # biSizeImage
            0, 0,       # biXPelsPerMeter, biYPelsPerMeter
            0, 0,       # biClrUsed, biClrImportant
        )

        blob = bih + pixel_data + mask_data
        # ICONDIRENTRY: width/height stored as u8, with 0 meaning 256.
        ww = 0 if w == 256 else w
        hh = 0 if h == 256 else h
        entries.append(struct.pack(
            "<BBBBHHII",
            ww, hh, 0, 0,
            1,          # color planes
            32,         # bits per pixel
            len(blob),  # bytes_in_res
            offset,     # image offset
        ))
        blobs.append(blob)
        offset += len(blob)

    with open(out_path, "wb") as f:
        f.write(icondir)
        for e in entries:
            f.write(e)
        for b in blobs:
            f.write(b)


def main():
    # Render the largest size at full detail, then downsample for the
    # smaller variants — small sizes lose detail if rendered directly,
    # because the per-pixel hill gradient assumes more pixels than
    # 16x16 has.
    master = render(max(SIZES))
    images = []
    for s in SIZES:
        if s == max(SIZES):
            images.append(master)
        else:
            images.append(master.resize((s, s), Image.LANCZOS))

    write_ico_bmp(images, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(SIZES)} sizes, BMP format)")


if __name__ == "__main__":
    main()
