#!/usr/bin/env python3
import math
import struct


def pixel(size, x, y):
    scale = size / 48.0

    def inside_rect(rx, ry, rw, rh):
        return rx <= x < rx + rw and ry <= y < ry + rh

    # Transparent rounded-ish corners.
    r = max(3, int(size * 0.12))
    if x < r and y < r and (x - r) ** 2 + (y - r) ** 2 > r * r:
        return (0, 0, 0, 0)
    if x >= size - r and y < r and (x - (size - r - 1)) ** 2 + (y - r) ** 2 > r * r:
        return (0, 0, 0, 0)
    if x < r and y >= size - r and (x - r) ** 2 + (y - (size - r - 1)) ** 2 > r * r:
        return (0, 0, 0, 0)
    if x >= size - r and y >= size - r and (x - (size - r - 1)) ** 2 + (y - (size - r - 1)) ** 2 > r * r:
        return (0, 0, 0, 0)

    # XP-ish blue tile background.
    t = y / max(1, size - 1)
    red = int(18 + 20 * (1 - t))
    green = int(74 + 42 * (1 - t))
    blue = int(142 + 58 * (1 - t))
    alpha = 255

    sx = lambda v: int(round(v * scale))

    # CRT shell.
    if inside_rect(sx(8), sx(9), sx(32), sx(24)):
        red, green, blue = 226, 229, 219
    if inside_rect(sx(11), sx(12), sx(26), sx(16)):
        red, green, blue = 9, 30, 34

    # Screen glow.
    if inside_rect(sx(13), sx(14), sx(22), sx(12)):
        glow = int(35 + 60 * math.sin((x + y) * 0.22))
        red, green, blue = 28, 145 + glow // 3, 92

    # Pixel prompt mark.
    if inside_rect(sx(15), sx(17), sx(6), max(1, sx(2))) or inside_rect(sx(20), sx(19), max(1, sx(2)), max(1, sx(2))) or inside_rect(sx(15), sx(21), sx(6), max(1, sx(2))):
        red, green, blue = 230, 255, 177

    # Stand.
    if inside_rect(sx(20), sx(33), sx(8), sx(4)) or inside_rect(sx(15), sx(38), sx(18), sx(4)):
        red, green, blue = 192, 196, 187

    # Spark.
    cx, cy = sx(38), sx(10)
    if abs(x - cx) + abs(y - cy) <= max(1, sx(4)):
        red, green, blue = 255, 218, 76

    return (red, green, blue, alpha)


def dib_for_size(size):
    header = struct.pack(
        "<IIIHHIIIIII",
        40,
        size,
        size * 2,
        1,
        32,
        0,
        size * size * 4,
        0,
        0,
        0,
        0,
    )
    rows = bytearray()
    for y in range(size - 1, -1, -1):
        for x in range(size):
            r, g, b, a = pixel(size, x, y)
            rows += bytes((b, g, r, a))
    mask_stride = ((size + 31) // 32) * 4
    mask = bytes(mask_stride * size)
    return header + rows + mask


def main():
    sizes = [16, 32, 48]
    images = [dib_for_size(s) for s in sizes]
    offset = 6 + len(sizes) * 16
    entries = bytearray()
    for size, image in zip(sizes, images):
        entries += struct.pack(
            "<BBBBHHII",
            size,
            size,
            0,
            0,
            1,
            32,
            len(image),
            offset,
        )
        offset += len(image)

    with open("xp_tiny_llm.ico", "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(sizes)))
        f.write(entries)
        for image in images:
            f.write(image)


if __name__ == "__main__":
    main()
