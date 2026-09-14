#!/usr/bin/env python3
"""
make_icon.py — generates the WayFinder launcher icon set.

DESIGN RATIONALE
The mark is a navigation chevron with a trail behind it. The trail starts
SOLID and becomes DASHED as it approaches the chevron. That is the product in
one image: the solid stretch is travel tracked by GNSS, the dashed stretch is
where the signal was lost and the system is dead reckoning — still moving,
still confident, just unaided.

Colours are taken from the app's dark map aesthetic: deep navy ground with a
cyan heading marker, which stays legible when Android shrinks the icon into a
48 dp mask.

Outputs (Android):
  mipmap-*/ic_launcher.png              legacy square-ish icon
  mipmap-*/ic_launcher_foreground.png   adaptive foreground (safe-zone aware)
  mipmap-*/ic_launcher_monochrome.png   Android 13+ themed icon
  mipmap-anydpi-v26/ic_launcher.xml     adaptive icon descriptor
  values/ic_launcher_background.xml     adaptive background colour

Run:  python3 tool/make_icon.py
"""

import math
import os

from PIL import Image, ImageDraw

SS = 4                      # supersampling factor for smooth edges
BASE = 1024                 # design canvas
R = BASE * SS

NAVY_TOP = (12, 32, 58)     # #0C203A
NAVY_BOT = (7, 18, 36)      # #071224
CYAN = (34, 211, 238)       # #22D3EE
CYAN_FOLD = (16, 165, 194)   # darker half, gives the chevron its centre fold
CYAN_DIM = (13, 116, 138)   # trail
WHITE = (255, 255, 255)

RES_DIR = os.path.join(os.path.dirname(__file__), '..',
                       'android', 'app', 'src', 'main', 'res')

# Android launcher densities: (dir suffix, legacy px, adaptive px)
DENSITIES = [
    ('mdpi', 48, 108),
    ('hdpi', 72, 162),
    ('xhdpi', 96, 216),
    ('xxhdpi', 144, 324),
    ('xxxhdpi', 192, 432),
]


def qbezier(p0, p1, p2, t):
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return x, y


def _cap(d, pt, width, colour):
    x, y = pt
    r = width / 2
    d.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def draw_trail(d, p0, p1, p2, width, colour, solid_frac=0.50):
    """A solid run that gives way to discrete dots.

    The solid stretch is travel tracked by GNSS; the dots are where the signal
    was lost and the system is dead reckoning.

    Dots rather than dashes-along-a-curve: at launcher sizes the round caps of
    a dashed stroke are wider than the gaps between them, so dashes merge back
    into a continuous worm. Discrete circles stay unambiguous down to 48 px.
    """
    steps = 900
    pts = [qbezier(p0, p1, p2, i / steps) for i in range(steps + 1)]

    cum = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    total = cum[-1]

    def at(length):
        for i in range(1, len(cum)):
            if cum[i] >= length:
                return pts[i]
        return pts[-1]

    solid_len = total * solid_frac
    seg = [pts[i] for i in range(len(cum)) if cum[i] <= solid_len]
    if len(seg) > 1:
        # Subsample before stroking. Feeding PIL hundreds of near-coincident
        # points makes each segment shorter than the stroke is wide, and the
        # joins leave a serrated edge along the outside of the curve.
        step = max(1, len(seg) // 24)
        thin = seg[::step]
        if thin[-1] != seg[-1]:
            thin.append(seg[-1])
        d.line(thin, fill=colour, width=width, joint='curve')
        _cap(d, thin[0], width, colour)
        _cap(d, thin[-1], width, colour)

    # three dots, easing outward and shrinking slightly as confidence decays
    for frac, scale in ((0.66, 0.92), (0.82, 0.78), (0.97, 0.64)):
        x, y = at(total * frac)
        r = width * 0.5 * scale
        d.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def chevron(cx, cy, size, angle_deg, draw_obj, fill, fold_fill=None):
    """Classic navigation chevron: a triangle notched at the tail.

    When [fold_fill] is given, the right half is drawn a shade darker to give
    the mark a crisp centre fold — this is what makes it read as a 3D heading
    marker rather than a flat paper-plane shape.
    """
    tip, right, notch, left = (0.0, -1.0), (0.62, 0.78), (0.0, 0.34), (-0.62, 0.78)
    a = math.radians(angle_deg)

    def rot(p):
        x, y = p
        return (cx + (x * math.cos(a) - y * math.sin(a)) * size,
                cy + (x * math.sin(a) + y * math.cos(a)) * size)

    draw_obj.polygon([rot(tip), rot(right), rot(notch), rot(left)], fill=fill)
    if fold_fill is not None:
        draw_obj.polygon([rot(tip), rot(right), rot(notch)], fill=fold_fill)


def render_mark(canvas_px, with_background, safe_scale=1.0, mono=False):
    """Render the logo. safe_scale shrinks artwork into the adaptive safe zone."""
    img = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if with_background:
        # vertical gradient ground
        for y in range(R):
            f = y / R
            col = tuple(int(NAVY_TOP[i] + (NAVY_BOT[i] - NAVY_TOP[i]) * f)
                        for i in range(3))
            d.line([(0, y), (R, y)], fill=col + (255,))

    art = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    ad = ImageDraw.Draw(art)

    # Trail sweeps from lower-left up into the tail of the chevron, so the
    # mark reads as one continuous movement rather than two stacked shapes.
    trail_w = int(46 * SS)
    p0 = (R * 0.200, R * 0.815)
    p1 = (R * 0.300, R * 0.520)
    p2 = (R * 0.490, R * 0.570)

    trail_col = WHITE if mono else CYAN_DIM
    head_col = WHITE if mono else CYAN
    fold_col = None if mono else CYAN_FOLD
    draw_trail(ad, p0, p1, p2, trail_w, trail_col)

    # heading chevron, pointing up-right along the trail's tangent
    chevron(R * 0.620, R * 0.400, R * 0.235, 40, ad, head_col, fold_col)

    if safe_scale != 1.0:
        s = int(R * safe_scale)
        art_s = art.resize((s, s), Image.LANCZOS)
        art = Image.new('RGBA', (R, R), (0, 0, 0, 0))
        art.paste(art_s, ((R - s) // 2, (R - s) // 2), art_s)

    img = Image.alpha_composite(img, art)
    return img.resize((canvas_px, canvas_px), Image.LANCZOS)


def rounded_legacy(px):
    """Legacy icon: the mark on a rounded-square plate."""
    img = render_mark(R // SS, with_background=True)
    img = img.resize((R, R), Image.LANCZOS)
    mask = Image.new('L', (R, R), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, R - 1, R - 1],
                                           radius=int(R * 0.22), fill=255)
    out = Image.new('RGBA', (R, R), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out.resize((px, px), Image.LANCZOS)


def main():
    res = os.path.abspath(RES_DIR)
    for suffix, legacy_px, adaptive_px in DENSITIES:
        out_dir = os.path.join(res, f'mipmap-{suffix}')
        os.makedirs(out_dir, exist_ok=True)

        rounded_legacy(legacy_px).save(os.path.join(out_dir, 'ic_launcher.png'))
        # adaptive foreground: artwork must sit inside the central 66/108
        render_mark(adaptive_px, with_background=False, safe_scale=0.62) \
            .save(os.path.join(out_dir, 'ic_launcher_foreground.png'))
        render_mark(adaptive_px, with_background=False, safe_scale=0.62, mono=True) \
            .save(os.path.join(out_dir, 'ic_launcher_monochrome.png'))
        print(f'  mipmap-{suffix}: {legacy_px}px legacy, {adaptive_px}px adaptive')

    anydpi = os.path.join(res, 'mipmap-anydpi-v26')
    os.makedirs(anydpi, exist_ok=True)
    with open(os.path.join(anydpi, 'ic_launcher.xml'), 'w') as f:
        f.write('''<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />
</adaptive-icon>
''')

    values = os.path.join(res, 'values')
    os.makedirs(values, exist_ok=True)
    with open(os.path.join(values, 'ic_launcher_background.xml'), 'w') as f:
        f.write('''<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="ic_launcher_background">#0A1B31</color>
</resources>
''')

    # a large preview for docs / review
    rounded_legacy(512).save(os.path.join(os.path.dirname(__file__),
                                          'wayfinder_icon_preview.png'))
    print('  adaptive-icon xml + background colour written')
    print('  preview: tool/wayfinder_icon_preview.png')


if __name__ == '__main__':
    main()
