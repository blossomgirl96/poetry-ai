"""The painting behind a downloaded poem.

v4 had one way to paint: blurred radial ellipses on off-white paper. However the
model placed them, every card read as the same soft beige fog, and Haiku reached
for the same handful of sepia hexes for almost every poem. Two fixes here:

1. Several *kinds* of picture, each drawn by its own renderer — a watercolour
   bloom, an ink blot, a field of colour, a sky at a given hour, contour lines,
   a two-colour print, a night sky, a photogram of grasses. The model picks the
   kind; the code draws it. Different kinds look different even in the same
   colours, which no amount of prompt-tuning got from one primitive.

2. Colour comes from the poem's own images. The model names the concrete things
   in the poem first and gives each a colour, then builds the palette from
   those — so a poem with a blue bicycle and a lemon tree can't come back brown.

Everything is SVG drawn here, seeded from the poem, so the same poem always
gets the same picture and Chrome prints it as it shows it. No full-page noise
texture: Chrome rasterises it at print resolution and it doesn't compress — a
paper-grain layer took the PDF from about 2 MB to 35 MB.
"""

import colorsys
import hashlib
import json
import math
import random
import re
from collections import deque

import base64
import os
import sys

import anthropic
import httpx

# Better taste than Haiku for this, and still a couple of seconds at low effort.
ART_MODEL = "claude-sonnet-5-5"

STYLES = {
    "watercolour": "Pools of wet pigment with ragged, bleeding edges and darker "
                   "tide-marks where they dried. Weather, tenderness, things "
                   "half-remembered.",
    "ink": "One or two heavy black-or-near-black ink blots with feathered edges "
           "and a few spatters; the rest bare paper. Grief, gravity, a single "
           "hard fact, restraint.",
    "colourfield": "Two or three soft-edged slabs of colour stacked across the "
                   "whole sheet, like a Rothko. Big plain feeling, a room, a "
                   "season, the colour of a life.",
    "horizon": "The sheet becomes a sky at a particular hour, graded top to "
               "bottom, with a horizon line and optionally a sun or moon. "
               "Distance, travel, waiting, dawn or dusk, the sea.",
    "contour": "Fine concentric lines like a topographic map, tree rings or "
               "ripples spreading on water. Time, place, memory, home ground, "
               "something growing slowly.",
    "riso": "A bold two-colour print: flat circles, arcs and bands, one in "
            "halftone dots, slightly misregistered. Playful, modern, city, "
            "music, joy, someone loud and bright.",
    "night": "Dark paper, a scatter of stars, faint coloured haze, optionally a "
             "moon. Night, sleep, insomnia, the dead, longing, wonder.",
    "photogram": "A cyanotype: deep blue (or another dark) sheet with pale "
                 "silhouettes of grasses and fronds rising from the bottom. "
                 "Gardens, summers, childhood, something preserved.",
}

ART_PROMPT = f"""You art-direct the background for a poem that will be printed \
and kept. The poem is set in a column down the middle; the picture lives around \
and behind it. Every poem should get a picture that could only belong to it.

Work in this order.

1. `images`: name two to four concrete, visible things in the poem itself — \
objects, places, light, weather, food, clothes — and give each the colour it \
actually is in the poem's world, as a six-digit hex. Literal, not symbolic: a \
yellow raincoat is yellow.

2. `style`: choose the kind of picture.
{chr(10).join(f"   - {k}: {v}" for k, v in STYLES.items())}

3. `colors`: two to five hexes, most important first, built from your images. \
For horizon, list them sky-top to ground instead. Choose saturated, mid-tone \
colour; the renderer handles the softening. Do not fall back on tan, sepia, \
beige or brown (#8B7355, #D4A574 and their neighbours) unless the poem is \
literally about wood, tea, sand, old photographs or similar. If your colours \
all came out brownish, look at the poem again for its real colours.

4. `paper`: the sheet. Light for most styles — off-white, but it may lean \
toward the palette (cool grey-white, pale green, warm pink). Dark for night \
and photogram (e.g. #14182B, #1C3B6B).

5. `focus_x`, `focus_y` (0–100): where the picture's weight sits on the sheet. \
Corners and edges keep the poem readable; the centre only for quiet styles.

6. `density` (0–1): how much of the sheet it covers. `intensity` (0–1): how \
strong the colour is. `feature`: true to include a sun or moon (horizon, night).

7. `image_prompt`: a prompt for an image model that will paint this background \
for real. Describe one portrait page in the medium of your chosen style \
(watercolour, sumi ink, colour-field painting, a sky, a topographic print, a \
two-colour risograph, a night sky, a cyanotype), using your colours and the \
poem's own images — the actual teddy bear, the actual kitchen door — painted \
loosely. Put everything near your focus point, at the edges or in one corner, \
and say plainly that the middle of the page, where the poem is set, stays \
nearly empty paper (or empty sky, for dark styles). Name the paper colour. End \
with: no text, no letters, no words, no signature, no border, no frame.

Fit the poem, not a house style. If you are told which styles were used \
recently for other people's poems, prefer a different one unless this poem \
clearly needs it."""

_HEX = {"type": "string", "description": "Six-digit hex like #3A6EA5"}

ART_SCHEMA = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"image": {"type": "string"}, "color": _HEX},
                "required": ["image", "color"],
                "additionalProperties": False,
            },
        },
        "style": {"type": "string", "enum": list(STYLES)},
        "why": {"type": "string"},
        "colors": {"type": "array", "items": _HEX},
        "paper": _HEX,
        "focus_x": {"type": "number"},
        "focus_y": {"type": "number"},
        "density": {"type": "number"},
        "intensity": {"type": "number"},
        "feature": {"type": "boolean"},
        "mood": {"type": "string"},
        "image_prompt": {"type": "string"},
    },
    "required": ["images", "style", "why", "colors", "paper", "focus_x",
                 "focus_y", "density", "intensity", "feature", "mood", "image_prompt"],
    "additionalProperties": False,
}

# Used only when the call fails — deliberately plain.
FALLBACK = {
    "style": "watercolour", "why": "fallback", "mood": "quiet", "images": [],
    "colors": ["#5F7F96", "#9AAFB8"], "paper": "#FBFAF7",
    "focus_x": 80, "focus_y": 12, "density": .45, "intensity": .5, "feature": False,
}

# The last few styles painted in this process, so one style doesn't take over.
_recent = deque(maxlen=4)

_HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}")


# ── colour helpers ───────────────────────────────────────────────────────────

def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _hex(rgb):
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02X}" for c in rgb)


def _mix(a, b, t):
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))


def _lum(h):
    def ch(c):
        return c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in _rgb(h))
    return .2126 * r + .7152 * g + .0722 * b


def _darken(h, t):
    r, g, b = _rgb(h)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    return _hex(colorsys.hls_to_rgb(hh, max(0, ll * (1 - t)), ss))


def _clamp(v, lo, hi, fallback):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return fallback


# ── art direction ────────────────────────────────────────────────────────────

def clean(data):
    """Clamp whatever came back into something every renderer can draw."""
    style = data.get("style") if data.get("style") in STYLES else FALLBACK["style"]
    colors = [c for c in (data.get("colors") or []) if isinstance(c, str) and _HEX_RE.fullmatch(c)]
    colors = colors[:5] or [i["color"] for i in data.get("images") or []
                            if isinstance(i.get("color"), str) and _HEX_RE.fullmatch(i["color"])][:5]
    if not colors:
        colors = list(FALLBACK["colors"])
    if len(colors) == 1:
        colors.append(_mix(colors[0], "#FFFFFF", .45))
    paper = data.get("paper")
    if not (isinstance(paper, str) and _HEX_RE.fullmatch(paper)):
        paper = "#121628" if style == "night" else "#1C3B6B" if style == "photogram" else FALLBACK["paper"]
    # These two only work dark; the rest only work light. Correct a mismatch
    # rather than print white text on white or black on navy.
    if style in ("night", "photogram") and _lum(paper) > .12:
        paper = _mix(paper, "#0B0E1A", .8)
    elif style not in ("night", "photogram", "horizon") and _lum(paper) < .6:
        paper = _mix(paper, "#FFFFFF", .85)
    return {
        "style": style,
        "why": str(data.get("why") or "")[:300],
        "mood": str(data.get("mood") or "")[:60] or "quiet",
        "images": [{"image": str(i.get("image", ""))[:60], "color": i.get("color")}
                   for i in (data.get("images") or [])[:4] if isinstance(i, dict)],
        "colors": colors,
        "paper": paper.upper(),
        "focus_x": _clamp(data.get("focus_x"), 0, 100, 70),
        "focus_y": _clamp(data.get("focus_y"), 0, 100, 20),
        "density": _clamp(data.get("density"), .1, 1, .5),
        "intensity": _clamp(data.get("intensity"), .15, 1, .6),
        "feature": bool(data.get("feature")),
        "image_prompt": str(data.get("image_prompt") or "")[:3000],
    }


def direct(client, text):
    """Ask for an art direction for this poem. Never raises."""
    note = ""
    if _recent:
        note = ("\n\n(Styles used recently for other poems: "
                + ", ".join(_recent) + ".)")
    try:
        msg = client.beta.messages.create(
            model=ART_MODEL,
            max_tokens=2000,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            system=ART_PROMPT,
            messages=[{"role": "user", "content": f"<poem>\n{text}\n</poem>{note}"}],
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": ART_SCHEMA}},
        )
        if msg.stop_reason == "refusal":
            print("[wash] art direction refused, using fallback", file=sys.stderr)
            return dict(FALLBACK)
        data = json.loads("".join(b.text for b in msg.content if b.type == "text"))
    except (anthropic.APIError, ValueError) as e:
        print(f"[wash] art direction failed, using fallback: {e!r}", file=sys.stderr)
        return dict(FALLBACK)
    art = clean(data)
    _recent.append(art["style"])
    return art


# ── painting it for real ─────────────────────────────────────────────────────
# An image model paints the art direction. The drawn SVG renderers below stay
# as the fallback: no key, an error, or a refusal and the card still has a
# picture, just a drawn one.

IMAGE_MODEL = "gpt-image-2"
IMAGES_URL = "https://api.openai.com/v1/images/generations"


def paint(art):
    """A data: URI for the painted background, or None. Never raises."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not art.get("image_prompt"):
        return None
    try:
        r = httpx.post(
            IMAGES_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": IMAGE_MODEL, "prompt": art["image_prompt"],
                  # Portrait, close to A4. JPEG at 85 keeps a card to ~200 KB;
                  # PNG was the size of the whole PDF budget on its own.
                  "size": "1024x1536", "quality": "medium",
                  "output_format": "jpeg", "output_compression": 85, "n": 1},
            timeout=120,
        )
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        base64.b64decode(b64, validate=True)
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        detail = getattr(getattr(e, "response", None), "text", "")[:300]
        print(f"[wash] image model failed, drawing instead: {e!r} {detail}", file=sys.stderr)
        return None
    return f"data:image/jpeg;base64,{b64}"


def text_theme(ground):
    """Ink colours that read on a sheet of this colour."""
    if _lum(ground) < .2:
        return {"ink": "#F4EEE4", "muted": "#CBC2B6", "accent": "#E6BC90", "rule": "rgba(255,255,255,.2)"}
    return {"ink": "#241F1A", "muted": "#6E645A", "accent": "#8A5A34", "rule": "rgba(36,31,26,.14)"}


def veil_css(paper):
    """A wash of the paper colour over the text column, on top of a painting.

    The image model is told to keep the middle bare, but doesn't always; this
    makes sure the poem reads either way without hiding the edges.
    """
    r, g, b = (round(c * 255) for c in _rgb(paper))
    c = lambda a: f"rgba({r},{g},{b},{a})"
    return (f"linear-gradient(90deg,{c(0)} 0%,{c(.62)} 9%,{c(.62)} 80%,{c(0)} 96%)")


# ── geometry ─────────────────────────────────────────────────────────────────

W, H = 1000, 1414   # A4 proportions; the SVG slices to whatever the page is


def _smooth(points):
    """A closed Catmull-Rom curve through the points, as an SVG path."""
    n = len(points)
    d = [f"M{points[0][0]:.1f},{points[0][1]:.1f}"]
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
    return "".join(d) + "Z"


def _blob(rng, cx, cy, rx, ry, wobble=.25, n=14, phase=None):
    """An organic closed shape: an ellipse whose radius wanders smoothly."""
    ph = phase or [rng.uniform(0, 6.28) for _ in range(3)]
    amp = [wobble, wobble * .55, wobble * .3]
    pts = []
    for i in range(n):
        a = 6.2832 * i / n
        k = 1 + sum(m * math.sin(f * a + p) for m, f, p in zip(amp, (2, 3, 5), ph))
        pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
    return _smooth(pts)


def _around(rng, art, spread):
    """A point near the focus, in sheet units."""
    fx, fy = art["focus_x"] / 100 * W, art["focus_y"] / 100 * H
    return (fx + rng.gauss(0, spread * W), fy + rng.gauss(0, spread * H))


def _bleed(fid, seed, freq, scale, blur):
    """Displace an edge with noise, then soften it — the wet-paper look."""
    return (f'<filter id="{fid}" x="-30%" y="-30%" width="160%" height="160%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="{freq}" numOctaves="3" seed="{seed}" result="n"/>'
            f'<feDisplacementMap in="SourceGraphic" in2="n" scale="{scale}" '
            f'xChannelSelector="R" yChannelSelector="G"/>'
            f'<feGaussianBlur stdDeviation="{blur}"/></filter>')


# ── the renderers ────────────────────────────────────────────────────────────
# Each returns (defs, body, paper_override_or_None).

def _watercolour(rng, art, seed):
    defs = [_bleed("wet", seed, .012, 70, 9), _bleed("edge", seed + 1, .02, 50, 2.5)]
    fills, edges = [], []
    n = 2 + round(art["density"] * 6)
    for i in range(n):
        col = art["colors"][i % len(art["colors"])]
        cx, cy = _around(rng, art, .14 + art["density"] * .2)
        size = rng.uniform(.16, .34) * W * (.7 + art["density"])
        d = _blob(rng, cx, cy, size, size * rng.uniform(.6, 1.1), wobble=.22)
        a = art["intensity"] * rng.uniform(.28, .5)
        fills.append(f'<path d="{d}" fill="{col}" opacity="{a:.2f}" style="mix-blend-mode:multiply"/>')
        edges.append(f'<path d="{d}" fill="none" stroke="{_darken(col, .25)}" stroke-width="5" opacity="{a * .55:.2f}"/>')
    # One filter per group, not per shape: Chrome rasterises every filtered
    # element separately when printing, and per-shape filters made 10 MB PDFs.
    body = [f'<g filter="url(#wet)">{"".join(fills)}</g>',
            f'<g filter="url(#edge)" style="mix-blend-mode:multiply">{"".join(edges)}</g>']
    return defs, body, None


def _ink(rng, art, seed):
    ink = art["colors"][0] if _lum(art["colors"][0]) < .08 else _darken(art["colors"][0], .7)
    defs = [_bleed("feather", seed, .03, 34, 1.2), _bleed("soak", seed + 3, .015, 50, 12)]
    body = []
    # Pulled in from the very edge: a blot centred on the corner is mostly
    # off the sheet and reads as a smudge, not a decision.
    art = dict(art, focus_x=min(88, max(12, art["focus_x"])), focus_y=min(90, max(10, art["focus_y"])))
    for i in range(1 + round(art["density"] * 2)):
        cx, cy = _around(rng, art, .07)
        r = rng.uniform(.17, .28) * W * (.8 + art["density"] * .6)
        d = _blob(rng, cx, cy, r, r * rng.uniform(.6, 1), wobble=.32, n=18)
        a = .5 + art["intensity"] * .4
        body.append(f'<path d="{d}" fill="{ink}" opacity="{a * .35:.2f}" filter="url(#soak)" transform="translate(6 8)"/>')
        body.append(f'<path d="{d}" fill="{ink}" opacity="{a:.2f}" filter="url(#feather)"/>')
    dots = []
    for _ in range(int(14 + art["density"] * 30)):
        x, y = _around(rng, art, .16)
        dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rng.paretovariate(2.2) * 2.2:.1f}" '
                    f'fill="{ink}" opacity="{rng.uniform(.4, .85):.2f}"/>')
    body.append(f'<g filter="url(#feather)">{"".join(dots)}</g>')
    if len(art["colors"]) > 1:   # one wash of the second colour, faint, under it
        cx, cy = _around(rng, art, .1)
        d = _blob(rng, cx, cy, .3 * W, .22 * W, wobble=.2)
        body.insert(0, f'<path d="{d}" fill="{art["colors"][1]}" opacity="{.12 + art["intensity"] * .15:.2f}" filter="url(#soak)"/>')
    return defs, body, None


def _colourfield(rng, art, seed):
    # Blots, not bars. Long smooth capsules read as sausages (or corn dogs), so
    # each colour is an ink-blot outline — same wobble and feathered edge as
    # _ink — kept no more than 1.5x longer than it is wide.
    defs = [_bleed("cf_edge", seed, .03, 34, 3), _bleed("cf_soak", seed + 3, .015, 50, 14)]
    count = min(len(art["colors"]), rng.choice([2, 3, 3, 4]))
    cols = art["colors"][:max(2, count)]
    # Stacked down the sheet or standing side by side; the seed decides, so two
    # colourfield poems don't come out as the same arrangement.
    vertical = rng.random() < .4
    span = W if vertical else H
    weights = [rng.uniform(.6, 1.4) for _ in cols]
    total = sum(weights)
    body, pos = [], -60
    a = .3 + art["intensity"] * .3
    for col, wgt in zip(cols, weights):
        along = (span + 120) * wgt / total / 2 * (1 + art["density"] * .2)
        across = min(along * 1.5, (W if not vertical else H) * rng.uniform(.3, .42))
        along = min(along, across * 1.5)
        mid = pos + (span + 120) * wgt / total / 2
        jitter = rng.gauss(0, .08)
        if vertical:
            cx, cy, rx, ry = mid, H * (.5 + jitter), along, across
        else:
            cx, cy, rx, ry = W * (.5 + jitter), mid, across, along
        d = _blob(rng, cx, cy, rx, ry, wobble=.3, n=18)
        body.append(f'<g style="mix-blend-mode:multiply">'
                    f'<path d="{d}" fill="{col}" opacity="{a * .45:.2f}" filter="url(#cf_soak)" transform="translate(8 10)"/>'
                    f'<path d="{d}" fill="{col}" opacity="{a:.2f}" filter="url(#cf_edge)"/></g>')
        pos += (span + 120) * wgt / total
    return defs, body, None


def _horizon(rng, art, seed):
    cols = art["colors"]
    # Push the sky to one side of mid-grey so the poem reads on it: a light sky
    # gets lighter, a dark one darker. Mid-tones fail with either ink.
    dark = sum(_lum(c) for c in cols) / len(cols) < .18
    target = "#0A0D18" if dark else "#FFFFFF"
    sky = [_mix(c, target, .55 if dark else .5) for c in cols]
    stops = "".join(f'<stop offset="{i / max(1, len(sky) - 1):.2f}" stop-color="{c}"/>'
                    for i, c in enumerate(sky))
    hy = (.55 + art["focus_y"] / 100 * .3) * H
    defs = [f'<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">{stops}</linearGradient>',
            _bleed("haze", seed, .006, 30, 16), _bleed("glow", seed + 2, .01, 10, 30)]
    body = [f'<rect width="{W}" height="{H}" fill="url(#sky)"/>']
    land = _mix(cols[-1], target, .25 if dark else .2)
    pts = [(-50, hy)] + [(x, hy + rng.gauss(0, 8 + art["density"] * 24))
                         for x in range(0, W + 100, 100)] + [(W + 50, hy)]
    ridge = "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in pts) + f" L{W + 50},{H + 50} L-50,{H + 50}Z"
    body.append(f'<path d="{ridge}" fill="{land}" opacity="{.35 + art["intensity"] * .35:.2f}" filter="url(#haze)"/>')
    if art["feature"]:
        # The poem runs down the middle, so the sun sits out to one side.
        sx = (min(art["focus_x"], 16) if art["focus_x"] < 50 else max(art["focus_x"], 84)) / 100 * W
        sy = hy - rng.uniform(.06, .2) * H
        disc = _mix(cols[0] if dark else cols[min(1, len(cols) - 1)], "#FFFFFF", .5 if dark else 0)
        body.append(f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="150" fill="{disc}" opacity=".35" filter="url(#glow)"/>')
        body.append(f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="{46 + art["intensity"] * 30:.0f}" fill="{disc}" opacity=".85"/>')
    return defs, body, sky[len(sky) // 2]


def _contour(rng, art, seed):
    body = []
    centres = 1 + round(art["density"] * 2)
    for c in range(centres):
        col = art["colors"][c % len(art["colors"])]
        cx, cy = _around(rng, art, .12) if c == 0 else (rng.uniform(0, W), rng.uniform(0, H))
        phase = [rng.uniform(0, 6.28) for _ in range(3)]
        rings = 8 + round(art["density"] * 14)
        step = rng.uniform(22, 38)
        for k in range(1, rings + 1):
            # Same phase each ring, so they nest like real contours; the wobble
            # grows outward so the outer lines drift.
            d = _blob(rng, cx, cy, k * step, k * step * .8, wobble=.08 + k * .008, n=20, phase=phase)
            a = (.45 + art["intensity"] * .45) * (1 - k / (rings + 6))
            w = 3 if k % 5 == 0 else 1.6
            body.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" opacity="{a:.2f}"/>')
    return [], body, None


def _riso(rng, art, seed):
    c1 = art["colors"][0]
    c2 = art["colors"][1 % len(art["colors"])]
    defs = [f'<pattern id="dots" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(22)">'
            f'<circle cx="4.5" cy="4.5" r="2.4" fill="{c2}"/></pattern>',
            _bleed("rough", seed, .9, 3, .3)]
    body = []
    a = .55 + art["intensity"] * .3
    off = rng.uniform(4, 9)
    shapes = 2 + round(art["density"] * 4)
    for i in range(shapes):
        x, y = _around(rng, art, .22)
        r = rng.uniform(.08, .22) * W
        kind = rng.choice(["circle", "half", "band", "arc"])
        fill = c1 if i % 2 == 0 else "url(#dots)"
        if kind == "circle":
            el = f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}"'
        elif kind == "half":
            el = (f'<path d="M{x - r:.0f},{y:.0f} A{r:.0f},{r:.0f} 0 0 1 {x + r:.0f},{y:.0f}Z" '
                  f'transform="rotate({rng.choice([0, 90, 180, 270])} {x:.0f} {y:.0f})"')
        elif kind == "band":
            el = (f'<rect x="{x - r * 1.6:.0f}" y="{y - r * .22:.0f}" width="{r * 3.2:.0f}" height="{r * .44:.0f}" '
                  f'rx="{r * .22:.0f}" transform="rotate({rng.uniform(-35, 35):.0f} {x:.0f} {y:.0f})"')
        else:
            el = (f'<path d="M{x - r:.0f},{y:.0f} A{r:.0f},{r:.0f} 0 0 1 {x + r:.0f},{y:.0f}" '
                  f'fill="none" stroke="{c1 if i % 2 else c2}" stroke-width="{r * .22:.0f}"')
        fill_attr = "" if kind == "arc" else f' fill="{fill}"'
        body.append(f'<g transform="translate({off:.1f} {off * .6:.1f})" style="mix-blend-mode:multiply" '
                    f'opacity="{a:.2f}" filter="url(#rough)">{el}{fill_attr}/></g>')
    return defs, body, None


def _night(rng, art, seed):
    defs = [_bleed("nebula", seed, .006, 90, 40)]
    body = []
    for col in art["colors"][:3]:
        cx, cy = _around(rng, art, .2)
        d = _blob(rng, cx, cy, rng.uniform(.25, .45) * W, rng.uniform(.15, .3) * W, wobble=.3)
        body.append(f'<path d="{d}" fill="{col}" opacity="{.12 + art["intensity"] * .2:.2f}" '
                    f'filter="url(#nebula)" style="mix-blend-mode:screen"/>')
    for _ in range(int(120 + art["density"] * 380)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.paretovariate(3) * .7
        body.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.2f}" fill="#FFF8EA" opacity="{rng.uniform(.25, .9):.2f}"/>')
    if art["feature"]:
        mx, my = art["focus_x"] / 100 * W, art["focus_y"] / 100 * H
        defs.append(f'<mask id="crescent"><rect width="{W}" height="{H}" fill="#fff"/>'
                    f'<circle cx="{mx + 26:.0f}" cy="{my - 14:.0f}" r="58" fill="#000"/></mask>')
        body.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="130" fill="#FFF3D6" opacity=".08" filter="url(#nebula)"/>')
        body.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="62" fill="#FFF3D6" opacity=".92" mask="url(#crescent)"/>')
    return defs, body, None


def _photogram(rng, art, seed):
    # Unfiltered on purpose: a full-sheet filter here printed at 5 MB, and a
    # contact print's edges are crisp anyway.
    defs = []
    body = []
    pale = _mix(art["colors"][0], "#FFFFFF", .8)
    stems = int(22 + art["density"] * 40)
    for _ in range(stems):
        x0 = art["focus_x"] / 100 * W + rng.gauss(0, W * .3)
        h = rng.uniform(.25, .7) * H * (.6 + art["density"] * .5)
        lean = rng.gauss(0, .25)
        x1, y1 = x0 + lean * h * .5, H - h * .55
        x2, y2 = x0 + lean * h, H - h
        a = rng.uniform(.35, .8) * (.7 + art["intensity"] * .4)
        body.append(f'<path d="M{x0:.0f},{H + 10} Q{x1:.0f},{y1:.0f} {x2:.0f},{y2:.0f}" fill="none" '
                    f'stroke="{pale}" stroke-width="{rng.uniform(2, 5):.1f}" opacity="{a:.2f}"/>')
        if rng.random() < .4:   # a seed head or leaf at the tip
            d = _blob(rng, x2, y2, rng.uniform(6, 16), rng.uniform(14, 34), wobble=.15, n=10)
            body.append(f'<path d="{d}" fill="{pale}" opacity="{a * .9:.2f}" '
                        f'transform="rotate({lean * 60:.0f} {x2:.0f} {y2:.0f})"/>')
    return defs, body, None


# How much of the picture survives behind the text column (1 = all of it).
# Dark-ink shapes and bold print need the most knocking back; a sky or a
# star field is meant to sit behind the words.
VEIL = {"watercolour": .55, "ink": .3, "colourfield": .75, "contour": .7,
        "riso": .35, "photogram": .5, "horizon": 1, "night": 1}


def _veil(strength):
    """A luminance mask: full strength at the edges, faded across the text."""
    if strength >= 1:
        return "", ""
    v = f"{round(strength * 255):02X}" * 3
    return (f'<linearGradient id="vg" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="#FFFFFF"/><stop offset=".1" stop-color="#{v}"/>'
            f'<stop offset=".72" stop-color="#{v}"/><stop offset=".9" stop-color="#FFFFFF"/>'
            f'</linearGradient><mask id="veil" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
            f'<rect width="{W}" height="{H}" fill="url(#vg)"/></mask>'), ' mask="url(#veil)"'


RENDERERS = {
    "watercolour": _watercolour, "ink": _ink, "colourfield": _colourfield,
    "horizon": _horizon, "contour": _contour, "riso": _riso,
    "night": _night, "photogram": _photogram,
}


def seed_for(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def render(art, seed):
    """The art direction as an SVG plus the ink colours that read on it."""
    rng = random.Random(seed)
    defs, body, behind_text = RENDERERS[art["style"]](rng, art, seed % 997)
    paper = art["paper"]
    ground = behind_text or paper
    dark = _lum(ground) < .2
    vdef, vattr = _veil(VEIL[art["style"]])
    svg = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" '
           f'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
           f'<defs>{"".join(defs)}{vdef}</defs>'
           f'<rect width="{W}" height="{H}" fill="{paper}"/>'
           f'<g{vattr}>{"".join(body)}</g></svg>')
    return svg, paper, text_theme(ground)
