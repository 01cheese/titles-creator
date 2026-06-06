"""
carousel_generator.py
Generates a TikTok carousel (1080×1920) from a list of games.
- Intro slide: random game cover from the list
- Game slides: one per game
- Outro slide: random game cover from the list (different from intro)
- All text in English
"""

import os
import io
import re
import random
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans

from dotenv import load_dotenv

STEAMGRIDDB_API_KEY = os.environ.get("STEAMGRIDDB_API_KEY")

if not STEAMGRIDDB_API_KEY:
    raise ValueError("ERROR: STEAMGRIDDB_API_KEY is missing in environment variables!")

TIKTOK_W, TIKTOK_H = 1080, 1920

COVER_DIMENSIONS = [
    "1080x1920",
    "1080x1350",
    "900x1350",
    "660x930",
    "600x900",
    "920x430",
]


# ── FONTS ─────────────────────────────────────────────────────────────────────

def find_font(bold=False):
    candidates_bold = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    candidates_reg = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        if os.path.exists(path):
            return path
    return None


FONT_BOLD = find_font(bold=True)
FONT_REG  = find_font(bold=False)


def load_font(font_path, size):
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def fit_text_size(draw, text, font_path, max_width, max_size=160, min_size=24):
    for size in range(max_size, min_size, -1):
        font = load_font(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font, size
    return load_font(font_path, min_size), min_size


def wrap_text(draw, text, font_path, max_width, max_size=160, min_size=40,
              max_lines=3):
    """
    Finds the largest font size where `text` wraps into at most `max_lines`
    lines, each fitting within `max_width`.

    Returns (font, size, lines: list[str]).
    Word-wraps greedily; if a single word is wider than max_width at min_size,
    it is left as-is on its own line.
    """
    words = text.split()

    for size in range(max_size, min_size - 1, -2):
        font = load_font(font_path, size)

        lines: list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            w = draw.textbbox((0, 0), test, font=font)[2]
            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        if len(lines) <= max_lines:
            return font, size, lines

    # Fallback: min_size, however many lines it takes
    font = load_font(font_path, min_size)
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        w = draw.textbbox((0, 0), test, font=font)[2]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return font, min_size, lines


def multiline_height(draw, lines, font, line_gap_ratio=0.15):
    """Total pixel height of wrapped lines block."""
    if not lines:
        return 0
    lh = draw.textbbox((0, 0), "A", font=font)[3]
    gap = int(lh * line_gap_ratio)
    return lh * len(lines) + gap * (len(lines) - 1), lh, gap


def draw_multiline(draw, lines, font, cx, start_y,
                   fill, shadow_fill=(0, 0, 0, 120),
                   shadow_offset=3, line_gap_ratio=0.15):
    """
    Draws centered multi-line text block.
    start_y = top of the first line's bounding box.
    Returns bottom y of the last line.
    """
    lh = draw.textbbox((0, 0), "A", font=font)[3]
    gap = int(lh * line_gap_ratio)
    y = start_y
    for line in lines:
        mid_y = y + lh // 2
        draw.text((cx + shadow_offset, mid_y + shadow_offset), line,
                  fill=shadow_fill, font=font, anchor="mm")
        draw.text((cx, mid_y), line,
                  fill=fill, font=font, anchor="mm")
        y += lh + gap
    return y  # bottom of last line


# ── ACCENT WORDS ───────────────────────────────────────────────────────────────

# Words that always get the accent color (case-insensitive match)
_ACCENT_KEYWORDS = {
    "games", "game", "top", "best", "worst", "hidden", "must",
    "epic", "legendary", "ultimate", "secret", "rare", "hidden",
}

def _word_is_accent(word: str) -> bool:
    """Return True if this word should be rendered in the accent color."""
    clean = re.sub(r"[^a-zA-Z0-9]", "", word)
    # Numbers (e.g. "10", "30", "100")
    if re.fullmatch(r"\d+", clean):
        return True
    # Accent keyword list
    if clean.lower() in _ACCENT_KEYWORDS:
        return True
    return False


def draw_multiline_accented(draw, lines, font, cx, start_y,
                            fill_white, fill_accent,
                            shadow_fill=(0, 0, 0, 130),
                            shadow_offset=3, line_gap_ratio=0.12):
    """
    Like draw_multiline but renders individual words in accent color
    when _word_is_accent() is True.
    Measures each word's width to position them correctly on the centered line.
    Returns bottom y of the last line.
    """
    lh = draw.textbbox((0, 0), "A", font=font)[3]
    gap = int(lh * line_gap_ratio)
    space_w = draw.textbbox((0, 0), " ", font=font)[2]

    y = start_y
    for line in lines:
        words = line.split()
        # Total line width for centering
        line_w = draw.textbbox((0, 0), line, font=font)[2]
        x = cx - line_w // 2
        mid_y = y + lh // 2

        for i, word in enumerate(words):
            ww = draw.textbbox((0, 0), word, font=font)[2]
            wx = x + ww // 2   # center of this word
            color = fill_accent if _word_is_accent(word) else fill_white
            # shadow
            draw.text((wx + shadow_offset, mid_y + shadow_offset), word,
                      fill=shadow_fill, font=font, anchor="mm")
            # text
            draw.text((wx, mid_y), word,
                      fill=color, font=font, anchor="mm")
            x += ww + space_w

        y += lh + gap

    return y


# ── COLOR ──────────────────────────────────────────────────────────────────────

def _rgb_to_hsv(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    df = mx - mn
    h = 0.0
    if df:
        if mx == r:   h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g: h = (60 * ((b - r) / df) + 120) % 360
        else:         h = (60 * ((r - g) / df) + 240) % 360
    s = 0.0 if mx == 0 else df / mx
    return h, s, mx


def _boost_color(r, g, b, min_s=0.80, min_v=0.90):
    h, s, v = _rgb_to_hsv(r, g, b)
    s = max(s, min_s)
    v = max(v, min_v)
    if s == 0:
        c = int(v * 255)
        return c, c, c
    i = int(h / 60) % 6
    f = h / 60 - int(h / 60)
    p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    rgb = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
    return tuple(int(x * 255) for x in rgb)


def extract_accent_color(image_source, n_clusters=10):
    """
    image_source: file path (str) or bytes-like / PIL Image.
    Returns (R, G, B, 255) accent color extracted from the cover.
    """
    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    elif isinstance(image_source, (bytes, bytearray, io.BytesIO)):
        buf = image_source if isinstance(image_source, io.BytesIO) else io.BytesIO(image_source)
        img = Image.open(buf).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    w, h = img.size
    top    = np.array(img.crop((0, 0,      w, h // 3)).resize((80, 40))).reshape(-1, 3)
    bottom = np.array(img.crop((0, h*2//3, w, h     )).resize((80, 40))).reshape(-1, 3)
    pixels = np.vstack([top, bottom]).astype(np.float32)

    bright = pixels.mean(axis=1)
    pixels = pixels[(bright > 25) & (bright < 235)]
    if len(pixels) < n_clusters:
        pixels = np.vstack([top, bottom]).astype(np.float32)

    k = min(n_clusters, len(pixels))
    km = KMeans(n_clusters=k, n_init=6, random_state=0)
    km.fit(pixels)
    counts = np.bincount(km.labels_)

    best_score = -1
    best_rgb   = (254, 44, 85)

    for i, center in enumerate(km.cluster_centers_):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        _, s, v = _rgb_to_hsv(r, g, b)
        weight  = counts[i] / len(pixels)
        score   = (s ** 1.5) * v * weight
        if s > 0.25:
            score *= 1.4
        if score > best_score:
            best_score = score
            best_rgb   = (r, g, b)

    boosted = _boost_color(*best_rgb)
    return (*boosted, 255)


# ── DRAW HELPERS ───────────────────────────────────────────────────────────────

def draw_gradient_rect(draw, x0, y0, x1, y1, color_top, color_bottom, steps=80):
    h = y1 - y0
    for i in range(steps):
        t  = i / steps
        r  = int(color_top[0] + (color_bottom[0] - color_top[0]) * t)
        g  = int(color_top[1] + (color_bottom[1] - color_top[1]) * t)
        b  = int(color_top[2] + (color_bottom[2] - color_top[2]) * t)
        a  = int(color_top[3] + (color_bottom[3] - color_top[3]) * t)
        y  = int(y0 + h * i / steps)
        y2 = int(y0 + h * (i + 1) / steps)
        draw.rectangle([(x0, y), (x1, y2)], fill=(r, g, b, a))


def draw_diamond_divider(draw, cx, y, half_width, color, line_w=2):
    ds = 6
    draw.line([(cx - half_width, y), (cx - ds - 4, y)], fill=color, width=line_w)
    draw.line([(cx + ds + 4, y),     (cx + half_width, y)], fill=color, width=line_w)
    draw.polygon([(cx, y-ds),(cx+ds, y),(cx, y+ds),(cx-ds, y)], fill=color)


# ── FIT TO TIKTOK ─────────────────────────────────────────────────────────────

def fit_to_tiktok(image_source):
    """
    image_source: file path (str), bytes, or PIL Image.
    Returns (RGBA canvas, fx, fy, fw, fh).
    """
    if isinstance(image_source, Image.Image):
        src = image_source.convert("RGB")
    elif isinstance(image_source, (bytes, bytearray, io.BytesIO)):
        buf = image_source if isinstance(image_source, io.BytesIO) else io.BytesIO(image_source)
        src = Image.open(buf).convert("RGB")
    else:
        src = Image.open(image_source).convert("RGB")

    sw, sh = src.size
    canvas = Image.new("RGB", (TIKTOK_W, TIKTOK_H))

    bg_scale = max(TIKTOK_W / sw, TIKTOK_H / sh)
    bg_w, bg_h = int(sw * bg_scale), int(sh * bg_scale)
    bg = src.resize((bg_w, bg_h), Image.LANCZOS)
    bx = (bg_w - TIKTOK_W) // 2
    by = (bg_h - TIKTOK_H) // 2
    bg = bg.crop((bx, by, bx + TIKTOK_W, by + TIKTOK_H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
    dark = Image.new("RGB", (TIKTOK_W, TIKTOK_H), (0, 0, 0))
    bg = Image.blend(bg, dark, alpha=0.45)
    canvas.paste(bg, (0, 0))

    fit_scale = min(TIKTOK_W / sw, TIKTOK_H / sh)
    fw, fh = int(sw * fit_scale), int(sh * fit_scale)
    front = src.resize((fw, fh), Image.LANCZOS)
    fx = (TIKTOK_W - fw) // 2
    fy = (TIKTOK_H - fh) // 2
    canvas.paste(front, (fx, fy))

    return canvas.convert("RGBA"), fx, fy, fw, fh


# ── GAME SLIDE ─────────────────────────────────────────────────────────────────

def create_game_slide(image_source, game_title, year, accent_color=None):
    """
    Returns JPEG bytes of a 1080×1920 game slide.
    image_source: path, bytes, or PIL Image.
    """
    if accent_color is None:
        accent_color = extract_accent_color(image_source)

    img, fx, fy, fw, fh = fit_to_tiktok(image_source)
    W, H = TIKTOK_W, TIKTOK_H

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    cover_cx = fx + fw // 2
    cover_cy = fy + fh // 2

    draw_gradient_rect(draw, fx, fy + int(fh * 0.50), fx + fw, fy + fh,
                       (0, 0, 0, 0), (0, 0, 0, 200))

    band_h = int(fh * 0.28)
    by0    = cover_cy - band_h // 2
    by1    = cover_cy + band_h // 2
    draw_gradient_rect(draw, fx, by0, fx + fw, by1,
                       (8, 8, 14, 185), (8, 8, 14, 215))

    draw.rectangle([(fx,          by0), (fx + 5,      by1)], fill=accent_color)
    draw.rectangle([(fx + fw - 5, by0), (fx + fw,     by1)], fill=accent_color)

    bc = (*accent_color[:3], 200)
    draw.line([(fx, by0), (fx + fw, by0)], fill=bc, width=2)
    draw.line([(fx, by1), (fx + fw, by1)], fill=bc, width=2)

    cx, cy   = cover_cx, cover_cy
    padding  = 56
    title_text = game_title.upper()
    year_text  = str(year)

    max_title_w = fw - padding * 2

    # ── Find font size that fits BOTH width and band height ──────────────────
    # Band height is fixed at 28% of cover height. Reserve space inside it:
    #   vertical padding (10% top+bottom) + year row + divider gap.
    band_h_px   = band_h   # already computed above
    band_pad_v  = int(band_h_px * 0.08)   # 8% breathing room top & bottom
    year_reserve = int(band_h_px * 0.20)  # ~20% for year + divider
    max_text_h  = band_h_px - band_pad_v * 2 - year_reserve

    font_title, title_size, title_lines = wrap_text(
        draw, title_text, FONT_BOLD, max_title_w,
        max_size=160, min_size=28, max_lines=3,
    )

    # Shrink further if the text block is taller than available space
    for _ in range(80):
        title_block_h, lh, lg = multiline_height(draw, title_lines, font_title)
        if title_block_h <= max_text_h:
            break
        title_size -= 2
        if title_size < 28:
            title_size = 28
            break
        font_title = load_font(FONT_BOLD, title_size)
        _, _, title_lines = wrap_text(
            draw, title_text, FONT_BOLD, max_title_w,
            max_size=title_size, min_size=28, max_lines=3,
        )

    title_block_h, lh, lg = multiline_height(draw, title_lines, font_title)

    font_year = load_font(FONT_BOLD, max(24, title_size // 3))
    tb_y   = draw.textbbox((0, 0), year_text, font=font_year)
    year_h = tb_y[3] - tb_y[1]

    gap     = max(10, title_size // 10)
    total_h = title_block_h + gap + year_h
    start_y = cy - total_h // 2

    shadow = max(2, title_size // 45)
    draw_multiline(
        draw, title_lines, font_title, cx, start_y,
        fill=(255, 255, 255, 255),
        shadow_fill=(0, 0, 0, 110),
        shadow_offset=shadow,
    )

    draw_diamond_divider(draw, cx, start_y + title_block_h + gap // 2,
                         int(fw * 0.28), (*accent_color[:3], 200), line_w=2)

    yy = start_y + title_block_h + gap + year_h // 2
    draw.text((cx, yy), year_text,
              fill=accent_color, font=font_year, anchor="mm")

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=97)
    return buf.getvalue()


# ── INTRO SLIDE ────────────────────────────────────────────────────────────────

def create_intro_slide(image_source, title_text, accent_color=None):
    """
    Returns JPEG bytes of a 1080×1920 intro slide.
    image_source: path, bytes, or PIL Image.
    """
    if accent_color is None:
        accent_color = extract_accent_color(image_source)

    if isinstance(image_source, Image.Image):
        src = image_source.convert("RGB")
    elif isinstance(image_source, (bytes, bytearray, io.BytesIO)):
        buf = image_source if isinstance(image_source, io.BytesIO) else io.BytesIO(image_source)
        src = Image.open(buf).convert("RGB")
    else:
        src = Image.open(image_source).convert("RGB")

    sw, sh = src.size
    W, H = TIKTOK_W, TIKTOK_H

    bg_scale = max(W / sw, H / sh)
    bg = src.resize((int(sw * bg_scale), int(sh * bg_scale)), Image.LANCZOS)
    bx = (bg.width  - W) // 2
    by = (bg.height - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=18))
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    bg = Image.blend(bg, dark, alpha=0.55)

    fit_scale = min(W / sw, (H * 0.55) / sh)
    fw, fh = int(sw * fit_scale), int(sh * fit_scale)
    front = src.resize((fw, fh), Image.LANCZOS)
    fx = (W - fw) // 2
    fy = int(H * 0.04)
    bg.paste(front, (fx, fy))

    canvas  = bg.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    grad_y = fy + fh - int(fh * 0.25)
    draw_gradient_rect(draw, 0, grad_y, W, H,
                       (0, 0, 0, 0), (0, 0, 0, 230))

    line_y = fy + fh + int(H * 0.015)
    draw.line([(W // 2 - 160, line_y), (W // 2 + 160, line_y)],
              fill=(*accent_color[:3], 220), width=3)

    cx = W // 2
    text_area_top    = line_y + int(H * 0.025)
    text_area_bottom = H - int(H * 0.06)
    text_area_h      = text_area_bottom - text_area_top

    # Intro: full topic, word-wrapped, accent on numbers/keywords
    # Cap max_size so text stays compact (≤ 35% of height)
    max_font = min(180, int(H * 0.095))
    font_title, title_size, title_lines = wrap_text(
        draw, title_text.upper(), FONT_BOLD, W - 100,
        max_size=max_font, min_size=52, max_lines=4,
    )

    title_block_h, lh, lg = multiline_height(draw, title_lines, font_title,
                                              line_gap_ratio=0.12)
    # Vertically center in lower portion
    start_y = text_area_top + (text_area_h - title_block_h) // 2

    shadow = max(3, title_size // 32)
    draw_multiline_accented(
        draw, title_lines, font_title, cx, start_y,
        fill_white=(255, 255, 255, 255),
        fill_accent=accent_color,
        shadow_fill=(0, 0, 0, 130),
        shadow_offset=shadow,
        line_gap_ratio=0.12,
    )

    dot_y = start_y + title_block_h + int(H * 0.022)
    for dx in [-24, 0, 24]:
        r2 = 6 if dx == 0 else 4
        draw.ellipse([(cx + dx - r2, dot_y - r2), (cx + dx + r2, dot_y + r2)],
                     fill=(*accent_color[:3], 200 if dx == 0 else 120))

    draw.rectangle([(0, 0), (4, H)], fill=(*accent_color[:3], 160))
    draw.rectangle([(W - 4, 0), (W, H)], fill=(*accent_color[:3], 160))

    result = Image.alpha_composite(canvas, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=97)
    return buf.getvalue()


# ── OUTRO SLIDE ────────────────────────────────────────────────────────────────

def create_outro_slide(main_text="What game would you add?",
                       sub_text="Drop it in the comments 👇",
                       accent_color=(254, 44, 85, 255),
                       bg_color=(8, 8, 14)):
    """Returns JPEG bytes of a 1080×1920 outro/CTA slide."""
    W, H = TIKTOK_W, TIKTOK_H
    canvas = Image.new("RGBA", (W, H), (*bg_color, 255))
    draw   = ImageDraw.Draw(canvas)

    cx, cy = W // 2, H // 2
    ar, ag, ab, _ = accent_color

    r_big = int(W * 0.72)
    draw.ellipse([(cx - r_big, cy - r_big), (cx + r_big, cy + r_big)],
                 fill=(ar, ag, ab, 18))
    r_mid = int(W * 0.48)
    draw.ellipse([(cx - r_mid, cy - r_mid), (cx + r_mid, cy + r_mid)],
                 fill=(ar, ag, ab, 28))

    draw.rectangle([(0, 0),     (5, H)], fill=(*accent_color[:3], 200))
    draw.rectangle([(W - 5, 0), (W, H)], fill=(*accent_color[:3], 200))

    for ly in [int(H * 0.18), int(H * 0.82)]:
        draw_diamond_divider(draw, cx, ly, int(W * 0.38),
                             (*accent_color[:3], 180), line_w=2)

    lines = main_text.split("\n")
    font_main, main_size = fit_text_size(draw, max(lines, key=len).upper(),
                                         FONT_BOLD, W - 100,
                                         max_size=200, min_size=60)
    font_sub = load_font(FONT_BOLD, max(34, main_size // 4))

    line_h_val = draw.textbbox((0, 0), "A", font=font_main)
    lh       = line_h_val[3] - line_h_val[1]
    line_gap = int(lh * 0.15)
    total_main_h = lh * len(lines) + line_gap * (len(lines) - 1)

    tb_sub = draw.textbbox((0, 0), sub_text, font=font_sub)
    sub_h  = tb_sub[3] - tb_sub[1]

    block_gap = int(H * 0.04)
    total_h   = total_main_h + block_gap + sub_h
    start_y   = cy - total_h // 2

    for i, line in enumerate(lines):
        ly     = start_y + i * (lh + line_gap) + lh // 2
        shadow = max(3, main_size // 30)
        draw.text((cx + shadow, ly + shadow), line.upper(),
                  fill=(0, 0, 0, 100), font=font_main, anchor="mm")
        draw.text((cx, ly), line.upper(),
                  fill=(255, 255, 255, 255), font=font_main, anchor="mm")

    sub_y = start_y + total_main_h + block_gap + sub_h // 2
    draw.text((cx, sub_y), sub_text,
              fill=accent_color, font=font_sub, anchor="mm")

    result = canvas.convert("RGB")
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=97)
    return buf.getvalue()


# ── STEAMGRIDDB API ────────────────────────────────────────────────────────────

def _sgdb_headers():
    return {"Authorization": f"Bearer {STEAMGRIDDB_API_KEY}"}


def get_game_data(game_name):
    url = f"https://www.steamgriddb.com/api/v2/search/autocomplete/{requests.utils.quote(game_name)}"
    try:
        res = requests.get(url, headers=_sgdb_headers(), timeout=10).json()
        if res.get("success") and res.get("data"):
            d = res["data"][0]
            return d["id"], d["name"]
    except Exception as e:
        print(f"  Search error: {e}")
    return None, None


def get_game_cover_bytes(game_id):
    url = f"https://www.steamgriddb.com/api/v2/grids/game/{game_id}"
    for dim in COVER_DIMENSIONS:
        params = {
            "dimensions": dim,
            "mimes": "image/png,image/jpeg,image/webp",
            "limit": 5,
        }
        try:
            res = requests.get(url, headers=_sgdb_headers(), params=params, timeout=10).json()
            if res.get("success") and res.get("data"):
                best = max(res["data"], key=lambda x: x.get("width", 0) * x.get("height", 0))
                cover_bytes = requests.get(best["url"], timeout=15).content
                print(f"  Cover: {best.get('width')}×{best.get('height')} px")
                return cover_bytes
        except Exception as e:
            print(f"  Cover error ({dim}): {e}")
    return None


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────────

def generate_carousel(topic: str, games: list[dict]) -> list[bytes]:
    """
    Main entry point.

    Args:
        topic: carousel topic string, e.g. "30 story games"
        games: list of dicts with keys "name" (str) and "year" (str | int)
               e.g. [{"name": "Elden Ring", "year": "2022"}, ...]

    Returns:
        List of JPEG bytes: [intro, *game_slides, outro]
        May be shorter if some games are not found.
    """
    print(f"[generate_carousel] topic='{topic}', {len(games)} games")
    slides: list[bytes] = []

    # Resolve all games first (name → id → cover bytes)
    resolved: list[dict] = []   # {name, year, cover_bytes, accent}
    for g in games:
        game_id, official_name = get_game_data(g["name"])
        if not game_id:
            print(f"  ✗ Not found: {g['name']}")
            continue
        cover = get_game_cover_bytes(game_id)
        if not cover:
            print(f"  ✗ No cover: {g['name']}")
            continue
        accent = extract_accent_color(cover)
        resolved.append({
            "name":         official_name,
            "year":         str(g.get("year", "")),
            "cover_bytes":  cover,
            "accent":       accent,
        })
        print(f"  ✓ Resolved: {official_name}")

    if not resolved:
        raise ValueError("No games could be resolved from the provided list.")

    # Pick two distinct random games for intro / outro backgrounds
    intro_game = random.choice(resolved)
    outro_candidates = [g for g in resolved if g["name"] != intro_game["name"]]
    outro_game = random.choice(outro_candidates) if outro_candidates else intro_game

    # ── Intro slide ──
    print("  ── Generating intro slide...")
    intro_bytes = create_intro_slide(
        intro_game["cover_bytes"],
        topic.strip(),          # full topic, word-wrapped automatically
        accent_color=intro_game["accent"],
    )
    slides.append(intro_bytes)

    # ── Game slides ──
    for g in resolved:
        print(f"  ── Game slide: {g['name']}...")
        slide_bytes = create_game_slide(
            g["cover_bytes"],
            g["name"],
            g["year"],
            accent_color=g["accent"],
        )
        slides.append(slide_bytes)

    # ── Outro slide ──
    print("  ── Generating outro slide...")
    outro_accent = outro_game["accent"]
    outro_bytes = create_outro_slide(
        main_text="What game\nwould you add?",
        sub_text="Drop it in the comments 👇",
        accent_color=outro_accent,
    )
    slides.append(outro_bytes)

    print(f"  ✓ Carousel ready: {len(slides)} slides total")
    return slides


# ── CLI USAGE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR = "tiktok_carousel"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TOPIC = "30 story games you must play"
    GAMES = [
        {"name": "Elden Ring",          "year": "2022"},
        {"name": "Voices of the Void",  "year": "2023"},
        {"name": "Hollow Knight",       "year": "2017"},
        {"name": "Disco Elysium",       "year": "2019"},
    ]

    slides = generate_carousel(TOPIC, GAMES)
    labels = ["00_intro"] + [f"{i+1:02d}_game" for i in range(len(slides) - 2)] + ["99_outro"]

    for label, data in zip(labels, slides):
        path = os.path.join(OUTPUT_DIR, f"{label}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path}")
