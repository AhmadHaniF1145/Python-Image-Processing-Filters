"""
╔══════════════════════════════════════════════════════════════╗
║    VIRAL PHOTOGRAPHIC EFFECTS — Instagram / TikTok Style     ║
║    Mata Kuliah: Pengolahan Citra Digital                     ║
╠══════════════════════════════════════════════════════════════╣
║  [1] Teal & Orange    — Hollywood cinematic (paling viral)   ║
║  [2] VHS / Retro      — Kaset video jadul 90s                ║
║  [3] Glitch           — Chromatic aberration split RGB       ║
║  [4] Vintage Film     — Kodak Portra analog simulation       ║
║  [5] Duotone          — Spotify-style two-color gradient     ║
║  [6] Dreamy / Aura    — Soft glow + pastel ethereal          ║
║  [7] Neon Cyberpunk   — Dark + neon edge glow                ║
║  [8] Bleach Bypass    — Cinematic desaturated high contrast  ║
║  [9] Lo-Fi            — Faded, grainy, warm aesthetic        ║
║  [0] Golden Hour      — Enhanced sunset warm light           ║
║  [Q] Film Burn        — Light leak + burn analog             ║
║  [W] Matrix Green     — Digital rain look                    ║
║  [E] Holographic      — RGB rainbow shift iridescent         ║
║  [R] Oil Paint        — Painterly brush stroke effect        ║
║  [T] Noir             — Classic black & white cinematic      ║
╠══════════════════════════════════════════════════════════════╣
║  [A]/[D]  Intensitas −/+                                     ║
║  [S]      Screenshot    [V] Record    [ESC] Keluar            ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import os
import math
import random
from datetime import datetime

os.makedirs("output", exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════

def make_lut(fn):
    """Buat Look-Up Table 256 nilai dari fungsi transformasi."""
    return np.array([np.clip(fn(i), 0, 255) for i in range(256)], dtype=np.uint8)

def apply_lut(channel, lut):
    return cv2.LUT(channel, lut)

def blend(a, b, t):
    """Linear blend: (1-t)*a + t*b"""
    return cv2.addWeighted(a, 1.0 - t, b, t, 0)

def to_float(img):
    return img.astype(np.float32) / 255.0

def to_uint8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def add_grain(frame, amount=0.04):
    """
    Film Grain — noise acak layaknya film analog.
    Teknik: Gaussian noise → ditambah ke float frame.
    Standar deviasi (sigma) kontrol seberapa kasar grainnya.
    """
    f = to_float(frame)
    noise = np.random.normal(0, amount, f.shape).astype(np.float32)
    return to_uint8(np.clip(f + noise, 0, 1))

def add_vignette(frame, strength=0.6, sigma_ratio=0.55):
    """
    Vignette — penggelapan sudut frame.
    Gaussian mask 2D: mask = outer_product(gauss_y, gauss_x)
    Dibalik (1 - mask) lalu dikalikan ke frame.
    """
    h, w = frame.shape[:2]
    gx = cv2.getGaussianKernel(w, int(w * sigma_ratio))
    gy = cv2.getGaussianKernel(h, int(h * sigma_ratio))
    mask = gy @ gx.T
    mask = (mask / mask.max()).astype(np.float32)
    vignette = 1.0 - strength * (1.0 - mask)
    vignette = np.dstack([vignette] * 3)
    return to_uint8(to_float(frame) * vignette)

def cinematic_bars(frame, bar_h=None):
    """
    Cinematic letterbox — garis hitam atas bawah (2.39:1 ratio).
    Standar aspect ratio film layar lebar Hollywood.
    """
    h, w = frame.shape[:2]
    if bar_h is None:
        bar_h = int(h * 0.09)
    out = frame.copy()
    out[:bar_h]     = 0
    out[h - bar_h:] = 0
    return out

def adjust_curves(frame, r_fn, g_fn, b_fn):
    """
    Color Curves — seperti Curves di Lightroom/Photoshop.
    Masing-masing kanal R, G, B punya kurva transformasinya sendiri.
    LUT dipakai agar tidak perlu loop per-piksel (sangat cepat).
    """
    b, g, r = cv2.split(frame)
    r = apply_lut(r, make_lut(r_fn))
    g = apply_lut(g, make_lut(g_fn))
    b = apply_lut(b, make_lut(b_fn))
    return cv2.merge([b, g, r])

def saturation_shift(frame, factor):
    """
    Ubah saturasi via HSV colorspace.
    S channel dikali faktor: >1 = lebih vivid, <1 = faded/desaturated.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def hue_shift(frame, degrees):
    """Geser hue di colorspace HSV. 0-180 di OpenCV (180 = 360°)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + degrees) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def soft_glow(frame, ksize=0, sigma=20, strength=0.35):
    """
    Soft Glow — blur kuat lalu di-blend ke atas frame asli.
    Hasilnya: highlight 'bleeding' ke area sekitar (dreamy look).
    Screen blend: 1 - (1-a)(1-b) → lebih terang dari additive.
    """
    if ksize == 0:
        ksize = 0
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    f = to_float(frame)
    g = to_float(blurred)
    # Screen blend
    screen = 1.0 - (1.0 - f) * (1.0 - g)
    return to_uint8(blend(f, screen, strength))


# ══════════════════════════════════════════════════════════════
#  15 EFEK VIRAL
# ══════════════════════════════════════════════════════════════

def fx_teal_orange(frame, t=1.0):
    """
    TEAL & ORANGE — Filter sinematik paling viral di dunia.
    Dipakai di hampir semua film Hollywood (Mad Max, Michael Bay, dll).

    Teknik Color Grading:
    • Skin tones (shadows, midtones) → push ke ORANGE (warm)
    • Background, shadows → push ke TEAL (cyan-green)
    Cara: manipulasi kurva R, G, B secara independen
    + Hue shift selective di area cyan dan orange
    + Vignette + cinematic bars untuk finishing.

    Rumus kurva:
      R: slight boost di midtone (orange push)
      G: slight dip di midtone
      B: boost shadows, dip highlights (teal push di gelap)
    """
    # Kurva per kanal (nilai input → nilai output)
    r_fn = lambda i: i * 1.08 + 8 * math.sin(i / 255 * math.pi)
    g_fn = lambda i: i * 0.95
    b_fn = lambda i: i * 0.85 + 25 * (1 - i / 255)   # teal di shadow

    graded = adjust_curves(frame, r_fn, g_fn, b_fn)

    # Boost saturasi agar skin dan background lebih pop
    graded = saturation_shift(graded, 1.0 + 0.4 * t)

    # Vignette sinematik
    graded = add_vignette(graded, strength=0.5 * t)

    # Cinematic bars
    graded = cinematic_bars(graded)

    return blend(frame, graded, t)


def fx_vhs(frame, t=1.0):
    """
    VHS / RETRO — Kaset video analog tahun 80-90an.
    Viral karena nostalgia aesthetic di TikTok.

    Teknik:
    1. Chroma shift: kanal U & V di YUV digeser → color bleeding
       (warna 'bocor' ke area sekitar — khas tape magnetik yang aus)
    2. Scan lines: garis horizontal gelap selang-seling
       (CRT TV hanya tampilkan garis ganjil atau genap per frame)
    3. Noise horizontal: setiap baris digeser random kiri/kanan
       (tracking error — pita kaset tidak terbaca sempurna)
    4. Grain + warna faded (saturasi turun)
    5. Timestamp watermark pojok bawah
    """
    h, w = frame.shape[:2]

    # Faded, low saturation
    out = saturation_shift(frame, 0.6 + 0.2 * (1 - t))

    # Chroma bleed: blur hanya kanal chrominance (YCrCb)
    ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    shift = int(t * 8)
    if shift > 0:
        cr = np.roll(cr, shift, axis=1)
        cb = np.roll(cb, -shift, axis=1)
    ycc = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    # Scan lines (CRT)
    mask = np.ones((h, w, 3), dtype=np.float32)
    mask[::2, :] = 1.0 - 0.25 * t
    out = to_uint8(to_float(out) * mask)

    # Horizontal tracking noise (per baris, random shift kecil)
    noisy = out.copy()
    for row in range(0, h, random.randint(20, 60)):
        shift_x = random.randint(-int(5 * t), int(5 * t))
        if abs(shift_x) > 0 and row + 8 < h:
            noisy[row:row+4, :] = np.roll(out[row:row+4, :], shift_x, axis=1)
    out = noisy

    # Grain
    out = add_grain(out, 0.05 * t)

    # Warm tint (tape warna kekuningan)
    out = adjust_curves(out,
        lambda i: min(255, i + 15 * t),
        lambda i: i,
        lambda i: max(0, i - 20 * t))

    # Timestamp (khas kamera VHS)
    ts = time.strftime("%Y %m %d   %H:%M:%S")
    cv2.putText(out, ts, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 80), 1)

    return out


def fx_glitch(frame, t=1.0):
    """
    GLITCH — Chromatic Aberration + Digital Glitch.
    Viral sebagai aesthetic 'broken screen' / cyberpunk di TikTok.

    Teknik:
    1. RGB Split (Chromatic Aberration):
       Setiap kanal R, G, B digeser arah berbeda.
       Ini simulasi lensa atau sensor yang tidak sejajar (lens fringing).
    2. Block glitch: blok-blok frame dipotong dan digeser secara acak
       (simulasi buffer video yang corrupt / packet loss)
    3. Scanline glitch: beberapa baris tiba-tiba bergeser horizontal
    4. Color inversion sesekali di area tertentu
    """
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame)

    # Chromatic aberration: geser R kanan, B kiri
    shift = int(t * 12)
    r_s = np.roll(r, shift,  axis=1)
    b_s = np.roll(b, -shift, axis=1)
    g_s = np.roll(g, int(shift * 0.3), axis=0)  # G slight vertical
    out = cv2.merge([b_s, g_s, r_s])

    # Block glitch (tampil tidak setiap frame agar lebih organik)
    if random.random() < 0.35 * t:
        n_blocks = random.randint(1, int(4 * t) + 1)
        for _ in range(n_blocks):
            bh = random.randint(5, 40)
            by = random.randint(0, h - bh - 1)
            bx_shift = random.randint(-int(60 * t), int(60 * t))
            out[by:by+bh] = np.roll(out[by:by+bh], bx_shift, axis=1)

    # Warna block: sesekali inversi warna blok kecil
    if random.random() < 0.2 * t:
        bh = random.randint(3, 15)
        by = random.randint(0, h - bh - 1)
        out[by:by+bh] = cv2.bitwise_not(out[by:by+bh])

    # Scan line dropout: baris hitam sesekali
    if random.random() < 0.4 * t:
        for _ in range(random.randint(1, 3)):
            ry = random.randint(0, h - 1)
            out[ry] = 0

    return out


def fx_vintage_film(frame, t=1.0):
    """
    VINTAGE FILM — Kodak Portra 400 Simulation.
    Analog film photography aesthetic yang super viral di IG.

    Kodak Portra karakteristiknya:
    • Skin tones warm & creamy (push Red + Yellow)
    • Shadow kebiruan/kehijauan (lift shadows)
    • Highlight tidak terlalu putih (roll-off yang halus)
    • Kontras sedang, saturasi tidak berlebihan
    • Grain yang terasa organik

    Teknik:
    1. Shadow lift: shadows tidak pernah benar-benar hitam
       (S-curve dengan toe yang diangkat)
    2. Highlight roll-off: kurva melengkung di atas
    3. Color shift: warm midtone + cool shadow
    4. Organik grain
    5. Vignette halus
    """
    # S-curve dengan shadow lift (Kodak signature)
    def s_curve_r(i):
        x = i / 255.0
        # Shadow lift + warm push
        y = x * 0.85 + 0.08 + 0.12 * math.sin(x * math.pi)
        return y * 255

    def s_curve_g(i):
        x = i / 255.0
        y = x * 0.9 + 0.05 + 0.05 * math.sin(x * math.pi)
        return y * 255

    def s_curve_b(i):
        x = i / 255.0
        # Cool shadows, slight green-blue push
        y = x * 0.78 + 0.06 + 0.04 * math.sin(x * math.pi)
        return y * 255

    graded = adjust_curves(frame, s_curve_r, s_curve_g, s_curve_b)

    # Saturasi dikurangi sedikit (film tidak segarang digital)
    graded = saturation_shift(graded, 0.75 + 0.15 * (1 - t))

    # Grain organik
    graded = add_grain(graded, 0.032 * t)

    # Vignette halus
    graded = add_vignette(graded, strength=0.38 * t)

    return blend(frame, graded, t)


def fx_duotone(frame, t=1.0):
    """
    DUOTONE — Filter dua warna ala Spotify / graphic design viral.
    Populer banget di konten editorial dan music content di IG.

    Teknik:
    1. Grayscale → nilai luminance 0-255
    2. Map shadow ke color_dark, highlight ke color_light
    3. Interpolasi linear antara dua warna berdasarkan brightness:
       output = lerp(color_dark, color_light, luma/255)
    Hasilnya: gambar hanya punya 2 warna dominan = sangat graphic.
    """
    # Pasangan warna duotone paling viral
    # (dark_color, light_color) dalam BGR
    palettes = [
        ((120, 20, 180),  (255, 200, 50)),   # Purple → Gold
        ((180, 30, 20),   (50, 220, 255)),   # Deep Red → Cyan
        ((20, 80, 200),   (240, 230, 50)),   # Dark Blue → Yellow
        ((10, 120, 30),   (220, 60, 200)),   # Dark Green → Pink
    ]
    # Ganti palet setiap 4 detik biar variatif
    idx = int(time.time() / 4) % len(palettes)
    dark, light = palettes[idx]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Buat gambar duotone
    out = np.zeros_like(frame, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = dark[c] / 255.0 * (1 - gray) + light[c] / 255.0 * gray

    result = to_uint8(out)
    return blend(frame, result, t)


def fx_dreamy(frame, t=1.0):
    """
    DREAMY / AURA — Ethereal soft pastel glow.
    Sangat viral di TikTok untuk konten aesthetic, fashion, art.

    Teknik:
    1. Pastel shift: warna didorong ke arah putih (wash out)
       Rumus: pastel = original * factor + white * (1-factor)
    2. Soft glow: screen blend antara frame dan blur-nya
       Screen blend: 1 - (1-a)(1-b) — lebih terang dari overlay
    3. Chromatic haze: R channel diblur lebih besar dari B → color fringe
    4. Highlight bloom: area terang bleeding ke sekitarnya
    """
    # Pastel wash
    white = np.ones_like(frame, dtype=np.float32)
    f = to_float(frame)
    pastel = f * (0.5 + 0.4 * t) + white * (0.15 * t)
    pastel = np.clip(pastel, 0, 1)

    out = to_uint8(pastel)

    # Chromatic soft glow (per channel blur berbeda)
    b, g, r = cv2.split(out)
    sigma = int(t * 18) + 5
    r_blur = cv2.GaussianBlur(r, (0, 0), sigma)
    b_blur = cv2.GaussianBlur(b, (0, 0), sigma // 2)
    g_blur = cv2.GaussianBlur(g, (0, 0), sigma * 2 // 3)

    # Screen blend per channel
    def screen_1ch(a, b_ch):
        fa = a.astype(np.float32) / 255.0
        fb = b_ch.astype(np.float32) / 255.0
        return np.clip((1 - (1 - fa) * (1 - fb)) * 255, 0, 255).astype(np.uint8)

    strength = 0.4 * t
    r_out = cv2.addWeighted(r, 1 - strength, screen_1ch(r, r_blur), strength, 0)
    g_out = cv2.addWeighted(g, 1 - strength, screen_1ch(g, g_blur), strength, 0)
    b_out = cv2.addWeighted(b, 1 - strength, screen_1ch(b, b_blur), strength, 0)

    result = cv2.merge([b_out, g_out, r_out])

    # Sedikit hue shift ke arah pink/lavender
    result = hue_shift(result, int(10 * t))

    # Vignette sangat halus
    result = add_vignette(result, strength=0.25 * t, sigma_ratio=0.7)

    return result


def fx_neon_cyberpunk(frame, t=1.0):
    """
    NEON CYBERPUNK — Dark city neon glow aesthetic.
    Viral untuk konten gaming, music, urban photography.

    Teknik:
    1. Darken: frame digelapkan drastis (shadows crushed)
    2. Edge glow: Canny edge detection → edge diberi warna neon
       Edge glow pakai Gaussian blur setelah deteksi
       → membuat edge 'bersinar' seperti lampu neon
    3. Neon color: warna edge di-cycle antara cyan, magenta, pink
    4. Boost saturasi ekstrem di area terang
    5. Tambah warna dingin (cyan-blue tint)
    """
    # Crush shadows
    f = to_float(frame)
    dark = np.power(f, 1.5 + t * 0.5)   # gamma > 1 = gelap
    dark = to_uint8(dark)

    # Deteksi edge (Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur dulu biar edge lebih bersih
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 60, 160)

    # Buat glow dari edge: blur edges → warna neon
    glow = cv2.GaussianBlur(edges, (0, 0), 6 + t * 8)

    # Warna neon berubah seiring waktu (cycle)
    cycle = (time.time() * 0.5) % 3
    if cycle < 1:
        neon_color = np.array([255 * (1-cycle), 50, 255 * cycle], dtype=np.float32)
    elif cycle < 2:
        c = cycle - 1
        neon_color = np.array([50, 255 * c, 255 * (1-c)], dtype=np.float32)
    else:
        c = cycle - 2
        neon_color = np.array([255 * c, 255 * (1-c), 50], dtype=np.float32)

    # Aplikasikan glow ke frame gelap
    glow_f = glow.astype(np.float32) / 255.0
    out_f  = to_float(dark)
    for c in range(3):
        out_f[:, :, c] += glow_f * (neon_color[c] / 255.0) * t * 1.2
    out = to_uint8(np.clip(out_f, 0, 1))

    # Cool tint
    out = adjust_curves(out,
        lambda i: max(0, i - 10 * t),
        lambda i: i,
        lambda i: min(255, i + 20 * t))

    return out


def fx_bleach_bypass(frame, t=1.0):
    """
    BLEACH BYPASS — Teknik darkroom analog paling ikonik.
    Dipakai di film: Saving Private Ryan, 1917, Minority Report.
    Viral karena feel 'perang/apocalyptic/raw' yang kuat.

    Teknik Darkroom:
    Proses cuci (bleach) film dilewati (di-bypass) →
    silver halide tetap di emulsi → mengurangi saturasi
    sambil meningkatkan kontras drastis.

    Simulasi digital:
    1. Grayscale = luminance layer
    2. Hard Light blend: grayscale di-blend ke original
       Hard light: gelap*gelap*2, terang = screen blend
    3. Kontras S-curve ekstrem
    4. Desaturasi
    5. Vignette keras + cinematic bars
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Hard light blend: original sebagai base, grayscale sebagai overlay
    f = to_float(frame)
    g = to_float(gray_bgr)

    # Hard light formula per piksel
    hard = np.where(g < 0.5,
                    2 * f * g,
                    1 - 2 * (1 - f) * (1 - g))
    hard = to_uint8(np.clip(hard, 0, 1))

    # S-curve kontras tinggi
    def s_hi(i):
        x = i / 255.0
        y = 0.5 + 1.5 * (x - 0.5) + 0.3 * math.sin((x - 0.5) * math.pi)
        return np.clip(y * 255, 0, 255)

    hard = adjust_curves(hard, s_hi, s_hi, s_hi)

    # Desaturasi (bleach mengurangi saturasi)
    hard = saturation_shift(hard, 0.2 + 0.3 * (1 - t))

    hard = add_vignette(hard, strength=0.7 * t)
    hard = cinematic_bars(hard)

    return blend(frame, hard, t)


def fx_lofi(frame, t=1.0):
    """
    LO-FI — Faded, warm, grainy aesthetic.
    Definisi Lo-Fi photography: 'imperfection is perfection'.
    Viral sebagai aesthetic tren Study/Chill di TikTok & YouTube.

    Karakteristik:
    • Highlight faded (tidak pernah benar-benar putih)
    • Shadow lifted (tidak pernah benar-benar hitam)
    • Warm, slightly yellowish tone
    • Heavy grain
    • Slight color shift: green-yellow di shadow

    Teknik: kurva fade (output tidak pernah mencapai 0 atau 255)
    + warm LUT + heavy grain
    """
    # Fade curve: shadow lift + highlight roll-off
    def fade(i):
        x = i / 255.0
        y = 0.08 + x * 0.82   # range 0.08 - 0.90 (tidak pernah pure black/white)
        return y * 255

    out = adjust_curves(frame,
        lambda i: min(255, fade(i) + 15 * t),   # warm red push
        lambda i: min(255, fade(i) + 8  * t),   # slight green
        lambda i: max(0,   fade(i) - 18 * t))   # cool blue reduction

    # Saturasi faded
    out = saturation_shift(out, 0.65 + 0.2 * (1 - t))

    # Heavy grain (ciri khas lo-fi)
    out = add_grain(out, 0.065 * t)

    # Vignette halus
    out = add_vignette(out, strength=0.3 * t)

    return blend(frame, out, t)


def fx_golden_hour(frame, t=1.0):
    """
    GOLDEN HOUR — Cahaya matahari terbenam yang hangat.
    Paling viral untuk portrait & outdoor content di IG.

    Golden hour asli: matahari rendah → cahaya merah/oranye memanjang
    → bayangan panjang, specular highlight hangat, suasana magis.

    Teknik:
    1. Boost Red channel ekstrem (terutama highlight)
    2. Midtone orange push (Red + Green, kurangi Blue)
    3. Highlight bloom: area terang makin bersinar warm
    4. Vignette hangat (bukan hitam biasa, tapi dark orange)
    5. Slight haze di area terang (golden atmosphere)
    """
    f = to_float(frame)

    # Orange-gold push
    r_boost = np.clip(f[:, :, 2] * (1 + 0.35 * t) + 0.08 * t, 0, 1)
    g_mid   = np.clip(f[:, :, 1] * (1 + 0.10 * t), 0, 1)
    b_cut   = np.clip(f[:, :, 0] * (1 - 0.30 * t), 0, 1)

    out = to_uint8(np.dstack([b_cut, g_mid, r_boost]))

    # Highlight bloom warm: area terang dibuat makin warm
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    highlight_mask = np.power(gray, 2.5)   # hanya area sangat terang
    highlight_mask = np.dstack([highlight_mask] * 3)
    warm_add = np.zeros_like(f)
    warm_add[:, :, 2] = 0.3 * t   # add red ke highlight
    warm_add[:, :, 1] = 0.1 * t
    out_f = to_float(out) + warm_add * highlight_mask
    out = to_uint8(np.clip(out_f, 0, 1))

    # Saturasi sedikit naik (golden hour warna vivid)
    out = saturation_shift(out, 1 + 0.3 * t)

    # Slight haze (golden atmosphere — blur terang)
    haze = cv2.GaussianBlur(out, (0, 0), 15)
    gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    haze_mask = np.dstack([gray2] * 3)
    out_f2 = to_float(out) * (1 - 0.15 * t * haze_mask) + \
             to_float(haze) * 0.15 * t * haze_mask
    out = to_uint8(np.clip(out_f2, 0, 1))

    out = add_vignette(out, strength=0.35 * t)
    out = cinematic_bars(out)

    return out


def fx_film_burn(frame, t=1.0):
    """
    FILM BURN / LIGHT LEAK — Kebocoran cahaya di kamera analog.
    Sangat viral di IG untuk foto portrait & aesthetic feed.

    Film burn terjadi ketika cahaya masuk ke kamar gelap kamera
    dan membakar emulsi film → efek oranye/merah/putih yang indah.

    Teknik simulasi:
    1. Base: Vintage Film tone
    2. Light leak overlay: gradient radial oranye di pojok
       • Dibuat dari Gaussian blur titik terang di luar frame
       • Beberapa sumber cahaya di posisi berbeda
    3. Blend dengan Screen blend mode (brighten only)
    4. Horizontal streak: garis cahaya tipis di 1/3 frame
    5. Grain analog
    """
    h, w = frame.shape[:2]

    # Base vintage tone
    base = fx_vintage_film(frame, t * 0.6)

    # Buat light leak canvas
    leak = np.zeros((h, w, 3), dtype=np.float32)

    # Light source 1: pojok kiri atas (orange-red)
    cx1, cy1 = int(w * 0.1), int(h * 0.05)
    for y in range(h):
        for x in range(0, w, 4):   # step 4 untuk kecepatan
            d = math.sqrt((x - cx1)**2 + (y - cy1)**2)
            falloff = math.exp(-d / (w * 0.35))
            leak[y, x, 2] += 0.7 * falloff * t   # R
            leak[y, x, 1] += 0.3 * falloff * t   # G
            leak[y, x, 0] += 0.1 * falloff * t   # B

    # Light source 2: pojok kanan atas (amber lebih terang)
    cx2, cy2 = int(w * 0.9), int(h * 0.1)
    for y in range(0, h, 2):
        for x in range(w // 2, w, 4):
            d = math.sqrt((x - cx2)**2 + (y - cy2)**2)
            falloff = math.exp(-d / (w * 0.25))
            leak[y, x, 2] += 0.5 * falloff * t
            leak[y, x, 1] += 0.35 * falloff * t
            leak[y, x, 0] += 0.05 * falloff * t

    # Horizontal streak (garis cahaya khas light leak)
    streak_y = int(h * 0.38)
    for y in range(max(0, streak_y - 8), min(h, streak_y + 8)):
        falloff_v = 1 - abs(y - streak_y) / 8
        for x in range(int(w * 0.2), w):
            falloff_h = (x - int(w * 0.2)) / (w * 0.8)
            leak[y, x, 2] = max(leak[y, x, 2], 0.45 * falloff_h * falloff_v * t)
            leak[y, x, 1] = max(leak[y, x, 1], 0.2  * falloff_h * falloff_v * t)

    leak = np.clip(leak, 0, 1)

    # Screen blend: base + light leak
    base_f = to_float(base)
    result = 1 - (1 - base_f) * (1 - leak)
    result = to_uint8(np.clip(result, 0, 1))

    result = add_grain(result, 0.035 * t)
    result = add_vignette(result, strength=0.3 * t)

    return result


def fx_matrix(frame, t=1.0):
    """
    MATRIX GREEN — Digital rain / hacker aesthetic.
    Viral sebagai overlay efek di konten tech/gaming/hacking.

    Teknik:
    1. Grayscale → hanya simpan luminance
    2. Duotone ke matrix green: hitam + hijau terang
    3. Scanline halus
    4. Rain overlay: karakter jatuh dari atas (simulasi character rain)
       Setiap kolom punya kecepatan berbeda
    5. Phosphor glow: layar CRT terminal lama
    """
    h, w = frame.shape[:2]

    # Convert ke 'terminal green' duotone
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    # Green phosphor: dark green shadow, bright green highlight
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[:, :, 1] = gray * (0.7 + 0.3 * t)          # G channel (green)
    out[:, :, 0] = gray * 0.05 * (1 - t * 0.8)     # B sedikit
    out = to_uint8(out)

    # Scanlines CRT
    mask = np.ones((h, w, 3), dtype=np.float32)
    mask[::2] *= (1 - 0.15 * t)
    out = to_uint8(to_float(out) * mask)

    # Phosphor glow
    glow = cv2.GaussianBlur(out, (0, 0), 3)
    out = to_uint8(np.clip(to_float(out) + to_float(glow) * 0.25 * t, 0, 1))

    return blend(frame, out, min(1.0, t * 1.1))


def fx_holographic(frame, t=1.0):
    """
    HOLOGRAPHIC / IRIDESCENT — Rainbow shift color effect.
    Viral di konten fashion, beauty, album art di IG/TikTok.

    Hologram efeknya berasal dari difraksi cahaya:
    sudut berbeda → panjang gelombang berbeda → warna berbeda.

    Simulasi:
    1. Hue shift berbeda untuk setiap baris (berdasarkan posisi Y)
       → menciptakan 'rainbow sweep' dari atas ke bawah
    2. Shift bergerak seiring waktu (animasi)
    3. Blend dengan frame asli → warna asli masih kelihatan
    4. Saturation boost ekstrem
    5. Slight chromatic aberration untuk efek prisma
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue shift per baris — berbeda tiap baris (rainbow sweep)
    time_offset = time.time() * 40   # kecepatan animasi
    hue_range   = 120 * t            # seberapa lebar rainbow

    for y in range(h):
        shift = (time_offset + y * hue_range / h) % 180
        hsv[y, :, 0] = (hsv[y, :, 0] + shift) % 180

    # Saturasi naik
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.8 * t), 0, 255)

    rainbow = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Chromatic aberration (prisma)
    b, g, r = cv2.split(rainbow)
    shift_px = int(t * 5)
    if shift_px > 0:
        r = np.roll(r, shift_px,  axis=1)
        b = np.roll(b, -shift_px, axis=1)
    rainbow = cv2.merge([b, g, r])

    return blend(frame, rainbow, 0.55 * t + 0.2)


def fx_oil_paint(frame, t=1.0):
    """
    OIL PAINT — Efek lukisan cat minyak.
    Viral sebagai artistic filter di TikTok & IG Stories.

    Teknik:
    1. Bilateral filter (edge-preserving smoothing) multi-pass
       Bilateral: smoothing tapi mempertahankan tepi
       Bobot = Gaussian(jarak spasial) × Gaussian(perbedaan warna)
    2. Detail layer: unsharp mask untuk texture
    3. Edge darkening: tepi dibuat lebih gelap (goresan kuas)
    4. Saturasi boost: cat minyak lebih vivid dari foto
    5. Slight warm tone: kanvas cenderung warm

    Semakin banyak iterasi bilateral → makin 'painterly'
    """
    # Multi-pass bilateral filter (semakin tebal efek lukisan)
    out = frame.copy()
    n_pass = max(1, int(t * 6))
    for _ in range(n_pass):
        out = cv2.bilateralFilter(out, 9, 75, 75)

    # Boost saturation (warna cat minyak lebih vibrant)
    out = saturation_shift(out, 1 + 0.5 * t)

    # Edge darkening (simulasi goresan kuas)
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 60, 140)
    edges_f = cv2.GaussianBlur(edges, (3, 3), 1).astype(np.float32) / 255
    edges_mask = np.dstack([1 - edges_f * 0.5 * t] * 3)
    out = to_uint8(to_float(out) * edges_mask)

    # Slight warm tone
    out = adjust_curves(out,
        lambda i: min(255, i + 8 * t),
        lambda i: i,
        lambda i: max(0, i - 10 * t))

    return out


def fx_noir(frame, t=1.0):
    """
    NOIR — Classic Hollywood black & white cinematic.
    Dipakai di film-film hitam putih berkontrast tinggi
    seperti Sin City, Schindler's List accent scene.
    Viral untuk portrait artistik di IG.

    Teknik:
    1. Luminance-weighted grayscale (bukan average biasa!)
       Bobot: 0.299R + 0.587G + 0.114B (perceptual luminance)
    2. High-contrast S-curve: shadows sangat gelap,
       highlights sangat terang (crushed blacks, blown whites)
    3. Grain berat (film noir pakai film ASA tinggi → noisy)
    4. Vignette sangat kuat di sudut
    5. Cinematic bars
    6. Slight green/silver tone (khas emulsi film orthochromatic)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # High contrast S-curve
    def noir_curve(i):
        x = i / 255.0
        # Shadows crushed, highlight lifted
        y = 1 / (1 + math.exp(-8 * (x - 0.45)))   # sigmoid sharp
        return y * 255

    gray_lut = make_lut(noir_curve)
    gray_hi  = apply_lut(gray, gray_lut)

    # Silver/green tone (orthochromatic film look)
    out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(gray_hi.astype(int) - int(8 * t), 0, 255)   # B
    out[:, :, 1] = np.clip(gray_hi.astype(int) + int(5 * t), 0, 255)   # G slight
    out[:, :, 2] = gray_hi                                               # R

    out = add_grain(out, 0.06 * t)
    out = add_vignette(out, strength=0.75 * t, sigma_ratio=0.45)
    out = cinematic_bars(out)

    return blend(frame, out, t)


# ══════════════════════════════════════════════════════════════
#  DAFTAR EFEK
# ══════════════════════════════════════════════════════════════
EFFECTS = [
    ("Teal & Orange",    fx_teal_orange),
    ("VHS / Retro",      fx_vhs),
    ("Glitch",           fx_glitch),
    ("Vintage Film",     fx_vintage_film),
    ("Duotone",          fx_duotone),
    ("Dreamy / Aura",    fx_dreamy),
    ("Neon Cyberpunk",   fx_neon_cyberpunk),
    ("Bleach Bypass",    fx_bleach_bypass),
    ("Lo-Fi",            fx_lofi),
    ("Golden Hour",      fx_golden_hour),
    ("Film Burn",        fx_film_burn),
    ("Matrix",           fx_matrix),
    ("Holographic",      fx_holographic),
    ("Oil Paint",        fx_oil_paint),
    ("Noir",             fx_noir),
]

KEY_MAP = {
    ord('1'): 0,  ord('2'): 1,  ord('3'): 2,
    ord('4'): 3,  ord('5'): 4,  ord('6'): 5,
    ord('7'): 6,  ord('8'): 7,  ord('9'): 8,
    ord('0'): 9,  ord('q'): 10, ord('w'): 11,
    ord('e'): 12, ord('r'): 13, ord('t'): 14,
}


# ══════════════════════════════════════════════════════════════
#  UI OVERLAY (transparan — kamera full frame)
# ══════════════════════════════════════════════════════════════
def draw_ui(frame, eidx, intensity, recording, fps):
    h, w = frame.shape[:2]

    # Panel kiri semi-transparan
    panel_w = 175
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (panel_w, h), (8, 8, 10), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, "IG / TIKTOK FILTERS", (7, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 210, 255), 1)
    cv2.line(frame, (7, 28), (panel_w - 7, 28), (50, 50, 70), 1)

    keys = ['1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T']
    for i, (name, _) in enumerate(EFFECTS):
        y = 48 + i * 25
        if i == eidx:
            cv2.rectangle(frame, (4, y-14), (panel_w-4, y+7), (35, 90, 210), -1)
            col = (255, 255, 255)
        else:
            col = (115, 115, 135)
        cv2.putText(frame, f"[{keys[i]}] {name}", (9, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # Bar bawah
    bh = 42
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-bh), (w, h), (8, 8, 10), -1)
    cv2.addWeighted(ov2, 0.60, frame, 0.40, 0, frame)

    name = EFFECTS[eidx][0]
    cv2.putText(frame, name, (panel_w + 10, h-24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 215, 255), 1)

    bx, by = panel_w + 10, h - 9
    blen = min(170, w - panel_w - 180)
    cv2.rectangle(frame, (bx, by-6), (bx+blen, by+3), (35, 35, 45), -1)
    cv2.rectangle(frame, (bx, by-6), (bx+int(blen*intensity), by+3), (40, 195, 110), -1)
    cv2.putText(frame, f"{int(intensity*100)}%", (bx+blen+6, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 150), 1)

    hint = "[A/D] Intensity   [S] Screenshot   [V] Record   [ESC] Exit"
    cv2.putText(frame, hint, (panel_w + 10, h-bh+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 90), 1)

    cv2.putText(frame, f"{fps:.0f}fps", (w-52, h-26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 170, 70), 1)

    if recording:
        cv2.circle(frame, (w-18, 18), 7, (0, 0, 210), -1)
        cv2.putText(frame, "REC", (w-52, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 50, 210), 1)

    return frame


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(__doc__)

    cap = None
    for idx in range(3):
        for backend, bname in [
            (cv2.CAP_MSMF,  "MSMF"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY,   "Auto"),
        ]:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ret, test = c.read()
                if ret and test is not None:
                    cap = c
                    print(f"Kamera OK: index={idx}, backend={bname}")
                    break
                c.release()
        if cap:
            break

    if not cap:
        print("ERROR: Kamera tidak ditemukan!")
        input("Enter untuk keluar...")
        return

    print("Warming up...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi: {actual_w}x{actual_h}")

    WIN = "IG/TikTok Viral Effects - Pengolahan Citra"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WIN, 1920, 1080)

    eidx       = 0
    intensity  = 0.7
    recording  = False
    vid_writer = None
    prev_t     = time.time()
    fps        = 30.0

    print("Siap! Tekan [1]-[T] untuk pilih filter, [A/D] intensitas, [ESC] keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            continue

        frame = cv2.flip(frame, 1)   # mirror selfie

        curr_t = time.time()
        fps    = 0.9 * fps + 0.1 / (curr_t - prev_t + 1e-9)
        prev_t = curr_t

        name, fn = EFFECTS[eidx]
        try:
            result = fn(frame, intensity)
        except Exception as e:
            result = frame.copy()
            cv2.putText(result, f"Error: {e}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        result = draw_ui(result, eidx, intensity, recording, fps)

        cv2.imshow(WIN, result)

        if recording and vid_writer:
            vid_writer.write(result)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in KEY_MAP:
            eidx = KEY_MAP[key]
            print(f"Filter: {EFFECTS[eidx][0]}")
        elif key == ord('a'):
            intensity = max(0.0, round(intensity - 0.05, 2))
        elif key == ord('d'):
            intensity = min(1.0, round(intensity + 0.05, 2))
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn_save = f"output/{name.replace('/','_').replace(' ','_')}_{ts}.jpg"
            cv2.imwrite(fn_save, result)
            print(f"Screenshot: {fn_save}")
        elif key == ord('v'):
            if not recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn_vid = f"output/video_{ts}.mp4"
                fh, fw = result.shape[:2]
                vid_writer = cv2.VideoWriter(
                    fn_vid, cv2.VideoWriter_fourcc(*'mp4v'), 20, (fw, fh))
                recording = True
                print(f"Recording: {fn_vid}")
            else:
                recording = False
                vid_writer.release()
                vid_writer = None
                print("Recording berhenti & tersimpan.")

    cap.release()
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()
    print("Selesai. File ada di folder output/")


if __name__ == "__main__":
    main()