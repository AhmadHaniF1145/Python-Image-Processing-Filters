"""
╔══════════════════════════════════════════════════════════╗
║        AR FACE GAMES - Filter Game TikTok/Instagram      ║
║        Mata Kuliah: Pengolahan Citra Digital              ║
║                                                          ║
║  Teknologi Utama:                                        ║
║  • Haar Cascade Face Detection (built-in OpenCV)         ║
║  • Collision Detection (circle-to-circle)                ║
║  • Particle System                                       ║
║  • Exponential Moving Average (smooth tracking)          ║
╚══════════════════════════════════════════════════════════╝

CARA MAIN:
  Gerakkan kepala/wajah untuk bermain — tidak butuh tangan!

  [1] METEOR DODGE   — Hindari meteor yang jatuh dari luar angkasa!
  [2] RING CATCHER   — Tangkap cincin dengan kepala kamu!
  [3] FRUIT SMASHER  — Hancurkan buah, hindari BOM!
  [R] Restart game saat ini
  [S] Screenshot
  [ESC] Keluar

Install: pip install opencv-python numpy
"""

import cv2
import numpy as np
import random
import time
import math
import os
from datetime import datetime

os.makedirs("output", exist_ok=True)

# ──────────────────────────────────────────────────────────
# FACE DETECTOR — Haar Cascade (built-in di OpenCV)
#
# Haar Cascade bekerja dengan cara:
# 1. Sliding window di berbagai skala gambar
# 2. Setiap window dievaluasi oleh 6000+ fitur "Haar-like"
#    (perbedaan intensitas antara area gelap dan terang)
# 3. Adaboost classifier memilah kandidat wajah dari non-wajah
# 4. Cascade structure: window yang gagal di tahap awal
#    langsung dibuang (sangat cepat!)
# Hasil: bounding box (x, y, w, h) setiap wajah yang terdeteksi
# ──────────────────────────────────────────────────────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


class SmoothFace:
    """
    Tracker wajah dengan Exponential Moving Average (EMA).

    Masalah: deteksi wajah raw sangat "jittery" (gemetar) frame ke frame.
    Solusi: EMA menghaluskan posisi dengan rumus:
        pos_smooth = alpha * pos_baru + (1 - alpha) * pos_lama
    alpha kecil (0.2) = sangat halus tapi lambat
    alpha besar (0.5) = responsif tapi masih ada guncangan
    Kita pakai 0.35 — keseimbangan yang bagus untuk game.
    """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.cx = self.cy = self.r = None
        self.detected = False

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaleFactor=1.1: cek skala 1x, 1.1x, 1.21x, ...
        # minNeighbors=5: butuh 5 deteksi bertetangga agar valid (kurangi false positive)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            # Ambil wajah terbesar (terdekat ke kamera)
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            ncx = fx + fw // 2
            ncy = fy + fh // 2
            nr  = max(fw, fh) // 2
            if self.cx is None:
                self.cx, self.cy, self.r = ncx, ncy, nr
            else:
                # EMA smoothing
                a = self.alpha
                self.cx = int(a * ncx + (1 - a) * self.cx)
                self.cy = int(a * ncy + (1 - a) * self.cy)
                self.r  = int(a * nr  + (1 - a) * self.r)
            self.detected = True
        else:
            self.detected = False
        return self.detected

    def get(self):
        """Return (center_x, center_y, radius)"""
        return self.cx, self.cy, self.r


# ──────────────────────────────────────────────────────────
# PARTICLE SYSTEM
#
# Digunakan untuk efek ledakan/sparkle saat terjadi event.
# Setiap partikel memiliki: posisi, kecepatan, warna, lifetime.
# Setiap frame: posisi += kecepatan, kecepatan_y += gravity
# Alpha = life/max_life → partikel memudar seiring waktu
# ──────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, size=4, life=35):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-8, -1)
        self.life = life
        self.max_life = life
        self.size = size

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.35       # gravitasi
        self.vx *= 0.97       # friction udara
        self.life -= 1

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = self.life / self.max_life
        s = max(1, int(self.size * alpha))
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), s, c, -1)


def burst_particles(particles, x, y, color, n=20, size=5):
    """Buat ledakan partikel di titik (x, y)"""
    for _ in range(n):
        particles.append(Particle(x, y, color, size=size))


def update_draw_particles(frame, particles):
    """Update semua partikel dan buang yang sudah mati"""
    alive = []
    for p in particles:
        p.update()
        p.draw(frame)
        if p.life > 0:
            alive.append(p)
    return alive


# ──────────────────────────────────────────────────────────
# UTILITY DRAWING
# ──────────────────────────────────────────────────────────
def text_shadow(frame, text, pos, scale, color, thick=2):
    """Teks dengan drop shadow agar terbaca di atas video apapun"""
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def text_center(frame, text, y, scale, color, thick=2):
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text_shadow(frame, text, ((w - tw) // 2, y), scale, color, thick)


def draw_face_halo(frame, cx, cy, r, color=(0, 255, 180)):
    """
    Lingkaran + crosshair di sekeliling wajah.
    Ini visual feedback bahwa wajah terdeteksi dan aktif sebagai kontroller.
    """
    cv2.circle(frame, (cx, cy), r + 6, color, 2)
    for angle in range(0, 360, 90):
        rad = math.radians(angle)
        px = int(cx + (r + 14) * math.cos(rad))
        py = int(cy + (r + 14) * math.sin(rad))
        cv2.circle(frame, (px, py), 4, color, -1)


def draw_hud(frame, score, lives, game_name, fps):
    """
    HUD (Heads Up Display) — informasi game di atas layar.
    Menggunakan alpha blending untuk panel semi-transparan:
    result = overlay * alpha + frame * (1 - alpha)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text_shadow(frame, game_name,        (10, 32), 0.65, (80, 220, 255))
    text_shadow(frame, f"SCORE: {score}", (w//2 - 65, 32), 0.7, (255, 220, 50))
    # Gambar hati sebagai indikator nyawa
    for i in range(max(0, lives)):
        hx = w - 28 - i * 28
        cv2.circle(frame, (hx, 22), 9, (60, 60, 255), -1)
        cv2.putText(frame, "v", (hx - 5, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 255), 1)
    cv2.putText(frame, f"{fps:.0f}fps", (w - 55, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)


# ══════════════════════════════════════════════════════════
# GAME 1: METEOR DODGE 🚀
# ══════════════════════════════════════════════════════════
class MeteorDodge:
    """
    METEOR DODGE
    ─────────────────────────────────────────────────────
    Konsep: Meteor jatuh dari atas layar, pemain harus
    menggerakkan kepala untuk menghindarinya.

    Teknik Pengolahan Citra:
    • Face Detection → dapatkan posisi & radius wajah
    • Collision Detection: 2 lingkaran bertabrakan jika
      jarak pusat < r1 + r2
      dist = sqrt((x1-x2)² + (y1-y2)²)
    • Alpha Blending untuk efek kilat saat kena hit
    • Particle System untuk efek ledakan

    Kesulitan meningkat: spawn lebih cepat + speed naik
    seiring bertambahnya skor.
    ─────────────────────────────────────────────────────
    """
    NAME = "METEOR DODGE"
    COLOR = (80, 120, 255)

    def __init__(self, w, h):
        self.w, self.h = w, h
        # Bintang latar — dibuat sekali, dipakai terus
        self.stars = [(random.randint(0, w), random.randint(0, h),
                       random.randint(1, 3), random.uniform(0.3, 1.0))
                      for _ in range(80)]
        self.reset()

    def reset(self):
        self.meteors   = []
        self.particles = []
        self.score     = 0
        self.lives     = 3
        self.game_over = False
        self.invincible = 0    # frame kebal setelah kena hit
        self.frame_count = 0
        self.spawn_timer = 0
        self.twinkling = 0

    def _spawn(self):
        x     = random.randint(40, self.w - 40)
        size  = random.randint(16, 38)
        speed = 2.5 + self.score * 0.025 + random.uniform(0, 2)
        color = random.choice([
            (30, 80, 210), (20, 60, 180), (50, 100, 230)
        ])
        n_pts = random.randint(7, 10)
        # Pre-generate bentuk irregular meteor (tidak berubah saat jatuh)
        shape_offsets = [
            (random.uniform(0.6, 1.0) if i % 2 == 0 else random.uniform(0.4, 0.7))
            for i in range(n_pts)
        ]
        self.meteors.append({
            'x': x, 'y': float(-size - 10), 'size': size, 'speed': speed,
            'color': color, 'rot': random.uniform(0, 360),
            'rot_speed': random.uniform(-4, 4),
            'n_pts': n_pts, 'shape': shape_offsets
        })

    def _draw_meteor(self, frame, m):
        x, y, s = int(m['x']), int(m['y']), m['size']
        n  = m['n_pts']
        # Buat polygon tidak beraturan seperti batu luar angkasa
        pts = []
        for i in range(n):
            angle = (i / n) * 2 * math.pi + math.radians(m['rot'])
            r = s * m['shape'][i]
            pts.append([int(x + r * math.cos(angle)),
                        int(y + r * math.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(frame, [pts], m['color'])
        # Highlight pinggir
        cv2.polylines(frame, [pts], True, (160, 180, 255), 1)
        # Fire trail (partikel api kecil di belakang meteor)
        for _ in range(4):
            tx = x + random.randint(-s//3, s//3)
            ty = y - random.randint(4, s + 4)
            ts = random.randint(2, 7)
            brightness = random.uniform(0.4, 1.0)
            fc = (0, int(100 * brightness), int(255 * brightness))
            cv2.circle(frame, (tx, ty), ts, fc, -1)

    def update(self, frame, face):
        h, w = frame.shape[:2]
        self.frame_count += 1
        self.twinkling   += 1

        # ── Background: starfield ──
        for sx, sy, ss, bright in self.stars:
            # Bintang "kedip" dengan sinus
            b = int(180 * bright * (0.5 + 0.5 * math.sin(
                self.twinkling * 0.05 + sx * 0.1)))
            cv2.circle(frame, (sx, sy), ss, (b, b, b + 40), -1)

        if self.game_over:
            return

        # ── Spawn meteor ──
        self.spawn_timer += 1
        interval = max(15, 55 - self.score // 4)
        if self.spawn_timer >= interval:
            self._spawn()
            self.spawn_timer = 0

        # ── Update & deteksi tabrakan ──
        alive = []
        cx, cy, fr = face.get() if face.detected else (0, 0, 0)
        for m in self.meteors:
            m['y']   += m['speed']
            m['rot'] += m['rot_speed']

            still_on = m['y'] < h + m['size']
            if still_on:
                self._draw_meteor(frame, m)

                # Collision: jarak antar pusat < jumlah radius
                if face.detected and self.invincible == 0:
                    dist = math.hypot(m['x'] - cx, m['y'] - cy)
                    hit_thresh = fr * 0.75 + m['size'] * 0.8
                    if dist < hit_thresh:
                        self.lives    -= 1
                        self.invincible = 70
                        burst_particles(self.particles, cx, cy,
                                        (80, 80, 255), n=25)
                        if self.lives <= 0:
                            self.game_over = True
                        continue   # meteor hancur, jangan ditambah ke alive
                alive.append(m)
            else:
                # Meteor lewat bawah layar = pemain berhasil dodge
                self.score += 2
        self.meteors = alive

        # ── Invincibility flash (layar merah berkedip) ──
        if self.invincible > 0:
            self.invincible -= 1
            if (self.invincible // 6) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # ── Skor bertambah seiring waktu bertahan ──
        if self.frame_count % 25 == 0:
            self.score += 1

        # ── Gambar indikator wajah ──
        if face.detected:
            color = (50, 50, 255) if self.invincible > 0 else (50, 255, 180)
            draw_face_halo(frame, cx, cy, fr, color)
            # Shield visual saat invincible
            if self.invincible > 0:
                cv2.circle(frame, (cx, cy), fr + 15,
                           (50, 50, 255), max(1, self.invincible // 10))

        self.particles = update_draw_particles(frame, self.particles)


# ══════════════════════════════════════════════════════════
# GAME 2: RING CATCHER 💍
# ══════════════════════════════════════════════════════════
class RingCatcher:
    """
    RING CATCHER
    ─────────────────────────────────────────────────────
    Konsep: Cincin jatuh dari atas — tangkap dengan kepala!
    Cincin dianggap tertangkap jika:
      1. Jarak pusat cincin ke pusat wajah < threshold
      2. Ukuran cincin ≈ ukuran wajah (size matching)

    Teknik Pengolahan Citra:
    • Face Detection + EMA smoothing
    • Dual-criteria collision (proximity + size matching)
      Ini lebih realistis dari sekedar circle overlap —
      seperti benar-benar memasukkan cincin ke kepala!
    • cv2.ellipse() untuk cincin 3D perspektif
    • Combo multiplier: tangkap berturut → poin berlipat

    Cincin yang lolos (jatuh ke bawah) = kehilangan nyawa.
    ─────────────────────────────────────────────────────
    """
    NAME = "RING CATCHER"
    COLOR = (180, 80, 255)

    RING_COLORS = [
        (0, 215, 255),    # gold
        (255, 100, 60),   # biru tua
        (50, 255, 120),   # hijau
        (255, 60, 200),   # pink
        (60, 220, 255),   # cyan
    ]

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.rings     = []
        self.particles = []
        self.score     = 0
        self.lives     = 5
        self.game_over = False
        self.spawn_timer  = 0
        self.frame_count  = 0
        self.combo        = 0
        self.combo_timer  = 0
        self.catch_flashes = []   # efek kilat saat berhasil tangkap

    def _spawn(self):
        x     = random.randint(80, self.w - 80)
        r     = random.randint(38, 80)    # radius cincin
        speed = 1.2 + self.score * 0.015 + random.uniform(0, 1.5)
        color = random.choice(self.RING_COLORS)
        self.rings.append({
            'x': float(x), 'y': float(-r - 10),
            'r': r, 'speed': speed, 'color': color,
            'shimmer': 0, 'wobble': random.uniform(0, math.pi * 2)
        })

    def _draw_ring(self, frame, ring):
        x, y   = int(ring['x']), int(ring['y'])
        r      = ring['r']
        c      = ring['color']
        bright = tuple(min(255, v + 80) for v in c)
        dark   = tuple(max(0, v - 60)  for v in c)

        # Ellipse untuk efek 3D perspektif cincin
        # Tinggi ellipse = r//3 memberikan ilusi cincin miring
        cv2.ellipse(frame, (x, y), (r, r // 3), 0, 0, 360, dark, 5)
        cv2.ellipse(frame, (x, y), (r, r // 3), 0, 0, 360, c,    3)
        cv2.ellipse(frame, (x, y), (r, r // 3), 0, 0, 360, bright, 1)

        # Bintik shimmer yang berputar mengelilingi cincin
        ring['shimmer'] += 8
        for i in range(5):
            angle = math.radians(ring['shimmer'] + i * 72)
            sx    = int(x + r * math.cos(angle))
            sy    = int(y + (r // 3) * math.sin(angle))
            cv2.circle(frame, (sx, sy), 3, (255, 255, 220), -1)

    def update(self, frame, face):
        h, w = frame.shape[:2]
        self.frame_count += 1

        # ── Background: gradient biru-ungu ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (30, 5, 50), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        if self.game_over:
            return

        # ── Spawn ──
        self.spawn_timer += 1
        interval = max(40, 100 - self.score * 2)
        if self.spawn_timer >= interval:
            self._spawn()
            self.spawn_timer = 0

        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo = 0

        cx, cy, fr = face.get() if face.detected else (0, 0, 0)

        alive = []
        for ring in self.rings:
            ring['y'] += ring['speed']
            caught = False

            if face.detected:
                dist = math.hypot(ring['x'] - cx, ring['y'] - cy)
                # Size match: ukuran cincin tidak terlalu beda dari radius wajah
                size_ok = abs(ring['r'] - fr) < fr * 0.65
                close   = dist < fr * 0.55

                if close and size_ok:
                    # TERTANGKAP!
                    self.combo      += 1
                    self.combo_timer = 70
                    pts              = 10 * max(1, self.combo)
                    self.score      += pts
                    burst_particles(self.particles, int(ring['x']),
                                    int(ring['y']), ring['color'], n=30)
                    self.catch_flashes.append({
                        'x': cx, 'y': cy - fr - 20,
                        'text': f"+{pts}" + (" COMBO!" if self.combo > 1 else ""),
                        'color': ring['color'], 'life': 40
                    })
                    caught = True

            if not caught:
                self._draw_ring(frame, ring)
                if ring['y'] <= h + ring['r']:
                    alive.append(ring)
                else:
                    # Cincin lolos = kehilangan nyawa
                    self.lives -= 1
                    burst_particles(self.particles, int(ring['x']),
                                    h - 10, (80, 80, 255), n=10)
                    if self.lives <= 0:
                        self.game_over = True
        self.rings = alive

        # ── Efek visual saat tangkap ──
        new_flashes = []
        for fl in self.catch_flashes:
            fl['life'] -= 1
            if fl['life'] > 0:
                alpha = fl['life'] / 40
                c = tuple(int(ch * alpha) for ch in fl['color'])
                text_shadow(frame, fl['text'], (fl['x'] - 30, fl['y']),
                            0.7, c)
                fl['y'] -= 1
                new_flashes.append(fl)
        self.catch_flashes = new_flashes

        # ── Combo display besar di tengah ──
        if self.combo >= 2:
            pulse = 0.9 + 0.2 * math.sin(self.frame_count * 0.3)
            text_center(frame, f"x{self.combo} COMBO!",
                       int(h * 0.45), 1.1 * pulse, (255, 200, 50), 2)

        # ── Face indicator + zona tangkap ──
        if face.detected:
            draw_face_halo(frame, cx, cy, fr, (180, 80, 255))
            # Zona tangkap (lingkaran tipis di area kepala)
            cv2.circle(frame, (cx, cy), int(fr * 0.55),
                       (180, 80, 255), 1)

        self.particles = update_draw_particles(frame, self.particles)


# ══════════════════════════════════════════════════════════
# GAME 3: FRUIT SMASHER 🍎💣
# ══════════════════════════════════════════════════════════
class FruitSmasher:
    """
    FRUIT SMASHER
    ─────────────────────────────────────────────────────
    Konsep: Buah terbang melintasi layar dari kiri/kanan.
    Gerakkan kepala untuk menabrak & menghancurkan buah.
    TAPI hindari BOM! (muncul setelah skor > 30)

    Teknik Pengolahan Citra:
    • Circle-to-circle collision detection
    • Velocity-based projectile motion
    • Alpha blending untuk screen flash saat kena bom
    • Procedural fruit drawing (tidak pakai gambar/aset!)
      Setiap buah digambar dengan cv2 shapes + warna

    Strategi: bom punya reward trap — gerak reflex smash
    buah harus dikontrol agar tidak kena bom!
    ─────────────────────────────────────────────────────
    """
    NAME = "FRUIT SMASHER"
    COLOR = (50, 230, 100)

    # Data setiap jenis buah: warna BGr, poin, radius, nama
    FRUIT_DATA = [
        dict(name='apple',      color=(50, 50, 220),   pts=10, r=26),
        dict(name='orange',     color=(30, 150, 255),  pts=15, r=24),
        dict(name='watermelon', color=(50, 195, 60),   pts=25, r=38),
        dict(name='grape',      color=(170, 40, 190),  pts=30, r=19),
        dict(name='banana',     color=(30, 230, 240),  pts=20, r=22),
    ]
    BOMB = dict(name='bomb', color=(25, 25, 25), pts=-1, r=30)

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.fruits    = []
        self.particles = []
        self.slashes   = []   # efek sabetan saat smash
        self.score     = 0
        self.lives     = 3
        self.game_over = False
        self.spawn_timer  = 0
        self.frame_count  = 0

    def _spawn(self):
        from_left = random.random() > 0.5
        x  = float(-60 if from_left else self.w + 60)
        y  = float(random.randint(int(self.h * 0.15), int(self.h * 0.85)))
        vx = random.uniform(3.5, 7.5) * (1 if from_left else -1)
        vy = random.uniform(-3, 3)

        # Bom mulai spawn setelah skor 30
        pool = self.FRUIT_DATA if self.score < 30 else (self.FRUIT_DATA + [self.BOMB] * 2)
        f = dict(random.choice(pool))  # copy agar tidak ubah template
        f.update({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'alive': True})
        self.fruits.append(f)

    def _draw_fruit(self, frame, f):
        x, y, r, c = int(f['x']), int(f['y']), f['r'], f['color']
        name = f['name']

        if name == 'bomb':
            # Bom: lingkaran hitam + sumbu api
            cv2.circle(frame, (x, y), r, (40, 40, 40), -1)
            cv2.circle(frame, (x, y), r, (120, 120, 120), 2)
            # Sumbu
            cv2.line(frame, (x, y - r), (x + 10, y - r - 14), (60, 160, 255), 2)
            # Api di ujung sumbu (berkedip)
            if self.frame_count % 6 < 3:
                cv2.circle(frame, (x + 10, y - r - 14), 4, (50, 200, 255), -1)
            cv2.putText(frame, "!", (x - 5, y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return

        # Badan buah
        cv2.circle(frame, (x, y), r, c, -1)
        # Highlight (pantulan cahaya)
        hx, hy = x - r // 3, y - r // 3
        bright = tuple(min(255, v + 90) for v in c)
        cv2.circle(frame, (hx, hy), r // 4, bright, -1)
        # Outline
        dark = tuple(max(0, v - 70) for v in c)
        cv2.circle(frame, (x, y), r, dark, 2)

        # Detail spesifik per buah
        if name == 'watermelon':
            # Garis-garis semangka
            for i in range(-r + 8, r - 8, 9):
                cv2.line(frame, (x + i, y - r + 4), (x + i, y + r - 4),
                         (30, 100, 25), 1)
            # Warna merah di dalam (setengah lingkaran)
            cv2.ellipse(frame, (x, y), (r - 4, r - 4), 0, 0, 180,
                        (50, 50, 200), -1)
        elif name == 'grape':
            # Tangkai
            cv2.line(frame, (x, y - r), (x + 4, y - r - 8), (40, 60, 140), 2)
        elif name == 'banana':
            # Pisang: ellipse miring
            cv2.ellipse(frame, (x, y), (r + 5, r - 8), 30, 0, 360, c, -1)
            cv2.ellipse(frame, (x, y), (r + 5, r - 8), 30, 0, 360, dark, 2)
        elif name == 'apple':
            # Daun apel
            cv2.ellipse(frame, (x + 5, y - r - 3), (6, 4), 30, 0, 360,
                        (30, 130, 30), -1)

    def update(self, frame, face):
        h, w = frame.shape[:2]
        self.frame_count += 1

        # ── Spawn ──
        self.spawn_timer += 1
        interval = max(20, 60 - self.score // 4)
        if self.spawn_timer >= interval:
            self._spawn()
            self.spawn_timer = 0

        cx, cy, fr = face.get() if face.detected else (0, 0, 0)

        alive = []
        for f in self.fruits:
            f['x'] += f['vx']
            f['y'] += f['vy']

            in_bounds = -120 < f['x'] < w + 120

            if face.detected and f['alive']:
                dist = math.hypot(f['x'] - cx, f['y'] - cy)
                # Collision: jarak < radius wajah * faktor + radius buah
                if dist < fr * 0.85 + f['r']:
                    f['alive'] = False
                    if f['name'] == 'bomb':
                        # Kena bom!
                        self.lives -= 1
                        burst_particles(self.particles, int(f['x']),
                                        int(f['y']), (40, 40, 255), n=40, size=7)
                        # Screen flash merah
                        ov = frame.copy()
                        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 200), -1)
                        cv2.addWeighted(ov, 0.30, frame, 0.70, 0, frame)
                        if self.lives <= 0:
                            self.game_over = True
                    else:
                        # Buah kena!
                        self.score += f['pts']
                        burst_particles(self.particles, int(f['x']),
                                        int(f['y']), f['color'], n=22)
                        self.slashes.append({
                            'x': int(f['x']), 'y': int(f['y']),
                            'life': 20, 'color': f['color'],
                            'pts': f['pts']
                        })
                    continue   # jangan tambah ke alive

            if in_bounds and f['alive']:
                self._draw_fruit(frame, f)
                alive.append(f)
        self.fruits = alive

        # ── Efek sabetan ──
        new_slashes = []
        for s in self.slashes:
            s['life'] -= 1
            if s['life'] > 0:
                a = s['life'] / 20
                r_ring = int(50 * (1 - a))
                c = tuple(int(ch * a) for ch in s['color'])
                cv2.circle(frame, (s['x'], s['y']), r_ring, c, 2)
                if s['life'] > 10:
                    text_shadow(frame, f"+{s['pts']} SMASH!",
                                (s['x'] - 40, s['y'] - 25), 0.6, s['color'])
                new_slashes.append(s)
        self.slashes = new_slashes

        # ── Peringatan bom muncul ──
        if self.score >= 25 and self.frame_count % 120 < 60:
            text_shadow(frame, "WATCH OUT FOR BOMBS!", (w // 2 - 130, h - 20),
                        0.55, (80, 80, 255))

        # ── Face indicator ──
        if face.detected:
            draw_face_halo(frame, cx, cy, fr, (50, 240, 100))

        self.particles = update_draw_particles(frame, self.particles)


# ══════════════════════════════════════════════════════════
# LAYAR GAME OVER & START
# ══════════════════════════════════════════════════════════
def draw_game_over(frame, score, game_name):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    text_center(frame, "GAME OVER",      h // 2 - 65, 1.9, (80, 80, 255), 3)
    text_center(frame, game_name,        h // 2 - 15, 0.65, (180, 180, 180))
    text_center(frame, f"SCORE: {score}", h // 2 + 35, 1.3, (255, 215, 50), 2)
    text_center(frame, "[R] Restart   [1/2/3] Ganti Game   [ESC] Keluar",
               h // 2 + 85, 0.48, (140, 140, 140))


def draw_start_screen(frame, blink):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    text_center(frame, "AR FACE GAMES",      h // 2 - 100, 1.5,  (80, 220, 255), 3)
    text_center(frame, "Pengolahan Citra Digital", h // 2 - 55, 0.55, (140, 140, 180))
    text_center(frame, "[1]  METEOR DODGE",  h // 2 - 5,   0.75, (255, 180, 60))
    text_center(frame, "[2]  RING CATCHER",  h // 2 + 38,  0.75, (190, 80, 255))
    text_center(frame, "[3]  FRUIT SMASHER", h // 2 + 81,  0.75, (60, 240, 110))
    if blink % 60 < 40:
        text_center(frame, "Posisikan wajah di depan kamera lalu pilih game!",
                   h // 2 + 130, 0.5, (200, 200, 200))


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print(__doc__)

    # ── Buka kamera (coba semua kombinasi backend/index) ──
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
                    print(f"Kamera: index={idx}, backend={bname}")
                    break
                c.release()
        if cap:
            break

    if not cap:
        print("ERROR: Kamera tidak ditemukan!")
        input("Tekan Enter untuk keluar...")
        return

    # Warm-up
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi: {cam_w}x{cam_h}")

    WIN = "AR Face Games - Pengolahan Citra"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 600)

    face  = SmoothFace(alpha=0.35)
    games = [
        MeteorDodge(cam_w, cam_h),
        RingCatcher(cam_w, cam_h),
        FruitSmasher(cam_w, cam_h),
    ]
    current = 0
    started = False
    frame_count = 0

    prev_t = time.time()
    fps    = 30.0

    print("Siap! Pilih game dengan tombol [1], [2], atau [3].")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            continue

        # Mirror (seperti selfie camera) — lebih intuitif untuk game
        frame = cv2.flip(frame, 1)

        curr_t = time.time()
        fps    = 0.9 * fps + 0.1 / (curr_t - prev_t + 1e-9)
        prev_t = curr_t
        frame_count += 1

        # Deteksi wajah setiap frame
        face.update(frame)

        game = games[current]

        if not started:
            draw_start_screen(frame, frame_count)
        else:
            game.update(frame, face)
            draw_hud(frame, game.score, game.lives, game.NAME, fps)
            if game.game_over:
                draw_game_over(frame, game.score, game.NAME)

        # Peringatan jika wajah tidak terdeteksi
        if not face.detected and started and not game.game_over:
            text_shadow(frame, "Wajah tidak terdeteksi!",
                        (10, frame.shape[0] - 12), 0.55, (100, 100, 255))

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                        # ESC
            break
        elif key == ord('1'):
            current = 0; started = True; games[0].reset()
        elif key == ord('2'):
            current = 1; started = True; games[1].reset()
        elif key == ord('3'):
            current = 2; started = True; games[2].reset()
        elif key in (ord('r'), ord('R')):
            game.reset()
        elif key == ord('s'):
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"output/game_screenshot_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Screenshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")


if __name__ == "__main__":
    main()