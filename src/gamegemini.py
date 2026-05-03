"""
╔══════════════════════════════════════════════════════════╗
║        AR FACE GAMES - Filter Game TikTok/Instagram      ║
║        Mata Kuliah: Pengolahan Citra Digital             ║
║                                                          ║
║  Teknologi Utama:                                        ║
║  • Haar Cascade Face Detection (built-in OpenCV)         ║
║  • Lane-based & Bounding Box Collision                   ║
║  • Particle System & EMA Smoothing                       ║
╚══════════════════════════════════════════════════════════╝

CARA MAIN:
  [1] FLAPPY FACE   — Naik-turunkan kepala lewati pipa!
  [2] SUBWAY DODGE  — Geser kiri/tengah/kanan hindari palang!
  [3] TOWER STACK   — Tangkap balok di atas kepala!
  [R] Restart game
  [ESC] Keluar
"""

import cv2
import numpy as np
import random
import time
import math
import os
from datetime import datetime

os.makedirs("output", exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class SmoothFace:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.cx = self.cy = self.r = None
        self.detected = False

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            ncx = fx + fw // 2
            ncy = fy + fh // 2
            nr  = max(fw, fh) // 2
            if self.cx is None:
                self.cx, self.cy, self.r = ncx, ncy, nr
            else:
                a = self.alpha
                self.cx = int(a * ncx + (1 - a) * self.cx)
                self.cy = int(a * ncy + (1 - a) * self.cy)
                self.r  = int(a * nr  + (1 - a) * self.r)
            self.detected = True
        else:
            self.detected = False
        return self.detected

    def get(self):
        return self.cx, self.cy, self.r


class Particle:
    def __init__(self, x, y, color, size=4, life=35):
        self.x, self.y = float(x), float(y)
        self.color = color
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-8, -1)
        self.life = self.max_life = life
        self.size = size

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.35
        self.vx *= 0.97
        self.life -= 1

    def draw(self, frame):
        if self.life <= 0: return
        alpha = self.life / self.max_life
        s = max(1, int(self.size * alpha))
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), s, c, -1)

def burst_particles(particles, x, y, color, n=20, size=5):
    for _ in range(n):
        particles.append(Particle(x, y, color, size=size))

def update_draw_particles(frame, particles):
    alive = []
    for p in particles:
        p.update()
        p.draw(frame)
        if p.life > 0: alive.append(p)
    return alive

def text_shadow(frame, text, pos, scale, color, thick=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def text_center(frame, text, y, scale, color, thick=2):
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text_shadow(frame, text, ((w - tw) // 2, y), scale, color, thick)

def draw_hud(frame, score, lives, game_name, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    text_shadow(frame, game_name, (10, 32), 0.65, (80, 220, 255))
    text_shadow(frame, f"SCORE: {score}", (w//2 - 65, 32), 0.7, (255, 220, 50))
    for i in range(max(0, lives)):
        hx = w - 28 - i * 28
        cv2.circle(frame, (hx, 22), 9, (60, 60, 255), -1)
    cv2.putText(frame, f"{fps:.0f}fps", (w - 55, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

# ══════════════════════════════════════════════════════════
# GAME 1: FLAPPY FACE
# ══════════════════════════════════════════════════════════
class FlappyFace:
    NAME = "FLAPPY FACE"
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.pipes = []
        self.particles = []
        self.score = 0
        self.lives = 1 # Flappy bird cuma 1 nyawa biasanya, tapi kita set 1
        self.game_over = False
        self.frame_count = 0
        self.pipe_speed = 8
        self.pipe_width = 80
        self.gap_size = 220

    def _spawn_pipe(self):
        gap_y = random.randint(100, self.h - self.gap_size - 100)
        self.pipes.append({
            'x': self.w, 
            'gap_y': gap_y,
            'passed': False
        })

    def update(self, frame, face):
        self.frame_count += 1
        if self.game_over: return

        # Spawn pipa baru
        if self.frame_count % max(40, 90 - self.score*2) == 0:
            self._spawn_pipe()

        cx, cy, fr = face.get() if face.detected else (self.w//4, self.h//2, 40)
        bird_r = int(fr * 0.6) # Hitbox sedikit lebih kecil dari wajah

        alive_pipes = []
        for p in self.pipes:
            p['x'] -= self.pipe_speed

            # Cek Tabrakan (Bounding Box)
            if p['x'] < cx + bird_r and p['x'] + self.pipe_width > cx - bird_r:
                if cy - bird_r < p['gap_y'] or cy + bird_r > p['gap_y'] + self.gap_size:
                    self.game_over = True
                    burst_particles(self.particles, cx, cy, (0,0,255), 40)

            # Tambah skor jika terlewati
            if not p['passed'] and p['x'] + self.pipe_width < cx:
                p['passed'] = True
                self.score += 1
                self.pipe_speed += 0.2 # Makin cepat

            # Gambar Pipa
            px, py_gap = int(p['x']), int(p['gap_y'])
            # Pipa Atas
            cv2.rectangle(frame, (px, 0), (px + self.pipe_width, py_gap), (50, 200, 50), -1)
            cv2.rectangle(frame, (px-5, py_gap-30), (px + self.pipe_width+5, py_gap), (50, 200, 50), -1)
            cv2.rectangle(frame, (px, 0), (px + self.pipe_width, py_gap), (0, 100, 0), 3)
            # Pipa Bawah
            cv2.rectangle(frame, (px, py_gap + self.gap_size), (px + self.pipe_width, self.h), (50, 200, 50), -1)
            cv2.rectangle(frame, (px-5, py_gap + self.gap_size), (px + self.pipe_width+5, py_gap + self.gap_size+30), (50, 200, 50), -1)
            cv2.rectangle(frame, (px, py_gap + self.gap_size), (px + self.pipe_width, self.h), (0, 100, 0), 3)

            if p['x'] > -self.pipe_width:
                alive_pipes.append(p)

        self.pipes = alive_pipes

        # Gambar Karakter / Indikator Wajah
        if face.detected:
            cv2.circle(frame, (cx, cy), bird_r, (50, 255, 255), 3)
            cv2.circle(frame, (cx+10, cy-10), 5, (255, 255, 255), -1) # Mata

        self.particles = update_draw_particles(frame, self.particles)

# ══════════════════════════════════════════════════════════
# GAME 2: SUBWAY DODGE
# ══════════════════════════════════════════════════════════
class SubwayDodge:
    NAME = "SUBWAY DODGE"
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.obs = []
        self.particles = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.frame_count = 0
        self.speed = 10

    def _spawn_obs(self):
        lane = random.randint(0, 2)
        self.obs.append({'y': -50, 'lane': lane})

    def update(self, frame, face):
        self.frame_count += 1
        if self.game_over: return

        lane_w = self.w // 3
        
        # Gambar garis jalur (perspektif sederhana)
        cv2.line(frame, (lane_w, 0), (lane_w, self.h), (255, 255, 255), 2)
        cv2.line(frame, (lane_w*2, 0), (lane_w*2, self.h), (255, 255, 255), 2)

        if self.frame_count % max(20, 50 - self.score) == 0:
            self._spawn_obs()

        cx, cy, fr = face.get() if face.detected else (self.w//2, self.h//2, 40)
        
        # Tentukan pemain ada di jalur mana
        player_lane = 0 if cx < lane_w else (1 if cx < lane_w*2 else 2)

        alive_obs = []
        for o in self.obs:
            o['y'] += self.speed
            oy = int(o['y'])
            ox = (o['lane'] * lane_w) + (lane_w // 2)

            # Cek Tabrakan di area bawah (zona wajah)
            if oy > cy - fr and oy < cy + fr and o['lane'] == player_lane:
                self.lives -= 1
                burst_particles(self.particles, ox, oy, (0,0,255), 30)
                if self.lives <= 0:
                    self.game_over = True
                continue # Hancur ditabrak

            # Gambar Rintangan (Palang Merah)
            cv2.rectangle(frame, (ox - 40, oy - 20), (ox + 40, oy + 20), (0, 0, 200), -1)
            cv2.rectangle(frame, (ox - 40, oy - 20), (ox + 40, oy + 20), (255, 255, 255), 2)
            # Motif palang
            cv2.line(frame, (ox - 30, oy - 20), (ox - 10, oy + 20), (255,255,255), 3)
            cv2.line(frame, (ox + 10, oy - 20), (ox + 30, oy + 20), (255,255,255), 3)

            if oy < self.h + 50:
                alive_obs.append(o)
            else:
                self.score += 1 # Lewat = dapat poin

        self.obs = alive_obs

        # Gambar Indikator Jalur Pemain
        target_x = (player_lane * lane_w) + (lane_w // 2)
        if face.detected:
            # Beri efek panah / glow di jalur yang sedang aktif
            cv2.circle(frame, (cx, cy), fr, (255, 200, 50), 3)
            cv2.line(frame, (target_x - 30, self.h - 20), (target_x + 30, self.h - 20), (50, 255, 50), 8)

        self.particles = update_draw_particles(frame, self.particles)

# ══════════════════════════════════════════════════════════
# GAME 3: TOWER STACK
# ══════════════════════════════════════════════════════════
class TowerStack:
    NAME = "TOWER STACK"
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.reset()

    def reset(self):
        self.tower = [] # list tinggi balok
        self.falling_block = None
        self.particles = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.speed = 5

    def update(self, frame, face):
        if self.game_over: return

        cx, cy, fr = face.get() if face.detected else (self.w//2, self.h - 100, 40)
        block_w, block_h = 60, 30

        if self.falling_block is None:
            self.falling_block = {'x': random.randint(100, self.w-100), 'y': -50, 'color': (random.randint(50,255), random.randint(50,255), random.randint(50,255))}

        # Update block jatuh
        fb = self.falling_block
        fb['y'] += self.speed
        
        # Hitung target tangkapan (puncak kepala / puncak tower)
        tower_height = len(self.tower) * block_h
        target_y = cy - fr - tower_height - (block_h // 2)

        # Cek Tangkapan
        if fb['y'] >= target_y and fb['y'] <= target_y + self.speed * 2:
            # Cek apakah blok sejajar dengan kepala
            if abs(fb['x'] - cx) < block_w:
                # Tertangkap!
                self.tower.append(fb['color'])
                self.score += 1
                self.speed += 0.5
                burst_particles(self.particles, int(fb['x']), int(fb['y']), fb['color'], 15)
                self.falling_block = None
            elif fb['y'] > target_y + block_h:
                # Meleset jatuh
                self.lives -= 1
                burst_particles(self.particles, int(fb['x']), int(fb['y']), (0,0,255), 20)
                self.falling_block = None
                if self.lives <= 0:
                    self.game_over = True

        # Gambar Blok Jatuh
        if self.falling_block:
            fx, fy = int(fb['x']), int(fb['y'])
            cv2.rectangle(frame, (fx - block_w//2, fy - block_h//2), (fx + block_w//2, fy + block_h//2), fb['color'], -1)
            cv2.rectangle(frame, (fx - block_w//2, fy - block_h//2), (fx + block_w//2, fy + block_h//2), (255,255,255), 2)

        # Gambar Tower di atas kepala
        if face.detected:
            cv2.ellipse(frame, (cx, cy-fr), (fr, 10), 0, 0, 360, (0, 255, 0), 2) # Piringan landasan
            for i, color in enumerate(self.tower):
                bx = cx
                by = cy - fr - (i * block_h) - (block_h // 2)
                cv2.rectangle(frame, (bx - block_w//2, by - block_h//2), (bx + block_w//2, by + block_h//2), color, -1)
                cv2.rectangle(frame, (bx - block_w//2, by - block_h//2), (bx + block_w//2, by + block_h//2), (255,255,255), 2)

        self.particles = update_draw_particles(frame, self.particles)

# ══════════════════════════════════════════════════════════
# MENU & MAIN
# ══════════════════════════════════════════════════════════
def draw_game_over(frame, score, game_name):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    text_center(frame, "GAME OVER",      h // 2 - 65, 1.9, (80, 80, 255), 3)
    text_center(frame, game_name,        h // 2 - 15, 0.65, (180, 180, 180))
    text_center(frame, f"SCORE: {score}", h // 2 + 35, 1.3, (255, 215, 50), 2)
    text_center(frame, "[R] Restart   [1/2/3] Ganti Game   [ESC] Keluar", h // 2 + 85, 0.48, (140, 140, 140))

def draw_start_screen(frame, blink):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    text_center(frame, "TIKTOK FILTER GAMES",      h // 2 - 100, 1.3,  (255, 50, 150), 3)
    text_center(frame, "[1]  FLAPPY FACE",  h // 2 - 5,   0.75, (50, 255, 50))
    text_center(frame, "[2]  SUBWAY DODGE",  h // 2 + 38,  0.75, (50, 200, 255))
    text_center(frame, "[3]  TOWER STACK", h // 2 + 81,  0.75, (255, 200, 50))
    if blink % 60 < 40:
        text_center(frame, "Pilih game dengan angka [1/2/3]", h // 2 + 130, 0.5, (200, 200, 200))

def main():
    cap = None
    for idx in range(2):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, test = c.read()
            if ret: cap = c; break

    if not cap:
        print("ERROR: Kamera tidak ditemukan!")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.namedWindow("TikTok Filter Games", cv2.WINDOW_NORMAL)
    face  = SmoothFace(alpha=0.35)
    
    games = [FlappyFace(cam_w, cam_h), SubwayDodge(cam_w, cam_h), TowerStack(cam_w, cam_h)]
    current = 0
    started = False
    frame_count = 0
    prev_t = time.time()
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)

        curr_t = time.time()
        fps = 0.9 * fps + 0.1 / (curr_t - prev_t + 1e-9)
        prev_t = curr_t
        frame_count += 1

        face.update(frame)
        game = games[current]

        if not started:
            draw_start_screen(frame, frame_count)
        else:
            game.update(frame, face)
            draw_hud(frame, game.score, game.lives, game.NAME, fps)
            if game.game_over: draw_game_over(frame, game.score, game.NAME)

        cv2.imshow("TikTok Filter Games", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('1'): current = 0; started = True; games[0].reset()
        elif key == ord('2'): current = 1; started = True; games[1].reset()
        elif key == ord('3'): current = 2; started = True; games[2].reset()
        elif key in (ord('r'), ord('R')): game.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()