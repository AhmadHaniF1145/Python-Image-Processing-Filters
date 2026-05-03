"""
================================================
  PHOTOGRAPHIC EFFECTS - Pengolahan Citra Digital
  Ahmad Hanif Abiyyu Khrisna
  2125640006
  STr-LJ Teknik Elektronika 2025
  Filter: Glitch & Golden Hour
  Kontrol: [1] Glitch  [2] Golden Hour
           [A]/[D] Intensitas  [S] Screenshot
           [V] Record  [ESC] Keluar
================================================
"""

import cv2
import numpy as np
import time
import math
import random
import os
from datetime import datetime

os.makedirs("output", exist_ok=True)


# ================================================
#  HELPER
# ================================================

def to_float(img):
    return img.astype(np.float32) / 255.0

def to_uint8(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def blend(original, filtered, t):
    return cv2.addWeighted(original, 1.0 - t, filtered, t, 0)


# ================================================
#  EFEK 1 — GLITCH
#
#  Glitch mensimulasikan kerusakan sinyal digital/analog.
#  Ada 3 sub-teknik yang digabung:
#
#  [A] CHROMATIC ABERRATION (RGB Split)
#      Lensa atau sensor kamera yang tidak sempurna
#      menyebabkan setiap panjang gelombang cahaya
#      (R, G, B) fokus di titik yang sedikit berbeda.
#      Simulasi: pisahkan frame menjadi 3 kanal R, G, B,
#      lalu geser kanal R ke kanan dan B ke kiri sebesar
#      beberapa piksel. Hasilnya: pinggiran objek
#      tampak "fringing" merah-biru.
#
#  [B] BLOCK CORRUPTION
#      Simulasi packet loss atau buffer error pada
#      transmisi video digital. Blok-blok piksel di
#      baris tertentu digeser secara horizontal secara
#      acak, seolah data video rusak/korup.
#      Tidak terjadi setiap frame — probabilitas 35%
#      agar terlihat organik (tidak terlalu sering).
#
#  [C] SCANLINE DROPOUT
#      Pada sinyal CRT atau tape VHS yang rusak,
#      beberapa baris kadang tidak terbaca sama sekali
#      dan tampil sebagai garis hitam.
#      Simulasi: baris tertentu di-set ke 0 (hitam).
# ================================================

def fx_glitch(frame, t=1.0):

    # [A] Chromatic Aberration
    b, g, r = cv2.split(frame)
    shift = int(t * 14)
    r = np.roll(r,  shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    g = np.roll(g,  int(shift * 0.3), axis=0)
    out = cv2.merge([b, g, r])

    # [B] Block Corruption
    h, w = out.shape[:2]
    if random.random() < 0.35 * t:
        jumlah_blok = random.randint(1, int(3 * t) + 1)
        for _ in range(jumlah_blok):
            tinggi_blok = random.randint(4, 35)
            posisi_y    = random.randint(0, h - tinggi_blok - 1)
            geser_x     = random.randint(-int(70 * t), int(70 * t))
            out[posisi_y : posisi_y + tinggi_blok] = np.roll(
                out[posisi_y : posisi_y + tinggi_blok], geser_x, axis=1)

    # [C] Scanline Dropout
    if random.random() < 0.4 * t:
        for _ in range(random.randint(1, 3)):
            baris = random.randint(0, h - 1)
            out[baris] = 0

    return out


# ================================================
#  EFEK 2 — GOLDEN HOUR
#
#  Golden Hour adalah kondisi cahaya ~1 jam setelah
#  matahari terbit atau sebelum terbenam. Matahari
#  posisinya rendah sehingga cahaya melewati atmosfer
#  lebih tebal. Panjang gelombang pendek (biru) tersebar,
#  hanya gelombang panjang (merah-oranye) yang sampai.
#
#  Simulasi dengan 4 tahap:
#
#  [A] WARM COLOR SHIFT (Kurva per kanal / LUT)
#      Setiap kanal R, G, B dimanipulasi secara
#      independen menggunakan Look-Up Table (LUT).
#      LUT = tabel 256 nilai: input piksel 0-255
#      dipetakan ke nilai output baru.
#      R dinaikkan  -> push warna ke merah/oranye
#      B diturunkan -> kurangi warna dingin
#
#  [B] HIGHLIGHT BLOOM (Masking area terang)
#      Di golden hour, area terang tampak "bleeding"
#      cahaya hangat ke sekitarnya.
#      Cara: buat mask hanya dari area terang
#      (luminance dipangkatkan -> makin selektif),
#      lalu tambahkan warna warm (R+) ke area tersebut.
#
#  [C] GOLDEN HAZE (Atmospheric scattering)
#      Atmosfer golden hour menciptakan haze kekuningan.
#      Simulasi: Gaussian blur -> blend ke frame asli
#      hanya di area terang via luminance mask.
#
#  [D] VIGNETTE (Gaussian 2D mask)
#      Penggelapan sudut frame.
#      Dibuat dari outer product dua vektor Gaussian:
#      mask = gauss_y . gauss_x_T   -> ellipse 2D
#      Piksel di tepi dikalikan nilai < 1 (lebih gelap).
# ================================================

def fx_golden_hour(frame, t=1.0):

    # [A] Warm Color Shift via LUT
    lut_r = np.clip(np.arange(256) * 1.12 + 15 * t, 0, 255).astype(np.uint8)
    lut_g = np.clip(np.arange(256) * 1.04 +  5 * t, 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) * 0.78 - 20 * t, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(frame)
    r = cv2.LUT(r, lut_r)
    g = cv2.LUT(g, lut_g)
    b = cv2.LUT(b, lut_b)
    out = cv2.merge([b, g, r])

    # [B] Highlight Bloom
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    highlight_mask = np.power(gray, 2.5)
    highlight_mask = np.dstack([highlight_mask] * 3)

    warm_layer = np.zeros_like(out, dtype=np.float32)
    warm_layer[:, :, 2] = 0.28 * t
    warm_layer[:, :, 1] = 0.10 * t

    out_f = to_float(out) + warm_layer * highlight_mask
    out = to_uint8(np.clip(out_f, 0, 1))

    # [C] Golden Haze
    haze     = cv2.GaussianBlur(out, (0, 0), 18)
    gray2    = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum_mask = np.dstack([gray2] * 3)
    strength = 0.18 * t
    out_f2   = to_float(out) * (1 - strength * lum_mask) + \
               to_float(haze) * strength * lum_mask
    out = to_uint8(np.clip(out_f2, 0, 1))

    # [D] Vignette
    h, w = out.shape[:2]
    gx   = cv2.getGaussianKernel(w, int(w * 0.55))
    gy   = cv2.getGaussianKernel(h, int(h * 0.55))
    mask = (gy @ gx.T).astype(np.float32)
    mask = mask / mask.max()
    vignette = 1.0 - 0.55 * t * (1.0 - mask)
    out = to_uint8(to_float(out) * np.dstack([vignette] * 3))

    return blend(frame, out, t)


# ================================================
#  DAFTAR EFEK
# ================================================
EFFECTS = [
    ("Glitch",       fx_glitch),
    ("Golden Hour",  fx_golden_hour),
]

KEY_MAP = {ord('1'): 0, ord('2'): 1}


# ================================================
#  UI OVERLAY
# ================================================
def draw_ui(frame, eidx, intensity, recording, fps):
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 52), (w, h), (8, 8, 10), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    name = EFFECTS[eidx][0]
    cv2.putText(frame, f"Filter: {name}", (12, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 215, 255), 2)

    for i, (n, _) in enumerate(EFFECTS):
        col = (255, 255, 255) if i == eidx else (100, 100, 120)
        cv2.putText(frame, f"[{i+1}] {n}", (12 + i * 170, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    bx, by = 360, h - 12
    cv2.rectangle(frame, (bx, by - 7), (bx + 150, by + 3), (35, 35, 45), -1)
    cv2.rectangle(frame, (bx, by - 7),
                  (bx + int(150 * intensity), by + 3), (40, 195, 110), -1)
    cv2.putText(frame, f"[A/D] {int(intensity*100)}%", (bx + 158, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 150), 1)

    cv2.putText(frame, f"{fps:.0f}fps", (w - 65, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 170, 70), 1)
    cv2.putText(frame, "[S] Screenshot  [V] Rec  [ESC] Exit",
                (w - 265, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 100), 1)

    if recording:
        cv2.circle(frame, (w - 18, 20), 8, (0, 0, 210), -1)
        cv2.putText(frame, "REC", (w - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 230), 1)

    return frame


# ================================================
#  MAIN
# ================================================
def main():
    print(__doc__)

    cap = None
    for idx in range(3):
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ret, test = c.read()
                if ret and test is not None:
                    cap = c
                    print(f"Kamera OK: index={idx}")
                    break
                c.release()
        if cap:
            break

    if not cap:
        print("ERROR: Kamera tidak ditemukan!")
        input("Enter untuk keluar...")
        return

    # Warm-up dulu SEBELUM set resolusi (Legion butuh ini)
    print("Warming up kamera...")
    import time as _t; _t.sleep(1.0)
    for _ in range(15):
        try:
            cap.read()
        except Exception:
            pass
        time.sleep(0.06)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi: {w}x{h}")

    WIN = "Photographic Effects - Pengolahan Citra"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    eidx       = 0
    intensity  = 0.85
    recording  = False
    vid_writer = None
    prev_t     = time.time()
    fps        = 30.0

    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            time.sleep(0.03)
            continue
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.01)
            continue
        if len(frame.shape) < 3 or frame.shape[0] < 10 or frame.shape[1] < 10:
            continue

        frame = cv2.flip(frame, 1)

        curr_t = time.time()
        fps    = 0.9 * fps + 0.1 / (curr_t - prev_t + 1e-9)
        prev_t = curr_t

        name, fn = EFFECTS[eidx]
        try:
            result = fn(frame, intensity)
        except Exception as e:
            result = frame.copy()
            cv2.putText(result, str(e), (10, 50),
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
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"output/{name.replace(' ','_')}_{ts}.jpg"
            cv2.imwrite(path, result)
            print(f"Screenshot: {path}")
        elif key == ord('v'):
            if not recording:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"output/video_{ts}.mp4"
                fh2, fw2 = result.shape[:2]
                vid_writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (fw2, fh2))
                recording = True
                print(f"Recording: {path}")
            else:
                recording = False
                vid_writer.release()
                vid_writer = None
                print("Recording selesai.")

    cap.release()
    if vid_writer:
        vid_writer.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()