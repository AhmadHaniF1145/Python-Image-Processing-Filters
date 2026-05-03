"""
========================================================
  PHOTOGRAPHIC EFFECTS à la Instagram/TikTok
  Mata Kuliah: Pengolahan Citra Digital
  Dibuat dengan: Python + OpenCV
========================================================

KONTROL KEYBOARD:
  [1-0] / [Q-R]  : Ganti efek filter
  [A] / [D]      : Intensitas filter -/+
  [S]            : Screenshot
  [V]            : Record video on/off
  [ESC]          : Keluar program
========================================================
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime

# ─────────────────────────────────────────────────────
# FUNGSI-FUNGSI EFEK FILTER
# ─────────────────────────────────────────────────────

def effect_normal(frame, intensity=1.0):
    """
    EFEK NORMAL - Tidak ada filter, tampilan kamera asli.
    Digunakan sebagai baseline/pembanding.
    """
    return frame.copy()


def effect_grayscale(frame, intensity=1.0):
    """
    EFEK GRAYSCALE (Hitam Putih)
    Teknik: Konversi warna BGR → Grayscale menggunakan rumus:
            Gray = 0.299*R + 0.587*G + 0.114*B
    Kemudian di-blend dengan frame asli berdasarkan intensitas.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1 - intensity, gray_bgr, intensity, 0)


def effect_sepia(frame, intensity=1.0):
    """
    EFEK SEPIA (Vintage Cokelat)
    Teknik: Transformasi matriks warna untuk menghasilkan tone hangat kecokelatan.
    Kernel matriks sepia menggeser kanal warna sehingga menghasilkan
    nuansa foto jadul (vintage).
    Formula:
      R_out = 0.393*R + 0.769*G + 0.189*B
      G_out = 0.349*R + 0.686*G + 0.168*B
      B_out = 0.272*R + 0.534*G + 0.131*B
    """
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.addWeighted(frame, 1 - intensity, sepia, intensity, 0)


def effect_blur(frame, intensity=1.0):
    """
    EFEK BLUR / BOKEH (Background Soft)
    Teknik: Gaussian Blur — menghaluskan gambar dengan kernel Gaussian.
    Kernel size menentukan seberapa blur. 
    Gaussian Blur menggunakan distribusi normal untuk pembobotan piksel.
    Semakin besar kernel → semakin blur.
    """
    k = max(1, int(intensity * 20))
    k = k if k % 2 == 1 else k + 1  # harus ganjil
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return blurred


def effect_edge_detection(frame, intensity=1.0):
    """
    EFEK EDGE DETECTION (Deteksi Tepi)
    Teknik: Algoritma Canny Edge Detection.
    1. Gaussian smoothing untuk mengurangi noise
    2. Gradient magnitude & direction (Sobel operator)
    3. Non-maximum suppression
    4. Double thresholding & edge tracking by hysteresis
    Menghasilkan garis-garis tepi objek seperti efek sketsa hitam putih.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low  = int(50  * (1 - intensity * 0.5))
    high = int(150 * (1 - intensity * 0.3))
    edges = cv2.Canny(gray, low, high)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr


def effect_cartoon(frame, intensity=1.0):
    """
    EFEK CARTOON / ANIME
    Teknik kombinasi dua proses:
    1. Bilateral Filter → menghaluskan warna tapi mempertahankan tepi (edge-preserving)
       - Ini beda dari Gaussian Blur karena mempertimbangkan jarak spatial DAN intensitas warna
    2. Canny Edge Detection → mengambil garis tepi
    Hasilnya: warna flat + garis hitam = tampilan kartun/anime!
    """
    # Step 1: Bilateral filter untuk warna flat
    n_iter = max(1, int(intensity * 7))
    color = frame.copy()
    for _ in range(n_iter):
        color = cv2.bilateralFilter(color, 9, 75, 75)
    
    # Step 2: Deteksi tepi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Gabungkan
    cartoon = cv2.bitwise_and(color, edges_bgr)
    return cartoon


def effect_warm(frame, intensity=1.0):
    """
    EFEK WARM (Filter Hangat / Golden Hour)
    Teknik: Manipulasi Look-Up Table (LUT) per kanal warna.
    - Meningkatkan kanal Merah (R) dan Hijau (G)
    - Menurunkan sedikit kanal Biru (B)
    LUT = tabel pemetaan nilai piksel 0-255 ke nilai baru.
    Seperti color grading di film/foto profesional!
    """
    factor = intensity * 50
    lut_r = np.clip(np.arange(256) + factor, 0, 255).astype(np.uint8)
    lut_g = np.clip(np.arange(256) + factor * 0.3, 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) - factor * 0.5, 0, 255).astype(np.uint8)
    
    result = frame.copy()
    result[:, :, 2] = cv2.LUT(frame[:, :, 2], lut_r)
    result[:, :, 1] = cv2.LUT(frame[:, :, 1], lut_g)
    result[:, :, 0] = cv2.LUT(frame[:, :, 0], lut_b)
    return result


def effect_cool(frame, intensity=1.0):
    """
    EFEK COOL (Filter Dingin / Cyan/Blue Tint)
    Teknik: Kebalikan dari warm filter.
    - Meningkatkan kanal Biru (B)
    - Menurunkan kanal Merah (R)
    Menghasilkan nuansa dingin, seperti filter "Winter" di Instagram.
    """
    factor = intensity * 50
    lut_r = np.clip(np.arange(256) - factor * 0.5, 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) + factor, 0, 255).astype(np.uint8)
    
    result = frame.copy()
    result[:, :, 2] = cv2.LUT(frame[:, :, 2], lut_r)
    result[:, :, 0] = cv2.LUT(frame[:, :, 0], lut_b)
    return result


def effect_sketch(frame, intensity=1.0):
    """
    EFEK SKETCH (Pensil/Gambar Tangan)
    Teknik: Dodge & Burn (Pencil Sketch)
    1. Konversi ke grayscale
    2. Invert gambar grayscale (negatif)
    3. Gaussian Blur pada gambar negatif
    4. Colour Dodge: gabungkan original gray dengan blur negatif
       Formula Dodge: result = gray / (1 - blur_neg/255)
    Menghasilkan efek seperti gambar pensil!
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    k = max(1, int(intensity * 40))
    k = k if k % 2 == 1 else k + 1
    blur = cv2.GaussianBlur(inv, (k, k), 0)
    
    # Dodge blend
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_bgr


def effect_vignette(frame, intensity=1.0):
    """
    EFEK VIGNETTE (Sudut Gelap)
    Teknik: Membuat mask ellips Gaussian yang gelap di pinggir.
    Vignette = efek darkening di sudut-sudut foto.
    Sering dipakai di fotografi portrait untuk fokus ke tengah.
    Menggunakan fungsi Gaussian 2D untuk gradien yang halus.
    """
    h, w = frame.shape[:2]
    
    # Buat mask Gaussian 2D
    sigma_x = w / (2 * intensity + 0.1)
    sigma_y = h / (2 * intensity + 0.1)
    
    x = cv2.getGaussianKernel(w, sigma_x)
    y = cv2.getGaussianKernel(h, sigma_y)
    mask = y * x.T
    mask = mask / mask.max()
    mask = np.dstack([mask] * 3)
    
    vignette = (frame * mask).astype(np.uint8)
    return vignette


def effect_hdr(frame, intensity=1.0):
    """
    EFEK HDR (High Dynamic Range)
    Teknik: Detail Enhancement menggunakan Bilateral Filter + CLAHE
    1. Detail Enhancement (cv2.detailEnhance) → mempertajam detail tekstur
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization):
       - Histogram equalization secara lokal (per blok)
       - Meningkatkan kontras di area gelap/terang secara adaptif
       - "Clip Limit" mencegah over-amplifikasi noise
    Seperti foto HDR yang sering dipakai di landscape photography!
    """
    # Detail enhance
    detail = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15 * intensity)
    
    # CLAHE pada channel L (lightness) di LAB colorspace
    lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0 * intensity, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def effect_negative(frame, intensity=1.0):
    """
    EFEK NEGATIVE (Foto Negatif)
    Teknik: Bitwise NOT / Inversi piksel.
    Formula: result = 255 - pixel_value
    Di-blend dengan frame asli berdasarkan intensitas.
    Seperti negatif film foto analog!
    """
    negative = cv2.bitwise_not(frame)
    return cv2.addWeighted(frame, 1 - intensity, negative, intensity, 0)


def effect_beauty(frame, intensity=1.0):
    """
    EFEK BEAUTY / SMOOTH SKIN (Filter Kecantikan)
    Teknik: Kombinasi Bilateral Filter + Unsharp Masking
    1. Bilateral Filter (multi-iterasi) → menghaluskan kulit/noise
       tanpa menghilangkan tepi/detail penting
    2. Unsharp Masking → mengembalikan ketajaman di area tepi
       Formula: sharp = original + amount * (original - blur)
    Seperti filter "Beauty" di kamera HP/TikTok!
    """
    # Bilateral smooth (edge-preserving)
    sigma = int(intensity * 80)
    smooth = cv2.bilateralFilter(frame, 15, sigma, sigma)
    
    # Unsharp mask untuk pertahankan ketajaman
    blur = cv2.GaussianBlur(smooth, (0, 0), 3)
    sharp_amount = 0.3 * intensity
    result = cv2.addWeighted(smooth, 1 + sharp_amount, blur, -sharp_amount, 0)
    return result


def effect_emboss(frame, intensity=1.0):
    """
    EFEK EMBOSS (Timbul / Relief)
    Teknik: Konvolusi dengan kernel Emboss.
    Kernel emboss memiliki nilai positif dan negatif yang menciptakan
    ilusi kedalaman 3D pada gambar.
    Hasilnya: gambar tampak seperti relief logam/batu!
    Ditambahkan ke nilai tengah (128) agar hasilnya netral.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2*intensity, -intensity,  0],
                       [-intensity,       1,        intensity],
                       [ 0,           intensity, 2*intensity]])
    embossed = cv2.filter2D(gray, -1, kernel) + 128
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)


def effect_pixelate(frame, intensity=1.0):
    """
    EFEK PIXELATE (Mozaik / 8-bit)
    Teknik: Downscale + Upscale menggunakan nearest-neighbor interpolation.
    1. Perkecil gambar drastis (nearest-neighbor, bukan linear)
    2. Perbesar kembali ke ukuran asli
    Nearest-neighbor tidak melakukan interpolasi → menghasilkan kotak pixel besar.
    Seperti efek censoring atau game retro 8-bit!
    """
    h, w = frame.shape[:2]
    scale = max(0.02, 1 - intensity * 0.95)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


# ─────────────────────────────────────────────────────
# DAFTAR SEMUA EFEK
# ─────────────────────────────────────────────────────
EFFECTS = [
    ("Normal",          effect_normal),
    ("Grayscale",       effect_grayscale),
    ("Sepia",           effect_sepia),
    ("Blur/Bokeh",      effect_blur),
    ("Edge Detection",  effect_edge_detection),
    ("Cartoon",         effect_cartoon),
    ("Warm",            effect_warm),
    ("Cool",            effect_cool),
    ("Sketch",          effect_sketch),
    ("Vignette",        effect_vignette),
    ("HDR",             effect_hdr),
    ("Negative",        effect_negative),
    ("Beauty",          effect_beauty),
    ("Emboss",          effect_emboss),
    ("Pixelate",        effect_pixelate),
]

KEY_MAP = {
    ord('1'): 0,  ord('2'): 1,  ord('3'): 2,
    ord('4'): 3,  ord('5'): 4,  ord('6'): 5,
    ord('7'): 6,  ord('8'): 7,  ord('9'): 8,
    ord('0'): 9,  ord('q'): 10, ord('w'): 11,
    ord('e'): 12, ord('r'): 13, ord('t'): 14,
}


# ─────────────────────────────────────────────────────
# FUNGSI UI OVERLAY
# ─────────────────────────────────────────────────────

def draw_ui(frame, effect_idx, intensity, recording, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Panel kiri — daftar efek
    panel_w = 210
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (0, 0), (panel_w, h), (15, 15, 15), -1)

    cv2.putText(frame, "PHOTO EFFECTS", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
    cv2.line(frame, (10, 35), (panel_w - 10, 35), (60, 60, 80), 1)

    keys = ['1','2','3','4','5','6','7','8','9','0','Q','W','E','R','T']
    for i, (name, _) in enumerate(EFFECTS):
        y = 58 + i * 28
        is_active = (i == effect_idx)
        if is_active:
            cv2.rectangle(frame, (6, y - 16), (panel_w - 6, y + 8),
                          (50, 100, 200), -1)
            color = (255, 255, 255)
        else:
            color = (140, 140, 160)
        cv2.putText(frame, f"[{keys[i]}] {name}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Panel bawah — info
    bar_h = 52
    cv2.rectangle(frame, (panel_w, h - bar_h), (w, h), (15, 15, 15), -1)

    eff_name = EFFECTS[effect_idx][0]
    cv2.putText(frame, f"Filter: {eff_name}", (panel_w + 12, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 1)

    # Intensity bar
    bar_x, bar_y = panel_w + 12, h - 14
    bar_len = 200
    cv2.rectangle(frame, (bar_x, bar_y - 8), (bar_x + bar_len, bar_y + 4),
                  (50, 50, 60), -1)
    fill = int(bar_len * intensity)
    cv2.rectangle(frame, (bar_x, bar_y - 8), (bar_x + fill, bar_y + 4),
                  (50, 200, 120), -1)
    cv2.putText(frame, f"Intensity [{int(intensity*100)}%]",
                (bar_x + bar_len + 10, bar_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

    # Kontrol hint
    hint = "[A/D] Intensity  [S] Screenshot  [V] Record  [ESC] Exit"
    cv2.putText(frame, hint, (panel_w + 12, h - bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 25, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 60, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

    return frame


# ─────────────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────────────

def main():
    print(__doc__)
    print("Membuka kamera...")

    # Buat folder output
    os.makedirs("output", exist_ok=True)

    # Coba berbagai backend & index kamera (fix untuk Windows)
    cap = None
    backends = [
        (cv2.CAP_MSMF,  "MSMF (Media Foundation)"),   # Windows native
        (cv2.CAP_DSHOW, "DirectShow"),                  # DirectShow
        (cv2.CAP_ANY,   "Auto"),                        # Otomatis
    ]
    for idx in range(3):          # coba index kamera 0, 1, 2
        for backend, bname in backends:
            print(f"  Mencoba index={idx}, backend={bname}...")
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ret, test = c.read()
                if ret and test is not None:
                    cap = c
                    print(f"  ✓ Kamera ditemukan! index={idx}, backend={bname}")
                    break
                c.release()
        if cap:
            break

    if cap is None or not cap.isOpened():
        print("\nERROR: Tidak ada kamera yang bisa dibuka.")
        print("Pastikan:")
        print("  1. Kamera/webcam terpasang dan tidak dipakai aplikasi lain")
        print("  2. Driver kamera sudah terinstall")
        print("  3. Izin kamera sudah diberikan di Pengaturan Windows")
        input("\nTekan Enter untuk keluar...")
        return

    # Biarkan kamera pakai resolusi default (jangan paksa 1280x720)
    # Warm-up: buang beberapa frame awal yang sering corrupt di Windows/Legion
    print("  Warming up kamera...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    # Cek resolusi aktual
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolusi kamera: {actual_w}x{actual_h}")

    effect_idx = 0
    intensity = 0.7
    recording = False
    video_writer = None

    prev_time = time.time()
    fps = 0

    print("Kamera aktif! Tekan tombol sesuai panduan di layar.")
    print("Tekan [ESC] untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        # Validasi frame: skip jika kosong atau dimensi tidak valid
        if not ret or frame is None or frame.size == 0:
            continue
        if len(frame.shape) < 3 or frame.shape[0] < 10 or frame.shape[1] < 10:
            continue

        # Hitung FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-9))
        prev_time = curr_time

        # Terapkan efek
        effect_name, effect_fn = EFFECTS[effect_idx]
        try:
            result = effect_fn(frame, intensity)
        except Exception as e:
            result = frame.copy()
            print(f"Error pada efek {effect_name}: {e}")

        # Gambar UI
        result = draw_ui(result, effect_idx, intensity, recording, fps)

        # Tampilkan
        cv2.imshow("Instagram/TikTok Effects - Pengolahan Citra", result)

        # Rekam video
        if recording and video_writer:
            video_writer.write(result)

        # Input keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in KEY_MAP:
            effect_idx = KEY_MAP[key]
            print(f"Filter: {EFFECTS[effect_idx][0]}")
        elif key == ord('a'):
            intensity = max(0.0, intensity - 0.05)
        elif key == ord('d'):
            intensity = min(1.0, intensity + 0.05)
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"output/screenshot_{effect_name.replace('/','_')}_{ts}.jpg"
            cv2.imwrite(fname, result)
            print(f"Screenshot disimpan: {fname}")
        elif key == ord('v'):
            if not recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"output/video_{ts}.mp4"
                h_v, w_v = result.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(fname, fourcc, 20, (w_v, h_v))
                recording = True
                print(f"Rekaman dimulai: {fname}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("Rekaman dihentikan & disimpan.")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("\nProgram selesai. File tersimpan di folder 'output/'")


if __name__ == "__main__":
    main()