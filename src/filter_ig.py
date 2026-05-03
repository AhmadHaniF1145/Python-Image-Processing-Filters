import cv2
import numpy as np
import os
import random
import time

# 1. FUNGSI UNTUK MENUMPUK GAMBAR (OVERLAY)
def overlay_transparent(background, overlay, x, y):
    """
    Fungsi untuk menempelkan gambar (overlay) ke atas frame kamera.
    Mendukung gambar transparan (PNG dengan Alpha Channel).
    """
    bg_h, bg_w, bg_channels = background.shape
    ol_h, ol_w = overlay.shape[:2]

    # Pastikan posisi overlay tidak keluar dari batas frame kamera
    if x >= bg_w or y >= bg_h:
        return background
    
    h, w = min(ol_h, bg_h - y), min(ol_w, bg_w - x)
    ol_x, ol_y = max(0, -x), max(0, -y)
    x, y = max(0, x), max(0, y)

    overlay_crop = overlay[ol_y:ol_y+h, ol_x:ol_x+w]
    background_crop = background[y:y+h, x:x+w]

    # Jika gambar overlay memiliki channel alpha (transparansi)
    if overlay.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            background_crop[:, :, c] = (alpha * overlay_crop[:, :, c] +
                                        alpha_inv * background_crop[:, :, c])
    else:
        # Jika gambar biasa (JPG), langsung timpa saja
        background_crop[:] = overlay_crop

    return background

def main():
    # 2. INISIALISASI CAMERA & FACE DETECTOR
    cap = cv2.VideoCapture(0)
    # Menggunakan model Haar Cascade bawaan OpenCV untuk mendeteksi wajah depan
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. MEMUAT GAMBAR-GAMBAR FILTER
    folder_path = "assets" # PASTIKAN FOLDER INI ADA DAN BERISI GAMBAR
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    target_size = (150, 150) # Ukuran kotak gambar di atas kepala
    
    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED agar alpha channel (transparan) terbaca
        img = cv2.resize(img, target_size)
        images.append((img, file.split('.')[0])) # Simpan gambar dan namanya

    if not images:
        print("Error: Tidak ada gambar di dalam folder 'assets'.")
        return

    # 4. VARIABEL STATE (STATUS PROGRAM)
    # state bisa berupa: 'idle' (menunggu), 'shuffling' (mengacak), 'result' (hasil akhir)
    state = 'idle'
    start_time = 0
    shuffle_duration = 3.0 # Berapa lama efek acak berjalan (dalam detik)
    current_image_index = 0
    final_result = None

    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Efek cermin agar pergerakan natural
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5. MENDETEKSI WAJAH
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Ambil wajah pertama yang terdeteksi
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # Koordinat untuk menaruh gambar/teks di atas kepala
            box_x = x + (w // 2) - (target_size[0] // 2)
            box_y = y - target_size[1] - 20 # 20 pixel di atas wajah

            # 6. LOGIKA STATE MACHINE (TAMPILAN BERDASARKAN STATUS)
            if state == 'idle':
                # Gambar kotak hitam dan teks "Apa Standmu?"
                cv2.rectangle(frame, (box_x, box_y), (box_x + target_size[0], box_y + target_size[1]), (0,0,0), -1)
                cv2.putText(frame, "Apa Standmu?", (box_x + 10, box_y + target_size[1]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "SPASI: Mulai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif state == 'shuffling':
                # Mengacak gambar dengan cepat
                current_image_index = (current_image_index + 1) % len(images)
                img_to_show, _ = images[current_image_index]
                frame = overlay_transparent(frame, img_to_show, box_x, box_y)
                
                cv2.putText(frame, "MENGACAK...", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Cek apakah waktu acak sudah selesai
                if time.time() - start_time > shuffle_duration:
                    state = 'result'
                    final_result = random.choice(images) # Pilih satu gambar acak sebagai hasil akhir

            elif state == 'result':
                # Tampilkan gambar final
                img_to_show, name = final_result
                frame = overlay_transparent(frame, img_to_show, box_x, box_y)
                
                # Tampilkan nama hasil di bawah gambar
                cv2.putText(frame, name.upper(), (box_x, box_y + target_size[1] + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPASI: Ulang Kembali", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            # Jika tidak ada wajah terdeteksi
            cv2.putText(frame, "Wajah tidak terdeteksi!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Tampilkan frame ke layar
        cv2.imshow('Filter Randomizer ala IG/TikTok', frame)

        # 7. MENANGKAP INPUT KEYBOARD
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '): # Tombol SPASI ditekan
            if state == 'idle' or state == 'result':
                state = 'shuffling'
                start_time = time.time()
        elif key == ord('q'): # Tombol Q ditekan
            break

    # Bersihkan memori saat program selesai
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()