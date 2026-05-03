import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

def draw_drone(frame, center_x, center_y):
    """Fungsi untuk menggambar FPV drone sederhana"""
    # Menggambar 4 baling-baling (propellers)
    cv2.circle(frame, (center_x - 20, center_y - 20), 12, (200, 200, 200), 2)
    cv2.circle(frame, (center_x + 20, center_y - 20), 12, (200, 200, 200), 2)
    cv2.circle(frame, (center_x - 20, center_y + 20), 12, (200, 200, 200), 2)
    cv2.circle(frame, (center_x + 20, center_y + 20), 12, (200, 200, 200), 2)
    # Menggambar frame/body drone (warna kuning)
    cv2.rectangle(frame, (center_x - 15, center_y - 15), 
                  (center_x + 15, center_y + 15), (0, 255, 255), -1)
    # Menggambar kamera FPV di depan (merah)
    cv2.circle(frame, (center_x + 15, center_y), 5, (0, 0, 255), -1)

def main():
    cap = cv2.VideoCapture(0)
    
    # Variabel Game
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    drone_x = 100
    drone_y = height // 2
    drone_radius = 25 # Estimasi ukuran hitbox drone
    
    obstacle_width = 70
    obstacle_x = width
    gap_size = 200
    obstacle_y_top = random.randint(50, height - gap_size - 50)
    obstacle_speed = 15
    
    score = 0
    game_over = False

    print("Kamera aktif! Naik-turunkan kepalamu untuk mengendalikan drone. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame agar seperti cermin
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not game_over:
            # 1. Deteksi Wajah untuk Kontrol
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    # Ambil titik tengah wajah (hidung)
                    nose_y = int((bboxC.ymin + bboxC.height / 2) * height)
                    
                    # Buat pergerakan drone lebih smooth mengikuti hidung
                    drone_y += int((nose_y - drone_y) * 0.2)
                    
                    # Gambar kotak target di wajah pemain
                    nose_x = int((bboxC.xmin + bboxC.width / 2) * width)
                    cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

            # 2. Update Rintangan (Obstacles)
            obstacle_x -= obstacle_speed
            
            # Jika rintangan lewat, reset ke kanan dan tambah skor
            if obstacle_x < -obstacle_width:
                obstacle_x = width
                obstacle_y_top = random.randint(50, height - gap_size - 50)
                score += 1
                obstacle_speed += 1 # Makin lama makin cepat!

            # 3. Deteksi Tabrakan (Collision Detection)
            # Hitbox rintangan atas
            rect_top = [obstacle_x, 0, obstacle_x + obstacle_width, obstacle_y_top]
            # Hitbox rintangan bawah
            rect_bottom = [obstacle_x, obstacle_y_top + gap_size, obstacle_x + obstacle_width, height]
            
            # Logika tabrakan sederhana
            if (drone_x + drone_radius > rect_top[0] and drone_x - drone_radius < rect_top[2]):
                if (drone_y - drone_radius < rect_top[3]) or (drone_y + drone_radius > rect_bottom[1]):
                    game_over = True

        # --- TAHAP PENGGAMBARAN (RENDERING) ---
        
        # Gambar Rintangan (Pilar Hijau)
        cv2.rectangle(frame, (obstacle_x, 0), (obstacle_x + obstacle_width, obstacle_y_top), (0, 200, 0), -1)
        cv2.rectangle(frame, (obstacle_x, obstacle_y_top + gap_size), (obstacle_x + obstacle_width, height), (0, 200, 0), -1)
        
        # Gambar Drone
        draw_drone(frame, drone_x, drone_y)
        
        # Gambar UI (Skor & Status)
        cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        if game_over:
            # Beri efek gelap/overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            cv2.putText(frame, "CRASHED!", (width//2 - 150, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.putText(frame, "Tekan 'r' untuk Restart", (width//2 - 150, height//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("TikTok Drone Game Filter", frame)

        # Kontrol Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and game_over:
            # Reset Game
            game_over = False
            score = 0
            obstacle_x = width
            obstacle_speed = 15
            drone_y = height // 2

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()