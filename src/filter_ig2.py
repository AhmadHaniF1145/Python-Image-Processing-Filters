    import cv2
import numpy as np

# =========================
# Fungsi Effects
# =========================

def effect_original(frame):
    return frame

def effect_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def effect_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(frame, kernel)

def effect_negative(frame):
    return cv2.bitwise_not(frame)

def effect_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def effect_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def effect_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def effect_warm(frame):
    increase = np.array([0, 10, 20])  # BGR
    return cv2.add(frame, increase)

def effect_cool(frame):
    decrease = np.array([20, 10, 0])
    return cv2.subtract(frame, decrease)

# =========================
# Inisialisasi Kamera
# =========================
cap = cv2.VideoCapture(0)

current_effect = 0

effects = [
    ("Original", effect_original),
    ("Grayscale", effect_grayscale),
    ("Sepia", effect_sepia),
    ("Negative", effect_negative),
    ("Blur", effect_blur),
    ("Edge", effect_edge),
    ("Cartoon", effect_cartoon),
    ("Warm", effect_warm),
    ("Cool", effect_cool),
]

print("Tekan angka 0-8 untuk ganti efek")
print("Tekan 'q' untuk keluar")

# =========================
# Loop Kamera
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, func = effects[current_effect]
    output = func(frame)

    # Handle grayscale biar tetap bisa ditampilkan
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    # Tampilkan nama efek
    cv2.putText(output, name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Camera Effects", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key >= ord('0') and key <= ord('8'):
        current_effect = key - ord('0')

cap.release()
cv2.destroyAllWindows()