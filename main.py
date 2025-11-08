import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer
from pathlib import Path

# Inisialisasi MediaPipe untuk deteksi wajah dan tangan
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Setup audio dengan pygame
mixer.init()
ASSET_PATH = Path('Asset/Images')

# Muat semua gambar aksesori dari folder sebagai BGRA (pastikan background transparan)
accessory_files = list(ASSET_PATH.glob("*.png"))
accessory_images = []
for file in accessory_files:
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)  # Memuat gambar dengan transparansi bila ada
    if img is None:
        continue
    # Jika tidak memiliki alpha, tambahkan alpha penuh (255)
    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.ones_like(b, dtype=b.dtype) * 255
        img = cv2.merge((b, g, r, a))
    accessory_images.append((file.stem, img))  # Simpan nama & gambar (BGRA)

# Setup video capture
cap = cv2.VideoCapture(0)

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlay img_overlay (BGRA) onto img (BGR) at position x,y with alpha handling and clipping."""
    h, w = img.shape[:2]
    ol_h, ol_w = img_overlay.shape[:2]

    # Compute overlay region in destination image
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ol_w)
    y2 = min(h, y + ol_h)

    if x1 >= x2 or y1 >= y2:
        return img  # di luar frame

    # Compute corresponding source region
    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    # Slice regions
    overlay_slice = img_overlay[src_y1:src_y2, src_x1:src_x2]
    img_slice = img[y1:y2, x1:x2]

    # Normalize alpha to 0..1
    alpha = overlay_slice[:, :, 3:] / 255.0
    inv_alpha = 1.0 - alpha

    # Blend
    for c in range(3):
        img_slice[:, :, c] = (alpha[:, :, 0] * overlay_slice[:, :, c] +
                              inv_alpha[:, :, 0] * img_slice[:, :, c])

    img[y1:y2, x1:x2] = img_slice
    return img

# Fungsi untuk menambahkan aksesori ke wajah (menggunakan bounding box dan landmark spesifik)
def add_accessory(frame, landmarks, accessory_img, scale=1.0, position=None):
    h_frame, w_frame, _ = frame.shape

    xs = np.array([p[0] for p in landmarks])
    ys = np.array([p[1] for p in landmarks])
    face_min_x, face_max_x = xs.min(), xs.max()
    face_min_y, face_max_y = ys.min(), ys.max()
    face_w = face_max_x - face_min_x
    face_h = face_max_y - face_min_y
    face_cx = int((face_min_x + face_max_x) / 2)

    # Default placement values
    if position == "hat":
        # Lebar topi berdasarkan lebar wajah
        width = int(face_w * 1.6 * scale)
        if width <= 0:
            return frame
        height = int(width * accessory_img.shape[0] / accessory_img.shape[1])
        x = int(face_cx - width / 2)
        # Tempatkan sedikit di atas bagian atas wajah (forehead)
        y = int(face_min_y - height * 0.75)
    elif position == "mustache":
        # Gunakan lebar mulut sebagai referensi
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_cx = int((mouth_left[0] + mouth_right[0]) / 2)
        mouth_w = abs(mouth_right[0] - mouth_left[0])
        width = int(max(mouth_w * 1.6, face_w * 0.25) * scale)
        if width <= 0:
            return frame
        height = int(width * accessory_img.shape[0] / accessory_img.shape[1])
        x = int(mouth_cx - width / 2)
        # Tempatkan sedikit di atas dan sejajar dengan mulut (agar berada di atas dagu)
        mouth_cy = int((mouth_left[1] + mouth_right[1]) / 2)
        nose_base = landmarks[1]
        # antara hidung dan mulut
        y = int(nose_base[1] + (mouth_cy - nose_base[1]) * 0.6 - height / 2)
    else:  # sunglasses / glasses
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_cx = int((left_eye[0] + right_eye[0]) / 2)
        eye_cy = int((left_eye[1] + right_eye[1]) / 2)
        eye_dist = np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        width = int(max(eye_dist * 2.2, face_w * 0.35) * scale)
        if width <= 0:
            return frame
        height = int(width * accessory_img.shape[0] / accessory_img.shape[1])
        x = int(eye_cx - width / 2)
        y = int(eye_cy - height / 2)

    # Resize aksesori
    accessory_resized = cv2.resize(accessory_img, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)

    # Overlay dengan alpha handling dan clipping
    frame = overlay_image_alpha(frame, accessory_resized, x, y)
    return frame

# Fungsi untuk mendeteksi jari telunjuk menunjuk ke gambar
def get_selected_accessory(frame, hand_landmarks, accessory_positions):
    if not hand_landmarks:
        return None
    for hand in hand_landmarks:
        finger_tip = hand.landmark[8]  # jari telunjuk
        x_tip = int(finger_tip.x * frame.shape[1])
        y_tip = int(finger_tip.y * frame.shape[0])

        for idx, (name, (x, y, w, h)) in enumerate(accessory_positions):
            if x < x_tip < x + w and y < y_tip < y + h:
                return idx
    return None

# --- Main Loop ---
selected_idx = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape
    accessory_positions = []

    # --- Tampilkan semua aksesori di bagian atas layar (thumbnail dengan alpha) ---
    thumb_size = 80
    margin = 10
    x_offset = 10

    # Membatasi x_offset agar thumbnail tidak keluar dari batas
    max_x_offset = w - thumb_size - margin  # Batas kanan frame

    for i, (name, img) in enumerate(accessory_images):
        if x_offset > max_x_offset:
            break  # Hentikan jika thumbnail melebihi batas

        # Siapkan thumbnail (BGRA)
        thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        # Overlay thumbnail di frame dengan alpha
        frame = overlay_image_alpha(frame, thumb, x_offset, 10)

        accessory_positions.append((name, (x_offset, 10, thumb_size, thumb_size)))
        x_offset += thumb_size + margin  # Pindahkan ke posisi berikutnya

    # --- Deteksi pilihan dengan tangan ---
    if hand_results.multi_hand_landmarks:
        idx = get_selected_accessory(frame, hand_results.multi_hand_landmarks, accessory_positions)
        if idx is not None:
            selected_idx = idx

    # --- Tambahkan aksesori ke wajah jika terpilih ---
    if selected_idx is not None and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
            _, accessory_img = accessory_images[selected_idx]
            accessory_name = accessory_images[selected_idx][0]

            # Tentukan posisi berdasarkan jenis aksesori (nama file)
            name_lower = accessory_name.lower()
            if 'hat' in name_lower or 'top' in name_lower:
                frame = add_accessory(frame, landmarks, accessory_img, scale=1.0, position="hat")
            elif 'mustache' in name_lower or 'moustache' in name_lower or 'kumis' in name_lower:
                frame = add_accessory(frame, landmarks, accessory_img, scale=1.0, position="mustache")
            else:
                frame = add_accessory(frame, landmarks, accessory_img, scale=1.0, position="glasses")

    # --- Tampilkan hasil ---
    cv2.imshow("Party Clown - Gesture Filter Selection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
