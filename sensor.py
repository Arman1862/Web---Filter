import cv2
import mediapipe as mp
import numpy as np
import requests
import pyvirtualcam

# Inisialisasi MediaPipe Hands dan Face Mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# URL gambar untuk masking wajah
IMAGE_URL = "https://i.pinimg.com/1200x/43/8b/c6/438bc647f7f36f1115ad28cd5ee8c059.jpg"
try:
    response = requests.get(IMAGE_URL)
    img_data = response.content
    img_array = np.frombuffer(img_data, np.uint8)
    overlay_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("Gambar berhasil diunduh!")
except Exception as e:
    print(f"Gagal mengunduh gambar dari URL: {e}")
    overlay_img = None

cap = cv2.VideoCapture(0) # Mengakses kamera fisik (index 0)

if not cap.isOpened():
    print("Error: Kamera fisik tidak dapat dibuka.")
    exit()

# Inisialisasi kamera virtual
# MirAi set ukurannya 640x480 untuk performa lebih baik
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    print(f'Virtual camera siap! Output ke: {cam.device}')
    
    while True:
        success, img_cam = cap.read()
        if not success:
            continue
        
        # Ubah ukuran gambar agar sesuai dengan virtual camera
        img_cam = cv2.resize(img_cam, (640, 480))
        img_cam = cv2.flip(img_cam, 1)
        
        imgRGB = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
        
        # Memproses deteksi tangan dan wajah
        hand_results = hands.process(imgRGB)
        face_results = face_mesh.process(imgRGB)

        # Mengaplikasikan filter ke wajah
        if face_results.multi_face_landmarks and overlay_img is not None:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = img_cam.shape
                face_oval_indices = set()
                for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
                    face_oval_indices.add(connection[0])
                    face_oval_indices.add(connection[1])
                
                face_oval_points = [
                    (int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h))
                    for p in face_oval_indices
                ]
                
                mask = np.zeros_like(img_cam, dtype=np.uint8)
                if face_oval_points:
                    hull = cv2.convexHull(np.array(face_oval_points, dtype=np.int32))
                    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
                
                (x, y, w_mask, h_mask) = cv2.boundingRect(mask[:, :, 0])
                
                if w_mask > 0 and h_mask > 0:
                    resized_overlay = cv2.resize(overlay_img, (w_mask, h_mask))
                    face_region = img_cam[y:y+h_mask, x:x+w_mask]
                    mask_region = mask[y:y+h_mask, x:x+w_mask]
                    
                    masked_overlay = cv2.bitwise_and(resized_overlay, mask_region)
                    face_region_masked = cv2.bitwise_and(face_region, cv2.bitwise_not(mask_region))
                    img_cam[y:y+h_mask, x:x+w_mask] = cv2.add(face_region_masked, masked_overlay)
        
        # Menggambar landmark tangan
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_cam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Kirim frame yang sudah dimodifikasi ke virtual camera
        frame_to_send = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
        cam.send(frame_to_send)