import pygame
# import chess
import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO

# Tambahkan library PyVirtualCam di sini
import pyvirtualcam

# Inisialisasi Pygame
pygame.init()

# Ukuran jendela dan papan
# MirAi ubah lebar jadi cuma 600 karena kita cuma butuh tampilan webcamnya
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sensor & Filter Camera")

# Warna
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0)
POINTER_COLOR = (255, 0, 255)

# Ukuran kotak catur
SQUARE_SIZE = HEIGHT // 8

# Variabel untuk penyesuaian
SENSITIVITY_MULTIPLIER = 1.5
PINCH_THRESHOLD = 0.08

# URL gambar untuk masking wajah
IMAGE_URL = "https://i.pinimg.com/1200x/43/8b/c6/438bc647f7f36f1115ad28cd5ee8c059.jpg"

# Memuat gambar dari URL
try:
    response = requests.get(IMAGE_URL)
    img_data = response.content
    img_array = np.frombuffer(img_data, np.uint8)
    overlay_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("Gambar berhasil diunduh!")
except Exception as e:
    print(f"Gagal mengunduh gambar dari URL: {e}")
    overlay_img = None

# Bagian kode ini MirAi biarin di-comment karena kita lagi fokus ke video virtual
# images = {}
# pieces = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']
# for piece in pieces:
#     images[piece] = pygame.image.load(f"images/{piece}.png")
#     images[piece] = pygame.transform.scale(images[piece], (SQUARE_SIZE, SQUARE_SIZE))

# def draw_board(screen, board, selected_square=None, valid_moves=None, pointer_pos=None):
#     for row in range(8):
#         for col in range(8):
#             color = WHITE if (row + col) % 2 == 0 else BLACK
#             pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
#             if selected_square and selected_square == chess.square(col, 7 - row):
#                 pygame.draw.rect(screen, HIGHLIGHT_COLOR, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
#             if valid_moves and chess.square(col, 7 - row) in valid_moves:
#                 pygame.draw.circle(screen, HIGHLIGHT_COLOR, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)

#     for row in range(8):
#         for col in range(8):
#             square = chess.square(col, 7 - row)
#             piece = board.piece_at(square)
#             if piece and square != selected_square:
#                 piece_code = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
#                 screen.blit(images[piece_code], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

#     if selected_square and pointer_pos:
#         piece = board.piece_at(selected_square)
#         if piece:
#             piece_code = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
#             screen.blit(images[piece_code], pygame.Rect(pointer_pos[0] - SQUARE_SIZE // 2, pointer_pos[1] - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE))
    
#     if pointer_pos:
#         pygame.draw.circle(screen, POINTER_COLOR, pointer_pos, 10)

# def get_square_from_pos(pos):
#     col = pos[0] // SQUARE_SIZE
#     row = 7 - (pos[1] // SQUARE_SIZE)
#     return chess.square(col, row)

# def get_chess_coord_from_hand_pos(hand_pos_x, hand_pos_y, width, height, sensitivity):
#     center_x_cam = 0.5
#     center_y_cam = 0.5
    
#     offset_x = (hand_pos_x - center_x_cam) * sensitivity
#     offset_y = (hand_pos_y - center_y_cam) * sensitivity
    
#     center_board_x = width // 4
#     center_board_y = height // 2
    
#     scaled_x = int(center_board_x + offset_x * (width // 2))
#     scaled_y = int(center_board_y + offset_y * height)
    
#     scaled_x = max(0, min(scaled_x, width // 2))
#     scaled_y = max(0, min(scaled_y, height))
    
#     return scaled_x, scaled_y

# Mengubah parameter MediaPipe untuk akurasi yang lebih baik
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# board = chess.Board()
selected_square = None
valid_moves = []
is_pinching = False
pointer_pos = None

running = True

# PENTING: Tambahkan bagian ini untuk inisialisasi virtual camera
# MirAi set ukurannya 640x480, kamu bisa ubah sesuai kebutuhan
with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f'Virtual camera siap! Output ke: {cam.device}')
    
    while running:
        success, img_cam = cap.read()
        if not success:
            continue
        
        # Ubah ukuran gambar agar sesuai dengan virtual camera
        img_cam = cv2.resize(img_cam, (1280, 720))
        img_cam = cv2.flip(img_cam, 1)
        
        # img_for_detection sekarang ukurannya 640x480
        img_for_detection = img_cam.copy()
        
        imgRGB = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
        
        # Memproses deteksi tangan
        hand_results = hands.process(imgRGB)
        
        # Memproses deteksi wajah dengan Face Mesh
        face_results = face_mesh.process(imgRGB)

        # draw_board(screen, board, selected_square, valid_moves, pointer_pos)
        
        # Mengaplikasikan gambar ke wajah
        if face_results.multi_face_landmarks and overlay_img is not None:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = img_for_detection.shape
                
                face_oval_indices = set()
                for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
                    face_oval_indices.add(connection[0])
                    face_oval_indices.add(connection[1])
                
                face_oval_points = [
                    (int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h))
                    for p in face_oval_indices
                ]
                
                mask = np.zeros_like(img_for_detection, dtype=np.uint8)
                if face_oval_points:
                    hull = cv2.convexHull(np.array(face_oval_points, dtype=np.int32))
                    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
                
                (x, y, w_mask, h_mask) = cv2.boundingRect(mask[:, :, 0])
                
                print(f"Wajah terdeteksi! Bounding Box: x={x}, y={y}, w={w_mask}, h={h_mask}")
                
                if w_mask > 0 and h_mask > 0:
                    resized_overlay = cv2.resize(overlay_img, (w_mask, h_mask))
                    face_region = img_for_detection[y:y+h_mask, x:x+w_mask]
                    mask_region = mask[y:y+h_mask, x:x+w_mask]
                    
                    masked_overlay = cv2.bitwise_and(resized_overlay, mask_region)
                    face_region_masked = cv2.bitwise_and(face_region, cv2.bitwise_not(mask_region))
                    img_for_detection[y:y+h_mask, x:x+w_mask] = cv2.add(face_region_masked, masked_overlay)
        
        # Logika deteksi tangan dan gestur
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_for_detection, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                middle_tip_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip_pos = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # pointer_pos = get_chess_coord_from_hand_pos(index_tip_pos.x, index_tip_pos.y, WIDTH, HEIGHT, SENSITIVITY_MULTIPLIER)
                
                print(f"Tangan terdeteksi! Pointer di koordinat: {pointer_pos}")
                
                distance = ((middle_tip_pos.x - thumb_tip_pos.x)**2 + (middle_tip_pos.y - thumb_tip_pos.y)**2)**0.5
                
                if distance < PINCH_THRESHOLD and not is_pinching:
                    is_pinching = True
                    # square_under_pointer = get_square_from_pos(pointer_pos)
                    # print(f"Gestur 'pinch' dimulai! Memilih kotak: {chess.square_name(square_under_pointer)}")
                    # if board.piece_at(square_under_pointer) and board.piece_at(square_under_pointer).color == board.turn:
                        # selected_square = square_under_pointer
                        # valid_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
                #     else:
                #         selected_square = None
                #         valid_moves = []
                        
                # elif distance >= PINCH_THRESHOLD and is_pinching:
                #     is_pinching = False
                #     # target_square = get_square_from_pos(pointer_pos)
                #     print(f"Gestur 'pinch' selesai! Melepaskan di kotak: {chess.square_name(target_square)}")
                #     if selected_square:
                #         move = chess.Move(selected_square, target_square)
                #         if move in board.legal_moves:
                #             board.push(move)
                        
                #     selected_square = None
                #     valid_moves = []
        else:
            if hand_results.multi_hand_landmarks is None:
                print("Tangan tidak terdeteksi.")
            if face_results.multi_face_landmarks is None:
                print("Wajah tidak terdeteksi.")
            pointer_pos = None
            selected_square = None
            valid_moves = []

        # Kirim frame yang sudah dimodifikasi ke virtual camera
        # Ubah BGR ke RGB karena PyVirtualCam butuh format RGB
        frame_to_send = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
        cam.send(frame_to_send)

        # Bagian ini untuk menampilkan di jendela Pygame, bisa kamu hapus kalau tidak butuh
        img_cam_pygame = pygame.image.frombuffer(img_for_detection.tobytes(), img_for_detection.shape[1::-1], "BGR")
        screen.blit(img_cam_pygame, (0, 0))
        
        font = pygame.font.Font(None, 36)
        # if board.is_checkmate():
        #     text = font.render("Checkmate!", True, (255, 0, 0))
        #     screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
        # elif board.is_check():
        #     text = font.render("Check!", True, (255, 0, 0))
        #     screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 20))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
cap.release()
pygame.quit()