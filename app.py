from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# URL gambar default untuk masking wajah
DEFAULT_IMAGE_URL = "https://i.pinimg.com/1200x/43/8b/c6/438bc647f7f36f1115ad28cd5ee8c059.jpg"

# Variabel global untuk menyimpan gambar filter
overlay_img = None
# Inisialisasi gambar awal saat aplikasi pertama kali dijalankan
try:
    response = requests.get(DEFAULT_IMAGE_URL)
    img_data = response.content
    img_array = np.frombuffer(img_data, np.uint8)
    overlay_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print("Gambar default berhasil diunduh!")
except Exception as e:
    print(f"Gagal mengunduh gambar default dari URL: {e}")
    overlay_img = None

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka.")
        return

    while True:
        success, img_cam = cap.read()
        if not success:
            break
        else:
            img_cam = cv2.resize(img_cam, (1280, 720))
            img_cam = cv2.flip(img_cam, 1)
            
            img_for_detection = img_cam.copy()
            imgRGB = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
            
            face_results = face_mesh.process(imgRGB)
            
            # Akses variabel global
            global overlay_img 

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
                    
                    if w_mask > 0 and h_mask > 0:
                        resized_overlay = cv2.resize(overlay_img, (w_mask, h_mask))
                        face_region = img_for_detection[y:y+h_mask, x:x+w_mask]
                        mask_region = mask[y:y+h_mask, x:x+w_mask]
                        
                        masked_overlay = cv2.bitwise_and(resized_overlay, mask_region)
                        face_region_masked = cv2.bitwise_and(face_region, cv2.bitwise_not(mask_region))
                        img_for_detection[y:y+h_mask, x:x+w_mask] = cv2.add(face_region_masked, masked_overlay)
            
            ret, buffer = cv2.imencode('.jpg', img_for_detection)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_filter', methods=['POST'])
def upload_filter():
    global overlay_img
    
    # Cek apakah ada file yang diunggah
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        new_overlay_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if new_overlay_img is not None:
            overlay_img = new_overlay_img
            print("Filter berhasil diganti dengan foto yang diunggah.")
            return redirect(url_for('index'))
    
    # Cek apakah ada URL yang dikirim
    url = request.form.get('url')
    if url:
        try:
            response = requests.get(url)
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            new_overlay_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if new_overlay_img is not None:
                overlay_img = new_overlay_img
                print("Filter berhasil diganti dengan foto dari URL.")
                return redirect(url_for('index'))
            else:
                print("URL tidak valid atau bukan gambar.")
        except Exception as e:
            print(f"Gagal mengunduh gambar dari URL: {e}")
            
    # Jika tidak ada file atau URL yang valid, kembalikan ke halaman utama
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)