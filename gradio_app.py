import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

# --- 1. LOAD MODELS ---
print("Sedang memuat model...")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define model paths with correct filenames
skin_model_path = os.path.join(script_dir, 'Skin Tone Model v2.h5')
yolo_model_path = os.path.join(script_dir, 'YOLOv8n.pt')

# Check if model files exist
if not os.path.exists(skin_model_path):
    raise FileNotFoundError(f"File '{skin_model_path}' tidak ditemukan! Pastikan file ada di folder yang sama dengan script ini.")

if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"File '{yolo_model_path}' tidak ditemukan! Pastikan file ada di folder yang sama dengan script ini.")

# Load Model Kulit
skin_model = tf.keras.models.load_model(skin_model_path)

# Load YOLO
yolo_model = YOLO(yolo_model_path) 

# Label (Sesuaikan urutan ini dengan hasil training di Colab kemarin)
LABELS = ['dark', 'light', 'mid-dark', 'mid-light']

# --- 2. LOGIKA REKOMENDASI ---
def get_recommendation(skin_label):
    recommendations = {
        "light": "Rekomendasi: Pastel (Pink, Baby Blue), Navy, Emerald Green. \nHindari: Neon, Kuning Pucat.",
        "mid-light": "Rekomendasi: Earth Tones, Mustard, Olive Green, Peach. \nHindari: Abu-abu flat.",
        "mid-dark": "Rekomendasi: Maroon, Gold, Ungu Tua, Orange Bata. \nHindari: Coklat mati.",
        "dark": "Rekomendasi: Putih Bersih, Kuning Lemon, Cobalt Blue, Merah. \nHindari: Coklat tua gelap."
    }
    return recommendations.get(skin_label, "Tidak ada rekomendasi.")

# --- 3. PROSES FRAME ---
def process_frame(frame):
    if frame is None: return frame, ""

    annotated_frame = frame.copy()
    info_text = "Mencari wajah..."
    
    # Deteksi wajah
    results = yolo_model(frame, classes=0, verbose=False)
    face_found = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Validasi koordinat agar tidak error saat crop
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            
            face_found = True

            # Prediksi Kulit
            resized_face = cv2.resize(face_crop, (224, 224))
            img_array = np.expand_dims(resized_face, axis=0) / 255.0
            
            prediction = skin_model.predict(img_array, verbose=0)
            idx = np.argmax(prediction)
            detected_skin = LABELS[idx]
            
            # Tampilkan Hasil
            rec = get_recommendation(detected_skin)
            
            # Gambar kotak & teks
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, detected_skin.upper(), (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            info_text = f"Deteksi: {detected_skin}\n{rec}"

    if not face_found:
        info_text = "Wajah tidak terdeteksi."

    # Gradio butuh format RGB, OpenCV pakai BGR. Jadi kita convert balik.
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), info_text

# --- 4. JALANKAN APP ---
# Di Mac Local, kita tidak perlu share=True kalau cuma buat sendiri
print("Aplikasi berjalan! Buka link lokal di browser.")

demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=[gr.Image(label="Output"), gr.Textbox(label="Rekomendasi")],
    live=True
)

demo.launch()
