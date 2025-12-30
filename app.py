import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import os
from pillow_heif import register_heif_opener
from collections import deque, Counter

# Register HEIC/HEIF support
register_heif_opener()

# Page configuration
st.set_page_config(
    page_title="💎 Skin Tone AI Professional",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Label skin tone (sesuai dengan model training)
LABELS = ['dark', 'light', 'mid-dark', 'mid-light']

# Settings untuk akurasi (ENHANCED)
HISTORY_LEN = 20  # Increased for more stable voting
CONFIDENCE_THRES = 0.50  # Lowered threshold for easier detection
MIN_FACE_SIZE = 90  # Minimum face size (pixel) - ensures user is close enough

# Brightness thresholds for gating
BRIGHTNESS_MIN = 70
BRIGHTNESS_MAX = 210

# Skin ratio threshold for obstruction detection
SKIN_RATIO_MIN = 0.4

# Colors for visualization (BGR format for OpenCV)
LABEL_COLORS = {
    'dark': (50, 75, 120),
    'light': (189, 224, 255),
    'mid-dark': (70, 110, 165),
    'mid-light': (148, 205, 255)
}

# State untuk voting history (webcam mode)
if 'skin_history' not in st.session_state:
    st.session_state.skin_history = deque(maxlen=HISTORY_LEN)
if 'final_decision' not in st.session_state:
    st.session_state.final_decision = "Mencari..."


# --- FITUR A: GRAY WORLD (AUTO WHITE BALANCE) ---
def correct_white_balance(img):
    """
    Koreksi white balance menggunakan Gray World Algorithm.
    Pengganti Calibration Card fisik untuk hasil lebih akurat di berbagai pencahayaan.
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    # Geser channel A dan B supaya rata-ratanya jadi 128 (netral)
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 1] / 255.0) * 1.2)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 2] / 255.0) * 1.2)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


# --- FITUR B: SKIN LOCUS (FILTER MATEMATIKA) ---
def create_skin_mask(image):
    """
    Membuat mask untuk area kulit menggunakan YCbCr color space.
    Rentang warna kulit universal di YCbCr untuk filter yang akurat.
    """
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Rentang warna kulit universal di YCbCr
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    
    mask = cv2.inRange(ycbcr, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# --- FITUR C: BRIGHTNESS GATING ---
def check_brightness(face_img):
    """
    Cek brightness wajah untuk menentukan apakah pencahayaan memadai.
    Returns: (is_valid, brightness_value, warning_message)
    """
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    
    if brightness < BRIGHTNESS_MIN:
        return False, brightness, "⚠️ GELAP: Cari Cahaya Lebih Terang"
    elif brightness > BRIGHTNESS_MAX:
        return False, brightness, "⚠️ SILAU: Kurangi Cahaya"
    else:
        return True, brightness, ""


# --- FITUR D: OBSTRUCTION DETECTION ---
def check_skin_visibility(face_img):
    """
    Deteksi apakah wajah terhalang oleh tangan, masker, atau objek lain.
    Menggunakan skin locus masking untuk menghitung rasio kulit terlihat.
    """
    mask = create_skin_mask(face_img)
    skin_ratio = cv2.countNonZero(mask) / (mask.size + 1e-5)
    
    if skin_ratio < SKIN_RATIO_MIN:
        return False, skin_ratio, "⚠️ Wajah Terhalang (Tangan/Masker)"
    else:
        return True, skin_ratio, ""


# --- FITUR E: SMART CROP (30% tengah wajah) ---
def smart_crop_face(x1, y1, x2, y2):
    """
    Mengambil 30% area tengah wajah (Pipi & Hidung) untuk sampling kulit.
    Membuang area jilbab, rambut, telinga, dan background.
    Returns: crop coordinates (crop_x1, crop_y1, crop_x2, crop_y2)
    """
    w_box = x2 - x1
    h_box = y2 - y1
    
    # Hanya ambil 30% area tengah wajah
    crop_x1 = int(x1 + w_box * 0.35)
    crop_x2 = int(x2 - w_box * 0.35)
    crop_y1 = int(y1 + h_box * 0.25)
    crop_y2 = int(y2 - h_box * 0.25)
    
    return crop_x1, crop_y1, crop_x2, crop_y2


def center_crop_face(face_img):
    """
    Ambil bagian tengah wajah (area pipi/hidung) untuk sampling kulit.
    Versi legacy untuk kompatibilitas dengan fungsi lama.
    """
    h, w = face_img.shape[:2]
    
    # Potong 35% dari setiap sisi horizontal, 25% atas/bawah (lebih agresif)
    crop_x1 = int(w * 0.35)
    crop_x2 = int(w * 0.65)
    crop_y1 = int(h * 0.25)
    crop_y2 = int(h * 0.75)
    
    center_crop = face_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Jika crop terlalu kecil, gunakan gambar asli
    if center_crop.size == 0:
        return face_img
    
    return center_crop


# Rekomendasi warna pakaian berdasarkan skin tone (diperluas dari Gradio)
def get_recommendation(skin_label):
    recommendations = {
        "light": "🎨 **COCOK:** Pastel, Jewel Tones (Emerald, Royal Blue), Hitam, Navy.\n\n🚫 **HINDARI:** Beige pucat, Kuning pudar, Neon.",
        "mid-light": "🎨 **COCOK:** Earth Tones (Khaki, Olive), Mustard, Coral, Peach.\n\n🚫 **HINDARI:** Abu-abu metalik, Neon dingin, Abu-abu flat.",
        "mid-dark": "🎨 **COCOK:** Warna Kaya (Maroon, Gold, Ungu Deep), Navy, Orange Bata.\n\n🚫 **HINDARI:** Coklat kusam (mirip kulit), Coklat mati.",
        "dark": "🎨 **COCOK:** Kontras Tinggi (Putih, Kuning Lemon, Fuchsia), Cobalt, Merah.\n\n🚫 **HINDARI:** Coklat tua gelap, Navy gelap."
    }
    return recommendations.get(skin_label, "Tidak ada rekomendasi.")


# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .skin-tone-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        margin: 5px;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load YOLOv8 Face dan Skin Tone classification models."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load YOLOv8 Face model (khusus deteksi wajah)
    yolo_path = os.path.join(base_path, "YOLOv8n Face.pt")
    if not os.path.exists(yolo_path):
        raise Exception(f"File {yolo_path} tidak ditemukan!")
    yolo_model = YOLO(yolo_path)
    
    # Load Skin Tone model
    skin_tone_path = os.path.join(base_path, "Skin Tone Model v2.h5")
    if not os.path.exists(skin_tone_path):
        raise Exception(f"File {skin_tone_path} tidak ditemukan!")
    skin_tone_model = tf.keras.models.load_model(skin_tone_path)
    
    return yolo_model, skin_tone_model


def process_frame(frame, yolo_model, skin_tone_model, use_voting=True, use_white_balance=True, 
                   use_center_crop=True, use_brightness_gate=True, use_obstruction_detect=True):
    """
    Process a single frame dengan fitur-fitur PROFESIONAL ENHANCED:
    - White balance correction (Gray World Algorithm)
    - Smart crop sampling (30% area tengah wajah)
    - Brightness gating (validasi pencahayaan)
    - Obstruction detection (deteksi halangan wajah)
    - Voting history untuk stabilisasi
    """
    if frame is None:
        return frame, "Tidak ada frame", None, 0.0, ""
    
    annotated_frame = frame.copy()
    info_text = "🔍 Mencari wajah..."
    detected_skin = None
    confidence = 0.0
    warning_msg = ""
    
    # Deteksi wajah menggunakan model Face
    results = yolo_model(frame, verbose=False)
    face_found = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Validasi ukuran minimum wajah (user harus mendekat)
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                # Tampilkan pesan untuk mendekat
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Mendekat!", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                warning_msg = "📏 Wajah terlalu jauh. Dekatkan ke kamera."
                continue
            
            # Validasi koordinat
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # --- SMART CROP: Mengambil 30% area tengah wajah ---
            crop_x1, crop_y1, crop_x2, crop_y2 = smart_crop_face(x1, y1, x2, y2)
            
            # Validasi koordinat crop
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(w, crop_x2)
            crop_y2 = min(h, crop_y2)
            
            face_center = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_center.size == 0:
                continue
            
            face_found = True
            
            # --- BRIGHTNESS GATING: Cek pencahayaan ---
            if use_brightness_gate:
                is_bright_ok, brightness_val, bright_warn = check_brightness(face_center)
                if not is_bright_ok:
                    # Kotak merah untuk kondisi pencahayaan buruk
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 1)
                    cv2.putText(annotated_frame, bright_warn.replace("⚠️ ", ""), (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    warning_msg = bright_warn
                    info_text = f"💡 Brightness: {brightness_val:.0f} (Ideal: {BRIGHTNESS_MIN}-{BRIGHTNESS_MAX})"
                    continue
            
            # Apply white balance correction
            if use_white_balance:
                face_for_prediction = correct_white_balance(face_center)
            else:
                face_for_prediction = face_center
            
            # --- OBSTRUCTION DETECTION: Cek apakah wajah terhalang ---
            if use_obstruction_detect:
                is_visible, skin_ratio, obstruct_warn = check_skin_visibility(face_for_prediction)
                if not is_visible:
                    # Kotak kuning untuk kondisi terhalang
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.rectangle(annotated_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 165, 255), 1)
                    cv2.putText(annotated_frame, "Wajah Terhalang", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    warning_msg = obstruct_warn
                    info_text = f"👋 Skin Ratio: {skin_ratio:.1%} (Min: {SKIN_RATIO_MIN:.0%})"
                    continue
            
            # --- PREDIKSI SKIN TONE ---
            try:
                resized_face = cv2.resize(face_for_prediction, (224, 224))
                img_array = np.expand_dims(resized_face, axis=0) / 255.0
                
                prediction = skin_tone_model.predict(img_array, verbose=0)
                idx = np.argmax(prediction)
                current_label = LABELS[idx]
                confidence = float(prediction[0][idx])
                
                # Voting mechanism untuk stabilisasi (webcam mode)
                if use_voting and confidence > CONFIDENCE_THRES:
                    st.session_state.skin_history.append(current_label)
                    if len(st.session_state.skin_history) > 0:
                        most_common = Counter(st.session_state.skin_history).most_common(1)[0][0]
                        detected_skin = most_common
                        st.session_state.final_decision = most_common
                else:
                    if confidence <= CONFIDENCE_THRES:
                        warning_msg = f"⏳ Confidence rendah ({confidence:.1%}). Analisis..."
                    detected_skin = current_label
                
                # --- VISUALISASI PROFESIONAL ---
                # Kotak hijau = kondisi bagus, siap deteksi
                color = (0, 255, 0)  # Hijau untuk kondisi baik
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Kotak biru = area sampling (30% tengah wajah)
                cv2.rectangle(annotated_frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 1)
                
                # Label hasil
                if warning_msg:
                    cv2.putText(annotated_frame, warning_msg.replace("⚠️ ", "").replace("⏳ ", ""), 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    label_text = f"{(detected_skin or current_label).upper()}"
                    cv2.putText(annotated_frame, label_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Tampilkan confidence di bawah kotak
                    cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                info_text = f"✅ Deteksi: {(detected_skin or current_label).upper()} (Confidence: {confidence:.1%})"
                
            except Exception as e:
                warning_msg = f"❌ Error prediksi: {str(e)}"
                info_text = warning_msg
    
    if not face_found and not warning_msg:
        info_text = "⚠️ Wajah tidak terdeteksi. Pastikan wajah terlihat jelas dan dekat ke kamera."
    
    return annotated_frame, info_text, detected_skin, confidence, warning_msg


def detect_and_classify_image(image, yolo_model, skin_tone_model, conf_threshold=0.5, 
                               use_white_balance=True, use_center_crop=True,
                               use_brightness_gate=True, use_obstruction_detect=True):
    """
    Detect faces and classify skin tones from uploaded image dengan fitur PROFESIONAL:
    - Smart crop sampling (30% area tengah wajah)
    - White balance correction  
    - Brightness gating
    - Obstruction detection
    - Enhanced visualization with sampling area
    """
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Deteksi wajah
    results = yolo_model(img_bgr, conf=conf_threshold)
    
    detections = []
    annotated_img = img_bgr.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detection_conf = float(box.conf[0])
            
            # Validasi ukuran minimum
            w_box = x2 - x1
            h_box = y2 - y1
            
            # Untuk gambar, gunakan threshold lebih rendah karena gambar bisa resolusi tinggi
            min_size_img = max(30, MIN_FACE_SIZE // 3)
            if w_box < min_size_img or h_box < min_size_img:
                continue
            
            # Validasi koordinat
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # --- SMART CROP: Mengambil 30% area tengah wajah ---
            crop_x1, crop_y1, crop_x2, crop_y2 = smart_crop_face(x1, y1, x2, y2)
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(w, crop_x2), min(h, crop_y2)
            
            face_center = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if face_center.size == 0:
                continue
            
            status_msg = ""
            brightness_val = 0
            skin_ratio_val = 0
            
            # --- BRIGHTNESS CHECK ---
            if use_brightness_gate:
                is_bright_ok, brightness_val, bright_warn = check_brightness(face_center)
                if not is_bright_ok:
                    status_msg = bright_warn
            
            # Apply white balance correction
            if use_white_balance:
                face_for_prediction = correct_white_balance(face_center)
            else:
                face_for_prediction = face_center
            
            # --- OBSTRUCTION CHECK ---
            if use_obstruction_detect and not status_msg:
                is_visible, skin_ratio_val, obstruct_warn = check_skin_visibility(face_for_prediction)
                if not is_visible:
                    status_msg = obstruct_warn
            else:
                # Calculate skin ratio anyway for diagnostics
                mask = create_skin_mask(face_for_prediction)
                skin_ratio_val = cv2.countNonZero(mask) / (mask.size + 1e-5)
            
            # --- PREDIKSI SKIN TONE ---
            resized_face = cv2.resize(face_for_prediction, (224, 224))
            img_input = np.expand_dims(resized_face, axis=0) / 255.0
            
            prediction = skin_tone_model.predict(img_input, verbose=0)
            skin_tone_idx = np.argmax(prediction[0])
            skin_tone_conf = float(prediction[0][skin_tone_idx])
            skin_tone_label = LABELS[skin_tone_idx]
            
            # --- VISUALISASI PROFESIONAL ---
            # Warna kotak berdasarkan status
            if status_msg:
                if "GELAP" in status_msg or "SILAU" in status_msg:
                    box_color = (0, 0, 255)  # Merah untuk lighting issue
                else:
                    box_color = (0, 165, 255)  # Orange untuk obstruction
            else:
                box_color = (0, 255, 0)  # Hijau untuk kondisi baik
            
            # Kotak wajah (luar)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
            
            # Kotak sampling biru (area yang dibaca AI)
            cv2.rectangle(annotated_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 2)
            
            # Label dengan background
            label = f"{skin_tone_label.upper()} ({skin_tone_conf:.1%})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
            cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status warning jika ada
            if status_msg:
                cv2.putText(annotated_img, status_msg.replace("⚠️ ", ""), (x1, y2 + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'sampling_area': (crop_x1, crop_y1, crop_x2, crop_y2),
                'detection_conf': detection_conf,
                'skin_tone': skin_tone_label,
                'skin_tone_conf': skin_tone_conf,
                'brightness': brightness_val,
                'skin_ratio': skin_ratio_val,
                'status': status_msg if status_msg else "OK"
            })
    
    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections


def main():
    # Header
    st.markdown('<h1 class="main-header">💎 Skin Tone AI Professional</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Menggunakan YOLO-Face + Smart Sampling + Auto White Balance</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selector
        mode = st.radio(
            "Pilih Mode Input:",
            ["📷 Webcam", "📁 Upload Gambar"],
            index=0
        )
        
        st.divider()
        
        conf_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.divider()
        
        # Advanced settings - Extended
        st.header("🔧 Advanced Features")
        
        use_white_balance = st.checkbox(
            "✨ Auto White Balance", 
            value=True, 
            help="Koreksi pencahayaan menggunakan Gray World Algorithm untuk hasil lebih akurat"
        )
        
        use_center_crop = st.checkbox(
            "🎯 Smart Crop Sampling", 
            value=True,
            help="Hanya mengambil 30% area tengah wajah (pipi & hidung) untuk menghindari rambut/jilbab/background"
        )
        
        use_brightness_gate = st.checkbox(
            "💡 Brightness Gating",
            value=True,
            help=f"Validasi pencahayaan (ideal: {BRIGHTNESS_MIN}-{BRIGHTNESS_MAX}). Kotak akan merah jika terlalu gelap/terang"
        )
        
        use_obstruction_detect = st.checkbox(
            "👋 Obstruction Detection",
            value=True,
            help=f"Deteksi jika wajah terhalang tangan/masker (min skin ratio: {SKIN_RATIO_MIN:.0%})"
        )
        
        if mode == "📷 Webcam":
            st.divider()
            if st.button("🔄 Reset Voting History"):
                st.session_state.skin_history.clear()
                st.session_state.final_decision = "Mencari..."
                st.success("History direset!")
        
        st.divider()
        
        st.header("📊 About")
        st.markdown("""
        **AI Professional Skin Analysis** menggunakan:
        - **YOLOv8 Face** untuk deteksi wajah real-time
        - **Custom CNN** untuk klasifikasi skin tone
        
        **🚀 Fitur Profesional:**
        1. ✅ Gray World White Balance
        2. ✅ Smart Crop (30% area sampel)
        3. ✅ Brightness Gating
        4. ✅ Skin Locus Masking (YCbCr)
        5. ✅ Obstruction Detection
        6. ✅ Voting History (Webcam)
        
        **🎨 Kategori Skin Tone:**
        - 🔵 Light
        - 🟢 Mid-Light  
        - 🟡 Mid-Dark
        - 🟤 Dark
        
        **📍 Panduan Visualisasi:**
        - 🟩 Kotak Hijau = Kondisi Baik
        - 🟥 Kotak Merah = Pencahayaan Buruk
        - 🟧 Kotak Orange = Wajah Terhalang
        - 🟦 Kotak Biru = Area Sampling AI
        """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        try:
            yolo_model, skin_tone_model = load_models()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.stop()
    
    # Mode: Webcam
    if mode == "📷 Webcam":
        st.markdown("### 📹 Live Webcam Detection")
        st.info("📍 Arahkan wajah ke webcam. Kotak akan berubah **HIJAU** jika kondisi siap untuk analisis. Kotak **BIRU** menunjukkan area sampling AI.")
        
        # Webcam input
        camera_input = st.camera_input("Capture from Webcam", label_visibility="collapsed")
        
        if camera_input is not None:
            # Convert to numpy array
            image = Image.open(camera_input)
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame with all professional features
            annotated_frame, info_text, detected_skin, confidence, warning_msg = process_frame(
                frame_bgr, yolo_model, skin_tone_model,
                use_voting=True,
                use_white_balance=use_white_balance,
                use_center_crop=use_center_crop,
                use_brightness_gate=use_brightness_gate,
                use_obstruction_detect=use_obstruction_detect
            )
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Detection Result")
                st.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Hasil Analisis")
                
                # Status dengan warna berdasarkan kondisi
                if warning_msg:
                    st.warning(warning_msg)
                else:
                    st.success(info_text)
                
                # Show voting info
                if len(st.session_state.skin_history) > 0:
                    vote_counts = Counter(st.session_state.skin_history)
                    st.markdown(f"**Voting History:** {len(st.session_state.skin_history)}/{HISTORY_LEN} samples")
                    st.caption(f"Votes: {dict(vote_counts)}")
                
                if detected_skin and not warning_msg:
                    st.markdown("---")
                    st.markdown(f"### 🎨 Skin Tone: **{detected_skin.upper()}**")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.markdown("### 👔 Rekomendasi Warna Pakaian")
                    recommendation = get_recommendation(detected_skin)
                    st.markdown(f"""
                    <div class="recommendation-box">
                        {recommendation.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Mode: Upload Image
    else:
        st.markdown("### 📤 Upload Image")
        st.info("📍 Upload gambar wajah. Kotak **BIRU** menunjukkan area sampling AI (30% area tengah wajah).")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=['jpg', 'jpeg', 'png', 'heic', 'heif'],
            help="Format: JPG, JPEG, PNG, HEIC, HEIF"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 Original Image")
                st.image(image, use_container_width=True)
            
            with st.spinner("🔍 Analyzing dengan fitur profesional..."):
                annotated_image, detections = detect_and_classify_image(
                    image, yolo_model, skin_tone_model, conf_threshold,
                    use_white_balance=use_white_balance,
                    use_center_crop=use_center_crop,
                    use_brightness_gate=use_brightness_gate,
                    use_obstruction_detect=use_obstruction_detect
                )
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                st.image(annotated_image, use_container_width=True)
            
            st.markdown("---")
            
            if detections:
                st.markdown(f"### 📋 Terdeteksi {len(detections)} Wajah")
                
                for idx, det in enumerate(detections):
                    # Status badge color
                    status = det.get('status', 'OK')
                    status_icon = "✅" if status == "OK" else "⚠️"
                    
                    with st.expander(f"👤 Face #{idx + 1} - {det['skin_tone'].upper()} {status_icon}", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown("**📊 Hasil Deteksi**")
                            st.markdown(f"**Skin Tone:** {det['skin_tone'].upper()}")
                            st.markdown(f"**Confidence:** {det['skin_tone_conf']:.1%}")
                            st.markdown(f"**Detection Score:** {det['detection_conf']:.1%}")
                        
                        with col_b:
                            st.markdown("**🔬 Diagnostik**")
                            brightness = det.get('brightness', 0)
                            skin_ratio = det.get('skin_ratio', 0)
                            st.markdown(f"**Brightness:** {brightness:.0f}")
                            st.markdown(f"**Skin Ratio:** {skin_ratio:.1%}")
                            st.markdown(f"**Status:** {status}")
                        
                        with col_c:
                            st.markdown("**👔 Rekomendasi:**")
                            st.markdown(get_recommendation(det['skin_tone']))
            else:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Coba sesuaikan threshold atau gunakan gambar lain.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        💎 <b>Skin Tone AI Professional</b> — Made with ❤️ using Streamlit, YOLOv8 Face, and TensorFlow<br>
        <small>Features: Gray World AWB | Smart Crop | Brightness Gating | Skin Locus | Obstruction Detection</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
