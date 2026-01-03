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
HISTORY_LEN = 30  # Increased for more stable voting (was 20)
CONFIDENCE_THRES = 0.35  # Lowered threshold - model confidence biasanya tidak terlalu tinggi
MIN_FACE_SIZE = 80  # Minimum face size (pixel) - sedikit lebih kecil untuk jangkauan lebih jauh

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


# --- FITUR F: CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
def apply_clahe(img):
    """
    Menerapkan CLAHE untuk meningkatkan kontras secara lokal.
    Sangat efektif untuk normalisasi pencahayaan yang tidak merata.
    """
    # Convert ke LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE hanya ke channel L (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Gabungkan kembali
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return result


# --- FITUR G: MULTI-REGION SAMPLING ---
def get_multi_region_crops(frame, x1, y1, x2, y2):
    """
    Mengambil sampel dari beberapa region wajah untuk prediksi ensemble.
    Region: Dahi, Pipi Kiri, Pipi Kanan, Hidung
    Returns: List of cropped regions
    """
    h, w = frame.shape[:2]
    w_box = x2 - x1
    h_box = y2 - y1
    
    regions = []
    region_names = []
    
    # Region 1: Dahi (atas tengah)
    forehead_x1 = int(x1 + w_box * 0.30)
    forehead_x2 = int(x1 + w_box * 0.70)
    forehead_y1 = int(y1 + h_box * 0.10)
    forehead_y2 = int(y1 + h_box * 0.30)
    
    # Region 2: Pipi Kiri
    left_cheek_x1 = int(x1 + w_box * 0.10)
    left_cheek_x2 = int(x1 + w_box * 0.35)
    left_cheek_y1 = int(y1 + h_box * 0.40)
    left_cheek_y2 = int(y1 + h_box * 0.70)
    
    # Region 3: Pipi Kanan
    right_cheek_x1 = int(x1 + w_box * 0.65)
    right_cheek_x2 = int(x1 + w_box * 0.90)
    right_cheek_y1 = int(y1 + h_box * 0.40)
    right_cheek_y2 = int(y1 + h_box * 0.70)
    
    # Region 4: Hidung (tengah)
    nose_x1 = int(x1 + w_box * 0.35)
    nose_x2 = int(x1 + w_box * 0.65)
    nose_y1 = int(y1 + h_box * 0.35)
    nose_y2 = int(y1 + h_box * 0.60)
    
    # Validasi dan crop
    region_coords = [
        (forehead_x1, forehead_y1, forehead_x2, forehead_y2, "Dahi"),
        (left_cheek_x1, left_cheek_y1, left_cheek_x2, left_cheek_y2, "Pipi Kiri"),
        (right_cheek_x1, right_cheek_y1, right_cheek_x2, right_cheek_y2, "Pipi Kanan"),
        (nose_x1, nose_y1, nose_x2, nose_y2, "Hidung")
    ]
    
    for rx1, ry1, rx2, ry2, name in region_coords:
        # Validasi koordinat
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
        
        if rx2 > rx1 and ry2 > ry1:
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size > 0:
                regions.append(crop)
                region_names.append(name)
    
    return regions, region_names


# --- FITUR H: ENSEMBLE PREDICTION ---
def ensemble_predict(regions, skin_model, use_clahe=True, use_white_balance=True):
    """
    Melakukan prediksi pada multiple regions dan menggabungkan hasilnya.
    Menggunakan AVERAGE RAW PREDICTIONS untuk confidence yang lebih stabil.
    Returns: (final_label, final_confidence, all_predictions)
    """
    if not regions:
        return None, 0.0, []
    
    predictions = []
    raw_predictions = []  # Untuk averaging
    
    for region in regions:
        # Preprocessing
        processed = region.copy()
        
        if use_white_balance:
            processed = correct_white_balance(processed)
        
        if use_clahe:
            processed = apply_clahe(processed)
        
        # Resize dan prediksi
        try:
            resized = cv2.resize(processed, (224, 224))
            img_array = np.expand_dims(resized, axis=0) / 255.0
            
            pred = skin_model.predict(img_array, verbose=0)
            idx = np.argmax(pred)
            conf = float(pred[0][idx])
            label = LABELS[idx]
            
            predictions.append({
                'label': label,
                'confidence': conf,
                'raw_pred': pred[0]
            })
            raw_predictions.append(pred[0])
        except Exception as e:
            continue
    
    if not predictions:
        return None, 0.0, []
    
    # METODE BARU: Average semua raw predictions
    # Ini menghasilkan confidence yang lebih stabil dan tinggi
    avg_pred = np.mean(raw_predictions, axis=0)
    final_idx = np.argmax(avg_pred)
    final_label = LABELS[final_idx]
    final_confidence = float(avg_pred[final_idx])
    
    # Boost confidence sedikit karena ensemble lebih reliable
    # Maximum boost 10%
    confidence_boost = min(0.10, (len(predictions) - 1) * 0.02)
    final_confidence = min(1.0, final_confidence + confidence_boost)
    
    return final_label, final_confidence, predictions


# Rekomendasi warna pakaian berdasarkan skin tone (diperluas dari Gradio)
def get_recommendation(skin_label):
    recommendations = {
        "light": "🎨 **COCOK:** Pastel, Jewel Tones (Emerald, Royal Blue), Hitam, Navy.\n\n🚫 **HINDARI:** Beige pucat, Kuning pudar, Neon.",
        "mid-light": "🎨 **COCOK:** Earth Tones (Khaki, Olive), Mustard, Coral, Peach.\n\n🚫 **HINDARI:** Abu-abu metalik, Neon dingin, Abu-abu flat.",
        "mid-dark": "🎨 **COCOK:** Warna Kaya (Maroon, Gold, Ungu Deep), Navy, Orange Bata.\n\n🚫 **HINDARI:** Coklat kusam (mirip kulit), Coklat mati.",
        "dark": "🎨 **COCOK:** Kontras Tinggi (Putih, Kuning Lemon, Fuchsia), Cobalt, Merah.\n\n🚫 **HINDARI:** Coklat tua gelap, Navy gelap."
    }
    return recommendations.get(skin_label, "Tidak ada rekomendasi.")


# Custom CSS for PREMIUM DARK MODE styling
st.markdown("""
<style>
    /* ===== IMPORT GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* ===== MAIN HEADER ===== */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #f472b6 70%, #fb7185 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
        filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.4));
    }
    
    .sub-header {
        font-size: 1.15rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        animation: fadeIn 1s ease-out;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* ===== GLASSMORPHISM CARDS (DARK) ===== */
    .glass-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.6s ease-out;
        color: #e2e8f0;
    }
    
    .glass-card h4, .glass-card h5, .glass-card strong {
        color: #f1f5f9 !important;
    }
    
    .glass-card-dark {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        padding: 24px;
        margin: 15px 0;
        color: #e2e8f0;
    }
    
    /* ===== RESULT CARD ===== */
    .result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 20px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* ===== SKIN TONE BADGES ===== */
    .skin-tone-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin: 8px 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .skin-tone-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    
    .badge-light { background: linear-gradient(135deg, #fde68a 0%, #fbbf24 100%); color: #78350f; }
    .badge-mid-light { background: linear-gradient(135deg, #fdba74 0%, #f97316 100%); }
    .badge-mid-dark { background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%); }
    .badge-dark { background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%); }
    
    /* ===== SKIN TONE COLOR INDICATOR ===== */
    .skin-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
        border: 2px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .skin-light { background: linear-gradient(135deg, #FFECD2 0%, #FCB69F 100%); }
    .skin-mid-light { background: linear-gradient(135deg, #E0C3A8 0%, #C9A882 100%); }
    .skin-mid-dark { background: linear-gradient(135deg, #A67C52 0%, #8B5A2B 100%); }
    .skin-dark { background: linear-gradient(135deg, #5D4037 0%, #3E2723 100%); }
    
    /* ===== RECOMMENDATION BOX ===== */
    .recommendation-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border-radius: 20px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .recommendation-box h4 {
        margin: 0 0 12px 0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 0 16px 16px 0;
        margin: 12px 0;
        font-weight: 500;
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%);
        border-left: 4px solid #22c55e;
        padding: 18px 22px;
        border-radius: 0 16px 16px 0;
        margin: 12px 0;
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        border-left: 4px solid #f59e0b;
        padding: 18px 22px;
        border-radius: 0 16px 16px 0;
        margin: 12px 0;
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c7d2fe;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #c7d2fe !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    /* ===== METRICS STYLING ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 8px;
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* ===== CAMERA INPUT ===== */
    .stCameraInput > div {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* ===== PROGRESS/LOADING ===== */
    .stProgress > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* ===== CUSTOM DIVIDER ===== */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #6366f1 50%, transparent 100%);
        border: none;
        margin: 30px 0;
        border-radius: 2px;
    }
    
    /* ===== FEATURE BADGE ===== */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 20px;
        margin-left: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ===== CONFIDENCE METER ===== */
    .confidence-meter {
        width: 100%;
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    .conf-high { background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%); }
    .conf-medium { background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }
    .conf-low { background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
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
                   use_center_crop=True, use_brightness_gate=True, use_obstruction_detect=True,
                   use_clahe=True, use_multi_region=True):
    """
    Process a single frame dengan fitur-fitur PROFESIONAL ENHANCED:
    - White balance correction (Gray World Algorithm)
    - Smart crop sampling (30% area tengah wajah)
    - Brightness gating (validasi pencahayaan)
    - Obstruction detection (deteksi halangan wajah)
    - CLAHE enhancement (normalisasi kontras)
    - Multi-region sampling (ensemble prediction)
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
                # Pilih metode prediksi: Multi-Region atau Single Region
                if use_multi_region:
                    # Multi-Region Ensemble Prediction
                    regions, region_names = get_multi_region_crops(frame, x1, y1, x2, y2)
                    
                    if regions:
                        current_label, confidence, all_preds = ensemble_predict(
                            regions, skin_tone_model, 
                            use_clahe=use_clahe, 
                            use_white_balance=use_white_balance
                        )
                        
                        # Visualisasi region yang disampling (optional, untuk debugging)
                        # Kotak-kotak kecil di area sampling
                        for i, (region, name) in enumerate(zip(regions, region_names)):
                            pass  # Bisa tambahkan visualisasi kotak kecil di sini
                    else:
                        # Fallback ke single region jika multi-region gagal
                        current_label, confidence = None, 0.0
                else:
                    # Single Region Prediction (metode lama)
                    processed_face = face_for_prediction.copy()
                    
                    if use_clahe:
                        processed_face = apply_clahe(processed_face)
                    
                    resized_face = cv2.resize(processed_face, (224, 224))
                    img_array = np.expand_dims(resized_face, axis=0) / 255.0
                    
                    prediction = skin_tone_model.predict(img_array, verbose=0)
                    idx = np.argmax(prediction)
                    current_label = LABELS[idx]
                    confidence = float(prediction[0][idx])
                
                # Skip jika prediksi gagal
                if current_label is None:
                    warning_msg = "⚠️ Gagal memprediksi, coba dekatkan wajah"
                    continue
                
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
                if warning_msg and "Confidence rendah" not in warning_msg:
                    cv2.putText(annotated_frame, warning_msg.replace("⚠️ ", "").replace("⏳ ", ""), 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    label_text = f"{(detected_skin or current_label).upper()}"
                    cv2.putText(annotated_frame, label_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Tampilkan confidence dan mode di bawah kotak
                    mode_text = "Multi" if use_multi_region else "Single"
                    cv2.putText(annotated_frame, f"Conf: {confidence:.2f} ({mode_text})", (x1, y2+20), 
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
                               use_brightness_gate=True, use_obstruction_detect=True,
                               use_clahe=True, use_multi_region=True):
    """
    Detect faces and classify skin tones from uploaded image dengan fitur PROFESIONAL:
    - Smart crop sampling (30% area tengah wajah)
    - White balance correction  
    - Brightness gating
    - Obstruction detection
    - CLAHE enhancement
    - Multi-region sampling (ensemble prediction)
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
            if use_multi_region:
                # Multi-Region Ensemble Prediction
                regions, region_names = get_multi_region_crops(img_bgr, x1, y1, x2, y2)
                
                if regions:
                    skin_tone_label, skin_tone_conf, all_preds = ensemble_predict(
                        regions, skin_tone_model,
                        use_clahe=use_clahe,
                        use_white_balance=use_white_balance
                    )
                else:
                    # Fallback
                    skin_tone_label, skin_tone_conf = "unknown", 0.0
            else:
                # Single Region Prediction
                processed_face = face_for_prediction.copy()
                
                if use_clahe:
                    processed_face = apply_clahe(processed_face)
                
                resized_face = cv2.resize(processed_face, (224, 224))
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
    # HERO SECTION - More Eye-catching
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #7c3aed 100%);
        border-radius: 30px;
        padding: 50px 30px;
        margin-bottom: 30px;
        text-align: center;
        border: 2px solid rgba(139, 92, 246, 0.5);
        box-shadow: 0 20px 60px rgba(124, 58, 237, 0.4);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: pulse 4s ease-in-out infinite;
        "></div>
        <h1 style="
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #fcd34d, #c4b5fd);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 3s ease infinite;
            margin: 0;
            position: relative;
            z-index: 1;
        ">💎 Skin Tone AI Professional</h1>
        <p style="
            color: #c4b5fd;
            font-size: 1.2rem;
            margin-top: 15px;
            position: relative;
            z-index: 1;
        ">Sistem Rekomendasi Warna Pakaian dengan Kecerdasan Buatan</p>
        <div style="
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 25px;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        ">
            <span style="background: rgba(34, 197, 94, 0.3); color: #86efac; padding: 8px 16px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #22c55e;">✅ YOLO Face Detection</span>
            <span style="background: rgba(59, 130, 246, 0.3); color: #93c5fd; padding: 8px 16px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #3b82f6;">✅ AI Skin Analysis</span>
            <span style="background: rgba(249, 115, 22, 0.3); color: #fdba74; padding: 8px 16px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #f97316;">✅ Smart Recommendations</span>
        </div>
    </div>
    <style>
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Quick Guide - Collapsible untuk user baru
    with st.expander("📖 **Panduan Cepat** - Klik untuk membuka", expanded=False):
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top: 0;">👋 Selamat Datang!</h4>
            <p>Aplikasi ini akan menganalisis warna kulit Anda dan memberikan rekomendasi warna pakaian yang paling cocok.</p>
            
            <h5>🚀 Cara Menggunakan:</h5>
            <ol>
                <li><strong>Pilih Mode Input</strong> di sidebar (Webcam atau Upload Gambar)</li>
                <li><strong>Pastikan pencahayaan cukup</strong> - tidak terlalu gelap atau terang</li>
                <li><strong>Posisikan wajah</strong> agar terlihat jelas di kamera</li>
                <li><strong>Tunggu deteksi</strong> - kotak hijau akan muncul di wajah Anda</li>
                <li><strong>Lihat hasil</strong> - skin tone dan rekomendasi warna akan ditampilkan</li>
            </ol>
            
            <h5>💡 Tips untuk Hasil Terbaik:</h5>
            <ul>
                <li>✅ Gunakan pencahayaan natural (dekat jendela)</li>
                <li>✅ Hindari lampu yang terlalu kuning/biru</li>
                <li>✅ Lepas kacamata jika ada</li>
                <li>✅ Jangan tutupi wajah dengan tangan</li>
                <li>✅ Dekatkan wajah ke kamera (30-50cm)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
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
        
        use_clahe = st.checkbox(
            "📊 CLAHE Enhancement",
            value=True,
            help="Contrast Limited Adaptive Histogram Equalization - Meningkatkan kontras untuk deteksi lebih akurat"
        )
        
        use_multi_region = st.checkbox(
            "🎯 Multi-Region Sampling",
            value=True,
            help="Ambil sampel dari 4 titik wajah (Dahi, Pipi Kiri, Pipi Kanan, Hidung) untuk hasil lebih stabil"
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
        7. ✅ CLAHE Enhancement (BARU!)
        8. ✅ Multi-Region Sampling (BARU!)

        
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
        
        # Status indicators
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.markdown("""
            <div class="metric-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">🟩</div>
                <div style="font-size: 0.75rem; color: #64748b;">Siap Analisis</div>
            </div>
            """, unsafe_allow_html=True)
        with col_stat2:
            st.markdown("""
            <div class="metric-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">🟦</div>
                <div style="font-size: 0.75rem; color: #64748b;">Area Sampling</div>
            </div>
            """, unsafe_allow_html=True)
        with col_stat3:
            st.markdown("""
            <div class="metric-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem;">🟥</div>
                <div style="font-size: 0.75rem; color: #64748b;">Perlu Perbaikan</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")  # Spacing
        
        # Quick tips inline
        st.markdown("""
        <div class="info-box" style="margin-bottom: 15px;">
            <strong>💡 Tips Cepat:</strong> Posisikan wajah di tengah • Pastikan pencahayaan cukup terang • Jarak ideal 30-50cm dari kamera
        </div>
        """, unsafe_allow_html=True)
        
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
                use_obstruction_detect=use_obstruction_detect,
                use_clahe=use_clahe,
                use_multi_region=use_multi_region
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
                
                # Show voting info dengan progress bar
                if len(st.session_state.skin_history) > 0:
                    vote_counts = Counter(st.session_state.skin_history)
                    progress = len(st.session_state.skin_history) / HISTORY_LEN
                    st.markdown(f"**📊 Voting Progress:** {len(st.session_state.skin_history)}/{HISTORY_LEN}")
                    st.progress(progress)
                    
                    # Vote distribution
                    st.caption(f"Distribution: {dict(vote_counts)}")
                
                if detected_skin and not warning_msg:
                    # Celebration effect untuk confidence tinggi
                    if confidence >= 0.7 and len(st.session_state.skin_history) >= HISTORY_LEN * 0.8:
                        st.balloons()
                    
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-box" style="text-align: center;">
                        <strong>🎉 Analisis Berhasil!</strong> Berikut adalah hasil dan rekomendasi untuk Anda.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Skin Tone Badge dengan warna
                    badge_class = f"badge-{detected_skin.replace(' ', '-')}"
                    skin_class = f"skin-{detected_skin.replace(' ', '-')}"
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <div class="skin-tone-badge {badge_class}">
                            <span class="skin-indicator {skin_class}"></span>
                            {detected_skin.upper()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Meter Visual
                    conf_percent = confidence * 100
                    conf_class = "conf-high" if conf_percent >= 70 else ("conf-medium" if conf_percent >= 50 else "conf-low")
                    
                    st.markdown(f"""
                    <div style="margin: 20px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: 600; color: #374151;">Confidence Level</span>
                            <span style="font-weight: 700; color: #6366f1;">{confidence:.1%}</span>
                        </div>
                        <div class="confidence-meter">
                            <div class="confidence-fill {conf_class}" style="width: {conf_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Rekomendasi
                    st.markdown("### 👔 Rekomendasi Warna Pakaian")
                    recommendation = get_recommendation(detected_skin)
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>✨ Cocok untuk {detected_skin.upper()}</h4>
                        {recommendation.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Mode: Upload Image
    else:
        st.markdown("### 📤 Upload Image")
        
        # Tips untuk upload
        col_tip1, col_tip2 = st.columns(2)
        with col_tip1:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Gambar yang Bagus:</strong><br>
                • Wajah terlihat jelas<br>
                • Pencahayaan merata<br>
                • Resolusi cukup tinggi
            </div>
            """, unsafe_allow_html=True)
        with col_tip2:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Hindari:</strong><br>
                • Wajah tertutup masker<br>
                • Foto terlalu gelap/terang<br>
                • Gambar blur/buram
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")  # Spacing
        
        # Detection Confidence Slider dengan penjelasan
        st.markdown("""
        <div class="info-box" style="margin-bottom: 10px;">
            <strong>🎚️ Detection Confidence</strong> — Sensitivitas deteksi wajah<br>
            <small style="color: #64748b;">
                • <strong>Tinggi (0.7-1.0):</strong> Hanya wajah yang sangat jelas terdeteksi<br>
                • <strong>Sedang (0.5):</strong> Keseimbangan (default)<br>
                • <strong>Rendah (0.2-0.4):</strong> Lebih sensitif, cocok untuk gambar jauh/kurang jelas
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        conf_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Turunkan nilai ini jika wajah tidak terdeteksi. Naikkan jika ada false detection."
        )
        
        st.markdown("")  # Spacing
        
        uploaded_file = st.file_uploader(
            "📁 Pilih gambar wajah Anda",
            type=['jpg', 'jpeg', 'png', 'heic', 'heif'],
            help="Format yang didukung: JPG, JPEG, PNG, HEIC, HEIF. Ukuran maksimal: 200MB"
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
                    use_obstruction_detect=use_obstruction_detect,
                    use_clahe=use_clahe,
                    use_multi_region=use_multi_region
                )
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                st.image(annotated_image, use_container_width=True)
            
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            if detections:
                st.markdown(f"### 📋 Terdeteksi {len(detections)} Wajah")
                
                for idx, det in enumerate(detections):
                    # Status badge color
                    status = det.get('status', 'OK')
                    status_icon = "✅" if status == "OK" else "⚠️"
                    skin_tone = det['skin_tone']
                    
                    with st.expander(f"👤 Face #{idx + 1} - {skin_tone.upper()} {status_icon}", expanded=True):
                        # Skin Tone Badge
                        badge_class = f"badge-{skin_tone.replace(' ', '-')}"
                        skin_class = f"skin-{skin_tone.replace(' ', '-')}"
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin: 15px 0;">
                            <div class="skin-tone-badge {badge_class}">
                                <span class="skin-indicator {skin_class}"></span>
                                {skin_tone.upper()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**📊 Hasil Deteksi**")
                            
                            # Confidence Meter
                            conf_percent = det['skin_tone_conf'] * 100
                            conf_class = "conf-high" if conf_percent >= 70 else ("conf-medium" if conf_percent >= 50 else "conf-low")
                            
                            st.markdown(f"""
                            <div style="margin: 10px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span style="font-weight: 500;">Confidence</span>
                                    <span style="font-weight: 700; color: #6366f1;">{det['skin_tone_conf']:.1%}</span>
                                </div>
                                <div class="confidence-meter">
                                    <div class="confidence-fill {conf_class}" style="width: {conf_percent}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Detection Score:** {det['detection_conf']:.1%}")
                        
                        with col_b:
                            st.markdown("**🔬 Diagnostik**")
                            brightness = det.get('brightness', 0)
                            skin_ratio = det.get('skin_ratio', 0)
                            
                            # Mini metrics
                            st.markdown(f"""
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0;">
                                <div class="metric-card" style="padding: 12px;">
                                    <div style="font-size: 1.2rem; font-weight: 700; color: #6366f1;">{brightness:.0f}</div>
                                    <div style="font-size: 0.75rem; color: #64748b;">Brightness</div>
                                </div>
                                <div class="metric-card" style="padding: 12px;">
                                    <div style="font-size: 1.2rem; font-weight: 700; color: #6366f1;">{skin_ratio:.0%}</div>
                                    <div style="font-size: 0.75rem; color: #64748b;">Skin Ratio</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if status != "OK":
                                st.warning(f"Status: {status}")
                        
                        # Rekomendasi
                        st.markdown("**👔 Rekomendasi Warna:**")
                        recommendation = get_recommendation(skin_tone)
                        st.markdown(f"""
                        <div class="recommendation-box" style="padding: 16px; margin-top: 10px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1rem;">✨ Cocok untuk {skin_tone.upper()}</h4>
                            {recommendation.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0 0 10px 0;">⚠️ Wajah Tidak Terdeteksi</h4>
                    <p style="margin: 0 0 10px 0;">AI tidak dapat menemukan wajah dalam gambar ini. Coba tips berikut:</p>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Pastikan wajah terlihat jelas dan tidak tertutup</li>
                        <li>Gunakan gambar dengan pencahayaan yang cukup</li>
                        <li>Turunkan nilai <strong>Detection Confidence</strong> di sidebar</li>
                        <li>Gunakan gambar dengan resolusi lebih tinggi</li>
                        <li>Pastikan wajah menghadap ke depan (frontal)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 8px;">
            💎 Skin Tone AI Professional
        </div>
        <div style="color: #64748b; font-size: 0.9rem;">
            Made with ❤️ using Streamlit, YOLOv8 Face, and TensorFlow
        </div>
        <div style="margin-top: 15px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <span class="feature-badge">Gray World AWB</span>
            <span class="feature-badge">Smart Crop</span>
            <span class="feature-badge">CLAHE</span>
            <span class="feature-badge">Multi-Region</span>
            <span class="feature-badge">Ensemble AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
