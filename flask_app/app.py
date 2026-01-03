"""
Flask App - Skin Tone AI Professional
Migrasi dari Streamlit ke Flask + TailwindCSS
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
CORS(app)

# ===== CONFIGURATION =====
LABELS = ['dark', 'light', 'mid-dark', 'mid-light']
CONFIDENCE_THRES = 0.35
MIN_FACE_SIZE = 80
BRIGHTNESS_MIN = 70
BRIGHTNESS_MAX = 210
SKIN_RATIO_MIN = 0.4

# ===== LOAD MODELS =====
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
yolo_model = None
skin_model = None

def load_models():
    global yolo_model, skin_model
    if yolo_model is None:
        yolo_path = os.path.join(BASE_PATH, "YOLOv8n Face.pt")
        yolo_model = YOLO(yolo_path)
    if skin_model is None:
        skin_path = os.path.join(BASE_PATH, "Skin Tone Model v2.h5")
        skin_model = tf.keras.models.load_model(skin_path)
    return yolo_model, skin_model

# ===== HELPER FUNCTIONS =====
def correct_white_balance(img):
    """Gray World Algorithm untuk white balance"""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def apply_clahe(img):
    """CLAHE untuk normalisasi kontras"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def smart_crop_face(x1, y1, x2, y2, ratio=0.3):
    """Crop 30% area tengah wajah"""
    w, h = x2 - x1, y2 - y1
    new_w, new_h = int(w * ratio), int(h * ratio)
    cx, cy = x1 + w // 2, y1 + h // 2
    return cx - new_w // 2, cy - new_h // 2, cx + new_w // 2, cy + new_h // 2

def check_brightness(face_crop):
    """Cek brightness"""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < BRIGHTNESS_MIN:
        return False, brightness, "Terlalu gelap"
    elif brightness > BRIGHTNESS_MAX:
        return False, brightness, "Terlalu terang"
    return True, brightness, "OK"

def create_skin_mask(img):
    """Skin locus mask menggunakan YCbCr"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_multi_region_crops(frame, x1, y1, x2, y2):
    """Ambil multiple regions dari wajah"""
    w, h = x2 - x1, y2 - y1
    regions = []
    region_names = []
    
    # Forehead (atas tengah)
    fw, fh = int(w * 0.4), int(h * 0.15)
    fx, fy = x1 + (w - fw) // 2, y1 + int(h * 0.1)
    if fx >= 0 and fy >= 0:
        forehead = frame[fy:fy+fh, fx:fx+fw]
        if forehead.size > 0:
            regions.append(forehead)
            region_names.append("forehead")
    
    # Left cheek
    cw, ch = int(w * 0.25), int(h * 0.25)
    lx, ly = x1 + int(w * 0.1), y1 + int(h * 0.4)
    if lx >= 0 and ly >= 0:
        left_cheek = frame[ly:ly+ch, lx:lx+cw]
        if left_cheek.size > 0:
            regions.append(left_cheek)
            region_names.append("left_cheek")
    
    # Right cheek
    rx, ry = x1 + int(w * 0.65), y1 + int(h * 0.4)
    if rx >= 0 and ry >= 0:
        right_cheek = frame[ry:ry+ch, rx:rx+cw]
        if right_cheek.size > 0:
            regions.append(right_cheek)
            region_names.append("right_cheek")
    
    # Nose
    nw, nh = int(w * 0.2), int(h * 0.2)
    nx, ny = x1 + (w - nw) // 2, y1 + int(h * 0.45)
    if nx >= 0 and ny >= 0:
        nose = frame[ny:ny+nh, nx:nx+nw]
        if nose.size > 0:
            regions.append(nose)
            region_names.append("nose")
    
    return regions, region_names

def ensemble_predict(regions, use_clahe=True, use_white_balance=True):
    """Ensemble prediction dari multiple regions"""
    global skin_model
    if not regions:
        return None, 0.0, []
    
    predictions = []
    raw_predictions = []
    
    for region in regions:
        processed = region.copy()
        
        if use_white_balance:
            processed = correct_white_balance(processed)
        if use_clahe:
            processed = apply_clahe(processed)
        
        try:
            resized = cv2.resize(processed, (224, 224))
            img_array = np.expand_dims(resized, axis=0) / 255.0
            
            pred = skin_model.predict(img_array, verbose=0)
            idx = np.argmax(pred)
            conf = float(pred[0][idx])
            label = LABELS[idx]
            
            predictions.append({'label': label, 'confidence': conf})
            raw_predictions.append(pred[0])
        except:
            continue
    
    if not predictions:
        return None, 0.0, []
    
    avg_pred = np.mean(raw_predictions, axis=0)
    final_idx = np.argmax(avg_pred)
    final_label = LABELS[final_idx]
    final_confidence = float(avg_pred[final_idx])
    
    # Boost for ensemble
    confidence_boost = min(0.10, (len(predictions) - 1) * 0.02)
    final_confidence = min(1.0, final_confidence + confidence_boost)
    
    return final_label, final_confidence, predictions

# ===== COLOR COMPATIBILITY SCORING =====
def hex_to_rgb(hex_color):
    """Convert hex to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for perceptual color distance"""
    r, g, b = [x / 255.0 for x in rgb]
    
    # Convert to XYZ
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = linearize(r), linearize(g), linearize(b)
    
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize for D65
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16/116)
    
    L = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))
    
    return L, a, b_val

def color_distance_lab(color1_hex, color2_hex):
    """Calculate Delta E (CIE76) - perceptual color difference"""
    lab1 = rgb_to_lab(hex_to_rgb(color1_hex))
    lab2 = rgb_to_lab(hex_to_rgb(color2_hex))
    
    delta_e = ((lab1[0] - lab2[0]) ** 2 + 
               (lab1[1] - lab2[1]) ** 2 + 
               (lab1[2] - lab2[2]) ** 2) ** 0.5
    return delta_e

def get_undertone_value(undertone_str):
    """Convert undertone description to numerical value (-1 cool, 0 neutral, 1 warm)"""
    undertone_lower = undertone_str.lower()
    if 'cool' in undertone_lower:
        return -0.7
    elif 'warm' in undertone_lower:
        return 0.7
    elif 'neutral' in undertone_lower:
        return 0.0
    elif 'golden' in undertone_lower or 'peachy' in undertone_lower:
        return 0.5
    elif 'rich' in undertone_lower or 'deep' in undertone_lower:
        return 0.3
    return 0.0

def get_color_temperature(hex_color):
    """Determine if color is warm, cool, or neutral (-1 to 1)"""
    r, g, b = hex_to_rgb(hex_color)
    
    # Warm colors have more red/yellow, cool colors have more blue
    warm_score = (r * 1.5 + g * 0.5 - b) / 255
    
    # Normalize to -1 to 1 range
    temp = max(-1, min(1, warm_score / 2))
    return temp

def calculate_contrast_ratio(skin_hex, color_hex):
    """Calculate contrast ratio between skin tone and color"""
    def luminance(hex_color):
        r, g, b = [x / 255.0 for x in hex_to_rgb(hex_color)]
        def adjust(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)
    
    l1 = luminance(skin_hex)
    l2 = luminance(color_hex)
    
    lighter = max(l1, l2)
    darker = min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)

def calculate_color_compatibility(skin_label, skin_hex, color_hex, undertone, is_recommended=True):
    """
    Calculate dynamic color compatibility score (0-5 stars, 0-100%)
    
    Factors:
    1. Undertone matching (30%)
    2. Contrast ratio (30%)
    3. Color harmony (20%)
    4. Base recommendation (20%)
    """
    score = 0.0
    
    # 1. Undertone matching (30 points)
    skin_temp = get_undertone_value(undertone)
    color_temp = get_color_temperature(color_hex)
    
    # Ideal matching: warm skin + warm color OR cool skin + cool color
    # Neutral skin can wear both
    if abs(skin_temp) < 0.3:  # Neutral skin
        undertone_score = 25  # Good with most colors
    else:
        temp_match = 1 - abs(skin_temp - color_temp) / 2
        undertone_score = temp_match * 30
    
    score += undertone_score
    
    # 2. Contrast ratio (30 points)
    # Skin tone approximate hex values
    skin_hex_approx = {
        'light': '#FFE0C4',
        'mid-light': '#D4A574',
        'mid-dark': '#8D5524',
        'dark': '#4A3728'
    }.get(skin_label, '#C68642')
    
    contrast = calculate_contrast_ratio(skin_hex_approx, color_hex)
    
    # Ideal contrast is 3:1 to 7:1
    if 3 <= contrast <= 7:
        contrast_score = 30
    elif 2 <= contrast < 3 or 7 < contrast <= 10:
        contrast_score = 22
    elif contrast < 2:
        contrast_score = 10  # Too similar
    else:
        contrast_score = 18  # Too high contrast
    
    score += contrast_score
    
    # 3. Color harmony (20 points)
    # Check if color is too close to skin tone (wash-out effect)
    distance = color_distance_lab(skin_hex_approx, color_hex)
    
    if distance < 15:  # Too similar
        harmony_score = 5
    elif distance < 30:  # Slightly similar
        harmony_score = 12
    elif distance < 60:  # Good separation
        harmony_score = 20
    else:  # Strong contrast
        harmony_score = 18
    
    score += harmony_score
    
    # 4. Base recommendation bonus (20 points)
    if is_recommended:
        score += 20
    else:
        score += 5  # Avoid colors get small base
    
    # Normalize to 0-100
    score = min(100, max(0, score))
    
    # Convert to stars (1-5)
    stars = round(score / 20)
    stars = max(1, min(5, stars))
    
    return {
        'score': round(score),
        'stars': stars,
        'undertone_match': round(undertone_score / 30 * 100),
        'contrast_quality': round(contrast_score / 30 * 100),
        'harmony': round(harmony_score / 20 * 100)
    }

def get_recommendation(skin_label):
    """
    Rekomendasi warna pakaian berdasarkan skin tone
    Sumber: Color Theory, Seasonal Color Analysis (Carole Jackson), Pantone Skin Tone Guide
    """
    recommendations = {
        "light": {
            "undertone": "Cool to Neutral",
            "season": "Winter / Summer",
            "description": "Kulit terang cenderung memiliki undertone pink atau peach. Warna-warna yang kontras tinggi akan menonjolkan kecerahan kulit tanpa membuatnya tampak pucat.",
            "good": [
                {"name": "Emerald", "hex": "#50C878", "reason": "Kontras kuat dengan kulit terang, membuat kulit terlihat lebih cerah dan sehat"},
                {"name": "Navy Blue", "hex": "#000080", "reason": "Warna klasik yang memberikan kesan elegan tanpa membuat kulit terlihat pucat"},
                {"name": "Burgundy", "hex": "#800020", "reason": "Warna deep yang menambah warmth tanpa konflik dengan undertone"},
                {"name": "Royal Blue", "hex": "#4169E1", "reason": "Saturasi tinggi membuat kulit terang tampak lebih vibrant"},
                {"name": "Hitam", "hex": "#000000", "reason": "Kontras maksimal, cocok untuk formal dan modern look"},
                {"name": "Soft Pink", "hex": "#FFB6C1", "reason": "Harmonis dengan undertone pink, lembut tapi tidak pucat"}
            ],
            "avoid": [
                {"name": "Beige Pucat", "hex": "#F5F5DC", "reason": "Terlalu mirip dengan warna kulit, membuat tampilan wash-out"},
                {"name": "Kuning Pudar", "hex": "#FFFACD", "reason": "Dapat membuat kulit terlihat sallow atau kekuningan"},
                {"name": "Neon Colors", "hex": "#FF6700", "reason": "Terlalu harsh, overwhelm kulit terang dan membuat terlihat pucat"}
            ],
            "theory": "Berdasarkan Color Theory, kulit terang membutuhkan warna dengan saturasi sedang-tinggi untuk menciptakan kontras yang seimbang. Seasonal Color Analysis mengkategorikan ini sebagai 'Winter' atau 'Summer' palette.",
            "source": "Carole Jackson - Color Me Beautiful (1980), Pantone Skin Tone Guide"
        },
        "mid-light": {
            "undertone": "Warm (Golden/Peachy)",
            "season": "Autumn / Spring",
            "description": "Kulit sawo matang terang memiliki undertone hangat keemasan. Warna-warna earth tone dan warm colors akan menonjolkan kehangatan alami kulit.",
            "good": [
                {"name": "Terracotta", "hex": "#E2725B", "reason": "Harmonis dengan undertone warm, membuat kulit glow natural"},
                {"name": "Olive Green", "hex": "#808000", "reason": "Earth tone yang complement kehangatan kulit"},
                {"name": "Mustard", "hex": "#FFDB58", "reason": "Warna hangat yang menonjolkan golden undertone"},
                {"name": "Coral", "hex": "#FF7F50", "reason": "Vibrant tapi tetap warm, membuat wajah terlihat segar"},
                {"name": "Camel", "hex": "#C19A6B", "reason": "Netral hangat yang sophisticated"},
                {"name": "Teal", "hex": "#008080", "reason": "Kontras yang menarik dengan skin warm undertone"}
            ],
            "avoid": [
                {"name": "Cool Grey", "hex": "#808080", "reason": "Undertone dingin konflik dengan warmth kulit, membuat kusam"},
                {"name": "Neon Pink", "hex": "#FF6EC7", "reason": "Terlalu cool-toned, tidak harmonis"},
                {"name": "Icy Blue", "hex": "#A5F2F3", "reason": "Warna dingin yang membuat kulit terlihat tidak sehat"}
            ],
            "theory": "Seasonal Color Analysis mengidentifikasi ini sebagai 'Autumn' atau 'Spring' palette. Warna-warna dengan base kuning/orange akan harmonis dengan undertone emas alami.",
            "source": "Munsell Color System, Color Me Beautiful methodology"
        },
        "mid-dark": {
            "undertone": "Warm to Neutral (Rich/Deep)",
            "season": "Deep Autumn / Deep Winter",
            "description": "Kulit sawo matang gelap memiliki kedalaman warna yang indah. Warna-warna rich dan jewel tones akan menonjolkan kekayaan warna kulit.",
            "good": [
                {"name": "Gold", "hex": "#FFD700", "reason": "Memberikan efek luminous, melengkapi warmth kulit"},
                {"name": "Maroon", "hex": "#800000", "reason": "Deep dan rich, menciptakan harmony yang elegan"},
                {"name": "Emerald", "hex": "#50C878", "reason": "Jewel tone yang membuat kulit terlihat radiant"},
                {"name": "Purple Deep", "hex": "#7B1FA2", "reason": "Warna regal yang complement richness kulit"},
                {"name": "Burnt Orange", "hex": "#CC5500", "reason": "Warm dan earthy, sangat flattering"},
                {"name": "Cobalt Blue", "hex": "#0047AB", "reason": "Kontras yang striking tapi sophisticated"}
            ],
            "avoid": [
                {"name": "Coklat Kusam", "hex": "#8B7355", "reason": "Terlalu dekat dengan skin tone, membuat flat"},
                {"name": "Beige Mati", "hex": "#C8AD7F", "reason": "Tidak memberikan kontras, terlihat monoton"},
                {"name": "Pastel Lemah", "hex": "#E6E6FA", "reason": "Kurang kontras dengan kedalaman kulit"}
            ],
            "theory": "Untuk kulit dengan depth yang kaya, dibutuhkan warna dengan intensitas yang sama untuk menciptakan balance. Jewel tones dan rich colors adalah pilihan optimal.",
            "source": "Pantone SkinTone Guide, Professional Image Consulting standards"
        },
        "dark": {
            "undertone": "Rich/Deep (Varies)",
            "season": "Deep Winter / Bright Spring",
            "description": "Kulit gelap yang kaya pigmen dapat memakai hampir semua warna dengan baik, terutama warna-warna bright dan kontras tinggi yang menonjolkan keindahan kulit.",
            "good": [
                {"name": "Putih", "hex": "#FFFFFF", "reason": "Kontras maksimal, clean dan striking look"},
                {"name": "Kuning Lemon", "hex": "#FFF44F", "reason": "Bright dan cheerful, beautiful contrast"},
                {"name": "Fuchsia", "hex": "#FF00FF", "reason": "Vibrant pink yang stunning dengan dark skin"},
                {"name": "Cobalt Blue", "hex": "#0047AB", "reason": "Electric dan powerful contrast"},
                {"name": "Merah Bright", "hex": "#FF0000", "reason": "Bold dan confident, sangat flattering"},
                {"name": "Orange Vibrant", "hex": "#FF6600", "reason": "Warm dan energetic, beautiful glow"}
            ],
            "avoid": [
                {"name": "Navy Sangat Gelap", "hex": "#000033", "reason": "Kurang kontras, blends dengan kulit"},
                {"name": "Coklat Gelap", "hex": "#3D2914", "reason": "Tidak memberikan separation yang cukup"},
                {"name": "Hitam Solid", "hex": "#000000", "reason": "Untuk casual, kurang pop (untuk formal OK)"}
            ],
            "theory": "Dark skin memiliki privilege bisa memakai warna-warna paling bright tanpa terlihat overwhelming. Kontras adalah kunci untuk menonjolkan beauty kulit gelap.",
            "source": "Pantone SkinTone Guide, Fashion Industry Standards"
        }
    }
    return recommendations.get(skin_label, {
        "undertone": "Unknown",
        "season": "Unknown", 
        "description": "Tidak dapat menentukan",
        "good": [],
        "avoid": [],
        "theory": "",
        "source": ""
    })

def get_recommendation_with_scores(skin_label):
    """Get recommendations with dynamic compatibility scores"""
    base_rec = get_recommendation(skin_label)
    
    undertone = base_rec.get('undertone', 'Unknown')
    
    # Calculate scores for good colors
    good_with_scores = []
    for color in base_rec.get('good', []):
        score_data = calculate_color_compatibility(
            skin_label, 
            None,  # Will use approximate
            color['hex'], 
            undertone, 
            is_recommended=True
        )
        good_with_scores.append({
            **color,
            'score': score_data['score'],
            'stars': score_data['stars'],
            'score_details': {
                'undertone_match': score_data['undertone_match'],
                'contrast': score_data['contrast_quality'],
                'harmony': score_data['harmony']
            }
        })
    
    # Calculate scores for avoid colors
    avoid_with_scores = []
    for color in base_rec.get('avoid', []):
        score_data = calculate_color_compatibility(
            skin_label, 
            None, 
            color['hex'], 
            undertone, 
            is_recommended=False
        )
        avoid_with_scores.append({
            **color,
            'score': score_data['score'],
            'stars': score_data['stars'],
            'score_details': {
                'undertone_match': score_data['undertone_match'],
                'contrast': score_data['contrast_quality'],
                'harmony': score_data['harmony']
            }
        })
    
    # Sort by score
    good_with_scores.sort(key=lambda x: x['score'], reverse=True)
    avoid_with_scores.sort(key=lambda x: x['score'], reverse=False)
    
    return {
        **base_rec,
        'good': good_with_scores,
        'avoid': avoid_with_scores
    }

def analyze_image(img_bgr, use_multi_region=True, use_clahe=True, use_white_balance=True):
    """Fungsi utama untuk analisis gambar"""
    global yolo_model, skin_model
    load_models()
    
    results = []
    annotated = img_bgr.copy()
    
    # Detect faces
    detections = yolo_model(img_bgr, verbose=False)
    
    for r in detections:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Size check
            w_box, h_box = x2 - x1, y2 - y1
            if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                continue
            
            # Smart crop
            crop_x1, crop_y1, crop_x2, crop_y2 = smart_crop_face(x1, y1, x2, y2)
            h, w = img_bgr.shape[:2]
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(w, crop_x2), min(h, crop_y2)
            
            face_center = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_center.size == 0:
                continue
            
            # Brightness check
            is_bright_ok, brightness, bright_msg = check_brightness(face_center)
            
            # Skin visibility check
            mask = create_skin_mask(face_center)
            skin_ratio = cv2.countNonZero(mask) / (mask.size + 1e-5)
            
            # Prediction
            if use_multi_region:
                regions, _ = get_multi_region_crops(img_bgr, x1, y1, x2, y2)
                if regions:
                    skin_label, skin_conf, _ = ensemble_predict(regions, use_clahe, use_white_balance)
                else:
                    skin_label, skin_conf = "unknown", 0.0
            else:
                processed = face_center.copy()
                if use_white_balance:
                    processed = correct_white_balance(processed)
                if use_clahe:
                    processed = apply_clahe(processed)
                
                resized = cv2.resize(processed, (224, 224))
                img_array = np.expand_dims(resized, axis=0) / 255.0
                pred = skin_model.predict(img_array, verbose=0)
                idx = np.argmax(pred)
                skin_label = LABELS[idx]
                skin_conf = float(pred[0][idx])
            
            # Draw annotations
            color = (0, 255, 0) if is_bright_ok and skin_ratio >= SKIN_RATIO_MIN else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(annotated, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 2)
            
            label_text = f"{skin_label.upper()} ({skin_conf:.0%})"
            cv2.putText(annotated, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            results.append({
                'skin_tone': skin_label,
                'confidence': skin_conf,
                'brightness': brightness,
                'skin_ratio': skin_ratio,
                'recommendation': get_recommendation_with_scores(skin_label),
                'bbox': [x1, y1, x2, y2]
            })
    
    # Encode annotated image
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return results, img_base64

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'image_base64' in request.json:
            # Base64 from webcam
            img_data = request.json['image_base64']
            img_data = img_data.split(',')[1] if ',' in img_data else img_data
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        results, annotated_base64 = analyze_image(img)
        
        return jsonify({
            'success': True,
            'results': results,
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Models loaded! Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
