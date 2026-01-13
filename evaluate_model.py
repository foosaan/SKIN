"""
📊 SKIN TONE MODEL EVALUATION SCRIPT
=====================================
Script untuk mengevaluasi akurasi model Skin Tone Classification
dan YOLOv8 Face Detection.

Cara Pakai:
1. Siapkan folder test_images dengan subfolder per kelas:
   test_images/
   ├── dark/
   ├── light/
   ├── mid-dark/
   └── mid-light/

2. Jalankan: python evaluate_model.py

Output:
- Classification Report (precision, recall, F1-score)
- Confusion Matrix (saved as PNG)
- Overall Accuracy percentage
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ===== CONFIGURATION =====
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_PATH, "YOLOv8n Face.pt")
SKIN_MODEL_PATH = os.path.join(BASE_PATH, "Skin_Tone_Model_v3.h5")
TEST_IMAGES_PATH = os.path.join(BASE_PATH, "test_images")  # Folder dengan subfolder per kelas

LABELS = ['dark', 'light', 'mid-dark', 'mid-light']
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.35

# ===== LOAD MODELS =====
print("📥 Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH)
print("✅ Models loaded!")

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

def predict_skin_tone(img_path):
    """
    Prediksi skin tone dari gambar.
    Returns: (predicted_label, confidence, face_detected)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, 0.0, False
    
    # Face detection
    results = yolo_model(img, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box, h_box = x2 - x1, y2 - y1
            
            if w_box < 80 or h_box < 80:
                continue
            
            # Smart crop
            crop_x1, crop_y1, crop_x2, crop_y2 = smart_crop_face(x1, y1, x2, y2)
            h, w = img.shape[:2]
            crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
            crop_x2, crop_y2 = min(w, crop_x2), min(h, crop_y2)
            
            face_center = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_center.size == 0:
                continue
            
            # Preprocessing
            processed = correct_white_balance(face_center)
            processed = apply_clahe(processed)
            
            # Predict
            resized = cv2.resize(processed, IMG_SIZE)
            img_array = np.expand_dims(resized, axis=0) / 255.0
            pred = skin_model.predict(img_array, verbose=0)
            idx = np.argmax(pred)
            confidence = float(pred[0][idx])
            label = LABELS[idx]
            
            return label, confidence, True
    
    return None, 0.0, False

def evaluate_yolo_detection(test_path):
    """Evaluate YOLO face detection rate"""
    print("\n" + "="*50)
    print("🎯 YOLO FACE DETECTION EVALUATION")
    print("="*50)
    
    total_images = 0
    faces_detected = 0
    
    for class_name in LABELS:
        class_path = os.path.join(test_path, class_name)
        if not os.path.exists(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            total_images += 1
            results = yolo_model(img, verbose=False)
            
            for r in results:
                if len(r.boxes) > 0:
                    faces_detected += 1
                    break
    
    if total_images > 0:
        detection_rate = (faces_detected / total_images) * 100
        print(f"\n📊 Detection Results:")
        print(f"   Total images: {total_images}")
        print(f"   Faces detected: {faces_detected}")
        print(f"   Detection rate: {detection_rate:.2f}%")
        return detection_rate
    else:
        print("❌ No images found in test folder!")
        return 0.0

def evaluate_skin_tone_classification(test_path):
    """Evaluate skin tone classification accuracy"""
    print("\n" + "="*50)
    print("🎨 SKIN TONE CLASSIFICATION EVALUATION")
    print("="*50)
    
    y_true = []
    y_pred = []
    confidences = []
    skipped = 0
    
    for class_name in LABELS:
        class_path = os.path.join(test_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ Folder tidak ditemukan: {class_path}")
            continue
        
        print(f"\n📂 Processing: {class_name}")
        class_images = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        for img_name in class_images:
            img_path = os.path.join(class_path, img_name)
            pred_label, confidence, face_detected = predict_skin_tone(img_path)
            
            if face_detected and pred_label:
                y_true.append(class_name)
                y_pred.append(pred_label)
                confidences.append(confidence)
                status = "✓" if pred_label == class_name else "✗"
                print(f"   {status} {img_name}: {pred_label} ({confidence:.1%})")
            else:
                skipped += 1
                print(f"   ⚠️ {img_name}: No face detected")
    
    if len(y_true) == 0:
        print("\n❌ Tidak ada gambar yang berhasil diproses!")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📊 CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))
    
    print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 Average Confidence: {np.mean(confidences) * 100:.2f}%")
    print(f"⚠️ Skipped (no face): {skipped}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Confusion Matrix - Skin Tone Classification\nAccuracy: {accuracy*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    output_path = os.path.join(BASE_PATH, 'confusion_matrix_evaluation.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Confusion matrix saved: {output_path}")
    plt.show()
    
    return accuracy

def main():
    print("\n" + "="*60)
    print("  📊 SKIN TONE AI - MODEL EVALUATION")
    print("="*60)
    
    # Check test folder
    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"\n❌ Test folder tidak ditemukan: {TEST_IMAGES_PATH}")
        print("\n📁 Buat folder dengan struktur:")
        print("   test_images/")
        print("   ├── dark/      (gambar wajah kulit gelap)")
        print("   ├── light/     (gambar wajah kulit terang)")
        print("   ├── mid-dark/  (gambar wajah kulit sawo matang gelap)")
        print("   └── mid-light/ (gambar wajah kulit sawo matang terang)")
        
        # Create folder structure
        create = input("\n🔧 Buat folder sekarang? (y/n): ")
        if create.lower() == 'y':
            for label in LABELS:
                os.makedirs(os.path.join(TEST_IMAGES_PATH, label), exist_ok=True)
            print("✅ Folder dibuat! Tambahkan gambar test dan jalankan lagi.")
        return
    
    # Count images
    total = 0
    for label in LABELS:
        class_path = os.path.join(TEST_IMAGES_PATH, label)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            print(f"   {label}: {count} images")
            total += count
    
    if total == 0:
        print("\n❌ Tidak ada gambar di folder test!")
        print("   Tambahkan gambar ke subfolder masing-masing kelas.")
        return
    
    print(f"\n📊 Total test images: {total}")
    
    # Run evaluations
    detection_rate = evaluate_yolo_detection(TEST_IMAGES_PATH)
    accuracy = evaluate_skin_tone_classification(TEST_IMAGES_PATH)
    
    # Summary
    print("\n" + "="*60)
    print("  📋 EVALUATION SUMMARY")
    print("="*60)
    print(f"   🎯 YOLO Face Detection Rate: {detection_rate:.2f}%")
    if accuracy:
        print(f"   🎨 Skin Tone Classification Accuracy: {accuracy * 100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
