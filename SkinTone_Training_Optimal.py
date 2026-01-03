"""
🎯 SKIN TONE CLASSIFICATION - OPTIMAL TRAINING NOTEBOOK
=========================================================
Notebook ini dioptimasi untuk Google Colab dengan GPU.
Fitur:
- Transfer Learning (EfficientNetB0/MobileNetV2)
- Data Augmentation lengkap
- Class Weights untuk data tidak seimbang
- Callbacks: Early Stopping, LR Scheduler, Model Checkpoint
- Evaluasi lengkap dengan Confusion Matrix

CARA PAKAI:
1. Upload ke Google Colab
2. Aktifkan GPU: Runtime > Change runtime type > GPU
3. Upload dataset ke Google Drive
4. Jalankan semua cell

Author: AI Assistant
"""

# ===============================================
# CELL 1: INSTALL & IMPORT
# ===============================================
# Uncomment jika di Colab:
# !pip install tensorflow scikit-learn matplotlib seaborn

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ===============================================
# CELL 2: KONFIGURASI
# ===============================================
# 📁 GANTI PATH INI SESUAI LOKASI DATASET KAMU
# Struktur folder harus:
# dataset/
#   ├── dark/
#   ├── light/
#   ├── mid-dark/
#   └── mid-light/

# Untuk Google Colab + Google Drive:
# from google.colab import drive 
# drive.mount('/content/drive')
# DATASET_PATH = '/content/drive/MyDrive/dataset_skin_tone'

# Untuk lokal:
DATASET_PATH = '/path/to/your/dataset'  # <-- GANTI INI!

# Konfigurasi Training
CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 100,  # Akan berhenti lebih awal jika sudah konvergen
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'model_type': 'EfficientNetB0',  # atau 'MobileNetV2'
    'fine_tune_layers': 20,  # Jumlah layer terakhir yang di-unfreeze saat fine-tuning
}

# Label sesuai dengan app.py
LABELS = ['dark', 'light', 'mid-dark', 'mid-light']

print("✅ Konfigurasi loaded!")
print(f"   Model: {CONFIG['model_type']}")
print(f"   Image Size: {CONFIG['img_size']}")
print(f"   Batch Size: {CONFIG['batch_size']}")
print(f"   Max Epochs: {CONFIG['epochs']}")

# ===============================================
# CELL 3: DATA AUGMENTATION
# ===============================================
"""
Data Augmentation sangat penting untuk:
- Mencegah overfitting
- Meningkatkan generalisasi model
- Meng-handle variasi pencahayaan real-world
"""

# Training augmentation (agresif)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=CONFIG['validation_split'],
    
    # Geometric transforms
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    
    # Color/Brightness transforms (PENTING untuk skin tone!)
    brightness_range=[0.7, 1.3],  # Simulasi berbagai pencahayaan
    channel_shift_range=30,  # Variasi warna
    
    fill_mode='nearest'
)

# Validation (tanpa augmentation, hanya rescale)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=CONFIG['validation_split']
)

print("✅ Data Augmentation configured!")

# ===============================================
# CELL 4: LOAD DATASET
# ===============================================
print(f"\n📂 Loading dataset dari: {DATASET_PATH}")

# Training data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=LABELS  # Pastikan urutan label konsisten
)

# Validation data
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=LABELS
)

print(f"\n✅ Dataset loaded!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")
print(f"   Classes: {train_generator.class_indices}")

# ===============================================
# CELL 5: HITUNG CLASS WEIGHTS
# ===============================================
"""
Class weights penting jika dataset tidak seimbang.
Misalnya: 1000 gambar 'light', tapi cuma 200 gambar 'dark'
"""

# Ambil semua label dari training data
train_labels = train_generator.classes

# Hitung class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

print("\n📊 Class Distribution:")
for idx, label in enumerate(LABELS):
    count = np.sum(train_labels == idx)
    weight = class_weight_dict[idx]
    print(f"   {label}: {count} samples (weight: {weight:.2f})")

# ===============================================
# CELL 6: BUILD MODEL (TRANSFER LEARNING)
# ===============================================
def build_model(model_type='EfficientNetB0', num_classes=4):
    """
    Build model dengan Transfer Learning.
    Base model sudah di-pretrain dengan ImageNet (14 juta gambar).
    """
    
    if model_type == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif model_type == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze semua layer base model dulu
    base_model.trainable = False
    
    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model, base_model

# Build model
model, base_model = build_model(CONFIG['model_type'], num_classes=len(LABELS))

# Compile
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n✅ Model built: {CONFIG['model_type']}")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

model.summary()

# ===============================================
# CELL 7: CALLBACKS
# ===============================================
"""
Callbacks untuk training yang optimal:
1. EarlyStopping: Berhenti jika tidak ada improvement
2. ReduceLROnPlateau: Kurangi LR jika stuck
3. ModelCheckpoint: Simpan model terbaik
"""

callbacks = [
    # Stop training jika val_accuracy tidak naik selama 15 epoch
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Kurangi learning rate jika val_loss tidak turun
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Simpan model terbaik
    ModelCheckpoint(
        'best_skin_tone_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("✅ Callbacks configured!")

# ===============================================
# CELL 8: PHASE 1 - TRAIN HEAD ONLY
# ===============================================
"""
Phase 1: Train hanya classification head (base model frozen)
Ini untuk "warm up" head layers sebelum fine-tuning.
"""

print("\n" + "="*50)
print("📚 PHASE 1: Training Classification Head")
print("="*50)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Cukup 20 epoch untuk phase 1
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Phase 1 completed!")

# ===============================================
# CELL 9: PHASE 2 - FINE-TUNING
# ===============================================
"""
Phase 2: Unfreeze beberapa layer terakhir dan train dengan LR rendah.
Ini memungkinkan model menyesuaikan fitur low-level untuk skin tone.
"""

print("\n" + "="*50)
print("🔧 PHASE 2: Fine-Tuning")
print("="*50)

# Unfreeze semua layer
base_model.trainable = True

# Freeze layer-layer awal, hanya unfreeze layer terakhir
fine_tune_at = len(base_model.layers) - CONFIG['fine_tune_layers']
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile dengan learning rate sangat rendah
model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate'] / 10),  # LR 10x lebih kecil
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"   Trainable layers: {len([l for l in model.layers if l.trainable])}")

# Train lagi
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=CONFIG['epochs'],
    initial_epoch=len(history_phase1.history['accuracy']),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Phase 2 (Fine-tuning) completed!")

# ===============================================
# CELL 10: EVALUASI
# ===============================================
print("\n" + "="*50)
print("📊 EVALUASI MODEL")
print("="*50)

# Evaluasi di validation set
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=LABELS, yticklabels=LABELS)
plt.title('Confusion Matrix - Skin Tone Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\n✅ Confusion matrix saved to 'confusion_matrix.png'")

# ===============================================
# CELL 11: PLOT TRAINING HISTORY
# ===============================================
def plot_history(history1, history2=None):
    """Plot training dan validation metrics."""
    
    # Gabungkan history jika ada 2 phase
    if history2:
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
    else:
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs_range, loss, 'b-', label='Training Loss')
    ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    
    print("\n✅ Training history saved to 'training_history.png'")

plot_history(history_phase1, history_phase2)

# ===============================================
# CELL 12: SAVE FINAL MODEL
# ===============================================
# Simpan model final
model.save('Skin Tone Model v3.h5')
print("\n✅ Model saved as 'Skin Tone Model v3.h5'")

# Untuk TensorFlow Lite (mobile deployment)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('skin_tone_model.tflite', 'wb') as f:
#     f.write(tflite_model)
# print("✅ TFLite model saved!")

print("\n" + "="*50)
print("🎉 TRAINING SELESAI!")
print("="*50)
print("""
Langkah selanjutnya:
1. Download 'Skin Tone Model v3.h5' dari Colab
2. Ganti nama file menjadi 'Skin Tone Model v2.h5' (atau update path di app.py)
3. Replace model lama dengan yang baru di folder project
4. Jalankan app.py dan test akurasinya!

Tips:
- Jika val_accuracy < 85%, coba tambah data atau augmentation
- Jika train_accuracy tinggi tapi val_accuracy rendah = overfitting
- Jika keduanya rendah = model perlu lebih kompleks atau data lebih banyak
""")
