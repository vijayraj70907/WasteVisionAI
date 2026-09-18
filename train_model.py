"""
WasteVision AI — Multi-class MobileNetV2 Trainer
=================================================
Supports 7 waste classes (expandable):
  Original 4: UnsedTablets, UnusedSyringe, WasteSyringe, WasteTablets
  New 3:      Plastic, Paper, Glass

To add Plastic / Paper / Glass training data from Kaggle:
  1. Download the "Garbage Classification" dataset from Kaggle:
       https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
  2. From the downloaded zip, copy the following subfolders into dataset/:
         plastic  →  dataset/Plastic
         paper    →  dataset/Paper
         glass    →  dataset/Glass
  3. Run augmentation first if any new class has < 100 images:
         python augment_dataset.py
  4. Then train:
         python train_model.py

The script auto-detects all subfolders in dataset/, trains MobileNetV2
with transfer learning, fine-tunes the last 30 layers, and saves:
  - model/biowaste_best_model.h5   (Keras model)
  - model/class_names.json         (index → class-name mapping for app.py)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS_TOP = 15
EPOCHS_FT  = 10
DATASET    = "dataset"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "biowaste_best_model.h5")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_names.json")

os.makedirs(MODEL_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.75, 1.25],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print(f"\nDetected {num_classes} classes: {train_gen.class_indices}\n")

folder_to_label = {
    "UnsedTablets":  "Unused Tablets",
    "UnusedSyringe": "Unused Syringe",
    "WasteSyringe":  "Waste Syringe",
    "WasteTablets":  "Waste Tablets",
    "Plastic":       "Plastic Waste",
    "Paper":         "Paper Waste",
    "Glass":         "Glass Waste",
}

class_names = {}
for folder_name, idx in train_gen.class_indices.items():
    label = folder_to_label.get(folder_name, folder_name.replace("_", " ").title())
    class_names[idx] = label

print("Class index → label mapping:")
for idx, label in sorted(class_names.items()):
    print(f"  {idx}: {label}")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump({str(k): v for k, v in class_names.items()}, f, indent=2)
print(f"\nClass map saved to {CLASS_MAP_PATH}")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_top = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
]

print("\n--- Phase 1: Training top layers (base frozen) ---")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_TOP,
    callbacks=callbacks_top
)

print("\n--- Phase 2: Fine-tuning last 30 layers ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_ft = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, verbose=1, min_lr=1e-7),
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FT,
    callbacks=callbacks_ft
)

loss, acc = model.evaluate(val_gen, verbose=0)
print(f"\nFinal validation — loss: {loss:.4f}  accuracy: {acc*100:.2f}%")
print(f"Model saved to {MODEL_PATH}")
print(f"Class names saved to {CLASS_MAP_PATH}")
print("\nUpdate MODEL_VERSION in app.py before deploying:")
print('  MODEL_VERSION = "v2.0 — Multi-class Waste Specialist"')
