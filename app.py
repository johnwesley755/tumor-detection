import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import the lightweight MobileNetV2 model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
dataset_path = 'brain_tumor_dataset/'
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Testing')

# Using a slightly smaller image size can also speed up training
IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32
# Reduced epochs for a much faster training session
EPOCHS = 15 
NUM_CLASSES = 4

# ---------------------------
# 2. DATA PREPROCESSING & AUGMENTATION
# ---------------------------
print("Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

print("Classes found:", train_generator.class_indices)

# ---------------------------
# 3. LIGHTWEIGHT MODEL BUILDING (MobileNetV2)
# ---------------------------
print("Building the lightweight MobileNetV2 model...")

# Define the input layer
inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Load MobileNetV2 with pre-trained ImageNet weights, excluding the top classification layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

# Freeze the layers of the base model
base_model.trainable = False

# Add our custom layers on top for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001), # A slightly higher learning rate can work well for MobileNetV2
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# 4. TRAINING THE MODEL
# ---------------------------
print("\nStarting model training...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint('lightweight_brain_tumor_best.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# ---------------------------
# 5. EVALUATION & VISUALIZATION
# ---------------------------
print("\nEvaluating final model...")
final_loss, final_accuracy = model.evaluate(validation_generator)
print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")
print(f"Final Validation Loss: {final_loss:.4f}")

# Plotting training history
print("\nVisualizing training results...")
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()