import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Import for the detailed statistics report ---
from sklearn.metrics import classification_report

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
dataset_path = 'brain_tumor_dataset/'
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Testing')

IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32
INITIAL_EPOCHS = 10 # Epochs for the initial training
FINE_TUNE_EPOCHS = 5  # Epochs for fine-tuning
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
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
    color_mode='rgb',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

print("Classes found:", train_generator.class_indices)

# ---------------------------
# 3. MODEL BUILDING
# ---------------------------
print("Building the MobileNetV2 model...")
inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)

# Freeze the base model initially
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# ---------------------------
# 4. INITIAL TRAINING (Top Layers)
# ---------------------------
print(f"\nStarting initial training for {INITIAL_EPOCHS} epochs...")

# We only use ModelCheckpoint now, no EarlyStopping
callbacks = [
    ModelCheckpoint('lightweight_brain_tumor_best.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ---------------------------
# 5. FINE-TUNING (For Better Results)
# ---------------------------
print("\nStarting fine-tuning...")

# Unfreeze the top layers of the model
base_model.trainable = True

# We'll fine-tune from this layer onwards. Deeper layers have more generic features.
fine_tune_at = 100 

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5), # 0.00001
    loss='categorical_crossentropy',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print(f"Fine-tuning for {FINE_TUNE_EPOCHS} more epochs...")

history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS, # Continues from where it left off
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=callbacks
)


# --------------------------------------------------------------------
# 6. FINAL EVALUATION & DETAILED STATS REPORT
#
# <<< THIS IS THE SECTION THAT PRINTS THE STATS YOU ASKED FOR >>>
# It uses the best model saved during training to give the most accurate report.
# --------------------------------------------------------------------
print("\n--- Loading best model for final evaluation ---")
model.load_weights('lightweight_brain_tumor_best.h5')

print("\nEvaluating final model on validation data...")
final_loss, final_accuracy, final_precision, final_recall = model.evaluate(validation_generator)
print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")
print(f"Final Validation Precision: {final_precision * 100:.2f}%")
print(f"Final Validation Recall: {final_recall * 100:.2f}%")

print("\n--- DETAILED CLASSIFICATION REPORT ---")
y_true = validation_generator.classes
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
class_names = list(validation_generator.class_indices.keys())

# Generate and print the report with F1-score, recall, etc.
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)