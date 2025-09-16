import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

DATA_DIR = 'brain_tumor_dataset/'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Testing')
IMG_SIZE = (224, 224)
BATCH = 16  # Smaller batch size for stability
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_FILE = 'model.h5'

def train():
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.15,
                                   height_shift_range=0.15, shear_range=0.15, zoom_range=0.15,
                                   horizontal_flip=True, fill_mode='nearest', validation_split=0.2)
    test_gen = ImageDataGenerator(rescale=1./255)
    train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
                                              class_mode='categorical', subset='training')
    val_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
                                            class_mode='categorical', subset='validation')
    weights = compute_class_weight(class_weight='balanced',
                               classes=np.unique(train_data.classes),
                               y=train_data.classes)
    class_weights = dict(enumerate(weights))


    inp = Input(shape=(*IMG_SIZE, 3))
    base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inp)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = Dropout(0.6)(x)
    out = Dense(len(CLASSES), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, min_lr=1e-6)
    ]

    h = model.fit(train_data, epochs=7, validation_data=val_data,
                  class_weight=class_weights, callbacks=callbacks)

    base.trainable = True
    for layer in base.layers[:-100]: layer.trainable = False
    model.compile(optimizer=Adam(5e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    hf = model.fit(train_data, epochs=4, validation_data=val_data, callbacks=callbacks)

    acc = h.history['accuracy'] + hf.history.get('accuracy', [])
    val_acc = h.history['val_accuracy'] + hf.history.get('val_accuracy', [])
    loss = h.history['loss'] + hf.history.get('loss', [])
    val_loss = h.history['val_loss'] + hf.history.get('val_loss', [])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(); plt.title('Loss')
    plt.tight_layout(); plt.show()

    model.load_weights(MODEL_FILE)
    val_labels = val_data.classes
    pred = np.argmax(model.predict(val_data), axis=1)
    print(classification_report(val_labels, pred, target_names=CLASSES))

def launch_app():
    model = load_model(MODEL_FILE)
    root = Tk(); root.title("Brain Tumor Classifier"); root.geometry("500x500")
    Label(root, text="Brain Tumor Detection", font=("Arial", 18, "bold")).pack(pady=10)
    img_lbl = Label(root); img_lbl.pack()
    res_lbl = Label(root, text="Upload an image to classify", font=("Arial", 14, "italic")); res_lbl.pack(pady=20)

    def predict():
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path: return
        img = Image.open(path); img.thumbnail((300, 300))
        img_lbl.config(image=ImageTk.PhotoImage(img)); img_lbl.image = ImageTk.PhotoImage(img)
        img_array = img_to_array(load_img(path, target_size=IMG_SIZE))/255.0
        pred = model.predict(np.expand_dims(img_array, axis=0))[0]
        cls = CLASSES[np.argmax(pred)].upper(); conf = 100 * np.max(pred)
        res_lbl.config(text=f"Prediction: {cls}\nConfidence: {conf:.2f}%", font=("Arial", 16, "bold"))

    Button(root, text="Upload & Predict", command=predict, font=("Arial", 12)).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    train()
    launch_app()
