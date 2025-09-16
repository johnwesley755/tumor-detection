import os
import numpy as np
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- Configuration ----------------
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_FILE = 'model.h5'

# ---------------- GUI Application ----------------
def launch_app():
    model = load_model(MODEL_FILE)
    root = Tk()
    root.title("Brain Tumor Classifier")
    root.geometry("500x500")

    Label(root, text="Brain Tumor Detection", font=("Arial", 18, "bold")).pack(pady=10)
    img_lbl = Label(root)
    img_lbl.pack()
    res_lbl = Label(root, text="Upload an image to classify", font=("Arial", 14, "italic"))
    res_lbl.pack(pady=20)

    def predict():
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img = Image.open(path)
        img.thumbnail((300, 300))
        img_lbl.config(image=ImageTk.PhotoImage(img))
        img_lbl.image = ImageTk.PhotoImage(img)

        img_array = img_to_array(load_img(path, target_size=IMG_SIZE)) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0))[0]
        cls = CLASSES[np.argmax(pred)].upper()
        conf = 100 * np.max(pred)
        res_lbl.config(text=f"Prediction: {cls}\nConfidence: {conf:.2f}%", font=("Arial", 16, "bold"))

    Button(root, text="Upload & Predict", command=predict, font=("Arial", 12)).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found!")
        print("Please ensure the model is trained and the file is in the correct location.")
    else:
        launch_app()
