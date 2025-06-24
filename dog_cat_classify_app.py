#import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

#load model
model = tf.keras.models.load_model('dogs_vs_cats_classifier.h5')

#create GUI window
window = tk.Tk()
window.title("🐶 Dogs vs 🐱 Cats Classifier")
window.geometry("600x600")
window.resizable(False, False)

#load background image
background_img = Image.open("background.jpg")  # ADD YOUR BACKGROUND IMAGE HERE
background_img = background_img.resize((600, 600))
background_tk = ImageTk.PhotoImage(background_img)

#insert all label
background_label = tk.Label(window, image=background_tk)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
result_label = tk.Label(window, text="Upload an image", font=("Comic Sans MS", 20, "bold"), bg="#000000", fg="#FFFFFF")
result_label.pack(pady=20)
img_label = tk.Label(window, bg="#000000")
img_label.pack(pady=10)

# Label to display image
img_label = tk.Label(window, bg="#000000")
img_label.pack(pady=10)

# Function to upload image
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Display image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # Prepare image for model
        img_model = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img_model)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)