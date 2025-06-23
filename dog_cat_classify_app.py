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