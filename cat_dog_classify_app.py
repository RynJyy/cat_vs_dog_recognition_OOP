#import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class CatDogClassifierGUI:
    def __init__(self, model_path='dogs_vs_cats_transfer.h5', image_size=(160, 160)):
        # Load trained model
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        #create GUI window
        self.window = tk.Tk()
        self.window.title("🐶 Dogs vs 🐱 Cats Classifier")
        self.window.geometry("700x700")
        self.window.resizable(False, False)

        #load background image
        background_img = Image.open("background.jpg")  #ADD YOUR BACKGROUND IMAGE HERE
        background_img = background_img.resize((700, 700))
        self.background_tk = ImageTk.PhotoImage(background_img)
        self.background_label = tk.Label(self.window, image=self.background_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        #insert all label
        self.result_label = tk.Label(self.window, text="Upload an image", font=("Comic Sans MS", 20, "bold"), bg="#000000", fg="#FFFFFF")
        self.result_label.pack(pady=20)
        self.img_label = tk.Label(self.window, bg="#000000")
        self.img_label.pack(pady=10)
        self.filename_label = tk.Label(self.window, text="", font=("Comic Sans MS", 12), bg="#000000", fg="#AAAAAA")
        self.filename_label.pack()

    #function to upload image
    def upload_image(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            #display image
            img = Image.open(file_path)
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.configure(image=img_tk)
            self.img_label.image = img_tk

            #prepare image for model
            img_model = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img_model)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            #predict
            prediction = self.model.predict(img_array)[0][0]

            #display result
            if prediction > 0.5:
                result = f"🐶 Prediction: Dog ({prediction:.2f})"
            else:
                result = f"🐱 Prediction: Cat ({1 - prediction:.2f})"

            self.result_label.config(text=result)

    #function to clear the image and result
    def clear_image(self): 
        self.img_label.config(image='')
        self.img_label.image = None
        self.result_label.config(text="Upload an image")
        self.filename_label.config(text="")
    
    #main looping
    def run(self):
        self.window.mainloop()

#run the GUI loop
if __name__ == "__main__":
    app = CatDogClassifierGUI()
    app.run()