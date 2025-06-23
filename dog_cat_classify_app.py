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