#import necessary modules for image classification
from tensorflow.keras import layers, models  # Used to build the CNN model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image loading and augmentation
import matplotlib.pyplot as plt  # For plotting training history

#class for image classification
class ImageClassifier:
    def __init__(self, image_size=(160, 160), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None