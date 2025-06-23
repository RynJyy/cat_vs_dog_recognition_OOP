#import necessary libraries
from image_classify import ImageClassifier
from tensorflow.keras import layers, models

#insert ImageClassifier class
class CatDogClassifier(ImageClassifier):
    def build_model(self):
        self.model = models.Sequential([
            #check the CNN architecture
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])