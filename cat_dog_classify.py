#import necessary libraries
from image_classify import ImageClassifier
from tensorflow.keras import layers, models
import tensorflow as tf


#insert ImageClassifier class
class CatDogClassifier(ImageClassifier):
    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # freeze the base model
        self.model = models.Sequential([
            #check the CNN architecture
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation = 'relu'),
            layers.Dropout(0,5), 
            layers.Dense(1, activation='sigmoid')
        ])

