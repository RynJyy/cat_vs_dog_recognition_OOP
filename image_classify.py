#import necessary modules for image classification
import tensorflow as tf
from tensorflow.keras import layers, models  # Used to build the CNN model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image loading and augmentation

#class for image classification
class ImageClassifier:
    def __init__(self, image_size=(160, 160), batch_size = 32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None

    #define method to load and validate data
    def load_data(self, train_dir, test_dir):
        train_datagen = ImageDataGenerator(
            #improves the model's performance
            rescale = 1./255,  
            rotation_range = 40, 
            width_shift_range = 0.2,  
            height_shift_range = 0.2,
            shear_range = 0.2, 
            zoom_range = 0.2,  
            horizontal_flip = True,
            fill_mode = 'nearest' 
        )
        val_datagen = ImageDataGenerator(rescale = 1./255)  #validation data rescaling

        #load test images
        self.test_generator = train_datagen.flow_from_directory(
            test_dir,
            target_size = self.image_size,
            batch_size = self.batch_size,
            class_mode = 'binary'
        )
        #load training images
        self.validation_generator = val_datagen.flow_from_directory( 
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

    #compile the model with loss function, optimizer and metrics
    def compile_model(self):
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
            metrics = ['accuracy']
        )
            
    #training model and saving the history
    def train(self, epochs = 5):
        self.history = self.model.fit(
            self.test_generator,
            epochs = epochs,
            validation_data = self.validation_generator
        )
    #save the model
    def save_model(self, filename):
        self.model.save(filename)