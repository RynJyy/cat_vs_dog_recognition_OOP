#import necessary class
from cat_dog_classify import CatDogClassifier

#main function to train the model
if __name__ == "__main__":
    #load the data
    classifier = CatDogClassifier()
    classifier.load_data(
        train_dir='archive/dogs_vs_cats/train',
        test_dir='archive/dogs_vs_cats/test'
    )  
    #define the training process
    classifier.build_model()
    classifier.compile_model()
    classifier.train(epochs=5)
    classifier.save_model('dogs_vs_cats_classifier.h5')
