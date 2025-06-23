#import necessary class
from cat_dog_classify import CatDogClassifier

#main function to train the model
if __name__ == "__main__":
    #load the data
    classifier = CatDogClassifier()
    classifier.load_data(
        train_dir='data/train',
        test_dir='data/test'
    )  
    #define the training process
    classifier.build_model()
    classifier.compile_model()
    classifier.train(epochs=10)
    classifier.save_model('dogs_vs_cats_classifier.h5')
