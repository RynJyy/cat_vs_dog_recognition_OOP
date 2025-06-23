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
    