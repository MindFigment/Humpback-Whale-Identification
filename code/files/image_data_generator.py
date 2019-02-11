from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from pandas import read_csv
from globals import *
from keras import backend as K
import numpy as np
from utils import read_raw_image
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
from PIL import Image
from utils import expand_path


class ImageGenerator():

    def __init__(self):
        self.img_shape = (384, 384, 1)
        self.datagen_train, self.datagen_test = self.create_image_generator()


    def create_image_generator(self):
            train_data = [img for _,img,_ in read_csv(train_csv).to_records()]
            size = len(train_data) // 90
            data = np.zeros((size, ) + self.img_shape, dtype=K.floatx()) 
            for i in tqdm(range(size)):
                # data[i] = load_img(train_dir + train_data[i], target_size=self.img_shape)
                I = Image.open(expand_path(train_data[i]))
                I = I.convert('L') # convert image to grayscale
                I = I.resize((384,384))
                I = img_to_array(I)
                data[i] = I

            datagen_train = ImageDataGenerator(
                featurewise_center=True, # set input mean to 0 over the dataset
                samplewise_center=True, # set each sample mean to 0
                featurewise_std_normalization=True, # divide inputs by std of the dataset
                samplewise_std_normalization=True, # divide each input by its std
                zca_whitening=False, # apply ZCA whitening
                rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
                horizontal_flip=True, # randomly flip images
                vertical_flip=False,
                rescale=1./255,
                fill_mode='nearest') # randomly flip images

            datagen_test = ImageDataGenerator(
                featurewise_center=True, # set input mean to 0 over the dataset
                samplewise_center=False, # set each sample mean to 0
                featurewise_std_normalization=True, # divide inputs by std of the dataset
                samplewise_std_normalization=False, # divide each input by its std
                rescale=1./255) # randomly flip images

            datagen_train.fit(data)

            datagen_test.fit(data)

            del data

            return datagen_train, datagen_test

    def read_for_training(self, img):
        I = read_raw_image(img)
        I = img_to_array(I)
        I = I.reshape((1, ) + self.img_shape)
        I = self.datagen_train.flow(I, batch_size=1)
        return I[0][0]

    def read_for_testing(self, img):
        I =read_raw_image(img).convert('L')
        I = img_to_array(I)
        I = I.reshape((1, ) + self.img_shape)
        I = self.datagen_test.flow(I, batch_size=1)
        return I[0][0]
    
