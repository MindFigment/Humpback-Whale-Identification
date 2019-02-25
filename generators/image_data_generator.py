from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import numpy as np
from pandas import read_csv
from tqdm import tqdm
from PIL import Image

from globals import train_csv
from utils import expand_path, read_raw_image


class ImageGenerator():
    def __init__(self, img_shape=(384, 384, 1), batch_size=1, mode='L'):
        """
        @param mode: type of images used, 'L' for grayscale, 'RGB' for color
        """
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.mode = mode
        self.datagen_train, self.datagen_test = self.__create_image_generators()

    def __create_image_generators(self):
        'Creation of training & test image generators'
        datagen_train = ImageDataGenerator(
            zca_whitening=False, # apply ZCA whitening
            rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)
            horizontal_flip=True, # randomly flip images
            vertical_flip=False,
            zoom_range=[1.0, 1.2],
            rescale=1./255,
            shear_range=0.2,
            fill_mode='nearest')

        datagen_test = ImageDataGenerator(
            rescale=1./255)

        # Fit generator on sample data
        # data = self.__get_data(train_csv)
        # datagen_train.fit(data)
        # datagen_test.fit(data)
        # del data

        return datagen_train, datagen_test

    def __get_data(self, data_file):
        # Data for fitting generator
        data = [img for _,img,_ in read_csv(data_file).to_records()]
        size = len(data)
        data = np.zeros((size, ) + self.img_shape, dtype=K.floatx()) 
        for i in tqdm(range(size)):
            I = Image.open(expand_path(data[i]))
            I = I.convert('L') # convert image to grayscale
            I = I.resize(self.img_shape[1:])
            I = img_to_array(I)
            data[i] = I

    def read_for_training(self, img):
        I = read_raw_image(img, self.img_shape, self.mode)
        I = img_to_array(I)
        I = I.reshape((1, ) + self.img_shape)
        I = self.datagen_train.flow(I, batch_size=self.batch_size)
        img = I[0][0]
        img  -= np.mean(img, keepdims=True)
        img  /= np.std(img, keepdims=True) + K.epsilon()
        return img

    def read_for_testing(self, img):
        I =  read_raw_image(img, self.img_shape, self.mode)
        I = img_to_array(I)
        I = I.reshape((1, ) + self.img_shape)
        I = self.datagen_test.flow(I, batch_size=self.batch_size)
        img = I[0][0]
        img  -= np.mean(img, keepdims=True)
        img  /= np.std(img, keepdims=True) + K.epsilon()
        return img

    
    
