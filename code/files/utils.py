from PIL import Image
from os.path import isfile
from pandas import read_csv
import numpy as np
from math import sqrt
from keras.preprocessing.image import img_to_array,array_to_img
import random
import pickle
from globals import *
from keras import backend as K

def expand_path(img):
    if isfile(train_dir + img):
         return train_dir + img
    if isfile(test_dir + img):
        return test_dir + img
    raise ValueError('No such image: {} in dataset'.format(img))

def load_pickle_file(file):
    with open (file, 'rb') as f:
        pickle_file = pickle.load(f)
        return pickle_file

def save_to_pickle(path, object):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def make_dicts():

    if isfile(joined_data_file):
        joined_data = load_pickle_file(joined_data_file)
    else:
        train_data = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
        test_data = [img for (_, img, _) in read_csv(sample_csv).to_records()]
        joined_data = list(train_data.keys()) + test_data
        save_to_pickle(joined_data_file, joined_data)

    # Load img2size dictionary if exists, or create it otherwise
    if isfile(img_to_size):
        img2size = load_pickle_file(img_to_size)
    else:
        img2size = {}
        for img in joined_data:
            size = Image.open(expand_path(img)).size
            img2size[img] = size
        assert len(img2size) == len(joined_data)
        save_to_pickle(img_to_size, img2size)

    train_data = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
    if isfile(whale_to_imgs):
        w2imgs = load_pickle_file(whale_to_imgs)
    else:
        w2imgs = {}
        for img, whale in train_data.items():
            if whale not in w2imgs:
                w2imgs[whale] = []
            if img not in w2imgs[whale]:
                w2imgs[whale].append(img)

    if isfile(train_examples_file):
        train = load_pickle_file(train_examples_file)
    else:
        train = []
        for whale, imgs in w2imgs.items():
            if len(imgs) > 1 and whale != 'new_whale':
                train += imgs
        random.shuffle(train)
        train_set = set(train)
        w2ts = {}
        for whale, imgs in w2imgs.items():
            for img in imgs:
                if img in train_set:
                    if whale not in w2ts:
                        w2ts[whale] = []
                    if img not in w2ts[whale]:
                        w2ts[whale].append(img)
        for w, ts in w2ts.items():
            w2ts[w] = np.array(ts)
        save_to_pickle(whale_to_training, w2ts)
        w2i = {}
        for i, w in enumerate(train):
            w2i[w] = i
        save_to_pickle(whale_to_index, w2i)



def read_raw_image(image):
    I = Image.open(expand_path(image)).convert('L')
    # I = I.convert('L') # convert image to grayscale
    I = I.resize((384,384))
    # I = img_to_array(I)
    # I -= np.mean(I, keepdims=True)
    # I /= np.std(I, keepdims=True) + K.epsilon()
    return I
