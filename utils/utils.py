from PIL import Image
from os.path import isfile
import pickle

from globals import train_dir, test_dir

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

def read_raw_image(img, img_shape=(384, 384, 1), mode='L'):
    I = Image.open(expand_path(img)).convert(mode)
    I = I.resize(img_shape[:-1])
    return I
