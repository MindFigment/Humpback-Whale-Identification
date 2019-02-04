from globals import *
from os.path import isfile
from pandas import read_csv
from utils import load_pickle_file, save_to_pickle, expand_path
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import random
import numpy as np

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
        for img in tqdm(joined_data):
            size = Image.open(expand_path(img)).size
            img2size[img] = size
        assert len(img2size) == len(joined_data)
        save_to_pickle(img_to_size, img2size)

    train_data = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
    if isfile(whale_to_imgs):
        w2imgs = load_pickle_file(whales_to_imgs)
    else:
        w2imgs = {}
        for img, whale in tqdm(train_data.items()):
            if whale not in w2imgs:
                w2imgs[whale] = []
            if img not in w2imgs[whale]:
                w2imgs[whale].append(img)

    if isfile(train_examples):
        train = load_pickle_file(train_examples)
    else:
        train = []
        for whale, imgs in tqdm(w2imgs.items()):
            if len(imgs) > 1 and whale != 'new_whale':
                train += imgs
        random.shuffle(train)
        train_set = set(train)
        w2ts = {}
        for whale, imgs in tqdm(w2imgs.items()):
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

if __name__ == "__main__":
    print("Executing as main program")
    print("Value of __name__ is: ", __name__)
    make_dicts()
