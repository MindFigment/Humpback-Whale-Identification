from globals import *
from os.path import isfile
from pandas import read_csv
from utils import load_pickle_file, save_to_pickle, expand_path
from tqdm import tqdm
from PIL import Image
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
        w2imgs = load_pickle_file(whale_to_imgs)
    else:
        w2imgs = {}
        for img, whale in tqdm(train_data.items()):
            if whale not in w2imgs:
                w2imgs[whale] = []
            if img not in w2imgs[whale]:
                w2imgs[whale].append(img)

    if isfile(train_examples_file) and isfile(validation_examples_file):
        train = load_pickle_file(train_examples_file)
        validation = load_pickle_file(validation_examples_file)
    else:
        all_examples = []
        for whale, imgs in tqdm(w2imgs.items()):
            if len(imgs) > 1 and whale != 'new_whale':
                all_examples += imgs
        random.shuffle(all_examples)
        num_validation_samples = int(len(all_examples) / 10)
        validation = all_examples[:num_validation_samples]
        train = all_examples[num_validation_samples:]
        train = all_examples

        print('Train size: ', len(train))
        print('Validation size: ', len(validation))

        save_to_pickle(train_examples_file, train)
        save_to_pickle(validation_examples_file, validation)
        save_to_pickle(all_examples_file, all_examples)

        # assert len(train) + len(validation) == len(all_examples)

        w2ts = {}
        for whale, imgs in tqdm(w2imgs.items()):
            for img in imgs:
                if img in train:
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

    if isfile(img_to_whales):
        print('HEY')
        # Find elements from training set not 'new_whale'
        train = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
        submit = [img for (_, img, _) in read_csv(sample_csv).to_records()]

        img2ws = {}
        for img,w in train.items():
            if w != 'new_whale':
                if img not in img2ws:
                    img2ws[img] = []
                if w not in img2ws[img]:
                    img2ws[img].append(w)
        known = sorted(list(img2ws.keys()))

        save_to_pickle(img_to_whales, img2ws)
        save_to_pickle(known_file, known)
        save_to_pickle(submit_file, submit)

        # Dictionary of image indices
        w2i = {}
        for i, w in enumerate(known):
            w2i[w] = i
        save_to_pickle(whale_to_index, w2i)

if __name__ == "__main__":
    make_dicts()
