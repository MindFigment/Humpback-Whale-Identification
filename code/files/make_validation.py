from globals import *
from os.path import isfile
from pandas import read_csv
from utils import load_pickle_file, save_to_pickle, expand_path
from tqdm import tqdm
from PIL import Image
import random
import numpy as np

def form_train_and_validation_data():

    train_data = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
    if isfile(whale_to_imgs + 'aaa'):
        w2imgs = load_pickle_file(whale_to_imgs)
    else:
        print('lets go!')
        w2imgs = {}
        for img, whale in tqdm(train_data.items()):
            if whale not in w2imgs:
                w2imgs[whale] = []
            if img not in w2imgs[whale]:
                w2imgs[whale].append(img)

        train_examples = []
        validation_examples = []
        lonely = []
        new_whale = []
        val_match = []
        lonely_count = 2073

        # lonely whales count = 2073
        # new whales count = 9664
        # couple whales count = 1285
        # aditional matching whales count = 2073 - 1285 = 788
        # lonely_count = 0
        # new_whale_count = 0
        # couple_count = 0
        val_known = []
        val_submit = []
        y_true = []
        matching_count = 0
        for whale, imgs in tqdm(w2imgs.items()):
            if whale == 'new_whale':
                new_whale += imgs
            elif len(imgs) == 1:
                lonely += imgs
                val_known += imgs    
            elif len(imgs) == 2:
                val_match.append((imgs[0], imgs[1], 1))
                val_submit.append(imgs[0])
                val_known.append(imgs[1])
                y_true.append(whale)
            elif len(imgs) >=4 and matching_count < 788:
                val_match.append((imgs[0], imgs[1], 1))
                val_known.append(imgs[0])
                y_true.append(whale)
                val_submit.append(imgs[1])
                matching_count += 1
                train_examples += imgs[2:]
            else:
                train_examples += imgs

        # print('lonely whales count: ', lonely_count)
        # print('new whales whales count: ', new_whale_count)
        # print('couple whales count: ', couple_count)
        # print('more then 20: ', more_then_20)
        random.shuffle(lonely)
        val_unmatch = list(zip(lonely, np.random.choice(new_whale, size=lonely_count, replace=False), np.zeros(lonely_count, dtype=np.int8)))
        validation_examples = val_match + val_unmatch
        random.shuffle(validation_examples)
        random.shuffle(train_examples)
        print('TRAIN')
        print(train_examples[:5])
        print('VALIDATION')
        print(validation_examples[:10])

        print('Train size: ', len(train_examples))
        print('Validation size: ', len(validation_examples))

        print('val_known size: ', len(val_known))
        print('val_submit size: ', len(val_submit))

        save_to_pickle(val_known_file, val_known)
        save_to_pickle(val_submit_file, val_submit)
        save_to_pickle(y_true_file, y_true)

        # save_to_pickle(train_examples, train)
        # save_to_pickle(validation_examples, validation)

        # assert len(train_examples) + len(validation_examples) == len(all_examples)

        # save_to_pickle(train_examples_file, train_examples)
        # save_to_pickle(validation_examples_file, validation_examples)

        w2ts = {}
        for whale, imgs in tqdm(w2imgs.items()):
            for img in imgs:
                if img in train_examples:
                    if whale not in w2ts:
                        w2ts[whale] = []
                    if img not in w2ts[whale]:
                        w2ts[whale].append(img)
        for w, ts in w2ts.items():
            w2ts[w] = np.array(ts)
        # save_to_pickle(whale_to_training, w2ts)
        w2i = {}
        for i, w in enumerate(train_examples):
            w2i[w] = i
        # save_to_pickle(whale_to_index, w2i)

if __name__ == "__main__":
    form_train_and_validation_data()
