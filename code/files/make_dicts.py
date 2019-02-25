from globals import *
from os.path import isfile
import os
from pandas import read_csv
from utils import load_pickle_file, save_to_pickle, expand_path
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import argparse

def make_dicts(reset_all, make_val):

    # Create meta directory if doesn't already exist for all dictionaries generated below
    os.makedirs(meta_dir, exist_ok=True)

    train_data = dict([(img, whale) for (_, img, whale) in read_csv(train_csv).to_records()])
    test_data = [img for (_, img, _) in read_csv(sample_csv).to_records()]

    # Load whale_to_imgs dictionary if exists, or create it otherwise
    if isfile(whale2imgs_file or not reset_all):
        whale2imgs = load_pickle_file(whale2imgs_file)
    else:
        whale2imgs = {}
        for img, whale in tqdm(train_data.items()):
            if whale not in whale2imgs:
                whale2imgs[whale] = []
            if img not in whale2imgs[whale]:
                whale2imgs[whale].append(img)

        save_to_pickle(whale2imgs_file, whale2imgs)

    if not isfile(img2whale_file) or reset_all:
        # Find elements from training set other then 'new_whale'
        img2whale = {}
        for img, whale in train_data.items():
            if whale != 'new_whale':
                if img not in img2whale:
                    img2whale[img] = whale
        train_known = sorted(list(img2whale.keys()))

        save_to_pickle(img2whale_file, img2whale)
        save_to_pickle(train_known_file, train_known)
        save_to_pickle(train_submit_file, test_data)

    if not (isfile(train_examples_file) and isfile(validation_examples_file) and reset_all == False):
        train_examples = []
        validation_examples = []
        lonely = []
        new_whale = []
        val_match = []
        lonely_count = len([ x for x in whale2imgs.values() if len(x) == 1 ]) # 2073
        couple_count = len([ x for x in whale2imgs.values() if len(x) == 2 ]) # 1285
        new_count = len([ x for x in train_data.values() if x == 'new_whale' ]) # 9664 
        # aditional matching whales count needed or creating balanced validation dataset (same matching and unmatching number of examples)
        extra_count = lonely_count - couple_count # 2073 - 1285 = 788

        val_known = []
        val_submit = []
        matching_count = 0

        if make_val:
            for whale, imgs in tqdm(whale2imgs.items()):
                if whale == 'new_whale':
                    new_whale += imgs
                elif len(imgs) == 1:
                    lonely += imgs
                    val_known += imgs    
                elif len(imgs) == 2:
                    val_match.append((imgs[0], imgs[1], 1))
                    val_known.append(imgs[1])
                    val_submit.append([imgs[0], whale])
                elif len(imgs) >=4 and matching_count < extra_count:
                    val_match.append((imgs[0], imgs[1], 1))
                    val_known.append(imgs[0])
                    val_submit.append([imgs[1], whale])
                    matching_count += 1
                    train_examples += imgs[2:]
                else:
                    train_examples += imgs
        else:
            for whale, imgs in tqdm(whale2imgs.items()):
                if whale == 'new_whale':
                    new_whale += imgs
                elif len(imgs) == 1:
                    lonely += imgs
                    val_known += imgs    
                elif len(imgs) == 2:
                    val_match.append((imgs[0], imgs[1], 1))
                    val_known.append(imgs[1])
                    val_submit.append([imgs[0], whale])
                    train_examples += imgs
                elif len(imgs) >=4 and matching_count < extra_count:
                    val_match.append((imgs[0], imgs[1], 1))
                    val_known.append(imgs[0])
                    val_submit.append([imgs[1], whale])
                    matching_count += 1
                    train_examples += imgs
                else:
                    train_examples += imgs

        print('lonely whales count: ', lonely_count)
        print('new whales count: ', new_count)
        print('couple whales count: ', couple_count)
        print('extra whales count: ', extra_count)

        random.shuffle(lonely)
        val_unmatch = list(zip(lonely, np.random.choice(new_whale, size=lonely_count, replace=False), np.zeros(lonely_count, dtype=np.int8)))
        validation_examples = val_match + val_unmatch
        random.shuffle(validation_examples)
        random.shuffle(train_examples)

        small_train_size = len(train_examples) // 10
        small_validation_size = len(validation_examples) // 10
        small_train_examples = train_examples[:small_train_size]
        small_validation_examples = validation_examples[:small_validation_size]

        # print('TRAIN')
        # print(train_examples[:10])
        # print('VALIDATION')
        # print(validation_examples[:10])

        print('Train size: ', len(train_examples))
        print('Validation size: ', len(validation_examples))

        print('val_known size: ', len(val_known))
        print('val_submit size: ', len(val_submit))

        save_to_pickle(train_examples_file, train_examples)
        save_to_pickle(validation_examples_file, validation_examples)

        save_to_pickle(train_examples_small_file, small_train_examples)
        save_to_pickle(validation_examples_small_file, small_validation_examples)

        save_to_pickle(val_known_file, val_known)
        save_to_pickle(val_submit_file, val_submit)

        w2ts = {}
        for whale, imgs in tqdm(whale2imgs.items()):
            for img in imgs:
                if img in train_examples:
                    if whale not in w2ts:
                        w2ts[whale] = []
                    if img not in w2ts[whale]:
                        w2ts[whale].append(img)
        for w, ts in w2ts.items():
            w2ts[w] = np.array(ts)

        save_to_pickle(whale2training_file, w2ts)


def parse_args():
    parser = argparse.ArgumentParser(description='hwi')

    parser.add_argument('--reset', dest='reset_all', action='store_true')
    parser.add_argument('--no-reset', dest='reset_all', action='store_false')
    parser.set_defaults(reset_all=False)

    parser.add_argument('--val', dest='make_val', action='store_true')
    parser.add_argument('--no-val', dest='make_val', action='store_false')
    parser.set_defaults(make_val=True)

    return parser.parse_args()

def main():
    args = parse_args()
    print(args.reset_all)
    print(args.make_val)
    make_dicts(args.reset_all, args.make_val)

if __name__ == "__main__":
    main()
