from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import time

from globals import *
from generators import FeatureGen, ScoreGen
from utils import load_pickle_file, save_to_pickle
from models import Model, contrastive_loss

def prepare_submission(threshold, filename, score, known, submit, model_file):
    """
    Generate kaggle submission file.
    @param threshold: threshold given to 'new_whale'
    @param filname: submission file name
    """
    image2whale = load_pickle_file(img2whale_file)

    # Create scores dir if doesn't exist
    scores_dir = callback_path + 'scores/'
    os.makedirs(scores_dir, exist_ok=True)

    # Create model_dir if doesn't exist
    model_dir = output_path + model_file.split('/')[0] + '/'
    os.makedirs(model_dir)

    new_whale = 'new_whale'

    # Prepare files paths`
    score_file = scores_dir + filename.replace('.h5', '.score')
    output_file = model_dir + filename
    
    # Code for creating submission file, 5 best scores for each whale image
    with open(score_file, 'w+') as sf:
        with open(output_file, 'wt', newline='\n') as f:
            f.write('Image,Id\n')
            for i, p in enumerate(tqdm(submit)):
                t = []
                s = set()
                a = score[i,:]
                probs = []
                for j in list(reversed(np.argsort(a))):
                    img = known[j]
                    if a[j] < threshold and new_whale not in s:
                        s.add(new_whale)
                        t.append(new_whale)
                        probs.append(a[j])
                        if len(t) == 5:
                            break
                    for w in image2whale[img]:
                        assert w != new_whale
                        if w not in s:
                            s.add(w)
                            t.append(w)
                            probs.append(a[j])
                            if len(t) == 5:
                                break
                    if len(t) == 5:
                        break
                assert len(t) == 5 and len(s) == 5
                f.write(p + ',' + ' '.join(t[:5]) + '\n')
                sf.write(p + ',' + ' '.join(map(str, probs)) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='hwi')
    parser.add_argument('--output', '-o', dest='output_filename',
                        help='output filename (.csv)',
                        default=None, type=str)
    parser.add_argument('--model', '-m', dest='model_filename',
                        help='model filename (.h5)',
                        default=None, type=str)
    parser.add_argument('--threshold', '-th', dest='threshold',
                        help='threshold for new_whale',
                        default=0.99, type=float)
    return parser.parse_args()
    
def main():
    tic = time.time()

    known = load_pickle_file(train_known_file)
    submit = load_pickle_file(train_submit_file)

    print('inference HWI')
    args = parse_args()
    model_path = models_path + args.model_filename
    #submission_path = output_path + args.output_filename
    threshold = args.threshold

    # Load model
    weights = load_model(model_path).get_weights()
    # weights =  keras.models.load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss}).get_weights()
    model = Model(0, 0, 'submission', use_val=False)
    model.model.set_weights(weights)

    # Evaluate model
    fknown = model.branch_model.predict_generator(FeatureGen(known, model.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
    fsubmit = model.branch_model.predict_generator(FeatureGen(submit, model.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
    score = model.head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=8, verbose=0)
    score = model.score_reshape(score, fknown, fsubmit)

    # Generate submission file
    prepare_submission(threshold, args.output_filename, score, known, submit, args.model_filename)
    toc = time.time()
    print("Inference time: ", (toc - tic) / 60, 'mins')

if __name__ == "__main__":
    main()
    
