from tqdm import tqdm
import numpy as np
import pandas as pd
from globals import *
from feature_generator import FeatureGen
from score_generator import ScoreGen
from keras.models import load_model
from utils import load_pickle_file, save_to_pickle
import time
from model import Model

def prepare_submission(threshold, filename, score, known, submit):
    """
    Generate kaggle submission file.
    @param threshold: threshold given to 'new_whale'
    @param filname: submission file name
    """
    img2ws = load_pickle_file(img_to_whales)

    new_whale = 'new_whale'
    with open(callback_path + 'score.txt', 'a+') as sf:
        with open(filename, 'wt', newline='\n') as f:
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
                    for w in img2ws[img]:
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

if __name__ == "__main__":
    
    tic = time.time()

    known = load_pickle_file(known_file)
    submit = load_pickle_file(submit_file)

    weights = load_model(my_model_file).get_weights()
    model = Model(0, 0)
    model.model.set_weights(weights)
    # Evaluate the model
    fknown = model.branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=8, verbose=0)
    fsubmit = model.branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=8, verbose=0)
    score = model.head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=8, verbose=0)
    score = model.score_reshape(score, fknown, fsubmit)

    save_to_pickle('score.pickle', score)
    # score = load_pickle_file('score.pickle')

    # Generate the submission file
    prepare_submission(0.99, output_path + 'second_submission_99.csv', score, known, submit)
    toc = time.time()
    print("Submission time: ", (toc - tic) / 60.)
