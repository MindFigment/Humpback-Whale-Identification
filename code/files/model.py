from keras import backend as K
import numpy as np
import random
from models_file import build_model
from globals import whale_to_imgs, whale_to_index, whale_to_training
from pandas import read_csv
from globals import train_csv
from utils import load_pickle_file, save_to_pickle
from feature_generator import FeatureGen
from score_generator import ScoreGen
from train_generator import TrainingData


class Model():
    def __init__(self, lr, l2, histories = []):
        self.model, self.branch_model, self.head_model = build_model(lr, l2)
        self.histories = histories
        self.step = 0

    def set_lr(lr):
        K.set_value(model.optimizer.lr, float(lr))

    def get_lr():
        return K.get_value(model.optimizer.lr)

    def score_reshape(self, score, x, y=None):
        """
        Tranformed the packed matrix 'score' into a square matrix.
        @param score the packed matrix
        @param x the first image feature tensor
        @param y the second image feature tensor if different from x
        @result the square matrix
        """
        if y is None:
            # When y is None score is a packed upper triangular matrix.
            # Unpack and transpose  to form the symmetrical lower triangular matrix.
            m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
            m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
            m += m.transpose()
        else:
            m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
            iy, ix = np.indices((y.shape[0], x.shape[0]))
            ix = ix.reshape((ix.size))
            iy = iy.reshape((iy.size))
            m[iy, ix] = score.squeeze()
        return m

    def compute_score(self, verbose=1):
        """
        Compute the score matrix by scoring every image from the training set against every other image O(n^2)
        """
        features = self.branch_model.predict_generator(FeatureGen(self.train, verbose=verbose), max_queue_size=12, workers=1, verbose=0)
        score = self.head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=1, verbose=0)
        score = self.score_reshape(score, features)
        return features, score

    def make_steps(self,steps, ampl):
        """
        Perform training epochs
        @param step: which epoch we are starting from
        @param steps: number of epochs to perform
        @param ampl: K, the randomized component of the score matrix
        """

        # Load train
        self.train = [img for (_, img, _) in read_csv(train_csv).to_records()]

        # shuffle training images
        random.shuffle(self.train)

        # Load whales to hashes dict
        w2ts = load_pickle_file(whale_to_training)

        # Map training image hash value to index n in 'train' array
        w2i = {}
        for i, w in enumerate(self.train):
            w2i[w] = i
        save_to_pickle(whale_to_index, w2i)

        # Compute the match score for each image pair
        features, score = self.compute_score()

        # Train the model for 'step' epochs
        history = model.fit_generator(
            TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=32),
            initial_epoch=step, epochs=step + steps, max_queue_size=12, workers=1, verbose=0)
        self.step += steps

        #Collect history data
        history['epochs'] = step
        history['ms'] = np.mean(score)
        history['lr'] = get_lr(model)
        print(history['epochs'], history['lr'], history['ms'])
        histories.append(history)
