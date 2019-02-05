from keras import backend as K
import numpy as np
import random
from models_file import build_model
from globals import whale_to_imgs, whale_to_index, whale_to_training, score_file, features_file, train_examples
from pandas import read_csv
from globals import train_csv
from utils import load_pickle_file, save_to_pickle
from feature_generator import FeatureGen
from score_generator import ScoreGen
from train_generator import TrainingData
from utils import save_to_pickle
# import tensorflow as tf  
# from keras.backend.tensorflow_backend import set_session  
# config = tf.ConfigProto()  
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
# config.log_device_placement = False  # to log device placement (on which device the operation ran)  
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)  
# set_session(sess)  # set this TensorFlow session as the default session for Keras  


class Model():
    def __init__(self, lr, l2, histories = []):
        self.model, self.branch_model, self.head_model = build_model(lr, l2)
        self.histories = histories
        self.step = 0

    def set_lr(self, lr):
        K.set_value(self.model.optimizer.lr, float(lr))

    def get_lr(self):
        return K.get_value(self.model.optimizer.lr)

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
        features = self.branch_model.predict_generator(FeatureGen(self.train, verbose=verbose), max_queue_size=12, workers=3, verbose=0)
        score = self.head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=3, verbose=0)
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
        self.train = load_pickle_file(train_examples)

        # shuffle training images
        random.shuffle(self.train)

        # Load whales to hashes dict
        #w2ts = load_pickle_file(whale_to_training)

        # Map training image hash value to index n in 'train' array
        w2i = {}
        for i, w in enumerate(self.train):
            w2i[w] = i
        save_to_pickle(whale_to_index, w2i)

        # Compute the match score for each image pair
        _, score = self.compute_score()
        # save_to_pickle(score_file, score)
        # save_to_pickle(features_file, features)
        # score = load_pickle_file(score_file) 
        # features = load_pickle_file(features_file)

        # Train the model for 'step' epochs
        history = self.model.fit_generator(
            TrainingData(score + ampl * np.random.random_sample(size=score.shape), self.train, steps=steps, batch_size=32),
            initial_epoch=self.step, epochs=self.step + steps, max_queue_size=12, workers=6, verbose=1).history
        self.step += steps

        # history = self.model.fit_generator(
        #     TrainingData(np.random.random_sample(size=(len(self.train), len(self.train))), self.train, steps=steps, batch_size=32),
        #     initial_epoch=self.step, epochs=self.step + steps, max_queue_size=12, workers=1, verbose=1).history

        #Collect history data
        history['epochs'] = self.step
        history['ms'] = np.mean(score)
        history['lr'] = self.get_lr()
        print(history['epochs'], history['lr'], history['ms'])
        self.histories.append(history)
