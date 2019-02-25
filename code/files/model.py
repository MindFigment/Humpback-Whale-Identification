from keras import backend as K
import numpy as np
import random
from models_file import build_model
from globals import *
from pandas import read_csv
from globals import train_csv
from utils import load_pickle_file, save_to_pickle
from feature_generator import FeatureGen
from score_generator import ScoreGen
from train_generator import TrainingData
from val_generator import ValData
from utils import save_to_pickle
import keras
from tqdm import tqdm
from image_data_generator import ImageGenerator
import os

class Model():
    """
    Class for representing model we are going to train
    """
    def __init__(self, lr, l2, model_name, histories = [], img_shape=(384, 384, 1), step=0, use_val=True, small_dataset=False):
        self.model, self.branch_model, self.head_model = build_model(lr, l2)
        self.histories = histories
        self.step = step
        self.img_shape = img_shape
        self.img_gen = ImageGenerator()
        self.best_map5 = 0
        self.model_name = model_name
        self.w2ts = load_pickle_file(whale2training_file)
        # Make callbacklist
        self.callbacklist = self.make_callback_list()
        # Load train
        if small_dataset:
            self.train = load_pickle_file(train_examples_small_file)
        else:
            self.train = load_pickle_file(train_examples_file)
        if small_dataset:
            validation_data = load_pickle_file(validation_examples_small_file)
        else:
            validation_data = load_pickle_file(validation_examples_file)
        if use_val:
            self.validation = ValData(validation_data, self.img_gen.read_for_testing, batch_size=16)
        else:
            self.validation = None

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

    def compute_score(self, data, verbose=1):
        """
        Compute the score matrix by scoring every image from the training set against every other image O(n^2)
        """
        features = self.branch_model.predict_generator(FeatureGen(data, self.img_gen.read_for_testing, verbose=verbose), max_queue_size=12, workers=8, verbose=0)
        score = self.head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=8, verbose=0)
        score = self.score_reshape(score, features)
        return features, score

    def make_steps(self,steps, ampl):
        """
        Perform training epochs
        @param step: which epoch we are starting from
        @param steps: number of epochs to perform
        @param ampl: K, the randomized component of the score matrix
        """

        # shuffle training images
        random.shuffle(self.train)

        # Map training images to index n in 'train' array TODO change w2i to img2index
        w2i = {}
        for i, w in enumerate(self.train):
            w2i[w] = i, 

        # Compute the match score for each image pair
        _, score = self.compute_score(self.train)

        # Train the model for 'step' epochs
        history = self.model.fit_generator(
            TrainingData(score + ampl * np.random.random_sample(size=score.shape), self.train, self.img_gen.read_for_training, steps=steps, batch_size=16, w2ts=self.w2ts, w2i=w2i),
            validation_data = self.validation, initial_epoch=self.step, epochs=self.step + steps, max_queue_size=12, workers=8, verbose=1, callbacks=self.callbacks_list).history
        self.step += steps

        # Compute MAP@5 for validation data
        map_5 = self.val_score()

        # Save model if MAP@5 score has improved
        if map_5 >= self.best_map5:
            self.best_map5 = map_5
            self.model.save(models_path + self.model_name + str(self.step) + '_' + str(map_5) + '.h5')

        #Collect history data
        history['epochs'] = self.step
        history['ms'] = np.mean(score)
        history['lr'] = self.get_lr()
        history['map5'] = map_5
        print('epochs: ', history['epochs'], 'lr:', history['lr'], 'ms: ', history['ms'], 'MAP@5 --> ', history['map5'])
        self.histories.append(history)

    def val_score(self):
        """
        Compute MAP@5 score for validation dataset
        """
        val_known = load_pickle_file(val_known_file)
        tmp = load_pickle_file(val_submit_file)
        val_submit = tmp[:, 0]
        y_true = tmp[:, 1]
        del tmp

        fknown = self.branch_model.predict_generator(FeatureGen(val_known, self.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
        fsubmit = self.branch_model.predict_generator(FeatureGen(val_submit, self.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
        score = self.head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=8, verbose=0)
        score = self.score_reshape(score, fknown, fsubmit)

        img2whale = load_pickle_file(img2whale_file)

        best_5 = []
        for i, _ in enumerate(tqdm(val_submit)):
            t = []
            s = set()
            a = score[i,:]
            for j in list(reversed(np.argsort(a))):
                img = val_known[j]
                whale = img2whale[img]
                if whale not in s:
                    s.add(whale)
                    t.append(whale)
                if len(t) == 5:
                    break
            assert len(t) == 5 and len(s) == 5
            best_5.append(t)

        map_5 = self.map5(best_5, y_true)
        return map_5

    def map5(self, best_5, y_true):
        """
        Function for computing MAP@5 given best 5 scoers for each whale image
        """
        epsilon = 1e-06
        average_precision = epsilon
        assert len(best_5) == len(y_true)
        U = len(best_5)
        for i in range(U):
            for j, w in enumerate(best_5[i]):
                if w == y_true[i]:
                    average_precision += 1 / (j + 1)
                    break
        mean_average_precision = average_precision / U

        return mean_average_precision

    def make_callback_list(self):
        checkpoint_path = models_path + self.model_name + '/' + str(self.step) + '.h5' 
        csv_logger_path = callback_path + self.model_name + '.log'
        tensorboard_path = tensorboard_dir + self.model_name
        # Make dir for tensorboard graphs
        os.makedirs(callback_path, exist_ok=True)
        os.makedirs(tensorboard_dir + self.model_name, exist_ok=True)
        os.makedirs(models_path + self.model_name, exist_ok=True)
        callbacks_list = [
            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
            keras.callbacks.CSVLogger(filename=csv_logger_path, append=True),
            keras.callbacks.TensorBoard(log_dir=tensorboard_path)
        ]

        return callbacks_list