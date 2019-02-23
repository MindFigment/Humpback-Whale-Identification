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
    def __init__(self, lr, l2, model_name, histories = []):
        self.model, self.branch_model, self.head_model = build_model(lr, l2)
        self.histories = histories
        self.step = 0
        self.img_shape = (384, 384, 1)
        self.img_gen = ImageGenerator()
        self.best_map5 = 0
        self.model_name = model_name
        os.makedirs(tensorboard_dir + model_name, exist_ok=True)

    def set_lr(self, lr):
        K.set_value(self.model.optimizer.lr, float(lr))

    def get_lr(self):
        return K.get_value(self.model.optimizer.lr)

    # def set_l2(self, l2):
    #     K.set_value(self.model.regularizer.l2, float(l2))

    # def get_l2(self):
    #     return K.get_value(self.model.regularizer.l2)

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

        # Load train
        train = load_pickle_file(train_examples_file)
        validation = load_pickle_file(validation_examples_file)

        # print('train len:', len(train))
        # print('validation len: ', len(validation))

        # shuffle training images
        random.shuffle(train)

        # Load whales to hashes dict
        #w2ts = load_pickle_file(whale_to_training)

        # Map training image hash value to index n in 'train' array
        w2i = {}
        for i, w in enumerate(train):
            w2i[w] = i
        save_to_pickle(whale_to_index, w2i)

        # Compute the match score for each image pair
        _, score = self.compute_score(train)
        # _, score_val = self.compute_score(validation)
        # save_to_pickle(score_file, score)
        # save_to_pickle(features_file, features)
        # score = load_pickle_file(score_file) 
        # features = load_pickle_file(features_file)

        callbacks_list = [
            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath=models_path + self.model_name + '_' + str(self.step) + '.h5', monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
            keras.callbacks.CSVLogger(filename=callback_path + self.model_name + '.log', append=True),
            keras.callbacks.TensorBoard(log_dir=tensorboard_dir + self.model_name)
        ]

        # Train the model for 'step' epochs
        history = self.model.fit_generator(
            TrainingData(score + ampl * np.random.random_sample(size=score.shape), train, self.img_gen.read_for_training, steps=steps, batch_size=16),
            validation_data = ValData(validation, self.img_gen.read_for_testing, batch_size=16),
            initial_epoch=self.step, epochs=self.step + steps, max_queue_size=12, workers=8, verbose=1, callbacks=callbacks_list).history
        self.step += steps

        map_5 = self.val_score()

        if map_5 >= self.best_map5:
            self.best_map5 = map_5
            self.model.save(models_path + self.model_name + str(self.step) + '_' + str(map_5) + '.h5')

        #Collect history data
        history['epochs'] = self.step
        history['ms'] = np.mean(score)
        history['lr'] = self.get_lr()
        # history['l2'] = self.get_l2()
        history['map5'] = map_5
        print('epochs: ', history['epochs'], 'lr:', history['lr'], 'l2: ', history['lr'], 'ms: ', history['ms'], 'MAP@5 --> ', history['map5'])
        self.histories.append(history)

    def val_score(self):
        val_known = load_pickle_file(val_known_file)
        val_submit = load_pickle_file(val_submit_file)
        y_true = load_pickle_file(y_true_file)
        fknown = self.branch_model.predict_generator(FeatureGen(val_known, self.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
        fsubmit = self.branch_model.predict_generator(FeatureGen(val_submit, self.img_gen.read_for_testing), max_queue_size=20, workers=8, verbose=0)
        score = self.head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=8, verbose=0)
        score = self.score_reshape(score, fknown, fsubmit)

        img2ws = load_pickle_file(img_to_whales)

        best_5 = []
        for i, _ in enumerate(tqdm(val_submit)):
            t = []
            s = set()
            a = score[i,:]
            for j in list(np.argsort(a)):
                img = val_known[j]
                for w in img2ws[img]:
                    if w not in s:
                        s.add(w)
                        t.append(w)
                        if len(t) == 5:
                            break
                if len(t) == 5:
                    break
            assert len(t) == 5 and len(s) == 5
            best_5.append(t)

        map_5 = self.map5(best_5, y_true)
        return map_5

    def map5(self, best_5, y_true):
        epsilon = 1e-06
        average_precision = epsilon
        # print('best5 len: ', len(best_5))
        # print('y_true len: ', len(y_true))
        assert len(best_5) == len(y_true)
        U = len(best_5)
        for i in range(U):
            for j, w in enumerate(best_5[i]):
                if w == y_true[i]:
                    average_precision += 1 / (j + 1)
                    break
        mean_average_precision = average_precision / U
        return mean_average_precision
