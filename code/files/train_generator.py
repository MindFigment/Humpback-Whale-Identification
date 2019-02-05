from keras.utils import Sequence
from scipy.optimize import linear_sum_assignment
from lap import lapjv
from globals import whale_to_training, whale_to_index
from keras import backend as K
from utils import read_raw_image
from utils import load_pickle_file, save_to_pickle
import numpy as np
import random

class TrainingData(Sequence):
    def __init__(self, score, train, steps=1000, batch_size=32):
        """
        @param score: cost matrix for the picture matching
        @param steps: number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score
        self.train = train
        self.steps = steps
        self.batch_size = batch_size
        self.img_shape = (384,384,1)
        self.w2ts = load_pickle_file(whale_to_training)
        self.w2i = load_pickle_file(whale_to_index)
        self.match = []
        self.unmatch = []
        for ts in self.w2ts.values():
            idxs = [self.w2i[w] for w in ts]
            for i in idxs:
                for j in idxs:
                    self.score[i, j] = 10000.0
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size, ) + self.img_shape, dtype=K.floatx())
        b = np.zeros((size, ) + self.img_shape, dtype=K.floatx())
        c = np.zeros((size,1), dtype=K.floatx())
        j = start // 2

        for i in range(0, size, 2):
            a[i,:,:,:] = read_raw_image(self.match[j][0])
            b[i,:,:,:] = read_raw_image(self.match[j][1])
            c[i,0] = 1
            a[i+1,:,:,:] = read_raw_image(self.unmatch[j][0])
            b[i+1,:,:,:] = read_raw_image(self.unmatch[j][1])
            c[i+1,0] = 0
            j += 1
        return [a,b], c

    def on_epoch_end(self):
        if self.steps <= 0:
            return

        self.steps -= 1
        self.match = []
        self.unmatch = []

        _,_,x = lapjv(self.score)

        y = np.arange(len(x), dtype=np.int32)

        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d):
                    break
            for ab in zip(ts, d):
                self.match.append(ab)

        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(self.unmatch), len(self.train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size
