from keras.utils import Sequence
from scipy.optimize import linear_sum_assignment
from keras import backend as K
import numpy as np
import random
import time
from lapjv import lapjv 

from globals import callback_path
from utils import load_pickle_file, save_to_pickle, read_raw_image

class TrainingData(Sequence):
    def __init__(self, score, train, img_gen, w2ts, w2i, steps=1000, batch_size=32, img_shape=(384, 384, 1)):
        """
        @param score: cost matrix for the picture matching
        @param steps: number of epoch we are planning with this score matrix
        @param w2ts: whale to training examples
        @param w2i: whale to index in the training dataset
        """
        super(TrainingData, self).__init__()
        self.score = -score # change to score if want to use contrastive loss instead of binary crossentropy
        self.train = train
        self.steps = steps
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.w2ts = w2ts
        self.w2i = w2i
        self.match = []
        self.unmatch = []
        self.img_gen = img_gen
        for ts in self.w2ts.values():
            idxs = [self.w2i[w] for w in ts]
            for i in idxs:
                for j in idxs:
                    self.score[i, j] = 10000.0
        self.on_epoch_end()

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        
        # Generate data
        [a, b], c = self.__data_generation(start, size)

        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0:
            return
        self.steps -= 1
        self.match = []
        self.unmatch = []

        start_time = time.time()
        _,x,_ = lapjv(self.score)
        end_time = time.time()
        exec_time = end_time - start_time

        with open(callback_path + 'lapjv.txt', 'w+') as f:
            f.write(str(self.steps) + ' ' + str(exec_time) + '\n')
     
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
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)

    def __data_generation(self, start, size):
        'Generates data containing (end - start) samples'
        # Generate data
        a = np.empty((size, ) + self.img_shape, dtype=K.floatx())
        b = np.empty((size, ) + self.img_shape, dtype=K.floatx())
        c = np.empty((size,1), dtype=K.floatx())
        j = start // 2

        for i in range(0, size, 2):
            a[i,:,:,:] = self.img_gen(self.match[j][0])
            b[i,:,:,:] = self.img_gen(self.match[j][1])
            c[i,0] = 1
            a[i+1,:,:,:] = self.img_gen(self.unmatch[j][0])
            b[i+1,:,:,:] = self.img_gen(self.unmatch[j][1])
            c[i+1,0] = 0
            j += 1

        return [a, b], c


