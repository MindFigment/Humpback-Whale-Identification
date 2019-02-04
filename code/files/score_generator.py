from keras.utils import Sequence
import numpy as np
from keras import backend as K
from utils import read_raw_image
# A keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only  the upper triangular matrix of the cost matrix if y is None
class ScoreGen(Sequence):

    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size))
            self.iy = self.iy.reshape((self.iy.size))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            print('verbose')
            # self.progress = tqdm_notebook(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            print('verbose')
            # self.progress.update()
            # if self.progress.n >= len(self):
            #     self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size
