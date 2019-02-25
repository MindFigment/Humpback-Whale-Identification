from keras.utils import Sequence
import numpy as np
from keras import backend as K
from utils import read_raw_image
from tqdm import tqdm

# A keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only  the upper triangular matrix of the cost matrix if y is None
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        """
        @param x: TODO
        @param y: TODO
        @param verbose: 1 for displaying progress bar, 0 for no information
        """
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
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='scores')

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate start index and number of batch samples
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))

        # Generate data
        a, b = self.__data_generation(start, end)

        # Update progress bar
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()
        return [a, b]


    def __data_generation(self, start, end):
        'Generates data containing (end - start) samples'
        # Generate data
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]

        return a, b

