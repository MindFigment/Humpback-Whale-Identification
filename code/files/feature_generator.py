from keras.utils import Sequence
import numpy as np
from keras import backend as K
from utils import read_raw_image
# A keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):

    def __init__(self, data, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.img_shape = (384, 384, 1)
        if self.verbose > 0:
            print('verbose')
            # self.progress = tqdm_notebook(total=len(self), desc="Features")

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size, ) + self.img_shape, dtype=K.floatx())
        for i in range(size):
            a[i,:,:,:] = read_raw_image(self.data[start + i])
        if self.verbose > 0:
            print('verbose')
            # self.progress.update()
            # if self.progress.n >= len(self):
            #     self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
