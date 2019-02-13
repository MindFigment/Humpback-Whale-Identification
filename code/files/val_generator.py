from keras.utils import Sequence
from keras import backend as K
from utils import read_raw_image
import numpy as np

class ValData(Sequence):
    def __init__(self, validation, img_gen, batch_size=32):
        """
        @param steps: number of epoch we are planning with this score matrix
        """
        super(ValData, self).__init__()
        self.validation = validation
        self.batch_size = batch_size
        self.img_shape = (384,384,1)
        self.img_gen = img_gen

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.validation))
        size = end - start
        assert size > 0
        a = np.zeros((size, ) + self.img_shape, dtype=K.floatx())
        b = np.zeros((size, ) + self.img_shape, dtype=K.floatx())
        c = np.zeros((size,1), dtype=K.floatx())

        for i in range(size):
            a[i,:,:,:] = self.img_gen(self.validation[start + i][0])
            b[i,:,:,:] = self.img_gen(self.validation[start + i][1])
            c[i,:] = self.validation[start + i][2]
        return [a,b], c

    def __len__(self):
        return (len(self.validation) + self.batch_size - 1) // self.batch_size
