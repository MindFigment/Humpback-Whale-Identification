from keras.utils import Sequence
from keras import backend as K
from utils import read_raw_image
import numpy as np

class ValData(Sequence):
    def __init__(self, validation, img_gen, batch_size=32, img_shape=(384, 384, 1)):
        """
        @param steps: number of epoch we are planning with this score matrix
        @param img_gen: image data generator function
        """
        super(ValData, self).__init__()
        self.validation = validation
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.img_gen = img_gen

    def __len__(self):
            return (len(self.validation) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate start and end index for batch of samples
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.validation))
        size = end - start
        assert size > 0

        # Generate data
        [a, b], c = self.__data_generation(start, size)

        return [a, b], c

    def __data_generation(self, start, size):
        'Generates data containing size samples'
        # Initialization
        a = np.empty((size, ) + self.img_shape, dtype=K.floatx())
        b = np.empty((size, ) + self.img_shape, dtype=K.floatx())
        c = np.empty((size,1), dtype=K.floatx())

        # Generate data
        for i in range(size):
            a[i,:,:,:] = self.img_gen(self.validation[start + i][0])
            b[i,:,:,:] = self.img_gen(self.validation[start + i][1])
            c[i,:] = self.validation[start + i][2]

        return [a, b], c
