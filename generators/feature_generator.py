from keras import backend as K
from keras.utils import Sequence
import numpy as np
from tqdm import tqdm

from utils.utils import read_raw_image

# A keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, img_gen, batch_size=64, img_shape=(384, 384, 1), verbose=1):
        """
        @param img_gen: image data generator function
        @param verbose: 1 for displaying progress bar, 0 for no information
        """
        super(FeatureGen, self).__init__()
        self.data = data
        self.img_gen = img_gen # image data generator function
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.verbose = verbose
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc="features")

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate start index and number of batch samples
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        
        # Generate data
        batch = self.__data_generation(start, size)

        # Update progress bar
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self):
                self.progress.close()

        return batch

    def __data_generation(self, start, size):
        'Generates data containing size samples'
        # Initialization
        batch = np.empty((size, ) + self.img_shape, dtype=K.floatx())

        # Generate data
        for i in range(size):
            batch[i,:,:,:] = self.img_gen(self.data[start + i])

        return batch