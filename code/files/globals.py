##################
## Folder paths ##
##################

data_path = '/home/stanislaw/kaggle-competitions/Humpback-Whale-Identification/data/' # main data folder path
input_path = data_path + 'input/'
cropped_imgs_path = data_path + 'cropped_imgs/'
# train_dir = input_path + 'train/'
# test_dir = input_path + 'test/'
train_dir = cropped_imgs_path + 'train/'
test_dir = cropped_imgs_path + 'test/'

callback_path = data_path + 'logs/' # logs generated during training
tensorboard_dir = callback_path + 'tensorboard/' 
models_path = data_path + 'models/' # models saved during training
output_path = data_path + 'output/' # csv submission files for

##################
## Dictionaries ##
##################

meta_dir = data_path + 'meta/'
whale2imgs_file = meta_dir + 'whale2imgs.pickle'
img2whale_file = meta_dir + 'img2whale.pickle'
whale2training_file = meta_dir + 'whale2training.pickle'
whale2index_file = meta_dir + 'whale2index.pickle'

################
## Data files ##
################

# Raw csv files
train_csv = input_path + 'train.csv'
sample_csv = input_path + 'sample_submission.csv'

# Generated pickle files
train_examples_file = meta_dir + 'train_examples.pickle'
validation_examples_file = meta_dir + 'validation_examples.pickle'
train_examples_small_file = meta_dir + 'train_examples_small.pickle'
validation_examples_small_file = meta_dir + 'validation_examples_small.pickle'

train_known_file = meta_dir + 'known.pickle'
train_submit_file = meta_dir + 'submit.pickle'
val_known_file = meta_dir + 'val_known.pickle'
val_submit_file = meta_dir + 'val_submit.pickle'

my_model = models_path + 'my_model.h5'
