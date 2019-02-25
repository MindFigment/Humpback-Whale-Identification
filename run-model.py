from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.models import load_model
import argparse
from os.path import isfile

from globals import data_path, models_path
from models import Model, contrastive_loss

def run(model_filename, use_val, small_data):

    model_path = models_path + model_filename
    model_name = ''.join(model_filename.split('.')[:-1])

    if isfile(model_path):
        model_weights = keras.models.load_model(model_path).get_weights()
        # model_weights =  keras.models.load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss}).get_weights()
        my_model = Model(64e-5, 2e-4, model_name, use_val=use_val, small_dataset=small_data)
        my_model.model.set_weights(model_weights)
        print('Loaded pretrained model: ', model_name)    
    else:
        print('Model name: ', model_name)
        my_model = Model(64e-5, 2e-4, model_name, use_val=use_val, small_dataset=small_data)
        
    ##################
    #### TRAINING ####
    ##################  

    # epoch 0-10
    my_model.make_steps(steps=10, ampl=1000)
    ampl = 100.0
    for _ in range(10):
        print('noise ampl = ', ampl)
        my_model.make_steps(steps=5, ampl=ampl)
        ampl = max(1.0, 100**-0.1 * ampl)
    for _ in range(18): 
        my_model.make_steps(steps=5, ampl=1.0)
    # epoch -> 200
    my_model.set_lr(16e-5)
    for _ in range(5):
        my_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 240
    my_model.set_lr(4e-5)
    for _ in range(4):
        my_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 250
    my_model.set_lr(1e-5)
    for _ in range(2):
        my_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 300
    for _ in range(18):
        my_model.make_steps(steps=8, ampl=1.0)
    # epoch -> 350
    my_model.set_lr(16e-5)
    for _ in range(15):
        my_model.make_steps(steps=8, ampl=0.5)
    # epoch -> 390
    my_model.set_lr(4e-5)
    for _ in range(15):
        my_model.make_steps(steps=8, ampl=0.25)
    # epoch -> 400
    my_model.set_lr(1e-5)
    for _ in range(10):
        my_model.make_steps(steps=8, ampl=0.25)

    print('Histories len: ', len(my_model.histories))

    my_model.model.save(model_path)

def parse_args():
    parser = argparse.ArgumentParser(description='train model')

    parser.add_argument('--model', '-m', dest='model_filename',
                        help='model filename (model.h5)',
                        default=None, type=str)

    parser.add_argument('--val', dest='use_val', action='store_true')
    parser.add_argument('--no-val', dest='use_val', action='store_false')
    parser.set_defaults(use_val=True)

    parser.add_argument('--small_data', dest='small_data', action='store_true')
    parser.add_argument('--no-small_data', dest='small_data', action='store_false')
    parser.set_defaults(small_data=False)
    
    return parser.parse_args()

def main():
    args = parse_args()
    run(args.model_filename, args.use_val, args.small_data)

if __name__ == '__main__':
  main()
