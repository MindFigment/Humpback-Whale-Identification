from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os.path import isfile
from model import Model
import keras
from globals import my_model
from globals import data_path, models_path
from keras.models import load_model
from models_file import contrastive_loss


def run(model_filename):

    model_path = models_path + model_filename
    model_name = model_filename.split('.')[0]
    if isfile(model_path):
        # # model_weights = keras.models.load_model(model_path).get_weights()
        model_weights =  keras.models.load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss}).get_weights()
        my_model = Model(64e-5, 0.0, model_name)
        my_model.model.set_weights(model_weights)
        print('Loaded pretrained model: ', model_name)    
    else:
        print('Model name: ', model_name)
        my_model = Model(64e-5, 0.0002, model_name)
        
    ##################
    #### TRAINING ####
    ##################

    # epoch 0-10
    # my_model.make_steps(steps=10, ampl=1000)
    # ampl = 100.0
    # for _ in range(10):
    #     print('noise ampl = ', ampl)
    #     my_model.make_steps(steps=5, ampl=ampl)
    #     ampl = max(1.0, 100**-0.1 * ampl)
    # for _ in range(18): 
    #     my_model.make_steps(steps=5, ampl=1.0)
    # # epoch -> 200
    # my_model.set_lr(16e-5)
    # for _ in range(5):
    #     my_model.make_steps(steps=5, ampl=0.5)
    # # epoch -> 240
    # my_model.set_lr(4e-5)
    # for _ in range(4):
    #     my_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 250
    # my_model.set_lr(1e-5)
    for _ in range(2):
        my_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 300
    # model_weights = my_model.model.get_weights()
    # my_model = Model(64e-5,0.0000002, model_name)
    # my_model.model.set_weights(model_weights)
    # my_model.set_l2(0.0002)
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
                        help='model filename',
                        default=None, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    run(args.model_filename)

if __name__ == '__main__':
  main()
