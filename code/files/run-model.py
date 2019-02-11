from os.path import isfile
from model import Model
import keras
from globals import mpiotte_standard_model
from globals import data_path, my_model_file
from keras.models import load_model

print(mpiotte_standard_model)

if isfile(mpiotte_standard_model + 'aa'):
    model = keras.models.load_model('../../data/input/mpiotte-standard.model')
    print('Loaded mpiotte-standard model')    
else:
    # weights = load_model(my_model_file).get_weights()
    mpiotte_model = Model(64e-5, 0)
    # mpiotte_model.model.set_weights(weights)
    # epoch 0-10
    mpiotte_model.make_steps(steps=10, ampl=1000)
    ampl = 100.0
    for _ in range(10):
        print('noise ampl = ', ampl)
        mpiotte_model.make_steps(steps=5, ampl=ampl)
    ampl = max(1.0, 100**-0.1 * ampl)
    for _ in range(10): # 18
        mpiotte_model.make_steps(steps=5, ampl=1.0)
    # epoch -> 200
    mpiotte_model.set_lr(16e-5)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 240
    mpiotte_model.set_lr(4e-5)
    for _ in range(8):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 250
    mpiotte_model.set_lr(1e-5)
    for _ in range(2):
        mpiotte_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 300
    weights = mpiotte_model.model.get_weights()
    mpiotte_model = Model(64e-5,0.0002)
    mpiotte_model.model.set_weights(weights)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=1.0)
    # epoch -> 350
    mpiotte_model.set_lr(16e-5)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 390
    mpiotte_model.set_lr(4e-5)
    for _ in range(8):
        mpiotte_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 400
    mpiotte_model.set_lr(1e-5)
    for _ in range(2):
        mpiotte_model.make_steps(steps=5, ampl=0.25)

    print(len(mpiotte_model.histories))

    mpiotte_model.model.save(data_path + 'models/tmp.h5')
