from os.path import isfile
from model import Model

mpiotte_standard_model = 'C:/Users/Maks/kaggle-competitions/Humpback-Whale-Identification/data/models/mpiotte-standard.model'
if isfile(mpiotte_standard_model):
    tmp = keras.models.load_model(mpiotte_standard_model)
    print('Loaded tmp model')
    # model.set_weights(tmp.get_weights)
else:
    mpiotte_model = Model(64e05, 0.0002)
    # epoch 0-10
    mpiotte_model.make_steps(steps=10, ampl=1000)
    ampl = 100.0
    for _ in range(10):
        print('noise ampl = ', ampl)
        mpiotte_model.make_steps(steps=5, ampl=ampl)
    ampl = max(1.0, 100**-0.1 * ampl)
    for _ in range(18):
        mpiotte_model.make_steps(steps=5, ampl=1.0)
    # epoch -> 200
    set_lr(model, 16e-5)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 240
    set_lr(model, 4e-5)
    for _ in range(8):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 250
    set_lr(model, 1e-5)
    for _ in range(2):
        mpiotte_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 300
    # weights = model.get_weights()
    # model, branch_model, head_model = build_model(64e-5,0.0002)
    # model.set_weights(weights)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=1.0)
    # epoch -> 350
    set_lr(model, 16e-5)
    for _ in range(10):
        mpiotte_model.make_steps(steps=5, ampl=0.5)
    # epoch -> 390
    set_lr(model, 4e-5)
    for _ in range(8):
        mpiotte_model.make_steps(steps=5, ampl=0.25)
    # epoch -> 400
    set_lr(model, 1e-5)
    for _ in range(2):
        mpiotte_model.make_steps(steps=5, ampl=0.25)

    print(len(history))

    model.save('mpiotte-standard.model')
